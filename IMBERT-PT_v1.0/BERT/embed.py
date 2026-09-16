"""Encode microbiome samples with a pre-trained IMBERT model.

Reads an abundance-rank table (the output of Preprocessing/preprocessing.py)
and writes one embedding vector per sample.  Runs on CPU by default, so the
released model can be used without a GPU.

Usage
-----
    python embed.py --checkpoint imbert_v1.0_pretrained.pt \
                    --input DL_input_20250728.tsv \
                    --output embeddings.tsv

    # a raw training checkpoint needs its vocabulary passed explicitly
    python embed.py --checkpoint imbert_v1.0_pretrained_epoch15000.ckp \
                    --vocab imbert_v1.0_pathway_vocab.tsv \
                    --input DL_input_20250728.tsv \
                    --output embeddings.tsv

Input format
------------
Tab separated; first column `Unnamed: 0` holds the pathway names, one further
column per sample holding the within-sample abundance rank (1 = most abundant,
0 = absent).  Pathway names must match the vocabulary that ships with the
checkpoint.

Pooling
-------
    mean (default)  average of all token positions -- the "geneformer" pooling
                    used by the IMBERT classification heads
    cls             the first (highest-ranked) position
"""

import argparse
import logging

import pandas as pd
import torch

import bert as Bert
import utils as u


def _silent_logger():
    logger = logging.getLogger('imbert.embed')
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    logger.setLevel(logging.CRITICAL)

    return logger


def load_pretrained(checkpoint_path, vocab_path=None, num_attention_heads=4, device='cpu'):
    """Rebuild the encoder described by a checkpoint and return (model, vocab)."""
    checkpoint = u.load_checkpoint(checkpoint_path, map_location='cpu')

    state_dict = u.strip_checkpoint_prefixes(checkpoint['model_state_dict'])

    if checkpoint.get('format') == 'imbert-checkpoint-v1' and vocab_path is None:
        vocab = checkpoint['vocab']
        hyperparams = dict(checkpoint['model_hyperparams'])
    else:
        if vocab_path is None:
            raise ValueError("this checkpoint does not embed its vocabulary; pass --vocab")
        vocab = u.read_vocab_file(vocab_path)
        hyperparams = u.infer_model_hyperparams(state_dict, num_attention_heads)

    if hyperparams['vocab_size'] != len(vocab):
        raise ValueError(f"the checkpoint has {hyperparams['vocab_size']} embedding rows but the "
                         f"vocabulary lists {len(vocab)} tokens -- they do not belong together")

    from transformers import BertConfig
    bert_config = BertConfig(hidden_act='gelu',
                             hidden_dropout_prob=0.0,
                             attention_probs_dropout_prob=0.0,
                             initializer_range=0.02,
                             **hyperparams)

    model = Bert.BertModel(bert_config, _silent_logger())
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    print(f"[IMBERT] loaded {checkpoint_path}: "
          f"{hyperparams['num_hidden_layers']} layers, hidden {hyperparams['hidden_size']}, "
          f"{hyperparams['num_attention_heads']} heads, vocab {len(vocab)}")

    return model, vocab, hyperparams


def encode_table(rank_df, vocab, seq_len):
    """Turn the abundance-rank table into padded token-id sequences."""
    sample_names = [c for c in rank_df.columns if c != 'Unnamed: 0']
    unk_id, pad_id = vocab['[UNK]'], vocab['[PAD]']

    sequences, masks = [], []
    n_unknown = 0
    for sample in sample_names:
        column = rank_df.loc[:, ['Unnamed: 0', sample]]
        ranked = column.loc[column[sample] > 0, :].sort_values(by=sample, ascending=True)
        ids = []
        for pathway in ranked['Unnamed: 0']:
            token_id = vocab.get(pathway)
            if token_id is None:
                token_id = unk_id
                n_unknown += 1
            ids.append(token_id)

        ids = ids[:seq_len]
        mask = [1] * len(ids) + [0] * (seq_len - len(ids))
        ids = ids + [pad_id] * (seq_len - len(ids))
        sequences.append(ids)
        masks.append(mask)

    if n_unknown:
        print(f"[IMBERT] {n_unknown} pathway occurrence(s) were not in the vocabulary "
              f"and were encoded as [UNK]")

    return sample_names, torch.tensor(sequences), torch.tensor(masks)


@torch.no_grad()
def embed(model, token_ids, attention_mask, pooling='mean', batch_size=16, device='cpu'):
    embeddings = []
    for start in range(0, token_ids.shape[0], batch_size):
        ids = token_ids[start:start + batch_size].to(device)
        mask = attention_mask[start:start + batch_size].to(device)
        segment = torch.zeros_like(ids)

        encoded_layers, _ = model(ids, segment, mask,
                                  output_all_encoded_layers=False,
                                  attention_show_flg=False,
                                  run_pooler=False)

        if pooling == 'cls':
            pooled = encoded_layers[:, 0, :]
        else:
            pooled = encoded_layers.mean(dim=1)
        embeddings.append(pooled.cpu())

    return torch.cat(embeddings, dim=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", default=None,
                        help="required for raw training checkpoints (.ckp)")
    parser.add_argument("--input", required=True, help="abundance-rank table (.tsv)")
    parser.add_argument("--output", required=True, help="embeddings to write (.tsv)")
    parser.add_argument("--pooling", choices=['mean', 'cls'], default='mean')
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--device", default='cpu',
                        help="cpu (default), cuda, or cuda:<id>")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("[IMBERT] CUDA is not available -> using CPU")
        device = torch.device('cpu')

    model, vocab, hyperparams = load_pretrained(args.checkpoint, args.vocab,
                                                args.num_attention_heads, device)

    rank_df = pd.read_csv(args.input, sep='\t')
    sample_names, token_ids, attention_mask = encode_table(
        rank_df, vocab, hyperparams['max_position_embeddings'])
    print(f"[IMBERT] encoding {len(sample_names)} samples")

    embeddings = embed(model, token_ids, attention_mask,
                       pooling=args.pooling, batch_size=args.batch_size, device=device)

    out = pd.DataFrame(embeddings.numpy(),
                       index=sample_names,
                       columns=[f"dim{i}" for i in range(embeddings.shape[1])])
    out.index.name = 'Sample'
    out.to_csv(args.output, sep='\t')
    print(f"[IMBERT] wrote {out.shape[0]} x {out.shape[1]} embeddings to {args.output}")


if __name__ == "__main__":
    main()
