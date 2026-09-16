"""Score samples with a fine-tuned IMBERT classifier.

Reads an abundance-rank table and a checkpoint written by one of the
finetune_*.py scripts and writes one probability per sample.  Runs on CPU by
default, so a released classifier can be applied without a GPU.

Usage
-----
    python predict.py --config finetune_config.ini \
                      --checkpoint Epoch14_checkpoint_FineTuning_....ckp \
                      --input DL_input_20250728.tsv \
                      --output predictions.tsv

The configuration file supplies the encoder geometry, the `output_type` of the
classification head and the pathway vocabulary, so it must be the same one that
produced the checkpoint.
"""

import argparse
import logging

import pandas as pd
import torch
from configobj import ConfigObj

import utils as u
from Fine_tuning import classifier as ftc

_CLASSIFIER_BY_OUTPUT_TYPE = {
    'geneformer': ftc.IMBERTBinaryClassifier_GF,
    'encoder': ftc.IMBERTBinaryClassifier_CLS,
    'encoderWithgeneformer': ftc.IMBERTBinaryClassifier_CLSWithGF,
    'geneformerMLP': ftc.IMBERTBinaryClassifier_GFMLP,
    'encoderMLP': ftc.IMBERTBinaryClassifier_CLSMLP,
}


def _silent_logger():
    logger = logging.getLogger('imbert.predict')
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    logger.setLevel(logging.CRITICAL)

    return logger


def load_finetuned(config, checkpoint_path, device):
    """Rebuild the fine-tuned classifier described by `config` and load its weights."""
    vocab = u.load_pretrained_vocab(config)
    if vocab is None:
        raise ValueError("set [PreTrain][[Dataset]] vocab_path to the vocabulary that "
                         "belongs to the checkpoint")

    logger = _silent_logger()
    bert = u.load_model(config, vocab, logger)

    output_type = config['FineTuning']['hyperparams']['output_type']
    if output_type not in _CLASSIFIER_BY_OUTPUT_TYPE:
        raise ValueError(f"unknown output_type {output_type!r}")
    model = _CLASSIFIER_BY_OUTPUT_TYPE[output_type](config, bert, logger)

    checkpoint = u.load_checkpoint(checkpoint_path, map_location='cpu')
    state_dict = {k[len('module.'):] if k.startswith('module.') else k: v
                  for k, v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    print(f"[IMBERT] loaded {checkpoint_path} "
          f"(epoch {checkpoint.get('epoch', 'n/a')}, output_type {output_type}, "
          f"vocab {len(vocab)})")

    return model, vocab


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
def predict(model, token_ids, attention_mask, batch_size=16, device='cpu'):
    logits = []
    for start in range(0, token_ids.shape[0], batch_size):
        ids = token_ids[start:start + batch_size].to(device)
        mask = attention_mask[start:start + batch_size].to(device)
        segment = torch.zeros_like(ids)

        _, batch_logits = model(ids, segment, mask)
        logits.append(batch_logits.squeeze(-1).cpu())

    return torch.cat(logits, dim=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="the finetune_config.ini used for training")
    parser.add_argument("--checkpoint", required=True, help="fine-tuned checkpoint (.ckp)")
    parser.add_argument("--input", required=True, help="abundance-rank table (.tsv)")
    parser.add_argument("--output", required=True, help="predictions to write (.tsv)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="probability above which a sample is called positive")
    parser.add_argument("--device", default='cpu', help="cpu (default), cuda, or cuda:<id>")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("[IMBERT] CUDA is not available -> using CPU")
        device = torch.device('cpu')

    config = ConfigObj(args.config, file_error=True)
    model, vocab = load_finetuned(config, args.checkpoint, device)

    seq_len = config['Model']['hyperparams'].as_int('max_position_embeddings')
    rank_df = pd.read_csv(args.input, sep='\t')
    sample_names, token_ids, attention_mask = encode_table(rank_df, vocab, seq_len)
    print(f"[IMBERT] scoring {len(sample_names)} samples")

    logits = predict(model, token_ids, attention_mask, args.batch_size, device)
    probabilities = torch.sigmoid(logits)

    out = pd.DataFrame({'Sample': sample_names,
                        'Logit': logits.numpy(),
                        'Probability': probabilities.numpy(),
                        'Pred_label': (probabilities >= args.threshold).int().numpy()})
    out.to_csv(args.output, sep='\t', index=False)
    print(f"[IMBERT] wrote {len(out)} predictions to {args.output}")


if __name__ == "__main__":
    main()
