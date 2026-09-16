"""Turn a training checkpoint into a small, portable release artefact.

A checkpoint written by pretrain.py holds the model weights, the AdamW state
(about two thirds of the file) and DistributedDataParallel key prefixes.  For
distribution none of that is wanted:

* dropping the optimizer state makes the file roughly three times smaller;
* dropping the `module.` / `bert.` prefixes makes it loadable by any script;
* a tensors-only file loads under `torch.load(..., weights_only=True)`, the
  safe default since PyTorch 2.6, so users are not asked to disable a security
  check to run the model.

The vocabulary is embedded in the exported file and its SHA-256 recorded, so a
downloaded checkpoint can never drift apart from the pathway ids it was trained
with.

Usage
-----
    python export_checkpoint.py \
        --checkpoint imbert_v1.0_pretrained_epoch15000.ckp \
        --vocab imbert_v1.0_pathway_vocab.tsv \
        --output imbert_v1.0_pretrained.pt
"""

import argparse
import hashlib
import json
import os

import torch

import utils as u


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="training checkpoint (.ckp)")
    parser.add_argument("--vocab", required=True, help="pathway vocabulary (.tsv)")
    parser.add_argument("--output", required=True, help="release file to write (.pt)")
    parser.add_argument("--num-attention-heads", type=int, default=4,
                        help="attention heads of the encoder; the only value that "
                             "cannot be recovered from the tensor shapes (default: 4)")
    parser.add_argument("--keep-mlm-head", action="store_true",
                        help="also export the masked-language-model head")
    args = parser.parse_args()

    checkpoint = u.load_checkpoint(args.checkpoint, map_location='cpu')
    state_dict = u.strip_checkpoint_prefixes(checkpoint['model_state_dict'],
                                             drop_mlm_head=not args.keep_mlm_head)

    vocab = u.read_vocab_file(args.vocab)
    hyperparams = u.infer_model_hyperparams(state_dict, args.num_attention_heads)

    if hyperparams['vocab_size'] != len(vocab):
        raise ValueError(f"the checkpoint has {hyperparams['vocab_size']} embedding rows but "
                         f"{args.vocab} lists {len(vocab)} tokens -- they do not belong together")

    payload = {
        'format': 'imbert-checkpoint-v1',
        'model_state_dict': state_dict,
        'model_hyperparams': hyperparams,
        'vocab': vocab,
        'vocab_sha256': sha256(args.vocab),
        'source_epoch': int(checkpoint.get('epoch', -1)),
        'source_checkpoint': os.path.basename(args.checkpoint),
    }
    torch.save(payload, args.output)

    original_mb = os.path.getsize(args.checkpoint) / 1e6
    exported_mb = os.path.getsize(args.output) / 1e6
    n_params = sum(t.numel() for t in state_dict.values())

    print("[export] note: torch.save is not byte-reproducible -- re-running this "
          "command yields the same tensors but a different file hash. Keep the file "
          "you publish and hash that one.")
    print(json.dumps({'output': args.output,
                      'tensors': len(state_dict),
                      'parameters': n_params,
                      'original_MB': round(original_mb, 1),
                      'exported_MB': round(exported_mb, 1),
                      'sha256': sha256(args.output),
                      **hyperparams}, indent=2))


if __name__ == "__main__":
    main()
