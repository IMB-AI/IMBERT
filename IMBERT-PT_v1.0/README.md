# IMBERT-PT — pre-training

Masked-language-model pre-training of the IMBERT encoder on microbial pathway
profiles, plus the preprocessing and inference utilities that go with it.
See the [repository README](../README.md) for the model description and the
released files.

## Contents

| File | Purpose |
|---|---|
| `Preprocessing/preprocessing.py` | per-sample abundance files → one abundance-rank table |
| `BERT/pretrain.py` | masked-language-model pre-training |
| `BERT/embed.py` | encode samples with a pre-trained model (CPU friendly) |
| `BERT/export_checkpoint.py` | training checkpoint → portable release file |
| `BERT/bert.py` | the BERT encoder |
| `BERT/optimizer.py` | inverse-square-root learning-rate schedule with warm-up |
| `BERT/Pathway_abundance/datasets.py` | `Dataset` that masks 15 % of the tokens |

## 1. Preprocessing

```bash
cd Preprocessing
python preprocessing.py --config preprocessing_config.ini [--save-tmp]
```

Reads every file in `<raw_input_dir>/DL_input_<version>/`, normalises the
pathway names to `PATHWAY NAME:PWY-ID`, ranks the non-zero pathways within each
sample (1 = most abundant) and breaks ties deterministically by pathway name.
Writes `<preprocessing_dir>/<version>/DL_input_<version>.tsv`.

`--save-tmp` additionally writes the per-sample intermediate tables, which is
useful when checking the ranking of a single sample.

## 2. Pre-training

```bash
cd BERT
python pretrain.py --config pretrain_config.ini
```

Key configuration entries:

| Entry | Meaning |
|---|---|
| `gpu_ids` | CUDA ids, one process per id; `cpu` to run on CPU |
| `Training_dataset_path` / `Validation_dataset_path` | abundance-rank tables |
| `dataset_dir` + `version` | used to find `DL_input_<version>_info.txt` |
| `vocab_path` | released vocabulary, or **empty** to build a new one from the training table |
| `batch_size` | total batch size, divided across the processes |
| `prediction_checkpoint` | save a checkpoint and dump per-sample predictions every N epochs |

Outputs land in `result_dir`:

```
<result_dir>/
├── <task>_<test_name>_IMB_pathway_vocab.tsv   the vocabulary used by this run
├── logs/                                      per-process training/validation logs
├── losses/                                    per-epoch loss tables
├── model_checkpoints/                         Epoch<N>_checkpoint_PreTrain_<test_name>.ckp
└── prediction_status/                         masked-token predictions per sample
```

A process group is created even for a single process, so `DistributedSampler`
and the loss `all_gather` behave the same on one CPU and on eight GPUs.

## 3. Exporting weights for release

```bash
python export_checkpoint.py \
    --checkpoint ../../imbert_v1.0_pretrained_epoch15000.ckp \
    --vocab      ../../imbert_v1.0_pathway_vocab.tsv \
    --output     ../../imbert_v1.0_pretrained.pt
```

Drops the optimizer state and the DistributedDataParallel key prefixes, embeds
the vocabulary and the architecture, and prints the SHA-256 of the result.
97.5 MB → 29.5 MB, and the result loads under `weights_only=True`.

## 4. Encoding new samples

```bash
python embed.py \
    --checkpoint ../../imbert_v1.0_pretrained.pt \
    --input      DL_input_<version>.tsv \
    --output     embeddings.tsv
```

A raw `.ckp` also works but needs `--vocab`. `--pooling {mean,cls}` selects the
pooling (`mean` matches the `geneformer` head used for classification),
`--device cuda` runs on a GPU.
