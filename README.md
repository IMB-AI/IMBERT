# IMBERT v1.0

IMBERT is a BERT-style transformer trained on **microbial metabolic pathway profiles**.
Each sample is treated as a "sentence": the pathways detected in that sample are
ordered by decreasing abundance, and each pathway is a token. The encoder is
pre-trained with masked-language modelling over that corpus and then fine-tuned
to classify host phenotypes from a stool metagenome.

This repository contains the pre-training code, the fine-tuning code, and the
scripts needed to apply the released model to new samples.

Repository: <https://github.com/IMB-AI/IMBERT>

> **Status:** the accompanying manuscript is under review. The reference and the
> archive DOIs in the [Citation](#citation) section are filled in on publication.

---

## Model

| | |
|---|---|
| Architecture | BERT encoder, 12 layers, hidden size 256, 4 attention heads, FFN 512 |
| Linear layers | no bias terms (`model_bias = False`) |
| Vocabulary | 3,233 tokens = 3,228 MetaCyc pathways + `[PAD] [UNK] [CLS] [SEP] [MASK]` |
| Max sequence length | 460 pathways per sample |
| Parameters | 7,315,712 (encoder only, without the MLM head) |
| Pre-training objective | masked language modelling (15 % masking) |
| Pre-training corpus | 27,149 metagenome samples (26,402 train / 747 validation) |
| Released checkpoint | epoch 15,000 |

The code in this repository reproduces the published results exactly: re-running
the healthy-versus-patient cross-validation with the released weights returned
identical AUROC, AUPRC, accuracy and precision at every epoch.

**Intended use:** research on microbiome–phenotype associations.
**Not** validated or approved for clinical or diagnostic use.

### Released files

| File | Size | Where | What it is |
|---|---|---|---|
| `imbert_v1.0_pretrained.pt` | 29.5 MB | **in this repository** | Recommended. Weights + vocabulary + architecture in one file; loads with `torch.load(..., weights_only=True)`. |
| `imbert_v1.0_pathway_vocab.tsv` | 165 KB | **in this repository** | Pathway → token-id mapping, one `id<TAB>pathway` per line. |
| `imbert_v1.0_pretrained_epoch15000.ckp` | 97.5 MB | Zenodo only | The raw training checkpoint (weights **and** AdamW state). Only needed to resume pre-training. |

So `git clone` gives you a working model: no separate download is needed for
either inference or fine-tuning.

> ⚠️ **The vocabulary is part of the model.** A pathway's row in the word-embedding
> matrix is fixed at pre-training time. Feeding the model a table whose pathway
> ids were derived some other way produces silently wrong results, so every
> script takes the vocabulary file explicitly.

> ⚠️ The raw `.ckp` carries optimizer state containing numpy objects, so
> PyTorch ≥ 2.6 refuses to load it under its default `weights_only=True`.
> The loaders in this repository fall back automatically and print a warning;
> `imbert_v1.0_pretrained.pt` avoids the issue entirely and is a third the size.

Regenerate the portable file from a training checkpoint with:

```bash
cd IMBERT-PT_v1.0/BERT
python export_checkpoint.py \
    --checkpoint ../../imbert_v1.0_pretrained_epoch15000.ckp \
    --vocab      ../../imbert_v1.0_pathway_vocab.tsv \
    --output     ../../imbert_v1.0_pretrained.pt
```

---

## Repository layout

```
IMBERT-PT_v1.0/                     pre-training
├── Preprocessing/
│   ├── preprocessing.py            abundance files  ->  abundance-rank table
│   └── preprocessing_config.ini
└── BERT/
    ├── pretrain.py                 masked-language-model pre-training
    ├── pretrain_config.ini
    ├── embed.py                    encode samples with a pre-trained model
    ├── export_checkpoint.py        make a portable release file
    ├── bert.py                     BERT encoder
    ├── optimizer.py                learning-rate schedule
    ├── utils.py
    └── Pathway_abundance/datasets.py

IMBERT-FT_v1.0/                     fine-tuning
└── BERT/
    ├── finetune_hvsp_cv.py         pooled healthy vs. patient
    ├── finetune_each_cohort_cv.py  one phenotype per cohort
    ├── finetune_multi_cohort_cv.py phenotypes pooled across cohorts
    ├── finetune_config.ini
    ├── predict.py                  score samples with a fine-tuned model
    ├── bert.py                     BERT encoder (fine-tuning variant)
    ├── utils.py
    └── Fine_tuning/
        ├── classifier.py           classification heads
        └── datasets.py
```

The two packages are deliberately self-contained: `bert.py` differs slightly
between them (the fine-tuning encoder also returns the embedding output and all
intermediate layers), so neither imports the other.

---

## Installation

```bash
git clone https://github.com/IMB-AI/IMBERT.git
cd IMBERT
pip install -r requirements.txt
```

To reproduce the published numbers exactly, install the pinned environment they
were produced in instead:

```bash
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements-paper.txt
```

Python 3.9+ is required; the published results used Python 3.10.3 with
PyTorch 2.0.1 on four NVIDIA RTX A5000 GPUs. Everything below runs on CPU; a GPU only makes it
faster. Set `gpu_ids = cpu` in the configuration file, or simply run on a
machine without CUDA, and the scripts fall back to a single CPU process.

---

## Quick start — use the released model

### 1. Prepare the input table

Start from per-sample pathway abundance files (for example HUMAnN
`*_pathabundance.tsv`) placed in `<raw_input_dir>/DL_input_<version>/`, one file
per sample named `<SampleId>_*`:

```
Unnamed: 0                                 SRR1234567
PWY-6990: (+)-camphor biosynthesis         0.0123
PWY-5484: glycolysis II                    0.4510
...
```

```bash
cd IMBERT-PT_v1.0/Preprocessing
python preprocessing.py --config preprocessing_config.ini
```

This writes `DL_input_<version>.tsv`: rows are pathways renamed to
`PATHWAY NAME:PWY-ID` (matching the released vocabulary), columns are samples,
and each cell is the **rank** of that pathway within that sample
(1 = most abundant, 0 = absent).

### 2. Get sample embeddings

```bash
cd IMBERT-PT_v1.0/BERT
python embed.py \
    --checkpoint ../../imbert_v1.0_pretrained.pt \
    --input      /path/to/DL_input_<version>.tsv \
    --output     embeddings.tsv
```

Produces one 256-dimensional vector per sample. `--pooling cls` uses the first
(highest-ranked) position instead of the mean over positions.

### 3. Score samples with a fine-tuned classifier

```bash
cd IMBERT-FT_v1.0/BERT
python predict.py \
    --config     finetune_config.ini \
    --checkpoint /path/to/Epoch14_checkpoint_FineTuning_<test_name>.ckp \
    --input      /path/to/DL_input_<version>.tsv \
    --output     predictions.tsv
```

---

## Training

### Pre-training

```bash
cd IMBERT-PT_v1.0/BERT
python pretrain.py --config pretrain_config.ini
```

Edit `pretrain_config.ini` first: `gpu_ids`, `result_dir`, the two dataset paths
and `vocab_path`. Leave `vocab_path` **empty** to train a brand-new model with a
vocabulary derived from your own table; set it to the released vocabulary to
reproduce or extend IMBERT v1.0.

### Fine-tuning

```bash
cd IMBERT-FT_v1.0/BERT
python finetune_hvsp_cv.py --config finetune_config.ini
```

`finetune_config.ini` needs `checkpoint_path` (the released model),
`vocab_path` (the released vocabulary), and the two data files described below.
See `IMBERT-FT_v1.0/README.md` for the three experiment layouts.

---

## Input data formats

**Abundance-rank table** (`total_samples_rank_path`, `Training_dataset_path`, …)
— tab separated, produced by `preprocessing.py`:

| `Unnamed: 0` | SRR1234567 | SRR1234568 |
|---|---|---|
| `GLYCOLYSIS II:PWY-5484` | 1 | 3 |
| `(+)-CAMPHOR BIOSYNTHESIS:PWY-6990` | 0 | 1 |

**Sample metadata** (`sample_information_path`, `DL_input_<version>_info.txt`)
— tab separated, `#` starts a comment line:

| Run_Id | Study_Accession | Subject_Health_Status | Phenotype | DL_Group |
|---|---|---|---|---|
| SRR1234567 | PRJNA516054 | Healthy | Healthy | Fine-tuning |
| SRR1234568 | PRJNA516054 | Non-healthy | Autism | Fine-tuning |

`Subject_Health_Status` is `Healthy` or `Non-healthy`; `DL_Group` is
`Fine-tuning` for samples used in the downstream tasks.

---

## Limitations

* The model only knows the 3,228 pathways in its vocabulary. Pathways outside it
  are encoded as `[UNK]`; a pathway database that differs substantially from the
  one used for pre-training will degrade performance.
* Samples are truncated to their 460 highest-ranked pathways.
* Only abundance **ranks** are used, so differences in absolute abundance between
  two pathways with adjacent ranks are not represented.
* Pre-training and fine-tuning data are public shotgun metagenomes; performance
  on amplicon (16S) data or on populations not represented in those cohorts has
  not been established here.

---

## Licence

**IMBERT is released for non-commercial research use only**, under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). One licence
covers everything: the source code, the pre-trained weights, the pathway
vocabulary, and any checkpoint fine-tuned from them. See `LICENSE`.

| | |
|---|---|
| Permitted | use, modification, fine-tuning and redistribution — including of derived models — for non-commercial purposes, with attribution |
| Not permitted | any use primarily directed towards commercial advantage or monetary compensation |
| Patents | **not licensed.** CC BY-NC 4.0 §2(b)(2) excludes patent and trademark rights, and no patent licence is granted by implication either |

Whether a use is permitted depends on the *purpose*, not on who you are: research
at a company can be non-commercial, and a paid service run by a non-profit is not.

> This is not one of the templates in GitHub's licence picker — every template
> there is an open-source licence that permits commercial use — so GitHub may
> show "Unknown licence" rather than a badge.

For commercial licensing enquiries, open an issue at
https://github.com/IMB-AI/IMBERT/issues.

This repository contains third-party code. See `NOTICE` for the derivation and
`THIRD-PARTY-LICENSES/` for the upstream MIT and Apache-2.0 texts, which must
be retained in any redistribution.

## Citation

> **TODO** — add the paper reference and the DOI of the archived weights.
