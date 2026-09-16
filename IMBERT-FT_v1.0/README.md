# IMBERT-FT — fine-tuning

Fine-tunes the pre-trained IMBERT encoder for **binary** phenotype
classification (healthy vs. patient) and scores new samples with the result.
See the [repository README](../README.md) for the model description and the
released files.

## Contents

| File | Purpose |
|---|---|
| `BERT/finetune_hvsp_cv.py` | pooled healthy vs. patient, `n_sampling_times` random splits |
| `BERT/finetune_each_cohort_cv.py` | one phenotype per BioProject, cross-validated within the cohort |
| `BERT/finetune_multi_cohort_cv.py` | phenotypes pooled over every cohort that studied them |
| `BERT/predict.py` | score samples with a fine-tuned checkpoint (CPU friendly) |
| `BERT/Fine_tuning/classifier.py` | the classification heads |
| `BERT/Fine_tuning/datasets.py` | `Dataset` producing the encoder inputs and the label |
| `BERT/bert.py`, `BERT/utils.py` | encoder and shared helpers |

## Running

```bash
cd BERT
python finetune_hvsp_cv.py --config finetune_config.ini
```

All three scripts take the same configuration file. Each one loops over its own
set of cohorts, and for every cohort trains `n_sampling_times` models on
different random train/validation splits. A split whose result directory
already exists is skipped, so an interrupted run can simply be restarted.

`finetune_each_cohort_cv.py` takes its cohort→phenotype mapping from the
`COHORT_INFO` dictionary at the bottom of the file.
`finetune_multi_cohort_cv.py` reads it instead from
`DL_info_summary_<version>.xlsx` (sheet `phenotype-bioprjs_summary`, columns
`Phenotype`, `Study_Accession`, `N_bioprjs`) and needs `openpyxl` installed.

## Configuration

| Entry | Meaning |
|---|---|
| `[PreTrain] checkpoint_path` | the pre-trained model to start from |
| `[PreTrain][[Dataset]] vocab_path` | **required** — the vocabulary belonging to that checkpoint |
| `[FineTuning] gpu_ids` | CUDA ids, one process per id; `cpu` to run on CPU |
| `total_samples_rank_path` | abundance-rank table (see the repository README) |
| `sample_information_path` | sample metadata table |
| `pred_type` | `cross_validation` (random splits) or `cross_cohort` (train/test by BioProject) |
| `cv_ratio`, `n_sampling_times` | split fraction and number of repeats |
| `exclude_bioprojects`, `exclude_phenotypes` | comma separated, or `None` |
| `grad_start_layer` | freeze encoder layers below this index (`0` = train everything, `-1` = head only) |
| `output_type` | which classification head to use (below) |
| `n_layers`, `dense_act`, `cls_dropout_prob` | MLP head geometry; `n_layers = 0` means no MLP |

### Classification heads

| `output_type` | Head |
|---|---|
| `geneformer` | mean over all positions → linear (used for IMBERT v1.0) |
| `encoder` | first (highest-ranked) position → linear |
| `encoderWithgeneformer` | first position and mean concatenated → linear |
| `geneformerMLP` | mean over positions → MLP → linear |
| `encoderMLP` | first position → MLP → linear |

All heads return raw logits; the training loop applies `BCEWithLogitsLoss`, and
`predict.py` applies `torch.sigmoid` to get probabilities.

## Outputs

```
<result_dir>/<cohort>/<split index>/
├── etc/                   merged predictions, loss/learning-rate tables,
│                          gradient norms, the vocabulary and the sample lists
├── logs/                  per-process log
├── losses/                per-epoch loss tables
├── model_checkpoints/     Epoch<N>_checkpoint_FineTuning_<test_name>.ckp
└── prediction_status/     per-step prediction tables
```

The merged prediction table
`etc/<test_name>_classification_result_in_{train,validation}.tsv` has one row
per sample per step, with the true label, the predicted label and the predicted
probability.

## Scoring new samples

```bash
python predict.py \
    --config     finetune_config.ini \
    --checkpoint <result_dir>/.../model_checkpoints/Epoch14_checkpoint_FineTuning_<test_name>.ckp \
    --input      DL_input_<version>.tsv \
    --output     predictions.tsv
```

Pass the same configuration file the checkpoint was trained with: it supplies
the encoder geometry, the head type and the vocabulary.
