"""Fine-tune IMBERT for binary phenotype classification.

Cross-validation over cohorts pooled per phenotype: the cohorts listed in
`cohort_info` are merged and split into train/validation `n_sampling_times` times.

Usage
-----
    python finetune_multi_cohort_cv.py --config finetune_config.ini

One process is spawned per CUDA device listed in `gpu_ids`; set `gpu_ids = cpu`
(or run on a machine without CUDA) to train in a single CPU process.
"""

import argparse
import glob
import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import tqdm
from configobj import ConfigObj
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_linear_schedule_with_warmup

import utils as u
from Fine_tuning import classifier as ftc

_CANDIDATE_TARGET_DISEASE_LIST = ['ACVD',
                                  'Autism',
                                  'Colorectal_cancer',
                                  'Crohns_disease',
                                  'Healthy',
                                  'HvsP',
                                  'Hypertension',
                                  'Impaired_glucose_tolerance',
                                  'Liver_Cirrhosis',
                                  'Obesity',
                                  'Rheumatoid_arthritis',
                                  'Symptomatic_atherosclerosis',
                                  'T1D',
                                  'T2D',
                                  'Ulcerative_colitis',
                                  'neuroblastoma']

_CLASSIFIER_BY_OUTPUT_TYPE = {
    'geneformer': ftc.IMBERTBinaryClassifier_GF,
    'encoder': ftc.IMBERTBinaryClassifier_CLS,
    'encoderWithgeneformer': ftc.IMBERTBinaryClassifier_CLSWithGF,
    'geneformerMLP': ftc.IMBERTBinaryClassifier_GFMLP,
    'encoderMLP': ftc.IMBERTBinaryClassifier_CLSMLP,
}


def _gather_classification_result(epoch,
                                  step,
                                  cls_output,
                                  data,
                                  sample_name_id_dict,
                                  bioprj_samples_dict,
                                  current_lr,
                                  rank: int):
    result_details_list = []
    for _, (cls_out, sample_name_id_tensor, sample_true_label_tensor) in enumerate(
            zip(cls_output, data['sample_name_id'], data['disease_label'])):
        sample_name_id = int(sum(sample_name_id_tensor.cpu().numpy()))
        sample_name = "".join([sample for sample, ids in sample_name_id_dict.items() if ids == sample_name_id])
        bioproject = u.find_bioproject_from_sample(bioprj_samples_dict, sample_name)
        true_label = int(sum(sample_true_label_tensor.cpu().numpy()))
        prob = float(torch.sigmoid(cls_out.detach()))
        pred_label = 1 if prob >= 0.5 else 0

        result = True if true_label == pred_label else False
        row = [rank,
               epoch,
               step,
               bioproject,
               sample_name_id,
               sample_name,
               true_label,
               pred_label,
               result,
               prob,
               current_lr]
        result_details_list.append(row)

    return result_details_list


def _write_classification_result_table(config, result_details_list, epoch, step, rank, mode: str):
    test_name, prediction_status_dir = u.get_test_name_and_result_dir(config, 'FineTuning',
                                                                     'prediction_status_dir')
    status_file_path = os.path.join(
        prediction_status_dir,
        f"{test_name}_{mode}_cls_results_table_epoch{epoch}_{step}steps_gpu{rank}.tsv")

    header = ['GPU', 'Epoch', 'Step', 'BioProject', 'Sample_id', 'Sample_name',
              'True_label', 'Pred_label', 'Result', 'Probability', 'Learning_rate']

    pd.DataFrame(result_details_list, columns=header).to_csv(status_file_path, sep='\t', index=False)


def _merge_prediction_results(config, mode: str):
    test_name, etc_dir = u.get_test_name_and_result_dir(config, 'FineTuning', 'etc_dir')
    _, ps_dir = u.get_test_name_and_result_dir(config, 'FineTuning', 'prediction_status_dir')
    df_list = [pd.read_csv(os.path.join(ps_dir, file), sep='\t') for file in os.listdir(ps_dir) if mode in file]
    merge_df = pd.concat(df_list, axis=0)

    final_merge_df = merge_df.sort_values(by=['GPU', 'Epoch', 'Step'])
    final_result_path = os.path.join(etc_dir, f"{test_name}_classification_result_in_{mode}.tsv")
    final_merge_df.to_csv(final_result_path, sep='\t', index=False)


def _write_loss_and_lr(config, post_fix_list, mode: str):
    test_name, etc_dir = u.get_test_name_and_result_dir(config, 'FineTuning', 'etc_dir')
    filename = f'{test_name}_{mode}_cls_loss_and_learning_rate.tsv'

    if mode == 'train':
        header = ['GPU', 'Epoch', 'Step', 'Training_CLS_loss', 'Learning_rate']
    else:
        header = ['GPU', 'Epoch', 'Step', 'Validation_CLS_loss', 'Learning_rate']

    row_list = []
    for post_fix in post_fix_list:
        gpu = str(post_fix['GPU'])
        epoch = str(post_fix['Epoch'])
        step = str(post_fix['Step'])

        if mode == 'train':
            cls_loss = str(post_fix['Training_CLS_loss'])
        else:
            cls_loss = str(post_fix['Validation_CLS_loss'])

        lr = str(post_fix['Learning_rate'])

        row = [gpu, epoch, step, cls_loss, lr]
        row_list.append(row)

    df = pd.DataFrame(row_list, columns=header)
    df.to_csv(os.path.join(etc_dir, filename), sep='\t', index=False)


def _write_gradient_flow_table(config, history):
    test_name, write_dir = u.get_test_name_and_result_dir(config, 'FineTuning', 'etc_dir')

    data = []
    for epoch, step_layer_grad_dict in history['gradients'].items():
        for step, layer_grad_dict in step_layer_grad_dict.items():
            for layer, grad in layer_grad_dict.items():
                data.append([epoch.replace('Epoch', ''), step.replace('Step', ''), layer, grad])

    df = pd.DataFrame(data, columns=['Epoch', 'Step', 'Layer', 'Gradients'])
    df.to_csv(os.path.join(write_dir, f'{test_name}_model_gradients_flow_during_train.tsv'), sep='\t', index=False)


def _check_hyperparams(config):
    task = config['COMMON']['task']
    dataset = config[task]['Dataset']
    hyperparams = config[task]['hyperparams']

    prediction_type = dataset['pred_type']
    if prediction_type not in ['cross_cohort', 'cross_validation']:
        raise ValueError(f"pred_type must be cross_cohort or cross_validation, got {prediction_type!r}")
    if prediction_type == 'cross_validation' and dataset['cv_ratio'] == 'none':
        raise ValueError("cv_ratio must be a float when pred_type is cross_validation")

    output_type = hyperparams['output_type']
    if output_type not in _CLASSIFIER_BY_OUTPUT_TYPE:
        raise ValueError(f"output_type must be one of "
                         f"{sorted(_CLASSIFIER_BY_OUTPUT_TYPE)}, got {output_type!r}")

    total_encoder_layers = config['Model']['hyperparams'].as_int('num_hidden_layers')
    layer_idx = hyperparams.as_int('grad_start_layer')
    encoder_result_layer = hyperparams.as_int('encoder_result_layer')
    if encoder_result_layer == -1:
        encoder_result_layer = total_encoder_layers - 1
    if encoder_result_layer >= total_encoder_layers:
        raise ValueError(f"encoder_result_layer ({encoder_result_layer}) is beyond the "
                         f"{total_encoder_layers} encoder layers of the model")
    if encoder_result_layer < layer_idx:
        raise ValueError("grad_start_layer must be smaller than encoder_result_layer")

    target_disease = dataset["target_disease"]
    if target_disease not in _CANDIDATE_TARGET_DISEASE_LIST:
        raise ValueError(f"{target_disease} is not in the known target disease list "
                         f"{_CANDIDATE_TARGET_DISEASE_LIST}")

    print("=" * 110)
    print('Hyperparams check complete')
    print("=" * 110 + "\n")


# Which utils helpers this experiment uses.
_LOAD_DATASETS = u.load_datasets
_PREPARE_INPUTS = u.prepare_binary_classification_v2
_COOLDOWN_SECONDS = 30


def main_worker(rank,
                world_size,
                config,
                train_df,
                validation_df,
                train_samples_info_df,
                valid_samples_info_df,
                train_sample_name_id_dict,
                validation_sample_name_id_dict,
                train_bioprj_samples_dict,
                validation_bioprj_samples_dict,
                vocab,
                addr, port):
    task = config['COMMON']['task']
    test_name = config[task]['test_name']

    # 1. Reproducibility
    random_seed = 1033
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 2. Devices and process group
    u.empty_cuda_cache()
    device_ids = u.resolve_device_ids(config, task)
    device = u.init_for_distributed(rank, device_ids, world_size, addr, port)

    # 3. Logger
    logger = u.make_logger(u.get_log_path(config, rank))

    # 4. Dataloaders
    train_sampler, train_dataloader = _LOAD_DATASETS(config, train_df, vocab,
                                                     train_sample_name_id_dict,
                                                     train_samples_info_df, 'train')
    validation_dataloader = _LOAD_DATASETS(config, validation_df, vocab,
                                           validation_sample_name_id_dict,
                                           valid_samples_info_df, 'validation')

    # 5. Encoder initialised from the pre-trained checkpoint
    bert = u.load_model(config, vocab, logger)
    imbert = u.load_pretrained_encoder(config, bert, device=device, logger=logger)

    output_type = config['FineTuning']['hyperparams']['output_type']
    cls_bert = _CLASSIFIER_BY_OUTPUT_TYPE[output_type](config, imbert, logger)
    print("output type: " + output_type)

    # 6. Freeze the lower encoder layers if requested
    total_encoder_layers = config['Model']['hyperparams'].as_int('num_hidden_layers')
    layer_idx = config[task]['hyperparams'].as_int('grad_start_layer')
    encoder_result_layer = config[task]['hyperparams'].as_int('encoder_result_layer')
    if encoder_result_layer == -1:
        encoder_result_layer = total_encoder_layers - 1

    if layer_idx == -1:
        # Train the classification head only.
        for name, param in cls_bert.named_parameters():
            if 'clf' not in name and 'mlp' not in name:
                param.requires_grad = False
    elif layer_idx > 0:
        logger.info(f'Encoder layers below index {layer_idx} are frozen.')
        for name, param in cls_bert.named_parameters():
            if name.startswith('imbert.encoder.layer') and int(name.split('.')[3]) < layer_idx:
                param.requires_grad = False
    else:
        for _, param in cls_bert.named_parameters():
            param.requires_grad = True

    for name, param in cls_bert.named_parameters():
        logger.info(f'{name}, param.requires_grad: {param.requires_grad}')

    cls_bert = cls_bert.to(device)
    cls_bert = DDP(module=cls_bert,
                   device_ids=[device.index] if device.type == 'cuda' else None,
                   find_unused_parameters=True)

    # 7. Criterion -- the heads return raw logits
    criterion = nn.BCEWithLogitsLoss().to(device)

    # 8. Optimizer and scheduler
    optimizer = u.load_optimizer(config, cls_bert, layer_idx)

    max_epochs = config[task]['hyperparams'].as_int('max_epochs')
    num_training_steps = len(train_dataloader) * max_epochs
    warmup_steps = int(num_training_steps * config[task]['hyperparams'].as_float('warmup_step_ratio'))
    scheduler_type = config[task]["hyperparams"]["scheduler_type"]
    print("scheduler: " + scheduler_type)
    if scheduler_type == "LinearWarmupDecay":
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)
    elif scheduler_type == "CosineAnnealingWarmRestart":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=1000, T_mult=2, eta_min=1e-6)
    elif scheduler_type == "CyclicLR":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=1e-6,
            max_lr=config["Training"]["hyperparams"].as_float("learning_rate"),
            step_size_up=500, mode="triangular2")
    else:
        raise ValueError(f"unknown scheduler_type {scheduler_type!r}")

    # 9. Training
    n_dense_layer = config[task]['hyperparams'].as_list('n_layers')
    max_clip_grad_norm = config[task]['hyperparams'].as_float('max_clip_grad_norm')
    dense_act = config[task]['hyperparams']['dense_act']

    logger.info("====================================================================")
    logger.info(f"Test name: {test_name}")
    logger.info(f"Sequence length: {config['Model']['hyperparams'].as_int('max_position_embeddings')}, "
                f"add_special_token: {config['COMMON'].as_bool('add_special_token')}")
    logger.info(f"Vocab size: {len(vocab)}, total parameters of CLS BERT: "
                f"{sum(p.numel() for p in cls_bert.parameters())}")
    logger.info(f"Init learning rate of AdamW: {optimizer.param_groups[0]['lr']}, "
                f"scheduler is {type(scheduler).__name__}")
    logger.info(f"Loss function is {criterion.__class__.__name__}")
    logger.info(f"Train dataloader has {len(train_dataloader)} batches per process")
    logger.info(f"num_training_steps: {num_training_steps},\twarmup_steps: {warmup_steps}")
    logger.info(f"Output type is {output_type}")
    logger.info(f"{encoder_result_layer}th layer of encoder will be used in fine-tuning")
    logger.info(f"{n_dense_layer} dense layers and {dense_act} function will be used")
    logger.info("====================================================================")

    training_post_fix_list, validation_post_fix_list = [], []
    train_gradient_history = {"gradients": {}}
    max_epochs_start_time = datetime.now()

    for epoch in range(max_epochs):
        epoch_start_time = datetime.now()
        train_sampler.set_epoch(epoch)

        train_data_iter = tqdm.tqdm(enumerate(train_dataloader),
                                    desc=f"EP_Training:{epoch}",
                                    total=len(train_dataloader),
                                    bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_train_loss_list, total_validation_loss_list = [], []
        train_gradient_history["gradients"][f'Epoch{epoch}'] = {}

        for step, data in train_data_iter:
            u.empty_cuda_cache()
            cls_bert.train()
            logger.info(f"============== Step{step} of Epoch{epoch} on {device} ==============")
            data = {key: value.to(device) for key, value in data.items()}
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Current learning rate: {current_lr}")

            disease_label_tensor = u.make_disease_label_loss_input(data, device)

            encoded_layer, cls_out = cls_bert(data["bert_answer"],
                                              data["segment_label"],
                                              data["attention_mask"])
            cls_loss = criterion(cls_out, disease_label_tensor.float())
            logger.info(f"==================== CLS loss: {cls_loss} ====================\n")

            cls_loss.backward()

            if rank == 0:
                gradients = {}
                for name, param in cls_bert.named_parameters():
                    if param.requires_grad:
                        gradients[name] = param.grad.norm().item() if param.grad is not None else 1e-13
                train_gradient_history["gradients"][f'Epoch{epoch}'][f'Step{step}'] = gradients

            _write_classification_result_table(
                config,
                _gather_classification_result(epoch, step, cls_out, data,
                                              train_sample_name_id_dict,
                                              train_bioprj_samples_dict,
                                              current_lr, rank),
                epoch, step, rank, 'train')

            nn.utils.clip_grad_norm_(cls_bert.parameters(), max_norm=max_clip_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            avg_loss += cls_loss.item()
            post_fix = {"GPU": rank,
                        "Epoch": epoch,
                        "Step": step,
                        "Training_CLS_loss": cls_loss.item(),
                        "Learning_rate": optimizer.param_groups[0]['lr']}
            training_post_fix_list.append(post_fix)
            if step % 10 == 0:
                train_data_iter.write(str(post_fix))

            cls_loss_tensor = torch.tensor([cls_loss.item()], device=device)
            gathered_cls_loss = [torch.zeros(1, device=device) for _ in range(world_size)]
            dist.all_gather(gathered_cls_loss, cls_loss_tensor)
            gathered_cls_loss = [tensor.item() for tensor in gathered_cls_loss]

            total_train_loss_list.append(
                pd.DataFrame([[epoch] * world_size, [step] * world_size,
                              list(np.arange(0, world_size)), gathered_cls_loss],
                             index=["epoch", "num_iter", "rank", "train_loss"]).T)

        train_epoch_loss = avg_loss / len(train_data_iter)
        train_data_iter.write(f"Learning rate is now {optimizer.param_groups[0]['lr']} and the train "
                              f"avg loss is {train_epoch_loss} in Epoch{epoch} at rank{rank}")
        pd.concat(total_train_loss_list, axis=0).to_csv(
            config["FineTuning"]["loss_dir"] + f"/epoch-{epoch}_train_loss.txt", sep="\t", index=False)

        if rank != 0:
            continue

        u.save_checkpoint(config, epoch, cls_bert, optimizer)
        print("\n" + "=" * 110)
        print(f"\t\t\t\t\t\tValidation in epoch{epoch}")
        print("=" * 110 + "\n")

        with torch.no_grad():
            cls_bert.eval()
            val_data_iter = tqdm.tqdm(enumerate(validation_dataloader),
                                      desc=f"EP_Validation:{epoch}",
                                      total=len(validation_dataloader),
                                      bar_format="{l_bar}{r_bar}")

            val_avg_loss = 0.0
            for step, data in val_data_iter:
                u.empty_cuda_cache()
                data = {key: value.to(device) for key, value in data.items()}
                disease_label_tensor = u.make_disease_label_loss_input(data, device)
                current_lr = optimizer.param_groups[0]['lr']

                encoded_layer, cls_out = cls_bert(data["bert_answer"],
                                                  data["segment_label"],
                                                  data["attention_mask"])
                cls_loss = criterion(cls_out, disease_label_tensor.float())
                val_avg_loss += cls_loss.item()

                total_validation_loss_list.append(
                    pd.DataFrame([epoch, step, rank, cls_loss.item()],
                                 index=["epoch", "num_iter", "rank", "validation_loss"]).T)

                _write_classification_result_table(
                    config,
                    _gather_classification_result(epoch, step, cls_out, data,
                                                  validation_sample_name_id_dict,
                                                  validation_bioprj_samples_dict,
                                                  current_lr, rank),
                    epoch, step, rank, 'validation')

                post_fix = {"GPU": rank,
                            "Epoch": epoch,
                            "Step": step,
                            "Validation_CLS_loss": cls_loss.item(),
                            "Learning_rate": current_lr}
                validation_post_fix_list.append(post_fix)
                if step % 10 == 0:
                    val_data_iter.write(str(post_fix))

            valid_epoch_loss = val_avg_loss / len(val_data_iter)
            val_data_iter.write(f"Validation learning rate is {optimizer.param_groups[0]['lr']} and "
                                f"the validation avg loss is {valid_epoch_loss} in Epoch{epoch}")

        print(f"Epoch{epoch} elapsed time: {datetime.now() - epoch_start_time}\n")
        pd.concat(total_validation_loss_list, axis=0).to_csv(
            config["FineTuning"]["loss_dir"] + f"/epoch-{epoch}_validation_loss.txt",
            sep="\t", index=False)

    max_epochs_end_time = datetime.now()
    logger.info(f"Start: {max_epochs_start_time}\nEnd: {max_epochs_end_time}\n"
                f"Elapsed time: {max_epochs_end_time - max_epochs_start_time} in {max_epochs} Epochs")

    print("=" * 110)
    print("Training gradient flow merging...")
    _write_gradient_flow_table(config, train_gradient_history)

    print("Training classification prediction merging...")
    _merge_prediction_results(config, 'train')
    _write_loss_and_lr(config, training_post_fix_list, 'train')

    print("Validation classification prediction merging...")
    _merge_prediction_results(config, 'validation')
    _write_loss_and_lr(config, validation_post_fix_list, 'validation')

    print("=" * 110)
    print(f"{test_name} elapsed time: {max_epochs_end_time - max_epochs_start_time} "
          f"in {max_epochs} Epochs")
    print("=" * 110 + "\n")


def _split_is_complete(split_dir):
    """True only once a split has written its merged validation table.

    The directory itself is created before training starts, so its mere
    existence means nothing -- an interrupted or failed run leaves one behind.
    The merged table is written at the very end of main_worker, which makes it
    a reliable completion marker.
    """
    pattern = os.path.join(split_dir, 'etc', '*_classification_result_in_validation.tsv')

    return bool(glob.glob(pattern))


def _run_random_splits(config, world_size):
    """Train one model per random train/validation split of the current cohort."""
    base_result_dir = config["FineTuning"]["result_dir"]
    u.make_task_result_directory(config)

    inputs = _PREPARE_INPUTS(config)
    for idx, split in enumerate(inputs):
        split_dir = os.path.join(base_result_dir, str(idx))
        config["FineTuning"]["result_dir"] = split_dir

        if _split_is_complete(split_dir):
            print(f"{idx}-th split already completed -> skipped")
            continue
        if os.path.exists(split_dir):
            print(f"{idx}-th split directory exists but holds no merged results "
                  f"(an earlier run was interrupted) -> running it again")
        u.make_task_result_directory(config)

        addr, port = u.find_free_port()
        mp.spawn(main_worker,
                 args=(world_size,
                       config,
                       split.train_samples_rank_df,
                       split.valid_samples_rank_df,
                       split.train_samples_info_df,
                       split.valid_samples_info_df,
                       split.train_sample_name_id_dict,
                       split.validation_sample_name_id_dict,
                       split.train_bioprj_samples_dict,
                       split.validation_bioprj_samples_dict,
                       split.vocab,
                       addr, port,),
                 nprocs=world_size,
                 join=True)

        print(f"{idx}-th random split completed; waiting for the devices to settle")
        time.sleep(_COOLDOWN_SECONDS)

    config["FineTuning"]["result_dir"] = base_result_dir


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=str,
                        help="path to the fine-tuning configuration file (.ini)")
    return parser.parse_args()


def _load_config(path):
    config = ConfigObj(path, file_error=True)
    task = config['COMMON']['task']
    if task != 'FineTuning':
        raise ValueError(f"expected [COMMON] task = FineTuning, found {task!r}")
    _check_hyperparams(config)
    return config


def _load_cohort_info(config):
    """Phenotype -> list of BioProjects, for every phenotype with >1 cohort.

    Read from the `DL_info_summary_<version>.xlsx` workbook that accompanies the
    preprocessed input table, sheet `phenotype-bioprjs_summary`, which must have
    the columns `Phenotype`, `Study_Accession` (comma separated) and `N_bioprjs`.
    Reading it needs the optional `openpyxl` dependency.
    """
    dataset = config["FineTuning"]["Dataset"]
    version = dataset["version"]
    summary_path = os.path.join(dataset["dataset_dir"], "preprocessed_bert_input_format",
                                version, f"DL_info_summary_{version}.xlsx")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"cohort summary workbook not found: {summary_path}")

    dataset_info = pd.read_excel(summary_path, sheet_name="phenotype-bioprjs_summary")
    dataset_info = dataset_info.loc[dataset_info["N_bioprjs"] > 1, :]

    return {phenotype: str(accessions).split(", ")
            for phenotype, accessions in zip(dataset_info["Phenotype"],
                                             dataset_info["Study_Accession"])}


def main():
    config = _load_config(_parse_args().config)
    world_size = u.get_world_size(config, 'FineTuning')
    base_result_dir = config["FineTuning"]["result_dir"]

    for target_disease, cohorts in _load_cohort_info(config).items():
        config["FineTuning"]["Dataset"]["train_bioprj"] = cohorts
        config["FineTuning"]["Dataset"]["valid_bioprj"] = cohorts
        config["FineTuning"]["Dataset"]["cv_bioprj"] = cohorts
        config["FineTuning"]["Dataset"]["target_disease"] = target_disease

        config["FineTuning"]["result_dir"] = os.path.join(base_result_dir, target_disease + "-CV")
        print(f"\n### phenotype {target_disease} | cohorts {', '.join(cohorts)} ###")

        _run_random_splits(config, world_size)

    config["FineTuning"]["result_dir"] = base_result_dir


if __name__ == "__main__":
    main()
