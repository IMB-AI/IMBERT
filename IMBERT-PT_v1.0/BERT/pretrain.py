"""Masked-language-model pre-training of IMBERT on microbial pathway profiles.

Usage
-----
    python pretrain.py --config pretrain_config.ini

The number of processes is derived from the `gpu_ids` entry of the [PreTrain]
section: one process per CUDA device, or a single CPU process when `gpu_ids`
is set to `cpu` (or when no CUDA device is available).

The masked-language-model head below (MaskedLanguageModel, BERTLM) is derived
from the same upstream implementation as bert.py and HAS BEEN MODIFIED: it
predicts metabolic pathway tokens rather than word pieces, and the
next-sentence-prediction head has been removed. See NOTICE.
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import tqdm
from configobj import ConfigObj
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.parallel import DistributedDataParallel as DDP

import optimizer as opt
import utils as u


class MaskedLanguageModel(nn.Module):
    """Predict the original token of a masked position.

    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, sequence_output):
        logits = self.linear(sequence_output) + self.bias
        prediction_scores = self.softmax(logits)

        return prediction_scores


class BERTLM(nn.Module):
    """IMBERT encoder + masked language model head."""

    def __init__(self, bert, vocab_size, hidden_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """
        super().__init__()
        self.bert = bert
        self.mask_lm = MaskedLanguageModel(hidden_size, vocab_size)

    def forward(self, x, segment_label, attention_mask):
        encoded_layers, pooled_output, attention_probs, attention_weight = self.bert(
            x, segment_label, attention_mask)

        return self.mask_lm(encoded_layers), pooled_output, attention_probs, attention_weight


def _save_prediction_status(config,
                            epoch,
                            num_iter,
                            mlm_output,
                            input_data,
                            sample_name_id_dict,
                            bioprj_samples_dict,
                            mask_token_id,
                            rank: int,
                            mode: str):
    test_name = config['PreTrain']['test_name']
    prediction_status_dir = config['PreTrain']['prediction_status_dir']
    status_file_path = os.path.join(
        prediction_status_dir,
        f"{test_name}_{mode}_epoch{epoch}_iter{num_iter}_prediction_gpu{rank}.txt")

    with open(status_file_path, 'w') as check:
        check.write('=' * 76 + '\n')
        check.write(f'Epoch: {epoch} | Iteration: {num_iter} | GPU: {rank}\n')
        for iter_idx, (mask_lm_output_sentence, answer_sentence, sample_name_id_tensor) in enumerate(
                zip(mlm_output, input_data['bert_answer'], input_data['sample_name_id'])):
            # mask_lm_output_sentence: [seq_len, vocab_size], answer_sentence: [seq_len]
            sample_name_id = list(sample_name_id_tensor.cpu().detach().numpy())[0]
            sample_name = "".join([sample for sample, ids in sample_name_id_dict.items()
                                   if ids == sample_name_id])
            bioproject = u.find_bioproject_from_sample(bioprj_samples_dict, sample_name)
            check.write(f"Sample ID: {sample_name_id}, Sample name: {sample_name} in {bioproject}\n")

            argmax_mask_lm_output_sentence = torch.argmax(mask_lm_output_sentence, dim=1)  # [seq_len]

            check.write(f"\nSentence order: {iter_idx}\n")
            input_sentence = input_data['bert_input'][iter_idx]
            check.write(f"\n1. Input sentence: \n{input_sentence}\n")

            y_pred = argmax_mask_lm_output_sentence
            check.write(f"\n2. Prediction sentence: \n{y_pred}\n")

            y_true = answer_sentence
            check.write(f"\n3. Answer sentence: \n{y_true}\n")

            masking_location = np.where(
                np.array(input_sentence.cpu().detach()) == mask_token_id)[0]
            check.write(f'\n4. Masking location: {masking_location}\n')

            y_pred_in_masking_location = y_pred[masking_location]
            check.write(f"\n5. Prediction in masking location: \n{y_pred_in_masking_location}\n")

            y_true_in_masking_location = y_true[masking_location]
            check.write(f"\n6. Answer in masking location: \n{y_true_in_masking_location}\n")

            y_true_np = y_true_in_masking_location.cpu().detach().numpy()
            y_pred_np = y_pred_in_masking_location.cpu().detach().numpy()
            check.write(f"\n7. Accuracy: {accuracy_score(y_true_np, y_pred_np) * 100}%\n")
            check.write(f"\n8. F1-score: {f1_score(y_true_np, y_pred_np, average='macro')}\n")
            check.write('=' * 76 + '\n')


def main_worker(rank,
                world_size,
                config,
                train_df,
                validation_df,
                true_train_samples_list,
                true_validation_samples_list,
                addr, port):
    task = config['COMMON']['task']
    pred_checkpoint = config[task].as_int("prediction_checkpoint")

    # 1. Devices and process group
    u.empty_cuda_cache()
    device_ids = u.resolve_device_ids(config, task)
    device = u.init_for_distributed(rank, device_ids, world_size, addr, port)

    # 2. Loggers
    train_log_path, valid_log_path = u.get_log_path(config, rank)
    logger = u.make_logger(train_log_path)
    valid_logger = u.make_logger(valid_log_path)

    # 3. Vocabulary
    vocab = u.make_vocabulary(config, train_df)
    logger.info(f"\nVocab size: {len(vocab)}\n")

    # 4. Sample bookkeeping
    train_sample_name_id_dict = u.make_sample_name_id_dict(true_train_samples_list, 'train')
    validation_sample_name_id_dict = u.make_sample_name_id_dict(true_validation_samples_list, 'validation')
    train_bioprj_samples_dict = u.gather_bioproject_samples(config, task, true_train_samples_list)
    validation_bioprj_samples_dict = u.gather_bioproject_samples(config, task, true_validation_samples_list)

    # 5. Dataloaders
    train_sampler, train_dataloader = u.load_datasets(config, train_df, vocab,
                                                      train_sample_name_id_dict, 'train')
    validation_dataloader = u.load_datasets(config, validation_df, vocab,
                                            validation_sample_name_id_dict, 'validation')

    # 6. Model
    hidden_size = config['Model']['hyperparams'].as_int('hidden_size')
    bert = u.load_model(config, vocab, logger)
    mlm_bert = BERTLM(bert, len(vocab), hidden_size)
    logger.info(f"Total parameters of MLM BERT: {sum(p.numel() for p in mlm_bert.parameters())}")
    mlm_bert = mlm_bert.to(device)
    mlm_bert = DDP(module=mlm_bert,
                   device_ids=[device.index] if device.type == 'cuda' else None,
                   find_unused_parameters=True)

    # 7. Criterion -- positions that are not masked carry the [PAD] label
    criterion = nn.NLLLoss(ignore_index=vocab['[PAD]']).to(device)

    # 8. Optimizer and scheduler
    warmup_steps, optimizer = u.load_optimizer(config, mlm_bert)
    scheduler = opt.ScheduledOptim(optimizer, hidden_size, logger, n_warmup_steps=warmup_steps)

    # 9. Training
    max_epochs = config[task]['hyperparams'].as_int('max_epochs')
    max_epochs_start_time = datetime.now()

    for epoch in range(max_epochs):
        epoch_start_time = datetime.now()
        train_sampler.set_epoch(epoch)

        train_data_iter = tqdm.tqdm(enumerate(train_dataloader),
                                    desc=f"EP_Training:{epoch}",
                                    total=len(train_dataloader),
                                    bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        mlm_bert.train()
        total_train_loss_list, total_validation_loss_list = [], []
        save_this_epoch = (epoch % pred_checkpoint == 0)

        for num_iter, data in train_data_iter:
            logger.info(f"GPU{rank}\t{num_iter}iter Start")
            data = {key: value.to(device) for key, value in data.items()}

            mask_lm_output, pooled_output, attention_probs, attention_weights = mlm_bert(
                data["bert_input"], data["segment_label"], data["attention_mask"])
            logger.info(f"GPU{rank}\t{num_iter}iter MLM output dimension: {mask_lm_output.shape}")

            # Back propagation
            logger.info(f"GPU{rank}\tBack propagation of {num_iter}iter")
            optimizer.zero_grad()
            mask_loss = criterion(mask_lm_output.transpose(1, 2), data["bert_label"])
            logger.info(f"GPU{rank}\t{num_iter}iter\tmask loss: {mask_loss}")

            # Collect the loss of every process
            mask_loss_tensor = torch.tensor([mask_loss.item()], device=device)
            gathered_mask_loss = [torch.zeros(1, device=device) for _ in range(world_size)]
            dist.all_gather(gathered_mask_loss, mask_loss_tensor)
            gathered_mask_loss = [tensor.item() for tensor in gathered_mask_loss]

            total_train_loss_list.append(
                pd.DataFrame([[epoch] * world_size,
                              [num_iter] * world_size,
                              list(np.arange(0, world_size)),
                              gathered_mask_loss],
                             index=["epoch", "num_iter", "rank", "train_loss"]).T)

            if save_this_epoch:
                _save_prediction_status(config, epoch, num_iter, mask_lm_output, data,
                                        train_sample_name_id_dict, train_bioprj_samples_dict,
                                        vocab['[MASK]'], rank, 'train')

            avg_loss += mask_loss.item()
            mask_loss.backward()
            optimizer.step()

            if num_iter % 100 == 0:
                print({"epoch": epoch,
                       "iter": num_iter,
                       "avg_loss": avg_loss / (int(num_iter) + 1),
                       "loss": mask_loss.item()})

            logger.info(f"GPU{rank}\t{num_iter}iter Done")
            logger.info("=" * 74)

        print(f"{config[task]['test_name']} | epoch {epoch} | "
              f"lr before update: {optimizer.param_groups[0]['lr']}")
        scheduler.step_and_update_lr()
        print(f"lr after update: {optimizer.param_groups[0]['lr']}")

        train_epoch_loss = avg_loss / len(train_data_iter)
        logger.info(f"GPU{rank}\tEP{epoch}_Training, avg_loss={train_epoch_loss}")

        if rank != 0:
            continue

        if save_this_epoch:
            print("start model saving")
            u.save_checkpoint(config, epoch, mlm_bert, optimizer)

        # Validation (master process only)
        mlm_bert.eval()
        with torch.no_grad():
            val_data_iter = tqdm.tqdm(enumerate(validation_dataloader),
                                      desc=f"EP_Validation:{epoch}",
                                      total=len(validation_dataloader),
                                      bar_format="{l_bar}{r_bar}")

            val_avg_loss = 0.0
            for num_iter, data in val_data_iter:
                valid_logger.info(f"GPU{rank}\tValidation {num_iter}iter Start")
                data = {key: value.to(device) for key, value in data.items()}

                mask_lm_output, pooled_output, attention_probs, attention_weights = mlm_bert(
                    data["bert_input"], data["segment_label"], data["attention_mask"])
                valid_logger.info(f"GPU{rank}\tValidation {num_iter}iter MLM output dimension: "
                                  f"{mask_lm_output.shape}")

                mask_loss = criterion(mask_lm_output.transpose(1, 2), data["bert_label"])
                valid_logger.info(f"GPU{rank}\tValidation {num_iter}iter\tmask loss: {mask_loss}")
                val_avg_loss += mask_loss.item()

                total_validation_loss_list.append(
                    pd.DataFrame([epoch, num_iter, rank, mask_loss.item()],
                                 index=["epoch", "num_iter", "rank", "validation_loss"]).T)

                if save_this_epoch:
                    _save_prediction_status(config, epoch, num_iter, mask_lm_output, data,
                                            validation_sample_name_id_dict,
                                            validation_bioprj_samples_dict,
                                            vocab['[MASK]'], rank, 'validation')

                if num_iter % 50 == 0:
                    print({"epoch": epoch,
                           "iter": num_iter,
                           "avg_loss": val_avg_loss / (int(num_iter) + 1),
                           "loss": mask_loss.item()})

            valid_epoch_loss = val_avg_loss / len(val_data_iter)
            valid_logger.info(f"EP{epoch}_Validation, avg_loss={valid_epoch_loss}")

        epoch_end_time = datetime.now()
        valid_logger.info(f"Epoch{epoch} elapsed time: {epoch_end_time - epoch_start_time}")

        loss_dir = config["PreTrain"]["loss_dir"]
        pd.concat(total_train_loss_list, axis=0).to_csv(
            f"{loss_dir}/epoch-{epoch}_train_loss.txt", sep="\t", index=False)
        pd.concat(total_validation_loss_list, axis=0).to_csv(
            f"{loss_dir}/epoch-{epoch}_validation_loss.txt", sep="\t", index=False)

    max_epochs_end_time = datetime.now()
    logger.info(f"Start: {max_epochs_start_time}\nEnd: {max_epochs_end_time}\n"
                f"Elapsed time: {max_epochs_end_time - max_epochs_start_time} in {max_epochs} Epochs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=str,
                        help="path to the pre-training configuration file (.ini)")
    args = parser.parse_args()

    config = ConfigObj(args.config, file_error=True)
    task = config['COMMON']['task']
    if task != 'PreTrain':
        raise ValueError(f"pretrain.py expects [COMMON] task = PreTrain, found {task!r}")

    world_size = u.get_world_size(config, task)
    addr, port = u.find_free_port()

    u.make_task_result_directory(config)

    train_df, true_train_samples_list = u.load_input_df_and_samples_list(config, task, 'train')
    validation_df, true_validation_samples_list = u.load_input_df_and_samples_list(config, task, 'validation')

    mp.spawn(main_worker,
             args=(world_size,
                   config,
                   train_df,
                   validation_df,
                   true_train_samples_list,
                   true_validation_samples_list,
                   addr, port,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    main()
