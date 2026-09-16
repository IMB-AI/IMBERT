"""Shared helpers for fine-tuning IMBERT on binary phenotype classification."""

import glob
import logging
import os
import socket
from dataclasses import dataclass

import pandas as pd
import torch
import torch.distributed as dist
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import BertConfig

import bert as Bert
from Fine_tuning import datasets as FTD

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SPECIAL_TOKENS = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4}
_SUB_DIRECTORY_LIST = ['etc',
                       'logs',
                       'losses',
                       'model_checkpoints',
                       'prediction_status']


@dataclass
class FineTuningInputData:
    vocab: dict
    train_samples_rank_df: pd.DataFrame
    valid_samples_rank_df: pd.DataFrame
    train_samples_info_df: pd.DataFrame
    valid_samples_info_df: pd.DataFrame
    train_sample_name_id_dict: dict
    validation_sample_name_id_dict: dict
    train_bioprj_samples_dict: dict
    validation_bioprj_samples_dict: dict


def prepare_binary_classification_v2(config):
    task = config['COMMON']['task']
    rank_path = config[task]['Dataset']['total_samples_rank_path']
    info_path = config[task]['Dataset']['sample_information_path']
    n_sampling_times = config[task]['Dataset'].as_int("n_sampling_times")

    rank_df = pd.read_csv(rank_path, sep='\t')
    total_sample_info_df = pd.read_csv(info_path, sep='\t', comment='#')
    rank_df = restrict_rank_df_to_pretrained_vocab(config, rank_df)

    input_data_list = []
    for i in range(0, n_sampling_times):
        train_samples_list, valid_samples_list = make_train_and_valid_samples_list(config, i) 
        train_samples_rank_df, valid_samples_rank_df = extract_samples_abundance_rank_v2(config,
                                                                                    task,
                                                                                    rank_df,
                                                                                    train_samples_list,
                                                                                    valid_samples_list)

        # Make vocab dict
        vocab = make_vocabulary(config, train_samples_rank_df)

        # sample name with id
        train_sample_name_id_dict = make_sample_name_id_dict(train_samples_list, 'train')
        validation_sample_name_id_dict = make_sample_name_id_dict(valid_samples_list, 'validation')

        write_true_train_and_validation_samples(config,
                                                train_sample_name_id_dict,
                                                'train')
        write_true_train_and_validation_samples(config,
                                                validation_sample_name_id_dict,
                                                'validation')

        train_samples_info_df, valid_samples_info_df = split_train_validation_samples_information(config,
                                                                                                task,
                                                                                                total_sample_info_df,
                                                                                                train_samples_list,
                                                                                                valid_samples_list)

        train_bioprj_samples_dict = gather_bioproject_samples(train_samples_info_df)
        validation_bioprj_samples_dict = gather_bioproject_samples(valid_samples_info_df)

        # Return as a dataclass
        input_data_list.append( FineTuningInputData(
            vocab=vocab,
            train_samples_rank_df=train_samples_rank_df,
            valid_samples_rank_df=valid_samples_rank_df,
            train_samples_info_df=train_samples_info_df,
            valid_samples_info_df=valid_samples_info_df,
            train_sample_name_id_dict=train_sample_name_id_dict,
            validation_sample_name_id_dict=validation_sample_name_id_dict,
            train_bioprj_samples_dict=train_bioprj_samples_dict,
            validation_bioprj_samples_dict=validation_bioprj_samples_dict
        ) )
        print(str(i) + " times sampling completed")
    
    return input_data_list



def prepare_binary_classification_HvsP(config):
    task = config['COMMON']['task']
    rank_path = config[task]['Dataset']['total_samples_rank_path']
    info_path = config[task]['Dataset']['sample_information_path']
    n_sampling_times = config[task]['Dataset'].as_int("n_sampling_times")

    excluding_datasets = config[task]["Dataset"]["exclude_bioprojects"]
    excluding_phenotypes = config[task]["Dataset"]["exclude_phenotypes"]

    rank_df = pd.read_csv(rank_path, sep='\t')
    total_sample_info_df = pd.read_csv(info_path, sep='\t', comment='#')
    if excluding_datasets != "None":
        print("excluding datasets: " + str(excluding_datasets))
        total_sample_info_df = total_sample_info_df.loc[~(total_sample_info_df["Study_Accession"].isin(excluding_datasets)), :]
    if excluding_phenotypes != "None":
        print("excluding phenotyes: " + str(excluding_phenotypes))
        total_sample_info_df = total_sample_info_df.loc[~(total_sample_info_df["Phenotype"].isin(excluding_phenotypes)), :]
    
    rank_df = restrict_rank_df_to_pretrained_vocab(config, rank_df)

    input_data_list = []
    for i in range(0, n_sampling_times):
        train_samples_list, valid_samples_list = make_train_and_valid_samples_list_HvsP(config, i) 
        train_samples_rank_df, valid_samples_rank_df = extract_samples_abundance_rank_v2(config,
                                                                                    task,
                                                                                    rank_df,
                                                                                    train_samples_list,
                                                                                    valid_samples_list)

        # Make vocab dict
        vocab = make_vocabulary(config, train_samples_rank_df)

        # sample name with id
        train_sample_name_id_dict = make_sample_name_id_dict(train_samples_list, 'train')
        validation_sample_name_id_dict = make_sample_name_id_dict(valid_samples_list, 'validation')

        write_true_train_and_validation_samples(config,
                                                train_sample_name_id_dict,
                                                'train')
        write_true_train_and_validation_samples(config,
                                                validation_sample_name_id_dict,
                                                'validation')

        
        train_samples_info_df, valid_samples_info_df = split_train_validation_samples_information(config,
                                                                                                task,
                                                                                                total_sample_info_df,
                                                                                                train_samples_list,
                                                                                                valid_samples_list)

        train_bioprj_samples_dict = gather_bioproject_samples(train_samples_info_df)
        validation_bioprj_samples_dict = gather_bioproject_samples(valid_samples_info_df)

        # Return as a dataclass
        input_data_list.append( FineTuningInputData(
            vocab=vocab,
            train_samples_rank_df=train_samples_rank_df,
            valid_samples_rank_df=valid_samples_rank_df,
            train_samples_info_df=train_samples_info_df,
            valid_samples_info_df=valid_samples_info_df,
            train_sample_name_id_dict=train_sample_name_id_dict,
            validation_sample_name_id_dict=validation_sample_name_id_dict,
            train_bioprj_samples_dict=train_bioprj_samples_dict,
            validation_bioprj_samples_dict=validation_bioprj_samples_dict
        ) )
        print(str(i) + " times sampling completed")
    
    return input_data_list

# --------------------------------------------------------------------------- #
# Devices / distributed setup
# --------------------------------------------------------------------------- #
def resolve_device_ids(config, task):
    """Return the CUDA device ids to use; an empty list means CPU.

    `gpu_ids` accepts a comma separated list of CUDA ids (`gpu_ids = 0, 1`), a
    single id, or the literal `cpu`/`none`.  When CUDA is unavailable the ids
    are ignored and the run falls back to CPU, so one configuration file works
    both on a laptop and on a multi-GPU server.
    """
    raw = config[task]['gpu_ids']
    if isinstance(raw, str):
        raw = [raw]
    device_ids = [str(x).strip() for x in raw if str(x).strip() != '']

    if len(device_ids) == 1 and device_ids[0].lower() in ('cpu', 'none'):
        device_ids = []

    if device_ids and not torch.cuda.is_available():
        print("[IMBERT] CUDA is not available -> running on CPU "
              f"(gpu_ids={device_ids} ignored).")
        device_ids = []

    if device_ids:
        n_visible = torch.cuda.device_count()
        invalid = [d for d in device_ids if not d.isdigit() or int(d) >= n_visible]
        if invalid:
            raise ValueError(
                f"gpu_ids contains ids that this machine does not have: {invalid} "
                f"(torch.cuda.device_count() == {n_visible})")

    return device_ids


def get_world_size(config, task):
    """Number of processes to spawn: one per GPU, or 1 on CPU."""
    return max(1, len(resolve_device_ids(config, task)))


def empty_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def init_for_distributed(rank, device_ids, world_size, addr, port):
    """Initialise the process group and return the `torch.device` for `rank`.

    A process group is created even for a single process so that
    `DistributedSampler` and `dist.all_gather` behave identically on one CPU
    and on four GPUs.
    """
    if device_ids:
        local_gpu_id = int(device_ids[rank])
        torch.cuda.set_device(local_gpu_id)
        device = torch.device(f'cuda:{local_gpu_id}')
        backend = 'nccl'
    else:
        device = torch.device('cpu')
        backend = 'gloo'

    print(f"[IMBERT] rank {rank}/{world_size} -> {device} (backend: {backend})")
    dist.init_process_group(backend=backend,
                            init_method=f"tcp://{addr}:{port}",
                            world_size=world_size,
                            rank=rank)
    dist.barrier()

    return device


def find_free_port():
    master_addr = "localhost"
    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    
    return master_addr, port


def get_log_path(config, rank):
    task = config['COMMON']['task']
    test_name = config[task]['test_name']
    log_dir = config[task]['log_dir']

    train_log_path = os.path.join(log_dir, f"{task}_{test_name}_training_gpu{rank}.log")

    return train_log_path


def make_logger(log_path):
    if os.path.exists(log_path):
        os.remove(log_path)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def write_true_train_and_validation_samples(config,
                                            sample_name_id_dict,
                                            mode: str):
    task = config['COMMON']['task']
    test_name, etc_dir = get_test_name_and_result_dir(config, task, 'etc_dir')

    samples_path = os.path.join(etc_dir, f"{test_name}_true_{mode}_samples_and_id.tsv")
    with open(samples_path, 'w') as true_set:
        header = ['Sample', 'ID']
        true_set.write("\t".join(header) + '\n')
        for sample, sid in sample_name_id_dict.items():
            row = [sample, sid]
            true_set.write("\t".join(list(map(str, row))) + '\n')


def make_task_result_directory(config):
    task = config['COMMON']['task']
    _, result_dir = get_test_name_and_result_dir(config, task, 'result_dir')

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for sub_directory in _SUB_DIRECTORY_LIST:
        sub_dir = os.path.join(result_dir, sub_directory)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)


def find_bioproject_from_sample(bioprj_samples_dict, sample_name):
    candidate_bioproject_list = [bioprj for bioprj, samples_list in bioprj_samples_dict.items() if
                                 sample_name in samples_list]
    if len(candidate_bioproject_list) == 1:
        return "".join(candidate_bioproject_list)

    elif len(candidate_bioproject_list) > 1:
        return ",".join(candidate_bioproject_list)

    elif len(candidate_bioproject_list) == 0:
        return 'Unknown'


def _build_dataloader(config, inhouse_df, vocab_dict, sample_name_id_dict,
                      disease_sample_list, mode: str):
    task = config['COMMON']['task']
    add_special_token = config['COMMON'].as_bool('add_special_token')
    seq_len = config["Model"]['hyperparams'].as_int("max_position_embeddings")
    batch_size = config[task]['hyperparams'].as_int('batch_size')

    world_size = get_world_size(config, task)
    use_cuda = len(resolve_device_ids(config, task)) > 0
    per_process_batch_size = max(1, int(batch_size / world_size))
    num_workers = 4 if use_cuda else 0

    print(f"\nCreating {task} {mode} Dataloader "
          f"(seq_len={seq_len}, batch_size/process={per_process_batch_size})")

    dataset = FTD.IMBDatasetForBinaryClassification(inhouse_df,
                                                    vocab_dict,
                                                    sample_name_id_dict,
                                                    disease_sample_list,
                                                    seq_len,
                                                    add_special_token)

    if mode == 'train':
        sampler = DistributedSampler(dataset, shuffle=True)
        dataloader = DataLoader(dataset,
                                batch_size=per_process_batch_size,
                                num_workers=num_workers,
                                shuffle=False,
                                pin_memory=use_cuda,
                                sampler=sampler,
                                drop_last=False)
        return sampler, dataloader

    return DataLoader(dataset,
                      batch_size=per_process_batch_size,
                      num_workers=num_workers,
                      shuffle=False,
                      pin_memory=use_cuda,
                      drop_last=False)


def load_datasets(config, inhouse_df, vocab_dict, sample_name_id_dict, sample_info_df, mode: str):
    """Dataloader for `target_disease` patients versus healthy controls.

    mode == 'train'      -> return (sampler, dataloader)
    mode == 'validation' -> return dataloader
    """
    disease = config["FineTuning"]["Dataset"]["target_disease"]
    disease_sample_list = [row.Run_Id for row in sample_info_df.itertuples()
                           if (row.Subject_Health_Status == 'Non-healthy')
                           and (row.Phenotype == disease)]

    return _build_dataloader(config, inhouse_df, vocab_dict, sample_name_id_dict,
                             disease_sample_list, mode)


def load_datasets_HvsP(config, inhouse_df, vocab_dict, sample_name_id_dict, sample_info_df, mode: str):
    """Dataloader for the pooled healthy-versus-patient task.

    Every non-healthy sample counts as a positive, whatever its phenotype.

    mode == 'train'      -> return (sampler, dataloader)
    mode == 'validation' -> return dataloader
    """
    disease_sample_list = [row.Run_Id for row in sample_info_df.itertuples()
                           if row.Subject_Health_Status == 'Non-healthy']

    return _build_dataloader(config, inhouse_df, vocab_dict, sample_name_id_dict,
                             disease_sample_list, mode)


def get_test_name_and_result_dir(config, task: str, key: str):
    """
    config_key; *_dir in task
    task: PreTrain or FineTuning
    """
    return config[task]['test_name'], config[task][key]


def save_checkpoint(config, epoch, model, optimizer):
    task = config['COMMON']['task']
    test_name, ckp_dir = get_test_name_and_result_dir(config, task, 'model_checkpoints_dir')

    checkpoint = {'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}

    file_name = f"Epoch{epoch}_checkpoint_{task}_{test_name}"
    torch.save(checkpoint, os.path.join(ckp_dir, f'{file_name}.ckp'))
    print(f"Save checkpoint at epoch{epoch} complete")


def read_vocab_file(vocab_path):
    """Read a `id<TAB>token` vocabulary file and return {token: id}."""
    vocab_dict = {}
    with open(vocab_path) as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.rstrip('\n')
            if not line:
                continue
            fields = line.split('\t')
            if len(fields) != 2:
                raise ValueError(f"{vocab_path}:{line_no}: expected 'id<TAB>token', got {line!r}")
            token_id, token = fields
            vocab_dict[token] = int(token_id)

    _validate_vocab(vocab_dict, vocab_path)
    return vocab_dict


def _validate_vocab(vocab_dict, source):
    for token, expected_id in SPECIAL_TOKENS.items():
        if token not in vocab_dict:
            raise ValueError(f"{source}: special token {token} is missing")
        if vocab_dict[token] != expected_id:
            raise ValueError(f"{source}: {token} must have id {expected_id}, "
                             f"found {vocab_dict[token]}")

    ids = sorted(vocab_dict.values())
    if len(ids) != len(set(ids)):
        raise ValueError(f"{source}: duplicated token ids")
    if ids != list(range(len(ids))):
        raise ValueError(f"{source}: token ids must be contiguous and start at 0 "
                         f"(got {ids[0]}..{ids[-1]} for {len(ids)} tokens)")


def get_pretrained_vocab_path(config):
    """Path of the vocabulary that belongs to the pre-trained checkpoint, or None."""
    return config['PreTrain']['Dataset'].get('vocab_path') or None


def load_pretrained_vocab(config):
    vocab_path = get_pretrained_vocab_path(config)
    if not vocab_path:
        return None
    return read_vocab_file(vocab_path)


def restrict_rank_df_to_pretrained_vocab(config, rank_df):
    """Drop pathways the pre-trained model has never seen.

    The model has one embedding row per pathway of its own vocabulary, so a
    fine-tuning table that contains extra pathways has to be reduced to that
    vocabulary first.
    """
    vocab_dict = load_pretrained_vocab(config)
    if vocab_dict is None:
        print("[IMBERT] no PreTrain vocab_path configured -> using the full input vocabulary")
        return rank_df

    known_pathways = set(vocab_dict) - set(SPECIAL_TOKENS)
    restricted = rank_df.loc[rank_df["Unnamed: 0"].isin(known_pathways), :].reset_index(drop=True)

    dropped = len(rank_df) - len(restricted)
    if dropped:
        print(f"[IMBERT] {dropped} pathway(s) of the input table are unknown to the "
              f"pre-trained model and were dropped")
    print(f"[IMBERT] {len(restricted)} of {len(known_pathways)} vocabulary pathways "
          f"are present in the input table")

    return restricted


def _save_vocab_dict(config, vocab_dict):
    """Record the vocabulary actually used, next to the run's other outputs."""
    task = config['COMMON']['task']
    test_name, etc_dir = get_test_name_and_result_dir(config, task, 'etc_dir')

    vocab_path = os.path.join(etc_dir, f"{test_name}_IMB_pathway_vocab.tsv")
    with open(vocab_path, 'w') as handle:
        for token, token_id in sorted(vocab_dict.items(), key=lambda kv: kv[1]):
            handle.write(f"{token_id}\t{token}\n")


def make_vocabulary(config, df):
    """Return the {pathway: token_id} mapping used to encode the samples.

    When `vocab_path` is configured under [PreTrain][[Dataset]] the ids come
    from that file and are therefore identical to the ones the encoder was
    pre-trained with -- this is what keeps a released checkpoint usable on a
    new cohort.  Without it a fresh vocabulary is derived from the row order of
    `df`, which is only correct when the encoder is trained from scratch.
    """
    vocab_dict = load_pretrained_vocab(config)

    if vocab_dict is None:
        print("[IMBERT] WARNING: building a vocabulary from the input table. "
              "Pre-trained weights will only line up if this table has exactly the "
              "same pathways, in the same order, as the pre-training table.")
        vocab_dict = {pwy: 5 + n for n, pwy in enumerate(df['Unnamed: 0'].to_list())}
        vocab_dict.update(SPECIAL_TOKENS)
        _validate_vocab(vocab_dict, 'input table')

    _save_vocab_dict(config, vocab_dict)

    return vocab_dict


def make_sample_name_id_dict(true_samples_list, mode: str):
    if mode == 'train':
        i = 1000
    elif mode == 'validation':
        i = 50000

    sample_name_id_dict = {}
    for sample in true_samples_list:
        sample_name_id_dict[sample] = i
        i += 1

    return sample_name_id_dict


def split_train_validation_samples_information(config,
                                               task,
                                               total_sample_info_df,
                                               train_samples_list,
                                               validation_samples_list):
    test_name, etc_dir = get_test_name_and_result_dir(config, task, 'etc_dir')
    train_samples_info_df = total_sample_info_df[
        total_sample_info_df['Run_Id'].isin(train_samples_list)].reset_index().drop(columns='index')
    valid_samples_info_df = total_sample_info_df[
        total_sample_info_df['Run_Id'].isin(validation_samples_list)].reset_index().drop(columns='index')

    train_samples_info_df.to_csv(os.path.join(etc_dir, f'{test_name}_train_samples_information.tsv'), sep='\t',
                                 index=False)
    valid_samples_info_df.to_csv(os.path.join(etc_dir, f'{test_name}_validation_samples_information.tsv'), sep='\t',
                                 index=False)

    return train_samples_info_df, valid_samples_info_df


def gather_bioproject_samples(samples_info_df):
    bioprj_samples_dict = {}
    for row in samples_info_df.itertuples():
        bioprj = row.Study_Accession
        sample = row.Run_Id
        if bioprj not in bioprj_samples_dict:
            bioprj_samples_dict[bioprj] = list()
        bioprj_samples_dict[bioprj].append(sample)

    return bioprj_samples_dict


def extract_samples_abundance_rank_v2(config, task, rank_df, train_samples_list, valid_samples_list):
    train_samples_rank_df = rank_df[['Unnamed: 0'] + train_samples_list]
    valid_samples_rank_df = rank_df[['Unnamed: 0'] + valid_samples_list]

    test_name, etc_dir = get_test_name_and_result_dir(config, task, 'etc_dir')
    train_samples_rank_df.to_csv(os.path.join(etc_dir, f'{test_name}_train_samples_rank.tsv'), sep='\t',
                                 index=False)
    valid_samples_rank_df.to_csv(os.path.join(etc_dir, f'{test_name}_validation_samples_rank.tsv'), sep='\t',
                                 index=False)

    return train_samples_rank_df, valid_samples_rank_df


def _convert_str_bioprojects_to_list(config, mode: str):
    """BioProject accessions of one split.

    configobj already splits a comma separated value into a list, but returns a
    plain string for a single value -- handle both.
    """
    task = config['COMMON']['task']
    key = 'train_bioprj' if mode == 'train' else 'valid_bioprj'
    bioprojects = config[task]['Dataset'][key]
    if isinstance(bioprojects, str):
        bioprojects = bioprojects.split(',')

    return [str(bioprj).strip() for bioprj in bioprojects if str(bioprj).strip()]


def make_train_and_valid_samples_list(config, i):
    task = config['COMMON']['task']
    target_disease = config[task]["Dataset"]["target_disease"]

    sample_info_path = config[task]['Dataset']['sample_information_path']
    prediction_type = config[task]['Dataset']['pred_type']  # 'cross_cohort' or 'cross_validation'
    if prediction_type == 'cross_cohort':
        train_bioproject_list = _convert_str_bioprojects_to_list(config, 'train')
        valid_bioproject_list = _convert_str_bioprojects_to_list(config, 'validation')

        train_samples_list, valid_samples_list = _get_cross_cohort_samples(target_disease,
                                                                           sample_info_path,
                                                                           train_bioproject_list,
                                                                           valid_bioproject_list)
    else:
        cv_bioproject = config[task]['Dataset'].as_list('cv_bioprj')
        cv_ratio = config[task]['Dataset'].as_float('cv_ratio')
        # train_samples_list, valid_samples_list = _get_cross_validation_samples(target_disease,
        #                                                                        cv_ratio,
        #                                                                        sample_info_path,
        #                                                                        cv_bioproject)
        train_samples_list, valid_samples_list = _get_cross_validation_samples_v2(target_disease,
                                                                               cv_ratio,
                                                                               sample_info_path,
                                                                               cv_bioproject, i)

    return train_samples_list, valid_samples_list


def make_train_and_valid_samples_list_HvsP(config, i):
    task = config['COMMON']['task']
    target_disease = config[task]["Dataset"]["target_disease"]

    sample_info_path = config[task]['Dataset']['sample_information_path']
    prediction_type = config[task]['Dataset']['pred_type']  # 'cross_cohort' or 'cross_validation'
    if prediction_type == 'cross_cohort':
        train_bioproject_list = _convert_str_bioprojects_to_list(config, 'train')
        valid_bioproject_list = _convert_str_bioprojects_to_list(config, 'validation')

        train_samples_list, valid_samples_list = _get_cross_cohort_samples(target_disease,
                                                                           sample_info_path,
                                                                           train_bioproject_list,
                                                                           valid_bioproject_list)
    else:
        cv_bioproject = config[task]['Dataset'].as_list('cv_bioprj')
        cv_ratio = config[task]['Dataset'].as_float('cv_ratio')
        # train_samples_list, valid_samples_list = _get_cross_validation_samples(target_disease,
        #                                                                        cv_ratio,
        #                                                                        sample_info_path,
        #                                                                        cv_bioproject)
        train_samples_list, valid_samples_list = _get_cross_validation_samples_HvsP(target_disease,
                                                                               cv_ratio,
                                                                               sample_info_path,
                                                                               cv_bioproject, i)

    return train_samples_list, valid_samples_list


def _get_cross_cohort_samples(target_disease, sample_info_path, train_bioproject_list, valid_bioproject_list):
    """Split by BioProject: train on one set of cohorts, validate on another.

    Selects the patients of `target_disease` plus every healthy control.
    (The original implementation read this file by column position and skipped
    the first three rows; it is read by column name here, which is what the
    rest of the code base does and what the shipped metadata file expects.)
    """
    info_df = pd.read_csv(sample_info_path, sep='\t', comment='#')
    selected = info_df.loc[(info_df['Phenotype'] == target_disease)
                           | (info_df['Subject_Health_Status'] == 'Healthy'), :]

    train_samples_list = selected.loc[selected['Study_Accession'].isin(train_bioproject_list),
                                      'Run_Id'].tolist()
    valid_samples_list = selected.loc[selected['Study_Accession'].isin(valid_bioproject_list),
                                      'Run_Id'].tolist()

    return train_samples_list, valid_samples_list


def _get_cross_validation_samples_HvsP(target_disease, cv_ratio, sample_info_path, cv_bioproject, i):
    info_df = pd.read_csv(sample_info_path, sep="\t", comment="#")
    target_info_df_h = info_df.loc[(info_df["Subject_Health_Status"] == "Healthy") & (info_df["DL_Group"] == "Fine-tuning"), :]
    target_info_df_d = info_df.loc[(info_df["Subject_Health_Status"] == "Non-healthy") & (info_df["DL_Group"] == "Fine-tuning"), :]

    x_train_h, x_test_h, y_train_h, y_test_h  = train_test_split(target_info_df_h, target_info_df_h["Subject_Health_Status"], 
                                                         train_size=cv_ratio, random_state=i+len(target_info_df_h))
    x_train_d, x_test_d, y_train_d, y_test_d  = train_test_split(target_info_df_d, target_info_df_d["Subject_Health_Status"], 
                                                         train_size=cv_ratio, random_state=i+len(target_info_df_d))

    x_train = pd.concat([x_train_h, x_train_d], axis=0)
    x_test = pd.concat([x_test_h, x_test_d], axis=0)
    
    #train_h_samples_list = random.sample(h_samples_list, round(len(h_samples_list) * cv_ratio))
    #train_d_samples_list = random.sample(d_samples_list, round(len(d_samples_list) * cv_ratio))

    #valid_h_samples_list = [sample for sample in h_samples_list if sample not in train_h_samples_list]
    #valid_d_samples_list = [sample for sample in d_samples_list if sample not in train_d_samples_list]

    #total_train_samples_list = train_h_samples_list + train_d_samples_list
    total_train_samples_list = x_train["Run_Id"].tolist()
    #total_valid_samples_list = valid_h_samples_list + valid_d_samples_list
    total_valid_samples_list = x_test["Run_Id"].tolist()

    return total_train_samples_list, total_valid_samples_list



def _get_cross_validation_samples_v2(target_disease, cv_ratio, sample_info_path, cv_bioproject, i):
    info_df = pd.read_csv(sample_info_path, sep="\t", comment="#")
    target_info_df = info_df.loc[info_df["Study_Accession"].isin(cv_bioproject), :]
    target_info_df_h = target_info_df.loc[target_info_df["Subject_Health_Status"] == "Healthy", :]
    target_info_df_d = target_info_df.loc[target_info_df["Phenotype"] == target_disease, :]

    x_train_h, x_test_h, y_train_h, y_test_h  = train_test_split(target_info_df_h, target_info_df_h["Phenotype"], 
                                                         train_size=cv_ratio, random_state=i+len(target_info_df_h))
    x_train_d, x_test_d, y_train_d, y_test_d  = train_test_split(target_info_df_d, target_info_df_d["Phenotype"], 
                                                         train_size=cv_ratio, random_state=i+len(target_info_df_d))

    x_train = pd.concat([x_train_h, x_train_d], axis=0)
    x_test = pd.concat([x_test_h, x_test_d], axis=0)
    
    #train_h_samples_list = random.sample(h_samples_list, round(len(h_samples_list) * cv_ratio))
    #train_d_samples_list = random.sample(d_samples_list, round(len(d_samples_list) * cv_ratio))

    #valid_h_samples_list = [sample for sample in h_samples_list if sample not in train_h_samples_list]
    #valid_d_samples_list = [sample for sample in d_samples_list if sample not in train_d_samples_list]

    #total_train_samples_list = train_h_samples_list + train_d_samples_list
    total_train_samples_list = x_train["Run_Id"].tolist()
    #total_valid_samples_list = valid_h_samples_list + valid_d_samples_list
    total_valid_samples_list = x_test["Run_Id"].tolist()

    return total_train_samples_list, total_valid_samples_list




def load_model(config, vocab, logger):
    task = config['COMMON']['task']

    if task == 'FineTuning':
        finetuning_encoder_result_layer = config['FineTuning']['hyperparams'].as_int('encoder_result_layer')
    else:
        finetuning_encoder_result_layer = -1

    model_hyperparams_obj = config['Model']['hyperparams']
    hidden_size = model_hyperparams_obj.as_int('hidden_size')
    num_hidden_layers = model_hyperparams_obj.as_int('num_hidden_layers')
    num_attention_heads = model_hyperparams_obj.as_int('num_attention_heads')
    intermediate_size = model_hyperparams_obj.as_int('intermediate_size')
    hidden_act = model_hyperparams_obj['hidden_act']
    hidden_dropout_prob = model_hyperparams_obj.as_float('hidden_dropout_prob')
    attention_probs_dropout_prob = model_hyperparams_obj.as_float('attention_probs_dropout_prob')
    max_position_embeddings = model_hyperparams_obj.as_int('max_position_embeddings')
    type_vocab_size = model_hyperparams_obj.as_int('type_vocab_size')
    initializer_range = model_hyperparams_obj.as_float('initializer_range')
    model_bias = model_hyperparams_obj.as_bool('model_bias')

    # Model
    bert_config = BertConfig(vocab_size=len(vocab),
                             hidden_size=hidden_size,
                             num_hidden_layers=num_hidden_layers,
                             num_attention_heads=num_attention_heads,
                             intermediate_size=intermediate_size,
                             hidden_act=hidden_act,
                             hidden_dropout_prob=hidden_dropout_prob,
                             attention_probs_dropout_prob=attention_probs_dropout_prob,
                             max_position_embeddings=max_position_embeddings,
                             type_vocab_size=type_vocab_size,
                             initializer_range=initializer_range,
                             encoder_result_layer=finetuning_encoder_result_layer,
                             model_bias=model_bias)

    bert = Bert.BertModel(bert_config, logger)

    logger.info("====================================================================")
    logger.info("\t\t\tHyperparameters of IMBERT")
    logger.info("====================================================================")
    logger.info(f"Hidden dimension: {bert_config.hidden_size}")
    logger.info(f"The number of transformer block: {bert_config.num_hidden_layers}")
    logger.info(f"The number of Attention heads: {bert_config.num_attention_heads}")
    logger.info(f"Intermediate dimension: {bert_config.intermediate_size}")
    logger.info(f"{bert_config.encoder_result_layer}th layer will be used in Fine-tuning")

    return bert


def load_optimizer(config, model, layer_idx):
    trainer_hyperparams_obj = config['Training']['hyperparams']
    learning_rate = trainer_hyperparams_obj.as_float('learning_rate')
    adam_beta1 = trainer_hyperparams_obj.as_float('adam_beta1')
    adam_beta2 = trainer_hyperparams_obj.as_float('adam_beta2')
    adam_weight_decay = trainer_hyperparams_obj.as_float('adam_weight_decay')

    betas = (adam_beta1, adam_beta2)
    if layer_idx > 0:
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=learning_rate,
                          betas=betas,
                          weight_decay=adam_weight_decay,
                          eps=1e-08)
    else:
        optimizer = AdamW(model.parameters(),
                          lr=learning_rate,
                          betas=betas,
                          weight_decay=adam_weight_decay,
                          eps=1e-08)

    return optimizer


def make_disease_label_loss_input(data, device):
    """Collapse the padded per-sample disease label into one value per sample."""
    disease_label_list = [[sum(disease_label)] for disease_label in data['disease_label']]

    return torch.tensor(disease_label_list).to(device)


# --------------------------------------------------------------------------- #
# Pre-trained weights
# --------------------------------------------------------------------------- #
def get_pretrained_checkpoint_path(config):
    """Path of the pre-trained checkpoint to start fine-tuning from.

    `checkpoint_path` wins when it is set; otherwise the path is rebuilt from
    the [PreTrain] directory, test name and epoch, which is how the checkpoints
    written by pretrain.py are named.
    """
    explicit = config['PreTrain'].get('checkpoint_path')
    if explicit:
        return explicit

    return os.path.join(config["PreTrain"]["model_checkpoints_dir"],
                        "Epoch" + config["FineTuning"]["weight_immigration_epoch"]
                        + "_checkpoint_PreTrain_" + config["PreTrain"]["test_name"] + ".ckp")


def strip_checkpoint_prefixes(state_dict):
    """Map pre-training parameter names onto the fine-tuning encoder.

    Pre-training runs under DistributedDataParallel and wraps the encoder in an
    MLM model, so its keys look like `module.bert.encoder...`.  The fine-tuning
    encoder is a bare `BertModel`, and the MLM head is dropped.
    """
    encoder_state = {}
    for key, value in state_dict.items():
        name = key
        for prefix in ('module.', 'bert.'):
            if name.startswith(prefix):
                name = name[len(prefix):]
        if name.startswith('mask_lm.'):
            continue
        encoder_state[name] = value

    return encoder_state


def load_checkpoint(path, map_location='cpu'):
    """`torch.load` that works across PyTorch versions.

    PyTorch 2.6 flipped the default of `weights_only` to True.  Checkpoints
    written by pretrain.py also carry the optimizer state, which contains numpy
    scalars, so they are rejected in that mode.  Try the safe mode first and
    only fall back -- with a warning -- when the file needs the legacy loader.

    Use tools/export_pretrained_checkpoint.py to turn a training checkpoint
    into a tensors-only file that always loads in safe mode.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # PyTorch < 1.13 has no weights_only argument.
        return torch.load(path, map_location=map_location)
    except Exception:
        print(f"[IMBERT] WARNING: {os.path.basename(path)} cannot be read in "
              f"weights_only mode (it carries optimizer state), falling back to the "
              f"legacy loader. Only do this for checkpoints you trust.")
        return torch.load(path, map_location=map_location, weights_only=False)


def _missing_checkpoint_message(weights_path):
    """A 'checkpoint not found' error that says what to do about it."""
    lines = [f"pre-trained checkpoint not found: {weights_path}", ""]

    # Look for something usable next to where the config pointed, and in the
    # repository root, so the message can name a concrete file.
    search_dirs, seen = [], set()
    for d in (os.path.dirname(os.path.abspath(weights_path)),
              os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))):
        if d not in seen:
            seen.add(d)
            search_dirs.append(d)

    found = []
    for d in search_dirs:
        if os.path.isdir(d):
            for pattern in ('*.pt', '*.ckp'):
                found.extend(sorted(glob.glob(os.path.join(d, pattern))))

    if found:
        lines.append("Checkpoint-looking files found nearby -- point "
                     "[PreTrain] checkpoint_path at one of these:")
        lines.extend(f"    {p}" for p in found)
    else:
        lines.append("Set [PreTrain] checkpoint_path in the configuration file to one of:")
        lines.append("    imbert_v1.0_pretrained.pt              (portable release file)")
        lines.append("    imbert_v1.0_pretrained_epoch15000.ckp  (raw training checkpoint)")

    lines += [
        "",
        "Both file types work. To turn a raw .ckp into the smaller portable .pt:",
        "    cd IMBERT-PT_v1.0/BERT",
        "    python export_checkpoint.py --checkpoint <file>.ckp \\",
        "                                --vocab <vocab>.tsv \\",
        "                                --output imbert_v1.0_pretrained.pt",
        "",
        "Or run pretrain.py first to train an encoder from scratch.",
    ]

    return "\n".join(lines)


def load_pretrained_encoder(config, model, device=None, logger=None):
    """Copy the pre-trained IMBERT weights into `model` (a `BertModel`).

    Parameters are matched **by name**, and `load_state_dict(strict=True)`
    refuses anything that does not line up, so a checkpoint trained with a
    different vocabulary or a different number of layers fails loudly instead
    of silently producing a mis-wired model.
    """
    weights_path = get_pretrained_checkpoint_path(config)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(_missing_checkpoint_message(weights_path))

    map_location = device if device is not None else 'cpu'
    checkpoint = load_checkpoint(weights_path, map_location=map_location)
    encoder_state = strip_checkpoint_prefixes(checkpoint['model_state_dict'])

    model_state = model.state_dict()
    checkpoint_vocab_size = encoder_state['embeddings.word_embeddings.weight'].shape[0]
    model_vocab_size = model_state['embeddings.word_embeddings.weight'].shape[0]
    if checkpoint_vocab_size != model_vocab_size:
        raise ValueError(
            f"vocabulary size mismatch: the checkpoint was trained with "
            f"{checkpoint_vocab_size} tokens but this run builds a model with "
            f"{model_vocab_size}. Point [PreTrain][[Dataset]] vocab_path at the "
            f"vocabulary file that was released with the checkpoint.")

    missing = sorted(set(model_state) - set(encoder_state))
    unexpected = sorted(set(encoder_state) - set(model_state))
    if missing or unexpected:
        raise ValueError(f"checkpoint does not match the model.\n"
                         f"  missing in checkpoint : {missing}\n"
                         f"  unexpected in checkpoint: {unexpected}")

    model.load_state_dict(encoder_state, strict=True)

    message = (f"Loaded {len(encoder_state)} pre-trained tensors from {weights_path} "
               f"(epoch {checkpoint.get('epoch', 'n/a')}, vocab size {checkpoint_vocab_size})")
    print(f"[IMBERT] {message}")
    if logger is not None:
        logger.info(message)

    return model
