"""Shared helpers for IMBERT masked-language-model pre-training."""

import logging
import os
import socket

import pandas as pd
import torch
import torch.distributed as dist

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import BertConfig

import bert as Bert
import Pathway_abundance.datasets as PAD

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SPECIAL_TOKENS = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4}


# --------------------------------------------------------------------------- #
# Devices / distributed setup
# --------------------------------------------------------------------------- #
def resolve_device_ids(config, task):
    """Return the list of CUDA device ids to use; an empty list means CPU.

    `gpu_ids` in the configuration file accepts either a comma separated list
    of CUDA device ids (``gpu_ids = 0, 1, 2, 3``), a single id (``gpu_ids = 0``)
    or the literal ``cpu`` / ``none`` to force CPU execution.  If CUDA is not
    available the requested ids are ignored and CPU is used instead, so the
    same configuration file runs on a workstation and on a multi-GPU server.
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


def setup_for_distributed(is_master):
    """Silence `print` in every process but the master one."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_for_distributed(rank, device_ids, world_size, addr, port):
    """Initialise the process group and return the `torch.device` for `rank`.

    A process group is created even for a single process (world_size == 1) so
    that `DistributedSampler` and the `dist.all_gather` calls in the training
    loop behave identically on one CPU and on eight GPUs.
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
    setup_for_distributed(rank == 0)

    return device


def find_free_port():
    master_addr = "localhost"

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()

    return master_addr, port


def empty_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_checkpoint(path, map_location='cpu'):
    """`torch.load` that works across PyTorch versions.

    PyTorch 2.6 flipped the default of `weights_only` to True.  Checkpoints
    written by pretrain.py also carry the optimizer state, which contains numpy
    scalars, so they are rejected in that mode.  Try the safe mode first and
    only fall back -- with a warning -- when the file needs the legacy loader.

    Use export_checkpoint.py to turn a training checkpoint into a tensors-only
    file that always loads in safe mode.
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


def strip_checkpoint_prefixes(state_dict, drop_mlm_head=True):
    """Map pre-training parameter names onto a bare `BertModel`.

    Pre-training runs under DistributedDataParallel and wraps the encoder in an
    MLM model, so its keys look like `module.bert.encoder...`.
    """
    encoder_state = {}
    for key, value in state_dict.items():
        name = key
        for prefix in ('module.', 'bert.'):
            if name.startswith(prefix):
                name = name[len(prefix):]
        if drop_mlm_head and name.startswith('mask_lm.'):
            continue
        encoder_state[name] = value

    return encoder_state


def infer_model_hyperparams(encoder_state, num_attention_heads):
    """Recover the encoder geometry from a state dict.

    Everything except the number of attention heads is written into the tensor
    shapes, so a checkpoint is almost self-describing.
    """
    vocab_size, hidden_size = encoder_state['embeddings.word_embeddings.weight'].shape
    layer_ids = [int(k.split('.')[2]) for k in encoder_state if k.startswith('encoder.layer.')]

    return {
        'vocab_size': int(vocab_size),
        'hidden_size': int(hidden_size),
        'num_hidden_layers': max(layer_ids) + 1,
        'num_attention_heads': num_attention_heads,
        'intermediate_size': int(encoder_state['encoder.layer.0.intermediate.dense.weight'].shape[0]),
        'max_position_embeddings': int(encoder_state['embeddings.position_embeddings.weight'].shape[0]),
        'type_vocab_size': int(encoder_state['embeddings.token_type_embeddings.weight'].shape[0]),
        'model_bias': 'encoder.layer.0.attention.selfattn.query.bias' in encoder_state,
    }


# --------------------------------------------------------------------------- #
# Result directories / logging
# --------------------------------------------------------------------------- #
def _get_test_name_and_result_dir(config, task: str, key: str):
    """`key` is one of the *_dir entries of the task section."""
    return config[task]['test_name'], config[task][key]


def make_task_result_directory(config):
    task = config['COMMON']['task']
    result_dir = config[task]['result_dir']

    sub_directory_list = ['attention_weights', 'logs', 'losses',
                          'model_checkpoints', 'prediction_status']
    os.makedirs(result_dir, exist_ok=True)
    for sub_directory in sub_directory_list:
        os.makedirs(os.path.join(result_dir, sub_directory), exist_ok=True)


def get_log_path(config, rank):
    task = config['COMMON']['task']
    test_name, log_dir = _get_test_name_and_result_dir(config, task, 'log_dir')

    train_log_path = os.path.join(log_dir, f"{task}_{test_name}_training_gpu{rank}.log")
    valid_log_path = os.path.join(log_dir, f"{task}_{test_name}_validation_gpu{rank}.log")

    return train_log_path, valid_log_path


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


# --------------------------------------------------------------------------- #
# Input data
# --------------------------------------------------------------------------- #
def load_input_df_and_samples_list(config, task, mode: str):
    if mode == 'train':
        samples_path = config[task]['Dataset']['Training_dataset_path']
    else:
        samples_path = config[task]['Dataset']['Validation_dataset_path']

    df = pd.read_csv(samples_path, sep='\t')
    samples_list = [sample for sample in df.columns.to_list() if sample != 'Unnamed: 0']

    return df, samples_list


def find_bioproject_from_sample(bioprj_samples_dict, sample_name):
    candidate_bioproject_list = [bioprj for bioprj, samples_list in bioprj_samples_dict.items()
                                 if sample_name in samples_list]
    if len(candidate_bioproject_list) == 1:
        return "".join(candidate_bioproject_list)
    elif len(candidate_bioproject_list) > 1:
        return ",".join(candidate_bioproject_list)
    return 'Unknown'


def make_sample_name_id_dict(true_samples_list, mode: str):
    i = 1000 if mode == 'train' else 50000

    sample_name_id_dict = {}
    for sample in true_samples_list:
        sample_name_id_dict[sample] = i
        i += 1

    return sample_name_id_dict


def gather_bioproject_samples(config, task, sample_list):
    info_path = os.path.join(config[task]["Dataset"]["dataset_dir"],
                             "DL_input_" + config[task]["Dataset"]["version"] + "_info.txt")
    sampleinfo = pd.read_csv(info_path, sep="\t", comment="#")

    selected_sampleinfo = sampleinfo.loc[sampleinfo["Run_Id"].isin(sample_list), :]
    if len(selected_sampleinfo) != len(sample_list):
        raise ValueError(
            f"{len(sample_list)} samples were requested but {len(selected_sampleinfo)} "
            f"were found in {info_path}")

    return {bioprj: selected_sampleinfo.loc[selected_sampleinfo["Study_Accession"] == bioprj, :]["Run_Id"].tolist()
            for bioprj in selected_sampleinfo["Study_Accession"].unique()}


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
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


def _write_vocab_dict(config, vocab_dict):
    """Always record the vocabulary that was actually used for this run."""
    task = config['COMMON']['task']
    test_name, result_dir = _get_test_name_and_result_dir(config, task, 'result_dir')

    vocab_path = os.path.join(result_dir, f"{task}_{test_name}_IMB_pathway_vocab.tsv")
    with open(vocab_path, 'w') as handle:
        for token, token_id in sorted(vocab_dict.items(), key=lambda kv: kv[1]):
            handle.write(f"{token_id}\t{token}\n")

    return vocab_path


def make_vocabulary(config, df):
    """Return the {pathway: token_id} mapping used to encode the samples.

    Two modes, selected by the optional `vocab_path` entry of the task's
    [[Dataset]] section:

    * `vocab_path` set  -- the vocabulary is read from that file and the ids are
      used exactly as stored.  This is REQUIRED whenever released IMBERT weights
      are involved: a pathway's row in the word-embedding matrix is fixed at
      pre-training time and cannot be re-derived from a different input table.
    * `vocab_path` unset -- a fresh vocabulary is built from the row order of
      `df`.  Only valid when pre-training a new model from scratch.

    The vocabulary actually used is always written to the result directory so
    that the checkpoints produced by this run remain usable later on.
    """
    task = config['COMMON']['task']
    vocab_path = config[task]['Dataset'].get('vocab_path')

    if vocab_path:
        vocab_dict = read_vocab_file(vocab_path)
        print(f"[IMBERT] vocabulary loaded from {vocab_path} ({len(vocab_dict)} tokens)")

        input_pathways = set(df['Unnamed: 0'].to_list())
        unknown = sorted(input_pathways - set(vocab_dict))
        if unknown:
            print(f"[IMBERT] WARNING: {len(unknown)} pathway(s) of the input table are not "
                  f"in the vocabulary and will be encoded as [UNK], e.g. {unknown[:5]}")
        missing = sorted(set(vocab_dict) - input_pathways - set(SPECIAL_TOKENS))
        if missing:
            print(f"[IMBERT] note: {len(missing)} vocabulary pathway(s) do not occur in the "
                  f"input table; their embeddings simply stay unused.")
    else:
        print("[IMBERT] no vocab_path configured -> building a new vocabulary from the "
              "input table (only valid when pre-training from scratch)")
        vocab_dict = {pwy: 5 + n for n, pwy in enumerate(df['Unnamed: 0'].to_list())}
        vocab_dict.update(SPECIAL_TOKENS)
        _validate_vocab(vocab_dict, 'input table')

    _write_vocab_dict(config, vocab_dict)

    return vocab_dict


# --------------------------------------------------------------------------- #
# Datasets / model / optimizer
# --------------------------------------------------------------------------- #
def load_datasets(config, inhouse_df, vocab_dict, sample_name_id_dict, mode: str):
    """
    mode == 'train'      -> return (sampler, dataloader)
    mode == 'validation' -> return dataloader
    """
    task = config['COMMON']['task']
    seq_len = config["Model"]["hyperparams"].as_int("max_position_embeddings")
    add_special_token = config["COMMON"].as_bool("add_special_token")
    batch_size = config[task]['hyperparams'].as_int('batch_size')

    world_size = get_world_size(config, task)
    use_cuda = len(resolve_device_ids(config, task)) > 0
    per_process_batch_size = max(1, int(batch_size / world_size))
    num_workers = 4 if use_cuda else 0

    print(f"\nCreating {task} {mode} Dataloader (seq_len={seq_len}, "
          f"batch_size/process={per_process_batch_size})")

    dataset = PAD.IMBDataset(inhouse_df,
                             vocab_dict,
                             sample_name_id_dict,
                             seq_len,
                             add_special_token)

    if mode == 'train':
        sampler = DistributedSampler(dataset, shuffle=True)
        dataloader = DataLoader(dataset,
                                batch_size=per_process_batch_size,
                                num_workers=num_workers,
                                shuffle=False,
                                pin_memory=use_cuda,
                                sampler=sampler)
        return sampler, dataloader

    dataloader = DataLoader(dataset,
                            batch_size=per_process_batch_size,
                            num_workers=num_workers,
                            shuffle=True,
                            pin_memory=use_cuda)
    return dataloader


def build_bert_config(config, vocab_size):
    """Build the `BertConfig` describing the IMBERT encoder."""
    hp = config['Model']['hyperparams']
    return BertConfig(vocab_size=vocab_size,
                      hidden_size=hp.as_int('hidden_size'),
                      num_hidden_layers=hp.as_int('num_hidden_layers'),
                      num_attention_heads=hp.as_int('num_attention_heads'),
                      intermediate_size=hp.as_int('intermediate_size'),
                      hidden_act=hp['hidden_act'],
                      hidden_dropout_prob=hp.as_float('hidden_dropout_prob'),
                      attention_probs_dropout_prob=hp.as_float('attention_probs_dropout_prob'),
                      max_position_embeddings=hp.as_int('max_position_embeddings'),
                      type_vocab_size=hp.as_int('type_vocab_size'),
                      initializer_range=hp.as_float('initializer_range'),
                      model_bias=hp.as_bool('model_bias'))


def load_model(config, vocab, logger):
    bert_config = build_bert_config(config, len(vocab))

    logger.info("====================================================================")
    logger.info("\t\t\tHyperparameters of IMBERT")
    logger.info("====================================================================")
    logger.info(f"Vocab size: {bert_config.vocab_size}")
    logger.info(f"Hidden dimension: {bert_config.hidden_size}")
    logger.info(f"The number of transformer blocks: {bert_config.num_hidden_layers}")
    logger.info(f"The number of attention heads: {bert_config.num_attention_heads}")
    logger.info(f"Intermediate dimension: {bert_config.intermediate_size}")
    logger.info(f"Linear layers use bias: {bert_config.model_bias}")

    return Bert.BertModel(bert_config, logger)


def save_checkpoint(config, epoch, model, optimizer):
    task = config['COMMON']['task']
    test_name, ckp_dir = _get_test_name_and_result_dir(config, task, 'model_checkpoints_dir')

    checkpoint = {'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}

    file_name = f"Epoch{epoch}_checkpoint_{task}_{test_name}"
    torch.save(checkpoint, os.path.join(ckp_dir, f'{file_name}.ckp'))
    print(f"Save checkpoint at epoch{epoch} complete")


def load_optimizer(config, model):
    hp = config['Training']['hyperparams']
    betas = (hp.as_float('adam_beta1'), hp.as_float('adam_beta2'))
    optimizer = AdamW(model.parameters(),
                      lr=hp.as_float('learning_rate'),
                      betas=betas,
                      weight_decay=hp.as_float('adam_weight_decay'),
                      eps=1e-08)

    return hp.as_int('warmup_steps'), optimizer
