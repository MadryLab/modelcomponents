import copy
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from fastargs import Section, Param, get_current_config
from fastargs.validation import And, OneOf
from tqdm import tqdm

import datasets
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader

from src.utils import eval_utils
from src.utils import ndm_utils

DEVICE = torch.device(0)
NUM_WORKERS = 2
DEBUG_MODE = True
ENABLE_AMP = True
NUM_COMPONENTS = 64_512
DATASET_SIZE = 500
DATASET_PATH = "roneneldan/TinyStories"  # download


def get_data(tokenizer, batch_size, num_workers=NUM_WORKERS):
    """
    Args
    - tokenizer
    - batch_size
    - num_workers
    Output
    - loader
    """
    # load dataset
    is_local = Path(DATASET_PATH).exists()

    if is_local:
        dataset = datasets.load_from_disk(DATASET_PATH)
    else:
        dataset = datasets.load_dataset(DATASET_PATH)

    dataset = dataset["validation"].train_test_split(
        test_size=DATASET_SIZE, shuffle=False
    )["test"]

    # tokenize
    def preprocess(examples, inds):
        out = tokenizer(
            examples["text"], truncation=True, max_length=tokenizer.model_max_length
        )
        out["index"] = inds
        return out

    tok_dataset = dataset.map(
        preprocess,
        with_indices=True,
        batched=True,
        batch_size=100,
        remove_columns=dataset.column_names,
    )

    # make dataloader
    collate_fn = DataCollatorForLanguageModeling(
        tokenizer, mlm=False, return_tensors="pt"
    )

    loader = DataLoader(
        tok_dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {"test": loader}


def evaluate(loader_map, model):
    """
    Args:
        - loader_map: dict (dataloaders)
        - model: nn.Module (model)
    Output:
        - stats: dict (dataloader -> model output, i.e., avg margin over all tokens in sequence)
    """

    def _get_sequence_length(attn_mask):
        inds = attn_mask.argmin(dim=1)
        inds[attn_mask.sum(dim=1) == attn_mask.shape[1]] = attn_mask.shape[1]
        return inds

    def _evaluate(model, dl, clone_logits=False):
        tracker = defaultdict(list)
        with torch.no_grad():
            for batch in dl:
                batch.pop("index")
                batch = batch.to(DEVICE)
                with torch.cuda.amp.autocast(ENABLE_AMP):
                    out = model(**batch)
                seq_lengths = _get_sequence_length(batch["attention_mask"])
                for idx, seq_len in enumerate(seq_lengths):
                    logits = out.logits[idx, : (seq_len - 1)]
                    labels = batch["labels"][idx, 1:seq_len]
                    margins = eval_utils.get_margins_given_logits(
                        logits, labels, clone_data=clone_logits
                    )
                    mean_margin = margins.mean().cpu().item()
                    tracker["margins"].append(mean_margin)

        return {k: np.array(v) for k, v in tracker.items()}

    model = model.to(DEVICE).eval()
    stats = {}
    for n, dl in loader_map.items():
        stats[n] = _evaluate(model, dl)
    return stats


def get_model_and_tokenizer():
    """
    Output
    - model
    - tokenizer
    """
    model_name_or_path = "gpt2"  # can change to local path
    is_local = Path(model_name_or_path).exists()
    model = GPT2LMHeadModel.from_pretrained(
        model_name_or_path, local_files_only=is_local
    )
    tokenizer = GPT2Tokenizer.from_pretrained(
        model_name_or_path, local_files_only=is_local
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # note: manually account for right padding in margin eval code (makes it a bit simpler)

    return model, tokenizer


def get_model_components(model):
    """
    Args
    - model: nn.Module (model)
    Output
    - mod_comps: dict (component group -> number of components)
    """
    suffixes = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]

    comp_dims = {}
    for k, v in model.named_modules():
        if any(k.endswith(suffix) for suffix in suffixes):
            comp_dims[f"{k}.weight"] = v.weight.shape[0]
    return comp_dims


def get_masked_model(model, mod_comps, mask):
    """
    Args
    - model: nn.Module (model)
    - mod_comps: dict (component group -> number of components)
    - mask: np.ndarray (mask for components to keep / ablate)
    Output
    - masked_model: nn.Module (masked model)
    """
    # deepcopy model before masking
    model = copy.deepcopy(model).cpu().eval()

    # extract components to ablate from mask
    rm_mask_inds = (~mask).nonzero()[0]
    comp_tups = [
        ndm_utils.get_component_from_index(mask_idx, mod_comps)
        for mask_idx in rm_mask_inds
    ]

    comp_map = defaultdict(list)
    for comp_name, comp_idx in comp_tups:
        comp_map[comp_name].append(comp_idx)

    # ablate components by zeroing out
    for comp_name, comp_inds in comp_map.items():
        comp = ndm_utils.recursive_getattr(model, comp_name)
        with torch.no_grad():
            comp[comp_inds] = 0

    return model


def evaluate_masked_model(model, mask, loaders, mod_comps):
    """
    Args
    - model: nn.Module (model)
    - mask: np.ndarray (mask for components to keep / ablate)
    - loaders: dict (dataloaders)
    - mod_comps: dict (component group -> number of components
    Output
    - eval_stats: dict (dataloader -> (masked) model output)
    """
    model = model.cpu().eval()
    masked_model = get_masked_model(model, mod_comps, mask)
    eval_stats = evaluate(loaders, masked_model)

    masked_model.cpu()
    torch.cuda.empty_cache()

    if DEBUG_MODE:
        print("masking done ({:.2f}%)".format(mask.mean() * 100), flush=True)

    return eval_stats


def get_args():
    sections = {}

    sections["expt"] = Section("expt", "experiment-specific arguments").params(
        base_dir=Param(str, "store directory", required=True),
        subsample_prob=Param(float, "subsample probability", default=1.0),
        start_index=Param(int, "start worker index", required=True),
        end_index=Param(int, "end worker index", required=True),
        mask_size=Param(int, "number of components", default=NUM_COMPONENTS),
        rerun=Param(And(int, OneOf([0, 1])), "re-run if completed flag", default=0),
        partition_index=Param(int, "partition index", default=0),
        num_partitions=Param(int, "number of partitions", default=1),
        batch_size=Param(int, "batch size", default=12),
        num_workers=Param(int, "number of workers", default=3),
    )

    return sections


def run():
    """
    Run experiment (save mask and masked model stats to data-store)
    """
    # get args
    sections = get_args()
    config = get_current_config()
    parser = argparse.ArgumentParser(description="GPT-2 TinyStories")
    config.augment_argparse(parser)
    config.validate(mode="stderr")
    config.summary()
    args = config.get()

    # update args
    base_dir = Path(args.expt.base_dir)
    assert base_dir.exists(), f"Use initialize_store.py to init base_dir"

    # get model and dataloader(s)
    model, tokenizer = get_model_and_tokenizer()
    mod_comps = get_model_components(model)
    loaders = get_data(tokenizer, args.expt.batch_size, args.expt.num_workers)

    # get data-store indices
    indices = list(range(args.expt.start_index, args.expt.end_index))

    if args.expt.num_partitions > 1:
        splits = np.array_split(indices, args.expt.num_partitions)
        indices = splits[args.expt.partition_index]

        if DEBUG_MODE:
            print(
                f"Partition {args.expt.partition_index} / {args.expt.num_partitions} ({len(indices)} indices)",
                flush=True,
            )

    # ablate and evaluate
    completed = np.lib.format.open_memmap(base_dir / "_completed.npy", mode="r")

    for index in tqdm(indices):
        if index >= len(completed):
            if DEBUG_MODE:
                print(f"Skip index {index} (out of bounds)")
            continue

        if not args.expt.rerun and completed[index]:
            if DEBUG_MODE:
                print(f"Skip index {index} (already completed)")
            continue

        # mask stats
        if DEBUG_MODE:
            print("Running index {}".format(index), flush=True)
        mask = ndm_utils.get_mask(args.expt.mask_size, args.expt.subsample_prob)
        eval_stats = evaluate_masked_model(model, mask, loaders, mod_comps)

        # update data-store
        out = {"masks": mask}
        split_list = ["test"]
        stat_list = ["margins"]

        for split in split_list:
            for stat in stat_list:
                out[f"{split}_{stat}"] = eval_stats[split][stat]

        ndm_utils.update_mmapped_file(base_dir, index, out)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
