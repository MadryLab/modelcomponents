import copy
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torchvision
from torch import nn
from fastargs import Section, Param, get_current_config
from fastargs.validation import And, OneOf
from tqdm import tqdm

from src.data import ffcv_pipelines
from src.utils import eval_utils
from src.utils import ndm_utils
from src.utils import data_utils
from src.utils import common_utils

DEVICE = torch.device(0)
DEBUG_MODE = True

BETON_URL = "https://www.dropbox.com/scl/fi/ovztah82lk6qav1bkhpn3/imagenet_val.beton?rlkey=eroq5kcusbwdfecxk0xscnu3n&dl=0"


def get_data(beton_path, subsample_skip, batch_size, num_workers):
    """
    Args
    - subsample_skip: int (skip factor for dataset subsampling)
    - batch_size: int (batch size)
    - num_workers: int (number of workers)
    Output
    - loaders: dict (dataloaders)
    """

    pipeline = ffcv_pipelines.get_pipelines("imagenet", "val", "cuda")
    indices = np.arange(0, 50_000, subsample_skip)

    loaders = {
        "val": data_utils.get_ffcv_loader(
            beton_path, batch_size, num_workers, pipeline, False, indices=indices
        )
    }

    return loaders


def evaluate(loader_map, model):
    """
    Args
    - loader_map: dict (dataloaders)
    - model: nn.Module (model)
    Output
    - stats: dict (dataloader -> model output)
    """
    model = model.to(DEVICE).eval()
    stats = defaultdict(dict)

    for n, dl in loader_map.items():
        stats[n]["margins"] = eval_utils.get_margins(model, dl, DEVICE, enable_amp=True)

    return dict(stats)


def get_model():
    """
    Output
    - model: nn.Module (model)
    """
    weights = torchvision.models.vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1
    model = torchvision.models.vit_b_16(weights=weights).eval()
    return model


def get_model_components(model):
    """
    Args
    - model: nn.Module (model)
    Output
    - mod_comps: dict (component group -> number of components)
    """
    mod_comps = {}

    for comp_name, comp in model.named_modules():
        if comp_name.startswith("heads"):
            continue
        if isinstance(comp, nn.Linear):
            mod_comps[f"{comp_name}.weight"] = comp.weight.shape[0]
        elif isinstance(comp, nn.MultiheadAttention):
            mod_comps[f"{comp_name}.in_proj_weight"] = comp.in_proj_weight.shape[0]

    return mod_comps


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
        print("masking done ({:.2f}%)".format(mask.mean()), flush=True)

    return eval_stats


def get_args():
    sections = {}

    sections["expt"] = Section("expt", "experiment-specific arguments").params(
        base_dir=Param(str, "store directory", required=True),
        subsample_prob=Param(float, "subsample probability", default=1.0),
        start_index=Param(int, "start worker index", required=True),
        end_index=Param(int, "end worker index", required=True),
        mask_size=Param(int, "number of components", default=82944),
        rerun=Param(And(int, OneOf([0, 1])), "re-run if completed flag", default=0),
        partition_index=Param(int, "partition index", default=0),
        num_partitions=Param(int, "number of partitions", default=1),
        subsample_skip=Param(int, "subsample skip", default=10),
        batch_size=Param(int, "batch size", default=500),
        num_workers=Param(int, "number of workers", default=3),
        beton_path=Param(str, "path to beton beton", default="TODO"),
    )

    return sections


def run():
    """
    Run experiment (save mask and masked model stats to data-store)
    """
    # get args
    sections = get_args()
    config = get_current_config()
    parser = argparse.ArgumentParser(description="ImageNet ViT")
    config.augment_argparse(parser)
    config.validate(mode="stderr")
    config.summary()
    args = config.get()

    # update args
    base_dir = Path(args.expt.base_dir)
    assert base_dir.exists(), base_dir

    # check beton path
    beton_path = args.expt.beton_path
    if beton_path == "TODO" or not Path(beton_path).exists():
        assert False, f"Download beton from: {BETON_URL}"

    # get model and dataloader(s)
    model = get_model()
    mod_comps = get_model_components(model)
    loaders = get_data(
        beton_path,
        args.expt.subsample_skip,
        args.expt.batch_size,
        args.expt.num_workers,
    )

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
        split_list = ["val"]
        stat_list = ["margins"]

        for split in split_list:
            for stat in stat_list:
                out[f"{split}_{stat}"] = eval_stats[split][stat]

        ndm_utils.update_mmapped_file(base_dir, index, out)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
