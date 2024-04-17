import numpy as np
import functools
from pathlib import Path
import torch


def recursive_getattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def get_component_from_index(comp_idx, comp_dims):
    """coponent_index -> component_key, component_index_in_component"""
    counter = 0
    for comp, comp_dim in comp_dims.items():
        if counter <= comp_idx < counter + comp_dim:
            return comp, comp_idx - counter
        counter += comp_dim
    assert False, "invalid component"


def get_mask(mask_dim, keep_prob):
    mask = np.random.rand(mask_dim) < keep_prob
    mask = mask.astype(bool)
    return mask


def get_component_segments(comp_dims, comp_name):
    """component_key -> [start_index, end_index]"""
    counter = 0
    for comp, comp_dim in comp_dims.items():
        if comp == comp_name:
            return counter, counter + comp_dim
        counter += comp_dim
    assert False, "invalid component"


def update_mmapped_file(dm_dir, index, out):
    dm_dir = Path(dm_dir)
    print("Updated: ")
    # out
    for k, v in out.items():
        mmap = np.lib.format.open_memmap(dm_dir / "{}.npy".format(k), mode="r+")
        mmap[index] = v
        mmap[index : (index + 1)].flush()
        if "margins" in k:
            s = "{} {:.2f}".format(k, (v > 0).mean())
        else:
            s = k
        print(s, end=",", flush=True)

    # completed
    cmp = np.lib.format.open_memmap(dm_dir / "_completed.npy", mode="r+")
    cmp[index] = True
    cmp[index : (index + 1)].flush()
    print()
