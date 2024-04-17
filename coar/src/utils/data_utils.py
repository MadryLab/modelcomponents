from collections import defaultdict
import itertools
import matplotlib.pyplot as plt
import numpy as np
import PIL
from pathlib import Path
import torch
from torch.utils.data import Dataset
from ffcv.loader import Loader, OrderOption
from ffcv.fields import IntField, RGBImageField
from ffcv.writer import DatasetWriter
from .label_utils import *

class ImagePatch(object):
    def __init__(self): pass
    def __call__(self, image): pass

class ConstantSquarePatch(ImagePatch):
    """
    Class to apply constant square patches to images
    - patch_rgb_tuple: tuple of size 3
    - patch_size: patch square length
    - patch_location: location corresponding to top left corner of patch in image
    """

    def __init__(self, patch_rgb_tuple, patch_size, patch_location):
        assert len(patch_rgb_tuple)==3, "invalid patch_rgb_tuple tuple"
        assert len(patch_location)==2, "invalid patch_location tuple"

        super().__init__()
        self.patch_rgb_tuple = patch_rgb_tuple
        self.patch_size = patch_size
        self.patch_location = patch_location

    def __call__(self, pil_image):
        # asserts
        assert type(pil_image) is PIL.Image.Image, "image type is not PIL.Image"
        assert 0 <= self.patch_location[0]+self.patch_size <= pil_image.size[0]
        assert 0 <= self.patch_location[1]+self.patch_size <= pil_image.size[1]

        # add patch to image
        pixels = list(range(self.patch_size))
        pixels = itertools.product(pixels, pixels)

        for (i, j) in pixels:
            px, py = self.patch_location[0]+i, self.patch_location[1]+j
            pil_image.putpixel((px, py), self.patch_rgb_tuple)

class UniformSquarePatch(ImagePatch):
    """
    Class to apply fixed uniform-noise square patch to images
    """

    def __init__(self, unif_min, unif_max, patch_size, patch_location):
        assert 0 <= unif_min <= unif_max, "invalid unif min val"
        assert unif_min <= unif_max <= 255, "invalid unif max val"
        assert len(patch_location)==2, "invalid patch_location tuple"

        super().__init__()
        self.unif_min = unif_min
        self.unif_max = unif_max
        self.patch_size = patch_size
        self.patch_location = patch_location

        self.U = defaultdict(dict)
        for i, j in itertools.product(range(self.patch_size), range(self.patch_size)):
            self.U[i][j] = tuple(np.random.randint(self.unif_min, self.unif_max+1, size=3))
        self.U = dict(self.U)

    def __call__(self, pil_image):
        # asserts
        assert type(pil_image) is PIL.Image.Image, "image type is not PIL.Image"
        assert 0 <= self.patch_location[0]+self.patch_size <= pil_image.size[0]
        assert 0 <= self.patch_location[1]+self.patch_size <= pil_image.size[1]

        # add patch to image
        pixels = list(range(self.patch_size))
        pixels = itertools.product(pixels, pixels)

        for (i, j) in pixels:
            px, py = self.patch_location[0]+i, self.patch_location[1]+j
            pil_image.putpixel((px, py), self.U[i][j])

class TransformDatasetWrapper(Dataset):
    """
    Dataset wrapper to apply additional custom
    transforms to target, label, or metadata
    - dataset: torch dataset
    - transform: generic tranform fa: data tuple -> modified data tuple
    """
    def __init__(self, dataset, arg_index, transform_fn):
        super().__init__()
        self.dset = dataset
        self.transform_fn = transform_fn
        self.arg_index = arg_index

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        data_tuple = [x for x in self.dset[idx]]
        data_tuple[self.arg_index] = self.transform_fn(data_tuple[self.arg_index])
        return data_tuple

def write_ffcv_beton(dset, save_path, num_workers=4, max_res=None, chunksize=100, **kw):
    save_path = Path(save_path)
    assert save_path.parent.exists()

    writer = DatasetWriter(str(save_path), {
        'image': RGBImageField(max_resolution=max_res),
        'label': IntField()
    }, num_workers=num_workers, **kw)

    writer.from_indexed_dataset(dset, chunksize=chunksize)
    return save_path

def get_ffcv_loader(beton_path, batch_size, num_workers, pipelines,
                    is_train, os_cache=True, indices=None, seed=None):

    order = OrderOption.RANDOM if is_train else OrderOption.SEQUENTIAL
    drop_last = is_train

    return Loader(
        fname=beton_path,
        batch_size=batch_size,
        num_workers=num_workers,
        order=order,
        os_cache=os_cache,
        indices=indices,
        pipelines=pipelines,
        drop_last=drop_last
    )

def split_dataset(dataset, fraction, dataset_keys=('train', 'test'), random_seed=0):
    train_size = int(fraction*len(dataset))
    indices = np.arange(len(dataset))

    if random_seed is not None:
        np.random.seed(random_seed)
    np.random.shuffle(indices)

    dsets_map = {}
    indices_map = {}
    inv_indices_map = {}

    for split in dataset_keys:
        indices_map[split] = indices[:train_size] if split==dataset_keys[0] else indices[train_size:]
        indices_map[split] = list(indices_map[split])
        dsets_map[split] = torch.utils.data.Subset(dataset, indices_map[split])

        for _idx in indices_map[split]:
            inv_indices_map[_idx] = split

    return indices_map, inv_indices_map, dsets_map

