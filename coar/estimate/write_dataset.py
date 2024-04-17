# Source: https://github.com/MadryLab/datamodels
import numpy as np
import os
from typing import Optional, Sequence
from torch.utils.data import Dataset
from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField, IntField

from argparse import ArgumentParser
from fastargs import Section, Param
from fastargs.decorators import param
from fastargs import get_current_config

Section('cfg', 'arguments to give the writer').params(
    data_dir=Param(str, 'Where to find the mmap datasets', required=True),
    out_path=Param(str, 'Where to write the FFCV dataset', required=True),
    x_name=Param(str, 'What portion of the data to write', default='masks'),
    y_name=Param(str, 'What portion of the data to write', required=True),
    y_slice=Param(int, 'If given, take a target slice', default=-1),
    completed_name=Param(str, 'Where to find completed datamodels', default='_completed'),
    ignore_completed=Param(bool, 'Whether to ignore the completed array', is_flag=True)
)


class RegressionDataset(Dataset):
    @param('cfg.y_slice')
    @param('cfg.ignore_completed')
    def __init__(self, *, masks_path: str, y_path: str,
                 completed_path: str,
                 subset: Optional[Sequence[int]] = None,
                 y_slice: Optional[int] = None,
                 ignore_completed: bool = False):
        super().__init__()
        self.comp_inds = None
        if not ignore_completed:
            print ('only using completed inds + rounding to closest 2000')
            comp_fp = np.lib.format.open_memmap(completed_path, mode='r')
            self.comp_inds = np.nonzero(comp_fp)[0]
            print ('Total completed before', len(self.comp_inds))
            N = 2000*(len(self.comp_inds)//2000)
            self.comp_inds = self.comp_inds[:N]
            print ('Total completed after', len(self.comp_inds))
        self.masks_fp = np.lib.format.open_memmap(masks_path, mode='r')
        self.x_dtype = self.masks_fp.dtype
        self.y_vals_fp = np.lib.format.open_memmap(y_path, mode='r')
        if y_slice is not None and y_slice > -1:
            self.y_vals_fp = self.y_vals_fp[:, y_slice]
        self.y_dtype = np.dtype('float32')

        total_len = len(self.masks_fp if self.comp_inds is None else self.comp_inds)
        self.subset = subset or range(total_len)

    def __getitem__(self, idx):
        ind = self.subset[idx]
        if self.comp_inds is not None:
            ind = self.comp_inds[ind]
        x_val = self.masks_fp[ind]
        y_val = self.y_vals_fp[ind].astype('float32')
        return x_val, y_val, ind

    def shape(self):
        return self.masks_fp.shape[1], self.y_vals_fp.shape[1]

    def __len__(self):
        return len(self.subset)


@param('cfg.data_dir')
@param('cfg.out_path')
@param('cfg.x_name')
@param('cfg.y_name')
@param('cfg.completed_name')
def write_dataset(data_dir: str, out_path: str,
                  x_name: str, y_name: str,
                  completed_name: str):
    ds = RegressionDataset(
            completed_path=os.path.join(data_dir, f'{completed_name}.npy'),
            masks_path=os.path.join(data_dir, f'{x_name}.npy'),
            y_path=os.path.join(data_dir, f'{y_name}.npy'))

    x_dim, y_dim = ds.shape()
    writer = DatasetWriter(out_path, {
        'mask': NDArrayField(dtype=ds.x_dtype, shape=(x_dim,)),
        'targets': NDArrayField(dtype=ds.y_dtype, shape=(y_dim,)),
        'idx': IntField()
    })

    writer.from_indexed_dataset(ds)


if __name__ == '__main__':
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    write_dataset()
