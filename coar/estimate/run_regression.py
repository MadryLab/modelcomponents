# Modification of https://github.com/MadryLab/datamodels/blob/main/datamodels/regression/compute_datamodels.py
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch as ch

from fastargs import Param, Section, get_current_config
from fastargs.decorators import param, section

from ffcv.fields.decoders import IntDecoder, NDArrayDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import Squeeze, ToDevice, ToTensor
from ffcv.pipeline.operation import Operation

from fast_l1 import regressor
from dataclasses import replace


class Slice(Operation):
    def __init__(self, start_ind, end_ind) -> None:
        super().__init__()
        self.start_ind = start_ind
        self.end_ind = end_ind

    def generate_code(self):
        start_ind = self.start_ind
        end_ind = self.end_ind

        def make_slice(inp, _):
            if end_ind == -1:
                return inp[:, start_ind:]
            return inp[:, start_ind:end_ind]

        return make_slice

    def declare_state_and_memory(self, previous_state):
        end_ind = previous_state.shape[0] if self.end_ind == -1 \
                  else self.end_ind
        new_shape = (int(end_ind) - self.start_ind,)
        return replace(previous_state, shape=new_shape), None


Section('data', 'source data info').params(
    data_path=Param(str, 'Path to beton file', required=True),
    num_train=Param(int, 'Number of models for training', required=True),
    num_val=Param(int, 'Number of models for validation', required=True),
    seed=Param(int, 'Random seed for picking validation set'),
    target_start_ind=Param(int, 'Start of target slice', default=0),
    target_end_ind=Param(int, 'End of target slice', default=-1)
)

Section('cfg', 'arguments to give the writer').params(
    k=Param(int, 'Number of lambdas on the regularization path',
            required=True),
    lr=Param(float, 'Learning rate to use', default=0.01),
    eps=Param(float, '(min lambda) / (max lambda)', default=1e-5),
    batch_size=Param(int, 'Batch size for regression', required=True),
    out_dir=Param(str, 'Where to write', required=True),
    num_workers=Param(int, 'Num of workers to use for dataloading', default=16),
    use_bias=Param(int, 'Whether to use the bias parameter', default=1),
    manual_max_lambda=Param(float, 'Manually set the max lambda', default=-1), # -1 = use max lambda from data
)

Section('early_stopping', 'arguments specific to early stopping').params(
    check_every=Param(int, 'How often to check for improvement', default=2),
    eps=Param(float, 'Improvement required at every check', default=1e-5)
)


@param('data.data_path')
@param('data.target_start_ind')
@param('data.target_end_ind')
@param('cfg.num_workers')
@param('cfg.batch_size')
def make_loader(subset, data_path=None, num_workers=None,
                target_start_ind=None, target_end_ind=None,
                drop_last=True, batch_size: int = 0) -> Loader:
    assert len(subset) % batch_size == 0, \
        f'Batch size ({batch_size}) should divide dataset size ({len(subset)})'
    return Loader(data_path,
                  batch_size=batch_size,
                  num_workers=num_workers,
                  order=OrderOption.RANDOM,
                  indices=subset,
                  drop_last=drop_last,
                  os_cache=True,
                  pipelines={
                      'mask': [NDArrayDecoder(),
                               ToTensor(),
                               ToDevice(ch.device('cuda:0'))],
                      'targets': [NDArrayDecoder(),
                                  ToTensor(),
                                  Slice(target_start_ind, target_end_ind),
                                  ToDevice(ch.device('cuda:0'))],
                      'idx': [IntDecoder(),
                              ToTensor(),
                              Squeeze(),
                              ToDevice(ch.device('cuda:0'))]
                  }, recompile=False)


@param('data.num_train')
@param('data.num_val')
def make_loaders(num_train: int = -1, num_val: int = -1):
    return make_loader(subset=np.arange(num_train)), \
           make_loader(subset=np.arange(num_train, num_train + num_val)), \
           make_loader(subset=np.arange(num_train + num_val))


@section('cfg')
@param('lr')
@param('k')
@param('eps')
@param('out_dir')
@param('use_bias')
@param('manual_max_lambda')
@section('early_stopping')
@param('check_every', alias='early_stop_freq')
@param('eps', alias='early_stop_eps')
@section('data')
@param('target_start_ind')
@param('target_end_ind')
def main(lr: float, k: int, eps: float,
         out_dir: str,
         use_bias: int,
         early_stop_freq: int,
         early_stop_eps: float,
         target_start_ind: int,
         target_end_ind: int,
         manual_max_lambda=-1):
    train_loader, val_loader, full_loader = make_loaders()

    # max lam
    max_lam_data = regressor.calc_max_lambda(train_loader)
    max_lam_zeros = (ch.ones_like(max_lam_data)*manual_max_lambda).cuda()
    use_manual_max_lambda = manual_max_lambda != -1
    max_lam = max_lam_zeros if use_manual_max_lambda else max_lam_data
    if use_manual_max_lambda: assert k==1, 'If max lambda is 0, k must be 1'

    n_features = train_loader.reader.handlers['mask'].shape[0]
    n_targets = train_loader.reader.handlers['targets'].shape[0]
    if target_end_ind == -1:
        n_targets -= target_start_ind
    else:
        n_targets = target_end_ind - target_start_ind

    n_targets = int(n_targets)
    print(n_features, n_targets)

    weight = ch.zeros(n_features, n_targets).cuda()
    bias = ch.zeros(n_targets).cuda()

    assert not os.path.exists(out_dir)
    log_path = Path(out_dir) / 'regularization_path/'
    final_log_path = Path(out_dir) / 'final_lambda/'
    os.makedirs(log_path)
    os.makedirs(final_log_path)
    best_lam = \
        regressor.train_saga(weight,
                             bias,
                             train_loader,
                             val_loader,
                             lr=lr,
                             start_lams=max_lam,
                             update_bias=(use_bias > 0),
                             lam_decay=np.exp(np.log(eps)/k),
                             num_lambdas=k,
                             early_stop_freq=early_stop_freq,
                             early_stop_eps=early_stop_eps,
                             logdir=str(log_path))

    ch.cuda.empty_cache()
    regressor.train_saga(weight,
                         bias,
                         full_loader,
                         None,
                         lr=lr,
                         start_lams=best_lam,
                         update_bias=(use_bias > 0),
                         lam_decay=1.,
                         num_lambdas=1,
                         early_stop_freq=early_stop_freq,
                         early_stop_eps=early_stop_eps,
                         logdir=str(final_log_path))
    ch.save({
        'weight': weight.cpu(),
        'bias':  bias.cpu(),
        'lam': best_lam.cpu()
    }, Path(out_dir) / 'datamodels.pt')


if __name__ == '__main__':
    config = get_current_config()
    parser = ArgumentParser(description='Regression')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    main()
