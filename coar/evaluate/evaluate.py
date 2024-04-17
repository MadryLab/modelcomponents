import copy
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from src.utils import eval_utils
from src.utils import ndm_utils
from src.utils import common_utils 

def get_masked_model(model, mask, comp_dims):
    model = model.cpu().eval()

    mask_inds = (~mask).nonzero()[0]
    comp_map = defaultdict(list)

    for idx in mask_inds:
        comp_name, comp_ind = ndm_utils.get_component_from_index(idx, comp_dims)
        comp_map[comp_name].append(comp_ind)

    for comp_name, comp_inds in comp_map.items():
        comp = common_utils.recursive_getattr(model, comp_name)
        with torch.no_grad(): 
            comp[comp_inds] = 0 

    return model

class EvaluateCoar(object):

    def __init__(self, attributions, get_model_func, mask_model_func, loader, num_components, device=torch.device(0)):
        """
        Args
            - attributions: attribution matrix [num_examples x num_components]
            - get_model_func: get_model()
            - mask_model_func: (model, mask) -> masked_model
            - loader: dataloader
            - num_components: number of components
        """
        self.attb = attributions
        self.get_model_func = get_model_func
        self.mask_model_func = mask_model_func
        self.loader = loader
        self.num_components = self.dim = num_components
        self.device = device

        self.og_model = get_model_func()
        self.attb = self.attb.to(device)

    def get_mask(self, prob):
        return ndm_utils.get_mask(self.dim, prob)

    def get_attb_margins(self, mask):
        mask_sum = self.attb[:,mask].sum(axis=1)
        return mask_sum.cpu().numpy()

    def get_margins(self, mask):
        model = copy.deepcopy(self.og_model)
        masked_model = self.mask_model_func(model, mask)
        masked_model = masked_model.to(self.device)
        margins = eval_utils.get_margins(masked_model, self.loader, self.device)
        return margins

    def run(self, mask_prob, num_runs, use_tqdm=False):
        margins_tracker = defaultdict(list)
        pbar = tqdm(range(num_runs)) if use_tqdm else range(num_runs)

        for _ in pbar:
            mask = self.get_mask(mask_prob)
            margins = self.get_margins(mask)
            margins_tracker['gt'].append(margins)

            margins = self.get_attb_margins(mask)
            margins_tracker['dm'].append(margins)

        gt = np.array(margins_tracker['gt']).astype(float).T
        dm = np.array(margins_tracker['dm']).astype(float).T

        # correlation: [num_examples]-length list of correlation over num_runs runs
        return {
            'gt': gt,
            'dm': dm,
            'correlation': [np.corrcoef(x, y)[1,0] for x, y in zip(gt, dm)]
        }

    def get_correlation_dataframe(self, subsample_fracs, num_runs, use_tqdm=False):
        frames = []

        for alpha in subsample_fracs:
            out = self.run(1-alpha, num_runs, use_tqdm=use_tqdm)
            df = pd.DataFrame()
            df['Model output correlation'] = out['correlation']
            df['Subsampling fraction'] = alpha
            df['Num. runs'] = num_runs
            frames.append(df)

        df = pd.concat(frames).reset_index(drop=True)
        return df
