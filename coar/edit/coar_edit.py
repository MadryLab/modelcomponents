import torch
from torch import nn
from collections import defaultdict
from src.utils import ndm_utils
from src.utils import common_utils


class CoarEdit(object):

    ABLATE_MODE = {'topk', 'bottomk'}

    def __init__(self):
        pass

    def score_components(self, target_attb, reference_attb):
        """
        Args
        - target_attb: attributions of target examples [num_examples x num_components]
        - reference_attb: attributions of reference examples [num_examples x num_components]
        Output
        - scores: scores for each component [num_components]
        """
        T, R = target_attb, reference_attb
        m_t, m_r = map(lambda A: A.mean(dim=0), [T,R])
        s_t, s_r = map(lambda A: A.std(dim=0, correction=0), [T,R])

        m_delta = m_t - m_r
        s_delta = ((((s_t)**2)/T.shape[0]) + (((s_r)**2)/R.shape[0]))**0.5

        scores = m_delta/s_delta
        return scores

    def get_subset(self, scores, num_ablate, ablate_mode):
        """
        Args
        - scores: scores for each component [num_components]
        - num_ablate: number of components to ablate
        - ablate_mode: 'topk' or 'bottomk'
        Output
        - subset: indices of components to ablate
        """
        assert ablate_mode in self.ABLATE_MODE, 'ablate_mode must be "topk" or "bottomk"'
        scores = torch.FloatTensor(scores)
        largest = ablate_mode == 'topk'
        subset = scores.topk(num_ablate, largest=largest)
        return subset

    @staticmethod
    def ablate_model_given_indices(model, components_map, ablate_indices):
        """
        Args
        - model: model to ablate (nn.Module)
        - components_map: dictionary mapping component attribute names to size of component
        - ablate_indices: indices of components to ablate
        """
        model = model.cpu().eval()

        comp_map = defaultdict(list)

        for idx in ablate_indices:
            comp_name, comp_ind = ndm_utils.get_component_from_index(idx, components_map)
            comp_map[comp_name].append(comp_ind)

        for comp_name, comp_inds in comp_map.items():
            comp = common_utils.recursive_getattr(model, comp_name)
            with torch.no_grad():
                comp[comp_inds] = 0

        return model
    
    @staticmethod
    def ablate_model_given_component(model, component_name, component_idx):
        """
        Args
        - model: model to ablate (nn.Module)
        - component_name: component group getattr name
        - component_idx: index of component in group
        """
        model = model.cpu().eval()

        comp = common_utils.recursive_getattr(model, component_name)
        with torch.no_grad():
            comp[component_idx] = 0

        return model
    
# helper functions

def get_resnet50_components(model):
    def is_component(comp_name):
        valid = 'conv' in comp_name
        return valid

    def get_comp_dim(comp):
        assert type(comp) is nn.Conv2d
        return comp.out_channels

    comp_dims = {}
    for comp_name, comp in model.named_modules():
        if not is_component(comp_name): continue
        key = comp_name+'.weight'
        comp_dims[key] = get_comp_dim(comp)

    return comp_dims
    
