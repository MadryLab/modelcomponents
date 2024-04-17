import functools
import time
import random
import os
import numpy as np
import torch
import seaborn as sns
from types import SimpleNamespace
import datetime as dt
from scipy.stats import linregress

def recursive_getattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def get_num_params(model):
    return sum(p.numel() for p in model.parameters())

def yield_array_chunks(m, split_size, split_axis=0, get_slice=False):
    num_chunks = m.shape[split_axis]//split_size
    if num_chunks*split_size < m.shape[split_axis]:
        num_chunks += 1

    num_dims = len(m.shape)
    all_slice = [slice(None,None) for _ in range(num_dims)]

    for chunk_id in range(num_chunks):
        s = all_slice[:]
        s[split_axis] = slice(chunk_id*split_size, (chunk_id+1)*split_size)
        s = tuple(s)

        if get_slice:
            yield m[s], s[split_axis]
            continue

        yield m[s]

def random_split(arr, p):
    n = int(len(arr)*p)
    arr = arr[:]
    random.shuffle(arr)
    return arr[:n], arr[n:]

def fit_line(x, y):
    m, c, corr, *_ = linregress(x, y)
    x_ = np.linspace(min(x), max(x), num=100)
    y_ = m*x_+c

    return {
        'slope': m,
        'intercept': c,
        'correlation': corr,
        'line': (x_, y_)
    }

def get_timestamp(fmt='%y-%b%d-%H%M', add_randint=True):
    r = random.randint(0, 100_000)
    ts = dt.datetime.now().strftime(fmt)
    if add_randint: ts = f'{ts}-r{r}'
    return ts

class NestedDict(dict):
    def __getitem__(self, key):
        if key in self:
            return self.get(key)
        return self.setdefault(key, NestedDict())

    def __add__(self, other):
        return other

    def __sub__(self, other):
        return other

class NestedNamespace(SimpleNamespace):

    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)

        for key, value in dictionary.items():
            is_dict = isinstance(value, dict)
            all_keys_str = is_dict and all(map(lambda k: type(k) is str, value.keys()))

            if is_dict and all_keys_str:
                self.__setattr__(key, NestedNamespace(value))
            else:
                self.__setattr__(key, value)

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def convert_namespace_to_dict(args):
    return {k: (convert_namespace_to_dict(v) if isinstance(v, SimpleNamespace) else v) for k, v in vars(args).items()}

def convert_dict_to_namespace(args):
    return NestedNamespace(args)

def deepcopy_namespace(ns):
    ns_dict  =convert_namespace_to_dict(ns)
    return convert_dict_to_namespace(ns_dict)

def flatten_dictionary(dictionary, join_str='.'):
    fd = {}
    for k, v in dictionary.items():
        is_dict = isinstance(v, dict)
        all_keys_str = is_dict and all(map(lambda c: type(c) is str, v.keys()))

        if is_dict and all_keys_str:
            fd_rec = flatten_dictionary(v, join_str=join_str)
            for k_rec, v_rec in fd_rec.items():
                fd[f'{k}{join_str}{k_rec}'] = v_rec
        else:
            fd[k] = v
    return fd

def get_random_partition(seq, num_partitions, sort=True):
    """
    - seq: sequence to partition
    - k: size of each partition
    """
    sort_func = lambda x: tuple(sorted(x)) if sorted else tuple(x)
    seq = seq[:]
    np.random.shuffle(seq)
    return sort_func([sort_func(seq[i::num_partitions])
                      for i in range(num_partitions)])
