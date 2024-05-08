# Source: https://github.com/MadryLab/datamodels
from fastargs.decorators import param
from fastargs import Param, Section
from pathlib import Path
from copy import deepcopy
import numpy as np
import json
from fastargs import get_current_config
from argparse import ArgumentParser
import types


def memmap_path(logdir, key):
    return Path(logdir) / f"{key}.npy"


def make_config(quiet=False, conf_path=None):

    def collect_known_args(self, parser, disable_help=False):
        args, _ = parser.parse_known_args()
        for fname in args.config_file:
            self.collect_config_file(fname)

        args = vars(args)
        self.collect(args)
        self.collect_env_variables()

    config = get_current_config()
    if conf_path is not None:
        config.collect_config_file(conf_path)

    f = types.MethodType(collect_known_args, config)
    config.collect_argparse_args = f

    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)

    config.validate(mode="stderr")
    if not quiet:
        config.summary()

    return config


def preprocess_spec(spec):

    COMPLETED = "_completed"
    NUM_MODELS = "num_models"
    SCHEMA = "schema"
    DTYPE = "dtype"
    SHAPE = "shape"

    def _add_completed(spec):
        spec = deepcopy(spec)
        spec[SCHEMA][COMPLETED] = {DTYPE: "bool_", SHAPE: []}
        return spec

    def verify_spec(spec):
        assert NUM_MODELS in spec, f"you need a {NUM_MODELS} attribute"
        assert SCHEMA in spec, f"you need a {SCHEMA} attribute"
        assert (
            not COMPLETED in spec[SCHEMA]
        ), f"no schema dtypes called {COMPLETED} allowed"

        schema = spec[SCHEMA]
        for _, v in schema.items():
            assert DTYPE in v, "you need a dtype"
            assert SHAPE in v, "you need a shape"

            this_dtype = v[DTYPE]
            this_shape = v[SHAPE]
            assert type(this_shape) is list, "your shape must be a list"
            assert hasattr(
                np, this_dtype
            ), f"your dtype {this_dtype} is not a numpy dtype"

    verify_spec(spec)
    spec = _add_completed(spec)
    return spec


Section("logging").params(
    logdir=Param(str, "file with main() to run"), spec=Param(str, "file with spec")
)


@param("logging.logdir")
@param("logging.spec")
def main(logdir, spec):
    assert logdir is not None
    assert spec is not None
    spec = json.loads(open(spec, "r").read())
    spec = preprocess_spec(spec)

    if Path(logdir).exists():
        print ('Log dir already exists, skipping')
        return 
    else:
        Path(logdir).mkdir(parents=False, exist_ok=True)

    num_models = spec["num_models"]
    for key, metadata in spec["schema"].items():
        dtype = getattr(np, metadata["dtype"])
        shape = (num_models,) + tuple(metadata["shape"])

        this_filename = memmap_path(logdir, key)
        mmap = np.lib.format.open_memmap(
            this_filename, mode="w+", dtype=dtype, shape=shape
        )
        mmap.flush()


if __name__ == "__main__":
    make_config()
    main()
    print("Done!")
