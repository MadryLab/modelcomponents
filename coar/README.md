## Estimating component attributions with COAR

Our code for estimating COAR attributions is split into three stages: initialization, dataset construction, and regression. We describe each stage in more detail below by applying COAR to a ResNet trained on CIFAR-10 ([`coar/estimate/cifar_resnet`](https://github.com/MadryLab/modelcomponents/tree/main/coar/estimate/cifar_resnet)).

### Stage 1: Initialize data stores

This stage involves initializing a directory of data stores (memory-mapped numpy arrays) using a JSON specification file.
We use the following spec file (`coar/estimate/cifar_resnet/spec_test.json`) to initialize data stores for CIFAR ResNet attributions:
```json
{
    "num_models": 100,
    "schema": {
        "masks": {
            "dtype": "bool_",
            "shape": [
                2304
            ]
        },
        "test_margins": {
            "dtype": "float16",
            "shape": [
                10000
            ]
        }
    }
}
```
This file specifies the dataset size and two data stores:
- `num_models` indicates that each data store comprises 100 rows, each corresponding to an ablated model.
-  `masks` is a `num_models x 2304` boolean array, where `2304` denotes the number of components in the ResNet model. Each mask (one per row) corresponds to a random subset of components ablated from the ResNet model.
- `val_margins` is a `num_models x 10000` float16 array. Each row records the outputs of ablated models on all 10k examples from the CIFAR10 test set.


To initialize these stores, you can run something like:
```bash
REPO_DIR=/mnt/xfs/home/harshay/repos_xfs/modelcomponents
STORE_DIR=/mnt/xfs/home/harshay/out/tmp/coar_test
EST_DIR=$REPO_DIR/coar/estimate
set -e

echo "Initializing store"
cd $EST_DIR

python -m initialize_store --logging.logdir $STORE_DIR --logging.spec $EST_DIR/cifar_resnet/spec_test.json
```

You will also need to download the model and ffcv dataloader
```bash
MODEL_PATH=$REPO_DIR/models/cifar_resnet.pt
BETON_PATH=$REPO_DIR/betons/cifar_resnet.pt

wget -O $MODEL_PATH "https://www.dropbox.com/scl/fi/ar7fput9rzyxebep0cgqf/cifar.pt?rlkey=y4hmrj94o4vxe4so55z1ebefw&dl=0"
wget -O $BETON_PATH "https://www.dropbox.com/scl/fi/4zj04xkgnb5mpw4aosvrt/cifar10.beton?rlkey=wspv74qs0h7l5cbxmzntmsywe&dl=0"
```

### Step 2: Make component datasets

In this step, we build a "component dataset" for each example in the test set using the initialized data stores from above.
This dataset includes tuples of $\alpha$-sized random subsets of model components $C'$ and their corresponding model outputs after ablating components in subset $C'$.

We implement this functionality for the CIFAR ResNet setup using a Python script called `make_dataset.py` in ([`coar/estimate/cifar_resnet`](https://github.com/MadryLab/modelcomponents/tree/main/coar/estimate/cifar_resnet)):
```bash
cd $REPO_DIR/coar

python -m estimate.cifar_resnet.make_dataset --expt.base_dir $STORE_DIR --expt.subsample_prob 0.95 --expt.batch_size 500 --expt.start_index 0 --expt.end_index 100 --expt.model_path $MODEL_PATH --expt.beton_path $BETON_PATH
```

### Step 3: Fit component attributions
We use the datasets described above to estimate component attributions (one per example) using [fast_l1](https://github.com/MadryLab/fast_l1), a SAGA-based GPU solver for linear regression.
We use the [FFCV library](http://ffcv.io/) for faster dataloading, so we first convert the component datasets from step 2 into FFCV-compatible `.beton` format as follows:
```bash
cd $EST_DIR

python -m write_dataset --cfg.data_dir $STORE_DIR --cfg.out_path $STORE_DIR/component_datasets.beton --cfg.x_name masks --cfg.y_name test_margins --cfg.ignore_completed
```
Then, we use `fast_l1` to estimate component attributions for the CIFAR ResNet setup as follows:
```bash
python -m run_regression --config $EST_DIR/cifar_resnet/regression_config_test.yaml --data.data_path $STORE_DIR/component_datasets.beton --cfg.out_dir $STORE_DIR/coar_attributions.pt
```
That's it! The file `component_attributions.pt` contains COAR-estimated attributions of the ResNet model on all 50k CIFAR test examples.

To run all of this in one go, check out [run.sh](https://github.com/MadryLab/modelcomponents/blob/main/coar/estimate/cifar_resnet/run.sh). You can speed things up by running `make_dataset` (i.e., construct component datasets) in parallel over a cluster of GPU machines. We provide an example SLURM script for this [here](https://github.com/MadryLab/modelcomponents/blob/main/coar/estimate/imagenet_vit/run.sbatch); note that this will require some modifications depending on your setup. If you just want pre-computed component attributions, check out the [README](https://github.com/MadryLab/modelcomponents/tree/main?tab=readme-ov-file#pre-computed-component-attributions).

We provide *example* run scripts for [ImageNet ResNet50](https://github.com/MadryLab/modelcomponents/tree/main/coar/estimate/imagenet_resnet), [ImageNet ViT-B/16](https://github.com/MadryLab/modelcomponents/tree/main/coar/estimate/imagenet_vit), and [GPT-2 evaluated on TinyStories](https://github.com/MadryLab/modelcomponents/blob/main/coar/estimate/gpt2_tinystories/run.sh).

