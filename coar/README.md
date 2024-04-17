## Estimating component attributions with COAR

Our code for estimating COAR attributions is split into three stages: initialization, dataset construction, and regression. We describe each stage in more detail below by applying COAR to a ViT-B/16 model trained on ImageNet ([`coar/estimate/imagenet_vit`](https://github.com/harshays/modelcomponents/tree/main/coar/estimate/imagenet_vit)).

### Stage 1: Initializing data stores

This stage involves initializing a directory of data stores (memory-mapped numpy arrays) using a JSON specification file.
We use the following spec file (`coar/estimate/imagenet_vit/spec.json`) to initialize data stores for ImageNet ViT attributions:
```json
{
    "num_models": 200000,
    "schema": {
        "masks": {
            "dtype": "bool_",
            "shape": [
                82944
            ]
        },
        "val_margins": {
            "dtype": "float16",
            "shape": [
                5000
            ]
        }
    }
}
```
This file specifies the dataset size and two data stores:
- `num_models` indicates that each data store comprises 200,000 rows, each corresponding to an ablated model.
-  `masks` is a `num_models x 82944` boolean array, where `82944` denotes the number of components in the ViT model. Each mask (one per row) corresponds to a random subset of components ablated from the ViT model.
- `val_margins` is a `num_models x 5000` float16 array. Each row records the outputs of ablated models on 5000 examples from the ImageNet validation set.

To initialize these stores, run:
```bash
REPO_DIR=...
STORE_DIR=...

cd $REPO_DIR/coar/estimate

python -m initialize_store \
    --logging.logdir $STORE_DIR \
    --logging.spec $REPO_DIR/coar/estimate/imagenet_vit/spec.json
```

### Step 2: Building component datasets

In this step, we incrementally build a "component dataset" for each example using the initialized data stores from above.
This dataset includes tuples of $\alpha$-sized random subsets of model components $C'$ and their corresponding model outputs after ablating components in subset $C'$.

We implement this functionality for the ImageNet ViT setup using a Python script called `make_dataset.py` in ([`coar/estimate/imagenet_vit`](https://github.com/harshays/modelcomponents/tree/main/coar/estimate/imagenet_vit)).
To run this script on a single GPU machine, run
```bash
cd $REPO_DIR/coar/estimate/imagenet_vit
bash run.sh
```
To construct component datasets for multiple examples in parallel over a cluster of GPU machines, we a provide a SLURM script:
```
cd $REPO_DIR/coar/estimate/imagenet_vit
sbatch run.sbatch
```
Note that both scripts—`run.sh` and `run.sbatch`—are just examples and will require some modifications depending on your compute setup.

### Step 3: Running linear regressions
We use the datasets described above to estimate component attributions (one per example) using [fast_l1](https://github.com/MadryLab/fast_l1), a SAGA-based GPU solver for linear regression.
We use the [FFCV library](http://ffcv.io/) for faster dataloading, so we first convert the component datasets from step 2 into FFCV-compatible `.beton` format as follows:
```bash
python -m write_dataset \
    --cfg.data_dir $STORE_DIR \
    --cfg.out_path $STORE_DIR/component_datasets.beton \
    --cfg.x_name masks \
    --cfg.y_name val_margins \
    --cfg.ignore_completed
```
Then, we use `fast_l1` to estimate component attributions for the ImageNet ViT setup as follows:
```bash
python -m run_regression \
        --config $REPO_DIR/regression_config.yaml \
        --data.data_path $STORE_DIR/component_datasets.beton \
        --cfg.out_dir $STORE_DIR/coar_attributions.pt
```
That's it! The file `component_attributions.pt` contains COAR-estimated attributions of the ViT model on 5000 ImageNet examples.

We also provide code to compute COAR attributions for CIFAR-10 ResNet in `coar/estimate/cifar_resnet` and ImageNet ResNet in `coar/estimate/imagenet_resnet` using the same commands as described above.


