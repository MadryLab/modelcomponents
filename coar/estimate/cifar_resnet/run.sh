REPO_DIR=/mnt/xfs/home/harshay/repos_xfs/modelcomponents
STORE_DIR=/mnt/xfs/home/harshay/out/tmp/coar_test
EST_DIR=$REPO_DIR/coar/estimate
set -e

echo "Initializing store"
cd $EST_DIR

python -m initialize_store --logging.logdir $STORE_DIR --logging.spec $EST_DIR/cifar_resnet/spec_test.json

# download model and beton
MODEL_PATH=$REPO_DIR/models/cifar_resnet.pt
BETON_PATH=$REPO_DIR/betons/cifar_resnet.pt

wget -O $MODEL_PATH "https://www.dropbox.com/scl/fi/ar7fput9rzyxebep0cgqf/cifar.pt?rlkey=y4hmrj94o4vxe4so55z1ebefw&dl=0"
wget -O $BETON_PATH "https://www.dropbox.com/scl/fi/4zj04xkgnb5mpw4aosvrt/cifar10.beton?rlkey=wspv74qs0h7l5cbxmzntmsywe&dl=0"

echo "Make component dataset"
cd $REPO_DIR/coar

python -m estimate.cifar_resnet.make_dataset --expt.base_dir $STORE_DIR --expt.subsample_prob 0.95 --expt.batch_size 500 --expt.start_index 0 --expt.end_index 100 --expt.model_path $MODEL_PATH --expt.beton_path $BETON_PATH

echo "Fit component attributions"
cd $EST_DIR

python -m write_dataset --cfg.data_dir $STORE_DIR --cfg.out_path $STORE_DIR/component_datasets.beton --cfg.x_name masks --cfg.y_name test_margins --cfg.ignore_completed

python -m run_regression --config $EST_DIR/cifar_resnet/regression_config_test.yaml --data.data_path $STORE_DIR/component_datasets.beton --cfg.out_dir $STORE_DIR/coar_attributions.pt

