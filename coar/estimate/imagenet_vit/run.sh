REPO_DIR=/mnt/xfs/home/harshay/repos_xfs/modelcomponents
STORE_DIR=/mnt/xfs/home/harshay/out/tmp/imagenet_vit_coar_test
EST_DIR=$REPO_DIR/coar/estimate
set -e

echo "Initializing store"
cd $EST_DIR

python -m initialize_store --logging.logdir $STORE_DIR --logging.spec $EST_DIR/imagenet_vit/spec_test.json

# download beton
BETON_PATH=$REPO_DIR/betons/imagenet_resnet.pt
if [ ! -f "$BETON_PATH" ]; then
    wget -O $BETON_PATH "https://www.dropbox.com/scl/fi/ovztah82lk6qav1bkhpn3/imagenet_val.beton?rlkey=eroq5kcusbwdfecxk0xscnu3n&dl=0"
else
    echo "Beton already downloaded"
fi

echo "Make component dataset"
cd $REPO_DIR/coar

python -m estimate.imagenet_vit.make_dataset --expt.base_dir $STORE_DIR --expt.subsample_prob 0.95 --expt.batch_size 200 --expt.start_index 0 --expt.end_index 100 --expt.beton_path $BETON_PATH

echo "Fit component attributions"
cd $EST_DIR

python -m write_dataset --cfg.data_dir $STORE_DIR --cfg.out_path $STORE_DIR/component_datasets.beton --cfg.x_name masks --cfg.y_name val_margins --cfg.ignore_completed

python -m run_regression --config $EST_DIR/imagenet_vit/regression_config_test.yaml --data.data_path $STORE_DIR/component_datasets.beton --cfg.out_dir $STORE_DIR/coar_attributions.pt