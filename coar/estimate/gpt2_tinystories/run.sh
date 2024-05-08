REPO_DIR=/mnt/xfs/home/harshay/repos_xfs/modelcomponents
STORE_DIR=/mnt/xfs/home/harshay/out/tmp/gpt2_tinystories_test
EST_DIR=$REPO_DIR/coar/estimate
set -e

echo "Initializing store"
cd $EST_DIR

python -m initialize_store --logging.logdir $STORE_DIR --logging.spec $EST_DIR/gpt2_tinystories/spec_test.json

echo "Make component dataset"
cd $REPO_DIR/coar

python -m estimate.gpt2_tinystories.make_dataset --expt.base_dir $STORE_DIR --expt.subsample_prob 0.975 --expt.start_index 0 --expt.end_index 100

echo "Fit component attributions"
cd $EST_DIR

python -m write_dataset --cfg.data_dir $STORE_DIR --cfg.out_path $STORE_DIR/component_datasets.beton --cfg.x_name masks --cfg.y_name test_margins --cfg.ignore_completed

python -m run_regression --config $EST_DIR/gpt2_tinystories/regression_config_test.yaml --data.data_path $STORE_DIR/component_datasets.beton --cfg.out_dir $STORE_DIR/coar_attributions.pt