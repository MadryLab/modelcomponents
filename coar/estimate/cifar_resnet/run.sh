GITHUB_REPO_DIR=/mnt/cfs/home/harshay/repos/modelcomponents/coar
BASE_DIR=/mnt/xfs/home/harshay/out/ndm/testing/cifar_resnet # data stores directory
SUBSAMPLE_PROB=0.95 # fraction of components not ablated
NUM_PARTITIONS=1 # one process
BATCH_SIZE=500
START_IDX=0
END_IDX=1000 # ablate + eval 1k times

cd $GITHUB_REPO_DIR

python -u -m estimate.cifar_resnet.make_dataset \
            --expt.base_dir $BASE_DIR \
            --expt.subsample_prob $SUBSAMPLE_PROB \
            --expt.batch_size $BATCH_SIZE \
            --expt.num_partitions $NUM_PARTITIONS \
            --expt.start_index $START_IDX \
            --expt.end_index $END_IDX \
            --expt.partition_index 0
