#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH -c 3
#SBATCH -a 0-7
#SBATCH -p background
#SBATCH --time=1-00:00:00
#SBATCH --job-name imagenet-vit
#SBATCH --output=<LOG_DIR>/in_vit_%a.out
#SBATCH --error=<LOG_DIR>/in_vit_%a.err
#SBATCH --requeue

source ~/.setup.sh

GITHUB_REPO_DIR=/mnt/cfs/home/harshay/repos/modelcomponents/coar
BASE_DIR=/mnt/xfs/home/harshay/out/ndm/testing/invit1 # data stores directory
SUBSAMPLE_PROB=0.95 # fraction of components not ablated
NUM_PARTITIONS=1 # one process
NUM_RUNS=1000 # each slurm task evals 1k ablated models

START_IDX=$((SLURM_ARRAY_TASK_ID*OFFSET))
END_IDX=$((SLURM_ARRAY_TASK_ID*OFFSET+OFFSET))

cd $GITHUB_REPO_DIR

parallel -j $NUM_PARTITIONS python -u -m estimate.imagenet_vit.make_dataset \
         --expt.base_dir $BASE_DIR \
         --expt.subsample_prob $SUBSAMPLE_PROB \
         --expt.batch_size $BATCH_SIZE \
         --expt.num_partitions $NUM_PARTITIONS \
         --expt.start_index $START_IDX \
         --expt.end_index $END_IDX \
         --expt.partition_index {} ::: $(seq 0 $((NUM_PARTITIONS-1)))