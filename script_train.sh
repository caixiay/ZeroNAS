#!/bin/bash
set -x
DATASET=$1
BATCHSIZE=$2
TIME=`date +'%Y-%m-%d_%H-%M-%S'`
PLOTDIR="./output/net_plot_${DATASET}_${TIME}"

mkdir $PLOTDIR

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}

export PYTHONUNBUFFERED=true
#checkpoint 30103 checkepoch 3 --r true --lr_decay_step 10 --epochs 12
LOG="${PLOTDIR}/train_log.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

python ./clswgan.py \
--dataset ${DATASET} \
--batch_size ${BATCHSIZE} \
--figure_dir ${PLOTDIR} \
  ${EXTRA_ARGS}

#2>&1 | tee $LOG



