#! /bin/bash



shopt -s nullglob

TASK=$1
MODEL_NAME=$2
METHOD=$3
nc=8 # number of clusters (if changing need to change LANGUAGE_MAP)


sbatch --array=0-${nc} job_clustered_downstream_eval.sh $TASK $MODEL_NAME $METHOD