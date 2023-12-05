#! /bin/bash



shopt -s nullglob

TASK=$1
MODEL_NAME=$2
METHOD=$3
CROSS=${4:-""} # TRG or SRC or ENSEMBLE or empty string. Now assume that demonstration in English.
nc=8 # number of clusters (if changing need to change LANGUAGE_MAP)
# parameter with default value



sbatch --array=0-$((nc-1)) job_clustered_downstream_eval.sh $TASK $MODEL_NAME $METHOD $CROSS