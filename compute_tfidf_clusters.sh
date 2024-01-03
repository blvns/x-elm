#! /bin/bash

LANG=$1 #lang=en, az, etc.
TASK=$2

NUM_CLUSTERS=8 # 4, 8, or 16
MIXTURE_FOLDER=/gscratch/zlab/tomlim/mbtm/mixtures/$TASK/$NUM_CLUSTERS
CLUSTERER_DIR=/gscratch/zlab/blvns/xl-btm/clusterers/mc4/$NUM_CLUSTERS

# INITIALIZE ENVIRONMENT
echo "Initializing environment"

cd /gscratch/zlab/tomlim/mbtm/xl-btm || exit

. "/gscratch/zlab/tomlim/miniconda3/etc/profile.d/conda.sh"
export PATH="/gscratch/zlab/tomlim/my_gs/miniconda3/bin:$PATH"


if [ "$TASK" == "xnli" ]; then
    COLUMNS=("premise" "hypothesis")
    SPLIT="test"
    DATASET_DIR=''
elif [ "$TASK" == "xstorycloze" ]; then
    COLUMNS=("input_sentence_1" "input_sentence_2" "input_sentence_3" "input_sentence_4")
    SPLIT="validation"
    DATASET_DIR='./xtstoycloze'
elif [ "$TASK" == "northeurlex" ]; then
    COLUMN=(src tgt)
    SPLIT="test"
    DATASET_DIR='./northeurlex'
else
    echo "Task not recognized"
    exit 1
fi

python downstream_eval/estimate_tfidf_cluster.py \
 --dataset-name $TASK \
 --dataset-dir $DATASET_DIR \
 --mixture-folder $MIXTURE_FOLDER \
 --path-to-clusterer $CLUSTERER_DIR \
 --columns $COLUMNS --lang $LANG --split $SPLIT --hf-format