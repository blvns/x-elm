#! /bin/bash


TASK=$1

NUM_CLUSTERS=8 # 4, 8, or 16
MIXTURE_FOLDER=/gscratch/zlab/tomlim/mbtm/mixtures/$TASK/$NUM_CLUSTERS
CLUSTERER_DIR=/gscratch/zlab/blvns/xl-btm/clusterers/mc4/$NUM_CLUSTERS

# INITIALIZE ENVIRONMENT
echo "Initializing environment"

cd /gscratch/zlab/tomlim/mbtm/xl-btm || exit

. "/gscratch/zlab/tomlim/miniconda3/etc/profile.d/conda.sh"
export PATH="/gscratch/zlab/tomlim/my_gs/miniconda3/bin:$PATH"

conda activate cbtm

if [ "$TASK" == "xnli" ]; then
  LANGUAGES=('ar' 'bg' 'de' 'el' 'en' 'es' 'fr' 'hi' 'ru' 'sw' 'th' 'tr' 'ur' 'vi' 'zh')
  COLUMNS="premise hypothesis"
  SPLIT="test"
  DATASET_DIR=''
elif [ "$TASK" == "xstorycloze" ]; then
  LANGUAGES=('ar' 'en' 'es' 'eu' 'hi' 'id' 'my' 'ru' 'sw' 'te' 'zh')
  COLUMNS="input_sentence_1 input_sentence_2 input_sentence_3 input_sentence_4"
  SPLIT="validation"
  DATASET_DIR='./xtstoycloze'
elif [ "$TASK" == "northeurlex" ]; then
  LANGUAGES=('de' 'ru' 'fr' 'es' 'tr' 'ko' 'ja' 'ar')
  COLUMNS="src tgt"
  SPLIT="test"
  DATASET_DIR='./northeurlex'
else
    echo "Task not recognized"
    exit 1
fi

for LANG in "${LANGUAGES[@]}"; do
  echo "Computing TF-IDF clusters for $LANG"
  if [ "$TASK" == "xnli" ]; then
    python xl-btm/downstream_eval/estimate_tfidf_cluster.py \
     --dataset-name $TASK \
     --mixture-folder $MIXTURE_FOLDER \
     --path-to-clusterer $CLUSTERER_DIR \
     --columns $COLUMNS --lang $LANG --split $SPLIT --hf-format
  else
    python xl-btm/downstream_eval/estimate_tfidf_cluster.py \
     --dataset-dir $DATASET_DIR \
     --dataset-name $TASK \
     --mixture-folder $MIXTURE_FOLDER \
     --path-to-clusterer $CLUSTERER_DIR \
     --columns $COLUMNS --lang $LANG --split $SPLIT --hf-format
  fi
done