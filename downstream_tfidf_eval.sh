#! /bin/bash


TASK=$1
TFIDF_MODE=$2
CROSS=${3:-""}




# INITIALIZE ENVIRONMENT
echo "Initializing environment"

cd /gscratch/zlab/tomlim/mbtm/xl-btm || exit

. "/gscratch/zlab/tomlim/miniconda3/etc/profile.d/conda.sh"
export PATH="/gscratch/zlab/tomlim/my_gs/miniconda3/bin:$PATH"

conda activate cbtm



# SET UP VARIABLES
echo "Setting up variables"

BASE_DIR="/gscratch/zlab/tomlim/mbtm"
MODEL_NAME="20.9B"
CLUSTERS=8
TFIDF_TASK_DIR="${BASE_DIR}/mixtures/${TASK}/${CLUSTERS}/"
#baseline


MODELS=""
for i in {0..7}
do
    MODELS="${MODELS} ${BASE_DIR}/experiments/${CLUSTERS}_clusters_tl/xlbtm.tfidf.mu40000.cluster${i}/hf"
done

if [ "$TASK" == "xnli" ]; then
    LANGUAGES=('ar' 'bg' 'de' 'el' 'en' 'es' 'fr' 'hi' 'ru' 'sw' 'th' 'tr' 'ur' 'vi' 'zh')
elif [ "$TASK" == "xstorycloze" ]; then
    LANGUAGES=('ar' 'en' 'es' 'eu' 'hi' 'id' 'my' 'ru' 'sw' 'te' 'zh')
elif [ "$TASK" == "northeurlex" ]; then
    LANGUAGES=('de' 'ru' 'fr' 'es' 'tr' 'ko' 'ja' 'ar')
else
    echo "Task not recognized"
    exit 1
fi

if [ "$CROSS" == "" ]; then
  RESULTS="${BASE_DIR}/results/${MODEL_NAME}/C${CLUSTERS}_tfidf_${TFIDF_MODE}"
  mkdir -p ${RESULTS}
  # RUN EVALUATION
  echo "Running evaluation"
  python3 xl-btm/downstream_eval/prompt.py --model_path ${MODELS} --output_dir ${RESULTS} \
    --task ${TASK} --eval_lang ${LANGUAGES[@]} \
    --tfidf_task_dir ${TFIDF_TASK_DIR} --tfidf_mode ${TFIDF_MODE}
else
  RESULTS="${BASE_DIR}/results/${MODEL_NAME}/C${CLUSTERS}_tfidf_${TFIDF_MODE}_from_en"
  mkdir -p ${RESULTS}
  # RUN EVALUATION
  echo "Running evaluation with en demonstrations"
  python3 xl-btm/downstream_eval/prompt.py --model_path ${MODELS} --output_dir ${RESULTS} \
    --task ${TASK} --eval_lang ${LANGUAGES[@]} --demo_lang "en" --k 8 \
    --tfidf_task_dir ${TFIDF_TASK_DIR} --tfidf_mode ${TFIDF_MODE}
fi

echo "Evaluation finished"
