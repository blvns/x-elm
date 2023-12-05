#! /bin/bash



TASK=$1
MODEL_NAME=$2
METHOD=$3
CROSS=$4
CLUSTER=$SLURM_ARRAY_TASK_ID
nc=8 # number of clusters (if changing need to change LANGUAGE_MAP)

shopt -s nullglob

# INITIALIZE ENVIRONMENT
echo "Initializing environment"

cd /gscratch/zlab/tomlim/mbtm/xl-btm || exit

. "/gscratch/zlab/tomlim/miniconda3/etc/profile.d/conda.sh"
export PATH="/gscratch/zlab/tomlim/my_gs/miniconda3/bin:$PATH"

conda activate cbtm

if [ "$TASK" == "xnli" ]; then
  LANGUAGE_MAP=('en es' 'de el' '' 'hi ur' 'bg ru' 'fr ar' 'tr zh' 'vi zh')
elif [ "$TASK" == "xstorycloze" ]; then
  LANGUAGE_MAP=('en es' '' '' 'hi' 'ru' 'ar' 'zh' 'sw')
else
  echo "Task not recognized"
  exit 1
fi

LANGUAGES=${LANGUAGE_MAP[$CLUSTER]}
# CROSS not equal to empty string means cross-lingual evaluation
if [ "$CROSS" != '' ]; then
    LANGUAGES=${LANGUAGE_MAP[$CLUSTER]}' '${LANGUAGE_MAP[$CROSS]}
fi


if [ "$MODEL_NAME" == "10.4B" ]; then
    MODEL_PREFIX="${BASE_DIR}/experiments/${nc}_clusters/xlbtm.${METHOD}.mu20000.cluster"
elif [ "$MODEL_NAME" == "20.9B" ] && [ "$METHOD" == "lang" ]; then
    MODEL_PREFIX="${BASE_DIR}/experiments/${nc}_clusters_tl/xlbtm.${METHOD}.mu40000.cluster"
elif [ "$MODEL_NAME" == "20.9B" ]; then
    MODEL_PREFIX="${BASE_DIR}/experiments/${nc}_clusters/xlbtm.${METHOD}.mu40000.cluster"
else
    echo "Model name not recognized"
    exit 1
fi

if [ "$CROSS" == "" ] || [ "$CROSS" == "SRC" ] ; then
    MODEL="${MODEL_PREFIX}${CLUSTER}/hf"
elif [ "$METHOD" == "TRG" ]; then
    MODEL="${MODEL_PREFIX}0/hf"
elif [ "$CROSS" == "ENSEMBLE" ]; then
    MODEL="${MODEL_PREFIX}0/hf ${MODEL_PREFIX}${CLUSTER}/hf"
else
  echo "Crosslingual mode not recognized"
  exit 1
fi


if [ "$CROSS" == "" ]; then
  RESULTS="${BASE_DIR}/results/${MODEL_NAME}/C${nc}_${METHOD}"
  mkdir -p ${RESULTS}
  # RUN EVALUATION
  echo "Running evaluation for cluster ${CLUSTER} languages: ${LANGUAGES}"
  python3 xl-btm/downstream_eval/prompt.py --model_path ${MODEL} --output_dir ${RESULTS} --task ${TASK} --eval_lang ${LANGUAGES}
else
  RESULTS="${BASE_DIR}/results/${MODEL_NAME}/C${nc}_${METHOD}_from_en_${CROSS}"
  mkdir -p ${RESULTS}
  # RUN EVALUATION
  echo "Running cross-lingual evaluation for cluster ${CLUSTER} languages: ${LANGUAGES}, from English examples with ${CROSS} model"
  python3 xl-btm/downstream_eval/prompt.py --model_path ${MODEL} --output_dir ${RESULTS} --task ${TASK} --eval_lang ${LANGUAGES} --demo_lang "en" --k 8
fi
echo "Evaluation finished"