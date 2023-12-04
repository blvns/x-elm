#! /bin/bash



TASK=$1
MODEL_NAME=$2
METHOD=$3
CLUSTER=$SLURM_ARRAY_TASK_ID
nc = 8 # number of clusters (if changing need to change LANGUAGE_MAP)

shopt -s nullglob

# INITIALIZE ENVIRONMENT
echo "Initializing environment"

cd /gscratch/zlab/tomlim/mbtm/xl-btm || exit

. "/gscratch/zlab/tomlim/miniconda3/etc/profile.d/conda.sh"
export PATH="/gscratch/zlab/tomlim/my_gs/miniconda3/bin:$PATH"

conda activate cbtm

if [ "$TASK" == "xnli" ]; then
  LANGUAGE_MAP = ('en es' 'de el' '' 'hi ur' 'bg ru' 'fr ar' 'tr zh' 'vi zh')
elif [ "$TASK" == "xstorycloze" ]; then
  LANGUAGE_MAP = ('en es' '' '' 'hi' 'ru' 'ar' 'zh' 'sw')
else
  echo "Task not recognized"
  exit 1
fi

LANGUAGES = LANGUAGE_MAP[$CLUSTER]

# check if there are languages to evaluate
if [ "$LANGUAGES" == '' ]; then
    echo "No languages to evaluate in cluster ${CLUSTER}"
    exit 0
fi


if [ "$MODEL_NAME" == "10.4B" ]; then
    MODEL="${BASE_DIR}/experiments/${nc}_clusters/xlbtm.${METHOD}.mu20000.cluster${CLUSTER}/hf"
elif [ "$MODEL_NAME" == "20.9B" ] && [ "$METHOD" == "lang" ]; then
    MODEL="${BASE_DIR}/experiments/${nc}_clusters_tl/xlbtm.${METHOD}.mu40000.cluster${CLUSTER}/hf"
elif [ "$MODEL_NAME" == "20.9B" ]; then
    MODEL="${BASE_DIR}/experiments/${nc}_clusters/xlbtm.${METHOD}.mu40000.cluster${CLUSTER}/hf"
else
    echo "Model name not recognized"
    exit 1
fi

RESULTS="${BASE_DIR}/results/${MODEL_NAME}/C${nc}_${METHOD}"

# RUN EVALUATION
echo "Running evaluation"
python3 xl-btm/downstream_eval/prompt.py --model_path ${MODEL} --output_dir ${RESULTS} --task ${TASK} --eval_lang ${LANGUAGES}

echo "Evaluation finished"