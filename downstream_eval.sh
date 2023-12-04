
cd /gscratch/zlab/tomlim/mbtm/xl-btm || exit

. "/gscratch/zlab/tomlim/miniconda3/etc/profile.d/conda.sh"
export PATH="/gscratch/zlab/tomlim/my_gs/miniconda3/bin:$PATH"

conda activate cbtm

# SET UP VARIABLES
echo "Setting up variables"

BASE_DIR="/gscratch/zlab/tomlim/mbtm"

#baseline
if [ "$MODEL_NAME" == "xglm-1.7B" ]; then
    MODEL="facebook/xglm-1.7B"
elif [ "$MODEL_NAME" == "10.4B" ]; then
    MODEL="${BASE_DIR}/experiments/1_clusters/xlbtm.dense.mu20000.cluster0/hf"
elif [ "$MODEL_NAME" == "20.9B" ]; then
    MODEL="${BASE_DIR}/experiments/1_clusters/xlbtm.dense.mu40000.cluster0/hf"
else
    echo "Model name not recognized"
    exit 1
fi

if [ "$TASK" == "xnli" ]; then
    LANGUAGES=('ar' 'bg' 'de' 'el' 'en' 'es' 'fr' 'hi' 'ru' 'sw' 'th' 'tr' 'ur' 'vi' 'zh')
elif [ "$TASK" == "xstorycloze" ]; then
    LANGUAGES=('ar' 'en' 'es' 'eu' 'hi' 'id' 'my' 'ru' 'sw' 'te' 'zh')
else
    echo "Task not recognized"
    exit 1
fi


RESULTS="${BASE_DIR}/results/${MODEL_NAME}/dense"


# RUN EVALUATION
echo "Running evaluation"
python3 xl-btm/downstream_eval/prompt.py --model_path ${MODEL} --output_dir ${RESULTS} --task ${TASK} --eval_lang ${LANGUAGES[@]}

echo "Evaluation finished"
