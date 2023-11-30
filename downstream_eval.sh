

BASE_DIR=<your_directory>
#baseline
#MODEL="facebook/xglm-1.7B"
# 10.4B model
#MODEL="${BASE_DIR}/experiments/1_clusters/xlbtm.dense.mu20000.cluster0/hf"
# 20.9B model
MODEL="${BASE_DIR}/experiments/1_clusters/xlbtm.dense.mu40000.cluster0/hf"
TASK="xnli"
LANGUAGES=('ar' 'bg' 'de' 'el' 'en' 'es' 'fr' 'hi' 'ru' 'sw' 'th' 'tr' 'ur' 'vi' 'zh')
RESULTS="${BASE_DIR}/results/20.9B"


# RUN EVALUATION
echo "Running evaluation"
python3 xl-btm/downstream_eval/prompt.py --model_path ${MODEL} --output_dir ${RESULTS} --task ${TASK} --eval_lang ${LANGUAGES[@]}

echo "Evaluation finished"
