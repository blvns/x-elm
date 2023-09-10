export NUM_CLUSTERS=$2;
#export VALID_LANG=$4
# we want as many GPUs as we have clusters
export NUM_GPUS=${NUM_CLUSTERS};
export DATASET=mc4;
#this was missing (might need to be num clusters? idk)
export SLURM_NTASKS=1;

# get model checkpoints
CONSOLIDATED_MODEL_PATHS=$3;
#CONSOLIDATED_MODEL_PATHS=`echo "${CONSOLIDATED_MODEL_PATHS}" | tr ',' ' '`
echo $CONSOLIDATED_MODEL_PATHS

# these model paths should be ordered by cluster ID!
#JOINED_MODEL_PATHS=$(join ${CONSOLIDATED_MODEL_PATHS[@]})
#echo $JOINED_MODEL_PATHS

#("ar" "bg" "de" "el" "en" "es" "fr" "hi" "ja" "ko" "ru" "sw" "tr" "ur" "vi" "zh")
declare -a langs=("fr") # "es"

for lang in "${langs[@]}"
do
   export EVAL_DIR=${SERIALIZATION_DIR}/${NUM_CLUSTERS}_clusters/eval_$1_"$lang";
   mkdir -p ${EVAL_DIR};
   python -m metaseq_cli.eval_cbtm \
    --data-dir ${DATA_DIR}/${DATASET} \
    --data-subset valid_"$lang" \
    --path-to-clusterer ${KMEANS_DIR}/${DATASET}/${NUM_CLUSTERS}/ \
    --model-paths $CONSOLIDATED_MODEL_PATHS \
    --job-dir ${EVAL_DIR} \
    --temperature 0.1 \
    --max-valid-steps 5000 \
    --ensemble-type clustering \
    #--submitit \
    #--topk 1 
   
done


