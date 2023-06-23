export NUM_CLUSTERS=$2;
# we want as many GPUs as we have clusters
export NUM_GPUS=${NUM_CLUSTERS};
export DATASET=mc4;
export EVAL_DIR=${SERIALIZATION_DIR}/${NUM_CLUSTERS}_clusters/eval_$1

mkdir -p ${EVAL_DIR};

# get model checkpoints
CONSOLIDATED_MODEL_PATHS=$3;
#CONSOLIDATED_MODEL_PATHS=`echo "${CONSOLIDATED_MODEL_PATHS}" | tr ',' ' '`
echo $CONSOLIDATED_MODEL_PATHS

# these model paths should be ordered by cluster ID!
#JOINED_MODEL_PATHS=$(join ${CONSOLIDATED_MODEL_PATHS[@]})
#echo $JOINED_MODEL_PATHS

#python -m metaseq_cli.eval_cbtm \
#    --data-dir ${DATA_DIR}/${DATASET} \
#    --data-subset valid/C4_small \
#    --path-to-clusterer ${KMEANS_DIR}/${DATASET}/${NUM_CLUSTERS}/ \
#    --model-paths $CONSOLIDATED_MODEL_PATHS \
#    --job-dir ${EVAL_DIR} \
#    --temperature 0.1 \
#    --max-valid-steps 200 \
#    --ensemble-type clustering \
#    --submitit
