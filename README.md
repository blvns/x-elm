# Cross-lingual Expert Language Models (X-ELM)

Code for the paper *Breaking the Curse of Multilinguality with Cross-lingual Expert Language Models*.

This repository is a fork of [c-BTM](https://github.com/kernelmachine/cbtm).

## Citation

If you use this code, please consider citing our work:

```
@article{blevins2024breaking,
 author = {Terra Blevins and Tomasz Limisiewicz and Suchin Gururangan and Margaret Li and Hila Gonen and Noah A. Smith and Luke Zettlemoyer.},
 title = {Breaking the Curse of Multilinguality with Cross-lingual Expert Language Models},
 year = {2024}
}
```

## Create a new conda env (recommended)

We supply an `environment.yml` file; this will create a conda environment with python 3.9 and a variety of dependencies. This will take a few minutes.

```bash
conda env create -f environment.yml
conda activate cbtm
```

### Install PyTorch

We tested this code with torch compiled with Cuda 11.3.

```bash
pip3 install torch==1.10.1+cu113  -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### Install Megatron

Make sure you have a GPU and CUDA visible for this step.

```bash
git clone --branch fairseq_v2 https://github.com/ngoyal2707/Megatron-LM.git
cd Megatron-LM
pip3 install six regex
pip3 install -e .
```

### Install fairscale

```bash
git clone https://github.com/facebookresearch/fairscale.git
cd fairscale
git checkout prefetch_fsdp_params_simple
pip3 install -e .
```

### Install balanced-kmeans

```bash
git clone https://github.com/kernelmachine/balanced-kmeans.git
cd balanced-kmeans
pip3 install -e .
```


### (Optional) Install Apex

NOTE: we don't use Apex for the X-ELM experiments, but you may need it for other functionalities in this repository.

Apex may not be compatible with all GPUs. In particular, if you're seeing that CUDA doesn't support your model during the forward pass, you might want to try uninstalling Apex and trying again.

```bash
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout e2083df5eb96643c61613b9df48dd4eea6b07690
```

Depending on your hardware, you may need to comment out lines 101-107 in setup.py before running the next pip install.

```bash
pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./
```

### Install X-ELM library

Build the X-ELM library. This won't really do anything if you've used the `environment.yml` file to build your conda environment.

```bash
cd /path/to/xelm
pip3 install -e .
```


# X-ELM Training and Evaluation

## Step 0: Set up data, models, and directories

We'll use the following environment variables in this tutorial, for simplicity. You can set these to whatever you want.

```bash
export XELM_DIR=$PWD; 
export DATA_DIR=${XELM_DIR}/data;
export SERIALIZATION_DIR=${XELM_DIR}/experiments;
export KMEANS_DIR=${XELM_DIR}/clusterers;
export CLUSTERS_DIR=${XELM_DIR}/clusters;
export VOCAB_DIR=$XELM_DIR/vocab
export PRETRAINED_MODELS_DIR=${XELM_DIR}/pretrained_models;
mkdir -p ${XELM_DIR} ${DATA_DIR} ${SERIALIZATION_DIR} ${KMEANS_DIR} ${CLUSTERS_DIR} ${VOCAB_DIR} ${PRETRAINED_MODELS_DIR};
```


### Configure xlbtm_constants.py

Next, the constants necessary to make this repo work are at `metaseq/xlbtm_constants.py`. Modify these to suit your local environment. 

Make sure the variables in `metaseq/xlbtm_constants.py` are consistent with the paths you set as environment variables above. 

### Download vocab files and seed models

Download the XGLM-1.7B checkpoint, which we use as our seed model:

```bash
mkdir -p ${PRETRAINED_MODELS_DIR}/xglm/1.7b/
wget https://nlp.cs.washington.edu/xl-elm/models/xglm_sharded.tar.gz
tar xvzf xglm_sharded.tar.gz -C ${PRETRAINED_MODELS_DIR}/xglm/1.7b/
```

The original XGLM model has been converted into Metaseq from Fairseq. We also provide the [unsharded Metaseq XGLM 1.7B model](https://nlp.cs.washington.edu/xl-elm/models/xglm.pt), and we document the full conversion process [here](conversion.md#Models), if you want to convert the model from Fairseq yourself or use a different XGLM size.


### Download data

We provide some sample mC4 data to get you started. Our model only expects (sharded) line-separated jsonl files, split into train and validation data. If you'd like to train on your own data, just follow the overall data layout in the example. 


```bash
mkdir -p ${DATA_DIR}/mc4_example/
wget https://nlp.cs.washington.edu/xl-elm/data/mc4_example.tar.gz
tar xvzf mc4_example.tar.gz -C ${DATA_DIR}/mc4_example/
```

This example dataset is a (subsampled) shard of mC4 for the 16 languages we train our X-ELMs on, and a small validation dataset. 

You can download the full mC4 dataset from Huggingface datasets at the following link: https://huggingface.co/datasets/mc4. Keep in mind that the dataset is very large, and comes as `json.gz` files. Our code expects raw jsonl files in the structure from the example directory, so make sure you have enough space (in total, it's about 1.4 terabytes of data uncompressed). We document the data conversion process used for the X-ELM experiments with mC4 [here](conversion.md#Data).

Metaseq expects the data to be in this general format:

```
{"text": "this is a document", "id": 0}
{"text": "this is another document", "id": 1}
```

For X-ELM, the data also contains "lang" and "url" data in each JSON document.

## Step 1: Train Clusterer

### 1a. TF-IDF Kmeans Clustering

This command trains a balanced k-means clusterer on a single shard of the C4 training data. Here we use k=8, and give as an argument a folder which contains a file called `mc4.jsonl`, as described above. 

Make sure you have access to a GPU here, to speed up training!

```bash
NUM_CLUSTERS=8;
DATASET=c4_example;
python -m metaseq.scripts.train_clusterer \
--data-dir ${DATA_DIR}/${DATASET}/train/00000 \
--num-clusters ${NUM_CLUSTERS} \
--balanced \
--output-dir ${KMEANS_DIR}/${DATASET}/${NUM_CLUSTERS}/
```

This will create `tfidf.pkl` (a tf-idf embedder) and `kmeans.pkl` (a kmeans clusterer) pickle files at `${KMEANS_DIR}/${DATASET}/${NUM_CLUSTERS}/`.

### 1b. Typological Clustering

TODO

## Step 2: Cluster data

### 2a TF-IDF Clustering

This code uses your trained clusterer to cluster the dataset's documents. This is _substantially_ faster if you can parallelize it as slurm jobs. 

We will do this automatically for you via `submitit`, just provide your slurm account and partition (either as a flag to the program, or in `metaseq/cbtm_constants.py`)

If you don't have access to slurm, you can cluster your data locally with the flag `--run local`, but it might take some time!

```bash
DATASET=c4_example;
NUM_CLUSTERS=8;

# Cluster train data
python -m metaseq.scripts.cluster \
--job-dir ${CBTM_DIR}/cluster_logs \
--data-dir ${DATA_DIR}/${DATASET} \
--path-to-clusterer ${KMEANS_DIR}/${DATASET}/${NUM_CLUSTERS}/ \
--num-clusters ${NUM_CLUSTERS} \
--output-prefix ${CLUSTERS_DIR}/${DATASET}/${NUM_CLUSTERS}/ \
--split train \
--run slurm; 

# Cluster validation data
python -m metaseq.scripts.cluster \
--job-dir ${CBTM_DIR}/cluster_logs \
--data-dir ${DATA_DIR}/${DATASET} \
--path-to-clusterer ${KMEANS_DIR}/${DATASET}/${NUM_CLUSTERS}/ \
--num-clusters ${NUM_CLUSTERS} \
--output-prefix ${CLUSTERS_DIR}/${DATASET}/${NUM_CLUSTERS}/ \
--split valid/C4_small \
--run slurm;
```

Logs for these clustering jobs appear in `${CBTM_DIR}/cluster_logs`.

After these jobs complete, open files in ${CLUSTERS_DIR}, e.g., `${CLUSTERS_DIR}/${DATASET}/${NUM_CLUSTERS}/train/00000/C4.jsonl`. You should see lines like the following:

```
{"sp_id":"\/gscratch\/zlab\/sg01\/data\/c4_example\/train\/00000\/C4.jsonl|0","cluster":5}
{"sp_id":"\/gscratch\/zlab\/sg01\/data\/c4_example\/train\/00000\/C4.jsonl|1","cluster":3}
{"sp_id":"\/gscratch\/zlab\/sg01\/data\/c4_example\/train\/00000\/C4.jsonl|2","cluster":2}
```

The field `sp_id` indicates a line (i.e., document) within a file, and the field `cluster` indicates its predicted cluster.

### 2a Typlogical Clustering

TODO

## Step 3: Train Models

Now we'll use the clustered data to train experts. You'll need at least 2 GPUs simultaneously to train each model.

This tutorial uses our `train_cbtm` script, which interfaces with SLURM. 

We have also provided an example sbatch script, if desired, in `metaseq/scripts/example_sbatch.sh`. You may need to edit this example sbatch command to include any additional slurm arguments you might need to get it working on your system.


### Train experts


The following command will train 8 expert models with 2 GPUs each for 50 steps (increase to 10000 steps to replicate our paper).


```bash
NUM_CLUSTERS=8;
DATASET=c4_example;
python -m metaseq.scripts.train_xlbtm \
   --model-size 1.7b \
   --run slurm   \
   --path-to-clusters-dir $CLUSTERS_DIR/${DATASET}/$NUM_CLUSTERS/ \
   --num-clusters $NUM_CLUSTERS  \
   --num-nodes 1 \
   --num-gpus 2 \
   --data-name ${DATASET}  \
   --path-to-data $DATA_DIR/${DATASET} \
   --learning-rate 1e-4 \
   --max-steps 50 \
   --valid-subset valid \
   --train-subset train
```

To train on a specific cluster(s), you can add the flag `--train-cluster 1,3,5`

To debug locally, change the `run` flag to `--run local`.

This command will output checkpoints and logs to `${SERIALIZATION_DIR}/8_clusters/`.


### Dense training

The following command will train a dense model with 2 GPUs for 50 steps (increase to 10000 steps to replicate our paper).

```bash
DATASET=c4_example;
python -m metaseq.scripts.train_xlbtm \
   --num-clusters 1 \
   --model-size 1.7b \
   --run slurm \
   --data-name $DATASET  \
   --num-nodes 1 \
   --num-gpus 2 \
   --data-name ${DATASET}  \
   --path-to-data $DATA_DIR/$DATASET  \
   --learning-rate 1e-4 \
   --max-steps 50 \
   --valid-subset valid \
   --train-subset train
```

To debug locally, change the `run` flag to `--run local`.

This command will output checkpoints to `${SERIALIZATION_DIR}/1_clusters/`.



## Evaluation

To evaluate your models, first consolidate your shards into a single checkpoint file.

The following script depends on the [`gnu-parallel`](https://www.gnu.org/software/parallel/) package.


```bash
NUM_CLUSTERS=8;
bash metaseq/scripts/consolidate_fsdp_shards.sh ${SERIALIZATION_DIR}/${NUM_CLUSTERS}_clusters/ "*ngpu4"
```

This will create a `consolidated.pt` checkpoint in each model's folder. 

Now the checkpoints are ready for eval. To launch on slurm:

```bash
export NUM_CLUSTERS=8;
# we want as many GPUs as we have clusters
export NUM_GPUS=${NUM_CLUSTERS};
export DATASET=c4_example;
export EVAL_DIR=${SERIALIZATION_DIR}/${NUM_CLUSTERS}_clusters/eval

mkdir -p ${EVAL_DIR};

# get model checkpoints
CONSOLIDATED_MODEL_PATHS=;
# this function gets all model checkpoint directories and sorts them by the cluster ID
# modify the folder pattern to match the directories names of your models, if you need.
FOLDER_PATTERN="cbtm\.c4_example\.*ngpu4"
mapfile -t MODEL_FOLDERS < <(find ${SERIALIZATION_DIR}/${NUM_CLUSTERS}_clusters/ -type d -name $FOLDER_PATTERN  -name "*\.cluster*" -printf "%f|%p\n" | sort -t "|" -k1,1 -t "|" -k2,2 | cut -d "|" -f 2)
for folder in "${MODEL_FOLDERS[@]}"; do
    # check if there are any consolidated.pt files in the model folder 
    if test -f "${folder}/consolidated.pt"; then
        CONSOLIDATED_MODEL_PATHS+="${folder}/consolidated.pt ";
    fi;
done
# function to join checkpoints into comma separated string
function join { local IFS=","; echo "$*"; }

# these model paths should be ordered by cluster ID!
JOINED_MODEL_PATHS=$(join ${CONSOLIDATED_MODEL_PATHS[@]})

python -m metaseq_cli.eval_cbtm \
    --data-dir ${DATA_DIR}/${DATASET} \
    --data-subset valid/C4_small \
    --path-to-clusterer ${KMEANS_DIR}/${DATASET}/${NUM_CLUSTERS}/ \
    --model-paths $(join ${CONSOLIDATED_MODEL_PATHS[@]}) \
    --job-dir ${EVAL_DIR} \
    --temperature 0.1 \
    --max-valid-steps 200 \
    --ensemble-type clustering \
    --submitit
```

You can check out logs for your slurm job at `${EVAL_DIR}`.

To launch locally, remove the flag `--submitit` in the command above. Make sure you have `$NUM_CLUSTERS` GPUs visible though!

This will output perplexity results to `${EVAL_DIR}/result.json`.

Use the same command as above to evaluate your dense models, just change the environment variable `NUM_CLUSTERS=1`.

## Open-sourced pretrained models

The models trained for the experiments in this paper are hosted [here](https://nlp.cs.washington.edu/xl-elm/).

### Converting from metaseq to Huggingface

We provide `xl-btm/convert_xglm_original_ckpt_to_trfms.py`, which can be used to convert consolidated XGLM models to the Huggingface format.


