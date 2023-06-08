from dataclasses import dataclass

# SLURM variables
DEFAULT_SLURM_ACCOUNT="zlab"
DEFAULT_SLURM_CONSTRAINT="[rtx6k|a40|a100]"
#DEFAULT_SLURM_PARTITION="gpu-rtx6k"
DEFAULT_SLURM_PARTITION="ckpt"

# path to data directory
DATA_DIR="/gscratch/zlab/blvns/xl-btm/data/"

# where models will be saved (we will add them under a folder called `opt_ft` in this directory)
SERIALIZATION_DIR="/gscratch/zlab/blvns/xl-btm/experiments/"

# where clusterers and clusters will be saved
KMEANS_DIR = f"/gscratch/zlab/blvns/xl-btm/clusterers"
CLUSTERS_DIR = f"/gscratch/zlab/blvns/xl-btm/clusters"

# path to vocabulary (gpt2-merges.txt and gpt2-encoder.json)
#DEBUGGING -- shouldn't be used but will break if used to init a tokenizer with xglm
#VOCAB_PATH="/gscratch/zlab/blvns/xlbtm/vocab/gpt2/"
#VOCAB_PATH="/gscratch/zlab/blvns/xl-btm/vocab/xglm/"
VOCAB_PATH = "."

# path to pretrained models
#PRETRAINED_MODELS_DIR="/gscratch/zlab/blvns/xlbtm/pretrained_models/opt/"
PRETRAINED_MODELS_DIR="/gscratch/zlab/blvns/xl-btm/pretrained_models/xglm/"

# path to 1.3B parameter OPT checkpoint
#PATH_TO_1_3B_MODEL="/gscratch/zlab/blvns/xlbtm/pretrained_models/opt/1.3b/checkpoint_last.pt"
# path to 6.7B parameter OPT checkpoint
#PATH_TO_6_7B_MODEL="/gscratch/zlab/blvns/opt/6.7b/checkpoint_last.pt"

# path to 1.7B parameter XGLM checkpoint
PATH_TO_1_7B_MODEL="/gscratch/zlab/blvns/xl-btm/pretrained_models/xglm/1.7b/xglm.pt"

# path to xlbtm library
PATH_TO_CBTM="/gscratch/zlab/blvns/xl-btm/"


#overriding model sizes from constants.py to include XGLM
@dataclass
class Size:
    n_layers: int
    emb_size: int
    n_heads: int
    d_head: int
    batch_size: int
    lr: float
    model_parallel: int

    @property
    def ffn_size(self):
        return 4 * self.emb_size

M = 1024 * 1024  # 1 million

#OPT model sizes
#MODEL_SIZES = {
#    "8m": Size(4, 128, 2, 64, int(0.5 * M), 2.0e-3, 2),  # tiny
#    "25m": Size(6, 256, 4, 64, int(0.5 * M), 1.0e-3, 2),  # 25m
#    "50m": Size(8, 512, 8, 64, int(0.5 * M), 9.0e-4, 2),  # 50m
#    "125m": Size(12, 768, 12, 64, int(0.5 * M), 6.0e-4, 2),  # small
#    "350m": Size(24, 1024, 16, 64, int(0.5 * M), 3.0e-4, 2),  # medium
#    "760m": Size(24, 1536, 16, 96, int(0.5 * M), 2.5e-4, 2),  # large
#    "1.3b": Size(24, 2048, 32, 64, int(1.0 * M), 2.0e-4, 2),  # xl
#    "2.7b": Size(32, 2560, 32, 80, int(1.0 * M), 1.6e-4, 4),  # 2.7b
#    "6.7b": Size(32, 4096, 32, 128, int(2.0 * M), 1.2e-4, 2),  # 6.7b
#    "13b": Size(40, 5120, 40, 128, int(4.0 * M), 1.0e-4, 2),  # 13b
#    "30b": Size(48, 7168, 56, 128, int(4.0 * M), 1.0e-4, 2),
#    "66b": Size(64, 9216, 72, 128, int(1.0 * M), 1.0e-4, 4),  # 66b on 512 GPUs in RSC
#    "175b": Size(96, 12288, 96, 128, int(0.25 * M), 3e-5, 8),  # GPTZ/GPT-3
#}

#adding XGLM
MODEL_SIZES = {
	'1.7b': Size(24, 2048, 16, 128, int(1.0 * M), 2.0e-4, 2)
}

# from appendix b of https://arxiv.org/pdf/2005.14165.pdf
# see table 2.1 in https://arxiv.org/pdf/2005.14165.pdf

for name, size in MODEL_SIZES.items():
    assert size.n_heads * size.d_head == size.emb_size, name
