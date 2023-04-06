INPUT_DIR=$1
OUTPUT_DIR=$2

python xl-btm/convert_xglm_original_ckpt_to_trfms.py \
 --fairseq_path $INPUT_DIR/consolidated.pt --pytorch_dump_folder_path $OUTPUT_DIR