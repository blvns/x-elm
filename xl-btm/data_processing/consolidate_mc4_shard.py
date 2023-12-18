import os
import subprocess
import pathlib
import joblib
from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm
import json

import random

SKIP_SHARD_MAX = 314


def _concat(input_fps, output_fp):
	trg_f = open(output_fp, 'w')
	for input_fp in input_fps:
		src_f = open(input_fp, 'r')
		for line in src_f: trg_f.write(line)
	return

def _reindex(lines):
	lines = [l for l in lines if len(l.split()) > 0]
	for i in range(0, len(lines)):
		line = lines[i]
		line_json = json.loads(line) #convert to json dict
		line_json['id'] = i #update index
		line = json.dumps(line_json)
		lines[i] = line
	return lines

def _shuf(input_fp, output_fp):
	src_lines = open(input_fp, 'r').readlines()
	random.shuffle(src_lines)
	lines = _reindex(src_lines)
	open(output_fp, 'w').writelines([l+'\n' for l in lines])
	return

def main():

	SOURCE_DIR = "/gscratch/zlab/blvns/xl-btm/data/mc4_adapt/"
	SOURCE_SUB_DIRS = ['train', 'valid']
	
	#for every subdirectory in source directory
	for sub_dir in SOURCE_SUB_DIRS:

		source_path_prefix = os.path.join(SOURCE_DIR, sub_dir)

		#get all shard dirs 
		shard_dirs = os.listdir(source_path_prefix) 
		shard_dirs = sorted(shard_dirs)
		#shard_dirs = [s for s in shard_dirs if int(s) > 598 and int(s) < 605]
		print(shard_dirs)

		#for every shard...
		for shard_dir in tqdm(shard_dirs):

			#manually skipping shards we already finished consolidating
			if sub_dir == 'train' and int(shard_dir) <= SKIP_SHARD_MAX: 
				print('skipping train {}'.format(shard_dir))
				continue

			source_path_prefix2 = os.path.join(source_path_prefix, shard_dir)
			target_path_prefix = source_path_prefix2

			#get all files in this subdirectory
			#source_files = [f for f in os.listdir(source_path_prefix2) if f.endswith('.jsonl') and os.path.isfile(os.path.join(source_path_prefix2, f))]

			#concat new files
			input_strs = [os.path.join(target_path_prefix, f) for f in os.listdir(target_path_prefix) if f.endswith('.jsonl') and os.path.isfile(os.path.join(target_path_prefix, f))]
			tmp_str = os.path.join(target_path_prefix, 'tmp.jsonl')
			shuffled_str = os.path.join(target_path_prefix, 'mc4.jsonl')

			#print(input_strs)
			#print(tmp_str) 
			#print(shuffled_str)
			#input('...')

			#subprocess.call("cat {} > {}".format(input_str, tmp_str), shell=True)
			#to ensure encodings don't get changed
			_ = _concat(input_strs, tmp_str)
			

			#shuffle new shard file
			_ = _shuf(tmp_str, shuffled_str)
			subprocess.call("rm {}".format(' '.join(input_strs)), shell=True)
			subprocess.call("rm {}".format(tmp_str), shell=True)
				



if __name__ == "__main__":
	main()

#EOF