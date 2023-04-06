import os
import pathlib
from tqdm import tqdm
import json
from joblib import Parallel, delayed
import subprocess

def clean_data(datapath, out_path):
	src_f = open(datapath, 'r')
	trg_f = open(out_path, 'w')

	for line in src_f:
		line_json = json.loads(line)
		if line_json['lang'] != 'fi':
			trg_f.write(line+'\n')
		
	src_f.close()
	trg_f.close()

	subprocess.call("mv {} {}".format(out_path, datapath), shell=True)
	return


def main():

	SOURCE_DIR = "/gscratch/zlab/blvns/xl-elm/data/mc4_adapt_neighbors/"
	SOURCE_SUB_DIRS = ['valid'] #['train', 'valid']
	TARGET_DIR = "/gscratch/zlab/blvns/xl-btm/data/mc4_adapt_neighbors/"

	for sub_dir in SOURCE_SUB_DIRS:
		source_path_prefix = os.path.join(SOURCE_DIR, sub_dir)

		#get all shard dirs 
		pl_files = os.listdir(source_path_prefix) 
		
		for plf in tqdm(pl_files):
			_, split, lang, shard, _ = plf.split('.')
			source_path = os.path.join(source_path_prefix, plf)
			shard_dir = str(shard).zfill(5)
			target_path = os.path.join(TARGET_DIR, sub_dir, shard_dir, '{}.jsonl'.format(lang))
			#print(source_path_prefix, plf)
			#print(target_path)
			#input('...')
			subprocess.call("mv {} {}".format(source_path, target_path), shell=True)


	'''
	#for every subdirectory in source directory
	for sub_dir in SOURCE_SUB_DIRS:
		source_path_prefix = os.path.join(SOURCE_DIR, sub_dir)

		#get all shard dirs 
		shard_dirs = os.listdir(source_path_prefix) 
		shard_dirs = sorted(shard_dirs)
		#shard_dirs = [s for s in shard_dirs if int(s) > 598 and int(s) < 605]
		print(shard_dirs)

		#for every shard...
		timeout=99999
		_ = Parallel(n_jobs=4, timeout=timeout)(delayed(clean_data)(os.path.join(source_path_prefix, shard_dir, 'mc4.jsonl'), os.path.join(source_path_prefix, shard_dir, 'mc4_nofi.jsonl')) for shard_dir in tqdm(shard_dirs))


		#for shard_dir in tqdm(shard_dirs):
		#	shard_path = os.path.join(source_path_prefix, shard_dir, 'mc4.jsonl')
		#	out_path = os.path.join(source_path_prefix, shard_dir, 'mc4_clean.jsonl')
		#	_ = clean_data(shard_path, out_path)
	'''
			




		
if __name__ == "__main__":
	main()

#EOF