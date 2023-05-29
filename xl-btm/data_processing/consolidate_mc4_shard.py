import os
import subprocess
import pathlib
import joblib
from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm

SKIP_SHARD_MAX = 533


def _move_and_add_lang_data(source_path, source_file, target_path, target_file):
	#parse information
	try:
		lang, _ = source_file.split('.')
	except ValueError:
		return

	shard = pd.read_json(path_or_buf=os.path.join(source_path, source_file), lines=True)

	#add lang code as meta data for each element in shard
	lang_arr = [lang]*len(shard)
	shard['lang'] = lang_arr

	#write json files out to new target shard file
	shard.to_json(os.path.join(target_path, target_file), lines=True, orient='records', force_ascii=False)

	return



def main():

	SOURCE_DIR = "/gscratch/zlab/blvns/xl-btm/data/mc4/"
	SOURCE_SUB_DIRS = ['train', 'valid']
	TARGET_DIR = "/gscratch/zlab/blvns/xl-btm/data/mc4_shuffled/"

	#make target dir if doesn't exist (and subdirs)
	#pathlib.Path(TARGET_DIR).mkdir(parents=True, exist_ok=True)
	#for sub_dir in SOURCE_SUB_DIRS:
	#	target_sub_dir = os.path.join(TARGET_DIR, sub_dir)
	#	pathlib.Path(target_sub_dir).mkdir(parents=True, exist_ok=True)
	
	#for every subdirectory in source directory
	for sub_dir in SOURCE_SUB_DIRS:

		source_path_prefix = os.path.join(SOURCE_DIR, sub_dir)

		#get all shard dirs 
		shard_dirs = os.listdir(source_path_prefix) 
		shard_dirs = sorted(shard_dirs)
		print(shard_dirs)

		#for every shard...
		for shard_dir in tqdm(shard_dirs):

			#manually skipping shards we already finished consolidating
			if sub_dir == 'train' and int(shard_dir) <= SKIP_SHARD_MAX: 
				print('skipping train {}'.format(shard_dir))
				continue

			source_path_prefix2 = os.path.join(source_path_prefix, shard_dir)

			#make target directory if it doesn't exist
			target_path_prefix = os.path.join(TARGET_DIR, sub_dir, shard_dir)
			pathlib.Path(target_path_prefix).mkdir(parents=True, exist_ok=True)

			#get all files in this subdirectory
			source_files = [f for f in os.listdir(source_path_prefix2) if os.path.isfile(os.path.join(source_path_prefix2, f))]

			#for every data file in shard directory...
			try:
				timeout=99999
				_ = Parallel(n_jobs=4, timeout=timeout)(delayed(_move_and_add_lang_data)(source_path_prefix2, sf, target_path_prefix, sf) for sf in source_files)
			#don't do parallel on the "problem"/large shards that cause OOM errors	
			except joblib.externals.loky.process_executor.TerminatedWorkerError:
				print('Shard causes OOM... running sequentially')
				for sf in source_files: 
					_move_and_add_lang_data(source_path_prefix2, sf, target_path_prefix, sf)


			#concat new files
			input_str = ' '.join([os.path.join(target_path_prefix, f) for f in os.listdir(target_path_prefix) if os.path.isfile(os.path.join(target_path_prefix, f))])
			tmp_str = os.path.join(target_path_prefix, 'tmp.jsonl')
			shuffled_str = os.path.join(target_path_prefix, 'mc4.jsonl')


			#print(input_str)

			subprocess.call("cat {} > {}".format(input_str, tmp_str), shell=True)
			subprocess.call("rm {}".format(input_str), shell=True)

			#shuffle new shard file
			subprocess.call("shuf {} > {}".format(tmp_str, shuffled_str), shell=True)
			subprocess.call("rm {}".format(tmp_str), shell=True)

				



if __name__ == "__main__":
	main()

#EOF