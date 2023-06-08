import os
from tqdm import tqdm
import subprocess
import json
from joblib import Parallel, delayed

SKIP_SHARD_MAX = 716

def _reindex(path_prefix, shard_dir, src_name, trg_name):

	dpath = os.path.join(path_prefix, shard_dir)
	src_path = os.path.join(dpath, src_name)

	#open source and target files
	src_f = open(src_path, 'r')
	trg_path = os.path.join(dpath, trg_name)
	trg_f = open(trg_path, 'w')

	#for each line in src_file, read in, update index, and write out to trg file
	reindex = 0
	for line in src_f:
		line_json = json.loads(line) #convert to json dict
		line_json['id'] = reindex #update index
		reindex += 1 #increment index for next doc
		reindexed_line = json.dumps(line_json) #convert json back to str
		trg_f.write(reindexed_line+'\n')

	src_f.close()
	trg_f.close()

	#remove original file after reindexing + rename
	#subprocess.call("rm {}".format(src_path), shell=True)
	subprocess.call("mv {} {}".format(trg_path, src_path), shell=True)

	return

def main():
	SOURCE_DIR = "/gscratch/zlab/blvns/xl-btm/data/mc4/"
	SOURCE_SUB_DIRS = ['valid'] #['train', 'valid']

	#for every subdirectory in source directory
	for sub_dir in SOURCE_SUB_DIRS:

		source_path_prefix = os.path.join(SOURCE_DIR, sub_dir)

		#get all shard dirs 
		shard_dirs = os.listdir(source_path_prefix) 
		shard_dirs = sorted(shard_dirs)

		if sub_dir == 'train':
			shard_dirs = [s for s in shard_dirs if int(s) > SKIP_SHARD_MAX]

		print(shard_dirs)	

		#for every shard...
		source_file = 'mc4.jsonl'
		target_file = 'mc4_reindexed.jsonl'

		timeout=99999
		_ = Parallel(n_jobs=4, timeout=timeout)(delayed(_reindex)(source_path_prefix, shard_dir, source_file, target_file) for shard_dir in tqdm(shard_dirs))

	return




if __name__ == "__main__":
	main()

#EOF