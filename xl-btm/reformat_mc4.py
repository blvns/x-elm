import os
import pathlib
from tqdm import tqdm

def main():

	SOURCE_DIR = "/gscratch/zlab/blvns/xl-btm/data/mc4/"
	SOURCE_SUB_DIRS = ['train', 'valid']
	TARGET_DIR = "/gscratch/zlab/blvns/xl-btm/data/mc4_reformatted/"

	#make target dir if doesn't exist (and subdirs)
	pathlib.Path(TARGET_DIR).mkdir(parents=True, exist_ok=True)
	for sub_dir in SOURCE_SUB_DIRS:
		target_sub_dir = os.path.join(TARGET_DIR, sub_dir)
		pathlib.Path(target_sub_dir).mkdir(parents=True, exist_ok=True)
	
	#for every subdirectory in source directory
	for sub_dir in SOURCE_SUB_DIRS:
		source_path_prefix = os.path.join(SOURCE_DIR, sub_dir)

		#get all files in this subdirectory
		source_files = [f for f in os.listdir(source_path_prefix) if os.path.isfile(os.path.join(source_path_prefix, f))]

		#for every file in subdirectory...
		for sf in tqdm(source_files):
			#parse information
			#print(sf)
			_, _, lang, shard, _ = sf.split('.')
			#print(lang, shard)

			#make target directory if it doesn't exist
			shard_dir = shard.zfill(5) 
			target_path_prefix = os.path.join(TARGET_DIR, sub_dir, shard_dir)
			#print(target_path_prefix)
			pathlib.Path(target_path_prefix).mkdir(parents=True, exist_ok=True)

			#TODO move file to target directory
			source_path = os.path.join(source_path_prefix, sf)
			tf = '{}.json'.format(lang)
			target_path = os.path.join(target_path_prefix, tf)
			#print(source_path)
			#print(target_path)
			os.rename(source_path, target_path)

if __name__ == "__main__":
	main()

#EOF