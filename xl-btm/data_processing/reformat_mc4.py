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

		#get all shard dirs 
		shard_dirs = os.listdir(source_path_prefix) 
		#print(shard_dirs)

		#for every shard...
		for shard_dir in tqdm(shard_dirs):

			source_path_prefix2 = os.path.join(source_path_prefix, shard_dir)

			#get all files in this subdirectory
			source_files = [f for f in os.listdir(source_path_prefix2) if os.path.isfile(os.path.join(source_path_prefix2, f))]

			#for every file in shard directory...
			for sf in source_files:
				#parse information
				lang, _ = sf.split('.')

				#make target directory if it doesn't exist
				target_path_prefix = os.path.join(TARGET_DIR, sub_dir, shard_dir)
				pathlib.Path(target_path_prefix).mkdir(parents=True, exist_ok=True)

				#TODO move file to target directory
				source_path = os.path.join(source_path_prefix2, sf)
				tf = '{}.jsonl'.format(lang)
				target_path = os.path.join(target_path_prefix, tf)
				#print(source_path)
				#print(target_path)
				os.rename(source_path, target_path)

if __name__ == "__main__":
	main()

#EOF