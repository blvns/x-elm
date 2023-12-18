import os
import pathlib
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

def main(args):
	DATA_DIR = "/gscratch/zlab/blvns/xl-btm/data/mc4_adapt/train"

	#get all shard dirs 
	shard_dirs = os.listdir(DATA_DIR) 
	shard_dirs = sorted(shard_dirs)
	print(shard_dirs)

	num_shards = len(shard_dirs)

	for lang in args.langs:
		lang_file = '{}.jsonl'.format(lang)
		all_lang_data = []

		#load and concat all data
		for sd in tqdm(shard_dirs):
			lang_fp = os.path.join(DATA_DIR, sd, lang_file)
			if os.path.isfile(lang_fp):
				x = pd.read_json(path_or_buf=lang_fp, lines=True)
				all_lang_data.append(x)
		all_lang_data = pd.concat(all_lang_data, ignore_index=True, sort=False)
		print(len(all_lang_data))

		#split along num Shards
		new_shards = np.array_split(all_lang_data, num_shards)

		#reindex data
		for shard in new_shards:
			element_idxs = list(range(0, len(shard)))
			shard['id'] = element_idxs
		print(new_shards[5])

		#write new shards to file
		for sd, shard in zip(shard_dirs, new_shards):
			output_path = os.path.join(DATA_DIR, sd, lang_file)
			#write out shard as .jsonl file
			shard.to_json(output_path, lines=True, orient='records', \
				force_ascii=False)
	return


				



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--langs', nargs='+', required=True)
	args = parser.parse_args()
	print(args)

	main(args)

#EOF