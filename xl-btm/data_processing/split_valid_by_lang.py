import os
import pathlib
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

LANGS = ['en', 'fr', 'es', 'de', 'el', 'bg', 'ru', 'tr', 'ar', 'vi', 'zh', 'hi', 'sw', 'ur', 'ja', 'ko']

def main():
	DATA_DIR = "/gscratch/zlab/blvns/xl-btm/data/mc4/valid"

	#get all shard dirs 
	shard_dirs = os.listdir(DATA_DIR) 
	shard_dirs = sorted(shard_dirs)
	print(shard_dirs)

	num_shards = len(shard_dirs)

	#load and concat all data
	for sd in tqdm(shard_dirs):
		shard_fp = os.path.join(DATA_DIR, sd, 'mc4.jsonl')
		shard = pd.read_json(path_or_buf=shard_fp, lines=True)
		for lang in LANGS:
			#print(lang)
			target_dir = "/gscratch/zlab/blvns/xl-btm/data/mc4/valid_{}/{}".format(lang, sd)
			pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)
			#filter to only get data for that lang
			x = shard.loc[shard['lang'] == lang]
			x.to_json(os.path.join(target_dir, 'mc4.jsonl'), lines=True, orient='records', force_ascii=False)

	return


				



if __name__ == "__main__":
	#parser = argparse.ArgumentParser()
	#parser.add_argument('--langs', nargs='+', required=True)
	#args = parser.parse_args()
	#print(args)

	main()

#EOF