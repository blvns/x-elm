import os
from tqdm import tqdm
import subprocess
import json
from joblib import Parallel, delayed
import spacy

SPACY_LANGS = ['zh', 'ko', 'ja']
SPACE_NAMES = {
	'zh': 'zh_core_web_sm',
	'ko': 'ko_core_news_sm',
	'ja': 'ja_core_news_sm',
}

def _count(path_prefix, shard_dir, src_name):
	count_dict = {}

	#open source files
	dpath = os.path.join(path_prefix, shard_dir)
	src_path = os.path.join(dpath, src_name)
	src_f = open(src_path, 'r')

	#for each line in src_file, read in, get lang, count num words
	for line in src_f:
		line_json = json.loads(line) #convert to json dict
		lang = line_json["lang"]
		text = line_json["text"].encode("utf-8")
		byte_count = len(list(text))

		if lang not in count_dict:
			count_dict[lang] = 0
		count_dict[lang] += byte_count

	src_f.close()

	return count_dict

def main():
	SOURCE_DIR = "/gscratch/zlab/blvns/xl-btm/data/mc4/"
	SOURCE_SUB_DIRS = ['train'] #['train', 'valid']

	#for every subdirectory in source directory
	for sub_dir in SOURCE_SUB_DIRS:

		source_path_prefix = os.path.join(SOURCE_DIR, sub_dir)

		#get all shard dirs 
		shard_dirs = os.listdir(source_path_prefix) 
		shard_dirs = sorted(shard_dirs, reverse=True)


		print(shard_dirs)	

		#for every shard...
		source_file = 'mc4.jsonl'
		#target_file = 'mc4_reindexed.jsonl'

		timeout=99999
		counts_arr = Parallel(n_jobs=8, timeout=timeout)(delayed(_count)(source_path_prefix, shard_dir, source_file) for shard_dir in tqdm(shard_dirs))

		count_dict = {}
		for d in counts_arr:
			for l in d:
				if l not in count_dict:
					count_dict[l] = 0
				count_dict[l] += d[l]
		print(count_dict)
		print(sum([v for _, v in count_dict.items()]))

		#convert to gb

		import math
		def convert_size(size_bytes):
		   if size_bytes == 0:
		       return "0B"
		   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
		   i = int(math.floor(math.log(size_bytes, 1024)))
		   p = math.pow(1024, i)
		   s = round(size_bytes / p, 2)
		   return "%s %s" % (s, size_name[i])

		for l, v in count_dict.items():
			print(l, convert_size(v))
		print('all', convert_size(sum([v for _, v in count_dict.items()])))

	return




if __name__ == "__main__":
	main()

#EOF