import os
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle

from matplotlib import pyplot as plt
import seaborn as sns

sns.set_context('talk')
sns.set_style("white")

def _count_data(data):
	wc = 0
	lc = 0

	shard_text = data['text'].to_list()
	for text in shard_text:
		wc += len(text.split())
		lc += len(text.split('\n'))

	dc = len(data)
	return wc, lc, dc

def _process_file(d, shard_path_prefix):
	lang, _ = d.split('.')
	d_path = os.path.join(shard_path_prefix, d) 
	data_json = pd.read_json(path_or_buf=d_path, lines=True)

	#count words, lines, docs in lang file
	wc, lc, dc = _count_data(data_json)
	return lang, wc, lc, dc

def main():

	DATA_DIR = "/gscratch/zlab/blvns/xl-btm/data/mc4/"
	DATA_SUB_DIRS = ['train', 'valid']
	
	#for every subdirectory in source directory
	for sub_dir in DATA_SUB_DIRS:
		path_prefix = os.path.join(DATA_DIR, sub_dir)
		word_counts = {}
		line_counts = {}
		doc_counts = {}


		#get all shard dirs 
		shard_dirs = os.listdir(path_prefix) 
		#print(shard_dirs)

		#for every shard...
		for sd in tqdm(shard_dirs):
			shard_path_prefix = os.path.join(path_prefix, sd)
			data_files = [f for f in os.listdir(shard_path_prefix) if os.path.isfile(os.path.join(shard_path_prefix, f)) and f.endswith('jsonl')]

			timeout=99999
			results_arr = Parallel(n_jobs=4, timeout=timeout)(delayed(_process_file)(d, shard_path_prefix) for d in data_files)

			for lang, wc, lc, dc in results_arr:
				#update word_counts, line_counts, doc_counts
				if lang in word_counts:
					word_counts[lang] += wc
					line_counts[lang] += lc
					doc_counts[lang] += dc
				else:
					word_counts[lang] = wc
					line_counts[lang] = lc
					doc_counts[lang] = dc

		with open("./mc4_{}_dists.pkl", 'wb') as f:
			pickle.dump((word_counts, line_counts, doc_counts), f)
						

		#make a histogram of data distributions for this split
		#https://seaborn.pydata.org/generated/seaborn.histplot.html
		wc_keys, wc_values = zip(**sorted(word_counts.items(), key=lambda x: x[1]))
		wc_df = pd.DataFrame.from_dict({'Lang': wc_keys, 'Count':wc_values})
		sns.histplot(data=wc_df, x='Lang', color='blue')
		plt.save_fig('./word_dist_{}.png'.format(sub_dir), dpi=800)
		plt.clf()

		lc_keys, lc_values = zip(**sorted(line_counts.items(), key=lambda x: x[1]))
		lc_df = pd.DataFrame.from_dict({'Lang': lc_keys, 'Count':lc_values})
		sns.histplot(data=lc_df, x='Lang', color='green')
		plt.save_fig('./line_dist_{}.png'.format(sub_dir), dpi=800)
		plt.clf()

		dc_keys, dc_values = zip(**sorted(doc_counts.items(), key=lambda x: x[1]))
		dc_df = pd.DataFrame.from_dict({'Lang': dc_keys, 'Count':dc_values})
		sns.histplot(data=dc_df, x='Lang', color='purple')
		plt.save_fig('./doc_dist_{}.png'.format(sub_dir), dpi=800)
		plt.clf()

if __name__ == "__main__":
	main()

#EOF
