import os
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

sns.set_context('talk')
sns.set_style("white")
	
def _get_id_lang_mapping(sd, path_prefix):
	shard_path_prefix = os.path.join(path_prefix, sd)
	d_path = os.path.join(shard_path_prefix, 'mc4.jsonl') 
	data_json = pd.read_json(path_or_buf=d_path, lines=True)

	doc_lang_mapping = {}
	for _, row in data_json.iterrows():
		x = row['id']
		y = row['lang']
		doc_lang_mapping[x] = y

	return doc_lang_mapping

def load_mapping(sd):
 	#get mapping of (shard, document_id) -> lang
	DATA_DIR = "/gscratch/zlab/blvns/xl-btm/data/mc4/"
	SPLIT_DIR = "train"
	doc_lang_mapping = {}

	#load mapping if already created
	if os.path.isfile("./data/mc4_train_{}_mapping.pkl".format(sd)):
		with open("./data/mc4_train_{}_mapping.pkl".format(sd), 'rb') as f:
			doc_lang_mapping = pickle.load(f)

	else:
		path_prefix = os.path.join(DATA_DIR, SPLIT_DIR)
		doc_lang_mapping = _get_id_lang_mapping(sd, path_prefix)

		#write this mapping out to file
		with open("./data/mc4_train_{}_mapping.pkl".format(sd), 'wb') as f:
			pickle.dump(doc_lang_mapping, f)

	return doc_lang_mapping

def process_cluster_file(path_prefix, shard_id):
	doc_lang_mapping = load_mapping(shard_id)

	shard_path_prefix = os.path.join(path_prefix, shard_id)
	d_path = os.path.join(shard_path_prefix, 'mc4.jsonl')
	data_json = pd.read_json(path_or_buf=d_path, lines=True)

	cluster_arr = []
	langs_arr = []
	for _, row in data_json.iterrows():
		x = int(row['sp_id'].split('|')[1])
		y = row['cluster']
		z = doc_lang_mapping[x]
		cluster_arr.append(y)
		langs_arr.append(z)

	return list(zip(cluster_arr, langs_arr))

def main():

	#get dist of langs in assigned clusters
	CLUSTER_DIR = "/gscratch/zlab/blvns/xl-btm/clusters/mc4"
	#dist of 1st valid shard
	CLUSTER_SIZE_DIRS = [4, 8, 16]

	#for every size of cluster
	for size_dir in CLUSTER_SIZE_DIRS:

		cluster_dist = {}

		if os.path.isfile("./data/mc4_cluster{}_lang_dists.pkl".format(size_dir)):
			with open("./data/mc4_cluster{}_lang_dists.pkl".format(size_dir), 'rb') as f:
				cluster_dist = pickle.load(f)

		else:
			path_prefix = os.path.join(CLUSTER_DIR, str(size_dir), 'train')
			#get all shard dirs in training data 
			shard_dirs = os.listdir(path_prefix) 
			shard_dirs = shard_dirs #DEBUGGING if we can do parallel with mappings cached
			
			#DEBUGGING if we can do parallel with mappings cached
			#timeout=99999
			#results_arr = Parallel(n_jobs=4, timeout=timeout)(delayed(process_cluster_file)(path_prefix, sd) for sd in tqdm(shard_dirs))

			for sd in tqdm(shard_dirs):
				results_arr = process_cluster_file(path_prefix, sd)
				
				for cluster_id, lang in results_arr:
					if lang not in cluster_dist:
						cluster_dist[lang] = {}
					if cluster_id not in cluster_dist[lang]:
						cluster_dist[lang][cluster_id] = 0
					cluster_dist[lang][cluster_id] += 1

			#write out train cluster dists
			with open("./data/mc4_cluster{}_lang_dists.pkl".format(size_dir), 'wb') as f:
				pickle.dump(cluster_dist, f)

		#make into heatmap format
		y_labels = ['en', 'es', 'bg', 'ru', 'de', 'el', 'fr', 'ar', 'hi', 'ur', 'vi', 'sw', 'ja', 'ko', 'tr', 'zh'] #all langs
		x_labels = [i for i in list(range(0, size_dir))]

		data = []

		#print(cluster_dist)

		for lang in y_labels:
			lang_data = []
			for cluster_id in x_labels:
				val = 0
				if lang in cluster_dist and cluster_id in cluster_dist[lang]:
					val = cluster_dist[lang][cluster_id]
				lang_data.append(val)
			
			#normalize lang_data
			lang_total = sum(lang_data)
			lang_data = [d/lang_total for d in lang_data]

			#print top 1 cluster for each expert
			lang_data = lang_data.index(max(lang_data))

			data.append(lang_data)

                '''
		#doing lang assign clusters
		cluster_dist = {
			4: {'en': 0, 'es': 0, 'bg': 0, 'ru': 0, 'ja': 1, 'ko': 1, 'tr': 1, 'zh': 1, 'de': 2, 'el': 2, 'fr': 2, 'ar': 2, 'hi': 3, 'ur': 3, 'vi': 3, 'sw': 3},
			8: {'en': 0, 'es': 0, 'de': 1, 'el': 1, 'ja': 2, 'ko': 2, 'hi': 3, 'ur': 3, 'bg': 4, 'ru': 4, 'fr': 5, 'ar': 5, 'tr': 6, 'zh': 6, 'vi': 7, 'sw': 7},
			16: {'en': 0, 'fr': 1, 'es': 2, 'de': 3, 'el': 4, 'bg': 5, 'ru': 6, 'tr': 7, 'ar': 8, 'vi': 9, 'zh': 10, 'hi': 11, 'sw': 12, 'ur': 13, 'ja': 14, 'ko': 15},
		}

		#make into heatmap format
		y_labels = ['en', 'es', 'bg', 'ru', 'de', 'el', 'fr', 'ar', 'hi', 'ur', 'vi', 'sw', 'ja', 'ko', 'tr', 'zh'] #all langs
		x_labels = [i for i in list(range(0, size_dir))]

		data = []
		for lang in y_labels:
			lang_data = []
			for cluster_id in x_labels:
				val = 0
				if cluster_id == cluster_dist[size_dir][lang]:
					val = 1
				lang_data.append(val)
			data.append(lang_data)
		'''

		#make a heatmap of auto clusters dists at k=4, 8, 16
		#cmap = sns.color_palette("rocket_r", as_cmap=True)
		#cmap = sns.light_palette("#9d053b", reverse=False, as_cmap=True) #, input='')
		cmap = sns.light_palette("#484f77", reverse=False, as_cmap=True) #, input='')
		

		sns.heatmap(data, xticklabels=x_labels, yticklabels=[l.upper() for l in y_labels], cmap=cmap)
		plt.xlabel('K = {}'.format(size_dir))
		plt.ylabel('Lang')
		#plt.xticks(rotation=0, fontsize=12)
		plt.xticks([])
		plt.tight_layout()
		plt.savefig('./tfidf_cluster{}_dist.png'.format(size_dir), dpi=800)
		plt.clf()

if __name__ == "__main__":
	main()

#EOF
