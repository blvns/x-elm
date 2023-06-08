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
	if os.path.isfile("./mc4_train_{}_mapping.pkl".format(sd)):
		with open("./mc4_train_{}_mapping.pkl".format(sd), 'rb') as f:
			doc_lang_mapping = pickle.load(f)

	else:
		path_prefix = os.path.join(DATA_DIR, SPLIT_DIR)

		#timeout=99999
		#results_arr = Parallel(n_jobs=, timeout=timeout)(delayed(_get_id_lang_mapping)(sd, path_prefix) for sd in tqdm(shard_dirs))
		doc_lang_mapping = _get_id_lang_mapping(sd, path_prefix)

		#write this mapping out to file
		with open("./mc4_train_{}_mapping.pkl".format(sd), 'wb') as f:
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

		if os.path.isfile("./mc4_cluster{}_lang_dists.pkl".format(size_dir)):
			with open("./mc4_train_mapping.pkl".format(sub_dir), 'rb') as f:
				cluster_dist = pickle.load(f)

		else:
			path_prefix = os.path.join(CLUSTER_DIR, str(size_dir), 'train')
			#get all shard dirs in training data 
			shard_dirs = os.listdir(path_prefix) 
			
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
			with open("./mc4_cluster{}_lang_dists.pkl".format(size_dir), 'wb') as f:
				pickle.dump(cluster_dist, f)

		#make into heatmap format
		x_labels = ['en', 'es', 'bg', 'ru', 'de', 'el', 'hi', 'ur', 'fr', 'ar', 'vi', 'sw', 'ja', 'ko', 'tr', 'zh'] #all langs
		y_labels = [i for i in list(range(0, size_dir))]

		data = []

		for lang in x_labels:
			lang_data = []
			for cluster_id in y_labels:
				val = 0
				if lang in cluster_dist and cluster_id in cluster_dist[lang]:
					val = cluster_dist[lang][cluster_id]
				lang_data.append(val)
			
			#normalize lang_data
			lang_total = sum(lang_data)
			lang_data = [d/lang_total for d in lang_data]
			data.append(lang_data)

		print(data)
		#make a heatmap of auto clusters dists at k=4, 8, 16
		sns.heatmap(data, xticklabels=[l.upper() for l in x_labels], yticklabels=y_labels)
		plt.tight_layout()
		plt.savefig('./cluster{}_dist.png'.format(size_dir), dpi=800)

if __name__ == "__main__":
	main()

#EOF