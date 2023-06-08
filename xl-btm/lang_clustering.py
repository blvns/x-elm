import os
from pathlib import Path
from tqdm import tqdm
import json
from joblib import Parallel, delayed
import argparse
import pickle

import lang2vec.lang2vec as l2v
from iso639 import Lang
import math

#def _pairwise(iterable):
#    #"s -> (s0, s1), (s2, s3), (s4, s5), ..."
#    a = iter(iterable)
#    return zip(a, a)

#greedy assignment to clusters based on distance
def _find_min_cluster_dist(cluster_map, cluster_ids):

	clustering = []
	dist = 0
	while len(cluster_map) > 0:
		z = sorted(cluster_map, key=lambda x: x[-1])[0]
		pair = [z[0], z[1]]
		clustering.append(pair)
		dist += z[2]

		#remove pairings that contain already assigned clusters
		cluster_map = [(x0, x1, y) for x0, x1, y in cluster_map if x0 not in pair and x1 not in pair]

	#return potential clustering with lowest overall distance
	return clustering, dist

def _avg_vectors(v_i, v_j):
	avg = []
	for x, y in zip(v_i, v_j):
		if x == '--':
			avg.append(y)
		elif y == '--':
			avg.append(x)
		else:
			avg.append((x+y)/2)
	return avg

def _cosine_dist(v_i, v_j):
	#skip missing values
	sum_xy = sum([x*y for x, y in zip(v_i, v_j) if x != '--' and y != '--'])
	sum_xx = sum([x*x for x in v_i if x != '--'])
	sum_yy = sum([y*y for y in v_j if y != '--'])
	return 1-(sum_xy/math.sqrt(sum_xx*sum_yy))

def _min_cosine_dist(vset_i, vset_j):
	distances = []
	for v_i in vset_i:
		for v_j in vset_j:
			d = _cosine_dist(v_i, v_j)
			distances.append(d)
	return min(distances)

def _train_clusterer(lang_codes, num_clusters):
	cluster_map = {}
	#skip learning clusters if every lang is a seperate cluster
	if num_clusters == len(lang_codes):
		cluster_map = {lang: i for i, lang in enumerate(lang_codes)}
		return cluster_map

	#else..
	#initialize clustering
	#and vector representations for each lang
	lang2_codes = [Lang(lang).pt3 for lang in lang_codes]
	v_dict = l2v.get_features(lang2_codes, "syntax_average", minimal=True)
	#cluster_vecs = [v_dict[Lang(lang).pt3] for lang in lang_codes]
	#min distance clustering
	cluster_vecs = [[v_dict[Lang(lang).pt3]] for lang in lang_codes]
	clusters = [[l] for l in lang_codes]

	#keep clustering into binary groups until we reach num_clusters
	while len(clusters) > num_clusters:
		print(clusters)
		#get distances between all pairs of clusters
		cluster_dists = []
		for i, vset_i in enumerate(cluster_vecs):
			for j, vset_j in enumerate(cluster_vecs):
				if i >= j: continue
				#d = _cosine_dist(v_i, v_j)
				#TODO implement this
				#min distance clustering
				d = _min_cosine_dist(vset_i, vset_j)
				cluster_dists.append((i, j, d))

		#find new set of clusters with min distance (greedy)
		cluster_ids = list(range(0, len(clusters)))
		assignments, dist = _find_min_cluster_dist(cluster_dists, cluster_ids)

		clusters = [clusters[x]+clusters[y] for x, y in assignments]
		#cluster_vecs = [_avg_vectors(cluster_vecs[x], cluster_vecs[y]) for x, y in assignments]
		#min distance clustering
		cluster_vecs = [cluster_vecs[x]+cluster_vecs[y] for x, y in assignments]

	print(clusters)
	#calculate cluster map
	cluster_map = {}
	for i, langs in enumerate(clusters):
		for l in langs:
			cluster_map[l] = i

	return cluster_map


def _assign_to_cluster(lang2cluster, source_path_prefix, cluster_path_prefix, shard_dir):
	source_file = 'mc4.jsonl'
	cluster_file = 'mc4.jsonl'

	source_path = os.path.join(source_path_prefix, shard_dir, source_file)
	cluster_path = os.path.join(cluster_path_prefix, shard_dir)
	Path(cluster_path).mkdir(parents=True, exist_ok=True) #make directory if doesn't exist
	cluster_path = os.path.join(cluster_path, cluster_file)

	src_f = open(source_path, 'r')
	trg_f = open(cluster_path, 'w')

	for line in src_f:
		line_json = json.loads(line) #convert to json dict
		line_id = line_json['id'] #get document id
		line_lang = line_json['lang'] #get language of document
		line_cluster = lang2cluster[line_lang] #lookup appropriate cluster for document
		#generate cluster assignment line
		cluster_json = {'sp_id': '{}|{}'.format(source_path, line_id), 'cluster': line_cluster}
		output_str = json.dumps(cluster_json)
		#write to cluster file
		trg_f.write(output_str+'\n')

	src_f.close()
	trg_f.close()
	return

def main(args):
	SOURCE_DIR = "/gscratch/zlab/blvns/xl-btm/data/mc4/"
	SUB_DIRS = ['train', 'valid']
	CLUSTER_DIR = "/gscratch/zlab/blvns/xl-btm/clusters_lang/mc4/{}".format(args.num_clusters)

	LANGS = ['en', 'fr', 'es', 'de', 'el', 'bg', 'ru', 'tr', 'ar', 'vi', 'zh', 'hi', 'sw', 'ur', 'ja', 'ko']

	#partition langs into clusters based on typological similarity
	cluster_map = _train_clusterer(LANGS, args.num_clusters)
	print(cluster_map)

	#log cluster map to file
	Path(CLUSTER_DIR).mkdir(parents=True, exist_ok=True) #make directory if doesn't exist
	output_path = os.path.join(CLUSTER_DIR, 'cluster_map.pkl')
	with open(output_path, 'wb') as f:
		pickle.dump(cluster_map, f)

	#assign all documents to their respective clusters
	for sub_dir in SUB_DIRS:

		source_path_prefix = os.path.join(SOURCE_DIR, sub_dir)
		cluster_path_prefix = os.path.join(CLUSTER_DIR, sub_dir)

		#get all shard dirs 
		shard_dirs = os.listdir(source_path_prefix) 
		shard_dirs = sorted(shard_dirs)

		print(shard_dirs)	
		
		#for every shard...
		timeout=99999
		_ = Parallel(n_jobs=4, timeout=timeout)(delayed(_assign_to_cluster)(cluster_map, source_path_prefix, cluster_path_prefix, shard_dir) for shard_dir in tqdm(shard_dirs))

	return




if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--num_clusters', type=int, required=True,
		choices=[2, 4, 8, 16])
	args = parser.parse_args()
	print(args)

	main(args)

#EOF