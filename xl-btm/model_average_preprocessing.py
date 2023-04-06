import pickle
import argparse

def main(args):

	with open("data/mc4_cluster{}_lang_dists.pkl".format(args.cluster_k), 'rb') as f:
		c = pickle.load(f)

	c = c[args.lang]

	c = [(x, y) for x, y in c.items()]
	c = sorted(c, key=lambda x: x[1], reverse=True)[:args.subset_k]
	#normalize values in c_dict
	c_sum = sum([z[1] for z in c])
	c = [(x, y/c_sum) for x,y in c]

	print(args.lang, args.cluster_k, args.subset_k)
	print([z[0] for z in c])
	print(','.join([str(z[1]) for z in c]))
	return




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-k", type=int, required=True)
    parser.add_argument("--subset-k", type=int, required=True)
    parser.add_argument("--lang", type=str, required=True)
    args = parser.parse_args()
    main(args)