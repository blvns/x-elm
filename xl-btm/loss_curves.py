import argparse
import json

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_context('talk')
sns.set_style("white")

FILTERS = {
	#'train_loss':('train_inner', 'loss', 'num_updates', 'Train Loss'),
	'train_ppl':('train_inner', 'ppl', 'num_updates', 'Train PPL'),
	#'valid_loss':('valid', 'valid_loss', 'valid_num_updates', 'Valid Loss'),
	'valid_ppl':('valid', 'valid_ppl', 'valid_num_updates', 'Valid PPL'),
}

def main(args):

	filter_name, metric_name, step_name, y_label = FILTERS[args.filter]

	colors = sns.color_palette("magma", len(args.data_paths))

	for i, data_path in enumerate(args.data_paths):
		metric_arr = []
		step_arr = []

		#process metrics from log file
		with open(data_path, 'r') as f:
			for line in f:
				line = line.strip()
				try:
					_, _, x, y = line.split(' | ')
					if x == filter_name:
						y_dict = json.loads(y)
						metric_arr.append(float(y_dict[metric_name]))
						step_arr.append(int(y_dict[step_name]))

				except ValueError:
					continue

		#generate viz of metric curve
		df = {'steps': step_arr, 'metric': metric_arr}
		df = pd.DataFrame(df)
		sns.lineplot(df, x='steps', y='metric', color=colors[i], label=str(i))

	plt.xlabel('Steps')
	plt.ylabel(y_label)
	if len(args.data_paths) > 1: plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
	plt.tight_layout()
	plt.savefig('{}_curve.png'.format(args.filter), dpi=800)
	plt.clf()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_paths', type=str, required=True, nargs='+')
	parser.add_argument('--filter', type=str, required=True,
		choices=['train_ppl', 'valid_ppl'])
	args = parser.parse_args() 

	main(args)

#EOF