import argparse
import json

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_context('talk')
sns.set_style("white")

FILTERS = {
	'train_loss':('train_inner', 'loss', 'num_updates', 'blue'),
	'train_ppl':('train_inner', 'ppl', 'num_updates', 'red'),
	'valid_loss':('valid', 'valid_loss', 'valid_num_updates', 'purple'),
	'valid_ppl':('valid', 'valid_ppl', 'valid_num_updates', 'orange'),
}

def main(args):

	filter_name, metric_name, step_name, plot_color = FILTERS[args.filter]
	metric_arr = []
	step_arr = []

	#process metrics from log file
	is_beginning = True
	with open(args.data_path, 'r') as f:
		for line in f:
			line = line.strip()
			if is_beginning: #skip setup stuff
				if "Start iterating over samples" in line: is_beginning = False
				continue
			else: #processing training outputs
				_, _, x, y = line.split(' | ')
				if x == filter_name:
					y_dict = json.loads(y)
					metric_arr.append(float(y_dict[metric_name]))
					step_arr.append(int(y_dict[step_name]))

	#generate viz of metric curve
	df = {'steps': step_arr, 'metric': metric_arr}
	df = pd.DataFrame(df)
	sns.lineplot(df, x='steps', y='metric', color=plot_color)
	plt.tight_layout()
	plt.savefig('{}_curve.png'.format(args.filter), dpi=800)
	plt.clf()





if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', type=str, required=True)
	parser.add_argument('--filter', type=str, required=True,
		choices=['train_loss', 'train_ppl', 'valid_loss', 'valid_ppl'])
	args = parser.parse_args() 

	main(args)

#EOF