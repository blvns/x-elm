#!/usr/bin/env python
# -*- coding: utf-8 -*- 

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
#from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
import torch
import torch.nn.functional as F
from datasets import load_dataset

import argparse
from tqdm import tqdm
import os
import numpy as np
import random
import pickle
import math

LABELS = {
	'xnli': ['Yes', 'Also', 'No'], #['entailment', 'neutral', 'contradiction']
	'xstorycloze': None
}

LANGS = {
	'xnli': ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh'],
	'xstorycloze': ['ar', 'en', 'es', 'eu', 'hi', 'id', 'my', 'ru', 'sw', 'te', 'zh'],
}

NUM_RUNS = 5
NUM_EVAL_EXAMPLES = 1000 #fixed across runs

def _create_context(task, k, split, eval_lang, datapath, rand_seed, run_id):
	context = ''
	if k == 0: return context

	data = load_dataset(task, eval_lang, data_dir=datapath)[split]
	data = data.shuffle(seed=rand_seed+run_id)
	data = data.select(list(range(k)))

	for example in data:
		if task == 'xnli': gold_label = example['label']
		elif task == 'xstorycloze': gold_label = example['sentence_quiz2'] if example['answer_right_ending'] == 2 else example['sentence_quiz1']
		example_text = _create_example(example, task, gold_label)
		context += example_text[0]+example_text[1]+'\n'

	return context

def _create_example(example, task, label):
	#based on buffet prompt
	if task == 'xnli':
		#format from xnli eval of XGLM
		prompt_template_shared = "{}, right? "
		prompt_template_scored = "{}, {}"
		example_text0 = prompt_template_shared.format(example['premise'])
		example_text1 = prompt_template_scored.format(label, example['hypothesis'])
	
	elif task == 'xstorycloze':
		#format from xstorycloze eval of XGLM
		prompt_template_shared = "{} {} {} {} "
		prompt_template_scored = "{}"
		example_text0 = prompt_template_shared.format(example['input_sentence_1'], example['input_sentence_2'], example['input_sentence_3'], example['input_sentence_4'])
		example_text1 = prompt_template_scored.format(label)

	return (example_text0, example_text1)

def _fwd(model, input_ids, context_key_values, apply_softmax):
	with torch.no_grad():
		input_ids = input_ids.to("cuda")
		output = model(input_ids, past_key_values=context_key_values, use_cache=True, return_dict=True)
	logits = output['logits'].squeeze()
	if apply_softmax: logits = F.log_softmax(logits, dim=-1)
	return logits.to("cpu")

def score(model, tokenizer, input_text, trg_lang, context_key_values, apply_softmax=False):
	scores = []

	#process shared subset first and get model state
	shared_text = input_text[0][0]
	example_key_values = _run_context(model, tokenizer, shared_text, history=context_key_values)

	#score example with each possible label
	for _, input_choice in input_text:
		#tokenize input
		input_ids = tokenizer(input_choice, return_tensors='pt')['input_ids']

		#pass example through model
		output = _fwd(model, input_ids[:,:-1], example_key_values, apply_softmax)	

		#get NLL of sequence from logits (conditioned on label)
		target = input_ids[:,1:].flatten()
		nll = F.nll_loss(output, target, reduction='mean')
		scores.append(nll.item())

	return scores

def _run_context(model, tokenizer, text, history=None):
	#tokenize input
	input_ids = tokenizer(text, return_tensors='pt')['input_ids']
	with torch.no_grad():
		input_ids = input_ids.to("cuda")
		output = model(input_ids, return_dict=True, use_cache=True, past_key_values=history)
	output = output['past_key_values']
	return output

def _calculate_acc(scores):
	acc = [1 if int(y) == probs.index(min(probs)) else 0 for y, probs in scores]
	acc = (sum(acc)/len(acc))*100
	return acc

def load_model(tokenizer_path, model_path):
	#tokenizer
	tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

	#model
	model = AutoModelForCausalLM.from_pretrained(model_path)
	model = model.to("cuda")
	model = model.eval()

	return tokenizer, model

def main(args):
	eval_task = args.task
	datapath = None
	apply_softmax=True

	if eval_task == 'xnli':
		context_split = 'validation'
		eval_split = 'test'
	elif eval_task == 'xstorycloze':
		context_split = 'train'
		eval_split = 'validation'
		datapath = './xstorycloze'

	#load model
	tokenizer, model = load_model('facebook/xglm-1.7B', args.model_path)

	for eval_lang in args.eval_lang:

		#demonstration langs
		demo_lang = args.demo_lang if args.demo_lang != None else eval_lang

		#load dataset 
		data = load_dataset(eval_task, eval_lang, data_dir=datapath)[eval_split]
		data = data.shuffle(seed=args.rand_seed)
		data = data.select(list(range(NUM_EVAL_EXAMPLES)))

		task_acc_by_run = []

		#multiple runs!!! 
		for run_id in range(NUM_RUNS):
			if args.k == 0 and run_id > 0:
				break

			#skip run if we already have results saved
			m = args.model_path.split('/')[-1]
			scores_filepath = '{}.{}.k{}.eval_{}.run{}.pkl'.format(m, eval_task, args.k, eval_lang, run_id)
			scores_filepath = os.path.join(args.output_dir, scores_filepath)
			if os.path.isfile(scores_filepath): 
				print('Skipping run {} of {}...'.format(run_id, NUM_RUNS))
				
				with open(scores_filepath, 'rb') as f:
					scores = pickle.load(f)
				acc = _calculate_acc(scores)
				task_acc_by_run.append(acc)
				continue

			context = None
			if args.k > 0:
				#create in-context examples text
				context = _create_context(eval_task, args.k, context_split, demo_lang, datapath, args.rand_seed, run_id)
				#cache context values to save on compute
				context = _run_context(model, tokenizer, context)

			#for each example -- get weight over all possible labels from all langs
			scores = []
			for example in tqdm(data):
				#format example for input
				if eval_task in ['xnli']:
					example_texts = [_create_example(example, eval_task, label=label_option) for label_option in LABELS[eval_task]]
					gold_label_idx = int(example['label'])
				elif eval_task == 'xstorycloze':
					example_texts = [_create_example(example, eval_task, label=example['sentence_quiz1']), _create_example(example, eval_task, label=example['sentence_quiz2'])]
					gold_label_idx = example['answer_right_ending']-1 #convert to 0-indexed
				#run through model + score
				x = score(model, tokenizer, example_texts, eval_lang, context, apply_softmax=apply_softmax)
				scores.append((gold_label_idx, x))

			#write out scores to file!
			print(scores_filepath)
			with open(scores_filepath, 'wb') as f:
				pickle.dump(scores, f)

			#calculate accuracy
			acc = _calculate_acc(scores)
			task_acc_by_run.append(acc)

		#get avg, var, std err across runs
		mean = sum(task_acc_by_run)/len(task_acc_by_run)
		var = sum([(s-mean)**2 for s in task_acc_by_run])/len(task_acc_by_run)
		std_dev = math.sqrt(var)
		std_err = std_dev/math.sqrt(len(task_acc_by_run))
		print('{} {} Acc: {:3.2f} ± {:4.3f}'.format(eval_task, eval_lang, mean, std_err))



if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument(
	    "--rand_seed", default=42, type=int
	)
	parser.add_argument(
	    "--k", default=0, type=int,
	    choices=[0, 1, 4, 8]
	)
	parser.add_argument(
	    "--model_path", type=str,
	    default="facebook/xglm-1.7B"
	    #pass in path from filesystem to eval custom models
	)
	parser.add_argument(
	    "--eval_lang", required=True, type=str, nargs='+'
	)
	parser.add_argument(
	    "--demo_lang", type=str,
	)
	parser.add_argument(
	    "--task", required=True, type=str,
	    choices=['xnli', 'xstorycloze']
	)
	parser.add_argument(
	    "--output_dir", default='.', type=str,
	)

	args = parser.parse_args()
	print(args)

	torch.manual_seed(args.rand_seed)
	os.environ['PYTHONHASHSEED'] = str(args.rand_seed)
	torch.cuda.manual_seed(args.rand_seed)
	torch.cuda.manual_seed_all(args.rand_seed)   
	np.random.seed(args.rand_seed)
	random.seed(args.rand_seed)
	torch.backends.cudnn.benchmark=False
	torch.backends.cudnn.deterministic=True

	main(args)

#EOF