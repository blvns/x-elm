import argparse
import itertools
import numpy as np
import os
import pickle
import random
import torch

from accelerate import Accelerator
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import load_dataset
from transformers import default_data_collator, AutoTokenizer


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def number_normalizer(tokens):
    """Map all numeric tokens to a placeholder.

    For many applications, tokens that begin with a number are not directly
    useful, but the fact that such a token exists can be relevant.  By applying
    this form of dimensionality reduction, some methods may perform better.
    """
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)


class NumberNormalizingVectorizer(TfidfVectorizer):
    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))

def load_data(dataset_name, dataset_dir, split, language, kmeans, vectorizer, hf_format=False, seed=42, max_eval_samples=None, group_texts=False, columns=None):
    
    set_seed(seed)
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained('facebook/xglm-1.7B')

    if not hf_format:
        file = os.path.join(args.dataset_dir, args.dataset_name)
        train_file = None
        validation_file = file

        data_files = {}
        dataset_args = {}
        data_files[split]= file
        extension = (
            train_file.split(".")[-1]
            if train_file is not None
            else validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = True
        if "json" in file:
            raw_datasets = load_dataset("json", data_files=data_files, cache_dir=None, use_auth_token=None, **dataset_args)
        elif "tsv" in file:
            raw_datasets = load_dataset("csv", delimiter="\t", data_files=data_files, cache_dir=None, use_auth_token=None, **dataset_args)
        elif "csv" in file:
            raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=None, use_auth_token=None, **dataset_args)
        elif "jsonl" in file:
            raw_datasets = load_dataset("json", data_files=data_files, cache_dir=None, use_auth_token=None, **dataset_args)
    else:
        raw_datasets = load_dataset(dataset_name, language, data_dir=dataset_dir)

    text_columns = columns
    
    remove_columns = [key for key in list(itertools.islice(raw_datasets[split], 1))[0].keys()]

    def tokenize_function(examples):
        texts = zip(*[examples[column] for column in text_columns])
        return tokenizer([" ".join(text) for text in texts])


    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=1,
            remove_columns=remove_columns,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

    block_size = 1024
    if block_size is None:
            block_size = tokenizer.model_max_length
            if block_size > 1024:
                print(
                    f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                    "Picking 1024 instead. You can change that default value by passing --block_size xxx."
                )
            block_size = 1024
    else:
        if block_size > tokenizer.model_max_length:
            print(
                f"The block_size passed ({block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(block_size, tokenizer.model_max_length)

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(itertools.chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    if group_texts:
        with accelerator.main_process_first():
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    #num_proc=1,
                    #load_from_cache_file=False,
                    desc=f"Grouping texts in chunks of {block_size}",
                )

    def generate_context_clusters(examples, tokenizer, vectorizer, kmeans):        
        clusters = []
        decoded_text = [tokenizer.decode(x) for x in examples['input_ids']]
        vectorized_text = vectorizer.transform(decoded_text)
        _, distances = kmeans.predict(torch.from_numpy(vectorized_text), return_distances=True)
        examples['clusters'] = distances
        del examples['input_ids']
        del examples['attention_mask']
        return examples

    with accelerator.main_process_first():
        tokenized_datasets = tokenized_datasets.map(
            lambda x: generate_context_clusters(x, tokenizer, vectorizer, kmeans),
            batched=True,
            desc="Generating context clusters",
        )
    return tokenized_datasets[split]


def load_model(path_to_model):
    with open(path_to_model, 'rb') as f:
        out = pickle.load(f)
    return out
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--dataset-name")
    parser.add_argument("--columns", nargs='+', default=['text'])
    parser.add_argument("--path-to-clusterer", type=Path)
    parser.add_argument("--mixture-folder")
    parser.add_argument("--hf-format", action='store_true')

    parser.add_argument("--split", default='validation')
    parser.add_argument("--seed", default=42)
    parser.add_argument('--lang', type=str)

    args = parser.parse_args()
    seed = args.seed
    print(args)

    # load vectorizer and clusterer
    vectorizer = load_model(args.path_to_clusterer / "tfidf.pkl")
    print('Loaded vectorizer!')
    kmeans = load_model(args.path_to_clusterer / "kmeans.pkl")
    print('Loaded kmeans clusters!')
   
    #eval_file = os.path.join(args.dataset_dir, '00000/mc4.jsonl')
    mixture_file_name = os.path.join(args.mixture_folder, f'{args.split}_{args.lang}', 'cluster.npy')

    # load dataset
    eval_dataset = load_data(args.dataset_name, args.dataset_dir, args.split, args.lang, kmeans, vectorizer,
                             hf_format=args.hf_format, group_texts=False, columns=args.columns)
    
    # batch dataset
    eval_dataloader = DataLoader(
            eval_dataset, collate_fn=default_data_collator, batch_size=1 #16
    )

    # cluster
    pbar = tqdm(eval_dataloader)
    clusters = []
    for batch in pbar:
        clusters.append(batch['clusters'])

    # build probability distribution and save
    cs = torch.cat(clusters, 0)
    # original: cs = torch.nn.functional.softmax(-cs ** 2 / 0.1, dim=1).cpu().numpy()
    cs = torch.nn.functional.softmax(-cs ** 2 / 1, dim=1).cpu().numpy()

    
    avg_cs = np.mean(cs, axis = 0)
    print(avg_cs)

    os.makedirs(os.path.dirname(mixture_file_name), exist_ok=True)
    np.save(mixture_file_name, cs)