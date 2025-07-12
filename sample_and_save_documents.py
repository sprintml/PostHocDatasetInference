import os
import random
import json
from data_loader.hf_dataloader import HFDataloader
from tqdm import tqdm
from itertools import chain
import numpy as np

from nltk.tokenize import NLTKWordTokenizer


def get_first_k_tokens_sentence(text, k):
    tokens_idx = list(NLTKWordTokenizer().span_tokenize(text))
    if len(tokens_idx) <= k:
        return text
    return text[:tokens_idx[k][0]]

def get_random_k_tokens_sentence(text, k):
    tokens_idx = list(NLTKWordTokenizer().span_tokenize(text))
    if len(tokens_idx) <= k:
        return text
    start_word_idx = random.randint(0, len(tokens_idx) - k)
    end_word_idx = start_word_idx+k-1
    return text[tokens_idx[start_word_idx][0]:tokens_idx[end_word_idx][1]]

def get_last_k_tokens_sentence(text, k):
    tokens_idx = list(NLTKWordTokenizer().span_tokenize(text))
    if len(tokens_idx) <= k:
        return text
    start_idx = len(tokens_idx) - k - 1
    return text[tokens_idx[start_idx][0]:]

def sentence_shorter_than_k_tokens(text, k):
    tokens_idx = list(NLTKWordTokenizer().span_tokenize(text))
    return len(tokens_idx) <= k


def split_texts_with_indices(texts, k, max_snippets=None):
    tokenizer = NLTKWordTokenizer()
    result = []
    print(f'Splitting {len(texts)} texts. Each snippet has {k} tokens. Maximum number of snippets is {max_snippets}')

    nums_snippets = []

    for text in tqdm(texts):
        # Tokenize the text and get spans
        spans = list(tokenizer.span_tokenize(text))

        # Create snippets of size k with their corresponding indices
        snippets_spans = [
            spans[i:i + k]
            for i in range(0, len(spans), k)
            if len(spans[i:i + k]) == k
        ]
        nums_snippets.append(len(snippets_spans))
        snippets = []
        for i in range(len(snippets_spans)):
            if i < len(snippets_spans) - 1:
                snippets.append(text[snippets_spans[i][0][0]:snippets_spans[i+1][0][0]])
            else:
                snippets.append(text[snippets_spans[i][0][0]:snippets_spans[i][-1][1]])

        # If max_snippets is specified, randomly sample the snippets
        if max_snippets:
            snippets = random.sample(snippets, min(max_snippets, len(snippets)))
        else:
            random.shuffle(snippets)

        result.append(snippets)
    print(nums_snippets)

    return result, np.mean(nums_snippets), np.median(nums_snippets)

def main(args):
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    dataloader = HFDataloader(data_dir=args.data_dir, dataset_name=args.dataset_name, split=args.split, subset_name=args.subset_name)
    dataset = dataloader.get_dataset(n_sample=args.n_sample, data_offset_idx=args.data_offset_idx)
    print(f'Sampled {len(dataset)} data points from dataset [{args.dataset_name}-{args.subset_name}-{args.split}], starting from {args.data_offset_idx}')

    extract_col_dict = {
        'cnn_dailymail': 'article', 
        'cc_news': 'text',
        'pile': 'text',
    }
    texts = [data[extract_col_dict[args.dataset_name]] for data in dataset]

    result, mean_num_snippets, median_num_snippets = split_texts_with_indices(texts, args.n_tokens, args.max_snippets)
    print(f'Mean snippets of dataset [{args.dataset_name}-{args.subset_name}-{args.split}]: {mean_num_snippets}')
    print(f'Median snippets of dataset [{args.dataset_name}-{args.subset_name}-{args.split}]: {median_num_snippets}')
    all_snippets = list(chain.from_iterable(result))
    random.shuffle(all_snippets)
    print(f'Total number of snippets: {len(all_snippets)}')

    # result file path
    root = os.path.join(args.data_dir, "datasets")
    if args.dataset_name == 'pile':
        val_pair_dir = os.path.join(root, f'{args.dataset_name}_{args.subset_name}')
    else:
        val_pair_dir = os.path.join(root, f'{args.dataset_name}')

    os.makedirs(val_pair_dir, exist_ok=True)

    file_name = os.path.join(val_pair_dir, f'{args.split}_{args.data_offset_idx}_{args.n_sample}_{args.n_tokens}token_max{args.max_snippets}.jsonl')
    
    print(f'Saving snippets...')
    with open(file_name, "w") as f:
        for snippet in tqdm(all_snippets):
            json_line = {}
            json_line['original'] = snippet

            f.write(json.dumps(json_line) + "\n")
        
    print(f'Snippets saved to {file_name}')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--dataset_name", type=str, default="pile")
    parser.add_argument("--subset_name", type=str, default='PubMed Abstracts') # choices=['USPTO Backgrounds', 'PubMed Abstracts', 'Wikipedia (en)', 'all']
    parser.add_argument("--split", type=str, default="train", help="For example, 'train','test','val','val+test'.")
    parser.add_argument("--data_offset_idx", type=int, default=0, help="Starting point of the documents")
    parser.add_argument("--n_sample", type=int, default=8192)
    parser.add_argument("--max_snippets", type=int, default=300, help="Maximum number of snippets per document. None means all snippets in a document will be extracted.")
    parser.add_argument("--n_tokens", type=int, default=32)

    args = parser.parse_args()

    subsets_list = [
        # "USPTO Backgrounds",
        # "PubMed Central",
        # "PubMed Abstracts",
        # "FreeLaw",
        # "Github",
        # "PhilPapers",
        # "ArXiv",
        # "StackExchange",
        # "DM Mathematics",
        # "NIH ExPorter",
        # "HackerNews",
        "Enron Emails",
        # "Wikipedia (en)",
        # "Pile-CC", 
        # "EuroParl"
        # "Ubuntu IRC"
        ]

    configs_list = [
        {"split": "train", "data_offset_idx": 0, "n_sample": 1200, "max_snippets": 30, "n_tokens": 32},
        {"split": "val+test", "data_offset_idx": 0, "n_sample": 1200, "max_snippets": 30, "n_tokens": 32},
        {"split": "train", "data_offset_idx": 0, "n_sample": 1200, "max_snippets": 30, "n_tokens": 64},
        {"split": "val+test", "data_offset_idx": 0, "n_sample": 1200, "max_snippets": 30, "n_tokens": 64},
    ]

    
    from multiprocessing import Pool, cpu_count

    def process_task(task_args):
        # Wrapper function to handle a single task
        main(task_args)

    # Prepare tasks for multiprocessing
    tasks = []
    for subsets_name in subsets_list:
        for config in configs_list:
            task_args = argparse.Namespace(**vars(args))  # Create a new Namespace object for each task
            task_args.subset_name = subsets_name
            vars(task_args).update(config)
            tasks.append(task_args)

    # Use multiprocessing Pool to process tasks in parallel
    num_workers = min(cpu_count(), len(tasks))  # Limit workers to the number of tasks
    with Pool(processes=num_workers) as pool:
        pool.map(process_task, tasks)
    