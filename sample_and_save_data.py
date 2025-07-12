import os
import random
import json
from data_loader.hf_dataloader import HFDataloader
from tqdm import tqdm

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

def main(args):
    for n_tokens in args.n_tokens_list:
        dataloader = HFDataloader(data_dir=args.data_dir, dataset_name=args.dataset_name, split=args.split, subset_name=args.subset_name)
        dataset = dataloader.get_dataset(n_sample=args.n_sample, data_offset_idx=args.data_offset_idx)
        print(f'Sampled {len(dataset)} data points from dataset [{args.dataset_name}-{args.subset_name}-{args.split}], starting from {args.data_offset_idx}')
        
        root = os.path.join(args.data_dir, "datasets")
        if args.dataset_name == 'pile':
            val_pair_dir = os.path.join(root, f'{args.dataset_name}_{args.subset_name}')
        else:
            val_pair_dir = os.path.join(root, f'{args.dataset_name}')

        os.makedirs(val_pair_dir, exist_ok=True)
        file_name = os.path.join(val_pair_dir, f'{args.split}_{args.data_offset_idx}_{args.n_sample}_{args.token_offset}{n_tokens}.jsonl')

        extract_col_dict = {
            'cnn_dailymail': 'article', 
            'cc_news': 'text',
            'pile': 'text',
            'timothy_sykes': 'main_text'
        }

        with open(file_name, "w") as f:
            for data in tqdm(dataset):
                json_line = {}
                if 'paraphrase' in file_name:
                    json_line['prefix'] = data['prefix']
                    # truncate original suffix
                    if args.token_offset == 'first':
                        json_line['original'] = get_first_k_tokens_sentence(data['original'], k=n_tokens)
                    elif args.token_offset == 'random':
                        json_line['original'] = get_random_k_tokens_sentence(data['original'], k=n_tokens)
                    elif args.token_offset == 'last':
                        json_line['original'] = get_last_k_tokens_sentence(data['original'], k=n_tokens)
                    elif args.token_offset == 'shorter':
                        if sentence_shorter_than_k_tokens(data['original'], k=n_tokens):
                            json_line['original'] = data['original']
                        else:
                            continue
                    para_texts = data['paraphrase']
                    para_list = []
                    for para_text in para_texts:
                        if args.token_offset == 'first':
                            para_text = get_first_k_tokens_sentence(para_text, k=n_tokens)
                        elif args.token_offset == 'random':
                            para_text = get_random_k_tokens_sentence(para_text, k=n_tokens)
                        elif args.token_offset == 'last':
                            para_text = get_last_k_tokens_sentence(para_text, k=n_tokens)
                        elif args.token_offset == 'shorter':
                            if sentence_shorter_than_k_tokens(para_text, k=n_tokens):
                                para_text = para_text
                            else:
                                continue
                        para_list.append(para_text)
                    json_line['paraphrase'] = para_list
                else:
                    if (args.dataset_name == "pile" and args.split not in ['train', 'val', 'test', 'val+test']) or args.dataset_name == 'timothy_sykes' and args.split not in ['train']:
                        col_name = "original"
                    else:
                        col_name = extract_col_dict[args.dataset_name]
                    if args.token_offset == 'first':
                        json_line['original'] = get_first_k_tokens_sentence(data[col_name], k=n_tokens)
                    elif args.token_offset == 'random':
                        json_line['original'] = get_random_k_tokens_sentence(data[col_name], k=n_tokens)
                    elif args.token_offset == 'last':
                        json_line['original'] = get_last_k_tokens_sentence(data[col_name], k=n_tokens)
                    elif args.token_offset == 'shorter':
                        if sentence_shorter_than_k_tokens(data[col_name], k=n_tokens):
                            json_line['original'] = data[col_name]
                        else:
                            continue
                    

                f.write(json.dumps(json_line) + "\n")
            
        print(f'Truncated {args.token_offset} {n_tokens} tokens for each sample.')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/storage2/bihe/llm_data_detect")
    parser.add_argument("--dataset_name", type=str, default="pile")
    parser.add_argument("--subset_name", type=str, default="USPTO Backgrounds", choices=['USPTO Backgrounds', 'PubMed Abstracts', 'Wikipedia (en)', 'all'])
    parser.add_argument("--split", type=str, default="val", help="For example, 'train','test','val','val+test'.")
    parser.add_argument("--token_offset", type=str, default="random", choices=["first", "random", "last", "shorter"])
    parser.add_argument("--data_offset_idx", type=int, default=5000)
    parser.add_argument("--n_sample", type=int, default=4096)
    parser.add_argument("--n_tokens_list", type=int, default=[64])

    args = parser.parse_args()

    dataset_name = "pile"
    subsets_list = [
        # "USPTO Backgrounds",
        # "StackExchange",
        # "EuroParl",
        # "Pile-CC",
        # "Wikipedia (en)",
        # "PhilPapers"
        # "NIH ExPorter",
        # "HackerNews",
        # "PubMed Central",
        # "PubMed Abstracts",
        # "FreeLaw",
        # "Github",
        # "ArXiv",
        # "Ubuntu IRC",
        # "Enron Emails",
        # "NIH ExPorter",
        # "DM Mathematics"
        ]

    with open('/home/bihe/LLM_data_detect/subset_config_search.json', 'r') as file:
        config_dict = json.load(file)
    for subset in subsets_list:
        gen_configs = config_dict[subset]["gen_configs"]
        for gen_config in gen_configs:
            doc_idx = gen_config["doc_idx"]
            max_snippets = gen_config["max_snippets"]
            n_tokens_list = gen_config["n_tokens_list"]
            for n_token in n_tokens_list:
                print(subset, doc_idx, max_snippets)
                configs_list = [
                    {"split": f"train_{doc_idx}_{n_token}token_max{max_snippets}", "data_offset_idx": 0, "n_sample": 2000, "n_tokens_list": [n_token]},
                    {"split": f"train_{doc_idx}_{n_token}token_max{max_snippets}", "data_offset_idx": 2000, "n_sample": 2000, "n_tokens_list": [n_token]},
                    {"split": f"train_{doc_idx}_{n_token}token_max{max_snippets}", "data_offset_idx": 0, "n_sample": 4000, "n_tokens_list": [n_token]},
                    {"split": f"train_{doc_idx}_{n_token}token_max{max_snippets}", "data_offset_idx": 4000, "n_sample": 200000, "n_tokens_list": [n_token]},
                    {"split": f"val+test_{doc_idx}_{n_token}token_max{max_snippets}", "data_offset_idx": 0, "n_sample": 2000, "n_tokens_list": [n_token]},
                    {"split": f"val+test_{doc_idx}_{n_token}token_max{max_snippets}", "data_offset_idx": 2000, "n_sample": 2000, "n_tokens_list": [n_token]},
                    {"split": f"val+test_{doc_idx}_{n_token}token_max{max_snippets}", "data_offset_idx": 0, "n_sample": 4000, "n_tokens_list": [n_token]},
                    {"split": f"val+test_{doc_idx}_{n_token}token_max{max_snippets}", "data_offset_idx": 4000, "n_sample": 200000, "n_tokens_list": [n_token]},
                ]
                
                from multiprocessing import Pool, cpu_count

                def process_task(task_args):
                    # Wrapper function to handle a single task
                    main(task_args)

                # Prepare tasks for multiprocessing
                tasks = []
                for config in configs_list:
                    task_args = argparse.Namespace(**vars(args))  # Create a new Namespace object for each task
                    task_args.dataset_name = dataset_name
                    task_args.subset_name = subset
                    vars(task_args).update(config)
                    tasks.append(task_args)

                # Use multiprocessing Pool to process tasks in parallel
                num_workers = min(cpu_count(), len(tasks))  # Limit workers to the number of tasks
                with Pool(processes=num_workers) as pool:
                    pool.map(process_task, tasks)
