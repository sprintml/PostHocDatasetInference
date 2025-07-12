'''
This file will call various perturbations, and add perturbed versions of the data to the dataset as different subsets.
'''
import random
from dataloader import load_data, pile_mapper
from transform import generate_perturbations
from datasets import load_dataset
import os
import json
import nltk


def main(args):    
    if args.dataset_name == 'pile' or args.dataset_name.startswith('dolma'):
        dataset_dir_name = f'{args.dataset_name}_{args.subset_name}'
    else:
        dataset_dir_name = f'{args.dataset_name}'

    root = os.path.join(args.data_dir, "datasets")
    val_pair_dir = os.path.join(root, f'{dataset_dir_name}')
    file_name = os.path.join(val_pair_dir, f'{args.split}.jsonl')
    # dataset_path = f"data/{dataset_name}_{split}.jsonl"
    dataset = load_dataset("json", data_files=file_name, split="train")
    if args.dataset_col != 'paraphrase':
        raw_texts = dataset[args.dataset_col]
        if args.prefix:
            updated_file_name = os.path.join(val_pair_dir, f'{args.split}_{args.dataset_col}_prefix.jsonl')
        else:
            updated_file_name = os.path.join(val_pair_dir, f'{args.split}_{args.dataset_col}.jsonl')
    else:
        raw_texts = [paraphrases[args.idx_paraphrase] for paraphrases in dataset[args.dataset_col]]
        # raw_texts = []
        # paraphrase_idx_list = []
        # for paraphrases in dataset[args.dataset_col]:
        #     idx = random.randint(0, len(paraphrases))
        #     raw_texts.append(paraphrases[idx])
        #     paraphrase_idx_list.append(idx)
        #     print(f'* Processing {idx}-th paraphrase...')
        if args.prefix:
            updated_file_name = os.path.join(val_pair_dir, f'{args.split}_{args.dataset_col}{args.idx_paraphrase}_prefix.jsonl')
        else:
            updated_file_name = os.path.join(val_pair_dir, f'{args.split}_{args.dataset_col}{args.idx_paraphrase}.jsonl')
    if args.prefix:
        for i, text in enumerate(raw_texts):
            raw_texts[i] = dataset['prefix'][i]+text
    print(type(raw_texts))
    print(type(raw_texts[0]))
    for text in raw_texts:
        if len(text) == 1:
            print(text)
    print(len(raw_texts))
    # add the perturbations
    perturbed_texts_dictionary = generate_perturbations(raw_texts)
    perturbation_styles = list(perturbed_texts_dictionary.keys())
    
    # save all the texts to a json lines file
    with open(updated_file_name, "w") as f:
        for i, text in enumerate(raw_texts):
            json_line = {}
            json_line["text"] = text
            for style in perturbation_styles:
                json_line[style] = perturbed_texts_dictionary[style][i]
            f.write(json.dumps(json_line) + "\n")
    
    # record which indexes are chosen
    # if args.prefix:
    #     updated_file_name = os.path.join(val_pair_dir, f'{args.split}_{args.dataset_col}_prefix_idx.jsonl')
    # else:
    #     updated_file_name = os.path.join(val_pair_dir, f'{args.split}_{args.dataset_col}_idx.jsonl')
    # with open(updated_file_name, "w") as f:
    #     json_line = {}
    #     json_line["idx_paraphrase"] = paraphrase_idx_list
    #     f.write(json.dumps(json_line) + "\n")
    # print(f"Data saved to {updated_file_name}")
                    

            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/storage2/bihe/llm_data_detect")
    parser.add_argument("--dataset_name", type=str, default="pile")
    parser.add_argument("--subset_name", type=str, default="Wikipedia (en)", help='Specify the subset name for PILE dataset')
    parser.add_argument('--split', type=str, default="train_original_4096_llama_pile_train_original_4096_200epoch_paraphrase", help='The split of the dataset to use')
    parser.add_argument('--dataset_col', type=str, default="original", choices=['original', 'paraphrase', 'prefix'])
    parser.add_argument('--prefix', type=int, default=0, help='whether to concatenate the prefix text')
    parser.add_argument('--idx_paraphrase', type=int, default=0, help='which paraphrase to process')
    args = parser.parse_args()
    
    main(args)
