import os
import random
import json
from copy import deepcopy

from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize

from data_loader.hf_dataloader import HFDataloader
from prompts.paraphrase import Paraphrase
from models.llama_hf_instruct import LlamaClientInstruct
from models.llama_hf_base import LlamaClientBase
from models.llama_hf_lora import LlamaClientLoRA
from models.llama_hb import LlamaClientHB
from helpers.process_text import get_first_k_tokens_sentence, get_first_k_tokens_sentence_with_tokenizer, count_tokens_with_tokenizer
from helpers.process_text import truncate_and_split_dataset_with_tokenizer, get_second_k_tokens_sentence_with_tokenizer, remove_k1_get_k2_tokens_sentence_with_tokenizer

nltk.download('punkt')

def split_string_by_words(s, word_count=100):
    words = s.split()
    result = []
    for i in range(0, len(words), word_count):
        chunk = words[i:i + word_count]
        result.append(' '.join(chunk))

    return result

def random_substring(s, length):
    if len(s) <= length:
        return s
    start_index = random.randint(0, len(s) - length)
    end_index = start_index + length
    return s[start_index:end_index]


def main(args):
    # save directory
    root = os.path.join(args.data_dir, "datasets")
    val_pair_dir = os.path.join(root, f'{args.dataset_name}_{args.subset_name}')
    os.makedirs(val_pair_dir, exist_ok=True)
    model_name = args.paraphrase_model if 'lora' not in args.paraphrase_model else args.lora_name
    file_name = os.path.join(val_pair_dir, f'{args.split}_{model_name}_{args.prefix_ratio}prefix_paraphrase.jsonl')
    # define paraphrasing model
    if 'llama' in args.paraphrase_model:
        if 'instruct' in args.paraphrase_model:
            model_client = LlamaClientInstruct(model_name=args.paraphrase_model, device=args.device)
        elif 'lora' in args.paraphrase_model:
            model_client = LlamaClientLoRA(model_name='llama3_8b', 
                data_dir=os.path.join(args.data_dir, f'model/{args.dataset_name}_{args.subset_name}/'), 
                lora_name=args.lora_name
            )
        elif 'hb' in args.paraphrase_model:
            model_name = args.paraphrase_model.replace('_hb', '')
            model_client = LlamaClientHB(model_name=model_name)
        else:
            model_client =  LlamaClientBase(model_name=args.paraphrase_model, device=args.device)
    tokenizer = model_client.tokenizer
    # load original data
    suspect_dataloader = HFDataloader(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name, split=args.split, subset_name=args.subset_name)
    suspect_dataset = suspect_dataloader.get_dataset(n_sample=args.n_sample)
    print(suspect_dataset)
    # sample tokens from the original data
    if args.prefix_ratio is not None:
        _, prefix_text_list, suspect_text_list, seq_lengths = truncate_and_split_dataset_with_tokenizer(
            data=suspect_dataset[suspect_dataloader.text_column_name],
            n_tokens=args.n_tokens,
            prefix_ratio=args.prefix_ratio,
            tokenizer=tokenizer)
    # get paraphrased data
    para_text_lists = [[] for _ in range(args.n_paraphrase)]
    for para_idx in tqdm(range(args.n_paraphrase)):
        print(f'Generating paraphrases for round {para_idx}...')
        para_text_chunks = []
        if 'instruct' in args.paraphrase_model:
            # construct prompts
            prompt_template = Paraphrase()
            formatted_prompt = prompt_template.prepare_prompt(prompt_type='keep_length', original_text=suspect_text_chunk)
            # query model
            if 'llama' in args.paraphrase_model:
                model_output = model_client.get_text(
                text=formatted_prompt
            )
            # print(suspect_text_chunk)
            # print(model_output)
            for i in range(args.n_paraphrase):
                current_prefix = 'Example {order}:'.format(order=i+2)
                current_idx = model_output.find(current_prefix)+len(current_prefix)
                model_output = model_output[current_idx:]
                if i < args.n_paraphrase-1:
                    next_idx = model_output.find('\n\nExample {order}:'.format(order=i+3))
                    para_text_chunk = model_output[:next_idx]
                else:
                    para_text_chunk = model_output
                para_text_chunks.append(para_text_chunk)
                print(para_text_chunk)
        else:
            if args.prefix_ratio is not None:
                # iterate over batches
                for i_batch in tqdm(range(0, len(prefix_text_list), args.batch_size)):
                    prefix_text_batch = prefix_text_list[i_batch: i_batch+args.batch_size]
                    seq_length_batch = seq_lengths[i_batch: i_batch+args.batch_size]
                    if 'llama' in args.paraphrase_model:
                        para_text_batch = model_client.get_text(
                            prefix_text_batch,
                            n_tokens=args.n_tokens+1
                        )
                        # remove prefixes in model outputs for current batch
                        for i_prefix in range(len(prefix_text_batch)):
                            para_text = para_text_batch[i_prefix]
                            # print(para_text)
                            n_tokens_prefix = int(seq_length_batch[i_prefix]*args.prefix_ratio)
                            n_tokens_suffix = seq_length_batch[i_prefix]-n_tokens_prefix
                            # para_text_trunc = get_first_k_tokens_sentence_with_tokenizer(para_text, n_tokens_trunc, tokenizer)
                            # para_text_batch[i_prefix] = para_text_trunc[len(prefix_text_batch[i_prefix]):]
                            para_text_batch[i_prefix] = remove_k1_get_k2_tokens_sentence_with_tokenizer(para_text, n_tokens_prefix, n_tokens_suffix, tokenizer)
                    para_text_lists[para_idx].extend(para_text_batch)
                    if state.is_main_process:
                        print(f'---prefix---\n {prefix_text_batch[5]}', flush=True)
                        print(f'---suspect---\n {suspect_text_list[i_batch: i_batch+args.batch_size][5]}', flush=True)
                        print(f'---paraphrase---\n {para_text_batch[5]}', flush=True)
            else: # use the complete text as prompt
                # iterate over batches
                for i_batch in tqdm(range(0, len(suspect_text_list), args.batch_size)):
                    suspect_text_batch = suspect_text_list[i_batch: i_batch+args.batch_size]
                    seq_length_batch = seq_lengths[i_batch: i_batch+args.batch_size]
                    if 'llama' in args.paraphrase_model:
                        para_text_batch = model_client.get_text(
                            suspect_text_batch,
                            n_tokens=args.n_tokens+1,
                            batch_size=args.batch_size
                        )
                        # remove original texts in model outputs for current batch
                        for i_suspect in range(len(suspect_text_batch)):
                            para_text = para_text_batch[i_suspect]
                            # print(para_text)
                            # n_tokens_trunc = len(word_tokenize(suspect_text_batch[i_suspect]))*2
                            # para_text_trunc = get_first_k_tokens_sentence(para_text, n_tokens_trunc)
                            # para_text_batch[i_suspect] = para_text_trunc[len(suspect_text_batch[i_suspect]):]
                            n_tokens_trunc = seq_length_batch[i_prefix]//2
                            # para_text_trunc = get_first_k_tokens_sentence_with_tokenizer(para_text, n_tokens_trunc, tokenizer)
                            # para_text_batch[i_prefix] = para_text_trunc[len(prefix_text_batch[i_prefix]):]
                            para_text_batch[i_prefix] = get_second_k_tokens_sentence_with_tokenizer(para_text, n_tokens_trunc, tokenizer)
                   
                    para_text_lists[para_idx].extend(para_text_batch)
    
    # save original and paraphrased data
    with open(file_name, "w") as f:
        for i in range(len(suspect_text_list)):
            json_line = {}
            if args.prefix_ratio is not None:
                json_line['prefix'] = prefix_text_list[i]
            json_line['original'] = suspect_text_list[i]
            json_line['paraphrase'] = [para_text_lists[para_idx][i] for para_idx in range(args.n_paraphrase)]
            f.write(json.dumps(json_line) + "\n")

        # para_data['text'] = ' '.join(para_text_chunks)
        # save data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/storage2/bihe/llm_data_detect/")
    parser.add_argument("--dataset_name", type=str, default="pile")
    parser.add_argument("--subset_name", type=str, default="USPTO Backgrounds")
    parser.add_argument("--split", type=str, default="test_9000_2048_random384")
    parser.add_argument("--paraphrase_model", type=str, default="llama3.1_405b_hb", choices=["llama3_8b", "llama3_8b_lora", "llama3_8b_instruct", "llama3.1_405b_hb"])
    parser.add_argument("--lora_name", type=str, default='llama_pile_test_original_32768_50epoch', help='Specify LoRA model name')
    parser.add_argument("--prefix_ratio", type=float, default=0.5, help='precentage of prefix sequence')
    parser.add_argument("--batch_size", type=int, default=4, help='Batch size of the paraphrase model')
    parser.add_argument("--n_tokens", type=int, default=384, help='Token length of each give sample')
    parser.add_argument("--n_paraphrase", type=int, default=3, help='Number of paraphrases')
    parser.add_argument("--use_parallel_inference", type=int, default=1)
    parser.add_argument("--n_sample", type=int, default=2048)
    # parser.add_argument("--n_val", type=int, default=600)
    parser.add_argument("--device", type=int, default=3)
    args = parser.parse_args()

    print(args, flush=True)
    
    main(args)
