import os
import random
import json
from copy import deepcopy

from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from accelerate import Accelerator
from accelerate import PartialState
from accelerate.utils import gather_object
import torch

from data_loader.hf_dataloader import HFDataloader
from prompts.paraphrase import Paraphrase
from models.llama_hf_instruct import LlamaClientInstruct
from models.llama_hf_base import LlamaClientBase
from models.llama_hf_lora import LlamaClientLoRA
from models.llama_hf_simple import LlamaClientSimple
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
    # print(f'available device: {os.environ["CUDA_VISIBLE_DEVICES"]}')
    root = os.path.join(args.data_dir, "datasets")
    if args.dataset_name == 'pile':
        dataset_dir_name = f'{args.dataset_name}_{args.subset_name}'
    else:
        dataset_dir_name = f'{args.dataset_name}'
    val_pair_dir = os.path.join(root, dataset_dir_name)
    os.makedirs(val_pair_dir, exist_ok=True)
    if 'lora' in args.paraphrase_model or 'simple'in args.paraphrase_model:
        if '/' in args.lora_name:
            model_name = args.lora_name.split('/')[1]
        else:
            model_name = args.lora_name
    else:
        model_name = args.paraphrase_model
    # model_name = args.paraphrase_model if 'lora' not in args.paraphrase_model else args.lora_name
    file_name = os.path.join(val_pair_dir, f'{args.split}_{model_name}_{args.n_tokens}_{args.prefix_ratio}prefix_{args.n_paraphrase}.jsonl')

    # file_name = os.path.join(val_pair_dir, f'{args.split}_{model_name}_{args.n_tokens}_{args.prefix_ratio}prefix_{args.repetition_penalty}rep_{args.temperature}temp_{args.top_p}top_{args.num_beams}beam_{args.load_in_8bit}8bit_{args.n_paraphrase}paraphrase.jsonl')
    # define paraphrasing model
    state = PartialState()
    if 'llama' in args.paraphrase_model:
        if 'instruct' in args.paraphrase_model:
            model_client =  LlamaClientInstruct(model_name=args.paraphrase_model, device=args.device)
        elif 'lora' in args.paraphrase_model:
            model_name_string =  args.paraphrase_model.replace("_lora", "")
            model_client =  LlamaClientLoRA(model_name=model_name_string, 
                data_dir=os.path.join(args.data_dir, f'model/{dataset_dir_name}/'), 
                lora_name=args.lora_name,
                device=state.device,
                load_in_8bit=args.load_in_8bit
            )
        elif 'simple' in args.paraphrase_model:
            model_name_string =  args.paraphrase_model.replace("_simple", "")
            model_client =  LlamaClientSimple(model_name=model_name_string, 
                data_dir=os.path.join(args.data_dir, f'model/{dataset_dir_name}/'), 
                lora_name=args.lora_name,
                device=state.device,
                load_in_8bit=args.load_in_8bit
            )
            print(f'device: {state.device}')
            print(f'torch.cuda.current_device(): {torch.cuda.current_device()}')
        else:
            model_client =  LlamaClientBase(model_name=args.paraphrase_model, device=args.device)
    tokenizer = model_client.tokenizer
    # load original data
    suspect_dataloader = HFDataloader(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name, split=args.split, subset_name=args.subset_name)
    suspect_dataset = suspect_dataloader.get_dataset(n_sample=args.n_sample, shuffle=False)
    # sample tokens from the original data
    if args.prefix_ratio is not None:
        _, prefix_text_list, suspect_text_list, seq_lengths = truncate_and_split_dataset_with_tokenizer(
            data=suspect_dataset[suspect_dataloader.text_column_name],
            n_tokens=args.n_tokens,
            prefix_ratio=args.prefix_ratio,
            tokenizer=tokenizer)
    print(seq_lengths)
    # get paraphrased data
    para_text_lists = [[] for _ in range(args.n_paraphrase)]
    # accelerator = Accelerator()
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
                        para_results = []
                        prefix_text_batch_indexed = list(enumerate(prefix_text_batch))
                        with state.split_between_processes(prefix_text_batch_indexed) as prefix_text_per_gpu_indexed:
                            indexes, prefix_text_per_gpu = zip(*prefix_text_per_gpu_indexed)
                            # print(f'input:{prefix_text_per_gpu}')
                            para_text_per_gpu = model_client.get_text(
                                prefix_text_per_gpu,
                                n_tokens=args.n_tokens+1,
                                repetition_penalty=args.repetition_penalty,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                top_k=args.top_k,
                                do_sample=args.do_sample,
                                num_beams=args.num_beams
                            )
                            
                            # print(f'output:{para_text_per_gpu}')
                            para_text_per_gpu_indexed = list(zip(indexes, para_text_per_gpu))
                            para_results.extend(para_text_per_gpu_indexed)
                        state.wait_for_everyone()
                        para_text_per_gpu_indexed = gather_object(para_results)
                        # print(f'all outputs: {para_text_per_gpu_indexed}')
                        if state.is_main_process:
                            para_text_batch = [None] * len(para_text_per_gpu_indexed)
                            for index, para_text in para_text_per_gpu_indexed:
                                # print(f'reordering {index}, {para_text}')
                                para_text_batch[index] = para_text
                            # remove prefixes in model outputs for current batch
                            assert len(prefix_text_batch) == len(para_text_batch)
                            assert len(prefix_text_batch) == len(seq_length_batch)
                            for i_prefix in range(len(prefix_text_batch)):
                                para_text = para_text_batch[i_prefix]
                                # print(para_text)
                                n_tokens_prefix = int(seq_length_batch[i_prefix]*args.prefix_ratio)
                                n_tokens_suffix = seq_length_batch[i_prefix]-n_tokens_prefix
                                # para_text_trunc = get_first_k_tokens_sentence_with_tokenizer(para_text, n_tokens_trunc, tokenizer)
                                # para_text_batch[i_prefix] = para_text_trunc[len(prefix_text_batch[i_prefix]):]
                                para_text_batch[i_prefix] = remove_k1_get_k2_tokens_sentence_with_tokenizer(para_text, n_tokens_prefix, n_tokens_suffix, tokenizer)
                                if len(para_text_batch[i_prefix]) == 0:
                                    print(f'prefix: {prefix_text_batch[i_prefix]}')
                                    print(f'sequence: {para_text}')
                    if state.is_main_process:
                        para_text_lists[para_idx].extend(para_text_batch)
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
    if state.is_main_process:
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
    parser.add_argument("--subset_name", type=str, default="Wikipedia (en)")
    parser.add_argument("--split", type=str, default="test_original_32768")
    parser.add_argument("--paraphrase_model", type=str, default="llama3_8b_lora") # "llama3.2_1b_lora", "llama3_8b_lora", "llama3_8b", "llama3_8b_instruct", "llama3_8b_simple"
    parser.add_argument("--lora_name", type=str, default='llama_pile_test_original_32768_50epoch', help='Specify LoRA model name')
    parser.add_argument("--prefix_ratio", type=float, default=0.5, help='precentage of prefix sequence')
    parser.add_argument("--batch_size", type=int, default=512, help='Batch size of the paraphrase model')
    parser.add_argument("--n_tokens", type=int, default=128, help='Token length of each give sample')
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--do_sample", type=int, default=1)
    parser.add_argument("--load_in_8bit", type=int, default=1)
    parser.add_argument("--n_paraphrase", type=int, default=1, help='Number of paraphrases')
    parser.add_argument("--use_parallel_inference", type=int, default=1)
    parser.add_argument("--n_sample", type=int, default=32768)
    # parser.add_argument("--n_val", type=int, default=600)
    parser.add_argument("--device", type=int, default=3)
    args = parser.parse_args()
    args.do_sample = bool(args.do_sample)
    args.load_in_8bit = bool(args.load_in_8bit)

    print(args, flush=True)
    
    main(args)
