"""
Loads various features for the train and val sets.
Trains a linear model on the train set and evaluates it on the val set.

Tests p value of differentiating train versus val on held out features.
"""

import os
import sys
import json
import random
import re

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel, chi2, norm, combine_pvalues
import scipy
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from datasets import Dataset

from selected_features import feature_list
from selected_feature_dict import feature_dict
from sklearn.metrics import roc_auc_score
from helpers.plot import plot_multi_hist_sorted, plot_multi_hist, plot_multi_pdf
from datasets import load_dataset
from helpers.process_text import count_tokens_with_tokenizer
from helpers.hf_token import hf_token
from helpers.type import str_or_other
from transformers import AutoTokenizer
from models.gpt2_xl import GPT2Classification
from models.llama import LlamaClassification
from models.pythia import PythiaClassification
from models.bert import BERTClassification
from metrics import raw_values
from utils import prepare_model
import torch.nn.functional as F


def get_args():
    parser = argparse.ArgumentParser(description='Dataset Inference on a language model')
    parser.add_argument("--data_dir", type=str, default="/storage2/bihe/llm_data_detect/")
    parser.add_argument("--dataset_name", type=str, default="pile")
    parser.add_argument("--subset_name", type=str, default="USPTO Backgrounds", help='Specify the subset name for PILE dataset')
    parser.add_argument('--model_name', type=str, default="EleutherAI/pythia-410m-deduped", help='The name of the model to use') # EleutherAI/pythia-410m-deduped
    parser.add_argument('--local_ckpt', type=str_or_other, default=None, help='local lora checkpoint')
    parser.add_argument('--suspect_split', type=str, default="test_9000_2048_random192_llama_pile_test_0_8092_random192_200epoch_16rank_192_0.5prefix_1.2rep_0.6temp_0.9top_1beam_True8bit_paraphrase_original")
    parser.add_argument('--validation_split', type=str, default="test_9000_2048_random192_llama_pile_test_0_8092_random192_200epoch_16rank_192_0.5prefix_1.2rep_0.6temp_0.9top_1beam_True8bit_paraphrase_paraphrase")
    parser.add_argument('--suspect_split_result_name', type=str, default="test_9k_2k_lora_0_8k_random192_200epoch_16rank_0.5pre_original")
    parser.add_argument('--validation_split_result_name', type=str, default="test_9k_2k_lora_0_8k_random192_200epoch_16rank_0.5pre_paraphrase")
    parser.add_argument('--num_samples_baseline', type=int, default=0, help='The number of samples used to train baseline classifier')
    parser.add_argument('--num_samples', type=int, default=1000, help='The number of samples to use')
    parser.add_argument("--normalize", type=str, default="train", help="Should you normalize?", choices=["no", "train", "combined"])
    parser.add_argument("--outliers", type=str, default="mean", help="The ablation to use", choices=["randomize", "keep", "zero", "mean", "clip", "mean+p-value", "p-value"])
    parser.add_argument("--outliers_remove_frac", type=float, default=0.05, help="How many percent of outliers we remove")
    parser.add_argument("--tail_remove_frac", type=float, default=0.05, help="How many percent of outliers we remove from the tail")
    parser.add_argument("--features", type=str, default="selected", help="The features to use")
    parser.add_argument("--false_positive", type=int, default=0, help="What if you gave two val splits?", choices=[0, 1])
    parser.add_argument("--num_random", type=int, default=10, help="How many random runs to do?")
    parser.add_argument("--n_tokens", type=int, default=192)
    parser.add_argument("--short_seq_tolerance", type=float, default=0.95)
    parser.add_argument("--prefix_ratio", type=float, default=0.5)
    parser.add_argument("--reorder_texts", type=int, default=1, help="Do you want to reorder the texts according to the key value and save them?")
    parser.add_argument("--positive_weights", type=str_or_other, default=True, help="only allow positive weight for linear model")
    parser.add_argument("--use_baseline_model", type=str_or_other, default=True, help="train a baseline model and a MIA model, then compare them")
    parser.add_argument("--pair_sus_val", type=str_or_other, default=True, help="There is a pairwise relationship between the suspect and validation sets")
    parser.add_argument('--cache_dir', type=str, default="/storage2/bihe/cache", help='The directory to cache the model')
    parser.add_argument('--baseline_model_type', type=str, default="gpt2", help='which type of baseline model to use')
    parser.add_argument('--train_split', type=str, default="4000_200000")
    parser.add_argument('--sus_split', type=str, default="0_2000")
    parser.add_argument('--val_split', type=str, default="2000_2000")
    parser.add_argument('--doc_idx', type=str, default="0_400")
    parser.add_argument('--max_snippets', type=int, default=10)
    parser.add_argument('--ttest_type', type=str, default="ind")
    parser.add_argument('--linear_activation', type=str, default="sigmoid")
    parser.add_argument('--linear_epochs', type=int, default=200)
    args = parser.parse_args()
    return args


def get_model(num_features, linear = True, positive=False, linear_activation='sigmoid'):
    if linear:
        model = nn.Linear(num_features, 1)
        if positive:
            # use pytorch parameterization to make sure all weights are positives
            import torch.nn.utils.parametrize as parametrize
            class WeightParameterization(nn.Module):
                def forward(self, X):
                    if linear_activation == 'softplus':
                        return nn.functional.softplus(X)
                    elif linear_activation == 'sigmoid':
                        return nn.functional.sigmoid(X)
                    else:
                        raise ValueError(f'Unknown activation function: {linear_activation}')
                    # return nn.functional.softplus(X)
                    # return nn.functional.sigmoid(X)
            # Example registration of this parameterization transform
            parametrize.register_parametrization(model, "weight", WeightParameterization())
            assert torch.all(model.weight>0) # now all > 0
    else:
        model = nn.Sequential(
            nn.Linear(num_features, 10),
            nn.ReLU(),
            nn.Linear(10, 1)  # Single output neuron
        )
    return model


def train_model(inputs, y, num_epochs=10000, positive_weights=False, linear_activation='sigmoid'):
    num_features = inputs.shape[1]
    print(f'num_features: {num_features}')
    model = get_model(num_features, positive=positive_weights, linear_activation=linear_activation)
        
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Convert y to float tensor for BCEWithLogitsLoss
    y_float = y.float()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()  # Squeeze the output to remove singleton dimension
        loss = criterion(outputs, y_float)
        loss.backward()
        optimizer.step()

    return model


def train_baseline_model(texts, labels, eval_texts, eval_labels, num_epochs=10000, data_dir=None, model_type='gpt2', model_name=None):
    # train_dataset = Dataset.from_dict({"text": texts, "extra_features":metrics, "label": labels})
    # decided by args.baseline_model_type
    if model_type.startswith('gpt2'):
        if 'layer' in model_type:
            n_layers = int(re.search(r'\d+', model_type).group())
            classification_trainer = GPT2Classification(
                num_labels=2,
                max_seq_len=64, batch_size=512,
                from_pretrained=False,
                use_lora=False,
                save_model=False,
                n_layers=n_layers)
        elif 'full' in model_type:
            classification_trainer = GPT2Classification(
                num_labels=2,
                max_seq_len=64, batch_size=32,
                from_pretrained=True,
                use_lora=False,
                save_model=False)
        elif 'lora' in model_type:
            classification_trainer = GPT2Classification(
                num_labels=2,
                max_seq_len=64, batch_size=32,
                from_pretrained=True,
                use_lora=True,
                save_model=False)
        else:
            raise ValueError(f'Invalid GPT2-based model type: {model_type}')
    elif model_type == 'llama':
        classification_trainer = LlamaClassification(
            num_labels=2,
            max_seq_len=64, batch_size=16,
            accelerator=None, save_dir=None, 
            save_model=False,
            from_pretrained=True,
            use_lora=True)
    elif model_type == 'bert':
        classification_trainer = BERTClassification(
            num_labels=2,
            max_seq_len=64, batch_size=128,
            accelerator=None, save_dir=None, 
            save_model=False,
            from_pretrained=True,
            use_lora=False)
    elif model_type == 'pythia':
        classification_trainer = PythiaClassification(
            model_name=model_name, 
            num_labels=2,
            max_seq_len=64, batch_size=128,
            accelerator=None, save_dir=None, 
            save_model=False,
            from_pretrained=True,
            use_lora=False)
    else:
        raise ValueError(f'Invalid model type: {model_type}')
    
    probabilities, labels, per_sample_loss, auc = classification_trainer.train_and_evaluate(
        train_dataset=Dataset.from_dict({"text": texts, "labels": labels.long()}),
        eval_dataset=Dataset.from_dict({"text": eval_texts, "labels": eval_labels.long()}),
        output_dir=os.path.join(data_dir, "tmp_results"), epochs=20, fold=0)
    
    return probabilities, labels, per_sample_loss, auc, classification_trainer


def get_predictions(model, val, y):
    with torch.no_grad():
        preds = model(val).detach().squeeze()
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    loss = criterion(preds, y.float())
    return F.sigmoid(preds).numpy(), loss.numpy()


def get_dataset_splits(
        _train_texts, _val_texts,
        _train_metrics, _val_metrics, 
        num_samples, num_samples_baseline):
    # get the train and val sets
    for_baseline_train_texts = _train_texts[:num_samples_baseline]
    for_baseline_val_texts = _val_texts[:num_samples_baseline]
    for_train_train_texts = _train_texts[num_samples_baseline:num_samples]
    for_train_val_texts = _val_texts[num_samples_baseline:num_samples]
    for_val_train_texts = _train_texts[num_samples:]
    for_val_val_texts = _val_texts[num_samples:]

    for_baseline_train_metrics = _train_metrics[:num_samples_baseline]
    for_baseline_val_metrics = _val_metrics[:num_samples_baseline]
    for_train_train_metrics = _train_metrics[num_samples_baseline:num_samples]
    for_train_val_metrics = _val_metrics[num_samples_baseline:num_samples]
    for_val_train_metrics = _train_metrics[num_samples:]
    for_val_val_metrics = _val_metrics[num_samples:]


    # create the train and val sets
    baseline_texts = for_baseline_train_texts+for_baseline_val_texts
    train_texts = for_train_train_texts+for_train_val_texts
    val_texts = for_val_train_texts+for_val_val_texts

    baseline_x = np.concatenate((for_baseline_train_metrics, for_baseline_val_metrics), axis=0)
    baseline_y = np.concatenate((-1*np.zeros(for_baseline_train_metrics.shape[0]), np.ones(for_baseline_val_metrics.shape[0])))
    train_x = np.concatenate((for_train_train_metrics, for_train_val_metrics), axis=0)
    train_y = np.concatenate((-1*np.zeros(for_train_train_metrics.shape[0]), np.ones(for_train_val_metrics.shape[0])))
    val_x = np.concatenate((for_val_train_metrics, for_val_val_metrics), axis=0)
    val_y = np.concatenate((-1*np.zeros(for_val_train_metrics.shape[0]), np.ones(for_val_val_metrics.shape[0])))
    
    # return tensors
    baseline_x = torch.tensor(baseline_x, dtype=torch.float32)
    baseline_y = torch.tensor(baseline_y, dtype=torch.float32)
    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    val_x = torch.tensor(val_x, dtype=torch.float32)
    val_y = torch.tensor(val_y, dtype=torch.float32)
    
    return (baseline_texts, baseline_x, baseline_y), (train_texts, train_x, train_y), (val_texts, val_x, val_y)


def normalize_and_stack(train_metrics, val_metrics, normalize="train"):
    '''
    excpects an input list of list of metrics
    normalize val with corre
    '''
    new_train_metrics = []
    new_val_metrics = []
    for (tm, vm) in zip(train_metrics, val_metrics):
        if normalize == "combined":
            combined_m = np.concatenate((tm, vm))
            mean_tm = np.mean(combined_m)
            std_tm = np.std(combined_m)
        else:
            mean_tm = np.mean(tm)
            std_tm = np.std(tm)
        
        if normalize == "no":
            normalized_vm = vm
            normalized_tm = tm
        else:
            #normalization should be done with respect to the train set statistics
            eps = 1e-20 # avoid zero devision
            normalized_vm = ((vm - mean_tm)+eps) / (std_tm+eps)
            normalized_tm = ((tm - mean_tm)+eps) / (std_tm+eps)
        
        new_train_metrics.append(normalized_tm)
        new_val_metrics.append(normalized_vm)

    train_metrics = np.stack(new_train_metrics, axis=1)
    val_metrics = np.stack(new_val_metrics, axis=1)
    return train_metrics, val_metrics, new_train_metrics, new_val_metrics

def remove_outliers(metrics, remove_frac=0.05, tail_remove_frac=0.05, outliers = "zero"):
    # Sort the array to work with ordered data
    sorted_ids = np.argsort(metrics)
    
    # Calculate the number of elements to remove from each side
    total_elements = len(metrics)
    elements_to_remove_each_side = int(total_elements * remove_frac / 2)
    elements_to_remove_each_side_tail = int(total_elements * tail_remove_frac)
    
    # Ensure we're not attempting to remove more elements than are present
    if elements_to_remove_each_side * 2 > total_elements:
        raise ValueError("remove_frac is too large, resulting in no elements left.")
    
    # Change the removed metrics to 0.
    lowest_ids = sorted_ids[:elements_to_remove_each_side]
    highest_ids = sorted_ids[-(elements_to_remove_each_side+elements_to_remove_each_side_tail):]
    all_ids = np.concatenate((lowest_ids, highest_ids))

    # import pdb; pdb.set_trace()
    
    trimmed_metrics = np.copy(metrics)
    
    if outliers == "zero":
        trimmed_metrics[all_ids] = 0
    elif outliers == "mean" or outliers == "mean+p-value":
        trimmed_metrics[all_ids] = np.mean(trimmed_metrics)
    elif outliers == "clip":
        highest_val_permissible = trimmed_metrics[highest_ids[0]]
        lowest_val_permissible = trimmed_metrics[lowest_ids[-1]]
        trimmed_metrics[highest_ids] =  highest_val_permissible
        trimmed_metrics[lowest_ids] =   lowest_val_permissible
    elif outliers == "randomize":
        #this will randomize the order of metrics
        raise Exception(f'the pairwise correlation would be broken for randomize')
        trimmed_metrics = np.delete(trimmed_metrics, all_ids)
    else:
        assert outliers in ["keep", "p-value"]
        pass
    
    return trimmed_metrics
    


def get_p_value_list(heldout_train, heldout_val, p_sample_list, ttest_type='ind'):
    p_value_list = []
    for num_samples in p_sample_list:
        heldout_train_curr = heldout_train[:num_samples]
        heldout_val_curr = heldout_val[:num_samples]
        if ttest_type == 'ind':
            t, p_value = ttest_ind(heldout_train_curr, heldout_val_curr, alternative='less')
        elif ttest_type == 'rel':
            t, p_value = ttest_rel(heldout_train_curr, heldout_val_curr, alternative='less')
        else:
            raise Exception(f'Unknown ttest type: {ttest_type}')
        p_value_list.append(p_value)
    return p_value_list
    
    

def split_train_val(metrics):
    keys = list(metrics.keys())
    num_elements = len(metrics[keys[0]])
    print (f"Using {num_elements} elements")
    # select a random subset of val_metrics (50% of ids)
    ids_train = np.random.choice(num_elements, num_elements//2, replace=False)
    ids_val = np.array([i for i in range(num_elements) if i not in ids_train])
    new_metrics_train = {}
    new_metrics_val = {}
    for key in keys:
        new_metrics_train[key] = np.array(metrics[key])[ids_train]
        new_metrics_val[key] = np.array(metrics[key])[ids_val]
    return new_metrics_train, new_metrics_val

def main():
    args = get_args()

    if args.dataset_name == 'pile' or args.dataset_name.startswith('dolma'):
        dataset_dir_name = f'{args.dataset_name}_{args.subset_name}'
    else:
        dataset_dir_name = f'{args.dataset_name}'

    if args.local_ckpt is not None:
        save_model_name = args.local_ckpt
    else:
        save_model_name = args.model_name

    suspect_file_path = os.path.join(args.data_dir, f"new_results/{save_model_name}/{dataset_dir_name}/{args.suspect_split}_metrics.json")
    validation_file_path = os.path.join(args.data_dir, f"new_results/{save_model_name}/{dataset_dir_name}/{args.validation_split}_metrics.json")

    if args.reorder_texts:
        suspect_column_name = re.sub(r"\d+", "", args.suspect_split.split('_')[-1]) # remove any possible numbers
        validation_column_name = re.sub(r"\d+", "", args.validation_split.split('_')[-1])
        suspect_file_name = '_'.join(args.suspect_split.split('_')[:-1])
        validation_file_name = '_'.join(args.validation_split.split('_')[:-1])
        # re.sub(r"_original.*\.jsonl", ".jsonl", filename)
        suspect_text_file_path = os.path.join(args.data_dir, f"datasets/{dataset_dir_name}/{suspect_file_name}.jsonl")
        validation_text_file_path = os.path.join(args.data_dir, f"datasets/{dataset_dir_name}/{validation_file_name}.jsonl")

        text_key_dir = os.path.join(args.data_dir, f"datasets_key/{save_model_name}/{dataset_dir_name}/")
        os.makedirs(text_key_dir, exist_ok=True)

    with open(suspect_file_path, 'r') as f:
        metrics_train = json.load(f)
    with open(validation_file_path, 'r') as f:
        metrics_val = json.load(f)

    if args.false_positive:
        metrics_train, metrics_val = split_train_val(metrics_val)

    keys = list(metrics_train.keys())
    train_metrics = []
    val_metrics = []

    # record which indexes are chosen
    # root = os.path.join(args.data_dir, "datasets")
    # val_pair_dir = os.path.join(root, f'{dataset_dir_name}')
    # if args.prefix:
    #     updated_file_name = os.path.join(val_pair_dir, f'{args.split}_{args.dataset_col}_prefix_idx.jsonl')
    # else:
    #     updated_file_name = os.path.join(val_pair_dir, f'{args.split}_{args.dataset_col}_idx.jsonl')
    # with open(updated_file_name, "w") as f:
    #     json_line = {}
    #     json_line["idx_paraphrase"] = paraphrase_idx_list
    #     f.write(json.dumps(json_line) + "\n")

    print(args.suspect_split)
    print(args.validation_split)
    print(suspect_text_file_path)
    print(validation_text_file_path)

    suspect_text_dataset = load_dataset("json", data_files=suspect_text_file_path, split='train')[suspect_column_name]
    if suspect_column_name == 'paraphrase':
        suspect_text_dataset = [data[0] for data in suspect_text_dataset]
    validation_text_dataset = load_dataset("json", data_files=validation_text_file_path, split='train')[validation_column_name]
    if validation_column_name == 'paraphrase':
        validation_text_dataset = [data[0] for data in validation_text_dataset]

    print(suspect_text_dataset[0:10])
    print(validation_text_dataset[0:10])

    # get output of the target model
    # model_name =  args.model_name
    # target_model, target_tokenizer = prepare_model(model_name, data_dir=args.data_dir, local_ckpt=args.local_ckpt, cache_dir=args.cache_dir)
    # suspect_loss_list = raw_values(target_model, target_tokenizer, suspect_text_dataset, batch_size = 128)
    # val_loss_list = raw_values(target_model, target_tokenizer, validation_text_dataset, batch_size = 128)
    

    for key in keys:
        if args.features == "selected":
            if key not in feature_list:
                continue
        elif args.features != "all":
            if key not in feature_dict[args.features]:
                continue
        print(key)
        metrics_train_key = metrics_train[key]
        metrics_val_key = metrics_val[key]

        
        # remove short sequences and save texts according to key value (from high to low)
        if args.reorder_texts:
            # for remove short sequences
            # Load tokenizer
            model_name = "meta-llama/Meta-Llama-3-8B"  # Replace with your chosen model
            tokenizer = AutoTokenizer.from_pretrained(model_name,
                token=hf_token,
                # padding_side = 'left',
            )
            # add padding tokens for llama model
            tokenizer.add_special_tokens({"pad_token":"<pad>"})
            def short_seq_id(texts, keys, tokenizer):
                assert len(texts) == len(keys), f'size mismatch: len(texts)=={len(texts)}, len(keys)=={len(keys)}'
                rm_idx = []
                for idx, text in enumerate(texts):
                    # print(type(text))
                    seq_len = count_tokens_with_tokenizer(text, tokenizer)
                    n_tokens = int(args.n_tokens*args.short_seq_tolerance)
                    if seq_len < int(n_tokens-n_tokens*args.prefix_ratio):
                        rm_idx.append(idx)
                return rm_idx

            def rm_short_seq(texts, keys, rm_idx):
                
                texts = [texts[i] for i in range(len(texts)) if i not in rm_idx]
                keys = [keys[i] for i in range(len(keys)) if i not in rm_idx]
                return texts, keys

            def sort_by_key(texts, keys):
                sorted_combined = sorted(list(zip(texts, keys)), key=lambda x: x[1], reverse=True)

                # Unzip the sorted list back into two separate lists if needed
                sorted_texts, sorted_keys = zip(*sorted_combined)

                # Convert back to lists if necessary
                sorted_texts = list(sorted_texts)
                sorted_keys = list(sorted_keys)
                return sorted_texts, sorted_keys

            
            print(f'suspect set size before removing short sequence:{len(suspect_text_dataset), len(metrics_train_key)}')
            # suspect_text_dataset, metrics_train_key = rm_short_seq(suspect_text_dataset, metrics_train_key, tokenizer)
            print(f'suspect set size after removing short sequence:{len(suspect_text_dataset), len(metrics_train_key)}')
            suspect_text_dataset_sorted, suspect_keys = sort_by_key(suspect_text_dataset, metrics_train_key)
            suspect_text_key_file_path = os.path.join(text_key_dir, f"{args.suspect_split}_{key}.jsonl")
            with open(suspect_text_key_file_path, "w") as f:
                for i in range(len(suspect_text_dataset_sorted)):
                    json_line = {}
                    json_line[suspect_column_name] = suspect_text_dataset_sorted[i]
                    json_line[key] = suspect_keys[i]
                    f.write(json.dumps(json_line) + "\n")

            print(f'validation set size before removing short sequence:{len(validation_text_dataset), len(metrics_val_key)}')
            # validation_text_dataset, metrics_val_key = rm_short_seq(validation_text_dataset, metrics_val_key, tokenizer)
            print(f'validation set size after removing short sequence:{len(validation_text_dataset), len(metrics_val_key)}')
            validation_text_dataset_sorted, validation_keys = sort_by_key(validation_text_dataset, metrics_val_key)
            validation_text_key_file_path = os.path.join(text_key_dir, f"{args.validation_split}_{key}.jsonl")
            with open(validation_text_key_file_path, "w") as f:
                for i in range(len(validation_text_dataset_sorted)):
                    json_line = {}
                    json_line[validation_column_name] = validation_text_dataset_sorted[i]
                    json_line[key] = validation_keys[i]
                    f.write(json.dumps(json_line) + "\n")

        

        # convert to numpy
        # print(metrics_train_key)
        # print(metrics_val_key)
        # print(f'after read: {suspect_text_dataset[0], metrics_train_key[0], suspect_text_dataset[1200], metrics_train_key[1200], validation_text_dataset[0], metrics_val_key[0], validation_text_dataset[1200], metrics_val_key[1200]}')
        metrics_train_key = np.array(metrics_train_key)
        metrics_val_key = np.array(metrics_val_key)
        # remove the top 2.5% and bottom 2.5% of the data
        
        metrics_train_key = remove_outliers(metrics_train_key, remove_frac = args.outliers_remove_frac, tail_remove_frac = args.tail_remove_frac, outliers = args.outliers)
        metrics_val_key = remove_outliers(metrics_val_key, remove_frac = args.outliers_remove_frac, tail_remove_frac = args.tail_remove_frac, outliers = args.outliers)

        print(f'after remove_outliers: {suspect_text_dataset[0], metrics_train_key[0], suspect_text_dataset[1200], metrics_train_key[1200], validation_text_dataset[0], metrics_val_key[0], validation_text_dataset[1200], metrics_val_key[1200]}')

        train_metrics.append(metrics_train_key)
        val_metrics.append(metrics_val_key)

        print(len(metrics_train_key))
        print(len(metrics_val_key))

        # plot probability distribution
        # if key in ['ppl', 'k_min_probs_0.1', 'zlib_ratio'] or 'k_max_probs' in key:
        figure_dir = os.path.join(args.data_dir, f'figures/{args.model_name}/{args.local_ckpt}/di_{key}/outlier_{args.outliers_remove_frac}/tail_{args.tail_remove_frac}/{dataset_dir_name}')
        os.makedirs(figure_dir, exist_ok=True)
        plot_multi_pdf(data_list=[metrics_train_key, metrics_val_key], 
                        label_list=['suspect(real)', 'validation(generated)'], 
                        title=f'{key}[{args.suspect_split_result_name}]-[{args.validation_split_result_name}]', 
                        xlabel=key, ylabel='probability', 
                        save_dir=figure_dir)
        print(f'* Plotted {key} probability distribution function.')



    # concatenate the train and val metrics by stacking them
    
    # train_metrics, val_metrics = new_train_metrics, new_val_metrics
    def append_loss_list(metric, dataset_loss_list, sorted=True):
        assert len(dataset_loss_list) == len(metric[0])
        for i in range(len(dataset_loss_list)):
            loss_list = dataset_loss_list[i]
            # truncate longer list
            loss_list = loss_list[:args.n_tokens//2]
            # extend short list
            for _ in range(args.n_tokens//2-len(loss_list)):
                loss_list.append(sum(loss_list)/len(loss_list))
            if sorted:
                loss_list.sort()
            dataset_loss_list[i] = loss_list
        dataset_loss_list = list(map(list, zip(*dataset_loss_list))) # transpose dataset_loss_list
        for dataset_losses in dataset_loss_list:
            dataset_losses = np.array(dataset_losses)
            dataset_losses = remove_outliers(dataset_losses, remove_frac = args.outliers_remove_frac, tail_remove_frac = args.tail_remove_frac, outliers = args.outliers)
            metric.append(dataset_losses)
        return metric
    # train_metrics = append_loss_list(train_metrics, suspect_loss_list)
    # val_metrics = append_loss_list(val_metrics, val_loss_list)

    print(f'len(unnormalized_train_metrics)={len(train_metrics)}')
    unnormalized_metric_mean_gap = []
    for i in range(len(train_metrics)):
        unnormalized_metrics_train_key = train_metrics[i]
        unnormalized_metrics_val_key = val_metrics[i]
        mean_metrics_train_key = np.mean(unnormalized_metrics_train_key)
        mean_metrics_val_key = np.mean(unnormalized_metrics_val_key)
        unnormalized_metric_mean_gap.append(mean_metrics_val_key - mean_metrics_train_key)

        
    print(train_metrics, val_metrics)
    train_metrics, val_metrics, unstacked_train_metrics, unstacked_val_metrics = normalize_and_stack(train_metrics, val_metrics, normalize=args.normalize)

    print(f'len(unstacked_train_metrics)={len(unstacked_train_metrics)}')
    metric_mean_gap = []
    for i in range(len(unstacked_train_metrics)):
        metrics_train_key = unstacked_train_metrics[i]
        metrics_val_key = unstacked_val_metrics[i]
        mean_metrics_train_key = np.mean(metrics_train_key)
        mean_metrics_val_key = np.mean(metrics_val_key)
        metric_mean_gap.append(mean_metrics_val_key - mean_metrics_train_key)

    
    p_sample_list = [2, 5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    p_file_dir = os.path.join(
        args.data_dir, 
        f"aggregated_results/p_values/{args.outliers}-outliers/{args.dataset_name}/{args.model_name}_{args.local_ckpt}/cls-{args.baseline_model_type}/pos-{args.positive_weights}/outlier_{args.outliers_remove_frac}/tail_{args.tail_remove_frac}/ttest-{args.ttest_type}/train-{args.train_split}_sus-{args.sus_split}_val-{args.val_split}_{args.num_samples}/linear_{args.linear_epochs}e_{args.linear_activation}/"
    )
    os.makedirs(p_file_dir, exist_ok=True)
    if 'train' in args.suspect_split:
        membership = 'member'
    elif 'val'  in args.suspect_split or 'test' in args.suspect_split or 'val+test' in args.suspect_split:
        membership = 'non-mem'
    else:
        membership = 'unknown'

    # save the gaps between (unnormalized) real and generated data for different MIA metrics
    p_file_metrics_gap = os.path.join(p_file_dir, f"{args.features}_{p_sample_list[-1]}samples_{args.num_random}runs_unnormalized_metrics_gaps.csv")
    feature_names = feature_dict[args.features] if args.features != "all" else keys
    # write column names
    if not os.path.exists(p_file_metrics_gap):
        with open(p_file_metrics_gap, 'w') as f:
            to_write = ",".join(
                [
                    "Subset", "Membership", "doc_idx", "Max. Snippets", "n_tokens", 
                    # "AUC_Text(%)", "AUC_onlyMIA(%)", "AUC_Comb(%)" 
                ]+feature_names
            ) + "\n"
            f.write(to_write)
    # write values
    with open(p_file_metrics_gap, 'a') as f:
        to_write = ",".join(
            [
                args.subset_name, membership, args.doc_idx, str(args.max_snippets), str(args.n_tokens), 
                # str(avg_auc_baseline), str(avg_auc_only_mia), str(avg_auc_mia)
            ]+[
                str(gap) for gap in unnormalized_metric_mean_gap
            ]
        ) + "\n"
        f.write(to_write)

    # save the gaps between real and generated data for different MIA metrics
    p_file_metrics_gap = os.path.join(p_file_dir, f"{args.features}_{p_sample_list[-1]}samples_{args.num_random}runs_metrics_gaps.csv")
    feature_names = feature_dict[args.features] if args.features != "all" else keys
    # write column names
    if not os.path.exists(p_file_metrics_gap):
        with open(p_file_metrics_gap, 'w') as f:
            to_write = ",".join(
                [
                    "Subset", "Membership", "doc_idx", "Max. Snippets", "n_tokens", 
                    # "AUC_Text(%)", "AUC_onlyMIA(%)", "AUC_Comb(%)" 
                ]+feature_names
            ) + "\n"
            f.write(to_write)
    # write values
    with open(p_file_metrics_gap, 'a') as f:
        to_write = ",".join(
            [
                args.subset_name, membership, args.doc_idx, str(args.max_snippets), str(args.n_tokens), 
                # str(avg_auc_baseline), str(avg_auc_only_mia), str(avg_auc_mia)
            ]+[
                str(gap) for gap in metric_mean_gap
            ]
        ) + "\n"
        f.write(to_write)

    
    all_p_value_list = []
    diff_all_p_value_list = []
    ratio_all_p_value_list = []
    loss_all_p_value_list = []
    no_pair_all_p_value_list = []
    auc_score_list = []
    baseline_auc_score_list = []
    only_mia_auc_score_list = []
    mia_auc_score_list = []
    print(suspect_text_dataset)
    print(validation_text_dataset)
    for i in range(args.num_random):
        
        if args.pair_sus_val:
            paired = list(zip(suspect_text_dataset, train_metrics, validation_text_dataset, val_metrics))
            np.random.shuffle(paired)
            suspect_text_dataset, train_metrics, validation_text_dataset, val_metrics = zip(*paired)
        else:
            train_paired = list(zip(suspect_text_dataset, train_metrics))
            val_paired = list(zip(validation_text_dataset, val_metrics))
            np.random.shuffle(train_paired)
            np.random.shuffle(val_paired)
            suspect_text_dataset, train_metrics = zip(*train_paired)
            validation_text_dataset, val_metrics = zip(*val_paired)

        train_metrics = np.array(train_metrics)
        val_metrics = np.array(val_metrics)
        (baseline_texts, baseline_x, baseline_y), (train_texts, train_x, train_y), (val_texts, val_x, val_y) = get_dataset_splits(
            suspect_text_dataset, validation_text_dataset,
            train_metrics, val_metrics, args.num_samples, args.num_samples_baseline
            )
        

        # sanity check
        assert len(baseline_texts) == len(baseline_x) and len(baseline_x) == len(baseline_y)
        assert len(train_texts) == len(train_x) and len(train_x) == len(train_y)
        assert len(val_texts) == len(val_x) and len(val_x) == len(val_y)

        print(f'baseline length after split: {len(baseline_texts), len(baseline_x), len(baseline_y)}')
        print(f'train length after split: {len(train_texts), len(train_x), len(train_y)}')
        print(f'val length after split: {len(val_texts), len(val_x), len(val_y)}')
        
        if args.use_baseline_model:

            # Define the path for saving logits
            logits_dir = os.path.join(
            args.data_dir, 
            # f"classifier_logits/{args.outliers}-outliers/{args.dataset_name}_{args.subset_name}/{args.model_name}_{args.local_ckpt}/cls-{args.baseline_model_type}/outlier_{args.outliers_remove_frac}/tail_{args.tail_remove_frac}/{args.suspect_split}/train-{args.train_split}_sus-{args.sus_split}_val-{args.val_split}_{args.num_samples}/random_{i}/"
            f"classifier_logits/{args.dataset_name}_{args.subset_name}/{args.model_name}_{args.local_ckpt}/cls-{args.baseline_model_type}/{args.suspect_split}/train-{args.train_split}_sus-{args.sus_split}_val-{args.val_split}_{args.num_samples}/random_{i}/"
        )
            baseline_logits_path = os.path.join(logits_dir, "baseline_logits.pt")
            baseline_val_logits_path = os.path.join(logits_dir, "baseline_val_logits.pt")

            # Check if the logits already exist
            if os.path.exists(baseline_logits_path) and os.path.exists(baseline_val_logits_path):
                # Load the logits directly
                print(f"Loading saved logits from {logits_dir}")
                baseline_logits = torch.load(baseline_logits_path)
                baseline_val_logits = torch.load(baseline_val_logits_path)
            else:
                # Train the model and generate logits
                print("Training baseline model...")
                baseline_logits, labels, baseline_losses, baseline_auc, baseline_trainer = train_baseline_model(
                            train_texts, train_y, 
                            train_texts, train_y,
                            data_dir=args.data_dir,
                            model_type=args.baseline_model_type,
                            model_name=args.model_name,)
                print(baseline_logits.shape)
                baseline_val_logits, _, _, _ = baseline_trainer.evaluate_model(Dataset.from_dict({"text": val_texts, "labels": val_y.long()}).map(baseline_trainer.tokenize_function, batched=True))
                baseline_logits = F.softmax(baseline_logits, dim=-1)[:,1].unsqueeze(1)
                baseline_val_logits = F.softmax(baseline_val_logits, dim=-1)[:,1].unsqueeze(1)
                
                # Save the logits for future use
                os.makedirs(logits_dir, exist_ok=True)
                torch.save(baseline_logits, baseline_logits_path)
                torch.save(baseline_val_logits, baseline_val_logits_path)
                print(f"Saved baseline logits to {logits_dir}")


            # construct the features
            baseline_train_features = baseline_logits
            mia_train_features = torch.cat([baseline_logits, train_x], dim=-1)
            baseline_val_features = baseline_val_logits
            mia_val_features = torch.cat([baseline_val_logits, val_x], dim=-1)

            linear_epochs = args.linear_epochs
            baseline_linear_model = train_model(baseline_train_features, train_y, num_epochs = linear_epochs, positive_weights=args.positive_weights, linear_activation=args.linear_activation)
            print(f'baseline_val_features.size:{baseline_val_features.size}')
            # baseline_preds = baseline_val_features[:,1].unsqueeze(1)
            # baseline_preds = baseline_val_features
            baseline_preds, baseline_loss = get_predictions(baseline_linear_model, baseline_val_features, val_y)
            only_mia_linear_model = train_model(train_x, train_y, num_epochs = linear_epochs, positive_weights=args.positive_weights, linear_activation=args.linear_activation)
            only_mia_preds, only_mia_loss = get_predictions(only_mia_linear_model, val_x, val_y)
            mia_linear_model = train_model(mia_train_features, train_y, num_epochs = linear_epochs, positive_weights=args.positive_weights, linear_activation=args.linear_activation)
            mia_preds, mia_loss = get_predictions(mia_linear_model, mia_val_features, val_y)
            print(f'preds: {baseline_preds, mia_preds}')

            baseline_auc_score = roc_auc_score(val_y, baseline_preds)
            only_mia_auc_score = roc_auc_score(val_y, only_mia_preds)
            mia_auc_score = roc_auc_score(val_y, mia_preds)
            baseline_heldout_suspect = baseline_preds[val_y == 0]
            baseline_heldout_val = baseline_preds[val_y == 1]
            only_mia_heldout_suspect = only_mia_preds[val_y == 0]
            only_mia_heldout_val = only_mia_preds[val_y == 1]
            mia_heldout_suspect = mia_preds[val_y == 0]
            mia_heldout_val = mia_preds[val_y == 1]
            # sanity check
            assert len(mia_heldout_suspect) == len(baseline_heldout_suspect)
            assert len(mia_heldout_val) == len(baseline_heldout_val)
            if len(baseline_heldout_val) != len(baseline_heldout_suspect):
                print(f'Warning: suspect ({len(baseline_heldout_suspect)}) and validation ({len(baseline_heldout_val)}) are not of the same lengths. Truncating...')
                baseline_heldout_suspect = baseline_heldout_suspect[:min(len(baseline_heldout_val), len(baseline_heldout_suspect))]
                baseline_heldout_val = baseline_heldout_val[:min(len(baseline_heldout_val), len(baseline_heldout_suspect))]
                # check
                only_mia_heldout_suspect = only_mia_heldout_suspect[:min(len(only_mia_heldout_val), len(only_mia_heldout_suspect))]
                only_mia_heldout_val = only_mia_heldout_val[:min(len(only_mia_heldout_val), len(only_mia_heldout_suspect))]
                mia_heldout_suspect = mia_heldout_suspect[:min(len(mia_heldout_val), len(mia_heldout_suspect))]
                mia_heldout_val = mia_heldout_val[:min(len(mia_heldout_val), len(mia_heldout_suspect))]
            # different comparison metrics
            for i in range((len(baseline_heldout_suspect)-1000) // 500):
                p_sample_list.append(1000+500*(i+1))
            # diff comparison
            baseline_diffs = baseline_heldout_val - baseline_heldout_suspect
            mia_diffs = mia_heldout_val - mia_heldout_suspect
            diff_p_value_list = get_p_value_list(baseline_diffs, mia_diffs, p_sample_list, ttest_type=args.ttest_type)
            print(f'classifier diff p-value: {diff_p_value_list}')
            diff_all_p_value_list.append(diff_p_value_list)
            # ratio comparison
            baseline_ratios = baseline_heldout_val / baseline_heldout_suspect
            mia_ratios = mia_heldout_val / mia_heldout_suspect
            ratio_p_value_list = get_p_value_list(baseline_ratios, mia_ratios, p_sample_list, ttest_type=args.ttest_type)
            print(f'classifier ratio p-value: {ratio_p_value_list}')
            ratio_all_p_value_list.append(ratio_p_value_list)
            # loss comparison
            loss_p_value_list = get_p_value_list(mia_loss, baseline_loss, p_sample_list, ttest_type=args.ttest_type)
            print(f'loss p-value: {loss_p_value_list}')
            loss_all_p_value_list.append(loss_p_value_list)
            # no pairwise comparison
            baseline_heldout_all = np.concatenate((1-baseline_heldout_val, baseline_heldout_suspect))
            mia_heldout_all = np.concatenate((1-mia_heldout_val, mia_heldout_suspect))
            no_pair_p_value_list = get_p_value_list(baseline_heldout_all, mia_heldout_all, p_sample_list, ttest_type=args.ttest_type)
            no_pair_all_p_value_list.append(no_pair_p_value_list)
            print(f'no pair p-value: {loss_p_value_list}')


            baseline_auc_score_list.append(baseline_auc_score)
            only_mia_auc_score_list.append(only_mia_auc_score)
            mia_auc_score_list.append(mia_auc_score)
            print(f'baseline auc: {baseline_auc_score}')
            print(f'only mia auc: {only_mia_auc_score}')
            print(f'mia auc: {mia_auc_score}')
            
        else: 
            model = train_model(train_x, train_y, num_epochs = 1000, positive_weights= args.positive_weights, linear_activation=args.linear_activation)
            preds, loss = get_predictions(model, val_x, val_y)
            preds_train, loss_train = get_predictions(model, train_x, train_y)
            og_train = preds_train[train_y == 0]
            og_val = preds_train[train_y == 1]

            heldout_train = preds[val_y == 0]
            heldout_val = preds[val_y == 1]
            # alternate hypothesis: heldout_train < heldout_val

            auc_score = roc_auc_score(val_y, preds)
            auc_score_list.append(auc_score)
            
            if args.outliers == "p-value" or args.outliers == "mean+p-value":
                heldout_train = remove_outliers(heldout_train, remove_frac = args.outliers_remove_frac, tail_remove_frac = args.tail_remove_frac, outliers = "randomize")
                heldout_val = remove_outliers(heldout_val, remove_frac = args.outliers_remove_frac, tail_remove_frac = args.tail_remove_frac, outliers = "randomize")

            p_value_list = get_p_value_list(heldout_train, heldout_val, ttest_type=args.ttest_type)
            all_p_value_list.append(p_value_list)

    avg_auc_baseline = np.mean(baseline_auc_score_list)*100
    avg_auc_only_mia = np.mean(only_mia_auc_score_list)*100
    avg_auc_mia = np.mean(mia_auc_score_list)*100
    print(f'avg. baseline AUC score: {avg_auc_baseline}%')
    print(f'avg. only AUC score: {avg_auc_only_mia}%')
    print(f'avg. mia AUC score: {avg_auc_mia}%')

    def aggregate_p_value(all_p_value_list):
        all_p_value_list_by_n_sample = zip(*all_p_value_list)
        aggregated_p_value_list = []
        for p_value_list in all_p_value_list_by_n_sample:
            # print(p_value_list)
            aggregated_p_value = 1-np.exp(np.sum(np.log(1-np.array(p_value_list, dtype=np.longdouble), dtype=np.longdouble)), dtype=np.longdouble) # no dividing by n
            aggregated_p_value_list.append(aggregated_p_value)
        return aggregated_p_value_list
    
    p_sample_list_auc = [2, 5, 10, 20]
    auc_p_value_list = get_p_value_list(baseline_auc_score_list, mia_auc_score_list, p_sample_list_auc, ttest_type=args.ttest_type)
    diff_aggregated_p_value_list = aggregate_p_value(diff_all_p_value_list)
    ratio_aggregated_p_value_list = aggregate_p_value(ratio_all_p_value_list)
    no_pair_aggregated_p_value_list = aggregate_p_value(no_pair_all_p_value_list)
    print(f'AUC p-value: {auc_p_value_list}')
    print(f'aggregated classifier diff p-value: {diff_aggregated_p_value_list}')
    print(f'aggregated classifier ratio p-value: {ratio_aggregated_p_value_list}')
    print(f'aggregated classifier no-pair p-value: {no_pair_aggregated_p_value_list}')

    # save final result
    
    # save ratio p-values with different p-value sample sizes
    p_file_aggr_ratio = os.path.join(p_file_dir, f"{args.features}_{p_sample_list[-1]}samples_{args.num_random}runs_p_ratio.csv")
    # write column names
    if not os.path.exists(p_file_aggr_ratio):
        with open(p_file_aggr_ratio, 'w') as f:
            to_write = ",".join(
                [
                    "Subset", "Membership", "doc_idx", "Max. Snippets", "n_tokens", "AUC_Text(%)", "AUC_onlyMIA(%)", "AUC_Comb(%)"
                ]+[
                    f"p_{str(p)}" for p in p_sample_list
                ]
            ) + "\n"
            f.write(to_write)
    # write values
    with open(p_file_aggr_ratio, 'a') as f:
        to_write = ",".join(
            [
                args.subset_name, membership, args.doc_idx, 
                str(args.max_snippets), str(args.n_tokens), str(avg_auc_baseline), str(avg_auc_only_mia), str(avg_auc_mia)
            ]+[
                str(p) for p in ratio_aggregated_p_value_list
            ]
        ) + "\n"
        f.write(to_write)
    
    # save auc p-values with different p-value sample sizes
    p_file_aggr_auc = os.path.join(p_file_dir, f"{args.features}_{p_sample_list[-1]}samples_{args.num_random}runs_p_auc.csv")
    # write column names
    if not os.path.exists(p_file_aggr_auc):
        with open(p_file_aggr_auc, 'w') as f:
            to_write = ",".join(
                [
                    "Subset", "Membership", "doc_idx", "Max. Snippets", "n_tokens", "AUC_Text(%)", "AUC_onlyMIA(%)", "AUC_Comb(%)"
                ]+[
                    f"p_{str(p)}" for p in p_sample_list_auc
                ]
            ) + "\n"
            f.write(to_write)
    # write values
    with open(p_file_aggr_auc, 'a') as f:
        to_write = ",".join(
            [
                args.subset_name, membership, args.doc_idx, 
                str(args.max_snippets), str(args.n_tokens), str(avg_auc_baseline), str(avg_auc_only_mia), str(avg_auc_mia)
            ]+[
                str(p) for p in auc_p_value_list
            ]
        ) + "\n"
        f.write(to_write)
    
    # save diff p-values with different p-value sample sizes
    p_file_aggr_diff = os.path.join(p_file_dir, f"{args.features}_{p_sample_list[-1]}samples_{args.num_random}runs_p_diff.csv")
    # write column names
    if not os.path.exists(p_file_aggr_diff):
        with open(p_file_aggr_diff, 'w') as f:
            to_write = ",".join(
                [
                    "Subset", "Membership", "doc_idx", "Max. Snippets", "n_tokens", "AUC_Text(%)", "AUC_onlyMIA(%)", "AUC_Comb(%)"
                ]+[
                    f"p_{str(p)}" for p in p_sample_list
                ]
            ) + "\n"
            f.write(to_write)
    # write values
    with open(p_file_aggr_diff, 'a') as f:
        to_write = ",".join(
            [
                args.subset_name, membership, args.doc_idx, 
                str(args.max_snippets), str(args.n_tokens), str(avg_auc_baseline), str(avg_auc_only_mia), str(avg_auc_mia)
            ]+[
                str(p) for p in diff_aggregated_p_value_list
            ]
        ) + "\n"
        f.write(to_write)

    # save all metrics to result file
    p_file_aggr_all = os.path.join(p_file_dir, f"{args.features}_{p_sample_list[-1]}samples_{args.num_random}runs_p_all.csv")
    # write column names
    if not os.path.exists(p_file_aggr_all):
        with open(p_file_aggr_all, 'w') as f:
            to_write = ",".join(
                [
                    "Subset", "Membership", "doc_idx", "Max. Snippets", "n_tokens", 
                    "AUC_Text(%)", "AUC_onlyMIA(%)", "AUC_Comb(%)",
                    f"P-value(Ratio)_{p_sample_list[-1]}", f"P-value(AUC)_{p_sample_list_auc[-1]}", f"P-value(Diff)_{p_sample_list[-1]}", 
                ]
            ) + "\n"
            f.write(to_write)
    # write values
    with open(p_file_aggr_all, 'a') as f:
        to_write = ",".join(
            [
                args.subset_name, membership, args.doc_idx, str(args.max_snippets), str(args.n_tokens), 
                str(avg_auc_baseline), str(avg_auc_only_mia), str(avg_auc_mia),
                str(auc_p_value_list[-1]), str(ratio_aggregated_p_value_list[-1]), str(diff_aggregated_p_value_list[-1])
            ]
        ) + "\n"
        f.write(to_write)


if __name__ == "__main__":
    main()