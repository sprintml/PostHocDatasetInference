"""
There were certain inconsitencies in the use of ppl and likelihood in the code
Correct all results to accommodate for the same
"""

import glob
import json
import os
import argparse
import torch
from helpers.type import str_or_other

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, default="/storage2/bihe/llm_data_detect/")
    parser.add_argument("--dataset_name", type=str, default="pile")
    parser.add_argument("--subset_name", type=str, default="Wikipedia (en)", help='Specify the subset name for PILE dataset')
    parser.add_argument('--model_name', type=str, default="EleutherAI/pythia-410m", help='The name of the model to use') # EleutherAI/pythia-410m-deduped
    parser.add_argument('--local_ckpt', type=str_or_other, default=None, help='local lora checkpoint')
    parser.add_argument('--split', type=str, default="test_original_4096_llama_pile_test_original_4096_400epoch_paraphrase_paraphrase", help='The split of the dataset to use')
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    # get all files in "results/EleutherAI/*/*.json"
    if args.local_ckpt is not None:
        save_model_name = args.local_ckpt
    else:
        save_model_name = args.model_name
    if args.dataset_name == 'pile' or args.dataset_name.startswith('dolma'):
        dataset_dir_name = f'{args.dataset_name}_{args.subset_name}'
    else:
        dataset_dir_name = f'{args.dataset_name}'
    old_file_path = os.path.join(args.data_dir, f"results/{save_model_name}/{dataset_dir_name}/{args.split}_metrics.json")
    file_list = glob.glob(old_file_path)

    '''
    dict_keys(['ppl', 'k_min_probs_0.05', 'k_min_probs_0.1', 'k_min_probs_0.2', 'k_min_probs_0.3', 'k_min_probs_0.4', 'k_min_probs_0.5', 'k_min_probs_0.6', 'k_max_probs_0.05', 'k_max_probs_0.1', 'k_max_probs_0.2', 'k_max_probs_0.3', 'k_max_probs_0.4', 'k_max_probs_0.5', 'k_max_probs_0.6', 'zlib_ratio', 'ppl_ratio_synonym_substitution', 'ppl_diff_synonym_substitution', 'ppl_ratio_butter_fingers', 'ppl_diff_butter_fingers', 'ppl_ratio_random_deletion', 'ppl_diff_random_deletion', 'ppl_ratio_change_char_case', 'ppl_diff_change_char_case', 'ppl_ratio_whitespace_perturbation', 'ppl_diff_whitespace_perturbation', 'ppl_ratio_underscore_trick', 'ppl_diff_underscore_trick', 'ref_ppl_ratio_silo', 'ref_ppl_diff_silo', 'ref_ppl_ratio_tinystories-33M', 'ref_ppl_diff_tinystories-33M', 'ref_ppl_ratio_tinystories-1M', 'ref_ppl_diff_tinystories-1M', 'ref_ppl_ratio_phi-1_5', 'ref_ppl_diff_phi-1_5'])
    '''



    # iterate over all files
    for file in file_list:
        with open(file, 'r') as f:
            metrics = json.load(f)
            ppl_list = torch.tensor(metrics['ppl'])
            eps = 1e-6 # avoid zero devision
            loss_list = torch.log(ppl_list+eps)
            keys = list(metrics.keys())
            for key in keys:
                if "ref_ppl_ratio" in key:
                    # pass
                    print(key)
                    current_ratio = torch.tensor(metrics[key]) # loss_list / ref_ppl
                    print(len(metrics[key]))
                    ref_ppl = (loss_list+eps) / (current_ratio+eps)
                    ppl_ratio = (ppl_list+eps) / (ref_ppl+eps)
                    loss_ratio = (torch.log(ref_ppl+eps)+eps) / (loss_list+eps)
                    metrics[key] = ppl_ratio.tolist()
                    metrics[key.replace("ppl", "loss")] = loss_ratio.tolist()
                elif "ref_ppl_diff" in key:
                    # pass
                    current_diff = torch.tensor(metrics[key]) # loss_list - ref_ppl
                    ref_ppl = loss_list - current_diff
                    ppl_diff = ppl_list - ref_ppl
                    loss_diff = torch.log(ref_ppl+eps) - loss_list
                    metrics[key] = ppl_diff.tolist()
                    metrics[key.replace("ppl", "loss")] = loss_diff.tolist()
                elif "ppl_ratio" in key:
                    print(key)
                    current_ratio = torch.tensor(metrics[key])
                    perturbation_loss = (loss_list+eps) / (current_ratio+eps)
                    perturbation_ppl = torch.exp(perturbation_loss)
                    ppl_ratio = (ppl_list+eps) / (perturbation_ppl+eps)
                    loss_ratio = (perturbation_loss+eps) / (loss_list+eps)
                    metrics[key] = ppl_ratio.tolist()
                    metrics[key.replace("ppl", "loss")] = loss_ratio.tolist()
                elif "ppl_diff" in key:
                    current_diff = torch.tensor(metrics[key])
                    perturbation_loss = loss_list - current_diff
                    perturbation_ppl = torch.exp(perturbation_loss)
                    ppl_diff = ppl_list - perturbation_ppl
                    loss_diff = perturbation_loss - loss_list
                    metrics[key] = ppl_diff.tolist()
                    metrics[key.replace("ppl", "loss")] = loss_diff.tolist()
            
            # save the new file at "new_results/EleutherAI/*/*.json"
            new_file = file.replace("results", "new_results")
            print(new_file)
            os.makedirs(os.path.dirname(new_file), exist_ok=True)
            with open(new_file, 'w') as f:
                json.dump(metrics, f)

if __name__ == "__main__":
    main()




