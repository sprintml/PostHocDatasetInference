from utils import prepare_model
from metrics import aggregate_metrics, reference_model_registry
import json, os
import argparse
from datasets import load_dataset
from helpers.type import str_or_other

def get_args():
    parser = argparse.ArgumentParser(description='Dataset Inference on a language model')
    
    parser.add_argument("--data_dir", type=str, default="/storage2/bihe/llm_data_detect/")
    parser.add_argument("--dataset_name", type=str, default="pile")
    parser.add_argument("--subset_name", type=str, default="Wikipedia (en)", help='Specify the subset name for PILE dataset')
    parser.add_argument('--model_name', type=str, default="EleutherAI/pythia-410m", help='The name of the model to use') # EleutherAI/pythia-410m-deduped
    parser.add_argument('--local_ckpt', type=str_or_other, default=None, help='local lora checkpoint for target model')
    parser.add_argument('--ref_ckpt', type=str_or_other, default=None, help='local lora checkpoint for reference model')
    parser.add_argument('--split', type=str, default="test_original_4096_llama_pile_test_original_4096_400epoch_paraphrase_paraphrase", help='The split of the dataset to use')
    # parser.add_argument('--dataset_col', type=str, default="original", choices=['original', 'paraphrase'])
    parser.add_argument('--num_samples', type=int, default=20000, help='The number of samples to use')
    parser.add_argument('--batch_size', type=int, default=64, help='The batch size to use')
    parser.add_argument('--from_hf', type=int, default=1, help='If set, will load the dataset from huggingface')
    parser.add_argument('--cache_dir', type=str, default="/storage3/bihe/cache", help='The directory to cache the model')
    args = parser.parse_args()
    return args



def main():
    args = get_args()
    print(type(args.local_ckpt))
    if args.dataset_name == 'pile' or args.dataset_name.startswith('dolma'):
        dataset_dir_name = f'{args.dataset_name}_{args.subset_name}'
    else:
        dataset_dir_name = f'{args.dataset_name}'
    results_file = os.path.join(args.data_dir, f"results/{args.model_name}/{dataset_dir_name}/{args.split}_metrics.json")
    # if os.path.exists(results_file):
        # print(f"Results file {results_file} already exists. Aborting...")
        # return
    model_name =  args.model_name
    
    # if model_name in ["microsoft/phi-1_5", "EleutherAI/pythia-12b", "EleutherAI/pythia-6.9b", "EleutherAI/pythia-410m"]:
    #     args.cache_dir = "/data/locus/llm_weights/pratyush"

    model, tokenizer = prepare_model(model_name, data_dir=args.data_dir, local_ckpt=args.local_ckpt, cache_dir=args.cache_dir)
    
    # load the data
    dataset_name = args.dataset_name
    split = args.split
    
    if not args.from_hf:
        from dataloader import load_data
        # if you want to load data directly from the PILE, use the following line
        num_samples = args.num_samples
        dataset = load_data(dataset_name, split, num_samples)
    else:
        root = os.path.join(args.data_dir, "datasets")
        val_pair_dir = os.path.join(root, f'{dataset_dir_name}')
        file_name = os.path.join(val_pair_dir, f'{args.split}.jsonl')
        # dataset_path = f"data/{dataset_name}_{split}.jsonl"
        dataset = load_dataset("json", data_files=file_name, split="train")
    print("Data loaded")

    # get the metrics
    if model_name in reference_model_registry.values():
        metric_list = ["ppl"]
    else:
        metric_list = ["k_min_probs", "ppl", "zlib_ratio", "k_max_probs", "perturbation", "reference_model"]

    dataset_col = 'text'
    metrics = aggregate_metrics(model, tokenizer, dataset, dataset_col, metric_list, args, batch_size = args.batch_size, ref_ckpt = args.ref_ckpt, data_dir=args.data_dir)
    
    # save the metrics
    if args.local_ckpt is not None:
        save_model_name = args.local_ckpt
    else:
        save_model_name = args.model_name
    results_file = os.path.join(args.data_dir, f"results/{save_model_name}/{dataset_dir_name}/{args.split}_metrics.json")
    os.makedirs(os.path.join(args.data_dir, f"results/{save_model_name}/{dataset_dir_name}"), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    main()

