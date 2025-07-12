import subprocess
import os
import json
import time

def execute_commands(commands):
    for command in commands:
        try:
            print(f"Executing command: {command}")
            # Use subprocess.run to execute the command
            result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE)
            print("Output:\n", result.stdout)
            print("Error (if any):\n", result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while executing '{command}': {e}")

        

if __name__ == "__main__":
    # Example list of commands
    commands_to_execute = []
    dataset_name = "pile"
    # subset_names = {
    #     # "USPTO Backgrounds": "uspto",
    #     # "Wikipedia (en)": "wiki",
    #     # "PubMed Abstracts": "pubmed_abs",
    #     # "Pile-CC": "pilecc",
    #     # "StackExchange": "stack",
    #     "PhilPapers": "phil"
    #     # "NIH ExPorter": "nih",
    #     # "HackerNews": "hacker"
    # }
    subset_names = [
        # "Pile-CC",
        "Wikipedia (en)",
        # "USPTO Backgrounds",
        # "FreeLaw",
        # "Enron Emails",
        # "EuroParl",
        # "StackExchange",
        # "Ubuntu IRC",
        # "ArXiv",
        # "Github",
        # "HackerNews",
        # "PubMed Abstracts",
        # "PhilPapers",
        # "PubMed Central",
        # "DM Mathematics",
        # "NIH ExPorter",
    ]

    model_names = [
        # "EleutherAI/pythia-70m-deduped",
        # "EleutherAI/pythia-160m-deduped",
        "EleutherAI/pythia-410m-deduped",
        # "EleutherAI/pythia-1b-deduped",
        # "EleutherAI/pythia-1.4b-deduped",
        # "EleutherAI/pythia-2.8b-deduped",
        # "EleutherAI/pythia-6.9b-deduped",
        # "EleutherAI/pythia-12b-deduped",
    ] # pythia-410m-deduped pythia-12b pythia-6.9b pythia-2.8b 1.4b
    local_ckpt = "None"
    batch_size = 16
    positive_weights = True
    num_random = 10
    outliers_remove_frac = 0.2
    tail_remove_frac = 0.0
    baseline_model_type = 'gpt2_2layer' #  gpt2_2layer  gpt2_full  gpt2_lora bert llama
    ttest_type = "ind" # ind rel
    baseline_type = "text" # text loss
    linear_epochs = 200
    linear_activation = "sigmoid" # sigmoid softplus
    n_paraphrases = 1 # 100

    # n_tokens_list = [32]
    # max_snippets_list = [500] # 20 500
    lora_epoch = 100
    lora_rank = 32
    sus_column = 'original' # original prefix
    val_column = 'paraphrase'
    feature_list = ["kmin234_ratio2_diff"] #all_ppl_no_max threshold_0.1_strict threshold_0.1 kmin23_ratio_diff
    # feature_list = ["ppl", "mink", "mink_0.2", "maxk", "zlib", "ppl_based", "ptb_ppl_ratio", "ref_ppl_ratio", "ptb_ppl_diff", "ref_ppl_diff", "ptb_loss_ratio", "ref_loss_ratio", "ptb_loss_diff", "ref_loss_diff", "all_no_diff", "all_custom", "all_ppl_no_max", "all"]
    commands_prefix = "sbatch --job-name=di --partition=all --mem=2G --cpus-per-task=2 --nodelist=sprint3 --gres=gpu:1 " # --nodelist=sprint3 --gres=gpu:1 
    file_configs = {
        "/home/bihe/LLM_data_detect/scripts/pile/run_di_pipeline_split_pile_gen_val_seg_first.sh":
        {
            "log_name": '--output="{log_dir}/{subset_short_name}_{doc_idx}_rand{n_tokens}_original[{sus_split}_{sus_column}]_para[{val_split}_gen_max_{max_snippets}_{lora_epoch}e_{lora_rank}r_{val_column}]_n_samp[{num_samples}]_{num_random}runs.out" ',
            "split_dict": [
                { # member-same
                    "split_name": "train",
                    # "doc_idx": "0_132",
                    "train_split": "4000_200000",
                    "sus_split": "0_2000",
                    "val_split": "0_2000",
                    "ref_split": "0_2000",
                    "num_samples": 1000},
                { # member-same
                    "split_name": "val+test",
                    # "doc_idx": "0_132",
                    "train_split": "4000_200000",
                    "sus_split": "0_2000",
                    "val_split": "0_2000",
                    "ref_split": "0_2000",
                    "num_samples": 1000},
                # { # member-same
                #     "split_name": "train",
                #     "train_split": "10000_200000",
                #     "sus_split": "0_10000",
                #     "val_split": "0_10000",
                #     "num_samples": 1000},
                # { # member-same
                #     "split_name": "val+test",
                #     "train_split": "10000_200000",
                #     "sus_split": "0_10000",
                #     "val_split": "0_10000",
                #     "num_samples": 1000},
                ]
        }
    }

    with open('/home/bihe/LLM_data_detect/subset_config.json', 'r') as file:
        config_dict = json.load(file)

    for model_name in model_names:
        for subset_name in subset_names:
            # subset_short_name = subset_names[subset_name]
            # doc_idx = splits["doc_idx"]
            subset_short_name = config_dict[subset_name]["short_name"]
            gen_configs = config_dict[subset_name]["gen_configs"]
            for gen_config in gen_configs:
                base_doc_idx = gen_config["doc_idx"]
                max_snippets = gen_config["max_snippets"]
                n_tokens_list = gen_config["n_tokens_list"]
                for n_tokens in n_tokens_list:
                    log_dir = f'/home/bihe/LLM_data_detect/logs/di_pipeline/pile/{model_name}_{local_ckpt}/seg_first/basetype-{baseline_type}/cls_{baseline_model_type}/outlier_{outliers_remove_frac}/tail_{tail_remove_frac}/{n_tokens}/{subset_name}/pos_{positive_weights}/ttest_{ttest_type}/linear_{linear_epochs}e_{linear_activation}'
                    os.makedirs(log_dir,exist_ok=True)
                    # for max_snippets in max_snippets_list:
                    for file_name in file_configs:
                        for splits in file_configs[file_name]["split_dict"]:
                            split_name = splits['split_name']
                            doc_idx = f'{split_name}_{base_doc_idx}'
                            # pass the variables
                            variable_string = ''
                            variable_string += f'--export=ALL,features={"\""+"/".join(feature_list)+"\""},n_tokens={n_tokens},max_snippets={max_snippets}'
                            variable_string += f',dataset_name={dataset_name},subset_name="{subset_name}"'
                            variable_string += f',model_name={model_name},local_ckpt={local_ckpt}'
                            variable_string += f',batch_size={batch_size}'
                            variable_string += f',positive_weights={positive_weights}'
                            variable_string += f',doc_idx={doc_idx}'
                            variable_string += f',num_random={num_random}'
                            variable_string += f',ttest_type={ttest_type}'
                            variable_string += f',baseline_type={baseline_type}'
                            variable_string += f',ref_split={splits["ref_split"]}'
                            variable_string += f',outliers_remove_frac={outliers_remove_frac}'
                            variable_string += f',tail_remove_frac={tail_remove_frac}'
                            variable_string += f',n_paraphrases={n_paraphrases}'
                            variable_string += f',linear_activation={linear_activation}'
                            variable_string += f',linear_epochs={linear_epochs}'
                            variable_string += f',baseline_model_type={baseline_model_type}'
                            variable_string += f',train_split={splits["train_split"]},sus_split={splits["sus_split"]},val_split={splits["val_split"]}'
                            variable_string += f',num_samples={splits["num_samples"]}'
                            variable_string += f',lora_epoch={lora_epoch},lora_rank={lora_rank}'
                            variable_string += f',sus_column={sus_column},val_column={val_column}'
                            variable_string += ' '
                            output_file = file_configs[file_name]["log_name"].format(
                                n_tokens=n_tokens,max_snippets=max_snippets,
                                log_dir=log_dir,
                                subset_short_name=subset_short_name,
                                doc_idx=doc_idx,
                                train_split=splits["train_split"],sus_split=splits["sus_split"],val_split=splits["val_split"],
                                num_samples=splits["num_samples"],
                                num_random=num_random,
                                lora_epoch=lora_epoch,lora_rank=lora_rank,
                                sus_column=sus_column,val_column=val_column)
                            command = commands_prefix+variable_string+output_file+file_name
                            commands_to_execute.append(command)


    execute_commands(commands_to_execute)