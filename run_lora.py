import os
import json
import random

from accelerate import Accelerator
import torch
# from torch.distributed.elastic.multiprocessing.errors import record
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import TrainerCallback, AutoConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset, concatenate_datasets

from helpers.plot import plot_multiple_line_charts
from helpers.hf_token import hf_token


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable (%): {100 * trainable_params / all_param}"
    )

class CustomTrainer(Trainer):
    def new_method(self):
        pass

# @record
def main(args):
    # load datasets
    if args.dataset_name == 'pile':
        dataset_dir_name = f'{args.dataset_name}_{args.subset_name}'
    else:
        dataset_dir_name = f'{args.dataset_name}'
    # read original and synthetic data
    root = os.path.join(args.data_dir, "datasets")
    val_pair_dir = os.path.join(root, dataset_dir_name)
    file_name = os.path.join(val_pair_dir, f'{args.split}.jsonl')
    dataset = load_dataset("json", data_files=file_name)
    # dataset = dataset.remove_columns("paraphrase")
    if args.eval_split is not None:
        eval_file_name = os.path.join(val_pair_dir, f'{args.eval_split}.jsonl')
        eval_dataset = load_dataset("json", data_files=eval_file_name)

    # load mixed dataset
    if args.mix_dataset_name is not None:
        if args.mix_dataset_name == 'pile':
            mix_dataset_dir_name = f'{args.mix_dataset_name}_{args.mix_subset_name}'
        else:
            mix_dataset_dir_name = f'{args.mix_dataset_name}'
        # read original and synthetic data
        root = os.path.join(args.data_dir, "datasets")
        mix_val_pair_dir = os.path.join(root, mix_dataset_dir_name)
        file_name = os.path.join(mix_val_pair_dir, f'{args.mix_split}.jsonl')
        mix_dataset = load_dataset("json", data_files=file_name)
    
    # if use accelerator, only run io operation for main process
    if args.use_accelerator:
        accelerator = Accelerator()
    io_flag = 1
    if args.use_accelerator:
        if not accelerator.is_main_process:
            io_flag = 0


    # load model and tokenizer
    model_name = args.model_name  # Replace with your chosen model
    config = AutoConfig.from_pretrained(model_name)
    config.num_hidden_layers = 2  # Restrict to only 2 layers
    config.hidden_size = 1024
    config.num_attention_heads = 16

    print(config)

    # Initialize the model from scratch (without pretrained weights)
    model = AutoModelForCausalLM.from_config(config)
    device_index = Accelerator().process_index
    # model.to(f"cuda:{device_index}") 

    # print(device_index)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name, 
    #     load_in_8bit=True, 
    #     # device_map='auto',
    #     device_map = {"": device_index},
    #     # torch_dtype=torch.float16,
    #     token=hf_token
    # )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
        token=hf_token,
        # padding_side = 'left',
    )
    tokenizer.pad_token = tokenizer.eos_token # add padding tokens for llama model
    # tokenizer.model_max_length = 256
    # print(tokenizer.model_max_length, flush=True)

    # tokenize the dataset
    if args.dataset_col == 'paraphrase':
        data = dataset.map(lambda samples: tokenizer([sample[0] for sample in samples[args.dataset_col]],
            # truncation=True,
            # padding='max_length',
            # max_length=args.n_tokens+1
            ), 
            batched=True)

        if args.eval_split is not None:
            eval_data = eval_dataset.map(lambda samples: tokenizer([sample[0] for sample in samples[args.dataset_col]],
                # truncation=True,
                # padding='max_length',
                # max_length=args.n_tokens+1
                ), 
                batched=True)
    else:
        data = dataset.map(lambda samples: tokenizer(samples[args.dataset_col],
            # truncation=True,
            # padding='max_length',
            # max_length=args.n_tokens+1
            ), 
            batched=True)

        if args.eval_split is not None:
            eval_data = eval_dataset.map(lambda samples: tokenizer(samples[args.dataset_col],
                # truncation=True,
                # padding='max_length',
                # max_length=args.n_tokens+1
                ), 
                batched=True)
    data = data['train']
            
    # tokenize and concatenate the mix dataset
    if args.mix_dataset_name is not None:
        if args.mix_dataset_col == 'paraphrase':
            mix_data = mix_dataset.map(lambda samples: tokenizer([sample[0] for sample in samples[args.mix_dataset_col]],
                # truncation=True,
                # padding='max_length',
                # max_length=args.n_tokens+1
                ), 
                batched=True)
        else:
            mix_data = mix_dataset.map(lambda samples: tokenizer(samples[args.mix_dataset_col],
                # truncation=True,
                # padding='max_length',
                # max_length=args.n_tokens+1
                ), 
                batched=True)
    
        mix_data = mix_data['train']
        data = data.remove_columns([col for col in data.column_names if col not in ['input_ids', 'attention_mask']])
        mix_data = mix_data.remove_columns([col for col in mix_data.column_names if col not in ['input_ids', 'attention_mask']])
        print(data, mix_data)
        data = concatenate_datasets([data, mix_data])

    data = data.shuffle()
    
    print(f'{len(data)} in total')

    print_trainable_parameters(model)
    # switch some layer params to 32bit precision
    # model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        # target_modules=["q_proj", "v_proj"],
        # lora_dropout=0.05,
        # bias="none",
        task_type="CAUSAL_LM"
    )

    # load/new LoRA model
    # if args.pretrained_lora_name is None:
    #     model = get_peft_model(model, config)
    # else:
    #     load_lora_path = os.path.join(args.data_dir, f'model/{dataset_dir_name}/{args.pretrained_lora_name}')
    #     model = PeftModel.from_pretrained(model,
    #                                       load_lora_path,
    #                                       is_trainable=True)
    #     if io_flag:
    #         print(f'* Loaded pretrained LoRA checkpoint: {args.pretrained_lora_name}')
    #         print(f'loaded lora parameters:')

    # model = get_peft_model(model, config)
    if io_flag:
        print_trainable_parameters(model)

    if args.eval_split is not None:
        evaluation_strategy="epoch"
    else:
        evaluation_strategy="no"

    # load trainer
    class SaveCheckpointCallback(TrainerCallback):
        def __init__(self, save_epochs, out_args):
            super().__init__()
            self.save_epochs = save_epochs  # List of epochs where checkpoints should be saved
            self.out_args = out_args

        def on_epoch_end(self, args, state, control, **kwargs):
            current_epoch = round(state.epoch)
            print(current_epoch)
            model_name = self.out_args.model_name.split('/')[1]
            if current_epoch in self.save_epochs:
                print(1)
                # Custom checkpoint name
                root = os.path.join(self.out_args.data_dir, 'model')
                val_pair_dir = os.path.join(root, dataset_dir_name)
                file_name = f'{model_name}_simple_{self.out_args.split}_{current_epoch}.0epoch_{self.out_args.lora_rank}rank'
                checkpoint_dir = os.path.join(val_pair_dir, file_name)
                kwargs['model'].save_pretrained(checkpoint_dir)
                print(f"Checkpoint saved at: {checkpoint_dir}")

    trainer = Trainer(
        model=model, 
        train_dataset=data, #['train'],
        eval_dataset=eval_data['train'],
        args=TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=2,
            # warmup_steps=100,
            warmup_ratio=0.03,
            num_train_epochs=args.n_epochs,
            # max_steps=10, 
            learning_rate=2e-4, # 5e-5
            # fp16=True,
            evaluation_strategy=evaluation_strategy,
            logging_strategy="epoch",
            save_strategy="no",
            # logging_steps=1, 
            output_dir='outputs',
            ddp_find_unused_parameters=False,
        ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[SaveCheckpointCallback(save_epochs=[50, 100], out_args=args)]

)
    dataloader = trainer.get_train_dataloader()
    
    # print(trainer._remove_unused_columns(data['train']))
    # print(next(iter(dataloader)))

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    trainer.train()
    trainer.evaluate()

    # define model checkpoint name
    model_name = args.model_name.split('/')[1]

    mix_data_string = ''
    if args.mix_dataset_name is not None:
        mix_data_string += args.mix_dataset_name
    if args.mix_subset_name is not None:
        mix_data_string += f'_{args.mix_subset_name}'
    if args.mix_split is not None:
        mix_data_string += f'_{args.mix_split}'
    if len(mix_data_string) > 0:
        file_name = f'{model_name}_simple_{args.split}_{mix_data_string}_{args.n_epochs}epoch_{args.lora_rank}rank'
    else:
        file_name = f'{model_name}_simple_{args.split}_{args.n_epochs}epoch_{args.lora_rank}rank'

    # define plot configurations
    if args.plot_loss:
        plots_dict = {
            "loss": {
                "train": [],
                "eval": []
            }
        }
        x_label = "Num. of epochs"
        y_label = "Loss"
        save_dir = os.path.join(args.data_dir, f'figures/lora_loss/{dataset_dir_name}')
        os.makedirs(save_dir, exist_ok=True)
        figure_title = file_name

    if args.plot_loss and io_flag:
        print(trainer.state.log_history)
        loss_list = []
        for obj in trainer.state.log_history:
            # print(f'cpo_trainer log: {obj},{type(obj)},{obj.keys()}')
            if 'loss' in obj and obj['epoch'] == len(plots_dict['loss']['train'])+1:
                plots_dict['loss']['train'].append(obj['loss'])
            elif 'eval_loss' in obj and obj['epoch'] == len(plots_dict['loss']['eval'])+1:
                plots_dict['loss']['eval'].append(obj['eval_loss'])
        
        fig_name = plot_multiple_line_charts(plots_dict, x_label, y_label, save_dir, figure_title)
        print(f'Plotted loss in {fig_name}')
    

    root = os.path.join(args.data_dir, 'model')
    val_pair_dir = os.path.join(root, dataset_dir_name)
    os.makedirs(val_pair_dir, exist_ok=True)
    model.save_pretrained(os.path.join(val_pair_dir, file_name))
    # trainer.save_model(os.path.join(args.data_dir, f'model/llama_{args.dataset_name}'))

    # load
    # model = PeftModel.from_pretrained(model, os.path.join(args.data_dir, f'model/llama_{args.dataset_name}'))
    # model = model.merge_and_unload()
    



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/storage2/bihe/llm_data_detect")
    parser.add_argument("--pretrained_lora_name", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="pile", help='for example pile, wikimia, bookmia, cnn_dailymail')
    parser.add_argument("--aligned", type=bool, default=False, help='Use the aligned synthesized data as test data.')
    parser.add_argument("--subset_name", type=str, default="Wikipedia (en)", help='Specify the subset name for PILE dataset') #choices=['PubMed Abstracts', 'Wikipedia (en)', 'USPTO Backgrounds'], 
    parser.add_argument("--dataset_col", type=str, default="original", help='which column of dataset is used for training lora')
    parser.add_argument("--mix_dataset_name", type=str, default=None, help='mix these data before finetuning')
    parser.add_argument("--mix_subset_name", type=str, default=None, help='mix these data before finetuning')
    parser.add_argument("--mix_split", type=str, default=None, help='which split of mixed dataset is used')
    parser.add_argument("--mix_dataset_col", type=str, default=None, help='which column of mixed dataset is used')
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--text_length", type=int, default=64, choices=[128, 256, 32, 64], help='Specify the text snippet length for WikiMIA dataset')
    # parser.add_argument("--device", type=int, default=3)
    parser.add_argument("--split", type=str, default="train_original_16384")
    parser.add_argument("--eval_split", type=str, default=None)
    # parser.add_argument("--n_tokens", type=int, default=128, help='Token length of each give sample')
    parser.add_argument("--n_epochs", type=float, default=100)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--use_accelerator", type=int, default=1)
    parser.add_argument("--plot_loss", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=18)
    
    # parser.add_argument("--continue_train", type=bool, default=True, help='Use the aligned synthesized data as test data.')
    args = parser.parse_args()

    print(args, flush=True)
    
    main(args)