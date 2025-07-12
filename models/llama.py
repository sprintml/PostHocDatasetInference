import os
import random
import time
import numpy as np
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import set_seed
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from helpers.model_utils import print_trainable_parameters
import torch.nn.functional as F

from helpers.hf_token import hf_token as access_token

class CustomTrainer(Trainer):
    def __init__(self, num_labels, *args, **kwargs):
        # Assuming Trainer's __init__ takes args and kwargs
        super(CustomTrainer, self).__init__(*args, **kwargs)
        self.num_labels = num_labels

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Forward pass to get model outputs
        outputs = model(**inputs)
        logits = outputs
        # logits = outputs.get('logits')
        
        # Get the labels from inputs
        labels = inputs.get('labels')
        if self.num_labels > 1:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()
            labels = labels.float()
        loss = criterion(logits, labels)
        # print(f'training: {logits, labels, loss}')
        
        return (loss, outputs) if return_outputs else loss


class LlamaClassification:
    def __init__(self, model_name='meta-llama/Meta-Llama-3-8B', 
                 num_labels=1,
                 max_seq_len=128, batch_size=12, 
                 accelerator=None, save_dir=None, 
                 save_model=False,
                 from_pretrained=False,
                 use_lora=False,
                 n_layers=2):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = None
        self.accelerator = accelerator
        self.save_dir = save_dir
        self.save_model = save_model
        self.from_pretrained = from_pretrained
        self.use_lora = use_lora
        self.n_layers = n_layers
        self.initialize_model()

    def initialize_model(self):
        
        if self.model is not None:
            del self.model

        seed = int(time.time())%1000
        print(f'seed: {seed}')
        set_seed(seed)
    
        if self.from_pretrained:
            base_model_config = AutoConfig.from_pretrained(self.model_name, num_labels=self.num_labels, token=access_token)
            base_model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=base_model_config, token=access_token)
            # model_config = GPT2Config(n_embd=1600, n_layer=36, n_head=25, num_labels=2)  # Customize the config as needed
            # self.model = GPT2ForSequenceClassification(config=model_config)
            # print(self.model)
            # self.model.transformer.h[0].load_state_dict(copy.deepcopy(base_model.transformer.h[0].state_dict()))
            # self.model.transformer.h[1].load_state_dict(copy.deepcopy(base_model.transformer.h[-1].state_dict()))
            if self.use_lora:
                # Prepare model for kbit training if applicable
                base_model = prepare_model_for_kbit_training(base_model)

                # Define LoRA configuration
                lora_config = LoraConfig(
                    r=4,  # LoRA rank
                    lora_alpha=4,
                    # target_modules=["classifier"],  # Target only specific layers (e.g., "classifier")
                    lora_dropout=0.1,
                    bias="none",
                )

                # Wrap the model with LoRA
                self.model = get_peft_model(base_model, lora_config)
            else:
                self.model = base_model
        else:
            raise ValueError(f"Self-defined model configuration not supported yet!")
            # model_config = GPT2Config(n_embd=1600, n_layer=2, n_head=25, num_labels=self.num_labels)  # Customize the config as needed
            # model_config = GPT2Config(n_embd=768, n_layer=self.n_layers, n_head=12, num_labels=2)  # Customize the config as needed
            # model_config = GPT2Config(n_embd=1600, n_layer=5, n_head=25, num_labels=2)  # Customize the config as needed
            # self.model = GPT2ForSequenceClassification(config=model_config)

        
 
        self.tokenizer.add_special_tokens({"pad_token":"<pad>"})

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        print_trainable_parameters(self.model)

        # Ensure all parameters are trainable
        if not self.use_lora:
            for param in self.model.parameters():
                param.requires_grad = True
        
        # print(self.model.score.weight)

    def tokenize_function(self, examples):
        return self.tokenizer(examples['text'], padding="max_length", truncation=True, max_length=self.max_seq_len)
    

    def custom_data_collator(self):
        def collate_batch(examples):
            # Assume tokenized input sequences typically include these fields after tokenization
            input_ids = [example['input_ids'] for example in examples]
            attention_mask = [example['attention_mask'] for example in examples]

            # Convert lists to tensors
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)

            # Convert labels into a tensor
            labels = [example['labels'] for example in examples]
            labels = torch.tensor(labels, dtype=torch.long)

            # Assemble the final batch dictionary
            batch = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }

            return batch

        return collate_batch

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        labels = labels[:len(logits)]
        if self.num_labels > 1:
            predictions = torch.argmax(torch.tensor(logits), dim=-1)
        else:
            predictions = (logits > 0.5).astype(np.int64)
        cm = confusion_matrix(labels, predictions, labels=list(range(2)))
        tn, fp, fn, tp = cm.ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        auc_score = roc_auc_score(labels, predictions)

        return {
            "True Positive Rate": tpr,
            "False Positive Rate": fpr,
            "AUC Score": auc_score,
        }


    def train_and_evaluate(self, train_dataset, eval_dataset, output_dir="./results", epochs=20, fold=0):
        train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        eval_dataset = eval_dataset.map(self.tokenize_function, batched=True)
        for data in train_dataset:
            # print(data)
            if 'extra_features' in data:
                extra_features = data['extra_features']
                for feature in extra_features:
                    # print(feature)
                    if not isinstance(feature, float):
                        print(f'nan detected! {type(feature)}')

        # print(f'length after mapping: {len(train_dataset), len(eval_dataset)}')

        self.best_auc = 0
        self.best_metrics = {}

        def compute_metrics_with_tracking(eval_pred):
            metrics = self.compute_metrics(eval_pred)
            current_auc = metrics.get("AUC Score", 0)
            if current_auc > self.best_auc:
                self.best_auc = current_auc
                self.best_metrics = metrics
            return metrics

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            warmup_ratio=0.03,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size*8,
            learning_rate=2e-4,
            evaluation_strategy="epoch",
            logging_dir="./outputs",
            logging_steps=10,
            save_strategy="no",  # Save checkpoints at each epoch
            # save_total_limit=1,    # Keep only the best checkpoint
            # load_best_model_at_end=self.load_best,  # Dynamically load the best model if load_best is True
            # metric_for_best_model="AUC Score",  # Define the metric to track for the best model
            # greater_is_better=True,           # Higher AUC is better
            save_safetensors=False
        )

        collator = self.custom_data_collator()
        
        # Use the collator when setting up the Trainer
        # trainer = CustomTrainer(
        #     model=self.model,
        #     args=training_args,
        #     train_dataset=train_dataset,
        #     eval_dataset=eval_dataset,
        #     compute_metrics=compute_metrics_with_tracking,
        #     data_collator=collator,  # Set the data_collator to our custom collator
        #     num_labels=self.num_labels,
        #     # compute_loss_func=self.compute_loss,
        # )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            data_collator=collator,  # Set the data_collator to our custom collator
            # num_labels=self.num_labels,
            # compute_loss_func=self.compute_loss,
        )
        trainer.train()

        all_logits, all_labels, all_losses, auc_score = self.evaluate_model(eval_dataset)

        print(all_logits, all_labels, all_losses, auc_score)

        return all_logits, all_labels, all_losses, auc_score


    def evaluate_model(self, eval_dataset):
        # Ensure the model is in evaluation mode
        self.model.eval()

        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.batch_size, collate_fn=self.custom_data_collator())
        all_logits = []
        all_labels = []
        all_losses = []

        # Use the model without gradient calculations
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                # Move the batch to the device
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                
                # Forward pass to get logits
                outputs = self.model(**batch)
                logits = outputs.logits
                labels = batch['labels']
                
                # Compute loss
                if self.num_labels > 1:
                    criterion = nn.CrossEntropyLoss(reduction="none")
                else:
                    criterion = nn.BCEWithLogitsLoss(reduction="none")
                    labels = labels.float()
                
                losses = criterion(logits, labels)
                
                # Collect logits, labels, and losses
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
                all_losses.append(losses.cpu())

        # Concatenate all batches
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_losses = torch.cat(all_losses, dim=0)

        # Compute AUC
        if self.num_labels > 1:
            predictions = torch.argmax(all_logits, dim=-1)
            auc_score = roc_auc_score(all_labels.numpy(), predictions.numpy(), multi_class="ovr")
        else:
            probabilities = torch.sigmoid(all_logits).numpy()
            auc_score = roc_auc_score(all_labels.numpy(), probabilities)

        # Summarize results
        print("Evaluation Results:")
        print(f"AUC Score: {auc_score}")
        print(f"Per-Sample Loss (mean): {all_losses.mean().item()}")

        return all_logits, all_labels, all_losses, auc_score

