import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from transformers import GPT2Config, GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from accelerate import Accelerator
from helpers.model_utils import print_trainable_parameters
import torch.nn.functional as F

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

class GPT2ForSequenceClassificationWithFeatures(nn.Module):
    def __init__(self, config, tokenizer, feature_size, output_size, positive_linear=False, use_hidden_logits=False):
        super(GPT2ForSequenceClassificationWithFeatures, self).__init__()
        
        # Load GPT2 model
        self.gpt2 = GPT2ForSequenceClassification(config=config)
        self.tokenizer = tokenizer

        self.gpt2.resize_token_embeddings(len(self.tokenizer))
        self.gpt2.config.pad_token_id = self.tokenizer.pad_token_id

        self.use_hidden_logits = use_hidden_logits

        if self.use_hidden_logits and feature_size > 0:
            self.feature_hidden_layer = torch.nn.Linear(feature_size, feature_size)
        
        # Define the linear layer that processes concatenated outputs
        # gpt2_hidden_size = self.gpt2.config.hidden_size
        self.linear_layer = nn.Linear(2 + feature_size, output_size)

        if positive_linear:
            # use pytorch parameterization to make sure all weights are positives
            import torch.nn.utils.parametrize as parametrize
            class SoftplusParameterization(nn.Module):
                def forward(self, X):
                    return nn.functional.softplus(X)
            # Example registration of this parameterization transform
            parametrize.register_parametrization(self.linear_layer, "weight", SoftplusParameterization())
            assert torch.all(self.linear_layer.weight>0) # now all > 0

    def forward(self, input_ids, attention_mask, labels, extra_features=None):
        # Forward pass through GPT2 model
        gpt2_outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        gpt2_logits = gpt2_outputs.logits

        if self.use_hidden_logits and extra_features is not None:
            extra_features = F.relu(self.feature_hidden_layer(extra_features))
        
        # Concatenate the GPT2 hidden states with extra features
        # Ensure extra_features is a tensor of the correct shape
        if extra_features is not None:
            concatenated_features = torch.cat((gpt2_logits, extra_features), dim=-1)
        else:
            concatenated_features = gpt2_logits
        
        # Pass through the linear layer
        final_output = self.linear_layer(concatenated_features).squeeze()
        # final_output = concatenated_features

        # criterion = nn.BCEWithLogitsLoss()
        # loss = criterion(final_output, label)
        
        return final_output

class GPT2ClassificationWithFeatures:
    def __init__(self, model_name='gpt2-xl', 
                 num_labels=1, num_extra_features=0,
                 max_seq_len=128, batch_size=12, 
                 accelerator=None, save_dir=None, 
                 save_model=False, load_best=True,
                 input_extra_features=False,
                 positive_linear=False):
        self.model_name = model_name
        self.num_labels = num_labels
        self.num_extra_features = num_extra_features
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = None
        self.accelerator = accelerator
        self.save_dir = save_dir
        self.save_model = save_model
        self.load_best = load_best
        self.input_extra_features = input_extra_features
        self.positive_linear = positive_linear
        self.initialize_model()

    def initialize_model(self):
        # model_config = GPT2Config(n_embd=1280, n_layer=36, n_head=20, num_labels=self.num_labels)  # Customize the config as needed
        model_config = GPT2Config(n_embd=1280, n_layer=2, n_head=20, num_labels=2)  # Customize the config as needed
        if self.model is not None:
            del self.model

        
        self.model = GPT2ForSequenceClassificationWithFeatures(
            config=model_config, 
            tokenizer=self.tokenizer,
            feature_size=self.num_extra_features, 
            output_size=self.num_labels,
            positive_linear=self.positive_linear)

        # if self.input_extra_features:
        #     self.model = GPT2ForSequenceClassificationWithFeatures(
        #         config=model_config, 
        #         tokenizer=self.tokenizer,
        #         feature_size=self.num_extra_features, 
        #         output_size=self.num_labels,
        #         positive_linear=self.positive_linear)
        # else:
        #     self.model = GPT2ForSequenceClassification(config=model_config)
        #     self.model.resize_token_embeddings(len(self.tokenizer))
        #     self.model.config.pad_token_id = self.tokenizer.pad_token_id


        print_trainable_parameters(self.model)

        # Ensure all parameters are trainable
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

            # If extra features are provided, handle them similarly
            if self.input_extra_features:
                extra_features = [example['extra_features'] for example in examples]
                extra_features = torch.tensor(extra_features, dtype=torch.float)

            # Convert labels into a tensor
            labels = [example['labels'] for example in examples]
            labels = torch.tensor(labels, dtype=torch.long)

            # Assemble the final batch dictionary
            batch = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }

            # print(f'len in data collator: {len(input_ids), len(attention_mask), len(labels)}')

            if self.input_extra_features:
                batch['extra_features'] = extra_features

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
        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics_with_tracking,
            data_collator=collator,  # Set the data_collator to our custom collator
            num_labels=self.num_labels,
            # compute_loss_func=self.compute_loss,
        )
        # trainer = Trainer(
        #     model=self.model,
        #     args=training_args,
        #     train_dataset=train_dataset,
        #     eval_dataset=eval_dataset,
        #     compute_metrics=self.compute_metrics,
        #     data_collator=collator,  # Set the data_collator to our custom collator
        #     # num_labels=self.num_labels,
        #     # compute_loss_func=self.compute_loss,
        # )
        trainer.train()

        # Load the best model or keep the last model
        # if self.load_best:
        #     print(f"Best model loaded with AUC: {self.best_auc:.4f}")
        # else:
        #     print(f"Using model from the last epoch with AUC: {self.best_auc:.4f}")

        # predictions, labels, metrics = trainer.predict(eval_dataset)

        # train the linear model for extra epochs
        # if self.num_extra_features > 0:
        #     for param in self.model.parameters():
        #         param.requires_grad = False
        #     for param in self.model.linear_layer.parameters():
        #         param.requires_grad = True
        #     training_args = TrainingArguments(
        #         output_dir=output_dir,
        #         num_train_epochs=1000,
        #         per_device_train_batch_size=4096,
        #         per_device_eval_batch_size=4096,
        #         evaluation_strategy="no",
        #         save_strategy="no",
        #         logging_strategy="no"
        #     )
            
        #     # Use the collator when setting up the Trainer
        #     trainer = CustomTrainer(
        #         model=self.model,
        #         args=training_args,
        #         train_dataset=train_dataset,
        #         eval_dataset=eval_dataset,
        #         compute_metrics=compute_metrics_with_tracking,
        #         data_collator=collator,  # Set the data_collator to our custom collator
        #         num_labels=self.num_labels,
        #         # compute_loss_func=self.compute_loss,
        #     )
        #     trainer.train()



        test_dataloader = trainer.get_test_dataloader(eval_dataset)
        self.model.eval()
        all_logits, all_labels = [], []
        for step, batch in enumerate(tqdm(test_dataloader)):
            with torch.no_grad():
                outputs = self.model(**batch)
                logits = outputs
                all_logits.append(logits)
                all_labels.append(batch['labels'])

        logits = torch.cat(all_logits, dim=0).cpu()
        labels = torch.cat(all_labels, dim=0).cpu().numpy()

        # calculate probabilities
        if self.num_labels > 1:  # Multi-class classification
            probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
        else:  # Binary classification
            probabilities = torch.sigmoid(torch.tensor(logits)).numpy()

        # get predictions
        if self.num_labels > 1:
            predictions = torch.argmax(torch.tensor(logits), dim=-1).numpy().astype(np.int64)
        else:
            predictions = (logits > 0.5).astype(np.int64)

        auc_score = roc_auc_score(labels, predictions)

        print(f'self predict: {logits.shape, labels.shape, auc_score}')

        if self.num_labels > 1:  # Multi-class classification
            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            per_sample_loss = loss_fn(
                torch.tensor(logits),
                torch.tensor(labels).long()
            ).numpy()
        else:  # Binary classification
            loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
            per_sample_loss = loss_fn(
                torch.tensor(logits).squeeze(),
                torch.tensor(labels).float()
            ).numpy()

        print(f'final eval result: {probabilities, labels, per_sample_loss, auc_score}')

        return probabilities, labels, per_sample_loss, auc_score
        


