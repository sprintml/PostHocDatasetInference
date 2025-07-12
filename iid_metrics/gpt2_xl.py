import os
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from transformers import GPT2Config, GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from accelerate import Accelerator
from helpers.model_utils import print_trainable_parameters

class GPT2Classification:
    def __init__(self, model_name='gpt2', num_labels=2, max_seq_len=128, batch_size=12, accelerator=None, save_dir=None, 
                 save_model=False, load_best=True):
        self.model_name = model_name
        self.num_labels = num_labels
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
        self.initialize_model()

    def initialize_model(self):
        model_config = GPT2Config(n_embd=1280, n_layer=36, n_head=20, num_labels=self.num_labels)  # Customize the config as needed
        if self.model is not None:
            del self.model
        self.model = GPT2ForSequenceClassification(config=model_config)
        
        print_trainable_parameters(self.model)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Ensure all parameters are trainable
        for param in self.model.parameters():
            param.requires_grad = True
        
        print(self.model.score.weight)

    def tokenize_function(self, examples):
        return self.tokenizer(examples['text'], padding="max_length", truncation=True, max_length=self.max_seq_len)

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        print(logits, labels)
        predictions = torch.argmax(torch.tensor(logits), dim=-1)
        cm = confusion_matrix(labels, predictions, labels=list(range(self.num_labels)))
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
            per_device_eval_batch_size=self.batch_size,
            evaluation_strategy="epoch",
            logging_dir="./outputs",
            logging_steps=10,
            save_strategy="epoch",  # Save checkpoints at each epoch
            save_total_limit=1,    # Keep only the best checkpoint
            load_best_model_at_end=self.load_best,  # Dynamically load the best model if load_best is True
            metric_for_best_model="AUC Score",  # Define the metric to track for the best model
            greater_is_better=True,           # Higher AUC is better
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics_with_tracking,
        )

        trainer.train()

        # Load the best model or keep the last model
        if self.load_best:
            print(f"Best model loaded with AUC: {self.best_auc:.4f}")
        else:
            print(f"Using model from the last epoch with AUC: {self.best_auc:.4f}")

        if self.save_model and self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            model_name = "best_model" if self.load_best else "last_model"
            model_path = os.path.join(self.save_dir, f"{model_name}_fold_{fold}.pt")
            self.model.save_pretrained(model_path)
            print(f"Saved {model_name} for fold {fold} at {model_path}")

        print(self.best_metrics)
        return self.best_metrics


    def cross_validate(self, texts_class0, texts_class1, num_splits=10):
        if len(texts_class0) < len(texts_class1):
            texts_class1 = texts_class1[: len(texts_class0)]
        elif len(texts_class0) > len(texts_class1):
            texts_class0 = texts_class0[: len(texts_class1)]

        texts = texts_class0 + texts_class1
        labels = [0] * len(texts_class0) + [1] * len(texts_class1)

        dataset = Dataset.from_dict({"text": texts, "label": labels})

        kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)
        tpr_list, fpr_list, auc_list = [], [], []

        for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(texts_class0)))):
            print(f"Processing fold {fold + 1}/{num_splits}")
            train_idx_full = np.concatenate([train_idx, train_idx + len(texts_class0)])
            val_idx_full = np.concatenate([val_idx, val_idx + len(texts_class0)])

            train_data = dataset.select(train_idx_full)
            eval_data = dataset.select(val_idx_full)

            self.initialize_model()

            results = self.train_and_evaluate(train_data, eval_data, fold=fold)

            tpr_list.append(results["True Positive Rate"])
            fpr_list.append(results["False Positive Rate"])
            auc_list.append(results["AUC Score"])

        avg_tpr = np.mean(tpr_list)
        avg_fpr = np.mean(fpr_list)
        avg_auc = np.mean(auc_list)

        return avg_tpr, avg_fpr, avg_auc


# Example usage
if __name__ == "__main__":
    texts_class0 = [
        "I love programming",
        "Python is amazing",
        "Debugging is fun",
        "I enjoy learning",
        "Machine learning is fascinating",
        "Coding is great",
        "I love solving problems",
        "Challenges are fun",
    ]
    texts_class1 = [
        "I hate bugs",
        "I dislike errors",
        "I hate bugs",
        "I dislike errors",
    ]

    model_name = "gpt2"
    classifier = GPT2Classification(model_name=model_name, num_labels=2, save_dir='./saved_models', save_model=True)

    avg_tpr, avg_fpr, avg_auc = classifier.cross_validate(texts_class0, texts_class1, num_splits=10)
    print(f"Average True Positive Rate: {avg_tpr:.4f}")
    print(f"Average False Positive Rate: {avg_fpr:.4f}")
    print(f"Average AUC score: {avg_auc:.4f}")