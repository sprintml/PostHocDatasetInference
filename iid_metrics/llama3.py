import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from transformers import set_seed, GPT2Config, GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments

class Llama3Classificaton:
    def __init__(self, model_name='gpt2-xl', num_labels=2, max_seq_len=128):

        self.model_name = model_name
        self.num_labels = num_labels
        self.max_seq_len = max_seq_len
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # self.tokenizer.padding_side = "left" # Very Important
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = None
        self.initialize_model()

    def initialize_model(self):
        model_config = GPT2Config.from_pretrained(self.model_name, num_labels=self.num_labels)
        if self.model is not None:
            del self.model
        self.model = GPT2ForSequenceClassification.from_pretrained(self.model_name, config=model_config)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.model.config.eos_token_id
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples['text'], padding='max_length', truncation=True, max_length=self.max_seq_len)
    
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=-1)
        cm = confusion_matrix(labels, predictions, labels=list(range(self.num_labels)))
        tn, fp, fn, tp = cm.ravel()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        return {
            'True Positive Rate': tpr,
            'False Positive Rate': fpr
        }

    def train_and_evaluate(self, train_dataset, eval_dataset, output_dir='./results', epochs=3):
        train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        eval_dataset = eval_dataset.map(self.tokenize_function, batched=True)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            warmup_ratio=0.03,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            evaluation_strategy="epoch",
            logging_dir='./logs',
            logging_steps=10,
            save_strategy = 'no',
            # load_best_model_at_end=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics
        )

        trainer.train()
        results = trainer.evaluate()
        return results

    def cross_validate(self, texts_class0, texts_class1, num_splits=10):
        # Combine texts and create labels
        texts = texts_class0 + texts_class1
        labels = [0] * len(texts_class0) + [1] * len(texts_class1)
        # Create a dataset
        dataset = Dataset.from_dict({'text': texts, 'label': labels})
        
        kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)
        tpr_list = []
        fpr_list = []
        
        for fold, (train_indices, test_indices) in enumerate(kf.split(np.arange(len(dataset)))):
            print(f"Processing fold {fold + 1}/{num_splits}")
            
            train_data = dataset.select(train_indices)
            eval_data = dataset.select(test_indices)
            
            # Initialize model and tokenizer for each fold
            self.initialize_model()
            
            # Train and evaluate
            results = self.train_and_evaluate(train_data, eval_data)
            
            tpr_list.append(results['eval_True Positive Rate'])
            fpr_list.append(results['eval_False Positive Rate'])
        
        # Compute average TPR and FPR across folds
        avg_tpr = np.mean(tpr_list)
        avg_fpr = np.mean(fpr_list)
        
        return avg_tpr, avg_fpr

# Example usage
if __name__ == "__main__":
    # Replace with your actual dataset loading logic
    # Sample data
    texts_class0 = [
        "I love programming", "Python is amazing", "Debugging is fun",
        "I enjoy learning", "Machine learning is fascinating",
        "Coding is great", "I love solving problems", "Challenges are fun",
        "I love programming", "Python is amazing", "Debugging is fun",
        "I enjoy learning", "Machine learning is fascinating",
        "Coding is great", "I love solving problems", "Challenges are fun",
        "I love programming", "Python is amazing", "Debugging is fun",
        "I enjoy learning", "Machine learning is fascinating",
        "Coding is great", "I love solving problems", "Challenges are fun",
        "I love programming", "Python is amazing", "Debugging is fun",
        "I enjoy learning", "Machine learning is fascinating",
        "Coding is great", "I love solving problems", "Challenges are fun",
    ]
    texts_class1 = [
        "I hate bugs", "I dislike errors",
        "I hate bugs", "I dislike errors",
        "I hate bugs", "I dislike errors",
        "I hate bugs", "I dislike errors",
        "I hate bugs", "I dislike errors",
        "I hate bugs", "I dislike errors",
        "I hate bugs", "I dislike errors",
        "I hate bugs", "I dislike errors",
    ]
    
    # Initialize classifier
    model_name = 'gpt2-xl'
    classifier = GPT2Classificaton(model_name=model_name, num_labels=2)
    
    # Cross-validation
    avg_tpr, avg_fpr = classifier.cross_validate(texts_class0, texts_class1, num_splits=10)
    print(f"Average True Positive Rate: {avg_tpr:.4f}")
    print(f"Average False Positive Rate: {avg_fpr:.4f}")