import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


from data_loader.text_dataloader import TextDataset
from models.bag_of_words_classifier import BoWClassifierModel


class BoWClassification:
    def __init__(self, hidden_dim=50, batch_size=32, epochs=10, lr=0.001, n_folds=10, threshold=0.5):
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vectorizer = None
        self.label_encoder = None
        self.model = None
        self.n_folds = n_folds
        self.threshold = threshold
    

    def train(self, texts_class0, texts_class1):
        assert len(texts_class0) == len(texts_class1), "Datasets must have the same length."
        # Combine texts and create labels
        texts = texts_class0 + texts_class1
        labels = [0] * len(texts_class0) + [1] * len(texts_class1)

        paired_indices = np.arange(len(texts_class0))

        # Initialize dataset
        dataset = TextDataset(texts, labels)
        self.vectorizer = dataset.get_vectorizer()
        self.label_encoder = dataset.get_label_encoder()
        
        input_dim = dataset.texts.shape[1]
        output_dim = len(set(labels))
        
        # K-Fold Cross Validation
        kfold = KFold(n_splits=self.n_folds, shuffle=True)
        tpr_list = []
        fpr_list = []
        auc_list = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(paired_indices)):
            print(f"FOLD {fold + 1}")
            print("-------------------------------")
            print(train_idx)
            print(val_idx)

            train_idx_full = np.concatenate([train_idx, train_idx + len(texts_class0)])
            val_idx_full = np.concatenate([val_idx, val_idx + len(texts_class0)])

            # Initialize model and optimizer
            self.model = BoWClassifierModel(input_dim, self.hidden_dim, output_dim).to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            
            # Subset the dataset for training and validation
            train_subset = Subset(dataset, train_idx_full)
            val_subset = Subset(dataset, val_idx_full)

            # for i in range(len(val_idx_full)//2):
            #     print(texts[val_idx_full[i]]+'\n'+texts[val_idx_full[i+len(val_idx_full)//2]]+'\n---\n')
            
            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=1, shuffle=True)
            
            for epoch in range(self.epochs):
                self.model.train()
                running_loss = 0.0
                
                for i, (inputs, labels) in enumerate(train_loader):
                    # print(inputs, labels)
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                
                print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {running_loss / len(train_loader)}')
            
            tpr, fpr, auc = self.evaluate(val_loader)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
            auc_list.append(auc)
            print(f'Validation TPR: {tpr*100}%')
            print(f'Validation FPR: {fpr*100}%')
            print(f'Validation AUC: {auc*100}%')
        
        average_tpr = np.mean(tpr_list)
        average_fpr = np.mean(fpr_list)
        average_auc = np.mean(auc_list)
        print(f'Average TPR over 10-folds: {average_tpr}%')
        print(f'Average FPR over 10-folds: {average_fpr}%')
        print(f'Average AUC over 10-folds: {average_auc}%')
        return average_tpr, average_fpr, average_auc

    def evaluate_accuracy(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy

    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                softmax_output = F.softmax(outputs, dim=1)
                positive_probs = softmax_output[:, 1]
                predicts = (positive_probs > self.threshold)
                # _, predicts = torch.max(outputs.data, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicts.cpu().numpy())
        
         # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        auc_score = roc_auc_score(y_true, y_pred)
        print(cm)
        
        # Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
        TN = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]
        TP = cm[1, 1]
        
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        
        return TPR, FPR, auc_score

    def predict(self, texts):
        self.model.eval()
        dataset = TextDataset(texts, labels=[0] * len(texts), vectorizer=self.vectorizer, label_encoder=self.label_encoder)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        predictions = []
        
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
        
        return self.label_encoder.inverse_transform(predictions)
    
if __name__ == '__main__':
    # Sample data
    texts_class0 = [
        "I love programming", "Python is amazing", "Debugging is fun",
        "I enjoy learning", "Machine learning is fascinating",
        "Coding is great", "I love solving problems", "Challenges are fun"
    ]
    texts_class1 = [
        "I hate bugs", "I dislike errors"
    ]

    # Initialize the model
    model = BoWClassifierModel(hidden_dim=50, batch_size=2, epochs=5, lr=0.001)

    # Train the model and perform 10-fold cross-validation
    average_accuracy = model.train(texts_class0, texts_class1)

    # Predict new texts
    new_texts = ["I enjoy coding", "Errors are annoying"]
    predictions = model.predict(new_texts)
    print(f'Predictions: {predictions}')
