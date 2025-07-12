import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd

class TextDataset(Dataset):
    def __init__(self, texts, labels, vectorizer=None, label_encoder=None):
        if vectorizer is None:
            self.vectorizer = CountVectorizer()
            self.texts = self.vectorizer.fit_transform(texts).toarray()
            count_array = self.texts
            print(self.vectorizer.get_feature_names_out())
            df = pd.DataFrame(data=count_array,columns = self.vectorizer.get_feature_names_out())
            print(df)
        else:
            self.vectorizer = vectorizer
            self.texts = self.vectorizer.transform(texts).toarray()

        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform(labels)
        else:
            self.label_encoder = label_encoder
            self.labels = self.label_encoder.transform(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

    def get_vectorizer(self):
        return self.vectorizer

    def get_label_encoder(self):
        return self.label_encoder


if __name__ == '__main__':
    member_set = ['hello today bye tomorrow today', 'good today bad the day after tomorrow']
    non_member_set = ['fish today mutton yesterday', 'vergetarian tomorrow']
    texts = member_set + non_member_set
    labels = [0] * len(member_set) + [1] * len(non_member_set)
    testset = TextDataset(texts, labels)
    print(len(testset))
    for data in testset:
        print(data)

