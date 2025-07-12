from typing import List, Tuple
from tqdm import tqdm
from collections import Counter
import nltk

# Ensure you have the necessary NLTK resources
nltk.download('punkt')  # This downloads the tokenizer model

class WordNGramSelector:
    def __init__(self, max_n: int = 1, top_k: int = 50, use_member_word=False):
        self.max_n = max_n
        self.top_k = top_k
        self.selected_ngrams_member = []
        self.selected_ngrams_non_member = []
        self.use_member_word = use_member_word

    def extract_ngrams(self, text: str) -> List[str]:
        words = nltk.word_tokenize(text)
        ngrams = []
        for n in range(1, self.max_n + 1):
            ngrams.extend([' '.join(words[i:i+n]) for i in range(len(words) - n + 1)])
        return ngrams

    def select_ngrams(self, members: List[str], non_members: List[str]):
        member_ngram_cnt = Counter({})
        non_member_ngram_cnt = Counter({})
        all_ngrams = set()
        
        print(f'Extracting n-grams for member data...')
        for text in tqdm(members):
            member_ngram_cnt.update(self.extract_ngrams(text))
        all_ngrams.update(member_ngram_cnt)
        
        print(f'Extracting n-grams for non-member data...')
        for text in tqdm(non_members):
            non_member_ngram_cnt.update(self.extract_ngrams(text))
        all_ngrams.update(non_member_ngram_cnt)

        ngram_ratios = []
        
        print(f'Computing TPR and FPR...')
        if self.use_member_word:
            for ngram in tqdm(all_ngrams):
                tpr = member_ngram_cnt[ngram]+1 / len(members)+1
                fpr = non_member_ngram_cnt[ngram]+1 / len(non_members)+1
                if fpr > 0:
                    ratio = tpr / fpr
                    ngram_ratios.append((ngram, ratio))

            ngram_ratios.sort(key=lambda x: x[1], reverse=True)
            self.selected_ngrams_member = [ngram for ngram, _ in ngram_ratios[:self.top_k]]
            print(self.selected_ngrams_member)
        else:
            # check the ngrams that has highest fpr-tpr rate
            ngram_ratios = []
            for ngram in tqdm(all_ngrams):
                tpr = member_ngram_cnt[ngram]+1 / len(members)+1
                fpr = non_member_ngram_cnt[ngram]+1 / len(non_members)+1
                if tpr > 0:
                    ratio = fpr / tpr
                    ngram_ratios.append((ngram, ratio))
            ngram_ratios.sort(key=lambda x: x[1], reverse=True)
            self.selected_ngrams_non_member = [ngram for ngram, ratio in ngram_ratios[:self.top_k]]
        print(ngram_ratios[:self.top_k])

    def predict(self, sample: str) -> bool:
        if self.use_member_word:
            is_member = any(ngram in sample for ngram in self.selected_ngrams_member)
        else:
            is_member = not any(ngram in sample for ngram in self.selected_ngrams_non_member)
        return is_member

    def evaluate(self, test_members: List[str], test_non_members: List[str]) -> Tuple[float, float]:
        tp = sum(1 for m in test_members if self.predict(m))
        fp = sum(1 for nm in test_non_members if self.predict(nm))
        
        tpr = tp / len(test_members)
        fpr = fp / len(test_non_members)
        
        return tpr, fpr
