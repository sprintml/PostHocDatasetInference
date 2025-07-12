from typing import List, Tuple
from tqdm import tqdm
from collections import Counter

class CharNGramSelector:
    '''
    Select the N-grams that appears more often in the suspect dataset and less often in the test dataset as membership verifiers.
    The N-grams are at character level.
    '''
    def __init__(self, max_n: int = 5, top_k: int = 10):
        self.max_n = max_n
        self.top_k = top_k
        self.selected_ngrams = []

    def extract_ngrams(self, text: str) -> List[str]:
        ngrams = []
        for n in range(1, self.max_n + 1):
            ngrams.extend([text[i:i+n] for i in range(len(text) - n + 1)])
        return ngrams

    def select_ngrams(self, members: List[str], non_members: List[str]):
        member_ngram_cnt = Counter({})
        non_member_ngram_cnt = Counter({})
        all_ngrams = set()
        print(f'Extracting n-grams for member data...')
        for text in tqdm(members):
            member_ngram_cnt.update(self.extract_ngrams(text))
        all_ngrams.update(member_ngram_cnt)
        for text in tqdm(non_members):
            non_member_ngram_cnt.update(self.extract_ngrams(text))
        all_ngrams.update(non_member_ngram_cnt)

        ngram_ratios = []
        print(f'Computing TPR and FPR...')
        for ngram in tqdm(all_ngrams):
            tpr = member_ngram_cnt[ngram]+1 / len(members)+1
            fpr = non_member_ngram_cnt[ngram]+1 / len(members)+1
            if fpr > 0:
                ratio = tpr / fpr
                ngram_ratios.append((ngram, ratio))

        ngram_ratios.sort(key=lambda x: x[1], reverse=True)
        self.selected_ngrams = [ngram for ngram, _ in ngram_ratios[:self.top_k]]
        print(ngram_ratios[:self.top_k])

    def predict(self, sample: str) -> bool:
        return any(ngram in sample for ngram in self.selected_ngrams)

    def evaluate(self, test_members: List[str], test_non_members: List[str]) -> Tuple[float, float]:
        tp = sum(1 for m in test_members if self.predict(m))
        fp = sum(1 for nm in test_non_members if self.predict(nm))
        
        tpr = tp / len(test_members)
        fpr = fp / len(test_non_members)
        
        return tpr, fpr
    
