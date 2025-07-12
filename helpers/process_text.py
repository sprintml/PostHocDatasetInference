import nltk
from nltk.tokenize import NLTKWordTokenizer, word_tokenize

nltk.download('punkt')

def get_first_k_tokens_sentence(text, k):
    tokens_idx = list(NLTKWordTokenizer().span_tokenize(text))
    return text[:tokens_idx[k][0]] if len(tokens_idx) > k else text

def count_tokens(text):
    return len(word_tokenize(text))

# if using a tokenizer, the begin of sentence token is not counted
def get_first_k_tokens_sentence_with_tokenizer(text, k, tokenizer):
    tokens = tokenizer(text, return_tensors="pt")
    truncated_tokens = tokens.input_ids[0][1:k+1]
    return tokenizer.decode(truncated_tokens, skip_special_tokens=True)

def remove_first_k_tokens_sentence_with_tokenizer(text, k, tokenizer):
    tokens = tokenizer(text, return_tensors="pt")
    if k > len(tokens.input_ids[0])-1:
        return ""
    else:
        truncated_tokens = tokens.input_ids[0][k+1:]
        return tokenizer.decode(truncated_tokens, skip_special_tokens=True)

def remove_k1_get_k2_tokens_sentence_with_tokenizer(text, k1, k2, tokenizer):
    tokens = tokenizer(text, return_tensors="pt")
    if k1 > len(tokens.input_ids[0])-1:
        print(f'k1 larger than whole sequence length.\nsequence:{text}\nk1:{k1}')
        return ""
    else:
        truncated_tokens = tokens.input_ids[0][k1+1:k1+k2+1]
        return tokenizer.decode(truncated_tokens, skip_special_tokens=True)

def get_second_k_tokens_sentence_with_tokenizer(text, k, tokenizer):
    tokens = tokenizer(text, return_tensors="pt")
    if k > len(tokens.input_ids[0])-1:
        return ""
    else:
        truncated_tokens = tokens.input_ids[0][k+1:2*k+1]
        return tokenizer.decode(truncated_tokens, skip_special_tokens=True)

def count_tokens_with_tokenizer(text, tokenizer):
    return len(tokenizer(text, return_tensors="pt").input_ids[0])-1

def truncate_and_split_dataset_with_tokenizer(data, n_tokens, prefix_ratio, tokenizer):
    trunc_texts, prefixes, suffixes, seq_lengths = [], [], [], []
    for i, text in enumerate(data):
        tokens = tokenizer(text, return_tensors="pt")
        truncated_tokens = tokens.input_ids[0][1:n_tokens+1] # begin of sentence token not counted
        seq_lengths.append(len(truncated_tokens))
        prefix_tokens = truncated_tokens[0:int(len(truncated_tokens)*prefix_ratio)]
        suffix_tokens = truncated_tokens[int(len(truncated_tokens)*prefix_ratio):]
        trunc_texts.append(tokenizer.decode(truncated_tokens, skip_special_tokens=True))
        prefixes.append(tokenizer.decode(prefix_tokens, skip_special_tokens=True))
        suffixes.append(tokenizer.decode(suffix_tokens, skip_special_tokens=True))
    return trunc_texts, prefixes, suffixes, seq_lengths
