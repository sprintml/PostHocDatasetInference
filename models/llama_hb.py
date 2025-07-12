import requests
import os
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
import transformers
import torch

from helpers.hf_token import hf_token as access_token



model_id_mapping = {
    'llama3_8b': "meta-llama/Meta-Llama-3-8B",
    'llama3.1_405b': "meta-llama/Meta-Llama-3.1-405B",
}
url = "https://api.hyperbolic.xyz/v1/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiZGJkaGkzODA4QGdtYWlsLmNvbSJ9.hbGJmlKVZqUGMuEdXtEPt8rlNQ7q5wUemSnYdcG7dSs"
}

class LlamaClientHB:
    '''
    Load checkpoints of base Llama models with LoRA adapter checkpoints.
    Note that the texts are formated as a list of strings and passed to the model as a batch.
    '''
    def __init__(self, 
                 model_name,
                 access_token=access_token,
                 ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id_mapping[model_name], token=access_token,
            # padding_side = 'left'
        )
        self.tokenizer.add_special_tokens({"pad_token":"<pad>"})

    def get_text(self, text_batch, n_tokens=256):
        print(text_batch)
        print(n_tokens)
        data = {
            "prompt": text_batch,
            "model": model_id_mapping[self.model_name],
            "max_tokens": n_tokens,
            "repetition_penalty": 1.2,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        response = requests.post(url, headers=headers, json=data)
        print(response)
        response = response.json()
        result = [prefix+response['choices'][i]['text'] for i, prefix in enumerate(text_batch)]
        return result

if __name__ == '__main__':
    llama_client = LlamaClientHB(model_name='llama3.1_405b')
    from tqdm import tqdm
    for i in tqdm(range(5)):
        print(i)
        texts = llama_client.get_text(['An old man is', 'a young man is', 'a woman is', 'I mean']*2)
        for j, text in enumerate(texts):
            print(f'--paraphrase {j}--')
            print(text)