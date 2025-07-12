import os
from tqdm import tqdm
import transformers
import torch
from helpers.hf_token import hf_token as access_token

# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model_id_mapping = {
    'llama3_8b': "meta-llama/Meta-Llama-3-8B",
    'llama3.1_405b': "meta-llama/Meta-Llama-3.1-405B",
}

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class LlamaClientBase:
    '''
    Load checkpoints of base Llama models (not instruction tuned).
    Note that the texts are formated as a list of strings and passed to the model as a batch.
    '''
    def __init__(self, 
                 model_name, 
                 access_token=access_token,
                 device=0
                 ):
        self.model_name = model_name
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id_mapping[model_name],
            # model_kwargs={"torch_dtype": torch.bfloat16},
            token=access_token,
            # batch_size=batch_size,
            device=device
        )
        self.pipeline.tokenizer.pad_token_id = self.pipeline.model.config.eos_token_id
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def get_text(self, text_batch, batch_size=2, n_tokens=256):
        # messages = [
        #     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        #     {"role": "user", "content": "Who are you?"},
        # ]
        # messages = [
        #     {"role": "user", "content": text},
        # ]
        outputs = self.pipeline(
            text_batch,
            batch_size=batch_size,
            max_new_tokens=n_tokens,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            # pad_token_id=self.pipeline.tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
        return [output[0]['generated_text'] for output in outputs]

if __name__ == '__main__':
    llama_client = LlamaClientBase(device=1, model_name='llama3.1_405b')
    from tqdm import tqdm
    for i in tqdm(range(5)):
        print(i)
        texts = llama_client.get_text(['An old man is', 'a young man is', 'a woman is', 'I mean']*16, batch_size=32)
        for j, text in enumerate(texts):
            print(f'--paraphrase {j}--')
            print(text)