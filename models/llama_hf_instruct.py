import transformers
import torch
from helpers.hf_token import hf_token as access_token

# model_id = "meta-llama/Meta-Llama-3-8B"
model_id_mapping = {
    'llama3_8b_instruct': "meta-llama/Meta-Llama-3-8B-Instruct"
}

class LlamaClientInstruct:
    '''
    Load checkpoints of instruction tuned Llama models.
    '''
    def __init__(self, 
                 model_name, access_token=access_token,
                 device=0
                 ):
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id_mapping[model_name],
            model_kwargs={"torch_dtype": torch.bfloat16},
            token=access_token,
            device=device
        )
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def get_text(self, text):
        # messages = [
        #     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        #     {"role": "user", "content": "Who are you?"},
        # ]
        messages = [
            {"role": "user", "content": text},
        ]
        outputs = self.pipeline(
            messages,
            # max_new_tokens=256,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=self.pipeline.tokenizer.eos_token_id
        )
        return outputs[0]["generated_text"][-1]['content']

if __name__ == '__main__':
    llama_client = LlamaClientInstruct()
    text = llama_client.get_text('hello!')
    print(text)