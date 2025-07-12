import os
from tqdm import tqdm

from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
import transformers
import torch
from peft import PeftModel
import torch.distributed as dist
from helpers.hf_token import hf_token as access_token

# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model_id_mapping = {
    "llama3_8b": "meta-llama/Meta-Llama-3-8B",
    "llama3.2_1b": "meta-llama/Llama-3.2-1B",
}

class LlamaClientSimple:
    '''
    Load checkpoints of base Llama models with LoRA adapter checkpoints.
    Note that the texts are formated as a list of strings and passed to the model as a batch.
    '''
    def __init__(self, 
                 model_name, 
                 data_dir,
                 lora_name,
                 access_token=access_token,
                 device=None,
                 load_in_8bit=True
                 ):
        cache_dir = '/storage2/bihe/cache/huggingface/hub/'
        self.model_name = model_name
        if device is None:
            accelerator = Accelerator()
            device_index = accelerator.process_index
            # load base model
            self.model = AutoModelForCausalLM.from_pretrained(
                os.path.join(data_dir, f'{lora_name}'),
                device_map = {"": device_index},
            )
            # self.model.to(f"cuda:{device_index}") 
        else:
            torch.cuda.set_device(device)
            self.model = AutoModelForCausalLM.from_pretrained(
                os.path.join(data_dir, f'{lora_name}'),
                device_map = {"": device},
            )
            # self.model.to(f"{device}") 
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id_mapping[model_name], token=access_token,
            # padding_side = 'left'
        )
        print(f'model loaded.')
        self.tokenizer.add_special_tokens({"pad_token":"<pad>"})
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        # self.model.resize_token_embeddings(len(self.tokenizer))
        self.tokenizer.pad_token_id = self.model.config.eos_token_id

        # self.pipeline = transformers.pipeline(
        #     "text-generation",
        #     model=self.model,
        #     tokenizer=self.tokenizer
        # )
        # self.pipeline = transformers.pipeline(
        #     "text-generation",
        #     model=model_id_mapping[model_name],
        #     model_kwargs={"torch_dtype": torch.bfloat16},
        #     token=access_token,
        #     # batch_size=batch_size,
        #     device=device
        # )
        # self.pipeline.tokenizer.pad_token_id = self.pipeline.model.config.eos_token_id
        # self.terminators = [
        #     self.pipeline.tokenizer.eos_token_id,
        #     self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        # ]

    def get_text(self, text_batch, n_tokens=256, repetition_penalty=1.2, temperature=0.6, top_p=0.9, top_k=50, do_sample=True, num_beams=1):
        input_ids = self.tokenizer(text_batch, return_tensors='pt',
            padding=True,
            # truncation=True
        ).input_ids.to("cuda")
        output_ids = self.model.generate(
            input_ids,
            max_length=n_tokens,
            # num_return_sequences=1,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            num_beams=num_beams,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        output_texts = [self.tokenizer.decode(output_id, skip_special_tokens=True) for output_id in output_ids]
        
        return output_texts

    # def get_text(self, text_batch, batch_size=32, n_tokens=256):
    #     outputs = self.pipeline(
    #         text_batch,
    #         batch_size=batch_size,
    #         max_new_tokens=n_tokens,
    #         eos_token_id=self.terminators,
    #         do_sample=True,
    #         temperature=0.6,
    #         top_p=0.9,
    #         # pad_token_id=self.pipeline.tokenizer.eos_token_id,
    #         repetition_penalty=1.2
    #     )
    #     return [output[0]['generated_text'] for output in outputs]

if __name__ == '__main__':
    lora_name = f'llama_pile_right'
    llama_client = LlamaClientLoRA(model_name='llama3_8b', 
        data_dir="/storage2/bihe/llm_data_detect/", lora_name=lora_name)
    from tqdm import tqdm
    for i in tqdm(range(5)):
        print(i, flush=True)
        texts = llama_client.get_text(["Archangel Michael may be depicted in Christian art alone or with other angels", 
        "John Jeffry Louis is the son of the Ambassador John J. Louis Jr. and Josephine Louis.", 
        '''Aaron Smith (musician)

Aaron "The A-Train" Smith (born September 3, 1950) is a Nashville-based drummer and percussionist.

At the age of 20, Aaron Smith played drums on The Temptations' megahit ''']*2)
        # texts = llama_client.get_text(["Archangel Michael may be depicted in Christian art alone or with other angels"]*2)
        for j, text in enumerate(texts):
            print(f'--paraphrase {j}--', flush=True)
            print(text, flush=True)