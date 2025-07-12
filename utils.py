import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from helpers.hf_token import hf_token as access_token


def prepare_model(model_name, cache_dir, data_dir=None, local_ckpt=None, quant="8bit"):
    '''load tokenizer'''
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    # pad token
    if 'llama' in model_name or 'Llama' in model_name:
        tokenizer.add_special_tokens({"pad_token":"<pad>"})
    elif not 'OLMo' in model_name:
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 512

    '''load model'''
    if quant is None:
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True, token=access_token).cuda()
    elif quant == "fp16":
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True, torch_dtype=torch.bfloat16, token=access_token).cuda()
    elif quant == "8bit":
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True, load_in_8bit=True, token=access_token)#.cuda()
    if local_ckpt is not None:
        print(f"* LOADING A LOCAL LORA CHECKPOINT: {local_ckpt}")
        load_lora_path = os.path.join(data_dir, f'model/{local_ckpt}')
        model = PeftModel.from_pretrained(model,
                                          load_lora_path,
                                          is_trainable=True)
        print(f"* LOADED A LOCAL LORA CHECKPOINT: {local_ckpt}")

    if 'llama' in model_name or 'Llama' in model_name:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
    print(f"Model loaded.")
    print(f"[model_name]: {model_name}")
    print(f"[local_ckpt]: {local_ckpt}")

    return model, tokenizer