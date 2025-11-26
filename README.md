# Unlocking Post-hoc Dataset Inference with Synthetic Data (ICML 2025 & ICML 2025 Workshop DIG-BUG Oral)



The remarkable capabilities of Large Language Models (LLMs) can be mainly attributed to their massive training datasets, which are often scraped from the internet without respecting data owners’ intellectual property rights. Dataset Inference (DI) offers a potential remedy by identifying whether a suspect dataset was used in training, thereby enabling data owners to verify unauthorized use. However, existing DI methods require a private set—known to be absent from training—that closely matches the compromised dataset’s distribution. Such in-distribution, held-out data is rarely available in practice, severely limiting the applicability of DI. In this work, we address this challenge by synthetically generating the required held-out set. Our approach tackles two key obstacles: (1) creating high-quality, diverse synthetic data that accurately reflects the original distribution, which we achieve via a data generator trained on a carefully designed suffix-based completion task, and (2) bridging likelihood gaps between real and synthetic data, which is realized through post-hoc calibration. Extensive experiments on diverse text datasets show that using our generated data as a held-out set enables DI to detect the original training sets with high confidence, while maintaining a low false positive rate. This result empowers copyright owners to make legitimate claims on data usage and demonstrates our method’s reliability for real-world litigations.



## 🚀 Quick Links 

- [**Paper**](https://openreview.net/forum?id=a5Kgv47d2e): View our paper at ICML 2025.
- [**GitHub Repository**](https://github.com/sprintml/PostHocDatasetInference): Access the source code, evaluation scripts, and additional resources for our work.

## 🆕 Updates
* [Nov 26, 2025] We released the generated held-out sets for the Pile dataset.
* [July 12, 2025] We released our initial code.

## 🗂️ Use Our Generated Held-out Sets
Download our generated held-out sets for the Pile dataset from [Google Drive](https://drive.google.com/file/d/1NijRPbnx4aSYdQuqj9jJa6LxaGBJinAA/view?usp=sharing).

Under each subdirectory (named as the subsets, e.g. `pile_Pile-CC`), there are four files:

1. `train_original.jsonl`: Sequences sampled from the **member** set, natural sequences.
2. `train_paraphrase0.jsonl`: Our generated held-out set for the above **member** sequences.
3. `val+test_original.jsonl`: Sequences sampled from the **non-member** set, natural sequences.
4. `val+test_paraphrase0.jsonl`: Our generated held-out set for the above **non-member** sequences.

In each file, the `text` entry denotes the clean texts, and the other entries denotes the texts with certain text augmentations.


## 🛠️ Usage 


### 1️⃣ Install the Dependencies
Create the conda virtual environment:
```console
conda create --name datadetect python=3.12
conda activate datadetect
```

Install Pytorch==2.3.0 and other dependencies with pip:
```console
pip install torch==2.3.0  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```


Please set your directories for saving data and set that as `data_dir` argument when you run the following python scripts.
```console
export DATA_DIR=YOUR_DATA_DIR
```
Please also create and specify your huggingface token in `helpers/hf_token.py` to use the Llama3-8B model:

```python
# change it to your own huggingface token
hf_token = '[YOUR_HUGGINGFACE_TOKEN]'
```

### 2️⃣ Prepare the Pile Dataset and Sample Sequences from it

#### 1.Segment the pile documents into sequences.

Please first change the `subsets_list` in `sample_and_save_documents.py` to indicate which Pile subsets you want to sample from, and also set the sampling configurations in `configs_list`. For example:

```python
'''in sample_and_save_documents.py'''

# this split the ArXiv and EuroParl subsets
subsets_list = ["ArXiv"] 
# set the config list
configs_list = [
    {"split": "train", "data_offset_idx": 0, "n_sample": 200, "max_snippets": 100, "n_tokens": 32},
    {"split": "val+test", "data_offset_idx": 0, "n_sample": 200, "max_snippets": 100, "n_tokens": 32},
]
```

This will sample 0-199 documents from the ArXiv subset of the Pile training, validation, and test splits. Here, validation and test splits are considered as a whole, as they are both non-member sets. The sampled documents are split into sequences of 32 tokens, and then we sample 100 sequences at maximum from each document. Please refer to `subset_config.json` for the hyperparameters we use for the experiments in the paper.

Then run the following command. The command will also automatically download the Pile training split (only the first partition), the validation split, and the test split. 

```console
python sample_and_save_documents.py --data_dir $DATA_DIR
```

It will produce two files: `[DATA_DIR]/pile_ArXiv/train_0_200_32token_max100.jsonl` and `[DATA_DIR]/pile_ArXiv/train_0_200_32token_max100.jsonl`.


#### 2.Split the sequences into the generator training split and generator inference split. 
Please first change the `subsets_list` in the python file to indicate which Pile subsets you want to use. The configuration for each subset will be automatically read from `subset_config.json`. 

```python
'''in sample_and_save_data.py'''

# this split the ArXiv and EuroParl subsets
subsets_list = ["ArXiv", "EuroParl"] 
```
Then run the following command to split the datasets. It will choose the first 2000 sequences as the generator inference split, 2000-4000 sequences as the generator validation split, and the others as the generator training split:
```console
python sample_and_save_data.py --data_dir $DATA_DIR 
```


### 3️⃣ Held-out Data Generation
Train a LoRA adapter on the generator training split. The adapter is based on the Llama3-8B model.
```console
python run_lora.py --data_dir $DATA_DIR 
```

Generate the held-out data based on the generator inference split.
```console
python gen_val_data.py --data_dir $DATA_DIR 
```

### 4️⃣ Compute the MIA Scores
Compute the MIA scores with the original dataset inference.
```console
python gen_perturbation.py --data_dir $DATA_DIR 
python di.py --data_dir $DATA_DIR 
python correction_script.py --data_dir $DATA_DIR 
```

### 5️⃣ Post-hoc Calibration
Run post-hoc calibration.
```console
python linear_di.py --data_dir $DATA_DIR 
```
### (Optional) Run 3 and 4 in Parallel with Script 
Alternatively, run post-hoc calibration on multiple subsets in parallel:
```console
python run_cmd_pile_gen_val_seg_first.py --data_dir $DATA_DIR 
```

### (Optional) Finetune Pythia Models
To evaluate our method on finetuned Pythia models, use the following command to finetune a Pythia model:
```console
python run_lora.py --data_dir $DATA_DIR 
```

### (Optional) IID Evaluation
Use the following command to run IID evaluation of two text datasets:
```console
python iid_verification_metrics.py --data_dir $DATA_DIR 
```

## 👨‍💻 TODOs
- [x] release the data
- [ ] clean up the args in sampling files
- [ ] release the checkpoints

## ❤️  Acknowledgement
The dataset inference part of this repo borrows largely from the Github repo [**LLM Dataset Inference: Did you train on my dataset?**](https://github.com/pratyushmaini/llm_dataset_inference/)
Thanks for their amazing work!

## ✒️ Citing Our Work 

If you find our codebase and dataset beneficial, please consider citing our work:
```
@inproceedings{
zhao2025unlocking,
title={Unlocking Post-hoc Dataset Inference with Synthetic Data},
author={Bihe Zhao and Pratyush Maini and Franziska Boenisch and Adam Dziedzic},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=a5Kgv47d2e}
}
```