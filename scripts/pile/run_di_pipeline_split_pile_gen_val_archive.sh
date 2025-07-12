#!/bin/bash
original_file_name="${sus_split}_random${n_tokens}_Meta-Llama-3-8B_${train_split}_random${n_tokens}_${lora_epoch}.0epoch_${lora_rank}rank_${n_tokens}_0.5prefix"
paraphrase_file_name="${val_split}_random${n_tokens}_Meta-Llama-3-8B_${train_split}_random${n_tokens}_${lora_epoch}.0epoch_${lora_rank}rank_${n_tokens}_0.5prefix"
original_short_name="${sus_split}_random${n_tokens}_0.5prefix_${sus_column:0:4}"
paraphrase_short_name="${val_split}_random${n_tokens}_0.5prefix_gen_val_${lora_epoch}e_${lora_rank}r_${val_column:0:4}"

data_dir="/storage2/bihe/llm_data_detect"
model_name="EleutherAI/pythia-410m-deduped"
local_ckpt="None"
batch_size=128

file_name_original="${original_file_name}_${sus_column}"
file_name_paraphrase="${paraphrase_file_name}_${val_column}"
short_name_original="${original_short_name}_${sus_column}"
short_name_paraphrase="${paraphrase_short_name}_${val_column}"

# mitigates activation problems
eval "$(conda shell.bash hook)"
source ~/.bashrc

# if perturbation data files don't exist, generate them
conda deactivate
conda activate perturb
# srun python /home/bihe/LLM_data_detect/gen_perturbation.py --split $original_file_name --dataset_name $dataset_name --subset_name "${subset_name}" --dataset_col original
# srun python /home/bihe/LLM_data_detect/gen_perturbation.py --split $paraphrase_file_name --dataset_name $dataset_name --subset_name "${subset_name}" --dataset_col paraphrase

if [ "${dataset_name}" = "pile" ]; then
    dataset_dir_name="${dataset_name}_${subset_name}"
else
    dataset_dir_name="${dataset_name}"
fi

dataset_dir="${data_dir}/datasets/${dataset_dir_name}"
echo "*PERTURBATION FILE PATH: ${dataset_dir}/${file_name_original}.jsonl"
echo "${dataset_name}"

if [ -e "${dataset_dir}/${file_name_original}.jsonl" ]; then
    echo "*PERTURBATION FILE EXISTS: ${file_name_original}"
else
    srun python /home/bihe/LLM_data_detect/gen_perturbation.py --split $original_file_name --dataset_name $dataset_name --subset_name "${subset_name}" --dataset_col "${sus_column}"
fi
if [ -e "${dataset_dir}/${file_name_paraphrase}.jsonl" ]; then
    echo "*PERTURBATION FILE EXISTS: ${file_name_paraphrase}"
else
    srun python /home/bihe/LLM_data_detect/gen_perturbation.py --split $paraphrase_file_name --dataset_name $dataset_name --subset_name "${subset_name}" --dataset_col "${val_column}"
fi


# if feature files don't exist, generate them by aggregating features
conda deactivate
conda activate llm_datadetect
# srun python /home/bihe/LLM_data_detect/di.py --split $file_name_original --model_name $model_name --dataset_name $dataset_name --subset_name "${subset_name}" --batch_size $batch_size
# srun python /home/bihe/LLM_data_detect/di.py --split $file_name_paraphrase --model_name $model_name --dataset_name $dataset_name --subset_name "${subset_name}" --batch_size $batch_size

if [ "${local_ckpt}" = "None" ]; then
    save_model_name=$model_name
else
    save_model_name=$local_ckpt
fi

feature_dir="${data_dir}/results/${save_model_name}/${dataset_dir_name}"
if [ -e "${feature_dir}/${file_name_original}_metrics.json" ]; then
    echo "FEATURE FILE EXISTS: ${file_name_original}"
else
    srun python /home/bihe/LLM_data_detect/di.py --split $file_name_original --model_name $model_name --dataset_name $dataset_name --subset_name "${subset_name}" --batch_size $batch_size --local_ckpt $local_ckpt
fi
if [ -e "${feature_dir}/${file_name_paraphrase}_metrics.json" ]; then
    echo "FEATURE FILE EXISTS: ${file_name_paraphrase}"
else
    srun python /home/bihe/LLM_data_detect/di.py --split $file_name_paraphrase --model_name $model_name --dataset_name $dataset_name --subset_name "${subset_name}" --batch_size $batch_size --local_ckpt $local_ckpt
fi


# correct metrics
srun python /home/bihe/LLM_data_detect/correction_script.py --split $file_name_original --model_name $model_name --dataset_name $dataset_name --subset_name "${subset_name}" --local_ckpt $local_ckpt
srun python /home/bihe/LLM_data_detect/correction_script.py --split $file_name_paraphrase --model_name $model_name --dataset_name $dataset_name --subset_name "${subset_name}" --local_ckpt $local_ckpt


# run linear model
# conda activate llm_datadetect
srun python /home/bihe/LLM_data_detect/linear_di.py --dataset_name $dataset_name --subset_name "${subset_name}" --suspect_split $file_name_original --validation_split $file_name_paraphrase --suspect_split_result_name $short_name_original --validation_split_result_name $short_name_paraphrase  --local_ckpt $local_ckpt --features $features --n_tokens=$n_tokens