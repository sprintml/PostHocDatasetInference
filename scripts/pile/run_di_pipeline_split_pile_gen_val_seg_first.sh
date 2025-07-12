#!/bin/bash
original_file_name="${doc_idx}_${n_tokens}token_max${max_snippets}_${sus_split}_random${n_tokens}_Meta-Llama-3-8B_${doc_idx}_${n_tokens}token_max${max_snippets}_${train_split}_random${n_tokens}_${lora_epoch}.0epoch_${lora_rank}rank_${n_tokens}_0.5prefix_${n_paraphrases}"
paraphrase_file_name="${doc_idx}_${n_tokens}token_max${max_snippets}_${val_split}_random${n_tokens}_Meta-Llama-3-8B_${doc_idx}_${n_tokens}token_max${max_snippets}_${train_split}_random${n_tokens}_${lora_epoch}.0epoch_${lora_rank}rank_${n_tokens}_0.5prefix_${n_paraphrases}"
original_short_name="${doc_idx}_${sus_split}_${n_tokens}token_max${max_snippets}_0.5prefix_${sus_column:0:4}"
paraphrase_short_name="${doc_idx}_${val_split}_${n_tokens}token_max${max_snippets}_0.5prefix_gen_val_${lora_epoch}e_${lora_rank}r_${val_column:0:4}"

data_dir="/storage2/bihe/llm_data_detect"

# file_name_original="${original_file_name}_${sus_column}"
# file_name_paraphrase="${paraphrase_file_name}_${val_column}"
# short_name_original="${original_short_name}_${sus_column}"
# short_name_paraphrase="${paraphrase_short_name}_${val_column}"

# idx_paraphrase=0

for idx_paraphrase in $(seq 0 $((n_paraphrases-1))); do

    if [ "${sus_column}" == "paraphrase" ]; then
        file_name_original="${original_file_name}_${sus_column}${idx_paraphrase}"
        short_name_original="${original_short_name}_${sus_column}${idx_paraphrase}"
    else
        file_name_original="${original_file_name}_${sus_column}"
        short_name_original="${original_short_name}_${sus_column}"
    fi

    if [ "${val_column}" == "paraphrase" ]; then
        file_name_paraphrase="${paraphrase_file_name}_${val_column}${idx_paraphrase}"
        short_name_paraphrase="${paraphrase_short_name}_${val_column}${idx_paraphrase}"
    else
        file_name_paraphrase="${paraphrase_file_name}_${val_column}"
        short_name_paraphrase="${paraphrase_short_name}_${val_column}"
    fi


    # mitigates activation problems
    eval "$(conda shell.bash hook)"
    source ~/.bashrc

    # if perturbation data files don't exist, generate them
    conda deactivate
    conda activate perturb
    # srun python /home/bihe/LLM_data_detect/gen_perturbation.py --split $original_file_name --dataset_name $dataset_name --subset_name "${subset_name}" --dataset_col "${sus_column}"
    # srun python /home/bihe/LLM_data_detect/gen_perturbation.py --split $paraphrase_file_name --dataset_name $dataset_name --subset_name "${subset_name}" --dataset_col "${val_column}"

    if [ "${dataset_name}" = "pile" ] || [[ "${dataset_name:0:5}" = "dolma" ]]; then
        dataset_dir_name="${dataset_name}_${subset_name}"
    else
        dataset_dir_name="${dataset_name}"
    fi

    ref_ckpt="${dataset_dir_name}/model_name_${doc_idx}_${n_tokens}token_max${max_snippets}_${ref_split}_random${n_tokens}_Meta-Llama-3-8B_${doc_idx}_${n_tokens}token_max${max_snippets}_${train_split}_random${n_tokens}_${lora_epoch}.0epoch_${lora_rank}rank_${n_tokens}_0.5prefix_1_original+paraphrase_1.0epoch_32rank"

    echo $ref_ckpt

    dataset_dir="${data_dir}/datasets/${dataset_dir_name}"
    echo "*PERTURBATION FILE PATH: ${dataset_dir}/${file_name_original}.jsonl"
    echo "${dataset_name}"

    if [ -e "${dataset_dir}/${file_name_original}.jsonl" ]; then
        echo "*PERTURBATION FILE EXISTS: ${file_name_original}"
    else
        srun python /home/bihe/LLM_data_detect/gen_perturbation.py --split $original_file_name --dataset_name $dataset_name --subset_name "${subset_name}" --dataset_col "${sus_column}" --idx_paraphrase $idx_paraphrase
    fi
    if [ -e "${dataset_dir}/${file_name_paraphrase}.jsonl" ]; then
        echo "*PERTURBATION FILE EXISTS: ${file_name_paraphrase}"
    else
        srun python /home/bihe/LLM_data_detect/gen_perturbation.py --split $paraphrase_file_name --dataset_name $dataset_name --subset_name "${subset_name}" --dataset_col "${val_column}" --idx_paraphrase $idx_paraphrase
    fi


    # if feature files don't exist, generate them by aggregating features
    conda deactivate
    conda activate llm_datadetect

    # srun python /home/bihe/LLM_data_detect/di.py --split $file_name_original --model_name $model_name --dataset_name $dataset_name --subset_name "${subset_name}" --batch_size $batch_size --local_ckpt $local_ckpt --ref_ckpt "${ref_ckpt}"
    # srun python /home/bihe/LLM_data_detect/di.py --split $file_name_paraphrase --model_name $model_name --dataset_name $dataset_name --subset_name "${subset_name}" --batch_size $batch_size --local_ckpt $local_ckpt --ref_ckpt "${ref_ckpt}"

    if [ "${local_ckpt}" = "None" ]; then
        save_model_name=$model_name
    else
        save_model_name=$local_ckpt
    fi

    feature_dir="${data_dir}/results/${save_model_name}/${dataset_dir_name}"
    if [ -e "${feature_dir}/${file_name_original}_metrics.json" ]; then
        echo "FEATURE FILE EXISTS: ${file_name_original}"
    else
        srun python /home/bihe/LLM_data_detect/di.py --split $file_name_original --model_name $model_name --dataset_name $dataset_name --subset_name "${subset_name}" --batch_size $batch_size --local_ckpt $local_ckpt --ref_ckpt "${ref_ckpt}"
    fi
    if [ -e "${feature_dir}/${file_name_paraphrase}_metrics.json" ]; then
        echo "FEATURE FILE EXISTS: ${file_name_paraphrase}"
    else
        srun python /home/bihe/LLM_data_detect/di.py --split $file_name_paraphrase --model_name $model_name --dataset_name $dataset_name --subset_name "${subset_name}" --batch_size $batch_size --local_ckpt $local_ckpt --ref_ckpt "${ref_ckpt}"
    fi


    # correct metrics
    srun python /home/bihe/LLM_data_detect/correction_script.py --split $file_name_original --model_name $model_name --dataset_name $dataset_name --subset_name "${subset_name}" --local_ckpt $local_ckpt
    srun python /home/bihe/LLM_data_detect/correction_script.py --split $file_name_paraphrase --model_name $model_name --dataset_name $dataset_name --subset_name "${subset_name}" --local_ckpt $local_ckpt

done

# run linear model
# conda activate llm_datadetect
features=$(echo $features | tr '/' ' ')
echo $features
if [ "${baseline_type}" = "text" ]; then
    for feature in $features; do
        echo $feature
        srun python /home/bihe/LLM_data_detect/linear_di.py \
            --dataset_name $dataset_name \
            --model_name "${model_name}" \
            --subset_name "${subset_name}" \
            --suspect_split $file_name_original \
            --validation_split $file_name_paraphrase \
            --suspect_split_result_name $short_name_original \
            --validation_split_result_name $short_name_paraphrase \
            --local_ckpt $local_ckpt \
            --num_samples $num_samples \
            --positive_weights $positive_weights \
            --features $feature \
            --linear_epochs $linear_epochs \
            --outliers_remove_frac $outliers_remove_frac \
            --linear_activation $linear_activation \
            --tail_remove_frac $tail_remove_frac \
            --train_split $train_split \
            --sus_split $sus_split \
            --val_split $val_split \
            --doc_idx $doc_idx \
            --max_snippets $max_snippets \
            --num_random $num_random \
            --baseline_model_type $baseline_model_type \
            --n_tokens=$n_tokens \
            --ttest_type=$ttest_type
    done
else
    for feature in $features; do
        echo $feature
        srun python /home/bihe/LLM_data_detect/linear_di_loss.py \
            --dataset_name $dataset_name \
            --model_name "${model_name}" \
            --subset_name "${subset_name}" \
            --suspect_split $file_name_original \
            --validation_split $file_name_paraphrase \
            --suspect_split_result_name $short_name_original \
            --validation_split_result_name $short_name_paraphrase \
            --local_ckpt $local_ckpt \
            --num_samples $num_samples \
            --positive_weights $positive_weights \
            --features $feature \
            --linear_epochs $linear_epochs \
            --outliers_remove_frac $outliers_remove_frac \
            --linear_activation $linear_activation \
            --tail_remove_frac $tail_remove_frac \
            --train_split $train_split \
            --sus_split $sus_split \
            --val_split $val_split \
            --doc_idx $doc_idx \
            --max_snippets $max_snippets \
            --num_random $num_random \
            --baseline_model_type $baseline_model_type \
            --n_tokens=$n_tokens \
            --ttest_type=$ttest_type
    done
fi
# srun python /home/bihe/LLM_data_detect/linear_di.py --dataset_name $dataset_name --subset_name "${subset_name}" --suspect_split $file_name_original --validation_split $file_name_paraphrase --suspect_split_result_name $short_name_original --validation_split_result_name $short_name_paraphrase  --local_ckpt $local_ckpt --features $features --n_tokens=$n_tokens