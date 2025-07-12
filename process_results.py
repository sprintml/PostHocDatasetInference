import os
import pandas as pd
import numpy as np
from collections import defaultdict

def aggregate_metrics(base_dir):
    # Define the required order of metrics
    metric_order = [
        "ppl", "mink", "mink_0.2", "maxk", "zlib", "ptb_ppl_ratio", "ref_ppl_ratio", "ptb_ppl_diff", "ref_ppl_diff",
        "ptb_loss_ratio", "ref_loss_ratio", "ptb_loss_diff", "ref_loss_diff", "ppl_based", "all", "all_no_diff",
        "all_custom"
    ]

    # Collect all CSV files by medium and suffix
    file_groups = defaultdict(lambda: {"test": [], "train": []})
    
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        for file in os.listdir(subdir_path):
            if file.startswith("all_metrics_test_") and "_test_" in file:
                parts = file.split("_test_")
                medium = parts[1]
                suffix = parts[-1]
                file_groups[(medium, suffix)]["test"].append(os.path.join(subdir_path, file))

            elif file.startswith("all_metrics_train_") and "_train_" in file:
                parts = file.split("_train_")
                medium = parts[1]
                suffix = parts[-1]
                file_groups[(medium, suffix)]["train"].append(os.path.join(subdir_path, file))

    # Process each group of files
    for (medium, suffix), files in file_groups.items():
        aggregated_p1000 = []
        aggregated_auc = []

        for membership, file_list in [("nonmember", files["test"]), ("member", files["train"])]:
            for filepath in file_list:
                df = pd.read_csv(filepath)

                # Ensure metrics are in the required order
                df = df[df['metric'].isin(metric_order)]
                df['metric'] = pd.Categorical(df['metric'], categories=metric_order, ordered=True)
                df = df.sort_values(by='metric')

                # Extract the last-column values for each metric
                metric_values_p1000 = {
                    metric: df.loc[df['metric'] == metric, 'p_1000'].values[-1] if metric in df['metric'].values else None
                    for metric in metric_order
                }
                metric_values_auc = {
                    metric: df.loc[df['metric'] == metric, 'AUC(%)'].values[-1] if metric in df['metric'].values else None
                    for metric in metric_order
                }

                # for key in metric_values_p1000:
                #     print(f"{metric_values_p1000[key]:.3g}")

                # Add p1000 information
                if any(metric_values_p1000.values()):
                    aggregated_p1000.append({
                        'dataset': os.path.basename(os.path.dirname(filepath)),
                        'membership': membership,
                        **metric_values_p1000
                    })

                # Add AUC information
                if any(metric_values_auc.values()):
                    aggregated_auc.append({
                        'dataset': os.path.basename(os.path.dirname(filepath)),
                        'membership': membership,
                        **metric_values_auc
                    })

        # Write the aggregated data to CSV files
        if aggregated_p1000:
            p1000_df = pd.DataFrame(aggregated_p1000)
            p1000_df = p1000_df.groupby(['dataset', 'membership'], as_index=False).mean()
            p1000_df = p1000_df[["dataset", "membership"] + metric_order]
            p1000_df.to_csv(os.path.join(base_dir, f"p1000_metrics_split_{medium}_split_{suffix}.csv"), float_format="%.2g", index=False)

        if aggregated_auc:
            auc_df = pd.DataFrame(aggregated_auc)
            auc_df = auc_df.groupby(['dataset', 'membership'], as_index=False).mean()
            auc_df = auc_df[["dataset", "membership"] + metric_order]
            auc_df.to_csv(os.path.join(base_dir, f"auc_metrics_split_{medium}_split_{suffix}.csv"), float_format="%.3g", index=False)

# Replace 'your_base_directory' with the actual path to the base directory
aggregate_metrics('/storage2/bihe/llm_data_detect/aggregated_results/p_values/mean-outliers/train-normalize-metrics_compare/EleutherAI_pythia-410m-deduped')