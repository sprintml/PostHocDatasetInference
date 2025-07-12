import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import itertools
from matplotlib.cm import get_cmap

# plt.rcParams.update({
# 	"pgf.texsystem": "pdflatex",
# 	'font.family': 'serif',
# 	'font.size' : 15,               	# Set font size to 11pt
# 	'axes.labelsize': 15,           	# -> axis labels
# 	'xtick.labelsize':12,
# 	'ytick.labelsize':12,
# 	'legend.fontsize': 12,
# 	'lines.linewidth':2,
# 	'text.usetex': False,
# 	'pgf.rcfonts': False,
# })
# plt.tight_layout(rect=[0, 0.03, 1, 0.85])


def plot_hist(data, title, xlabel, ylabel, save_dir):

    # Create the histogram
    plt.hist(data, bins=30, edgecolor='black')

    # Add titles and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Save the plot to a file
    plt.savefig(os.path.join(save_dir, f'{title}.png'))


def plot_multi_hist(data_list, label_list, title, xlabel, ylabel, save_dir):

    # Define a list of colors
    colors = plt.cm.get_cmap('Set2', len(data_list))
    
    # Create the plot
    plt.figure(figsize=(10, 6))

    for i, data in enumerate(data_list):
        plt.hist(data, bins=100, alpha=0.4, color=colors(i), label=label_list[i])

    # Add titles and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    # Save the plot to a file
    plt.savefig(os.path.join(save_dir, f'{title}.png'))


def plot_multi_hist_sorted(data_list, label_list, title, xlabel, ylabel, save_dir, width=0.8):

    def sort_and_reorder(first_list, second_list):
        # Pair each element from first_list with the corresponding element from second_list
        paired_list = list(zip(first_list, second_list))
        
        # Sort the pairs based on the elements of the first list using a lambda function
        paired_list.sort(key=lambda x: x[0])
        
        # Separate the pairs back into two lists
        sorted_first_list, reordered_second_list = zip(*paired_list)
        
        return list(sorted_first_list), list(reordered_second_list)
    
    first_list, second_list = sort_and_reorder(data_list[0], data_list[1])

    # Define a list of colors
    colors = plt.cm.get_cmap('Set2', len(data_list))
    
    # Create the plot
    plt.figure(figsize=(10, 6))

    r1 = np.arange(len(first_list))

    # Create the bar plots
    plt.bar(r1, first_list, width=width, alpha=0.4, color=colors(0), label=label_list[0])
    plt.bar(r1, second_list, width=width, alpha=0.4, color=colors(1), label=label_list[1])

    # Add titles and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    # Save the plot to a file
    plt.savefig(os.path.join(save_dir, f'{title}.png'))
    plt.close()

def plot_multiple_line_charts(plots_dict, x_label, y_label, save_dir, figure_title, x_range=None, y_range=None):
    n = len(plots_dict)  # Number of subplots
    
    # Create a new figure with subplots
    fig, axs = plt.subplots(n, 1, figsize=(10, 5 * n))
    
    # Ensure axs is iterable by making it a list if there's only one subplot
    if n == 1:
        axs = [axs]

    # Iterate over the dictionary items
    for ax, (title, lines) in zip(axs, plots_dict.items()):
        for line_label, data in lines.items():
            ax.plot(data, label=line_label)
        
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()

        # Set axis ranges if specified
        if x_range is not None:  # For example, x_range could be a tuple like (0, 100)
            ax.set_xlim(x_range[0], x_range[1])
        if y_range is not None:  # Likewise, y_range could be a tuple like (0, 10)
            ax.set_ylim(y_range[0], y_range[1])


    # Set the main title and layout adjustments
    plt.suptitle(figure_title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for the figure suptitle
    plt.subplots_adjust(hspace=0.5)
    
    # Save the figure in the specified directory with the figure title as filename
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, figure_title.replace(' ', '_') + '.png')
    plt.savefig(file_path)
    
    return file_path

# def plot_multi_pdf(data_list, label_list, title, xlabel, ylabel, save_dir):
#     # Define a list of colors
#     colors = plt.cm.get_cmap('Set2', len(data_list))
    
#     # Create the plot
#     plt.figure(figsize=(10, 6))

#     # Plot the probability distribution function for each data list
#     for idx, data in enumerate(data_list):
#         # Plot the histograms with normalized count (density)
#         sns.kdeplot(data, color=colors(idx), label=label_list[idx], fill=True, alpha=0.5)

#     # Add titles and labels
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.legend()

#     # Make sure the directory exists before saving
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
    
#     # Save the plot to a file
#     plt.savefig(os.path.join(save_dir, f'{title}.png'))

#     # Optionally display the plot
#     plt.show()
    


def plot_multi_pdf(data_list, label_list, title, xlabel, ylabel, save_dir=None):
    # Define a list of colors
    colors = plt.cm.get_cmap('Set2', len(data_list))
    
    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the probability distribution function for each data list
    for idx, data in enumerate(data_list):
        # Plot the KDE
        sns.kdeplot(data, color=colors(idx), label=label_list[idx], fill=True, alpha=0.5)
        
        # Calculate the mean and plot a dotted line
        mean_value = np.mean(data)
        plt.axvline(mean_value, color=colors(idx), linestyle='dotted', linewidth=2, 
                    label=f'Mean {label_list[idx]}: {mean_value:.2f}')

    # Add titles and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    if save_dir is not None:
        # Make sure the directory exists before saving
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save the plot to a file
        plt.savefig(os.path.join(save_dir, f'{title}.png'))

    # Optionally display the plot
    plt.show()


def plot_line_chart(data, title, x_label, y_label, save_dir, threshold=None):
    """
    Plots a line chart based on the input dictionary.
    
    Parameters:
        data (dict): A dictionary with keys "member" and "nonmember". Each contains metrics as keys and a list of (x, y) points as values.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
    """
    # Define line styles
    line_styles = {"member": "solid", "nonmember": "dotted"}
    
    # Extended list of markers and colors
    markers = itertools.cycle(["o", "s", "D", "P", "*", "X", "^", "v", "h", ">", "<", "H", "|", "_"])
    cmap = get_cmap("tab10")
    colors = itertools.cycle(cmap.colors)  # Cycle through Set2 colors
    
    # Generate consistent color and marker combinations for each metric
    metric_styles = {}
    for group in ["member", "nonmember"]:
        for metric in data.get(group, {}):
            if metric not in metric_styles:
                metric_styles[metric] = (next(colors), next(markers))
    
    plt.figure(figsize=(12, 8))
    
    # Plot the data
    for group, group_data in data.items():
        for metric, points in group_data.items():
            x, y = zip(*points)  # Unpack points into x and y
            color, marker = metric_styles[metric]
            plt.plot(
                x, y,
                label=f"{metric}-{group}",
                linestyle=line_styles[group],
                marker=marker,
                color=color
            )

    if threshold is not None:
        plt.axhline(y=threshold, color="black", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold})")

    # Set labels, legend, and grid
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{y_label} vs {x_label}")
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1), borderaxespad=0, fontsize='small')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Make sure the directory exists before saving
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save the plot to a file
    plt.savefig(os.path.join(save_dir, f'{title}.png'))

if __name__ == '__main__':
    data = {
        "member": {
            "ppl": [(0.1, 0.99), (0.2, 0.98)],
            "mink": [(0.1, 0.95), (0.2, 0.93)],
            "maxk": [(0.1, 0.92), (0.2, 0.91)],
            "zlib": [(0.1, 0.90), (0.2, 0.89)],
            "ptb": [(0.1, 0.88), (0.2, 0.87)],
            "ref": [(0.1, 0.86), (0.2, 0.85)],
        },
        "nonmember": {
            "ppl": [(0.1, 0.89), (0.2, 0.85)],
            "mink": [(0.1, 0.83), (0.2, 0.80)],
            "maxk": [(0.1, 0.78), (0.2, 0.76)],
            "zlib": [(0.1, 0.75), (0.2, 0.74)],
            "ptb": [(0.1, 0.73), (0.2, 0.72)],
            "ref": [(0.1, 0.71), (0.2, 0.70)],
        }
    }

    save_dir = '/storage2/bihe/llm_data_detect/figures/gen_data_ratio-p_value'
    title = 'cnn_dailymail_100k_4k_random192_0.5prefix'

    plot_line_chart(data, x_label="gen_data_ratio", y_label="p-value")
