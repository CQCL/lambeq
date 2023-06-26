import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np


PLOT_SAVE_PATH = 'notebook_execution_time_comparison.png'

# Function to load data from CSV files
def load_data(csv_path):
    data = pd.read_csv(csv_path)
    # cut path from notebook name
    data['notebook'] = data['notebook'].apply(lambda x: x.rpartition('/')[2])
    data['source'] = csv_path
    data['median'] = data.filter(regex='run_').median(1)
    data['min'] = data.filter(regex='run_').min(1)
    data['std'] = data.filter(regex='run_').std(1)
    return data


def plot_data(data: pd.DataFrame, title='Notebook Execution Time Comparison'):
    # Read out mean and standard deviation
    mean_data = data.groupby(['notebook', 'source'])['median'].mean().unstack().fillna(0)
    std_data = data.groupby(['notebook', 'source'])['std'].mean().unstack().fillna(0)
    min_data = data.groupby(['notebook', 'source'])['min'].mean().unstack().fillna(0)

    # Create a figure and a set of subplots
    _, ax = plt.subplots(figsize=(15, 8))

    # Define variable bar width
    samples = len(mean_data.keys())
    bar_width = 1/(2*samples)

    # Define notebook positions
    notebook_positions = np.arange(len(mean_data.index))

    n_samples = len(mean_data.columns)

    # Generate bar plots with error bars for each source
    for i, source in enumerate(mean_data.columns):
        label = source.split('/')[1]
        offset = ((i - n_samples//2 + 1/2) if n_samples % 2 == 0 else
                    (n_samples//2 - i))
        # Plot medium values as bar plot
        ax.bar(notebook_positions - offset * bar_width,
               mean_data[source],
               bar_width,
               alpha=0.2,
               color='bgrcmyk'[i],
               yerr=std_data[source],
               ecolor='gray',
               label='_nolegend_')
        # Plot minimum values as bar plot
        ax.bar(notebook_positions - offset * bar_width,
               min_data[source],
               bar_width,
               alpha=0.8,
               color='bgrcmyk'[i],
               label=label)

    ax.set_title(title)
    ax.set_xlabel('Notebook')
    ax.set_ylabel('Time (s)')
    ax.set_xticks(notebook_positions)
    ax.set_xticklabels(mean_data.index, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH)
    plt.show()


def main(notebook_runtimes_dir):
    # Traverse the 'notebook_runtimes_dir' and its subdirectories for CSV files
    data_frames = []
    for root, _, files in os.walk(notebook_runtimes_dir):
        for file in files:
            if file.endswith(".csv"):
                csv_path = os.path.join(root, file)
                data = load_data(csv_path)
                data_frames.append(data)

    # Concatenate all dataframes
    all_data = pd.concat(data_frames, ignore_index=True)

    # Plot the data
    plot_data(all_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare notebook execution times.')
    parser.add_argument('notebook_runtimes_dir', type=str,
                        help='Path to the directory containing the CSV files.')

    args = parser.parse_args()

    main(args.notebook_runtimes_dir)
