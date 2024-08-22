from typing import Optional
import argparse
from dataclasses import dataclass, asdict, field

@dataclass
class CFG:
    solutions_filepath: str = 'submission.json'
    ground_truth_filepath: str = '/mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7.json'

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Configuration")
    parser.add_argument('--solutions_filepath', type=str, help="Path to the model")
    parser.add_argument('--ground_truth_filepath', type=str, help="Path to the dataset to make inference")
    return parser.parse_args()

# Override default configuration using arguments
args = parse_args()
cfg = CFG(**{k: v for k, v in vars(args).items() if v is not None})
print(asdict(cfg))


from abc import ABC, abstractmethod
import json
import os
from tqdm.auto import tqdm
import numpy as np
from itertools import islice, product
import matplotlib.pyplot as plt
from matplotlib import colors
from termcolor import colored
import shutil

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# %%
import logging

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info('Started logging')


# %% [markdown]
# ### Plots

# %%
def plot_task(task):
    samples = task['train'] + task['test']
    for plot_idx, sample in enumerate(samples):
        plt.subplot(2, len(samples), plot_idx + 1)
        plot_grid(sample['input'])
        if 'output' in sample:
            plt.subplot(2, len(samples), plot_idx + 1 + len(samples))
            plot_grid(sample['output'])

def plot_grids(grids):
    for plot_idx, grid in enumerate(grids):
        plt.subplot(1, len(grids), plot_idx + 1)
        plot_grid(grid)

def plot_grid(grid):
    grid = np.array(grid)
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    plt.imshow(grid, cmap=cmap, norm=norm)
    plt.grid(True,which='both',color='lightgrey', linewidth=0.5)
    plt.xticks(np.arange(-0.5, grid.shape[1]), [])
    plt.yticks(np.arange(-0.5, grid.shape[0]), [])
    plt.xlim(-0.5, grid.shape[1]-0.5)

# %% [markdown]
# ### Evaluation

# %%
def analyze_number_of_predictions_per_task(data, texts):
    number_of_predictions = dict()
    for task_id, task in data.items():
        number_of_predictions[task_id] = len(texts[task_id]['responses'])/len(task['test'])
    plt.title('Distribution of the number of predictions per task')
    plt.hist(number_of_predictions.values(), bins=np.arange(1.5, 9))
    plt.xlabel('number of predictions')
    plt.ylabel('count')
    return number_of_predictions

# %%
def evaluate(ground_truth, solutions):
    """
    Computes the following metrics:

    - Accuracy
    - Correct pixels
    - Correct size
    """
    metrics = []
    for task_id, task_ground_truth in ground_truth.items():
        task_metrics = []
        #plot_task(data[task_id]); plt.suptitle(f'{task_id}'); plt.show()
        for idx, correct_grid in enumerate(task_ground_truth):
            predicted_grids = list(solutions[task_id][idx].values())
            predicted_grids = [grid for grid in predicted_grids if grid]

            task_metrics.append(evaluate_grid(correct_grid, predicted_grids))
            print_metrics(task_metrics[-1], f'{task_id}_{idx}')
            #plot_grids([correct_grid] + predicted_grids)
            plt.suptitle(f'{task_id}_{idx}')
            plt.show()
        metrics.append(average_metrics(task_metrics))
    print('\n'*3 + '# Aggregated metrics:')
    print_metrics(average_metrics(metrics))
    save_metrics(metrics, solutions)
    #plot_metrics_distribution(metrics)
    print_metrics(average_metrics(metrics))

def plot_metrics_distribution(metrics):
    for key in metrics[0]:
        values = [x[key] for x in metrics]
        plt.title(f'Distribution of {key}')
        plt.hist(values, bins=np.linspace(0, 1, 10))
        plt.xlabel(key)
        plt.ylabel('count')
        plt.show()

def average_metrics(metrics):
    averaged_metrics = dict()
    for key in metrics[0]:
        averaged_metrics[key] = np.mean([x[key] for x in metrics])
    return averaged_metrics

def save_metrics(metrics, solutions):
    formatted_metrics = dict(global_metrics=average_metrics(metrics))
    for task_id, task_metrics in zip(solutions, metrics):
        formatted_metrics[task_id] = task_metrics
    with open('metrics.json', 'w') as f:
        json.dump(formatted_metrics, f)

def print_metrics(metrics, prefix=''):
    text = f'{prefix}'
    for key, value in metrics.items():
        text += f'{key}: {value*100:.1f}%\t'
    print(text)


def evaluate_grid(correct_grid, predicted_grids):
    correct_grid = np.array(correct_grid)
    metrics = dict(accuracy=0, correct_pixels=0, correct_size=0, unanswered=(2 - len(predicted_grids))/2)
    for predicted_grid in predicted_grids:
        predicted_grid = np.array(predicted_grid)
        if correct_grid.shape == predicted_grid.shape:
            metrics['accuracy'] = max(metrics['accuracy'], np.all(predicted_grid == correct_grid))
            metrics['correct_pixels'] = max(metrics['correct_pixels'], np.mean(predicted_grid == correct_grid))
            metrics['correct_size'] = max(metrics['correct_size'], correct_grid.shape == predicted_grid.shape)
    return metrics



# %% [markdown]
# ## Evaluation
with open(cfg.solutions_filepath, 'r') as f:
    solutions = json.load(f)

with open(cfg.ground_truth_filepath, 'r') as f:
    ground_truth = json.load(f)

ground_truth = {key: ground_truth[key] for key in solutions}
evaluate(ground_truth, solutions)
