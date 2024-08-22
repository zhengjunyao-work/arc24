from typing import Optional
import argparse
from dataclasses import dataclass, asdict, field

@dataclass
class CFG:
    # Model
    model_path: str = "/home/gbarbadillo/data/Qwen2-0.5B-arc"
    max_model_len: int = 8192 #61000 for phi-3
    # Dataset
    #dataset_path: str = '/mnt/hdd0/Kaggle/arc24/data/arc-agi_evaluation_challenges.json'
    dataset_path: str = '/mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7.json'
    n_tasks: Optional[int] = None # Optional parameter to limit the number of task in the inference, set it to None to use all the tasks
    # Inference params
    max_predictions_per_task: int = 2 #
    sampling_params: dict = field(default_factory=lambda: dict(temperature=0.0, max_tokens=1000)) # https://docs.vllm.ai/en/latest/dev/sampling_params.html

# %%
from jinja2 import Template

system_prompt = """You are a helpful AI assistant. Your job is to solve tasks from the Abstraction and Reasoning Challenge (ARC). 
The user will present you with sample input and output grids for each task. 
Your job will be to understand the transformation between the input and the output and apply it to the last input grid given by the user. 
The puzzle-like inputs and outputs present a grid where each square can be one of ten colors. A grid can be any height or width between 1x1 and 30x30.
The background of the grid is typically colored with 0.
The tasks from ARC are based on the following priors:

- Objectness: Objects persist and cannot appear or disappear without reason. Objects can interact or not depending on the circumstances.
- Goal-directed: Objects can be animate or inanimate. Some objects are "agents" - they have intentions and they pursue goals.
- Numbers & counting: Objects can be counted or sorted by their shape, appearance, or movement using basic mathematics like addition, subtraction, and comparison.
- Basic geometry & topology: Objects can be shapes like rectangles, triangles, and circles which can be mirrored, rotated, translated, deformed, combined, repeated, etc. Differences in distances can be detected.

The transformations between input and output should be based on these priors.
"""

prompt_template = Template("""Let's see if you can solve this simple ARC task. These are some input-output grid examples that define the task.
{% for sample in train_samples %}
## Example {{ loop.index }}

### Input

{{ sample.input }}

### Output

{{ sample.output }}
{% endfor %}
## Test case

### Input

{{ test_input }}
""")

answer_template = Template("""### Output

{{ test_output }}""")

train_samples = [dict(input=[0], output=[1]), dict(input=[2], output=[3])]
prompt = prompt_template.render(train_samples=train_samples, test_input=[4])
print(prompt)
print(answer_template.render(test_output=[5]))

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Configuration")
    parser.add_argument('--model_path', type=str, help="Path to the model")
    parser.add_argument('--dataset_path', type=str, help="Path to the dataset to make inference")
    parser.add_argument('--n_tasks', type=int, help="If given only the first n tasks will be evaluated")
    return parser.parse_args()


# Override default configuration using arguments
args = parse_args()
cfg = CFG(**{k: v for k, v in vars(args).items() if v is not None})
print(asdict(cfg))


# %%
import os
is_dry_run = cfg.dataset_path == '/kaggle/input/arc-prize-2024/arc-agi_test_challenges.json' and not os.getenv('KAGGLE_IS_COMPETITION_RERUN')
if is_dry_run:
    print('This is a dry run, no inference nor installation of packages will be done')

# %% [markdown]
# ## Install

# %%
if not is_dry_run:
    # model imports
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

# %% [markdown]
# ## Imports

# %%
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

# %%
import logging

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info('Started logging')

# %% [markdown]
# ## Code

# %% [markdown]
# ### Grid encoding

# %% [markdown]
# There are many ways to encode/format the grid as input to the LLM. In this section we are going to define several encoders so we can sistematically try them all.

# %%
class GridEncoder(ABC):
    @abstractmethod
    def to_text(self, grid):
        pass

    @abstractmethod
    def to_grid(self, text):
        pass

# %%
sample_grid = np.eye(3, dtype=int).tolist()

def test_translator(translator):
    assert sample_grid == translator.to_grid(translator.to_text(sample_grid))
    print(translator.to_text(sample_grid))

# %%
class MinimalGridEncoder(GridEncoder):
    @staticmethod
    def to_text(grid):
        text = '\n'.join([''.join([str(x) for x in line]) for line in grid])
        return text

    @staticmethod
    def to_grid(text):
        lines = text.strip().splitlines()
        grid = [[int(x) for x in line] for line in lines]
        return grid

test_translator(MinimalGridEncoder())

# %%
class GridWithSeparationEncoder(GridEncoder):
    def __init__(self, split_symbol):
        self.split_symbol = split_symbol

    def to_text(self, grid):
        text = '\n'.join([self.split_symbol.join([str(x) for x in line]) for line in grid])
        return text

    def to_grid(self, text):
        lines = text.strip().splitlines()
        grid = [[int(x) for x in line.split(self.split_symbol)] for line in lines]
        return grid

test_translator(GridWithSeparationEncoder('|'))

# %%
class GridCodeBlockEncoder(GridEncoder):
    def __init__(self, base_encoder):
        self.encoder = base_encoder

    def to_text(self, grid):
        text = f'```grid\n{self.encoder.to_text(grid)}\n```'
        return text

    def to_grid(self, text):
        grid_text = text.split('```grid\n')[1].split('\n```')[0]
        grid = self.encoder.to_grid(grid_text)
        return grid

test_translator(GridCodeBlockEncoder(MinimalGridEncoder()))

test_translator(GridCodeBlockEncoder(GridWithSeparationEncoder('|')))

# %% [markdown]
# ### Prompting

# %% [markdown]
# There are also many ways to build a prompt for the ARC challenge. The class that builds the prompt will receive a grid encoder as input, this way we can try different prompts with different grid encoders.
# The class that builds the prompts needs to be also capable of parsing the response from the model.

# %%
class PromptCreator(ABC):
    def __init__(self, grid_encoder: GridEncoder):
        self.grid_encoder = grid_encoder

    @abstractmethod
    def create_task_prompts(self, task):
        pass

    @abstractmethod
    def parse_response(self, text):
        pass

# %%
class SimplePromptCreator(PromptCreator):
    def __init__(self, grid_encoder):
        super().__init__(grid_encoder)

    def create_task_prompts(self, task):
        train_samples = [{key: self.grid_encoder.to_text(grid) for key, grid in sample.items()} for sample in task['train']]
        prompts = []
        for test_sample in task['test']:
            user_message = prompt_template.render(train_samples=train_samples,
                                                  test_input=self.grid_encoder.to_text(test_sample['input']))
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": """### Output\n```grid\n"""}]
            # TODO: add start of assistant reply
            prompt = tokenizer.apply_chat_template(messages,
                                                   tokenize=False,
                                                   add_generation_prompt=False)
            prompts.append(remove_assistant_ending(prompt))
        return prompts

    def parse_response(self, text):
        return self.grid_encoder.to_grid('```grid\n' + text)


def remove_assistant_ending(text):
    """
phi-3

<|assistant|>
### Output
```grid
<|end|>
<|endoftext|>

llama 3.1

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

### Output
```grid<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    # TODO: better way to solve this, model_path could be not informative
    if 'llama' in cfg.model_path.lower():
        split_text = '<|eot_id|>'
    elif 'qwen' in cfg.model_path.lower():
        split_text = '<|im_end|>'
    else:
        split_text = '<|end|>'
    return split_text.join(text.split(split_text)[:-1])

# %%
def print_sample_prompt(data, prompt_creator):
    prompts = [prompt_creator.create_task_prompts(task)[0] for task in data.values()]
    prompts = sorted(prompts, key=lambda x: len(x))
    pretty_print_prompt(prompts[0])

def pretty_print_prompt(text, default_color='black'):
    color = default_color
    attrs = None
    for line in text.splitlines():
        if line.startswith('<|assistant|>'):
            color = 'blue'
        elif line.startswith('<|user|>'):
            color = default_color
        elif line.startswith('<|system|>'):
            color = 'green'
        if line.startswith('<'):
            attrs = ['bold']
        else:
            attrs = None
        print(colored(line, color, attrs=attrs))

# %%
def plot_input_token_length_distribution(data, prompt_creator):
    prompts = []
    for task in data.values():
        prompts.extend(prompt_creator.create_task_prompts(task))
    token_length_distribution = [len(tokenizer.tokenize(prompt)) for prompt in tqdm(prompts)]
    plt.title('Prompt token length distribution')
    plt.hist(token_length_distribution)
    plt.xlabel('n tokens')

# %%
if not is_dry_run:
    print(f'Loading {cfg.model_path}')
    llm = LLM(model=cfg.model_path,
              trust_remote_code=True,
              dtype='half',
              tensor_parallel_size=2, # to use 2 gpus
              max_model_len=cfg.max_model_len,
              #kv_cache_dtype='fp8_e5m2', I have disabled kv cache quantization because it is hurtful
              enforce_eager=True, # without this 13.9GB of memory is used on each GPU, with this is 13.3GB,
              disable_log_stats=True,
             )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    for number in '0123456789':
        print(f'{number}: {[key for key in tokenizer.get_vocab().keys() if number in key and not key.startswith("<")]}')

# %% [markdown]
# The tokenizer from phi-3 encodes each digit indepently, it does not group numbers such as 10 or 100.

# %% [markdown]
# ### Data augmentation

# %% [markdown]
# We need data augmentation to make multiple predictions for each task.

# %%
class DataAugmentation():
    def __init__(self, flip, n_rot90):
        self.flip = flip
        self.n_rot90 = n_rot90

    def augment_task(self, task):
        augmented_task = dict()
        for partition, samples in task.items():
            augmented_task[partition] = [{name:self.augment_grid(grid) for name,grid in sample.items()} for sample in samples]
        return augmented_task

    def augment_grid(self, grid):
        grid = np.array(grid)
        if self.flip:
            grid = np.flip(grid, axis=1)
        grid = np.rot90(grid, k=self.n_rot90)
        return grid.tolist()

    def revert_augmentation(self, grid):
        grid = np.array(grid)
        grid = np.rot90(grid, k=-self.n_rot90)
        if self.flip:
            grid = np.flip(grid, axis=1)
        return grid.tolist()


for flip in [True, False]:
    for n_rot90 in range(4):
        data_augmentation = DataAugmentation(flip, n_rot90)
        assert sample_grid == data_augmentation.revert_augmentation(data_augmentation.augment_grid(sample_grid))

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
# ## Inference

# %% [markdown]
# We need to generate 2 different predictions for each task. The model could fail to generate a prediction, or the parsing can fail... Thus we need a method that is robust to fails.
#
# One way to solve this would be to use data augmentation. By applying rotations and flips we could generate up to 8 variations of each task. So we could try with different data augmentations until we have 2 predictions for each task. Another alternative would be to make inference with the 8 variations and use majority voting.

# %%
def solve_task(task_id, task, prompt_creator, sampling_params):
    data_augmentation_params = product([False, True], [0, 1, 2, 3])
    solution = {task_id:[{"attempt_1": [], "attempt_2": []} for _ in task['test']]}
    texts = dict(prompts=[], responses=[], exceptions=[])
    for flip, n_rot90 in islice(data_augmentation_params, cfg.max_predictions_per_task):
        data_augmentation = DataAugmentation(flip, n_rot90)
        augmented_task = data_augmentation.augment_task(task)
        prompts = prompt_creator.create_task_prompts(augmented_task)
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        responses = [output.outputs[0].text for output in outputs]
        for idx, response in enumerate(responses):
            try:
                augmented_grid = prompt_creator.parse_response(response)
                grid = data_augmentation.revert_augmentation(augmented_grid)
                if not solution[task_id][idx]["attempt_1"]:
                    solution[task_id][idx]["attempt_1"] = grid
                elif solution[task_id][idx]["attempt_1"] != grid and not solution[task_id][idx]["attempt_2"]:
                    solution[task_id][idx]["attempt_2"] = grid
            except Exception as e:
                print(f'Exception when parsing response from {task_id}: {e}')
                texts['exceptions'].append(str(e))
        texts['prompts'].append(prompts)
        texts['responses'].append(responses)
        if is_solution_done(solution):
            break
    return solution, {task_id:texts}

def is_solution_done(solution):
    for task_id, predictions in solution.items():
        for prediction in predictions:
            for grid in prediction.values():
                if not grid:
                    return False
    return True

# %%
def inference(data, prompt_creator, sampling_params):
    solutions, texts = dict(), dict()
    for idx, (task_id, task) in tqdm(enumerate(data.items()), total=len(data), desc='Solving tasks', smoothing=0):
        logging.info(f'Solving {task_id}, {idx+1}/{len(data)}')
        task_solution, task_texts = solve_task(task_id, task, prompt_creator, sampling_params)
        solutions.update(task_solution)
        texts.update(task_texts)
    return solutions, texts

# %%
with open(cfg.dataset_path) as f:
    data = json.load(f)
if cfg.n_tasks is not None and cfg.n_tasks > 0:
    data = dict(islice(data.items(), cfg.n_tasks))
print(f'There are {len(data)} tasks to solve.')

# %%
if not is_dry_run:
    prompt_creator = SimplePromptCreator(GridCodeBlockEncoder(MinimalGridEncoder()))
    print_sample_prompt(data, prompt_creator)
    #plot_input_token_length_distribution(data, prompt_creator)

# %%
if is_dry_run:
    with open('submission.json', 'w') as f:
        json.dump(dict(dry_run=True), f)
else:
    sampling_params = SamplingParams(n=1, **cfg.sampling_params)
    solutions, texts = inference(data, prompt_creator, sampling_params)
    with open('submission.json', 'w') as f:
        json.dump(solutions, f)

# %%
if not is_dry_run:
    number_of_predictions_per_task = analyze_number_of_predictions_per_task(data, texts)
    number_of_predictions_per_task

# %% [markdown]
# ## Evaluation

# %%
ground_truth_path = cfg.dataset_path.replace('challenges.json', 'solutions.json')
if os.path.exists(ground_truth_path):
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    ground_truth = {key: ground_truth[key] for key in solutions}
    evaluate(ground_truth, solutions)

    with open('texts.json', 'w') as f:
        json.dump(texts, f)
    with open('number_of_predictions_per_task.json', 'w') as f:
        json.dump(number_of_predictions_per_task, f)

# %% [markdown]
# ## Clean

# %%
def clear_vllm_gpu_memory():
    global llm
    # https://github.com/vllm-project/vllm/issues/1908
    from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
    import torch
    import gc
    destroy_model_parallel()
    destroy_distributed_environment()
    del llm.llm_engine.model_executor
    del llm
    gc.collect()
    torch.cuda.empty_cache()

if not is_dry_run:
    clear_vllm_gpu_memory()