from typing import Optional
import argparse
from dataclasses import dataclass, asdict, field

@dataclass
class CFG:
    output_filepath: str = 'submission.json'
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
    #sampling_params: dict = field(default_factory=lambda: dict(temperature=0.0, max_tokens=1000, use_beam_search=True, best_of=4)) # https://docs.vllm.ai/en/latest/dev/sampling_params.html

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


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Configuration")
    parser.add_argument('--model_path', type=str, help="Path to the model")
    parser.add_argument('--dataset_path', type=str, help="Path to the dataset to make inference")
    parser.add_argument('--n_tasks', type=int, help="If given only the first n tasks will be evaluated")
    parser.add_argument('--output_filepath', type=str, help="Path to the json file with the predictions")
    parser.add_argument('--max_predictions_per_task', type=int, help="Max number of predictions per task")
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

class GridEncoder(ABC):
    @abstractmethod
    def to_text(self, grid):
        pass

    @abstractmethod
    def to_grid(self, text):
        pass

sample_grid = np.eye(3, dtype=int).tolist()

def test_translator(translator):
    assert sample_grid == translator.to_grid(translator.to_text(sample_grid))
    print(translator.to_text(sample_grid))

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

class PromptCreator(ABC):
    def __init__(self, grid_encoder: GridEncoder):
        self.grid_encoder = grid_encoder

    @abstractmethod
    def create_task_prompts(self, task):
        pass

    @abstractmethod
    def parse_response(self, text):
        pass

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

def plot_input_token_length_distribution(data, prompt_creator):
    prompts = []
    for task in data.values():
        prompts.extend(prompt_creator.create_task_prompts(task))
    token_length_distribution = [len(tokenizer.tokenize(prompt)) for prompt in tqdm(prompts)]
    plt.title('Prompt token length distribution')
    plt.hist(token_length_distribution)
    plt.xlabel('n tokens')


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
# ## Inference

# %% [markdown]
# We need to generate 2 different predictions for each task. The model could fail to generate a prediction, or the parsing can fail... Thus we need a method that is robust to fails.
#
# One way to solve this would be to use data augmentation. By applying rotations and flips we could generate up to 8 variations of each task. So we could try with different data augmentations until we have 2 predictions for each task. Another alternative would be to make inference with the 8 variations and use majority voting.

def solve_task(task_id, task, prompt_creator, sampling_params):
    data_augmentation_params = product([False, True], [0, 1, 2, 3])
    solution = {task_id:[{f"attempt_{i}": [] for i in range(cfg.max_predictions_per_task)} for _ in task['test']]}
    texts = dict(prompts=[], responses=[], exceptions=[])
    for attempt_idx, (flip, n_rot90) in islice(enumerate(data_augmentation_params), cfg.max_predictions_per_task):
        data_augmentation = DataAugmentation(flip, n_rot90)
        augmented_task = data_augmentation.augment_task(task)
        prompts = prompt_creator.create_task_prompts(augmented_task)
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        responses = [output.outputs[0].text for output in outputs]
        for test_idx, response in enumerate(responses):
            try:
                augmented_grid = prompt_creator.parse_response(response)
                grid = data_augmentation.revert_augmentation(augmented_grid)
                solution[task_id][test_idx][f"attempt_{attempt_idx}"] = grid
            except Exception as e:
                print(f'Exception when parsing response from {task_id}: {e}')
                texts['exceptions'].append(str(e))
        texts['prompts'].append(prompts)
        texts['responses'].append(responses)
    return solution, {task_id:texts}


def inference(data, prompt_creator, sampling_params):
    solutions, texts = dict(), dict()
    for idx, (task_id, task) in tqdm(enumerate(data.items()), total=len(data), desc='Solving tasks', smoothing=0):
        logging.info(f'Solving {task_id}, {idx+1}/{len(data)}')
        task_solution, task_texts = solve_task(task_id, task, prompt_creator, sampling_params)
        solutions.update(task_solution)
        texts.update(task_texts)
    return solutions, texts

with open(cfg.dataset_path) as f:
    data = json.load(f)
if cfg.n_tasks is not None and cfg.n_tasks > 0:
    data = dict(islice(data.items(), cfg.n_tasks))
print(f'There are {len(data)} tasks to solve.')


prompt_creator = SimplePromptCreator(GridCodeBlockEncoder(MinimalGridEncoder()))
print_sample_prompt(data, prompt_creator)

sampling_params = SamplingParams(n=1, **cfg.sampling_params)
solutions, texts = inference(data, prompt_creator, sampling_params)
with open(cfg.output_filepath, 'w') as f:
    json.dump(solutions, f)
with open('texts.json', 'w') as f:
    json.dump(texts, f)

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

clear_vllm_gpu_memory()