
# %%
import os
import random
import json
from abc import ABC, abstractmethod
import numpy as np
from termcolor import colored
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import wandb
from typing import Optional
from itertools import product, islice, permutations, chain

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, pipeline
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset

plt.plot()
plt.close('all')
plt.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['font.size'] = 16

# %% [markdown]
# ## Configuration

# %%
class cfg:
    model_path = "/home/gbarbadillo/data/Phi-3-mini-128k-instruct"
    adapter_path: Optional[str] = '/mnt/hdd0/Kaggle/arc24/models/20240729_arc_fine_tuning/08_phi-3_rearc100_lr2e-4_data_augmentation/checkpoint-1073'
    train_dataset = '/mnt/hdd0/Kaggle/arc24/data/arc-agi_training_challenges.json'
    # train_dataset = '/mnt/hdd0/Kaggle/arc24/data/rearc/re_arc_100.json'
    val_dataset = '/mnt/hdd0/Kaggle/arc24/data/arc-agi_evaluation_challenges.json'
    output_dir = '/mnt/hdd0/Kaggle/arc24/models/20240729_arc_fine_tuning/10_phi-3_1rearc100_2train_lr5e-5_color-swap-no-preserve_continue'
    max_seq_len = 4096
    epochs = 0
    max_steps : Optional[int] =  1000 # If given it will override epochs
    eval_steps = 100
    warmup_ratio = 0.1
    learning_rate = 5e-5
    # LoRA
    use_rslora = True,
    use_dora = True,
    lora_r = 32
    # data augmentation
    use_data_augmentation = True
    max_train_permutations = 2 # tipically 2
    color_swaps = 1
    preserve_original_colors = False
    geometric_transforms = 8 # 0-8
    swap_train_and_test = True
    repeat_prompts = False # if bigger than 0 it will repeat the prompts that many times

# %%
# overfit conf
class cfg:
    model_path = "/home/gbarbadillo/data/Phi-3-mini-128k-instruct"
    adapter_path: Optional[str] = '/mnt/hdd0/Kaggle/arc24/models/20240729_arc_fine_tuning/10_phi-3_1rearc100_2train_lr5e-5_color-swap-no-preserve_continue/checkpoint-1000'
    train_dataset = '/mnt/hdd0/Kaggle/arc24/data/arc-agi_training_challenges.json'
    val_dataset = '/mnt/hdd0/Kaggle/arc24/data/arc-agi_training_challenges.json'
    output_dir = '/mnt/hdd0/Kaggle/arc24/models/20240729_arc_fine_tuning/11_phi-3_overfit-to-train_continue_lr5e-5_longer'
    max_seq_len = 4096
    epochs = 0
    max_steps : Optional[int] =  1200 # If given it will override epochs
    eval_steps = 100
    warmup_ratio = 0.1
    learning_rate = 1e-5
    # LoRA
    use_rslora = True,
    use_dora = True,
    lora_r = 32
    # data augmentation
    use_data_augmentation = False
    max_train_permutations = 2 # tipically 2
    color_swaps = 1
    preserve_original_colors = False
    geometric_transforms = 8 # 0-8
    swap_train_and_test = True
    repeat_prompts = 10 # if bigger than 0 it will repeat the prompts that many times, useful to induce variation in the order of the prompts

# %%
# test time fine-tuning conf
class cfg:
    model_path = "/home/gbarbadillo/data/Phi-3-mini-128k-instruct"
    adapter_path: Optional[str] = '/mnt/hdd0/Kaggle/arc24/models/20240729_arc_fine_tuning/10_phi-3_1rearc100_2train_lr5e-5_color-swap-no-preserve_continue/checkpoint-1000'
    train_dataset = '/mnt/hdd0/Kaggle/arc24/data/test_time_fine-tuning/evaluation_n-2.json'
    val_dataset = '/mnt/hdd0/Kaggle/arc24/data/arc-agi_evaluation_challenges.json'
    output_dir = '/mnt/hdd0/Kaggle/arc24/models/20240802_test_time_fine-tuning/05_n-2-dataset_lr5e-5_5e3steps'
    max_seq_len = 4096
    epochs = 0
    max_steps : Optional[int] =  5000 # If given it will override epochs
    eval_steps = 50
    warmup_ratio = 0.1
    learning_rate = 5e-5
    # LoRA
    use_rslora = True,
    use_dora = True,
    lora_r = 32
    # data augmentation
    use_data_augmentation = True
    max_train_permutations = 2 # tipically 2
    color_swaps = 1
    preserve_original_colors = False
    geometric_transforms = 8 # 0-8
    swap_train_and_test = False
    repeat_prompts = 0 # if bigger than 0 it will repeat the prompts that many times, useful to induce variation in the order of the prompts

# fast test time fine-tuning conf
batch_size = 32
class cfg:
    model_path = "/home/gbarbadillo/data/Phi-3-mini-128k-instruct"
    adapter_path: Optional[str] = '/mnt/hdd0/Kaggle/arc24/models/20240729_arc_fine_tuning/10_phi-3_1rearc100_2train_lr5e-5_color-swap-no-preserve_continue/checkpoint-1000'
    train_dataset = '/mnt/hdd0/Kaggle/arc24/data/test_time_fine-tuning/evaluation_n-1.json'
    val_dataset = '/mnt/hdd0/Kaggle/arc24/data/arc-agi_evaluation_challenges.json'
    output_dir = f'/mnt/hdd0/Kaggle/arc24/models/20240802_test_time_fine-tuning/07_fast_bs{batch_size}_lr1e-3_1e3steps'
    max_seq_len = 4096
    epochs = 0
    batch_size = batch_size
    max_steps : Optional[int] =  int(1000*16/batch_size) # If given it will override epochs
    eval_steps = int(50*16/batch_size)
    warmup_ratio = 0.2
    learning_rate = 1e-3
    # LoRA
    use_rslora = True,
    use_dora = True,
    lora_r = 32
    # data augmentation
    use_data_augmentation = True
    max_train_permutations = 2 # tipically 2
    color_swaps = 1
    preserve_original_colors = False
    geometric_transforms = 8 # 0-8
    swap_train_and_test = False
    repeat_prompts = 0 # if bigger than 0 it will repeat the prompts that many times, useful to induce variation in the order of the prompts

# Smaller models config
class cfg:
    #model_path = "HuggingFaceTB/SmolLM-135M-Instruct"
    model_path = 'Qwen/Qwen2-0.5B-Instruct'
    adapter_path: Optional[str] = None
    train_dataset = '/mnt/hdd0/Kaggle/arc24/data/arc-agi_training_challenges.json'
    # train_dataset = '/mnt/hdd0/Kaggle/arc24/data/rearc/re_arc_100.json'
    val_dataset = '/mnt/hdd0/Kaggle/arc24/data/arc-agi_evaluation_challenges.json'
    output_dir = '/mnt/hdd0/Kaggle/arc24/models/20240809_smaller_models/04_Qwen2-0.5B-Instruct_lr1e-4_r4_8e3steps'
    max_seq_len = 4096
    epochs = 0
    max_steps : Optional[int] =  8000 #1000 # If given it will override epochs
    eval_steps = 100 #100
    warmup_ratio = 0.1
    batch_size = 16
    # SmolLM-135M-Instruct: (4, 4); Qwen/Qwen2-0.5B-Instruct: (1, 2)
    per_device_train_batch_size = 1
    per_device_eval_batch_size = 2
    learning_rate = 1e-4
    # LoRA
    use_rslora = True,
    use_dora = True,
    lora_r = 4
    # data augmentation
    use_data_augmentation = True #True
    max_train_permutations = 2 # tipically 2
    color_swaps = 4
    preserve_original_colors = False
    geometric_transforms = 8 # 0-8
    swap_train_and_test = True
    repeat_prompts = False # if bigger than 0 it will repeat the prompts that many times



# Smaller models 2º stage test-time fine-tuning config
class cfg:
    model_path = 'Qwen/Qwen2-0.5B-Instruct'
    adapter_path: Optional[str] = '/mnt/hdd0/Kaggle/arc24/models/20240812_smaller_models_test-time_fine-tuning/01_Qwen2-0.5B-Instruct_lr1e-4_r128_2e3steps/checkpoint-2000'
    train_dataset = '/mnt/hdd0/Kaggle/arc24/data/test_time_fine-tuning/evaluation_n-1.json'
    val_dataset = '/mnt/hdd0/Kaggle/arc24/data/arc-agi_evaluation_challenges.json'
    output_dir = '/mnt/hdd0/Kaggle/arc24/models/20240812_smaller_models_test-time_fine-tuning/debug_memory_2'
    max_seq_len = 4096
    epochs = 0
    max_steps : Optional[int] =  50 #1000 # If given it will override epochs
    eval_steps = 100 #100
    warmup_ratio = 0.1
    batch_size = 16
    # SmolLM-135M-Instruct: (4, 4); Qwen/Qwen2-0.5B-Instruct: (1, 2)
    per_device_train_batch_size = 1
    per_device_eval_batch_size = 2
    learning_rate = 1e-6
    # # LoRA
    # use_rslora = True,
    # use_dora = True,
    # lora_r = 4
    # data augmentation
    use_data_augmentation = False #True
    max_train_permutations = 2 # tipically 2
    color_swaps = 2
    preserve_original_colors = False
    geometric_transforms = 8 # 0-8
    swap_train_and_test = False
    repeat_prompts = 5 # if bigger than 0 it will repeat the prompts that many times

# Smaller models config 1.5B
class cfg:
    #model_path = "HuggingFaceTB/SmolLM-135M-Instruct"
    model_path = 'Qwen/Qwen2-1.5B-Instruct'
    adapter_path: Optional[str] = None
    train_dataset = '/mnt/hdd0/Kaggle/arc24/data/arc-agi_training_challenges.json'
    # train_dataset = '/mnt/hdd0/Kaggle/arc24/data/rearc/re_arc_100.json'
    val_dataset = '/mnt/hdd0/Kaggle/arc24/data/arc-agi_evaluation_challenges.json'
    output_dir = '/mnt/hdd0/Kaggle/arc24/models/20240809_smaller_models/05_Qwen2-1.5B-Instruct_lr1e-4_r32_8e3steps'
    max_seq_len = 4096
    epochs = 0
    max_steps : Optional[int] =  8000 #1000 # If given it will override epochs
    eval_steps = 100 #100
    warmup_ratio = 0.1
    batch_size = 16
    # SmolLM-135M-Instruct: (4, 4); Qwen/Qwen2-0.5B-Instruct: (1, 2)
    per_device_train_batch_size = 1
    per_device_eval_batch_size = 2
    learning_rate = 1e-4
    # LoRA
    use_rslora = True,
    use_dora = True,
    lora_r = 32
    # data augmentation
    use_data_augmentation = True #True
    max_train_permutations = 2 # tipically 2
    color_swaps = 4
    preserve_original_colors = False
    geometric_transforms = 8 # 0-8
    swap_train_and_test = True
    repeat_prompts = False # if bigger than 0 it will repeat the prompts that many times




# Smaller models test-time fine-tuning config
class cfg:
    model_path = 'Qwen/Qwen2-0.5B-Instruct'
    adapter_path: Optional[str] = '/mnt/hdd0/Kaggle/arc24/models/20240809_smaller_models/04_Qwen2-0.5B-Instruct_lr2e-4_r128_8e3steps/checkpoint-6700'
    train_dataset = '/mnt/hdd0/Kaggle/arc24/data/test_time_fine-tuning/evaluation_n-1.json'
    val_dataset = '/mnt/hdd0/Kaggle/arc24/data/arc-agi_evaluation_challenges.json'
    output_dir = '/mnt/hdd0/Kaggle/arc24/models/20240812_smaller_models_test-time_fine-tuning/01_Qwen2-0.5B-Instruct_lr1e-4_r128_4e3steps'
    max_seq_len = 4096
    epochs = 0
    max_steps : Optional[int] =  4000 #1000 # If given it will override epochs
    eval_steps = 100 #100
    warmup_ratio = 0.1
    batch_size = 16
    # SmolLM-135M-Instruct: (4, 4); Qwen/Qwen2-0.5B-Instruct: (1, 2)
    per_device_train_batch_size = 1
    per_device_eval_batch_size = 2
    learning_rate = 1e-4
    # # LoRA
    # use_rslora = True,
    # use_dora = True,
    # lora_r = 4
    # data augmentation
    use_data_augmentation = True #True
    max_train_permutations = 2 # tipically 2
    color_swaps = 4
    preserve_original_colors = False
    geometric_transforms = 8 # 0-8
    swap_train_and_test = False
    repeat_prompts = 0 # if bigger than 0 it will repeat the prompts that many times

# Smaller models 1.5 test-time fine-tuning config
class cfg:
    model_path = 'Qwen/Qwen2-1.5B-Instruct'
    adapter_path: Optional[str] = '/mnt/hdd0/Kaggle/arc24/models/20240809_smaller_models/05_Qwen2-1.5B-Instruct_lr1e-4_r32_8e3steps/checkpoint-4000'
    train_dataset = '/mnt/hdd0/Kaggle/arc24/data/test_time_fine-tuning/evaluation_n-1.json'
    val_dataset = '/mnt/hdd0/Kaggle/arc24/data/arc-agi_evaluation_challenges.json'
    output_dir = '/mnt/hdd0/Kaggle/arc24/models/20240812_smaller_models_test-time_fine-tuning/03_Qwen2-1.5B-Instruct_lr1e-4_r32_4e3steps'
    max_seq_len = 4096
    epochs = 0
    max_steps : Optional[int] =  4000 #1000 # If given it will override epochs
    eval_steps = 100 #100
    warmup_ratio = 0.1
    batch_size = 16
    # SmolLM-135M-Instruct: (4, 4); Qwen/Qwen2-0.5B-Instruct: (1, 2)
    per_device_train_batch_size = 1
    per_device_eval_batch_size = 2
    learning_rate = 1e-4
    # # LoRA
    # use_rslora = True,
    # use_dora = True,
    # lora_r = 4
    # data augmentation
    use_data_augmentation = True #True
    max_train_permutations = 2 # tipically 2
    color_swaps = 2
    preserve_original_colors = False
    geometric_transforms = 8 # 0-8
    swap_train_and_test = False
    repeat_prompts = 0 # if bigger than 0 it will repeat the prompts that many times

# %%
os.makedirs(cfg.output_dir, exist_ok=True)
with open(os.path.join(cfg.output_dir, 'cfg.json'), 'w') as f:
    json.dump({key:value for key, value in cfg.__dict__.items() if not key.startswith('__')}, f, indent=4)

# %% [markdown]
# ## Model

# %%
if 'llama' in cfg.model_path:
    device_map = {
        'model.embed_tokens': 0,
        'model.layers.0': 0,
        'model.layers.1': 0,
        'model.layers.2': 0,
        'model.layers.3': 0,
        'model.layers.4': 0,
        'model.layers.5': 0,
        'model.layers.6': 0,
        'model.layers.7': 0,
        'model.layers.8': 0,
        'model.layers.9': 0,
        'model.layers.10': 0,
        'model.layers.11': 0,
        'model.layers.12': 0,
        'model.layers.13': 0,
        'model.layers.14': 0,
        'model.layers.15': 0,
        'model.layers.16': 0,
        'model.layers.17': 1,
        'model.layers.18': 1,
        'model.layers.19': 1,
        'model.layers.20': 1,
        'model.layers.21': 1,
        'model.layers.22': 1,
        'model.layers.23': 1,
        'model.layers.24': 1,
        'model.layers.25': 1,
        'model.layers.26': 1,
        'model.layers.27': 1,
        'model.layers.28': 1,
        'model.layers.29': 1,
        'model.layers.30': 1,
        'model.layers.31': 1,
        'model.norm': 1,
        'model.rotary_emb': 1,
        'lm_head': 1,
    }
elif cfg.model_path == 'Qwen/Qwen2-0.5B-Instruct':
    device_map = {
        'model.embed_tokens': 0,
        'lm_head': 0,
        'model.layers.0': 0,
        'model.layers.1': 0,
        'model.layers.2': 0,
        'model.layers.3': 0,
        'model.layers.4': 0,
        'model.layers.5': 0,
        'model.layers.6': 0,
        'model.layers.7': 0,
        'model.layers.8': 1,
        'model.layers.9': 1,
        'model.layers.10': 1,
        'model.layers.11': 1,
        'model.layers.12': 1,
        'model.layers.13': 1,
        'model.layers.14': 1,
        'model.layers.15': 1,
        'model.layers.16': 1,
        'model.layers.17': 1,
        'model.layers.18': 1,
        'model.layers.19': 1,
        'model.layers.20': 1,
        'model.layers.21': 1,
        'model.layers.22': 1,
        'model.layers.23': 1,
        'model.norm': 1
    }
elif cfg.model_path == 'Qwen/Qwen2-1.5B-Instruct':
    device_map = {
        'model.embed_tokens': 0,
        'lm_head': 0,
        'model.layers.0': 0,
        'model.layers.1': 0,
        'model.layers.2': 0,
        'model.layers.3': 0,
        'model.layers.4': 0,
        'model.layers.5': 0,
        'model.layers.6': 0,
        'model.layers.7': 0,
        'model.layers.8': 0,
        'model.layers.9': 0,
        'model.layers.10': 0,
        'model.layers.11': 1,
        'model.layers.12': 1,
        'model.layers.13': 1,
        'model.layers.14': 1,
        'model.layers.15': 1,
        'model.layers.16': 1,
        'model.layers.17': 1,
        'model.layers.18': 1,
        'model.layers.19': 1,
        'model.layers.20': 1,
        'model.layers.21': 1,
        'model.layers.22': 1,
        'model.layers.23': 1,
        'model.layers.24': 1,
        'model.layers.25': 1,
        'model.layers.26': 1,
        'model.layers.27': 1,
        'model.norm': 1}
else:
    device_map = 'balanced'

# device_map = 'balanced'

model = AutoModelForCausalLM.from_pretrained(
    cfg.model_path,
    #quantization_config=bnb_config,
    device_map=device_map,
    # max_memory={0: '9GB', 1: '8GB'},
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    )

print(model.hf_device_map)

# %%
tokenizer = AutoTokenizer.from_pretrained(
    cfg.model_path,
    trust_remote_code=True)
if 'llama' in cfg.model_path:
    print('Adding <|pad|> token to tokenizer')
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.padding_side = 'right'
print(tokenizer.special_tokens_map)
print('Verification of number tokens')
for number in '0123456789':
        print(f'{number}: {[key for key in tokenizer.get_vocab().keys() if number in key and not key.startswith("<")]}')


# %%
def print_gpu_memory():
    for device in range(torch.cuda.device_count()):
        print(f'GPU {device} memory allocated: {torch.cuda.memory_allocated(device)/1024**3:.1f} GB, max memory allocated: {torch.cuda.max_memory_allocated(device)/1024**3:.1f} GB')
print_gpu_memory()

# %% [markdown]
# ## Data

# %% [markdown]
# ### Load data

# %%
def load_arc_data_with_solutions(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    solutions_filepath = filepath.replace('challenges.json', 'solutions.json')
    if filepath != solutions_filepath and os.path.exists(solutions_filepath):
        with open(solutions_filepath, 'r') as f:
            solutions = json.load(f)
        for sample_id, task in data.items():
            for idx, sample in enumerate(task['test']):
                sample['output'] = solutions[sample_id][idx]
    else:
        print('No solutions file found, the solutions should already be in the data')
    return data

# %%
train_data = load_arc_data_with_solutions(cfg.train_dataset)
sample_task = list(train_data.values())[0]

# %% [markdown]
# ### Grid encoders

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
# ### Plot

# %%


# %% [markdown]
# ### Data augmentation

# %% [markdown]
# There are many ways to augment the available arc tasks:
# 
# - Rotations and flips
# - Change the order of the train samples
# - Swap one of the train samples with one of the test samples
# - Remap the colors

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


# %%
def swap_one_train_and_test_sample(task):
    augmented_tasks = [task]
    for train_idx, train_sample in enumerate(task['train']):
        for test_idx, test_sample in enumerate(task['test']):
            augmented_task = dict()
            augmented_task['train'] = task['train'][:train_idx] + [test_sample] + task['train'][train_idx+1:]
            augmented_task['test'] = task['test'][:test_idx] + [train_sample] + task['test'][test_idx+1:]
            augmented_tasks.append(augmented_task)
    return augmented_tasks


# %%
def swap_task_colors(task, change_background_probability=0.1):
    colors = list(range(10))
    if random.random() < change_background_probability:
        new_colors = list(range(10))
        random.shuffle(new_colors)
    else:
        new_colors = list(range(1, 10))
        random.shuffle(new_colors)
        new_colors = [0] + new_colors

    color_map = {x: y for x, y in zip(colors, new_colors)}
    vectorized_mapping = np.vectorize(color_map.get)

    new_task = dict()
    for key in task.keys():
        new_task[key] = [{name:vectorized_mapping(grid) for name, grid in sample.items()} for sample in task[key]]
    return new_task


# %%
def permute_train_samples(task, max_permutations=6):
    augmented_tasks = []
    for _ in range(max_permutations):
        train_order = np.arange(len(task['train']))
        np.random.shuffle(train_order)
        augmented_task = dict()
        augmented_task['train'] = [task['train'][idx] for idx in train_order]
        augmented_task['test'] = task['test']
        augmented_tasks.append(augmented_task)
    return augmented_tasks


# %%
def apply_geometric_augmentations(task, n_augmentations=8):
    augmented_tasks = []
    data_augmentation_params = product([False, True], [0, 1, 2, 3])
    if n_augmentations < 8:
        data_augmentation_params = list(data_augmentation_params)
        indices = np.random.choice(np.arange(len(data_augmentation_params)), n_augmentations, replace=False)
        data_augmentation_params = [data_augmentation_params[idx] for idx in indices]
    for flip, n_rot90 in data_augmentation_params:
        data_augmentation = DataAugmentation(flip, n_rot90)
        augmented_task = data_augmentation.augment_task(task)
        augmented_tasks.append(augmented_task)
    return augmented_tasks

# %%
def apply_all_data_augmentations(tasks):
    print('Applying all data augmentations, initial number of tasks is', len(tasks))
    augmented_tasks = tasks
    if cfg.geometric_transforms:
        augmented_tasks = list(chain(*[apply_geometric_augmentations(task, cfg.geometric_transforms) for task in tqdm(augmented_tasks, desc='geometric augmentations')]))
        print(f'After applying geometric augmentations there are {len(augmented_tasks)} tasks')
    if cfg.swap_train_and_test:
        augmented_tasks = list(chain(*[swap_one_train_and_test_sample(task) for task in tqdm(augmented_tasks, desc='swap train and test')]))
        print(f'After swapping train and test samples there are {len(augmented_tasks)} tasks')
    if cfg.max_train_permutations:
        augmented_tasks = list(chain(*[permute_train_samples(task, max_permutations=cfg.max_train_permutations) for task in tqdm(augmented_tasks, desc='permute train samples')]))
        print(f'After permuting train samples there are {len(augmented_tasks)} tasks')
    if cfg.color_swaps:
        if cfg.preserve_original_colors:
            augmented_tasks.extend([swap_task_colors(task) for task in tqdm(augmented_tasks*cfg.color_swaps, desc='swap colors')])
        else:
            augmented_tasks = [swap_task_colors(task) for task in tqdm(augmented_tasks*cfg.color_swaps, desc='swap colors')]
        print(f'After swapping colors there are {len(augmented_tasks)} tasks')
    return augmented_tasks

# %% [markdown]
# ### Format data

# %%
task_description = """You are a helpful AI assistant. Your job is to solve tasks from the Abstraction and Reasoning Challenge (ARC). 
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

def create_prompts_from_task(task, grid_encoder):
    prompts = []
    for test_sample in task['test']:
        messages = [{"role": "system", "content": task_description}]
        user_message = "Let's see if you can solve this simple ARC task. These are some input-output grid examples that define the task.\n"
        for example_idx, sample in enumerate(task['train']):
            user_message += f"\n## Example {example_idx}\n\n### Input\n\n{grid_encoder.to_text(sample['input'])}\n"
            user_message += f"### Output\n\n{grid_encoder.to_text(sample['output'])}\n"
        user_message += f"\n## Test case\n\n### Input\n\n{grid_encoder.to_text(test_sample['input'])}\n"
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": f"### Output\n\n{grid_encoder.to_text(test_sample['output'])}\n"})

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        prompts.append(prompt)
    return prompts

# %%
def create_dataset(filepath, grid_encoder, use_data_augmentation=True, repeat_prompts=0):
    data = load_arc_data_with_solutions(filepath)

    tasks = list(data.values())
    if use_data_augmentation:
        tasks = apply_all_data_augmentations(tasks)

    prompts = []
    for task in tqdm(tasks, desc='create prompts'):
        prompts.extend(create_prompts_from_task(task, grid_encoder))
    print(len(prompts))

    np.random.shuffle(prompts)
    pretty_print_prompt(prompts[0])

    prompt_lengths = [len(tokenizer.encode(prompt)) for prompt in tqdm(prompts, desc='Calculating prompt lengths')]
    plt.hist(prompt_lengths, bins=100);
    plt.title('Prompt length distribution')
    plt.xlabel('Number of tokens');
    plt.show()

    prompts = [prompt for prompt, prompt_length in zip(prompts, prompt_lengths) if prompt_length < cfg.max_seq_len]
    print(f'Leaving {len(prompts)} prompts after removing those longer than {cfg.max_seq_len} tokens')

    if repeat_prompts:
        repeated_prompts = prompts.copy()
        for _ in range(repeat_prompts):
            repeated_prompts = repeated_prompts.copy()
            np.random.shuffle(repeated_prompts)
            prompts.extend(repeated_prompts)
        print(f'Repeating prompts {repeat_prompts} times, now there are {len(prompts)} prompts')

    dataset = Dataset.from_dict({'text': prompts})
    return dataset

def pretty_print_prompt(text, default_color='white'):
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
if 'llama' in cfg.model_path:
    # we need to add separation between numbers in the grid
    grid_encoder = GridCodeBlockEncoder(GridWithSeparationEncoder('|'))
else:
    grid_encoder = GridCodeBlockEncoder(MinimalGridEncoder())
train_dataset = create_dataset(
    cfg.train_dataset, grid_encoder,
    use_data_augmentation=cfg.use_data_augmentation,
    repeat_prompts=cfg.repeat_prompts)

# %%
print(f'One epoch would be {len(train_dataset)/16:n} steps')

# %%
val_dataset = create_dataset(cfg.val_dataset, grid_encoder, use_data_augmentation=False)

# %% [markdown]
# ## Train

# %%
#raise

# %%
if cfg.adapter_path is None:
    peft_config = LoraConfig(
        # lora_alpha: LoRA scaling factor.
        lora_alpha=64, #64,
        lora_dropout=0.1, # 0.1, althought Vaca suggested to use 0.05 for big models
        # r: the rank of the update matrices, expressed in int. Lower rank results in smaller update matrices with fewer trainable parameters.
        r=cfg.lora_r, #16
        bias="none",
        task_type="CAUSAL_LM",
        # target_modules: The modules (for example, attention blocks) to apply the LoRA update matrices.
        target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj'],
        use_rslora=cfg.use_rslora,
        use_dora=cfg.use_dora,
    )
else:
    print(f'Loading adapter from {cfg.adapter_path}')
    peft_config = None
    model = PeftModel.from_pretrained(model, cfg.adapter_path, is_trainable=True)

# %%
batch_size_kwargs = dict(
    # 4-16 batch size should be fine for lora.
    per_device_train_batch_size=cfg.per_device_train_batch_size, 
    gradient_accumulation_steps=cfg.batch_size//cfg.per_device_train_batch_size,
    per_device_eval_batch_size=cfg.per_device_eval_batch_size,
)

training_arguments = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        warmup_ratio=cfg.warmup_ratio,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="linear",
        optim="paged_adamw_8bit",

        do_eval=True,
        evaluation_strategy="steps",
        save_steps=cfg.eval_steps,
        logging_steps=10, #50,
        eval_steps=cfg.eval_steps,
        log_level="debug",

        **batch_size_kwargs
)

# %%
if 'llama' in cfg.model_path:
    print('Using llama template for collator')
    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        instruction_template='<|start_header_id|>user<|end_header_id|>',
        response_template='<|start_header_id|>assistant<|end_header_id|>',
    )
elif 'SmolLM' in cfg.model_path or 'qwen' in cfg.model_path.lower():
    print('Using SmolLM template for collator')
    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        instruction_template='<|im_start|>user',
        response_template='<|im_start|>assistant',
    )
else:
    print('Using Phi-3 template for collator')
    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        instruction_template='<|user|>',
        response_template='<|assistant|>'
    )

# %%
w = wandb.init(reinit=True,
               dir=cfg.output_dir,
               project=os.path.basename(os.path.dirname(cfg.output_dir)),
               name=os.path.basename(cfg.output_dir))
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=cfg.max_seq_len,
    data_collator=data_collator,
    args=training_arguments,
    # packing=True, # ValueError: You passed a `DataCollatorForCompletionOnlyLM` to the SFTTrainer. This is not compatible with the `packing` argument.
)

trainer.train()
w.finish()


