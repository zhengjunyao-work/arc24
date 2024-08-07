
class cfg:
    # Model
    #model_path = '/kaggle/input/phi-3/transformers/phi-3-mini-128k-instruct/1/Phi-3-mini-128k-instruct'
    #model_path = '/kaggle/input/mistral/pytorch/7b-instruct-v0.1-hf/1'
    #model_path = '/kaggle/input/llama-3/transformers/8b-chat-hf/1'
    # model_path = "/home/gbarbadillo/data/Phi-3-mini-128k-instruct"
    # model_path = "/home/gbarbadillo/data/Phi-3-count"
    # model_path = '/home/gbarbadillo/data/llama-3.1-transformers-8b-instruct-v1'
    # model_path = '/home/gbarbadillo/data/llama-3.1-count'
    model_path = "/home/gbarbadillo/data/Phi-3-arc"
    max_model_len = 8192 #61000 for phi-3
    # Dataset
    # dataset_path = '/mnt/hdd0/Kaggle/arc24/data/arc-agi_training_challenges.json'
    dataset_path = '/mnt/hdd0/Kaggle/arc24/data/arc-agi_evaluation_challenges.json'
    #dataset_path = '/kaggle/input/arc-prize-2024/arc-agi_evaluation_challenges.json'
    n_tasks = None # Optional parameter to limit the number of task in the inference, set it to None to use all the tasks
    # Few-shot
    few_shot_dataset_path = '/mnt/hdd0/Kaggle/arc24/data/arc-agi_evaluation_challenges.json'
    n_shots = 0
    # Inference params
    max_predictions_per_task = 2 # 
    sampling_params = dict(temperature=0.0, max_tokens=1000) # https://docs.vllm.ai/en/latest/dev/sampling_params.html

# %%
import os
is_dry_run = cfg.dataset_path == '/kaggle/input/arc-prize-2024/arc-agi_test_challenges.json' and not os.getenv('KAGGLE_IS_COMPETITION_RERUN')
if is_dry_run:
    print('This is a dry run, no inference nor installation of packages will be done')

# %% [markdown]
# ## Install

# %%
import vllm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
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

# %% [markdown]
# ## Code

# %% [markdown]
# ### Wait for gpu available

# %%
# import torch
# import time

# import logging

# for handler in logging.root.handlers[:]:
#     logging.root.removeHandler(handler)
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')

# def is_gpu_memory_available(required_memory=14):
#     for device in range(torch.cuda.device_count()):
#         available_memory = torch.cuda.mem_get_info(device)[0]/1024**3
#         logging.info(f'Available memory on GPU {device} is {available_memory:.1f} GB')
#         if available_memory < required_memory:
#             return False
#     return True

# def wait_for_gpu_memory(wait_time=60, required_memory=14):
#     while not is_gpu_memory_available(required_memory):
#         logging.info(f'Waiting for GPU memory to be available...')
#         time.sleep(wait_time)
#     logging.info(f'GPU memory is available. Let\'s go training!')
#     time.sleep(1) # wait a bit more to ensure the server is ready

# wait_for_gpu_memory(wait_time=360)

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

    def create_task_prompts(self, task):
        if cfg.model_path == '/kaggle/input/mistral/pytorch/7b-instruct-v0.1-hf/1':
            # Mistral does not have system prompt
            messages = []
        else:
            messages = [ 
                {"role": "system", "content": "You are a helpful AI assistant. Your task is to answer to the user using always the same transformation of the user input."}, 
            ] 
        for sample in task['train']:
            messages.append({"role": "user", "content": f"Input:\n{self.grid_encoder.to_text(sample['input'])}"})
            messages.append({"role": "assistant", "content": f"Output:\n{self.grid_encoder.to_text(sample['output'])}"})

        prompts = []
        for test_sample in task['test']:
            final_message = {"role": "user", "content": f"Input:\n{self.grid_encoder.to_text(test_sample['input'])}"}
            prompt = tokenizer.apply_chat_template(messages + [final_message],
                                                   tokenize=False,
                                                   add_generation_prompt=True)
            prompts.append(prompt)
        return prompts
    
    def parse_response(self, text):
        grid_text = text.split('Output:\n')[1]
        return self.grid_encoder.to_grid(grid_text)

# %%
puzzle_explanations = {
    '00576224': """The pattern of the input is repeated to generate the output.

1. The first two rows are obtained by simply repeating the 2x2 pattern 3 times along the cols axis.
2. The following two rows are obtained by flipping the pattern horizontally and repeating it 3 times
3. The final two rows are identical to the first ones, simply repeat the 2x2 pattern 3 times.

Thus the output is 3 times bigger than the input (6x6 vs 2x2) because the pattern is repeated 3 times in the row and col axis.""",
    '009d5c81': """To create the output we have to copy the input with two modifications:

1. The object with color 1 is removed and replaced with the background color 0
2. The color of the other object (there are only two objects in the grid) is modified.
  The new color of this object depends on the shape of the object of color 1. There is a mapping
  between shapes and colors. Just look at the train examples for an object of the same shape
  and see the color that is applied on the output.""",
    '00dbd492': """The input shows a square with color 2 that is empty except from a point in the center.
The output is created by colorizing the inside of the square. The color is chosen depending on the size of the squares.
The larger square is painted with 3, the medium with 4 and the small with 8.""",
    '03560426': """The input shows objects of different colors at the bottom of the grid.
The output is created by moving the objects to the top left corner. The objects are moved from left to right order.
The first object is placed at the top left corner, the second object is placed at the lower right corner of the first object,
the third object is placed at the lower right corner of the second object and so on. There is oclusion between the objects,
in those oclusions we see the rightmost object.""",
    '0607ce86': """This is a denoising task. The input shows the same object repeated many times, but there are noisy pixels in the grid.
The output is created by removing all the noise in the grid. The background should be completely 0.
The real object without noise can be guessed because there are many repetitions of the object, so we simply have to
look at the majority pixel on each location.""",
    '0692e18c': """The ouptut is created following this steps.

1. The input is upscaled x3. So if the input is 3x3 the output should be an upscaled version of the input 9x9
2. We apply an AND function in a sliding window fashion over the output using the inverted input pattern (take the input and swicth the background color 0 with the other color and viceversa)
    """,
    '070dd51e': """The output is created by simply drawing horizontal and vertical lines between cells with the same color.
If there is an intersection between lines the vertical line will be shown.""",
    '08573cc6': """The output is created by drawing an spiral that starts at the cell with color 1.
The colors of the spiral are taken from the first two cells of the grid, which will be removed in the output.""",
    '0a2355a6': """The output is created by copying the input and changing the color of the objects.
The new color will be chosen depending on the number of holes of the object. There is a mapping between number of holes and color that can be observed from the input examples.""",
}

# %%
class FewShotPromptCreator(PromptCreator):
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
    def __init__(self, grid_encoder):
        super().__init__(grid_encoder)
        with open(cfg.few_shot_dataset_path, 'r') as f:
            self.few_shot_tasks = json.load(f)
        with open(cfg.few_shot_dataset_path.replace('challenges.json', 'solutions.json'), 'r') as f:
            self.few_shot_solutions = json.load(f)
        self.few_shot_tasks = {task_id: self.few_shot_tasks[task_id] for task_id in puzzle_explanations}
        self.few_shot_solutions = {task_id: self.few_shot_solutions[task_id] for task_id in puzzle_explanations}
        self.few_shot_task_ids = list(self.few_shot_tasks.keys())
        self.n_shots = cfg.n_shots
    
    def create_task_prompts(self, task):
        messages = [{"role": "system", "content": self.task_description}]
        
        for task_id in np.random.choice(self.few_shot_task_ids, self.n_shots):
            few_shot_task = self.few_shot_tasks[task_id]
            user_message = self.create_user_message_for_train_examples(few_shot_task)
            for test_idx, test_sample in enumerate(few_shot_task['test']):
                user_message += self.create_input_message('Test case', test_sample)
                messages.append({"role": "user", "content": user_message})
                user_message = ''
                assistant_message = f'{puzzle_explanations[task_id]}\n\n' + self.create_output_message(self.few_shot_solutions[task_id][test_idx])
                messages.append({"role": "assistant", "content": assistant_message})

        user_message = self.create_user_message_for_train_examples(task)        
        prompts = []
        for test_sample in task['test']:
            user_message += self.create_input_message('Test case', test_sample)
            messages.append({"role": "user", "content": user_message})
            prompt = tokenizer.apply_chat_template(messages,
                                                   tokenize=False,
                                                   add_generation_prompt=True)
            prompts.append(prompt)
        return prompts
    
    def create_user_message_for_train_examples(self, task):
        user_message = "Let's see if you can solve this simple ARC task. These are some input-output grid examples that define the task.\n"
        for example_idx, sample in enumerate(task['train']):
            user_message += self.create_input_message(f'Example {example_idx}', sample)
            user_message += '\n' + self.create_output_message(sample['output'])
        return user_message

    def create_input_message(self, title, sample):
        return f"\n## {title}\n\n### Input\n\n{self.grid_encoder.to_text(sample['input'])}\n"
    
    def create_output_message(self, grid):
        return f"### Output\n\n{self.grid_encoder.to_text(grid)}\n"
    
    def parse_response(self, text):
        return self.grid_encoder.to_grid(text)

# %%
def print_sample_prompt(data, prompt_creator):
    prompts = [prompt_creator.create_task_prompts(task)[0] for task in data.values()]
    prompts = sorted(prompts, key=lambda x: len(x))
    pretty_print_prompt(prompts[0])
    
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
def plot_input_token_length_distribution(data, prompt_creator):
    prompts = []
    for task in data.values():
        prompts.extend(prompt_creator.create_task_prompts(task))
    token_length_distribution = [len(tokenizer.tokenize(prompt)) for prompt in tqdm(prompts)]
    plt.title('Prompt token length distribution')
    plt.hist(token_length_distribution)
    plt.xlabel('n tokens')

# %% [markdown]
# ### Model

# %%
if not is_dry_run:
    llm = LLM(model=cfg.model_path,
              trust_remote_code=True, 
              dtype='half', 
              tensor_parallel_size=2, # to use 2 gpus
              max_model_len=cfg.max_model_len,
            #   kv_cache_dtype='fp8_e5m2',
              enforce_eager=True, # without this 13.9GB of memory is used on each GPU, with this is 13.3GB,
              disable_log_stats=True,
            #   enable_lora=True,
            #   max_lora_rank=32,
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
    accuracy, correct_pixels, correct_size = [], [], []
    for task_id, task_ground_truth in ground_truth.items():
        task_metrics = []
        for idx, correct_grid in enumerate(task_ground_truth):
            predicted_grids = list(solutions[task_id][idx].values())
            predicted_grids = [grid for grid in predicted_grids if grid]
            
            task_metrics.append(evaluate_grid(correct_grid, predicted_grids))
            print_metrics(task_metrics[-1], f'{task_id}_{idx}')
        metrics.append(average_metrics(task_metrics))
    print('\n'*3 + '# Aggregated metrics:')
    print_metrics(average_metrics(metrics))
    save_metrics(metrics, solutions)
    print_metrics(average_metrics(metrics))
    return metrics
    
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
# TODO: lora
# lora_request = LoRARequest("learn-to-count", 1, '/mnt/hdd0/Kaggle/arc24/models/20240724_first_trainings/15_continue_training_phi3_4e5/checkpoint-22800')


def solve_task(task_id, task, prompt_creator, sampling_params):
    data_augmentation_params = product([False, True], [0, 1, 2, 3])
    solution = {task_id:[{"attempt_1": [], "attempt_2": []} for _ in task['test']]}
    texts = dict(prompts=[], responses=[], exceptions=[])
    for flip, n_rot90 in islice(data_augmentation_params, cfg.max_predictions_per_task):
        data_augmentation = DataAugmentation(flip, n_rot90)
        augmented_task = data_augmentation.augment_task(task)
        prompts = prompt_creator.create_task_prompts(augmented_task)
        outputs = llm.generate(prompts, sampling_params) #TODO:, lora_request=lora_request)
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
    for task_id, task in tqdm(data.items(), total=len(data), smoothing=0):
        task_solution, task_texts = solve_task(task_id, task, prompt_creator, sampling_params)
        solutions.update(task_solution)
        texts.update(task_texts)
    return solutions, texts

# %%
with open(cfg.dataset_path) as f:
    data = json.load(f)
if cfg.n_tasks is not None:
    data = dict(islice(data.items(), cfg.n_tasks))
print(f'There are {len(data)} tasks to solve.')

# %%
if not is_dry_run:
    prompt_creator = FewShotPromptCreator(GridCodeBlockEncoder(MinimalGridEncoder()))
    # prompt_creator = SimplePromptCreator(GridCodeBlockEncoder(MinimalGridEncoder()))
    # prompt_creator = SimplePromptCreator(GridCodeBlockEncoder(GridWithSeparationEncoder('|')))
    print_sample_prompt(data, prompt_creator)
    plot_input_token_length_distribution(data, prompt_creator)

# %%
if is_dry_run:
    with open('submission.json', 'w') as f:
        json.dump(dict(dry_run=True), f)
else:
    sampling_params = SamplingParams(**cfg.sampling_params)
    solutions, texts = inference(data, prompt_creator, sampling_params)
    with open('submission.json', 'w') as f:
        json.dump(solutions, f)

# %%
if not is_dry_run:
    number_of_predictions_per_task = analyze_number_of_predictions_per_task(data, texts)
    number_of_predictions_per_task

# %% [markdown]
# ## Evaluation

# %% [markdown]
# ### Visualization

# %%
ground_truth_path = cfg.dataset_path.replace('challenges.json', 'solutions.json')
if os.path.exists(ground_truth_path):
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    ground_truth = {key: ground_truth[key] for key in solutions}
    metrics = evaluate(ground_truth, solutions)
    
    with open('texts.json', 'w') as f:
        json.dump(texts, f)
    with open('number_of_predictions_per_task.json', 'w') as f:
        json.dump(number_of_predictions_per_task, f)

# %% [markdown]
# ### Just metrics

# %%
print_metrics(average_metrics(metrics))

# %% [markdown]
# ## Close

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

clear_vllm_gpu_memory()
