from typing import Optional
import argparse
from dataclasses import dataclass, asdict



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
    max_predictions_per_task: int = 8 #
    best_of: int = 1 # controls the number of beams used in beam search

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Configuration")
    parser.add_argument('--model_path', type=str, help="Path to the model")
    parser.add_argument('--dataset_path', type=str, help="Path to the dataset to make inference")
    parser.add_argument('--n_tasks', type=int, help="If given only the first n tasks will be evaluated")
    parser.add_argument('--output_filepath', type=str, help="Path to the json file with the predictions")
    parser.add_argument('--max_predictions_per_task', type=int, help="Max number of predictions per task")
    parser.add_argument('--best_of', type=int, help="controls the number of beams used in beam search")
    return parser.parse_args()


# Override default configuration using arguments
args = parse_args()
cfg = CFG(**{k: v for k, v in vars(args).items() if v is not None})
print(asdict(cfg))



import json
import os
from tqdm.auto import tqdm
from itertools import islice, product

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from arc24.data_augmentation import DataAugmentation
from arc24.prompting import SimplePromptCreator, print_sample_prompt
from arc24.encoders import GridCodeBlockEncoder, MinimalGridEncoder

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info('Started logging')


def solve_task(task_id, task, prompt_creator, sampling_params):
    data_augmentation_params = product([False, True], [0, 1, 2, 3])
    solution = {task_id:[{f"attempt_{i}": [] for i in range(cfg.max_predictions_per_task)} for _ in task['test']]}

    prompts, data_augmentations = [], []
    for attempt_idx, (flip, n_rot90) in islice(enumerate(data_augmentation_params), cfg.max_predictions_per_task):
        data_augmentation = DataAugmentation(flip, n_rot90)
        augmented_task = data_augmentation.augment_task(task)
        prompts.extend(prompt_creator.create_task_prompts(augmented_task))
        data_augmentations.append(data_augmentation)
    # group all the prompts in a batch for faster inference
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    n_test = len(task['test'])

    texts = dict(prompts=prompts, responses=[], exceptions=[])
    for idx, output in enumerate(outputs):
        response = output.outputs[0].text
        test_idx = idx % n_test
        attempt_idx = idx // n_test
        data_augmentation = data_augmentations[attempt_idx]
        try:
            augmented_grid = prompt_creator.parse_response(response)
            grid = data_augmentation.revert_augmentation(augmented_grid)
            solution[task_id][test_idx][f"attempt_{attempt_idx}"] = grid
        except Exception as e:
            print(f'Exception when parsing response from {task_id}: {e}')
            texts['exceptions'].append(str(e))
        texts['responses'].append(response)
    return solution, {task_id:texts}


def inference(data, prompt_creator, sampling_params):
    solutions, texts = dict(), dict()
    for idx, (task_id, task) in tqdm(enumerate(data.items()), total=len(data), desc='Solving tasks', smoothing=0):
        logging.info(f'Solving {task_id}, {idx+1}/{len(data)}')
        task_solution, task_texts = solve_task(task_id, task, prompt_creator, sampling_params)
        solutions.update(task_solution)
        texts.update(task_texts)
    return solutions, texts


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


with open(cfg.dataset_path) as f:
    data = json.load(f)
if cfg.n_tasks is not None and cfg.n_tasks > 0:
    data = dict(islice(data.items(), cfg.n_tasks))
print(f'There are {len(data)} tasks to solve.')


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


prompt_creator = SimplePromptCreator(GridCodeBlockEncoder(MinimalGridEncoder()), tokenizer, cfg.model_path)
print_sample_prompt(data, prompt_creator)

if cfg.best_of == 1:
    # # https://docs.vllm.ai/en/latest/dev/sampling_params.html
    print('Using greedy search')
    sampling_params = SamplingParams(n=1, temperature=0.0, max_tokens=1000)
else:
    print(f'Using beam search with best_of={cfg.best_of}')
    sampling_params = SamplingParams(n=1, temperature=0.0, max_tokens=1000, use_beam_search=True, best_of=cfg.best_of)

solutions, texts = inference(data, prompt_creator, sampling_params)
with open(cfg.output_filepath, 'w') as f:
    json.dump(solutions, f)
with open('texts.json', 'w') as f:
    json.dump(texts, f)

clear_vllm_gpu_memory()