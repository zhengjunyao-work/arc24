import sys
import argparse
import json
from tqdm.auto import tqdm
import numpy as np

from vllm import LLM
from transformers import AutoTokenizer


from inference import (
    clear_vllm_gpu_memory,
    get_sampling_params,
    generate_outputs_with_batches
)
from voting import get_unique_matrices_and_counts_sorted
from arc24.encoders import create_grid_encoder
from arc24.data_augmentation import (
    get_random_geometric_augmentation_params,
    get_random_color_map,
    apply_data_augmentation
)
from arc24.prompting import create_prompts_from_task
from arc24.logging import log_execution_time, logging

logger = logging.getLogger(__name__)

@log_execution_time
def main():
    cfg = parse_args()
    with open(cfg.dataset_path, 'r') as f:
        dataset = json.load(f)
    with open(cfg.predictions_path, 'r') as f:
        predictions = json.load(f)
    unique_predictions = leave_only_unique_predictions(predictions)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    grid_encoder = create_grid_encoder(cfg.grid_encoder)
    llm = LLM(
        model=cfg.model_path,
        trust_remote_code=True,
        dtype='half',
        tensor_parallel_size=2, # to use 2 gpus
        max_model_len=cfg.max_model_len,
        #kv_cache_dtype='fp8_e5m2', I have disabled kv cache quantization because it is hurtful
        enforce_eager=True, # without this 13.9GB of memory is used on each GPU, with this is 13.3GB,
        disable_log_stats=True,
        max_num_seqs=255, # default is supposed to be 256 I have used it to solve some weird illegal memory error
        swap_space=cfg.swap_space, # CPU swap space size (GiB) per GPU, has great influence on RAM but I haven't noticed any performance difference
    )
    sampling_params = get_sampling_params(best_of=1, temperature=cfg.temperature, n=1, max_output_tokens=cfg.max_output_tokens)
    prompts = create_prompts(
        unique_predictions, dataset, grid_encoder, tokenizer,
        prompt_version=cfg.prompt_version, verifications_per_prediction=cfg.verifications_per_prediction)
    outputs = generate_outputs_with_batches(llm, prompts, sampling_params, batch_size=cfg.batch_size)
    aggregated_verifications = aggregate_verification_predictions(outputs, prompts, unique_predictions)
    selected_predictions = select_predictions_with_verifications(unique_predictions, aggregated_verifications, cfg.n_top)

    with open(cfg.output_path, 'w') as f:
        json.dump(selected_predictions, f, indent=4)
    # TODO: save richer information for later analysis

    del llm.llm_engine.model_executor
    del llm
    clear_vllm_gpu_memory()


def parse_args(args=None):
    if args is None:
        args = sys.argv[1:]
    epilog = """
    """
    description = """
Creates a ranking of the predictions using a model that verifies if a prediction is correct or not.
This works because verifying that a prediction is correct is an easier task than making the prediction itself.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('--model-path', type=str, help="Path to the verifier model")
    parser.add_argument('--max-model-len', default=10240,
                        type=int, help="Maximum number of tokens in the model")
    parser.add_argument('--grid-encoder', default='GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))',
                        type=str, help="Name of the grid encoder")
    parser.add_argument('--prompt-version', default='verify-output-from-examples-v0',
                        type=str, help="Prompt version")
    parser.add_argument('--dataset-path', type=str, help="Path to the dataset to make inference")
    parser.add_argument('--predictions-path', type=str, help="Path to the json file with the predictions")
    parser.add_argument('--output-path', type=str, help="Path to the json file with the predictions")
    parser.add_argument('--verifications-per-prediction', default=8, type=int, help="Number of verifications per prediction")
    parser.add_argument('--temperature', default=0, type=float, help="temperature for sampling, 0.0 for greedy search")
    parser.add_argument('--batch-size', default=512, type=int, help="batch size for inference")
    parser.add_argument('--max-output-tokens', default=5, type=int, help="Maximum number of tokens to generate")
    parser.add_argument('--random-seed', default=None, type=int, help="Random seed for data augmentation")
    parser.add_argument('--swap-space', default=0, type=int, help="CPU swap space size (GiB) per GPU")
    parser.add_argument('--verbose', action='store_true', help="Print verbose output")
    parser.add_argument('--n-top', default=2, type=int, help="Number of top predictions to select")
    print(args)
    return parser.parse_args()


def leave_only_unique_predictions(predictions):
    unique_predictions = dict()
    for task_id, task_predictions in predictions.items():
        unique_predictions[task_id] = []
        for sample_predictions in task_predictions:
            sample_predictions = list(sample_predictions.values())
            sample_predictions = [prediction for prediction in sample_predictions if prediction]
            sample_predictions, _ = get_unique_matrices_and_counts_sorted(sample_predictions)
            unique_predictions[task_id].append(sample_predictions)
    return unique_predictions


def create_prompts(predictions, dataset, grid_encoder, tokenizer, prompt_version, verifications_per_prediction):
    prompts = []
    for task_id, task_predictions in tqdm(predictions.items(), total=len(predictions), desc='Creating prompts'):
        for sample_idx, sample_predictions in enumerate(task_predictions):
            for prediction_idx, prediction in enumerate(sample_predictions):
                for _ in range(verifications_per_prediction):
                    task = dataset[task_id].copy()
                    task['test'] = [dict(input=task['test'][sample_idx]['input'], output=prediction)]
                    data_augmentation_kwargs = get_random_geometric_augmentation_params()
                    data_augmentation_kwargs['color_map'] = get_random_color_map(change_background_probability=0.1)
                    augmented_task = apply_data_augmentation(task, **data_augmentation_kwargs)
                    augmented_task['test_output'] = augmented_task['test'][0]['output']
                    prompt = create_prompts_from_task(
                        augmented_task, grid_encoder=grid_encoder, tokenizer=tokenizer,
                        is_train_prompt=False, prompt_version=prompt_version)[0]
                    prompts.append(dict(task_id=task_id,
                                        data_augmentation_kwargs=data_augmentation_kwargs,
                                        prompt=prompt,
                                        sample_idx=sample_idx,
                                        prediction_idx=prediction_idx))
    logger.info(f'Created {len(prompts)} prompts')
    return prompts


def aggregate_verification_predictions(outputs, prompts, unique_predictions):
    text_outputs = [output.outputs[0].text for output in outputs]
    logger.info(f'Unique text outputs: {np.unique(text_outputs, return_counts=True)}')
    verifications = [output == 'yes' for output in text_outputs]
    print(np.unique(verifications, return_counts=True))
    aggregated_verifications = {task_id: [np.zeros(len(sample_predictions)) for sample_predictions in task_predictions] for task_id, task_predictions in unique_predictions.items()}
    for verification, prompt in zip(verifications, prompts):
        if verification:
            aggregated_verifications[prompt['task_id']][prompt['sample_idx']][prompt['prediction_idx']] += 1
    return aggregated_verifications


def select_predictions_with_verifications(unique_predictions, aggregated_verifications, n):
    selected_predictions = dict()
    for task_id, task_predictions in unique_predictions.items():
        selected_predictions[task_id] = []
        for sample_predictions, sample_verifications in zip(task_predictions, aggregated_verifications[task_id]):
            ranking = np.argsort(sample_verifications)[::-1][:n]
            selected_predictions[task_id].append({f'attempt_{attempt_idx}': sample_predictions[idx] for attempt_idx, idx in enumerate(ranking, 1)})
    return selected_predictions


if __name__ == '__main__':
    main()
