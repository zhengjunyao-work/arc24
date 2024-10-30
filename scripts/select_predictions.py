import sys
import argparse
import json
import random
from tqdm.auto import tqdm
import numpy as np

from inference import (
    clear_vllm_gpu_memory,
    generate_outputs_with_batches
)
from arc24.data_augmentation import (
    get_random_geometric_augmentation_params,
    get_random_color_map,
    apply_data_augmentation
)
from arc24.prompting import create_prompts_from_task
from arc24.logging import log_execution_time, logging
from verify_predictions import (
    load_data,
    create_inference_artifacts,
    select_predictions_with_verifications,
    create_rich_output,
    create_empty_aggregated_verifications
)

logger = logging.getLogger(__name__)

@log_execution_time
def main():
    cfg = parse_args()
    dataset, unique_predictions = load_data(cfg.dataset_path, cfg.predictions_path)
    # unique_predictions = {key: unique_predictions[key] for key in list(unique_predictions.keys())[:10]}
    aggregated_verifications = create_empty_aggregated_verifications(unique_predictions)
    tokenizer, grid_encoder, llm, sampling_params = create_inference_artifacts(cfg)
    for round_idx in range(cfg.n_rounds):
        prompts = create_prompts(
            unique_predictions, dataset, grid_encoder, tokenizer, prompt_version=cfg.prompt_version)
        logger.info(f'Round {round_idx+1}/{cfg.n_rounds}: {len(prompts)} prompts')
        if not prompts:
            break
        outputs = generate_outputs_with_batches(llm, prompts, sampling_params, batch_size=cfg.batch_size)
        aggregated_verifications = update_aggregate_verification_predictions(outputs, prompts, aggregated_verifications)
    selected_predictions = select_predictions_with_verifications(unique_predictions, aggregated_verifications, cfg.n_top)

    with open(cfg.output_path, 'w') as f:
        json.dump(selected_predictions, f, indent=4)
    rich_output = create_rich_output(unique_predictions, aggregated_verifications)
    with open(cfg.output_path.replace('.json', '_rich_output.json'), 'w') as f:
        json.dump(rich_output, f, indent=4)

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
    parser.add_argument('--prompt-version', default='select-output-from-examples-v0',
                        type=str, help="Prompt version")
    parser.add_argument('--dataset-path', type=str, help="Path to the dataset to make inference")
    parser.add_argument('--predictions-path', type=str, help="Path to the json file with the predictions")
    parser.add_argument('--output-path', type=str, help="Path to the json file with the predictions")
    parser.add_argument('--temperature', default=0, type=float, help="temperature for sampling, 0.0 for greedy search")
    parser.add_argument('--batch-size', default=512, type=int, help="batch size for inference")
    parser.add_argument('--max-output-tokens', default=5, type=int, help="Maximum number of tokens to generate")
    parser.add_argument('--random-seed', default=None, type=int, help="Random seed for data augmentation")
    parser.add_argument('--swap-space', default=0, type=int, help="CPU swap space size (GiB) per GPU")
    parser.add_argument('--verbose', action='store_true', help="Print verbose output")
    parser.add_argument('--n-top', default=2, type=int, help="Number of top predictions to select")
    parser.add_argument('--n-rounds', default=4, type=int, help="Number of all vs all rounds to select predictions")
    print(args)
    return parser.parse_args()


def create_prompts(predictions, dataset, grid_encoder, tokenizer, prompt_version):
    """ Creates prompt doing an all vs all comparison of the predictions """
    prompts = []
    for task_id, task_predictions in predictions.items():
        for sample_idx, sample_predictions in enumerate(task_predictions):
            for prediction_idx, prediction in enumerate(sample_predictions):
                for other_prediction_idx, other_prediction in enumerate(sample_predictions[prediction_idx + 1:], prediction_idx + 1):
                    task = dataset[task_id].copy()
                    task['test'] = [dict(input=task['test'][sample_idx]['input'],
                                         output_1=prediction,
                                         output_2=other_prediction)]
                    data_augmentation_kwargs = get_random_geometric_augmentation_params()
                    data_augmentation_kwargs['color_map'] = get_random_color_map(change_background_probability=0.1)
                    augmented_task = apply_data_augmentation(task, **data_augmentation_kwargs)
                    if random.random() < 0.5:
                        augmented_task['test_output_choices'] = [
                            augmented_task['test'][0]['output_1'],
                            augmented_task['test'][0]['output_2'],
                        ]
                        prediction_indices=[prediction_idx, other_prediction_idx]
                    else:
                        augmented_task['test_output_choices'] = [
                            augmented_task['test'][0]['output_2'],
                            augmented_task['test'][0]['output_1'],
                        ]
                        prediction_indices=[other_prediction_idx, prediction_idx]
                    prompt = create_prompts_from_task(
                        augmented_task, grid_encoder=grid_encoder, tokenizer=tokenizer,
                        is_train_prompt=False, prompt_version=prompt_version)[0]
                    prompts.append(dict(task_id=task_id,
                                        data_augmentation_kwargs=data_augmentation_kwargs,
                                        prompt=prompt,
                                        sample_idx=sample_idx,
                                        prediction_indices=prediction_indices))
    return prompts


def update_aggregate_verification_predictions(outputs, prompts, aggregated_verifications):
    text_outputs = [output.outputs[0].text for output in outputs]
    logger.info(f'Unique text outputs: {np.unique(text_outputs, return_counts=True)}')
    for output, prompt in zip(text_outputs, prompts):
        if output == '1':
            aggregated_verifications[prompt['task_id']][prompt['sample_idx']][prompt['prediction_indices'][0]].update(True)
            aggregated_verifications[prompt['task_id']][prompt['sample_idx']][prompt['prediction_indices'][1]].update(False)
        elif output == '2':
            aggregated_verifications[prompt['task_id']][prompt['sample_idx']][prompt['prediction_indices'][0]].update(False)
            aggregated_verifications[prompt['task_id']][prompt['sample_idx']][prompt['prediction_indices'][1]].update(True)
        else:
            logger.warning(f'Invalid output: {output}')
    return aggregated_verifications


if __name__ == '__main__':
    main()
