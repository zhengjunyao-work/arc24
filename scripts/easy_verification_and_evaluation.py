import sys
import os
import json
import shutil
import argparse
import hashlib

from arc24.logging import logging
from easy_inference_and_evaluation import (
    merge_lora_with_model,
    evaluation
)

logger = logging.getLogger(__name__)


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    train_conf_path = os.path.join(os.path.dirname(args.checkpoint_path), 'cfg.json')
    with open(train_conf_path, 'r') as f:
        cfg = json.load(f)
    print(cfg)
    output_filepath = args.predictions_path.replace(
        '.json', f'_m{generate_small_hash(args.checkpoint_path)}_v{args.verifications_per_prediction:03d}_verification.json')
    logger.info(f'Output file path: {output_filepath}')
    if not os.path.exists(output_filepath):
        model_path = merge_lora_with_model(args.checkpoint_path, cfg['model_path'])
        verification(
            model_path, output_filepath,
            verifications_per_prediction=args.verifications_per_prediction,
            dataset_path=args.dataset_path,
            predictions_path=args.predictions_path)
    else:
        print('Output file already exists, skipping inference')
    evaluation(output_filepath, args.dataset_path)
    # TODO: add vote 1 evaluation


def verification(model_path, output_filepath, verifications_per_prediction,
                 dataset_path, predictions_path):
    print('-'*80)
    logger.info(f'Verification with model {model_path}')
    cmd = f'python verify_predictions.py --model-path {model_path} --output-path {output_filepath}'
    cmd += f' --verifications-per-prediction {verifications_per_prediction}'
    cmd += f' --dataset-path {dataset_path} --predictions-path {predictions_path}'
    # cmd += ' --random-seed 7' # TODO: remove the random seed
    print(cmd)
    os.system(cmd)
    if not os.path.exists(output_filepath):
        raise Exception('Output file not found, error running inference')




def generate_small_hash(input_string: str) -> str:
    # Create an MD5 hash object
    hash_object = hashlib.md5(input_string.encode())
    # Convert the hash object to a hexadecimal string
    return hash_object.hexdigest()[:8]  # Slice to get a shorter hash (first 8 characters)



def parse_args(args):
    epilog = """
Alternative datasets for evaluation:

--dataset_path /mnt/hdd0/Kaggle/arc24/data/arc-agi_evaluation_challenges.json
    """
    description = "Runs inference and evaluation on a given checkpoint"
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('checkpoint_path', help='Path to folder with the checkpoint that we want to use to verify predictions')
    parser.add_argument('--verifications-per-prediction', type=int, default=8,
                        help="Number of verifications per prediction")
    parser.add_argument('--dataset-path', type=str, help="Path to the dataset to make inference and evaluation",
                        default='/mnt/hdd0/Kaggle/arc24/data/new_partitions/arc-agi_all_challenges.json')
    parser.add_argument('--predictions-path', type=str, help="Path to the json file with the predictions")
    args = parser.parse_args(args)
    print(args)
    return args


if __name__ == '__main__':
    main()
