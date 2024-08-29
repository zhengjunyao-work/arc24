import sys
import os
import json
import shutil
import argparse


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    train_conf_path = os.path.join(os.path.dirname(args.checkpoint_path), 'cfg.json')
    with open(train_conf_path, 'r') as f:
        cfg = json.load(f)
    print(cfg)
    model_path = merge_lora_with_model(args.checkpoint_path, cfg['model_path'])
    output_folder = args.checkpoint_path.replace('arc24/models', 'arc24/evaluations')
    output_filepath = inference(
        model_path, output_folder,
        cfg.get('grid_encoder', 'GridCodeBlockEncoder(MinimalGridEncoder())'),
        args.predictions_per_task)
    copy_train_conf(train_conf_path)
    evaluation(output_filepath)
    voting_output_filepath = voting(output_filepath)
    evaluation(voting_output_filepath)
    # v2 voting
    voting_output_filepath = voting(output_filepath.replace('.json', '_task_results.json'))
    evaluation(voting_output_filepath)
    print('-'*80)
    print('Done!\n\n\n')


def merge_lora_with_model(lora_path, model_path):
    print('-'*80)
    print(f'Merging lora with model {lora_path}')
    model_path = os.path.join('/home/gbarbadillo/data', os.path.basename(model_path))
    output_path = os.path.join('/home/gbarbadillo/data/temp_model')
    if os.path.exists(output_path):
        os.system(f'rm -r {output_path}')
    ret = os.system(f'python merge_lora.py --base_model_path {model_path} --lora_path {lora_path} --output_path {output_path}')
    if ret != 0:
        raise Exception('Error merging lora with model')
    return output_path


def inference(model_path, output_folder, grid_encoder, predictions_per_task):
    print('-'*80)
    print(f'Inference with model {model_path}')
    os.makedirs(output_folder, exist_ok=True)
    output_filepath = os.path.join(output_folder, f'inference_x{predictions_per_task}.json')
    if os.path.exists(output_filepath):
        print('Output file already exists, skipping inference')
        return output_filepath
    cmd = f'python inference.py --model_path {model_path} --output_filepath {output_filepath} --predictions_per_task {predictions_per_task} --grid_encoder "{grid_encoder}"'
    print(cmd)
    ret = os.system(cmd)
    if ret != 0:
        raise Exception('Error running inference')
    return output_filepath


def copy_train_conf(train_conf_path):
    shutil.copy(train_conf_path, train_conf_path.replace('arc24/models', 'arc24/evaluations'))


def evaluation(filepath):
    print('-'*80)
    print(f'Evaluating {filepath}')
    cmd = f'python evaluation.py {filepath}'
    print(cmd)
    ret = os.system(cmd)
    if ret != 0:
        raise Exception('Error running evaluation')


def voting(filepath):
    print('-'*80)
    print(f'Voting {filepath}')
    output_filepath = filepath.replace('.json', '_voting.json')
    cmd = f'python voting.py --input_filepath={filepath} --output_filepath={output_filepath}'
    print(cmd)
    ret = os.system(cmd)
    if ret != 0:
        raise Exception('Error running voting')
    return output_filepath


def parse_args(args):
    epilog = """
    """
    description = """
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('checkpoint_path', help='Path to folder with the checkpoint that we want to evaluate')
    parser.add_argument('--predictions_per_task', type=int, default=64,
                        help="Number of predictions per task, use a multiple of 8")
    args = parser.parse_args(args)
    print(args)
    return args


if __name__ == '__main__':
    main()
