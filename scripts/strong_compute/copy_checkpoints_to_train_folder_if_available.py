import sys
import argparse
import glob
import os
import shutil


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    output_dir = os.path.realpath(args.output_dir)
    print(output_dir)
    src_path = '/root/models/' + output_dir.split('/models/')[-1]
    print(src_path)
    checkpoint_paths = sorted(glob.glob(f'{src_path}/checkpoint-*'), key=lambda x:int(x.split('checkpoint-')[-1]))
    print(f'Found {len(checkpoint_paths)} checkpoints')
    if checkpoint_paths:
        src_path = checkpoint_paths[-1]
        print(f'Source path: {src_path}')
        shutil.copytree(src_path, os.path.join(output_dir, os.path.basename(src_path)))


def parse_args(args):
    epilog = """
    """
    description = """
    If there are checkpoints from previous runs of the same training, they will be copied
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('output_dir', help='Directory where the output of the training will be saved')
    args = parser.parse_args(args)
    print(args)
    return args


if __name__ == '__main__':
    main()