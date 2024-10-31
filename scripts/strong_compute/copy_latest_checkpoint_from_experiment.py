import sys
import argparse
import glob
import shutil


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    checkpoint_paths = sorted(glob.glob(f'/root/exports/{args.experiment}/outputs/models/*/*/checkpoint-*'), key=lambda x:int(x.split('checkpoint-')[-1]))
    print(f'Found {len(checkpoint_paths)} checkpoints')
    if checkpoint_paths:
        src_path = checkpoint_paths[-1]
        print(f'Source path: {src_path}')
        dst_path = '/root/models/' + src_path.split('/models/')[-1]
        print(f'Destination path: {dst_path}')
        shutil.copytree(src_path, dst_path)
        print('Done!')


def parse_args(args):
    epilog = """
    """
    description = """
    Searches all the checkpoints from a experiment and copies the last one to the models directory.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('experiment', help='Experiment identifier')
    args = parser.parse_args(args)
    print(args)
    return args


if __name__ == '__main__':
    main()
