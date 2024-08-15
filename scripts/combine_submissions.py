import sys
import argparse
import json

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    combine_submissions(args.sub_1, args.sub_2, args.output)


def combine_submissions(sub_1, sub_2, output):
    with open(sub_1, 'r') as f:
        combined_sub = json.load(f)
    with open(sub_2, 'r') as f:
        sub_2 = json.load(f)

    for task_id, values in combined_sub.items():
        for sample_idx, _ in enumerate(values):
            if sub_2[task_id][sample_idx]['attempt_1'] and sub_2[task_id][sample_idx]['attempt_1'] != combined_sub[task_id][sample_idx]['attempt_1']:
                combined_sub[task_id][sample_idx]['attempt_2'] = sub_2[task_id][sample_idx]['attempt_1']
            elif sub_2[task_id][sample_idx]['attempt_2'] and sub_2[task_id][sample_idx]['attempt_2'] != combined_sub[task_id][sample_idx]['attempt_1']:
                combined_sub[task_id][sample_idx]['attempt_2'] = sub_2[task_id][sample_idx]['attempt_2']

    with open(output, 'w') as f:
        json.dump(combined_sub, f)


def parse_args(args):
    epilog = """
    """
    description = """
Creates a single submission taking the first attempt of each submission

Please give as a first option the 2020 solution, because it gives preference to the first submission.
It will verify that the second submission creates valid grids and that they are different from the first submission.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('--sub_1', help='Path to json file with first submission', required=True)
    parser.add_argument('--sub_2', help='Path to json file with second submission', required=True)
    parser.add_argument('--output', help='Path to save the combined submission', required=True)
    args = parser.parse_args(args)
    print(args)
    return args


if __name__ == '__main__':
    main()
