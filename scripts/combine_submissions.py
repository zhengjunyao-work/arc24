import sys
import argparse
import json


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    with open(args.sub_1, 'r') as f:
        sub_1 = json.load(f)
    with open(args.sub_2, 'r') as f:
        sub_2 = json.load(f)

    combined_sub = combine_submissions(sub_1, sub_2, args.give_preference_to_second_submission_on_second_attempt)
    with open(args.output, 'w') as f:
        json.dump(combined_sub, f)


def combine_submissions(sub_1, sub_2, give_preference_to_second_submission_on_second_attempt):
    combined_sub = sub_1.copy()
    for task_id, values in combined_sub.items():
        for i, _ in enumerate(values):
            # If the first submission is empty, we take the second submission
            if not combined_sub[task_id][i]['attempt_1']:
                combined_sub[task_id][i] = sub_2[task_id][i]
            # If the second attempt is empty, we try to fill it with the second submission
            elif not combined_sub[task_id][i]['attempt_2'] or give_preference_to_second_submission_on_second_attempt:
                # Otherwise If the first attempt from second submission is valid and different from the first submission, we take the second submission as the second attempt
                if sub_2[task_id][i]['attempt_1'] and sub_2[task_id][i]['attempt_1'] != combined_sub[task_id][i]['attempt_1']:
                    combined_sub[task_id][i]['attempt_2'] = sub_2[task_id][i]['attempt_1']
                # Otherwise we take the second attempt from the second submission if it is different and valid
                elif sub_2[task_id][i]['attempt_2'] and sub_2[task_id][i]['attempt_2'] != combined_sub[task_id][i]['attempt_1']:
                    combined_sub[task_id][i]['attempt_2'] = sub_2[task_id][i]['attempt_2']
    return combined_sub


def parse_args(args):
    epilog = """
    """
    description = """
Creates a single submission by combining two submissions.

There are two modes of operation:

- The default mode will give total preference to the first submission, and will only use the second submission if the first submission is empty.
- If the flag --give_preference_to_second_submission_on_second_attempt is set, the second submission will be given preference on the second attempt.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('--sub_1', help='Path to json file with first submission', required=True)
    parser.add_argument('--sub_2', help='Path to json file with second submission', required=True)
    parser.add_argument('--output', help='Path to save the combined submission', required=True)
    parser.add_argument('--give_preference_to_second_submission_on_second_attempt', action='store_true', help='If True, the second submission will be given preference on the second attempt')
    args = parser.parse_args(args)
    print(args)
    return args


if __name__ == '__main__':
    main()
