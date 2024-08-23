import sys
import argparse
import json
from collections import defaultdict

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    with open(args.input_filepath) as f:
        solutions = json.load(f)
    solutions = select_most_voted_solutions(solutions, args.n)
    with open(args.output_filepath, 'w') as f:
        json.dump(solutions, f)


def select_most_voted_solutions(solutions, n):
    most_voted_solutions = dict()
    for task_id, task_solutions in solutions.items():
        most_voted_solutions[task_id] = list()
        for sample_solutions in task_solutions:
            unique_solutions, counts = get_unique_matrices_and_counts_sorted(list(sample_solutions.values()))
            most_voted_sample_solutions = {f'attempt_{i+1}': solution for i, solution in enumerate(unique_solutions[:n])}
            if len(most_voted_sample_solutions) < n:
                for i in range(len(most_voted_sample_solutions), n):
                    most_voted_sample_solutions[f'attempt_{i+1}'] = []
            most_voted_solutions[task_id].append(most_voted_sample_solutions)
    return most_voted_solutions


def get_unique_matrices_and_counts_sorted(matrices):
    # Dictionary to store counts of unique matrices
    matrix_count = defaultdict(int)

    # Convert each matrix to a tuple of tuples and count occurrences
    for matrix in matrices:
        matrix_tuple = tuple(tuple(row) for row in matrix)
        matrix_count[matrix_tuple] += 1

    # Sort the matrices by their counts in descending order
    sorted_matrices = sorted(matrix_count.items(), key=lambda item: item[1], reverse=True)

    # Convert the sorted tuples back to lists for easier interpretation
    unique_matrices = [list([list(row) for row in matrix]) for matrix, _ in sorted_matrices]
    counts = [count for _, count in sorted_matrices]

    return unique_matrices, counts

def parse_args(args):
    epilog = """
    """
    description = """
Creates a new solution by voting and choosing the most common solutions for each task.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('--input_filepath', required=True, help='Path to input json file with the solutions')
    parser.add_argument('--output_filepath', required=True, help='Path to output json file with the solutions')
    parser.add_argument('--n', type=int, default=2, help='Number of final solutions to keep')
    args = parser.parse_args(args)
    print(description)
    print(args)
    return args


if __name__ == '__main__':
    main()
