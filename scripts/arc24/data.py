import os
import json


def load_arc_data_with_solutions(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    solutions_filepath = filepath.replace('challenges.json', 'solutions.json')
    if filepath != solutions_filepath and os.path.exists(solutions_filepath):
        with open(solutions_filepath, 'r') as f:
            solutions = json.load(f)
        for sample_id, task in data.items():
            for idx, sample in enumerate(task['test']):
                sample['output'] = solutions[sample_id][idx]
    verify_that_all_samples_have_output(data)
    return data


def verify_that_all_samples_have_output(data):
    for task in data.values():
        for samples in task.values():
            for sample in samples:
                if 'output' not in sample:
                    raise ValueError('Not all samples have output')

