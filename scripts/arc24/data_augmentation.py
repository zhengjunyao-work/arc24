import numpy as np
import random
from functools import partial
from itertools import product


def apply_data_augmentation(task, hflip, n_rot90, color_map=None):
    augmented_task = _apply_augmentation_to_task(task, partial(geometric_augmentation, hflip=hflip, n_rot90=n_rot90))
    if color_map is not None:
        augmented_task = swap_task_colors(augmented_task, color_map)
    augmented_task = permute_train_samples(augmented_task, max_permutations=1)[0]
    return augmented_task


def revert_data_augmentation(grid, hflip, n_rot90, color_map=None):
    grid = revert_geometric_augmentation(grid, hflip, n_rot90)
    if color_map is not None:
        grid = revert_color_swap(grid, color_map)
    return grid


def random_augment_task(task):
    task = swap_task_colors(task)
    task = _apply_augmentation_to_task(task, partial(geometric_augmentation,
                                                     hflip=random.choice([True, False]),
                                                     n_rot90=random.choice([0, 1, 2, 3])))
    task = permute_train_samples(task, max_permutations=1)[0]
    task = random_swap_train_and_test(task)
    return task


def _apply_augmentation_to_task(task, augmentation):
    augmented_task = dict()
    for partition, samples in task.items():
        augmented_task[partition] = [{name:augmentation(grid) for name,grid in sample.items()} for sample in samples]
    return augmented_task


def geometric_augmentation(grid, hflip, n_rot90):
    grid = np.array(grid)
    if hflip:
        grid = np.flip(grid, axis=1)
    grid = np.rot90(grid, k=n_rot90)
    return grid.tolist()


def revert_geometric_augmentation(grid, hflip, n_rot90):
    grid = np.array(grid)
    grid = np.rot90(grid, k=-n_rot90)
    if hflip:
        grid = np.flip(grid, axis=1)
    return grid.tolist()


def swap_task_colors(task, color_map=None, change_background_probability=0.1):
    if color_map is None:
        color_map = get_random_color_map(change_background_probability)
    vectorized_mapping = np.vectorize(color_map.get)

    new_task = dict()
    for key in task.keys():
        new_task[key] = [{name:vectorized_mapping(grid) for name, grid in sample.items()} for sample in task[key]]
    return new_task


def revert_color_swap(grid, color_map):
    reverse_color_map = {v: k for k, v in color_map.items()}
    vectorized_mapping = np.vectorize(reverse_color_map.get)
    return vectorized_mapping(grid).tolist()


def get_random_color_map(change_background_probability):
    colors = list(range(10))
    if random.random() < change_background_probability:
        new_colors = list(range(10))
        random.shuffle(new_colors)
    else:
        new_colors = list(range(1, 10))
        random.shuffle(new_colors)
        new_colors = [0] + new_colors

    color_map = {x: y for x, y in zip(colors, new_colors)}
    return color_map


def permute_train_samples(task, max_permutations=6):
    augmented_tasks = []
    for _ in range(max_permutations):
        train_order = np.arange(len(task['train']))
        np.random.shuffle(train_order)
        augmented_task = dict()
        augmented_task['train'] = [task['train'][idx] for idx in train_order]
        augmented_task['test'] = task['test']
        augmented_tasks.append(augmented_task)
    return augmented_tasks


def random_swap_train_and_test(task):
    augmented_task = task.copy()
    train_idx = np.random.randint(len(task['train']))
    test_idx = np.random.randint(len(task['test']))
    augmented_task['train'] = task['train'][:train_idx] + [task['test'][test_idx]] + task['train'][train_idx+1:]
    augmented_task['test'] = task['test'][:test_idx] + [task['train'][train_idx]] + task['test'][test_idx+1:]
    return augmented_task


if __name__ == '__main__':
    sample_grid = np.eye(3, dtype=int).tolist()
    for flip in [True, False]:
        for n_rot90 in range(4):
            assert sample_grid == revert_geometric_augmentation(geometric_augmentation(sample_grid, flip, n_rot90), flip, n_rot90)
