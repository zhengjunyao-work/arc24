import numpy as np
import random
from functools import partial


def apply_data_augmentation(task, hflip, n_rot90, color_map=None):
    augmented_task = _apply_augmentation_to_task(task, partial(geometric_augmentation, hflip=hflip, n_rot90=n_rot90))
    if color_map is not None:
        augmented_task = swap_task_colors(augmented_task, color_map)
    augmented_task = permute_train_samples(augmented_task)
    return augmented_task


def revert_data_augmentation(grid, hflip, n_rot90, color_map=None):
    grid = revert_geometric_augmentation(grid, hflip, n_rot90)
    if color_map is not None:
        grid = revert_color_swap(grid, color_map)
    return grid


def random_augment_task(task):
    augmented_task = apply_data_augmentation(
        task, color_map=get_random_color_map(), **get_random_geometric_augmentation_params())
    augmented_task = random_swap_train_and_test(augmented_task)
    return augmented_task


def random_compose_new_task_by_adding_additional_transformation(task, augmentation_target=None):
    """
    Creates a new task by randomly applying transformations to the inputs or the outputs

    Parameters
    ----------
    task : dict
        The task to be transformed
    augmentation_target : str
        The target of the transformation. Either 'input' or 'output'
    """
    if augmentation_target is None:
        augmentation_target = random.choice(['input', 'output'])
    # new_task = _apply_augmentation_to_task(
    #     task,
    #     partial(geometric_augmentation, **get_random_geometric_augmentation_params()),
    #     augmentation_target=augmentation_target)
    # new_task = _apply_augmentation_to_task(
    #     task,
    #     partial(add_padding, **get_random_padding_params()),
    #     augmentation_target=augmentation_target)
    new_task = _apply_augmentation_to_task(
        task,
        partial(upscale, **get_random_upscale_params()),
        augmentation_target=augmentation_target)
    # TODO: finish this function
    return new_task


def _apply_augmentation_to_task(task, augmentation, augmentation_target=None):
    augmented_task = dict()
    for partition, samples in task.items():
        augmented_task[partition] = [_augment_sample(sample, augmentation, augmentation_target) for sample in samples]
    return augmented_task


def _augment_sample(sample, augmentation, augmentation_target=None):
    if augmentation_target is None:
        return {name:augmentation(grid) for name, grid in sample.items()}
    else:
        if augmentation_target not in sample:
            raise ValueError(f"augmentation_target {augmentation_target} not found in sample")
        return {name:augmentation(grid) if name == augmentation_target else grid for name, grid in sample.items()}


def get_random_geometric_augmentation_params():
    return dict(hflip=random.choice([True, False]), n_rot90=random.choice([0, 1, 2, 3]))


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


def get_random_color_map(change_background_probability=0.1):
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


def permute_train_samples(task):
    train_order = np.arange(len(task['train']))
    np.random.shuffle(train_order)
    augmented_task = dict()
    augmented_task['train'] = [task['train'][idx] for idx in train_order]
    augmented_task['test'] = task['test']
    return augmented_task


def random_swap_train_and_test(task):
    augmented_task = task.copy()
    train_idx = np.random.randint(len(task['train']))
    test_idx = np.random.randint(len(task['test']))
    augmented_task['train'] = task['train'][:train_idx] + [task['test'][test_idx]] + task['train'][train_idx+1:]
    augmented_task['test'] = task['test'][:test_idx] + [task['train'][train_idx]] + task['test'][test_idx+1:]
    return augmented_task


def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)


def add_padding(grid, color, size):
    rows, cols = len(grid), len(grid[0])
    padded_grid = [[color]*(cols + size*2) for _ in range(size)]
    for row in grid:
        padded_grid.append([color]*size + row + [color]*size)
    padded_grid += [[color]*(cols + size*2) for _ in range(size)]
    return padded_grid


def get_random_padding_params():
    # TODO: verify that the grid won't be too big
    return dict(color=random.randint(0, 9), size=random.randint(1, 5))


def upscale(grid, scale):
    if isinstance(scale, int):
        scale = (scale, scale)
    grid = np.array(grid, dtype=int)
    for axis, scale in enumerate(scale):
        grid = np.repeat(grid, scale, axis=axis)
    return grid.tolist()

def get_random_upscale_params():
    # TODO: verify that the grid won't be too big
    if random.random() < 0.5:
        return dict(scale=random.randint(2, 4))
    else:
        return dict(scale=(random.randint(2, 4), random.randint(2, 4)))


if __name__ == '__main__':
    sample_grid = np.eye(3, dtype=int).tolist()
    for flip in [True, False]:
        for n_rot90 in range(4):
            assert sample_grid == revert_geometric_augmentation(geometric_augmentation(sample_grid, flip, n_rot90), flip, n_rot90)
