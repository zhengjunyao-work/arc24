import numpy as np
from functools import partial


def apply_data_augmentation(task, hflip, n_rot90):
    augmented_task = _apply_augmentation_to_task(task, partial(geometric_augmentation, hflip=hflip, n_rot90=n_rot90))
    return augmented_task


def revert_data_augmentation(grid, hflip, n_rot90):
    return revert_geometric_augmentation(grid, hflip, n_rot90)


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


sample_grid = np.eye(3, dtype=int).tolist()
for flip in [True, False]:
    for n_rot90 in range(4):
        assert sample_grid == revert_geometric_augmentation(geometric_augmentation(sample_grid, flip, n_rot90), flip, n_rot90)

