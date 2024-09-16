import numpy as np

def task_007bbfb7(grid):
    output = []
    output = np.repeat(grid, len(grid), axis=0)
    output = np.repeat(output, len(grid[0]), axis=1)

    mask = np.tile(grid, np.array(grid).shape) > 0
    output = output * mask
    return output
