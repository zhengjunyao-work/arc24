import numpy as np
from primitives import *

def task_007bbfb7(grid):
    output = []
    output = np.repeat(grid, len(grid), axis=0)
    output = np.repeat(output, len(grid[0]), axis=1)

    mask = np.tile(grid, np.array(grid).shape) > 0
    output = output * mask
    return output


def task_00d62c1b(grid):
    holes = detect_holes(grid)
    output = change_color(grid, holes, color=4)
    return output


def task_017c7c7b(grid):
    output = apply_color_map(grid, color_map={1: 2})
    output = np.vstack((output, output[:3]))
    # output = np.tile(output, (2, 1))
    # TODO: each column is a series with a different period, we have to continue each series
    return output

def task_025d127b(grid):
    objects = detect_objects(grid)
    new_objects = []
    for object in objects:
        new_objects.append(deform_object(object, part='top', direction='right', amount=1))
    output = draw_objects(np.zeros_like(grid), new_objects)
    return output
