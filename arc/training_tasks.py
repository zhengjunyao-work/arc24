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

# def task_045e512c(grid):
#     objects = detect_objects(grid)
#     objects = sort_objects_by_size(objects, ascending=False)
#     biggest_object = objects[0]
#     output = draw_objects(np.zeros_like(grid), [biggest_object])
#     for other_object in objects[1:]:
#         biggest_object = merge_objects(biggest_object, other_object)

def task_0520fde7(grid):
    side_1, side_2 = split_grid(grid, axis=1)
    output = np.minimum(side_1, side_2)
    output = apply_color_map(output, color_map={1: 2})
    return output

def task_05269061(grid):
    diagonal_color_map = dict()
    n_colors = len(np.unique(grid)) - 1
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            color = grid[i][j]
            if color != 0:
                diagonal_color_map[(i + j) % n_colors] = color
    output = np.zeros_like(grid)
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            output[i][j] = diagonal_color_map[(i + j) % n_colors]
    return output.tolist()

def task_05f2a901(grid):
    objects = detect_objects(grid)
    moving_object_color = 2
    moving_object = [object for object in objects if object.color == moving_object_color][0]
    still_object_color = 1
    still_object = [object for object in objects if object.color == still_object_color][0]
    moving_object = move_object_until_collision(moving_object, still_object)
    output = draw_objects(np.zeros_like(grid), [moving_object, still_object])
    return output

# def task_06df4c85(grid):

def task_08ed6ac7(grid):
    objects = detect_objects(grid)
    colors = [1, 2, 3, 4]
    output = np.zeros_like(grid)
    for object, color in zip(sort_objects_by_size(objects, ascending=False), colors):
        output = draw_objects(output, [object], color=color)
    return output

def task_09629e4f(grid):
    cells = np.concatenate(detect_cells(grid, grid_color=5))
    color_to_skip = 8
    chosen_cell = [cell for cell in cells if color_to_skip not in np.unique(cell)][0]
    new_cells = [np.tile(color, (3, 3)) for color in np.ravel(chosen_cell)]
    output = create_grid(new_cells, grid_shape=(3, 3), grid_color=5, grid_width=1)
    return output

# def task_0962bcdd(grid):
#     objects = detect_objects(grid)
#     # TODO: is there an easy way to implement the grow pattern?

# def task_0a938d79(grid):
#     objects = detect_objects(grid)

def task_0b148d64(grid):
    objects = detect_objects(grid)
    colors, counts = np.unique([object.color for object in objects], return_counts=True)
    less_common_color = colors[np.argmin(counts)]
    chosen_object = [object for object in objects if object.color == less_common_color][0]
    output = draw_objects(np.zeros_like(grid), [chosen_object])
    output = crop_object(output, chosen_object)
    return output

# def task_0ca9ddb6(grid):
#     objects = detect_objects(grid)
#     # it is a simple grow pattern

def task_0d3d703e(grid):
    color_map = {3: 4, 1: 5, 2: 6, 8:9, 5:1, 6:2, 9:8, 4:3, }
    output = apply_color_map(grid, color_map)
    return output

def task_0dfd9992(grid):
    output = fill_the_gaps_of_symmetry_pattern(grid)
    return output

# def task_0e206a2e(grid):

# def task_10fcaaa3(grid):
#     output = np.tile(grid, (2, 2))
#     # some growing pattern
#     return output

# def task_11852cab(grids):
#     # another growing pattern

def task_1190e5a7(grid):
    colors = np.unique(grid)
    less_common_color = colors[np.argmin(np.unique(grid, return_counts=True)[1])]
    cells = detect_cells(grid, grid_color=less_common_color)
    most_common_color = colors[np.argmax(np.unique(grid, return_counts=True)[1])]
    output = np.tile(most_common_color, (len(cells), len(cells[0])))
    return output

# def task_137eaa0f(grid):
#     # some tetris puzzle

# def task_150deff5(grid):
#     # some puzzle where we have to color an object based on some rules

def task_178fcbfb(grid):
    objects = detect_objects(grid)
    vertical_color = 2
    output = np.zeros_like(grid)
    for object in objects:
        if object.color == vertical_color:
            vertical_object = object
            output = draw_line(object, axis=1, color=object.color)
    for object in objects:
        if object.color != vertical_color:
            output = draw_line(object, axis=0, color=object.color)
    return output

def task_1a07d186(grid):
    objects = detect_objects(grid)
    lines = [object for object in objects if object.is_line()]
    output = draw_objects(np.zeros_like(grid), lines)

    for object in objects:
        if object.is_line():
            continue
        for line in lines:
            if object.color == line.color:
                new_object = move_object_until_collision(object, line)
                output = draw_objects(output, [new_object])
                break
    return output

