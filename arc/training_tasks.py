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

def task_1b2d62fb(grid):
    side_1, side_2 = split_grid(grid, axis=1)
    output = np.maximum(side_1, side_2)
    output = apply_color_map(output, color_map={9: 0, 0:8})
    return output

# def task_1b60fb0c(grid):
#     # this requires the ability to divide an object in parts based on simmetry

# def task_1bfc4729(grid):
#     objects = detect_objects(grid)

def task_1c786137(grid):
    objects = detect_objects(grid)
    for object in objects:
        if object.is_rectangle() and object.size > 4:
            output = crop_object(grid, object)
            return output

def task_1caeab9d(grid):
    reference_object_color = 1
    objects = detect_objects(grid)
    reference_object = [object for object in objects if object.color == reference_object_color][0]
    output = draw_objects(np.zeros_like(grid), [reference_object])
    for object in objects:
        if object.color != reference_object_color:
            new_object = move_object(object, axis=1,
                                     amount=reference_object.y - new_object.y)
            output = draw_objects(output, [new_object])
    return output

def task_1cf80156(grid):
    object = detect_objects(grid)[0]
    output = crop_object(grid, object)
    return output

def task_1e0a9b12(grid):
    output = np.sort(grid, axis=0)
    return output.tolist()

# def task_1e32b0e9(grid):
#     cells = detect_cells(grid)

# def task_1f0c79e5(grid):
#     # repeat the object in the direction of the red pixels

def task_1f642eb9(grid):
    objects = detect_objects(grid)
    still_object = [object for object in objects if object.size > 1][0]
    output = grid.copy()
    for object in objects:
        if object.size == 1:
            distance = get_distance_between_objects(object, still_object)
            object.move(distance + np.sign(distance))
            output = draw_objects(output, [object])
    return output

def task_1f85a75f(grid):
    objects = detect_objects(grid)
    objects = sort_objects_by_size(objects, ascending=False)
    biggest_object = objects[0]
    output = crop_object(grid, biggest_object)
    return output

def task_1f876c06(grid):
    objects = detect_objects(grid)
    objects = sorted(objects, key=lambda x: x.color)
    output = grid.copy()
    for point1, point2 in zip(objects[::2], objects[1::2]):
        output = draw_line_between_points(output, point1, point2)
    return output

def task_1fad071e(grid):
    objects = detect_objects(grid)
    relevant_color = 1
    n_squares_with_size_4 = 0
    for object in objects:
        if object.color == relevant_color and object.size == 4:
            n_squares_with_size_4 += 1
    output = np.zeros(shape=(1, 5))
    output[:, :n_squares_with_size_4] = relevant_color
    return output

# def task_2013d3e2(grid):

def task_2204b7a8(grid):
    objects = detect_objects(grid)
    lines = [object for object in objects if object.is_line()]
    output = grid.copy()
    for object in objects:
        if object.is_line():
            continue
        min_distance = np.inf
        new_color = object.color
        for line in lines:
            if get_distance_between_objects(object, line) < min_distance:
                min_distance = get_distance_between_objects(object, line)
                new_color = line.color
        object.color = new_color
        output = draw_objects(output, [object])
    return output

# def task_22168020(grid):
#     # this requires some kind of filling function, it could be consider some growing pattern

# def task_22233c11(grid):
#     # another growing pattern

def task_2281f1f4(grid):
    reference_color = 5
    new_color = 2
    cols = np.where(np.array(grid[0]) == reference_color)[0]
    rows = np.where(np.array(grid)[:, -1] == reference_color)[0]
    output = grid.copy()
    for col in cols:
        for row in rows:
            output[row][col] = new_color
    return output

# def task_228f6490(grid):
#     # this requires some kind of puzzle solving

def task_22eb0ac0(grid):
    objects = detect_objects(grid)
    objects = sorted(objects, key=lambda x: x.color)
    output = grid.copy()
    for point1, point2 in zip(objects[::2], objects[1::2]):
        if point1.y == point2.y:
            output = draw_line_between_points(output, point1, point2, color=point1.color)
    return output

# def task_234bbc79(grid):
#     # this requires non trivial puzzle solving

def task_23581191(grid):
    objects = detect_objects(grid)
    output = grid.copy()
    for object in objects:
        output = draw_line(object, axis=0, color=object.color)
        output = draw_line(object, axis=1, color=object.color)
    new_color = 2
    output[objects[0].y][objects[0].x] = new_color
    output[objects[1].y][objects[1].x] = new_color
    return output

# def task_239be575(grid):
#     # check if the red squares are connected

def task_23b5c85d(grid):
    objects = detect_objects(grid)
    smallest_object = sorted(objects, key=lambda x: x.size)[0]
    output = crop_object(grid, smallest_object)
    return output

def task_253bf280(grid):
    objects = detect_objects(grid)
    new_line_color = 3
    output = grid.copy()
    for idx, object in enumerate(objects):
        for other_object in objects[idx + 1:]:
            if object.y == other_object.y or object.x == other_object.x:
                output = draw_line_between_points(output, object, other_object, color=new_line_color)
    return output

# def task_25d487eb(grid):
#     # growing pattern

def task_25d8a9c8(grid):
    output = np.zeros_like(grid)
    color = 5
    for object in detect_objects(grid):
        if object.isline() and object.size == 3:
            object.color = color
            output = draw_objects(output, [object])
    return output

def task_25ff71a9(grid):
    output = np.zeros_like(grid)
    movement = (-1, 0)
    for object in detect_objects(grid):
        object.move(movement)
        output = draw_objects(output, [object])
    return output

# def task_264363fd(grid):
#     # not easy to implement in code

def task_272f95fa(grid):
    cells = np.concatenate(detect_cells(grid))
    cell_index_to_color = {1: 2, 3:4, 4:6, 5:3, 7:1}
    new_cells = [np.ones_like(cell)*cell_index_to_color.get(idx, 0) for idx, cell in enumerate(cells)]
    output = create_grid(new_cells, grid_shape=(3, 3), grid_color=8, grid_width=1)
    return output

# def task_27a28665(grid):
#     # maps the pattern to the color, best explained by example

def task_28bf18c6(grid):
    object = detect_objects(grid)[0]
    output = crop_object(grid, object)
    output = np.tile(output, (1, 2))
    return output

# def task_28e73c20(grid):
#     # draw an spiral
#     color = 3
#     output = grid.copy()
#     directions = [(0, 1), (-1, 0), (0, -1), (1, 0)]
#     position = (0, 0)
#     for direction in directions:
#         while True:
#             new_position = (position[0] + direction[0], position[1] + direction[1])
#             if output[new_position[0]][new_position[1]] != color:
#                 output[position[0]][position[1]] = color
#                 position = new_position
#             else:
#                 break
#     return output

def task_29623171(grid):
    cells = np.concatenate(detect_cells(grid))
    max_objects_in_cell = max([len(detect_objects(cell)) for cell in cells])
    new_cells = []
    for cell in cells:
        objects = detect_objects(cell)
        if len(objects) == max_objects_in_cell:
            new_cells.append(np.ones_like(cell)*objects[0].color)
        else:
            new_cells.append(np.zeros_like(cell))
    output = create_grid(new_cells, grid_shape=(3, 3), grid_color=5, grid_width=1)
    return output

def task_29c11459(grid):
    objects = detect_objects(grid)
    output = grid.copy()
    new_point_color = 5
    for idx, point_1 in enumerate(objects):
        for point_2 in objects[idx + 1:]:
            if point_1.y == point_2.y:
                middle_position = (point_1.position + point_2.position)/2
                output = draw_line_between_points(grid, point_1, middle_position, color=point_1.color)
                output = draw_line_between_points(grid, middle_position, point_2, color=point_2.color)
                output[middle_position.x][middle_position.y] = new_point_color
    return output

def task_29ec7d0e(grid):
    output = fill_the_gaps_of_symmetry_pattern(grid)
    return output

# def task_2bcee788(grid):
#     #this requires some symmetry and changing background color
#     objects = detect_objects(grid)
#     biggest_object = sorted(objects, key=lambda x: x.size)[-1]
#     # how to choose the simmetry axis?

# def task_2bee17df(grid):
#     # some kind of painting the maximum space between objects in two axis

def task_2c608aff(grid):
    objects = detect_objects(grid)
    objects = sorted(objects, key=lambda x: x.size)
    biggest_object = objects[-1]
    output = grid.copy()
    for object in objects[:-1]:
        distance = get_distance_between_objects(object, biggest_object)
        if min(distance) == 0:
            output = draw_line_between_points(output, object, biggest_object)
    return output

def task_2dc579da(grid):
    cells = np.concatenate(detect_cells(grid))
    for cell in cells:
        if np.unique(cell) == 2:
            output = cell
            return output

def task_2dd70a9a(grid):
    objects = detect_objects(grid)
    start_color = 3
    goal_color = 2
    start_object = [object for object in objects if object.color == start_color][0]
    goal_object = [object for object in objects if object.color == goal_color][0]
    output = grid.copy()
    output = draw_path_between_objects(output, start_object, goal_object, color=start_color)
    return output

# def task_2dee498d(grid):
#     # this requires to detect what is the biggest repeating element

def task_31aa019c(grid):
    objects = detect_objects(grid)
    colors, counts = np.unique([object.color for object in objects], return_counts=True)
    less_common_color = colors[np.argmin(counts)]
    chosen_object = [object for object in objects if object.color == less_common_color][0]
    output = draw_objects(np.zeros_like(grid), [chosen_object])
    output = draw_enclosing_rectangle(output, chosen_object, color=2)
    return output

def task_321b1fc6(grid):
    objects = detect_objects(grid, ignore_colors=True)
    chosen_object = [object for object in objects if len(object.colors) > 1][0]
    output = grid.copy()
    for object in objects:
        if len(object.colors) == 1:
            output = draw_objects(output, chosen_object.move(object.position))
    return output

# def task_32597951(grid):
#     # this requires non trivial dealing with objects