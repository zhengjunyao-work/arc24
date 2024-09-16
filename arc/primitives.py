import numpy as np

def apply_color_map(grid, color_map):
    output, grid = np.array(grid), np.array(grid)
    for previous_color, new_color in color_map.items():
        output[grid == previous_color] = new_color
    return output.tolist()

