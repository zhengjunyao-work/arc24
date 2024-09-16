import random
import numpy as np

def task_007bbfb7(shape=(3, 3)):
    color = random.randint(1, 9)
    grid = color * np.random.binomial(1, 0.5, shape)
    return grid.tolist()

def task_00d62c1b():
    side = random.randint(6, 20)
    return create_random_objects_with_holes(color=3, background_color=0, shape=(side, side))
