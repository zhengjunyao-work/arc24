import random
import numpy as np

def task_007bbfb7():
    color = random.randint(1, 9)
    shape = (random.randint(1, 5), random.randint(1, 5))
    grid = color * np.random.binomial(1, 0.5, shape)
    return grid.tolist()
