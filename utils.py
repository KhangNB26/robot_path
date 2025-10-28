# utils.py
import time
import random
import numpy as np
from functools import wraps
from contextlib import contextmanager


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        res = func(*args, **kwargs)
        t1 = time.perf_counter()
        return res, (t1 - t0)
    return wrapper

@contextmanager
def timing(description: str):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{description}: {elapsed:.3f} seconds")

def neighbors4(pos, grid_shape):
    r, c = pos
    R, C = grid_shape
    for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
        nr, nc = r+dr, c+dc
        if 0 <= nr < R and 0 <= nc < C:
            yield (nr, nc)

def manhattan(a,b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def set_seed(seed=None):
    if seed is None:
        seed = np.random.SeedSequence().entropy
    random.seed(seed)
    np.random.seed(seed)
    return seed
