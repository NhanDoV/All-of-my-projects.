import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

def random_pairs(n):
    idx = list(range(n))
    random.shuffle(idx)
    return [(idx[i], idx[i+1]) for i in range(0, len(idx)-1, 2)]

def generate_points(w, h, n_areas, d):
    """
    Generate top-left points (y, x) for blocks size d x d
    """
    points = []
    for _ in range(n_areas):
        x = random.randint(0, w - d)
        y = random.randint(0, h - d)
        points.append((y, x))  # (row, col)
    return points

def pairwise_perm_blocks(arr, points, d):
    arr = arr.copy()
    pairs = random_pairs(len(points))
    
    for a, b in pairs:
        y0, x0 = points[a]
        y1, x1 = points[b]
        
        slc_a = (slice(y0, y0+d), slice(x0, x0+d), ...)
        slc_b = (slice(y1, y1+d), slice(x1, x1+d), ...)
        
        tmp = arr[slc_a].copy()
        arr[slc_a] = arr[slc_b]
        arr[slc_b] = tmp
    
    return arr