# Scripts related to the Convex Hull
#
# Author: Indrajit Ghosh
#
# Date: Jul 05, 2022
#

import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.path import Path

def generate_uniform_sample_points(shape=(5, 2), lower:float=-1, upper:float=1):
    """
    Generates ```Unif[`lower`, `upper`)``` distributed random points

    Returns
    -------
        `np.array`

    Example
    -------
    >>> generate_uniform_sample_points()

        array([
            [-0.42917239, -0.9402378 ],
            [ 0.29796083, -0.85400552],
            [-0.98060496,  0.76676783],
            [ 0.82573971,  0.7479717 ],
            [ 0.36098848, -0.97295949]
            ])

    >>> generate_uniform_sample_points(shape=(6, 3))

        array([
            [ 0.81626309,  0.57074435, -0.51679046, -0.14502065],
            [ 0.17765438,  0.68677077, -0.78349613,  0.6564779 ],
            [ 0.36337345,  0.72131454,  0.36628025, -0.48209883],
            [ 0.59299661, -0.08121728,  0.394873  ,  0.02575999],
            [-0.92177366, -0.63444988, -0.9856509 ,  0.36512726],
            [-0.35511079, -0.11933817,  0.2604979 ,  0.82616689]
            ])

    """
    a, b = lower, upper
    ran_arr = (b - a) * np.random.random_sample(size=shape) + a # Uniform[a, b); b>a

    return ran_arr

def get_vertices_of_convex_hull(points:np.array=np.random.random_sample(size=(5, 2))):
    """
    Returns the extreme points of the ConvexHull of the given set of ```points```

    Returns
    -------
        `np.array`

    Reference
    ---------
        `Quickhull algorithm for computing the convex hull`

    """
    hull = ConvexHull(points)
    vertices = points[hull.vertices]

    return vertices


def is_inside_convex_hull(points:np.array=np.random.random_sample(size=(5, 2)), checkpoint=(0, 0)):
    """
    Checks whether the `checkpoint` is inside the ConvexHull of `points`

    Returns
    -------
        `Bool`: True/False
    """
    vertices = get_vertices_of_convex_hull(points)
    hall_path = Path(vertices)

    return hall_path.contains_point(checkpoint)


def main():
    print('hello world!')


if __name__ == '__main__':
    main()