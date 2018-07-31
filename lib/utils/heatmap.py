import numpy as np
import cv2


def compute_heatmap(points, shape, crop=None, scale=None, k_ratio=1.0):
    # Put points on in the numpy array.
    heatmap = np.zeros((shape[0], shape[1]), dtype=np.float32)
    n_points = points.shape[0]
    value = 1.0

    for i in xrange(n_points):
        x = points[i, 0]
        y = points[i, 1]
        row = int(y)
        col = int(x)
        heatmap[row, col] += value

    # Compute kernel size of the Gaussian filter. The kernel size must be odd.
    k_size = int(np.sqrt(shape[0] * shape[1]) / k_ratio)

    if k_size % 2 == 0:
        k_size += 1

    # Compute the heatmap using the Gaussian filter.
    heatmap = cv2.GaussianBlur(heatmap, (k_size, k_size), 0)

    if crop:
        heatmap = heatmap[crop[0]:crop[0]+crop[2], crop[1]:crop[1]+crop[3]]

    if scale:
        heatmap = cv2.resize(heatmap, None, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_LINEAR)

    heatmap /= np.sum(heatmap)

    return heatmap

