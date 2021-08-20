from typing import Tuple

import numpy as np
from numba import njit
from math import ceil


@njit
def _fill_block_maxes(
    grad_norm: np.ndarray,
    maxes: np.ndarray,
    args: np.ndarray,
    xmin: int,
    xmax: int,
    ymin: int,
    ymax: int,
    blocksize: int,
):
    """Iterates over regions of an image, calculating and storing the argmax of each."""
    w = grad_norm.shape[1]
    imin = xmin // blocksize
    imax = ceil(xmax / blocksize)
    jmin = ymin // blocksize
    jmax = ceil(ymax / blocksize)
    irng = imax - imin
    jrng = jmax - jmin

    bh, bw = maxes.shape

    # if it's the first time we have to do all blocks
    # note this can possibly be fooled by the edge case where there is only one block
    # but in that case, it is probably running tremendously fast anyway
    not_first_time = args[bh - 1, bw - 1, 0] > 0 or args[bh - 1, bw - 1, 1] > 0

    for kdx in range(jrng * irng):
        # how wide is the block we're looking at (relevant at border)
        jdx = kdx // irng + jmin
        idx = kdx % irng + imin

        # if the current max of this block falls out of the bounds of the update,
        # don't calculate argmax
        y, x = args[jdx, idx]
        out_of_bounds = (
            (idx == imin and xmin > x)
            or (jdx == imax and ymin > y)
            or (idx == imax - 1 and xmax < x)
            or (jdx == jmax - 1 and ymax < y)
        )
        skip = not_first_time and out_of_bounds
        # This seems to save about 10%, not bad
        if skip:
            continue

        # if the block is near the edge, its width is smaller
        xsize = min(w - idx * blocksize, blocksize)

        # find the argmax within the block, but set args relative to whole image
        argmax = np.argmax(
            grad_norm[
                jdx * blocksize : (jdx + 1) * blocksize,
                idx * blocksize : (idx + 1) * blocksize,
            ]
        )
        x, y = idx * blocksize + argmax % xsize, jdx * blocksize + argmax // xsize
        args[jdx, idx] = np.array([y, x])

        # find the max value for the block, and set maxes
        maxes[jdx, idx] = grad_norm[y, x]


@njit
def _meshgrid(xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Equivalent to np.meshgrid for the 2D case, numba friendly."""
    x_out = np.expand_dims(xs, 0) + np.zeros((ys.size, 1), dtype=np.float32)
    y_out = np.expand_dims(ys, 1) + np.zeros((1, xs.size), dtype=np.float32)
    return x_out, y_out


@njit
def _pick_points_winblock(
    grads: np.ndarray, grad_norm: np.ndarray, dot_dist: float, bkg: float
) -> np.ndarray:
    """Picks points corresponding to the most intense edges in an image."""
    h, w, _ = grads.shape

    num_points = 4
    max_points = int(h * w * 0.1)
    points = np.zeros((max_points, 2), dtype=np.int32)

    # start with the corners
    points[:4] = np.array(
        [[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=np.int32
    )

    # construct mesh and gassians outside of the loop, only need to once
    xsn, ysn = _meshgrid(
        np.arange(-2 * dot_dist, 2 * dot_dist + 1).astype(np.float32),
        np.arange(-2 * dot_dist, 2 * dot_dist + 1).astype(np.float32),
    )
    gsn_dot = np.exp(-0.5 * (xsn * xsn + ysn * ysn) / (dot_dist * dot_dist))
    gsn_dst = 1.0 - np.exp(-0.5 * (xsn * xsn + ysn * ysn) / (dot_dist * dot_dist / 16))

    # calculate the normalized gradients, just need this once
    normals = grads / np.expand_dims(grad_norm, 2)

    # calculate the blocks to speed up argmax
    # why x2? Empirical finding, 3.5% speedup.
    blocksize = int(np.sqrt(np.sqrt(h * w)) * 2.0)
    maxes = np.zeros(
        (int(np.ceil(h / blocksize)), int(np.ceil(w / blocksize))), dtype=np.float32
    )
    args = np.zeros(
        (int(np.ceil(h / blocksize)), int(np.ceil(w / blocksize)), 2), dtype=np.int32
    )
    _fill_block_maxes(grad_norm, maxes, args, 0, w, 0, h, blocksize)

    # while the highest remaining gradient norm is greater than bkg, pick another point.
    while num_points < max_points:
        # find the coordinates and normal vector of the highest point
        y, x = np.reshape(args, (-1, 2))[np.argmax(maxes)]
        if grad_norm[y, x] < bkg:
            break
        points[num_points] = np.array((x, y))
        normal = normals[y, x]

        # calculate the windows
        xmin = np.maximum(x - 2 * dot_dist, 0)
        xmax = np.minimum(x + 2 * dot_dist + 1, w)
        ymin = np.maximum(y - 2 * dot_dist, 0)
        ymax = np.minimum(y + 2 * dot_dist + 1, h)
        gxmin = xmin - (x - 2 * dot_dist)
        gxmax = (4 * dot_dist + 1) - ((x + 2 * dot_dist + 1) - xmax)
        gymin = ymin - (y - 2 * dot_dist)
        gymax = (4 * dot_dist + 1) - ((y + 2 * dot_dist + 1) - ymax)
        winh = ymax - ymin
        winw = xmax - xmin

        # calculate the gradient norm window
        dot_distwin = np.zeros((winh, winw), dtype=np.float32)

        # calculate the points within the window that match the gradient
        for ydx in range(winh):
            dot_distwin[ydx] = np.maximum(
                0.0, np.dot(normals[ydx + ymin, xmin:xmax], normal)
            )

        # calculate the decrease in the gradient, and apply it
        dot_dec = 1.0 - gsn_dot[gymin:gymax, gxmin:gxmax] * dot_distwin
        dst_dec = gsn_dst[gymin:gymax, gxmin:gxmax]
        factor = dot_dec * dst_dec
        grad_norm[ymin:ymax, xmin:xmax] *= factor
        _fill_block_maxes(grad_norm, maxes, args, xmin, xmax, ymin, ymax, blocksize)

        # recalculate the norm
        num_points += 1
    return points[:num_points]


@njit
def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Implements the crossing algorithm to determine whether points are in polygon."""
    # setup
    n_p = len(points)
    n_v = len(polygon) - 1
    count = np.zeros((n_p,), dtype=np.int32)
    xs = polygon[:n_v, 0]
    xe = polygon[1:, 0]
    ys = polygon[:n_v, 1]
    ye = polygon[1:, 1]

    minx = np.min(polygon[:, 0])
    maxx = np.max(polygon[:, 0])
    miny = np.min(polygon[:, 1])
    maxy = np.max(polygon[:, 1])

    # only need to calculate these things once, then just grab items from them
    dx = xe - xs
    dy = ye - ys
    dxdy = dx / dy

    # iterate over the points
    for pdx in range(n_p):
        if (
            points[pdx, 0] > maxx
            or points[pdx, 0] < minx
            or points[pdx, 1] > maxy
            or points[pdx, 1] < miny
        ):
            continue
        # find the polygon edges which a horizontal line through point intersects with
        # if line slopes up, can intersect with the starting point but not the end
        # if line slopes down, can intersect with the end but not the start
        # horizontal lines will be left out, but that's okay! One of the neighbouring
        # segments will capture the appropriate interaction
        level_with = np.logical_or(
            np.logical_and(ys <= points[pdx, 1], ye > points[pdx, 1]),
            np.logical_and(ys > points[pdx, 1], ye <= points[pdx, 1]),
        )
        # iterate over those edges, only if edges are level with
        for vdx in range(n_v):
            if level_with[vdx]:
                # if so, calculate the x coordinate, check we are left of that
                vt = points[pdx, 1] - ys[vdx]
                if points[pdx, 0] < xs[vdx] + vt * dxdy[vdx]:
                    count[pdx] += 1

    # if crossings are odd, the point is in the polygon
    return np.equal(np.remainder(count, 2), 1)
