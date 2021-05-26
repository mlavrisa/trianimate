"""Produce nice clean triangulations of images, with tunable triangle sizes.

The triangulations produced by this module are cleaner than triangulations produced by
other methods, which can result in arbitrarily small triangles. This module focuses on
producing triangles with larger sizes, while trying to capture small image features as
well. Various supporting functions can be used as well to change the properties of the
resulting triangulations, including colour modifications.

    Typical usage examples:
        verts, faces, colours = classic_process(img, 0.6, 0.4, 0.3, 512)
        verts, faces = triangulate(img)
        colours = get_triangle_means(img, verts, faces)
        img = warp_colours(img)
"""

import cv2 as cv
import numpy as np
import numpy.linalg as linalg
from scipy.spatial import Delaunay

from trianimate.triangulate_utils import _pick_points_winblock


def _sobel_process(src: np.ndarray) -> np.ndarray:
    scale = 1
    delta = 0
    ksize = 5
    ddepth = cv.CV_32F  # cv.CV_16S

    # for reducing noise
    src = cv.GaussianBlur(src, (3, 3), 0, borderType=cv.BORDER_REPLICATE)

    x_grads = [
        cv.Sobel(
            src[..., cdx],
            ddepth,
            1,
            0,
            ksize=ksize,
            scale=scale,
            delta=delta,
            borderType=cv.BORDER_REPLICATE,
        )
        for cdx in range(src.shape[-1])
    ]

    y_grads = [
        cv.Sobel(
            src[..., cdx],
            ddepth,
            0,
            1,
            ksize=ksize,
            scale=scale,
            delta=delta,
            borderType=cv.BORDER_REPLICATE,
        )
        for cdx in range(src.shape[-1])
    ]

    grad = np.stack(x_grads + y_grads, axis=-1)

    return grad


def _warp_exp(cols: np.ndarray, factor: float) -> np.ndarray:
    # smooth function which curves between 0 and 255
    # the farther factor is from 0, the more extreme the warp
    return np.int32(255 * (1 - np.exp(-cols * factor)) / (1 - np.exp(-255.0 * factor)))


def warp_colours(
    img: np.ndarray, colour_boost: float = 0.5, brightness_boost: float = 0.5
) -> np.ndarray:
    """Warps the colours and brightness of an RGB image or array of RGB colours.
    
    Converts the image or array to HSV, and warps the saturation and value channels.
    The degree of boosting corresponds to `colour_boost` and  `brightness_boost` values
    respectively, with a value of 0 being no change, and 1 being an extreme change.
    Values less than 0 are also accepted, used to reduce the saturation or brightness.
    
    Args:
        img: np.ndarray containing image or colour data to operate on. dims: either
            (h, w, 3) or (L, 3), dtype: np.uint8
        colour_boost: float value clipped to the range [-1, 1] indicating the decrease
            (negative) or increase (positive) in the saturation.
        brightness_boost: same as above but for the value channel.
    
    Returns:
        np.ndarray containing the data with warped colours, also np.uint8
    
    Raises:
        ValueError: img must have 2 or 3 dimensions.
    """

    sat_val = np.clip(colour_boost, -1.0, 1.0) * 2.5e-2
    val_val = np.clip(brightness_boost, -1.0, 1.0) * 1.25e-2

    dims = len(img.shape)

    if dims == 2:
        hls = cv.cvtColor(img[None, :, :], cv.COLOR_RGB2HLS).astype(np.int32)
    elif dims == 3:
        hls = cv.cvtColor(img, cv.COLOR_RGB2HLS).astype(np.int32)
    else:
        raise ValueError(f"img must have 2 or 3 dimensions, got {dims}.")

    if abs(val_val) > 1e-6:
        hls[:, :, 1] = _warp_exp(hls[:, :, 1], val_val)
    if abs(sat_val) > 1e-6:
        hls[:, :, 2] = _warp_exp(hls[:, :, 2], sat_val)

    if dims == 2:
        return cv.cvtColor(hls.astype(np.uint8), cv.COLOR_HLS2RGB)[0]
    else:
        return cv.cvtColor(hls.astype(np.uint8), cv.COLOR_HLS2RGB)


def find_colours(
    img: np.ndarray, vertices: np.ndarray, faces: np.ndarray
) -> np.ndarray:
    """Get triangle colours.
    
    From an image, and a triangulation of it, get the colours of each of the associated
    triangle faces by taking the mean of the colours of the triangles within the image.
    
    Args:
        img: np.ndarray corresponding to the image. dims: (h, w, 3)
        vertices: np.ndarray corresponding to the scaled points, which will be scaled up
            to the width and height of the image. dims: (Npts, 2), where vertices[:, 0]
            is x and vertices[:, 1] is y.
        faces: np.ndarray of the indices of the vertices which make up the triangles in
            the image. dims: (Ntri, 3), dtype is an integer type
    
    Returns:
        np.ndarray with the colours of each of the faces. dims: (Ntri, 3)
    """

    h, w, _ = img.shape
    points = vertices * np.array([[w, h]])

    cols = np.zeros((faces.shape[0], 3), dtype=np.uint8)
    cv_pts = np.around(points).astype(np.int_)

    for tdx, pts in enumerate(faces):
        curr_pts = cv_pts[pts]
        ymin = np.min(curr_pts[:, 1])
        xmin = np.min(curr_pts[:, 0])
        ymax = np.minimum(np.max(curr_pts[:, 1]), h - 1)
        xmax = np.minimum(np.max(curr_pts[:, 0]), w - 1)
        win_w = xmax - xmin + 1
        win_h = ymax - ymin + 1
        window_pts = curr_pts - np.array([[xmin, ymin]])
        tri_mask = cv.fillConvexPoly(
            np.zeros((win_h, win_w), dtype=np.uint8),
            np.int32([window_pts]),
            color=255,
            lineType=cv.LINE_AA,
        )
        col = np.uint8(cv.mean(img[ymin : ymax + 1, xmin : xmax + 1], mask=tri_mask))
        cols[tdx] = col[:3]

    return cols


def find_vertices(
    img: np.ndarray, detail: float = 0.5, threshold: float = 0.5, max_dim: int = 1024
) -> np.ndarray:
    """Find key points on an image and create a triangulation.
    
    Picks points such that they tend to be far from one another, and such that they lie
    above certain detail thresholds. Higher detail, larger images take longer to
    compute. Capture more detail by setting a higher `detail` value. Ignore more gently
    varying image regions by setting a higher `threshold`.
    
    Args:
        img: `np.ndarray` with the image in question. dims: (h, w, [1 or more])
        detail: `float` in the range 0 to 1, indicating how much detail to capture.
        threshold: `float` in the range 0 to 1, indicating what fraction of the image is 
            considered 'background' to be ignored.
        max_dim: if the image is larger than this integer, it will be rescaled so that
            its maximum dimension is this size. This reduces computation time.
    
    Returns:
        an `np.ndarray` with points on the image, making up the triangle vertices as
        [x, y] pairs. This is scaled to the range [0, 1] to be easily resized to other
        image dimensions or aspect ratios. Its dimensions are (Npts, 2).
    """

    # calculate the constants used in the various calculations
    det_val = int(2.0 ** (np.clip(detail, 0.0, 1.0) * 3.0 + 4.0))
    thresh_val = np.clip(threshold, 0.0, 1.0) * 100.0

    # resizing step, scale the image to max_dim to speed up processing
    # then calculate inter-point distance from det_val, which controls triangle size
    oh, ow = img.shape[:2]
    if oh > max_dim or ow > max_dim:
        h, w = int(oh * max_dim / max(oh, ow)), int(ow * max_dim / max(oh, ow))
        img = cv.resize(img, (w, h), interpolation=cv.INTER_AREA)
    else:
        h, w = oh, ow
    spread = max(max(h, w) // det_val, 1)

    # calculate the grads and norm, and the threshold value
    grads = _sobel_process(img)
    grad_norm = linalg.norm(grads, axis=2) + 1.0
    bkg = np.percentile(grad_norm, thresh_val)

    # determine the image's critical points
    points = _pick_points_winblock(grads, grad_norm, spread, bkg)

    # scale the points to be in [0, 1)
    scale_pts = points / np.array([[w - 1, h - 1]])

    return scale_pts


def find_faces(vertices: np.ndarray) -> np.ndarray:
    """Wrapper of scipy.spatial.Delaunay which returns the simplex indices.
    
    Args:
        vertices: `np.ndarray` points on a 2D plane to triangulate
        
    Returns:
        The Delaunay triangulation of the points, as an (Nf, 3) array of indices."""
    tri = Delaunay(vertices)
    return tri.simplices


def classic_triangulate(
    img: np.ndarray,
    detail: float = 0.5,
    threshold: float = 0.5,
    colour_boost: float = 0.5,
    max_dim: int = 1024,
) -> np.ndarray:
    """Helper function which runs through a standard triangulation.

    Warps the colours to make them more saturated and brighter. Finds the corresponding
    key points and triangulation of them. Determines the mean colours of the triangles.
    Finally applies a second boost to the saturation. See the corresponding arguments of
    `triangulate`, `warp_colours` and `get_triangle means` for a detailed description of
    the arguments and return values.

    Args:
        img: `np.ndarray` input image
        detail: `float` level of detail (0., 1.)
        threshold: `float` fraction of img considered background (0., 1.)
        colour_boost: `float` how much to boost the colour by (-1., 1.)
        max_dim: `int` shrink the image if it's above this size
    
    Returns:
        vertices, faces and colours of the triangulation, as `np.ndarray`s
    """
    img = warp_colours(img, colour_boost, colour_boost)
    vertices = find_vertices(img, detail, threshold, max_dim)
    faces = find_faces(vertices)
    cols = find_colours(img, vertices, faces)
    cols = warp_colours(cols, colour_boost, 0.0)

    return vertices, faces, cols

