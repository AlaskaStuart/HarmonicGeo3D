"""Fit a logical grid and deterministic CCW triangle net to drill points."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan, cos, sin

import numpy as np
from scipy import stats
from scipy.signal import find_peaks

from .mesh_topology import build_mesh_topology


def _validate_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {points.shape}.")
    if len(points) < 3:
        raise ValueError("points must contain at least three XY locations.")
    if not np.isfinite(points).all():
        raise ValueError("points contains non-finite coordinates.")
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError("points contains duplicate XY coordinates.")
    return points


def _validate_logical_grid(
    nearest_grids: np.ndarray,
    *,
    grid_shape: tuple[int, int],
) -> None:
    if nearest_grids.ndim != 2 or nearest_grids.shape[1] != 2:
        raise ValueError(
            f"nearest_grids must have shape (N, 2), got {nearest_grids.shape}."
        )
    if not np.issubdtype(nearest_grids.dtype, np.integer):
        raise ValueError("nearest_grids must contain integer grid coordinates.")
    if len(np.unique(nearest_grids, axis=0)) != len(nearest_grids):
        raise ValueError("nearest_grids contains duplicate logical coordinates.")
    if np.any(nearest_grids < 0):
        raise ValueError("nearest_grids contains negative logical coordinates.")
    limits = np.asarray(grid_shape, dtype=int)
    if np.any(nearest_grids >= limits):
        raise ValueError(
            f"nearest_grids contains coordinates outside grid_shape {grid_shape}."
        )


def hough_transform(points: np.ndarray) -> np.ndarray:
    """Map point pairs to centred Hough angle/radius coordinates.

    Parameters
    ----------
    points : ndarray, shape (N, 2)
        Planar borehole coordinates in metres.

    Returns
    -------
    ndarray, shape (N * (N - 1) // 2, 4)
        Angle (radians), radius (metres), and the two endpoint indices.
    """
    centre = np.mean(points, axis=0)
    points_num = points.shape[0]
    row_transformed_data = int(points_num * (points_num - 1) / 2)
    hough_space = np.zeros(shape=(row_transformed_data, 4))

    for i in range(points_num):
        for j in range(i + 1, points_num):
            idx = int(i * points_num - i * (i + 1) / 2 + j - i - 1)

            x_i, y_i = points[i] - centre
            x_j, y_j = points[j] - centre

            theta = atan((x_j - x_i) / (y_i - y_j))

            rho = x_i * cos(theta) + y_i * sin(theta)

            hough_space[idx][0] = theta
            hough_space[idx][1] = rho
            hough_space[idx][2] = i
            hough_space[idx][3] = j

    return hough_space


def find_kde_peaks(data: np.ndarray, bw: float = 0.05) -> np.ndarray:
    """Locate maxima of a one-dimensional Gaussian KDE.

    Parameters
    ----------
    data : ndarray, shape (N,)
        Finite samples in the coordinate's units.
    bw : float, optional
        Gaussian KDE bandwidth factor.

    Returns
    -------
    ndarray, shape (P, 2)
        Peak coordinate and density, evaluated on the established 1000-point grid.
    """

    kde = stats.gaussian_kde(data, bw_method=bw)
    x_range_min = min(data) - 0.1 * abs(min(data))
    x_range_max = max(data) + 0.1 * abs(max(data))

    x_range = np.linspace(x_range_min, x_range_max, 1000)
    kde_values = kde(x_range)

    peaks, _ = find_peaks(kde_values)

    x_peaks = x_range[peaks].reshape(-1, 1)
    kde_values_peaks = kde_values[peaks].reshape(-1, 1)

    kde_peaks = np.hstack((x_peaks, kde_values_peaks))

    return kde_peaks


def find_top_two_kde_peaks(kde_peaks: np.ndarray) -> np.ndarray:
    """Return the two highest-density peaks in descending density order.

    Parameters
    ----------
    kde_peaks : ndarray, shape (P, 2)
        Peak coordinates and densities from ``find_kde_peaks``.

    Returns
    -------
    ndarray, shape (min(P, 2), 2)
        Selected coordinates and densities.
    """
    sorted_indices = np.argsort(kde_peaks[:, 1])[::-1]
    top_two_indices = sorted_indices[:2]
    top_two_kde_peaks = kde_peaks[top_two_indices]

    return top_two_kde_peaks


def transform_along_basis(
    points: np.ndarray, basis_i: np.ndarray, basis_j: np.ndarray
) -> np.ndarray:
    """Project XY points onto two fitted grid directions.

    Parameters
    ----------
    points : ndarray, shape (N, 2)
        Centred coordinates in metres.
    basis_i, basis_j : ndarray, shape (2,)
        Fitted direction vectors, which need not be orthogonal.

    Returns
    -------
    ndarray, shape (N, 2)
        Projection coefficients along the two directions.
    """
    transformed_points = []

    for point in points:
        coeff_i = np.dot(point, basis_i) / np.dot(basis_i, basis_i)
        coeff_j = np.dot(point, basis_j) / np.dot(basis_j, basis_j)
        transformed_points.append((coeff_i, coeff_j))

    return np.array(transformed_points)


def calculate_nearest_location(
    points: np.ndarray, grids_x_axes: np.ndarray, grids_y_axes: np.ndarray
) -> np.ndarray:
    """Assign logical grid indices using the established collision rule.

    Parameters
    ----------
    points : ndarray, shape (N, 2)
        Projected coordinates.
    grids_x_axes, grids_y_axes : ndarray
        Candidate axis coordinates derived from KDE peaks.

    Returns
    -------
    ndarray, shape (N, 2)
        Integer logical coordinates. Collisions move forward along the axis
        with the greater squared displacement; this preserves the original rule.
    """
    nearest_grids = []

    for pt in points:
        x_min_idx = (np.square(grids_x_axes - pt[0])).argmin()
        y_min_idx = (np.square(grids_y_axes - pt[1])).argmin()

        nearest_grids.append((x_min_idx, y_min_idx))

    nearest_grids = np.array(nearest_grids)

    while True:
        unique_grids, counts = np.unique(nearest_grids, axis=0, return_counts=True)
        repeated_grids = unique_grids[counts > 1]

        if len(repeated_grids) == 0:
            break
        for rep_grid in repeated_grids:
            indices = np.where((nearest_grids == rep_grid).all(axis=1))[0]
            idx_pt_a, idx_pt_b = indices[0], indices[1]

            idx_grid_i, idx_grid_j = rep_grid[0], rep_grid[1]

            pt_a, pt_b = points[idx_pt_a], points[idx_pt_b]
            grid_i, grid_j = grids_x_axes[idx_grid_i], grids_y_axes[idx_grid_j]

            # Resolve collisions using the established forward-grid rule.
            # Move along the axis with the greater squared displacement.
            if pt_a[0] < pt_b[0]:
                x_to_grid_dist = np.square(pt_b[0] - grid_i)
                y_to_grid_dist = np.square(pt_b[1] - grid_j)

                if x_to_grid_dist < y_to_grid_dist:
                    new_nearest_grid = [idx_grid_i, idx_grid_j + 1]
                else:
                    new_nearest_grid = [idx_grid_i + 1, idx_grid_j]
                nearest_grids[idx_pt_b] = np.array(new_nearest_grid)
            else:
                x_to_grid_dist = np.square(pt_a[0] - grid_i)
                y_to_grid_dist = np.square(pt_a[1] - grid_j)

                if x_to_grid_dist < y_to_grid_dist:
                    new_nearest_grid = [idx_grid_i, idx_grid_j + 1]
                else:
                    new_nearest_grid = [idx_grid_i + 1, idx_grid_j]
                nearest_grids[idx_pt_a] = np.array(new_nearest_grid)

    return nearest_grids


def corner_in_grids(rectangle: list, grids: np.ndarray) -> list:
    """Return point indices for the occupied corners of a grid rectangle.

    The returned indices follow ``rectangle``'s corner order.  A rectangle is
    usable only when three or all four of its corners are occupied; otherwise
    an empty list is returned.  ``calculate_nearest_location`` normally makes
    grid coordinates unique, but keeping the first index here also makes the
    lookup deterministic if duplicate coordinates are supplied.
    """
    grids = np.asarray(grids)
    if grids.ndim != 2 or grids.shape[1] != 2:
        raise ValueError(f"grids must have shape (N, 2), got {grids.shape}.")

    grid_indices = {}
    for index, grid in enumerate(grids):
        grid_indices.setdefault(tuple(grid), index)

    result = []
    for corner in rectangle:
        index = grid_indices.get(tuple(corner))
        if index is not None:
            result.append(index)

    if len(result) in (3, 4):
        return result
    return []


@dataclass(frozen=True, slots=True)
class _LogicalGridFit:
    points: np.ndarray
    nearest_grids: np.ndarray
    grid_shape: tuple[int, int]


def _fit_logical_grid(points: np.ndarray) -> _LogicalGridFit:
    """Fit the expensive Hough/KDE logical grid exactly once."""

    points = _validate_points(points)

    hough_space = hough_transform(points)

    hough_space_theta = hough_space[:, 0]

    theta_kde_peaks = find_kde_peaks(hough_space_theta)
    top_two_kde_peaks = find_top_two_kde_peaks(theta_kde_peaks)

    theta_1 = top_two_kde_peaks[0, 0]
    theta_2 = top_two_kde_peaks[1, 0]

    basis_i = np.array([cos(theta_1), sin(theta_1)])
    basis_j = np.array([cos(theta_2), sin(theta_2)])
    centre = np.mean(points, axis=0)
    points_centered = points - centre
    transformed_points = transform_along_basis(points_centered, basis_i, basis_j)

    grids_i_axes = find_kde_peaks(transformed_points[:, 0], bw=0.07)[:, 0]

    grid_j_axes = find_kde_peaks(transformed_points[:, 1], bw=0.07)[:, 0]

    nearest_grids = calculate_nearest_location(
        transformed_points, grids_i_axes, grid_j_axes
    )
    _validate_logical_grid(
        nearest_grids,
        grid_shape=(len(grids_i_axes), len(grid_j_axes)),
    )
    return _LogicalGridFit(
        points=points,
        nearest_grids=nearest_grids,
        grid_shape=(len(grids_i_axes), len(grid_j_axes)),
    )


def generate_tri_net(
    points: np.ndarray, *, _fit: _LogicalGridFit | None = None
) -> list[list[int]]:
    """Construct a deterministic counter-clockwise logical-grid triangle net.

    Parameters
    ----------
    points : ndarray, shape (N, 2)
        Unique finite planar coordinates in metres.
    _fit : _LogicalGridFit, optional
        Internal precomputed grid fit.

    Returns
    -------
    list of list of int
        Triangle node indices, with three indices per cell.

    Notes
    -----
    The method expects two recoverable grid directions and is intended for
    approximately grid-arranged surveys. It is not a general scattered-data
    triangulator. Rectangles with fewer than three occupied corners are omitted.
    """
    fit = _fit_logical_grid(points) if _fit is None else _fit
    points = fit.points
    nearest_grids = fit.nearest_grids
    grids_i_count, grids_j_count = fit.grid_shape

    triangle_net_list = []

    for rect_i in range(grids_i_count - 1):
        for rect_j in range(grids_j_count - 1):
            ith_rect = [
                [rect_i, rect_j],
                [rect_i + 1, rect_j],
                [rect_i + 1, rect_j + 1],
                [rect_i, rect_j + 1],
            ]

            corners = corner_in_grids(ith_rect, nearest_grids)
            if len(corners) == 4:
                triangle_net_list.append([corners[0], corners[1], corners[2]])
                triangle_net_list.append([corners[0], corners[2], corners[3]])
            elif len(corners) == 3:
                triangle_net_list.append(corners)
            else:
                continue

    topology = build_mesh_topology(points, triangle_net_list)
    return topology.triangles.tolist()
