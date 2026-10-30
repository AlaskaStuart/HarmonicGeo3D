"""Validated, deterministic adjacency for a shared triangular XY mesh."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


class MeshTopologyError(ValueError):
    """Raised when nodes and triangles cannot form a valid manifold mesh."""


@dataclass(frozen=True, slots=True)
class MeshTopology:
    """CCW triangles and their immutable edge/node adjacency."""

    triangles: NDArray[np.int64]
    edges: NDArray[np.int64]
    edge_to_cells: tuple[tuple[int, ...], ...]
    node_to_cells: tuple[tuple[int, ...], ...]


def _readonly(array: NDArray) -> NDArray:
    array.setflags(write=False)
    return array


def _validated_xy(xy: Sequence[Sequence[float]] | NDArray) -> NDArray[np.float64]:
    nodes = np.asarray(xy, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise MeshTopologyError(f"xy must have shape (N, 2), got {nodes.shape}.")
    if len(nodes) < 3:
        raise MeshTopologyError("xy must contain at least three nodes.")
    if not np.isfinite(nodes).all():
        raise MeshTopologyError("xy contains non-finite coordinates.")
    if len(np.unique(nodes, axis=0)) != len(nodes):
        raise MeshTopologyError("xy contains duplicate coordinates.")
    return nodes


def _validated_triangles(
    triangles: Sequence[Sequence[int]] | NDArray,
    *,
    node_count: int,
) -> NDArray[np.int64]:
    source = np.asarray(triangles)
    if source.ndim != 2 or source.shape[1] != 3:
        raise MeshTopologyError(
            f"triangles must have shape (M, 3), got {source.shape}."
        )
    if len(source) == 0:
        raise MeshTopologyError("triangles must contain at least one cell.")
    if not np.issubdtype(source.dtype, np.integer):
        raise MeshTopologyError("triangle node indices must use an integer dtype.")

    cells = np.asarray(source, dtype=np.int64).copy()
    repeated_node_cells = [
        cell_id for cell_id, cell in enumerate(cells) if len(set(cell)) != 3
    ]
    if repeated_node_cells:
        raise MeshTopologyError(
            f"triangles contain repeated node indices in cells {repeated_node_cells}."
        )
    if np.any(cells < 0) or np.any(cells >= node_count):
        bad_cells = np.flatnonzero(
            np.any((cells < 0) | (cells >= node_count), axis=1)
        ).tolist()
        raise MeshTopologyError(
            f"triangles reference nodes outside [0, {node_count - 1}] "
            f"in cells {bad_cells}."
        )

    canonical = np.sort(cells, axis=1)
    _, inverse, counts = np.unique(
        canonical, axis=0, return_inverse=True, return_counts=True
    )
    duplicate_groups = [
        np.flatnonzero(inverse == group_id).tolist()
        for group_id, count in enumerate(counts)
        if count > 1
    ]
    if duplicate_groups:
        raise MeshTopologyError(
            f"duplicate triangles detected independent of winding: {duplicate_groups}."
        )
    return cells


def _orient_ccw(
    xy: NDArray[np.float64],
    triangles: NDArray[np.int64],
    *,
    area_atol: float,
) -> NDArray[np.int64]:
    if not np.isfinite(area_atol) or area_atol < 0:
        raise ValueError("area_atol must be finite and non-negative.")

    points = xy[triangles]
    signed_double_area = (points[:, 1, 0] - points[:, 0, 0]) * (
        points[:, 2, 1] - points[:, 0, 1]
    ) - (points[:, 1, 1] - points[:, 0, 1]) * (points[:, 2, 0] - points[:, 0, 0])
    degenerate = np.flatnonzero(0.5 * np.abs(signed_double_area) <= area_atol)
    if len(degenerate):
        raise MeshTopologyError(
            "zero-area or below-tolerance triangles detected in cells "
            f"{degenerate.tolist()}."
        )

    ccw = triangles.copy()
    clockwise = signed_double_area < 0
    ccw[clockwise, 1], ccw[clockwise, 2] = (
        triangles[clockwise, 2],
        triangles[clockwise, 1],
    )
    return ccw


def build_mesh_topology(
    xy: Sequence[Sequence[float]] | NDArray,
    triangles: Sequence[Sequence[int]] | NDArray,
    *,
    area_atol: float = 0.0,
) -> MeshTopology:
    """Validate triangle geometry, orient cells CCW, and build adjacency.

    Parameters
    ----------
    xy : array_like, shape (N, 2)
        Finite, unique XY locations in metres.
    triangles : array_like, shape (M, 3)
        Integer node indices, using every input node.
    area_atol : float, optional
        Minimum accepted triangle area in square metres.

    Returns
    -------
    MeshTopology
        Counter-clockwise triangles, unique undirected edges, and adjacency.
        Edge IDs are lexicographic; adjacency IDs are sorted.

    Raises
    ------
    MeshTopologyError
        If geometry, indices, or mesh connectivity are invalid.
    """

    nodes = _validated_xy(xy)
    source_cells = _validated_triangles(triangles, node_count=len(nodes))
    cells = _orient_ccw(nodes, source_cells, area_atol=area_atol)

    used_nodes = np.unique(cells)
    if len(used_nodes) != len(nodes):
        unused = np.setdiff1d(np.arange(len(nodes)), used_nodes).tolist()
        raise MeshTopologyError(f"mesh leaves nodes unused: {unused}.")

    edge_cells: dict[tuple[int, int], list[int]] = {}
    cell_edge_keys: list[tuple[tuple[int, int], ...]] = []
    node_cells: list[list[int]] = [[] for _ in range(len(nodes))]

    for cell_id, (node_a, node_b, node_c) in enumerate(cells.tolist()):
        local_edges = (
            tuple(sorted((node_a, node_b))),
            tuple(sorted((node_b, node_c))),
            tuple(sorted((node_c, node_a))),
        )
        cell_edge_keys.append(local_edges)
        for edge in local_edges:
            edge_cells.setdefault(edge, []).append(cell_id)
        for node_id in (node_a, node_b, node_c):
            node_cells[node_id].append(cell_id)

    nonmanifold = {
        edge: tuple(cell_ids)
        for edge, cell_ids in edge_cells.items()
        if len(cell_ids) > 2
    }
    if nonmanifold:
        raise MeshTopologyError(f"non-manifold edges detected: {nonmanifold}.")

    edge_keys = sorted(edge_cells)
    edges = np.asarray(edge_keys, dtype=np.int64)
    edge_to_cells = tuple(tuple(sorted(edge_cells[edge])) for edge in edge_keys)
    node_to_cells = tuple(tuple(sorted(cell_ids)) for cell_ids in node_cells)

    return MeshTopology(
        triangles=_readonly(cells),
        edges=_readonly(edges),
        edge_to_cells=edge_to_cells,
        node_to_cells=node_to_cells,
    )
