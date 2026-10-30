"""Marching-triangles extraction of inferred pinch-out fronts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np
from numpy.typing import NDArray

from .mesh_topology import MeshTopology


class PinchFrontError(ValueError):
    """Raised when a unit-presence contour is geometrically ambiguous."""


@dataclass(frozen=True, slots=True)
class PinchFront:
    """A deduplicated threshold contour used to build the conformal mesh."""

    body_index: int
    tau: float
    points_xy: NDArray[np.float64]
    point_node_ids: NDArray[np.int64]
    point_edge_ids: NDArray[np.int64]
    point_cell_ids: tuple[tuple[int, ...], ...]
    segments: NDArray[np.int64]
    segment_cell_ids: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class _PointSpec:
    key: tuple[Hashable, ...]
    xy: tuple[float, float]
    node_id: int
    edge_id: int


def _readonly(array: NDArray) -> NDArray:
    array.setflags(write=False)
    return array


def _validate_inputs(
    xy: Sequence[Sequence[float]] | NDArray,
    topology: MeshTopology,
    presence_values: Sequence[float] | NDArray,
    *,
    tau: float,
    level_atol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    nodes = np.asarray(xy, dtype=float)
    values = np.asarray(presence_values, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise PinchFrontError(f"xy must have shape (N, 2), got {nodes.shape}.")
    if values.shape != (len(nodes),):
        raise PinchFrontError(
            f"presence_values must have shape ({len(nodes)},), got {values.shape}."
        )
    if len(topology.node_to_cells) != len(nodes):
        raise PinchFrontError("topology node count does not match xy.")
    if not np.isfinite(nodes).all() or not np.isfinite(values).all():
        raise PinchFrontError("xy and unit-presence values must be finite.")
    if not np.isfinite(tau) or not 0 < tau < 1:
        raise ValueError("tau must be finite and strictly between zero and one.")
    if not np.isfinite(level_atol) or level_atol < 0:
        raise ValueError("level_atol must be finite and non-negative.")
    return nodes, values


def extract_pinch_front(
    xy: Sequence[Sequence[float]] | NDArray,
    topology: MeshTopology,
    presence_values: Sequence[float] | NDArray,
    *,
    tau: float,
    body_index: int,
    level_atol: float = 1.0e-10,
) -> PinchFront:
    """Extract a threshold contour by piecewise-linear marching triangles.

    Parameters
    ----------
    xy : array_like, shape (N, 2)
        Original planar coordinates in metres.
    topology : MeshTopology
        Original triangle connectivity and unique edges.
    presence_values : array_like, shape (N,)
        Dimensionless harmonic values for one body.
    tau : float
        Dimensionless contour threshold.
    body_index : int
        Dense body index.
    level_atol : float, optional
        Tolerance for values lying exactly on the contour level.

    Returns
    -------
    PinchFront
        Contour points (P, 2) in metres, segments (S, 2), and parent provenance.

    Notes
    -----
    A shared original edge has one intersection point ID. Exact threshold nodes
    retain their original node identity. Duplicate segments are merged while
    all contributing parent cells are retained.
    """

    nodes, values = _validate_inputs(
        xy, topology, presence_values, tau=tau, level_atol=level_atol
    )
    if body_index < 0:
        raise ValueError("body_index must be non-negative.")

    edge_lookup = {
        (int(edge[0]), int(edge[1])): edge_id
        for edge_id, edge in enumerate(topology.edges)
    }
    point_ids: dict[tuple[Hashable, ...], int] = {}
    point_specs: list[_PointSpec] = []
    point_cells: list[set[int]] = []
    segment_ids: dict[tuple[int, int], int] = {}
    segments: list[tuple[int, int]] = []
    segment_cells: list[set[int]] = []

    def node_spec(node_id: int) -> _PointSpec:
        return _PointSpec(
            key=("node", node_id),
            xy=(float(nodes[node_id, 0]), float(nodes[node_id, 1])),
            node_id=node_id,
            edge_id=-1,
        )

    def edge_spec(node_a: int, node_b: int) -> _PointSpec:
        edge_nodes = tuple(sorted((node_a, node_b)))
        edge_id = edge_lookup[edge_nodes]
        value_a = float(values[edge_nodes[0]])
        value_b = float(values[edge_nodes[1]])
        denominator = value_b - value_a
        if denominator == 0:
            raise PinchFrontError(
                f"cannot interpolate threshold on constant edge {edge_nodes}."
            )
        fraction = (tau - value_a) / denominator
        if fraction < -level_atol or fraction > 1 + level_atol:
            raise PinchFrontError(
                f"edge {edge_nodes} threshold fraction is outside [0, 1]: {fraction}."
            )
        fraction = min(1.0, max(0.0, fraction))
        point = (1.0 - fraction) * nodes[edge_nodes[0]] + fraction * nodes[
            edge_nodes[1]
        ]
        return _PointSpec(
            key=("edge", edge_id),
            xy=(float(point[0]), float(point[1])),
            node_id=-1,
            edge_id=edge_id,
        )

    def register_point(spec: _PointSpec) -> int:
        existing = point_ids.get(spec.key)
        if existing is not None:
            return existing
        point_id = len(point_specs)
        point_ids[spec.key] = point_id
        point_specs.append(spec)
        point_cells.append(set())
        return point_id

    def register_segment(
        spec_a: _PointSpec,
        spec_b: _PointSpec,
        *,
        cell_id: int,
    ) -> None:
        point_a = register_point(spec_a)
        point_b = register_point(spec_b)
        if point_a == point_b:
            return
        point_cells[point_a].add(cell_id)
        point_cells[point_b].add(cell_id)
        key = tuple(sorted((point_a, point_b)))
        existing = segment_ids.get(key)
        if existing is None:
            segment_id = len(segments)
            segment_ids[key] = segment_id
            segments.append(key)
            segment_cells.append({cell_id})
        else:
            segment_cells[existing].add(cell_id)

    for cell_id, triangle in enumerate(topology.triangles):
        cell_nodes = [int(node) for node in triangle]
        cell_values = values[cell_nodes]
        equal = np.abs(cell_values - tau) <= level_atol
        if np.all(equal):
            raise PinchFrontError(
                f"cell {cell_id} lies entirely on unit-presence threshold {tau}; "
                "the front is not uniquely defined inside that triangle."
            )

        local_pairs = (
            (cell_nodes[0], cell_nodes[1]),
            (cell_nodes[1], cell_nodes[2]),
            (cell_nodes[2], cell_nodes[0]),
        )
        level_edges = [
            (node_a, node_b)
            for node_a, node_b in local_pairs
            if abs(values[node_a] - tau) <= level_atol
            and abs(values[node_b] - tau) <= level_atol
        ]
        if level_edges:
            if len(level_edges) != 1:
                raise PinchFrontError(
                    f"cell {cell_id} has multiple threshold edges and an "
                    "ambiguous contour."
                )
            node_a, node_b = level_edges[0]
            register_segment(
                node_spec(node_a),
                node_spec(node_b),
                cell_id=cell_id,
            )
            continue

        intersections: dict[tuple[Hashable, ...], _PointSpec] = {}
        for node_a, node_b in local_pairs:
            value_a = float(values[node_a])
            value_b = float(values[node_b])
            equal_a = abs(value_a - tau) <= level_atol
            equal_b = abs(value_b - tau) <= level_atol
            if equal_a:
                spec = node_spec(node_a)
                intersections[spec.key] = spec
            elif equal_b:
                spec = node_spec(node_b)
                intersections[spec.key] = spec
            elif (value_a - tau) * (value_b - tau) < 0:
                spec = edge_spec(node_a, node_b)
                intersections[spec.key] = spec

        if len(intersections) <= 1:
            continue
        if len(intersections) != 2:
            raise PinchFrontError(
                f"cell {cell_id} has {len(intersections)} unique contour "
                "intersections; expected zero, one, or two."
            )
        spec_a, spec_b = intersections.values()
        register_segment(
            spec_a,
            spec_b,
            cell_id=cell_id,
        )

    points_xy = np.asarray([spec.xy for spec in point_specs], dtype=float).reshape(
        -1, 2
    )
    point_node_ids = np.asarray([spec.node_id for spec in point_specs], dtype=np.int64)
    point_edge_ids = np.asarray([spec.edge_id for spec in point_specs], dtype=np.int64)
    segment_array = np.asarray(segments, dtype=np.int64).reshape(-1, 2)

    return PinchFront(
        body_index=body_index,
        tau=float(tau),
        points_xy=_readonly(points_xy),
        point_node_ids=_readonly(point_node_ids),
        point_edge_ids=_readonly(point_edge_ids),
        point_cell_ids=tuple(tuple(sorted(cells)) for cells in point_cells),
        segments=_readonly(segment_array),
        segment_cell_ids=tuple(tuple(sorted(cells)) for cells in segment_cells),
    )
