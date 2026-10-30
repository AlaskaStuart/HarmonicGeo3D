"""Deterministic local remeshing that makes all pinch fronts true mesh edges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .mesh_topology import MeshTopology, MeshTopologyError, build_mesh_topology
from .pinch_front import PinchFront

ORIGINAL_NODE = 0
ORIGINAL_EDGE_NODE = 1
ORIGINAL_CELL_NODE = 2


class ConformalMeshError(ValueError):
    """Raised when front constraints cannot form an unambiguous shared mesh."""


def _readonly(array: NDArray) -> NDArray:
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class ConformalMesh:
    """One XY topology shared by all interfaces and downstream model data."""

    nodes_xy: NDArray[np.float64]
    topology: MeshTopology
    unit_presence_values: NDArray[np.float64]
    cell_presence_mask: NDArray[np.bool_]
    pinch_edge_mask: NDArray[np.bool_]
    zero_thickness_mask: NDArray[np.bool_]
    observed_z: NDArray[np.float64]
    z_known: NDArray[np.bool_]
    source_row_indices: NDArray[np.int64]
    drill_names: NDArray[np.str_]
    node_source_kind: NDArray[np.int8]
    node_canonical_parent_cells: NDArray[np.int64]
    node_parent_barycentric: NDArray[np.float64]
    cell_parent_ids: NDArray[np.int64]
    geometry_tolerance: float
    area_tolerance: float
    presence_tolerance: float
    original_node_count: int


@dataclass(slots=True)
class _RawSegment:
    node_a: int
    node_b: int
    body_mask: NDArray[np.bool_]
    parent_cells: tuple[int, ...]
    split_points: dict[float, int]


def _cross(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _barycentric(
    point: NDArray[np.float64], triangle: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Compute parent-triangle barycentric provenance in relative coordinates."""
    matrix = np.column_stack((triangle[1] - triangle[0], triangle[2] - triangle[0]))
    try:
        uv = np.linalg.solve(matrix, point - triangle[0])
    except np.linalg.LinAlgError as exc:  # guarded by base topology validation
        raise ConformalMeshError(
            "cannot compute provenance in a degenerate cell."
        ) from exc
    return np.asarray((1.0 - uv[0] - uv[1], uv[0], uv[1]), dtype=float)


def _point_on_segment(
    point: NDArray[np.float64],
    start: NDArray[np.float64],
    end: NDArray[np.float64],
    tolerance: float,
) -> tuple[bool, float]:
    vector = end - start
    length2 = float(vector @ vector)
    if length2 == 0:
        return False, float("nan")
    parameter = float((point - start) @ vector / length2)
    projected = start + parameter * vector
    return (
        -tolerance / np.sqrt(length2) <= parameter <= 1 + tolerance / np.sqrt(length2)
        and float(np.linalg.norm(point - projected)) <= tolerance,
        parameter,
    )


def _ear_clip(
    polygon: list[int],
    nodes: list[NDArray[np.float64]],
    area_atol: float,
) -> list[tuple[int, int, int]]:
    """Triangulate one simple CCW polygon without adding Steiner nodes."""

    cleaned: list[int] = []
    for node in polygon:
        if not cleaned or node != cleaned[-1]:
            cleaned.append(node)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    if len(cleaned) < 3:
        raise ConformalMeshError("local planar graph contains a face with <3 nodes.")

    def triangle_area(a: int, b: int, c: int) -> float:
        return 0.5 * _cross(nodes[b] - nodes[a], nodes[c] - nodes[a])

    def inside(point: NDArray[np.float64], a: int, b: int, c: int) -> bool:
        pa, pb, pc = nodes[a], nodes[b], nodes[c]
        return (
            _cross(pb - pa, point - pa) >= -area_atol
            and _cross(pc - pb, point - pb) >= -area_atol
            and _cross(pa - pc, point - pc) >= -area_atol
        )

    remaining = cleaned.copy()
    result: list[tuple[int, int, int]] = []
    while len(remaining) > 3:
        found = False
        for index in range(len(remaining)):
            previous = remaining[index - 1]
            current = remaining[index]
            following = remaining[(index + 1) % len(remaining)]
            if triangle_area(previous, current, following) <= area_atol:
                continue
            if any(
                inside(nodes[other], previous, current, following)
                for other in remaining
                if other not in (previous, current, following)
            ):
                continue
            result.append((previous, current, following))
            del remaining[index]
            found = True
            break
        if not found:
            raise ConformalMeshError(
                "deterministic ear clipping failed; the local face is self-"
                "intersecting, ambiguous, or below area tolerance."
            )
    if triangle_area(*remaining) <= area_atol:
        raise ConformalMeshError("ear clipping produced a below-tolerance triangle.")
    result.append(tuple(remaining))
    return result


def _planar_faces(
    graph_edges: set[tuple[int, int]],
    nodes: list[NDArray[np.float64]],
    area_atol: float,
) -> list[list[int]]:
    """Walk planar half-edges to recover bounded counter-clockwise faces."""
    adjacency: dict[int, set[int]] = {}
    for node_a, node_b in graph_edges:
        if node_a == node_b:
            raise ConformalMeshError("local graph contains a zero-length edge.")
        adjacency.setdefault(node_a, set()).add(node_b)
        adjacency.setdefault(node_b, set()).add(node_a)
    ordered: dict[int, list[int]] = {}
    for node, neighbours in adjacency.items():
        ordered[node] = sorted(
            neighbours,
            key=lambda neighbour: (
                np.arctan2(
                    nodes[neighbour][1] - nodes[node][1],
                    nodes[neighbour][0] - nodes[node][0],
                ),
                neighbour,
            ),
        )

    visited: set[tuple[int, int]] = set()
    faces: list[list[int]] = []
    for edge in sorted(graph_edges):
        for start in (edge, (edge[1], edge[0])):
            if start in visited:
                continue
            face: list[int] = []
            current = start
            for _ in range(2 * len(graph_edges) + 2):
                if current in visited:
                    if current != start:
                        raise ConformalMeshError(
                            "local half-edge walk closed inconsistently."
                        )
                    break
                visited.add(current)
                node_a, node_b = current
                face.append(node_a)
                neighbours = ordered[node_b]
                reverse_index = neighbours.index(node_a)
                node_c = neighbours[(reverse_index - 1) % len(neighbours)]
                current = (node_b, node_c)
                if current == start:
                    break
            else:
                raise ConformalMeshError("local half-edge walk did not terminate.")
            coordinates = np.asarray([nodes[node] for node in face])
            signed_area = 0.5 * float(
                np.sum(
                    coordinates[:, 0] * np.roll(coordinates[:, 1], -1)
                    - coordinates[:, 1] * np.roll(coordinates[:, 0], -1)
                )
            )
            if signed_area > area_atol:
                faces.append(face)
    if not faces:
        raise ConformalMeshError("local planar graph contains no bounded CCW face.")
    return faces


def build_conformal_mesh(
    xy: Sequence[Sequence[float]] | NDArray,
    topology: MeshTopology,
    unit_presence_values: Sequence[Sequence[float]] | NDArray,
    fronts: Sequence[PinchFront],
    *,
    tau: Sequence[float],
    observed_z: Sequence[Sequence[float]] | NDArray | None = None,
    z_known: Sequence[Sequence[bool]] | NDArray | None = None,
    source_row_indices: Sequence[int] | NDArray | None = None,
    drill_names: Sequence[str] | NDArray | None = None,
    geometry_atol: float,
    area_atol: float,
    presence_atol: float = 1.0e-9,
) -> ConformalMesh:
    """Insert pinch-front intersections and retriangulate affected cells.

    Parameters
    ----------
    xy : array_like, shape (N, 2)
        Original planar coordinates in metres.
    topology : MeshTopology
        Original counter-clockwise triangle connectivity.
    unit_presence_values : array_like, shape (N, B)
        Dimensionless harmonic fields at original nodes.
    fronts : sequence of PinchFront
        One threshold-front description for each body.
    tau : sequence of float, shape (B,)
        Dimensionless presence thresholds.
    observed_z, z_known : array_like, optional
        Elevations in metres and observation flags, both of shape (N, I).
    source_row_indices, drill_names : array_like, optional
        Observation row identifiers and borehole names, each of length N.
    geometry_atol, area_atol : float
        Geometric length (m) and area (m2) tolerances.
    presence_atol : float, optional
        Dimensionless threshold tolerance.

    Returns
    -------
    ConformalMesh
        Shared final mesh, presence and zero-thickness masks, and geometric
        provenance. New nodes receive no observed interface elevations.

    Raises
    ------
    ConformalMeshError
        If constraints cannot form an unambiguous shared mesh.
    """

    base_xy = np.asarray(xy, dtype=float)
    presence_values = np.asarray(unit_presence_values, dtype=float)
    thresholds = np.asarray(tau, dtype=float)
    node_count = len(base_xy)
    body_count = len(thresholds)
    if base_xy.shape != (node_count, 2) or node_count != len(topology.node_to_cells):
        raise ConformalMeshError("xy and topology node counts do not agree.")
    if presence_values.shape != (node_count, body_count):
        raise ConformalMeshError(
            f"unit_presence_values must have shape ({node_count}, {body_count})."
        )
    if not np.isfinite(base_xy).all() or not np.isfinite(presence_values).all():
        raise ConformalMeshError("xy and unit-presence values must be finite.")
    if thresholds.ndim != 1 or not np.isfinite(thresholds).all():
        raise ConformalMeshError("tau must be a finite one-dimensional array.")
    if geometry_atol <= 0 or area_atol < 0 or presence_atol < 0:
        raise ValueError(
            "Conformal mesh tolerances must be non-negative and geometry_atol positive."
        )
    if len(fronts) != body_count:
        raise ConformalMeshError("one PinchFront is required for every body.")

    nodes = [point.copy() for point in base_xy]
    source_kind = [ORIGINAL_NODE] * node_count
    canonical_cells = [min(cells) for cells in topology.node_to_cells]
    barycentric = [
        _barycentric(base_xy[i], base_xy[topology.triangles[canonical_cells[i]]])
        for i in range(node_count)
    ]
    edge_nodes: dict[int, list[int]] = {
        edge_id: [] for edge_id in range(len(topology.edges))
    }

    def locate_original_edge(point: NDArray[np.float64]) -> int | None:
        matches: list[tuple[int, float]] = []
        for edge_id, edge in enumerate(topology.edges):
            on_edge, fraction = _point_on_segment(
                point, base_xy[int(edge[0])], base_xy[int(edge[1])], geometry_atol
            )
            if on_edge:
                matches.append((edge_id, min(1.0, max(0.0, fraction))))
        if not matches:
            return None
        interior = [
            item for item in matches if geometry_atol < item[1] < 1 - geometry_atol
        ]
        return min(interior or matches)[0]

    def register_point(
        point: NDArray[np.float64],
        *,
        cells: set[int],
        edge_hint: int = -1,
    ) -> int:
        distances = np.linalg.norm(base_xy - point, axis=1)
        if float(np.min(distances)) <= geometry_atol:
            node_id = int(np.argmin(distances))
            if float(distances[node_id]) > geometry_atol:
                raise ConformalMeshError("ambiguous original-node snap.")
            return node_id
        for node_id in range(node_count, len(nodes)):
            if float(np.linalg.norm(nodes[node_id] - point)) <= geometry_atol:
                return node_id

        edge_id = None
        if edge_hint >= 0:
            edge = topology.edges[edge_hint]
            on_edge, _ = _point_on_segment(
                point, base_xy[int(edge[0])], base_xy[int(edge[1])], geometry_atol
            )
            if not on_edge:
                raise ConformalMeshError(
                    "front edge provenance does not match geometry."
                )
            edge_id = edge_hint
        else:
            edge_id = locate_original_edge(point)

        node_id = len(nodes)
        nodes.append(point.copy())
        if edge_id is not None:
            source_kind.append(ORIGINAL_EDGE_NODE)
            cells.update(topology.edge_to_cells[edge_id])
            edge_nodes[edge_id].append(node_id)
        else:
            source_kind.append(ORIGINAL_CELL_NODE)
        if not cells:
            raise ConformalMeshError("new front node has no original parent cell.")
        canonical = min(cells)
        canonical_cells.append(canonical)
        barycentric.append(_barycentric(point, base_xy[topology.triangles[canonical]]))
        return node_id

    raw_segments: list[_RawSegment] = []
    for body, front in enumerate(fronts):
        if front.body_index != body:
            raise ConformalMeshError("front body indices must match sequence order.")
        if abs(front.tau - thresholds[body]) > presence_atol:
            raise ConformalMeshError(
                f"front {body} threshold {front.tau} does not match tau "
                f"{thresholds[body]}."
            )
        body_mask = np.zeros(body_count, dtype=bool)
        body_mask[body] = True
        mapped: list[int] = []
        for point_id, point in enumerate(front.points_xy):
            cells = set(front.point_cell_ids[point_id])
            node_id = int(front.point_node_ids[point_id])
            if node_id >= 0:
                if (
                    node_id >= node_count
                    or np.linalg.norm(base_xy[node_id] - point) > geometry_atol
                ):
                    raise ConformalMeshError(
                        "front original-node provenance is invalid."
                    )
                mapped.append(node_id)
            else:
                mapped.append(
                    register_point(
                        np.asarray(point, dtype=float),
                        cells=cells,
                        edge_hint=int(front.point_edge_ids[point_id]),
                    )
                )
        for segment_id, segment in enumerate(front.segments):
            node_a, node_b = mapped[int(segment[0])], mapped[int(segment[1])]
            if (
                node_a == node_b
                or np.linalg.norm(nodes[node_a] - nodes[node_b]) <= geometry_atol
            ):
                raise ConformalMeshError("front contains a zero-length segment.")
            cells = tuple(sorted(set(front.segment_cell_ids[segment_id])))
            if not cells:
                raise ConformalMeshError("front segment has no parent cell.")
            raw_segments.append(
                _RawSegment(
                    node_a, node_b, body_mask.copy(), cells, {0.0: node_a, 1.0: node_b}
                )
            )

    # Split every pair that coexists in an original cell. Collinear overlaps are
    # split at all overlap endpoints; coincident child edges are merged later.
    for first in range(len(raw_segments)):
        segment_a = raw_segments[first]
        point_a = nodes[segment_a.node_a]
        vector_a = nodes[segment_a.node_b] - point_a
        length_a = float(np.linalg.norm(vector_a))
        for second in range(first + 1, len(raw_segments)):
            segment_b = raw_segments[second]
            common_cells = set(segment_a.parent_cells) & set(segment_b.parent_cells)
            if not common_cells:
                continue
            point_b = nodes[segment_b.node_a]
            vector_b = nodes[segment_b.node_b] - point_b
            length_b = float(np.linalg.norm(vector_b))
            denominator = _cross(vector_a, vector_b)
            offset = point_b - point_a
            scaled_tolerance = geometry_atol * max(length_a, length_b, 1.0)
            if abs(denominator) > scaled_tolerance:
                t = _cross(offset, vector_b) / denominator
                u = _cross(offset, vector_a) / denominator
                parameter_tolerance = geometry_atol / min(length_a, length_b)
                if (
                    -parameter_tolerance <= t <= 1 + parameter_tolerance
                    and -parameter_tolerance <= u <= 1 + parameter_tolerance
                ):
                    t = min(1.0, max(0.0, t))
                    u = min(1.0, max(0.0, u))
                    point = point_a + t * vector_a
                    node_id = register_point(point, cells=common_cells)
                    segment_a.split_points[float(t)] = node_id
                    segment_b.split_points[float(u)] = node_id
                continue
            if abs(_cross(offset, vector_a)) > scaled_tolerance:
                continue
            # Collinear: project all endpoints and split both segments at every
            # endpoint that lies on the other segment.
            events = 0
            for source, target, source_point, source_vector in (
                (segment_a, segment_b, point_a, vector_a),
                (segment_b, segment_a, point_b, vector_b),
            ):
                target_start = nodes[target.node_a]
                target_vector = nodes[target.node_b] - target_start
                target_length2 = float(target_vector @ target_vector)
                for parameter, node_id in ((0.0, source.node_a), (1.0, source.node_b)):
                    point = source_point + parameter * source_vector
                    target_parameter = float(
                        (point - target_start) @ target_vector / target_length2
                    )
                    target_length = float(np.sqrt(target_length2))
                    if (
                        -geometry_atol / target_length
                        <= target_parameter
                        <= 1 + geometry_atol / target_length
                    ):
                        target_parameter = min(1.0, max(0.0, target_parameter))
                        target.split_points[target_parameter] = node_id
                        events += 1
            if events == 0:
                continue

    constrained_edges: dict[tuple[int, int], NDArray[np.bool_]] = {}
    constraint_cells: dict[tuple[int, int], set[int]] = {}
    for segment in raw_segments:
        ordered = sorted(segment.split_points.items())
        for (_, node_a), (_, node_b) in zip(ordered[:-1], ordered[1:]):
            if (
                node_a == node_b
                or np.linalg.norm(nodes[node_a] - nodes[node_b]) <= geometry_atol
            ):
                continue
            edge = tuple(sorted((node_a, node_b)))
            if edge in constrained_edges:
                constrained_edges[edge] |= segment.body_mask
                constraint_cells[edge].update(segment.parent_cells)
            else:
                constrained_edges[edge] = segment.body_mask.copy()
                constraint_cells[edge] = set(segment.parent_cells)

    final_triangles: list[tuple[int, int, int]] = []
    cell_parent_ids: list[int] = []
    for parent_id, triangle_array in enumerate(topology.triangles):
        triangle = tuple(map(int, triangle_array))
        graph_edges: set[tuple[int, int]] = set()
        has_boundary_split = False
        for index in range(3):
            start = triangle[index]
            end = triangle[(index + 1) % 3]
            edge_key = tuple(sorted((start, end)))
            edge_id = int(np.flatnonzero(np.all(topology.edges == edge_key, axis=1))[0])
            additions = edge_nodes[edge_id]
            directed = [start, *additions, end]
            vector = nodes[end] - nodes[start]
            directed.sort(key=lambda node: float((nodes[node] - nodes[start]) @ vector))
            if len(directed) > 2:
                has_boundary_split = True
            graph_edges.update(
                tuple(sorted((node_a, node_b)))
                for node_a, node_b in zip(directed[:-1], directed[1:])
            )
        local_constraints = [
            edge for edge, cells in constraint_cells.items() if parent_id in cells
        ]
        graph_edges.update(local_constraints)
        if not has_boundary_split and not local_constraints:
            final_triangles.append(triangle)
            cell_parent_ids.append(parent_id)
            continue
        for face in _planar_faces(graph_edges, nodes, area_atol):
            for child in _ear_clip(face, nodes, area_atol):
                final_triangles.append(child)
                cell_parent_ids.append(parent_id)

    node_array = np.asarray(nodes, dtype=float)
    try:
        final_topology = build_mesh_topology(
            node_array, np.asarray(final_triangles, dtype=np.int64), area_atol=area_atol
        )
    except MeshTopologyError as exc:
        raise ConformalMeshError(f"final conformal topology is invalid: {exc}") from exc

    final_presence_values = np.empty((len(nodes), body_count), dtype=float)
    final_presence_values[:node_count] = presence_values
    for node_id in range(node_count, len(nodes)):
        parent = canonical_cells[node_id]
        weights = barycentric[node_id]
        final_presence_values[node_id] = (
            weights @ presence_values[topology.triangles[parent]]
        )

    cell_presence_mask = np.empty(
        (len(final_topology.triangles), body_count), dtype=bool
    )
    for cell_id, triangle in enumerate(final_topology.triangles):
        values = final_presence_values[triangle]
        centroid_values = np.mean(values, axis=0)
        near = np.abs(centroid_values - thresholds) <= presence_atol
        if np.any(near):
            raise ConformalMeshError(
                f"final cell {cell_id} centroid lies on a unit-presence threshold for "
                f"bodies {np.flatnonzero(near).tolist()}."
            )
        crosses = (np.min(values, axis=0) < thresholds - presence_atol) & (
            np.max(values, axis=0) > thresholds + presence_atol
        )
        if np.any(crosses):
            raise ConformalMeshError(
                f"final cell {cell_id} crosses unit-presence boundaries for bodies "
                f"{np.flatnonzero(crosses).tolist()}."
            )
        cell_presence_mask[cell_id] = centroid_values > thresholds

    edge_lookup = {
        tuple(map(int, edge)): edge_id
        for edge_id, edge in enumerate(final_topology.edges)
    }
    pinch_edge_mask = np.zeros((len(final_topology.edges), body_count), dtype=bool)
    for edge, mask in constrained_edges.items():
        edge_id = edge_lookup.get(edge)
        if edge_id is None:
            raise ConformalMeshError(f"front segment {edge} is not a final mesh edge.")
        pinch_edge_mask[edge_id] |= mask

    zero_mask = np.zeros((len(nodes), body_count), dtype=bool)
    for body in range(body_count):
        absent_cells = np.flatnonzero(~cell_presence_mask[:, body])
        if len(absent_cells):
            zero_mask[np.unique(final_topology.triangles[absent_cells]), body] = True
        pinch_edges = np.flatnonzero(pinch_edge_mask[:, body])
        if len(pinch_edges):
            zero_mask[np.unique(final_topology.edges[pinch_edges]), body] = True

    interface_count = 0 if observed_z is None else np.asarray(observed_z).shape[1]
    source_z = (
        np.empty((node_count, interface_count), dtype=float)
        if observed_z is None
        else np.asarray(observed_z, dtype=float)
    )
    source_known = (
        np.zeros_like(source_z, dtype=bool)
        if z_known is None
        else np.asarray(z_known, dtype=bool)
    )
    if (
        source_z.shape != (node_count, interface_count)
        or source_known.shape != source_z.shape
    ):
        raise ConformalMeshError(
            "observed_z and z_known shapes do not match original nodes."
        )
    expanded_z = np.full((len(nodes), interface_count), np.nan, dtype=float)
    expanded_known = np.zeros((len(nodes), interface_count), dtype=bool)
    expanded_z[:node_count] = source_z
    expanded_known[:node_count] = source_known
    source_rows = (
        np.arange(node_count, dtype=np.int64)
        if source_row_indices is None
        else np.asarray(source_row_indices, dtype=np.int64)
    )
    if source_rows.shape != (node_count,):
        raise ConformalMeshError(
            "source_row_indices shape does not match original nodes."
        )
    expanded_rows = np.full(len(nodes), -1, dtype=np.int64)
    expanded_rows[:node_count] = source_rows
    source_names = (
        np.asarray([str(i) for i in range(node_count)], dtype=str)
        if drill_names is None
        else np.asarray(drill_names, dtype=str)
    )
    if source_names.shape != (node_count,):
        raise ConformalMeshError("drill_names shape does not match original nodes.")
    expanded_names = np.full(
        len(nodes), "", dtype=f"<U{max(1, source_names.dtype.itemsize // 4)}"
    )
    expanded_names[:node_count] = source_names
    return ConformalMesh(
        nodes_xy=_readonly(node_array),
        topology=final_topology,
        unit_presence_values=_readonly(final_presence_values),
        cell_presence_mask=_readonly(cell_presence_mask),
        pinch_edge_mask=_readonly(pinch_edge_mask),
        zero_thickness_mask=_readonly(zero_mask),
        observed_z=_readonly(expanded_z),
        z_known=_readonly(expanded_known),
        source_row_indices=_readonly(expanded_rows),
        drill_names=_readonly(expanded_names),
        node_source_kind=_readonly(np.asarray(source_kind, dtype=np.int8)),
        node_canonical_parent_cells=_readonly(
            np.asarray(canonical_cells, dtype=np.int64)
        ),
        node_parent_barycentric=_readonly(np.asarray(barycentric, dtype=float)),
        cell_parent_ids=_readonly(np.asarray(cell_parent_ids, dtype=np.int64)),
        geometry_tolerance=float(geometry_atol),
        area_tolerance=float(area_atol),
        presence_tolerance=float(presence_atol),
        original_node_count=node_count,
    )
