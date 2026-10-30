"""Geometric, numerical, and solid-topology validation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .conformal_mesh import ConformalMesh, ConformalMeshError
from .interface_qp import InterfaceSolution
from .mesh_topology import MeshTopology
from .solid_builder import SolidModel


class CompleteModelValidationError(ValueError):
    """Raised when the complete geological model fails an acceptance check."""


def _cross(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _proper_intersection(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    c: NDArray[np.float64],
    d: NDArray[np.float64],
    tolerance: float,
) -> bool:
    ab = b - a
    cd = d - c
    scale = max(float(np.linalg.norm(ab)), float(np.linalg.norm(cd)), 1.0)
    denominator = _cross(ab, cd)
    if abs(denominator) <= tolerance * scale:
        return False
    offset = c - a
    t = _cross(offset, cd) / denominator
    u = _cross(offset, ab) / denominator
    parameter_tolerance = tolerance / scale
    return (
        parameter_tolerance < t < 1 - parameter_tolerance
        and parameter_tolerance < u < 1 - parameter_tolerance
    )


def validate_conformal_mesh(
    mesh: ConformalMesh,
    original_xy: Sequence[Sequence[float]] | NDArray,
    original_topology: MeshTopology,
    original_presence_values: Sequence[Sequence[float]] | NDArray,
    *,
    tau: Sequence[float],
) -> None:
    """Check remeshing geometry, provenance, and zero-thickness memberships.

    Parameters
    ----------
    mesh : ConformalMesh
        Final mesh with geometric tolerances and parent provenance.
    original_xy : array_like, shape (N, 2)
        Original coordinates in metres.
    original_topology : MeshTopology
        Parent-cell topology.
    original_presence_values : array_like, shape (N, B)
        Harmonic fields at original nodes.
    tau : sequence of float, shape (B,)
        Presence thresholds.

    Raises
    ------
    ConformalMeshError
        If original values change, parent area is not conserved, fields are
        inconsistent, edges cross, or zero-thickness memberships are missing.
    """

    xy = mesh.nodes_xy
    original_xy = np.asarray(original_xy, dtype=float)
    original_presence_values = np.asarray(original_presence_values, dtype=float)
    thresholds = np.asarray(tau, dtype=float)
    errors: list[str] = []
    original_count = len(original_xy)
    if not np.array_equal(xy[:original_count], original_xy):
        errors.append("original node coordinates changed")
    if not np.array_equal(
        mesh.unit_presence_values[:original_count], original_presence_values
    ):
        errors.append("original unit-presence values changed")
    if np.any(mesh.z_known[original_count:]):
        errors.append("new nodes were marked as observed")
    if not np.isnan(mesh.observed_z[original_count:]).all():
        errors.append("new nodes received interface elevations")
    if np.any(mesh.source_row_indices[original_count:] != -1):
        errors.append("new nodes received source-row identifiers")

    original_areas = np.empty(len(original_topology.triangles), dtype=float)
    for parent, triangle in enumerate(original_topology.triangles):
        points = original_xy[triangle]
        original_areas[parent] = 0.5 * abs(
            _cross(points[1] - points[0], points[2] - points[0])
        )
    child_areas = np.zeros_like(original_areas)
    for cell_id, triangle in enumerate(mesh.topology.triangles):
        points = xy[triangle]
        double_area = _cross(points[1] - points[0], points[2] - points[0])
        child_areas[mesh.cell_parent_ids[cell_id]] += 0.5 * double_area
    area_residual = float(np.max(np.abs(child_areas - original_areas)))
    if area_residual > max(mesh.area_tolerance * 10, np.max(original_areas) * 1e-11):
        errors.append(f"parent-cell area conservation residual is {area_residual}")

    presence_residual = 0.0
    for node_id in range(original_count, len(xy)):
        parent = int(mesh.node_canonical_parent_cells[node_id])
        expected = (
            mesh.node_parent_barycentric[node_id]
            @ original_presence_values[original_topology.triangles[parent]]
        )
        presence_residual = max(
            presence_residual,
            float(np.max(np.abs(expected - mesh.unit_presence_values[node_id]))),
        )
    if presence_residual > 1e-10:
        errors.append(f"unit-presence interpolation residual is {presence_residual}")

    crossing_cells = 0
    threshold_centroid_cells = 0
    for triangle in mesh.topology.triangles:
        values = mesh.unit_presence_values[triangle]
        crossing_cells += int(
            np.any(
                (np.min(values, axis=0) < thresholds - mesh.presence_tolerance)
                & (np.max(values, axis=0) > thresholds + mesh.presence_tolerance)
            )
        )
        threshold_centroid_cells += int(
            np.any(
                np.abs(np.mean(values, axis=0) - thresholds) <= mesh.presence_tolerance
            )
        )
    if crossing_cells:
        errors.append(f"{crossing_cells} cells cross unit-presence boundaries")
    if threshold_centroid_cells:
        errors.append(f"{threshold_centroid_cells} cell centroids lie on thresholds")

    crossing_edge_pairs = 0
    edges = mesh.topology.edges
    for first, edge_a in enumerate(edges):
        nodes_a = set(map(int, edge_a))
        for edge_b in edges[first + 1 :]:
            if nodes_a & set(map(int, edge_b)):
                continue
            if _proper_intersection(
                xy[edge_a[0]],
                xy[edge_a[1]],
                xy[edge_b[0]],
                xy[edge_b[1]],
                mesh.geometry_tolerance,
            ):
                crossing_edge_pairs += 1
    if crossing_edge_pairs:
        errors.append(f"{crossing_edge_pairs} final edge pairs cross")

    missing_zero_memberships = 0
    for body in range(mesh.cell_presence_mask.shape[1]):
        absent = np.flatnonzero(~mesh.cell_presence_mask[:, body])
        expected_nodes: set[int] = set()
        if len(absent):
            expected_nodes.update(map(int, np.unique(mesh.topology.triangles[absent])))
        pinch = np.flatnonzero(mesh.pinch_edge_mask[:, body])
        if len(pinch):
            expected_nodes.update(map(int, np.unique(mesh.topology.edges[pinch])))
        missing_zero_memberships += sum(
            not mesh.zero_thickness_mask[node, body] for node in expected_nodes
        )
    if missing_zero_memberships:
        errors.append("zero-thickness mask misses absent/front nodes")

    if errors:
        raise ConformalMeshError(
            "Conformal mesh validation failed: " + "; ".join(errors)
        )


def _oriented_tetra_faces(
    tetrahedron: Sequence[int],
) -> tuple[tuple[int, int, int], ...]:
    a, b, c, d = map(int, tetrahedron)
    return ((b, c, d), (a, d, c), (a, b, d), (a, c, b))


def _orientation_parity(face: Sequence[int]) -> int:
    values = list(map(int, face))
    inversions = sum(
        values[first] > values[second]
        for first in range(3)
        for second in range(first + 1, 3)
    )
    return -1 if inversions % 2 else 1


def _body_face_data(
    tetrahedra: NDArray[np.int64], tetra_body_ids: NDArray[np.int64], body: int
) -> tuple[Counter[tuple[int, int, int]], dict[tuple[int, int, int], list[int]]]:
    counter: Counter[tuple[int, int, int]] = Counter()
    orientations: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for tetrahedron in tetrahedra[tetra_body_ids == body]:
        for face in _oriented_tetra_faces(tetrahedron):
            key = tuple(sorted(face))
            counter[key] += 1
            orientations[key].append(_orientation_parity(face))
    return counter, orientations


@dataclass(frozen=True, slots=True)
class SolidBoundaryAnalysis:
    """Tetrahedral boundary topology needed by the closure checks."""

    face_counters: tuple[Counter[tuple[int, int, int]], ...]
    boundary_closed: NDArray[np.bool_]
    nonmanifold_face_count: int
    inconsistent_internal_orientation_count: int


def analyze_solid_boundaries(
    solid_or_tetrahedra: SolidModel | NDArray[np.int64],
    tetra_body_ids: NDArray[np.int64] | None = None,
    *,
    body_count: int | None = None,
) -> SolidBoundaryAnalysis:
    """Analyze body boundary topology from tetrahedra.

    Parameters
    ----------
    solid_or_tetrahedra : SolidModel or ndarray, shape (T, 4)
        A solid model or raw tetrahedral node indices.
    tetra_body_ids : ndarray, shape (T,), optional
        Required body indices when raw tetrahedra are supplied.
    body_count : int, optional
        Declared body count; required with a SolidModel. Inferred from raw
        nonempty body indices when omitted.

    Returns
    -------
    SolidBoundaryAnalysis
        Face counters, per-body closure, and manifold/orientation diagnostics.

    Notes
    -----
    Face incidence is recomputed without using stored validation summaries.
    """

    if isinstance(solid_or_tetrahedra, SolidModel):
        tetrahedra = solid_or_tetrahedra.tetrahedra
        body_ids = solid_or_tetrahedra.tetra_body_ids
        if body_count is None:
            raise ValueError("body_count is required with a solid model.")
    else:
        tetrahedra = np.asarray(solid_or_tetrahedra, dtype=np.int64)
        if tetra_body_ids is None:
            raise ValueError("tetra_body_ids are required with raw tetrahedra.")
        body_ids = np.asarray(tetra_body_ids, dtype=np.int64)
        if body_count is None:
            if not len(body_ids):
                raise ValueError("body_count is required when no tetrahedra exist.")
            body_count = int(np.max(body_ids)) + 1
    if body_count < 1:
        raise ValueError("body_count must be positive.")
    if np.any(body_ids < 0) or np.any(body_ids >= body_count):
        raise ValueError("tetra_body_ids lie outside the declared body count.")
    face_counters: list[Counter[tuple[int, int, int]]] = []
    closed = np.zeros(body_count, dtype=bool)
    nonmanifold = 0
    inconsistent = 0
    for body in range(body_count):
        face_counter, orientations = _body_face_data(tetrahedra, body_ids, body)
        face_counters.append(face_counter)
        nonmanifold += sum(count > 2 for count in face_counter.values())
        inconsistent += sum(
            len(values) == 2 and values[0] == values[1]
            for values in orientations.values()
        )
        faces = np.asarray(
            sorted(face for face, count in face_counter.items() if count == 1),
            dtype=np.int64,
        ).reshape(-1, 3)
        boundary_edges: Counter[tuple[int, int]] = Counter()
        for face in faces:
            for edge in combinations(face, 2):
                boundary_edges[tuple(sorted(map(int, edge)))] += 1
        closed[body] = bool(boundary_edges) and all(
            count == 2 for count in boundary_edges.values()
        )
    closed.setflags(write=False)
    return SolidBoundaryAnalysis(
        tuple(face_counters),
        closed,
        int(nonmanifold),
        int(inconsistent),
    )


def _expected_side_faces(
    solid: SolidModel, edge: Sequence[int], body: int
) -> tuple[tuple[int, int, int], ...]:
    node_a, node_b = sorted(map(int, edge))
    lower_a = int(solid.surface_vertex_ids[node_a, body])
    lower_b = int(solid.surface_vertex_ids[node_b, body])
    upper_a = int(solid.surface_vertex_ids[node_a, body + 1])
    upper_b = int(solid.surface_vertex_ids[node_b, body + 1])
    candidates = (
        (lower_a, lower_b, upper_b),
        (lower_a, upper_a, upper_b),
    )
    return tuple(tuple(sorted(face)) for face in candidates if len(set(face)) == 3)


def validate_complete_model(
    mesh: ConformalMesh,
    solution: InterfaceSolution,
    solid: SolidModel,
    *,
    tau: Sequence[float],
    residual_atol: float = 1.0e-7,
    volume_abs_atol: float = 1.0e-5,
    volume_rel_atol: float = 1.0e-9,
    thickness_atol: float = 1.0e-7,
    tetra_volume_atol: float = 1.0e-12,
) -> None:
    """Check numerical constraints, topology, volumes, closure, and interfaces.

    Parameters
    ----------
    mesh : ConformalMesh
        Shared mesh and scientific constraint masks.
    solution : InterfaceSolution
        Elevations and thicknesses in metres.
    solid : SolidModel
        Constructed tetrahedral bodies and per-cell volume accounting.
    tau : sequence of float
        Dimensionless presence thresholds, one per body.
    residual_atol, thickness_atol : float, optional
        Constraint and zero-thickness tolerances in metres.
    volume_abs_atol, tetra_volume_atol : float, optional
        Absolute volume tolerances in cubic metres.
    volume_rel_atol : float, optional
        Relative cell-volume tolerance.

    Raises
    ------
    CompleteModelValidationError
        If observations, ordering, pinch edges, tetrahedra, cell volumes,
        closed body boundaries, or shared interfaces fail validation.

    Notes
    -----
    Tetrahedron volumes are independently recomputed to check stored values.
    Boundary face analysis is calculated once and reused for all closure checks.
    """

    errors: list[str] = []
    body_count = mesh.cell_presence_mask.shape[1]
    thresholds = np.asarray(tau, dtype=float)
    if thresholds.shape != (body_count,) or not np.isfinite(thresholds).all():
        raise ValueError(f"tau must contain {body_count} finite thresholds.")
    triangles = mesh.topology.triangles
    known = mesh.z_known
    observation_residual = (
        0.0
        if not np.any(known)
        else float(np.max(np.abs(solution.z[known] - mesh.observed_z[known])))
    )
    ordering_violation = max(0.0, -float(np.min(solution.h)))
    zero_residuals = np.zeros(body_count, dtype=float)
    for body in range(body_count):
        constrained = mesh.zero_thickness_mask[:, body]
        if np.any(constrained):
            zero_residuals[body] = float(np.max(np.abs(solution.h[constrained, body])))
    if observation_residual > residual_atol:
        errors.append(f"observation residual is {observation_residual}")
    if ordering_violation > residual_atol:
        errors.append(f"interface ordering violation is {ordering_violation}")
    if float(np.max(zero_residuals)) > residual_atol:
        errors.append(f"zero-thickness residual is {float(np.max(zero_residuals))}")

    canonical_triangles = np.sort(triangles, axis=1)
    duplicate_triangles = int(
        len(canonical_triangles) - len(np.unique(canonical_triangles, axis=0))
    )
    points = mesh.nodes_xy[triangles]
    signed_double_areas = (points[:, 1, 0] - points[:, 0, 0]) * (
        points[:, 2, 1] - points[:, 0, 1]
    ) - (points[:, 1, 1] - points[:, 0, 1]) * (points[:, 2, 0] - points[:, 0, 0])
    degenerate_triangles = int(np.sum(signed_double_areas <= mesh.area_tolerance * 2))
    nonmanifold_mesh_edges = int(
        sum(len(cells) > 2 for cells in mesh.topology.edge_to_cells)
    )
    if duplicate_triangles:
        errors.append(f"mesh has {duplicate_triangles} duplicate triangles")
    if degenerate_triangles:
        errors.append(f"mesh has {degenerate_triangles} degenerate triangles")
    if nonmanifold_mesh_edges:
        errors.append(f"mesh has {nonmanifold_mesh_edges} non-manifold edges")

    invalid_pinch_edges = 0
    unmarked_presence_boundary_edges = 0
    for edge_id, edge in enumerate(mesh.topology.edges):
        for body in range(body_count):
            marked = bool(mesh.pinch_edge_mask[edge_id, body])
            if marked and np.any(
                np.abs(mesh.unit_presence_values[edge, body] - thresholds[body])
                > mesh.presence_tolerance
            ):
                invalid_pinch_edges += 1
            cells = mesh.topology.edge_to_cells[edge_id]
            separates_presence = len(cells) == 2 and bool(
                mesh.cell_presence_mask[cells[0], body]
            ) != bool(mesh.cell_presence_mask[cells[1], body])
            if separates_presence and not marked:
                unmarked_presence_boundary_edges += 1
    if invalid_pinch_edges:
        errors.append(f"{invalid_pinch_edges} pinch edges are not threshold edges")
    if unmarked_presence_boundary_edges:
        errors.append(
            f"{unmarked_presence_boundary_edges} unit-presence-boundary edges are unmarked"
        )

    near_zero = np.abs(solution.h) <= thickness_atol
    unexpected_near_zero = int(np.sum(near_zero & ~mesh.zero_thickness_mask))
    missing_near_zero = int(np.sum(~near_zero & mesh.zero_thickness_mask))
    if unexpected_near_zero or missing_near_zero:
        errors.append(
            "solid zero classification differs from the model zero-thickness mask"
        )
    present_all_zero = int(
        sum(
            bool(mesh.cell_presence_mask[cell, body])
            and bool(np.all(near_zero[triangle, body]))
            for cell, triangle in enumerate(triangles)
            for body in range(body_count)
        )
    )
    if present_all_zero:
        errors.append(f"{present_all_zero} present cells have three zero vertices")

    if solid.tetrahedra.ndim != 2 or solid.tetrahedra.shape[1:] != (4,):
        errors.append("tetrahedra must have shape (T, 4)")
    if np.any(solid.tetrahedra < 0) or np.any(
        solid.tetrahedra >= len(solid.vertices_xyz)
    ):
        errors.append("tetrahedra reference out-of-range solid vertices")
    recomputed_volumes = np.asarray(
        [
            float(
                np.dot(
                    p[1] - p[0],
                    np.cross(p[2] - p[0], p[3] - p[0]),
                )
                / 6.0
            )
            for tetrahedron in solid.tetrahedra
            for p in (solid.vertices_xyz[tetrahedron],)
        ],
        dtype=float,
    )
    negative_tetrahedra = int(np.sum(recomputed_volumes < -tetra_volume_atol))
    zero_tetrahedra = int(np.sum(recomputed_volumes <= tetra_volume_atol))
    stored_volume_residual = float(
        np.max(np.abs(recomputed_volumes - solid.tetra_volumes))
    )
    if negative_tetrahedra:
        errors.append(f"{negative_tetrahedra} tetrahedra have negative volume")
    if zero_tetrahedra:
        errors.append(f"{zero_tetrahedra} tetrahedra have zero volume")
    if stored_volume_residual > volume_abs_atol:
        errors.append(f"stored tetra-volume residual is {stored_volume_residual}")

    accumulated_cell_volumes = np.zeros_like(solid.cell_tetra_volumes)
    for volume, body, cell in zip(
        recomputed_volumes,
        solid.tetra_body_ids,
        solid.tetra_parent_cell_ids,
    ):
        if not mesh.cell_presence_mask[int(cell), int(body)]:
            errors.append(f"tetrahedron generated in absent cell {cell}, body {body}")
        accumulated_cell_volumes[int(cell), int(body)] += volume
    cell_storage_residual = float(
        np.max(np.abs(accumulated_cell_volumes - solid.cell_tetra_volumes))
    )
    cell_volume_residuals = np.abs(
        accumulated_cell_volumes - solid.cell_theoretical_volumes
    )
    cell_volume_limits = volume_abs_atol + volume_rel_atol * np.maximum(
        np.abs(solid.cell_theoretical_volumes), 1.0
    )
    failed_volume_cells = int(
        np.sum((cell_volume_residuals > cell_volume_limits) & mesh.cell_presence_mask)
    )
    if cell_storage_residual > volume_abs_atol:
        errors.append(f"cell-volume storage residual is {cell_storage_residual}")
    if failed_volume_cells:
        errors.append(f"{failed_volume_cells} cells fail theoretical volume checks")

    boundary_analysis = analyze_solid_boundaries(solid, body_count=body_count)
    body_boundary_closed = boundary_analysis.boundary_closed
    nonmanifold_tetra_faces = boundary_analysis.nonmanifold_face_count
    inconsistent_internal_face_orientations = (
        boundary_analysis.inconsistent_internal_orientation_count
    )
    missing_or_inconsistent_side_faces = 0
    face_counters = boundary_analysis.face_counters

    for body in range(body_count):
        face_counter = face_counters[body]

        for edge_id, edge in enumerate(mesh.topology.edges):
            present_cells = sum(
                bool(mesh.cell_presence_mask[cell, body])
                for cell in mesh.topology.edge_to_cells[edge_id]
            )
            for face in _expected_side_faces(solid, edge, body):
                if face_counter.get(face, 0) != present_cells:
                    missing_or_inconsistent_side_faces += 1

    if nonmanifold_tetra_faces:
        errors.append(f"{nonmanifold_tetra_faces} tetra faces are non-manifold")
    if inconsistent_internal_face_orientations:
        errors.append(
            f"{inconsistent_internal_face_orientations} shared faces have equal orientation"
        )
    if missing_or_inconsistent_side_faces:
        errors.append(
            f"{missing_or_inconsistent_side_faces} side faces are missing or inconsistent"
        )
    present_bodies = np.any(mesh.cell_presence_mask, axis=0)
    if not np.all(body_boundary_closed[present_bodies]):
        errors.append("one or more geological-body boundaries are open")

    adjacent_interface_expected = np.zeros(body_count - 1, dtype=np.int64)
    adjacent_interface_matched = np.zeros(body_count - 1, dtype=np.int64)
    for lower_body in range(body_count - 1):
        interface = lower_body + 1
        for cell, triangle in enumerate(triangles):
            if not (
                mesh.cell_presence_mask[cell, lower_body]
                and mesh.cell_presence_mask[cell, lower_body + 1]
            ):
                continue
            adjacent_interface_expected[lower_body] += 1
            face = tuple(
                sorted(
                    map(
                        int,
                        solid.surface_vertex_ids[triangle, interface],
                    )
                )
            )
            if (
                face_counters[lower_body].get(face, 0) == 1
                and face_counters[lower_body + 1].get(face, 0) == 1
            ):
                adjacent_interface_matched[lower_body] += 1
    if not np.array_equal(adjacent_interface_expected, adjacent_interface_matched):
        errors.append("adjacent geological bodies disagree on shared interface faces")

    if errors:
        raise CompleteModelValidationError(
            "Model validation failed: " + "; ".join(errors)
        )
