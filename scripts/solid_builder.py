"""Build positive-volume tetrahedral geological bodies from the shared mesh."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .conformal_mesh import ConformalMesh
from .interface_qp import InterfaceSolution


class SolidBuildError(ValueError):
    """Raised when the solved surfaces cannot form valid solid cells."""


def _readonly(array: NDArray) -> NDArray:
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class SolidModel:
    """Tetrahedral bodies and the geometry needed to validate them."""

    vertices_xyz: NDArray[np.float64]
    surface_vertex_ids: NDArray[np.int64]
    tetrahedra: NDArray[np.int64]
    tetra_body_ids: NDArray[np.int64]
    tetra_parent_cell_ids: NDArray[np.int64]
    tetra_volumes: NDArray[np.float64]
    cell_tetra_volumes: NDArray[np.float64]
    cell_theoretical_volumes: NDArray[np.float64]


def _signed_tetra_volume(
    points: NDArray[np.float64],
) -> float:
    return float(
        np.dot(
            points[1] - points[0],
            np.cross(points[2] - points[0], points[3] - points[0]),
        )
        / 6.0
    )


def _validate_inputs(
    mesh: ConformalMesh,
    solution: InterfaceSolution,
    *,
    thickness_atol: float,
    tetra_volume_atol: float,
) -> None:
    node_count = len(mesh.nodes_xy)
    cell_count = len(mesh.topology.triangles)
    if mesh.nodes_xy.shape != (node_count, 2):
        raise SolidBuildError("mesh nodes_xy must have shape (N, 2).")
    if solution.z.ndim != 2 or solution.z.shape[0] != node_count:
        raise SolidBuildError("solution Z must have shape (N, I).")
    interface_count = solution.z.shape[1]
    body_count = interface_count - 1
    if interface_count < 2:
        raise SolidBuildError("at least two interfaces are required.")
    if solution.h.shape != (node_count, body_count):
        raise SolidBuildError(
            f"solution H must have shape ({node_count}, {body_count})."
        )
    if mesh.cell_presence_mask.shape != (cell_count, body_count):
        raise SolidBuildError(
            f"cell_presence_mask must have shape ({cell_count}, {body_count})."
        )
    for name, values in (
        ("nodes_xy", mesh.nodes_xy),
        ("solution Z", solution.z),
        ("solution H", solution.h),
    ):
        if not np.isfinite(values).all():
            raise SolidBuildError(f"{name} contains a non-finite value.")
    if not np.isfinite(thickness_atol) or thickness_atol <= 0:
        raise SolidBuildError("thickness_atol must be finite and positive.")
    if not np.isfinite(tetra_volume_atol) or tetra_volume_atol <= 0:
        raise SolidBuildError("tetra_volume_atol must be finite and positive.")
    difference_residual = float(
        np.max(np.abs(solution.h - np.diff(solution.z, axis=1)))
    )
    if difference_residual > thickness_atol:
        raise SolidBuildError(
            "solution H is inconsistent with Z; maximum residual is "
            f"{difference_residual}."
        )
    minimum_thickness = float(np.min(solution.h))
    if minimum_thickness < -thickness_atol:
        raise SolidBuildError(
            "solution contains thickness below the accepted tolerance: "
            f"{minimum_thickness}."
        )


def _build_welded_vertices(
    mesh: ConformalMesh,
    solution: InterfaceSolution,
    *,
    thickness_atol: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
]:
    """Weld coincident interface vertices without changing solved elevations."""
    node_count = len(mesh.nodes_xy)
    interface_count = solution.z.shape[1]
    surface_vertex_ids = np.empty((node_count, interface_count), dtype=np.int64)
    vertices: list[tuple[float, float, float]] = []

    for node in range(node_count):
        groups: list[list[int]] = [[0]]
        for interface in range(1, interface_count):
            if abs(solution.h[node, interface - 1]) <= thickness_atol:
                groups[-1].append(interface)
            else:
                groups.append([interface])
        for interfaces in groups:
            vertex_id = len(vertices)
            welded_z = float(np.mean(solution.z[node, interfaces]))
            vertices.append(
                (float(mesh.nodes_xy[node, 0]), float(mesh.nodes_xy[node, 1]), welded_z)
            )
            surface_vertex_ids[node, interfaces] = vertex_id

    return (
        np.asarray(vertices, dtype=float),
        surface_vertex_ids,
    )


def build_solid_model(
    mesh: ConformalMesh,
    solution: InterfaceSolution,
    *,
    thickness_atol: float = 1.0e-7,
    tetra_volume_atol: float = 1.0e-12,
) -> SolidModel:
    """Construct conforming tetrahedra between adjacent shared interfaces.

    Parameters
    ----------
    mesh : ConformalMesh
        Shared XY mesh and per-cell body presence.
    solution : InterfaceSolution
        Elevations and thicknesses in metres.
    thickness_atol : float, optional
        Metre tolerance used only for derived solid vertex welding.
    tetra_volume_atol : float, optional
        Minimum accepted tetrahedron volume in cubic metres.

    Returns
    -------
    SolidModel
        Vertices (V, 3), tetrahedra (T, 4), body/parent-cell assignments,
        and volumes used by the independent complete-model validator.

    Raises
    ------
    SolidBuildError
        If valid positive-volume tetrahedra cannot be constructed.
    """

    _validate_inputs(
        mesh,
        solution,
        thickness_atol=thickness_atol,
        tetra_volume_atol=tetra_volume_atol,
    )
    vertices, surface_vertex_ids = _build_welded_vertices(
        mesh, solution, thickness_atol=thickness_atol
    )

    cell_count = len(mesh.topology.triangles)
    body_count = solution.h.shape[1]
    cell_tetra_volumes = np.zeros((cell_count, body_count), dtype=float)
    cell_theoretical_volumes = np.zeros((cell_count, body_count), dtype=float)
    tetrahedra: list[tuple[int, int, int, int]] = []
    tetra_body_ids: list[int] = []
    tetra_parent_cell_ids: list[int] = []
    tetra_volumes: list[float] = []

    for body in range(body_count):
        for cell, triangle in enumerate(mesh.topology.triangles):
            if not mesh.cell_presence_mask[cell, body]:
                continue
            nodes = sorted(map(int, triangle))
            zero_count = int(np.sum(np.abs(solution.h[nodes, body]) <= thickness_atol))
            if zero_count == 3:
                raise SolidBuildError(
                    f"present cell {cell} in body {body} has three zero-thickness vertices."
                )

            lower = [int(surface_vertex_ids[node, body]) for node in nodes]
            upper = [int(surface_vertex_ids[node, body + 1]) for node in nodes]
            templates = (
                (lower[0], lower[1], lower[2], upper[2]),
                (lower[0], lower[1], upper[1], upper[2]),
                (lower[0], upper[0], upper[1], upper[2]),
            )
            kept_count = 0
            cell_volume = 0.0
            for template in templates:
                if len(set(template)) < 4:
                    continue
                tetrahedron = list(template)
                signed_volume = _signed_tetra_volume(vertices[tetrahedron])
                if abs(signed_volume) <= tetra_volume_atol:
                    raise SolidBuildError(
                        f"cell {cell}, body {body} produced a non-repeated "
                        f"tetrahedron below volume tolerance: {signed_volume}."
                    )
                if signed_volume < 0:
                    tetrahedron[1], tetrahedron[2] = (tetrahedron[2], tetrahedron[1])
                    signed_volume = -signed_volume
                tetrahedra.append(tuple(tetrahedron))
                tetra_body_ids.append(body)
                tetra_parent_cell_ids.append(cell)
                tetra_volumes.append(signed_volume)
                kept_count += 1
                cell_volume += signed_volume

            expected_count = 3 - zero_count
            if kept_count != expected_count:
                raise SolidBuildError(
                    f"cell {cell}, body {body} produced {kept_count} tetrahedra; "
                    f"expected {expected_count}."
                )
            xy = mesh.nodes_xy[np.asarray(triangle, dtype=np.int64)]
            area = 0.5 * abs(
                float(
                    (xy[1, 0] - xy[0, 0]) * (xy[2, 1] - xy[0, 1])
                    - (xy[1, 1] - xy[0, 1]) * (xy[2, 0] - xy[0, 0])
                )
            )
            theoretical_volume = area * float(np.sum(solution.h[triangle, body])) / 3.0
            cell_tetra_volumes[cell, body] = cell_volume
            cell_theoretical_volumes[cell, body] = theoretical_volume

    return SolidModel(
        vertices_xyz=_readonly(vertices.copy()),
        surface_vertex_ids=_readonly(surface_vertex_ids.copy()),
        tetrahedra=_readonly(np.asarray(tetrahedra, dtype=np.int64).reshape((-1, 4))),
        tetra_body_ids=_readonly(np.asarray(tetra_body_ids, dtype=np.int64)),
        tetra_parent_cell_ids=_readonly(
            np.asarray(tetra_parent_cell_ids, dtype=np.int64)
        ),
        tetra_volumes=_readonly(np.asarray(tetra_volumes, dtype=float)),
        cell_tetra_volumes=_readonly(cell_tetra_volumes),
        cell_theoretical_volumes=_readonly(cell_theoretical_volumes),
    )
