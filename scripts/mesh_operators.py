"""Sparse mesh operators used by shared-surface modelling stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import sparse


class MeshOperatorError(ValueError):
    """Raised when a requested mesh operator cannot be constructed."""


@dataclass(frozen=True, slots=True)
class EdgeDifferenceOperator:
    """Weighted edge differences and the geometry used to build them.

    Edge weights are defined as

    ``weight[e] = median(edge_length) / edge_length[e]``.

    Each row of ``matrix`` has ``-sqrt(weight)`` at the first edge node and
    ``+sqrt(weight)`` at the second.  Multiplying the matrix by a nodal field
    therefore returns weighted differences along every valid mesh edge.
    """

    matrix: sparse.csr_matrix
    reference_length: float


@dataclass(frozen=True, slots=True)
class FinalMeshOperators:
    """Sparse numerical operators assembled on the final shared conformal mesh."""

    difference: EdgeDifferenceOperator
    curvature_operator: sparse.csr_matrix


def _readonly(array: NDArray) -> NDArray:
    array.setflags(write=False)
    return array


def _readonly_csr(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    matrix.data.setflags(write=False)
    matrix.indices.setflags(write=False)
    matrix.indptr.setflags(write=False)
    return matrix


def build_weighted_edge_difference(
    xy: Sequence[Sequence[float]] | NDArray,
    edges: Sequence[Sequence[int]] | NDArray,
) -> EdgeDifferenceOperator:
    """Build a dimensionless inverse-distance edge-difference operator.

    Parameters
    ----------
    xy : array_like, shape (N, 2)
        Finite planar coordinates in metres.
    edges : array_like, shape (E, 2)
        Unique undirected integer node pairs.

    Returns
    -------
    EdgeDifferenceOperator
        CSR matrix (E, N) with weights ``sqrt(L_ref / edge_length)`` and median
        edge length ``L_ref`` in metres.

    Notes
    -----
    All supplied edges participate. No fault or crossing-edge filter is applied.
    The median-length numerator changes only the overall objective scale
    relative to inverse-length weights.
    """

    nodes = np.asarray(xy, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise MeshOperatorError(f"xy must have shape (N, 2), got {nodes.shape}.")
    if not np.isfinite(nodes).all():
        raise MeshOperatorError("xy contains non-finite coordinates.")

    source_edges = np.asarray(edges)
    if source_edges.ndim != 2 or source_edges.shape[1] != 2:
        raise MeshOperatorError(
            f"edges must have shape (E, 2), got {source_edges.shape}."
        )
    if len(source_edges) == 0:
        raise MeshOperatorError("edges must contain at least one mesh edge.")
    if not np.issubdtype(source_edges.dtype, np.integer):
        raise MeshOperatorError("edge node indices must use an integer dtype.")

    edge_nodes = np.asarray(source_edges, dtype=np.int64)
    if np.any(edge_nodes[:, 0] == edge_nodes[:, 1]):
        bad = np.flatnonzero(edge_nodes[:, 0] == edge_nodes[:, 1]).tolist()
        raise MeshOperatorError(f"edges repeat a node at edge IDs {bad}.")
    if np.any(edge_nodes < 0) or np.any(edge_nodes >= len(nodes)):
        bad = np.flatnonzero(
            np.any((edge_nodes < 0) | (edge_nodes >= len(nodes)), axis=1)
        ).tolist()
        raise MeshOperatorError(f"edges reference out-of-range nodes at IDs {bad}.")

    canonical = np.sort(edge_nodes, axis=1)
    if len(np.unique(canonical, axis=0)) != len(canonical):
        raise MeshOperatorError("edges contain duplicate undirected node pairs.")

    vectors = nodes[edge_nodes[:, 1]] - nodes[edge_nodes[:, 0]]
    lengths = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0):
        bad = np.flatnonzero(~np.isfinite(lengths) | (lengths <= 0)).tolist()
        raise MeshOperatorError(f"edges have invalid geometric lengths at IDs {bad}.")

    reference_length = float(np.median(lengths))
    weights = reference_length / lengths
    roots = np.sqrt(weights)
    edge_count = len(edge_nodes)
    row_ids = np.repeat(np.arange(edge_count, dtype=np.int64), 2)
    column_ids = edge_nodes.reshape(-1)
    values = np.column_stack((-roots, roots)).reshape(-1)
    matrix = sparse.csr_matrix(
        (values, (row_ids, column_ids)),
        shape=(edge_count, len(nodes)),
        dtype=np.float64,
    )
    matrix.sort_indices()

    return EdgeDifferenceOperator(
        matrix=matrix,
        reference_length=reference_length,
    )


def build_final_mesh_operators(
    xy: Sequence[Sequence[float]] | NDArray,
    triangles: Sequence[Sequence[int]] | NDArray,
    edges: Sequence[Sequence[int]] | NDArray,
    *,
    area_atol: float = 0.0,
    matrix_atol: float = 1.0e-10,
) -> FinalMeshOperators:
    """Build edge, cotangent, lumped-mass, and curvature operators.

    Parameters
    ----------
    xy : array_like, shape (N, 2)
        Final planar coordinates in metres.
    triangles : array_like, shape (M, 3)
        Counter-clockwise triangle node indices.
    edges : array_like, shape (E, 2)
        Unique edge node pairs.
    area_atol : float, optional
        Minimum triangle area in square metres.
    matrix_atol : float, optional
        Accepted matrix consistency residual.

    Returns
    -------
    FinalMeshOperators
        Weighted differences, cotangent Laplacian, lumped mass, and curvature.

    Notes
    -----
    Cotangent weights retain the unmodified linear-triangle FEM values,
    including negative contributions from obtuse angles. No edge filter is used.
    """

    nodes = np.asarray(xy, dtype=float)
    cells_source = np.asarray(triangles)
    if nodes.ndim != 2 or nodes.shape[1] != 2 or len(nodes) < 3:
        raise MeshOperatorError(f"xy must have shape (N, 2), got {nodes.shape}.")
    if not np.isfinite(nodes).all():
        raise MeshOperatorError("xy contains non-finite coordinates.")
    if cells_source.ndim != 2 or cells_source.shape[1] != 3:
        raise MeshOperatorError(
            f"triangles must have shape (M, 3), got {cells_source.shape}."
        )
    if len(cells_source) == 0:
        raise MeshOperatorError("triangles must contain at least one cell.")
    if not np.issubdtype(cells_source.dtype, np.integer):
        raise MeshOperatorError("triangle node indices must use an integer dtype.")
    cells = np.asarray(cells_source, dtype=np.int64)
    if np.any(cells < 0) or np.any(cells >= len(nodes)):
        raise MeshOperatorError("triangles reference out-of-range nodes.")
    if any(len(set(map(int, cell))) != 3 for cell in cells):
        raise MeshOperatorError("triangles contain repeated node indices.")
    if not np.isfinite(area_atol) or area_atol < 0:
        raise ValueError("area_atol must be finite and non-negative.")
    if not np.isfinite(matrix_atol) or matrix_atol < 0:
        raise ValueError("matrix_atol must be finite and non-negative.")

    difference = build_weighted_edge_difference(nodes, edges)
    row_ids: list[int] = []
    column_ids: list[int] = []
    values: list[float] = []
    mass = np.zeros(len(nodes), dtype=float)

    for cell_id, cell in enumerate(cells):
        point_ids = list(map(int, cell))
        points = nodes[point_ids]
        edge_01 = points[1] - points[0]
        edge_02 = points[2] - points[0]
        signed_double_area = float(edge_01[0] * edge_02[1] - edge_01[1] * edge_02[0])
        if signed_double_area <= 2.0 * area_atol:
            reason = "clockwise" if signed_double_area < 0 else "degenerate"
            raise MeshOperatorError(
                f"triangle {cell_id} is {reason} or below area tolerance."
            )
        area = 0.5 * signed_double_area
        mass[point_ids] += area / 3.0

        cotangents = np.empty(3, dtype=float)
        for local_node in range(3):
            vector_a = points[(local_node + 1) % 3] - points[local_node]
            vector_b = points[(local_node + 2) % 3] - points[local_node]
            cotangents[local_node] = float(
                np.dot(vector_a, vector_b) / signed_double_area
            )
        local = 0.5 * np.asarray(
            (
                (
                    cotangents[1] + cotangents[2],
                    -cotangents[2],
                    -cotangents[1],
                ),
                (
                    -cotangents[2],
                    cotangents[0] + cotangents[2],
                    -cotangents[0],
                ),
                (
                    -cotangents[1],
                    -cotangents[0],
                    cotangents[0] + cotangents[1],
                ),
            ),
            dtype=float,
        )
        for local_row, node_row in enumerate(point_ids):
            for local_column, node_column in enumerate(point_ids):
                row_ids.append(node_row)
                column_ids.append(node_column)
                values.append(float(local[local_row, local_column]))

    if not np.isfinite(mass).all() or np.any(mass <= 0):
        invalid = np.flatnonzero(~np.isfinite(mass) | (mass <= 0)).tolist()
        raise MeshOperatorError(
            f"lumped mass must be finite and positive at every node: {invalid}."
        )
    laplacian = sparse.coo_matrix(
        (values, (row_ids, column_ids)),
        shape=(len(nodes), len(nodes)),
        dtype=np.float64,
    ).tocsr()
    laplacian.sum_duplicates()
    laplacian.sort_indices()
    if not np.isfinite(laplacian.data).all():
        raise MeshOperatorError("cotangent Laplacian contains non-finite values.")
    symmetry = laplacian - laplacian.T
    symmetry_residual = (
        0.0 if symmetry.nnz == 0 else float(np.max(np.abs(symmetry.data)))
    )
    row_sum_residual = float(
        np.max(np.abs(np.asarray(laplacian.sum(axis=1)).reshape(-1)))
    )
    if symmetry_residual > matrix_atol:
        raise MeshOperatorError(
            f"cotangent Laplacian symmetry residual is {symmetry_residual}."
        )
    if row_sum_residual > matrix_atol:
        raise MeshOperatorError(
            f"cotangent Laplacian row-sum residual is {row_sum_residual}."
        )

    curvature = sparse.diags(1.0 / np.sqrt(mass), format="csr") @ laplacian
    curvature = curvature.tocsr()
    curvature.sort_indices()
    if not np.isfinite(curvature.data).all():
        raise MeshOperatorError("mass-normalized curvature operator is non-finite.")

    return FinalMeshOperators(
        difference=difference,
        curvature_operator=_readonly_csr(curvature),
    )
