"""CVXPY joint solve for an ordered geological-interface stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from .conformal_mesh import ConformalMesh
from .mesh_operators import FinalMeshOperators


class InterfaceSolveError(ValueError):
    """Raised when the joint interface optimization problem is invalid, infeasible, or inaccurate."""


def _readonly(array: NDArray) -> NDArray:
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class InterfaceSolution:
    """Jointly optimized interface elevations on one shared mesh."""

    z: NDArray[np.float64]
    h: NDArray[np.float64]


def _validate_problem_inputs(
    mesh: ConformalMesh,
    operators: FinalMeshOperators,
    alpha: Sequence[float],
    beta: Sequence[float],
    *,
    residual_atol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    node_count = len(mesh.nodes_xy)
    if mesh.observed_z.ndim != 2 or mesh.observed_z.shape[0] != node_count:
        raise InterfaceSolveError(f"observed_z must have shape ({node_count}, I).")
    interface_count = mesh.observed_z.shape[1]
    body_count = interface_count - 1
    if interface_count < 2:
        raise InterfaceSolveError("at least two interfaces are required.")
    if mesh.z_known.shape != mesh.observed_z.shape:
        raise InterfaceSolveError("z_known shape does not match observed_z.")
    if mesh.zero_thickness_mask.shape != (node_count, body_count):
        raise InterfaceSolveError(
            f"zero_thickness_mask must have shape ({node_count}, {body_count})."
        )
    if np.any(mesh.z_known & ~np.isfinite(mesh.observed_z)):
        raise InterfaceSolveError("known observations must be finite.")
    if operators.difference.matrix.shape != (len(mesh.topology.edges), node_count):
        raise InterfaceSolveError("D_final shape does not match the final mesh.")
    expected_square = (node_count, node_count)
    if operators.curvature_operator.shape != expected_square:
        raise InterfaceSolveError(
            "curvature_operator must have shape "
            f"{expected_square}, got {operators.curvature_operator.shape}."
        )
    if not np.isfinite(residual_atol) or residual_atol <= 0:
        raise ValueError("residual_atol must be finite and positive.")

    alpha_values = np.asarray(alpha, dtype=float)
    beta_values = np.asarray(beta, dtype=float)
    if alpha_values.shape != (interface_count,) or not np.isfinite(alpha_values).all():
        raise InterfaceSolveError(
            f"alpha must contain {interface_count} finite values."
        )
    if beta_values.shape != (body_count,) or not np.isfinite(beta_values).all():
        raise InterfaceSolveError(f"beta must contain {body_count} finite values.")
    if np.any(alpha_values <= 0) or np.any(beta_values <= 0):
        raise InterfaceSolveError("all alpha and beta values must be positive.")

    known = mesh.z_known
    observations = mesh.observed_z
    for lower in range(body_count):
        for upper in range(lower + 1, interface_count):
            comparable = known[:, lower] & known[:, upper]
            invalid = np.flatnonzero(
                comparable
                & (observations[:, upper] < observations[:, lower] - residual_atol)
            )
            if len(invalid):
                raise InterfaceSolveError(
                    "known interface observations violate bottom-to-top order "
                    f"at nodes {invalid.tolist()}."
                )
    for body in range(body_count):
        both_known = known[:, body] & known[:, body + 1]
        conflicts = np.flatnonzero(
            mesh.zero_thickness_mask[:, body]
            & both_known
            & (
                np.abs(observations[:, body + 1] - observations[:, body])
                > residual_atol
            )
        )
        if len(conflicts):
            raise InterfaceSolveError(
                f"body {body} zero-thickness constraints conflict with known "
                f"observations at nodes {conflicts.tolist()}."
            )
    return alpha_values, beta_values


def solve_interface_stack(
    mesh: ConformalMesh,
    operators: FinalMeshOperators,
    *,
    alpha: Sequence[float],
    beta: Sequence[float],
    solver: str = "OSQP",
    eps_abs: float = 1.0e-9,
    eps_rel: float = 1.0e-9,
    max_iter: int = 200_000,
    polishing: bool = True,
    warm_start: bool = False,
    residual_atol: float = 1.0e-7,
) -> InterfaceSolution:
    """Jointly reconstruct ordered interfaces with hard observations.

    Parameters
    ----------
    mesh : ConformalMesh
        Shared XY mesh, observation mask, and zero-thickness constraints.
    operators : FinalMeshOperators
        Curvature and weighted thickness-difference operators on the final mesh.
    alpha, beta : sequence of float
        Positive curvature weights (I,) and thickness weights (I - 1,).
    solver : str, optional
        Required solver, ``OSQP``.
    eps_abs, eps_rel : float, optional
        Positive OSQP convergence tolerances.
    max_iter : int, optional
        Maximum solver iterations.
    polishing, warm_start : bool, optional
        OSQP polishing and warm-start controls.
    residual_atol : float, optional
        Accepted observation, ordering, and zero-thickness residual in metres.

    Returns
    -------
    InterfaceSolution
        Read-only elevations ``z`` (N, I) and thicknesses ``h`` (N, I - 1), in metres.

    Raises
    ------
    InterfaceSolveError
        If constraints conflict, optimal status is not reached, or residuals fail.
    """

    alpha_values, beta_values = _validate_problem_inputs(
        mesh, operators, alpha, beta, residual_atol=residual_atol
    )
    if solver != "OSQP":
        raise ValueError("interface solver is fixed to OSQP.")
    for name, value in (("eps_abs", eps_abs), ("eps_rel", eps_rel)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    node_count = len(mesh.nodes_xy)
    interface_count = mesh.observed_z.shape[1]
    body_count = interface_count - 1
    z_variable = cp.Variable((node_count, interface_count), name="Z")
    thickness = z_variable[:, 1:] - z_variable[:, :-1]
    constraints: list[cp.Constraint] = [thickness >= 0]
    for interface in range(interface_count):
        known_nodes = np.flatnonzero(mesh.z_known[:, interface])
        if len(known_nodes):
            constraints.append(
                z_variable[known_nodes, interface]
                == mesh.observed_z[known_nodes, interface]
            )
    for body in range(body_count):
        zero_nodes = np.flatnonzero(mesh.zero_thickness_mask[:, body])
        if len(zero_nodes):
            constraints.append(thickness[zero_nodes, body] == 0)

    curvature_terms = [
        alpha_values[interface]
        * cp.sum_squares(operators.curvature_operator @ z_variable[:, interface])
        for interface in range(interface_count)
    ]
    thickness_terms = [
        beta_values[body]
        * cp.sum_squares(operators.difference.matrix @ thickness[:, body])
        for body in range(body_count)
    ]
    problem = cp.Problem(
        cp.Minimize(sum(curvature_terms) + sum(thickness_terms)), constraints
    )
    if not problem.is_dcp():
        raise InterfaceSolveError("joint interface problem is not DCP compliant.")
    try:
        problem.solve(
            solver=cp.OSQP,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            max_iter=max_iter,
            polishing=polishing,
            warm_start=warm_start,
            verbose=False,
        )
    except cp.error.SolverError as exc:
        raise InterfaceSolveError(
            f"OSQP failed to solve interface stack: {exc}"
        ) from exc
    if problem.status != cp.OPTIMAL:
        raise InterfaceSolveError(
            f"interface solve did not reach optimal status: {problem.status}."
        )
    if z_variable.value is None:
        raise InterfaceSolveError("optimal solve returned no numerical solution.")

    z = np.asarray(z_variable.value, dtype=float)
    h = z[:, 1:] - z[:, :-1]
    if not np.isfinite(z).all() or not np.isfinite(h).all():
        raise InterfaceSolveError("interface solution contains non-finite values.")
    observation_residual = (
        0.0
        if not np.any(mesh.z_known)
        else float(np.max(np.abs(z[mesh.z_known] - mesh.observed_z[mesh.z_known])))
    )
    zero_residuals = np.zeros(body_count, dtype=float)
    for body in range(body_count):
        zero_nodes = mesh.zero_thickness_mask[:, body]
        if np.any(zero_nodes):
            zero_residuals[body] = float(np.max(np.abs(h[zero_nodes, body])))
    minimum_thickness = float(np.min(h))
    ordering_violation = max(0.0, -minimum_thickness)
    maximum_residual = max(
        observation_residual,
        float(np.max(zero_residuals)),
        ordering_violation,
    )
    if maximum_residual > residual_atol:
        raise InterfaceSolveError(
            f"interface constraints exceed residual tolerance: {maximum_residual}."
        )

    return InterfaceSolution(
        z=_readonly(z),
        h=_readonly(h),
    )
