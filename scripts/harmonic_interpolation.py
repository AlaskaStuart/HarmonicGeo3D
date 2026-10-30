"""Constrained graph harmonic interpolation on the original shared triangle mesh."""

from __future__ import annotations

from typing import Sequence

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from .mesh_operators import EdgeDifferenceOperator


class HarmonicInterpolationError(RuntimeError):
    """Raised when unit-presence controls are invalid or a solve is unacceptable."""


def _readonly(array: NDArray) -> NDArray:
    array.setflags(write=False)
    return array


def _validated_masks(
    zero_controls: NDArray,
    present_controls: NDArray,
    unknown_controls: NDArray,
    *,
    node_count: int,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    masks = tuple(
        np.asarray(mask, dtype=bool)
        for mask in (zero_controls, present_controls, unknown_controls)
    )
    shapes = {mask.shape for mask in masks}
    if len(shapes) != 1:
        raise HarmonicInterpolationError(
            "zero, present, and unknown control masks must have identical shapes."
        )
    shape = masks[0].shape
    if len(shape) != 2 or shape[0] != node_count:
        raise HarmonicInterpolationError(
            f"control masks must have shape ({node_count}, B), got {shape}."
        )
    if shape[1] == 0:
        raise HarmonicInterpolationError(
            "control masks must contain at least one body."
        )

    membership_count = sum(mask.astype(np.int8) for mask in masks)
    if not np.all(membership_count == 1):
        bad = np.argwhere(membership_count != 1).tolist()
        raise HarmonicInterpolationError(
            "zero, present, and unknown controls must be mutually exclusive "
            f"and exhaustive; invalid entries: {bad}."
        )
    return masks


def solve_harmonic_fields(
    operator: EdgeDifferenceOperator,
    zero_controls: Sequence[Sequence[bool]] | NDArray,
    present_controls: Sequence[Sequence[bool]] | NDArray,
    unknown_controls: Sequence[Sequence[bool]] | NDArray,
    *,
    solver: str = "OSQP",
    eps_abs: float = 1.0e-9,
    eps_rel: float = 1.0e-9,
    max_iter: int = 100_000,
    residual_atol: float = 1.0e-7,
) -> NDArray[np.float64]:
    """Infer bounded harmonic presence fields for every geological body.

    Parameters
    ----------
    operator : EdgeDifferenceOperator
        Weighted edge-difference matrix of shape (E, N).
    zero_controls, present_controls, unknown_controls : array_like, shape (N, B)
        Mutually exclusive and exhaustive body-control masks. Known equal
        interface elevations imply zero thickness; missing elevations are unknown.
    solver : str, optional
        Required solver, ``OSQP``.
    eps_abs, eps_rel : float, optional
        Positive OSQP tolerances.
    max_iter : int, optional
        Maximum iterations per body.
    residual_atol : float, optional
        Maximum accepted bound and control residual (dimensionless).

    Returns
    -------
    ndarray, shape (N, B)
        Read-only, dimensionless presence fields. These are not probabilities.

    Raises
    ------
    HarmonicInterpolationError
        If controls or solver results are unacceptable.

    Notes
    -----
    Bodies without zero-thickness controls are assigned one everywhere without
    optimization. A body with zero controls also needs positive-thickness controls.
    """

    if solver != "OSQP":
        raise ValueError("Harmonic fields must use the OSQP solver.")
    if not np.isfinite([eps_abs, eps_rel, residual_atol]).all():
        raise ValueError("solver and residual tolerances must be finite.")
    if eps_abs <= 0 or eps_rel <= 0 or residual_atol < 0 or max_iter <= 0:
        raise ValueError("solver tolerances/max_iter must be positive and valid.")

    difference = operator.matrix
    node_count = difference.shape[1]
    zero, present, _ = _validated_masks(
        zero_controls,
        present_controls,
        unknown_controls,
        node_count=node_count,
    )
    body_count = zero.shape[1]

    values = np.empty((node_count, body_count), dtype=float)
    for body in range(body_count):
        zero_ids = np.flatnonzero(zero[:, body])
        present_ids = np.flatnonzero(present[:, body])
        if len(zero_ids) == 0:
            field = np.ones(node_count, dtype=float)
        else:
            if len(present_ids) == 0:
                raise HarmonicInterpolationError(
                    f"body {body} has zero-thickness controls but no positive-"
                    "thickness controls."
                )
            presence_values = cp.Variable(node_count, name=f"presence_body_{body:02d}")
            problem = cp.Problem(
                cp.Minimize(cp.sum_squares(difference @ presence_values)),
                [
                    presence_values >= 0,
                    presence_values <= 1,
                    presence_values[zero_ids] == 0,
                    presence_values[present_ids] == 1,
                ],
            )
            try:
                problem.solve(
                    solver=cp.OSQP,
                    eps_abs=eps_abs,
                    eps_rel=eps_rel,
                    max_iter=max_iter,
                    polishing=True,
                    warm_start=False,
                    verbose=False,
                )
            except cp.error.SolverError as exc:
                raise HarmonicInterpolationError(
                    f"OSQP failed while solving harmonic field for body {body}."
                ) from exc
            status = str(problem.status)
            if status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                raise HarmonicInterpolationError(
                    f"body {body} harmonic solve returned status {status!r}."
                )
            if presence_values.value is None:
                raise HarmonicInterpolationError(
                    f"body {body} harmonic solve returned no nodal values."
                )
            field = np.asarray(presence_values.value, dtype=float).reshape(node_count)
        if not np.isfinite(field).all():
            raise HarmonicInterpolationError(
                f"body {body} harmonic solve produced non-finite values."
            )
        zero_residual = float(np.max(np.abs(field[zero_ids]))) if len(zero_ids) else 0.0
        present_residual = (
            float(np.max(np.abs(field[present_ids] - 1.0))) if len(present_ids) else 0.0
        )
        control_residual = max(zero_residual, present_residual)
        lower_violation = max(0.0, float(-np.min(field)))
        upper_violation = max(0.0, float(np.max(field) - 1.0))
        if max(control_residual, lower_violation, upper_violation) > residual_atol:
            raise HarmonicInterpolationError(
                f"body {body} violates unit-presence constraints: control="
                f"{control_residual:.3e}, lower={lower_violation:.3e}, "
                f"upper={upper_violation:.3e}."
            )

        values[:, body] = field

    return _readonly(values)
