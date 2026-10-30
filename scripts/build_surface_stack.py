#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the observation arrays used by joint stratigraphic-surface modelling.

This module does not interpolate missing elevations.  It performs the data
assembly step that must happen before the constrained solver:

* use the XY and interface observations stored together in ``layers.csv``;
* optionally align those rows with an explicitly supplied coordinate file;
* derive the ordered interface stack from the CSV header;
* build ``Z`` with shape ``(n_points, n_interfaces)`` and ``NaN`` at missing
  values;
* build observation, missing, and zero-thickness control classifications;
* reject known elevations that violate the bottom-to-top stratigraphic order.

The first three columns must be ``drill, coord_x, coord_y``.  Every following
column is an interface ordered from bottom to top, and the final interface must
be ``Ground``.  Each non-ground interface is named ``layer-<number>-bot`` and
defines the stable body name above it, for example ``layer-01-bot`` defines
``body-01``.

Blank values mean unknown elevations.  A zero-thickness control is recorded
only when both bounding elevations are observed and equal within
``pinchout_atol``.  When one of the two elevations is missing, the body status
remains unknown rather than being treated as present or zero thickness.  The
resulting object keeps only the control masks consumed by the harmonic interpolation
solver; the inferred front geometry is built later from those fields.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

METADATA_COLUMNS = ("drill", "coord_x", "coord_y")
GROUND_INTERFACE = "Ground"
LAYER_INTERFACE_PATTERN = re.compile(r"layer-(\d+)-bot")

MISSING_TOKENS = frozenset({"", "na", "n/a", "nan", "null", "none"})


class SurfaceStackDataError(ValueError):
    """Raised when source data cannot form a valid surface-stack input."""


class LayerOrderError(SurfaceStackDataError):
    """Raised when two known interfaces are in an invalid vertical order."""


@dataclass(frozen=True, slots=True)
class SurfaceStackSchema:
    """Ordered interface and stable body identities derived from a CSV header."""

    interface_names: tuple[str, ...]
    body_names: tuple[str, ...]
    body_boundaries: tuple[tuple[str, str], ...]

    @property
    def interface_count(self) -> int:
        return len(self.interface_names)


def parse_surface_stack_schema(fieldnames: Sequence[str]) -> SurfaceStackSchema:
    """Validate the authoritative header and derive its ordered body schema."""

    names = tuple(str(name).strip() for name in fieldnames)
    if names[: len(METADATA_COLUMNS)] != METADATA_COLUMNS:
        raise SurfaceStackDataError(
            "Layer CSV must begin with columns "
            f"{list(METADATA_COLUMNS)}, found {list(names[:3])}."
        )
    interface_names = names[len(METADATA_COLUMNS) :]
    if len(interface_names) < 2:
        raise SurfaceStackDataError(
            "Layer CSV must contain at least one layer interface followed by Ground."
        )
    if interface_names[-1] != GROUND_INTERFACE:
        raise SurfaceStackDataError(
            "Ground must be the final interface column in the layer CSV."
        )
    if len(set(names)) != len(names):
        raise SurfaceStackDataError("Layer CSV header contains duplicate column names.")

    body_names: list[str] = []
    numeric_ids: dict[int, str] = {}
    for interface_name in interface_names[:-1]:
        match = LAYER_INTERFACE_PATTERN.fullmatch(interface_name)
        if match is None:
            raise SurfaceStackDataError(
                "Non-ground interface columns must match "
                f"'layer-<non-negative integer>-bot', got {interface_name!r}."
            )
        number_text = match.group(1)
        number = int(number_text)
        previous = numeric_ids.get(number)
        if previous is not None:
            raise SurfaceStackDataError(
                "Layer interface numeric IDs must be unique; "
                f"{previous!r} and {interface_name!r} both identify {number}."
            )
        numeric_ids[number] = interface_name
        body_names.append(f"body-{number_text}")

    boundaries = tuple(zip(interface_names[:-1], interface_names[1:]))
    body_tuple = tuple(body_names)
    return SurfaceStackSchema(
        interface_names=interface_names,
        body_names=body_tuple,
        body_boundaries=boundaries,
    )


@dataclass(frozen=True, slots=True)
class SurfaceStackObservations:
    """Observed and missing values on one common XY mesh.

    Array shapes are:

    * ``nodes_xy``: ``(N, 2)``
    * ``z``, ``observed_mask``: ``(N, I)``
    * control masks: ``(N, I - 1)``

    ``source_row_indices[i]`` is the zero-based data-row index (header not
    counted) in the layer CSV that was matched to mesh point ``i``.
    """

    interface_names: tuple[str, ...]
    body_names: tuple[str, ...]
    body_boundaries: tuple[tuple[str, str], ...]
    drill_names: NDArray[np.str_]
    nodes_xy: NDArray[np.float64]
    source_row_indices: NDArray[np.int64]
    z: NDArray[np.float64]
    observed_mask: NDArray[np.bool_]
    zero_thickness_control_mask: NDArray[np.bool_]
    present_control_mask: NDArray[np.bool_]
    unknown_control_mask: NDArray[np.bool_]

    @property
    def interface_count(self) -> int:
        return len(self.interface_names)

    @property
    def body_count(self) -> int:
        return len(self.body_names)


def _parse_number(
    raw_value: str | None,
    *,
    field_label: str,
    allow_missing: bool,
) -> float:
    text = "" if raw_value is None else raw_value.strip()
    if text.casefold() in MISSING_TOKENS:
        if allow_missing:
            return float("nan")
        raise SurfaceStackDataError(f"{field_label} is required but missing.")

    try:
        value = float(text)
    except ValueError as exc:
        raise SurfaceStackDataError(
            f"{field_label} must be numeric or blank, got {text!r}."
        ) from exc

    if not np.isfinite(value):
        raise SurfaceStackDataError(
            f"{field_label} must be finite or blank, got {text!r}."
        )
    return value


def _read_layer_csv(
    layer_csv: Path,
) -> tuple[
    SurfaceStackSchema,
    NDArray[np.str_],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    if not layer_csv.is_file():
        raise FileNotFoundError(f"Layer CSV does not exist: {layer_csv}")

    with layer_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SurfaceStackDataError(f"Layer CSV has no header: {layer_csv}")
        schema = parse_surface_stack_schema(reader.fieldnames)
        rows = list(reader)

    if not rows:
        raise SurfaceStackDataError(f"Layer CSV contains no data rows: {layer_csv}")

    drill_names: list[str] = []
    layer_xy = np.empty((len(rows), 2), dtype=float)
    source_z = np.empty((len(rows), schema.interface_count), dtype=float)

    for row_index, row in enumerate(rows):
        csv_row = row_index + 2
        if None in row or any(value is None for value in row.values()):
            raise SurfaceStackDataError(
                f"Layer CSV row {csv_row} has a different field count from the header."
            )
        drill_name = (row.get("drill") or "").strip()
        if not drill_name:
            raise SurfaceStackDataError(
                f"Layer CSV row {csv_row} has an empty drill identifier."
            )
        drill_names.append(drill_name)

        layer_xy[row_index, 0] = _parse_number(
            row.get("coord_x"),
            field_label=f"row {csv_row} coord_x",
            allow_missing=False,
        )
        layer_xy[row_index, 1] = _parse_number(
            row.get("coord_y"),
            field_label=f"row {csv_row} coord_y",
            allow_missing=False,
        )

        for interface_index, interface_name in enumerate(schema.interface_names):
            source_z[row_index, interface_index] = _parse_number(
                row.get(interface_name),
                field_label=f"row {csv_row} {interface_name}",
                allow_missing=True,
            )

    duplicate_drills = sorted(
        name for name in set(drill_names) if drill_names.count(name) > 1
    )
    if duplicate_drills:
        raise SurfaceStackDataError(
            f"Layer CSV contains duplicate drill identifiers: {duplicate_drills}"
        )

    if len(np.unique(layer_xy, axis=0)) != len(layer_xy):
        raise SurfaceStackDataError(
            "Layer CSV contains duplicate (coord_x, coord_y) coordinates; "
            "point-to-row matching would be ambiguous."
        )

    return schema, np.asarray(drill_names, dtype=str), layer_xy, source_z


def _read_mesh_coordinates(coords_csv: Path) -> NDArray[np.float64]:
    if not coords_csv.is_file():
        raise FileNotFoundError(f"Coordinate CSV does not exist: {coords_csv}")

    parsed_rows: list[list[float]] = []
    with coords_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for file_row, row in enumerate(csv.reader(handle), start=1):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != 2:
                raise SurfaceStackDataError(
                    f"Coordinate CSV row {file_row} must contain exactly two "
                    f"columns, got {len(row)}."
                )
            parsed_rows.append(
                [
                    _parse_number(
                        row[0],
                        field_label=f"coordinate row {file_row} x",
                        allow_missing=False,
                    ),
                    _parse_number(
                        row[1],
                        field_label=f"coordinate row {file_row} y",
                        allow_missing=False,
                    ),
                ]
            )

    if not parsed_rows:
        raise SurfaceStackDataError(
            f"Coordinate CSV contains no coordinate rows: {coords_csv}"
        )

    nodes_xy = np.asarray(parsed_rows, dtype=float)
    if len(np.unique(nodes_xy, axis=0)) != len(nodes_xy):
        raise SurfaceStackDataError(
            "Coordinate CSV contains duplicate XY points; triangle point IDs "
            "would be ambiguous."
        )
    return nodes_xy


def _match_layer_rows_to_mesh(
    nodes_xy: NDArray[np.float64],
    layer_xy: NDArray[np.float64],
    *,
    coordinate_atol: float,
) -> NDArray[np.int64]:
    if coordinate_atol < 0:
        raise ValueError("coordinate_atol must be non-negative.")
    if len(nodes_xy) != len(layer_xy):
        raise SurfaceStackDataError(
            "Coordinate and layer CSV row counts differ: "
            f"{len(nodes_xy)} mesh points versus {len(layer_xy)} layer rows."
        )

    source_row_indices: list[int] = []
    used_rows: set[int] = set()
    for point_index, point_xy in enumerate(nodes_xy):
        matches = np.flatnonzero(
            np.all(
                np.isclose(layer_xy, point_xy, rtol=0.0, atol=coordinate_atol),
                axis=1,
            )
        )
        if len(matches) == 0:
            raise SurfaceStackDataError(
                f"Mesh point P{point_index} at {point_xy.tolist()} has no matching "
                "row in the layer CSV."
            )
        if len(matches) > 1:
            raise SurfaceStackDataError(
                f"Mesh point P{point_index} at {point_xy.tolist()} matches more "
                "than one layer row within coordinate_atol."
            )

        source_row = int(matches[0])
        if source_row in used_rows:
            raise SurfaceStackDataError(
                f"Layer row {source_row + 2} was matched more than once."
            )
        used_rows.add(source_row)
        source_row_indices.append(source_row)

    return np.asarray(source_row_indices, dtype=np.int64)


def _find_order_violations(
    drill_names: NDArray[np.str_],
    z: NDArray[np.float64],
    observed_mask: NDArray[np.bool_],
    interface_names: Sequence[str],
    *,
    order_atol: float,
) -> list[str]:
    violations: list[str] = []
    for point_index, drill_name in enumerate(drill_names):
        for lower_index in range(len(interface_names) - 1):
            for upper_index in range(lower_index + 1, len(interface_names)):
                if not (
                    observed_mask[point_index, lower_index]
                    and observed_mask[point_index, upper_index]
                ):
                    continue
                lower_value = z[point_index, lower_index]
                upper_value = z[point_index, upper_index]
                if lower_value > upper_value + order_atol:
                    violations.append(
                        f"P{point_index} {drill_name}: "
                        f"{interface_names[lower_index]}={lower_value:g} > "
                        f"{interface_names[upper_index]}={upper_value:g}"
                    )
    return violations


def load_surface_stack_observations(
    layer_csv: str | Path,
    coords_csv: str | Path | None = None,
    *,
    coordinate_atol: float = 1e-6,
    order_atol: float = 1e-9,
    pinchout_atol: float = 1e-9,
) -> SurfaceStackObservations:
    """Load and validate the arrays required by the joint surface solver.

    Parameters
    ----------
    layer_csv:
        CSV beginning with ``drill``, ``coord_x``, ``coord_y`` followed by two
        or more bottom-to-top interface columns ending in ``Ground``.
    coords_csv:
        Headerless two-column XY file used to generate ``triangle_net``.  When
        supplied, layer rows are matched and reordered to this point order by
        coordinates.  When omitted, the layer CSV row order is used directly.
    coordinate_atol:
        Absolute tolerance for matching XY coordinates.  Relative tolerance is
        deliberately disabled.
    order_atol:
        Numerical tolerance used only when validating known vertical order.
        Source elevations are never altered.
    pinchout_atol:
        Absolute elevation-difference tolerance used to identify known
        pinchouts.  Keep this small unless a geological equality tolerance has
        been explicitly defined.
    """

    if order_atol < 0:
        raise ValueError("order_atol must be non-negative.")
    if pinchout_atol < 0:
        raise ValueError("pinchout_atol must be non-negative.")

    layer_csv = Path(layer_csv)
    schema, source_drills, layer_xy, source_z = _read_layer_csv(layer_csv)

    if coords_csv is None:
        nodes_xy = layer_xy.copy()
        source_row_indices = np.arange(len(layer_xy), dtype=np.int64)
    else:
        nodes_xy = _read_mesh_coordinates(Path(coords_csv))
        source_row_indices = _match_layer_rows_to_mesh(
            nodes_xy,
            layer_xy,
            coordinate_atol=coordinate_atol,
        )

    drill_names = source_drills[source_row_indices]
    z = source_z[source_row_indices].copy()
    observed_mask = np.isfinite(z)
    violations = _find_order_violations(
        drill_names,
        z,
        observed_mask,
        schema.interface_names,
        order_atol=order_atol,
    )
    if violations:
        preview = "\n  - ".join(violations[:20])
        remainder = (
            "" if len(violations) <= 20 else f"\n  ... {len(violations) - 20} more"
        )
        raise LayerOrderError(
            "Known interface elevations violate the required bottom-to-top "
            f"order:\n  - {preview}{remainder}"
        )

    known_control_mask = observed_mask[:, :-1] & observed_mask[:, 1:]
    adjacent_differences = z[:, 1:] - z[:, :-1]
    zero_thickness_control_mask = known_control_mask & (
        np.abs(adjacent_differences) <= pinchout_atol
    )
    present_control_mask = known_control_mask & ~zero_thickness_control_mask
    unknown_control_mask = ~known_control_mask

    return SurfaceStackObservations(
        interface_names=schema.interface_names,
        body_names=schema.body_names,
        body_boundaries=schema.body_boundaries,
        drill_names=drill_names,
        nodes_xy=nodes_xy,
        source_row_indices=source_row_indices,
        z=z,
        observed_mask=observed_mask,
        zero_thickness_control_mask=zero_thickness_control_mask,
        present_control_mask=present_control_mask,
        unknown_control_mask=unknown_control_mask,
    )
