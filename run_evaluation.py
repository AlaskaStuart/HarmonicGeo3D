"""Evaluate one full model and leave-one-borehole-out reconstructions."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.tri import Triangulation
from scipy.spatial import cKDTree

from run_model import check_output_directory, run_model
from scripts.build_surface_stack import load_surface_stack_observations

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "examples" / "input" / "completed_drill_surfaces.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "evaluation"
QUALITY_THRESHOLD = 0.20
VOLUME_ATOL = 1.0e-12
GEOMETRY_ATOL = 1.0e-10
INTERFACE_SOLVER_EPS = 1.0e-10


@dataclass(frozen=True, slots=True)
class EvaluationInput:
    """Complete reference observations and source text for masking.

    Coordinates have shape (N, 2) in metres; reference elevations have
    shape (N, I), in metres, ordered from bottom to top.
    """

    path: Path
    rows: list[dict[str, str]]
    fields: tuple[str, ...]
    drills: tuple[str, ...]
    interfaces: tuple[str, ...]
    coords: np.ndarray
    reference_z: np.ndarray


def read_input(path: Path) -> EvaluationInput:
    """Read a complete reference table using the common input validator."""
    observations = load_surface_stack_observations(path)
    if not observations.observed_mask.all():
        raise ValueError("Evaluation requires a complete reference elevation table.")
    drills = tuple(str(drill) for drill in observations.drill_names)
    if any(
        drill in {"full", ".", ".."} or any(char in drill for char in "/\\:")
        for drill in drills
    ):
        raise ValueError(
            "Evaluation drill identifiers must be safe, unique folder names."
        )
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = tuple(reader.fieldnames)
    return EvaluationInput(
        path=path,
        rows=rows,
        fields=fields,
        drills=drills,
        interfaces=observations.interface_names,
        coords=observations.nodes_xy,
        reference_z=observations.z,
    )


def write_masked_inputs(info: EvaluationInput, output: Path) -> dict[str, Path]:
    """Write fresh inputs retaining XY and masking all elevations of one drill."""
    directory = output / "masked_inputs"
    directory.mkdir(parents=True)
    paths = {}
    for drill in info.drills:
        path = directory / f"{drill}.csv"
        with path.open("x", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=info.fields)
            writer.writeheader()
            for source_row in info.rows:
                row = dict(source_row)
                if row["drill"].strip() == drill:
                    for name in info.interfaces:
                        row[name] = ""
                writer.writerow(row)
        paths[drill] = path
    return paths


def build_evaluation_models(
    info: EvaluationInput, output: Path
) -> tuple[dict[str, Path], dict[str, float]]:
    """Reconstruct every scenario once without consulting existing results."""
    masked_inputs = write_masked_inputs(info, output)
    inputs = {"full": info.path, **masked_inputs}
    paths = {}
    times = {}
    for fold, source in inputs.items():
        directory = output / "full" if fold == "full" else output / "masked" / fold
        print(f"Building {fold}", flush=True)
        start = time.perf_counter()
        paths[fold], _, _ = run_model(
            layer_csv=source,
            output_dir=directory,
            interface_solver_eps=INTERFACE_SOLVER_EPS,
            save_surface_diagnostics=True,
        )
        times[fold] = time.perf_counter() - start
    return paths, times


def load_model(path: Path) -> dict[str, np.ndarray]:
    """Read an NPZ model without enabling pickle deserialization."""
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def triangle_quality(
    nodes_xy: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate triangle quality and signed double area.

    Parameters
    ----------
    nodes_xy : ndarray, shape (N, 2)
        Coordinates in metres.
    triangles : ndarray, shape (M, 3)
        Triangle node indices.

    Returns
    -------
    quality : ndarray, shape (M,)
        Dimensionless ``4 * sqrt(3) * area / sum(edge_length**2)``.
    signed_double_area : ndarray, shape (M,)
        Twice the oriented area in square metres.
    """
    points = np.asarray(nodes_xy)[np.asarray(triangles, dtype=int)]
    vectors = points[:, 1:] - points[:, :1]
    signed_double_area = (
        vectors[:, 0, 0] * vectors[:, 1, 1] - vectors[:, 0, 1] * vectors[:, 1, 0]
    )
    edge_lengths_sq = np.sum(np.diff(points[:, [0, 1, 2, 0]], axis=1) ** 2, axis=2)
    area = np.abs(signed_double_area) / 2.0
    quality = 4.0 * np.sqrt(3.0) * area / np.sum(edge_lengths_sq, axis=1)
    return quality, signed_double_area


def tetra_signed_volumes(model: dict[str, np.ndarray]) -> np.ndarray:
    """Calculate signed tetrahedron volumes in cubic metres from stored geometry."""
    vertices = model["vertices_xyz"][model["tetrahedra"]]
    return (
        np.einsum(
            "ij,ij->i",
            np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]),
            vertices[:, 3] - vertices[:, 0],
        )
        / 6.0
    )


def duplicate_count(cells: np.ndarray) -> int:
    """Count repeated cells after ignoring vertex order."""
    canonical = np.sort(np.asarray(cells, dtype=int), axis=1)
    return int(len(canonical) - len(np.unique(canonical, axis=0)))


def nonmanifold_edges(triangles: np.ndarray) -> int:
    """Count undirected triangle edges incident to more than two cells."""
    triangles = np.asarray(triangles, dtype=int)
    edges = np.sort(
        np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])),
        axis=1,
    )
    return int(np.sum(np.unique(edges, axis=0, return_counts=True)[1] > 2))


def nonmanifold_faces(tetrahedra: np.ndarray) -> int:
    """Count tetrahedral faces incident to more than two cells."""
    tetrahedra = np.asarray(tetrahedra, dtype=int)
    faces = np.sort(
        np.vstack(
            (
                tetrahedra[:, [1, 2, 3]],
                tetrahedra[:, [0, 3, 2]],
                tetrahedra[:, [0, 1, 3]],
                tetrahedra[:, [0, 2, 1]],
            )
        ),
        axis=1,
    )
    return int(np.sum(np.unique(faces, axis=0, return_counts=True)[1] > 2))


def mesh_metrics(model: dict[str, np.ndarray]) -> tuple[dict[str, Any], np.ndarray]:
    """Measure mesh quality and topology, preserving surface-only status explicitly."""
    quality, signed_area = triangle_quality(
        model["final_nodes_xy"], model["final_triangles"]
    )
    has_solid = "tetrahedra" in model
    volumes = tetra_signed_volumes(model) if has_solid else np.empty(0)
    negative_area = int(np.sum(signed_area < -GEOMETRY_ATOL))
    degenerate = int(np.sum(np.abs(signed_area) <= GEOMETRY_ATOL))
    zero_volume = int(np.sum(np.abs(volumes) <= VOLUME_ATOL))
    negative_volume = int(np.sum(volumes < -VOLUME_ATOL))
    duplicate_triangles = duplicate_count(model["final_triangles"])
    nonmanifold_edge = nonmanifold_edges(model["final_triangles"])
    nonmanifold_face = nonmanifold_faces(model["tetrahedra"]) if has_solid else np.nan
    invalid = (
        negative_area
        + degenerate
        + zero_volume
        + negative_volume
        + duplicate_triangles
        + nonmanifold_edge
        + nonmanifold_face
    )
    metrics = {
        "initial_nodes": int(len(model["initial_nodes_xy"])),
        "final_nodes": int(len(model["final_nodes_xy"])),
        "initial_triangles": int(len(model["initial_triangles"])),
        "final_triangles": int(len(model["final_triangles"])),
        "added_nodes": int(
            len(model["final_nodes_xy"]) - len(model["initial_nodes_xy"])
        ),
        "added_triangles": int(
            len(model["final_triangles"]) - len(model["initial_triangles"])
        ),
        "tetrahedra": int(len(model["tetrahedra"])) if has_solid else np.nan,
        "model_status": str(model.get("model_status", "valid")),
        "solid_error": str(model.get("solid_error", "")),
        "q_min": float(np.min(quality)),
        "q_p05": float(np.percentile(quality, 5)),
        "q_lt_020_count": int(np.sum(quality < QUALITY_THRESHOLD)),
        "q_lt_020_ratio": float(np.mean(quality < QUALITY_THRESHOLD)),
        "negative_area_count": negative_area,
        "degenerate_triangle_count": degenerate,
        "duplicate_triangle_count": duplicate_triangles,
        "nonmanifold_edge_count": nonmanifold_edge,
        "zero_volume_tetra_count": zero_volume if has_solid else np.nan,
        "negative_volume_tetra_count": negative_volume if has_solid else np.nan,
        "nonmanifold_face_count": nonmanifold_face,
        "invalid_unit_count": int(invalid) if has_solid else np.nan,
    }
    return metrics, quality


@dataclass(frozen=True, slots=True)
class PointPrediction:
    """A query status, containing-cell index, and interface elevations (m)."""

    status: str
    triangle_id: int
    predicted: np.ndarray


def barycentric_query(
    model: dict[str, np.ndarray], point_xy: np.ndarray
) -> PointPrediction:
    """Query interface elevations at an XY location using triangle interpolation.

    Parameters
    ----------
    model : dict of str to ndarray
        Stored final mesh and interface arrays.
    point_xy : ndarray, shape (2,)
        Query coordinates in metres.

    Returns
    -------
    PointPrediction
        Query status, triangle index, and elevations (I,) in metres. An invalid
        query retains NaN elevations and is included in coverage accounting.
    """
    nodes = np.asarray(model["final_nodes_xy"], dtype=float)
    triangles = np.asarray(model["final_triangles"], dtype=int)
    z = np.asarray(model["Z"], dtype=float)
    point = np.asarray(point_xy, dtype=float)
    finder = Triangulation(nodes[:, 0], nodes[:, 1], triangles).get_trifinder()
    candidate = int(finder(float(point[0]), float(point[1])))
    candidates = [candidate] if candidate >= 0 else range(len(triangles))
    for triangle_id in candidates:
        if triangle_id < 0:
            continue
        triangle = triangles[int(triangle_id)]
        tri_xy = nodes[triangle]
        matrix = np.vstack((tri_xy.T, np.ones(3)))
        try:
            weights = np.linalg.solve(matrix, np.r_[point, 1.0])
        except np.linalg.LinAlgError:
            continue
        if (
            np.all(np.isfinite(weights))
            and np.min(weights) >= -1.0e-8
            and np.max(weights) <= 1.0 + 1.0e-8
        ):
            predicted = weights @ z[triangle]
            if np.all(np.isfinite(predicted)):
                return PointPrediction("valid", int(triangle_id), predicted)
    return PointPrediction("invalid", -1, np.full(z.shape[1], np.nan))


def front_blocks(model: dict[str, np.ndarray]) -> list[np.ndarray] | None:
    """Unpack per-body front segments as arrays of shape (S, 2, 2), in metres."""
    required = (
        "front_points_xy",
        "front_segments",
        "front_point_offsets",
        "front_segment_offsets",
    )
    if not all(key in model for key in required):
        return None
    points = model["front_points_xy"]
    segments = model["front_segments"]
    point_offsets = model["front_point_offsets"]
    segment_offsets = model["front_segment_offsets"]
    return [
        points[point_offsets[body] : point_offsets[body + 1]][
            segments[segment_offsets[body] : segment_offsets[body + 1]]
        ]
        for body in range(len(segment_offsets) - 1)
    ]


def evaluation_bounds(coords: np.ndarray) -> tuple[float, float, float, float]:
    """Return the survey extent as (xmin, xmax, ymin, ymax), in metres."""
    return (
        float(np.min(coords[:, 0])),
        float(np.max(coords[:, 0])),
        float(np.min(coords[:, 1])),
        float(np.max(coords[:, 1])),
    )


def inside_bounds(
    points: np.ndarray, bounds: tuple[float, float, float, float]
) -> np.ndarray:
    """Select finite XY points inside the common evaluation extent."""
    points = np.asarray(points, dtype=float)
    xmin, xmax, ymin, ymax = bounds
    return (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= xmin - GEOMETRY_ATOL)
        & (points[:, 0] <= xmax + GEOMETRY_ATOL)
        & (points[:, 1] >= ymin - GEOMETRY_ATOL)
        & (points[:, 1] <= ymax + GEOMETRY_ATOL)
    )


def front_records(
    model: dict[str, np.ndarray], bounds: tuple[float, float, float, float]
) -> tuple[list[dict[str, Any]], np.ndarray, int]:
    """Return front segments, endpoint/midpoint samples, and invalid-segment count."""
    blocks = front_blocks(model)
    if blocks is None:
        return [], np.empty((0, 2)), 0
    records = []
    selected = []
    invalid = 0
    for body, block in enumerate(blocks):
        if len(block) == 0:
            continue
        lengths = np.linalg.norm(block[:, 1] - block[:, 0], axis=1)
        invalid += int(
            np.sum(~np.isfinite(block).all(axis=(1, 2)) | (lengths <= GEOMETRY_ATOL))
        )
        keep = inside_bounds(block.mean(axis=1), bounds)
        for segment_id, segment in enumerate(block[keep]):
            p, q = segment
            records.append(
                {
                    "body": body,
                    "segment_id": segment_id,
                    "x1": float(p[0]),
                    "y1": float(p[1]),
                    "x2": float(q[0]),
                    "y2": float(q[1]),
                }
            )
            selected.extend((p, q, segment.mean(axis=0)))
    return (
        records,
        np.asarray(selected, dtype=float) if selected else np.empty((0, 2)),
        invalid,
    )


def unit_presence_area(
    model: dict[str, np.ndarray], bounds: tuple[float, float, float, float]
) -> np.ndarray:
    """Sum supported triangle areas per body within the survey extent, in m2."""
    triangles = model["final_triangles"]
    points = model["final_nodes_xy"][triangles]
    signed = (points[:, 1, 0] - points[:, 0, 0]) * (
        points[:, 2, 1] - points[:, 0, 1]
    ) - (points[:, 1, 1] - points[:, 0, 1]) * (points[:, 2, 0] - points[:, 0, 0])
    inside = inside_bounds(points.mean(axis=1), bounds)
    return np.sum(
        (np.abs(signed) / 2.0)[inside, None] * model["cell_presence_mask"][inside],
        axis=0,
    )


def component_count(
    model: dict[str, np.ndarray], bounds: tuple[float, float, float, float]
) -> tuple[int, int]:
    """Count front graph components using the established six-decimal node grouping."""
    blocks = front_blocks(model)
    if blocks is None:
        return 0, 0
    total = 0
    invalid = 0
    for block in blocks:
        if len(block) == 0:
            continue
        lengths = np.linalg.norm(block[:, 1] - block[:, 0], axis=1)
        invalid += int(
            np.sum(~np.isfinite(block).all(axis=(1, 2)) | (lengths <= GEOMETRY_ATOL))
        )
        selected = block[
            inside_bounds(block.mean(axis=1), bounds) & (lengths > GEOMETRY_ATOL)
        ]
        parent = []
        nodes = {}

        def add(key: tuple[float, float]) -> int:
            if key not in nodes:
                nodes[key] = len(parent)
                parent.append(len(parent))
            return nodes[key]

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for segment in selected:
            a = add(tuple(np.round(segment[0], 6)))
            b = add(tuple(np.round(segment[1], 6)))
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                parent[root_b] = root_a
        total += len({find(index) for index in range(len(parent))})
    return total, invalid


@dataclass(frozen=True, slots=True)
class FrontSummary:
    """Boundary records, sampled XY (m), supported area (m2), and connectivity."""

    records: list[dict[str, Any]]
    points: np.ndarray
    area: float
    components: int
    invalid_count: int


def summarize_fronts(
    model: dict[str, np.ndarray], bounds: tuple[float, float, float, float]
) -> FrontSummary:
    """Compute each scenario's boundary geometry once for all comparisons."""
    records, points, invalid = front_records(model, bounds)
    components, invalid_graph = component_count(model, bounds)
    return FrontSummary(
        records,
        points,
        float(np.sum(unit_presence_area(model, bounds))),
        components,
        max(invalid, invalid_graph),
    )


def front_stability(full: FrontSummary, masked: FrontSummary) -> dict[str, Any]:
    """Measure boundary changes relative to the full reconstruction.

    Parameters
    ----------
    full, masked : FrontSummary
        Precomputed geometry over the same survey extent.

    Returns
    -------
    dict
        Symmetric nearest-sample mean/P90 distance (m), total supported-area
        change (%), graph component change, and invalid-front count.

    Notes
    -----
    Samples comprise endpoints and midpoints. This is sensitivity relative to
    the full reconstruction, not accuracy against independent geological fronts.
    """
    full_points, mask_points = full.points, masked.points
    if len(full_points) == 0 or len(mask_points) == 0:
        return {
            "chamfer_mean_m": np.nan,
            "chamfer_p90_m": np.nan,
            "area_change_rate_pct": np.nan,
            "component_change": np.nan,
            "invalid_front": np.nan,
        }
    tree_full, tree_mask = cKDTree(full_points), cKDTree(mask_points)
    distances = np.r_[tree_full.query(mask_points)[0], tree_mask.query(full_points)[0]]
    return {
        "chamfer_mean_m": float(np.mean(distances)),
        "chamfer_p90_m": float(np.percentile(distances, 90)),
        "area_change_rate_pct": 100.0 * abs(masked.area - full.area) / full.area
        if full.area > 0
        else np.nan,
        "component_change": int(masked.components - full.components),
        "invalid_front": int(max(full.invalid_count, masked.invalid_count)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Write a metric table with a stable column order and UTF-8 BOM."""
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in fields} for row in rows
        )


def error_metrics(values: list[float], total: int) -> dict[str, Any]:
    """Summarize valid errors in metres, retaining the full denominator for coverage."""
    errors = np.asarray(values, dtype=float)
    return {
        "valid_n": len(errors),
        "coverage": len(errors) / total,
        "MAE": float(np.mean(np.abs(errors))) if len(errors) else np.nan,
        "RMSE": float(np.sqrt(np.mean(errors**2))) if len(errors) else np.nan,
        "P90_abs_error": float(np.percentile(np.abs(errors), 90))
        if len(errors)
        else np.nan,
    }


def run_evaluation(
    input_path: Path = DEFAULT_INPUT, output: Path = DEFAULT_OUTPUT
) -> None:
    """Rebuild all leave-one-borehole-out scenarios and write evaluation tables.

    Parameters
    ----------
    input_path : pathlib.Path, optional
        Complete elevation reference table. The example includes model-completed
        values; it is not entirely independent observed truth.
    output : pathlib.Path, optional
        New or empty output directory; existing results are never reused.

    Notes
    -----
    Every fold retains all XY coordinates and clears every interface elevation
    of one borehole. Borehole and interface counts come from the input table.
    All scenarios retain their model status; runtime is not a regression metric.
    """
    input_path, output = Path(input_path), Path(output)
    info = read_input(input_path)
    check_output_directory(output)
    output.mkdir(parents=True, exist_ok=True)
    evaluation_start = time.perf_counter()
    paths, run_times = build_evaluation_models(info, output)
    models = {key: load_model(path) for key, path in paths.items()}
    # Use the entire survey extent for all full-versus-masked evaluations.
    bounds = evaluation_bounds(info.coords)
    drills = info.drills
    bodies = [str(name) for name in models["full"]["body_names"]]
    mesh_rows, quality_rows = [], []
    for fold, model in models.items():
        state = "full" if fold == "full" else "mask"
        metrics, quality = mesh_metrics(model)
        mesh_rows.append(
            {
                "state": state,
                "fold": fold,
                "runtime_s": run_times.get(fold, np.nan),
                **metrics,
            }
        )
        quality_rows.extend(
            {"state": state, "fold": fold, "quality": float(q)} for q in quality
        )

    predictions = {
        drill: barycentric_query(models[drill], info.coords[i])
        for i, drill in enumerate(drills)
    }
    interface_rows, thickness_rows, error_rows, thickness_error_rows = [], [], [], []
    for i, drill in enumerate(drills):
        result = predictions[drill]
        reference = info.reference_z[i]
        predicted = result.predicted
        for j, name in enumerate(info.interfaces):
            signed = predicted[j] - reference[j]
            error_rows.append(
                {
                    "drill": drill,
                    "interface": name,
                    "reference": float(reference[j]),
                    "status": result.status,
                    "triangle_id": result.triangle_id,
                    "predicted": float(predicted[j]),
                    "signed_error": float(signed),
                    "abs_error": float(abs(signed)),
                }
            )
        for j, name in enumerate(bodies):
            predicted_h, reference_h = np.diff(predicted)[j], np.diff(reference)[j]
            signed = predicted_h - reference_h
            thickness_error_rows.append(
                {
                    "drill": drill,
                    "body": name,
                    "reference": float(reference_h),
                    "status": result.status,
                    "predicted": float(predicted_h),
                    "signed_error": float(signed),
                    "abs_error": float(abs(signed)),
                }
            )
    for name in info.interfaces:
        interface_rows.append(
            {
                "interface": name,
                **error_metrics(
                    [
                        row["signed_error"]
                        for row in error_rows
                        if row["interface"] == name and row["status"] == "valid"
                    ],
                    len(drills),
                ),
            }
        )
    for name in bodies:
        thickness_rows.append(
            {
                "body": name,
                **error_metrics(
                    [
                        row["signed_error"]
                        for row in thickness_error_rows
                        if row["body"] == name and row["status"] == "valid"
                    ],
                    len(drills),
                ),
            }
        )

    front_rows, overlay_rows = [], []
    front_summaries = {
        fold: summarize_fronts(model, bounds) for fold, model in models.items()
    }
    for fold, model in models.items():
        state = "full" if fold == "full" else "mask"
        records = front_summaries[fold].records
        overlay_rows.extend(
            {"state": state, "fold": fold, **record} for record in records
        )
        if state == "mask":
            front_rows.append(
                {
                    "fold": fold,
                    **front_stability(front_summaries["full"], front_summaries[fold]),
                }
            )

    tables = {
        "mesh_metrics.csv": mesh_rows,
        "triangle_quality.csv": quality_rows,
        "interface_metrics.csv": interface_rows,
        "thickness_metrics.csv": thickness_rows,
        "masked_prediction_errors.csv": error_rows,
        "masked_thickness_errors.csv": thickness_error_rows,
        "front_metrics.csv": front_rows,
        "front_overlay.csv": overlay_rows,
    }
    for filename, rows in tables.items():
        write_csv(output / filename, rows, list(rows[0]))
    metadata = {
        "interfaces": list(info.interfaces),
        "masked_drills": list(drills),
        "fold_count": len(drills),
        "masking": "retain XY; clear all interface elevations for one drill per fold",
        "reference": "completed input elevations; full model for front stability",
        "evaluation_bounds": list(bounds),
        "quality_threshold": QUALITY_THRESHOLD,
        "body_names": bodies,
        "runtime_s": run_times,
        "total_runtime_s": time.perf_counter() - evaluation_start,
        "interface_solver_eps": INTERFACE_SOLVER_EPS,
        "model_status": {
            fold: str(model["model_status"]) for fold, model in models.items()
        },
        "surface_only_folds": [
            fold for fold, model in models.items() if "tetrahedra" not in model
        ],
    }
    (output / "evaluation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"evaluation data written to {output}")


def main() -> None:
    """Parse input/output paths and run the complete masking experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_evaluation(args.input, args.output_dir)


if __name__ == "__main__":
    main()
