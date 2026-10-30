"""Build a shared-mesh geological model and completed surface tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from scripts.build_surface_stack import (
    SurfaceStackObservations,
    load_surface_stack_observations,
)
from scripts.conformal_mesh import ConformalMesh, build_conformal_mesh
from scripts.generate_tri_net import generate_tri_net
from scripts.harmonic_interpolation import solve_harmonic_fields
from scripts.interface_qp import InterfaceSolution, solve_interface_stack
from scripts.mesh_operators import (
    build_final_mesh_operators,
    build_weighted_edge_difference,
)
from scripts.mesh_topology import build_mesh_topology
from scripts.pinch_front import PinchFront, extract_pinch_front
from scripts.solid_builder import SolidBuildError, build_solid_model
from scripts.validation import validate_complete_model, validate_conformal_mesh

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "examples" / "input" / "layers.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "model"
TAU_VALUE = 0.25
NODE_SOURCE_LABELS = ("original_node", "original_edge", "original_cell")


def _generated(value: float) -> str:
    return f"{float(value):.6f}"


def _write_csv_outputs(
    output_dir: Path,
    layer_csv: Path,
    observations: SurfaceStackObservations,
    mesh: ConformalMesh,
    solution: InterfaceSolution,
) -> None:
    """Write completed elevations while retaining the text of known observations."""
    with layer_csv.open(newline="", encoding="utf-8-sig") as stream:
        source_rows = list(csv.DictReader(stream))

    drill_path = output_dir / "completed_drill_surfaces.csv"
    mesh_path = output_dir / "completed_mesh_surfaces.csv"
    interfaces = observations.interface_names

    with drill_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(("drill", "coord_x", "coord_y", *interfaces))
        for node, drill in enumerate(observations.drill_names):
            source = source_rows[int(observations.source_row_indices[node])]
            values = [
                source[name].strip()
                if observations.observed_mask[node, interface]
                else _generated(solution.z[node, interface])
                for interface, name in enumerate(interfaces)
            ]
            writer.writerow(
                (drill, source["coord_x"].strip(), source["coord_y"].strip(), *values)
            )

    with mesh_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("node_id", "drill", "node_source_kind", "coord_x", "coord_y", *interfaces)
        )
        for node, xy in enumerate(mesh.nodes_xy):
            original = node < mesh.original_node_count
            source = None
            drill = ""
            if original:
                drill = str(observations.drill_names[node])
                source = source_rows[int(observations.source_row_indices[node])]
            values = [
                source[name].strip()
                if source is not None and observations.observed_mask[node, interface]
                else _generated(solution.z[node, interface])
                for interface, name in enumerate(interfaces)
            ]
            coords = (
                (source["coord_x"].strip(), source["coord_y"].strip())
                if source is not None
                else (_generated(xy[0]), _generated(xy[1]))
            )
            writer.writerow(
                (
                    node,
                    drill,
                    NODE_SOURCE_LABELS[int(mesh.node_source_kind[node])],
                    *coords,
                    *values,
                )
            )


def _front_arrays(fronts: tuple[PinchFront, ...]) -> dict[str, np.ndarray]:
    """Pack per-body front geometry with offsets and body-local segment indices."""
    point_offsets = [0]
    segment_offsets = [0]
    points = []
    segments = []
    for front in fronts:
        points.append(front.points_xy)
        segments.append(front.segments)
        point_offsets.append(point_offsets[-1] + len(front.points_xy))
        segment_offsets.append(segment_offsets[-1] + len(front.segments))
    return {
        "front_points_xy": np.concatenate(points) if points else np.empty((0, 2)),
        "front_segments": np.concatenate(segments)
        if segments
        else np.empty((0, 2), dtype=np.int64),
        "front_point_offsets": np.asarray(point_offsets, dtype=np.int64),
        "front_segment_offsets": np.asarray(segment_offsets, dtype=np.int64),
    }


def run_model(
    layer_csv: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    interface_solver_eps: float = 1.0e-9,
    save_surface_diagnostics: bool = False,
) -> tuple[Path, Path, Path]:
    """Build a geological model and two completed surface tables.

    Parameters
    ----------
    layer_csv : pathlib.Path, optional
        Observation table, including blank unknown elevations.
    output_dir : pathlib.Path, optional
        New or empty output directory. Fixed reference outputs are protected.
    interface_solver_eps : float, optional
        Absolute and relative interface-solver tolerance. The evaluation uses
        1e-10; ordinary modelling retains the original 1e-9 default.
    save_surface_diagnostics : bool, optional
        Preserve validated interface results as ``surface_model.npz`` if solid
        construction raises ``SolidBuildError``. Other exceptions propagate.

    Returns
    -------
    tuple of pathlib.Path
        NPZ path and two completed CSV paths. Only a complete solid is named
        ``demo_model.npz``; the alternative has explicit ``surface_only`` status.

    Raises
    ------
    FileExistsError
        If the output directory contains existing content.
    SolidBuildError
        If solid construction fails and diagnostic saving is disabled.
    """

    layer_csv = Path(layer_csv)
    output_dir = Path(output_dir)
    targets = (
        output_dir / "demo_model.npz",
        output_dir / "completed_drill_surfaces.csv",
        output_dir / "completed_mesh_surfaces.csv",
    )
    check_output_directory(output_dir)

    observations = load_surface_stack_observations(layer_csv, coords_csv=None)
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_triangles = np.asarray(
        generate_tri_net(observations.nodes_xy), dtype=np.int64
    )
    initial_topology = build_mesh_topology(observations.nodes_xy, initial_triangles)
    tau = (TAU_VALUE,) * observations.body_count

    edge_operator = build_weighted_edge_difference(
        observations.nodes_xy, initial_topology.edges
    )
    harmonic_solution = solve_harmonic_fields(
        edge_operator,
        observations.zero_thickness_control_mask,
        observations.present_control_mask,
        observations.unknown_control_mask,
        solver="OSQP",
        eps_abs=1.0e-9,
        eps_rel=1.0e-9,
        max_iter=100_000,
        residual_atol=1.0e-7,
    )
    fronts = tuple(
        extract_pinch_front(
            observations.nodes_xy,
            initial_topology,
            harmonic_solution[:, body],
            tau=tau[body],
            body_index=body,
            level_atol=1.0e-10,
        )
        for body in range(observations.body_count)
    )

    geometry_atol = 1.0e-9 * edge_operator.reference_length
    conformal_mesh = build_conformal_mesh(
        observations.nodes_xy,
        initial_topology,
        harmonic_solution,
        fronts,
        tau=tau,
        observed_z=observations.z,
        z_known=observations.observed_mask,
        source_row_indices=observations.source_row_indices,
        drill_names=observations.drill_names,
        geometry_atol=geometry_atol,
        area_atol=geometry_atol * edge_operator.reference_length,
        presence_atol=1.0e-9,
    )
    validate_conformal_mesh(
        conformal_mesh,
        observations.nodes_xy,
        initial_topology,
        harmonic_solution,
        tau=tau,
    )

    operators = build_final_mesh_operators(
        conformal_mesh.nodes_xy,
        conformal_mesh.topology.triangles,
        conformal_mesh.topology.edges,
        area_atol=conformal_mesh.area_tolerance,
        matrix_atol=1.0e-10,
    )
    beta = (1.0 / operators.difference.reference_length**2,) * observations.body_count
    model_solution = solve_interface_stack(
        conformal_mesh,
        operators,
        alpha=(1.0,) * observations.interface_count,
        beta=beta,
        solver="OSQP",
        eps_abs=interface_solver_eps,
        eps_rel=interface_solver_eps,
        max_iter=200_000,
        polishing=True,
        warm_start=False,
        residual_atol=1.0e-7,
    )
    solid_error = ""
    try:
        solid = build_solid_model(
            conformal_mesh,
            model_solution,
            thickness_atol=1.0e-7,
            tetra_volume_atol=1.0e-12,
        )
        validate_complete_model(
            conformal_mesh,
            model_solution,
            solid,
            tau=tau,
            residual_atol=1.0e-7,
            volume_abs_atol=1.0e-5,
            volume_rel_atol=1.0e-9,
        )
    except SolidBuildError as exc:
        if not save_surface_diagnostics:
            raise
        solid = None
        solid_error = str(exc)

    arrays = {
        "initial_nodes_xy": observations.nodes_xy,
        "initial_triangles": initial_topology.triangles,
        "drill_names": observations.drill_names,
        "interface_names": np.asarray(observations.interface_names),
        "body_names": np.asarray(observations.body_names),
        "body_boundaries": np.asarray(observations.body_boundaries),
        "tau": np.asarray(tau),
        "unit_presence_values": harmonic_solution,
        "final_nodes_xy": conformal_mesh.nodes_xy,
        "final_triangles": conformal_mesh.topology.triangles,
        "original_node_count": np.asarray(conformal_mesh.original_node_count),
        "pinch_edge_mask": conformal_mesh.pinch_edge_mask,
        "Z": model_solution.z,
        "cell_presence_mask": conformal_mesh.cell_presence_mask,
        "model_status": np.asarray("valid" if solid is not None else "surface_only"),
        "solid_error": np.asarray(solid_error),
        **_front_arrays(fronts),
    }
    if solid is not None:
        arrays.update(
            {
                "vertices_xyz": solid.vertices_xyz,
                "tetrahedra": solid.tetrahedra,
                "tetra_body_ids": solid.tetra_body_ids,
            }
        )
    else:
        targets = (output_dir / "surface_model.npz", *targets[1:])
    model_path = targets[0]
    np.savez_compressed(model_path, **arrays)
    _write_csv_outputs(
        output_dir, layer_csv, observations, conformal_mesh, model_solution
    )
    solid_summary = (
        f"{len(solid.tetrahedra)} tetrahedra"
        if solid is not None
        else f"surface only; solid failed: {solid_error}"
    )
    print(
        f"model: {len(observations.nodes_xy)} initial nodes, "
        f"{len(conformal_mesh.nodes_xy)} final nodes, {solid_summary}"
    )
    return targets


def check_output_directory(output_dir: Path) -> None:
    """Reject existing content and protect the fixed reference results."""
    directory = Path(output_dir).resolve()
    reference = (ROOT / "examples" / "expected_output").resolve()
    if directory.is_relative_to(reference) or reference.is_relative_to(directory):
        raise ValueError("The fixed reference directory cannot be used for output.")
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise FileExistsError("Output directory is not empty; choose a new directory.")


def main() -> None:
    """Parse paths and build the model with the documented defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_model(args.input, args.output_dir)


if __name__ == "__main__":
    main()
