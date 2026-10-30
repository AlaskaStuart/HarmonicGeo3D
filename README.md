# HarmonicGeo3D

**Three-dimensional geological modelling with harmonic interpolation**

HarmonicGeo3D provides a Python workflow for three-dimensional geological modelling from borehole coordinates and stratigraphic interface elevations. It combines graph harmonic interpolation, pinch-out boundary extraction, local triangular mesh refinement and constrained joint reconstruction of geological interfaces. The repository includes example datasets, modelling and evaluation scripts, and Jupyter notebooks for visualising the workflow and results.

## Environment and dependencies

HarmonicGeo3D can be run directly from the project directory without installing it as a Python package.

Reference environment: Python 3.11.15 on macOS 26.7.2 (ARM64). Package dependencies and their versions are specified in `requirements.txt`. Optimisation problems are solved with CVXPY using OSQP.

If the required dependencies are already available, no additional setup is needed. Otherwise, install them in the active Python environment:

```bash
python -m pip install -r requirements.txt
```

When running the Jupyter notebooks, select a Python kernel that uses the same environment.

## Usage

Run the following commands from the project directory using the prepared Python environment.

### 1. Build the geological model

```bash
python run_model.py
```

This reads `examples/input/layers.csv` and writes the modelling results to `outputs/model/`.

### 2. Run the borehole masking evaluation

```bash
python run_evaluation.py
```

This reads `examples/input/completed_drill_surfaces.csv` and writes the evaluation results to `outputs/evaluation/`.

### 3. Visualise the results

Open the required notebook in a Jupyter-compatible editor and run its cells in order.

- Notebooks `01`–`05` visualise the modelling results.
- Notebook `06` visualises the evaluation results.

Figures are saved to `outputs/figures/` by default.

Output directories are created automatically. Modelling and evaluation require new or empty output directories; existing results are not overwritten.

## Input data

Example input datasets are provided in `examples/input/`.

| File                           | Purpose                                                                                                    |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| `layers.csv`                   | Input for geological modelling, retaining 15 unknown interface elevations as blank cells.                  |
| `completed_drill_surfaces.csv` | Complete reference table for the borehole masking evaluation, including the 15 model-completed elevations. |

- `drill`: unique borehole identifier.
- `coord_x`, `coord_y`: local planar coordinates in metres.
- Interface columns: elevations in metres, ordered from bottom to top, with `Ground` representing the ground surface.

Blank interface elevations indicate unknown values and should remain blank rather than being replaced with zero. A unit is identified as having zero thickness when its bounding interface elevations are both known and equal within the numerical tolerance.

The example datasets use translated local planar coordinates. Original absolute coordinates and coordinate-translation constants are not included.

The evaluation requires a complete elevation table. Reconstruction errors are calculated relative to the supplied reference table, which combines supplied interface elevations and model-completed values.

## Modelling workflow

The modelling workflow consists of five main steps:

1. Construct an initial triangular mesh from borehole coordinates and the dominant directions of the borehole arrangement.
2. Solve a graph harmonic field for each geological unit using known unit-presence and unit-absence controls.
3. Extract model-inferred pinch-out boundaries using a prescribed contour threshold, and locally refine the mesh along these boundaries.
4. Jointly reconstruct geological interfaces subject to known elevations, stratigraphic ordering and zero-thickness constraints.
5. Construct three-dimensional geological bodies from the reconstructed interfaces.

The harmonic fields are dimensionless auxiliary fields used for boundary inference; they do not represent unit thickness or calibrated occurrence probabilities.

The contour threshold and modelling settings for the supplied example are defined in `run_model.py`. Evaluation-specific settings are defined in `run_evaluation.py`.

## Outputs

### Modelling outputs

By default, `run_model.py` writes the following files to `outputs/model/`:

| File | Content |
| --- | --- |
| `demo_model.npz` | Compressed model data containing the mesh, reconstructed interfaces, inferred pinch-out boundaries and tetrahedral geological bodies. |
| `completed_drill_surfaces.csv` | Interface elevations at boreholes, including reconstructed values for previously unknown elevations. |
| `completed_mesh_surfaces.csv` | Interface elevations at all refined mesh nodes. |

### Evaluation outputs

By default, `run_evaluation.py` writes its results to `outputs/evaluation/`.

| File | Content |
| --- | --- |
| `interface_metrics.csv`, `thickness_metrics.csv` | Reconstruction error summaries for interface elevations and unit thicknesses. |
| `masked_prediction_errors.csv`, `masked_thickness_errors.csv` | Individual predictions, reference values and reconstruction errors. |
| `mesh_metrics.csv`, `triangle_quality.csv` | Mesh statistics, reconstruction status and triangle shape quality. |
| `front_metrics.csv`, `front_overlay.csv` | Changes in inferred pinch-out boundaries and boundary coordinates for plotting. |
| `evaluation_metadata.json` | Evaluation settings, borehole and interface identifiers, runtime information and reconstruction statuses. |

The `full/` and `masked/` subdirectories contain the unmasked and individual masked reconstructions. The `masked_inputs/` subdirectory contains the corresponding masked input tables.

Evaluation runs may also produce `surface_model.npz`, labelled `surface_only`, when interface reconstruction succeeds but solid construction cannot be completed. These files contain interface results without tetrahedral solid cells.

Pre-generated numerical reference outputs are provided in `examples/expected_output/`. Normal runs write their results to `outputs/`.

Figures generated by the notebooks are saved to `outputs/figures/` by default.

## Scope

HarmonicGeo3D is intended for layered geological structures represented by a consistent bottom-to-top sequence of interfaces. Initial mesh construction relies on two recognisable dominant directions in the borehole arrangement.

Each interface is represented by a single elevation at a given planar location. The current implementation does not explicitly represent faults, overturned strata or interfaces with multiple elevations at the same planar location.

Stratigraphic correlation and interface identification should be completed before modelling. The plotting notebooks are configured for the supplied example and may require adjustments when unit names, unit counts or plotting layouts differ.

## Licence

The source code is licensed under the MIT License. See [LICENSE](LICENSE) for details.
