# Example data and numerical references

These files provide input data and numerical reference outputs for the
geological modelling example and borehole masking evaluation.

The supplied example data and derived numerical reference outputs are **not
covered by the source code's MIT licence**.

## Input tables

Both tables contain 39 boreholes, identical XY locations and row ordering, and
six interfaces ordered from bottom to top. Coordinates and elevations are in
metres. The first three columns are `drill,coord_x,coord_y`; remaining columns
are `layer-00-bot` through `layer-04-bot`, followed by `Ground`.

| File | Purpose |
| --- | --- |
| `input/layers.csv` | Modelling observations, retaining 15 missing elevations as blank cells. |
| `input/completed_drill_surfaces.csv` | Complete evaluation reference, including the 15 model-completed values. |

Blank elevations mean unknown values. They are never replaced by zero. A body
has a zero-thickness observation only when both bounding elevations are known
and equal within the numerical tolerance. Body `body-00` lies between
`layer-00-bot` and `layer-01-bot`; the same naming rule continues upward,
with `body-04` between `layer-04-bot` and `Ground`.

Both tables use translated local XY coordinates with the same translation.
No rotation, scaling, or random perturbation was applied. Borehole names,
row order, interface values, field names, and blank positions are retained.
Elevations keep their original datum. Translation is not irreversible
anonymisation. The original absolute coordinates and translation constants
are not included.

## Evaluation interpretation

The reference table includes model-completed elevations, so its entire content
must not be described as independent field measurements. Each masking fold
retains every borehole XY location and blanks every interface elevation at one
borehole. The other rows use the complete reference values. Evaluation is
therefore a reconstruction sensitivity experiment relative to this completed
table, rather than independent validation at previously unseen XY locations.

Interface and thickness errors are measured against the reference table.
Boundary changes compare each masked reconstruction with a full reconstruction
of that same complete table. That full reconstruction differs conceptually
from modelling `layers.csv`, which retains missing observations. Neither
boundary Chamfer distance nor supported-area change establishes accuracy
against an independently mapped geological front.

Model status is explicit in every NPZ and evaluation metadata. A `valid`
package includes validated tetrahedral bodies. A `surface_only` package stores
the reconstructed interfaces under `surface_model.npz`, with the solid
construction diagnostic in `solid_error`; it contains no tetrahedral arrays.
Consult `outputs/evaluation/evaluation_metadata.json` and `mesh_metrics.csv`
when interpreting individual scenarios. Missing solid metrics remain NaN.

## Fixed reference outputs

`expected_output/` contains numerical outputs regenerated with the public local
input copies and the reorganized implementation:

- `model/demo_model.npz` and both completed surface CSV tables;
- `evaluation/interface_metrics.csv`, `thickness_metrics.csv`,
  `front_metrics.csv`, and `mesh_metrics.csv`.

These are fixed comparison targets. Normal modelling, evaluation, and plotting
write to `outputs/` and never update the reference directory. The two CLI
entry points reject attempts to use the fixed reference directory for output.
