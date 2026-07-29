# EFR-OpInf

Evolve Filter Relax Operator Inference (EFR-OpInf) for reduced-order
modeling of fluid flow problems, built on [FEniCSx](https://fenicsproject.org/)
and [OpInf](https://operator-inference.github.io/opinf/source/index.html).

## Install

Dependencies include the FEniCSx stack (`dolfinx`, `mpi4py`, `petsc4py`, `basix`,
`ufl`, `adios4dolfinx`), which are conda-forge packages and aren't
pip-installable. This project uses [pixi](https://pixi.sh) to manage both the
conda-forge stack and the pip-installable Python package in one place. Pixi is similar 
to conda environments. 

```bash
curl -fsSL https://pixi.sh/install.sh | bash   # if pixi isn't installed yet
pixi install
```

This creates a pixi environment with the full FEniCSx stack plus an editable
install of the `efr_opinf` package.

Note that pyproject.toml has been pinned to specific version numbers known to work with
this package arround May 2026. 

## Running examples

Mesh/data/results paths (`Meshes/`, `NSE_data/`, `Results/`) are resolved via
`src/efr_opinf/_paths.py`, which anchors them to the repository root based on
the installed package's own location — not the current working directory. So
scripts and notebooks under `examples/` can be run from anywhere:

```bash
pixi run python examples/test_CDR_parametric.py
```

`examples/test_function_load.py` references `rbnicsx`, which isn't declared in
`pyproject.toml`. It needs to be set up separately if you want to run that
specific example.

## Package layout

- `src/efr_opinf/_paths.py` — `PROJECT_ROOT`, `MESH_DIR`, `DATA_DIR`, `RESULTS_DIR`
  constants, anchored to the package's own location so path resolution doesn't
  depend on the current working directory.
- `src/efr_opinf/interfaces/` — FEniCSx mesh/function-space interfaces:
  `nse.py` (current), `nse_old.py` (superseded, kept for manual comparison
  before removal), `cdr.py` (separate convection-diffusion-reaction case).
- `src/efr_opinf/rom/` — Operator Inference reduced-order models built on top of
  the interfaces above.
- `src/efr_opinf/fom/` — full-order-model Navier-Stokes cylinder flow drivers.
- `examples/` — standalone experiment scripts and notebooks.
- `Meshes/` — mesh files used by the examples.
