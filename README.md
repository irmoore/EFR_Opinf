# EFR-OpInf

Evolve Filter Relax Operator Inference (EFR-OpInf) for reduced-order
modeling of fluid flow problems, built on [FEniCSx](https://fenicsproject.org/)
and [OpInf](https://operator-inference.github.io/opinf/source/index.html).

## Install

Major dependencies are the FEniCSx stack (`dolfinx`, `mpi4py`, `petsc4py`, `basix`,
`ufl`, `adios4dolfinx`). This library contains C++ code that requires downloading 
conda-forge binaries, so this package is not pip installable. 

This project uses [pixi](https://pixi.sh) to manage both the
conda-forge stack and Python (mainly OpInf). Pixi is similar 
to conda. If you have any concerns about using the curl script below, please go
to their website and determine your installation preferences. 

```bash
curl -fsSL https://pixi.sh/install.sh | bash   # if pixi isn't installed yet
```

```bash
git clone git@github.com:irmoore/EFR_Opinf.git
cd EFR_Opinf
pixi install
```

This creates a pixi environment with the full FEniCSx stack plus an editable
install of the `efr_opinf` package. This environment uses package versions known
to be compatible as of May 2026.

## Running examples

Mesh/data/results paths (`Meshes/`, `NSE_data/`, `Results/`) are resolved via
`src/efr_opinf/_paths.py`, which anchors them to the repository root based on
the installed package's own location, not the current working directory. So
scripts and notebooks under `examples/` can be run from anywhere:

```bash
pixi run python examples/test_CDR_parametric.py
```

If you want to run multiple tests, type

```bash
pixi shell
#your commands here
exit
```

This functions almost identically to a conda environment. 

## How to reproduce paper results

1. CDR: go to examples directory and run:

```bash
pixi run python examples/test_CDR_parametric.py
```

2. NSE FPC: Go to EFR_OpInf_script.ipynb. follow the instructions to generate FOM data and run the cells. 

## Package layout

- `src/efr_opinf/_paths.py`: `PROJECT_ROOT`, `MESH_DIR`, `DATA_DIR`, `RESULTS_DIR`
  constants, anchored to the package's own location so path resolution doesn't
  depend on the current working directory.
- `src/efr_opinf/interfaces/`: This code interfaces between FEniCSx and OpInf. The classes in here handle
passing data between them and performing FE calculations. `nse.py` and `cdr.py` are used for the flow past a cylinder and convection diffusion examples respectively.  
- `src/efr_opinf/rom/`: Operator Inference reduced-order models built on top of
  the interfaces above. Again, there is an option for NSE and CDR. 
- `src/efr_opinf/fom/`: Data collection for full-order-model Navier-Stokes cylinder flow.
- `examples/`: standalone experiment scripts and notebooks.
- `Meshes/`: mesh files used by the examples. CDR is saved, NSE will be generated on the fly. 
- `Results/`: Stores all results, organized by problem and parameters. 
- `NSE_data/`: NSE results use data which is computed once, then loaded. This is expensive. 
