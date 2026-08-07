#!/usr/bin/env python
"""Generate NSE cylinder FOM training data from a YAML config.

Usage:
    pixi run mpirun -n <N> python examples/generate_fom_data.py <config.yaml>

See examples/fom_configs/ for example configs -- full_options.yaml documents
every field, standard_T20.yaml / efr_T20.yaml are minimal examples.
"""
import argparse
from pathlib import Path

from efr_opinf.fom.cylinder import load_config, run_cylinder_fom


def main():
    parser = argparse.ArgumentParser(
        description="Generate NSE cylinder FOM training data from a YAML config."
    )
    parser.add_argument(
        "config", type=Path,
        help="Path to a YAML config file (see examples/fom_configs/).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_cylinder_fom(config)


if __name__ == "__main__":
    main()
