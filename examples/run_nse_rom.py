#!/usr/bin/env python
"""Run the NSE OpInf ROM pipeline (train/validate/test) from a YAML config.

Usage:
    pixi run python examples/run_nse_rom.py <config.yaml>

See examples/rom_configs/ for example configs -- full_options.yaml documents
every field, nse_r20.yaml is a minimal example. Requires FOM data already
generated -- see examples/generate_fom_data.py.
"""
import argparse
from pathlib import Path

from efr_opinf.rom.nse import load_config, run_nse_rom


def main():
    parser = argparse.ArgumentParser(
        description="Run the NSE OpInf ROM pipeline from a YAML config."
    )
    parser.add_argument(
        "config", type=Path,
        help="Path to a YAML config file (see examples/rom_configs/).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_nse_rom(config)


if __name__ == "__main__":
    main()
