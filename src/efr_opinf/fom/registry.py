"""Locate previously generated Cylinder FOM snapshot data by time horizon."""

import json
from pathlib import Path

from efr_opinf._paths import DATA_DIR


class FOMDataNotFoundError(RuntimeError):
    pass


def find_fom_data(min_T: float, efr: bool = False) -> Path:
    """Return the .bp snapshot file covering at least [0, min_T].

    Only T is checked against the sidecar metadata written by
    efr_opinf.fom.cylinder.run_cylinder_fom -- dt/sample_gap are recorded
    there for provenance but not enforced here. Among matches, the one
    with the smallest sufficient T is returned.
    """
    flavor = "efr" if efr else "standard"
    subdir = DATA_DIR / "Cylinder" / flavor

    candidates = []
    if subdir.exists():
        for sidecar in subdir.glob("*.json"):
            with open(sidecar) as f:
                meta = json.load(f)
            bp_path = sidecar.with_suffix(".bp")
            if meta["T"] >= min_T and bp_path.exists():
                candidates.append((meta["T"], bp_path))

    if not candidates:
        raise FOMDataNotFoundError(
            f"No {flavor} FOM data found covering T >= {min_T} in {subdir}. "
            "Generate it with:\n"
            "  pixi run python -m efr_opinf.fom.cylinder --config <path_to_config.yaml>\n"
            f"using a config with time.T >= {min_T}"
            + (" and an `efr:` block." if efr else ".")
        )

    candidates.sort(key=lambda pair: pair[0])
    return candidates[0][1]
