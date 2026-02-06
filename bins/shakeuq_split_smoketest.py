"""Smoke test harness for SHAKEuq split + SHAKElab."""

from __future__ import annotations

import argparse
from pathlib import Path

from modules.SHAKEuq import SHAKEuq
from modules.SHAKElab import SHAKElab


def parse_args():
    parser = argparse.ArgumentParser(description="SHAKEuq split smoke test")
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--shakemap-folder", required=True)
    parser.add_argument("--stations-folder", required=True)
    parser.add_argument("--rupture-folder", required=True)
    parser.add_argument("--dyfi-cdi-file", default=None)
    parser.add_argument("--versions", nargs="+", type=int, required=True)
    parser.add_argument("--lab-root", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    uq = SHAKEuq(
        event_id=args.event_id,
        shakemap_folder=args.shakemap_folder,
        stations_folder=args.stations_folder,
        rupture_folder=args.rupture_folder,
        dyfi_cdi_file=args.dyfi_cdi_file,
        version_list=args.versions,
    )

    uq.uq_build_dataset(
        event_id=args.event_id,
        version_list=args.versions,
        base_folder="./export/SHAKEuq",
        stations_folder=args.stations_folder,
        rupture_folder=args.rupture_folder,
        imts=("MMI", "PGA"),
    )

    lab = SHAKElab(args.lab_root, export_base="./export/SHAKElab")
    events_df = lab.index_events()
    subset = events_df.head(3)
    sweep = lab.run_cdi_filter_sweep(
        subset,
        filter_configs=[
            {"config_id": "loose", "max_dist_km": 400, "min_nresp": 1},
            {"config_id": "strict", "max_dist_km": 200, "min_nresp": 5},
        ],
        imt="MMI",
    )
    if not sweep.empty:
        lab.plot_cdi_sweep_summary(sweep, Path("./export/SHAKElab") / "cdi_sweep")


if __name__ == "__main__":
    main()
