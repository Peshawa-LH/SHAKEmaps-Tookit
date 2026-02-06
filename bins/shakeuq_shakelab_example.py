"""Example script for SHAKEuq + SHAKElab workflows."""

from __future__ import annotations

import argparse

from modules.SHAKEuq import SHAKEuq
from modules.SHAKElab import SHAKElab


def parse_args():
    parser = argparse.ArgumentParser(description="SHAKEuq/SHAKElab demo")
    parser.add_argument("--event-id", required=True, help="USGS event id")
    parser.add_argument("--shakemap-folder", required=True, help="Root folder for shakemap versions")
    parser.add_argument("--stations-folder", required=True, help="Root folder for stationlist versions")
    parser.add_argument("--rupture-folder", required=True, help="Root folder for rupture versions")
    parser.add_argument("--dyfi-cdi-file", default=None, help="Optional CDI file for event")
    parser.add_argument("--versions", nargs="+", type=int, required=True, help="Version list (e.g., 1 2 3)")
    parser.add_argument("--lab-root", default=None, help="Root folder for SHAKElab multi-event runs")
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
        imts=("MMI",),
    )

    if args.lab_root:
        lab = SHAKElab(args.lab_root, export_base="./export/SHAKElab")
        events_df = lab.index_events()
        sweep_df = lab.run_cdi_filter_sweep(
            events_df.head(2),
            filter_configs=[
                {"config_id": "loose", "max_dist_km": 400, "min_nresp": 1},
                {"config_id": "strict", "max_dist_km": 200, "min_nresp": 5},
            ],
            imt="MMI",
        )
        if not sweep_df.empty:
            lab.plot_cdi_sweep_summary(sweep_df, "./export/SHAKElab/cdi_sweep")


if __name__ == "__main__":
    main()
