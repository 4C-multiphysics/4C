# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import argparse
import os
import pathlib
import yaml
import json
from datetime import datetime, timezone
import platform
import multiprocessing


def extract_performance_test_result(timings: dict, file_path: pathlib.Path) -> dict:
    """
    Extracts all relevant performance data into the format that 4C expects for storing and visualization.
    """

    context = {
        "date": datetime.fromtimestamp(
            file_path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        "machine": platform.node(),
        "num_procs": timings["Number of processes"],
        "num_cpu": multiprocessing.cpu_count(),
        "processor": platform.processor(),
    }

    data = []
    for name, timing_data in timings["Total times"].items():
        data.append(
            {
                "name": name,
                "unit": timings["Time unit"],
                "measurements": {
                    "min": timing_data["MinOverProcs"],
                    "max": timing_data["MaxOverProcs"],
                    "mean": timing_data["MeanOverProcs"],
                },
            }
        )

    return {
        "context": context,
        "data": data,
    }


def main():
    parser = argparse.ArgumentParser(description="Collect performance test results.")
    parser.add_argument(
        "base_dir",
        help="A directory that contains the performance test results to be collected. Performance test data is recursively collected in all subdirectories.",
    )
    parser.add_argument(
        "target_file",
        help="The file to which the collected performance test results should be written.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Don't fail if no performance test results are found.",
    )
    args = parser.parse_args()

    # we have to search for all files named /test_name/*-timings.yaml
    suffix = "-timings.yaml"
    path_root = pathlib.Path(args.base_dir)
    files = list(path_root.rglob(f"*{suffix}"))

    performance_test_results = {}

    for file in files:
        with open(file, "r") as f:
            timings = yaml.safe_load(f)

        test_name = file.parent.name

        if test_name in performance_test_results:
            raise RuntimeError(
                f"Duplicate test name {test_name} found. Test names are derived from the parent directory of the performance test result files, so make sure that there are no two performance test result files with the same parent directory name."
            )

        performance_test_results[test_name] = extract_performance_test_result(
            timings, file
        )

    if len(performance_test_results) == 0 and not args.allow_empty:
        raise RuntimeError(
            f"No performance test results found in {args.base_dir} (searched for files with suffix {suffix})."
        )

    with open(args.target_file, "w") as f:
        json.dump(performance_test_results, f)


if __name__ == "__main__":
    main()
