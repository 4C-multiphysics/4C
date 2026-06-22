# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import argparse
import json
import pathlib
import yaml


def extract_benchmark_test_result(timings: dict) -> dict:
    """
    Extracts all relevant benchmark data into the format that
    4C expects for storing and visualization.
    """
    if "context" not in timings:
        raise RuntimeError(
            f"Benchmark test result file {timings} does not contain a "
            "'context' field, which is required for extracting the benchmark "
            "test result."
        )

    # we can just take the context of the benchmark itself
    context = timings["context"]

    # make sure it contains the date field
    if "date" not in timings["context"]:
        raise RuntimeError(
            f"Benchmark test result file {timings} does not contain a 'date' "
            "field in the 'context' field, which is required for extracting the "
            "benchmark test result."
        )

    data = []
    for benchmark in timings["benchmarks"]:
        data.append(
            {
                "name": benchmark["name"],
                "unit": benchmark["time_unit"],
                "measurements": {
                    "min": benchmark["real_time"],
                    "max": benchmark["real_time"],
                    "mean": benchmark["real_time"],
                },
            }
        )

    return {
        "context": context,
        "data": data,
    }


def main():
    """
    Collects all benchmark results and converts them into a collection
    file in the respective format that 4C benchmark (and performance)
    reporting utilities expect.
    """
    parser = argparse.ArgumentParser(description="Collect benchmark test results.")
    parser.add_argument(
        "base_dir",
        help="A directory that contains the benchmark test results to be "
        "collected. Benchmark test data is recursively collected in all subdirectories.",
    )
    parser.add_argument(
        "target_file",
        help="The file to which the collected benchmark test results should be written.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Don't fail if no benchmark test results are found.",
    )
    args = parser.parse_args()

    # we have to search for all files named /test_name/*-benchmark.json
    suffix = "-benchmark.json"
    path_root = pathlib.Path(args.base_dir)
    files = list(path_root.rglob(f"*{suffix}"))

    benchmark_test_results = {}

    for file in files:
        with open(file, "r") as f:
            timings = yaml.safe_load(f)

        test_name = file.name.removesuffix(suffix)

        if test_name in benchmark_test_results:
            raise RuntimeError(
                f"Duplicate test name {test_name} found. Test names are derived from the "
                "parent directory of the benchmark test result files, so make sure that "
                "there are no two benchmark test result files with the same parent directory "
                "name."
            )

        benchmark_test_results[test_name] = extract_benchmark_test_result(timings)

    if not benchmark_test_results and not args.allow_empty:
        raise RuntimeError(
            f"No benchmark test results found in {args.base_dir}. Use --allow-empty to allow "
            "writing an empty result file."
        )

    with open(args.target_file, "w") as f:
        json.dump(benchmark_test_results, f)


if __name__ == "__main__":
    main()
