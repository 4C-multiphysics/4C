# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import json
import argparse
import junitparser
import os


def detect_processors(properties):
    for prop in properties:
        if prop["name"] == "PROCESSORS":
            return int(prop["value"])

    return 1


def detect_timeout(properties):
    for prop in properties:
        if prop["name"] == "TIMEOUT":
            return float(prop["value"])

    return 1


def main():
    parser = argparse.ArgumentParser(description="Chunk Test Suite")
    parser.add_argument(
        "--chunks",
        "-c",
        default=15,
        type=int,
        help="Number of chunks to divide the test suite into",
    )
    parser.add_argument(
        "--junit-report",
        type=str,
        default=None,
        help="Junit report of the test suite for actual runtime of tests",
    )
    parser.add_argument(
        "ctest_info",
        type=str,
        help="Test summary file from ctest via ctest --show-only=json-v1",
    )

    args = parser.parse_args()

    # read junit report
    test_runtime = {}
    if args.junit_report is not None:
        xml = junitparser.JUnitXml.fromfile(args.junit_report)
        for suite in xml:
            for case in suite:
                test_runtime[case.name] = case.time

    # read general test info
    with open(args.ctest_info, "r", encoding="utf8") as file:
        test_summary = json.load(file)

    # determine expected processor runtimes per test
    expected_test_processor_times = []
    for test_number, test in enumerate(test_summary["tests"]):
        properties = test["properties"]
        name = test["name"]
        num_procs = detect_processors(properties)
        timeout = detect_timeout(properties)

        if name in test_runtime:
            expected_proc_runtime = test_runtime[name] * num_procs
        else:
            expected_proc_runtime = timeout * num_procs

        expected_test_processor_times.append(expected_proc_runtime)

    # compute average processor time per chunk
    average_processor_time = sum(expected_test_processor_times) / args.chunks

    # estimate optimal chunks
    chunks = []
    current_chunk_run_time = 0
    chunk_start = 1
    for test_number, expected_proc_time in enumerate(
        expected_test_processor_times, start=1
    ):
        current_chunk_run_time += expected_proc_time
        if current_chunk_run_time > average_processor_time:
            chunks.append((chunk_start, test_number, current_chunk_run_time))
            chunk_start = test_number + 1
            current_chunk_run_time = 0

    chunks.append(
        (chunk_start, len(expected_test_processor_times), current_chunk_run_time)
    )

    print("[", end="")
    for chunk in chunks[:-1]:
        print(f'"{chunk[0]},{chunk[1]}"', end=",")
    print(f'"{chunks[-1][0]},"]')


if __name__ == "__main__":
    main()
