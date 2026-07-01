# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import argparse
import sys
import yaml


def main():
    parser = argparse.ArgumentParser(
        description="Check timings for four_c_building_testing."
    )
    parser.add_argument("timings_file", type=str, help="Path to the timings file.")
    parser.add_argument(
        "expected_timers",
        type=str,
        nargs="+",
        help="List of expected timers to check for.",
    )
    parser.add_argument(
        "--expected-max-time",
        type=float,
        nargs="*",
        default=[],
        help="Expected maximum time for each timer.",
    )
    parser.add_argument(
        "--expected-max-num-calls",
        type=int,
        nargs="*",
        default=[],
        help="Expected maximum number of calls for each timer.",
    )
    parser.add_argument(
        "--expected-min-num-calls",
        type=int,
        nargs="*",
        default=[],
        help="Expected minimum number of calls for each timer.",
    )
    args = parser.parse_args()

    assert len(args.expected_max_time) == 0 or len(args.expected_max_time) == len(
        args.expected_timers
    ), "Expected max time list must be empty or match the length of expected timers"
    assert len(args.expected_max_num_calls) == 0 or len(
        args.expected_max_num_calls
    ) == len(
        args.expected_timers
    ), "Expected max num calls list must be empty or match the length of expected timers"
    assert len(args.expected_min_num_calls) == 0 or len(
        args.expected_min_num_calls
    ) == len(
        args.expected_timers
    ), "Expected min num calls list must be empty or match the length of expected timers"

    with open(args.timings_file, "r") as f:
        timings = yaml.safe_load(f)

    max_times = (
        args.expected_max_time
        if len(args.expected_max_time) > 0
        else [sys.float_info.max] * len(args.expected_timers)
    )
    max_num_calls = (
        args.expected_max_num_calls
        if len(args.expected_max_num_calls) > 0
        else [sys.maxsize] * len(args.expected_timers)
    )
    min_num_calls = (
        args.expected_min_num_calls
        if len(args.expected_min_num_calls) > 0
        else [0] * len(args.expected_timers)
    )

    assert "MinOverProcs" in timings["Statistics collected"]
    assert "MeanOverProcs" in timings["Statistics collected"]
    assert "MaxOverProcs" in timings["Statistics collected"]
    assert "MeanOverCallCounts" in timings["Statistics collected"]

    for timer, max_time, max_calls, min_calls in zip(
        args.expected_timers, max_times, max_num_calls, min_num_calls
    ):
        print(
            f"Checking timer '{timer}' with expected max time {max_time}, expected max num calls {max_calls}, and expected min num calls {min_calls}..."
        )
        assert (
            timer in timings["Timer names"]
        ), f"Timer '{timer}' not found in timings file. Timers found: {timings['Timer names']}"

        assert (
            timings["Total times"][timer]["MinOverProcs"] <= max_time
        ), f"Timer '{timer}' has MinOverProcs time {timings['Total times'][timer]['MinOverProcs']} which exceeds the expected max time {max_time}."
        assert (
            timings["Total times"][timer]["MeanOverProcs"] <= max_time
        ), f"Timer '{timer}' has MeanOverProcs time {timings['Total times'][timer]['MeanOverProcs']} which exceeds the expected max time {max_time}."
        assert (
            timings["Total times"][timer]["MaxOverProcs"] <= max_time
        ), f"Timer '{timer}' has MaxOverProcs time {timings['Total times'][timer]['MaxOverProcs']} which exceeds the expected max time {max_time}."
        assert (
            timings["Total times"][timer]["MeanOverCallCounts"] <= max_time
        ), f"Timer '{timer}' has MeanOverCallCounts time {timings['Total times'][timer]['MeanOverCallCounts']} which exceeds the expected mean time per call count {max_time} / {timings['Total times'][timer]['Total calls']} = {max_time / timings['Total times'][timer]['Total calls']}."

        assert (
            timings["Call counts"][timer]["MinOverProcs"] <= max_calls
        ), f"Timer '{timer}' has MinOverProcs call count {timings['Call counts'][timer]['MinOverProcs']} which exceeds the expected max call count {max_calls}."
        assert (
            timings["Call counts"][timer]["MeanOverProcs"] <= max_calls
        ), f"Timer '{timer}' has MeanOverProcs call count {timings['Call counts'][timer]['MeanOverProcs']} which exceeds the expected max call count {max_calls}."
        assert (
            timings["Call counts"][timer]["MaxOverProcs"] <= max_calls
        ), f"Timer '{timer}' has MaxOverProcs call count {timings['Call counts'][timer]['MaxOverProcs']} which exceeds the expected max call count {max_calls}."
        assert (
            timings["Call counts"][timer]["MinOverProcs"] >= min_calls
        ), f"Timer '{timer}' has MinOverProcs call count {timings['Call counts'][timer]['MinOverProcs']} which is below the expected min call count {min_calls}."
        assert (
            timings["Call counts"][timer]["MeanOverProcs"] >= min_calls
        ), f"Timer '{timer}' has MeanOverProcs call count {timings['Call counts'][timer]['MeanOverProcs']} which is below the expected min call count {min_calls}."
        assert (
            timings["Call counts"][timer]["MaxOverProcs"] >= min_calls
        ), f"Timer '{timer}' has MaxOverProcs call count {timings['Call counts'][timer]['MaxOverProcs']} which is below the expected min call count {min_calls}."


if __name__ == "__main__":
    main()
