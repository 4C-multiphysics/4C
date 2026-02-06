# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
This script checks that the restart step is correctly recorded in an output.control file.
"""

import argparse
import sys
from four_c_common_utils.io import load_yaml


def cli():
    """
    Main execution function.
    """
    parser = argparse.ArgumentParser(
        description="Check that the restart step is correctly recorded in an output.control file (YAML format)."
    )
    parser.add_argument(
        "control_file",
        type=str,
        help="Path to the output.control file from the restarted test run",
    )
    parser.add_argument(
        "expected_restart_step",
        type=int,
        help="Expected restart step number",
    )

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Checking restart step in output.control file")
    print(f"{'='*70}")
    print(f"File: {args.control_file}")
    print(f"Expected restart step: {args.expected_restart_step}")
    print(f"{'='*70}\n")

    # Load control file as YAML file
    try:
        data = load_yaml(args.control_file)
    except Exception as e:
        print(f"ERROR: Failed to load YAML file: {e}")
        sys.exit(1)

    # Check that data is a list
    if not isinstance(data, list):
        print(f"ERROR: Expected data to be a list, got {type(data)}")
        sys.exit(1)

    # Check that we have at least 2 elements (index 0 and 1)
    if len(data) < 2:
        print(
            f"ERROR: Control file has only {len(data)} element(s), expected at least 2"
        )
        sys.exit(1)

    # Get the general section at index 1
    general_section = data[1]
    if not isinstance(general_section, dict):
        print(
            f"ERROR: Expected element at index 1 to be a dict, got {type(general_section)}"
        )
        sys.exit(1)

    if "general" not in general_section:
        print(f"ERROR: 'general' key not found in element at index 1")
        print(f"Available keys: {list(general_section.keys())}")
        sys.exit(1)

    general = general_section["general"]
    if not isinstance(general, dict):
        print(f"ERROR: Expected 'general' to be a dict, got {type(general)}")
        sys.exit(1)

    # Check for restarted_from_step
    if "restarted_from_step" not in general:
        print(f"ERROR: 'restarted_from_step' key not found in general section")
        print(f"Available keys: {list(general.keys())}")
        sys.exit(1)

    actual_restart_step = general["restarted_from_step"]

    # Compare
    if actual_restart_step == args.expected_restart_step:
        print(f"[PASS] Restart step matches: {actual_restart_step}")
        print(f"{'='*70}\n")
        sys.exit(0)
    else:
        print(f"[FAIL] Restart step mismatch!")
        print(f"  Expected: {args.expected_restart_step}")
        print(f"  Actual:   {actual_restart_step}")
        print(f"{'='*70}\n")
        sys.exit(1)


if __name__ == "__main__":
    cli()
