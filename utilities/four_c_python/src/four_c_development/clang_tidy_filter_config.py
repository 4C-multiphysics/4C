# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import yaml
import argparse


def main():
    args = argparse.ArgumentParser(
        prog="filter_config.py",
        description="Filters the .clang-tidy file to only include the checks that are enforced",
    )

    args.add_argument("in_file", help="The path to the .clang-tidy file")
    args.add_argument("out_file", help="The path to the output .clang-tidy file")

    args = args.parse_args()

    with open(args.in_file, "r") as f:
        clang_tidy_options = yaml.safe_load(f)

    # deactivate all checks except the ones that are enforced
    filtered_options = {
        "Checks": "-*, " + clang_tidy_options["WarningsAsErrors"],
        "CheckOptions": clang_tidy_options["CheckOptions"],
        "WarningsAsErrors": "*",
    }

    with open(args.out_file, "w") as f:
        yaml.dump(filtered_options, f)


if __name__ == "__main__":
    main()
