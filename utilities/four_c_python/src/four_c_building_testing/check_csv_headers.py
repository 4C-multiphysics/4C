# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Check that a CSV file contains the expected header row."""

import argparse
import csv
import os


def read_header(path):
    """Read and return the first row of a CSV file."""
    if not os.path.isfile(path):
        raise ValueError(f"The file {path} does not exist.")

    with open(path, "r", newline="") as csv_file:
        reader = csv.reader(csv_file)
        try:
            return next(reader)
        except StopIteration as exc:
            raise ValueError(f"The file {path} is empty.") from exc


def check_csv_headers(path, expected_headers):
    """Check that the first CSV row matches the expected header list."""
    actual_headers = read_header(path)
    if actual_headers != expected_headers:
        raise AssertionError(
            "CSV header comparison failed:\n"
            f"expected: {expected_headers}\n"
            f"actual:   {actual_headers}"
        )


def cli():
    """Execution part of script."""
    parser = argparse.ArgumentParser(description="Check the headers of a CSV file")
    parser.add_argument("csv_file", type=str)
    parser.add_argument("headers", nargs="+", help="Expected header strings")
    args = parser.parse_args()

    check_csv_headers(args.csv_file, args.headers)
    print("CSV header comparison successful!")


if __name__ == "__main__":
    cli()
