#!/usr/bin/env python3
# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check that all Python files are located within utilities/four_c_python."""

import sys


def main():
    """Check that all Python files are located within utilities/four_c_python."""

    wrong_files = [
        file for file in sys.argv[1:] if "utilities/four_c_python/" not in file
    ]

    if wrong_files:
        print("The following Python files are outside 'utilities/four_c_python':\n")
        for file in wrong_files:
            print(f"    {file}")

        print(
            "\nPlease move these files into the 'utilities/four_c_python' directory.\n"
            "This ensures all Python utilities are centralized in one place."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
