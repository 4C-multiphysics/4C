# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check that C++ files have correct includes"""

import re
import sys

from pathlib import Path
from typing import Tuple

from four_c_utils.common_utils import file_contents


def replace_line_in_file(file: Path, line_no: int, line: str):
    """Replace a line in a file.

    Args:
        file: The file to replace the line in
        line_no: The line number to replace
        line: The new line
    """

    lines = file_contents(file)

    lines[line_no] = line

    with open(file, "w") as f:
        f.writelines(lines)


def check_and_fix_includes(file: Path) -> Tuple[list, bool]:
    """Check that the file has valid includes.

    If quotes are missing for internal imports fix them.

    Check that 4C files are included with quotes, external
    ones are included with angle brackets. Overall, no relative
    directory navigation (using '.' or '..') is allowed in
    includes.

    Args:
        file: The file to check

    Returns:
        List of line numbers with invalid includes, bool if
        includes were fixed
    """

    invalid_include_lines = []
    fixed_include = False

    for line_no, line in enumerate(file_contents(file)):

        if not line.startswith("#include"):
            continue

        if ".." in line or "./" in line:
            invalid_include_lines.append([line_no + 1, line])
            continue

        if "4C" in line:
            match = re.search(r"#include <(4C_\w+\.hpp)>", line)
            if match:
                new_line = f'#include "{match[1]}"\n'
                replace_line_in_file(file, line_no, new_line)
                fixed_include = True

        else:
            if not "<" in line:
                invalid_include_lines.append([line_no + 1, line])

    return invalid_include_lines, fixed_include


def main():
    """Check that C++ files have correct includes."""

    invalid_file_includes_data = {}
    fixed_file_includes_data = {}

    for file in sys.argv[1:]:
        invalid_includes, fixed_includes = check_and_fix_includes(Path(file))

        if invalid_includes:
            invalid_file_includes_data[file] = invalid_includes
        if fixed_includes:
            fixed_file_includes_data[file] = fixed_includes

    if invalid_file_includes_data:
        print("The following includes do not adhere to our include convention:\n")
        for wrong_file in invalid_file_includes_data.keys():
            for invalid_include in invalid_file_includes_data[wrong_file]:
                print(f"    * {wrong_file}:{invalid_include[0]}: {invalid_include[1]}")

        print(
            "    Please fix the includes in the files to match the naming convention.\n"
            "    A valid include must:\n"
            "        - not contain relative navigation (. or ../) \n"
            "        - include external files with angle brackets\n"
        )

    if fixed_file_includes_data:
        print("Includes from the following files have been automatically fixed:\n")
        for fixed_file in fixed_file_includes_data.keys():
            print(f"    * {fixed_file}")

        print("\n    Please review the changes and commit them.\n")

    if invalid_file_includes_data or fixed_file_includes_data:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    sys.exit(main())
