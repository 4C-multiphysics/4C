# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check that C++ files have correct filenames"""

import sys

from pathlib import Path


def get_module_name(file: Path) -> str | None:
    """Get the corresponding module name for a file.

    The module name is determined by the parent folder
    name which is on the same level as the first
    .contains_modules file.

    Args:
        file: The file to get the module name from

    Returns:
        The module name or None if no module is found
    """

    current_dir = file.parent.absolute()

    while True:
        if current_dir == Path("/"):
            return None
        if (current_dir / ".contains_modules").is_file():
            return file.absolute().relative_to(current_dir).parts[0]
        current_dir = current_dir.parent


def check_file_name(file: Path) -> bool:
    """Check that the file has a valid filename.

    A valid filename is prefixed by '4C' and the module name.
    If it is a test file it is postfixed by '_test'.

    Args:
        file: The file to check

    Returns:
        True if the file has a valid filename, False otherwise
    """

    module_name = get_module_name(file)

    if not file.name.startswith("4C_" if module_name is None else f"4C_{module_name}"):
        return False

    if "/tests/" in str(file) or "/unittests/" in str(file):
        if not file.name.split(".")[0].endswith("_test"):
            return False

    return True


def main():
    """Check that C++ files have correct filenames."""

    wrong_file_names = []

    for file in sys.argv[1:]:
        if not check_file_name(Path(file)):
            wrong_file_names.append(file)

    if wrong_file_names:
        print("The following files do not adhere to our file naming convention:\n")
        for wrong_file in wrong_file_names:
            print(f"    {wrong_file}")

        print(
            "\nPlease rename the files to match the naming convention.\n"
            "A valid filename must:\n"
            "   - Be prefixed with '4C_' followed by the module name\n"
            "   - If it is a test file, it should end with '_test'"
        )

        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    sys.exit(main())
