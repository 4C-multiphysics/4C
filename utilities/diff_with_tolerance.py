# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# -*- coding: utf-8 -*-
"""
This script compares two files with a tolerance.
"""

# Import python modules.
import os
import numpy as np
import argparse


def convert_line_to_array(line):
    """
    Try to convert a string to an array with floats.
    """

    # Split up the line.
    if "," in line:
        line = line.split(",")
    else:
        line = line.split()
    line = [item.strip() for item in line]

    # Try to parse all parts of the line to a float.
    line_array = []
    try:
        for item in line:
            line_array.append(float(item))
    except ValueError:
        return False, None

    # All floats could be converted. We still have to check if the array is empty.
    if len(line_array) == 0:
        return False, None
    else:
        return True, line_array


def read_csv(path):
    """
    Load a csv file as a numpy array.
    """
    if not os.path.isfile(path):
        raise ValueError("The file {} does not exist.".format(path))

    # Load the lines in the file.
    with open(path, "r") as f:
        lines = f.readlines()

    # Go through each line an try to convert it to float.
    data = []
    for line in lines:
        is_float_line, line_array = convert_line_to_array(line)
        if is_float_line:
            data.append(line_array)

    return np.array(data)


if __name__ == "__main__":
    """
    Execution part of script.
    """

    parser = argparse.ArgumentParser(
        description="Compare csv files with absolute and relative tolerances"
    )
    parser.add_argument("file_a", type=str)
    parser.add_argument("file_b", type=str)
    parser.add_argument("r_tol", type=float, help="Relative tolerance")
    parser.add_argument("a_tol", type=float, help="Absolute tolerance")
    args = parser.parse_args()

    file_a = args.file_a
    file_b = args.file_b
    r_tol = args.r_tol
    a_tol = args.a_tol

    # Load each file as a real array.
    data_a = read_csv(file_a)
    data_b = read_csv(file_b)

    # Compare the data values.
    if np.allclose(data_a, data_b, rtol=r_tol, atol=a_tol):
        print("CSV comparison successful!")
    else:
        raise ValueError("CSV comparison failed!")
