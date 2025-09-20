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
import sys
import numpy as np
import yaml
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


def compare_nested_dicts_or_lists(
    obj,
    reference_obj,
    rtol,
    atol,
) -> bool:
    """Recursively compare two nested dictionaries or lists.

    In case objects are not within the provided tolerance an `AssertionError` is raised.

    Args:
        obj: Object for comparison
        reference_obj: Reference object
        rtol: The relative tolerance parameter for numpy.isclose
        atol: The absolute tolerance parameter for numpy.isclose

    Returns:
        True if the dictionaries are equal
    """

    # Ensures the types are the same
    if not type(obj) is type(reference_obj):
        if not isinstance(obj, (float, int)) or not isinstance(
            reference_obj, (float, int)
        ):
            raise AssertionError(
                f"Object is of type {type(obj)}, but the reference is of type {type(reference_obj)}"
            )

    # Objects are numerics
    if isinstance(obj, (float, int)):
        if not np.isclose(obj, reference_obj, rtol, atol):
            raise AssertionError(
                f"The numerics are not close:\n\nobj = {obj}\n\nreference_obj = {reference_obj}"
            )
        return True

    # Object are dicts
    if isinstance(obj, dict):
        # ^ is the symmetric difference operator, i.e. union of the sets without the intersection
        if missing_keys := set(obj.keys()) ^ set(reference_obj.keys()):
            raise AssertionError(
                f"The following keys could not be found in both dicts {missing_keys}:"
                f"\nobj: {obj}\n\nreference_obj:{reference_obj}"
            )
        for key in obj:
            compare_nested_dicts_or_lists(
                obj[key],
                reference_obj[key],
                rtol,
                atol,
            )
        return True

    # Objects are lists
    if isinstance(obj, list):
        if len(obj) != len(reference_obj):
            raise AssertionError(
                f"The list lengths differ (got {len(obj)} and {len(reference_obj)}).\nobj "
                f"{obj}\n\nreference_obj:{reference_obj}"
            )
        for obj_item, reference_obj_item in zip(obj, reference_obj):
            compare_nested_dicts_or_lists(
                obj_item,
                reference_obj_item,
                rtol,
                atol,
            )
        return True

    # Otherwise compare the objects directly
    if obj != reference_obj:
        raise AssertionError(
            f"The objects are not equal:\n\nobj = {obj}\n\nreference_obj = {reference_obj}"
        )

    return True


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

    _, file_ending_a = os.path.splitext(file_a)
    _, file_ending_b = os.path.splitext(file_b)

    if file_ending_a != file_ending_b:
        raise ValueError(
            "You are trying to compare files with different file endings: {} and {}.".format(
                file_ending_a, file_ending_b
            )
        )

    if file_ending_a == ".csv":
        # Load each file as a real array.
        data_a = read_csv(file_a)
        data_b = read_csv(file_b)

        # Compare the data values.
        if np.allclose(data_a, data_b, rtol=r_tol, atol=a_tol):
            print("CSV comparison successful!")
        else:
            raise ValueError("CSV comparison failed!")

    elif file_ending_a == ".yaml":
        data_a = []
        data_b = []
        with open(file_a, "r") as f:
            data_a = yaml.safe_load(f)

        with open(file_b, "r") as f:
            data_b = yaml.safe_load(f)

        compare_nested_dicts_or_lists(data_a, data_b, r_tol, a_tol)

    else:
        raise ValueError("File ending not recognized!")
