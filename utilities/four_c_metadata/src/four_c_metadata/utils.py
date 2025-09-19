# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Utils."""

import ryml
import pathlib
import json
import re


def load_yaml(path_to_yaml_file: str | pathlib.Path) -> dict:
    """
    Duplicated from fourcipp.
    """

    json_str = ryml.emit_json(
        ryml.parse_in_arena(pathlib.Path(path_to_yaml_file).read_bytes())
    )

    # Convert `inf` to a string to avoid JSON parsing errors, see https://github.com/biojppm/rapidyaml/issues/312
    json_str = re.sub(r":\s*(-?)inf\b", r': "\1inf"', json_str)
    # Convert floats that are missing digits on either side of the decimal point
    json_str = re.sub(r":\s*(-?)\.([0-9]+)", r": \g<1>0.\2", json_str)
    json_str = re.sub(r":\s*(-?)([0-9]+)\.(\D)", r": \1\2.0\3", json_str)

    data = json.loads(json_str)

    return data
