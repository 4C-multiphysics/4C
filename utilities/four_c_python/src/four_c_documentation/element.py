# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
python include script containing the following class to be used
in the function `defineElements(filename)`,
which returns a dictionary of all elements containing the ElementData information

ElementData    : provides the geometrical information for an element
"""
import yaml
from typing import List
from dataclasses import dataclass


@dataclass
class ElementData:
    # note: using np.ndarray here is not possible, since the length is not known before
    # and can even be different per item (e.g., lines with 2 and 3 nodes)
    dimension: int = None
    coords: List[List[float]] = None
    lines: List[List[int]] = None
    surfaces: List[List[int]] = None

    def checkdefinition(self):
        if not self.lines:
            return False
        nodemax = max([n for line in self.lines for n in line])

        if nodemax >= len(self.coords):
            print("Line definition contains unknown nodes")
            return False

        if self.dimension == 3:
            if not self.surfaces:
                return False
            nodemax = max([n for surf in self.surfaces for n in surf])
            if nodemax >= len(self.coords):
                print("Surface definition contains unknown nodes")
                return False
        return True


# read the yaml file containing element information
def defineElements(element_info_stream):
    element_dict = yaml.load(element_info_stream, Loader=yaml.Loader)

    element_definitions = {}
    for elementname, elementdata in element_dict.items():
        if not elementdata:
            continue
        if "dimension" not in elementdata:
            continue
        if "nodes" not in elementdata:
            continue
        if "lines" not in elementdata:
            continue
        if elementdata["dimension"] == 3:
            if "surfaces" not in elementdata:
                continue
            else:
                element_info = ElementData(
                    elementdata["dimension"],
                    elementdata["nodes"],
                    elementdata["lines"],
                    elementdata["surfaces"],
                )
        else:
            element_info = ElementData(
                elementdata["dimension"], elementdata["nodes"], elementdata["lines"]
            )
        if not element_info.checkdefinition():
            continue

        element_definitions[elementname] = element_info

    return element_definitions
