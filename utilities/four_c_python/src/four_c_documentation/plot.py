# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Tool for drawing cell types with node, line and surface numbers.
The corresponding data is taken from yaml file elementinformation.yaml,
which contains a map of cell types.
Each cell type entry consists of four named arrays:

- coords         : coordinates of the element nodes (implicitly defining the node numbers)
- lines          : a vector of vectors of node numbers defining the edges of the element
- surfaces       : a vector of vectors of node numbers defining the surfaces of the element
- surfacecorners : a vector of integers giving the number of corners of each surface

If called directly, it needs two parameters:
yamlfiledirectory : The path of the yaml file
figuredirectory   : The path where the figures are stored

Within the process of generating the documentation (`ninja documentation`),
the yaml file `elementinformation.yaml` is created by the program `create_rtd`.
Then, the function `main(yamlfiledirectory, figuredirectory)` given below is called
by the sphinx generator (`conf.py`). The directories are defined within cmake,
which adapts the conf.py accordingly.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial import ConvexHull
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from four_c_documentation.element import defineElements
from pathlib import Path, PurePath
from sys import argv, exit
import argparse
from dataclasses import dataclass


@dataclass
class PlotParameters:
    """
    Defines all parameters, which may be adjusted for the plots
    """

    tab10_cmap = mpl.colormaps["tab10"].colors
    tab20_cmap = mpl.colormaps["tab20"].colors[1::2]
    colorlist = tab10_cmap + tab20_cmap
    nodecolor: str = "black"
    nodehiddencolor: str = "grey"
    originalstructure_in_faceplot: str = "grey"
    plotmargin: float = 0.1
    fontsize: int = 18
    axesfontsize: int = 20
    shrink_factor: float = 0.4
    dpi: int = 150


def find_center_and_move(coordinates, move_percentage):
    # Helper function:
    # Find the indices of the two points that are furthest away from each other
    indices = np.unravel_index(
        np.argmax(np.linalg.norm(coordinates[:, np.newaxis] - coordinates, axis=-1)),
        coordinates.shape,
    )

    # Calculate the line length as the distance between the two furthest points
    line_length = np.linalg.norm(coordinates[indices[0]] - coordinates[indices[1]])

    # Find the center position by calculating the mean of the coordinates along each axis
    center_position = np.mean(coordinates, axis=0)

    # Calculate the total distance from each point to the center
    distances_to_center = np.linalg.norm(coordinates - center_position, axis=1)

    # Find the index of the point that is furthest from the center
    closest_point_index = np.argmax(distances_to_center)

    # Calculate the direction vector from the center towards the closest point
    direction_vector = (
        coordinates[closest_point_index] - center_position
    ) / np.linalg.norm(coordinates[closest_point_index] - center_position)

    # Calculate the new position after moving 10% towards one end
    new_position = center_position + move_percentage * line_length * direction_vector

    return new_position


def plot2D_lines_and_nodes(ax, lines, coords, parameter):
    # Generates a 2D plot of a 2D element with node and line numbers
    # First the nodes and node numbers
    for i, coord in enumerate(coords):
        ax.scatter(coord[0], coord[1], color=parameter.nodecolor)
        ax.text(
            coord[0],
            coord[1],
            str(i),
            color=parameter.nodecolor,
            va="top",
            ha="left",
            fontsize=parameter.fontsize,
            bbox=dict(boxstyle="circle", facecolor="None", edgecolor="None", pad=10),
        )
    # now the lines and line numbers
    for i, line in enumerate(lines):
        line_coords = np.array([coords[node] for node in line])
        x, y = zip(*line_coords)
        ax.plot(x, y, color=parameter.colorlist[i])
        annotation_coord = find_center_and_move(line_coords, 0.2)
        ax.text(
            annotation_coord[0],
            annotation_coord[1],
            str(i),
            color=parameter.colorlist[i],
            va="center",
            ha="center",
            fontsize=parameter.fontsize,
            bbox=dict(boxstyle="circle", facecolor=(1, 1, 1, 0.7), edgecolor="None"),
        )


def plot3D_lines_and_nodes(ax, lines, coords, parameter):
    # Generates a 3D plot of a 3D element with node and line numbers
    xmax = max([c[0] for c in coords])
    ymin = min([c[1] for c in coords])
    zmax = max([c[2] for c in coords])

    for i, coord in enumerate(coords):
        nodecolor = (
            parameter.nodecolor
            if (coord[0] == xmax or coord[1] == ymin or coord[2] == zmax)
            else parameter.nodehiddencolor
        )
        ax.scatter(coord[0], coord[1], coord[2], color=nodecolor)
        ax.text(
            coord[0],
            coord[1],
            coord[2] - 0.05,
            str(i),
            color=nodecolor,
            va="top",
            ha="left",
            fontsize=parameter.fontsize,
            bbox=dict(boxstyle="circle", facecolor="None", edgecolor="None", pad=10),
        )

    for i, line in enumerate(lines):
        line_coords = np.array([coords[node] for node in line])
        x, y, z = zip(*line_coords)
        ax.plot(x, y, z, color=parameter.colorlist[i])
        annotation_coord = find_center_and_move(line_coords, 0.2)
        ax.text(
            annotation_coord[0],
            annotation_coord[1],
            annotation_coord[2],
            str(i),
            color=parameter.colorlist[i],
            va="center",
            ha="center",
            fontsize=parameter.fontsize,
            bbox=dict(boxstyle="circle", facecolor=(1, 1, 1, 0.7), edgecolor="None"),
        )


def plot3D_surfaces(ax, surfaces, coords, parameter):
    # Generates a plot of a 3D element with extra surface
    for k, surf in enumerate(surfaces):
        surf_coords = np.array([coords[node] for node in surf])
        surf_center_coord = np.mean(surf_coords, axis=0)
        # in order to be independent of the node numbering, we create a convex hull
        # around the surface nodes.
        # This must be done in 2D, so we simply skip the coordinate with the least difference
        # (all surfaces are straight)
        min_diff_coordinates = np.argmin(
            np.max(surf_coords, axis=0) - np.min(surf_coords, axis=0)
        )
        dims = [0, 1, 2]
        dims.remove(min_diff_coordinates)
        coords2d = np.array(surf_coords[:, dims])
        hull = ConvexHull(coords2d)
        verts = [surf_coords[i] for i in hull.vertices]

        # draw original cell structure
        ax.add_collection3d(
            Poly3DCollection(
                [verts],
                edgecolor=parameter.originalstructure_in_faceplot,
                facecolor=(1, 1, 1, 0.01),
            )
        )

        # draw shrunk faces
        shrunk_verts = [
            surf_center_coord
            + parameter.shrink_factor * (np.array(verts) - surf_center_coord)
        ]
        ax.add_collection3d(
            Poly3DCollection(
                shrunk_verts,
                edgecolor=parameter.colorlist[k],
                facecolor=(1, 1, 1, 0.01),
            )
        )
        ax.text(
            surf_center_coord[0],
            surf_center_coord[1],
            surf_center_coord[2],
            str(k),
            color=parameter.colorlist[k],
            va="center",
            ha="center",
            fontsize=parameter.fontsize,
        )


def generate_figures(element_information_file, figure_dir):
    plotpar = PlotParameters()

    with open(element_information_file, "r") as element_stream:
        element_definitions = defineElements(element_stream)

    for element_name, element_data in element_definitions.items():
        print("Creating cell type ", element_name, " plot")

        element_coords = np.array(element_data.coords)
        if element_data.dimension == 3:
            # creating the plots for 3D elements
            fig = plt.figure(figsize=(12, 6))
            ax1 = fig.add_subplot(121, projection="3d")
            ax2 = fig.add_subplot(122, projection="3d")

            plot3D_lines_and_nodes(
                ax1,
                element_data.lines,
                element_coords,
                plotpar,
            )
            plot3D_surfaces(
                ax2,
                element_data.surfaces,
                element_coords,
                plotpar,
            )
            for ax in [ax1, ax2]:
                ax.set_xlim(
                    [
                        np.min(element_coords[:, 0]) - 0.1,
                        np.max(element_coords[:, 0]) + 0.1,
                    ]
                )
                ax.set_ylim(
                    [
                        np.min(element_coords[:, 1]) - 0.1,
                        np.max(element_coords[:, 1]) + 0.1,
                    ]
                )
                ax.set_zlim(
                    [
                        np.min(element_coords[:, 2]) - 0.1,
                        np.max(element_coords[:, 2]) + 0.1,
                    ]
                )
                ax.set_xticks(element_coords[:, 0])
                ax.set_yticks(element_coords[:, 1])
                ax.set_zticks(element_coords[:, 2])
                ax.tick_params(labelsize=plotpar.axesfontsize)
                ax.grid(False)

        elif element_data.dimension == 2:
            # creating the plot for 2D elements
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111)
            plot2D_lines_and_nodes(
                ax,
                element_data.lines,
                element_coords,
                plotpar,
            )
            ax.set_xticks(element_coords[:, 0])
            ax.set_yticks(element_coords[:, 1])
            ax.tick_params(axis="both", labelsize=plotpar.axesfontsize)
            ax.grid(False)
            plt.margins(plotpar.plotmargin)

        else:
            # No plot for 0D/1D cell types yet
            pass

        plt.tight_layout()
        plt.savefig(PurePath(figure_dir, element_name + ".png"), dpi=plotpar.dpi)
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        prog="plot.py",
        description="This tool generates element line and surface plots from a yaml control file",
    )
    parser.add_argument("information_file_path", type=str)
    parser.add_argument("figures_directory", type=Path)
    args = parser.parse_args()

    yaml_control_file = Path(args.information_file_path).joinpath(
        "elementinformation.yaml"
    )
    figure_dir = args.figures_directory

    if not yaml_control_file.exists():
        print(str(yaml_control_file) + " does not exist\n")
        parser.print_help()
        exit(1)

    if not figure_dir.is_dir():
        print(str(figure_dir) + " is not a directory\n")
        parser.print_help()
        exit(1)

    generate_figures(yaml_control_file, figure_dir)


if __name__ == "__main__":
    main()
