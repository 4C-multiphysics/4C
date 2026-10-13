# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Import python modules.
import os
import sys
import numpy as np
import xml.etree.ElementTree as ET
from vtk import vtkXMLPPolyDataReader
from vtk import vtkXMLPUnstructuredGridReader
from vtk import vtkXMLUnstructuredGridReader
from vtk import vtkXMLGenericDataObjectReader
from vtk.util import numpy_support as VN
import argparse

# Import pure vtk data comparison.
from four_c_building_testing.vtk_data_compare import compare_vtk_data

# we expect the following input:
# link_to_python link_to_this_script input_pvd_file reference_pvd_file tolerance_for_data_comparison number_of_timesteps[optional] list_of_points_in_time[optional]
# /home/user/anaconda3/envs/vtk-test/bin/python /home/user/sim/vtk-tests/python/vtk_compare.py /home/user/sim/vtk-tests/sohex8/xxx-structure.pvd /home/user/sim/vtk-tests/sohex8/sohex8fbar_cooks_nl_new_struc-structure.pvd 1e-8 3 10.0 35.0 100.0
# Supports both vtu_per_rank output (.pvd + .pvtu master + per-rank .vtu files) and
# vtu_collective output (.pvd + single shared multi-piece .vtu per time step).


def compare_vtk(path1, path2, points_in_time, tol_float=1e-8, raise_error=True):
    """
    Compare the vtk files at path1 and path2.

    Args
    ----
    raise_error: bool
        If true, then an error will be raised in case the files do not match. User can read it in verbose ctest.
        Otherwise False will be returned.
    tol_float: float
        If given, numbers will be considered equal if the difference between
        them is smaller than tol_float.
    timesteps: float list
        Only given timesteps will be compared. If empty all timesteps are compared.
    """

    # Check that both arguments are paths and .pvd files exist.
    if not (os.path.isfile(path1) and os.path.isfile(path2)):
        raise ValueError("The .pvd paths given are not OK!")

    def compare_pvd(path_comp, path_ref):
        """
        check content of pvd files, exclude file links
        """
        comp_data = ET.parse(path_comp)
        ref_data = ET.parse(path_ref)

        comp_root = comp_data.getroot()
        ref_root = ref_data.getroot()

        num_collections = len(comp_root.findall("Collection"))
        num_collections_ref = len(ref_root.findall("Collection"))

        if num_collections != num_collections_ref:
            raise ValueError(
                f"Number of Collections in PVD file differ! {num_collections} != {num_collections_ref}"
            )

        for collection, collection_ref in zip(
            comp_root.findall("Collection"), ref_root.findall("Collection")
        ):
            num_datasets = len(collection.findall("DataSet"))
            num_datasets_ref = len(collection_ref.findall("DataSet"))
            if num_datasets != num_datasets_ref:
                raise ValueError(
                    f"Number of DataSets in Collections of PVD file differ! {num_datasets} != {num_datasets_ref}"
                )

        # remove file attrib to compare the rest
        for collection in comp_root.findall("Collection"):
            for dataset in collection.findall("DataSet"):
                dataset.attrib.pop("file")

        for collection in ref_root.findall("Collection"):
            for dataset in collection.findall("DataSet"):
                dataset.attrib.pop("file")

        # Check that both etrees are the same except from file links
        if not (ET.tostring(comp_root) == ET.tostring(ref_root)):
            raise ValueError(
                f"XML structures in PVD files differ!\n\n{ET.tostring(comp_root).decode()}\n\nvs.\n\n{ET.tostring(ref_root).decode()}"
            )

    def find_pvtk(pvdpath, points_in_time):
        """
        Find list of master data files (.pvtu for vtu_per_rank, .vtu for vtu_collective)
        referenced by the given .pvd file, filtered to the given points in time
        """
        mydata = ET.parse(pvdpath)

        filearray = []

        # find relative paths to pvtk files
        for type_tag in mydata.findall("Collection/DataSet"):
            stepfile = type_tag.get("file")
            timepoint = float(type_tag.get("timestep"))
            # only use timestep if it is close to values in given list or all are used
            if (
                np.isclose(
                    timepoint,
                    points_in_time,
                    atol=1e-10,
                    rtol=0.0,
                ).any()
            ) or (len(points_in_time) == 0):
                filearray.append(stepfile)

        # if something did not go as intended
        if (len(points_in_time) != len(filearray)) and not (len(points_in_time) == 0):
            raise ValueError(
                "Number of time steps given does not match number of files found! Check input or adjust tolerance."
            )

        return filearray

    def compare_pvtk(pvtkpath_comp, pvtkpath_ref):
        """
        compare file content of pvtk files
        """
        # Check that both .pvtk files exist
        if not (os.path.isfile(pvtkpath_comp) and os.path.isfile(pvtkpath_ref)):
            raise ValueError("The .pvtk paths given are not OK!")

        comp_data = ET.parse(pvtkpath_comp)
        ref_data = ET.parse(pvtkpath_ref)

        comp_root = comp_data.getroot()
        ref_root = ref_data.getroot()

        # remove piece elements (links) to compare the rest
        for child in comp_root:
            for grandchild in child.findall("Piece"):
                child.remove(grandchild)

        for child in ref_root:
            for grandchild in child.findall("Piece"):
                child.remove(grandchild)

        # Check that both etrees are the same except from file links
        if not (ET.tostring(comp_root) == ET.tostring(ref_root)):
            raise ValueError("XML structures in PVTK files differ!")

    def merge_vtk(pvtkpath):
        """
        Read a parallel master file (.pvtu, vtu_per_rank) or a single shared multi-piece file
        (.vtu, vtu_collective) and return the merged unstructured grid
        """

        is_parallel_master = os.path.splitext(pvtkpath)[1] == ".pvtu"

        if is_parallel_master:
            # find all desired vtk files and check if they exist
            vtkfiles = find_vtk_and_check(pvtkpath)
            type_probe_file = os.path.join(os.path.dirname(pvtkpath), vtkfiles[0])
        else:
            type_probe_file = pvtkpath

        # read the probe file once to determine its data type
        type_reader = vtkXMLGenericDataObjectReader()
        type_reader.SetFileName(type_probe_file)
        type_reader.Update()
        type = type_reader.GetOutput().GetDataObjectType()

        # use the respective reader for the vtk types used in 4C
        if type == 4:
            if is_parallel_master:
                preader = vtkXMLPUnstructuredGridReader()
            else:
                preader = vtkXMLUnstructuredGridReader()
            preader.SetFileName(pvtkpath)
            preader.Update()
        else:
            raise ValueError("Unknown VTK result type. known types: UnstructuredGrid")

        # A matching sorting could be introduced here to get rid of dependency on number of processors

        return preader.GetOutput()

    def find_vtk_and_check(pvtkpath):
        """
        get list of every vtk file from every processor
        """

        mydata = ET.parse(pvtkpath)

        filearray = []

        for type_attrib in mydata.findall("*/Piece"):
            procfile = type_attrib.get("Source")
            # check if file exists
            if not (os.path.isfile(os.path.join(os.path.dirname(pvtkpath), procfile))):
                raise ValueError("The .vtk paths given are not OK!")
            filearray.append(procfile)

        return filearray

    # Perform all checks, catch errors.
    try:
        # compare content of .pvd files excluding filepaths
        compare_pvd(path_comp=path1, path_ref=path2)

        dir1 = os.path.dirname(path1)
        dir2 = os.path.dirname(path2)

        # create list of step files from .pvd file content (master .pvtu for vtu_per_rank,
        # single shared multi-piece .vtu for vtu_collective)
        stepfiles1 = find_pvtk(pvdpath=path1, points_in_time=points_in_time)
        stepfiles2 = find_pvtk(pvdpath=path2, points_in_time=points_in_time)

        # Load the vtk files.
        data1array = []
        data2array = []

        # For the parallel master layout (.pvtu), compare the master XML (minus <Piece> links).
        # A collective .vtu has no separate master file, so this step is skipped. Only compare the
        # XML structure when both step files are parallel masters; a mixed layout (.pvtu vs. .vtu)
        # compares data via merge_vtk below.
        for i in range(0, len(stepfiles1)):
            if (
                os.path.splitext(stepfiles1[i])[1] == ".pvtu"
                and os.path.splitext(stepfiles2[i])[1] == ".pvtu"
            ):
                compare_pvtk(
                    pvtkpath_comp=os.path.join(dir1, stepfiles1[i]),
                    pvtkpath_ref=os.path.join(dir2, stepfiles2[i]),
                )

        for iter in enumerate(stepfiles1):
            data1array.append(merge_vtk(os.path.join(dir1, iter[1])))

        for iter in enumerate(stepfiles2):
            data2array.append(merge_vtk(os.path.join(dir2, iter[1])))

        for i in range(0, len(data1array)):
            compare_vtk_data(data1array[i], data2array[i], tol_float=tol_float)

    except Exception as error:
        if raise_error:
            raise error
        return False

    return True


def cli():
    parser = argparse.ArgumentParser(
        description="Compare two .pvd files and their linked .pvtu/.vtu data files. Supports both vtu_per_rank (.pvtu master + per-rank .vtu) and vtu_collective (.vtu) layouts. Only given points in time are compared. If no points in time are given, all are compared."
    )
    parser.add_argument(
        "vtk_result", help="Path to .pvd file of the result to be compared."
    )
    parser.add_argument(
        "vtk_reference",
        help="Path to .pvd file of the reference to be compared against.",
    )
    parser.add_argument(
        "tolerance", help="Tolerance for data comparison, e.g. 1e-8.", type=float
    )
    parser.add_argument(
        "--points_in_time",
        nargs="*",
        help="List of points in time to be compared (If empty, all timesteps are compared).",
        type=float,
        default=[],
    )

    args = parser.parse_args()
    # Read arguments.
    file_comp = args.vtk_result
    file_ref = args.vtk_reference
    tolerance = args.tolerance
    points_in_time = args.points_in_time
    compare_vtk(
        path1=file_comp,
        path2=file_ref,
        points_in_time=points_in_time,
        tol_float=tolerance,
    )

    print(
        "SUCCESS: VTK results match for given .pvd files, points in time and tolerance"
    )


if __name__ == "__main__":
    cli()
