// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_vtu_reader.hpp"

#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_io_element_vtk_cell_type_register.hpp"
#include "4C_io_mesh.hpp"
#include "4C_utils_exceptions.hpp"

#ifdef FOUR_C_WITH_VTK
#include <vtkArrayDispatch.h>
#include <vtkCell.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkIdList.h>
#include <vtkLongLongArray.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#include <vtkType.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkXMLUnstructuredGridWriter.h>
#endif


FOUR_C_NAMESPACE_OPEN

#ifdef FOUR_C_WITH_VTK
namespace
{
  // Returns a reference to the specified array in the given vtkDataArrayCollection
  vtkDataArray& get_array(auto* data, const std::string& name)
  {
    vtkDataArray* data_array = data->GetArray(name.c_str());
    FOUR_C_ASSERT_ALWAYS(data_array,
        "Array {} not found!\n\n4C requires:\n\n * Zero or more integer-type point-arrays "
        "'point_set_#' that define the points in a set with id #. A point is in the set if the "
        "respective value is not zero.\n * Exactly one integer-typed cell-array `block_id` that "
        "defines blocks of the mesh. Each block can only contain cells of the same type.",
        name);

    return *data_array;
  }

  // Extract the value of type @p T from the given vtkDataArray
  template <typename T>
  T extract_component_from_integral_array(vtkDataArray& array, vtkIdType tupleIdx, int compIdx = 0)
  {
    T result;

    // Dispatch only over integer arrays
    if (!vtkArrayDispatch::DispatchByValueType<vtkArrayDispatch::Integrals>::Execute(&array,
            [&](auto* typedArray)
            {
              using ArrayT = std::decay_t<decltype(*typedArray)>;
              using ValueT = typename ArrayT::ValueType;
              static_assert(std::is_integral_v<ValueT>, "Expecting only integer arrays");

              ValueT value = typedArray->GetComponent(tupleIdx, compIdx);
              result = static_cast<T>(value);
            }))
    {
      FOUR_C_THROW("Array {} is of type {}, but expecting an integer-type array!", array.GetName(),
          array.GetDataTypeAsString());
    }

    return result;
  }

  // Returns a map of all numbered arrays with a specific prefix (e.g. "point_set_1", "point_set_2")
  std::map<int, std::reference_wrapper<vtkDataArray>> get_numbered_arrays_with_prefix(
      auto* data, const std::string& prefix)
  {
    std::map<int, std::reference_wrapper<vtkDataArray>> arrays;

    int numArrays = data->GetNumberOfArrays();
    for (int i = 0; i < numArrays; ++i)
    {
      const char* name = data->GetArrayName(i);
      if (name)
      {
        std::string_view name_str(name);

        if (name_str.size() > prefix.size() + 1 && name_str.starts_with(prefix + "_") &&
            name_str.substr(prefix.size() + 1).find_first_not_of("0123456789") ==
                std::string_view::npos)
        {
          int set_id = std::stoi(name_str.substr(prefix.size() + 1).data());

          arrays.emplace(set_id, *data->GetArray(name));
        }
      }
    }

    return arrays;
  }

  // Translates from vtk cell connectivity ordering to 4C connectivity ordering
  std::vector<int> translate_vtk_connectivity(
      Core::FE::CellType cell_type, std::span<const vtkIdType> vtk_connectivity)
  {
    return Core::FE::cell_type_switch<Core::IO::VTKSupportedCellTypes>(cell_type,
        [&](auto celltype_t)
        {
          std::vector<int> four_c_connectivity(vtk_connectivity.size(), 0);

          for (std::size_t i = 0; i < vtk_connectivity.size(); ++i)
          {
            four_c_connectivity[i] =
                vtk_connectivity[Core::IO::vtk_connectivity_reverse_mapping<celltype_t()>[i]];
          }

          return four_c_connectivity;
        });
  }
}  // namespace

Core::IO::MeshInput::Mesh<3> Core::IO::VTU::read_vtu_file(const std::filesystem::path& vtu_file)
{
  FOUR_C_ASSERT_ALWAYS(
      std::filesystem::exists(vtu_file), "File {} does not exist.", vtu_file.string());

  Core::IO::MeshInput::Mesh<3> mesh{};

  // Read the VTU file
  vtkSmartPointer<vtkXMLUnstructuredGridReader> reader =
      vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
  reader->SetFileName(vtu_file.c_str());
  reader->Update();
  vtkUnstructuredGrid* vtk_mesh = reader->GetOutput();

  // first read the points of the mesh and the point-sets
  auto point_sets = get_numbered_arrays_with_prefix(vtk_mesh->GetPointData(), "point_set");
  mesh.points.resize(vtk_mesh->GetNumberOfPoints());
  for (vtkIdType i = 0; i < vtk_mesh->GetNumberOfPoints(); ++i)
  {
    vtk_mesh->GetPoint(i, mesh.points[static_cast<int>(i)].coords.data());
    mesh.points[static_cast<int>(i)].external_id = static_cast<int>(i);

    // check whether this point is part of a point-set
    for (const auto& [set_id, array_ref] : point_sets)
    {
      bool is_part_of_point_set =
          extract_component_from_integral_array<bool>(array_ref.get(), i, 0);

      if (is_part_of_point_set)
      {
        mesh.point_sets[set_id].point_ids.emplace(i);
      }
    }
  }

  // now read the cell and their blocks
  vtkDataArray& cell_block_info = get_array(vtk_mesh->GetCellData(), "block_id");

  for (vtkIdType i = 0; i < vtk_mesh->GetNumberOfCells(); i++)
  {
    const int block_id = extract_component_from_integral_array<int>(cell_block_info, i, 0);

    const auto cell_type = get_celltype_from_vtk(vtk_mesh->GetCellType(i));

    // Try to emplace the cell block into the mesh
    const auto [emplaced_item, inserted] =
        mesh.cell_blocks.try_emplace(block_id, MeshInput::CellBlock{cell_type});

    MeshInput::CellBlock& cell_block = emplaced_item->second;

    FOUR_C_ASSERT_ALWAYS(emplaced_item->second.cell_type == cell_type,
        "Cell block {} has mixed cell types: {} and {}.", block_id,
        Core::FE::cell_type_to_string(emplaced_item->second.cell_type),
        Core::FE::cell_type_to_string(cell_type));

    // extract connectivity (note that we need to adapt the node-ordering according to our
    // convention)
    vtkIdType number_of_points;
    const vtkIdType* connectivity = nullptr;
    vtk_mesh->GetCellPoints(i, number_of_points, connectivity);

    // add to cell_block
    cell_block.add_cell(
        i, translate_vtk_connectivity(
               cell_type, std::span{connectivity, static_cast<std::size_t>(number_of_points)}));
  }

  return mesh;
}

#else
Core::IO::MeshInput::Mesh<3> Core::IO::VTU::read_vtu_file(const std::filesystem::path& vtu_file)
{
  FOUR_C_THROW(
      "You have to enable VTK to support vtu mesh file input. Reconfigure 4C with the CMake option "
      "'FOUR_C_WITH_VTK' set to 'ON'.");
}
#endif

FOUR_C_NAMESPACE_CLOSE
