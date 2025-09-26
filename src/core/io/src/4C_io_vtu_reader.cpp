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
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor_internals.hpp"
#include "4C_utils_exceptions.hpp"

#include <type_traits>

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

  std::unordered_map<std::string, std::reference_wrapper<vtkDataArray>> get_vtk_data(auto* vtk_data)
  {
    std::unordered_map<std::string, std::reference_wrapper<vtkDataArray>> data;

    int numArrays = vtk_data->GetNumberOfArrays();

    for (int i = 0; i < numArrays; ++i)
    {
      const char* name = vtk_data->GetArrayName(i);
      if (name) data.emplace(name, *vtk_data->GetArray(name));
    }
    return data;
  }


  /*!
   * @brief Given a templated container (templated for the scalar type), create an empty instance of
   * the container. Optionally reserve space for the given number of entries.
   */
  template <template <typename> typename Container>
  Core::IO::MeshInput::FieldDataVariantType<3> make_container_with_supported_scalar_type(
      vtkDataArray& array, bool reserve = true)
  {
    Core::IO::MeshInput::FieldDataVariantType<3> type;
    if (!vtkArrayDispatch::DispatchByValueType<vtkArrayDispatch::AllTypes>::Execute(&array,
            [&](auto* typed_array)
            {
              using ValueType = typename std::decay_t<decltype(*typed_array)>::ValueType;
              if constexpr (std::is_integral_v<ValueType>)
              {
                Container<int> container{};
                if (reserve) container.reserve(array.GetSize());
                type = std::move(container);
              }
              else if constexpr (std::is_floating_point_v<ValueType>)
              {
                Container<double> container{};
                if (reserve) container.reserve(array.GetSize());
                type = std::move(container);
              }
              else
              {
                FOUR_C_THROW(
                    "Array {} is of type {}. 4C currently only supports the input of integral- or "
                    "floating point types.",
                    array.GetName(), array.GetDataTypeAsString());
              }
            }))
    {
      FOUR_C_THROW("Array {} is of type {}, which is not supported!", array.GetName(),
          array.GetDataTypeAsString());
    }

    return type;
  }

  struct ScalarFieldType
  {
    template <typename T>
    using type = std::vector<T>;
  };

  template <unsigned dim>
  struct VectorFieldType
  {
    template <typename T>
    using type = std::vector<Core::LinAlg::Tensor<T, dim>>;
  };

  template <unsigned dim>
  struct SymmetricTensorFieldType
  {
    template <typename T>
    using type = std::vector<Core::LinAlg::SymmetricTensor<T, dim, dim>>;
  };

  template <unsigned dim>
  struct TensorFieldType
  {
    template <typename T>
    using type = std::vector<Core::LinAlg::Tensor<T, dim, dim>>;
  };


  /*!
   * @brief Create an empty field data variant of the appropriate type according to the number of
   * given components in the array. Optionally, the vector can already reserve space for the number
   * of entries in the array.
   */
  Core::IO::MeshInput::FieldDataVariantType<3> make_empty_field_data_variant(
      vtkDataArray& array, bool reserve = true)
  {
    const int n_components = array.GetNumberOfComponents();
    switch (n_components)
    {
      case 1:
        return make_container_with_supported_scalar_type<ScalarFieldType::type>(
            array, /*reserve=*/reserve);
      case 3:
        return make_container_with_supported_scalar_type<VectorFieldType<3>::type>(
            array, /*reserve=*/reserve);
      case 6:
        return make_container_with_supported_scalar_type<SymmetricTensorFieldType<3>::type>(
            array, /*reserve=*/reserve);
      case 9:
        return make_container_with_supported_scalar_type<TensorFieldType<3>::type>(
            array, /*reserve=*/reserve);
      default:
        FOUR_C_THROW(
            "Array {} has {} components, but can only handle 1 (scalar), 3 (vector), 6 (symmetric "
            "tensor) or 9 (tensor) "
            "components!",
            array.GetName(), n_components);
    }
  }

  template <typename T>
  T extract_data_item(vtkDataArray& array, vtkIdType index)
  {
    const int n_components = array.GetNumberOfComponents();

    auto extract_entry = [&](auto& typed_array) -> T
    {
      switch (n_components)
      {
        case 1:
        {
          // scalar
          if constexpr (std::is_arithmetic_v<T>)
          {
            return T(typed_array.GetValue(index));
          }
          FOUR_C_THROW("Implementation error: Here we expect that the types are convertible.");
        }
        case 3:
        {
          // vector
          if constexpr (Core::LinAlg::is_tensor<T>)
          {
            using ScalarType = typename T::value_type;
            if constexpr (T::rank() == 1)
            {
              return T{{ScalarType(typed_array.GetComponent(index, 0)),
                  ScalarType(typed_array.GetComponent(index, 1)),
                  ScalarType(typed_array.GetComponent(index, 2))}};
            }
          }
          FOUR_C_THROW("Implementation error: Here we expect that T is a Tensor type.");
        }
        case 6:
        {
          // symmetric tensor
          if constexpr (Core::LinAlg::is_symmetric_tensor<T>)
          {
            using ScalarType = typename T::value_type;
            if constexpr (T::rank() == 2)
            {
              std::array<ScalarType, 6> components{ScalarType(typed_array.GetComponent(index, 0)),
                  ScalarType(typed_array.GetComponent(index, 1)),
                  ScalarType(typed_array.GetComponent(index, 2)),
                  ScalarType(typed_array.GetComponent(index, 3)),
                  ScalarType(typed_array.GetComponent(index, 4)),
                  ScalarType(typed_array.GetComponent(index, 5))};
              return Core::LinAlg::make_symmetric_tensor_view<3, 3>(components.data());
            }
          }
          FOUR_C_THROW("Implementation error: Here we expect that T is a SymmetricTensor type.");
        }
        case 9:
        {
          // tensor
          if constexpr (Core::LinAlg::is_tensor<T>)
          {
            using ScalarType = typename T::value_type;
            if constexpr (T::rank() == 2 && !Core::LinAlg::is_compressed_tensor<T>)
            {
              return T{{
                  {
                      ScalarType(typed_array.GetComponent(index, 0)),
                      ScalarType(typed_array.GetComponent(index, 1)),
                      ScalarType(typed_array.GetComponent(index, 2)),
                  },
                  {
                      ScalarType(typed_array.GetComponent(index, 3)),
                      ScalarType(typed_array.GetComponent(index, 4)),
                      ScalarType(typed_array.GetComponent(index, 5)),
                  },
                  {
                      ScalarType(typed_array.GetComponent(index, 6)),
                      ScalarType(typed_array.GetComponent(index, 7)),
                      ScalarType(typed_array.GetComponent(index, 8)),
                  },
              }};
            }
          }
          FOUR_C_THROW("Implementation error: Here we expect that T is a SymmetricTensor type.");
        }
        default:
          FOUR_C_THROW(
              "Array {} has {} components, but can only handle 1 (scalar), 3 "
              "(vector), 6 (symmetric tensor) or 9 (tensor) components!",
              array.GetName(), n_components);
      }
    };

    T result;
    if (vtkArrayDispatch::DispatchByValueType<vtkArrayDispatch::AllTypes>::Execute(
            &array, [&](auto* typed_array) { result = extract_entry(*typed_array); }))
    {
      return result;
    }
    FOUR_C_THROW("Failed to extract data from array {} of type {}!", array.GetName(),
        array.GetDataTypeAsString());
  }

  // Returns a map of all numbered arrays with a specific prefix (e.g. "point_set_1",
  // "point_set_2")
  std::unordered_map<int, std::reference_wrapper<vtkDataArray>> get_numbered_arrays_with_prefix(
      const std::unordered_map<std::string, std::reference_wrapper<vtkDataArray>>& data,
      const std::string& prefix)
  {
    std::unordered_map<int, std::reference_wrapper<vtkDataArray>> arrays;

    auto prefix_filter = std::views::filter(
        [&](const auto& kv)
        {
          const auto& name = kv.first;
          return name.size() > prefix.size() + 1 && name.starts_with(prefix + "_") &&
                 name.substr(prefix.size() + 1).find_first_not_of("0123456789") ==
                     std::string_view::npos;
        });
    for (const auto& [name, data] : data | prefix_filter)
    {
      int set_id = std::stoi(name.substr(prefix.size() + 1).data());

      arrays.emplace(set_id, data);
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

Core::IO::MeshInput::RawMesh<3> Core::IO::VTU::read_vtu_file(const std::filesystem::path& vtu_file)
{
  FOUR_C_ASSERT_ALWAYS(
      std::filesystem::exists(vtu_file), "File {} does not exist.", vtu_file.string());

  Core::IO::MeshInput::RawMesh<3> mesh{};

  // Read the VTU file
  vtkSmartPointer<vtkXMLUnstructuredGridReader> reader =
      vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
  reader->SetFileName(vtu_file.c_str());
  reader->Update();
  vtkUnstructuredGrid* vtk_mesh = reader->GetOutput();

  // first read the points of the mesh and the point-sets
  auto point_data = get_vtk_data(vtk_mesh->GetPointData());
  mesh.points.resize(vtk_mesh->GetNumberOfPoints());

  auto point_sets = get_numbered_arrays_with_prefix(point_data, "point_set");
  for (const auto& [name, vtk_data] : point_data)
  {
    mesh.point_data[name] = make_empty_field_data_variant(vtk_data.get(), /*reserve=*/true);
  }

  for (vtkIdType i = 0; i < vtk_mesh->GetNumberOfPoints(); ++i)
  {
    vtk_mesh->GetPoint(i, mesh.points[static_cast<int>(i)].data());

    // process all point data
    for (const auto& [name, array_ref] : point_data)
    {
      // Avoid capturing structured binding for clang OpenMP
      vtkDataArray& array = array_ref.get();
      std::visit(
          [&](auto& value)
          {
            value.emplace_back(
                extract_data_item<typename std::remove_reference_t<decltype(value)>::value_type>(
                    array, i));
          },
          mesh.point_data[name]);
    }

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
  auto cell_data = get_vtk_data(vtk_mesh->GetCellData());
  vtkDataArray& cell_block_info = get_array(vtk_mesh->GetCellData(), "block_id");

  for (vtkIdType i = 0; i < vtk_mesh->GetNumberOfCells(); i++)
  {
    const int block_id = extract_component_from_integral_array<int>(cell_block_info, i, 0);

    const auto cell_type = get_celltype_from_vtk(vtk_mesh->GetCellType(i));

    const auto [emplaced_item, inserted] =
        mesh.cell_blocks.try_emplace(block_id, MeshInput::CellBlock<3>{cell_type});

    MeshInput::CellBlock<3>& cell_block = emplaced_item->second;

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
    cell_block.add_cell(translate_vtk_connectivity(
        cell_type, std::span{connectivity, static_cast<std::size_t>(number_of_points)}));

    // process all cell data
    for (const auto& [name, array_ref] : cell_data)
    {
      // Avoid capturing structured binding for clang OpenMP
      vtkDataArray& array = array_ref.get();
      if (cell_block.size() == 1)
      {
        // This is a new cell-block: Prepare the data container
        cell_block.cell_data[name] = make_empty_field_data_variant(array, /*reserve=*/false);
      }

      std::visit(
          [&](auto& value)
          {
            using ValueType = typename std::remove_reference_t<decltype(value)>::value_type;
            value.emplace_back(extract_data_item<ValueType>(array, i));
          },
          cell_block.cell_data[name]);
    }
  }

  MeshInput::assert_valid(mesh);
  return mesh;
}

#else
Core::IO::MeshInput::RawMesh<3> Core::IO::VTU::read_vtu_file(const std::filesystem::path& vtu_file)
{
  FOUR_C_THROW(
      "You have to enable VTK to support vtu mesh file input. Reconfigure 4C with the CMake option "
      "'FOUR_C_WITH_VTK' set to 'ON'.");
}
#endif

FOUR_C_NAMESPACE_CLOSE
