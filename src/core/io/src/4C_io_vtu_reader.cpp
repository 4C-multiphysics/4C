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
#include "4C_utils_demangle.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_vector2D.hpp"

#include <algorithm>
#include <type_traits>
#include <typeindex>

#ifdef FOUR_C_WITH_VTK
#include <vtkArrayDispatch.h>
#include <vtkBitArray.h>
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
      // Handle vtkBitArray separately as it is not covered by vtkArrayDispatch::Integrals
      if (vtkBitArray* bitArray = vtkBitArray::SafeDownCast(&array))
      {
        FOUR_C_ASSERT_ALWAYS(compIdx == 0,
            "vtkBitArray only has one component, but component {} was requested.", compIdx);
        result = static_cast<T>(bitArray->GetValue(tupleIdx));
      }
      else
      {
        FOUR_C_THROW("Array {} is of type {}, but expecting an integer- or bit-type array!",
            array.GetName(), array.GetDataTypeAsString());
      }
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
  Core::IO::MeshInput::FieldDataVariantType make_container_with_supported_scalar_type(
      vtkDataArray& array, bool reserve = true)
  {
    std::optional<Core::IO::MeshInput::FieldDataVariantType> type{};
    if (!vtkArrayDispatch::DispatchByValueType<vtkArrayDispatch::AllTypes>::Execute(&array,
            [&](auto* typed_array)
            {
              using ValueType = typename std::decay_t<decltype(*typed_array)>::ValueType;
              if constexpr (std::is_integral_v<ValueType>)
              {
                Core::Utils::Vector2D<int> container{
                    static_cast<size_t>(array.GetNumberOfComponents())};
                if (reserve) container.reserve(array.GetNumberOfTuples());
                type.emplace(std::move(container));
              }
              else if constexpr (std::is_floating_point_v<ValueType>)
              {
                Core::Utils::Vector2D<double> container{
                    static_cast<size_t>(array.GetNumberOfComponents())};
                if (reserve) container.reserve(array.GetNumberOfTuples());
                type.emplace(std::move(container));
              }
              else
              {
                FOUR_C_THROW(
                    "Array {} is of type {}. 4C currently only supports the input of bit-, "
                    "integral- or floating point types.",
                    array.GetName(), array.GetDataTypeAsString());
              }
            }))
    {
      // Handle vtkBitArray separately as it is not covered by vtkArrayDispatch::AllTypes
      if (vtkBitArray::SafeDownCast(&array))
      {
        Core::Utils::Vector2D<bool> container{static_cast<size_t>(array.GetNumberOfComponents())};
        if (reserve) container.reserve(array.GetNumberOfTuples());
        type.emplace(std::move(container));
      }
      else
      {
        FOUR_C_THROW("Array {} is of type {}, which is not supported!", array.GetName(),
            array.GetDataTypeAsString());
      }
    }

    FOUR_C_ASSERT(type.has_value(), "Failed to create container for array {} of type {}!",
        array.GetName(), array.GetDataTypeAsString());

    return *type;
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
  Core::IO::MeshInput::FieldDataVariantType make_empty_field_data_variant(
      vtkDataArray& array, bool reserve = true)
  {
    return make_container_with_supported_scalar_type(array, /*reserve=*/reserve);
  }

  template <typename T>
  std::vector<T> extract_raw_data(vtkDataArray& array, vtkIdType index)
  {
    const int n_components = array.GetNumberOfComponents();

    std::vector<T> raw_data;


    auto extract_entry = [&](auto& typed_array) -> std::vector<T>
    {
      std::vector<T> raw_data_item(n_components);
      for (int c = 0; c < n_components; ++c)
      {
        raw_data_item[c] = static_cast<T>(typed_array.GetComponent(index, c));
      }
      return raw_data_item;
    };

    if (vtkArrayDispatch::DispatchByValueType<vtkArrayDispatch::AllTypes>::Execute(
            &array, [&](auto* typed_array) { raw_data = extract_entry(*typed_array); }))
    {
      return raw_data;
    }
    // Handle vtkBitArray separately as it is not covered by vtkArrayDispatch::AllTypes
    if (vtkBitArray* bit_array = vtkBitArray::SafeDownCast(&array))
    {
      return extract_entry(*bit_array);
    }
    FOUR_C_THROW("Failed to extract data from array {} of type {}!", array.GetName(),
        array.GetDataTypeAsString());
  }

  // Returns true if the given array stores integer values (any width, signed or unsigned)
  bool is_integral_array(vtkDataArray& array)
  {
    if (vtkBitArray::SafeDownCast(&array)) return true;
    bool result = false;
    vtkArrayDispatch::DispatchByValueType<vtkArrayDispatch::Integrals>::Execute(
        &array, [&](auto*) { result = true; });
    return result;
  }

  // Holds information about a single point-set array. Every entry carries the full array
  // name so the resulting PointSet is referenceable via NODE_SET_NAME (this also applies
  // to "point_set_<N>" arrays, which additionally keep their numeric ID for NODE_SET_ID).
  struct PointSetEntry
  {
    int id;
    std::string name;
    std::reference_wrapper<vtkDataArray> array;
  };

  // Returns all single-component integer PointData arrays as point-set entries.
  // Arrays named "point_set_<N>" keep their numeric ID; all other arrays are assigned
  // IDs after the highest numeric ID, in alphabetical order of the array name so that
  // IDs are reproducible across runs and compilers.
  std::vector<PointSetEntry> get_point_set_arrays(
      const std::unordered_map<std::string, std::reference_wrapper<vtkDataArray>>& data)
  {
    std::vector<PointSetEntry> numbered;
    std::vector<PointSetEntry> named;
    int max_numeric_id = 0;
    const std::string prefix = "point_set_";

    for (const auto& [array_name, array_ref] : data)
    {
      vtkDataArray& array = array_ref.get();

      if (array.GetNumberOfComponents() != 1) continue;
      if (!is_integral_array(array)) continue;

      if (array_name.size() > prefix.size() && array_name.starts_with(prefix))
      {
        const std::string suffix = array_name.substr(prefix.size());
        if (!suffix.empty() && suffix.find_first_not_of("0123456789") == std::string::npos)
        {
          int id = std::stoi(suffix);
          max_numeric_id = std::max(max_numeric_id, id);
          numbered.emplace_back(id, array_name, array_ref);
          continue;
        }
      }
      named.emplace_back(0, array_name, array_ref);
    }

    std::sort(named.begin(), named.end(),
        [](const PointSetEntry& a, const PointSetEntry& b) { return a.name < b.name; });

    int next_id = max_numeric_id + 1;
    for (auto& entry : named)
    {
      entry.id = next_id++;
      numbered.push_back(entry);
    }
    return numbered;
  }

  template <typename SourceType, typename TargetTensor>
  std::pair<std::type_index, std::function<std::any(std::any)>> make_tensor_conversion_item()
  {
    return std::make_pair(
        Core::IO::MeshInput::make_converter_type_index<SourceType, TargetTensor>(),
        [](std::any source) -> std::any
        {
          const auto& typed_source = std::any_cast<std::span<const SourceType>>(source);

          FOUR_C_ASSERT(typed_source.size() == TargetTensor::compressed_size,
              "Expecting {} components to convert to {}, but got {} components.",
              TargetTensor::compressed_size, Core::Utils::try_demangle(typeid(TargetTensor).name()),
              typed_source.size());

          TargetTensor result{};
          std::copy(typed_source.begin(), typed_source.end(), result.container().begin());

          // vtu uses row major ordering for tensors, but 4C uses column major, so we need to
          // transpose the result if it is a non-symmetric rank-2 tensor
          constexpr bool is_transpose = TargetTensor::rank() == 2 && !TargetTensor::is_compressed;
          if constexpr (is_transpose)
            return Core::LinAlg::transpose(result);
          else
            return result;
        });
  }

  template <unsigned dim>
  [[nodiscard]] std::unordered_map<std::type_index, std::function<std::any(std::any)>>
  make_type_converters()
  {
    std::unordered_map<std::type_index, std::function<std::any(std::any)>> converters{};

    // from double to double tensor
    converters.insert(make_tensor_conversion_item<double, Core::LinAlg::Tensor<double, dim>>());
    converters.insert(
        make_tensor_conversion_item<double, Core::LinAlg::SymmetricTensor<double, dim, dim>>());
    converters.insert(
        make_tensor_conversion_item<double, Core::LinAlg::Tensor<double, dim, dim>>());

    // from int to int tensor
    converters.insert(make_tensor_conversion_item<int, Core::LinAlg::Tensor<int, dim>>());
    converters.insert(
        make_tensor_conversion_item<int, Core::LinAlg::SymmetricTensor<int, dim, dim>>());
    converters.insert(make_tensor_conversion_item<int, Core::LinAlg::Tensor<int, dim, dim>>());

    // from bool to bool tensor
    converters.insert(make_tensor_conversion_item<bool, Core::LinAlg::Tensor<bool, dim>>());
    converters.insert(
        make_tensor_conversion_item<bool, Core::LinAlg::SymmetricTensor<bool, dim, dim>>());
    converters.insert(make_tensor_conversion_item<bool, Core::LinAlg::Tensor<bool, dim, dim>>());


    return converters;
  };
}  // namespace

Core::IO::MeshInput::RawMesh<3> Core::IO::VTU::read_vtu_file(const std::filesystem::path& vtu_file)
{
  FOUR_C_ASSERT_ALWAYS(
      std::filesystem::exists(vtu_file), "File {} does not exist.", vtu_file.string());

  Core::IO::MeshInput::RawMesh<3> mesh{};

  // append type converters for vtu
  mesh.converters = make_type_converters<3>();

  // Read the VTU file
  vtkSmartPointer<vtkXMLUnstructuredGridReader> reader =
      vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
  reader->SetFileName(vtu_file.c_str());
  reader->Update();
  vtkUnstructuredGrid* vtk_mesh = reader->GetOutput();

  // first read the points of the mesh and the point-sets
  auto point_data = get_vtk_data(vtk_mesh->GetPointData());
  mesh.points.resize(vtk_mesh->GetNumberOfPoints());

  auto point_sets = get_point_set_arrays(point_data);
  for (const auto& [name, vtk_data] : point_data)
  {
    mesh.point_data.emplace(name, make_empty_field_data_variant(vtk_data.get(), /*reserve=*/true));
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
            value.push_back(
                extract_raw_data<typename std::remove_reference_t<decltype(value)>::value_type>(
                    array, i));
          },
          mesh.point_data[name]);
    }

    // check whether this point is part of a point-set. The PointSet entry (and its name)
    // is created lazily on the first matched point, so arrays that mark no points at all
    // do not produce empty sets (which would fail assert_valid).
    for (const auto& entry : point_sets)
    {
      bool is_part_of_point_set =
          extract_component_from_integral_array<bool>(entry.array.get(), i, 0);

      if (is_part_of_point_set)
      {
        auto& point_set = mesh.point_sets[entry.id];
        point_set.point_ids.emplace(i);
        point_set.name = entry.name;
      }
    }
  }

  // now read the cell and their blocks
  auto cell_data = get_vtk_data(vtk_mesh->GetCellData());
  vtkDataArray& cell_block_info = get_array(vtk_mesh->GetCellData(), "block_id");

  for (vtkIdType i = 0; i < vtk_mesh->GetNumberOfCells(); i++)
  {
    const int block_id = extract_component_from_integral_array<int>(cell_block_info, i, 0);

    // extract connectivity
    vtkIdType number_of_points;
    const vtkIdType* connectivity = nullptr;
    vtk_mesh->GetCellPoints(i, number_of_points, connectivity);

    // Get 4C connectivity and cell type.
    const auto vtk_cell_type = vtk_mesh->GetCellType(i);
    const auto [cell_type, four_c_connectivity] = get_celltype_from_vtk_and_translate_connectivity(
        vtk_cell_type, std::span{connectivity, static_cast<std::size_t>(number_of_points)});

    const auto [emplaced_item, inserted] =
        mesh.cell_blocks.try_emplace(block_id, MeshInput::CellBlock<3>{cell_type});

    MeshInput::CellBlock<3>& cell_block = emplaced_item->second;

    FOUR_C_ASSERT_ALWAYS(emplaced_item->second.cell_type == cell_type,
        "Cell block {} has mixed cell types: {} and {}.", block_id,
        Core::FE::cell_type_to_string(emplaced_item->second.cell_type),
        Core::FE::cell_type_to_string(cell_type));

    // add to cell_block
    cell_block.add_cell(four_c_connectivity);

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
            value.push_back(extract_raw_data<ValueType>(array, i));
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
