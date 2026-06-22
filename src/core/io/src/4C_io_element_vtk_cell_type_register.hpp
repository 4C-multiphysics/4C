// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_ELEMENT_VTK_CELL_TYPE_REGISTER_HPP
#define FOUR_C_IO_ELEMENT_VTK_CELL_TYPE_REGISTER_HPP

#include "4C_config.hpp"

#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_std23_unreachable.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  using VtkCellType = uint8_t;

  /*!
   * @brief This is a list of all known mapping from vtk-celltypes to 4C CellTypes
   *
   * @note This map only contains 1-to-1 mappings (bijective) of the celltype.
   */
  constexpr std::array vtk_celltype_mapping = {
      std::make_pair<Core::FE::CellType, VtkCellType>(Core::FE::CellType::quad4, 9),
      std::make_pair<Core::FE::CellType, VtkCellType>(Core::FE::CellType::quad6, 30),
      std::make_pair<Core::FE::CellType, VtkCellType>(Core::FE::CellType::quad8, 23),
      std::make_pair<Core::FE::CellType, VtkCellType>(Core::FE::CellType::quad9, 28),
      std::make_pair<Core::FE::CellType, VtkCellType>(Core::FE::CellType::tri3, 5),
      std::make_pair<Core::FE::CellType, VtkCellType>(Core::FE::CellType::tri6, 22),
      std::make_pair<Core::FE::CellType, VtkCellType>(Core::FE::CellType::hex8, 12),
      std::make_pair<Core::FE::CellType, VtkCellType>(Core::FE::CellType::hex20, 25),
      std::make_pair<Core::FE::CellType, VtkCellType>(Core::FE::CellType::hex27, 29),
      std::make_pair<Core::FE::CellType, VtkCellType>(Core::FE::CellType::tet4, 10),
      std::make_pair<Core::FE::CellType, VtkCellType>(Core::FE::CellType::tet10, 24),
      std::make_pair<Core::FE::CellType, VtkCellType>(Core::FE::CellType::wedge6, 13),
      std::make_pair<Core::FE::CellType, VtkCellType>(Core::FE::CellType::wedge15, 26),
      std::make_pair<Core::FE::CellType, VtkCellType>(Core::FE::CellType::pyramid5, 14),
      std::make_pair<Core::FE::CellType, VtkCellType>(Core::FE::CellType::line2, 3),
      std::make_pair<Core::FE::CellType, VtkCellType>(Core::FE::CellType::line3, 21),
      std::make_pair<Core::FE::CellType, VtkCellType>(Core::FE::CellType::point1, 1),
  };

  namespace Internal
  {

    template <std::array, typename>
    struct SupportedCellTypesHelper;

    template <std::array mapping, std::size_t... is>
    struct SupportedCellTypesHelper<mapping, std::integer_sequence<std::size_t, is...>>
    {
      using type = Core::FE::CelltypeSequence<mapping[is].first...>;
    };

    template <Core::FE::CellType cell_type>
    struct VTKConnectivityMapping;

    template <>
    struct VTKConnectivityMapping<Core::FE::CellType::quad4>
    {
      static constexpr std::array value = {0, 1, 2, 3};
    };
    template <>
    struct VTKConnectivityMapping<Core::FE::CellType::quad6>
    {
      static constexpr std::array value = {0, 1, 4, 3, 2, 5};
    };
    template <>
    struct VTKConnectivityMapping<Core::FE::CellType::quad8>
    {
      static constexpr std::array value = {0, 1, 2, 3, 4, 5, 6, 7};
    };
    template <>
    struct VTKConnectivityMapping<Core::FE::CellType::quad9>
    {
      static constexpr std::array value = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    };
    template <>
    struct VTKConnectivityMapping<Core::FE::CellType::tri3>
    {
      static constexpr std::array value = {0, 1, 2};
    };
    template <>
    struct VTKConnectivityMapping<Core::FE::CellType::tri6>
    {
      static constexpr std::array value = {0, 1, 2, 3, 4, 5};
    };
    template <>
    struct VTKConnectivityMapping<Core::FE::CellType::hex8>
    {
      static constexpr std::array value = {0, 1, 2, 3, 4, 5, 6, 7};
    };
    template <>
    struct VTKConnectivityMapping<Core::FE::CellType::hex20>
    {
      static constexpr std::array value = {
          0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15};
    };
    template <>
    struct VTKConnectivityMapping<Core::FE::CellType::hex27>
    {
      static constexpr std::array value = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 12,
          13, 14, 15, 24, 22, 21, 23, 20, 25, 26};
    };
    template <>
    struct VTKConnectivityMapping<Core::FE::CellType::tet4>
    {
      static constexpr std::array value = {0, 1, 2, 3};
    };
    template <>
    struct VTKConnectivityMapping<Core::FE::CellType::tet10>
    {
      static constexpr std::array value = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    };
    template <>
    struct VTKConnectivityMapping<Core::FE::CellType::wedge6>
    {
      static constexpr std::array value = {0, 2, 1, 3, 5, 4};
    };
    template <>
    struct VTKConnectivityMapping<Core::FE::CellType::wedge15>
    {
      static constexpr std::array value = {0, 2, 1, 3, 5, 4, 8, 7, 6, 14, 13, 12, 9, 11, 10};
    };
    template <>
    struct VTKConnectivityMapping<Core::FE::CellType::pyramid5>
    {
      static constexpr std::array value = {0, 1, 2, 3, 4};
    };
    template <>
    struct VTKConnectivityMapping<Core::FE::CellType::line2>
    {
      static constexpr std::array value = {0, 1};
    };
    template <>
    struct VTKConnectivityMapping<Core::FE::CellType::line3>
    {
      static constexpr std::array value = {0, 1, 2};
    };
    template <>
    struct VTKConnectivityMapping<Core::FE::CellType::point1>
    {
      static constexpr std::array value = {0};
    };

    /*!
     * @brief Special mapping for VTK cell types for input which can have variable number of points,
     * e.g., polylines for beams.
     *
     * This maps the vtk cell type and the number of points to the corresponding 4C cell type and
     * the connectivity mapping.
     */
    static std::map<std::pair<VtkCellType, long long>,
        std::pair<Core::FE::CellType, std::vector<std::size_t>>>
        vtk_celltype_special_mapping_for_input = {
            {{4, 2}, {Core::FE::CellType::line2, {0, 1}}},
            {{4, 3}, {Core::FE::CellType::line3, {0, 2, 1}}},
            {{4, 4}, {Core::FE::CellType::line4, {0, 3, 1, 2}}},
            {{4, 5}, {Core::FE::CellType::line5, {0, 4, 1, 2, 3}}},
    };

    /*!
     * @brief Special mapping for VTK cell types for output since some 4C cell types don't have a
     * direct VTK counterpart.
     */
    constexpr std::array vtk_celltype_special_mapping_for_output = {
        std::make_pair<Core::FE::CellType, VtkCellType>(Core::FE::CellType::hex16, 12),
        std::make_pair<Core::FE::CellType, VtkCellType>(Core::FE::CellType::hex18, 12),
        std::make_pair<Core::FE::CellType, VtkCellType>(Core::FE::CellType::line4, 4),
        std::make_pair<Core::FE::CellType, VtkCellType>(Core::FE::CellType::line5, 4),
        std::make_pair<Core::FE::CellType, VtkCellType>(Core::FE::CellType::line6, 4),
    };

    /*!
     * @brief Special mapping for VTK connectivity for output since some 4C cell types don't have a
     * direct VTK counterpart.
     *
     * @tparam cell_type The 4C cell type.
     */
    template <Core::FE::CellType cell_type>
    struct VTKConnectivitySpecialMappingForOutput;
    template <>
    struct VTKConnectivitySpecialMappingForOutput<Core::FE::CellType::hex16>
    {
      static constexpr std::array value = {0, 1, 2, 3, 8, 9, 10, 11};
    };
    template <>
    struct VTKConnectivitySpecialMappingForOutput<Core::FE::CellType::hex18>
    {
      static constexpr std::array value = {0, 1, 2, 3, 9, 10, 11, 12};
    };
    template <>
    struct VTKConnectivitySpecialMappingForOutput<Core::FE::CellType::line4>
    {
      static constexpr std::array value = {0, 1, 2, 4};
    };
    template <>
    struct VTKConnectivitySpecialMappingForOutput<Core::FE::CellType::line5>
    {
      static constexpr std::array value =
          VTKConnectivitySpecialMappingForOutput<Core::FE::CellType::line4>::value;
    };
    template <>
    struct VTKConnectivitySpecialMappingForOutput<Core::FE::CellType::line6>
    {
      static constexpr std::array value =
          VTKConnectivitySpecialMappingForOutput<Core::FE::CellType::line4>::value;
    };
  }  // namespace Internal

  using VTKSupportedCellTypes = Internal::SupportedCellTypesHelper<vtk_celltype_mapping,
      std::make_index_sequence<vtk_celltype_mapping.size()>>::type;

  template <Core::FE::CellType celltype, std::array mapping = vtk_celltype_mapping>
  static constexpr VtkCellType vtk_cell_type = []() consteval
  {
    for (const auto& [fe_type, vtk_type] : mapping)
    {
      if (fe_type == celltype)
      {
        return vtk_type;
      }
    }
    FOUR_C_THROW("Unsupported cell type");
  }();

  /*!
   * @brief Creates a mapping of the 4C connectivity ordering to the VTK connectivity ordering.
   *
   * @tparam cell_type The 4C cell type.
   */
  template <Core::FE::CellType cell_type>
  constexpr std::array vtk_connectivity_mapping =
      Internal::VTKConnectivityMapping<cell_type>::value;

  /*!
   * @brief Defines the reverse connectivity mapping from VTK to 4C
   *
   * @note this is implicitly defined by the @p vtk_connectivity_mapping
   */
  template <Core::FE::CellType cell_type>
  static constexpr std::array vtk_connectivity_reverse_mapping = []() consteval
  {
    constexpr std::size_t size = vtk_connectivity_mapping<cell_type>.size();
    std::array<std::size_t, size> reverse_mapping{};

    for (std::size_t i = 0; i < vtk_connectivity_mapping<cell_type>.size(); ++i)
    {
      reverse_mapping[vtk_connectivity_mapping<cell_type>[i]] = i;
    }

    return reverse_mapping;
  }();


  /*!
   * @brief Get the vtk cell type and the connectivity mapping from 4C cell type for output
   *
   * @param four_c_ele_shape_type
   * @return constexpr std::pair<VtkCellType, std::vector<int>>
   */
  constexpr std::pair<VtkCellType,
      std::vector<int>> inline get_vtk_cell_type_from_element_cell_type(Core::FE::CellType
          four_c_ele_shape_type)
  {
    return Core::FE::cell_type_switch<VTKSupportedCellTypes>(
        four_c_ele_shape_type,
        [](auto celltype_t) -> std::pair<VtkCellType, std::vector<int>>
        {
          // this cell-type is directly supported by VTK
          constexpr Core::FE::CellType celltype = celltype_t();
          return std::make_pair(
              vtk_cell_type<celltype>, std::vector<int>(vtk_connectivity_mapping<celltype>.begin(),
                                           vtk_connectivity_mapping<celltype>.end()));
        },
        [](auto celltype_t) -> std::pair<VtkCellType, std::vector<int>>
        {
          // this cell-type is NOT directly supported
          // check whether this cell-type can be mapped to a supported cell-type
          constexpr Core::FE::CellType celltype = celltype_t();

          constexpr bool is_specially_supported_for_output =
              std::ranges::any_of(Internal::vtk_celltype_special_mapping_for_output,
                  [](const auto& pair) { return celltype == pair.first; });

          if constexpr (is_specially_supported_for_output)
          {
            for (const auto& [supported_celltype, vtk_type] :
                Internal::vtk_celltype_special_mapping_for_output)
            {
              return std::make_pair(vtk_type,
                  std::vector<int>(
                      Internal::VTKConnectivitySpecialMappingForOutput<celltype_t()>::value.begin(),
                      Internal::VTKConnectivitySpecialMappingForOutput<celltype_t()>::value.end()));
            }

            std23::unreachable();
          }

          FOUR_C_THROW(
              "Unsupported cell type {} for VTK output.", Core::FE::cell_type_to_string(celltype));
        });
  }

  /*!
   * @brief Get the 4C celltype from vtk cell-type and the 4C connectivity
   *
   * @param vtk_cell_type
   * @param vtk_connectivity
   * @return Core::FE::CellType
   */
  inline std::pair<Core::FE::CellType, std::vector<int>>
  get_celltype_from_vtk_and_translate_connectivity(
      const VtkCellType vtk_cell_type, const std::span<const long long> vtk_connectivity)
  {
    constexpr VtkCellType max_id = []() consteval
    {
      VtkCellType max_id = 0;
      for (auto [_, id] : vtk_celltype_mapping)
      {
        if (id > max_id) max_id = id;
      }
      return max_id;
    }();

    constexpr std::array<std::optional<Core::FE::CellType>, max_id + 1> celltype_mapping =
        []() consteval
    {
      std::array<std::optional<Core::FE::CellType>, max_id + 1> mapping{};
      for (const auto& pair : vtk_celltype_mapping)
      {
        mapping[pair.second] = pair.first;
      }
      return mapping;
    }();

    std::optional<Core::FE::CellType> cell_type =
        (vtk_cell_type <= max_id) ? celltype_mapping[vtk_cell_type] : std::nullopt;

    if (cell_type.has_value())
    {
      // A cell type could be found in the bijective mapping, now we can transform the connectivity.
      return {*cell_type,
          Core::FE::cell_type_switch<Core::IO::VTKSupportedCellTypes>(*cell_type,
              [&](auto celltype_t)
              {
                std::vector<int> four_c_connectivity(vtk_connectivity.size(), 0);

                for (std::size_t i = 0; i < vtk_connectivity.size(); ++i)
                {
                  four_c_connectivity[i] =
                      vtk_connectivity[Core::IO::vtk_connectivity_reverse_mapping<celltype_t()>[i]];
                }

                return four_c_connectivity;
              })};
    }
    else
    {
      // Could not find the element type in the bijective mapping, look if the element is given in
      // the special input mapping.
      auto map_item = Internal::vtk_celltype_special_mapping_for_input.find(
          {vtk_cell_type, vtk_connectivity.size()});
      if (map_item != Internal::vtk_celltype_special_mapping_for_input.end())
      {
        cell_type = map_item->second.first;
        std::vector<int> four_c_connectivity(vtk_connectivity.size(), 0);
        for (std::size_t i = 0; i < vtk_connectivity.size(); ++i)
        {
          four_c_connectivity[i] = vtk_connectivity[map_item->second.second[i]];
        }
        return {*cell_type, four_c_connectivity};
      }
    }

    FOUR_C_THROW(
        "VTK cell type {} with {} points not found in 4C.", vtk_cell_type, vtk_connectivity.size());
  }

}  // namespace Core::IO

FOUR_C_NAMESPACE_CLOSE

#endif
