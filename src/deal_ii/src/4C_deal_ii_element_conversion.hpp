// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_DEAL_II_ELEMENT_CONVERSION_HPP
#define FOUR_C_DEAL_II_ELEMENT_CONVERSION_HPP

#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "cut/4C_cut_find_cycles.hpp"

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>

FOUR_C_NAMESPACE_OPEN



namespace DealiiWrappers
{
  namespace ConversionTools
  {
    namespace Internal
    {
      template <bool index_runs_over_four_c, int dim, int spacedim = dim>
      void reindex_shape_functions(Core::FE::CellType, unsigned int n_components,
          const std::span<const int>& reindex_scalar, std::vector<int>& reindexing);

      /**
       * reinidexings of the shape functions from 4C to deal.II for different cell types.
       */
      namespace FourCToDeal
      {
        // dimension 1
        static constexpr std::array reindex_line2{0, 1};

        // -----------------------------------------------------------------------------------------
        // -----------------------------------------------------------------------------------------
        // -----------------------------------------------------------------------------------------

        // dimension 2
        static constexpr std::array reindex_quad4{0, 1, 3, 2};
        static constexpr std::array reindex_quad9{0, 1, 3, 2, 6, 5, 7, 4, 8};

        // --------------------------------------------------------------------------------------
        // --------------------------------------------------------------------------------------
        // --------------------------------------------------------------------------------------

        // dimension 3
        static constexpr std::array reindex_tet4{0, 1, 2, 3};
        static constexpr std::array reindex_hex8{0, 1, 3, 2, 4, 5, 7, 6};
        static constexpr std::array reindex_hex27{0, 1, 3, 2, 4, 5, 7, 6, 10, 9, 11, 8, 16, 17, 19,
            18, 14, 13, 15, 12, 24, 22, 21, 23, 20, 25, 26};
      }  // namespace FourCToDeal
      namespace DealToFourC
      {
        // dimension 1
        static constexpr std::array reindex_line2{0, 1};

        // --------------------------------------------------------------------------------------
        // --------------------------------------------------------------------------------------
        // --------------------------------------------------------------------------------------

        // dimension 2
        static constexpr std::array reindex_quad4{0, 1, 3, 2};
        static constexpr std::array reindex_quad9{0, 1, 3, 2, 7, 5, 4, 6, 8};

        // --------------------------------------------------------------------------------------
        // --------------------------------------------------------------------------------------
        // --------------------------------------------------------------------------------------

        // dimension 3
        static constexpr std::array reindex_tet4{0, 1, 2, 3};
        static constexpr std::array reindex_hex8{0, 1, 3, 2, 4, 5, 7, 6};
        static constexpr std::array reindex_hex27{// vertices
            0, 1, 3, 2, 4, 5, 7, 6,
            // lines
            11, 9, 8, 10, 19, 17, 16, 18, 12, 13, 15, 14,
            // faces
            24, 22, 21, 23, 20, 25,
            // center
            26};


      }  // namespace DealToFourC

    }  // namespace Internal

    namespace FourCToDeal
    {
      /**
       * Get the local reindexing of the shape functions for a given cell type.
       * This function computes the mapping from 4C to deal.II, i.e.,
       * i-th shape function in 4C corresponds to the
       * reindex_shape_functions_scalar(cell_type)[i]-th shape function in deal.II.
       */
      constexpr std::span<const int> reindex_shape_functions_scalar(Core::FE::CellType cell_type);

      /**
       * Get the local reindexing of the shape functions for a given cell type and number of
       * components. This function allows handling finite elements with multiple components e.g.
       * vector valued finite elements. As reindex_shape_functions_scalar(), this function computes
       * the mapping from 4C to deal.II. I.e. the i-th shape function in 4C corresponds to the
       * reindex_shape_functions(cell_type)[i]-th shape function in deal.II.
       *
       * @param n_components The number of components that are present in the finite element.
       * @param reindexing vector that will be filled with the reindexing information. It will
       * be resized to the correct size (i.e. number of shape functions in 4C times n_components).
       *
       * @tparam dim spatial dimension of the finite element. This is necessary since we need to
       * construct a dealii::FESystem object to get the correct reindexing as the indexing is an
       * implementation detail that must be queried from the FESystem object.
       * @tparam spacedim as dim
       */
      template <int dim, int spacedim = dim>
      void reindex_shape_functions(
          Core::FE::CellType, unsigned int n_components, std::vector<int>& reindexing);

      /**
       * Return the deal.II finite element name for a given 4C cell type in a format so that the
       * function dealii::FETools::get_fe_by_name<dim, spacedim>(fe_name) can be used to extract
       * the finite element.
       */
      constexpr std::string_view dealii_fe_name(Core::FE::CellType cell_type);

      /**
       * Given a 4C element, extract the GIDs of its nodes and rearrange them to be compatible with
       * deal.II. Also, return the element center which we assume uniquely identifies the element.
       */
      template <int spacedim>
      dealii::Point<spacedim> vertices_to_dealii(
          const Core::Elements::Element* element, std::vector<unsigned>& vertex_gids);

    }  // namespace FourCToDeal

    namespace DealToFourC
    {
      std::span<const int> reindex_shape_functions_scalar(Core::FE::CellType cell_type);

      template <int dim, int spacedim = dim>
      void reindex_shape_functions(
          Core::FE::CellType, unsigned int n_components, std::vector<int>& reindexing);

      /**
       * Extract the 4C cell type from a given deal.II finite element name.
       * There is a caveat to this functions as deal.II it might not always be possible to extract
       * the 4C cell type as deal.II uses the same finite element names for different spatial
       * dimensions. (e.g. FE_Q(1) is the liner tensor product element in any spatial dimension
       * i.e. dim=1 -> line2, dim=2 -> quad4, dim=3 -> hex8).
       * This function thus can be provided with a dim and spacedim template parameter so that it
       * can resolve the ambiguity.
       * @Note that is not always necessary as sometimes the dimension is already encoded in the
       * fe_name, e.g. FE_Q<2>(1) is the linear tensor product element in 2D (quad4).
       * For that reason there is also a similar function below without the template parameters that
       * attempts to resolve the dimension from the fe_name string.
       */
      template <int dim, int spacedim = dim>
      constexpr Core::FE::CellType four_c_cell_type_for_dim(std::string_view finite_element_name);

      /**
       * Extract the 4C cell type from a given deal.II finite element name.
       * See the above function for details.
       */
      constexpr Core::FE::CellType four_c_cell_type(std::string_view finite_element_name);


    }  // namespace DealToFourC
  }  // namespace ConversionTools
}  // namespace DealiiWrappers



namespace DealiiWrappers
{
  namespace ConversionTools
  {
    namespace Internal
    {
      template <bool index_runs_over_four_c, int dim, int spacedim>
      void reindex_shape_functions(Core::FE::CellType cell_type, unsigned int n_components,
          const std::span<const int>& reindex_scalar, std::vector<int>& reindexing)
      {
        FOUR_C_ASSERT(n_components > 0, "The number of components must be greater than zero.");
        reindexing.resize(reindex_scalar.size() * n_components);
        if (n_components == 1)
        {
          reindexing.assign(reindex_scalar.begin(), reindex_scalar.end());
        }

        const auto fe_name = ConversionTools::FourCToDeal::dealii_fe_name(cell_type);
        const auto fe = dealii::FETools::get_fe_by_name<dim, spacedim>(std::string(fe_name));
        dealii::FESystem<dim, spacedim> fe_system(*fe, n_components);

        FOUR_C_ASSERT(fe_system.dofs_per_cell == reindex_scalar.size() * n_components,
            "The number of dofs per cell in the deal.II finite element does not match the "
            "number of dofs per cell in the 4C finite element.");

        for (unsigned int dof = 0; dof < reindex_scalar.size(); ++dof)
        {
          for (unsigned int c = 0; c < n_components; ++c)
          {
            // get the dealii index through the FESystem since the
            // actual indexing is an implementation detail within deal.II that may change
            const auto current_dealii_index =
                fe_system.component_to_system_index(c, reindex_scalar[dof]);
            // in 4C, the dof indices are grouped per node, so we need to iterate over the
            // components
            const auto current_four_c_index = c + dof * n_components;
            const unsigned int index =
                index_runs_over_four_c ? current_four_c_index : current_dealii_index;
            const unsigned int value =
                index_runs_over_four_c ? current_dealii_index : current_four_c_index;

            reindexing[index] = value;
          }
        }
      }

      constexpr int extract_dimension_from_fe_name(const std::string_view s)
      {
        auto l = s.find('<');
        auto r = s.find('>', l);
        if (l == std::string_view::npos || r == std::string_view::npos || l + 1 >= r) return -1;
        int value = 0;
        for (size_t i = l + 1; i < r; ++i)
        {
          char c = s[i];
          if (c < '0' || c > '9') throw std::invalid_argument("Non-digit in int");
          value = value * 10 + (c - '0');
        }
        return value;
      }
    }  // namespace Internal

    namespace FourCToDeal
    {
      constexpr inline std::span<const int> reindex_shape_functions_scalar(
          Core::FE::CellType cell_type)
      {
        switch (cell_type)
        {
          // dimension 1
          {
            case Core::FE::CellType::line2:
            {
              return Internal::FourCToDeal::reindex_line2;
            }
          }

          // -----------------------------------------------------------------------------------------
          // -----------------------------------------------------------------------------------------
          // -----------------------------------------------------------------------------------------

          // dimension 2
          {
            case Core::FE::CellType::quad4:
            {
              return Internal::FourCToDeal::reindex_quad4;
            }
            case Core::FE::CellType::quad9:
            {
              return Internal::FourCToDeal::reindex_quad9;
            }
          }

          // --------------------------------------------------------------------------------------
          // --------------------------------------------------------------------------------------
          // --------------------------------------------------------------------------------------

          // dimension 3
          {
            case Core::FE::CellType::tet4:
            {
              return Internal::FourCToDeal::reindex_tet4;
            }
            case Core::FE::CellType::hex8:
            {
              return Internal::FourCToDeal::reindex_hex8;
            }
            case Core::FE::CellType::hex27:
            {
              return Internal::FourCToDeal::reindex_hex27;
            }
          }
          default:
          {
            FOUR_C_THROW(
                "Unsupported cell type '{}'.", Core::FE::cell_type_to_string(cell_type).c_str());
          }
        }
      }

      template <int dim, int spacedim>
      inline void reindex_shape_functions(
          Core::FE::CellType cell_type, unsigned int n_components, std::vector<int>& reindexing)
      {
        Internal::reindex_shape_functions<true, dim, spacedim>(
            cell_type, n_components, reindex_shape_functions_scalar(cell_type), reindexing);
      }

      inline constexpr std::string_view dealii_fe_name(Core::FE::CellType cell_type)
      {
        switch (cell_type)
        {
          // dimension 1
          {
            case Core::FE::CellType::line2:
              return "FE_Q<1>(1)";
          }

          // -----------------------------------------------------------------------------------------
          // -----------------------------------------------------------------------------------------
          // -----------------------------------------------------------------------------------------

          // dimension 2
          {
            case Core::FE::CellType::quad4:
              return "FE_Q<2>(1)";
            case Core::FE::CellType::quad9:
              return "FE_Q<2>(2)";
          }

          // -----------------------------------------------------------------------------------------
          // -----------------------------------------------------------------------------------------
          // -----------------------------------------------------------------------------------------

          // dimension 3
          {
            case Core::FE::CellType::tet4:
              return "FE_SimplexP(1)";
            case Core::FE::CellType::hex8:
              return "FE_Q<3>(1)";
            case Core::FE::CellType::hex27:
              return "FE_Q<3>(2)";
          }
          default:
            FOUR_C_THROW(
                "Unsupported cell type '{}'.", Core::FE::cell_type_to_string(cell_type).c_str());
        }
      }

      template <int spacedim>
      dealii::Point<spacedim> vertices_to_dealii(
          const Core::Elements::Element* element, std::vector<unsigned>& vertex_gids)
      {
        auto reindexing = reindex_shape_functions_scalar(element->shape());

        switch (element->shape())
        {
          case Core::FE::CellType::line2:
          case Core::FE::CellType::quad4:
          case Core::FE::CellType::tet4:
          case Core::FE::CellType::hex8:
          {
            dealii::Point<spacedim> element_center;
            vertex_gids.resize(element->num_node());

            for (int lid = 0; lid < element->num_node(); ++lid)
            {
              const auto& node = element->nodes()[reindexing[lid]];
              vertex_gids[lid] = node->id();
              for (unsigned d = 0; d < spacedim; ++d) element_center[d] += node->x()[d];
            }

            // Normalize the center
            element_center /= element->num_node();
            return element_center;
          }
          case Core::FE::CellType::hex27:
          {
            dealii::Point<spacedim> element_center;
            vertex_gids.resize(8);

            // Only require the first 8 nodes for deal.II
            for (int lid = 0; lid < 8; ++lid)
            {
              const auto& node = element->nodes()[reindexing[lid]];
              vertex_gids[lid] = node->id();
              for (unsigned d = 0; d < spacedim; ++d) element_center[d] += node->x()[d];
            }
            // Normalize the center
            element_center *= 0.125;
            return element_center;
          }
          default:
            FOUR_C_THROW("Unsupported cell type '{}'.",
                Core::FE::cell_type_to_string(element->shape()).c_str());
        }
      }


    }  // namespace FourCToDeal

    namespace DealToFourC
    {
      inline std::span<const int> reindex_shape_functions_scalar(Core::FE::CellType cell_type)
      {
        switch (cell_type)
        {
          // dimension 1
          {
            case Core::FE::CellType::line2:
            {
              return Internal::DealToFourC::reindex_line2;
            }
          }

          // --------------------------------------------------------------------------------------
          // --------------------------------------------------------------------------------------
          // --------------------------------------------------------------------------------------

          // dimension 2
          {
            case Core::FE::CellType::quad4:
            {
              return Internal::DealToFourC::reindex_quad4;
            }
            case Core::FE::CellType::quad9:
            {
              return Internal::DealToFourC::reindex_quad9;
            }
          }

          // --------------------------------------------------------------------------------------
          // --------------------------------------------------------------------------------------
          // --------------------------------------------------------------------------------------

          // dimension 3
          {
            case Core::FE::CellType::tet4:
            {
              return Internal::DealToFourC::reindex_tet4;
            }
            case Core::FE::CellType::hex8:
            {
              return Internal::DealToFourC::reindex_hex8;
            }
            case Core::FE::CellType::hex27:
            {
              return Internal::DealToFourC::reindex_hex27;
            }
          }
          default:
          {
            FOUR_C_THROW(
                "Unsupported cell type '{}'.", Core::FE::cell_type_to_string(cell_type).c_str());
          }
        }
      }

      template <int dim, int spacedim>
      inline void reindex_shape_functions(
          Core::FE::CellType cell_type, unsigned int n_components, std::vector<int>& reindexing)
      {
        Internal::reindex_shape_functions<false, dim, spacedim>(
            cell_type, n_components, reindex_shape_functions_scalar(cell_type), reindexing);
      }


      template <int dim, int spacedim>
      constexpr Core::FE::CellType four_c_cell_type_for_dim(std::string_view finite_element_name)
      {
        if constexpr (dim == -1 || spacedim == -1)
        {
          FOUR_C_THROW(
              "The finite element name {} must contain a dimension (dim) in the form of "
              "FE_NAME<dim>(degree) within the '< >' brackets. If thats not the case use the "
              "templated version of this function, where you can specify the dimension explicitly.",
              finite_element_name);
          return Core::FE::CellType::dis_none;
        }
        if constexpr (dim == 1 && spacedim == 1)
        {
          if (finite_element_name == "FE_Q(1)" || finite_element_name == "FE_Q<1>(1)")
          {
            return Core::FE::CellType::line2;
          }
          FOUR_C_THROW("Unsupported finite element type '{}' for dim = {} and spacedim = {}.",
              finite_element_name, 1, 1);
          return Core::FE::CellType::dis_none;
        }
        if constexpr (dim == 2 && spacedim == 2)
        {
          if (finite_element_name == "FE_Q(1)" || finite_element_name == "FE_Q<2>(1)")
          {
            return Core::FE::CellType::quad4;
          }
          FOUR_C_THROW("Unsupported finite element type '{}' for dim = {} and spacedim = {}.",
              finite_element_name, 2, 2);
          return Core::FE::CellType::dis_none;
        }
        if constexpr (dim == 3 && spacedim == 3)
        {
          if (finite_element_name == "FE_Q(1)" || finite_element_name == "FE_Q<3>(1)")
          {
            return Core::FE::CellType::hex8;
          }
          if (finite_element_name == "FE_Q(2)" || finite_element_name == "FE_Q<3>(2)")
          {
            return Core::FE::CellType::hex27;
          }
          if (finite_element_name == "FE_SimplexP(1)" || finite_element_name == "FE_SimplexP<3>(1)")
          {
            return Core::FE::CellType::tet4;
          }
          FOUR_C_THROW("Unsupported finite element type '{}' for dim = {} and spacedim = {}.",
              finite_element_name, 3, 3);
          return Core::FE::CellType::dis_none;
        }
        FOUR_C_THROW("Unsupported finite element type '{}' for dim = {} and spacedim = {}.",
            finite_element_name, dim, spacedim);
        return Core::FE::CellType::dis_none;
      }


      constexpr Core::FE::CellType four_c_cell_type(std::string_view finite_element_name)
      {
        auto dim = Internal::extract_dimension_from_fe_name(finite_element_name);
        if (dim == -1)
        {
          four_c_cell_type_for_dim<-1, -1>(finite_element_name);
        }
        if (dim == 1)
        {
          return four_c_cell_type_for_dim<1, 1>(finite_element_name);
        }
        if (dim == 2)
        {
          return four_c_cell_type_for_dim<2, 2>(finite_element_name);
        }
        if (dim == 3)
        {
          return four_c_cell_type_for_dim<3, 3>(finite_element_name);
        }
        return Core::FE::CellType::dis_none;
      }
    }  // namespace DealToFourC
  }  // namespace ConversionTools
}  // namespace DealiiWrappers

FOUR_C_NAMESPACE_CLOSE

#endif
