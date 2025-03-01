// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLID_PORO_3D_ELE_CALC_LIB_P1_HPP
#define FOUR_C_SOLID_PORO_3D_ELE_CALC_LIB_P1_HPP


#include "4C_config.hpp"

#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_utils_exceptions.hpp"

#include <ranges>

FOUR_C_NAMESPACE_OPEN



namespace Discret::Elements
{
  enum class SolidPoroDofType
  {
    displacement,
    porosity,
    none
  };

  template <SolidPoroDofType type, int dim>
  int get_global_index(int local_index)
  {
    if constexpr (type == SolidPoroDofType::displacement)
    {
      return local_index + local_index / dim;
    }
    else if constexpr (type == SolidPoroDofType::porosity)
    {
      return (local_index + 1) * (dim + 1) - 1;
    }
    else if constexpr (type == SolidPoroDofType::none)
    {
      return local_index;
    }
  };

  template <SolidPoroDofType type, int dim>
  int get_expected_target_matrix_size(int num_local_size)
  {
    if constexpr (type == SolidPoroDofType::displacement)
    {
      FOUR_C_ASSERT_ALWAYS(num_local_size % dim == 0,
          "Local size does not match the dimension. Expecting a multiple of %i but got %i", dim,
          num_local_size);
      return num_local_size / dim * (dim + 1);
    }
    else if constexpr (type == SolidPoroDofType::porosity)
    {
      return num_local_size * (dim + 1);
    }
    else if constexpr (type == SolidPoroDofType::none)
    {
      return num_local_size;
    }
  };

  template <SolidPoroDofType row_type, SolidPoroDofType col_type, int dim = 3>
  void assemble_mixed_displacement_porosity_matrix(
      Core::LinAlg::SerialDenseMatrix& target, const Core::LinAlg::SerialDenseMatrix& source)
  {
    FOUR_C_ASSERT(
        (get_expected_target_matrix_size<row_type, dim>(source.num_rows()) == target.num_rows()),
        "Source and target matrix dimensions do not match for the specified row type. Got %i but "
        "expected %i.",
        target.num_rows(), get_expected_target_matrix_size<row_type, dim>(source.num_rows()));

    FOUR_C_ASSERT(
        (get_expected_target_matrix_size<col_type, dim>(source.num_cols()) == target.num_cols()),
        "Source and target matrix dimensions do not match for the specified col type. Got %i but "
        "expected %i",
        target.num_cols(), get_expected_target_matrix_size<col_type, dim>(source.num_cols()));

    for (int i = 0; i < source.num_rows(); ++i)
    {
      for (int j = 0; j < source.num_cols(); ++j)
      {
        target(get_global_index<row_type, dim>(i), get_global_index<col_type, dim>(j)) +=
            source(i, j);
      }
    }
  };

  template <SolidPoroDofType dof_type, int dim = 3>
  void assemble_mixed_displacement_porosity_vector(
      Core::LinAlg::SerialDenseVector& target, const Core::LinAlg::SerialDenseVector& source)
  {
    FOUR_C_ASSERT(
        (get_expected_target_matrix_size<dof_type, dim>(source.num_rows()) == target.num_rows()),
        "Source and target vector dimensions do not match for the specified dof type. Got %i but "
        "expected %i",
        target.num_rows(), get_expected_target_matrix_size<dof_type, dim>(source.num_rows()));

    for (int i = 0; i < source.num_rows(); ++i)
    {
      target(get_global_index<dof_type, dim>(i)) += source(i);
    }
  };

  inline std::vector<int> get_reduced_displacement_location_array(
      const std::vector<int>& la, int num_nodes, int dim)
  {
    auto displacement_dofs = [i = -1, dim](const auto&) mutable
    {
      i = (i + 1) % (dim + 1);
      return i < dim;
    };

    std::vector<int> displacement_location_array;
    displacement_location_array.reserve(num_nodes * dim);
    std::ranges::copy(la | std::views::filter(displacement_dofs),
        std::back_inserter(displacement_location_array));

    return displacement_location_array;
  }

  inline std::vector<int> get_reduced_porosity_location_array(
      const std::vector<int>& la, int num_nodes, int dim)
  {
    auto porosity_dofs = [i = -1, dim](const auto&) mutable
    {
      i = (i + 1) % (dim + 1);
      return i == dim;
    };

    std::vector<int> displacement_location_array;
    displacement_location_array.reserve(num_nodes * dim);
    std::ranges::copy(
        la | std::views::filter(porosity_dofs), std::back_inserter(displacement_location_array));

    return displacement_location_array;
  }
}  // namespace Discret::Elements

FOUR_C_NAMESPACE_CLOSE

#endif