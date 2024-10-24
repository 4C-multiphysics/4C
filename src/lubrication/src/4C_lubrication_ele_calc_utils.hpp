// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LUBRICATION_ELE_CALC_UTILS_HPP
#define FOUR_C_LUBRICATION_ELE_CALC_UTILS_HPP

#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_utils_integration.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Lubrication
{
  //! Template Meta Programming version of switch over discretization type
  template <Core::FE::CellType distype>
  struct DisTypeToOptGaussRule
  {
  };
  template <>
  struct DisTypeToOptGaussRule<Core::FE::CellType::quad4>
  {
    static constexpr Core::FE::GaussRule2D rule = Core::FE::GaussRule2D::quad_9point;
  };
  template <>
  struct DisTypeToOptGaussRule<Core::FE::CellType::quad8>
  {
    static constexpr Core::FE::GaussRule2D rule = Core::FE::GaussRule2D::quad_25point;
  };
  template <>
  struct DisTypeToOptGaussRule<Core::FE::CellType::quad9>
  {
    static constexpr Core::FE::GaussRule2D rule = Core::FE::GaussRule2D::quad_25point;
  };
  template <>
  struct DisTypeToOptGaussRule<Core::FE::CellType::tri3>
  {
    static constexpr Core::FE::GaussRule2D rule = Core::FE::GaussRule2D::tri_7point;
  };
  template <>
  struct DisTypeToOptGaussRule<Core::FE::CellType::tri6>
  {
    static constexpr Core::FE::GaussRule2D rule = Core::FE::GaussRule2D::tri_16point;
  };
  template <>
  struct DisTypeToOptGaussRule<Core::FE::CellType::line2>
  {
    static constexpr Core::FE::GaussRule1D rule = Core::FE::GaussRule1D::line_3point;
  };
  template <>
  struct DisTypeToOptGaussRule<Core::FE::CellType::line3>
  {
    static constexpr Core::FE::GaussRule1D rule = Core::FE::GaussRule1D::line_5point;
  };

  //! Template Meta Programming version of switch over discretization type
  template <Core::FE::CellType distype>
  struct DisTypeToGaussRuleForExactSol
  {
  };
  template <>
  struct DisTypeToGaussRuleForExactSol<Core::FE::CellType::quad4>
  {
    static constexpr Core::FE::GaussRule2D rule = Core::FE::GaussRule2D::quad_9point;
  };
  template <>
  struct DisTypeToGaussRuleForExactSol<Core::FE::CellType::quad8>
  {
    static constexpr Core::FE::GaussRule2D rule = Core::FE::GaussRule2D::quad_25point;
  };
  template <>
  struct DisTypeToGaussRuleForExactSol<Core::FE::CellType::quad9>
  {
    static constexpr Core::FE::GaussRule2D rule = Core::FE::GaussRule2D::quad_25point;
  };
  template <>
  struct DisTypeToGaussRuleForExactSol<Core::FE::CellType::tri3>
  {
    static constexpr Core::FE::GaussRule2D rule = Core::FE::GaussRule2D::tri_7point;
  };
  template <>
  struct DisTypeToGaussRuleForExactSol<Core::FE::CellType::tri6>
  {
    static constexpr Core::FE::GaussRule2D rule = Core::FE::GaussRule2D::tri_16point;
  };
  template <>
  struct DisTypeToGaussRuleForExactSol<Core::FE::CellType::line2>
  {
    static constexpr Core::FE::GaussRule1D rule = Core::FE::GaussRule1D::line_3point;
  };  // not tested
  template <>
  struct DisTypeToGaussRuleForExactSol<Core::FE::CellType::line3>
  {
    static constexpr Core::FE::GaussRule1D rule = Core::FE::GaussRule1D::line_5point;
  };  // not tested


}  // namespace Lubrication


FOUR_C_NAMESPACE_CLOSE

#endif
