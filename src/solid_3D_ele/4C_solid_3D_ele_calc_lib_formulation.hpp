// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLID_3D_ELE_CALC_LIB_FORMULATION_HPP
#define FOUR_C_SOLID_3D_ELE_CALC_LIB_FORMULATION_HPP

#include "4C_config.hpp"

#include "4C_solid_3D_ele_calc_lib.hpp"
#include "4C_solid_3D_ele_formulation.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret::Elements
{

  template <Core::FE::CellType celltype, typename SolidFormulation>
  class ElementFormulationDerivativeEvaluator
  {
   public:
    ElementFormulationDerivativeEvaluator(const Core::Elements::Element& ele,
        const Discret::Elements::ElementNodes<celltype>& element_nodes,
        const Core::LinAlg::Tensor<double, 3>& xi,
        const Discret::Elements::ShapeFunctionsAndDerivatives<celltype>& shape_functions,
        const Discret::Elements::JacobianMapping<celltype>& jacobian_mapping,
        const Core::LinAlg::Tensor<double, 3, 3>& deformation_gradient,
        const Discret::Elements::PreparationData<SolidFormulation>& preparation_data,
        Discret::Elements::SolidFormulationHistory<SolidFormulation>& history_data)
        : ele_(ele),
          element_nodes_(element_nodes),
          xi_(xi),
          shape_functions_(shape_functions),
          jacobian_mapping_(jacobian_mapping),
          deformation_gradient_(deformation_gradient),
          preparation_data_(preparation_data),
          history_data_(history_data)
    {
    }

    [[nodiscard]] Core::LinAlg::Matrix<9, Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>>
    evaluate_d_deformation_gradient_d_displacements() const
    {
      return Discret::Elements::evaluate_d_deformation_gradient_d_displacements(ele_,
          element_nodes_, xi_, shape_functions_, jacobian_mapping_, deformation_gradient_,
          preparation_data_, history_data_);
    }

    [[nodiscard]] Core::LinAlg::Matrix<9, Core::FE::dim<celltype>>
    evaluate_d_deformation_gradient_d_xi() const
    {
      return Discret::Elements::evaluate_d_deformation_gradient_d_xi(ele_, element_nodes_, xi_,
          shape_functions_, jacobian_mapping_, deformation_gradient_, preparation_data_,
          history_data_);
    }

    [[nodiscard]] Core::LinAlg::Matrix<9,
        Core::FE::num_nodes(celltype) * Core::FE::dim<celltype> * Core::FE::dim<celltype>>
    evaluate_d2_deformation_gradient_d_displacements_d_xi() const
    {
      return Discret::Elements::evaluate_d_deformation_gradient_d_displacements_d_xi(ele_,
          element_nodes_, xi_, shape_functions_, jacobian_mapping_, deformation_gradient_,
          preparation_data_, history_data_);
    }

   private:
    const Core::Elements::Element& ele_;
    const Discret::Elements::ElementNodes<celltype>& element_nodes_;
    const Core::LinAlg::Tensor<double, 3>& xi_;
    const Discret::Elements::ShapeFunctionsAndDerivatives<celltype>& shape_functions_;
    const Discret::Elements::JacobianMapping<celltype>& jacobian_mapping_;
    const Core::LinAlg::Tensor<double, 3, 3>& deformation_gradient_;
    const Discret::Elements::PreparationData<SolidFormulation>& preparation_data_;
    Discret::Elements::SolidFormulationHistory<SolidFormulation>& history_data_;
  };
}  // namespace Discret::Elements

FOUR_C_NAMESPACE_CLOSE

#endif