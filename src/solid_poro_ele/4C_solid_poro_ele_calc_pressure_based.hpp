// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLID_PORO_ELE_CALC_PRESSURE_BASED_HPP
#define FOUR_C_SOLID_PORO_ELE_CALC_PRESSURE_BASED_HPP


#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_utils_gausspoints.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_solid_ele_calc_lib.hpp"
#include "4C_utils_function.hpp"
FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  class FluidPoroMultiPhase;
  class StructPoro;
}  // namespace Mat
namespace Solid::Elements
{
  class ParamsInterface;
}  // namespace Solid::Elements

namespace Discret
{

  namespace Elements
  {
    template <Core::FE::CellType celltype>
    class SolidPoroPressureBasedEleCalc
    {
     public:
      SolidPoroPressureBasedEleCalc()
        requires(Core::FE::dim<celltype> == 3);
      SolidPoroPressureBasedEleCalc(
          double reference_thickness, Discret::Elements::PlaneAssumption plane_assumption)
        requires(Core::FE::dim<celltype> == 2);

      /*!
       * @brief This method adds the solid pressure contribution to the nonlinear force vector and
       * stiffness matrix.
       */
      void add_solidpressure_contribution_to_nonlinear_force_stiffness(
          const Core::Elements::Element& ele, Mat::StructPoro& porostructmat,
          Mat::FluidPoroMultiPhase& porofluidmat, const Inpar::Solid::KinemType& kinematictype,
          const Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
          Teuchos::ParameterList& params, Core::LinAlg::SerialDenseVector* force_vector,
          Core::LinAlg::SerialDenseMatrix* stiffness_matrix);

      /*!
       * @brief This method adds a possible body force contribution to the nonlinear force vector
       * and stiffness matrix.
       */
      void add_bodyforce_contribution_to_nonlinear_force_stiffness(
          const Core::Elements::Element& ele, Mat::StructPoro& porostructmat,
          Mat::FluidPoroMultiPhase& porofluidmat, const Inpar::Solid::KinemType& kinematictype,
          const Core::Utils::FunctionOfSpaceTime* bodyforce_contribution_function,
          const double time, const Core::FE::Discretization& discretization,
          Core::Elements::LocationArray& la, Teuchos::ParameterList& params,
          Core::LinAlg::SerialDenseVector* force_vector,
          Core::LinAlg::SerialDenseMatrix* stiffness_matrix);

      void evaluate_nonlinear_force_stiffness_od(const Core::Elements::Element& ele,
          Mat::StructPoro& porostructmat, Mat::FluidPoroMultiPhase& porofluidmat,
          const Inpar::Solid::KinemType& kinematictype,
          const Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
          Teuchos::ParameterList& params, Core::LinAlg::SerialDenseMatrix& stiffness_matrix);

      void poro_setup(
          Mat::StructPoro& porostructmat, const Core::IO::InputParameterContainer& container);

     private:
      /// static values for matrix sizes
      static constexpr int num_nodes_ = Core::FE::num_nodes(celltype);
      static constexpr int num_dim_ = Core::FE::dim<celltype>;
      static constexpr int num_dof_per_ele_ = num_nodes_ * num_dim_;
      static constexpr int num_str_ = num_dim_ * (num_dim_ + 1) / 2;

      Core::FE::GaussIntegration gauss_integration_;

      ElementProperties<celltype> element_properties_;
    };
  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif