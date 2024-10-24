// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLID_PORO_3D_ELE_CALC_PRESSURE_VELOCITY_BASED_HPP
#define FOUR_C_SOLID_PORO_3D_ELE_CALC_PRESSURE_VELOCITY_BASED_HPP


#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_utils_gausspoints.hpp"
#include "4C_solid_poro_3D_ele_calc_interface.hpp"
#include "4C_solid_poro_3D_ele_calc_lib_io.hpp"
#include "4C_solid_poro_3D_ele_properties.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  class FluidPoro;
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
    class SolidPoroPressureVelocityBasedEleCalc
    {
     public:
      SolidPoroPressureVelocityBasedEleCalc();

      void evaluate_nonlinear_force_stiffness(const Core::Elements::Element& ele,
          Mat::StructPoro& porostructmat, Mat::FluidPoro& porofluidmat,
          AnisotropyProperties anisotropy_properties, const Inpar::Solid::KinemType& kinematictype,
          const Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
          Teuchos::ParameterList& params, Core::LinAlg::SerialDenseVector* force_vector,
          Core::LinAlg::SerialDenseMatrix* stiffness_matrix,
          Core::LinAlg::SerialDenseMatrix* reactive_matrix);

      void evaluate_nonlinear_force_stiffness_od(const Core::Elements::Element& ele,
          Mat::StructPoro& porostructmat, Mat::FluidPoro& porofluidmat,
          AnisotropyProperties anisotropy_properties, const Inpar::Solid::KinemType& kinematictype,
          const Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
          Teuchos::ParameterList& params, Core::LinAlg::SerialDenseMatrix* stiffness_matrix);

      void poro_setup(
          Mat::StructPoro& porostructmat, const Core::IO::InputParameterContainer& container);



      void coupling_stress_poroelast(const Core::Elements::Element& ele,
          Mat::StructPoro& porostructmat, const Inpar::Solid::KinemType& kinematictype,
          const CouplStressIO& couplingstressIO, const Core::FE::Discretization& discretization,
          Core::Elements::LocationArray& la, Teuchos::ParameterList& params);



     private:
      /// static values for matrix sizes
      static constexpr int num_nodes_ = Core::FE::num_nodes<celltype>;
      static constexpr int num_dim_ = Core::FE::dim<celltype>;
      static constexpr int num_dof_per_ele_ = num_nodes_ * num_dim_;
      static constexpr int num_str_ = num_dim_ * (num_dim_ + 1) / 2;

      Core::FE::GaussIntegration gauss_integration_;
    };
  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE


#endif