// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_discretization.hpp"
#include "4C_porofluid_pressure_based_ele.hpp"
#include "4C_porofluid_pressure_based_ele_action.hpp"
#include "4C_porofluid_pressure_based_ele_factory.hpp"
#include "4C_porofluid_pressure_based_ele_interface.hpp"
#include "4C_porofluid_pressure_based_ele_parameter.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  evaluate the element (public)                           vuong 08/16 |
 *----------------------------------------------------------------------*/
int Discret::Elements::PoroFluidMultiPhase::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  // we assume here, that numdofpernode is equal for every node within
  // the element and does not change during the computations
  const int numdofpernode = num_dof_per_node(*(nodes()[0]));

  // check for the action parameter
  const auto action = Teuchos::getIntegralValue<PoroPressureBased::Action>(params, "action");
  switch (action)
  {
    // all physics-related stuff is included in the implementation class(es) that can
    // be used in principle inside any element (at the moment: only PoroFluidMultiPhase element)
    case PoroPressureBased::calc_mat_and_rhs:
    case PoroPressureBased::calc_fluid_struct_coupl_mat:
    case PoroPressureBased::calc_fluid_scatra_coupl_mat:
    case PoroPressureBased::calc_error:
    case PoroPressureBased::calc_pres_and_sat:
    case PoroPressureBased::calc_solidpressure:
    case PoroPressureBased::calc_porosity:
    case PoroPressureBased::calc_determinant_of_deformationgradient:
    case PoroPressureBased::calc_volfrac_blood_lung:
    case PoroPressureBased::recon_flux_at_nodes:
    case PoroPressureBased::calc_phase_velocities:
    case PoroPressureBased::calc_initial_time_deriv:
    case PoroPressureBased::calc_valid_dofs:
    case PoroPressureBased::calc_domain_integrals:
    {
      std::vector<Core::LinAlg::SerialDenseMatrix*> elemat(2);
      elemat[0] = &elemat1;
      elemat[1] = &elemat2;

      std::vector<Core::LinAlg::SerialDenseVector*> elevec(3);
      elevec[0] = &elevec1;
      elevec[1] = &elevec2;
      elevec[2] = &elevec3;

      return Discret::Elements::PoroFluidMultiPhaseFactory::provide_impl(
          shape(), numdofpernode, discretization.name())
          ->evaluate(this, params, discretization, la, elemat, elevec);
      break;
    }
    default:
    {
      FOUR_C_THROW("Unknown type of action '{}' for PoroFluidMultiPhase", action);
      break;
    }
  }  // switch(action)

  return 0;
}  // Discret::Elements::PoroFluidMultiPhase::Evaluate


/*----------------------------------------------------------------------*
 |  dummy                                                   vuong 08/16 |
 *----------------------------------------------------------------------*/
int Discret::Elements::PoroFluidMultiPhase::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, const Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  FOUR_C_THROW("evaluate_neumann for PoroFluidMultiPhase  not yet implemented!");
  //    The function is just a dummy. For PoroFluidMultiPhase elements, the integration
  //    integration of volume Neumann conditions (body forces) takes place
  //    in the element. We need it there for potential stabilisation terms!
  return 0;
}

FOUR_C_NAMESPACE_CLOSE
