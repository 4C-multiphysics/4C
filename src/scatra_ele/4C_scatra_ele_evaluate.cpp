// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_discretization.hpp"
#include "4C_mat_elchmat.hpp"
#include "4C_scatra_ele.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_ele_factory.hpp"
#include "4C_scatra_ele_interface.hpp"
#include "4C_scatra_ele_parameter_boundary.hpp"
#include "4C_scatra_ele_parameter_elch.hpp"
#include "4C_scatra_ele_parameter_elch_diffcond.hpp"
#include "4C_scatra_ele_parameter_lsreinit.hpp"
#include "4C_scatra_ele_parameter_std.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_scatra_ele_parameter_turbulence.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN



/*----------------------------------------------------------------------*
 |  evaluate the element (public)                              gjb 01/09|
 *----------------------------------------------------------------------*/
int Discret::Elements::Transport::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  FOUR_C_THROW("not implemented. Use the evaluate() method with Location Array instead!");
  return -1;
}

/*----------------------------------------------------------------------*
 |  evaluate the element (public)                              gjb 01/09|
 *----------------------------------------------------------------------*/
int Discret::Elements::Transport::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  // we assume here, that numdofpernode is equal for every node within
  // the discretization and does not change during the computations
  const int numdofpernode = num_dof_per_node(*(nodes()[0]));
  int numscal = numdofpernode;

  // perform additional operations specific to implementation type
  switch (impltype_)
  {
    case Inpar::ScaTra::impltype_elch_diffcond:
    case Inpar::ScaTra::impltype_elch_diffcond_multiscale:
    case Inpar::ScaTra::impltype_elch_diffcond_thermo:
    case Inpar::ScaTra::impltype_elch_electrode:
    case Inpar::ScaTra::impltype_elch_electrode_growth:
    case Inpar::ScaTra::impltype_elch_electrode_thermo:
    case Inpar::ScaTra::impltype_elch_NP:
    case Inpar::ScaTra::impltype_elch_scl:
    {
      // adapt number of transported scalars for electrochemistry problems
      numscal -= 1;

      // get the material of the first element
      // we assume here, that the material is equal for all elements in this discretization
      std::shared_ptr<Core::Mat::Material> material = Transport::material();
      if (material->material_type() == Core::Materials::m_elchmat)
      {
        const auto* actmat = static_cast<const Mat::ElchMat*>(material.get());

        numscal = actmat->num_scal();
      }

      break;
    }
    case Inpar::ScaTra::impltype_levelset:
    case Inpar::ScaTra::impltype_lsreinit:
    {
      // decide whether reinitialization is active or not
      if (not params.get<bool>("solve reinit eq", false))
        impltype_ = Inpar::ScaTra::impltype_levelset;
      else
        impltype_ = Inpar::ScaTra::impltype_lsreinit;

      break;
    }

    case Inpar::ScaTra::impltype_std:
    case Inpar::ScaTra::impltype_thermo_elch_electrode:
    case Inpar::ScaTra::impltype_thermo_elch_diffcond:
    case Inpar::ScaTra::impltype_advreac:
    case Inpar::ScaTra::impltype_chemo:
    case Inpar::ScaTra::impltype_chemoreac:
    case Inpar::ScaTra::impltype_aniso:
    case Inpar::ScaTra::impltype_cardiac_monodomain:
    case Inpar::ScaTra::impltype_loma:
    case Inpar::ScaTra::impltype_poro:
    case Inpar::ScaTra::impltype_pororeac:
    case Inpar::ScaTra::impltype_pororeacECM:
    case Inpar::ScaTra::impltype_multipororeac:
    case Inpar::ScaTra::impltype_one_d_artery:
    case Inpar::ScaTra::impltype_no_physics:
      // do nothing in these cases
      break;

    default:
    {
      // other implementation types are invalid
      FOUR_C_THROW("Invalid implementation type!");
      break;
    }
  }

  // check for the action parameter
  const auto action = Teuchos::getIntegralValue<ScaTra::Action>(params, "action");
  switch (action)
  {
    // all physics-related stuff is included in the implementation class(es) that can
    // be used in principle inside any element (at the moment: only Transport element)
    case ScaTra::Action::calc_mat_and_rhs:
    case ScaTra::Action::calc_rhs:
    {
      return ScaTraFactory::provide_impl(
          shape(), impltype_, numdofpernode, numscal, discretization.name())
          ->evaluate(this, params, discretization, la, elemat1, elemat2, elevec1, elevec2, elevec3);
      break;
    }

    case ScaTra::Action::calc_scatra_mono_odblock_fluid:
    case ScaTra::Action::calc_scatra_mono_odblock_mesh:
    case ScaTra::Action::calc_scatra_mono_odblock_scatrathermo:
    case ScaTra::Action::calc_scatra_mono_odblock_thermoscatra:
    {
      return ScaTraFactory::provide_impl(
          shape(), impltype_, numdofpernode, numscal, discretization.name())
          ->evaluate_od(
              this, params, discretization, la, elemat1, elemat2, elevec1, elevec2, elevec3);
      break;
    }

    case ScaTra::Action::check_scatra_element_parameter:
    case ScaTra::Action::calc_initial_time_deriv:
    case ScaTra::Action::integrate_shape_functions:
    case ScaTra::Action::calc_flux_domain:
    case ScaTra::Action::calc_total_and_mean_scalars:
    case ScaTra::Action::calc_mean_scalar_time_derivatives:
    case ScaTra::Action::calc_domain_and_bodyforce:
    case ScaTra::Action::calc_scatra_box_filter:
    case ScaTra::Action::calc_turbulent_prandtl_number:
    case ScaTra::Action::calc_vreman_scatra:
    case ScaTra::Action::calc_subgrid_diffusivity_matrix:
    case ScaTra::Action::get_material_parameters:
    case ScaTra::Action::calc_mean_Cai:
    case ScaTra::Action::calc_dissipation:
    case ScaTra::Action::calc_mat_and_rhs_lsreinit_correction_step:
    case ScaTra::Action::calc_node_based_reinit_velocity:
    case ScaTra::Action::calc_domain_integral:
    case ScaTra::Action::calc_error:
    case ScaTra::Action::calc_elch_conductivity:
    case ScaTra::Action::calc_elch_electrode_soc_and_c_rate:
    case ScaTra::Action::calc_elch_elctrode_mean_concentration:
    case ScaTra::Action::calc_elch_domain_kinetics:
    case ScaTra::Action::calc_mass_center_smoothingfunct:
    case ScaTra::Action::get_material_internal_state:
    case ScaTra::Action::set_material_internal_state:
    case ScaTra::Action::get_material_ionic_currents:
    case ScaTra::Action::time_update_material:
    case ScaTra::Action::calc_immersed_element_source:
    case ScaTra::Action::calc_elch_boundary_kinetics_point:
    case ScaTra::Action::micro_scale_initialize:
    case ScaTra::Action::micro_scale_prepare_time_step:
    case ScaTra::Action::micro_scale_solve:
    case ScaTra::Action::micro_scale_update:
    case ScaTra::Action::micro_scale_output:
    case ScaTra::Action::collect_micro_scale_output:
    case ScaTra::Action::micro_scale_read_restart:
    case ScaTra::Action::micro_scale_set_time:
    case ScaTra::Action::calc_heteroreac_mat_and_rhs:
    case ScaTra::Action::calc_mass_matrix:
    case ScaTra::Action::transform_real_to_reference_point:
    case ScaTra::Action::evaluate_field_in_point:
    {
      return ScaTraFactory::provide_impl(
          shape(), impltype_, numdofpernode, numscal, discretization.name())
          ->evaluate_service(
              this, params, discretization, la, elemat1, elemat2, elevec1, elevec2, elevec3);
      break;
    }

    default:
    {
      FOUR_C_THROW("Unknown type of action '{}' for ScaTra", action);
      break;
    }
  }  // switch(action)

  return 0;
}  // Discret::Elements::Transport::Evaluate


/*----------------------------------------------------------------------*
 |  do nothing (public)                                        gjb 01/09|
 |                                                                      |
 |  The function is just a dummy. For the transport elements, the       |
 |  integration of the volume Neumann (body forces) loads takes place   |
 |  in the element. We need it there for the stabilization terms!       |
 *----------------------------------------------------------------------*/
int Discret::Elements::Transport::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  return 0;
}

FOUR_C_NAMESPACE_CLOSE
