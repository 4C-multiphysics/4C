// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_calc_elch_electrode_sti_thermo.hpp"

#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_utils_singleton_owner.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | singleton access method                                   fang 11/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::ELEMENTS::ScaTraEleCalcElchElectrodeSTIThermo<distype>*
Discret::ELEMENTS::ScaTraEleCalcElchElectrodeSTIThermo<distype>::instance(
    const int numdofpernode, const int numscal, const std::string& disname)
{
  static auto singleton_map = Core::Utils::make_singleton_map<std::string>(
      [](const int numdofpernode, const int numscal, const std::string& disname)
      {
        return std::unique_ptr<ScaTraEleCalcElchElectrodeSTIThermo<distype>>(
            new ScaTraEleCalcElchElectrodeSTIThermo<distype>(numdofpernode, numscal, disname));
      });

  return singleton_map[disname].instance(
      Core::Utils::SingletonAction::create, numdofpernode, numscal, disname);
}


/*----------------------------------------------------------------------*
 | extract quantities for element evaluation                 fang 11/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::ELEMENTS::ScaTraEleCalcElchElectrodeSTIThermo<
    distype>::extract_element_and_node_values(Core::Elements::Element* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la)
{
  // call base class routine to extract scatra-related quantities
  myelch::extract_element_and_node_values(ele, params, discretization, la);

  // call base class routine to extract thermo-related quantitites
  mythermo::extract_element_and_node_values(ele, params, discretization, la);
}


/*----------------------------------------------------------------------*
 | get material parameters                                   fang 11/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::ELEMENTS::ScaTraEleCalcElchElectrodeSTIThermo<distype>::get_material_params(
    const Core::Elements::Element* ele, std::vector<double>& densn, std::vector<double>& densnp,
    std::vector<double>& densam, double& visc, const int iquad)
{
  // Set GP values to mat_electrode
  myelectrode::utils()->mat_electrode(
      ele->material(), var_manager()->phinp(0), var_manager()->temp(), myelectrode::diff_manager());

  // get parameters of secondary, thermodynamic electrolyte material
  Teuchos::RCP<const Core::Mat::Material> material = ele->material(1);
  materialtype_ = material->material_type();
  if (materialtype_ == Core::Materials::m_soret) mythermo::mat_soret(material);
}  // Discret::ELEMENTS::ScaTraEleCalcElchElectrodeSTIThermo<distype>::get_material_params


/*--------------------------------------------------------------------------*
 | calculate element matrix and element right-hand side vector   fang 11/15 |
 *--------------------------------------------------------------------------*/

template <Core::FE::CellType distype>
void Discret::ELEMENTS::ScaTraEleCalcElchElectrodeSTIThermo<distype>::calc_mat_and_rhs(
    Core::LinAlg::SerialDenseMatrix& emat, Core::LinAlg::SerialDenseVector& erhs, const int k,
    const double fac, const double timefacfac, const double rhsfac, const double taufac,
    const double timetaufac, const double rhstaufac, Core::LinAlg::Matrix<nen_, 1>& tauderpot,
    double& rhsint)
{
  // call base class routine for isothermal problems
  myelectrode::calc_mat_and_rhs(
      emat, erhs, k, fac, timefacfac, rhsfac, taufac, timetaufac, rhstaufac, tauderpot, rhsint);

  if (materialtype_ == Core::Materials::m_soret)
  {
    // matrix and vector contributions arising from additional, thermodynamic term for Soret effect
    mythermo::calc_mat_soret(emat, timefacfac, var_manager()->phinp(0),
        myelectrode::diff_manager()->get_isotropic_diff(0),
        myelectrode::diff_manager()->get_conc_deriv_iso_diff_coef(0, 0), var_manager()->temp(),
        var_manager()->grad_temp(), my::funct_, my::derxy_);
    mythermo::calc_rhs_soret(erhs, var_manager()->phinp(0),
        myelectrode::diff_manager()->get_isotropic_diff(0), rhsfac, var_manager()->temp(),
        var_manager()->grad_temp(), my::derxy_);
  }
}


/*----------------------------------------------------------------------*
 | evaluate action for off-diagonal system matrix block      fang 11/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::ELEMENTS::ScaTraEleCalcElchElectrodeSTIThermo<distype>::evaluate_action_od(
    Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, const ScaTra::Action& action,
    Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra,
    Core::LinAlg::SerialDenseVector& elevec2_epetra,
    Core::LinAlg::SerialDenseVector& elevec3_epetra)
{
  // determine and evaluate action
  switch (action)
  {
    case ScaTra::Action::calc_scatra_mono_odblock_scatrathermo:
    {
      sysmat_od_scatra_thermo(ele, elemat1_epetra);

      break;
    }

    default:
    {
      // call base class routine
      my::evaluate_action_od(ele, params, discretization, action, la, elemat1_epetra,
          elemat2_epetra, elevec1_epetra, elevec2_epetra, elevec3_epetra);

      break;
    }
  }  // switch(action)

  return 0;
}


/*------------------------------------------------------------------------------------------------------*
 | fill element matrix with linearizations of discrete scatra residuals w.r.t. thermo dofs   fang
 11/15 |
 *------------------------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::ELEMENTS::ScaTraEleCalcElchElectrodeSTIThermo<distype>::sysmat_od_scatra_thermo(
    Core::Elements::Element* ele, Core::LinAlg::SerialDenseMatrix& emat)
{
  // integration points and weights
  Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(ScaTra::DisTypeToOptGaussRule<distype>::rule);

  // loop over integration points
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    // evaluate shape functions, their derivatives, and domain integration factor at current
    // integration point
    const double fac = my::eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

    // evaluate overall integration factor
    const double timefacfac = my::scatraparatimint_->time_fac() * fac;

    // evaluate internal variables at current integration point
    set_internal_variables_for_mat_and_rhs();

    // evaluate material parameters at current integration point
    double dummy(0.);
    std::vector<double> dummyvec(my::numscal_, 0.);
    get_material_params(ele, dummyvec, dummyvec, dummyvec, dummy, iquad);

    // calculating the off diagonal for the temperature derivative of concentration and electric
    // potential
    mythermo::calc_mat_diff_thermo_od(emat, my::numdofpernode_, timefacfac, var_manager()->inv_f(),
        var_manager()->grad_phi(0), var_manager()->grad_pot(),
        myelectrode::diff_manager()->get_temp_deriv_iso_diff_coef(0, 0),
        myelectrode::diff_manager()->get_temp_deriv_cond(0), my::funct_, my::derxy_, 1.);

    if (materialtype_ == Core::Materials::m_soret)
    {
      // provide element matrix with linearizations of Soret term in discrete scatra residuals
      // w.r.t. thermo dofs
      mythermo::calc_mat_soret_od(emat, timefacfac, var_manager()->phinp(0),
          myelectrode::diff_manager()->get_isotropic_diff(0), var_manager()->temp(),
          var_manager()->grad_temp(), my::funct_, my::derxy_);
    }
  }
}


/*------------------------------------------------------------------------------*
 | set internal variables for element evaluation                     fang 11/15 |
 *------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::ELEMENTS::ScaTraEleCalcElchElectrodeSTIThermo<
    distype>::set_internal_variables_for_mat_and_rhs()
{
  // set internal variables for element evaluation
  var_manager()->set_internal_variables(my::funct_, my::derxy_, my::ephinp_, my::ephin_,
      mythermo::etempnp_, my::econvelnp_, my::ehist_);
}

/*----------------------------------------------------------------------*
 | private constructor for singletons                        fang 11/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::ELEMENTS::ScaTraEleCalcElchElectrodeSTIThermo<
    distype>::ScaTraEleCalcElchElectrodeSTIThermo(const int numdofpernode, const int numscal,
    const std::string& disname)
    :  // constructors of base classes
      ScaTraEleCalcElchElectrode<distype>::ScaTraEleCalcElchElectrode(
          numdofpernode, numscal, disname),
      ScaTraEleSTIThermo<distype>::ScaTraEleSTIThermo(numscal)
{
  // safety check
  if (numscal != 1 or numdofpernode != 2)
    FOUR_C_THROW("Invalid number of transported scalars or degrees of freedom per node!");

  // replace internal variable manager for isothermal electrodes by internal variable manager for
  // thermodynamic electrodes
  my::scatravarmanager_ =
      Teuchos::make_rcp<ScaTraEleInternalVariableManagerElchElectrodeSTIThermo<nsd_, nen_>>(
          my::numscal_, myelch::elchparams_);
}


// template classes
// 1D elements
template class Discret::ELEMENTS::ScaTraEleCalcElchElectrodeSTIThermo<Core::FE::CellType::line2>;
template class Discret::ELEMENTS::ScaTraEleCalcElchElectrodeSTIThermo<Core::FE::CellType::line3>;

// 2D elements
template class Discret::ELEMENTS::ScaTraEleCalcElchElectrodeSTIThermo<Core::FE::CellType::tri3>;
template class Discret::ELEMENTS::ScaTraEleCalcElchElectrodeSTIThermo<Core::FE::CellType::tri6>;
template class Discret::ELEMENTS::ScaTraEleCalcElchElectrodeSTIThermo<Core::FE::CellType::quad4>;
// template class
// Discret::ELEMENTS::ScaTraEleCalcElchElectrodeSTIThermo<Core::FE::CellType::quad8>;
template class Discret::ELEMENTS::ScaTraEleCalcElchElectrodeSTIThermo<Core::FE::CellType::quad9>;
template class Discret::ELEMENTS::ScaTraEleCalcElchElectrodeSTIThermo<Core::FE::CellType::nurbs9>;

// 3D elements
template class Discret::ELEMENTS::ScaTraEleCalcElchElectrodeSTIThermo<Core::FE::CellType::hex8>;
// template class
// Discret::ELEMENTS::ScaTraEleCalcElchElectrodeSTIThermo<Core::FE::CellType::hex20>;
template class Discret::ELEMENTS::ScaTraEleCalcElchElectrodeSTIThermo<Core::FE::CellType::hex27>;
template class Discret::ELEMENTS::ScaTraEleCalcElchElectrodeSTIThermo<Core::FE::CellType::tet4>;
template class Discret::ELEMENTS::ScaTraEleCalcElchElectrodeSTIThermo<Core::FE::CellType::tet10>;
// template class
// Discret::ELEMENTS::ScaTraEleCalcElchElectrodeSTIThermo<Core::FE::CellType::wedge6>;
template class Discret::ELEMENTS::ScaTraEleCalcElchElectrodeSTIThermo<Core::FE::CellType::pyramid5>;
// template class
// Discret::ELEMENTS::ScaTraEleCalcElchElectrodeSTIThermo<Core::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE
