// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_boundary_calc_sti_electrode.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_utils_boundary_integration.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_inpar_s2i.hpp"
#include "4C_mat_electrode.hpp"
#include "4C_mat_fourier.hpp"
#include "4C_mat_soret.hpp"
#include "4C_scatra_ele_boundary_calc_elch_electrode_utils.hpp"
#include "4C_scatra_ele_parameter_boundary.hpp"
#include "4C_scatra_ele_parameter_elch.hpp"
#include "4C_scatra_ele_parameter_std.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_utils_singleton_owner.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::Elements::ScaTraEleBoundaryCalcSTIElectrode<distype, probdim>*
Discret::Elements::ScaTraEleBoundaryCalcSTIElectrode<distype, probdim>::instance(
    const int numdofpernode, const int numscal, const std::string& disname)
{
  static auto singleton_map = Core::Utils::make_singleton_map<std::string>(
      [](const int numdofpernode, const int numscal, const std::string& disname)
      {
        return std::unique_ptr<ScaTraEleBoundaryCalcSTIElectrode<distype, probdim>>(
            new ScaTraEleBoundaryCalcSTIElectrode<distype, probdim>(
                numdofpernode, numscal, disname));
      });

  return singleton_map[disname].instance(
      Core::Utils::SingletonAction::create, numdofpernode, numscal, disname);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::Elements::ScaTraEleBoundaryCalcSTIElectrode<distype,
    probdim>::ScaTraEleBoundaryCalcSTIElectrode(const int numdofpernode, const int numscal,
    const std::string& disname)
    :  // constructor of base class
      my::ScaTraEleBoundaryCalc(numdofpernode, numscal, disname),

      // initialize member variable
      eelchnp_(2, Core::LinAlg::Matrix<nen_, 1>(Core::LinAlg::Initialization::zero))
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalcSTIElectrode<distype, probdim>::evaluate_s2_i_coupling(
    const Core::Elements::FaceElement* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& eslavematrix, Core::LinAlg::SerialDenseMatrix& emastermatrix,
    Core::LinAlg::SerialDenseVector& eslaveresidual)
{
  // access primary and secondary materials of parent element
  std::shared_ptr<const Mat::Soret> matsoret =
      std::dynamic_pointer_cast<const Mat::Soret>(ele->parent_element()->material());
  std::shared_ptr<const Mat::Fourier> matfourier =
      std::dynamic_pointer_cast<const Mat::Fourier>(ele->parent_element()->material());
  std::shared_ptr<const Mat::Electrode> matelectrode =
      std::dynamic_pointer_cast<const Mat::Electrode>(ele->parent_element()->material(1));
  if ((matsoret == nullptr and matfourier == nullptr) or matelectrode == nullptr)
    FOUR_C_THROW("Invalid electrode material for scatra-scatra interface coupling!");

  // extract local nodal values on present and opposite side of scatra-scatra interface
  extract_node_values(discretization, la);
  std::vector<Core::LinAlg::Matrix<nen_, 1>> emasterscatra(
      2, Core::LinAlg::Matrix<nen_, 1>(Core::LinAlg::Initialization::zero));
  my::extract_node_values(
      emasterscatra, discretization, la, "imasterscatra", my::scatraparams_->nds_scatra());

  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  const int kineticmodel = my::scatraparamsboundary_->kinetic_model();

  Core::LinAlg::Matrix<nen_, 1> emastertemp(Core::LinAlg::Initialization::zero);
  if (kineticmodel == Inpar::S2I::kinetics_butlervolmerreducedthermoresistance)
    my::extract_node_values(emastertemp, discretization, la, "imastertemp", 3);

  // element slave mechanical stress tensor
  const bool is_pseudo_contact = my::scatraparamsboundary_->is_pseudo_contact();
  std::vector<Core::LinAlg::Matrix<nen_, 1>> eslavestress_vector(
      6, Core::LinAlg::Matrix<nen_, 1>(Core::LinAlg::Initialization::zero));
  if (is_pseudo_contact)
    my::extract_node_values(eslavestress_vector, discretization, la, "mechanicalStressState",
        my::scatraparams_->nds_two_tensor_quantity());

  Core::LinAlg::Matrix<nsd_, 1> normal;

  // loop over integration points
  for (int gpid = 0; gpid < intpoints.ip().nquad; ++gpid)
  {
    // evaluate values of shape functions and domain integration factor at current integration point
    const double fac = my::eval_shape_func_and_int_fac(intpoints, gpid, &normal);
    const double detF = my::calculate_det_f_of_parent_element(ele, intpoints.point(gpid));

    const double pseudo_contact_fac = my::calculate_pseudo_contact_factor(
        is_pseudo_contact, eslavestress_vector, normal, my::funct_);

    // evaluate overall integration factors
    const double timefacfac = my::scatraparamstimint_->time_fac() * fac;
    const double timefacrhsfac = my::scatraparamstimint_->time_fac_rhs() * fac;
    if (timefacfac < 0. or timefacrhsfac < 0.) FOUR_C_THROW("Integration factor is negative!");

    evaluate_s2_i_coupling_at_integration_point<distype>(*matelectrode, my::ephinp_[0], emastertemp,
        eelchnp_, emasterscatra, pseudo_contact_fac, my::funct_, my::funct_,
        my::scatraparamsboundary_, timefacfac, timefacrhsfac, detF, eslavematrix, emastermatrix,
        eslaveresidual);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
template <Core::FE::CellType distype_master>
void Discret::Elements::ScaTraEleBoundaryCalcSTIElectrode<distype,
    probdim>::evaluate_s2_i_coupling_at_integration_point(const Mat::Electrode& matelectrode,
    const Core::LinAlg::Matrix<nen_, 1>& eslavetempnp,
    const Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>& emastertempnp,
    const std::vector<Core::LinAlg::Matrix<nen_, 1>>& eslavephinp,
    const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>>& emasterphinp,
    const double pseudo_contact_fac, const Core::LinAlg::Matrix<nen_, 1>& funct_slave,
    const Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>& funct_master,
    const Discret::Elements::ScaTraEleParameterBoundary* const scatra_parameter_boundary,
    const double timefacfac, const double timefacrhsfac, const double detF,
    Core::LinAlg::SerialDenseMatrix& k_ss, Core::LinAlg::SerialDenseMatrix& k_sm,
    Core::LinAlg::SerialDenseVector& r_s)
{
  // get condition specific parameters
  const int kineticmodel = scatra_parameter_boundary->kinetic_model();
  const double kr = scatra_parameter_boundary->charge_transfer_constant();
  const double alphaa = scatra_parameter_boundary->alphadata();
  const double alphac = scatra_parameter_boundary->alpha_c();
  const double peltier = scatra_parameter_boundary->peltier();
  const double thermoperm = scatra_parameter_boundary->thermo_perm();
  const double molar_heat_capacity = scatra_parameter_boundary->molar_heat_capacity();

  // evaluate dof values at current integration point on present and opposite side of scatra-scatra
  // interface
  const double eslavetempint = funct_slave.dot(eslavetempnp);
  const double emastertempint = funct_master.dot(emastertempnp);
  if (eslavetempint <= 0.) FOUR_C_THROW("Temperature is non-positive!");
  const double eslavephiint = funct_slave.dot(eslavephinp[0]);
  const double eslavepotint = funct_slave.dot(eslavephinp[1]);
  const double emasterphiint = funct_master.dot(emasterphinp[0]);
  const double emasterpotint = funct_master.dot(emasterphinp[1]);

  const int nen_master = Core::FE::num_nodes<distype_master>;

  // access input parameters associated with current condition
  const double faraday = Discret::Elements::ScaTraEleParameterElch::instance("scatra")->faraday();
  const double gasconstant =
      Discret::Elements::ScaTraEleParameterElch::instance("scatra")->gas_constant();

  // compute matrix and vector contributions according to kinetic model for current scatra-scatra
  // interface coupling condition
  switch (kineticmodel)
  {
    // Butler-Volmer-Peltier kinetics
    case Inpar::S2I::kinetics_butlervolmerpeltier:
    {
      // extract saturation value of intercalated lithium concentration from electrode material
      const double cmax = matelectrode.c_max();

      // evaluate factor F/RT
      const double frt = faraday / (gasconstant * eslavetempint);

      // equilibrium electric potential difference at electrode surface
      const double epd =
          matelectrode.compute_open_circuit_potential(eslavephiint, faraday, frt, detF);

      // electrode-electrolyte overpotential at integration point
      const double eta = eslavepotint - emasterpotint - epd;

      // Butler-Volmer exchange current density
      const double i0 = kr * faraday * pow(emasterphiint, alphaa) *
                        pow(cmax - eslavephiint, alphaa) * pow(eslavephiint, alphac);

      // exponential Butler-Volmer terms
      const double expterm1 = exp(alphaa * frt * eta);
      const double expterm2 = exp(-alphac * frt * eta);
      const double expterm = expterm1 - expterm2;

      // core residual term
      const double residual_timefacrhsfac =
          pseudo_contact_fac * timefacrhsfac * i0 * expterm * (eta + peltier);

      // core linearization w.r.t. temperature
      const double linearization_timefacfac =
          -pseudo_contact_fac * timefacfac * i0 * frt / eslavetempint * eta *
          (alphaa * expterm1 + alphac * expterm2) * (eta + peltier);

      // compute matrix and vector contributions
      for (int vi = 0; vi < nen_; ++vi)
      {
        for (int ui = 0; ui < nen_; ++ui)
          k_ss(vi, ui) -= funct_slave(vi) * linearization_timefacfac * funct_slave(ui);
        r_s[vi] += funct_slave(vi) * residual_timefacrhsfac;
      }

      break;
    }
    case Inpar::S2I::kinetics_butlervolmerreducedthermoresistance:
    {
      // Flux of energy from mass flux and from difference in temperature
      double j_timefacrhsfac(0.0);
      double djdT_slave_timefacfac(0.0);
      double djdT_master_timefacfac(0.0);

      // Part 1: Energy flux from mass flux
      const double etempint = (eslavetempint + emastertempint) / 2.0;
      const double frt = faraday / (etempint * gasconstant);

      // equilibrium electric potential difference at electrode surface
      const double epd =
          matelectrode.compute_open_circuit_potential(eslavephiint, faraday, frt, detF);
      const double depddT = matelectrode.compute_d_open_circuit_potential_d_temperature(
          eslavephiint, faraday, gasconstant);

      // skip further computation in case equilibrium electric potential difference is outside
      // physically meaningful range
      if (std::isinf(epd)) break;

      // Butler-Volmer exchange mass flux density
      const double j0 = kr;

      // electrode-electrolyte overpotential at integration point
      const double eta = eslavepotint - emasterpotint - epd;

      // exponential Butler-Volmer terms
      const double expterm1 = std::exp(alphaa * frt * eta);
      const double expterm2 = std::exp(-alphac * frt * eta);
      const double expterm = expterm1 - expterm2;

      // core residual term associated with Butler-Volmer mass flux density
      const double j_mass = j0 * expterm;

      // temperature from source side of flux
      const double j_mass_energy = j_mass * molar_heat_capacity * etempint;

      j_timefacrhsfac += pseudo_contact_fac * timefacrhsfac * j_mass_energy;

      // forward declarations
      double dj_dT_slave(0.0);

      // calculate linearizations of Butler-Volmer kinetics w.r.t. temperature dofs
      calculate_butler_volmer_temp_linearizations(
          alphaa, alphac, depddT, eta, etempint, faraday, frt, gasconstant, j0, dj_dT_slave);

      const double dj_mass_energydT_slave =
          dj_dT_slave * molar_heat_capacity * etempint * 0.5 + j0 * expterm * 0.5;
      const double dj_mass_energydT_master =
          dj_dT_slave * molar_heat_capacity * etempint * 0.5 - j0 * expterm * 0.5;

      djdT_slave_timefacfac += pseudo_contact_fac * dj_mass_energydT_slave * timefacfac;
      djdT_master_timefacfac += pseudo_contact_fac * dj_mass_energydT_master * timefacfac;

      // Part 2: Energy flux from temperature drop
      // core residual
      j_timefacrhsfac +=
          pseudo_contact_fac * timefacrhsfac * thermoperm * (eslavetempint - emastertempint);

      // core linearizations
      djdT_slave_timefacfac += pseudo_contact_fac * timefacfac * thermoperm;
      djdT_master_timefacfac -= pseudo_contact_fac * timefacfac * thermoperm;

      for (int vi = 0; vi < nen_; ++vi)
      {
        for (int ui = 0; ui < nen_; ++ui)
          k_ss(vi, ui) += funct_slave(vi) * djdT_slave_timefacfac * funct_slave(ui);
        r_s[vi] -= funct_slave(vi) * j_timefacrhsfac;

        for (int ui = 0; ui < nen_master; ++ui)
          k_sm(vi, ui) += funct_slave(vi) * djdT_master_timefacfac * funct_master(ui);
      }

      break;
    }
    case Inpar::S2I::kinetics_butlervolmerreduced:
    case Inpar::S2I::kinetics_constantinterfaceresistance:
    case Inpar::S2I::kinetics_nointerfaceflux:
    {
      // do nothing
      break;
    }

    default:
    {
      FOUR_C_THROW("Kinetic model for scatra-scatra interface coupling is not yet implemented!");
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalcSTIElectrode<distype,
    probdim>::evaluate_s2_i_coupling_od(const Core::Elements::FaceElement* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& eslavematrix,
    Core::LinAlg::SerialDenseMatrix& emastermatrix)
{
  // access primary and secondary materials of parent element
  std::shared_ptr<const Mat::Soret> matsoret =
      std::dynamic_pointer_cast<const Mat::Soret>(ele->parent_element()->material());
  std::shared_ptr<const Mat::Fourier> matfourier =
      std::dynamic_pointer_cast<const Mat::Fourier>(ele->parent_element()->material());
  std::shared_ptr<const Mat::Electrode> matelectrode =
      std::dynamic_pointer_cast<const Mat::Electrode>(ele->parent_element()->material(1));
  if ((matsoret == nullptr and matfourier == nullptr) or matelectrode == nullptr)
    FOUR_C_THROW("Invalid electrode material for scatra-scatra interface coupling!");

  // extract local nodal values on present and opposite side of scatra-scatra interface
  extract_node_values(discretization, la);
  std::vector<Core::LinAlg::Matrix<nen_, 1>> emasterscatra(
      2, Core::LinAlg::Matrix<nen_, 1>(Core::LinAlg::Initialization::zero));
  my::extract_node_values(
      emasterscatra, discretization, la, "imasterscatra", my::scatraparams_->nds_scatra());

  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  const int kineticmodel = my::scatraparamsboundary_->kinetic_model();

  Core::LinAlg::Matrix<nen_, 1> emastertemp(Core::LinAlg::Initialization::zero);
  if (kineticmodel == Inpar::S2I::kinetics_butlervolmerreducedthermoresistance)
    my::extract_node_values(emastertemp, discretization, la, "imastertemp", 3);

  // get primary variable to derive the linearization
  const auto differentiationtype =
      Teuchos::getIntegralValue<ScaTra::DifferentiationType>(params, "differentiationtype");

  // element slave mechanical stress tensor
  const bool is_pseudo_contact = my::scatraparamsboundary_->is_pseudo_contact();
  std::vector<Core::LinAlg::Matrix<nen_, 1>> eslavestress_vector(
      6, Core::LinAlg::Matrix<nen_, 1>(Core::LinAlg::Initialization::zero));
  if (is_pseudo_contact)
    my::extract_node_values(eslavestress_vector, discretization, la, "mechanicalStressState",
        my::scatraparams_->nds_two_tensor_quantity());

  Core::LinAlg::Matrix<nsd_, 1> normal;

  // loop over integration points
  for (int gpid = 0; gpid < intpoints.ip().nquad; ++gpid)
  {
    // evaluate values of shape functions and domain integration factor at current integration point
    const double fac = my::eval_shape_func_and_int_fac(intpoints, gpid, &normal);
    const double detF = my::calculate_det_f_of_parent_element(ele, intpoints.point(gpid));

    const double pseudo_contact_fac = my::calculate_pseudo_contact_factor(
        is_pseudo_contact, eslavestress_vector, normal, my::funct_);

    // evaluate overall integration factor
    const double timefacfac = my::scatraparamstimint_->time_fac() * fac;
    if (timefacfac < 0.0) FOUR_C_THROW("Integration factor is negative!");

    const double timefacwgt = my::scatraparamstimint_->time_fac() * intpoints.ip().qwgt[gpid];

    static Core::LinAlg::Matrix<nsd_, nen_> dsqrtdetg_dd;
    if (differentiationtype == ScaTra::DifferentiationType::disp)
    {
      static Core::LinAlg::Matrix<nen_, nsd_> xyze_transposed;
      xyze_transposed.update_t(my::xyze_);
      Core::FE::evaluate_shape_function_spatial_derivative_in_prob_dim<distype, nsd_>(
          my::derxy_, my::deriv_, xyze_transposed, normal);
      my::evaluate_spatial_derivative_of_area_integration_factor(intpoints, gpid, dsqrtdetg_dd);
    }

    evaluate_s2_i_coupling_od_at_integration_point<distype>(*matelectrode, my::ephinp_[0],
        emastertemp, eelchnp_, emasterscatra, pseudo_contact_fac, my::funct_, my::funct_,
        my::scatraparamsboundary_, timefacfac, timefacwgt, detF, differentiationtype, dsqrtdetg_dd,
        my::derxy_, eslavematrix, emastermatrix);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
template <Core::FE::CellType distype_master>
void Discret::Elements::ScaTraEleBoundaryCalcSTIElectrode<distype,
    probdim>::evaluate_s2_i_coupling_od_at_integration_point(const Mat::Electrode& matelectrode,
    const Core::LinAlg::Matrix<nen_, 1>& eslavetempnp,
    const Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>& emastertempnp,
    const std::vector<Core::LinAlg::Matrix<nen_, 1>>& eslavephinp,
    const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>>& emasterphinp,
    const double pseudo_contact_fac, const Core::LinAlg::Matrix<nen_, 1>& funct_slave,
    const Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>& funct_master,
    const Discret::Elements::ScaTraEleParameterBoundary* const scatra_parameter_boundary,
    const double timefacfac, const double timefacwgt, const double detF,
    const ScaTra::DifferentiationType differentiationtype,
    const Core::LinAlg::Matrix<nsd_, nen_>& dsqrtdetg_dd,
    const Core::LinAlg::Matrix<nsd_, nen_>& shape_spatial_derivatives,
    Core::LinAlg::SerialDenseMatrix& k_ss, Core::LinAlg::SerialDenseMatrix& k_sm)
{
  // get condition specific parameters
  const int kineticmodel = scatra_parameter_boundary->kinetic_model();
  const double kr = scatra_parameter_boundary->charge_transfer_constant();
  const double alphaa = scatra_parameter_boundary->alphadata();
  const double alphac = scatra_parameter_boundary->alpha_c();
  const double peltier = scatra_parameter_boundary->peltier();
  const double thermoperm = scatra_parameter_boundary->thermo_perm();
  const double molar_heat_capacity = scatra_parameter_boundary->molar_heat_capacity();

  // number of nodes of master-side element
  const int nen_master = Core::FE::num_nodes<distype_master>;

  // evaluate dof values at current integration point on present and opposite side of scatra-scatra
  // interface
  const double eslavetempint = funct_slave.dot(eslavetempnp);
  if (eslavetempint <= 0.0) FOUR_C_THROW("Temperature is non-positive!");
  const double emastertempint = funct_master.dot(emastertempnp);
  const double eslavephiint = funct_slave.dot(eslavephinp[0]);
  const double eslavepotint = funct_slave.dot(eslavephinp[1]);
  const double emasterphiint = funct_master.dot(emasterphinp[0]);
  const double emasterpotint = funct_master.dot(emasterphinp[1]);

  // compute derivatives of scatra-scatra interface coupling residuals w.r.t. concentration and
  // electric potential according to kinetic model for current thermo-thermo interface coupling
  // condition
  switch (kineticmodel)
  {
    // Butler-Volmer-Peltier kinetics
    case Inpar::S2I::kinetics_butlervolmerpeltier:
    {
      switch (differentiationtype)
      {
        case ScaTra::DifferentiationType::elch:
        {
          // access input parameters associated with current condition
          const double faraday =
              Discret::Elements::ScaTraEleParameterElch::instance("scatra")->faraday();
          Discret::Elements::ScaTraEleParameterElch::instance("scatra")->faraday();
          const double gasconstant =
              Discret::Elements::ScaTraEleParameterElch::instance("scatra")->gas_constant();

          // extract saturation value of intercalated lithium concentration from electrode material
          const double cmax = matelectrode.c_max();

          // evaluate factor F/RT
          const double frt = faraday / (gasconstant * eslavetempint);

          // equilibrium electric potential difference at electrode surface and its derivative
          // w.r.t. concentration at electrode surface
          const double epd =
              matelectrode.compute_open_circuit_potential(eslavephiint, faraday, frt, detF);
          const double epdderiv = matelectrode.compute_d_open_circuit_potential_d_concentration(
              eslavephiint, faraday, frt, detF);

          // electrode-electrolyte overpotential at integration point
          const double eta = eslavepotint - emasterpotint - epd;

          // Butler-Volmer exchange current density
          const double i0 = kr * faraday * std::pow(emasterphiint, alphaa) *
                            std::pow(cmax - eslavephiint, alphaa) * std::pow(eslavephiint, alphac);

          // exponential Butler-Volmer terms
          const double expterm1 = std::exp(alphaa * frt * eta);
          const double expterm2 = std::exp(-alphac * frt * eta);
          const double expterm = expterm1 - expterm2;

          // core linearizations w.r.t. master-side and slave-side concentrations and electric
          // potentials
          const double dres_dc_slave =
              (kr * faraday * std::pow(emasterphiint, alphaa) *
                      std::pow(cmax - eslavephiint, alphaa - 1.) *
                      std::pow(eslavephiint, alphac - 1.) *
                      (-alphaa * eslavephiint + alphac * (cmax - eslavephiint)) * expterm +
                  i0 * (-alphaa * frt * epdderiv * expterm1 - alphac * frt * epdderiv * expterm2)) *
                  (eta + peltier) -
              i0 * expterm * epdderiv;
          const double dres_dc_slave_timefacfac = pseudo_contact_fac * timefacfac * dres_dc_slave;

          const double dres_dc_master = i0 * alphaa / emasterphiint * expterm * (eta + peltier);
          const double dres_dc_master_timefacfac = pseudo_contact_fac * timefacfac * dres_dc_master;

          const double dres_dpot_slave =
              i0 * frt * (alphaa * expterm1 + alphac * expterm2) * (eta + peltier) + i0 * expterm;
          const double dres_dpot_slave_timefacfac =
              pseudo_contact_fac * timefacfac * dres_dpot_slave;

          const double dres_dpot_master_timefacfac = -dres_dpot_slave_timefacfac;

          // compute matrix contributions associated with slave-side residuals
          for (int vi = 0; vi < nen_; ++vi)
          {
            for (int ui = 0; ui < nen_; ++ui)
            {
              // compute linearizations w.r.t. slave-side concentrations
              k_ss(vi, ui * 2) -= funct_slave(vi) * dres_dc_slave_timefacfac * funct_slave(ui);

              // compute linearizations w.r.t. slave-side electric potentials
              k_ss(vi, ui * 2 + 1) -=
                  funct_slave(vi) * dres_dpot_slave_timefacfac * funct_slave(ui);
            }

            for (int ui = 0; ui < nen_master; ++ui)
            {
              // compute linearizations w.r.t. master-side concentrations
              k_sm(vi, ui * 2) -= funct_slave(vi) * dres_dc_master_timefacfac * funct_master(ui);

              // compute linearizations w.r.t. master-side electric potentials
              k_sm(vi, ui * 2 + 1) -=
                  funct_slave(vi) * dres_dpot_master_timefacfac * funct_master(ui);
            }
          }
          break;
        }
        default:
        {
          FOUR_C_THROW("Unknown primary quantity to calculate derivative");
        }
      }
      break;
    }
    case Inpar::S2I::kinetics_butlervolmerreducedthermoresistance:
    {
      switch (differentiationtype)
      {
        case ScaTra::DifferentiationType::disp:
        {
          // Part 1
          // compute factor F/(RT)
          const double gasconstant =
              Discret::Elements::ScaTraEleParameterElch::instance("scatra")->gas_constant();
          const double faraday =
              Discret::Elements::ScaTraEleParameterElch::instance("scatra")->faraday();
          const double etempint = (eslavetempint + emastertempint) / 2;
          const double frt = faraday / (etempint * gasconstant);

          // equilibrium electric potential difference at electrode surface
          const double epd =
              matelectrode.compute_open_circuit_potential(eslavephiint, faraday, frt, detF);

          // skip further computation in case equilibrium electric potential difference is outside
          // physically meaningful range
          if (not std::isinf(epd))
          {
            const double depd_ddetF = matelectrode.compute_d_open_circuit_potential_d_det_f(
                eslavephiint, faraday, frt, detF);

            // electrode-electrolyte overpotential at integration point
            const double eta = eslavepotint - emasterpotint - epd;

            // Butler-Volmer exchange mass flux density
            const double j0 = kr;

            // exponential Butler-Volmer terms
            const double expterm1 = std::exp(alphaa * frt * eta);
            const double expterm2 = std::exp(-alphac * frt * eta);
            const double expterm = expterm1 - expterm2;

            // core linearization associated with Butler-Volmer mass flux density
            const double dj_dsqrtdetg_timefacwgt =
                pseudo_contact_fac * timefacwgt * j0 * expterm * molar_heat_capacity * etempint;
            const double dj_depd = -j0 * frt * (alphaa * expterm1 + alphac * expterm2) *
                                   molar_heat_capacity * etempint;
            const double dj_ddetF = dj_depd * depd_ddetF;
            const double dj_ddetF_timefacfac = pseudo_contact_fac * dj_ddetF * timefacfac;

            // loop over matrix columns
            for (int ui = 0; ui < nen_; ++ui)
            {
              const int fui = ui * 3;

              // loop over matrix rows
              for (int vi = 0; vi < nen_; ++vi)
              {
                const double vi_dj_dsqrtdetg = funct_slave(vi) * dj_dsqrtdetg_timefacwgt;
                const double vi_dj_ddetF = funct_slave(vi) * dj_ddetF_timefacfac;

                // loop over spatial dimensions
                for (int dim = 0; dim < 3; ++dim)
                {
                  // compute linearizations w.r.t. slave-side structural displacements
                  k_ss(vi, fui + dim) += vi_dj_dsqrtdetg * dsqrtdetg_dd(dim, ui);
                  k_ss(vi, fui + dim) += vi_dj_ddetF * detF * shape_spatial_derivatives(dim, ui);
                }
              }
            }
          }
          // Part 2
          const double dj_dsqrtdetg_timefacwgt =
              pseudo_contact_fac * timefacwgt * (eslavetempint - emastertempint) * thermoperm;

          // loop over matrix columns
          for (int ui = 0; ui < nen_; ++ui)
          {
            const int fui = ui * 3;

            // loop over matrix rows
            for (int vi = 0; vi < nen_; ++vi)
            {
              const double vi_dj_dsqrtdetg = funct_slave(vi) * dj_dsqrtdetg_timefacwgt;

              // loop over spatial dimensions
              for (int dim = 0; dim < 3; ++dim)
              {
                // finalize linearizations w.r.t. slave-side structural displacements
                k_ss(vi, fui + dim) += vi_dj_dsqrtdetg * dsqrtdetg_dd(dim, ui);
              }
            }
          }
          break;
        }
        case ScaTra::DifferentiationType::elch:
        {
          const double gasconstant =
              Discret::Elements::ScaTraEleParameterElch::instance("scatra")->gas_constant();
          const double faraday =
              Discret::Elements::ScaTraEleParameterElch::instance("scatra")->faraday();
          const double etempint = (eslavetempint + emastertempint) / 2;
          const double frt = faraday / (etempint * gasconstant);

          const double epd =
              matelectrode.compute_open_circuit_potential(eslavephiint, faraday, frt, detF);

          const double epdderiv = matelectrode.compute_d_open_circuit_potential_d_concentration(
              eslavephiint, faraday, frt, detF);

          const double cmax = matelectrode.c_max();

          if (not std::isinf(epd))
          {
            const double eta = eslavepotint - emasterpotint - epd;

            const double expterm1 = std::exp(alphaa * frt * eta);
            const double expterm2 = std::exp(-alphac * frt * eta);

            // Butler-Volmer exchange mass flux density
            const double j0 = kr;

            // forward declarations
            double dj_dc_slave(0.0);
            double dj_dc_master(0.0);
            double dj_dpot_slave(0.0);
            double dj_dpot_master(0.0);

            // calculate linearizations of Butler-Volmer kinetics w.r.t. elch dofs
            calculate_butler_volmer_elch_linearizations(kineticmodel, j0, frt, epdderiv, alphaa,
                alphac, 0.0, expterm1, expterm2, kr, faraday, emasterphiint, eslavephiint, cmax,
                eta, dj_dc_slave, dj_dc_master, dj_dpot_slave, dj_dpot_master);

            const double dj_energydc_slave =
                pseudo_contact_fac * dj_dc_slave * molar_heat_capacity * etempint;
            const double dj_energydpot_slave =
                pseudo_contact_fac * dj_dpot_slave * molar_heat_capacity * etempint;
            const double dj_energydc_master =
                pseudo_contact_fac * dj_dc_master * molar_heat_capacity * etempint;
            const double dj_energydpot_master =
                pseudo_contact_fac * dj_dpot_master * molar_heat_capacity * etempint;
            // compute matrix contributions associated with slave-side residuals
            for (int vi = 0; vi < nen_; ++vi)
            {
              for (int ui = 0; ui < nen_; ++ui)
              {
                k_ss(vi, ui * 2) -= funct_slave(vi) * dj_energydc_slave * funct_slave(ui);
                // compute linearizations w.r.t. slave-side electric potentials
                k_ss(vi, ui * 2 + 1) -= funct_slave(vi) * dj_energydpot_slave * funct_slave(ui);
              }

              for (int ui = 0; ui < nen_master; ++ui)
              {
                k_sm(vi, ui * 2) -= funct_slave(vi) * dj_energydc_master * funct_master(ui);
                // compute linearizations w.r.t. master-side electric potentials
                k_sm(vi, ui * 2 + 1) -= funct_slave(vi) * dj_energydpot_master * funct_master(ui);
              }
            }
          }
          break;
        }
        default:
        {
          FOUR_C_THROW("Unknown type of primary variable");
        }
      }
      break;
    }
    case Inpar::S2I::kinetics_butlervolmerreduced:
    case Inpar::S2I::kinetics_constantinterfaceresistance:
    case Inpar::S2I::kinetics_nointerfaceflux:
    {
      // do nothing
      break;
    }

    default:
    {
      FOUR_C_THROW("Kinetic model for scatra-scatra interface coupling is not yet implemented!");
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleBoundaryCalcSTIElectrode<distype, probdim>::evaluate_action(
    Core::Elements::FaceElement* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, ScaTra::BoundaryAction action,
    Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra,
    Core::LinAlg::SerialDenseVector& elevec2_epetra,
    Core::LinAlg::SerialDenseVector& elevec3_epetra)
{
  // determine and evaluate action
  switch (action)
  {
    case ScaTra::BoundaryAction::calc_s2icoupling:
    {
      evaluate_s2_i_coupling(
          ele, params, discretization, la, elemat1_epetra, elemat2_epetra, elevec1_epetra);
      break;
    }

    case ScaTra::BoundaryAction::calc_s2icoupling_od:
    {
      evaluate_s2_i_coupling_od(ele, params, discretization, la, elemat1_epetra, elemat2_epetra);
      break;
    }

    default:
    {
      my::evaluate_action(ele, params, discretization, action, la, elemat1_epetra, elemat2_epetra,
          elevec1_epetra, elevec2_epetra, elevec3_epetra);
      break;
    }
  }

  return 0;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalcSTIElectrode<distype, probdim>::extract_node_values(
    const Core::FE::Discretization& discretization, Core::Elements::LocationArray& la)
{
  // call base class routine
  my::extract_node_values(discretization, la);

  // extract nodal electrochemistry variables associated with time t_{n+1} or t_{n+alpha_f}
  my::extract_node_values(eelchnp_, discretization, la, "scatra", my::scatraparams_->nds_scatra());
}


// template classes
template class Discret::Elements::ScaTraEleBoundaryCalcSTIElectrode<Core::FE::CellType::quad4, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcSTIElectrode<Core::FE::CellType::quad8, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcSTIElectrode<Core::FE::CellType::quad9, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcSTIElectrode<Core::FE::CellType::tri3, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcSTIElectrode<Core::FE::CellType::tri6, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcSTIElectrode<Core::FE::CellType::line2, 2>;
template class Discret::Elements::ScaTraEleBoundaryCalcSTIElectrode<Core::FE::CellType::line2, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcSTIElectrode<Core::FE::CellType::line3, 2>;
template class Discret::Elements::ScaTraEleBoundaryCalcSTIElectrode<Core::FE::CellType::nurbs3, 2>;
template class Discret::Elements::ScaTraEleBoundaryCalcSTIElectrode<Core::FE::CellType::nurbs9, 3>;

// explicit instantiation of template methods
template void Discret::Elements::ScaTraEleBoundaryCalcSTIElectrode<Core::FE::CellType::quad4>::
    evaluate_s2_i_coupling_at_integration_point<Core::FE::CellType::quad4>(const Mat::Electrode&,
        const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>&,
        const std::vector<Core::LinAlg::Matrix<nen_, 1>>&,
        const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>>&,
        const double, const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>&,
        const Discret::Elements::ScaTraEleParameterBoundary* const, const double, const double,
        const double, Core::LinAlg::SerialDenseMatrix&, Core::LinAlg::SerialDenseMatrix&,
        Core::LinAlg::SerialDenseVector&);
template void Discret::Elements::ScaTraEleBoundaryCalcSTIElectrode<Core::FE::CellType::quad4>::
    evaluate_s2_i_coupling_at_integration_point<Core::FE::CellType::tri3>(const Mat::Electrode&,
        const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>&,
        const std::vector<Core::LinAlg::Matrix<nen_, 1>>&,
        const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>>&,
        const double, const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>&,
        const Discret::Elements::ScaTraEleParameterBoundary* const, const double, const double,
        const double, Core::LinAlg::SerialDenseMatrix&, Core::LinAlg::SerialDenseMatrix&,
        Core::LinAlg::SerialDenseVector&);
template void Discret::Elements::ScaTraEleBoundaryCalcSTIElectrode<Core::FE::CellType::tri3>::
    evaluate_s2_i_coupling_at_integration_point<Core::FE::CellType::quad4>(const Mat::Electrode&,
        const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>&,
        const std::vector<Core::LinAlg::Matrix<nen_, 1>>&,
        const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>>&,
        const double, const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>&,
        const Discret::Elements::ScaTraEleParameterBoundary* const, const double, const double,
        const double, Core::LinAlg::SerialDenseMatrix&, Core::LinAlg::SerialDenseMatrix&,
        Core::LinAlg::SerialDenseVector&);
template void Discret::Elements::ScaTraEleBoundaryCalcSTIElectrode<Core::FE::CellType::tri3>::
    evaluate_s2_i_coupling_at_integration_point<Core::FE::CellType::tri3>(const Mat::Electrode&,
        const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>&,
        const std::vector<Core::LinAlg::Matrix<nen_, 1>>&,
        const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>>&,
        const double, const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>&,
        const Discret::Elements::ScaTraEleParameterBoundary* const, const double, const double,
        const double, Core::LinAlg::SerialDenseMatrix&, Core::LinAlg::SerialDenseMatrix&,
        Core::LinAlg::SerialDenseVector&);
template void Discret::Elements::ScaTraEleBoundaryCalcSTIElectrode<Core::FE::CellType::quad4>::
    evaluate_s2_i_coupling_od_at_integration_point<Core::FE::CellType::quad4>(const Mat::Electrode&,
        const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>&,
        const std::vector<Core::LinAlg::Matrix<nen_, 1>>&,
        const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>>&,
        const double, const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>&,
        const Discret::Elements::ScaTraEleParameterBoundary* const, const double, const double,
        const double, const ScaTra::DifferentiationType, const Core::LinAlg::Matrix<nsd_, nen_>&,
        const Core::LinAlg::Matrix<nsd_, nen_>&, Core::LinAlg::SerialDenseMatrix&,
        Core::LinAlg::SerialDenseMatrix&);
template void Discret::Elements::ScaTraEleBoundaryCalcSTIElectrode<Core::FE::CellType::quad4>::
    evaluate_s2_i_coupling_od_at_integration_point<Core::FE::CellType::tri3>(const Mat::Electrode&,
        const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>&,
        const std::vector<Core::LinAlg::Matrix<nen_, 1>>&,
        const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>>&,
        const double, const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>&,
        const Discret::Elements::ScaTraEleParameterBoundary* const, const double, const double,
        const double, const ScaTra::DifferentiationType, const Core::LinAlg::Matrix<nsd_, nen_>&,
        const Core::LinAlg::Matrix<nsd_, nen_>&, Core::LinAlg::SerialDenseMatrix&,
        Core::LinAlg::SerialDenseMatrix&);
template void Discret::Elements::ScaTraEleBoundaryCalcSTIElectrode<Core::FE::CellType::tri3>::
    evaluate_s2_i_coupling_od_at_integration_point<Core::FE::CellType::quad4>(const Mat::Electrode&,
        const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>&,
        const std::vector<Core::LinAlg::Matrix<nen_, 1>>&,
        const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>>&,
        const double, const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>&,
        const Discret::Elements::ScaTraEleParameterBoundary* const, const double, const double,
        const double, const ScaTra::DifferentiationType, const Core::LinAlg::Matrix<nsd_, nen_>&,
        const Core::LinAlg::Matrix<nsd_, nen_>&, Core::LinAlg::SerialDenseMatrix&,
        Core::LinAlg::SerialDenseMatrix&);
template void Discret::Elements::ScaTraEleBoundaryCalcSTIElectrode<Core::FE::CellType::tri3>::
    evaluate_s2_i_coupling_od_at_integration_point<Core::FE::CellType::tri3>(const Mat::Electrode&,
        const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>&,
        const std::vector<Core::LinAlg::Matrix<nen_, 1>>&,
        const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>>&,
        const double, const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>&,
        const Discret::Elements::ScaTraEleParameterBoundary* const, const double, const double,
        const double, const ScaTra::DifferentiationType, const Core::LinAlg::Matrix<nsd_, nen_>&,
        const Core::LinAlg::Matrix<nsd_, nen_>&, Core::LinAlg::SerialDenseMatrix&,
        Core::LinAlg::SerialDenseMatrix&);

FOUR_C_NAMESPACE_CLOSE
