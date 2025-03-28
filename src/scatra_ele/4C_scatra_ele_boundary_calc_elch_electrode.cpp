// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_boundary_calc_elch_electrode.hpp"

#include "4C_fem_general_utils_boundary_integration.hpp"
#include "4C_mat_electrode.hpp"
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
Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<distype, probdim>*
Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<distype, probdim>::instance(
    const int numdofpernode, const int numscal, const std::string& disname)
{
  static auto singleton_map = Core::Utils::make_singleton_map<std::string>(
      [](const int numdofpernode, const int numscal, const std::string& disname)
      {
        return std::unique_ptr<ScaTraEleBoundaryCalcElchElectrode<distype, probdim>>(
            new ScaTraEleBoundaryCalcElchElectrode<distype, probdim>(
                numdofpernode, numscal, disname));
      });

  return singleton_map[disname].instance(
      Core::Utils::SingletonAction::create, numdofpernode, numscal, disname);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<distype,
    probdim>::ScaTraEleBoundaryCalcElchElectrode(const int numdofpernode, const int numscal,
    const std::string& disname)
    : myelch::ScaTraEleBoundaryCalcElch(numdofpernode, numscal, disname)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<distype,
    probdim>::evaluate_s2_i_coupling(const Core::Elements::FaceElement* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& eslavematrix,
    Core::LinAlg::SerialDenseMatrix& emastermatrix, Core::LinAlg::SerialDenseVector& eslaveresidual)
{
  // safety check
  if (myelch::elchparams_->equ_pot() != Inpar::ElCh::equpot_divi)
    FOUR_C_THROW("Invalid closing equation for electric potential!");

  // get condition specific parameter
  const int kineticmodel = my::scatraparamsboundary_->kinetic_model();

  // access material of parent element
  std::shared_ptr<const Mat::Electrode> matelectrode = nullptr;
  if (ele->parent_element()->material()->material_type() ==
      Core::Materials::MaterialType::m_electrode)
  {
    matelectrode =
        std::dynamic_pointer_cast<const Mat::Electrode>(ele->parent_element()->material());
  }

  // extract local nodal values on present and opposite side of scatra-scatra interface
  this->extract_node_values(discretization, la);
  std::vector<Core::LinAlg::Matrix<nen_, 1>> emasterphinp(
      my::numdofpernode_, Core::LinAlg::Matrix<nen_, 1>(Core::LinAlg::Initialization::zero));
  if (params.isParameter("evaluate_manifold_coupling"))
    my::extract_node_values(emasterphinp, discretization, la, "manifold_on_scatra");
  else
    my::extract_node_values(emasterphinp, discretization, la, "imasterphinp");

  Core::LinAlg::Matrix<nen_, 1> eslavetempnp(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nen_, 1> emastertempnp(Core::LinAlg::Initialization::zero);
  if (kineticmodel == Inpar::S2I::kinetics_butlervolmerreducedthermoresistance)
  {
    my::extract_node_values(
        eslavetempnp, discretization, la, "islavetemp", my::scatraparams_->nds_thermo());
    my::extract_node_values(
        emastertempnp, discretization, la, "imastertemp", my::scatraparams_->nds_thermo());
  }

  // dummy element matrix and vector
  Core::LinAlg::SerialDenseMatrix dummymatrix;
  Core::LinAlg::SerialDenseVector dummyvector;

  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  Core::LinAlg::Matrix<nsd_, 1> normal;

  // element slave mechanical stress tensor
  const bool is_pseudo_contact = my::scatraparamsboundary_->is_pseudo_contact();
  std::vector<Core::LinAlg::Matrix<nen_, 1>> eslavestress_vector(
      6, Core::LinAlg::Matrix<nen_, 1>(Core::LinAlg::Initialization::zero));
  if (is_pseudo_contact)
    my::extract_node_values(eslavestress_vector, discretization, la, "mechanicalStressState",
        my::scatraparams_->nds_two_tensor_quantity());

  // loop over integration points
  for (int gpid = 0; gpid < intpoints.ip().nquad; ++gpid)
  {
    // evaluate values of shape functions and domain integration factor at current integration point
    const double fac = my::eval_shape_func_and_int_fac(intpoints, gpid, &normal);
    const double detF = my::calculate_det_f_of_parent_element(ele, intpoints.point(gpid));

    // evaluate overall integration factors
    const double timefacfac = my::scatraparamstimint_->time_fac() * fac;
    const double timefacrhsfac = my::scatraparamstimint_->time_fac_rhs() * fac;
    if (timefacfac < 0.0 or timefacrhsfac < 0.0) FOUR_C_THROW("Integration factor is negative!");

    const double pseudo_contact_fac = my::calculate_pseudo_contact_factor(
        is_pseudo_contact, eslavestress_vector, normal, my::funct_);

    evaluate_s2_i_coupling_at_integration_point<distype>(matelectrode, my::ephinp_, emasterphinp,
        eslavetempnp, emastertempnp, pseudo_contact_fac, my::funct_, my::funct_, my::funct_,
        my::funct_, my::scatraparamsboundary_, timefacfac, timefacrhsfac, detF, get_frt(),
        my::numdofpernode_, eslavematrix, emastermatrix, dummymatrix, dummymatrix, eslaveresidual,
        dummyvector);
  }  // loop over integration points
}  // Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<distype,
   // probdim>::evaluate_s2_i_coupling

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
template <Core::FE::CellType distype_master>
void Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<distype, probdim>::
    evaluate_s2_i_coupling_at_integration_point(
        const std::shared_ptr<const Mat::Electrode>& matelectrode,
        const std::vector<Core::LinAlg::Matrix<nen_, 1>>& eslavephinp,
        const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>>&
            emasterphinp,
        const Core::LinAlg::Matrix<nen_, 1>& eslavetempnp,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>& emastertempnp,
        const double pseudo_contact_fac, const Core::LinAlg::Matrix<nen_, 1>& funct_slave,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>& funct_master,
        const Core::LinAlg::Matrix<nen_, 1>& test_slave,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>& test_master,
        const Discret::Elements::ScaTraEleParameterBoundary* const scatra_parameter_boundary,
        const double timefacfac, const double timefacrhsfac, const double detF, double frt,
        const int num_dof_per_node, Core::LinAlg::SerialDenseMatrix& k_ss,
        Core::LinAlg::SerialDenseMatrix& k_sm, Core::LinAlg::SerialDenseMatrix& k_ms,
        Core::LinAlg::SerialDenseMatrix& k_mm, Core::LinAlg::SerialDenseVector& r_s,
        Core::LinAlg::SerialDenseVector& r_m)
{
  // get condition specific parameters
  const auto condition_type = scatra_parameter_boundary->condition_type();
  const int kineticmodel = scatra_parameter_boundary->kinetic_model();
  const int numelectrons = scatra_parameter_boundary->num_electrons();
  const double kr = scatra_parameter_boundary->charge_transfer_constant();
  const double alphaa = scatra_parameter_boundary->alphadata();
  const double alphac = scatra_parameter_boundary->alpha_c();
  const double resistance = scatra_parameter_boundary->resistance();
  const double itemaxmimplicitBV = scatra_parameter_boundary->itemaximplicit_bv();
  const double convtolimplicitBV = scatra_parameter_boundary->convtolimplicit_bv();
  const std::vector<int>* onoff = scatra_parameter_boundary->on_off();

  // number of nodes of master-side mortar element
  const int nen_master = Core::FE::num_nodes<distype_master>;

  // evaluate dof values at current integration point on present and opposite side of scatra-scatra
  // interface
  const double eslavephiint = funct_slave.dot(eslavephinp[0]);
  const double eslavepotint = funct_slave.dot(eslavephinp[1]);
  const double emasterphiint = funct_master.dot(emasterphinp[0]);
  const double emasterpotint = funct_master.dot(emasterphinp[1]);
  const double eslavetempint = funct_slave.dot(eslavetempnp);
  const double emastertempint = funct_master.dot(emastertempnp);

  const double etempint = 0.5 * (eslavetempint + emastertempint);

  // get faraday constant
  const double faraday = Discret::Elements::ScaTraEleParameterElch::instance("scatra")->faraday();

  if (kineticmodel == Inpar::S2I::kinetics_butlervolmerreducedthermoresistance)
  {
    const double gasconstant =
        Discret::Elements::ScaTraEleParameterElch::instance("scatra")->gas_constant();
    frt = faraday / (etempint * gasconstant);
  }

  // compute matrix and vector contributions according to kinetic model for current scatra-scatra
  // interface coupling condition
  switch (kineticmodel)
  {
    // Butler-Volmer kinetics
    case Inpar::S2I::kinetics_butlervolmer:
    case Inpar::S2I::kinetics_butlervolmerlinearized:
    case Inpar::S2I::kinetics_butlervolmerpeltier:
    case Inpar::S2I::kinetics_butlervolmerreducedthermoresistance:
    case Inpar::S2I::kinetics_butlervolmerreduced:
    case Inpar::S2I::kinetics_butlervolmerreducedcapacitance:
    case Inpar::S2I::kinetics_butlervolmerreducedlinearized:
    case Inpar::S2I::kinetics_butlervolmerresistance:
    case Inpar::S2I::kinetics_butlervolmerreducedresistance:
    {
      if (matelectrode == nullptr)
        FOUR_C_THROW("Invalid electrode material for scatra-scatra interface coupling!");

      // extract saturation value of intercalated lithium concentration from electrode material
      const double cmax = matelectrode->c_max();

      // equilibrium electric potential difference at electrode surface
      const double epd =
          matelectrode->compute_open_circuit_potential(eslavephiint, faraday, frt, detF);

      // skip further computation in case equilibrium electric potential difference is outside
      // physically meaningful range
      if (std::isinf(epd)) break;

      // derivative of equilibrium electric potential difference w.r.t. concentration at
      // electrode surface
      const double epdderiv = matelectrode->compute_d_open_circuit_potential_d_concentration(
          eslavephiint, faraday, frt, detF);

      // Butler-Volmer exchange mass flux density
      const double j0 = calculate_butler_volmer_exchange_mass_flux_density(
          kr, alphaa, alphac, cmax, eslavephiint, emasterphiint, kineticmodel, condition_type);

      switch (kineticmodel)
      {
        case Inpar::S2I::kinetics_butlervolmer:
        case Inpar::S2I::kinetics_butlervolmerlinearized:
        case Inpar::S2I::kinetics_butlervolmerpeltier:
        case Inpar::S2I::kinetics_butlervolmerreducedthermoresistance:
        case Inpar::S2I::kinetics_butlervolmerreduced:
        case Inpar::S2I::kinetics_butlervolmerreducedcapacitance:
        case Inpar::S2I::kinetics_butlervolmerreducedlinearized:
        {
          // electrode-electrolyte overpotential at integration point
          const double eta = eslavepotint - emasterpotint - epd;

          // exponential Butler-Volmer terms
          const double expterm1 = std::exp(alphaa * frt * eta);
          const double expterm2 = std::exp(-alphac * frt * eta);
          const double expterm = expterm1 - expterm2;

          // core residual term associated with Butler-Volmer mass flux density
          const double j =
              is_butler_volmer_linearized(kineticmodel) ? j0 * frt * eta : j0 * expterm;

          // forward declarations
          double dj_dc_slave(0.0);
          double dj_dc_master(0.0);
          double dj_dpot_slave(0.0);
          double dj_dpot_master(0.0);

          // calculate linearizations of Butler-Volmer kinetics w.r.t. elch dofs
          calculate_butler_volmer_elch_linearizations(kineticmodel, j0, frt, epdderiv, alphaa,
              alphac, resistance, expterm1, expterm2, kr, faraday, emasterphiint, eslavephiint,
              cmax, eta, dj_dc_slave, dj_dc_master, dj_dpot_slave, dj_dpot_master);

          // calculate RHS and linearizations of master and slave-side residuals
          calculate_rh_sand_global_system<distype_master>(funct_slave, funct_master, test_slave,
              test_master, pseudo_contact_fac, numelectrons, nen_master, timefacfac, timefacrhsfac,
              dj_dc_slave, dj_dc_master, dj_dpot_slave, dj_dpot_master, j, num_dof_per_node, k_ss,
              k_sm, k_ms, k_mm, r_s, r_m);

          break;
        }

        case Inpar::S2I::kinetics_butlervolmerresistance:
        case Inpar::S2I::kinetics_butlervolmerreducedresistance:
        {
          // compute Butler-Volmer mass flux density via Newton-Raphson method
          const double j = calculate_modified_butler_volmer_mass_flux_density(j0, alphaa, alphac,
              frt, eslavepotint, emasterpotint, epd, resistance, itemaxmimplicitBV,
              convtolimplicitBV, faraday);

          // electrode-electrolyte overpotential at integration point
          const double eta = eslavepotint - emasterpotint - epd - j * faraday * resistance;

          // exponential Butler-Volmer terms
          const double expterm1 = std::exp(alphaa * frt * eta);
          const double expterm2 = std::exp(-alphac * frt * eta);

          // forward declarations
          double dj_dc_slave(0.0);
          double dj_dc_master(0.0);
          double dj_dpot_slave(0.0);
          double dj_dpot_master(0.0);

          // calculate linearizations of Butler-Volmer kinetics w.r.t. elch dofs
          calculate_butler_volmer_elch_linearizations(kineticmodel, j0, frt, epdderiv, alphaa,
              alphac, resistance, expterm1, expterm2, kr, faraday, emasterphiint, eslavephiint,
              cmax, eta, dj_dc_slave, dj_dc_master, dj_dpot_slave, dj_dpot_master);

          // calculate RHS and linearizations of master and slave-side residuals
          calculate_rh_sand_global_system<distype_master>(funct_slave, funct_master, test_slave,
              test_master, pseudo_contact_fac, numelectrons, nen_master, timefacfac, timefacrhsfac,
              dj_dc_slave, dj_dc_master, dj_dpot_slave, dj_dpot_master, j, num_dof_per_node, k_ss,
              k_sm, k_ms, k_mm, r_s, r_m);

          break;
        }  // case Inpar::S2I::kinetics_butlervolmerresistance:
        default:
        {
          FOUR_C_THROW("something went wrong");
        }
      }
      break;
    }

    case Inpar::S2I::kinetics_constantinterfaceresistance:
    {
      // core residual
      const double inv_massfluxresistance = 1.0 / (resistance * faraday);
      const double jtimefacrhsfac = pseudo_contact_fac * timefacrhsfac *
                                    (eslavepotint - emasterpotint) * inv_massfluxresistance;

      // calculate core linearizations
      const double dj_dpot_slave_timefacfac =
          pseudo_contact_fac * timefacfac * inv_massfluxresistance;
      const double dj_dpot_master_timefacfac = -dj_dpot_slave_timefacfac;

      // calculate RHS and linearizations of master and slave-side residuals
      if (k_ss.numRows() and k_sm.numRows() and r_s.length())
      {
        for (int vi = 0; vi < nen_; ++vi)
        {
          const int row_conc = vi * num_dof_per_node;
          const int row_pot = vi * num_dof_per_node + 1;

          for (int ui = 0; ui < nen_; ++ui)
          {
            const int col_pot = ui * num_dof_per_node + 1;

            if ((*onoff)[0] == 1)
            {
              k_ss(row_conc, col_pot) +=
                  test_slave(vi) * dj_dpot_slave_timefacfac * funct_slave(ui);
            }
            if ((*onoff)[1] == 1)
            {
              k_ss(row_pot, col_pot) +=
                  numelectrons * test_slave(vi) * dj_dpot_slave_timefacfac * funct_slave(ui);
            }
          }

          for (int ui = 0; ui < nen_master; ++ui)
          {
            const int col_pot = ui * num_dof_per_node + 1;

            if ((*onoff)[0] == 1)
            {
              k_sm(row_conc, col_pot) +=
                  test_slave(vi) * dj_dpot_master_timefacfac * funct_master(ui);
            }
            if ((*onoff)[1] == 1)
            {
              k_sm(row_pot, col_pot) +=
                  numelectrons * test_slave(vi) * dj_dpot_master_timefacfac * funct_master(ui);
            }
          }

          if ((*onoff)[0] == 1) r_s[row_conc] -= test_slave(vi) * jtimefacrhsfac;
          if ((*onoff)[1] == 1) r_s[row_pot] -= numelectrons * test_slave(vi) * jtimefacrhsfac;
        }
      }
      else if (k_ss.numRows() or k_sm.numRows() or r_s.length())
        FOUR_C_THROW(
            "Must provide both slave-side matrices and slave-side vector or none of them!");

      if (k_ms.numRows() and k_mm.numRows() and r_m.length())
      {
        for (int vi = 0; vi < nen_master; ++vi)
        {
          const int row_conc = vi * num_dof_per_node;
          const int row_pot = vi * num_dof_per_node + 1;

          for (int ui = 0; ui < nen_; ++ui)
          {
            const int col_pot = ui * num_dof_per_node + 1;

            if ((*onoff)[0] == 1)
            {
              k_ms(row_conc, col_pot) -=
                  numelectrons * test_master(vi) * dj_dpot_slave_timefacfac * funct_slave(ui);
            }
            if ((*onoff)[1] == 1)
            {
              k_ms(row_pot, col_pot) -=
                  numelectrons * test_master(vi) * dj_dpot_slave_timefacfac * funct_slave(ui);
            }
          }

          for (int ui = 0; ui < nen_master; ++ui)
          {
            const int col_pot = ui * num_dof_per_node + 1;

            if ((*onoff)[0] == 1)
            {
              k_mm(row_conc, col_pot) -=
                  test_master(vi) * dj_dpot_master_timefacfac * funct_master(ui);
            }
            if ((*onoff)[1] == 1)
            {
              k_mm(row_pot, col_pot) -=
                  numelectrons * test_master(vi) * dj_dpot_master_timefacfac * funct_master(ui);
            }
          }
          if ((*onoff)[0] == 1) r_m[row_conc] += test_master(vi) * jtimefacrhsfac;
          if ((*onoff)[1] == 1) r_m[row_pot] += numelectrons * test_master(vi) * jtimefacrhsfac;
        }
      }
      else if (k_ms.numRows() or k_mm.numRows() or r_m.length())
        FOUR_C_THROW(
            "Must provide both master-side matrices and master-side vector or none of them!");

      break;
    }  // case Inpar::S2I::kinetics_constantinterfaceresistance

    case Inpar::S2I::kinetics_nointerfaceflux:
    {
      // do nothing
      break;
    }  // case Inpar::S2I::kinetics_nointerfaceflux

    default:
    {
      FOUR_C_THROW("Kinetic model for scatra-scatra interface coupling is not yet implemented!");
    }
  }  // switch(kineticmodel)
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<distype,
    probdim>::evaluate_s2_i_coupling_capacitance(const Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& eslavematrix,
    Core::LinAlg::SerialDenseMatrix& emastermatrix, Core::LinAlg::SerialDenseVector& eslaveresidual,
    Core::LinAlg::SerialDenseVector& emasterresidual)
{
  // get condition specific parameter
  const int kineticmodel = my::scatraparamsboundary_->kinetic_model();

  // extract local nodal values of temporal derivatives at current timestep on both sides of the
  // scatra-scatra interface
  std::vector<Core::LinAlg::Matrix<nen_, 1>> eslavephidtnp(
      my::numdofpernode_, Core::LinAlg::Matrix<nen_, 1>(Core::LinAlg::Initialization::zero));
  std::vector<Core::LinAlg::Matrix<nen_, 1>> emasterphidtnp(
      my::numdofpernode_, Core::LinAlg::Matrix<nen_, 1>(Core::LinAlg::Initialization::zero));
  if (kineticmodel == Inpar::S2I::kinetics_butlervolmerreducedcapacitance)
  {
    my::extract_node_values(eslavephidtnp, discretization, la, "islavephidtnp");
    my::extract_node_values(emasterphidtnp, discretization, la, "imasterphidtnp");
  }

  // extract local nodal values of current time step at master-side of scatra-scatra interface
  this->extract_node_values(discretization, la);
  std::vector<Core::LinAlg::Matrix<nen_, 1>> emasterphinp(
      my::numdofpernode_, Core::LinAlg::Matrix<nen_, 1>(Core::LinAlg::Initialization::zero));
  my::extract_node_values(emasterphinp, discretization, la, "imasterphinp");

  Core::LinAlg::Matrix<nsd_, 1> normal;

  // element slave mechanical stress tensor
  const bool is_pseudo_contact = my::scatraparamsboundary_->is_pseudo_contact();
  std::vector<Core::LinAlg::Matrix<nen_, 1>> eslavestress_vector(
      6, Core::LinAlg::Matrix<nen_, 1>(Core::LinAlg::Initialization::zero));
  if (is_pseudo_contact)
    my::extract_node_values(eslavestress_vector, discretization, la, "mechanicalStressState",
        my::scatraparams_->nds_two_tensor_quantity());

  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  // loop over integration points
  for (int gpid = 0; gpid < intpoints.ip().nquad; ++gpid)
  {
    // evaluate values of shape functions and domain integration factor at current integration point
    const double fac = my::eval_shape_func_and_int_fac(intpoints, gpid, &normal);
    const double timefacfac = my::scatraparamstimint_->time_fac() * fac;
    const double timefacrhsfac = my::scatraparamstimint_->time_fac_rhs() * fac;

    const double pseudo_contact_fac = my::calculate_pseudo_contact_factor(
        is_pseudo_contact, eslavestress_vector, normal, my::funct_);

    evaluate_s2_i_coupling_capacitance_at_integration_point<distype>(eslavephidtnp, emasterphidtnp,
        my::ephinp_, emasterphinp, pseudo_contact_fac, my::funct_, my::funct_, my::funct_,
        my::funct_, my::scatraparamsboundary_, my::scatraparamstimint_->time_derivative_fac(),
        timefacfac, timefacrhsfac, my::numdofpernode_, eslavematrix, emastermatrix, eslaveresidual,
        emasterresidual);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
template <Core::FE::CellType distype_master>
void Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<distype, probdim>::
    evaluate_s2_i_coupling_capacitance_at_integration_point(
        const std::vector<Core::LinAlg::Matrix<nen_, 1>>& eslavephidtnp,
        const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>>&
            emasterphidtnp,
        const std::vector<Core::LinAlg::Matrix<nen_, 1>>& eslavephinp,
        const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>>&
            emasterphinp,
        const double pseudo_contact_fac, const Core::LinAlg::Matrix<nen_, 1>& funct_slave,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>& funct_master,
        const Core::LinAlg::Matrix<nen_, 1>& test_slave,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>& test_master,
        const Discret::Elements::ScaTraEleParameterBoundary* const scatra_parameter_boundary,
        const double timederivfac, const double timefacfac, const double timefacrhsfac,
        const int num_dof_per_node, Core::LinAlg::SerialDenseMatrix& k_ss,
        Core::LinAlg::SerialDenseMatrix& k_ms, Core::LinAlg::SerialDenseVector& r_s,
        Core::LinAlg::SerialDenseVector& r_m)
{
  // get condition specific parameters
  const int kineticmodel = scatra_parameter_boundary->kinetic_model();
  const int numelectrons = scatra_parameter_boundary->num_electrons();
  const double capacitance = scatra_parameter_boundary->capacitance();

  // number of nodes of master-side mortar element
  const int nen_master = Core::FE::num_nodes<distype_master>;

  // get faraday constant
  const double faraday = Discret::Elements::ScaTraEleParameterElch::instance("scatra")->faraday();

  // compute matrix and vector contributions according to kinetic model for current scatra-scatra
  // interface coupling condition
  switch (kineticmodel)
  {
    case Inpar::S2I::kinetics_butlervolmerreducedcapacitance:
    {
      // evaluate time derivative of potential values at current integration point on slave- and
      // master-side of scatra-scatra interface
      const double eslavepotdtintnp = funct_slave.dot(eslavephidtnp[1]);
      const double emasterpotdtintnp = funct_master.dot(emasterphidtnp[1]);

      // core residual term associated with capacitive mass flux density
      const double jC =
          capacitance * (eslavepotdtintnp - emasterpotdtintnp) / (numelectrons * faraday);

      // calculate non-zero linearization of capacitive mass flux density w.r.t. slave-side dofs
      const double djC_dpot_slave = capacitance * timederivfac / (numelectrons * faraday);

      calculate_rh_sand_global_system_capacitive_flux<distype_master>(funct_slave, test_slave,
          test_master, pseudo_contact_fac, numelectrons, timefacfac, timefacrhsfac, nen_master, jC,
          djC_dpot_slave, num_dof_per_node, k_ss, k_ms, r_s, r_m);

      break;
    }

    default:
    {
      FOUR_C_THROW(
          "Kinetic model for capacitance of scatra-scatra interface coupling is not yet "
          "implemented!");
    }
  }  // switch(kineticmodel)
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<distype,
    probdim>::evaluate_s2_i_coupling_od(const Core::Elements::FaceElement* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& eslavematrix)
{
  std::shared_ptr<const Mat::Electrode> matelectrode = nullptr;
  if (ele->parent_element()->material()->material_type() ==
      Core::Materials::MaterialType::m_electrode)
  {
    matelectrode =
        std::dynamic_pointer_cast<const Mat::Electrode>(ele->parent_element()->material());
  }

  // get condition specific parameters
  const int kineticmodel = my::scatraparamsboundary_->kinetic_model();
  const auto differentiationtype =
      Teuchos::getIntegralValue<ScaTra::DifferentiationType>(params, "differentiationtype");
  const bool is_pseudo_contact = my::scatraparamsboundary_->is_pseudo_contact();

  // extract local nodal values on present and opposite side of scatra-scatra interface
  this->extract_node_values(discretization, la);
  std::vector<Core::LinAlg::Matrix<nen_, 1>> emasterphinp(
      my::numdofpernode_, Core::LinAlg::Matrix<nen_, 1>(Core::LinAlg::Initialization::zero));
  if (params.isParameter("evaluate_manifold_coupling"))
    my::extract_node_values(emasterphinp, discretization, la, "manifold_on_scatra");
  else
    my::extract_node_values(emasterphinp, discretization, la, "imasterphinp");

  // element slave mechanical stress tensor
  std::vector<Core::LinAlg::Matrix<nen_, 1>> eslavestress_vector(
      6, Core::LinAlg::Matrix<nen_, 1>(Core::LinAlg::Initialization::zero));
  if (is_pseudo_contact)
    my::extract_node_values(eslavestress_vector, discretization, la, "mechanicalStressState",
        my::scatraparams_->nds_two_tensor_quantity());

  Core::LinAlg::Matrix<nsd_, 1> normal;

  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  // loop over integration points
  for (int gpid = 0; gpid < intpoints.ip().nquad; ++gpid)
  {
    // evaluate values of shape functions at current integration point
    const double facwgt = my::eval_shape_func_and_int_fac(intpoints, gpid, &normal);

    const double pseudo_contact_fac = my::calculate_pseudo_contact_factor(
        is_pseudo_contact, eslavestress_vector, normal, my::funct_);

    static Core::LinAlg::Matrix<nsd_, nen_> dsqrtdetg_dd;
    if (differentiationtype == ScaTra::DifferentiationType::disp)
    {
      static Core::LinAlg::Matrix<nen_, nsd_> xyze_transposed;
      xyze_transposed.update_t(my::xyze_);
      Core::FE::evaluate_shape_function_spatial_derivative_in_prob_dim<distype, nsd_>(
          my::derxy_, my::deriv_, xyze_transposed, normal);
      my::evaluate_spatial_derivative_of_area_integration_factor(intpoints, gpid, dsqrtdetg_dd);
    }

    // evaluate overall integration factors
    const double timefacwgt = my::scatraparamstimint_->time_fac() * intpoints.ip().qwgt[gpid];
    if (timefacwgt < 0.0) FOUR_C_THROW("Integration factor is negative!");

    // evaluate dof values at current integration point on present and opposite side of
    // scatra-scatra interface
    const double eslavephiint = my::funct_.dot(my::ephinp_[0]);
    const double eslavepotint = my::funct_.dot(my::ephinp_[1]);
    const double emasterphiint = my::funct_.dot(emasterphinp[0]);
    const double emasterpotint = my::funct_.dot(emasterphinp[1]);

    // compute matrix and vector contributions according to kinetic
    // model for current scatra-scatra interface coupling condition
    switch (kineticmodel)
    {
      // Butler-Volmer kinetics
      case Inpar::S2I::kinetics_butlervolmer:
      case Inpar::S2I::kinetics_butlervolmerlinearized:
      case Inpar::S2I::kinetics_butlervolmerreduced:
      case Inpar::S2I::kinetics_butlervolmerreducedcapacitance:
      case Inpar::S2I::kinetics_butlervolmerreducedlinearized:
      {
        // access input parameters associated with current condition
        const auto conditiontype = my::scatraparamsboundary_->condition_type();
        const int numelectrons = my::scatraparamsboundary_->num_electrons();
        const double faraday = myelch::elchparams_->faraday();
        const double alphaa = my::scatraparamsboundary_->alphadata();
        const double alphac = my::scatraparamsboundary_->alpha_c();
        const double kr = my::scatraparamsboundary_->charge_transfer_constant();

        if (matelectrode == nullptr)
          FOUR_C_THROW("Invalid electrode material for scatra-scatra interface coupling!");

        // extract saturation value of intercalated lithium concentration from electrode material
        const double cmax = matelectrode->c_max();

        // compute factor F/(RT)
        const double frt = myelch::elchparams_->frt();
        const double detF = my::calculate_det_f_of_parent_element(ele, intpoints.point(gpid));

        // equilibrium electric potential difference at electrode surface
        const double epd =
            matelectrode->compute_open_circuit_potential(eslavephiint, faraday, frt, detF);

        // skip further computation in case equilibrium electric potential difference is
        // outside physically meaningful range
        if (std::isinf(epd)) break;

        const double depd_ddetF = matelectrode->compute_d_open_circuit_potential_d_det_f(
            eslavephiint, faraday, frt, detF);

        // Butler-Volmer exchange mass flux density
        const double j0 = calculate_butler_volmer_exchange_mass_flux_density(
            kr, alphaa, alphac, cmax, eslavephiint, emasterphiint, kineticmodel, conditiontype);

        // electrode-electrolyte overpotential at integration point
        const double eta = eslavepotint - emasterpotint - epd;

        // derivative of interface flux w.r.t. displacement
        switch (differentiationtype)
        {
          case ScaTra::DifferentiationType::disp:
          {
            double dj_dsqrtdetg(0.0), dj_ddetF(0.0);
            calculate_butler_volmer_disp_linearizations(
                kineticmodel, alphaa, alphac, frt, j0, eta, depd_ddetF, dj_dsqrtdetg, dj_ddetF);

            const double dj_dsqrtdetg_timefacwgt = pseudo_contact_fac * dj_dsqrtdetg * timefacwgt;
            const double dj_ddetF_timefacfac =
                pseudo_contact_fac * dj_ddetF * facwgt * my::scatraparamstimint_->time_fac();

            // loop over matrix columns
            for (int ui = 0; ui < nen_; ++ui)
            {
              const int fui = ui * 3;

              // loop over matrix rows
              for (int vi = 0; vi < nen_; ++vi)
              {
                const int row_conc = vi * my::numdofpernode_;
                const int row_pot = row_conc + 1;
                const double vi_dj_dsqrtdetg = my::funct_(vi) * dj_dsqrtdetg_timefacwgt;
                const double vi_dj_ddetF = my::funct_(vi) * dj_ddetF_timefacfac;

                // loop over spatial dimensions
                for (int dim = 0; dim < 3; ++dim)
                {
                  // compute linearizations w.r.t. slave-side structural displacements
                  eslavematrix(row_conc, fui + dim) += vi_dj_dsqrtdetg * dsqrtdetg_dd(dim, ui);
                  eslavematrix(row_conc, fui + dim) += vi_dj_ddetF * detF * my::derxy_(dim, ui);
                  eslavematrix(row_pot, fui + dim) +=
                      numelectrons * vi_dj_dsqrtdetg * dsqrtdetg_dd(dim, ui);
                  eslavematrix(row_pot, fui + dim) +=
                      numelectrons * vi_dj_ddetF * detF * my::derxy_(dim, ui);
                }
              }
            }
            break;
          }
          default:
          {
            FOUR_C_THROW("Unknown differentiation type");
          }
        }
        break;
      }
      case Inpar::S2I::kinetics_constantinterfaceresistance:
      {
        switch (differentiationtype)
        {
          case ScaTra::DifferentiationType::disp:
          {
            const std::vector<int>* onoff = my::scatraparamsboundary_->on_off();

            // calculate linearizations
            const double inv_massfluxresistance =
                1.0 / (my::scatraparamsboundary_->resistance() * myelch::elchparams_->faraday());
            const double dj_dsqrtdetg_timefacwgt = pseudo_contact_fac * timefacwgt *
                                                   (eslavepotint - emasterpotint) *
                                                   inv_massfluxresistance;

            // loop over matrix columns
            for (int ui = 0; ui < nen_; ++ui)
            {
              const int fui = ui * 3;

              // loop over matrix rows
              for (int vi = 0; vi < nen_; ++vi)
              {
                const int row_conc = vi * my::numdofpernode_;
                const int row_pot = vi * my::numdofpernode_ + 1;
                const double vi_dj_dsqrtdetg = my::funct_(vi) * dj_dsqrtdetg_timefacwgt;

                // loop over spatial dimensions
                for (int dim = 0; dim < 3; ++dim)
                {
                  // finalize linearizations w.r.t. slave-side structural displacements
                  if ((*onoff)[0] == 1)
                  {
                    eslavematrix(row_conc, fui + dim) += vi_dj_dsqrtdetg * dsqrtdetg_dd(dim, ui);
                  }
                  if ((*onoff)[1] == 1)
                  {
                    eslavematrix(row_pot, fui + dim) += my::scatraparamsboundary_->num_electrons() *
                                                        vi_dj_dsqrtdetg * dsqrtdetg_dd(dim, ui);
                  }
                }
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
      case Inpar::S2I::kinetics_nointerfaceflux:
      {
        // nothing to do
        break;
      }
      default:
      {
        FOUR_C_THROW("Kinetic model for scatra-scatra interface coupling is not yet implemented!");
      }
    }  // switch(kineticmodel)
  }  // loop over integration points
}  // Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<distype,
   // probdim>::evaluate_s2_i_coupling_od

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<distype,
    probdim>::evaluate_s2_i_coupling_capacitance_od(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& eslavematrix, Core::LinAlg::SerialDenseMatrix& emastermatrix)
{
  const auto differentiationtype =
      Teuchos::getIntegralValue<ScaTra::DifferentiationType>(params, "differentiationtype");

  const int kineticmodel = my::scatraparamsboundary_->kinetic_model();
  const int numelectrons = my::scatraparamsboundary_->num_electrons();
  const double capacitance = my::scatraparamsboundary_->capacitance();
  const double faraday = Discret::Elements::ScaTraEleParameterElch::instance("scatra")->faraday();
  const bool is_pseudo_contact = my::scatraparamsboundary_->is_pseudo_contact();

  // extract local nodal values of time derivatives at current time step on both sides of the
  // scatra-scatra interface
  std::vector<Core::LinAlg::Matrix<nen_, 1>> eslavephidtnp(
      my::numdofpernode_, Core::LinAlg::Matrix<nen_, 1>(Core::LinAlg::Initialization::zero));
  std::vector<Core::LinAlg::Matrix<nen_, 1>> emasterphidtnp(
      my::numdofpernode_, Core::LinAlg::Matrix<nen_, 1>(Core::LinAlg::Initialization::zero));
  if (kineticmodel == Inpar::S2I::kinetics_butlervolmerreducedcapacitance)
  {
    my::extract_node_values(eslavephidtnp, discretization, la, "islavephidtnp");
    my::extract_node_values(emasterphidtnp, discretization, la, "imasterphidtnp");
  }

  // extract local nodal values of current time step on master side of scatra-scatra interface
  this->extract_node_values(discretization, la);
  std::vector<Core::LinAlg::Matrix<nen_, 1>> emasterphinp(
      my::numdofpernode_, Core::LinAlg::Matrix<nen_, 1>(Core::LinAlg::Initialization::zero));
  my::extract_node_values(emasterphinp, discretization, la, "imasterphinp");

  // element slave mechanical stress tensor
  std::vector<Core::LinAlg::Matrix<nen_, 1>> eslavestress_vector(
      6, Core::LinAlg::Matrix<nen_, 1>(Core::LinAlg::Initialization::zero));
  if (is_pseudo_contact)
    my::extract_node_values(eslavestress_vector, discretization, la, "mechanicalStressState",
        my::scatraparams_->nds_two_tensor_quantity());

  Core::LinAlg::Matrix<nsd_, 1> normal;

  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  // loop over integration points
  for (int gpid = 0; gpid < intpoints.ip().nquad; ++gpid)
  {
    // evaluate values of shape functions at current integration point
    my::eval_shape_func_and_int_fac(intpoints, gpid, &normal);

    const double pseudo_contact_fac = my::calculate_pseudo_contact_factor(
        is_pseudo_contact, eslavestress_vector, normal, my::funct_);

    // evaluate shape derivatives
    static Core::LinAlg::Matrix<nsd_, nen_> dsqrtdetg_dd;
    if (differentiationtype == ScaTra::DifferentiationType::disp)
      my::evaluate_spatial_derivative_of_area_integration_factor(intpoints, gpid, dsqrtdetg_dd);

    // evaluate overall integration factors
    const double timefacwgt = my::scatraparamstimint_->time_fac() * intpoints.ip().qwgt[gpid];
    if (timefacwgt < 0.0) FOUR_C_THROW("Integration factor is negative!");

    // compute matrix and vector contributions according to kinetic
    // model for current scatra-scatra interface coupling condition
    switch (kineticmodel)
    {
      // Butler-Volmer kinetics
      case Inpar::S2I::kinetics_butlervolmerreducedcapacitance:
      {
        // evaluate time derivative of potential values at current integration point on slave- and
        // master-side of scatra-scatra interface
        const double eslavepotdtintnp = my::funct_.dot(eslavephidtnp[1]);
        const double emasterpotdtintnp = my::funct_.dot(emasterphidtnp[1]);

        // core residual term associated with capacitive mass flux density
        const double jC =
            capacitance * (eslavepotdtintnp - emasterpotdtintnp) / (numelectrons * faraday);

        // derivative of interface flux w.r.t. displacement
        switch (differentiationtype)
        {
          case ScaTra::DifferentiationType::disp:
          {
            const double djC_dsqrtdetg_timefacwgt = jC * timefacwgt;

            // loop over matrix columns
            for (int ui = 0; ui < nen_; ++ui)
            {
              const int fui = ui * 3;

              // loop over matrix rows
              for (int vi = 0; vi < nen_; ++vi)
              {
                const int row_conc = vi * my::numdofpernode_;
                const int row_pot = row_conc + 1;
                const double vi_djC_dsqrtdetg =
                    my::funct_(vi) * pseudo_contact_fac * djC_dsqrtdetg_timefacwgt;

                // loop over spatial dimensions
                for (int dim = 0; dim < 3; ++dim)
                {
                  // compute linearizations w.r.t. slave-side structural displacements
                  eslavematrix(row_pot, fui + dim) +=
                      numelectrons * vi_djC_dsqrtdetg * dsqrtdetg_dd(dim, ui);
                  // compute linearizations w.r.t. master-side structural displacements
                  emastermatrix(row_conc, fui + dim) -= vi_djC_dsqrtdetg * dsqrtdetg_dd(dim, ui);
                  emastermatrix(row_pot, fui + dim) -=
                      numelectrons * vi_djC_dsqrtdetg * dsqrtdetg_dd(dim, ui);
                }
              }
            }

            break;
          }
          default:
          {
            FOUR_C_THROW("Unknown differentiation type");
          }
        }
        break;
      }

      default:
      {
        FOUR_C_THROW(
            "Kinetic model for scatra-scatra interface coupling with capacitance is not yet "
            "implemented!");
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
double Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<distype, probdim>::get_valence(
    const std::shared_ptr<const Core::Mat::Material>& material, const int k) const
{
  // valence cannot be computed for electrode material
  FOUR_C_THROW("Valence cannot be computed for electrode material!");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
double Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<distype, probdim>::get_frt() const
{
  // fetch factor F/RT from electrochemistry parameter list in isothermal case
  return myelch::elchparams_->frt();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
template <Core::FE::CellType distype_master>
void Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<distype,
    probdim>::calculate_rh_sand_global_system(const Core::LinAlg::Matrix<nen_, 1>& funct_slave,
    const Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>& funct_master,
    const Core::LinAlg::Matrix<nen_, 1>& test_slave,
    const Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>& test_master,
    const double pseudo_contact_fac, const double numelectrons, const int nen_master,
    const double timefacfac, const double timefacrhsfac, const double dj_dc_slave,
    const double dj_dc_master, const double dj_dpot_slave, const double dj_dpot_master,
    const double j, const int num_dof_per_node, Core::LinAlg::SerialDenseMatrix& k_ss,
    Core::LinAlg::SerialDenseMatrix& k_sm, Core::LinAlg::SerialDenseMatrix& k_ms,
    Core::LinAlg::SerialDenseMatrix& k_mm, Core::LinAlg::SerialDenseVector& r_s,
    Core::LinAlg::SerialDenseVector& r_m)
{
  // pre calculate integrand values
  const double jtimefacrhsfac = pseudo_contact_fac * j * timefacrhsfac;
  const double dj_dc_slave_timefacfac = pseudo_contact_fac * dj_dc_slave * timefacfac;
  const double dj_dc_master_timefacfac = pseudo_contact_fac * dj_dc_master * timefacfac;
  const double dj_dpot_slave_timefacfac = pseudo_contact_fac * dj_dpot_slave * timefacfac;
  const double dj_dpot_master_timefacfac = pseudo_contact_fac * dj_dpot_master * timefacfac;

  // assemble slave side element rhs and linearizations
  if (k_ss.numRows() and k_sm.numRows() and r_s.length())
  {
    for (int vi = 0; vi < nen_; ++vi)
    {
      const int row_conc = vi * num_dof_per_node;
      const int row_pot = row_conc + 1;

      for (int ui = 0; ui < nen_; ++ui)
      {
        const int col_conc = ui * num_dof_per_node;
        const int col_pot = col_conc + 1;

        k_ss(row_conc, col_conc) += test_slave(vi) * dj_dc_slave_timefacfac * funct_slave(ui);
        k_ss(row_conc, col_pot) += test_slave(vi) * dj_dpot_slave_timefacfac * funct_slave(ui);
        k_ss(row_pot, col_conc) +=
            numelectrons * test_slave(vi) * dj_dc_slave_timefacfac * funct_slave(ui);
        k_ss(row_pot, col_pot) +=
            numelectrons * test_slave(vi) * dj_dpot_slave_timefacfac * funct_slave(ui);
      }

      for (int ui = 0; ui < nen_master; ++ui)
      {
        const int col_conc = ui * num_dof_per_node;
        const int col_pot = col_conc + 1;

        k_sm(row_conc, col_conc) += test_slave(vi) * dj_dc_master_timefacfac * funct_master(ui);
        k_sm(row_conc, col_pot) += test_slave(vi) * dj_dpot_master_timefacfac * funct_master(ui);
        k_sm(row_pot, col_conc) +=
            numelectrons * test_slave(vi) * dj_dc_master_timefacfac * funct_master(ui);
        k_sm(row_pot, col_pot) +=
            numelectrons * test_slave(vi) * dj_dpot_master_timefacfac * funct_master(ui);
      }

      r_s[row_conc] -= test_slave(vi) * jtimefacrhsfac;
      r_s[row_pot] -= numelectrons * test_slave(vi) * jtimefacrhsfac;
    }
  }
  else if (k_ss.numRows() or k_sm.numRows() or r_s.length())
    FOUR_C_THROW("Must provide both slave-side matrices and slave-side vector or none of them!");

  // assemble master side element rhs and linearizations
  if (k_ms.numRows() and k_mm.numRows() and r_m.length())
  {
    for (int vi = 0; vi < nen_master; ++vi)
    {
      const int row_conc = vi * num_dof_per_node;
      const int row_pot = row_conc + 1;

      for (int ui = 0; ui < nen_; ++ui)
      {
        const int col_conc = ui * num_dof_per_node;
        const int col_pot = col_conc + 1;

        k_ms(row_conc, col_conc) -= test_master(vi) * dj_dc_slave_timefacfac * funct_slave(ui);
        k_ms(row_conc, col_pot) -= test_master(vi) * dj_dpot_slave_timefacfac * funct_slave(ui);
        k_ms(row_pot, col_conc) -=
            numelectrons * test_master(vi) * dj_dc_slave_timefacfac * funct_slave(ui);
        k_ms(row_pot, col_pot) -=
            numelectrons * test_master(vi) * dj_dpot_slave_timefacfac * funct_slave(ui);
      }

      for (int ui = 0; ui < nen_master; ++ui)
      {
        const int col_conc = ui * num_dof_per_node;
        const int col_pot = col_conc + 1;

        k_mm(row_conc, col_conc) -= test_master(vi) * dj_dc_master_timefacfac * funct_master(ui);
        k_mm(row_conc, col_pot) -= test_master(vi) * dj_dpot_master_timefacfac * funct_master(ui);
        k_mm(row_pot, col_conc) -=
            numelectrons * test_master(vi) * dj_dc_master_timefacfac * funct_master(ui);
        k_mm(row_pot, col_pot) -=
            numelectrons * test_master(vi) * dj_dpot_master_timefacfac * funct_master(ui);
      }

      r_m[row_conc] += test_master(vi) * jtimefacrhsfac;
      r_m[row_pot] += numelectrons * test_master(vi) * jtimefacrhsfac;
    }
  }
  else if (k_ms.numRows() or k_mm.numRows() or r_m.length())
    FOUR_C_THROW("Must provide both master-side matrices and master-side vector or none of them!");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
template <Core::FE::CellType distype_master>
void Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<distype,
    probdim>::calculate_rh_sand_global_system_capacitive_flux(const Core::LinAlg::Matrix<nen_, 1>&
                                                                  funct_slave,
    const Core::LinAlg::Matrix<nen_, 1>& test_slave,
    const Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>& test_master,
    const double pseudo_contact_fac, const int numelectrons, const double timefacfac,
    const double timefacrhsfac, const int nen_master, const double jC, const double djC_dpot_slave,
    const int num_dof_per_node, Core::LinAlg::SerialDenseMatrix& k_ss,
    Core::LinAlg::SerialDenseMatrix& k_ms, Core::LinAlg::SerialDenseVector& r_s,
    Core::LinAlg::SerialDenseVector& r_m)
{
  const double jCtimefacrhsfac = pseudo_contact_fac * jC * timefacrhsfac;
  const double djC_dpot_slave_timefacfac = pseudo_contact_fac * djC_dpot_slave * timefacfac;

  // assemble slave side element rhs and linearizations
  if (k_ss.numRows() and k_ms.numRows() and r_s.length() and r_m.length())
  {
    for (int vi = 0; vi < nen_; ++vi)
    {
      const int row_conc = vi * num_dof_per_node;
      const int row_pot = row_conc + 1;

      for (int ui = 0; ui < nen_; ++ui)
      {
        const int col_conc = ui * num_dof_per_node;
        const int col_pot = col_conc + 1;

        k_ss(row_pot, col_pot) +=
            numelectrons * test_slave(vi) * djC_dpot_slave_timefacfac * funct_slave(ui);
      }

      // only charge conservation equation at slave side
      r_s[row_pot] -= numelectrons * test_slave(vi) * jCtimefacrhsfac;
    }

    for (int vi = 0; vi < nen_master; ++vi)
    {
      const int row_conc = vi * num_dof_per_node;
      const int row_pot = row_conc + 1;

      for (int ui = 0; ui < nen_; ++ui)
      {
        const int col_conc = ui * num_dof_per_node;
        const int col_pot = col_conc + 1;

        k_ms(row_conc, col_pot) -= test_master(vi) * djC_dpot_slave_timefacfac * funct_slave(ui);
        k_ms(row_pot, col_pot) -=
            numelectrons * test_master(vi) * djC_dpot_slave_timefacfac * funct_slave(ui);
      }

      r_m[row_conc] += test_master(vi) * jCtimefacrhsfac;
      r_m[row_pot] += numelectrons * test_master(vi) * jCtimefacrhsfac;
    }
  }
  else if (k_ss.numRows() or k_ms.numRows() or r_s.length() or r_m.length())
    FOUR_C_THROW("You did not provide the correct set of matrices and vectors!");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<distype,
    probdim>::calc_s2_i_coupling_flux(const Core::Elements::FaceElement* ele,
    const Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseVector& scalars)
{
  // get condition specific parameters
  const auto condition_type = my::scatraparamsboundary_->condition_type();
  const int kineticmodel = my::scatraparamsboundary_->kinetic_model();
  const int numelectrons = my::scatraparamsboundary_->num_electrons();
  const double kr = my::scatraparamsboundary_->charge_transfer_constant();
  const double alphaa = my::scatraparamsboundary_->alphadata();
  const double alphac = my::scatraparamsboundary_->alpha_c();
  const double resistance = my::scatraparamsboundary_->resistance();
  const double itemaxmimplicitBV = my::scatraparamsboundary_->itemaximplicit_bv();
  const double convtolimplicitBV = my::scatraparamsboundary_->convtolimplicit_bv();
  const std::vector<int>* onoff = my::scatraparamsboundary_->on_off();
  const bool only_positive_fluxes =
      params.isParameter("only_positive_fluxes") and params.get<bool>("only_positive_fluxes");

  const double faraday = Discret::Elements::ScaTraEleParameterElch::instance("scatra")->faraday();

  // access material of parent element
  std::shared_ptr<const Mat::Electrode> matelectrode = nullptr;
  if (ele->parent_element()->material()->material_type() ==
      Core::Materials::MaterialType::m_electrode)
  {
    matelectrode =
        std::dynamic_pointer_cast<const Mat::Electrode>(ele->parent_element()->material());
  }

  // extract local nodal values on present and opposite side of scatra-scatra interface
  this->extract_node_values(discretization, la);
  std::vector<Core::LinAlg::Matrix<nen_, 1>> emasterphinp(
      my::numdofpernode_, Core::LinAlg::Matrix<nen_, 1>(Core::LinAlg::Initialization::zero));
  if (params.isParameter("evaluate_manifold_coupling"))
    my::extract_node_values(emasterphinp, discretization, la, "manifold_on_scatra");
  else
    my::extract_node_values(emasterphinp, discretization, la, "imasterphinp");

  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  for (int gpid = 0; gpid < intpoints.ip().nquad; ++gpid)
  {
    // evaluate values of shape functions and domain integration factor at current integration point
    const double fac = my::eval_shape_func_and_int_fac(intpoints, gpid);

    Core::LinAlg::Matrix<nen_, 1> eslavetempnp(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<nen_, 1> emastertempnp(Core::LinAlg::Initialization::zero);
    if (kineticmodel == Inpar::S2I::kinetics_butlervolmerreducedthermoresistance)
    {
      my::extract_node_values(
          eslavetempnp, discretization, la, "islavetemp", my::scatraparams_->nds_thermo());
      my::extract_node_values(
          emastertempnp, discretization, la, "imastertemp", my::scatraparams_->nds_thermo());
    }

    const double eslavephiint = my::funct_.dot(my::ephinp_[0]);
    const double eslavepotint = my::funct_.dot(my::ephinp_[1]);
    const double emasterphiint = my::funct_.dot(emasterphinp[0]);
    const double emasterpotint = my::funct_.dot(emasterphinp[1]);
    const double eslavetempint = my::funct_.dot(eslavetempnp);
    const double emastertempint = my::funct_.dot(emastertempnp);

    const double etempint = 0.5 * (eslavetempint + emastertempint);

    double frt = 0.0;
    if (kineticmodel == Inpar::S2I::kinetics_butlervolmerreducedthermoresistance)
    {
      const double gasconstant =
          Discret::Elements::ScaTraEleParameterElch::instance("scatra")->gas_constant();
      frt = faraday / (etempint * gasconstant);
    }
    else
      frt = get_frt();

    const double detF = my::calculate_det_f_of_parent_element(ele, intpoints.point(gpid));

    switch (kineticmodel)
    {
        // Butler-Volmer kinetics
      case Inpar::S2I::kinetics_butlervolmer:
      case Inpar::S2I::kinetics_butlervolmerlinearized:
      case Inpar::S2I::kinetics_butlervolmerpeltier:
      case Inpar::S2I::kinetics_butlervolmerreducedthermoresistance:
      case Inpar::S2I::kinetics_butlervolmerreduced:
      case Inpar::S2I::kinetics_butlervolmerreducedcapacitance:
      case Inpar::S2I::kinetics_butlervolmerreducedlinearized:
      case Inpar::S2I::kinetics_butlervolmerresistance:
      case Inpar::S2I::kinetics_butlervolmerreducedresistance:
      {
        if (matelectrode == nullptr)
          FOUR_C_THROW("Invalid electrode material for scatra-scatra interface coupling!");

        // extract saturation value of intercalated lithium concentration from electrode material
        const double cmax = matelectrode->c_max();

        // equilibrium electric potential difference at electrode surface
        const double epd =
            matelectrode->compute_open_circuit_potential(eslavephiint, faraday, frt, detF);

        // skip further computation in case equilibrium electric potential difference is outside
        // physically meaningful range
        if (std::isinf(epd)) break;

        // Butler-Volmer exchange mass flux density
        const double j0 = calculate_butler_volmer_exchange_mass_flux_density(
            kr, alphaa, alphac, cmax, eslavephiint, emasterphiint, kineticmodel, condition_type);

        switch (kineticmodel)
        {
          case Inpar::S2I::kinetics_butlervolmer:
          case Inpar::S2I::kinetics_butlervolmerlinearized:
          case Inpar::S2I::kinetics_butlervolmerpeltier:
          case Inpar::S2I::kinetics_butlervolmerreducedthermoresistance:
          case Inpar::S2I::kinetics_butlervolmerreduced:
          case Inpar::S2I::kinetics_butlervolmerreducedcapacitance:
          case Inpar::S2I::kinetics_butlervolmerreducedlinearized:
          {
            // electrode-electrolyte overpotential at integration point
            const double eta = eslavepotint - emasterpotint - epd;

            // exponential Butler-Volmer terms
            const double expterm1 = std::exp(alphaa * frt * eta);
            const double expterm2 = std::exp(-alphac * frt * eta);
            const double expterm = expterm1 - expterm2;

            // core residual term associated with Butler-Volmer mass flux density
            const double j =
                is_butler_volmer_linearized(kineticmodel) ? j0 * frt * eta : j0 * expterm;

            if (only_positive_fluxes and j < 0.0) break;

            const double jfac = fac * j;

            for (int vi = 0; vi < nen_; ++vi)
            {
              const double jfac_funct = jfac * my::funct_(vi);

              scalars[0] += jfac_funct;
              scalars[1] += numelectrons * jfac_funct;
            }
            break;
          }

          case Inpar::S2I::kinetics_butlervolmerresistance:
          case Inpar::S2I::kinetics_butlervolmerreducedresistance:
          {
            // compute Butler-Volmer mass flux density via Newton-Raphson method
            const double j = calculate_modified_butler_volmer_mass_flux_density(j0, alphaa, alphac,
                frt, eslavepotint, emasterpotint, epd, resistance, itemaxmimplicitBV,
                convtolimplicitBV, faraday);

            if (only_positive_fluxes and j < 0.0) break;

            const double jfac = fac * j;

            for (int vi = 0; vi < nen_; ++vi)
            {
              const double jfac_funct = jfac * my::funct_(vi);

              scalars[0] += jfac_funct;
              scalars[1] += numelectrons * jfac_funct;
            }

            break;
          }  // case Inpar::S2I::kinetics_butlervolmerresistance:
          default:
          {
            FOUR_C_THROW("something went wrong");
          }
        }
        break;
      }
      case Inpar::S2I::kinetics_constantinterfaceresistance:
      {
        const double inv_massfluxresistance = 1.0 / (resistance * faraday);

        const double j = (eslavepotint - emasterpotint) * inv_massfluxresistance;

        // only add positive fluxes
        if (only_positive_fluxes and j < 0.0) break;

        const double jfac = fac * j;

        for (int vi = 0; vi < nen_; ++vi)
        {
          const double jfac_funct = jfac * my::funct_(vi);

          if ((*onoff)[0] == 1) scalars[0] += jfac_funct;
          if ((*onoff)[1] == 1) scalars[1] += numelectrons * jfac_funct;
        }
      }

      break;

      case Inpar::S2I::kinetics_nointerfaceflux:
      {
        // do nothing
        break;
      }
      default:
      {
        FOUR_C_THROW("kinetic model not implemented.");
      }
    }
  }
}

// explicit instantiation of template methods
template void Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<Core::FE::CellType::tri3>::
    evaluate_s2_i_coupling_at_integration_point<Core::FE::CellType::tri3>(
        const std::shared_ptr<const Mat::Electrode>&,
        const std::vector<Core::LinAlg::Matrix<nen_, 1>>&,
        const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>>&,
        const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>&, const double,
        const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>&,
        const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>&,
        const Discret::Elements::ScaTraEleParameterBoundary* const, const double, const double,
        const double, double, const int, Core::LinAlg::SerialDenseMatrix&,
        Core::LinAlg::SerialDenseMatrix&, Core::LinAlg::SerialDenseMatrix&,
        Core::LinAlg::SerialDenseMatrix&, Core::LinAlg::SerialDenseVector&,
        Core::LinAlg::SerialDenseVector&);
template void Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<Core::FE::CellType::tri3>::
    evaluate_s2_i_coupling_at_integration_point<Core::FE::CellType::quad4>(
        const std::shared_ptr<const Mat::Electrode>&,
        const std::vector<Core::LinAlg::Matrix<nen_, 1>>&,
        const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>>&,
        const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>&,
        const double, const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>&,
        const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>&,
        const Discret::Elements::ScaTraEleParameterBoundary* const, const double, const double,
        const double, double, const int, Core::LinAlg::SerialDenseMatrix&,
        Core::LinAlg::SerialDenseMatrix&, Core::LinAlg::SerialDenseMatrix&,
        Core::LinAlg::SerialDenseMatrix&, Core::LinAlg::SerialDenseVector&,
        Core::LinAlg::SerialDenseVector&);
template void Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<Core::FE::CellType::quad4>::
    evaluate_s2_i_coupling_at_integration_point<Core::FE::CellType::tri3>(
        const std::shared_ptr<const Mat::Electrode>&,
        const std::vector<Core::LinAlg::Matrix<nen_, 1>>&,
        const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>>&,
        const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>&, const double,
        const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>&,
        const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>&,
        const Discret::Elements::ScaTraEleParameterBoundary* const, const double, const double,
        const double, double, const int, Core::LinAlg::SerialDenseMatrix&,
        Core::LinAlg::SerialDenseMatrix&, Core::LinAlg::SerialDenseMatrix&,
        Core::LinAlg::SerialDenseMatrix&, Core::LinAlg::SerialDenseVector&,
        Core::LinAlg::SerialDenseVector&);
template void Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<Core::FE::CellType::quad4>::
    evaluate_s2_i_coupling_at_integration_point<Core::FE::CellType::quad4>(
        const std::shared_ptr<const Mat::Electrode>&,
        const std::vector<Core::LinAlg::Matrix<nen_, 1>>&,
        const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>>&,
        const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>&,
        const double, const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>&,
        const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>&,
        const Discret::Elements::ScaTraEleParameterBoundary* const, const double, const double,
        const double, double, const int, Core::LinAlg::SerialDenseMatrix&,
        Core::LinAlg::SerialDenseMatrix&, Core::LinAlg::SerialDenseMatrix&,
        Core::LinAlg::SerialDenseMatrix&, Core::LinAlg::SerialDenseVector&,
        Core::LinAlg::SerialDenseVector&);

// template classes
template class Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<Core::FE::CellType::quad4, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<Core::FE::CellType::quad8, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<Core::FE::CellType::quad9, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<Core::FE::CellType::tri3, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<Core::FE::CellType::tri6, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<Core::FE::CellType::line2, 2>;
template class Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<Core::FE::CellType::line2, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<Core::FE::CellType::line3, 2>;
template class Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<Core::FE::CellType::nurbs3, 2>;
template class Discret::Elements::ScaTraEleBoundaryCalcElchElectrode<Core::FE::CellType::nurbs9, 3>;

FOUR_C_NAMESPACE_CLOSE
