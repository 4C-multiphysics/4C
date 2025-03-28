// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_boundary_calc_elch_electrode_sti_thermo.hpp"

#include "4C_fem_general_utils_boundary_integration.hpp"
#include "4C_inpar_s2i.hpp"
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
Discret::Elements::ScaTraEleBoundaryCalcElchElectrodeSTIThermo<distype, probdim>*
Discret::Elements::ScaTraEleBoundaryCalcElchElectrodeSTIThermo<distype, probdim>::instance(
    const int numdofpernode, const int numscal, const std::string& disname)
{
  static auto singleton_map = Core::Utils::make_singleton_map<std::string>(
      [](const int numdofpernode, const int numscal, const std::string& disname)
      {
        return std::unique_ptr<ScaTraEleBoundaryCalcElchElectrodeSTIThermo<distype, probdim>>(
            new ScaTraEleBoundaryCalcElchElectrodeSTIThermo<distype, probdim>(
                numdofpernode, numscal, disname));
      });

  return singleton_map[disname].instance(
      Core::Utils::SingletonAction::create, numdofpernode, numscal, disname);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::Elements::ScaTraEleBoundaryCalcElchElectrodeSTIThermo<distype,
    probdim>::ScaTraEleBoundaryCalcElchElectrodeSTIThermo(const int numdofpernode,
    const int numscal, const std::string& disname)
    :  // constructor of base class
      myelectrode::ScaTraEleBoundaryCalcElchElectrode(numdofpernode, numscal, disname)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalcElchElectrodeSTIThermo<distype,
    probdim>::evaluate_s2_i_coupling_od(const Core::Elements::FaceElement* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& eslavematrix)
{
  // safety checks
  if (my::numscal_ != 1) FOUR_C_THROW("Invalid number of transported scalars!");
  if (my::numdofpernode_ != 2) FOUR_C_THROW("Invalid number of degrees of freedom per node!");
  if (myelch::elchparams_->equ_pot() != Inpar::ElCh::equpot_divi)
    FOUR_C_THROW("Invalid closing equation for electric potential!");

  // access material of parent element
  std::shared_ptr<const Mat::Electrode> matelectrode =
      std::dynamic_pointer_cast<const Mat::Electrode>(ele->parent_element()->material());
  if (matelectrode == nullptr)
    FOUR_C_THROW("Invalid electrode material for scatra-scatra interface coupling!");

  const int kineticmodel = my::scatraparamsboundary_->kinetic_model();
  const auto differentiationtype =
      Teuchos::getIntegralValue<ScaTra::DifferentiationType>(params, "differentiationtype");

  // extract local nodal values on present and opposite side of scatra-scatra interface
  extract_node_values(discretization, la);
  std::vector<Core::LinAlg::Matrix<nen_, 1>> emasterphinp(
      my::numdofpernode_, Core::LinAlg::Matrix<nen_, 1>(Core::LinAlg::Initialization::zero));
  my::extract_node_values(emasterphinp, discretization, la, "imasterphinp");

  Core::LinAlg::Matrix<nen_, 1> emastertempnp(Core::LinAlg::Initialization::zero);
  if (kineticmodel == Inpar::S2I::kinetics_butlervolmerreducedthermoresistance)
    my::extract_node_values(
        emastertempnp, discretization, la, "imastertemp", my::scatraparams_->nds_thermo());

  // element slave mechanical stress tensor
  const bool is_pseudo_contact = my::scatraparamsboundary_->is_pseudo_contact();
  std::vector<Core::LinAlg::Matrix<nen_, 1>> eslavestress_vector(
      6, Core::LinAlg::Matrix<nen_, 1>(Core::LinAlg::Initialization::zero));
  if (is_pseudo_contact)
    my::extract_node_values(eslavestress_vector, discretization, la, "mechanicalStressState",
        my::scatraparams_->nds_two_tensor_quantity());

  Core::LinAlg::Matrix<nsd_, 1> normal;

  // dummy element matrix
  Core::LinAlg::SerialDenseMatrix dummymatrix;

  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  static Core::LinAlg::Matrix<nsd_, nen_> dsqrtdetg_dd;

  // loop over integration points
  for (int gpid = 0; gpid < intpoints.ip().nquad; ++gpid)
  {
    // evaluate values of shape functions and domain integration factor at current integration point
    const double fac = my::eval_shape_func_and_int_fac(intpoints, gpid, &normal);
    const double detF = my::calculate_det_f_of_parent_element(ele, intpoints.point(gpid));

    const double pseudo_contact_fac = my::calculate_pseudo_contact_factor(
        is_pseudo_contact, eslavestress_vector, normal, my::funct_);

    if (differentiationtype == ScaTra::DifferentiationType::disp)
    {
      static Core::LinAlg::Matrix<nen_, nsd_> xyze_transposed;
      xyze_transposed.update_t(my::xyze_);
      Core::FE::evaluate_shape_function_spatial_derivative_in_prob_dim<distype, nsd_>(
          my::derxy_, my::deriv_, xyze_transposed, normal);
      my::evaluate_spatial_derivative_of_area_integration_factor(intpoints, gpid, dsqrtdetg_dd);
    }

    // evaluate overall integration factor
    const double timefacfac = my::scatraparamstimint_->time_fac() * fac;
    if (timefacfac < 0.0) FOUR_C_THROW("Integration factor is negative!");

    const double timefacwgt = my::scatraparamstimint_->time_fac() * intpoints.ip().qwgt[gpid];
    if (timefacwgt < 0.0) FOUR_C_THROW("Integration factor is negative!");

    evaluate_s2_i_coupling_od_at_integration_point<distype>(*matelectrode, my::ephinp_, etempnp_,
        emastertempnp, emasterphinp, pseudo_contact_fac, my::funct_, my::funct_, my::funct_,
        my::funct_, dsqrtdetg_dd, my::derxy_, my::scatraparamsboundary_, differentiationtype,
        timefacfac, timefacwgt, detF, my::numdofpernode_, eslavematrix, dummymatrix);
  }  // loop over integration points
}  // Discret::Elements::ScaTraEleBoundaryCalcElchElectrodeSTIThermo<distype,
   // probdim>::evaluate_s2_i_coupling_od

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
template <Core::FE::CellType distype_master>
void Discret::Elements::ScaTraEleBoundaryCalcElchElectrodeSTIThermo<distype,
    probdim>::evaluate_s2_i_coupling_od_at_integration_point(const Mat::Electrode& matelectrode,
    const std::vector<Core::LinAlg::Matrix<nen_, 1>>& eslavephinp,
    const Core::LinAlg::Matrix<nen_, 1>& eslavetempnp,
    const Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>& emastertempnp,
    const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>>& emasterphinp,
    const double pseudo_contact_fac, const Core::LinAlg::Matrix<nen_, 1>& funct_slave,
    const Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>& funct_master,
    const Core::LinAlg::Matrix<nen_, 1>& test_slave,
    const Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>& test_master,
    const Core::LinAlg::Matrix<nsd_, nen_>& dsqrtdetg_dd,
    const Core::LinAlg::Matrix<nsd_, nen_>& shape_spatial_derivatives,
    const Discret::Elements::ScaTraEleParameterBoundary* const scatra_parameter_boundary,
    const ScaTra::DifferentiationType differentiationtype, const double timefacfac,
    const double timefacwgt, const double detF, const int num_dof_per_node,
    Core::LinAlg::SerialDenseMatrix& k_ss, Core::LinAlg::SerialDenseMatrix& k_ms)
{
  // get condition specific parameters
  const int kineticmodel = scatra_parameter_boundary->kinetic_model();
  const int numelectrons = scatra_parameter_boundary->num_electrons();
  const double kr = scatra_parameter_boundary->charge_transfer_constant();
  const double alphaa = scatra_parameter_boundary->alphadata();
  const double alphac = scatra_parameter_boundary->alpha_c();

  // number of nodes of master-side element
  const int nen_master = Core::FE::num_nodes<distype_master>;

  // evaluate dof values at current integration point on present and opposite side of scatra-scatra
  // interface
  const double eslavephiint = funct_slave.dot(eslavephinp[0]);
  const double eslavepotint = funct_slave.dot(eslavephinp[1]);
  const double eslavetempint = funct_slave.dot(eslavetempnp);
  const double emastertempint = funct_master.dot(emastertempnp);
  const double emasterphiint = funct_master.dot(emasterphinp[0]);
  const double emasterpotint = funct_master.dot(emasterphinp[1]);

  const double faraday = Discret::Elements::ScaTraEleParameterElch::instance("scatra")->faraday();
  const double gasconstant =
      Discret::Elements::ScaTraEleParameterElch::instance("scatra")->gas_constant();

  // extract saturation value of intercalated lithium concentration from electrode material
  const double cmax = matelectrode.c_max();

  // compute derivatives of scatra-scatra interface coupling residuals w.r.t. thermo dofs according
  // to kinetic model for current scatra-scatra interface coupling condition
  switch (kineticmodel)
  {
    // Butler-Volmer-Peltier kinetics
    case Inpar::S2I::kinetics_butlervolmerpeltier:
    {
      switch (differentiationtype)
      {
        case ScaTra::DifferentiationType::temp:
        {
          // evaluate factor F/RT
          const double frt = faraday / (gasconstant * eslavetempint);

          // equilibrium electric potential difference at electrode surface
          const double epd =
              matelectrode.compute_open_circuit_potential(eslavephiint, faraday, frt, detF);

          // electrode-electrolyte overpotential at integration point
          const double eta = eslavepotint - emasterpotint - epd;

          // Butler-Volmer exchange mass flux density
          const double j0 = kr * std::pow(emasterphiint, alphaa) *
                            std::pow(cmax - eslavephiint, alphaa) * std::pow(eslavephiint, alphac);

          // exponential Butler-Volmer terms
          const double expterm1 = std::exp(alphaa * frt * eta);
          const double expterm2 = std::exp(-alphac * frt * eta);

          // linearization of Butler-Volmer mass flux density w.r.t. temperature
          const double dj_dT_timefacfac = -pseudo_contact_fac * timefacfac * j0 * frt /
                                          eslavetempint * eta *
                                          (alphaa * expterm1 + alphac * expterm2);

          // compute matrix contributions associated with slave-side residuals
          for (int vi = 0; vi < nen_; ++vi)
          {
            const int row_conc = vi * num_dof_per_node;

            for (int ui = 0; ui < nen_; ++ui)
            {
              // compute linearizations associated with slave-side equations for lithium transport
              k_ss(row_conc, ui) += test_slave(vi) * dj_dT_timefacfac * funct_slave(ui);

              // compute linearizations associated with slave-side closing equations for electric
              // potential
              k_ss(row_conc + 1, ui) +=
                  numelectrons * test_slave(vi) * dj_dT_timefacfac * funct_slave(ui);
            }
          }

          // compute matrix contributions associated with master-side residuals if necessary
          if (k_ms.numRows())
          {
            for (int vi = 0; vi < nen_master; ++vi)
            {
              const int row_conc = vi * num_dof_per_node;

              for (int ui = 0; ui < nen_; ++ui)
              {
                // compute linearizations associated with master-side equations for lithium
                // transport
                k_ms(row_conc, ui) -= test_master(vi) * dj_dT_timefacfac * funct_slave(ui);

                // compute linearizations associated with master-side closing equations for electric
                // potential
                k_ms(row_conc + 1, ui) -=
                    numelectrons * test_master(vi) * dj_dT_timefacfac * funct_slave(ui);
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
    case Inpar::S2I::kinetics_butlervolmerreducedthermoresistance:
    {
      // average temperature at interface
      const double etempint = 0.5 * (eslavetempint + emastertempint);

      // factor FRT
      const double frt = faraday / (etempint * gasconstant);

      // equilibrium electric potential difference at electrode surface
      const double epd =
          matelectrode.compute_open_circuit_potential(eslavephiint, faraday, frt, detF);

      // skip further computation in case equilibrium electric potential difference is
      // outside physically meaningful range
      if (std::isinf(epd)) break;

      const double depd_ddetF =
          matelectrode.compute_d_open_circuit_potential_d_det_f(eslavephiint, faraday, frt, detF);

      // Butler-Volmer exchange mass flux density
      const double j0 = kr;

      // electrode-electrolyte overpotential at integration point
      const double eta = eslavepotint - emasterpotint - epd;

      // dervivative of interface flux w.r.t. displacement
      switch (differentiationtype)
      {
        case ScaTra::DifferentiationType::disp:
        {
          double dj_dsqrtdetg(0.0), dj_ddetF(0.0);
          calculate_butler_volmer_disp_linearizations(
              kineticmodel, alphaa, alphac, frt, j0, eta, depd_ddetF, dj_dsqrtdetg, dj_ddetF);

          const double dj_dsqrtdetg_timefacwgt = pseudo_contact_fac * dj_dsqrtdetg * timefacwgt;
          const double dj_ddetF_timefacfac = pseudo_contact_fac * dj_ddetF * timefacfac;

          // loop over matrix columns
          for (int ui = 0; ui < nen_; ++ui)
          {
            const int fui = ui * 3;

            // loop over matrix rows
            for (int vi = 0; vi < nen_; ++vi)
            {
              const int row_conc = vi * num_dof_per_node;
              const int row_pot = row_conc + 1;
              const double vi_dj_dsqrtdetg = test_slave(vi) * dj_dsqrtdetg_timefacwgt;
              const double vi_dj_ddetF = test_slave(vi) * dj_ddetF_timefacfac;

              // loop over spatial dimensions
              for (int dim = 0; dim < 3; ++dim)
              {
                // compute linearizations w.r.t. slave-side structural displacements
                k_ss(row_conc, fui + dim) += vi_dj_dsqrtdetg * dsqrtdetg_dd(dim, ui);
                k_ss(row_conc, fui + dim) +=
                    vi_dj_ddetF * detF * shape_spatial_derivatives(dim, ui);
                k_ss(row_pot, fui + dim) += numelectrons * vi_dj_dsqrtdetg * dsqrtdetg_dd(dim, ui);
                k_ss(row_pot, fui + dim) +=
                    numelectrons * vi_dj_ddetF * detF * shape_spatial_derivatives(dim, ui);
              }
            }
          }
          break;
        }
        case ScaTra::DifferentiationType::temp:
        {
          // derivative of epd w.r.t temperature
          const double depddT = matelectrode.compute_d_open_circuit_potential_d_temperature(
              eslavephiint, faraday, gasconstant);

          // forward declarations
          double dj_dT_slave(0.0);

          // calculate linearizations of Butler-Volmer kinetics w.r.t. tmperature dofs
          calculate_butler_volmer_temp_linearizations(
              alphaa, alphac, depddT, eta, etempint, faraday, frt, gasconstant, j0, dj_dT_slave);

          const double djdT_slave_timefacfac = dj_dT_slave * timefacfac;

          // loop over matrix columns
          for (int ui = 0; ui < nen_; ++ui)
          {
            // loop over matrix rows
            for (int vi = 0; vi < nen_; ++vi)
            {
              const int row_conc = vi * num_dof_per_node;
              const int row_pot = row_conc + 1;
              const double vi_dj_dT_slave =
                  test_slave(vi) * pseudo_contact_fac * djdT_slave_timefacfac;

              // compute linearizations w.r.t. temperature
              k_ss(row_conc, ui) += vi_dj_dT_slave * funct_slave(ui);
              k_ss(row_pot, ui) += numelectrons * vi_dj_dT_slave * funct_slave(ui);
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
    case Inpar::S2I::kinetics_nointerfaceflux:
      break;

    default:
    {
      FOUR_C_THROW("Kinetic model for scatra-scatra interface coupling is not yet implemented!");
    }
  }  // select kinetic model
}  // Discret::Elements::ScaTraEleBoundaryCalcElchElectrodeSTIThermo<distype,
   // probdim>::evaluate_s2_i_coupling_od_at_integration_point

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleBoundaryCalcElchElectrodeSTIThermo<distype,
    probdim>::evaluate_action(Core::Elements::FaceElement* ele, Teuchos::ParameterList& params,
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
    case ScaTra::BoundaryAction::calc_s2icoupling_od:
    {
      evaluate_s2_i_coupling_od(ele, params, discretization, la, elemat1_epetra);
      break;
    }

    default:
    {
      myelectrode::evaluate_action(ele, params, discretization, action, la, elemat1_epetra,
          elemat2_epetra, elevec1_epetra, elevec2_epetra, elevec3_epetra);
      break;
    }
  }  // switch action

  return 0;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalcElchElectrodeSTIThermo<distype,
    probdim>::extract_node_values(const Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la)
{
  // call base class routine
  my::extract_node_values(discretization, la);

  // extract nodal temperature variables associated with time t_{n+1} or t_{n+alpha_f}
  my::extract_node_values(etempnp_, discretization, la, "thermo", my::scatraparams_->nds_thermo());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
double Discret::Elements::ScaTraEleBoundaryCalcElchElectrodeSTIThermo<distype, probdim>::get_frt()
    const
{
  // evaluate local temperature value
  const double temperature = my::funct_.dot(etempnp_);

  // safety check
  if (temperature <= 0.) FOUR_C_THROW("Temperature is non-positive!");

  const double faraday = myelch::elchparams_->faraday();
  const double gasconstant = myelch::elchparams_->gas_constant();

  // evaluate factor F/RT
  return faraday / (gasconstant * temperature);
}


// template classes
template class Discret::Elements::ScaTraEleBoundaryCalcElchElectrodeSTIThermo<
    Core::FE::CellType::quad4, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcElchElectrodeSTIThermo<
    Core::FE::CellType::quad8, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcElchElectrodeSTIThermo<
    Core::FE::CellType::quad9, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcElchElectrodeSTIThermo<
    Core::FE::CellType::tri3, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcElchElectrodeSTIThermo<
    Core::FE::CellType::tri6, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcElchElectrodeSTIThermo<
    Core::FE::CellType::line2, 2>;
template class Discret::Elements::ScaTraEleBoundaryCalcElchElectrodeSTIThermo<
    Core::FE::CellType::line2, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcElchElectrodeSTIThermo<
    Core::FE::CellType::line3, 2>;
template class Discret::Elements::ScaTraEleBoundaryCalcElchElectrodeSTIThermo<
    Core::FE::CellType::nurbs3, 2>;
template class Discret::Elements::ScaTraEleBoundaryCalcElchElectrodeSTIThermo<
    Core::FE::CellType::nurbs9, 3>;

// explicit instantiation of template methods
template void
Discret::Elements::ScaTraEleBoundaryCalcElchElectrodeSTIThermo<Core::FE::CellType::quad4>::
    evaluate_s2_i_coupling_od_at_integration_point<Core::FE::CellType::quad4>(const Mat::Electrode&,
        const std::vector<Core::LinAlg::Matrix<nen_, 1>>&, const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>&,
        const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>>&,
        const double, const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>&,
        const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>&,
        const Core::LinAlg::Matrix<nsd_, nen_>&, const Core::LinAlg::Matrix<nsd_, nen_>&,
        const Discret::Elements::ScaTraEleParameterBoundary* const,
        const ScaTra::DifferentiationType, const double, const double, const double, const int,
        Core::LinAlg::SerialDenseMatrix&, Core::LinAlg::SerialDenseMatrix&);
template void
Discret::Elements::ScaTraEleBoundaryCalcElchElectrodeSTIThermo<Core::FE::CellType::quad4>::
    evaluate_s2_i_coupling_od_at_integration_point<Core::FE::CellType::tri3>(const Mat::Electrode&,
        const std::vector<Core::LinAlg::Matrix<nen_, 1>>&, const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>&,
        const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>>&,
        const double, const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>&,
        const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>&,
        const Core::LinAlg::Matrix<nsd_, nen_>&, const Core::LinAlg::Matrix<nsd_, nen_>&,
        const Discret::Elements::ScaTraEleParameterBoundary* const,
        const ScaTra::DifferentiationType, const double, const double, const double, const int,
        Core::LinAlg::SerialDenseMatrix&, Core::LinAlg::SerialDenseMatrix&);
template void
Discret::Elements::ScaTraEleBoundaryCalcElchElectrodeSTIThermo<Core::FE::CellType::tri3>::
    evaluate_s2_i_coupling_od_at_integration_point<Core::FE::CellType::quad4>(const Mat::Electrode&,
        const std::vector<Core::LinAlg::Matrix<nen_, 1>>&, const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>&,
        const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>>&,
        const double, const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>&,
        const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>&,
        const Core::LinAlg::Matrix<nsd_, nen_>&, const Core::LinAlg::Matrix<nsd_, nen_>&,
        const Discret::Elements::ScaTraEleParameterBoundary* const,
        const ScaTra::DifferentiationType, const double, const double, const double, const int,
        Core::LinAlg::SerialDenseMatrix&, Core::LinAlg::SerialDenseMatrix&);
template void
Discret::Elements::ScaTraEleBoundaryCalcElchElectrodeSTIThermo<Core::FE::CellType::tri3>::
    evaluate_s2_i_coupling_od_at_integration_point<Core::FE::CellType::tri3>(const Mat::Electrode&,
        const std::vector<Core::LinAlg::Matrix<nen_, 1>>&, const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>&,
        const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>>&,
        const double, const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>&,
        const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>&,
        const Core::LinAlg::Matrix<nsd_, nen_>&, const Core::LinAlg::Matrix<nsd_, nen_>&,
        const Discret::Elements::ScaTraEleParameterBoundary* const,
        const ScaTra::DifferentiationType, const double, const double, const double, const int,
        Core::LinAlg::SerialDenseMatrix&, Core::LinAlg::SerialDenseMatrix&);

FOUR_C_NAMESPACE_CLOSE
