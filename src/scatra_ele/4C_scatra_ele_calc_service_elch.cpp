// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_discretization.hpp"  // for time curve in body force
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_geometry_position_array.hpp"
#include "4C_fem_nurbs_discretization_utils.hpp"
#include "4C_fluid_rotsym_periodicbc.hpp"
#include "4C_global_data.hpp"  // consistency check of formulation and material
#include "4C_io_input_parameter_container.hpp"
#include "4C_mat_elchmat.hpp"
#include "4C_scatra_ele.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_ele_calc_elch.hpp"
#include "4C_scatra_ele_parameter_elch.hpp"
#include "4C_scatra_ele_parameter_std.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_scatra_ele_utils_elch.hpp"
#include "4C_utils_function_of_time.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 | evaluate action                                           fang 02/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleCalcElch<distype, probdim>::evaluate_action(
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
    case ScaTra::Action::check_scatra_element_parameter:
    {
      check_elch_element_parameter(ele);
      break;
    }

    case ScaTra::Action::calc_flux_domain:
    {
      //--------------------------------------------------------------------------------
      // extract element based or nodal values
      //--------------------------------------------------------------------------------

      // get number of dofset associated with velocity related dofs
      const int ndsvel = my::scatrapara_->nds_vel();

      // get velocity values at nodes
      const std::shared_ptr<const Core::LinAlg::Vector<double>> convel =
          discretization.get_state(ndsvel, "convective velocity field");
      const std::shared_ptr<const Core::LinAlg::Vector<double>> vel =
          discretization.get_state(ndsvel, "velocity field");

      // safety check
      if (convel == nullptr or vel == nullptr) FOUR_C_THROW("Cannot get state vector");

      // determine number of velocity related dofs per node
      const int numveldofpernode = la[ndsvel].lm_.size() / nen_;

      // construct location vector for velocity related dofs
      std::vector<int> lmvel(nsd_ * nen_, -1);
      for (unsigned inode = 0; inode < nen_; ++inode)
        for (unsigned idim = 0; idim < nsd_; ++idim)
          lmvel[inode * nsd_ + idim] = la[ndsvel].lm_[inode * numveldofpernode + idim];

      // extract local values of (convective) velocity field from global state vector
      Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_, nen_>>(*convel, my::econvelnp_, lmvel);
      Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_, nen_>>(*vel, my::evelnp_, lmvel);

      // rotate the vector field in the case of rotationally symmetric boundary conditions
      my::rotsymmpbc_->rotate_my_values_if_necessary(my::econvelnp_);
      my::rotsymmpbc_->rotate_my_values_if_necessary(my::evelnp_);

      // need current values of transported scalar
      // -> extract local values from global vectors
      std::shared_ptr<const Core::LinAlg::Vector<double>> phinp = discretization.get_state("phinp");
      if (phinp == nullptr) FOUR_C_THROW("Cannot get state vector 'phinp'");
      Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phinp, my::ephinp_, la[0].lm_);

      //----------------------------------------------------------------------
      // calculation of element volume both for tau at ele. cent. and int. pt.
      //----------------------------------------------------------------------
      my::eval_shape_func_and_derivs_at_ele_center();

      //----------------------------------------------------------------------
      // get material and stabilization parameters (evaluation at element center)
      //----------------------------------------------------------------------
      // density at t_(n)
      std::vector<double> densn(my::numscal_, 1.0);
      // density at t_(n+1) or t_(n+alpha_F)
      std::vector<double> densnp(my::numscal_, 1.0);
      // density at t_(n+alpha_M)
      std::vector<double> densam(my::numscal_, 1.0);

      // fluid viscosity
      double visc(0.0);

      // material parameter at the element center
      if (not my::scatrapara_->mat_gp())
      {
        set_internal_variables_for_mat_and_rhs();

        get_material_params(ele, densn, densnp, densam, visc);
      }

      //----------------------------------------------------------------------
      // integration loop for one element
      //----------------------------------------------------------------------
      // integration points and weights
      const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
          ScaTra::DisTypeToOptGaussRule<distype>::rule);

      for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
      {
        const double fac = my::eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

        // set internal variables
        set_internal_variables_for_mat_and_rhs();

        //----------------------------------------------------------------------
        // get material parameters (evaluation at integration point)
        //----------------------------------------------------------------------
        if (my::scatrapara_->mat_gp()) get_material_params(ele, densn, densnp, densam, visc, iquad);

        // access control parameter for flux calculation
        Inpar::ScaTra::FluxType fluxtype = my::scatrapara_->calc_flux_domain();
        std::shared_ptr<std::vector<int>> writefluxids = my::scatrapara_->write_flux_ids();

        // do a loop for systems of transported scalars
        for (int& writefluxid : *writefluxids)
        {
          int k = 0;
          // Actually, we compute here a weighted (and integrated) form of the fluxes!
          // On time integration level, these contributions are then used to calculate
          // an L2-projected representation of fluxes.
          // Thus, this method here DOES NOT YET provide flux values that are ready to use!!

          // allocate and initialize!
          Core::LinAlg::Matrix<nsd_, 1> q(Core::LinAlg::Initialization::zero);

          if (writefluxid != my::numdofpernode_)
          {
            k = writefluxid - 1;
            calculate_flux(q, fluxtype, k);
          }
          else if (writefluxid == my::numdofpernode_)
          {
            k = my::numdofpernode_ - 1;
            calculate_current(q, fluxtype, fac);
          }
          else
            FOUR_C_THROW("Flux id, which should be calculated, does not exit in the dof set.");

          // integrate and assemble everything into the "flux" vector
          for (unsigned vi = 0; vi < nen_; vi++)
          {
            const int fvi = vi * my::numdofpernode_ + k;
            elevec1_epetra[fvi] += fac * my::funct_(vi) * q(0);
            elevec2_epetra[fvi] += fac * my::funct_(vi) * q(1);
            if (nsd_ < 3)
              elevec3_epetra[fvi] = 0.0;
            else
              elevec3_epetra[fvi] += fac * my::funct_(vi) * q(2);
          }  // vi
        }
      }

      break;
    }
    case ScaTra::Action::calc_error:
    {
      // check if length suffices
      if (elevec1_epetra.length() < 1) FOUR_C_THROW("Result vector too short");

      // need current solution
      std::shared_ptr<const Core::LinAlg::Vector<double>> phinp = discretization.get_state("phinp");
      if (phinp == nullptr) FOUR_C_THROW("Cannot get state vector 'phinp'");
      Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phinp, my::ephinp_, la[0].lm_);

      cal_error_compared_to_analyt_solution(ele, params, elevec1_epetra);

      break;
    }

    case ScaTra::Action::calc_elch_conductivity:
    {
      // get flag if effective conductivity should be calculated
      bool effCond = params.get<bool>("effCond");
      // get flag if the inverse of the conductivity should be calculated -> specific resistance
      bool specresist = params.get<bool>("specresist");

      // extract quantities for element evaluation
      this->extract_element_and_node_values(ele, params, discretization, la);

      // elevec1_epetra(0):          conductivity of ionic species 0
      // elevec1_epetra(numscal_-1): conductivity of ionic species (numscal_-1)
      // elevec1_epetra(numscal_):   conductivity of the electrolyte solution (sum_k sigma(k))
      // elevec1_epetra(numscal_+1): domain integral
      calculate_conductivity(ele, elchparams_->equ_pot(), elevec1_epetra, effCond, specresist);
      break;
    }

    case ScaTra::Action::calc_elch_boundary_kinetics_point:
    {
      // process electrode boundary kinetics point condition
      calc_elch_boundary_kinetics_point(
          ele, params, discretization, la[0].lm_, elemat1_epetra, elevec1_epetra, 1.);

      break;
    }

    default:
    {
      my::evaluate_action(ele, params, discretization, action, la, elemat1_epetra, elemat2_epetra,
          elevec1_epetra, elevec2_epetra, elevec3_epetra);
      break;
    }
  }  // switch(action)

  return 0;
}


/*----------------------------------------------------------------------------------------*
 | calculate error of numerical solution with respect to analytical solution   fang 10/16 |
 *----------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcElch<distype, probdim>::cal_error_compared_to_analyt_solution(
    const Core::Elements::Element* ele,      //!< element
    Teuchos::ParameterList& params,          //!< parameter list
    Core::LinAlg::SerialDenseVector& errors  //!< vector containing L2 and H1 error norms
)
{
  // call base class routine
  my::cal_error_compared_to_analyt_solution(ele, params, errors);
}  // Discret::Elements::ScaTraEleCalcElch<distype>::cal_error_compared_to_analyt_solution


/*----------------------------------------------------------------------*
  |  Calculate conductivity (ELCH) (private)                   gjb 07/09 |
  *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcElch<distype, probdim>::calculate_conductivity(
    const Core::Elements::Element* ele, const enum Inpar::ElCh::EquPot equpot,
    Core::LinAlg::SerialDenseVector& sigma_domint, bool effCond, bool specresist)
{
  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  // integration loop
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    const double fac = my::eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

    //----------------------------------------------------------------------
    // get material and stabilization parameters (evaluation at element center)
    //----------------------------------------------------------------------
    // density at t_(n)
    std::vector<double> densn(my::numscal_, 1.0);
    // density at t_(n+1) or t_(n+alpha_F)
    std::vector<double> densnp(my::numscal_, 1.0);
    // density at t_(n+alpha_M)
    std::vector<double> densam(my::numscal_, 1.0);

    // fluid viscosity
    double visc(0.0);

    // set internal variables at integration point
    set_internal_variables_for_mat_and_rhs();

    // material parameter at integration point
    get_material_params(ele, densn, densnp, densam, visc, iquad);

    // calculate integrals of (inverted) scalar(s) and domain
    for (unsigned i = 0; i < nen_; i++)
    {
      double sigma_all(0.0);
      std::vector<double> sigma(my::numscal_, 0.0);
      // compute the conductivity (1/(\Omega m) = 1 Siemens / m)
      get_conductivity(equpot, sigma_all, sigma, effCond);

      const double fac_funct_i = fac * my::funct_(i);

      // sigma_domint(0):          conductivity of ionic species 0
      // sigma_domint(numscal_-1): conductivity of ionic species (numscal_-1)
      // sigma_domint(numscal_):   conductivity of the electrolyte solution (sum_k sigma(k))
      // sigma_domint(numscal_+1): domain integral
      for (int k = 0; k < my::numscal_; k++)
      {
        sigma_domint[k] += sigma[k] * fac_funct_i;
      }

      // calculation of conductivity or specific resistance of electrolyte solution
      if (!specresist)
        sigma_domint[my::numscal_] += sigma_all * fac_funct_i;
      else
        sigma_domint[my::numscal_] += 1 / sigma_all * fac_funct_i;

      // domain integral
      sigma_domint[my::numscal_ + 1] += fac_funct_i;
    }
  }  // loop over integration points
}  // ScaTraEleCalcElch()


/*----------------------------------------------------------------------*
 | process an electrode boundary kinetics point condition    fang 08/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcElch<distype, probdim>::calc_elch_boundary_kinetics_point(
    Core::Elements::Element* ele,                     ///< current element
    Teuchos::ParameterList& params,                   ///< parameter list
    Core::FE::Discretization& discretization,         ///< discretization
    std::vector<int>& lm,                             ///< location vector
    Core::LinAlg::SerialDenseMatrix& elemat1_epetra,  ///< element matrix
    Core::LinAlg::SerialDenseVector& elevec1_epetra,  ///< element right-hand side vector
    const double scalar  ///< scaling factor for element matrix and right-hand side contributions
)
{
  // get actual values of transported scalars
  std::shared_ptr<const Core::LinAlg::Vector<double>> phinp = discretization.get_state("phinp");
  if (phinp == nullptr) FOUR_C_THROW("Cannot get state vector 'phinp'");

  // extract local values from the global vector
  std::vector<Core::LinAlg::Matrix<nen_, 1>> ephinp(
      my::numdofpernode_, Core::LinAlg::Matrix<nen_, 1>(Core::LinAlg::Initialization::zero));
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phinp, ephinp, lm);

  // get history variable (needed for double layer modeling)
  std::shared_ptr<const Core::LinAlg::Vector<double>> hist = discretization.get_state("hist");
  if (phinp == nullptr) FOUR_C_THROW("Cannot get state vector 'hist'");

  // extract local values from the global vector
  std::vector<Core::LinAlg::Matrix<nen_, 1>> ehist(
      my::numdofpernode_, Core::LinAlg::Matrix<nen_, 1>(Core::LinAlg::Initialization::zero));
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*hist, ehist, lm);

  // get current condition
  std::shared_ptr<Core::Conditions::Condition> cond =
      params.get<std::shared_ptr<Core::Conditions::Condition>>("condition");
  if (cond == nullptr) FOUR_C_THROW("Cannot access condition 'ElchBoundaryKineticsPoint'!");

  // access parameters of the condition
  const int kinetics = cond->parameters().get<Inpar::ElCh::ElectrodeKinetics>("KINETIC_MODEL");
  double pot0 = cond->parameters().get<double>("POT");
  const auto functnum = cond->parameters().get<std::optional<int>>("FUNCT");
  const int nume = cond->parameters().get<int>("E-");
  // if zero=1=true, the current flow across the electrode is zero (comparable to do-nothing Neuman
  // condition) but the electrode status is evaluated
  const int zerocur = cond->parameters().get<int>("ZERO_CUR");
  if (nume < 0)
  {
    FOUR_C_THROW(
        "The convention for electrochemical reactions at the electrodes does not allow \n"
        "a negative number of transferred electrons");
  }

  // convention for stoichiometric coefficients s_i:
  // Sum_i (s_i  M_i^(z_i)) -> n e- (n needs to be positive)
  const auto* stoich = &cond->parameters().get<std::vector<int>>("STOICH");
  if ((unsigned int)my::numscal_ != (*stoich).size())
  {
    FOUR_C_THROW(
        "Electrode kinetics: number of stoichiometry coefficients {} does not match"
        " the number of ionic species {}",
        (*stoich).size(), my::numscal_);
  }

  // the classical implementations of kinetic electrode models does not support
  // more than one reagent or product!! There are alternative formulations
  // as e.g. Newman (2004), pp. 205, eq. 8.6 with 8.10
  {
    int reactspecies = 0;
    for (int kk = 0; kk < my::numscal_; ++kk) reactspecies += abs((*stoich)[kk]);

    if (reactspecies > 1 and (kinetics == Inpar::ElCh::butler_volmer or
                                 kinetics == Inpar::ElCh::butler_volmer_yang1997 or
                                 kinetics == Inpar::ElCh::tafel or kinetics == Inpar::ElCh::linear))
    {
      FOUR_C_THROW(
          "Kinetic model Butler-Volmer / Butler-Volmer-Yang / Tafel and Linear: \n"
          "Only one educt and no product is allowed in the implemented version");
    }
  }

  // access input parameter
  const double frt = elchparams_->frt();
  if (frt <= 0.0) FOUR_C_THROW("A negative factor frt is not possible by definition");

  // get control parameter from parameter list
  const bool is_stationary = my::scatraparatimint_->is_stationary();
  const double time = my::scatraparatimint_->time();
  double timefac = 1.0;
  double rhsfac = 1.0;
  // find out whether we shell use a time curve and get the factor
  // this feature can be also used for stationary "pseudo time loops"
  if (functnum.has_value() && functnum.value() > 0)
  {
    const double functfac = Global::Problem::instance()
                                ->function_by_id<Core::Utils::FunctionOfTime>(functnum.value())
                                .evaluate(time);

    // adjust potential at metal side accordingly
    pot0 *= functfac;
  }

  if (!(params.get<bool>("calc_status", false)))
  {
    if (not is_stationary)
    {
      // One-step-Theta:    timefac = theta*dt
      // BDF2:              timefac = 2/3 * dt
      // generalized-alpha: timefac = (gamma*alpha_F/alpha_M) * dt
      timefac = my::scatraparatimint_->time_fac();
      if (timefac < 0.0) FOUR_C_THROW("time factor is negative.");
      // for correct scaling of rhs contribution (see below)
      rhsfac = 1 / my::scatraparatimint_->alpha_f();
    }

    if (zerocur == 0)
    {
      evaluate_elch_boundary_kinetics_point(ele, elemat1_epetra, elevec1_epetra, ephinp, ehist,
          timefac, cond, nume, *stoich, kinetics, pot0, frt, scalar);
    }

    // realize correct scaling of rhs contribution for gen.alpha case
    // with dt*(gamma/alpha_M) = timefac/alpha_F
    // matrix contributions are already scaled correctly with
    // timefac=dt*(gamma*alpha_F/alpha_M)
    elevec1_epetra.scale(rhsfac);
  }
  else
  {
    // get actual values of transported scalars
    std::shared_ptr<const Core::LinAlg::Vector<double>> phidtnp =
        discretization.get_state("phidtnp");
    if (phidtnp == nullptr) FOUR_C_THROW("Cannot get state vector 'ephidtnp'");
    // extract local values from the global vector
    std::vector<Core::LinAlg::Matrix<nen_, 1>> ephidtnp(
        my::numdofpernode_, Core::LinAlg::Matrix<nen_, 1>(Core::LinAlg::Initialization::zero));
    Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phidtnp, ephidtnp, lm);

    if (not is_stationary)
    {
      // One-step-Theta:    timefacrhs = theta*dt
      // BDF2:              timefacrhs = 2/3 * dt
      // generalized-alpha: timefacrhs = (gamma/alpha_M) * dt
      timefac = my::scatraparatimint_->time_fac_rhs();
      if (timefac < 0.) FOUR_C_THROW("time factor is negative.");
    }

    evaluate_electrode_status_point(ele, elevec1_epetra, params, *cond, ephinp, ephidtnp, kinetics,
        *stoich, nume, pot0, frt, timefac, scalar);
  }
}  // Discret::Elements::ScaTraEleCalcElch<distype>::calc_elch_boundary_kinetics_point


/*----------------------------------------------------------------------*
 | evaluate an electrode boundary kinetics point condition   fang 08/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcElch<distype, probdim>::evaluate_elch_boundary_kinetics_point(
    const Core::Elements::Element* ele,                        ///< current element
    Core::LinAlg::SerialDenseMatrix& emat,                     ///< element matrix
    Core::LinAlg::SerialDenseVector& erhs,                     ///< element right-hand side vector
    const std::vector<Core::LinAlg::Matrix<nen_, 1>>& ephinp,  ///< state variables at element nodes
    const std::vector<Core::LinAlg::Matrix<nen_, 1>>&
        ehist,                                          ///< history variables at element nodes
    double timefac,                                     ///< time factor
    std::shared_ptr<Core::Conditions::Condition> cond,  ///< electrode kinetics boundary condition
    const int nume,                                     ///< number of transferred electrons
    const std::vector<int> stoich,                      ///< stoichiometry of the reaction
    const int kinetics,                                 ///< desired electrode kinetics model
    const double pot0,                                  ///< electrode potential on metal side
    const double frt,                                   ///< factor F/RT
    const double scalar  ///< scaling factor for element matrix and right-hand side contributions
)
{
  // get boundary porosity from condition if available, or set equal to volume porosity otherwise
  double epsilon = cond->parameters().get<double>("EPSILON");
  if (epsilon == -1)
    epsilon = scalar;
  else if (epsilon <= 0 or epsilon > 1)
    FOUR_C_THROW("Boundary porosity has to be between 0 and 1, or -1 by default!");

  // extract nodal cloud of current condition
  const std::vector<int>* nodeids = cond->get_nodes();

  // safety checks
  if (!nodeids)
    FOUR_C_THROW("Electrode kinetics point boundary condition doesn't have nodal cloud!");
  if (nodeids->size() != 1)
    FOUR_C_THROW(
        "Electrode kinetics point boundary condition must be associated with exactly one node!");
  if (nsd_ele_ != 1)
  {
    FOUR_C_THROW(
        "Electrode kinetics point boundary conditions are applicable to one-dimensional problems "
        "only!");
  }

  // extract global ID of conditioned node
  const int nodeid = (*nodeids)[0];

  // find out whether conditioned node is the leftmost (position 0) or rightmost (position 1) node
  // of the current line element
  int position(-1);
  if (nodeid == ele->nodes()[0]->id())
    position = 0;
  else if (nodeid == ele->nodes()[1]->id())
    position = 1;
  else
  {
    FOUR_C_THROW(
        "Electrode kinetics point boundary condition must be imposed either on the leftmost or on "
        "the rightmost node of a line element!");
  }

  // manipulate shape function array according to node position
  my::funct_.clear();
  my::funct_(position) = 1.;

  // loop over all scalars
  for (int k = 0; k < my::numscal_; ++k)
  {
    if (stoich[k] == 0) continue;

    //(- N^(d+m)*n) = j = s_k / (nume * faraday * z_e-) * i
    //                  = s_k / (nume * faraday * (-1)) * i
    //                    |_______fns_________________|
    // see, e.g. in Ehrl et al., "A computational approach for the simulation of natural convection
    // in electrochemical cells", JCP, 2012
    double fns = -1.0 / elchparams_->faraday() / nume;
    // stoichiometry as a consequence of the reaction convention
    fns *= stoich[k];

    // get valence of the single reactant
    const double valence_k = diff_manager()->get_valence(k);

    // call utility class for evaluation of electrode boundary kinetics point condition
    utils_->evaluate_elch_kinetics_at_integration_point(ele, emat, erhs, ephinp, ehist, timefac, 1.,
        my::funct_, *cond, nume, stoich, valence_k, kinetics, pot0, frt, fns, epsilon, k);
  }  // loop over all scalars
}  // Discret::Elements::ScaTraEleCalcElch<distype>::evaluate_elch_boundary_kinetics_point


/*----------------------------------------------------------------------*
 | evaluate status information on point electrode            fang 09/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcElch<distype, probdim>::evaluate_electrode_status_point(
    const Core::Elements::Element* ele,                        ///< current element
    Core::LinAlg::SerialDenseVector& scalars,                  ///< scalars to be integrated
    Teuchos::ParameterList& params,                            ///< parameter list
    Core::Conditions::Condition& cond,                         ///< condition
    const std::vector<Core::LinAlg::Matrix<nen_, 1>>& ephinp,  ///< state variables at element nodes
    const std::vector<Core::LinAlg::Matrix<nen_, 1>>& ephidtnp,  ///< nodal time derivative vector
    const int kinetics,             ///< desired electrode kinetics model
    const std::vector<int> stoich,  ///< stoichiometry of the reaction
    const int nume,                 ///< number of transferred electrons
    const double pot0,              ///< electrode potential on metal side
    const double frt,               ///< factor F/RT
    const double timefac,           ///< time factor
    const double scalar             ///< scaling factor for current related quantities
)
{
  // Warning:
  // Specific time integration parameter are set in the following function.
  // In the case of a genalpha-time integration scheme the solution vector phiaf_ at time n+af
  // is passed to the element evaluation routine. Therefore, the electrode status is evaluate at a
  // different time (n+af) than our output routine (n+1), resulting in slightly different values at
  // the electrode. A different approach is not possible (without major hacks) since the
  // time-integration scheme is necessary to perform galvanostatic simulations, for instance. Think
  // about: double layer effects for genalpha time-integration scheme

  // if zero=1=true, the current flow across the electrode is zero (comparable to do-nothing Neumann
  // condition) but the electrode status is evaluated
  const int zerocur = cond.parameters().get<int>("ZERO_CUR");

  // get boundary porosity from condition if available, or set equal to volume porosity otherwise
  double epsilon = cond.parameters().get<double>("EPSILON");
  if (epsilon == -1)
    epsilon = scalar;
  else if (epsilon <= 0 or epsilon > 1)
    FOUR_C_THROW("Boundary porosity has to be between 0 and 1, or -1 by default!");

  bool statistics = false;

  // extract nodal cloud of current condition
  const std::vector<int>* nodeids = cond.get_nodes();

  // safety checks
  if (!nodeids)
    FOUR_C_THROW("Electrode kinetics point boundary condition doesn't have nodal cloud!");
  if (nodeids->size() != 1)
    FOUR_C_THROW(
        "Electrode kinetics point boundary condition must be associated with exactly one node!");
  if (nsd_ele_ != 1)
  {
    FOUR_C_THROW(
        "Electrode kinetics point boundary conditions are applicable to one-dimensional problems "
        "only!");
  }

  // extract global ID of conditioned node
  const int nodeid = (*nodeids)[0];

  // find out whether conditioned node is the leftmost (position 0) or rightmost (position 1) node
  // of the current line element
  int position(-1);
  if (nodeid == ele->nodes()[0]->id())
    position = 0;
  else if (nodeid == ele->nodes()[1]->id())
    position = 1;
  else
  {
    FOUR_C_THROW(
        "Electrode kinetics point boundary condition must be imposed either on the leftmost or on "
        "the rightmost node of a line element!");
  }

  // manipulate shape function array according to node position
  my::funct_.clear();
  my::funct_(position) = 1.;

  // index of reactive species (starting from zero)
  for (int k = 0; k < my::numscal_; ++k)
  {
    // only the first oxidized species O is considered for statistics
    // statistics of other species result directly from the oxidized species (current density, ...)
    // or need to be implemented (surface concentration, OCV, ...)
    if (stoich[k] >= 0) continue;

    statistics = true;

    // call utility class for element evaluation
    utils_->evaluate_electrode_status_at_integration_point(ele, scalars, params, cond, ephinp,
        ephidtnp, my::funct_, zerocur, kinetics, stoich, nume, pot0, frt, timefac, 1., epsilon, k);

    // stop loop over ionic species after one evaluation (see also comment above)
    break;
  }  // loop over scalars

  // safety check
  if (!statistics)
  {
    FOUR_C_THROW(
        "There is no oxidized species O (stoich<0) defined in your input file!! \n"
        " Statistics could not be evaluated");
  }
}  // Discret::Elements::ScaTraEleCalcElch<distype>::evaluate_electrode_status_point


/*----------------------------------------------------------------------------------------*
 | finite difference check on element level (for debugging only) (protected)   fang 10/14 |
 *----------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcElch<distype, probdim>::fd_check(Core::Elements::Element* ele,
    Core::LinAlg::SerialDenseMatrix& emat, Core::LinAlg::SerialDenseVector& erhs,
    Core::LinAlg::SerialDenseVector& subgrdiff)
{
  // screen output
  std::cout << "FINITE DIFFERENCE CHECK FOR ELEMENT " << ele->id();

  // make a copy of state variables to undo perturbations later
  std::vector<Core::LinAlg::Matrix<nen_, 1>> ephinp_original(my::numdofpernode_);
  for (int k = 0; k < my::numdofpernode_; ++k)
    for (unsigned i = 0; i < nen_; ++i) ephinp_original[k](i, 0) = my::ephinp_[k](i, 0);

  // generalized-alpha time integration requires a copy of history variables as well
  std::vector<Core::LinAlg::Matrix<nen_, 1>> ehist_original(my::numscal_);
  if (my::scatraparatimint_->is_gen_alpha())
  {
    for (int k = 0; k < my::numscal_; ++k)
      for (unsigned i = 0; i < nen_; ++i) ehist_original[k](i, 0) = my::ehist_[k](i, 0);
  }

  // initialize element matrix and vectors for perturbed state
  Core::LinAlg::SerialDenseMatrix emat_dummy(emat);
  Core::LinAlg::SerialDenseVector erhs_perturbed(erhs);
  Core::LinAlg::SerialDenseVector subgrdiff_dummy(subgrdiff);

  // initialize counter for failed finite difference checks
  unsigned counter(0);

  // initialize tracking variable for maximum absolute and relative errors
  double maxabserr(0.);
  double maxrelerr(0.);

  // loop over columns of element matrix by first looping over nodes and then over dofs at each node
  for (unsigned inode = 0; inode < nen_; ++inode)
  {
    for (int idof = 0; idof < my::numdofpernode_; ++idof)
    {
      // number of current column of element matrix
      int col = inode * my::numdofpernode_ + idof;

      // clear element matrix and vectors for perturbed state
      emat_dummy.putScalar(0.0);
      erhs_perturbed.putScalar(0.0);
      subgrdiff_dummy.putScalar(0.0);

      // fill state vectors with original state variables
      for (int k = 0; k < my::numdofpernode_; ++k)
        for (unsigned i = 0; i < nen_; ++i) my::ephinp_[k](i, 0) = ephinp_original[k](i, 0);
      if (my::scatraparatimint_->is_gen_alpha())
        for (int k = 0; k < my::numscal_; ++k)
          for (unsigned i = 0; i < nen_; ++i) my::ehist_[k](i, 0) = ehist_original[k](i, 0);

      // impose perturbation
      if (my::scatraparatimint_->is_gen_alpha())
      {
        // perturbation of phi(n+alphaF), not of phi(n+1) => scale epsilon by factor alphaF
        my::ephinp_[idof](inode, 0) +=
            my::scatraparatimint_->alpha_f() * my::scatrapara_->fd_check_eps();

        // perturbation of phi(n+alphaF) by alphaF*epsilon corresponds to perturbation of phidtam
        // (stored in ehist_) by alphaM*epsilon/(gamma*dt); note: alphaF/timefac = alphaM/(gamma*dt)
        if (idof < my::numscal_)
        {
          my::ehist_[idof](inode, 0) += my::scatraparatimint_->alpha_f() /
                                        my::scatraparatimint_->time_fac() *
                                        my::scatrapara_->fd_check_eps();
        }
      }
      else
        my::ephinp_[idof](inode, 0) += my::scatrapara_->fd_check_eps();

      // calculate element right-hand side vector for perturbed state
      sysmat(ele, emat_dummy, erhs_perturbed, subgrdiff_dummy);

      // Now we compare the difference between the current entries in the element matrix
      // and their finite difference approximations according to
      // entries ?= (-erhs_perturbed + erhs_original) / epsilon

      // Note that the element right-hand side equals the negative element residual.
      // To account for errors due to numerical cancellation, we additionally consider
      // entries - erhs_original / epsilon ?= -erhs_perturbed / epsilon

      // Note that we still need to evaluate the first comparison as well. For small entries in the
      // element matrix, the second comparison might yield good agreement in spite of the entries
      // being wrong!
      for (int row = 0; row < my::numdofpernode_ * static_cast<int>(nen_); ++row)
      {
        // get current entry in original element matrix
        const double entry = emat(row, col);

        // finite difference suggestion (first divide by epsilon and then subtract for better
        // conditioning)
        const double fdval = -erhs_perturbed(row) / my::scatrapara_->fd_check_eps() +
                             erhs(row) / my::scatrapara_->fd_check_eps();

        // confirm accuracy of first comparison
        if (abs(fdval) > 1.e-17 and abs(fdval) < 1.e-15)
          FOUR_C_THROW("Finite difference check involves values too close to numerical zero!");

        // absolute and relative errors in first comparison
        const double abserr1 = entry - fdval;
        if (abs(abserr1) > abs(maxabserr)) maxabserr = abserr1;
        double relerr1(0.);
        if (abs(entry) > 1.e-17)
          relerr1 = abserr1 / abs(entry);
        else if (abs(fdval) > 1.e-17)
          relerr1 = abserr1 / abs(fdval);
        if (abs(relerr1) > abs(maxrelerr)) maxrelerr = relerr1;

        // evaluate first comparison
        if (abs(relerr1) > my::scatrapara_->fd_check_tol())
        {
          if (!counter) std::cout << " --> FAILED AS FOLLOWS:" << std::endl;
          std::cout << "emat[" << row << "," << col << "]:  " << entry << "   ";
          std::cout << "finite difference suggestion:  " << fdval << "   ";
          std::cout << "absolute error:  " << abserr1 << "   ";
          std::cout << "relative error:  " << relerr1 << std::endl;

          counter++;
        }

        // first comparison OK
        else
        {
          // left-hand side in second comparison
          const double left = entry - erhs(row) / my::scatrapara_->fd_check_eps();

          // right-hand side in second comparison
          const double right = -erhs_perturbed(row) / my::scatrapara_->fd_check_eps();

          // confirm accuracy of second comparison
          if (abs(right) > 1.e-17 and abs(right) < 1.e-15)
            FOUR_C_THROW("Finite difference check involves values too close to numerical zero!");

          // absolute and relative errors in second comparison
          const double abserr2 = left - right;
          if (abs(abserr2) > abs(maxabserr)) maxabserr = abserr2;
          double relerr2(0.);
          if (abs(left) > 1.e-17)
            relerr2 = abserr2 / abs(left);
          else if (abs(right) > 1.e-17)
            relerr2 = abserr2 / abs(right);
          if (abs(relerr2) > abs(maxrelerr)) maxrelerr = relerr2;

          // evaluate second comparison
          if (abs(relerr2) > my::scatrapara_->fd_check_tol())
          {
            if (!counter) std::cout << " --> FAILED AS FOLLOWS:" << std::endl;
            std::cout << "emat[" << row << "," << col << "]-erhs[" << row << "]/eps:  " << left
                      << "   ";
            std::cout << "-erhs_perturbed[" << row << "]/eps:  " << right << "   ";
            std::cout << "absolute error:  " << abserr2 << "   ";
            std::cout << "relative error:  " << relerr2 << std::endl;

            counter++;
          }
        }
      }
    }
  }

  // screen output in case finite difference check is passed
  if (!counter)
    std::cout << " --> PASSED WITH MAXIMUM ABSOLUTE ERROR " << maxabserr
              << " AND MAXIMUM RELATIVE ERROR " << maxrelerr << std::endl;

  // undo perturbations of state variables
  for (int k = 0; k < my::numdofpernode_; ++k)
    for (unsigned i = 0; i < nen_; ++i) my::ephinp_[k](i, 0) = ephinp_original[k](i, 0);
  if (my::scatraparatimint_->is_gen_alpha())
    for (int k = 0; k < my::numscal_; ++k)
      for (unsigned i = 0; i < nen_; ++i) my::ehist_[k](i, 0) = ehist_original[k](i, 0);
}


// template classes

// 1D elements
template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::line2, 1>;
template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::line2, 2>;
template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::line2, 3>;
template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::line3, 1>;

// 2D elements
template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::tri3, 2>;
template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::tri3, 3>;
template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::tri6, 2>;
template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::quad4, 2>;
template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::quad4, 3>;
// template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::quad8>;
template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::quad9, 2>;
template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::nurbs9, 2>;

// 3D elements
template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::hex8, 3>;
// template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::hex20>;
template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::hex27, 3>;
template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::tet4, 3>;
template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::tet10, 3>;
// template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::wedge6>;
template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::pyramid5, 3>;
// template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE
