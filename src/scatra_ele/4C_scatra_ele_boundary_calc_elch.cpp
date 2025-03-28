// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_boundary_calc_elch.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_scatra_ele_parameter_elch.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_scatra_ele_utils_elch.hpp"
#include "4C_utils_function_of_time.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | protected constructor for singletons                      fang 01/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::Elements::ScaTraEleBoundaryCalcElch<distype, probdim>::ScaTraEleBoundaryCalcElch(
    const int numdofpernode, const int numscal, const std::string& disname)
    :  // constructor of base class
      my::ScaTraEleBoundaryCalc(numdofpernode, numscal, disname),

      // instance of parameter class for electrochemistry problems
      elchparams_(Discret::Elements::ScaTraEleParameterElch::instance(disname)),

      // instance of utility class supporting element evaluation
      utils_(ScaTraEleUtilsElch<distype>::instance(numdofpernode, numscal, disname))
{
}


/*----------------------------------------------------------------------*
 | evaluate action                                           fang 02/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleBoundaryCalcElch<distype, probdim>::evaluate_action(
    Core::Elements::FaceElement* ele,                 //!< boundary element
    Teuchos::ParameterList& params,                   //!< parameter list
    Core::FE::Discretization& discretization,         //!< discretization
    ScaTra::BoundaryAction action,                    //!< action
    Core::Elements::LocationArray& la,                //!< location array
    Core::LinAlg::SerialDenseMatrix& elemat1_epetra,  //!< element matrix 1
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,  //!< element matrix 2
    Core::LinAlg::SerialDenseVector& elevec1_epetra,  //!< element right-hand side vector 1
    Core::LinAlg::SerialDenseVector& elevec2_epetra,  //!< element right-hand side vector 2
    Core::LinAlg::SerialDenseVector& elevec3_epetra   //!< element right-hand side vector 3
)
{
  // determine and evaluate action
  switch (action)
  {
    case ScaTra::BoundaryAction::calc_elch_linearize_nernst:
    {
      calc_nernst_linearization(ele, params, discretization, la, elemat1_epetra, elevec1_epetra);

      break;
    }

    case ScaTra::BoundaryAction::calc_elch_cell_voltage:
    {
      calc_cell_voltage(ele, params, discretization, la, elevec1_epetra);

      break;
    }

    default:
    {
      my::evaluate_action(ele, params, discretization, action, la, elemat1_epetra, elemat2_epetra,
          elevec1_epetra, elevec2_epetra, elevec3_epetra);

      break;
    }
  }  // switch action

  return 0;
}


/*----------------------------------------------------------------------*
 | process an electrode kinetics boundary condition          fang 08/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalcElch<distype, probdim>::calc_elch_boundary_kinetics(
    Core::Elements::FaceElement* ele,                 ///< current element
    Teuchos::ParameterList& params,                   ///< parameter list
    Core::FE::Discretization& discretization,         ///< discretization
    Core::Elements::LocationArray& la,                ///< location array
    Core::LinAlg::SerialDenseMatrix& elemat1_epetra,  ///< element matrix
    Core::LinAlg::SerialDenseVector& elevec1_epetra,  ///< element right-hand side vector
    const double scalar  ///< scaling factor for element matrix and right-hand side contributions
)
{
  // state and history variables at element nodes
  my::extract_node_values(discretization, la);
  std::vector<Core::LinAlg::Matrix<nen_, 1>> ehist(
      my::numdofpernode_, Core::LinAlg::Matrix<nen_, 1>(Core::LinAlg::Initialization::zero));
  my::extract_node_values(ehist, discretization, la, "hist");

  // get current condition
  std::shared_ptr<Core::Conditions::Condition> cond =
      params.get<std::shared_ptr<Core::Conditions::Condition>>("condition");
  if (cond == nullptr) FOUR_C_THROW("Cannot access condition 'ElchBoundaryKinetics'");

  // access parameters of the condition
  const auto kinetics = cond->parameters().get<Inpar::ElCh::ElectrodeKinetics>("KINETIC_MODEL");
  auto pot0 = cond->parameters().get<double>("POT");
  const auto curvenum = cond->parameters().get<std::optional<int>>("FUNCT");
  const auto nume = cond->parameters().get<int>("E-");
  // if zero=1=true, the current flow across the electrode is zero (comparable to do-nothing Neuman
  // condition) but the electrode status is evaluated
  const auto zerocur = cond->parameters().get<int>("ZERO_CUR");
  if (nume < 0)
    FOUR_C_THROW(
        "The convention for electrochemical reactions at the electrodes does not allow \n"
        "a negative number of transferred electrons");

  // convention for stoichiometric coefficients s_i:
  // Sum_i (s_i  M_i^(z_i)) -> n e- (n needs to be positive)
  const auto* stoich = &cond->parameters().get<std::vector<int>>("STOICH");
  if ((unsigned int)my::numscal_ != (*stoich).size())
    FOUR_C_THROW(
        "Electrode kinetics: number of stoichiometry coefficients {} does not match"
        " the number of ionic species {}",
        (*stoich).size(), my::numscal_);

  // the classical implementations of kinetic electrode models does not support
  // more than one reagent or product!! There are alternative formulations
  // as e.g. Newman (2004), pp. 205, eq. 8.6 with 8.10
  {
    int reactspecies = 0;
    for (int kk = 0; kk < my::numscal_; ++kk) reactspecies += abs((*stoich)[kk]);

    if (reactspecies > 1 and (kinetics == Inpar::ElCh::butler_volmer or
                                 kinetics == Inpar::ElCh::butler_volmer_yang1997 or
                                 kinetics == Inpar::ElCh::tafel or kinetics == Inpar::ElCh::linear))
      FOUR_C_THROW(
          "Kinetic model Butler-Volmer / Butler-Volmer-Yang / Tafel and Linear: \n"
          "Only one educt and no product is allowed in the implemented version");
  }

  // access input parameter
  const double frt = elchparams_->frt();

  // get control parameter from parameter list
  const bool is_stationary = my::scatraparamstimint_->is_stationary();
  const double time = my::scatraparamstimint_->time();
  double timefac = 1.0;
  double rhsfac = 1.0;
  // find out whether we shell use a time curve and get the factor
  // this feature can be also used for stationary "pseudo time loops"
  if (curvenum.has_value() && curvenum.value() > 0)
  {
    // function_by_id takes a zero-based index
    const double curvefac = Global::Problem::instance()
                                ->function_by_id<Core::Utils::FunctionOfTime>(curvenum.value())
                                .evaluate(time);
    // adjust potential at metal side accordingly
    pot0 *= curvefac;
  }

  if (!(params.get<bool>("calc_status", false)))
  {
    if (not is_stationary)
    {
      // One-step-Theta:    timefac = theta*dt
      // BDF2:              timefac = 2/3 * dt
      // generalized-alpha: timefac = (gamma*alpha_F/alpha_M) * dt
      timefac = my::scatraparamstimint_->time_fac();
      if (timefac < 0.0) FOUR_C_THROW("time factor is negative.");
      // for correct scaling of rhs contribution (see below)
      rhsfac = 1 / my::scatraparamstimint_->alpha_f();
    }

    if (zerocur == 0)
    {
      evaluate_elch_boundary_kinetics(ele, elemat1_epetra, elevec1_epetra, my::ephinp_, ehist,
          timefac, ele->parent_element()->material(), cond, nume, *stoich, kinetics, pot0, frt,
          scalar);
    }

    // realize correct scaling of rhs contribution for gen.alpha case
    // with dt*(gamma/alpha_M) = timefac/alpha_F
    // matrix contributions are already scaled correctly with
    // timefac=dt*(gamma*alpha_F/alpha_M)
    elevec1_epetra.scale(rhsfac);
  }
  else
  {
    // extract local values from the global vector
    std::vector<Core::LinAlg::Matrix<nen_, 1>> ephidtnp(
        my::numdofpernode_, Core::LinAlg::Matrix<nen_, 1>(Core::LinAlg::Initialization::zero));
    my::extract_node_values(ephidtnp, discretization, la, "phidtnp");

    if (not is_stationary)
    {
      // One-step-Theta:    timefacrhs = theta*dt
      // BDF2:              timefacrhs = 2/3 * dt
      // generalized-alpha: timefacrhs = (gamma/alpha_M) * dt
      timefac = my::scatraparamstimint_->time_fac_rhs();
      if (timefac < 0.) FOUR_C_THROW("time factor is negative.");
    }

    evaluate_electrode_status(ele, elevec1_epetra, params, *cond, my::ephinp_, ephidtnp, kinetics,
        *stoich, nume, pot0, frt, timefac, scalar);
  }

  return;
}  // Discret::Elements::ScaTraEleBoundaryCalcElch<distype, probdim>::calc_elch_boundary_kinetics


/*----------------------------------------------------------------------*
 | calculate linearization of nernst equation                     ehrl  |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalcElch<distype, probdim>::calc_nernst_linearization(
    Core::Elements::FaceElement* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra)
{
  std::shared_ptr<Core::Conditions::Condition> cond =
      params.get<std::shared_ptr<Core::Conditions::Condition>>("condition");
  if (cond == nullptr) FOUR_C_THROW("Cannot access condition 'ElchBoundaryKinetics'");

  const auto kinetics = cond->parameters().get<Inpar::ElCh::ElectrodeKinetics>("KINETIC_MODEL");

  // Nernst-BC
  if (kinetics == Inpar::ElCh::nernst)
  {
    // extract local values from the global vector
    my::extract_node_values(discretization, la);

    // access parameters of the condition
    auto pot0 = cond->parameters().get<double>("POT");
    const auto curvenum = cond->parameters().get<std::optional<int>>("FUNCT");
    const auto nume = cond->parameters().get<int>("E-");
    const auto e0 = cond->parameters().get<double>("E0");
    const auto c0 = cond->parameters().get<double>("C0");

    if (nume < 0)
      FOUR_C_THROW(
          "The convention for electrochemical reactions at the electrodes does not allow \n"
          "a negative number of transferred electrons");

    const auto* stoich = &cond->parameters().get<std::vector<int>>("STOICH");
    if ((unsigned int)my::numscal_ != (*stoich).size())
      FOUR_C_THROW(
          "Electrode kinetics: number of stoichiometry coefficients {} does not match"
          " the number of ionic species {}",
          (*stoich).size(), my::numscal_);

    // access input parameter
    const double frt = elchparams_->frt();

    const double time = my::scatraparamstimint_->time();

    if (curvenum.has_value() && curvenum.value() > 0)
    {
      // function_by_id takes a zero-based index
      const double curvefac = Global::Problem::instance()
                                  ->function_by_id<Core::Utils::FunctionOfTime>(curvenum.value())
                                  .evaluate(time);
      // adjust potential at metal side accordingly
      pot0 *= curvefac;
    }

    // concentration of active species at integration point
    std::vector<double> conint(my::numscal_, 0.0);

    // index of reactive species (starting from zero)
    // loop over all scalars
    for (int k = 0; k < my::numscal_; ++k)
    {
      if ((*stoich)[k] == 0) continue;

      /*----------------------------------------------------------------------*
       |               start loop over integration points                     |
       *----------------------------------------------------------------------*/
      const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
          ScaTra::DisTypeToOptGaussRule<distype>::rule);
      for (int gpid = 0; gpid < intpoints.ip().nquad; gpid++)
      {
        const double fac = my::eval_shape_func_and_int_fac(intpoints, gpid);

        // elch-specific values at integration point:
        // concentration is evaluated at all GP since some reaction models depend on all
        // concentrations
        for (int kk = 0; kk < my::numscal_; ++kk) conint[kk] = my::funct_.dot(my::ephinp_[kk]);

        // el. potential at integration point
        const double potint = my::funct_.dot(my::ephinp_[my::numscal_]);

        if (c0 < 1e-12)
          FOUR_C_THROW("reference concentration is too small (c0 < 1.0E-12) : {}", c0);

        for (int vi = 0; vi < nen_; ++vi)
        {
          for (int ui = 0; ui < nen_; ++ui)
          {
            elemat1_epetra(vi * my::numdofpernode_ + my::numscal_, ui * my::numdofpernode_ + k) +=
                fac * my::funct_(vi) / (frt * conint[k] * nume) * my::funct_(ui);
            elemat1_epetra(vi * my::numdofpernode_ + my::numscal_,
                ui * my::numdofpernode_ + my::numscal_) += fac * my::funct_(vi) * my::funct_(ui);
          }

          // -----right-hand-side
          elevec1_epetra[vi * my::numdofpernode_ + my::numscal_] +=
              fac * my::funct_(vi) * (pot0 - e0 - potint - log(conint[k] / c0) / (frt * nume));
        }
      }  // end of loop over integration points gpid
    }  // end loop over scalars
  }  // end if(kinetics == Inpar::ElCh::nernst)
}


/*----------------------------------------------------------------------*
 | calculate cell voltage                                    fang 01/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalcElch<distype, probdim>::calc_cell_voltage(
    const Core::Elements::Element* ele,        //!< the element we are dealing with
    Teuchos::ParameterList& params,            //!< parameter list
    Core::FE::Discretization& discretization,  //!< discretization
    Core::Elements::LocationArray& la,         //!< location array
    Core::LinAlg::SerialDenseVector& scalars  //!< result vector for scalar integrals to be computed
)
{
  // extract local nodal values of electric potential from global state vector
  my::extract_node_values(discretization, la);

  // initialize variables for electric potential and domain integrals
  double intpotential(0.);
  double intdomain(0.);

  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  // loop over integration points
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    // evaluate values of shape functions and domain integration factor at current integration point
    const double fac = my::eval_shape_func_and_int_fac(intpoints, iquad);

    // calculate potential and domain integrals
    for (int vi = 0; vi < nen_; ++vi)
      // potential integral
      intpotential += my::ephinp_[my::numscal_](vi, 0) * my::funct_(vi) * fac;

    // domain integral
    intdomain += fac;
  }  // loop over integration points

  // safety check
  if (scalars.length() != 2)
    FOUR_C_THROW("Result vector for cell voltage computation has invalid length!");

  // write results for electric potential and domain integrals into result vector
  scalars(0) = intpotential;
  scalars(1) = intdomain;

  return;
}


/*----------------------------------------------------------------------*
 | evaluate an electrode kinetics boundary condition          gjb 01/09 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalcElch<distype,
    probdim>::evaluate_elch_boundary_kinetics(const Core::Elements::Element*
                                                  ele,  ///< current element
    Core::LinAlg::SerialDenseMatrix& emat,              ///< element matrix
    Core::LinAlg::SerialDenseVector& erhs,              ///< element right-hand side vector
    const std::vector<Core::LinAlg::Matrix<nen_, 1>>&
        ephinp,  ///< nodal values of concentration and electric potential
    const std::vector<Core::LinAlg::Matrix<nen_, 1>>& ehist,  ///< nodal history vector
    double timefac,                                           ///< time factor
    std::shared_ptr<const Core::Mat::Material> material,      ///< material
    std::shared_ptr<Core::Conditions::Condition> cond,  ///< electrode kinetics boundary condition
    const int nume,                                     ///< number of transferred electrons
    const std::vector<int> stoich,                      ///< stoichiometry of the reaction
    const int kinetics,                                 ///< desired electrode kinetics model
    const double pot0,                                  ///< electrode potential on metal side
    const double frt,                                   ///< factor F/RT
    const double scalar  ///< scaling factor for element matrix and right-hand side contributions
)
{
  // for pre-multiplication of i0 with 1/(F z_k)
  const double faraday = elchparams_->faraday();

  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  // loop over all scalars
  for (int k = 0; k < my::numscal_; ++k)
  {
    if (stoich[k] == 0) continue;

    //(- N^(d+m)*n) = j = s_k / (nume * faraday * z_e-) * i
    //                  = s_k / (nume * faraday * (-1)) * i
    //                    |_______fns_________________|
    // see, e.g. in Ehrl et al., "A computational approach for the simulation of natural convection
    // in electrochemical cells", JCP, 2012
    double fns = -1.0 / faraday / nume;
    // stoichiometry as a consequence of the reaction convention
    fns *= stoich[k];

    // get valence of the single reactant
    const double valence_k = get_valence(material, k);

    /*----------------------------------------------------------------------*
     |               start loop over integration points                     |
     *----------------------------------------------------------------------*/
    for (int gpid = 0; gpid < intpoints.ip().nquad; gpid++)
    {
      const double fac = my::eval_shape_func_and_int_fac(intpoints, gpid);

      // get boundary porosity from condition if available, or set equal to volume porosity
      // otherwise
      auto epsilon = cond->parameters().get<double>("EPSILON");
      if (epsilon == -1)
        epsilon = scalar;
      else if (epsilon <= 0 or epsilon > 1)
        FOUR_C_THROW("Boundary porosity has to be between 0 and 1, or -1 by default!");

      // call utility class for element evaluation
      utils_->evaluate_elch_kinetics_at_integration_point(ele, emat, erhs, ephinp, ehist, timefac,
          fac, my::funct_, *cond, nume, stoich, valence_k, kinetics, pot0, frt, fns, epsilon, k);
    }  // loop over integration points
  }  // loop over all scalars

  return;
}  // Discret::Elements::ScaTraEleBoundaryCalcElch<distype,
   // probdim>::evaluate_elch_boundary_kinetics


/*----------------------------------------------------------------------*
 | evaluate electrode kinetics status information             gjb 01/09 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalcElch<distype, probdim>::evaluate_electrode_status(
    const Core::Elements::Element* ele,        ///< current element
    Core::LinAlg::SerialDenseVector& scalars,  ///< scalars to be integrated
    Teuchos::ParameterList& params,            ///< parameter list
    Core::Conditions::Condition& cond,         ///< condition
    const std::vector<Core::LinAlg::Matrix<nen_, 1>>&
        ephinp,  ///< nodal values of concentration and electric potential
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
  auto epsilon = cond.parameters().get<double>("EPSILON");
  if (epsilon == -1)
    epsilon = scalar;
  else if (epsilon <= 0 or epsilon > 1)
    FOUR_C_THROW("Boundary porosity has to be between 0 and 1, or -1 by default!");

  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  bool statistics = false;

  // index of reactive species (starting from zero)
  for (int k = 0; k < my::numscal_; ++k)
  {
    // only the first oxidized species O is considered for statistics
    // statistics of other species result directly from the oxidized species (current density, ...)
    // or need to be implemented (surface concentration, OCV, ...)
    if (stoich[k] >= 0) continue;

    statistics = true;

    // loop over integration points
    for (int gpid = 0; gpid < intpoints.ip().nquad; gpid++)
    {
      const double fac = my::eval_shape_func_and_int_fac(intpoints, gpid);

      // call utility class for element evaluation
      utils_->evaluate_electrode_status_at_integration_point(ele, scalars, params, cond, ephinp,
          ephidtnp, my::funct_, zerocur, kinetics, stoich, nume, pot0, frt, timefac, fac, epsilon,
          k);
    }  // loop over integration points

    // stop loop over ionic species after one evaluation (see also comment above)
    break;
  }  // loop over scalars

  // safety check
  if (statistics == false)
    FOUR_C_THROW(
        "There is no oxidized species O (stoich<0) defined in your input file!! \n"
        " Statistics could not be evaluated");
}  // Discret::Elements::ScaTraEleBoundaryCalcElch<distype, probdim>::evaluate_electrode_status


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
// template classes
template class Discret::Elements::ScaTraEleBoundaryCalcElch<Core::FE::CellType::quad4, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcElch<Core::FE::CellType::quad8, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcElch<Core::FE::CellType::quad9, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcElch<Core::FE::CellType::tri3, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcElch<Core::FE::CellType::tri6, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcElch<Core::FE::CellType::line2, 2>;
template class Discret::Elements::ScaTraEleBoundaryCalcElch<Core::FE::CellType::line2, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcElch<Core::FE::CellType::line3, 2>;
template class Discret::Elements::ScaTraEleBoundaryCalcElch<Core::FE::CellType::nurbs3, 2>;
template class Discret::Elements::ScaTraEleBoundaryCalcElch<Core::FE::CellType::nurbs9, 3>;

FOUR_C_NAMESPACE_CLOSE
