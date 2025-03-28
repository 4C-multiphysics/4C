// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_ele_boundary_calc.hpp"

#include "4C_fem_condition.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_boundary_integration.hpp"
#include "4C_global_data.hpp"  // for curves and functions
#include "4C_porofluid_pressure_based_ele_action.hpp"
#include "4C_porofluid_pressure_based_ele_parameter.hpp"
#include "4C_utils_function.hpp"
#include "4C_utils_parameter_list.hpp"
#include "4C_utils_singleton_owner.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | singleton access method                                   vuong 08/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::PoroFluidMultiPhaseEleBoundaryCalc<distype>*
Discret::Elements::PoroFluidMultiPhaseEleBoundaryCalc<distype>::instance(
    const int numdofpernode, const std::string& disname)
{
  static auto singleton_map = Core::Utils::make_singleton_map<std::string>(
      [](const int numdofpernode, const std::string& disname)
      {
        return std::unique_ptr<PoroFluidMultiPhaseEleBoundaryCalc<distype>>(
            new PoroFluidMultiPhaseEleBoundaryCalc<distype>(numdofpernode, disname));
      });

  return singleton_map[disname].instance(
      Core::Utils::SingletonAction::create, numdofpernode, disname);
}


/*----------------------------------------------------------------------*
 | protected constructor for singletons                      vuong 08/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::PoroFluidMultiPhaseEleBoundaryCalc<distype>::PoroFluidMultiPhaseEleBoundaryCalc(
    const int numdofpernode, const std::string& disname)
    : params_(Discret::Elements::PoroFluidMultiPhaseEleParameter::instance(disname)),
      numdofpernode_(numdofpernode),
      xyze_(Core::LinAlg::Initialization::zero),  // initialize to zero
      edispnp_(Core::LinAlg::Initialization::zero),
      xsi_(Core::LinAlg::Initialization::zero),
      funct_(Core::LinAlg::Initialization::zero),
      deriv_(Core::LinAlg::Initialization::zero),
      derxy_(Core::LinAlg::Initialization::zero),
      normal_(Core::LinAlg::Initialization::zero),
      velint_(Core::LinAlg::Initialization::zero),
      metrictensor_(Core::LinAlg::Initialization::zero)
{
  return;
}


/*----------------------------------------------------------------------*
 | setup element evaluation                                  vuong 08/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::PoroFluidMultiPhaseEleBoundaryCalc<distype>::setup_calc(
    Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization)
{
  // get node coordinates (we have a nsd_+1 dimensional domain!)
  Core::Geo::fill_initial_position_array<distype, nsd_ + 1, Core::LinAlg::Matrix<nsd_ + 1, nen_>>(
      ele, xyze_);

  return 0;
}

/*----------------------------------------------------------------------*
 * Evaluate element                                          vuong 08/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::PoroFluidMultiPhaseEleBoundaryCalc<distype>::evaluate(
    Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    std::vector<Core::LinAlg::SerialDenseVector*>& elevec)
{
  //--------------------------------------------------------------------------------
  // preparations for element
  //--------------------------------------------------------------------------------
  if (setup_calc(ele, params, discretization) == -1) return 0;

  //--------------------------------------------------------------------------------
  // extract element based or nodal values
  //--------------------------------------------------------------------------------
  extract_element_and_node_values(ele, params, discretization, la);

  // check for the action parameter
  const auto action =
      Teuchos::getIntegralValue<POROFLUIDMULTIPHASE::BoundaryAction>(params, "action");
  // evaluate action
  evaluate_action(ele, params, discretization, action, la, elemat, elevec);

  return 0;
}

/*----------------------------------------------------------------------*
 | extract element based or nodal values                     vuong 08/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::PoroFluidMultiPhaseEleBoundaryCalc<
    distype>::extract_element_and_node_values(Core::Elements::Element* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la)
{
  // get additional state vector for ALE case: grid displacement
  if (params_->is_ale())
  {
    // get number of dof-set associated with displacement related dofs
    const int ndsdisp = params_->nds_disp();

    std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp =
        discretization.get_state(ndsdisp, "dispnp");
    if (dispnp == nullptr) FOUR_C_THROW("Cannot get state vector 'dispnp'");

    // determine number of displacement related dofs per node
    const int numdispdofpernode = la[ndsdisp].lm_.size() / nen_;

    // construct location vector for displacement related dofs
    std::vector<int> lmdisp((nsd_ + 1) * nen_, -1);
    for (int inode = 0; inode < nen_; ++inode)
      for (int idim = 0; idim < nsd_ + 1; ++idim)
        lmdisp[inode * (nsd_ + 1) + idim] = la[ndsdisp].lm_[inode * numdispdofpernode + idim];

    // extract local values of displacement field from global state vector
    Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_ + 1, nen_>>(*dispnp, edispnp_, lmdisp);

    // add nodal displacements to point coordinates
    xyze_ += edispnp_;
  }
  else
    edispnp_.clear();
}

/*----------------------------------------------------------------------*
 * Action type: Evaluate                                     vuong 08/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::PoroFluidMultiPhaseEleBoundaryCalc<distype>::evaluate_action(
    Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, POROFLUIDMULTIPHASE::BoundaryAction action,
    Core::Elements::LocationArray& la, std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    std::vector<Core::LinAlg::SerialDenseVector*>& elevec)
{
  // switch over action type
  switch (action)
  {
    case POROFLUIDMULTIPHASE::bd_calc_Neumann:
    {
      // check if the neumann conditions were set
      Core::Conditions::Condition* condition =
          params.get<Core::Conditions::Condition*>("condition");
      if (condition == nullptr) FOUR_C_THROW("Cannot access Neumann boundary condition!");

      // evaluate neumann loads
      evaluate_neumann(ele, params, discretization, *condition, la, *elevec[0]);

      break;
    }
  }

  return 0;
}

/*----------------------------------------------------------------------*
 | evaluate Neumann boundary condition                        vuong 08/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::PoroFluidMultiPhaseEleBoundaryCalc<distype>::evaluate_neumann(
    Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Conditions::Condition& condition,
    Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseVector& elevec1)
{
  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_> intpoints(
      POROFLUIDMULTIPHASE::ElementUtils::DisTypeToOptGaussRule<distype>::rule);

  // find out whether we will use a time curve
  const double time = params_->time();

  // get values, switches and spatial functions from the condition
  // (assumed to be constant on element boundary)
  const int numdof = condition.parameters().get<int>("NUMDOF");
  const auto onoff = condition.parameters().get<std::vector<int>>("ONOFF");
  const auto val = condition.parameters().get<std::vector<double>>("VAL");
  const auto func = condition.parameters().get<std::vector<std::optional<int>>>("FUNCT");

  if (numdofpernode_ != numdof)
    FOUR_C_THROW(
        "The NUMDOF you have entered in your NEUMANN CONDITION does not equal the number of "
        "scalars.");

  // integration loop
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    double fac = eval_shape_func_and_int_fac(intpoints, iquad);

    // factor given by spatial function
    double functfac = 1.0;

    // determine global coordinates of current Gauss point
    const int nsd_vol_ele = nsd_ + 1;
    Core::LinAlg::Matrix<nsd_vol_ele, 1> coordgp;  // coordinate has always to be given in 3D!
    coordgp.multiply_nn(xyze_, funct_);

    const double* coordgpref = &coordgp(0);  // needed for function evaluation

    for (int dof = 0; dof < numdofpernode_; ++dof)
    {
      if (onoff[dof])  // is this dof activated?
      {
        // factor given by spatial function
        if (func[dof].has_value() && func[dof].value() > 0)
        {
          // evaluate function at current Gauss point (provide always 3D coordinates!)
          functfac = Global::Problem::instance()
                         ->function_by_id<Core::Utils::FunctionOfSpaceTime>(func[dof].value())
                         .evaluate(coordgpref, time, dof);
        }
        else
          functfac = 1.;

        const double val_fac_funct_fac = val[dof] * fac * functfac;

        for (int node = 0; node < nen_; ++node)
          elevec1[node * numdofpernode_ + dof] += funct_(node) * val_fac_funct_fac;
      }  // if (onoff[dof])
    }  // loop over dofs
  }  // loop over integration points

  return 0;
}

/*----------------------------------------------------------------------*
 | evaluate shape functions and int. factor at int. point     vuong 08/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
double Discret::Elements::PoroFluidMultiPhaseEleBoundaryCalc<distype>::eval_shape_func_and_int_fac(
    const Core::FE::IntPointsAndWeights<nsd_>& intpoints,  ///< integration points
    const int iquad,                                       ///< id of current Gauss point
    Core::LinAlg::Matrix<1 + nsd_, 1>* normalvec  ///< normal vector at Gauss point(optional)
)
{
  // coordinates of the current integration point
  const double* gpcoord = (intpoints.ip().qxg)[iquad];
  for (int idim = 0; idim < nsd_; idim++)
  {
    xsi_(idim) = gpcoord[idim];
  }

  // shape functions and their first derivatives
  Core::FE::shape_function<distype>(xsi_, funct_);
  Core::FE::shape_function_deriv1<distype>(xsi_, deriv_);

  // the metric tensor and the area of an infinitesimal surface/line element
  // optional: get normal at integration point as well
  double drs(0.0);
  Core::FE::compute_metric_tensor_for_boundary_ele<distype>(
      xyze_, deriv_, metrictensor_, drs, normalvec);

  // return the integration factor
  return intpoints.ip().qwgt[iquad] * drs;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
// template classes
template class Discret::Elements::PoroFluidMultiPhaseEleBoundaryCalc<Core::FE::CellType::quad4>;
template class Discret::Elements::PoroFluidMultiPhaseEleBoundaryCalc<Core::FE::CellType::quad8>;
template class Discret::Elements::PoroFluidMultiPhaseEleBoundaryCalc<Core::FE::CellType::quad9>;
template class Discret::Elements::PoroFluidMultiPhaseEleBoundaryCalc<Core::FE::CellType::tri3>;
template class Discret::Elements::PoroFluidMultiPhaseEleBoundaryCalc<Core::FE::CellType::tri6>;
template class Discret::Elements::PoroFluidMultiPhaseEleBoundaryCalc<Core::FE::CellType::line2>;
template class Discret::Elements::PoroFluidMultiPhaseEleBoundaryCalc<Core::FE::CellType::line3>;

FOUR_C_NAMESPACE_CLOSE
