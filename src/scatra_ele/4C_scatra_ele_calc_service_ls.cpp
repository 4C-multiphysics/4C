// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_geometry_position_array.hpp"
#include "4C_levelset_input.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_ele_calc_ls.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <numbers>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 | evaluate action                                           fang 02/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::ScaTraEleCalcLS<distype>::evaluate_action(Core::Elements::Element* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
    const ScaTra::Action& action, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  //(for now) only first dof set considered
  const std::vector<int>& lm = la[0].lm_;

  // determine and evaluate action
  switch (action)
  {
    case ScaTra::Action::calc_error:
    {
      // extract local values from the global vectors
      std::shared_ptr<const Core::LinAlg::Vector<double>> phizero =
          discretization.get_state("phiref");
      std::shared_ptr<const Core::LinAlg::Vector<double>> phinp = discretization.get_state("phinp");
      if (phizero == nullptr or phinp == nullptr)
        FOUR_C_THROW("Cannot get state vector 'phizero' and/ or 'phinp'!");

      std::vector<Core::LinAlg::Matrix<nen_, 1>> ephizero(my::numscal_);

      Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phinp, my::ephinp_, lm);
      Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phizero, ephizero, lm);

      // check if length suffices
      if (elevec1.length() < 1) FOUR_C_THROW("Result vector too short");

      cal_error_compared_to_analyt_solution(ele, ephizero, params, elevec1);

      break;
    }
    default:
    {
      my::evaluate_action(
          ele, params, discretization, action, la, elemat1, elemat2, elevec1, elevec2, elevec3);
      break;
    }
  }  // switch(action)

  return 0;
}


/*---------------------------------------------------------------------*
 |  calculate error compared to analytical solution    rasthofer 04/14 |
 *---------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcLS<distype>::cal_error_compared_to_analyt_solution(
    const Core::Elements::Element* ele, const std::vector<Core::LinAlg::Matrix<nen_, 1>>& ephizero,
    Teuchos::ParameterList& params, Core::LinAlg::SerialDenseVector& errors)
{
  // get element volume
  const double vol = my::eval_shape_func_and_derivs_at_ele_center();

  // get element length
  // cast dimension to a double variable -> pow()
  const double dim = static_cast<double>(nsd_);
  const double h = std::pow(vol, 1.0 / dim);

  // integration points and weights
  // more GP than usual due to (possible) cos/exp fcts in analytical solutions
  Core::FE::IntPointsAndWeights<nsd_> intpoints(
      ScaTra::DisTypeToGaussRuleForExactSol<distype>::rule);

  const auto errortype =
      Teuchos::getIntegralValue<LevelSet::CalcErrorLevelSet>(params, "calcerrorflag");
  switch (errortype)
  {
    case LevelSet::calcerror_initial_field:
    {
      // start loop over integration points
      for (int iquad = 0; iquad < intpoints.ip().nquad; iquad++)
      {
        const double fac = my::eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

        const double phizero =
            my::funct_.dot(ephizero[0]);  // only one scalar available for level set
        double smoothH_exact = 0.0;
        smooth_heaviside_function(h, phizero, smoothH_exact);

        const double phinp =
            my::funct_.dot(my::ephinp_[0]);  // only one scalar available for level set
        double smoothH = 0.0;
        smooth_heaviside_function(h, phinp, smoothH);

        // add error
        errors[0] += std::abs(smoothH_exact - smoothH) * fac;
      }  // end of loop over integration points

      errors[1] += vol;
    }
    break;
    default:
      FOUR_C_THROW("Unknown analytical solution!");
      break;
  }  // switch(errortype)

  return;
}


/*----------------------------------------------------------------------*
 | smoothed heaviside function                          rasthofer 04/12 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcLS<distype>::smooth_heaviside_function(
    const double charelelength, const double phi, double& smoothH)
{
  // assume interface thickness
  const double epsilon = charelelength;

  if (phi < -epsilon)
    smoothH = 0.0;
  else if (phi > epsilon)
    smoothH = 1.0;
  else
    smoothH =
        0.5 * (1.0 + phi / epsilon + sin(phi * std::numbers::pi / epsilon) / std::numbers::pi);

  return;
}


// template classes

// 1D elements
template class Discret::Elements::ScaTraEleCalcLS<Core::FE::CellType::line2>;
template class Discret::Elements::ScaTraEleCalcLS<Core::FE::CellType::line3>;

// 2D elements
template class Discret::Elements::ScaTraEleCalcLS<Core::FE::CellType::tri3>;
template class Discret::Elements::ScaTraEleCalcLS<Core::FE::CellType::tri6>;
template class Discret::Elements::ScaTraEleCalcLS<Core::FE::CellType::quad4>;
// template class Discret::Elements::ScaTraEleCalcLS<Core::FE::CellType::quad8>;
template class Discret::Elements::ScaTraEleCalcLS<Core::FE::CellType::quad9>;
template class Discret::Elements::ScaTraEleCalcLS<Core::FE::CellType::nurbs9>;

// 3D elements
template class Discret::Elements::ScaTraEleCalcLS<Core::FE::CellType::hex8>;
// template class Discret::Elements::ScaTraEleCalcLS<Core::FE::CellType::hex20>;
template class Discret::Elements::ScaTraEleCalcLS<Core::FE::CellType::hex27>;
template class Discret::Elements::ScaTraEleCalcLS<Core::FE::CellType::tet4>;
template class Discret::Elements::ScaTraEleCalcLS<Core::FE::CellType::tet10>;
// template class Discret::Elements::ScaTraEleCalcLS<Core::FE::CellType::wedge6>;
template class Discret::Elements::ScaTraEleCalcLS<Core::FE::CellType::pyramid5>;
// template class Discret::Elements::ScaTraEleCalcLS<Core::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE
