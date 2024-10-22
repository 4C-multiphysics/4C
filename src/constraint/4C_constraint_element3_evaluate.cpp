// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_constraint_element3.hpp"
#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_linalg_serialdensevector.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Discret::ELEMENTS::ConstraintElement3::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  ActionType act = none;

  // get the required action and distinguish between 2d and 3d MPC's
  std::string action = params.get<std::string>("action", "none");
  if (action == "none")
    return 0;
  else if (action == "calc_MPC_stiff")
  {
    act = calc_MPC_stiff;
  }
  else if (action == "calc_MPC_state")
  {
    act = calc_MPC_state;
  }
  else
    FOUR_C_THROW("Unknown type of action for ConstraintElement3");

  switch (act)
  {
    case none:
    {
      return (0);
    }
    break;
    case calc_MPC_state:
    {
      Teuchos::RCP<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == Teuchos::null) FOUR_C_THROW("Cannot get state vector 'displacement'");
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);
      const int numnod = num_node();

      if (numnod == 4)
      {
        const int numdim = 3;
        Core::LinAlg::Matrix<4, numdim> xscurr;  // material coord. of element
        spatial_configuration(xscurr, mydisp);
        Core::LinAlg::Matrix<numdim, 1> elementnormal;

        compute_normal(xscurr, elementnormal);
        if (abs(elementnormal.norm2()) < 1E-6)
        {
          FOUR_C_THROW("Bad plane, points almost on a line!");
        }

        elevec3[0] = compute_normal_dist(xscurr, elementnormal);
      }
      else if (numnod == 2)
      {
        Teuchos::RCP<Core::Conditions::Condition> condition =
            params.get<Teuchos::RCP<Core::Conditions::Condition>>("condition");
        const auto& direct = condition->parameters().get<std::vector<double>>("direction");
        const auto& value = condition->parameters().get<std::string>("value");
        if (value == "disp")
          elevec3[0] = compute_weighted_distance(mydisp, direct);
        else if (value == "x")
        {
          Core::LinAlg::Matrix<2, 3> xscurr;  // material coord. of element
          spatial_configuration(xscurr, mydisp);
          elevec3[0] = compute_weighted_distance(xscurr, direct);
        }
        else
          FOUR_C_THROW("MPC cannot compute state!");
      }
    }
    break;
    case calc_MPC_stiff:
    {
      Teuchos::RCP<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == Teuchos::null) FOUR_C_THROW("Cannot get state vector 'displacement'");
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);
      const int numnod = num_node();

      if (numnod == 4)
      {
        const int numdim = 3;
        const int numnode = 4;

        Core::LinAlg::Matrix<numnode, numdim> xscurr;  // material coord. of element
        spatial_configuration(xscurr, mydisp);

        Core::LinAlg::Matrix<numdim, 1> elementnormal;
        compute_normal(xscurr, elementnormal);
        if (abs(elementnormal.norm2()) < 1E-6)
        {
          FOUR_C_THROW("Bad plane, points almost on a line!");
        }
        double normaldistance = compute_normal_dist(xscurr, elementnormal);

        compute_first_deriv(xscurr, elevec1, elementnormal);
        compute_second_deriv(xscurr, elemat1, elementnormal);

        // update corresponding column in "constraint" matrix
        elevec2 = elevec1;
        elevec3[0] = normaldistance;
      }
      else if (numnod == 2)
      {
        Teuchos::RCP<Core::Conditions::Condition> condition =
            params.get<Teuchos::RCP<Core::Conditions::Condition>>("condition");
        const std::vector<double>& direct =
            condition->parameters().get<std::vector<double>>("direction");

        // Compute weighted difference between masternode and other node and it's derivative
        compute_first_deriv_weighted_distance(elevec1, direct);
        elevec2 = elevec1;

        const std::string& value = condition->parameters().get<std::string>("value");
        if (value == "disp")
          elevec3[0] = compute_weighted_distance(mydisp, direct);
        else if (value == "x")
        {
          Core::LinAlg::Matrix<2, 3> xscurr;  // spatial coord. of element
          spatial_configuration(xscurr, mydisp);
          elevec3[0] = compute_weighted_distance(xscurr, direct);
        }
        else
          FOUR_C_THROW("MPC cannot compute state!");
      }
    }
    break;
    default:
      FOUR_C_THROW("Unimplemented type of action");
  }
  return 0;


}  // end of Discret::ELEMENTS::ConstraintElement3::Evaluate

/*----------------------------------------------------------------------*
 * Evaluate Neumann (->FOUR_C_THROW) */
int Discret::ELEMENTS::ConstraintElement3::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  FOUR_C_THROW("You called Evaluate Neumann of constraint element.");
  return 0;
}

/*----------------------------------------------------------------------*
 * compute 3d normal */
void Discret::ELEMENTS::ConstraintElement3::compute_normal(
    const Core::LinAlg::Matrix<4, 3>& xc, Core::LinAlg::Matrix<3, 1>& elenorm)
{
  elenorm(0, 0) = -(xc(0, 2) * xc(1, 1)) + xc(0, 1) * xc(1, 2) + xc(0, 2) * xc(2, 1) -
                  xc(1, 2) * xc(2, 1) - xc(0, 1) * xc(2, 2) + xc(1, 1) * xc(2, 2);
  elenorm(1, 0) = xc(0, 2) * xc(1, 0) - xc(0, 0) * xc(1, 2) - xc(0, 2) * xc(2, 0) +
                  xc(1, 2) * xc(2, 0) + xc(0, 0) * xc(2, 2) - xc(1, 0) * xc(2, 2);
  elenorm(2, 0) = -(xc(0, 1) * xc(1, 0)) + xc(0, 0) * xc(1, 1) + xc(0, 1) * xc(2, 0) -
                  xc(1, 1) * xc(2, 0) - xc(0, 0) * xc(2, 1) + xc(1, 0) * xc(2, 1);
  return;
}

/*----------------------------------------------------------------------*
 * normal distance between fourth point and plane */
double Discret::ELEMENTS::ConstraintElement3::compute_normal_dist(
    const Core::LinAlg::Matrix<4, 3>& xc, const Core::LinAlg::Matrix<3, 1>& normal)
{
  return (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
             normal(2, 0) * (xc(0, 2) - xc(3, 2))) /
         (-normal.norm2());
}

/*----------------------------------------------------------------------*
 * first derivatives */
void Discret::ELEMENTS::ConstraintElement3::compute_first_deriv(
    const Core::LinAlg::Matrix<4, 3>& xc, Core::LinAlg::SerialDenseVector& elevector,
    const Core::LinAlg::Matrix<3, 1>& normal)
{
  double normsquare = pow(normal.norm2(), 2);
  double normcube = pow(normal.norm2(), 3);

  elevector[0] =
      (-((-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) + 2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
           (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
               normal(2, 0) * (xc(0, 2) - xc(3, 2)))) +
          2 * normsquare *
              (xc(1, 2) * (xc(2, 1) - xc(3, 1)) + xc(2, 2) * xc(3, 1) - xc(2, 1) * xc(3, 2) +
                  xc(1, 1) * (-xc(2, 2) + xc(3, 2)))) /
      (2. * normcube);

  elevector[1] =
      (-((-2 * (normal(2, 0)) * (xc(1, 0) - xc(2, 0)) - 2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
           (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
               normal(2, 0) * (xc(0, 2) - xc(3, 2)))) +
          2 * normsquare *
              (-(xc(2, 2) * xc(3, 0)) + xc(1, 2) * (-xc(2, 0) + xc(3, 0)) +
                  xc(1, 0) * (xc(2, 2) - xc(3, 2)) + xc(2, 0) * xc(3, 2))) /
      (2. * normcube);

  elevector[2] =
      (2 * normsquare *
              (xc(1, 1) * (xc(2, 0) - xc(3, 0)) + xc(2, 1) * xc(3, 0) - xc(2, 0) * xc(3, 1) +
                  xc(1, 0) * (-xc(2, 1) + xc(3, 1))) -
          (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) - 2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2)))) /
      (2. * normcube);

  elevector[3] =
      (-((-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) + 2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
           (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
               normal(2, 0) * (xc(0, 2) - xc(3, 2)))) +
          2 * normsquare *
              ((xc(0, 2) - xc(2, 2)) * (-xc(0, 1) + xc(3, 1)) +
                  (xc(0, 1) - xc(2, 1)) * (xc(0, 2) - xc(3, 2)))) /
      (2. * normcube);

  elevector[4] =
      (-((-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) - 2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
           (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
               normal(2, 0) * (xc(0, 2) - xc(3, 2)))) +
          2 * normsquare *
              ((xc(0, 2) - xc(2, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (-xc(0, 0) + xc(2, 0)) * (xc(0, 2) - xc(3, 2)))) /
      (2. * normcube);

  elevector[5] =
      (2 * normsquare *
              ((-xc(0, 1) + xc(2, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(2, 0)) * (xc(0, 1) - xc(3, 1))) -
          (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) - 2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2)))) /
      (2. * normcube);

  elevector[6] =
      (-((-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) + 2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
           (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
               normal(2, 0) * (xc(0, 2) - xc(3, 2)))) +
          2 * normsquare *
              ((xc(0, 2) - xc(1, 2)) * (xc(0, 1) - xc(3, 1)) +
                  (-xc(0, 1) + xc(1, 1)) * (xc(0, 2) - xc(3, 2)))) /
      (2. * normcube);

  elevector[7] =
      (-((-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) - 2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
           (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
               normal(2, 0) * (xc(0, 2) - xc(3, 2)))) +
          2 * normsquare *
              ((-xc(0, 2) + xc(1, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (xc(0, 2) - xc(3, 2)))) /
      (2. * normcube);

  elevector[8] =
      (2 * normsquare *
              ((xc(0, 1) - xc(1, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (-xc(0, 1) + xc(3, 1))) -
          (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) - 2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2)))) /
      (2. * normcube);

  elevector[9] = (-(xc(1, 2) * xc(2, 1)) + xc(0, 2) * (-xc(1, 1) + xc(2, 1)) +
                     xc(0, 1) * (xc(1, 2) - xc(2, 2)) + xc(1, 1) * xc(2, 2)) /
                 normal.norm2();

  elevector[10] = normal(1, 0) / normal.norm2();

  elevector[11] = (-(xc(1, 1) * xc(2, 0)) + xc(0, 1) * (-xc(1, 0) + xc(2, 0)) +
                      xc(0, 0) * (xc(1, 1) - xc(2, 1)) + xc(1, 0) * xc(2, 1)) /
                  normal.norm2();

  return;
}

/*----------------------------------------------------------------------*
 * second derivatives */
void Discret::ELEMENTS::ConstraintElement3::compute_second_deriv(
    const Core::LinAlg::Matrix<4, 3>& xc, Core::LinAlg::SerialDenseMatrix& elematrix,
    const Core::LinAlg::Matrix<3, 1>& normal)
{
  double normsquare = pow(normal.norm2(), 2);
  double normcube = pow(normal.norm2(), 3);
  double normpowfour = pow(normal.norm2(), 4);
  double normpowfive = pow(normal.norm2(), 5);

  elematrix(0, 0) =
      (-4 * normsquare * (pow(xc(1, 1) - xc(2, 1), 2) + pow(xc(1, 2) - xc(2, 2), 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              pow(-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                      2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2)),
                  2) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          4 * normsquare *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              (xc(1, 2) * (xc(2, 1) - xc(3, 1)) + xc(2, 2) * xc(3, 1) - xc(2, 1) * xc(3, 2) +
                  xc(1, 1) * (-xc(2, 2) + xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(0, 1) =
      (-4 * normsquare * (xc(1, 0) - xc(2, 0)) * (-xc(1, 1) + xc(2, 1)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(xc(2, 2) * xc(3, 0)) + xc(1, 2) * (-xc(2, 0) + xc(3, 0)) +
                  xc(1, 0) * (xc(2, 2) - xc(3, 2)) + xc(2, 0) * xc(3, 2)) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              (xc(1, 2) * (xc(2, 1) - xc(3, 1)) + xc(2, 2) * xc(3, 1) - xc(2, 1) * xc(3, 2) +
                  xc(1, 1) * (-xc(2, 2) + xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(0, 2) =
      (-2 * normsquare *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              (xc(1, 1) * (xc(2, 0) - xc(3, 0)) + xc(2, 1) * xc(3, 0) - xc(2, 0) * xc(3, 1) +
                  xc(1, 0) * (-xc(2, 1) + xc(3, 1))) -
          4 * normsquare * (xc(1, 0) - xc(2, 0)) * (-xc(1, 2) + xc(2, 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              (xc(1, 2) * (xc(2, 1) - xc(3, 1)) + xc(2, 2) * xc(3, 1) - xc(2, 1) * xc(3, 2) +
                  xc(1, 1) * (-xc(2, 2) + xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(0, 3) =
      (3 * (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) + 2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * (xc(0, 1) - xc(2, 1)) * (-xc(1, 1) + xc(2, 1)) +
                  2 * (xc(0, 2) - xc(2, 2)) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              ((xc(0, 2) - xc(2, 2)) * (-xc(0, 1) + xc(3, 1)) +
                  (xc(0, 1) - xc(2, 1)) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) +
                  2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              (xc(1, 2) * (xc(2, 1) - xc(3, 1)) + xc(2, 2) * xc(3, 1) - xc(2, 1) * xc(3, 2) +
                  xc(1, 1) * (-xc(2, 2) + xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(0, 4) =
      (-4 * normsquare *
              (-2 * xc(1, 1) * xc(2, 0) + xc(0, 1) * (-xc(1, 0) + xc(2, 0)) +
                  2 * xc(0, 0) * (xc(1, 1) - xc(2, 1)) + xc(1, 0) * xc(2, 1) +
                  xc(2, 0) * xc(2, 1)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              ((xc(0, 2) - xc(2, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (-xc(0, 0) + xc(2, 0)) * (xc(0, 2) - xc(3, 2))) +
          4 * normpowfour * (-xc(2, 2) + xc(3, 2)) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              (xc(1, 2) * (xc(2, 1) - xc(3, 1)) + xc(2, 2) * xc(3, 1) - xc(2, 1) * xc(3, 2) +
                  xc(1, 1) * (-xc(2, 2) + xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(0, 5) =
      (-2 * normsquare *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              ((-xc(0, 1) + xc(2, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(2, 0)) * (xc(0, 1) - xc(3, 1))) +
          4 * normpowfour * (xc(2, 1) - xc(3, 1)) -
          4 * normsquare *
              (-2 * xc(1, 2) * xc(2, 0) + xc(0, 2) * (-xc(1, 0) + xc(2, 0)) +
                  2 * xc(0, 0) * (xc(1, 2) - xc(2, 2)) + xc(1, 0) * xc(2, 2) +
                  xc(2, 0) * xc(2, 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              (xc(1, 2) * (xc(2, 1) - xc(3, 1)) + xc(2, 2) * xc(3, 1) - xc(2, 1) * xc(3, 2) +
                  xc(1, 1) * (-xc(2, 2) + xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(0, 6) =
      (-2 * normsquare *
              (2 * (xc(0, 1) - xc(1, 1)) * (xc(1, 1) - xc(2, 1)) +
                  2 * (xc(0, 2) - xc(1, 2)) * (xc(1, 2) - xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              ((xc(0, 2) - xc(1, 2)) * (xc(0, 1) - xc(3, 1)) +
                  (-xc(0, 1) + xc(1, 1)) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              (xc(1, 2) * (xc(2, 1) - xc(3, 1)) + xc(2, 2) * xc(3, 1) - xc(2, 1) * xc(3, 2) +
                  xc(1, 1) * (-xc(2, 2) + xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(0, 7) =
      (-4 * normsquare *
              (xc(1, 0) * xc(1, 1) + xc(0, 1) * (xc(1, 0) - xc(2, 0)) + xc(1, 1) * xc(2, 0) -
                  2 * xc(0, 0) * (xc(1, 1) - xc(2, 1)) - 2 * xc(1, 0) * xc(2, 1)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              ((-xc(0, 2) + xc(1, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (xc(0, 2) - xc(3, 2))) +
          4 * normpowfour * (xc(1, 2) - xc(3, 2)) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              (xc(1, 2) * (xc(2, 1) - xc(3, 1)) + xc(2, 2) * xc(3, 1) - xc(2, 1) * xc(3, 2) +
                  xc(1, 1) * (-xc(2, 2) + xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(0, 8) =
      (4 * normpowfour * (-xc(1, 1) + xc(3, 1)) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              ((xc(0, 1) - xc(1, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (-xc(0, 1) + xc(3, 1))) -
          4 * normsquare *
              (xc(1, 0) * xc(1, 2) + xc(0, 2) * (xc(1, 0) - xc(2, 0)) + xc(1, 2) * xc(2, 0) -
                  2 * xc(0, 0) * (xc(1, 2) - xc(2, 2)) - 2 * xc(1, 0) * xc(2, 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              (xc(1, 2) * (xc(2, 1) - xc(3, 1)) + xc(2, 2) * xc(3, 1) - xc(2, 1) * xc(3, 2) +
                  xc(1, 1) * (-xc(2, 2) + xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(0, 9) = -((-(xc(1, 2) * xc(2, 1)) + xc(0, 2) * (-xc(1, 1) + xc(2, 1)) +
                          xc(0, 1) * (xc(1, 2) - xc(2, 2)) + xc(1, 1) * xc(2, 2)) *
                        (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                            2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2)))) /
                    (2. * normcube);

  elematrix(0, 10) =
      (normal(0, 0) *
          (xc(0, 2) * xc(1, 1) * xc(1, 2) - xc(1, 0) * xc(1, 1) * xc(2, 0) +
              xc(1, 1) * pow(xc(2, 0), 2) +
              xc(0, 0) * (xc(1, 0) - xc(2, 0)) * (xc(1, 1) - xc(2, 1)) +
              pow(xc(1, 0), 2) * xc(2, 1) - xc(0, 2) * xc(1, 2) * xc(2, 1) +
              pow(xc(1, 2), 2) * xc(2, 1) - xc(1, 0) * xc(2, 0) * xc(2, 1) -
              xc(0, 2) * xc(1, 1) * xc(2, 2) - xc(1, 1) * xc(1, 2) * xc(2, 2) +
              xc(0, 2) * xc(2, 1) * xc(2, 2) - xc(1, 2) * xc(2, 1) * xc(2, 2) +
              xc(1, 1) * pow(xc(2, 2), 2) -
              xc(0, 1) * (pow(xc(1, 0), 2) + pow(xc(1, 2), 2) - 2 * xc(1, 0) * xc(2, 0) +
                             pow(xc(2, 0), 2) - 2 * xc(1, 2) * xc(2, 2) + pow(xc(2, 2), 2)))) /
      normcube;

  elematrix(0, 11) =
      -((normal(0, 0) *
            (-(xc(0, 1) * xc(1, 1) * xc(1, 2)) + xc(1, 0) * xc(1, 2) * xc(2, 0) -
                xc(1, 2) * pow(xc(2, 0), 2) + xc(0, 1) * xc(1, 2) * xc(2, 1) +
                xc(1, 1) * xc(1, 2) * xc(2, 1) - xc(1, 2) * pow(xc(2, 1), 2) +
                xc(0, 2) * (pow(xc(1, 0), 2) + pow(xc(1, 1), 2) - 2 * xc(1, 0) * xc(2, 0) +
                               pow(xc(2, 0), 2) - 2 * xc(1, 1) * xc(2, 1) + pow(xc(2, 1), 2)) -
                xc(0, 0) * (xc(1, 0) - xc(2, 0)) * (xc(1, 2) - xc(2, 2)) -
                pow(xc(1, 0), 2) * xc(2, 2) + xc(0, 1) * xc(1, 1) * xc(2, 2) -
                pow(xc(1, 1), 2) * xc(2, 2) + xc(1, 0) * xc(2, 0) * xc(2, 2) -
                xc(0, 1) * xc(2, 1) * xc(2, 2) + xc(1, 1) * xc(2, 1) * xc(2, 2))) /
          normcube);

  elematrix(1, 0) =
      (-4 * normsquare * (xc(1, 0) - xc(2, 0)) * (-xc(1, 1) + xc(2, 1)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(xc(2, 2) * xc(3, 0)) + xc(1, 2) * (-xc(2, 0) + xc(3, 0)) +
                  xc(1, 0) * (xc(2, 2) - xc(3, 2)) + xc(2, 0) * xc(3, 2)) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              (xc(1, 2) * (xc(2, 1) - xc(3, 1)) + xc(2, 2) * xc(3, 1) - xc(2, 1) * xc(3, 2) +
                  xc(1, 1) * (-xc(2, 2) + xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(1, 1) =
      (-4 * normsquare * (pow(xc(1, 0) - xc(2, 0), 2) + pow(xc(1, 2) - xc(2, 2), 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              pow(-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                      2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2)),
                  2) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          4 * normsquare *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(xc(2, 2) * xc(3, 0)) + xc(1, 2) * (-xc(2, 0) + xc(3, 0)) +
                  xc(1, 0) * (xc(2, 2) - xc(3, 2)) + xc(2, 0) * xc(3, 2))) /
      (4. * normpowfive);

  elematrix(1, 2) =
      (-2 * normsquare *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              (xc(1, 1) * (xc(2, 0) - xc(3, 0)) + xc(2, 1) * xc(3, 0) - xc(2, 0) * xc(3, 1) +
                  xc(1, 0) * (-xc(2, 1) + xc(3, 1))) -
          4 * normsquare * (xc(1, 1) - xc(2, 1)) * (-xc(1, 2) + xc(2, 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              (-(xc(2, 2) * xc(3, 0)) + xc(1, 2) * (-xc(2, 0) + xc(3, 0)) +
                  xc(1, 0) * (xc(2, 2) - xc(3, 2)) + xc(2, 0) * xc(3, 2))) /
      (4. * normpowfive);

  elematrix(1, 3) =
      (-4 * normsquare *
              (2 * xc(0, 1) * (xc(1, 0) - xc(2, 0)) + xc(1, 1) * xc(2, 0) -
                  2 * xc(1, 0) * xc(2, 1) + xc(2, 0) * xc(2, 1) +
                  xc(0, 0) * (-xc(1, 1) + xc(2, 1))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) +
                  2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              ((xc(0, 2) - xc(2, 2)) * (-xc(0, 1) + xc(3, 1)) +
                  (xc(0, 1) - xc(2, 1)) * (xc(0, 2) - xc(3, 2))) +
          4 * normpowfour * (xc(2, 2) - xc(3, 2)) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) +
                  2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              (-(xc(2, 2) * xc(3, 0)) + xc(1, 2) * (-xc(2, 0) + xc(3, 0)) +
                  xc(1, 0) * (xc(2, 2) - xc(3, 2)) + xc(2, 0) * xc(3, 2))) /
      (4. * normpowfive);

  elematrix(1, 4) =
      (3 * (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) - 2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * (xc(1, 0) - xc(2, 0)) * (-xc(0, 0) + xc(2, 0)) +
                  2 * (xc(0, 2) - xc(2, 2)) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              ((xc(0, 2) - xc(2, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (-xc(0, 0) + xc(2, 0)) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              (-(xc(2, 2) * xc(3, 0)) + xc(1, 2) * (-xc(2, 0) + xc(3, 0)) +
                  xc(1, 0) * (xc(2, 2) - xc(3, 2)) + xc(2, 0) * xc(3, 2))) /
      (4. * normpowfive);

  elematrix(1, 5) =
      (4 * normpowfour * (-xc(2, 0) + xc(3, 0)) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              ((-xc(0, 1) + xc(2, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(2, 0)) * (xc(0, 1) - xc(3, 1))) -
          4 * normsquare *
              (-2 * xc(1, 2) * xc(2, 1) + xc(0, 2) * (-xc(1, 1) + xc(2, 1)) +
                  2 * xc(0, 1) * (xc(1, 2) - xc(2, 2)) + xc(1, 1) * xc(2, 2) +
                  xc(2, 1) * xc(2, 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              (-(xc(2, 2) * xc(3, 0)) + xc(1, 2) * (-xc(2, 0) + xc(3, 0)) +
                  xc(1, 0) * (xc(2, 2) - xc(3, 2)) + xc(2, 0) * xc(3, 2))) /
      (4. * normpowfive);

  elematrix(1, 6) =
      (-4 * normsquare *
              (xc(1, 0) * xc(1, 1) - 2 * xc(0, 1) * (xc(1, 0) - xc(2, 0)) -
                  2 * xc(1, 1) * xc(2, 0) + xc(0, 0) * (xc(1, 1) - xc(2, 1)) +
                  xc(1, 0) * xc(2, 1)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              ((xc(0, 2) - xc(1, 2)) * (xc(0, 1) - xc(3, 1)) +
                  (-xc(0, 1) + xc(1, 1)) * (xc(0, 2) - xc(3, 2))) +
          4 * normpowfour * (-xc(1, 2) + xc(3, 2)) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              (-(xc(2, 2) * xc(3, 0)) + xc(1, 2) * (-xc(2, 0) + xc(3, 0)) +
                  xc(1, 0) * (xc(2, 2) - xc(3, 2)) + xc(2, 0) * xc(3, 2))) /
      (4. * normpowfive);

  elematrix(1, 7) =
      (-2 * normsquare *
              (2 * (xc(0, 0) - xc(1, 0)) * (xc(1, 0) - xc(2, 0)) +
                  2 * (xc(0, 2) - xc(1, 2)) * (xc(1, 2) - xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              ((-xc(0, 2) + xc(1, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              (-(xc(2, 2) * xc(3, 0)) + xc(1, 2) * (-xc(2, 0) + xc(3, 0)) +
                  xc(1, 0) * (xc(2, 2) - xc(3, 2)) + xc(2, 0) * xc(3, 2))) /
      (4. * normpowfive);

  elematrix(1, 8) =
      (4 * normpowfour * (xc(1, 0) - xc(3, 0)) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              ((xc(0, 1) - xc(1, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (-xc(0, 1) + xc(3, 1))) -
          4 * normsquare *
              (xc(1, 1) * xc(1, 2) + xc(0, 2) * (xc(1, 1) - xc(2, 1)) + xc(1, 2) * xc(2, 1) -
                  2 * xc(0, 1) * (xc(1, 2) - xc(2, 2)) - 2 * xc(1, 1) * xc(2, 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              (-(xc(2, 2) * xc(3, 0)) + xc(1, 2) * (-xc(2, 0) + xc(3, 0)) +
                  xc(1, 0) * (xc(2, 2) - xc(3, 2)) + xc(2, 0) * xc(3, 2))) /
      (4. * normpowfive);

  elematrix(1, 9) =
      (normal(1, 0) *
          (xc(0, 2) * xc(1, 0) * xc(1, 2) + pow(xc(1, 1), 2) * xc(2, 0) -
              xc(0, 2) * xc(1, 2) * xc(2, 0) + pow(xc(1, 2), 2) * xc(2, 0) +
              xc(0, 1) * (xc(1, 0) - xc(2, 0)) * (xc(1, 1) - xc(2, 1)) -
              xc(1, 0) * xc(1, 1) * xc(2, 1) - xc(1, 1) * xc(2, 0) * xc(2, 1) +
              xc(1, 0) * pow(xc(2, 1), 2) - xc(0, 2) * xc(1, 0) * xc(2, 2) -
              xc(1, 0) * xc(1, 2) * xc(2, 2) + xc(0, 2) * xc(2, 0) * xc(2, 2) -
              xc(1, 2) * xc(2, 0) * xc(2, 2) + xc(1, 0) * pow(xc(2, 2), 2) -
              xc(0, 0) * (pow(xc(1, 1), 2) + pow(xc(1, 2), 2) - 2 * xc(1, 1) * xc(2, 1) +
                             pow(xc(2, 1), 2) - 2 * xc(1, 2) * xc(2, 2) + pow(xc(2, 2), 2)))) /
      normcube;

  elematrix(1, 10) = -(normal(1, 0) * (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                                          2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2)))) /
                     (2. * normcube);

  elematrix(1, 11) =
      -((normal(1, 0) *
            (-(xc(0, 1) * xc(1, 1) * xc(1, 2)) + xc(1, 0) * xc(1, 2) * xc(2, 0) -
                xc(1, 2) * pow(xc(2, 0), 2) + xc(0, 1) * xc(1, 2) * xc(2, 1) +
                xc(1, 1) * xc(1, 2) * xc(2, 1) - xc(1, 2) * pow(xc(2, 1), 2) +
                xc(0, 2) * (pow(xc(1, 0), 2) + pow(xc(1, 1), 2) - 2 * xc(1, 0) * xc(2, 0) +
                               pow(xc(2, 0), 2) - 2 * xc(1, 1) * xc(2, 1) + pow(xc(2, 1), 2)) -
                xc(0, 0) * (xc(1, 0) - xc(2, 0)) * (xc(1, 2) - xc(2, 2)) -
                pow(xc(1, 0), 2) * xc(2, 2) + xc(0, 1) * xc(1, 1) * xc(2, 2) -
                pow(xc(1, 1), 2) * xc(2, 2) + xc(1, 0) * xc(2, 0) * xc(2, 2) -
                xc(0, 1) * xc(2, 1) * xc(2, 2) + xc(1, 1) * xc(2, 1) * xc(2, 2))) /
          normcube);

  elematrix(2, 0) =
      (-2 * normsquare *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              (xc(1, 1) * (xc(2, 0) - xc(3, 0)) + xc(2, 1) * xc(3, 0) - xc(2, 0) * xc(3, 1) +
                  xc(1, 0) * (-xc(2, 1) + xc(3, 1))) -
          4 * normsquare * (xc(1, 0) - xc(2, 0)) * (-xc(1, 2) + xc(2, 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              (xc(1, 2) * (xc(2, 1) - xc(3, 1)) + xc(2, 2) * xc(3, 1) - xc(2, 1) * xc(3, 2) +
                  xc(1, 1) * (-xc(2, 2) + xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(2, 1) =
      (-2 * normsquare *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              (xc(1, 1) * (xc(2, 0) - xc(3, 0)) + xc(2, 1) * xc(3, 0) - xc(2, 0) * xc(3, 1) +
                  xc(1, 0) * (-xc(2, 1) + xc(3, 1))) -
          4 * normsquare * (xc(1, 1) - xc(2, 1)) * (-xc(1, 2) + xc(2, 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              (-(xc(2, 2) * xc(3, 0)) + xc(1, 2) * (-xc(2, 0) + xc(3, 0)) +
                  xc(1, 0) * (xc(2, 2) - xc(3, 2)) + xc(2, 0) * xc(3, 2))) /
      (4. * normpowfive);

  elematrix(2, 2) =
      (-4 * normsquare *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              (xc(1, 1) * (xc(2, 0) - xc(3, 0)) + xc(2, 1) * xc(3, 0) - xc(2, 0) * xc(3, 1) +
                  xc(1, 0) * (-xc(2, 1) + xc(3, 1))) +
          3 *
              pow(2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                      2 * normal(0, 0) * (xc(1, 1) - xc(2, 1)),
                  2) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          4 * normsquare * (pow(xc(1, 0) - xc(2, 0), 2) + pow(xc(1, 1) - xc(2, 1), 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(2, 3) =
      (4 * normpowfour * (-xc(2, 1) + xc(3, 1)) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) +
                  2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              (xc(1, 1) * (xc(2, 0) - xc(3, 0)) + xc(2, 1) * xc(3, 0) - xc(2, 0) * xc(3, 1) +
                  xc(1, 0) * (-xc(2, 1) + xc(3, 1))) +
          3 *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) +
                  2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          4 * normsquare *
              (2 * xc(0, 2) * (xc(1, 0) - xc(2, 0)) + xc(1, 2) * xc(2, 0) -
                  2 * xc(1, 0) * xc(2, 2) + xc(2, 0) * xc(2, 2) +
                  xc(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              ((xc(0, 2) - xc(2, 2)) * (-xc(0, 1) + xc(3, 1)) +
                  (xc(0, 1) - xc(2, 1)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(2, 4) =
      (4 * normpowfour * (xc(2, 0) - xc(3, 0)) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              (xc(1, 1) * (xc(2, 0) - xc(3, 0)) + xc(2, 1) * xc(3, 0) - xc(2, 0) * xc(3, 1) +
                  xc(1, 0) * (-xc(2, 1) + xc(3, 1))) +
          3 *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          4 * normsquare *
              (2 * xc(0, 2) * (xc(1, 1) - xc(2, 1)) + xc(1, 2) * xc(2, 1) -
                  2 * xc(1, 1) * xc(2, 2) + xc(2, 1) * xc(2, 2) +
                  xc(0, 1) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              ((xc(0, 2) - xc(2, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (-xc(0, 0) + xc(2, 0)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(2, 5) =
      (-2 * normsquare *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              ((-xc(0, 1) + xc(2, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(2, 0)) * (xc(0, 1) - xc(3, 1))) -
          2 * normsquare *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              (xc(1, 1) * (xc(2, 0) - xc(3, 0)) + xc(2, 1) * xc(3, 0) - xc(2, 0) * xc(3, 1) +
                  xc(1, 0) * (-xc(2, 1) + xc(3, 1))) +
          3 *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * (xc(1, 0) - xc(2, 0)) * (-xc(0, 0) + xc(2, 0)) +
                  2 * (xc(1, 1) - xc(2, 1)) * (-xc(0, 1) + xc(2, 1))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(2, 6) =
      (4 * normpowfour * (xc(1, 1) - xc(3, 1)) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              (xc(1, 1) * (xc(2, 0) - xc(3, 0)) + xc(2, 1) * xc(3, 0) - xc(2, 0) * xc(3, 1) +
                  xc(1, 0) * (-xc(2, 1) + xc(3, 1))) +
          3 *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          4 * normsquare *
              (xc(1, 0) * xc(1, 2) - 2 * xc(0, 2) * (xc(1, 0) - xc(2, 0)) -
                  2 * xc(1, 2) * xc(2, 0) + xc(0, 0) * (xc(1, 2) - xc(2, 2)) +
                  xc(1, 0) * xc(2, 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              ((xc(0, 2) - xc(1, 2)) * (xc(0, 1) - xc(3, 1)) +
                  (-xc(0, 1) + xc(1, 1)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(2, 7) =
      (4 * normpowfour * (-xc(1, 0) + xc(3, 0)) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              (xc(1, 1) * (xc(2, 0) - xc(3, 0)) + xc(2, 1) * xc(3, 0) - xc(2, 0) * xc(3, 1) +
                  xc(1, 0) * (-xc(2, 1) + xc(3, 1))) +
          3 *
              (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          4 * normsquare *
              (xc(1, 1) * xc(1, 2) - 2 * xc(0, 2) * (xc(1, 1) - xc(2, 1)) -
                  2 * xc(1, 2) * xc(2, 1) + xc(0, 1) * (xc(1, 2) - xc(2, 2)) +
                  xc(1, 1) * xc(2, 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              ((-xc(0, 2) + xc(1, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(2, 8) =
      (-2 * normsquare *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              ((xc(0, 1) - xc(1, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (-xc(0, 1) + xc(3, 1))) -
          2 * normsquare *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              (xc(1, 1) * (xc(2, 0) - xc(3, 0)) + xc(2, 1) * xc(3, 0) - xc(2, 0) * xc(3, 1) +
                  xc(1, 0) * (-xc(2, 1) + xc(3, 1))) +
          3 *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * (xc(0, 0) - xc(1, 0)) * (xc(1, 0) - xc(2, 0)) +
                  2 * (xc(0, 1) - xc(1, 1)) * (xc(1, 1) - xc(2, 1))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(2, 9) =
      (normal(2, 0) *
          (xc(0, 2) * xc(1, 0) * xc(1, 2) + pow(xc(1, 1), 2) * xc(2, 0) -
              xc(0, 2) * xc(1, 2) * xc(2, 0) + pow(xc(1, 2), 2) * xc(2, 0) +
              xc(0, 1) * (xc(1, 0) - xc(2, 0)) * (xc(1, 1) - xc(2, 1)) -
              xc(1, 0) * xc(1, 1) * xc(2, 1) - xc(1, 1) * xc(2, 0) * xc(2, 1) +
              xc(1, 0) * pow(xc(2, 1), 2) - xc(0, 2) * xc(1, 0) * xc(2, 2) -
              xc(1, 0) * xc(1, 2) * xc(2, 2) + xc(0, 2) * xc(2, 0) * xc(2, 2) -
              xc(1, 2) * xc(2, 0) * xc(2, 2) + xc(1, 0) * pow(xc(2, 2), 2) -
              xc(0, 0) * (pow(xc(1, 1), 2) + pow(xc(1, 2), 2) - 2 * xc(1, 1) * xc(2, 1) +
                             pow(xc(2, 1), 2) - 2 * xc(1, 2) * xc(2, 2) + pow(xc(2, 2), 2)))) /
      normcube;

  elematrix(2, 10) =
      -((normal(2, 0) *
            (-(xc(0, 2) * xc(1, 1) * xc(1, 2)) + xc(1, 0) * xc(1, 1) * xc(2, 0) -
                xc(1, 1) * pow(xc(2, 0), 2) -
                xc(0, 0) * (xc(1, 0) - xc(2, 0)) * (xc(1, 1) - xc(2, 1)) -
                pow(xc(1, 0), 2) * xc(2, 1) + xc(0, 2) * xc(1, 2) * xc(2, 1) -
                pow(xc(1, 2), 2) * xc(2, 1) + xc(1, 0) * xc(2, 0) * xc(2, 1) +
                xc(0, 2) * xc(1, 1) * xc(2, 2) + xc(1, 1) * xc(1, 2) * xc(2, 2) -
                xc(0, 2) * xc(2, 1) * xc(2, 2) + xc(1, 2) * xc(2, 1) * xc(2, 2) -
                xc(1, 1) * pow(xc(2, 2), 2) +
                xc(0, 1) * (pow(xc(1, 0), 2) + pow(xc(1, 2), 2) - 2 * xc(1, 0) * xc(2, 0) +
                               pow(xc(2, 0), 2) - 2 * xc(1, 2) * xc(2, 2) + pow(xc(2, 2), 2)))) /
          normcube);

  elematrix(2, 11) =
      -((2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) - 2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
          (-(xc(1, 1) * xc(2, 0)) + xc(0, 1) * (-xc(1, 0) + xc(2, 0)) +
              xc(0, 0) * (xc(1, 1) - xc(2, 1)) + xc(1, 0) * xc(2, 1))) /
      (2. * normcube);

  elematrix(3, 0) =
      (3 * (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) + 2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * (xc(0, 1) - xc(2, 1)) * (-xc(1, 1) + xc(2, 1)) +
                  2 * (xc(0, 2) - xc(2, 2)) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              ((xc(0, 2) - xc(2, 2)) * (-xc(0, 1) + xc(3, 1)) +
                  (xc(0, 1) - xc(2, 1)) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) +
                  2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              (xc(1, 2) * (xc(2, 1) - xc(3, 1)) + xc(2, 2) * xc(3, 1) - xc(2, 1) * xc(3, 2) +
                  xc(1, 1) * (-xc(2, 2) + xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(3, 1) =
      (-4 * normsquare *
              (2 * xc(0, 1) * (xc(1, 0) - xc(2, 0)) + xc(1, 1) * xc(2, 0) -
                  2 * xc(1, 0) * xc(2, 1) + xc(2, 0) * xc(2, 1) +
                  xc(0, 0) * (-xc(1, 1) + xc(2, 1))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) +
                  2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              ((xc(0, 2) - xc(2, 2)) * (-xc(0, 1) + xc(3, 1)) +
                  (xc(0, 1) - xc(2, 1)) * (xc(0, 2) - xc(3, 2))) +
          4 * normpowfour * (xc(2, 2) - xc(3, 2)) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) +
                  2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              (-(xc(2, 2) * xc(3, 0)) + xc(1, 2) * (-xc(2, 0) + xc(3, 0)) +
                  xc(1, 0) * (xc(2, 2) - xc(3, 2)) + xc(2, 0) * xc(3, 2))) /
      (4. * normpowfive);

  elematrix(3, 2) =
      (4 * normpowfour * (-xc(2, 1) + xc(3, 1)) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) +
                  2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              (xc(1, 1) * (xc(2, 0) - xc(3, 0)) + xc(2, 1) * xc(3, 0) - xc(2, 0) * xc(3, 1) +
                  xc(1, 0) * (-xc(2, 1) + xc(3, 1))) +
          3 *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) +
                  2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          4 * normsquare *
              (2 * xc(0, 2) * (xc(1, 0) - xc(2, 0)) + xc(1, 2) * xc(2, 0) -
                  2 * xc(1, 0) * xc(2, 2) + xc(2, 0) * xc(2, 2) +
                  xc(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              ((xc(0, 2) - xc(2, 2)) * (-xc(0, 1) + xc(3, 1)) +
                  (xc(0, 1) - xc(2, 1)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(3, 3) =
      (3 *
              pow(-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) +
                      2 * normal(1, 0) * (xc(0, 2) - xc(2, 2)),
                  2) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          4 * normsquare * (pow(xc(0, 1) - xc(2, 1), 2) + pow(xc(0, 2) - xc(2, 2), 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          4 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) +
                  2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              ((xc(0, 2) - xc(2, 2)) * (-xc(0, 1) + xc(3, 1)) +
                  (xc(0, 1) - xc(2, 1)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(3, 4) =
      (-4 * normsquare * (-xc(0, 0) + xc(2, 0)) * (xc(0, 1) - xc(2, 1)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) +
                  2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) +
                  2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              ((xc(0, 2) - xc(2, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (-xc(0, 0) + xc(2, 0)) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              ((xc(0, 2) - xc(2, 2)) * (-xc(0, 1) + xc(3, 1)) +
                  (xc(0, 1) - xc(2, 1)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(3, 5) =
      (-2 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) +
                  2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              ((-xc(0, 1) + xc(2, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(2, 0)) * (xc(0, 1) - xc(3, 1))) +
          3 *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) +
                  2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          4 * normsquare * (-xc(0, 0) + xc(2, 0)) * (xc(0, 2) - xc(2, 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              ((xc(0, 2) - xc(2, 2)) * (-xc(0, 1) + xc(3, 1)) +
                  (xc(0, 1) - xc(2, 1)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(3, 6) =
      (3 *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) +
                  2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * (-xc(0, 1) + xc(1, 1)) * (xc(0, 1) - xc(2, 1)) +
                  2 * (-xc(0, 2) + xc(1, 2)) * (xc(0, 2) - xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) +
                  2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              ((xc(0, 2) - xc(1, 2)) * (xc(0, 1) - xc(3, 1)) +
                  (-xc(0, 1) + xc(1, 1)) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              ((xc(0, 2) - xc(2, 2)) * (-xc(0, 1) + xc(3, 1)) +
                  (xc(0, 1) - xc(2, 1)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(3, 7) =
      (-4 * normsquare *
              (-(xc(1, 1) * xc(2, 0)) + xc(0, 1) * (-2 * xc(1, 0) + xc(2, 0)) +
                  xc(0, 0) * (xc(0, 1) + xc(1, 1) - 2 * xc(2, 1)) + 2 * xc(1, 0) * xc(2, 1)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) +
                  2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) +
                  2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              ((-xc(0, 2) + xc(1, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              ((xc(0, 2) - xc(2, 2)) * (-xc(0, 1) + xc(3, 1)) +
                  (xc(0, 1) - xc(2, 1)) * (xc(0, 2) - xc(3, 2))) +
          4 * normpowfour * (-xc(0, 2) + xc(3, 2))) /
      (4. * normpowfive);

  elematrix(3, 8) =
      (4 * normpowfour * (xc(0, 1) - xc(3, 1)) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) +
                  2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              ((xc(0, 1) - xc(1, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (-xc(0, 1) + xc(3, 1))) +
          3 *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) +
                  2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          4 * normsquare *
              (-(xc(1, 2) * xc(2, 0)) + xc(0, 2) * (-2 * xc(1, 0) + xc(2, 0)) +
                  xc(0, 0) * (xc(0, 2) + xc(1, 2) - 2 * xc(2, 2)) + 2 * xc(1, 0) * xc(2, 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              ((xc(0, 2) - xc(2, 2)) * (-xc(0, 1) + xc(3, 1)) +
                  (xc(0, 1) - xc(2, 1)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(3, 9) =
      -((-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) + 2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
          (-(xc(1, 2) * xc(2, 1)) + xc(0, 2) * (-xc(1, 1) + xc(2, 1)) +
              xc(0, 1) * (xc(1, 2) - xc(2, 2)) + xc(1, 1) * xc(2, 2))) /
      (2. * normcube);

  elematrix(3, 10) =
      -((normal(0, 0) *
            (xc(0, 1) * xc(1, 0) * xc(2, 0) - xc(0, 1) * pow(xc(2, 0), 2) +
                xc(1, 1) * pow(xc(2, 0), 2) + pow(xc(0, 0), 2) * (xc(1, 1) - xc(2, 1)) +
                pow(xc(0, 2), 2) * (xc(1, 1) - xc(2, 1)) - xc(1, 0) * xc(2, 0) * xc(2, 1) +
                xc(0, 0) * (-2 * xc(1, 1) * xc(2, 0) + xc(0, 1) * (-xc(1, 0) + xc(2, 0)) +
                               (xc(1, 0) + xc(2, 0)) * xc(2, 1)) +
                xc(0, 1) * xc(1, 2) * xc(2, 2) - xc(1, 2) * xc(2, 1) * xc(2, 2) -
                xc(0, 1) * pow(xc(2, 2), 2) + xc(1, 1) * pow(xc(2, 2), 2) +
                xc(0, 2) * (xc(1, 2) * xc(2, 1) + (-2 * xc(1, 1) + xc(2, 1)) * xc(2, 2) +
                               xc(0, 1) * (-xc(1, 2) + xc(2, 2))))) /
          normcube);

  elematrix(3, 11) = -(
      (normal(0, 0) *
          (xc(0, 2) * xc(1, 0) * xc(2, 0) - xc(0, 2) * pow(xc(2, 0), 2) +
              xc(1, 2) * pow(xc(2, 0), 2) + xc(0, 2) * xc(1, 1) * xc(2, 1) -
              xc(0, 2) * pow(xc(2, 1), 2) + xc(1, 2) * pow(xc(2, 1), 2) +
              pow(xc(0, 0), 2) * (xc(1, 2) - xc(2, 2)) + pow(xc(0, 1), 2) * (xc(1, 2) - xc(2, 2)) -
              xc(1, 0) * xc(2, 0) * xc(2, 2) - xc(1, 1) * xc(2, 1) * xc(2, 2) +
              xc(0, 0) * (-2 * xc(1, 2) * xc(2, 0) + xc(0, 2) * (-xc(1, 0) + xc(2, 0)) +
                             (xc(1, 0) + xc(2, 0)) * xc(2, 2)) +
              xc(0, 1) * (-2 * xc(1, 2) * xc(2, 1) + xc(0, 2) * (-xc(1, 1) + xc(2, 1)) +
                             (xc(1, 1) + xc(2, 1)) * xc(2, 2)))) /
      normcube);

  elematrix(4, 0) =
      (-4 * normsquare *
              (-2 * xc(1, 1) * xc(2, 0) + xc(0, 1) * (-xc(1, 0) + xc(2, 0)) +
                  2 * xc(0, 0) * (xc(1, 1) - xc(2, 1)) + xc(1, 0) * xc(2, 1) +
                  xc(2, 0) * xc(2, 1)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              ((xc(0, 2) - xc(2, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (-xc(0, 0) + xc(2, 0)) * (xc(0, 2) - xc(3, 2))) +
          4 * normpowfour * (-xc(2, 2) + xc(3, 2)) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              (xc(1, 2) * (xc(2, 1) - xc(3, 1)) + xc(2, 2) * xc(3, 1) - xc(2, 1) * xc(3, 2) +
                  xc(1, 1) * (-xc(2, 2) + xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(4, 1) =
      (3 * (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) - 2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * (xc(1, 0) - xc(2, 0)) * (-xc(0, 0) + xc(2, 0)) +
                  2 * (xc(0, 2) - xc(2, 2)) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              ((xc(0, 2) - xc(2, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (-xc(0, 0) + xc(2, 0)) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              (-(xc(2, 2) * xc(3, 0)) + xc(1, 2) * (-xc(2, 0) + xc(3, 0)) +
                  xc(1, 0) * (xc(2, 2) - xc(3, 2)) + xc(2, 0) * xc(3, 2))) /
      (4. * normpowfive);

  elematrix(4, 2) =
      (4 * normpowfour * (xc(2, 0) - xc(3, 0)) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              (xc(1, 1) * (xc(2, 0) - xc(3, 0)) + xc(2, 1) * xc(3, 0) - xc(2, 0) * xc(3, 1) +
                  xc(1, 0) * (-xc(2, 1) + xc(3, 1))) +
          3 *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          4 * normsquare *
              (2 * xc(0, 2) * (xc(1, 1) - xc(2, 1)) + xc(1, 2) * xc(2, 1) -
                  2 * xc(1, 1) * xc(2, 2) + xc(2, 1) * xc(2, 2) +
                  xc(0, 1) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              ((xc(0, 2) - xc(2, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (-xc(0, 0) + xc(2, 0)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(4, 3) =
      (-4 * normsquare * (-xc(0, 0) + xc(2, 0)) * (xc(0, 1) - xc(2, 1)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) +
                  2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) +
                  2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              ((xc(0, 2) - xc(2, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (-xc(0, 0) + xc(2, 0)) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              ((xc(0, 2) - xc(2, 2)) * (-xc(0, 1) + xc(3, 1)) +
                  (xc(0, 1) - xc(2, 1)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(4, 4) =
      (3 *
              pow(-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                      2 * normal(0, 0) * (xc(0, 2) - xc(2, 2)),
                  2) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          4 * normsquare * (pow(xc(0, 0) - xc(2, 0), 2) + pow(xc(0, 2) - xc(2, 2), 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          4 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              ((xc(0, 2) - xc(2, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (-xc(0, 0) + xc(2, 0)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(4, 5) =
      (-2 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              ((-xc(0, 1) + xc(2, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(2, 0)) * (xc(0, 1) - xc(3, 1))) +
          3 *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          4 * normsquare * (-xc(0, 1) + xc(2, 1)) * (xc(0, 2) - xc(2, 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              ((xc(0, 2) - xc(2, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (-xc(0, 0) + xc(2, 0)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(4, 6) =
      (-4 * normsquare *
              (xc(0, 1) * (xc(1, 0) - 2 * xc(2, 0)) + 2 * xc(1, 1) * xc(2, 0) -
                  xc(1, 0) * xc(2, 1) + xc(0, 0) * (xc(0, 1) - 2 * xc(1, 1) + xc(2, 1))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              ((xc(0, 2) - xc(1, 2)) * (xc(0, 1) - xc(3, 1)) +
                  (-xc(0, 1) + xc(1, 1)) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              ((xc(0, 2) - xc(2, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (-xc(0, 0) + xc(2, 0)) * (xc(0, 2) - xc(3, 2))) +
          4 * normpowfour * (xc(0, 2) - xc(3, 2))) /
      (4. * normpowfive);

  elematrix(4, 7) =
      (3 * (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) - 2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * (xc(0, 0) - xc(1, 0)) * (-xc(0, 0) + xc(2, 0)) +
                  2 * (-xc(0, 2) + xc(1, 2)) * (xc(0, 2) - xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              ((-xc(0, 2) + xc(1, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              ((xc(0, 2) - xc(2, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (-xc(0, 0) + xc(2, 0)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(4, 8) =
      (4 * normpowfour * (-xc(0, 0) + xc(3, 0)) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              ((xc(0, 1) - xc(1, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (-xc(0, 1) + xc(3, 1))) +
          3 *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          4 * normsquare *
              (-(xc(1, 2) * xc(2, 1)) + xc(0, 2) * (-2 * xc(1, 1) + xc(2, 1)) +
                  xc(0, 1) * (xc(0, 2) + xc(1, 2) - 2 * xc(2, 2)) + 2 * xc(1, 1) * xc(2, 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              ((xc(0, 2) - xc(2, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (-xc(0, 0) + xc(2, 0)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(4, 9) =
      -((normal(1, 0) *
            (pow(xc(0, 1), 2) * (xc(1, 0) - xc(2, 0)) + pow(xc(0, 2), 2) * (xc(1, 0) - xc(2, 0)) +
                xc(0, 0) * xc(1, 1) * xc(2, 1) - xc(1, 1) * xc(2, 0) * xc(2, 1) -
                xc(0, 0) * pow(xc(2, 1), 2) + xc(1, 0) * pow(xc(2, 1), 2) +
                xc(0, 1) * (xc(1, 1) * xc(2, 0) + (-2 * xc(1, 0) + xc(2, 0)) * xc(2, 1) +
                               xc(0, 0) * (-xc(1, 1) + xc(2, 1))) +
                xc(0, 0) * xc(1, 2) * xc(2, 2) - xc(1, 2) * xc(2, 0) * xc(2, 2) -
                xc(0, 0) * pow(xc(2, 2), 2) + xc(1, 0) * pow(xc(2, 2), 2) +
                xc(0, 2) * (xc(1, 2) * xc(2, 0) + (-2 * xc(1, 0) + xc(2, 0)) * xc(2, 2) +
                               xc(0, 0) * (-xc(1, 2) + xc(2, 2))))) /
          normcube);

  elematrix(4, 10) = -(normal(1, 0) * (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                                          2 * normal(0, 0) * (xc(0, 2) - xc(2, 2)))) /
                     (2. * normcube);

  elematrix(4, 11) = -(
      (normal(1, 0) *
          (xc(0, 2) * xc(1, 0) * xc(2, 0) - xc(0, 2) * pow(xc(2, 0), 2) +
              xc(1, 2) * pow(xc(2, 0), 2) + xc(0, 2) * xc(1, 1) * xc(2, 1) -
              xc(0, 2) * pow(xc(2, 1), 2) + xc(1, 2) * pow(xc(2, 1), 2) +
              pow(xc(0, 0), 2) * (xc(1, 2) - xc(2, 2)) + pow(xc(0, 1), 2) * (xc(1, 2) - xc(2, 2)) -
              xc(1, 0) * xc(2, 0) * xc(2, 2) - xc(1, 1) * xc(2, 1) * xc(2, 2) +
              xc(0, 0) * (-2 * xc(1, 2) * xc(2, 0) + xc(0, 2) * (-xc(1, 0) + xc(2, 0)) +
                             (xc(1, 0) + xc(2, 0)) * xc(2, 2)) +
              xc(0, 1) * (-2 * xc(1, 2) * xc(2, 1) + xc(0, 2) * (-xc(1, 1) + xc(2, 1)) +
                             (xc(1, 1) + xc(2, 1)) * xc(2, 2)))) /
      normcube);

  elematrix(5, 0) =
      (-2 * normsquare *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              ((-xc(0, 1) + xc(2, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(2, 0)) * (xc(0, 1) - xc(3, 1))) +
          4 * normpowfour * (xc(2, 1) - xc(3, 1)) -
          4 * normsquare *
              (-2 * xc(1, 2) * xc(2, 0) + xc(0, 2) * (-xc(1, 0) + xc(2, 0)) +
                  2 * xc(0, 0) * (xc(1, 2) - xc(2, 2)) + xc(1, 0) * xc(2, 2) +
                  xc(2, 0) * xc(2, 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              (xc(1, 2) * (xc(2, 1) - xc(3, 1)) + xc(2, 2) * xc(3, 1) - xc(2, 1) * xc(3, 2) +
                  xc(1, 1) * (-xc(2, 2) + xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(5, 1) =
      (4 * normpowfour * (-xc(2, 0) + xc(3, 0)) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              ((-xc(0, 1) + xc(2, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(2, 0)) * (xc(0, 1) - xc(3, 1))) -
          4 * normsquare *
              (-2 * xc(1, 2) * xc(2, 1) + xc(0, 2) * (-xc(1, 1) + xc(2, 1)) +
                  2 * xc(0, 1) * (xc(1, 2) - xc(2, 2)) + xc(1, 1) * xc(2, 2) +
                  xc(2, 1) * xc(2, 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              (-(xc(2, 2) * xc(3, 0)) + xc(1, 2) * (-xc(2, 0) + xc(3, 0)) +
                  xc(1, 0) * (xc(2, 2) - xc(3, 2)) + xc(2, 0) * xc(3, 2))) /
      (4. * normpowfive);

  elematrix(5, 2) =
      (-2 * normsquare *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              ((-xc(0, 1) + xc(2, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(2, 0)) * (xc(0, 1) - xc(3, 1))) -
          2 * normsquare *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              (xc(1, 1) * (xc(2, 0) - xc(3, 0)) + xc(2, 1) * xc(3, 0) - xc(2, 0) * xc(3, 1) +
                  xc(1, 0) * (-xc(2, 1) + xc(3, 1))) +
          3 *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * (xc(1, 0) - xc(2, 0)) * (-xc(0, 0) + xc(2, 0)) +
                  2 * (xc(1, 1) - xc(2, 1)) * (-xc(0, 1) + xc(2, 1))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(5, 3) =
      (-2 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) +
                  2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              ((-xc(0, 1) + xc(2, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(2, 0)) * (xc(0, 1) - xc(3, 1))) +
          3 *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) +
                  2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          4 * normsquare * (-xc(0, 0) + xc(2, 0)) * (xc(0, 2) - xc(2, 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              ((xc(0, 2) - xc(2, 2)) * (-xc(0, 1) + xc(3, 1)) +
                  (xc(0, 1) - xc(2, 1)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(5, 4) =
      (-2 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              ((-xc(0, 1) + xc(2, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(2, 0)) * (xc(0, 1) - xc(3, 1))) +
          3 *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          4 * normsquare * (-xc(0, 1) + xc(2, 1)) * (xc(0, 2) - xc(2, 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              ((xc(0, 2) - xc(2, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (-xc(0, 0) + xc(2, 0)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(5, 5) =
      (-4 * normsquare *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              ((-xc(0, 1) + xc(2, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(2, 0)) * (xc(0, 1) - xc(3, 1))) -
          4 * normsquare * (pow(xc(0, 0) - xc(2, 0), 2) + pow(xc(0, 1) - xc(2, 1), 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              pow(2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                      2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1)),
                  2) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(5, 6) =
      (-2 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              ((-xc(0, 1) + xc(2, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(2, 0)) * (xc(0, 1) - xc(3, 1))) +
          4 * normpowfour * (-xc(0, 1) + xc(3, 1)) +
          3 *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          4 * normsquare *
              (xc(0, 2) * (xc(1, 0) - 2 * xc(2, 0)) + 2 * xc(1, 2) * xc(2, 0) -
                  xc(1, 0) * xc(2, 2) + xc(0, 0) * (xc(0, 2) - 2 * xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              ((xc(0, 2) - xc(1, 2)) * (xc(0, 1) - xc(3, 1)) +
                  (-xc(0, 1) + xc(1, 1)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(5, 7) =
      (4 * normpowfour * (xc(0, 0) - xc(3, 0)) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              ((-xc(0, 1) + xc(2, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(2, 0)) * (xc(0, 1) - xc(3, 1))) +
          3 *
              (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          4 * normsquare *
              (xc(0, 2) * (xc(1, 1) - 2 * xc(2, 1)) + 2 * xc(1, 2) * xc(2, 1) -
                  xc(1, 1) * xc(2, 2) + xc(0, 1) * (xc(0, 2) - 2 * xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              ((-xc(0, 2) + xc(1, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(5, 8) =
      (-2 * normsquare *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              ((-xc(0, 1) + xc(2, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(2, 0)) * (xc(0, 1) - xc(3, 1))) -
          2 * normsquare *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              ((xc(0, 1) - xc(1, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (-xc(0, 1) + xc(3, 1))) +
          3 *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * (xc(0, 0) - xc(1, 0)) * (-xc(0, 0) + xc(2, 0)) +
                  2 * (xc(0, 1) - xc(1, 1)) * (-xc(0, 1) + xc(2, 1))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(5, 9) =
      -((normal(2, 0) *
            (pow(xc(0, 1), 2) * (xc(1, 0) - xc(2, 0)) + pow(xc(0, 2), 2) * (xc(1, 0) - xc(2, 0)) +
                xc(0, 0) * xc(1, 1) * xc(2, 1) - xc(1, 1) * xc(2, 0) * xc(2, 1) -
                xc(0, 0) * pow(xc(2, 1), 2) + xc(1, 0) * pow(xc(2, 1), 2) +
                xc(0, 1) * (xc(1, 1) * xc(2, 0) + (-2 * xc(1, 0) + xc(2, 0)) * xc(2, 1) +
                               xc(0, 0) * (-xc(1, 1) + xc(2, 1))) +
                xc(0, 0) * xc(1, 2) * xc(2, 2) - xc(1, 2) * xc(2, 0) * xc(2, 2) -
                xc(0, 0) * pow(xc(2, 2), 2) + xc(1, 0) * pow(xc(2, 2), 2) +
                xc(0, 2) * (xc(1, 2) * xc(2, 0) + (-2 * xc(1, 0) + xc(2, 0)) * xc(2, 2) +
                               xc(0, 0) * (-xc(1, 2) + xc(2, 2))))) /
          normcube);

  elematrix(5, 10) =
      -((normal(2, 0) *
            (xc(0, 1) * xc(1, 0) * xc(2, 0) - xc(0, 1) * pow(xc(2, 0), 2) +
                xc(1, 1) * pow(xc(2, 0), 2) + pow(xc(0, 0), 2) * (xc(1, 1) - xc(2, 1)) +
                pow(xc(0, 2), 2) * (xc(1, 1) - xc(2, 1)) - xc(1, 0) * xc(2, 0) * xc(2, 1) +
                xc(0, 0) * (-2 * xc(1, 1) * xc(2, 0) + xc(0, 1) * (-xc(1, 0) + xc(2, 0)) +
                               (xc(1, 0) + xc(2, 0)) * xc(2, 1)) +
                xc(0, 1) * xc(1, 2) * xc(2, 2) - xc(1, 2) * xc(2, 1) * xc(2, 2) -
                xc(0, 1) * pow(xc(2, 2), 2) + xc(1, 1) * pow(xc(2, 2), 2) +
                xc(0, 2) * (xc(1, 2) * xc(2, 1) + (-2 * xc(1, 1) + xc(2, 1)) * xc(2, 2) +
                               xc(0, 1) * (-xc(1, 2) + xc(2, 2))))) /
          normcube);

  elematrix(5, 11) =
      -((-(xc(1, 1) * xc(2, 0)) + xc(0, 1) * (-xc(1, 0) + xc(2, 0)) +
            xc(0, 0) * (xc(1, 1) - xc(2, 1)) + xc(1, 0) * xc(2, 1)) *
          (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) - 2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1)))) /
      (2. * normcube);

  elematrix(6, 0) =
      (-2 * normsquare *
              (2 * (xc(0, 1) - xc(1, 1)) * (xc(1, 1) - xc(2, 1)) +
                  2 * (xc(0, 2) - xc(1, 2)) * (xc(1, 2) - xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              ((xc(0, 2) - xc(1, 2)) * (xc(0, 1) - xc(3, 1)) +
                  (-xc(0, 1) + xc(1, 1)) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              (xc(1, 2) * (xc(2, 1) - xc(3, 1)) + xc(2, 2) * xc(3, 1) - xc(2, 1) * xc(3, 2) +
                  xc(1, 1) * (-xc(2, 2) + xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(6, 1) =
      (-4 * normsquare *
              (xc(1, 0) * xc(1, 1) - 2 * xc(0, 1) * (xc(1, 0) - xc(2, 0)) -
                  2 * xc(1, 1) * xc(2, 0) + xc(0, 0) * (xc(1, 1) - xc(2, 1)) +
                  xc(1, 0) * xc(2, 1)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              ((xc(0, 2) - xc(1, 2)) * (xc(0, 1) - xc(3, 1)) +
                  (-xc(0, 1) + xc(1, 1)) * (xc(0, 2) - xc(3, 2))) +
          4 * normpowfour * (-xc(1, 2) + xc(3, 2)) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              (-(xc(2, 2) * xc(3, 0)) + xc(1, 2) * (-xc(2, 0) + xc(3, 0)) +
                  xc(1, 0) * (xc(2, 2) - xc(3, 2)) + xc(2, 0) * xc(3, 2))) /
      (4. * normpowfive);

  elematrix(6, 2) =
      (4 * normpowfour * (xc(1, 1) - xc(3, 1)) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              (xc(1, 1) * (xc(2, 0) - xc(3, 0)) + xc(2, 1) * xc(3, 0) - xc(2, 0) * xc(3, 1) +
                  xc(1, 0) * (-xc(2, 1) + xc(3, 1))) +
          3 *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          4 * normsquare *
              (xc(1, 0) * xc(1, 2) - 2 * xc(0, 2) * (xc(1, 0) - xc(2, 0)) -
                  2 * xc(1, 2) * xc(2, 0) + xc(0, 0) * (xc(1, 2) - xc(2, 2)) +
                  xc(1, 0) * xc(2, 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              ((xc(0, 2) - xc(1, 2)) * (xc(0, 1) - xc(3, 1)) +
                  (-xc(0, 1) + xc(1, 1)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(6, 3) =
      (3 *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) +
                  2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * (-xc(0, 1) + xc(1, 1)) * (xc(0, 1) - xc(2, 1)) +
                  2 * (-xc(0, 2) + xc(1, 2)) * (xc(0, 2) - xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) +
                  2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              ((xc(0, 2) - xc(1, 2)) * (xc(0, 1) - xc(3, 1)) +
                  (-xc(0, 1) + xc(1, 1)) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              ((xc(0, 2) - xc(2, 2)) * (-xc(0, 1) + xc(3, 1)) +
                  (xc(0, 1) - xc(2, 1)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(6, 4) =
      (-4 * normsquare *
              (xc(0, 1) * (xc(1, 0) - 2 * xc(2, 0)) + 2 * xc(1, 1) * xc(2, 0) -
                  xc(1, 0) * xc(2, 1) + xc(0, 0) * (xc(0, 1) - 2 * xc(1, 1) + xc(2, 1))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              ((xc(0, 2) - xc(1, 2)) * (xc(0, 1) - xc(3, 1)) +
                  (-xc(0, 1) + xc(1, 1)) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              ((xc(0, 2) - xc(2, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (-xc(0, 0) + xc(2, 0)) * (xc(0, 2) - xc(3, 2))) +
          4 * normpowfour * (xc(0, 2) - xc(3, 2))) /
      (4. * normpowfive);

  elematrix(6, 5) =
      (-2 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              ((-xc(0, 1) + xc(2, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(2, 0)) * (xc(0, 1) - xc(3, 1))) +
          4 * normpowfour * (-xc(0, 1) + xc(3, 1)) +
          3 *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          4 * normsquare *
              (xc(0, 2) * (xc(1, 0) - 2 * xc(2, 0)) + 2 * xc(1, 2) * xc(2, 0) -
                  xc(1, 0) * xc(2, 2) + xc(0, 0) * (xc(0, 2) - 2 * xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              ((xc(0, 2) - xc(1, 2)) * (xc(0, 1) - xc(3, 1)) +
                  (-xc(0, 1) + xc(1, 1)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(6, 6) =
      (-4 * normsquare * (pow(xc(0, 1) - xc(1, 1), 2) + pow(xc(0, 2) - xc(1, 2), 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              pow(-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                      2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2)),
                  2) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          4 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              ((xc(0, 2) - xc(1, 2)) * (xc(0, 1) - xc(3, 1)) +
                  (-xc(0, 1) + xc(1, 1)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(6, 7) =
      (-4 * normsquare * (xc(0, 0) - xc(1, 0)) * (-xc(0, 1) + xc(1, 1)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              ((-xc(0, 2) + xc(1, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              ((xc(0, 2) - xc(1, 2)) * (xc(0, 1) - xc(3, 1)) +
                  (-xc(0, 1) + xc(1, 1)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(6, 8) =
      (-2 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              ((xc(0, 1) - xc(1, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (-xc(0, 1) + xc(3, 1))) -
          4 * normsquare * (xc(0, 0) - xc(1, 0)) * (-xc(0, 2) + xc(1, 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              ((xc(0, 2) - xc(1, 2)) * (xc(0, 1) - xc(3, 1)) +
                  (-xc(0, 1) + xc(1, 1)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(6, 9) =
      -((-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) + 2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
          (-(xc(1, 2) * xc(2, 1)) + xc(0, 2) * (-xc(1, 1) + xc(2, 1)) +
              xc(0, 1) * (xc(1, 2) - xc(2, 2)) + xc(1, 1) * xc(2, 2))) /
      (2. * normcube);

  elematrix(6, 10) =
      (normal(0, 0) * (pow(xc(0, 2), 2) * xc(1, 1) - xc(0, 2) * xc(1, 1) * xc(1, 2) +
                          xc(1, 0) * xc(1, 1) * xc(2, 0) -
                          xc(0, 0) * (xc(0, 1) * (xc(1, 0) - xc(2, 0)) + xc(1, 1) * xc(2, 0) +
                                         xc(1, 0) * (xc(1, 1) - 2 * xc(2, 1))) +
                          pow(xc(0, 0), 2) * (xc(1, 1) - xc(2, 1)) - pow(xc(0, 2), 2) * xc(2, 1) -
                          pow(xc(1, 0), 2) * xc(2, 1) + 2 * xc(0, 2) * xc(1, 2) * xc(2, 1) -
                          pow(xc(1, 2), 2) * xc(2, 1) +
                          xc(0, 1) * (pow(xc(1, 0), 2) - xc(1, 0) * xc(2, 0) -
                                         (xc(0, 2) - xc(1, 2)) * (xc(1, 2) - xc(2, 2))) -
                          xc(0, 2) * xc(1, 1) * xc(2, 2) + xc(1, 1) * xc(1, 2) * xc(2, 2))) /
      normcube;

  elematrix(6, 11) = ((-(xc(1, 2) * xc(2, 1)) + xc(0, 2) * (-xc(1, 1) + xc(2, 1)) +
                          xc(0, 1) * (xc(1, 2) - xc(2, 2)) + xc(1, 1) * xc(2, 2)) *
                         (pow(xc(0, 1), 2) * xc(1, 2) - xc(0, 1) * xc(1, 1) * xc(1, 2) +
                             xc(1, 0) * xc(1, 2) * xc(2, 0) +
                             xc(0, 2) * (pow(xc(1, 0), 2) - xc(1, 0) * xc(2, 0) -
                                            (xc(0, 1) - xc(1, 1)) * (xc(1, 1) - xc(2, 1))) -
                             xc(0, 1) * xc(1, 2) * xc(2, 1) + xc(1, 1) * xc(1, 2) * xc(2, 1) -
                             xc(0, 0) * (xc(0, 2) * (xc(1, 0) - xc(2, 0)) + xc(1, 2) * xc(2, 0) +
                                            xc(1, 0) * (xc(1, 2) - 2 * xc(2, 2))) +
                             pow(xc(0, 0), 2) * (xc(1, 2) - xc(2, 2)) -
                             pow(xc(0, 1), 2) * xc(2, 2) - pow(xc(1, 0), 2) * xc(2, 2) +
                             2 * xc(0, 1) * xc(1, 1) * xc(2, 2) - pow(xc(1, 1), 2) * xc(2, 2))) /
                     normcube;

  elematrix(7, 0) =
      (-4 * normsquare *
              (xc(1, 0) * xc(1, 1) + xc(0, 1) * (xc(1, 0) - xc(2, 0)) + xc(1, 1) * xc(2, 0) -
                  2 * xc(0, 0) * (xc(1, 1) - xc(2, 1)) - 2 * xc(1, 0) * xc(2, 1)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              ((-xc(0, 2) + xc(1, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (xc(0, 2) - xc(3, 2))) +
          4 * normpowfour * (xc(1, 2) - xc(3, 2)) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              (xc(1, 2) * (xc(2, 1) - xc(3, 1)) + xc(2, 2) * xc(3, 1) - xc(2, 1) * xc(3, 2) +
                  xc(1, 1) * (-xc(2, 2) + xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(7, 1) =
      (-2 * normsquare *
              (2 * (xc(0, 0) - xc(1, 0)) * (xc(1, 0) - xc(2, 0)) +
                  2 * (xc(0, 2) - xc(1, 2)) * (xc(1, 2) - xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              ((-xc(0, 2) + xc(1, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              (-(xc(2, 2) * xc(3, 0)) + xc(1, 2) * (-xc(2, 0) + xc(3, 0)) +
                  xc(1, 0) * (xc(2, 2) - xc(3, 2)) + xc(2, 0) * xc(3, 2))) /
      (4. * normpowfive);

  elematrix(7, 2) =
      (4 * normpowfour * (-xc(1, 0) + xc(3, 0)) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              (xc(1, 1) * (xc(2, 0) - xc(3, 0)) + xc(2, 1) * xc(3, 0) - xc(2, 0) * xc(3, 1) +
                  xc(1, 0) * (-xc(2, 1) + xc(3, 1))) +
          3 *
              (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          4 * normsquare *
              (xc(1, 1) * xc(1, 2) - 2 * xc(0, 2) * (xc(1, 1) - xc(2, 1)) -
                  2 * xc(1, 2) * xc(2, 1) + xc(0, 1) * (xc(1, 2) - xc(2, 2)) +
                  xc(1, 1) * xc(2, 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              ((-xc(0, 2) + xc(1, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(7, 3) =
      (-4 * normsquare *
              (-(xc(1, 1) * xc(2, 0)) + xc(0, 1) * (-2 * xc(1, 0) + xc(2, 0)) +
                  xc(0, 0) * (xc(0, 1) + xc(1, 1) - 2 * xc(2, 1)) + 2 * xc(1, 0) * xc(2, 1)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) +
                  2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) +
                  2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              ((-xc(0, 2) + xc(1, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              ((xc(0, 2) - xc(2, 2)) * (-xc(0, 1) + xc(3, 1)) +
                  (xc(0, 1) - xc(2, 1)) * (xc(0, 2) - xc(3, 2))) +
          4 * normpowfour * (-xc(0, 2) + xc(3, 2))) /
      (4. * normpowfive);

  elematrix(7, 4) =
      (3 * (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) - 2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * (xc(0, 0) - xc(1, 0)) * (-xc(0, 0) + xc(2, 0)) +
                  2 * (-xc(0, 2) + xc(1, 2)) * (xc(0, 2) - xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              ((-xc(0, 2) + xc(1, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              ((xc(0, 2) - xc(2, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (-xc(0, 0) + xc(2, 0)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(7, 5) =
      (4 * normpowfour * (xc(0, 0) - xc(3, 0)) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              ((-xc(0, 1) + xc(2, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(2, 0)) * (xc(0, 1) - xc(3, 1))) +
          3 *
              (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          4 * normsquare *
              (xc(0, 2) * (xc(1, 1) - 2 * xc(2, 1)) + 2 * xc(1, 2) * xc(2, 1) -
                  xc(1, 1) * xc(2, 2) + xc(0, 1) * (xc(0, 2) - 2 * xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              ((-xc(0, 2) + xc(1, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(7, 6) =
      (-4 * normsquare * (xc(0, 0) - xc(1, 0)) * (-xc(0, 1) + xc(1, 1)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              ((-xc(0, 2) + xc(1, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              ((xc(0, 2) - xc(1, 2)) * (xc(0, 1) - xc(3, 1)) +
                  (-xc(0, 1) + xc(1, 1)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(7, 7) =
      (-4 * normsquare * (pow(xc(0, 0) - xc(1, 0), 2) + pow(xc(0, 2) - xc(1, 2), 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              pow(-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                      2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2)),
                  2) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          4 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              ((-xc(0, 2) + xc(1, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(7, 8) =
      (-2 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              ((xc(0, 1) - xc(1, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (-xc(0, 1) + xc(3, 1))) -
          4 * normsquare * (xc(0, 1) - xc(1, 1)) * (-xc(0, 2) + xc(1, 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              ((-xc(0, 2) + xc(1, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(7, 9) =
      (normal(1, 0) *
          (xc(0, 0) * pow(xc(1, 1), 2) + xc(0, 0) * pow(xc(1, 2), 2) +
              pow(xc(0, 1), 2) * (xc(1, 0) - xc(2, 0)) + pow(xc(0, 2), 2) * (xc(1, 0) - xc(2, 0)) -
              pow(xc(1, 1), 2) * xc(2, 0) - pow(xc(1, 2), 2) * xc(2, 0) -
              xc(0, 0) * xc(1, 1) * xc(2, 1) + xc(1, 0) * xc(1, 1) * xc(2, 1) -
              xc(0, 1) * (-2 * xc(1, 1) * xc(2, 0) + xc(0, 0) * (xc(1, 1) - xc(2, 1)) +
                             xc(1, 0) * (xc(1, 1) + xc(2, 1))) -
              xc(0, 0) * xc(1, 2) * xc(2, 2) + xc(1, 0) * xc(1, 2) * xc(2, 2) -
              xc(0, 2) * (-2 * xc(1, 2) * xc(2, 0) + xc(0, 0) * (xc(1, 2) - xc(2, 2)) +
                             xc(1, 0) * (xc(1, 2) + xc(2, 2))))) /
      normcube;

  elematrix(7, 10) = -(normal(1, 0) * (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                                          2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2)))) /
                     (2. * normcube);

  elematrix(7, 11) =
      (normal(1, 0) * (pow(xc(0, 1), 2) * xc(1, 2) - xc(0, 1) * xc(1, 1) * xc(1, 2) +
                          xc(1, 0) * xc(1, 2) * xc(2, 0) +
                          xc(0, 2) * (pow(xc(1, 0), 2) - xc(1, 0) * xc(2, 0) -
                                         (xc(0, 1) - xc(1, 1)) * (xc(1, 1) - xc(2, 1))) -
                          xc(0, 1) * xc(1, 2) * xc(2, 1) + xc(1, 1) * xc(1, 2) * xc(2, 1) -
                          xc(0, 0) * (xc(0, 2) * (xc(1, 0) - xc(2, 0)) + xc(1, 2) * xc(2, 0) +
                                         xc(1, 0) * (xc(1, 2) - 2 * xc(2, 2))) +
                          pow(xc(0, 0), 2) * (xc(1, 2) - xc(2, 2)) - pow(xc(0, 1), 2) * xc(2, 2) -
                          pow(xc(1, 0), 2) * xc(2, 2) + 2 * xc(0, 1) * xc(1, 1) * xc(2, 2) -
                          pow(xc(1, 1), 2) * xc(2, 2))) /
      normcube;

  elematrix(8, 0) =
      (4 * normpowfour * (-xc(1, 1) + xc(3, 1)) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              ((xc(0, 1) - xc(1, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (-xc(0, 1) + xc(3, 1))) -
          4 * normsquare *
              (xc(1, 0) * xc(1, 2) + xc(0, 2) * (xc(1, 0) - xc(2, 0)) + xc(1, 2) * xc(2, 0) -
                  2 * xc(0, 0) * (xc(1, 2) - xc(2, 2)) - 2 * xc(1, 0) * xc(2, 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                  2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              (xc(1, 2) * (xc(2, 1) - xc(3, 1)) + xc(2, 2) * xc(3, 1) - xc(2, 1) * xc(3, 2) +
                  xc(1, 1) * (-xc(2, 2) + xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(8, 1) =
      (4 * normpowfour * (xc(1, 0) - xc(3, 0)) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              ((xc(0, 1) - xc(1, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (-xc(0, 1) + xc(3, 1))) -
          4 * normsquare *
              (xc(1, 1) * xc(1, 2) + xc(0, 2) * (xc(1, 1) - xc(2, 1)) + xc(1, 2) * xc(2, 1) -
                  2 * xc(0, 1) * (xc(1, 2) - xc(2, 2)) - 2 * xc(1, 1) * xc(2, 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              (-(xc(2, 2) * xc(3, 0)) + xc(1, 2) * (-xc(2, 0) + xc(3, 0)) +
                  xc(1, 0) * (xc(2, 2) - xc(3, 2)) + xc(2, 0) * xc(3, 2))) /
      (4. * normpowfive);

  elematrix(8, 2) =
      (-2 * normsquare *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              ((xc(0, 1) - xc(1, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (-xc(0, 1) + xc(3, 1))) -
          2 * normsquare *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              (xc(1, 1) * (xc(2, 0) - xc(3, 0)) + xc(2, 1) * xc(3, 0) - xc(2, 0) * xc(3, 1) +
                  xc(1, 0) * (-xc(2, 1) + xc(3, 1))) +
          3 *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              (2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) -
                  2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * (xc(0, 0) - xc(1, 0)) * (xc(1, 0) - xc(2, 0)) +
                  2 * (xc(0, 1) - xc(1, 1)) * (xc(1, 1) - xc(2, 1))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(8, 3) =
      (4 * normpowfour * (xc(0, 1) - xc(3, 1)) -
          2 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) +
                  2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              ((xc(0, 1) - xc(1, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (-xc(0, 1) + xc(3, 1))) +
          3 *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              (-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) +
                  2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          4 * normsquare *
              (-(xc(1, 2) * xc(2, 0)) + xc(0, 2) * (-2 * xc(1, 0) + xc(2, 0)) +
                  xc(0, 0) * (xc(0, 2) + xc(1, 2) - 2 * xc(2, 2)) + 2 * xc(1, 0) * xc(2, 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              ((xc(0, 2) - xc(2, 2)) * (-xc(0, 1) + xc(3, 1)) +
                  (xc(0, 1) - xc(2, 1)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(8, 4) =
      (4 * normpowfour * (-xc(0, 0) + xc(3, 0)) -
          2 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              ((xc(0, 1) - xc(1, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (-xc(0, 1) + xc(3, 1))) +
          3 *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (xc(0, 2) - xc(2, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          4 * normsquare *
              (-(xc(1, 2) * xc(2, 1)) + xc(0, 2) * (-2 * xc(1, 1) + xc(2, 1)) +
                  xc(0, 1) * (xc(0, 2) + xc(1, 2) - 2 * xc(2, 2)) + 2 * xc(1, 1) * xc(2, 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              ((xc(0, 2) - xc(2, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (-xc(0, 0) + xc(2, 0)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(8, 5) =
      (-2 * normsquare *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              ((-xc(0, 1) + xc(2, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(2, 0)) * (xc(0, 1) - xc(3, 1))) -
          2 * normsquare *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              ((xc(0, 1) - xc(1, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (-xc(0, 1) + xc(3, 1))) +
          3 *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) -
                  2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * (xc(0, 0) - xc(1, 0)) * (-xc(0, 0) + xc(2, 0)) +
                  2 * (xc(0, 1) - xc(1, 1)) * (-xc(0, 1) + xc(2, 1))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(8, 6) =
      (-2 * normsquare *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              ((xc(0, 1) - xc(1, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (-xc(0, 1) + xc(3, 1))) -
          4 * normsquare * (xc(0, 0) - xc(1, 0)) * (-xc(0, 2) + xc(1, 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              (-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) +
                  2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              ((xc(0, 2) - xc(1, 2)) * (xc(0, 1) - xc(3, 1)) +
                  (-xc(0, 1) + xc(1, 1)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(8, 7) =
      (-2 * normsquare *
              (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              ((xc(0, 1) - xc(1, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (-xc(0, 1) + xc(3, 1))) -
          4 * normsquare * (xc(0, 1) - xc(1, 1)) * (-xc(0, 2) + xc(1, 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) +
          3 *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2))) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          2 * normsquare *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              ((-xc(0, 2) + xc(1, 2)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(8, 8) =
      (-4 * normsquare *
              (2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                  2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
              ((xc(0, 1) - xc(1, 1)) * (xc(0, 0) - xc(3, 0)) +
                  (xc(0, 0) - xc(1, 0)) * (-xc(0, 1) + xc(3, 1))) +
          3 *
              pow(2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) -
                      2 * normal(0, 0) * (xc(0, 1) - xc(1, 1)),
                  2) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2))) -
          4 * normsquare * (pow(xc(0, 0) - xc(1, 0), 2) + pow(xc(0, 1) - xc(1, 1), 2)) *
              (-(normal(0, 0) * (xc(0, 0) - xc(3, 0))) + normal(1, 0) * (-xc(0, 1) + xc(3, 1)) -
                  normal(2, 0) * (xc(0, 2) - xc(3, 2)))) /
      (4. * normpowfive);

  elematrix(8, 9) = -(
      (normal(2, 0) * (-(xc(0, 0) * pow(xc(1, 1), 2)) - xc(0, 0) * pow(xc(1, 2), 2) +
                          pow(xc(1, 1), 2) * xc(2, 0) + pow(xc(1, 2), 2) * xc(2, 0) +
                          pow(xc(0, 1), 2) * (-xc(1, 0) + xc(2, 0)) +
                          pow(xc(0, 2), 2) * (-xc(1, 0) + xc(2, 0)) +
                          xc(0, 0) * xc(1, 1) * xc(2, 1) - xc(1, 0) * xc(1, 1) * xc(2, 1) +
                          xc(0, 1) * (-2 * xc(1, 1) * xc(2, 0) + xc(0, 0) * (xc(1, 1) - xc(2, 1)) +
                                         xc(1, 0) * (xc(1, 1) + xc(2, 1))) +
                          xc(0, 0) * xc(1, 2) * xc(2, 2) - xc(1, 0) * xc(1, 2) * xc(2, 2) +
                          xc(0, 2) * (-2 * xc(1, 2) * xc(2, 0) + xc(0, 0) * (xc(1, 2) - xc(2, 2)) +
                                         xc(1, 0) * (xc(1, 2) + xc(2, 2))))) /
      normcube);

  elematrix(8, 10) =
      (normal(2, 0) * (pow(xc(0, 2), 2) * xc(1, 1) - xc(0, 2) * xc(1, 1) * xc(1, 2) +
                          xc(1, 0) * xc(1, 1) * xc(2, 0) -
                          xc(0, 0) * (xc(0, 1) * (xc(1, 0) - xc(2, 0)) + xc(1, 1) * xc(2, 0) +
                                         xc(1, 0) * (xc(1, 1) - 2 * xc(2, 1))) +
                          pow(xc(0, 0), 2) * (xc(1, 1) - xc(2, 1)) - pow(xc(0, 2), 2) * xc(2, 1) -
                          pow(xc(1, 0), 2) * xc(2, 1) + 2 * xc(0, 2) * xc(1, 2) * xc(2, 1) -
                          pow(xc(1, 2), 2) * xc(2, 1) +
                          xc(0, 1) * (pow(xc(1, 0), 2) - xc(1, 0) * xc(2, 0) -
                                         (xc(0, 2) - xc(1, 2)) * (xc(1, 2) - xc(2, 2))) -
                          xc(0, 2) * xc(1, 1) * xc(2, 2) + xc(1, 1) * xc(1, 2) * xc(2, 2))) /
      normcube;

  elematrix(8, 11) =
      -((2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) - 2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
          (-(xc(1, 1) * xc(2, 0)) + xc(0, 1) * (-xc(1, 0) + xc(2, 0)) +
              xc(0, 0) * (xc(1, 1) - xc(2, 1)) + xc(1, 0) * xc(2, 1))) /
      (2. * normcube);

  elematrix(9, 0) = -((-(xc(1, 2) * xc(2, 1)) + xc(0, 2) * (-xc(1, 1) + xc(2, 1)) +
                          xc(0, 1) * (xc(1, 2) - xc(2, 2)) + xc(1, 1) * xc(2, 2)) *
                        (-2 * normal(2, 0) * (-xc(1, 1) + xc(2, 1)) +
                            2 * normal(1, 0) * (-xc(1, 2) + xc(2, 2)))) /
                    (2. * normcube);

  elematrix(9, 1) =
      (normal(1, 0) *
          (xc(0, 2) * xc(1, 0) * xc(1, 2) + pow(xc(1, 1), 2) * xc(2, 0) -
              xc(0, 2) * xc(1, 2) * xc(2, 0) + pow(xc(1, 2), 2) * xc(2, 0) +
              xc(0, 1) * (xc(1, 0) - xc(2, 0)) * (xc(1, 1) - xc(2, 1)) -
              xc(1, 0) * xc(1, 1) * xc(2, 1) - xc(1, 1) * xc(2, 0) * xc(2, 1) +
              xc(1, 0) * pow(xc(2, 1), 2) - xc(0, 2) * xc(1, 0) * xc(2, 2) -
              xc(1, 0) * xc(1, 2) * xc(2, 2) + xc(0, 2) * xc(2, 0) * xc(2, 2) -
              xc(1, 2) * xc(2, 0) * xc(2, 2) + xc(1, 0) * pow(xc(2, 2), 2) -
              xc(0, 0) * (pow(xc(1, 1), 2) + pow(xc(1, 2), 2) - 2 * xc(1, 1) * xc(2, 1) +
                             pow(xc(2, 1), 2) - 2 * xc(1, 2) * xc(2, 2) + pow(xc(2, 2), 2)))) /
      normcube;

  elematrix(9, 2) =
      (normal(2, 0) *
          (xc(0, 2) * xc(1, 0) * xc(1, 2) + pow(xc(1, 1), 2) * xc(2, 0) -
              xc(0, 2) * xc(1, 2) * xc(2, 0) + pow(xc(1, 2), 2) * xc(2, 0) +
              xc(0, 1) * (xc(1, 0) - xc(2, 0)) * (xc(1, 1) - xc(2, 1)) -
              xc(1, 0) * xc(1, 1) * xc(2, 1) - xc(1, 1) * xc(2, 0) * xc(2, 1) +
              xc(1, 0) * pow(xc(2, 1), 2) - xc(0, 2) * xc(1, 0) * xc(2, 2) -
              xc(1, 0) * xc(1, 2) * xc(2, 2) + xc(0, 2) * xc(2, 0) * xc(2, 2) -
              xc(1, 2) * xc(2, 0) * xc(2, 2) + xc(1, 0) * pow(xc(2, 2), 2) -
              xc(0, 0) * (pow(xc(1, 1), 2) + pow(xc(1, 2), 2) - 2 * xc(1, 1) * xc(2, 1) +
                             pow(xc(2, 1), 2) - 2 * xc(1, 2) * xc(2, 2) + pow(xc(2, 2), 2)))) /
      normcube;

  elematrix(9, 3) =
      -((-2 * normal(2, 0) * (xc(0, 1) - xc(2, 1)) + 2 * normal(1, 0) * (xc(0, 2) - xc(2, 2))) *
          (-(xc(1, 2) * xc(2, 1)) + xc(0, 2) * (-xc(1, 1) + xc(2, 1)) +
              xc(0, 1) * (xc(1, 2) - xc(2, 2)) + xc(1, 1) * xc(2, 2))) /
      (2. * normcube);

  elematrix(9, 4) =
      -((normal(1, 0) *
            (pow(xc(0, 1), 2) * (xc(1, 0) - xc(2, 0)) + pow(xc(0, 2), 2) * (xc(1, 0) - xc(2, 0)) +
                xc(0, 0) * xc(1, 1) * xc(2, 1) - xc(1, 1) * xc(2, 0) * xc(2, 1) -
                xc(0, 0) * pow(xc(2, 1), 2) + xc(1, 0) * pow(xc(2, 1), 2) +
                xc(0, 1) * (xc(1, 1) * xc(2, 0) + (-2 * xc(1, 0) + xc(2, 0)) * xc(2, 1) +
                               xc(0, 0) * (-xc(1, 1) + xc(2, 1))) +
                xc(0, 0) * xc(1, 2) * xc(2, 2) - xc(1, 2) * xc(2, 0) * xc(2, 2) -
                xc(0, 0) * pow(xc(2, 2), 2) + xc(1, 0) * pow(xc(2, 2), 2) +
                xc(0, 2) * (xc(1, 2) * xc(2, 0) + (-2 * xc(1, 0) + xc(2, 0)) * xc(2, 2) +
                               xc(0, 0) * (-xc(1, 2) + xc(2, 2))))) /
          normcube);

  elematrix(9, 5) =
      -((normal(2, 0) *
            (pow(xc(0, 1), 2) * (xc(1, 0) - xc(2, 0)) + pow(xc(0, 2), 2) * (xc(1, 0) - xc(2, 0)) +
                xc(0, 0) * xc(1, 1) * xc(2, 1) - xc(1, 1) * xc(2, 0) * xc(2, 1) -
                xc(0, 0) * pow(xc(2, 1), 2) + xc(1, 0) * pow(xc(2, 1), 2) +
                xc(0, 1) * (xc(1, 1) * xc(2, 0) + (-2 * xc(1, 0) + xc(2, 0)) * xc(2, 1) +
                               xc(0, 0) * (-xc(1, 1) + xc(2, 1))) +
                xc(0, 0) * xc(1, 2) * xc(2, 2) - xc(1, 2) * xc(2, 0) * xc(2, 2) -
                xc(0, 0) * pow(xc(2, 2), 2) + xc(1, 0) * pow(xc(2, 2), 2) +
                xc(0, 2) * (xc(1, 2) * xc(2, 0) + (-2 * xc(1, 0) + xc(2, 0)) * xc(2, 2) +
                               xc(0, 0) * (-xc(1, 2) + xc(2, 2))))) /
          normcube);

  elematrix(9, 6) =
      -((-2 * normal(2, 0) * (-xc(0, 1) + xc(1, 1)) + 2 * normal(1, 0) * (-xc(0, 2) + xc(1, 2))) *
          (-(xc(1, 2) * xc(2, 1)) + xc(0, 2) * (-xc(1, 1) + xc(2, 1)) +
              xc(0, 1) * (xc(1, 2) - xc(2, 2)) + xc(1, 1) * xc(2, 2))) /
      (2. * normcube);

  elematrix(9, 7) =
      (normal(1, 0) *
          (xc(0, 0) * pow(xc(1, 1), 2) + xc(0, 0) * pow(xc(1, 2), 2) +
              pow(xc(0, 1), 2) * (xc(1, 0) - xc(2, 0)) + pow(xc(0, 2), 2) * (xc(1, 0) - xc(2, 0)) -
              pow(xc(1, 1), 2) * xc(2, 0) - pow(xc(1, 2), 2) * xc(2, 0) -
              xc(0, 0) * xc(1, 1) * xc(2, 1) + xc(1, 0) * xc(1, 1) * xc(2, 1) -
              xc(0, 1) * (-2 * xc(1, 1) * xc(2, 0) + xc(0, 0) * (xc(1, 1) - xc(2, 1)) +
                             xc(1, 0) * (xc(1, 1) + xc(2, 1))) -
              xc(0, 0) * xc(1, 2) * xc(2, 2) + xc(1, 0) * xc(1, 2) * xc(2, 2) -
              xc(0, 2) * (-2 * xc(1, 2) * xc(2, 0) + xc(0, 0) * (xc(1, 2) - xc(2, 2)) +
                             xc(1, 0) * (xc(1, 2) + xc(2, 2))))) /
      normcube;

  elematrix(9, 8) = -(
      (normal(2, 0) * (-(xc(0, 0) * pow(xc(1, 1), 2)) - xc(0, 0) * pow(xc(1, 2), 2) +
                          pow(xc(1, 1), 2) * xc(2, 0) + pow(xc(1, 2), 2) * xc(2, 0) +
                          pow(xc(0, 1), 2) * (-xc(1, 0) + xc(2, 0)) +
                          pow(xc(0, 2), 2) * (-xc(1, 0) + xc(2, 0)) +
                          xc(0, 0) * xc(1, 1) * xc(2, 1) - xc(1, 0) * xc(1, 1) * xc(2, 1) +
                          xc(0, 1) * (-2 * xc(1, 1) * xc(2, 0) + xc(0, 0) * (xc(1, 1) - xc(2, 1)) +
                                         xc(1, 0) * (xc(1, 1) + xc(2, 1))) +
                          xc(0, 0) * xc(1, 2) * xc(2, 2) - xc(1, 0) * xc(1, 2) * xc(2, 2) +
                          xc(0, 2) * (-2 * xc(1, 2) * xc(2, 0) + xc(0, 0) * (xc(1, 2) - xc(2, 2)) +
                                         xc(1, 0) * (xc(1, 2) + xc(2, 2))))) /
      normcube);

  elematrix(9, 9) = 0;

  elematrix(9, 10) = 0;

  elematrix(9, 11) = 0;

  elematrix(10, 0) =
      (normal(0, 0) *
          (xc(0, 2) * xc(1, 1) * xc(1, 2) - xc(1, 0) * xc(1, 1) * xc(2, 0) +
              xc(1, 1) * pow(xc(2, 0), 2) +
              xc(0, 0) * (xc(1, 0) - xc(2, 0)) * (xc(1, 1) - xc(2, 1)) +
              pow(xc(1, 0), 2) * xc(2, 1) - xc(0, 2) * xc(1, 2) * xc(2, 1) +
              pow(xc(1, 2), 2) * xc(2, 1) - xc(1, 0) * xc(2, 0) * xc(2, 1) -
              xc(0, 2) * xc(1, 1) * xc(2, 2) - xc(1, 1) * xc(1, 2) * xc(2, 2) +
              xc(0, 2) * xc(2, 1) * xc(2, 2) - xc(1, 2) * xc(2, 1) * xc(2, 2) +
              xc(1, 1) * pow(xc(2, 2), 2) -
              xc(0, 1) * (pow(xc(1, 0), 2) + pow(xc(1, 2), 2) - 2 * xc(1, 0) * xc(2, 0) +
                             pow(xc(2, 0), 2) - 2 * xc(1, 2) * xc(2, 2) + pow(xc(2, 2), 2)))) /
      normcube;

  elematrix(10, 1) = -(normal(1, 0) * (-2 * normal(2, 0) * (xc(1, 0) - xc(2, 0)) -
                                          2 * normal(0, 0) * (-xc(1, 2) + xc(2, 2)))) /
                     (2. * normcube);

  elematrix(10, 2) =
      -((normal(2, 0) *
            (-(xc(0, 2) * xc(1, 1) * xc(1, 2)) + xc(1, 0) * xc(1, 1) * xc(2, 0) -
                xc(1, 1) * pow(xc(2, 0), 2) -
                xc(0, 0) * (xc(1, 0) - xc(2, 0)) * (xc(1, 1) - xc(2, 1)) -
                pow(xc(1, 0), 2) * xc(2, 1) + xc(0, 2) * xc(1, 2) * xc(2, 1) -
                pow(xc(1, 2), 2) * xc(2, 1) + xc(1, 0) * xc(2, 0) * xc(2, 1) +
                xc(0, 2) * xc(1, 1) * xc(2, 2) + xc(1, 1) * xc(1, 2) * xc(2, 2) -
                xc(0, 2) * xc(2, 1) * xc(2, 2) + xc(1, 2) * xc(2, 1) * xc(2, 2) -
                xc(1, 1) * pow(xc(2, 2), 2) +
                xc(0, 1) * (pow(xc(1, 0), 2) + pow(xc(1, 2), 2) - 2 * xc(1, 0) * xc(2, 0) +
                               pow(xc(2, 0), 2) - 2 * xc(1, 2) * xc(2, 2) + pow(xc(2, 2), 2)))) /
          normcube);

  elematrix(10, 3) =
      -((normal(0, 0) *
            (xc(0, 1) * xc(1, 0) * xc(2, 0) - xc(0, 1) * pow(xc(2, 0), 2) +
                xc(1, 1) * pow(xc(2, 0), 2) + pow(xc(0, 0), 2) * (xc(1, 1) - xc(2, 1)) +
                pow(xc(0, 2), 2) * (xc(1, 1) - xc(2, 1)) - xc(1, 0) * xc(2, 0) * xc(2, 1) +
                xc(0, 0) * (-2 * xc(1, 1) * xc(2, 0) + xc(0, 1) * (-xc(1, 0) + xc(2, 0)) +
                               (xc(1, 0) + xc(2, 0)) * xc(2, 1)) +
                xc(0, 1) * xc(1, 2) * xc(2, 2) - xc(1, 2) * xc(2, 1) * xc(2, 2) -
                xc(0, 1) * pow(xc(2, 2), 2) + xc(1, 1) * pow(xc(2, 2), 2) +
                xc(0, 2) * (xc(1, 2) * xc(2, 1) + (-2 * xc(1, 1) + xc(2, 1)) * xc(2, 2) +
                               xc(0, 1) * (-xc(1, 2) + xc(2, 2))))) /
          normcube);

  elematrix(10, 4) = -(normal(1, 0) * (-2 * normal(2, 0) * (-xc(0, 0) + xc(2, 0)) -
                                          2 * normal(0, 0) * (xc(0, 2) - xc(2, 2)))) /
                     (2. * normcube);

  elematrix(10, 5) =
      -((normal(2, 0) *
            (xc(0, 1) * xc(1, 0) * xc(2, 0) - xc(0, 1) * pow(xc(2, 0), 2) +
                xc(1, 1) * pow(xc(2, 0), 2) + pow(xc(0, 0), 2) * (xc(1, 1) - xc(2, 1)) +
                pow(xc(0, 2), 2) * (xc(1, 1) - xc(2, 1)) - xc(1, 0) * xc(2, 0) * xc(2, 1) +
                xc(0, 0) * (-2 * xc(1, 1) * xc(2, 0) + xc(0, 1) * (-xc(1, 0) + xc(2, 0)) +
                               (xc(1, 0) + xc(2, 0)) * xc(2, 1)) +
                xc(0, 1) * xc(1, 2) * xc(2, 2) - xc(1, 2) * xc(2, 1) * xc(2, 2) -
                xc(0, 1) * pow(xc(2, 2), 2) + xc(1, 1) * pow(xc(2, 2), 2) +
                xc(0, 2) * (xc(1, 2) * xc(2, 1) + (-2 * xc(1, 1) + xc(2, 1)) * xc(2, 2) +
                               xc(0, 1) * (-xc(1, 2) + xc(2, 2))))) /
          normcube);

  elematrix(10, 6) =
      (normal(0, 0) * (pow(xc(0, 2), 2) * xc(1, 1) - xc(0, 2) * xc(1, 1) * xc(1, 2) +
                          xc(1, 0) * xc(1, 1) * xc(2, 0) -
                          xc(0, 0) * (xc(0, 1) * (xc(1, 0) - xc(2, 0)) + xc(1, 1) * xc(2, 0) +
                                         xc(1, 0) * (xc(1, 1) - 2 * xc(2, 1))) +
                          pow(xc(0, 0), 2) * (xc(1, 1) - xc(2, 1)) - pow(xc(0, 2), 2) * xc(2, 1) -
                          pow(xc(1, 0), 2) * xc(2, 1) + 2 * xc(0, 2) * xc(1, 2) * xc(2, 1) -
                          pow(xc(1, 2), 2) * xc(2, 1) +
                          xc(0, 1) * (pow(xc(1, 0), 2) - xc(1, 0) * xc(2, 0) -
                                         (xc(0, 2) - xc(1, 2)) * (xc(1, 2) - xc(2, 2))) -
                          xc(0, 2) * xc(1, 1) * xc(2, 2) + xc(1, 1) * xc(1, 2) * xc(2, 2))) /
      normcube;

  elematrix(10, 7) = -(normal(1, 0) * (-2 * normal(2, 0) * (xc(0, 0) - xc(1, 0)) -
                                          2 * normal(0, 0) * (-xc(0, 2) + xc(1, 2)))) /
                     (2. * normcube);

  elematrix(10, 8) =
      (normal(2, 0) * (pow(xc(0, 2), 2) * xc(1, 1) - xc(0, 2) * xc(1, 1) * xc(1, 2) +
                          xc(1, 0) * xc(1, 1) * xc(2, 0) -
                          xc(0, 0) * (xc(0, 1) * (xc(1, 0) - xc(2, 0)) + xc(1, 1) * xc(2, 0) +
                                         xc(1, 0) * (xc(1, 1) - 2 * xc(2, 1))) +
                          pow(xc(0, 0), 2) * (xc(1, 1) - xc(2, 1)) - pow(xc(0, 2), 2) * xc(2, 1) -
                          pow(xc(1, 0), 2) * xc(2, 1) + 2 * xc(0, 2) * xc(1, 2) * xc(2, 1) -
                          pow(xc(1, 2), 2) * xc(2, 1) +
                          xc(0, 1) * (pow(xc(1, 0), 2) - xc(1, 0) * xc(2, 0) -
                                         (xc(0, 2) - xc(1, 2)) * (xc(1, 2) - xc(2, 2))) -
                          xc(0, 2) * xc(1, 1) * xc(2, 2) + xc(1, 1) * xc(1, 2) * xc(2, 2))) /
      normcube;

  elematrix(10, 9) = 0;

  elematrix(10, 10) = 0;

  elematrix(10, 11) = 0;

  elematrix(11, 0) =
      -((normal(0, 0) *
            (-(xc(0, 1) * xc(1, 1) * xc(1, 2)) + xc(1, 0) * xc(1, 2) * xc(2, 0) -
                xc(1, 2) * pow(xc(2, 0), 2) + xc(0, 1) * xc(1, 2) * xc(2, 1) +
                xc(1, 1) * xc(1, 2) * xc(2, 1) - xc(1, 2) * pow(xc(2, 1), 2) +
                xc(0, 2) * (pow(xc(1, 0), 2) + pow(xc(1, 1), 2) - 2 * xc(1, 0) * xc(2, 0) +
                               pow(xc(2, 0), 2) - 2 * xc(1, 1) * xc(2, 1) + pow(xc(2, 1), 2)) -
                xc(0, 0) * (xc(1, 0) - xc(2, 0)) * (xc(1, 2) - xc(2, 2)) -
                pow(xc(1, 0), 2) * xc(2, 2) + xc(0, 1) * xc(1, 1) * xc(2, 2) -
                pow(xc(1, 1), 2) * xc(2, 2) + xc(1, 0) * xc(2, 0) * xc(2, 2) -
                xc(0, 1) * xc(2, 1) * xc(2, 2) + xc(1, 1) * xc(2, 1) * xc(2, 2))) /
          normcube);

  elematrix(11, 1) =
      -((normal(1, 0) *
            (-(xc(0, 1) * xc(1, 1) * xc(1, 2)) + xc(1, 0) * xc(1, 2) * xc(2, 0) -
                xc(1, 2) * pow(xc(2, 0), 2) + xc(0, 1) * xc(1, 2) * xc(2, 1) +
                xc(1, 1) * xc(1, 2) * xc(2, 1) - xc(1, 2) * pow(xc(2, 1), 2) +
                xc(0, 2) * (pow(xc(1, 0), 2) + pow(xc(1, 1), 2) - 2 * xc(1, 0) * xc(2, 0) +
                               pow(xc(2, 0), 2) - 2 * xc(1, 1) * xc(2, 1) + pow(xc(2, 1), 2)) -
                xc(0, 0) * (xc(1, 0) - xc(2, 0)) * (xc(1, 2) - xc(2, 2)) -
                pow(xc(1, 0), 2) * xc(2, 2) + xc(0, 1) * xc(1, 1) * xc(2, 2) -
                pow(xc(1, 1), 2) * xc(2, 2) + xc(1, 0) * xc(2, 0) * xc(2, 2) -
                xc(0, 1) * xc(2, 1) * xc(2, 2) + xc(1, 1) * xc(2, 1) * xc(2, 2))) /
          normcube);

  elematrix(11, 2) =
      -((2 * normal(1, 0) * (xc(1, 0) - xc(2, 0)) - 2 * normal(0, 0) * (xc(1, 1) - xc(2, 1))) *
          (-(xc(1, 1) * xc(2, 0)) + xc(0, 1) * (-xc(1, 0) + xc(2, 0)) +
              xc(0, 0) * (xc(1, 1) - xc(2, 1)) + xc(1, 0) * xc(2, 1))) /
      (2. * normcube);

  elematrix(11, 3) = -(
      (normal(0, 0) *
          (xc(0, 2) * xc(1, 0) * xc(2, 0) - xc(0, 2) * pow(xc(2, 0), 2) +
              xc(1, 2) * pow(xc(2, 0), 2) + xc(0, 2) * xc(1, 1) * xc(2, 1) -
              xc(0, 2) * pow(xc(2, 1), 2) + xc(1, 2) * pow(xc(2, 1), 2) +
              pow(xc(0, 0), 2) * (xc(1, 2) - xc(2, 2)) + pow(xc(0, 1), 2) * (xc(1, 2) - xc(2, 2)) -
              xc(1, 0) * xc(2, 0) * xc(2, 2) - xc(1, 1) * xc(2, 1) * xc(2, 2) +
              xc(0, 0) * (-2 * xc(1, 2) * xc(2, 0) + xc(0, 2) * (-xc(1, 0) + xc(2, 0)) +
                             (xc(1, 0) + xc(2, 0)) * xc(2, 2)) +
              xc(0, 1) * (-2 * xc(1, 2) * xc(2, 1) + xc(0, 2) * (-xc(1, 1) + xc(2, 1)) +
                             (xc(1, 1) + xc(2, 1)) * xc(2, 2)))) /
      normcube);

  elematrix(11, 4) = -(
      (normal(1, 0) *
          (xc(0, 2) * xc(1, 0) * xc(2, 0) - xc(0, 2) * pow(xc(2, 0), 2) +
              xc(1, 2) * pow(xc(2, 0), 2) + xc(0, 2) * xc(1, 1) * xc(2, 1) -
              xc(0, 2) * pow(xc(2, 1), 2) + xc(1, 2) * pow(xc(2, 1), 2) +
              pow(xc(0, 0), 2) * (xc(1, 2) - xc(2, 2)) + pow(xc(0, 1), 2) * (xc(1, 2) - xc(2, 2)) -
              xc(1, 0) * xc(2, 0) * xc(2, 2) - xc(1, 1) * xc(2, 1) * xc(2, 2) +
              xc(0, 0) * (-2 * xc(1, 2) * xc(2, 0) + xc(0, 2) * (-xc(1, 0) + xc(2, 0)) +
                             (xc(1, 0) + xc(2, 0)) * xc(2, 2)) +
              xc(0, 1) * (-2 * xc(1, 2) * xc(2, 1) + xc(0, 2) * (-xc(1, 1) + xc(2, 1)) +
                             (xc(1, 1) + xc(2, 1)) * xc(2, 2)))) /
      normcube);

  elematrix(11, 5) =
      -((-(xc(1, 1) * xc(2, 0)) + xc(0, 1) * (-xc(1, 0) + xc(2, 0)) +
            xc(0, 0) * (xc(1, 1) - xc(2, 1)) + xc(1, 0) * xc(2, 1)) *
          (2 * normal(1, 0) * (-xc(0, 0) + xc(2, 0)) - 2 * normal(0, 0) * (-xc(0, 1) + xc(2, 1)))) /
      (2. * normcube);

  elematrix(11, 6) = ((-(xc(1, 2) * xc(2, 1)) + xc(0, 2) * (-xc(1, 1) + xc(2, 1)) +
                          xc(0, 1) * (xc(1, 2) - xc(2, 2)) + xc(1, 1) * xc(2, 2)) *
                         (pow(xc(0, 1), 2) * xc(1, 2) - xc(0, 1) * xc(1, 1) * xc(1, 2) +
                             xc(1, 0) * xc(1, 2) * xc(2, 0) +
                             xc(0, 2) * (pow(xc(1, 0), 2) - xc(1, 0) * xc(2, 0) -
                                            (xc(0, 1) - xc(1, 1)) * (xc(1, 1) - xc(2, 1))) -
                             xc(0, 1) * xc(1, 2) * xc(2, 1) + xc(1, 1) * xc(1, 2) * xc(2, 1) -
                             xc(0, 0) * (xc(0, 2) * (xc(1, 0) - xc(2, 0)) + xc(1, 2) * xc(2, 0) +
                                            xc(1, 0) * (xc(1, 2) - 2 * xc(2, 2))) +
                             pow(xc(0, 0), 2) * (xc(1, 2) - xc(2, 2)) -
                             pow(xc(0, 1), 2) * xc(2, 2) - pow(xc(1, 0), 2) * xc(2, 2) +
                             2 * xc(0, 1) * xc(1, 1) * xc(2, 2) - pow(xc(1, 1), 2) * xc(2, 2))) /
                     normcube;

  elematrix(11, 7) =
      (normal(1, 0) * (pow(xc(0, 1), 2) * xc(1, 2) - xc(0, 1) * xc(1, 1) * xc(1, 2) +
                          xc(1, 0) * xc(1, 2) * xc(2, 0) +
                          xc(0, 2) * (pow(xc(1, 0), 2) - xc(1, 0) * xc(2, 0) -
                                         (xc(0, 1) - xc(1, 1)) * (xc(1, 1) - xc(2, 1))) -
                          xc(0, 1) * xc(1, 2) * xc(2, 1) + xc(1, 1) * xc(1, 2) * xc(2, 1) -
                          xc(0, 0) * (xc(0, 2) * (xc(1, 0) - xc(2, 0)) + xc(1, 2) * xc(2, 0) +
                                         xc(1, 0) * (xc(1, 2) - 2 * xc(2, 2))) +
                          pow(xc(0, 0), 2) * (xc(1, 2) - xc(2, 2)) - pow(xc(0, 1), 2) * xc(2, 2) -
                          pow(xc(1, 0), 2) * xc(2, 2) + 2 * xc(0, 1) * xc(1, 1) * xc(2, 2) -
                          pow(xc(1, 1), 2) * xc(2, 2))) /
      normcube;

  elematrix(11, 8) =
      -((2 * normal(1, 0) * (xc(0, 0) - xc(1, 0)) - 2 * normal(0, 0) * (xc(0, 1) - xc(1, 1))) *
          (-(xc(1, 1) * xc(2, 0)) + xc(0, 1) * (-xc(1, 0) + xc(2, 0)) +
              xc(0, 0) * (xc(1, 1) - xc(2, 1)) + xc(1, 0) * xc(2, 1))) /
      (2. * normcube);

  elematrix(11, 9) = 0;

  elematrix(11, 10) = 0;

  elematrix(11, 11) = 0;
  return;
}

double Discret::ELEMENTS::ConstraintElement3::compute_weighted_distance(
    const std::vector<double> disp, const std::vector<double> direct)
{
  // norm of direct
  double norm = sqrt(pow(direct.at(0), 2) + pow(direct.at(1), 2) + pow(direct.at(2), 2));
  double result = 0.0;

  for (int i = 0; i < 3; i++)
  {
    result += (disp.at(i) - disp.at(i + 3)) * direct.at(i);
  }
  result /= norm;
  return result;
}

double Discret::ELEMENTS::ConstraintElement3::compute_weighted_distance(
    const Core::LinAlg::Matrix<2, 3> disp, const std::vector<double> direct)
{
  // norm of direct
  double norm = sqrt(pow(direct.at(0), 2) + pow(direct.at(1), 2) + pow(direct.at(2), 2));
  double result = 0.0;

  for (int i = 0; i < 3; i++)
  {
    result += (disp(0, i) - disp(1, i)) * direct.at(i);
  }
  result /= norm;
  return result;
}

void Discret::ELEMENTS::ConstraintElement3::compute_first_deriv_weighted_distance(
    Core::LinAlg::SerialDenseVector& elevector, const std::vector<double> direct)
{
  // norm of direct
  double norm = sqrt(pow(direct.at(0), 2) + pow(direct.at(1), 2) + pow(direct.at(2), 2));

  for (int i = 0; i < 3; i++)
  {
    elevector(i) = -direct.at(i) / norm;
    elevector(3 + i) = direct.at(i) / norm;
  }

  return;
}

FOUR_C_NAMESPACE_CLOSE
