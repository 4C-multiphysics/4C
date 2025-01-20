// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_bele_bele3.hpp"
#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function.hpp"

FOUR_C_NAMESPACE_OPEN



/*----------------------------------------------------------------------*
 |  evaluate the element (public)                            g.bau 07/07|
 *----------------------------------------------------------------------*/
int Discret::Elements::Bele3Line::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  Discret::Elements::Bele3Line::ActionType act = Bele3Line::none;
  std::string action = params.get<std::string>("action", "none");
  if (action == "none")
    FOUR_C_THROW("No action supplied");
  else if (action == "integrate_Shapefunction")
    act = Bele3Line::integrate_Shapefunction;
  else
    FOUR_C_THROW("Unknown type of action for Bele3Line");

  switch (act)
  {
    case integrate_Shapefunction:
    {
      std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp;
      std::vector<double> mydispnp;

      //      if (parent_->IsMoving())
      {
        dispnp = discretization.get_state("dispnp");
        if (dispnp != nullptr)
        {
          mydispnp.resize(lm.size());
          Core::FE::extract_my_values(*dispnp, mydispnp, lm);
        }
        else
        {
          std::cout << "could not get displacement vector to compute current positions!"
                    << std::endl;
        }
      }

      integrate_shape_function(params, discretization, lm, elevec1, mydispnp);
      break;
    }
    default:
      FOUR_C_THROW("Unknown type of action for Bele3Line");
      break;
  }  // end of switch(act)

  return 0;

}  // Discret::Elements::Bele3Line::Evaluate



/*----------------------------------------------------------------------*
 |  Integrate a Line Neumann boundary condition (public)     gammi 04/07|
 *----------------------------------------------------------------------*/
int Discret::Elements::Bele3Line::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  // there are 2 velocities and 1 pressure
  const int numdf = 3;

  const double thsl = params.get("thsl", 0.0);

  // find out whether we will use a time curve
  const double time = params.get("total time", -1.0);

  // get values and switches from the condition
  // (assumed to be constant on element boundary)
  const auto& onoff = condition.parameters().get<std::vector<int>>("ONOFF");
  const auto& val = condition.parameters().get<std::vector<double>>("VAL");
  const auto& functions = condition.parameters().get<std::vector<Core::IO::Noneable<int>>>("FUNCT");

  // set number of nodes
  const size_t iel = this->num_node();

  const Core::FE::CellType distype = this->shape();

  // gaussian points
  const Core::FE::GaussRule1D gaussrule = get_optimal_gaussrule(distype);
  const Core::FE::IntegrationPoints1D intpoints(gaussrule);


  // allocate vector for shape functions and for derivatives
  Core::LinAlg::SerialDenseVector funct(iel);
  Core::LinAlg::SerialDenseMatrix deriv(1, iel);

  // node coordinates
  Core::LinAlg::SerialDenseMatrix xye(2, iel);

  // get node coordinates
  for (size_t i = 0; i < iel; ++i)
  {
    xye(0, i) = this->nodes()[i]->x()[0];
    xye(1, i) = this->nodes()[i]->x()[1];
  }


  // loop over integration points
  for (int gpid = 0; gpid < intpoints.nquad; gpid++)
  {
    const double e1 = intpoints.qxg[gpid][0];
    // get shape functions and derivatives in the line
    Core::FE::shape_function_1d(funct, e1, distype);
    Core::FE::shape_function_1d_deriv1(deriv, e1, distype);

    // compute infinitesimal line element dr for integration along the line
    const double dr = f2_substitution(xye, deriv, iel);

    // values are multiplied by the product from inf. area element,
    // the gauss weight, the timecurve factor and the constant
    // belonging to the time integration algorithm (theta*dt for
    // one step theta, 2/3 for bdf with dt const.)

    const double fac = intpoints.qwgt[gpid] * dr * thsl;

    // factor given by spatial function
    double functionfac = 1.0;
    // determine coordinates of current Gauss point
    double coordgp[3];
    coordgp[0] = 0.0;
    coordgp[1] = 0.0;
    coordgp[2] = 0.0;
    for (size_t i = 0; i < iel; i++)
    {
      coordgp[0] += xye(0, i) * funct[i];
      coordgp[1] += xye(1, i) * funct[i];
    }

    const double* coordgpref = coordgp;  // needed for function evaluation

    for (size_t node = 0; node < iel; ++node)
    {
      for (size_t dim = 0; dim < 3; dim++)
      {
        // evaluate function at current gauss point if it is not none
        if (functions[dim].has_value() && functions[dim].value() > 0)
        {
          functionfac =
              Global::Problem::instance()
                  ->function_by_id<Core::Utils::FunctionOfSpaceTime>(functions[dim].value() - 1)
                  .evaluate(coordgpref, time, dim);
        }
        else
          functionfac = 1.0;

        elevec1[node * numdf + dim] += funct[node] * onoff[dim] * val[dim] * fac * functionfac;
      }
    }
  }  // end of loop over integrationen points


  // FOUR_C_THROW("Line Neumann condition not yet implemented for Bele3");
  return 0;
}

Core::FE::GaussRule1D Discret::Elements::Bele3Line::get_optimal_gaussrule(
    const Core::FE::CellType& distype)
{
  Core::FE::GaussRule1D rule = Core::FE::GaussRule1D::undefined;
  switch (distype)
  {
    case Core::FE::CellType::line2:
      rule = Core::FE::GaussRule1D::line_2point;
      break;
    case Core::FE::CellType::line3:
      rule = Core::FE::GaussRule1D::line_3point;
      break;
    default:
      FOUR_C_THROW("unknown number of nodes for gaussrule initialization");
      break;
  }
  return rule;
}


double Discret::Elements::Bele3Line::f2_substitution(const Core::LinAlg::SerialDenseMatrix xye,
    const Core::LinAlg::SerialDenseMatrix deriv, const int iel)
{
  // compute derivative of parametrization
  double dr = 0.0;
  Core::LinAlg::SerialDenseVector der_par(iel);
  Core::LinAlg::multiply_nt(der_par, xye, deriv);
  dr = Core::LinAlg::norm2(der_par);
  return dr;
}

/*----------------------------------------------------------------------*
 |  Integrate shapefunctions over line (public)              g.bau 07/07|
 *----------------------------------------------------------------------*/
void Discret::Elements::Bele3Line::integrate_shape_function(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, const std::vector<int>& lm,
    Core::LinAlg::SerialDenseVector& elevec1, const std::vector<double>& edispnp)
{
  // there are 2 velocities and 1 pressure
  const int numdf = 3;

  //  const double thsl = params.get("thsl",1.0);

  /*
    // find out whether we will use a time curve
    bool usetime = true;
    const double time = params.get("total time",-1.0);
    if (time<0.0) usetime = false;
  */

  // set number of nodes
  const size_t iel = this->num_node();

  // gaussian points
  const Core::FE::CellType distype = this->shape();
  const Core::FE::GaussRule1D gaussrule = get_optimal_gaussrule(distype);
  const Core::FE::IntegrationPoints1D intpoints(gaussrule);

  // allocate vector for shape functions and for derivatives
  Core::LinAlg::SerialDenseVector funct(iel);
  Core::LinAlg::SerialDenseMatrix deriv(1, iel);

  // node coordinates
  Core::LinAlg::SerialDenseMatrix xye(2, iel);

  // get node coordinates
  for (size_t i = 0; i < iel; ++i)
  {
    xye(0, i) = this->nodes()[i]->x()[0];
    xye(1, i) = this->nodes()[i]->x()[1];
  }

  //  if (parent_->IsMoving())
  {
    FOUR_C_ASSERT(edispnp.size() != 0, "paranoid");

    for (size_t i = 0; i < iel; i++)
    {
      xye(0, i) += edispnp[3 * i];
      xye(1, i) += edispnp[3 * i + 1];
    }
  }

  // loop over integration points
  for (int gpid = 0; gpid < intpoints.nquad; gpid++)
  {
    const double e1 = intpoints.qxg[gpid][0];
    // get shape functions and derivatives in the line
    Core::FE::shape_function_1d(funct, e1, distype);
    Core::FE::shape_function_1d_deriv1(deriv, e1, distype);

    // compute infinitesimal line element dr for integration along the line
    const double dr = f2_substitution(xye, deriv, iel);

    // values are multiplied by the product from inf. area element,
    // the gauss weight, the timecurve factor and the constant
    // belonging to the time integration algorithm (theta*dt for
    // one step theta, 2/3 for bdf with dt const.)

    // double fac = intpoints.qwgt[gpid] *dr * thsl;
    const double fac = intpoints.qwgt[gpid] * dr;

    for (size_t node = 0; node < iel; ++node)
    {
      for (size_t dim = 0; dim < 3; dim++)
      {
        elevec1[node * numdf + dim] += funct[node] * fac;
      }
    }
  }  // end of loop over integrationen points

  return;
}  // Discret::Elements::Bele3Line::integrate_shape_function

FOUR_C_NAMESPACE_CLOSE
