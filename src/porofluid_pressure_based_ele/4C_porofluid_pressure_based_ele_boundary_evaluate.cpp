// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_porofluid_pressure_based_ele.hpp"
#include "4C_porofluid_pressure_based_ele_action.hpp"
#include "4C_porofluid_pressure_based_ele_boundary_factory.hpp"
#include "4C_porofluid_pressure_based_ele_interface.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  evaluate the element (public)                             vuong 08/16 |
 *----------------------------------------------------------------------*/
int Discret::Elements::PoroFluidMultiPhaseBoundary::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  FOUR_C_THROW("not implemented. Use the evaluate() method with Location Array instead!");
  return -1;
}

/*----------------------------------------------------------------------*
 |  evaluate the element (public)                             vuong 08/16 |
 *----------------------------------------------------------------------*/
int Discret::Elements::PoroFluidMultiPhaseBoundary::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  // we assume here, that numdofpernode is equal for every node within
  // the element and does not change during the computations
  const int numdofpernode = num_dof_per_node(*(nodes()[0]));

  // copy pointers to matrices and vectors into std::vector
  std::vector<Core::LinAlg::SerialDenseMatrix*> elemat(2);
  elemat[0] = &elemat1;
  elemat[1] = &elemat2;
  std::vector<Core::LinAlg::SerialDenseVector*> elevec(3);
  elevec[0] = &elevec1;
  elevec[1] = &elevec2;
  elevec[2] = &elevec3;

  // all physics-related stuff is included in the implementation class that can
  // be used in principle inside any element (at the moment: only Transport
  // boundary element)
  // If this element has special features/ methods that do not fit in the
  // generalized implementation class, you have to do a switch here in order to
  // call element-specific routines
  return Discret::Elements::PoroFluidMultiPhaseBoundaryFactory::provide_impl(
      this, numdofpernode, discretization.name())
      ->evaluate(this, params, discretization, la, elemat, elevec);
}


/*----------------------------------------------------------------------*
 | evaluate Neumann boundary condition on boundary element   vuong 08/16 |
 *----------------------------------------------------------------------*/
int Discret::Elements::PoroFluidMultiPhaseBoundary::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, const Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  // add Neumann boundary condition to parameter list
  params.set<const Core::Conditions::Condition*>("condition", &condition);

  // build the location array
  Core::Elements::LocationArray la(discretization.num_dof_sets());
  Core::Elements::Element::location_vector(discretization, la);

  // evaluate boundary element
  return evaluate(params, discretization, la, *elemat1, *elemat1, elevec1, elevec1, elevec1);
}

FOUR_C_NAMESPACE_CLOSE
