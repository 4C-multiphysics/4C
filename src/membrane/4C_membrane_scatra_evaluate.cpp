// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_membrane_scatra.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  pre-evaluate the element (public)                                   |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::MembraneScatra<distype>::pre_evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la)
{
  if (la.size() > 1)
  {
    // ask for the number of dofs of second dofset (scatra)
    const int numscal = discretization.num_dof(1, nodes()[0]);

    if (la[1].size() != Membrane<distype>::numnod_ * numscal)
      FOUR_C_THROW("location vector length does not match!");

    // name of scalarfield
    std::string scalarfield = "scalarfield";

    if (discretization.has_state(1, scalarfield))
    {
      // get the scalar state
      std::shared_ptr<const Core::LinAlg::Vector<double>> scalarnp =
          discretization.get_state(1, scalarfield);

      if (scalarnp == nullptr) FOUR_C_THROW("can not get state vector {}", scalarfield.c_str());

      // extract local values of the global vectors
      std::vector<double> myscalar = Core::FE::extract_values(*scalarnp, la[1].lm_);

      // element vector for k-th scalar
      std::vector<Core::LinAlg::Matrix<Membrane<distype>::numnod_, 1>> elescalar(numscal);
      for (int k = 0; k < numscal; ++k)
      {
        for (int i = 0; i < Membrane<distype>::numnod_; ++i)
        {
          (elescalar.at(k))(i, 0) = myscalar.at(numscal * i + k);
        }
      }

      // number of gauss points
      const int numgp = (Membrane<distype>::intpoints_).nquad;

      // create vector of gauss point values to be set in params list
      std::shared_ptr<std::vector<std::vector<double>>> gpscalar =
          std::make_shared<std::vector<std::vector<double>>>(
              numgp, std::vector<double>(numscal, 0.0));

      // allocate vector for shape functions and matrix for derivatives at gp
      Core::LinAlg::Matrix<Membrane<distype>::numnod_, 1> shapefcts(
          Core::LinAlg::Initialization::zero);

      // loop over gauss points
      for (int gp = 0; gp < numgp; ++gp)
      {
        // get gauss points from integration rule
        Core::LinAlg::Matrix<2, 1> xi_eta;
        xi_eta(0) = (Membrane<distype>::intpoints_).qxg[gp][0];
        xi_eta(1) = (Membrane<distype>::intpoints_).qxg[gp][1];

        // get shape functions and derivatives in the plane of the element
        Core::FE::shape_function<distype>(xi_eta, shapefcts);

        // scalar at current gp
        std::vector<double> scalar_curr_gp(numscal, 0.0);

        for (int k = 0; k < numscal; ++k)
        {
          // identical shapefunctions for displacements and scalar fields
          scalar_curr_gp.at(k) = shapefcts.dot(elescalar.at(k));
        }

        gpscalar->at(gp) = scalar_curr_gp;
      }

      // set scalar states at gp to params list
      params.set<std::shared_ptr<std::vector<std::vector<double>>>>("gp_scalar", gpscalar);
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  evaluate the element (public)                                       |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::MembraneScatra<distype>::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra,
    Core::LinAlg::SerialDenseVector& elevec2_epetra,
    Core::LinAlg::SerialDenseVector& elevec3_epetra)
{
  // in some cases we need to write/change some data before evaluating
  pre_evaluate(params, discretization, la);

  Membrane<distype>::evaluate(params, discretization, la[0].lm_, elemat1_epetra, elemat2_epetra,
      elevec1_epetra, elevec2_epetra, elevec3_epetra);

  return 0;
}

template class Discret::Elements::MembraneScatra<Core::FE::CellType::tri3>;
template class Discret::Elements::MembraneScatra<Core::FE::CellType::tri6>;
template class Discret::Elements::MembraneScatra<Core::FE::CellType::quad4>;
template class Discret::Elements::MembraneScatra<Core::FE::CellType::quad9>;

FOUR_C_NAMESPACE_CLOSE
