// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_shell7p_ele_scatra_preevaluator.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_shell7p_ele_calc_lib.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN


void Discret::Elements::Shell::pre_evaluate_scatra_by_element(Core::Elements::Element& ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& dof_index_array)
{
  switch (ele.shape())
  {
    case Core::FE::CellType::quad4:
      return pre_evaluate_scatra<Core::FE::CellType::quad4>(
          ele, params, discretization, dof_index_array);
    case Core::FE::CellType::quad8:
      return pre_evaluate_scatra<Core::FE::CellType::quad8>(
          ele, params, discretization, dof_index_array);
    case Core::FE::CellType::quad9:
      return pre_evaluate_scatra<Core::FE::CellType::quad9>(
          ele, params, discretization, dof_index_array);
    case Core::FE::CellType::tri3:
      return pre_evaluate_scatra<Core::FE::CellType::tri3>(
          ele, params, discretization, dof_index_array);
    case Core::FE::CellType::tri6:
      return pre_evaluate_scatra<Core::FE::CellType::tri6>(
          ele, params, discretization, dof_index_array);
    default:
      FOUR_C_THROW(
          "The discretization type you are trying to pre-evaluate for shell7p scatra is not yet "
          "implemented.");
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::Shell::pre_evaluate_scatra(Core::Elements::Element& ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& dof_index_array)
{
  Core::FE::IntegrationPoints2D intpoints_midsurface_ =
      create_gauss_integration_points<distype>(get_gauss_rule<distype>());

  if (dof_index_array.size() > 1)
  {
    // ask for the number of dofs of second dofset (scatra)
    const int numscal = discretization.num_dof(1, ele.nodes()[0]);

    if (dof_index_array[1].size() != Shell::Internal::num_node<distype> * numscal)
      FOUR_C_THROW("location vector length does not match!");

    // name of scalarfield
    std::string scalarfield = "scalarfield";

    if (discretization.has_state(1, scalarfield))
    {
      // get the scalar state
      std::shared_ptr<const Core::LinAlg::Vector<double>> scalarnp =
          discretization.get_state(1, scalarfield);

      if (scalarnp == nullptr) FOUR_C_THROW("can not get state vector {}", scalarfield);

      // extract local values of the global vectors
      std::vector<double> myscalar = Core::FE::extract_values(*scalarnp, dof_index_array[1].lm_);

      // element vector for k-th scalar
      std::vector<Core::LinAlg::Matrix<Shell::Internal::num_node<distype>, 1>> elescalar(numscal);
      for (int k = 0; k < numscal; ++k)
      {
        for (int i = 0; i < Shell::Internal::num_node<distype>; ++i)
        {
          (elescalar.at(k))(i, 0) = myscalar.at(numscal * i + k);
        }
      }

      // create vector of gauss point values to be set in params list
      std::shared_ptr<std::vector<std::vector<double>>> gpscalar =
          std::make_shared<std::vector<std::vector<double>>>(
              intpoints_midsurface_.num_points(), std::vector<double>(numscal, 0.0));

      // allocate vector for shape functions and matrix for derivatives at gp
      Core::LinAlg::Matrix<Shell::Internal::num_node<distype>, 1> shapefunctions(
          Core::LinAlg::Initialization::zero);

      // loop over gauss points
      for (int gp = 0; gp < intpoints_midsurface_.num_points(); ++gp)
      {
        // get gauss points from integration rule
        double xi_gp = intpoints_midsurface_.qxg[gp][0];
        double eta_gp = intpoints_midsurface_.qxg[gp][1];

        // get shape functions and derivatives in the plane of the element
        Core::FE::shape_function_2d(shapefunctions, xi_gp, eta_gp, distype);

        // scalar at current gp
        std::vector<double> scalar_curr_gp(numscal, 0.0);

        for (int k = 0; k < numscal; ++k)
        {
          // identical shapefunctions for displacements and scalar fields
          scalar_curr_gp.at(k) = shapefunctions.dot(elescalar.at(k));
        }

        gpscalar->at(gp) = scalar_curr_gp;
      }

      // set scalar states at gp to params list
      params.set<std::shared_ptr<std::vector<std::vector<double>>>>("gp_conc", gpscalar);
    }
  }
  Core::LinAlg::Tensor<double, 2> center{};
  params.set("elecenter_coords_ref", center);
}

FOUR_C_NAMESPACE_CLOSE
