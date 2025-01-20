// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_elements_paramsinterface.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_fem_nurbs_discretization_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_mat_shell_kl.hpp"
#include "4C_shell_kl_nurbs.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function.hpp"

#include <array>

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
int Discret::Elements::KirchhoffLoveShellNurbs::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  set_params_interface_ptr(params);
  Core::Elements::ActionType act = Core::Elements::none;
  if (is_params_interface())
  {
    act = interface_ptr_->get_action_type();
  }
  else
  {
    FOUR_C_THROW("KirchhoffLoveShellNurbs are only implemented for the new solid time integration");
  }

  switch (act)
  {
    // Calculate the residuum and/or tangent stiffness matrix
    case Core::Elements::struct_calc_internalforce:
    case Core::Elements::struct_calc_nlnstiff:
    {
      // Get NURBS stuff
      std::vector<Core::LinAlg::SerialDenseVector> myknots;
      Core::LinAlg::Matrix<9, 1> weights(true);
      const bool zero_size =
          Core::FE::Nurbs::get_my_nurbs_knots_and_weights(discretization, this, myknots, weights);
      if (zero_size) return 0;

      // Get current displacement
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == nullptr) FOUR_C_THROW("Cannot get state vectors 'displacement'");
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);
      Core::LinAlg::Matrix<9 * 3, 1> displacement(mydisp.data(), true);

      // Get reference configuration
      Core::LinAlg::Matrix<9, 3, double> XI(true);
      for (unsigned int i_node = 0; i_node < 9; i_node++)
      {
        for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
        {
          XI(i_node, i_dim) = nodes()[i_node]->x()[i_dim];
        }
      }

      // Get material
      const auto shell_material =
          std::dynamic_pointer_cast<const Mat::KirchhoffLoveShell>(material());

      // Get integration points
      const std::array<Core::FE::IntegrationPoints1D, 2> integration_points = {
          Core::FE::IntegrationPoints1D(gaussrule_[0]),
          Core::FE::IntegrationPoints1D(gaussrule_[1])};

      if (act == Core::Elements::struct_calc_internalforce)
      {
        // Evaluate the residuum
        evaluate_residuum_auto_generated(shell_material->young_modulus(),
            shell_material->poisson_ratio(), shell_material->thickness(), integration_points[0],
            integration_points[1], myknots, weights, XI, displacement, elevec1);
      }
      else if (act == Core::Elements::struct_calc_nlnstiff)
      {
        // Evaluate the residuum and the tangent stiffness matrix
        evaluate_residuum_and_jacobian_auto_generated(shell_material->young_modulus(),
            shell_material->poisson_ratio(), shell_material->thickness(), integration_points[0],
            integration_points[1], myknots, weights, XI, displacement, elevec1, elemat1);
      }
      else
        FOUR_C_THROW("Got unexpected action type");

      break;
    }
    case Core::Elements::struct_calc_predict:
    case Core::Elements::struct_calc_recover:
    case Core::Elements::struct_calc_update_istep:
      break;
    default:
    {
      FOUR_C_THROW("Unknown type of action for KirchhoffLoveShellNurbs element: %s",
          action_type_to_string(act).c_str());
      break;
    }
  }
  return 0;
}

/**
 *
 */
int Discret::Elements::KirchhoffLoveShellNurbs::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  constexpr int n_nodal_dof = 3;

  set_params_interface_ptr(params);
  double time = -1.0;
  if (is_params_interface())
  {
    time = interface_ptr_->get_total_time();
  }
  else
  {
    FOUR_C_THROW("KirchhoffLoveShellNurbs are only implemented for the new solid time integration");
  }

  // get values and switches from the condition
  const auto& onoff = condition.parameters().get<std::vector<int>>("ONOFF");
  const auto& val = condition.parameters().get<std::vector<double>>("VAL");
  const auto& funct_id = condition.parameters().get<std::vector<Core::IO::Noneable<int>>>("FUNCT");

  // ensure that the boundary condition has the correct size
  if (onoff.size() != n_nodal_dof)
    FOUR_C_THROW("Wrong number of BC values. Expected %d, got %d.", n_nodal_dof, onoff.size());

  // Get the functions from the global problem
  std::array<const Core::Utils::FunctionOfSpaceTime*, n_nodal_dof> functions;
  for (int i_dof = 0; i_dof < n_nodal_dof; ++i_dof)
  {
    if (onoff[i_dof] == 1 and funct_id[i_dof].has_value() and funct_id[i_dof].value() > 0)
    {
      functions[i_dof] =
          &(Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfSpaceTime>(
              funct_id[i_dof].value() - 1));
    }
  }

  // Lambda to allow the evaluation of the body load
  auto evaluate_body_load = [&](const double* reference_position) -> Core::LinAlg::Matrix<3, 1>
  {
    Core::LinAlg::Matrix<3, 1> force(true);

    for (int i_dof = 0; i_dof < n_nodal_dof; ++i_dof)
    {
      if (onoff[i_dof] == 1 and funct_id[i_dof] > 0)
      {
        force(i_dof) = val[i_dof] * functions[i_dof]->evaluate(reference_position, time, i_dof);
      }
      else if (onoff[i_dof] == 1)
      {
        force(i_dof) = val[i_dof];
      }
      else
      {
        // No load in this direction
      }
    }

    return force;
  };

  // Get NURBS stuff
  const auto* nurbsdis = dynamic_cast<Core::FE::Nurbs::NurbsDiscretization*>(&(discretization));
  std::vector<Core::LinAlg::SerialDenseVector> myknots(2);
  bool zero_sized = (*((*nurbsdis).get_knot_vector())).get_ele_knots(myknots, id());
  // skip zero sized elements in knot span, as they correspond to interpolated nodes
  if (zero_sized) return (0);
  Core::LinAlg::Matrix<9, 1> weights(true);
  for (int i_node = 0; i_node < 9; ++i_node)
  {
    const auto* cp = dynamic_cast<Core::FE::Nurbs::ControlPoint*>(nodes()[i_node]);
    weights(i_node) = cp->w();
  }

  // Get nodal reference configuration
  Core::LinAlg::Matrix<9, 3, double> XI(true);
  for (unsigned int i_node = 0; i_node < 9; i_node++)
  {
    for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
    {
      XI(i_node, i_dim) = nodes()[i_node]->x()[i_dim];
    }
  }

  // Get integration points, we use the same integration rule (6GP) in both parameter directions
  const Core::FE::IntegrationPoints1D integration_points = Core::FE::IntegrationPoints1D(
      Core::FE::num_gauss_points_to_gauss_rule<Core::FE::CellType::line2>(6));

  // Evaluate the residuum and the tangent stiffness matrix
  evaluate_body_load_auto_generated(
      integration_points, myknots, weights, XI, evaluate_body_load, elevec1);

  // finished
  return 0;
}

FOUR_C_NAMESPACE_CLOSE
