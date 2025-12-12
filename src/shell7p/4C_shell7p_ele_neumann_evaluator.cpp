// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_shell7p_ele_neumann_evaluator.hpp"

#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_shell7p_ele_calc_lib.hpp"
#include "4C_utils_function.hpp"

FOUR_C_NAMESPACE_OPEN

void Discret::Elements::Shell::evaluate_neumann_by_element(Core::Elements::Element& ele,
    const Core::FE::Discretization& discretization, const Core::Conditions::Condition& condition,
    const std::vector<int>& dof_index_array, Core::LinAlg::SerialDenseVector& element_force_vector,
    Core::LinAlg::SerialDenseMatrix* element_stiffness_matrix, double total_time)
{
  switch (ele.shape())
  {
    case Core::FE::CellType::quad4:
      return evaluate_neumann<Core::FE::CellType::quad4>(ele, discretization, condition,
          dof_index_array, element_force_vector, element_stiffness_matrix, total_time);
    case Core::FE::CellType::quad8:
      return evaluate_neumann<Core::FE::CellType::quad8>(ele, discretization, condition,
          dof_index_array, element_force_vector, element_stiffness_matrix, total_time);
    case Core::FE::CellType::quad9:
      return evaluate_neumann<Core::FE::CellType::quad9>(ele, discretization, condition,
          dof_index_array, element_force_vector, element_stiffness_matrix, total_time);
    case Core::FE::CellType::tri3:
      return evaluate_neumann<Core::FE::CellType::tri3>(ele, discretization, condition,
          dof_index_array, element_force_vector, element_stiffness_matrix, total_time);
    case Core::FE::CellType::tri6:
      return evaluate_neumann<Core::FE::CellType::tri6>(ele, discretization, condition,
          dof_index_array, element_force_vector, element_stiffness_matrix, total_time);
    default:
      FOUR_C_THROW(
          "The discretization type you are trying to evaluate the Neumann condition for is not yet "
          "implemented in evaluate_neumann_by_element.");
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::Shell::evaluate_neumann(Core::Elements::Element& ele,
    const Core::FE::Discretization& discretization, const Core::Conditions::Condition& condition,
    const std::vector<int>& dof_index_array, Core::LinAlg::SerialDenseVector& element_force_vector,
    Core::LinAlg::SerialDenseMatrix* element_stiffness_matrix, double total_time)
{
  constexpr auto num_dim = Shell::Internal::num_dim;
  constexpr auto numnod = Shell::Internal::num_node<distype>;
  constexpr auto noddof = Shell::Internal::node_dof;

  Core::FE::IntegrationPoints2D intpoints =
      create_gauss_integration_points<distype>(get_gauss_rule<distype>());

  // IMPORTANT: The 'neum_orthopressure' case represents a truly nonlinear follower-load
  // acting on the spatial configuration. Therefore, it needs to be linearized. On the
  // contrary, the simplified 'neum_pseudo_orthopressure' option allows for an approximative
  // modeling of an orthopressure load without the need to do any linearization. However,
  // this can only be achieved by referring the 'neum_pseudo_orthopressure' load to the last
  // converged configuration, which introduces an error as compared with 'neum_orthopressure'.
  bool loadlin = (element_stiffness_matrix != nullptr);

  // get type of condition
  enum LoadType
  {
    neum_none,
    neum_live,
    neum_orthopressure,
    neum_pseudo_orthopressure,
  };
  LoadType ltype = neum_none;

  // type of configurations
  enum Configuration
  {
    config_none,
    config_material,       // material configuration
    config_lastconverged,  // last converged configuration
    config_spatial         // spatial configuration
  };

  Configuration config = config_none;

  const std::string& type = condition.parameters().get<std::string>("TYPE");
  if (type == "Live")
  {
    ltype = neum_live;
    config = config_material;
  }
  else if (type == "pseudo_orthopressure")
  {
    ltype = neum_pseudo_orthopressure;
    config = config_lastconverged;
  }
  else if (type == "orthopressure")
  {
    ltype = neum_orthopressure;
    config = config_spatial;
  }
  else
  {
    FOUR_C_THROW("Unknown type of SurfaceNeumann condition");
  }
  // get values and switches from the condition
  const auto& onoff = condition.parameters().get<std::vector<int>>("ONOFF");
  const auto& value = condition.parameters().get<std::vector<double>>("VAL");

  // ensure that at least as many curves/functs as dofs are available
  if (onoff.size() < Internal::node_dof)
    FOUR_C_THROW(
        "Fewer functions or curves defined than the element's nodal degree of freedoms (6).");

  for (std::size_t checkdof = num_dim; checkdof < onoff.size(); ++checkdof)
  {
    if (onoff[checkdof] != 0)
    {
      FOUR_C_THROW(
          "You have activated more than {} dofs in your Neumann boundary condition. This is higher "
          "than the dimension of the element.",
          num_dim);
    }
  }
  // get ids of functions of space and time
  const auto function_ids = condition.parameters().get<std::vector<std::optional<int>>>("FUNCT");

  // integration loops
  std::array<double, 2> xi_gp;
  for (int gp = 0; gp < intpoints.num_points(); ++gp)
  {
    // get gauss points from integration rule
    xi_gp[0] = intpoints.qxg[gp][0];
    xi_gp[1] = intpoints.qxg[gp][1];
    // get gauss weight at current gp
    double gpweights = intpoints.qwgt[gp];
    // get shape functions and derivatives at gaussian points
    ShapefunctionsAndDerivatives<distype> shapefunctions =
        evaluate_shapefunctions_and_derivs<distype>(xi_gp);
    Core::LinAlg::Matrix<num_dim, num_dim> g_metrics_covariant(Core::LinAlg::Initialization::zero);

    Core::LinAlg::Matrix<numnod, num_dim> x_refe(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<numnod, num_dim> x_curr(Core::LinAlg::Initialization::zero);

    for (int i = 0; i < numnod; ++i)
    {
      x_refe(i, 0) = ele.nodes()[i]->x()[0];
      x_refe(i, 1) = ele.nodes()[i]->x()[1];
      x_refe(i, 2) = ele.nodes()[i]->x()[2];
    }

    switch (config)
    {
      case config_material:
      {
        // no linearization needed for load in material configuration
        loadlin = false;
        g_metrics_covariant.multiply_nn(1.0, shapefunctions.derivatives_, x_refe, 0.0);
      }
      break;
      case config_lastconverged:
      {
        // no linearization needed for load in last converged configuration
        loadlin = false;
        const Core::LinAlg::Vector<double>& disp = *discretization.get_state("displacement");
        std::vector<double> displacements = Core::FE::extract_values(disp, dof_index_array);

        spatial_configuration<distype>(x_curr, x_refe, displacements, 0);

        g_metrics_covariant.multiply_nn(1.0, shapefunctions.derivatives_, x_curr, 0.0);
      }
      break;
      case config_spatial:
      {
        const Core::LinAlg::Vector<double>& disp = *discretization.get_state("displacement new");
        std::vector<double> displacements = Core::FE::extract_values(disp, dof_index_array);

        spatial_configuration<distype>(x_curr, x_refe, displacements, 0);

        g_metrics_covariant.multiply_nn(1.0, shapefunctions.derivatives_, x_curr, 0.0);
      }
      break;
      default:
        FOUR_C_THROW("Unknown type of configuration");
    }

    // get thickness direction derivative perpendicular to g1 and g2 (with area as length)
    // -> g3 = (g1 x g2) / (|g1 x g2 |)
    Core::LinAlg::Matrix<num_dim, 1> g3(Core::LinAlg::Initialization::zero);
    g3(0) = g_metrics_covariant(0, 1) * g_metrics_covariant(1, 2) -
            g_metrics_covariant(0, 2) * g_metrics_covariant(1, 1);
    g3(1) = g_metrics_covariant(0, 2) * g_metrics_covariant(1, 0) -
            g_metrics_covariant(0, 0) * g_metrics_covariant(1, 2);
    g3(2) = g_metrics_covariant(0, 0) * g_metrics_covariant(1, 1) -
            g_metrics_covariant(0, 1) * g_metrics_covariant(1, 0);
    // compute line increment ds
    double ds = g3.norm2();
    if (ds <= 1.0e-14) FOUR_C_THROW("Element Area equal 0.0 or negative detected");

    // material/reference coordinates of Gaussian point
    Core::LinAlg::Matrix<num_dim, 1> gauss_point_reference_coordinates;
    gauss_point_reference_coordinates.multiply_tn(x_refe, shapefunctions.shapefunctions_);
    // current coordinates of Gaussian point
    Core::LinAlg::Matrix<num_dim, 1> gauss_point_current_coordinates;
    if (ltype != neum_live)
      gauss_point_current_coordinates.multiply_tn(x_curr, shapefunctions.shapefunctions_);

    // evaluate the spatial function in the reference state
    std::vector<double> function_scale_factors(3, 1.0);
    for (auto dim = 0; dim < num_dim; dim++)
    {
      if (onoff[dim])
      {
        // function evaluation
        if (function_ids[dim].has_value() && function_ids[dim].value() > 0)
        {
          function_scale_factors[dim] =
              Global::Problem::instance()
                  ->function_by_id<Core::Utils::FunctionOfSpaceTime>(function_ids[dim].value())
                  .evaluate(gauss_point_reference_coordinates.as_span(), total_time, dim);
        }
      }
    }

    double value_times_integration_factor = 0.0;
    for (int dim = 0; dim < Shell::Internal::num_dim; ++dim)
    {
      switch (ltype)
      {
        case neum_live:  // uniform load on reference configuration
        {
          value_times_integration_factor =
              value[dim] * function_scale_factors[dim] * gpweights * ds;
        }
        break;
        // pressure orthogonal to surface
        case neum_pseudo_orthopressure:
        case neum_orthopressure:
        {
          if (onoff[2] != 1) FOUR_C_THROW("orthopressure must be on third dof");
          value_times_integration_factor =
              value[2] * function_scale_factors[2] * gpweights * g3(dim, 0);
        }
        break;
        default:
          FOUR_C_THROW("Unknown type of SurfaceNeumann load");
      }  // end switch(eletype)

      for (auto nodeid = 0; nodeid < Shell::Internal::num_node<distype>; ++nodeid)
      {
        // Evaluates the Neumann boundary condition: f_{x,y,z}^i=\sum_j N^i(xi^j) * value(t) *
        // integration_factor_j
        // assembles the element force vector [f_x^1, f_y^1, f_z^1, ..., f_x^n, f_y^n, f_z^n]
        element_force_vector[nodeid * noddof + dim] +=
            shapefunctions.shapefunctions_(nodeid) * value_times_integration_factor;
      }
    }
    // load linerization (if necessary)
    if (loadlin)
    {
      constexpr auto numdof = Internal::numdofperelement<distype>;
      Core::LinAlg::Matrix<num_dim, numdof> dnormal;
      Core::LinAlg::Matrix<num_dim, numdof> dg1;
      Core::LinAlg::Matrix<num_dim, numdof> dg2;

      // linearization of basis vectors
      for (int nodeid = 0; nodeid < numnod; ++nodeid)
      {
        for (int dim = 0; dim < num_dim; ++dim)
        {
          dg1(dim, nodeid * noddof + dim) = shapefunctions.derivatives_(0, nodeid);
          dg2(dim, nodeid * noddof + dim) = shapefunctions.derivatives_(1, nodeid);
        }
      }
      // linearization of local surface normal vector
      for (int dof = 0; dof < numdof; ++dof)
      {
        dnormal(0, dof) =
            dg1(1, dof) * g_metrics_covariant(1, 2) + g_metrics_covariant(0, 1) * dg2(2, dof) -
            dg1(2, dof) * g_metrics_covariant(1, 1) - g_metrics_covariant(0, 2) * dg2(1, dof);
        dnormal(1, dof) =
            dg1(2, dof) * g_metrics_covariant(1, 0) + g_metrics_covariant(0, 2) * dg2(0, dof) -
            dg1(0, dof) * g_metrics_covariant(1, 2) - g_metrics_covariant(0, 0) * dg2(2, dof);
        dnormal(2, dof) =
            dg1(0, dof) * g_metrics_covariant(1, 1) + g_metrics_covariant(0, 0) * dg2(1, dof) -
            dg1(1, dof) * g_metrics_covariant(1, 0) - g_metrics_covariant(0, 1) * dg2(0, dof);
      }
      // build surface element load linearization matrix
      // (CAREFUL: Minus sign due to the fact that external forces enter the global
      // residual vector with a minus sign, too! However, the load linaerization is
      // simply added to the global tangent stiffness matrix, thus we explicitly
      // need to set the minus sign here.)
      for (int nodeid = 0; nodeid < numnod; ++nodeid)
        for (int dim = 0; dim < num_dim; dim++)
          for (int dof = 0; dof < element_force_vector.numRows(); dof++)
            (*element_stiffness_matrix)(nodeid* noddof + dim, dof) -=
                shapefunctions.shapefunctions_(nodeid) * value[2] * function_scale_factors[2] *
                gpweights * dnormal(dim, dof);
    }
  }
}
FOUR_C_NAMESPACE_CLOSE
