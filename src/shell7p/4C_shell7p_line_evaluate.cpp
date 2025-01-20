// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_global_data.hpp"
#include "4C_shell7p_line.hpp"
#include "4C_utils_function.hpp"
#include "4C_utils_function_of_time.hpp"

FOUR_C_NAMESPACE_OPEN

int Discret::Elements::Shell7pLine::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Conditions::Condition& condition,
    std::vector<int>& dof_index_array, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  // set the interface ptr in the parent element
  parent_element()->set_params_interface_ptr(params);

  // we need the displacement at the previous step
  std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
      discretization.get_state("displacement");
  if (disp == nullptr) FOUR_C_THROW("Cannot get state vector 'displacement'");
  std::vector<double> displacements(dof_index_array.size());
  Core::FE::extract_my_values(*disp, displacements, dof_index_array);

  // get values and switches from the condition
  const auto onoff = condition.parameters().get<std::vector<int>>("ONOFF");
  const auto val = condition.parameters().get<std::vector<double>>("VAL");
  const auto spa_func = condition.parameters().get<std::vector<Core::IO::Noneable<int>>>("FUNCT");

  // time curve business
  // find out whether we will use a time curve
  double time = -1.0;
  if (parent_element()->is_params_interface())
    time = parent_element()->str_params_interface().get_total_time();
  else
    time = params.get<double>("total time", -1.0);

  // ensure that at least as many curves/functs as dofs are available
  if (int(onoff.size()) < node_dof_)
    FOUR_C_THROW(
        "Fewer functions or curves defined than the element's nodal degree of freedoms (6).");

  for (int checkdof = num_dim_; checkdof < int(onoff.size()); ++checkdof)
  {
    if (onoff[checkdof] != 0)
    {
      FOUR_C_THROW(
          "Number of Dimensions in Neumann_Evaluation is 3. Further DoFs are not considered.");
    }
  }

  // element geometry update - currently only material configuration
  const int numnode = num_node();
  Core::LinAlg::SerialDenseMatrix x(numnode, num_dim_);
  material_configuration(x);

  // integration parameters
  const Core::FE::IntegrationPoints1D intpoints(Core::FE::GaussRule1D::line_2point);
  Core::LinAlg::SerialDenseVector shape_functions(numnode);
  Core::LinAlg::SerialDenseMatrix derivatives(1, numnode);
  const Core::FE::CellType shape = Shell7pLine::shape();

  // integration
  for (int gp = 0; gp < intpoints.num_points(); ++gp)
  {
    // get shape functions and derivatives of element surface
    const double e = intpoints.qxg[gp][0];
    Core::FE::shape_function_1d(shape_functions, e, shape);
    Core::FE::shape_function_1d_deriv1(derivatives, e, shape);

    // covariant basis vectors and metric of shell body
    // g1,g2,g3 stored in Jacobian matrix  = (g1,g2,g3)
    double dL;
    line_integration(dL, x, derivatives);
    std::vector<double> a;
    // loop through the dofs of a node
    for (int i = 0; i < num_dim_; ++i)
    {
      // check if this dof is activated
      if (onoff[i])
      {
        // factor given by spatial function
        double functfac = 1.0;

        if (spa_func[i].has_value() && spa_func[i].value() > 0)
        {
          // calculate reference position of gaussian point
          Core::LinAlg::SerialDenseMatrix gp_coord(1, num_dim_);
          gp_coord.multiply(Teuchos::TRANS, Teuchos::NO_TRANS, 1.0, shape_functions, x, 0.0);

          // write coordinates in another datatype
          double gp_coord2[num_dim_];
          for (int k = 0; k < num_dim_; k++) gp_coord2[k] = gp_coord(0, k);
          const double* coordgpref = gp_coord2;  // needed for function evaluation

          // evaluate function at current gauss point
          functfac = Global::Problem::instance()
                         ->function_by_id<Core::Utils::FunctionOfSpaceTime>(spa_func[i].value() - 1)
                         .evaluate(coordgpref, time, i);
        }

        const double fac = val[i] * intpoints.qwgt[gp] * dL * functfac;

        for (int node = 0; node < numnode; ++node)
        {
          elevec1[node * node_dof_ + i] += shape_functions[node] * fac;
        }
      }
    }
  }

  return 0;
}


void Discret::Elements::Shell7pLine::line_integration(double& dL,
    const Core::LinAlg::SerialDenseMatrix& x, const Core::LinAlg::SerialDenseMatrix& derivatives)
{
  // compute dXYZ / drs
  Core::LinAlg::SerialDenseMatrix dxyzdrs(1, num_dim_);
  dxyzdrs.multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0, derivatives, x, 0.0);
  dL = 0.0;

  for (int i = 0; i < 3; ++i) dL += dxyzdrs(0, i) * dxyzdrs(0, i);

  dL = sqrt(dL);
}
FOUR_C_NAMESPACE_CLOSE
