// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_elements_paramsinterface.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_so3_line.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function.hpp"

FOUR_C_NAMESPACE_OPEN


/*-----------------------------------------------------------------------*
 * Integrate a Line Neumann boundary condition (public)         gee 04/08|
 * ----------------------------------------------------------------------*/
int Discret::Elements::StructuralLine::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  // set the interface ptr in the parent element
  parent_element()->set_params_interface_ptr(params);
  // get type of condition
  enum LoadType
  {
    neum_none,
    neum_live
  };
  LoadType ltype;

  // spatial or material configuration depends on the type of load
  // currently only material frame used
  // enum Configuration
  //{
  //  config_none,
  //  config_material,
  //  config_spatial,
  //  config_both
  //};
  // Configuration config = config_none;

  const auto& type = condition.parameters().get<std::string>("TYPE");
  if (type == "neum_live")
  {
    ltype = neum_live;
    // config = config_material;
  }
  else
    FOUR_C_THROW("Unknown type of LineNeumann condition");

  // get values and switches from the condition
  const auto onoff = condition.parameters().get<std::vector<int>>("ONOFF");
  const auto val = condition.parameters().get<std::vector<double>>("VAL");
  const auto spa_func = condition.parameters().get<std::vector<Core::IO::Noneable<int>>>("FUNCT");

  /*
  **    TIME CURVE BUSINESS
  */
  // find out whether we will use a time curve
  double time = -1.0;
  if (parent_element()->is_params_interface())
    time = parent_element()->params_interface_ptr()->get_total_time();
  else
    time = params.get("total time", -1.0);

  const int numdim = 3;

  // ensure that at least as many curves/functs as dofs are available
  if (int(onoff.size()) < numdim)
    FOUR_C_THROW("Fewer functions or curves defined than the element has dofs.");

  for (int checkdof = numdim; checkdof < int(onoff.size()); ++checkdof)
  {
    if (onoff[checkdof] != 0)
      FOUR_C_THROW(
          "Number of Dimensions in Neumann_Evaluation is 3. Further DoFs are not considered.");
  }

  // element geometry update - currently only material configuration
  const int numnode = num_node();
  Core::LinAlg::SerialDenseMatrix x(numnode, numdim);
  material_configuration(x);

  // integration parameters
  const Core::FE::IntegrationPoints1D intpoints(gaussrule_);
  Core::LinAlg::SerialDenseVector shapefcts(numnode);
  Core::LinAlg::SerialDenseMatrix deriv(1, numnode);
  const Core::FE::CellType shape = StructuralLine::shape();

  // integration
  for (int gp = 0; gp < intpoints.nquad; ++gp)
  {
    // get shape functions and derivatives of element surface
    const double e = intpoints.qxg[gp][0];
    Core::FE::shape_function_1d(shapefcts, e, shape);
    Core::FE::shape_function_1d_deriv1(deriv, e, shape);
    switch (ltype)
    {
      case neum_live:
      {  // uniform load on reference configuration

        double dL;
        line_integration(dL, x, deriv);

        // loop the dofs of a node
        for (int i = 0; i < numdim; ++i)
        {
          if (onoff[i])  // is this dof activated?
          {
            // factor given by spatial function
            double functfac = 1.0;

            if (spa_func[i].has_value() && spa_func[i].value() > 0)
            {
              // calculate reference position of GP
              Core::LinAlg::SerialDenseMatrix gp_coord(1, numdim);
              Core::LinAlg::multiply_tn(gp_coord, shapefcts, x);

              // write coordinates in another datatype
              double gp_coord2[numdim];
              for (int k = 0; k < numdim; k++) gp_coord2[k] = gp_coord(0, k);
              const double* coordgpref = gp_coord2;  // needed for function evaluation

              // evaluate function at current gauss point
              functfac =
                  Global::Problem::instance()
                      ->function_by_id<Core::Utils::FunctionOfSpaceTime>(spa_func[i].value() - 1)
                      .evaluate(coordgpref, time, i);
            }

            const double fac = val[i] * intpoints.qwgt[gp] * dL * functfac;
            for (int node = 0; node < numnode; ++node)
            {
              elevec1[node * numdim + i] += shapefcts[node] * fac;
            }
          }
        }

        break;
      }

      default:
        FOUR_C_THROW("Unknown type of LineNeumann load");
        break;
    }
  }

  return 0;
}

/*----------------------------------------------------------------------*
 *  (private)                                                  gee 04/08|
 * ---------------------------------------------------------------------*/
void Discret::Elements::StructuralLine::line_integration(double& dL,
    const Core::LinAlg::SerialDenseMatrix& x, const Core::LinAlg::SerialDenseMatrix& deriv)
{
  // compute dXYZ / drs
  Core::LinAlg::SerialDenseMatrix dxyzdrs(1, 3);
  Core::LinAlg::multiply(dxyzdrs, deriv, x);
  dL = 0.0;
  for (int i = 0; i < 3; ++i) dL += dxyzdrs(0, i) * dxyzdrs(0, i);
  dL = sqrt(dL);
  return;
}

FOUR_C_NAMESPACE_CLOSE
