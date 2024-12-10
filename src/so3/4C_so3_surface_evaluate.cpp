// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_boundary_integration.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_general_utils_nurbs_shapefunctions.hpp"
#include "4C_fem_nurbs_discretization.hpp"
#include "4C_fluid_ele_action.hpp"
#include "4C_global_data.hpp"
#include "4C_immersed_problem_immersed_base.hpp"
#include "4C_inpar_fsi.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_mat_structporo.hpp"
#include "4C_so3_prestress_service.hpp"
#include "4C_so3_surface.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function.hpp"
#include "4C_utils_function_of_time.hpp"

#include <Sacado.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 * Integrate a Surface Neumann boundary condition (public)     gee 04/08|
 * ---------------------------------------------------------------------*/
int Discret::Elements::StructuralSurface::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  // set the interface ptr in the parent element
  parent_element()->set_params_interface_ptr(params);

  // IMPORTANT: The 'neum_orthopressure' case represents a truly nonlinear follower-load
  // acting on the spatial configuration. Therefore, it needs to be linearized. On the
  // contrary, the simplified 'neum_pseudo_orthopressure' option allows for an approximative
  // modeling of an orthopressure load without the need to do any linearization. However,
  // this can only be achieved by referring the 'neum_pseudo_orthopressure' load to the last
  // converged configuration, which introduces an error as compared with 'neum_orthopressure'.
  bool loadlin = (elemat1 != nullptr);

  // type of Neumann conditions
  enum LoadType
  {
    neum_none,
    neum_live,                  // standard Neumann load
    neum_pseudo_orthopressure,  // pseudo-orthopressure load
    neum_orthopressure,         // orthopressure load
    neum_torque                 // torque
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

  // get type of condition
  const auto& type = condition.parameters().get<std::string>("TYPE");
  if (type == "neum_live")
  {
    ltype = neum_live;
    config = config_material;
  }
  else if (type == "neum_pseudo_orthopressure")
  {
    ltype = neum_pseudo_orthopressure;
    config = config_lastconverged;
  }
  else if (type == "neum_orthopressure")
  {
    ltype = neum_orthopressure;
    config = config_spatial;
  }
  else if (type == "neum_torque")
  {
    ltype = neum_torque;
    config = config_spatial;
  }
  else
  {
    FOUR_C_THROW("Unknown type of SurfaceNeumann condition");
  }

  // get values and switches from the condition
  const auto* onoff = &condition.parameters().get<std::vector<int>>("ONOFF");
  const auto* val = &condition.parameters().get<std::vector<double>>("VAL");
  const auto* spa_func = condition.parameters().get_if<std::vector<int>>("FUNCT");

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
  if (int(onoff->size()) < numdim)
    FOUR_C_THROW("Fewer functions or curves defined than the element has dofs.");

  // element geometry update
  const int numnode = num_node();
  const int numdf = num_dof_per_node(*nodes()[0]);
  Core::LinAlg::SerialDenseMatrix x(numnode, numdim);
  Core::LinAlg::SerialDenseMatrix xc;
  switch (config)
  {
    case config_material:
    {
      // no linearization needed for load in material configuration
      loadlin = false;

      // evaluate material configuration
      material_configuration(x);
    }
    break;
    case config_lastconverged:
    {
      // initialize last converged configuration
      xc.shape(numnode, numdim);

      // no linearization needed for load in last converged configuration
      loadlin = false;

      // evaluate last converged configuration
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == nullptr) FOUR_C_THROW("Cannot get state vector 'displacement'");
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);
      spatial_configuration(xc, mydisp);
    }
    break;
    case config_spatial:
    {
      // initialize spatial configuration
      xc.shape(numnode, numdim);


      // The true spatial configuration is the material configuration for mulf
      if (Prestress::is_mulf_active(time))
      {
        // no linearization needed for mulf
        loadlin = false;

        // evaluate material configuration
        material_configuration(xc);
      }
      else  // standard case
      {
        std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
            discretization.get_state("displacement new");
        if (disp == nullptr)
          FOUR_C_THROW(
              "Cannot get state vector 'displacement new'\n"
              "Did you forget to set the 'LOADLIN yes' in '--STRUCTURAL DYNAMIC' input section???");
        std::vector<double> mydisp(lm.size());
        Core::FE::extract_my_values(*disp, mydisp, lm);
        spatial_configuration(xc, mydisp);
      }
    }
    break;
    default:
      FOUR_C_THROW("Unknown case of frame");
      break;
  }

  // --------------------------------------------------
  // Now do the nurbs specific stuff
  bool is_nurbs_element = Core::FE::is_nurbs_celltype(shape());

  // factor for surface orientation
  double normalfac = 1.0;

  // local surface id
  const int surfaceid = l_surf_number();

  // knot vectors for parent volume and this surface
  std::vector<Core::LinAlg::SerialDenseVector> mypknots(3);
  std::vector<Core::LinAlg::SerialDenseVector> myknots(2);

  // NURBS control point weights for all nodes, ie. CPs
  Core::LinAlg::SerialDenseVector weights(numnode);

  if (is_nurbs_element)
  {
    // --------------------------------------------------
    // get knotvector
    auto& nurbsdis = dynamic_cast<Core::FE::Nurbs::NurbsDiscretization&>(discretization);
    std::shared_ptr<Core::FE::Nurbs::Knotvector> knots = (nurbsdis).get_knot_vector();
    bool zero_size = knots->get_boundary_ele_and_parent_knots(
        mypknots, myknots, normalfac, parent_element()->id(), surfaceid);
    // elements that have zero size in knotspan are skipped
    // (only required to enforce interpolation at certain points
    //  using repeated knots)
    if (zero_size) return 0;

    // --------------------------------------------------
    // get node weights for nurbs elements
    for (int inode = 0; inode < numnode; inode++)
    {
      auto* cp = dynamic_cast<Core::FE::Nurbs::ControlPoint*>(nodes()[inode]);
      weights(inode) = cp->w();
    }
  }
  // --------------------------------------------------


  // allocate vector for shape functions and matrix for derivatives
  Core::LinAlg::SerialDenseVector funct(numnode);
  Core::LinAlg::SerialDenseMatrix deriv(2, numnode);

  /*----------------------------------------------------------------------*
  |               start loop over integration points                     |
  *----------------------------------------------------------------------*/
  const Core::FE::IntegrationPoints2D intpoints(gaussrule_);
  for (int gp = 0; gp < intpoints.nquad; gp++)
  {
    // set gausspoints from integration rule
    Core::LinAlg::SerialDenseVector e(2);
    e(0) = intpoints.qxg[gp][0];
    e(1) = intpoints.qxg[gp][1];

    // get shape functions and derivatives in the plane of the element
    if (!is_nurbs_element)
    {
      Core::FE::shape_function_2d(funct, e(0), e(1), shape());
      Core::FE::shape_function_2d_deriv1(deriv, e(0), e(1), shape());
    }
    else
    {
      Core::FE::Nurbs::nurbs_get_2d_funct_deriv(
          funct, deriv, e, myknots, weights, Core::FE::CellType::nurbs9);
    }

    // Stuff to get spatial Neumann
    Core::LinAlg::SerialDenseMatrix gp_coord(1, numdim);

    switch (ltype)
    {
      case neum_live:
      {
        // check for correct input
        for (int checkdof = numdim; checkdof < int(onoff->size()); ++checkdof)
        {
          if ((*onoff)[checkdof] != 0)
            FOUR_C_THROW(
                "Number of Dimensions in Neumann_Evalutaion is 3. Further DoFs are not "
                "considered.");
        }

        Core::LinAlg::SerialDenseMatrix dxyzdrs(2, 3);
        Core::LinAlg::multiply(dxyzdrs, deriv, x);
        Core::LinAlg::SerialDenseMatrix metrictensor(2, 2);
        Core::LinAlg::multiply_nt(metrictensor, dxyzdrs, dxyzdrs);
        const double detA =
            sqrt(metrictensor(0, 0) * metrictensor(1, 1) - metrictensor(0, 1) * metrictensor(1, 0));

        double functfac = 1.0;
        int functnum = -1;

        for (int dof = 0; dof < numdim; dof++)
        {
          if ((*onoff)[dof])  // is this dof activated?
          {
            // factor given by spatial function
            if (spa_func) functnum = (*spa_func)[dof];

            if (functnum > 0)
            {
              // Calculate reference position of GP
              Core::LinAlg::multiply_tn(gp_coord, funct, x);
              // write coordinates in another datatype
              double gp_coord2[numdim];
              for (int i = 0; i < numdim; i++)
              {
                gp_coord2[i] = gp_coord(0, i);
              }
              const double* coordgpref = gp_coord2;  // needed for function evaluation

              // evaluate function at current gauss point
              functfac = Global::Problem::instance()
                             ->function_by_id<Core::Utils::FunctionOfSpaceTime>(functnum - 1)
                             .evaluate(coordgpref, time, dof);
            }
            else
              functfac = 1.0;

            const double fac = intpoints.qwgt[gp] * detA * (*val)[dof] * functfac;
            for (int node = 0; node < numnode; ++node)
            {
              elevec1[node * numdf + dof] += funct[node] * fac;
            }
          }
        }
      }
      break;

      case neum_pseudo_orthopressure:
      case neum_orthopressure:
      {
        if ((*onoff)[0] != 1) FOUR_C_THROW("orthopressure on 1st dof only!");
        for (int checkdof = 1; checkdof < 3; ++checkdof)
          if ((*onoff)[checkdof] != 0) FOUR_C_THROW("orthopressure on 1st dof only!");
        double ortho_value = (*val)[0];
        // if (!ortho_value) FOUR_C_THROW("no orthopressure value given!"); // in case of coupling
        // with redairways, there is a zero orthoval in the beginning!!!!
        std::vector<double> normal(3);
        surface_integration(normal, xc, deriv);
        // Calculate spatial position of GP
        double functfac = 1.0;
        int functnum = -1;

        // factor given by spatial function
        if (spa_func) functnum = (*spa_func)[0];

        if (functnum > 0)
        {
          Core::LinAlg::multiply_tn(gp_coord, funct, xc);
          // write coordinates in another datatype
          double gp_coord2[numdim];
          for (int i = 0; i < numdim; i++)
          {
            gp_coord2[i] = gp_coord(0, i);
          }
          const double* coordgpref = gp_coord2;  // needed for function evaluation

          // evaluate function at current gauss point
          functfac = Global::Problem::instance()
                         ->function_by_id<Core::Utils::FunctionOfSpaceTime>(functnum - 1)
                         .evaluate(coordgpref, time, 0);
        }

        const double fac = intpoints.qwgt[gp] * functfac * ortho_value * normalfac;
        for (int node = 0; node < numnode; ++node)
          for (int dim = 0; dim < 3; dim++)
            elevec1[node * numdf + dim] += funct[node] * normal[dim] * fac;

        // load linearization (if necessary)
        if (loadlin)
        {
          Core::LinAlg::SerialDenseMatrix Dnormal(numdf, numdf * numnode);
          analytical_d_surface_integration(Dnormal, xc, deriv);  // --> analytical derivative
          // automatic_d_surface_integration(Dnormal, xc, deriv);    // --> automatic derivative
          // (Sacado)

          // build surface element load linearization matrix
          // (CAREFUL: Minus sign due to the fact that external forces enter the global
          // residual vector with a minus sign, too! However, the load linaerization is
          // simply added to the global tangent stiffness matrix, thus we explicitly
          // need to set the minus sign here.)
          for (int node = 0; node < numnode; ++node)
            for (int dim = 0; dim < 3; dim++)
              for (int dof = 0; dof < elevec1.numRows(); dof++)
                (*elemat1)(node * numdf + dim, dof) -= funct[node] * Dnormal(dim, dof) * fac;
        }
      }
      break;

      case neum_torque:
      {
        // check whether only first, fourth, fifth and sixth value is set
        if ((*onoff)[0] != 1) FOUR_C_THROW("Torque value not provided!");
        if ((*onoff)[3] != 1) FOUR_C_THROW("X-coordinate of axis for torque not provided!");
        if ((*onoff)[4] != 1) FOUR_C_THROW("Y-coordinate of axis for torque not provided!");
        if ((*onoff)[5] != 1) FOUR_C_THROW("Z-coordinate of axis for torque not provided!");
        for (int checkdof = 1; checkdof < 3; ++checkdof)
          if ((*onoff)[checkdof] != 0) FOUR_C_THROW("Incorrect value for torque!");

        // get values for torque and coordinates of axis
        double torque_value = (*val)[0];
        Core::LinAlg::Matrix<3, 1> axis(true);
        axis(0) = (*val)[3];
        axis(1) = (*val)[4];
        axis(2) = (*val)[5];

        // compute normal vector (with area as length)
        std::vector<double> normal(3);
        surface_integration(normal, xc, deriv);

        // compute cross product of axis and (negative) normal
        Core::LinAlg::Matrix<3, 1> crossprod(true);
        crossprod(0) = -(axis(1) * normal[2] - axis(2) * normal[1]);
        crossprod(1) = -(axis(2) * normal[0] - axis(0) * normal[2]);
        crossprod(2) = -(axis(0) * normal[1] - axis(1) * normal[0]);

        // factor given by spatial function
        double functfac = 1.0;
        int functnum = -1;

        if (spa_func) functnum = (*spa_func)[0];
        {
          if (functnum > 0)
          {
            Core::LinAlg::multiply_tn(gp_coord, funct, xc);
            // write coordinates in another datatype
            double gp_coord2[numdim];
            for (int i = 0; i < numdim; i++)
            {
              gp_coord2[i] = gp_coord(0, i);
            }
            const double* coordgpref = gp_coord2;  // needed for function evaluation

            // evaluate function at current gauss point
            functfac = Global::Problem::instance()
                           ->function_by_id<Core::Utils::FunctionOfSpaceTime>(functnum - 1)
                           .evaluate(coordgpref, time, 0);
          }
          else
            functfac = 1.0;
        }

        const double fac = intpoints.qwgt[gp] * functfac * torque_value;
        for (int node = 0; node < numnode; ++node)
          for (int dim = 0; dim < 3; dim++)
            elevec1[node * numdf + dim] += funct[node] * crossprod(dim) * fac;
      }
      break;

      default:
        FOUR_C_THROW("Unknown type of SurfaceNeumann load");
        break;
    }

  } /* end of loop over integration points gp */

  return 0;
}

/*----------------------------------------------------------------------*
 * Evaluate normal at gp (private)                             gee 08/08|
 * ---------------------------------------------------------------------*/
void Discret::Elements::StructuralSurface::surface_integration(std::vector<double>& normal,
    const Core::LinAlg::SerialDenseMatrix& x, const Core::LinAlg::SerialDenseMatrix& deriv)
{
  // note that the length of this normal is the area dA

  // compute dXYZ / drs
  Core::LinAlg::SerialDenseMatrix dxyzdrs(2, 3);
  if (Core::LinAlg::multiply(dxyzdrs, deriv, x)) FOUR_C_THROW("multiply failed");

  normal[0] = dxyzdrs(0, 1) * dxyzdrs(1, 2) - dxyzdrs(0, 2) * dxyzdrs(1, 1);
  normal[1] = dxyzdrs(0, 2) * dxyzdrs(1, 0) - dxyzdrs(0, 0) * dxyzdrs(1, 2);
  normal[2] = dxyzdrs(0, 0) * dxyzdrs(1, 1) - dxyzdrs(0, 1) * dxyzdrs(1, 0);

  return;
}

/*----------------------------------------------------------------------*
 * Evaluate sqrt of determinant of metric at gp (private)      gee 04/08|
 * ---------------------------------------------------------------------*/
void Discret::Elements::StructuralSurface::surface_integration(double& detA,
    std::vector<double>& normal, const Core::LinAlg::SerialDenseMatrix& x,
    const Core::LinAlg::SerialDenseMatrix& deriv)
{
  // compute dXYZ / drs
  Core::LinAlg::SerialDenseMatrix dxyzdrs(2, 3);
  if (Core::LinAlg::multiply(dxyzdrs, deriv, x)) FOUR_C_THROW("multiply failed");

  /* compute covariant metric tensor G for surface element
  **                        | g11   g12 |
  **                    G = |           |
  **                        | g12   g22 |
  ** where (o denotes the inner product, xyz a vector)
  **
  **       dXYZ   dXYZ          dXYZ   dXYZ          dXYZ   dXYZ
  ** g11 = ---- o ----    g12 = ---- o ----    g22 = ---- o ----
  **        dr     dr            dr     ds            ds     ds
  */
  Core::LinAlg::SerialDenseMatrix metrictensor(2, 2);
  Core::LinAlg::multiply_nt(metrictensor, dxyzdrs, dxyzdrs);
  detA = sqrt(metrictensor(0, 0) * metrictensor(1, 1) - metrictensor(0, 1) * metrictensor(1, 0));
  normal[0] = dxyzdrs(0, 1) * dxyzdrs(1, 2) - dxyzdrs(0, 2) * dxyzdrs(1, 1);
  normal[1] = dxyzdrs(0, 2) * dxyzdrs(1, 0) - dxyzdrs(0, 0) * dxyzdrs(1, 2);
  normal[2] = dxyzdrs(0, 0) * dxyzdrs(1, 1) - dxyzdrs(0, 1) * dxyzdrs(1, 0);

  return;
}

/*----------------------------------------------------------------------*
 * Calculates dnormal/dx_j with Sacado DFAD                   popp 06/13|
 * ---------------------------------------------------------------------*/
void Discret::Elements::StructuralSurface::automatic_d_surface_integration(
    Core::LinAlg::SerialDenseMatrix& d_normal, const Core::LinAlg::SerialDenseMatrix& x,
    const Core::LinAlg::SerialDenseMatrix& deriv)
{
  // some parameters
  const int numnode = num_node();
  const int numdim = 3;
  const int numdof = numdim * numnode;

  // create vectors of Sacado type
  std::vector<Sacado::Fad::DFad<double>> saccado_x(x.numCols() * x.numRows());
  std::vector<Sacado::Fad::DFad<double>> saccado_deriv(deriv.numCols() * deriv.numRows());
  std::vector<Sacado::Fad::DFad<double>> saccado_g1(numdim);
  std::vector<Sacado::Fad::DFad<double>> saccado_g2(numdim);
  std::vector<Sacado::Fad::DFad<double>> saccado_normal(numdim);

  // copy data of coordinate matrix x
  for (int row = 0; row < x.numRows(); row++)
  {
    for (int column = 0; column < x.numCols(); column++)
    {
      saccado_x[x.numCols() * row + column] = x(row, column);
      saccado_x[x.numCols() * row + column].diff(
          x.numCols() * row + column, x.numCols() * x.numRows());
    }
  }

  // copy data of shape function derivatives matrix deriv
  for (int row = 0; row < deriv.numRows(); row++)
  {
    for (int column = 0; column < deriv.numCols(); column++)
    {
      saccado_deriv[deriv.numCols() * row + column] = deriv(row, column);
    }
  }

  // re-compute local basis vectors g1 and g2
  for (int dim = 0; dim < numdim; dim++)
  {
    for (int column = 0; column < deriv.numCols(); column++)
    {
      saccado_g1[dim] += saccado_deriv[column] * saccado_x[column * x.numCols() + dim];
      saccado_g2[dim] +=
          saccado_deriv[column + deriv.numCols()] * saccado_x[column * x.numCols() + dim];
    }
  }

  // re-compute normal vector (cross product g1 x g2)
  saccado_normal[0] = saccado_g1[1] * saccado_g2[2] - saccado_g1[2] * saccado_g2[1];
  saccado_normal[1] = saccado_g1[2] * saccado_g2[0] - saccado_g1[0] * saccado_g2[2];
  saccado_normal[2] = saccado_g1[0] * saccado_g2[1] - saccado_g1[1] * saccado_g2[0];

  // direct access to the Sacado derivatives
  for (int dim = 0; dim < numdim; dim++)
  {
    for (int dxyz = 0; dxyz < numdof; dxyz++)
    {
      d_normal(dim, dxyz) = saccado_normal[dim].fastAccessDx(dxyz);
    }
  }

  return;
}


/*----------------------------------------------------------------------*
 * Calculates dnormal/dx_j analytically                       popp 06/13|
 * ---------------------------------------------------------------------*/
void Discret::Elements::StructuralSurface::analytical_d_surface_integration(
    Core::LinAlg::SerialDenseMatrix& d_normal, const Core::LinAlg::SerialDenseMatrix& x,
    const Core::LinAlg::SerialDenseMatrix& deriv)
{
  // some parameters
  const int numnode = num_node();
  const int numsurfdim = 2;
  const int numdim = 3;
  const int numdof = numdim * numnode;

  // compute dXYZ / drs (defining the two local basis vectors)
  Core::LinAlg::SerialDenseMatrix dxyzdrs(numsurfdim, numdim);
  Core::LinAlg::multiply(dxyzdrs, deriv, x);

  // basis vectors (just ouf of laziness)
  std::vector<double> g1(numdim);
  std::vector<double> g2(numdim);

  for (int k = 0; k < numdim; ++k)
  {
    g1[k] = dxyzdrs(0, k);
    g2[k] = dxyzdrs(1, k);
  }

  // linearization of basis vectors
  Core::LinAlg::SerialDenseMatrix dg1(numdim, numdof);
  Core::LinAlg::SerialDenseMatrix dg2(numdim, numdof);

  for (int node = 0; node < numnode; ++node)
  {
    for (int k = 0; k < numdim; ++k)
    {
      dg1(k, node * numdim + k) = deriv(0, node);
      dg2(k, node * numdim + k) = deriv(1, node);
    }
  }

  // linearization of local surface normal vector
  for (int dof = 0; dof < numdof; ++dof)
  {
    d_normal(0, dof) =
        dg1(1, dof) * g2[2] + g1[1] * dg2(2, dof) - dg1(2, dof) * g2[1] - g1[2] * dg2(1, dof);
    d_normal(1, dof) =
        dg1(2, dof) * g2[0] + g1[2] * dg2(0, dof) - dg1(0, dof) * g2[2] - g1[0] * dg2(2, dof);
    d_normal(2, dof) =
        dg1(0, dof) * g2[1] + g1[0] * dg2(1, dof) - dg1(1, dof) * g2[0] - g1[1] * dg2(0, dof);
  }

  return;
}

/*----------------------------------------------------------------------*
 * Evaluate method for StructuralSurface-Elements               tk 10/07*
 * ---------------------------------------------------------------------*/
int Discret::Elements::StructuralSurface::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elematrix1, Core::LinAlg::SerialDenseMatrix& elematrix2,
    Core::LinAlg::SerialDenseVector& elevector1, Core::LinAlg::SerialDenseVector& elevector2,
    Core::LinAlg::SerialDenseVector& elevector3)
{
  // start with "none"
  Discret::Elements::StructuralSurface::ActionType act = StructuralSurface::none;

  // get the required action
  std::string action = params.get<std::string>("action", "none");
  if (action == "none")
    FOUR_C_THROW("No action supplied");
  else if (action == "calc_struct_constrvol")
    act = StructuralSurface::calc_struct_constrvol;
  else if (action == "calc_struct_volconstrstiff")
    act = StructuralSurface::calc_struct_volconstrstiff;
  else if (action == "calc_struct_monitarea")
    act = StructuralSurface::calc_struct_monitarea;
  else if (action == "calc_struct_constrarea")
    act = StructuralSurface::calc_struct_constrarea;
  else if (action == "calc_struct_areaconstrstiff")
    act = StructuralSurface::calc_struct_areaconstrstiff;
  else if (action == "calc_init_vol")
    act = StructuralSurface::calc_init_vol;
  else if (action == "calc_brownian_motion")
    act = StructuralSurface::calc_brownian_motion;
  else if (action == "calc_brownian_motion_damping")
    act = StructuralSurface::calc_brownian_motion_damping;
  else if (action == "calc_struct_centerdisp")
    act = StructuralSurface::calc_struct_centerdisp;
  else if (action == "calc_struct_rotation")
    act = StructuralSurface::calc_struct_rotation;
  else if (action == "calc_undo_struct_rotation")
    act = StructuralSurface::calc_undo_struct_rotation;
  else if (action == "calc_struct_area")
    act = StructuralSurface::calc_struct_area;
  else if (action == "calc_ref_nodal_normals")
    act = StructuralSurface::calc_ref_nodal_normals;
  else if (action == "calc_cur_normal_at_point")
    act = StructuralSurface::calc_cur_normal_at_point;
  else if (action == "calc_cur_nodal_normals")
    act = StructuralSurface::calc_cur_nodal_normals;
  else if (action == "calc_fluid_traction")
    act = StructuralSurface::calc_fluid_traction;
  else if (action == "mark_immersed_elements")
    act = StructuralSurface::mark_immersed_elements;
  else if (action == "calc_struct_robinforcestiff")
    act = StructuralSurface::calc_struct_robinforcestiff;
  else
  {
    std::cout << action << std::endl;
    FOUR_C_THROW("Unknown type of action for StructuralSurface");
  }

  // create communicator
  MPI_Comm Comm = discretization.get_comm();
  // what the element has to do
  switch (act)
  {
    // gives the center displacement for SlideALE
    case calc_struct_centerdisp:
    {
      // We are not interested in ghosted elements
      if (Core::Communication::my_mpi_rank(Comm) == owner())
      {
        // element geometry update
        std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
            discretization.get_state("displacementtotal");
        if (disp == nullptr) FOUR_C_THROW("Cannot get state vector 'displacementtotal'");
        std::vector<double> mydisp(lm.size());
        Core::FE::extract_my_values(*disp, mydisp, lm);
        const int numnode = num_node();
        const int numdf = 3;
        Core::LinAlg::SerialDenseMatrix xc(numnode, numdf);
        spatial_configuration(xc, mydisp);

        // integration of the displacements over the surface
        // allocate vector for shape functions and matrix for derivatives
        Core::LinAlg::SerialDenseVector funct(numnode);
        Core::LinAlg::SerialDenseMatrix deriv(2, numnode);

        /*----------------------------------------------------------------------*
          |               start loop over integration points                     |
          *----------------------------------------------------------------------*/
        const Core::FE::IntegrationPoints2D intpoints(gaussrule_);

        std::shared_ptr<const Core::LinAlg::Vector<double>> dispincr =
            discretization.get_state("displacementincr");
        std::vector<double> edispincr(lm.size());
        Core::FE::extract_my_values(*dispincr, edispincr, lm);
        elevector2[0] = 0;

        for (int gp = 0; gp < intpoints.nquad; gp++)
        {
          const double e0 = intpoints.qxg[gp][0];
          const double e1 = intpoints.qxg[gp][1];

          // get shape functions and derivatives in the plane of the element
          Core::FE::shape_function_2d(funct, e0, e1, shape());
          Core::FE::shape_function_2d_deriv1(deriv, e0, e1, shape());

          std::vector<double> normal(3);
          double detA;
          surface_integration(detA, normal, xc, deriv);

          elevector2[0] +=
              sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);

          for (int dim = 0; dim < 3; dim++)
          {
            for (int j = 0; j < numnode; ++j)
            {
              elevector3[dim] += funct[j] * intpoints.qwgt[gp] * edispincr[j * numdf + dim] * detA;
            }
          }
        }
      }
    }
    break;
    case calc_struct_rotation:
    {
      // We are not interested in ghosted elements
      if (Core::Communication::my_mpi_rank(Comm) == owner())
      {
        const double maxcoord = params.get<double>("maxcoord");
        Inpar::FSI::SlideALEProj aletype = params.get<Inpar::FSI::SlideALEProj>("aletype");
        const int numnode = num_node();
        const int numdf = 3;
        double tol = 1.0E-5;

        //  element geometry update for time t_n
        std::shared_ptr<const Core::LinAlg::Vector<double>> dispn =
            discretization.get_state("displacementnp");
        if (dispn == nullptr) FOUR_C_THROW("Cannot get state vector 'displacementnp");
        std::vector<double> edispn(lm.size());
        Core::FE::extract_my_values(*dispn, edispn, lm);
        Core::LinAlg::SerialDenseMatrix xcn(numnode, numdf);
        spatial_configuration(xcn, edispn);

        std::shared_ptr<const Core::LinAlg::Vector<double>> dispincr =
            discretization.get_state("displacementincr");
        if (dispn == nullptr) FOUR_C_THROW("Cannot get state vector 'displacementincr");
        std::vector<double> edispincr(lm.size());
        Core::FE::extract_my_values(*dispincr, edispincr, lm);

        // integration of the displacements over the surface
        // allocate vector for shape functions and matrix for derivatives
        Core::LinAlg::SerialDenseVector funct(numnode);
        Core::LinAlg::SerialDenseMatrix deriv(2, numnode);

        // loop over integration points
        const Core::FE::IntegrationPoints2D intpoints(gaussrule_);
        for (int gp = 0; gp < intpoints.nquad; gp++)
        {
          const double e0 = intpoints.qxg[gp][0];
          const double e1 = intpoints.qxg[gp][1];

          // get shape functions and derivatives in the plane of the element
          Core::FE::shape_function_2d(funct, e0, e1, shape());
          Core::FE::shape_function_2d_deriv1(deriv, e0, e1, shape());

          std::vector<double> normal(3);
          double detA;
          surface_integration(detA, normal, xcn, deriv);

          Core::LinAlg::SerialDenseVector tangent(3);
          if (aletype == Inpar::FSI::ALEprojection_rot_z ||
              aletype == Inpar::FSI::ALEprojection_rot_zsphere)
          {
            // compute tangential direction in xy-plane from normal
            tangent[0] = -normal[1];
            tangent[1] = normal[0];
            tangent[2] = 0.0;
          }
          else
          {
            FOUR_C_THROW("rotation not yet implemented!");
          }

          if (Core::LinAlg::norm2(tangent) > tol)
          {
            tangent.scale(1.0 / (Core::LinAlg::norm2(tangent)));
          }
          elevector2[0] +=
              sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);

          for (int node = 0; node < numnode; ++node)
          {
            double scalarprod =
                tangent[0] * edispincr[node * numdf] + tangent[1] * edispincr[node * numdf + 1];
            if (aletype == Inpar::FSI::ALEprojection_rot_zsphere)
            {
              double circ(0.0);
              const double val = (1.0 - pow(xcn(node, 2) / maxcoord, 2.0));
              if (val < 0.0)  // negative doubles can happen due to round-off errors
              {
                if (val > -1e-10)  // seems to be a round-off error, we proceed assuming val=0.0
                {
                  circ = 0.0;
                }
                else  // severe error
                  FOUR_C_THROW("Do not use sqrt() with a negative number");
              }
              else
              {
                circ = sqrt(val);
              }
              if (circ > tol)
                elevector3[0] += funct[node] * intpoints.qwgt[gp] * scalarprod * detA / circ;
            }
            else
              elevector3[0] += funct[node] * intpoints.qwgt[gp] * scalarprod * detA;
          }
        }
      }
    }
    break;
    case calc_undo_struct_rotation:
    {
      const double maxcoord = params.get<double>("maxcoord");
      Inpar::FSI::SlideALEProj aletype = params.get<Inpar::FSI::SlideALEProj>("aletype");
      const int numnode = num_node();
      const int numdf = 3;
      double tol = 1.0E-5;

      //  element geometry update for time t_n
      std::shared_ptr<const Core::LinAlg::Vector<double>> dispn =
          discretization.get_state("displacementnp");
      if (dispn == nullptr) FOUR_C_THROW("Cannot get state vector 'displacementnp");
      std::vector<double> edispn(lm.size());
      Core::FE::extract_my_values(*dispn, edispn, lm);
      Core::LinAlg::SerialDenseMatrix xcn(numnode, numdf);
      spatial_configuration(xcn, edispn);

      std::shared_ptr<const Core::LinAlg::Vector<double>> dispincr =
          discretization.get_state("displacementincr");
      if (dispn == nullptr) FOUR_C_THROW("Cannot get state vector 'displacementincr");
      std::vector<double> edispincr(lm.size());
      Core::FE::extract_my_values(*dispincr, edispincr, lm);

      // integration of the displacements over the surface
      // allocate vector for shape functions and matrix for derivatives
      Core::LinAlg::SerialDenseVector funct(numnode);
      Core::LinAlg::SerialDenseMatrix deriv(2, numnode);

      std::vector<double> nodalrot(numnode);
      for (int node = 0; node < numnode; node++)
      {
        std::vector<double> normal(3);  // normal in element center
        // get shape functions and derivatives in the plane of the element
        Core::FE::shape_function_2d(funct, 0.0, 0.0, shape());
        Core::FE::shape_function_2d_deriv1(deriv, 0.0, 0.0, shape());

        surface_integration(normal, xcn, deriv);

        Core::LinAlg::SerialDenseVector tangent(3);
        if (aletype == Inpar::FSI::ALEprojection_rot_z ||
            aletype == Inpar::FSI::ALEprojection_rot_zsphere)
        {
          // compute tangential direction in xy-plane from normal
          tangent[0] = -normal[1];
          tangent[1] = normal[0];
          tangent[2] = 0.0;
        }
        else
        {
          FOUR_C_THROW("rotation not yet implemented!");
        }

        if (Core::LinAlg::norm2(tangent) > tol)
        {
          tangent.scale(1.0 / (Core::LinAlg::norm2(tangent) * nodes()[node]->num_element()));
        }

        if (aletype == Inpar::FSI::ALEprojection_rot_zsphere)
        {
          double circ(0.0);
          const double val = (1.0 - pow(xcn(node, 2) / maxcoord, 2.0));
          if (val < 0.0)  // negative doubles can happen due to round-off errors
          {
            if (val > -1e-10)  // seems to be a round-off error, we proceed assuming val=0.0
            {
              circ = 0.0;
            }
            else  // severe error
              FOUR_C_THROW("Do not use sqrt() with a negative number");
          }
          else
          {
            circ = sqrt(val);
          }
          if (circ > tol)
          {
            for (int dof = 0; dof < 2; dof++) elevector1[node * numdf + dof] = tangent[dof] * circ;
          }
        }
        else
        {
          for (int dof = 0; dof < 2; dof++) elevector1[node * numdf + dof] = tangent[dof];
        }
      }
    }
    break;
    case calc_struct_constrvol:
    {
      // We are not interested in volume of ghosted elements
      if (Core::Communication::my_mpi_rank(Comm) == owner())
      {
        // element geometry update
        std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
            discretization.get_state("displacement");
        if (disp == nullptr) FOUR_C_THROW("Cannot get state vector 'displacement'");
        std::vector<double> mydisp(lm.size());
        Core::FE::extract_my_values(*disp, mydisp, lm);
        const int numdim = 3;
        Core::LinAlg::SerialDenseMatrix xscurr(num_node(), numdim);  // material coord. of element
        spatial_configuration(xscurr, mydisp);
        // call submethod for volume evaluation and store rseult in third systemvector
        double volumeele = compute_constr_vols(xscurr, num_node());
        elevector3[0] = volumeele;
      }
    }
    break;
    case calc_struct_volconstrstiff:
    {
      // element geometry update
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == nullptr) FOUR_C_THROW("Cannot get state vector 'displacement'");
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);
      const int numdim = 3;
      Core::LinAlg::SerialDenseMatrix xscurr(num_node(), numdim);  // material coord. of element
      spatial_configuration(xscurr, mydisp);
      double volumeele;
      // first partial derivatives
      Core::LinAlg::SerialDenseVector Vdiff1;
      // second partial derivatives
      std::shared_ptr<Core::LinAlg::SerialDenseMatrix> Vdiff2 =
          std::make_shared<Core::LinAlg::SerialDenseMatrix>();

      // get projection method
      std::shared_ptr<Core::Conditions::Condition> condition =
          params.get<std::shared_ptr<Core::Conditions::Condition>>("condition");
      const auto* projtype = condition->parameters().get_if<std::string>("projection");

      if (projtype != nullptr)
      {
        // call submethod to compute volume and its derivatives w.r.t. to current displ.
        if (*projtype == "yz")
        {
          compute_vol_deriv(
              xscurr, num_node(), numdim * num_node(), volumeele, Vdiff1, Vdiff2, 0, 0);
        }
        else if (*projtype == "xz")
        {
          compute_vol_deriv(
              xscurr, num_node(), numdim * num_node(), volumeele, Vdiff1, Vdiff2, 1, 1);
        }
        else if (*projtype == "xy")
        {
          compute_vol_deriv(
              xscurr, num_node(), numdim * num_node(), volumeele, Vdiff1, Vdiff2, 2, 2);
        }
        else
        {
          compute_vol_deriv(xscurr, num_node(), numdim * num_node(), volumeele, Vdiff1, Vdiff2);
        }
      }
      else
        compute_vol_deriv(xscurr, num_node(), numdim * num_node(), volumeele, Vdiff1, Vdiff2);

      // update rhs vector and corresponding column in "constraint" matrix
      elevector1 = Vdiff1;
      elevector2 = Vdiff1;
      elematrix1 = *Vdiff2;
      // call submethod for volume evaluation and store result in third systemvector
      elevector3[0] = volumeele;
    }
    break;
    case calc_init_vol:
    {
      if (Core::Communication::my_mpi_rank(Comm) == owner())
      {
        // the reference volume of the RVE (including inner
        // holes) is calculated by evaluating the following
        // surface integral:
        // V = 1/3*int(div(X))dV = 1/3*int(N*X)dA
        // with X being the reference coordinates and N the
        // normal vector of the surface element (exploiting the
        // fact that div(X)=1.0)

        // this is intended to be used in the serial case (microstructure)

        // NOTE: there must not be any holes penetrating the boundary!

        double V = params.get<double>("V0", 0.0);
        double dV = 0.0;
        const int numnode = num_node();
        Core::LinAlg::SerialDenseMatrix x(numnode, 3);
        material_configuration(x);

        // allocate vector for shape functions and matrix for derivatives
        Core::LinAlg::SerialDenseVector funct(numnode);
        Core::LinAlg::SerialDenseMatrix deriv(2, numnode);

        /*----------------------------------------------------------------------*
             |               start loop over integration points                     |
         *----------------------------------------------------------------------*/
        const Core::FE::IntegrationPoints2D intpoints(gaussrule_);

        for (int gp = 0; gp < intpoints.nquad; gp++)
        {
          const double e0 = intpoints.qxg[gp][0];
          const double e1 = intpoints.qxg[gp][1];

          // get shape functions and derivatives in the plane of the element
          Core::FE::shape_function_2d(funct, e0, e1, shape());
          Core::FE::shape_function_2d_deriv1(deriv, e0, e1, shape());

          std::vector<double> normal(3);
          double detA;
          surface_integration(detA, normal, x, deriv);
          const double fac = intpoints.qwgt[gp] * detA;

          double temp = 0.0;
          std::vector<double> X(3, 0.);

          for (int i = 0; i < numnode; i++)
          {
            X[0] += funct[i] * x(i, 0);
            X[1] += funct[i] * x(i, 1);
            X[2] += funct[i] * x(i, 2);
          }

          for (int i = 0; i < 3; ++i)
          {
            temp += normal[i] * normal[i];
          }

          if (temp < 0.) FOUR_C_THROW("calculation of initial volume failed in surface element");
          double absnorm = sqrt(temp);

          for (int i = 0; i < 3; ++i)
          {
            normal[i] /= absnorm;
          }
          for (int i = 0; i < 3; ++i)
          {
            dV += 1 / 3.0 * fac * normal[i] * X[i];
          }
        }
        params.set("V0", V + dV);
      }
    }
    break;
    // compute stochastical forces due to Brownian Motion
    case calc_brownian_motion:
    {
      FOUR_C_THROW("not commited");
    }
    break;
    // compute damping matrix due to Brownian Motion
    case calc_brownian_motion_damping:
    {
      FOUR_C_THROW("not yet comitted");
    }
    break;
    // compute the area (e.g. for initialization)
    case calc_struct_monitarea:
    {
      // We are not interested in volume of ghosted elements
      if (Core::Communication::my_mpi_rank(Comm) == owner())
      {
        // element geometry update
        std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
            discretization.get_state("displacement");
        if (disp == nullptr) FOUR_C_THROW("Cannot get state vector 'displacement'");
        std::vector<double> mydisp(lm.size());
        Core::FE::extract_my_values(*disp, mydisp, lm);
        const int numdim = 3;
        Core::LinAlg::SerialDenseMatrix xscurr(num_node(), numdim);  // material coord. of element
        spatial_configuration(xscurr, mydisp);

        std::shared_ptr<Core::Conditions::Condition> condition =
            params.get<std::shared_ptr<Core::Conditions::Condition>>("condition");
        const auto* projtype = condition->parameters().get_if<std::string>("projection");

        // To compute monitored area consider required projection method
        // and set according coordinates to zero
        if (*projtype == "yz")
        {
          xscurr(0, 0) = 0;
          xscurr(1, 0) = 0;
          xscurr(2, 0) = 0;
          xscurr(3, 0) = 0;
        }
        else if (*projtype == "xz")
        {
          xscurr(0, 1) = 0;
          xscurr(1, 1) = 0;
          xscurr(2, 1) = 0;
          xscurr(3, 1) = 0;
        }
        else if (*projtype == "xy")
        {
          xscurr(0, 2) = 0;
          xscurr(1, 2) = 0;
          xscurr(2, 2) = 0;
          xscurr(3, 2) = 0;
        }

        double areaele = 0.0;
        const Core::FE::IntegrationPoints2D intpoints(gaussrule_);
        // allocate matrix for derivatives of shape functions
        Core::LinAlg::SerialDenseMatrix deriv(2, num_node());

        // Compute area
        for (int gp = 0; gp < intpoints.nquad; gp++)
        {
          const double e0 = intpoints.qxg[gp][0];
          const double e1 = intpoints.qxg[gp][1];

          // get shape functions and derivatives in the plane of the element
          Core::FE::shape_function_2d_deriv1(deriv, e0, e1, shape());

          std::vector<double> normal(3);
          double detA;
          surface_integration(detA, normal, xscurr, deriv);
          const double fac = intpoints.qwgt[gp] * detA;
          areaele += fac;
        }

        // store result in third systemvector
        elevector3[0] = areaele;
      }
    }
    break;
    case calc_struct_constrarea:
    {
      // element geometry update
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == nullptr) FOUR_C_THROW("Cannot get state vector 'displacement'");
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);
      const int numdim = 3;
      Core::LinAlg::SerialDenseMatrix xscurr(num_node(), numdim);  // material coord. of element
      spatial_configuration(xscurr, mydisp);
      // initialize variables
      double elearea;
      // first partial derivatives
      Core::LinAlg::SerialDenseVector Adiff;
      // second partial derivatives
      std::shared_ptr<Core::LinAlg::SerialDenseMatrix> Adiff2 =
          std::make_shared<Core::LinAlg::SerialDenseMatrix>();

      // call submethod
      compute_area_deriv(xscurr, num_node(), numdim * num_node(), elearea, Adiff, Adiff2);
      // store result
      elevector3[0] = elearea;
    }
    break;
    case calc_struct_areaconstrstiff:
    {
      // element geometry update
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == nullptr) FOUR_C_THROW("Cannot get state vector 'displacement'");
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);
      const int numdim = 3;
      Core::LinAlg::SerialDenseMatrix xscurr(num_node(), numdim);  // material coord. of element
      spatial_configuration(xscurr, mydisp);
      // initialize variables
      double elearea;
      // first partial derivatives
      Core::LinAlg::SerialDenseVector Adiff;
      // second partial derivatives
      std::shared_ptr<Core::LinAlg::SerialDenseMatrix> Adiff2 =
          std::make_shared<Core::LinAlg::SerialDenseMatrix>();

      // call submethod
      compute_area_deriv(xscurr, num_node(), numdim * num_node(), elearea, Adiff, Adiff2);
      // update elematrices and elevectors
      elevector1 = Adiff;
      elevector1.scale(-1.0);
      elevector2 = elevector1;
      elematrix1 = *Adiff2;
      elematrix1.scale(-1.0);
      elevector3[0] = elearea;
    }
    break;
    case calc_struct_area:
    {
      const int numnode = num_node();
      Core::LinAlg::SerialDenseMatrix x(numnode, 3);
      material_configuration(x);
      // Core::LinAlg::SerialDenseVector  funct(numnode);
      Core::LinAlg::SerialDenseMatrix deriv(2, numnode);
      const Core::FE::IntegrationPoints2D intpoints(gaussrule_);
      double a = 0.0;
      for (int gp = 0; gp < intpoints.nquad; gp++)
      {
        const double e0 = intpoints.qxg[gp][0];
        const double e1 = intpoints.qxg[gp][1];
        Core::FE::shape_function_2d_deriv1(deriv, e0, e1, shape());
        std::vector<double> normal(3);
        double detA;
        surface_integration(detA, normal, x, deriv);
        a += (intpoints.qwgt[gp] * detA);
      }
      double atmp = params.get("area", -1.0);
      a += atmp;
      params.set("area", a);
    }
    break;
    case calc_ref_nodal_normals:
    {
      // dummy vector
      std::vector<double> dummy(lm.size());
      build_normals_at_nodes(elevector1, dummy, true);
    }
    break;
    case calc_cur_nodal_normals:
    {
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == nullptr) FOUR_C_THROW("Cannot get state vector 'displacement'");
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);
      build_normals_at_nodes(elevector1, mydisp, false);
    }
    break;
    case calc_cur_normal_at_point:
    {
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == nullptr) FOUR_C_THROW("Cannot get state vector 'displacement'");
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);

      const int numnode = num_node();
      const int numdim = 3;

      Core::LinAlg::SerialDenseMatrix x(numnode, 3);
      spatial_configuration(x, mydisp);

      const double e0 = elevector2(0);
      const double e1 = elevector2(1);

      // allocate vector for shape functions and matrix for derivatives
      Core::LinAlg::SerialDenseVector funct(numnode);
      Core::LinAlg::SerialDenseMatrix deriv(2, numnode);

      // get shape functions and derivatives in the plane of the element
      Core::FE::shape_function_2d(funct, e0, e1, shape());
      Core::FE::shape_function_2d_deriv1(deriv, e0, e1, shape());

      double detA;
      std::vector<double> normal(3);
      surface_integration(detA, normal, x, deriv);

      for (int j = 0; j < numdim; ++j)
      {
        elevector1(j) = normal[j];
      }
    }
    break;
    case calc_fluid_traction:
    {
      Global::Problem* globalproblem = Global::Problem::instance();
      std::string backgrddisname(params.get<std::string>("backgrddisname"));
      std::string immerseddisname(params.get<std::string>("immerseddisname"));
      ////////////////////////////////////////////////////////////////////////////
      // rauch 05/14 originally for immersed method
      //
      // this action integrates the fluid stress over the structural surface.
      // the fluid velocities and pressure are interpolated to the structural
      // integration point externally by the method:
      //
      // InterpolateToImmersedIntPoint(...)
      //
      // which is a generally applicable method for this purpose returning
      // the desired quantitiy from another field interpolated to the current
      // integration point belonging to THIS field.
      //
      // fluiddis  = backgrddis
      // structdis = immerseddis
      //
      // if discretizations named differently are to be dealt with, provide
      // their names via the parameter list.
      //
      // CALC:   /
      //        /
      //        |
      //        | P * N * du  dA = (J * \sigma * F^-T * N, du)_{d B_0}
      //        |
      //       /
      //      / d B_0
      //
      // \sigma can be calculated from interpolated velocity gradient and pressure.
      // J is the usual jacobian determinant.
      // F is the deformation gradient of the structural surface ele.
      // N is the normal in reference configuration of the structural surface ele.
      // du denotes the virtual displacement field over the structural surface ele.
      //
      //
      /////////////////////////////////////////////////////////////////////////////

      // get integration rule
      const Core::FE::IntPointsAndWeights<2> intpoints(
          Discret::Elements::DisTypeToOptGaussRule<Core::FE::CellType::quad4>::rule);

      const std::shared_ptr<Core::FE::Discretization> backgrddis =
          globalproblem->get_dis(backgrddisname);
      const std::shared_ptr<Core::FE::Discretization> immerseddis =
          globalproblem->get_dis(immerseddisname);

#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (backgrddis == nullptr)
        FOUR_C_THROW(
            "Pointer to background dis empty. Correct disname in parameter list 'params'?");
      if (immerseddis == nullptr)
        FOUR_C_THROW("Pointer to immersed dis empty. Correct disname in parameter list 'params'?");
#endif

      const int nen = num_node();
      const int parent_nen = this->parent_element()->num_node();
      const int numdofpernode = num_dof_per_node(*(nodes()[0]));
      const int globdim = globalproblem->n_dim();

      Core::LinAlg::Matrix<2, 4> deriv;
      Core::LinAlg::Matrix<1, 4> funct;
      Core::LinAlg::Matrix<1, 8> parent_funct;
      Core::LinAlg::Matrix<3, 8> parent_deriv;
      Core::LinAlg::Matrix<3, 8> parent_deriv_notrafo;

      // get parent location matrix
      Core::Elements::LocationArray parent_la(immerseddis->num_dof_sets());
      this->parent_element()->location_vector(*immerseddis, parent_la, false);

      // get structural state and element displacements (parent element)
      std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp =
          discretization.get_state("displacement");
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (dispnp == nullptr) FOUR_C_THROW("Cannot get state vector 'displacement'");
#endif
      std::vector<double> parenteledisp(lm.size());
      std::vector<double> bdryeledisp(lm.size());
      Core::FE::extract_my_values(*dispnp, bdryeledisp, lm);
      Core::FE::extract_my_values(*dispnp, parenteledisp, parent_la[0].lm_);

      // geometry (surface ele)
      Core::LinAlg::Matrix<3, 4> xrefe;  // material coord. of element

      Core::Nodes::Node** nodes = this->nodes();
      for (int i = 0; i < nen; ++i)
      {
        const auto& x = nodes[i]->x();
        xrefe(0, i) = x[0];
        xrefe(1, i) = x[1];
        xrefe(2, i) = x[2];
      }

      // update element geometry (parent ele)
      Core::LinAlg::Matrix<3, 8> parent_xrefe;  // material coord. of element
      Core::LinAlg::Matrix<3, 8> parent_xcurr;  // current  coord. of element

      nodes = this->parent_element()->nodes();
      for (int i = 0; i < parent_nen; ++i)
      {
        const auto& x = nodes[i]->x();
        parent_xrefe(0, i) = x[0];
        parent_xrefe(1, i) = x[1];
        parent_xrefe(2, i) = x[2];

        parent_xcurr(0, i) = parent_xrefe(0, i) + parenteledisp[i * numdofpernode + 0];
        parent_xcurr(1, i) = parent_xrefe(1, i) + parenteledisp[i * numdofpernode + 1];
        parent_xcurr(2, i) = parent_xrefe(2, i) + parenteledisp[i * numdofpernode + 2];
      }

      // get coordinates of gauss points w.r.t. local parent coordinate system
      Core::LinAlg::SerialDenseMatrix parent_xi(intpoints.ip().nquad, globdim);
      Core::LinAlg::SerialDenseMatrix derivtrafo(3, 3);

      Core::FE::boundary_gp_to_parent_gp<3>(parent_xi, derivtrafo, intpoints,
          Core::FE::CellType::hex8, Core::FE::CellType::quad4, this->face_parent_number());

      ////////////////////////////////////////////////////////////////////
      /////   gauss point loop
      ///////////////////////////////////////////////////////////////////
      for (int gp = 0; gp < intpoints.ip().nquad; gp++)
      {
        std::vector<double> bdryxi(globdim - 1);
        bdryxi[0] = intpoints.ip().qxg[gp][0];
        bdryxi[1] = intpoints.ip().qxg[gp][1];

        std::vector<double> interpolationresult(7);
        FLD::Action action = FLD::interpolate_velgrad_to_given_point;

        Immersed::interpolate_to_immersed_int_point<Core::FE::CellType::hex8,  // source
            Core::FE::CellType::quad4>                                         // target
            (*backgrddis, *immerseddis, *this, bdryxi, bdryeledisp, action,
                interpolationresult  // result
            );

        // get shape functions and derivatives in the plane of the element
        Core::FE::shape_function_2d(funct, bdryxi[0], bdryxi[1], shape());
        Core::FE::shape_function_2d_deriv1(deriv, bdryxi[0], bdryxi[1], shape());

        Core::FE::shape_function_3d(parent_funct, parent_xi(gp, 0), parent_xi(gp, 1),
            parent_xi(gp, 2), this->parent_element()->shape());
        Core::FE::shape_function_3d_deriv1(parent_deriv_notrafo, parent_xi(gp, 0), parent_xi(gp, 1),
            parent_xi(gp, 2), this->parent_element()->shape());

        ////////////////////////////////////////////////////
        // calc unitnormal N in material configuration
        ///////////////////////////////////////////////////
        Core::LinAlg::Matrix<3, 1> unitnormal;
        std::vector<double> normal(globdim);

        // note that the length of this normal is the area dA

        // compute dXYZ / drs
        Core::LinAlg::Matrix<2, 3> dxyzdrs;
        dxyzdrs.multiply_nt(
            deriv, xrefe);  // to calculate unitnormal in current config. argument must be xcurr

        normal[0] = dxyzdrs(0, 1) * dxyzdrs(1, 2) - dxyzdrs(0, 2) * dxyzdrs(1, 1);
        normal[1] = dxyzdrs(0, 2) * dxyzdrs(1, 0) - dxyzdrs(0, 0) * dxyzdrs(1, 2);
        normal[2] = dxyzdrs(0, 0) * dxyzdrs(1, 1) - dxyzdrs(0, 1) * dxyzdrs(1, 0);

        Core::LinAlg::Matrix<2, 2> metrictensor;
        metrictensor.multiply_nt(dxyzdrs, dxyzdrs);
        double detA =
            sqrt(metrictensor(0, 0) * metrictensor(1, 1) - metrictensor(0, 1) * metrictensor(1, 0));

        for (int i = 0; i < globdim; ++i) unitnormal(i, 0) = normal[i];
        const double norm2 = unitnormal.norm2();
        unitnormal.scale(1 / norm2);

        /////////////////////////////////////////////////////////////////////
        // extract cauchy stress tensor from result vector of interpolation
        /////////////////////////////////////////////////////////////////////
        Core::LinAlg::Matrix<3, 3> cauchystress;

        cauchystress(0, 0) = interpolationresult[0];
        cauchystress(1, 1) = interpolationresult[1];
        cauchystress(2, 2) = interpolationresult[2];
        cauchystress(0, 1) = interpolationresult[3];
        cauchystress(1, 2) = interpolationresult[4];
        cauchystress(0, 2) = interpolationresult[5];
        cauchystress(1, 0) = cauchystress(0, 1);
        cauchystress(2, 1) = cauchystress(1, 2);
        cauchystress(2, 0) = cauchystress(0, 2);

        /* get the inverse of the Jacobian matrix which looks like:
        **            [ x_,r  y_,r  z_,r ]^-1
        **     J^-1 = [ x_,s  y_,s  z_,s ]
        **            [ x_,t  y_,t  z_,t ]
        */
        // compute derivatives N_XYZ at gp w.r.t. material coordinates
        // by N_XYZ = J^-1 * N_rst
        Core::LinAlg::Matrix<3, 3> invJ(true);
        Core::LinAlg::Matrix<3, 8> N_XYZ(true);
        invJ.multiply_nt(parent_deriv_notrafo, parent_xrefe);
        invJ.invert();
        N_XYZ.multiply(invJ, parent_deriv_notrafo);

        // (material) deformation gradient F = d xcurr / d xrefe = xcurr * N_XYZ^T
        Core::LinAlg::Matrix<3, 3> defgrd_inv;
        // deformation gradient
        defgrd_inv.multiply_nt(parent_xcurr, N_XYZ);
        // Jacobian determinant
        double J = defgrd_inv.determinant();
        // invert deformation gradient
        defgrd_inv.invert();

        Core::LinAlg::Matrix<3, 1> tempvec;
        Core::LinAlg::Matrix<3, 3> tempmat;
        tempmat.multiply_nt(cauchystress, defgrd_inv);

        tempvec.multiply_nn(tempmat, unitnormal);

        double gpweight = intpoints.ip().qwgt[gp];
        // fill element vector
        for (int node = 0; node < nen; node++)
        {
          for (int dof = 0; dof < globdim; dof++)
          {
            elevector1(node * numdofpernode + dof) +=
                (gpweight * detA) * J * tempvec(dof, 0) * funct(node);
          }
        }

        if (elevector2.numRows() > 0)
        {
          // just pressure part of traction
          Core::LinAlg::Matrix<3, 3> pressure_part;
          pressure_part(0, 0) = interpolationresult[6];
          pressure_part(1, 1) = interpolationresult[6];
          pressure_part(2, 2) = interpolationresult[6];

          tempmat.clear();
          tempvec.clear();

          tempmat.multiply_nt(pressure_part, defgrd_inv);
          tempvec.multiply_nn(tempmat, unitnormal);

          // fill element vector
          for (int node = 0; node < nen; node++)
          {
            for (int dof = 0; dof < globdim; dof++)
            {
              elevector2(node * numdofpernode + dof) +=
                  (gpweight * detA) * J * tempvec(dof, 0) * funct(node);
            }
          }
        }  // if elevector2 exists

      }  // gauss point loop
    }
    break;
      break;
    case mark_immersed_elements:
    {
      // obsolete
    }
    break;

    // a surface element routine that calculates the force and stiff contributions of a structural
    // Robin boundary condition (a spring and/or dashpot per unit reference area) - mhv 08/2016
    case calc_struct_robinforcestiff:
    {
      std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp =
          discretization.get_state("displacement");
      std::shared_ptr<const Core::LinAlg::Vector<double>> velonp =
          discretization.get_state("velocity");
      std::shared_ptr<const Core::LinAlg::Vector<double>> offset_prestress =
          discretization.get_state("offset_prestress");

      // time-integration factor for stiffness contribution of dashpot, d(v_{n+1})/d(d_{n+1})
      const double time_fac = params.get("time_fac", 0.0);

      const auto* onoff = params.get<const std::vector<int>*>("ONOFF");
      auto springstiff = *(params.get<const std::vector<double>*>("springstiff"));
      auto dashpotvisc = *(params.get<const std::vector<double>*>("dashpotvisc"));
      auto disploffset = *(params.get<const std::vector<double>*>("DISPLOFFSET"));
      const auto* numfuncstiff = params.get<const std::vector<int>*>("TIMEFUNCTSTIFF");
      const auto* numfuncvisco = params.get<const std::vector<int>*>("TIMEFUNCTVISCO");
      const auto* numfuncdisploffset = params.get<const std::vector<int>*>("TIMEFUNCTDISPLOFFSET");
      const auto* numfuncnonlinstiff = params.get<const std::vector<int>*>("FUNCTNONLINSTIFF");

      const double time = parent_element()->is_params_interface()
                              ? parent_element()->params_interface_ptr()->get_total_time()
                              : params.get("total time", 0.0);


      // scale coefficients with time function if activated
      for (auto i = 0U; i < numfuncstiff->size(); ++i)
      {
        if ((*numfuncnonlinstiff)[i] == 0)
        {
          springstiff[i] = (*numfuncstiff)[i] != 0
                               ? springstiff[i] * Global::Problem::instance()
                                                      ->function_by_id<Core::Utils::FunctionOfTime>(
                                                          (*numfuncstiff)[i] - 1)
                                                      .evaluate(time)
                               : springstiff[i];
        }
      }

      for (auto i = 0U; i < numfuncvisco->size(); ++i)
        dashpotvisc[i] = (*numfuncvisco)[i] != 0
                             ? dashpotvisc[i] * Global::Problem::instance()
                                                    ->function_by_id<Core::Utils::FunctionOfTime>(
                                                        (*numfuncvisco)[i] - 1)
                                                    .evaluate(time)
                             : dashpotvisc[i];

      for (auto i = 0U; i < numfuncdisploffset->size(); ++i)
        disploffset[i] = (*numfuncdisploffset)[i] != 0
                             ? disploffset[i] * Global::Problem::instance()
                                                    ->function_by_id<Core::Utils::FunctionOfTime>(
                                                        (*numfuncdisploffset)[i] - 1)
                                                    .evaluate(time)
                             : disploffset[i];

      // type of Robin conditions
      enum RobinType
      {
        none,
        xyz,
        refsurfnormal,
        cursurfnormal
      };

      RobinType rtype = none;

      // get type of Robin condition
      const std::string* direction = params.get<const std::string*>("DIRECTION");
      if (*direction == "xyz")
        rtype = xyz;
      else if (*direction == "refsurfnormal")
        rtype = refsurfnormal;
      else if (*direction == "cursurfnormal")
        rtype = cursurfnormal;
      else
        FOUR_C_THROW("Unknown type of Robin condition");


      // element geometry update
      const int numdim = 3;
      const int numnode = num_node();

      const int numdf = num_dof_per_node(*nodes()[0]);
      Core::LinAlg::SerialDenseMatrix x(numnode, numdim);

      std::vector<double> mydisp(lm.size());
      std::vector<double> myvelo(lm.size());
      std::vector<double> myoffprestr(lm.size());
      Core::FE::extract_my_values(*dispnp, mydisp, lm);
      Core::FE::extract_my_values(*velonp, myvelo, lm);
      Core::FE::extract_my_values(*offset_prestress, myoffprestr, lm);

      // set material configuration
      material_configuration(x);

      // --------------------------------------------------
      // Now do the nurbs specific stuff
      bool nurbsele = false;

      auto* nurbsdis = dynamic_cast<Core::FE::Nurbs::NurbsDiscretization*>(&(discretization));

      if (nurbsdis != nullptr) nurbsele = true;

      // knot vectors for parent volume and this surface
      std::vector<Core::LinAlg::SerialDenseVector> mypknots(3);
      std::vector<Core::LinAlg::SerialDenseVector> myknots(2);

      // NURBS control point weights for all nodes, ie. CPs
      Core::LinAlg::SerialDenseVector weights(numnode);

      if (nurbsele)
      {
        // --------------------------------------------------
        // get node weights for nurbs elements
        for (int inode = 0; inode < numnode; inode++)
        {
          auto* cp = dynamic_cast<Core::FE::Nurbs::ControlPoint*>(nodes()[inode]);
          weights(inode) = cp->w();
        }
      }
      // --------------------------------------------------


      std::vector<double> refnormal(lm.size());
      std::vector<double> mydisp_refnormal(lm.size());
      std::vector<double> myvelo_refnormal(lm.size());
      std::vector<double> myoffprestr_refnormal(lm.size());
      Core::LinAlg::SerialDenseMatrix N_otimes_N;
      N_otimes_N.shape(lm.size(), lm.size());

      if (rtype == refsurfnormal)
      {
        std::vector<double> dummy(lm.size());  // dummy vector - we only want the reference normals!
        build_normals_at_nodes(elevector2, dummy, true);

        // norm of nodal subvectors of element normal vector
        Core::LinAlg::SerialDenseVector norm_refnormal_sq;
        norm_refnormal_sq.size(numnode);
        for (int node = 0; node < numnode; ++node)
        {
          for (int dim = 0; dim < numdim; dim++)
            norm_refnormal_sq[node] +=
                elevector2[node * numdf + dim] * elevector2[node * numdf + dim];
        }

        // normalize nodal subvectors of element normal vector
        for (int node = 0; node < numnode; ++node)
          for (int dim = 0; dim < numdim; dim++)
            elevector2[node * numdf + dim] /= sqrt(norm_refnormal_sq[node]);

        // build nodal N \otimes N matrix
        for (int node = 0; node < numnode; ++node)
        {
          for (int dim1 = 0; dim1 < numdf; dim1++)
          {
            for (int dim2 = 0; dim2 < numdf; dim2++)
              N_otimes_N(node * numdf + dim1, node * numdf + dim2) =
                  elevector2[node * numdf + dim1] * elevector2[node * numdf + dim2];
          }
        }

        // (N \otimes N) disp, (N \otimes N) velo
        for (int node = 0; node < numnode; ++node)
        {
          for (int dim1 = 0; dim1 < numdim; dim1++)
          {
            refnormal[node * numdf + dim1] += elevector2[node * numdf + dim1];
            for (int dim2 = 0; dim2 < numdim; dim2++)
            {
              mydisp_refnormal[node * numdf + dim1] +=
                  N_otimes_N(node * numdf + dim1, node * numdf + dim2) *
                  (mydisp[node * numdf + dim2]);
              myvelo_refnormal[node * numdf + dim1] +=
                  N_otimes_N(node * numdf + dim1, node * numdf + dim2) *
                  myvelo[node * numdf + dim2];
              myoffprestr_refnormal[node * numdf + dim1] +=
                  N_otimes_N(node * numdf + dim1, node * numdf + dim2) *
                  myoffprestr[node * numdf + dim2];
            }
          }
        }
      }

      // allocate vector for shape functions and matrix for derivatives
      Core::LinAlg::SerialDenseVector funct(numnode);
      Core::LinAlg::SerialDenseMatrix deriv(2, numnode);

      /*----------------------------------------------------------------------*
      |               start loop over integration points                     |
      *----------------------------------------------------------------------*/
      const Core::FE::IntegrationPoints2D intpoints(gaussrule_);
      for (int gp = 0; gp < intpoints.nquad; gp++)
      {
        // set gausspoints from integration rule
        Core::LinAlg::SerialDenseVector e(2);
        e(0) = intpoints.qxg[gp][0];
        e(1) = intpoints.qxg[gp][1];

        // get shape functions and derivatives in the plane of the element
        if (!nurbsele)
        {
          Core::FE::shape_function_2d(funct, e(0), e(1), shape());
          Core::FE::shape_function_2d_deriv1(deriv, e(0), e(1), shape());
        }
        else
        {
          Core::FE::Nurbs::nurbs_get_2d_funct_deriv(
              funct, deriv, e, myknots, weights, Core::FE::CellType::nurbs9);
        }

        // check for correct input
        for (int checkdof = numdim; checkdof < int(onoff->size()); ++checkdof)
        {
          if ((*onoff)[checkdof] != 0)
            FOUR_C_THROW(
                "Number of dimensions in Robin evaluation is 3. Further DoFs are not considered.");
        }

        Core::LinAlg::SerialDenseMatrix dxyzdrs(2, 3);
        Core::LinAlg::multiply(dxyzdrs, deriv, x);
        Core::LinAlg::SerialDenseMatrix metrictensor(2, 2);
        Core::LinAlg::multiply_nt(metrictensor, dxyzdrs, dxyzdrs);
        const double detA =
            sqrt(metrictensor(0, 0) * metrictensor(1, 1) - metrictensor(0, 1) * metrictensor(1, 0));

        switch (rtype)
        {
          case xyz:
          {
            for (int dim = 0; dim < numdim; dim++)
            {
              if ((*onoff)[dim])  // is this dof activated?
              {
                // displacement and velocity at Gauss point
                double dispnp_gp = 0.0;
                double velonp_gp = 0.0;
                double offprestrn_gp = 0.0;
                for (int node = 0; node < numnode; ++node)
                {
                  dispnp_gp += funct[node] * mydisp[node * numdf + dim];
                  velonp_gp += funct[node] * myvelo[node * numdf + dim];
                  offprestrn_gp += funct[node] * myoffprestr[node * numdf + dim];
                }

                // displacement related forces and derivatives
                double force_disp;
                double force_disp_deriv;
                if ((*numfuncnonlinstiff)[dim] == 0)
                {
                  force_disp = springstiff[dim] * (dispnp_gp - disploffset[dim] + offprestrn_gp);
                  force_disp_deriv = springstiff[dim];
                }
                else
                {
                  double displ[3] = {std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::infinity()};
                  displ[dim] = dispnp_gp - disploffset[dim] + offprestrn_gp;
                  force_disp = Global::Problem::instance()
                                   ->function_by_id<Core::Utils::FunctionOfSpaceTime>(
                                       (*numfuncnonlinstiff)[dim] - 1)
                                   .evaluate(displ, time, 0);

                  force_disp_deriv = (Global::Problem::instance()
                                          ->function_by_id<Core::Utils::FunctionOfSpaceTime>(
                                              (*numfuncnonlinstiff)[dim] - 1)
                                          .evaluate_spatial_derivative(displ, time, 0))[dim];
                }

                // velocity related forces and derivatives
                const double force_vel = dashpotvisc[dim] * velonp_gp;
                const double force_vel_deriv = dashpotvisc[dim];

                // multiply integration factor
                const double fac = intpoints.qwgt[gp] * detA;
                const double force_disp_fac = force_disp * fac;
                const double force_disp_deriv_fac = force_disp_deriv * fac;
                const double force_vel_fac = force_vel * fac;
                const double force_vel_deriv_fac = force_vel_deriv * fac;

                // residual
                for (int node = 0; node < numnode; ++node)
                  elevector1[node * numdf + dim] += funct[node] * (force_disp_fac + force_vel_fac);

                // spring stress for output (const per element)
                for (int node = 0; node < numnode; ++node)
                  elevector3[node * numdf + dim] -= force_disp + force_vel;

                // stiffness matrix
                for (int node1 = 0; node1 < numnode; ++node1)
                  for (int node2 = 0; node2 < numnode; ++node2)
                    (elematrix1)(node1 * numdf + dim, node2 * numdf + dim) +=
                        funct[node1] * funct[node2] *
                        (force_disp_deriv_fac + force_vel_deriv_fac * time_fac);
              }
            }
          }
          break;

          case refsurfnormal:
          {
            if ((*onoff)[0] != 1) FOUR_C_THROW("refsurfnormal Robin condition on 1st dof only!");
            for (int checkdof = 1; checkdof < 3; ++checkdof)
              if ((*onoff)[checkdof] != 0)
                FOUR_C_THROW("refsurfnormal Robin condition on 1st dof only!");

            // all parameters are in dof No. 1
            for (int dim = 0; dim < numdim; dim++)
            {
              // displacement and velocity in normal direction at Gauss point
              double refnormal_gp = 0.0;
              double dispnp_refnormal_gp = 0.0;
              double velonp_refnormal_gp = 0.0;
              double offprestrn_refnormal_gp = 0.0;
              for (int node = 0; node < numnode; ++node)
              {
                refnormal_gp += funct[node] * refnormal[node * numdf + dim];
                dispnp_refnormal_gp += funct[node] * mydisp_refnormal[node * numdf + dim];
                velonp_refnormal_gp += funct[node] * myvelo_refnormal[node * numdf + dim];
                offprestrn_refnormal_gp += funct[node] * myoffprestr_refnormal[node * numdf + dim];
              }

              // displacement related forces and derivatives
              double force_disp;
              double force_disp_deriv;
              if ((*numfuncnonlinstiff)[0] == 0)
              {
                force_disp =
                    springstiff[0] * (dispnp_refnormal_gp + -disploffset[0] * refnormal_gp +
                                         offprestrn_refnormal_gp);
                force_disp_deriv = springstiff[0];
              }
              else
              {
                double displ[3] = {
                    dispnp_refnormal_gp + -disploffset[0] * refnormal_gp + offprestrn_refnormal_gp,
                    std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::infinity()};
                force_disp = Global::Problem::instance()
                                 ->function_by_id<Core::Utils::FunctionOfSpaceTime>(
                                     (*numfuncnonlinstiff)[0] - 1)
                                 .evaluate(displ, time, 0);

                force_disp_deriv = (Global::Problem::instance()
                                        ->function_by_id<Core::Utils::FunctionOfSpaceTime>(
                                            (*numfuncnonlinstiff)[0] - 1)
                                        .evaluate_spatial_derivative(displ, time, 0))[0];
              }

              // velocity related forces
              const double force_vel = dashpotvisc[0] * velonp_refnormal_gp;
              const double force_vel_deriv = dashpotvisc[0];

              // multiply integration factor
              const double fac = intpoints.qwgt[gp] * detA;
              const double force_disp_fac = force_disp * fac;
              const double force_disp_deriv_fac = force_disp_deriv * fac;
              const double force_vel_fac = force_vel * fac;
              const double force_vel_deriv_fac = force_vel_deriv * fac;


              // residual
              for (int node = 0; node < numnode; ++node)
                elevector1[node * numdf + dim] += funct[node] * (force_disp_fac + force_vel_fac);

              // spring stress for output (const per element)
              for (int node = 0; node < numnode; ++node)
                elevector3[node * numdf + dim] -= force_disp + force_vel;

              // stiffness matrix
              const int dim1 = dim;
              for (int dim2 = 0; dim2 < numdim; dim2++)
                for (int node1 = 0; node1 < numnode; ++node1)
                  for (int node2 = 0; node2 < numnode; ++node2)
                    (elematrix1)(node1 * numdf + dim1, node2 * numdf + dim2) +=
                        funct[node1] * funct[node2] *
                        (force_disp_deriv_fac + force_vel_deriv_fac * time_fac) *
                        N_otimes_N(node2 * numdf + dim1, node2 * numdf + dim2);
            }
          }
          break;

          case cursurfnormal:
          {
            FOUR_C_THROW(
                "cursurfnormal option not (yet) implemented in calc_struct_robinforcestiff "
                "routine!");
          }
          break;

          default:
            FOUR_C_THROW("Unknown type of Robin direction");
            break;
        }
      }
    }
    break;

    default:
      FOUR_C_THROW("Unimplemented type of action for StructuralSurface");
      break;
  }
  return 0;
}

/*----------------------------------------------------------------------*
 * Evaluate method for StructuralSurface-Elements               tk 10/07*
 * ---------------------------------------------------------------------*/
int Discret::Elements::StructuralSurface::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& elematrix1, Core::LinAlg::SerialDenseMatrix& elematrix2,
    Core::LinAlg::SerialDenseVector& elevector1, Core::LinAlg::SerialDenseVector& elevector2,
    Core::LinAlg::SerialDenseVector& elevector3)
{
  if (la.size() == 1)
  {
    return evaluate(params, discretization,
        la[0].lm_,  // location vector is build by the first column of la
        elematrix1, elematrix2, elevector1, elevector2, elevector3);
  }

  // start with "none"
  Discret::Elements::StructuralSurface::ActionType act = StructuralSurface::none;

  // get the required action
  std::string action = params.get<std::string>("action", "none");
  if (action == "none")
    FOUR_C_THROW("No action supplied");
  else if (action == "calc_struct_area_poro")
    act = StructuralSurface::calc_struct_area_poro;
  else if (action == "calc_cur_nodal_normals")
    act = StructuralSurface::calc_cur_nodal_normals;
  else if (action == "calc_ref_nodal_normals")
    act = StructuralSurface::calc_ref_nodal_normals;
  else
    FOUR_C_THROW("Unknown type of action for StructuralSurface");

  // what the element has to do
  switch (act)
  {
    case calc_struct_area_poro:
    {
      calculate_surface_porosity(params, discretization, la);
    }
    break;
    case calc_ref_nodal_normals:
    case calc_cur_nodal_normals:
    {
      evaluate(params, discretization,
          la[0].lm_,  // location vector is build by the first column of la
          elematrix1, elematrix2, elevector1, elevector2, elevector3);
    }
    break;
    default:
      FOUR_C_THROW("Unimplemented type of action for StructuralSurface");
      break;
  }
  return 0;
}

/*----------------------------------------------------------------------*
 * Compute Volume enclosed by surface.                          tk 10/07*
 * ---------------------------------------------------------------------*/
double Discret::Elements::StructuralSurface::compute_constr_vols(
    const Core::LinAlg::SerialDenseMatrix& xc, const int numnode)
{
  double V = 0.0;

  // Volume is calculated by evaluating the integral
  // 1/3*int_A(x dydz + y dxdz + z dxdy)

  // we compute the three volumes separately
  for (int indc = 0; indc < 3; indc++)
  {
    // split current configuration between "ab" and "c"
    // where a!=b!=c and a,b,c are in {x,y,z}
    Core::LinAlg::SerialDenseMatrix ab = xc;
    Core::LinAlg::SerialDenseVector c(numnode);
    for (int i = 0; i < numnode; i++)
    {
      ab(i, indc) = 0.0;   // project by z_i = 0.0
      c(i) = xc(i, indc);  // extract z coordinate
    }
    // index of variables a and b
    int inda = (indc + 1) % 3;
    int indb = (indc + 2) % 3;

    // get gaussrule
    const Core::FE::IntegrationPoints2D intpoints(gaussrule_);
    int ngp = intpoints.nquad;

    // allocate vector for shape functions and matrix for derivatives
    Core::LinAlg::SerialDenseVector funct(numnode);
    Core::LinAlg::SerialDenseMatrix deriv(2, numnode);

    /*----------------------------------------------------------------------*
     |               start loop over integration points                     |
     *----------------------------------------------------------------------*/
    for (int gpid = 0; gpid < ngp; ++gpid)
    {
      const double e0 = intpoints.qxg[gpid][0];
      const double e1 = intpoints.qxg[gpid][1];

      // get shape functions and derivatives of shape functions in the plane of
      // the element
      Core::FE::shape_function_2d(funct, e0, e1, shape());
      Core::FE::shape_function_2d_deriv1(deriv, e0, e1, shape());

      double detA;
      // compute "metric tensor" deriv*ab, which is a 2x3 matrix with zero indc'th
      // column
      Core::LinAlg::SerialDenseMatrix metrictensor(2, 3);
      Core::LinAlg::multiply(metrictensor, deriv, ab);
      // Core::LinAlg::SerialDenseMatrix metrictensor(2,2);
      // metrictensor.multiply('N','T',1.0,dxyzdrs,dxyzdrs,0.0);
      detA = metrictensor(0, inda) * metrictensor(1, indb) -
             metrictensor(0, indb) * metrictensor(1, inda);
      const double dotprodc = funct.dot(c);
      // add weighted volume at gausspoint
      V -= dotprodc * detA * intpoints.qwgt[gpid];
    }
  }
  return V / 3.0;
}

/*----------------------------------------------------------------------*
 * Compute volume and its first and second derivatives          tk 02/09*
 * with respect to the displacements                                    *
 * ---------------------------------------------------------------------*/
void Discret::Elements::StructuralSurface::compute_vol_deriv(
    const Core::LinAlg::SerialDenseMatrix& xc, const int numnode, const int ndof, double& V,
    Core::LinAlg::SerialDenseVector& Vdiff1,
    const std::shared_ptr<Core::LinAlg::SerialDenseMatrix>& Vdiff2, const int minindex,
    const int maxindex)
{
  // necessary constants
  const int numdim = 3;
  const double invnumind = 1.0 / (maxindex - minindex + 1.0);

  // initialize
  V = 0.0;
  Vdiff1.size(ndof);
  if (Vdiff2 != nullptr) Vdiff2->shape(ndof, ndof);

  // Volume is calculated by evaluating the integral
  // 1/3*int_A(x dydz + y dxdz + z dxdy)

  // we compute the three volumes separately
  for (int indc = minindex; indc < maxindex + 1; indc++)
  {
    // split current configuration between "ab" and "c"
    // where a!=b!=c and a,b,c are in {x,y,z}
    Core::LinAlg::SerialDenseMatrix ab = xc;
    Core::LinAlg::SerialDenseVector c(numnode);
    for (int i = 0; i < numnode; i++)
    {
      ab(i, indc) = 0.0;   // project by z_i = 0.0
      c(i) = xc(i, indc);  // extract z coordinate
    }
    // index of variables a and b
    int inda = (indc + 1) % 3;
    int indb = (indc + 2) % 3;

    // get gaussrule
    const Core::FE::IntegrationPoints2D intpoints(gaussrule_);
    int ngp = intpoints.nquad;

    // allocate vector for shape functions and matrix for derivatives
    Core::LinAlg::SerialDenseVector funct(numnode);
    Core::LinAlg::SerialDenseMatrix deriv(2, numnode);

    /*----------------------------------------------------------------------*
     |               start loop over integration points                     |
     *----------------------------------------------------------------------*/
    for (int gpid = 0; gpid < ngp; ++gpid)
    {
      const double e0 = intpoints.qxg[gpid][0];
      const double e1 = intpoints.qxg[gpid][1];

      // get shape functions and derivatives of shape functions in the plane of
      // the element
      Core::FE::shape_function_2d(funct, e0, e1, shape());
      Core::FE::shape_function_2d_deriv1(deriv, e0, e1, shape());

      // evaluate Jacobi determinant, for projected dA*
      std::vector<double> normal(numdim);
      double detA;
      // compute "metric tensor" deriv*xy, which is a 2x3 matrix with zero 3rd
      // column
      Core::LinAlg::SerialDenseMatrix metrictensor(2, numdim);
      Core::LinAlg::multiply(metrictensor, deriv, ab);
      // metrictensor.multiply('N','T',1.0,dxyzdrs,dxyzdrs,0.0);
      detA = metrictensor(0, inda) * metrictensor(1, indb) -
             metrictensor(0, indb) * metrictensor(1, inda);
      const double dotprodc = funct.dot(c);
      // add weighted volume at gausspoint
      V -= dotprodc * detA * intpoints.qwgt[gpid];

      //-------- compute first derivative
      for (int i = 0; i < numnode; i++)
      {
        (Vdiff1)[3 * i + inda] +=
            invnumind * intpoints.qwgt[gpid] * dotprodc *
            (deriv(0, i) * metrictensor(1, indb) - metrictensor(0, indb) * deriv(1, i));
        (Vdiff1)[3 * i + indb] +=
            invnumind * intpoints.qwgt[gpid] * dotprodc *
            (deriv(1, i) * metrictensor(0, inda) - metrictensor(1, inda) * deriv(0, i));
        (Vdiff1)[3 * i + indc] += invnumind * intpoints.qwgt[gpid] * funct[i] * detA;
      }

      //-------- compute second derivative
      if (Vdiff2 != nullptr)
      {
        for (int i = 0; i < numnode; i++)
        {
          for (int j = 0; j < numnode; j++)
          {
            //"diagonal" (dV)^2/(dx_i dx_j) = 0, therefore only six entries have
            // to be specified
            (*Vdiff2)(3 * i + inda, 3 * j + indb) +=
                invnumind * intpoints.qwgt[gpid] * dotprodc *
                (deriv(0, i) * deriv(1, j) - deriv(1, i) * deriv(0, j));
            (*Vdiff2)(3 * i + indb, 3 * j + inda) +=
                invnumind * intpoints.qwgt[gpid] * dotprodc *
                (deriv(0, j) * deriv(1, i) - deriv(1, j) * deriv(0, i));
            (*Vdiff2)(3 * i + inda, 3 * j + indc) +=
                invnumind * intpoints.qwgt[gpid] * funct[j] *
                (deriv(0, i) * metrictensor(1, indb) - metrictensor(0, indb) * deriv(1, i));
            (*Vdiff2)(3 * i + indc, 3 * j + inda) +=
                invnumind * intpoints.qwgt[gpid] * funct[i] *
                (deriv(0, j) * metrictensor(1, indb) - metrictensor(0, indb) * deriv(1, j));
            (*Vdiff2)(3 * i + indb, 3 * j + indc) +=
                invnumind * intpoints.qwgt[gpid] * funct[j] *
                (deriv(1, i) * metrictensor(0, inda) - metrictensor(1, inda) * deriv(0, i));
            (*Vdiff2)(3 * i + indc, 3 * j + indb) +=
                invnumind * intpoints.qwgt[gpid] * funct[i] *
                (deriv(1, j) * metrictensor(0, inda) - metrictensor(1, inda) * deriv(0, j));
          }
        }
      }
    }
  }
  V *= invnumind;
  return;
}


/*----------------------------------------------------------------------*
 * Compute surface area and its first and second derivatives    lw 05/08*
 * with respect to the displacements                                    *
 * ---------------------------------------------------------------------*/
void Discret::Elements::StructuralSurface::compute_area_deriv(
    const Core::LinAlg::SerialDenseMatrix& x, const int numnode, const int ndof, double& A,
    Core::LinAlg::SerialDenseVector& Adiff,
    const std::shared_ptr<Core::LinAlg::SerialDenseMatrix>& Adiff2)
{
  // initialization
  A = 0.;
  Adiff.size(ndof);

  if (Adiff2 != nullptr) Adiff2->shape(ndof, ndof);

  const Core::FE::IntegrationPoints2D intpoints(gaussrule_);

  int ngp = intpoints.nquad;

  // allocate vector for shape functions and matrix for derivatives
  Core::LinAlg::SerialDenseMatrix deriv(2, numnode);
  Core::LinAlg::SerialDenseMatrix dxyzdrs(2, 3);

  /*----------------------------------------------------------------------*
   |               start loop over integration points                     |
   *----------------------------------------------------------------------*/

  for (int gpid = 0; gpid < ngp; ++gpid)
  {
    const double e0 = intpoints.qxg[gpid][0];
    const double e1 = intpoints.qxg[gpid][1];

    // get derivatives of shape functions in the plane of the element
    Core::FE::shape_function_2d_deriv1(deriv, e0, e1, shape());

    std::vector<double> normal(3);
    double detA;
    surface_integration(detA, normal, x, deriv);
    A += detA * intpoints.qwgt[gpid];

    Core::LinAlg::SerialDenseMatrix ddet(3, ndof, true);
    Core::LinAlg::SerialDenseMatrix ddet2(3 * ndof, ndof, true);
    Core::LinAlg::SerialDenseVector jacobi_deriv(ndof, true);

    Core::LinAlg::multiply(dxyzdrs, deriv, x);

    /*--------------- derivation of minor determiants of the Jacobian
     *----------------------------- with respect to the displacements */
    for (int i = 0; i < numnode; ++i)
    {
      ddet(0, 3 * i) = 0.;
      ddet(0, 3 * i + 1) = deriv(0, i) * dxyzdrs(1, 2) - deriv(1, i) * dxyzdrs(0, 2);
      ddet(0, 3 * i + 2) = deriv(1, i) * dxyzdrs(0, 1) - deriv(0, i) * dxyzdrs(1, 1);

      ddet(1, 3 * i) = deriv(1, i) * dxyzdrs(0, 2) - deriv(0, i) * dxyzdrs(1, 2);
      ddet(1, 3 * i + 1) = 0.;
      ddet(1, 3 * i + 2) = deriv(0, i) * dxyzdrs(1, 0) - deriv(1, i) * dxyzdrs(0, 0);

      ddet(2, 3 * i) = deriv(0, i) * dxyzdrs(1, 1) - deriv(1, i) * dxyzdrs(0, 1);
      ddet(2, 3 * i + 1) = deriv(1, i) * dxyzdrs(0, 0) - deriv(0, i) * dxyzdrs(1, 0);
      ddet(2, 3 * i + 2) = 0.;

      jacobi_deriv(i * 3) = 1 / detA * (normal[2] * ddet(2, 3 * i) + normal[1] * ddet(1, 3 * i));
      jacobi_deriv(i * 3 + 1) =
          1 / detA * (normal[2] * ddet(2, 3 * i + 1) + normal[0] * ddet(0, 3 * i + 1));
      jacobi_deriv(i * 3 + 2) =
          1 / detA * (normal[0] * ddet(0, 3 * i + 2) + normal[1] * ddet(1, 3 * i + 2));
    }

    /*--- calculation of first derivatives of current interfacial area
     *----------------------------- with respect to the displacements */
    for (int i = 0; i < ndof; ++i)
    {
      (Adiff)[i] += jacobi_deriv(i) * intpoints.qwgt[gpid];
    }

    if (Adiff2 != nullptr)
    {
      /*--------- second derivates of minor determiants of the Jacobian
       *----------------------------- with respect to the displacements */
      for (int n = 0; n < numnode; ++n)
      {
        for (int o = 0; o < numnode; ++o)
        {
          ddet2(n * 3 + 1, o * 3 + 2) = deriv(0, n) * deriv(1, o) - deriv(1, n) * deriv(0, o);
          ddet2(n * 3 + 2, o * 3 + 1) = -ddet2(n * 3 + 1, o * 3 + 2);

          ddet2(ndof + n * 3, o * 3 + 2) = deriv(1, n) * deriv(0, o) - deriv(0, n) * deriv(1, o);
          ddet2(ndof + n * 3 + 2, o * 3) = -ddet2(ndof + n * 3, o * 3 + 2);

          ddet2(2 * ndof + n * 3, o * 3 + 1) = ddet2(n * 3 + 1, o * 3 + 2);
          ddet2(2 * ndof + n * 3 + 1, o * 3) = -ddet2(2 * ndof + n * 3, o * 3 + 1);
        }
      }

      /*- calculation of second derivatives of current interfacial areas
       *----------------------------- with respect to the displacements */
      for (int i = 0; i < ndof; ++i)
      {
        int var1, var2;

        if (i % 3 == 0)  // displacement in x-direction
        {
          var1 = 1;
          var2 = 2;
        }
        else if ((i - 1) % 3 == 0)  // displacement in y-direction
        {
          var1 = 0;
          var2 = 2;
        }
        else if ((i - 2) % 3 == 0)  // displacement in z-direction
        {
          var1 = 0;
          var2 = 1;
        }
        else
        {
          FOUR_C_THROW("calculation of second derivatives of interfacial area failed");
          exit(1);
        }

        for (int j = 0; j < ndof; ++j)
        {
          (*Adiff2)(i, j) +=
              (-1 / detA * jacobi_deriv(j) * jacobi_deriv(i) +
                  1 / detA *
                      (ddet(var1, i) * ddet(var1, j) + normal[var1] * ddet2(var1 * ndof + i, j) +
                          ddet(var2, i) * ddet(var2, j) +
                          normal[var2] * ddet2(var2 * ndof + i, j))) *
              intpoints.qwgt[gpid];
        }
      }
    }
  }

  return;
}


void Discret::Elements::StructuralSurface::build_normals_at_nodes(
    Core::LinAlg::SerialDenseVector& nodenormals, const std::vector<double>& mydisp, bool refconfig)
{
  const int numnode = num_node();
  const int numdim = 3;

  Core::LinAlg::SerialDenseMatrix x(numnode, 3);
  if (refconfig)
    material_configuration(x);
  else
  {
    spatial_configuration(x, mydisp);
  }

  for (int i = 0; i < numnode; ++i)
  {
    Core::LinAlg::Matrix<3, 1> loc_coor;
    loc_coor = Core::FE::get_node_coordinates(i, shape());

    const double e0 = loc_coor(0);
    const double e1 = loc_coor(1);

    // allocate vector for shape functions and matrix for derivatives
    Core::LinAlg::SerialDenseVector funct(numnode);
    Core::LinAlg::SerialDenseMatrix deriv(2, numnode);

    // get shape functions and derivatives in the plane of the element
    Core::FE::shape_function_2d(funct, e0, e1, shape());
    Core::FE::shape_function_2d_deriv1(deriv, e0, e1, shape());

    double detA;
    std::vector<double> normal(3);
    surface_integration(detA, normal, x, deriv);

    for (int j = 0; j < numdim; ++j)
    {
      nodenormals(numdim * i + j) = normal[j];
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::StructuralSurface::calculate_surface_porosity(
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la)
{
  // get the parent element
  Core::Elements::Element* parentele = parent_element();
  const int nenparent = parentele->num_node();
  // get element location vector and ownerships
  std::vector<int> lmpar;
  std::vector<int> lmowner;
  std::vector<int> lmstride;
  parentele->location_vector(discretization, lmpar, lmowner, lmstride);

  const Core::FE::IntegrationPoints2D intpoints(gaussrule_);
  const int ngp = intpoints.nquad;
  Core::LinAlg::SerialDenseVector poro(ngp);
  const int numdim = 3;
  const int numnode = num_node();
  const int noddof = num_dof_per_node(*(nodes()[0]));

  // element geometry update
  std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
      discretization.get_state("displacement");
  if (disp == nullptr) FOUR_C_THROW("Cannot get state vector 'displacement'");
  std::vector<double> mydisp(lmpar.size());
  Core::FE::extract_my_values(*disp, mydisp, lmpar);

  // update element geometry
  Core::LinAlg::SerialDenseMatrix xrefe(numdim, nenparent);  // material coord. of element
  Core::LinAlg::SerialDenseMatrix xcurr(numdim, nenparent);  // current  coord. of element

  Core::Nodes::Node** nodes = parentele->nodes();
  for (int i = 0; i < nenparent; ++i)
  {
    const auto& x = nodes[i]->x();
    xrefe(0, i) = x[0];
    xrefe(1, i) = x[1];
    xrefe(2, i) = x[2];

    xcurr(0, i) = xrefe(0, i) + mydisp[i * noddof + 0];
    xcurr(1, i) = xrefe(1, i) + mydisp[i * noddof + 1];
    xcurr(2, i) = xrefe(2, i) + mydisp[i * noddof + 2];
  }

  const int numdofpernode = 4;

  std::shared_ptr<const Core::LinAlg::Vector<double>> velnp =
      discretization.get_state(1, "fluidvel");
  if (velnp == nullptr) FOUR_C_THROW("Cannot get state vector 'fluidvel'");
  // extract local values of the global vectors
  std::vector<double> myvelpres(la[1].lm_.size());
  Core::FE::extract_my_values(*velnp, myvelpres, la[1].lm_);

  Core::LinAlg::SerialDenseVector mypres(numnode);
  for (int inode = 0; inode < numnode; ++inode)  // number of nodes
  {
    (mypres)(inode) = myvelpres[numdim + (inode * numdofpernode)];
  }

  // get coordinates of gauss points w.r.t. local parent coordinate system
  Core::LinAlg::SerialDenseMatrix pqxg(intpoints.nquad, 3);
  Core::LinAlg::SerialDenseMatrix derivtrafo(3, 3);

  Core::FE::surface_gp_to_parent_gp(
      pqxg, derivtrafo, intpoints, parentele->shape(), shape(), l_surf_number());

  std::shared_ptr<Mat::StructPoro> structmat =
      std::dynamic_pointer_cast<Mat::StructPoro>(parentele->material(1));

  for (int gp = 0; gp < ngp; ++gp)
  {
    // get shape functions and derivatives in the plane of the element
    Core::LinAlg::SerialDenseMatrix deriv(3, nenparent);
    Core::FE::shape_function_3d_deriv1(
        deriv, pqxg(gp, 0), pqxg(gp, 1), pqxg(gp, 2), parentele->shape());

    Core::LinAlg::SerialDenseVector funct2D(numnode);
    Core::FE::shape_function_2d(funct2D, intpoints.qxg[gp][0], intpoints.qxg[gp][1], shape());

    // pressure at integration point
    double press = funct2D.dot(mypres);

    // get Jacobian matrix and determinant w.r.t. spatial configuration
    //! transposed jacobian "dx/ds"
    Core::LinAlg::SerialDenseMatrix xjm(numdim, numdim);
    Core::LinAlg::multiply_nt(xjm, deriv, xcurr);
    Core::LinAlg::SerialDenseMatrix Jmat(numdim, numdim);
    Core::LinAlg::multiply_nt(Jmat, deriv, xrefe);

    double det = 0.0;
    double detJ = 0.0;

    if (numdim == 3)
    {
      det = xjm(0, 0) * (xjm(1, 1) * xjm(2, 2) - xjm(2, 1) * xjm(1, 2)) +
            xjm(0, 1) * (-xjm(1, 0) * xjm(2, 2) + xjm(2, 0) * xjm(1, 2)) +
            xjm(0, 2) * (xjm(1, 0) * xjm(2, 1) - xjm(2, 0) * xjm(1, 1));
      detJ = Jmat(0, 0) * (Jmat(1, 1) * Jmat(2, 2) - Jmat(2, 1) * Jmat(1, 2)) +
             Jmat(0, 1) * (-Jmat(1, 0) * Jmat(2, 2) + Jmat(2, 0) * Jmat(1, 2)) +
             Jmat(0, 2) * (Jmat(1, 0) * Jmat(2, 1) - Jmat(2, 0) * Jmat(1, 1));
    }
    else
      FOUR_C_THROW("not implemented");

    const double J = det / detJ;

    double porosity = 0.0;

    structmat->compute_surf_porosity(params, press, J, l_surf_number(), gp, porosity, nullptr,
        nullptr, nullptr, nullptr, nullptr, true);
  }
}

FOUR_C_NAMESPACE_CLOSE
