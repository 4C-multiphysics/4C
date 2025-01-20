// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_elements_paramsinterface.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_boundary_integration.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_nurbs_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_mat_structporo.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function.hpp"
#include "4C_w1.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  Integrate a Line Neumann boundary condition (public)      popp 06/13|
 *----------------------------------------------------------------------*/
int Discret::Elements::Wall1Line::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  // set the interface pointer in the parent element
  parent_element()->set_params_interface_ptr(params);
  // IMPORTANT: The 'neum_orthopressure' case represents a truly nonlinear follower-load
  // acting on the spatial configuration. Therefore, it needs to be linearized. On the
  // contrary, the simplified 'neum_pseudo_orthopressure' option allows for an approximative
  // modeling of an orthopressure load without the need to do any linearization. However,
  // this can only be achieved by referring the 'neum_pseudo_orthopressure' load to the last
  // converged configuration, which introduces an error as compared with 'neum_orthopressure'.
  bool loadlin = (elemat1 != nullptr);

  // get type of condition
  enum LoadType
  {
    neum_none,
    neum_live,                  // standard Neumann load
    neum_pseudo_orthopressure,  // pseudo-orthopressure load
    neum_orthopressure
    // orthopressure load
  };

  LoadType ltype = neum_none;
  const std::string& type = condition.parameters().get<std::string>("TYPE");
  if (type == "neum_live")
    ltype = neum_live;
  else if (type == "neum_pseudo_orthopressure")
    ltype = neum_pseudo_orthopressure;
  else if (type == "neum_orthopressure")
    ltype = neum_orthopressure;
  else
    FOUR_C_THROW("Unknown type of SurfaceNeumann condition");

  // get values and switches from the condition
  const auto onoff = condition.parameters().get<std::vector<int>>("ONOFF");
  const auto val = condition.parameters().get<std::vector<double>>("VAL");
  const auto funct = condition.parameters().get<std::vector<Core::IO::Noneable<int>>>("FUNCT");

  // check total time
  double time = -1.0;
  if (parent_element()->is_params_interface())
    time = parent_element()->params_interface_ptr()->get_total_time();
  else
    time = params.get("total time", -1.0);

  // set number of dofs per node
  const int noddof = num_dof_per_node(*nodes()[0]);

  // ensure that at least as many curves/functs as dofs are available
  if (int(onoff.size()) < noddof)
    FOUR_C_THROW("Fewer functions or curves defined than the element has dofs.");

  // set number of nodes
  const int numnod = num_node();
  const Core::FE::CellType distype = shape();

  // gaussian points
  const Core::FE::GaussRule1D gaussrule = get_optimal_gaussrule(distype);
  const Core::FE::IntegrationPoints1D intpoints(gaussrule);

  // allocate vector for shape functions and for derivatives
  Core::LinAlg::SerialDenseVector shapefcts(numnod);
  Core::LinAlg::SerialDenseMatrix deriv(1, numnod);

  // prepare element geometry 1
  // --> we always need the material configuration
  Core::LinAlg::SerialDenseMatrix xye(Wall1::numdim_, numnod);
  for (int i = 0; i < numnod; ++i)
  {
    xye(0, i) = nodes()[i]->x()[0];
    xye(1, i) = nodes()[i]->x()[1];
  }

  // prepare element geometry 2
  // --> depending on the type of Neumann condition, we might not need a spatial
  // configuration at all (standard Neumann), we might need the last converged
  // spatial position (pseudo-orthopressure) or the true current geometry (orthopressure).
  Core::LinAlg::SerialDenseMatrix xyecurr(Wall1::numdim_, numnod);

  // (1) standard Neumann --> we need only material configuration
  if (ltype == neum_live)
  {
    loadlin = false;  // no linearization needed for load in material configuration
  }

  // (2) pseudo orthopressure --> we need last converged configuration
  else if (ltype == neum_pseudo_orthopressure)
  {
    loadlin = false;  // no linearization needed for load in last converged configuration

    std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
        discretization.get_state("displacement");
    if (disp == nullptr) FOUR_C_THROW("Cannot get state vector 'displacement'");
    std::vector<double> mydisp(lm.size());
    Core::FE::extract_my_values(*disp, mydisp, lm);

    for (int i = 0; i < numnod; ++i)
    {
      xyecurr(0, i) = xye(0, i) + mydisp[i * noddof + 0];
      xyecurr(1, i) = xye(1, i) + mydisp[i * noddof + 1];
    }
  }

  // (3) true orthopressure --> we need spatial configuration
  else if (ltype == neum_orthopressure)
  {
    if (!loadlin)
      FOUR_C_THROW(
          "No linearization provided for orthopressure load (add 'LOADLIN yes' to input file)");

    std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
        discretization.get_state("displacement new");
    if (disp == nullptr) FOUR_C_THROW("Cannot get state vector 'displacement new'");
    std::vector<double> mydisp(lm.size());
    Core::FE::extract_my_values(*disp, mydisp, lm);

    for (int i = 0; i < numnod; ++i)
    {
      xyecurr(0, i) = xye(0, i) + mydisp[i * noddof + 0];
      xyecurr(1, i) = xye(1, i) + mydisp[i * noddof + 1];
    }
  }

  // loop over integration points //new
  for (int gpid = 0; gpid < intpoints.nquad; gpid++)
  {
    const double e1 = intpoints.qxg[gpid][0];

    // get shape functions and derivatives in the line
    if (distype == Core::FE::CellType::line2 || distype == Core::FE::CellType::line3)
    {
      Core::FE::shape_function_1d(shapefcts, e1, distype);
      Core::FE::shape_function_1d_deriv1(deriv, e1, distype);
    }
    else if (distype == Core::FE::CellType::nurbs2 || distype == Core::FE::CellType::nurbs3)
    {
      Core::FE::Nurbs::NurbsDiscretization* nurbsdis =
          dynamic_cast<Core::FE::Nurbs::NurbsDiscretization*>(&(discretization));

      std::shared_ptr<Core::FE::Nurbs::Knotvector> knots = (*nurbsdis).get_knot_vector();
      std::vector<Core::LinAlg::SerialDenseVector> parentknots(2);
      std::vector<Core::LinAlg::SerialDenseVector> boundknots(1);

      double normalfac = 0.0;
      bool zero_size = knots->get_boundary_ele_and_parent_knots(
          parentknots, boundknots, normalfac, parent_master_element()->id(), face_master_number());

      if (zero_size) return (0);

      Core::LinAlg::SerialDenseVector weights(num_node());
      for (int inode = 0; inode < num_node(); ++inode)
      {
        Core::FE::Nurbs::ControlPoint* cp =
            dynamic_cast<Core::FE::Nurbs::ControlPoint*>(nodes()[inode]);
        weights(inode) = cp->w();
      }

      Core::FE::Nurbs::nurbs_get_1d_funct_deriv(
          shapefcts, deriv, e1, boundknots[0], weights, distype);
    }
    else
      FOUR_C_THROW("Wrong distype!");

    switch (ltype)
    {
      case neum_live:
      {  // uniform load on reference configuration

        // compute infinitesimal line element dr for integration along the line
        const double dr = w1_substitution(xye, deriv, nullptr, numnod);

        double functfac = 1.0;

        // loop the dofs of a node
        for (int i = 0; i < noddof; ++i)
        {
          if (onoff[i])  // is this dof activated?
          {
            if (funct[i].has_value() && funct[i].value() > 0)
            {
              // factor given by spatial function
              const int functnum = funct[i].value();

              // calculate reference position of GP
              Core::LinAlg::SerialDenseMatrix gp_coord(1, Wall1::numdim_);
              gp_coord.multiply(Teuchos::TRANS, Teuchos::TRANS, 1.0, shapefcts, xye, 0.0);

              // write coordinates in another datatype
              double gp_coord2[3];  // the position vector has to be given in 3D!!!
              const int numdim = 2;
              for (int k = 0; k < numdim; k++) gp_coord2[k] = gp_coord(0, k);
              for (int k = numdim; k < 3;
                  k++)  // set a zero value for the remaining spatial directions
                gp_coord2[k] = 0.0;
              const double* coordgpref = gp_coord2;  // needed for function evaluation

              // evaluate function at current gauss point
              functfac = Global::Problem::instance()
                             ->function_by_id<Core::Utils::FunctionOfSpaceTime>(functnum - 1)
                             .evaluate(coordgpref, time, i);
            }
            else
              functfac = 1.0;

            const double fac = intpoints.qwgt[gpid] * dr * val[i] * functfac;
            for (int node = 0; node < numnod; ++node)
            {
              elevec1[node * noddof + i] += shapefcts[node] * fac;
            }
          }
        }
        break;
      }


      case neum_pseudo_orthopressure:  // pseudo-orthogonal pressure on last converged config.
      case neum_orthopressure:
      {  // orthogonal pressure (nonlinear load) on current config.

        // check for correct input
        if (onoff[0] != 1) FOUR_C_THROW("orthopressure on 1st dof only!");
        for (int checkdof = 1; checkdof < noddof; ++checkdof)
        {
          if (onoff[checkdof] != 0) FOUR_C_THROW("orthopressure on 1st dof only!");
        }
        double ortho_value = val[0];
        if (!ortho_value) FOUR_C_THROW("no orthopressure value given!");

        // outward normal vector (unit vector)
        std::vector<double> unrm(Wall1::numdim_);

        // compute infinitesimal line element dr for integration along the line
        const double dr = w1_substitution(xyecurr, deriv, &unrm, numnod);

        double functfac = 1.0;
        if (funct[0].has_value() && funct[0].value() > 0)
        {
          // factor given by spatial function
          const int functnum = funct[0].value();

          // calculate reference position of GP
          Core::LinAlg::SerialDenseMatrix gp_coord(1, Wall1::numdim_);
          gp_coord.multiply(Teuchos::TRANS, Teuchos::TRANS, 1.0, shapefcts, xye, 0.0);

          // write coordinates in another datatype
          double gp_coord2[3];  // the position vector has to be given in 3D!!!
          const int numdim = 2;
          for (int k = 0; k < numdim; k++) gp_coord2[k] = gp_coord(0, k);
          for (int k = numdim; k < 3; k++)  // set a zero value for the remaining spatial directions
            gp_coord2[k] = 0.0;
          const double* coordgpref = gp_coord2;  // needed for function evaluation

          // evaluate function at current gauss point
          functfac = Global::Problem::instance()
                         ->function_by_id<Core::Utils::FunctionOfSpaceTime>(functnum - 1)
                         .evaluate(coordgpref, time, 0);
        }

        // constant factor for integration
        const double fac = intpoints.qwgt[gpid] * ortho_value * functfac;

        // add load components
        for (int node = 0; node < numnod; ++node)
          for (int j = 0; j < noddof; ++j)
            elevec1[node * noddof + j] += shapefcts[node] * unrm[j] * dr * fac;

        // linearization if needed
        if (loadlin)
        {
          // total number of element DOFs
          int numdof = noddof * numnod;

          // directional derivative of surface
          Core::LinAlg::SerialDenseMatrix a_Dnormal(Wall1::numdim_, numdof);

          //******************************************************************
          // compute directional derivative
          //******************************************************************

          // linearization of basis vector
          Core::LinAlg::SerialDenseMatrix dg(Wall1::numdim_, numdof);
          for (int node = 0; node < numnod; ++node)
            for (int k = 0; k < Wall1::numdim_; ++k) dg(k, node * noddof + k) = deriv(0, node);

          // linearization of local surface normal vector
          for (int dof = 0; dof < numdof; ++dof)
          {
            a_Dnormal(0, dof) = dg(1, dof);
            a_Dnormal(1, dof) = -dg(0, dof);
          }

          // build surface element load linearization matrix
          // (CAREFUL: Minus sign due to the fact that external forces enter the global
          // residual vector with a minus sign, too! However, the load linaerization is
          // simply added to the global tangent stiffness matrix, thus we explicitly
          // need to set the minus sign here.)
          for (int node = 0; node < numnod; ++node)
            for (int dim = 0; dim < 2; dim++)
              for (int dof = 0; dof < elevec1.length(); dof++)
                (*elemat1)(node* noddof + dim, dof) -= shapefcts[node] * a_Dnormal(dim, dof) * fac;
        }

        break;
      }

      default:
      {
        FOUR_C_THROW("Unknown type of SurfaceNeumann load");
        break;
      }
    }
  }

  return 0;
}

Core::FE::GaussRule1D Discret::Elements::Wall1Line::get_optimal_gaussrule(
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
    case Core::FE::CellType::nurbs2:
      rule = Core::FE::GaussRule1D::line_2point;
      break;
    case Core::FE::CellType::nurbs3:
      rule = Core::FE::GaussRule1D::line_3point;
      break;
    default:
      FOUR_C_THROW("unknown number of nodes for gaussrule initialization");
      break;
  }
  return rule;
}

// determinant of jacobian matrix

double Discret::Elements::Wall1Line::w1_substitution(const Core::LinAlg::SerialDenseMatrix& xyze,
    const Core::LinAlg::SerialDenseMatrix& deriv,
    std::vector<double>* unrm,  // unit normal
    const int iel)
{
  /*
   |                                            0 1
   |                                           +-+-+
   |       0 1              0...iel-1          | | | 0
   |      +-+-+             +-+-+-+-+          +-+-+
   |      | | | 1     =     | | | | | 0        | | | .
   |      +-+-+             +-+-+-+-+       *  +-+-+ .
   |                                           | | | .
   |                                           +-+-+
   |                                           | | | iel-1
   |                                           +-+-+
   |
   |       dxyzdrs             deriv^T          xye^T
   |
   |
   |                       +-        -+
   |                        | dx   dy  |
   |      yields   dxydr =  | --   --  |
   |                        | dr   dr  |
   |                       +-        -+
   |
   */
  // compute derivative of parametrization
  double dr = 0.0;
  Core::LinAlg::SerialDenseMatrix der_par(1, 2);
  int err = Core::LinAlg::multiply_nt(der_par, deriv, xyze);
  if (err != 0) FOUR_C_THROW("Multiply failed");
  dr = sqrt(der_par(0, 0) * der_par(0, 0) + der_par(0, 1) * der_par(0, 1));
  if (unrm != nullptr)
  {
    (*unrm)[0] = 1 / dr * der_par(0, 1);
    (*unrm)[1] = -1 / dr * der_par(0, 0);
  }
  return dr;
}

/*======================================================================*/
int Discret::Elements::Wall1Line::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elematrix1, Core::LinAlg::SerialDenseMatrix& elematrix2,
    Core::LinAlg::SerialDenseVector& elevector1, Core::LinAlg::SerialDenseVector& elevector2,
    Core::LinAlg::SerialDenseVector& elevector3)
{
  const Core::FE::CellType distype = shape();

  // set number of dofs per node
  const int noddof = num_dof_per_node(*nodes()[0]);

  // start with "none"
  Discret::Elements::Wall1Line::ActionType act = Wall1Line::none;

  // get the required action
  std::string action = params.get<std::string>("action", "none");
  if (action == "none")
    FOUR_C_THROW("No action supplied");
  else if (action == "calc_struct_constrarea")
    act = Wall1Line::calc_struct_constrarea;
  else if (action == "calc_struct_centerdisp")
    act = Wall1Line::calc_struct_centerdisp;
  else if (action == "calc_struct_areaconstrstiff")
    act = Wall1Line::calc_struct_areaconstrstiff;
  else
    FOUR_C_THROW("Unknown type of action for Wall1_Line");
  // create communicator
  MPI_Comm Comm = discretization.get_comm();
  // what the element has to do
  switch (act)
  {
    // just compute the enclosed volume (e.g. for initialization)
    case calc_struct_constrarea:
    {
      if (distype != Core::FE::CellType::line2)
      {
        FOUR_C_THROW("Area Constraint only works for line2 curves!");
      }
      // We are not interested in volume of ghosted elements
      if (Core::Communication::my_mpi_rank(Comm) == owner())
      {
        // element geometry update
        std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
            discretization.get_state("displacement");
        if (disp == nullptr) FOUR_C_THROW("Cannot get state vector 'displacement'");
        std::vector<double> mydisp(lm.size());
        Core::FE::extract_my_values(*disp, mydisp, lm);
        const int numnod = num_node();
        Core::LinAlg::SerialDenseMatrix xsrefe(
            numnod, Wall1::numdim_);  // material coord. of element
        Core::LinAlg::SerialDenseMatrix xscurr(
            numnod, Wall1::numdim_);  // material coord. of element
        for (int i = 0; i < numnod; ++i)
        {
          xsrefe(i, 0) = nodes()[i]->x()[0];
          xsrefe(i, 1) = nodes()[i]->x()[1];

          xscurr(i, 0) = xsrefe(i, 0) + mydisp[i * noddof];
          xscurr(i, 1) = xsrefe(i, 1) + mydisp[i * noddof + 1];
        }
        // compute area between line and x-Axis
        double areaele = 0.5 * (xscurr(0, 1) + xscurr(1, 1)) * (xscurr(1, 0) - xscurr(0, 0));
        elevector3[0] = areaele;
      }
    }
    break;
    case calc_struct_centerdisp:
    {
      if (Core::Communication::my_mpi_rank(Comm) == owner())
      {
        // element geometry update
        std::shared_ptr<const Core::LinAlg::Vector<double>> disptotal =
            discretization.get_state("displacementtotal");
        if (disptotal == nullptr) FOUR_C_THROW("Cannot get state vector 'displacementtotal'");
        std::vector<double> mydisp(lm.size());
        Core::FE::extract_my_values(*disptotal, mydisp, lm);
        const int numnod = num_node();
        Core::LinAlg::SerialDenseMatrix xsrefe(
            Wall1::numdim_, numnod);  // material coord. of element
        Core::LinAlg::SerialDenseMatrix xscurr(
            Wall1::numdim_, numnod);  // current coord. of element
        for (int i = 0; i < numnod; ++i)
        {
          xsrefe(0, i) = nodes()[i]->x()[0];
          xsrefe(1, i) = nodes()[i]->x()[1];

          xscurr(0, i) = xsrefe(0, i) + mydisp[i * noddof];
          xscurr(1, i) = xsrefe(1, i) + mydisp[i * noddof + 1];
        }

        // integration of the displacements over the surface
        const int dim = Wall1::numdim_;
        const Core::FE::CellType distype = shape();

        // gaussian points
        const Core::FE::GaussRule1D gaussrule = get_optimal_gaussrule(distype);
        const Core::FE::IntegrationPoints1D intpoints(gaussrule);  //

        // allocate vector for shape functions and for derivatives
        Core::LinAlg::SerialDenseVector funct(numnod);
        Core::LinAlg::SerialDenseMatrix deriv(1, numnod);

        std::shared_ptr<const Core::LinAlg::Vector<double>> dispincr =
            discretization.get_state("displacementincr");
        std::vector<double> edispincr(lm.size());
        Core::FE::extract_my_values(*dispincr, edispincr, lm);

        elevector2[0] = 0;

        for (int gpid = 0; gpid < intpoints.nquad; gpid++)
        {
          const double e1 = intpoints.qxg[gpid][0];  // coordinate of GP

          // get values of shape functions and derivatives in the line at specific GP
          Core::FE::shape_function_1d(funct, e1, distype);
          Core::FE::shape_function_1d_deriv1(deriv, e1, distype);

          double dr = w1_substitution(xscurr, deriv, nullptr, numnod);

          elevector2[0] += intpoints.qwgt[gpid] * dr;

          for (int d = 0; d < dim; d++)
          {
            if (gpid == 0) elevector3[d] = 0;

            for (int j = 0; j < numnod; ++j)
            {
              elevector3[d] += funct[j] * intpoints.qwgt[gpid] * edispincr[j * dim + d] * dr;
            }
          }
        }
      }
    }
    break;

    case calc_struct_areaconstrstiff:
    {
      if (distype != Core::FE::CellType::line2)
      {
        FOUR_C_THROW("Area Constraint only works for line2 curves!");
      }  // element geometry update
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == nullptr)
      {
        FOUR_C_THROW("Cannot get state vector 'displacement'");
      }
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);
      const int numnod = num_node();
      Core::LinAlg::SerialDenseMatrix xsrefe(numnod, Wall1::numdim_);  // material coord. of element
      Core::LinAlg::SerialDenseMatrix xscurr(numnod, Wall1::numdim_);  // material coord. of element
      for (int i = 0; i < numnod; ++i)
      {
        xsrefe(i, 0) = nodes()[i]->x()[0];
        xsrefe(i, 1) = nodes()[i]->x()[1];

        xscurr(i, 0) = xsrefe(i, 0) + mydisp[i * noddof];
        xscurr(i, 1) = xsrefe(i, 1) + mydisp[i * noddof + 1];
      }
      // call submethods
      compute_area_constr_stiff(xscurr, elematrix1);
      compute_area_constr_deriv(xscurr, elevector1);
      elevector2 = elevector1;
      // compute area between line and x-Axis
      double areaele = 0.5 * (xscurr(0, 1) + xscurr(1, 1)) * (xscurr(1, 0) - xscurr(0, 0));
      elevector3[0] = areaele;
    }
    break;
    default:
      FOUR_C_THROW("Unimplemented type of action for Soh8Surface");
      break;
  }
  return 0;
}

/*----------------------------------------------------------------------*
 * Evaluate method on multiple dofsets                       vuong 11/12*
 * ---------------------------------------------------------------------*/
int Discret::Elements::Wall1Line::evaluate(Teuchos::ParameterList& params,
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

  const Core::FE::CellType distype = shape();

  // start with "none"
  Discret::Elements::Wall1Line::ActionType act = Wall1Line::none;

  // get the required action
  std::string action = params.get<std::string>("action", "none");
  if (action == "none")
    FOUR_C_THROW("No action supplied");
  else if (action == "calc_struct_area_poro")
    act = Wall1Line::calc_struct_area_poro;
  else
    FOUR_C_THROW("Unknown type of action for StructuralSurface");

  // what the element has to do
  switch (act)
  {
    case calc_struct_area_poro:
    {
      // get the parent element
      const Core::Elements::Element* parentele = parent_element();
      const int nenparent = parentele->num_node();
      // get element location vector and ownerships
      std::vector<int> lmpar;
      std::vector<int> lmowner;
      std::vector<int> lmstride;
      parentele->location_vector(discretization, lmpar, lmowner, lmstride);

      // gaussian points
      const Core::FE::GaussRule1D gaussrule = get_optimal_gaussrule(distype);
      // get integration rule
      const Core::FE::IntPointsAndWeights<1> intpoints(gaussrule);

      const int ngp = intpoints.ip().nquad;
      Core::LinAlg::SerialDenseVector poro(ngp);
      const int numdim = 2;
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

      const Core::Nodes::Node* const* nodes = parentele->nodes();
      for (int i = 0; i < nenparent; ++i)
      {
        const auto& x = nodes[i]->x();
        xrefe(0, i) = x[0];
        xrefe(1, i) = x[1];

        xcurr(0, i) = xrefe(0, i) + mydisp[i * noddof + 0];
        xcurr(1, i) = xrefe(1, i) + mydisp[i * noddof + 1];
      }

      // number of degrees of freedom per node of fluid
      const int numdofpernode = 3;

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

      Core::LinAlg::SerialDenseMatrix pqxg;
      Core::LinAlg::SerialDenseMatrix derivtrafo;

      Core::FE::boundary_gp_to_parent_gp<2>(
          pqxg, derivtrafo, intpoints, parentele->shape(), distype, face_parent_number());

      for (int gp = 0; gp < ngp; ++gp)
      {
        // get shape functions and derivatives in the plane of the element
        Core::LinAlg::SerialDenseVector funct(nenparent);
        Core::LinAlg::SerialDenseMatrix deriv(2, nenparent);
        Core::FE::shape_function_2d(funct, pqxg(gp, 0), pqxg(gp, 1), parentele->shape());
        Core::FE::shape_function_2d_deriv1(deriv, pqxg(gp, 0), pqxg(gp, 1), parentele->shape());

        Core::LinAlg::SerialDenseVector funct1D(numnode);
        Core::FE::shape_function_1d(funct1D, intpoints.ip().qxg[gp][0], shape());

        // pressure at integration point
        double press = funct1D.dot(mypres);

        // get Jacobian matrix and determinant w.r.t. spatial configuration
        //! transposed jacobian "dx/ds"
        Core::LinAlg::SerialDenseMatrix xjm(numdim, numdim);
        Core::LinAlg::multiply_nt(xjm, deriv, xcurr);
        Core::LinAlg::SerialDenseMatrix Jmat(numdim, numdim);
        Core::LinAlg::multiply_nt(Jmat, deriv, xrefe);

        double det = 0.0;
        double detJ = 0.0;

        if (numdim == 2)
        {
          det = xjm(0, 0) * xjm(1, 1) - xjm(0, 1) * xjm(1, 0);
          detJ = Jmat(0, 0) * Jmat(1, 1) - Jmat(0, 1) * Jmat(1, 0);
        }
        else
          FOUR_C_THROW("not implemented");

        const double J = det / detJ;

        // get structure material
        std::shared_ptr<Mat::StructPoro> structmat =
            std::static_pointer_cast<Mat::StructPoro>(parentele->material());
        if (structmat == nullptr) FOUR_C_THROW("invalid structure material for poroelasticity");
        double porosity = 0.0;
        structmat->compute_surf_porosity(params, press, J, face_parent_number(), gp, porosity);
      }
    }
    break;
    default:
      FOUR_C_THROW("Unimplemented type of action for Soh8Surface");
      break;
  }
  return 0;
}

/*----------------------------------------------------------------------*
 * Compute first derivatives of area                            tk 10/07*
 * with respect to the displacements                                    *
 * ---------------------------------------------------------------------*/
void Discret::Elements::Wall1Line::compute_area_constr_deriv(
    Core::LinAlg::SerialDenseMatrix xscurr, Core::LinAlg::SerialDenseVector& elevector)
{
  if (elevector.length() != 4)
  {
    std::cout << "Length of element Vector: " << elevector.length() << std::endl;
    FOUR_C_THROW("That is not the right size!");
  }
  // implementation of simple analytic solution
  elevector[0] = -xscurr(0, 1) - xscurr(1, 1);
  elevector[1] = xscurr(1, 0) - xscurr(0, 0);
  elevector[2] = xscurr(0, 1) + xscurr(1, 1);
  elevector[3] = xscurr(1, 0) - xscurr(0, 0);
  elevector.scale(-0.5);
  return;
}

/*----------------------------------------------------------------------*
 * Compute influence of area constraint on stiffness matrix.    tk 10/07*
 * Second derivatives of areas with respect to the displacements        *
 * ---------------------------------------------------------------------*/
void Discret::Elements::Wall1Line::compute_area_constr_stiff(
    Core::LinAlg::SerialDenseMatrix xscurr, Core::LinAlg::SerialDenseMatrix& elematrix)
{
  elematrix(0, 0) = 0.0;
  elematrix(0, 1) = -0.5;
  elematrix(0, 2) = 0.0;
  elematrix(0, 3) = -0.5;

  elematrix(1, 0) = -0.5;
  elematrix(1, 1) = 0.0;
  elematrix(1, 2) = 0.5;
  elematrix(1, 3) = 0.0;

  elematrix(2, 0) = 0.0;
  elematrix(2, 1) = 0.5;
  elematrix(2, 2) = 0.0;
  elematrix(2, 3) = 0.5;

  elematrix(3, 0) = -0.5;
  elematrix(3, 1) = 0.0;
  elematrix(3, 2) = 0.5;
  elematrix(3, 3) = 0.0;

  elematrix.scale(-1.0);
  return;
}

FOUR_C_NAMESPACE_CLOSE
