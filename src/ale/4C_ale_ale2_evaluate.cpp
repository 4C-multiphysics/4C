// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ale_ale2.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_general_utils_nurbs_shapefunctions.hpp"
#include "4C_fem_nurbs_discretization.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_mat_elasthyper.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_mat_stvenantkirchhoff.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
int Discret::Elements::Ale2::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  Discret::Elements::Ale2::ActionType act = Ale2::none;

  // get the action required
  std::string action = params.get<std::string>("action", "none");
  if (action == "none")
    FOUR_C_THROW("No action supplied");
  else if (action == "calc_ale_solid")
    act = Ale2::calc_ale_solid;
  else if (action == "calc_ale_solid_linear")
    act = Ale2::calc_ale_solid_linear;
  else if (action == "calc_ale_laplace_material")
    act = Ale2::calc_ale_laplace_material;
  else if (action == "calc_ale_laplace_spatial")
    act = Ale2::calc_ale_laplace_spatial;
  else if (action == "calc_ale_springs_material")
    act = Ale2::calc_ale_springs_material;
  else if (action == "calc_ale_springs_spatial")
    act = Ale2::calc_ale_springs_spatial;
  else if (action == "setup_material")
    act = Ale2::setup_material;
  else if (action == "calc_jacobian_determinant")
    act = Ale2::calc_det_jac;
  else
    FOUR_C_THROW("{} is an unknown type of action for Ale2", action);

  bool spatialconfiguration = true;
  if (params.isParameter("use spatial configuration"))
    spatialconfiguration = params.get<bool>("use spatial configuration");


  // get the material
  std::shared_ptr<Core::Mat::Material> mat = material();

  switch (act)
  {
    case calc_ale_solid:
    {
      std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp =
          discretization.get_state("dispnp");
      std::vector<double> my_dispnp = Core::FE::extract_values(*dispnp, lm);

      static_ke_nonlinear(lm, my_dispnp, &elemat1, &elevec1, params, true, false);

      break;
    }
    case calc_ale_solid_linear:
    {
      std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp =
          discretization.get_state("dispnp");
      std::vector<double> my_dispnp = Core::FE::extract_values(*dispnp, lm);

      static_ke_nonlinear(lm, my_dispnp, &elemat1, &elevec1, params, spatialconfiguration, true);

      break;
    }
    case calc_ale_laplace_material:
    {
      std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp =
          discretization.get_state("dispnp");
      std::vector<double> my_dispnp = Core::FE::extract_values(*dispnp, lm);
      static_ke_laplace(discretization, lm, &elemat1, elevec1, my_dispnp, spatialconfiguration);

      break;
    }
    case calc_ale_laplace_spatial:
    {
      std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp =
          discretization.get_state("dispnp");
      std::vector<double> my_dispnp = Core::FE::extract_values(*dispnp, lm);
      static_ke_laplace(discretization, lm, &elemat1, elevec1, my_dispnp, true);

      break;
    }
    case calc_ale_springs_material:
    {
      std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp =
          discretization.get_state("dispnp");  // get the displacements
      std::vector<double> my_dispnp = Core::FE::extract_values(*dispnp, lm);

      static_ke_spring(&elemat1, elevec1, my_dispnp, spatialconfiguration);

      break;
    }
    case calc_ale_springs_spatial:
    {
      std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp =
          discretization.get_state("dispnp");  // get the displacements
      std::vector<double> my_dispnp = Core::FE::extract_values(*dispnp, lm);

      static_ke_spring(&elemat1, elevec1, my_dispnp, true);

      break;
    }
    case setup_material:
    {
      // get material
      std::shared_ptr<Mat::So3Material> so3mat = std::dynamic_pointer_cast<Mat::So3Material>(mat);

      if (so3mat->material_type() != Core::Materials::m_elasthyper and
          so3mat->material_type() !=
              Core::Materials::m_stvenant)  // ToDo (mayr): allow only materials without history
      {
        FOUR_C_THROW(
            "Illegal material type for ALE. Only materials allowed that do "
            "not store Gauss point data and do not need additional data from the "
            "element line definition.");
      }

      if (so3mat->material_type() == Core::Materials::m_elasthyper)
      {
        so3mat = std::dynamic_pointer_cast<Mat::ElastHyper>(mat);
        so3mat->setup(0, {}, {});
      }
      break;  // no setup for St-Venant / classic_lin required
    }
    case calc_det_jac:
    {
      std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp =
          discretization.get_state("dispnp");
      std::vector<double> my_dispnp = Core::FE::extract_values(*dispnp, lm);

      compute_det_jac(elevec1, lm, my_dispnp);

      break;
    }
    default:
    {
      FOUR_C_THROW("Unknown type of action for Ale2");
      break;
    }
  }

  return 0;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
int Discret::Elements::Ale2::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, const Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  return 0;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Discret::Elements::Ale2::edge_geometry(int i, int j,
    const Core::LinAlg::SerialDenseMatrix& xyze, double* length, double* sin_alpha,
    double* cos_alpha)
{
  double delta_x, delta_y;
  /*---------------------------------------------- x- and y-difference ---*/
  delta_x = xyze(0, j) - xyze(0, i);
  delta_y = xyze(1, j) - xyze(1, i);
  /*------------------------------- determine distance between i and j ---*/
  *length = sqrt(delta_x * delta_x + delta_y * delta_y);
  if (*length < (1.0E-14)) FOUR_C_THROW("edge or diagonal of element has zero length");
  /*--------------------------------------- determine direction of i-j ---*/
  *sin_alpha = delta_y / *length;
  *cos_alpha = delta_x / *length;
  /*----------------------------------------------------------------------*/
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
double Discret::Elements::Ale2::ale2_area_tria(
    const Core::LinAlg::SerialDenseMatrix& xyze, int i, int j, int k)
{
  double a, b, c; /* geometrical values */
  double el_area; /* element area */
  /*----------------------------------------------------------------------*/

  a = (xyze(0, i) - xyze(0, j)) * (xyze(0, i) - xyze(0, j)) +
      (xyze(1, i) - xyze(1, j)) * (xyze(1, i) - xyze(1, j)); /* line i-j squared */
  b = (xyze(0, j) - xyze(0, k)) * (xyze(0, j) - xyze(0, k)) +
      (xyze(1, j) - xyze(1, k)) * (xyze(1, j) - xyze(1, k)); /* line j-k squared */
  c = (xyze(0, k) - xyze(0, i)) * (xyze(0, k) - xyze(0, i)) +
      (xyze(1, k) - xyze(1, i)) * (xyze(1, k) - xyze(1, i)); /* line k-i squared */
  el_area = 0.25 * sqrt(2.0 * a * b + 2.0 * b * c + 2.0 * c * a - a * a - b * b - c * c);
  return el_area;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Discret::Elements::Ale2::ale2_torsional(int i, int j, int k,
    const Core::LinAlg::SerialDenseMatrix& xyze, Core::LinAlg::SerialDenseMatrix* k_torsion)
{
  /*
                             k
                             *
                      / \
         y,v ^      l_ki /   \  l_jk
             |           /     \
        --->     i *-------* j
              x,u          l_ij
  */

  double x_ij, x_jk, x_ki; /* x-differences between nodes */
  double y_ij, y_jk, y_ki; /* y-differences between nodes */
  double l_ij, l_jk, l_ki; /* side lengths */
  double a_ij, a_jk, a_ki; /* auxiliary values same as in Farhat et al. */
  double b_ij, b_jk, b_ki; /*                  - " -                    */
  double area;             /* area of the triangle */


  Core::LinAlg::SerialDenseMatrix R(3, 6); /* rotation matrix same as in Farhat et al. */
  Core::LinAlg::SerialDenseMatrix C(3, 3); /* torsion stiffness matrix same as in Farhat et al. */
  Core::LinAlg::SerialDenseMatrix A(6, 3); /* auxiliary array of intermediate results */


  /*--------------------------------- determine basic geometric values ---*/
  x_ij = xyze(0, j) - xyze(0, i);
  x_jk = xyze(0, k) - xyze(0, j);
  x_ki = xyze(0, i) - xyze(0, k);
  y_ij = xyze(1, j) - xyze(1, i);
  y_jk = xyze(1, k) - xyze(1, j);
  y_ki = xyze(1, i) - xyze(1, k);

  l_ij = sqrt(x_ij * x_ij + y_ij * y_ij);
  l_jk = sqrt(x_jk * x_jk + y_jk * y_jk);
  l_ki = sqrt(x_ki * x_ki + y_ki * y_ki);

  /*----------------------------------------------- check edge lengths ---*/
  if (l_ij < (1.0E-14)) FOUR_C_THROW("edge or diagonal of element has zero length");
  if (l_jk < (1.0E-14)) FOUR_C_THROW("edge or diagonal of element has zero length");
  if (l_ki < (1.0E-14)) FOUR_C_THROW("edge or diagonal of element has zero length");

  /*-------------------------------------------- fill auxiliary values ---*/
  a_ij = x_ij / (l_ij * l_ij);
  a_jk = x_jk / (l_jk * l_jk);
  a_ki = x_ki / (l_ki * l_ki);
  b_ij = y_ij / (l_ij * l_ij);
  b_jk = y_jk / (l_jk * l_jk);
  b_ki = y_ki / (l_ki * l_ki);

  /*--------------------------------------------------- determine area ---*/
  area = ale2_area_tria(xyze, i, j, k);

  /*---------------------------------- determine torsional stiffnesses ---*/
  C(0, 0) = l_ij * l_ij * l_ki * l_ki / (4.0 * area * area);
  C(1, 1) = l_ij * l_ij * l_jk * l_jk / (4.0 * area * area);
  C(2, 2) = l_ki * l_ki * l_jk * l_jk / (4.0 * area * area);

  C(0, 1) = C(0, 2) = C(1, 0) = C(1, 2) = C(2, 0) = C(2, 1) = 0;

  /*--------------------------------------- fill transformation matrix ---*/
  R(0, 0) = -b_ki - b_ij;
  R(0, 1) = a_ij + a_ki;
  R(0, 2) = b_ij;
  R(0, 3) = -a_ij;
  R(0, 4) = b_ki;
  R(0, 5) = -a_ki;

  R(1, 0) = b_ij;
  R(1, 1) = -a_ij;
  R(1, 2) = -b_ij - b_jk;
  R(1, 3) = a_jk + a_ij;
  R(1, 4) = b_jk;
  R(1, 5) = -a_jk;

  R(2, 0) = b_ki;
  R(2, 1) = -a_ki;
  R(2, 2) = b_jk;
  R(2, 3) = -a_jk;
  R(2, 4) = -b_jk - b_ki;
  R(2, 5) = a_ki + a_jk;

  /*----------------------------------- perform matrix multiplications ---*/


  int err = Core::LinAlg::multiply_tn(A, R, C);  // A = R^t * C
  if (err != 0) FOUR_C_THROW("Multiply failed");
  err = Core::LinAlg::multiply(*k_torsion, A, R);  // stiff = A * R
  if (err != 0) FOUR_C_THROW("Multiply failed");
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Discret::Elements::Ale2::ale2_tors_spring_tri3(
    Core::LinAlg::SerialDenseMatrix* sys_mat, const Core::LinAlg::SerialDenseMatrix& xyze)
{
  int i, j; /* counters */

  Core::LinAlg::SerialDenseMatrix k_tria(6, 6);  // rotational stiffness matrix of one triangle

  /*-------------------------------- evaluate torsional stiffness part ---*/
  ale2_torsional(0, 1, 2, xyze, &k_tria);

  /*-------------------------- add everything to the element stiffness ---*/
  for (i = 0; i < 6; i++)
    for (j = 0; j < 6; j++) (*sys_mat)(i, j) += k_tria(i, j);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Discret::Elements::Ale2::ale2_tors_spring_quad4(
    Core::LinAlg::SerialDenseMatrix* sys_mat, const Core::LinAlg::SerialDenseMatrix& xyze)
{
  int i, j; /* counters */

  Core::LinAlg::SerialDenseMatrix k_tria(6, 6);  // rotational stiffness matrix of one triangle

  /*--- pass all nodes and determine the triangle defined by the node i and
  adjunct nodes... ---*/
  ale2_torsional(0, 1, 2, xyze, &k_tria);

  /*---------- ...sort everything into the element stiffness matrix... ---*/
  for (i = 0; i < 6; i++)
    for (j = 0; j < 6; j++) (*sys_mat)(i, j) += k_tria(i, j);

  /*--------------------------------- ...repeat for second triangle... ---*/
  ale2_torsional(1, 2, 3, xyze, &k_tria);
  for (i = 2; i < 8; i++)
    for (j = 2; j < 8; j++) (*sys_mat)(i, j) += k_tria(i - 2, j - 2);

  /*------------------------------------------ ...and for the third... ---*/
  ale2_torsional(2, 3, 0, xyze, &k_tria);
  for (i = 4; i < 8; i++)
    for (j = 4; j < 8; j++) (*sys_mat)(i, j) += k_tria(i - 4, j - 4);
  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++) (*sys_mat)(i, j) += k_tria(i + 4, j + 4);
  for (i = 4; i < 8; i++)
  {
    for (j = 0; j < 2; j++)
    {
      (*sys_mat)(i, j) += k_tria(i - 4, j + 4);
      (*sys_mat)(j, i) += k_tria(i - 4, j + 4);
    }
  }

  /*------------------------------- ...and eventually for a forth time ---*/
  ale2_torsional(3, 0, 1, xyze, &k_tria);
  for (i = 0; i < 4; i++)
    for (j = 0; j < 4; j++) (*sys_mat)(i, j) += k_tria(i + 2, j + 2);
  for (i = 6; i < 8; i++)
    for (j = 6; j < 8; j++) (*sys_mat)(i, j) += k_tria(i - 6, j - 6);
  for (i = 6; i < 8; i++)
  {
    for (j = 0; j < 4; j++)
    {
      (*sys_mat)(i, j) += k_tria(i - 6, j + 2);
      (*sys_mat)(j, i) += k_tria(i - 6, j + 2);
    }
  }
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Discret::Elements::Ale2::static_ke_spring(Core::LinAlg::SerialDenseMatrix* sys_mat,
    Core::LinAlg::SerialDenseVector& residual, std::vector<double>& displacements,
    const bool spatialconfiguration)
{
  const int iel = num_node();  // numnode of this element
  const Core::FE::CellType distype = this->shape();
  int numcnd;          // number of corner nodes
  int node_i, node_j;  // end nodes of actual spring
  double length;       // length of actual edge
  double sin, cos;     // direction of actual edge
  double factor;


  // number of corner nodes
  switch (distype)
  {
    case Core::FE::CellType::quad4:
    case Core::FE::CellType::quad8:
    case Core::FE::CellType::quad9:
      numcnd = 4;
      break;
    case Core::FE::CellType::tri3:
    case Core::FE::CellType::tri6:
      numcnd = 3;
      break;
    default:
      numcnd = 0;
      FOUR_C_THROW("distype unknown");
      break;
  }

  // Actual element coordinates
  Core::LinAlg::SerialDenseMatrix xyze(2, iel);

  for (int i = 0; i < iel; i++)
  {
    xyze(0, i) = nodes()[i]->x()[0];
    xyze(1, i) = nodes()[i]->x()[1];
  }

  // compute spatial configuration (if necessary)
  if (spatialconfiguration)
  {
    for (int i = 0; i < iel; i++)
    {
      xyze(0, i) += displacements[i * 2];
      xyze(1, i) += displacements[i * 2 + 1];
    }
  }

  // Linear springs from all corner nodes to all corner nodes
  // loop over all edges and diagonals of the element
  for (node_i = 0; node_i < numcnd; node_i++)
  {
    for (node_j = node_i + 1; node_j < numcnd; node_j++)
    {
      edge_geometry(node_i, node_j, xyze, &length, &sin, &cos);
      factor = 1.0 / length;
      // put values in 'element stiffness'
      (*sys_mat)(node_i * 2, node_i * 2) += cos * cos * factor;
      (*sys_mat)(node_i * 2 + 1, node_i * 2 + 1) += sin * sin * factor;
      (*sys_mat)(node_i * 2, node_i * 2 + 1) += sin * cos * factor;
      (*sys_mat)(node_i * 2 + 1, node_i * 2) += sin * cos * factor;

      (*sys_mat)(node_j * 2, node_j * 2) += cos * cos * factor;
      (*sys_mat)(node_j * 2 + 1, node_j * 2 + 1) += sin * sin * factor;
      (*sys_mat)(node_j * 2, node_j * 2 + 1) += sin * cos * factor;
      (*sys_mat)(node_j * 2 + 1, node_j * 2) += sin * cos * factor;

      (*sys_mat)(node_i * 2, node_j * 2) -= cos * cos * factor;
      (*sys_mat)(node_i * 2 + 1, node_j * 2 + 1) -= sin * sin * factor;
      (*sys_mat)(node_i * 2, node_j * 2 + 1) -= sin * cos * factor;
      (*sys_mat)(node_i * 2 + 1, node_j * 2) -= sin * cos * factor;

      (*sys_mat)(node_j * 2, node_i * 2) -= cos * cos * factor;
      (*sys_mat)(node_j * 2 + 1, node_i * 2 + 1) -= sin * sin * factor;
      (*sys_mat)(node_j * 2, node_i * 2 + 1) -= sin * cos * factor;
      (*sys_mat)(node_j * 2 + 1, node_i * 2) -= sin * cos * factor;
    }
  }

  // build in torsional springs
  // and put edge nodes on the middle of the respective edge
  switch (distype)
  {
    case Core::FE::CellType::quad8:
      (*sys_mat)(8, 8) = 1.0;
      (*sys_mat)(8, 0) = -0.5;
      (*sys_mat)(8, 2) = -0.5;
      (*sys_mat)(9, 9) = 1.0;
      (*sys_mat)(9, 1) = -0.5;
      (*sys_mat)(9, 3) = -0.5;
      (*sys_mat)(10, 10) = 1.0;
      (*sys_mat)(10, 2) = -0.5;
      (*sys_mat)(10, 4) = -0.5;
      (*sys_mat)(11, 11) = 1.0;
      (*sys_mat)(11, 3) = -0.5;
      (*sys_mat)(11, 5) = -0.5;
      (*sys_mat)(12, 12) = 1.0;
      (*sys_mat)(12, 4) = -0.5;
      (*sys_mat)(12, 6) = -0.5;
      (*sys_mat)(13, 13) = 1.0;
      (*sys_mat)(13, 5) = -0.5;
      (*sys_mat)(13, 7) = -0.5;
      (*sys_mat)(14, 14) = 1.0;
      (*sys_mat)(14, 6) = -0.5;
      (*sys_mat)(14, 0) = -0.5;
      (*sys_mat)(15, 15) = 1.0;
      (*sys_mat)(15, 7) = -0.5;
      (*sys_mat)(15, 1) = -0.5;
      ale2_tors_spring_quad4(sys_mat, xyze);
      break;
    case Core::FE::CellType::quad9:
      (*sys_mat)(8, 8) = 1.0;
      (*sys_mat)(8, 0) = -0.5;
      (*sys_mat)(8, 2) = -0.5;
      (*sys_mat)(9, 9) = 1.0;
      (*sys_mat)(9, 1) = -0.5;
      (*sys_mat)(9, 3) = -0.5;
      (*sys_mat)(10, 10) = 1.0;
      (*sys_mat)(10, 2) = -0.5;
      (*sys_mat)(10, 4) = -0.5;
      (*sys_mat)(11, 11) = 1.0;
      (*sys_mat)(11, 3) = -0.5;
      (*sys_mat)(11, 5) = -0.5;
      (*sys_mat)(12, 12) = 1.0;
      (*sys_mat)(12, 4) = -0.5;
      (*sys_mat)(12, 6) = -0.5;
      (*sys_mat)(13, 13) = 1.0;
      (*sys_mat)(13, 5) = -0.5;
      (*sys_mat)(13, 7) = -0.5;
      (*sys_mat)(14, 14) = 1.0;
      (*sys_mat)(14, 6) = -0.5;
      (*sys_mat)(14, 0) = -0.5;
      (*sys_mat)(15, 15) = 1.0;
      (*sys_mat)(15, 7) = -0.5;
      (*sys_mat)(15, 1) = -0.5;
      (*sys_mat)(16, 16) = 1.0;
      (*sys_mat)(16, 8) = -0.5;
      (*sys_mat)(16, 12) = -0.5;
      (*sys_mat)(17, 17) = 1.0;
      (*sys_mat)(17, 9) = -0.5;
      (*sys_mat)(17, 13) = -0.5;
      ale2_tors_spring_quad4(sys_mat, xyze);
      break;
    case Core::FE::CellType::quad4:
      ale2_tors_spring_quad4(sys_mat, xyze);
      break;
    case Core::FE::CellType::tri3:
      ale2_tors_spring_tri3(sys_mat, xyze);
      break;
    case Core::FE::CellType::tri6:
      (*sys_mat)(6, 6) = 1.0;
      (*sys_mat)(6, 0) = -0.5;
      (*sys_mat)(6, 2) = -0.5;
      (*sys_mat)(7, 7) = 1.0;
      (*sys_mat)(7, 1) = -0.5;
      (*sys_mat)(7, 3) = -0.5;
      (*sys_mat)(8, 8) = 1.0;
      (*sys_mat)(8, 2) = -0.5;
      (*sys_mat)(8, 4) = -0.5;
      (*sys_mat)(9, 9) = 1.0;
      (*sys_mat)(9, 3) = -0.5;
      (*sys_mat)(9, 5) = -0.5;
      (*sys_mat)(10, 10) = 1.0;
      (*sys_mat)(10, 4) = -0.5;
      (*sys_mat)(10, 0) = -0.5;
      (*sys_mat)(11, 11) = 1.0;
      (*sys_mat)(11, 5) = -0.5;
      (*sys_mat)(11, 1) = -0.5;
      ale2_tors_spring_tri3(sys_mat, xyze);
      break;
    default:
      FOUR_C_THROW("unknown distype in ale spring dynamic");
      break;
  }

  // compute residual vector
  residual.putScalar(0.0);
  for (int i = 0; i < 2 * iel; ++i)
    for (int j = 0; j < 2 * iel; ++j) residual[i] += (*sys_mat)(i, j) * displacements[j];

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Discret::Elements::Ale2::static_ke_nonlinear(const std::vector<int>& lm,
    const std::vector<double>& disp, Core::LinAlg::SerialDenseMatrix* stiffmatrix,
    Core::LinAlg::SerialDenseVector* force, Teuchos::ParameterList& params,
    const bool spatialconfiguration, const bool pseudolinear)
{
  const int numnode = num_node();
  const int numdf = 2;
  const int nd = numnode * numdf;

  const double* total_time =
      params.isParameter("total time") ? &params.get<double>("total time") : nullptr;
  const double* time_step_size =
      params.isParameter("delta time") ? &params.get<double>("delta time") : nullptr;

  // general arrays
  Core::LinAlg::SerialDenseVector funct(numnode);
  Core::LinAlg::SerialDenseMatrix deriv;
  deriv.shape(2, numnode);
  Core::LinAlg::SerialDenseMatrix xjm;
  xjm.shape(2, 2);
  Core::LinAlg::SerialDenseMatrix boplin;
  boplin.shape(4, 2 * numnode);
  Core::LinAlg::SerialDenseVector F;
  F.size(4);
  Core::LinAlg::SerialDenseVector strain;
  strain.size(4);
  double det;
  Core::LinAlg::SerialDenseMatrix xrefe(2, numnode);
  Core::LinAlg::SerialDenseMatrix xcure(2, numnode);
  const int numeps = 4;
  Core::LinAlg::SerialDenseMatrix b_cure;
  b_cure.shape(numeps, nd);
  Core::LinAlg::SerialDenseMatrix stress;
  stress.shape(4, 4);
  Core::LinAlg::SerialDenseMatrix C;
  C.shape(4, 4);


  /*------- get integration data ---------------------------------------- */
  const Core::FE::CellType distype = shape();

  // gaussian points
  const Core::FE::GaussRule2D gaussrule = get_optimal_gaussrule(distype);
  const Core::FE::IntegrationPoints2D intpoints(gaussrule);

  /*----------------------------------------------------- geometry update */
  for (int k = 0; k < numnode; ++k)
  {
    xrefe(0, k) = nodes()[k]->x()[0];
    xrefe(1, k) = nodes()[k]->x()[1];

    xcure(0, k) = xrefe(0, k);
    xcure(1, k) = xrefe(1, k);

    if (spatialconfiguration)
    {
      xcure(0, k) += disp[k * numdf + 0];
      xcure(1, k) += disp[k * numdf + 1];
    }
  }

  /*--------------------------------- get node weights for nurbs elements */
  Core::LinAlg::SerialDenseVector weights(numnode);
  if (distype == Core::FE::CellType::nurbs4 || distype == Core::FE::CellType::nurbs9)
  {
    for (int inode = 0; inode < numnode; ++inode)
    {
      Core::FE::Nurbs::ControlPoint* cp =
          dynamic_cast<Core::FE::Nurbs::ControlPoint*>(nodes()[inode]);

      weights(inode) = cp->w();
    }
  }


  /*=================================================== integration loops */
  for (int ip = 0; ip < intpoints.nquad; ++ip)
  {
    /*================================== gaussian point and weight at it */
    const double e1 = intpoints.qxg[ip][0];
    const double e2 = intpoints.qxg[ip][1];
    const double wgt = intpoints.qwgt[ip];

    // get values of shape functions and derivatives in the gausspoint
    if (distype != Core::FE::CellType::nurbs4 && distype != Core::FE::CellType::nurbs9)
    {
      // shape functions and their derivatives for polynomials
      Core::FE::shape_function_2d(funct, e1, e2, distype);
      Core::FE::shape_function_2d_deriv1(deriv, e1, e2, distype);
    }
    else
    {
      // nurbs version
      FOUR_C_THROW("Not implemented yet!");
    }

    /*--------------------------------------- compute jacobian Matrix */
    jacobian_matrix(xrefe, deriv, xjm, &det, numnode);

    /*------------------------------------ integration factor  -------*/
    const double fac = wgt * det;
    /*----------------------------------- calculate operator Blin  ---*/
    calc_b_op_lin(boplin, deriv, xjm, det, numnode);
    // cout.precision(16);
    /*------------ calculate defgrad F^u, Green-Lagrange-strain E^u --*/
    def_grad(F, strain, xrefe, xcure, boplin, numnode);


    /*-calculate defgrad F in matrix notation and Blin in current conf.*/
    b_op_lin_cure(b_cure, boplin, F, numeps, nd);

    Core::LinAlg::Tensor<double, 3> xi = {{e1, e2, 0.0}};
    Mat::EvaluationContext context{.total_time = total_time,
        .time_step_size = time_step_size,
        .xi = &xi,
        .ref_coords = nullptr};
    call_mat_geo_nonl(strain, stress, C, numeps, material(), params, context, ip);



    /*---------------------- geometric part of stiffness matrix kg ---*/
    if (stiffmatrix) kg(*stiffmatrix, boplin, stress, fac, nd, numeps);

    /*------------------ elastic+displacement stiffness matrix keu ---*/
    if (stiffmatrix) keu(*stiffmatrix, b_cure, C, fac, nd, numeps);

    /*--------------- nodal forces fi from integration of stresses ---*/
    if (not pseudolinear and force) fint(stress, b_cure, *force, fac, nd);


  }  // for (int ip=0; ip<totngp; ++ip)

  if (pseudolinear and force)
  {
    Core::LinAlg::SerialDenseVector displacements;
    displacements.resize(nd);
    for (int i = 0; i < nd; ++i) displacements(i) = disp[i];

    Core::LinAlg::multiply(*force, *stiffmatrix, displacements);
  }

  return;
}

///*----------------------------------------------------------------------------*/
///*----------------------------------------------------------------------------*/
// static void ale2_min_jaco(Core::FE::CellType distype,
//    Core::LinAlg::SerialDenseMatrix xyz, double *min_detF)
//{
//  double  detF[4]; // Jacobian determinant at nodes
//
//  switch (distype)
//  {
//    case Core::FE::CellType::quad4:
//    case Core::FE::CellType::quad8:
//    case Core::FE::CellType::quad9:
//      /*--------------------- evaluate Jacobian determinant at nodes ---*/
//      detF[0] = 0.25 * ( (xyz[0][0]-xyz[0][1]) * (xyz[1][0]-xyz[1][3])
//                       - (xyz[1][0]-xyz[1][1]) * (xyz[0][0]-xyz[0][3]) );
//      detF[1] = 0.25 * ( (xyz[0][0]-xyz[0][1]) * (xyz[1][1]-xyz[1][2])
//                       - (xyz[1][0]-xyz[1][1]) * (xyz[0][1]-xyz[0][2]) );
//      detF[2] = 0.25 * ( (xyz[0][3]-xyz[0][2]) * (xyz[1][1]-xyz[1][2])
//                       - (xyz[1][3]-xyz[1][2]) * (xyz[0][1]-xyz[0][2]) );
//      detF[3] = 0.25 * ( (xyz[0][3]-xyz[0][2]) * (xyz[1][0]-xyz[1][3])
//                       - (xyz[1][3]-xyz[1][2]) * (xyz[0][0]-xyz[0][3]) );
//
//      std::cout << "detF[0] = " << detF[0] << std::endl;
//      std::cout << "detF[1] = " << detF[1] << std::endl;
//      std::cout << "detF[2] = " << detF[2] << std::endl;
//      std::cout << "detF[3] = " << detF[3] << std::endl;
//
//      /*------------------------------------------------- check sign ---*/
//      if (detF[0] <= 0.0) FOUR_C_THROW("Negative JACOBIAN ");
//      if (detF[1] <= 0.0) FOUR_C_THROW("Negative JACOBIAN ");
//      if (detF[2] <= 0.0) FOUR_C_THROW("Negative JACOBIAN ");
//      if (detF[3] <= 0.0) FOUR_C_THROW("Negative JACOBIAN ");
//      /*-------------------------------------- look for the smallest ---*/
//      *min_detF = ( detF[0]  < detF[1]) ?  detF[0]  : detF[1];
//      *min_detF = (*min_detF < detF[2]) ? *min_detF : detF[2];
//      *min_detF = (*min_detF < detF[3]) ? *min_detF : detF[3];
//      /*----------------------------------------------------------------*/
//      break;
//    case Core::FE::CellType::tri3:
//      *min_detF = (-xyz[0][0]+xyz[0][1]) * (-xyz[1][0]+xyz[1][2])
//                - (-xyz[0][0]+xyz[0][2]) * (-xyz[1][0]+xyz[1][1]);
//      if (*min_detF <= 0.0) FOUR_C_THROW("Negative JACOBIAN ");
//      break;
//    default:
//      FOUR_C_THROW("minimal Jacobian determinant for this distype not implemented");
//      break;
//  }
//  return;
//}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Discret::Elements::Ale2::static_ke_laplace(Core::FE::Discretization& dis, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix* sys_mat, Core::LinAlg::SerialDenseVector& residual,
    std::vector<double>& displacements, const bool spatialconfiguration)
{
  //  FOUR_C_THROW("We don't know what is really done in the element evaluation"
  //      "of the Laplace smoothing strategy. Check this CAREFULLY before"
  //      "using it.");

  const int iel = num_node();
  const Core::FE::CellType distype = this->shape();

  Core::LinAlg::SerialDenseMatrix xyze(2, iel);

  // get node coordinates
  for (int i = 0; i < iel; i++)
  {
    xyze(0, i) = nodes()[i]->x()[0];
    xyze(1, i) = nodes()[i]->x()[1];
  }

  // update spatial configuration if necessary
  if (spatialconfiguration)
  {
    for (int i = 0; i < iel; i++)
    {
      xyze(0, i) += displacements[2 * i + 0];
      xyze(1, i) += displacements[2 * i + 1];
    }
  }

  // --------------------------------------------------
  // Now do the nurbs specific stuff
  std::vector<Core::LinAlg::SerialDenseVector> myknots(2);
  Core::LinAlg::SerialDenseVector weights(iel);

  if (distype == Core::FE::CellType::nurbs4 || distype == Core::FE::CellType::nurbs9)
  {
    Core::FE::Nurbs::NurbsDiscretization* nurbsdis =
        dynamic_cast<Core::FE::Nurbs::NurbsDiscretization*>(&(dis));

    (*((*nurbsdis).get_knot_vector())).get_ele_knots(myknots, id());

    for (int inode = 0; inode < iel; ++inode)
    {
      Core::FE::Nurbs::ControlPoint* cp =
          dynamic_cast<Core::FE::Nurbs::ControlPoint*>(nodes()[inode]);

      weights(inode) = cp->w();
    }
  }

  /*----------------------------------------- declaration of variables ---*/
  Core::LinAlg::SerialDenseVector funct(iel);
  Core::LinAlg::SerialDenseMatrix deriv(2, iel);
  Core::LinAlg::SerialDenseMatrix deriv_xy(2, iel);
  Core::LinAlg::SerialDenseMatrix xjm(2, 2);
  Core::LinAlg::SerialDenseMatrix xji(2, 2);

  // Gauss quadrature points
  const Core::FE::GaussRule2D gaussrule = get_optimal_gaussrule(distype);
  const Core::FE::IntegrationPoints2D intpoints(gaussrule);
  //  double min_detF = 0.0;         /* minimal Jacobian determinant   */
  //  ale2_min_jaco(Shape(),xyze,&min_detF);

  // integration loop
  for (int iquad = 0; iquad < intpoints.nquad; ++iquad)
  {
    // parameter coordinates in 1- and 2-direction at quadrature point 'iquad'
    const double e1 = intpoints.qxg[iquad][0];
    const double e2 = intpoints.qxg[iquad][1];

    // get values of shape functions and derivatives at the gausspoint
    if (distype != Core::FE::CellType::nurbs4 && distype != Core::FE::CellType::nurbs9)
    {
      // shape functions and their derivatives for polynomials
      Core::FE::shape_function_2d(funct, e1, e2, distype);
      Core::FE::shape_function_2d_deriv1(deriv, e1, e2, distype);
    }
    else
    {
      // nurbs version
      Core::LinAlg::SerialDenseVector gp(2);
      gp(0) = e1;
      gp(1) = e2;

      Core::FE::Nurbs::nurbs_get_2d_funct_deriv(funct, deriv, gp, myknots, weights, distype);
    }

    // compute jacobian matrix

    // determine jacobian at point r,s,t
    for (int i = 0; i < 2; ++i)
    {
      for (int j = 0; j < 2; ++j)
      {
        double inv_det = 0.;
        for (int l = 0; l < iel; ++l)
        {
          inv_det += deriv(i, l) * xyze(j, l);
        }
        xjm(i, j) = inv_det;
      }
    }

    // determinant of jacobian
    const double det = xjm(0, 0) * xjm(1, 1) - xjm(0, 1) * xjm(1, 0);
    const double fac = intpoints.qwgt[iquad] * det;

    // inverse of jacobian
    const double inv_det = 1.0 / det;
    xji(0, 0) = xjm(1, 1) * inv_det;
    xji(0, 1) = -xjm(0, 1) * inv_det;
    xji(1, 0) = -xjm(1, 0) * inv_det;
    xji(1, 1) = xjm(0, 0) * inv_det;

    for (int isd = 0; isd < 2; isd++)
      for (int jsd = 0; jsd < 2; jsd++)
        for (int inode = 0; inode < iel; inode++)
          deriv_xy(isd, inode) = xji(isd, jsd) * deriv(jsd, inode);

    /*------------------------- diffusivity depends on displacement ---*/
    const double k_diff = 1.0;  // 1.0/min_detF/min_detF;

    /*------------------------------- sort it into stiffness matrix ---*/
    for (int i = 0; i < iel; ++i)
    {
      for (int j = 0; j < iel; ++j)
      {
        (*sys_mat)(i * 2, j * 2) +=
            (deriv_xy(0, i) * deriv_xy(0, j) + deriv_xy(1, i) * deriv_xy(1, j)) * fac * k_diff;
        (*sys_mat)(i * 2 + 1, j * 2 + 1) +=
            (deriv_xy(0, i) * deriv_xy(0, j) + deriv_xy(1, i) * deriv_xy(1, j)) * fac * k_diff;
      }
    }
  }

  residual.putScalar(0.0);
  for (int i = 0; i < 2 * iel; ++i)
    for (int j = 0; j < 2 * iel; ++j) residual[i] += (*sys_mat)(i, j) * displacements[j];

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Discret::Elements::Ale2::calc_b_op_lin(Core::LinAlg::SerialDenseMatrix& boplin,
    Core::LinAlg::SerialDenseMatrix& deriv, Core::LinAlg::SerialDenseMatrix& xjm, double& det,
    const int iel)
{
  double inv_det;
  double xji[2][2];
  /*---------------------------------------------- inverse of jacobian ---*/
  inv_det = 1.0 / det;
  xji[0][0] = xjm(1, 1) * inv_det;
  xji[0][1] = -xjm(0, 1) * inv_det;
  xji[1][0] = -xjm(1, 0) * inv_det;
  xji[1][1] = xjm(0, 0) * inv_det;
  /*----------------------------- get operator boplin of global derivatives -*/
  /*-------------- some comments, so that even fluid people are able to
   understand this quickly :-)
   the Boplin looks like
       | Nk,x    0   |
       |   0    Nk,y |
       | Nk,y    0   |
       |  0     Nk,x |
  */
  for (int inode = 0; inode < iel; inode++)
  {
    int dnode = inode * 2;

    boplin(0, dnode + 0) = deriv(0, inode) * xji[0][0] + deriv(1, inode) * xji[0][1];
    boplin(1, dnode + 1) = deriv(0, inode) * xji[1][0] + deriv(1, inode) * xji[1][1];
    boplin(2, dnode + 0) = boplin(1, dnode + 1);
    boplin(3, dnode + 1) = boplin(0, dnode + 0);
  } /* end of loop over nodes */
  return;
}


/*-----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Discret::Elements::Ale2::jacobian_matrix(const Core::LinAlg::SerialDenseMatrix& xrefe,
    const Core::LinAlg::SerialDenseMatrix& deriv, Core::LinAlg::SerialDenseMatrix& xjm, double* det,
    const int iel)
{
  xjm.putScalar(0.0);

  for (int k = 0; k < iel; k++)
  {
    xjm(0, 0) += deriv(0, k) * xrefe(0, k);
    xjm(0, 1) += deriv(0, k) * xrefe(1, k);
    xjm(1, 0) += deriv(1, k) * xrefe(0, k);
    xjm(1, 1) += deriv(1, k) * xrefe(1, k);
  }

  /*------------------------------------------ determinant of jacobian ---*/
  *det = xjm(0, 0) * xjm(1, 1) - xjm(1, 0) * xjm(0, 1);

  if (*det < 0.0) FOUR_C_THROW("NEGATIVE JACOBIAN DETERMINANT {:8.5f} in ELEMENT {}\n", *det, id());
  /*----------------------------------------------------------------------*/

  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Ale2::def_grad(Core::LinAlg::SerialDenseVector& F,
    Core::LinAlg::SerialDenseVector& strain, const Core::LinAlg::SerialDenseMatrix& xrefe,
    const Core::LinAlg::SerialDenseMatrix& xcure, Core::LinAlg::SerialDenseMatrix& boplin,
    const int iel)
{
  /*------------------calculate defgrad --------- (Summenschleife->+=) ---*
  defgrad looks like:

        |  1 + Ux,X  |
        |  1 + Uy,Y  |
        |      Ux,Y  |
        |      Uy,X  |
  */

  F.putScalar(0.0);

  F[0] = 1;
  F[1] = 1;
  for (int inode = 0; inode < iel; inode++)
  {
    F[0] += boplin(0, 2 * inode) * (xcure(0, inode) - xrefe(0, inode));      // F_11
    F[1] += boplin(1, 2 * inode + 1) * (xcure(1, inode) - xrefe(1, inode));  // F_22
    F[2] += boplin(2, 2 * inode) * (xcure(0, inode) - xrefe(0, inode));      // F_12
    F[3] += boplin(3, 2 * inode + 1) * (xcure(1, inode) - xrefe(1, inode));  // F_21
  } /* end of loop over nodes */

  /*-----------------------calculate Green-Lagrange strain E -------------*/
  strain[0] = 0.5 * (F[0] * F[0] + F[3] * F[3] - 1.0);  // E_11
  strain[1] = 0.5 * (F[2] * F[2] + F[1] * F[1] - 1.0);  // E_22
  strain[2] = 0.5 * (F[0] * F[2] + F[3] * F[1]);        // E_12
  strain[3] = strain[2];                                // E_21

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Ale2::kg(Core::LinAlg::SerialDenseMatrix& estif,
    const Core::LinAlg::SerialDenseMatrix& boplin, const Core::LinAlg::SerialDenseMatrix& stress,
    const double fac, const int nd, const int numeps)
{
  /*---------------------------------------------- perform B^T * SIGMA * B*/
  for (int i = 0; i < nd; i++)
    for (int j = 0; j < nd; j++)
      for (int r = 0; r < numeps; r++)
        for (int m = 0; m < numeps; m++)
          estif(i, j) += boplin(r, i) * stress(r, m) * boplin(m, j) * fac;

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Ale2::keu(Core::LinAlg::SerialDenseMatrix& estif,
    const Core::LinAlg::SerialDenseMatrix& b_cure, const Core::LinAlg::SerialDenseMatrix& C,
    const double fac, const int nd, const int numeps)
{
  /*------------- perform B_cure^T * D * B_cure, whereas B_cure = F^T * B */
  for (int i = 0; i < nd; i++)
    for (int j = 0; j < nd; j++)
      for (int k = 0; k < numeps; k++)
        for (int m = 0; m < numeps; m++) estif(i, j) += b_cure(k, i) * C(k, m) * b_cure(m, j) * fac;

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Ale2::fint(const Core::LinAlg::SerialDenseMatrix& stress,
    const Core::LinAlg::SerialDenseMatrix& b_cure, Core::LinAlg::SerialDenseVector& intforce,
    const double fac, const int nd)

{
  Core::LinAlg::SerialDenseVector st;
  st.size(4);

  st[0] = fac * stress(0, 0);
  st[1] = fac * stress(1, 1);
  st[2] = fac * stress(0, 2);
  st[3] = fac * stress(0, 2);

  for (int i = 0; i < nd; i++)
    for (int j = 0; j < 4; j++) intforce[i] += b_cure(j, i) * st[j];

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Ale2::call_mat_geo_nonl(
    const Core::LinAlg::SerialDenseVector& strain,        ///< Green-Lagrange strain vector
    Core::LinAlg::SerialDenseMatrix& stress,              ///< stress vector
    Core::LinAlg::SerialDenseMatrix& C,                   ///< elasticity matrix
    const int numeps,                                     ///< number of strains
    std::shared_ptr<const Core::Mat::Material> material,  ///< the material data
    Teuchos::ParameterList& params,                       ///< element parameter list
    const Mat::EvaluationContext& context,                ///< context for material evaluation
    const int gp)
{
  /*--------------------------- call material law -> get tangent modulus--*/
  switch (material->material_type())
  {
    case Core::Materials::m_stvenant: /*----------------------- linear elastic ---*/
    {
      const Mat::StVenantKirchhoff* actmat =
          static_cast<const Mat::StVenantKirchhoff*>(material.get());
      double ym = actmat->youngs();
      double pv = actmat->poisson_ratio();

      /*----------- material-tangente - plane strain, rotational symmetry ---*/

      const double c1 = ym / (1.0 + pv);
      const double b1 = c1 * pv / (1.0 - 2.0 * pv);
      const double a1 = b1 + c1;

      C(0, 0) = a1;
      C(0, 1) = b1;
      C(0, 2) = 0.;
      C(0, 3) = 0.;

      C(1, 0) = b1;
      C(1, 1) = a1;
      C(1, 2) = 0.;
      C(1, 3) = 0.;

      C(2, 0) = 0.;
      C(2, 1) = 0.;
      C(2, 2) = c1 / 2.;
      C(2, 3) = c1 / 2.;

      C(3, 0) = 0.;
      C(3, 1) = 0.;
      C(3, 2) = c1 / 2;
      C(3, 3) = c1 / 2;


      /*-------------------------- evaluate 2.PK-stresses -------------------*/
      /*------------------ Summenschleife -> += (2.PK stored as vector) ------*/

      Core::LinAlg::SerialDenseVector svector;
      svector.size(3);

      for (int k = 0; k < 3; k++)
      {
        for (int i = 0; i < numeps; i++)
        {
          svector(k) += C(k, i) * strain(i);
        }
      }
      /*------------------ 2.PK stored as matrix -----------------------------*/
      stress(0, 0) = svector(0);
      stress(0, 2) = svector(2);
      stress(1, 1) = svector(1);
      stress(1, 3) = svector(2);
      stress(2, 0) = svector(2);
      stress(2, 2) = svector(1);
      stress(3, 1) = svector(2);
      stress(3, 3) = svector(0);


      break;
    }
    case Core::Materials::m_elasthyper:  // general hyperelastic material (bborn, 06/09)
    {
      material_response3d_plane(stress, C, strain, params, context, gp);
      break;
    }
    default:
    {
      FOUR_C_THROW("Invalid type of material law for wall element");
      break;
    }
  }  // switch(material->material_type())

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Discret::Elements::Ale2::material_response3d_plane(Core::LinAlg::SerialDenseMatrix& stress,
    Core::LinAlg::SerialDenseMatrix& C, const Core::LinAlg::SerialDenseVector& strain,
    Teuchos::ParameterList& params, const Mat::EvaluationContext& context, const int gp)
{
  // make 3d equivalent of Green-Lagrange strain
  Core::LinAlg::Matrix<6, 1> gl(Core::LinAlg::Initialization::uninitialized);
  green_lagrange_plane3d(strain, gl);

  // call 3d stress response
  Core::LinAlg::Matrix<6, 1> pk2(Core::LinAlg::Initialization::zero);   // must be zerofied!!!
  Core::LinAlg::Matrix<6, 6> cmat(Core::LinAlg::Initialization::zero);  // must be zerofied!!!
  material_response3d(&pk2, &cmat, &gl, params, context, gp);

  // we have plain strain

  // transform 2nd Piola--Kirchhoff stress back to 2d stress matrix
  stress.putScalar(0.0);                                               // zerofy
  stress(0, 0) = stress(3, 3) = pk2(0);                                // S_{11}
  stress(1, 1) = stress(2, 2) = pk2(1);                                // S_{22}
  stress(0, 2) = stress(1, 3) = stress(3, 1) = stress(2, 0) = pk2(3);  // S_{12}

  // transform elasticity matrix  back to 2d matrix
  C(0, 0) = cmat(0, 0);  // C_{1111}
  C(0, 1) = cmat(0, 1);  // C_{1122}
  C(0, 2) = cmat(0, 3);  // C_{1112}
  C(0, 3) = cmat(0, 3);  // C_{1112} = C_{1121}

  C(1, 0) = cmat(1, 0);  // C_{2211}
  C(1, 1) = cmat(1, 1);  // C_{2222}
  C(1, 2) = cmat(1, 3);  // C_{2212}
  C(1, 3) = cmat(1, 3);  // C_{2221} = C_{2212}

  C(2, 0) = cmat(3, 0);  // C_{1211}
  C(2, 1) = cmat(3, 1);  // C_{1222}
  C(2, 2) = cmat(3, 3);  // C_{1212}
  C(2, 3) = cmat(3, 3);  // C_{1221} = C_{1212}

  C(3, 0) = cmat(3, 0);  // C_{2111} = C_{1211}
  C(3, 1) = cmat(3, 1);  // C_{2122} = C_{1222}
  C(3, 2) = cmat(3, 3);  // C_{2112} = C_{1212}
  C(3, 3) = cmat(3, 3);  // C_{2121} = C_{1212}

  // leave this dump
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Discret::Elements::Ale2::material_response3d(Core::LinAlg::Matrix<6, 1>* stress,
    Core::LinAlg::Matrix<6, 6>* cmat, const Core::LinAlg::Matrix<6, 1>* glstrain,
    Teuchos::ParameterList& params, const Mat::EvaluationContext& context, const int gp)
{
  std::shared_ptr<Mat::So3Material> so3mat =
      std::dynamic_pointer_cast<Mat::So3Material>(material());
  if (so3mat == nullptr) FOUR_C_THROW("cast to So3Material failed!");

  Core::LinAlg::Matrix<6, 1> stress_like_glstrain;
  Core::LinAlg::Voigt::Strains::to_stress_like(*glstrain, stress_like_glstrain);

  Core::LinAlg::SymmetricTensor<double, 3, 3> stress_t{};
  Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> cmat_t{};
  so3mat->evaluate(nullptr,
      Core::LinAlg::make_symmetric_tensor_from_stress_like_voigt_matrix(stress_like_glstrain),
      params, context, stress_t, cmat_t, gp, id());

  *stress = Core::LinAlg::Matrix<6, 1>(Core::LinAlg::make_stress_like_voigt_view(stress_t));
  *cmat = Core::LinAlg::Matrix<6, 6>(Core::LinAlg::make_stress_like_voigt_view(cmat_t));
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void Discret::Elements::Ale2::green_lagrange_plane3d(
    const Core::LinAlg::SerialDenseVector& glplane, Core::LinAlg::Matrix<6, 1>& gl3d)
{
  gl3d(0) = glplane(0);               // E_{11}
  gl3d(1) = glplane(1);               // E_{22}
  gl3d(2) = 0.0;                      // E_{33}
  gl3d(3) = glplane(2) + glplane(3);  // 2*E_{12}=E_{12}+E_{21}
  gl3d(4) = 0.0;                      // 2*E_{23}
  gl3d(5) = 0.0;                      // 2*E_{31}

  return;
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void Discret::Elements::Ale2::b_op_lin_cure(Core::LinAlg::SerialDenseMatrix& b_cure,
    const Core::LinAlg::SerialDenseMatrix& boplin, const Core::LinAlg::SerialDenseVector& F,
    const int numeps, const int nd)
{
  Core::LinAlg::SerialDenseMatrix Fmatrix;
  Fmatrix.shape(4, 4);


  /*---------------------------write Vector F as a matrix Fmatrix*/

  Fmatrix(0, 0) = F[0];
  Fmatrix(0, 2) = 0.5 * F[2];
  Fmatrix(0, 3) = 0.5 * F[2];
  Fmatrix(1, 1) = F[1];
  Fmatrix(1, 2) = 0.5 * F[3];
  Fmatrix(1, 3) = 0.5 * F[3];
  Fmatrix(2, 1) = F[2];
  Fmatrix(2, 2) = 0.5 * F[0];
  Fmatrix(2, 3) = 0.5 * F[0];
  Fmatrix(3, 0) = F[3];
  Fmatrix(3, 2) = 0.5 * F[1];
  Fmatrix(3, 3) = 0.5 * F[1];

  /*-------------------------------------------------int_b_cure operator*/
  b_cure.putScalar(0.0);
  for (int i = 0; i < numeps; i++)
    for (int j = 0; j < nd; j++)
      for (int k = 0; k < numeps; k++) b_cure(i, j) += Fmatrix(k, i) * boplin(k, j);
  /*----------------------------------------------------------------*/

  return;
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void Discret::Elements::Ale2::compute_det_jac(Core::LinAlg::SerialDenseVector& elevec1,
    const std::vector<int>& lm, const std::vector<double>& disp)
{
  const int numnode = num_node();
  const int numdf = 2;

  // general arrays
  Core::LinAlg::SerialDenseVector funct(numnode);
  Core::LinAlg::SerialDenseMatrix deriv;
  deriv.shape(2, numnode);
  Core::LinAlg::SerialDenseMatrix xjm;
  xjm.shape(2, 2);
  double det;
  double qm = 0.0;
  Core::LinAlg::SerialDenseMatrix xrefe(2, numnode);
  Core::LinAlg::SerialDenseMatrix xcure(2, numnode);

  Core::LinAlg::SerialDenseVector detjac(4);
  Core::LinAlg::SerialDenseVector quality(4);

  /*------- get integration data ---------------------------------------- */
  const Core::FE::CellType distype = shape();
  if (distype != Core::FE::CellType::quad4)
    FOUR_C_THROW("Quality metric is currently implemented for Quad4 elements, only.");

  /*----------------------------------------------------- geometry update */
  for (int k = 0; k < numnode; ++k)
  {
    xrefe(0, k) = nodes()[k]->x()[0];
    xrefe(1, k) = nodes()[k]->x()[1];

    // We always evaluate the current configuration
    xcure(0, k) = xrefe(0, k) += disp[k * numdf + 0];
    xcure(1, k) = xrefe(1, k) += disp[k * numdf + 1];
  }

  // array with x- and y-coordinates of nodes in parameter space
  double nodepos[4][2];
  nodepos[0][0] = -1.0;
  nodepos[0][1] = -1.0;
  nodepos[1][0] = 1.0;
  nodepos[1][1] = -1.0;
  nodepos[2][0] = 1.0;
  nodepos[2][1] = 1.0;
  nodepos[3][0] = -1.0;
  nodepos[3][1] = 1.0;

  /*------------- Loop over all nodes -----------------------------------*/
  for (int node = 0; node < 4; ++node)
  {
    /*================================== gaussian point and weight at it */
    const double e1 = nodepos[node][0];
    const double e2 = nodepos[node][1];

    // get values of shape functions and derivatives in the gausspoint
    // shape functions and their derivatives for polynomials
    Core::FE::shape_function_2d(funct, e1, e2, distype);
    Core::FE::shape_function_2d_deriv1(deriv, e1, e2, distype);

    /*--------------------------------------- compute jacobian Matrix */
    jacobian_matrix(xrefe, deriv, xjm, &det, numnode);

    /*---------------------------- Evaluate quality measure */
    evaluate_oddy(xjm, det, qm);


    detjac[node] = det;
    quality[node] = qm;
  }  // loop over nodes

  // assign results
  double mindetjac = detjac[0];
  double minqm = quality[0];
  for (int i = 1; i < 4; ++i)
  {
    if (detjac[i] < mindetjac) mindetjac = detjac[i];
    if (quality[i] < minqm) minqm = quality[i];
  }

  elevec1[0] = mindetjac;
  elevec1[1] = minqm;

  return;
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void Discret::Elements::Ale2::evaluate_oddy(
    const Core::LinAlg::SerialDenseMatrix& xjm, double det, double& qm)
{
  // compute C
  Core::LinAlg::SerialDenseMatrix c(2, 2, true);
  for (int i = 0; i < 2; ++i)
  {
    for (int j = 0; j < 2; ++j)
    {
      for (int k = 0; k < 2; ++k) c(i, j) += xjm(i, k) * xjm(j, k);
      c(i, j) /= det;
    }
  }

  // compute D
  double d = 0.0;
  for (int i = 0; i < 2; ++i)
  {
    for (int j = 0; j < 2; ++j)
    {
      d += c(i, j) * c(i, j);
    }
  }

  for (int k = 0; k < 2; ++k) d -= 0.5 * c(k, k);

  qm = d;
}

FOUR_C_NAMESPACE_CLOSE
