// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cut_boundarycell.hpp"

#include "4C_cut_cycle.hpp"
#include "4C_cut_element.hpp"
#include "4C_cut_output.hpp"
#include "4C_cut_side.hpp"
#include "4C_cut_volumecell.hpp"
#include "4C_fem_geometry_element_volume.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Cut::BoundaryCell::BoundaryCell(const Core::LinAlg::SerialDenseMatrix& xyz, Facet* facet,
    const std::vector<Point*>& points, int cubature_degree)
    : facet_(facet), points_(std::make_shared<Cycle>(points)), cubature_degree_(cubature_degree)
{
  xyz_.shape(3, xyz.numCols());

  /* This is necessary, because it is possible, that the given
   * LinAlg::SerialDenseMatrix has a total row number smaller than 3 (equal the
   * actual problem dimension).                              hiermeier 11/16 */
  for (unsigned c = 0; c < static_cast<unsigned>(xyz.numCols()); ++c)
  {
    std::copy(&xyz(0, c), &xyz(0, c) + xyz.numRows(), &xyz_(0, c));
    std::fill(&xyz_(0, c) + xyz.numRows(), &xyz_(0, c) + 3, 0.0);
  }

  // Assign reference position of boundary cell
  xyz_ref_ = xyz_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::BoundaryCell::clear() { points_->clear(); }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::BoundaryCell::is_valid() const { return points_->size() > 0; }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int Cut::BoundaryCell::get_global_boundary_cell_id() { return facet_->side_id(); }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <Core::FE::CellType celldistype>
void Cut::BoundaryCell::transform_local_coords(Element* elem1,
    const Core::LinAlg::Matrix<2, 1>& eta, Core::LinAlg::Matrix<3, 1>& x_gp_lin,
    Core::LinAlg::Matrix<3, 1>& normal, double& drs, bool shadow)
{
  // TEUCHOS_FUNC_TIME_MONITOR( "Cut::BoundaryCell::transform_local_coords" );


  const int numnodes = Core::FE::num_nodes<celldistype>;
  Core::LinAlg::Matrix<3, numnodes> xyzeGlo(this->xyz_, true), xyze;

  for (int i = 0; i < numnodes; i++)
  {
    Core::LinAlg::Matrix<3, 1> glo(&xyzeGlo(0, i)), loc;

    // map w.r to linear shadow element
    if (not shadow)
    {
      elem1->local_coordinates(glo, loc);
    }
    // map w.r to parent quad element
    else
    {
      if (not elem1->is_shadow())
        elem1->local_coordinates(glo, loc);
      else
        elem1->local_coordinates_quad(glo, loc);
    }
    xyze(0, i) = loc(0);
    xyze(1, i) = loc(1);
    xyze(2, i) = loc(2);
  }
  Core::LinAlg::Matrix<numnodes, 1> funct;
  Core::LinAlg::Matrix<2, numnodes> deriv;
  Core::LinAlg::Matrix<2, 2> metrictensor;

  Core::FE::shape_function_2d(funct, eta(0), eta(1), celldistype);
  Core::FE::shape_function_2d_deriv1(deriv, eta(0), eta(1), celldistype);
  Core::FE::compute_metric_tensor_for_boundary_ele<celldistype>(
      xyze, deriv, metrictensor, drs, &normal);

  x_gp_lin.multiply(xyze, funct);

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const std::vector<Cut::Point*>& Cut::BoundaryCell::points() const { return (*points_)(); }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Cut::Line2BoundaryCell::area() { return Core::Geo::element_area(shape(), xyz_); }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Cut::Tri3BoundaryCell::area()
{
  const int numnodes = Core::FE::num_nodes<Core::FE::CellType::tri3>;

  const std::vector<Point*> points = this->points();

  // create planes consisting of 3 nodes each
  Core::LinAlg::Matrix<numnodes, 1> p0(points[0]->x());
  Core::LinAlg::Matrix<numnodes, 1> p1(points[1]->x());
  Core::LinAlg::Matrix<numnodes, 1> p2(points[2]->x());

  Core::LinAlg::Matrix<numnodes, 1> v01;
  Core::LinAlg::Matrix<numnodes, 1> v02;
  Core::LinAlg::Matrix<numnodes, 1> v12;

  v01.update(1, p1, -1, p0, 0);
  v02.update(1, p2, -1, p0, 0);
  v12.update(1, p1, -1, p2, 0);


  double a = v01.norm2();
  double b = v02.norm2();
  double c = v12.norm2();

  // Herons formula:
  //    sqrt((a+(b+c))(c-(a-b))(c+(a-b))(a+(b-c)))/4
  //    double s=0.5*(v01.norm2() + v02.norm2() + v12.norm2());
  //    double area = sqrt( s*(s-v01.norm2())*(s-v02.norm2())*(s-v12.norm2()) );

  double areasqr = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c));
  if (areasqr < 0.0)
  {
    if (std::abs(areasqr) > AREA_BC_TOL)
    {
      std::ofstream file("invalid_boundary_cell.pos");
      // Write BCs for outside VolumeCell:
      Cut::Output::gmsh_new_section(file, "BoundaryCells");
      dump_gmsh(file);
      Cut::Output::gmsh_end_section(file, false);
      Cut::Output::gmsh_new_section(file, "BoundaryCellsNormal");
      dump_gmsh_normal(file);
      Cut::Output::gmsh_end_section(file);
      FOUR_C_THROW("Boundary Cell not valid! Written GMSH output in Invalid_boundary_cell.pos!");
    }
    else
    {
#if EXTENDED_CUT_DEBUG_OUTPUT
      std::cout << "NOTE: Boundary cell area calculated with Heron's formula is negative"
                << std::endl;
#endif
      // TODO: What should we really do here?
      // For now we create really small boundary cells
      areasqr *= -1;
    }
  }
  double area = 0.25 * sqrt(areasqr);

  // Alternative options:
  //  - cross-product (see cut_tetmesh.cpp in is_valid_tet())

  return area;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::BoundaryCell::dump_gmsh(std::ofstream& file, int* value)
{
  int default_value = facet_->side_id();
  if (not value) value = &default_value;

  Output::gmsh_cell_dump(file, shape(), xyz_, nullptr, value);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Point1BoundaryCell::dump_gmsh_normal(std::ofstream& file)
{
  // there is no normal for one point
  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Tri3BoundaryCell::dump_gmsh_normal(std::ofstream& file)
{
  file.precision(16);

  file << "VP(";
  Core::LinAlg::Matrix<3, 1> midpoint_triag(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; ++i)
  {
    Core::LinAlg::Matrix<3, 1> cur(Core::LinAlg::Initialization::zero);
    cur(0, 0) = xyz_(0, i);
    cur(1, 0) = xyz_(1, i);
    cur(2, 0) = xyz_(2, i);
    midpoint_triag.update(1.0, cur, 1.0);
  }

  midpoint_triag.scale(1.0 / 3.0);
  file << midpoint_triag(0, 0) << "," << midpoint_triag(1, 0) << "," << midpoint_triag(2, 0);
  file << "){";

  // Choose midpoint of triangle as normal for now. Not best choice possibly.
  Core::LinAlg::Matrix<2, 1> eta;
  eta(0, 0) = 0.5;
  eta(1, 0) = 0.5;
  Core::LinAlg::Matrix<3, 1> normal;
  this->normal(eta, normal);
  file << normal(0) << "," << normal(1) << "," << normal(2);

  file << "};\n";
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Quad4BoundaryCell::dump_gmsh_normal(std::ofstream& file)
{
  file.precision(16);

  file << "VP(";
  Core::LinAlg::Matrix<3, 1> midpoint_quad(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 4; ++i)
  {
    Core::LinAlg::Matrix<3, 1> cur(Core::LinAlg::Initialization::zero);
    cur(0, 0) = xyz_(0, i);
    cur(1, 0) = xyz_(1, i);
    cur(2, 0) = xyz_(2, i);
    midpoint_quad.update(1.0, cur, 1.0);
  }

  midpoint_quad.scale(1.0 / 4.0);
  file << midpoint_quad(0, 0) << "," << midpoint_quad(1, 0) << "," << midpoint_quad(2, 0);
  file << "){";

  // Choose midpoint of triangle as normal for now. Not best choice possibly.
  Core::LinAlg::Matrix<2, 1> eta;
  eta(0, 0) = 0.5;
  eta(1, 0) = 0.5;
  Core::LinAlg::Matrix<3, 1> normal;
  this->normal(eta, normal);
  file << normal(0) << "," << normal(1) << "," << normal(2);

  file << "};\n";
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Line2BoundaryCell::dump_gmsh_normal(std::ofstream& file)
{
  file.precision(16);

  const unsigned num_nodes = Core::FE::num_nodes<Core::FE::CellType::line2>;

  file << "VP(";
  Core::LinAlg::Matrix<3, 1> midpoint(Core::LinAlg::Initialization::zero);
  for (unsigned i = 0; i < num_nodes; ++i)
  {
    Core::LinAlg::Matrix<3, 1> xyz(&xyz_(0, i), true);
    midpoint.update(1.0, xyz, 1.0);
  }

  midpoint.scale(1.0 / num_nodes);
  for (unsigned j = 0; j < 3; ++j)
  {
    if (j > 0) file << ",";
    file << midpoint(j, 0);
  }
  file << "){";

  // Choose midpoint of line2
  Core::LinAlg::Matrix<2, 1> eta;
  eta = 0.0;
  Core::LinAlg::Matrix<3, 1> normal;
  this->normal(eta, normal);
  for (unsigned j = 0; j < 3; ++j)
  {
    if (j > 0) file << ",";
    file << normal(j, 0);
  }

  file << "};\n";
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::ArbitraryBoundaryCell::dump_gmsh_normal(std::ofstream& file)
{
  // TO DO: implement gmsh output for arbitrarily shaped bcell
  //  FOUR_C_THROW("not implemented");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Point1BoundaryCell::normal(
    const Core::LinAlg::Matrix<2, 1>& xsi, Core::LinAlg::Matrix<3, 1>& normal) const
{
  normal.put_scalar(0.0);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Line2BoundaryCell::normal(
    const Core::LinAlg::Matrix<2, 1>& xsi, Core::LinAlg::Matrix<3, 1>& normal) const
{
  const unsigned probdim = facet_->parent_side()->n_prob_dim();
  switch (probdim)
  {
    case 2:
      eval_normal_vectors<2, Core::FE::CellType::line2>(xyz_, xsi, normal);
      break;
    case 3:
      eval_normal_vectors<3, Core::FE::CellType::line2>(xyz_, xsi, normal);
      break;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Tri3BoundaryCell::normal(
    const Core::LinAlg::Matrix<2, 1>& xsi, Core::LinAlg::Matrix<3, 1>& normal) const
{
  // get derivatives at pos
  Core::LinAlg::Matrix<3, 3> side_xyze(xyz_.values(), true);

  Core::LinAlg::Matrix<2, 3> deriv;
  Core::LinAlg::Matrix<2, 3> A;

  Core::FE::shape_function_2d_deriv1(deriv, xsi(0), xsi(1), Core::FE::CellType::tri3);
  A.multiply_nt(deriv, side_xyze);

  // cross product to get the normal at the point
  normal(0) = A(0, 1) * A(1, 2) - A(0, 2) * A(1, 1);
  normal(1) = A(0, 2) * A(1, 0) - A(0, 0) * A(1, 2);
  normal(2) = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);

  double norm = normal.norm2();
  normal.scale(1. / norm);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Quad4BoundaryCell::normal(
    const Core::LinAlg::Matrix<2, 1>& xsi, Core::LinAlg::Matrix<3, 1>& normal) const
{
  // get derivatives at pos
  Core::LinAlg::Matrix<3, 4> side_xyze(xyz_.values(), true);
  // Position2d<Core::FE::CellType::quad4> position( side_xyze, xsi );
  // position.Normal( xsi, normal );

  Core::LinAlg::Matrix<2, 4> deriv;
  Core::LinAlg::Matrix<2, 3> A;

  Core::FE::shape_function_2d_deriv1(deriv, xsi(0), xsi(1), Core::FE::CellType::quad4);
  A.multiply_nt(deriv, side_xyze);

  // cross product to get the normal at the point
  normal(0) = A(0, 1) * A(1, 2) - A(0, 2) * A(1, 1);
  normal(1) = A(0, 2) * A(1, 0) - A(0, 0) * A(1, 2);
  normal(2) = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);

  double norm = normal.norm2();
  normal.scale(1. / norm);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::ArbitraryBoundaryCell::normal(
    const Core::LinAlg::Matrix<2, 1>& xsi, Core::LinAlg::Matrix<3, 1>& normal) const
{
  FOUR_C_THROW("Call get_normal_vector() to get normal for arbitrary boundarycells");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::FE::GaussIntegration Cut::Point1BoundaryCell::gauss_rule(int cubaturedegree)
{
  Core::FE::GaussIntegration gi(Core::FE::CellType::point1, cubaturedegree);
  return gi;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::FE::GaussIntegration Cut::Line2BoundaryCell::gauss_rule(int cubaturedegree)
{
  Core::FE::GaussIntegration gi(Core::FE::CellType::line2, cubaturedegree);
  return gi;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::FE::GaussIntegration Cut::Tri3BoundaryCell::gauss_rule(int cubaturedegree)
{
  Core::FE::GaussIntegration gi(Core::FE::CellType::tri3, cubaturedegree);
  return gi;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::FE::GaussIntegration Cut::Quad4BoundaryCell::gauss_rule(int cubaturedegree)
{
  Core::FE::GaussIntegration gi(Core::FE::CellType::quad4, cubaturedegree);
  return gi;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::FE::GaussIntegration Cut::ArbitraryBoundaryCell::gauss_rule(int cubaturedegree)
{
  return gauss_rule_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::ArbitraryBoundaryCell::element_center(Core::LinAlg::Matrix<3, 1>& midpoint)
{
  FOUR_C_THROW("Element Center for ArbitraryBoundaryCells not implemented!");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Point1BoundaryCell::element_center(Core::LinAlg::Matrix<3, 1>& midpoint)
{
  std::copy(xyz_.values(), xyz_.values() + 3, midpoint.data());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Line2BoundaryCell::element_center(Core::LinAlg::Matrix<3, 1>& midpoint)
{
  Core::LinAlg::Matrix<3, 1> center_rst(Core::LinAlg::Initialization::zero);
  my_element_center<Core::FE::CellType::line2>(center_rst, midpoint);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Tri3BoundaryCell::element_center(Core::LinAlg::Matrix<3, 1>& midpoint)
{
  Core::LinAlg::Matrix<3, 1> center;
  center(0, 0) = 0.25;
  center(1, 0) = 0.25;
  center(2, 0) = 0.0;
  my_element_center<Core::FE::CellType::tri3>(center, midpoint);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Quad4BoundaryCell::element_center(Core::LinAlg::Matrix<3, 1>& midpoint)
{
  Core::LinAlg::Matrix<3, 1> center;
  center(0, 0) = 0.0;
  center(1, 0) = 0.0;
  center(2, 0) = 0.0;
  my_element_center<Core::FE::CellType::quad4>(center, midpoint);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::LinAlg::Matrix<3, 1> Cut::Point1BoundaryCell::get_normal_vector()
{
  FOUR_C_THROW("There is no normal for Point1 boundarycell");
  exit(EXIT_FAILURE);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::LinAlg::Matrix<3, 1> Cut::Line2BoundaryCell::get_normal_vector()
{
  Core::LinAlg::Matrix<3, 1> normal_v(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<2, 1> xsi(Core::LinAlg::Initialization::zero);

  normal(xsi, normal_v);

  return normal_v;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::LinAlg::Matrix<3, 1> Cut::Tri3BoundaryCell::get_normal_vector()
{
  FOUR_C_THROW("Call Transform function to get normal for Tri3 boundarycell");
  exit(EXIT_FAILURE);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::LinAlg::Matrix<3, 1> Cut::Quad4BoundaryCell::get_normal_vector()
{
  FOUR_C_THROW("Call Transform function to get normal for Quad4 boundarycell");
  exit(EXIT_FAILURE);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::LinAlg::Matrix<3, 1> Cut::ArbitraryBoundaryCell::get_normal_vector() { return normal_; }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::BoundaryCell::print(std::ostream& stream)
{
  stream << "--- boundary cell ( address: " << std::setw(10) << this << " )\n";
  for (unsigned i = 0; i < points_->size(); i++)
  {
    (*points_)()[i]->print(stream);
    stream << "\n";
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::vector<std::vector<double>> Cut::BoundaryCell::coordinates_v()
{
  std::vector<std::vector<double>> corners;
  for (std::vector<Point*>::const_iterator j = (*points_)().begin(); j != (*points_)().end(); ++j)
  {
    std::vector<double> cornerLocal(3);
    cornerLocal.assign((*j)->x(), (*j)->x() + 3);
    corners.push_back(cornerLocal);
  }
  return corners;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::Tri3BoundaryCell::is_valid_boundary_cell()
{
  const int numnodes = Core::FE::num_nodes<Core::FE::CellType::tri3>;

  const std::vector<Point*> points = this->points();

  // create planes consisting of 3 nodes each
  Core::LinAlg::Matrix<numnodes, 1> p0(points[0]->x());
  Core::LinAlg::Matrix<numnodes, 1> p1(points[1]->x());
  Core::LinAlg::Matrix<numnodes, 1> p2(points[2]->x());

  Core::LinAlg::Matrix<numnodes, 1> v01;
  Core::LinAlg::Matrix<numnodes, 1> v02;
  Core::LinAlg::Matrix<numnodes, 1> v12;

  v01.update(1, p1, -1, p0, 0);
  v02.update(1, p2, -1, p0, 0);
  v12.update(1, p1, -1, p2, 0);

  // Get distance to origin
  Core::LinAlg::Matrix<numnodes, 1> temp(Core::LinAlg::Initialization::zero);
  temp(0, 0) = p0.norm2();  // Distance of points to origin
  temp(1, 0) = p1.norm2();
  temp(2, 0) = p2.norm2();

  // This is to scale the tolerance, it determines our maximum precision (i.e. machine precision)
  //  double max_dist_to_origin = temp.NormInf();

  // Distance between points in triangle
  temp(0, 0) = v01.norm2();
  temp(1, 0) = v02.norm2();
  temp(2, 0) = v12.norm2();

  double min_dist_in_tri = temp.min_value();
  // We want to test with this one I think... But might lead to problems.
  //  double tolerance = LINSOLVETOL*max_dist_to_origin;

  temp(0, 0) = points[0]->tolerance();
  temp(1, 0) = points[1]->tolerance();
  temp(2, 0) = points[2]->tolerance();
  double max_tol_points = temp.max_value();


  //  std::cout << "min_dist_in_tri: " << min_dist_in_tri << std::endl;
  //  std::cout << "tolerance: " << tolerance << std::endl;
  //  std::cout << "max_tol_points: " << max_tol_points << std::endl;

  if (min_dist_in_tri < max_tol_points)
  {
    return false;
  }

  return true;
}

// function specializations
template void Cut::BoundaryCell::transform_local_coords<Core::FE::CellType::tri3>(Element* elem1,
    const Core::LinAlg::Matrix<2, 1>& eta, Core::LinAlg::Matrix<3, 1>& x_gp_lin,
    Core::LinAlg::Matrix<3, 1>& normal, double& drs, bool shadow);
template void Cut::BoundaryCell::transform_local_coords<Core::FE::CellType::quad4>(Element* elem1,
    const Core::LinAlg::Matrix<2, 1>& eta, Core::LinAlg::Matrix<3, 1>& x_gp_lin,
    Core::LinAlg::Matrix<3, 1>& normal, double& drs, bool shadow);

FOUR_C_NAMESPACE_CLOSE
