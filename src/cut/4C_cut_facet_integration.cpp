// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cut_facet_integration.hpp"

#include "4C_cut_boundarycell.hpp"
#include "4C_cut_kernel.hpp"
#include "4C_cut_line_integration.hpp"
#include "4C_cut_options.hpp"
#include "4C_cut_side.hpp"
#include "4C_cut_triangulateFacet.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*-----------------------------------------------------------------------------------------------------*
      compute the equation of the plane Ax+By+Cz=D with the local coordinates of corner
points
*------------------------------------------------------------------------------------------------------*/
std::vector<double> Cut::FacetIntegration::equation_plane(
    const std::vector<std::vector<double>>& cornersLocal)
{
  // TODO: use references for return!!!
  // Newell's method of determining equation of plane
  std::vector<double> eqn_plane = Kernel::eqn_plane_of_polygon(cornersLocal);

  return eqn_plane;
}

/*---------------------------------------------------------------------------------------------------------------------*
  A facet is clockwise ordered meaning that the normal vector of the facet is acting on the wrong
direction. This routine checks if the facet is ordered clockwise sudhakar 05/15

  We have two cases:

  Case 1 : the facet is formed from a cut side
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      We compute the normal vectors of the parent side and the facet. If they are not acting on the
opposite directions, then the facet is clockwise ordered. REMEMBER: This idea cannot be used in the
case 2 below because of the use of Shards element topology, wherein all the surfaces are not ensured
to have correct normal

  Case 2 : the facet is formed from the element side
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      Construct a reference vector from pointing from the element centre to the geometric centre of
the facet. This vector must be in the correct normal direction. Take the dot product of reference
vector and the normal vector of facet. If the dot product < 0, then the facet is clockwise ordered
*----------------------------------------------------------------------------------------------------------------------*/
void Cut::FacetIntegration::is_clockwise(
    const std::vector<double>& eqn_plane, const std::vector<std::vector<double>>& cornersLocal)
{
  ordering_computed_ = true;
  clockwise_ = false;

  bool iscut = face1_->on_cut_side();

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Case 1: Facet is formed from the cut side
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if (iscut)
  {
    Side* parent = face1_->parent_side();

    double dotProduct = 0.0;

    if (not face1_->belongs_to_level_set_side())
    {
      const std::vector<Node*>& par_nodes = parent->nodes();
      std::vector<std::vector<double>> corners(par_nodes.size());
      int mm = 0;
      for (std::vector<Node*>::const_iterator i = par_nodes.begin(); i != par_nodes.end(); i++)
      {
        Node* nod = *i;
        double x1[3];
        nod->coordinates(x1);

        std::vector<double> pt_local(3);
#ifdef LOCAL
        Core::LinAlg::Matrix<3, 1> glo, loc;

        for (int nodno = 0; nodno < 3; nodno++) glo(nodno, 0) = x1[nodno];
        elem1_->local_coordinates(glo, loc);

        pt_local[0] = loc(0, 0);
        pt_local[1] = loc(1, 0);
        pt_local[2] = loc(2, 0);
#else
        pt_local[0] = x1[0];
        pt_local[1] = x1[1];
        pt_local[2] = x1[2];
#endif

        corners[mm] = pt_local;
        mm++;
      }

      std::vector<double> eqn_par = equation_plane(corners);

#ifdef LOCAL
      dotProduct = eqn_plane[0] * eqn_par[0];
#else
      dotProduct =
          eqn_plane[0] * eqn_par[0] + eqn_plane[1] * eqn_par[1] + eqn_plane[2] * eqn_par[2];
#endif
    }
    else  // If the facet is on a cut-side which also is a level set side.
    {
      std::vector<double> phi_deriv1;

      Core::LinAlg::Matrix<3, 1> coord;

      // First entry is midpoint of triangulation!!
      // Third entry could possibly be a concave point (take care of in Triangulation of Facet).
      // Thus-> Choose second entry, which should be "safe".
      coord(0, 0) = cornersLocal[1][0];
      coord(1, 0) = cornersLocal[1][1];
      coord(2, 0) = cornersLocal[1][2];
#ifdef LOCAL
      phi_deriv1 = elem1_->get_level_set_gradient_at_local_coords_in_local_coords(coord);
#else
      phi_deriv1 = elem1_->get_level_set_gradient(coord);
#endif

      dotProduct = eqn_plane[0] * phi_deriv1[0] + eqn_plane[1] * phi_deriv1[1] +
                   eqn_plane[2] * phi_deriv1[2];

#ifdef DIRECTDIV_EXTENDED_DEBUG_OUTPUT
      std::cout << "cornersLocalFacetForGrad: " << cornersLocal[1][0] << ", " << cornersLocal[1][1]
                << ", " << cornersLocal[1][2] << std::endl;
      std::cout << "phi_deriv1: " << phi_deriv1[0] << ", " << phi_deriv1[1] << ", " << phi_deriv1[2]
                << std::endl;
      std::cout << "eqn_plane_IC: " << eqn_plane[0] << ", " << eqn_plane[1] << ", " << eqn_plane[2]
                << std::endl;

      std::cout << "dotProduct: " << dotProduct << std::endl;
      std::cout << "position_ : " << position_ << std::endl;
#endif
    }

    // We use the fact that the normal to any cut side is always pointing towards fluid domain
    if (position_ == Cut::Point::inside)
    {
      if (dotProduct < 0.0) clockwise_ = true;
    }
    else if (position_ == Cut::Point::outside)
    {
      if (dotProduct > 0.0) clockwise_ = true;
    }
    else
    {
      FOUR_C_THROW("VC position not defined!");
    }
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Case 2: Facet is formed from the element  side
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  else
  {
    //-----
    // STEP 1: Get centroid of the parent element
    //-----
    Core::LinAlg::Matrix<3, 1> elecen;
#ifdef LOCAL
    switch (elem1_->Shape())
    {
      case Core::FE::CellType::hex8:
      {
        elecen = Core::FE::get_local_center_position<3>(Core::FE::CellType::hex8);
        break;
      }
      case Core::FE::CellType::tet4:
      {
        elecen = Core::FE::get_local_center_position<3>(Core::FE::CellType::tet4);
        break;
      }
      default:
      {
        FOUR_C_THROW(
            "Add other elements not only here but in complete direct divergence procedure");
        break;
      }
    }
#else
    elem1_->element_center(elecen);
#endif

    //-----
    // STEP 2: Get geometric centre point of the facet
    // For concave facets, this is not the actual centre, but it does not matter
    //-----
    Core::LinAlg::Matrix<3, 1> facecen;
    unsigned npts = cornersLocal.size();
    for (std::vector<std::vector<double>>::const_iterator fit = cornersLocal.begin();
        fit != cornersLocal.end(); fit++)
    {
      const std::vector<double>& ftp = *fit;
      for (unsigned dim = 0; dim < 3; dim++) facecen(dim, 0) += ftp[dim];
    }

    for (unsigned dim = 0; dim < 3; dim++) facecen(dim, 0) = facecen(dim, 0) / npts;

    //-----
    // STEP 3: Construct a unit vector that points FROM element centre TO the facet centre
    // This reference vector is in the correct normal direction
    //-----
    Core::LinAlg::Matrix<3, 1> ref_vec;
    ref_vec.update(1.0, facecen, -1.0, elecen);
    double l2_ref_vec = ref_vec.norm2();
    ref_vec.scale(1.0 / l2_ref_vec);

    //-----
    // STEP 4: Take dot product with the normal of facet
    // If both are in the opposite direction, then the facet nodes are arranged clockwise
    //-----
    Core::LinAlg::Matrix<3, 1> norm_fac;
    for (unsigned dim = 0; dim < 3; dim++) norm_fac(dim, 0) = eqn_plane[dim];

    double dotProduct = ref_vec.dot(norm_fac);
    if (dotProduct < 0.0) clockwise_ = 1;
  }
  // std::cout<<"clockwise = "<<clockwise_<<"\t"<<"is cut side = "<<iscut<<"\n";
  return;
}

/*-----------------------------------------------------------------------------------------------*
          Returns true if vertices of facets are ordered clockwise              sudhakar 07/12
*------------------------------------------------------------------------------------------------*/
bool Cut::FacetIntegration::is_clockwise_ordering()
{
  if (ordering_computed_) return clockwise_;

  std::vector<std::vector<double>> cornersLocal;
  face1_->corner_points_local(elem1_, cornersLocal);
  if (eqn_plane_.size() == 0)
  {
    eqn_plane_ = equation_plane(cornersLocal);
  }

  this->is_clockwise(eqn_plane_, cornersLocal);
  return clockwise_;
}

/*-----------------------------------------------------------------------------------------------*
                            computes x=f(y,z) from the plane equation
                  equation of this form is used to replace x in the line integral
*------------------------------------------------------------------------------------------------*/
std::vector<double> Cut::FacetIntegration::compute_alpha(
    std::vector<double>& eqn_plane, Cut::ProjectionDirection intType)
{
  std::vector<double> alfa(3);
  double a = eqn_plane[0];
  double b = eqn_plane[1];
  double c = eqn_plane[2];
  double d = eqn_plane[3];


  if (intType == Cut::proj_x)
  {
    alfa[0] = d / a;
    alfa[1] = -1.0 * b / a;
    alfa[2] = -1.0 * c / a;
  }
  else if (intType == Cut::proj_y)
  {
    alfa[0] = d / b;
    alfa[1] = -1.0 * c / b;
    alfa[2] = -1.0 * a / b;
  }
  else if (intType == Cut::proj_z)
  {
    alfa[0] = d / c;
    alfa[1] = -1.0 * a / c;
    alfa[2] = -1.0 * b / c;
  }
  else
  {
    FOUR_C_THROW("The facet integration type undefined");
    exit(1);
  }
  return alfa;
}

/*---------------------------------------------------------------------------------*
      return the absolute normal of the facet in a particular direction
*----------------------------------------------------------------------------------*/
double Cut::FacetIntegration::get_normal(Cut::ProjectionDirection intType)
{
  double normalScale = 0.0;
  for (unsigned i = 0; i < 3; i++) normalScale += eqn_plane_[i] * eqn_plane_[i];
  normalScale = sqrt(normalScale);

  if (intType == Cut::proj_x)
    return eqn_plane_[0] / normalScale;
  else if (intType == Cut::proj_y)
    return eqn_plane_[1] / normalScale;
  else if (intType == Cut::proj_z)
    return eqn_plane_[2] / normalScale;
  else
  {
    FOUR_C_THROW("The normal direction is unspecified");
    return 0.0;
  }
}

/*-------------------------------------------------------------------------------------*
                      perform integration over the facet
*--------------------------------------------------------------------------------------*/
double Cut::FacetIntegration::integrate_facet()
{
#ifdef LOCAL
  std::vector<std::vector<double>> cornersLocal;
  face1_->CornerPointsLocal(elem1_, cornersLocal, true);

  // Special case when using moment fitting for boundary integration
  if (global_)
  {
    cornersLocal.clear();
    cornersLocal = face1_->CornerPointsGlobal(elem1_, true);
  }

#else
  std::vector<std::vector<double>> cornersLocal = face1_->corner_points_global(elem1_, true);
#endif

  eqn_plane_ = equation_plane(cornersLocal);

  // the face is in the x-y or in y-z plane which gives zero facet integral
  if (fabs(eqn_plane_[0]) < TOL_EQN_PLANE && bcell_int_ == false) return 0.0;
  // x=0 plane which also do not contribute to facet integral
  if (fabs(eqn_plane_[1]) < TOL_EQN_PLANE && fabs(eqn_plane_[2]) < TOL_EQN_PLANE &&
      fabs(eqn_plane_[3]) < TOL_EQN_PLANE && bcell_int_ == false)
    return 0.0;

  if (bcell_int_ ==
      true)  // the integral value of boundarycell will not change w.r.t the ordering of vertices
    clockwise_ = false;
  else
    is_clockwise(eqn_plane_, cornersLocal);

  // integrating over each line of the facet
  double facet_integ = 0.0;
  if (bcell_int_ == false)
  {
    std::vector<double> alpha;
    alpha = compute_alpha(eqn_plane_, Cut::proj_x);
    for (std::vector<std::vector<double>>::const_iterator k = cornersLocal.begin();
        k != cornersLocal.end(); k++)
    {
      const std::vector<double> coords1 = *k;
      std::vector<double> coords2;
      // for the last line the end point is the first point of the facet
      if (k != (cornersLocal.end() - 1))
        coords2 = *(k + 1);
      else
        coords2 = *(cornersLocal.begin());

      // first index decides the x or y coordinate, second index decides the start point or end
      // point
      Core::LinAlg::Matrix<2, 2> coordLine;

      // The facet is projected over y-z plane and then the integration is performed
      // so only y- and z-coordinates are passed to make the lines
      //[0]-- indicates y and [1] indicate z because we are now in y-z plane
      coordLine(0, 0) = coords1[1];
      coordLine(1, 0) = coords1[2];
      coordLine(0, 1) = coords2[1];
      coordLine(1, 1) = coords2[2];

      LineIntegration line1(coordLine, inte_num_, alpha, bcell_int_);
      facet_integ += line1.integrate_line();
    }
  }
  else
  {
    // to reduce the truncation error introduced during the projection of plane,
    // the plane, along which the normal component is maximum, is chosen
    Cut::ProjectionDirection projType;
    if (fabs(eqn_plane_[0]) < 1e-8)
    {
      if (fabs(eqn_plane_[1]) < 1e-8)
        projType = Cut::proj_z;
      else if (fabs(eqn_plane_[2]) < 1e-8)
        projType = Cut::proj_y;
      else
      {
        if (fabs(eqn_plane_[1]) > fabs(eqn_plane_[2]))
          projType = Cut::proj_y;
        else
          projType = Cut::proj_z;
      }
    }
    else if (fabs(eqn_plane_[1]) < 1e-8)
    {
      if (fabs(eqn_plane_[2]) < 1e-8)
        projType = Cut::proj_x;
      else
      {
        if (fabs(eqn_plane_[0]) > fabs(eqn_plane_[2]))
          projType = Cut::proj_x;
        else
          projType = Cut::proj_z;
      }
    }
    else if (fabs(eqn_plane_[2]) < 1e-8)
    {
      if (fabs(eqn_plane_[1]) > fabs(eqn_plane_[0]))
        projType = Cut::proj_y;
      else
        projType = Cut::proj_x;
    }
    else
    {
      if (fabs(eqn_plane_[0]) >= fabs(eqn_plane_[1]) && fabs(eqn_plane_[0]) >= fabs(eqn_plane_[2]))
        projType = Cut::proj_x;
      else if (fabs(eqn_plane_[1]) >= fabs(eqn_plane_[2]) &&
               fabs(eqn_plane_[1]) >= fabs(eqn_plane_[0]))
        projType = Cut::proj_y;
      else
        projType = Cut::proj_z;
    }

    if (projType != Cut::proj_x and projType != Cut::proj_y and projType != Cut::proj_z)
    {
      FOUR_C_THROW("projection plane is not defined");
      exit(1);
    }

    boundary_facet_integration(cornersLocal, facet_integ, projType);
  }

  // this condition results in negative normal for all the lines in the line integral
  if (clockwise_ && bcell_int_ == false)
  {
    facet_integ = -1.0 * facet_integ;
    for (int i = 0; i < 4; i++) eqn_plane_[i] = -1.0 * eqn_plane_[i];
  }

  return facet_integ;
}

/*-----------------------------------------------------------------------------------------------*
                            Performs integration over the boundarycell
*------------------------------------------------------------------------------------------------*/
void Cut::FacetIntegration::boundary_facet_integration(
    const std::vector<std::vector<double>>& cornersLocal, double& facet_integ,
    Cut::ProjectionDirection intType)
{
  std::vector<double> alpha;
  double abs_normal = 0.0;

  for (std::vector<std::vector<double>>::const_iterator k = cornersLocal.begin();
      k != cornersLocal.end(); k++)
  {
    const std::vector<double> coords1 = *k;
    std::vector<double> coords2;

    // for the last line the end point is the first point of the facet
    if (k != (cornersLocal.end() - 1))
      coords2 = *(k + 1);
    else
      coords2 = *(cornersLocal.begin());

    // first index decides the x or y coordinate, second index decides the start point or end point
    Core::LinAlg::Matrix<2, 2> coordLine;
    if (intType == Cut::proj_x)
    {
      if (k == cornersLocal.begin())
      {
        alpha = compute_alpha(eqn_plane_, Cut::proj_x);
        abs_normal = get_normal(intType);
      }
      // The facet is projected over y-z plane and then the integration is performed
      // so only y- and z-coordinates are passed to make the lines
      //(0,i)-- indicates y and (1,i) indicate z because we are now in y-z plane
      coordLine(0, 0) = coords1[1];
      coordLine(1, 0) = coords1[2];
      coordLine(0, 1) = coords2[1];
      coordLine(1, 1) = coords2[2];

      LineIntegration line1(coordLine, inte_num_, alpha, bcell_int_);
      line1.set_integ_type(Cut::proj_x);
      facet_integ += line1.integrate_line();
    }
    else if (intType == Cut::proj_y)
    {
      if (k == cornersLocal.begin())
      {
        alpha = compute_alpha(eqn_plane_, Cut::proj_y);
        abs_normal = get_normal(intType);
      }
      // The facet is projected over y-z plane and then the integration is performed
      // so only y- and z-coordinates are passed to make the lines
      //(0,i)-- indicates y and (1,i) indicate z because we are now in y-z plane
      coordLine(0, 0) = coords1[2];
      coordLine(1, 0) = coords1[0];
      coordLine(0, 1) = coords2[2];
      coordLine(1, 1) = coords2[0];
      LineIntegration line1(coordLine, inte_num_, alpha, bcell_int_);
      line1.set_integ_type(Cut::proj_y);
      facet_integ += line1.integrate_line();
    }

    else if (intType == Cut::proj_z)
    {
      if (k == cornersLocal.begin())
      {
        alpha = compute_alpha(eqn_plane_, Cut::proj_z);
        abs_normal = get_normal(intType);
      }
      // The facet is projected over y-z plane and then the integration is performed
      // so only y- and z-coordinates are passed to make the lines
      //(0,i)-- indicates y and (1,i) indicate z because we are now in y-z plane
      coordLine(0, 0) = coords1[0];
      coordLine(1, 0) = coords1[1];
      coordLine(0, 1) = coords2[0];
      coordLine(1, 1) = coords2[1];
      LineIntegration line1(coordLine, inte_num_, alpha, bcell_int_);
      line1.set_integ_type(Cut::proj_z);
      facet_integ += line1.integrate_line();
    }
    else
    {
      FOUR_C_THROW("The facet integration type not supported");
      exit(1);
    }
  }

  facet_integ = facet_integ / abs_normal;
}

/*-------------------------------------------------------------------------------------------------*
      Generate integration rule for the facet if the divergence theorem is used      Sudhakar 03/12
      directly to generate Gauss integration rule for the facet
*--------------------------------------------------------------------------------------------------*/
void Cut::FacetIntegration::divergence_integration_rule(
    Mesh& mesh, Core::FE::CollectedGaussPoints& cgp)
{
  TEUCHOS_FUNC_TIME_MONITOR("Cut::FacetIntegration::divergence_integration_rule");

  std::list<std::shared_ptr<BoundaryCell>> divCells;

  // the last two parameters has no influence when called from the first parameter is set to true
  generate_divergence_cells(true, mesh, divCells);

  double normalX =
      get_normal(Cut::proj_x);  // make sure eqn of plane is available before calling this

  if (clockwise_)  // because if ordering is clockwise the contribution of this facet must be
                   // subtracted
    normalX = -1.0 * normalX;

#ifdef DIRECTDIV_EXTENDED_DEBUG_OUTPUT
  static int mm = 0;
  std::stringstream str;
  str << "divCell" << mm << ".pos";
  std::ofstream file(str.str().c_str());
  file << "View \""
       << "FacetBCellInfo"
       << "\" {\n";
#endif
  for (std::list<std::shared_ptr<BoundaryCell>>::iterator i = divCells.begin(); i != divCells.end();
      ++i)
  {
    BoundaryCell* bcell = &**i;

#ifdef DIRECTDIV_EXTENDED_DEBUG_OUTPUT
    if (face1_->IsFacetSplit())
    {
      std::vector<std::vector<Point*>> facetSplit = face1_->GetSplitCells();
      for (std::vector<std::vector<Point*>>::iterator k = facetSplit.begin(); k != facetSplit.end();
          k++)
      {
        std::cout << "Split cell output." << std::endl;
        Cut::Output::GmshTriSideDump(file, *k);
      }
    }
    else
      Cut::Output::GmshTriSideDump(file, bcell->Points());

    Core::LinAlg::Matrix<3, 1> midpt;
    bcell->element_center(midpt);
    Cut::Output::GmshVector(file, midpt, Cut::Output::GetEqOfPlane(bcell->Points()), true);
#endif

    Core::FE::GaussIntegration gi_temp =
        Core::FE::GaussIntegration(bcell->shape(), DIRECTDIV_GAUSSRULE);

    if (bcell->area() < REF_AREA_BCELL) continue;

    for (Core::FE::GaussIntegration::iterator iquad = gi_temp.begin(); iquad != gi_temp.end();
        ++iquad)
    {
      double drs = 0.0;
      Core::LinAlg::Matrix<3, 1> x_gp_loc(true), normal(true);
      const Core::LinAlg::Matrix<2, 1> eta(iquad.point());

      switch (bcell->shape())
      {
        case Core::FE::CellType::tri3:
        {
#ifdef LOCAL
          bcell->transform_local_coords<Core::FE::CellType::tri3>(
              elem1_, eta, x_gp_loc, normal, drs, true);
#else
          bcell->transform<Core::FE::CellType::tri3>(eta, x_gp_loc, normal, drs);
#endif
          break;
        }
        case Core::FE::CellType::quad4:
        {
#ifdef LOCAL
          bcell->transform_local_coords<Core::FE::CellType::quad4>(
              elem1_, eta, x_gp_loc, normal, drs, true);
#else
          bcell->transform<Core::FE::CellType::quad4>(eta, x_gp_loc, normal, drs);
#endif
          break;
        }
        default:
          FOUR_C_THROW("unsupported integration cell type");
      }
      double wei = iquad.weight() * drs * normalX;

      cgp.append(x_gp_loc, wei);
    }
  }
#ifdef DIRECTDIV_EXTENDED_DEBUG_OUTPUT
  file << "};\n";
  mm++;
#endif
}

/*-----------------------------------------------------------------------------------------------------*
      Split the facet into auxiliary divergence cells which is either Tri or Quad         Sudhakar
03/12
*------------------------------------------------------------------------------------------------------*/
void Cut::FacetIntegration::generate_divergence_cells(
    bool divergenceRule,  // if called to generate direct divergence rule
    Mesh& mesh, std::list<std::shared_ptr<BoundaryCell>>& divCells)
{
#ifdef LOCAL
  std::vector<std::vector<double>> cornersLocal;
  face1_->CornerPointsLocal(elem1_, cornersLocal);
#else
  std::vector<std::vector<double>> cornersLocal = face1_->corner_points_global(elem1_);
#endif

  eqn_plane_ = equation_plane(cornersLocal);

  // the face is in the x-y or in y-z plane which will not be considered when divergence theorem is
  // applied
  if (divergenceRule)
  {
    if (fabs(eqn_plane_[0]) < TOL_EQN_PLANE)
    {
      return;
    }
  }

  if (!divergenceRule && !face1_->on_cut_side()) return;

  is_clockwise(eqn_plane_, cornersLocal);

  std::vector<Point*> corners = face1_->corner_points();

  if (divergenceRule)
  {
    //-----------------------------------------------------------------
    // only facets with 3 corners are considered as the special case and a Tri3 is created directly.
    // we cannot directly create a Quad4 by checking the number of corners, because it can be a
    // concave facet with 4 corners for which Gaussian rules are not available
    //-----------------------------------------------------------------
    if (corners.size() == 3)
    {
      temporary_tri3(corners, divCells);
    }
    else  // split the arbitrary noded facet into cells
    {
      std::string splitMethod;

      std::vector<std::vector<Point*>> split;

      // if the facet is warped, do centre point triangulation --> reduced error (??)
      if (not face1_->is_planar(mesh, face1_->corner_points()))
      {
        if (!face1_->is_triangulated()) face1_->do_triangulation(mesh, corners);
        split = face1_->triangulation();
      }
      // the facet is not warped
      else
      {
        // split facet
        if (!face1_->is_facet_split()) face1_->split_facet(corners);
        split = face1_->get_split_cells();
        splitMethod = "split";
      }

      for (std::vector<std::vector<Point*>>::const_iterator j = split.begin(); j != split.end();
          ++j)
      {
        const std::vector<Point*>& tri = *j;
        if (tri.size() == 3)
          temporary_tri3(tri, divCells);
        else if (tri.size() == 4)  // split algorithm always gives convex quad
          temporary_quad4(tri, divCells);
        else
        {
          std::cout << "number of sides = " << tri.size();
          FOUR_C_THROW("Splitting created neither tri3 or quad4");
        }
      }
    }
  }
}

/*-------------------------------------------------------------------------------------------*
                            temporarily create a tri3 cell
                    this is temporary because this is not stored for the volumecell
*--------------------------------------------------------------------------------------------*/
void Cut::FacetIntegration::temporary_tri3(
    const std::vector<Point*>& corners, std::list<std::shared_ptr<BoundaryCell>>& divCells)
{
  Core::LinAlg::SerialDenseMatrix xyz(3, 3);
  for (int i = 0; i < 3; ++i) corners[i]->coordinates(&xyz(0, i));
  divCells.push_back(std::make_shared<Tri3BoundaryCell>(xyz, face1_, corners));
}

/*-------------------------------------------------------------------------------------------*
                            temporarily create a quad4 cell
                    this is temporary because this is not stored for the volumecell
*--------------------------------------------------------------------------------------------*/
void Cut::FacetIntegration::temporary_quad4(
    const std::vector<Point*>& corners, std::list<std::shared_ptr<BoundaryCell>>& divCells)
{
  Core::LinAlg::SerialDenseMatrix xyz(3, 4);
  for (int i = 0; i < 4; ++i) corners[i]->coordinates(&xyz(0, i));
  divCells.push_back(std::make_shared<Quad4BoundaryCell>(xyz, face1_, corners));
}

/*-------------------------------------------------------------------------------------------------*
      Generate integration rule for the facet if the divergence theorem is used      Sudhakar 03/12
      directly to generate Gauss integration rule for the facet
*--------------------------------------------------------------------------------------------------*/
void Cut::FacetIntegration::divergence_integration_rule_new(
    Mesh& mesh, Core::FE::CollectedGaussPoints& cgp)
{
  TEUCHOS_FUNC_TIME_MONITOR("Cut::FacetIntegration::divergence_integration_rule");

  std::list<std::shared_ptr<BoundaryCell>> divCells;

  // the last two parameters has no influence when called from the first parameter is set to true
  std::vector<std::vector<double>> eqn_plane_divCell;

  // If the facet is not planar it will be triangulated in DirectDivergence::list_facets().
  // Might want to split the facet for the case it is a planar quad -> less divCells.

#ifndef TRIANGULATE_ALL_FACETS_FOR_DIVERGENCECELLS
  bool triangulate_and_levelset =
      (face1_->corner_points().size() > 3 and face1_->belongs_to_level_set_side());
#else
  bool triangulate_and_levelset = (face1_->CornerPoints().size() > 3);
#endif

  //  if(face1_->CornerPoints().size()>3 and face1_->belongs_to_level_set_side())
  if (triangulate_and_levelset) face1_->do_triangulation(mesh, face1_->corner_points());

#ifndef TRIANGULATE_ALL_FACETS_FOR_DIVERGENCECELLS
  triangulate_and_levelset = ((face1_->is_triangulated() or face1_->is_facet_split()) and
                              face1_->belongs_to_level_set_side());
#else
  triangulate_and_levelset = (face1_->IsTriangulated() or face1_->IsFacetSplit());
#endif

  //  if((face1_->IsTriangulated() or face1_->IsFacetSplit()) and
  //  face1_->belongs_to_level_set_side())
  if (triangulate_and_levelset)
  {
    std::vector<std::vector<Point*>> facet_triang;
    if (face1_->is_triangulated())
      facet_triang = face1_->triangulation();
    else
      facet_triang = face1_->get_split_cells();
#ifdef DIRECTDIV_EXTENDED_DEBUG_OUTPUT
    std::cout << "TRIANGULATED OR SPLIT FACET!" << std::endl;
#endif

    unsigned counterDivCells = 0;
    for (std::vector<std::vector<Point*>>::const_iterator j = facet_triang.begin();
        j != facet_triang.end(); ++j)
    {
      generate_divergence_cells_new(true, mesh, divCells, *j);
      while (counterDivCells < divCells.size())
      {
        counterDivCells++;
        eqn_plane_divCell.push_back(eqn_plane_);
      }
    }
  }
  else
  {
    // OLD IMPLEMENTATION!
    // Maybe triangulate for cut side?
#ifdef DIRECTDIV_EXTENDED_DEBUG_OUTPUT
    std::cout << "NOT TRIANGULATED FACET." << std::endl;
#endif

    generate_divergence_cells_new(true, mesh, divCells, face1_->corner_points());
    // generate_divergence_cells(true,mesh,divCells);
    for (unsigned i = 0; i < divCells.size(); i++)
    {
      eqn_plane_divCell.push_back(eqn_plane_);
    }
  }

  // SAFETY-CHECK
  if (eqn_plane_divCell.size() != divCells.size())
    FOUR_C_THROW("Something wrong with divCell and clockwise assignment.");

  double normalX;

  int zz = 0;
  for (std::list<std::shared_ptr<BoundaryCell>>::iterator i = divCells.begin(); i != divCells.end();
      ++i)
  {
    BoundaryCell* bcell = &**i;

    // Get equation of plane for divergence Cell.
    std::vector<double> eqn_plane_bcell = eqn_plane_divCell[zz];

    double normalScale = 0.0;
    for (unsigned i = 0; i < 3; i++) normalScale += eqn_plane_bcell[i] * eqn_plane_bcell[i];
    normalScale = sqrt(normalScale);
    normalX = eqn_plane_bcell[0] / normalScale;

#ifdef DIRECTDIV_EXTENDED_DEBUG_OUTPUT
    if (face1_->IsFacetSplit())
    {
      std::vector<std::vector<Point*>> facetSplit = face1_->GetSplitCells();
      for (std::vector<std::vector<Point*>>::iterator k = facetSplit.begin(); k != facetSplit.end();
          k++)
      {
        Core::LinAlg::Matrix<3, 1> midpt(Core::LinAlg::Initialization::set_zero);

        std::vector<std::vector<double>> corners_split;
        for (std::vector<Point*>::iterator l = (*k).begin(); l != (*k).end(); l++)
        {
          const double* coords = (*l)->X();
          midpt(0, 0) += coords[0];
          midpt(1, 0) += coords[1];
          midpt(2, 0) += coords[2];
        }

        midpt.scale(1.0 / (*k).size());

        //        bcell->element_center(midpt);
        Cut::Output::GmshTriSideDump(file, *k);
        Cut::Output::GmshVector(file, midpt, Cut::Output::GetEqOfPlane(*k), true);
      }
    }
    else
    {
      Core::LinAlg::Matrix<3, 1> midpt(Core::LinAlg::Initialization::set_zero);
      bcell->element_center(midpt);
      Cut::Output::GmshTriSideDump(file, bcell->Points());
      Cut::Output::GmshVector(file, midpt, Cut::Output::GetEqOfPlane(bcell->Points()), true);
    }
#endif

    Core::FE::GaussIntegration gi_temp =
        Core::FE::GaussIntegration(bcell->shape(), DIRECTDIV_GAUSSRULE);

    if (bcell->area() < REF_AREA_BCELL) continue;

    for (Core::FE::GaussIntegration::iterator iquad = gi_temp.begin(); iquad != gi_temp.end();
        ++iquad)
    {
      double drs = 0.0;
      Core::LinAlg::Matrix<3, 1> x_gp_loc(true), normal(true);
      const Core::LinAlg::Matrix<2, 1> eta(iquad.point());

      switch (bcell->shape())
      {
        case Core::FE::CellType::tri3:
        {
#ifdef LOCAL
          bcell->transform_local_coords<Core::FE::CellType::tri3>(
              elem1_, eta, x_gp_loc, normal, drs, true);
#else
          bcell->transform<Core::FE::CellType::tri3>(eta, x_gp_loc, normal, drs);
#endif
          break;
        }
        case Core::FE::CellType::quad4:
        {
#ifdef LOCAL
          bcell->transform_local_coords<Core::FE::CellType::quad4>(
              elem1_, eta, x_gp_loc, normal, drs, true);
#else
          bcell->transform<Core::FE::CellType::quad4>(eta, x_gp_loc, normal, drs);
#endif
          break;
        }
        default:
          FOUR_C_THROW("unsupported integration cell type ( cell type = {} )",
              Core::FE::cell_type_to_string(bcell->shape()).c_str());
          exit(EXIT_FAILURE);
      }
      double wei = iquad.weight() * drs * normalX;

      cgp.append(x_gp_loc, wei);
    }
    zz++;  // Iterator of divCells.
  }
#ifdef DIRECTDIV_EXTENDED_DEBUG_OUTPUT
  file << "};\n";
  mm++;
#endif
}

void Cut::FacetIntegration::generate_divergence_cells_new(bool divergenceRule, Mesh& mesh,
    std::list<std::shared_ptr<BoundaryCell>>& divCells, const std::vector<Point*>& cornersGlobal)
{
// First convert format...
#ifdef LOCAL
  std::vector<std::vector<double>> cornersLocal;      //(cornersGlobal.size(),3);
  std::vector<std::vector<double>> cornersGlobalVec;  //(cornersGlobal.size(),3);

  for (std::vector<Point*>::const_iterator j = cornersGlobal.begin(); j != cornersGlobal.end(); ++j)
  {
    Core::LinAlg::Matrix<3, 1> cornersLocMatrix;
    Core::LinAlg::Matrix<3, 1> cornersGloMatrix((*j)->X());

    elem1_->local_coordinates(cornersGloMatrix, cornersLocMatrix);

    std::vector<double> cornerLocal(3);
    cornerLocal[0] = cornersLocMatrix(0, 0);
    cornerLocal[1] = cornersLocMatrix(1, 0);
    cornerLocal[2] = cornersLocMatrix(2, 0);
    cornersLocal.push_back(cornerLocal);

    //    std::cout << "cornerLocal: " << cornerLocal[0] << ", " << cornerLocal[1] << ", " <<
    //    cornerLocal[2] << std::endl;

    std::vector<double> cornerGlobal(3);
    cornerGlobal[0] = cornersGloMatrix(0, 0);
    cornerGlobal[1] = cornersGloMatrix(1, 0);
    cornerGlobal[2] = cornersGloMatrix(2, 0);
    cornersGlobalVec.push_back(cornerGlobal);
  }
#else
  std::vector<std::vector<double>> cornersGlobalVec;  //(cornersGlobal.size(),3);

  for (std::vector<Point*>::const_iterator j = cornersGlobal.begin(); j != cornersGlobal.end(); ++j)
  {
    Core::LinAlg::Matrix<3, 1> cornersGloMatrix((*j)->x());

    std::vector<double> cornerGlobal(3);
    cornerGlobal[0] = cornersGloMatrix(0, 0);
    cornerGlobal[1] = cornersGloMatrix(1, 0);
    cornerGlobal[2] = cornersGloMatrix(2, 0);
    cornersGlobalVec.push_back(cornerGlobal);
  }
  std::vector<std::vector<double>> cornersLocal =
      cornersGlobalVec;  // face1_->CornerPointsGlobal(elem1_);
#endif

  // Equation of plane for triangulated/split facet or full facet
  eqn_plane_ = equation_plane(cornersLocal);

  // the face is in the x-y or in y-z plane which will not be considered when divergence theorem is
  // applied
  if (divergenceRule)
  {
    if (fabs(eqn_plane_[0]) < TOL_EQN_PLANE / 10.0)
    {
#ifdef DIRECTDIV_EXTENDED_DEBUG_OUTPUT
      for (std::vector<std::vector<double>>::const_iterator j = cornersLocal.begin();
          j != cornersLocal.end(); ++j)
      {
        std::cout << "cornerLocal: " << (*j)[0] << ", " << (*j)[1] << ", " << (*j)[2] << std::endl;
      }
      std::cout << "eqn_plane_: " << eqn_plane_[0] << ", " << eqn_plane_[1] << ", " << eqn_plane_[2]
                << ", " << eqn_plane_[3] << std::endl;
      std::cout << "RETURN BECAUSE eqn_plane_ too small in x-dir." << std::endl;
#endif
      return;
    }
  }

  if (!divergenceRule && !face1_->on_cut_side()) return;

  is_clockwise(eqn_plane_, cornersLocal);

  std::vector<Point*> corners = cornersGlobal;

  if (clockwise_)
  {
    // CHANGE SIGN For equation of plane.
    eqn_plane_[0] = -eqn_plane_[0];
    eqn_plane_[1] = -eqn_plane_[1];
    eqn_plane_[2] = -eqn_plane_[2];
    eqn_plane_[3] = -eqn_plane_[3];
  }

  if (divergenceRule)
  {
    //-----------------------------------------------------------------
    // only facets with 3 corners are considered as the special case and a Tri3 is created directly.
    // we cannot directly create a Quad4 by checking the number of corners, because it can be a
    // concave facet with 4 corners for which Gaussian rules are not available
    //-----------------------------------------------------------------
    if (corners.size() == 3)
    {
      temporary_tri3(corners, divCells);
    }
    else  // split the arbitrary noded facet into cells
    {
      std::string splitMethod;

      std::vector<std::vector<Point*>> split;

      // if the facet is warped, do centre point triangulation --> reduced error (??)
      if (not face1_->is_planar(mesh, face1_->corner_points()))
      {
        if (!face1_->is_triangulated()) face1_->do_triangulation(mesh, corners);
        split = face1_->triangulation();
      }
      // the facet is not warped
      else
      {
        // split facet
        if (!face1_->is_facet_split()) face1_->split_facet(corners);
        split = face1_->get_split_cells();
        splitMethod = "split";
      }

      for (std::vector<std::vector<Point*>>::const_iterator j = split.begin(); j != split.end();
          ++j)
      {
        const std::vector<Point*>& tri = *j;
        if (tri.size() == 3)
          temporary_tri3(tri, divCells);
        else if (tri.size() == 4)  // split algorithm always gives convex quad
          temporary_quad4(tri, divCells);
        else
        {
          std::cout << "number of sides = " << tri.size();
          FOUR_C_THROW("Splitting created neither tri3 or quad4");
        }
      }
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
