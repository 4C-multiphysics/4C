// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cut_elementhandle.hpp"

#include "4C_cut_boundarycell.hpp"
#include "4C_cut_integrationcell.hpp"
#include "4C_cut_mesh.hpp"
#include "4C_cut_node.hpp"
#include "4C_cut_point.hpp"
#include "4C_cut_position.hpp"
#include "4C_cut_tolerance.hpp"
#include "4C_cut_volumecell.hpp"
#include "4C_utils_enum.hpp"

#include <Teuchos_TimeMonitor.hpp>

#include <fstream>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
// Project the integration rule available in the local coordinates of the
// integration-cells to the local coordinates of background element
/*----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
std::shared_ptr<Core::FE::GaussPoints> Cut::ElementHandle::create_projected(
    const std::vector<Cut::Point*>& cpoints, std::shared_ptr<Core::FE::GaussPoints> gp_ic)
{
  const unsigned nen = Core::FE::num_nodes(distype);
  const unsigned dim = Core::FE::dim<distype>;
  Core::LinAlg::Matrix<dim, nen> xie;
  if (cpoints.size() != nen) FOUR_C_THROW("non-matching number of points");

  // Find the local coordinates of given corner points w.r. to background ElementHandle
  for (unsigned i = 0; i < nen; ++i)
  {
    Cut::Point* p = cpoints[i];
    const Core::LinAlg::Matrix<3, 1>& xi = local_coordinates(p);

    // copy first dim entries into xie
    std::copy(xi.data(), xi.data() + dim, &xie(0, i));
  }

  Core::FE::GaussIntegration intpoints(gp_ic);
  std::shared_ptr<Core::FE::CollectedGaussPoints> cgp =
      std::make_shared<Core::FE::CollectedGaussPoints>(gp_ic->num_points());

  // Perform actual mapping to correct local coordinates
  Core::FE::GaussIntegration::project_gauss_points_local_to_global<distype>(xie, intpoints, *cgp);
  return cgp;
}


/*----------------------------------------------------------------------*/
// Collect the Gaussian points of all volume-cells belonging to this element in such a way
// that Gaussian rule for every volume-cell can be separated
/*----------------------------------------------------------------------*/
void Cut::ElementHandle::volume_cell_gauss_points(
    plain_volumecell_set& cells, std::vector<Core::FE::GaussIntegration>& intpoints)
{
  intpoints.clear();
  intpoints.reserve(cells.size());

  for (plain_volumecell_set::iterator i = cells.begin(); i != cells.end(); ++i)
  {
    Cut::VolumeCell* vc = *i;

    std::shared_ptr<Core::FE::GaussPointsComposite> gpc =
        std::make_shared<Core::FE::GaussPointsComposite>(0);

    switch (vc->parent_element()->get_element_integration_type())
    {
      case EleIntType_Tessellation:
      {
        append_volume_cell_gauss_points_tessellation(*gpc, vc);
        break;
      }
      case EleIntType_MomentFitting:
      {
        append_volume_cell_gauss_points_moment_fitting(*gpc, vc);
        break;
      }
      case EleIntType_DirectDivergence:
      {
        append_volume_cell_gauss_points_direct_divergence(gpc, vc);
        break;
      }
      default:
      {
        FOUR_C_THROW("non supported element integration type for given volume-cell {}",
            vc->parent_element()->get_element_integration_type());
      }
    }

#ifdef QUADCOMP
    if (gpc->NumPoints() > 56)
    {
      QuadratureCompression qc;
      bool quad_comp_success = qc.perform_compression_of_quadrature(*gpc, vc);
      if (quad_comp_success)
      {
        intpoints.push_back(Core::FE::GaussIntegration(qc.get_compressed_quadrature()));

        // reset the Gauss points for the volumecell so that the compression need not be performed
        // for each iteration within the Newton loop
        vc->SetGaussRule(qc.get_compressed_quadrature());
      }
      else
        intpoints.push_back(Core::FE::GaussIntegration(gpc));
    }
    else
    {
      intpoints.push_back(Core::FE::GaussIntegration(gpc));
    }
#else
    intpoints.push_back(Core::FE::GaussIntegration(gpc));
#endif
  }
}


void Cut::ElementHandle::append_volume_cell_gauss_points_tessellation(
    Core::FE::GaussPointsComposite& gpc, Cut::VolumeCell* vc)
{
  //---------------
  // For tessellation, we have Gauss points calculated at local coordinates of each integrationcells
  // we transform this to local coordinates of background ElementHandle
  //----------------
  const Cut::plain_integrationcell_set& cells = vc->integration_cells();
  for (Cut::plain_integrationcell_set::const_iterator i = cells.begin(); i != cells.end(); ++i)
  {
    Cut::IntegrationCell* ic = *i;

    std::shared_ptr<Core::FE::GaussPoints> gp_ic =
        Core::FE::create_gauss_points(ic->shape(), ic->cubature_degree(ic->shape()));
    const std::vector<Cut::Point*>& cpoints = ic->points();

    switch (ic->shape())
    {
      case Core::FE::CellType::tri3:
      {
        std::shared_ptr<Core::FE::GaussPoints> gp =
            create_projected<Core::FE::CellType::tri3>(cpoints, gp_ic);
        gpc.append(gp);
        break;
      }
      case Core::FE::CellType::quad4:
      {
        std::shared_ptr<Core::FE::GaussPoints> gp =
            create_projected<Core::FE::CellType::quad4>(cpoints, gp_ic);
        gpc.append(gp);
        break;
      }
      case Core::FE::CellType::hex8:
      {
        std::shared_ptr<Core::FE::GaussPoints> gp =
            create_projected<Core::FE::CellType::hex8>(cpoints, gp_ic);
        gpc.append(gp);
        break;
      }
      case Core::FE::CellType::tet4:
      {
        std::shared_ptr<Core::FE::GaussPoints> gp =
            create_projected<Core::FE::CellType::tet4>(cpoints, gp_ic);
        gpc.append(gp);
        break;
      }
      case Core::FE::CellType::wedge6:
      {
        std::shared_ptr<Core::FE::GaussPoints> gp =
            create_projected<Core::FE::CellType::wedge6>(cpoints, gp_ic);
        gpc.append(gp);
        break;
      }
      case Core::FE::CellType::pyramid5:
      {
        std::shared_ptr<Core::FE::GaussPoints> gp =
            create_projected<Core::FE::CellType::pyramid5>(cpoints, gp_ic);
        gpc.append(gp);
        break;
      }
      default:
        FOUR_C_THROW("unsupported integration cell type ( cell type = {} )",
            Core::FE::cell_type_to_string(ic->shape()));
    }
  }
}

void Cut::ElementHandle::append_volume_cell_gauss_points_moment_fitting(
    Core::FE::GaussPointsComposite& gpc, Cut::VolumeCell* vc)
{
  //-------------------
  // For MomentFitting, we have Gauss points that are calculated w.r to local coordinates of linear
  // shadow element If background ElementHandle is linear, then no need for any mapping Else, we map
  // these points to local coordinates of corresponding Quad element
  //-------------------

  //---------------------------------------------
  const std::vector<Cut::Point*>& cpoints = vc->parent_element()->points();
  std::shared_ptr<Core::FE::GaussPoints> gp_ic = vc->get_gauss_rule();


  switch (shape())
  {
    case Core::FE::CellType::hex8:
    case Core::FE::CellType::tet4:
    case Core::FE::CellType::wedge6:
    case Core::FE::CellType::pyramid5:
    {
      gpc.append(gp_ic);
      break;
    }

    case Core::FE::CellType::hex20:
    case Core::FE::CellType::hex27:
    {
      std::shared_ptr<Core::FE::GaussPoints> gp =
          create_projected<Core::FE::CellType::hex8>(cpoints, gp_ic);
      gpc.append(gp);
      break;
    }
    case Core::FE::CellType::tet10:
    {
      std::shared_ptr<Core::FE::GaussPoints> gp =
          create_projected<Core::FE::CellType::tet4>(cpoints, gp_ic);
      gpc.append(gp);
      break;
    }
    default:
    {
      FOUR_C_THROW("element handle for this element is not available\n");
      break;
    }
  }
}


void Cut::ElementHandle::append_volume_cell_gauss_points_direct_divergence(
    std::shared_ptr<Core::FE::GaussPointsComposite> gpc, Cut::VolumeCell* vc)
{
  //-------------------
  // For DirectDivergence, we calculate Gauss points at the correct local coord. during construction
  // itself This method is handled separately because
  // 1. main Gauss pts should be mapped w.r to each facet of vcell
  //         --> element volume mapping as done for tessellation and moment fitting do not work
  // 2. Internal Gauss pts can be obtained only if we have correctly mapped main Gauss points
  //-------------------
  std::shared_ptr<Core::FE::GaussPoints> gp = vc->get_gauss_rule();

  // volume cell gausspoints are identified to be negligible in
  // Cut::VolumeCell::direct_divergence_gauss_rule
  if (gp == nullptr) return;

  gpc->append(gp);
}

/*----------------------------------------------------------------------*/
// Collect the Gaussian points of all the volume-cells belonging to this element.
// The integration rules over all the volume-cells are connected.
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::FE::GaussPointsComposite> Cut::ElementHandle::gauss_points_connected(
    plain_volumecell_set& cells, VCellGaussPts gausstype)
{
  std::shared_ptr<Core::FE::GaussPointsComposite> gpc =
      std::make_shared<Core::FE::GaussPointsComposite>(0);

  for (plain_volumecell_set::iterator i = cells.begin(); i != cells.end(); ++i)
  {
    Cut::VolumeCell* vc = *i;

    const plain_integrationcell_set& cells = vc->integration_cells();


    if (gausstype == VCellGaussPts_Tessellation)
    {
      for (plain_integrationcell_set::const_iterator i = cells.begin(); i != cells.end(); ++i)
      {
        Cut::IntegrationCell* ic = *i;

        std::shared_ptr<Core::FE::GaussPoints> gp_ic =
            Core::FE::create_gauss_points(ic->shape(), ic->cubature_degree(ic->shape()));
        const std::vector<Cut::Point*>& cpoints = ic->points();

        switch (ic->shape())
        {
          case Core::FE::CellType::hex8:
          {
            std::shared_ptr<Core::FE::GaussPoints> gp =
                create_projected<Core::FE::CellType::hex8>(cpoints, gp_ic);
            gpc->append(gp);
            break;
          }
          case Core::FE::CellType::tet4:
          {
            std::shared_ptr<Core::FE::GaussPoints> gp =
                create_projected<Core::FE::CellType::tet4>(cpoints, gp_ic);
            gpc->append(gp);
            break;
          }
          case Core::FE::CellType::wedge6:
          {
            std::shared_ptr<Core::FE::GaussPoints> gp =
                create_projected<Core::FE::CellType::wedge6>(cpoints, gp_ic);
            gpc->append(gp);
            break;
          }
          case Core::FE::CellType::pyramid5:
          {
            std::shared_ptr<Core::FE::GaussPoints> gp =
                create_projected<Core::FE::CellType::pyramid5>(cpoints, gp_ic);
            gpc->append(gp);
            break;
          }
          default:
            FOUR_C_THROW("unsupported integration cell type ( cell type = {} )",
                Core::FE::cell_type_to_string(ic->shape()));
        }
      }
    }
    else if (gausstype == VCellGaussPts_MomentFitting ||
             gausstype == VCellGaussPts_DirectDivergence)
    {
      std::shared_ptr<Core::FE::GaussPoints> gp = vc->get_gauss_rule();
      gpc->append(gp);
    }
  }

  return gpc;
}

/*----------------------------------------------------------------------*/
// Collect the Gauss points of all the boundary-cells belong to this element.
// This is the method used now in the new implementation
/*----------------------------------------------------------------------*/
void Cut::ElementHandle::boundary_cell_gauss_points_lin(
    const std::map<int, std::vector<Cut::BoundaryCell*>>& bcells,
    std::map<int, std::vector<Core::FE::GaussIntegration>>& intpoints, const int bc_cubaturedegree)
{
  // TEUCHOS_FUNC_TIME_MONITOR( "Cut::ElementHandle::boundary_cell_gauss_points_lin" );

  for (std::map<int, std::vector<Cut::BoundaryCell*>>::const_iterator i = bcells.begin();
      i != bcells.end(); ++i)
  {
    int sid = i->first;
    const std::vector<Cut::BoundaryCell*>& cells = i->second;
    std::vector<Core::FE::GaussIntegration>& cell_points = intpoints[sid];

    //    // safety check
    //    if(sid < 0)
    //    {
    //       FOUR_C_THROW("invalid sid: {}", sid);
    //    }
    //    else
    //    {
    //      // ask for a mesh cutting side
    //      SideHandle * side = wizard->GetMeshCuttingSide( sid, 0 );
    //
    //      // ask for level-set cutting side
    //      bool is_ls_side = wizard->HasLSCuttingSide( sid );
    //      if ( side==nullptr and !is_ls_side )
    //      {
    //        FOUR_C_THROW( "no side with given id available for combined mesh and
    //        level-set cut" );
    //      }
    //    }

    cell_points.clear();
    cell_points.reserve(cells.size());

    for (std::vector<Cut::BoundaryCell*>::const_iterator i = cells.begin(); i != cells.end(); ++i)
    {
      Cut::BoundaryCell* bc = *i;

      // Create (unmodified) gauss points for integration cell with requested
      // polynomial order. This is supposed to be fast, since there is a cache.
      cell_points.push_back(bc->gauss_rule(bc_cubaturedegree));
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::ElementHandle::get_boundary_cell_sets(
    const std::vector<Cut::Point::PointPosition>& desired_positions,
    std::vector<plain_boundarycell_set>& bcellsets)
{
  for (std::vector<Cut::Point::PointPosition>::const_iterator ip = desired_positions.begin();
      ip != desired_positions.end(); ++ip)
  {
    const std::vector<plain_boundarycell_set>& ele_bcellsets = get_boundary_cell_set(*ip);

    std::copy(
        ele_bcellsets.begin(), ele_bcellsets.end(), std::inserter(bcellsets, bcellsets.end()));
  }
}

/*----------------------------------------------------------------------*/
// get all the element' sets of volume-cells and nds-vectors
/*----------------------------------------------------------------------*/
void Cut::ElementHandle::get_volume_cells_dof_sets(std::vector<plain_volumecell_set>& cellsets,
    std::vector<std::vector<int>>& nds_sets, bool include_inner)
{
  const std::vector<plain_volumecell_set>& ele_vc_sets_inside = get_vc_sets_inside();
  const std::vector<plain_volumecell_set>& ele_vc_sets_outside = get_vc_sets_outside();

  std::vector<std::vector<int>>& nodaldofset_vc_sets_inside = get_nodal_dof_set_vc_sets_inside();
  std::vector<std::vector<int>>& nodaldofset_vc_sets_outside = get_nodal_dof_set_vc_sets_outside();

  if (include_inner)
  {
    std::copy(ele_vc_sets_inside.begin(), ele_vc_sets_inside.end(),
        std::inserter(cellsets, cellsets.end()));
    std::copy(nodaldofset_vc_sets_inside.begin(), nodaldofset_vc_sets_inside.end(),
        std::inserter(nds_sets, nds_sets.end()));
  }

  std::copy(ele_vc_sets_outside.begin(), ele_vc_sets_outside.end(),
      std::inserter(cellsets, cellsets.end()));
  std::copy(nodaldofset_vc_sets_outside.begin(), nodaldofset_vc_sets_outside.end(),
      std::inserter(nds_sets, nds_sets.end()));
}



/*----------------------------------------------------------------------*/
//! Collect all volume-cells belonging to this elements
/*----------------------------------------------------------------------*/
void Cut::LinearElementHandle::get_volume_cells(plain_volumecell_set& cells)
{
  const plain_volumecell_set& cs = element_->volume_cells();
  std::copy(cs.begin(), cs.end(), std::inserter(cells, cells.begin()));
}


/*----------------------------------------------------------------------*/
//! Collect all volume-cells belonging to this element ordered by position
/*----------------------------------------------------------------------*/
void Cut::LinearElementHandle::collect_volume_cells(
    plain_volumecell_set& cells_inside, plain_volumecell_set& cells_outside)
{
  const plain_volumecell_set& ecells = element_->volume_cells();

  // sort for inside and outside volume cells
  for (plain_volumecell_set::const_iterator i = ecells.begin(); i != ecells.end(); i++)
  {
    if ((*i)->position() == Cut::Point::outside)
    {
      cells_outside.insert(*i);
    }
    else  // inside vc
    {
      cells_inside.insert(*i);
    }
  }
}


/*----------------------------------------------------------------------------*
  get all the element sets of volume-cells, nds-vectors and integration points
  return true if a specific XFEM-Gaussrule is available and necessary
 *----------------------------------------------------------------------------*/
bool Cut::ElementHandle::get_cell_sets_dof_sets_gauss_points(
    std::vector<plain_volumecell_set>& cell_sets, std::vector<std::vector<int>>& nds_sets,
    std::vector<std::vector<Core::FE::GaussIntegration>>& intpoints_sets, bool include_inner)
{
  TEUCHOS_FUNC_TIME_MONITOR("Cut::ElementHandle::get_cell_sets_dof_sets_gauss_points");

  get_volume_cells_dof_sets(cell_sets, nds_sets, include_inner);

  intpoints_sets.clear();

  /* switch this on to only integrate cut elements
   * only one cell_sets for current element and element is not intersected, then
   * use a non-XFEM Gaussrule */
  if (!is_intersected())
  {
    // perform  standard integration on an uncut element
    if (cell_sets.size() == 1)
      return false;
    else if (cell_sets.size() == 0)
    {
      /* the element does not have a physical but only a ghost set, therefore
       * no integration is necessary and intpoints remains empty */
    }
    else
      FOUR_C_THROW(
          "number of cell_sets for a non-intersected element is invalid: {}", cell_sets.size());
  }


  intpoints_sets.reserve(cell_sets.size());

  for (std::vector<plain_volumecell_set>::iterator i = cell_sets.begin(); i != cell_sets.end(); i++)
  {
    plain_volumecell_set& cells = *i;

    std::vector<Core::FE::GaussIntegration> gaussCellsets;
    volume_cell_gauss_points(cells, gaussCellsets);

    intpoints_sets.push_back(gaussCellsets);
  }

  // return true if specific XFEM-integration rule available
  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::ElementHandle::get_gauss_rule_integration_cells(
    std::vector<Core::FE::GaussIntegration>& intpoints_vec, bool integrate_inside_volumecells)
{
  // Get the volume cells for this background element
  plain_volumecell_set cell_sets;
  get_volume_cells(cell_sets);

  // Check if this element is a whole element, e.g. it was not cut
  if (!is_intersected())
  {
    // perform  standard integration on an uncut element
    if (cell_sets.size() == 1)
      return false;
    else
      FOUR_C_THROW(
          "Number of cell_sets for a non-intersected element is invalid: {}", cell_sets.size());
  }
  // if the element is cut, then check how many volume sets it has. Here, we assume
  // that there is only one cut interface and the element should be cut by two, there should
  // be only two volume cells, one representing the inside and the outside sides
  FOUR_C_ASSERT_ALWAYS(cell_sets.size() == 2,
      "We expect exactly two volume cells for this element, but {} where found.", cell_sets.size());

  // Now, erase the volume cells that should not be integrated depending on
  // integrate_inside_volumecells
  if (integrate_inside_volumecells)
  {
    cell_sets.erase(std::remove_if(cell_sets.begin(), cell_sets.end(),
                        [](VolumeCell* volcell) { return volcell->position() == Point::outside; }),
        cell_sets.end());
  }
  else
  {
    cell_sets.erase(std::remove_if(cell_sets.begin(), cell_sets.end(),
                        [](VolumeCell* volcell) { return volcell->position() == Point::inside; }),
        cell_sets.end());
  }

  // At this point, the gauss points of the cells with physical meaning
  // are projected to the local space of their background element
  volume_cell_gauss_points(cell_sets, intpoints_vec);

  // return true if specific integration rule available
  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::LinearElementHandle::boundary_cell_set(Point::PointPosition position)
{
  // boundary cell sets were already added
  if (bcell_sets_.find(position) != bcell_sets_.end()) return;

  // increase the size of the paired vector and extract the std::vector
  // corresponding to the given position
  const unsigned curr_size = bcell_sets_.size();
  bcell_sets_.resize(curr_size + 1);
  std::vector<plain_boundarycell_set>& bcell_sets = bcell_sets_[position];
  bcell_sets.resize(1, plain_boundarycell_set());

  // get the volume cells of this linear element
  const plain_volumecell_set& evolcells = element_->volume_cells();

  plain_boundarycell_set& bcells = bcell_sets[0];
  for (plain_volumecell_set::const_iterator citvol = evolcells.begin(); citvol != evolcells.end();
      ++citvol)
  {
    const VolumeCell& evolcell = **citvol;
    if (evolcell.position() == position)
    {
      const plain_boundarycell_set& ebcells = evolcell.boundary_cells();
      for (plain_boundarycell_set::const_iterator citbc = ebcells.begin(); citbc != ebcells.end();
          ++citbc)
      {
        // avoid to add boundary cells twice
        if (bcells.find(*citbc) == bcells.end()) bcells.insert(*citbc);
      }
    }
  }
}

/*----------------------------------------------------------------------*/
// get the element's sets of volume-cells ordered by inside/outside position
/*----------------------------------------------------------------------*/
void Cut::LinearElementHandle::volume_cell_sets()
{
  if (!cells_set_)
  {
    const plain_volumecell_set& ecells = element_->volume_cells();

    // sort for inside and outside volume cells
    for (plain_volumecell_set::const_iterator i = ecells.begin(); i != ecells.end(); i++)
    {
      if ((*i)->position() == Cut::Point::outside)
      {
        plain_volumecell_set s;  // plain volume cell set with only one entry
        s.insert(*i);
        vc_sets_outside_.push_back(s);
      }
      else  // inside vc
      {
        plain_volumecell_set s;  // plain volume cell set with only one entry
        s.insert(*i);
        vc_sets_inside_.push_back(s);
      }
    }

    cells_set_ = true;
  }
}



/*----------------------------------------------------------------------*/
// returns true in case that any cut-side cut with the element produces cut points,
// i.e. also for touched cases (at points, edges or sides),
// or when an element side has more than one facet or is touched by fully/partially by the cut side
/*----------------------------------------------------------------------*/
bool Cut::QuadraticElementHandle::is_cut()
{
  for (std::vector<Element*>::iterator i = subelements_.begin(); i != subelements_.end(); ++i)
  {
    Element* e = *i;
    if (e->is_cut())
    {
      return true;
    }
  }
  return false;
}


/*----------------------------------------------------------------------*/
// return true if one of the sub-elements is intersected or the sub-elements
// have different positions and therefore the global element is intersected by a cut-side
/*----------------------------------------------------------------------*/
bool Cut::QuadraticElementHandle::is_intersected()
{
  Cut::Point::PointPosition unique_pos = Cut::Point::undecided;

  for (std::vector<Element*>::iterator i = subelements_.begin(); i != subelements_.end(); ++i)
  {
    Element* e = *i;
    if (e->is_intersected())
    {
      return true;
    }


    // do the sub-elements have different positions ? (e.g. when the cut side directly cuts between
    // sub-elements)
    if (unique_pos == Cut::Point::undecided)
    {
      // assume a new unique position for all sub elements
      unique_pos = (*e->volume_cells().begin())->position();
    }
    else if ((*e->volume_cells().begin())->position() != unique_pos)
    {
      return true;
    }
  }
  return false;
}


/*----------------------------------------------------------------------*/
// Collect all volume-cells belonging to this elements
/*----------------------------------------------------------------------*/
void Cut::QuadraticElementHandle::get_volume_cells(plain_volumecell_set& cells)
{
  for (std::vector<Element*>::iterator i = subelements_.begin(); i != subelements_.end(); ++i)
  {
    Element* e = *i;
    const plain_volumecell_set& cs = e->volume_cells();
    std::copy(cs.begin(), cs.end(), std::inserter(cells, cells.begin()));
  }
}


/*----------------------------------------------------------------------*/
// Collect all volume-cells belonging to this element ordered by position
/*----------------------------------------------------------------------*/
void Cut::QuadraticElementHandle::collect_volume_cells(
    plain_volumecell_set& cells_inside, plain_volumecell_set& cells_outside)
{
  for (std::vector<Element*>::const_iterator i = subelements_.begin(); i != subelements_.end(); ++i)
  {
    Element* e = *i;
    const plain_volumecell_set& ecells = e->volume_cells();

    // sort for inside and outside volume-cells
    for (plain_volumecell_set::const_iterator i = ecells.begin(); i != ecells.end(); i++)
    {
      if ((*i)->position() == Cut::Point::outside)
      {
        cells_outside.insert(*i);
      }
      else  // inside vc
      {
        cells_inside.insert(*i);
      }
    }  // volume-cells
  }  // sub-elements
}


/*----------------------------------------------------------------------*/
//  get the quadratic element's volumetric integration cells (just for Tessellation)
/*----------------------------------------------------------------------*/
void Cut::QuadraticElementHandle::get_integration_cells(plain_integrationcell_set& cells)
{
  for (std::vector<Element*>::iterator i = subelements_.begin(); i != subelements_.end(); ++i)
  {
    Element* e = *i;
    e->get_integration_cells(cells);
  }
}


/*----------------------------------------------------------------------*/
//  get all the quadratic element's boundary integration cells
//  TODO: this has to be corrected such that just the bcs which belong the
//  outside vcs will be returned
/*----------------------------------------------------------------------*/
void Cut::QuadraticElementHandle::get_boundary_cells(plain_boundarycell_set& bcells)
{
  FOUR_C_THROW("Deprecated version!");
  for (std::vector<Element*>::iterator i = subelements_.begin(); i != subelements_.end(); ++i)
  {
    Element* e = *i;
    e->get_boundary_cells(bcells);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::QuadraticElementHandle::boundary_cell_set(Point::PointPosition position)
{
  if (connected_bcell_sets_.find(position) != connected_bcell_sets_.end()) return;

  connect_boundary_cells(position);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::QuadraticElementHandle::connect_boundary_cells(Point::PointPosition position)
{
  plain_volumecell_set evolcells_position;
  collect_volume_cells(position, evolcells_position);

  std::vector<plain_volumecell_set> connected_evolcells_position;
  build_cell_sets(evolcells_position, connected_evolcells_position);

  build_boundary_cell_sets(connected_evolcells_position, connected_bcell_sets_[position]);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::QuadraticElementHandle::build_boundary_cell_sets(
    const std::vector<plain_volumecell_set>& connected_vcell_set,
    std::vector<plain_boundarycell_set>& connected_bcell_set) const
{
  connected_bcell_set.resize(connected_vcell_set.size(), plain_boundarycell_set());

  unsigned vcell_set_count = 0;
  for (std::vector<plain_volumecell_set>::const_iterator citvset = connected_vcell_set.begin();
      citvset != connected_vcell_set.end(); ++citvset)
  {
    const plain_volumecell_set& vcell_set = *citvset;
    plain_boundarycell_set& bcell_set = connected_bcell_set[vcell_set_count++];
    for (plain_volumecell_set::const_iterator citvc = vcell_set.begin(); citvc != vcell_set.end();
        ++citvc)
    {
      const VolumeCell& vcell = **citvc;
      const plain_boundarycell_set& bcells = vcell.boundary_cells();
      for (plain_boundarycell_set::const_iterator citbc = bcells.begin(); citbc != bcells.end();
          ++citbc)
      {
        // avoid to add bcells twice
        if (bcell_set.find(*citbc) == bcell_set.end()) bcell_set.insert(*citbc);
      }
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::QuadraticElementHandle::collect_volume_cells(
    Point::PointPosition position, plain_volumecell_set& evolcells_position) const
{
  for (std::vector<Element*>::const_iterator i = subelements_.begin(); i != subelements_.end(); ++i)
  {
    Element* e = *i;
    const plain_volumecell_set& ecells = e->volume_cells();

    // sort for inside and outside volume-cells
    for (plain_volumecell_set::const_iterator i = ecells.begin(); i != ecells.end(); i++)
    {
      VolumeCell* evolcell = *i;
      if (evolcell->position() == position)
      {
        evolcells_position.insert(evolcell);
      }
    }
  }
}



/*----------------------------------------------------------------------*/
//! get the element's sets of volume-cells ordered by inside/outside position
/*----------------------------------------------------------------------*/
void Cut::QuadraticElementHandle::volume_cell_sets()
{
  // connect volumecells of subelements
  connect_volume_cells();
}


/*----------------------------------------------------------------------*/
//! connect volume-cells to sets of volume-cells
/*----------------------------------------------------------------------*/
void Cut::QuadraticElementHandle::connect_volume_cells()
{
  // find the connection between volumecells of all subelements for the current element (hex8,
  // hex20, tet4, tet10 etc.) remark: this function determines not the connection outside this
  // element

  // get the volumecells of all subelements stored in two plainvolume_sets (inside and outside)
  if (!cells_connected_)
  {
    plain_volumecell_set e_vcs_inside;
    plain_volumecell_set e_vcs_outside;

    collect_volume_cells(e_vcs_inside, e_vcs_outside);

    build_cell_sets(e_vcs_inside, connected_vc_sets_inside_);
    build_cell_sets(e_vcs_outside, connected_vc_sets_outside_);

    cells_connected_ = true;
  }
}

/*----------------------------------------------------------------------*/
//! build sets
/*----------------------------------------------------------------------*/
void Cut::QuadraticElementHandle::build_cell_sets(
    plain_volumecell_set& cells_to_connect, std::vector<plain_volumecell_set>& connected_sets)
{
  plain_volumecell_set done;

  for (plain_volumecell_set::const_iterator i = cells_to_connect.begin();
      i != cells_to_connect.end(); ++i)
  {
    VolumeCell* cell = *i;
    if (done.count(cell) == 0)  // cell currently not-done
    {
      plain_volumecell_set connected;
      // REMARK: here use the version without! elements check:
      // here we build cell sets within one global element with vcs of subelements
      // maybe the vcs of one subelement are not connected within one subelement,
      // but within one global element, therefore more than one vc of one subelements
      // may be connected.
      //        cell->Neighbors( nullptr, cells_to_connect, done, connected, elements );
      cell->neighbors(nullptr, cells_to_connect, done, connected);

      if (connected.size() > 0)
      {
        connected_sets.push_back(connected);
        std::copy(connected.begin(), connected.end(), std::inserter(done, done.begin()));
      }
    }
  }
}


/*----------------------------------------------------------------------*/
// constructor
/*----------------------------------------------------------------------*/
Cut::Hex20ElementHandle::Hex20ElementHandle(Mesh& mesh, int eid, const std::vector<int>& node_ids)
    : QuadraticElementHandle()
{
  subelements_.reserve(8);

  const CellTopologyData* top_data = shards::getCellTopologyData<shards::Hexahedron<8>>();

  // create middle nodes

  Core::LinAlg::Matrix<3, 1> xyz;
  Core::LinAlg::Matrix<1, 1> lsv;

  plain_int_set node_nids;

  Core::LinAlg::Matrix<3, 8> side_xyze;
  Core::LinAlg::Matrix<1, 8> side_lsvs;
  Core::LinAlg::Matrix<8, 1> side_funct;
  std::vector<Node*> side_nodes(8);

  std::vector<Node*> center_nodes(6);

  // get the real nodes of the hex20 element and get the center nodes (shadow nodes) of its
  // quadratic sides which stored in the shadow_nodes list in cut_mesh.H using the eight side nodes
  // of the side as key special handling for the inner center node of hex20 element required, see
  // below, the inner center node is stored in the shadow_nodes map using the 20 nodes of the hex20
  // element as key

  // loop the six sides of the hex 20 element
  for (int localsideid = 0; localsideid < 6; ++localsideid)
  {
    node_nids.clear();
    // loop the eight nodes of each quad8 side of the hex20 element
    for (int i = 0; i < 8; ++i)
    {
      int localnodeid = Core::FE::eleNodeNumbering_hex27_surfaces[localsideid][i];
      Node* n = mesh.get_node(node_ids[localnodeid], static_cast<double*>(nullptr));
      side_nodes[i] = n;
      node_nids.insert(node_ids[localnodeid]);
      n->coordinates(&side_xyze(0, i));
      side_lsvs(i) = n->lsv();
    }

    Core::FE::shape_function_2d(side_funct, 0.0, 0.0, Core::FE::CellType::quad8);
    xyz.multiply(side_xyze, side_funct);
    lsv.multiply(side_lsvs, side_funct);

    // find the unique center node of the quadratic quad8 sides
    center_nodes[localsideid] = mesh.get_node(node_nids, xyz.data(), lsv(0));
  }

  Node* node20 = center_nodes[0];
  int node20_id = node20->id();

  Node* node21 = center_nodes[1];
  int node21_id = node21->id();

  Node* node22 = center_nodes[2];
  int node22_id = node22->id();

  Node* node23 = center_nodes[3];
  int node23_id = node23->id();

  Node* node24 = center_nodes[4];
  int node24_id = node24->id();

  Node* node25 = center_nodes[5];
  int node25_id = node25->id();

  Core::LinAlg::Matrix<3, 20> xyze;
  Core::LinAlg::Matrix<1, 20> lsvs;
  nodes_.reserve(20);
  for (int i = 0; i < 20; ++i)
  {
    Node* n = mesh.get_node(node_ids[i], static_cast<double*>(nullptr));
    nodes_.push_back(n);
    n->coordinates(&xyze(0, i));
    lsvs(i) = n->lsv();
  }

  // special handling for the inner center node of hex20 element
  // Remark: this node is also stored as shadow node in cut_mesh, however, the key for this shadow
  // node is the set of all 20 nodes of the hex20 element in contrast to the shadow nodes of sides,
  // for that the key are the eight nodes of the quad7 side
  Core::LinAlg::Matrix<20, 1> funct;
  Core::FE::shape_function_3d(funct, 0.0, 0.0, 0.0, Core::FE::CellType::hex20);

  xyz.multiply(xyze, funct);
  lsv.multiply(lsvs, funct);
  node_nids.clear();
  std::copy(node_ids.begin(), node_ids.end(), std::inserter(node_nids, node_nids.begin()));
  Node* node26 = mesh.get_node(node_nids, xyz.data(), lsv(0));
  int node26_id = node26->id();


  std::vector<int> nids(8);

  nids[0] = node_ids[0];
  nids[1] = node_ids[8];
  nids[2] = node20_id;
  nids[3] = node_ids[11];
  nids[4] = node_ids[12];
  nids[5] = node21_id;
  nids[6] = node26_id;
  nids[7] = node24_id;
  Element* sub1 = mesh.get_element(-1, nids, *top_data);
  sub1->set_as_shadow_elem();
  sub1->set_quad_corners(mesh, node_ids);
  sub1->set_quad_shape(Core::FE::CellType::hex20);
  subelements_.push_back(sub1);

  nids[0] = node_ids[8];
  nids[1] = node_ids[1];
  nids[2] = node_ids[9];
  nids[3] = node20_id;
  nids[4] = node21_id;
  nids[5] = node_ids[13];
  nids[6] = node22_id;
  nids[7] = node26_id;
  Element* sub2 = mesh.get_element(-1, nids, *top_data);
  sub2->set_as_shadow_elem();
  sub2->set_quad_corners(mesh, node_ids);
  sub2->set_quad_shape(Core::FE::CellType::hex20);
  subelements_.push_back(sub2);

  nids[0] = node20_id;
  nids[1] = node_ids[9];
  nids[2] = node_ids[2];
  nids[3] = node_ids[10];
  nids[4] = node26_id;
  nids[5] = node22_id;
  nids[6] = node_ids[14];
  nids[7] = node23_id;
  Element* sub3 = mesh.get_element(-1, nids, *top_data);
  sub3->set_as_shadow_elem();
  sub3->set_quad_corners(mesh, node_ids);
  sub3->set_quad_shape(Core::FE::CellType::hex20);
  subelements_.push_back(sub3);

  nids[0] = node_ids[11];
  nids[1] = node20_id;
  nids[2] = node_ids[10];
  nids[3] = node_ids[3];
  nids[4] = node24_id;
  nids[5] = node26_id;
  nids[6] = node23_id;
  nids[7] = node_ids[15];
  Element* sub4 = mesh.get_element(-1, nids, *top_data);
  sub4->set_as_shadow_elem();
  sub4->set_quad_corners(mesh, node_ids);
  sub4->set_quad_shape(Core::FE::CellType::hex20);
  subelements_.push_back(sub4);

  /////////////////////////////////////////////////////////////////

  nids[0] = node_ids[12];
  nids[1] = node21_id;
  nids[2] = node26_id;
  nids[3] = node24_id;
  nids[4] = node_ids[4];
  nids[5] = node_ids[16];
  nids[6] = node25_id;
  nids[7] = node_ids[19];
  Element* sub5 = mesh.get_element(-1, nids, *top_data);
  sub5->set_as_shadow_elem();
  sub5->set_quad_corners(mesh, node_ids);
  sub5->set_quad_shape(Core::FE::CellType::hex20);
  subelements_.push_back(sub5);

  nids[0] = node21_id;
  nids[1] = node_ids[13];
  nids[2] = node22_id;
  nids[3] = node26_id;
  nids[4] = node_ids[16];
  nids[5] = node_ids[5];
  nids[6] = node_ids[17];
  nids[7] = node25_id;
  Element* sub6 = mesh.get_element(-1, nids, *top_data);
  sub6->set_as_shadow_elem();
  sub6->set_quad_corners(mesh, node_ids);
  sub6->set_quad_shape(Core::FE::CellType::hex20);
  subelements_.push_back(sub6);

  nids[0] = node26_id;
  nids[1] = node22_id;
  nids[2] = node_ids[14];
  nids[3] = node23_id;
  nids[4] = node25_id;
  nids[5] = node_ids[17];
  nids[6] = node_ids[6];
  nids[7] = node_ids[18];
  Element* sub7 = mesh.get_element(-1, nids, *top_data);
  sub7->set_as_shadow_elem();
  sub7->set_quad_corners(mesh, node_ids);
  sub7->set_quad_shape(Core::FE::CellType::hex20);
  subelements_.push_back(sub7);

  nids[0] = node24_id;
  nids[1] = node26_id;
  nids[2] = node23_id;
  nids[3] = node_ids[15];
  nids[4] = node_ids[19];
  nids[5] = node25_id;
  nids[6] = node_ids[18];
  nids[7] = node_ids[7];
  Element* sub8 = mesh.get_element(-1, nids, *top_data);
  sub8->set_as_shadow_elem();
  sub8->set_quad_corners(mesh, node_ids);
  sub8->set_quad_shape(Core::FE::CellType::hex20);
  subelements_.push_back(sub8);

  // each subelement should know its parents id
  for (std::vector<Element*>::iterator i = subelements_.begin(); i != subelements_.end(); ++i)
  {
    Element* subelement = *i;
    subelement->parent_id(eid);
  }
}


/*----------------------------------------------------------------------*/
// constructor
/*----------------------------------------------------------------------*/
Cut::Hex27ElementHandle::Hex27ElementHandle(Mesh& mesh, int eid, const std::vector<int>& node_ids)
    : QuadraticElementHandle()
{
  subelements_.reserve(8);

  nodes_.reserve(27);
  for (int i = 0; i < 27; ++i)
  {
    Node* n = mesh.get_node(node_ids[i], static_cast<double*>(nullptr));
    nodes_.push_back(n);
  }

  const CellTopologyData* top_data = shards::getCellTopologyData<shards::Hexahedron<8>>();

  std::vector<int> nids(8);

  nids[0] = node_ids[0];
  nids[1] = node_ids[8];
  nids[2] = node_ids[20];
  nids[3] = node_ids[11];
  nids[4] = node_ids[12];
  nids[5] = node_ids[21];
  nids[6] = node_ids[26];
  nids[7] = node_ids[24];
  Element* sub1 = mesh.get_element(-1, nids, *top_data);
  sub1->set_as_shadow_elem();
  sub1->set_quad_corners(mesh, node_ids);
  sub1->set_quad_shape(Core::FE::CellType::hex27);
  subelements_.push_back(sub1);

  nids[0] = node_ids[8];
  nids[1] = node_ids[1];
  nids[2] = node_ids[9];
  nids[3] = node_ids[20];
  nids[4] = node_ids[21];
  nids[5] = node_ids[13];
  nids[6] = node_ids[22];
  nids[7] = node_ids[26];
  Element* sub2 = mesh.get_element(-1, nids, *top_data);
  sub2->set_as_shadow_elem();
  sub2->set_quad_corners(mesh, node_ids);
  sub2->set_quad_shape(Core::FE::CellType::hex27);
  subelements_.push_back(sub2);

  nids[0] = node_ids[20];
  nids[1] = node_ids[9];
  nids[2] = node_ids[2];
  nids[3] = node_ids[10];
  nids[4] = node_ids[26];
  nids[5] = node_ids[22];
  nids[6] = node_ids[14];
  nids[7] = node_ids[23];
  Element* sub3 = mesh.get_element(-1, nids, *top_data);
  sub3->set_as_shadow_elem();
  sub3->set_quad_corners(mesh, node_ids);
  sub3->set_quad_shape(Core::FE::CellType::hex27);
  subelements_.push_back(sub3);

  nids[0] = node_ids[11];
  nids[1] = node_ids[20];
  nids[2] = node_ids[10];
  nids[3] = node_ids[3];
  nids[4] = node_ids[24];
  nids[5] = node_ids[26];
  nids[6] = node_ids[23];
  nids[7] = node_ids[15];
  Element* sub4 = mesh.get_element(-1, nids, *top_data);
  sub4->set_as_shadow_elem();
  sub4->set_quad_corners(mesh, node_ids);
  sub4->set_quad_shape(Core::FE::CellType::hex27);
  subelements_.push_back(sub4);

  /////////////////////////////////////////////////////////////////

  nids[0] = node_ids[12];
  nids[1] = node_ids[21];
  nids[2] = node_ids[26];
  nids[3] = node_ids[24];
  nids[4] = node_ids[4];
  nids[5] = node_ids[16];
  nids[6] = node_ids[25];
  nids[7] = node_ids[19];
  Element* sub5 = mesh.get_element(-1, nids, *top_data);
  sub5->set_as_shadow_elem();
  sub5->set_quad_corners(mesh, node_ids);
  sub5->set_quad_shape(Core::FE::CellType::hex27);
  subelements_.push_back(sub5);

  nids[0] = node_ids[21];
  nids[1] = node_ids[13];
  nids[2] = node_ids[22];
  nids[3] = node_ids[26];
  nids[4] = node_ids[16];
  nids[5] = node_ids[5];
  nids[6] = node_ids[17];
  nids[7] = node_ids[25];
  Element* sub6 = mesh.get_element(-1, nids, *top_data);
  sub6->set_as_shadow_elem();
  sub6->set_quad_corners(mesh, node_ids);
  sub6->set_quad_shape(Core::FE::CellType::hex27);
  subelements_.push_back(sub6);

  nids[0] = node_ids[26];
  nids[1] = node_ids[22];
  nids[2] = node_ids[14];
  nids[3] = node_ids[23];
  nids[4] = node_ids[25];
  nids[5] = node_ids[17];
  nids[6] = node_ids[6];
  nids[7] = node_ids[18];
  Element* sub7 = mesh.get_element(-1, nids, *top_data);
  sub7->set_as_shadow_elem();
  sub7->set_quad_corners(mesh, node_ids);
  sub7->set_quad_shape(Core::FE::CellType::hex27);
  subelements_.push_back(sub7);

  nids[0] = node_ids[24];
  nids[1] = node_ids[26];
  nids[2] = node_ids[23];
  nids[3] = node_ids[15];
  nids[4] = node_ids[19];
  nids[5] = node_ids[25];
  nids[6] = node_ids[18];
  nids[7] = node_ids[7];
  Element* sub8 = mesh.get_element(-1, nids, *top_data);
  sub8->set_as_shadow_elem();
  sub8->set_quad_corners(mesh, node_ids);
  sub8->set_quad_shape(Core::FE::CellType::hex27);
  subelements_.push_back(sub8);

  // each subelement should know its parents id
  for (std::vector<Element*>::iterator i = subelements_.begin(); i != subelements_.end(); ++i)
  {
    Element* subelement = *i;
    subelement->parent_id(eid);
  }
}


/*----------------------------------------------------------------------*/
// constructor
/*----------------------------------------------------------------------*/
Cut::Tet10ElementHandle::Tet10ElementHandle(Mesh& mesh, int eid, const std::vector<int>& nids)
    : QuadraticElementHandle()
{
  subelements_.reserve(8);

  nodes_.reserve(10);
  for (int i = 0; i < 10; ++i)
  {
    Node* n = mesh.get_node(nids[i], static_cast<double*>(nullptr));
    nodes_.push_back(n);
  }

  const CellTopologyData* top_data = shards::getCellTopologyData<shards::Tetrahedron<4>>();

  std::vector<int> subnids(4);

  subnids[0] = nids[0];
  subnids[1] = nids[4];
  subnids[2] = nids[6];
  subnids[3] = nids[7];
  Element* sub1 = mesh.get_element(-1, subnids, *top_data);
  sub1->set_as_shadow_elem();
  sub1->set_quad_corners(mesh, nids);
  sub1->set_quad_shape(Core::FE::CellType::tet10);
  subelements_.push_back(sub1);

  subnids[0] = nids[4];
  subnids[1] = nids[1];
  subnids[2] = nids[5];
  subnids[3] = nids[8];
  Element* sub2 = mesh.get_element(-1, subnids, *top_data);
  sub2->set_as_shadow_elem();
  sub2->set_quad_corners(mesh, nids);
  sub2->set_quad_shape(Core::FE::CellType::tet10);
  subelements_.push_back(sub2);

  subnids[0] = nids[6];
  subnids[1] = nids[5];
  subnids[2] = nids[2];
  subnids[3] = nids[9];
  Element* sub3 = mesh.get_element(-1, subnids, *top_data);
  sub3->set_as_shadow_elem();
  sub3->set_quad_corners(mesh, nids);
  sub3->set_quad_shape(Core::FE::CellType::tet10);
  subelements_.push_back(sub3);

  subnids[0] = nids[7];
  subnids[1] = nids[8];
  subnids[2] = nids[9];
  subnids[3] = nids[3];
  Element* sub4 = mesh.get_element(-1, subnids, *top_data);
  sub4->set_as_shadow_elem();
  sub4->set_quad_corners(mesh, nids);
  sub4->set_quad_shape(Core::FE::CellType::tet10);
  subelements_.push_back(sub4);

  /////////////////////////////////////////////////////////////////

  subnids[0] = nids[4];
  subnids[1] = nids[5];
  subnids[2] = nids[6];
  subnids[3] = nids[8];
  Element* sub5 = mesh.get_element(-1, subnids, *top_data);
  sub5->set_as_shadow_elem();
  sub5->set_quad_corners(mesh, nids);
  sub5->set_quad_shape(Core::FE::CellType::tet10);
  subelements_.push_back(sub5);

  subnids[0] = nids[6];
  subnids[1] = nids[9];
  subnids[2] = nids[7];
  subnids[3] = nids[8];
  Element* sub6 = mesh.get_element(-1, subnids, *top_data);
  sub6->set_as_shadow_elem();
  sub6->set_quad_corners(mesh, nids);
  sub6->set_quad_shape(Core::FE::CellType::tet10);
  subelements_.push_back(sub6);

  subnids[0] = nids[4];
  subnids[1] = nids[6];
  subnids[2] = nids[7];
  subnids[3] = nids[8];
  Element* sub7 = mesh.get_element(-1, subnids, *top_data);
  sub7->set_as_shadow_elem();
  sub7->set_quad_corners(mesh, nids);
  sub7->set_quad_shape(Core::FE::CellType::tet10);
  subelements_.push_back(sub7);

  subnids[0] = nids[9];
  subnids[1] = nids[6];
  subnids[2] = nids[5];
  subnids[3] = nids[8];
  Element* sub8 = mesh.get_element(-1, subnids, *top_data);
  sub8->set_as_shadow_elem();
  sub8->set_quad_corners(mesh, nids);
  sub8->set_quad_shape(Core::FE::CellType::tet10);
  subelements_.push_back(sub8);

  // each subelement should know its parents id
  for (std::vector<Element*>::iterator i = subelements_.begin(); i != subelements_.end(); ++i)
  {
    Element* subelement = *i;
    subelement->parent_id(eid);
  }
}

/*----------------------------------------------------------------------*/
// constructor
/*----------------------------------------------------------------------*/
Cut::Wedge15ElementHandle::Wedge15ElementHandle(
    Mesh& mesh, int eid, const std::vector<int>& node_ids)
    : QuadraticElementHandle()
{
  subelements_.reserve(8);  // subdivide into 8 wedge 6 elements

  const CellTopologyData* top_data = shards::getCellTopologyData<shards::Wedge<6>>();

  // create middle nodes

  Core::LinAlg::Matrix<3, 1> xyz;
  Core::LinAlg::Matrix<1, 1> lsv;

  plain_int_set node_nids;

  Core::LinAlg::Matrix<3, 8> side_xyze;
  Core::LinAlg::Matrix<1, 8> side_lsvs;
  Core::LinAlg::Matrix<8, 1> side_funct;
  std::vector<Node*> side_nodes(8);

  std::vector<Node*> center_nodes(3);

  // get the real nodes of the wedge15 element and get the center nodes (shadow nodes) of its
  // quadratic quad8 sides which stored in the shadow_nodes list in cut_mesh.H using the eight side
  // nodes of the side as key special handling for the inner center node of wedge15 element
  // required, see below,

  // loop the 5 sides of the wedge15 element

  // three quadratic sides
  for (int localsideid = 0; localsideid < 3; ++localsideid)
  {
    node_nids.clear();
    // loop the eight nodes of each quad8 side of the wedge15 element
    for (int i = 0; i < 8; ++i)
    {
      int localnodeid = Core::FE::eleNodeNumbering_wedge18_quadsurfaces[localsideid][i];
      Node* n = mesh.get_node(node_ids[localnodeid], static_cast<double*>(nullptr));
      side_nodes[i] = n;
      node_nids.insert(node_ids[localnodeid]);
      n->coordinates(&side_xyze(0, i));
      side_lsvs(i) = n->lsv();
    }

    Core::FE::shape_function_2d(side_funct, 0.0, 0.0, Core::FE::CellType::quad8);
    xyz.multiply(side_xyze, side_funct);
    lsv.multiply(side_lsvs, side_funct);

    // find the unique center node of the quadratic quad8 sides
    center_nodes[localsideid] = mesh.get_node(node_nids, xyz.data(), lsv(0));
  }

  Core::LinAlg::Matrix<3, 6> tb_side_xyze;  // top_bottom_sides
  Core::LinAlg::Matrix<1, 6> tb_side_lsvs;
  Core::LinAlg::Matrix<6, 1> tb_side_funct;
  std::vector<Node*> tb_side_nodes(6);

  // two quadratic sides on top and bottom
  for (int localsideid = 0; localsideid < 2; ++localsideid)
  {
    node_nids.clear();
    // loop the 6 nodes of each tri6 side of the wedge15 element
    for (int i = 0; i < 6; ++i)
    {
      int localnodeid = Core::FE::eleNodeNumbering_wedge18_trisurfaces[localsideid][i];
      Node* n = mesh.get_node(node_ids[localnodeid], static_cast<double*>(nullptr));
      tb_side_nodes[i] = n;
      node_nids.insert(node_ids[localnodeid]);
      n->coordinates(&tb_side_xyze(0, i));
      tb_side_lsvs(i) = n->lsv();
    }
  }

  Node* node15 = center_nodes[0];
  int node15_id = node15->id();

  Node* node16 = center_nodes[1];
  int node16_id = node16->id();

  Node* node17 = center_nodes[2];
  int node17_id = node17->id();


  Core::LinAlg::Matrix<3, 15> xyze;
  Core::LinAlg::Matrix<1, 15> lsvs;
  nodes_.reserve(15);
  for (int i = 0; i < 15; ++i)
  {
    Node* n = mesh.get_node(node_ids[i], static_cast<double*>(nullptr));
    nodes_.push_back(n);
    n->coordinates(&xyze(0, i));
    lsvs(i) = n->lsv();
  }


  std::vector<int> nids(6);

  nids[0] = node_ids[0];
  nids[1] = node_ids[6];
  nids[2] = node_ids[8];
  nids[3] = node_ids[9];
  nids[4] = node15_id;
  nids[5] = node17_id;
  Element* sub1 = mesh.get_element(-1, nids, *top_data);
  sub1->set_as_shadow_elem();
  sub1->set_quad_corners(mesh, node_ids);
  sub1->set_quad_shape(Core::FE::CellType::wedge6);
  subelements_.push_back(sub1);

  nids[0] = node_ids[6];
  nids[1] = node_ids[1];
  nids[2] = node_ids[7];
  nids[3] = node15_id;
  nids[4] = node_ids[10];
  nids[5] = node16_id;
  Element* sub2 = mesh.get_element(-1, nids, *top_data);
  sub2->set_as_shadow_elem();
  sub2->set_quad_corners(mesh, node_ids);
  sub2->set_quad_shape(Core::FE::CellType::wedge6);
  subelements_.push_back(sub2);

  nids[0] = node_ids[6];
  nids[1] = node_ids[7];
  nids[2] = node_ids[8];
  nids[3] = node15_id;
  nids[4] = node16_id;
  nids[5] = node17_id;
  Element* sub3 = mesh.get_element(-1, nids, *top_data);
  sub3->set_as_shadow_elem();
  sub3->set_quad_corners(mesh, node_ids);
  sub3->set_quad_shape(Core::FE::CellType::wedge6);
  subelements_.push_back(sub3);

  nids[0] = node_ids[8];
  nids[1] = node_ids[7];
  nids[2] = node_ids[2];
  nids[3] = node17_id;
  nids[4] = node16_id;
  nids[5] = node_ids[11];
  Element* sub4 = mesh.get_element(-1, nids, *top_data);
  sub4->set_as_shadow_elem();
  sub4->set_quad_corners(mesh, node_ids);
  sub4->set_quad_shape(Core::FE::CellType::wedge6);
  subelements_.push_back(sub4);

  /////////////////////////////////////////////////////////////////

  nids[0] = node_ids[9];
  nids[1] = node15_id;
  nids[2] = node17_id;
  nids[3] = node_ids[3];
  nids[4] = node_ids[12];
  nids[5] = node_ids[14];
  Element* sub5 = mesh.get_element(-1, nids, *top_data);
  sub5->set_as_shadow_elem();
  sub5->set_quad_corners(mesh, node_ids);
  sub5->set_quad_shape(Core::FE::CellType::wedge6);
  subelements_.push_back(sub5);

  nids[0] = node15_id;
  nids[1] = node_ids[10];
  nids[2] = node16_id;
  nids[3] = node_ids[12];
  nids[4] = node_ids[4];
  nids[5] = node_ids[13];
  Element* sub6 = mesh.get_element(-1, nids, *top_data);
  sub6->set_as_shadow_elem();
  sub6->set_quad_corners(mesh, node_ids);
  sub6->set_quad_shape(Core::FE::CellType::wedge6);
  subelements_.push_back(sub6);

  nids[0] = node15_id;
  nids[1] = node16_id;
  nids[2] = node17_id;
  nids[3] = node_ids[12];
  nids[4] = node_ids[13];
  nids[5] = node_ids[14];
  Element* sub7 = mesh.get_element(-1, nids, *top_data);
  sub7->set_as_shadow_elem();
  sub7->set_quad_corners(mesh, node_ids);
  sub7->set_quad_shape(Core::FE::CellType::wedge6);
  subelements_.push_back(sub7);

  nids[0] = node17_id;
  nids[1] = node16_id;
  nids[2] = node_ids[11];
  nids[3] = node_ids[14];
  nids[4] = node_ids[13];
  nids[5] = node_ids[5];
  Element* sub8 = mesh.get_element(-1, nids, *top_data);
  sub8->set_as_shadow_elem();
  sub8->set_quad_corners(mesh, node_ids);
  sub8->set_quad_shape(Core::FE::CellType::wedge6);
  subelements_.push_back(sub8);

  // each subelement should know its parents id
  for (std::vector<Element*>::iterator i = subelements_.begin(); i != subelements_.end(); ++i)
  {
    Element* subelement = *i;
    subelement->parent_id(eid);
  }
}


/*----------------------------------------------------------------------*/
// compute local coordinates of the element for given global coordinates
/*----------------------------------------------------------------------*/
void Cut::Hex20ElementHandle::local_coordinates(
    const Core::LinAlg::Matrix<3, 1>& xyz, Core::LinAlg::Matrix<3, 1>& rst)
{
  std::shared_ptr<Cut::Position> pos =
      Cut::PositionFactory::build_position<3, Core::FE::CellType::hex20>(nodes_, xyz);

  bool success = pos->compute(1e-10);
  if (not success)
  {
    std::cout << "local coordinates for hex20 element could not be determined" << std::endl;
    for (int i = 0; i < (int)(nodes_.size()); i++)
    {
      std::cout << " node " << i << std::endl;
      nodes_[i]->plot(std::cout);
    }

    std::cout << "point in xyz: " << xyz << std::endl;

    FOUR_C_THROW("local coordinates for hex20 element could not be determined");
  }
  pos->local_coordinates(rst);
}


/*----------------------------------------------------------------------*/
// compute local coordinates of the element for given global coordinates
/*----------------------------------------------------------------------*/
void Cut::Hex27ElementHandle::local_coordinates(
    const Core::LinAlg::Matrix<3, 1>& xyz, Core::LinAlg::Matrix<3, 1>& rst)
{
  std::shared_ptr<Cut::Position> pos =
      Cut::PositionFactory::build_position<3, Core::FE::CellType::hex27>(nodes_, xyz);

  bool success = pos->compute();
  if (not success)
  {
  }
  pos->local_coordinates(rst);
}


/*----------------------------------------------------------------------*/
// compute local coordinates of the element for given global coordinates
/*----------------------------------------------------------------------*/
void Cut::Tet10ElementHandle::local_coordinates(
    const Core::LinAlg::Matrix<3, 1>& xyz, Core::LinAlg::Matrix<3, 1>& rst)
{
  std::shared_ptr<Cut::Position> pos =
      Cut::PositionFactory::build_position<3, Core::FE::CellType::tet10>(nodes_, xyz);

  bool success = pos->compute();
  if (not success)
  {
  }
  pos->local_coordinates(rst);
}

/*----------------------------------------------------------------------*/
// compute local coordinates of the element for given global coordinates
/*----------------------------------------------------------------------*/
void Cut::Wedge15ElementHandle::local_coordinates(
    const Core::LinAlg::Matrix<3, 1>& xyz, Core::LinAlg::Matrix<3, 1>& rst)
{
  std::shared_ptr<Cut::Position> pos =
      Cut::PositionFactory::build_position<3, Core::FE::CellType::wedge15>(nodes_, xyz);

  bool success = pos->compute();
  if (not success)
  {
  }
  pos->local_coordinates(rst);
}

FOUR_C_NAMESPACE_CLOSE
