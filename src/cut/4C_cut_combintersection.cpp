// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cut_combintersection.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_cut_levelsetintersection.hpp"
#include "4C_cut_levelsetside.hpp"
#include "4C_cut_meshintersection.hpp"
#include "4C_cut_tolerance.hpp"  // for EXTENDED_CUT_DEBUG_OUTPUT

#include <mpi.h>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*-----------------------------------------------------------------------------------------*
 * constructor for Combined intersection class (Levelset and Mesh intersection in one class)
 *-----------------------------------------------------------------------------------------*/

Cut::CombIntersection::CombIntersection(MPI_Comm comm)
    : ParentIntersection(Core::Communication::my_mpi_rank(comm)),
      LevelSetIntersection(comm, false),
      MeshIntersection(1, Core::Communication::my_mpi_rank(comm))
{
  // call also the ParentIntersection-constructor first, otherwise according to the public virtual
  // inheritance of diamond shape the standard constructor of ParentIntersection() with default
  // value of myrank is called
}


void Cut::CombIntersection::cut(bool screenoutput)
{
  TEUCHOS_FUNC_TIME_MONITOR("Cut --- 4/6 --- Cut_Intersection");

  if (myrank_ == 0 and screenoutput) Core::IO::cout << "\t * 4/6 Cut_Intersection ...";

  Mesh& m = normal_mesh();

  // Remark: we assume that there is no overlap between levelset-isocontour and mesh

  // find cut points with levelset-side
  if (side_ != nullptr)
  {
    m.cut(*side_);
  }
//
// find cut points with cut mesh
#if EXTENDED_CUT_DEBUG_OUTPUT
  if (myrank_ == 0 and screenoutput) Core::IO::cout << "\t\t...Finding cut points";
  double t_start = Teuchos::Time::wallTime();
#endif
  m.find_cut_points();
#if EXTENDED_CUT_DEBUG_OUTPUT
  double t_diff = Teuchos::Time::wallTime() - t_start;
  if (myrank_ == 0 and screenoutput)
    Core::IO::cout << "\t\t\t... Success (" << t_diff << " secs)" << Core::IO::endl;
  if (myrank_ == 0 and screenoutput) Core::IO::cout << "\t\t...Finding cut lines" << Core::IO::endl;
  t_start = Teuchos::Time::wallTime();
#endif
  m.make_cut_lines();
#if EXTENDED_CUT_DEBUG_OUTPUT
  t_diff = Teuchos::Time::wallTime() - t_start;
  if (myrank_ == 0 and screenoutput)
    Core::IO::cout << "\t\t\t... Success (" << t_diff << " secs)" << Core::IO::endl;
  if (myrank_ == 0 and screenoutput)
    Core::IO::cout << "\t\t...Finding cut facets" << Core::IO::endl;
  t_start = Teuchos::Time::wallTime();
#endif
  m.make_facets();
#if EXTENDED_CUT_DEBUG_OUTPUT
  t_diff = Teuchos::Time::wallTime() - t_start;
  if (myrank_ == 0 and screenoutput)
    Core::IO::cout << "\t\t\t... Success (" << t_diff << " secs)" << Core::IO::endl;
  if (myrank_ == 0 and screenoutput) Core::IO::cout << "\t\t...Finding cut cells" << Core::IO::endl;
  t_start = Teuchos::Time::wallTime();
#endif
  m.make_volume_cells();
#if EXTENDED_CUT_DEBUG_OUTPUT
  t_diff = Teuchos::Time::wallTime() - t_start;
  if (myrank_ == 0 and screenoutput)
    Core::IO::cout << "\t\t\t... Success (" << t_diff << " secs)" << Core::IO::endl;
#endif
}


void Cut::CombIntersection::find_node_positions()
{
  // TODO: this function and the overall inside-outside position strategy still has to be adapted
  // for more complex cases

  // NOTE: this will only work if mesh-cut area and level-set cut area are not overlapping

  Mesh& m = normal_mesh();

  // first, set the position for the mesh cut
  m.find_node_positions();

  // second, set the position for the level-set cut
  m.find_ls_node_positions();
}


void Cut::CombIntersection::add_element(int eid, const std::vector<int>& nids,
    const Core::LinAlg::SerialDenseMatrix& xyz, Core::FE::CellType distype, const double* lsv,
    const bool lsv_only_plus_domain)
{
  Cut::ElementHandle* e = nullptr;

  // consider level-set values to decide whether the element has to be added or not
  if (lsv != nullptr)
  {
    // NOTE: dependent on whether one or two phases are used for the computation, the number of
    // degrees of freedom is determined via the cut status of elements, if both fluid phases have to
    // be considered, we have to add only cut elements, as uncut elements always carry physical
    // degrees of freedom if only the plus domain is a physical field, we have to add also elements
    // with pure negative level-set values (nodes in the ghost-domain) such that the
    // Dofset-Management does not produce degrees of freedom for such nodes

    e = LevelSetIntersection::add_element(eid, nids, xyz, distype, lsv, lsv_only_plus_domain);
  }

  // no check necessary if element lies within bounding box of cut surface
  if (e != nullptr) return;

  MeshIntersection::add_element(eid, nids, xyz, distype, lsv);
}

void Cut::CombIntersection::add_level_set_side(int levelset_side)
{
  LevelSetIntersection::add_cut_side(levelset_side);
}

void Cut::CombIntersection::add_mesh_cutting_side(int sid, const std::vector<int>& nids,
    const Core::LinAlg::SerialDenseMatrix& xyz, Core::FE::CellType distype, int mi)
{
  MeshIntersection::add_cut_side(sid, nids, xyz, distype, mi);
}

FOUR_C_NAMESPACE_CLOSE
