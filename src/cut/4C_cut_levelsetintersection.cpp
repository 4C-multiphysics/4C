// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cut_levelsetintersection.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_cut_side.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Cut::LevelSetIntersection::LevelSetIntersection(MPI_Comm comm, bool create_side)
    : ParentIntersection(Core::Communication::my_mpi_rank(comm)), side_(nullptr), comm_(comm)
{
  if (create_side) add_cut_side(1);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::LevelSetIntersection::add_cut_side(int levelset_sid)
{
  if (side_) FOUR_C_THROW("currently only one levelset-side is supported");

  // create the levelset-side
  side_ = std::shared_ptr<Side>(Side::create_level_set_side(levelset_sid));
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Cut::ElementHandle* Cut::LevelSetIntersection::add_element(int eid, const std::vector<int>& nids,
    const Core::LinAlg::SerialDenseMatrix& xyz, Core::FE::CellType distype, const double* lsv,
    const bool lsv_only_plus_domain, const bool& check_lsv)
{
  int numnode = nids.size();
  if (numnode != xyz.numCols()) FOUR_C_THROW("node coordinate number mismatch");

  bool ltz = false;
  bool gtz = false;

  if (check_lsv)
  {
    // make sure element has LSV +ve and -ve in one of its nodes
    // ensures this is a cut element
    for (int i = 0; i < numnode; ++i)
    {
      if (lsv[i] <= REFERENCETOL) ltz = true;
      if (lsv[i] >= -REFERENCETOL) gtz = true;
    }
  }

  /* add all cut elements (different signs of levelset values) OR
   * if only plus domain is a physical field we have to add also
   * elements with only negative values (as they are not allowed to
   * carry DOFS at the end) */
  if ((not check_lsv) or (ltz and gtz) or (lsv_only_plus_domain and ltz))
  {
    // add all nodes to mesh
    for (int i = 0; i < numnode; ++i)
    {
      normal_mesh().get_node(nids[i], &xyz(0, i), lsv[i]);
    }

    // create element
    return mesh_.create_element(eid, nids, distype);
  }

  return nullptr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::LevelSetIntersection::cut_mesh(bool screenoutput)
{
  TEUCHOS_FUNC_TIME_MONITOR("Cut --- 1/3 --- Cut");

  Mesh& m = normal_mesh();

  double t_diff = 0.0;
  {
    const double t_start = Teuchos::Time::wallTime();
    if (myrank_ == 0 and screenoutput)
      std::cout << "\n\t * 1/6 Cut ......................" << std::flush;

    m.cut(*side_);

    if (myrank_ == 0 and screenoutput)
    {
      if (comm_) Core::Communication::barrier(comm_);
      t_diff = Teuchos::Time::wallTime() - t_start;

      printf("success! ( %10.4e secs )", t_diff);
    }
  }

  {
    const double t_start = Teuchos::Time::wallTime();
    if (myrank_ == 0 and screenoutput)
      std::cout << "\n\t * 2/6 MakeCutLines ............." << std::flush;

    m.make_cut_lines();

    if (myrank_ == 0 and screenoutput)
    {
      if (comm_) Core::Communication::barrier(comm_);
      t_diff = Teuchos::Time::wallTime() - t_start;

      printf("success! ( %10.4e secs )", t_diff);
    }
  }

  {
    const double t_start = Teuchos::Time::wallTime();
    if (myrank_ == 0 and screenoutput)
      std::cout << "\n\t * 3/6 MakeFacets ..............." << std::flush;

    m.make_facets();

    if (myrank_ == 0 and screenoutput)
    {
      if (comm_) Core::Communication::barrier(comm_);
      t_diff = Teuchos::Time::wallTime() - t_start;

      printf("success! ( %10.4e secs )", t_diff);
    }
  }

  {
    const double t_start = Teuchos::Time::wallTime();
    if (myrank_ == 0 and screenoutput)
      std::cout << "\n\t * 4/6 MakeVolumeCells .........." << std::flush;

    m.make_volume_cells();

    if (myrank_ == 0 and screenoutput)
    {
      if (comm_) Core::Communication::barrier(comm_);
      t_diff = Teuchos::Time::wallTime() - t_start;

      printf("success! ( %10.4e secs )", t_diff);
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::LevelSetIntersection::cut(bool include_inner, bool screenoutput, VCellGaussPts VCellGP)
{
  // ###########################################################################
  //  STEP 1/3 CUT THE MESH
  // ###########################################################################
  cut_mesh(screenoutput);

  // ###########################################################################
  //  STEP 2/3 ASSIGN DOFS
  // ###########################################################################

  Mesh& m = normal_mesh();

  if (options_.find_positions())
  {
    m.find_ls_node_positions();

    //=====================================================================

    m.find_nodal_dof_sets(include_inner);

    //=====================================================================
  }
  // #############################################################################
  //  STEP 3/3 FINALIZE, ASSIGN INTEGRATIONRULES
  // #############################################################################

  double t_diff = 0.0;
  {
    const double t_start = Teuchos::Time::wallTime();
    if (myrank_ == 0 and screenoutput)
      std::cout << "\n\t * 5/6 create_integration_cells ..." << std::flush;

    if (VCellGP == VCellGaussPts_Tessellation)
      m.create_integration_cells(0);
    else
      m.direct_divergence_gauss_rule(true, BCellGaussPts_Tessellation);

    if (myrank_ == 0 and screenoutput)
    {
      if (comm_) Core::Communication::barrier(comm_);
      t_diff = Teuchos::Time::wallTime() - t_start;

      printf("success! ( %10.4e secs )", t_diff);
    }
  }

  {
    const double t_start = Teuchos::Time::wallTime();
    if (myrank_ == 0 and screenoutput)
      std::cout << "\n\t * 6/6 TestElementVolume ........" << std::flush;

    m.test_element_volume(true, VCellGP);

    if (myrank_ == 0 and screenoutput)
    {
      if (comm_) Core::Communication::barrier(comm_);
      t_diff = Teuchos::Time::wallTime() - t_start;

      printf("success! ( %10.4e secs )\n\n", t_diff);
    }
  }
  // ######################################################################################

}  // Cut::LevelSetIntersection::Cut

FOUR_C_NAMESPACE_CLOSE
