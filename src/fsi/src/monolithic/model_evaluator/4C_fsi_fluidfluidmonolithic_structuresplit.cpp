// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fsi_fluidfluidmonolithic_structuresplit.hpp"

#include "4C_adapter_ale_xffsi.hpp"
#include "4C_adapter_fld_fluid_fluid_fsi.hpp"
#include "4C_adapter_str_fsiwrapper.hpp"
#include "4C_ale_input.hpp"
#include "4C_ale_utils_mapextractor.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_fsi_input.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_io_pstream.hpp"
#include "4C_structure_aux.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
FSI::FluidFluidMonolithicStructureSplit::FluidFluidMonolithicStructureSplit(
    MPI_Comm comm, Global::Problem& problem, const Teuchos::ParameterList& timeparams)
    : MonolithicStructureSplit(comm, problem, timeparams)
{
  // cast to problem-specific fluid-wrapper
  fluid_ =
      std::dynamic_pointer_cast<Adapter::FluidFluidFSI>(MonolithicStructureSplit::fluid_field());

  // cast to problem-specific ALE-wrapper
  ale_ = std::dynamic_pointer_cast<Adapter::AleXFFsiWrapper>(MonolithicStructureSplit::ale_field());

  // XFFSI_Full_Newton is an invalid choice together with NOX,
  // because DOF-maps can change from one iteration step to the other (XFEM cut)
  if (fluid_field()->monolithic_xffsi_approach() == XFEM::XFFSI_Full_Newton)
    FOUR_C_THROW("NOX-based XFFSI Approach does not work with XFFSI_Full_Newton!");
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicStructureSplit::update()
{
  // time to relax the ALE-mesh?
  if (fluid_field()->is_ale_relaxation_step(step()))
  {
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      Core::IO::cout << "Relaxing Ale" << Core::IO::endl;

    ale_field()->solve();
    fluid_field()->apply_mesh_displacement(ale_to_fluid(ale_field()->dispnp()));
  }

  // update fields
  FSI::MonolithicStructureSplit::update();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicStructureSplit::prepare_time_step()
{
  // prepare time step on subsequent field & increment
  FSI::MonolithicStructureSplit::prepare_time_step();

  // when this is the first call or we haven't relaxed the ALE-mesh
  // previously, the DOF-maps have not
  // changed since system setup
  if (step() == 0 || !fluid_field()->is_ale_relaxation_step(step() - 1)) return;

  // REMARK:
  // as the new xfem-cut may lead to a change in the fluid dof-map,
  // we have to refresh the block system matrix,
  // rebuild the merged DOF map & update map extractor for combined
  // Dirichlet maps
  FSI::MonolithicStructureSplit::create_combined_dof_row_map();
  setup_dbc_map_extractor();
  FSI::MonolithicStructureSplit::create_system_matrix();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicStructureSplit::setup_dbc_map_extractor()
{
  // merge Dirichlet maps of structure, fluid and ALE to global FSI Dirichlet map
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> dbcmaps;

  // structure DBC
  dbcmaps.push_back(structure_field()->get_dbc_map_extractor()->cond_map());
  // fluid DBC (including background & embedded discretization)
  dbcmaps.push_back(fluid_field()->get_dbc_map_extractor()->cond_map());
  // ALE-DBC-maps, free of FSI DOF
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> aleintersectionmaps;
  aleintersectionmaps.push_back(ale_field()->get_dbc_map_extractor()->cond_map());
  aleintersectionmaps.push_back(ale_field()->interface()->other_map());
  std::shared_ptr<Core::LinAlg::Map> aleintersectionmap =
      Core::LinAlg::MultiMapExtractor::intersect_maps(aleintersectionmaps);
  dbcmaps.push_back(aleintersectionmap);

  std::shared_ptr<const Core::LinAlg::Map> dbcmap =
      Core::LinAlg::MultiMapExtractor::merge_maps(dbcmaps);

  // finally, create the global FSI Dirichlet map extractor
  dbcmaps_ = std::make_shared<Core::LinAlg::MapExtractor>(*dof_row_map(), dbcmap, true);
  if (dbcmaps_ == nullptr)
  {
    FOUR_C_THROW("Creation of Dirichlet map extractor failed.");
  }
}

FOUR_C_NAMESPACE_CLOSE
