// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fsi_fluidfluidmonolithic_fluidsplit.hpp"

#include "4C_adapter_ale_xffsi.hpp"
#include "4C_adapter_fld_fluid_fluid_fsi.hpp"
#include "4C_adapter_str_fsiwrapper.hpp"
#include "4C_ale_input.hpp"
#include "4C_ale_utils_mapextractor.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_constraint_manager.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_fsi.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_io_pstream.hpp"
#include "4C_structure_aux.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
FSI::FluidFluidMonolithicFluidSplit::FluidFluidMonolithicFluidSplit(
    MPI_Comm comm, const Teuchos::ParameterList& timeparams)
    : MonolithicFluidSplit(comm, timeparams)
{
  // cast to problem-specific fluid-wrapper
  fluid_ = std::dynamic_pointer_cast<Adapter::FluidFluidFSI>(MonolithicFluidSplit::fluid_field());

  // cast to problem-specific ALE-wrapper
  ale_ = std::dynamic_pointer_cast<Adapter::AleXFFsiWrapper>(MonolithicFluidSplit::ale_field());

  // XFFSI_Full_Newton is an invalid choice together with NOX,
  // because DOF-maps can change from one iteration step to the other (XFEM cut)
  if (fluid_field()->monolithic_xffsi_approach() == Inpar::XFEM::XFFSI_Full_Newton)
    FOUR_C_THROW("NOX-based XFFSI Approach does not work with XFFSI_Full_Newton!");
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicFluidSplit::update()
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
  FSI::MonolithicFluidSplit::update();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicFluidSplit::prepare_time_step()
{
  // prepare time step on subsequent field & increment
  FSI::MonolithicFluidSplit::prepare_time_step();

  // when this is the first call or we haven't relaxed the ALE-mesh
  // previously, the DOF-maps have not
  // changed since system setup
  if (step() == 0 || !fluid_field()->is_ale_relaxation_step(step() - 1)) return;

  // as the new xfem-cut may lead to a change in the fluid dof-map,
  // we have to refresh the block system matrix,
  // rebuild the merged DOF map & update map extractor for combined
  // Dirichlet maps
  FSI::MonolithicFluidSplit::create_combined_dof_row_map();
  setup_dbc_map_extractor();
  FSI::MonolithicFluidSplit::create_system_matrix();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicFluidSplit::setup_dbc_map_extractor()
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

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicFluidSplit::output()
{
  structure_field()->output();
  fluid_field()->output();

  // output Lagrange multiplier
  {
    // the Lagrange multiplier lives on the FSI interface
    // for output, we want to insert lambda into a full vector, defined on the embedded fluid field
    // 1. insert into vector containing all fluid DOF
    std::shared_ptr<Core::LinAlg::Vector<double>> lambdafull =
        fluid_field()->interface()->insert_fsi_cond_vector(
            *FSI::MonolithicFluidSplit::get_lambda());
    // 2. extract the embedded fluid part
    std::shared_ptr<Core::LinAlg::Vector<double>> lambdaemb =
        fluid_field()->x_fluid_fluid_map_extractor()->extract_fluid_vector(*lambdafull);

    const Teuchos::ParameterList& fsidyn = Global::Problem::instance()->fsi_dynamic_params();
    const int uprestart = fsidyn.get<int>("RESTARTEVERY");
    const int upres = fsidyn.get<int>("RESULTSEVERY");
    if ((uprestart != 0 && fluid_field()->step() % uprestart == 0) ||
        fluid_field()->step() % upres == 0)
      fluid_field()->disc_writer()->write_vector("fsilambda", lambdaemb);
  }
  ale_field()->output();

  if (structure_field()->get_constraint_manager()->have_monitor())
  {
    structure_field()->get_constraint_manager()->compute_monitor_values(
        *structure_field()->dispnp());
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      structure_field()->get_constraint_manager()->print_monitor_values();
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicFluidSplit::read_restart(int step)
{
  // Read Lagrange Multiplier (associated with embedded fluid)
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> lambdaemb =
        std::make_shared<Core::LinAlg::Vector<double>>(
            *(fluid_field()->x_fluid_fluid_map_extractor()->fluid_map()), true);
    Core::IO::DiscretizationReader reader = Core::IO::DiscretizationReader(
        fluid_field()->discretization(), Global::Problem::instance()->input_control_file(), step);
    reader.read_vector(lambdaemb, "fsilambda");
    // Insert into vector containing the whole merged fluid DOF
    std::shared_ptr<Core::LinAlg::Vector<double>> lambdafull =
        fluid_field()->x_fluid_fluid_map_extractor()->insert_fluid_vector(*lambdaemb);
    FSI::MonolithicFluidSplit::set_lambda(
        fluid_field()->interface()->extract_fsi_cond_vector(*lambdafull));
  }

  structure_field()->read_restart(step);
  fluid_field()->read_restart(step);
  ale_field()->read_restart(step);

  set_time_step(fluid_field()->time(), fluid_field()->step());
}

FOUR_C_NAMESPACE_CLOSE
