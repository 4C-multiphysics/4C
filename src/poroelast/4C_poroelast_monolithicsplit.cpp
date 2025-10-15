// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_poroelast_monolithicsplit.hpp"

#include "4C_adapter_fld_base_algorithm.hpp"
#include "4C_adapter_fld_poro.hpp"
#include "4C_adapter_str_fpsiwrapper.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_structure_aux.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

PoroElast::MonolithicSplit::MonolithicSplit(MPI_Comm comm, const Teuchos::ParameterList& timeparams,
    std::shared_ptr<Core::LinAlg::MapExtractor> porosity_splitter)
    : Monolithic(comm, timeparams, porosity_splitter)
{
  icoupfs_ = std::make_shared<Coupling::Adapter::Coupling>();

  evaluateinterface_ = false;

  fgcur_ = nullptr;
  ddiinc_ = nullptr;
  solipre_ = nullptr;
  ddginc_ = nullptr;
  solgpre_ = nullptr;
  duiinc_ = nullptr;
  solivelpre_ = nullptr;
  duginc_ = nullptr;
  solgvelpre_ = nullptr;

  fsibcmap_ = nullptr;
  fsibcextractor_ = nullptr;

  ddi_ = nullptr;
}

void PoroElast::MonolithicSplit::prepare_time_step()
{
  // call base class
  PoroElast::Monolithic::prepare_time_step();

  //  // call the predictor
  //  structure_field()->prepare_time_step();
  //  fluid_field()->prepare_time_step();

  if (evaluateinterface_)
  {
    // here we account for DBCs and preconditioning on the FSI-Interface. In both cases the
    // structure field decides, what to do (I don't think this is the best solution, but at least
    // the
    // easiest one)

    double timescale = fluid_field()->time_scaling();

    std::shared_ptr<const Core::LinAlg::Vector<double>> idispnp =
        structure_field()->interface()->extract_fsi_cond_vector(*structure_field()->dispnp());
    std::shared_ptr<const Core::LinAlg::Vector<double>> idispn =
        structure_field()->interface()->extract_fsi_cond_vector(*structure_field()->dispn());
    std::shared_ptr<const Core::LinAlg::Vector<double>> ivelnp =
        structure_field()->interface()->extract_fsi_cond_vector(*structure_field()->velnp());
    std::shared_ptr<Core::LinAlg::Vector<double>> ifvelnp =
        fluid_field()->extract_interface_velnp();
    std::shared_ptr<Core::LinAlg::Vector<double>> ifveln = fluid_field()->extract_interface_veln();

    ddi_->update(1.0, *idispnp, -1.0, *idispn, 0.0);
    ddi_->update(-1.0, *ifveln, timescale);

    if (fsibcmap_->num_global_elements())
    {
      // if there are DBCs on FSI conditioned nodes, they have to be treated separately

      std::shared_ptr<Core::LinAlg::Vector<double>> ibcveln =
          fsibcextractor_->extract_cond_vector(*structure_to_fluid_at_interface(*ivelnp));
      std::shared_ptr<Core::LinAlg::Vector<double>> inobcveln =
          fsibcextractor_->extract_other_vector(*structure_to_fluid_at_interface(*ddi_));

      // DBCs at FSI-Interface
      fsibcextractor_->insert_cond_vector(*ibcveln, *ifvelnp);
      // any preconditioned values at the FSI-Interface
      fsibcextractor_->insert_other_vector(*inobcveln, *ifvelnp);
    }
    else
      // no DBCs on FSI interface -> just make preconditioners consistent (structure decides)
      ifvelnp = structure_to_fluid_at_interface(*ddi_);

    fluid_field()->apply_interface_velocities(ifvelnp);
  }
}

std::shared_ptr<Core::LinAlg::Vector<double>>
PoroElast::MonolithicSplit::structure_to_fluid_at_interface(
    const Core::LinAlg::Vector<double>& iv) const
{
  return icoupfs_->master_to_slave(iv);
}

std::shared_ptr<Core::LinAlg::Vector<double>>
PoroElast::MonolithicSplit::fluid_to_structure_at_interface(
    const Core::LinAlg::Vector<double>& iv) const
{
  return icoupfs_->slave_to_master(iv);
}

std::shared_ptr<Core::LinAlg::Map> PoroElast::MonolithicSplit::fsidbc_map()
{
  TEUCHOS_FUNC_TIME_MONITOR("PoroElast::MonolithicSplit::FSIDBCMap");

  // get interface map and DBC map of fluid
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> fluidmaps;
  fluidmaps.push_back(fluid_field()->interface()->fsi_cond_map());
  fluidmaps.push_back(fluid_field()->get_dbc_map_extractor()->cond_map());

  // build vector of dbc and fsi coupling of fluid field
  std::shared_ptr<Core::LinAlg::Map> fluidfsibcmap =
      Core::LinAlg::MultiMapExtractor::intersect_maps(fluidmaps);

  if (fluidfsibcmap->num_my_elements())
    FOUR_C_THROW("Dirichlet boundary conditions on fluid and FSI interface not supported!!");

  // get interface map and DBC map of structure
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> structmaps;
  structmaps.push_back(structure_field()->interface()->fsi_cond_map());
  structmaps.push_back(structure_field()->get_dbc_map_extractor()->cond_map());

  // vector of dbc and fsi coupling of structure field
  std::shared_ptr<Core::LinAlg::Map> structfsibcmap =
      Core::LinAlg::MultiMapExtractor::intersect_maps(structmaps);

  Core::LinAlg::Vector<double> gidmarker_struct(
      *structure_field()->interface()->fsi_cond_map(), true);

  // Todo this is ugly, fix it!
  const int mylength = structfsibcmap->num_my_elements();  // on each processor (lids)
  const int* mygids = structfsibcmap->my_global_elements();

  // mark gids with fsi and DBC Condition
  for (int i = 0; i < mylength; ++i)
  {
    int gid = mygids[i];
    // FOUR_C_ASSERT(slavemastermap.count(gid),"master gid not found on slave side");
    gidmarker_struct.replace_global_value(gid, 1.0);
  }

  // transfer to fluid side
  std::shared_ptr<Core::LinAlg::Vector<double>> gidmarker_fluid =
      structure_to_fluid_at_interface(gidmarker_struct);

  std::vector<int> structfsidbcvector;
  const int numgids = gidmarker_fluid->local_length();  // on each processor (lids)
  double* mygids_fluid = gidmarker_fluid->get_values();
  const int* fluidmap = gidmarker_fluid->get_map().my_global_elements();
  for (int i = 0; i < numgids; ++i)
  {
    double val = mygids_fluid[i];
    if (val == 1.0) structfsidbcvector.push_back(fluidmap[i]);
  }

  std::shared_ptr<Core::LinAlg::Map> structfsidbcmap = std::make_shared<Core::LinAlg::Map>(
      -1, structfsidbcvector.size(), structfsidbcvector.data(), 0, get_comm());
  // FOUR_C_ASSERT(fluidfsidbcmap->UniqueGIDs(),"fsidbcmap is not unique!");

  return structfsidbcmap;
}

void PoroElast::MonolithicSplit::setup_coupling_and_matrices()
{
  const int ndim = Global::Problem::instance()->n_dim();
  icoupfs_->setup_condition_coupling(*structure_field()->discretization(),
      structure_field()->interface()->fsi_cond_map(), *fluid_field()->discretization(),
      fluid_field()->interface()->fsi_cond_map(), "FSICoupling", ndim);

  fsibcmap_ = fsidbc_map();

  evaluateinterface_ = structure_field()->interface()->fsi_cond_relevant();

  if (evaluateinterface_)
  {
    if (fsibcmap_->num_global_elements())
    {
      const std::shared_ptr<Adapter::FluidPoro>& fluidfield =
          std::dynamic_pointer_cast<Adapter::FluidPoro>(fluid_field());
      fluidfield->add_dirich_cond(fsibcmap_);

      fsibcextractor_ = std::make_shared<Core::LinAlg::MapExtractor>(
          *fluid_field()->interface()->fsi_cond_map(), fsibcmap_);
    }

    std::shared_ptr<const Core::LinAlg::Vector<double>> idispnp =
        structure_field()->interface()->extract_fsi_cond_vector(*structure_field()->dispnp());
    ddi_ = std::make_shared<Core::LinAlg::Vector<double>>(idispnp->get_map(), true);
  }

  // initialize Poroelasticity-systemmatrix_
  systemmatrix_ =
      std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          *extractor(), *extractor(), 81, false, true);

  // initialize coupling matrices
  k_fs_ =
      std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          *(structure_field()->interface()), *(fluid_field()->interface()), 81, false, true);

  k_sf_ =
      std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          *(fluid_field()->interface()), *(structure_field()->interface()), 81, false, true);
}

void PoroElast::MonolithicSplit::build_combined_dbc_map()
{
  TEUCHOS_FUNC_TIME_MONITOR("PoroElast::MonolithicSplit::combined_dbc_map");

  // first, get DBC-maps from structure and fluid field and merge them to one map
  const std::shared_ptr<const Core::LinAlg::Map> scondmap =
      structure_field()->get_dbc_map_extractor()->cond_map();
  const std::shared_ptr<const Core::LinAlg::Map> fcondmap =
      fluid_field()->get_dbc_map_extractor()->cond_map();

  std::vector<std::shared_ptr<const Core::LinAlg::Map>> vectoroverallfsimaps;
  vectoroverallfsimaps.push_back(scondmap);
  vectoroverallfsimaps.push_back(fcondmap);

  std::shared_ptr<Core::LinAlg::Map> overallfsidbcmaps =
      Core::LinAlg::MultiMapExtractor::merge_maps(vectoroverallfsimaps);

  // now we intersect the global dof map with the DBC map to get all dofs with DBS applied, which
  // are in the global
  // system, i.e. are not condensed
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> vectordbcmaps;
  vectordbcmaps.emplace_back(overallfsidbcmaps);
  vectordbcmaps.emplace_back(fullmap_);

  combinedDBCMap_ = Core::LinAlg::MultiMapExtractor::intersect_maps(vectordbcmaps);
}

void PoroElast::MonolithicSplit::solve()
{
  // solve monolithic system by newton iteration
  Monolithic::solve();

  // recover Lagrange multiplier \lambda_\Gamma at the interface at the end of each time step
  // (i.e. condensed forces onto the structure) needed for rhs in next time step
  recover_lagrange_multiplier_after_time_step();
}

FOUR_C_NAMESPACE_CLOSE
