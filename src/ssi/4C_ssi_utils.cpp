// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ssi_utils.hpp"

#include "4C_adapter_scatra_base_algorithm.hpp"
#include "4C_adapter_str_ssiwrapper.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_utils_gid_vector.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_fem_general_utils_createdis.hpp"
#include "4C_geometric_search_matchingoctree.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_s2i.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_scatra_timint_implicit.hpp"
#include "4C_scatra_timint_meshtying_strategy_s2i.hpp"
#include "4C_ssi_monolithic.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <utility>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int SSI::Utils::check_time_stepping(double dt1, double dt2)
{
  const double workdt1 = std::min(dt1, dt2);
  const double workdt2 = std::max(dt1, dt2);
  int i = 0;

  while (true)
  {
    i++;
    const double t1 = i * workdt1;

    if (std::abs(t1 - workdt2) < 10E-10) break;
    if (t1 > workdt2)
      FOUR_C_THROW("Chosen time steps {} and {} are not a multiplicative of each other", dt1, dt2);
  }
  return i;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::Utils::change_time_parameter(MPI_Comm comm, Teuchos::ParameterList& ssiparams,
    Teuchos::ParameterList& scatradyn, Teuchos::ParameterList& sdyn)
{
  // Create sub problems with different time steps
  if (ssiparams.get<bool>("DIFFTIMESTEPSIZE"))
  {
    // Check correct choice of time stepping for single fields
    double scatrastep = scatradyn.get<double>("TIMESTEP");
    double solidstep = sdyn.get<double>("TIMESTEP");

    check_time_stepping(scatrastep, solidstep);

    // modify global time step size
    ssiparams.set<double>("TIMESTEP", std::min(scatrastep, solidstep));
  }
  else
  {
    // -------------------------------------------------------------------
    // overrule certain parameters for coupled problems
    // -------------------------------------------------------------------
    // the default time step size
    scatradyn.set<double>("TIMESTEP", ssiparams.get<double>("TIMESTEP"));
    sdyn.set<double>("TIMESTEP", ssiparams.get<double>("TIMESTEP"));
    // maximum simulation time
    scatradyn.set<double>("MAXTIME", ssiparams.get<double>("MAXTIME"));
    sdyn.set<double>("MAXTIME", ssiparams.get<double>("MAXTIME"));
    // maximum number of timesteps
    scatradyn.set<int>("NUMSTEP", ssiparams.get<int>("NUMSTEP"));
    sdyn.set<int>("NUMSTEP", ssiparams.get<int>("NUMSTEP"));
  }

  // Check correct input of restart. Code relies that both time value RESTARTEVERYTIME and
  // RESULTSEVERYTIME are given if restart from time is applied
  double restarttime = ssiparams.get<double>("RESTARTEVERYTIME");
  double updatetime = ssiparams.get<double>("RESULTSEVERYTIME");
  if ((updatetime > 0.0) or (restarttime > 0.0))
  {
    if (updatetime <= 0.0 and restarttime <= 0.0)
    {
      FOUR_C_THROW(
          "If time controlled output and restart is desired, both parameters RESTARTEVERYTIME and "
          "RESULTSEVERYTIME has to be set");
    }
  }

  // set restart params
  int scatrarestart;
  int structurerestart;

  if (restarttime > 0.0)
  {
    scatrarestart = check_time_stepping(scatradyn.get<double>("TIMESTEP"), restarttime);
    structurerestart = check_time_stepping(sdyn.get<double>("TIMESTEP"), restarttime);
  }
  else
  {
    int restart = ssiparams.get<int>("RESTARTEVERY");
    scatrarestart = restart;
    structurerestart = restart;
  }

  // set output params
  int scatraupres;
  int structureupres;

  if (updatetime > 0.0)
  {
    scatraupres = check_time_stepping(scatradyn.get<double>("TIMESTEP"), updatetime);
    structureupres = check_time_stepping(sdyn.get<double>("TIMESTEP"), updatetime);
  }
  else
  {
    int update = ssiparams.get<int>("RESULTSEVERY");
    scatraupres = update;
    structureupres = update;
  }

  // restart
  scatradyn.set<int>("RESTARTEVERY", scatrarestart);
  sdyn.set<int>("RESTARTEVERY", structurerestart);
  // solution output
  scatradyn.set<int>("RESULTSEVERY", scatraupres);
  sdyn.set<int>("RESULTSEVERY", structureupres);

  if (Core::Communication::my_mpi_rank(comm) == 0)
  {
    std::cout << "====================== Overview of chosen time stepping: "
                 "==============================\n"
              << "\t Timestep scatra:           " << scatradyn.get<double>("TIMESTEP") << "\n"
              << "\t Timestep structure:        " << sdyn.get<double>("TIMESTEP") << "\n"
              << "\t Result step scatra:        " << scatradyn.get<int>("RESULTSEVERY") << "\n"
              << "\t Result step structure:     " << sdyn.get<int>("RESULTSEVERY") << "\n"
              << "\t Restart step scatra:       " << scatradyn.get<int>("RESTARTEVERY") << "\n"
              << "\t Restart step structure:    " << sdyn.get<int>("RESTARTEVERY") << "\n"
              << "================================================================================="
                 "=======\n \n";
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::ParameterList SSI::Utils::clone_scatra_manifold_params(
    const Teuchos::ParameterList& scatraparams,
    const Teuchos::ParameterList& sublist_manifold_params)
{
  Teuchos::ParameterList scatra_manifold_params(scatraparams);

  auto initial_field = Teuchos::getIntegralValue<Inpar::ScaTra::InitialField>(
      sublist_manifold_params, "INITIALFIELD");
  scatra_manifold_params.set("INITIALFIELD", initial_field);
  switch (initial_field)
  {
    case Inpar::ScaTra::initfield_zero_field:
    case Inpar::ScaTra::initfield_field_by_condition:
    {
      scatra_manifold_params.set<int>("INITFUNCNO", -1);
      break;
    }
    case Inpar::ScaTra::initfield_field_by_function:
    {
      scatra_manifold_params.set<int>("INITFUNCNO", sublist_manifold_params.get<int>("INITFUNCNO"));
      break;
    }
    default:
      FOUR_C_THROW("Initial field type on manifold not supported.");
  }

  if (Teuchos::getIntegralValue<Inpar::ScaTra::OutputScalarType>(scatraparams, "OUTPUTSCALARS") !=
      Inpar::ScaTra::outputscalars_none)
    scatra_manifold_params.set<bool>("output_file_name_discretization", true);

  scatra_manifold_params.set<bool>("OUTPUTSCALARSMEANGRAD", false);

  scatra_manifold_params.set<bool>("ADAPTIVE_TIMESTEPPING", false);

  return scatra_manifold_params;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::ParameterList SSI::Utils::modify_scatra_params(const Teuchos::ParameterList& scatraparams)
{
  auto scatraparams_mutable = Teuchos::ParameterList(scatraparams);

  if (Teuchos::getIntegralValue<Inpar::ScaTra::OutputScalarType>(scatraparams, "OUTPUTSCALARS") !=
      Inpar::ScaTra::outputscalars_none)
    scatraparams_mutable.set<bool>("output_file_name_discretization", true);

  return scatraparams_mutable;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
SSI::Utils::SSIMatrices::SSIMatrices(const SSIMaps& ssi_maps,
    const Core::LinAlg::MatrixType ssi_matrixtype, const Core::LinAlg::MatrixType scatra_matrixtype,
    const bool is_scatra_manifold)
    : is_scatra_manifold_(is_scatra_manifold),
      scatra_matrixtype_(scatra_matrixtype),
      scatra_dofrowmap_(ssi_maps.scatra_dof_row_map()),
      structure_dofrowmap_(ssi_maps.structure_dof_row_map())
{
  // fill maps related to scalar transport manifold if relevant
  if (is_scatra_manifold_) scatramanifold_dofrowmap_ = ssi_maps.scatra_manifold_dof_row_map();

  initialize_system_matrix(ssi_maps, ssi_matrixtype);

  initialize_main_diag_matrices(ssi_maps);

  initialize_off_diag_matrices(ssi_maps);
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
void SSI::Utils::SSIMatrices::initialize_main_diag_matrices(const SSIMaps& ssi_maps)
{
  structure_matrix_ = setup_sparse_matrix(*structure_dofrowmap_);

  switch (scatra_matrixtype_)
  {
    case Core::LinAlg::MatrixType::block_condition:
    case Core::LinAlg::MatrixType::block_condition_dof:
    {
      scatra_matrix_ =
          setup_block_matrix(*ssi_maps.block_map_scatra(), *ssi_maps.block_map_scatra());
      if (is_scatra_manifold_)
        manifold_matrix_ = setup_block_matrix(
            *ssi_maps.block_map_scatra_manifold(), *ssi_maps.block_map_scatra_manifold());

      break;
    }

    case Core::LinAlg::MatrixType::sparse:
    {
      scatra_matrix_ = setup_sparse_matrix(*scatra_dofrowmap_);

      if (is_scatra_manifold_) manifold_matrix_ = setup_sparse_matrix(*scatramanifold_dofrowmap_);

      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
    }
  }
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
void SSI::Utils::SSIMatrices::initialize_off_diag_matrices(const SSIMaps& ssi_maps)
{
  switch (scatra_matrixtype_)
  {
    case Core::LinAlg::MatrixType::block_condition:
    case Core::LinAlg::MatrixType::block_condition_dof:
    {
      scatra_structure_matrix_ =
          setup_block_matrix(*ssi_maps.block_map_scatra(), *ssi_maps.block_map_structure());

      structure_scatra_matrix_ =
          setup_block_matrix(*ssi_maps.block_map_structure(), *ssi_maps.block_map_scatra());

      if (is_scatra_manifold_)
      {
        scatramanifold_structure_matrix_ = setup_block_matrix(
            *ssi_maps.block_map_scatra_manifold(), *ssi_maps.block_map_structure());
        scatramanifold_scatra_matrix_ =
            setup_block_matrix(*ssi_maps.block_map_scatra_manifold(), *ssi_maps.block_map_scatra());
        scatra_scatramanifold_matrix_ =
            setup_block_matrix(*ssi_maps.block_map_scatra(), *ssi_maps.block_map_scatra_manifold());
      }

      break;
    }

    case Core::LinAlg::MatrixType::sparse:
    {
      scatra_structure_matrix_ = setup_sparse_matrix(*scatra_dofrowmap_);
      structure_scatra_matrix_ = setup_sparse_matrix(*structure_dofrowmap_);

      if (is_scatra_manifold_)
      {
        scatramanifold_structure_matrix_ = setup_sparse_matrix(*scatramanifold_dofrowmap_);
        scatramanifold_scatra_matrix_ = setup_sparse_matrix(*scatramanifold_dofrowmap_);
        scatra_scatramanifold_matrix_ = setup_sparse_matrix(*scatra_dofrowmap_);
      }

      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::Utils::SSIMatrices::complete_scatra_manifold_scatra_matrix() const
{
  switch (scatra_matrixtype_)
  {
    case Core::LinAlg::MatrixType::sparse:
      scatra_manifold_scatra_matrix()->complete(*scatra_dofrowmap_, *scatramanifold_dofrowmap_);
      break;
    case Core::LinAlg::MatrixType::block_condition:
    case Core::LinAlg::MatrixType::block_condition_dof:
      scatra_manifold_scatra_matrix()->complete();
      break;
    default:
      FOUR_C_THROW("Not supported Core::LinAlg::MatrixType!");
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::Utils::SSIMatrices::complete_scatra_manifold_structure_matrix() const
{
  switch (scatra_matrixtype_)
  {
    case Core::LinAlg::MatrixType::sparse:
      scatra_manifold_structure_matrix()->complete(
          *structure_dofrowmap_, *scatramanifold_dofrowmap_);
      break;
    case Core::LinAlg::MatrixType::block_condition:
    case Core::LinAlg::MatrixType::block_condition_dof:
      scatra_manifold_structure_matrix()->complete();
      break;
    default:
      FOUR_C_THROW("Not supported Core::LinAlg::MatrixType!");
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::Utils::SSIMatrices::complete_scatra_scatra_manifold_matrix() const
{
  switch (scatra_matrixtype_)
  {
    case Core::LinAlg::MatrixType::sparse:
      scatra_scatra_manifold_matrix()->complete(*scatramanifold_dofrowmap_, *scatra_dofrowmap_);
      break;
    case Core::LinAlg::MatrixType::block_condition:
    case Core::LinAlg::MatrixType::block_condition_dof:
      scatra_scatra_manifold_matrix()->complete();
      break;
    default:
      FOUR_C_THROW("Not supported Core::LinAlg::MatrixType!");
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::Utils::SSIMatrices::complete_scatra_structure_matrix() const
{
  switch (scatra_matrixtype_)
  {
    case Core::LinAlg::MatrixType::sparse:
      scatra_structure_matrix()->complete(*structure_dofrowmap_, *scatra_dofrowmap_);
      break;
    case Core::LinAlg::MatrixType::block_condition:
    case Core::LinAlg::MatrixType::block_condition_dof:
      scatra_structure_matrix()->complete();
      break;
    default:
      FOUR_C_THROW("Not supported Core::LinAlg::MatrixType!");
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::Utils::SSIMatrices::complete_structure_scatra_matrix() const
{
  switch (scatra_matrixtype_)
  {
    case Core::LinAlg::MatrixType::sparse:
      structure_scatra_matrix()->complete(*scatra_dofrowmap_, *structure_dofrowmap_);
      break;
    case Core::LinAlg::MatrixType::block_condition:
    case Core::LinAlg::MatrixType::block_condition_dof:
      structure_scatra_matrix()->complete();
      break;
    default:
      FOUR_C_THROW("Not supported Core::LinAlg::MatrixType!");
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::Utils::SSIMatrices::clear_matrices() const
{
  system_matrix_->zero();
  scatra_matrix_->zero();
  scatra_structure_matrix_->zero();
  structure_scatra_matrix_->zero();
  structure_matrix_->zero();

  if (is_scatra_manifold_)
  {
    scatramanifold_structure_matrix_->zero();
    manifold_matrix_->zero();
    scatra_scatramanifold_matrix_->zero();
    scatramanifold_scatra_matrix_->zero();
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
SSI::Utils::SSIVectors::SSIVectors(const SSIMaps& ssi_maps, const bool is_scatra_manifold)
    : increment_(std::make_shared<Core::LinAlg::Vector<double>>(
          *ssi_maps.maps_sub_problems()->full_map(), true)),
      is_scatra_manifold_(is_scatra_manifold),
      manifold_residual_(is_scatra_manifold ? std::make_shared<Core::LinAlg::Vector<double>>(
                                                  *ssi_maps.scatra_manifold_dof_row_map(), true)
                                            : nullptr),
      residual_(std::make_shared<Core::LinAlg::Vector<double>>(
          *ssi_maps.maps_sub_problems()->full_map(), true)),
      scatra_residual_(
          std::make_shared<Core::LinAlg::Vector<double>>(*ssi_maps.scatra_dof_row_map(), true)),
      structure_residual_(
          std::make_shared<Core::LinAlg::Vector<double>>(*ssi_maps.structure_dof_row_map(), true))
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::Utils::SSIVectors::clear_increment() const { increment_->put_scalar(0.0); }

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::Utils::SSIVectors::clear_residuals() const
{
  residual_->put_scalar(0.0);
  scatra_residual_->put_scalar(0.0);
  structure_residual_->put_scalar(0.0);
  if (is_scatra_manifold_) manifold_residual_->put_scalar(0.0);
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
void SSI::Utils::SSIMatrices::initialize_system_matrix(
    const SSIMaps& ssi_maps, const Core::LinAlg::MatrixType ssi_matrixtype)
{
  switch (ssi_matrixtype)
  {
    case Core::LinAlg::MatrixType::block_field:
    {
      system_matrix_ = setup_block_matrix(
          *ssi_maps.block_map_system_matrix(), *ssi_maps.block_map_system_matrix());
      break;
    }

    case Core::LinAlg::MatrixType::sparse:
    {
      system_matrix_ = setup_sparse_matrix(*ssi_maps.map_system_matrix());
      break;
    }

    default:
    {
      FOUR_C_THROW("Type of global system matrix for scalar-structure interaction not recognized!");
    }
  }
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> SSI::Utils::SSIMatrices::setup_block_matrix(
    const Core::LinAlg::MultiMapExtractor& row_map, const Core::LinAlg::MultiMapExtractor& col_map)
{
  constexpr int expected_entries_per_row = 81;
  constexpr bool explicitdirichlet = false;
  constexpr bool savegraph = true;

  return std::make_shared<
      Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
      col_map, row_map, expected_entries_per_row, explicitdirichlet, savegraph);
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> SSI::Utils::SSIMatrices::setup_sparse_matrix(
    const Core::LinAlg::Map& row_map)
{
  constexpr int expected_entries_per_row = 27;
  constexpr bool explicitdirichlet = false;
  constexpr bool savegraph = true;

  return std::make_shared<Core::LinAlg::SparseMatrix>(
      row_map, expected_entries_per_row, explicitdirichlet, savegraph);
}

/* ----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
SSI::Utils::SSIMaps::SSIMaps(const SsiMono& ssi_mono_algorithm)
    : scatra_matrixtype_(ssi_mono_algorithm.scatra_field()->matrix_type()),
      scatra_manifold_matrixtype_(ssi_mono_algorithm.is_scatra_manifold()
                                      ? ssi_mono_algorithm.scatra_manifold()->matrix_type()
                                      : Core::LinAlg::MatrixType::undefined),
      ssi_matrixtype_(ssi_mono_algorithm.matrix_type())
{
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> partial_maps(
      ssi_mono_algorithm.is_scatra_manifold() ? 3 : 2, nullptr);
  std::shared_ptr<const Core::LinAlg::Map> merged_map;

  partial_maps[get_problem_position(Subproblem::scalar_transport)] =
      std::make_shared<Core::LinAlg::Map>(*ssi_mono_algorithm.scatra_field()->dof_row_map());
  partial_maps[get_problem_position(Subproblem::structure)] =
      std::make_shared<Core::LinAlg::Map>(*ssi_mono_algorithm.structure_field()->dof_row_map());
  if (ssi_mono_algorithm.is_scatra_manifold())
  {
    partial_maps[get_problem_position(Subproblem::manifold)] =
        std::make_shared<Core::LinAlg::Map>(*ssi_mono_algorithm.scatra_manifold()->dof_row_map());
    auto temp_map = merge_map(partial_maps[0], partial_maps[1], false);
    merged_map = merge_map(temp_map, partial_maps[2], false);
  }
  else
  {
    merged_map = merge_map(partial_maps[0], partial_maps[1], false);
  }

  maps_sub_problems_ = std::make_shared<Core::LinAlg::MultiMapExtractor>(*merged_map, partial_maps);
  // check global map extractor
  maps_sub_problems_->check_for_valid_map_extractor();

  switch (ssi_matrixtype_)
  {
    case Core::LinAlg::MatrixType::block_field:
    {
      auto block_map_structure = std::make_shared<Core::LinAlg::MultiMapExtractor>(
          *ssi_mono_algorithm.structure_field()->discretization()->dof_row_map(),
          std::vector(1, ssi_mono_algorithm.structure_field()->dof_row_map()));

      block_map_structure->check_for_valid_map_extractor();

      block_maps_sub_problems_.insert(std::make_pair(Subproblem::structure, block_map_structure));

      switch (scatra_matrixtype_)
      {
        case Core::LinAlg::MatrixType::sparse:
        {
          auto block_map_scatra = std::make_shared<Core::LinAlg::MultiMapExtractor>(
              *ssi_mono_algorithm.scatra_field()->discretization()->dof_row_map(),
              std::vector(1, ssi_mono_algorithm.scatra_field()->dof_row_map()));

          block_map_scatra->check_for_valid_map_extractor();

          block_maps_sub_problems_.insert(
              std::make_pair(Subproblem::scalar_transport, block_map_scatra));

          if (ssi_mono_algorithm.is_scatra_manifold())
          {
            auto block_map_scatra_manifold = std::make_shared<Core::LinAlg::MultiMapExtractor>(
                *ssi_mono_algorithm.scatra_manifold()->discretization()->dof_row_map(),
                std::vector(1, ssi_mono_algorithm.scatra_manifold()->dof_row_map()));

            block_map_scatra_manifold->check_for_valid_map_extractor();

            block_maps_sub_problems_.insert(
                std::make_pair(Subproblem::manifold, block_map_scatra_manifold));
          }
          break;
        }
        case Core::LinAlg::MatrixType::block_condition:
        case Core::LinAlg::MatrixType::block_condition_dof:
        {
          block_maps_sub_problems_.insert(std::make_pair(
              Subproblem::scalar_transport, ssi_mono_algorithm.scatra_field()->dof_block_maps()));

          if (ssi_mono_algorithm.is_scatra_manifold())
          {
            block_maps_sub_problems_.insert(std::make_pair(
                Subproblem::manifold, ssi_mono_algorithm.scatra_manifold()->dof_block_maps()));
          }
          else
          {
            block_maps_sub_problems_.insert(std::make_pair(Subproblem::manifold, nullptr));
          }

          break;
        }
        default:
        {
          FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
        }
      }

      create_and_check_block_maps_sub_problems(ssi_mono_algorithm);

      break;
    }

    case Core::LinAlg::MatrixType::sparse:
    {
      block_maps_sub_problems_.insert(std::make_pair(Subproblem::structure, nullptr));
      block_maps_sub_problems_.insert(std::make_pair(Subproblem::scalar_transport, nullptr));
      block_maps_sub_problems_.insert(std::make_pair(Subproblem::manifold, nullptr));
      break;
    }

    default:
    {
      FOUR_C_THROW("Type of global system matrix for scalar-structure interaction not recognized!");
    }
  }

  map_system_matrix_ = maps_sub_problems_->full_map();
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
std::vector<int> SSI::Utils::SSIMaps::get_block_positions(const Subproblem subproblem) const
{
  FOUR_C_ASSERT(
      ssi_matrixtype_ != Core::LinAlg::MatrixType::sparse, "Sparse matrices have just one block");

  std::vector<int> block_position;

  switch (subproblem)
  {
    case Subproblem::structure:
    {
      block_position.emplace_back(0);
      break;
    }
    case Subproblem::scalar_transport:
    {
      if (scatra_matrixtype_ == Core::LinAlg::MatrixType::sparse)
        block_position.emplace_back(1);
      else
      {
        for (int i = 0; i < block_map_scatra()->num_maps(); ++i) block_position.emplace_back(i + 1);
      }
      break;
    }
    case Subproblem::manifold:
    {
      if (scatra_manifold_matrixtype_ == Core::LinAlg::MatrixType::sparse)
        block_position.emplace_back(2);
      else
      {
        auto scatra_manifold_num_block_maps = block_map_scatra_manifold()->num_maps();

        for (int i = 0; i < scatra_manifold_num_block_maps; ++i)
          block_position.emplace_back(block_map_scatra()->num_maps() + 1 + i);
      }
      break;
    }
    default:
    {
      FOUR_C_THROW("Unknown type of subproblem");
    }
  }

  return block_position;
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
int SSI::Utils::SSIMaps::get_problem_position(const Subproblem subproblem)
{
  switch (subproblem)
  {
    case Subproblem::structure:
    {
      return 0;
    }
    case Subproblem::scalar_transport:
    {
      return 1;
    }
    case Subproblem::manifold:
    {
      return 2;
    }
    default:
    {
      FOUR_C_THROW("Unknown type of sub problem");
    }
  }
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
void SSI::Utils::SSIMaps::create_and_check_block_maps_sub_problems(
    const SsiMono& ssi_mono_algorithm)
{
  const int num_blocks_systemmatrix =
      block_map_scatra()->num_maps() + block_map_structure()->num_maps() +
      (ssi_mono_algorithm.is_scatra_manifold() ? block_map_scatra_manifold()->num_maps() : 0);

  std::vector<std::shared_ptr<const Core::LinAlg::Map>> partial_maps_system_matrix(
      num_blocks_systemmatrix, nullptr);

  for (int i = 0; i < block_map_scatra()->num_maps(); ++i)

  {
    auto block_positions_scatra = get_block_positions(Subproblem::scalar_transport);
    partial_maps_system_matrix[block_positions_scatra.at(i)] = block_map_scatra()->map(i);
  }

  partial_maps_system_matrix.at(get_block_positions(Subproblem::structure).at(0)) =
      block_map_structure()->full_map();

  if (ssi_mono_algorithm.is_scatra_manifold())
  {
    for (int i = 0; i < block_map_scatra_manifold()->num_maps(); ++i)
    {
      auto block_positions_manifold = get_block_positions(Subproblem::manifold);
      partial_maps_system_matrix[block_positions_manifold.at(i)] =
          block_map_scatra_manifold()->map(i);
    }
  }

  block_map_system_matrix_ = std::make_shared<Core::LinAlg::MultiMapExtractor>(
      *maps_sub_problems_->full_map(), partial_maps_system_matrix);

  block_map_system_matrix_->check_for_valid_map_extractor();
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::MultiMapExtractor> SSI::Utils::SSIMaps::block_map_scatra() const
{
  return block_maps_sub_problems_.at(Subproblem::scalar_transport);
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::MultiMapExtractor>
SSI::Utils::SSIMaps::block_map_scatra_manifold() const
{
  return block_maps_sub_problems_.at(Subproblem::manifold);
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::MultiMapExtractor> SSI::Utils::SSIMaps::block_map_structure()
    const
{
  return block_maps_sub_problems_.at(Subproblem::structure);
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map> SSI::Utils::SSIMaps::scatra_dof_row_map() const
{
  return maps_sub_problems()->map(get_problem_position(Subproblem::scalar_transport));
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map> SSI::Utils::SSIMaps::scatra_manifold_dof_row_map() const
{
  return maps_sub_problems()->map(get_problem_position(Subproblem::manifold));
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map> SSI::Utils::SSIMaps::structure_dof_row_map() const
{
  return maps_sub_problems()->map(get_problem_position(Subproblem::structure));
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
void SSI::Utils::check_consistency_of_ssi_interface_contact_condition(
    const std::vector<const Core::Conditions::Condition*>& conditionsToBeTested,
    const Core::FE::Discretization& structdis)
{
  // get conditions to check against
  std::vector<const Core::Conditions::Condition*> s2ikinetics_conditions;
  structdis.get_condition("S2IKinetics", s2ikinetics_conditions);
  std::vector<const Core::Conditions::Condition*> contactconditions;
  structdis.get_condition("Contact", contactconditions);

  // loop over all ssi conditions and check them
  for (const auto* conditionToBeTested : conditionsToBeTested)
  {
    std::vector<const Core::Conditions::Condition*> InterfaceS2IConditions;
    std::vector<const Core::Conditions::Condition*> InterfaceContactConditions;

    const int S2IKineticsID = conditionToBeTested->parameters().get<int>("S2I_KINETICS_ID");
    const int contactconditionID =
        conditionToBeTested->parameters().get<int>("CONTACT_CONDITION_ID");

    if (S2IKineticsID != contactconditionID)
    {
      FOUR_C_THROW(
          "For the 'SSIInterfaceContact' condition we have to demand, that the 'S2ICouplingID' and "
          "the 'CONTACT_CONDITION_ID' have the same value as the contact strategy factory relies "
          "on this to set the scatra interface parameters correctly.");
    }

    // loop over all scatra-scatra interface conditions and add them to the vector if IDs match
    for (const auto* s2ikinetics_cond : s2ikinetics_conditions)
    {
      if (s2ikinetics_cond->parameters().get<int>("ConditionID") != S2IKineticsID) continue;

      InterfaceS2IConditions.push_back(s2ikinetics_cond);
    }

    // loop over all contact conditions and add them to the vector if IDs match
    for (const auto* contactcondition : contactconditions)
    {
      if (contactcondition->parameters().get<int>("InterfaceID") != contactconditionID) continue;

      InterfaceContactConditions.push_back(contactcondition);
    }

    if (InterfaceContactConditions.empty())
      FOUR_C_THROW(
          "Did not find 'Contact' condition as defined in 'SSIInterfaceContact' condition!");

    if (InterfaceS2IConditions.empty())
      FOUR_C_THROW(
          "Did not find 'S2ICoupling' condition as defined in 'SSIInterfaceContact' condition!");

    // now get the nodes
    auto InterfaceS2INodes = find_conditioned_node_ids(
        structdis, InterfaceS2IConditions, Core::Conditions::LookFor::locally_owned);
    auto InterfaceContactNodes = find_conditioned_node_ids(
        structdis, InterfaceContactConditions, Core::Conditions::LookFor::locally_owned);

    // and compare whether same nodes are defined
    for (const auto InterfaceS2INode : InterfaceS2INodes)
    {
      bool foundit(false);

      for (const auto InterfaceContactNode : InterfaceContactNodes)
      {
        if (InterfaceS2INode == InterfaceContactNode) foundit = true;
      }

      if (!foundit)
      {
        FOUR_C_THROW(
            "The following conditions are defined on a non-consistent set of nodes!\n"
            "The Conditions defined from 'SSIInterfaceContact' condition with the condition-ID: "
            "{}:\n"
            "'S2ICoupling' conditions with ID: {};\n"
            "'Contact' conditions with ID: {};\n"
            "The last two conditions are NOT defined on the same node-sets which is not "
            "reasonable. Check your Input-File!",
            conditionToBeTested->parameters().get<int>("ConditionID"), S2IKineticsID,
            contactconditionID);
      }
    }
  }
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
SSI::Utils::SSIMeshTying::SSIMeshTying(const std::string& conditionname_coupling,
    const Core::FE::Discretization& dis, const bool build_slave_slave_transformation,
    const bool check_over_constrained)
    : comm_(dis.get_comm()),
      do_print_(Core::Communication::my_mpi_rank(dis.get_comm()) == 0),
      my_rank_(Core::Communication::my_mpi_rank(dis.get_comm())),
      num_proc_(Core::Communication::num_mpi_ranks(dis.get_comm()))
{
  setup_mesh_tying_handlers(
      dis, conditionname_coupling, build_slave_slave_transformation, check_over_constrained);

  // construct full slave, master, and interior maps
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> slave_maps;
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> master_maps;
  for (const auto& meshtying : meshtying_handlers_)
  {
    slave_maps.emplace_back(meshtying->slave_master_coupling()->slave_dof_map());
    master_maps.emplace_back(meshtying->slave_master_coupling()->master_dof_map());
  }

  full_master_side_map_ = Core::LinAlg::MultiMapExtractor::merge_maps(master_maps);
  full_slave_side_map_ = Core::LinAlg::MultiMapExtractor::merge_maps(slave_maps);
  auto interface_map = merge_map(full_master_side_map_, full_slave_side_map_);

  interior_map_ = split_map(*dis.dof_row_map(), *interface_map);
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
void SSI::Utils::SSIMeshTying::setup_mesh_tying_handlers(const Core::FE::Discretization& dis,
    const std::string& name_meshtying_condition, const bool build_slave_slave_transformation,
    const bool check_over_constrained)
{
  Teuchos::Time timer("MeshtypingHandlers", true);
  const double t0 = timer.wallTime();

  if (do_print_)
  {
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << '\n';
    std::cout << "| Starting to setup mesh tying   |" << '\n';
  }

  // find pairwise matching nodes
  std::vector<std::pair<int, int>> coupling_pairs;
  find_matching_node_pairs(dis, name_meshtying_condition, coupling_pairs);

  // all nodes on one geometrical point: outer vector defines geometric position, and inner vector
  // all nodes there
  std::vector<std::vector<int>> grouped_matching_nodes;
  group_matching_nodes(coupling_pairs, grouped_matching_nodes);

  if (do_print_) std::cout << "| Finished: group nodes          |" << '\n';

  // define master gids and pair them with the remaining (slave) nodes
  std::vector<int> master_gids;
  std::map<int, int> slave_master_pair;
  define_master_slave_pairing(
      dis, grouped_matching_nodes, master_gids, slave_master_pair, check_over_constrained);

  if (do_print_) std::cout << "| Finished: master-slave pairing |" << '\n';

  // get number of slave nodes per master node -> max. number gives number of needed adapters
  std::map<int, int> num_assigned_slave_to_master_nodes;
  int glob_max_adapters = 0;
  get_num_assigned_slave_to_master_nodes(
      slave_master_pair, num_assigned_slave_to_master_nodes, glob_max_adapters);

  // setup mesh tying:
  // - coupling adapter
  // - map extractor
  // - slave_slave_transformation
  std::map<int, int> created_adapters;  // numbers of created coupling adapters per master node
  for (const auto& key : num_assigned_slave_to_master_nodes | std::views::keys)
    created_adapters.insert(std::make_pair(key, 0));

  for (int iadapter = 0; iadapter < glob_max_adapters; ++iadapter)
  {
    // create vectors of master and slave nodes for this coupling adapter
    std::vector<int> inodegidvec_master;
    std::vector<int> inodegidvec_slave;

    for (auto& pair : slave_master_pair)
    {
      const int master_gid = pair.second;
      if (master_gid != -1)  // this pair is already considered
      {
        const int num_created_adapters = created_adapters[master_gid];
        if (num_assigned_slave_to_master_nodes[master_gid] >= iadapter + 1 and
            num_created_adapters == iadapter)
        {
          const int slave_gid = pair.first;
          Core::Communication::add_owned_node_gid(dis, master_gid, inodegidvec_master);
          Core::Communication::add_owned_node_gid(dis, slave_gid, inodegidvec_slave);

          pair.second = -1;  // do not consider this pair in next iteration
          created_adapters[master_gid] = num_created_adapters + 1;
        }
      }
    }

    // setup coupling adapter
    auto coupling_adapter = std::make_shared<Coupling::Adapter::Coupling>();
    const int num_dofs =
        static_cast<int>(static_cast<double>(dis.dof_row_map()->num_global_elements()) /
                         static_cast<double>(dis.node_row_map()->num_global_elements()));
    coupling_adapter->setup_coupling(
        dis, dis, inodegidvec_master, inodegidvec_slave, num_dofs, true, 1.0e-8);

    // setup multimap extractor for each coupling adapter
    auto slave_map = coupling_adapter->slave_dof_map();
    auto master_map = coupling_adapter->master_dof_map();
    auto interior_map = split_map(*dis.dof_row_map(), *merge_map(slave_map, master_map));

    std::vector<std::shared_ptr<const Core::LinAlg::Map>> maps;
    maps.emplace_back(interior_map);
    maps.emplace_back(slave_map);
    maps.emplace_back(master_map);

    auto coupling_map_extractor =
        std::make_shared<Core::LinAlg::MultiMapExtractor>(*dis.dof_row_map(), maps);
    coupling_map_extractor->check_for_valid_map_extractor();

    auto slave_slave_transformation = std::make_shared<Coupling::Adapter::Coupling>();
    if (build_slave_slave_transformation)
    {
      // coupling adapter between new slave nodes (master) and old slave nodes from input file
      // (slave). Needed, to map the lineariaztion dphi/du at the interface to the correct dofs
      std::vector<int> all_coupled_original_slave_gids;
      find_slave_slave_transformation_nodes(
          dis, name_meshtying_condition, inodegidvec_slave, all_coupled_original_slave_gids);

      std::vector<int> my_coupled_original_slave_gids;
      for (const int& slave_gid : all_coupled_original_slave_gids)
        Core::Communication::add_owned_node_gid(dis, slave_gid, my_coupled_original_slave_gids);

      slave_slave_transformation->setup_coupling(dis, dis, inodegidvec_slave,
          my_coupled_original_slave_gids, Global::Problem::instance()->n_dim(), true, 1.0e-8);
    }

    // combine coupling adapters and multimap extractor to mesh tying object
    meshtying_handlers_.emplace_back(std::make_shared<SSIMeshTyingHandler>(
        coupling_adapter, coupling_map_extractor, slave_slave_transformation));
  }

  const double total_time = timer.wallTime() - t0;
  if (do_print_)
  {
    std::cout << "|--------------------------------|" << '\n';
    std::cout << "|  Mesh tying setup successful   |" << '\n';
    std::cout << std::scientific << std::setprecision(3);
    std::cout << "|        (" << total_time << " sec.)        |" << '\n';
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << '\n' << '\n';
  }
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
int SSI::Utils::SSIMeshTying::has_gid(
    const int gid, const std::vector<std::vector<int>>& matching_nodes) const
{
  const int size = static_cast<int>(matching_nodes.size());
  const int load_per_proc = std::floor(static_cast<double>(size) / static_cast<double>(num_proc_));

  // define load of this proc
  const int upper_bound = my_rank_ == num_proc_ - 1 ? size : load_per_proc * (my_rank_ + 1);
  const int lower_bound = load_per_proc * my_rank_;

  const int my_return_value = has_gid_partial(gid, lower_bound, upper_bound, matching_nodes);

  return Core::Communication::max_all(my_return_value, comm_);
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
int SSI::Utils::SSIMeshTying::has_gid_partial(const int gid, const int start, const int end,
    const std::vector<std::vector<int>>& matching_nodes) const
{
  for (int i = start; i < end; ++i)
  {
    for (const int& j : matching_nodes[i])
      if (gid == j) return i;
  }
  return -1;
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
void SSI::Utils::SSIMeshTying::find_matching_node_pairs(const Core::FE::Discretization& dis,
    const std::string& name_meshtying_condition,
    std::vector<std::pair<int, int>>& coupling_pairs) const
{
  // coupled nodes on this proc from input
  std::vector<std::pair<int, int>> my_coupling_pairs;

  // get all mesh tying conditions
  std::vector<const Core::Conditions::Condition*> meshtying_conditions;
  dis.get_condition(name_meshtying_condition, meshtying_conditions);

  // match nodes between all mesh tying conditions (named with "a" and "b")
  for (std::size_t a = 0; a < meshtying_conditions.size(); ++a)
  {
    const auto* meshtying_condition_a = meshtying_conditions.at(a);

    // nodes of meshtying_condition_a owned by this proc
    std::vector<int> inodegidvec_a;
    Core::Communication::add_owned_node_gid_from_list(
        dis, *meshtying_condition_a->get_nodes(), inodegidvec_a);

    // init node matching octree with nodes from condition a
    auto tree = Core::GeometricSearch::NodeMatchingOctree();
    tree.init(dis, inodegidvec_a, 150, 1.0e-8);
    tree.setup();

    // find nodes from condition b that match nodes from condition a
    for (std::size_t b = a + 1; b < meshtying_conditions.size(); ++b)
    {
      const auto* meshtying_condition_b = meshtying_conditions.at(b);

      // nodes of meshtying_condition_b owned by this proc
      std::vector<int> inodegidvec_b;
      Core::Communication::add_owned_node_gid_from_list(
          dis, *meshtying_condition_b->get_nodes(), inodegidvec_b);

      // key: master node gid, value: slave node gid and distance
      std::map<int, std::pair<int, double>> coupled_gid_nodes;
      tree.find_match(dis, inodegidvec_b, coupled_gid_nodes);

      // loop over all nodal couplings and find coupled nodes
      for (const auto& pair : coupled_gid_nodes)
      {
        const double distance = pair.second.second;
        if (distance < 1.0e-16)
        {
          const int gid1 = pair.first;
          const int gid2 = pair.second.first;

          my_coupling_pairs.emplace_back(gid1, gid2);
        }
      }
    }
  }

  // communicate to all other procs
  const auto all_coupling_pairs = Core::Communication::all_reduce(my_coupling_pairs, comm_);

  // remove duplicates (slave node = master node)
  for (const auto& pair : all_coupling_pairs)
    if (pair.first != pair.second) coupling_pairs.emplace_back(pair);
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
void SSI::Utils::SSIMeshTying::group_matching_nodes(
    const std::vector<std::pair<int, int>>& coupling_pairs,
    std::vector<std::vector<int>>& grouped_matching_nodes) const
{
  // all nodes on one geometrical point: outer vector defines geometric position, and inner vector
  // all nodes there
  for (const auto& [gid1, gid2] : coupling_pairs)
  {
    std::vector<int> nodes_at_same_point;

    // initial fill of grouped_matching_nodes
    if (grouped_matching_nodes.empty())
    {
      nodes_at_same_point.resize(2);
      nodes_at_same_point[0] = gid1;
      nodes_at_same_point[1] = gid2;
      grouped_matching_nodes.emplace_back(nodes_at_same_point);
    }
    else
    {
      // check if gid1 or gid2 is already part of the matching_nodes vector
      const int index1 = has_gid(gid1, grouped_matching_nodes);
      const int index2 = has_gid(gid2, grouped_matching_nodes);

      // do not add, if both indices are equal, meaning that the different gids are already in the
      // same group
      if (index1 == index2 and index1 != -1)
      {
        continue;
      }
      // if both gids are not part of matching_nodes -> create new entry
      else if (index1 == -1 and index2 == -1)
      {
        nodes_at_same_point.resize(2);
        nodes_at_same_point[0] = gid1;
        nodes_at_same_point[1] = gid2;
        grouped_matching_nodes.emplace_back(nodes_at_same_point);
      }
      // if gid1 is part of matching_nodes and gid2 is not -> add gid2 to entry of gid1
      else if (index1 != -1 and index2 == -1)
      {
        grouped_matching_nodes[index1].emplace_back(gid2);
      }
      // if gid2 is part of matching_nodes and gid1 is not -> add gid1 to entry of gid2
      else if (index1 == -1 and index2 != -1)
      {
        grouped_matching_nodes[index2].emplace_back(gid1);
      }
      // if gid1 and gid2 are part of matching_nodes -> remove nodes from entry of gid1 and move
      // them to entry of gid2
      else if (index1 != -1 and index2 != -1 and index1 != index2)
      {
        for (const int& node : grouped_matching_nodes[index1])
          grouped_matching_nodes[index2].emplace_back(node);

        grouped_matching_nodes.erase(std::next(grouped_matching_nodes.begin(), index1));
      }
      else
      {
        FOUR_C_THROW("This should not be the case");
      }
    }
  }
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
void SSI::Utils::SSIMeshTying::get_num_assigned_slave_to_master_nodes(
    const std::map<int, int>& slave_master_pair,
    std::map<int, int>& num_assigned_slave_to_master_nodes, int& max_assigned_slave_nodes) const
{
  for (const auto& val : slave_master_pair | std::views::values)
  {
    const int master_node_gid = val;

    // initial fill
    if (num_assigned_slave_to_master_nodes.empty())
    {
      num_assigned_slave_to_master_nodes.insert(std::make_pair(master_node_gid, 1));
      max_assigned_slave_nodes = 1;
    }
    else
    {
      // master node not in map so far -> create entry
      if (num_assigned_slave_to_master_nodes.contains(master_node_gid))
      {
        num_assigned_slave_to_master_nodes[master_node_gid]++;
        if (max_assigned_slave_nodes < num_assigned_slave_to_master_nodes[master_node_gid])
          max_assigned_slave_nodes = num_assigned_slave_to_master_nodes[master_node_gid];
      }
      // master node already in map-> increase counter by one
      else
      {
        num_assigned_slave_to_master_nodes.insert(std::make_pair(master_node_gid, 1));
      }
    }
  }
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
void SSI::Utils::SSIMeshTying::define_master_slave_pairing(const Core::FE::Discretization& dis,
    const std::vector<std::vector<int>>& grouped_matching_nodes, std::vector<int>& master_gids,
    std::map<int, int>& slave_master_pair, const bool check_over_constrained) const
{
  // get Dirichlet nodes -> they define the master side
  std::vector<const Core::Conditions::Condition*> dbc_conds;
  dis.get_condition("Dirichlet", dbc_conds);
  std::set<int> dbc_nodes;
  for (const auto* dbc_cond : dbc_conds)
    for (const int& dbc_node : *dbc_cond->get_nodes()) dbc_nodes.insert(dbc_node);

  std::vector<int> my_master_gids;
  std::map<int, int> my_slave_master_pair;

  const int size = static_cast<int>(grouped_matching_nodes.size());
  const int load_per_proc = std::ceil(static_cast<double>(size) / static_cast<double>(num_proc_));

  // define load for this proc in for loop
  const int upper_bound = my_rank_ == num_proc_ - 1 ? size : load_per_proc * (my_rank_ + 1);
  const int lower_bound = load_per_proc * my_rank_;

  for (int i = lower_bound; i < upper_bound; ++i)
  {
    const auto& nodes_at_same_point = grouped_matching_nodes[i];

    // define which of the nodes from nodes_at_same_point is master node
    // in case one node is a Dirichlet node -> use this one
    bool found_dbc_node = false;
    int new_master_gid = -1;
    for (const int& node : nodes_at_same_point)
    {
      for (const int& dbc_node : dbc_nodes)
      {
        if (dbc_node == node)
        {
          if (found_dbc_node and check_over_constrained)
            FOUR_C_THROW("Mesh tying and DBCs are over constrained.");
          else
          {
            found_dbc_node = true;
            new_master_gid = dbc_node;
            break;
          }
#ifndef FOUR_C_ENABLE_ASSERTIONS
          break;  // in release version, we can break here and subsequently skip the safety check
#endif
        }
      }
    }
    // otherwise -> use the first (random) node
    if (!found_dbc_node) new_master_gid = nodes_at_same_point[0];

    // insert new master node to list of master nodes
    my_master_gids.emplace_back(new_master_gid);

    // build slave-master pairs with defined master node
    for (const int& node : nodes_at_same_point)
      if (node != new_master_gid) my_slave_master_pair.insert(std::make_pair(node, new_master_gid));
  }

  master_gids = Core::Communication::all_reduce(my_master_gids, comm_);
  slave_master_pair = Core::Communication::all_reduce(my_slave_master_pair, comm_);

#ifdef FOUR_C_ENABLE_ASSERTIONS
  // check if everything worked fine
  std::set<int> unique_master_gids;
  for (const int& master_gid : master_gids) unique_master_gids.insert(master_gid);
  if (unique_master_gids.size() != master_gids.size()) FOUR_C_THROW("Master gids not unique");
#endif
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
void SSI::Utils::SSIMeshTying::find_slave_slave_transformation_nodes(
    const Core::FE::Discretization& dis, const std::string& name_meshtying_condition,
    const std::vector<int>& inodegidvec_slave,
    std::vector<int>& all_coupled_original_slave_gids) const
{
  // store nodes that are slave nodes from the input
  std::vector<const Core::Conditions::Condition*> meshtying_conditions;
  dis.get_condition(name_meshtying_condition, meshtying_conditions);

  std::vector<int> original_slave_gids;
  for (auto* meshtying_condition : meshtying_conditions)
  {
    if (meshtying_condition->parameters().get<Inpar::S2I::InterfaceSides>("INTERFACE_SIDE") ==
        Inpar::S2I::side_slave)
    {
      Core::Communication::add_owned_node_gid_from_list(
          dis, *meshtying_condition->get_nodes(), original_slave_gids);
    }
  }

  auto tree = Core::GeometricSearch::NodeMatchingOctree();
  tree.init(dis, inodegidvec_slave, 150, 1.0e-8);
  tree.setup();
  std::map<int, std::pair<int, double>> coupled_gid_nodes;
  tree.find_match(dis, original_slave_gids, coupled_gid_nodes);

  // find matches if distance < 1.0e-16
  std::vector<int> my_coupled_original_slave_gids;
  for (const auto& [gid, distance] : coupled_gid_nodes | std::views::values)
  {
    if (distance < 1.0e-16) my_coupled_original_slave_gids.emplace_back(gid);
  }

  // distribute gids from original slave nodes to all procs (matching might be on different proc)
  all_coupled_original_slave_gids =
      Core::Communication::all_reduce(my_coupled_original_slave_gids, comm_);
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
void SSI::Utils::SSIMeshTying::check_slave_side_has_dirichlet_conditions(
    std::shared_ptr<const Core::LinAlg::Map> struct_dbc_map) const
{
  // check if slave side dofs are part of DBC maps
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> maps(2, nullptr);
  maps[0] = std::move(struct_dbc_map);
  for (const auto& meshtying : meshtying_handlers_)
  {
    maps[1] = meshtying->slave_master_coupling()->slave_dof_map();
    if (Core::LinAlg::MultiMapExtractor::intersect_maps(maps)->num_global_elements() > 0)
      FOUR_C_THROW("Must not apply Dirichlet conditions to slave-side structural displacements!");
  }
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
SSI::Utils::SSIMeshTyingHandler::SSIMeshTyingHandler(
    std::shared_ptr<Coupling::Adapter::Coupling> slave_master_coupling,
    std::shared_ptr<Core::LinAlg::MultiMapExtractor> slave_master_extractor,
    std::shared_ptr<Coupling::Adapter::Coupling> slave_slave_transformation)
    : slave_master_coupling_(std::move(slave_master_coupling)),
      slave_master_extractor_(std::move(slave_master_extractor)),
      slave_slave_transformation_(std::move(slave_slave_transformation))
{
  slave_side_converter_ =
      std::make_shared<Coupling::Adapter::CouplingSlaveConverter>(*slave_master_coupling_);
}

FOUR_C_NAMESPACE_CLOSE
