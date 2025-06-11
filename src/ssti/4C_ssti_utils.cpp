// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ssti_utils.hpp"

#include "4C_adapter_str_ssiwrapper.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_scatra_ele.hpp"
#include "4C_scatra_timint_implicit.hpp"
#include "4C_scatra_timint_meshtying_strategy_s2i.hpp"
#include "4C_ssi_utils.hpp"
#include "4C_ssti_monolithic.hpp"

FOUR_C_NAMESPACE_OPEN



/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
SSTI::SSTIMaps::SSTIMaps(const SSTI::SSTIMono& ssti_mono_algorithm)
    : block_map_scatra_(nullptr),
      block_map_structure_(nullptr),
      block_map_thermo_(nullptr),
      maps_subproblems_(nullptr)
{
  // setup maps containing dofs of subproblems
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> partial_maps(3, nullptr);
  partial_maps[ssti_mono_algorithm.get_problem_position(Subproblem::scalar_transport)] =
      std::make_shared<Core::LinAlg::Map>(*ssti_mono_algorithm.scatra_field()->dof_row_map());
  partial_maps[ssti_mono_algorithm.get_problem_position(Subproblem::structure)] =
      std::make_shared<Core::LinAlg::Map>(*ssti_mono_algorithm.structure_field()->dof_row_map());
  partial_maps[ssti_mono_algorithm.get_problem_position(Subproblem::thermo)] =
      std::make_shared<Core::LinAlg::Map>(*ssti_mono_algorithm.thermo_field()->dof_row_map());
  std::shared_ptr<const Core::LinAlg::Map> temp_map =
      Core::LinAlg::merge_map(partial_maps[0], partial_maps[1], false);
  std::shared_ptr<const Core::LinAlg::Map> merged_map =
      Core::LinAlg::merge_map(temp_map, partial_maps[2], false);
  // initialize global map extractor
  maps_subproblems_ = std::make_shared<Core::LinAlg::MultiMapExtractor>(*merged_map, partial_maps);
  // check global map extractor
  maps_subproblems_->check_for_valid_map_extractor();

  // initialize map extractors associated with blocks of subproblems
  block_map_structure_ = std::make_shared<Core::LinAlg::MultiMapExtractor>(
      *ssti_mono_algorithm.structure_field()->dof_row_map(),
      std::vector<std::shared_ptr<const Core::LinAlg::Map>>(
          1, ssti_mono_algorithm.structure_field()->dof_row_map()));
  switch (ssti_mono_algorithm.scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::sparse:
    {
      block_map_scatra_ = std::make_shared<Core::LinAlg::MultiMapExtractor>(
          *ssti_mono_algorithm.scatra_field()->dof_row_map(),
          std::vector<std::shared_ptr<const Core::LinAlg::Map>>(
              1, ssti_mono_algorithm.scatra_field()->dof_row_map()));
      block_map_thermo_ = std::make_shared<Core::LinAlg::MultiMapExtractor>(
          *ssti_mono_algorithm.thermo_field()->dof_row_map(),
          std::vector<std::shared_ptr<const Core::LinAlg::Map>>(
              1, ssti_mono_algorithm.thermo_field()->dof_row_map()));
      break;
    }
    case Core::LinAlg::MatrixType::block_condition:
    {
      block_map_scatra_ = ssti_mono_algorithm.scatra_field()->dof_block_maps();
      block_map_thermo_ = ssti_mono_algorithm.thermo_field()->dof_block_maps();
      break;
    }
    default:
    {
      FOUR_C_THROW("Matrix type not supported");
    }
  }

  block_map_scatra_->check_for_valid_map_extractor();
  block_map_structure_->check_for_valid_map_extractor();
  block_map_thermo_->check_for_valid_map_extractor();
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Map> SSTI::SSTIMaps::map_interface(
    std::shared_ptr<const ScaTra::MeshtyingStrategyS2I> meshtyingstrategy) const
{
  auto mergedInterfaceMap = Core::LinAlg::MultiMapExtractor::merge_maps(
      {meshtyingstrategy->coupling_adapter()->master_dof_map(),
          meshtyingstrategy->coupling_adapter()->slave_dof_map()});
  if (not mergedInterfaceMap->unique_gids()) FOUR_C_THROW("Map not unique");
  return mergedInterfaceMap;
}


/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::MultiMapExtractor> SSTI::SSTIMaps::maps_interface_blocks(
    std::shared_ptr<const ScaTra::MeshtyingStrategyS2I> meshtyingstrategy,
    Core::LinAlg::MatrixType scatramatrixtype, unsigned nummaps) const
{
  std::shared_ptr<Core::LinAlg::MultiMapExtractor> blockmapinterface(nullptr);

  std::shared_ptr<Core::LinAlg::Map> interfacemap = map_interface(meshtyingstrategy);

  switch (scatramatrixtype)
  {
    case Core::LinAlg::MatrixType::sparse:
    {
      blockmapinterface = std::make_shared<Core::LinAlg::MultiMapExtractor>(
          *interfacemap, std::vector<std::shared_ptr<const Core::LinAlg::Map>>(1, interfacemap));
      break;
    }
    case Core::LinAlg::MatrixType::block_condition:
    {
      std::vector<std::shared_ptr<const Core::LinAlg::Map>> partial_blockmapinterface(
          nummaps, nullptr);
      for (int iblockmap = 0; iblockmap < static_cast<int>(nummaps); ++iblockmap)
      {
        partial_blockmapinterface[iblockmap] = Core::LinAlg::MultiMapExtractor::merge_maps(
            {meshtyingstrategy->block_maps_slave().map(iblockmap),
                meshtyingstrategy->block_maps_master().map(iblockmap)});
      }
      blockmapinterface = std::make_shared<Core::LinAlg::MultiMapExtractor>(
          *interfacemap, partial_blockmapinterface);
      break;
    }
    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
      break;
    }
  }

  blockmapinterface->check_for_valid_map_extractor();

  return blockmapinterface;
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::MultiMapExtractor> SSTI::SSTIMaps::maps_interface_blocks_slave(
    const ScaTra::MeshtyingStrategyS2I& meshtyingstrategy,
    Core::LinAlg::MatrixType scatramatrixtype, unsigned nummaps) const
{
  std::shared_ptr<Core::LinAlg::MultiMapExtractor> blockmapinterfaceslave(nullptr);

  switch (scatramatrixtype)
  {
    case Core::LinAlg::MatrixType::sparse:
    {
      const auto slavedofmap = meshtyingstrategy.coupling_adapter()->slave_dof_map();
      blockmapinterfaceslave = std::make_shared<Core::LinAlg::MultiMapExtractor>(
          *slavedofmap, std::vector<std::shared_ptr<const Core::LinAlg::Map>>(1, slavedofmap));
      break;
    }
    case Core::LinAlg::MatrixType::block_condition:
    {
      blockmapinterfaceslave =
          std::make_shared<Core::LinAlg::MultiMapExtractor>(meshtyingstrategy.block_maps_slave());
      break;
    }
    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
      break;
    }
  }

  blockmapinterfaceslave->check_for_valid_map_extractor();

  return blockmapinterfaceslave;
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
SSTI::SSTIMapsMono::SSTIMapsMono(const SSTI::SSTIMono& ssti_mono_algorithm)
    : SSTIMaps(ssti_mono_algorithm), block_map_system_matrix_(nullptr)
{
  // initialize map extractors associated with blocks of global system matrix
  switch (ssti_mono_algorithm.scatra_field()->matrix_type())
  {
    // one single main-diagonal matrix block associated with scalar transport field
    case Core::LinAlg::MatrixType::sparse:
    {
      block_map_system_matrix_ = maps_sub_problems();
      break;
    }
      // many main-diagonal matrix blocks associated with scalar transport field
    case Core::LinAlg::MatrixType::block_condition:
    {
      auto block_positions_scatra =
          ssti_mono_algorithm.get_block_positions(Subproblem::scalar_transport);
      auto block_positions_structure =
          ssti_mono_algorithm.get_block_positions(Subproblem::structure);
      auto block_positions_thermo = ssti_mono_algorithm.get_block_positions(Subproblem::thermo);

      std::vector<std::shared_ptr<const Core::LinAlg::Map>> maps_systemmatrix(
          block_positions_scatra.size() + block_positions_structure.size() +
          block_positions_thermo.size());
      for (int imap = 0; imap < static_cast<int>(block_positions_scatra.size()); ++imap)
        maps_systemmatrix[block_positions_scatra.at(imap)] = block_map_scatra()->map(imap);

      // extract map underlying single main-diagonal matrix block associated with structural
      // field
      maps_systemmatrix[block_positions_structure.at(0)] = block_map_structure()->full_map();

      for (int imap = 0; imap < static_cast<int>(block_positions_thermo.size()); ++imap)
        maps_systemmatrix[block_positions_thermo.at(imap)] = block_map_thermo()->map(imap);

      // initialize map extractor associated with blocks of global system matrix
      block_map_system_matrix_ = std::make_shared<Core::LinAlg::MultiMapExtractor>(
          *maps_sub_problems()->full_map(), maps_systemmatrix);

      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
      break;
    }
  }
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
SSTI::SSTIMatrices::SSTIMatrices(std::shared_ptr<SSTI::SSTIMapsMono> ssti_maps_mono,
    const Core::LinAlg::MatrixType matrixtype_global,
    const Core::LinAlg::MatrixType matrixtype_scatra, bool interfacemeshtying)
    : matrixtype_scatra_(matrixtype_scatra),
      ssti_maps_mono_(ssti_maps_mono),
      systemmatrix_(nullptr),
      scatrastructuredomain_(nullptr),
      scatrastructureinterface_(nullptr),
      scatrathermodomain_(nullptr),
      scatrathermointerface_(nullptr),
      structurescatradomain_(nullptr),
      structurethermodomain_(nullptr),
      thermoscatradomain_(nullptr),
      thermoscatrainterface_(nullptr),
      thermostructuredomain_(nullptr),
      thermostructureinterface_(nullptr),
      interfacemeshtying_(interfacemeshtying)
{
  // perform initializations associated with global system matrix
  switch (matrixtype_global)
  {
    case Core::LinAlg::MatrixType::block_field:
    {
      systemmatrix_ = setup_block_matrix(
          *ssti_maps_mono->block_map_system_matrix(), *ssti_maps_mono->block_map_system_matrix());
      break;
    }

    case Core::LinAlg::MatrixType::sparse:
    {
      systemmatrix_ = setup_sparse_matrix(*ssti_maps_mono->maps_sub_problems()->full_map());
      break;
    }

    default:
    {
      FOUR_C_THROW("Type of global system matrix for scalar-structure interaction not recognized!");
      break;
    }
  }

  // setup blocks for coupling matrices
  switch (matrixtype_scatra)
  {
    case Core::LinAlg::MatrixType::block_condition:
    {
      scatrastructuredomain_ = setup_block_matrix(
          *ssti_maps_mono->block_map_scatra(), *ssti_maps_mono->block_map_structure());
      structurescatradomain_ = setup_block_matrix(
          *ssti_maps_mono->block_map_structure(), *ssti_maps_mono->block_map_scatra());
      structurethermodomain_ = setup_block_matrix(
          *ssti_maps_mono->block_map_structure(), *ssti_maps_mono->block_map_thermo());
      thermostructuredomain_ = setup_block_matrix(
          *ssti_maps_mono->block_map_thermo(), *ssti_maps_mono->block_map_structure());
      scatrathermodomain_ = setup_block_matrix(
          *ssti_maps_mono->block_map_scatra(), *ssti_maps_mono->block_map_thermo());
      thermoscatradomain_ = setup_block_matrix(
          *ssti_maps_mono->block_map_thermo(), *ssti_maps_mono->block_map_scatra());

      if (interfacemeshtying_)
      {
        scatrastructureinterface_ = setup_block_matrix(
            *ssti_maps_mono->block_map_scatra(), *ssti_maps_mono->block_map_structure());
        thermostructureinterface_ = setup_block_matrix(
            *ssti_maps_mono->block_map_thermo(), *ssti_maps_mono->block_map_structure());
        scatrathermointerface_ = setup_block_matrix(
            *ssti_maps_mono->block_map_scatra(), *ssti_maps_mono->block_map_thermo());
        thermoscatrainterface_ = setup_block_matrix(
            *ssti_maps_mono->block_map_thermo(), *ssti_maps_mono->block_map_scatra());
      }
      break;
    }
    case Core::LinAlg::MatrixType::sparse:
    {
      scatrastructuredomain_ = setup_sparse_matrix(*ssti_maps_mono->block_map_scatra()->full_map());
      structurescatradomain_ =
          setup_sparse_matrix(*ssti_maps_mono->block_map_structure()->full_map());
      structurethermodomain_ =
          setup_sparse_matrix(*ssti_maps_mono->block_map_structure()->full_map());
      thermostructuredomain_ = setup_sparse_matrix(*ssti_maps_mono->block_map_thermo()->full_map());
      scatrathermodomain_ = setup_sparse_matrix(*ssti_maps_mono->block_map_scatra()->full_map());
      thermoscatradomain_ = setup_sparse_matrix(*ssti_maps_mono->block_map_thermo()->full_map());

      if (interfacemeshtying_)
      {
        scatrastructureinterface_ =
            setup_sparse_matrix(*ssti_maps_mono->block_map_scatra()->full_map());
        thermostructureinterface_ =
            setup_sparse_matrix(*ssti_maps_mono->block_map_thermo()->full_map());
        scatrathermointerface_ =
            setup_sparse_matrix(*ssti_maps_mono->block_map_scatra()->full_map());
        thermoscatrainterface_ =
            setup_sparse_matrix(*ssti_maps_mono->block_map_thermo()->full_map());
      }
      break;
    }
    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
      break;
    }
  }
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
void SSTI::SSTIMatrices::clear_matrices()
{
  systemmatrix_->zero();
  scatrastructuredomain_->zero();
  scatrathermodomain_->zero();
  structurescatradomain_->zero();
  structurethermodomain_->zero();
  thermoscatradomain_->zero();
  thermostructuredomain_->zero();

  if (interfacemeshtying_)
  {
    scatrastructureinterface_->zero();
    scatrathermointerface_->zero();
    thermoscatrainterface_->zero();
    thermostructureinterface_->zero();
  }
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
void SSTI::SSTIMatrices::complete_coupling_matrices()
{
  switch (matrixtype_scatra_)
  {
    case Core::LinAlg::MatrixType::block_condition:
    case Core::LinAlg::MatrixType::block_condition_dof:
    {
      scatrastructuredomain_->complete();
      scatrathermodomain_->complete();
      structurescatradomain_->complete();
      structurethermodomain_->complete();
      thermoscatradomain_->complete();
      thermostructuredomain_->complete();

      if (interfacemeshtying_)
      {
        scatrastructureinterface_->complete();
        scatrathermointerface_->complete();
        thermoscatrainterface_->complete();
        thermostructureinterface_->complete();
      }
      break;
    }

    case Core::LinAlg::MatrixType::sparse:
    {
      scatrastructuredomain_->complete(*ssti_maps_mono_->block_map_structure()->full_map(),
          *ssti_maps_mono_->block_map_scatra()->full_map());
      scatrathermodomain_->complete(*ssti_maps_mono_->block_map_thermo()->full_map(),
          *ssti_maps_mono_->block_map_scatra()->full_map());
      structurescatradomain_->complete(*ssti_maps_mono_->block_map_scatra()->full_map(),
          *ssti_maps_mono_->block_map_structure()->full_map());
      structurethermodomain_->complete(*ssti_maps_mono_->block_map_thermo()->full_map(),
          *ssti_maps_mono_->block_map_structure()->full_map());
      thermoscatradomain_->complete(*ssti_maps_mono_->block_map_scatra()->full_map(),
          *ssti_maps_mono_->block_map_thermo()->full_map());
      thermostructuredomain_->complete(*ssti_maps_mono_->block_map_structure()->full_map(),
          *ssti_maps_mono_->block_map_thermo()->full_map());

      if (interfacemeshtying_)
      {
        scatrastructureinterface_->complete(*ssti_maps_mono_->block_map_structure()->full_map(),
            *ssti_maps_mono_->block_map_scatra()->full_map());
        scatrathermointerface_->complete(*ssti_maps_mono_->block_map_thermo()->full_map(),
            *ssti_maps_mono_->block_map_scatra()->full_map());
        thermoscatrainterface_->complete(*ssti_maps_mono_->block_map_scatra()->full_map(),
            *ssti_maps_mono_->block_map_thermo()->full_map());
        thermostructureinterface_->complete(*ssti_maps_mono_->block_map_structure()->full_map(),
            *ssti_maps_mono_->block_map_thermo()->full_map());
      }
      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
      break;
    }
  }
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
void SSTI::SSTIMatrices::un_complete_coupling_matrices()
{
  scatrastructuredomain_->un_complete();
  scatrathermodomain_->un_complete();
  structurescatradomain_->un_complete();
  structurethermodomain_->un_complete();
  thermoscatradomain_->un_complete();
  thermostructuredomain_->un_complete();

  if (interfacemeshtying_)
  {
    scatrastructureinterface_->un_complete();
    scatrathermointerface_->un_complete();
    thermoscatrainterface_->un_complete();
    thermostructureinterface_->un_complete();
  }
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> SSTI::SSTIMatrices::setup_block_matrix(
    const Core::LinAlg::MultiMapExtractor& row_map, const Core::LinAlg::MultiMapExtractor& col_map)
{
  const int expected_entries_per_row = 81;
  const bool explicitdirichlet = false;
  const bool savegraph = true;

  return std::make_shared<
      Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
      col_map, row_map, expected_entries_per_row, explicitdirichlet, savegraph);
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> SSTI::SSTIMatrices::setup_sparse_matrix(
    const Core::LinAlg::Map& row_map)
{
  const int expected_entries_per_row = 27;
  const bool explicitdirichlet = false;
  const bool savegraph = true;

  return std::make_shared<Core::LinAlg::SparseMatrix>(
      row_map, expected_entries_per_row, explicitdirichlet, savegraph);
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
SSTI::ConvCheckMono::ConvCheckMono(Teuchos::ParameterList params)
    : itermax_(params.get<int>("ITEMAX")),
      itertol_(params.sublist("MONOLITHIC").get<double>("CONVTOL")),
      restol_(params.sublist("MONOLITHIC").get<double>("ABSTOLRES"))
{
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
bool SSTI::ConvCheckMono::converged(const SSTI::SSTIMono& ssti_mono)
{
  bool is_converged(false);

  // compute L2 norm of concentration state vector
  double concdofnorm(0.0);
  ssti_mono.scatra_field()
      ->splitter()
      ->extract_other_vector(*ssti_mono.scatra_field()->phinp())
      ->norm_2(&concdofnorm);

  // compute L2 norm of concentration increment vector
  double concincnorm(0.0);
  ssti_mono.scatra_field()
      ->splitter()
      ->extract_other_vector(*ssti_mono.all_maps()->maps_sub_problems()->extract_vector(
          *ssti_mono.increment(), ssti_mono.get_problem_position(Subproblem::scalar_transport)))
      ->norm_2(&concincnorm);

  // compute L2 norm of concentration residual vector
  double concresnorm(0.0);
  ssti_mono.scatra_field()
      ->splitter()
      ->extract_other_vector(*ssti_mono.all_maps()->maps_sub_problems()->extract_vector(
          *ssti_mono.residual(), ssti_mono.get_problem_position(Subproblem::scalar_transport)))
      ->norm_2(&concresnorm);

  // compute L2 norm of potential state vector
  double potdofnorm(0.0);
  ssti_mono.scatra_field()
      ->splitter()
      ->extract_cond_vector(*ssti_mono.scatra_field()->phinp())
      ->norm_2(&potdofnorm);

  // compute L2 norm of potential increment vector
  double potincnorm(0.0);
  ssti_mono.scatra_field()
      ->splitter()
      ->extract_cond_vector(*ssti_mono.all_maps()->maps_sub_problems()->extract_vector(
          *ssti_mono.increment(), ssti_mono.get_problem_position(Subproblem::scalar_transport)))
      ->norm_2(&potincnorm);

  // compute L2 norm of potential residual vector
  double potresnorm(0.0);
  ssti_mono.scatra_field()
      ->splitter()
      ->extract_cond_vector(*ssti_mono.all_maps()->maps_sub_problems()->extract_vector(
          *ssti_mono.residual(), ssti_mono.get_problem_position(Subproblem::scalar_transport)))
      ->norm_2(&potresnorm);

  // compute L2 norm of structural state vector
  double structuredofnorm(0.0);
  ssti_mono.structure_field()->dispnp()->norm_2(&structuredofnorm);

  // compute L2 norm of structural residual vector
  double structureresnorm(0.0);
  ssti_mono.all_maps()
      ->maps_sub_problems()
      ->extract_vector(*ssti_mono.residual(), ssti_mono.get_problem_position(Subproblem::structure))
      ->norm_2(&structureresnorm);

  // compute L2 norm of structural increment vector
  double structureincnorm(0.0);
  ssti_mono.all_maps()
      ->maps_sub_problems()
      ->extract_vector(
          *ssti_mono.increment(), ssti_mono.get_problem_position(Subproblem::structure))
      ->norm_2(&structureincnorm);

  // compute L2 norm of thermo state vector
  double thermodofnorm(0.0);
  ssti_mono.thermo_field()->phinp()->norm_2(&thermodofnorm);

  // compute L2 norm of thermo residual vector
  double thermoresnorm(0.0);
  ssti_mono.all_maps()
      ->maps_sub_problems()
      ->extract_vector(*ssti_mono.residual(), ssti_mono.get_problem_position(Subproblem::thermo))
      ->norm_2(&thermoresnorm);

  // compute L2 norm of thermo increment vector
  double thermoincnorm(0.0);
  ssti_mono.all_maps()
      ->maps_sub_problems()
      ->extract_vector(*ssti_mono.increment(), ssti_mono.get_problem_position(Subproblem::thermo))
      ->norm_2(&thermoincnorm);

  // compute L2 norm of total residual vector
  double totresnorm(0.0);
  ssti_mono.residual()->norm_2(&totresnorm);

  // safety checks
  if (std::isnan(concdofnorm) or std::isnan(concresnorm) or std::isnan(concincnorm) or
      std::isnan(potdofnorm) or std::isnan(potresnorm) or std::isnan(potincnorm) or
      std::isnan(structuredofnorm) or std::isnan(structureresnorm) or
      std::isnan(structureincnorm) or std::isnan(thermodofnorm) or std::isnan(thermoresnorm) or
      std::isnan(thermoincnorm))
    FOUR_C_THROW("Vector norm is not a number!");
  if (std::isinf(concdofnorm) or std::isinf(concresnorm) or std::isinf(concincnorm) or
      std::isinf(potdofnorm) or std::isinf(potresnorm) or std::isinf(potincnorm) or
      std::isinf(structuredofnorm) or std::isinf(structureresnorm) or
      std::isinf(structureincnorm) or std::isnan(thermodofnorm) or std::isnan(thermoresnorm) or
      std::isnan(thermoincnorm))
    FOUR_C_THROW("Vector norm is infinity!");

  // prevent division by zero
  if (concdofnorm < 1.e-10) concdofnorm = 1.e-10;
  if (potdofnorm < 1.e-10) potdofnorm = 1.e-10;
  if (structuredofnorm < 1.e-10) structuredofnorm = 1.e-10;
  if (thermodofnorm < 1.e-10) thermodofnorm = 1.e-10;

  // first Newton-Raphson iteration
  if (ssti_mono.newton_iteration() == 1)
  {
    if (Core::Communication::my_mpi_rank(ssti_mono.get_comm()) == 0)
    {
      // print header of convergence table to screen
      std::cout << "+------------+-------------------+--------------+--------------+--------------+"
                   "--------------+--------------+--------------+--------------+--------------+----"
                   "----------+"
                << std::endl;
      std::cout << "|- step/max -|- tolerance[norm] -|-- conc-res --|-- conc-inc --|-- pot-res "
                   "---|-- pot-inc ---|- struct-res -|- struct-inc -|- thermo-res -|- thermo-inc "
                   "-|-  tot. res  -|"
                << std::endl;

      // print first line of convergence table to screen
      // solution increment not yet available during first Newton-Raphson iteration
      std::cout << "|  " << std::setw(3) << ssti_mono.newton_iteration() << "/" << std::setw(3)
                << itermax_ << "   | " << std::setw(10) << std::setprecision(3) << std::scientific
                << itertol_ << "[L_2 ]  | " << std::setw(10) << std::setprecision(3)
                << std::scientific << concresnorm << "   |      --      | " << std::setw(10)
                << std::setprecision(3) << std::scientific << potresnorm << "   |      --      | "
                << std::setw(10) << std::setprecision(3) << std::scientific << structureresnorm
                << "   |      --      | " << std::setw(10) << std::setprecision(3)
                << std::scientific << thermoresnorm << "   |      --      | " << std::setw(10)
                << std::setprecision(3) << std::scientific << totresnorm << "   |    | "
                << "(       --      , te = " << std::setw(10) << std::setprecision(3)
                << ssti_mono.time_statistics()[0] << ")" << std::endl;
    }
  }

  // subsequent Newton-Raphson iterations
  else
  {
    // print current line of convergence table to screen
    if (Core::Communication::my_mpi_rank(ssti_mono.get_comm()) == 0)
    {
      std::cout << "|  " << std::setw(3) << ssti_mono.newton_iteration() << "/" << std::setw(3)
                << itermax_ << "   | " << std::setw(10) << std::setprecision(3) << std::scientific
                << itertol_ << "[L_2 ]  | " << std::setw(10) << std::setprecision(3)
                << std::scientific << concresnorm << "   | " << std::setw(10)
                << std::setprecision(3) << std::scientific << concincnorm / concdofnorm << "   | "
                << std::setw(10) << std::setprecision(3) << std::scientific << potresnorm << "   | "
                << std::setw(10) << std::setprecision(3) << std::scientific
                << potincnorm / potdofnorm << "   | " << std::setw(10) << std::setprecision(3)
                << std::scientific << structureresnorm << "   | " << std::setw(10)
                << std::setprecision(3) << std::scientific << structureincnorm / structuredofnorm
                << "   | " << std::setw(10) << std::setprecision(3) << std::scientific
                << thermoresnorm << "   | " << std::setw(10) << std::setprecision(3)
                << std::scientific << thermoincnorm / thermodofnorm << "   | " << std::setw(10)
                << std::setprecision(3) << std::scientific << totresnorm << "   | "
                << "   | (ts = " << std::setw(10) << std::setprecision(3)
                << ssti_mono.time_statistics()[1] << ", te = " << std::setw(10)
                << std::setprecision(3) << ssti_mono.time_statistics()[0] << ")" << std::endl;
    }

    // convergence check
    if (concresnorm <= itertol_ and potresnorm <= itertol_ and structureresnorm <= itertol_ and
        thermoresnorm <= itertol_ and concincnorm / concdofnorm <= itertol_ and
        potincnorm / potdofnorm <= itertol_ and structureincnorm / structuredofnorm <= itertol_ and
        thermoincnorm / thermodofnorm <= itertol_)
      // exit Newton-Raphson iteration upon convergence
      is_converged = true;
  }

  // exit Newton-Raphson iteration when residuals are small enough to prevent unnecessary additional
  // solver calls
  if (concresnorm < restol_ and potresnorm < restol_ and structureresnorm < restol_ and
      thermoresnorm < restol_)
    is_converged = true;

  // print warning to screen if maximum number of Newton-Raphson iterations is reached without
  // convergence
  if (ssti_mono.newton_iteration() == itermax_ and !is_converged)
  {
    if (Core::Communication::my_mpi_rank(ssti_mono.get_comm()) == 0)
    {
      std::cout << "+------------+-------------------+--------------+--------------+--------------+"
                   "--------------+--------------+--------------+--------------+--------------+----"
                   "----------+"
                << std::endl;
      std::cout << "|                     Newton-Raphson method has not converged after a maximum "
                   "number of "
                << std::setw(2) << itermax_ << " iterations!                     |" << std::endl;
    }

    // proceed to next time step
    is_converged = true;
  }

  return is_converged;
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
std::map<std::string, std::string> SSTI::SSTIScatraStructureCloneStrategy::conditions_to_copy()
    const
{
  // call base class
  std::map<std::string, std::string> conditions_to_copy =
      SSI::ScatraStructureCloneStrategy::conditions_to_copy();

  conditions_to_copy.insert({"ThermoDirichlet", "ThermoDirichlet"});
  conditions_to_copy.insert({"ThermoPointNeumann", "ThermoPointNeumann"});
  conditions_to_copy.insert({"ThermoLineNeumann", "ThermoLineNeumann"});
  conditions_to_copy.insert({"ThermoSurfaceNeumann", "ThermoSurfaceNeumann"});
  conditions_to_copy.insert({"ThermoVolumeNeumann", "ThermoVolumeNeumann"});
  conditions_to_copy.insert({"ThermoInitfield", "ThermoInitfield"});
  conditions_to_copy.insert({"SSTIMeshtying3DomainIntersection", "Meshtying3DomainIntersection"});
  conditions_to_copy.insert({"SSTIInterfaceMeshtying", "S2IMeshtying"});

  return conditions_to_copy;
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
void SSTI::SSTIScatraStructureCloneStrategy::set_element_data(
    std::shared_ptr<Core::Elements::Element> newele, Core::Elements::Element* oldele,
    const int matid, const bool isnurbsdis)
{
  // We need to set material and possibly other things to complete element setup.
  // This is again really ugly as we have to extract the actual
  // element type in order to access the material property

  // note: set_material() was reimplemented by the transport element!
  auto* trans = dynamic_cast<Discret::Elements::Transport*>(newele.get());
  if (trans != nullptr)
  {
    // set distype as well!
    trans->set_dis_type(oldele->shape());

    // now check whether ImplType is reasonable and if set the ImplType
    Inpar::ScaTra::ImplType impltype = SSI::ScatraStructureCloneStrategy::get_impl_type(oldele);

    if (impltype == Inpar::ScaTra::impltype_undefined)
    {
      FOUR_C_THROW(
          "ScatraStructureCloneStrategy copies scatra discretization from structure "
          "discretization, but the STRUCTURE elements that are defined in the input file are "
          "either "
          "not meant to be copied to scatra elements or the ImplType is set 'Undefined' which is "
          "not meaningful for the created scatra discretization! Use SOLIDSCATRA, WALLSCATRA or "
          "SHELLSCATRA elements with meaningful ImplType instead!");
    }
    else
    {
      // find the appropriate thermo type
      if (impltype == Inpar::ScaTra::impltype_elch_electrode)
        trans->set_impl_type(Inpar::ScaTra::impltype_elch_electrode_thermo);
      else if (impltype == Inpar::ScaTra::impltype_elch_diffcond)
        trans->set_impl_type(Inpar::ScaTra::impltype_elch_diffcond_thermo);
      else
        FOUR_C_THROW("Something went wrong");
    }

    // set material
    trans->set_material(matid, oldele);
  }
  else
  {
    FOUR_C_THROW("unsupported element type '{}'", Core::Utils::get_dynamic_type_name(*newele));
  }
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
std::map<std::string, std::string> SSTI::SSTIScatraThermoCloneStrategy::conditions_to_copy() const
{
  // call base class
  std::map<std::string, std::string> conditions_to_copy =
      STI::ScatraThermoCloneStrategy::conditions_to_copy();

  conditions_to_copy.insert({"Meshtying3DomainIntersection", "Meshtying3DomainIntersection"});
  conditions_to_copy.insert({"SSTIInterfaceMeshtying", "S2IMeshtying"});
  conditions_to_copy.insert({"TotalAndMeanScalar", "TotalAndMeanScalar"});

  return conditions_to_copy;
}
FOUR_C_NAMESPACE_CLOSE
