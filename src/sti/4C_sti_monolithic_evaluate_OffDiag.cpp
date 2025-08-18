// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_sti_monolithic_evaluate_OffDiag.hpp"

#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_assemblestrategy.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_timint_implicit.hpp"
#include "4C_scatra_timint_meshtying_strategy_s2i.hpp"
#include "4C_utils_parameter_list.hpp"

#include <utility>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
STI::ScatraThermoOffDiagCoupling::ScatraThermoOffDiagCoupling(
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> block_map_thermo,
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> block_map_thermo_interface,
    std::shared_ptr<const Core::LinAlg::Map> full_map_scatra,
    std::shared_ptr<const Core::LinAlg::Map> full_map_thermo,
    std::shared_ptr<const Core::LinAlg::Map> interface_map_scatra,
    std::shared_ptr<const Core::LinAlg::Map> interface_map_thermo, bool isale,
    std::shared_ptr<const ScaTra::MeshtyingStrategyS2I> meshtying_strategy_scatra,
    std::shared_ptr<const ScaTra::MeshtyingStrategyS2I> meshtying_strategy_thermo,
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra,
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> thermo)
    : block_map_thermo_(std::move(block_map_thermo)),
      block_map_thermo_interface_(std::move(block_map_thermo_interface)),
      full_map_scatra_(std::move(full_map_scatra)),
      full_map_thermo_(std::move(full_map_thermo)),
      interface_map_scatra_(std::move(interface_map_scatra)),
      interface_map_thermo_(std::move(interface_map_thermo)),
      isale_(isale),
      meshtying_strategy_scatra_(std::move(meshtying_strategy_scatra)),
      meshtying_strategy_thermo_(std::move(meshtying_strategy_thermo)),
      scatra_(std::move(scatra)),
      thermo_(std::move(thermo))
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void STI::ScatraThermoOffDiagCoupling::evaluate_off_diag_block_scatra_thermo_domain(
    std::shared_ptr<Core::LinAlg::SparseOperator> scatrathermoblock)
{
  // initialize scatra-thermo matrix block
  scatrathermoblock->zero();

  // create parameter list for element evaluation
  Teuchos::ParameterList eleparams;

  // action for elements
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::calc_scatra_mono_odblock_scatrathermo, eleparams);

  // remove state vectors from scatra discretization
  scatra_field()->discretization()->clear_state();

  // add state vectors to scatra discretization
  scatra_field()->add_time_integration_specific_vectors();

  // create strategy for assembly of scatra-thermo matrix block
  Core::FE::AssembleStrategy strategyscatrathermo(
      0,  // row assembly based on number of dofset associated with scatra dofs on scatra
          // discretization
      2,  // column assembly based on number of dofset associated with thermo dofs on scatra
          // discretization
      scatrathermoblock,  // scatra-thermo matrix block
      nullptr,            // no additional matrices or vectors
      nullptr, nullptr, nullptr);

  // assemble scatra-thermo matrix block
  scatra_field()->discretization()->evaluate(eleparams, strategyscatrathermo);

  // remove state vectors from scalar transport discretization
  scatra_field()->discretization()->clear_state();

  // finalize scatra-thermo block
  switch (scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    {
      scatrathermoblock->complete();
      break;
    }

    case Core::LinAlg::MatrixType::sparse:
    {
      scatrathermoblock->complete(*full_map_thermo(), *full_map_scatra());
      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
      break;
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void STI::ScatraThermoOffDiagCoupling::evaluate_off_diag_block_thermo_scatra_domain(
    std::shared_ptr<Core::LinAlg::SparseOperator> thermoscatrablock)
{
  // initialize thermo-scatra matrix block
  thermoscatrablock->zero();

  // create parameter list for element evaluation
  Teuchos::ParameterList eleparams;

  // action for elements
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::calc_scatra_mono_odblock_thermoscatra, eleparams);

  // remove state vectors from thermo discretization
  thermo_field()->discretization()->clear_state();

  // add state vectors to thermo discretization
  thermo_field()->add_time_integration_specific_vectors();

  // create strategy for assembly of thermo-scatra matrix block
  Core::FE::AssembleStrategy strategythermoscatra(
      0,  // row assembly based on number of dofset associated with thermo dofs on thermo
          // discretization
      2,  // column assembly based on number of dofset associated with scatra dofs on thermo
          // discretization
      thermoscatrablock,  // thermo-scatra matrix block
      nullptr,            // no additional matrices or vectors
      nullptr, nullptr, nullptr);

  // assemble thermo-scatra matrix block
  thermo_field()->discretization()->evaluate(eleparams, strategythermoscatra);

  // finalize thermo-scatra matrix block
  switch (scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    {
      thermoscatrablock->complete();
      break;
    }

    case Core::LinAlg::MatrixType::sparse:
    {
      thermoscatrablock->complete(*full_map_scatra(), *full_map_thermo());
      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
      break;
    }
  }

  // remove state vectors from thermo discretization
  thermo_field()->discretization()->clear_state();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
STI::ScatraThermoOffDiagCouplingMatchingNodes::ScatraThermoOffDiagCouplingMatchingNodes(
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> block_map_thermo,
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> block_map_thermo_interface,
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> block_map_thermo_interface_slave,
    std::shared_ptr<const Core::LinAlg::Map> full_map_scatra,
    std::shared_ptr<const Core::LinAlg::Map> full_map_thermo,
    std::shared_ptr<const Core::LinAlg::Map> interface_map_scatra,
    std::shared_ptr<const Core::LinAlg::Map> interface_map_thermo, bool isale,
    std::shared_ptr<const ScaTra::MeshtyingStrategyS2I> meshtying_strategy_scatra,
    std::shared_ptr<const ScaTra::MeshtyingStrategyS2I> meshtying_strategy_thermo,
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra,
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> thermo)
    : ScatraThermoOffDiagCoupling(std::move(block_map_thermo),
          std::move(block_map_thermo_interface), std::move(full_map_scatra),
          std::move(full_map_thermo), std::move(interface_map_scatra),
          std::move(interface_map_thermo), isale, std::move(meshtying_strategy_scatra),
          std::move(meshtying_strategy_thermo), std::move(scatra), std::move(thermo)),
      block_map_thermo_interface_slave_(std::move(block_map_thermo_interface_slave))
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void STI::ScatraThermoOffDiagCouplingMatchingNodes::evaluate_off_diag_block_scatra_thermo_interface(
    std::shared_ptr<Core::LinAlg::SparseOperator> scatrathermoblockinterface)
{
  // zero out matrix
  scatrathermoblockinterface->zero();

  // slave and master matrix for evaluation of conditions
  std::shared_ptr<Core::LinAlg::SparseOperator> slavematrix(nullptr);
  std::shared_ptr<Core::LinAlg::SparseOperator> mastermatrix(nullptr);
  switch (scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    {
      slavematrix = std::make_shared<
          Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(

          *block_map_thermo_interface(), meshtying_strategy_scatra()->block_maps_slave(), 81, false,
          true);
      mastermatrix = std::make_shared<
          Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(

          *block_map_thermo_interface(), meshtying_strategy_scatra()->block_maps_master(), 81,
          false, true);
      break;
    }
    case Core::LinAlg::MatrixType::sparse:
    {
      slavematrix = std::make_shared<Core::LinAlg::SparseMatrix>(
          *meshtying_strategy_scatra()->coupling_adapter()->slave_dof_map(), 27, false, true);
      mastermatrix = std::make_shared<Core::LinAlg::SparseMatrix>(
          *meshtying_strategy_scatra()->coupling_adapter()->master_dof_map(), 27, false, true);
      break;
    }
    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
      break;
    }
  }

  // evaluate interface contibutions on slave side
  evaluate_scatra_thermo_interface_slave_side(slavematrix);

  // copy interface contributions from slave side to master side
  copy_slave_to_master_scatra_thermo_interface(slavematrix, mastermatrix);

  // add contributions from slave side and master side
  scatrathermoblockinterface->add(*slavematrix, false, 1.0, 1.0);
  scatrathermoblockinterface->add(*mastermatrix, false, 1.0, 1.0);

  // finalize scatra-thermo matrix block
  switch (scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    {
      scatrathermoblockinterface->complete();
      break;
    }
    case Core::LinAlg::MatrixType::sparse:
    {
      scatrathermoblockinterface->complete(*interface_map_thermo(), *interface_map_scatra());
      break;
    }
    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
      break;
    }
  }

  // remove state vectors from scalar transport discretization
  scatra_field()->discretization()->clear_state();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void STI::ScatraThermoOffDiagCouplingMatchingNodes::evaluate_scatra_thermo_interface_slave_side(
    std::shared_ptr<Core::LinAlg::SparseOperator> slavematrix)
{
  // zero out slavematrtix
  slavematrix->zero();

  // create parameter list for element evaluation
  Teuchos::ParameterList condparams;

  // action for elements
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
      "action", ScaTra::BoundaryAction::calc_s2icoupling_od, condparams);

  // set type of differentiation to temperature
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::DifferentiationType>(
      "differentiationtype", ScaTra::DifferentiationType::temp, condparams);

  // remove state vectors from scalar transport discretization
  scatra_field()->discretization()->clear_state();

  // add state vectors to scalar transport discretization
  scatra_field()->add_time_integration_specific_vectors();

  // create strategy for assembly of auxiliary system matrix
  Core::FE::AssembleStrategy strategyscatrathermos2i(
      0,            // row assembly based on number of dofset associated with scatra dofs on scatra
                    // discretization
      2,            // column assembly based on number of dofset associated with thermo dofs on
                    // scatra discretization
      slavematrix,  // auxiliary system matrix
      nullptr,      // no additional matrices of vectors
      nullptr, nullptr, nullptr);

  // evaluate scatra-scatra interface kinetics
  std::vector<Core::Conditions::Condition*> conditions;
  for (const auto& kinetics_slave_cond :
      meshtying_strategy_scatra()->kinetics_conditions_meshtying_slave_side())
  {
    if (kinetics_slave_cond.second->parameters().get<Inpar::S2I::KineticModels>("KINETIC_MODEL") !=
        static_cast<int>(Inpar::S2I::kinetics_nointerfaceflux))
    {
      // collect condition specific data and store to scatra boundary parameter class
      meshtying_strategy_scatra()->set_condition_specific_scatra_parameters(
          *kinetics_slave_cond.second);
      // evaluate the condition
      scatra_field()->discretization()->evaluate_condition(condparams, strategyscatrathermos2i,
          "S2IKinetics", kinetics_slave_cond.second->parameters().get<int>("ConditionID"));
    }
  }

  // finalize slave matrix
  switch (scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    {
      slavematrix->complete();
      break;
    }
    case Core::LinAlg::MatrixType::sparse:
    {
      slavematrix->complete(*interface_map_thermo(),
          *meshtying_strategy_scatra()->coupling_adapter()->slave_dof_map());
      break;
    }
    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
      break;
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void STI::ScatraThermoOffDiagCouplingMatchingNodes::copy_slave_to_master_scatra_thermo_interface(
    std::shared_ptr<const Core::LinAlg::SparseOperator> slavematrix,
    std::shared_ptr<Core::LinAlg::SparseOperator>& mastermatrix)
{
  // zero out master matrix
  mastermatrix->zero();

  switch (scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    {
      // cast master and slave matrix
      const auto blockslavematrix =
          std::dynamic_pointer_cast<const Core::LinAlg::BlockSparseMatrixBase>(slavematrix);
      auto blockmastermatrix =
          std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(mastermatrix);

      // initialize auxiliary system matrix for linearizations of master-side scatra
      // fluxes w.r.t. slave-side thermo dofs
      Core::LinAlg::SparseMatrix mastermatrixsparse(
          *meshtying_strategy_scatra()->coupling_adapter()->master_dof_map(), 27, false, true);

      // derive linearizations of master-side scatra fluxes w.r.t. slave-side thermo dofs
      // and assemble into auxiliary system matrix
      for (int iblock = 0; iblock < meshtying_strategy_scatra()->block_maps_slave().num_maps();
          ++iblock)
      {
        Coupling::Adapter::MatrixRowTransform()(blockslavematrix->matrix(iblock, 0), -1.0,
            Coupling::Adapter::CouplingSlaveConverter(
                *meshtying_strategy_scatra()->coupling_adapter()),
            mastermatrixsparse, true);
      }

      // finalize auxiliary system matrix
      mastermatrixsparse.complete(*meshtying_strategy_thermo()->coupling_adapter()->slave_dof_map(),
          *meshtying_strategy_scatra()->coupling_adapter()->master_dof_map());

      // split auxiliary system matrix and assemble into scatra-thermo matrix block
      blockmastermatrix = Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
          mastermatrixsparse, *block_map_thermo(), *scatra_field()->dof_block_maps());

      // finalize master matrix
      mastermatrix->complete();

      break;
    }
    case Core::LinAlg::MatrixType::sparse:
    {
      // cast master and slave matrix
      const auto sparseslavematrix =
          std::dynamic_pointer_cast<const Core::LinAlg::SparseMatrix>(slavematrix);
      auto sparsemastermatrix = std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(mastermatrix);

      // derive linearizations of master-side scatra fluxes w.r.t. slave-side thermo dofs
      // and assemble into scatra-thermo matrix block
      Coupling::Adapter::MatrixRowTransform()(*sparseslavematrix, -1.0,
          Coupling::Adapter::CouplingSlaveConverter(
              *meshtying_strategy_scatra()->coupling_adapter()),
          *sparsemastermatrix, false);

      // finalize master matrix
      mastermatrix->complete(
          *interface_map_thermo(), *meshtying_strategy_scatra()->interface_maps()->map(2));

      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");

      break;
    }
  }
  // linearizations of scatra fluxes w.r.t. master-side thermo dofs are not needed,
  // since these dofs will be condensed out later
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void STI::ScatraThermoOffDiagCouplingMatchingNodes::evaluate_off_diag_block_thermo_scatra_interface(
    std::shared_ptr<Core::LinAlg::SparseOperator> thermoscatrablockinterface)
{
  // zero out matrix
  thermoscatrablockinterface->zero();

  // initialize slave and master matrix
  std::shared_ptr<Core::LinAlg::SparseOperator> slavematrix(nullptr);
  meshtying_strategy_thermo()->master_matrix()->zero();
  auto mastermatrix = meshtying_strategy_thermo()->master_matrix();
  switch (scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    {
      slavematrix = std::make_shared<
          Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(

          meshtying_strategy_scatra()->block_maps_slave(), *block_map_thermo_interface_slave(), 81,
          false, true);
      break;
    }
    case Core::LinAlg::MatrixType::sparse:
    {
      meshtying_strategy_thermo()->slave_matrix()->zero();
      slavematrix = meshtying_strategy_thermo()->slave_matrix();
      break;
    }
    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
      break;
    }
  }

  // remove state vectors from thermo discretization
  thermo_field()->discretization()->clear_state();

  // add state vectors to thermo discretization
  thermo_field()->add_time_integration_specific_vectors();

  // create parameter list for element evaluation
  Teuchos::ParameterList condparams;

  // action for elements
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
      "action", ScaTra::BoundaryAction::calc_s2icoupling_od, condparams);

  // set differentiation type to elch
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::DifferentiationType>(
      "differentiationtype", ScaTra::DifferentiationType::elch, condparams);

  // create strategy for assembly of auxiliary system matrices
  Core::FE::AssembleStrategy strategythermoscatras2i(
      0,             // row assembly based on number of dofset associated with thermo dofs on
                     // thermo discretization
      2,             // column assembly based on number of dofset associated with scatra dofs on
                     // thermo discretization
      slavematrix,   // auxiliary system matrix for slave side
      mastermatrix,  // auxiliary system matrix for master side
      nullptr,       // no additional matrices of vectors
      nullptr, nullptr);

  // evaluate scatra-scatra interface kinetics
  for (const auto& kinetics_slave_cond :
      meshtying_strategy_thermo()->kinetics_conditions_meshtying_slave_side())
  {
    if (kinetics_slave_cond.second->parameters().get<Inpar::S2I::KineticModels>("KINETIC_MODEL") !=
        static_cast<int>(Inpar::S2I::kinetics_nointerfaceflux))
    {
      // collect condition specific data and store to scatra boundary parameter class
      meshtying_strategy_thermo()->set_condition_specific_scatra_parameters(
          *kinetics_slave_cond.second);
      // evaluate the condition
      thermo_field()->discretization()->evaluate_condition(condparams, strategythermoscatras2i,
          "S2IKinetics", kinetics_slave_cond.second->parameters().get<int>("ConditionID"));
    }
  }

  switch (scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    {
      // finalize auxiliary system matrices
      slavematrix->complete();
      mastermatrix->complete(*meshtying_strategy_scatra()->coupling_adapter()->slave_dof_map(),
          *meshtying_strategy_thermo()->coupling_adapter()->slave_dof_map());

      // assemble linearizations of slave-side thermo fluxes w.r.t. slave-side scatra dofs
      // into thermo-scatra matrix block
      thermoscatrablockinterface->add(*slavematrix, false, 1.0, 1.0);

      // initialize temporary matrix
      Core::LinAlg::SparseMatrix ksm(
          *meshtying_strategy_thermo()->coupling_adapter()->slave_dof_map(), 27, false, true);

      // transform linearizations of slave-side thermo fluxes w.r.t. master-side scatra dofs
      Coupling::Adapter::MatrixColTransform()(mastermatrix->row_map(), mastermatrix->col_map(),
          *mastermatrix, 1.0,
          Coupling::Adapter::CouplingSlaveConverter(
              *meshtying_strategy_scatra()->coupling_adapter()),
          ksm, true, false);

      // finalize temporary matrix
      ksm.complete(*meshtying_strategy_scatra()->coupling_adapter()->master_dof_map(),
          *meshtying_strategy_thermo()->coupling_adapter()->slave_dof_map());

      // split temporary matrix and assemble into thermo-scatra matrix block
      const auto blockksm = Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
          ksm, meshtying_strategy_scatra()->block_maps_master(),
          *block_map_thermo_interface_slave());
      blockksm->complete();
      thermoscatrablockinterface->add(*blockksm, false, 1.0, 1.0);

      // finalize matrix
      thermoscatrablockinterface->complete();

      break;
    }
    case Core::LinAlg::MatrixType::sparse:
    {
      slavematrix->complete(*meshtying_strategy_scatra()->coupling_adapter()->slave_dof_map(),
          *meshtying_strategy_thermo()->coupling_adapter()->slave_dof_map());
      mastermatrix->complete(*meshtying_strategy_scatra()->coupling_adapter()->slave_dof_map(),
          *meshtying_strategy_thermo()->coupling_adapter()->slave_dof_map());

      // assemble linearizations of slave-side thermo fluxes w.r.t. slave-side scatra dofs
      // into thermo-scatra matrix block
      thermoscatrablockinterface->add(*slavematrix, false, 1.0, 1.0);

      // derive linearizations of slave-side thermo fluxes w.r.t. master-side scatra dofs
      // and assemble into thermo-scatra matrix block
      Coupling::Adapter::MatrixColTransform()(mastermatrix->row_map(), mastermatrix->col_map(),
          *mastermatrix, 1.0,
          Coupling::Adapter::CouplingSlaveConverter(
              *meshtying_strategy_scatra()->coupling_adapter()),
          *std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(thermoscatrablockinterface), true,
          true);

      // finalize matrix
      thermoscatrablockinterface->complete(*interface_map_scatra(),
          *meshtying_strategy_thermo()->coupling_adapter()->slave_dof_map());

      break;
    }
    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
      break;
    }
  }

  // remove state vectors from thermo discretization
  thermo_field()->discretization()->clear_state();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
STI::ScatraThermoOffDiagCouplingMortarStandard::ScatraThermoOffDiagCouplingMortarStandard(
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> block_map_thermo,
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> block_map_thermo_interface,
    std::shared_ptr<const Core::LinAlg::Map> full_map_scatra,
    std::shared_ptr<const Core::LinAlg::Map> full_map_thermo,
    std::shared_ptr<const Core::LinAlg::Map> interface_map_scatra,
    std::shared_ptr<const Core::LinAlg::Map> interface_map_thermo, bool isale,
    std::shared_ptr<const ScaTra::MeshtyingStrategyS2I> meshtying_strategy_scatra,
    std::shared_ptr<const ScaTra::MeshtyingStrategyS2I> meshtying_strategy_thermo,
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra,
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> thermo)
    : ScatraThermoOffDiagCoupling(std::move(block_map_thermo),
          std::move(block_map_thermo_interface), std::move(full_map_scatra),
          std::move(full_map_thermo), std::move(interface_map_scatra),
          std::move(interface_map_thermo), isale, std::move(meshtying_strategy_scatra),
          std::move(meshtying_strategy_thermo), std::move(scatra), std::move(thermo))
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void STI::ScatraThermoOffDiagCouplingMortarStandard::
    evaluate_off_diag_block_scatra_thermo_interface(
        std::shared_ptr<Core::LinAlg::SparseOperator> scatrathermoblockinterface)
{
  // zero out matrix
  scatrathermoblockinterface->zero();

  // initialize auxiliary system matrices for linearizations of slave-side and master-side
  // scatra fluxes w.r.t. slave-side thermo dofs
  std::shared_ptr<Core::LinAlg::SparseOperator> slavematrix(nullptr);
  meshtying_strategy_scatra()->master_matrix()->zero();
  std::shared_ptr<Core::LinAlg::SparseMatrix> mastermatrix_sparse =
      meshtying_strategy_scatra()->master_matrix();
  switch (scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    {
      slavematrix = std::make_shared<
          Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(

          *block_map_thermo_interface(), meshtying_strategy_scatra()->block_maps_slave(), 81, false,
          true);
      break;
    }

    case Core::LinAlg::MatrixType::sparse:
    {
      slavematrix = meshtying_strategy_scatra()->slave_matrix();
      slavematrix->zero();
      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
      break;
    }
  }

  // create parameter list for element evaluation
  Teuchos::ParameterList condparams;

  // action for elements
  condparams.set<Inpar::S2I::EvaluationActions>("action", Inpar::S2I::evaluate_condition_od);

  // create strategy for assembly of auxiliary system matrices
  ScaTra::MortarCellAssemblyStrategy strategyscatrathermos2i(slavematrix, Inpar::S2I::side_slave,
      Inpar::S2I::side_slave, nullptr, Inpar::S2I::side_undefined, Inpar::S2I::side_undefined,
      mastermatrix_sparse, Inpar::S2I::side_master, Inpar::S2I::side_slave, nullptr,
      Inpar::S2I::side_undefined, Inpar::S2I::side_undefined, nullptr, Inpar::S2I::side_undefined,
      nullptr, Inpar::S2I::side_undefined, 0, 1);

  // extract scatra-scatra interface kinetics conditions
  std::vector<const Core::Conditions::Condition*> conditions;
  scatra_field()->discretization()->get_condition("S2IKinetics", conditions);

  // loop over all conditions
  for (const auto& condition : conditions)
  {
    // consider conditions for slave side only
    if (condition->parameters().get<Inpar::S2I::InterfaceSides>("INTERFACE_SIDE") ==
        Inpar::S2I::side_slave)
    {
      // add condition to parameter list
      condparams.set<const Core::Conditions::Condition*>("condition", condition);

      // collect condition specific data and store to scatra boundary parameter class
      meshtying_strategy_scatra()->set_condition_specific_scatra_parameters(*condition);
      // evaluate mortar integration cells
      meshtying_strategy_scatra()->evaluate_mortar_cells(
          meshtying_strategy_scatra()->mortar_discretization(
              condition->parameters().get<int>("ConditionID")),
          condparams, strategyscatrathermos2i);
    }
  }

  // finalize auxiliary system matrices
  mastermatrix_sparse->complete(
      *interface_map_thermo(), *meshtying_strategy_scatra()->interface_maps()->map(2));

  std::shared_ptr<Core::LinAlg::SparseOperator> mastermatrix(nullptr);
  switch (scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    {
      slavematrix->complete();
      mastermatrix =

          Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
              *meshtying_strategy_scatra()->master_matrix(), *block_map_thermo_interface(),
              meshtying_strategy_scatra()->block_maps_master());
      mastermatrix->complete();

      break;
    }
    case Core::LinAlg::MatrixType::sparse:
    {
      slavematrix->complete(
          *interface_map_thermo(), *meshtying_strategy_scatra()->interface_maps()->map(1));
      mastermatrix = mastermatrix_sparse;

      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
      break;
    }
  }

  // assemble linearizations of slave-side and master-side scatra fluxes w.r.t. slave-side
  // thermo dofs into scatra-thermo matrix block
  scatrathermoblockinterface->add(*slavematrix, false, 1.0, 1.0);
  scatrathermoblockinterface->add(*mastermatrix, false, 1.0, 1.0);

  // linearizations of scatra fluxes w.r.t. master-side thermo dofs are not needed, since
  // these dofs will be condensed out later
  // finalize scatra-thermo matrix block
  switch (scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    {
      scatrathermoblockinterface->complete();
      break;
    }

    case Core::LinAlg::MatrixType::sparse:
    {
      scatrathermoblockinterface->complete(*interface_map_thermo(), *interface_map_scatra());
      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
      break;
    }
  }

  // remove state vectors from scatra discretization
  scatra_field()->discretization()->clear_state();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void STI::ScatraThermoOffDiagCouplingMortarStandard::
    evaluate_off_diag_block_thermo_scatra_interface(
        std::shared_ptr<Core::LinAlg::SparseOperator> thermoscatrablockinterface)
{
  // zero out matrix
  thermoscatrablockinterface->zero();

  // remove state vectors from thermo discretization
  thermo_field()->discretization()->clear_state();

  // add state vectors to thermo discretization
  thermo_field()->add_time_integration_specific_vectors();

  // initialize auxiliary system matrix for linearizations of slave-side thermo fluxes
  // w.r.t. slave-side and master-side scatra dofs
  std::shared_ptr<Core::LinAlg::SparseOperator> slavematrix(nullptr);
  switch (scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    {
      slavematrix = std::make_shared<
          Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          *scatra_field()->dof_block_maps(), *block_map_thermo_interface(), 81, false, true);
      break;
    }

    case Core::LinAlg::MatrixType::sparse:
    {
      slavematrix = meshtying_strategy_thermo()->slave_matrix();
      slavematrix->zero();
      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
    }
  }

  // create parameter list for element evaluation
  Teuchos::ParameterList condparams;

  // action for elements
  condparams.set<Inpar::S2I::EvaluationActions>("action", Inpar::S2I::evaluate_condition_od);

  // create strategy for assembly of auxiliary system matrix
  ScaTra::MortarCellAssemblyStrategy strategythermoscatras2i(slavematrix, Inpar::S2I::side_slave,
      Inpar::S2I::side_slave, slavematrix, Inpar::S2I::side_slave, Inpar::S2I::side_master, nullptr,
      Inpar::S2I::side_undefined, Inpar::S2I::side_undefined, nullptr, Inpar::S2I::side_undefined,
      Inpar::S2I::side_undefined, nullptr, Inpar::S2I::side_undefined, nullptr,
      Inpar::S2I::side_undefined, 0, 1);

  // extract scatra-scatra interface kinetics conditions
  std::vector<const Core::Conditions::Condition*> conditions;
  thermo_field()->discretization()->get_condition("S2IKinetics", conditions);

  // loop over all conditions
  for (const auto& condition : conditions)
  {
    // consider conditions for slave side only
    if (condition->parameters().get<Inpar::S2I::InterfaceSides>("INTERFACE_SIDE") ==
        Inpar::S2I::side_slave)
    {
      // add condition to parameter list
      condparams.set<const Core::Conditions::Condition*>("condition", condition);

      // collect condition specific data and store to scatra boundary parameter class
      meshtying_strategy_thermo()->set_condition_specific_scatra_parameters(*condition);
      // evaluate mortar integration cells
      meshtying_strategy_thermo()->evaluate_mortar_cells(
          meshtying_strategy_thermo()->mortar_discretization(
              condition->parameters().get<int>("ConditionID")),
          condparams, strategythermoscatras2i);
    }
  }

  // finalize auxiliary system matrix
  switch (scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    {
      slavematrix->complete();
      break;
    }

    case Core::LinAlg::MatrixType::sparse:
    {
      slavematrix->complete(
          *interface_map_scatra(), *meshtying_strategy_thermo()->interface_maps()->map(1));
      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
      break;
    }
  }

  // assemble linearizations of slave-side thermo fluxes w.r.t. slave-side and master-side
  // scatra dofs into thermo-scatra matrix block
  thermoscatrablockinterface->add(*slavematrix, false, 1.0, 1.0);

  // linearizations of master-side thermo fluxes w.r.t. scatra dofs are not needed, since
  // thermo fluxes are source terms and thus only evaluated once on slave side

  // finalize thermo-scatra matrix block
  switch (scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    {
      thermoscatrablockinterface->complete();
      break;
    }

    case Core::LinAlg::MatrixType::sparse:
    {
      thermoscatrablockinterface->complete(*interface_map_scatra(), *interface_map_thermo());
      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
      break;
    }
  }

  // remove state vectors from thermo discretization
  thermo_field()->discretization()->clear_state();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<STI::ScatraThermoOffDiagCoupling> STI::build_scatra_thermo_off_diag_coupling(
    const Inpar::S2I::CouplingType& couplingtype,
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> block_map_thermo,
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> block_map_thermo_interface,
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> block_map_thermo_interface_slave,
    std::shared_ptr<const Core::LinAlg::Map> full_map_scatra,
    std::shared_ptr<const Core::LinAlg::Map> full_map_thermo,
    std::shared_ptr<const Core::LinAlg::Map> interface_map_scatra,
    std::shared_ptr<const Core::LinAlg::Map> interface_map_thermo, bool isale,
    std::shared_ptr<const ScaTra::MeshtyingStrategyS2I> meshtying_strategy_scatra,
    std::shared_ptr<const ScaTra::MeshtyingStrategyS2I> meshtying_strategy_thermo,
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra,
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> thermo)
{
  std::shared_ptr<STI::ScatraThermoOffDiagCoupling> scatrathermooffdiagcoupling = nullptr;

  switch (couplingtype)
  {
    case Inpar::S2I::coupling_matching_nodes:
    {
      scatrathermooffdiagcoupling = std::make_shared<STI::ScatraThermoOffDiagCouplingMatchingNodes>(
          block_map_thermo, block_map_thermo_interface, block_map_thermo_interface_slave,
          full_map_scatra, full_map_thermo, interface_map_scatra, interface_map_thermo, isale,
          meshtying_strategy_scatra, meshtying_strategy_thermo, scatra, thermo);
      break;
    }
    case Inpar::S2I::coupling_mortar_standard:
    {
      scatrathermooffdiagcoupling =
          std::make_shared<STI::ScatraThermoOffDiagCouplingMortarStandard>(block_map_thermo,
              block_map_thermo_interface, full_map_scatra, full_map_thermo, interface_map_scatra,
              interface_map_thermo, isale, meshtying_strategy_scatra, meshtying_strategy_thermo,
              scatra, thermo);
      break;
    }
    default:
    {
      FOUR_C_THROW("Not supported coupling type");
      break;
    }
  }

  return scatrathermooffdiagcoupling;
}

FOUR_C_NAMESPACE_CLOSE
