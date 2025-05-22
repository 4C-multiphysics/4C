// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_elast_artery_coupling.hpp"

#include "4C_adapter_art_net.hpp"
#include "4C_adapter_porofluid_pressure_based_wrapper.hpp"
#include "4C_adapter_str_wrapper.hpp"
#include "4C_fem_condition_locsys.hpp"
#include "4C_fem_general_assemblestrategy.hpp"
#include "4C_fem_general_elements_paramsminimal.hpp"
#include "4C_global_data.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_equilibrate.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_print.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_linear_solver_method_input.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_linear_solver_method_parameters.hpp"
#include "4C_porofluid_pressure_based_elast_utils.hpp"
#include "4C_porofluid_pressure_based_utils.hpp"
#include "4C_utils_enum.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 | constructor                                         kremheller 05/18 |
 *----------------------------------------------------------------------*/
PoroPressureBased::PorofluidElastArteryCouplingAlgorithm::PorofluidElastArteryCouplingAlgorithm(
    MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams)
    : PorofluidElastMonolithicAlgorithm(comm, globaltimeparams)
{
  blockrowdofmap_artporo_ = std::make_shared<Core::LinAlg::MultiMapExtractor>();

  return;
}

/*----------------------------------------------------------------------*
 | setup the map                                       kremheller 04/18 |
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastArteryCouplingAlgorithm::setup_maps()
{
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> vecSpaces;

  vecSpaces.push_back(structure_dof_row_map());

  vecSpaces.push_back(porofluid_dof_row_map());

  vecSpaces.push_back(artery_dof_row_map());

  if (vecSpaces[0]->num_global_elements() == 0) FOUR_C_THROW("No structure equation. Panic.");
  if (vecSpaces[1]->num_global_elements() == 0) FOUR_C_THROW("No fluid equation. Panic.");
  if (vecSpaces[2]->num_global_elements() == 0) FOUR_C_THROW("No fluid equation. Panic.");

  // full Poromultiphase-elasticity-map
  fullmap_ = Core::LinAlg::MultiMapExtractor::merge_maps(vecSpaces);

  // full Poromultiphase-elasticity-blockmap
  blockrowdofmap_->setup(*fullmap_, vecSpaces);

  // full map of artery and poromulti DOFs
  fullmap_artporo_ = Core::LinAlg::MultiMapExtractor::merge_maps({vecSpaces[1], vecSpaces[2]});

  // full artery-poromulti-blockmap
  blockrowdofmap_artporo_->setup(*fullmap_artporo_, {vecSpaces[1], vecSpaces[2]});

  return;
}

/*----------------------------------------------------------------------*
 | extract field vectors for calling evaluate() of the  kremheller 04/18|
 | single fields                                                        |
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastArteryCouplingAlgorithm::build_convergence_norms()
{
  std::shared_ptr<const Core::LinAlg::Vector<double>> arteryrhs =
      extractor()->extract_vector(*rhs_, 2);
  std::shared_ptr<const Core::LinAlg::Vector<double>> arteryinc =
      extractor()->extract_vector(*iterinc_, 2);

  // build also norms for artery
  normrhsart_ = calculate_vector_norm(vectornormfres_, *arteryrhs);
  normincart_ = calculate_vector_norm(vectornorminc_, *arteryinc);
  arterypressnorm_ =
      calculate_vector_norm(vectornorminc_, (*porofluid_algo()->art_net_tim_int()->pressurenp()));

  // call base class
  PorofluidElastMonolithicAlgorithm::build_convergence_norms();

  return;
}
/*----------------------------------------------------------------------*
 | extract field vectors for calling evaluate() of the  kremheller 04/18|
 | single fields                                                        |
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastArteryCouplingAlgorithm::extract_field_vectors(
    std::shared_ptr<const Core::LinAlg::Vector<double>> x,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& sx,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& fx)
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "POROMULTIPHASE::PoroMultiPhaseMonolithicTwoWayArteryCoupling::extract_field_vectors");

  // process structure unknowns of the first field
  sx = extractor()->extract_vector(*x, 0);

  // process artery and porofluid unknowns
  std::shared_ptr<const Core::LinAlg::Vector<double>> porofluid =
      extractor()->extract_vector(*x, 1);
  std::shared_ptr<const Core::LinAlg::Vector<double>> artery = extractor()->extract_vector(*x, 2);

  std::shared_ptr<Core::LinAlg::Vector<double>> dummy =
      std::make_shared<Core::LinAlg::Vector<double>>(*fullmap_artporo_);

  blockrowdofmap_artporo_->insert_vector(*porofluid, 0, *dummy);
  blockrowdofmap_artporo_->insert_vector(*artery, 1, *dummy);

  fx = dummy;

  return;
}

/*----------------------------------------------------------------------*
 | setup system matrix of poromultiphase-elasticity with artery         |
 | coupling                                           kremheller 05/18  |
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastArteryCouplingAlgorithm::setup_system_matrix(
    Core::LinAlg::BlockSparseMatrixBase& mat)
{
  PorofluidElastMonolithicAlgorithm::setup_system_matrix(mat);

  // pure artery part
  mat.assign(2, 2, Core::LinAlg::DataAccess::View, artery_porofluid_sysmat()->matrix(1, 1));
  // artery-porofluid part
  mat.assign(2, 1, Core::LinAlg::DataAccess::View, artery_porofluid_sysmat()->matrix(1, 0));
  // porofluid-artery part
  mat.assign(1, 2, Core::LinAlg::DataAccess::View, artery_porofluid_sysmat()->matrix(0, 1));

  return;
}

/*----------------------------------------------------------------------*
 | setup rhs of poromultiphase-elasticity with artery coupling          |
 |                                                    kremheller 05/18  |
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastArteryCouplingAlgorithm::setup_rhs()
{
  // get structure part
  std::shared_ptr<Core::LinAlg::Vector<double>> str_rhs = setup_structure_partof_rhs();

  // insert and scale
  extractor()->insert_vector(*str_rhs, 0, *rhs_);
  rhs_->scale(-1.0);

  // insert artery part and porofluid part
  extractor()->insert_vector(
      *(blockrowdofmap_artporo_->extract_vector(*porofluid_algo()->artery_porofluid_rhs(), 0)), 1,
      *rhs_);
  extractor()->insert_vector(
      *(blockrowdofmap_artporo_->extract_vector(*porofluid_algo()->artery_porofluid_rhs(), 1)), 2,
      *rhs_);

  return;
}

/*-----------------------------------------------------------------------/
|  build the combined dbcmap                           kremheller 05/18  |
/-----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastArteryCouplingAlgorithm::build_combined_dbc_map()
{
  PorofluidElastMonolithicAlgorithm::build_combined_dbc_map();

  const std::shared_ptr<const Core::LinAlg::Map> artcondmap =
      porofluid_algo()->art_net_tim_int()->get_dbc_map_extractor()->cond_map();

  // merge them
  combinedDBCMap_ = Core::LinAlg::merge_map(combinedDBCMap_, artcondmap, false);

  return;
}
/*----------------------------------------------------------------------------*
 | build null space for artery block of global system matrix kremheller 05/18 |
 *--------------------------------------------------------------------------- */
void PoroPressureBased::PorofluidElastArteryCouplingAlgorithm::build_artery_block_null_space(
    std::shared_ptr<Core::LinAlg::Solver>& solver, const int& arteryblocknum)
{
  Teuchos::ParameterList& block_smoother_params =
      solver->params().sublist("Inverse" + std::to_string(arteryblocknum));

  Core::LinearSolver::Parameters::compute_solver_parameters(
      *porofluid_algo()->art_net_tim_int()->discretization(), block_smoother_params);

  // fix the null space if some DOFs are condensed out
  Core::LinearSolver::Parameters::fix_null_space("Artery",
      *(porofluid_algo()->art_net_tim_int()->discretization()->dof_row_map(0)),
      *(porofluid_algo()->artery_dof_row_map()), block_smoother_params);
}

/*----------------------------------------------------------------------*
 | Create linear (iterative) solver                    kremheller 05/18 |
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastArteryCouplingAlgorithm::create_linear_solver(
    const Teuchos::ParameterList& solverparams, const Core::LinearSolver::SolverType solvertype)
{
  PorofluidElastMonolithicAlgorithm::create_linear_solver(solverparams, solvertype);

  // build also the artery null space
  build_artery_block_null_space(solver_, 3);
}

FOUR_C_NAMESPACE_CLOSE
