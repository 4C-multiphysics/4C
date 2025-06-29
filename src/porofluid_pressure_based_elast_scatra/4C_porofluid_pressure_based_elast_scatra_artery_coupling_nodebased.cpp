// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_elast_scatra_artery_coupling_nodebased.hpp"

#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_fem_condition_selector.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
PoroPressureBased::PorofluidElastScatraArteryCouplingNodeBasedAlgorithm::
    PorofluidElastScatraArteryCouplingNodeBasedAlgorithm(
        std::shared_ptr<Core::FE::Discretization> artery_dis,
        std::shared_ptr<Core::FE::Discretization> homogenized_dis,
        const Teuchos::ParameterList& meshtying_params, const std::string& condition_name)
    : PorofluidElastScatraArteryCouplingBaseAlgorithm(
          artery_dis, homogenized_dis, meshtying_params),
      condition_name_(condition_name)
{
  // user info
  if (my_mpi_rank_ == 0)
  {
    std::cout << "<                                                  >" << '\n';
    print_coupling_method();
    std::cout << "<                                                  >" << '\n';
    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>" << '\n';
    std::cout << "\n";
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNodeBasedAlgorithm::init()
{
  // ARTERY COUPLING CONDITIONS
  std::vector condition_ids = {std::vector<int>(), std::vector<int>()};

  // check if conditions are defined on both discretizations --------------------------
  // 1) 1D artery discretization
  std::vector<const Core::Conditions::Condition*> artery_coupling_condition;
  artery_dis_->get_condition(condition_name_, artery_coupling_condition);

  for (const auto& iter : artery_coupling_condition)
  {
    int myID = iter->parameters().get<int>("COUPID");
    condition_ids[0].push_back(myID);
  }

  // 2) 2D, 3D homogenized field discretization
  std::vector<const Core::Conditions::Condition*> homogenized_coupling_condition;
  homogenized_dis_->get_condition(condition_name_, homogenized_coupling_condition);

  for (const auto& iter : homogenized_coupling_condition)
  {
    int myID = iter->parameters().get<int>("COUPID");
    condition_ids[1].push_back(myID);
  }

  if (condition_ids[0].size() != condition_ids[1].size())
    FOUR_C_THROW("Artery coupling conditions need to be defined on both discretizations");

  // -----------------------------------------------------------------------------------------------------------------
  // create map extractors needed for artery condition coupling --> homogenized field part
  homogenized_field_extractor_ = std::make_shared<Core::LinAlg::MultiMapExtractor>();
  setup_map_extractor(*homogenized_field_extractor_, *homogenized_dis_, coupled_dofs_homogenized_);
  check_dbc_on_coupled_dofs(*homogenized_dis_, homogenized_field_extractor_->map(1));

  // -----------------------------------------------------------------------------------------------------------------
  // create map extractors needed for artery condition coupling --> artery part
  artery_field_extractor_ = std::make_shared<Core::LinAlg::MultiMapExtractor>();
  setup_map_extractor(*artery_field_extractor_, *artery_dis_, coupled_dofs_artery_);
  check_dbc_on_coupled_dofs(*artery_dis_, artery_field_extractor_->map(1));

  // setup coupling adapter
  coupling_artery_homogenized_ = std::make_shared<Coupling::Adapter::Coupling>();
  coupling_artery_homogenized_->setup_condition_coupling(*homogenized_dis_,
      homogenized_field_extractor_->map(1), *artery_dis_, artery_field_extractor_->map(1),
      condition_name_, coupled_dofs_homogenized_, coupled_dofs_artery_);

  // full map of the homogenized field and uncoupled dofs of artery
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> maps;
  maps.push_back(homogenized_field_extractor_->full_map());
  maps.push_back(artery_field_extractor_->map(0));

  fullmap_ = Core::LinAlg::MultiMapExtractor::merge_maps(maps);
  /// dof row map of coupled problem split in (field) blocks
  global_extractor_ = std::make_shared<Core::LinAlg::MultiMapExtractor>();
  global_extractor_->setup(*fullmap_, maps);

  // check global map extractor
  global_extractor_->check_for_valid_map_extractor();

  // needed for matrix transformations
  row_col_transform_ = std::make_shared<Coupling::Adapter::MatrixRowColTransform>();
  row_transform_ = std::make_shared<Coupling::Adapter::MatrixRowTransform>();
  col_transform_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNodeBasedAlgorithm::setup_map_extractor(
    Core::LinAlg::MultiMapExtractor& map_extractor, Core::FE::Discretization& dis,
    const std::vector<int>& coupled_dofs) const
{
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> coupled_maps;

  // build coupled maps for all coupled dofs
  for (int idof = 0; idof < num_coupled_dofs_; idof++)
  {
    Core::LinAlg::MultiMapExtractor current_map_extractor;
    Core::Conditions::setup_extractor(dis, current_map_extractor,
        {Core::Conditions::Selector(condition_name_, coupled_dofs[idof], coupled_dofs[idof] + 1)});

    coupled_maps.push_back(current_map_extractor.map(1));
  }
  // full map coupled -> all coupled dofs
  const std::shared_ptr<Core::LinAlg::Map> full_coupled_maps =
      Core::LinAlg::MultiMapExtractor::merge_maps(coupled_maps);

  // full map uncoupled -> all uncoupled dofs
  const Core::LinAlg::MapExtractor full_uncoupled_map_extractor(
      *dis.dof_row_map(), full_coupled_maps, false);
  const auto full_uncoupled_map =
      std::make_shared<Core::LinAlg::Map>(*full_uncoupled_map_extractor.cond_map());

  // vector for setup of extractor
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> full_map_vector;
  full_map_vector.push_back(full_uncoupled_map);
  full_map_vector.push_back(full_coupled_maps);

  map_extractor.setup(*dis.dof_row_map(), full_map_vector);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNodeBasedAlgorithm::setup_system(
    const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
    const std::shared_ptr<Core::LinAlg::Vector<double>> rhs,
    const std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_cont,
    const std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_art,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_cont,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_art,
    std::shared_ptr<const Core::LinAlg::MapExtractor> dbcmap_cont,
    std::shared_ptr<const Core::LinAlg::MapExtractor> dbcmap_art)
{
  setup_rhs(rhs, rhs_cont, rhs_art);
  setup_matrix(sysmat, sysmat_cont, *sysmat_art);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNodeBasedAlgorithm::setup_rhs(
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs,
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_homogenized,
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_artery)
{
  setup_global_vector(rhs, rhs_homogenized, rhs_artery);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNodeBasedAlgorithm::setup_global_vector(
    const std::shared_ptr<Core::LinAlg::Vector<double>> global_vector,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> homogenized_vector,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> artery_vector)
{
  // zero out
  global_vector->put_scalar(0.0);

  // inner (uncoupled) DOFs of artery
  std::shared_ptr<Core::LinAlg::Vector<double>> vec2_uncoupled =
      artery_field_extractor_->extract_vector(*artery_vector, 0);

  // boundary (coupled) DOFs of artery
  std::shared_ptr<Core::LinAlg::Vector<double>> vec2_coupled =
      artery_field_extractor_->extract_vector(*artery_vector, 1);

  // transform boundary DOFs to continuous dis
  std::shared_ptr<Core::LinAlg::Vector<double>> temp = homogenized_field_extractor_->insert_vector(
      *coupling_artery_homogenized_->slave_to_master(*vec2_coupled), 1);

  // add to continuous vec
  temp->update(1.0, *homogenized_vector, 1.0);

  // set up global vector
  global_extractor_->insert_vector(*temp, 0, *global_vector);
  global_extractor_->insert_vector(*vec2_uncoupled, 1, *global_vector);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNodeBasedAlgorithm::setup_matrix(
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
    std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_homogenized,
    Core::LinAlg::SparseMatrix& sysmat_artery) const
{
  sysmat_homogenized->un_complete();

  // artery
  // first split the matrix into 2x2 blocks (boundary vs. inner dofs)
  const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> artery_blocks =
      Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
          sysmat_artery, *(artery_field_extractor_), *(artery_field_extractor_));
  artery_blocks->complete();

  // inner artery dofs
  sysmat->assign(1, 1, Core::LinAlg::DataAccess::View, artery_blocks->matrix(0, 0));

  (*col_transform_)(artery_blocks->full_row_map(), artery_blocks->full_col_map(),
      artery_blocks->matrix(0, 1), 1.0,
      Coupling::Adapter::CouplingSlaveConverter(*coupling_artery_homogenized_),
      sysmat->matrix(1, 0));

  (*row_transform_)(artery_blocks->matrix(1, 0), 1.0,
      Coupling::Adapter::CouplingSlaveConverter(*coupling_artery_homogenized_),
      sysmat->matrix(0, 1));

  (*row_col_transform_)(artery_blocks->matrix(1, 1), 1.0,
      Coupling::Adapter::CouplingSlaveConverter(*coupling_artery_homogenized_),
      Coupling::Adapter::CouplingSlaveConverter(*coupling_artery_homogenized_), *sysmat_homogenized,
      true, true);

  // continuous field
  sysmat->assign(0, 0, Core::LinAlg::DataAccess::View, *sysmat_homogenized);
  // complete
  sysmat->complete();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNodeBasedAlgorithm::
    extract_single_field_vectors(std::shared_ptr<const Core::LinAlg::Vector<double>> global_vector,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& homogenized_vector,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& artery_vector)
{
  // process second field (continuous)
  homogenized_vector = global_extractor_->extract_vector(*global_vector, 0);

  // process coupled (boundary) DOFs of the second field
  const std::shared_ptr<Core::LinAlg::Vector<double>> boundary_dofs =
      homogenized_field_extractor_->extract_vector(*homogenized_vector, 1);

  // process inner (uncoupled) and boundary (coupled) DOFs of artery
  const std::shared_ptr<const Core::LinAlg::Vector<double>> artery_inner_dofs =
      global_extractor_->extract_vector(*global_vector, 1);
  const std::shared_ptr<Core::LinAlg::Vector<double>> artery_boundary_dofs =
      coupling_artery_homogenized_->master_to_slave(*boundary_dofs);

  // build vector for artery
  // 1) inner DOFs
  const std::shared_ptr<Core::LinAlg::Vector<double>> artery_temp =
      artery_field_extractor_->insert_vector(*artery_inner_dofs, 0);
  // 2) boundary DOFs
  artery_field_extractor_->insert_vector(*artery_boundary_dofs, 1, *artery_temp);

  artery_vector = artery_temp;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNodeBasedAlgorithm::
    check_dbc_on_coupled_dofs(const Core::FE::Discretization& dis,
        const std::shared_ptr<const Core::LinAlg::Map>& coupled_dof_map) const
{
  // object holds maps/subsets for DOFs subjected to Dirichlet BCs and otherwise
  const auto dbc_maps = std::make_shared<Core::LinAlg::MapExtractor>();
  {
    const std::shared_ptr<Core::LinAlg::Vector<double>> zeros =
        Core::LinAlg::create_vector(*dis.dof_row_map(), true);
    Teuchos::ParameterList ele_params;
    // other parameters needed by the elements
    ele_params.set("total time", 0.0);
    ele_params.set<const Core::Utils::FunctionManager*>(
        "function_manager", &Global::Problem::instance()->function_manager());
    dis.evaluate_dirichlet(ele_params, zeros, nullptr, nullptr, nullptr, dbc_maps);
  }
  // intersect DBC maps and coupled dof map to check if coupling and DBC are applied on same dofs
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> maps_vector;
  maps_vector.push_back(dbc_maps->cond_map());
  maps_vector.push_back(coupled_dof_map);
  const std::shared_ptr<Core::LinAlg::Map> intersect_dbc_coupled =
      Core::LinAlg::MultiMapExtractor::intersect_maps(maps_vector);

  if (intersect_dbc_coupled->num_global_elements() > 0)
  {
    if (my_mpi_rank_ == 0)
    {
      std::cout << "\n\n";
      std::cout << "You cannot define DBC and nodal coupling conditions on the same node\n"
                   "for discretization "
                << dis.name()
                << "\n"
                   "The problematic DOFs are:"
                << '\n';
    }
    intersect_dbc_coupled->print(std::cout);
    FOUR_C_THROW("Review you input file.");
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNodeBasedAlgorithm::check_initial_fields(
    const std::shared_ptr<const Core::LinAlg::Vector<double>> homogenized_vector,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> artery_vector)
{
  // boundary (coupled) DOFs of artery
  const std::shared_ptr<Core::LinAlg::Vector<double>> artery_coupled_dofs =
      artery_field_extractor_->extract_vector(*artery_vector, 1);

  // transform boundary DOFs to homogenized discretization
  const std::shared_ptr<Core::LinAlg::Vector<double>> transformed_coupled_dofs =
      coupling_artery_homogenized_->slave_to_master(*artery_coupled_dofs);

  // process coupled (boundary) DOFs of the second field
  const std::shared_ptr<Core::LinAlg::Vector<double>> homogenized_coupled_dofs =
      homogenized_field_extractor_->extract_vector(*homogenized_vector, 1);

  // subtract artery DOF values from continuous DOF values
  homogenized_coupled_dofs->update(-1.0, *transformed_coupled_dofs, 1.0);

  // build L2 norm
  double difference(0.0);
  homogenized_coupled_dofs->norm_2(&difference);

  if (difference > 1.0e-9)
  {
    FOUR_C_THROW("Your initial fields differ with an L2 norm of {}", difference);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map>
PoroPressureBased::PorofluidElastScatraArteryCouplingNodeBasedAlgorithm::artery_dof_row_map() const
{
  return artery_field_extractor_->map(0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map>
PoroPressureBased::PorofluidElastScatraArteryCouplingNodeBasedAlgorithm::dof_row_map() const
{
  return fullmap_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNodeBasedAlgorithm::apply_mesh_movement()
{
  if (!evaluate_in_ref_config_)
    FOUR_C_THROW("Evaluation in current configuration not possible for node-based coupling");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>> PoroPressureBased::
    PorofluidElastScatraArteryCouplingNodeBasedAlgorithm::blood_vessel_volume_fraction()
{
  FOUR_C_THROW("Output of vessel volume fraction not possible for node-based coupling");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNodeBasedAlgorithm::
    print_coupling_method() const
{
  std::cout << "<   Coupling-Method : Node-based                   >" << '\n';
}

FOUR_C_NAMESPACE_CLOSE
