// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_str_model_evaluator.hpp"

#include "4C_beam3_base.hpp"
#include "4C_beamcontact_input.hpp"
#include "4C_beaminteraction_beam_to_solid_edge_contact_params.hpp"
#include "4C_beaminteraction_beam_to_solid_volume_meshtying_params.hpp"
#include "4C_beaminteraction_calc_utils.hpp"
#include "4C_beaminteraction_contact_params.hpp"
#include "4C_beaminteraction_crosslinker_handler.hpp"
#include "4C_beaminteraction_crosslinker_node.hpp"
#include "4C_beaminteraction_data.hpp"
#include "4C_beaminteraction_str_model_evaluator_datastate.hpp"
#include "4C_beaminteraction_submodel_evaluator_beamcontact.hpp"
#include "4C_beaminteraction_submodel_evaluator_crosslinking.hpp"
#include "4C_beaminteraction_submodel_evaluator_factory.hpp"
#include "4C_beaminteraction_submodel_evaluator_generic.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_fem_general_utils_createdis.hpp"
#include "4C_fem_geometry_periodic_boundingbox.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_beam_to_solid.hpp"
#include "4C_io.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_rebalance_print.hpp"
#include "4C_rigidsphere.hpp"
#include "4C_scatra_ele.hpp"
#include "4C_solid_3D_ele.hpp"
#include "4C_structure_new_model_evaluator_data.hpp"
#include "4C_structure_new_timint_base.hpp"
#include "4C_structure_new_utils.hpp"
#include "4C_utils_parameter_list.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::ModelEvaluator::BeamInteraction::BeamInteraction()
    : discret_ptr_(nullptr),
      beaminteraction_params_ptr_(nullptr),
      submodeltypes_(nullptr),
      me_map_ptr_(nullptr),
      me_vec_ptr_(nullptr),
      myrank_(-1),
      coupsia_(nullptr),
      siatransform_(nullptr),
      ia_discret_(nullptr),
      eletypeextractor_(nullptr),
      ia_state_ptr_(nullptr),
      ia_force_beaminteraction_(nullptr),
      force_beaminteraction_(nullptr),
      stiff_beaminteraction_(nullptr),
      beam_crosslinker_handler_(nullptr),
      binstrategy_(nullptr),
      bindis_(nullptr),
      rowbins_(nullptr),
      dis_at_last_redistr_(nullptr),
      half_interaction_distance_(-1.0)
{
  // empty
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BeamInteraction::setup()
{
  check_init();

  // -------------------------------------------------------------------------
  // setup variables
  // -------------------------------------------------------------------------
  // discretization pointer
  discret_ptr_ = std::dynamic_pointer_cast<Core::FE::Discretization>(discret_ptr());
  // stiff
  stiff_beaminteraction_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *global_state().dof_row_map_view(), 81, true, true);
  // force and displacement at last redistribution
  force_beaminteraction_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*global_state().dof_row_map(), true);
  dis_at_last_redistr_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*global_state().dof_row_map(), true);
  // get myrank
  myrank_ = Core::Communication::my_mpi_rank(discret_ptr()->get_comm());

  beaminteraction_params_ptr_ = std::make_shared<FourC::BeamInteraction::BeamInteractionParams>();
  beaminteraction_params_ptr_->init();
  beaminteraction_params_ptr_->setup();

  // print logo
  logo();

  // set submodel types
  set_sub_model_types();

  // -------------------------------------------------------------------------
  // clone problem discretization, the idea is simple: we redistribute only
  // the new discretization to enable all interactions (including the required
  // search), calculate the resulting force and stiffness contributions, export
  // them to our initial discretization where all evaluation, assembly and
  // solving is done. Therefore the maps of our initial discretization don't
  // change, i.e. there is no need to rebuild the global state.
  // -------------------------------------------------------------------------
  std::shared_ptr<Core::FE::DiscretizationCreatorBase> discloner =
      std::make_shared<Core::FE::DiscretizationCreatorBase>();
  ia_discret_ = discloner->create_matching_discretization(
      *discret_ptr_, "ia_structure", true, true, false, true);
  // create discretization writer
  ia_discret_->set_writer(std::make_shared<Core::IO::DiscretizationWriter>(ia_discret_,
      Global::Problem::instance()->output_control_file(),
      Global::Problem::instance()->spatial_approximation_type()));

  // init data container
  ia_state_ptr_ = std::make_shared<Solid::ModelEvaluator::BeamInteractionDataState>();
  ia_state_ptr_->init();
  ia_state_ptr_->setup(ia_discret_);

  ia_state_ptr_->get_dis_np() =
      std::make_shared<Core::LinAlg::Vector<double>>(*global_state_ptr()->get_dis_np());
  FourC::BeamInteraction::Utils::periodic_boundary_consistent_dis_vector(
      *ia_state_ptr_->get_dis_np(), *tim_int().get_data_sdyn_ptr()->get_periodic_bounding_box(),
      *ia_discret_);

  // -------------------------------------------------------------------------
  // initialize coupling adapter to transform matrices between the two discrets
  // (with distinct parallel distribution)
  // -------------------------------------------------------------------------
  coupsia_ = std::make_shared<Coupling::Adapter::Coupling>();
  siatransform_ = std::make_shared<Coupling::Adapter::MatrixRowTransform>();

  // -------------------------------------------------------------------------
  // initialize and setup binning strategy and beam crosslinker handler
  // -------------------------------------------------------------------------
  // construct, init and setup binning strategy
  std::vector<std::shared_ptr<Core::FE::Discretization>> discret_vec(1, ia_discret_);

  // We have to pass the displacement column vector to the initialization of the binning strategy.
  ia_state_ptr_->get_dis_col_np() =
      std::make_shared<Core::LinAlg::Vector<double>>(*ia_discret_->dof_col_map());
  Core::LinAlg::export_to(*ia_state_ptr_->get_dis_np(), *ia_state_ptr_->get_dis_col_np());

  std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>> disp_vec(
      1, ia_state_ptr_->get_dis_col_np());
  Teuchos::ParameterList binning_params = Global::Problem::instance()->binning_strategy_params();
  Core::Utils::add_enum_class_to_parameter_list<Core::FE::ShapeFunctionType>(
      "spatial_approximation_type", Global::Problem::instance()->spatial_approximation_type(),
      binning_params);

  auto correct_node = [](const Core::Nodes::Node& node) -> decltype(auto)
  {
    const Core::Elements::Element* element = node.adjacent_elements()[0].user_element();
    const auto* beamelement = dynamic_cast<const Discret::Elements::Beam3Base*>(element);
    if (beamelement != nullptr && !beamelement->is_centerline_node(node))
      return *element->nodes()[0];
    else
      return node;
  };

  auto determine_relevant_points = [correct_node](const Core::FE::Discretization& discret,
                                       const Core::Elements::Element& ele,
                                       std::shared_ptr<const Core::LinAlg::Vector<double>> disnp)
      -> std::vector<std::array<double, 3>>
  {
    if (dynamic_cast<const Discret::Elements::Beam3Base*>(&ele))
    {
      return Core::Binstrategy::DefaultRelevantPoints{
          .correct_node = correct_node,
      }(discret, ele, disnp);
    }
    else if (ele.element_type() == Discret::Elements::RigidsphereType::instance())
    {
      double currpos[3] = {0.0, 0.0, 0.0};
      Core::Binstrategy::Utils::get_current_node_pos(discret, ele.nodes()[0], disnp, currpos);
      const double radius = dynamic_cast<const Discret::Elements::Rigidsphere&>(ele).radius();
      return {{currpos[0] - radius, currpos[1] - radius, currpos[2] - radius},
          {currpos[0] + radius, currpos[1] + radius, currpos[2] + radius}};
    }
    else
      return Core::Binstrategy::DefaultRelevantPoints{}(discret, ele, disnp);
  };

  binstrategy_ = std::make_shared<Core::Binstrategy::BinningStrategy>(binning_params,
      Global::Problem::instance()->output_control_file(), ia_discret_->get_comm(),
      Core::Communication::my_mpi_rank(ia_discret_->get_comm()), correct_node,
      determine_relevant_points, discret_vec, disp_vec);

  binstrategy_->set_deforming_binning_domain_handler(
      tim_int().get_data_sdyn_ptr()->get_periodic_bounding_box());

  bindis_ = binstrategy_->bin_discret();

  // construct, init and setup beam crosslinker handler and binning strategy
  // todo: move this and its single call during partition to crosslinker submodel
  if (have_sub_model_type(Inpar::BeamInteraction::submodel_crosslinking))
  {
    beam_crosslinker_handler_ = std::make_shared<FourC::BeamInteraction::BeamCrosslinkerHandler>();
    beam_crosslinker_handler_->init(global_state().get_my_rank(), binstrategy_);
    beam_crosslinker_handler_->setup();
  }

  // some screen output for binning
  print_binning_info_to_screen();

  // extract map for each eletype that is in discretization
  eletypeextractor_ = std::make_shared<FourC::BeamInteraction::Utils::MapExtractor>();
  FourC::BeamInteraction::Utils::setup_ele_type_map_extractor(*ia_discret_, *eletypeextractor_);

  // initialize and setup submodel evaluators
  init_and_setup_sub_model_evaluators();

  // distribute problem according to bin distribution to procs ( in case of restart
  // partitioning is done during read_restart() )
  if (not Global::Problem::instance()->restart()) partition_problem();

  // some actions need a partitioned system followed by a renewal of the partition
  if (not Global::Problem::instance()->restart() and post_partition_problem()) partition_problem();

  // only call post_setup at this stage for non restarted simulations
  if (not Global::Problem::instance()->restart()) post_setup_submodels();

  // some screen output
  Core::Rebalance::print_parallel_distribution(*ia_discret_);
  Core::Rebalance::print_parallel_distribution(*bindis_);

  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BeamInteraction::post_setup_submodels()
{
  check_init();

  // post setup submodel loop
  Vector::iterator some_iter;
  for (Vector::iterator some_iter = me_vec_ptr_->begin(); some_iter != me_vec_ptr_->end();
      ++some_iter)
    (*some_iter)->post_setup();

  if (beaminteraction_params_ptr_->get_repartition_strategy() ==
      Inpar::BeamInteraction::repstr_adaptive)
  {
    // submodel loop to determine half interaction radius
    for (some_iter = me_vec_ptr_->begin(); some_iter != me_vec_ptr_->end(); ++some_iter)
      (*some_iter)->get_half_interaction_distance(half_interaction_distance_);

    if (global_state().get_my_rank() == 0)
      std::cout << " half min bin size " << 0.5 * binstrategy_->get_min_bin_size() << std::endl;

    // safety checks
    if (half_interaction_distance_ > (0.5 * binstrategy_->get_min_bin_size()))
      FOUR_C_THROW(
          "Your half interaction distance {} is larger than half your smallest bin {}. You will "
          "not be\n"
          "able to track all interactions like this. Increase bin size?",
          half_interaction_distance_, 0.5 * binstrategy_->get_min_bin_size());
    if (half_interaction_distance_ < 0)
      FOUR_C_THROW("At least one model needs to define half interaction radius");
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BeamInteraction::set_sub_model_types()
{
  check_init();

  submodeltypes_ = std::make_shared<std::set<enum Inpar::BeamInteraction::SubModelType>>();

  // ---------------------------------------------------------------------------
  // check for crosslinking in biopolymer networks
  // ---------------------------------------------------------------------------
  if (Global::Problem::instance()
          ->beam_interaction_params()
          .sublist("SPHERE BEAM LINK")
          .get<bool>("SPHEREBEAMLINKING"))
    submodeltypes_->insert(Inpar::BeamInteraction::submodel_spherebeamlink);

  // ---------------------------------------------------------------------------
  // check for crosslinking in biopolymer networks
  // ---------------------------------------------------------------------------
  if (Global::Problem::instance()
          ->beam_interaction_params()
          .sublist("CROSSLINKING")
          .get<bool>("CROSSLINKER"))
    submodeltypes_->insert(Inpar::BeamInteraction::submodel_crosslinking);

  // ---------------------------------------------------------------------------
  // check for point to point penalty coupling conditions
  // ---------------------------------------------------------------------------

  // conditions for beam penalty point coupling
  std::vector<const Core::Conditions::Condition*> beampenaltycouplingconditions;
  discret_ptr_->get_condition("PenaltyPointCouplingCondition", beampenaltycouplingconditions);
  if (beampenaltycouplingconditions.size() > 0)
    submodeltypes_->insert(Inpar::BeamInteraction::submodel_beamcontact);

  // ---------------------------------------------------------------------------
  // check for beam contact
  // ---------------------------------------------------------------------------
  if (Teuchos::getIntegralValue<Inpar::BeamInteraction::Strategy>(
          Global::Problem::instance()->beam_interaction_params().sublist("BEAM TO BEAM CONTACT"),
          "STRATEGY") != Inpar::BeamInteraction::bstr_none or
      Teuchos::getIntegralValue<Inpar::BeamInteraction::Strategy>(
          Global::Problem::instance()->beam_interaction_params().sublist("BEAM TO SPHERE CONTACT"),
          "STRATEGY") != Inpar::BeamInteraction::bstr_none or
      Teuchos::getIntegralValue<Inpar::BeamToSolid::BeamToSolidContactDiscretization>(
          Global::Problem::instance()->beam_interaction_params().sublist(
              "BEAM TO SOLID VOLUME MESHTYING"),
          "CONTACT_DISCRETIZATION") != Inpar::BeamToSolid::BeamToSolidContactDiscretization::none or
      Teuchos::getIntegralValue<Inpar::BeamToSolid::BeamToSolidContactDiscretization>(
          Global::Problem::instance()->beam_interaction_params().sublist(
              "BEAM TO SOLID SURFACE MESHTYING"),
          "CONTACT_DISCRETIZATION") != Inpar::BeamToSolid::BeamToSolidContactDiscretization::none or
      Teuchos::getIntegralValue<Inpar::BeamToSolid::BeamToSolidContactDiscretization>(
          Global::Problem::instance()->beam_interaction_params().sublist(
              "BEAM TO SOLID SURFACE CONTACT"),
          "CONTACT_DISCRETIZATION") != Inpar::BeamToSolid::BeamToSolidContactDiscretization::none or
      Global::Problem::instance()->parameters().isParameter(
          "BEAM INTERACTION/BEAM TO SOLID EDGE CONTACT"))
    submodeltypes_->insert(Inpar::BeamInteraction::submodel_beamcontact);

  // ---------------------------------------------------------------------------
  // check for beam potential-based interactions
  // ---------------------------------------------------------------------------
  std::vector<const Core::Conditions::Condition*> beampotconditions;
  discret().get_condition("BeamPotentialLineCharge", beampotconditions);
  if (beampotconditions.size() > 0)
    submodeltypes_->insert(Inpar::BeamInteraction::submodel_potential);

  // extract beam to beam contact conditions
  std::vector<const Core::Conditions::Condition*> beamtobeamcontactconditions;
  discret_ptr_->get_condition("BeamToBeamContact", beamtobeamcontactconditions);

  // Ensure that no point coupling condition connects two beams that are possibly in contact with
  // each other.
  for (auto beam_point_coupling_condition : beampenaltycouplingconditions)
  {
    // get all nodes of beam to beam point coupling condition
    const std::vector<int>& nodes = *(beam_point_coupling_condition->get_nodes());
    if (nodes.size() != 2)
      FOUR_C_THROW("Beam-to-beam point coupling condition should contain exactly two nodes.");
    for (auto beam_to_beam_contact_condition : beamtobeamcontactconditions)
    {
      if (beam_to_beam_contact_condition->contains_node(nodes[0]) and
          beam_to_beam_contact_condition->contains_node(nodes[1]))
      {
        FOUR_C_THROW(
            "It is not possible to use beam-to-beam contact in combination with "
            "beam-to-beam point coupling for the same node pair. Please reconsider the specified "
            "interaction conditions with the ids {} and {}, since they are intersecting at the "
            "nodes with ids {} and {}",
            beam_to_beam_contact_condition->id(), beam_point_coupling_condition->id(), nodes[0],
            nodes[1]);
      }
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BeamInteraction::init_and_setup_sub_model_evaluators()
{
  check_init();

  // model map
  me_map_ptr_ = FourC::BeamInteraction::SubmodelEvaluator::build_model_evaluators(*submodeltypes_);
  std::vector<enum Inpar::BeamInteraction::SubModelType> sorted_submodeltypes;

  // build and sort submodel vector
  me_vec_ptr_ = transform_to_vector(*me_map_ptr_, sorted_submodeltypes);

  Vector::iterator some_iter;
  for (some_iter = (*me_vec_ptr_).begin(); some_iter != (*me_vec_ptr_).end(); ++some_iter)
  {
    (*some_iter)
        ->init(ia_discret_, bindis_, global_state_ptr(), global_in_output_ptr(), ia_state_ptr_,
            beam_crosslinker_handler_, binstrategy_,
            tim_int().get_data_sdyn_ptr()->get_periodic_bounding_box(),
            std::dynamic_pointer_cast<FourC::BeamInteraction::Utils::MapExtractor>(
                eletypeextractor_));
    (*some_iter)->setup();
  }

  // submodels build their pointer to other submodel objects to enable submodel dependencies
  // this is not particularly nice, at least the nicest way to handle such dependencies
  for (some_iter = me_vec_ptr_->begin(); some_iter != me_vec_ptr_->end(); ++some_iter)
    (*some_iter)->init_submodel_dependencies(me_map_ptr_);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Solid::ModelEvaluator::BeamInteraction::Vector>
Solid::ModelEvaluator::BeamInteraction::transform_to_vector(
    Solid::ModelEvaluator::BeamInteraction::Map submodel_map,
    std::vector<Inpar::BeamInteraction::SubModelType>& sorted_submodel_types) const
{
  std::shared_ptr<Solid::ModelEvaluator::BeamInteraction::Vector> me_vec_ptr =
      std::make_shared<Solid::ModelEvaluator::BeamInteraction::Vector>(0);

  Solid::ModelEvaluator::BeamInteraction::Map::iterator miter;

  // if there is a contractile cell submodel, put in first place
  miter = submodel_map.find(Inpar::BeamInteraction::submodel_spherebeamlink);
  if (miter != submodel_map.end())
  {
    // put it in first place
    me_vec_ptr->push_back(miter->second);
    sorted_submodel_types.push_back(miter->first);
    submodel_map.erase(miter);
  }

  // insert the remaining submodel evaluators into the model vector
  for (miter = submodel_map.begin(); miter != submodel_map.end(); ++miter)
  {
    me_vec_ptr->push_back(miter->second);
    sorted_submodel_types.push_back(miter->first);
  }

  return me_vec_ptr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::BeamInteraction::have_sub_model_type(
    Inpar::BeamInteraction::SubModelType const& submodeltype) const
{
  check_init();
  return (submodeltypes_->find(submodeltype) != submodeltypes_->end());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BeamInteraction::partition_problem()
{
  check_init();

  // store structure discretization in vector
  std::vector<std::shared_ptr<Core::FE::Discretization>> discret_vec(1, ia_discret_);

  // displacement vector according to periodic boundary conditions
  std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> mutabledisnp(
      1, std::make_shared<Core::LinAlg::Vector<double>>(*ia_discret_->dof_col_map()));
  Core::LinAlg::export_to(*ia_state_ptr_->get_dis_np(), *mutabledisnp[0]);

  std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>> disnp(
      1, std::make_shared<Core::LinAlg::Vector<double>>(*mutabledisnp[0]));

  // nodes, that are owned by a proc, are distributed to the bins of this proc
  std::vector<std::map<int, std::vector<int>>> nodesinbin(1);

  // weight for load balancing regarding the distribution of bins to procs
  // (this is experimental, choose what gives you best results)
  double const weight = 1.0;
  // get optimal row distribution of bins to procs
  rowbins_ =
      binstrategy_->weighted_distribution_of_bins_to_procs(discret_vec, disnp, nodesinbin, weight);

  // extract noderowmap because it will be called reset() after adding elements
  std::shared_ptr<Core::LinAlg::Map> noderowmap =
      std::make_shared<Core::LinAlg::Map>(*bindis_->node_row_map());
  // delete old bins ( in case you partition during your simulation or after a restart)
  bindis_->delete_elements();
  binstrategy_->fill_bins_into_bin_discretization(*rowbins_);

  // now node (=crosslinker) to bin (=element) relation needs to be
  // established in binning discretization. Therefore some nodes need to
  // change their owner according to the bins owner they reside in
  if (have_sub_model_type(Inpar::BeamInteraction::submodel_crosslinking))
    beam_crosslinker_handler_->distribute_linker_to_bins(noderowmap);

  // determine boundary bins (physical boundary as well as boundary to other procs)
  binstrategy_->determine_boundary_row_bins();

  // determine one layer ghosting around boundary bins determined in previous step
  binstrategy_->determine_boundary_col_bins();

  // standard ghosting (if a proc owns a part of nodes (and therefore dofs) of
  // an element, the element and the rest of its nodes and dofs are ghosted
  std::shared_ptr<Core::LinAlg::Map> stdelecolmap;
  std::shared_ptr<Core::LinAlg::Map> stdnodecolmapdummy;
  binstrategy_->standard_discretization_ghosting(
      ia_discret_, *rowbins_, ia_state_ptr_->get_dis_np(), stdelecolmap, stdnodecolmapdummy);

  // distribute elements that can be cut by the periodic boundary to bins
  std::shared_ptr<Core::LinAlg::Vector<double>> iadiscolnp =
      std::make_shared<Core::LinAlg::Vector<double>>(*ia_discret_->dof_col_map());
  Core::LinAlg::export_to(*ia_state_ptr_->get_dis_np(), *iadiscolnp);

  binstrategy_->distribute_elements_to_bins_using_ele_aabb(*ia_discret_,
      ia_discret_->my_row_element_range(), ia_state_ptr_->get_bin_to_row_ele_map(), iadiscolnp);

  // build row elements to bin map
  build_row_ele_to_bin_map();

  // extend ghosting
  extend_ghosting();

  // assign Elements to bins
  binstrategy_->remove_all_eles_from_bins();
  binstrategy_->assign_eles_to_bins(*ia_discret_, ia_state_ptr_->get_extended_bin_to_row_ele_map(),
      FourC::BeamInteraction::Utils::convert_element_to_bin_content_type);

  // update maps of state vectors and matrices
  update_maps();

  // reset transformation
  update_coupling_adapter_and_matrix_transformation();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::BeamInteraction::post_partition_problem()
{
  check_init();

  bool repartition = false;

  Vector::iterator some_iter;
  for (some_iter = me_vec_ptr_->begin(); some_iter != me_vec_ptr_->end(); ++some_iter)
    repartition = (*some_iter)->post_partition_problem() ? true : repartition;

  return repartition;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BeamInteraction::extend_ghosting()
{
  TEUCHOS_FUNC_TIME_MONITOR("Solid::ModelEvaluator::BeamInteraction::extend_ghosting");

  ia_state_ptr_->get_extended_bin_to_row_ele_map().clear();

  check_init();

  std::set<int> colbins;

  // first, add default one layer ghosting
  std::set<int> const& boundcolbins = binstrategy_->boundary_col_bins_ids();
  colbins.insert(boundcolbins.begin(), boundcolbins.end());

  // if the bounding box of a row element of myrank touches a boundary col bin, we need
  // to ghost its neighborhood as well
  std::map<int, std::set<int>>::const_iterator it;
  std::vector<int> binvec(27);
  for (it = ia_state_ptr_->get_bin_to_row_ele_map().begin();
      it != ia_state_ptr_->get_bin_to_row_ele_map().end(); ++it)
  {
    // not doing the following if is only valid if you ensure that the largest element
    // in the discretization (in deformed state) is smaller than the smallest bin size
    // which is not necessarily needed e.g. for beam contact
    //    if( boundcolbins.find( it->first ) != boundcolbins.end() )
    {
      binstrategy_->get_neighbor_and_own_bin_ids(it->first, binvec);
      colbins.insert(binvec.begin(), binvec.end());
      binvec.clear();
    }
  }

  // enable submodel specific ghosting contributions to bin col map
  Vector::iterator some_iter;
  for (some_iter = me_vec_ptr_->begin(); some_iter != me_vec_ptr_->end(); ++some_iter)
    (*some_iter)->add_bins_to_bin_col_map(colbins);

  // 1) extend ghosting of bin discretization
  // todo: think about if you really need to assign degrees of freedom for crosslinker
  // (now only needed in case you want to write output)
  binstrategy_->extend_ghosting_of_binning_discretization(*rowbins_, colbins, true);

  // add submodel specific bins whose content should be ghosted in problem discret
  for (some_iter = me_vec_ptr_->begin(); some_iter != me_vec_ptr_->end(); ++some_iter)
    (*some_iter)->add_bins_with_relevant_content_for_ia_discret_col_map(colbins);

  // build auxiliary bin col map
  std::vector<int> auxgids(colbins.begin(), colbins.end());
  std::shared_ptr<Core::LinAlg::Map> auxmap = std::make_shared<Core::LinAlg::Map>(
      -1, static_cast<int>(auxgids.size()), auxgids.data(), 0, bindis_->get_comm());

  std::shared_ptr<Core::LinAlg::Map> ia_elecolmap = binstrategy_->extend_element_col_map(
      ia_state_ptr_->get_bin_to_row_ele_map(), ia_state_ptr_->get_bin_to_row_ele_map(),
      ia_state_ptr_->get_extended_bin_to_row_ele_map(), auxmap);

  // 2) extend ghosting of discretization
  Core::Binstrategy::Utils::extend_discretization_ghosting(
      *ia_discret_, *ia_elecolmap, true, false, true);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BeamInteraction::reset(const Core::LinAlg::Vector<double>& x)
{
  check_init_setup();

  // todo: somewhat illegal as of const correctness
  tim_int().get_data_sdyn_ptr()->get_periodic_bounding_box()->apply_dirichlet(
      global_state().get_time_n(), Global::Problem::instance()->function_manager());

  // get current displacement state and export to interaction discretization dofmap
  FourC::BeamInteraction::Utils::update_dof_map_of_vector(
      *ia_discret_, ia_state_ptr_->get_dis_np(), global_state().get_dis_np());
  FourC::BeamInteraction::Utils::periodic_boundary_consistent_dis_vector(
      *ia_state_ptr_->get_dis_np(), *tim_int().get_data_sdyn_ptr()->get_periodic_bounding_box(),
      *ia_discret_);

  // update column vector
  ia_state_ptr_->get_dis_col_np() =
      std::make_shared<Core::LinAlg::Vector<double>>(*ia_discret_->dof_col_map());
  Core::LinAlg::export_to(*ia_state_ptr_->get_dis_np(), *ia_state_ptr_->get_dis_col_np());

  // update restart displacement vector
  if (ia_state_ptr_->get_restart_coupling_flag())
  {
    ia_state_ptr_->get_dis_restart_col() =
        std::make_shared<Core::LinAlg::Vector<double>>(*ia_discret_->dof_col_map());
    Core::LinAlg::export_to(
        *ia_state_ptr_->get_dis_restart(), *ia_state_ptr_->get_dis_restart_col());
  }

  // submodel loop
  Vector::iterator some_iter;
  for (some_iter = me_vec_ptr_->begin(); some_iter != me_vec_ptr_->end(); ++some_iter)
    (*some_iter)->reset();

  // Zero out force and stiffness contributions
  force_beaminteraction_->put_scalar(0.0);
  ia_force_beaminteraction_->put_scalar(0.0);
  ia_state_ptr_->get_force_np()->put_scalar(0.0);
  stiff_beaminteraction_->zero();
  ia_state_ptr_->get_stiff()->zero();

  // update gidmap_ and exporter in matrix transform object
  // Note: we need this in every evaluation call (i.e. every iteration) because a change
  //       in the active set of element pairs changes the entries of the used coarse
  //       system stiffness matrix (because we only assemble non-zero values).
  //       Therefore, the graph of the matrix changes and also the required gidmap
  //       (even in computation with one processor)
  // note: this is only necessary if active sets change in consecutive iteration steps
  // ( as crosslinker for example are only updated each time step, we only need to do this
  // every time step)
  if (have_sub_model_type(Inpar::BeamInteraction::submodel_potential) ||
      have_sub_model_type(Inpar::BeamInteraction::submodel_beamcontact))
    update_coupling_adapter_and_matrix_transformation();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::BeamInteraction::evaluate_force()
{
  check_init_setup();

  Vector::iterator some_iter;
  for (some_iter = me_vec_ptr_->begin(); some_iter != me_vec_ptr_->end(); ++some_iter)
    (*some_iter)->evaluate_force();

  // do communication
  if (ia_state_ptr_->get_force_np()->complete() != 0) FOUR_C_THROW("GlobalAssemble failed");
  // add to non fe vector
  if (ia_force_beaminteraction_->update(1., *ia_state_ptr_->get_force_np(), 1.))
    FOUR_C_THROW("update went wrong");

  // transformation from ia_discret to problem discret
  transform_force();

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::BeamInteraction::evaluate_stiff()
{
  check_init_setup();

  ia_state_ptr_->get_stiff()->un_complete();

  Vector::iterator some_iter;
  for (some_iter = me_vec_ptr_->begin(); some_iter != me_vec_ptr_->end(); ++some_iter)
    (*some_iter)->evaluate_stiff();

  if (not ia_state_ptr_->get_stiff()->filled()) ia_state_ptr_->get_stiff()->complete();

  transform_stiff();

  if (not stiff_beaminteraction_->filled()) stiff_beaminteraction_->complete();

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::BeamInteraction::evaluate_force_stiff()
{
  check_init_setup();

  ia_state_ptr_->get_stiff()->un_complete();

  Vector::iterator some_iter;
  for (some_iter = me_vec_ptr_->begin(); some_iter != me_vec_ptr_->end(); ++some_iter)
    (*some_iter)->evaluate_force_stiff();

  // do communication
  if (ia_state_ptr_->get_force_np()->complete() != 0) FOUR_C_THROW("GlobalAssemble failed");

  // add to non fe vector
  if (ia_force_beaminteraction_->update(1., *ia_state_ptr_->get_force_np(), 1.))
    FOUR_C_THROW("update went wrong");
  if (not ia_state_ptr_->get_stiff()->filled()) ia_state_ptr_->get_stiff()->complete();

  transform_force_stiff();

  if (not stiff_beaminteraction_->filled()) stiff_beaminteraction_->complete();

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::BeamInteraction::have_lagrange_dofs() const
{
  bool lagrange_flag = false;
  for (std::size_t i = 0; i < me_vec_ptr_->size(); ++i)
  {
    auto beam_contact_model =
        std::dynamic_pointer_cast<FourC::BeamInteraction::SubmodelEvaluator::BeamContact const>(
            (*me_vec_ptr_)[i]);
    if (!beam_contact_model) continue;

    auto params = beam_contact_model->beam_contact_params().beam_to_solid_volume_meshtying_params();
    if (params && params->get_constraint_enforcement() ==
                      Inpar::BeamToSolid::BeamToSolidConstraintEnforcement::lagrange)
    {
      lagrange_flag = true;
      if (me_vec_ptr_->size() > 1)
        FOUR_C_THROW(
            "Currently Lagrange multipliers within beam interaction are exclusively supported for "
            "beam-to-solid interactions.");
    }
  }

  return lagrange_flag;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::BeamInteraction::assemble_force(
    Core::LinAlg::Vector<double>& f, const double& timefac_np) const
{
  check_init_setup();

  Core::LinAlg::assemble_my_vector(1.0, f, timefac_np, *force_beaminteraction_);

  if (have_lagrange_dofs())
  {
    auto beam_contact_model =
        std::dynamic_pointer_cast<FourC::BeamInteraction::SubmodelEvaluator::BeamContact const>(
            (*me_vec_ptr_)[0]);

    beam_contact_model->assemble_force(f);
  }

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::BeamInteraction::assemble_jacobian(
    Core::LinAlg::SparseOperator& jac, const double& timefac_np) const
{
  check_init_setup();

  std::shared_ptr<Core::LinAlg::SparseMatrix> jac_dd_ptr = global_state().extract_displ_block(jac);
  jac_dd_ptr->add(*stiff_beaminteraction_, false, timefac_np, 1.0);

  if (have_lagrange_dofs())
  {
    auto beam_contact_model =
        std::dynamic_pointer_cast<FourC::BeamInteraction::SubmodelEvaluator::BeamContact const>(
            (*me_vec_ptr_)[0]);

    beam_contact_model->assemble_stiff(jac);
  }

  // no need to keep it
  stiff_beaminteraction_->zero();
  ia_state_ptr_->get_stiff()->zero();

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BeamInteraction::write_restart(
    Core::IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const
{
  check_init_setup();

  int const stepn = global_state().get_step_n();
  double const timen = global_state().get_time_n();
  std::shared_ptr<Core::IO::DiscretizationWriter> ia_writer = ia_discret_->writer();
  std::shared_ptr<Core::IO::DiscretizationWriter> bin_writer = bindis_->writer();

  // write restart of ia_discret
  ia_writer->write_mesh(stepn, timen);
  ia_writer->new_step(stepn, timen);

  // mesh is not written to disc, only maximum node id is important for output
  // fixme: can we just write mesh
  bin_writer->write_only_nodes_in_new_field_group_to_control_file(stepn, timen, true);
  bin_writer->new_step(stepn, timen);

  // as we know that our maps have changed every time we write output, we can empty
  // the map cache as we can't get any advantage saving the maps anyway
  ia_writer->clear_map_cache();
  bin_writer->clear_map_cache();

  // sub model loop
  Vector::iterator some_iter;
  for (some_iter = me_vec_ptr_->begin(); some_iter != me_vec_ptr_->end(); ++some_iter)
    (*some_iter)->write_restart(*ia_writer, *bin_writer);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BeamInteraction::read_restart(Core::IO::DiscretizationReader& ioreader)
{
  check_init_setup();

  int const stepn = global_state().get_step_n();
  auto input_control_file = Global::Problem::instance()->input_control_file();

  // pre sub model loop
  Vector::iterator some_iter;
  for (some_iter = me_vec_ptr_->begin(); some_iter != me_vec_ptr_->end(); ++some_iter)
    (*some_iter)->pre_read_restart();

  // read interaction discretization
  Core::IO::DiscretizationReader ia_reader(ia_discret_, input_control_file, stepn);
  // includes fill_complete()
  ia_reader.read_history_data(stepn);

  // rebuild bin discret correctly in case crosslinker were present
  // Fixme: do just read history data like with ia discret
  // read correct nodes
  Core::IO::DiscretizationReader bin_reader(bindis_, input_control_file, stepn);
  bin_reader.read_nodes_only(stepn);
  bindis_->fill_complete(Core::FE::OptionsFillComplete::none());

  // need to read step next (as it was written next, do safety check)
  if (stepn != ia_reader.read_int("step") or stepn != bin_reader.read_int("step"))
    FOUR_C_THROW("Restart step not consistent with read restart step. ");

  // rebuild binning
  partition_problem();

  // sub model loop
  for (some_iter = me_vec_ptr_->begin(); some_iter != me_vec_ptr_->end(); ++some_iter)
    (*some_iter)->read_restart(ia_reader, bin_reader);

  // post sub model loop
  for (some_iter = me_vec_ptr_->begin(); some_iter != me_vec_ptr_->end(); ++some_iter)
    (*some_iter)->post_read_restart();

  // Check if we need to store the restart displacement in the data state container.
  const Teuchos::ParameterList& beam_interaction_params =
      Global::Problem::instance()->beam_interaction_params();
  if (have_sub_model_type(Inpar::BeamInteraction::submodel_beamcontact) &&
      beam_interaction_params.sublist("BEAM TO SOLID VOLUME MESHTYING")
          .get<bool>("COUPLE_RESTART_STATE"))
  {
    ia_state_ptr_->set_restart_coupling_flag(true);
    ia_state_ptr_->get_dis_restart() =
        std::make_shared<Core::LinAlg::Vector<double>>(*ia_state_ptr_->get_dis_np());
    ia_state_ptr_->get_dis_restart_col() =
        std::make_shared<Core::LinAlg::Vector<double>>(*ia_state_ptr_->get_dis_np());
  }
}

void Solid::ModelEvaluator::BeamInteraction::run_pre_compute_x(
    const Core::LinAlg::Vector<double>& xold, Core::LinAlg::Vector<double>& dir_mutable,
    const NOX::Nln::Group& curr_grp)
{
  if (have_lagrange_dofs()) Core::LinAlg::export_to(dir_mutable, *ia_state_ptr_->get_lambda());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BeamInteraction::run_post_compute_x(
    const Core::LinAlg::Vector<double>& xold, const Core::LinAlg::Vector<double>& dir,
    const Core::LinAlg::Vector<double>& xnew)
{
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BeamInteraction::run_post_iterate(const ::NOX::Solver::Generic& solver)
{
  check_init_setup();

  // submodel loop
  Vector::iterator some_iter;
  for (some_iter = me_vec_ptr_->begin(); some_iter != me_vec_ptr_->end(); ++some_iter)
    (*some_iter)->run_post_iterate(solver);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BeamInteraction::update_step_state(const double& timefac_n)
{
  check_init_setup();

  // add the old time factor scaled contributions to the residual
  std::shared_ptr<Core::LinAlg::Vector<double>>& fstructold_ptr =
      global_state().get_fstructure_old();

  fstructold_ptr->update(timefac_n, *force_beaminteraction_, 1.0);

  // submodel loop
  Vector::iterator some_iter;
  for (some_iter = me_vec_ptr_->begin(); some_iter != me_vec_ptr_->end(); ++some_iter)
    (*some_iter)->update_step_state(timefac_n);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BeamInteraction::update_step_element()
{
  check_init_setup();

  Vector::iterator some_iter;

  /* the idea is the following: redistribution of elements is only necessary if
   * one node on any proc has moved "too far" compared to the time step of the
   * last redistribution. Therefore we only do the expensive redistribution,
   * change of ghosting and assigning of elements if necessary.
   * This holds for both the beam discretization as well as the linker/binning
   * discretization.
   */

  // repartition every time
  bool beam_redist = check_if_beam_discret_redistribution_needs_to_be_done();

  // submodel loop
  bool binning_redist = false;
  for (some_iter = me_vec_ptr_->begin(); some_iter != me_vec_ptr_->end(); ++some_iter)
    binning_redist = (*some_iter)->pre_update_step_element(beam_redist) ? true : binning_redist;

  if (beam_redist)
  {
    binstrategy_->transfer_nodes_and_elements(
        *ia_discret_, ia_state_ptr_->get_dis_col_np(), ia_state_ptr_->get_bin_to_row_ele_map());

    build_row_ele_to_bin_map();

    // extend ghosting
    extend_ghosting();

    // assign Elements to bins
    binstrategy_->remove_all_eles_from_bins();
    binstrategy_->assign_eles_to_bins(*ia_discret_,
        ia_state_ptr_->get_extended_bin_to_row_ele_map(),
        FourC::BeamInteraction::Utils::convert_element_to_bin_content_type);

    // current displacement state gets new reference state
    dis_at_last_redistr_ =
        std::make_shared<Core::LinAlg::Vector<double>>(*global_state().get_dis_n());

    if (global_state().get_my_rank() == 0)
    {
      Core::IO::cout(Core::IO::verbose) << "\n************************************************\n"
                                        << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose) << "Complete redistribution was done " << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose) << "\n************************************************\n"
                                        << Core::IO::endl;
    }
  }
  else if (binning_redist)
  {
    extend_ghosting();
    binstrategy_->remove_all_eles_from_bins();
    binstrategy_->assign_eles_to_bins(*ia_discret_,
        ia_state_ptr_->get_extended_bin_to_row_ele_map(),
        FourC::BeamInteraction::Utils::convert_element_to_bin_content_type);

    if (global_state().get_my_rank() == 0)
    {
      Core::IO::cout(Core::IO::verbose) << "\n************************************************\n"
                                        << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose) << " binning redistribution was done " << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose) << "\n************************************************\n"
                                        << Core::IO::endl;
    }
  }

  // update maps of state vectors and matrices
  update_maps();

  // update coupling adapter, this should be done every time step as
  // interacting elements and therefore system matrix can change every time step
  update_coupling_adapter_and_matrix_transformation();

  // submodel loop update
  for (some_iter = me_vec_ptr_->begin(); some_iter != me_vec_ptr_->end(); ++some_iter)
    (*some_iter)->update_step_element(binning_redist || beam_redist);

  // submodel post update
  for (some_iter = me_vec_ptr_->begin(); some_iter != me_vec_ptr_->end(); ++some_iter)
    (*some_iter)->post_update_step_element();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::BeamInteraction::check_if_beam_discret_redistribution_needs_to_be_done()
{
  if (beaminteraction_params_ptr_->get_repartition_strategy() !=
      Inpar::BeamInteraction::repstr_adaptive)
    return true;

  Core::LinAlg::Vector<double> dis_increment(*global_state().dof_row_map(), true);
  int doflid[3];
  for (int i = 0; i < discret_ptr_->num_my_row_nodes(); ++i)
  {
    // get a pointer at i-th row node
    Core::Nodes::Node* node = discret_ptr_->l_row_node(i);

    /* Hermite Interpolation: Check whether node is a beam node which is NOT
     * used for centerline interpolation if so, we simply skip it because
     * it does not have position DoFs */
    if (FourC::BeamInteraction::Utils::is_beam_node(*node) and
        not FourC::BeamInteraction::Utils::is_beam_centerline_node(*node))
      continue;

    // get GIDs of this node's degrees of freedom
    std::vector<int> dofnode = discret_ptr_->dof(node);

    for (int dim = 0; dim < 3; ++dim)
    {
      // note: we are using the displacement vector of the problem discretization
      // ( this one also does not get shifted, therefore we do not need to worry
      // about a periodic boundary shift of a node between dis_at_last_redistr_ and the current
      // disp)
      doflid[dim] = dis_at_last_redistr_->get_map().lid(dofnode[dim]);
      (dis_increment).get_values()[doflid[dim]] =
          (*global_state().get_dis_np())[doflid[dim]] - (*dis_at_last_redistr_)[doflid[dim]];
    }
  }

  // get maximal displacement increment since last redistribution over all procs
  std::array<double, 2> extrema = {0.0, 0.0};
  dis_increment.min_value(&extrema[0]);
  dis_increment.max_value(&extrema[1]);
  double gmaxdisincr = std::max(-extrema[0], extrema[1]);

  // some verbose screen output
  if (global_state().get_my_rank() == 0)
  {
    Core::IO::cout(Core::IO::debug)
        << " half interaction distance " << half_interaction_distance_ << Core::IO::endl;
    Core::IO::cout(Core::IO::debug) << " gmaxdisincr " << gmaxdisincr << Core::IO::endl;
    Core::IO::cout(Core::IO::debug)
        << " half min bin size " << 0.5 * binstrategy_->get_min_bin_size() << Core::IO::endl;
  }

  return ((half_interaction_distance_ + gmaxdisincr) > (0.5 * binstrategy_->get_min_bin_size()));
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BeamInteraction::determine_stress_strain()
{
  // empty
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BeamInteraction::determine_energy()
{
  check_init_setup();

  std::map<Solid::EnergyType, double> energy_this_submodel;

  for (auto& submodel : (*me_vec_ptr_))
  {
    energy_this_submodel = submodel->get_energy();

    for (auto const& energy_type : energy_this_submodel)
      eval_data().add_contribution_to_energy_type(energy_type.second, energy_type.first);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BeamInteraction::determine_optional_quantity()
{
  // empty
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BeamInteraction::output_step_state(
    Core::IO::DiscretizationWriter& iowriter) const
{
  check_init_setup();

  Vector::iterator some_iter;
  for (some_iter = me_vec_ptr_->begin(); some_iter != me_vec_ptr_->end(); ++some_iter)
    (*some_iter)->output_step_state(iowriter);

  // visualize bins according to specification in input file ( MESHFREE -> WRITEBINS "" )
  binstrategy_->write_bin_output(global_state().get_step_n(), global_state().get_time_n());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BeamInteraction::runtime_output_step_state() const
{
  check_init_setup();

  Vector::iterator some_iter;
  for (some_iter = me_vec_ptr_->begin(); some_iter != me_vec_ptr_->end(); ++some_iter)
    (*some_iter)->runtime_output_step_state();

  tim_int().get_data_sdyn_ptr()->get_periodic_bounding_box()->runtime_output_step_state(
      global_state().get_time_n(), global_state().get_step_n());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map>
Solid::ModelEvaluator::BeamInteraction::get_block_dof_row_map_ptr() const
{
  check_init_setup();

  if (have_lagrange_dofs())
  {
    auto beam_contact_model =
        std::dynamic_pointer_cast<FourC::BeamInteraction::SubmodelEvaluator::BeamContact const>(
            (*me_vec_ptr_)[0]);

    return beam_contact_model->get_lagrange_map();
  }
  else
  {
    return global_state().dof_row_map();
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
Solid::ModelEvaluator::BeamInteraction::get_current_solution_ptr() const
{
  // there are no model specific solution entries
  return nullptr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
Solid::ModelEvaluator::BeamInteraction::get_last_time_step_solution_ptr() const
{
  // there are no model specific solution entries
  return nullptr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BeamInteraction::post_output()
{
  //  tim_int().get_data_sdyn_ptr()->get_periodic_bounding_box()->ApplyDirichlet(
  //  global_state().get_time_n()
  //  );
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BeamInteraction::reset_step_state()
{
  Vector::iterator some_iter;
  for (some_iter = me_vec_ptr_->begin(); some_iter != me_vec_ptr_->end(); ++some_iter)
    (*some_iter)->reset_step_state();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BeamInteraction::update_coupling_adapter_and_matrix_transformation()
{
  check_init();

  TEUCHOS_FUNC_TIME_MONITOR(
      "Solid::ModelEvaluator::BeamInteraction::update_coupling_adapter_and_matrix_transformation");

  // reset transformation member variables (eg. exporter) by rebuilding
  // and provide new maps for coupling adapter
  siatransform_ = std::make_shared<Coupling::Adapter::MatrixRowTransform>();
  coupsia_->setup_coupling(*ia_discret_, *discret_ptr_);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BeamInteraction::build_row_ele_to_bin_map()
{
  check_init();

  // delete old map
  ia_state_ptr_->get_row_ele_to_bin_map().clear();
  // loop over bins
  std::map<int, std::set<int>>::const_iterator biniter;
  for (biniter = ia_state_ptr_->get_bin_to_row_ele_map().begin();
      biniter != ia_state_ptr_->get_bin_to_row_ele_map().end(); ++biniter)
  {
    // loop over ele content of this bin
    std::set<int>::const_iterator eleiter;
    for (eleiter = biniter->second.begin(); eleiter != biniter->second.end(); ++eleiter)
    {
      int elegid = *eleiter;
      int bingid = biniter->first;
      // assign bins to elements
      ia_state_ptr_->get_row_ele_to_bin_map()[elegid].insert(bingid);
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BeamInteraction::update_maps()
{
  check_init();

  // todo: performance improvement by using the same exporter object every time
  // and not doing the safety checks in Linalg::Export.
  // todo: check if update is necessary (->SameAs())

  // beam displacement
  FourC::BeamInteraction::Utils::update_dof_map_of_vector(
      *ia_discret_, ia_state_ptr_->get_dis_np());

  // get current displacement state and export to interaction discretization dofmap
  FourC::BeamInteraction::Utils::update_dof_map_of_vector(
      *ia_discret_, ia_state_ptr_->get_dis_np(), global_state().get_dis_np());
  FourC::BeamInteraction::Utils::periodic_boundary_consistent_dis_vector(
      *ia_state_ptr_->get_dis_np(), *tim_int().get_data_sdyn_ptr()->get_periodic_bounding_box(),
      *ia_discret_);

  // update column vector
  ia_state_ptr_->get_dis_col_np() =
      std::make_shared<Core::LinAlg::Vector<double>>(*ia_discret_->dof_col_map());
  Core::LinAlg::export_to(*ia_state_ptr_->get_dis_np(), *ia_state_ptr_->get_dis_col_np());

  // update restart displacement vector
  if (ia_state_ptr_->get_restart_coupling_flag())
  {
    ia_state_ptr_->get_dis_restart_col() =
        std::make_shared<Core::LinAlg::Vector<double>>(*ia_discret_->dof_col_map());
    Core::LinAlg::export_to(
        *ia_state_ptr_->get_dis_restart(), *ia_state_ptr_->get_dis_restart_col());
  }

  // force
  ia_force_beaminteraction_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*ia_discret_->dof_row_map(), true);
  ia_state_ptr_->get_force_np() =
      std::make_shared<Core::LinAlg::FEVector<double>>(*ia_discret_->dof_row_map(), true);

  // stiff
  ia_state_ptr_->get_stiff() = std::make_shared<Core::LinAlg::SparseMatrix>(
      *ia_discret_->dof_row_map(), 81, true, true, Core::LinAlg::SparseMatrix::FE_MATRIX);

  FourC::BeamInteraction::Utils::setup_ele_type_map_extractor(*ia_discret_, *eletypeextractor_);
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BeamInteraction::transform_force()
{
  check_init();

  TEUCHOS_FUNC_TIME_MONITOR("Solid::ModelEvaluator::BeamInteraction::transform_force");

  // transform force vector to problem discret layout/distribution
  force_beaminteraction_ = coupsia_->master_to_slave(*ia_force_beaminteraction_);
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BeamInteraction::transform_stiff()
{
  check_init();

  TEUCHOS_FUNC_TIME_MONITOR("Solid::ModelEvaluator::BeamInteraction::transform_stiff");

  stiff_beaminteraction_->un_complete();
  // transform stiffness matrix to problem discret layout/distribution
  (*siatransform_)(*ia_state_ptr_->get_stiff(), 1.0,
      Coupling::Adapter::CouplingMasterConverter(*coupsia_), *stiff_beaminteraction_, false);
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BeamInteraction::transform_force_stiff()
{
  check_init();

  transform_force();
  transform_stiff();
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BeamInteraction::print_binning_info_to_screen() const
{
  std::vector<std::shared_ptr<Core::FE::Discretization>> discret_vec(1, ia_discret_);
  std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>> disnp_vec(1, nullptr);

  double bin_size_lower_bound =
      binstrategy_->compute_lower_bound_for_bin_size_as_max_edge_length_of_aabb_of_largest_ele(
          discret_vec, disnp_vec);
  Core::LinAlg::Matrix<3, 2> XAABB(Core::LinAlg::Initialization::zero);
  binstrategy_->compute_min_binning_domain_containing_all_elements_of_multiple_discrets(
      discret_vec, disnp_vec, XAABB, false);
  if (global_state().get_my_rank() == 0)
  {
    Core::IO::cout(Core::IO::verbose)
        << " \n---------------------------------------------------------- " << Core::IO::endl;
    Core::IO::cout(Core::IO::verbose) << " chosen/computed cutoff radius                      : "
                                      << binstrategy_->get_min_bin_size() << Core::IO::endl;
    Core::IO::cout(Core::IO::verbose)
        << " largest edge length of largest element xaabb       : " << bin_size_lower_bound
        << Core::IO::endl;
    Core::IO::cout(Core::IO::verbose)
        << "DOMAINBOUNDINGBOX containing all elements of input discretization:\n " << XAABB(0, 0)
        << " " << XAABB(1, 0) << " " << XAABB(2, 0) << " " << XAABB(0, 1) << " " << XAABB(1, 1)
        << " " << XAABB(2, 1) << Core::IO::endl;
    Core::IO::cout(Core::IO::verbose)
        << " ----------------------------------------------------------\n " << Core::IO::endl;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BeamInteraction::logo() const
{
  check_init();

  if (myrank_ == 0)
  {
    Core::IO::cout << "\n****************************************************************"
                   << Core::IO::endl;
    Core::IO::cout << "*                                                              *"
                   << Core::IO::endl;
    Core::IO::cout << "*          Welcome to the Beam Interaction Model Evaluator     *"
                   << Core::IO::endl;
    Core::IO::cout << "*                                                              *"
                   << Core::IO::endl;
    Core::IO::cout << "****************************************************************"
                   << Core::IO::endl;
    Core::IO::cout << "                                                                  "
                   << Core::IO::endl;
    Core::IO::cout << "                                                                  "
                   << Core::IO::endl;
    Core::IO::cout << "                      0=========================0                 "
                   << Core::IO::endl;
    Core::IO::cout << "                    //|   \\            /       /||                "
                   << Core::IO::endl;
    Core::IO::cout << "                   // |    \\ |       |/       //||                "
                   << Core::IO::endl;
    Core::IO::cout << "                  //  |  /  \\|       /       // ||                "
                   << Core::IO::endl;
    Core::IO::cout << "                 //   |  \\   \\   /  /|\\     //  ||                "
                   << Core::IO::endl;
    Core::IO::cout << "                //    |  /   |\\ /  / | \\   //   ||                "
                   << Core::IO::endl;
    Core::IO::cout << "               //     |  \\   | \\     |  \\ //  / ||                "
                   << Core::IO::endl;
    Core::IO::cout << "              //  \\  /|  /   |/      |   //  /  ||                "
                   << Core::IO::endl;
    Core::IO::cout << "              0=========================0 \\ /   ||                "
                   << Core::IO::endl;
    Core::IO::cout << "             ||    /\\ |____          |  || \\    ||                "
                   << Core::IO::endl;
    Core::IO::cout << "             ||   /  \\|    \\   ------   ||/ \\   ||                "
                   << Core::IO::endl;
    Core::IO::cout << "             ||  /    |                 ||      ||                "
                   << Core::IO::endl;
    Core::IO::cout << "             || /     0----------/------||------0-                "
                   << Core::IO::endl;
    Core::IO::cout << "             ||      /   /       \\      ||     //                 "
                   << Core::IO::endl;
    Core::IO::cout << "             ||     /___/  \\     /    / ||    //                  "
                   << Core::IO::endl;
    Core::IO::cout << "             ||    /        \\    \\   /  ||   //                   "
                   << Core::IO::endl;
    Core::IO::cout << "             ||   /  \\/\\/\\/  \\   /  /   ||  //                    "
                   << Core::IO::endl;
    Core::IO::cout << "             ||  /      /     \\  \\ /    || //                     "
                   << Core::IO::endl;
    Core::IO::cout << "             || /      /         /      ||//                      "
                   << Core::IO::endl;
    Core::IO::cout << "             ||/                       /||/                       "
                   << Core::IO::endl;
    Core::IO::cout << "              0=========================0                         "
                   << Core::IO::endl;
    Core::IO::cout << "                                                                     "
                   << Core::IO::endl;
    Core::IO::cout << "                                                                     "
                   << Core::IO::endl;
  }
}

FOUR_C_NAMESPACE_CLOSE
