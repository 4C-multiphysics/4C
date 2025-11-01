// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_submodel_evaluator_crosslinking.hpp"

#include "4C_beam3_base.hpp"
#include "4C_beaminteraction_calc_utils.hpp"
#include "4C_beaminteraction_crosslinker_handler.hpp"
#include "4C_beaminteraction_crosslinker_node.hpp"
#include "4C_beaminteraction_crosslinking_params.hpp"
#include "4C_beaminteraction_data.hpp"
#include "4C_beaminteraction_link.hpp"
#include "4C_beaminteraction_link_beam3_reissner_line2_pinjointed.hpp"
#include "4C_beaminteraction_link_beam3_reissner_line2_rigidjointed.hpp"
#include "4C_beaminteraction_str_model_evaluator_datastate.hpp"
#include "4C_binstrategy_meshfree_multibin.hpp"
#include "4C_fem_condition.hpp"
#include "4C_fem_condition_utils.hpp"
#include "4C_fem_geometry_intersection_math.hpp"
#include "4C_fem_geometry_periodic_boundingbox.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_beaminteraction.hpp"
#include "4C_io.hpp"
#include "4C_io_discretization_visualization_writer_nodes.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_multi_vector.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_mat_crosslinkermat.hpp"
#include "4C_structure_new_timint_basedataglobalstate.hpp"
#include "4C_structure_new_timint_basedataio.hpp"
#include "4C_structure_new_timint_basedataio_runtime_vtp_output.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_TimeMonitor.hpp>

#include <unordered_set>

FOUR_C_NAMESPACE_OPEN

/*-------------------------------------------------------------------------------*
 *-------------------------------------------------------------------------------*/
BeamInteraction::SubmodelEvaluator::Crosslinking::Crosslinking()
    : crosslinking_params_ptr_(nullptr),
      cl_exporter_(nullptr),
      beam_exporter_(nullptr),
      visualization_output_writer_ptr_(nullptr),
      linker_disnp_(nullptr),
      dis_at_last_redistr_(nullptr),
      half_interaction_distance_(0.0),
      cl_noderowmap_prior_redistr_(nullptr),
      cl_nodecolmap_prior_redistr_(nullptr),
      beam_elerowmap_prior_redistr_(nullptr),
      beam_elecolmap_prior_redistr_(nullptr)
{
  crosslinker_data_.clear();
  beam_data_.clear();
  doublebondcl_.clear();
}

/*-------------------------------------------------------------------------------*
 *-------------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::setup()
{
  check_init();

  // construct, init and setup data container for crosslinking
  crosslinking_params_ptr_ = std::make_shared<BeamInteraction::CrosslinkingParams>();
  crosslinking_params_ptr_->init(g_state());
  crosslinking_params_ptr_->setup();

  // set binding spot positions on filament elements according input file specifications
  set_filament_types();
  // this includes temporary change in ghosting
  BeamInteraction::Utils::set_filament_binding_spot_positions(
      discret_ptr(), *crosslinking_params_ptr_);

  // add crosslinker to bin discretization
  add_crosslinker_to_bin_discretization();

  // build runtime visualization output writer
  if (g_in_output().get_runtime_vtp_output_params() != nullptr) init_output_runtime_structure();

  // store old maps prior to redistribution
  cl_noderowmap_prior_redistr_ = std::make_shared<Core::LinAlg::Map>(*bin_discret().node_row_map());
  cl_nodecolmap_prior_redistr_ = std::make_shared<Core::LinAlg::Map>(*bin_discret().node_col_map());
  beam_elerowmap_prior_redistr_ =
      std::make_shared<Core::LinAlg::Map>(*ele_type_map_extractor().beam_map());
  beam_elecolmap_prior_redistr_ = std::make_shared<Core::LinAlg::Map>(*discret().element_col_map());

  // set flag
  issetup_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool BeamInteraction::SubmodelEvaluator::Crosslinking::post_partition_problem()
{
  check_init_setup();

  // init beam data container
  unsigned int numcolele = discret().num_my_col_elements();
  beam_data_.clear();
  beam_data_.resize(discret().num_my_col_elements());
  for (unsigned int i = 0; i < numcolele; ++i)
  {
    // beam element i for which data will be collected
    Discret::Elements::Beam3Base* beamele_i =
        dynamic_cast<Discret::Elements::Beam3Base*>(discret_ptr()->l_col_element(i));

    // go to next element in case the current one is not a beam element
    if (beamele_i == nullptr) continue;

    std::vector<double> eledisp;
    BeamInteraction::Utils::get_current_unshifted_element_dis(discret(), beamele_i,
        *beam_interaction_data_state_ptr()->get_dis_col_np(), periodic_bounding_box(), eledisp);

    beam_data_[i] = std::make_shared<BeamInteraction::Data::BeamData>();
    beam_data_[i]->set_id(beamele_i->id());

    // loop over all binding spots of current element
    // loop over binding spot types of current element
    for (auto const& iter : beamele_i->get_binding_spots())
    {
      // loop over all binding spots of current type j of current element
      int unsigned const numbbspot = beamele_i->get_number_of_binding_spots(iter.first);
      Core::LinAlg::Matrix<3, 1> pos(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<3, 3> triad(Core::LinAlg::Initialization::zero);
      for (int unsigned k = 0; k < numbbspot; ++k)
      {
        BeamInteraction::Utils::get_pos_and_triad_of_binding_spot(
            beamele_i, *periodic_bounding_box_ptr(), iter.first, k, pos, triad, eledisp);

        beam_data_[i]->set_b_spot_position(iter.first, k, pos);
        beam_data_[i]->set_b_spot_triad(iter.first, k, triad);
        beam_data_[i]->set_b_spot_status(iter.first, k, std::set<int>());
      }
    }
  }

  std::vector<std::shared_ptr<BeamInteraction::Data::CrosslinkerData>> newlinker;
  // map key is crosslinker gid to be able to uniquely address one entry over all procs
  std::map<int, NewDoubleBonds> mynewdbondcl;
  set_all_possible_initial_double_bonded_crosslinker(newlinker, mynewdbondcl);

  // set row map of newly created linker discretization
  bin_discret_ptr()->fill_complete(Core::FE::OptionsFillComplete::none());

  // init crosslinker data container
  crosslinker_data_.clear();
  unsigned int numrowcl = bin_discret().num_my_row_nodes();
  crosslinker_data_.resize(bin_discret().num_my_col_nodes());
  for (unsigned int i = 0; i < numrowcl; ++i)
  {
    Core::Nodes::Node* cl = bin_discret().l_row_node(i);
    crosslinker_data_[cl->lid()] = std::make_shared<BeamInteraction::Data::CrosslinkerData>();

    crosslinker_data_[cl->lid()]->set_id(cl->id());
  }

  // set initially set crosslinker
  for (auto& iter : newlinker)
  {
    int cl_lid = bin_discret().node_col_map()->lid(iter->get_id());
    crosslinker_data_[cl_lid]->set_b_spots(iter->get_b_spots());
    crosslinker_data_[cl_lid]->set_number_of_bonds(iter->get_number_of_bonds());
    crosslinker_data_[cl_lid]->set_position(iter->get_position());
  }

  // setup new double bonds and insert them in doublebondcl_
  create_new_double_bonded_crosslinker_element_pairs(mynewdbondcl);

  // store maps
  store_maps_prior_redistribution();
  update_and_export_crosslinker_data();
  update_and_export_beam_data(false);

  // local flag if one proc has new linker
  int loc_newlinks = static_cast<int>(newlinker.size() > 0);
  // global flag
  int g_newlinks = 0;
  g_newlinks = Core::Communication::max_all(loc_newlinks, bin_discret().get_comm());

  newlinker.clear();

  // we need to partition our problem again
  return static_cast<bool>(g_newlinks);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::post_setup()
{
  check_init_setup();

  if (not Global::Problem::instance()->restart())
  {
    // in case of initially set crosslinker
    if (crosslinking_params_ptr_->total_num_init_crosslinker() > 0)
      update_my_double_bonds_remote_id_list();

    // store displacement of restart step as displacement state of last redistribution
    dis_at_last_redistr_ =
        std::make_shared<Core::LinAlg::Vector<double>>(*bin_discret().dof_row_map(), true);
    for (int i = 0; i < bin_discret().num_my_row_nodes(); ++i)
    {
      CrossLinking::CrosslinkerNode* crosslinker_i =
          dynamic_cast<CrossLinking::CrosslinkerNode*>(bin_discret().l_row_node(i));

      // std::vector holding gids of dofs
      std::vector<int> dofnode = bin_discret().dof(crosslinker_i);

      // loop over all dofs
      for (unsigned int dim = 0; dim < 3; ++dim)
      {
        int doflid = dis_at_last_redistr_->get_map().lid(dofnode[dim]);
        (*dis_at_last_redistr_).get_values()[doflid] = crosslinker_i->x()[dim];
      }
    }
    // build up column linker information
    update_and_export_crosslinker_data();
    update_and_export_beam_data(false);
  }

  // store maps
  store_maps_prior_redistribution();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::init_submodel_dependencies(
    std::shared_ptr<Solid::ModelEvaluator::BeamInteraction::Map> const submodelmap)
{
  check_init_setup();
  // no active influence on other submodels
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::set_filament_types()
{
  check_init();

  std::vector<const Core::Conditions::Condition*> filament_conditions;
  discret().get_condition("BeamLineFilamentCondition", filament_conditions);
  auto filament_nodes_and_conditions = Core::Conditions::find_conditioned_node_ids_and_conditions(
      discret(), filament_conditions, Core::Conditions::LookFor::locally_owned_and_ghosted);

  // loop over all col nodes
  for (int coln = 0; coln < discret().num_my_col_nodes(); ++coln)
  {
    Core::Nodes::Node* currnode = discret().l_col_node(coln);
    // get filament number of current node ( requirement: node belongs to only one filament)
    if (!filament_nodes_and_conditions.contains(currnode->id())) continue;

    const Core::Conditions::Condition* cond =
        filament_nodes_and_conditions.find(currnode->id())->second;

    // get filament type
    Inpar::BeamInteraction::FilamentType filetype = Inpar::BeamInteraction::string_to_filament_type(
        (cond->parameters().get<std::string>("TYPE")));

    for (int i = 0; i < currnode->num_element(); ++i)
    {
      Discret::Elements::Beam3Base* beamele = dynamic_cast<Discret::Elements::Beam3Base*>(
          currnode->adjacent_elements()[i].user_element());

#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (beamele == nullptr)
        FOUR_C_THROW(" DESIGN LINE BEAM FILAMENT CONDITIONS only suitable for beam elements.");
#endif

      beamele->set_filament_type(filetype);
    }
  }
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::
    set_all_possible_initial_double_bonded_crosslinker(
        std::vector<std::shared_ptr<BeamInteraction::Data::CrosslinkerData>>& newlinker,
        std::map<int, NewDoubleBonds>& mynewdbondcl)
{
  check_init();

  // in case no initial crosslinker are supposed to be set, return, no repartitioning
  // of discretization required in this case
  if (crosslinking_params_ptr_->total_num_init_crosslinker() == 0) return;

  // get all possible bspot partners
  std::vector<BeamInteraction::Data::BspotLinkerData> my_bspot_linker;
  std::map<int, std::vector<BeamInteraction::Data::BspotLinkerData>> global_bspot_linker;

  // loop over all binding spots and find matching ones
  get_all_possible_bspot_links(my_bspot_linker);

  // create same list on procs
  communicate_initial_linker(my_bspot_linker, global_bspot_linker);

  // all procs make decisions for a valid link based the same rule
  std::vector<int> newlinkermatid;
  unambiguous_decisions_on_all_procs(newlinker, global_bspot_linker, newlinkermatid);

  // setupt new double bonded linker
  setup_my_initial_double_bonded_linker(newlinker, mynewdbondcl, newlinkermatid);
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::get_all_possible_bspot_links(
    std::vector<BeamInteraction::Data::BspotLinkerData>& my_bspot_linker)
{
  check_init();

  my_bspot_linker.clear();

  // loop over all row beam elements
  int unsigned const numbeams = ele_type_map_extractor_ptr()->beam_map()->num_my_elements();
  my_bspot_linker.reserve(numbeams);
  for (unsigned int rowbeam_i = 0; rowbeam_i < numbeams; ++rowbeam_i)
  {
    Discret::Elements::Beam3Base* beamele = dynamic_cast<Discret::Elements::Beam3Base*>(
        discret().g_element(ele_type_map_extractor_ptr()->beam_map()->gid(rowbeam_i)));

    BeamInteraction::Data::BeamData const* beamdata_i = beam_data_[beamele->lid()].get();

    // exclude current element/filament if filetype_none
    if (beamele->get_filament_type() == Inpar::BeamInteraction::filetype_none) continue;

    // loop over all binding spot types of current filament
    for (auto const& iter : beamdata_i->get_b_spot_status())
    {
      // loop over all binding spots of current binding spot type of current element
      unsigned int numbspots = beamele->get_number_of_binding_spots(iter.first);
      for (unsigned int locbspot_i = 0; locbspot_i < numbspots; ++locbspot_i)
      {
        // get bin of current bspot
        int const bingid = bin_strategy().convert_pos_to_gid(
            beamdata_i->get_b_spot_position(iter.first, locbspot_i));

        // get neighboring bins
        // note: interaction distance cl to beam needs to be smaller than the bin size
        std::vector<int> neighboring_binIds;
        neighboring_binIds.reserve(27);
        // do not check on existence here -> shifted to GetBinContent
        bin_strategy_ptr()->get_neighbor_and_own_bin_ids(bingid, neighboring_binIds);

        // get set of neighboring beam elements (i.e. elements that somehow touch nb bins)
        // we also need col elements (flag = false) here (in contrast to "normal" crosslinking)
        std::set<Core::Elements::Element*> neighboring_beams;
        std::vector<Core::Binstrategy::Utils::BinContentType> bc(
            1, Core::Binstrategy::Utils::BinContentType::Beam);
        bin_strategy_ptr()->get_bin_content(neighboring_beams, bc, neighboring_binIds, false);

        // in case there are no neighbors, go to next binding spot
        if (neighboring_beams.empty()) continue;

        // loop over all neighboring beam elements
        for (auto const& nb_ele_i : neighboring_beams)
        {
          Discret::Elements::Beam3Base* nb_beamele =
              dynamic_cast<Discret::Elements::Beam3Base*>(nb_ele_i);

          BeamInteraction::Data::BeamData const* nb_beamdata_i =
              beam_data_[nb_beamele->lid()].get();

          // exclude linking of touching elements
          if (BeamInteraction::Utils::do_beam_elements_share_nodes(beamele, nb_beamele)) continue;

          // exclude neighbor element/filament if filetype_none
          if (nb_beamele->get_filament_type() == Inpar::BeamInteraction::filetype_none) continue;

          // loop over binding spots of neighboring element
          for (unsigned int nb_locbspot_i = 0;
              nb_locbspot_i < nb_beamele->get_number_of_binding_spots(iter.first); ++nb_locbspot_i)
          {
            // loop over different linker types and check for feasibility of link
            std::vector<int> const& matcrosslinkerpertype =
                crosslinking_params_ptr_->mat_crosslinker_per_type();
            for (unsigned int type_i = 0; type_i < matcrosslinkerpertype.size(); ++type_i)
            {
              // check criteria for feasible bond
              std::shared_ptr<Mat::CrosslinkerMat> mat =
                  std::dynamic_pointer_cast<Mat::CrosslinkerMat>(
                      Mat::factory(matcrosslinkerpertype[type_i]));

              // linker type needs to match binding spot type
              if (mat->linker_type() != iter.first) continue;

              // distance
              // minimum and maximum distance at which a double-bond crosslink can be established
              double const linkdistmin = mat->linking_length() - mat->linking_length_tolerance();
              double const linkdistmax = mat->linking_length() + mat->linking_length_tolerance();

#ifdef FOUR_C_ENABLE_ASSERTIONS
              if (linkdistmax > bin_strategy().bin_size_lower_bound())
                FOUR_C_THROW(
                    "The allowed binding distance of your linker material is greater than the "
                    "lower bound for bin size, this can lead to missed binding events");
#endif

              if (BeamInteraction::Utils::is_distance_out_of_range(
                      beamdata_i->get_b_spot_position(iter.first, locbspot_i),
                      nb_beamdata_i->get_b_spot_position(iter.first, nb_locbspot_i), linkdistmin,
                      linkdistmax))
                continue;

              // orientation of centerline tangent vectors at binding spots:
              // a crosslink (double-bonded crosslinker) will only be established if the
              // enclosed angle is in the specified range
              double const linkanglemin = mat->linking_angle() - mat->linking_angle_tolerance();
              double const linkanglemax = mat->linking_angle() + mat->linking_angle_tolerance();

              // get tangent of binding spot on beamele
              Core::LinAlg::Matrix<3, 1> bindingspot_beam_tangent(
                  Core::LinAlg::Initialization::zero);
              for (unsigned int idim = 0; idim < 3; ++idim)
                bindingspot_beam_tangent(idim) =
                    beamdata_i->get_b_spot_triad(iter.first, locbspot_i)(idim, 0);

              // get tangent of binding spot on nb_beamele
              Core::LinAlg::Matrix<3, 1> nb_bindingspot_beam_tangent(
                  Core::LinAlg::Initialization::zero);
              for (unsigned int idim = 0; idim < 3; ++idim)
                nb_bindingspot_beam_tangent(idim) =
                    nb_beamdata_i->get_b_spot_triad(iter.first, nb_locbspot_i)(idim, 0);

              if (BeamInteraction::Utils::is_enclosed_angle_out_of_range(bindingspot_beam_tangent,
                      nb_bindingspot_beam_tangent, linkanglemin, linkanglemax))
                continue;

              // if all criteria were met for a certain linker, add this bond to linker specific
              // bond list
              BeamInteraction::Data::BspotLinkerData bspotlinkerdata;
              bspotlinkerdata.set_ele_gid1(beamele->id());
              bspotlinkerdata.set_loc_bspot_id1(locbspot_i);
              bspotlinkerdata.set_ele_gid2(nb_beamele->id());
              bspotlinkerdata.set_loc_bspot_id2(nb_locbspot_i);
              bspotlinkerdata.set_type(static_cast<int>(iter.first));
              bspotlinkerdata.set_mat_id(matcrosslinkerpertype[type_i]);

              // append to all my linker
              my_bspot_linker.push_back(bspotlinkerdata);

              // go to next binding spot, if current one will possibly bind
              break;
            }
          }
        }
      }
    }
  }
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::communicate_initial_linker(
    std::vector<BeamInteraction::Data::BspotLinkerData> const& my_bspot_linker,
    std::map<int, std::vector<BeamInteraction::Data::BspotLinkerData>>& global_bspot_linker)
{
  check_init();

  global_bspot_linker.clear();

  // gather all data over all procs
  MPI_Comm com = discret().get_comm();
  const int numproc = Core::Communication::num_mpi_ranks(com);
  int numpairs = static_cast<int>(my_bspot_linker.size());
  std::vector<int> elegid_1, elegid_2, locbspot_1, locbspot_2, type, mat;
  for (int iproc = 0; iproc < numproc; ++iproc)
  {
    numpairs = my_bspot_linker.size();
    std::vector<int> elegid_1_i(numpairs, 0), elegid_2_i(numpairs, 0), locbspot_1_i(numpairs, 0),
        locbspot_2_i(numpairs, 0), type_i(numpairs, 0), mat_i(numpairs, 0);
    if (iproc == g_state().get_my_rank())
    {
      for (unsigned int i = 0; i < my_bspot_linker.size(); ++i)
      {
        elegid_1_i[i] = my_bspot_linker[i].get_ele_gid1();
        elegid_2_i[i] = my_bspot_linker[i].get_ele_gid2();
        locbspot_1_i[i] = my_bspot_linker[i].get_loc_bspot_id1();
        locbspot_2_i[i] = my_bspot_linker[i].get_loc_bspot_id2();
        type_i[i] = my_bspot_linker[i].get_type();
        mat_i[i] = my_bspot_linker[i].get_mat_id();
      }
    }

    // first: proc i tells all procs how many pairs it has
    Core::Communication::broadcast(&numpairs, 1, iproc, com);
    // second: proc i tells all procs which pairs it has
    elegid_1_i.resize(numpairs);
    elegid_2_i.resize(numpairs);
    locbspot_1_i.resize(numpairs);
    locbspot_2_i.resize(numpairs);
    type_i.resize(numpairs);
    mat_i.resize(numpairs);

    Core::Communication::broadcast(elegid_1_i.data(), numpairs, iproc, com);
    Core::Communication::broadcast(elegid_2_i.data(), numpairs, iproc, com);
    Core::Communication::broadcast(locbspot_1_i.data(), numpairs, iproc, com);
    Core::Communication::broadcast(locbspot_2_i.data(), numpairs, iproc, com);
    Core::Communication::broadcast(type_i.data(), numpairs, iproc, com);
    Core::Communication::broadcast(mat_i.data(), numpairs, iproc, com);

    elegid_1.insert(elegid_1.end(), elegid_1_i.begin(), elegid_1_i.end());
    elegid_2.insert(elegid_2.end(), elegid_2_i.begin(), elegid_2_i.end());
    locbspot_1.insert(locbspot_1.end(), locbspot_1_i.begin(), locbspot_1_i.end());
    locbspot_2.insert(locbspot_2.end(), locbspot_2_i.begin(), locbspot_2_i.end());
    type.insert(type.end(), type_i.begin(), type_i.end());
    mat.insert(mat.end(), mat_i.begin(), mat_i.end());
  }

  // rebuild bspot_linker_data
  int numglobalpairs = elegid_1.size();
  std::map<int, std::map<long long, BeamInteraction::Data::BspotLinkerData>> sort_data;
  long long bspotgid = -1;
  for (int i = 0; i < numglobalpairs; ++i)
  {
    BeamInteraction::Data::BspotLinkerData bspotlinkerdata;
    bspotlinkerdata.set_ele_gid1(elegid_1[i]);
    bspotlinkerdata.set_loc_bspot_id1(locbspot_1[i]);
    bspotlinkerdata.set_ele_gid2(elegid_2[i]);
    bspotlinkerdata.set_loc_bspot_id2(locbspot_2[i]);
    bspotlinkerdata.set_type(type[i]);
    bspotlinkerdata.set_mat_id(mat[i]);

    bspotgid = BeamInteraction::Utils::cantor_pairing(std::make_pair(elegid_2[i], locbspot_2[i]));

    sort_data[elegid_1[i]][bspotgid] = bspotlinkerdata;
  }

  for (auto const& iter_sort_i : sort_data)
    for (auto const& iter_sort_j : iter_sort_i.second)
      global_bspot_linker[iter_sort_i.first].push_back(iter_sort_j.second);
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::unambiguous_decisions_on_all_procs(
    std::vector<std::shared_ptr<BeamInteraction::Data::CrosslinkerData>>& newlinker,
    std::map<int, std::vector<BeamInteraction::Data::BspotLinkerData>> const& global_bspot_linker,
    std::vector<int>& newlinkermatid)
{
  // initialize a box within linker are spawned
  std::vector<bool> dummy(3, false);
  Core::Geo::MeshFree::BoundingBox linker_init_box;
  linker_init_box.init(
      crosslinking_params_ptr_->linker_initialization_box(), dummy);  // no setup() call needed here

  // loop over bspotpairs and make decision
  newlinker.reserve(global_bspot_linker.size());
  std::map<long long, int> bondsperbindingspot;
  std::set<std::pair<long long, long long>, BeamInteraction::Utils::StdPairComparatorOrderCounts>
      doublebonds;
  std::map<int, int> numpertype;
  Core::LinAlg::Matrix<3, 1> clpos(Core::LinAlg::Initialization::zero);
  for (auto const& iter_sort : global_bspot_linker)
  {
    for (auto const& iter : iter_sort.second)
    {
      long long bspotgid = BeamInteraction::Utils::cantor_pairing(
          std::make_pair(iter.get_ele_gid1(), iter.get_loc_bspot_id1()));
      long long nb_bspotgid = BeamInteraction::Utils::cantor_pairing(
          std::make_pair(iter.get_ele_gid2(), iter.get_loc_bspot_id2()));

      Inpar::BeamInteraction::CrosslinkerType linkertype =
          static_cast<Inpar::BeamInteraction::CrosslinkerType>(iter.get_type());

      // check if binding spot has reached its maximum number of bonds
      if (bondsperbindingspot.find(bspotgid) != bondsperbindingspot.end() and
          bondsperbindingspot[bspotgid] >=
              crosslinking_params_ptr_->max_number_of_bonds_per_filament_bspot(linkertype))
        continue;

      // check if maximal number of crosslinker for current type is already exceeded
      if ((numpertype[iter.get_mat_id()] + 1) >
          crosslinking_params_ptr_->num_init_crosslinker_per_crosslinker_mat_id(iter.get_mat_id()))
        continue;

      // to ensure that double bonds are recognized as such independent of the order of the
      // bspotgids
      std::pair<long long, long long> currdoublebond = (bspotgid < nb_bspotgid)
                                                           ? std::make_pair(bspotgid, nb_bspotgid)
                                                           : std::make_pair(nb_bspotgid, bspotgid);
      // to check:
      // i)  check if binding spot has reached its maximum number of bonds
      // ii) check if identical bond already exists
      if ((bondsperbindingspot.find(nb_bspotgid) != bondsperbindingspot.end() and
              bondsperbindingspot[nb_bspotgid] >=
                  crosslinking_params_ptr_->max_number_of_bonds_per_filament_bspot(linkertype)) or
          doublebonds.find(currdoublebond) != doublebonds.end())
        continue;

      // update variables
      doublebonds.insert(currdoublebond);
      ++bondsperbindingspot[nb_bspotgid];
      ++bondsperbindingspot[bspotgid];
      ++numpertype[iter.get_mat_id()];

      // check ownership
      if (discret().element_row_map()->lid(iter.get_ele_gid1()) == -1) continue;

      // store data of new crosslinker
      std::shared_ptr<BeamInteraction::Data::CrosslinkerData> cldata =
          std::make_shared<BeamInteraction::Data::CrosslinkerData>();

      // set positions
      clpos.clear();
      set_position_of_double_bonded_crosslinker_pb_cconsistent(clpos,
          beam_data_[discret_ptr()->g_element(iter.get_ele_gid1())->lid()]->get_b_spot_position(
              linkertype, iter.get_loc_bspot_id1()),
          beam_data_[discret_ptr()->g_element(iter.get_ele_gid2())->lid()]->get_b_spot_position(
              linkertype, iter.get_loc_bspot_id2()));

      // check if linker is within linker initialization box
      if (not linker_init_box.within(clpos, dummy)) continue;

      // set current binding spot status of crosslinker
      cldata->set_position(clpos);
      cldata->set_bspot(0, std::make_pair(iter.get_ele_gid1(), iter.get_loc_bspot_id1()));
      cldata->set_bspot(1, std::make_pair(iter.get_ele_gid2(), iter.get_loc_bspot_id2()));
      // set number of bonds
      cldata->set_number_of_bonds(2);

      newlinker.push_back(cldata);
      newlinkermatid.push_back(iter.get_mat_id());
    }
  }
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::setup_my_initial_double_bonded_linker(
    std::vector<std::shared_ptr<BeamInteraction::Data::CrosslinkerData>>& newlinker,
    std::map<int, NewDoubleBonds>& mynewdbondcl, std::vector<int> const& newlinkermatid)
{
  // determine unique gids on each proc (ascending over all procs)
  // gather numbers of new linker on each proc
  MPI_Comm com = discret().get_comm();
  std::vector<int> nummynewlinks(1);
  nummynewlinks[0] = static_cast<int>(newlinker.size());
  // initialize std::vector for communication
  std::vector<int> numnewlinks(Core::Communication::num_mpi_ranks(com), 0);
  // communicate
  Core::Communication::gather_all(
      nummynewlinks.data(), numnewlinks.data(), nummynewlinks.size(), com);
  Core::Communication::barrier(com);

  // calculate starting index on myrank
  int mystartgid = 0;
  for (int i = 0; i < g_state().get_my_rank(); ++i) mystartgid += numnewlinks[i];

  // loop over new linker on myrank
  int gid = bin_discret_ptr()->node_row_map()->max_all_gid() + 1 + mystartgid;
  std::vector<double> X(3);
  for (unsigned int i = 0; i < newlinker.size(); ++i)
  {
    for (unsigned int dim = 0; dim < 3; ++dim) X[dim] = newlinker[i]->get_position()(dim);

    std::shared_ptr<CrossLinking::CrosslinkerNode> newcrosslinker =
        std::make_shared<CrossLinking::CrosslinkerNode>(gid, X, g_state().get_my_rank());
    newcrosslinker->set_material(std::dynamic_pointer_cast<Mat::CrosslinkerMat>(
        Mat::factory(newlinkermatid[i])));  // HACK HACK HACK
    bin_discret_ptr()->add_node(newcrosslinker->x(), newcrosslinker->id(), newcrosslinker);

    NewDoubleBonds dbondcl;
    std::vector<std::pair<int, int>> bspots = newlinker[i]->get_b_spots();
    dbondcl.id = gid;
    dbondcl.eleids.push_back(bspots[0]);
    dbondcl.eleids.push_back(bspots[1]);

    int colelelid = discret_ptr()->element_col_map()->lid(bspots[0].first);
    int nb_colelelid = discret_ptr()->element_col_map()->lid(bspots[1].first);
    dbondcl.bspotposs.push_back(beam_data_[colelelid]->get_b_spot_position(
        newcrosslinker->get_material()->linker_type(), bspots[0].second));
    dbondcl.bspotposs.push_back(beam_data_[nb_colelelid]->get_b_spot_position(
        newcrosslinker->get_material()->linker_type(), bspots[1].second));
    dbondcl.bspottriads.push_back(beam_data_[colelelid]->get_b_spot_triad(
        newcrosslinker->get_material()->linker_type(), bspots[0].second));
    dbondcl.bspottriads.push_back(beam_data_[nb_colelelid]->get_b_spot_triad(
        newcrosslinker->get_material()->linker_type(), bspots[1].second));

    mynewdbondcl[dbondcl.id] = dbondcl;

    // set correct states for linker
    newlinker[i]->set_id(gid);

    std::vector<double> newpos(3, 0.0);
    for (int dim = 0; dim < 3; ++dim) newpos[dim] = newlinker[i]->get_position()(dim);
    newcrosslinker->set_pos(newpos);

    beam_data_[discret_ptr()->element_col_map()->lid(newlinker[i]->get_b_spots()[0].first)]
        ->add_bond_to_binding_spot(newcrosslinker->get_material()->linker_type(),
            newlinker[i]->get_b_spots()[0].second, gid);

    if (discret().have_global_element(newlinker[i]->get_b_spots()[1].first))
      beam_data_[discret_ptr()->element_col_map()->lid(newlinker[i]->get_b_spots()[1].first)]
          ->add_bond_to_binding_spot(newcrosslinker->get_material()->linker_type(),
              newlinker[i]->get_b_spots()[1].second, gid);

    // update gid
    ++gid;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::add_crosslinker_to_bin_discretization()
{
  check_init();

  // initialize a box within linker are spawned
  std::vector<bool> dummy(3, false);
  Core::Geo::MeshFree::BoundingBox linker_init_box;
  linker_init_box.init(
      crosslinking_params_ptr_->linker_initialization_box(), dummy);  // no setup() call needed here

  // loop over all linker types that should be added to simulation volume
  // (only proc 0 is doing this (as the number of crosslinker is manageable)
  std::vector<int> const& numcrosslinkerpertype =
      crosslinking_params_ptr_->num_crosslinker_per_type();
  std::vector<int> const& matcrosslinkerpertype =
      crosslinking_params_ptr_->mat_crosslinker_per_type();
  int gid = 0;
  if (g_state().get_my_rank() == 0)
  {
    for (int cltype_i = 0; cltype_i < static_cast<int>(numcrosslinkerpertype.size()); ++cltype_i)
    {
      for (int cltype_i_cl_j = 0; cltype_i_cl_j < numcrosslinkerpertype[cltype_i]; ++cltype_i_cl_j)
      {
        // random reference position of crosslinker in bounding box
        Core::LinAlg::Matrix<3, 1> Xmat;
        linker_init_box.random_pos_within(Xmat, Global::Problem::instance()->random());

        std::vector<double> X(3);
        for (int dim = 0; dim < 3; ++dim) X[dim] = Xmat(dim);

        // construct node, init data container, set material and add to bin discret
        std::shared_ptr<CrossLinking::CrosslinkerNode> newcrosslinker =
            std::make_shared<CrossLinking::CrosslinkerNode>(gid++, X, g_state().get_my_rank());
        newcrosslinker->set_material(matcrosslinkerpertype[cltype_i]);
        bin_discret_ptr()->add_node(newcrosslinker->x(), newcrosslinker->id(), newcrosslinker);
      }
    }
  }

  // set row map of newly created linker discretization
  bin_discret_ptr()->fill_complete(Core::FE::OptionsFillComplete::none());
}

/*-------------------------------------------------------------------------------*
 *-------------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::reset()
{
  check_init_setup();

  // reset crosslinker pairs
  for (auto const& iter : doublebondcl_)
  {
    std::shared_ptr<BeamInteraction::BeamLink> elepairptr = iter.second;

#ifdef FOUR_C_ENABLE_ASSERTIONS

    CrossLinking::CrosslinkerNode* cl_i =
        dynamic_cast<CrossLinking::CrosslinkerNode*>(bin_discret_ptr()->g_node(elepairptr->id()));
    // safety check
    BeamInteraction::Data::CrosslinkerData const* cldata_i = crosslinker_data_[cl_i->lid()].get();

    if (cldata_i->get_number_of_bonds() != 2)
      FOUR_C_THROW("Cl with gid {} Owner {} on myrank {} and numbonds {}", elepairptr->id(),
          cl_i->owner(), g_state_ptr()->get_my_rank(), cldata_i->get_number_of_bonds());
#endif

    // init positions and triads
    std::vector<Core::LinAlg::Matrix<3, 1>> pos(2);
    std::vector<Core::LinAlg::Matrix<3, 3>> triad(2);

    for (int i = 0; i < 2; ++i)
    {
      int elegid = elepairptr->get_ele_gid(i);

      // safety check
      if (discret_ptr()->element_col_map()->lid(elegid) < 0)
      {
        elepairptr->print(std::cout);
        FOUR_C_THROW("reset(): elegid {} not there on proc {} ", elegid, g_state().get_my_rank());
      }

      int locbspotnum = elepairptr->get_loc_b_spot_num(i);
      Core::Elements::Element* ele = discret_ptr()->g_element(elegid);

      BeamInteraction::Utils::get_pos_and_triad_of_binding_spot(discret(), ele,
          *beam_interaction_data_state_ptr()->get_dis_col_np(), *periodic_bounding_box_ptr(),
          elepairptr->get_linker_type(), locbspotnum, pos[i], triad[i]);
    }

    // unshift one of the positions if both are separated by a periodic boundary
    // condition, i.e. have been shifted before
    periodic_bounding_box_ptr()->un_shift_3d(pos[1], pos[0]);

    // safety check until code is better tested for potential problems with periodic boundary
    // conditions
    // **************************** DEBUG ****************************************
    Core::LinAlg::Matrix<3, 1> dist(Core::LinAlg::Initialization::zero);
    dist.update(1.0, pos[0], -1.0, pos[1]);
    for (unsigned int i = 0; i < 3; ++i)
    {
      if (std::abs(dist(i)) > 0.5 * periodic_bounding_box_ptr()->edge_length(i))
      {
        FOUR_C_THROW(
            "You are trying to set the binding spot positions of this crosslinker "
            "in at least one direction\n at a distance larger than {}, which is "
            " half of the period length in the respective direction",
            0.5 * periodic_bounding_box_ptr()->edge_length(i));
      }
    }
    // ********************** END DEBUG ****************************************

    // finally reset state
    elepairptr->reset_state(pos, triad);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool BeamInteraction::SubmodelEvaluator::Crosslinking::evaluate_force()
{
  check_init_setup();

  // force and moment exerted on the two connection sites due to the mechanical connection
  std::vector<Core::LinAlg::SerialDenseVector> bspotforce(2, Core::LinAlg::SerialDenseVector(6));

  // resulting discrete element force vectors of the two parent elements
  std::vector<Core::LinAlg::SerialDenseVector> eleforce(2);

  std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>> dummystiff;

  // element gids of interacting elements
  std::vector<int> elegids(2);

  for (auto const& iter : doublebondcl_)
  {
    std::shared_ptr<BeamInteraction::BeamLink> elepairptr = iter.second;

    for (int i = 0; i < 2; ++i)
    {
      elegids[i] = elepairptr->get_ele_gid(i);
      bspotforce[i].putScalar(0.0);
    }

    // evaluate beam linkage object to get forces and moments on binding spots
    elepairptr->evaluate_force(bspotforce[0], bspotforce[1]);

    // apply forces on binding spots to parent elements
    // and get their discrete element force vectors
    BeamInteraction::Utils::apply_binding_spot_force_to_parent_elements<
        Discret::Elements::Beam3Base, Discret::Elements::Beam3Base>(discret(),
        *periodic_bounding_box_ptr(), *beam_interaction_data_state_ptr()->get_dis_col_np(),
        *elepairptr, bspotforce, eleforce);

    // assemble the contributions into force vector class variable
    // f_crosslink_np_ptr_, i.e. in the DOFs of the connected nodes
    BeamInteraction::Utils::fe_assemble_ele_force_stiff_into_system_vector_matrix(discret(),
        elegids, eleforce, dummystiff, beam_interaction_data_state_ptr()->get_force_np(), nullptr);
  }

  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool BeamInteraction::SubmodelEvaluator::Crosslinking::evaluate_stiff()
{
  check_init_setup();

  /* linearizations, i.e. stiffness contributions due to forces on the two
   * connection sites due to the mechanical connection */
  std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>> bspotstiff(
      2, std::vector<Core::LinAlg::SerialDenseMatrix>(2, Core::LinAlg::SerialDenseMatrix(6, 6)));

  // linearizations, i.e. discrete stiffness contributions to the two parent elements
  // we can't handle this separately for both elements because there are entries which
  // couple the two element stiffness blocks
  std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>> elestiff(
      2, std::vector<Core::LinAlg::SerialDenseMatrix>(2));

  std::vector<Core::LinAlg::SerialDenseVector> dummyforce;

  // element gids of interacting elements
  std::vector<int> elegids(2);

  for (auto const& iter : doublebondcl_)
  {
    std::shared_ptr<BeamInteraction::BeamLink> elepairptr = iter.second;

    for (int i = 0; i < 2; ++i)
    {
      elegids[i] = elepairptr->get_ele_gid(i);

      for (int j = 0; j < 2; ++j) bspotstiff[i][j].putScalar(0.0);
    }

    // evaluate beam linkage object to get linearizations of forces and moments on binding spots
    elepairptr->evaluate_stiff(
        bspotstiff[0][0], bspotstiff[0][1], bspotstiff[1][0], bspotstiff[1][1]);

    // apply linearizations to parent elements and get their discrete element stiffness matrices
    BeamInteraction::Utils::apply_binding_spot_stiff_to_parent_elements<
        Discret::Elements::Beam3Base, Discret::Elements::Beam3Base>(discret(),
        *periodic_bounding_box_ptr(), *beam_interaction_data_state_ptr()->get_dis_col_np(),
        *elepairptr, bspotstiff, elestiff);

    // assemble the contributions into stiffness matrix class variable
    // stiff_crosslink_ptr_, i.e. in the DOFs of the connected nodes
    BeamInteraction::Utils::fe_assemble_ele_force_stiff_into_system_vector_matrix(discret(),
        elegids, dummyforce, elestiff, nullptr, beam_interaction_data_state_ptr()->get_stiff());
  }

  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool BeamInteraction::SubmodelEvaluator::Crosslinking::evaluate_force_stiff()
{
  check_init_setup();

  // force and moment exerted on the two connection sites due to the mechanical connection
  std::vector<Core::LinAlg::SerialDenseVector> bspotforce(2, Core::LinAlg::SerialDenseVector(6));

  /* linearizations, i.e. stiffness contributions due to forces on the two
   * connection sites due to the mechanical connection */
  std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>> bspotstiff(
      2, std::vector<Core::LinAlg::SerialDenseMatrix>(2, Core::LinAlg::SerialDenseMatrix(6, 6)));

  // resulting discrete element force vectors of the two parent elements
  std::vector<Core::LinAlg::SerialDenseVector> eleforce(2);

  // linearizations, i.e. discrete stiffness contributions to the two parent elements
  // we can't handle this separately for both elements because there are entries which
  // couple the two element stiffness blocks
  std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>> elestiff(
      2, std::vector<Core::LinAlg::SerialDenseMatrix>(2));

  // element gids of interacting elements
  std::vector<int> elegids(2);

  for (auto const& iter : doublebondcl_)
  {
    std::shared_ptr<BeamInteraction::BeamLink> elepairptr = iter.second;
    for (int i = 0; i < 2; ++i)
    {
      elegids[i] = elepairptr->get_ele_gid(i);
      bspotforce[i].putScalar(0.0);

      for (int j = 0; j < 2; ++j) bspotstiff[i][j].putScalar(0.0);
    }

    // evaluate beam linkage object to get forces and moments on binding spots
    elepairptr->evaluate_force_stiff(bspotforce[0], bspotforce[1], bspotstiff[0][0],
        bspotstiff[0][1], bspotstiff[1][0], bspotstiff[1][1]);

    // apply forces on binding spots and corresponding linearizations to parent elements
    // and get their discrete element force vectors and stiffness matrices
    BeamInteraction::Utils::apply_binding_spot_force_stiff_to_parent_elements<
        Discret::Elements::Beam3Base, Discret::Elements::Beam3Base>(discret(),
        *periodic_bounding_box_ptr(), *beam_interaction_data_state_ptr()->get_dis_col_np(),
        *elepairptr, bspotforce, bspotstiff, eleforce, elestiff);

    // assemble the contributions into force and stiffness class variables
    // f_crosslink_np_ptr_, stiff_crosslink_ptr_, i.e. in the DOFs of the connected nodes
    BeamInteraction::Utils::fe_assemble_ele_force_stiff_into_system_vector_matrix(discret(),
        elegids, eleforce, elestiff, beam_interaction_data_state_ptr()->get_force_np(),
        beam_interaction_data_state_ptr()->get_stiff());
  }

  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::update_step_state(const double& timefac_n)
{
  check_init_setup();

  // crosslinker diffusion: - according to browninan dyn for free cl
  //                        - according to beams for single and double bonded
  diffuse_crosslinker();
}
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool BeamInteraction::SubmodelEvaluator::Crosslinking::pre_update_step_element(bool beam_redist)
{
  check_init_setup();

#ifdef FOUR_C_ENABLE_ASSERTIONS
  // safety check
  if (not dis_at_last_redistr_->get_map().same_as(*bin_discret().dof_row_map()))
    FOUR_C_THROW(
        "current linker dof map and map of disp vector after last redistribution are\n "
        "are not the same. Something went wrong");
#endif

  linker_disnp_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*bin_discret().dof_row_map(), true);
  Core::LinAlg::Vector<double> dis_increment(*bin_discret().dof_row_map(), true);

  Core::LinAlg::Matrix<3, 1> d;
  Core::LinAlg::Matrix<3, 1> ref;
  int doflid[3];

  for (int i = 0; i < bin_discret().num_my_row_nodes(); ++i)
  {
    d.clear();
    ref.clear();

    // get a pointer at i-th row node
    Core::Nodes::Node* node = bin_discret().l_row_node(i);

    // get GIDs of this node's degrees of freedom
    std::vector<int> dofnode = bin_discret().dof(node);

    for (int dim = 0; dim < 3; ++dim)
    {
      doflid[dim] = dis_at_last_redistr_->get_map().lid(dofnode[dim]);
      d(dim) = (*dis_at_last_redistr_).local_values_as_span()[doflid[dim]];
      (*linker_disnp_).get_values()[doflid[dim]] = ref(dim) = node->x()[dim];
    }
    // unshift
    periodic_bounding_box().un_shift_3d(d, ref);

    for (int dim = 0; dim < 3; ++dim) (dis_increment).get_values()[doflid[dim]] = d(dim) - ref(dim);
  }

  // get maximal displacement increment since last redistribution over all procs
  std::array<double, 2> extrema = {0.0, 0.0};
  dis_increment.min_value(&extrema[0]);
  dis_increment.max_value(&extrema[1]);
  const double gmaxdisincr = std::max(-extrema[0], extrema[1]);

  // some screen output
  if (g_state().get_my_rank() == 0)
    Core::IO::cout(Core::IO::debug) << " max linker movement " << gmaxdisincr << Core::IO::endl;

  bool linker_redist =
      ((half_interaction_distance_ + gmaxdisincr) > (0.5 * bin_strategy().bin_size_lower_bound()));

  // store old maps prior to potential redistribution
  // this needs to be stored even no redistribution takes place later one
  store_maps_prior_redistribution();

  if (linker_redist or beam_redist)
  {
    // current displacement state gets new reference state
    dis_at_last_redistr_ = std::make_shared<Core::LinAlg::Vector<double>>(*linker_disnp_);
    // transfer crosslinker to new bins
    std::shared_ptr<std::list<int>> lostcl = beam_crosslinker_handler_ptr()->transfer_linker(true);
    if (not lostcl->empty())
      FOUR_C_THROW("Crosslinker got lost during transfer, something went wrong");
  }

  return linker_redist;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::update_step_element(
    bool repartition_was_done)
{
  check_init_setup();

  if (repartition_was_done)
  {
    // adapt map of vector to map after redistribution
    BeamInteraction::Utils::update_dof_map_of_vector(*bin_discret_ptr(), dis_at_last_redistr_);

    // update double bonded linker
    update_my_double_bonds_after_redistribution();
  }

  // gather data for all column crosslinker and column beams initially
  // this needs to be done every time since e.g. beam positions and triads
  // change every time
  update_and_export_crosslinker_data();
  update_and_export_beam_data();

  // manage binding and unbinding events of crosslinker
  bind_and_unbind_crosslinker();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::post_update_step_element()
{
  // empty
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::map<Solid::EnergyType, double> BeamInteraction::SubmodelEvaluator::Crosslinking::get_energy()
    const
{
  check_init_setup();
  std::map<Solid::EnergyType, double> cl_energies;

  for (auto db_iter : doublebondcl_)
  {
    cl_energies[Solid::beam_to_beam_link_internal_energy] += db_iter.second->get_internal_energy();
    cl_energies[Solid::beam_to_beam_link_kinetic_energy] += db_iter.second->get_kinetic_energy();
  }

  return cl_energies;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::output_step_state(
    Core::IO::DiscretizationWriter& iowriter) const
{
  check_init_setup();
  // not used, we are writing output during runtime
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::runtime_output_step_state() const
{
  check_init_setup();

  TEUCHOS_FUNC_TIME_MONITOR(
      "BeamInteraction::SubmodelEvaluator::Crosslinking::"
      "runtime_output_step_state");

  if (visualization_output_writer_ptr_ != nullptr) write_output_runtime_structure();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::init_output_runtime_structure()
{
  check_init();

  visualization_output_writer_ptr_ =
      std::make_shared<Core::IO::DiscretizationVisualizationWriterNodes>(bin_discret_ptr(),
          Core::IO::visualization_parameters_factory(
              Global::Problem::instance()->io_params().sublist("RUNTIME VTK OUTPUT"),
              *Global::Problem::instance()->output_control_file(), g_state().get_time_n()));
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::write_output_runtime_structure() const
{
  check_init_setup();

  // ************** BEGIN RUNTIME VTP OUTPUT *** OPTION 1: DISCRETIZATION *********
  // this section compiles and seems to do the job correctly :-)

  // initialize the writer object
  visualization_output_writer_ptr_->set_geometry_from_discretization();

  // append all desired node and dof output data to the writer object's storage
  Core::FE::Discretization const& bindis = bin_discret();
  // node
  std::shared_ptr<Core::LinAlg::Vector<double>> numbond =
      Core::LinAlg::create_vector(*bindis.node_row_map(), true);
  std::shared_ptr<Core::LinAlg::Vector<double>> owner =
      Core::LinAlg::create_vector(*bindis.node_row_map(), true);

  // dof
  std::shared_ptr<Core::LinAlg::Vector<double>> dis =
      Core::LinAlg::create_vector(*bindis.dof_row_map(), true);
  std::shared_ptr<Core::LinAlg::Vector<double>> orientation =
      Core::LinAlg::create_vector(*bindis.dof_row_map(), true);
  std::shared_ptr<Core::LinAlg::Vector<double>> force =
      Core::LinAlg::create_vector(*bindis.dof_row_map(), true);

  fill_state_data_vectors_for_output(*dis, *orientation, *numbond, *owner, *force);

  // append displacement vector if desired
  // append displacement if desired
  //   if ( global_in_output().get_runtime_vtp_output_params()->output_displacement_state() )
  //     visualization_output_writer_ptr_-->append_dof_based_result_data_vector( dis, 3,
  //     "displacement"
  //     );

  // append owner if desired
  if (g_in_output().get_runtime_vtp_output_params()->output_owner())
    visualization_output_writer_ptr_->append_node_based_result_data_vector(*owner, 1, "owner");

  // append orientation vector if desired
  if (g_in_output().get_runtime_vtp_output_params()->output_orientation_and_length())
    visualization_output_writer_ptr_->append_dof_based_result_data_vector(
        *orientation, 3, "orientation");

  // append number of bonds if desired
  if (g_in_output().get_runtime_vtp_output_params()->output_number_of_bonds())
    visualization_output_writer_ptr_->append_node_based_result_data_vector(
        *numbond, 1, "numberofbonds");

  // append number of bonds if desired
  if (g_in_output().get_runtime_vtp_output_params()->output_linking_force())
    visualization_output_writer_ptr_->append_dof_based_result_data_vector(*force, 3, "force");

  // finalize everything and write all required files to file system
  visualization_output_writer_ptr_->write_to_disk(g_state().get_time_n(), g_state().get_step_n());

  // ************** BEGIN RUNTIME VTP OUTPUT *** OPTION 2: DIRECTLY *********
  // this section is just to get the idea and needs some minor modifications (indicated by Fixme)
  // this may serve as a template for visualization of stochastic&viscous forces, contact forces,
  // ...

  //  std::shared_ptr<RuntimeVtpWriter> vtp_writer_ptr =
  //      Teuchos::rcp( new RuntimeVtpWriter() );
  //
  //  // Todo: we need a better upper bound for total number of time steps here
  //  // however, this 'only' affects the number of leading zeros in the vtk file names
  //  const unsigned int num_timesteps_in_simulation_upper_bound = 1000000;
  //
  //
  //  // initialize the writer object
  //  vtp_writer_ptr->initialize();   // Fixme
  //
  //  // set geometry manually
  //  const unsigned int num_spatial_dimensions = 3;
  //  unsigned int num_row_points = 2000;
  //
  //  // get and prepare storage for point coordinate values
  //  std::vector< double >& point_coordinates = vtp_writer_ptr->GetPointCoordinateVector();
  //  point_coordinates.clear();
  //  point_coordinates.reserve( num_spatial_dimensions * num_row_points );
  //
  //
  //  // loop over my points and collect the geometry/grid data, i.e. reference positions of nodes
  //  for (unsigned int inode=0; inode < num_row_points; ++inode)
  //  {
  //    const Core::Nodes::Node* node = BinDiscretPtr()->lRowNode(inode);
  //
  //    for (unsigned int idim=0; idim<num_spatial_dimensions; ++idim)
  //      point_coordinates.push_back( node->X()[idim] );
  //  }
  //
  //
  //  // reset time and time step and geometry name in the writer object
  //  vtp_writer_ptr->SetupForNewTimeStepAndGeometry(
  //      global_state().get_time_n(), global_state().get_step_n(), BinDiscretPtr()->name() );
  //
  //
  //
  //  // append all desired output data to the writer object's storage
  //
  //  // number of bonds: collect data and append to visualization results if desired
  //  std::vector< double > num_bonds( num_row_points );
  //
  //  for ( unsigned int i = 0; i < num_row_points; ++i )
  //  {
  //    CrossLinking::CrosslinkerNode *crosslinker_i =
  //        dynamic_cast< CrossLinking::CrosslinkerNode* >( bindis.lRowNode(i) );
  //
  //    num_bonds[i] = crosslinker_i->ClData()->GetNumberOfBonds();
  //  }
  //
  //  vtp_writer_ptr->AppendVisualizationPointDataVector( num_bonds, 1, "num_bonds" );
  //
  //
  //
  //  // finalize everything and write all required VTU files to filesystem
  //  vtp_writer_ptr->WriteFiles();
  //
  //
  //  // write a collection file summarizing all previously written files
  //  vtp_writer_ptr->WriteCollectionFileOfAllWrittenFiles();   // Fixme
  // ************** END RUNTIME VTP OUTPUT ***************************************
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::fill_state_data_vectors_for_output(
    Core::LinAlg::Vector<double>& displacement, Core::LinAlg::Vector<double>& orientation,
    Core::LinAlg::Vector<double>& numberofbonds, Core::LinAlg::Vector<double>& owner,
    Core::LinAlg::Vector<double>& force) const
{
  check_init_setup();
  Core::FE::Discretization const& bindis = bin_discret();

  const unsigned int num_spatial_dim = 3;
  Core::LinAlg::SerialDenseVector bspotforce(num_spatial_dim);

  // todo: this is of course not nice, this needs to be done somewhere else
  for (int i = 0; i < bindis.num_my_row_nodes(); ++i)
  {
    Core::Nodes::Node* crosslinker_i = bindis.l_row_node(i);
    // std::vector holding gids of dofs
    std::vector<int> dofnode = bindis.dof(crosslinker_i);
    int numbonds = crosslinker_data_[crosslinker_i->lid()]->get_number_of_bonds();

    std::shared_ptr<BeamInteraction::BeamLink> beamlink = nullptr;

    if (numbonds == 2)
    {
      beamlink = doublebondcl_.at(crosslinker_i->id());
      beamlink->get_binding_spot_force(0, bspotforce);
    }

    // loop over all dofs
    for (unsigned int dim = 0; dim < num_spatial_dim; ++dim)
    {
      int doflid = displacement.get_map().lid(dofnode[dim]);
      (displacement).get_values()[doflid] = crosslinker_i->x()[dim];

      if (numbonds == 2)
      {
        (orientation).get_values()[doflid] =
            beamlink->get_bind_spot_pos1()(dim) - beamlink->get_bind_spot_pos2()(dim);
        (force).get_values()[doflid] = bspotforce(dim);
      }
      else
      {
        (orientation).get_values()[doflid] = 0.0;
        (force).get_values()[doflid] = 0.0;
      }
    }

    (numberofbonds).get_values()[i] = numbonds;
    (owner).get_values()[i] = crosslinker_i->owner();
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::reset_step_state() { check_init_setup(); }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::write_restart(
    Core::IO::DiscretizationWriter& ia_writer, Core::IO::DiscretizationWriter& bin_writer) const
{
  check_init_setup();

  // -------------------------------------------------------------------------
  // 1) write list of double bonded crosslinker on each proc
  // -------------------------------------------------------------------------
  Core::Communication::PackBuffer linker_buffer;
  for (auto const& iter : doublebondcl_)
  {
    std::shared_ptr<BeamInteraction::BeamLink> btbl = iter.second;
    btbl->pack(linker_buffer);
  }

  std::shared_ptr<std::vector<char>> db_linker = std::make_shared<std::vector<char>>();
  std::swap(*db_linker, linker_buffer());

  // -------------------------------------------------------------------------
  // 2) write crosslinker data
  // -------------------------------------------------------------------------
  Core::Communication::PackBuffer cldata_buffer;
  unsigned int numrowcl = bin_discret().num_my_row_nodes();
  for (unsigned int i = 0; i < numrowcl; ++i)
  {
    int const clgid = bin_discret().node_row_map()->gid(i);
    std::shared_ptr<BeamInteraction::Data::CrosslinkerData> cl_data_i =
        crosslinker_data_[bin_discret().node_col_map()->lid(clgid)];

    cl_data_i->pack(cldata_buffer);
  }

  std::shared_ptr<std::vector<char>> cldata = std::make_shared<std::vector<char>>();
  std::swap(*cldata, cldata_buffer());

  // -------------------------------------------------------------------------
  // 3) beam data
  // -------------------------------------------------------------------------
  Core::Communication::PackBuffer beamdata_buffer;
  unsigned int numrowbeam = ele_type_map_extractor().beam_map()->num_my_elements();
  for (unsigned int i = 0; i < numrowbeam; ++i)
  {
    int const beamgid = ele_type_map_extractor().beam_map()->gid(i);
    std::shared_ptr<BeamInteraction::Data::BeamData> beam_data_i =
        beam_data_[discret().element_col_map()->lid(beamgid)];

#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (beam_data_i == nullptr)
      FOUR_C_THROW("beam data of row beam with gid {} not there", beamgid);
#endif

    beam_data_i->pack(beamdata_buffer);
  }

  std::shared_ptr<std::vector<char>> beamdata = std::make_shared<std::vector<char>>();
  std::swap(*beamdata, beamdata_buffer());

  // -------------------------------------------------------------------------
  // write data
  // -------------------------------------------------------------------------
  bin_writer.write_char_data("Linker", *db_linker);
  bin_writer.write_char_data("ClData", *cldata);
  bin_writer.write_char_data("BeamData", *beamdata);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::pre_read_restart()
{
  check_init_setup();
  beam_crosslinker_handler_ptr()->remove_all_linker();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::read_restart(
    Core::IO::DiscretizationReader& ia_reader, Core::IO::DiscretizationReader& bin_reader)
{
  check_init_setup();

  // -------------------------------------------------------------------------
  // 1) read list of double bonded crosslinker on each proc
  // -------------------------------------------------------------------------
  std::shared_ptr<std::vector<char>> linkercharvec;
  bin_reader.read_char_vector(linkercharvec, "Linker");

  {
    Core::Communication::UnpackBuffer buffer(*linkercharvec);
    while (!buffer.at_end())
    {
      std::shared_ptr<Core::Communication::ParObject> object(Core::Communication::factory(buffer));
      std::shared_ptr<BeamInteraction::BeamLink> beamtobeamlink =
          std::dynamic_pointer_cast<BeamInteraction::BeamLink>(object);
      if (beamtobeamlink == nullptr) FOUR_C_THROW("Failed to build a node from the node data");

      // insert in my list of double bonded crosslinker
      doublebondcl_[beamtobeamlink->id()] = beamtobeamlink;
    }
  }

  // -------------------------------------------------------------------------
  // 2) read crosslinker data
  // -------------------------------------------------------------------------

  std::shared_ptr<std::vector<char>> cldata_charvec;
  bin_reader.read_char_vector(cldata_charvec, "ClData");
  crosslinker_data_.resize(bin_discret().num_my_col_nodes());

  std::map<int, std::shared_ptr<BeamInteraction::Data::CrosslinkerData>> cl_not_owned;
  std::set<int> not_owned_gids;
  std::map<int, std::vector<char>> cl_datapacks;
  std::vector<int> read_node_ids;
  Core::Communication::UnpackBuffer buffer(*cldata_charvec);
  while (!buffer.at_end())
  {
    // unpack
    std::shared_ptr<BeamInteraction::Data::CrosslinkerData> cl_data(
        BeamInteraction::Data::create_data_container<BeamInteraction::Data::CrosslinkerData>(
            buffer));

    int const cl_gid = cl_data->get_id();
    read_node_ids.push_back(cl_gid);

    // repack it for communication to find owner
    Core::Communication::PackBuffer data;
    cl_data->pack(data);
    cl_datapacks[cl_gid].insert(cl_datapacks[cl_gid].end(), data().begin(), data().end());
  }

  // build dummy map according to read data on myrank
  Core::LinAlg::Map dummy_cl_map(
      -1, read_node_ids.size(), read_node_ids.data(), 0, bin_discret().get_comm());

  // build exporter object
  std::shared_ptr<Core::Communication::Exporter> exporter =
      std::make_shared<Core::Communication::Exporter>(
          dummy_cl_map, *bin_discret().node_col_map(), bin_discret().get_comm());

  // export
  exporter->do_export(cl_datapacks);

  // rebuild data container
  crosslinker_data_.resize(bin_discret().num_my_col_nodes());
  for (auto& iter : cl_datapacks)
  {
    Core::Communication::UnpackBuffer cl_buffer(iter.second);
    std::shared_ptr<BeamInteraction::Data::CrosslinkerData> cl_data(
        BeamInteraction::Data::create_data_container<BeamInteraction::Data::CrosslinkerData>(
            cl_buffer));
    crosslinker_data_[bin_discret().node_col_map()->lid(cl_data->get_id())] = cl_data;
  }

  // -------------------------------------------------------------------------
  // 3) read beam data
  // -------------------------------------------------------------------------
  std::shared_ptr<std::vector<char>> beamdata_charvec;
  bin_reader.read_char_vector(beamdata_charvec, "BeamData");
  beam_data_.resize(discret().num_my_col_elements());

  std::map<int, std::shared_ptr<BeamInteraction::Data::BeamData>> beams_not_owned;
  not_owned_gids.clear();
  std::map<int, std::vector<char>> beam_datapacks;
  std::vector<int> read_ele_ids;
  Core::Communication::UnpackBuffer beamdata_buffer(*beamdata_charvec);
  while (!beamdata_buffer.at_end())
  {
    // unpack
    std::shared_ptr<BeamInteraction::Data::BeamData> beam_data(
        BeamInteraction::Data::create_data_container<BeamInteraction::Data::BeamData>(
            beamdata_buffer));

    int const beam_gid = beam_data->get_id();
    read_ele_ids.push_back(beam_gid);

    // repack it for communication to find owner
    Core::Communication::PackBuffer data;
    beam_data->pack(data);
    beam_datapacks[beam_gid].insert(beam_datapacks[beam_gid].end(), data().begin(), data().end());
  }

  // build dummy map according to read data on myrank
  std::shared_ptr<Core::LinAlg::Map> dummy_beam_map = std::make_shared<Core::LinAlg::Map>(
      -1, read_ele_ids.size(), read_ele_ids.data(), 0, discret().get_comm());

  // build exporter object
  exporter = std::make_shared<Core::Communication::Exporter>(
      *dummy_beam_map, *discret().element_col_map(), discret().get_comm());

  // export
  exporter->do_export(beam_datapacks);

  // rebuild data container
  beam_data_.resize(discret().num_my_col_elements());
  for (auto& iter : beam_datapacks)
  {
    Core::Communication::UnpackBuffer buffer(iter.second);
    std::shared_ptr<BeamInteraction::Data::BeamData> beam_data(
        BeamInteraction::Data::create_data_container<BeamInteraction::Data::BeamData>(buffer));
    beam_data_[discret().element_col_map()->lid(beam_data->get_id())] = beam_data;
  }

  // init maps
  store_maps_prior_redistribution();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::post_read_restart()
{
  check_init_setup();

  // bring each object in doublebondcl_ map to its correct owner
  update_my_double_bonds_remote_id_list();

  // store displacement of restart step as displacement state of last redistribution
  dis_at_last_redistr_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*bin_discret().dof_row_map(), true);
  for (int i = 0; i < bin_discret().num_my_row_nodes(); ++i)
  {
    Core::Nodes::Node* crosslinker_i = bin_discret().l_row_node(i);

    // std::vector holding gids of dofs
    std::vector<int> dofnode = bin_discret().dof(crosslinker_i);

    // loop over all dofs
    for (unsigned int dim = 0; dim < 3; ++dim)
    {
      int doflid = dis_at_last_redistr_->get_map().lid(dofnode[dim]);
      (*dis_at_last_redistr_).get_values()[doflid] = crosslinker_i->x()[dim];
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::add_bins_to_bin_col_map(
    std::set<int>& colbins)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::
    add_bins_with_relevant_content_for_ia_discret_col_map(std::set<int>& colbins) const
{
  check_init_setup();

  bool map_changed = false;
  if (not(cl_noderowmap_prior_redistr_->same_as(*bin_discret().node_row_map()) and
          cl_nodecolmap_prior_redistr_->same_as(*bin_discret().node_col_map())))
    map_changed = true;

  std::set<int> colbinsext;
  for (auto const& iter : colbins)
  {
    // get current bin
    Core::Elements::Element* currbin = bin_discret().g_element(iter);
    // get all crosslinker in current bin
    Core::Nodes::Node** clincurrentbin = currbin->nodes();

    // loop over all crosslinker in CurrentBin
    for (int i = 0; i < currbin->num_node(); ++i)
    {
      if (map_changed or (crosslinker_data_.size() == 0) or
          (crosslinker_data_[clincurrentbin[i]->lid()]->get_number_of_bonds() > 0))
      {
        std::vector<int> binvec;
        bin_strategy().get_neighbor_bin_ids(iter, binvec);
        colbinsext.insert(binvec.begin(), binvec.end());
        // go to next bin
        break;
      }
    }
  }

  colbins.insert(colbinsext.begin(), colbinsext.end());

  // the following would be a default additional one ghost layer

  //
  //  std::set< int > colbinsext(colbins.begin(),colbins.end());
  //  std::set< int >::const_iterator biniter;
  //  for ( biniter = colbinsext.begin(); biniter != colbinsext.end() ; ++biniter )
  //  {
  //    std::vector< int > binvec;
  //    BinStrategy().get_neighbor_and_own_bin_ids( *biniter, binvec );
  //    colbinsext.insert( binvec.begin(), binvec.end() );
  //  }
  //
  //  colbins.insert( colbinsext.begin(), colbinsext.end() );
}

/*-------------------------------------------------------------------------------*
 *-------------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::get_half_interaction_distance(
    double& half_interaction_distance)
{
  check_init_setup();

  // loop over all linker of all linker types and get largest linker (also
  // considering tolerance)
  double curr_ia_dist = -1.0;
  double local_half_interaction_distance = 0.0;
  const int numrowcl = bin_discret_ptr()->num_my_row_nodes();
  for (int rowcli = 0; rowcli < numrowcl; ++rowcli)
  {
    // get current linker
    CrossLinking::CrosslinkerNode* crosslinker_i =
        dynamic_cast<CrossLinking::CrosslinkerNode*>(bin_discret_ptr()->l_row_node(rowcli));

    curr_ia_dist = 0.5 * (crosslinker_i->get_material()->linking_length() +
                             crosslinker_i->get_material()->linking_length_tolerance());

    local_half_interaction_distance = (curr_ia_dist > local_half_interaction_distance)
                                          ? curr_ia_dist
                                          : local_half_interaction_distance;
  }

  // get global maximum
  double global_half_interaction_distance = 0.0;
  // build sum over all procs
  MPI_Allreduce(&local_half_interaction_distance, &global_half_interaction_distance, 1, MPI_DOUBLE,
      MPI_MAX, discret().get_comm());
  half_interaction_distance_ = global_half_interaction_distance;

  // some screen output
  if (g_state().get_my_rank() == 0)
    Core::IO::cout(Core::IO::verbose) << " beam to beam crosslinking half interaction distance "
                                      << global_half_interaction_distance << Core::IO::endl;

  half_interaction_distance = (half_interaction_distance_ > half_interaction_distance)
                                  ? half_interaction_distance_
                                  : half_interaction_distance;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::diffuse_crosslinker()
{
  check_init_setup();

  // loop over all row crosslinker (beam binding status not touched here)
  const int numrowcl = bin_discret_ptr()->num_my_row_nodes();
  for (int rowcli = 0; rowcli < numrowcl; ++rowcli)
  {
    // get current linker
    CrossLinking::CrosslinkerNode* crosslinker_i =
        dynamic_cast<CrossLinking::CrosslinkerNode*>(bin_discret_ptr()->l_row_node(rowcli));

#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (crosslinker_i->num_element() != 1)
      FOUR_C_THROW("More than one element for this crosslinker");
#endif

    BeamInteraction::Data::CrosslinkerData* cldata_i =
        crosslinker_data_[crosslinker_i->lid()].get();

    // different treatment according to number of bonds a crosslinker has
    switch (cldata_i->get_number_of_bonds())
    {
      case 0:
      {
        // crosslinker has zero bonds, i.e. is free to diffuse according to
        // brownian dynamics
        diffuse_unbound_crosslinker(crosslinker_i, cldata_i);
        break;
      }
      case 1:
      {
        // get clbspot that is currently bonded
        int occbspotid = get_single_occupied_cl_bspot(cldata_i->get_b_spots());

        // get current position of binding spot of filament partner
        // note: we can not use our beam data container, as bspot position is not current position
        // (as this is the result of a sum, you can not have a reference to that)
        const int elegid = cldata_i->get_b_spots()[occbspotid].first;

        Discret::Elements::Beam3Base* ele =
            dynamic_cast<Discret::Elements::Beam3Base*>(discret_ptr()->g_element(elegid));

#ifdef FOUR_C_ENABLE_ASSERTIONS
        // safety check
        const int colelelid = discret_ptr()->element_col_map()->lid(elegid);
        if (colelelid < 0)
          FOUR_C_THROW(
              "Crosslinker has {} bonds but its binding partner with gid {} "
              "is \nnot ghosted/owned on proc {} (owner of crosslinker)",
              cldata_i->get_number_of_bonds(), elegid, g_state().get_my_rank());
        // safety check
        if (ele == nullptr)
          FOUR_C_THROW(
              "Dynamic cast of ele with gid {} failed on proc ", elegid, g_state().get_my_rank());
#endif

        // get current position of filament binding spot
        Core::LinAlg::Matrix<3, 1> bbspotpos;
        std::vector<double> eledisp;
        BeamInteraction::Utils::get_current_unshifted_element_dis(discret(), ele,
            *beam_interaction_data_state_ptr()->get_dis_col_np(), periodic_bounding_box(), eledisp);
        ele->get_pos_of_binding_spot(bbspotpos, eledisp,
            crosslinker_i->get_material()->linker_type(),
            cldata_i->get_b_spots()[occbspotid].second, periodic_bounding_box());

        // note: a crosslinker can not leave the computational domain here, as no beam binding
        // spot can be outside the periodic box at this point
        std::vector<double> newpos(3, 0.0);
        for (int dim = 0; dim < 3; ++dim) newpos[dim] = bbspotpos(dim);

        crosslinker_i->set_pos(newpos);
        cldata_i->set_position(bbspotpos);

        break;
      }
      case 2:
      {
        // crosslinker has two bonds (cl gets current mid position between the filament
        // binding spot it is attached to)
        // -----------------------------------------------------------------
        // partner one
        // -----------------------------------------------------------------
        int elegid = cldata_i->get_b_spots()[0].first;

#ifdef FOUR_C_ENABLE_ASSERTIONS
        if (elegid < 0 or cldata_i->get_b_spots()[0].second < 0)
          FOUR_C_THROW(
              " double bonded crosslinker has stored beam partner gid or loc bsponum of -1, "
              " something went wrong");
        // safety check
        int colelelid = discret_ptr()->element_col_map()->lid(elegid);
        if (colelelid < 0)
          FOUR_C_THROW(
              "Crosslinker has {} bonds but its binding partner with gid {} "
              "is not \nghosted/owned on proc {} (owner of crosslinker)",
              cldata_i->get_number_of_bonds(), elegid, g_state().get_my_rank());
#endif

        Discret::Elements::Beam3Base* ele =
            dynamic_cast<Discret::Elements::Beam3Base*>(discret_ptr()->g_element(elegid));

#ifdef FOUR_C_ENABLE_ASSERTIONS
        // safety check
        if (ele == nullptr)
          FOUR_C_THROW(
              "Dynamic cast of ele with gid {} failed on proc ", elegid, g_state().get_my_rank());
#endif

        // get current position of filament binding spot
        Core::LinAlg::Matrix<3, 1> bbspotposone;
        std::vector<double> eledisp;
        BeamInteraction::Utils::get_current_unshifted_element_dis(discret(), ele,
            *beam_interaction_data_state_ptr()->get_dis_col_np(), periodic_bounding_box(), eledisp);
        ele->get_pos_of_binding_spot(bbspotposone, eledisp,
            crosslinker_i->get_material()->linker_type(), cldata_i->get_b_spots()[0].second,
            periodic_bounding_box());

        // -----------------------------------------------------------------
        // partner two
        // -----------------------------------------------------------------
        elegid = cldata_i->get_b_spots()[1].first;

#ifdef FOUR_C_ENABLE_ASSERTIONS
        // safety check
        if (elegid < 0 or cldata_i->get_b_spots()[1].second < 0)
          FOUR_C_THROW(
              " double bonded crosslinker has stored beam partner gid or loc bsponum of -1, "
              " something went wrong");
        colelelid = discret_ptr()->element_col_map()->lid(elegid);
        if (colelelid < 0)
          FOUR_C_THROW(
              "Crosslinker has {} bonds but its binding partner with gid {} "
              "is \nnot ghosted/owned on proc {} (owner of crosslinker)",
              cldata_i->get_number_of_bonds(), elegid, g_state().get_my_rank());
#endif

        ele = dynamic_cast<Discret::Elements::Beam3Base*>(discret_ptr()->g_element(elegid));

#ifdef FOUR_C_ENABLE_ASSERTIONS
        // safety check
        if (ele == nullptr)
          FOUR_C_THROW(
              "Dynamic cast of ele with gid {} failed on proc ", elegid, g_state().get_my_rank());
#endif

        // get current position of filament binding spot
        Core::LinAlg::Matrix<3, 1> bbspotpostwo;
        BeamInteraction::Utils::get_current_unshifted_element_dis(discret(), ele,
            *beam_interaction_data_state_ptr()->get_dis_col_np(), periodic_bounding_box(), eledisp);
        ele->get_pos_of_binding_spot(bbspotpostwo, eledisp,
            crosslinker_i->get_material()->linker_type(), cldata_i->get_b_spots()[1].second,
            periodic_bounding_box());

        Core::LinAlg::Matrix<3, 1> clpos(Core::LinAlg::Initialization::zero);
        set_position_of_double_bonded_crosslinker_pb_cconsistent(clpos, bbspotposone, bbspotpostwo);

        std::vector<double> newpos(3, 0.0);
        for (int dim = 0; dim < 3; ++dim) newpos[dim] = clpos(dim);

        crosslinker_i->set_pos(newpos);
        cldata_i->set_position(clpos);

        break;
      }
      default:
      {
        FOUR_C_THROW(
            "Unrealistic number {} of bonds for a crosslinker.", cldata_i->get_number_of_bonds());
      }
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::diffuse_unbound_crosslinker(
    Core::Nodes::Node* crosslinker_i, BeamInteraction::Data::CrosslinkerData* cldata_i)
{
  check_init();

  CrossLinking::CrosslinkerNode* crosslinker =
      dynamic_cast<CrossLinking::CrosslinkerNode*>(crosslinker_i);

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (crosslinker == nullptr) FOUR_C_THROW("Dynamic cast to CrosslinkerNode failed");
#endif

  // get standard deviation and mean value for crosslinker that are free to
  // diffuse
  double standarddev = std::sqrt(2.0 * crosslinking_params_ptr_->kt() /
                                 (3.0 * std::numbers::pi * crosslinking_params_ptr_->viscosity() *
                                     crosslinker->get_material()->linking_length()) *
                                 crosslinking_params_ptr_->delta_time());
  double meanvalue = 0.0;

  // diffuse crosslinker according to brownian dynamics
  Core::LinAlg::Matrix<3, 1> newclpos(Core::LinAlg::Initialization::zero);
  int count = 3;
  std::vector<double> randvec(count);
  // maximal diffusion given by cutoff radius (sqrt(3) = 1.73..)
  double const maxmov = bin_strategy().bin_size_lower_bound() / 1.74;
  if (standarddev > 0.0)
  {
    Global::Problem::instance()->random()->set_mean_stddev(meanvalue, standarddev);
    Global::Problem::instance()->random()->normal(randvec, count);
  }
  for (int dim = 0; dim < 3; ++dim)
  {
    if (abs(randvec[dim]) > maxmov)
    {
      double old = randvec[dim];
      randvec[dim] = (abs(randvec[dim]) / randvec[dim]) * maxmov;
      Core::IO::cout(Core::IO::verbose) << "Movement of free crosslinker " << crosslinker->id()
                                        << " was restricted by cutoff radius"
                                           " in "
                                        << dim << " direction. " << old << " to " << randvec[dim]
                                        << "\nThis should not happen to often "
                                           "to stay physical. Increase cutoff or reduce movement"
                                        << Core::IO::endl;
    }
    newclpos(dim) = crosslinker->x()[dim] + randvec[dim];
  }

  // check compliance with periodic boundary conditions
  periodic_bounding_box().shift_3d(newclpos);
  std::vector<double> newpos(3, 0.0);
  for (int dim = 0; dim < 3; ++dim) newpos[dim] = newclpos(dim);
  crosslinker->set_pos(newpos);

  cldata_i->set_position(newclpos);
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
int BeamInteraction::SubmodelEvaluator::Crosslinking::get_single_occupied_cl_bspot(
    std::vector<std::pair<int, int>> const& clbspots) const
{
  check_init();

  if (clbspots[0].first > -1)
    return 0;
  else if (clbspots[1].first > -1)
    return 1;
  else
    FOUR_C_THROW("numbond = 1 but both binding spots store invalid element GIDs!");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::
    set_position_of_double_bonded_crosslinker_pb_cconsistent(Core::LinAlg::Matrix<3, 1>& clpos,
        Core::LinAlg::Matrix<3, 1> const& bspot1pos,
        Core::LinAlg::Matrix<3, 1> const& bspot2pos) const
{
  /* the position of (the center) of a double-bonded crosslinker is defined as
   * midpoint between the two given binding spot positions. (imagine a linker
   * being a slender body with a binding domain at each of both ends) */

  /* if the two binding spots are separated by a periodic boundary, we need to
   * shift one position back to get the interpolation right */
  clpos = bspot2pos;
  periodic_bounding_box().un_shift_3d(clpos, bspot1pos);

  // fixme: to avoid senseless FOUR_C_THROW in debug mode
  Core::LinAlg::Matrix<3, 1> dummy(clpos);
  clpos.update(0.5, bspot1pos, 0.5, dummy);

  // shift the interpolated position back in the periodic box if necessary
  periodic_bounding_box().shift_3d(clpos);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::set_position_of_newly_free_crosslinker(
    CrossLinking::CrosslinkerNode* crosslinker, BeamInteraction::Data::CrosslinkerData* cldata)
{
  check_init();

  // generate vector in random direction
  // of length half the linking length to "reset" crosslink molecule position: it may now
  // reenter or leave the bonding proximity
  // todo: does this make sense?
  Core::LinAlg::Matrix<3, 1> clpos(cldata->get_position());
  Core::LinAlg::Matrix<3, 1> cldeltapos_i;
  std::vector<double> randunivec(3);
  int count = 3;
  Global::Problem::instance()->random()->uni(randunivec, count);
  for (unsigned int dim = 0; dim < 3; ++dim) cldeltapos_i(dim) = randunivec[dim];

  cldeltapos_i.scale(crosslinker->get_material()->linking_length() / cldeltapos_i.norm2());

  clpos.update(1.0, cldeltapos_i, 1.0);

  periodic_bounding_box().shift_3d(clpos);

  std::vector<double> newpos(3, 0.0);
  for (int dim = 0; dim < 3; ++dim) newpos[dim] = clpos(dim);
  crosslinker->set_pos(newpos);

  cldata->set_position(clpos);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::
    set_position_of_newly_single_bonded_crosslinker(CrossLinking::CrosslinkerNode* crosslinker,
        BeamInteraction::Data::CrosslinkerData* cldata, int stayoccpotid)
{
  check_init();

  // update position
  const int collidoccbeam =
      discret_ptr()->element_col_map()->lid(cldata->get_b_spots()[stayoccpotid].first);

#ifdef FOUR_C_ENABLE_ASSERTIONS
  // safety check
  if (collidoccbeam < 0)
    FOUR_C_THROW("element with gid {} not ghosted on proc {}",
        cldata->get_b_spots()[stayoccpotid].first, g_state().get_my_rank());
#endif

  BeamInteraction::Data::BeamData const* beamdata_i = beam_data_[collidoccbeam].get();
  Core::LinAlg::Matrix<3, 1> clpos(beamdata_i->get_b_spot_position(
      crosslinker->get_material()->linker_type(), cldata->get_b_spots()[stayoccpotid].second));

  std::vector<double> newpos(3, 0.0);
  for (int dim = 0; dim < 3; ++dim) newpos[dim] = clpos(dim);
  crosslinker->set_pos(newpos);

  cldata->set_position(clpos);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::store_maps_prior_redistribution()
{
  check_init();

  *cl_noderowmap_prior_redistr_ = *bin_discret().node_row_map();
  *cl_nodecolmap_prior_redistr_ = *bin_discret().node_col_map();
  *beam_elerowmap_prior_redistr_ = *ele_type_map_extractor().beam_map();
  *beam_elecolmap_prior_redistr_ = *discret().element_col_map();
}
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::update_and_export_crosslinker_data()
{
  check_init();

  TEUCHOS_FUNC_TIME_MONITOR(
      "BeamInteraction::SubmodelEvaluator::Crosslinking::"
      "update_and_export_crosslinker_data");

  //  // if one proc has changed maps, all procs have to update exporter
  //  int loc_changed_maps = static_cast< int >( (cl_exporter_ == nullptr) or not
  //      ( cl_exporter_->SourceMap().SameAs( *cl_noderowmap_prior_redistr_ ) and
  //        cl_nodecolmap_prior_redistr_->SameAs(*BinDiscret().NodeColMap() ) ) );
  //
  //  //global filled flag (is true / one if and only if loc_changed_maps == true on each processor
  //  int g_changed_maps = 0;
  //
  //  /*the global flag is set to the maximal value of any local flag
  //   * i.e. if on any processor loc_changed_maps == true, the flag g_changed_maps is set to
  //   * one*/
  //  Core::Communication::max_all( &loc_changed_maps, &g_changed_maps, 1 , BinDiscret().Comm());
  //
  //  if ( g_changed_maps )
  cl_exporter_ = std::make_shared<Core::Communication::Exporter>(
      *cl_noderowmap_prior_redistr_, *bin_discret().node_col_map(), bin_discret().get_comm());

  // we first need to pack our stuff into and std::vector< char > for communication
  std::map<int, std::vector<char>> allpacks;
  unsigned int numrowcl = cl_noderowmap_prior_redistr_->num_my_elements();
  for (unsigned int i = 0; i < numrowcl; ++i)
  {
    int const clgid = cl_noderowmap_prior_redistr_->gid(i);
    std::shared_ptr<BeamInteraction::Data::CrosslinkerData> cl_data_i =
        crosslinker_data_[cl_nodecolmap_prior_redistr_->lid(clgid)];

    Core::Communication::PackBuffer data;
    cl_data_i->pack(data);
    allpacks[clgid].insert(allpacks[clgid].end(), data().begin(), data().end());
  }

  // export
  cl_exporter_->do_export(allpacks);

  // rebuild data container
  crosslinker_data_.resize(bin_discret().num_my_col_nodes());
  for (auto& iter : allpacks)
  {
    Core::Communication::UnpackBuffer buffer(iter.second);
    std::shared_ptr<BeamInteraction::Data::CrosslinkerData> cl_data(
        BeamInteraction::Data::create_data_container<BeamInteraction::Data::CrosslinkerData>(
            buffer));
    crosslinker_data_[bin_discret().node_col_map()->lid(cl_data->get_id())] = cl_data;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::update_and_export_beam_data(
    bool update_states)
{
  check_init();

  //  TEUCHOS_FUNC_TIME_MONITOR("BeamInteraction::SubmodelEvaluator::Crosslinking::"
  //      "update_and_export_beam_data");
  //
  //  // if maps have change on one proc, all procs have to rebuild the exporter object
  //  int loc_changed_maps = static_cast< int >( (beam_exporter_ == nullptr) or not
  //      ( beam_exporter_->SourceMap().SameAs( *beam_elerowmap_prior_redistr_ ) and
  //        beam_elecolmap_prior_redistr_->SameAs( *discret().ElementColMap() ) ) );
  //
  //  //global filled flag (is true / one if and only if loc_changed_maps == true on each processor
  //  int g_changed_maps = 0;
  //
  //  /*the global flag is set to the maximal value of any local flag
  //   * i.e. if on any processor loc_changed_maps == true, the flag g_changed_maps is set to
  //   * one*/
  //  Core::Communication::max_all( &loc_changed_maps, &g_changed_maps, 1 , discret().Comm());
  //
  //  if ( g_changed_maps )
  beam_exporter_ = std::make_shared<Core::Communication::Exporter>(
      *beam_elerowmap_prior_redistr_, *discret().element_col_map(), discret().get_comm());

  // we first need to pack our row stuff into and std::vector< char > for communication
  std::map<int, std::vector<char>> allpacks;
  unsigned int numrowele = beam_elerowmap_prior_redistr_->num_my_elements();
  for (unsigned int i = 0; i < numrowele; ++i)
  {
    int const elegid = beam_elerowmap_prior_redistr_->gid(i);

    std::shared_ptr<BeamInteraction::Data::BeamData> beam_data_i =
        beam_data_[beam_elecolmap_prior_redistr_->lid(elegid)];

    // safety check
    if (beam_data_i == nullptr)
      FOUR_C_THROW("beam data container for beam with gid {} not there on rank {} ", elegid,
          g_state().get_my_rank());

    if (update_states)
    {
      // safety check
      if (discret().element_col_map()->lid(elegid) < 0)
        FOUR_C_THROW(" Element {} has moved too far between two redistributions.", elegid);

      // beam element i for which data will be collected
      Discret::Elements::Beam3Base* beamele_i =
          dynamic_cast<Discret::Elements::Beam3Base*>(discret().g_element(elegid));

      // go to next element in case the current one is not a beam element
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (beamele_i == nullptr) FOUR_C_THROW("cast did not work");
#endif

      std::vector<double> eledisp;
      BeamInteraction::Utils::get_current_unshifted_element_dis(discret(), beamele_i,
          *beam_interaction_data_state_ptr()->get_dis_col_np(), periodic_bounding_box(), eledisp);

      // loop over binding spot types of current element
      for (auto const& iter : beamele_i->get_binding_spots())
      {
        // loop over all binding spots of current type j of current element
        int unsigned const numbbspot = beamele_i->get_number_of_binding_spots(iter.first);
        Core::LinAlg::Matrix<3, 1> pos(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<3, 3> triad(Core::LinAlg::Initialization::zero);
        for (int unsigned k = 0; k < numbbspot; ++k)
        {
          BeamInteraction::Utils::get_pos_and_triad_of_binding_spot(
              beamele_i, *periodic_bounding_box_ptr(), iter.first, k, pos, triad, eledisp);

          beam_data_i->set_b_spot_position(iter.first, k, pos);
          beam_data_i->set_b_spot_triad(iter.first, k, triad);
        }
      }
    }

    Core::Communication::PackBuffer data;
    beam_data_i->pack(data);
    allpacks[elegid].insert(allpacks[elegid].end(), data().begin(), data().end());
    //    allpacks[elegid] = data();
  }

  // export
  beam_exporter_->do_export(allpacks);

  // rebuild data container
  beam_data_.resize(discret().num_my_col_elements());
  for (auto& iter : allpacks)
  {
    Core::Communication::UnpackBuffer buffer(iter.second);
    std::shared_ptr<BeamInteraction::Data::BeamData> beam_data(
        BeamInteraction::Data::create_data_container<BeamInteraction::Data::BeamData>(buffer));
    beam_data_[discret().element_col_map()->lid(beam_data->get_id())] = beam_data;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::bind_and_unbind_crosslinker()
{
  check_init_setup();

  TEUCHOS_FUNC_TIME_MONITOR(
      "BeamInteraction::SubmodelEvaluator::"
      "Crosslinking::bind_and_unbind_crosslinker");

  // manage binding events
  int num_new_linker = bind_crosslinker();

  // manage unbinding events
  int num_dissolved_linker = un_bind_crosslinker();

  // write some information to screen
  std::vector<int> num_local(3, 0);
  std::vector<int> num_global(3, 0);
  num_local[0] = static_cast<int>(doublebondcl_.size());
  num_local[1] = num_new_linker;
  num_local[2] = num_dissolved_linker;
  MPI_Reduce(num_local.data(), num_global.data(), 3, MPI_INT, MPI_SUM, 0, discret().get_comm());
  if (g_state().get_my_rank() == 0)
  {
    Core::IO::cout(Core::IO::standard)
        << "\n************************************************" << Core::IO::endl;
    Core::IO::cout(Core::IO::standard) << "Beam to Beam Links: " << num_global[0];
    Core::IO::cout(Core::IO::standard) << " (New: " << num_global[1];
    Core::IO::cout(Core::IO::standard) << " Dissolved: " << num_global[2];
    Core::IO::cout(Core::IO::standard) << ")\n************************************************\n"
                                       << Core::IO::endl;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int BeamInteraction::SubmodelEvaluator::Crosslinking::bind_crosslinker()
{
  check_init();

  Global::Problem::instance()->random()->set_rand_range(0.0, 1.0);

  // intended bonds of row crosslinker on myrank (key is clgid)
  std::map<int, std::shared_ptr<BeamInteraction::Data::BindEventData>> mybonds;
  mybonds.clear();
  // intended bond col crosslinker to row element (key is owner of crosslinker != myrank)
  std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::BindEventData>>> undecidedbonds;
  undecidedbonds.clear();

  // fill binding event maps
  find_potential_binding_events(mybonds, undecidedbonds);

  // bind events where myrank only owns the elements, cl are taken care
  // of by their owner (key is clgid)
  std::map<int, std::shared_ptr<BeamInteraction::Data::BindEventData>> myelebonds;

  // now each row owner of a linker gets requests, makes a random decision and
  // informs back its requesters
  manage_binding_in_parallel(mybonds, undecidedbonds, myelebonds);

  // actual update of binding states is done here
  return update_my_crosslinker_and_element_binding_states(mybonds, myelebonds);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::find_potential_binding_events(
    std::map<int, std::shared_ptr<BeamInteraction::Data::BindEventData>>& mybonds,
    std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::BindEventData>>>&
        undecidedbonds)
{
  check_init();

  TEUCHOS_FUNC_TIME_MONITOR(
      "BeamInteraction::SubmodelEvaluator::Crosslinking::find_potential_binding_events");

  // this variable is used to check if a beam binding spot is linked twice on
  // myrank during a time step
  // ( first key is linkertype, second key is locbspotid, set holds gids of bonded crosslinker)
  std::map<int, std::vector<std::map<int, std::set<int>>>> intendedbeambonds;
  for (unsigned int i = 0; i < crosslinking_params_ptr_->linker_types().size(); ++i)
    intendedbeambonds[crosslinking_params_ptr_->linker_types()[i]].resize(
        discret_ptr()->num_my_row_elements());

  // store bins that have already been examined
  std::vector<int> examinedbins(bin_discret_ptr()->num_my_col_elements(), 0);
  // loop over all column crosslinker in random order
  // create random order of indices
  std::vector<int> rordercolcl =
      BeamInteraction::Utils::permutation(bin_discret_ptr()->num_my_col_nodes());

  for (auto const& icl : rordercolcl)
  {
    Core::Nodes::Node* currcrosslinker = bin_discret_ptr()->l_col_node(icl);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (currcrosslinker == nullptr) FOUR_C_THROW("Node not there");
    if (currcrosslinker->num_element() != 1)
      FOUR_C_THROW("More than one element for this crosslinker");
#endif

    // get bin that contains this crosslinker (can only be one)
    Core::Elements::Element* currentbin = currcrosslinker->adjacent_elements()[0].user_element();

#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (currentbin->id() < 0) FOUR_C_THROW(" negative bin id number {} ", currentbin->id());
#endif

    // if a bin has already been examined --> continue with next crosslinker
    if (examinedbins[currentbin->lid()]) continue;
    // else: bin is examined for the first time --> new entry in examinedbins_
    else
      examinedbins[currentbin->lid()] = 1;

    find_potential_binding_events_in_bin_and_neighborhood(
        currentbin, mybonds, undecidedbonds, intendedbeambonds, true);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::
    find_potential_binding_events_in_bin_and_neighborhood(Core::Elements::Element* bin,
        std::map<int, std::shared_ptr<BeamInteraction::Data::BindEventData>>& mybonds,
        std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::BindEventData>>>&
            undecidedbonds,
        std::map<int, std::vector<std::map<int, std::set<int>>>>& intendedbeambonds,
        bool checklinkingprop)
{
  check_init();

  // get neighboring bins
  // note: interaction distance cl to beam needs to be smaller than the bin size
  std::vector<int> neighboring_binIds;
  neighboring_binIds.reserve(27);
  // do not check on existence here -> shifted to GetBinContent
  bin_strategy_ptr()->get_neighbor_and_own_bin_ids(bin->id(), neighboring_binIds);

  // get set of neighboring beam elements (i.e. elements that somehow touch nb bins)
  // as explained above, we only need row elements (true flag in GetBinContent())
  std::set<Core::Elements::Element*> neighboring_row_beams;
  std::vector<Core::Binstrategy::Utils::BinContentType> bc_beam(
      1, Core::Binstrategy::Utils::BinContentType::Beam);
  bin_strategy_ptr()->get_bin_content(neighboring_row_beams, bc_beam, neighboring_binIds, true);
  std::set<Core::Elements::Element*> neighboring_col_spheres;
  std::vector<Core::Binstrategy::Utils::BinContentType> bc_sphere(
      1, Core::Binstrategy::Utils::BinContentType::RigidSphere);
  bin_strategy_ptr()->get_bin_content(
      neighboring_col_spheres, bc_sphere, neighboring_binIds, false);


  // in case there are no neighbors, go to next crosslinker (an therefore bin)
  if (neighboring_row_beams.empty()) return;

  // get all crosslinker in current bin
  Core::Nodes::Node** clincurrentbin = bin->nodes();
  const int numcrosslinker = bin->num_node();

  // obtain random order in which crosslinker are addressed
  std::vector<int> randorder = BeamInteraction::Utils::permutation(numcrosslinker);

  // loop over all crosslinker in CurrentBin in random order
  for (auto const& randcliter : randorder)
  {
    // get random crosslinker in current bin
    Core::Nodes::Node* crosslinker_i = clincurrentbin[randcliter];

    // todo: this can be done more efficiently
    if (check_if_sphere_prohibits_binding(neighboring_col_spheres, crosslinker_i)) continue;

    // get all potential binding events on myrank
    prepare_binding(crosslinker_i, neighboring_row_beams, mybonds, undecidedbonds,
        intendedbeambonds, checklinkingprop);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool BeamInteraction::SubmodelEvaluator::Crosslinking::check_if_sphere_prohibits_binding(
    std::set<Core::Elements::Element*> const& neighboring_col_spheres,
    Core::Nodes::Node* node_i) const
{
  check_init();

  CrossLinking::CrosslinkerNode* crosslinker_i =
      dynamic_cast<CrossLinking::CrosslinkerNode*>(node_i);

  if (std::abs(crosslinker_i->get_material()->no_bond_dist_sphere()) < 1.0e-8) return false;

  BeamInteraction::Data::CrosslinkerData* cldata_i = crosslinker_data_[crosslinker_i->lid()].get();

  for (auto const& sphere_iter : neighboring_col_spheres)
  {
    // init position of linker nodes
    Core::LinAlg::Matrix<3, 1> sphere_pos(Core::LinAlg::Initialization::zero);

    // sphere current position
    std::vector<double> sphereeledisp;
    BeamInteraction::Utils::get_current_element_dis(
        discret(), sphere_iter, *beam_interaction_data_state().get_dis_col_np(), sphereeledisp);

    // note: sphere has just one node (with three translational dofs)
    for (unsigned int dim = 0; dim < 3; ++dim)
      sphere_pos(dim) = sphere_iter->nodes()[0]->x()[dim] + sphereeledisp[dim];

    Core::LinAlg::Matrix<3, 1> dist_vec(Core::LinAlg::Initialization::zero);
    dist_vec.update(1.0, sphere_pos, -1.0, cldata_i->get_position());
    const double distance = dist_vec.norm2();

    if (distance < crosslinker_i->get_material()->no_bond_dist_sphere()) return true;
  }

  return false;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::prepare_binding(Core::Nodes::Node* node_i,
    std::set<Core::Elements::Element*> const& neighboring_beams,
    std::map<int, std::shared_ptr<BeamInteraction::Data::BindEventData>>& mybonds,
    std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::BindEventData>>>&
        undecidedbonds,
    std::map<int, std::vector<std::map<int, std::set<int>>>>& intendedbeambonds,
    bool checklinkingprop)
{
  check_init();

  CrossLinking::CrosslinkerNode* crosslinker_i =
      dynamic_cast<CrossLinking::CrosslinkerNode*>(node_i);

  // get precomputed data of crosslinker i
  BeamInteraction::Data::CrosslinkerData* cldata_i = crosslinker_data_[crosslinker_i->lid()].get();

  // -------------------------------------------------------------------------
  // We now check all criteria that need to be passed for a binding event one
  // after the other
  // -------------------------------------------------------------------------
  // 1. criterion: in case crosslinker is double bonded, we can leave here
  if (cldata_i->get_number_of_bonds() == 2) return;

  // loop over all neighboring beam elements in random order (keep in mind
  // we are only looping over row elements)
  std::vector<Core::Elements::Element*> beamvec(neighboring_beams.begin(), neighboring_beams.end());

  // -------------------------------------------------------------------------
  // NOTE: This is crucial for reproducibility to ensure that computation does
  // not depend on pointer addresses (see also comment of class Less)
  // -------------------------------------------------------------------------
  std::sort(beamvec.begin(), beamvec.end(), BeamInteraction::Utils::Less());

  std::vector<int> randorder =
      BeamInteraction::Utils::permutation(static_cast<int>(beamvec.size()));
  for (auto const& randiter : randorder)
  {
    // get neighboring (nb) beam element
    Core::Elements::Element* nbbeam = beamvec[randiter];

    // get pre computed data of current nbbeam
    BeamInteraction::Data::BeamData* beamdata_i = beam_data_[nbbeam->lid()].get();

    if (cldata_i->get_number_of_bonds() == 1)
    {
      int cl_bondedtogid =
          cldata_i->get_b_spots()[get_single_occupied_cl_bspot(cldata_i->get_b_spots())].first;

      // safety check
      if (discret().element_col_map()->lid(cl_bondedtogid) < 0)
        FOUR_C_THROW("Element {} not ghosted on rank {}", cl_bondedtogid, g_state().get_my_rank());

      // 2. criterion:
      // exclude binding of a single bonded crosslinker in close proximity on the
      // same filament (i.e. element cloud of old element binding partner is excluded)
      if (BeamInteraction::Utils::do_beam_elements_share_nodes(
              discret().g_element(cl_bondedtogid), nbbeam))
        continue;
    }

    // loop over all binding spots of current element in random order
    std::vector<int> randbspot =
        BeamInteraction::Utils::permutation(beamdata_i->get_number_of_binding_spots_of_type(
            crosslinker_i->get_material()->linker_type()));

    for (auto const& rbspotiter : randbspot)
    {
      // get local number of binding spot in element
      const int locnbspot = rbspotiter;

      // we are now doing some additional checks if a binding event is feasible
      if (not check_bind_event_criteria(crosslinker_i, nbbeam, cldata_i, beamdata_i, locnbspot,
              intendedbeambonds, checklinkingprop))
        continue;

      // ---------------------------------------------------------------------
      // if we made it this far, we can add this potential binding event to its
      // corresponding map
      // ---------------------------------------------------------------------
      std::shared_ptr<BeamInteraction::Data::BindEventData> bindeventdata =
          std::make_shared<BeamInteraction::Data::BindEventData>();
      // default permission is true, is changed if owner of cl has something against it
      bindeventdata->init(crosslinker_i->id(), nbbeam->id(), locnbspot, g_state().get_my_rank(), 1);

      // in case myrank is owner, we add it to the mybonds map
      if (crosslinker_i->owner() == g_state().get_my_rank())
      {
        mybonds[bindeventdata->get_cl_id()] = bindeventdata;
      }
      else
      {
        // myrank is not owner, we add it to the map of events that need to be
        // communicated to make a decision
        undecidedbonds[crosslinker_i->owner()].push_back(bindeventdata);
      }

      // as we allow only one binding event for each cl in one time step,
      // we are done here, if we made it so far (i.e met criteria 1. - 7.)
      return;
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool BeamInteraction::SubmodelEvaluator::Crosslinking::check_bind_event_criteria(
    CrossLinking::CrosslinkerNode const* const crosslinker_i,
    Core::Elements::Element const* const potbeampartner,
    BeamInteraction::Data::CrosslinkerData* cldata_i,
    BeamInteraction::Data::BeamData const* beamdata_i, int locnbspot,
    std::map<int, std::vector<std::map<int, std::set<int>>>>& intendedbeambonds,
    bool checklinkingprop) const
{
  check_init();

  int const potbeampartnerrowlid = discret().element_row_map()->lid(potbeampartner->id());
  Inpar::BeamInteraction::CrosslinkerType linkertype = crosslinker_i->get_material()->linker_type();

  // check compatibility of crosslinker type and filament type (some linker can only
  // bind to certain filament types)
  if (not check_linker_and_filament_type_compatibility(
          linkertype, dynamic_cast<Discret::Elements::Beam3Base const* const>(potbeampartner)
                          ->get_filament_type()))
    return false;

  // a crosslink is set if and only if it passes the probability check
  // for a binding event to happen
  double plink = 1.0 - exp((-1.0) * crosslinking_params_ptr_->delta_time() *
                           crosslinker_i->get_material()->k_on());

  if (checklinkingprop and (Global::Problem::instance()->random()->uni() > plink)) return false;

  // criterion:
  // first check if binding spot has free bonds left
  if (static_cast<int>(beamdata_i->get_b_spot_status_at(linkertype, locnbspot).size()) >=
      crosslinking_params_ptr_->max_number_of_bonds_per_filament_bspot(linkertype))
    return false;

  // exclude multiple identical crosslinks
  if (not return_false_if_identical_bond_already_exists(
          crosslinker_i, cldata_i, intendedbeambonds, beamdata_i, locnbspot, potbeampartnerrowlid))
    return false;

  /* check RELEVANT distance criterion
   * if free:
   *   distance between crosslinker center and current beam binding spot
   * if singly bound:
   *   distance between already bound bspot of crosslinker and current beam binding spot
   * note: as we set the crosslinker position to coincide with beam bspot position if singly bound,
   *       we can also use cldata_i.clpos in the second case*/

  // get current position and tangent vector of filament axis at free binding spot
  Core::LinAlg::Matrix<3, 1> const& currbbspos =
      beamdata_i->get_b_spot_position(linkertype, locnbspot);

  // minimum and maximum distance at which a double-bond crosslink can be established
  double const linkdistmin = crosslinker_i->get_material()->linking_length() -
                             crosslinker_i->get_material()->linking_length_tolerance();
  double const linkdistmax = crosslinker_i->get_material()->linking_length() +
                             crosslinker_i->get_material()->linking_length_tolerance();

#ifdef FOUR_C_ENABLE_ASSERTIONS
  // safety check
  if (linkdistmax > bin_strategy().bin_size_lower_bound())
    FOUR_C_THROW(
        "The allowed binding distance of linker {} (in case it is single bonded) is"
        "\ngreater than the lower bound for bin size, this could lead to missing a binding event",
        crosslinker_i->id());
#endif

  if ((cldata_i->get_number_of_bonds() == 0 and
          BeamInteraction::Utils::is_distance_out_of_range(
              cldata_i->get_position(), currbbspos, 0.5 * linkdistmin, 0.5 * linkdistmax)) or
      (cldata_i->get_number_of_bonds() == 1 and
          BeamInteraction::Utils::is_distance_out_of_range(
              cldata_i->get_position(), currbbspos, linkdistmin, linkdistmax)))
    return false;

  // orientation of centerline tangent vectors at binding spots
  // a crosslink (double-bonded crosslinker) will only be established if the
  // enclosed angle is in the specified range
  double const linkanglemin = crosslinker_i->get_material()->linking_angle() -
                              crosslinker_i->get_material()->linking_angle_tolerance();
  double const linkanglemax = crosslinker_i->get_material()->linking_angle() +
                              crosslinker_i->get_material()->linking_angle_tolerance();

  // if crosslinker is singly bound, we fetch the orientation vector
  Core::LinAlg::Matrix<3, 1> occ_bindingspot_beam_tangent(Core::LinAlg::Initialization::zero);
  if (cldata_i->get_number_of_bonds() == 1)
    get_occupied_cl_b_spot_beam_tangent(
        crosslinker_i, cldata_i, occ_bindingspot_beam_tangent, crosslinker_i->id());

  // note: we use first base vector instead of tangent vector here
  Core::LinAlg::Matrix<3, 1> curr_bindingspot_beam_tangent(Core::LinAlg::Initialization::zero);
  for (unsigned int idim = 0; idim < 3; ++idim)
    curr_bindingspot_beam_tangent(idim) =
        beamdata_i->get_b_spot_triad(linkertype, locnbspot)(idim, 0);

  if (cldata_i->get_number_of_bonds() == 1 and
      BeamInteraction::Utils::is_enclosed_angle_out_of_range(
          occ_bindingspot_beam_tangent, curr_bindingspot_beam_tangent, linkanglemin, linkanglemax))
    return false;

  // check if current beam binding spot yet intended to bind this timestep
  // by a crosslinker that came before in this random order
  if (static_cast<int>(intendedbeambonds.at(linkertype)[potbeampartnerrowlid][locnbspot].size() +
                       beamdata_i->get_b_spot_status_at(linkertype, locnbspot).size()) >=
      crosslinking_params_ptr_->max_number_of_bonds_per_filament_bspot(
          crosslinker_i->get_material()->linker_type()))
  {
    /* note: it is possible that the binding event that rejects the current one is rejected itself
     * later during communication with other procs and therefore the current one could be
     * valid. Just neglecting this here is a slight inconsistency, but should be ok as such an
     * coincidence is extremely rare in a simulation with realistic proportion of crosslinker
     * to beam binding spots. Additionally missing one event would not change any physics.
     * (Could be cured with additional communication)
     */
    if (Core::Communication::num_mpi_ranks(discret().get_comm()) > 1)
      Core::IO::cout(Core::IO::verbose)
          << " Warning: There is a minimal chance of missing a regular binding event on "
             "rank "
          << g_state().get_my_rank() << Core::IO::endl;
    return false;
  }
  else
  {
    intendedbeambonds.at(linkertype)[potbeampartnerrowlid][locnbspot].insert(crosslinker_i->id());
  }

  // bind event can happen
  // (rejection afterwards only possible in case of parallel simulation)
  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool BeamInteraction::SubmodelEvaluator::Crosslinking::
    return_false_if_identical_bond_already_exists(
        CrossLinking::CrosslinkerNode const* const crosslinker_i,
        BeamInteraction::Data::CrosslinkerData* cldata_i,
        std::map<int, std::vector<std::map<int, std::set<int>>>>& intendedbeambonds,
        BeamInteraction::Data::BeamData const* beamdata_i, int locnbspot,
        int potbeampartnerrowlid) const
{
  Inpar::BeamInteraction::CrosslinkerType linkertype = crosslinker_i->get_material()->linker_type();

  if (not(cldata_i->get_number_of_bonds() == 1 and
          crosslinking_params_ptr_->max_number_of_bonds_per_filament_bspot(linkertype) > 1))
    return true;

  // get element and gid of beam element to which current linker is already bonded to
  int occbspotid = get_single_occupied_cl_bspot(cldata_i->get_b_spots());
  int const elegid = cldata_i->get_b_spots()[occbspotid].first;
  int const locbspotnum = cldata_i->get_b_spots()[occbspotid].second;

  int const elecollid = discret().element_col_map()->lid(elegid);

#ifdef FOUR_C_ENABLE_ASSERTIONS
  // safety check
  if (elecollid < 0)
    FOUR_C_THROW("element with gid {} not on proc {}", elegid, g_state().get_my_rank());
#endif

  // loop over crosslinker that are already bonded to binding spot to which current linker is bonded
  // to
  for (auto const& iter : beam_data_[elecollid]->get_b_spot_status_at(linkertype, locbspotnum))
  {
    // this is needed in case a binding event was allowed in this time step in opposite direction
    if (intendedbeambonds.at(linkertype)[potbeampartnerrowlid][locnbspot].find(iter) !=
        intendedbeambonds.at(linkertype)[potbeampartnerrowlid][locnbspot].end())
      return false;
  }

  // loop over crosslinker that are already bonded to potential new binding spot
  for (auto const iter : beamdata_i->get_b_spot_status_at(linkertype, locnbspot))
  {
    Core::Nodes::Node* bonded_crosslinker_i = bin_discret().g_node(iter);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    // safety check
    if (bonded_crosslinker_i == nullptr)
      FOUR_C_THROW(" Linker with gid {} not on rank {}", iter, g_state().get_my_rank());
#endif

    BeamInteraction::Data::CrosslinkerData const* bondedcl_data_i =
        crosslinker_data_[bonded_crosslinker_i->lid()].get();

#ifdef FOUR_C_ENABLE_ASSERTIONS
    // safety check
    if (bonded_crosslinker_i == nullptr)
      FOUR_C_THROW("Data for crosslinker {} not there on rank {}", bonded_crosslinker_i->id(),
          g_state().get_my_rank());
#endif

    if (bondedcl_data_i->get_number_of_bonds() == 1)
    {
      // this is needed in case a binding event was allowed in this time step in opposite direction
      int const elelid = discret().element_row_map()->lid(elegid);
      if (elelid != -1 and
          intendedbeambonds.at(linkertype)[elelid][locbspotnum].find(bonded_crosslinker_i->id()) !=
              intendedbeambonds.at(linkertype)[elelid][locbspotnum].end())
        return false;
    }
    else if (bondedcl_data_i->get_number_of_bonds() == 2)
    {
      // if intended bond between two filament binding spots already exists, reject current intended
      // bond
      if ((bondedcl_data_i->get_b_spots()[0].first == elegid and
              bondedcl_data_i->get_b_spots()[0].second == locbspotnum) or
          (bondedcl_data_i->get_b_spots()[1].first == elegid and
              bondedcl_data_i->get_b_spots()[1].second == locbspotnum))
        return false;
    }
    else
    {
      FOUR_C_THROW(
          " unrealistic number of bonds ({}) for crosslinker (gid {}) at this point. Beam {} local "
          "{} ",
          bondedcl_data_i->get_number_of_bonds(), bonded_crosslinker_i->owner(), elegid,
          locbspotnum);
    }
  }

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool BeamInteraction::SubmodelEvaluator::Crosslinking::check_linker_and_filament_type_compatibility(
    Inpar::BeamInteraction::CrosslinkerType linkertype,
    Inpar::BeamInteraction::FilamentType filamenttype) const
{
  switch (linkertype)
  {
    case Inpar::BeamInteraction::linkertype_arbitrary:
    {
      // no check of filament type necessary
      return true;
      break;
    }
    case Inpar::BeamInteraction::linkertype_actin:
    {
      if (filamenttype == Inpar::BeamInteraction::filetype_actin)
        return true;
      else
        return false;
      break;
    }
    case Inpar::BeamInteraction::linkertype_collagen:
    {
      if (filamenttype == Inpar::BeamInteraction::filetype_collagen)
        return true;
      else
        return false;
      break;
    }
    default:
    {
      FOUR_C_THROW("Unknown linker type.");
    }
  }

  // default false
  return false;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::get_occupied_cl_b_spot_beam_tangent(
    CrossLinking::CrosslinkerNode const* const crosslinker_i,
    BeamInteraction::Data::CrosslinkerData* cldata_i,
    Core::LinAlg::Matrix<3, 1>& occ_bindingspot_beam_tangent, int clgid) const
{
  check_init_setup();

  int occbspotid = get_single_occupied_cl_bspot(cldata_i->get_b_spots());

  const int locbspotnum = cldata_i->get_b_spots()[occbspotid].second;
  const int elegid = cldata_i->get_b_spots()[occbspotid].first;
  const int elecollid = discret().element_col_map()->lid(elegid);

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (elecollid < 0)
    FOUR_C_THROW(" Element with gid {} bonded to cl {} on rank {} not even ghosted", elegid, clgid,
        g_state().get_my_rank());
#endif

  // note: we use first base vector instead of tangent vector here
  for (unsigned int idim = 0; idim < 3; ++idim)
    occ_bindingspot_beam_tangent(idim) = beam_data_[elecollid]->get_b_spot_triad(
        crosslinker_i->get_material()->linker_type(), locbspotnum)(idim, 0);
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::manage_binding_in_parallel(
    std::map<int, std::shared_ptr<BeamInteraction::Data::BindEventData>>& mybonds,
    std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::BindEventData>>>&
        undecidedbonds,
    std::map<int, std::shared_ptr<BeamInteraction::Data::BindEventData>>& myelebonds) const
{
  check_init();

  TEUCHOS_FUNC_TIME_MONITOR(
      "BeamInteraction::SubmodelEvaluator::Crosslinking::manage_binding_in_parallel");

  // -------------------------------------------------------------------------
  // 1) each procs makes its requests and receives the request of other procs
  // -------------------------------------------------------------------------
  // store requested cl and its data
  std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::BindEventData>>> requestedcl;
  communicate_undecided_bonds(undecidedbonds, requestedcl);

  // -------------------------------------------------------------------------
  // 2) now myrank needs to decide which proc is allowed to set the requested
  //    link
  // -------------------------------------------------------------------------
  std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::BindEventData>>> decidedbonds;
  decide_binding_in_parallel(requestedcl, mybonds, decidedbonds);

  // -------------------------------------------------------------------------
  // 3) communicate the binding decisions made on myrank, receive decisions
  //    made for its own requests and create colbondmap accordingly
  // -------------------------------------------------------------------------
  communicate_decided_bonds(decidedbonds, myelebonds);
  decidedbonds.clear();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int BeamInteraction::SubmodelEvaluator::Crosslinking::
    update_my_crosslinker_and_element_binding_states(
        std::map<int, std::shared_ptr<BeamInteraction::Data::BindEventData>>& mybonds,
        std::map<int, std::shared_ptr<BeamInteraction::Data::BindEventData>>& myelebonds)
{
  check_init();

  // map key is crosslinker gid to be able to uniquely address one entry over all procs
  std::map<int, NewDoubleBonds> mynewdbondcl;

  // myrank owner of crosslinker and most elements
  update_my_crosslinker_binding_states(mybonds, mynewdbondcl);

  // myrank only owner of current binding partner ele
  update_my_element_binding_states(myelebonds);

  // setup new double bonds and insert them in doublebondcl_
  create_new_double_bonded_crosslinker_element_pairs(mynewdbondcl);

  return static_cast<int>(mynewdbondcl.size());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::update_my_crosslinker_binding_states(
    std::map<int, std::shared_ptr<BeamInteraction::Data::BindEventData>> const& mybonds,
    std::map<int, NewDoubleBonds>& mynewdbondcl)
{
  check_init();

  for (auto const& cliter : mybonds)
  {
    // get binding event data
    BeamInteraction::Data::BindEventData* binevdata = cliter.second.get();

#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (binevdata->get_permission() != 1)
      FOUR_C_THROW(
          " Rank {} wants to bind crosslinker {} without permission, "
          " something went wrong",
          g_state().get_my_rank(), cliter.first);
#endif

    // get current linker and beam data
    const int clcollid = bin_discret_ptr()->node_col_map()->lid(cliter.first);
    const int colelelid = discret_ptr()->element_col_map()->lid(binevdata->get_ele_id());

#ifdef FOUR_C_ENABLE_ASSERTIONS
    // safety checks
    if (clcollid < 0) FOUR_C_THROW("Crosslinker not even ghosted, should be owned here.");
    if (colelelid < 0) FOUR_C_THROW("Element with gid {} not ghosted.", binevdata->get_ele_id());
#endif
    CrossLinking::CrosslinkerNode* crosslinker_i =
        dynamic_cast<CrossLinking::CrosslinkerNode*>(bin_discret_ptr()->l_col_node(clcollid));

    // get crosslinker data
    BeamInteraction::Data::CrosslinkerData* cldata_i = crosslinker_data_[clcollid].get();

    Core::Elements::Element* beamele_i = discret_ptr()->l_col_element(colelelid);
    BeamInteraction::Data::BeamData* beamdata_i = beam_data_[colelelid].get();

#ifdef FOUR_C_ENABLE_ASSERTIONS
    // safety checks
    if (cliter.first != binevdata->get_cl_id())
      FOUR_C_THROW("Map key does not match crosslinker gid of current binding event.");

    if (crosslinker_i->owner() != g_state().get_my_rank())
      FOUR_C_THROW("Only row owner of crosslinker is changing its status");

    if (colelelid < 0)
      FOUR_C_THROW(
          "Binding element partner of current row crosslinker is not ghosted, "
          "this must be the case though.");
#endif

    // -------------------------------------------------------------------------
    // different treatment according to number of bonds crosslinker had before
    // this binding event
    // -------------------------------------------------------------------------
    switch (cldata_i->get_number_of_bonds())
    {
      case 0:
      {
        // -----------------------------------------------------------------
        // update crosslinker status
        // -----------------------------------------------------------------
        // store gid and bspot local number of this element, first binding spot
        // always bonded first
        cldata_i->set_bspot(
            0, std::make_pair(binevdata->get_ele_id(), binevdata->get_b_spot_loc_n()));

        // update number of bonds
        cldata_i->set_number_of_bonds(1);

        // update position
        std::vector<double> newpos(3, 0.0);
        for (int dim = 0; dim < 3; ++dim)
          newpos[dim] = beamdata_i->get_b_spot_position(
              crosslinker_i->get_material()->linker_type(), binevdata->get_b_spot_loc_n())(dim);
        crosslinker_i->set_pos(newpos);
        cldata_i->set_position(beamdata_i->get_b_spot_position(
            crosslinker_i->get_material()->linker_type(), binevdata->get_b_spot_loc_n()));

        // -----------------------------------------------------------------
        // update beam status
        // -----------------------------------------------------------------
        // store crosslinker gid in status of beam binding spot if myrank
        // is owner of beam
        if (beamele_i->owner() == g_state().get_my_rank())
          beamdata_i->add_bond_to_binding_spot(crosslinker_i->get_material()->linker_type(),
              binevdata->get_b_spot_loc_n(), binevdata->get_cl_id());

#ifdef FOUR_C_ENABLE_ASSERTIONS
        // safety check
        if (not(cldata_i->get_b_spots()[1].first < 0))
          FOUR_C_THROW("Numbond does not fit to clbspot vector.");
#endif

        break;
      }
      case 1:
      {
        // get clbspot that is currently bonded
        int occbspotid = get_single_occupied_cl_bspot(cldata_i->get_b_spots());
        int freebspotid = 1;
        if (occbspotid == 1) freebspotid = 0;

        // -----------------------------------------------------------------
        // update crosslinker status
        // -----------------------------------------------------------------
        // store gid and bspot local number of this element
        cldata_i->set_bspot(
            freebspotid, std::make_pair(binevdata->get_ele_id(), binevdata->get_b_spot_loc_n()));

        // update number of bonds
        cldata_i->set_number_of_bonds(2);

        // update position
        Core::LinAlg::Matrix<3, 1> clpos(cldata_i->get_position());
        set_position_of_double_bonded_crosslinker_pb_cconsistent(clpos,
            beamdata_i->get_b_spot_position(crosslinker_i->get_material()->linker_type(),
                cldata_i->get_b_spots()[freebspotid].second),
            cldata_i->get_position());

        std::vector<double> newpos(3, 0.0);
        for (int dim = 0; dim < 3; ++dim) newpos[dim] = clpos(dim);
        crosslinker_i->set_pos(newpos);

        cldata_i->set_position(clpos);

        // create double bond cl data
        int occ_colelelid =
            discret_ptr()->element_col_map()->lid(cldata_i->get_b_spots()[occbspotid].first);
        NewDoubleBonds dbondcl;
        dbondcl.id = binevdata->get_cl_id();
        if (cldata_i->get_b_spots()[freebspotid].first > cldata_i->get_b_spots()[occbspotid].first)
        {
          dbondcl.eleids.push_back(cldata_i->get_b_spots()[freebspotid]);
          dbondcl.eleids.push_back(cldata_i->get_b_spots()[occbspotid]);
          dbondcl.bspotposs.push_back(beam_data_[colelelid]->get_b_spot_position(
              crosslinker_i->get_material()->linker_type(),
              cldata_i->get_b_spots()[freebspotid].second));
          dbondcl.bspotposs.push_back(beam_data_[occ_colelelid]->get_b_spot_position(
              crosslinker_i->get_material()->linker_type(),
              cldata_i->get_b_spots()[occbspotid].second));
          dbondcl.bspottriads.push_back(
              beam_data_[colelelid]->get_b_spot_triad(crosslinker_i->get_material()->linker_type(),
                  cldata_i->get_b_spots()[freebspotid].second));
          dbondcl.bspottriads.push_back(beam_data_[occ_colelelid]->get_b_spot_triad(
              crosslinker_i->get_material()->linker_type(),
              cldata_i->get_b_spots()[occbspotid].second));
        }
        else
        {
          dbondcl.eleids.push_back(cldata_i->get_b_spots()[occbspotid]);
          dbondcl.eleids.push_back(cldata_i->get_b_spots()[freebspotid]);
          dbondcl.bspotposs.push_back(beam_data_[occ_colelelid]->get_b_spot_position(
              crosslinker_i->get_material()->linker_type(),
              cldata_i->get_b_spots()[occbspotid].second));
          dbondcl.bspotposs.push_back(beam_data_[colelelid]->get_b_spot_position(
              crosslinker_i->get_material()->linker_type(),
              cldata_i->get_b_spots()[freebspotid].second));
          dbondcl.bspottriads.push_back(beam_data_[occ_colelelid]->get_b_spot_triad(
              crosslinker_i->get_material()->linker_type(),
              cldata_i->get_b_spots()[occbspotid].second));
          dbondcl.bspottriads.push_back(
              beam_data_[colelelid]->get_b_spot_triad(crosslinker_i->get_material()->linker_type(),
                  cldata_i->get_b_spots()[freebspotid].second));
        }

        // insert pair in mypairs
        mynewdbondcl[dbondcl.id] = dbondcl;

        // first check if myrank is owner of element of current binding event
        // (additionally to being owner of cl)
        if (beamele_i->owner() == g_state().get_my_rank())
          beamdata_i->add_bond_to_binding_spot(crosslinker_i->get_material()->linker_type(),
              binevdata->get_b_spot_loc_n(), binevdata->get_cl_id());

        break;
      }
      default:
      {
        FOUR_C_THROW(
            "You should not be here, crosslinker has unrealistic number "
            "{} of bonds.",
            cldata_i->get_number_of_bonds());
      }
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::update_my_element_binding_states(
    std::map<int, std::shared_ptr<BeamInteraction::Data::BindEventData>> const& myelebonds)
{
  check_init();

  /*
   * 1| 2__|2  or  1| 3__|2  or  1| 2__|1
   * 1|    |2      1|    |2      1|    |1
   * legend: | = beam; __= cl; 2,3 = owner; 1 = myrank
   */
  // loop through all binding events
  for (auto const& cliter : myelebonds)
  {
    // get binding event data
    BeamInteraction::Data::BindEventData* binevdata = cliter.second.get();

    // get linker data and beam data
    CrossLinking::CrosslinkerNode* linker =
        dynamic_cast<CrossLinking::CrosslinkerNode*>(bin_discret_ptr()->g_node(cliter.first));
    int const colelelid = discret_ptr()->element_col_map()->lid(binevdata->get_ele_id());

#ifdef FOUR_C_ENABLE_ASSERTIONS
    // safety checks
    if (linker == nullptr)
      FOUR_C_THROW("Crosslinker needs to be ghosted, but this isn't the case.");
    if (colelelid < 0)
      FOUR_C_THROW("element with gid {} not ghosted on proc {}", binevdata->get_ele_id(),
          g_state().get_my_rank());
#endif

    // linker
    int const clcollid = linker->lid();
    BeamInteraction::Data::CrosslinkerData* cldata_i = crosslinker_data_[clcollid].get();

    BeamInteraction::Data::BeamData* beamdata_i = beam_data_[colelelid].get();

#ifdef FOUR_C_ENABLE_ASSERTIONS
    // safety checks
    if (discret_ptr()->l_col_element(colelelid)->owner() != g_state().get_my_rank())
      FOUR_C_THROW("Only row owner of element is allowed to change its status");
    if (linker->owner() == g_state().get_my_rank())
      FOUR_C_THROW("myrank should not be owner of this crosslinker");
#endif

    // different treatment according to number of bonds crosslinker had before
    // this binding event
    switch (cldata_i->get_number_of_bonds())
    {
      case 0:
      {
        beamdata_i->add_bond_to_binding_spot(linker->get_material()->linker_type(),
            binevdata->get_b_spot_loc_n(), binevdata->get_cl_id());
        break;
      }
      case 1:
      {
        beamdata_i->add_bond_to_binding_spot(linker->get_material()->linker_type(),
            binevdata->get_b_spot_loc_n(), binevdata->get_cl_id());
        break;
      }
      default:
      {
        FOUR_C_THROW(
            "You should not be here, crosslinker has unrealistic number "
            "{} of bonds.",
            cldata_i->get_number_of_bonds());
      }
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::
    create_new_double_bonded_crosslinker_element_pairs(
        std::map<int, NewDoubleBonds> const& mynewdbondcl)
{
  check_init();

  for (auto const& iter : mynewdbondcl)
  {
    NewDoubleBonds const& newdoublebond_i = iter.second;
    CrossLinking::CrosslinkerNode* cl_node =
        dynamic_cast<CrossLinking::CrosslinkerNode*>(bin_discret_ptr()->g_node(iter.first));

    // create and initialize objects of beam-to-beam connections
    // Todo move this inside the create routines (or one create routine in BeamLink class)
    std::shared_ptr<BeamInteraction::BeamLink> linkelepairptr;
    if (cl_node->get_material()->joint_type() == Inpar::BeamInteraction::beam3r_line2_rigid)
      linkelepairptr = BeamInteraction::BeamLinkRigidJointed::create();
    else if (cl_node->get_material()->joint_type() == Inpar::BeamInteraction::beam3r_line2_pin or
             cl_node->get_material()->joint_type() == Inpar::BeamInteraction::truss)
      linkelepairptr =
          BeamInteraction::BeamLinkPinJointed::create(cl_node->get_material()->joint_type());

    // finally initialize and setup object
    linkelepairptr->init(iter.first, newdoublebond_i.eleids, newdoublebond_i.bspotposs,
        newdoublebond_i.bspottriads, cl_node->get_material()->linker_type(),
        g_state().get_time_np());
    linkelepairptr->setup(cl_node->get_material()->beam_elast_hyper_mat_num());

    // add to my double bonds
    doublebondcl_[linkelepairptr->id()] = linkelepairptr;

#ifdef FOUR_C_ENABLE_ASSERTIONS
    // safety check
    Core::Nodes::Node* crosslinker_i = bin_discret_ptr()->g_node(linkelepairptr->id());

    // safety check
    BeamInteraction::Data::CrosslinkerData const* cldata_i =
        crosslinker_data_[crosslinker_i->lid()].get();

    if (cldata_i->get_number_of_bonds() != 2)
      FOUR_C_THROW("Setup: Cl with gid {} Owner {} on myrank {} and numbonds {}",
          linkelepairptr->id(), crosslinker_i->owner(), g_state_ptr()->get_my_rank(),
          cldata_i->get_number_of_bonds());
#endif
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::
    double_bind_crosslinker_in_bins_and_neighborhood(std::set<int> const& bingids)
{
  check_init();

  unsigned int maxbonds = 2;

  for (unsigned int bond_i = 0; bond_i < maxbonds; ++bond_i)
  {
    // intended bonds of row crosslinker on myrank (key is clgid)
    std::map<int, std::shared_ptr<BeamInteraction::Data::BindEventData>> mybonds;
    // intended bond col crosslinker to row element (key is owner of crosslinker != myrank)
    std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::BindEventData>>>
        undecidedbonds;

    // store bins that have already been examined
    std::vector<int> examinedbins(bin_discret_ptr()->num_my_col_elements(), 0);
    // this variable is used to check if a beam binding spot is linked twice on
    // myrank during a time step
    // ( first key is linkertype, second key is locbspotid, set holds gids of bonded crosslinker)
    std::map<int, std::vector<std::map<int, std::set<int>>>> intendedbeambonds;
    for (unsigned int i = 0; i < crosslinking_params_ptr_->linker_types().size(); ++i)
      intendedbeambonds[crosslinking_params_ptr_->linker_types()[i]].resize(
          discret_ptr()->num_my_row_elements());

    for (auto const& b_iter : bingids)
    {
      // get neighboring bins
      std::vector<int> nb_binIds;
      nb_binIds.reserve(27);
      // do not check on existence here -> shifted to GetBinContent
      bin_strategy_ptr()->get_neighbor_and_own_bin_ids(b_iter, nb_binIds);

      for (auto const& nb_iter : nb_binIds)
      {
        // check on existence of bin on this proc
        if (not bin_discret_ptr()->have_global_element(nb_iter)) continue;

        Core::Elements::Element* currentbin = bin_discret_ptr()->g_element(nb_iter);

        // if a bin has already been examined --> continue with next crosslinker
        if (examinedbins[currentbin->lid()]) continue;
        // else: bin is examined for the first time --> new entry in examinedbins_
        else
          examinedbins[currentbin->lid()] = 1;

        find_potential_binding_events_in_bin_and_neighborhood(
            currentbin, mybonds, undecidedbonds, intendedbeambonds, false);
      }
    }

    // bind events where myrank only owns the elements, cl are taken care
    // of by their owner (key is clgid)
    std::map<int, std::shared_ptr<BeamInteraction::Data::BindEventData>> myelebonds;

    // now each row owner of a linker gets requests, makes a random decision and
    // informs back its requesters
    manage_binding_in_parallel(mybonds, undecidedbonds, myelebonds);

    // actual update of binding states is done here
    update_my_crosslinker_and_element_binding_states(mybonds, myelebonds);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int BeamInteraction::SubmodelEvaluator::Crosslinking::un_bind_crosslinker()
{
  check_init();

  Global::Problem::instance()->random()->set_rand_range(0.0, 1.0);

  // data containing information about elements that need to be updated on
  // procs != myrank
  std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::UnBindEventData>>>
      sendunbindevents;
  sendunbindevents.clear();
  // elements that need to be updated on myrank
  std::vector<std::shared_ptr<BeamInteraction::Data::UnBindEventData>> myrankunbindevents;
  myrankunbindevents.clear();
  int num_db_dissolved = 0;

  // loop over all row linker (in random order) and dissolve bond if probability
  // criterion is met
  /* note: we loop over all row crosslinker, i.e. myrank needs to update all
   * crosslinker information. As it possible that a row crosslinker is linked
   * to col element, we potentially need to communicate if such an element
   * needs to be updated*/
  const int numrowcl = bin_discret_ptr()->num_my_row_nodes();
  std::vector<int> rorderrowcl = BeamInteraction::Utils::permutation(numrowcl);
  for (auto const& rowcli : rorderrowcl)
  {
    CrossLinking::CrosslinkerNode* linker =
        dynamic_cast<CrossLinking::CrosslinkerNode*>(bin_discret_ptr()->l_row_node(rowcli));

    // only consider unbinding in case off rate is unequal zero
    if (linker->get_material()->k_off() < 1e-08) continue;

    const int clcollid = linker->lid();
    BeamInteraction::Data::CrosslinkerData* cldata_i = crosslinker_data_[clcollid].get();

    // probability with which a crosslink breaks up in the current time step
    double p_unlink = 1.0 - exp((-1.0) * crosslinking_params_ptr_->delta_time() *
                                linker->get_material()->k_off());

    // different treatment according to number of bonds of a crosslinker
    switch (cldata_i->get_number_of_bonds())
    {
      case 0:
      {
        // nothing to do here
        break;
      }
      case 1:
      {
        // if probability criterion is not met, we are done here
        if (Global::Problem::instance()->random()->uni() > p_unlink) break;

        // dissolve bond and update states
        dissolve_bond(linker, get_single_occupied_cl_bspot(cldata_i->get_b_spots()), 1,
            sendunbindevents, myrankunbindevents);

        break;
      }
      case 2:
      {
        // calc unbind probability in case of force dependent off rate
        std::vector<double> p_unlink_db(2, 0.0);
        if (abs(linker->get_material()->delta_bell_eq()) > 1.0e-8)
          calc_bells_force_dependent_unbind_probability(
              linker, *doublebondcl_[linker->id()], p_unlink_db);
        else
          p_unlink_db[0] = p_unlink_db[1] = p_unlink;

        // loop through crosslinker bonds in random order
        std::vector<int> ro = BeamInteraction::Utils::permutation(cldata_i->get_number_of_bonds());
        for (auto const& clbspotiter : ro)
        {
          // if probability criterion isn't met, go to next spot
          if (Global::Problem::instance()->random()->uni() > p_unlink_db[clbspotiter]) continue;

          // dissolve bond and update states
          dissolve_bond(linker, clbspotiter, 2, sendunbindevents, myrankunbindevents);
          ++num_db_dissolved;

          // we only want to dissolve one bond per timestep, therefore we go to
          // next crosslinker if we made it so far (i.e. a bond got dissolved)
          break;
        }

        break;
      }
      default:
      {
        FOUR_C_THROW(
            "Unrealistic number {} of bonds for a crosslinker.", cldata_i->get_number_of_bonds());
      }
    }
  }

  // communicate which elements need to be updated on rank != myrank
  communicate_crosslinker_unbinding(sendunbindevents, myrankunbindevents);

  // update binding status of beam binding partners on myrank
  update_beam_binding_status_after_unbinding(myrankunbindevents);

  return num_db_dissolved;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::
    calc_bells_force_dependent_unbind_probability(CrossLinking::CrosslinkerNode* linker,
        BeamInteraction::BeamLink& elepairptr, std::vector<double>& punlinkforcedependent) const
{
  check_init_setup();

  /* characteristic vond length (=nodereldis)
   * from B. Gui and W. Guilford: Mechanics of actomyosin bonds in different nucleotide states are
   * tuned to muscle contraction Fig 2: slip pathway, ADP, delta = 0.0004; Note: delta < 0 -> catch
   * bond, delta > 0 -> bond-weakening see Kai Mueller Dis p. 67/68 */
  double const delta = linker->get_material()->delta_bell_eq();
  double const kt = crosslinking_params_ptr_->kt();
  double const koff = linker->get_material()->k_off();
  double const dt = crosslinking_params_ptr_->delta_time();

  // safety check
  if (kt < 1e-08)
    FOUR_C_THROW(" Thermal energy (KT) set to zero, although you are about to divide by it. ");

  // force and moment exerted on the two binding sites of crosslinker with clgid
  std::vector<Core::LinAlg::SerialDenseVector> bspotforce(2, Core::LinAlg::SerialDenseVector(6));
  elepairptr.evaluate_force(bspotforce[0], bspotforce[1]);

  // check if linker is stretched -> sgn+ or compressed -> sgn- by checking orientation of force
  // vector note: this works only if there are no other forces (like inertia, stochastic, damping)
  // acting on the cl
  Core::LinAlg::Matrix<3, 1> dist_vec(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> bspotforceone(Core::LinAlg::Initialization::zero);
  dist_vec.update(1.0, elepairptr.get_bind_spot_pos1(), -1.0, elepairptr.get_bind_spot_pos2());
  for (unsigned int j = 0; j < 3; ++j) bspotforceone(j) = bspotforce[0](j);
  double sgn = (dist_vec.dot(bspotforceone) < 0.0) ? -1.0 : 1.0;

  /* note: you have a sign criterion that is dependent on the axial strain, but the force you are
   * using also contains shear parts. This means in cases of axial strains near 0 and (large) shear
   * forces you are doing something strange (e.g. jumping between behaviour). Think about a two or
   * three dimensional calculation of the new koff, considering shear and axial forces independently
   */
  //  if ( global_state().get_my_rank() == 0 )
  //    std::cout << "Warning: in cases of high shear forces and low axial strain you might "
  //                 "be doing something strange using the force dependent off rate ..." <<
  //                 std::endl;


  // NOTE if you want to add a different force dependent unbinding law: The forces of your linker
  // that were newly set this time step are zero as they are set stress free. Dissolving them here
  // because of a zero force does not make sense. This needs to be considered.


  // calculate new off rate
  std::vector<double> clbspotforcenorm(2, 0.0);
  std::vector<double> forcedependentkoff(2, 0.0);
  for (unsigned int i = 0; i < 2; ++i)
  {
    // currently, only forces (not moments) considered
    bspotforce[i].reshape(3, 1);
    clbspotforcenorm[i] = Core::LinAlg::norm2(bspotforce[i]);

    // adjusted off-rate according to Bell's equation (Howard, eq 5.10, p.89)
    forcedependentkoff[i] = koff * exp(sgn * clbspotforcenorm[i] * delta / kt);

    // get respective force dependent unbind probability for each cl binding spot
    punlinkforcedependent[i] = 1.0 - exp((-1.0) * dt * forcedependentkoff[i]);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::update_beam_binding_status_after_unbinding(
    std::vector<std::shared_ptr<BeamInteraction::Data::UnBindEventData>> const& unbindevent)
{
  check_init();

  // loop through all unbinding events on myrank
  for (auto const& iter : unbindevent)
  {
    // get data
    const int elegidtoupdate = iter->get_ele_toupdate().first;
    const int bspotlocn = iter->get_ele_toupdate().second;
    const int colelelid = discret().element_col_map()->lid(elegidtoupdate);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    // safety check
    if (discret().element_row_map()->lid(elegidtoupdate) < 0)
      FOUR_C_THROW(
          "element with gid {} not owned by proc {}", elegidtoupdate, g_state().get_my_rank());
#endif

    // erase current bond
    beam_data_[colelelid]->erase_bond_from_binding_spot(
        iter->get_linker_type(), bspotlocn, iter->get_cl_id());
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::update_my_double_bonds_after_redistribution()
{
  check_init();

  std::map<int, std::vector<std::shared_ptr<BeamInteraction::BeamLink>>> dbcltosend;
  std::set<int> dbtoerase;

  // loop over all double bonds on myrank
  std::map<int, std::shared_ptr<BeamInteraction::BeamLink>>::iterator iter;
  for (iter = doublebondcl_.begin(); iter != doublebondcl_.end(); ++iter)
  {
    const int clgid = iter->first;

    // safety check
    if (bin_discret_ptr()->node_col_map()->lid(clgid) < 0)
      FOUR_C_THROW(
          "Crosslinker {} moved further than the bin length in one time step on rank {}, "
          "this is not allowed (maybe increase cutoff radius). ",
          clgid, g_state().get_my_rank());

    Core::Nodes::Node* doublebondedcl_i = bin_discret_ptr()->g_node(clgid);

    // check ownership
    int owner = doublebondedcl_i->owner();
    if (owner != g_state().get_my_rank())
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (not doublebondcl_.count(clgid))
        FOUR_C_THROW("willing to delete double bond {} which is not existing", clgid);
#endif
      dbcltosend[owner].push_back(iter->second);
      dbtoerase.insert(clgid);
    }
  }

  for (auto const& i : dbtoerase) doublebondcl_.erase(i);

  // add new double bonds
  communicate_beam_link_after_redistribution(dbcltosend);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::update_my_double_bonds_remote_id_list()
{
  check_init();

  // loop over all double bonded crosslinker that were read during restart and
  // get the ones that do not belong to myrank
  std::set<int> notonmyrank;
  std::map<int, std::shared_ptr<BeamInteraction::BeamLink>>::iterator iter;
  for (iter = doublebondcl_.begin(); iter != doublebondcl_.end(); ++iter)
  {
    // double bonded crosslinker gid
    const int clgid = iter->first;

    // not owned
    if (bin_discret_ptr()->node_row_map()->lid(clgid) < 0) notonmyrank.insert(clgid);
  }

  int const size = static_cast<int>(notonmyrank.size());
  std::vector<int> unique_clgidlist(notonmyrank.begin(), notonmyrank.end());
  std::vector<int> unique_pidlist(size);

  // find new host procs for double bonded crosslinker by communication
  int err = bin_discret_ptr()->node_row_map()->remote_id_list(
      size, unique_clgidlist.data(), unique_pidlist.data(), nullptr);
  if (err < 0) FOUR_C_THROW("Core::LinAlg::Map::RemoteIDList returned err={}", err);

  std::map<int, std::vector<std::shared_ptr<BeamInteraction::BeamLink>>> dbcltosend;
  for (unsigned int i = 0; i < static_cast<unsigned int>(unique_clgidlist.size()); ++i)
    dbcltosend[unique_pidlist[i]].push_back(doublebondcl_[unique_clgidlist[i]]);

  // update myrank's map
  for (auto const& i : notonmyrank) doublebondcl_.erase(i);

  // send and receive double bonds
  communicate_beam_link_after_redistribution(dbcltosend);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::unbind_crosslinker_in_bins_and_neighborhood(
    std::set<int> const& bingids)
{
  check_init();

  unbind_crosslinker_in_bins_and_neighborhood(bingids, false);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::unbind_crosslinker_in_bins_and_neighborhood(
    std::set<int> const& bingids, bool doubleunbind)
{
  check_init_setup();

  std::set<int> binsonmyrank;
  determine_responsilbe_procs_for_forced_crosslinker_unbinding(bingids, binsonmyrank);

  // data containing information about elements that need to be updated on
  // procs != myrank
  std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::UnBindEventData>>>
      sendunbindevents;
  // elements that need to be updated on myrank
  std::vector<std::shared_ptr<BeamInteraction::Data::UnBindEventData>> myrankunbindevents;

  for (auto const& nb_iter : binsonmyrank)
  {
    // get all crosslinker in current bin
    Core::Elements::Element* currentbin = bin_discret_ptr()->g_element(nb_iter);
    Core::Nodes::Node** clincurrentbin = currentbin->nodes();
    const int numcrosslinker = currentbin->num_node();

    // loop over all crosslinker in current bin
    for (int i = 0; i < numcrosslinker; ++i)
    {
      // get crosslinker in current bin
      Core::Nodes::Node* crosslinker_i = clincurrentbin[i];
      BeamInteraction::Data::CrosslinkerData* cldata_i =
          crosslinker_data_[crosslinker_i->lid()].get();

#ifdef FOUR_C_ENABLE_ASSERTIONS
      // safety checks
      if (crosslinker_i->owner() != g_state().get_my_rank())
        FOUR_C_THROW(
            " Only row owner of crosslinker changes its state, rank {} is not owner "
            "of linker with gid {}, but rank {}",
            g_state().get_my_rank(), crosslinker_i->id(), crosslinker_i->owner());
#endif

      switch (cldata_i->get_number_of_bonds())
      {
        case 0:
        {
          break;
        }
        case 1:
        {
          // dissolve bond and update states
          dissolve_bond(crosslinker_i, get_single_occupied_cl_bspot(cldata_i->get_b_spots()),
              cldata_i->get_number_of_bonds(), sendunbindevents, myrankunbindevents);
          break;
        }
        case 2:
        {
          // dissolve random bond and update states
          dissolve_bond(crosslinker_i,
              BeamInteraction::Utils::permutation(cldata_i->get_number_of_bonds())[0],
              cldata_i->get_number_of_bonds(), sendunbindevents, myrankunbindevents);

          // in case we want to allow transition from double bonded to free, take same linker
          // again
          if (doubleunbind) --i;
          break;
        }
        default:
        {
          FOUR_C_THROW(" Unrealistic number {} of bonds for a crosslinker.",
              cldata_i->get_number_of_bonds());
        }
      }
    }
  }

  // communicate which elements need to be updated on rank != myrank
  communicate_crosslinker_unbinding(sendunbindevents, myrankunbindevents);

  // update binding status of beam binding partners on myrank
  update_beam_binding_status_after_unbinding(myrankunbindevents);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::
    determine_responsilbe_procs_for_forced_crosslinker_unbinding(
        std::set<int> const& bingids, std::set<int>& binsonmyrank) const
{
  check_init_setup();

  std::set<int> checkedbins;
  std::map<int, std::vector<int>> binstosend;

  // determine all bins that need to dissolve all bonds on myrank and
  // on other procs (these need to be informed by myrank)
  for (auto const& b_iter : bingids)
  {
    std::vector<int> nb_binIds;
    nb_binIds.reserve(27);
    // do not check on existence here
    bin_strategy().get_neighbor_and_own_bin_ids(b_iter, nb_binIds);

    for (auto const& nb_iter : nb_binIds)
    {
      // safety check
      if (not bin_discret().have_global_element(nb_iter))
        FOUR_C_THROW("Not entire neighborhood ghosted, this is a problem in the following ");

      // if a bin has already been examined --> continue with next bin
      // like this we get a unique vector that myrank sends
      if (checkedbins.find(nb_iter) != checkedbins.end()) continue;
      // else: bin is examined for the first time --> new entry in examinedbins_
      else
        checkedbins.insert(nb_iter);

      // decide who needs to dissolve bonds
      const int owner = bin_discret().g_element(nb_iter)->owner();
      if (owner == g_state().get_my_rank())
        binsonmyrank.insert(nb_iter);
      else
        binstosend[owner].push_back(nb_iter);
    }
  }

  communicate_bin_ids(binstosend, binsonmyrank);
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::communicate_bin_ids(
    std::map<int, std::vector<int>> const& binstosend, std::set<int>& binsonmyrank) const
{
  check_init_setup();

  // build exporter
  Core::Communication::Exporter exporter(discret().get_comm());
  int const numproc = Core::Communication::num_mpi_ranks(discret().get_comm());
  int const myrank = g_state().get_my_rank();

  // ---- send ---- ( we do not need to pack anything)
  int const length = binstosend.size();
  std::vector<MPI_Request> request(length);
  int tag = 0;
  std::map<int, std::vector<int>>::const_iterator p;
  std::vector<int> targetprocs(numproc, 0);
  for (p = binstosend.begin(); p != binstosend.end(); ++p)
  {
    targetprocs[p->first] = 1;
    exporter.i_send(myrank, p->first, (p->second).data(), static_cast<int>((p->second).size()),
        1234, request[tag]);
    ++tag;
  }
  if (tag != length) FOUR_C_THROW("Number of messages is mixed up");

  // ---- prepare receiving procs -----
  std::vector<int> summedtargets(numproc, 0);
  summedtargets = Core::Communication::sum_all(targetprocs, discret().get_comm());

  // ---- receive ----- (we do not need to unpack anything)
  for (int rec = 0; rec < summedtargets[myrank]; ++rec)
  {
    std::vector<int> rdata;
    int length = 0;
    int tag = -1;
    int from = -1;
    exporter.receive_any(from, tag, rdata, length);
    if (tag != 1234)
      FOUR_C_THROW("Received on proc {} data with wrong tag from proc {}", myrank, from);

    // insert in binsonmyrank
    binsonmyrank.insert(rdata.begin(), rdata.end());
  }

  // wait for all communication to finish
  wait(exporter, request, static_cast<int>(binstosend.size()));
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::dissolve_bond(Core::Nodes::Node* linker,
    int freedbspotid, int numbondsold,
    std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::UnBindEventData>>>&
        sendunbindevents,
    std::vector<std::shared_ptr<BeamInteraction::Data::UnBindEventData>>& myrankunbindevents)
{
  check_init_setup();

#ifdef FOUR_C_ENABLE_ASSERTIONS
  // safety check
  if (numbondsold < 1) FOUR_C_THROW("dissolution of free crosslinker does not make any sense");
#endif

  CrossLinking::CrosslinkerNode* crosslinker = dynamic_cast<CrossLinking::CrosslinkerNode*>(linker);

  // get linker data
  const int clcollid = crosslinker->lid();
  BeamInteraction::Data::CrosslinkerData* cldata = crosslinker_data_[clcollid].get();

  // store unbinding event data
  std::shared_ptr<BeamInteraction::Data::UnBindEventData> unbindevent =
      std::make_shared<BeamInteraction::Data::UnBindEventData>();
  unbindevent->set_cl_id(linker->id());
  unbindevent->set_ele_toupdate(cldata->get_b_spots()[freedbspotid]);
  unbindevent->set_linker_type(crosslinker->get_material()->linker_type());

  // owner of beam
  const int beamowner = discret_ptr()->g_element(unbindevent->get_ele_toupdate().first)->owner();

  // check who needs to update the element status
  if (beamowner == g_state().get_my_rank())
    myrankunbindevents.push_back(unbindevent);
  else
    sendunbindevents[beamowner].push_back(unbindevent);

  // -----------------------------------------------------------------
  // update crosslinker status
  // -----------------------------------------------------------------
  // update binding status of linker
  cldata->set_bspot(freedbspotid, std::make_pair(-1, -1));

  // update number of bonds
  cldata->set_number_of_bonds(numbondsold - 1);

  if (numbondsold == 1)
  {
    set_position_of_newly_free_crosslinker(crosslinker, cldata);
  }
  else if (numbondsold == 2)
  {
    int stayoccpotid = 0;
    if (freedbspotid == 0) stayoccpotid = 1;

    set_position_of_newly_single_bonded_crosslinker(crosslinker, cldata, stayoccpotid);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    // safety check
    if (not doublebondcl_.count(linker->id()))
      FOUR_C_THROW("crosslinker {} with {} bonds is not in double bonded map of rank {}",
          linker->id(), cldata->get_number_of_bonds() + 1, g_state_ptr()->get_my_rank());
#endif

    // erase crosslinker from double bonded crosslinker list
    doublebondcl_.erase(linker->id());
  }
  else
  {
    FOUR_C_THROW("dissolution of free linker does not make any sense, something went wrong.");
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::communicate_undecided_bonds(
    std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::BindEventData>>>&
        undecidedbonds,
    std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::BindEventData>>>& requestedcl)
    const
{
  check_init();

  // do communication
  std::vector<std::shared_ptr<BeamInteraction::Data::BindEventData>> recvbindevent;
  i_send_recv_any(undecidedbonds, recvbindevent);

  for (unsigned int i = 0; i < recvbindevent.size(); ++i)
    requestedcl[recvbindevent[i]->get_cl_id()].push_back(recvbindevent[i]);

  recvbindevent.clear();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::communicate_decided_bonds(
    std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::BindEventData>>>& decidedbonds,
    std::map<int, std::shared_ptr<BeamInteraction::Data::BindEventData>>& myelebonds) const
{
  check_init();

  // communicate decided bonds
  std::vector<std::shared_ptr<BeamInteraction::Data::BindEventData>> recvbindevent;
  i_send_recv_any(decidedbonds, recvbindevent);

  // loop over received binding events
  for (unsigned int i = 0; i < recvbindevent.size(); ++i)
  {
    // add binding events to new colbond map
    if (recvbindevent[i]->get_permission())
      myelebonds[recvbindevent[i]->get_cl_id()] = recvbindevent[i];
  }
  recvbindevent.clear();
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::decide_binding_in_parallel(
    std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::BindEventData>>>& requestedcl,
    std::map<int, std::shared_ptr<BeamInteraction::Data::BindEventData>>& mybonds,
    std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::BindEventData>>>& decidedbonds)
    const
{
  check_init();

  std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::BindEventData>>>::iterator
      cliter;
  // loop over all requested cl (note myrank is owner of these)
  for (cliter = requestedcl.begin(); cliter != requestedcl.end(); ++cliter)
  {
    // check if myrank wants to bind this crosslinker
    bool myrankbond = false;
    if (mybonds.find(cliter->first) != mybonds.end()) myrankbond = true;

    // ---------------------------------------------------------------------
    // if only one request and myrank does not want to bind this cl,
    // requesting proc gets the permission to do so
    // ---------------------------------------------------------------------
    if (static_cast<int>(cliter->second.size()) == 1 and not myrankbond)
    {
      // we send back the permission to the relevant proc, because myrank as row
      // owner of bspot needs to set the respective stuff for the element of this
      // binding event
      // note: permission = true was send as default, so this can be sent back
      // without changes
      decidedbonds[cliter->second[0]->get_request_proc()].push_back(cliter->second[0]);

#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (cliter->second[0]->get_permission() != 1)
        FOUR_C_THROW(
            " something during communication went wrong, default true permission "
            " not received");
#endif

      // insert this new binding event in map of myrank, because as row owner of
      // this cl he is responsible to set the respective stuff for the crosslinker
      // of this binding event
      mybonds[cliter->first] = cliter->second[0];

      // go to next crosslinker
      continue;
    }

    // ---------------------------------------------------------------------
    // in case number of requesting procs >1 for this cl or myrank wants to
    // set it itself
    // ---------------------------------------------------------------------
    int numrequprocs = static_cast<int>(cliter->second.size());
    if (myrankbond) numrequprocs += 1;

    // get random proc out of affected ones
    Global::Problem::instance()->random()->set_rand_range(0.0, 1.0);
    // fixme: what if random number exactly = 1?
    int rankwithpermission =
        std::floor(numrequprocs * Global::Problem::instance()->random()->uni());

    // myrank is allowed to set link
    if (myrankbond and rankwithpermission == (numrequprocs - 1))
    {
      // note: this means link is set between row cl and row ele on myrank,
      // all relevant information for myrank is stored in mybonds
      // loop over all requesters and store their veto
      std::vector<std::shared_ptr<BeamInteraction::Data::BindEventData>>::iterator iter;
      for (iter = cliter->second.begin(); iter != cliter->second.end(); ++iter)
      {
        (*iter)->set_permission(0);
        decidedbonds[(*iter)->get_request_proc()].push_back(*iter);
      }
    }
    // certain requester is allowed to set the link
    else
    {
      // loop over all requesters and store veto for all requester except for one
      std::vector<std::shared_ptr<BeamInteraction::Data::BindEventData>>::iterator iter;

      int counter = 0;
      for (iter = cliter->second.begin(); iter != cliter->second.end(); ++iter)
      {
        if (rankwithpermission == counter)
        {
          // permission for this random proc
          decidedbonds[(*iter)->get_request_proc()].push_back(*iter);

#ifdef FOUR_C_ENABLE_ASSERTIONS
          if ((*iter)->get_permission() != 1)
            FOUR_C_THROW(
                " something during communication went wrong, default true permission "
                " not received");
#endif

          // erase old binding event
          if (myrankbond) mybonds.erase(cliter->first);

          // insert new binding event
          mybonds[cliter->first] = *iter;
        }
        else
        {
          (*iter)->set_permission(0);
          decidedbonds[(*iter)->get_request_proc()].push_back(*iter);
        }
        counter++;
      }
    }
  }
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::communicate_beam_link_after_redistribution(
    std::map<int, std::vector<std::shared_ptr<BeamInteraction::BeamLink>>>& dbondcltosend)
{
  check_init();

  // build exporter
  Core::Communication::Exporter exporter(discret_ptr()->get_comm());
  int const numproc = Core::Communication::num_mpi_ranks(discret_ptr()->get_comm());

  // -----------------------------------------------------------------------
  // send
  // -----------------------------------------------------------------------
  // ---- pack data for sending -----
  std::map<int, std::vector<char>> sdata;
  std::vector<int> targetprocs(numproc, 0);
  std::map<int, std::vector<std::shared_ptr<BeamInteraction::BeamLink>>>::const_iterator p;
  for (p = dbondcltosend.begin(); p != dbondcltosend.end(); ++p)
  {
    std::vector<std::shared_ptr<BeamInteraction::BeamLink>>::const_iterator iter;
    for (iter = p->second.begin(); iter != p->second.end(); ++iter)
    {
      Core::Communication::PackBuffer data;
      (*iter)->pack(data);
      sdata[p->first].insert(sdata[p->first].end(), data().begin(), data().end());
    }
    targetprocs[p->first] = 1;
  }

  // ---- send ----
  const int length = sdata.size();
  std::vector<MPI_Request> request(length);
  int tag = 0;
  for (std::map<int, std::vector<char>>::const_iterator p = sdata.begin(); p != sdata.end(); ++p)
  {
    exporter.i_send(g_state().get_my_rank(), p->first, (p->second).data(), (int)(p->second).size(),
        1234, request[tag]);
    ++tag;
  }
  if (tag != length) FOUR_C_THROW("Number of messages is mixed up");

  // -----------------------------------------------------------------------
  // receive
  // -----------------------------------------------------------------------
  // ---- prepare receiving procs -----
  std::vector<int> summedtargets(numproc, 0);
  summedtargets = Core::Communication::sum_all(targetprocs, discret_ptr()->get_comm());

  // myrank receive all packs that are sent to him
  for (int rec = 0; rec < summedtargets[g_state().get_my_rank()]; ++rec)
  {
    std::vector<char> rdata;
    int length = 0;
    int tag = -1;
    int from = -1;
    exporter.receive_any(from, tag, rdata, length);
    if (tag != 1234)
      FOUR_C_THROW(
          "Received on proc {} data with wrong tag from proc {}", g_state().get_my_rank(), from);

    // store received data

    Core::Communication::UnpackBuffer buffer(rdata);
    while (!buffer.at_end())
    {
      std::shared_ptr<Core::Communication::ParObject> object =
          std::shared_ptr<Core::Communication::ParObject>(Core::Communication::factory(buffer));
      std::shared_ptr<BeamInteraction::BeamLink> beamtobeamlink =
          std::dynamic_pointer_cast<BeamInteraction::BeamLink>(object);
      if (beamtobeamlink == nullptr) FOUR_C_THROW("Received object is not a beam to beam linkage");

#ifdef FOUR_C_ENABLE_ASSERTIONS
      // some safety checks
      if (bin_discret_ptr()->g_node(beamtobeamlink->id())->owner() != g_state_ptr()->get_my_rank())
        FOUR_C_THROW(
            " A double bond was sent to rank {}, although it is not the owner of "
            "the cl with gid {} ",
            g_state_ptr()->get_my_rank(), beamtobeamlink->id());
      if (doublebondcl_.count(beamtobeamlink->id()))
        FOUR_C_THROW(" Rank {} got sent double bonded crosslinker {} which it already has ",
            g_state_ptr()->get_my_rank(), beamtobeamlink->id());
#endif

      // insert new double bonds in my list
      doublebondcl_[beamtobeamlink->id()] = beamtobeamlink;
    }
  }

  // wait for all communication to finish
  wait(exporter, request, static_cast<int>(sdata.size()));
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::communicate_crosslinker_unbinding(
    std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::UnBindEventData>>>&
        sendunbindevent,
    std::vector<std::shared_ptr<BeamInteraction::Data::UnBindEventData>>& myrankunbindevent) const
{
  check_init();

  i_send_recv_any(sendunbindevent, myrankunbindevent);
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
template <typename T>
void BeamInteraction::SubmodelEvaluator::Crosslinking::i_send(
    Core::Communication::Exporter& exporter, std::vector<MPI_Request>& request,
    std::map<int, std::vector<std::shared_ptr<T>>> const& send) const
{
  check_init();

  // ---- pack data for sending -----
  std::map<int, std::vector<char>> sdata;
  typename std::map<int, std::vector<std::shared_ptr<T>>>::const_iterator p;
  for (p = send.begin(); p != send.end(); ++p)
  {
    typename std::vector<std::shared_ptr<T>>::const_iterator iter;

    for (iter = p->second.begin(); iter != p->second.end(); ++iter)
    {
      Core::Communication::PackBuffer data;
      (*iter)->pack(data);
      sdata[p->first].insert(sdata[p->first].end(), data().begin(), data().end());
    }
  }

  // ---- send ----
  const int length = sdata.size();
  request.resize(length);
  int tag = 0;
  for (std::map<int, std::vector<char>>::const_iterator p = sdata.begin(); p != sdata.end(); ++p)
  {
    exporter.i_send(g_state().get_my_rank(), p->first, (p->second).data(),
        static_cast<int>((p->second).size()), 1234, request[tag]);
    ++tag;
  }
  if (tag != length) FOUR_C_THROW("Number of messages is mixed up");
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
template <typename T>
void BeamInteraction::SubmodelEvaluator::Crosslinking::prepare_receiving_procs(
    std::map<int, std::vector<std::shared_ptr<T>>> const& datasenttorank,
    std::vector<int>& summedtargets) const
{
  check_init();

  const int numproc = Core::Communication::num_mpi_ranks(discret().get_comm());

  // get number of procs from which myrank receives data
  std::vector<int> targetprocs(numproc, 0);
  typename std::map<int, std::vector<std::shared_ptr<T>>>::const_iterator prociter;
  for (prociter = datasenttorank.begin(); prociter != datasenttorank.end(); ++prociter)
    targetprocs[prociter->first] = 1;
  // store number of messages myrank receives
  summedtargets.resize(numproc, 0);
  summedtargets = Core::Communication::sum_all(targetprocs, discret().get_comm());
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
template <typename T>
void BeamInteraction::SubmodelEvaluator::Crosslinking::recv_any(
    Core::Communication::Exporter& exporter, int receivesize,
    std::vector<std::shared_ptr<T>>& recv) const
{
  check_init();

  // myrank receive all packs that are sent to him
  for (int rec = 0; rec < receivesize; ++rec)
  {
    std::vector<char> rdata;
    int length = 0;
    int tag = -1;
    int from = -1;
    exporter.receive_any(from, tag, rdata, length);
    if (tag != 1234)
      FOUR_C_THROW(
          "Received on proc {} data with wrong tag from proc {}", g_state().get_my_rank(), from);

    // store received data

    Core::Communication::UnpackBuffer buffer(rdata);
    while (!buffer.at_end())
    {
      std::shared_ptr<T> data_container(BeamInteraction::Data::create_data_container<T>(buffer));

      // add received data to list
      recv.push_back(data_container);
    }
  }
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
template <typename T>
void BeamInteraction::SubmodelEvaluator::Crosslinking::i_send_recv_any(
    std::map<int, std::vector<std::shared_ptr<T>>> const& send,
    std::vector<std::shared_ptr<T>>& recv) const
{
  check_init();

  // build exporter
  Core::Communication::Exporter exporter(bin_discret().get_comm());

  // -----------------------------------------------------------------------
  // send
  // -----------------------------------------------------------------------
  // unblocking send
  std::vector<MPI_Request> request;
  i_send(exporter, request, send);

  // -----------------------------------------------------------------------
  // prepare receive
  // -----------------------------------------------------------------------
  std::vector<int> summedtargets;
  prepare_receiving_procs(send, summedtargets);

  // -----------------------------------------------------------------------
  // receive
  // -----------------------------------------------------------------------
  int receivesize = summedtargets[g_state().get_my_rank()];
  recv_any(exporter, receivesize, recv);

  // wait for all communication to finish
  wait(exporter, request, static_cast<int>(send.size()));
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::wait(
    Core::Communication::Exporter& exporter, std::vector<MPI_Request>& request, int length) const
{
  check_init();

  // wait for all communication to finish
  for (int i = 0; i < length; ++i) exporter.wait(request[i]);

  // note: if we have done everything correct, this should be a no time operation
  Core::Communication::barrier(bin_discret().get_comm());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Crosslinking::print_and_check_bind_event_data(
    BeamInteraction::Data::BindEventData& bindeventdata) const
{
  check_init();

  // extract data
  std::cout << "\n Rank: " << g_state().get_my_rank() << std::endl;
  std::cout << " crosslinker gid " << bindeventdata.get_cl_id() << std::endl;
  std::cout << " element gid " << bindeventdata.get_ele_id() << std::endl;
  std::cout << " bspot local number " << bindeventdata.get_b_spot_loc_n() << std::endl;
  std::cout << " requesting proc " << bindeventdata.get_request_proc() << std::endl;
  std::cout << " permission " << bindeventdata.get_permission() << std::endl;

  if (bindeventdata.get_cl_id() < 0 or bindeventdata.get_ele_id() < 0 or
      bindeventdata.get_b_spot_loc_n() < 0 or bindeventdata.get_request_proc() < 0 or
      not(bindeventdata.get_permission() == 0 or bindeventdata.get_permission() == 1))
    FOUR_C_THROW(" your bindevent does not make sense.");
}


//-----------------------------------------------------------------------------
// explicit template instantiation (to please every compiler)
//-----------------------------------------------------------------------------
template void BeamInteraction::SubmodelEvaluator::Crosslinking::i_send(
    Core::Communication::Exporter&, std::vector<MPI_Request>&,
    std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::CrosslinkerData>>> const&)
    const;
template void BeamInteraction::SubmodelEvaluator::Crosslinking::i_send(
    Core::Communication::Exporter&, std::vector<MPI_Request>&,
    std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::BeamData>>> const&) const;
template void BeamInteraction::SubmodelEvaluator::Crosslinking::i_send(
    Core::Communication::Exporter&, std::vector<MPI_Request>&,
    std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::BindEventData>>> const&) const;
template void BeamInteraction::SubmodelEvaluator::Crosslinking::i_send(
    Core::Communication::Exporter&, std::vector<MPI_Request>&,
    std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::UnBindEventData>>> const&)
    const;


template void BeamInteraction::SubmodelEvaluator::Crosslinking::prepare_receiving_procs(
    std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::CrosslinkerData>>> const&,
    std::vector<int>&) const;
template void BeamInteraction::SubmodelEvaluator::Crosslinking::prepare_receiving_procs(
    std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::BeamData>>> const&,
    std::vector<int>&) const;
template void BeamInteraction::SubmodelEvaluator::Crosslinking::prepare_receiving_procs(
    std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::BindEventData>>> const&,
    std::vector<int>&) const;
template void BeamInteraction::SubmodelEvaluator::Crosslinking::prepare_receiving_procs(
    std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::UnBindEventData>>> const&,
    std::vector<int>&) const;


template void BeamInteraction::SubmodelEvaluator::Crosslinking::recv_any(
    Core::Communication::Exporter&, int const,
    std::vector<std::shared_ptr<BeamInteraction::Data::CrosslinkerData>>&) const;
template void BeamInteraction::SubmodelEvaluator::Crosslinking::recv_any(
    Core::Communication::Exporter&, int const,
    std::vector<std::shared_ptr<BeamInteraction::Data::BeamData>>&) const;
template void BeamInteraction::SubmodelEvaluator::Crosslinking::recv_any(
    Core::Communication::Exporter&, int const,
    std::vector<std::shared_ptr<BeamInteraction::Data::BindEventData>>&) const;
template void BeamInteraction::SubmodelEvaluator::Crosslinking::recv_any(
    Core::Communication::Exporter&, int const,
    std::vector<std::shared_ptr<BeamInteraction::Data::UnBindEventData>>&) const;


template void BeamInteraction::SubmodelEvaluator::Crosslinking::i_send_recv_any(
    std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::CrosslinkerData>>> const&,
    std::vector<std::shared_ptr<BeamInteraction::Data::CrosslinkerData>>&) const;
template void BeamInteraction::SubmodelEvaluator::Crosslinking::i_send_recv_any(
    std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::BeamData>>> const&,
    std::vector<std::shared_ptr<BeamInteraction::Data::BeamData>>&) const;
template void BeamInteraction::SubmodelEvaluator::Crosslinking::i_send_recv_any(
    std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::BindEventData>>> const&,
    std::vector<std::shared_ptr<BeamInteraction::Data::BindEventData>>&) const;
template void BeamInteraction::SubmodelEvaluator::Crosslinking::i_send_recv_any(
    std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::UnBindEventData>>> const&,
    std::vector<std::shared_ptr<BeamInteraction::Data::UnBindEventData>>&) const;

FOUR_C_NAMESPACE_CLOSE
