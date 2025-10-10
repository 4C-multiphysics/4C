// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_xfem_condition_manager.hpp"

#include "4C_cut_volumecell.hpp"
#include "4C_io.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_xfem_utils.hpp"


FOUR_C_NAMESPACE_OPEN


// constructor
XFEM::ConditionManager::ConditionManager(
    const std::map<std::string, int>& dofset_coupling_map,  ///< ???
    std::shared_ptr<Core::FE::Discretization>& bg_dis,      ///< background discretization
    std::vector<std::shared_ptr<Core::FE::Discretization>>&
        meshcoupl_dis,  ///< mesh coupling discretizations
    std::vector<std::shared_ptr<Core::FE::Discretization>>&
        levelsetcoupl_dis,  ///< levelset coupling discretizations
    const double time,      ///< time
    const int step          ///< time step
    )
    : dofset_coupling_map_(dofset_coupling_map),
      bg_dis_(bg_dis),
      levelset_gid_(-1),
      time_(time),
      step_(step),
      is_levelset_uptodate_(false),
      ele_lsc_coup_idx_col_(nullptr),
      bg_phinp_(nullptr),
      isinit_(false),
      issetup_(false)
{
  // create Levelset Coupling objects
  {
    std::vector<std::string> conditions_to_check;

    // NOTE: CHANGING THE ORDER HERE CAN INFLUENCE BOOLEAN COMBINATIONS WHEN CREATING UNIQUE
    // LS-FIELD
    conditions_to_check.push_back(
        "XFEMLevelsetNeumann");  // Neumann before Dirichlet, otherwise artificial error at outflow
                                 // due to rough approximation by level-set
    conditions_to_check.push_back("XFEMLevelsetWeakDirichlet");
    conditions_to_check.push_back("XFEMLevelsetNavierSlip");
    conditions_to_check.push_back("XFEMLevelsetTwophase");
    conditions_to_check.push_back("XFEMLevelsetCombustion");

    create_couplings(levelsetcoupl_dis, conditions_to_check, false);
  }

  // create Mesh Coupling objects
  {
    std::vector<std::string> conditions_to_check;
    conditions_to_check.push_back("XFEMSurfNeumann");
    conditions_to_check.push_back("XFEMSurfWeakDirichlet");
    conditions_to_check.push_back("XFEMSurfFSIPart");
    conditions_to_check.push_back("XFEMSurfFSIMono");
    conditions_to_check.push_back("XFEMSurfFPIMono");
    conditions_to_check.push_back("XFEMSurfFluidFluid");
    conditions_to_check.push_back("XFEMSurfNavierSlip");
    conditions_to_check.push_back("XFEMSurfNavierSlipTwoPhase");
    conditions_to_check.push_back("EmbeddedMeshSolidSurfCoupling");
    conditions_to_check.push_back("EmbeddedMeshSolidVolBackground");

    create_couplings(meshcoupl_dis, conditions_to_check, true);
  }

  set_dof_set_coupling_map(dofset_coupling_map);
}


void XFEM::ConditionManager::create_couplings(
    std::vector<std::shared_ptr<Core::FE::Discretization>>&
        coupl_dis,  ///< coupling discretizations
    const std::vector<std::string>&
        conditions_to_check,   ///< conditions for which coupling objects shall be created
    bool create_mesh_coupling  ///< create mesh coupling or level-set coupling object
)
{
  // check if a coupling discretization has relevant conditioned nodes
  // create new coupling object for each type of condition and each coupling discretization
  for (size_t c_idx = 0; c_idx < coupl_dis.size();
      c_idx++)  // loop all specified mesh coupling discretizations
  {
    if (coupl_dis[c_idx] == nullptr) continue;

    std::vector<std::string> names;
    coupl_dis[c_idx]->get_condition_names(names);

    for (size_t c = 0; c < conditions_to_check.size(); c++)
    {
      if (std::find(names.begin(), names.end(), conditions_to_check[c]) == names.end()) continue;

      // get all conditions of this type, if several conditions with different coupling ids
      std::set<int> coupling_ids;
      get_coupling_ids(*(coupl_dis[c_idx]), conditions_to_check[c], coupling_ids);

      // create new coupling object for each composite
      for (std::set<int>::iterator cid = coupling_ids.begin(); cid != coupling_ids.end(); ++cid)
      {
        if (create_mesh_coupling)
          create_new_mesh_coupling(conditions_to_check[c], coupl_dis[c_idx], *cid);
        else
          create_new_level_set_coupling(conditions_to_check[c], coupl_dis[c_idx], *cid);
      }
    }
  }
}

void XFEM::ConditionManager::get_coupling_ids(const Core::FE::Discretization& cond_dis,
    const std::string& condition_name, std::set<int>& coupling_ids)
{
  // get all conditions of this type, if several conditions with different coupling ids
  // create an own coupling object for each coupling id
  std::vector<const Core::Conditions::Condition*> conditions;
  cond_dis.get_condition(condition_name, conditions);

  // CompositeByCouplingId
  for (size_t s = 0; s < conditions.size(); ++s)
  {
    const Core::Conditions::Condition* cond = conditions[s];
    const int couplingID = cond->parameters().get<int>("COUPLINGID");

    coupling_ids.insert(couplingID);
  }
}

void XFEM::ConditionManager::set_dof_set_coupling_map(
    const std::map<std::string, int>& dofset_coupling_map)
{
  for (int m = 0; m < num_mesh_coupling(); m++)
  {
    mesh_coupl_[m]->set_dof_set_coupling_map(dofset_coupling_map_);
  }

  for (int l = 0; l < num_level_set_coupling(); l++)
  {
    levelset_coupl_[l]->set_dof_set_coupling_map(dofset_coupling_map_);
  }
}



void XFEM::ConditionManager::status()
{
  int myrank = Core::Communication::my_mpi_rank(bg_dis_->get_comm());

  // -------------------------------------------------------------------
  //                       output to screen
  // -------------------------------------------------------------------
  if (myrank == 0)
  {
    printf(
        "   "
        "+-----------------------------------------------------------------------------------------"
        "------------------------------------------------------------------------------------------"
        "---+\n");
    printf(
        "   +----------------------------------------------------XFEM::ConditionManager - Created "
        "Coupling "
        "objects---------------------------------------------------------------------------------+"
        "\n");
    printf(
        "   "
        "+----------+-----------+-----------------------------+---------+--------------------------"
        "---+-----------------------------+-----------------------------+--------------------------"
        "---+\n");
    printf(
        "   | COUP-IDX | START-SID |       CONDITION-TYPE        | COUP-ID |          CUTTER-DIS   "
        "      | created from CONDITION-DIS  |        COUPLING-DIS         |     "
        "AVERAGING-STRATEGY      |\n");

    if (has_mesh_coupling())
    {
      printf(
          "   "
          "+----------+-----------+-----------------------------+---------+------------------------"
          "-----+-----------------------------+-----------------------------+----------------------"
          "-------+\n");
      printf(
          "   |Mesh Coupling Objects                                                               "
          "                                                                                        "
          "          |\n");
    }

    // loop all mesh coupling objects
    for (int mc = 0; mc < (int)mesh_coupl_.size(); mc++)
    {
      mesh_coupl_[mc]->status(mc, mesh_coupl_start_gid_[mc]);
    }

    if (has_level_set_coupling())
    {
      printf(
          "   "
          "+----------+-----------+-----------------------------+---------+------------------------"
          "-----+-----------------------------+-----------------------------+----------------------"
          "-------+\n");
      printf(
          "   |Levelset Coupling Objects                                                           "
          "                                                                                        "
          "          |\n");
    }

    // loop all levelset coupling objects
    for (int lsc = 0; lsc < (int)levelset_coupl_.size(); lsc++)
    {
      levelset_coupl_[lsc]->status(lsc, levelset_gid_);
    }

    printf(
        "   "
        "+----------+-----------+-----------------------------+---------+--------------------------"
        "---+-----------------------------+-----------------------------+--------------------------"
        "---+\n");
    printf(
        "   "
        "+-----------------------------------------------------------------------------------------"
        "------------------------------------------------------------------------------------------"
        "---+\n");
  }
}


void XFEM::ConditionManager::increment_time_and_step(const double dt)
{
  step_ += 1;
  time_ += dt;

  // loop all mesh coupling objects
  for (int mc = 0; mc < (int)mesh_coupl_.size(); mc++)
  {
    mesh_coupl_[mc]->increment_time_and_step(dt);
  }

  // loop all levelset coupling objects
  for (int lsc = 0; lsc < (int)levelset_coupl_.size(); lsc++)
  {
    levelset_coupl_[lsc]->increment_time_and_step(dt);
  }
}


void XFEM::ConditionManager::set_time_and_step(const double time, const int step)
{
  time_ = time;
  step_ = step;

  // loop all mesh coupling objects
  for (int mc = 0; mc < (int)mesh_coupl_.size(); mc++)
  {
    mesh_coupl_[mc]->set_time_and_step(time, step);
  }

  // loop all levelset coupling objects
  for (int lsc = 0; lsc < (int)levelset_coupl_.size(); lsc++)
  {
    levelset_coupl_[lsc]->set_time_and_step(time, step);
  }
}



void XFEM::ConditionManager::create_new_level_set_coupling(const std::string& cond_name,
    std::shared_ptr<Core::FE::Discretization> cond_dis, const int coupling_id)
{
  add_level_set_coupling(cond_name, cond_dis, coupling_id);
}


void XFEM::ConditionManager::create_new_mesh_coupling(const std::string& cond_name,
    std::shared_ptr<Core::FE::Discretization>
        cond_dis,  ///< discretization from which the cutter discretization can be derived
    const int coupling_id)
{
  add_mesh_coupling(cond_name, cond_dis, coupling_id);
}

void XFEM::ConditionManager::init()
{
  issetup_ = false;

  //--------------------------------------------------------
  // loop all mesh coupling objects
  for (int mc = 0; mc < (int)mesh_coupl_.size(); mc++)
  {
    mesh_coupl_[mc]->init();
  }

  //--------------------------------------------------------
  // loop all levelset coupling objects
  for (int lc = 0; lc < (int)levelset_coupl_.size(); lc++)
  {
    levelset_coupl_[lc]->init();
  }

  isinit_ = true;
}


void XFEM::ConditionManager::setup()
{
  check_init();

  // do setup

  //--------------------------------------------------------
  // loop all mesh coupling objects
  for (int mc = 0; mc < (int)mesh_coupl_.size(); mc++)
  {
    mesh_coupl_[mc]->setup();
  }

  //--------------------------------------------------------
  // loop all levelset coupling objects
  for (int lc = 0; lc < (int)levelset_coupl_.size(); lc++)
  {
    levelset_coupl_[lc]->setup();
  }

  create();

  issetup_ = true;
}



void XFEM::ConditionManager::create()
{
  numglobal_coupling_sides_ = 0;
  mesh_coupl_start_gid_.resize(mesh_coupl_.size());
  levelset_gid_ = -1;

  // set global side Ids for all Mesh coupling discretizations and level-set sides

  //--------------------------------------------------------
  // loop all mesh coupling objects
  for (int mc = 0; mc < (int)mesh_coupl_.size(); mc++)
  {
    std::shared_ptr<Core::FE::Discretization> mc_cutdis = mesh_coupl_[mc]->get_cutter_dis();
    if (mc_cutdis == nullptr) FOUR_C_THROW("cutter dis is nullptr");

    // set current number of global coupling sides as start index for global id this coupling object
    mesh_coupl_start_gid_[mc] = numglobal_coupling_sides_;

    // increase total number of sides with number of global side elements of this mesh coupling
    // object
    numglobal_coupling_sides_ += mc_cutdis->num_global_elements();
  }

  //--------------------------------------------------------
  // combine the level-set values

  if (levelset_coupl_.size() > 0)
  {
    // add one global levelset side used in the cut library
    levelset_gid_ = numglobal_coupling_sides_;
    numglobal_coupling_sides_ += 1;

    bg_phinp_ = Core::LinAlg::create_vector(*bg_dis_->node_row_map(), true);

    // information about the coupling condition for level-sets is obtained via the background
    // element for which we store the index of the level-set coupling object we allow for multiple
    // level-set coupling objects however only for one level-set side

    ele_lsc_coup_idx_col_ =
        std::make_shared<Core::LinAlg::Vector<int>>(*bg_dis_->element_col_map(), true);
  }

  //--------------------------------------------------------
  // print status of conditionManager to screen
  status();
}


void XFEM::ConditionManager::set_level_set_field(const double time)
{
  if (levelset_coupl_.size() != 1)
    FOUR_C_THROW("level-set field is not unique, which level-set field to be set?");

  is_levelset_uptodate_ = false;

  // update all level-set fields
  // loop all levelset coupling objects
  for (int lsc = 0; lsc < (int)levelset_coupl_.size(); lsc++)
  {
    levelset_coupl_[lsc]->set_level_set_field(time);
  }
}


void XFEM::ConditionManager::write_access_geometric_quantities(
    Core::LinAlg::Vector<double>& scalaraf, Core::LinAlg::MultiVector<double>& smoothed_gradphiaf,
    Core::LinAlg::Vector<double>& curvatureaf)
{
  // TODO: when using two-phase in combination with other levelset, how to access to the right
  // coupling twophase coupling object?
  // TODO: safety check, that there is a unique two-phase coupling object!?
  if (levelset_coupl_.size() != 1)
    FOUR_C_THROW("level-set field is not unique, which level-set field to be set by given vector?");

  is_levelset_uptodate_ = false;
}


std::shared_ptr<const Core::LinAlg::Vector<double>>
XFEM::ConditionManager::get_level_set_field_col()
{
  if (levelset_coupl_.size() == 0) return nullptr;

  // export nodal level-set values to node column map
  std::shared_ptr<Core::LinAlg::Vector<double>> bg_phinp_col =
      std::make_shared<Core::LinAlg::Vector<double>>(*bg_dis_->node_col_map());
  Core::LinAlg::export_to(*get_level_set_field(), *bg_phinp_col);

  return bg_phinp_col;
}

void XFEM::ConditionManager::update_level_set_field()
{
  //-------------------------------------------------------------------------------------------------
  // Boolean operations like \cap \cup \complementary \ ... are used to combine level-set functions
  //-------------------------------------------------------------------------------------------------
  // NOTE:
  // * we proceed for the coupling objects as they have been created (WDBC, NEUMANN, TWOPHASE,
  // COMBUSTION...)
  // * within one TYPE of conditions (e.g. WDBC) we combine single fields sorted by their coupling
  // ID
  // * the single groups are combined via MAX (\cap) operations,
  //   such that negative value are within fluid and positive values are non-fluid, or the second
  //   fluid phase!!!

  // assume same maps between background fluid dis and the cutterdis (scatra dis)

  // note: not only level-set values have to updated, but also which coupling condition is active in
  // which background element
  // -> 1st: store for each node from which level-set coupling the dominating level set values stems
  // from
  // -> 2nd: based on nodal information, we decide which coupling condition has to be evaluated on
  // an element for which the conditions are not unique

  Core::LinAlg::Vector<int> node_lsc_coup_idx(*bg_dis_->node_row_map(), true);
  Core::LinAlg::Vector<int> node_lsc_coup_idx_col(*bg_dis_->node_col_map(), true);
  Core::LinAlg::Vector<int> ele_lsc_coup_idx(*bg_dis_->element_row_map(), true);

  // for each row node, the dominating levelset coupling index
  for (int lsc = 0; lsc < num_level_set_coupling(); ++lsc)
  {
    std::shared_ptr<LevelSetCoupling>& coupling = levelset_coupl_[lsc];

    // get boolean combination w.r.t previously updated combination
    CouplingBase::LevelSetBooleanType ls_boolean_type = coupling->get_boolean_combination();

    if (lsc == 0)  // initialize with the first coupling!
    {
      if (coupling->get_boolean_combination() != CouplingBase::ls_none)
        FOUR_C_THROW(
            "the first Boundary-Condition level-set coupling (WDBC or NEUMANN) should always use "
            "BOOLEANTYPE=none ! Check your boolean operations");

      std::shared_ptr<Core::LinAlg::Vector<double>> tmp =
          coupling->get_level_set_field_as_node_row_vector();
      const int err = bg_phinp_->update(1.0, *tmp, 0.0);
      if (err) FOUR_C_THROW("update did not work - vectors based on wrong maps?");
    }
    else  // apply boolean combinations for the further level-set fields
    {
      if (ls_boolean_type == CouplingBase::ls_none)
        FOUR_C_THROW(
            "there is a level-set coupling for which you did not specify the the BOOLEANTYPE! "
            "Check your boolean operations");

      // need to live here
      std::shared_ptr<Core::LinAlg::Vector<double>> tmp =
          coupling->get_level_set_field_as_node_row_vector();
      combine_level_set_field(*bg_phinp_, *tmp, lsc, node_lsc_coup_idx, ls_boolean_type);
    }

    if (coupling->apply_complementary_operator()) build_complementary_level_set(*bg_phinp_);
  }

  // export to column vector
  Core::LinAlg::export_to(node_lsc_coup_idx, node_lsc_coup_idx_col);

  // set the levelset coupling index for all row elements
  const Core::LinAlg::Map* elerowmap = bg_dis_->element_row_map();
  const Core::LinAlg::Map* nodecolmap = bg_dis_->node_col_map();

  // loop all row elements on the processor
  for (int leleid = 0; leleid < bg_dis_->num_my_row_elements(); ++leleid)
  {
    const int gid = elerowmap->gid(leleid);
    Core::Elements::Element* ele = bg_dis_->g_element(gid);
    const int numnode = ele->num_node();
    const int* nodeids = ele->node_ids();

    std::set<int> lsc_coupling_indices;

    for (int n = 0; n < numnode; ++n)
    {
      int nlid = nodecolmap->lid(nodeids[n]);
      lsc_coupling_indices.insert(node_lsc_coup_idx_col.get_local_values()[nlid]);
    }

    //    if(lsc_coupling_indices.size() > 1)
    //    {
    //      for(std::set<int>::iterator i= lsc_coupling_indices.begin();
    //          i!= lsc_coupling_indices.end();
    //          ++i)
    //        std::cout << "for element: " << ele->Id() << " following lsc-indices: " << *i <<
    //        std::endl;
    //    }

    // take the one with the lowest coupling index!
    (ele_lsc_coup_idx).get_local_values()[leleid] = *(lsc_coupling_indices.begin());
  }

  Core::LinAlg::export_to(ele_lsc_coup_idx, *ele_lsc_coup_idx_col_);

  is_levelset_uptodate_ = true;
}

void XFEM::ConditionManager::combine_level_set_field(Core::LinAlg::Vector<double>& vec1,
    Core::LinAlg::Vector<double>& vec2, const int lsc_index_2,
    Core::LinAlg::Vector<int>& node_lsc_coup_idx,
    XFEM::CouplingBase::LevelSetBooleanType ls_boolean_type)
{
  switch (ls_boolean_type)
  {
    case XFEM::CouplingBase::ls_cut:
      set_maximum(vec1, vec2, lsc_index_2, node_lsc_coup_idx);
      break;
    case XFEM::CouplingBase::ls_union:
      set_minimum(vec1, vec2, lsc_index_2, node_lsc_coup_idx);
      break;
    case XFEM::CouplingBase::ls_difference:
      set_difference(vec1, vec2, lsc_index_2, node_lsc_coup_idx);
      break;
    case XFEM::CouplingBase::ls_sym_difference:
      set_symmetric_difference(vec1, vec2, lsc_index_2, node_lsc_coup_idx);
      break;
    default:
      FOUR_C_THROW("unsupported type of boolean operation between two level-sets");
      break;
  }
}


void XFEM::ConditionManager::check_for_equal_maps(
    Core::LinAlg::Vector<double>& vec1, Core::LinAlg::Vector<double>& vec2)
{
  if (not vec1.get_map().point_same_as(vec2.get_map())) FOUR_C_THROW("maps do not match!");
}


void XFEM::ConditionManager::set_minimum(Core::LinAlg::Vector<double>& vec1,
    Core::LinAlg::Vector<double>& vec2, const int lsc_index_2,
    Core::LinAlg::Vector<int>& node_lsc_coup_idx)
{
  int err = -1;

  check_for_equal_maps(vec1, vec2);

  // loop all nodes on the processor
  for (int lnodeid = 0; lnodeid < bg_dis_->num_my_row_nodes(); lnodeid++)
  {
    double val1 = (vec1)[lnodeid];
    double val2 = (vec2)[lnodeid];


    // std::min(val1, val2);
    int arg = -1;
    double final_val = XFEM::argmin(val1, val2, arg);

    if (arg == 2)
      (node_lsc_coup_idx).get_local_values()[lnodeid] =
          lsc_index_2;  // else keep the old lsc coupling

    // now copy the values
    err = vec1.replace_local_value(lnodeid, final_val);
    if (err != 0) FOUR_C_THROW("error while inserting value into phinp_");
  }
}


void XFEM::ConditionManager::set_maximum(Core::LinAlg::Vector<double>& vec1,
    Core::LinAlg::Vector<double>& vec2, const int lsc_index_2,
    Core::LinAlg::Vector<int>& node_lsc_coup_idx)
{
  int err = -1;

  check_for_equal_maps(vec1, vec2);

  // loop all nodes on the processor
  for (int lnodeid = 0; lnodeid < bg_dis_->num_my_row_nodes(); lnodeid++)
  {
    double val1 = (vec1)[lnodeid];
    double val2 = (vec2)[lnodeid];

    // std::max(val1, val2);
    int arg = -1;
    double final_val = XFEM::argmax(val1, val2, arg);

    if (arg == 2)
      (node_lsc_coup_idx).get_local_values()[lnodeid] =
          lsc_index_2;  // else keep the old lsc coupling

    // now copy the values
    err = vec1.replace_local_value(lnodeid, final_val);
    if (err != 0) FOUR_C_THROW("error while inserting value into phinp_");
  }
}


void XFEM::ConditionManager::set_difference(Core::LinAlg::Vector<double>& vec1,
    Core::LinAlg::Vector<double>& vec2, const int lsc_index_2,
    Core::LinAlg::Vector<int>& node_lsc_coup_idx)
{
  int err = -1;

  check_for_equal_maps(vec1, vec2);

  // loop all nodes on the processor
  for (int lnodeid = 0; lnodeid < bg_dis_->num_my_row_nodes(); lnodeid++)
  {
    double val1 = (vec1)[lnodeid];
    double val2 = (vec2)[lnodeid];

    // std::max(val1, -val2);
    int arg = -1;
    double final_val = XFEM::argmax(val1, -val2, arg);

    if (arg == 2)
      (node_lsc_coup_idx).get_local_values()[lnodeid] =
          lsc_index_2;  // else keep the old lsc coupling

    // now copy the values
    err = vec1.replace_local_value(lnodeid, final_val);
    if (err != 0) FOUR_C_THROW("error while inserting value into phinp_");
  }
}

void XFEM::ConditionManager::set_symmetric_difference(Core::LinAlg::Vector<double>& vec1,
    Core::LinAlg::Vector<double>& vec2, const int lsc_index_2,
    Core::LinAlg::Vector<int>& node_lsc_coup_idx)
{
  int err = -1;

  check_for_equal_maps(vec1, vec2);

  // loop all nodes on the processor
  for (int lnodeid = 0; lnodeid < bg_dis_->num_my_row_nodes(); lnodeid++)
  {
    double val1 = (vec1)[lnodeid];
    double val2 = (vec2)[lnodeid];

    int arg_tmp1 = -1;
    int arg_tmp2 = -1;
    double val_tmp1 = XFEM::argmax(val1, -val2, arg_tmp1);
    double val_tmp2 = XFEM::argmax(-val1, val2, arg_tmp2);

    int arg_tmp3 = -1;
    double final_val = XFEM::argmin(val_tmp1, val_tmp2, arg_tmp3);

    if (arg_tmp3 == 2)
      if (arg_tmp2 == 2)
        (node_lsc_coup_idx).get_local_values()[lnodeid] =
            lsc_index_2;  // else keep the old lsc coupling

    if (arg_tmp3 == 1)
      if (arg_tmp1 == 2)
        (node_lsc_coup_idx).get_local_values()[lnodeid] =
            lsc_index_2;  // else keep the old lsc coupling


    // now copy the values
    err = vec1.replace_local_value(lnodeid, final_val);
    if (err != 0) FOUR_C_THROW("error while inserting value into phinp_");
  }
}


void XFEM::ConditionManager::build_complementary_level_set(Core::LinAlg::Vector<double>& vec1)
{
  vec1.scale(-1.0);
}


void XFEM::ConditionManager::clear_state()
{
  // loop all mesh coupling objects
  for (int mc = 0; mc < (int)mesh_coupl_.size(); mc++)
  {
    mesh_coupl_[mc]->clear_state();
  }
}

void XFEM::ConditionManager::set_state()
{
  // loop all mesh coupling objects
  for (int mc = 0; mc < (int)mesh_coupl_.size(); mc++)
  {
    mesh_coupl_[mc]->set_state();
  }
}

void XFEM::ConditionManager::set_state_displacement()
{
  // loop all mesh coupling objects
  for (int mc = 0; mc < (int)mesh_coupl_.size(); mc++)
  {
    mesh_coupl_[mc]->set_state_displacement();
  }
}

void XFEM::ConditionManager::update_state_vectors()
{
  // loop all mesh coupling objects
  for (int mc = 0; mc < (int)mesh_coupl_.size(); mc++)
  {
    mesh_coupl_[mc]->update_state_vectors();
  }
}

void XFEM::ConditionManager::complete_state_vectors()
{
  // loop all mesh coupling objects
  for (int mc = 0; mc < (int)mesh_coupl_.size(); mc++)
  {
    mesh_coupl_[mc]->complete_state_vectors();
  }
}

void XFEM::ConditionManager::zero_state_vectors_fsi()
{
  // loop all mesh coupling objects
  for (int mc = 0; mc < (int)mesh_coupl_.size(); mc++)
  {
    mesh_coupl_[mc]->zero_state_vectors_fsi();
  }
}

void XFEM::ConditionManager::export_geometric_quantities()
{
  // loop all mesh coupling objects
  for (int lsc = 0; lsc < (int)levelset_coupl_.size(); lsc++)
  {
    levelset_coupl_[lsc]->export_geometric_quantities();
  }
}

void XFEM::ConditionManager::gmsh_output(const std::string& filename_base, const int step,
    const int gmsh_step_diff, const bool gmsh_debug_out_screen)
{
  // loop all mesh coupling objects
  for (int mc = 0; mc < (int)mesh_coupl_.size(); mc++)
  {
    mesh_coupl_[mc]->gmsh_output(filename_base, step, gmsh_step_diff, gmsh_debug_out_screen);
  }
  // loop all level set coupling objects
  for (int lc = 0; lc < (int)levelset_coupl_.size(); lc++)
  {
    levelset_coupl_[lc]->gmsh_output(filename_base, step, gmsh_step_diff, gmsh_debug_out_screen);
  }
}

void XFEM::ConditionManager::gmsh_output_discretization(std::ostream& gmshfilecontent)
{
  // loop all mesh coupling objects
  for (int mc = 0; mc < (int)mesh_coupl_.size(); mc++)
  {
    mesh_coupl_[mc]->gmsh_output_discretization(gmshfilecontent);
  }
}

void XFEM::ConditionManager::output(
    const int step, const double time, const bool write_restart_data)
{
  // loop all mesh coupling objects
  for (int mc = 0; mc < (int)mesh_coupl_.size(); mc++)
  {
    mesh_coupl_[mc]->output(step, time, write_restart_data);
  }

  // output for combined levelset field
  // no restart as bg_phinp can be rebuild from single level-set fields
  if (levelset_coupl_.size() > 0)
  {
    std::shared_ptr<Core::IO::DiscretizationWriter> output = bg_dis_->writer();
    output->write_vector("fluid_levelset_boundary", bg_phinp_);
  }

  // loop all level set coupling objects
  for (int lc = 0; lc < (int)levelset_coupl_.size(); lc++)
  {
    levelset_coupl_[lc]->output(step, time, write_restart_data, lc);
  }
}

void XFEM::ConditionManager::lift_drag(const int step, const double time)
{
  // loop all mesh coupling objects
  for (int mc = 0; mc < (int)mesh_coupl_.size(); mc++)
  {
    mesh_coupl_[mc]->lift_drag(step, time);
  }
}

void XFEM::ConditionManager::read_restart(const int step)
{
  // loop all mesh coupling objects
  for (int mc = 0; mc < (int)mesh_coupl_.size(); mc++)
  {
    mesh_coupl_[mc]->read_restart(step);
  }

  // loop all levelset coupling objects
  for (int lsc = 0; lsc < (int)levelset_coupl_.size(); lsc++)
  {
    levelset_coupl_[lsc]->read_restart(step, lsc);
  }
}


void XFEM::ConditionManager::prepare_solve()
{
  // loop all mesh coupling objects
  for (int mc = 0; mc < (int)mesh_coupl_.size(); mc++)
  {
    mesh_coupl_[mc]->prepare_solve();
  }

  // loop all levelset coupling objects
  for (int lsc = 0; lsc < (int)levelset_coupl_.size(); lsc++)
  {
    levelset_coupl_[lsc]->prepare_solve();
  }

  is_levelset_uptodate_ = false;
}

bool XFEM::ConditionManager::has_moving_interface()
{
  // loop all mesh coupling objects
  for (int mc = 0; mc < (int)mesh_coupl_.size(); mc++)
  {
    if (mesh_coupl_[mc]->has_moving_interface()) return true;
  }

  // loop all levelset coupling objects
  for (int lsc = 0; lsc < (int)levelset_coupl_.size(); lsc++)
  {
    if (levelset_coupl_[lsc]->has_moving_interface()) return true;
  }

  return false;
}

bool XFEM::ConditionManager::has_averaging_strategy(Inpar::XFEM::AveragingStrategy strategy)
{
  if (has_level_set_coupling())
  {
    for (size_t il = 0; il < levelset_coupl_.size(); ++il)
    {
      if (levelset_coupl_[il]->get_averaging_strategy() == strategy) return true;
    }
  }

  if (has_mesh_coupling())
  {
    for (size_t im = 0; im < mesh_coupl_.size(); ++im)
    {
      if (mesh_coupl_[im]->get_averaging_strategy() == strategy) return true;
    }
  }

  return false;
}

void XFEM::ConditionManager::get_volume_cell_material(Core::Elements::Element* actele,
    std::shared_ptr<Core::Mat::Material>& mat, const Cut::VolumeCell* vc)
{
  XFEM::Utils::get_volume_cell_material(actele, mat, vc->position());
}

void XFEM::ConditionManager::get_interface_master_material(Core::Elements::Element* actele,
    std::shared_ptr<Core::Mat::Material>& mat, const Cut::VolumeCell* vc)
{
  XFEM::Utils::get_volume_cell_material(actele, mat, vc->position());
}

void XFEM::ConditionManager::get_interface_slave_material(
    Core::Elements::Element* actele, std::shared_ptr<Core::Mat::Material>& mat, int coup_sid)
{
  if (is_mesh_coupling(coup_sid))
  {
    int mc = get_mesh_coupling_index(coup_sid);
    mesh_coupl_[mc]->get_interface_slave_material(actele, mat);
  }
  else if (is_level_set_coupling(coup_sid))
  {
    int lc = get_level_set_coupling_index(actele->id());
    levelset_coupl_[lc]->get_interface_slave_material(actele, mat);
  }
  else
    FOUR_C_THROW(
        "The coupling-side id: {} does not correspond to a mesh or levelset coupling object.",
        coup_sid);
}

bool XFEM::ConditionManager::initialize_fluid_state(std::shared_ptr<Cut::CutWizard> cutwizard,
    std::shared_ptr<Core::FE::Discretization> fluiddis,
    std::shared_ptr<XFEM::ConditionManager> condition_manager,
    std::shared_ptr<Teuchos::ParameterList> fluidparams)
{
  for (int m = 0; m < num_mesh_coupling(); m++)
  {
    std::shared_ptr<MeshCouplingFSI> mc_fsi =
        std::dynamic_pointer_cast<XFEM::MeshCouplingFSI>(mesh_coupl_[m]);
    std::shared_ptr<MeshCouplingFPI> mc_fpi =
        std::dynamic_pointer_cast<XFEM::MeshCouplingFPI>(mesh_coupl_[m]);
    if (mc_fsi != nullptr)
    {
      if (mc_fsi->initialize_fluid_state(cutwizard, fluiddis, condition_manager, fluidparams))
        return true;
    }
    if (mc_fpi != nullptr)
    {
      if (mc_fpi->initialize_fluid_state(cutwizard, fluiddis, condition_manager, fluidparams))
        return true;
    }
  }
  return false;
}

// Get Boundary Cell Clone Information <clone_coup_idx, clone_coup_sid>
std::vector<std::pair<int, int>> XFEM::ConditionManager::get_bc_clone_information(
    const int coup_sid, const int back_eid, int coup_idx)
{
  std::vector<std::pair<int, int>> BCCloneInformationvector;
  if (coup_idx == -1) coup_idx = get_coupling_index(coup_sid, back_eid);
  std::shared_ptr<CouplingBase> coupling = get_coupling_by_idx(coup_idx);
  if (coupling != nullptr)
  {
    // if there are other reasons than mcfpi ... feel free to add your case just here
    std::shared_ptr<MeshCouplingFPI> mcfpicoupling =
        std::dynamic_pointer_cast<MeshCouplingFPI>(coupling);
    if (mcfpicoupling != nullptr)
    {
      if (mcfpicoupling
              ->cut_geometry())  // if this is the ps_ps_block which was loaded into the CUT
                                 // --> clone from this
      {
        BCCloneInformationvector.push_back(std::pair<int, int>(coup_idx + 1,
            coup_sid + mesh_coupl_start_gid_[coup_idx + 1] - mesh_coupl_start_gid_[coup_idx]));
        BCCloneInformationvector.push_back(std::pair<int, int>(coup_idx + 2,
            coup_sid + mesh_coupl_start_gid_[coup_idx + 2] - mesh_coupl_start_gid_[coup_idx]));
        BCCloneInformationvector.push_back(std::pair<int, int>(coup_idx + 3,
            coup_sid + mesh_coupl_start_gid_[coup_idx + 3] - mesh_coupl_start_gid_[coup_idx]));
        return BCCloneInformationvector;
      }
      else
        FOUR_C_THROW("get_bc_clone_information: Try to clone from FPI MC != PoroStructure?");
    }
    else
      return BCCloneInformationvector;
  }
  else
    FOUR_C_THROW("get_bc_clone_information: Coupling is empty!");
  return BCCloneInformationvector;
}

Core::Elements::Element* XFEM::ConditionManager::get_coupling_element(
    const int coup_sid,  ///< the overall global coupling side id
    Core::Elements::Element* ele)
{
  if (is_mesh_coupling(coup_sid))
  {
    // get the mesh coupling object index
    const int mc_idx = get_mesh_coupling_index(coup_sid);

    // compute the side id w.r.t the cutter discretization the side belongs to
    const int cutterdis_sid = get_cutter_dis_ele_id(coup_sid, mc_idx);

    // get the boundary discretization, the side belongs to
    return mesh_coupl_[mc_idx]->get_coupling_element(cutterdis_sid);
  }
  else if (is_level_set_coupling(coup_sid))
  {
    // coupling of element with the element itself!
    const int coupldis_eid = ele->id();

    // get the level-set coupling object index for given background element
    const int lsc_idx = get_level_set_coupling_index(coupldis_eid);

    return levelset_coupl_[lsc_idx]->get_coupling_element(coupldis_eid);
  }
  else
    FOUR_C_THROW(
        "there is no valid mesh-/levelset-coupling condition object for side: {}", coup_sid);


  return nullptr;
}

void XFEM::ConditionManager::get_coupling_ele_location_vector(
    const int coup_sid, std::vector<int>& patchlm)
{
  int mc = get_mesh_coupling_index(coup_sid);
  int sid = get_cutter_dis_ele_id(coup_sid, mc);

  mesh_coupl_[mc]->get_coupling_ele_location_vector(sid, patchlm);
  return;
}

// Get the average weights from the coupling objects
// comment: as soon as we start doing mesh and levelset coupling with overlapping interfaces, we
// need to provide the position of the volumecells down here as well to choose the correct material.
// atm the position this is hardcoded in the coupling objects
// (all assume Cut::Point::outside, except TwoPhaseFlow Master outside, Slave inside)
void XFEM::ConditionManager::get_average_weights(
    const int coup_sid,              ///< the overall global coupling side id
    Core::Elements::Element* xfele,  ///< xfluid ele
    double& kappa_m,                 ///< Weight parameter (parameter +/master side)
    double& kappa_s,                 ///< Weight parameter (parameter -/slave  side)
    bool& non_xfluid_coupling)
{
  Core::Elements::Element* coup_ele = get_coupling_element(coup_sid, xfele);
  const int coup_idx = get_coupling_index(coup_sid, xfele->id());

  get_coupling_by_idx(coup_idx)->get_average_weights(
      xfele, coup_ele, kappa_m, kappa_s, non_xfluid_coupling);

  return;
}

/*--------------------------------------------------------------------------------
 * compute viscous part of Nitsche's penalty term scaling for Nitsche's method
 *--------------------------------------------------------------------------------*/
void XFEM::ConditionManager::get_visc_penalty_stabfac(
    const int coup_sid,              ///< the overall global coupling side id
    Core::Elements::Element* xfele,  ///< xfluid ele
    const double& kappa_m,           ///< Weight parameter (parameter +/master side)
    const double& kappa_s,           ///< Weight parameter (parameter -/slave  side)
    const double& inv_h_k,           ///< the inverse characteristic element length h_k
    const Discret::Elements::FluidEleParameterXFEM*
        params,                     ///< parameterlist which specifies interface configuration
    double& NIT_visc_stab_fac,      ///< viscous part of Nitsche's penalty term
    double& NIT_visc_stab_fac_tang  ///< viscous part of Nitsche's penalty term in tang direction
)
{
  Core::Elements::Element* coup_ele = get_coupling_element(coup_sid, xfele);
  const int coup_idx = get_coupling_index(coup_sid, xfele->id());

  get_coupling_by_idx(coup_idx)->get_visc_penalty_stabfac(xfele, coup_ele, kappa_m, kappa_s,
      inv_h_k, params, NIT_visc_stab_fac, NIT_visc_stab_fac_tang);

  return;
}

/*--------------------------------------------------------------------------------
 * get the estimation of the penalty scaling in Nitsche's method from the trace inequality for a
 *specific coupling side
 *--------------------------------------------------------------------------------*/
double XFEM::ConditionManager::get_trace_estimate_max_eigenvalue(
    const int coup_sid  ///< the overall global coupling side id
)
{
  // get the mesh coupling object index
  const int mc_idx = get_mesh_coupling_index(coup_sid);
  // compute the side id w.r.t the cutter discretization the side belongs to
  const int cutterdis_sid = get_cutter_dis_ele_id(coup_sid, mc_idx);
  // get the boundary discretization, the side belongs to
  std::shared_ptr<MeshVolCoupling> mvolcoupling =
      std::dynamic_pointer_cast<MeshVolCoupling>(mesh_coupl_[mc_idx]);
  if (mvolcoupling == nullptr) FOUR_C_THROW("Cast to MeshVolCoupling failed!");

  return mvolcoupling->get_estimate_nitsche_trace_max_eigenvalue(
      mvolcoupling->get_side(cutterdis_sid));
}

FOUR_C_NAMESPACE_CLOSE
