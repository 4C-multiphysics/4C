// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_coupling_volmortar.hpp"

#include "4C_coupling_volmortar_cell.hpp"
#include "4C_coupling_volmortar_defines.hpp"
#include "4C_coupling_volmortar_integrator.hpp"
#include "4C_coupling_volmortar_utils.hpp"
#include "4C_cut_cutwizard.hpp"
#include "4C_cut_elementhandle.hpp"
#include "4C_cut_volumecell.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_dofset_predefineddofnumber.hpp"
#include "4C_fem_geometry_searchtree.hpp"
#include "4C_fem_geometry_searchtree_service.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_mortar_calc_utils.hpp"
#include "4C_mortar_coupling3d.hpp"
#include "4C_mortar_element.hpp"
#include "4C_mortar_utils.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_pairedvector.hpp"
#include "4C_utils_parameter_list.hpp"

#include <Teuchos_Time.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  ctor (public)                                            farah 10/13|
 *----------------------------------------------------------------------*/
Coupling::VolMortar::VolMortarCoupl::VolMortarCoupl(int dim,
    std::shared_ptr<Core::FE::Discretization> dis1,  // on Omega_1
    std::shared_ptr<Core::FE::Discretization> dis2,  // on Omega_2
    const Teuchos::ParameterList& volmortar_parameters,
    const Teuchos::ParameterList& cut_parameters,
    std::vector<int>* coupleddof12,  // 2-->1
    std::vector<int>* coupleddof21,  // 1-->2
    std::pair<int, int>* dofset12,   // 2-->1
    std::pair<int, int>* dofset21,   // 1-->2
    std::shared_ptr<Coupling::VolMortar::Utils::DefaultMaterialStrategy>
        materialstrategy  // strategy for element information transfer
    )
    : dim_(dim),
      cut_params_(cut_parameters),
      dis1_(dis1),
      dis2_(dis2),
      materialstrategy_(materialstrategy)
{
  // check
  if (not dis1_->filled() or not dis2_->filled())
    FOUR_C_THROW(
        "ERROR: fill_complete() has to be called on both discretizations before setup of "
        "VolMortarCoupl");

  // its the same communicator for all discr.
  comm_ = dis1->get_comm();
  myrank_ = Core::Communication::my_mpi_rank(comm_);

  // define dof sets
  if (dofset21 == nullptr)
    dofset21_ = std::pair<int, int>(1, 0);
  else
    dofset21_ = *dofset21;

  if (dofset12 == nullptr)
    dofset12_ = std::pair<int, int>(1, 0);
  else
    dofset12_ = *dofset12;

  const int* rownodes1 = dis1->node_row_map()->my_global_elements();
  const int numrownode1 = dis1->node_row_map()->num_my_elements();

  const int* rownodes2 = dis2->node_row_map()->my_global_elements();
  const int numrownode2 = dis2->node_row_map()->num_my_elements();

  // determine row map of projection operators
  build_maps(dis1, p12_dofrowmap_, coupleddof21, rownodes1, numrownode1, dofset12_.first);

  build_maps(dis2, p21_dofrowmap_, coupleddof12, rownodes2, numrownode2, dofset21_.first);

  // determine domain map of projection operators
  build_maps(dis1, p21_dofdomainmap_, coupleddof21, rownodes1, numrownode1, dofset21_.second);
  build_maps(dis2, p12_dofdomainmap_, coupleddof12, rownodes2, numrownode2, dofset12_.second);

  const int* colnodes1 = dis1->node_col_map()->my_global_elements();
  const int numcolnode1 = dis1->node_col_map()->num_my_elements();

  const int* colnodes2 = dis2->node_col_map()->my_global_elements();
  const int numcolnode2 = dis2->node_col_map()->num_my_elements();

  // determine column map of projection operators
  build_maps(dis1, p21_dofcolmap_, coupleddof21, colnodes1, numcolnode1, dofset21_.second);
  build_maps(dis2, p12_dofcolmap_, coupleddof12, colnodes2, numcolnode2, dofset12_.second);

  // get required parameter list
  read_and_check_input(volmortar_parameters);

  // init dop normals
  init_dop_normals();

  // init aux normal TODO: no fixed direction!!! ONLY FOR 2D CASE !!!
  auxn_[0] = 0.0;
  auxn_[1] = 0.0;
  auxn_[2] = 1.0;

  // initialize the counter
  polygoncounter_ = 0;
  cellcounter_ = 0;
  inteles_ = 0;
  volume_ = 0.0;

  return;
}

/*----------------------------------------------------------------------*
 |  Build maps based on coupling dofs                        farah 03/15|
 *----------------------------------------------------------------------*/
void Coupling::VolMortar::VolMortarCoupl::build_maps(std::shared_ptr<Core::FE::Discretization>& dis,
    std::shared_ptr<const Core::LinAlg::Map>& dofmap, const std::vector<int>* coupleddof,
    const int* nodes, int numnode, int dofset)
{
  std::vector<int> dofmapvec;
  std::map<int, std::vector<int>> dofs;

  const bool couplalldofs = (coupleddof == nullptr);

  unsigned numcoupleddofs = 0;
  if (not couplalldofs)
  {
    // count number of dofs per node that are coupled
    for (unsigned j = 0; j < coupleddof->size(); ++j)
      if ((*coupleddof)[j] == 1) numcoupleddofs++;
  }
  else
    numcoupleddofs = dis->num_dof(dofset, dis->l_row_node(0));

  for (int i = 0; i < numnode; ++i)
  {
    const Core::Nodes::Node* actnode = dis->g_node(nodes[i]);

    const std::vector<int> dof = dis->dof(dofset, actnode);
    if (numcoupleddofs > dof.size())
      FOUR_C_THROW("got just {} dofs at node {} (lid={}) but expected {}", dof.size(), nodes[i], i,
          numcoupleddofs);

    for (unsigned j = 0; j < dof.size(); ++j)
    {
      if (not couplalldofs)
        if ((*coupleddof)[j] == 0) continue;

      dofmapvec.push_back(dof[j]);
    }
  }
  // dof map is the original, unpermuted distribution of dofs
  dofmap = std::make_shared<Core::LinAlg::Map>(-1, dofmapvec.size(), dofmapvec.data(), 0, comm_);

  return;
}

/*----------------------------------------------------------------------*
 |  Evaluate (public)                                        farah 10/13|
 *----------------------------------------------------------------------*/
void Coupling::VolMortar::VolMortarCoupl::evaluate_volmortar()
{
  /***********************************************************
   * Welcome                                                 *
   ***********************************************************/
  if (myrank_ == 0)
  {
    std::cout << "**************************************************" << std::endl;
    std::cout << "*****     Welcome to Core::VOLMORTAR-Coupling!     *****" << std::endl;
    std::cout << "**************************************************" << std::endl;
  }

  // time measurement
  Core::Communication::barrier(comm_);
  const double t_start = Teuchos::Time::wallTime();

  /***********************************************************
   * Check initial residuum and perform mesh init             *
   ***********************************************************/
  // check_initial_residuum();
  // mesh initialization procedure
  if (params().get<bool>("MESH_INIT")) mesh_init();

  /***********************************************************
   * initialize global matrices                              *
   ***********************************************************/
  initialize();

  /***********************************************************
   * Segment-based integration                               *
   ***********************************************************/
  if (Teuchos::getIntegralValue<IntType>(params(), "INTTYPE") == inttype_segments)
    evaluate_segments();

  /***********************************************************
   * Element-based Integration                               *
   ***********************************************************/
  else if (Teuchos::getIntegralValue<IntType>(params(), "INTTYPE") == inttype_elements)
    evaluate_elements();

  // no other possibility
  else
    FOUR_C_THROW("ERROR: Chosen INTTYPE not provided");

  /***********************************************************
   * complete global matrices and create projection operator *
   ***********************************************************/
  complete();
  create_projection_operator();

  /**************************************************
   * Bye                                            *
   **************************************************/
  Core::Communication::barrier(comm_);
  const double inttime = Teuchos::Time::wallTime() - t_start;

  if (myrank_ == 0)
  {
    std::cout << "**************************************************" << std::endl;
    std::cout << "*****       Core::VOLMORTAR-Coupling Done!!!       *****" << std::endl;
    std::cout << "**************************************************" << std::endl;

    // output
    std::cout << "required evaluation time: " << inttime << std::endl;
    //  std::cout << "Polyogns/Polyhedra = " << polygoncounter_ << std::endl;
    //  std::cout << "Created Cells      = " << cellcounter_    << std::endl;
    //  std::cout << "Integr. Elements   = " << inteles_        << std::endl;
    std::cout << "Integrated Volume  = " << volume_ << "\n" << std::endl;
  }

  // reset counter
  polygoncounter_ = 0;
  cellcounter_ = 0;
  inteles_ = 0;
  volume_ = 0.0;

  // coupling done
  return;
}

/*----------------------------------------------------------------------*
 |  Init normals for Dop calculation                         farah 05/14|
 *----------------------------------------------------------------------*/
void Coupling::VolMortar::VolMortarCoupl::init_dop_normals()
{
  dopnormals_(0, 0) = 1.0;
  dopnormals_(0, 1) = 0.0;
  dopnormals_(0, 2) = 0.0;

  dopnormals_(1, 0) = 0.0;
  dopnormals_(1, 1) = 1.0;
  dopnormals_(1, 2) = 0.0;

  dopnormals_(2, 0) = 0.0;
  dopnormals_(2, 1) = 0.0;
  dopnormals_(2, 2) = 1.0;

  dopnormals_(3, 0) = 1.0;
  dopnormals_(3, 1) = 1.0;
  dopnormals_(3, 2) = 0.0;

  dopnormals_(4, 0) = 1.0;
  dopnormals_(4, 1) = 0.0;
  dopnormals_(4, 2) = 1.0;

  dopnormals_(5, 0) = 0.0;
  dopnormals_(5, 1) = 1.0;
  dopnormals_(5, 2) = 1.0;

  dopnormals_(6, 0) = 1.0;
  dopnormals_(6, 1) = 0.0;
  dopnormals_(6, 2) = -1.0;

  dopnormals_(7, 0) = 1.0;
  dopnormals_(7, 1) = -1.0;
  dopnormals_(7, 2) = 0.0;

  dopnormals_(8, 0) = 0.0;
  dopnormals_(8, 1) = 1.0;
  dopnormals_(8, 2) = -1.0;

  return;
}
/*----------------------------------------------------------------------*
 |  Init search tree                                         farah 05/14|
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Geo::SearchTree> Coupling::VolMortar::VolMortarCoupl::init_search(
    std::shared_ptr<Core::FE::Discretization> searchdis)
{
  // init current positions
  std::map<int, Core::LinAlg::Matrix<3, 1>> currentpositions;

  for (int lid = 0; lid < searchdis->num_my_col_elements(); ++lid)
  {
    Core::Elements::Element* sele = searchdis->l_col_element(lid);

    // calculate slabs for every node on every element
    for (int k = 0; k < sele->num_node(); k++)
    {
      Core::Nodes::Node* node = sele->nodes()[k];
      Core::LinAlg::Matrix<3, 1> currpos;

      const auto x = node->x();
      currpos(0) = x[0];
      currpos(1) = x[1];
      if (x.size() > 2) currpos(2) = x[2];

      currentpositions[node->id()] = currpos;
    }
  }

  // init of 3D search tree
  std::shared_ptr<Core::Geo::SearchTree> searchTree = std::make_shared<Core::Geo::SearchTree>(5);

  // find the bounding box of the elements and initialize the search tree
  const Core::LinAlg::Matrix<3, 2> rootBox =
      Core::Geo::get_xaab_bof_dis(*searchdis, currentpositions);
  searchTree->initialize_tree(rootBox, *searchdis, Core::Geo::TreeType(Core::Geo::OCTTREE));

  return searchTree;
}

/*----------------------------------------------------------------------*
 |  Calculate Dops for background mesh                       farah 05/14|
 *----------------------------------------------------------------------*/
std::map<int, Core::LinAlg::Matrix<9, 2>> Coupling::VolMortar::VolMortarCoupl::calc_background_dops(
    std::shared_ptr<Core::FE::Discretization> searchdis)
{
  std::map<int, Core::LinAlg::Matrix<9, 2>> currentKDOPs;

  for (int lid = 0; lid < searchdis->num_my_col_elements(); ++lid)
  {
    Core::Elements::Element* sele = searchdis->l_col_element(lid);

    currentKDOPs[sele->id()] = calc_dop(*sele);
  }

  return currentKDOPs;
}

/*----------------------------------------------------------------------*
 |  Calculate Dop for one Element                            farah 05/14|
 *----------------------------------------------------------------------*/
Core::LinAlg::Matrix<9, 2> Coupling::VolMortar::VolMortarCoupl::calc_dop(
    Core::Elements::Element& ele)
{
  Core::LinAlg::Matrix<9, 2> dop;

  // calculate slabs
  for (int j = 0; j < 9; j++)
  {
    // initialize slabs
    dop(j, 0) = 1.0e12;
    dop(j, 1) = -1.0e12;
  }

  // calculate slabs for every node on every element
  for (int k = 0; k < ele.num_node(); k++)
  {
    Core::Nodes::Node* node = ele.nodes()[k];

    // get current node position
    std::array<double, 3> pos = {0.0, 0.0, 0.0};
    for (int j = 0; j < dim_; ++j) pos[j] = node->x()[j];

    // calculate slabs
    for (int j = 0; j < 9; j++)
    {
      //= ax+by+cz=d/sqrt(aa+bb+cc)
      double num =
          dopnormals_(j, 0) * pos[0] + dopnormals_(j, 1) * pos[1] + dopnormals_(j, 2) * pos[2];
      double denom =
          sqrt((dopnormals_(j, 0) * dopnormals_(j, 0)) + (dopnormals_(j, 1) * dopnormals_(j, 1)) +
               (dopnormals_(j, 2) * dopnormals_(j, 2)));
      double dcurrent = num / denom;

      if (dcurrent > dop(j, 1)) dop(j, 1) = dcurrent;
      if (dcurrent < dop(j, 0)) dop(j, 0) = dcurrent;
    }
  }

  return dop;
}

/*----------------------------------------------------------------------*
 |  Perform searching procedure                              farah 05/14|
 *----------------------------------------------------------------------*/
std::vector<int> Coupling::VolMortar::VolMortarCoupl::search(Core::Elements::Element& ele,
    std::shared_ptr<Core::Geo::SearchTree> SearchTree,
    std::map<int, Core::LinAlg::Matrix<9, 2>>& currentKDOPs)
{
  // vector of global ids of found elements
  std::vector<int> gids;
  gids.clear();
  std::set<int> gid;
  gid.clear();

  Core::LinAlg::Matrix<9, 2> queryKDOP;

  // calc dop for considered element
  queryKDOP = calc_dop(ele);

  //**********************************************************
  // search for near elements to the background node's coord
  SearchTree->search_collisions(currentKDOPs, queryKDOP, 0, gid);

  for (std::set<int>::iterator iter = gid.begin(); iter != gid.end(); ++iter) gids.push_back(*iter);

  return gids;
}

/*----------------------------------------------------------------------*
 |  Assign materials for both fields                         vuong 09/14|
 *----------------------------------------------------------------------*/
void Coupling::VolMortar::VolMortarCoupl::assign_materials()
{
  if (dis1_ == nullptr or dis2_ == nullptr)
    FOUR_C_THROW("no discretization for assigning materials!");

  // init search trees
  std::shared_ptr<Core::Geo::SearchTree> SearchTreeA = init_search(dis1_);
  std::shared_ptr<Core::Geo::SearchTree> SearchTreeB = init_search(dis2_);

  // calculate DOPs for search algorithm
  std::map<int, Core::LinAlg::Matrix<9, 2>> CurrentDOPsA = calc_background_dops(dis1_);
  std::map<int, Core::LinAlg::Matrix<9, 2>> CurrentDOPsB = calc_background_dops(dis2_);

  /**************************************************
   * loop over all Adis elements                    *
   **************************************************/
  for (int j = 0; j < dis1_->num_my_col_elements(); ++j)
  {
    // get master element
    Core::Elements::Element* Aele = dis1_->l_col_element(j);

    std::vector<int> found = search(*Aele, SearchTreeB, CurrentDOPsB);

    /***********************************************************
     * Assign materials                                        *
     ***********************************************************/
    materialstrategy_->assign_material2_to1(this, Aele, found, dis1_, dis2_);
  }

  /**************************************************
   * loop over all Bdis elements                    *
   **************************************************/
  for (int j = 0; j < dis2_->num_my_col_elements(); ++j)
  {
    // get master element
    Core::Elements::Element* Bele = dis2_->l_col_element(j);

    std::vector<int> found = search(*Bele, SearchTreeA, CurrentDOPsA);

    /***********************************************************
     * Assign materials                                        *
     ***********************************************************/
    materialstrategy_->assign_material1_to2(this, Bele, found, dis1_, dis2_);
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Calculate trafo matrix for quadr. elements               farah 05/14|
 *----------------------------------------------------------------------*/
std::vector<int> Coupling::VolMortar::VolMortarCoupl::get_adjacent_nodes(
    Core::FE::CellType shape, int& lid)
{
  // vector of adjacent node ids
  std::vector<int> ids;

  switch (shape)
  {
    case Core::FE::CellType::hex20:
    {
      if (lid == 8)
      {
        ids.push_back(0);
        ids.push_back(1);
      }
      else if (lid == 9)
      {
        ids.push_back(1);
        ids.push_back(2);
      }
      else if (lid == 10)
      {
        ids.push_back(2);
        ids.push_back(3);
      }
      else if (lid == 11)
      {
        ids.push_back(3);
        ids.push_back(0);
      }
      else if (lid == 12)
      {
        ids.push_back(0);
        ids.push_back(4);
      }
      else if (lid == 13)
      {
        ids.push_back(1);
        ids.push_back(5);
      }
      else if (lid == 14)
      {
        ids.push_back(2);
        ids.push_back(6);
      }
      else if (lid == 15)
      {
        ids.push_back(3);
        ids.push_back(7);
      }
      else if (lid == 16)
      {
        ids.push_back(4);
        ids.push_back(5);
      }
      else if (lid == 17)
      {
        ids.push_back(5);
        ids.push_back(6);
      }
      else if (lid == 18)
      {
        ids.push_back(6);
        ids.push_back(7);
      }
      else if (lid == 19)
      {
        ids.push_back(4);
        ids.push_back(7);
      }
      else
        FOUR_C_THROW("ERROR: Given Id is wrong!!!");

      break;
    }
    case Core::FE::CellType::tet10:
    {
      if (lid == 4)
      {
        ids.push_back(0);
        ids.push_back(1);
      }
      else if (lid == 5)
      {
        ids.push_back(1);
        ids.push_back(2);
      }
      else if (lid == 6)
      {
        ids.push_back(0);
        ids.push_back(2);
      }
      else if (lid == 7)
      {
        ids.push_back(0);
        ids.push_back(3);
      }
      else if (lid == 8)
      {
        ids.push_back(1);
        ids.push_back(3);
      }
      else if (lid == 9)
      {
        ids.push_back(2);
        ids.push_back(3);
      }
      else
        FOUR_C_THROW("ERROR: Given Id is wrong!!!");

      break;
    }
    default:
      FOUR_C_THROW("ERROR: shape unknown\n");
      break;
  }

  return ids;
}
/*----------------------------------------------------------------------*
 |  Calculate trafo matrix for quadr. elements               farah 05/14|
 *----------------------------------------------------------------------*/
void Coupling::VolMortar::VolMortarCoupl::create_trafo_operator(Core::Elements::Element& ele,
    std::shared_ptr<Core::FE::Discretization> searchdis, bool dis, std::set<int>& donebefore)
{
  // trafo parameter
  const double alpha = 0.3;

  // ids for corner and edge nodes
  int corner_min = 0;
  int corner_max = 0;
  int edge_min = 0;
  int edge_max = 0;

  if (ele.shape() == Core::FE::CellType::hex20)
  {
    corner_min = 0;
    corner_max = 7;
    edge_min = 8;
    edge_max = 19;
  }
  else if (ele.shape() == Core::FE::CellType::tet10)
  {
    corner_min = 0;
    corner_max = 3;
    edge_min = 4;
    edge_max = 9;
  }
  else
    FOUR_C_THROW("ERROR: Unknown shape for trafo matrix");

  // loop over element nodes
  for (int i = 0; i < ele.num_node(); ++i)
  {
    Core::Nodes::Node* cnode = ele.nodes()[i];
    if (cnode->owner() != myrank_) continue;

    std::set<int>::iterator iter = donebefore.find(cnode->id());
    if (iter != donebefore.end()) continue;

    donebefore.insert(cnode->id());

    //************************************************
    // corner nodes
    if (i >= corner_min and i <= corner_max)
    {
      int nsdof = searchdis->num_dof(1, cnode);

      // loop over slave dofs
      for (int jdof = 0; jdof < nsdof; ++jdof)
      {
        int row = searchdis->dof(1, cnode, jdof);

        if (dis)
          t1_->assemble(1.0, row, row);
        else
          t2_->assemble(1.0, row, row);
      }
    }
    //************************************************
    // edge nodes
    else if (i >= edge_min and i <= edge_max)
    {
      // get ids
      std::vector<int> ids = get_adjacent_nodes(ele.shape(), i);

      int nsdof = searchdis->num_dof(1, cnode);

      // loop over dofs
      for (int jdof = 0; jdof < nsdof; ++jdof)
      {
        int row = searchdis->dof(1, cnode, jdof);

        // assemble diagonal entries
        if (dis)
          t1_->assemble(1.0 - 2.0 * alpha, row, row);
        else
          t2_->assemble(1.0 - 2.0 * alpha, row, row);

        // found ids
        for (int id = 0; id < (int)ids.size(); ++id)
        {
          Core::Nodes::Node* fnode = ele.nodes()[ids[id]];
          int nfdof = searchdis->num_dof(1, fnode);

          for (int fdof = 0; fdof < nfdof; ++fdof)
          {
            int col = searchdis->dof(1, fnode, fdof);

            if (jdof == fdof)
            {
              // assemble off-diagonal entries
              if (dis)
                t1_->assemble(alpha, row, col);
              else
                t2_->assemble(alpha, row, col);
            }
          }
        }
      }
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Consistent interpolation routine                         farah 06/14|
 *----------------------------------------------------------------------*/
void Coupling::VolMortar::VolMortarCoupl::evaluate_consistent_interpolation()
{
  /***********************************************************
   * Welcome                                                 *
   ***********************************************************/
  if (myrank_ == 0)
  {
    std::cout << "*********************************************************" << std::endl;
    std::cout << "***** Welcome to Consistent-Interpolation-Coupling! *****" << std::endl;
    std::cout << "*********************************************************" << std::endl;
  }

  /***********************************************************
   * Init P-matrices                                         *
   ***********************************************************/
  p12_ = std::make_shared<Core::LinAlg::SparseMatrix>(*p12_dofrowmap_, 10);
  p21_ = std::make_shared<Core::LinAlg::SparseMatrix>(*p21_dofrowmap_, 100);

  /***********************************************************
   * create search tree and current dops                     *
   ***********************************************************/
  std::shared_ptr<Core::Geo::SearchTree> SearchTreeA = init_search(dis1_);
  std::shared_ptr<Core::Geo::SearchTree> SearchTreeB = init_search(dis2_);
  std::map<int, Core::LinAlg::Matrix<9, 2>> CurrentDOPsA = calc_background_dops(dis1_);
  std::map<int, Core::LinAlg::Matrix<9, 2>> CurrentDOPsB = calc_background_dops(dis2_);

  /***********************************************************
   * Create P operators                                      *
   ***********************************************************/
  for (int i = 0; i < dis1_->num_my_col_nodes(); ++i)
  {
    // 1 map node into bele
    int gid = dis1_->node_col_map()->gid(i);
    Core::Nodes::Node* anode = dis1_->g_node(gid);

    // get found elements from other discr.
    std::vector<int> found =
        search(*anode->adjacent_elements()[0].user_element(), SearchTreeB, CurrentDOPsB);

    assemble_consistent_interpolation_p12(anode, found);
  }  // end node loop

  //================================================
  for (int i = 0; i < dis2_->num_my_col_nodes(); ++i)
  {
    // 1 map node into bele
    int gid = dis2_->node_col_map()->gid(i);
    Core::Nodes::Node* bnode = dis2_->g_node(gid);

    // get found elements from other discr.
    std::vector<int> found =
        search(*bnode->adjacent_elements()[0].user_element(), SearchTreeA, CurrentDOPsA);

    assemble_consistent_interpolation_p21(bnode, found);
  }  // end node loop

  /***********************************************************
   * Complete                                                *
   ***********************************************************/
  p12_->complete(*p12_dofdomainmap_, *p12_dofrowmap_);
  p21_->complete(*p21_dofdomainmap_, *p21_dofrowmap_);

  return;
}

/*----------------------------------------------------------------------*
 |  Element-based routine                                    farah 04/14|
 *----------------------------------------------------------------------*/
void Coupling::VolMortar::VolMortarCoupl::evaluate_elements()
{
  // output
  if (myrank_ == 0)
  {
    std::cout << "*****       Element-based Integration        *****" << std::endl;
    std::cout << "*****       Calc First Projector:            *****" << std::endl;
  }

  // init search trees
  std::shared_ptr<Core::Geo::SearchTree> SearchTreeA = init_search(dis1_);
  std::shared_ptr<Core::Geo::SearchTree> SearchTreeB = init_search(dis2_);

  // calculate DOPs for search algorithm
  std::map<int, Core::LinAlg::Matrix<9, 2>> CurrentDOPsA = calc_background_dops(dis1_);
  std::map<int, Core::LinAlg::Matrix<9, 2>> CurrentDOPsB = calc_background_dops(dis2_);

  /**************************************************
   * loop over all Adis elements                    *
   **************************************************/
  std::set<int> donebeforea;
  for (int j = 0; j < dis1_->num_my_col_elements(); ++j)
  {
    // print_status(j,false);

    // get master element
    Core::Elements::Element* Aele = dis1_->l_col_element(j);

    std::vector<int> found = search(*Aele, SearchTreeB, CurrentDOPsB);
    integrate_3d_ele_based_p12(*Aele, found);

    // create trafo operator for quadr. modification
    if (dualquad_ != dualquad_no_mod) create_trafo_operator(*Aele, dis1_, true, donebeforea);
  }

  // half-time output
  if (myrank_ == 0)
  {
    std::cout << "**************************************************" << std::endl;
    std::cout << "*****       Calc Second Projector:           *****" << std::endl;
  }

  /**************************************************
   * loop over all Bdis elements                    *
   **************************************************/
  std::set<int> donebeforeb;
  for (int j = 0; j < dis2_->num_my_col_elements(); ++j)
  {
    // print_status(j,true);

    // get master element
    Core::Elements::Element* Bele = dis2_->l_col_element(j);

    std::vector<int> found = search(*Bele, SearchTreeA, CurrentDOPsA);
    integrate_3d_ele_based_p21(*Bele, found);

    // create trafo operator for quadr. modification
    if (dualquad_ != dualquad_no_mod) create_trafo_operator(*Bele, dis2_, false, donebeforeb);
  }

  // stats
  inteles_ += dis1_->num_global_elements();
  inteles_ += dis2_->num_global_elements();
}

/*----------------------------------------------------------------------*
 |  Segment-based routine                                    farah 04/14|
 *----------------------------------------------------------------------*/
void Coupling::VolMortar::VolMortarCoupl::evaluate_segments()
{
  // create search tree and current dops
  std::shared_ptr<Core::Geo::SearchTree> SearchTreeB = init_search(dis2_);
  std::map<int, Core::LinAlg::Matrix<9, 2>> CurrentDOPsB = calc_background_dops(dis2_);

  /**************************************************
   * loop over all slave elements                   *
   **************************************************/
  for (int i = 0; i < dis1_->num_my_col_elements(); ++i)
  {
    // output of coupling status
    // print_status(i);

    // get slave element
    Core::Elements::Element* Aele = dis1_->l_col_element(i);

    // get found elements from other discr.
    std::vector<int> found = search(*Aele, SearchTreeB, CurrentDOPsB);

    /***********************************************************
     * Assign materials                                        *
     ***********************************************************/
    materialstrategy_->assign_material2_to1(this, Aele, found, dis1_, dis2_);

    /**************************************************
     * loop over all master elements                  *
     **************************************************/
    for (int foundeles = 0; foundeles < (int)found.size(); ++foundeles)
    {
      // get b element
      Core::Elements::Element* Bele = dis2_->g_element(found[foundeles]);

      /***********************************************************
       * Assign materials                                        *
       ***********************************************************/
      materialstrategy_->assign_material1_to2(
          this, Bele, std::vector<int>(1, Aele->id()), dis1_, dis2_);

      /**************************************************
       *                    2D                          *
       **************************************************/
      if (dim_ == 2)
      {
        evaluate_segments_2d(*Aele, *Bele);
      }

      /**************************************************
       *                    3D                          *
       **************************************************/
      else if (dim_ == 3)
      {
        evaluate_segments_3d(Aele, Bele);
      }

      /**************************************************
       * Wrong Dimension !!!                            *
       **************************************************/
      else
        FOUR_C_THROW("ERROR: Problem dimension is not correct!");

    }  // end master element loop
  }  // end slave element loop
}

/*----------------------------------------------------------------------*
 |  Segment-based routine 2D                                 farah 04/14|
 *----------------------------------------------------------------------*/
void Coupling::VolMortar::VolMortarCoupl::evaluate_segments_2d(
    Core::Elements::Element& Aele, Core::Elements::Element& Bele)
{
  // define polygon vertices
  static std::vector<Mortar::Vertex> slave_vertices;
  static std::vector<Mortar::Vertex> master_vertices;
  static std::vector<Mortar::Vertex> ClippedPolygon;
  static std::vector<std::shared_ptr<Mortar::IntCell>> cells;

  // clear old polygons
  slave_vertices.clear();
  master_vertices.clear();
  ClippedPolygon.clear();
  cells.clear();
  cells.resize(0);

  // build new polygons
  define_vertices_master(Bele, master_vertices);
  define_vertices_slave(Aele, slave_vertices);

  double tol = 1e-12;
  polygon_clipping_convex_hull(slave_vertices, master_vertices, ClippedPolygon, Aele, Bele, tol);
  int clipsize = (int)(ClippedPolygon.size());

  // proceed only if clipping polygon is at least a triangle
  if (clipsize < 3)
    return;
  else
    polygoncounter_ += 1;

  // triangulation
  bool success = delaunay_triangulation(cells, ClippedPolygon, tol);
  if (!success)
  {
    bool backup = center_triangulation(cells, ClippedPolygon, tol);
    if (!backup) FOUR_C_THROW("ERROR: Triangulation failed!");
  }

  cellcounter_ += (int)cells.size();

  // integrate cells
  integrate_2d(Aele, Bele, cells);
}

/*----------------------------------------------------------------------*
 |  Segment-based routine 3D                                 farah 04/14|
 *----------------------------------------------------------------------*/
void Coupling::VolMortar::VolMortarCoupl::evaluate_segments_3d(
    Core::Elements::Element* Aele, Core::Elements::Element* Bele)
{
  // check need element-based integration over sele:
  bool integrateA = check_ele_integration(*Aele, *Bele);

  // check need element-based integration over mele:
  bool integrateB = check_ele_integration(*Bele, *Aele);

  // check need for cut:
  bool performcut = check_cut(*Aele, *Bele);

  /**************************************************
   * Integrate element-based Sele                   *
   **************************************************/
  if (integrateA)
  {
    // integrate over Aele
    integrate_3d(*Aele, *Bele, 0);
  }
  /**************************************************
   * Integrate element-based Mele                   *
   **************************************************/
  else if (integrateB)
  {
    // integrate over Bele
    integrate_3d(*Aele, *Bele, 1);
  }
  /**************************************************
   * Start Cut                                      *
   **************************************************/
  else if (performcut)
  {
    // call cut and create integration cells
    try
    {
      bool switched_conf = false;
      perform_cut(Aele, Bele, switched_conf);
    }
    catch (Core::Exception& err1)
    {
      try
      {
        bool switched_conf = true;
        perform_cut(Bele, Aele, switched_conf);
      }
      catch (Core::Exception& err2)
      {
        std::cout << "runtime error 2 = " << err2.what() << std::endl;
      }
    }
  }  // end performcut
  else
  {
    // do nothing...
  }

  return;
}

/*----------------------------------------------------------------------*
 |  get parameters and check for validity                    farah 04/14|
 *----------------------------------------------------------------------*/
void Coupling::VolMortar::VolMortarCoupl::read_and_check_input(
    const Teuchos::ParameterList& volmortar_parameters)
{
  // check validity
  if (Teuchos::getIntegralValue<IntType>(volmortar_parameters, "INTTYPE") == inttype_segments)
  {
    if (myrank_ == 0)
    {
      std::cout
          << "WARNING: The chosen integration type for volmortar coupling requires cut procedure !"
          << std::endl;
      std::cout
          << "WARNING: The cut is up to now not able to exactly calculate the required segments!"
          << std::endl;
    }
  }

  if (volmortar_parameters.get<bool>("MESH_INIT") and
      Teuchos::getIntegralValue<IntType>(volmortar_parameters, "INTTYPE") == inttype_segments)
  {
    FOUR_C_THROW("ERROR: mesh_init only for ele-based integration!!!");
  }

  if (Teuchos::getIntegralValue<Coupling::VolMortar::Shapefcn>(volmortar_parameters, "SHAPEFCN") ==
      shape_std)
  {
    std::cout << "WARNING: Standard shape functions are employed! D is lumped!" << std::endl;
  }

  // merge to global parameter list
  params_.setParameters(volmortar_parameters);

  // get specific and frequently reused parameters
  dualquad_ = Teuchos::getIntegralValue<DualQuad>(params_, "DUALQUAD");
}

/*----------------------------------------------------------------------*
 |  check initial residuum                                   farah 04/14|
 *----------------------------------------------------------------------*/
void Coupling::VolMortar::VolMortarCoupl::check_initial_residuum()
{
  // create vectors of initial primary variables
  std::shared_ptr<Core::LinAlg::Vector<double>> var_A =
      Core::LinAlg::create_vector(*discret1()->dof_row_map(0), true);
  std::shared_ptr<Core::LinAlg::Vector<double>> var_B =
      Core::LinAlg::create_vector(*discret2()->dof_row_map(1), true);

  // solution
  std::shared_ptr<Core::LinAlg::Vector<double>> result_A =
      Core::LinAlg::create_vector(*discret2()->dof_row_map(1), true);
  std::shared_ptr<Core::LinAlg::Vector<double>> result_B =
      Core::LinAlg::create_vector(*discret2()->dof_row_map(1), true);

  // node positions for Discr A
  for (int i = 0; i < discret1()->num_my_row_elements(); ++i)
  {
    Core::Elements::Element* Aele = discret1()->l_row_element(i);
    for (int j = 0; j < Aele->num_node(); ++j)
    {
      Core::Nodes::Node* cnode = Aele->nodes()[j];
      int nsdof = discret1()->num_dof(0, cnode);

      // loop over slave dofs
      for (int jdof = 0; jdof < nsdof; ++jdof)
      {
        int id = discret1()->dof(0, cnode, jdof);
        double val = cnode->x()[jdof];

        var_A->replace_global_value(id, val);
      }
    }
  }

  // node positions for Discr B
  for (int i = 0; i < discret2()->num_my_row_elements(); ++i)
  {
    Core::Elements::Element* Bele = discret2()->l_row_element(i);
    for (int j = 0; j < Bele->num_node(); ++j)
    {
      Core::Nodes::Node* cnode = Bele->nodes()[j];
      int nsdof = discret2()->num_dof(1, cnode);

      // loop over slave dofs
      for (int jdof = 0; jdof < nsdof; ++jdof)
      {
        int id = discret2()->dof(1, cnode, jdof);
        double val = cnode->x()[jdof];

        var_B->replace_global_value(id, val);
      }
    }
  }

  // std::cout << "vector of slave positions " << *var_B << std::endl;

  // do: D*db - M*da
  // int err1 = dmatrixB_->Multiply(false, *var_B, *result_B);
  // int err2 = mmatrixB_->Multiply(false, *var_A, *result_A);

  // if(err1!=0 or err2!=0)
  //  FOUR_C_THROW("error");

  int err = p21_->multiply(false, *var_A, *result_A);
  if (err != 0) FOUR_C_THROW("error");

  // subtract both results
  result_A->update(-1.0, *var_B, 1.0);

  std::cout << "Result of init check= " << std::endl;
  result_A->print(std::cout);

  return;
}

/*----------------------------------------------------------------------*
 |  check initial residuum                                   farah 04/14|
 *----------------------------------------------------------------------*/
void Coupling::VolMortar::VolMortarCoupl::mesh_init()
{
  // create merged map:
  std::shared_ptr<Core::DOFSets::DofSetInterface> dofsetaux;
  dofsetaux = std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(dim_, 0, 0, true);
  int dofseta = dis1_->add_dof_set(dofsetaux);
  dofsetaux = std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(dim_, 0, 0, true);
  int dofsetb = dis2_->add_dof_set(dofsetaux);
  dis1_->fill_complete({
      .assign_degrees_of_freedom = true,
      .init_elements = false,
      .do_boundary_conditions = false,
  });
  dis2_->fill_complete({
      .assign_degrees_of_freedom = true,
      .init_elements = false,
      .do_boundary_conditions = false,
  });

  double omega = 1.0;
  double nu = 0.0;
  int maxiter = 1;

  // init zero residuum vector for old iteration
  std::shared_ptr<Core::LinAlg::Vector<double>> ResoldA =
      Core::LinAlg::create_vector(*discret1()->dof_row_map(dofseta), true);
  std::shared_ptr<Core::LinAlg::Vector<double>> ResoldB =
      Core::LinAlg::create_vector(*discret2()->dof_row_map(dofsetb), true);

  // output
  if (myrank_ == 0)
  {
    std::cout << "*****       MESHINIT       *****" << std::endl;
  }

  // global mesh init loop
  for (int mi = 0; mi < maxiter; mi++)
  {
    // init
    dmatrix_xa_ =
        std::make_shared<Core::LinAlg::SparseMatrix>(*discret1()->dof_row_map(dofseta), 10);
    mmatrix_xa_ =
        std::make_shared<Core::LinAlg::SparseMatrix>(*discret1()->dof_row_map(dofseta), 100);

    dmatrix_xb_ =
        std::make_shared<Core::LinAlg::SparseMatrix>(*discret2()->dof_row_map(dofsetb), 10);
    mmatrix_xb_ =
        std::make_shared<Core::LinAlg::SparseMatrix>(*discret2()->dof_row_map(dofsetb), 100);

    // output
    if (myrank_ == 0) std::cout << "*****       step " << mi << std::endl;

    // init search trees
    std::shared_ptr<Core::Geo::SearchTree> SearchTreeA = init_search(dis1_);
    std::shared_ptr<Core::Geo::SearchTree> SearchTreeB = init_search(dis2_);

    // calculate DOPs for search algorithm
    std::map<int, Core::LinAlg::Matrix<9, 2>> CurrentDOPsA = calc_background_dops(dis1_);
    std::map<int, Core::LinAlg::Matrix<9, 2>> CurrentDOPsB = calc_background_dops(dis2_);

    /**************************************************
     * loop over all Adis elements                    *
     **************************************************/
    std::set<int> donebeforea;
    for (int j = 0; j < dis1_->num_my_col_elements(); ++j)
    {
      // get master element
      Core::Elements::Element* Aele = dis1_->l_col_element(j);

      std::vector<int> found = search(*Aele, SearchTreeB, CurrentDOPsB);
      integrate_3d_ele_based_a_dis_mesh_init(*Aele, found, dofseta, dofsetb);
    }

    /**************************************************
     * loop over all Bdis elements                    *
     **************************************************/
    std::set<int> donebeforeb;
    for (int j = 0; j < dis2_->num_my_col_elements(); ++j)
    {
      // get master element
      Core::Elements::Element* Bele = dis2_->l_col_element(j);

      std::vector<int> found = search(*Bele, SearchTreeA, CurrentDOPsA);
      integrate_3d_ele_based_b_dis_mesh_init(*Bele, found, dofseta, dofsetb);
    }

    // complete...
    dmatrix_xa_->complete();
    mmatrix_xa_->complete(*discret2()->dof_row_map(dofsetb), *discret1()->dof_row_map(dofseta));
    dmatrix_xb_->complete();
    mmatrix_xb_->complete(*discret1()->dof_row_map(dofseta), *discret2()->dof_row_map(dofsetb));

    mergedmap_ = Core::LinAlg::merge_map(
        *discret1()->dof_row_map(dofseta), *discret2()->dof_row_map(dofsetb), false);
    std::shared_ptr<Core::LinAlg::Vector<double>> mergedsol =
        Core::LinAlg::create_vector(*mergedmap_);
    std::shared_ptr<Core::LinAlg::Vector<double>> mergedX =
        Core::LinAlg::create_vector(*mergedmap_);
    std::shared_ptr<Core::LinAlg::Vector<double>> mergedXa =
        Core::LinAlg::create_vector(*discret1()->dof_row_map(dofseta));
    std::shared_ptr<Core::LinAlg::Vector<double>> mergedXb =
        Core::LinAlg::create_vector(*discret2()->dof_row_map(dofsetb));

    for (int n = 0; n < dis1_->node_row_map()->num_my_elements(); ++n)
    {
      int gid = dis1_->node_row_map()->gid(n);
      Core::Nodes::Node* node = dis1_->g_node(gid);

      Core::LinAlg::SerialDenseVector pos(3);
      pos(0) = node->x()[0];
      pos(1) = node->x()[1];
      pos(2) = node->x()[2];

      std::vector<int> id(3);
      id[0] = dis1_->dof(dofseta, node, 0);
      id[1] = dis1_->dof(dofseta, node, 1);
      id[2] = dis1_->dof(dofseta, node, 2);

      std::vector<int> owner(3);
      owner[0] = node->owner();
      owner[1] = node->owner();
      owner[2] = node->owner();

      Core::LinAlg::assemble(*mergedX, pos, id, owner);
      Core::LinAlg::assemble(*mergedXa, pos, id, owner);
    }

    for (int n = 0; n < dis2_->node_row_map()->num_my_elements(); ++n)
    {
      int gid = dis2_->node_row_map()->gid(n);
      Core::Nodes::Node* node = dis2_->g_node(gid);

      Core::LinAlg::SerialDenseVector pos(3);
      pos(0) = node->x()[0];
      pos(1) = node->x()[1];
      pos(2) = node->x()[2];

      std::vector<int> id(3);
      id[0] = dis2_->dof(dofsetb, node, 0);
      id[1] = dis2_->dof(dofsetb, node, 1);
      id[2] = dis2_->dof(dofsetb, node, 2);

      std::vector<int> owner(3);
      owner[0] = node->owner();
      owner[1] = node->owner();
      owner[2] = node->owner();

      Core::LinAlg::assemble(*mergedX, pos, id, owner);
      Core::LinAlg::assemble(*mergedXb, pos, id, owner);
    }

    //--------------------------------------------------------------
    //--------------------------------------------------------------
    // Check:
    std::shared_ptr<Core::LinAlg::Vector<double>> solDA =
        Core::LinAlg::create_vector(*discret1()->dof_row_map(dofseta));
    std::shared_ptr<Core::LinAlg::Vector<double>> solMA =
        Core::LinAlg::create_vector(*discret1()->dof_row_map(dofseta));

    int err = 0;
    err = dmatrix_xa_->multiply(false, *mergedXa, *solDA);
    if (err != 0) FOUR_C_THROW("stop");

    err = mmatrix_xa_->multiply(false, *mergedXb, *solMA);
    if (err != 0) FOUR_C_THROW("stop");

    std::shared_ptr<Core::LinAlg::Vector<double>> solDB =
        Core::LinAlg::create_vector(*discret2()->dof_row_map(dofsetb));
    std::shared_ptr<Core::LinAlg::Vector<double>> solMB =
        Core::LinAlg::create_vector(*discret2()->dof_row_map(dofsetb));

    err = dmatrix_xb_->multiply(false, *mergedXb, *solDB);
    if (err != 0) FOUR_C_THROW("stop");

    err = mmatrix_xb_->multiply(false, *mergedXa, *solMB);
    if (err != 0) FOUR_C_THROW("stop");

    solDA->update(-1.0, *solMA, 1.0);
    solDB->update(-1.0, *solMB, 1.0);

    double ra = 0.0;
    double rb = 0.0;

    // Residuum k+1
    solDA->norm_2(&ra);
    solDB->norm_2(&rb);

    std::cout << "ra= " << ra << "    rb= " << rb << std::endl;

    if (ra < 1e-10 and rb < 1e-10) return;

    solDA->scale(-1.0 * omega);
    solDB->scale(-1.0 * omega);

    // Calc relaxation parameter nu:
    if (mi > 0)
    {
      double fac = 0.0;
      std::shared_ptr<Core::LinAlg::Vector<double>> DiffA =
          Core::LinAlg::create_vector(*discret1()->dof_row_map(dofseta), true);
      std::shared_ptr<Core::LinAlg::Vector<double>> DiffB =
          Core::LinAlg::create_vector(*discret2()->dof_row_map(dofsetb), true);
      DiffA->update(1.0, *solDA, 0.0);
      DiffA->update(-1.0, *ResoldA, 1.0);
      DiffB->update(1.0, *solDB, 0.0);
      DiffB->update(-1.0, *ResoldB, 1.0);

      double topa = 0.0;
      DiffA->dot(*solDA, &topa);

      double topb = 0.0;
      DiffB->dot(*solDB, &topb);

      double top = -(topa + topb);

      double downa = 0.0;
      DiffA->dot(*DiffA, &downa);

      double downb = 0.0;
      DiffB->dot(*DiffB, &downb);
      double down = (downa + downb);

      fac = top / down;
      std::cout << "fac= " << fac << std::endl;
      nu = nu + (nu - 1.0) * fac;
    }

    ResoldA->update(1.0, *solDA, 0.0);
    ResoldB->update(1.0, *solDB, 0.0);
    //--------------------------------------------------------------
    //--------------------------------------------------------------
    // relaxation parameter
    //    omega = 1.0 - nu;
    omega = 1.0;

    Core::LinAlg::SparseMatrix k(*mergedmap_, 100, false, true);
    k.add(*dmatrix_xa_, false, -1.0 * omega, 1.0);
    k.add(*mmatrix_xa_, false, 1.0 * omega, 1.0);
    k.add(*dmatrix_xb_, false, -1.0 * omega, 1.0);
    k.add(*mmatrix_xb_, false, 1.0 * omega, 1.0);

    Core::LinAlg::Vector<double> ones(*mergedmap_);
    ones.put_scalar(1.0);
    Core::LinAlg::SparseMatrix onesdiag(ones);
    onesdiag.complete();
    k.add(onesdiag, false, 1.0, 1.0);
    k.complete();

    // solve with default solver

    Teuchos::ParameterList solvparams;
    Core::Utils::add_enum_class_to_parameter_list<Core::LinearSolver::SolverType>(
        "SOLVER", Core::LinearSolver::SolverType::umfpack, solvparams);
    Core::LinAlg::Solver solver(solvparams, comm_, nullptr, Core::IO::Verbositylevel::standard);

    Core::LinAlg::SolverParams solver_params;
    solver_params.refactor = true;
    solver.solve(Core::Utils::shared_ptr_from_ref(k), mergedsol, mergedX, solver_params);

    std::shared_ptr<Core::LinAlg::Vector<double>> sola =
        Core::LinAlg::create_vector(*discret1()->dof_row_map(dofseta));
    std::shared_ptr<Core::LinAlg::Vector<double>> solb =
        Core::LinAlg::create_vector(*discret2()->dof_row_map(dofsetb));

    Core::LinAlg::MapExtractor mapext(*mergedmap_,
        std::make_shared<Core::LinAlg::Map>(*(discret1()->dof_row_map(dofseta))),
        std::make_shared<Core::LinAlg::Map>(*(discret2()->dof_row_map(dofsetb))));
    mapext.extract_cond_vector(*mergedsol, *sola);
    mapext.extract_other_vector(*mergedsol, *solb);

    //--------------------------------------------------------------
    // Post Processing... Node Relocation
    // node positions
    for (int i = 0; i < discret1()->num_my_row_elements(); ++i)
    {
      Core::Elements::Element* Aele = discret1()->l_row_element(i);
      for (int j = 0; j < Aele->num_node(); ++j)
      {
        Core::Nodes::Node* cnode = Aele->nodes()[j];
        int nsdof = discret1()->num_dof(dofseta, cnode);
        std::vector<double> nvector(3);

        // loop over slave dofs
        for (int jdof = 0; jdof < nsdof; ++jdof)
        {
          const int lid = sola->get_map().lid(discret1()->dof(dofseta, cnode, jdof));
          nvector[jdof] = (*sola)[lid] - cnode->x()[jdof];
        }
        cnode->change_pos(nvector);
      }
    }

    for (int i = 0; i < discret2()->num_my_row_elements(); ++i)
    {
      Core::Elements::Element* Bele = discret2()->l_row_element(i);
      for (int j = 0; j < Bele->num_node(); ++j)
      {
        Core::Nodes::Node* cnode = Bele->nodes()[j];
        int nsdof = discret2()->num_dof(dofsetb, cnode);
        std::vector<double> nvector(3);

        // loop over slave dofs
        for (int jdof = 0; jdof < nsdof; ++jdof)
        {
          const int lid = solb->get_map().lid(discret2()->dof(dofsetb, cnode, jdof));
          nvector[jdof] = (*solb)[lid] - cnode->x()[jdof];
        }
        cnode->change_pos(nvector);
      }
    }

    dis1_->fill_complete({
        .assign_degrees_of_freedom = false,
        .init_elements = true,
        .do_boundary_conditions = true,
    });
    dis2_->fill_complete({
        .assign_degrees_of_freedom = false,
        .init_elements = true,
        .do_boundary_conditions = true,
    });

    // last check:
    std::shared_ptr<Core::LinAlg::Vector<double>> checka =
        Core::LinAlg::create_vector(*discret1()->dof_row_map(dofseta));
    std::shared_ptr<Core::LinAlg::Vector<double>> checkb =
        Core::LinAlg::create_vector(*discret2()->dof_row_map(dofsetb));

    for (int n = 0; n < dis1_->node_row_map()->num_my_elements(); ++n)
    {
      int gid = dis1_->node_row_map()->gid(n);
      Core::Nodes::Node* node = dis1_->g_node(gid);

      Core::LinAlg::SerialDenseVector pos(3);
      pos(0) = node->x()[0];
      pos(1) = node->x()[1];
      pos(2) = node->x()[2];

      std::vector<int> id(3);
      id[0] = dis1_->dof(dofseta, node, 0);
      id[1] = dis1_->dof(dofseta, node, 1);
      id[2] = dis1_->dof(dofseta, node, 2);

      std::vector<int> owner(3);
      owner[0] = node->owner();
      owner[1] = node->owner();
      owner[2] = node->owner();

      Core::LinAlg::assemble(*checka, pos, id, owner);
    }

    for (int n = 0; n < dis2_->node_row_map()->num_my_elements(); ++n)
    {
      int gid = dis2_->node_row_map()->gid(n);
      Core::Nodes::Node* node = dis2_->g_node(gid);

      Core::LinAlg::SerialDenseVector pos(3);
      pos(0) = node->x()[0];
      pos(1) = node->x()[1];
      pos(2) = node->x()[2];

      std::vector<int> id(3);
      id[0] = dis2_->dof(dofsetb, node, 0);
      id[1] = dis2_->dof(dofsetb, node, 1);
      id[2] = dis2_->dof(dofsetb, node, 2);

      std::vector<int> owner(3);
      owner[0] = node->owner();
      owner[1] = node->owner();
      owner[2] = node->owner();

      Core::LinAlg::assemble(*checkb, pos, id, owner);
    }

    //--------------------------------------------------------------
    //--------------------------------------------------------------
    // Check:
    std::shared_ptr<Core::LinAlg::Vector<double>> finalDA =
        Core::LinAlg::create_vector(*discret1()->dof_row_map(dofseta));
    std::shared_ptr<Core::LinAlg::Vector<double>> finalMA =
        Core::LinAlg::create_vector(*discret1()->dof_row_map(dofseta));

    err = dmatrix_xa_->multiply(false, *checka, *finalDA);
    if (err != 0) FOUR_C_THROW("stop");

    err = mmatrix_xa_->multiply(false, *checkb, *finalMA);
    if (err != 0) FOUR_C_THROW("stop");

    std::shared_ptr<Core::LinAlg::Vector<double>> finalDB =
        Core::LinAlg::create_vector(*discret2()->dof_row_map(dofsetb));
    std::shared_ptr<Core::LinAlg::Vector<double>> finalMB =
        Core::LinAlg::create_vector(*discret2()->dof_row_map(dofsetb));

    err = dmatrix_xb_->multiply(false, *checkb, *finalDB);
    if (err != 0) FOUR_C_THROW("stop");

    err = mmatrix_xb_->multiply(false, *checka, *finalMB);
    if (err != 0) FOUR_C_THROW("stop");

    finalDA->update(-1.0, *finalMA, 1.0);

    finalDB->update(-1.0, *finalMB, 1.0);

    double finalra = 0.0;
    double finalrb = 0.0;

    // Residuum k+1
    finalDA->norm_2(&finalra);
    finalDB->norm_2(&finalrb);

    std::cout << "final ra= " << finalra << "   final rb= " << finalrb << std::endl;
  }

  return;
}
/*----------------------------------------------------------------------*
 |  print_status (public)                                     farah 02/14|
 *----------------------------------------------------------------------*/
void Coupling::VolMortar::VolMortarCoupl::print_status(int& i, bool dis_switch)
{
  static int percent_counter = 0;
  static int EleSum = 0;

  if (i == 0)
  {
    if (dis_switch)
      EleSum = dis2_->num_global_elements();
    else
      EleSum = dis1_->num_global_elements();

    percent_counter = 0;
  }

  if ((int)((i * 100) / EleSum) > (int)(10 * percent_counter))
  {
    std::cout << "---------------------------" << std::endl;
    std::cout << (int)((i * 100) / EleSum) - 1 << "% of Coupling Evaluations are done!"
              << std::endl;
    std::cout << "---------------------------" << std::endl;

    percent_counter++;
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Start Cut routine                                        farah 01/14|
 *----------------------------------------------------------------------*/
void Coupling::VolMortar::VolMortarCoupl::perform_cut(
    Core::Elements::Element* sele, Core::Elements::Element* mele, bool switched_conf)
{
  // create empty vector of integration cells
  std::vector<std::shared_ptr<Cell>> IntCells;

  // the cut wizard wants discretizations to perform the cut. One is supposed to be the background
  // discretization and the other the interface discretization. As we want to cut only two 3D
  // elements, we build two auxiliary discretizations. The first holds only a copy of the master
  // element, acting as background mesh, and the other one is build of the surface elements of the
  // slave element, being the interface discretization. We use temporary copies of all elements and
  // nodes, as we only need the geometry to perform the cut, but want to make sure that the gids
  // and dofs of the original elements are kept untouched.

  std::shared_ptr<Core::FE::Discretization> sauxdis =
      std::make_shared<Core::FE::Discretization>((std::string) "slaveauxdis", comm_, dim_);
  std::shared_ptr<Core::FE::Discretization> mauxdis =
      std::make_shared<Core::FE::Discretization>((std::string) "masterauxdis", comm_, dim_);

  // build surface elements for all surfaces of slave element
  std::vector<std::shared_ptr<Core::Elements::Element>> sele_surfs = sele->surfaces();
  const int numsurf = sele_surfs.size();
  for (int isurf = 0; isurf < numsurf; ++isurf)
  {
    // add all surface elements to auxiliary slave discretization (no need for cloning, as surface
    // elements are
    // rebuild every time when calling Surfaces(), anyway)
    sauxdis->add_element(sele_surfs[isurf]);
  }

  // add clone of element to auxiliary discretization
  mauxdis->add_element(std::shared_ptr<Core::Elements::Element>(mele->clone()));

  // add clones of nodes to auxiliary discretizations
  for (int node = 0; node < sele->num_node(); ++node)
  {
    auto cloned_node = std::shared_ptr<Core::Nodes::Node>(sele->nodes()[node]->clone());
    sauxdis->add_node(cloned_node->x(), cloned_node->id(), cloned_node);
  }
  for (int node = 0; node < mele->num_node(); ++node)
  {
    auto cloned_node = std::shared_ptr<Core::Nodes::Node>(mele->nodes()[node]->clone());
    mauxdis->add_node(cloned_node->x(), cloned_node->id(), cloned_node);
  }

  // complete dis
  sauxdis->fill_complete({
      .assign_degrees_of_freedom = true,
      .init_elements = false,
      .do_boundary_conditions = false,
  });
  mauxdis->fill_complete({
      .assign_degrees_of_freedom = true,
      .init_elements = false,
      .do_boundary_conditions = false,
  });

  //--------------------------------------------------------------------------------------
  // Initialize the cut wizard

  // create new cut wizard
  Cut::CutWizard wizard(mauxdis);

  // *************************************
  // TESSELATION *************************
  if (Teuchos::getIntegralValue<CutType>(params(), "CUTTYPE") == cuttype_tessellation)
  {
    // Set options for the cut wizard
    wizard.set_options(cut_params_, Cut::NDS_Strategy_full,
        Cut::VCellGaussPts_Tessellation,  // how to create volume cell Gauss points?
        Cut::BCellGaussPts_Tessellation,  // how to create boundary cell Gauss points?
        "invalid_file",                   // no file needed
        false,                            // gmsh output for cut library
        true,                             // find point positions
        true,                             // generate only tet cells
        false                             // print screen output
    );

    // cut in reference configuration
    wizard.set_background_state(nullptr, nullptr, -1);
    wizard.add_cutter_state(0, sauxdis, nullptr);

    wizard.prepare();
    wizard.cut(true);  // include_inner

    Cut::plain_volumecell_set mcells_out;
    Cut::plain_volumecell_set mcells_in;
    Cut::ElementHandle* em = wizard.get_element(mele);

    // is mele in cut involved?
    if (em != nullptr)
    {
      em->collect_volume_cells(mcells_in, mcells_out);

      int count = 0;

      for (Cut::plain_volumecell_set::iterator u = mcells_in.begin(); u != mcells_in.end(); u++)
      {
        Cut::VolumeCell* vc = *u;
        const Cut::plain_integrationcell_set& intcells = vc->integration_cells();

        for (Cut::plain_integrationcell_set::const_iterator z = intcells.begin();
            z != intcells.end(); z++)
        {
          Cut::IntegrationCell* ic = *z;

          IntCells.push_back(std::make_shared<Coupling::VolMortar::Cell>(
              count, 4, ic->coordinates(), ic->shape()));
          volume_ += IntCells[count]->vol();

          count++;
        }
      }
      // integrate found cells - tesselation
      if (!switched_conf)
        integrate_3d_cell(*sele, *mele, IntCells);
      else
        integrate_3d_cell(*mele, *sele, IntCells);

      // count the cells and the polygons/polyhedra
      polygoncounter_ += mcells_in.size();
      cellcounter_ += count;
    }
  }

  // *******************************************
  // DIRECT DIVERGENCE *************************
  else if (Teuchos::getIntegralValue<CutType>(params(), "CUTTYPE") == cuttype_directdivergence)
  {
    // Set options for the cut wizard
    wizard.set_options(cut_params_, Cut::NDS_Strategy_full,
        Cut::VCellGaussPts_DirectDivergence,  // how to create volume cell Gauss points?
        Cut::BCellGaussPts_Tessellation,      // how to create boundary cell Gauss points?
        "invalid_file",                       // no file needed
        false,                                // gmsh output for cut library
        true,                                 // find point positions
        false,                                // generate only tet cells
        false                                 // print screen output
    );

    // cut in reference configuration
    wizard.set_background_state(nullptr, nullptr, -1);
    wizard.add_cutter_state(0, sauxdis, nullptr);

    wizard.cut(true);  // include_inner

    Cut::plain_volumecell_set mcells_out;
    Cut::plain_volumecell_set mcells_in;
    Cut::ElementHandle* em = wizard.get_element(mele);

    // for safety
    volcell_.clear();
    if (em != nullptr)
    {
      em->collect_volume_cells(volcell_, mcells_out);

      // start integration
      if (switched_conf)
        integrate_3d_cell_direct_divergence(*mele, *sele, switched_conf);
      else
        integrate_3d_cell_direct_divergence(*sele, *mele, switched_conf);

      polygoncounter_ += volcell_.size();
    }
  }

  // *******************************************
  // DEFAULT           *************************
  else
    FOUR_C_THROW("ERROR: Chosen Cuttype for volmortar not supported!");

  return;
}

/*----------------------------------------------------------------------*
 |  Check need for element-based integration                 farah 01/14|
 *----------------------------------------------------------------------*/
bool Coupling::VolMortar::VolMortarCoupl::check_ele_integration(
    Core::Elements::Element& sele, Core::Elements::Element& mele)
{
  bool integrateele = true;
  bool converged = false;

  double xi[3] = {0.0, 0.0, 0.0};
  double xgl[3] = {0.0, 0.0, 0.0};

  //--------------------------------------------------------
  // 1. all slave nodes within with master ele ?
  for (int u = 0; u < sele.num_node(); ++u)
  {
    xgl[0] = sele.nodes()[u]->x()[0];
    xgl[1] = sele.nodes()[u]->x()[1];
    xgl[2] = sele.nodes()[u]->x()[2];

    // global to local:
    if (mele.shape() == Core::FE::CellType::hex8)
      Mortar::Utils::global_to_local<Core::FE::CellType::hex8>(mele, xgl, xi, converged);
    else if (mele.shape() == Core::FE::CellType::hex20)
      Mortar::Utils::global_to_local<Core::FE::CellType::hex20>(mele, xgl, xi, converged);
    else if (mele.shape() == Core::FE::CellType::hex27)
      Mortar::Utils::global_to_local<Core::FE::CellType::hex27>(mele, xgl, xi, converged);
    else if (mele.shape() == Core::FE::CellType::tet4)
      Mortar::Utils::global_to_local<Core::FE::CellType::tet4>(mele, xgl, xi, converged);
    else if (mele.shape() == Core::FE::CellType::tet10)
      Mortar::Utils::global_to_local<Core::FE::CellType::tet10>(mele, xgl, xi, converged);
    else if (mele.shape() == Core::FE::CellType::pyramid5)
      Mortar::Utils::global_to_local<Core::FE::CellType::pyramid5>(mele, xgl, xi, converged);
    else
      FOUR_C_THROW("ERROR: Shape function not supported!");

    if (converged == true)
    {
      if (mele.shape() == Core::FE::CellType::hex8 or mele.shape() == Core::FE::CellType::hex20 or
          mele.shape() == Core::FE::CellType::hex27)
      {
        if (xi[0] > -1.0 - VOLMORTARELETOL and xi[0] < 1.0 + VOLMORTARELETOL and
            xi[1] > -1.0 - VOLMORTARELETOL and xi[1] < 1.0 + VOLMORTARELETOL and
            xi[2] > -1.0 - VOLMORTARELETOL and xi[2] < 1.0 + VOLMORTARELETOL)
          integrateele = true;
        else
          return false;
      }
      else if (mele.shape() == Core::FE::CellType::tet4 or
               mele.shape() == Core::FE::CellType::tet10)
      {
        if (xi[0] > 0.0 - VOLMORTARELETOL and xi[0] < 1.0 + VOLMORTARELETOL and
            xi[1] > 0.0 - VOLMORTARELETOL and xi[1] < 1.0 + VOLMORTARELETOL and
            xi[2] > 0.0 - VOLMORTARELETOL and xi[2] < 1.0 + VOLMORTARELETOL and
            (xi[0] + xi[1] + xi[2]) < 1.0 + 3 * VOLMORTARELETOL)
          integrateele = true;
        else
          return false;
      }
      else
        FOUR_C_THROW("ERROR: Element type not supported!");
    }
    else
    {
      //      std::cout << "!!! GLOBAL TO LOCAL NOT CONVERGED !!!" << std::endl;
      return false;
    }
  }

  return integrateele;
}

/*----------------------------------------------------------------------*
 |  Check need for cut and element-based integration         farah 01/14|
 *----------------------------------------------------------------------*/
bool Coupling::VolMortar::VolMortarCoupl::check_cut(
    Core::Elements::Element& sele, Core::Elements::Element& mele)
{
  double xi[3] = {0.0, 0.0, 0.0};
  double xgl[3] = {0.0, 0.0, 0.0};
  bool converged = false;

  {
    //--------------------------------------------------------
    // 1. check if all nodes are outside a parameter space surface
    // bools for tet
    bool xi0 = false;
    bool xi1 = false;
    bool xi2 = false;
    bool all = false;

    // bools for hex
    bool xi0n = false;
    bool xi1n = false;
    bool xi2n = false;

    for (int u = 0; u < mele.num_node(); ++u)
    {
      xgl[0] = mele.nodes()[u]->x()[0];
      xgl[1] = mele.nodes()[u]->x()[1];
      xgl[2] = mele.nodes()[u]->x()[2];

      // global to local:
      if (sele.shape() == Core::FE::CellType::hex8)
        Mortar::Utils::global_to_local<Core::FE::CellType::hex8>(sele, xgl, xi, converged);
      else if (sele.shape() == Core::FE::CellType::hex20)
        Mortar::Utils::global_to_local<Core::FE::CellType::hex20>(sele, xgl, xi, converged);
      else if (sele.shape() == Core::FE::CellType::hex27)
        Mortar::Utils::global_to_local<Core::FE::CellType::hex27>(sele, xgl, xi, converged);
      else if (sele.shape() == Core::FE::CellType::tet4)
        Mortar::Utils::global_to_local<Core::FE::CellType::tet4>(sele, xgl, xi, converged);
      else if (sele.shape() == Core::FE::CellType::tet10)
        Mortar::Utils::global_to_local<Core::FE::CellType::tet10>(sele, xgl, xi, converged);
      else if (sele.shape() == Core::FE::CellType::pyramid5)
        Mortar::Utils::global_to_local<Core::FE::CellType::pyramid5>(sele, xgl, xi, converged);
      else
        FOUR_C_THROW("ERROR: Shape function not supported!");

      if (converged == true)
      {
        if (sele.shape() == Core::FE::CellType::hex8 or sele.shape() == Core::FE::CellType::hex20 or
            sele.shape() == Core::FE::CellType::hex27)
        {
          if (xi[0] > -1.0 + VOLMORTARCUTTOL) xi0 = true;
          if (xi[1] > -1.0 + VOLMORTARCUTTOL) xi1 = true;
          if (xi[2] > -1.0 + VOLMORTARCUTTOL) xi2 = true;

          if (xi[0] < 1.0 - VOLMORTARCUTTOL) xi0n = true;
          if (xi[1] < 1.0 - VOLMORTARCUTTOL) xi1n = true;
          if (xi[2] < 1.0 - VOLMORTARCUTTOL) xi2n = true;
        }
        else if (sele.shape() == Core::FE::CellType::tet4 or
                 sele.shape() == Core::FE::CellType::tet10)
        {
          if (xi[0] > 0.0 + VOLMORTARCUTTOL) xi0 = true;
          if (xi[1] > 0.0 + VOLMORTARCUTTOL) xi1 = true;
          if (xi[2] > 0.0 + VOLMORTARCUTTOL) xi2 = true;
          if ((xi[0] + xi[1] + xi[2]) < 1.0 - 3.0 * VOLMORTARCUTTOL) all = true;
        }
        else
          FOUR_C_THROW("ERROR: Element not supported!");
      }
    }  // end node loop

    if (sele.shape() == Core::FE::CellType::tet4 or sele.shape() == Core::FE::CellType::tet10)
    {
      if (!xi0 or !xi1 or !xi2 or !all) return false;
    }
    else if (sele.shape() == Core::FE::CellType::hex8 or
             sele.shape() == Core::FE::CellType::hex20 or sele.shape() == Core::FE::CellType::hex27)
    {
      if (!xi0 or !xi1 or !xi2 or !xi0n or !xi1n or !xi2n) return false;
    }
    else
      FOUR_C_THROW("ERROR: Element not supported!");
  }

  {
    //--------------------------------------------------------
    // 2. check if all nodes are outside a parameter space surface (changed elements)
    bool xi0 = false;
    bool xi1 = false;
    bool xi2 = false;
    bool all = false;

    // bools for hex
    bool xi0n = false;
    bool xi1n = false;
    bool xi2n = false;

    for (int u = 0; u < sele.num_node(); ++u)
    {
      xgl[0] = sele.nodes()[u]->x()[0];
      xgl[1] = sele.nodes()[u]->x()[1];
      xgl[2] = sele.nodes()[u]->x()[2];

      // global to local:
      if (mele.shape() == Core::FE::CellType::hex8)
        Mortar::Utils::global_to_local<Core::FE::CellType::hex8>(mele, xgl, xi, converged);
      else if (mele.shape() == Core::FE::CellType::hex20)
        Mortar::Utils::global_to_local<Core::FE::CellType::hex20>(mele, xgl, xi, converged);
      else if (mele.shape() == Core::FE::CellType::hex27)
        Mortar::Utils::global_to_local<Core::FE::CellType::hex27>(mele, xgl, xi, converged);
      else if (mele.shape() == Core::FE::CellType::tet4)
        Mortar::Utils::global_to_local<Core::FE::CellType::tet4>(mele, xgl, xi, converged);
      else if (mele.shape() == Core::FE::CellType::tet10)
        Mortar::Utils::global_to_local<Core::FE::CellType::tet10>(mele, xgl, xi, converged);
      else if (mele.shape() == Core::FE::CellType::pyramid5)
        Mortar::Utils::global_to_local<Core::FE::CellType::pyramid5>(mele, xgl, xi, converged);
      else
        FOUR_C_THROW("ERROR: Shape function not supported!");

      if (converged == true)
      {
        if (mele.shape() == Core::FE::CellType::hex8 or mele.shape() == Core::FE::CellType::hex20 or
            mele.shape() == Core::FE::CellType::hex27)
        {
          if (xi[0] > -1.0 + VOLMORTARCUTTOL) xi0 = true;
          if (xi[1] > -1.0 + VOLMORTARCUTTOL) xi1 = true;
          if (xi[2] > -1.0 + VOLMORTARCUTTOL) xi2 = true;

          if (xi[0] < 1.0 - VOLMORTARCUTTOL) xi0n = true;
          if (xi[1] < 1.0 - VOLMORTARCUTTOL) xi1n = true;
          if (xi[2] < 1.0 - VOLMORTARCUTTOL) xi2n = true;
        }

        else if (mele.shape() == Core::FE::CellType::tet4 or
                 mele.shape() == Core::FE::CellType::tet10)
        {
          if (xi[0] > 0.0 + VOLMORTARCUTTOL) xi0 = true;
          if (xi[1] > 0.0 + VOLMORTARCUTTOL) xi1 = true;
          if (xi[2] > 0.0 + VOLMORTARCUTTOL) xi2 = true;
          if ((xi[0] + xi[1] + xi[2]) < 1.0 - 3.0 * VOLMORTARCUTTOL) all = true;
        }
        else
          FOUR_C_THROW("ERROR: Element not supported!");
      }
    }

    if (mele.shape() == Core::FE::CellType::tet4 or mele.shape() == Core::FE::CellType::tet10)
    {
      if (!xi0 or !xi1 or !xi2 or !all) return false;
    }
    else if (mele.shape() == Core::FE::CellType::hex8 or
             mele.shape() == Core::FE::CellType::hex20 or mele.shape() == Core::FE::CellType::hex27)
    {
      if (!xi0 or !xi1 or !xi2 or !xi0n or !xi1n or !xi2n) return false;
    }
    else
      FOUR_C_THROW("ERROR: Element not supported!");
  }

  //--------------------------------------------------------
  // 3. master nodes within slave parameter space?
  for (int u = 0; u < mele.num_node(); ++u)
  {
    xgl[0] = mele.nodes()[u]->x()[0];
    xgl[1] = mele.nodes()[u]->x()[1];
    xgl[2] = mele.nodes()[u]->x()[2];

    // global to local:
    if (sele.shape() == Core::FE::CellType::hex8)
      Mortar::Utils::global_to_local<Core::FE::CellType::hex8>(sele, xgl, xi, converged);
    else if (sele.shape() == Core::FE::CellType::hex20)
      Mortar::Utils::global_to_local<Core::FE::CellType::hex20>(sele, xgl, xi, converged);
    else if (sele.shape() == Core::FE::CellType::hex27)
      Mortar::Utils::global_to_local<Core::FE::CellType::hex27>(sele, xgl, xi, converged);
    else if (sele.shape() == Core::FE::CellType::tet4)
      Mortar::Utils::global_to_local<Core::FE::CellType::tet4>(sele, xgl, xi, converged);
    else if (sele.shape() == Core::FE::CellType::tet10)
      Mortar::Utils::global_to_local<Core::FE::CellType::tet10>(sele, xgl, xi, converged);
    else if (sele.shape() == Core::FE::CellType::pyramid5)
      Mortar::Utils::global_to_local<Core::FE::CellType::pyramid5>(sele, xgl, xi, converged);
    else
      FOUR_C_THROW("ERROR: Shape function not supported!");

    if (converged == true)
    {
      if (sele.shape() == Core::FE::CellType::hex8 or sele.shape() == Core::FE::CellType::hex20 or
          sele.shape() == Core::FE::CellType::hex27)
      {
        if (abs(xi[0]) < 1.0 - VOLMORTARCUT2TOL and abs(xi[1]) < 1.0 - VOLMORTARCUT2TOL and
            abs(xi[2]) < 1.0 - VOLMORTARCUT2TOL)
          return true;
      }
      else if (sele.shape() == Core::FE::CellType::tet4 or
               sele.shape() == Core::FE::CellType::tet10)
      {
        if (xi[0] > 0.0 + VOLMORTARCUT2TOL and xi[0] < 1.0 - VOLMORTARCUT2TOL and
            xi[1] > 0.0 + VOLMORTARCUT2TOL and xi[1] < 1.0 - VOLMORTARCUT2TOL and
            xi[2] > 0.0 + VOLMORTARCUT2TOL and xi[2] < 1.0 - VOLMORTARCUT2TOL and
            (xi[0] + xi[1] + xi[2]) < 1.0 - 3.0 * VOLMORTARCUT2TOL)
          return true;
      }
      else
        FOUR_C_THROW("ERROR: Element not supported!");
    }
  }

  //--------------------------------------------------------
  // 4. slave nodes within master parameter space?
  for (int u = 0; u < sele.num_node(); ++u)
  {
    xgl[0] = sele.nodes()[u]->x()[0];
    xgl[1] = sele.nodes()[u]->x()[1];
    xgl[2] = sele.nodes()[u]->x()[2];

    // global to local:
    if (mele.shape() == Core::FE::CellType::hex8)
      Mortar::Utils::global_to_local<Core::FE::CellType::hex8>(mele, xgl, xi, converged);
    else if (mele.shape() == Core::FE::CellType::hex20)
      Mortar::Utils::global_to_local<Core::FE::CellType::hex20>(mele, xgl, xi, converged);
    else if (mele.shape() == Core::FE::CellType::hex27)
      Mortar::Utils::global_to_local<Core::FE::CellType::hex27>(mele, xgl, xi, converged);
    else if (mele.shape() == Core::FE::CellType::tet4)
      Mortar::Utils::global_to_local<Core::FE::CellType::tet4>(mele, xgl, xi, converged);
    else if (mele.shape() == Core::FE::CellType::tet10)
      Mortar::Utils::global_to_local<Core::FE::CellType::tet10>(mele, xgl, xi, converged);
    else if (mele.shape() == Core::FE::CellType::pyramid5)
      Mortar::Utils::global_to_local<Core::FE::CellType::pyramid5>(mele, xgl, xi, converged);
    else
      FOUR_C_THROW("ERROR: Shape function not supported!");

    if (converged == true)
    {
      if (mele.shape() == Core::FE::CellType::hex8 or mele.shape() == Core::FE::CellType::hex20 or
          mele.shape() == Core::FE::CellType::hex27)
      {
        if (abs(xi[0]) < 1.0 - VOLMORTARCUT2TOL and abs(xi[1]) < 1.0 - VOLMORTARCUT2TOL and
            abs(xi[2]) < 1.0 - VOLMORTARCUT2TOL)
          return true;
      }
      else if (mele.shape() == Core::FE::CellType::tet4 or
               mele.shape() == Core::FE::CellType::tet10)
      {
        if (xi[0] > 0.0 + VOLMORTARCUT2TOL and xi[0] < 1.0 - VOLMORTARCUT2TOL and
            xi[1] > 0.0 + VOLMORTARCUT2TOL and xi[1] < 1.0 - VOLMORTARCUT2TOL and
            xi[2] > 0.0 + VOLMORTARCUT2TOL and xi[2] < 1.0 - VOLMORTARCUT2TOL and
            (xi[0] + xi[1] + xi[2]) < 1.0 - 3.0 * VOLMORTARCUT2TOL)
          return true;
      }
      else
        FOUR_C_THROW("ERROR: Element not supported!");
    }
  }

  return false;
}

/*----------------------------------------------------------------------*
 |  integrate_2d Cells                                        farah 01/14|
 *----------------------------------------------------------------------*/
void Coupling::VolMortar::VolMortarCoupl::integrate_2d(Core::Elements::Element& sele,
    Core::Elements::Element& mele, std::vector<std::shared_ptr<Mortar::IntCell>>& cells)
{
  //--------------------------------------------------------------------
  // loop over cells for A Field
  // loop over cells
  for (int q = 0; q < (int)cells.size(); ++q)
  {
    switch (sele.shape())
    {
      // 2D surface elements
      case Core::FE::CellType::quad4:
      {
        switch (mele.shape())
        {
          // 2D surface elements
          case Core::FE::CellType::quad4:
          {
            static VolMortarIntegrator<Core::FE::CellType::quad4, Core::FE::CellType::quad4>
                integrator(params());
            integrator.integrate_cells_2d(sele, mele, *cells[q], *d1_, *m12_, *dis1_, *dis2_,
                dofset12_.first, dofset12_.second);
            break;
          }
          case Core::FE::CellType::tri3:
          {
            static VolMortarIntegrator<Core::FE::CellType::quad4, Core::FE::CellType::tri3>
                integrator(params());
            integrator.integrate_cells_2d(sele, mele, *cells[q], *d1_, *m12_, *dis1_, *dis2_,
                dofset12_.first, dofset12_.second);
            break;
          }
          default:
          {
            FOUR_C_THROW("ERROR: unknown shape!");
            break;
          }
        }
        break;
      }
      case Core::FE::CellType::tri3:
      {
        switch (mele.shape())
        {
          // 2D surface elements
          case Core::FE::CellType::quad4:
          {
            static VolMortarIntegrator<Core::FE::CellType::tri3, Core::FE::CellType::quad4>
                integrator(params());
            integrator.integrate_cells_2d(sele, mele, *cells[q], *d1_, *m12_, *dis1_, *dis2_,
                dofset12_.first, dofset12_.second);
            break;
          }
          case Core::FE::CellType::tri3:
          {
            static VolMortarIntegrator<Core::FE::CellType::tri3, Core::FE::CellType::tri3>
                integrator(params());
            integrator.integrate_cells_2d(sele, mele, *cells[q], *d1_, *m12_, *dis1_, *dis2_,
                dofset12_.first, dofset12_.second);
            break;
          }
          default:
          {
            FOUR_C_THROW("ERROR: unknown shape!");
            break;
          }
        }
        break;
      }
      default:
      {
        FOUR_C_THROW("ERROR: unknown shape!");
        break;
      }
    }

    //--------------------------------------------------------------------
    // loop over cells for A Field
    switch (mele.shape())
    {
      // 2D surface elements
      case Core::FE::CellType::quad4:
      {
        switch (sele.shape())
        {
          // 2D surface elements
          case Core::FE::CellType::quad4:
          {
            static VolMortarIntegrator<Core::FE::CellType::quad4, Core::FE::CellType::quad4>
                integrator(params());
            integrator.integrate_cells_2d(mele, sele, *cells[q], *d2_, *m21_, *dis2_, *dis1_,
                dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::tri3:
          {
            static VolMortarIntegrator<Core::FE::CellType::quad4, Core::FE::CellType::tri3>
                integrator(params());
            integrator.integrate_cells_2d(mele, sele, *cells[q], *d2_, *m21_, *dis2_, *dis1_,
                dofset21_.first, dofset21_.second);
            break;
          }
          default:
          {
            FOUR_C_THROW("ERROR: unknown shape!");
            break;
          }
        }
        break;
      }
      case Core::FE::CellType::tri3:
      {
        switch (sele.shape())
        {
          // 2D surface elements
          case Core::FE::CellType::quad4:
          {
            static VolMortarIntegrator<Core::FE::CellType::tri3, Core::FE::CellType::quad4>
                integrator(params());
            integrator.integrate_cells_2d(mele, sele, *cells[q], *d2_, *m21_, *dis2_, *dis1_,
                dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::tri3:
          {
            static VolMortarIntegrator<Core::FE::CellType::tri3, Core::FE::CellType::tri3>
                integrator(params());
            integrator.integrate_cells_2d(mele, sele, *cells[q], *d2_, *m21_, *dis2_, *dis1_,
                dofset21_.first, dofset21_.second);
            break;
          }
          default:
          {
            FOUR_C_THROW("ERROR: unknown shape!");
            break;
          }
        }
        break;
      }
      default:
      {
        FOUR_C_THROW("ERROR: unknown shape!");
        break;
      }
    }
  }  // end loop

  return;
}

/*----------------------------------------------------------------------*
 |  integrate_3d Cells                                        farah 01/14|
 *----------------------------------------------------------------------*/
void Coupling::VolMortar::VolMortarCoupl::integrate_3d_cell(Core::Elements::Element& sele,
    Core::Elements::Element& mele, std::vector<std::shared_ptr<Cell>>& cells)
{
  //--------------------------------------------------------------------
  // loop over cells for A Field
  for (int q = 0; q < (int)cells.size(); ++q)
  {
    switch (sele.shape())
    {
      // 2D surface elements
      case Core::FE::CellType::hex8:
      {
        switch (mele.shape())
        {
          // 2D surface elements
          case Core::FE::CellType::hex8:
          {
            static VolMortarIntegrator<Core::FE::CellType::hex8, Core::FE::CellType::hex8>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::hex20:
          {
            static VolMortarIntegrator<Core::FE::CellType::hex8, Core::FE::CellType::hex20>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::hex27:
          {
            static VolMortarIntegrator<Core::FE::CellType::hex8, Core::FE::CellType::hex27>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::tet4:
          {
            static VolMortarIntegrator<Core::FE::CellType::hex8, Core::FE::CellType::tet4>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::tet10:
          {
            static VolMortarIntegrator<Core::FE::CellType::hex8, Core::FE::CellType::tet10>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::pyramid5:
          {
            static VolMortarIntegrator<Core::FE::CellType::hex8, Core::FE::CellType::pyramid5>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          default:
          {
            FOUR_C_THROW("ERROR: unknown shape!");
            break;
          }
        }
        break;
      }
      // #######################################################
      case Core::FE::CellType::hex20:
      {
        switch (mele.shape())
        {
          // 2D surface elements
          case Core::FE::CellType::hex8:
          {
            static VolMortarIntegrator<Core::FE::CellType::hex20, Core::FE::CellType::hex8>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::hex20:
          {
            static VolMortarIntegrator<Core::FE::CellType::hex20, Core::FE::CellType::hex20>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::hex27:
          {
            static VolMortarIntegrator<Core::FE::CellType::hex20, Core::FE::CellType::hex27>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::tet4:
          {
            static VolMortarIntegrator<Core::FE::CellType::hex20, Core::FE::CellType::tet4>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::tet10:
          {
            static VolMortarIntegrator<Core::FE::CellType::hex20, Core::FE::CellType::tet10>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::pyramid5:
          {
            static VolMortarIntegrator<Core::FE::CellType::hex20, Core::FE::CellType::pyramid5>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          default:
          {
            FOUR_C_THROW("ERROR: unknown shape!");
            break;
          }
        }
        break;
      }
      // #######################################################
      case Core::FE::CellType::hex27:
      {
        switch (mele.shape())
        {
          // 2D surface elements
          case Core::FE::CellType::hex8:
          {
            static VolMortarIntegrator<Core::FE::CellType::hex27, Core::FE::CellType::hex8>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::hex20:
          {
            static VolMortarIntegrator<Core::FE::CellType::hex27, Core::FE::CellType::hex20>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::hex27:
          {
            static VolMortarIntegrator<Core::FE::CellType::hex27, Core::FE::CellType::hex27>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::tet4:
          {
            static VolMortarIntegrator<Core::FE::CellType::hex27, Core::FE::CellType::tet4>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::tet10:
          {
            static VolMortarIntegrator<Core::FE::CellType::hex27, Core::FE::CellType::tet10>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::pyramid5:
          {
            static VolMortarIntegrator<Core::FE::CellType::hex27, Core::FE::CellType::pyramid5>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          default:
          {
            FOUR_C_THROW("ERROR: unknown shape!");
            break;
          }
        }
        break;
      }
      // #######################################################
      case Core::FE::CellType::tet4:
      {
        switch (mele.shape())
        {
          // 2D surface elements
          case Core::FE::CellType::hex8:
          {
            static VolMortarIntegrator<Core::FE::CellType::tet4, Core::FE::CellType::hex8>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::hex20:
          {
            static VolMortarIntegrator<Core::FE::CellType::tet4, Core::FE::CellType::hex20>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::hex27:
          {
            static VolMortarIntegrator<Core::FE::CellType::tet4, Core::FE::CellType::hex27>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::tet4:
          {
            static VolMortarIntegrator<Core::FE::CellType::tet4, Core::FE::CellType::tet4>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::tet10:
          {
            static VolMortarIntegrator<Core::FE::CellType::tet4, Core::FE::CellType::tet10>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::pyramid5:
          {
            static VolMortarIntegrator<Core::FE::CellType::tet4, Core::FE::CellType::pyramid5>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          default:
          {
            FOUR_C_THROW("ERROR: unknown shape!");
            break;
          }
        }
        break;
      }
      // #######################################################
      case Core::FE::CellType::tet10:
      {
        switch (mele.shape())
        {
          // 2D surface elements
          case Core::FE::CellType::hex8:
          {
            static VolMortarIntegrator<Core::FE::CellType::tet10, Core::FE::CellType::hex8>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::hex20:
          {
            static VolMortarIntegrator<Core::FE::CellType::tet10, Core::FE::CellType::hex20>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::hex27:
          {
            static VolMortarIntegrator<Core::FE::CellType::tet10, Core::FE::CellType::hex27>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::tet4:
          {
            static VolMortarIntegrator<Core::FE::CellType::tet10, Core::FE::CellType::tet4>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::tet10:
          {
            static VolMortarIntegrator<Core::FE::CellType::tet10, Core::FE::CellType::tet10>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::pyramid5:
          {
            static VolMortarIntegrator<Core::FE::CellType::tet10, Core::FE::CellType::pyramid5>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          default:
          {
            FOUR_C_THROW("ERROR: unknown shape!");
            break;
          }
        }
        break;
      }
      // #######################################################
      case Core::FE::CellType::pyramid5:
      {
        switch (mele.shape())
        {
          // 2D surface elements
          case Core::FE::CellType::hex8:
          {
            static VolMortarIntegrator<Core::FE::CellType::pyramid5, Core::FE::CellType::hex8>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::hex20:
          {
            static VolMortarIntegrator<Core::FE::CellType::pyramid5, Core::FE::CellType::hex20>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::hex27:
          {
            static VolMortarIntegrator<Core::FE::CellType::pyramid5, Core::FE::CellType::hex27>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::tet4:
          {
            static VolMortarIntegrator<Core::FE::CellType::pyramid5, Core::FE::CellType::tet4>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::tet10:
          {
            static VolMortarIntegrator<Core::FE::CellType::pyramid5, Core::FE::CellType::tet10>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::pyramid5:
          {
            static VolMortarIntegrator<Core::FE::CellType::pyramid5, Core::FE::CellType::pyramid5>
                integrator(params());
            integrator.initialize_gp(false, 0, cells[q]->shape());
            integrator.integrate_cells_3d(sele, mele, *cells[q], *d1_, *m12_, *d2_, *m21_, *dis1_,
                *dis2_, dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          default:
          {
            FOUR_C_THROW("ERROR: unknown shape!");
            break;
          }
        }
        break;
      }
      default:
      {
        FOUR_C_THROW("ERROR: unknown shape!");
        break;
      }
    }
  }  // end cell loop
  return;
}

/*----------------------------------------------------------------------*
 |  integrate_3d elebased                                     farah 04/14|
 *----------------------------------------------------------------------*/
void Coupling::VolMortar::VolMortarCoupl::integrate_3d_ele_based_p12(
    Core::Elements::Element& Aele, std::vector<int>& foundeles)
{
  switch (Aele.shape())
  {
    // 2D "volume" elements
    case Core::FE::CellType::quad4:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::quad4> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Aele, foundeles, *d1_, *m12_, *dis1_, *dis2_,
          dofset12_.first, dofset12_.second, *p12_dofrowmap_, *p12_dofcolmap_);
      break;
    }
    case Core::FE::CellType::quad8:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::quad8> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Aele, foundeles, *d1_, *m12_, *dis1_, *dis2_,
          dofset12_.first, dofset12_.second, *p12_dofrowmap_, *p12_dofcolmap_);
      break;
    }
    case Core::FE::CellType::quad9:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::quad9> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Aele, foundeles, *d1_, *m12_, *dis1_, *dis2_,
          dofset12_.first, dofset12_.second, *p12_dofrowmap_, *p12_dofcolmap_);
      break;
    }
    case Core::FE::CellType::tri3:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::tri3> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Aele, foundeles, *d1_, *m12_, *dis1_, *dis2_,
          dofset12_.first, dofset12_.second, *p12_dofrowmap_, *p12_dofcolmap_);
      break;
    }
    case Core::FE::CellType::tri6:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::tri6> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Aele, foundeles, *d1_, *m12_, *dis1_, *dis2_,
          dofset12_.first, dofset12_.second, *p12_dofrowmap_, *p12_dofcolmap_);
      break;
    }
    // 3D volume elements
    case Core::FE::CellType::hex8:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::hex8> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Aele, foundeles, *d1_, *m12_, *dis1_, *dis2_,
          dofset12_.first, dofset12_.second, *p12_dofrowmap_, *p12_dofcolmap_);
      break;
    }
    case Core::FE::CellType::hex27:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::hex27> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Aele, foundeles, *d1_, *m12_, *dis1_, *dis2_,
          dofset12_.first, dofset12_.second, *p12_dofrowmap_, *p12_dofcolmap_);
      break;
    }
    case Core::FE::CellType::hex20:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::hex20> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Aele, foundeles, *d1_, *m12_, *dis1_, *dis2_,
          dofset12_.first, dofset12_.second, *p12_dofrowmap_, *p12_dofcolmap_);
      break;
    }
    case Core::FE::CellType::tet4:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::tet4> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Aele, foundeles, *d1_, *m12_, *dis1_, *dis2_,
          dofset12_.first, dofset12_.second, *p12_dofrowmap_, *p12_dofcolmap_);
      break;
    }
    case Core::FE::CellType::tet10:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::tet10> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Aele, foundeles, *d1_, *m12_, *dis1_, *dis2_,
          dofset12_.first, dofset12_.second, *p12_dofrowmap_, *p12_dofcolmap_);
      break;
    }
    case Core::FE::CellType::pyramid5:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::pyramid5> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Aele, foundeles, *d1_, *m12_, *dis1_, *dis2_,
          dofset12_.first, dofset12_.second, *p12_dofrowmap_, *p12_dofcolmap_);
      break;
    }
    default:
    {
      FOUR_C_THROW("ERROR: unknown shape!");
      break;
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  integrate_3d element based for projector P21              farah 04/14|
 *----------------------------------------------------------------------*/
void Coupling::VolMortar::VolMortarCoupl::integrate_3d_ele_based_p21(
    Core::Elements::Element& Bele, std::vector<int>& foundeles)
{
  switch (Bele.shape())
  {
    // 2D "volume" elements
    case Core::FE::CellType::quad4:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::quad4> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Bele, foundeles, *d2_, *m21_, *dis2_, *dis1_,
          dofset21_.first, dofset21_.second, *p21_dofrowmap_, *p21_dofcolmap_);
      break;
    }
    case Core::FE::CellType::quad8:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::quad8> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Bele, foundeles, *d2_, *m21_, *dis2_, *dis1_,
          dofset21_.first, dofset21_.second, *p21_dofrowmap_, *p21_dofcolmap_);
      break;
    }
    case Core::FE::CellType::quad9:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::quad9> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Bele, foundeles, *d2_, *m21_, *dis2_, *dis1_,
          dofset21_.first, dofset21_.second, *p21_dofrowmap_, *p21_dofcolmap_);
      break;
    }
    case Core::FE::CellType::tri3:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::tri3> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Bele, foundeles, *d2_, *m21_, *dis2_, *dis1_,
          dofset21_.first, dofset21_.second, *p21_dofrowmap_, *p21_dofcolmap_);
      break;
    }
    case Core::FE::CellType::tri6:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::tri6> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Bele, foundeles, *d2_, *m21_, *dis2_, *dis1_,
          dofset21_.first, dofset21_.second, *p21_dofrowmap_, *p21_dofcolmap_);
      break;
    }
    // 3D volume elements
    case Core::FE::CellType::hex8:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::hex8> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Bele, foundeles, *d2_, *m21_, *dis2_, *dis1_,
          dofset21_.first, dofset21_.second, *p21_dofrowmap_, *p21_dofcolmap_);
      break;
    }
    case Core::FE::CellType::hex20:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::hex20> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Bele, foundeles, *d2_, *m21_, *dis2_, *dis1_,
          dofset21_.first, dofset21_.second, *p21_dofrowmap_, *p21_dofcolmap_);
      break;
    }
    case Core::FE::CellType::hex27:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::hex27> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Bele, foundeles, *d2_, *m21_, *dis2_, *dis1_,
          dofset21_.first, dofset21_.second, *p21_dofrowmap_, *p21_dofcolmap_);
      break;
    }
    case Core::FE::CellType::tet4:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::tet4> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Bele, foundeles, *d2_, *m21_, *dis2_, *dis1_,
          dofset21_.first, dofset21_.second, *p21_dofrowmap_, *p21_dofcolmap_);
      break;
    }
    case Core::FE::CellType::tet10:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::tet10> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Bele, foundeles, *d2_, *m21_, *dis2_, *dis1_,
          dofset21_.first, dofset21_.second, *p21_dofrowmap_, *p21_dofcolmap_);
      break;
    }
    case Core::FE::CellType::pyramid5:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::pyramid5> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Bele, foundeles, *d2_, *m21_, *dis2_, *dis1_,
          dofset21_.first, dofset21_.second, *p21_dofrowmap_, *p21_dofcolmap_);
      break;
    }
    default:
    {
      FOUR_C_THROW("ERROR: unknown shape!");
      break;
    }
  }
  return;
}

/*----------------------------------------------------------------------*
 |  integrate_3d elebased                                     farah 03/15|
 *----------------------------------------------------------------------*/
void Coupling::VolMortar::VolMortarCoupl::integrate_3d_ele_based_a_dis_mesh_init(
    Core::Elements::Element& Aele, std::vector<int>& foundeles, int dofseta, int dofsetb)
{
  switch (Aele.shape())
  {
    // 2D surface elements
    case Core::FE::CellType::hex8:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::hex8> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Aele, foundeles, *dmatrix_xa_, *mmatrix_xa_, *dis1_, *dis2_,
          dofseta, dofsetb, *p12_dofrowmap_, *p12_dofcolmap_);
      break;
    }
    case Core::FE::CellType::tet4:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::tet4> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Aele, foundeles, *dmatrix_xa_, *mmatrix_xa_, *dis1_, *dis2_,
          dofseta, dofsetb, *p12_dofrowmap_, *p12_dofcolmap_);
      break;
    }
    case Core::FE::CellType::hex27:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::hex27> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Aele, foundeles, *dmatrix_xa_, *mmatrix_xa_, *dis1_, *dis2_,
          dofseta, dofsetb, *p12_dofrowmap_, *p12_dofcolmap_);
      break;
    }
    case Core::FE::CellType::hex20:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::hex20> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Aele, foundeles, *dmatrix_xa_, *mmatrix_xa_, *dis1_, *dis2_,
          dofseta, dofsetb, *p12_dofrowmap_, *p12_dofcolmap_);
      break;
    }
    case Core::FE::CellType::tet10:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::tet10> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Aele, foundeles, *dmatrix_xa_, *mmatrix_xa_, *dis1_, *dis2_,
          dofseta, dofsetb, *p12_dofrowmap_, *p12_dofcolmap_);
      break;
    }
    case Core::FE::CellType::pyramid5:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::pyramid5> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Aele, foundeles, *dmatrix_xa_, *mmatrix_xa_, *dis1_, *dis2_,
          dofseta, dofsetb, *p12_dofrowmap_, *p12_dofcolmap_);
      break;
    }
    default:
    {
      FOUR_C_THROW("ERROR: unknown shape!");
      break;
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  integrate_3d Cells                                        farah 03/15|
 *----------------------------------------------------------------------*/
void Coupling::VolMortar::VolMortarCoupl::integrate_3d_ele_based_b_dis_mesh_init(
    Core::Elements::Element& Bele, std::vector<int>& foundeles, int dofsetb, int dofseta)
{
  switch (Bele.shape())
  {
    // 2D surface elements
    case Core::FE::CellType::hex8:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::hex8> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Bele, foundeles, *dmatrix_xb_, *mmatrix_xb_, *dis2_, *dis1_,
          dofseta, dofsetb, *p21_dofrowmap_, *p21_dofcolmap_);
      break;
    }
    case Core::FE::CellType::tet4:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::tet4> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Bele, foundeles, *dmatrix_xb_, *mmatrix_xb_, *dis2_, *dis1_,
          dofseta, dofsetb, *p21_dofrowmap_, *p21_dofcolmap_);
      break;
    }
    case Core::FE::CellType::hex27:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::hex27> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Bele, foundeles, *dmatrix_xb_, *mmatrix_xb_, *dis2_, *dis1_,
          dofseta, dofsetb, *p21_dofrowmap_, *p21_dofcolmap_);
      break;
    }
    case Core::FE::CellType::hex20:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::hex20> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Bele, foundeles, *dmatrix_xb_, *mmatrix_xb_, *dis2_, *dis1_,
          dofseta, dofsetb, *p21_dofrowmap_, *p21_dofcolmap_);
      break;
    }
    case Core::FE::CellType::tet10:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::tet10> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Bele, foundeles, *dmatrix_xb_, *mmatrix_xb_, *dis2_, *dis1_,
          dofseta, dofsetb, *p21_dofrowmap_, *p21_dofcolmap_);
      break;
    }
    case Core::FE::CellType::pyramid5:
    {
      static VolMortarIntegratorEleBased<Core::FE::CellType::pyramid5> integrator(params());
      integrator.initialize_gp();
      integrator.integrate_ele_based_3d(Bele, foundeles, *dmatrix_xb_, *mmatrix_xb_, *dis2_, *dis1_,
          dofseta, dofsetb, *p21_dofrowmap_, *p21_dofcolmap_);
      break;
    }
    default:
    {
      FOUR_C_THROW("ERROR: unknown shape!");
      break;
    }
  }
  return;
}

/*----------------------------------------------------------------------*
 |  Assemble p matrix for cons. interpolation approach       farah 06/14|
 *----------------------------------------------------------------------*/
void Coupling::VolMortar::VolMortarCoupl::assemble_consistent_interpolation_p12(
    Core::Nodes::Node* node, std::vector<int>& foundeles)
{
  static ConsInterpolator interpolator;
  interpolator.interpolate(
      node, *p12_, *dis1_, *dis2_, foundeles, dofset12_, *p12_dofrowmap_, *p12_dofcolmap_);

  return;
}

/*----------------------------------------------------------------------*
 |  Assemble p matrix for cons. interpolation approach       farah 06/14|
 *----------------------------------------------------------------------*/
void Coupling::VolMortar::VolMortarCoupl::assemble_consistent_interpolation_p21(
    Core::Nodes::Node* node, std::vector<int>& foundeles)
{
  static ConsInterpolator interpolator;
  interpolator.interpolate(
      node, *p21_, *dis2_, *dis1_, foundeles, dofset21_, *p21_dofrowmap_, *p21_dofcolmap_);

  return;
}

/*----------------------------------------------------------------------*
 |  integrate_3d Cells for direct divergence approach         farah 04/14|
 *----------------------------------------------------------------------*/
void Coupling::VolMortar::VolMortarCoupl::integrate_3d_cell_direct_divergence(
    Core::Elements::Element& sele, Core::Elements::Element& mele, bool switched_conf)
{
  if ((int)(volcell_.size()) > 1)
    std::cout << "****************************   CELL SIZE > 1 ***************************"
              << std::endl;

  for (Cut::plain_volumecell_set::iterator i = volcell_.begin(); i != volcell_.end(); i++)
  {
    if (*i == nullptr) continue;

    Cut::VolumeCell* vc = *i;

    if (vc->is_negligibly_small()) continue;

    // main gp rule
    std::shared_ptr<Core::FE::GaussPoints> intpoints = vc->get_gauss_rule();

    //--------------------------------------------------------------------
    // loop over cells for A Field

    switch (sele.shape())
    {
      // 2D surface elements
      case Core::FE::CellType::hex8:
      {
        switch (mele.shape())
        {
          // 2D surface elements
          case Core::FE::CellType::hex8:
          {
            static VolMortarIntegrator<Core::FE::CellType::hex8, Core::FE::CellType::hex8>
                integrator(params());
            integrator.integrate_cells_3d_direct_diveregence(sele, mele, *vc, intpoints,
                switched_conf, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_, dofset12_.first,
                dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::tet4:
          {
            static VolMortarIntegrator<Core::FE::CellType::hex8, Core::FE::CellType::tet4>
                integrator(params());
            integrator.integrate_cells_3d_direct_diveregence(sele, mele, *vc, intpoints,
                switched_conf, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_, dofset12_.first,
                dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::pyramid5:
          {
            static VolMortarIntegrator<Core::FE::CellType::hex8, Core::FE::CellType::pyramid5>
                integrator(params());
            integrator.integrate_cells_3d_direct_diveregence(sele, mele, *vc, intpoints,
                switched_conf, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_, dofset12_.first,
                dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          default:
          {
            FOUR_C_THROW("ERROR: unknown shape!");
            break;
          }
        }
        break;
      }
      case Core::FE::CellType::tet4:
      {
        switch (mele.shape())
        {
          // 2D surface elements
          case Core::FE::CellType::hex8:
          {
            static VolMortarIntegrator<Core::FE::CellType::tet4, Core::FE::CellType::hex8>
                integrator(params());
            integrator.integrate_cells_3d_direct_diveregence(sele, mele, *vc, intpoints,
                switched_conf, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_, dofset12_.first,
                dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::tet4:
          {
            static VolMortarIntegrator<Core::FE::CellType::tet4, Core::FE::CellType::tet4>
                integrator(params());
            integrator.integrate_cells_3d_direct_diveregence(sele, mele, *vc, intpoints,
                switched_conf, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_, dofset12_.first,
                dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::pyramid5:
          {
            static VolMortarIntegrator<Core::FE::CellType::tet4, Core::FE::CellType::pyramid5>
                integrator(params());
            integrator.integrate_cells_3d_direct_diveregence(sele, mele, *vc, intpoints,
                switched_conf, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_, dofset12_.first,
                dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          default:
          {
            FOUR_C_THROW("ERROR: unknown shape!");
            break;
          }
        }
        break;
      }
      case Core::FE::CellType::pyramid5:
      {
        switch (mele.shape())
        {
          // 2D surface elements
          case Core::FE::CellType::hex8:
          {
            static VolMortarIntegrator<Core::FE::CellType::pyramid5, Core::FE::CellType::hex8>
                integrator(params());
            integrator.integrate_cells_3d_direct_diveregence(sele, mele, *vc, intpoints,
                switched_conf, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_, dofset12_.first,
                dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::tet4:
          {
            static VolMortarIntegrator<Core::FE::CellType::pyramid5, Core::FE::CellType::tet4>
                integrator(params());
            integrator.integrate_cells_3d_direct_diveregence(sele, mele, *vc, intpoints,
                switched_conf, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_, dofset12_.first,
                dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          case Core::FE::CellType::pyramid5:
          {
            static VolMortarIntegrator<Core::FE::CellType::pyramid5, Core::FE::CellType::pyramid5>
                integrator(params());
            integrator.integrate_cells_3d_direct_diveregence(sele, mele, *vc, intpoints,
                switched_conf, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_, dofset12_.first,
                dofset12_.second, dofset21_.first, dofset21_.second);
            break;
          }
          default:
          {
            FOUR_C_THROW("ERROR: unknown shape!");
            break;
          }
        }
        break;
      }
      default:
      {
        FOUR_C_THROW("ERROR: unknown shape!");
        break;
      }
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Integrate over A-element domain                          farah 01/14|
 *----------------------------------------------------------------------*/
void Coupling::VolMortar::VolMortarCoupl::integrate_3d(
    Core::Elements::Element& sele, Core::Elements::Element& mele, int domain)
{
  //--------------------------------------------------------------------
  // loop over cells for A Field
  switch (sele.shape())
  {
    // 2D surface elements
    case Core::FE::CellType::hex8:
    {
      switch (mele.shape())
      {
        // 2D surface elements
        case Core::FE::CellType::hex8:
        {
          static VolMortarIntegrator<Core::FE::CellType::hex8, Core::FE::CellType::hex8> integrator(
              params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        case Core::FE::CellType::hex20:
        {
          static VolMortarIntegrator<Core::FE::CellType::hex8, Core::FE::CellType::hex20>
              integrator(params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        case Core::FE::CellType::hex27:
        {
          static VolMortarIntegrator<Core::FE::CellType::hex8, Core::FE::CellType::hex27>
              integrator(params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        case Core::FE::CellType::tet4:
        {
          static VolMortarIntegrator<Core::FE::CellType::hex8, Core::FE::CellType::tet4> integrator(
              params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        case Core::FE::CellType::tet10:
        {
          static VolMortarIntegrator<Core::FE::CellType::hex8, Core::FE::CellType::tet10>
              integrator(params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        case Core::FE::CellType::pyramid5:
        {
          static VolMortarIntegrator<Core::FE::CellType::hex8, Core::FE::CellType::pyramid5>
              integrator(params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        default:
        {
          FOUR_C_THROW("ERROR: unknown shape!");
          break;
        }
      }
      break;
    }
    // ########################################################
    case Core::FE::CellType::hex20:
    {
      switch (mele.shape())
      {
        // 2D surface elements
        case Core::FE::CellType::hex8:
        {
          static VolMortarIntegrator<Core::FE::CellType::hex20, Core::FE::CellType::hex8>
              integrator(params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        case Core::FE::CellType::hex20:
        {
          static VolMortarIntegrator<Core::FE::CellType::hex20, Core::FE::CellType::hex20>
              integrator(params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        case Core::FE::CellType::hex27:
        {
          static VolMortarIntegrator<Core::FE::CellType::hex20, Core::FE::CellType::hex27>
              integrator(params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        case Core::FE::CellType::tet4:
        {
          static VolMortarIntegrator<Core::FE::CellType::hex20, Core::FE::CellType::tet4>
              integrator(params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        case Core::FE::CellType::tet10:
        {
          static VolMortarIntegrator<Core::FE::CellType::hex20, Core::FE::CellType::tet10>
              integrator(params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        case Core::FE::CellType::pyramid5:
        {
          static VolMortarIntegrator<Core::FE::CellType::hex20, Core::FE::CellType::pyramid5>
              integrator(params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        default:
        {
          FOUR_C_THROW("ERROR: unknown shape!");
          break;
        }
      }
      break;
    }
    // ########################################################
    case Core::FE::CellType::hex27:
    {
      switch (mele.shape())
      {
        // 2D surface elements
        case Core::FE::CellType::hex8:
        {
          static VolMortarIntegrator<Core::FE::CellType::hex27, Core::FE::CellType::hex8>
              integrator(params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        case Core::FE::CellType::hex20:
        {
          static VolMortarIntegrator<Core::FE::CellType::hex27, Core::FE::CellType::hex20>
              integrator(params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        case Core::FE::CellType::hex27:
        {
          static VolMortarIntegrator<Core::FE::CellType::hex27, Core::FE::CellType::hex27>
              integrator(params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        case Core::FE::CellType::tet4:
        {
          static VolMortarIntegrator<Core::FE::CellType::hex27, Core::FE::CellType::tet4>
              integrator(params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        case Core::FE::CellType::tet10:
        {
          static VolMortarIntegrator<Core::FE::CellType::hex27, Core::FE::CellType::tet10>
              integrator(params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        case Core::FE::CellType::pyramid5:
        {
          static VolMortarIntegrator<Core::FE::CellType::hex27, Core::FE::CellType::pyramid5>
              integrator(params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        default:
        {
          FOUR_C_THROW("ERROR: unknown shape!");
          break;
        }
      }
      break;
    }
    // ########################################################
    case Core::FE::CellType::tet4:
    {
      switch (mele.shape())
      {
        // 2D surface elements
        case Core::FE::CellType::hex8:
        {
          static VolMortarIntegrator<Core::FE::CellType::tet4, Core::FE::CellType::hex8> integrator(
              params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        case Core::FE::CellType::hex20:
        {
          static VolMortarIntegrator<Core::FE::CellType::tet4, Core::FE::CellType::hex20>
              integrator(params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        case Core::FE::CellType::hex27:
        {
          static VolMortarIntegrator<Core::FE::CellType::tet4, Core::FE::CellType::hex27>
              integrator(params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        case Core::FE::CellType::tet4:
        {
          static VolMortarIntegrator<Core::FE::CellType::tet4, Core::FE::CellType::tet4> integrator(
              params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        case Core::FE::CellType::tet10:
        {
          static VolMortarIntegrator<Core::FE::CellType::tet4, Core::FE::CellType::tet10>
              integrator(params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        case Core::FE::CellType::pyramid5:
        {
          static VolMortarIntegrator<Core::FE::CellType::tet4, Core::FE::CellType::pyramid5>
              integrator(params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        default:
        {
          FOUR_C_THROW("ERROR: unknown shape!");
          break;
        }
      }
      break;
    }
    // ########################################################
    case Core::FE::CellType::tet10:
    {
      switch (mele.shape())
      {
        // 2D surface elements
        case Core::FE::CellType::hex8:
        {
          static VolMortarIntegrator<Core::FE::CellType::tet10, Core::FE::CellType::hex8>
              integrator(params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        case Core::FE::CellType::hex20:
        {
          static VolMortarIntegrator<Core::FE::CellType::tet10, Core::FE::CellType::hex20>
              integrator(params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        case Core::FE::CellType::hex27:
        {
          static VolMortarIntegrator<Core::FE::CellType::tet10, Core::FE::CellType::hex27>
              integrator(params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        case Core::FE::CellType::tet4:
        {
          static VolMortarIntegrator<Core::FE::CellType::tet10, Core::FE::CellType::tet4>
              integrator(params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        case Core::FE::CellType::tet10:
        {
          static VolMortarIntegrator<Core::FE::CellType::tet10, Core::FE::CellType::tet10>
              integrator(params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        case Core::FE::CellType::pyramid5:
        {
          static VolMortarIntegrator<Core::FE::CellType::tet10, Core::FE::CellType::pyramid5>
              integrator(params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        default:
        {
          FOUR_C_THROW("ERROR: unknown shape!");
          break;
        }
      }
      break;
    }
    // ########################################################
    case Core::FE::CellType::pyramid5:
    {
      switch (mele.shape())
      {
        // 2D surface elements
        case Core::FE::CellType::hex8:
        {
          static VolMortarIntegrator<Core::FE::CellType::pyramid5, Core::FE::CellType::hex8>
              integrator(params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        case Core::FE::CellType::hex20:
        {
          static VolMortarIntegrator<Core::FE::CellType::pyramid5, Core::FE::CellType::hex20>
              integrator(params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        case Core::FE::CellType::hex27:
        {
          static VolMortarIntegrator<Core::FE::CellType::pyramid5, Core::FE::CellType::hex27>
              integrator(params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        case Core::FE::CellType::tet4:
        {
          static VolMortarIntegrator<Core::FE::CellType::pyramid5, Core::FE::CellType::tet4>
              integrator(params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        case Core::FE::CellType::tet10:
        {
          static VolMortarIntegrator<Core::FE::CellType::pyramid5, Core::FE::CellType::tet10>
              integrator(params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        case Core::FE::CellType::pyramid5:
        {
          static VolMortarIntegrator<Core::FE::CellType::pyramid5, Core::FE::CellType::pyramid5>
              integrator(params());
          integrator.initialize_gp(true, domain);
          integrator.integrate_ele_3d(domain, sele, mele, *d1_, *m12_, *d2_, *m21_, *dis1_, *dis2_,
              dofset12_.first, dofset12_.second, dofset21_.first, dofset21_.second);
          break;
        }
        default:
        {
          FOUR_C_THROW("ERROR: unknown shape!");
          break;
        }
      }
      break;
    }
    default:
    {
      FOUR_C_THROW("ERROR: unknown shape!");
      break;
    }
  }

  // integration element counter
  inteles_++;

  return;
}

/*----------------------------------------------------------------------*
 |  init (public)                                            farah 10/13|
 *----------------------------------------------------------------------*/
void Coupling::VolMortar::VolMortarCoupl::initialize()
{
  /* ******************************************************************
   * (re)setup global Mortar Core::LinAlg::SparseMatrices                   *
   * unknowns which are going to be condensed are defined on the slave*
   * side. Therefore, the rows are the auxiliary variables on the     *
   * slave side!                                                      *
   * ******************************************************************/

  d1_ = std::make_shared<Core::LinAlg::SparseMatrix>(*p12_dofrowmap_, 10);
  m12_ = std::make_shared<Core::LinAlg::SparseMatrix>(*p12_dofrowmap_, 100);

  d2_ = std::make_shared<Core::LinAlg::SparseMatrix>(*p21_dofrowmap_, 10);
  m21_ = std::make_shared<Core::LinAlg::SparseMatrix>(*p21_dofrowmap_, 100);

  // initialize trafo operator for quadr. modification
  if (dualquad_ != dualquad_no_mod)
  {
    t1_ = std::make_shared<Core::LinAlg::SparseMatrix>(*p12_dofrowmap_, 10);
    t2_ = std::make_shared<Core::LinAlg::SparseMatrix>(*p21_dofrowmap_, 10);
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Complete (public)                                        farah 01/14|
 *----------------------------------------------------------------------*/
void Coupling::VolMortar::VolMortarCoupl::complete()
{
  // complete...
  d1_->complete(*p12_dofrowmap_, *p12_dofrowmap_);
  m12_->complete(*p12_dofdomainmap_, *p12_dofrowmap_);

  d2_->complete(*p21_dofrowmap_, *p21_dofrowmap_);
  m21_->complete(*p21_dofdomainmap_, *p21_dofrowmap_);

  // complete trafo operator for quadr. modification
  if (dualquad_ != dualquad_no_mod)
  {
    t1_->complete(*p12_dofrowmap_, *p12_dofrowmap_);
    t2_->complete(*p21_dofrowmap_, *p21_dofrowmap_);
  }

  return;
}

/*----------------------------------------------------------------------*
 |  compute projection operator P                            farah 01/14|
 *----------------------------------------------------------------------*/
void Coupling::VolMortar::VolMortarCoupl::create_projection_operator()
{
  /********************************************************************/
  /* Multiply Mortar matrices: P = inv(D) * M         A               */
  /********************************************************************/
  Core::LinAlg::SparseMatrix invd1(*d1_);
  std::shared_ptr<Core::LinAlg::Vector<double>> diag1 =
      Core::LinAlg::create_vector(*p12_dofrowmap_, true);

  // extract diagonal of invd into diag
  invd1.extract_diagonal_copy(*diag1);

  // set zero diagonal values to dummy 1.0
  for (int i = 0; i < diag1->local_length(); ++i)
    if (abs((*diag1)[i]) < 1e-12) (*diag1).get_values()[i] = 1.0;

  // scalar inversion of diagonal values
  diag1->reciprocal(*diag1);

  // re-insert inverted diagonal into invd
  invd1.replace_diagonal_values(*diag1);

  // do the multiplication P = inv(D) * M
  std::shared_ptr<Core::LinAlg::SparseMatrix> aux12 =
      Core::LinAlg::matrix_multiply(invd1, false, *m12_, false, false, false, true);

  /********************************************************************/
  /* Multiply Mortar matrices: P = inv(D) * M         B               */
  /********************************************************************/
  Core::LinAlg::SparseMatrix invd2(*d2_);
  std::shared_ptr<Core::LinAlg::Vector<double>> diag2 =
      Core::LinAlg::create_vector(*p21_dofrowmap_, true);

  // extract diagonal of invd into diag
  invd2.extract_diagonal_copy(*diag2);

  // set zero diagonal values to dummy 1.0
  for (int i = 0; i < diag2->local_length(); ++i)
    if (abs((*diag2)[i]) < 1e-12) (*diag2).get_values()[i] = 1.0;

  // scalar inversion of diagonal values
  diag2->reciprocal(*diag2);

  // re-insert inverted diagonal into invd
  invd2.replace_diagonal_values(*diag2);

  // do the multiplication P = inv(D) * M
  std::shared_ptr<Core::LinAlg::SparseMatrix> aux21 =
      Core::LinAlg::matrix_multiply(invd2, false, *m21_, false, false, false, true);

  // initialize trafo operator for quadr. modification
  if (dualquad_ != dualquad_no_mod)
  {
    p12_ = Core::LinAlg::matrix_multiply(*t1_, false, *aux12, false, false, false, true);
    p21_ = Core::LinAlg::matrix_multiply(*t2_, false, *aux21, false, false, false, true);
  }
  else
  {
    p12_ = aux12;
    p21_ = aux21;
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Define polygon of mortar vertices                        farah 01/14|
 *----------------------------------------------------------------------*/
void Coupling::VolMortar::VolMortarCoupl::define_vertices_slave(
    Core::Elements::Element& ele, std::vector<Mortar::Vertex>& slave_vertices)
{
  // project slave nodes onto auxiliary plane
  int nnodes = ele.num_node();
  Core::Nodes::Node** mynodes = ele.nodes();
  if (!mynodes) FOUR_C_THROW("ERROR: project_slave: Null pointer!");

  // initialize storage for slave coords + their ids
  std::vector<double> vertices(3);
  std::vector<int> snodeids(1);

  for (int i = 0; i < nnodes; ++i)
  {
    // compute projection
    const auto x = mynodes[i]->x();
    for (size_t k = 0; k < x.size(); ++k) vertices[k] = x[k];

    // get node id, too
    snodeids[0] = mynodes[i]->id();

    // store into vertex data structure
    slave_vertices.push_back(Mortar::Vertex(
        vertices, Mortar::Vertex::slave, snodeids, nullptr, nullptr, false, false, nullptr, -1.0));
  }
  return;
}

/*----------------------------------------------------------------------*
 |  Define polygon of mortar vertices                        farah 01/14|
 *----------------------------------------------------------------------*/
void Coupling::VolMortar::VolMortarCoupl::define_vertices_master(
    Core::Elements::Element& ele, std::vector<Mortar::Vertex>& slave_vertices)
{
  // project slave nodes onto auxiliary plane
  int nnodes = ele.num_node();
  Core::Nodes::Node** mynodes = ele.nodes();
  if (!mynodes) FOUR_C_THROW("ERROR: project_slave: Null pointer!");

  // initialize storage for slave coords + their ids
  std::vector<double> vertices(3);
  std::vector<int> snodeids(1);

  for (int i = 0; i < nnodes; ++i)
  {
    // compute projection
    const auto x = mynodes[i]->x();
    for (size_t k = 0; k < x.size(); ++k) vertices[k] = x[k];

    // get node id, too
    snodeids[0] = mynodes[i]->id();

    // store into vertex data structure
    slave_vertices.push_back(Mortar::Vertex(vertices, Mortar::Vertex::projmaster, snodeids, nullptr,
        nullptr, false, false, nullptr, -1.0));
  }
  return;
}

/*----------------------------------------------------------------------*
 |  Clipping of two polygons (NEW version)                    popp 11/09|
 *----------------------------------------------------------------------*/
bool Coupling::VolMortar::VolMortarCoupl::polygon_clipping_convex_hull(
    std::vector<Mortar::Vertex>& poly1, std::vector<Mortar::Vertex>& poly2,
    std::vector<Mortar::Vertex>& respoly, Core::Elements::Element& sele,
    Core::Elements::Element& mele, double& tol)
{
  //**********************************************************************
  // STEP1: Input check
  // - input polygons must consist of min. 3 vertices each
  // - rotation of poly1 must be c-clockwise w.r.t. (0,0,1) or Auxn()
  // - rotation of poly 2 changed to c-clockwise w.r.t. (0,0,1) or Auxn()
  // - both input polygons must be convex
  //**********************************************************************

  // check input variables
  if ((int)poly1.size() < 3 || (int)poly2.size() < 3)
    FOUR_C_THROW("ERROR: Input Polygons must consist of min. 3 vertices each");

  // check for rotation of polygon1 (slave) and polygon 2 (master)
  // note that we implicitly already rely on convexity here!
  // first get geometric centers of polygon1 and polygon2
  std::array<double, 3> center1 = {0.0, 0.0, 0.0};
  std::array<double, 3> center2 = {0.0, 0.0, 0.0};

  for (int i = 0; i < (int)poly1.size(); ++i)
    for (int k = 0; k < 3; ++k) center1[k] += poly1[i].coord()[k] / ((int)poly1.size());

  for (int i = 0; i < (int)poly2.size(); ++i)
    for (int k = 0; k < 3; ++k) center2[k] += poly2[i].coord()[k] / ((int)poly2.size());

  // then we compute the counter-clockwise plane normal
  std::array<double, 3> diff1 = {0.0, 0.0, 0.0};
  std::array<double, 3> edge1 = {0.0, 0.0, 0.0};
  std::array<double, 3> diff2 = {0.0, 0.0, 0.0};
  std::array<double, 3> edge2 = {0.0, 0.0, 0.0};

  for (int k = 0; k < 3; ++k)
  {
    diff1[k] = poly1[0].coord()[k] - center1[k];
    edge1[k] = poly1[1].coord()[k] - poly1[0].coord()[k];
    diff2[k] = poly2[0].coord()[k] - center2[k];
    edge2[k] = poly2[1].coord()[k] - poly2[0].coord()[k];
  }

  std::array<double, 3> cross1 = {0.0, 0.0, 0.0};
  std::array<double, 3> cross2 = {0.0, 0.0, 0.0};

  cross1[0] = diff1[1] * edge1[2] - diff1[2] * edge1[1];
  cross1[1] = diff1[2] * edge1[0] - diff1[0] * edge1[2];
  cross1[2] = diff1[0] * edge1[1] - diff1[1] * edge1[0];

  cross2[0] = diff2[1] * edge2[2] - diff2[2] * edge2[1];
  cross2[1] = diff2[2] * edge2[0] - diff2[0] * edge2[2];
  cross2[2] = diff2[0] * edge2[1] - diff2[1] * edge2[0];

  // check against auxiliary plane normal
  double check1 = cross1[0] * auxn()[0] + cross1[1] * auxn()[1] + cross1[2] * auxn()[2];
  double check2 = cross2[0] * auxn()[0] + cross2[1] * auxn()[1] + cross2[2] * auxn()[2];

  // check polygon 1 and throw dserror if not c-clockwise
  if (check1 <= 0) FOUR_C_THROW("ERROR: Polygon 1 (slave) not ordered counter-clockwise!");

  // check polygon 2 and reorder in c-clockwise direction
  if (check2 < 0) std::reverse(poly2.begin(), poly2.end());

  // check if the two input polygons are convex
  // a polygon is convex if the scalar product of an edge normal and the
  // next edge direction is negative for all edges
  for (int i = 0; i < (int)poly1.size(); ++i)
  {
    // we need the edge vector first
    std::array<double, 3> edge = {0.0, 0.0, 0.0};
    for (int k = 0; k < 3; ++k)
    {
      if (i != (int)poly1.size() - 1)
        edge[k] = poly1[i + 1].coord()[k] - poly1[i].coord()[k];
      else
        edge[k] = poly1[0].coord()[k] - poly1[i].coord()[k];
    }

    // edge normal is result of cross product
    std::array<double, 3> n = {0.0, 0.0, 0.0};
    n[0] = edge[1] * auxn()[2] - edge[2] * auxn()[1];
    n[1] = edge[2] * auxn()[0] - edge[0] * auxn()[2];
    n[2] = edge[0] * auxn()[1] - edge[1] * auxn()[0];

    // we need the next edge vector now
    std::array<double, 3> nextedge = {0.0, 0.0, 0.0};
    for (int k = 0; k < 3; ++k)
    {
      if (i < (int)poly1.size() - 2)
        nextedge[k] = poly1[i + 2].coord()[k] - poly1[i + 1].coord()[k];
      else if (i == (int)poly1.size() - 2)
        nextedge[k] = poly1[0].coord()[k] - poly1[i + 1].coord()[k];
      else
        nextedge[k] = poly1[1].coord()[k] - poly1[0].coord()[k];
    }

    // check scalar product
    double check = n[0] * nextedge[0] + n[1] * nextedge[1] + n[2] * nextedge[2];
    if (check > 0) FOUR_C_THROW("ERROR: Input polygon 1 not convex");
  }

  for (int i = 0; i < (int)poly2.size(); ++i)
  {
    // we need the edge vector first
    std::array<double, 3> edge = {0.0, 0.0, 0.0};
    for (int k = 0; k < 3; ++k)
    {
      if (i != (int)poly2.size() - 1)
        edge[k] = poly2[i + 1].coord()[k] - poly2[i].coord()[k];
      else
        edge[k] = poly2[0].coord()[k] - poly2[i].coord()[k];
    }

    // edge normal is result of cross product
    std::array<double, 3> n = {0.0, 0.0, 0.0};
    n[0] = edge[1] * auxn()[2] - edge[2] * auxn()[1];
    n[1] = edge[2] * auxn()[0] - edge[0] * auxn()[2];
    n[2] = edge[0] * auxn()[1] - edge[1] * auxn()[0];

    // we need the next edge vector now
    std::array<double, 3> nextedge = {0.0, 0.0, 0.0};
    for (int k = 0; k < 3; ++k)
    {
      if (i < (int)poly2.size() - 2)
        nextedge[k] = poly2[i + 2].coord()[k] - poly2[i + 1].coord()[k];
      else if (i == (int)poly2.size() - 2)
        nextedge[k] = poly2[0].coord()[k] - poly2[i + 1].coord()[k];
      else
        nextedge[k] = poly2[1].coord()[k] - poly2[0].coord()[k];
    }

    // check scalar product
    double check = n[0] * nextedge[0] + n[1] * nextedge[1] + n[2] * nextedge[2];
    if (check > 0)
    {
      // this may happen, so do NOT throw an error immediately
      // but instead check if the two elements to be clipped are
      // close to each other at all. If so, then really throw the
      // dserror, if not, simply continue with the next pair!
      int sid = sele.id();
      int mid = mele.id();
      bool nearcheck = true;  // rough_check_nodes();
      if (nearcheck)
      {
        std::cout << "***WARNING*** Input polygon 2 not convex! (S/M-pair: " << sid << "/" << mid
                  << ")" << std::endl;
      }

      return false;
    }
  }

  //**********************************************************************
  // STEP2: Extend Vertex data structures
  // - note that poly1 is the slave element and poly2 the master element
  // - assign Next() and Prev() pointers to initialize linked structure
  //**********************************************************************
  // set previous and next Vertex pointer for all elements in lists
  for (int i = 0; i < (int)poly1.size(); ++i)
  {
    // standard case
    if (i != 0 && i != (int)poly1.size() - 1)
    {
      poly1[i].assign_next(&poly1[i + 1]);
      poly1[i].assign_prev(&poly1[i - 1]);
    }
    // first element in list
    else if (i == 0)
    {
      poly1[i].assign_next(&poly1[i + 1]);
      poly1[i].assign_prev(&poly1[(int)poly1.size() - 1]);
    }
    // last element in list
    else
    {
      poly1[i].assign_next(&poly1[0]);
      poly1[i].assign_prev(&poly1[i - 1]);
    }
  }
  for (int i = 0; i < (int)poly2.size(); ++i)
  {
    // standard case
    if (i != 0 && i != (int)poly2.size() - 1)
    {
      poly2[i].assign_next(&poly2[i + 1]);
      poly2[i].assign_prev(&poly2[i - 1]);
    }
    // first element in list
    else if (i == 0)
    {
      poly2[i].assign_next(&poly2[i + 1]);
      poly2[i].assign_prev(&poly2[(int)poly2.size() - 1]);
    }
    // last element in list
    else
    {
      poly2[i].assign_next(&poly2[0]);
      poly2[i].assign_prev(&poly2[i - 1]);
    }
  }

  //**********************************************************************
  // STEP3: Perform line intersection of all edge pairs
  // - this yields a new vector of intersection vertices
  // - by default the respective edge end vertices are assumed to be
  //   the next/prev vertices and connectivity is set up accordingly
  //**********************************************************************
  std::vector<Mortar::Vertex> intersec;

  for (int i = 0; i < (int)poly1.size(); ++i)
  {
    for (int j = 0; j < (int)poly2.size(); ++j)
    {
      // we need two edges first
      std::array<double, 3> edge1 = {0.0, 0.0, 0.0};
      std::array<double, 3> edge2 = {0.0, 0.0, 0.0};
      for (int k = 0; k < 3; ++k)
      {
        edge1[k] = (poly1[i].next())->coord()[k] - poly1[i].coord()[k];
        edge2[k] = (poly2[j].next())->coord()[k] - poly2[j].coord()[k];
      }

      // outward edge normals of polygon 1 and 2 edges
      std::array<double, 3> n1 = {0.0, 0.0, 0.0};
      std::array<double, 3> n2 = {0.0, 0.0, 0.0};
      n1[0] = edge1[1] * auxn()[2] - edge1[2] * auxn()[1];
      n1[1] = edge1[2] * auxn()[0] - edge1[0] * auxn()[2];
      n1[2] = edge1[0] * auxn()[1] - edge1[1] * auxn()[0];
      n2[0] = edge2[1] * auxn()[2] - edge2[2] * auxn()[1];
      n2[1] = edge2[2] * auxn()[0] - edge2[0] * auxn()[2];
      n2[2] = edge2[0] * auxn()[1] - edge2[1] * auxn()[0];

      // check for parallelity of edges
      double parallel = edge1[0] * n2[0] + edge1[1] * n2[1] + edge1[2] * n2[2];
      if (abs(parallel) < tol) continue;

      // check for intersection of non-parallel edges
      double wec_p1 = 0.0;
      double wec_p2 = 0.0;
      for (int k = 0; k < 3; ++k)
      {
        wec_p1 += (poly1[i].coord()[k] - poly2[j].coord()[k]) * n2[k];
        wec_p2 += ((poly1[i].next())->coord()[k] - poly2[j].coord()[k]) * n2[k];
      }

      if (wec_p1 * wec_p2 <= 0)
      {
        double wec_q1 = 0.0;
        double wec_q2 = 0.0;
        for (int k = 0; k < 3; ++k)
        {
          wec_q1 += (poly2[j].coord()[k] - poly1[i].coord()[k]) * n1[k];
          wec_q2 += ((poly2[j].next())->coord()[k] - poly1[i].coord()[k]) * n1[k];
        }

        if (wec_q1 * wec_q2 <= 0)
        {
          double alphap = wec_p1 / (wec_p1 - wec_p2);
          double alphaq = wec_q1 / (wec_q1 - wec_q2);
          std::vector<double> ip(3);
          std::vector<double> iq(3);
          for (int k = 0; k < 3; ++k)
          {
            ip[k] = (1 - alphap) * poly1[i].coord()[k] + alphap * (poly1[i].next())->coord()[k];
            iq[k] = (1 - alphaq) * poly2[j].coord()[k] + alphaq * (poly2[j].next())->coord()[k];
            if (abs(ip[k]) < tol) ip[k] = 0.0;
            if (abs(iq[k]) < tol) iq[k] = 0.0;
          }

          // generate vectors of underlying node ids for lineclip (2x slave, 2x master)
          std::vector<int> lcids(4);
          lcids[0] = (int)(poly1[i].nodeids()[0]);
          lcids[1] = (int)((poly1[i].next())->nodeids()[0]);
          lcids[2] = (int)(poly2[j].nodeids()[0]);
          lcids[3] = (int)((poly2[j].next())->nodeids()[0]);

          // store intersection points
          intersec.push_back(Mortar::Vertex(ip, Mortar::Vertex::lineclip, lcids, poly1[i].next(),
              &poly1[i], true, false, nullptr, alphap));
        }
      }
    }
  }

  //**********************************************************************
  // STEP4: Collapse line intersections
  // - this yields a collapsed vector of intersection vertices
  // - those intersection points close to poly1/poly2 vertices are deleted
  //**********************************************************************
  std::vector<Mortar::Vertex> collintersec;

  for (int i = 0; i < (int)intersec.size(); ++i)
  {
    // keep track of comparisons
    bool close = false;

    // check against all poly1 (slave) points
    for (int j = 0; j < (int)poly1.size(); ++j)
    {
      // distance vector
      std::array<double, 3> diff = {0.0, 0.0, 0.0};
      for (int k = 0; k < 3; ++k) diff[k] = intersec[i].coord()[k] - poly1[j].coord()[k];
      double dist = sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);

      // only keep intersection point if not close
      if (dist <= tol)
      {
        close = true;
        break;
      }
    }

    // do only if no close poly1 (slave) point found
    if (!close)
    {
      // check against all poly2 (master) points
      for (int j = 0; j < (int)poly2.size(); ++j)
      {
        // distance vector
        std::array<double, 3> diff = {0.0, 0.0, 0.0};
        for (int k = 0; k < 3; ++k) diff[k] = intersec[i].coord()[k] - poly2[j].coord()[k];
        double dist = sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);

        // only keep intersection point if not close
        if (dist <= tol)
        {
          close = true;
          break;
        }
      }
    }

    // keep intersection point only if not close to any poly1/poly2 point
    if (!close) collintersec.push_back(intersec[i]);
  }

  //**********************************************************************
  // STEP5: Create points of convex hull
  // - check all poly1 points against all poly1/poly2 edges
  // - check all poly2 points against all poly1/poly2 edges
  // - check all collintersec points against all poly1/poly2 edges
  // - points outside any poly1/poly2 edge are NOT in the convex hull
  // - as a result we obtain all points forming the convex hull
  //**********************************************************************
  std::vector<Mortar::Vertex> convexhull;

  //----------------------------------------------------check poly1 points
  for (int i = 0; i < (int)poly1.size(); ++i)
  {
    // keep track of inside / outside status
    bool outside = false;

    // check against all poly1 (slave) edges
    for (int j = 0; j < (int)poly1.size(); ++j)
    {
      // we need diff vector and edge2 first
      std::array<double, 3> diff = {0.0, 0.0, 0.0};
      std::array<double, 3> edge = {0.0, 0.0, 0.0};
      for (int k = 0; k < 3; ++k)
      {
        diff[k] = poly1[i].coord()[k] - poly1[j].coord()[k];
        edge[k] = (poly1[j].next())->coord()[k] - poly1[j].coord()[k];
      }

      // compute distance from point on poly1 to edge
      std::array<double, 3> n = {0.0, 0.0, 0.0};
      n[0] = edge[1] * auxn()[2] - edge[2] * auxn()[1];
      n[1] = edge[2] * auxn()[0] - edge[0] * auxn()[2];
      n[2] = edge[0] * auxn()[1] - edge[1] * auxn()[0];
      double ln = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
      for (int k = 0; k < 3; ++k) n[k] /= ln;

      double dist = diff[0] * n[0] + diff[1] * n[1] + diff[2] * n[2];

      // only keep point if not in outside halfspace
      if (dist > tol)
      {
        outside = true;
        break;
      }
    }

    // do only if not already outside w.r.t. to a poly1 (slave) edge
    if (!outside)
    {
      // check against all poly2 (master) edges
      for (int j = 0; j < (int)poly2.size(); ++j)
      {
        // we need diff vector and edge2 first
        std::array<double, 3> diff = {0.0, 0.0, 0.0};
        std::array<double, 3> edge = {0.0, 0.0, 0.0};
        for (int k = 0; k < 3; ++k)
        {
          diff[k] = poly1[i].coord()[k] - poly2[j].coord()[k];
          edge[k] = (poly2[j].next())->coord()[k] - poly2[j].coord()[k];
        }

        // compute distance from point on poly1 to edge
        std::array<double, 3> n = {0.0, 0.0, 0.0};
        n[0] = edge[1] * auxn()[2] - edge[2] * auxn()[1];
        n[1] = edge[2] * auxn()[0] - edge[0] * auxn()[2];
        n[2] = edge[0] * auxn()[1] - edge[1] * auxn()[0];
        double ln = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        for (int k = 0; k < 3; ++k) n[k] /= ln;

        double dist = diff[0] * n[0] + diff[1] * n[1] + diff[2] * n[2];

        // only keep point if not in outside halfspace
        if (dist > tol)
        {
          outside = true;
          break;
        }
      }
    }

    // only keep point if never in outside halfspace
    if (!outside) convexhull.push_back(poly1[i]);
  }

  //----------------------------------------------------check poly2 points
  for (int i = 0; i < (int)poly2.size(); ++i)
  {
    // keep track of inside / outside status
    bool outside = false;

    // check against all poly1 (slave) edges
    for (int j = 0; j < (int)poly1.size(); ++j)
    {
      // we need diff vector and edge2 first
      std::array<double, 3> diff = {0.0, 0.0, 0.0};
      std::array<double, 3> edge = {0.0, 0.0, 0.0};
      for (int k = 0; k < 3; ++k)
      {
        diff[k] = poly2[i].coord()[k] - poly1[j].coord()[k];
        edge[k] = (poly1[j].next())->coord()[k] - poly1[j].coord()[k];
      }

      // compute distance from point on poly1 to edge
      std::array<double, 3> n = {0.0, 0.0, 0.0};
      n[0] = edge[1] * auxn()[2] - edge[2] * auxn()[1];
      n[1] = edge[2] * auxn()[0] - edge[0] * auxn()[2];
      n[2] = edge[0] * auxn()[1] - edge[1] * auxn()[0];
      double ln = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
      for (int k = 0; k < 3; ++k) n[k] /= ln;

      double dist = diff[0] * n[0] + diff[1] * n[1] + diff[2] * n[2];

      // only keep point if not in outside halfspace
      if (dist > tol)
      {
        outside = true;
        break;
      }
    }

    // do only if not already outside w.r.t. to a poly1 (slave) edge
    if (!outside)
    {
      // check against all poly2 (master) edges
      for (int j = 0; j < (int)poly2.size(); ++j)
      {
        // we need diff vector and edge2 first
        std::array<double, 3> diff = {0.0, 0.0, 0.0};
        std::array<double, 3> edge = {0.0, 0.0, 0.0};
        for (int k = 0; k < 3; ++k)
        {
          diff[k] = poly2[i].coord()[k] - poly2[j].coord()[k];
          edge[k] = (poly2[j].next())->coord()[k] - poly2[j].coord()[k];
        }

        // compute distance from point on poly1 to edge
        std::array<double, 3> n = {0.0, 0.0, 0.0};
        n[0] = edge[1] * auxn()[2] - edge[2] * auxn()[1];
        n[1] = edge[2] * auxn()[0] - edge[0] * auxn()[2];
        n[2] = edge[0] * auxn()[1] - edge[1] * auxn()[0];
        double ln = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        for (int k = 0; k < 3; ++k) n[k] /= ln;

        double dist = diff[0] * n[0] + diff[1] * n[1] + diff[2] * n[2];

        // only keep point if not in outside halfspace
        if (dist > tol)
        {
          outside = true;
          break;
        }
      }
    }

    // only keep point if never in outside halfspace
    if (!outside) convexhull.push_back(poly2[i]);
  }

  //---------------------------------------------check collintersec points
  for (int i = 0; i < (int)collintersec.size(); ++i)
  {
    // keep track of inside / outside status
    bool outside = false;

    // check against all poly1 (slave) edges
    for (int j = 0; j < (int)poly1.size(); ++j)
    {
      // we need diff vector and edge2 first
      std::array<double, 3> diff = {0.0, 0.0, 0.0};
      std::array<double, 3> edge = {0.0, 0.0, 0.0};
      for (int k = 0; k < 3; ++k)
      {
        diff[k] = collintersec[i].coord()[k] - poly1[j].coord()[k];
        edge[k] = (poly1[j].next())->coord()[k] - poly1[j].coord()[k];
      }

      // compute distance from point on poly1 to edge
      std::array<double, 3> n = {0.0, 0.0, 0.0};
      n[0] = edge[1] * auxn()[2] - edge[2] * auxn()[1];
      n[1] = edge[2] * auxn()[0] - edge[0] * auxn()[2];
      n[2] = edge[0] * auxn()[1] - edge[1] * auxn()[0];
      double ln = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
      for (int k = 0; k < 3; ++k) n[k] /= ln;

      double dist = diff[0] * n[0] + diff[1] * n[1] + diff[2] * n[2];

      // only keep point if not in outside halfspace
      if (dist > tol)
      {
        outside = true;
        break;
      }
    }

    // do only if not already outside w.r.t. to a poly1 (slave) edge
    if (!outside)
    {
      // check against all poly2 (master) edges
      for (int j = 0; j < (int)poly2.size(); ++j)
      {
        // we need diff vector and edge2 first
        std::array<double, 3> diff = {0.0, 0.0, 0.0};
        std::array<double, 3> edge = {0.0, 0.0, 0.0};
        for (int k = 0; k < 3; ++k)
        {
          diff[k] = collintersec[i].coord()[k] - poly2[j].coord()[k];
          edge[k] = (poly2[j].next())->coord()[k] - poly2[j].coord()[k];
        }

        // compute distance from point on poly1 to edge
        std::array<double, 3> n = {0.0, 0.0, 0.0};
        n[0] = edge[1] * auxn()[2] - edge[2] * auxn()[1];
        n[1] = edge[2] * auxn()[0] - edge[0] * auxn()[2];
        n[2] = edge[0] * auxn()[1] - edge[1] * auxn()[0];
        double ln = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        for (int k = 0; k < 3; ++k) n[k] /= ln;

        double dist = diff[0] * n[0] + diff[1] * n[1] + diff[2] * n[2];

        // only keep point if not in outside halfspace
        if (dist > tol)
        {
          outside = true;
          break;
        }
      }
    }

    // only keep point if never in outside halfspace
    if (!outside) convexhull.push_back(collintersec[i]);
  }

  //**********************************************************************
  // STEP6: Collapse convex hull points
  // - this yields a collapsed vector of convex hull vertices
  // - up to now only duplicate intersection points have been eliminated
  // - this operation now removes ALL kinds of duplicate points
  // - intersection points close to poly2/poly1 points are deleted
  // - poly2 points close to poly1 vertices are deleted
  //**********************************************************************
  std::vector<Mortar::Vertex> collconvexhull;

  for (int i = 0; i < (int)convexhull.size(); ++i)
  {
    // keep track of comparisons
    bool close = false;

    // do not collapse poly1 (slave) points
    if (convexhull[i].v_type() == Mortar::Vertex::slave)
    {
      collconvexhull.push_back(convexhull[i]);
      continue;
    }

    // check remaining poly2 (master) and intersec points against poly1 (slave) points
    for (int j = 0; j < (int)convexhull.size(); ++j)
    {
      // only collapse with poly1 (slave) points
      if (convexhull[j].v_type() != Mortar::Vertex::slave) continue;

      // distance vector
      std::array<double, 3> diff = {0.0, 0.0, 0.0};
      for (int k = 0; k < 3; ++k) diff[k] = convexhull[i].coord()[k] - convexhull[j].coord()[k];
      double dist = sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);

      // only keep point if not close
      if (dist <= tol)
      {
        close = true;
        break;
      }
    }

    // do not check poly2 (master) points
    if (convexhull[i].v_type() == Mortar::Vertex::projmaster)
    {
      if (!close) collconvexhull.push_back(convexhull[i]);
      continue;
    }

    // check intersec points against poly2 (master) points
    if (!close && convexhull[i].v_type() == Mortar::Vertex::lineclip)
    {
      for (int j = 0; j < (int)convexhull.size(); ++j)
      {
        // only collapse with poly2 (master) points
        if (convexhull[j].v_type() != Mortar::Vertex::projmaster) continue;

        // distance vector
        std::array<double, 3> diff = {0.0, 0.0, 0.0};
        for (int k = 0; k < 3; ++k) diff[k] = convexhull[i].coord()[k] - convexhull[j].coord()[k];
        double dist = sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);

        // only keep intersection point if not close
        if (dist <= tol)
        {
          close = true;
          break;
        }
      }
    }

    // keep intersection point only if not collapsed
    if (!close) collconvexhull.push_back(convexhull[i]);
  }

  //**********************************************************************
  // STEP7: Transform convex hull points to auxiliary plane
  // - x* = A * (x - p1) where p1 = translation, A = rotation
  // - this yields transformed points with coords (x_bar,y_bar,0)
  //**********************************************************************
  // only continue if more than two points remaining
  if ((int)collconvexhull.size() < 3)
  {
    // no clip polygon if less than 3 points
    std::vector<Mortar::Vertex> empty;
    respoly = empty;
  }
  else if ((int)collconvexhull.size() == 3)
  {
    // 3 points already ARE the convex hull
    respoly = collconvexhull;
  }
  else
  {
    // do transformation to auxiliary plane
    std::array<double, 3> newzero = {
        collconvexhull[0].coord()[0], collconvexhull[0].coord()[1], collconvexhull[0].coord()[2]};
    std::array<double, 3> newxaxis = {collconvexhull[1].coord()[0] - collconvexhull[0].coord()[0],
        collconvexhull[1].coord()[1] - collconvexhull[0].coord()[1],
        collconvexhull[1].coord()[2] - collconvexhull[0].coord()[2]};
    std::array<double, 3> newyaxis = {auxn()[1] * newxaxis[2] - auxn()[2] * newxaxis[1],
        auxn()[2] * newxaxis[0] - auxn()[0] * newxaxis[2],
        auxn()[0] * newxaxis[1] - auxn()[1] * newxaxis[0]};
    double lnewxaxis =
        sqrt(newxaxis[0] * newxaxis[0] + newxaxis[1] * newxaxis[1] + newxaxis[2] * newxaxis[2]);
    double lnewyaxis =
        sqrt(newyaxis[0] * newyaxis[0] + newyaxis[1] * newyaxis[1] + newyaxis[2] * newyaxis[2]);

    //    for(int h=0;h<(int)collconvexhull.size();++h)
    //      std::cout << "Coord "<< h <<"= " << collconvexhull[h].Coord()[0] << " " <<
    //      collconvexhull[h].Coord()[1] << " " << collconvexhull[h].Coord()[2] << std::endl;

    // normalize
    for (int k = 0; k < 3; ++k)
    {
      newxaxis[k] /= lnewxaxis;
      newyaxis[k] /= lnewyaxis;
    }

    // trafo matrix
    Core::LinAlg::Matrix<3, 3> trafo;
    for (int k = 0; k < 3; ++k)
    {
      trafo(0, k) = newxaxis[k];
      trafo(1, k) = newyaxis[k];
      trafo(2, k) = auxn()[k];
    }

    // temporary storage for transformed points
    int np = (int)collconvexhull.size();
    Core::LinAlg::SerialDenseMatrix transformed(2, np);

    // transform each convex hull point
    for (int i = 0; i < np; ++i)
    {
      std::array<double, 3> newpoint = {0.0, 0.0, 0.0};

      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k)
          newpoint[j] += trafo(j, k) * (collconvexhull[i].coord()[k] - newzero[k]);

      if (abs(newpoint[2]) > tol)
        FOUR_C_THROW("ERROR: Transformation to aux. plane failed: z!=0 !");
      transformed(0, i) = newpoint[0];
      transformed(1, i) = newpoint[1];
    }

    //**********************************************************************
    // STEP8: Sort convex hull points to obtain final clip polygon
    // - this yields the final clip polygon
    // - sanity of the generated output is checked
    //**********************************************************************
    Mortar::sort_convex_hull_points(false, transformed, collconvexhull, respoly, tol);
  }

  return true;
}


/*----------------------------------------------------------------------*
 |  Triangulation of clip polygon (3D) - CENTER               popp 08/11|
 *----------------------------------------------------------------------*/
bool Coupling::VolMortar::VolMortarCoupl::center_triangulation(
    std::vector<std::shared_ptr<Mortar::IntCell>>& cells, std::vector<Mortar::Vertex>& clip,
    double tol)
{
  // preparations
  cells.resize(0);
  int clipsize = (int)(clip.size());
  std::vector<Core::Gen::Pairedvector<int, double>> lincenter(3, 100);

  std::vector<Core::Gen::Pairedvector<int, double>> derivauxn;

  std::vector<std::vector<Core::Gen::Pairedvector<int, double>>> linvertex(
      clipsize, std::vector<Core::Gen::Pairedvector<int, double>>(3, 100));

  //**********************************************************************
  // (1) Trivial clipping polygon -> IntCells
  //**********************************************************************
  // clip polygon = triangle
  // no triangulation necessary -> 1 IntCell
  if (clipsize == 3)
  {
    // IntCell vertices = clip polygon vertices
    Core::LinAlg::Matrix<3, 3> coords;
    for (int i = 0; i < clipsize; ++i)
      for (int k = 0; k < 3; ++k) coords(k, i) = clip[i].coord()[k];

    // create IntCell object and push back
    cells.push_back(std::make_shared<Mortar::IntCell>(0, 3, coords, auxn(),
        Core::FE::CellType::tri3, linvertex[0], linvertex[1], linvertex[2], derivauxn));

    // get out of here
    return true;
  }

  /*
   // clip polygon = quadrilateral
   // no triangulation necessary -> 2 IntCells
   else if (clipsize==4)
   {
   // IntCell 1 vertices = clip polygon vertices 0,1,2
   Core::LinAlg::SerialDenseMatrix coords(3,3);
   for (int k=0;k<3;++k)
   {
   coords(k,0) = Clip()[0].Coord()[k];
   coords(k,1) = Clip()[1].Coord()[k];
   coords(k,2) = Clip()[2].Coord()[k];
   }

   // create 1st IntCell object and push back
   Cells().push_back(Teuchos::rcp(new
   IntCell(0,3,coords,Auxn(),Core::FE::CellType::tri3,
   linvertex[0],linvertex[1],linvertex[2],get_deriv_auxn())));

   // IntCell vertices = clip polygon vertices 2,3,0
   for (int k=0;k<3;++k)
   {
   coords(k,0) = Clip()[2].Coord()[k];
   coords(k,1) = Clip()[3].Coord()[k];
   coords(k,2) = Clip()[0].Coord()[k];
   }

   // create 2nd IntCell object and push back
   Cells().push_back(Teuchos::rcp(new
   IntCell(1,3,coords,Auxn(),Core::FE::CellType::tri3,
   linvertex[2],linvertex[3],linvertex[0],get_deriv_auxn())));

   // get out of here
   return true;
   }
   */

  //**********************************************************************
  // (2) Find center of clipping polygon (centroid formula)
  //**********************************************************************
  std::vector<double> clipcenter(3);
  for (int k = 0; k < 3; ++k) clipcenter[k] = 0.0;
  double fac = 0.0;

  // first we need node averaged center
  std::array<double, 3> nac = {0.0, 0.0, 0.0};
  for (int i = 0; i < clipsize; ++i)
    for (int k = 0; k < 3; ++k) nac[k] += (clip[i].coord()[k] / clipsize);

  // loop over all triangles of polygon
  for (int i = 0; i < clipsize; ++i)
  {
    std::array<double, 3> xi_i = {0.0, 0.0, 0.0};
    std::array<double, 3> xi_ip1 = {0.0, 0.0, 0.0};

    // standard case
    if (i < clipsize - 1)
    {
      for (int k = 0; k < 3; ++k) xi_i[k] = clip[i].coord()[k];
      for (int k = 0; k < 3; ++k) xi_ip1[k] = clip[i + 1].coord()[k];
    }
    // last vertex of clip polygon
    else
    {
      for (int k = 0; k < 3; ++k) xi_i[k] = clip[clipsize - 1].coord()[k];
      for (int k = 0; k < 3; ++k) xi_ip1[k] = clip[0].coord()[k];
    }

    // triangle area
    std::array<double, 3> diff1 = {0.0, 0.0, 0.0};
    std::array<double, 3> diff2 = {0.0, 0.0, 0.0};
    for (int k = 0; k < 3; ++k) diff1[k] = xi_ip1[k] - xi_i[k];
    for (int k = 0; k < 3; ++k) diff2[k] = xi_i[k] - nac[k];

    std::array<double, 3> cross = {0.0, 0.0, 0.0};
    cross[0] = diff1[1] * diff2[2] - diff1[2] * diff2[1];
    cross[1] = diff1[2] * diff2[0] - diff1[0] * diff2[2];
    cross[2] = diff1[0] * diff2[1] - diff1[1] * diff2[0];

    double Atri = 0.5 * sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);

    // add contributions to clipcenter and fac
    fac += Atri;
    for (int k = 0; k < 3; ++k) clipcenter[k] += 1.0 / 3.0 * (xi_i[k] + xi_ip1[k] + nac[k]) * Atri;
  }

  // final clipcenter
  for (int k = 0; k < 3; ++k) clipcenter[k] /= fac;

  //**********************************************************************
  // (4) General clipping polygon: Triangulation -> IntCells
  //**********************************************************************
  // center-based triangulation if clip polygon > quadrilateral
  // No. of IntCells is equal to no. of clip polygon vertices
  for (int num = 0; num < clipsize; ++num)
  {
    // the first vertex is always the clip center
    // the second vertex is always the current clip vertex
    Core::LinAlg::Matrix<3, 3> coords;
    for (int k = 0; k < 3; ++k)
    {
      coords(k, 0) = clipcenter[k];
      coords(k, 1) = clip[num].coord()[k];
    }

    // the third vertex is the next vertex on clip polygon
    int numplus1 = num + 1;
    if (num == clipsize - 1)
    {
      for (int k = 0; k < 3; ++k) coords(k, 2) = clip[0].coord()[k];
      numplus1 = 0;
    }
    else
      for (int k = 0; k < 3; ++k) coords(k, 2) = clip[num + 1].coord()[k];

    // create IntCell object and push back
    cells.push_back(std::make_shared<Mortar::IntCell>(num, 3, coords, auxn(),
        Core::FE::CellType::tri3, lincenter, linvertex[num], linvertex[numplus1], derivauxn));
  }

  // triangulation successful
  return true;
}

/*----------------------------------------------------------------------*
 |  Triangulation of clip polygon (3D) - DELAUNAY             popp 08/11|
 *----------------------------------------------------------------------*/
bool Coupling::VolMortar::VolMortarCoupl::delaunay_triangulation(
    std::vector<std::shared_ptr<Mortar::IntCell>>& cells, std::vector<Mortar::Vertex>& clip,
    double tol)
{
  // preparations
  cells.resize(0);
  int clipsize = (int)(clip.size());

  std::vector<Core::Gen::Pairedvector<int, double>> derivauxn;

  std::vector<std::vector<Core::Gen::Pairedvector<int, double>>> linvertex(
      clipsize, std::vector<Core::Gen::Pairedvector<int, double>>(3, 100));
  //**********************************************************************
  // (1) Trivial clipping polygon -> IntCells
  //**********************************************************************
  // clip polygon = triangle
  // no triangulation necessary -> 1 IntCell
  if (clipsize == 3)
  {
    // IntCell vertices = clip polygon vertices
    Core::LinAlg::Matrix<3, 3> coords;
    for (int i = 0; i < clipsize; ++i)
      for (int k = 0; k < 3; ++k) coords(k, i) = clip[i].coord()[k];

    // create IntCell object and push back
    cells.push_back(std::make_shared<Mortar::IntCell>(0, 3, coords, auxn(),
        Core::FE::CellType::tri3, linvertex[0], linvertex[1], linvertex[2], derivauxn));

    // get out of here
    return true;
  }

  //**********************************************************************
  // (2) General clipping polygon: Triangulation -> IntCells
  //**********************************************************************
  // store Delaunay triangles here
  std::vector<std::vector<int>> triangles;

  // start with first triangle v0,v1,v2
  std::vector<int> currtriangle(3);
  currtriangle[0] = 0;
  currtriangle[1] = 1;
  currtriangle[2] = 2;
  triangles.push_back(currtriangle);

  // build Delaunay triangulation recursively (starting from a triangle
  // and then adding all remaining nodes of the clipping polygon 1-by-1)
  // loop over clip vertices v3,..,vN
  for (int c = 3; c < clipsize; ++c)
  {
    // current size of triangulated polygon
    int currsize = c + 1;

    // add next triangle v(c-1),vc,v0
    std::vector<int> nexttriangle(3);
    nexttriangle[0] = c - 1;
    nexttriangle[1] = c;
    nexttriangle[2] = 0;
    triangles.push_back(nexttriangle);

    // check Delaunay criterion for all triangles and sort
    // triangles accordingly (good / bad)
    int numt = (int)triangles.size();
    std::vector<bool> bad(numt);
    std::vector<double> close(numt);
    for (int t = 0; t < numt; ++t)
    {
      // initialize values indicating a close decision
      // these are needed to later introduce some tolerance into
      // the Delaunay criterion decision (which is needed in the
      // cases where this decision becomes non-unique / arbitrary).
      close[t] = 1.0e12;

      // indices of current triangle
      int idx0 = triangles[t][0];
      int idx1 = triangles[t][1];
      int idx2 = triangles[t][2];

      // coordinates of current triangle
      Core::LinAlg::SerialDenseMatrix coords(3, 3);
      for (int k = 0; k < 3; ++k)
      {
        coords(k, 0) = clip[idx0].coord()[k];
        coords(k, 1) = clip[idx1].coord()[k];
        coords(k, 2) = clip[idx2].coord()[k];
      }

      // define center of circumcircle of current triangle
      double x1 = coords(0, 0);
      double y1 = coords(1, 0);
      double z1 = coords(2, 0);
      double x2 = coords(0, 1);
      double y2 = coords(1, 1);
      double z2 = coords(2, 1);
      double x3 = coords(0, 2);
      double y3 = coords(1, 2);
      double z3 = coords(2, 2);

      // a=vector P1P2, b=vector P2P3
      double a1 = x2 - x1;
      double a2 = y2 - y1;
      double a3 = z2 - z1;
      double b1 = x3 - x2;
      double b2 = y3 - y2;
      double b3 = z3 - z2;

      // normal vector of plane P1P2P3 via cross product
      double no1 = a2 * b3 - b2 * a3;
      double no2 = a3 * b1 - b3 * a1;
      double no3 = a1 * b2 - b1 * a2;

      // perpendicular bisector of P1P2 via cross product
      double c1 = a2 * no3 - no2 * a3;
      double c2 = a3 * no1 - no3 * a1;
      double c3 = a1 * no2 - no1 * a2;

      // perpendicular bisector of P2P3 via cross product
      double d1 = b2 * no3 - no2 * b3;
      double d2 = b3 * no1 - no3 * b1;
      double d3 = b1 * no2 - no1 * b2;

      // mid-points of P1P2 and P2P3
      double m1 = (x1 + x2) / 2.0;
      double m2 = (y1 + y2) / 2.0;
      double m3 = (z1 + z2) / 2.0;
      double n1 = (x2 + x3) / 2.0;
      double n2 = (y2 + y3) / 2.0;
      double n3 = (z2 + z3) / 2.0;

      // try to minimize error
      int direction = 0;
      if (abs(auxn()[0]) >= abs(auxn()[1]) && abs(auxn()[0]) >= abs(auxn()[2])) direction = 1;
      if (abs(auxn()[1]) >= abs(auxn()[0]) && abs(auxn()[1]) >= abs(auxn()[2])) direction = 2;
      if (abs(auxn()[2]) >= abs(auxn()[0]) && abs(auxn()[2]) >= abs(auxn()[1])) direction = 3;
      if (direction == 0) FOUR_C_THROW("ERROR: Did not find best direction");

      // intersection of the two perpendicular bisections
      // (solution of m1+s*c1 = n1+t*d1 and m2+s*c2 = n2+t*d2)
      double s = 0.0;
      if (direction == 1)
      {
        // minimize error in yz-plane by solving
        // m2+s*c2 = n2+t*d2 and m3+s*c3 = n3+t*d3
        s = (m3 * d2 - n3 * d2 - d3 * m2 + d3 * n2) / (c2 * d3 - c3 * d2);
      }
      else if (direction == 2)
      {
        // minimize error in xz-plane by solving
        // m1+s*c1 = n1+t*d1 and m3+s*c3 = n3+t*d3
        s = (m3 * d1 - n3 * d1 - d3 * m1 + d3 * n1) / (c1 * d3 - c3 * d1);
      }
      else /* (direction==3)*/
      {
        // minimize error in xy-plane by solving
        // m1+s*c1 = n1+t*d1 and m2+s*c2 = n2+t*d2
        s = (m2 * d1 - n2 * d1 - d2 * m1 + d2 * n1) / (c1 * d2 - c2 * d1);
      }

      // center of the circumcircle
      double xcenter = m1 + s * c1;
      double ycenter = m2 + s * c2;
      double zcenter = m3 + s * c3;

      // radius of the circumcircle
      double radius1 = sqrt((xcenter - x1) * (xcenter - x1) + (ycenter - y1) * (ycenter - y1) +
                            (zcenter - z1) * (zcenter - z1));
      double radius2 = sqrt((xcenter - x2) * (xcenter - x2) + (ycenter - y2) * (ycenter - y2) +
                            (zcenter - z2) * (zcenter - z2));
      double radius3 = sqrt((xcenter - x3) * (xcenter - x3) + (ycenter - y3) * (ycenter - y3) +
                            (zcenter - z3) * (zcenter - z3));

      // check radius computation
      if (abs(radius2 - radius1) > tol || abs(radius3 - radius1) > tol)
      {
        std::cout << "***WARNING*** Delaunay triangulation failed (no well-defined circumcircles)"
                  << " -> using backup" << std::endl;

        // if Delaunay triangulation failed, use old center-based
        // triangulation as backup (therefore return false)
        return false;
      }

      // check Delaunay criterion for all other vertices
      // (of current polygon, NOT the full clipping polygon)
      for (int k = 0; k < currsize; ++k)
      {
        // no check needed for triangle vertices
        if (k == idx0 || k == idx1 || k == idx2) continue;

        // compute distance
        double dist = sqrt((xcenter - clip[k].coord()[0]) * (xcenter - clip[k].coord()[0]) +
                           (ycenter - clip[k].coord()[1]) * (ycenter - clip[k].coord()[1]) +
                           (zcenter - clip[k].coord()[2]) * (zcenter - clip[k].coord()[2]));

        // monitor critical Delaunay criterion decision
        // (necessary to avoid inconsistent good/bad grouping later)
        double diff = abs(dist - radius1);
        if (diff < close[t]) close[t] = diff;

        // check for bad triangle (without tolerance)
        if (dist < radius1) bad[t] = true;
      }
    }

    // make good/bad decision consistent (with tolerance)
    // (problems might occur if more than 3 vertices on circumcircle)
    for (int t = 0; t < numt; ++t)
    {
      // check if this good decision was really close
      if (!bad[t] && close[t] < tol)
      {
        // check if any bad decision was really close, too
        bool foundpartner = false;
        for (int u = 0; u < numt; ++u)
        {
          if (bad[u] && close[u] < tol) foundpartner = true;
        }

        // set good->bad if partner found
        if (foundpartner) bad[t] = true;
      }
    }

    // now we build vector of all good / bad triangles
    std::vector<std::vector<int>> goodtriangles;
    std::vector<std::vector<int>> badtriangles;
    for (int t = 0; t < numt; ++t)
    {
      if (bad[t])
        badtriangles.push_back(triangles[t]);
      else
        goodtriangles.push_back(triangles[t]);
    }

    // find vertices in bad triangles: ALL vertices
    // find vertices in bad triangles: NOT connected with current vertex
    std::vector<int> badv;
    std::vector<int> ncv;
    for (int t = 0; t < numt; ++t)
    {
      if (bad[t])
      {
        // indices of current bad triangle
        int idx0 = triangles[t][0];
        int idx1 = triangles[t][1];
        int idx2 = triangles[t][2];

        // collect ALL vertices
        bool foundbefore0 = false;
        for (int k = 0; k < (int)badv.size(); ++k)
        {
          if (badv[k] == idx0) foundbefore0 = true;
        }
        if (!foundbefore0) badv.push_back(idx0);

        bool foundbefore1 = false;
        for (int k = 0; k < (int)badv.size(); ++k)
        {
          if (badv[k] == idx1) foundbefore1 = true;
        }
        if (!foundbefore1) badv.push_back(idx1);

        bool foundbefore2 = false;
        for (int k = 0; k < (int)badv.size(); ++k)
        {
          if (badv[k] == idx2) foundbefore2 = true;
        }
        if (!foundbefore2) badv.push_back(idx2);

        // indices of current vertex neighbors
        int neighbor0 = c - 1;
        int neighbor1 = 0;

        // collect NOT connected vertices
        if (idx0 != c && idx0 != neighbor0 && idx0 != neighbor1)
        {
          bool foundbefore = false;
          for (int k = 0; k < (int)ncv.size(); ++k)
          {
            if (ncv[k] == idx0) foundbefore = true;
          }
          if (!foundbefore) ncv.push_back(idx0);
        }
        if (idx1 != c && idx1 != neighbor0 && idx1 != neighbor1)
        {
          bool foundbefore = false;
          for (int k = 0; k < (int)ncv.size(); ++k)
          {
            if (ncv[k] == idx1) foundbefore = true;
          }
          if (!foundbefore) ncv.push_back(idx1);
        }
        if (idx2 != c && idx2 != neighbor0 && idx2 != neighbor1)
        {
          bool foundbefore = false;
          for (int k = 0; k < (int)ncv.size(); ++k)
          {
            if (ncv[k] == idx2) foundbefore = true;
          }
          if (!foundbefore) ncv.push_back(idx2);
        }
      }
    }

    // build triangles formed by current vertex and ncv vertices
    std::vector<std::vector<int>> addtriangles;
    for (int k = 0; k < (int)ncv.size(); ++k)
    {
      // find ncv vertex neighbor0
      bool validneighbor0 = false;
      int off0 = 0;
      int neighbor0 = 0;
      do
      {
        // set neighbor
        neighbor0 = ncv[k] - 1 - off0;
        if ((ncv[k] - off0) == 0) neighbor0 = currsize - 1 - off0;

        // check if neighbor is in bad vertices
        for (int k = 0; k < (int)badv.size(); ++k)
        {
          if (badv[k] == neighbor0) validneighbor0 = true;
        }

        // increase counter
        ++off0;

      } while (!validneighbor0);

      // find ncv vertex neighbor1
      bool validneighbor1 = false;
      int off1 = 0;
      int neighbor1 = 0;
      do
      {
        // set neighbor
        neighbor1 = ncv[k] + 1 + off1;
        if ((ncv[k] + off1) == currsize - 1) neighbor1 = 0 + off1;

        // check if neighbor is in bad vertices
        for (int k = 0; k < (int)badv.size(); ++k)
        {
          if (badv[k] == neighbor1) validneighbor1 = true;
        }

        // increase counter
        ++off1;

      } while (!validneighbor1);

      // plausibility check
      if (neighbor0 == c || neighbor1 == c)
        FOUR_C_THROW("ERROR: Connected nodes not possible here");

      // add triangles
      std::vector<int> add1(3);
      add1[0] = c;
      add1[1] = ncv[k];
      add1[2] = neighbor0;
      addtriangles.push_back(add1);
      std::vector<int> add2(3);
      add2[0] = c;
      add2[1] = ncv[k];
      add2[2] = neighbor1;
      addtriangles.push_back(add2);
    }

    // collapse addtriangles (remove double entries)
    for (int k = 0; k < (int)addtriangles.size(); ++k)
    {
      bool addbefore = false;
      int idx0 = addtriangles[k][0];
      int idx1 = addtriangles[k][1];
      int idx2 = addtriangles[k][2];

      // check against all other goodtriangles
      for (int l = 0; l < (int)goodtriangles.size(); ++l)
      {
        // do not check against itself
        int lidx0 = goodtriangles[l][0];
        int lidx1 = goodtriangles[l][1];
        int lidx2 = goodtriangles[l][2];

        if (idx0 == lidx0 && idx1 == lidx1 && idx2 == lidx2) addbefore = true;
        if (idx0 == lidx0 && idx1 == lidx2 && idx2 == lidx1) addbefore = true;
        if (idx0 == lidx1 && idx1 == lidx0 && idx2 == lidx2) addbefore = true;
        if (idx0 == lidx1 && idx1 == lidx2 && idx2 == lidx0) addbefore = true;
        if (idx0 == lidx2 && idx1 == lidx0 && idx2 == lidx1) addbefore = true;
        if (idx0 == lidx2 && idx1 == lidx1 && idx2 == lidx0) addbefore = true;
      }

      // add to good triangles
      if (!addbefore)
      {
        goodtriangles.push_back(addtriangles[k]);
      }
    }

    // store final triangulation
    triangles.resize(0);
    for (int k = 0; k < (int)goodtriangles.size(); ++k) triangles.push_back(goodtriangles[k]);
  }

  // create intcells for all triangle
  int numt = (int)triangles.size();
  for (int t = 0; t < numt; ++t)
  {
    // indices of current triangle
    int idx0 = triangles[t][0];
    int idx1 = triangles[t][1];
    int idx2 = triangles[t][2];

    // coordinates of current triangle
    Core::LinAlg::Matrix<3, 3> coords;
    for (int k = 0; k < 3; ++k)
    {
      coords(k, 0) = clip[idx0].coord()[k];
      coords(k, 1) = clip[idx1].coord()[k];
      coords(k, 2) = clip[idx2].coord()[k];
    }

    // create IntCell object and push back
    cells.push_back(std::make_shared<Mortar::IntCell>(t, 3, coords, auxn(),
        Core::FE::CellType::tri3, linvertex[idx0], linvertex[idx1], linvertex[idx2], derivauxn));
  }

  // double check number of triangles
  if (numt != clipsize - 2)
  {
    std::cout << "***WARNING*** Delaunay triangulation failed (" << clipsize << " vertices, "
              << numt << " triangles)"
              << " -> using backup" << std::endl;

    // if Delaunay triangulation failed, use old center-based
    // triangulation as backup (therefore return false)
    return false;
  }

  // triangulation successful
  return true;
}

FOUR_C_NAMESPACE_CLOSE
