// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_utils.hpp"

#include "4C_adapter_porofluid_pressure_based.hpp"
#include "4C_fem_general_utils_createdis.hpp"
#include "4C_fem_geometry_intersection_service.hpp"
#include "4C_fem_geometry_intersection_service.templates.hpp"
#include "4C_fem_geometry_position_array.hpp"
#include "4C_fem_geometry_searchtree.hpp"
#include "4C_fem_geometry_searchtree_service.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_bio.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_mat_cnst_1d_art.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_material_base.hpp"
#include "4C_porofluid_pressure_based_algorithm.hpp"
#include "4C_rebalance_binning_based.hpp"
#include "4C_rebalance_print.hpp"

FOUR_C_NAMESPACE_OPEN


namespace
{
  std::vector<int> get_coupling_arteries_node_to_point(
      std::shared_ptr<Core::FE::Discretization> artdis, Core::FE::Discretization& artsearchdis)
  {
    // this vector will be filled
    std::vector<int> artEleGIDs_help;

    // get 1D coupling IDs from Input
    std::vector<const Core::Conditions::Condition*> artCoupcond;

    artdis->get_condition("ArtPorofluidCouplConNodeToPoint", artCoupcond);

    artEleGIDs_help.reserve(artCoupcond.size());

    // get global element Ids from artery coupling nodes
    for (const auto& iter : artCoupcond)
    {
      const std::vector<int>* ArteryNodeIds = iter->get_nodes();

      for (auto const nodeid : *ArteryNodeIds)
      {
        Core::Nodes::Node* artnode = artsearchdis.g_node(nodeid);
        Core::Elements::Element** artele = artnode->elements();
        // get Id of corresponding element; Note: in lung modeling only most distal nodes
        // are coupled, so coupling nodes can only belong to one element
        const int elementID = artele[0]->id();
        // safety check if assertion is true
        FOUR_C_ASSERT(elementID >= 0, "It is not possible to have a negative element ID!");
        artEleGIDs_help.push_back(elementID);
      }
    }
    return artEleGIDs_help;
  }
}  // namespace

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void PoroPressureBased::setup_material(
    MPI_Comm comm, const std::string& struct_disname, const std::string& fluid_disname)
{
  // get the fluid discretization
  std::shared_ptr<Core::FE::Discretization> fluiddis =
      Global::Problem::instance()->get_dis(fluid_disname);

  // initialize material map
  std::map<int, int> matmap;
  {
    // get the cloning material map from the input file
    std::map<std::pair<std::string, std::string>, std::map<int, int>> clonefieldmatmap =
        Global::Problem::instance()->cloning_material_map();
    if (clonefieldmatmap.size() < 1)
      FOUR_C_THROW("At least one material pairing required in --CLONING MATERIAL MAP.");

    // check if the current discretization is included in the material map
    std::pair<std::string, std::string> key(fluid_disname, struct_disname);
    matmap = clonefieldmatmap[key];
    if (matmap.size() < 1)
      FOUR_C_THROW(
          "Key pair '{}/{}' not defined in --CLONING MATERIAL MAP.", fluid_disname, struct_disname);
  }


  // number of column elements within fluid discretization
  const int numelements = fluiddis->num_my_col_elements();

  // loop over column elements
  for (int i = 0; i < numelements; ++i)
  {
    // get current element
    Core::Elements::Element* ele = fluiddis->l_col_element(i);

    // find the corresponding material in the matmap
    int src_matid = ele->material()->parameter()->id();
    std::map<int, int>::iterator mat_iter = matmap.find(src_matid);
    if (mat_iter != matmap.end())
    {
      // get the ID of the secondary material
      const int tar_matid = mat_iter->second;
      // build the material usilng the factory
      std::shared_ptr<Core::Mat::Material> mat = Mat::factory(tar_matid);

      // add secondary material to poro fluid element
      if (ele->add_material(mat) != 2) FOUR_C_THROW("unexpected number of materials!");
    }
    else
    {
      // before we stop, print the material id map
      std::cout << "Material map on PROC " << Core::Communication::my_mpi_rank(comm) << ":"
                << std::endl;
      for (mat_iter = matmap.begin(); mat_iter != matmap.end(); mat_iter++)
        std::cout << mat_iter->first << " -> " << mat_iter->second << std::endl;

      FOUR_C_THROW("no matching material ID ({}) in map", src_matid);
    }

  }  // end loop over column elements

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::MultiVector<double>>
PoroPressureBased::convert_dof_vector_to_node_based_multi_vector(
    const Core::FE::Discretization& dis, const Core::LinAlg::Vector<double>& vector, const int nds,
    const int numdofpernode)
{
  // initialize multi vector
  std::shared_ptr<Core::LinAlg::MultiVector<double>> multi =
      std::make_shared<Core::LinAlg::MultiVector<double>>(*dis.node_row_map(), numdofpernode, true);

  // get maps
  const Core::LinAlg::Map& vectormap = vector.get_map();

  // loop over nodes of the discretization
  for (int inode = 0; inode < dis.num_my_row_nodes(); ++inode)
  {
    // get current node
    Core::Nodes::Node* node = dis.l_row_node(inode);
    // copy each dof value of node
    for (int idof = 0; idof < numdofpernode; ++idof)
      (*multi)(idof)[inode] = vector[vectormap.lid(dis.dof(nds, node, idof))];
  }

  return multi;
}

/*----------------------------------------------------------------------*
 | create algorithm                                                      |
 *----------------------------------------------------------------------*/
std::shared_ptr<Adapter::PoroFluidMultiphase> PoroPressureBased::create_algorithm(
    PoroPressureBased::TimeIntegrationScheme timintscheme,
    std::shared_ptr<Core::FE::Discretization> dis, const int linsolvernumber,
    const Teuchos::ParameterList& probparams, const Teuchos::ParameterList& poroparams,
    std::shared_ptr<Core::IO::DiscretizationWriter> output)
{
  // Creation of Coupled Problem algorithm.
  std::shared_ptr<Adapter::PoroFluidMultiphase> algo = nullptr;

  // -------------------------------------------------------------------
  // algorithm construction depending on
  // time-integration (or stationary) scheme
  // -------------------------------------------------------------------

  switch (timintscheme)
  {
    case TimeIntegrationScheme::one_step_theta:
    {
      // create algorithm
      algo = std::make_shared<PorofluidAlgorithm>(
          dis, linsolvernumber, probparams, poroparams, output);
      break;
    }
    default:
      FOUR_C_THROW("Unknown time-integration scheme for multiphase poro fluid problem");
      break;
  }

  return algo;
}

/*--------------------------------------------------------------------------*
 | perform extended ghosting for artery dis                kremheller 03/19 |
 *--------------------------------------------------------------------------*/
std::map<int, std::set<int>> PoroPressureBased::extended_ghosting_artery_discretization(
    Core::FE::Discretization& contdis, std::shared_ptr<Core::FE::Discretization> artdis,
    const bool evaluate_on_lateral_surface,
    const Inpar::ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod couplingmethod)
{
  // user output
  if (Core::Communication::my_mpi_rank(contdis.get_comm()) == 0)
  {
    std::cout
        << "\n<<<<<<<<<<<<<<< Starting extended ghosting of artery discretization >>>>>>>>>>>>>>>\n"
        << std::endl;
  }

  artdis->fill_complete();
  if (!contdis.filled()) contdis.fill_complete();

  // create the fully overlapping search discretization
  std::shared_ptr<Core::FE::Discretization> artsearchdis =
      create_fully_overlapping_artery_discretization(*artdis, "artsearchdis", false);

  // to be filled with additional elements to be ghosted
  std::set<int> elecolset;
  const Core::LinAlg::Map* elecolmap = artdis->element_col_map();
  for (int lid = 0; lid < elecolmap->num_my_elements(); ++lid)
  {
    int gid = elecolmap->gid(lid);
    elecolset.insert(gid);
  }

  // to be filled with additional nodes to be ghosted
  std::set<int> nodecolset;
  const Core::LinAlg::Map* nodecolmap = artdis->node_col_map();
  for (int lid = 0; lid < nodecolmap->num_my_elements(); ++lid)
  {
    int gid = nodecolmap->gid(lid);
    nodecolset.insert(gid);
  }

  // get artEleGIDs depending on the coupling method
  const std::vector<int> artEleGIDs = std::invoke(
      [&]()
      {
        if (couplingmethod == Inpar::ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::ntp)
        {
          return get_coupling_arteries_node_to_point(artdis, *artsearchdis);
        }
        else
        {
          std::vector<int> artEleGIDs_help;
          artEleGIDs_help.reserve(artsearchdis->element_col_map()->num_my_elements());
          for (int iart = 0; iart < artsearchdis->element_col_map()->num_my_elements(); ++iart)
          {
            artEleGIDs_help.push_back(artsearchdis->element_col_map()->gid(iart));
          }
          return artEleGIDs_help;
        }
      });

  // search with the fully overlapping discretization
  std::map<int, std::set<int>> nearby_ele_pairs = oct_tree_search(contdis, *artdis, *artsearchdis,
      evaluate_on_lateral_surface, artEleGIDs, elecolset, nodecolset);

  // extended ghosting for elements
  std::vector<int> coleles(elecolset.begin(), elecolset.end());
  const Core::LinAlg::Map extendedelecolmap(
      -1, coleles.size(), coleles.data(), 0, contdis.get_comm());

  artdis->export_column_elements(extendedelecolmap);

  // extended ghosting for nodes
  std::vector<int> colnodes(nodecolset.begin(), nodecolset.end());
  const Core::LinAlg::Map extendednodecolmap(
      -1, colnodes.size(), colnodes.data(), 0, contdis.get_comm());

  artdis->export_column_nodes(extendednodecolmap);

  // fill and inform user
  artdis->fill_complete();
  Core::Rebalance::Utils::print_parallel_distribution(*artdis);

  // user output
  if (Core::Communication::my_mpi_rank(contdis.get_comm()) == 0)
  {
    std::cout << "<<<<<<<<<<<<<<< Finished extended ghosting of artery discretization "
                 ">>>>>>>>>>>>>>>\n"
              << std::endl;
  }

  return nearby_ele_pairs;
}

/*--------------------------------------------------------------------------*
 | create the fully overlapping artery discretization      kremheller 03/19 |
 *--------------------------------------------------------------------------*/
std::shared_ptr<Core::FE::Discretization>
PoroPressureBased::create_fully_overlapping_artery_discretization(
    Core::FE::Discretization& artdis, std::string disname, bool doboundaryconditions)
{
  // we clone a search discretization of the artery discretization on which the search will be
  // performed in a brute force way fully overlapping
  Core::FE::DiscretizationCreatorBase discloner;
  std::shared_ptr<Core::FE::Discretization> artsearchdis =
      discloner.create_matching_discretization(artdis, disname, false, false, false, false);

  // ghost on all procs.
  Core::Rebalance::ghost_discretization_on_all_procs(*artsearchdis);
  artsearchdis->fill_complete(false, false, doboundaryconditions);

  return artsearchdis;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::map<int, std::set<int>> PoroPressureBased::oct_tree_search(Core::FE::Discretization& contdis,
    Core::FE::Discretization& artdis, Core::FE::Discretization& artsearchdis,
    const bool evaluate_on_lateral_surface, const std::vector<int> artEleGIDs,
    std::set<int>& elecolset, std::set<int>& nodecolset)
{
  // this map will be filled
  std::map<int, std::set<int>> nearby_ele_pairs;

  // search tree
  Core::Geo::SearchTree searchTree(5);

  // nodal positions of 2D/3D-discretization
  std::map<int, Core::LinAlg::Matrix<3, 1>> my_positions_cont =
      get_nodal_positions(contdis, contdis.node_col_map());
  // axis-aligned bounding boxes of all elements of 2D/3D discretization
  std::map<int, Core::LinAlg::Matrix<3, 2>> aabb_cont =
      Core::Geo::get_current_xaab_bs(contdis, my_positions_cont);

  // find the bounding box of the 2D/3D discretization
  const Core::LinAlg::Matrix<3, 2> sourceEleBox = Core::Geo::get_xaab_bof_dis(contdis);
  searchTree.initialize_tree(sourceEleBox, contdis, Core::Geo::TreeType(Core::Geo::OCTTREE));

  // user info and timer
  if (Core::Communication::my_mpi_rank(contdis.get_comm()) == 0)
    std::cout << "Starting with OctTree search for coupling ... " << std::endl;
  Teuchos::Time timersearch("OctTree_search", true);
  // *********** time measurement ***********
  double dtcpu = timersearch.wallTime();
  // *********** time measurement ***********

  // nodal positions of artery-discretization (fully overlapping)
  std::map<int, Core::LinAlg::Matrix<3, 1>> positions_artery;
  // nodal positions of artery-discretization (row-map format)
  std::map<int, Core::LinAlg::Matrix<3, 1>> my_positions_artery =
      get_nodal_positions(artdis, artdis.node_row_map());

  // gather
  std::vector<int> procs(Core::Communication::num_mpi_ranks(contdis.get_comm()));
  for (int i = 0; i < Core::Communication::num_mpi_ranks(contdis.get_comm()); i++) procs[i] = i;
  Core::LinAlg::gather<int, Core::LinAlg::Matrix<3, 1>>(my_positions_artery, positions_artery,
      Core::Communication::num_mpi_ranks(contdis.get_comm()), procs.data(), contdis.get_comm());

  // do the actual search on fully overlapping artery discretization
  for (unsigned int iart = 0; iart < artEleGIDs.size(); ++iart)
  {
    const int artelegid = artEleGIDs[iart];
    Core::Elements::Element* artele = artsearchdis.g_element(artelegid);

    // axis-aligned bounding box of artery
    const Core::LinAlg::Matrix<3, 2> aabb_artery =
        get_aabb(artele, positions_artery, evaluate_on_lateral_surface);

    // get elements nearby
    std::set<int> closeeles;
    searchTree.search_collisions(aabb_cont, aabb_artery, 0, closeeles);

    // nearby elements found
    if (closeeles.size() > 0)
    {
      nearby_ele_pairs[artelegid] = closeeles;

      // add elements and nodes for extended ghosting of artery discretization
      if (not artdis.have_global_element(artelegid))
      {
        elecolset.insert(artelegid);
        const int* nodeids = artele->node_ids();
        for (int inode = 0; inode < artele->num_node(); ++inode) nodecolset.insert(nodeids[inode]);
      }
    }

    // estimate of duration for search (check how long the search took for 1/20 of all elements, the
    // estimated total time of the search is then 20 times this time)
    if (iart == (0.05 * artEleGIDs.size()))
    {
      double mydtsearch = timersearch.wallTime() - dtcpu;
      double maxdtsearch = 0.0;
      Core::Communication::max_all(&mydtsearch, &maxdtsearch, 1, contdis.get_comm());
      if (Core::Communication::my_mpi_rank(contdis.get_comm()) == 0)
        std::cout << "Estimated duration: " << 20.0 * (maxdtsearch) << "s" << std::endl;
    }
  }

  // *********** time measurement ***********
  double mydtsearch = timersearch.wallTime() - dtcpu;
  double maxdtsearch = 0.0;
  Core::Communication::max_all(&mydtsearch, &maxdtsearch, 1, contdis.get_comm());
  // *********** time measurement ***********
  if (Core::Communication::my_mpi_rank(contdis.get_comm()) == 0)
    std::cout << "Completed in " << maxdtsearch << "s" << std::endl;

  return nearby_ele_pairs;
}

/*----------------------------------------------------------------------*
 | get axis-aligned bounding box of element            kremheller 03/19 |
 *----------------------------------------------------------------------*/
Core::LinAlg::Matrix<3, 2> PoroPressureBased::get_aabb(Core::Elements::Element* ele,
    std::map<int, Core::LinAlg::Matrix<3, 1>>& positions, const bool evaluate_on_lateral_surface)
{
  const Core::LinAlg::SerialDenseMatrix xyze_element(
      Core::Geo::get_current_nodal_positions(ele, positions));
  Core::Geo::EleGeoType eleGeoType(Core::Geo::HIGHERORDER);
  Core::Geo::check_rough_geo_type(ele, xyze_element, eleGeoType);

  Core::LinAlg::Matrix<3, 2> aabb_artery =
      Core::Geo::compute_fast_xaabb(ele->shape(), xyze_element, eleGeoType);

  // add radius to axis aligned bounding box of artery element (in all coordinate directions) in
  // case of evaluation on lateral surface
  if (evaluate_on_lateral_surface)
  {
    std::shared_ptr<Mat::Cnst1dArt> arterymat =
        std::static_pointer_cast<Mat::Cnst1dArt>(ele->material());
    if (arterymat == nullptr) FOUR_C_THROW("Cast to artery material failed!");
    const double radius = arterymat->diam() / 2.0;
    for (int idim = 0; idim < 3; idim++)
    {
      aabb_artery(idim, 0) = aabb_artery(idim, 0) - radius;
      aabb_artery(idim, 1) = aabb_artery(idim, 1) + radius;
    }
  }
  return aabb_artery;
}

/*----------------------------------------------------------------------*
 | get nodal positions                                 kremheller 10/19 |
 *----------------------------------------------------------------------*/
std::map<int, Core::LinAlg::Matrix<3, 1>> PoroPressureBased::get_nodal_positions(
    Core::FE::Discretization& dis, const Core::LinAlg::Map* nodemap)
{
  std::map<int, Core::LinAlg::Matrix<3, 1>> positions;
  for (int lid = 0; lid < nodemap->num_my_elements(); ++lid)
  {
    const Core::Nodes::Node* node = dis.g_node(nodemap->gid(lid));
    Core::LinAlg::Matrix<3, 1> currpos;

    currpos(0) = node->x()[0];
    currpos(1) = node->x()[1];
    currpos(2) = node->x()[2];

    positions[node->id()] = currpos;
  }
  return positions;
}

/*----------------------------------------------------------------------*
 | get maximum nodal distance                          kremheller 05/18 |
 *----------------------------------------------------------------------*/
double PoroPressureBased::get_max_nodal_distance(
    Core::Elements::Element* ele, Core::FE::Discretization& dis)
{
  double maxdist = 0.0;

  for (int inode = 0; inode < ele->num_node() - 1; inode++)
  {
    // get first node and its position
    int node0_gid = ele->node_ids()[inode];
    Core::Nodes::Node* node0 = dis.g_node(node0_gid);

    static Core::LinAlg::Matrix<3, 1> pos0;
    pos0(0) = node0->x()[0];
    pos0(1) = node0->x()[1];
    pos0(2) = node0->x()[2];

    // loop over second node to numnode to compare distances with first node
    for (int jnode = inode + 1; jnode < ele->num_node(); jnode++)
    {
      int node1_gid = ele->node_ids()[jnode];
      Core::Nodes::Node* node1 = dis.g_node(node1_gid);

      static Core::LinAlg::Matrix<3, 1> pos1;
      pos1(0) = node1->x()[0];
      pos1(1) = node1->x()[1];
      pos1(2) = node1->x()[2];

      static Core::LinAlg::Matrix<3, 1> dist;
      dist.update(1.0, pos0, -1.0, pos1, 0.0);

      maxdist = std::max(maxdist, dist.norm2());
    }
  }

  return maxdist;
}

/*----------------------------------------------------------------------*
 | calculate vector norm                             kremheller 12/17   |
 *----------------------------------------------------------------------*/
double PoroPressureBased::calculate_vector_norm(
    const enum PoroPressureBased::VectorNorm norm, const Core::LinAlg::Vector<double>& vect)
{
  // L1 norm
  // norm = sum_0^i vect[i]
  if (norm == VectorNorm::l1)
  {
    double vectnorm;
    vect.norm_1(&vectnorm);
    return vectnorm;
  }
  // L2/Euclidian norm
  // norm = sqrt{sum_0^i vect[i]^2 }
  else if (norm == VectorNorm::l2)
  {
    double vectnorm;
    vect.norm_2(&vectnorm);
    return vectnorm;
  }
  // RMS norm
  // norm = sqrt{sum_0^i vect[i]^2 }/ sqrt{length_vect}
  else if (norm == VectorNorm::rms)
  {
    double vectnorm;
    vect.norm_2(&vectnorm);
    return vectnorm / sqrt((double)vect.global_length());
  }
  // infinity/maximum norm
  // norm = max( vect[i] )
  else if (norm == VectorNorm::inf)
  {
    double vectnorm;
    vect.norm_inf(&vectnorm);
    return vectnorm;
  }
  // norm = sum_0^i vect[i]/length_vect
  else if (norm == VectorNorm::l1_scaled)
  {
    double vectnorm;
    vect.norm_1(&vectnorm);
    return vectnorm / ((double)vect.global_length());
  }
  else
  {
    FOUR_C_THROW("Cannot handle vector norm");
    return 0;
  }  // calculate_vector_norm()
}

FOUR_C_NAMESPACE_CLOSE
