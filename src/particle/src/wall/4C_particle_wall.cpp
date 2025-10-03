// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_wall.hpp"

#include "4C_binstrategy.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_fem_condition_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_dofset_transparent.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_geometry_searchtree_service.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_discretization_visualization_writer_mesh.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_enums.hpp"
#include "4C_particle_engine_interface.hpp"
#include "4C_particle_input.hpp"
#include "4C_particle_wall_datastate.hpp"
#include "4C_particle_wall_discretization_runtime_vtu_writer.hpp"
#include "4C_solid_3D_ele.hpp"

#include <Teuchos_TimeMonitor.hpp>

#include <memory>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
PARTICLEWALL::WallHandlerBase::WallHandlerBase(MPI_Comm comm, const Teuchos::ParameterList& params)
    : comm_(comm),
      myrank_(Core::Communication::my_mpi_rank(comm)),
      params_(params),
      validwallelements_(false),
      validwallneighbors_(false)
{
  // empty constructor
}

PARTICLEWALL::WallHandlerBase::~WallHandlerBase() = default;

void PARTICLEWALL::WallHandlerBase::init(
    const std::shared_ptr<Core::Binstrategy::BinningStrategy> binstrategy)
{
  // set interface to binning strategy
  binstrategy_ = binstrategy;

  // init wall discretization
  init_wall_discretization();

  // init wall data state container
  init_wall_data_state();
}

void PARTICLEWALL::WallHandlerBase::setup(
    const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface,
    const double restart_time)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;

  // setup wall discretization
  setup_wall_discretization();

  // create wall discretization runtime vtu writer
  create_wall_discretization_runtime_vtu_writer(restart_time);

  // setup wall data state container
  walldatastate_->setup();
}

void PARTICLEWALL::WallHandlerBase::write_restart(const int step, const double time) const
{
  // get wall discretization writer
  std::shared_ptr<Core::IO::DiscretizationWriter> walldiscretizationwriter =
      walldiscretization_->writer();

  walldiscretizationwriter->new_step(step, time);
}

void PARTICLEWALL::WallHandlerBase::read_restart(const int restartstep) {}

void PARTICLEWALL::WallHandlerBase::insert_particle_states_of_particle_types(
    std::map<PARTICLEENGINE::TypeEnum, std::set<PARTICLEENGINE::StateEnum>>& particlestatestotypes)
    const
{
  // get flags defining considered states of particle wall
  const bool ismoving = params_.get<bool>("PARTICLE_WALL_MOVING");
  const bool isloaded = params_.get<bool>("PARTICLE_WALL_LOADED");

  if (not(ismoving and isloaded)) return;

  // iterate over particle types
  for (auto& typeIt : particlestatestotypes)
  {
    // set of particle states for current particle type
    std::set<PARTICLEENGINE::StateEnum>& particlestates = typeIt.second;

    // insert states needed for iteration in particle structure interaction
    particlestates.insert({
        PARTICLEENGINE::LastIterPosition,
        PARTICLEENGINE::LastIterVelocity,
        PARTICLEENGINE::LastIterAcceleration,
    });

    if (particlestates.count(PARTICLEENGINE::AngularVelocity))
      particlestates.insert(PARTICLEENGINE::LastIterAngularVelocity);

    if (particlestates.count(PARTICLEENGINE::AngularAcceleration))
      particlestates.insert(PARTICLEENGINE::LastIterAngularAcceleration);

    if (particlestates.count(PARTICLEENGINE::ModifiedAcceleration))
      particlestates.insert(PARTICLEENGINE::LastIterModifiedAcceleration);

    if (particlestates.count(PARTICLEENGINE::DensityDot))
      particlestates.insert(PARTICLEENGINE::LastIterDensity);

    if (particlestates.count(PARTICLEENGINE::TemperatureDot))
      particlestates.insert(PARTICLEENGINE::LastIterTemperature);
  }
}

void PARTICLEWALL::WallHandlerBase::write_wall_runtime_output(
    const int step, const double time) const
{
  // write wall discretization runtime output
  walldiscretizationruntimevtuwriter_->write_wall_discretization_runtime_output(step, time);
}

void PARTICLEWALL::WallHandlerBase::update_bin_row_and_col_map(
    const std::shared_ptr<Core::LinAlg::Map> binrowmap,
    const std::shared_ptr<Core::LinAlg::Map> bincolmap)
{
  binrowmap_ = binrowmap;
  bincolmap_ = bincolmap;
}

void PARTICLEWALL::WallHandlerBase::check_wall_nodes_located_in_bounding_box() const
{
  // get bounding box dimension
  Core::LinAlg::Matrix<3, 2> boundingbox = binstrategy_->domain_bounding_box_corner_positions();

  // iterate over row wall nodes
  for (int rowlidofnode = 0; rowlidofnode < walldiscretization_->num_my_row_nodes(); ++rowlidofnode)
  {
    // get pointer to current row wall node
    Core::Nodes::Node* node = walldiscretization_->l_row_node(rowlidofnode);

    // init current position of node
    Core::LinAlg::Matrix<3, 1> currpos;
    for (int dim = 0; dim < 3; ++dim) currpos(dim) = node->x()[dim];

    if (walldatastate_->get_disp_row() != nullptr)
    {
      // get nodal dofs
      std::vector<int> lm;
      walldiscretization_->dof(static_cast<unsigned int>(0), node, lm);

      // iterate over spatial directions
      for (int dim = 0; dim < 3; ++dim)
      {
        // local id of nodal dof in current spatial direction
        const int lid = walldiscretization_->dof_row_map()->lid(lm[dim]);

#ifdef FOUR_C_ENABLE_ASSERTIONS
        // safety check
        if (lid < 0) FOUR_C_THROW("dof gid={} not in dof row map!", lm[dim]);
#endif

        currpos(dim) += walldatastate_->get_disp_row()->operator[](lid);
      }
    }

    // safety check
    for (int dim = 0; dim < 3; ++dim)
      if (currpos(dim) < boundingbox(dim, 0) or boundingbox(dim, 1) < currpos(dim))
        FOUR_C_THROW("node gid={} resides outside of bounding box!", node->id());
  }
}

void PARTICLEWALL::WallHandlerBase::get_max_wall_position_increment(
    double& allprocmaxpositionincrement) const
{
  if (walldatastate_->get_disp_row() != nullptr)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    // safety checks
    if (walldatastate_->get_disp_row_last_transfer() == nullptr)
      FOUR_C_THROW("vector of wall displacements after last transfer not set!");

    if (not walldatastate_->get_disp_row()->get_map().same_as(
            walldatastate_->get_disp_row_last_transfer()->get_map()))
      FOUR_C_THROW("maps are not equal as expected!");
#endif

    // maximum position increment since last particle transfer
    double maxpositionincrement = 0.0;

    // iterate over coordinate values of wall displacements
    for (int i = 0; i < walldatastate_->get_disp_row()->local_length(); ++i)
    {
      // get position increment of wall node in current spatial dimension since last transfer
      double absolutpositionincrement =
          std::abs(walldatastate_->get_disp_row()->operator[](i) -
                   walldatastate_->get_disp_row_last_transfer()->operator[](i));

      // compare to current maximum
      maxpositionincrement = std::max(maxpositionincrement, absolutpositionincrement);
    }

    // bin size safety check
    if (maxpositionincrement > particleengineinterface_->min_bin_size())
      FOUR_C_THROW("a wall node traveled more than one bin on this processor!");

    // get maximum position increment on all processors
    allprocmaxpositionincrement =
        Core::Communication::max_all(maxpositionincrement, walldiscretization_->get_comm());
  }
}

void PARTICLEWALL::WallHandlerBase::relate_bins_to_col_wall_eles()
{
  // valid flag denoting validity of map relating bins to column wall elements
  if (validwallelements_) return;

  TEUCHOS_FUNC_TIME_MONITOR("PARTICLEWALL::WallHandlerBase::relate_bins_to_col_wall_eles");

#ifdef FOUR_C_ENABLE_ASSERTIONS
  walldatastate_->check_for_correct_maps();
#endif

  // clear vector relating column wall elements to bins
  binstocolwalleles_.assign(walldiscretization_->num_my_col_elements(), std::vector<int>(0));

  // invalidate flag denoting validity of wall neighbors map
  validwallneighbors_ = false;

  // iterate over column wall elements
  for (int collidofele = 0; collidofele < walldiscretization_->num_my_col_elements(); ++collidofele)
  {
    // get pointer to current column wall element
    Core::Elements::Element* ele = walldiscretization_->l_col_element(collidofele);

    // get corresponding bin ids for element
    std::vector<int> binids;
    binstrategy_->distribute_single_element_to_bins_using_ele_aabb(
        *walldiscretization_, ele, binids, walldatastate_->get_disp_col());

    // relate ids of owned bins to column wall elements
    for (int gidofbin : binids)
      if (not(bincolmap_->lid(gidofbin) < 0)) binstocolwalleles_[collidofele].push_back(gidofbin);
  }

  // validate flag denoting validity of map relating bins to column wall elements
  validwallelements_ = true;
}

void PARTICLEWALL::WallHandlerBase::build_particle_to_wall_neighbors(
    const PARTICLEENGINE::ParticlesToBins& particlestobins)
{
  TEUCHOS_FUNC_TIME_MONITOR("PARTICLEWALL::WallHandlerBase::build_particle_to_wall_neighbors");

#ifdef FOUR_C_ENABLE_ASSERTIONS
  walldatastate_->check_for_correct_maps();
#endif

  // safety check
  if ((not validwallelements_)) FOUR_C_THROW("invalid relation of bins to column wall elements!");

  // clear potential neighboring column wall elements
  potentialwallneighbors_.clear();

  // invalidate flag denoting validity of wall neighbors map
  validwallneighbors_ = false;

  // get particle container bundle
  PARTICLEENGINE::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->get_particle_container_bundle();

  // get minimum relevant bin size
  const double minbinsize = particleengineinterface_->min_bin_size();

  // iterate over column wall elements
  for (int collidofele = 0; collidofele < walldiscretization_->num_my_col_elements(); ++collidofele)
  {
    // set of neighboring bins of current column wall element
    std::set<int> neighborbins;

    // iterate over bins related to current column wall element
    for (int gidofbin : binstocolwalleles_[collidofele])
    {
      // get neighboring bins to current bin
      std::vector<int> binvec;
      binstrategy_->get_neighbor_and_own_bin_ids(gidofbin, binvec);

      // insert into set of neighboring bins of current column wall element
      neighborbins.insert(binvec.begin(), binvec.end());
    }

    // get pointer to current column wall element
    Core::Elements::Element* ele = walldiscretization_->l_col_element(collidofele);

    // determine nodal positions of column wall element
    std::map<int, Core::LinAlg::Matrix<3, 1>> colelenodalpos;
    determine_col_wall_ele_nodal_pos(ele, colelenodalpos);

    // iterate over neighboring bins
    for (int gidofneighborbin : neighborbins)
    {
      // consider only owned neighboring bins
      if (binrowmap_->lid(gidofneighborbin) < 0) continue;

      // get local id of neighboring bin
      const int collidofneighboringbin = bincolmap_->lid(gidofneighborbin);

      // check if current neighboring bin contains particles
      if (particlestobins[collidofneighboringbin].empty()) continue;

      // iterate over particles in current neighboring bin
      for (auto& neighborParticleIt : particlestobins[collidofneighboringbin])
      {
        // get type of neighboring particle
        PARTICLEENGINE::TypeEnum neighborTypeEnum = neighborParticleIt.first;

        // get local index of neighboring particle
        const int neighborindex = neighborParticleIt.second;

        // get container of neighboring particle of current particle type
        PARTICLEENGINE::ParticleContainer* neighborcontainer =
            particlecontainerbundle->get_specific_container(
                neighborTypeEnum, PARTICLEENGINE::Owned);

        // get position of neighboring particle
        const Core::LinAlg::Matrix<3, 1> currpos(
            neighborcontainer->get_ptr_to_state(PARTICLEENGINE::Position, neighborindex));

        // get coordinates of closest point on current column wall element to particle
        Core::LinAlg::Matrix<3, 1> closestpos;
        Core::Geo::nearest_3d_object_on_element(ele, colelenodalpos, currpos, closestpos);

        // distance vector from particle to closest point on current column wall element
        double dist[3];
        for (int i = 0; i < 3; i++) dist[i] = closestpos(i) - currpos(i);

        // distance between particle and wall element larger than minimum bin size
        if (dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2] > (minbinsize * minbinsize))
          continue;

        // append potential wall neighbor pair
        potentialwallneighbors_.push_back(std::make_pair(
            std::make_tuple(neighborTypeEnum, PARTICLEENGINE::Owned, neighborindex), ele));
      }
    }
  }

  // validate flag denoting validity of wall neighbors map
  validwallneighbors_ = true;
}

const PARTICLEENGINE::PotentialWallNeighbors&
PARTICLEWALL::WallHandlerBase::get_potential_wall_neighbors() const
{
  // safety check
  if (not validwallneighbors_) FOUR_C_THROW("invalid wall neighbors!");

  return potentialwallneighbors_;
}

void PARTICLEWALL::WallHandlerBase::determine_col_wall_ele_nodal_pos(
    Core::Elements::Element* ele, std::map<int, Core::LinAlg::Matrix<3, 1>>& colelenodalpos) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (walldiscretization_->element_col_map()->lid(ele->id()) < 0)
    FOUR_C_THROW("element gid={} not in element column map!", ele->id());
#endif

  // get pointer to nodes of current column wall element
  Core::Nodes::Node** nodes = ele->nodes();
  const int numnodes = ele->num_node();

#ifdef FOUR_C_ENABLE_ASSERTIONS
  for (int i = 0; i < numnodes; ++i)
    if (walldiscretization_->node_col_map()->lid(nodes[i]->id()) < 0)
      FOUR_C_THROW(
          "node gid={} of column element gid={} not in node column map", nodes[i]->id(), ele->id());
#endif

  // determine nodal displacements
  std::vector<double> nodal_disp(numnodes * 3, 0.0);
  if (walldatastate_->get_disp_col() != nullptr)
  {
    std::vector<int> lm_wall;
    lm_wall.reserve(numnodes * 3);
    std::vector<int> lmowner;
    std::vector<int> lmstride;
    ele->location_vector(*walldiscretization_, lm_wall, lmowner, lmstride);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    for (int i = 0; i < numnodes * 3; ++i)
      if (walldiscretization_->dof_col_map()->lid(lm_wall[i]) < 0)
        FOUR_C_THROW("dof gid={} not in dof column map!", lm_wall[i]);
#endif

    nodal_disp = Core::FE::extract_values(*walldatastate_->get_disp_col(), lm_wall);
  }

  // iterate over nodes of current column wall element
  for (int k = 0; k < numnodes; ++k)
  {
    // get reference to current nodal position
    Core::LinAlg::Matrix<3, 1>& currpos = colelenodalpos[nodes[k]->id()];

    // determine nodal position
    for (int dim = 0; dim < 3; ++dim) currpos(dim) = nodes[k]->x()[dim] + nodal_disp[k * 3 + dim];
  }
}

void PARTICLEWALL::WallHandlerBase::init_wall_data_state()
{
  // create wall data state container
  walldatastate_ = std::make_shared<PARTICLEWALL::WallDataState>(params_);

  // init wall data state container
  walldatastate_->init(walldiscretization_);
}

void PARTICLEWALL::WallHandlerBase::create_wall_discretization_runtime_vtu_writer(
    const double restart_time)
{
  // create wall discretization runtime vtu writer
  walldiscretizationruntimevtuwriter_ =
      std::make_unique<PARTICLEWALL::WallDiscretizationRuntimeVtuWriter>(
          walldiscretization_, walldatastate_, restart_time);
}

void PARTICLEWALL::WallHandlerBase::create_wall_discretization()
{
  // create wall discretization
  walldiscretization_ = std::make_shared<Core::FE::Discretization>(
      "particlewalls", comm_, Global::Problem::instance()->n_dim());

  // create wall discretization writer
  walldiscretization_->set_writer(std::make_shared<Core::IO::DiscretizationWriter>(
      walldiscretization_, Global::Problem::instance()->output_control_file(),
      Global::Problem::instance()->spatial_approximation_type()));
}

PARTICLEWALL::WallHandlerDiscretCondition::WallHandlerDiscretCondition(
    MPI_Comm comm, const Teuchos::ParameterList& params)
    : PARTICLEWALL::WallHandlerBase(comm, params)
{
  // empty constructor
}

void PARTICLEWALL::WallHandlerDiscretCondition::distribute_wall_elements_and_nodes()
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "PARTICLEWALL::WallHandlerDiscretCondition::distribute_wall_elements_and_nodes");

  // invalidate flags
  validwallelements_ = false;
  validwallneighbors_ = false;

  // distribute wall elements to bins with standard ghosting
  std::shared_ptr<Core::LinAlg::Map> stdelecolmap;
  std::shared_ptr<Core::LinAlg::Map> stdnodecolmapdummy;
  binstrategy_->standard_discretization_ghosting(walldiscretization_, *binrowmap_,
      walldatastate_->get_ref_disp_row(), stdelecolmap, stdnodecolmapdummy);

  // export displacement vector
  std::shared_ptr<Core::LinAlg::Vector<double>> disn_col = nullptr;
  if (walldatastate_->get_disp_row() != nullptr)
  {
    disn_col = std::make_shared<Core::LinAlg::Vector<double>>(*walldiscretization_->dof_col_map());
    Core::LinAlg::export_to(*walldatastate_->get_disp_row(), *disn_col);
  }

  // determine bin to row wall element distribution
  std::map<int, std::set<int>> bintorowelemap;
  binstrategy_->distribute_elements_to_bins_using_ele_aabb(
      *walldiscretization_, walldiscretization_->my_row_element_range(), bintorowelemap, disn_col);

  // extend wall element ghosting
  extend_wall_element_ghosting(bintorowelemap);

  // update maps of state vectors
  walldatastate_->update_maps_of_state_vectors();
}

void PARTICLEWALL::WallHandlerDiscretCondition::transfer_wall_elements_and_nodes()
{
  // transfer wall elements and nodes only if wall displacements are set
  if (walldatastate_->get_disp_row() == nullptr) return;

  TEUCHOS_FUNC_TIME_MONITOR(
      "PARTICLEWALL::WallHandlerDiscretCondition::transfer_wall_elements_and_nodes");

  // invalidate flags
  validwallelements_ = false;
  validwallneighbors_ = false;

  // transfer wall elements and nodes
  std::map<int, std::set<int>> bintorowelemap;
  binstrategy_->transfer_nodes_and_elements(
      *walldiscretization_, walldatastate_->get_disp_col(), bintorowelemap);

  // extend wall element ghosting
  extend_wall_element_ghosting(bintorowelemap);

  // update maps of state vectors
  walldatastate_->update_maps_of_state_vectors();
}

void PARTICLEWALL::WallHandlerDiscretCondition::extend_wall_element_ghosting(
    std::map<int, std::set<int>>& bintorowelemap)
{
  std::map<int, std::set<int>> colbintoelemap;
  std::shared_ptr<Core::LinAlg::Map> extendedelecolmap = binstrategy_->extend_element_col_map(
      bintorowelemap, bintorowelemap, colbintoelemap, bincolmap_);

  Core::Binstrategy::Utils::extend_discretization_ghosting(
      *walldiscretization_, *extendedelecolmap, true, false, false);
}

void PARTICLEWALL::WallHandlerDiscretCondition::init_wall_discretization()
{
  // create wall discretization
  create_wall_discretization();

  // access the structural discretization
  std::shared_ptr<Core::FE::Discretization> structurediscretization =
      Global::Problem::instance()->get_dis("structure");

  // finalize structure discretization construction
  if (not structurediscretization->filled()) structurediscretization->fill_complete();

  // get all particle wall conditions
  std::vector<const Core::Conditions::Condition*> conditions;
  structurediscretization->get_condition("ParticleWall", conditions);

  // iterate over particle wall conditions
  for (int i = 0; i < static_cast<int>(conditions.size()); ++i)
  {
    // set current particle wall condition
    std::vector<const Core::Conditions::Condition*> currcondition;
    currcondition.push_back(conditions[i]);

    // get material id for current particle wall condition
    const int mat = currcondition[0]->parameters().get<int>("MAT");

    // initialize maps for particle wall conditions
    std::map<int, Core::Nodes::Node*> nodes;
    std::map<int, Core::Nodes::Node*> colnodes;
    std::map<int, std::shared_ptr<Core::Elements::Element>> colelements;

    // get structure objects in wall condition
    Core::Conditions::find_condition_objects(
        *structurediscretization, nodes, colnodes, colelements, currcondition);

    // iterate over column wall nodes
    for (auto& nodeit : colnodes)
    {
      // get current node
      Core::Nodes::Node* currnode = nodeit.second;

      // add current node to wall discretization
      walldiscretization_->add_node(currnode->x(), currnode->id(),
          std::make_shared<Core::Nodes::Node>(currnode->id(), currnode->x(), currnode->owner()));
    }

    // iterate over column wall elements
    for (auto& eleit : colelements)
    {
      // get current element
      std::shared_ptr<Core::Elements::Element> currele = eleit.second;

      // create wall element
      std::shared_ptr<Core::Elements::Element> wallele =
          Core::Communication::factory("BELE3_3", "Polynomial", currele->id(), currele->owner());

      // set node ids to element
      wallele->set_node_ids(currele->num_node(), currele->node_ids());

      // create material for current wall element
      if (not(mat < 0)) wallele->set_material(0, Mat::factory(mat));

      // add wall element to discretization
      walldiscretization_->add_element(wallele);
    }
  }

  // reuse dofs of structural discretization for wall discretization
  bool parallel = (Core::Communication::num_mpi_ranks(comm_) == 1) ? false : true;
  std::shared_ptr<Core::DOFSets::DofSet> newdofset =
      std::make_shared<Core::DOFSets::TransparentDofSet>(structurediscretization, parallel);
  walldiscretization_->replace_dof_set(newdofset);

  // finalize wall discretization construction
  walldiscretization_->fill_complete({
      .assign_degrees_of_freedom = true,
      .init_elements = false,
      .do_boundary_conditions = false,
  });
}

void PARTICLEWALL::WallHandlerDiscretCondition::setup_wall_discretization() const
{
  // short screen output
  if (binstrategy_->have_periodic_boundary_conditions_applied() and myrank_ == 0)
    Core::IO::cout << "Warning: particle wall not transferred over periodic boundary!"
                   << Core::IO::endl;
}

PARTICLEWALL::WallHandlerBoundingBox::WallHandlerBoundingBox(
    MPI_Comm comm, const Teuchos::ParameterList& params)
    : PARTICLEWALL::WallHandlerBase(comm, params)
{
  // empty constructor
}

void PARTICLEWALL::WallHandlerBoundingBox::distribute_wall_elements_and_nodes()
{
  // no need to distribute wall elements and nodes
}

void PARTICLEWALL::WallHandlerBoundingBox::transfer_wall_elements_and_nodes()
{
  // no need to transfer wall elements and nodes
}

void PARTICLEWALL::WallHandlerBoundingBox::init_wall_discretization()
{
  // create wall discretization
  create_wall_discretization();

  // init vector of node and element ids
  std::vector<int> nodeids;
  std::vector<int> eleids;

  // generate wall discretization from bounding box on first processor
  if (myrank_ == 0)
  {
    // prepare vector of node and element ids
    nodeids.reserve(8);
    eleids.reserve(6);

    // get bounding box dimension
    Core::LinAlg::Matrix<3, 2> boundingbox = binstrategy_->domain_bounding_box_corner_positions();

    // reduce bounding box size to account for round-off errors
    for (int dim = 0; dim < 3; ++dim)
    {
      // periodic boundary conditions in current spatial direction
      if (binstrategy_->have_periodic_boundary_conditions_applied_in_spatial_direction(dim))
        continue;

      boundingbox(dim, 0) += 1.0e-12;
      boundingbox(dim, 1) -= 1.0e-12;
    }

    // init vector of corner node positions
    std::vector<std::array<double, 3>> nodepositions;
    nodepositions.reserve(8);

    // determine corner node positions from bounding box dimension
    nodepositions.push_back({boundingbox(0, 0), boundingbox(1, 0), boundingbox(2, 0)});
    nodepositions.push_back({boundingbox(0, 0), boundingbox(1, 1), boundingbox(2, 0)});
    nodepositions.push_back({boundingbox(0, 0), boundingbox(1, 1), boundingbox(2, 1)});
    nodepositions.push_back({boundingbox(0, 0), boundingbox(1, 0), boundingbox(2, 1)});
    nodepositions.push_back({boundingbox(0, 1), boundingbox(1, 0), boundingbox(2, 0)});
    nodepositions.push_back({boundingbox(0, 1), boundingbox(1, 1), boundingbox(2, 0)});
    nodepositions.push_back({boundingbox(0, 1), boundingbox(1, 1), boundingbox(2, 1)});
    nodepositions.push_back({boundingbox(0, 1), boundingbox(1, 0), boundingbox(2, 1)});

    int nodeid = 0;
    for (auto& nodepos : nodepositions)
    {
      // add corner node to wall discretization
      walldiscretization_->add_node(
          nodepos, nodeid, std::make_shared<Core::Nodes::Node>(nodeid, nodepos, myrank_));

      // add node id
      nodeids.push_back(nodeid++);
    }

    // init vector of node ids to corresponding wall elements
    std::vector<std::vector<int>> nodeidsofelements;
    nodeidsofelements.reserve(6);

    // set corner node ids for each wall element
    nodeidsofelements.push_back({0, 3, 2, 1});
    nodeidsofelements.push_back({4, 5, 6, 7});
    nodeidsofelements.push_back({0, 4, 7, 3});
    nodeidsofelements.push_back({1, 2, 6, 5});
    nodeidsofelements.push_back({0, 1, 5, 4});
    nodeidsofelements.push_back({2, 3, 7, 6});

    // get material id for particle wall
    const int mat = params_.get<int>("PARTICLE_WALL_MAT");

    int eleid = 0;
    for (int dim = 0; dim < 3; ++dim)
    {
      // periodic boundary conditions in current spatial direction
      if (binstrategy_->have_periodic_boundary_conditions_applied_in_spatial_direction(dim))
        continue;

      // positive and negative end of bounding box in current spatial direction
      for (int sign = 0; sign < 2; ++sign)
      {
        // create wall element
        std::shared_ptr<Core::Elements::Element> wallele =
            Core::Communication::factory("BELE3_3", "Polynomial", eleid, myrank_);

        // set node ids to element
        wallele->set_node_ids(4, nodeidsofelements[dim * 2 + sign].data());

        // create material for current wall element
        if (not(mat < 0)) wallele->set_material(0, Mat::factory(mat));

        // add wall element to discretization
        walldiscretization_->add_element(wallele);

        // add element id
        eleids.push_back(eleid++);
      }
    }

    // no wall elements added
    if (eleids.empty()) FOUR_C_THROW("no wall elements added, check periodic boundary conditions!");
  }

  // node row map of wall elements
  Core::LinAlg::Map noderowmap(
      -1, nodeids.size(), nodeids.data(), 0, walldiscretization_->get_comm());

  // fully overlapping node column map
  std::shared_ptr<Core::LinAlg::Map> nodecolmap = Core::LinAlg::allreduce_e_map(noderowmap);

  // element row map of wall elements
  Core::LinAlg::Map elerowmap(-1, eleids.size(), eleids.data(), 0, walldiscretization_->get_comm());

  // fully overlapping element column map
  std::shared_ptr<Core::LinAlg::Map> elecolmap = Core::LinAlg::allreduce_e_map(elerowmap);

  // fully overlapping ghosting of the wall elements to have everything redundant
  walldiscretization_->export_column_nodes(*nodecolmap);
  walldiscretization_->export_column_elements(*elecolmap);

  // finalize wall discretization construction
  walldiscretization_->fill_complete({
      .assign_degrees_of_freedom = true,
      .init_elements = false,
      .do_boundary_conditions = false,
  });
}

void PARTICLEWALL::WallHandlerBoundingBox::setup_wall_discretization() const
{
  // nothing to do
}

FOUR_C_NAMESPACE_CLOSE
