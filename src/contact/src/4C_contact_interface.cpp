// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_interface.hpp"

#include "4C_binstrategy.hpp"
#include "4C_contact_coupling2d.hpp"
#include "4C_contact_coupling3d.hpp"
#include "4C_contact_element.hpp"
#include "4C_contact_friction_node.hpp"
#include "4C_contact_input.hpp"
#include "4C_contact_integrator.hpp"
#include "4C_contact_interpolator.hpp"
#include "4C_contact_line_coupling.hpp"
#include "4C_contact_nitsche_utils.hpp"
#include "4C_contact_node.hpp"
#include "4C_contact_selfcontact_binarytree_unbiased.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_linalg_graph.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_mortar_binarytree.hpp"
#include "4C_mortar_defines.hpp"
#include "4C_mortar_dofset.hpp"
#include "4C_mortar_projector.hpp"
#include "4C_rebalance_graph_based.hpp"
#include "4C_scatra_ele_parameter_boundary.hpp"

#include <Teuchos_Time.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
CONTACT::InterfaceDataContainer::InterfaceDataContainer()
    : selfcontact_(false),
      friction_(false),
      non_smooth_contact_(false),
      two_half_pass_(false),
      constr_direction_(CONTACT::ConstraintDirection::vague),
      activenodes_(nullptr),
      activedofs_(nullptr),
      inactivenodes_(nullptr),
      inactivedofs_(nullptr),
      activen_(nullptr),
      activet_(nullptr),
      slipnodes_(nullptr),
      slipdofs_(nullptr),
      slipt_(nullptr),
      nonsmoothnodes_(nullptr),
      smoothnodes_(nullptr),
      sdof_vertex_rowmap_(nullptr),
      sdof_vertex_colmap_(nullptr),
      sdof_edge_rowmap_(nullptr),
      sdof_edge_colmap_(nullptr),
      sdof_surf_rowmap_(nullptr),
      sdof_surf_colmap_(nullptr),
      nextendedghosting_(nullptr),
      eextendedghosting_(nullptr),
      binarytreeself_(nullptr),
      cn_values_(nullptr),
      ct_values_(nullptr),
      smpairs_(0),
      smintpairs_(0),
      intcells_(0)
{
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<CONTACT::Interface> CONTACT::Interface::create(const int id, MPI_Comm comm,
    const int spatialDim, const Teuchos::ParameterList& icontact, const bool selfcontact)
{
  std::shared_ptr<Mortar::InterfaceDataContainer> interfaceData_ptr =
      std::make_shared<CONTACT::InterfaceDataContainer>();
  return std::make_shared<Interface>(
      interfaceData_ptr, id, comm, spatialDim, icontact, selfcontact);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
CONTACT::Interface::Interface(const std::shared_ptr<CONTACT::InterfaceDataContainer>& interfaceData)
    : Mortar::Interface(interfaceData),
      interface_data_(interfaceData),
      selfcontact_(interface_data_->is_self_contact()),
      friction_(interface_data_->is_friction()),
      nonSmoothContact_(interface_data_->is_non_smooth_contact()),
      two_half_pass_(interface_data_->is_two_half_pass()),
      constr_direction_(interface_data_->constraint_direction()),
      activenodes_(interface_data_->active_nodes()),
      activedofs_(interface_data_->active_dofs()),
      inactivenodes_(interface_data_->inactive_nodes()),
      inactivedofs_(interface_data_->inactive_dofs()),
      activen_(interface_data_->active_n()),
      activet_(interface_data_->active_t()),
      slipnodes_(interface_data_->slip_nodes()),
      slipdofs_(interface_data_->slip_dofs()),
      slipt_(interface_data_->slip_t()),
      nonsmoothnodes_(interface_data_->non_smooth_nodes()),
      smoothnodes_(interface_data_->smooth_nodes()),
      sdofVertexRowmap_(interface_data_->sdof_vertex_rowmap()),
      sdofVertexColmap_(interface_data_->sdof_vertex_colmap()),
      sdofEdgeRowmap_(interface_data_->sdof_edge_rowmap()),
      sdofEdgeColmap_(interface_data_->sdof_edge_colmap()),
      sdofSurfRowmap_(interface_data_->sdof_surf_rowmap()),
      sdofSurfColmap_(interface_data_->sdof_surf_colmap()),
      nextendedghosting_(interface_data_->n_extended_ghosting()),
      eextendedghosting_(interface_data_->e_extended_ghosting()),
      binarytreeself_(interface_data_->binary_tree_self()),
      cnValues_(interface_data_->cn_values()),
      ctValues_(interface_data_->ct_values()),
      smpairs_(interface_data_->sm_int_pairs()),
      smintpairs_(interface_data_->sm_int_pairs()),
      intcells_(interface_data_->int_cells())
{ /* do nothing */
}

/*----------------------------------------------------------------------*
 |  ctor (public)                                            mwgee 10/07|
 *----------------------------------------------------------------------*/
CONTACT::Interface::Interface(const std::shared_ptr<Mortar::InterfaceDataContainer>& interfaceData,
    const int id, MPI_Comm comm, const int spatialDim, const Teuchos::ParameterList& icontact,
    bool selfcontact)
    : Mortar::Interface(interfaceData, id, comm, spatialDim, icontact,
          Global::Problem::instance()->output_control_file(),
          Global::Problem::instance()->spatial_approximation_type()),
      interface_data_(std::dynamic_pointer_cast<CONTACT::InterfaceDataContainer>(interfaceData)),
      selfcontact_(interface_data_->is_self_contact()),
      friction_(interface_data_->is_friction()),
      nonSmoothContact_(interface_data_->is_non_smooth_contact()),
      two_half_pass_(interface_data_->is_two_half_pass()),
      constr_direction_(interface_data_->constraint_direction()),
      activenodes_(interface_data_->active_nodes()),
      activedofs_(interface_data_->active_dofs()),
      inactivenodes_(interface_data_->inactive_nodes()),
      inactivedofs_(interface_data_->inactive_dofs()),
      activen_(interface_data_->active_n()),
      activet_(interface_data_->active_t()),
      slipnodes_(interface_data_->slip_nodes()),
      slipdofs_(interface_data_->slip_dofs()),
      slipt_(interface_data_->slip_t()),
      nonsmoothnodes_(interface_data_->non_smooth_nodes()),
      smoothnodes_(interface_data_->smooth_nodes()),
      sdofVertexRowmap_(interface_data_->sdof_vertex_rowmap()),
      sdofVertexColmap_(interface_data_->sdof_vertex_colmap()),
      sdofEdgeRowmap_(interface_data_->sdof_edge_rowmap()),
      sdofEdgeColmap_(interface_data_->sdof_edge_colmap()),
      sdofSurfRowmap_(interface_data_->sdof_surf_rowmap()),
      sdofSurfColmap_(interface_data_->sdof_surf_colmap()),
      nextendedghosting_(interface_data_->n_extended_ghosting()),
      eextendedghosting_(interface_data_->e_extended_ghosting()),
      binarytreeself_(interface_data_->binary_tree_self()),
      cnValues_(interface_data_->cn_values()),
      ctValues_(interface_data_->ct_values()),
      smpairs_(interface_data_->sm_int_pairs()),
      smintpairs_(interface_data_->sm_int_pairs()),
      intcells_(interface_data_->int_cells())
{
  selfcontact_ = selfcontact;
  nonSmoothContact_ = icontact.get<bool>("NONSMOOTH_GEOMETRIES");
  two_half_pass_ = icontact.get<bool>("Two_half_pass");
  constr_direction_ =
      Teuchos::getIntegralValue<CONTACT::ConstraintDirection>(icontact, "CONSTRAINT_DIRECTIONS");
  smpairs_ = 0;
  smintpairs_ = 0;
  intcells_ = 0;

  // set frictional contact status
  auto ftype = Teuchos::getIntegralValue<CONTACT::FrictionType>(icontact, "FRICTION");
  if (ftype != CONTACT::FrictionType::none) friction_ = true;

  // set poro contact
  if (icontact.get<CONTACT::Problemtype>("PROBTYPE") == CONTACT::Problemtype::poroelast ||
      icontact.get<CONTACT::Problemtype>("PROBTYPE") == CONTACT::Problemtype::poroscatra ||
      icontact.get<CONTACT::Problemtype>("PROBTYPE") == CONTACT::Problemtype::fpi)
  {
    set_poro_flag(true);
    set_poro_type(Inpar::Mortar::poroelast);
  }
  if (icontact.get<CONTACT::Problemtype>("PROBTYPE") == CONTACT::Problemtype::poroscatra)
    set_poro_type(Inpar::Mortar::poroscatra);

  // set ehl contact
  if (icontact.get<CONTACT::Problemtype>("PROBTYPE") == CONTACT::Problemtype::ehl)
    set_ehl_flag(true);

  // check for redundant slave storage
  // needed for self contact but not wanted for general contact
  // for self contact this is ensured in BuildInterfaces in contact_strategy_factory.cpp
  // so we only print a warning here, as it is possible to have another contact interface with a
  // different ID that does not need to be a self contact interface
  if (!(selfcontact_ or nonSmoothContact_) &&
      interface_data_->get_extend_ghosting() == Inpar::Mortar::ExtendGhosting::redundant_all)
  {
    if (Core::Communication::my_mpi_rank(Interface::get_comm()) == 0)
    {
      std::cout << "\n\nWARNING: We do not want redundant interface storage for contact where not "
                   "needed, as it is very expensive. But we need it e.g. for self contact."
                << '\n';
    }
  }

  // initialize extended ghosting for RR loop
  eextendedghosting_ = nullptr;
  nextendedghosting_ = nullptr;
}

/*----------------------------------------------------------------------*
 |  update master and slave sets (nodes etc.)                farah 10/16|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::update_master_slave_sets()
{
  // call mortar function
  Mortar::Interface::update_master_slave_sets();

  //********************************************************************
  // DOFS
  //********************************************************************
  // do the same business for dofs
  // (get row and column maps of slave and master dofs separately)
  if (nonSmoothContact_)
  {
    std::vector<int> sVc;  // slave column map
    std::vector<int> sVr;  // slave row map
    std::vector<int> sEc;  // master column map
    std::vector<int> sEr;  // master row map
    std::vector<int> sSc;  // master column map
    std::vector<int> sSr;  // master row map

    for (int i = 0; i < discret().node_col_map()->num_my_elements(); ++i)
    {
      int gid = discret().node_col_map()->gid(i);
      Core::Nodes::Node* node = discret().g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      auto* mrtrnode = dynamic_cast<Node*>(node);
      const bool isslave = mrtrnode->is_slave();
      const int numdof = mrtrnode->num_dof();

      if (isslave)
      {
        // vertex
        if (mrtrnode->is_on_corner())
        {
          for (int j = 0; j < numdof; ++j) sVc.push_back(mrtrnode->dofs()[j]);

          if (discret().node_row_map()->my_gid(gid))
            for (int j = 0; j < numdof; ++j) sVr.push_back(mrtrnode->dofs()[j]);
        }
        // edge
        else if (mrtrnode->is_on_edge())
        {
          for (int j = 0; j < numdof; ++j) sEc.push_back(mrtrnode->dofs()[j]);

          if (discret().node_row_map()->my_gid(gid))
            for (int j = 0; j < numdof; ++j) sEr.push_back(mrtrnode->dofs()[j]);
        }
        // surface
        else if (!mrtrnode->is_on_corner_edge())
        {
          for (int j = 0; j < numdof; ++j) sSc.push_back(mrtrnode->dofs()[j]);

          if (discret().node_row_map()->my_gid(gid))
            for (int j = 0; j < numdof; ++j) sSr.push_back(mrtrnode->dofs()[j]);
        }
        else
        {
          FOUR_C_THROW("unknown case!");
        }
      }
    }

    sdofVertexRowmap_ =
        std::make_shared<Core::LinAlg::Map>(-1, (int)sVr.size(), sVr.data(), 0, get_comm());
    sdofVertexColmap_ =
        std::make_shared<Core::LinAlg::Map>(-1, (int)sVc.size(), sVc.data(), 0, get_comm());
    sdofEdgeRowmap_ =
        std::make_shared<Core::LinAlg::Map>(-1, (int)sEr.size(), sEr.data(), 0, get_comm());
    sdofEdgeColmap_ =
        std::make_shared<Core::LinAlg::Map>(-1, (int)sEc.size(), sEc.data(), 0, get_comm());
    sdofSurfRowmap_ =
        std::make_shared<Core::LinAlg::Map>(-1, (int)sSr.size(), sSr.data(), 0, get_comm());
    sdofSurfColmap_ =
        std::make_shared<Core::LinAlg::Map>(-1, (int)sSc.size(), sSc.data(), 0, get_comm());
  }
}

/*----------------------------------------------------------------------*
 |  create and fill cn vector                                farah 07/16|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::set_cn_ct_values(const int& iter)
{
  // get cn from the input file
  const double cn = interface_params().get<double>("SEMI_SMOOTH_CN");
  const double ct = interface_params().get<double>("SEMI_SMOOTH_CT");

  // set all nodal cn-values to the input value
  get_cn() = std::make_shared<Core::LinAlg::Vector<double>>(*slave_row_nodes(), true);
  get_cn()->put_scalar(cn);

  // set all nodal ct-values to the input value
  if (friction_)
  {
    get_ct() = std::make_shared<Core::LinAlg::Vector<double>>(*slave_row_nodes(), true);
    get_ct()->put_scalar(ct);
  }

  // modification for edge/corner nodes
  for (int i = 0; i < slave_row_nodes()->num_my_elements(); ++i)
  {
    int gid = slave_row_nodes()->gid(i);
    Core::Nodes::Node* node = discret().g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid {}", gid);
    Node* cnode = dynamic_cast<Node*>(node);

    // calculate characteristic edge length:
    Core::Elements::Element* ele = cnode->adjacent_elements()[0].user_element();
    auto* cele = dynamic_cast<Element*>(ele);
    std::array<double, 3> pos1 = {0.0, 0.0, 0.0};
    std::array<double, 3> pos2 = {0.0, 0.0, 0.0};
    std::array<double, 3> vec = {0.0, 0.0, 0.0};

    pos1[0] = dynamic_cast<Node*>(cele->nodes()[0])->x()[0];
    pos1[1] = dynamic_cast<Node*>(cele->nodes()[0])->x()[1];

    if (dim_ == 3) pos1[2] = dynamic_cast<Node*>(cele->nodes()[0])->x()[2];

    pos2[0] = dynamic_cast<Node*>(cele->nodes()[1])->x()[0];
    pos2[1] = dynamic_cast<Node*>(cele->nodes()[1])->x()[1];
    if (dim_ == 3) pos2[2] = dynamic_cast<Node*>(cele->nodes()[1])->x()[2];

    vec[0] = pos1[0] - pos2[0];
    vec[1] = pos1[1] - pos2[1];
    vec[2] = pos1[2] - pos2[2];

    const double length = sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
    if (length < 1e-12)
    {
      std::cout << "*** WARNING *** element edge nearly zero" << '\n';
      continue;
    }

    if (cnode->is_on_edge())
    {
      get_cn_ref().get_values()[get_cn_ref().get_map().lid(cnode->id())] = cn * (length * length);
      if (friction_)
        get_ct_ref().get_values()[get_ct_ref().get_map().lid(cnode->id())] = ct * (length * length);
    }

    if (cnode->is_on_corner())
    {
      get_cn_ref().get_values()[get_cn_ref().get_map().lid(cnode->id())] =
          cn * (length * length * length * length);
      if (friction_)
        get_ct_ref().get_values()[get_ct_ref().get_map().lid(cnode->id())] =
            ct * (length * length * length * length);
    }
  }
}

/*----------------------------------------------------------------------*
 |  << operator                                              mwgee 10/07|
 *----------------------------------------------------------------------*/
std::ostream& operator<<(std::ostream& os, const CONTACT::Interface& interface)
{
  interface.print(os);
  return os;
}

/*----------------------------------------------------------------------*
 |  print interface (public)                                 mwgee 10/07|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::print(std::ostream& os) const
{
  if (Core::Communication::my_mpi_rank(get_comm()) == 0) os << "Contact ";
  Mortar::Interface::print(os);
}

/*----------------------------------------------------------------------*
 |  add contact node (public)                                mwgee 10/07|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::add_node(std::shared_ptr<CONTACT::Node> cnode)
{
  idiscret_->add_node(cnode->x(), cnode->id(), cnode);
}

/*----------------------------------------------------------------------*
 |  add contact element (public)                             mwgee 10/07|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::add_element(std::shared_ptr<CONTACT::Element> cele)
{
  // check for quadratic 2d slave elements to be modified
  if (cele->is_slave() && (cele->shape() == Core::FE::CellType::line3)) quadslave_ = true;

  // check for quadratic 3d slave elements to be modified
  if (cele->is_slave() &&
      (cele->shape() == Core::FE::CellType::quad8 || cele->shape() == Core::FE::CellType::tri6))
    quadslave_ = true;

  idiscret_->add_element(cele);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::Interface::update_parallel_layout_and_data_structures(const bool perform_rebalancing,
    const bool enforce_ghosting_update, const int maxdof, const double meanVelocity)
{
  if (perform_rebalancing)
  {
    redistribute();
    fill_complete_new(false, maxdof);
  }

  if (perform_rebalancing || enforce_ghosting_update)
  {
    // Assure that at least some maps are available
    if (!filled()) fill_complete_new(false, maxdof);

    // Finalize interface maps
    extend_interface_ghosting_safely(meanVelocity);
    fill_complete_new(true, maxdof);
  }

  // print new parallel distribution
  if (perform_rebalancing)
  {
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      std::cout << "Interface parallel distribution after rebalancing:" << '\n';
    print_parallel_distribution();
  }

  if (perform_rebalancing || enforce_ghosting_update) create_search_tree();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::Interface::fill_complete_new(const bool isFinalParallelDistribution, const int maxdof)
{
  std::stringstream ss;
  ss << "CONTACT::Interface::fill_complete_new of '" << discret().name() << "'";
  TEUCHOS_FUNC_TIME_MONITOR(ss.str());

  // store maximum global dof ID handed in
  // this ID is later needed when setting up the Lagrange multiplier
  // dof map, which of course must not overlap with existing dof ranges
  maxdofglobal_ = maxdof;

  /* We'd like to call idiscret_.fill_complete(true,false,false) but this will assign all nodes new
   * degrees of freedom which we don't want. We would like to use the degrees of freedom that were
   * stored in the mortar nodes. To do so, we have to create and set our own version of a
   * Mortar::DofSet class before we call fill_complete on the interface discretization. The
   * specialized DofSet class will not assign new dofs but will assign the dofs stored in the nodes.
   */
  {
    std::shared_ptr<Mortar::DofSet> mrtrdofset = std::make_shared<Mortar::DofSet>();
    discret().replace_dof_set(mrtrdofset);
  }

  // fill_complete the interface discretization
  discret().fill_complete({
      .assign_degrees_of_freedom = isFinalParallelDistribution,
      .init_elements = false,
      .do_boundary_conditions = false,
  });

  // check whether crosspoints / edge nodes shall be considered or not
  initialize_cross_points();

  // check for const/linear interpolation of 2D/3D quadratic Lagrange multipliers
  initialize_lag_mult_const();
  initialize_lag_mult_lin();

  // check/init corner/edge modification
  initialize_corner_edge();

  // need row and column maps of slave and master nodes / elements / dofs
  // separately so we can easily address them
  update_master_slave_sets();

  // initialize node and element data container
  initialize_data_container();

  // Communicate quadslave status among ALL processors
  communicate_quad_slave_status_among_all_procs();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::Interface::extend_interface_ghosting_safely(const double meanVelocity)
{
  if (discret().node_col_map() == nullptr) FOUR_C_THROW("NodeColMap not set.");
  if (discret().element_col_map() == nullptr) FOUR_C_THROW("ElementColMap not set.");

  // later we might export node and element column map to extended or even FULL overlap,
  // thus store the standard column maps first
  {
    // get standard nodal column map (overlap=1)
    oldnodecolmap_ = std::make_shared<Core::LinAlg::Map>(*(discret().node_col_map()));

    // get standard element column map (overlap=1)
    oldelecolmap_ = std::make_shared<Core::LinAlg::Map>(*(discret().element_col_map()));
  }

  switch (interface_data_->get_extend_ghosting())
  {
    case Inpar::Mortar::ExtendGhosting::redundant_all:
    {
      // to ease our search algorithms we'll afford the luxury to ghost all nodes
      // on all processors. To do so, we'll take the node row map and export it to
      // full overlap. Then we export the discretization to full overlap column map.
      // This way, also all mortar elements will be fully ghosted on all processors.

      // we want to do full ghosting on all procs
      std::vector<int> allproc(Core::Communication::num_mpi_ranks(get_comm()));
      for (int i = 0; i < Core::Communication::num_mpi_ranks(get_comm()); ++i) allproc[i] = i;

      // fill my own row node ids
      const Core::LinAlg::Map* noderowmap = discret().node_row_map();
      std::vector<int> sdata(noderowmap->num_my_elements());
      for (int i = 0; i < noderowmap->num_my_elements(); ++i) sdata[i] = noderowmap->gid(i);

      // gather all gids of nodes redundantly
      std::vector<int> rdata;
      Core::LinAlg::gather<int>(sdata, rdata, (int)allproc.size(), allproc.data(), get_comm());

      // build completely overlapping map of nodes (on ALL processors)
      Core::LinAlg::Map newnodecolmap(-1, (int)rdata.size(), rdata.data(), 0, get_comm());
      sdata.clear();
      rdata.clear();

      // fill my own row element ids
      const Core::LinAlg::Map* elerowmap = discret().element_row_map();
      sdata.resize(elerowmap->num_my_elements());
      for (int i = 0; i < elerowmap->num_my_elements(); ++i) sdata[i] = elerowmap->gid(i);

      // gather all gids of elements redundantly
      rdata.resize(0);
      Core::LinAlg::gather<int>(sdata, rdata, (int)allproc.size(), allproc.data(), get_comm());

      // build complete overlapping map of elements (on ALL processors)
      Core::LinAlg::Map newelecolmap(-1, (int)rdata.size(), rdata.data(), 0, get_comm());
      sdata.clear();
      rdata.clear();
      allproc.clear();

      // redistribute the discretization of the interface according to the
      // new column layout
      discret().export_column_nodes(newnodecolmap);
      discret().export_column_elements(newelecolmap);

      break;
    }
    case Inpar::Mortar::ExtendGhosting::redundant_master:
    {
      // to ease our search algorithms we'll afford the luxury to ghost all master
      // nodes on all processors. To do so, we'll take the master node row map and
      // export it to full overlap. Then we export the discretization to partially
      // full overlap column map. This way, also all master elements will be fully
      // ghosted on all processors.

      // at least for master, we want to do full ghosting on all procs
      std::vector<int> allproc(Core::Communication::num_mpi_ranks(get_comm()));
      for (int i = 0; i < Core::Communication::num_mpi_ranks(get_comm()); ++i) allproc[i] = i;

      // fill my own master row node ids
      const Core::LinAlg::Map* noderowmap = discret().node_row_map();
      std::vector<int> sdata;
      for (int i = 0; i < noderowmap->num_my_elements(); ++i)
      {
        int gid = noderowmap->gid(i);
        Core::Nodes::Node* node = discret().g_node(gid);
        if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
        auto* mrtrnode = dynamic_cast<Mortar::Node*>(node);
        if (!mrtrnode->is_slave()) sdata.push_back(gid);
      }

      // gather all master row node gids redundantly
      std::vector<int> rdata;
      Core::LinAlg::gather<int>(sdata, rdata, (int)allproc.size(), allproc.data(), get_comm());

      // add my own slave column node ids (non-redundant, standard overlap)
      const Core::LinAlg::Map* nodecolmap = discret().node_col_map();
      for (int i = 0; i < nodecolmap->num_my_elements(); ++i)
      {
        int gid = nodecolmap->gid(i);
        Core::Nodes::Node* node = discret().g_node(gid);
        if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
        auto* mrtrnode = dynamic_cast<Mortar::Node*>(node);
        if (mrtrnode->is_slave()) rdata.push_back(gid);
      }

      // build new node column map (on ALL processors)
      Core::LinAlg::Map newnodecolmap(-1, (int)rdata.size(), rdata.data(), 0, get_comm());
      sdata.clear();
      rdata.clear();

      // fill my own master row element ids
      const Core::LinAlg::Map* elerowmap = discret().element_row_map();
      sdata.resize(0);
      for (int i = 0; i < elerowmap->num_my_elements(); ++i)
      {
        int gid = elerowmap->gid(i);
        Core::Elements::Element* ele = discret().g_element(gid);
        if (!ele) FOUR_C_THROW("Cannot find element with gid %", gid);
        auto* mrtrele = dynamic_cast<Mortar::Element*>(ele);
        if (!mrtrele->is_slave()) sdata.push_back(gid);
      }

      // gather all gids of elements redundantly
      rdata.resize(0);
      Core::LinAlg::gather<int>(sdata, rdata, (int)allproc.size(), allproc.data(), get_comm());

      // add my own slave column node ids (non-redundant, standard overlap)
      const Core::LinAlg::Map* elecolmap = discret().element_col_map();
      for (int i = 0; i < elecolmap->num_my_elements(); ++i)
      {
        int gid = elecolmap->gid(i);
        Core::Elements::Element* ele = discret().g_element(gid);
        if (!ele) FOUR_C_THROW("Cannot find element with gid %", gid);
        auto* mrtrele = dynamic_cast<Mortar::Element*>(ele);
        if (mrtrele->is_slave()) rdata.push_back(gid);
      }

      // build new element column map (on ALL processors)
      Core::LinAlg::Map newelecolmap(-1, (int)rdata.size(), rdata.data(), 0, get_comm());
      sdata.clear();
      rdata.clear();
      allproc.clear();

      // redistribute the discretization of the interface according to the
      // new node / element column layout (i.e. master = full overlap)
      discret().export_column_nodes(newnodecolmap);
      discret().export_column_elements(newelecolmap);

      break;
    }
    case Inpar::Mortar::ExtendGhosting::roundrobin:
    {
      // Nothing to do in case of Round-Robin
      break;
    }
    case Inpar::Mortar::ExtendGhosting::binning:
    {
      // Extend master column map via binning

      // Create the binning strategy
      auto binningstrategy =
          setup_binning_strategy(Global::Problem::instance()->binning_strategy_params(),
              meanVelocity, Global::Problem::instance()->output_control_file(),
              Global::Problem::instance()->spatial_approximation_type());

      // fill master and slave elements into bins
      std::map<int, std::set<int>> slavebinelemap;
      binningstrategy->distribute_elements_to_bins_using_ele_aabb(discret(),
          std::views::filter(discret().my_col_element_range(), [](auto ele)
              { return dynamic_cast<const Mortar::Element*>(ele.user_element())->is_slave(); }),
          slavebinelemap);

      std::map<int, std::set<int>> masterbinelemap;
      binningstrategy->distribute_elements_to_bins_using_ele_aabb(discret(),
          std::views::filter(discret().my_col_element_range(), [](auto ele)
              { return !dynamic_cast<const Mortar::Element*>(ele.user_element())->is_slave(); }),
          masterbinelemap);

      // Extend ghosting of the master elements
      std::map<int, std::set<int>> ext_bin_to_ele_map;
      auto extendedmastercolmap = binningstrategy->extend_element_col_map(slavebinelemap,
          masterbinelemap, ext_bin_to_ele_map, nullptr, nullptr, discret().element_col_map());

      discret().export_column_elements(*extendedmastercolmap);

      // get the node ids of the elements that are to be ghosted and create a proper node column
      // map for their export
      std::set<int> nodes;
      const int numMasterColElements = extendedmastercolmap->num_my_elements();
      for (int lid = 0; lid < numMasterColElements; ++lid)
      {
        Core::Elements::Element* ele = discret().g_element(extendedmastercolmap->gid(lid));
        const int* nodeids = ele->node_ids();
        for (int inode = 0; inode < ele->num_node(); ++inode) nodes.insert(nodeids[inode]);
      }

      std::vector<int> colnodes(nodes.begin(), nodes.end());
      Core::LinAlg::Map nodecolmap(-1, (int)colnodes.size(), colnodes.data(), 0, get_comm());

      discret().export_column_nodes(nodecolmap);
      break;
    }
    default:
    {
      FOUR_C_THROW(
          "This case of redundant interface storage has not been covered, yet. Implement it!");
      break;
    }
  }
}


/*----------------------------------------------------------------------*
 |  redistribute contact interface (public)                   popp 08/10|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::redistribute()
{
  const Teuchos::ParameterList& mortarParallelRedistParams =
      interface_params().sublist("PARALLEL REDISTRIBUTION");

  // make sure we are supposed to be here
  if (Teuchos::getIntegralValue<Inpar::Mortar::ParallelRedist>(mortarParallelRedistParams,
          "PARALLEL_REDIST") == Inpar::Mortar::ParallelRedist::redist_none)
  {
    FOUR_C_THROW(
        "You are not supposed to be here since you did not enable PARALLEL_REDIST in the "
        "input file. ");
  }

  // some local variables
  MPI_Comm comm(Interface::get_comm());
  const int myrank = Core::Communication::my_mpi_rank(comm);
  const int numproc = Core::Communication::num_mpi_ranks(comm);
  Teuchos::Time time("", true);
  std::set<int>::const_iterator iter;

  // vector containing all proc ids
  std::vector<int> allproc(numproc);
  for (int i = 0; i < numproc; ++i) allproc[i] = i;

  //**********************************************************************
  // (1) SLAVE splitting in close / non-close parts
  //**********************************************************************
  // perform contact search (still with non-optimal distribution)
  initialize();
  if (search_alg() == Inpar::Mortar::search_bfele)
    evaluate_search_brute_force(search_param());
  else if (search_alg() == Inpar::Mortar::search_binarytree)
    evaluate_search_binarytree();
  else
    FOUR_C_THROW("Invalid search algorithm");

  // split slave element row map and build redundant vector of
  // all close / non-close slave node ids on all procs
  std::vector<int> closeele, noncloseele;
  std::vector<int> localcns, localfns;

  split_into_far_and_close_sets(closeele, noncloseele, localcns, localfns);

  // loop over all elements to reset candidates / search lists
  // (use standard slave column map)
  const int numMySlaveColElements = slave_col_elements()->num_my_elements();
  for (int i = 0; i < numMySlaveColElements; ++i)
  {
    int gid = slave_col_elements()->gid(i);
    Core::Elements::Element* ele = discret().g_element(gid);
    if (!ele) FOUR_C_THROW("Cannot find ele with gid {}", gid);
    auto* mele = dynamic_cast<Mortar::Element*>(ele);

    mele->mo_data().search_elements().resize(0);
  }

  // we need an arbitrary preliminary element row map
  Core::LinAlg::Map slaveCloseRowEles(
      -1, (int)closeele.size(), closeele.data(), 0, Interface::get_comm());
  Core::LinAlg::Map slaveNonCloseRowEles(
      -1, (int)noncloseele.size(), noncloseele.data(), 0, Interface::get_comm());
  Core::LinAlg::Map masterRowEles(*master_row_elements());

  // check for consistency
  if (slaveCloseRowEles.num_global_elements() == 0 &&
      slaveNonCloseRowEles.num_global_elements() == 0)
    FOUR_C_THROW("CONTACT redistribute: Both slave sets (close/non-close) are empty");

  //**********************************************************************
  // (2) SPECIAL CASES and output to screen
  //**********************************************************************
  // print element overview
  if (!myrank)
  {
    int cl = slaveCloseRowEles.num_global_elements();
    int ncl = slaveNonCloseRowEles.num_global_elements();
    int ma = masterRowEles.num_global_elements();
    std::cout << "Element overview: " << cl << " / " << ncl << " / " << ma
              << "  (close-S / non-close-S / M)";
  }

  // print old parallel distribution
  if (myrank == 0)
  {
    std::cout << "\nInterface parallel distribution before rebalancing:" << '\n';
  }
  print_parallel_distribution();

  // use simple base class method if there are ONLY close or non-close elements
  // (return value TRUE, because redistribution performed)
  if (slaveCloseRowEles.num_global_elements() == 0 ||
      slaveNonCloseRowEles.num_global_elements() == 0)
  {
    Mortar::Interface::redistribute();
    return;
  }

  //**********************************************************************
  // (3a) PREPARATIONS decide how many procs are used
  //**********************************************************************
  // first we assume that all procs will be used
  int scproc = numproc;
  int sncproc = numproc;
  int mproc = numproc;

  // minimum number of elements per proc
  const int minele = mortarParallelRedistParams.get<int>("MIN_ELEPROC");

  // Max. relative imbalance between subdomain sizes
  const double imbalance_tol = mortarParallelRedistParams.get<double>("IMBALANCE_TOL");

  // calculate real number of procs to be used
  if (minele > 0)
  {
    scproc = static_cast<int>((slaveCloseRowEles.num_global_elements()) / minele);
    sncproc = static_cast<int>((slaveNonCloseRowEles.num_global_elements()) / minele);
    mproc = static_cast<int>((masterRowEles.num_global_elements()) / minele);
    if (slaveCloseRowEles.num_global_elements() < 2 * minele) scproc = 1;
    if (slaveNonCloseRowEles.num_global_elements() < 2 * minele) sncproc = 1;
    if (masterRowEles.num_global_elements() < 2 * minele) mproc = 1;
    if (scproc > numproc) scproc = numproc;
    if (sncproc > numproc) sncproc = numproc;
    if (mproc > numproc) mproc = numproc;
  }

  // print message
  if (!myrank)
  {
    std::cout << "\nRedistributing interface '" << discret().name() << "' .........\n";
    std::cout << "Procs used for redistribution: " << scproc << " / " << sncproc << " / " << mproc
              << " (close-S / non-close-S / M)\n";
  }

  //**********************************************************************
  // (3b) PREPARATIONS build initial node graph
  //**********************************************************************
  // create graph object
  std::shared_ptr<Core::LinAlg::Graph> graph =
      std::make_shared<Core::LinAlg::Graph>(*slave_row_nodes(), 108);

  // loop over all row nodes to fill graph
  const int numMySlaveRowNodes = slave_row_nodes()->num_my_elements();
  for (int slaveRowNode = 0; slaveRowNode < numMySlaveRowNodes; ++slaveRowNode)
  {
    int gid = slave_row_nodes()->gid(slaveRowNode);
    Core::Nodes::Node* node = discret().g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);

    // find adjacent elements first
    for (auto ele : node->adjacent_elements())
    {
      // store adjacent nodes
      int numnode = ele.num_nodes();
      std::vector<int> nodeids(numnode);
      std::ranges::transform(
          ele.nodes(), nodeids.begin(), [](auto node) { return node.global_id(); });

      auto indices = std::span(nodeids.data(), numnode);
      graph->insert_global_indices(gid, indices);
    }
  }

  // fill graph and optimize storage
  graph->fill_complete();
  graph->optimize_storage();

  //**********************************************************************
  // (4) CLOSE SLAVE redistribution
  //**********************************************************************

  // build redundant vector of all close slave node ids on all procs
  // (there must not be any double entries in the node lists, thus
  // transform to sets and then back to vectors)
  std::vector<int> globalcns;
  std::set<int> setglobalcns;
  std::vector<int> scnids;
  Core::LinAlg::gather<int>(localcns, globalcns, numproc, allproc.data(), Interface::get_comm());
  for (int globalcn : globalcns)
  {
    setglobalcns.insert(globalcn);
  }
  for (iter = setglobalcns.begin(); iter != setglobalcns.end(); ++iter)
  {
    scnids.push_back(*iter);
  }

  //**********************************************************************
  // call parallel redistribution
  std::shared_ptr<const Core::LinAlg::Graph> slaveCloseNodeGraph =
      Core::Rebalance::build_graph(*idiscret_, slaveCloseRowEles);

  Teuchos::ParameterList slaveCloseRebalanceParams;
  slaveCloseRebalanceParams.set<std::string>("num_global_parts", std::to_string(scproc));
  slaveCloseRebalanceParams.set<std::string>("imbalance_tolerance", std::to_string(imbalance_tol));
  slaveCloseRebalanceParams.set<std::string>("algorithm", "parmetis");

  const auto& [slaveCloseRowNodes, slaveCloseColNodes] =
      Core::Rebalance::rebalance_node_maps(*slaveCloseNodeGraph, slaveCloseRebalanceParams);
  //**********************************************************************

  //**********************************************************************
  // (5) NON-CLOSE SLAVE redistribution
  //**********************************************************************

  // build redundant vector of all non-close slave node ids on all procs
  // (there must not be any double entries in the node lists, thus
  // transform to sets and then back to vectors)
  std::vector<int> globalfns;
  std::set<int> setglobalfns;
  std::vector<int> sncnids;
  Core::LinAlg::gather<int>(localfns, globalfns, numproc, allproc.data(), Interface::get_comm());
  for (int globalfn : globalfns) setglobalfns.insert(globalfn);
  for (iter = setglobalfns.begin(); iter != setglobalfns.end(); ++iter) sncnids.push_back(*iter);

  //**********************************************************************
  // call parallel redistribution
  std::shared_ptr<const Core::LinAlg::Graph> slaveNonCloseNodeGraph =
      Core::Rebalance::build_graph(*idiscret_, slaveNonCloseRowEles);

  Teuchos::ParameterList slaveNonCloseRebalanceParams;
  slaveNonCloseRebalanceParams.set<std::string>("num_global_parts", std::to_string(sncproc));
  slaveNonCloseRebalanceParams.set<std::string>(
      "imbalance_tolerance", std::to_string(imbalance_tol));
  slaveNonCloseRebalanceParams.set<std::string>("algorithm", "parmetis");

  const auto& [slaveNonCloseRowNodes, snccolnodes] =
      Core::Rebalance::rebalance_node_maps(*slaveNonCloseNodeGraph, slaveNonCloseRebalanceParams);
  //**********************************************************************

  //**********************************************************************
  // (6) MASTER redistribution
  //**********************************************************************
  std::shared_ptr<Core::LinAlg::Map> mrownodes = nullptr;
  std::shared_ptr<Core::LinAlg::Map> mcolnodes = nullptr;

  redistribute_master_side(mrownodes, mcolnodes, masterRowEles, comm, mproc, imbalance_tol);

  //**********************************************************************
  // (7) Merge global interface node row and column map
  //**********************************************************************
  // merge slave node row map from close and non-close parts
  std::shared_ptr<Core::LinAlg::Map> srownodes = nullptr;

  //----------------------------------CASE 1: ONE OR BOTH SLAVE SETS EMPTY
  if (slaveCloseRowNodes == nullptr || slaveNonCloseRowNodes == nullptr)
  {
    FOUR_C_THROW("CONTACT redistribute: Both slave sets (close/non-close) are empty");
  }
  //-------------------------------------CASE 2: BOTH SLAVE SETS NON-EMPTY
  else
  {
    // find intersection set of close and non-close nodes
    std::set<int> intersec;
    for (iter = setglobalcns.begin(); iter != setglobalcns.end(); ++iter)
    {
      auto found = setglobalfns.find(*iter);
      if (found != setglobalfns.end()) intersec.insert(*found);
    }

    // build slave node row map
    const int numMySlaveCloseRowNodes = slaveCloseRowNodes->num_my_elements();
    const int numMySlaveNonCloseRowNodes = slaveNonCloseRowNodes->num_my_elements();
    std::vector<int> mygids(numMySlaveCloseRowNodes + numMySlaveNonCloseRowNodes);
    int count = slaveCloseRowNodes->num_my_elements();

    // first get GIDs of input slaveCloseRowNodes
    for (int i = 0; i < count; ++i) mygids[i] = slaveCloseRowNodes->gid(i);

    // then add GIDs of input slaveNonCloseRowNodes (only new ones)
    for (int i = 0; i < numMySlaveNonCloseRowNodes; ++i)
    {
      // check for intersection gid
      // don't do anything for intersection gids (slaveCloseRowNodes dominates!!!)
      auto found = intersec.find(slaveNonCloseRowNodes->gid(i));
      if (found != intersec.end()) continue;

      // check for overlap
      if (slaveCloseRowNodes->my_gid(slaveNonCloseRowNodes->gid(i)))
        FOUR_C_THROW("Core::LinAlg::merge_map: Result map is overlapping");

      // add new GIDs to mygids
      mygids[count] = slaveNonCloseRowNodes->gid(i);
      ++count;
    }
    mygids.resize(count);
    sort(mygids.begin(), mygids.end());
    srownodes = std::make_shared<Core::LinAlg::Map>(
        -1, (int)mygids.size(), mygids.data(), 0, slaveCloseRowNodes->get_comm());
  }

  // merge interface node row map from slave and master parts
  std::shared_ptr<Core::LinAlg::Map> rownodes =
      Core::LinAlg::merge_map(srownodes, mrownodes, false);

  // IMPORTANT NOTE:
  // While merging from the two different slave parts of the discretization
  // (close slave, non-close slave) is feasible for the node row map,
  // this is not possible for the node column map. Some necessary
  // information on ghosting at the transition between close and non-close
  // slave region would always be missed! Thus, we reconstruct a
  // suitable slave node column map "by hand" here. This is quite simply
  // done by exporting the initial node graph to the new distribution
  // and by then asking for its column map.

  // create the output graph (with new slave node row map) and export to it
  std::shared_ptr<Core::LinAlg::Graph> outgraph =
      std::make_shared<Core::LinAlg::Graph>(*srownodes, 108);
  Core::LinAlg::Export exporter(graph->row_map(), *srownodes);
  outgraph->export_to(*graph, exporter, Core::LinAlg::CombineMode::add);

  // trash old graph
  graph = nullptr;

  // call fill complete and optimize storage
  outgraph->fill_complete();
  outgraph->optimize_storage();

  // get column map from the graph -> build slave node column map
  const Core::LinAlg::Map& bcol = outgraph->col_map();
  std::shared_ptr<Core::LinAlg::Map> scolnodes =
      std::make_shared<Core::LinAlg::Map>(bcol.num_global_elements(), bcol.num_my_elements(),
          bcol.my_global_elements(), 0, Interface::get_comm());

  // trash new graph
  outgraph = nullptr;

  // merge interface node column map from slave and master parts
  std::shared_ptr<Core::LinAlg::Map> colnodes =
      Core::LinAlg::merge_map(scolnodes, mcolnodes, false);

  //**********************************************************************
  // (8) Get partitioning information into discretization
  //**********************************************************************
  // build reasonable element maps from the already valid and final node maps
  // (note that nothing is actually redistributed in here)
  const auto& [roweles, coleles] = discret().build_element_row_column(*rownodes, *colnodes);

  // Redistribute but do NOT call fill_complete at all
  discret().redistribute(
      {*rownodes, *colnodes}, {*roweles, *coleles}, {.fill_complete = std::nullopt});
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::Interface::split_into_far_and_close_sets(std::vector<int>& closeele,
    std::vector<int>& noncloseele, std::vector<int>& localcns, std::vector<int>& localfns) const
{
  const bool performSplitting =
      interface_params().sublist("PARALLEL REDISTRIBUTION").get<bool>("EXPLOIT_PROXIMITY");

  if (performSplitting)
  {
    // loop over all row elements to gather the local information
    for (int i = 0; i < slave_row_elements()->num_my_elements(); ++i)
    {
      // get element
      int gid = slave_row_elements()->gid(i);
      Core::Elements::Element* ele = discret().g_element(gid);
      if (!ele) FOUR_C_THROW("Cannot find element with gid %", gid);
      auto* cele = dynamic_cast<Mortar::Element*>(ele);

      // store element id and adjacent node ids
      int close = cele->mo_data().num_search_elements();
      if (close > 0)
      {
        closeele.push_back(gid);
        for (int k = 0; k < cele->num_node(); ++k) localcns.push_back(cele->node_ids()[k]);
      }
      else
      {
        noncloseele.push_back(gid);
        for (int k = 0; k < cele->num_node(); ++k) localfns.push_back(cele->node_ids()[k]);
      }
    }
  }
  else
  {
    // loop over all row elements to gather the local information
    for (int i = 0; i < slave_row_elements()->num_my_elements(); ++i)
    {
      // get element
      int gid = slave_row_elements()->gid(i);
      Core::Elements::Element* ele = discret().g_element(gid);
      if (!ele) FOUR_C_THROW("Cannot find element with gid %", gid);
      auto* cele = dynamic_cast<Mortar::Element*>(ele);

      // store element id and adjacent node ids
      noncloseele.push_back(gid);
      for (int k = 0; k < cele->num_node(); ++k) localfns.push_back(cele->node_ids()[k]);
    }
  }
}

/*----------------------------------------------------------------------*
 | collect distribution data (public)                         popp 10/10|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::collect_distribution_data(int& numColElements, int& numRowElements)
{
  // loop over proc's column slave elements of the interface
  for (int i = 0; i < selecolmap_->num_my_elements(); ++i)
  {
    int gid1 = selecolmap_->gid(i);
    Core::Elements::Element* ele1 = idiscret_->g_element(gid1);
    if (!ele1) FOUR_C_THROW("Cannot find slave element with gid %", gid1);
    auto* slaveElement = dynamic_cast<Element*>(ele1);

    // bool indicating coupling partners
    bool add = (slaveElement->mo_data().num_search_elements() > 0);

    // Check if this element has any coupling partners.
    // Increment element counter if so.
    if (add) ++numColElements;

    // check if - in addition - the active proc owns this element.
    // Increment input variable rowele if so.
    if (add && slaveElement->owner() == Core::Communication::my_mpi_rank(get_comm()))
      ++numRowElements;
  }
}

/*----------------------------------------------------------------------*
 |  create search tree (public)                               popp 01/10|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::create_search_tree()
{
  // binary tree search
  if (search_alg() == Inpar::Mortar::search_binarytree)
  {
    //*****SELF CONTACT*****
    if (self_contact())
    {
      // set state in interface to initialize all kinds of quantities
      Core::LinAlg::Vector<double> zero(*idiscret_->dof_row_map());
      set_state(Mortar::state_new_displacement, zero);

      // create fully overlapping map of all contact elements
      std::shared_ptr<Core::LinAlg::Map> elefullmap =
          Core::LinAlg::allreduce_e_map(*idiscret_->element_row_map());

      // create binary tree object for self contact search
      if (!two_half_pass())
      {
        // (NOTE THAT SELF CONTACT SEARCH IS NOT YET FULLY PARALLELIZED!)
        binarytreeself_ = std::make_shared<CONTACT::SelfBinaryTree>(
            discret(), interface_params(), elefullmap, n_dim(), search_param());
      }
      else
      {
        // if we use the two half pass algorithm, we use the unbiased self binary tree
        // implementation
        binarytreeself_ = std::make_shared<CONTACT::UnbiasedSelfBinaryTree>(
            discret(), interface_params(), elefullmap, n_dim(), search_param());
      }
      // initialize the self binary tree
      binarytreeself_->init();
    }
    //*****TWO BODY CONTACT*****
    else
    {
      std::shared_ptr<Core::LinAlg::Map> melefullmap = nullptr;
      switch (interface_data_->get_extend_ghosting())
      {
        case Inpar::Mortar::ExtendGhosting::roundrobin:
        case Inpar::Mortar::ExtendGhosting::binning:
        {
          melefullmap = melecolmap_;
          break;
        }
        case Inpar::Mortar::ExtendGhosting::redundant_all:
        case Inpar::Mortar::ExtendGhosting::redundant_master:
        {
          melefullmap = Core::LinAlg::allreduce_e_map(*melerowmap_);
          break;
        }
        default:
        {
          FOUR_C_THROW("Chosen parallel strategy not supported!");
          break;
        }
      }

      {
        // get update type of binary tree
        auto updatetype = Teuchos::getIntegralValue<Inpar::Mortar::BinaryTreeUpdateType>(
            interface_params(), "BINARYTREE_UPDATETYPE");

        // create binary tree object for contact search and setup tree
        binarytree_ = std::make_shared<Mortar::BinaryTree>(discret(), selecolmap_, melefullmap,
            n_dim(), search_param(), updatetype, search_use_aux_pos());
        // initialize the binary tree
        binarytree_->init();
      }
    }
  }

  // no binary tree search
  else
  {
    if (self_contact()) FOUR_C_THROW("Binarytree search needed for self contact");
  }
}


/*----------------------------------------------------------------------*
 |  Initialize Data Container for nodes and elements         farah 02/16|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::initialize_data_container()
{
  // call base class functionality
  Mortar::Interface::initialize_data_container();

  // ==================
  // non-smooth contact:
  // we need this master node data container to create an averaged
  // nodal normal field on the master side for the smoothed cpp
  // normal field!
  if (interface_params().get<bool>("CPP_NORMALS") || nonSmoothContact_)
  {
    const std::shared_ptr<Core::LinAlg::Map> masternodes =
        Core::LinAlg::allreduce_e_map(*(master_row_nodes()));

    for (int i = 0; i < masternodes->num_my_elements(); ++i)
    {
      int gid = masternodes->gid(i);
      Core::Nodes::Node* node = discret().g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid {}", gid);
      auto* mnode = dynamic_cast<CONTACT::Node*>(node);
      mnode->initialize_data_container();
    }
  }
}


/*----------------------------------------------------------------------*
 |  initialize / reset interface for contact                  popp 01/08|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::initialize()
{
  // loop over all nodes to reset stuff (fully overlapping column map)
  // (use fully overlapping column map)

  for (int i = 0; i < idiscret_->num_my_col_nodes(); ++i)
  {
    auto* node = dynamic_cast<CONTACT::Node*>(idiscret_->l_col_node(i));

    // reset feasible projection and segmentation status
    node->has_proj() = false;
    node->has_segment() = false;
  }

  // init normal data in master node data container for cpp calculation
  if (interface_params().get<bool>("CPP_NORMALS"))
  {
    for (int i = 0; i < master_col_nodes()->num_my_elements(); ++i)
    {
      int gid = master_col_nodes()->gid(i);
      Core::Nodes::Node* node = discret().g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      Node* cnode = dynamic_cast<Node*>(node);

      // reset derivative maps of normal vector
      for (auto& j : cnode->data().get_deriv_n())
      {
        j.clear();
      }
      (cnode->data().get_deriv_n()).resize(0, 0);

      // reset derivative maps of tangent vectors
      for (auto& j : cnode->data().get_deriv_txi())
      {
        j.clear();
      }
      (cnode->data().get_deriv_txi()).resize(0, 0);
      for (auto& j : cnode->data().get_deriv_teta())
      {
        j.clear();
      }
      (cnode->data().get_deriv_teta()).resize(0, 0);

      for (auto& j : cnode->data().get_deriv_tangent())
      {
        j.clear();
      }
      (cnode->data().get_deriv_tangent()).resize(0, 0);
    }
  }

  // loop over all slave nodes to reset stuff (standard column map)
  // (include slave side boundary nodes / crosspoints)
  for (int i = 0; i < slave_col_nodes_bound()->num_my_elements(); ++i)
  {
    int gid = slave_col_nodes_bound()->gid(i);
    Core::Nodes::Node* node = discret().g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Node* cnode = dynamic_cast<Node*>(node);

    // reset nodal Mortar maps
    // for sts
    cnode->mo_data().get_d().clear();
    cnode->mo_data().get_m().clear();
    cnode->mo_data().get_mmod().clear();
    // for nts
    cnode->mo_data().get_dnts().clear();
    cnode->mo_data().get_mnts().clear();
    // for lts
    cnode->mo_data().get_dlts().clear();
    cnode->mo_data().get_mlts().clear();
    // for ltl
    cnode->mo_data().get_dltl().clear();
    cnode->mo_data().get_mltl().clear();

    // reset derivative maps of normal vector
    for (auto& j : cnode->data().get_deriv_n())
    {
      j.clear();
    }
    (cnode->data().get_deriv_n()).resize(0, 0);

    // reset derivative maps of tangent vectors
    for (auto& j : cnode->data().get_deriv_txi())
    {
      j.clear();
    }
    (cnode->data().get_deriv_txi()).resize(0, 0);
    for (auto& j : cnode->data().get_deriv_teta())
    {
      j.clear();
    }
    (cnode->data().get_deriv_teta()).resize(0, 0);

    // reset derivative map of Mortar matrices
    (cnode->data().get_deriv_d()).clear();
    (cnode->data().get_deriv_dlts()).clear();
    (cnode->data().get_deriv_dltl()).clear();
    (cnode->data().get_deriv_m()).clear();
    (cnode->data().get_deriv_mnts()).clear();
    (cnode->data().get_deriv_mlts()).clear();
    (cnode->data().get_deriv_mltl()).clear();

    // reset nodal weighted gap and derivative
    cnode->data().getg() = 1.0e12;
    cnode->data().getgnts() = 1.0e12;
    cnode->data().getglts() = 1.0e12;
    cnode->data().getgltl()[0] = 1.0e12;
    cnode->data().getgltl()[1] = 1.0e12;
    cnode->data().getgltl()[2] = 1.0e12;
    cnode->mo_data().get_dscale() = 0.0;
    (cnode->data().get_deriv_g()).clear();
    (cnode->data().get_deriv_gnts()).clear();
    (cnode->data().get_deriv_glts()).clear();
    for (auto& j : cnode->data().get_deriv_gltl())
    {
      j.clear();
    }
    for (auto& j : cnode->data().get_deriv_jumpltl())
    {
      j.clear();
    }

    // reset nodal jump
    cnode->data().getjumpltl()[0] = 1.0e12;
    cnode->data().getjumpltl()[1] = 1.0e12;
    cnode->data().getjumpltl()[2] = 1.0e12;

    // hybrid formulation
    cnode->data().get_alpha_n() = -1.0;
    cnode->data().get_alpha().clear();

    // reset derivative map of lagrange multipliers
    for (auto& j : cnode->data().get_deriv_z())
    {
      j.clear();
    }
    (cnode->data().get_deriv_z()).resize(0);

    if (friction_)
    {
      auto* frinode = dynamic_cast<FriNode*>(cnode);

      // reset SNodes and Mnodes
      frinode->fri_data().get_s_nodes().clear();
      frinode->fri_data().get_m_nodes().clear();

      // for gp slip
      if (interface_params().get<bool>("GP_SLIP_INCR"))
      {
        // reset jump deriv.
        for (auto& j : frinode->fri_data().get_deriv_var_jump())
        {
          j.clear();
        }

        (frinode->fri_data().get_deriv_var_jump()).resize(2);

        // reset jumps
        frinode->fri_data().jump_var()[0] = 0.0;
        frinode->fri_data().jump_var()[1] = 0.0;
      }
    }

    // just do poro contact relevant stuff!
    if (interface_data().is_poro())
    {
      cnode->poro_data().getn_coup() = 0.0;
      cnode->poro_data().get_derivn_coup().clear();
      cnode->poro_data().get_vel_derivn_coup().clear();
      cnode->poro_data().get_pres_derivn_coup().clear();
    }

    // just do ehl relevant stuff!
    if (ehl_) cnode->ehl_data().clear();
  }

  //**********************************************************************
  // In general, it is sufficient to reset search candidates only for
  // all elements in the standard slave column map. However, self contact
  // is an exception here, and we need to reset the search candidates of
  // all slave elements in the fully overlapping column map there. This
  // is due to the fact that self contact search is NOT parallelized.
  //**********************************************************************
  if (self_contact())
  {
    // loop over all elements to reset candidates / search lists
    // (use fully overlapping column map of S+M elements)
    for (int i = 0; i < idiscret_->num_my_col_elements(); ++i)
    {
      Core::Elements::Element* ele = idiscret_->l_col_element(i);
      auto* mele = dynamic_cast<Mortar::Element*>(ele);

      mele->mo_data().search_elements().resize(0);

      // dual shape function coefficient matrix
      mele->mo_data().reset_dual_shape();
      mele->mo_data().reset_deriv_dual_shape();
    }
  }
  else
  {
    // loop over all elements to reset candidates / search lists
    // (use standard slave column map)
    for (int i = 0; i < slave_col_elements()->num_my_elements(); ++i)
    {
      int gid = slave_col_elements()->gid(i);
      Core::Elements::Element* ele = discret().g_element(gid);
      if (!ele) FOUR_C_THROW("Cannot find ele with gid {}", gid);
      auto* mele = dynamic_cast<Mortar::Element*>(ele);

      mele->mo_data().search_elements().resize(0);

      // dual shape function coefficient matrix
      mele->mo_data().reset_dual_shape();
      mele->mo_data().reset_deriv_dual_shape();
    }
  }

  // clear all Nitsche data
  if (Teuchos::getIntegralValue<Inpar::Mortar::AlgorithmType>(imortar_, "ALGORITHM") ==
      Inpar::Mortar::algorithm_gpts)
  {
    for (int e = 0; e < discret().element_col_map()->num_my_elements(); ++e)
    {
      dynamic_cast<Mortar::Element*>(discret().g_element(discret().element_col_map()->gid(e)))
          ->get_nitsche_container()
          .clear();
    }
  }

  // reset s/m pairs and intcell counters
  smpairs_ = 0;
  smintpairs_ = 0;
  intcells_ = 0;
}

/*----------------------------------------------------------------------*
 |  compute element areas (public)                            popp 11/07|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::set_element_areas()
{
  //**********************************************************************
  // In general, it is sufficient to compute element areas only for
  // all elements in the standard slave column map. However, self contact
  // is an exception here, and we need the element areas of all elements
  // (slave and master) in the fully overlapping column map there. At the
  // same time we initialize the element data containers for self contact.
  // This is due to the fact that self contact search is NOT parallelized.
  //**********************************************************************
  if (self_contact() or interface_params().get<bool>("CPP_NORMALS") or nonSmoothContact_)
  {
    // loop over all elements to set current element length / area
    // (use fully overlapping column map)
    for (int i = 0; i < idiscret_->num_my_col_elements(); ++i)
    {
      auto* element = dynamic_cast<Mortar::Element*>(idiscret_->l_col_element(i));
      element->initialize_data_container();
      element->mo_data().area() = element->compute_area();
    }
  }
  else
  {
    // refer call back to base class version
    Mortar::Interface::set_element_areas();
  }
}


/*----------------------------------------------------------------------*
 |  pre evaluate to calc normals                            farah 02/16 |
 *----------------------------------------------------------------------*/
void CONTACT::Interface::pre_evaluate(const int& step, const int& iter)
{
  TEUCHOS_FUNC_TIME_MONITOR("CONTACT::Interface::pre_evaluate");

  //**********************************************************************
  // search algorithm
  //**********************************************************************
  if (search_alg() == Inpar::Mortar::search_bfele)
    evaluate_search_brute_force(search_param());
  else if (search_alg() == Inpar::Mortar::search_binarytree)
    evaluate_search_binarytree();
  else
    FOUR_C_THROW("Invalid search algorithm");

  // set global vector of cn values
  set_cn_ct_values(iter);

  // cpp normals or averaged normal field?
  if (interface_params().get<bool>("CPP_NORMALS"))
  {
    // evaluate cpp nodal normals on slave side
    evaluate_cpp_normals();
  }
  else
  {
    // evaluate averaged nodal normals on slave side
    evaluate_nodal_normals();

    // export nodal normals to slave node column map
    // this call is very expensive and the computation
    // time scales directly with the proc number !
    export_nodal_normals();
  }

  // set condition specific parameters needed for evaluation
  set_condition_specific_parameters();
}


/*----------------------------------------------------------------------*
 |  store nts values into sts data container for assembly   farah 07/16 |
 *----------------------------------------------------------------------*/
void CONTACT::Interface::store_nt_svalues()
{
  // loop over all possibly non-smooth nodes
  for (int i = 0; i < slave_row_nodes()->num_my_elements(); ++i)
  {
    int gid = slave_row_nodes()->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Node* cnode = dynamic_cast<Node*>(node);

    // check if integration is done
    if (cnode->mo_data().get_dnts().size() < 1) continue;

    // if nonsmooth contact is activated and the node is no corner node continue
    // if non-smooth contact is not activated go on
    if (!cnode->is_on_corner() and nonSmoothContact_) continue;

    //-------------------------------------------------------------------------------------
    // store D matrix entries
    // resize Pairedvector to nts size
    if ((int)cnode->mo_data().get_d().size() == 0)
    {
      cnode->mo_data().get_d().resize(cnode->mo_data().get_dnts().size());
    }

    for (auto p = cnode->mo_data().get_dnts().begin(); p != cnode->mo_data().get_dnts().end(); ++p)
    {
      cnode->mo_data().get_d()[p->first] += (p->second);
    }

    //-------------------------------------------------------------------------------------
    // store M matrix entries
    for (auto p = cnode->mo_data().get_mnts().begin(); p != cnode->mo_data().get_mnts().end(); ++p)
    {
      cnode->mo_data().get_m()[p->first] += (p->second);
    }

    //-------------------------------------------------------------------------------------
    // store weighted gap
    cnode->data().getg() = cnode->data().getgnts();

    //-------------------------------------------------------------------------------------
    // store weighted gap linearization
    for (auto p = cnode->data().get_deriv_gnts().begin(); p != cnode->data().get_deriv_gnts().end();
        ++p)
    {
      cnode->data().get_deriv_g()[p->first] += (p->second);
    }

    //-------------------------------------------------------------------------------------
    // store D deriv
    // --> No D linearization!

    //-------------------------------------------------------------------------------------
    // store M deriv
    {
      // Mortar M derivatives
      std::map<int, std::map<int, double>>& mntsderiv = cnode->data().get_deriv_mnts();

      // get sizes and iterator start
      int mastersize = (int)mntsderiv.size();
      auto mntscurr = mntsderiv.begin();

      /********************************************** LinMMatrix **********/
      // loop over all master nodes in the DerivM-map of the current LM slave node
      for (int l = 0; l < mastersize; ++l)
      {
        int mgid = mntscurr->first;
        ++mntscurr;

        // Mortar matrix M derivatives
        std::map<int, double>& thismderivnts = cnode->data().get_deriv_mnts()[mgid];
        std::map<int, double>& thismderivmortar = cnode->data().get_deriv_m()[mgid];

        int mapsize = (int)(thismderivnts.size());

        auto mcolcurr = thismderivnts.begin();

        // loop over all directional derivative entries
        for (int c = 0; c < mapsize; ++c)
        {
          thismderivmortar[mcolcurr->first] += (mcolcurr->second);
          ++mcolcurr;
        }

        // check for completeness of DerivM-Derivatives-iteration
        if (mcolcurr != thismderivnts.end())
          FOUR_C_THROW("StoreNTS: Not all derivative entries of DerivM considered!");
      }
    }
  }  // end node loop
}


/*----------------------------------------------------------------------*
 |  store lts values into sts data container for assembly   farah 07/16 |
 *----------------------------------------------------------------------*/
void CONTACT::Interface::store_lt_svalues()
{
  // loop over all possibly non-smooth nodes
  for (int i = 0; i < slave_row_nodes()->num_my_elements(); ++i)
  {
    double msum = 0.0;
    double ssum = 0.0;

    int gid = slave_row_nodes()->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Node* cnode = dynamic_cast<Node*>(node);

    // check if this is an edge or a corner and nonsmooth contact is activated
    if (!cnode->is_on_edge() and nonSmoothContact_) continue;
    if (cnode->is_on_corner() and nonSmoothContact_) continue;

    // check if integration is done
    if (cnode->mo_data().get_dlts().size() < 1) continue;

    //-------------------------------------------------------------------------------------
    // store D matrix entries
    // resize Pairedvector to nts size
    if ((int)cnode->mo_data().get_d().size() == 0)
      cnode->mo_data().get_d().resize(
          cnode->mo_data().get_dlts().size() + cnode->mo_data().get_dltl().size());

    for (auto p = cnode->mo_data().get_dlts().begin(); p != cnode->mo_data().get_dlts().end(); ++p)
    {
      cnode->mo_data().get_d()[p->first] += (p->second);
      ssum += (p->second);
    }

    //-------------------------------------------------------------------------------------
    // store M matrix entries
    for (auto p = cnode->mo_data().get_mlts().begin(); p != cnode->mo_data().get_mlts().end(); ++p)
    {
      cnode->mo_data().get_m()[p->first] += (p->second);
      msum += (p->second);
    }

    //-------------------------------------------------------------------------------------
    // store weighted gap
    cnode->data().getg() = cnode->data().getglts();

    //-------------------------------------------------------------------------------------
    // store weighted gap linearization
    for (auto p = cnode->data().get_deriv_glts().begin(); p != cnode->data().get_deriv_glts().end();
        ++p)
      cnode->data().get_deriv_g()[p->first] += (p->second);

    //-------------------------------------------------------------------------------------
    // store D deriv
    {
      // Mortar M derivatives
      std::map<int, std::map<int, double>>& mntsderiv = cnode->data().get_deriv_dlts();

      // get sizes and iterator start
      int mastersize = (int)mntsderiv.size();
      auto mntscurr = mntsderiv.begin();

      /********************************************** LinMMatrix **********/
      // loop over all master nodes in the DerivM-map of the current LM slave node
      for (int l = 0; l < mastersize; ++l)
      {
        int mgid = mntscurr->first;
        ++mntscurr;

        // Mortar matrix M derivatives
        std::map<int, double>& thismderivnts = cnode->data().get_deriv_dlts()[mgid];
        std::map<int, double>& thismderivmortar = cnode->data().get_deriv_d()[mgid];

        int mapsize = (int)(thismderivnts.size());

        auto mcolcurr = thismderivnts.begin();

        // loop over all directional derivative entries
        for (int c = 0; c < mapsize; ++c)
        {
          thismderivmortar[mcolcurr->first] += (mcolcurr->second);
          ++mcolcurr;
        }

        // check for completeness of DerivM-Derivatives-iteration
        if (mcolcurr != thismderivnts.end())
          FOUR_C_THROW("StoreNTS: Not all derivative entries of DerivM considered!");
      }
    }

    //-------------------------------------------------------------------------------------
    // store M deriv
    {
      // Mortar M derivatives
      std::map<int, std::map<int, double>>& mntsderiv = cnode->data().get_deriv_mlts();

      // get sizes and iterator start
      int mastersize = (int)mntsderiv.size();
      auto mntscurr = mntsderiv.begin();

      /********************************************** LinMMatrix **********/
      // loop over all master nodes in the DerivM-map of the current LM slave node
      for (int l = 0; l < mastersize; ++l)
      {
        int mgid = mntscurr->first;
        ++mntscurr;

        // Mortar matrix M derivatives
        std::map<int, double>& thismderivnts = cnode->data().get_deriv_mlts()[mgid];
        std::map<int, double>& thismderivmortar = cnode->data().get_deriv_m()[mgid];

        int mapsize = (int)(thismderivnts.size());

        auto mcolcurr = thismderivnts.begin();

        // loop over all directional derivative entries
        for (int c = 0; c < mapsize; ++c)
        {
          thismderivmortar[mcolcurr->first] += (mcolcurr->second);
          ++mcolcurr;
        }

        // check for completeness of DerivM-Derivatives-iteration
        if (mcolcurr != thismderivnts.end())
          FOUR_C_THROW("StoreNTS: Not all derivative entries of DerivM considered!");
      }
    }
    //    std::cout << "ssum = " << ssum << "  msum = " << msum << "  balance= " << ssum-msum <<
    //    std::endl;
    if (abs(ssum - msum) > 1e-12) FOUR_C_THROW("no slave master balance!");

  }  // end node loop
}

/*----------------------------------------------------------------------*
 |  Add line to line penalty forces                         farah 10/16 |
 *----------------------------------------------------------------------*/
void CONTACT::Interface::add_ltl_forces_friction(Core::LinAlg::FEVector<double>& feff)
{
  const double penalty = interface_params().get<double>("PENALTYPARAM");
  const double penaltytan = interface_params().get<double>("PENALTYPARAMTAN");
  const double frcoeff = interface_params().get<double>("FRCOEFF");

  std::array<double, 3> oldtraction = {0.0, 0.0, 0.0};

  // loop over all slave nodes
  for (int j = 0; j < snoderowmap_->num_my_elements(); ++j)
  {
    int gid = snoderowmap_->gid(j);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    auto* cnode = dynamic_cast<FriNode*>(node);

    double x = cnode->fri_data().tractionold_ltl()[0] * cnode->fri_data().tractionold_ltl()[0];
    double y = cnode->fri_data().tractionold_ltl()[1] * cnode->fri_data().tractionold_ltl()[1];
    double z = cnode->fri_data().tractionold_ltl()[2] * cnode->fri_data().tractionold_ltl()[2];
    double tracvalue = sqrt(x + y + z);

    if (tracvalue > 1e-8)
    {
      oldtraction[0] = cnode->fri_data().tractionold_ltl()[0];
      oldtraction[1] = cnode->fri_data().tractionold_ltl()[1];
      oldtraction[2] = cnode->fri_data().tractionold_ltl()[2];
      break;
    }
  }

  // maybe the old traction is here zero (first contact step)
  // loop over all slave nodes
  for (int j = 0; j < snoderowmap_->num_my_elements(); ++j)
  {
    int gid = snoderowmap_->gid(j);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    auto* cnode = dynamic_cast<FriNode*>(node);

    // check if this is active node
    if (cnode->data().getgltl()[0] < 1e8 and cnode->data().getgltl()[1] < 1e8 and
        cnode->data().getgltl()[2] < 1e8)
    {
      // normal force
      std::array<double, 3> fn = {0.0, 0.0, 0.0};
      for (int dim = 0; dim < n_dim(); ++dim) fn[dim] = -penalty * cnode->data().getgltl()[dim];

      // f trial tangential
      std::array<double, 3> ftrial = {0.0, 0.0, 0.0};
      for (int dim = 0; dim < n_dim(); ++dim)
        ftrial[dim] = oldtraction[dim] - penaltytan * cnode->data().getjumpltl()[dim];

      // trial norm
      double trialnorm =
          sqrt(ftrial[0] * ftrial[0] + ftrial[1] * ftrial[1] + ftrial[2] * ftrial[2]);

      // maxtrac
      double maxtrac = sqrt(fn[0] * fn[0] + fn[1] * fn[1] + fn[2] * fn[2]);

      // real traction
      std::array<double, 3> ftan = {0.0, 0.0, 0.0};

      if (trialnorm - frcoeff * maxtrac <= 0.0)
      {
        for (int dim = 0; dim < n_dim(); ++dim) ftan[dim] = ftrial[dim];
      }
      else
      {
        double coeff = frcoeff * maxtrac / trialnorm;
        for (int dim = 0; dim < n_dim(); ++dim) ftan[dim] = coeff * ftrial[dim];
      }

      // store
      cnode->fri_data().traction()[0] = ftan[0];
      cnode->fri_data().traction()[1] = ftan[1];
      cnode->fri_data().traction()[2] = ftan[2];

      // ASSEMBLE
      /**************************************************** D-matrix ******/
      if ((cnode->mo_data().get_dltl()).size() > 0)
      {
        Core::Gen::Pairedvector<int, double> map = cnode->mo_data().get_dltl();

        for (const auto& p : map)
        {
          // node id
          int gid3 = p.first;
          Core::Nodes::Node* snode = idiscret_->g_node(gid3);
          if (!snode) FOUR_C_THROW("Cannot find node with gid");
          Node* csnode = dynamic_cast<Node*>(snode);

          for (int dim = 0; dim < n_dim(); ++dim)
          {
            double value = (p.second) * ftan[dim];
            const int ltlid = csnode->dofs()[dim];
            feff.sum_into_global_values(1, &ltlid, &value);
          }
        }
      }
      else
      {
        FOUR_C_THROW("no d matrix entries available for ltlt contact");
      }

      /**************************************************** M-matrix ******/
      if ((cnode->mo_data().get_mltl()).size() > 0)
      {
        std::map<int, double> map = cnode->mo_data().get_mltl();

        for (auto& p : map)
        {
          // node id
          int gid3 = p.first;
          Core::Nodes::Node* snode = idiscret_->g_node(gid3);
          if (!snode) FOUR_C_THROW("Cannot find node with gid");
          Node* csnode = dynamic_cast<Node*>(snode);

          for (int dim = 0; dim < n_dim(); ++dim)
          {
            double value = -(p.second) * ftan[dim];
            const int ltlid = csnode->dofs()[dim];
            feff.sum_into_global_values(1, &ltlid, &value);
          }
        }
      }
      else
      {
        FOUR_C_THROW("no m matrix entries available for ltlt contact");
      }

      break;
    }
  }
}

/*----------------------------------------------------------------------*
 |  Add line to line penalty forces                         farah 10/16 |
 *----------------------------------------------------------------------*/
void CONTACT::Interface::add_ltl_stiffness_friction(Core::LinAlg::SparseMatrix& kteff)
{
  const double penalty = interface_params().get<double>("PENALTYPARAM");
  const double penaltytan = interface_params().get<double>("PENALTYPARAMTAN");
  const double frcoeff = interface_params().get<double>("FRCOEFF");

  std::array<double, 3> oldtraction = {0.0, 0.0, 0.0};

  // loop over all slave nodes
  for (int j = 0; j < snoderowmap_->num_my_elements(); ++j)
  {
    int gid = snoderowmap_->gid(j);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    auto* cnode = dynamic_cast<FriNode*>(node);

    double x = cnode->fri_data().tractionold_ltl()[0] * cnode->fri_data().tractionold_ltl()[0];
    double y = cnode->fri_data().tractionold_ltl()[1] * cnode->fri_data().tractionold_ltl()[1];
    double z = cnode->fri_data().tractionold_ltl()[2] * cnode->fri_data().tractionold_ltl()[2];
    double tracvalue = sqrt(x + y + z);

    if (tracvalue > 1e-8)
    {
      oldtraction[0] = cnode->fri_data().tractionold_ltl()[0];
      oldtraction[1] = cnode->fri_data().tractionold_ltl()[1];
      oldtraction[2] = cnode->fri_data().tractionold_ltl()[2];
      break;
    }
  }

  // maybe the old traction is here zero (first contact step)
  // loop over all slave nodes
  for (int j = 0; j < snoderowmap_->num_my_elements(); ++j)
  {
    int gid = snoderowmap_->gid(j);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    auto* cnode = dynamic_cast<FriNode*>(node);

    // check if this is active node
    if (cnode->data().getgltl()[0] < 1e8 and cnode->data().getgltl()[1] < 1e8 and
        cnode->data().getgltl()[2] < 1e8)
    {
      // state
      bool stick = true;
      Core::Gen::Pairedvector<int, double> coefflin(100);

      // normal force
      std::array<double, 3> fn = {0.0, 0.0, 0.0};
      for (int dim = 0; dim < n_dim(); ++dim) fn[dim] = -penalty * cnode->data().getgltl()[dim];

      // f trial tangential
      std::array<double, 3> ftrial = {0.0, 0.0, 0.0};
      for (int dim = 0; dim < n_dim(); ++dim)
        ftrial[dim] = oldtraction[dim] - penaltytan * cnode->data().getjumpltl()[dim];

      double coeff = 0.0;

      // trial norm
      double trialnorm =
          sqrt(ftrial[0] * ftrial[0] + ftrial[1] * ftrial[1] + ftrial[2] * ftrial[2]);

      // maxtrac
      double maxtrac = sqrt(fn[0] * fn[0] + fn[1] * fn[1] + fn[2] * fn[2]);

      // real traction
      std::array<double, 3> ftan = {0.0, 0.0, 0.0};

      if (trialnorm - frcoeff * maxtrac <= 0.0)
      {
        stick = true;
        for (int dim = 0; dim < n_dim(); ++dim) ftan[dim] = ftrial[dim];
      }
      else
      {
        stick = false;
        coeff = frcoeff * maxtrac / trialnorm;
        for (int dim = 0; dim < n_dim(); ++dim) ftan[dim] = coeff * ftrial[dim];

        Core::Gen::Pairedvector<int, double> fn_x(100);
        Core::Gen::Pairedvector<int, double> fn_y(100);
        Core::Gen::Pairedvector<int, double> fn_z(100);

        Core::Gen::Pairedvector<int, double> ft_x(100);
        Core::Gen::Pairedvector<int, double> ft_y(100);
        Core::Gen::Pairedvector<int, double> ft_z(100);

        for (auto& pp : cnode->data().get_deriv_gltl()[0])
        {
          fn_x[pp.first] -= penalty * (pp.second);
        }
        for (auto& pp : cnode->data().get_deriv_gltl()[1])
        {
          fn_y[pp.first] -= penalty * (pp.second);
        }
        for (auto& pp : cnode->data().get_deriv_gltl()[2])
        {
          fn_z[pp.first] -= penalty * (pp.second);
        }

        for (auto& pp : cnode->data().get_deriv_jumpltl()[0])
        {
          ft_x[pp.first] -= penaltytan * (pp.second);
        }
        for (auto& pp : cnode->data().get_deriv_jumpltl()[1])
        {
          ft_y[pp.first] -= penaltytan * (pp.second);
        }
        for (auto& pp : cnode->data().get_deriv_jumpltl()[2])
        {
          ft_z[pp.first] -= penaltytan * (pp.second);
        }

        Core::Gen::Pairedvector<int, double> maxtraclin(100);
        for (const auto& pp : fn_x)
        {
          maxtraclin[pp.first] -= 0.5 * (1.0 / maxtrac) * (pp.second) * 2.0 * fn[0] * pp.second;
        }
        for (const auto& pp : fn_y)
        {
          maxtraclin[pp.first] -= 0.5 * (1.0 / maxtrac) * (pp.second) * 2.0 * fn[1] * pp.second;
        }
        for (const auto& pp : fn_z)
        {
          maxtraclin[pp.first] -= 0.5 * (1.0 / maxtrac) * (pp.second) * 2.0 * fn[2] * pp.second;
        }

        Core::Gen::Pairedvector<int, double> trialnormlin(100);
        for (const auto& pp : ft_x)
        {
          trialnormlin[pp.first] -=
              0.5 * (1.0 / trialnorm) * (pp.second) * 2.0 * ftrial[0] * pp.second;
        }
        for (const auto& pp : ft_y)
        {
          trialnormlin[pp.first] -=
              0.5 * (1.0 / trialnorm) * (pp.second) * 2.0 * ftrial[1] * pp.second;
        }
        for (const auto& pp : ft_z)
        {
          trialnormlin[pp.first] -=
              0.5 * (1.0 / trialnorm) * (pp.second) * 2.0 * ftrial[2] * pp.second;
        }

        for (const auto& pp : maxtraclin)
        {
          coefflin[pp.first] += frcoeff * pp.second * (1.0 / trialnorm);
        }

        for (const auto& pp : trialnormlin)
        {
          coefflin[pp.first] -= frcoeff * pp.second * (1.0 / (trialnorm * trialnorm)) * maxtrac;
        }
      }


      std::map<int, std::map<int, double>>& dderiv = cnode->data().get_deriv_dltl();

      // get sizes and iterator start
      int slavesize = (int)dderiv.size();
      auto scurr = dderiv.begin();

      /********************************************** LinDMatrix **********/
      // loop over all DISP slave nodes in the DerivD-map of the current LM slave node
      for (int k = 0; k < slavesize; ++k)
      {
        int sgid = scurr->first;
        ++scurr;

        Core::Nodes::Node* snode = idiscret_->g_node(sgid);
        if (!snode) FOUR_C_THROW("Cannot find node with gid %", sgid);
        Node* csnode = dynamic_cast<Node*>(snode);

        // Mortar matrix D derivatives
        std::map<int, double>& thisdderiv = cnode->data().get_deriv_dltl()[sgid];
        int mapsize = (int)(thisdderiv.size());

        // inner product D_{jk,c} * z_j for index j
        for (int prodj = 0; prodj < n_dim(); ++prodj)
        {
          int row = csnode->dofs()[prodj];
          auto scolcurr = thisdderiv.begin();

          // loop over all directional derivative entries
          for (int c = 0; c < mapsize; ++c)
          {
            int col = scolcurr->first;
            double val = ftan[prodj] * (scolcurr->second);
            ++scolcurr;

            kteff.fe_assemble(val, row, col);
          }

          // check for completeness of DerivD-Derivatives-iteration
          if (scolcurr != thisdderiv.end())
            FOUR_C_THROW("AssembleLinDM: Not all derivative entries of DerivD considered!");
        }
      }

      // Mortar matrix D and M derivatives
      std::map<int, std::map<int, double>>& mderiv = cnode->data().get_deriv_mltl();

      // get sizes and iterator start
      int mastersize = (int)mderiv.size();
      auto mcurr = mderiv.begin();

      /********************************************** LinMMatrix **********/
      // loop over all master nodes in the DerivM-map of the current LM slave node
      for (int l = 0; l < mastersize; ++l)
      {
        int mgid = mcurr->first;
        ++mcurr;

        Core::Nodes::Node* mnode = idiscret_->g_node(mgid);
        if (!mnode) FOUR_C_THROW("Cannot find node with gid %", mgid);
        Node* cmnode = dynamic_cast<Node*>(mnode);

        // Mortar matrix M derivatives
        std::map<int, double>& thismderiv = cnode->data().get_deriv_mltl()[mgid];
        int mapsize = (int)(thismderiv.size());

        // inner product M_{jl,c} * z_j for index j
        for (int prodj = 0; prodj < n_dim(); ++prodj)
        {
          int row = cmnode->dofs()[prodj];
          auto mcolcurr = thismderiv.begin();

          // loop over all directional derivative entries
          for (int c = 0; c < mapsize; ++c)
          {
            int col = mcolcurr->first;
            double val = ftan[prodj] * (mcolcurr->second);
            ++mcolcurr;

            kteff.fe_assemble(-val, row, col);
          }

          // check for completeness of DerivM-Derivatives-iteration
          if (mcolcurr != thismderiv.end())
            FOUR_C_THROW("AssembleLinDM: Not all derivative entries of DerivM considered!");
        }
      }

      // ****************************************************************
      // **************************************************** stick state
      // ****************************************************************
      if (stick)
      {
        /**************************************************** D-matrix ******/
        if ((cnode->mo_data().get_dltl()).size() > 0)
        {
          Core::Gen::Pairedvector<int, double> map = cnode->mo_data().get_dltl();

          for (const auto& p : map)
          {
            // node id
            int gid3 = p.first;
            Core::Nodes::Node* snode = idiscret_->g_node(gid3);
            if (!snode) FOUR_C_THROW("Cannot find node with gid");
            Node* csnode = dynamic_cast<Node*>(snode);

            for (int dim = 0; dim < n_dim(); ++dim)
            {
              for (auto pp = cnode->data().get_deriv_jumpltl()[dim].begin();
                  pp != cnode->data().get_deriv_jumpltl()[dim].end(); ++pp)
              {
                double value = penaltytan * (p.second) * (pp->second);
                kteff.fe_assemble(value, csnode->dofs()[dim], pp->first);
              }
            }
          }
        }
        else
        {
          FOUR_C_THROW("no d matrix entries available for ltlt contact");
        }
        /**************************************************** D-matrix ******/
        if ((cnode->mo_data().get_mltl()).size() > 0)
        {
          std::map<int, double> map = cnode->mo_data().get_mltl();

          for (auto& p : map)
          {
            // node id
            int gid3 = p.first;
            Core::Nodes::Node* snode = idiscret_->g_node(gid3);
            if (!snode) FOUR_C_THROW("Cannot find node with gid");
            Node* csnode = dynamic_cast<Node*>(snode);

            for (int dim = 0; dim < n_dim(); ++dim)
            {
              for (auto pp = cnode->data().get_deriv_jumpltl()[dim].begin();
                  pp != cnode->data().get_deriv_jumpltl()[dim].end(); ++pp)
              {
                double value = -penaltytan * (p.second) * (pp->second);
                kteff.fe_assemble(value, csnode->dofs()[dim], pp->first);
              }
            }
          }
        }
        else
        {
          FOUR_C_THROW("no m matrix entries available for ltlt contact");
        }
      }
      // ****************************************************************
      // **************************************************** slip state
      // ****************************************************************
      else
      {
        /**************************************************** D-matrix ******/
        if ((cnode->mo_data().get_dltl()).size() > 0)
        {
          Core::Gen::Pairedvector<int, double> map = cnode->mo_data().get_dltl();

          for (const auto& p : map)
          {
            // node id
            int gid3 = p.first;
            Core::Nodes::Node* snode = idiscret_->g_node(gid3);
            if (!snode) FOUR_C_THROW("Cannot find node with gid");
            Node* csnode = dynamic_cast<Node*>(snode);

            for (int dim = 0; dim < n_dim(); ++dim)
            {
              for (auto pp = cnode->data().get_deriv_jumpltl()[dim].begin();
                  pp != cnode->data().get_deriv_jumpltl()[dim].end(); ++pp)
              {
                double value = penaltytan * coeff * (p.second) * (pp->second);
                kteff.fe_assemble(value, csnode->dofs()[dim], pp->first);
              }
            }
          }
        }
        else
        {
          FOUR_C_THROW("no d matrix entries available for ltlt contact");
        }

        /**************************************************** D-matrix ******/
        if ((cnode->mo_data().get_dltl()).size() > 0)
        {
          Core::Gen::Pairedvector<int, double> map = cnode->mo_data().get_dltl();

          for (const auto& p : map)
          {
            // node id
            int gid3 = p.first;
            Core::Nodes::Node* snode = idiscret_->g_node(gid3);
            if (!snode) FOUR_C_THROW("Cannot find node with gid");
            Node* csnode = dynamic_cast<Node*>(snode);

            for (int dim = 0; dim < n_dim(); ++dim)
            {
              for (const auto& pp : coefflin)
              {
                double value = -penaltytan * ftan[dim] * (p.second) * (pp.second);
                kteff.fe_assemble(value, csnode->dofs()[dim], pp.first);
              }
            }
          }
        }
        else
        {
          FOUR_C_THROW("no d matrix entries available for ltlt contact");
        }

        /**************************************************** D-matrix ******/
        if ((cnode->mo_data().get_mltl()).size() > 0)
        {
          std::map<int, double> map = cnode->mo_data().get_mltl();

          for (auto& p : map)
          {
            // node id
            int gid3 = p.first;
            Core::Nodes::Node* snode = idiscret_->g_node(gid3);
            if (!snode) FOUR_C_THROW("Cannot find node with gid");
            Node* csnode = dynamic_cast<Node*>(snode);

            for (int dim = 0; dim < n_dim(); ++dim)
            {
              for (auto pp = cnode->data().get_deriv_jumpltl()[dim].begin();
                  pp != cnode->data().get_deriv_jumpltl()[dim].end(); ++pp)
              {
                double value = -penaltytan * coeff * (p.second) * (pp->second);
                kteff.fe_assemble(value, csnode->dofs()[dim], pp->first);
              }
            }
          }
        }
        else
        {
          FOUR_C_THROW("no m matrix entries available for ltlt contact");
        }
        if ((cnode->mo_data().get_mltl()).size() > 0)
        {
          std::map<int, double> map = cnode->mo_data().get_mltl();

          for (auto& p : map)
          {
            // node id
            int gid3 = p.first;
            Core::Nodes::Node* snode = idiscret_->g_node(gid3);
            if (!snode) FOUR_C_THROW("Cannot find node with gid");
            Node* csnode = dynamic_cast<Node*>(snode);

            for (int dim = 0; dim < n_dim(); ++dim)
            {
              for (const auto& pp : coefflin)
              {
                double value = penaltytan * ftan[dim] * (p.second) * (pp.second);
                kteff.fe_assemble(value, csnode->dofs()[dim], pp.first);
              }
            }
          }
        }
        else
        {
          FOUR_C_THROW("no m matrix entries available for ltlt contact");
        }
      }
    }
  }
}

/*----------------------------------------------------------------------*
 |  Add nts penalty forces master                           farah 11/16 |
 *----------------------------------------------------------------------*/
void CONTACT::Interface::add_nts_forces_master(Core::LinAlg::FEVector<double>& feff)
{
  const double penalty = interface_params().get<double>("PENALTYPARAM");

  // loop over all slave nodes
  for (int j = 0; j < mnoderowmap_->num_my_elements(); ++j)
  {
    int gid = mnoderowmap_->gid(j);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Node* cnode = dynamic_cast<Node*>(node);

    // only for corners
    if (!cnode->is_on_corner()) continue;

    // is gap is in contact
    if (cnode->data().getgnts() < 1e-12)
    {
      /**************************************************** D-matrix ******/
      if ((cnode->mo_data().get_dnts()).size() > 0)
      {
        Core::Gen::Pairedvector<int, double> map = cnode->mo_data().get_dnts();

        for (const auto& p : map)
        {
          // node id
          int gid3 = p.first;
          Core::Nodes::Node* snode = idiscret_->g_node(gid3);
          if (!snode) FOUR_C_THROW("Cannot find node with gid");
          Node* csnode = dynamic_cast<Node*>(snode);

          for (int dim = 0; dim < n_dim(); ++dim)
          {
            double value =
                penalty * (p.second) * cnode->data().getgnts() * cnode->mo_data().n()[dim];
            int ltlid = csnode->dofs()[dim];
            feff.sum_into_global_values(1, &ltlid, &value);
          }
        }
      }
      else
      {
        FOUR_C_THROW("no d matrix entries available for ltlt contact");
      }

      /**************************************************** M-matrix ******/
      if ((cnode->mo_data().get_mnts()).size() > 0)
      {
        std::map<int, double> map = cnode->mo_data().get_mnts();

        for (auto& p : map)
        {
          // node id
          int gid3 = p.first;
          Core::Nodes::Node* snode = idiscret_->g_node(gid3);
          if (!snode) FOUR_C_THROW("Cannot find node with gid");
          Node* csnode = dynamic_cast<Node*>(snode);

          for (int dim = 0; dim < n_dim(); ++dim)
          {
            double value =
                -penalty * (p.second) * cnode->data().getgnts() * cnode->mo_data().n()[dim];
            int ltlid = csnode->dofs()[dim];
            feff.sum_into_global_values(1, &ltlid, &value);
          }
        }
      }
      else
      {
        FOUR_C_THROW("no m matrix entries available for ltlt contact");
      }
    }
  }
}

/*----------------------------------------------------------------------*
 |  Add line to line penalty forces master                  farah 11/16 |
 *----------------------------------------------------------------------*/
void CONTACT::Interface::add_lts_forces_master(Core::LinAlg::FEVector<double>& feff)
{
  const double penalty = interface_params().get<double>("PENALTYPARAM");

  // loop over all slave nodes
  for (int j = 0; j < mnoderowmap_->num_my_elements(); ++j)
  {
    int gid = mnoderowmap_->gid(j);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Node* cnode = dynamic_cast<Node*>(node);

    // only for edges (without corner)
    if (!cnode->is_on_corner_edge()) continue;
    if (cnode->is_on_corner()) continue;

    // scale penalty
    double penaltyLts = penalty * cnode->data().kappa();

    // is gap is in contact
    if (cnode->data().getglts() < 1e-12)
    {
      /**************************************************** D-matrix ******/
      if ((cnode->mo_data().get_dlts()).size() > 0)
      {
        Core::Gen::Pairedvector<int, double> map = cnode->mo_data().get_dlts();

        for (const auto& p : map)
        {
          // node id
          int gid3 = p.first;
          Core::Nodes::Node* snode = idiscret_->g_node(gid3);
          if (!snode) FOUR_C_THROW("Cannot find node with gid");
          Node* csnode = dynamic_cast<Node*>(snode);

          for (int dim = 0; dim < n_dim(); ++dim)
          {
            double value =
                penaltyLts * (p.second) * cnode->data().getglts() * cnode->mo_data().n()[dim];
            int ltlid = csnode->dofs()[dim];
            feff.sum_into_global_values(1, &ltlid, &value);
          }
        }
      }
      else
      {
        FOUR_C_THROW("no d matrix entries available for ltlt contact");
      }

      /**************************************************** M-matrix ******/
      if ((cnode->mo_data().get_mlts()).size() > 0)
      {
        std::map<int, double> map = cnode->mo_data().get_mlts();

        for (auto& p : map)
        {
          // node id
          int gid3 = p.first;
          Core::Nodes::Node* snode = idiscret_->g_node(gid3);
          if (!snode) FOUR_C_THROW("Cannot find node with gid");
          Node* csnode = dynamic_cast<Node*>(snode);

          for (int dim = 0; dim < n_dim(); ++dim)
          {
            double value =
                -penaltyLts * (p.second) * cnode->data().getglts() * cnode->mo_data().n()[dim];
            int ltlid = csnode->dofs()[dim];
            feff.sum_into_global_values(1, &ltlid, &value);
          }
        }
      }
      else
      {
        FOUR_C_THROW("no m matrix entries available for ltlt contact");
      }
    }
  }
}

/*----------------------------------------------------------------------*
 |  Add line to line penalty forces                         farah 10/16 |
 *----------------------------------------------------------------------*/
void CONTACT::Interface::add_ltl_forces(Core::LinAlg::FEVector<double>& feff)
{
  // gap = g_n * n
  // D/M = sval/mval
  const double penalty = interface_params().get<double>("PENALTYPARAM");

  // loop over all slave nodes
  for (int j = 0; j < snoderowmap_->num_my_elements(); ++j)
  {
    int gid = snoderowmap_->gid(j);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Node* cnode = dynamic_cast<Node*>(node);

    // check if this is valid node
    if (cnode->data().getgltl()[0] < 1e8 and cnode->data().getgltl()[1] < 1e8 and
        cnode->data().getgltl()[2] < 1e8)
    {
      /**************************************************** D-matrix ******/
      if ((cnode->mo_data().get_dltl()).size() > 0)
      {
        Core::Gen::Pairedvector<int, double> map = cnode->mo_data().get_dltl();

        for (const auto& p : map)
        {
          // node id
          int gid3 = p.first;
          Core::Nodes::Node* snode = idiscret_->g_node(gid3);
          if (!snode) FOUR_C_THROW("Cannot find node with gid");
          Node* csnode = dynamic_cast<Node*>(snode);

          for (int dim = 0; dim < n_dim(); ++dim)
          {
            double value = penalty * (p.second) * cnode->data().getgltl()[dim];
            int ltlid = csnode->dofs()[dim];
            feff.sum_into_global_values(1, &ltlid, &value);
          }
        }
      }
      else
      {
        FOUR_C_THROW("no d matrix entries available for ltlt contact");
      }

      /**************************************************** M-matrix ******/
      if ((cnode->mo_data().get_mltl()).size() > 0)
      {
        std::map<int, double> map = cnode->mo_data().get_mltl();

        for (auto& p : map)
        {
          // node id
          int gid3 = p.first;
          Core::Nodes::Node* snode = idiscret_->g_node(gid3);
          if (!snode) FOUR_C_THROW("Cannot find node with gid");
          Node* csnode = dynamic_cast<Node*>(snode);

          for (int dim = 0; dim < n_dim(); ++dim)
          {
            double value = -penalty * (p.second) * cnode->data().getgltl()[dim];
            int ltlid = {csnode->dofs()[dim]};
            feff.sum_into_global_values(1, &ltlid, &value);
          }
        }
      }
      else
      {
        FOUR_C_THROW("no m matrix entries available for ltlt contact");
      }
    }
  }
}

/*----------------------------------------------------------------------*
 |  Add line to line penalty forces                         farah 11/16 |
 *----------------------------------------------------------------------*/
void CONTACT::Interface::add_lts_stiffness_master(Core::LinAlg::SparseMatrix& kteff)
{
  const double penalty = interface_params().get<double>("PENALTYPARAM");

  // loop over all slave nodes
  for (int j = 0; j < mnoderowmap_->num_my_elements(); ++j)
  {
    int gid = mnoderowmap_->gid(j);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Node* cnode = dynamic_cast<Node*>(node);

    // only for edges (without corner)
    if (!cnode->is_on_corner_edge()) continue;
    if (cnode->is_on_corner()) continue;

    // scale penalty
    double penaltyLts = penalty * cnode->data().kappa();

    // is gap is in contact
    if (cnode->data().getglts() < 1e-12)
    {
      std::array<double, 3> lm = {0.0, 0.0, 0.0};
      lm[0] = penaltyLts * cnode->data().getglts() * cnode->mo_data().n()[0];
      lm[1] = penaltyLts * cnode->data().getglts() * cnode->mo_data().n()[1];
      lm[2] = penaltyLts * cnode->data().getglts() * cnode->mo_data().n()[2];

      std::map<int, std::map<int, double>>& dderiv = cnode->data().get_deriv_dlts();

      // get sizes and iterator start
      int slavesize = (int)dderiv.size();
      auto scurr = dderiv.begin();

      /********************************************** LinDMatrix **********/
      // loop over all DISP slave nodes in the DerivD-map of the current LM slave node
      for (int k = 0; k < slavesize; ++k)
      {
        int sgid = scurr->first;
        ++scurr;

        Core::Nodes::Node* snode = idiscret_->g_node(sgid);
        if (!snode) FOUR_C_THROW("Cannot find node with gid %", sgid);
        Node* csnode = dynamic_cast<Node*>(snode);

        // Mortar matrix D derivatives
        std::map<int, double>& thisdderiv = cnode->data().get_deriv_dlts()[sgid];
        int mapsize = (int)(thisdderiv.size());

        // inner product D_{jk,c} * z_j for index j
        for (int prodj = 0; prodj < n_dim(); ++prodj)
        {
          int row = csnode->dofs()[prodj];
          auto scolcurr = thisdderiv.begin();

          // loop over all directional derivative entries
          for (int c = 0; c < mapsize; ++c)
          {
            int col = scolcurr->first;
            double val = lm[prodj] * (scolcurr->second);
            ++scolcurr;

            kteff.fe_assemble(-val, row, col);
          }

          // check for completeness of DerivD-Derivatives-iteration
          if (scolcurr != thisdderiv.end())
            FOUR_C_THROW("AssembleLinDM: Not all derivative entries of DerivD considered!");
        }
      }

      // Mortar matrix D and M derivatives
      std::map<int, std::map<int, double>>& mderiv = cnode->data().get_deriv_mlts();

      // get sizes and iterator start
      int mastersize = (int)mderiv.size();
      auto mcurr = mderiv.begin();

      /********************************************** LinMMatrix **********/
      // loop over all master nodes in the DerivM-map of the current LM slave node
      for (int l = 0; l < mastersize; ++l)
      {
        int mgid = mcurr->first;
        ++mcurr;

        Core::Nodes::Node* mnode = idiscret_->g_node(mgid);
        if (!mnode) FOUR_C_THROW("Cannot find node with gid %", mgid);
        Node* cmnode = dynamic_cast<Node*>(mnode);

        // Mortar matrix M derivatives
        std::map<int, double>& thismderiv = cnode->data().get_deriv_mlts()[mgid];
        int mapsize = (int)(thismderiv.size());

        // inner product M_{jl,c} * z_j for index j
        for (int prodj = 0; prodj < n_dim(); ++prodj)
        {
          int row = cmnode->dofs()[prodj];
          auto mcolcurr = thismderiv.begin();

          // loop over all directional derivative entries
          for (int c = 0; c < mapsize; ++c)
          {
            int col = mcolcurr->first;
            double val = lm[prodj] * (mcolcurr->second);
            ++mcolcurr;

            kteff.fe_assemble(val, row, col);
          }

          // check for completeness of DerivM-Derivatives-iteration
          if (mcolcurr != thismderiv.end())
            FOUR_C_THROW("AssembleLinDM: Not all derivative entries of DerivM considered!");
        }
      }

      /**************************************************** D-matrix ******/
      if ((cnode->mo_data().get_dlts()).size() > 0)
      {
        Core::Gen::Pairedvector<int, double> map = cnode->mo_data().get_dlts();

        for (const auto& p : map)
        {
          // node id
          int gid3 = p.first;
          Core::Nodes::Node* snode = idiscret_->g_node(gid3);
          if (!snode) FOUR_C_THROW("Cannot find node with gid");
          Node* csnode = dynamic_cast<Node*>(snode);

          for (int dim = 0; dim < n_dim(); ++dim)
          {
            // gap linearization
            for (auto pp = cnode->data().get_deriv_glts().begin();
                pp != cnode->data().get_deriv_glts().end(); ++pp)
            {
              double value = -penaltyLts * (p.second) * (pp->second) * cnode->mo_data().n()[dim];
              kteff.fe_assemble(value, csnode->dofs()[dim], pp->first);
            }
            // normal linearization
            for (auto pp = cnode->data().get_deriv_n()[dim].begin();
                pp != cnode->data().get_deriv_n()[dim].end(); ++pp)
            {
              double value = -penaltyLts * (p.second) * (pp->second) * cnode->data().getglts();
              kteff.fe_assemble(value, csnode->dofs()[dim], pp->first);
            }
          }
        }
      }
      else
      {
        FOUR_C_THROW("no d matrix entries available for ltlt contact");
      }
      /**************************************************** D-matrix ******/
      if ((cnode->mo_data().get_mlts()).size() > 0)
      {
        std::map<int, double> map = cnode->mo_data().get_mlts();

        for (auto& p : map)
        {
          // node id
          int gid3 = p.first;
          Core::Nodes::Node* snode = idiscret_->g_node(gid3);
          if (!snode) FOUR_C_THROW("Cannot find node with gid");
          Node* csnode = dynamic_cast<Node*>(snode);

          for (int dim = 0; dim < n_dim(); ++dim)
          {
            for (auto pp = cnode->data().get_deriv_glts().begin();
                pp != cnode->data().get_deriv_glts().end(); ++pp)
            {
              double value = penaltyLts * (p.second) * (pp->second) * cnode->mo_data().n()[dim];
              kteff.fe_assemble(value, csnode->dofs()[dim], pp->first);
            }
            // normal linearization
            for (auto pp = cnode->data().get_deriv_n()[dim].begin();
                pp != cnode->data().get_deriv_n()[dim].end(); ++pp)
            {
              double value = penaltyLts * (p.second) * (pp->second) * cnode->data().getglts();
              kteff.fe_assemble(value, csnode->dofs()[dim], pp->first);
            }
          }
        }
      }
      else
      {
        FOUR_C_THROW("no m matrix entries available for ltlt contact");
      }
    }
  }
}

/*----------------------------------------------------------------------*
 |  Add line to line penalty forces                         farah 11/16 |
 *----------------------------------------------------------------------*/
void CONTACT::Interface::add_nts_stiffness_master(Core::LinAlg::SparseMatrix& kteff)
{
  const double penalty = interface_params().get<double>("PENALTYPARAM");

  // loop over all slave nodes
  for (int j = 0; j < mnoderowmap_->num_my_elements(); ++j)
  {
    int gid = mnoderowmap_->gid(j);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Node* cnode = dynamic_cast<Node*>(node);

    // only for corners
    if (!cnode->is_on_corner()) continue;

    // is gap is in contact
    if (cnode->data().getgnts() < 1e-12)
    {
      std::array<double, 3> lm = {0.0, 0.0, 0.0};
      lm[0] = penalty * cnode->data().getgnts() * cnode->mo_data().n()[0];
      lm[1] = penalty * cnode->data().getgnts() * cnode->mo_data().n()[1];
      lm[2] = penalty * cnode->data().getgnts() * cnode->mo_data().n()[2];

      // Mortar matrix D and M derivatives
      std::map<int, std::map<int, double>>& mderiv = cnode->data().get_deriv_mnts();

      // get sizes and iterator start
      int mastersize = (int)mderiv.size();
      auto mcurr = mderiv.begin();

      /********************************************** LinMMatrix **********/
      // loop over all master nodes in the DerivM-map of the current LM slave node
      for (int l = 0; l < mastersize; ++l)
      {
        int mgid = mcurr->first;
        ++mcurr;

        Core::Nodes::Node* mnode = idiscret_->g_node(mgid);
        if (!mnode) FOUR_C_THROW("Cannot find node with gid %", mgid);
        Node* cmnode = dynamic_cast<Node*>(mnode);

        // Mortar matrix M derivatives
        std::map<int, double>& thismderiv = cnode->data().get_deriv_mnts()[mgid];
        int mapsize = (int)(thismderiv.size());

        // inner product M_{jl,c} * z_j for index j
        for (int prodj = 0; prodj < n_dim(); ++prodj)
        {
          int row = cmnode->dofs()[prodj];
          auto mcolcurr = thismderiv.begin();

          // loop over all directional derivative entries
          for (int c = 0; c < mapsize; ++c)
          {
            int col = mcolcurr->first;
            double val = lm[prodj] * (mcolcurr->second);
            ++mcolcurr;

            kteff.fe_assemble(val, row, col);
          }

          // check for completeness of DerivM-Derivatives-iteration
          if (mcolcurr != thismderiv.end())
            FOUR_C_THROW("AssembleLinDM: Not all derivative entries of DerivM considered!");
        }
      }

      /**************************************************** D-matrix ******/
      if ((cnode->mo_data().get_dnts()).size() > 0)
      {
        Core::Gen::Pairedvector<int, double> map = cnode->mo_data().get_dnts();

        for (const auto& p : map)
        {
          // node id
          int gid3 = p.first;
          Core::Nodes::Node* snode = idiscret_->g_node(gid3);
          if (!snode) FOUR_C_THROW("Cannot find node with gid");
          Node* csnode = dynamic_cast<Node*>(snode);

          for (int dim = 0; dim < n_dim(); ++dim)
          {
            // gap linearization
            for (auto pp = cnode->data().get_deriv_gnts().begin();
                pp != cnode->data().get_deriv_gnts().end(); ++pp)
            {
              double value = -penalty * (p.second) * (pp->second) * cnode->mo_data().n()[dim];
              kteff.fe_assemble(value, csnode->dofs()[dim], pp->first);
            }
            // normal linearization
            for (auto pp = cnode->data().get_deriv_n()[dim].begin();
                pp != cnode->data().get_deriv_n()[dim].end(); ++pp)
            {
              double value = -penalty * (p.second) * (pp->second) * cnode->data().getgnts();
              kteff.fe_assemble(value, csnode->dofs()[dim], pp->first);
            }
          }
        }
      }
      else
      {
        FOUR_C_THROW("no d matrix entries available for ltlt contact");
      }
      /**************************************************** D-matrix ******/
      if ((cnode->mo_data().get_mnts()).size() > 0)
      {
        std::map<int, double> map = cnode->mo_data().get_mnts();

        for (auto& p : map)
        {
          // node id
          int gid3 = p.first;
          Core::Nodes::Node* snode = idiscret_->g_node(gid3);
          if (!snode) FOUR_C_THROW("Cannot find node with gid");
          Node* csnode = dynamic_cast<Node*>(snode);

          for (int dim = 0; dim < n_dim(); ++dim)
          {
            for (auto pp = cnode->data().get_deriv_gnts().begin();
                pp != cnode->data().get_deriv_gnts().end(); ++pp)
            {
              double value = penalty * (p.second) * (pp->second) * cnode->mo_data().n()[dim];
              kteff.fe_assemble(value, csnode->dofs()[dim], pp->first);
            }
            // normal linearization
            for (auto pp = cnode->data().get_deriv_n()[dim].begin();
                pp != cnode->data().get_deriv_n()[dim].end(); ++pp)
            {
              double value = penalty * (p.second) * (pp->second) * cnode->data().getgnts();
              kteff.fe_assemble(value, csnode->dofs()[dim], pp->first);
            }
          }
        }
      }
      else
      {
        FOUR_C_THROW("no m matrix entries available for ltlt contact");
      }
    }
  }
}

/*----------------------------------------------------------------------*
 |  Add line to line penalty forces                         farah 10/16 |
 *----------------------------------------------------------------------*/
void CONTACT::Interface::add_ltl_stiffness(Core::LinAlg::SparseMatrix& kteff)
{
  const double penalty = interface_params().get<double>("PENALTYPARAM");

  // loop over all slave nodes
  for (int j = 0; j < snoderowmap_->num_my_elements(); ++j)
  {
    int gid = snoderowmap_->gid(j);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Node* cnode = dynamic_cast<Node*>(node);

    // check if this is valid node
    if (cnode->data().getgltl()[0] < 1e8 and cnode->data().getgltl()[1] < 1e8 and
        cnode->data().getgltl()[2] < 1e8)
    {
      std::array<double, 3> lm = {0.0, 0.0, 0.0};
      lm[0] = penalty * cnode->data().getgltl()[0];
      lm[1] = penalty * cnode->data().getgltl()[1];
      lm[2] = penalty * cnode->data().getgltl()[2];

      std::map<int, std::map<int, double>>& dderiv = cnode->data().get_deriv_dltl();

      // get sizes and iterator start
      int slavesize = (int)dderiv.size();
      auto scurr = dderiv.begin();

      /********************************************** LinDMatrix **********/
      // loop over all DISP slave nodes in the DerivD-map of the current LM slave node
      for (int k = 0; k < slavesize; ++k)
      {
        int sgid = scurr->first;
        ++scurr;

        Core::Nodes::Node* snode = idiscret_->g_node(sgid);
        if (!snode) FOUR_C_THROW("Cannot find node with gid %", sgid);
        Node* csnode = dynamic_cast<Node*>(snode);

        // Mortar matrix D derivatives
        std::map<int, double>& thisdderiv = cnode->data().get_deriv_dltl()[sgid];
        int mapsize = (int)(thisdderiv.size());

        // inner product D_{jk,c} * z_j for index j
        for (int prodj = 0; prodj < n_dim(); ++prodj)
        {
          int row = csnode->dofs()[prodj];
          auto scolcurr = thisdderiv.begin();

          // loop over all directional derivative entries
          for (int c = 0; c < mapsize; ++c)
          {
            int col = scolcurr->first;
            double val = lm[prodj] * (scolcurr->second);
            ++scolcurr;

            kteff.fe_assemble(-val, row, col);
          }

          // check for completeness of DerivD-Derivatives-iteration
          if (scolcurr != thisdderiv.end())
            FOUR_C_THROW("AssembleLinDM: Not all derivative entries of DerivD considered!");
        }
      }

      // Mortar matrix D and M derivatives
      std::map<int, std::map<int, double>>& mderiv = cnode->data().get_deriv_mltl();

      // get sizes and iterator start
      int mastersize = (int)mderiv.size();
      auto mcurr = mderiv.begin();

      /********************************************** LinMMatrix **********/
      // loop over all master nodes in the DerivM-map of the current LM slave node
      for (int l = 0; l < mastersize; ++l)
      {
        int mgid = mcurr->first;
        ++mcurr;

        Core::Nodes::Node* mnode = idiscret_->g_node(mgid);
        if (!mnode) FOUR_C_THROW("Cannot find node with gid %", mgid);
        Node* cmnode = dynamic_cast<Node*>(mnode);

        // Mortar matrix M derivatives
        std::map<int, double>& thismderiv = cnode->data().get_deriv_mltl()[mgid];
        int mapsize = (int)(thismderiv.size());

        // inner product M_{jl,c} * z_j for index j
        for (int prodj = 0; prodj < n_dim(); ++prodj)
        {
          int row = cmnode->dofs()[prodj];
          auto mcolcurr = thismderiv.begin();

          // loop over all directional derivative entries
          for (int c = 0; c < mapsize; ++c)
          {
            int col = mcolcurr->first;
            double val = lm[prodj] * (mcolcurr->second);
            ++mcolcurr;

            kteff.fe_assemble(val, row, col);
          }

          // check for completeness of DerivM-Derivatives-iteration
          if (mcolcurr != thismderiv.end())
            FOUR_C_THROW("AssembleLinDM: Not all derivative entries of DerivM considered!");
        }
      }

      /**************************************************** D-matrix ******/
      if ((cnode->mo_data().get_dltl()).size() > 0)
      {
        Core::Gen::Pairedvector<int, double> map = cnode->mo_data().get_dltl();

        for (const auto& p : map)
        {
          // node id
          int gid3 = p.first;
          Core::Nodes::Node* snode = idiscret_->g_node(gid3);
          if (!snode) FOUR_C_THROW("Cannot find node with gid");
          Node* csnode = dynamic_cast<Node*>(snode);

          for (int dim = 0; dim < n_dim(); ++dim)
          {
            for (auto pp = cnode->data().get_deriv_gltl()[dim].begin();
                pp != cnode->data().get_deriv_gltl()[dim].end(); ++pp)
            {
              double value = -penalty * (p.second) * (pp->second);
              kteff.fe_assemble(value, csnode->dofs()[dim], pp->first);
            }
          }
        }
      }
      else
      {
        FOUR_C_THROW("no d matrix entries available for ltlt contact");
      }
      /**************************************************** D-matrix ******/
      if ((cnode->mo_data().get_mltl()).size() > 0)
      {
        std::map<int, double> map = cnode->mo_data().get_mltl();

        for (auto& p : map)
        {
          // node id
          int gid3 = p.first;
          Core::Nodes::Node* snode = idiscret_->g_node(gid3);
          if (!snode) FOUR_C_THROW("Cannot find node with gid");
          Node* csnode = dynamic_cast<Node*>(snode);

          for (int dim = 0; dim < n_dim(); ++dim)
          {
            for (auto pp = cnode->data().get_deriv_gltl()[dim].begin();
                pp != cnode->data().get_deriv_gltl()[dim].end(); ++pp)
            {
              double value = penalty * (p.second) * (pp->second);
              kteff.fe_assemble(value, csnode->dofs()[dim], pp->first);
            }
          }
        }
      }
      else
      {
        FOUR_C_THROW("no m matrix entries available for ltlt contact");
      }
    }
  }
}

/*----------------------------------------------------------------------*
 |  post evaluate to scale calculated terms                 farah 02/16 |
 *----------------------------------------------------------------------*/
void CONTACT::Interface::post_evaluate(const int step, const int iter)
{
  // decide which type of coupling should be evaluated
  auto algo = Teuchos::getIntegralValue<Inpar::Mortar::AlgorithmType>(imortar_, "ALGORITHM");

  switch (algo)
  {
    //*********************************
    // Mortar Coupling (STS)    (2D/3D)
    //*********************************
    case Inpar::Mortar::algorithm_mortar:
    {
      // non-smooth contact
      if (nonSmoothContact_)
      {
        // store lts into mortar data container
        store_lt_svalues();

        // store nts into mortar data container
        store_nt_svalues();
      }
      return;
      break;
    }
    //*********************************
    // Gauss-Point-To-Segment (GPTS)
    //*********************************
    case Inpar::Mortar::algorithm_gpts:
    {
      // already stored
      return;
      break;
    }
    //*********************************
    // Line-to-Segment Coupling (3D)
    //*********************************
    case Inpar::Mortar::algorithm_lts:
    {
      // store lts into mortar data container
      store_lt_svalues();
      break;
    }
    //*********************************
    // Node-to-Segment Coupling (2D/3D)
    //*********************************
    case Inpar::Mortar::algorithm_nts:
    {
      // store nts into mortar data container
      store_nt_svalues();
      break;
    }
    //*********************************
    // line-to-line Coupling (3D)
    //*********************************
    case Inpar::Mortar::algorithm_ltl:
    {
      return;
      break;
    }
    //*********************************
    // Node-to-Line Coupling (3D)
    //*********************************
    case Inpar::Mortar::algorithm_ntl:
    {
      FOUR_C_THROW("not yet implemented!");
      break;
    }
    //*********************************
    // Segment-to-Line Coupling (3D)
    //*********************************
    case Inpar::Mortar::algorithm_stl:
    {
      // store lts into mortar data container
      store_lt_svalues();
      break;
    }
    //*********************************
    // Default case
    //*********************************
    default:
    {
      FOUR_C_THROW("Unknown discr. type for constraints!");
      break;
    }
  }
}


/*----------------------------------------------------------------------*
 |  evaluate coupling type segment-to-segment coupl          farah 02/16|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::evaluate_sts(const Core::LinAlg::Map& selecolmap,
    const std::shared_ptr<Mortar::ParamsInterface>& mparams_ptr)
{
  Mortar::Interface::evaluate_sts(selecolmap, mparams_ptr);
}

/*----------------------------------------------------------------------*
 |  protected evaluate routine                               farah 07/16|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::evaluate_coupling(const Core::LinAlg::Map& selecolmap,
    const Core::LinAlg::Map* snoderowmap,
    const std::shared_ptr<Mortar::ParamsInterface>& mparams_ptr)
{
  // ask if non-smooth contact is activated!
  if (nonSmoothContact_)
  {
    // 2D: only STS and nts has to be performed
    if (n_dim() == 2)
    {
      //********************************************************************
      // 1) perform coupling (projection + overlap detection for sl/m pairs)
      // 2) integrate Mortar matrix M and weighted gap g
      // 3) compute directional derivative of M and g and store into nodes
      //    (only for contact setting)
      //********************************************************************
      evaluate_sts(selecolmap, mparams_ptr);

      //********************************************************************
      // 1) perform coupling (find the closest point between to lines)
      // 2) evaluate gap and shape functions at this point
      // 3) compute directional derivative of entries and store into nodes
      //    (only for contact setting)
      //********************************************************************
      evaluate_nts();

      //********************************************************************
      // NTN is a special case of NTS and an additional implementation is
      // not required!
      //********************************************************************
    }
    else if (n_dim() == 3)
    {
      //********************************************************************
      // TODO: remove this hack!
      // HACK: LTL is not yet included in nonsmooth contact framework!
      // However, we want to test the LTL code separately. Thus, the "if"-
      // statement is included:
      // decide which type of coupling should be evaluated
      //********************************************************************
      auto algo = Teuchos::getIntegralValue<Inpar::Mortar::AlgorithmType>(imortar_, "ALGORITHM");
      if (algo == Inpar::Mortar::algorithm_ltl)
      {
        evaluate_ltl();
        return;
      }

      //********************************************************************
      // 1) try to project slave nodes onto master elements
      // 2) evaluate shape functions at projected positions
      // 3) compute directional derivative of M and g and store into nodes
      //********************************************************************
      evaluate_nts();

      //********************************************************************
      // 1) perform coupling (find the closest point between to lines)
      // 2) evaluate gap and shape functions at this point
      // 3) compute directional derivative of entries and store into nodes
      //********************************************************************
      evaluate_ltl();

      //********************************************************************
      // 1) perform coupling (projection + line clipping edge surface pairs)
      // 2) integrate Mortar matrices D + M and weighted gap g
      // 3) compute directional derivative of D + M and g and store into nodes
      //********************************************************************
      evaluate_lts();

      //********************************************************************
      // 1) perform coupling (projection + overlap detection for sl/m pairs)
      // 2) integrate Mortar matrix M and weighted gap g
      // 3) compute directional derivative of M and g and store into nodes
      //********************************************************************
      evaluate_sts(selecolmap, mparams_ptr);

      //********************************************************************
      // perform LTS steps for master edges
      //********************************************************************
      //      evaluate_lts_master();

      //********************************************************************
      // perform NTS steps for master edges
      //********************************************************************
      //      evaluate_nts_master();

      //********************************************************************
      // NTN is a special case of NTS and an additional implementation is
      // not required!
      //********************************************************************
    }
    else
    {
      FOUR_C_THROW("Wrong dimension!");
    }
  }
  else
  {
    //********************************************************************
    // call base routine for standard mortar/nts evaluation
    //********************************************************************
    Mortar::Interface::evaluate_coupling(selecolmap, snoderowmap, mparams_ptr);
  }
}

/*----------------------------------------------------------------------*
 |  Check and initialize corner/edge contact                 farah 07/16|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::initialize_corner_edge()
{
  // return if nonsmooth contact is activated
  if (nonSmoothContact_) return;

  // call base function
  Mortar::Interface::initialize_corner_edge();
}

/*----------------------------------------------------------------------*
 |  cpp to edge + Lin                                       farah 11/16 |
 *----------------------------------------------------------------------*/
double CONTACT::Interface::compute_normal_node_to_edge(const Mortar::Node& snode,
    const Mortar::Element& mele, double* normal,
    std::vector<Core::Gen::Pairedvector<int, double>>& normaltonodelin) const
{
  // define tolerance
  const double tol = 1e-8;
  double dist = 1e12;
  int nrow = mele.num_node();

  const Node* node1 = dynamic_cast<const Node*>(mele.nodes()[0]);
  const Node* node2 = dynamic_cast<const Node*>(mele.nodes()[1]);

  double length1 = sqrt(node1->mo_data().edge_tangent()[0] * node1->mo_data().edge_tangent()[0] +
                        node1->mo_data().edge_tangent()[1] * node1->mo_data().edge_tangent()[1] +
                        node1->mo_data().edge_tangent()[2] * node1->mo_data().edge_tangent()[2]);
  double length2 = sqrt(node2->mo_data().edge_tangent()[0] * node2->mo_data().edge_tangent()[0] +
                        node2->mo_data().edge_tangent()[1] * node2->mo_data().edge_tangent()[1] +
                        node2->mo_data().edge_tangent()[2] * node2->mo_data().edge_tangent()[2]);

  if (length1 < 1e-12 or length2 < 1e-12) return dist;

  // calc angle between tangents
  std::array<double, 3> t1 = {0.0, 0.0, 0.0};
  std::array<double, 3> t2 = {0.0, 0.0, 0.0};

  t1[0] = node1->mo_data().edge_tangent()[0];
  t1[1] = node1->mo_data().edge_tangent()[1];
  t1[2] = node1->mo_data().edge_tangent()[2];

  t2[0] = node2->mo_data().edge_tangent()[0];
  t2[1] = node2->mo_data().edge_tangent()[1];
  t2[2] = node2->mo_data().edge_tangent()[2];

  double test = t1[0] * t2[0] + t1[1] * t2[1] + t1[2] * t2[2];
  if (test < tol) FOUR_C_THROW("tangents have wrong direction!");


  double f = 0.0;
  double df = 0.0;
  double xi = 0.0;

  // newton loop
  for (int k = 0; k < MORTARMAXITER; ++k)
  {
    //    std::cout << "k= " << k << std::endl;
    //**********************************************
    //  F CALCULATION                             //
    //**********************************************
    Core::LinAlg::SerialDenseVector sval(nrow);
    Core::LinAlg::SerialDenseMatrix sderiv(nrow, 1);
    mele.evaluate_shape(&xi, sval, sderiv, nrow);

    // tangent part
    std::array<double, 3> tangent = {0.0, 0.0, 0.0};
    tangent[0] += sval[0] * dynamic_cast<const Node*>(mele.nodes()[0])->mo_data().edge_tangent()[0];
    tangent[1] += sval[0] * dynamic_cast<const Node*>(mele.nodes()[0])->mo_data().edge_tangent()[1];
    tangent[2] += sval[0] * dynamic_cast<const Node*>(mele.nodes()[0])->mo_data().edge_tangent()[2];

    tangent[0] += sval[1] * dynamic_cast<const Node*>(mele.nodes()[1])->mo_data().edge_tangent()[0];
    tangent[1] += sval[1] * dynamic_cast<const Node*>(mele.nodes()[1])->mo_data().edge_tangent()[1];
    tangent[2] += sval[1] * dynamic_cast<const Node*>(mele.nodes()[1])->mo_data().edge_tangent()[2];

    double tangentSlave = 0.0;
    tangentSlave = tangent[0] * snode.xspatial()[0] + tangent[1] * snode.xspatial()[1] +
                   tangent[2] * snode.xspatial()[2];

    // master part
    std::array<double, 3> master = {0.0, 0.0, 0.0};
    master[0] += sval[0] * node1->xspatial()[0];
    master[1] += sval[0] * node1->xspatial()[1];
    master[2] += sval[0] * node1->xspatial()[2];

    master[0] += sval[1] * node2->xspatial()[0];
    master[1] += sval[1] * node2->xspatial()[1];
    master[2] += sval[1] * node2->xspatial()[2];

    double tangentMaster = 0.0;
    tangentMaster = tangent[0] * master[0] + tangent[1] * master[1] + tangent[2] * master[2];

    f = tangentSlave - tangentMaster;
    if (abs(f) <= MORTARCONVTOL) break;
    //**********************************************
    //   F GRADIENT CALCULATION                   //
    //**********************************************
    // lin tangent part
    std::array<double, 3> lintangent = {0.0, 0.0, 0.0};
    lintangent[0] += sderiv(0, 0) * node1->mo_data().edge_tangent()[0];
    lintangent[1] += sderiv(0, 0) * node1->mo_data().edge_tangent()[1];
    lintangent[2] += sderiv(0, 0) * node1->mo_data().edge_tangent()[2];

    lintangent[0] += sderiv(1, 0) * node2->mo_data().edge_tangent()[0];
    lintangent[1] += sderiv(1, 0) * node2->mo_data().edge_tangent()[1];
    lintangent[2] += sderiv(1, 0) * node2->mo_data().edge_tangent()[2];

    double lintangentSlave = 0.0;
    lintangentSlave = lintangent[0] * snode.xspatial()[0] + lintangent[1] * snode.xspatial()[1] +
                      lintangent[2] * snode.xspatial()[2];

    // lin master part
    std::array<double, 3> linmaster = {0.0, 0.0, 0.0};
    linmaster[0] += sderiv(0, 0) * node1->xspatial()[0];
    linmaster[1] += sderiv(0, 0) * node1->xspatial()[1];
    linmaster[2] += sderiv(0, 0) * node1->xspatial()[2];

    linmaster[0] += sderiv(1, 0) * node2->xspatial()[0];
    linmaster[1] += sderiv(1, 0) * node2->xspatial()[1];
    linmaster[2] += sderiv(1, 0) * node2->xspatial()[2];

    double lintangentMaster = 0.0;
    lintangentMaster =
        lintangent[0] * master[0] + lintangent[1] * master[1] + lintangent[2] * master[2];

    double tangentlinMaster = 0.0;
    tangentlinMaster =
        tangent[0] * linmaster[0] + tangent[1] * linmaster[1] + tangent[2] * linmaster[2];

    df = lintangentSlave - lintangentMaster - tangentlinMaster;
    if (abs(df) < 1e-12) FOUR_C_THROW("df zero");
    xi += -f / df;
  }

  //**********************************************
  //   CHECK XI                                 //
  //**********************************************
  if (-1.0 - tol > xi or xi > 1.0 + tol) return dist;

  //**********************************************
  //   LINEARIZATION   df                       //
  //**********************************************

  Core::LinAlg::SerialDenseVector sval(nrow);
  Core::LinAlg::SerialDenseMatrix sderiv(nrow, 1);
  mele.evaluate_shape(&xi, sval, sderiv, nrow);

  // tangent part
  std::array<double, 3> tangent = {0.0, 0.0, 0.0};
  tangent[0] += sval[0] * dynamic_cast<const Node*>(mele.nodes()[0])->mo_data().edge_tangent()[0];
  tangent[1] += sval[0] * dynamic_cast<const Node*>(mele.nodes()[0])->mo_data().edge_tangent()[1];
  tangent[2] += sval[0] * dynamic_cast<const Node*>(mele.nodes()[0])->mo_data().edge_tangent()[2];

  tangent[0] += sval[1] * dynamic_cast<const Node*>(mele.nodes()[1])->mo_data().edge_tangent()[0];
  tangent[1] += sval[1] * dynamic_cast<const Node*>(mele.nodes()[1])->mo_data().edge_tangent()[1];
  tangent[2] += sval[1] * dynamic_cast<const Node*>(mele.nodes()[1])->mo_data().edge_tangent()[2];

  // master part
  std::array<double, 3> master = {0.0, 0.0, 0.0};
  master[0] += sval[0] * node1->xspatial()[0];
  master[1] += sval[0] * node1->xspatial()[1];
  master[2] += sval[0] * node1->xspatial()[2];

  master[0] += sval[1] * node2->xspatial()[0];
  master[1] += sval[1] * node2->xspatial()[1];
  master[2] += sval[1] * node2->xspatial()[2];

  // lin tangent part
  std::array<double, 3> lintangent = {0.0, 0.0, 0.0};
  lintangent[0] +=
      sderiv(0, 0) * dynamic_cast<const Node*>(mele.nodes()[0])->mo_data().edge_tangent()[0];
  lintangent[1] +=
      sderiv(0, 0) * dynamic_cast<const Node*>(mele.nodes()[0])->mo_data().edge_tangent()[1];
  lintangent[2] +=
      sderiv(0, 0) * dynamic_cast<const Node*>(mele.nodes()[0])->mo_data().edge_tangent()[2];

  lintangent[0] +=
      sderiv(1, 0) * dynamic_cast<const Node*>(mele.nodes()[1])->mo_data().edge_tangent()[0];
  lintangent[1] +=
      sderiv(1, 0) * dynamic_cast<const Node*>(mele.nodes()[1])->mo_data().edge_tangent()[1];
  lintangent[2] +=
      sderiv(1, 0) * dynamic_cast<const Node*>(mele.nodes()[1])->mo_data().edge_tangent()[2];

  // lin master part
  std::array<double, 3> linmaster = {0.0, 0.0, 0.0};
  linmaster[0] += sderiv(0, 0) * node1->xspatial()[0];
  linmaster[1] += sderiv(0, 0) * node1->xspatial()[1];
  linmaster[2] += sderiv(0, 0) * node1->xspatial()[2];

  linmaster[0] += sderiv(1, 0) * node2->xspatial()[0];
  linmaster[1] += sderiv(1, 0) * node2->xspatial()[1];
  linmaster[2] += sderiv(1, 0) * node2->xspatial()[2];


  //**********************************************
  //   LINEARIZATION    f                       //
  //**********************************************
  std::vector<Core::Gen::Pairedvector<int, double>> linT(3, 100);  // added all sizes

  for (const auto& p : node1->data().get_deriv_tangent()[0])
  {
    linT[0][p.first] += sval[0] * p.second;
  }
  for (const auto& p : node1->data().get_deriv_tangent()[1])
  {
    linT[1][p.first] += sval[0] * p.second;
  }
  for (const auto& p : node1->data().get_deriv_tangent()[2])
  {
    linT[2][p.first] += sval[0] * p.second;
  }

  for (const auto& p : node2->data().get_deriv_tangent()[0])
  {
    linT[0][p.first] += sval[1] * p.second;
  }
  for (const auto& p : node2->data().get_deriv_tangent()[1])
  {
    linT[1][p.first] += sval[1] * p.second;
  }
  for (const auto& p : node2->data().get_deriv_tangent()[2])
  {
    linT[2][p.first] += sval[1] * p.second;
  }

  std::vector<Core::Gen::Pairedvector<int, double>> linXsl(3, 100);  // added all sizes
  linXsl[0][snode.dofs()[0]] += 1.0;
  linXsl[1][snode.dofs()[1]] += 1.0;
  linXsl[2][snode.dofs()[2]] += 1.0;

  std::vector<Core::Gen::Pairedvector<int, double>> linXm(3, 100);  // added all sizes
  linXm[0][node1->dofs()[0]] += sval[0];
  linXm[1][node1->dofs()[1]] += sval[0];
  linXm[2][node1->dofs()[2]] += sval[0];

  linXm[0][node2->dofs()[0]] += sval[1];
  linXm[1][node2->dofs()[1]] += sval[1];
  linXm[2][node2->dofs()[2]] += sval[1];

  Core::Gen::Pairedvector<int, double> linf(100);  // added all sizes
  for (const auto& p : linT[0])
  {
    linf[p.first] += snode.xspatial()[0] * p.second;
  }
  for (const auto& p : linT[1])
  {
    linf[p.first] += snode.xspatial()[1] * p.second;
  }
  for (const auto& p : linT[2])
  {
    linf[p.first] += snode.xspatial()[2] * p.second;
  }

  for (const auto& p : linXsl[0])
  {
    linf[p.first] += tangent[0] * p.second;
  }
  for (const auto& p : linXsl[1])
  {
    linf[p.first] += tangent[1] * p.second;
  }
  for (const auto& p : linXsl[2])
  {
    linf[p.first] += tangent[2] * p.second;
  }

  for (const auto& p : linT[0])
  {
    linf[p.first] -= master[0] * p.second;
  }
  for (const auto& p : linT[1])
  {
    linf[p.first] -= master[1] * p.second;
  }
  for (const auto& p : linT[2])
  {
    linf[p.first] -= master[2] * p.second;
  }

  for (const auto& p : linXm[0])
  {
    linf[p.first] -= tangent[0] * p.second;
  }
  for (const auto& p : linXm[1])
  {
    linf[p.first] -= tangent[1] * p.second;
  }
  for (const auto& p : linXm[2])
  {
    linf[p.first] -= tangent[2] * p.second;
  }

  Core::Gen::Pairedvector<int, double> linXi(100);  // added all sizes
  for (const auto& p : linf)
  {
    linXi[p.first] -= p.second / df;
  }

  //**********************************************
  //   CALC NORMAL                              //
  //**********************************************
  std::array<double, 3> auxnormal = {0.0, 0.0, 0.0};
  auxnormal[0] = snode.xspatial()[0] - master[0];
  auxnormal[1] = snode.xspatial()[1] - master[1];
  auxnormal[2] = snode.xspatial()[2] - master[2];

  // calc distance
  dist =
      sqrt(auxnormal[0] * auxnormal[0] + auxnormal[1] * auxnormal[1] + auxnormal[2] * auxnormal[2]);

  if (abs(dist) < 1e-12) return 1e12;

  //*******************************************
  // Lin:
  std::vector<Core::Gen::Pairedvector<int, double>> auxlin(3, 100);  // added all sizes

  // xslave
  for (int k = 0; k < 3; ++k) (auxlin[k])[snode.dofs()[k]] += 1.0;

  // xmaster n1
  for (int k = 0; k < 3; ++k) (auxlin[k])[node1->dofs()[k]] -= sval[0];
  // xmaster n2
  for (int k = 0; k < 3; ++k) (auxlin[k])[node2->dofs()[k]] -= sval[1];

  for (const auto& p : linXi)
  {
    for (int k = 0; k < 3; ++k)
    {
      (auxlin[k])[p.first] -= sderiv(0, 0) * node1->xspatial()[k] * p.second;
      (auxlin[k])[p.first] -= sderiv(1, 0) * node2->xspatial()[k] * p.second;
    }
  }

  normal[0] = auxnormal[0];
  normal[1] = auxnormal[1];
  normal[2] = auxnormal[2];

  //******************************
  // Orientation check:
  std::array<double, 3> slavebasednormal = {0.0, 0.0, 0.0};
  int nseg = snode.num_element();

  // we need to store some stuff here
  //**********************************************************************
  // elens(0,i): x-coord of element normal
  // elens(1,i): y-coord of element normal
  // elens(2,i): z-coord of element normal
  // elens(3,i): id of adjacent element i
  // elens(4,i): length of element normal
  // elens(5,i): length/area of element itself
  //**********************************************************************
  Core::LinAlg::SerialDenseMatrix elens(6, nseg);
  const auto* adjmrtrele =
      dynamic_cast<const Mortar::Element*>(snode.adjacent_elements()[0].user_element());

  // build element normal at current node
  // (we have to pass in the index i to be able to store the
  // normal and other information at the right place in elens)
  int i = 0;
  adjmrtrele->build_normal_at_node(snode.id(), i, elens);

  // add (weighted) element normal to nodal normal n
  for (int j = 0; j < 3; ++j) slavebasednormal[j] += elens(j, 0) / elens(4, 0);

  // create unit normal vector
  const double length =
      sqrt(slavebasednormal[0] * slavebasednormal[0] + slavebasednormal[1] * slavebasednormal[1] +
           slavebasednormal[2] * slavebasednormal[2]);
  if (abs(length) < 1e-12)
  {
    FOUR_C_THROW("Nodal normal length 0, node ID {}", snode.id());
  }
  else
  {
    for (int j = 0; j < 3; ++j) slavebasednormal[j] /= length;
  }

  const double dotprod =
      -(normal[0] / dist * slavebasednormal[0] + normal[1] / dist * slavebasednormal[1] +
          normal[2] / dist * slavebasednormal[2]);

  if (dotprod < -1e-12)
  {
    // get the cpp normal
    normal[0] = -normal[0];
    normal[1] = -normal[1];
    normal[2] = -normal[2];

    for (int j = 0; j < 3; ++j)
    {
      for (auto p = auxlin[j].begin(); p != auxlin[j].end(); ++p)
      {
        (normaltonodelin[j])[p->first] -= (p->second);
      }
    }
  }
  else
  {
    // linearization
    for (int j = 0; j < 3; ++j)
    {
      for (auto p = auxlin[j].begin(); p != auxlin[j].end(); ++p)
      {
        (normaltonodelin[j])[p->first] += (p->second);
      }
    }
  }

  return dist;
}

/*----------------------------------------------------------------------*
 |  cpp to node + Lin                                       farah 01/16 |
 *----------------------------------------------------------------------*/
double CONTACT::Interface::compute_normal_node_to_node(const Mortar::Node& snode,
    const Mortar::Node& mnode, double* normal,
    std::vector<Core::Gen::Pairedvector<int, double>>& normaltonodelin) const
{
  const int dim = n_dim();

  // distance between node and surface
  double gdist = 1e12;
  std::array<double, 3> gnormal = {0.0, 0.0, 0.0};
  std::vector<Core::Gen::Pairedvector<int, double>> glin(3, 1000);

  double dist = 1e12;
  std::array<double, 3> auxnormal = {0.0, 0.0, 0.0};

  // loop over found master nodes
  std::vector<Core::Gen::Pairedvector<int, double>> auxlin(3, 1000);

  // calc vector
  auxnormal[0] = snode.xspatial()[0] - mnode.xspatial()[0];
  auxnormal[1] = snode.xspatial()[1] - mnode.xspatial()[1];
  auxnormal[2] = snode.xspatial()[2] - mnode.xspatial()[2];

  // remove numerical artifacts
  if (abs(auxnormal[0]) < 1e-12) auxnormal[0] = 0.0;
  if (abs(auxnormal[1]) < 1e-12) auxnormal[1] = 0.0;
  if (abs(auxnormal[2]) < 1e-12) auxnormal[2] = 0.0;

  // calc distance
  dist =
      sqrt(auxnormal[0] * auxnormal[0] + auxnormal[1] * auxnormal[1] + auxnormal[2] * auxnormal[2]);

  // if nodes lying on each other: continue to next master node
  if (abs(dist) < 1e-12) return dist;

  //*******************************************
  // Lin:
  // xslave
  for (int k = 0; k < dim; ++k) (auxlin[k])[snode.dofs()[k]] += 1.0;

  // xmaster
  for (int k = 0; k < dim; ++k) (auxlin[k])[mnode.dofs()[k]] -= 1.0;

  // get normal
  gdist = dist;

  // normalize vector
  gnormal[0] = auxnormal[0];  /// dist;
  gnormal[1] = auxnormal[1];  /// dist;
  gnormal[2] = auxnormal[2];  /// dist;

  // linearization
  glin = auxlin;

  // get the cpp normal
  normal[0] = gnormal[0];
  normal[1] = gnormal[1];
  normal[2] = gnormal[2];

  //******************************
  // Orientation check:
  std::array<double, 3> slavebasednormal = {0.0, 0.0, 0.0};
  int nseg = snode.num_element();

  // we need to store some stuff here
  //**********************************************************************
  // elens(0,i): x-coord of element normal
  // elens(1,i): y-coord of element normal
  // elens(2,i): z-coord of element normal
  // elens(3,i): id of adjacent element i
  // elens(4,i): length of element normal
  // elens(5,i): length/area of element itself
  //**********************************************************************
  Core::LinAlg::SerialDenseMatrix elens(6, nseg);
  const auto* adjmrtrele =
      dynamic_cast<const Mortar::Element*>(snode.adjacent_elements()[0].user_element());

  // build element normal at current node
  // (we have to pass in the index i to be able to store the
  // normal and other information at the right place in elens)
  int i = 0;
  adjmrtrele->build_normal_at_node(snode.id(), i, elens);

  // add (weighted) element normal to nodal normal n
  for (int j = 0; j < 3; ++j) slavebasednormal[j] += elens(j, 0) / elens(4, 0);

  // create unit normal vector
  const double length =
      sqrt(slavebasednormal[0] * slavebasednormal[0] + slavebasednormal[1] * slavebasednormal[1] +
           slavebasednormal[2] * slavebasednormal[2]);
  if (abs(length) < 1e-12)
  {
    FOUR_C_THROW("Nodal normal length 0, node ID {}", snode.id());
  }
  else
  {
    for (int j = 0; j < 3; ++j) slavebasednormal[j] /= length;
  }

  const double dotprod = -(normal[0] * slavebasednormal[0] + normal[1] * slavebasednormal[1] +
                           normal[2] * slavebasednormal[2]);

  if (dotprod < -1e-12)
  {
    // get the cpp normal
    normal[0] = -normal[0];
    normal[1] = -normal[1];
    normal[2] = -normal[2];

    for (int j = 0; j < dim; ++j)
      for (auto p = glin[j].begin(); p != glin[j].end(); ++p)
        (normaltonodelin[j])[p->first] -= (p->second);
  }
  else
  {
    // linearization
    for (int j = 0; j < dim; ++j)
      for (auto p = glin[j].begin(); p != glin[j].end(); ++p)
        (normaltonodelin[j])[p->first] += (p->second);
  }


  return gdist;
}

/*----------------------------------------------------------------------*
 |  evaluate closest point normals                          farah 08/16 |
 *----------------------------------------------------------------------*/
void CONTACT::Interface::evaluate_cpp_normals()
{
  // Build averaged normal field on physically smooth surface
  // loop over proc's master nodes of the interface
  // use row map and export to column map later
  for (int i = 0; i < master_row_nodes()->num_my_elements(); ++i)
  {
    int gid = master_row_nodes()->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Node* mrtrnode = dynamic_cast<Node*>(node);

    // build averaged normal at each master node
    mrtrnode->build_averaged_normal();

    // build tangent
    if (mrtrnode->is_on_edge()) mrtrnode->build_averaged_edge_tangent();
  }

  // export nodal normals
  export_master_nodal_normals();

  // loop over slave nodes
  for (int i = 0; i < slave_row_nodes()->num_my_elements(); ++i)
  {
    int gid = slave_row_nodes()->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    auto* mrtrnode = dynamic_cast<Mortar::Node*>(node);

    if (mrtrnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
      FOUR_C_THROW("Node ownership inconsistency!");

    // vector with possible contacting master eles/nodes
    std::vector<Mortar::Element*> meles;
    std::vector<Mortar::Node*> mnodes;

    // fill vector with possibly contacting meles
    find_master_elements(*mrtrnode, meles);

    // fallback solution if no mele is available
    if (meles.size() < 1)  // or !mrtrnode->IsOnCornerEdge())
    {
      Node* cnode = dynamic_cast<Node*>(mrtrnode);
      cnode->build_averaged_normal();
      continue;
    }


    // Here we have all found master elements for one slave node.
    // distance for cpp
    double normaltoline[3] = {0.0, 0.0, 0.0};
    std::vector<Core::Gen::Pairedvector<int, double>> normaltolineLin(3, 1);  // 1 dummy

    // Now, calculate distance between node and master line
    double dist = compute_cpp_normal(*mrtrnode, meles, normaltoline, normaltolineLin);

    // if no projection was possible
    if (dist > 1e11)
    {
      Node* cnode = dynamic_cast<Node*>(mrtrnode);
      cnode->build_averaged_normal();
      continue;
    }

    // set the normal and its lineratization
    set_cpp_normal(*mrtrnode, normaltoline, normaltolineLin);
  }

  // export slave normals
  export_nodal_normals();
}


/*----------------------------------------------------------------------*
 |  export master nodal normals (protected)                  farah 08/16|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::export_master_nodal_normals() const
{
  std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>> triad;

  std::map<int, std::vector<int>> n_x_key;
  std::map<int, std::vector<int>> n_y_key;
  std::map<int, std::vector<int>> n_z_key;
  std::map<int, std::vector<int>> txi_x_key;
  std::map<int, std::vector<int>> txi_y_key;
  std::map<int, std::vector<int>> txi_z_key;
  std::map<int, std::vector<int>> teta_x_key;
  std::map<int, std::vector<int>> teta_y_key;
  std::map<int, std::vector<int>> teta_z_key;

  std::map<int, std::vector<double>> n_x_val;
  std::map<int, std::vector<double>> n_y_val;
  std::map<int, std::vector<double>> n_z_val;
  std::map<int, std::vector<double>> txi_x_val;
  std::map<int, std::vector<double>> txi_y_val;
  std::map<int, std::vector<double>> txi_z_val;
  std::map<int, std::vector<double>> teta_x_val;
  std::map<int, std::vector<double>> teta_y_val;
  std::map<int, std::vector<double>> teta_z_val;

  Core::Gen::Pairedvector<int, double>::iterator iter;

  const std::shared_ptr<Core::LinAlg::Map> masternodes =
      Core::LinAlg::allreduce_e_map(*(mnoderowmap_));

  // build info on row map
  for (int i = 0; i < mnoderowmap_->num_my_elements(); ++i)
  {
    int gid = mnoderowmap_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    auto* cnode = dynamic_cast<CONTACT::Node*>(node);

    // fill nodal matrix
    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> loc =
        std::make_shared<Core::LinAlg::SerialDenseMatrix>(3, 3);
    (*loc)(0, 0) = cnode->mo_data().n()[0];
    (*loc)(1, 0) = cnode->mo_data().n()[1];
    (*loc)(2, 0) = cnode->mo_data().n()[2];
    (*loc)(0, 1) = cnode->data().txi()[0];
    (*loc)(1, 1) = cnode->data().txi()[1];
    (*loc)(2, 1) = cnode->data().txi()[2];
    (*loc)(0, 2) = cnode->data().teta()[0];
    (*loc)(1, 2) = cnode->data().teta()[1];
    (*loc)(2, 2) = cnode->data().teta()[2];

    triad[gid] = loc;

    // fill nodal derivative vectors
    std::vector<Core::Gen::Pairedvector<int, double>>& derivn = cnode->data().get_deriv_n();
    std::vector<Core::Gen::Pairedvector<int, double>>& derivtxi = cnode->data().get_deriv_txi();
    std::vector<Core::Gen::Pairedvector<int, double>>& derivteta = cnode->data().get_deriv_teta();

    for (iter = derivn[0].begin(); iter != derivn[0].end(); ++iter)
    {
      n_x_key[gid].push_back(iter->first);
      n_x_val[gid].push_back(iter->second);
    }
    for (iter = derivn[1].begin(); iter != derivn[1].end(); ++iter)
    {
      n_y_key[gid].push_back(iter->first);
      n_y_val[gid].push_back(iter->second);
    }
    for (iter = derivn[2].begin(); iter != derivn[2].end(); ++iter)
    {
      n_z_key[gid].push_back(iter->first);
      n_z_val[gid].push_back(iter->second);
    }

    for (iter = derivtxi[0].begin(); iter != derivtxi[0].end(); ++iter)
    {
      txi_x_key[gid].push_back(iter->first);
      txi_x_val[gid].push_back(iter->second);
    }
    for (iter = derivtxi[1].begin(); iter != derivtxi[1].end(); ++iter)
    {
      txi_y_key[gid].push_back(iter->first);
      txi_y_val[gid].push_back(iter->second);
    }
    for (iter = derivtxi[2].begin(); iter != derivtxi[2].end(); ++iter)
    {
      txi_z_key[gid].push_back(iter->first);
      txi_z_val[gid].push_back(iter->second);
    }

    for (iter = derivteta[0].begin(); iter != derivteta[0].end(); ++iter)
    {
      teta_x_key[gid].push_back(iter->first);
      teta_x_val[gid].push_back(iter->second);
    }
    for (iter = derivteta[1].begin(); iter != derivteta[1].end(); ++iter)
    {
      teta_y_key[gid].push_back(iter->first);
      teta_y_val[gid].push_back(iter->second);
    }
    for (iter = derivteta[2].begin(); iter != derivteta[2].end(); ++iter)
    {
      teta_z_key[gid].push_back(iter->first);
      teta_z_val[gid].push_back(iter->second);
    }
  }

  // communicate from master node row to column map
  Core::Communication::Exporter ex(*mnoderowmap_, *masternodes, get_comm());
  ex.do_export(triad);

  ex.do_export(n_x_key);
  ex.do_export(n_x_val);
  ex.do_export(n_y_key);
  ex.do_export(n_y_val);
  ex.do_export(n_z_key);
  ex.do_export(n_z_val);

  ex.do_export(txi_x_key);
  ex.do_export(txi_x_val);
  ex.do_export(txi_y_key);
  ex.do_export(txi_y_val);
  ex.do_export(txi_z_key);
  ex.do_export(txi_z_val);

  ex.do_export(teta_x_key);
  ex.do_export(teta_x_val);
  ex.do_export(teta_y_key);
  ex.do_export(teta_y_val);
  ex.do_export(teta_z_key);
  ex.do_export(teta_z_val);

  // extract info on column map
  for (int i = 0; i < masternodes->num_my_elements(); ++i)
  {
    // only do something for ghosted nodes
    int gid = masternodes->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    auto* cnode = dynamic_cast<CONTACT::Node*>(node);
    int linsize = cnode->get_linsize() + (int)(n_x_key[gid].size());

    if (cnode->owner() == Core::Communication::my_mpi_rank(get_comm())) continue;

    // extract info
    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> loc = triad[gid];
    cnode->mo_data().n()[0] = (*loc)(0, 0);
    cnode->mo_data().n()[1] = (*loc)(1, 0);
    cnode->mo_data().n()[2] = (*loc)(2, 0);
    cnode->data().txi()[0] = (*loc)(0, 1);
    cnode->data().txi()[1] = (*loc)(1, 1);
    cnode->data().txi()[2] = (*loc)(2, 1);
    cnode->data().teta()[0] = (*loc)(0, 2);
    cnode->data().teta()[1] = (*loc)(1, 2);
    cnode->data().teta()[2] = (*loc)(2, 2);

    // extract derivative info
    std::vector<Core::Gen::Pairedvector<int, double>>& derivn = cnode->data().get_deriv_n();
    std::vector<Core::Gen::Pairedvector<int, double>>& derivtxi = cnode->data().get_deriv_txi();
    std::vector<Core::Gen::Pairedvector<int, double>>& derivteta = cnode->data().get_deriv_teta();

    for (auto& k : derivn)
    {
      k.clear();
    }
    derivn.resize(3, linsize);
    for (auto& k : derivtxi)
    {
      k.clear();
    }
    derivtxi.resize(3, linsize);
    for (auto& k : derivteta)
    {
      k.clear();
    }
    derivteta.resize(3, linsize);

    cnode->data().get_deriv_n()[0].resize(linsize);
    cnode->data().get_deriv_n()[1].resize(linsize);
    cnode->data().get_deriv_n()[2].resize(linsize);

    cnode->data().get_deriv_txi()[0].resize(linsize);
    cnode->data().get_deriv_txi()[1].resize(linsize);
    cnode->data().get_deriv_txi()[2].resize(linsize);

    cnode->data().get_deriv_teta()[0].resize(linsize);
    cnode->data().get_deriv_teta()[1].resize(linsize);
    cnode->data().get_deriv_teta()[2].resize(linsize);

    for (int k = 0; k < (int)(n_x_key[gid].size()); ++k)
      (cnode->data().get_deriv_n()[0])[n_x_key[gid][k]] = n_x_val[gid][k];
    for (int k = 0; k < (int)(n_y_key[gid].size()); ++k)
      (cnode->data().get_deriv_n()[1])[n_y_key[gid][k]] = n_y_val[gid][k];
    for (int k = 0; k < (int)(n_z_key[gid].size()); ++k)
      (cnode->data().get_deriv_n()[2])[n_z_key[gid][k]] = n_z_val[gid][k];

    for (int k = 0; k < (int)(txi_x_key[gid].size()); ++k)
      (cnode->data().get_deriv_txi()[0])[txi_x_key[gid][k]] = txi_x_val[gid][k];
    for (int k = 0; k < (int)(txi_y_key[gid].size()); ++k)
      (cnode->data().get_deriv_txi()[1])[txi_y_key[gid][k]] = txi_y_val[gid][k];
    for (int k = 0; k < (int)(txi_z_key[gid].size()); ++k)
      (cnode->data().get_deriv_txi()[2])[txi_z_key[gid][k]] = txi_z_val[gid][k];

    for (int k = 0; k < (int)(teta_x_key[gid].size()); ++k)
      (cnode->data().get_deriv_teta()[0])[teta_x_key[gid][k]] = teta_x_val[gid][k];
    for (int k = 0; k < (int)(teta_y_key[gid].size()); ++k)
      (cnode->data().get_deriv_teta()[1])[teta_y_key[gid][k]] = teta_y_val[gid][k];
    for (int k = 0; k < (int)(teta_z_key[gid].size()); ++k)
      (cnode->data().get_deriv_teta()[2])[teta_z_key[gid][k]] = teta_z_val[gid][k];
  }

  // free memory
  triad.clear();

  n_x_key.clear();
  n_y_key.clear();
  n_z_key.clear();
  txi_x_key.clear();
  txi_y_key.clear();
  txi_z_key.clear();
  teta_x_key.clear();
  teta_y_key.clear();
  teta_z_key.clear();

  n_x_val.clear();
  n_y_val.clear();
  n_z_val.clear();
  txi_x_val.clear();
  txi_y_val.clear();
  txi_z_val.clear();
  teta_x_val.clear();
  teta_y_val.clear();
  teta_z_val.clear();
}

/*----------------------------------------------------------------------*
 |  cpp to line based on averaged nodal normal field        farah 08/16 |
 *----------------------------------------------------------------------*/
double CONTACT::Interface::compute_cpp_normal_2d(const Mortar::Node& mrtrnode,
    std::vector<Mortar::Element*> meles, double* normal,
    std::vector<Core::Gen::Pairedvector<int, double>>& normaltolineLin) const
{
  // define tolerance
  const double tol = 1e-8;
  const double validAngle = 20;

  // distance between node and surface
  double gdist = 1e12;  // distance
  std::array<double, 3> gnormal = {0.0, 0.0, 0.0};
  std::vector<Core::Gen::Pairedvector<int, double>> glin(3, 1);  // 1 dummy
  std::set<int> donebeforeMasterCorner;

  bool nodeOnNode = false;     // flag for node on node (corner on corner) setting
  bool pathdependent = false;  // flag if we have to check path from last converged check
  std::array<double, 3> vect = {0.0, 0.0, 0.0};  // patch

  // calc trajectory and node-to-node distance for corner nodes
  if (mrtrnode.is_on_corner())
  {
    pathdependent = true;
    const auto& coNode = dynamic_cast<const CONTACT::Node&>(mrtrnode);
    if (coNode.active()) pathdependent = false;

    // calculate path
    // check trajectory from considered node
    std::array<double, 3> Posn = {0.0, 0.0, 0.0};
    std::array<double, 3> Posnp = {0.0, 0.0, 0.0};
    Posn[0] = mrtrnode.x()[0] + mrtrnode.uold()[0];
    Posn[1] = mrtrnode.x()[1] + mrtrnode.uold()[1];
    if (dim_ == 3) Posn[2] = mrtrnode.x()[2] + mrtrnode.uold()[2];
    Posnp[0] = mrtrnode.xspatial()[0];
    Posnp[1] = mrtrnode.xspatial()[1];
    if (dim_ == 3) Posnp[2] = mrtrnode.xspatial()[2];

    vect[0] = Posnp[0] - Posn[0];
    vect[1] = Posnp[1] - Posn[1];
    vect[2] = Posnp[2] - Posn[2];

    double lvec = sqrt(vect[0] * vect[0] + vect[1] * vect[1] + vect[2] * vect[2]);
    if (lvec < 1e-12)
    {
      pathdependent = false;
    }
    else
    {
      vect[0] /= lvec;
      vect[1] /= lvec;
      vect[2] /= lvec;
    }

    // Compute normal to node:
    // loop over found eles
    for (auto& mele : meles)
    {
      // get linsize
      int linsize = 0;
      for (int i = 0; i < mele->num_node(); ++i)
      {
        Core::Nodes::Node* node = mele->nodes()[i];
        if (!node) FOUR_C_THROW("Cannot find master node");
        auto* mnode = dynamic_cast<CONTACT::Node*>(node);
        linsize += mnode->get_linsize();

        // if master node is also corner node
        if (mnode->is_on_corner())
        {
          auto iter = donebeforeMasterCorner.find(mnode->id());

          // if not then create ele
          if (iter != donebeforeMasterCorner.end()) continue;

          // set master corner node id
          donebeforeMasterCorner.insert(mnode->id());

          // aux variables
          double dist = 1e12;
          double auxnormal[3] = {0.0, 0.0, 0.0};
          std::vector<Core::Gen::Pairedvector<int, double>> auxlin(
              3, linsize + 1 + mele->num_node());

          // compute distance between corners
          dist = compute_normal_node_to_node(mrtrnode, *mnode, auxnormal, auxlin);

          // if nodes lying on each other
          if (abs(dist) < 1e-12)
          {
            nodeOnNode = true;
            continue;
          }

          // angle between trajectory and normal
          if (pathdependent)
          {
            double auxl = sqrt(auxnormal[0] * auxnormal[0] + auxnormal[1] * auxnormal[1]);
            if (auxl < 1e-12) continue;
            double angle = acos(-vect[0] * auxnormal[0] / auxl - vect[1] * auxnormal[1] / auxl);

            angle = 180 * (angle / 3.14159265359);
            if (abs(angle) > validAngle) continue;
          }

          // get closest valid distance
          if (abs(dist) < abs(gdist))
          {
            gdist = dist;
            gnormal[0] = auxnormal[0];
            gnormal[1] = auxnormal[1];
            gnormal[2] = auxnormal[2];
            glin = auxlin;
          }
        }
      }  // end node loop
    }  // end element loop
  }

  // loop over found eles
  for (auto& mele : meles)
  {
    bool cornerele = false;
    // get linsize
    int linsize = 0;
    for (int i = 0; i < mele->num_node(); ++i)
    {
      Core::Nodes::Node* node = mele->nodes()[i];
      if (!node) FOUR_C_THROW("Cannot find master node");
      auto* mnode = dynamic_cast<CONTACT::Node*>(node);
      linsize += mnode->get_linsize();

      if (mnode->is_on_corner()) cornerele = true;
    }

    double xi[2] = {0.0, 0.0};
    double dist = 1e12;
    double auxnormal[3] = {0.0, 0.0, 0.0};
    std::vector<Core::Gen::Pairedvector<int, double>> auxlin(3, linsize + 1 + mele->num_node());

    // check for nonsmooth mele
    if (cornerele and !nodeOnNode)
    {
      // perform CPP to find normals based on element normals
      Mortar::Projector::impl(*mele)->project_s_node_by_m_normal_lin(
          mrtrnode, *mele, xi, auxnormal, dist, auxlin);
    }
    // compute normal with averaged nodal normal field from master surface
    else
    {
      // perform CPP to find normals based on averaged nodal normal field
      Mortar::Projector::impl(*mele)->project_s_node_by_m_nodal_normal_lin(
          mrtrnode, *mele, xi, auxnormal, dist, auxlin);
    }

    // check if found parameter space coordinate is within element domain
    if (mele->shape() == Core::FE::CellType::line2 or mele->shape() == Core::FE::CellType::line3)
    {
      if (-1.0 - tol > xi[0] or xi[0] > 1.0 + tol) continue;
    }
    else
    {
      FOUR_C_THROW("Unknown ele type!");
    }

    // angle between trajectory and normal
    if (pathdependent)
    {
      double auxl = sqrt(auxnormal[0] * auxnormal[0] + auxnormal[1] * auxnormal[1]);
      if (auxl < 1e-12) continue;
      double angle = acos(-vect[0] * auxnormal[0] / auxl - vect[1] * auxnormal[1] / auxl);

      angle = 180 * (angle / 3.14159265359);
      if (abs(angle) > validAngle) continue;
    }

    // get closest valid distance
    if (abs(dist) < abs(gdist))
    {
      gdist = dist;
      gnormal[0] = auxnormal[0];
      gnormal[1] = auxnormal[1];
      gnormal[2] = auxnormal[2];
      glin = auxlin;
    }
  }  // end mele loop

  // get the cpp normal
  normal[0] = gnormal[0];
  normal[1] = gnormal[1];
  normal[2] = gnormal[2];
  normaltolineLin = glin;

  return gdist;
}

/*----------------------------------------------------------------------*
 |  cpp to line based on averaged nodal normal field        farah 08/16 |
 *----------------------------------------------------------------------*/
double CONTACT::Interface::compute_cpp_normal_3d(Mortar::Node& mrtrnode,
    std::vector<Mortar::Element*> meles, double* normal,
    std::vector<Core::Gen::Pairedvector<int, double>>& normaltolineLin)
{
  // define tolerance
  const double tol = 1e-8;

  // distance between node and surface
  bool pathdependent = true;
  const double validAngle = 5.0;
  double gdist = 1e12;  // distance
  std::array<double, 3> gnormal = {0.0, 0.0, 0.0};
  std::vector<Core::Gen::Pairedvector<int, double>> glin(3, 1);  // 1 dummy

  //******************************************************
  //             CALC TRAJECTORY
  //******************************************************
  std::array<double, 3> Posn = {0.0, 0.0, 0.0};
  std::array<double, 3> Posnp = {0.0, 0.0, 0.0};
  std::array<double, 3> vect = {0.0, 0.0, 0.0};

  Posn[0] = mrtrnode.x()[0] + mrtrnode.uold()[0];
  Posn[1] = mrtrnode.x()[1] + mrtrnode.uold()[1];
  if (dim_ == 3) Posn[2] = mrtrnode.x()[2] + mrtrnode.uold()[2];
  Posnp[0] = mrtrnode.xspatial()[0];
  Posnp[1] = mrtrnode.xspatial()[1];
  if (dim_ == 3) Posnp[2] = mrtrnode.xspatial()[2];

  vect[0] = Posnp[0] - Posn[0];
  vect[1] = Posnp[1] - Posn[1];
  vect[2] = Posnp[2] - Posn[2];

  double lvec = sqrt(vect[0] * vect[0] + vect[1] * vect[1] + vect[2] * vect[2]);
  if (lvec < 1e-12)
  {
    pathdependent = false;
  }
  else
  {
    vect[0] /= lvec;
    vect[1] /= lvec;
    vect[2] /= lvec;
  }

  if (dynamic_cast<CONTACT::Node&>(mrtrnode).active()) pathdependent = false;

  //******************************************************
  //             COMPUTE NORMAL TO SURFACE
  //******************************************************
  // loop over found eles for all geometrical nodes
  for (auto& mele : meles)
  {
    double xi[2] = {0.0, 0.0};
    double dist = 1e12;
    double auxnormal[3] = {0.0, 0.0, 0.0};
    std::vector<Core::Gen::Pairedvector<int, double>> auxlin(3, 1000);

    // perform CPP to find normals
    bool success = Mortar::Projector::impl(*mele)->project_s_node_by_m_nodal_normal_lin(
        mrtrnode, *mele, xi, auxnormal, dist, auxlin);

    // newton not converged
    if (!success) continue;

    // check if found parameter space coordinate is within element domain
    if (mele->shape() == Core::FE::CellType::quad4 or mele->shape() == Core::FE::CellType::quad8 or
        mele->shape() == Core::FE::CellType::quad9)
    {
      if (-1.0 - tol > xi[0] or xi[0] > 1.0 + tol or -1.0 - tol > xi[1] or xi[1] > 1.0 + tol)
        continue;
    }
    else if (mele->shape() == Core::FE::CellType::tri3 or mele->shape() == Core::FE::CellType::tri6)
    {
      if (xi[0] < 0.0 - tol or xi[1] < 0.0 - tol or xi[0] > 1.0 + tol or xi[1] > 1.0 + tol or
          xi[0] + xi[1] > 1.0 + 2 * tol)
        continue;
    }
    else
    {
      FOUR_C_THROW("Unknown ele type!");
    }

    // angle between trajectory and normal
    if (pathdependent)
    {
      double auxl = sqrt(
          auxnormal[0] * auxnormal[0] + auxnormal[1] * auxnormal[1] + auxnormal[2] * auxnormal[2]);
      if (auxl < 1e-12) continue;
      double angle = acos(-vect[0] * auxnormal[0] / auxl - vect[1] * auxnormal[1] / auxl -
                          vect[2] * auxnormal[2] / auxl);

      angle = 180 * (angle / 3.14159265359);
      if (abs(angle) > validAngle) continue;
    }

    if (dist < gdist)
    {
      gdist = dist;
      gnormal[0] = auxnormal[0];
      gnormal[1] = auxnormal[1];
      gnormal[2] = auxnormal[2];
      glin = auxlin;
    }
  }  // end mele loop
  //******************************************************
  //             COMPUTE NORMAL TO LINE
  //******************************************************
  if (mrtrnode.is_on_corner_edge())  // only for edge or corner nodes possible
  {
    // guarantee uniqueness
    std::set<std::pair<int, int>> donebefore;

    // calc
    for (auto& mele : meles)
    {
      // loop over master edges -> match node number for quad4
      for (int j = 0; j < mele->num_node(); ++j)
      {
        int nodeIds[2] = {0, 0};
        int nodeLIds[2] = {0, 0};

        if (mele->shape() == Core::FE::CellType::quad4)
        {
          if (j == 0)
          {
            nodeIds[0] = mele->node_ids()[0];
            nodeIds[1] = mele->node_ids()[1];

            nodeLIds[0] = 0;
            nodeLIds[1] = 1;
          }
          else if (j == 1)
          {
            nodeIds[0] = mele->node_ids()[1];
            nodeIds[1] = mele->node_ids()[2];

            nodeLIds[0] = 1;
            nodeLIds[1] = 2;
          }
          else if (j == 2)
          {
            nodeIds[0] = mele->node_ids()[2];
            nodeIds[1] = mele->node_ids()[3];

            nodeLIds[0] = 2;
            nodeLIds[1] = 3;
          }
          else if (j == 3)
          {
            nodeIds[0] = mele->node_ids()[3];
            nodeIds[1] = mele->node_ids()[0];

            nodeLIds[0] = 3;
            nodeLIds[1] = 0;
          }
          else
          {
            FOUR_C_THROW("loop counter and edge number do not match!");
          }
        }

        // check if both nodes on edge geometry
        bool node0Edge =
            dynamic_cast<Mortar::Node*>(mele->nodes()[nodeLIds[0]])->is_on_corner_edge();
        bool node1Edge =
            dynamic_cast<Mortar::Node*>(mele->nodes()[nodeLIds[1]])->is_on_corner_edge();

        if (!node0Edge or !node1Edge) continue;

        // create pair
        std::pair<int, int> actIDs = std::pair<int, int>(nodeIds[0], nodeIds[1]);
        std::pair<int, int> actIDstw = std::pair<int, int>(nodeIds[1], nodeIds[0]);

        // check if processed before
        auto iter = donebefore.find(actIDs);
        auto itertw = donebefore.find(actIDstw);

        // if not then create ele
        if (iter == donebefore.end() and itertw == donebefore.end())
        {
          // add to set of processed nodes
          donebefore.insert(actIDs);
          donebefore.insert(actIDstw);

          // create line ele:
          Mortar::Element lineEle(j, mele->owner(), Core::FE::CellType::line2, 2, nodeIds, false);

          // get nodes
          Core::Nodes::Node* nodes[2] = {mele->nodes()[nodeLIds[0]], mele->nodes()[nodeLIds[1]]};
          lineEle.build_nodal_pointers(nodes);

          // init data container for dual shapes
          lineEle.initialize_data_container();

          // call cpp function for edge to edge

          double dist = 1e12;
          double auxnormal[3] = {0.0, 0.0, 0.0};
          std::vector<Core::Gen::Pairedvector<int, double>> auxlin(3, 100 + 1 + mele->num_node());

          // compute distance between node and edge
          dist = compute_normal_node_to_edge(mrtrnode, lineEle, auxnormal, auxlin);

          // angle between trajectory and normal
          if (pathdependent)
          {
            double auxl = sqrt(auxnormal[0] * auxnormal[0] + auxnormal[1] * auxnormal[1] +
                               auxnormal[2] * auxnormal[2]);
            if (auxl < 1e-12) continue;
            double angle = acos(-vect[0] * auxnormal[0] / auxl - vect[1] * auxnormal[1] / auxl -
                                vect[2] * auxnormal[2] / auxl);

            angle = 180 * (angle / 3.14159265359);
            if (abs(angle) > validAngle) continue;
          }

          if (dist <= gdist + tol)
          {
            std::cout << "CLOSE TO EDGE!!!" << '\n';
            gdist = dist;
            gnormal[0] = auxnormal[0];
            gnormal[1] = auxnormal[1];
            gnormal[2] = auxnormal[2];
            glin = auxlin;
          }
        }
      }  // end mele node loop
    }
  }

  //******************************************************
  //             COMPUTE NORMAL TO NODE
  //******************************************************
  if (mrtrnode.is_on_corner())  // only for corner nodes possible
  {
    std::set<int> donebeforeMasterCorner;

    for (auto& mele : meles)
    {
      // get linsize
      int linsize = 0;
      for (int i = 0; i < mele->num_node(); ++i)
      {
        Core::Nodes::Node* node = mele->nodes()[i];
        if (!node) FOUR_C_THROW("Cannot find master node");
        auto* mnode = dynamic_cast<CONTACT::Node*>(node);
        linsize += mnode->get_linsize();

        // if master node is also corner node
        if (mnode->is_on_corner())
        {
          auto iter = donebeforeMasterCorner.find(mnode->id());

          // if not then create ele
          if (iter != donebeforeMasterCorner.end()) continue;

          // set master corner node id
          donebeforeMasterCorner.insert(mnode->id());

          double dist = 1e12;
          double auxnormal[3] = {0.0, 0.0, 0.0};
          std::vector<Core::Gen::Pairedvector<int, double>> auxlin(
              3, linsize + 1 + mele->num_node());

          // compute distance between corners
          dist = compute_normal_node_to_node(mrtrnode, *mnode, auxnormal, auxlin);

          // angle between trajectory and normal
          if (pathdependent)
          {
            double auxl = sqrt(auxnormal[0] * auxnormal[0] + auxnormal[1] * auxnormal[1] +
                               auxnormal[2] * auxnormal[2]);
            if (auxl < 1e-12) continue;
            double angle = acos(-vect[0] * auxnormal[0] / auxl - vect[1] * auxnormal[1] / auxl -
                                vect[2] * auxnormal[2] / auxl);

            angle = 180 * (angle / 3.14159265359);
            if (abs(angle) > validAngle) continue;
          }

          if (dist < gdist)
          {
            gdist = dist;
            gnormal[0] = auxnormal[0];
            gnormal[1] = auxnormal[1];
            gnormal[2] = auxnormal[2];
            glin = auxlin;
          }
        }
      }
    }
  }

  //******************************************************
  //             FINAL STORAGE
  //******************************************************

  // get the cpp normal
  normal[0] = gnormal[0];
  normal[1] = gnormal[1];
  normal[2] = gnormal[2];
  normaltolineLin = glin;

  // bye bye
  return gdist;
}

/*----------------------------------------------------------------------*
 |  cpp to line based on averaged nodal normal field        farah 05/16 |
 *----------------------------------------------------------------------*/
double CONTACT::Interface::compute_cpp_normal(Mortar::Node& mrtrnode,
    std::vector<Mortar::Element*> meles, double* normal,
    std::vector<Core::Gen::Pairedvector<int, double>>& normaltolineLin)
{
  // define distance
  double gdist = 1e12;

  //===================================================================
  //===================================================================
  //                           2D case
  //===================================================================
  //===================================================================
  if (n_dim() == 2)
  {
    gdist = compute_cpp_normal_2d(mrtrnode, meles, normal, normaltolineLin);
  }
  //===================================================================
  //===================================================================
  //                           3D case
  //===================================================================
  //===================================================================
  else if (n_dim() == 3)
  {
    gdist = compute_cpp_normal_3d(mrtrnode, meles, normal, normaltolineLin);
  }
  //===================================================================
  //===================================================================
  //                           Invalid
  //===================================================================
  //===================================================================
  else
  {
    FOUR_C_THROW("invalid dimension!");
  }

  // return distance
  return gdist;
}

/*----------------------------------------------------------------------*
 |  set cpp normal                                           farah 01/16|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::set_cpp_normal(Mortar::Node& snode, double* normal,
    std::vector<Core::Gen::Pairedvector<int, double>>& normallin)
{
  Node& cnode = dynamic_cast<Node&>(snode);

  const double length = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
  if (length < 1e-12) FOUR_C_THROW("normal length is zero!!!");

  // negative sign because it is a master normal!
  cnode.mo_data().n()[0] = -normal[0] / length;
  cnode.mo_data().n()[1] = -normal[1] / length;
  cnode.mo_data().n()[2] = -normal[2] / length;

  //  if (cnode.IsOnEdge())
  //    std::cout << "normal =  " << cnode.MoData().n()[0] << "  "<< cnode.MoData().n()[1] << "  "<<
  //    cnode.MoData().n()[2] << std::endl;

  // prepare nodal storage maps for derivative
  if ((int)cnode.data().get_deriv_n().size() == 0)
    cnode.data().get_deriv_n().resize(3, normallin[0].size() * 3);
  if ((int)cnode.data().get_deriv_txi().size() == 0)
    cnode.data().get_deriv_txi().resize(3, normallin[0].size() * 3);
  if ((int)cnode.data().get_deriv_teta().size() == 0)
    cnode.data().get_deriv_teta().resize(3, normallin[0].size() * 3);

  // init tangent length
  double ltxi = -1.0;

  //------------------------------------------------------------------
  // 2D Tangent!
  if (cnode.num_dof() == 2)
  {
    // simple definition for txi
    cnode.data().txi()[0] = -cnode.mo_data().n()[1];
    cnode.data().txi()[1] = cnode.mo_data().n()[0];
    cnode.data().txi()[2] = 0.0;

    // teta is z-axis
    cnode.data().teta()[0] = 0.0;
    cnode.data().teta()[1] = 0.0;
    cnode.data().teta()[2] = 1.0;
  }
  // 3D Tangent!
  else
  {
    if (abs(cnode.mo_data().n()[0]) > 1.0e-6 || abs(cnode.mo_data().n()[1]) > 1.0e-6)
    {
      cnode.data().txi()[0] = -cnode.mo_data().n()[1];
      cnode.data().txi()[1] = cnode.mo_data().n()[0];
      cnode.data().txi()[2] = 0.0;
    }
    else
    {
      cnode.data().txi()[0] = 0.0;
      cnode.data().txi()[1] = -cnode.mo_data().n()[2];
      cnode.data().txi()[2] = cnode.mo_data().n()[1];
    }

    ltxi = sqrt(cnode.data().txi()[0] * cnode.data().txi()[0] +
                cnode.data().txi()[1] * cnode.data().txi()[1] +
                cnode.data().txi()[2] * cnode.data().txi()[2]);
    if (ltxi < 1e-12) FOUR_C_THROW("tangent txi length is zero!!!");
    for (int j = 0; j < 3; ++j) cnode.data().txi()[j] /= ltxi;

    // teta follows from corkscrew rule (teta = n x txi)
    cnode.data().teta()[0] = cnode.mo_data().n()[1] * cnode.data().txi()[2] -
                             cnode.mo_data().n()[2] * cnode.data().txi()[1];
    cnode.data().teta()[1] = cnode.mo_data().n()[2] * cnode.data().txi()[0] -
                             cnode.mo_data().n()[0] * cnode.data().txi()[2];
    cnode.data().teta()[2] = cnode.mo_data().n()[0] * cnode.data().txi()[1] -
                             cnode.mo_data().n()[1] * cnode.data().txi()[0];
  }


  //------------------------------------------------------------------
  for (auto p = normallin[0].begin(); p != normallin[0].end(); ++p)
  {
    (cnode.data().get_deriv_n()[0])[p->first] -= (p->second);
  }
  for (auto p = normallin[1].begin(); p != normallin[1].end(); ++p)
  {
    (cnode.data().get_deriv_n()[1])[p->first] -= (p->second);
  }
  for (auto p = normallin[2].begin(); p != normallin[2].end(); ++p)
  {
    (cnode.data().get_deriv_n()[2])[p->first] -= (p->second);
  }

  // normalize directional derivative
  // (length differs for weighted/unweighted case bot not the procedure!)
  // (be careful with reference / copy of derivative maps!)
  Core::Gen::Pairedvector<int, double>& derivnx = cnode.data().get_deriv_n()[0];
  Core::Gen::Pairedvector<int, double>& derivny = cnode.data().get_deriv_n()[1];
  Core::Gen::Pairedvector<int, double>& derivnz = cnode.data().get_deriv_n()[2];
  Core::Gen::Pairedvector<int, double> cderivnx = cnode.data().get_deriv_n()[0];
  Core::Gen::Pairedvector<int, double> cderivny = cnode.data().get_deriv_n()[1];
  Core::Gen::Pairedvector<int, double> cderivnz = cnode.data().get_deriv_n()[2];
  const double nxnx = cnode.mo_data().n()[0] * cnode.mo_data().n()[0];
  const double nxny = cnode.mo_data().n()[0] * cnode.mo_data().n()[1];
  const double nxnz = cnode.mo_data().n()[0] * cnode.mo_data().n()[2];
  const double nyny = cnode.mo_data().n()[1] * cnode.mo_data().n()[1];
  const double nynz = cnode.mo_data().n()[1] * cnode.mo_data().n()[2];
  const double nznz = cnode.mo_data().n()[2] * cnode.mo_data().n()[2];

  // build a vector with all keys from x,y,z maps
  // (we need this in order not to miss any entry!)
  std::vector<int> allkeysn;
  for (const auto& p : derivnx)
  {
    bool found = false;
    for (int j : allkeysn)
      if ((p.first) == j) found = true;
    if (!found) allkeysn.push_back(p.first);
  }
  for (const auto& p : derivny)
  {
    bool found = false;
    for (int j : allkeysn)
      if ((p.first) == j) found = true;
    if (!found) allkeysn.push_back(p.first);
  }
  for (const auto& p : derivnz)
  {
    bool found = false;
    for (int j : allkeysn)
      if ((p.first) == j) found = true;
    if (!found) allkeysn.push_back(p.first);
  }

  // normalize x-components
  for (int j : allkeysn)
  {
    double val = cderivnx[j];
    derivnx[j] = (val - nxnx * val - nxny * cderivny[j] - nxnz * cderivnz[j]) / length;
  }

  // normalize y-components
  for (int j : allkeysn)
  {
    double val = cderivny[j];
    derivny[j] = (val - nxny * cderivnx[j] - nyny * val - nynz * cderivnz[j]) / length;
  }

  // normalize z-components
  for (int j : allkeysn)
  {
    double val = cderivnz[j];
    derivnz[j] = (val - nxnz * cderivnx[j] - nynz * cderivny[j] - nznz * val) / length;
  }

  //------------------------------------------------------------------
  // 2D Tangent!
  if (cnode.num_dof() == 2)
  {
    for (auto p = cnode.data().get_deriv_n()[1].begin(); p != cnode.data().get_deriv_n()[1].end();
        ++p)
    {
      (cnode.data().get_deriv_txi()[0])[p->first] -= (p->second);
    }
    for (auto p = cnode.data().get_deriv_n()[0].begin(); p != cnode.data().get_deriv_n()[0].end();
        ++p)
    {
      (cnode.data().get_deriv_txi()[1])[p->first] += (p->second);
    }
  }
  // 3D Tangent!
  else
  {
    // unnormalized tangent derivative txi
    // use definitions for txi from BuildAveragedNormal()
    if (abs(cnode.mo_data().n()[0]) > 1.0e-6 || abs(cnode.mo_data().n()[1]) > 1.0e-6)
    {
      Core::Gen::Pairedvector<int, double>& derivtxix = cnode.data().get_deriv_txi()[0];
      Core::Gen::Pairedvector<int, double>& derivtxiy = cnode.data().get_deriv_txi()[1];

      for (const auto& p : derivny)
      {
        derivtxix[p.first] -= (p.second);
      }

      for (const auto& p : derivnx)
      {
        derivtxiy[p.first] += (p.second);
      }
    }
    else
    {
      Core::Gen::Pairedvector<int, double>& derivtxiy = cnode.data().get_deriv_txi()[1];
      Core::Gen::Pairedvector<int, double>& derivtxiz = cnode.data().get_deriv_txi()[2];

      for (const auto& p : derivnz)
      {
        derivtxiy[p.first] -= (p.second);
      }

      for (const auto& p : derivny)
      {
        derivtxiz[p.first] += (p.second);
      }
    }

    if (ltxi < 1e-12)
    {
      FOUR_C_THROW("tangent txi length is zero!!!");
    }

    // normalize txi directional derivative
    // (identical to normalization of normal derivative)
    Core::Gen::Pairedvector<int, double>& derivtxix = cnode.data().get_deriv_txi()[0];
    Core::Gen::Pairedvector<int, double>& derivtxiy = cnode.data().get_deriv_txi()[1];
    Core::Gen::Pairedvector<int, double>& derivtxiz = cnode.data().get_deriv_txi()[2];
    Core::Gen::Pairedvector<int, double> cderivtxix = cnode.data().get_deriv_txi()[0];
    Core::Gen::Pairedvector<int, double> cderivtxiy = cnode.data().get_deriv_txi()[1];
    Core::Gen::Pairedvector<int, double> cderivtxiz = cnode.data().get_deriv_txi()[2];
    const double txtx = cnode.data().txi()[0] * cnode.data().txi()[0];
    const double txty = cnode.data().txi()[0] * cnode.data().txi()[1];
    const double txtz = cnode.data().txi()[0] * cnode.data().txi()[2];
    const double tyty = cnode.data().txi()[1] * cnode.data().txi()[1];
    const double tytz = cnode.data().txi()[1] * cnode.data().txi()[2];
    const double tztz = cnode.data().txi()[2] * cnode.data().txi()[2];

    // build a vector with all keys from x,y,z maps
    // (we need this in order not to miss any entry!)
    std::vector<int> allkeyst;
    for (const auto& p : derivtxix)
    {
      bool found = false;
      for (int j : allkeyst)
      {
        if ((p.first) == j)
        {
          found = true;
        }
      }
      if (!found)
      {
        allkeyst.push_back(p.first);
      }
    }
    for (const auto& p : derivtxiy)
    {
      bool found = false;
      for (int j : allkeyst)
      {
        if ((p.first) == j)
        {
          found = true;
        }
      }
      if (!found)
      {
        allkeyst.push_back(p.first);
      }
    }
    for (const auto& p : derivtxiz)
    {
      bool found = false;
      for (int j : allkeyst)
      {
        if ((p.first) == j)
        {
          found = true;
        }
      }
      if (!found)
      {
        allkeyst.push_back(p.first);
      }
    }

    // normalize x-components
    for (int j : allkeyst)
    {
      double val = cderivtxix[j];
      derivtxix[j] = (val - txtx * val - txty * cderivtxiy[j] - txtz * cderivtxiz[j]) / ltxi;
    }

    // normalize y-components
    for (int j : allkeyst)
    {
      double val = cderivtxiy[j];
      derivtxiy[j] = (val - txty * cderivtxix[j] - tyty * val - tytz * cderivtxiz[j]) / ltxi;
    }

    // normalize z-components
    for (int j : allkeyst)
    {
      double val = cderivtxiz[j];
      derivtxiz[j] = (val - txtz * cderivtxix[j] - tytz * cderivtxiy[j] - tztz * val) / ltxi;
    }

    // get normalized tangent derivative teta
    // use corkscrew rule from BuildAveragedNormal()
    Core::Gen::Pairedvector<int, double>& derivtetax = cnode.data().get_deriv_teta()[0];
    Core::Gen::Pairedvector<int, double>& derivtetay = cnode.data().get_deriv_teta()[1];
    Core::Gen::Pairedvector<int, double>& derivtetaz = cnode.data().get_deriv_teta()[2];

    for (const auto& p : derivnx)
    {
      derivtetay[p.first] -= cnode.data().txi()[2] * (p.second);
      derivtetaz[p.first] += cnode.data().txi()[1] * (p.second);
    }
    for (const auto& p : derivny)
    {
      derivtetax[p.first] += cnode.data().txi()[2] * (p.second);
      derivtetaz[p.first] -= cnode.data().txi()[0] * (p.second);
    }
    for (const auto& p : derivnz)
    {
      derivtetax[p.first] -= cnode.data().txi()[1] * (p.second);
      derivtetay[p.first] += cnode.data().txi()[0] * (p.second);
    }
    for (const auto& p : derivtxix)
    {
      derivtetay[p.first] += cnode.mo_data().n()[2] * (p.second);
      derivtetaz[p.first] -= cnode.mo_data().n()[1] * (p.second);
    }
    for (const auto& p : derivtxiy)
    {
      derivtetax[p.first] -= cnode.mo_data().n()[2] * (p.second);
      derivtetaz[p.first] += cnode.mo_data().n()[0] * (p.second);
    }
    for (const auto& p : derivtxiz)
    {
      derivtetax[p.first] += cnode.mo_data().n()[1] * (p.second);
      derivtetay[p.first] -= cnode.mo_data().n()[0] * (p.second);
    }
  }
}


/*----------------------------------------------------------------------*
 |  export nodal normals (public)                             popp 11/10|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::export_nodal_normals() const
{
  // create empty data objects
  std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>> triad;
  std::map<int, std::map<int, std::vector<double>>> node_data_collection;

  // Define keys for the nodal data map
  enum NodalQuantity : std::uint8_t
  {
    N_X_KEY,
    N_Y_KEY,
    N_Z_KEY,
    TXI_X_KEY,
    TXI_Y_KEY,
    TXI_Z_KEY,
    TETA_X_KEY,
    TETA_Y_KEY,
    TETA_Z_KEY,
    N_X_VAL,
    N_Y_VAL,
    N_Z_VAL,
    TXI_X_VAL,
    TXI_Y_VAL,
    TXI_Z_VAL,
    TETA_X_VAL,
    TETA_Y_VAL,
    TETA_Z_VAL
  };

  // Build info on row map
  for (int i = 0; i < snoderowmapbound_->num_my_elements(); ++i)
  {
    int gid = snoderowmapbound_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Node* cnode = dynamic_cast<Node*>(node);

    // Fill nodal matrix
    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> loc =
        std::make_shared<Core::LinAlg::SerialDenseMatrix>(3, 3);
    (*loc)(0, 0) = cnode->mo_data().n()[0];
    (*loc)(1, 0) = cnode->mo_data().n()[1];
    (*loc)(2, 0) = cnode->mo_data().n()[2];
    (*loc)(0, 1) = cnode->data().txi()[0];
    (*loc)(1, 1) = cnode->data().txi()[1];
    (*loc)(2, 1) = cnode->data().txi()[2];
    (*loc)(0, 2) = cnode->data().teta()[0];
    (*loc)(1, 2) = cnode->data().teta()[1];
    (*loc)(2, 2) = cnode->data().teta()[2];

    triad[gid] = loc;

    // Fill nodal derivative vectors
    std::vector<Core::Gen::Pairedvector<int, double>>& derivn = cnode->data().get_deriv_n();
    std::vector<Core::Gen::Pairedvector<int, double>>& derivtxi = cnode->data().get_deriv_txi();
    std::vector<Core::Gen::Pairedvector<int, double>>& derivteta = cnode->data().get_deriv_teta();

    std::map<int, std::vector<double>>& node_data = node_data_collection[gid];

    for (const auto& _iter : derivn[0])
    {
      node_data[N_X_KEY].push_back(_iter.first);
      node_data[N_X_VAL].push_back(_iter.second);
    }
    for (const auto& _iter : derivn[1])
    {
      node_data[N_Y_KEY].push_back(_iter.first);
      node_data[N_Y_VAL].push_back(_iter.second);
    }
    for (const auto& _iter : derivn[2])
    {
      node_data[N_Z_KEY].push_back(_iter.first);
      node_data[N_Z_VAL].push_back(_iter.second);
    }

    for (const auto& _iter : derivtxi[0])
    {
      node_data[TXI_X_KEY].push_back(_iter.first);
      node_data[TXI_X_VAL].push_back(_iter.second);
    }
    for (const auto& _iter : derivtxi[1])
    {
      node_data[TXI_Y_KEY].push_back(_iter.first);
      node_data[TXI_Y_VAL].push_back(_iter.second);
    }
    for (const auto& _iter : derivtxi[2])
    {
      node_data[TXI_Z_KEY].push_back(_iter.first);
      node_data[TXI_Z_VAL].push_back(_iter.second);
    }

    for (const auto& _iter : derivteta[0])
    {
      node_data[TETA_X_KEY].push_back(_iter.first);
      node_data[TETA_X_VAL].push_back(_iter.second);
    }
    for (const auto& _iter : derivteta[1])
    {
      node_data[TETA_Y_KEY].push_back(_iter.first);
      node_data[TETA_Y_VAL].push_back(_iter.second);
    }
    for (const auto& _iter : derivteta[2])
    {
      node_data[TETA_Z_KEY].push_back(_iter.first);
      node_data[TETA_Z_VAL].push_back(_iter.second);
    }
  }

  // communicate from slave node row to column map
  Core::Communication::Exporter& ex = interface_data_->exporter();
  ex.do_export(triad);
  ex.do_export(node_data_collection);

  // extract info on column map
  for (int i = 0; i < snodecolmapbound_->num_my_elements(); ++i)
  {
    // only do something for ghosted nodes
    int gid = snodecolmapbound_->gid(i);
    if (snoderowmapbound_->my_gid(gid)) continue;

    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Node* cnode = dynamic_cast<Node*>(node);

    // extract info
    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> loc = triad[gid];
    cnode->mo_data().n()[0] = (*loc)(0, 0);
    cnode->mo_data().n()[1] = (*loc)(1, 0);
    cnode->mo_data().n()[2] = (*loc)(2, 0);
    cnode->data().txi()[0] = (*loc)(0, 1);
    cnode->data().txi()[1] = (*loc)(1, 1);
    cnode->data().txi()[2] = (*loc)(2, 1);
    cnode->data().teta()[0] = (*loc)(0, 2);
    cnode->data().teta()[1] = (*loc)(1, 2);
    cnode->data().teta()[2] = (*loc)(2, 2);

    // extract derivative info
    std::vector<Core::Gen::Pairedvector<int, double>>& derivn = cnode->data().get_deriv_n();
    std::vector<Core::Gen::Pairedvector<int, double>>& derivtxi = cnode->data().get_deriv_txi();
    std::vector<Core::Gen::Pairedvector<int, double>>& derivteta = cnode->data().get_deriv_teta();

    std::map<int, std::vector<double>>& node_data = node_data_collection[gid];
    size_t linsize = static_cast<size_t>(cnode->get_linsize()) + node_data[N_X_KEY].size();

    // Clear and resize operations on vectors derivn, derivtxi, derivteta
    for (auto& k : derivn)
    {
      k.clear();
    }
    derivn.resize(3, linsize);

    for (auto& k : derivtxi)
    {
      k.clear();
    }
    derivtxi.resize(3, linsize);

    for (auto& k : derivteta)
    {
      k.clear();
    }
    derivteta.resize(3, linsize);

    // Assignments using node_data arrays
    for (size_t k = 0; k < node_data[N_X_KEY].size(); ++k)
    {
      (cnode->data().get_deriv_n()[0])[node_data[N_X_KEY][k]] = node_data[N_X_VAL][k];
    }

    for (size_t k = 0; k < node_data[N_Y_KEY].size(); ++k)
    {
      (cnode->data().get_deriv_n()[1])[node_data[N_Y_KEY][k]] = node_data[N_Y_VAL][k];
    }

    for (size_t k = 0; k < node_data[N_Z_KEY].size(); ++k)
    {
      (cnode->data().get_deriv_n()[2])[node_data[N_Z_KEY][k]] = node_data[N_Z_VAL][k];
    }

    for (size_t k = 0; k < node_data[TXI_X_KEY].size(); ++k)
    {
      (cnode->data().get_deriv_txi()[0])[node_data[TXI_X_KEY][k]] = node_data[TXI_X_VAL][k];
    }

    for (size_t k = 0; k < node_data[TXI_Y_KEY].size(); ++k)
    {
      (cnode->data().get_deriv_txi()[1])[node_data[TXI_Y_KEY][k]] = node_data[TXI_Y_VAL][k];
    }

    for (size_t k = 0; k < node_data[TXI_Z_KEY].size(); ++k)
    {
      (cnode->data().get_deriv_txi()[2])[node_data[TXI_Z_KEY][k]] = node_data[TXI_Z_VAL][k];
    }

    for (size_t k = 0; k < node_data[TETA_X_KEY].size(); ++k)
    {
      (cnode->data().get_deriv_teta()[0])[node_data[TETA_X_KEY][k]] = node_data[TETA_X_VAL][k];
    }

    for (size_t k = 0; k < node_data[TETA_Y_KEY].size(); ++k)
    {
      (cnode->data().get_deriv_teta()[1])[node_data[TETA_Y_KEY][k]] = node_data[TETA_Y_VAL][k];
    }

    for (size_t k = 0; k < node_data[TETA_Z_KEY].size(); ++k)
    {
      (cnode->data().get_deriv_teta()[2])[node_data[TETA_Z_KEY][k]] = node_data[TETA_Z_VAL][k];
    }
  }
}

/*----------------------------------------------------------------------*
 |  Search for potentially contacting sl/ma pairs (public)    popp 10/08|
 *----------------------------------------------------------------------*/
bool CONTACT::Interface::evaluate_search_binarytree()
{
  // *********************************************************************
  // self contact:
  // *********************************************************************
  // We call evaluate_search(), which does both, the bottom-up update (on the whole interface) and
  // the search. Then the dynamic master/slave assignment routine update_master_slave_sets() is
  // called and the new slave nodes' data containers are initialized.
  // *********************************************************************
  if (self_contact())
  {
    // evaluate search itself
    binarytreeself_->evaluate_search();

    // update master/slave sets of interface
    update_master_slave_sets();

    // initialize node data container
    // (include slave side boundary nodes / crosspoints)
    for (int i = 0; i < slave_col_nodes_bound()->num_my_elements(); ++i)
    {
      int gid = slave_col_nodes_bound()->gid(i);
      Core::Nodes::Node* node = discret().g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid {}", gid);
      auto* mnode = dynamic_cast<Mortar::Node*>(node);

      // initialize container if not yet initialized before
      mnode->initialize_data_container();
    }

    // no initialization of element data container as this would
    // possibly destroy the information on search elements again
    // (this was already done in set_element_areas())
  }
  else
  {
    // call mortar routine
    Mortar::Interface::evaluate_search_binarytree();
  }

  return true;
}

/*----------------------------------------------------------------------*
 |  evaluate coupling type segment-to-line coupl             farah 07/16|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::evaluate_stl()
{
  // check
  if (n_dim() == 2) FOUR_C_THROW("LTS algorithm only for 3D simulations!");

  // loop over slave elements
  for (int i = 0; i < selecolmap_->num_my_elements(); ++i)
  {
    int gid1 = selecolmap_->gid(i);
    Core::Elements::Element* ele1 = idiscret_->g_element(gid1);
    if (!ele1) FOUR_C_THROW("Cannot find slave element with gid %", gid1);
    auto* selement = dynamic_cast<Element*>(ele1);

    // guarantee uniqueness
    std::set<std::pair<int, int>> donebefore;

    // loop over found meles
    for (int j = 0; j < selement->mo_data().num_search_elements(); ++j)
    {
      int gid2 = selement->mo_data().search_elements()[j];
      Core::Elements::Element* ele2 = idiscret_->g_element(gid2);
      if (!ele2) FOUR_C_THROW("Cannot find master element with gid %", gid2);
      auto* melement = dynamic_cast<Element*>(ele2);

      if (melement->shape() == Core::FE::CellType::quad4)
      {
        for (int nodeId = 0; nodeId < 4; ++nodeId)
        {
          int nodeIds[2] = {0, 0};
          int nodeLIds[2] = {0, 0};

          if (nodeId == 0)
          {
            nodeIds[0] = melement->node_ids()[0];
            nodeIds[1] = melement->node_ids()[1];

            nodeLIds[0] = 0;
            nodeLIds[1] = 1;
          }
          else if (nodeId == 1)
          {
            nodeIds[0] = melement->node_ids()[1];
            nodeIds[1] = melement->node_ids()[2];

            nodeLIds[0] = 1;
            nodeLIds[1] = 2;
          }
          else if (nodeId == 2)
          {
            nodeIds[0] = melement->node_ids()[2];
            nodeIds[1] = melement->node_ids()[3];

            nodeLIds[0] = 2;
            nodeLIds[1] = 3;
          }
          else if (nodeId == 3)
          {
            nodeIds[0] = melement->node_ids()[3];
            nodeIds[1] = melement->node_ids()[0];

            nodeLIds[0] = 3;
            nodeLIds[1] = 0;
          }

          // create pair
          std::pair<int, int> actIDs = std::pair<int, int>(nodeIds[0], nodeIds[1]);
          std::pair<int, int> actIDstw = std::pair<int, int>(nodeIds[1], nodeIds[0]);

          // check if processed before
          auto iter = donebefore.find(actIDs);
          auto itertw = donebefore.find(actIDstw);

          // if not then create ele
          if (iter == donebefore.end() and itertw == donebefore.end())
          {
            // add to set of processed nodes
            donebefore.insert(actIDs);
            donebefore.insert(actIDstw);

            // create line ele:
            std::shared_ptr<Mortar::Element> lineEle = std::make_shared<Mortar::Element>(
                j, melement->owner(), Core::FE::CellType::line2, 2, nodeIds, false);

            // get nodes
            std::array<Core::Nodes::Node*, 2> nodes = {
                melement->nodes()[nodeLIds[0]], melement->nodes()[nodeLIds[1]]};
            lineEle->build_nodal_pointers(nodes.data());

            // init data container for dual shapes
            lineEle->initialize_data_container();

            std::vector<Element*> seleElements;
            seleElements.push_back(selement);

            // create coupling object
            LineToSurfaceCoupling3d coup(*idiscret_, 3, interface_params(), *melement, lineEle,
                seleElements, LineToSurfaceCoupling3d::stl);

            // perform evaluate!
            coup.evaluate_coupling();
          }
        }  // end edge loop
      }
      else
      {
        FOUR_C_THROW("LTS only for quad4!");
      }

    }  // end found mele loop
  }  // end slave ele loop
}

/*----------------------------------------------------------------------*
 |  evaluate coupling type line-to-segment coupl             farah 07/16|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::evaluate_lts()
{
  // check
  if (n_dim() == 2) FOUR_C_THROW("LTS algorithm only for 3D simulations!");

  // guarantee uniqueness
  std::set<std::pair<int, int>> donebefore;

  // loop over slave elements
  for (int i = 0; i < selecolmap_->num_my_elements(); ++i)
  {
    int gid1 = selecolmap_->gid(i);
    Core::Elements::Element* ele1 = idiscret_->g_element(gid1);
    if (!ele1) FOUR_C_THROW("Cannot find slave element with gid %", gid1);
    auto* selement = dynamic_cast<Element*>(ele1);

    // ele check
    if (selement->shape() != Core::FE::CellType::quad4 and
        selement->shape() != Core::FE::CellType::tri3)
      FOUR_C_THROW("LTS algorithm only for tri3/quad4!");

    // empty vector of master element pointers
    std::vector<std::shared_ptr<Mortar::Element>> lineElements;
    std::vector<Element*> meleElements;

    // compute slave normal
    double slaveN[3] = {0.0, 0.0, 0.0};
    double loccenter[2] = {0.0, 0.0};

    Core::FE::CellType dt = selement->shape();
    if (dt == Core::FE::CellType::tri3 || dt == Core::FE::CellType::tri6)
    {
      loccenter[0] = 1.0 / 3.0;
      loccenter[1] = 1.0 / 3.0;
    }
    else if (dt == Core::FE::CellType::quad4 || dt == Core::FE::CellType::quad8 ||
             dt == Core::FE::CellType::quad9)
    {
      loccenter[0] = 0.0;
      loccenter[1] = 0.0;
    }
    else
    {
      FOUR_C_THROW("auxiliary_plane called for unknown element type");
    }

    // we then compute the unit normal vector at the element center
    selement->compute_unit_normal_at_xi(loccenter, slaveN);

    // loop over the candidate master elements of sele_
    // use slave element's candidate list SearchElements !!!
    for (int j = 0; j < selement->mo_data().num_search_elements(); ++j)
    {
      int gid2 = selement->mo_data().search_elements()[j];
      Core::Elements::Element* ele2 = idiscret_->g_element(gid2);
      if (!ele2) FOUR_C_THROW("Cannot find master element with gid %", gid2);
      auto* melement = dynamic_cast<Element*>(ele2);

      // check orientation
      // compute slave normal
      double masterN[3] = {0.0, 0.0, 0.0};
      double loccenterM[2] = {0.0, 0.0};

      Core::FE::CellType dt = melement->shape();
      if (dt == Core::FE::CellType::tri3 || dt == Core::FE::CellType::tri6)
      {
        loccenterM[0] = 1.0 / 3.0;
        loccenterM[1] = 1.0 / 3.0;
      }
      else if (dt == Core::FE::CellType::quad4 || dt == Core::FE::CellType::quad8 ||
               dt == Core::FE::CellType::quad9)
      {
        loccenterM[0] = 0.0;
        loccenterM[1] = 0.0;
      }
      else
      {
        FOUR_C_THROW("auxiliary_plane called for unknown element type");
      }

      // we then compute the unit normal vector at the element center
      melement->compute_unit_normal_at_xi(loccenterM, masterN);

      double scaprod = slaveN[0] * masterN[0] + slaveN[1] * masterN[1] + slaveN[2] * masterN[2];

      // tolerance for line clipping
      const double sminedge = selement->min_edge_size();
      const double mminedge = melement->min_edge_size();
      const double tol = 0.001 * std::min(sminedge, mminedge);
      if (abs(scaprod) < tol) continue;

      // if orientation is okay
      meleElements.push_back(melement);
    }

    // no valid maste elements?
    if (meleElements.size() < 1) continue;

    // loop over slave edges -> match node number for tri3/quad4
    for (int j = 0; j < selement->num_node(); ++j)
    {
      int nodeIds[2] = {0, 0};
      int nodeLIds[2] = {0, 0};

      if (selement->shape() == Core::FE::CellType::quad4)
      {
        if (j == 0)
        {
          nodeIds[0] = selement->node_ids()[0];
          nodeIds[1] = selement->node_ids()[1];

          nodeLIds[0] = 0;
          nodeLIds[1] = 1;
        }
        else if (j == 1)
        {
          nodeIds[0] = selement->node_ids()[1];
          nodeIds[1] = selement->node_ids()[2];

          nodeLIds[0] = 1;
          nodeLIds[1] = 2;
        }
        else if (j == 2)
        {
          nodeIds[0] = selement->node_ids()[2];
          nodeIds[1] = selement->node_ids()[3];

          nodeLIds[0] = 2;
          nodeLIds[1] = 3;
        }
        else if (j == 3)
        {
          nodeIds[0] = selement->node_ids()[3];
          nodeIds[1] = selement->node_ids()[0];

          nodeLIds[0] = 3;
          nodeLIds[1] = 0;
        }
        else
        {
          FOUR_C_THROW("loop counter and edge number do not match!");
        }
      }
      else if (selement->shape() == Core::FE::CellType::tri3)
      {
        if (j == 0)
        {
          nodeIds[0] = selement->node_ids()[0];
          nodeIds[1] = selement->node_ids()[1];

          nodeLIds[0] = 0;
          nodeLIds[1] = 1;
        }
        else if (j == 1)
        {
          nodeIds[0] = selement->node_ids()[1];
          nodeIds[1] = selement->node_ids()[2];

          nodeLIds[0] = 1;
          nodeLIds[1] = 2;
        }
        else if (j == 2)
        {
          nodeIds[0] = selement->node_ids()[2];
          nodeIds[1] = selement->node_ids()[0];

          nodeLIds[0] = 2;
          nodeLIds[1] = 0;
        }
        else
        {
          FOUR_C_THROW("loop counter and edge number do not match!");
        }
      }

      // check if both nodes on edge geometry
      bool node0Edge = dynamic_cast<Mortar::Node*>(selement->nodes()[nodeLIds[0]])->is_on_edge();
      bool node1Edge = dynamic_cast<Mortar::Node*>(selement->nodes()[nodeLIds[1]])->is_on_edge();

      if (nonSmoothContact_ and (!node0Edge or !node1Edge)) continue;

      // create pair
      std::pair<int, int> actIDs = std::pair<int, int>(nodeIds[0], nodeIds[1]);
      std::pair<int, int> actIDstw = std::pair<int, int>(nodeIds[1], nodeIds[0]);

      // check if processed before
      auto iter = donebefore.find(actIDs);
      auto itertw = donebefore.find(actIDstw);

      // if not then create ele
      if (iter == donebefore.end() and itertw == donebefore.end())
      {
        // add to set of processed nodes
        donebefore.insert(actIDs);
        donebefore.insert(actIDstw);

        // create line ele:
        std::shared_ptr<Mortar::Element> lineEle = std::make_shared<Mortar::Element>(
            j, selement->owner(), Core::FE::CellType::line2, 2, nodeIds, false);

        // get nodes
        std::array<Core::Nodes::Node*, 2> nodes = {
            selement->nodes()[nodeLIds[0]], selement->nodes()[nodeLIds[1]]};
        lineEle->build_nodal_pointers(nodes.data());

        // init data container for dual shapes
        lineEle->initialize_data_container();

        // push back into vector
        lineElements.push_back(lineEle);
      }
    }  // end edge loop

    // loop over all created line elements
    for (auto& lineElement : lineElements)
    {
      // create coupling object
      LineToSurfaceCoupling3d coup(*idiscret_, 3, interface_params(), *selement, lineElement,
          meleElements, LineToSurfaceCoupling3d::lts);

      // perform evaluate!
      coup.evaluate_coupling();
    }
  }  // slave ele loop
}

/*----------------------------------------------------------------------*
 |  evaluate coupling type line-to-line coupl                farah 07/16|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::evaluate_ltl()
{
  // check
  if (n_dim() == 2) FOUR_C_THROW("LTL algorithm only for 3D simulations!");

  // guarantee uniqueness of slave edges
  std::set<std::pair<int, int>> donebeforeS;

  // loop over slave elements
  for (int i = 0; i < selecolmap_->num_my_elements(); ++i)
  {
    int gid1 = selecolmap_->gid(i);
    Core::Elements::Element* ele1 = idiscret_->g_element(gid1);
    if (!ele1) FOUR_C_THROW("Cannot find slave element with gid %", gid1);
    auto* selement = dynamic_cast<Element*>(ele1);

    // empty vector of slave element pointers
    std::vector<std::shared_ptr<Mortar::Element>> lineElementsS;

    if (selement->shape() == Core::FE::CellType::quad4)
    {
      for (int j = 0; j < 4; ++j)
      {
        int nodeIds[2] = {0, 0};
        int nodeLIds[2] = {0, 0};

        if (j == 0)
        {
          nodeIds[0] = selement->node_ids()[0];
          nodeIds[1] = selement->node_ids()[1];

          nodeLIds[0] = 0;
          nodeLIds[1] = 1;
        }
        else if (j == 1)
        {
          nodeIds[0] = selement->node_ids()[1];
          nodeIds[1] = selement->node_ids()[2];

          nodeLIds[0] = 1;
          nodeLIds[1] = 2;
        }
        else if (j == 2)
        {
          nodeIds[0] = selement->node_ids()[2];
          nodeIds[1] = selement->node_ids()[3];

          nodeLIds[0] = 2;
          nodeLIds[1] = 3;
        }
        else if (j == 3)
        {
          nodeIds[0] = selement->node_ids()[3];
          nodeIds[1] = selement->node_ids()[0];

          nodeLIds[0] = 3;
          nodeLIds[1] = 0;
        }

        // check if both nodes on edge geometry
        bool node0Edge = dynamic_cast<Mortar::Node*>(selement->nodes()[nodeLIds[0]])->is_on_edge();
        bool node1Edge = dynamic_cast<Mortar::Node*>(selement->nodes()[nodeLIds[1]])->is_on_edge();

        if (!node0Edge or !node1Edge) continue;

        // create pair
        std::pair<int, int> actIDs = std::pair<int, int>(nodeIds[0], nodeIds[1]);
        std::pair<int, int> actIDstw = std::pair<int, int>(nodeIds[1], nodeIds[0]);

        // check if processed before
        auto iter = donebeforeS.find(actIDs);
        auto itertw = donebeforeS.find(actIDstw);

        // if not then create ele
        if (iter == donebeforeS.end() and itertw == donebeforeS.end())
        {
          // add to set of processed nodes
          donebeforeS.insert(actIDs);
          donebeforeS.insert(actIDstw);

          // create line ele:
          std::shared_ptr<Mortar::Element> lineEle = std::make_shared<Mortar::Element>(
              j, selement->owner(), Core::FE::CellType::line2, 2, nodeIds, false);

          // get nodes
          std::array<Core::Nodes::Node*, 2> nodes = {
              selement->nodes()[nodeLIds[0]], selement->nodes()[nodeLIds[1]]};
          lineEle->build_nodal_pointers(nodes.data());

          // init data container for dual shapes
          lineEle->initialize_data_container();

          // push back into vector
          lineElementsS.push_back(lineEle);
        }
      }  // end edge loop
    }
    else if (selement->shape() == Core::FE::CellType::tri3)
    {
      for (int j = 0; j < 3; ++j)
      {
        int nodeIds[2] = {0, 0};
        int nodeLIds[2] = {0, 0};

        if (j == 0)
        {
          nodeIds[0] = selement->node_ids()[0];
          nodeIds[1] = selement->node_ids()[1];

          nodeLIds[0] = 0;
          nodeLIds[1] = 1;
        }
        else if (j == 1)
        {
          nodeIds[0] = selement->node_ids()[1];
          nodeIds[1] = selement->node_ids()[2];

          nodeLIds[0] = 1;
          nodeLIds[1] = 2;
        }
        else if (j == 2)
        {
          nodeIds[0] = selement->node_ids()[2];
          nodeIds[1] = selement->node_ids()[0];

          nodeLIds[0] = 2;
          nodeLIds[1] = 0;
        }

        // check if both nodes on edge geometry
        bool node0Edge = dynamic_cast<Mortar::Node*>(selement->nodes()[nodeLIds[0]])->is_on_edge();
        bool node1Edge = dynamic_cast<Mortar::Node*>(selement->nodes()[nodeLIds[1]])->is_on_edge();

        if (!node0Edge or !node1Edge) continue;

        // create pair
        std::pair<int, int> actIDs = std::pair<int, int>(nodeIds[0], nodeIds[1]);
        std::pair<int, int> actIDstw = std::pair<int, int>(nodeIds[1], nodeIds[0]);

        // check if processed before
        auto iter = donebeforeS.find(actIDs);
        auto itertw = donebeforeS.find(actIDstw);

        // if not then create ele
        if (iter == donebeforeS.end() and itertw == donebeforeS.end())
        {
          // add to set of processed nodes
          donebeforeS.insert(actIDs);
          donebeforeS.insert(actIDstw);

          // create line ele:
          std::shared_ptr<Mortar::Element> lineEle = std::make_shared<Mortar::Element>(
              j, selement->owner(), Core::FE::CellType::line2, 2, nodeIds, false);

          // get nodes
          std::array<Core::Nodes::Node*, 2> nodes = {
              selement->nodes()[nodeLIds[0]], selement->nodes()[nodeLIds[1]]};
          lineEle->build_nodal_pointers(nodes.data());

          // init data container for dual shapes
          lineEle->initialize_data_container();

          // push back into vector
          lineElementsS.push_back(lineEle);
        }
      }  // end edge loop
    }
    else
    {
      FOUR_C_THROW("LTL only for quad4 and tri3!");
    }

    // guarantee uniqueness of master edges
    std::set<std::pair<int, int>> donebeforeM;

    // empty vector of slave element pointers
    std::vector<std::shared_ptr<Mortar::Element>> lineElementsM;

    // loop over the candidate master elements of sele_
    // use slave element's candidate list SearchElements !!!
    for (int k = 0; k < selement->mo_data().num_search_elements(); ++k)
    {
      int gid2 = selement->mo_data().search_elements()[k];
      Core::Elements::Element* ele2 = idiscret_->g_element(gid2);
      if (!ele2) FOUR_C_THROW("Cannot find master element with gid %", gid2);
      auto* melement = dynamic_cast<Element*>(ele2);

      if (melement->shape() == Core::FE::CellType::quad4)
      {
        for (int j = 0; j < 4; ++j)
        {
          int nodeIds[2] = {0, 0};
          int nodeLIds[2] = {0, 0};

          if (j == 0)
          {
            nodeIds[0] = melement->node_ids()[0];
            nodeIds[1] = melement->node_ids()[1];

            nodeLIds[0] = 0;
            nodeLIds[1] = 1;
          }
          else if (j == 1)
          {
            nodeIds[0] = melement->node_ids()[1];
            nodeIds[1] = melement->node_ids()[2];

            nodeLIds[0] = 1;
            nodeLIds[1] = 2;
          }
          else if (j == 2)
          {
            nodeIds[0] = melement->node_ids()[2];
            nodeIds[1] = melement->node_ids()[3];

            nodeLIds[0] = 2;
            nodeLIds[1] = 3;
          }
          else if (j == 3)
          {
            nodeIds[0] = melement->node_ids()[3];
            nodeIds[1] = melement->node_ids()[0];

            nodeLIds[0] = 3;
            nodeLIds[1] = 0;
          }

          // check if both nodes on edge geometry
          bool node0Edge =
              dynamic_cast<Mortar::Node*>(melement->nodes()[nodeLIds[0]])->is_on_edge();
          bool node1Edge =
              dynamic_cast<Mortar::Node*>(melement->nodes()[nodeLIds[1]])->is_on_edge();

          if (!node0Edge or !node1Edge) continue;

          // create pair
          std::pair<int, int> actIDs = std::pair<int, int>(nodeIds[0], nodeIds[1]);
          std::pair<int, int> actIDstw = std::pair<int, int>(nodeIds[1], nodeIds[0]);

          // check if processed before
          auto iter = donebeforeM.find(actIDs);
          auto itertw = donebeforeM.find(actIDstw);

          // if not then create ele
          if (iter == donebeforeM.end() and itertw == donebeforeM.end())
          {
            // add to set of processed nodes
            donebeforeM.insert(actIDs);
            donebeforeM.insert(actIDstw);

            // create line ele:
            std::shared_ptr<Mortar::Element> lineEle = std::make_shared<Mortar::Element>(
                j, melement->owner(), Core::FE::CellType::line2, 2, nodeIds, false);

            // get nodes
            std::array<Core::Nodes::Node*, 2> nodes = {
                melement->nodes()[nodeLIds[0]], melement->nodes()[nodeLIds[1]]};
            lineEle->build_nodal_pointers(nodes.data());

            // init data container for dual shapes
            lineEle->initialize_data_container();

            // push back into vector
            lineElementsM.push_back(lineEle);
          }
        }  // end edge loop
      }
      else if (melement->shape() == Core::FE::CellType::tri3)
      {
        for (int j = 0; j < 3; ++j)
        {
          int nodeIds[2] = {0, 0};
          int nodeLIds[2] = {0, 0};

          if (j == 0)
          {
            nodeIds[0] = melement->node_ids()[0];
            nodeIds[1] = melement->node_ids()[1];

            nodeLIds[0] = 0;
            nodeLIds[1] = 1;
          }
          else if (j == 1)
          {
            nodeIds[0] = melement->node_ids()[1];
            nodeIds[1] = melement->node_ids()[2];

            nodeLIds[0] = 1;
            nodeLIds[1] = 2;
          }
          else if (j == 2)
          {
            nodeIds[0] = melement->node_ids()[2];
            nodeIds[1] = melement->node_ids()[0];

            nodeLIds[0] = 2;
            nodeLIds[1] = 0;
          }

          // check if both nodes on edge geometry
          bool node0Edge =
              dynamic_cast<Mortar::Node*>(melement->nodes()[nodeLIds[0]])->is_on_edge();
          bool node1Edge =
              dynamic_cast<Mortar::Node*>(melement->nodes()[nodeLIds[1]])->is_on_edge();

          if (!node0Edge or !node1Edge) continue;

          // create pair
          std::pair<int, int> actIDs = std::pair<int, int>(nodeIds[0], nodeIds[1]);
          std::pair<int, int> actIDstw = std::pair<int, int>(nodeIds[1], nodeIds[0]);

          // check if processed before
          auto iter = donebeforeM.find(actIDs);
          auto itertw = donebeforeM.find(actIDstw);

          // if not then create ele
          if (iter == donebeforeM.end() and itertw == donebeforeM.end())
          {
            // add to set of processed nodes
            donebeforeM.insert(actIDs);
            donebeforeM.insert(actIDstw);

            // create line ele:
            std::shared_ptr<Mortar::Element> lineEle = std::make_shared<Mortar::Element>(
                j, melement->owner(), Core::FE::CellType::line2, 2, nodeIds, false);

            // get nodes
            std::array<Core::Nodes::Node*, 2> nodes = {
                melement->nodes()[nodeLIds[0]], melement->nodes()[nodeLIds[1]]};
            lineEle->build_nodal_pointers(nodes.data());

            // init data container for dual shapes
            lineEle->initialize_data_container();

            // push back into vector
            lineElementsM.push_back(lineEle);
          }
        }  // end edge loop
      }
      else
      {
        FOUR_C_THROW("LTL only for quad4 and tri3!");
      }
    }  // end found mele loop

    // loop over slave edges
    for (auto& s : lineElementsS)
    {
      // loop over master edges
      for (auto& m : lineElementsM)
      {
        // create coupling object
        LineToLineCouplingPoint3d coup(*idiscret_, 3, interface_params(), s, m);

        // perform evaluate!
        coup.evaluate_coupling();
      }
    }

  }  // end slave loop
}


/*----------------------------------------------------------------------*
 |  evaluate coupling type node-to-segment coupl             farah 02/16|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::evaluate_nts()
{
  // create one interpolator instance which is valid for all nodes!
  NTS::Interpolator interpolator(interface_params(), n_dim());

  // loop over slave nodes
  for (int i = 0; i < snoderowmap_->num_my_elements(); ++i)
  {
    int gid = snoderowmap_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    auto* mrtrnode = dynamic_cast<Mortar::Node*>(node);

    if (!mrtrnode->is_on_corner() and nonSmoothContact_) continue;

    if (mrtrnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
      FOUR_C_THROW("Node ownership inconsistency!");

    // vector with possible contacting master eles
    std::vector<Mortar::Element*> meles;

    // fill vector with possibly contacting meles
    find_master_elements(*mrtrnode, meles);

    // skip calculation if no meles vector is empty
    if (meles.size() < 1) continue;

    // call interpolation functions
    interpolator.interpolate(*mrtrnode, meles);
  }
}


/*----------------------------------------------------------------------*
 |  Integrate matrix M and gap g on slave/master overlaps     popp 11/08|
 *----------------------------------------------------------------------*/
bool CONTACT::Interface::mortar_coupling(Mortar::Element* sele, std::vector<Mortar::Element*> mele,
    const std::shared_ptr<Mortar::ParamsInterface>& mparams_ptr)
{
  // do stuff before the actual coupling is going to be evaluated
  pre_mortar_coupling(sele, mele, mparams_ptr);

  // increase counter of slave/master pairs
  smpairs_ += (int)mele.size();

  // check if quadratic interpolation is involved
  bool quadratic = false;
  if (sele->is_quad())
  {
    quadratic = true;
  }
  for (auto& m : mele)
  {
    if (m->is_quad())
    {
      quadratic = true;
    }
  }

  // *********************************************************************
  // do interface coupling within a new class
  // (projection slave and master, overlap detection, integration and
  // linearization of the Mortar matrix M)
  // ************************************************************** 2D ***
  if (n_dim() == 2)
  {
    // *************************************************** linear 2D ***
    // ************************************************ quadratic 2D ***
    // neither quadratic interpolation nor mixed linear and quadratic
    // interpolation need any special treatment in the 2d case

    // create Coupling2dManager
    CONTACT::Coupling2dManager coup(discret(), n_dim(), quadratic, interface_params(), sele, mele);
    // evaluate
    coup.evaluate_coupling(mparams_ptr);

    // increase counter of slave/master integration pairs and intcells
    smintpairs_ += (int)mele.size();
    intcells_ += (int)mele.size();
  }
  // ************************************************************** 3D ***
  else if (n_dim() == 3)
  {
    // *************************************************** linear 3D ***
    if (!quadratic)
    {
      // create Coupling3dManager
      CONTACT::Coupling3dManager coup(
          discret(), n_dim(), quadratic, interface_params(), sele, mele);
      // evaluate
      coup.evaluate_coupling(mparams_ptr);

      // increase counter of slave/master integration pairs and intcells
      smintpairs_ += (int)mele.size();
      intcells_ += coup.integration_cells();
    }

    // ************************************************** quadratic 3D ***
    else
    {
      // create Coupling3dQuadManager
      CONTACT::Coupling3dQuadManager coup(
          discret(), n_dim(), quadratic, interface_params(), sele, mele);
      // evaluate
      coup.evaluate_coupling(mparams_ptr);
    }  // quadratic
  }  // 3D
  else
  {
    FOUR_C_THROW("Dimension for Mortar coupling must be 2D or 3D!");
  }
  // *********************************************************************

  // do stuff after the coupling evaluation
  post_mortar_coupling(sele, mele, mparams_ptr);

  return true;
}

/*----------------------------------------------------------------------*
 |  Integrate penalty scaling factor kapp (public)            popp 11/09|
 *----------------------------------------------------------------------*/
bool CONTACT::Interface::integrate_kappa_penalty(CONTACT::Element& sele)
{
  // create correct integration limits
  double sxia[2] = {0.0, 0.0};
  double sxib[2] = {0.0, 0.0};
  if (sele.shape() == Core::FE::CellType::tri3 || sele.shape() == Core::FE::CellType::tri6)
  {
    // parameter space is [0,1] for triangles
    sxib[0] = 1.0;
    sxib[1] = 1.0;
  }
  else
  {
    // parameter space is [-1,1] for quadrilaterals
    sxia[0] = -1.0;
    sxia[1] = -1.0;
    sxib[0] = 1.0;
    sxib[1] = 1.0;
  }

  // ************************************************** quadratic 3D ***
  if (n_dim() == 3 && sele.is_quad())
  {
    // get LM interpolation and testing type
    auto lmtype =
        Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(interface_params(), "LM_QUAD");

    // build linear integration elements from quadratic CElements
    std::vector<std::shared_ptr<Mortar::IntElement>> sauxelements;
    split_int_elements(sele, sauxelements);

    // different options for mortar integration
    if (lmtype == Inpar::Mortar::lagmult_quad || lmtype == Inpar::Mortar::lagmult_lin)
    {
      // do the element integration of kappa and store into gap
      int nrow = sele.num_node();
      Core::LinAlg::SerialDenseVector gseg(nrow);

      // create a CONTACT integrator instance with correct num_gp and Dim
      CONTACT::Integrator integrator(imortar_, sele.shape(), get_comm());
      integrator.integrate_kappa_penalty(sele, sxia, sxib, gseg);

      // do the assembly into the slave nodes
      integrator.assemble_g(get_comm(), sele, gseg);
    }
    else if (lmtype == Inpar::Mortar::lagmult_pwlin)
    {
      // integrate each int element separately
      for (auto& sauxelement : sauxelements)
      {
        // do the int element integration of kappa and store into gap
        int nrow = sauxelement->num_node();
        Core::LinAlg::SerialDenseVector gseg(nrow);

        // create a CONTACT integrator instance with correct num_gp and Dim
        CONTACT::Integrator integrator(imortar_, sauxelement->shape(), get_comm());
        integrator.integrate_kappa_penalty(sele, *sauxelement, sxia, sxib, gseg);

        // do the assembly into the slave nodes
        integrator.assemble_g(get_comm(), *sauxelement, gseg);
      }
    }
    else
    {
      FOUR_C_THROW("integrate_kappa_penalty: Invalid case for 3D mortar contact LM interpolation");
    }
  }

  // *************************************************** other cases ***
  else
  {
    // do the element integration of kappa and store into gap
    int nrow = sele.num_node();
    Core::LinAlg::SerialDenseVector gseg(nrow);

    // create a CONTACT integrator instance with correct num_gp and Dim
    CONTACT::Integrator integrator(imortar_, sele.shape(), get_comm());
    integrator.integrate_kappa_penalty(sele, sxia, sxib, gseg);

    // do the assembly into the slave nodes
    integrator.assemble_g(get_comm(), sele, gseg);
  }

  return true;
}

/*----------------------------------------------------------------------*
 |  Evaluate relative movement (jump) of a slave node     gitterle 10/09|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::evaluate_relative_movement(
    const std::shared_ptr<Core::LinAlg::Vector<double>> xsmod,
    const std::shared_ptr<Core::LinAlg::SparseMatrix> dmatrixmod,
    const std::shared_ptr<Core::LinAlg::SparseMatrix> doldmod)
{
  if (not friction_)
  {
    FOUR_C_THROW(
        "Error in Interface::evaluate_relative_movement(): Only evaluated for frictional contact");
  }


  // parameters
  double pp = interface_params().get<double>("PENALTYPARAM");

  // loop over all slave row nodes on the current interface
  for (int i = 0; i < slave_row_nodes()->num_my_elements(); ++i)
  {
    int gid = slave_row_nodes()->gid(i);
    Core::Nodes::Node* node = discret().g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    auto* cnode = dynamic_cast<FriNode*>(node);
    double cn = get_cn_ref().local_values_as_span()[get_cn_ref().get_map().lid(cnode->id())];

    // get some information form the node
    double gap = cnode->data().getg();

    const int dim = cnode->num_dof();

    // compute normal part of Lagrange multiplier
    double nz = 0.0;
    for (int k = 0; k < 3; ++k)
    {
      nz += cnode->mo_data().n()[k] * cnode->mo_data().lm()[k];
    }

    std::vector<double> jump(dim);
    for (int dim = 0; dim < n_dim(); dim++) jump[dim] = 0;

    double lmuzawan = 0.0;
    for (int k = 0; k < dim; ++k)
      lmuzawan += cnode->mo_data().lmuzawa()[k] * cnode->mo_data().n()[k];

    double kappa = cnode->data().kappa();

    // evaluate jump (relative displacement) of this node
    // only when the node is going to be active, otherwise,
    // this value isn't needed.
    bool activeinfuture = false;

    const auto contact_strategy =
        Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(interface_params(), "STRATEGY");

    if (contact_strategy == CONTACT::SolvingStrategy::penalty ||
        contact_strategy == CONTACT::SolvingStrategy::multiscale ||
        (contact_strategy == CONTACT::SolvingStrategy::lagmult and
            not interface_params().get<bool>("SEMI_SMOOTH_NEWTON")))
    {
      if (-gap >= 0)
      {
        activeinfuture = true;
      }
    }
    else if (contact_strategy == CONTACT::SolvingStrategy::lagmult and
             interface_params().get<bool>("SEMI_SMOOTH_NEWTON"))
    {
      if ((nz - cn * gap > 0) or cnode->active())
      {
        activeinfuture = true;
      }
    }
    else if (contact_strategy == CONTACT::SolvingStrategy::uzawa)
    {
      if (lmuzawan - kappa * pp * gap >= 0)
      {
        activeinfuture = true;
      }
    }
    else
    {
      FOUR_C_THROW(
          "Error in Interface::evaluate_relative_movement(): Solution strategy not known!");
    }

    if (activeinfuture)
    {
      Core::Gen::Pairedvector<int, double>& dmap = cnode->mo_data().get_d();
      Core::Gen::Pairedvector<int, double>& dmapold = cnode->fri_data().get_d_old();

      std::set<int> snodes = cnode->fri_data().get_s_nodes();

      // check if there are entries in the old D map
      if (dmapold.size() < 1)
        FOUR_C_THROW("Error in Interface::evaluate_relative_movement(): No old D-Map!");

      std::map<int, double>::iterator colcurr;
      std::set<int>::iterator scurr;

      // loop over all slave nodes with an entry adjacent to this node
      for (scurr = snodes.begin(); scurr != snodes.end(); scurr++)
      {
        int gid = *scurr;
        Core::Nodes::Node* snode = idiscret_->g_node(gid);
        if (!snode) FOUR_C_THROW("Cannot find node with gid %", gid);
        Node* csnode = dynamic_cast<Node*>(snode);

        double dik = dmap[csnode->id()];
        double dikold = dmapold[csnode->id()];

        std::map<int, double>::iterator mcurr;

        for (int dim = 0; dim < csnode->num_dof(); ++dim)
        {
          int locid = (xsmod->get_map()).lid(csnode->dofs()[dim]);
          jump[dim] -= (dik - dikold) * xsmod->local_values_as_span()[locid];
        }
      }  //  loop over adjacent slave nodes

      std::map<int, double>& mmap = cnode->mo_data().get_m();
      std::map<int, double>& mmapold = cnode->fri_data().get_m_old();

      const std::set<int>& mnodescurrent = cnode->fri_data().get_m_nodes();
      const std::set<int>& mnodesold = cnode->fri_data().get_m_nodes_old();

      // check if there are entries in the M map
      if (mmap.size() < 1)
      {
        FOUR_C_THROW("Error in Interface::evaluate_relative_movement(): No M-Map!");
      }

      // check if there are entries in the old M map
      if (mmapold.size() < 1)
      {
        FOUR_C_THROW("Error in Interface::evaluate_relative_movement(): No old M-Map!");
      }

      if (mnodesold.size() < 1)
      {
        FOUR_C_THROW("Error in Interface::evaluate_relative_movement(): No old M-Set!");
      }

      std::set<int> mnodes;
      std::set<int>::iterator mcurr;

      for (mcurr = mnodescurrent.begin(); mcurr != mnodescurrent.end(); mcurr++)
      {
        mnodes.insert(*mcurr);
      }

      for (mcurr = mnodesold.begin(); mcurr != mnodesold.end(); mcurr++) mnodes.insert(*mcurr);

      // loop over all master nodes (find adjacent ones to this slip node)
      for (mcurr = mnodes.begin(); mcurr != mnodes.end(); mcurr++)
      {
        int gid = *mcurr;
        Core::Nodes::Node* mnode = idiscret_->g_node(gid);
        if (!mnode) FOUR_C_THROW("Cannot find node with gid %", gid);
        Node* cmnode = dynamic_cast<Node*>(mnode);

        double mik = mmap[cmnode->id()];
        double mikold = mmapold[cmnode->id()];

        std::map<int, double>::iterator mcurr;

        for (int dim = 0; dim < cnode->num_dof(); ++dim)
        {
          jump[dim] += (mik - mikold) * (cmnode->xspatial()[dim]);
        }
      }  //  loop over master nodes

      // write it to nodes
      for (int dim = 0; dim < n_dim(); dim++) cnode->fri_data().jump()[dim] = jump[dim];

      // linearization of jump vector

      // reset derivative map of jump
      for (auto& j : cnode->fri_data().get_deriv_jump())
      {
        j.clear();
      }
      (cnode->fri_data().get_deriv_jump()).resize(0);

      /*** 01  **********************************************************/

      if (dmatrixmod == nullptr)
      {
        // loop over according slave nodes
        for (scurr = snodes.begin(); scurr != snodes.end(); scurr++)
        {
          int gid = *scurr;
          Core::Nodes::Node* snode = idiscret_->g_node(gid);
          if (!snode) FOUR_C_THROW("Cannot find node with gid %", gid);
          Node* csnode = dynamic_cast<Node*>(snode);

          double dik = dmap[csnode->id()];
          double dikold = dmapold[csnode->id()];

          for (int dimrow = 0; dimrow < cnode->num_dof(); ++dimrow)
          {
            int col = csnode->dofs()[dimrow];
            double val = -(dik - dikold);
            if (abs(val) > 1e-14) cnode->add_deriv_jump_value(dimrow, col, val);
          }
        }
      }
      // in the 3D quadratic case, the values are obtained from the
      // global matrices Dmod and Doldmod
      else
      {
        // loop over dimension of the node
        for (int dim = 0; dim < cnode->num_dof(); ++dim)
        {
          int NumEntries = 0;
          int NumEntriesOld = 0;
          std::vector<double> Values((dmatrixmod->max_num_entries()));
          std::vector<int> Indices((dmatrixmod->max_num_entries()));
          std::vector<double> ValuesOld((dmatrixmod->max_num_entries()));
          std::vector<int> IndicesOld((dmatrixmod->max_num_entries()));

          // row
          int row = cnode->dofs()[dim];

          // extract entries of this row from matrix
          dmatrixmod->extract_global_row_copy(
              row, dmatrixmod->max_num_entries(), NumEntries, Values.data(), Indices.data());

          doldmod->extract_global_row_copy(
              row, doldmod->max_num_entries(), NumEntriesOld, ValuesOld.data(), IndicesOld.data());

          // loop over entries of this vector
          for (int j = 0; j < NumEntries; ++j)
          {
            double ValueOld = 0;
            bool found = false;

            // find value with the same index in vector of Dold
            for (int k = 0; k < NumEntriesOld; ++k)
            {
              if (Indices[k] == Indices[j])
              {
                ValueOld = ValuesOld[k];
                found = true;
                break;
              }
            }

            if (not found or abs(ValueOld) < 1e-12)
            {
              FOUR_C_THROW("Error in EvaluareRelMov(): No old D value exists");
            }

            // write to node
            cnode->add_deriv_jump_value(dim, Indices[j], (Values[j] - ValueOld));
          }
        }
      }

      /*** 02  **********************************************************/
      // loop over according master nodes
      for (mcurr = mnodes.begin(); mcurr != mnodes.end(); mcurr++)
      {
        int gid = *mcurr;
        Core::Nodes::Node* mnode = idiscret_->g_node(gid);
        if (!mnode) FOUR_C_THROW("Cannot find node with gid %", gid);
        Node* cmnode = dynamic_cast<Node*>(mnode);

        double mik = mmap[cmnode->id()];
        double mikold = mmapold[cmnode->id()];

        for (int dimrow = 0; dimrow < cnode->num_dof(); ++dimrow)
        {
          int col = cmnode->dofs()[dimrow];
          double val = (mik - mikold);
          if (abs(val) > 1e-14) cnode->add_deriv_jump_value(dimrow, col, val);
        }
      }

      /*** 03 ***********************************************************/
      // we need the Lin(D-matrix) entries of this node
      std::map<int, std::map<int, double>>& ddmap = cnode->data().get_deriv_d();
      std::map<int, std::map<int, double>>::iterator dscurr;

      // loop over all slave nodes in the DerivM-map of the stick slave node
      for (dscurr = ddmap.begin(); dscurr != ddmap.end(); ++dscurr)
      {
        int gid = dscurr->first;
        Core::Nodes::Node* snode = idiscret_->g_node(gid);
        if (!snode) FOUR_C_THROW("Cannot find node with gid %", gid);
        Node* csnode = dynamic_cast<Node*>(snode);

        // compute entry of the current stick node / slave node pair
        std::map<int, double>& thisdmmap = cnode->data().get_deriv_d(gid);

        // loop over all entries of the current derivative map
        for (colcurr = thisdmmap.begin(); colcurr != thisdmmap.end(); ++colcurr)
        {
          int col = colcurr->first;

          // loop over dimensions
          for (int dim = 0; dim < cnode->num_dof(); ++dim)
          {
            int locid = (xsmod->get_map()).lid(csnode->dofs()[dim]);
            double val = -colcurr->second * xsmod->local_values_as_span()[locid];
            if (abs(val) > 1e-14) cnode->add_deriv_jump_value(dim, col, val);
          }
        }
      }

      /*** 04 ***********************************************************/
      // we need the Lin(M-matrix) entries of this node
      std::map<int, std::map<int, double>>& dmmap = cnode->data().get_deriv_m();
      std::map<int, std::map<int, double>>::iterator dmcurr;

      // loop over all master nodes in the DerivM-map of the stick slave node
      for (dmcurr = dmmap.begin(); dmcurr != dmmap.end(); ++dmcurr)
      {
        int gid = dmcurr->first;
        Core::Nodes::Node* mnode = idiscret_->g_node(gid);
        if (!mnode) FOUR_C_THROW("Cannot find node with gid %", gid);
        Node* cmnode = dynamic_cast<Node*>(mnode);
        double* mxi = cmnode->xspatial();

        // compute entry of the current stick node / master node pair
        std::map<int, double>& thisdmmap = cnode->data().get_deriv_m(gid);

        // loop over all entries of the current derivative map
        for (colcurr = thisdmmap.begin(); colcurr != thisdmmap.end(); ++colcurr)
        {
          int col = colcurr->first;

          // loop over dimensions
          for (int dimrow = 0; dimrow < cnode->num_dof(); ++dimrow)
          {
            double val = colcurr->second * mxi[dimrow];
            if (abs(val) > 1e-14) cnode->add_deriv_jump_value(dimrow, col, val);
          }
        }
      }

      if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
      {
        for (int j = 0; j < n_dim(); j++)
        {
          if (cnode->dbc_dofs()[j])
          {
            cnode->fri_data().jump()[j] = 0.;
            cnode->fri_data().get_deriv_jump()[j].clear();
          }
        }
      }

    }  // active nodes
  }  // loop over slave nodes
}

/*----------------------------------------------------------------------*
 |  calculate nodal distances (public)                     pfaller Jan15|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::evaluate_distances(
    const std::shared_ptr<const Core::LinAlg::Vector<double>>& vec,
    std::map<int, std::vector<double>>& mynormals,
    std::map<int, std::vector<Core::Gen::Pairedvector<int, double>>>& dmynormals,
    std::map<int, double>& mygap, std::map<int, std::map<int, double>>& dmygap)
{
  set_state(Mortar::state_new_displacement, *vec);
  initialize();

  // interface needs to be complete
  if (!filled() && Core::Communication::my_mpi_rank(get_comm()) == 0)
    FOUR_C_THROW("fill_complete() not called on interface %", id_);

  // create an interpolator instance
  NTS::Interpolator interpolator(imortar_, n_dim());

  // create normals
  pre_evaluate(-1, -1);  // dummy values

  // loop over proc's slave elements of the interface for integration
  // use standard column map to include processor's ghosted elements
  Core::Communication::barrier(get_comm());

  for (int i = 0; i < selecolmap_->num_my_elements(); ++i)
  {
    int gid1 = selecolmap_->gid(i);
    Core::Elements::Element* ele1 = idiscret_->g_element(gid1);
    if (!ele1) FOUR_C_THROW("Cannot find slave element with gid %", gid1);
    auto* selement = dynamic_cast<CONTACT::Element*>(ele1);

    if (selement->mo_data().num_search_elements() < 1)
    {
      std::cout << "WARNING: No elements found!" << '\n';
      continue;
    }

    // skip zero-sized nurbs elements (slave)
    if (selement->zero_sized()) continue;

    // empty vector of master element pointers
    std::vector<CONTACT::Element*> melements;

    // loop over the candidate master elements of sele_
    // use slave element's candidate list SearchElements !!!
    for (int j = 0; j < selement->mo_data().num_search_elements(); ++j)
    {
      int gid2 = selement->mo_data().search_elements()[j];
      Core::Elements::Element* ele2 = idiscret_->g_element(gid2);
      if (!ele2) FOUR_C_THROW("Cannot find master element with gid %", gid2);
      auto* melement = dynamic_cast<CONTACT::Element*>(ele2);

      // skip zero-sized nurbs elements (master)
      if (melement->zero_sized()) continue;

      melements.push_back(melement);
    }

    //**************************************************************
    //                loop over all Slave nodes
    //**************************************************************
    for (int snodes = 0; snodes < selement->num_node(); ++snodes)
    {
      auto* mynode = dynamic_cast<CONTACT::Node*>(selement->nodes()[snodes]);

      // skip this node if already considered
      if (mynode->has_proj()) continue;

      //                store node normals
      //**************************************************************
      //      int gid = snoderowmapbound_->GID(snodes);
      int gid = mynode->id();

      int numdofs = mynode->num_dof();
      std::vector<double> temp(numdofs, 0.0);
      for (int kk = 0; kk < numdofs; kk++)
      {
        temp[kk] = mynode->mo_data().n()[kk];
      }
      mynormals.insert(std::pair<int, std::vector<double>>(gid, temp));
      dmynormals.insert(std::pair<int, std::vector<Core::Gen::Pairedvector<int, double>>>(
          gid, mynode->data().get_deriv_n()));

      //**************************************************************
      double sxi[2] = {0.0, 0.0};

      if (selement->shape() == Core::FE::CellType::quad4 or
          selement->shape() == Core::FE::CellType::quad8 or
          selement->shape() == Core::FE::CellType::quad9)
      {
        // TODO (pfaller): switch case
        if (snodes == 0)
        {
          sxi[0] = -1;
          sxi[1] = -1;
        }
        else if (snodes == 1)
        {
          sxi[0] = 1;
          sxi[1] = -1;
        }
        else if (snodes == 2)
        {
          sxi[0] = 1;
          sxi[1] = 1;
        }
        else if (snodes == 3)
        {
          sxi[0] = -1;
          sxi[1] = 1;
        }
        else if (snodes == 4)
        {
          sxi[0] = 0;
          sxi[1] = -1;
        }
        else if (snodes == 5)
        {
          sxi[0] = 1;
          sxi[1] = 0;
        }
        else if (snodes == 6)
        {
          sxi[0] = 0;
          sxi[1] = 1;
        }
        else if (snodes == 7)
        {
          sxi[0] = -1;
          sxi[1] = 0;
        }
        else if (snodes == 8)
        {
          sxi[0] = 0;
          sxi[1] = 0;
        }
        else
        {
          FOUR_C_THROW("ERROR: wrong node LID");
        }
      }
      else if (selement->shape() == Core::FE::CellType::tri3 or
               selement->shape() == Core::FE::CellType::tri6)
      {
        if (snodes == 0)
        {
          sxi[0] = 0;
          sxi[1] = 0;
        }
        else if (snodes == 1)
        {
          sxi[0] = 1;
          sxi[1] = 0;
        }
        else if (snodes == 2)
        {
          sxi[0] = 0;
          sxi[1] = 1;
        }
        else if (snodes == 3)
        {
          sxi[0] = 0.5;
          sxi[1] = 0;
        }
        else if (snodes == 4)
        {
          sxi[0] = 0.5;
          sxi[1] = 0.5;
        }
        else if (snodes == 5)
        {
          sxi[0] = 0;
          sxi[1] = 0.5;
        }
        else
        {
          FOUR_C_THROW("ERROR: wrong node LID");
        }
      }
      else
      {
        FOUR_C_THROW("Chosen element type not supported for NTS!");
      }

      //**************************************************************
      //                loop over all Master Elements
      //**************************************************************
      // create vectors to store projection information for several master elements in case
      // projection is not unique
      std::vector<double> gap_vec;
      std::vector<std::map<int, double>> dgap_vec;

      for (auto& melement : melements)
      {
        // project Gauss point onto master element
        double mxi[2] = {0.0, 0.0};
        double projalpha = 0.0;
        bool is_projected = Mortar::Projector::impl(*selement, *melement)
                                ->project_gauss_point_3d(*selement, sxi, *melement, mxi, projalpha);

        bool is_on_mele = true;

        // check GP projection
        Core::FE::CellType dt = melement->shape();
        const double tol = 1e-8;
        if (dt == Core::FE::CellType::quad4 || dt == Core::FE::CellType::quad8 ||
            dt == Core::FE::CellType::quad9)
        {
          if (mxi[0] < -1.0 - tol || mxi[1] < -1.0 - tol || mxi[0] > 1.0 + tol ||
              mxi[1] > 1.0 + tol)
          {
            is_on_mele = false;
          }
        }
        else
        {
          if (mxi[0] < -tol || mxi[1] < -tol || mxi[0] > 1.0 + tol || mxi[1] > 1.0 + tol ||
              mxi[0] + mxi[1] > 1.0 + 2 * tol)
          {
            is_on_mele = false;
          }
        }

        // node on mele?
        if (is_on_mele && is_projected)
        {
          // store information of projection so that this node is not considered again
          mynode->has_proj() = true;

          int ndof = 3;
          int ncol = melement->num_node();
          Core::LinAlg::SerialDenseVector mval(ncol);
          Core::LinAlg::SerialDenseMatrix mderiv(ncol, 2);
          melement->evaluate_shape(mxi, mval, mderiv, ncol, false);

          //          int linsize    = mynode->GetLinsize();
          int linsize = 100;
          double gpn[3] = {0.0, 0.0, 0.0};
          //**************************************************************

          // evaluate the GP slave coordinate derivatives --> no entries
          std::vector<Core::Gen::Pairedvector<int, double>> dsxi(2, 0);
          std::vector<Core::Gen::Pairedvector<int, double>> dmxi(2, 4 * linsize + ncol * ndof);

          (interpolator).deriv_xi_gp_3d(*selement, *melement, sxi, mxi, dsxi, dmxi, projalpha);
          (interpolator).nw_gap_3d(*mynode, *melement, mval, mderiv, dmxi, gpn);

          // store linearization for node
          std::map<int, double> dgap = mynode->data().get_deriv_gnts();  // (dof,value)

          // store gap information
          gap_vec.push_back(mynode->data().getgnts());
          dgap_vec.push_back(dgap);

          // reset nodal weighted gap and derivative
          mynode->data().getgnts() = 1.0e12;
          (mynode->data().get_deriv_gnts()).clear();
        }  // End hit ele
      }  // End Loop over all Master Elements

      if (gap_vec.size() > 0)
      {
        // find projection with smallest absolute value of gap
        auto iter_min = std::min_element(gap_vec.begin(), gap_vec.end(), abs_compare);
        const int i_min = std::distance(gap_vec.begin(), iter_min);

        // save to map at GID
        mygap.insert(std::pair<int, double>(gid, gap_vec[i_min]));
        dmygap.insert(std::pair<int, std::map<int, double>>(gid, dgap_vec[i_min]));
      }
    }
  }

  Core::Communication::barrier(get_comm());
}

/*----------------------------------------------------------------------*
 |  Evaluate L2 Norm of tangential contact conditions     gitterle 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::evaluate_tangent_norm(double& cnormtan)
{
  // friction coefficient
  double frcoeff = interface_params().get<double>("FRCOEFF");

  // loop over all slave row nodes on the current interface
  for (int i = 0; i < slave_row_nodes()->num_my_elements(); ++i)
  {
    int gid = slave_row_nodes()->gid(i);
    Core::Nodes::Node* node = discret().g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    auto* cnode = dynamic_cast<FriNode*>(node);

    // get some information from node
    double* n = cnode->mo_data().n();
    const int dim = cnode->num_dof();

    // tangential plane
    Core::LinAlg::SerialDenseMatrix tanplane(dim, dim);
    if (dim == 3)
    {
      tanplane(0, 0) = 1 - (n[0] * n[0]);
      tanplane(0, 1) = -(n[0] * n[1]);
      tanplane(0, 2) = -(n[0] * n[2]);
      tanplane(1, 0) = -(n[1] * n[0]);
      tanplane(1, 1) = 1 - (n[1] * n[1]);
      tanplane(1, 2) = -(n[1] * n[2]);

      tanplane(2, 0) = -(n[2] * n[0]);
      tanplane(2, 1) = -(n[2] * n[1]);
      tanplane(2, 2) = 1 - (n[2] * n[2]);
    }
    else if (dim == 2)
    {
      tanplane(0, 0) = 1 - (n[0] * n[0]);
      tanplane(0, 1) = -(n[0] * n[1]);

      tanplane(1, 0) = -(n[1] * n[0]);
      tanplane(1, 1) = 1 - (n[1] * n[1]);
    }
    else
    {
      FOUR_C_THROW("Error in AssembleTangentForces: Unknown dimension.");
    }

    // jump vector
    Core::LinAlg::SerialDenseMatrix jumpvec(dim, 1);
    for (int i = 0; i < dim; i++)
    {
      jumpvec(i, 0) = cnode->fri_data().jump()[i];
    }

    // evaluate jump in tangential direction
    Core::LinAlg::SerialDenseMatrix jumptan(dim, 1);
    Core::LinAlg::multiply(jumptan, tanplane, jumpvec);

    // force vector
    Core::LinAlg::SerialDenseMatrix forcevec(dim, 1);
    for (int i = 0; i < dim; i++)
    {
      forcevec(i, 0) = cnode->mo_data().lm()[i];
    }

    // evaluate force in normal direction
    double forcen = 0.0;
    for (int k = 0; k < dim; ++k) forcen += forcevec(k, 0) * n[k];

    // norm of constraint violation for stick nodes
    if (cnode->active() and not cnode->fri_data().slip())
    {
      for (int j = 0; j < dim; j++) cnormtan += jumptan(j, 0) * jumptan(j, 0);
    }
    // norm of constraint violation for slip nodes
    else if (cnode->active() and cnode->fri_data().slip())
    {
      double part1 = 0.0;
      double jumpnorm = 0.0;

      for (int j = 0; j < dim; j++)
      {
        jumpnorm += jumptan(j, 0) * jumptan(j, 0);
        part1 += jumptan(j, 0) * forcevec(j, 0);
      }

      jumpnorm = sqrt(jumpnorm);
      cnormtan += (part1 - frcoeff * forcen * jumpnorm) * (part1 - frcoeff * forcen * jumpnorm);
    }
  }  // loop over slave nodes

  // get cnorm from all procs
  double sumcnormtanallprocs = 0.0;
  sumcnormtanallprocs = Core::Communication::sum_all(cnormtan, get_comm());
  cnormtan = sumcnormtanallprocs;
}


/*----------------------------------------------------------------------*
 |  Update active set and check for convergence (public)      popp 06/08|
 *----------------------------------------------------------------------*/
bool CONTACT::Interface::update_active_set_semi_smooth()
{
  // get input parameter ftype
  auto ftype = Teuchos::getIntegralValue<CONTACT::FrictionType>(interface_params(), "FRICTION");

  // this is the complementarity parameter we use for the decision.
  // it might be scaled with a mesh-size dependent factor
  double cn = 0.;
  double ct = 0.;

  // assume that active set has converged and check for opposite
  int localcheck = true;

  // loop over all slave nodes on the current interface
  for (int j = 0; j < slave_row_nodes()->num_my_elements(); ++j)
  {
    int gid = slave_row_nodes()->gid(j);
    Core::Nodes::Node* node = discret().g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);

    Node* cnode = dynamic_cast<Node*>(node);

    cn = get_cn_ref().local_values_as_span()[get_cn_ref().get_map().lid(cnode->id())];
    if (friction_)
      ct = get_ct_ref().local_values_as_span()[get_ct_ref().get_map().lid(cnode->id())];

    // get weighted gap
    double wgap = cnode->data().getg();

    // compute normal part of Lagrange multiplier
    double nz = 0.0;
    for (int k = 0; k < 3; ++k)
    {
      nz += cnode->mo_data().n()[k] * cnode->mo_data().lm()[k];
    }

    // friction
    std::vector<double> tz(n_dim() - 1, 0);
    std::vector<double> tjump(n_dim() - 1, 0);
    double euclidean = 0.0;

    if (friction_)
    {
      // static cast
      auto* frinode = dynamic_cast<FriNode*>(cnode);

      // compute tangential parts and of Lagrange multiplier and incremental jumps
      for (int i = 0; i < n_dim(); ++i)
      {
        tz[0] += frinode->data().txi()[i] * frinode->mo_data().lm()[i];
        if (n_dim() == 3) tz[1] += frinode->data().teta()[i] * frinode->mo_data().lm()[i];

        if (not interface_params().get<bool>("GP_SLIP_INCR"))
        {
          tjump[0] += frinode->data().txi()[i] * frinode->fri_data().jump()[i];
          if (n_dim() == 3) tjump[1] += frinode->data().teta()[i] * frinode->fri_data().jump()[i];
        }
      }

      if (interface_params().get<bool>("GP_SLIP_INCR"))
      {
        tjump[0] = frinode->fri_data().jump_var()[0];
        if (n_dim() == 3) tjump[1] = frinode->fri_data().jump_var()[1];
      }

      // evaluate euclidean norm |tz+ct.tjump|
      std::vector<double> sum(n_dim() - 1, 0);
      sum[0] = tz[0] + ct * tjump[0];
      if (n_dim() == 3) sum[1] = tz[1] + ct * tjump[1];
      if (n_dim() == 2) euclidean = abs(sum[0]);
      if (n_dim() == 3) euclidean = sqrt(sum[0] * sum[0] + sum[1] * sum[1]);
    }

    // adhesion
    double adhbound = 0.0;
    if (Teuchos::getIntegralValue<CONTACT::AdhesionType>(interface_params(), "ADHESION") ==
        CONTACT::AdhesionType::bounded)
      adhbound = interface_params().get<double>("ADHESION_BOUND");

    // check nodes of inactive set *************************************
    if (not cnode->active())
    {
      // check for penetration and/or tensile contact forces
      if (nz - cn * wgap > 0)  // no averaging of Lagrange multipliers
      // if ((0.5*nz+0.5*nzold) - cn*wgap > 0) // averaging of Lagrange multipliers
      {
        cnode->active() = true;
        localcheck = false;

        // friction
        if (friction_)
        {
          // nodes coming into contact
          dynamic_cast<FriNode*>(cnode)->fri_data().slip() = true;
        }
      }
    }

    // check nodes of active set ***************************************
    else
    {
      // adhesion modification
      nz += adhbound;

      // check for tensile contact forces and/or penetration
      if (nz - cn * wgap <= 0)  // no averaging of Lagrange multipliers
      // if ((0.5*nz+0.5*nzold) - cn*wgap <= 0) // averaging of Lagrange multipliers
      {
        cnode->active() = false;
        localcheck = false;

        // friction
        if (friction_) dynamic_cast<FriNode*>(cnode)->fri_data().slip() = false;
      }

      // only do something for friction
      else
      {
        // friction tresca
        if (ftype == CONTACT::FrictionType::tresca)
        {
          auto* frinode = dynamic_cast<FriNode*>(cnode);

          // CAREFUL: friction bound is now interface-local (popp 08/2012)
          double frbound = interface_params().get<double>("FRBOUND");

          if (not frinode->fri_data().slip())
          {
            // check (euclidean)-frbound <= 0
            if (euclidean - frbound <= 0)
            {
            }
            // do nothing (stick was correct)
            else
            {
              frinode->fri_data().slip() = true;
              localcheck = false;
            }
          }
          else
          {
            // check (euclidean)-frbound > 0
            if (euclidean - frbound > 0)
            {
            }
            // do nothing (slip was correct)
            else
            {
              frinode->fri_data().slip() = false;
              localcheck = false;
            }
          }
        }  // if (fytpe=="tresca")

        // friction coulomb
        if (ftype == CONTACT::FrictionType::coulomb)
        {
          auto* frinode = dynamic_cast<FriNode*>(cnode);

          // CAREFUL: friction coefficient is now interface-local (popp 08/2012)
          double frcoeff = frinode->fr_coeff(interface_params().get<double>("FRCOEFF"));
          double frbound;
          static const bool regularization =
              interface_params().get<bool>("REGULARIZED_NORMAL_CONTACT");
          if (!regularization)
            frbound = frcoeff * (nz - cn * wgap);
          else
          {
            static const double k = 1. / interface_params().get<double>("REGULARIZATION_STIFFNESS");
            static const double gmax = interface_params().get<double>("REGULARIZATION_THICKNESS");
            if (cnode->mo_data().get_d().size() != 1)
            {
              FOUR_C_THROW(
                  "we need to have a D-value for active contact nodes\nAnd exactly one due to "
                  "biorthogonality");
            }
            double dval = cnode->mo_data().get_d().at(cnode->id());
            const double gLM = gmax * (1. - exp(-k / gmax * nz));
            frbound = frcoeff * std::max(0., nz - cn * (wgap + dval * gLM));
          }

          if (not frinode->fri_data().slip())
          {
            // check (euclidean)-frbound <= 0
            if (euclidean - frbound <= 1e-10)
            {
            }
            // do nothing (stick was correct)
            else
            {
              frinode->fri_data().slip() = true;
              localcheck = false;
            }
          }
          else
          {
            // check (euclidean)-frbound > 0
            if (euclidean - frbound > -1e-10)
            {
            }
            // do nothing (slip was correct)
            else
            {
              frinode->fri_data().slip() = false;
              localcheck = false;
            }
          }
        }  // if (ftype == CONTACT::FrictionType::coulomb)
      }  // if (nz - cn*wgap <= 0)
    }  // if (cnode->Active()==false)
  }  // loop over all slave nodes

  // broadcast convergence status among processors
  int convcheck = 0;
  convcheck = Core::Communication::min_all(localcheck, get_comm());

  return convcheck;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::Interface::update_active_set_initial_status() const
{
  // List of GIDs of all my slave row nodes
  int* my_slave_row_node_gids = slave_row_nodes()->my_global_elements();

  // loop over all slave nodes on the current interface
  for (int j = 0; j < slave_row_nodes()->num_my_elements(); ++j)
  {
    // Grab the current slave node
    const int gid = my_slave_row_node_gids[j];
    Node* cnode = dynamic_cast<Node*>(discret().g_node(gid));
    if (!cnode) FOUR_C_THROW("Cannot find node with gid %", gid);

    set_node_initially_active(*cnode);
  }
}

/*----------------------------------------------------------------------*
 |  build active set (nodes / dofs)                           popp 02/08|
 *----------------------------------------------------------------------*/
bool CONTACT::Interface::build_active_set(bool init)
{
  // define local variables
  std::vector<int> mynodegids;
  std::vector<int> mynodegidsInactive;
  std::vector<int> mydofgids;
  std::vector<int> mydofgidsInactive;
  std::vector<int> myslipnodegids;
  std::vector<int> myslipdofgids;
  std::vector<int> mymnodegids;
  std::vector<int> mymdofgids;

  // loop over all slave nodes
  for (int i = 0; i < snoderowmap_->num_my_elements(); ++i)
  {
    int gid = snoderowmap_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Node* cnode = dynamic_cast<Node*>(node);
    const int numdof = cnode->num_dof();

    // *******************************************************************
    // INITIALIZATION OF THE ACTIVE SET (t=0)
    // *******************************************************************
    // This is given by the Node member variable IsInitActive(), which
    // has been introduced via the contact conditions in the input file.
    // Thus, if no design line has been chosen to be active at t=0,
    // the active node set will be empty at t=0. Yet, if one or more
    // design lines have been specified as "Slave" AND "Active" then
    // the corresponding Nodes are put into an initial active set!
    // This yields a very flexible solution for contact initialization.
    // *******************************************************************
    if (init)
    {
      // flag for initialization of init active nodes with nodal gaps
      const bool initcontactbygap = interface_params().get<bool>("INITCONTACTBYGAP");
      // value
      double initcontactval = interface_params().get<double>("INITCONTACTGAPVALUE");

      // Either init contact by definition or by gap
      if (cnode->is_init_active() and initcontactbygap)
        FOUR_C_THROW("Init contact either by definition in condition or by gap!");

      // check if node is initially active or, if initialization with nodal, gap,
      // the gap is smaller than the prescribed value
      if (cnode->is_init_active() or (initcontactbygap and cnode->data().getg() < initcontactval))
      {
        cnode->active() = true;
        mynodegids.push_back(cnode->id());

        for (int j = 0; j < numdof; ++j) mydofgids.push_back(cnode->dofs()[j]);
      }

      // check if frictional node is initially in slip state
      if (friction_)
      {
        // do nothing: we always assume STICK at t=0
      }
    }

    // *******************************************************************
    // RE-BUILDING OF THE ACTIVE SET
    // *******************************************************************
    else
    {
      // check if node is active
      if (cnode->active())
      {
        mynodegids.push_back(cnode->id());

        for (int j = 0; j < numdof; ++j) mydofgids.push_back(cnode->dofs()[j]);
      }
      else
      {
        mynodegidsInactive.push_back(cnode->id());

        for (int j = 0; j < numdof; ++j) mydofgidsInactive.push_back(cnode->dofs()[j]);
      }

      // check if frictional node is in slip state
      if (friction_)
      {
        if (dynamic_cast<FriNode*>(cnode)->fri_data().slip())
        {
          myslipnodegids.push_back(cnode->id());

          for (int j = 0; j < numdof; ++j) myslipdofgids.push_back(cnode->dofs()[j]);
        }
      }
    }
  }

  // create active node map and active dof map
  activenodes_ = Core::LinAlg::create_map(mynodegids, get_comm());
  activedofs_ = Core::LinAlg::create_map(mydofgids, get_comm());
  inactivenodes_ = Core::LinAlg::create_map(mynodegidsInactive, get_comm());
  inactivedofs_ = Core::LinAlg::create_map(mydofgidsInactive, get_comm());

  if (friction_)
  {
    // create slip node map and slip dof map
    slipnodes_ = Core::LinAlg::create_map(myslipnodegids, get_comm());
    slipdofs_ = Core::LinAlg::create_map(myslipdofgids, get_comm());
  }

  // split active dofs and slip dofs
  split_active_dofs();

  return true;
}

/*----------------------------------------------------------------------*
 |  split active dofs into Ndofs, Tdofs and slipTdofs         popp 02/08|
 *----------------------------------------------------------------------*/
bool CONTACT::Interface::split_active_dofs()
{
  // get out of here if active set is empty
  if (activenodes_ == nullptr or activenodes_->num_global_elements() == 0)
  {
    activen_ = std::make_shared<Core::LinAlg::Map>(0, 0, get_comm());
    activet_ = std::make_shared<Core::LinAlg::Map>(0, 0, get_comm());
    slipt_ = std::make_shared<Core::LinAlg::Map>(0, 0, get_comm());
    return true;
  }

  // define local variables
  int countN = 0;
  int countT = 0;
  std::vector<int> myNgids(activenodes_->num_my_elements());
  std::vector<int> myTgids((n_dim() - 1) * activenodes_->num_my_elements());

  // dimension check
  double dimcheck = (activedofs_->num_global_elements()) / (activenodes_->num_global_elements());
  if (dimcheck != n_dim()) FOUR_C_THROW("SplitActiveDofs: Nodes <-> Dofs dimension mismatch!");

  // loop over all active row nodes
  for (int i = 0; i < activenodes_->num_my_elements(); ++i)
  {
    int gid = activenodes_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Node* cnode = dynamic_cast<Node*>(node);
    const int numdof = cnode->num_dof();

    // add first dof to Nmap
    myNgids[countN] = cnode->dofs()[0];
    ++countN;

    // add remaining dofs to Tmap
    for (int j = 1; j < numdof; ++j)
    {
      myTgids[countT] = cnode->dofs()[j];
      ++countT;
    }
  }

  // resize the temporary vectors
  myNgids.resize(countN);
  myTgids.resize(countT);

  // communicate countN and countT among procs
  int gcountN, gcountT;
  gcountN = Core::Communication::sum_all(countN, get_comm());
  gcountT = Core::Communication::sum_all(countT, get_comm());

  // check global dimensions
  if ((gcountN + gcountT) != activedofs_->num_global_elements())
    FOUR_C_THROW("split_active_dofs: Splitting went wrong!");

  // create Nmap and Tmap objects
  activen_ = std::make_shared<Core::LinAlg::Map>(gcountN, countN, myNgids.data(), 0, get_comm());
  activet_ = std::make_shared<Core::LinAlg::Map>(gcountT, countT, myTgids.data(), 0, get_comm());

  // *******************************************************************
  // FRICTION - EXTRACTING TANGENTIAL DOFS FROM SLIP DOFS
  // *******************************************************************

  // get out of here if there is no friction
  if (not friction_)
  {
    return true;
  }

  // get out of here if slip set is empty
  if (slipnodes_ == nullptr)
  {
    slipt_ = std::make_shared<Core::LinAlg::Map>(0, 0, get_comm());
    return true;
  }

  if (slipnodes_->num_global_elements() == 0)
  {
    slipt_ = std::make_shared<Core::LinAlg::Map>(0, 0, get_comm());
    return true;
  }

  // define local variables
  int countslipT = 0;
  std::vector<int> myslipTgids((n_dim() - 1) * slipnodes_->num_my_elements());

  // dimension check
  dimcheck = (slipdofs_->num_global_elements()) / (slipnodes_->num_global_elements());
  if (dimcheck != n_dim()) FOUR_C_THROW("SplitActiveDofs: Nodes <-> Dofs dimension mismatch!");

  // loop over all slip row nodes
  for (int i = 0; i < slipnodes_->num_my_elements(); ++i)
  {
    int gid = slipnodes_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Node* cnode = dynamic_cast<Node*>(node);
    const int numdof = cnode->num_dof();

    // add dofs to slipTmap
    for (int j = 1; j < numdof; ++j)
    {
      myslipTgids[countslipT] = cnode->dofs()[j];
      ++countslipT;
    }
  }

  // resize the temporary vectors
  myslipTgids.resize(countslipT);

  // communicate countslipT among procs
  int gcountslipT;
  gcountslipT = Core::Communication::sum_all(countslipT, get_comm());

  // create Tslipmap objects
  slipt_ = std::make_shared<Core::LinAlg::Map>(
      gcountslipT, countslipT, myslipTgids.data(), 0, get_comm());

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const CONTACT::Interface& CONTACT::Interface::get_ma_sharing_ref_interface() const
{
  return dynamic_cast<const Interface&>(interface_data_->get_ma_sharing_ref_interface());
}


/*----------------------------------------------------------------------*
 | Store nodal quant. to old ones (last conv. time step)  gitterle 02/09|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::store_to_old(Mortar::StrategyBase::QuantityType type)
{
  // loop over all slave row nodes on the current interface
  for (int j = 0; j < slave_col_nodes()->num_my_elements(); ++j)
  {
    int gid = slave_col_nodes()->gid(j);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);

    switch (type)
    {
      case Mortar::StrategyBase::dm:
      {
        // store D and M entries
        dynamic_cast<FriNode*>(node)->store_dm_old();
        break;
      }
      case Mortar::StrategyBase::pentrac:
      {
        // store penalty tractions to old ones
        dynamic_cast<FriNode*>(node)->store_trac_old();
        break;
      }
      case Mortar::StrategyBase::n_old:
      {
        dynamic_cast<Node*>(node)->store_old_normal();
        break;
      }
      case Mortar::StrategyBase::activeold:
      {
        dynamic_cast<Node*>(node)->data().active_old() = dynamic_cast<Node*>(node)->active();
        break;
      }
      default:
        FOUR_C_THROW("StoreDMToNodes: Unknown state std::string variable!");
        break;
    }  // switch
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::Interface::update_self_contact_lag_mult_set(
    const Core::LinAlg::Map& gref_lmmap, const Core::LinAlg::Map& gref_smmap)
{
  if (gref_lmmap.num_my_elements() != gref_smmap.num_my_elements()) FOUR_C_THROW("Size mismatch!");

  const int num_sgids = sdofrowmap_->num_my_elements();
  const int* sgids = sdofrowmap_->my_global_elements();
  const int* ref_lmgids = gref_lmmap.my_global_elements();

  std::vector<int> lmdofs;
  lmdofs.reserve(num_sgids);

  for (int i = 0; i < num_sgids; ++i)
  {
    const int sgid = sgids[i];
    const int ref_lid = gref_smmap.lid(sgid);
    if (ref_lid == -1)
    {
      FOUR_C_THROW(
          "Couldn't find the current slave gid #{} in the reference self "
          "contact slave master map.",
          sgid);
    }
    lmdofs.push_back(ref_lmgids[ref_lid]);
  }

  lmdofmap_ = std::make_shared<Core::LinAlg::Map>(
      -1, static_cast<int>(lmdofs.size()), lmdofs.data(), 0, get_comm());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::Interface::set_node_initially_active(CONTACT::Node& cnode) const
{
  static const bool init_contact_by_gap = interface_params().get<bool>("INITCONTACTBYGAP");

  const bool node_init_active = cnode.is_init_active();

  // Either init contact by definition or by gap
  if (node_init_active and init_contact_by_gap)
    FOUR_C_THROW("Init contact either by definition in condition or by gap!");
  else if (node_init_active)
    cnode.active() = true;
  else if (init_contact_by_gap)
    set_node_initially_active_by_gap(cnode);

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (node_init_active)
  {
    std::cout << "Node #" << std::setw(5) << cnode.id()
              << " is set initially active via the condition line.\n";
  }
  else if (init_contact_by_gap)
  {
    std::cout << "Node #" << std::setw(5) << cnode.id() << " is set initially active by gap.\n";
  }
#endif
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::Interface::set_node_initially_active_by_gap(CONTACT::Node& cnode) const
{
  static const double initcontactval = interface_params().get<double>("INITCONTACTGAPVALUE");

  if (cnode.data().getg() < initcontactval) cnode.active() = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::Interface::set_condition_specific_parameters()
{
  if (interface_params().isSublist("ContactS2ICoupling"))
  {
    // read interface parameters and set them to the scatra boundary parameter class
    auto& s2icouplinglist = interface_params().sublist("ContactS2ICoupling", true);
    Discret::Elements::ScaTraEleParameterBoundary::instance("scatra")->set_parameters(
        s2icouplinglist);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::Interface::postprocess_quantities(const Teuchos::ParameterList& outputParams) const
{
  using std::shared_ptr;

  // Check if the given parameter list contains all required data to be written to output
  {
    // Vector with names of all required parameter entries
    std::vector<std::string> requiredEntries;
    requiredEntries.emplace_back("step");
    requiredEntries.emplace_back("time");
    requiredEntries.emplace_back("displacement");
    requiredEntries.emplace_back("interface traction");
    requiredEntries.emplace_back("slave forces");
    requiredEntries.emplace_back("master forces");

    check_output_list(outputParams, requiredEntries);
  }

  // Get the discretization writer and get ready for writing
  std::shared_ptr<Core::IO::DiscretizationWriter> writer = idiscret_->writer();

  // Get output for this time step started
  {
    const int step = outputParams.get<int>("step");
    const double time = outputParams.get<double>("time");

    writer->clear_map_cache();
    writer->write_mesh(step, time);
    writer->new_step(step, time);
  }

  /* Write interface displacement
   *
   * The interface displacement has been handed in via the parameter list outParams.
   * Grab it from there, then use Core::LinAlg::export_to() to extract the interface
   * portion from the global displacement vector. Finally, write the interface
   * portion using this interfaces' discretization writer.
   */
  {
    // Get full displacement vector and extract interface displacement
    std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
        outputParams.get<std::shared_ptr<const Core::LinAlg::Vector<double>>>("displacement");
    std::shared_ptr<Core::LinAlg::Vector<double>> iDisp =
        std::make_shared<Core::LinAlg::Vector<double>>(*idiscret_->dof_row_map());
    Core::LinAlg::export_to(*disp, *iDisp);

    // Write the interface displacement field
    writer->write_vector("displacement", iDisp, Core::IO::VectorType::dofvector);
  }

  // Write Lagrange multiplier field
  {
    // Get full Lagrange multiplier vector and extract values of this interface
    std::shared_ptr<const Core::LinAlg::Vector<double>> lagMult =
        outputParams.get<std::shared_ptr<const Core::LinAlg::Vector<double>>>("interface traction");
    std::shared_ptr<Core::LinAlg::Vector<double>> iLagMult =
        std::make_shared<Core::LinAlg::Vector<double>>(*idiscret_->dof_row_map());
    Core::LinAlg::export_to(*lagMult, *iLagMult);

    // Write this interface's Lagrange multiplier field
    writer->write_vector("interfacetraction", iLagMult, Core::IO::VectorType::dofvector);
  }

  // Write normal contact stress
  {
    // Get values from parameter list and export to interface dof_row_map
    std::shared_ptr<const Core::LinAlg::Vector<double>> normalStresses =
        outputParams.get<std::shared_ptr<const Core::LinAlg::Vector<double>>>("norcontactstress");
    std::shared_ptr<Core::LinAlg::Vector<double>> iNormalStresses =
        std::make_shared<Core::LinAlg::Vector<double>>(*idiscret_->dof_row_map());
    Core::LinAlg::export_to(*normalStresses, *iNormalStresses);

    // Write this interface's normal contact stress field
    writer->write_vector("norcontactstress", iNormalStresses, Core::IO::VectorType::dofvector);
  }

  // Write tangential contact stress
  {
    // Get values from parameter list and export to interface dof_row_map
    std::shared_ptr<const Core::LinAlg::Vector<double>> tangentialStresses =
        outputParams.get<std::shared_ptr<const Core::LinAlg::Vector<double>>>("tancontactstress");
    std::shared_ptr<Core::LinAlg::Vector<double>> iTangentialStresses =
        std::make_shared<Core::LinAlg::Vector<double>>(*idiscret_->dof_row_map());
    Core::LinAlg::export_to(*tangentialStresses, *iTangentialStresses);

    // Write this interface's normal contact stress field
    writer->write_vector("tancontactstress", iTangentialStresses, Core::IO::VectorType::dofvector);
  }

  // Write nodal forces of slave side
  {
    // Get nodal forces
    std::shared_ptr<const Core::LinAlg::Vector<double>> slaveforces =
        outputParams.get<std::shared_ptr<const Core::LinAlg::Vector<double>>>("slave forces");
    std::shared_ptr<Core::LinAlg::Vector<double>> forces =
        std::make_shared<Core::LinAlg::Vector<double>>(*idiscret_->dof_row_map());
    Core::LinAlg::export_to(*slaveforces, *forces);

    // Write to output
    writer->write_vector("slaveforces", forces, Core::IO::VectorType::dofvector);
  }

  // Write nodal forces of master side
  {
    // Get nodal forces
    std::shared_ptr<const Core::LinAlg::Vector<double>> masterforces =
        outputParams.get<std::shared_ptr<const Core::LinAlg::Vector<double>>>("master forces");
    std::shared_ptr<Core::LinAlg::Vector<double>> forces =
        std::make_shared<Core::LinAlg::Vector<double>>(*idiscret_->dof_row_map());
    Core::LinAlg::export_to(*masterforces, *forces);

    // Write to output
    writer->write_vector("masterforces", forces, Core::IO::VectorType::dofvector);
  }


  // Nodes: node-based vector with '0' at slave nodes and '1' at master nodes
  {
    Core::LinAlg::Vector<double> masterVec(*mnoderowmap_);
    masterVec.put_scalar(1.0);

    std::shared_ptr<const Core::LinAlg::Map> nodeRowMap =
        Core::LinAlg::merge_map(snoderowmap_, mnoderowmap_, false);
    std::shared_ptr<Core::LinAlg::Vector<double>> masterSlaveVec =
        std::make_shared<Core::LinAlg::Vector<double>>(*nodeRowMap, true);
    Core::LinAlg::export_to(masterVec, *masterSlaveVec);

    writer->write_vector("slavemasternodes", masterSlaveVec, Core::IO::VectorType::nodevector);
  }

  // Write active set
  {
    // evaluate active set and slip set
    Core::LinAlg::Vector<double> activeset(*activenodes_);
    activeset.put_scalar(1.0);

    if (is_friction())
    {
      Core::LinAlg::Vector<double> slipset(*slipnodes_);
      slipset.put_scalar(1.0);
      Core::LinAlg::Vector<double> slipsetexp(*activenodes_);
      Core::LinAlg::export_to(slipset, slipsetexp);
      activeset.update(1.0, slipsetexp, 1.0);
    }

    // export to interface node row map
    std::shared_ptr<Core::LinAlg::Vector<double>> activesetexp =
        std::make_shared<Core::LinAlg::Vector<double>>(*(idiscret_->node_row_map()));
    Core::LinAlg::export_to(activeset, *activesetexp);

    writer->write_vector("activeset", activesetexp, Core::IO::VectorType::nodevector);
  }

  // Elements: element-based vector with '0' at slave elements and '1' at master elements
  {
    Core::LinAlg::Vector<double> masterVec(*melerowmap_);
    masterVec.put_scalar(1.0);

    std::shared_ptr<const Core::LinAlg::Map> eleRowMap =
        Core::LinAlg::merge_map(selerowmap_, melerowmap_, false);
    std::shared_ptr<Core::LinAlg::Vector<double>> masterSlaveVec =
        std::make_shared<Core::LinAlg::Vector<double>>(*eleRowMap, true);
    Core::LinAlg::export_to(masterVec, *masterSlaveVec);

    writer->write_vector(
        "slavemasterelements", masterSlaveVec, Core::IO::VectorType::elementvector);
  }

  // Write element owners
  {
    std::shared_ptr<const Core::LinAlg::Map> eleRowMap =
        Core::LinAlg::merge_map(selerowmap_, melerowmap_, false);
    std::shared_ptr<Core::LinAlg::Vector<double>> owner =
        std::make_shared<Core::LinAlg::Vector<double>>(*eleRowMap);

    for (int i = 0; i < idiscret_->element_row_map()->num_my_elements(); ++i)
      (*owner).get_values()[i] = idiscret_->l_row_element(i)->owner();

    writer->write_vector("Owner", owner, Core::IO::VectorType::elementvector);
  }
}

FOUR_C_NAMESPACE_CLOSE
