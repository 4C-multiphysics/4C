// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_coupling_adapter_mortar.hpp"

#include "4C_fem_condition_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_io.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_mortar_coupling3d_classes.hpp"
#include "4C_mortar_element.hpp"
#include "4C_mortar_interface.hpp"
#include "4C_mortar_node.hpp"
#include "4C_mortar_utils.hpp"
#include "4C_utils_function_manager.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Coupling::Adapter::CouplingMortar::CouplingMortar(int spatial_dimension,
    Teuchos::ParameterList mortar_coupling_params, Teuchos::ParameterList contact_dynamic_params,
    Core::FE::ShapeFunctionType shape_function_type)
    : spatial_dimension_(spatial_dimension),
      mortar_coupling_params_(mortar_coupling_params),
      contact_dynamic_params_(contact_dynamic_params),
      shape_function_type_(shape_function_type),
      issetup_(false)
{
  ;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Coupling::Adapter::CouplingMortar::setup(
    const std::shared_ptr<Core::FE::Discretization>& target_dis,
    const std::shared_ptr<Core::FE::Discretization>& source_dis,
    const std::shared_ptr<Core::FE::Discretization>& aledis, const std::vector<int>& coupleddof,
    const std::string& couplingcond, MPI_Comm comm,
    const Core::Utils::FunctionManager& function_manager,
    const Teuchos::ParameterList& binning_params,
    const std::map<std::string, std::shared_ptr<Core::FE::Discretization>>& discretization_map,
    std::shared_ptr<Core::IO::OutputControl> output_control,
    const Core::FE::ShapeFunctionType spatial_approximation_type, const bool source_is_ale,
    const bool slidingale, const int target_dofset_number, const int source_dofset_number)
{
  // initialize maps for row nodes
  std::map<int, Core::Nodes::Node*> target_nodes;
  std::map<int, Core::Nodes::Node*> source_nodes;

  // initialize maps for column nodes
  std::map<int, Core::Nodes::Node*> target_global_nodes;
  std::map<int, Core::Nodes::Node*> source_global_nodes;

  // initialize maps for elements
  std::map<int, std::shared_ptr<Core::Elements::Element>> target_elements;
  std::map<int, std::shared_ptr<Core::Elements::Element>> source_elements;

  // Coupling condition is defined by "MORTAR COUPLING CONDITIONS"
  // There is only one discretization (target_dis == source_dis). Therefore, the node set have to be
  // separated beforehand.
  if (couplingcond == "Mortar" || couplingcond == "MortarMulti")
  {
    std::vector<const Core::Conditions::Condition*> conds;
    std::vector<const Core::Conditions::Condition*> conds_target;
    std::vector<const Core::Conditions::Condition*> conds_source;
    target_dis->get_condition(couplingcond, conds);

    for (unsigned i = 0; i < conds.size(); i++)
    {
      const std::string& side = conds[i]->parameters().get<std::string>("Side");

      if (side == "Master")
        conds_target.push_back(conds[i]);
      else if (side == "Slave")
        conds_source.push_back(conds[i]);
    }

    // Fill maps based on condition for target side (target_dis == source_dis)
    Core::Conditions::find_condition_objects(
        *target_dis, target_nodes, target_global_nodes, target_elements, conds_target);

    // Fill maps based on condition for source side (target_dis == source_dis)
    Core::Conditions::find_condition_objects(
        *source_dis, source_nodes, source_global_nodes, source_elements, conds_source);
  }
  // Coupling condition is defined by "FSI COUPLING CONDITIONS"
  // There are two discretizations for the target and source side. Therefore, the target/source
  // nodes are chosen based on the discretization.
  else
  {
    // Fill maps based on condition for target side (target_dis != source_dis)
    Core::Conditions::find_condition_objects(
        *target_dis, target_nodes, target_global_nodes, target_elements, couplingcond);

    // Fill maps based on condition for source side (target_dis != source_dis)
    Core::Conditions::find_condition_objects(
        *source_dis, source_nodes, source_global_nodes, source_elements, couplingcond);
  }

  // number of coupled dofs (defined in coupleddof by a 1)
  int numcoupleddof = 0;
  for (unsigned ii = 0; ii < coupleddof.size(); ++ii)
    if (coupleddof[ii] == 1) ++numcoupleddof;

  // setup mortar interface
  setup_interface(target_dis, source_dis, coupleddof, target_global_nodes, source_global_nodes,
      target_elements, source_elements, comm, binning_params, discretization_map, output_control,
      spatial_approximation_type, source_is_ale, slidingale, target_dofset_number,
      source_dofset_number);

  // all the following stuff has to be done once in setup
  // in order to get initial D_ and M_

  // processor ID
  const int myrank = Core::Communication::my_mpi_rank(target_dis->get_comm());

  // get mortar coupling parameters
  Teuchos::ParameterList inputmortar;
  inputmortar.setParameters(contact_dynamic_params_);
  inputmortar.setParameters(mortar_coupling_params_);

  // interface displacement (=0) has to be merged from source and target discretization
  std::shared_ptr<Core::LinAlg::Map> dofrowmap =
      Core::LinAlg::merge_map(target_dof_row_map_, source_dof_row_map_, false);
  std::shared_ptr<Core::LinAlg::Vector<double>> dispn =
      std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);

  // set displacement state in mortar interface
  interface_->set_state(Mortar::state_new_displacement, *dispn);

  // print message
  if (myrank == 0)
  {
    std::cout << "\nPerforming mortar coupling...............";
    fflush(stdout);
  }

  // evaluate interface
  evaluate();

  // print message
  if (myrank == 0) std::cout << "done!" << std::endl << std::endl;

  // initial mesh relocation:
  // For curved internal or fsi coupling interfaces, a mesh relocation is critical,
  // since the integration over curved interface (generation of mortar coupling
  // matrices) results in inaccuracies. These inaccuracies may lead to undesired node
  // displacements.
  // Example: nodes at the interface are also moved for matching discretizations
  // (P should be "unity matrix")!
  if (Teuchos::getIntegralValue<Mortar::MeshRelocation>(inputmortar, "MESH_RELOCATION") ==
      Mortar::relocation_initial)
  {
    // Warning:
    // Mesh relocation is not possible if coupled degrees of freedom are less than
    // the spatial dimensions!
    if (numcoupleddof < spatial_dimension_)
    {
      std::cout << "Warning: " << std::endl;
      std::cout
          << "Initial mesh relocation is not possible, since the coupled degrees of freedom are "
          << std::endl;
      std::cout << "less than the spatial dimensions!!" << std::endl;
      std::cout << "Additional information is provided by comments in the code!" << std::endl;
    }

    // Originally, this method was written for structural problems coupling the
    // spatial displacements. Therefore, the source and target map could be also used
    // to store the coordinates of the interface nodes, which is necessary to perform
    // the mesh relocation. Hence, this method cannot be used for problem types such
    // as elch, scatra, etc., having less coupling degrees of freedom than spatial
    // dimensions.
    std::shared_ptr<Core::LinAlg::Vector<double>> idisp(nullptr);
    mesh_relocation(
        *source_dis, aledis, target_dof_row_map_, source_dof_row_map_, idisp, comm, source_is_ale);
  }

  // matrix transformation to initial parallel distribution
  matrix_row_col_transform();

  // check if source dofs have dirichlet constraints
  check_source_dirichlet_overlap(source_dis, comm, function_manager);
}


/*----------------------------------------------------------------------*
 | check for overlap of source and Dirichlet boundaries      farah 02/16 |
 *----------------------------------------------------------------------*/
void Coupling::Adapter::CouplingMortar::check_source_dirichlet_overlap(
    const std::shared_ptr<Core::FE::Discretization>& source_dis, MPI_Comm comm,
    const Core::Utils::FunctionManager& function_manager)
{
  // safety check
  check_setup();

  // check for overlap of source and Dirichlet boundaries
  // (this is not allowed in order to avoid over-constraint)
  bool overlap = false;
  Teuchos::ParameterList p;
  p.set("total time", 0.0);
  p.set<const Core::Utils::FunctionManager*>("function_manager", &function_manager);
  std::shared_ptr<Core::LinAlg::MapExtractor> dbcmaps =
      std::make_shared<Core::LinAlg::MapExtractor>();
  std::shared_ptr<Core::LinAlg::Vector<double>> temp =
      std::make_shared<Core::LinAlg::Vector<double>>(*(source_dis->dof_row_map()), true);
  source_dis->evaluate_dirichlet(p, temp, nullptr, nullptr, nullptr, dbcmaps);

  // loop over all source row nodes of the interface
  for (int j = 0; j < interface_->source_row_nodes()->num_my_elements(); ++j)
  {
    int gid = interface_->source_row_nodes()->gid(j);
    Core::Nodes::Node* node = interface_->discret().g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Mortar::Node* mtnode = static_cast<Mortar::Node*>(node);

    // check if this node's dofs are in dbcmap
    for (int k = 0; k < mtnode->num_dof(); ++k)
    {
      int currdof = mtnode->dofs()[k];
      int lid = (dbcmaps->cond_map())->lid(currdof);

      // found source node with dbc
      if (lid >= 0)
      {
        overlap = true;
        break;
      }
    }
  }

  // print warning message to screen
  if (overlap && Core::Communication::my_mpi_rank(comm) == 0)
  {
    if (Core::Communication::my_mpi_rank(comm) == 0)
      FOUR_C_THROW(
          "Source boundary and Dirichlet boundary conditions overlap!\n"
          "This leads to an over-constraint problem setup");
  }

  return;
}


/*----------------------------------------------------------------------*
 | setup routine for mortar framework                        ehrl 08/13 |
 *----------------------------------------------------------------------*/
void Coupling::Adapter::CouplingMortar::setup_interface(
    const std::shared_ptr<Core::FE::Discretization>& target_dis,
    const std::shared_ptr<Core::FE::Discretization>& source_dis, const std::vector<int>& coupleddof,
    const std::map<int, Core::Nodes::Node*>& target_global_nodes,
    const std::map<int, Core::Nodes::Node*>& source_global_nodes,
    const std::map<int, std::shared_ptr<Core::Elements::Element>>& target_elements,
    const std::map<int, std::shared_ptr<Core::Elements::Element>>& source_elements, MPI_Comm comm,
    const Teuchos::ParameterList& binning_params,
    const std::map<std::string, std::shared_ptr<Core::FE::Discretization>>& discretization_map,
    std::shared_ptr<Core::IO::OutputControl> output_control,
    const Core::FE::ShapeFunctionType spatial_approximation_type, const bool source_is_ale,
    const bool slidingale, const int target_dofset_number, const int source_dofset_number)
{
  // vector coupleddof defines degree of freedom which are coupled (1: coupled; 0: not coupled),
  // e.g.:
  // - fluid 3D meshtying: coupleddof = [1, 1, 1, 1] -> all degrees of freedom (velocity and
  // pressure) are coupled
  // - fluid 3D meshtying: coupleddof = [1, 1, 1, 0] -> only velocity degrees of freedom are coupled
  // - fsi 3D: coupleddof = [1, 1, 1] -> at the interface only displacements are coupled
  // - ....

  // get mortar coupling parameters
  Teuchos::ParameterList input;
  input.setParameters(mortar_coupling_params_);
  input.setParameters(contact_dynamic_params_);

  // is this a nurbs problem?
  const bool nurbs = shape_function_type_ == Core::FE::ShapeFunctionType::nurbs;
  input.set<bool>("NURBS", nurbs);

  // set valid parameter values
  input.set<Mortar::ShapeFcn>("LM_SHAPEFCN", Mortar::ShapeFcn::shape_dual);
  input.set<Mortar::ConsistentDualType>(
      "LM_DUAL_CONSISTENT", Mortar::ConsistentDualType::consistent_none);
  input.sublist("PARALLEL REDISTRIBUTION")
      .set<Mortar::ParallelRedist>("PARALLEL_REDIST", Mortar::ParallelRedist::redist_none);
  input.set<int>("DIMENSION", spatial_dimension_);

  // create an empty mortar interface
  interface_ = Mortar::Interface::create(
      0, comm, spatial_dimension_, input, output_control, spatial_approximation_type);

  // number of dofs per node based on the coupling vector coupleddof
  const int dof = coupleddof.size();
  if ((target_dis->num_my_row_nodes() > 0 and
          (target_dis->num_dof(target_dofset_number, target_dis->l_row_node(0)) != dof and
              source_is_ale == true and slidingale == false)) or
      (source_dis->num_my_row_nodes() > 0 and
          (source_dis->num_dof(source_dofset_number, source_dis->l_row_node(0)) != dof and
              source_is_ale == false and slidingale == false)))
  {
    FOUR_C_THROW(
        "The size of the coupling vector coupleddof and dof defined in the discretization does not "
        "fit!! \n"
        "dof defined in the discretization: {} \n"
        "length of coupleddof: {}",
        target_dis->num_dof(target_dofset_number, target_dis->l_row_node(0)), dof);
  }

  // special case: sliding ale
  // In the sliding ale framework two mortar discretizations are generated from identical
  // target element and source element sets. Since node-, dof- and element ids of the original
  // elements are the same, an offset have to be defined
  int nodeoffset = 0;
  int dofoffset = 0;
  if (slidingale == true)
  {
    nodeoffset = target_dis->node_row_map()->max_all_gid() + 1;
    dofoffset = target_dis->dof_row_map(target_dofset_number)->max_all_gid() + 1;
  }

  // number of coupled dofs (defined in coupleddof by a 1)
  int numcoupleddof = 0;
  for (int ii = 0; ii < dof; ++ii)
    if (coupleddof[ii] == 1) ++numcoupleddof;

  // feeding target nodes to the interface including ghosted nodes
  std::map<int, Core::Nodes::Node*>::const_iterator nodeiter;
  for (nodeiter = target_global_nodes.begin(); nodeiter != target_global_nodes.end(); ++nodeiter)
  {
    Core::Nodes::Node* node = nodeiter->second;
    // vector containing only the gids of the coupled dofs (size numcoupleddof)
    std::vector<int> dofids(numcoupleddof);
    int ii = 0;
    for (int k = 0; k < dof; ++k)
    {
      // Should this dof be coupled? (==1),
      if (coupleddof[k] == 1)
      {
        // get the gid of the coupled dof (size dof)
        // and store it in the vector dofids containing only coupled dofs (size numcoupleddof)
        dofids[ii] = target_dis->dof(target_dofset_number, node)[k];
        ii += 1;
      }
    }
    std::shared_ptr<Mortar::Node> mrtrnode =
        std::make_shared<Mortar::Node>(node->id(), node->x(), node->owner(), dofids, false);

    if (nurbs) Mortar::Utils::prepare_nurbs_node(node, *mrtrnode);
    interface_->add_mortar_node(mrtrnode);
  }

  // feeding source nodes to the interface including ghosted nodes
  for (nodeiter = source_global_nodes.begin(); nodeiter != source_global_nodes.end(); ++nodeiter)
  {
    Core::Nodes::Node* node = nodeiter->second;
    // vector containing only the gids of the coupled dofs (size numcoupleddof)
    std::vector<int> dofids(numcoupleddof);
    int ii = 0;
    for (int k = 0; k < dof; ++k)
    {
      // Should this dof be coupled? (==1)
      if (coupleddof[k] == 1)
      {
        // get the gid of the coupled dof (size dof)
        // and store it in the vector dofids containing only coupled dofs (size numcoupleddof)
        dofids[ii] = source_dis->dof(source_dofset_number, node)[k] + dofoffset;
        ii += 1;
      }
    }
    std::shared_ptr<Mortar::Node> mrtrnode = std::make_shared<Mortar::Node>(
        node->id() + nodeoffset, node->x(), node->owner(), dofids, true);

    if (nurbs) Mortar::Utils::prepare_nurbs_node(node, *mrtrnode);
    interface_->add_mortar_node(mrtrnode);
  }

  // We need to determine an element offset to start the numbering of the source
  // mortar elements AFTER the target mortar elements in order to ensure unique
  // eleIDs in the interface discretization. The element offset equals the
  // overall number of target mortar elements (which is not equal to the number
  // of elements in the field that is chosen as target side).
  //
  // If target_dis==source_dis, the element numbering is right without offset
  int eleoffset = 0;
  if (target_dis.get() != source_dis.get())
  {
    int num_target_mortar_elements = target_elements.size();
    eleoffset = Core::Communication::sum_all(num_target_mortar_elements, comm);
  }

  if (slidingale == true) eleoffset = target_dis->element_row_map()->max_all_gid() + 1;

  // feeding target elements to the interface
  std::map<int, std::shared_ptr<Core::Elements::Element>>::const_iterator elemiter;
  for (elemiter = target_elements.begin(); elemiter != target_elements.end(); ++elemiter)
  {
    std::shared_ptr<Core::Elements::Element> ele = elemiter->second;
    std::shared_ptr<Mortar::Element> mrtrele = std::make_shared<Mortar::Element>(
        ele->id(), ele->owner(), ele->shape(), ele->num_node(), ele->node_ids(), false, nurbs);

    if (nurbs) Mortar::Utils::prepare_nurbs_element(*target_dis, ele, *mrtrele, spatial_dimension_);
    interface_->add_mortar_element(mrtrele);
  }

  // feeding source elements to the interface
  for (elemiter = source_elements.begin(); elemiter != source_elements.end(); ++elemiter)
  {
    std::shared_ptr<Core::Elements::Element> ele = elemiter->second;

    // Here, we have to distinguish between standard and sliding ale since mortar elements are
    // generated from the identical element sets in the case of sliding ale Therefore, we introduce
    // an element offset AND a node offset for the the source mortar elements
    if (slidingale == false)
    {
      std::shared_ptr<Mortar::Element> mrtrele =
          std::make_shared<Mortar::Element>(ele->id() + eleoffset, ele->owner(), ele->shape(),
              ele->num_node(), ele->node_ids(), true, nurbs);

      if (nurbs)
        Mortar::Utils::prepare_nurbs_element(*source_dis, ele, *mrtrele, spatial_dimension_);
      interface_->add_mortar_element(mrtrele);
    }
    else
    {
      std::vector<int> nidsoff;
      for (int i = 0; i < ele->num_node(); i++)
      {
        nidsoff.push_back(ele->node_ids()[ele->num_node() - 1 - i] + nodeoffset);
      }

      std::shared_ptr<Mortar::Element> mrtrele =
          std::make_shared<Mortar::Element>(ele->id() + eleoffset, ele->owner(), ele->shape(),
              ele->num_node(), nidsoff.data(), true, nurbs);

      interface_->add_mortar_element(mrtrele);
    }
  }

  /* Finalize the interface construction
   *
   * If this is the final parallel distribution, we need to assign degrees of freedom during
   * during fill_complete(). If parallel redistribution is enabled, there will be another call to
   * fill_complete(), so we skip this expensive operation here and do it later. DOFs have to be
   * assigned only once!
   */
  const Mortar::ParallelRedist parallelRedist = Teuchos::getIntegralValue<Mortar::ParallelRedist>(
      input.sublist("PARALLEL REDISTRIBUTION"), "PARALLEL_REDIST");
  {
    bool isFinalDistribution = false;
    if (parallelRedist == Mortar::ParallelRedist::redist_none or
        Core::Communication::num_mpi_ranks(comm) == 1)
      isFinalDistribution = true;

    interface_->fill_complete(discretization_map, binning_params, output_control,
        spatial_approximation_type, isFinalDistribution);
  }

  // set setup flag!
  issetup_ = true;

  // store old row maps (before parallel redistribution)
  psource_dof_row_map_ = std::make_shared<Core::LinAlg::Map>(*interface_->source_row_dofs());
  ptarget_dof_row_map_ = std::make_shared<Core::LinAlg::Map>(*interface_->target_row_dofs());

  // print parallel distribution
  interface_->print_parallel_distribution();

  //**********************************************************************
  // PARALLEL REDISTRIBUTION OF INTERFACE
  //**********************************************************************
  if (parallelRedist != Mortar::ParallelRedist::redist_none and
      Core::Communication::num_mpi_ranks(comm) > 1)
  {
    // redistribute optimally among all procs
    interface_->redistribute();

    // call fill complete again
    interface_->fill_complete(
        discretization_map, binning_params, output_control, spatial_approximation_type, true);

    // print parallel distribution again
    interface_->print_parallel_distribution();
  }
  //**********************************************************************

  // store row maps (after parallel redistribution)
  source_dof_row_map_ = std::make_shared<Core::LinAlg::Map>(*interface_->source_row_dofs());
  target_dof_row_map_ = std::make_shared<Core::LinAlg::Map>(*interface_->target_row_dofs());

  // create binary search tree
  interface_->create_search_tree();

  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Coupling::Adapter::CouplingMortar::mesh_relocation(Core::FE::Discretization& source_dis,
    std::shared_ptr<Core::FE::Discretization> aledis,
    std::shared_ptr<const Core::LinAlg::Map> target_dof_row_map,
    std::shared_ptr<const Core::LinAlg::Map> source_dof_row_map,
    std::shared_ptr<Core::LinAlg::Vector<double>>& idisp, MPI_Comm comm, bool source_is_ale)
{
  // safety check
  check_setup();

  // problem dimension
  const int dim = spatial_dimension_;

  //**********************************************************************
  // (0) check constraints in reference configuration
  //**********************************************************************
  // build global vectors of source and target coordinates
  std::shared_ptr<Core::LinAlg::Vector<double>> xs =
      std::make_shared<Core::LinAlg::Vector<double>>(*source_dof_row_map, true);
  std::shared_ptr<Core::LinAlg::Vector<double>> xm =
      std::make_shared<Core::LinAlg::Vector<double>>(*target_dof_row_map, true);

  // loop over all source row nodes
  for (int j = 0; j < interface_->source_row_nodes()->num_my_elements(); ++j)
  {
    int gid = interface_->source_row_nodes()->gid(j);
    Core::Nodes::Node* node = interface_->discret().g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Mortar::Node* mtnode = static_cast<Mortar::Node*>(node);

    // prepare assembly
    // minimum three coupling dof's otherwise this method is not working
    // since the source map is based on dof<dim
    // -> spacial coordinates cannot be assembled in vector based on this map
    Core::LinAlg::SerialDenseVector val(dim);
    std::vector<int> lm(dim, 0.0);
    std::vector<int> lmowner(dim, 0.0);

    for (int k = 0; k < dim; ++k)
    {
      val[k] = mtnode->x()[k];
      lm[k] = mtnode->dofs()[k];
      lmowner[k] = mtnode->owner();
    }


    // add ALE displacements, if required
    if (idisp != nullptr)
    {
      // get degrees of freedom of a node
      std::vector<int> gdofs = interface_->discret().dof(node);

      for (int k = 0; k < dim; ++k)
      {
        val[k] += idisp->local_values_as_span()[(idisp->get_map()).lid(gdofs[k])];
      }
    }

    // do assembly
    Core::LinAlg::assemble(*xs, val, lm, lmowner);
  }

  // loop over all target row nodes
  for (int j = 0; j < interface_->target_row_nodes()->num_my_elements(); ++j)
  {
    int gid = interface_->target_row_nodes()->gid(j);
    Core::Nodes::Node* node = interface_->discret().g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Mortar::Node* mtnode = static_cast<Mortar::Node*>(node);

    // prepare assembly
    // minimum three coupling dof's otherwise this method is not working
    // since the source map is based on dof<dim
    // -> spacial coordinates cannot be assembled in vector based on this map
    Core::LinAlg::SerialDenseVector val(dim);
    std::vector<int> lm(dim, 0.0);
    std::vector<int> lmowner(dim, 0.0);

    for (int k = 0; k < dim; ++k)
    {
      val[k] = mtnode->x()[k];
      lm[k] = mtnode->dofs()[k];
      lmowner[k] = mtnode->owner();
    }

    // add ALE displacements, if required
    if (idisp != nullptr)
    {
      // get degrees of freedom of a node
      std::vector<int> gdofs = interface_->discret().dof(node);

      for (int k = 0; k < dim; ++k)
      {
        val[k] += idisp->local_values_as_span()[(idisp->get_map()).lid(gdofs[k])];
      }
    }

    // do assembly
    Core::LinAlg::assemble(*xm, val, lm, lmowner);
  }

  // compute g-vector at global level
  std::shared_ptr<Core::LinAlg::Vector<double>> Dxs =
      std::make_shared<Core::LinAlg::Vector<double>>(*source_dof_row_map);
  D_->multiply(false, *xs, *Dxs);
  std::shared_ptr<Core::LinAlg::Vector<double>> Mxm =
      std::make_shared<Core::LinAlg::Vector<double>>(*source_dof_row_map);
  M_->multiply(false, *xm, *Mxm);
  std::shared_ptr<Core::LinAlg::Vector<double>> gold =
      std::make_shared<Core::LinAlg::Vector<double>>(*source_dof_row_map, true);
  gold->update(1.0, *Dxs, 1.0);
  gold->update(-1.0, *Mxm, 1.0);
  double gnorm = 0.0;
  gold->norm_2(&gnorm);
  gnorm /= sqrt((double)gold->global_length());  // scale with length of vector

  const double tol = 1.0e-12;
  // no need to do mesh relocation if g already very small

  if (Core::Communication::my_mpi_rank(comm) == 0)
  {
    std::cout << "Analyze interface quality: L2-norm of gap vector = " << gnorm
              << " whereas tol = " << tol << std::endl;

    if (gnorm < tol) std::cout << "  --> Mesh relocation is not necessary. " << std::endl;
  }

  if (gnorm < tol) return;

  // print message
  if (Core::Communication::my_mpi_rank(comm) == 0)
  {
    std::cout << "Performing mesh relocation...........";
    fflush(stdout);
  }

  //**********************************************************************
  // perform mesh relocation node by node
  //**********************************************************************
  // IMPORTANT NOTE:
  // We have to be very careful on which nodes on which processor to
  // relocate! Basically, every processor needs to know about relocation
  // of all its column nodes in the standard column map with overlap=1,
  // because all these nodes participate in the processor's element
  // evaluation! Thus, the modified source positions are first exported
  // to the column map of the respective interface and the modification
  // loop is then also done with respect to this node column map!
  // A second concern is that we are dealing with a special interface
  // discretization (including special meshtying nodes, too) here, This
  // interface discretization has been set up for dealing with meshtying
  // ONLY, and there is still the underlying problem discretization
  // dealing with the classical finite element evaluation. Thus, it is
  // very important that we apply the nodal relocation to BOTH the
  // Mortar::Node(s) in the meshtying interface discretization AND to the
  // Nodes in the underlying problem discretization.
  // Finally, we have to ask ourselves whether the node column distribution
  // of the source nodes in the interface discretization is IDENTICAL
  // to the distribution in the underlying problem discretization. This
  // is NOT necessarily the case, as we might have redistributed the
  // interface among all processors. Thus, we loop over the fully over-
  // lapping source column map here to keep all processors around. Then,
  // the first modification (Mortar::Node) is always performed, but the
  // second modification (Core::Nodes::Node) is only performed if the respective
  // node in contained in the problem node column map.
  //**********************************************************************

  //**********************************************************************
  // (1) get target positions on global level
  //**********************************************************************
  // fill X_target first
  std::shared_ptr<Core::LinAlg::Vector<double>> X_target =
      std::make_shared<Core::LinAlg::Vector<double>>(*target_dof_row_map, true);

  // loop over all target row nodes on the current interface
  for (int j = 0; j < interface_->target_row_nodes()->num_my_elements(); ++j)
  {
    int gid = interface_->target_row_nodes()->gid(j);
    Core::Nodes::Node* node = interface_->discret().g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Mortar::Node* mtnode = static_cast<Mortar::Node*>(node);

    // do assembly (overwrite duplicate nodes)
    // minimum three coupling dof's otherwise this method is not working
    // since the source map is based on dof<dim
    // -> spacial coordinates cannot be assembled in vector based on this map
    for (int k = 0; k < dim; ++k)
    {
      int dof = mtnode->dofs()[k];
      (*X_target).get_values()[(X_target->get_map()).lid(dof)] = mtnode->x()[k];

      // add ALE displacements, if required
      if (idisp != nullptr)
        (*X_target).get_values()[(X_target->get_map()).lid(dof)] +=
            idisp->local_values_as_span()[(idisp->get_map()).lid(dof)];
    }
  }

  //**********************************************************************
  // (2) solve for modified source positions on global level
  //**********************************************************************
  // relocate modified source positions
  std::shared_ptr<Core::LinAlg::Vector<double>> X_source_mod =
      std::make_shared<Core::LinAlg::Vector<double>>(*source_dof_row_map, true);

  // this is trivial for dual Lagrange multipliers
  P_->multiply(false, *X_target, *X_source_mod);


  //**********************************************************************
  // (3) perform mesh relocation node by node
  //**********************************************************************
  // export X_source_mod to fully overlapping column map for current interface
  std::shared_ptr<Core::LinAlg::Map> fullsdofs =
      Core::LinAlg::allreduce_e_map(*(interface_->source_row_dofs()));
  std::shared_ptr<Core::LinAlg::Map> fullsnodes =
      Core::LinAlg::allreduce_e_map(*(interface_->source_row_nodes()));
  Core::LinAlg::Vector<double> X_source_mod_col(*fullsdofs, false);
  Core::LinAlg::export_to(*X_source_mod, X_source_mod_col);

  // loop over all source nodes on the current interface
  for (int j = 0; j < fullsnodes->num_my_elements(); ++j)
  {
    // get global ID of current node
    int gid = fullsnodes->gid(j);

    // be careful to modify BOTH mtnode in interface discret ...
    // (check if the node is available on this processor)
    bool isininterfacecolmap = false;
    int ilid = interface_->source_col_nodes()->lid(gid);
    if (ilid >= 0) isininterfacecolmap = true;
    Core::Nodes::Node* node = nullptr;
    Mortar::Node* mtnode = nullptr;
    if (isininterfacecolmap)
    {
      node = interface_->discret().g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      mtnode = static_cast<Mortar::Node*>(node);
    }

    // ... AND standard node in underlying source discret
    // (check if the node is available on this processor)
    bool isinproblemcolmap = false;
    int lid = source_dis.node_col_map()->lid(gid);
    if (lid >= 0) isinproblemcolmap = true;
    Core::Nodes::Node* pnode = nullptr;
    if (isinproblemcolmap)
    {
      pnode = source_dis.g_node(gid);
      if (!pnode) FOUR_C_THROW("Cannot find node with gid %", gid);
    }

    // ... AND standard node in ALE discret if fluid=source
    // (check if the node is available on this processor)
    bool isinproblemcolmap2 = false;
    Core::Nodes::Node* alenode = nullptr;
    if (aledis != nullptr)
    {
      int lid2 = aledis->node_col_map()->lid(gid);
      if (lid2 >= 0) isinproblemcolmap2 = true;
      if (isinproblemcolmap2)
      {
        alenode = aledis->g_node(gid);
        if (source_is_ale and not alenode) FOUR_C_THROW("Cannot find node with gid %", gid);
      }
    }

    // old and new nodal position and problem dimension
    std::array<double, 3> Xold = {0.0, 0.0, 0.0};
    std::array<double, 3> Xnew = {0.0, 0.0, 0.0};
    std::array<double, 3> Xnewglobal = {0.0, 0.0, 0.0};

    //******************************************************************
    // compute new nodal position
    //******************************************************************
    // first sort out procs that do not know of mtnode
    if (isininterfacecolmap)
    {
      // owner processor of this node will do computation
      if (Core::Communication::my_mpi_rank(comm) == mtnode->owner())
      {
        // get corresponding entries from X_source_mod
        int numdim = mtnode->n_dim();

        // find DOFs of current node in X_source_mod and extract this node's position
        std::vector<int> locindex(numdim);

        for (int k = 0; k < numdim; ++k)
        {
          locindex[k] = (X_source_mod_col.get_map()).lid(mtnode->dofs()[k]);
          if (locindex[k] < 0) FOUR_C_THROW("Did not find dof in map");
          Xnew[k] = X_source_mod_col.local_values_as_span()[locindex[k]];
          Xold[k] = mtnode->x()[k];
          if (idisp != nullptr)
            Xold[k] += idisp->local_values_as_span()[(idisp->get_map())
                    .lid(interface_->discret().dof(node)[k])];
        }

        // check is mesh distortion is still OK
        // (throw a FOUR_C_THROW if length of relocation is larger than 80%
        // of an adjacent element edge -> see Puso, IJNME, 2004)
        const double limit = 0.8;
        double relocation = 0.0;
        if (dim == 2)
        {
          relocation = sqrt((Xnew[0] - Xold[0]) * (Xnew[0] - Xold[0]) +
                            (Xnew[1] - Xold[1]) * (Xnew[1] - Xold[1]));
        }
        else if (dim == 3)
        {
          relocation = sqrt((Xnew[0] - Xold[0]) * (Xnew[0] - Xold[0]) +
                            (Xnew[1] - Xold[1]) * (Xnew[1] - Xold[1]) +
                            (Xnew[2] - Xold[2]) * (Xnew[2] - Xold[2]));
        }
        else
          FOUR_C_THROW("Problem dimension must be either 2 or 3!");
        bool isok = mtnode->check_mesh_distortion(relocation, limit);
        if (!isok) FOUR_C_THROW("Mesh distortion generated by relocation is too large!");
      }
    }

    // communicate new position Xnew to all procs
    // (we can use SumAll here, as Xnew will be zero on all processors
    // except for the owner processor of the current node)
    Xnewglobal = Core::Communication::sum_all(Xnew, comm);

    // const_cast to force modified X() into mtnode
    // const_cast to force modified xspatial() into mtnode
    // const_cast to force modified X() into pnode
    // const_cast to force modified X() into alenode if fluid=source
    // (remark: this is REALLY BAD coding)
    if (Teuchos::getIntegralValue<Mortar::MeshRelocation>(
            mortar_coupling_params_, "MESH_RELOCATION") == Mortar::relocation_initial)
    {
      for (int k = 0; k < dim; ++k)
      {
        // modification in interface discretization
        if (isininterfacecolmap)
        {
          const_cast<double&>(mtnode->x()[k]) = Xnewglobal[k];
          const_cast<double&>(mtnode->xspatial()[k]) = Xnewglobal[k];
        }

        // modification in problem discretization
        if (isinproblemcolmap) const_cast<double&>(pnode->x()[k]) = Xnewglobal[k];

        // modification in ALE discretization
        if (isinproblemcolmap2 and source_is_ale)
          const_cast<double&>(alenode->x()[k]) = Xnewglobal[k];
      }
    }
    else
      FOUR_C_THROW("wrong input parameter for mortar-based MESH_RELOCATION!");
  }

  //**********************************************************************
  // (4) re-evaluate constraints in reference configuration
  //**********************************************************************
  // build global vectors of source and target coordinates
  xs = std::make_shared<Core::LinAlg::Vector<double>>(*source_dof_row_map, true);
  xm = std::make_shared<Core::LinAlg::Vector<double>>(*target_dof_row_map, true);

  // loop over all source row nodes
  for (int j = 0; j < interface_->source_row_nodes()->num_my_elements(); ++j)
  {
    int gid = interface_->source_row_nodes()->gid(j);
    Core::Nodes::Node* node = interface_->discret().g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Mortar::Node* mtnode = static_cast<Mortar::Node*>(node);

    // prepare assembly
    Core::LinAlg::SerialDenseVector val(dim);
    std::vector<int> lm(dim);
    std::vector<int> lmowner(dim);

    for (int k = 0; k < dim; ++k)
    {
      val[k] = mtnode->x()[k];
      lm[k] = mtnode->dofs()[k];
      lmowner[k] = mtnode->owner();
    }

    // add ALE displacements, if required
    if (idisp != nullptr)
    {
      // get degrees of freedom of a node
      std::vector<int> gdofs = interface_->discret().dof(node);

      for (int k = 0; k < dim; ++k)
      {
        val[k] += idisp->local_values_as_span()[(idisp->get_map()).lid(gdofs[k])];
      }
    }

    // do assembly
    Core::LinAlg::assemble(*xs, val, lm, lmowner);
  }

  // loop over all target row nodes
  for (int j = 0; j < interface_->target_row_nodes()->num_my_elements(); ++j)
  {
    int gid = interface_->target_row_nodes()->gid(j);
    Core::Nodes::Node* node = interface_->discret().g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Mortar::Node* mtnode = static_cast<Mortar::Node*>(node);

    // prepare assembly
    Core::LinAlg::SerialDenseVector val(dim);
    std::vector<int> lm(dim);
    std::vector<int> lmowner(dim);

    for (int k = 0; k < dim; ++k)
    {
      val[k] = mtnode->x()[k];
      lm[k] = mtnode->dofs()[k];
      lmowner[k] = mtnode->owner();
    }

    // add ALE displacements, if required
    if (idisp != nullptr)
    {
      // get degrees of freedom of a node
      std::vector<int> gdofs = interface_->discret().dof(node);

      for (int k = 0; k < dim; ++k)
      {
        val[k] += idisp->local_values_as_span()[(idisp->get_map()).lid(gdofs[k])];
      }
    }

    // do assembly
    Core::LinAlg::assemble(*xm, val, lm, lmowner);
  }

  // compute g-vector at global level
  Dxs = std::make_shared<Core::LinAlg::Vector<double>>(*source_dof_row_map);
  D_->multiply(false, *xs, *Dxs);
  Mxm = std::make_shared<Core::LinAlg::Vector<double>>(*source_dof_row_map);
  M_->multiply(false, *xm, *Mxm);
  std::shared_ptr<Core::LinAlg::Vector<double>> gnew =
      std::make_shared<Core::LinAlg::Vector<double>>(*source_dof_row_map, true);
  gnew->update(1.0, *Dxs, 1.0);
  gnew->update(-1.0, *Mxm, 1.0);
  gnew->norm_2(&gnorm);
  gnorm /= sqrt((double)gnew->global_length());  // scale with length of vector

  if (gnorm > tol)
    FOUR_C_THROW(
        "Mesh relocation was not successful! \n "
        "Gap norm {} is larger than tolerance {}",
        gnorm, tol);

  //**********************************************************************
  // (5) re-relocate finite elements (if source=structure)
  //**********************************************************************
  // if source=fluid, we are lucky because fluid elements do not
  // need any re-relocation (unlike structural elements)
  // fluid elements: empty implementation (return 0)
  Core::Communication::ParObjectFactory::instance().initialize_elements(source_dis);

  // print message
  if (Core::Communication::my_mpi_rank(comm) == 0)
  {
    std::cout << "done!" << std::endl
              << std::endl
              << "Analyze interface quality: L2-norm of gap vector = " << gnorm
              << " whereas tol = " << tol << std::endl;

    if (gnorm < tol) std::cout << "  --> Mesh relocation was successful. " << std::endl;
  }

  return;
}


/*----------------------------------------------------------------------*
 |  compute projection operator P                            farah 01/16|
 *----------------------------------------------------------------------*/
void Coupling::Adapter::CouplingMortar::create_p()
{
  // safety check
  check_setup();

  // check
  if (Teuchos::getIntegralValue<Mortar::ShapeFcn>(interface()->interface_params(), "LM_SHAPEFCN") !=
      Mortar::shape_dual)
    FOUR_C_THROW("Creation of P operator only for dual shape functions!");

  /********************************************************************/
  /* Multiply Mortar matrices: P = inv(D) * M         A               */
  /********************************************************************/
  D_->complete();
  Dinv_ = std::make_shared<Core::LinAlg::SparseMatrix>(*D_);
  std::shared_ptr<Core::LinAlg::Vector<double>> diag =
      std::make_shared<Core::LinAlg::Vector<double>>(*source_dof_row_map_, true);
  int err = 0;

  // extract diagonal of invd into diag
  Dinv_->extract_diagonal_copy(*diag);

  // set zero diagonal values to dummy 1.0
  for (int i = 0; i < diag->local_length(); ++i)
  {
    if (abs(diag->local_values_as_span()[i]) < 1e-12)
    {
      std::cout << "WARNING: Diagonal entry of D matrix (value = "
                << diag->local_values_as_span()[i]
                << ") is skipped because it is less than 1e-12!!!" << std::endl;
      (*diag).get_values()[i] = 1.0;
    }
  }

  // scalar inversion of diagonal values
  diag->reciprocal(*diag);

  // re-insert inverted diagonal into invd
  err = Dinv_->replace_diagonal_values(*diag);
  if (err != 0) FOUR_C_THROW("replace_diagonal_values() failed with error code {}.", err);

  // complete inverse D matrix
  Dinv_->complete();

  // do the multiplication P = inv(D) * M
  P_ = Core::LinAlg::matrix_multiply(*Dinv_, false, *M_, false, false, false, true);

  // complete the matrix
  P_->complete(*target_dof_row_map_, *source_dof_row_map_);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Coupling::Adapter::CouplingMortar::evaluate(
    std::shared_ptr<Core::LinAlg::Vector<double>> idisp)
{
  // safety check
  check_setup();

  // set new displacement state in mortar interface
  interface_->set_state(Mortar::state_new_displacement, *idisp);
  evaluate();
  matrix_row_col_transform();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Coupling::Adapter::CouplingMortar::evaluate(
    std::shared_ptr<Core::LinAlg::Vector<double>> idispma,
    std::shared_ptr<Core::LinAlg::Vector<double>> idispsl)
{
  // safety checks
  check_setup();
  FOUR_C_ASSERT(idispma->get_map().point_same_as(*ptarget_dof_row_map_),
      "Map of incoming target vector does not match the stored target dof row map.");
  FOUR_C_ASSERT(idispsl->get_map().point_same_as(*psource_dof_row_map_),
      "Map of incoming source vector does not match the stored source dof row map.");

  const Core::LinAlg::Map stdmap = idispsl->get_map();
  idispsl->replace_map(*source_dof_row_map_);

  std::shared_ptr<Core::LinAlg::Map> dofrowmap =
      Core::LinAlg::merge_map(*ptarget_dof_row_map_, *psource_dof_row_map_, false);
  Core::LinAlg::Import target_importer(*dofrowmap, *ptarget_dof_row_map_);
  Core::LinAlg::Import source_importer(*dofrowmap, *psource_dof_row_map_);

  // Import target and source displacements into a single vector
  std::shared_ptr<Core::LinAlg::Vector<double>> idisp_target_source =
      std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);
  idisp_target_source->import(*idispma, target_importer, Core::LinAlg::CombineMode::add);
  idisp_target_source->import(*idispsl, source_importer, Core::LinAlg::CombineMode::add);

  // set new displacement state in mortar interface
  interface_->set_state(Mortar::state_new_displacement, *idisp_target_source);

  evaluate();
  matrix_row_col_transform();

  idispsl->replace_map(stdmap);
}


/*----------------------------------------------------------------------*
 *  Create integration cells for mortar interface           farah 01/16 |
 *----------------------------------------------------------------------*/
void Coupling::Adapter::CouplingMortar::evaluate_geometry(
    std::vector<std::shared_ptr<Mortar::IntCell>>& intcells  //!< vector of mortar integration cells
)
{
  // safety check
  check_setup();

  // evaluate geometry information
  interface_->evaluate_geometry(intcells);

  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Coupling::Adapter::CouplingMortar::evaluate()
{
  // safety check
  check_setup();

  // in the following two steps MORTAR does all the work for new interface displacements
  interface_->initialize();
  interface_->evaluate();

  // preparation for AssembleDM
  // (Note that redistsource and redisttarget are the source and target row maps
  // after parallel redistribution. If no redistribution was performed, they
  // are of course identical to source_dof_row_map_/target_dof_row_map_!)
  std::shared_ptr<Core::LinAlg::SparseMatrix> dmatrix =
      std::make_shared<Core::LinAlg::SparseMatrix>(*source_dof_row_map_, 10);
  std::shared_ptr<Core::LinAlg::SparseMatrix> mmatrix =
      std::make_shared<Core::LinAlg::SparseMatrix>(*source_dof_row_map_, 100);
  interface_->assemble_dm(*dmatrix, *mmatrix);

  // Complete() global Mortar matrices
  dmatrix->complete();
  mmatrix->complete(*target_dof_row_map_, *source_dof_row_map_);
  D_ = dmatrix;
  M_ = mmatrix;

  // create projection operator and Dinv
  create_p();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Coupling::Adapter::CouplingMortar::matrix_row_col_transform()
{
  // safety check
  check_setup();

  // check for parallel redistribution
  bool parredist = false;
  const Teuchos::ParameterList& input = mortar_coupling_params_.sublist("PARALLEL REDISTRIBUTION");
  if (Teuchos::getIntegralValue<Mortar::ParallelRedist>(input, "PARALLEL_REDIST") !=
      Mortar::ParallelRedist::redist_none)
    parredist = true;

  // only for parallel redistribution case
  if (parredist)
  {
    if (psource_dof_row_map_ == nullptr or ptarget_dof_row_map_ == nullptr)
      FOUR_C_THROW("Dof maps based on initial parallel distribution are wrong!");

    // transform everything back to old distribution
    D_ = Core::LinAlg::matrix_row_col_transform(*D_, *psource_dof_row_map_, *psource_dof_row_map_);
    M_ = Core::LinAlg::matrix_row_col_transform(*M_, *psource_dof_row_map_, *ptarget_dof_row_map_);
    Dinv_ = Core::LinAlg::matrix_row_col_transform(
        *Dinv_, *psource_dof_row_map_, *psource_dof_row_map_);
    P_ = Core::LinAlg::matrix_row_col_transform(*P_, *psource_dof_row_map_, *ptarget_dof_row_map_);
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Coupling::Adapter::CouplingMortar::evaluate_with_mesh_relocation(
    std::shared_ptr<Core::FE::Discretization> source_dis,
    std::shared_ptr<Core::FE::Discretization> aledis,
    std::shared_ptr<Core::LinAlg::Vector<double>>& idisp, MPI_Comm comm, bool source_is_ale)
{
  // safety check
  check_setup();

  // set new displacement state in mortar interface
  interface_->set_state(Mortar::state_new_displacement, *idisp);

  // in the following two steps MORTAR does all the work for new interface displacements
  interface_->initialize();
  interface_->evaluate();

  // preparation for AssembleDM
  // (Note that redistsource and redisttarget are the source and target row maps
  // after parallel redistribution. If no redistribution was performed, they
  // are of course identical to source_dof_row_map_/target_dof_row_map_!)
  std::shared_ptr<Core::LinAlg::SparseMatrix> dmatrix =
      std::make_shared<Core::LinAlg::SparseMatrix>(*source_dof_row_map_, 10);
  std::shared_ptr<Core::LinAlg::SparseMatrix> mmatrix =
      std::make_shared<Core::LinAlg::SparseMatrix>(*source_dof_row_map_, 100);
  interface_->assemble_dm(*dmatrix, *mmatrix);

  // Complete() global Mortar matrices
  dmatrix->complete();
  mmatrix->complete(*target_dof_row_map_, *source_dof_row_map_);
  D_ = dmatrix;
  M_ = mmatrix;

  // Build Dinv
  Dinv_ = std::make_shared<Core::LinAlg::SparseMatrix>(*D_);

  // extract diagonal of invd into diag
  std::shared_ptr<Core::LinAlg::Vector<double>> diag =
      std::make_shared<Core::LinAlg::Vector<double>>(*source_dof_row_map_, true);
  Dinv_->extract_diagonal_copy(*diag);

  // set zero diagonal values to dummy 1.0
  for (int i = 0; i < diag->local_length(); ++i)
    if ((*diag).local_values_as_span()[i] == 0.0) (*diag).get_values()[i] = 1.0;

  // scalar inversion of diagonal values
  diag->reciprocal(*diag);
  Dinv_->replace_diagonal_values(*diag);
  Dinv_->complete(D_->range_map(), D_->domain_map());
  P_ = Core::LinAlg::matrix_multiply(*Dinv_, false, *M_, false, false, false, true);

  // only for parallel redistribution case
  matrix_row_col_transform();

  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::MultiVector<double>>
Coupling::Adapter::CouplingMortar::target_to_source(
    const Core::LinAlg::MultiVector<double>& mv) const
{
  // safety check
  check_setup();

  FOUR_C_ASSERT(target_dof_row_map_->same_as(mv.get_map()), "Vector with target dof map expected");

  Core::LinAlg::MultiVector<double> tmp =
      Core::LinAlg::MultiVector<double>(M_->row_map(), mv.num_vectors());

  M_->multiply(false, mv, tmp);

  std::shared_ptr<Core::LinAlg::MultiVector<double>> sv =
      std::make_shared<Core::LinAlg::MultiVector<double>>(*psource_dof_row_map_, mv.num_vectors());

  Dinv_->multiply(false, tmp, *sv);

  return sv;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Coupling::Adapter::CouplingMortar::target_to_source(
    const Core::LinAlg::Vector<double>& mv) const
{
  // safety check
  check_setup();

  FOUR_C_ASSERT(target_dof_row_map_->same_as(mv.get_map()), "Vector with target dof map expected");

  Core::LinAlg::Vector<double> tmp = Core::LinAlg::Vector<double>(M_->row_map());

  M_->multiply(false, mv, tmp);

  std::shared_ptr<Core::LinAlg::Vector<double>> sv =
      std::make_shared<Core::LinAlg::Vector<double>>(*psource_dof_row_map_);

  Dinv_->multiply(false, tmp, *sv);

  return sv;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::CouplingMortar::target_to_source(
    const Core::LinAlg::MultiVector<double>& mv, Core::LinAlg::MultiVector<double>& sv) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (not mv.get_map().point_same_as(P_->col_map())) FOUR_C_THROW("target dof map vector expected");
  if (not sv.get_map().point_same_as(D_->col_map())) FOUR_C_THROW("source dof map vector expected");
#endif

  // safety check
  check_setup();

  // source vector with auxiliary dofmap
  Core::LinAlg::MultiVector<double> sv_aux(P_->row_map(), sv.num_vectors());

  // project
  P_->multiply(false, mv, sv_aux);

  // copy from auxiliary to physical map (needed for coupling in fluid ale algorithm)
  std::copy(sv_aux.get_values(),
      sv_aux.get_values() + (sv_aux.local_length() * sv_aux.num_vectors()), sv.get_values());

  // in contrast to the Adapter::Coupling class we do not need to export here, as
  // the mortar interface itself has (or should have) guaranteed the same distribution of target and
  // source dis on all procs
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::CouplingMortar::source_to_target(
    const Core::LinAlg::MultiVector<double>& sv, Core::LinAlg::MultiVector<double>& mv) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (not mv.get_map().point_same_as(P_->col_map())) FOUR_C_THROW("target dof map vector expected");
  if (not sv.get_map().point_same_as(D_->col_map())) FOUR_C_THROW("source dof map vector expected");
#endif

  // safety check
  check_setup();

  Core::LinAlg::Vector<double> tmp = Core::LinAlg::Vector<double>(M_->range_map());
  std::copy(sv.get_values(), sv.get_values() + sv.local_length(), tmp.get_values());

  Core::LinAlg::Vector<double> tempm(*ptarget_dof_row_map_);
  M_->multiply(true, tmp, tempm);

  // copy from auxiliary to physical map (needed for coupling in fluid ale algorithm)
  std::copy(tempm.get_values(), tempm.get_values() + (tempm.local_length() * tempm.num_vectors()),
      mv.get_values());

  // in contrast to the Adapter::Coupling class we do not need to export here, as
  // the mortar interface itself has (or should have) guaranteed the same distribution of target and
  // source dis on all procs
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Coupling::Adapter::CouplingMortar::source_to_target(
    const Core::LinAlg::Vector<double>& sv) const
{
  // safety check
  check_setup();

  Core::LinAlg::Vector<double> tmp = Core::LinAlg::Vector<double>(M_->range_map());
  std::copy(sv.get_values(), sv.get_values() + sv.local_length(), tmp.get_values());

  std::shared_ptr<Core::LinAlg::Vector<double>> mv =
      std::make_shared<Core::LinAlg::Vector<double>>(*ptarget_dof_row_map_);
  M_->multiply(true, tmp, *mv);

  return mv;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::MultiVector<double>>
Coupling::Adapter::CouplingMortar::source_to_target(
    const Core::LinAlg::MultiVector<double>& sv) const
{
  // safety check
  check_setup();

  Core::LinAlg::MultiVector<double> tmp =
      Core::LinAlg::MultiVector<double>(M_->range_map(), sv.num_vectors());
  std::copy(sv.get_values(), sv.get_values() + sv.local_length(), tmp.get_values());

  std::shared_ptr<Core::LinAlg::MultiVector<double>> mv =
      std::make_shared<Core::LinAlg::MultiVector<double>>(*ptarget_dof_row_map_, sv.num_vectors());
  M_->multiply(true, tmp, *mv);

  return mv;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Coupling::Adapter::CouplingMortar::mortar_condensation(
    std::shared_ptr<Core::LinAlg::SparseMatrix>& k, Core::LinAlg::Vector<double>& rhs) const
{
  Mortar::Utils::mortar_matrix_condensation(k, P_, P_);
  Mortar::Utils::mortar_rhs_condensation(rhs, *P_);

  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Coupling::Adapter::CouplingMortar::mortar_recover(
    Core::LinAlg::SparseMatrix& k, Core::LinAlg::Vector<double>& inc) const
{
  Mortar::Utils::mortar_recover(inc, *P_);
  return;
}

FOUR_C_NAMESPACE_CLOSE
