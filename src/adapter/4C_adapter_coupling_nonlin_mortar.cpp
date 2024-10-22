// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_adapter_coupling_nonlin_mortar.hpp"

#include "4C_contact_element.hpp"
#include "4C_contact_friction_node.hpp"
#include "4C_contact_integrator.hpp"
#include "4C_contact_interface.hpp"
#include "4C_coupling_adapter_mortar.hpp"
#include "4C_fem_condition_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_nurbs_discretization.hpp"
#include "4C_fem_nurbs_discretization_control_point.hpp"
#include "4C_fem_nurbs_discretization_knotvector.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_contact.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_mortar_utils.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 |  ctor                                                     farah 10/14|
 *----------------------------------------------------------------------*/
Adapter::CouplingNonLinMortar::CouplingNonLinMortar(int spatial_dimension,
    Teuchos::ParameterList mortar_coupling_params, Teuchos::ParameterList contact_dynamic_params,
    Core::FE::ShapeFunctionType shape_function_type)
    : Coupling::Adapter::CouplingMortar(
          spatial_dimension, mortar_coupling_params, contact_dynamic_params, shape_function_type),
      issetup_(false),
      comm_(Teuchos::null),
      myrank_(-1),
      slavenoderowmap_(Teuchos::null),
      DLin_(Teuchos::null),
      MLin_(Teuchos::null),
      H_(Teuchos::null),
      T_(Teuchos::null),
      N_(Teuchos::null),
      gap_(Teuchos::null),
      interface_(Teuchos::null)
{
  // empty...
}


/*----------------------------------------------------------------------*
 |  initialize nonlinear mortar framework                    farah 10/14|
 *----------------------------------------------------------------------*/
void Adapter::CouplingNonLinMortar::setup(Teuchos::RCP<Core::FE::Discretization> masterdis,
    Teuchos::RCP<Core::FE::Discretization> slavedis, std::vector<int> coupleddof,
    const std::string& couplingcond)
{
  myrank_ = masterdis->get_comm().MyPID();
  comm_ = Teuchos::RCP(masterdis->get_comm().Clone());

  // ParameterList
  Teuchos::ParameterList input;

  // initialize maps for column nodes
  std::map<int, Core::Nodes::Node*> mastergnodes;
  std::map<int, Core::Nodes::Node*> slavegnodes;

  // initialize maps for elements
  std::map<int, Teuchos::RCP<Core::Elements::Element>> masterelements;
  std::map<int, Teuchos::RCP<Core::Elements::Element>> slaveelements;

  Teuchos::RCP<CONTACT::Interface> interface;

  // number of dofs per node based on the coupling vector coupleddof
  const int dof = coupleddof.size();

  // number of coupled dofs (defined in coupleddof by a 1)
  int numcoupleddof = 0;
  for (int ii = 0; ii < dof; ++ii)
    if (coupleddof[ii] == 1) numcoupleddof += 1;

  // read the mortar conditions and set probtype
  read_mortar_condition(masterdis, slavedis, coupleddof, couplingcond, input, mastergnodes,
      slavegnodes, masterelements, slaveelements);

  // add contact nodes to interface discr.
  add_mortar_nodes(masterdis, slavedis, coupleddof, input, mastergnodes, slavegnodes,
      masterelements, slaveelements, interface, numcoupleddof);

  // add contact eles to interface discr.
  add_mortar_elements(
      masterdis, slavedis, input, masterelements, slaveelements, interface, numcoupleddof);

  // complete interface, store as int. var. and do
  // parallel red.
  complete_interface(masterdis, interface);

  // Initialize matrices
  init_matrices();

  // create stratgy object if required
  create_strategy(masterdis, slavedis, input, numcoupleddof);

  // set setup flag
  issetup_ = true;

  return;
}


/*----------------------------------------------------------------------*
 |  read mortar condition                                    farah 10/14|
 *----------------------------------------------------------------------*/
void Adapter::CouplingNonLinMortar::create_strategy(
    Teuchos::RCP<Core::FE::Discretization> masterdis,
    Teuchos::RCP<Core::FE::Discretization> slavedis, Teuchos::ParameterList& input,
    int numcoupleddof)
{
  // nothing to do for pure adapter
  return;
}


/*----------------------------------------------------------------------*
 |  read mortar condition                                    farah 10/14|
 *----------------------------------------------------------------------*/
void Adapter::CouplingNonLinMortar::read_mortar_condition(
    Teuchos::RCP<Core::FE::Discretization> masterdis,
    Teuchos::RCP<Core::FE::Discretization> slavedis, std::vector<int> coupleddof,
    const std::string& couplingcond, Teuchos::ParameterList& input,
    std::map<int, Core::Nodes::Node*>& mastergnodes, std::map<int, Core::Nodes::Node*>& slavegnodes,
    std::map<int, Teuchos::RCP<Core::Elements::Element>>& masterelements,
    std::map<int, Teuchos::RCP<Core::Elements::Element>>& slaveelements)
{
  // TODO: extend this to sliding ale + ALE-dis
  // vector coupleddof defines degree of freedom which are coupled (1: coupled; 0: not coupled),
  // e.g.:
  // - fluid 3D meshtying: coupleddof = [1, 1, 1, 1] -> all degrees of freedom (velocity and
  // pressure) are coupled
  // - fluid 3D meshtying: coupleddof = [1, 1, 1, 0] -> only velocity degrees of freedom are coupled
  // - fsi 3D: coupleddof = [1, 1, 1] -> at the interface only displacements are coupled
  // - ....

  // initialize maps for row nodes
  std::map<int, Core::Nodes::Node*> masternodes;
  std::map<int, Core::Nodes::Node*> slavenodes;

  // Coupling condition is defined by "MORTAR COUPLING CONDITIONS"
  // There is only one discretization (masterdis == slavedis). Therefore, the node set have to be
  // separated beforehand.
  if (couplingcond == "Mortar" || couplingcond == "Contact" || couplingcond == "EHLCoupling")
  {
    std::vector<Core::Conditions::Condition*> conds;
    std::vector<Core::Conditions::Condition*> conds_master(0);
    std::vector<Core::Conditions::Condition*> conds_slave(0);
    masterdis->get_condition(couplingcond, conds);

    for (unsigned i = 0; i < conds.size(); i++)
    {
      const std::string& side = conds[i]->parameters().get<std::string>("Side");

      if (side == "Master") conds_master.push_back(conds[i]);

      if (side == "Slave") conds_slave.push_back(conds[i]);
    }

    // Fill maps based on condition for master side (masterdis == slavedis)
    Core::Conditions::find_condition_objects(
        *masterdis, masternodes, mastergnodes, masterelements, conds_master);

    // Fill maps based on condition for slave side (masterdis == slavedis)
    Core::Conditions::find_condition_objects(
        *slavedis, slavenodes, slavegnodes, slaveelements, conds_slave);
  }
  // Coupling condition is defined by "FSI COUPLING CONDITIONS"
  // There are two discretizations for the master and slave side. Therefore, the master/slave nodes
  // are chosen based on the discretization.
  else
  {
    // Fill maps based on condition for slave side (masterdis != slavedis)
    if (slavedis != Teuchos::null)
      Core::Conditions::find_condition_objects(
          *slavedis, slavenodes, slavegnodes, slaveelements, couplingcond);
  }

  // get mortar coupling parameters
  const Teuchos::ParameterList& inputmortar = Global::Problem::instance()->mortar_coupling_params();
  const Teuchos::ParameterList& meshtying = Global::Problem::instance()->contact_dynamic_params();
  const Teuchos::ParameterList& wearlist = Global::Problem::instance()->wear_params();

  input.setParameters(inputmortar);
  input.setParameters(meshtying);
  input.setParameters(wearlist);

  input.set<int>("PROBTYPE", Inpar::CONTACT::other);  // if other probtypes, this will be
                                                      // overwritten in overloaded function

  // is this a nurbs problem?
  bool isnurbs = false;
  Core::FE::ShapeFunctionType distype = Global::Problem::instance()->spatial_approximation_type();
  if (distype == Core::FE::ShapeFunctionType::nurbs) isnurbs = true;
  input.set<bool>("NURBS", isnurbs);
  input.set<int>("DIMENSION", Global::Problem::instance()->n_dim());

  // check for invalid parameter values
  if (Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(input, "LM_SHAPEFCN") !=
          Inpar::Mortar::shape_dual and
      Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(input, "LM_SHAPEFCN") !=
          Inpar::Mortar::shape_petrovgalerkin)
    if (myrank_ == 0) FOUR_C_THROW("Mortar coupling adapter only works for dual shape functions");

  // as two half pass approach is not implemented for this approach set false
  input.set<bool>("Two_half_pass", false);

  return;
}


/*----------------------------------------------------------------------*
 |  add mortar nodes                                         farah 10/14|
 *----------------------------------------------------------------------*/
void Adapter::CouplingNonLinMortar::add_mortar_nodes(
    Teuchos::RCP<Core::FE::Discretization> masterdis,
    Teuchos::RCP<Core::FE::Discretization> slavedis, std::vector<int> coupleddof,
    Teuchos::ParameterList& input, std::map<int, Core::Nodes::Node*>& mastergnodes,
    std::map<int, Core::Nodes::Node*>& slavegnodes,
    std::map<int, Teuchos::RCP<Core::Elements::Element>>& masterelements,
    std::map<int, Teuchos::RCP<Core::Elements::Element>>& slaveelements,
    Teuchos::RCP<CONTACT::Interface>& interface, int numcoupleddof)
{
  const bool isnurbs = input.get<bool>("NURBS");

  // get problem dimension (2D or 3D) and create (Mortar::Interface)
  const int dim = Global::Problem::instance()->n_dim();

  // create an empty mortar interface
  interface = CONTACT::Interface::create(0, *comm_, dim, input, false);

  //  if((masterdis->NumDof(masterdis->lRowNode(0))!=dof and slavewithale==true and
  //  slidingale==false) or
  //      (slavedis->NumDof(slavedis->lRowNode(0))!=dof and slavewithale==false and
  //      slidingale==false))
  //  {
  //    FOUR_C_THROW("The size of the coupling vector coupleddof and dof defined in the
  //    discretization does not fit!! \n"
  //            "dof defined in the discretization: %i \n"
  //            "length of coupleddof: %i",masterdis->NumDof(masterdis->lRowNode(0)), dof);
  //  }

  // ########## CHECK for a better implementation of this ###################
  // If this option is used, check functionality ... not sure if this is correct!
  // special case: sliding ale
  // In the sliding ale framework two mortar discretizations are generated from identical
  // masterelement and slaveelement sets. Since node-, dof- and element ids of the original
  // elements are the same, an offset have to be defined int nodeoffset=0;
  int dofoffset = 0;
  //  if(slidingale==true)
  //  {
  //    nodeoffset = masterdis->NodeRowMap()->MaxAllGID()+1;
  //    dofoffset = masterdis->dof_row_map()->MaxAllGID()+1;
  //  }
  // ########## CHECK for a better implementation of this ###################

  // feeding master nodes to the interface including ghosted nodes
  std::map<int, Core::Nodes::Node*>::const_iterator nodeiter;
  for (nodeiter = mastergnodes.begin(); nodeiter != mastergnodes.end(); ++nodeiter)
  {
    Core::Nodes::Node* node = nodeiter->second;
    // vector containing only the gids of the coupled dofs (size numcoupleddof)
    std::vector<int> dofids(numcoupleddof);
    int ii = 0;
    for (unsigned int k = 0; k < coupleddof.size(); ++k)
    {
      // Should this dof be coupled? (==1),
      if (coupleddof[k] == 1)
      {
        // get the gid of the coupled dof (size dof)
        // and store it in the vector dofids containing only coupled dofs (size numcoupleddof)
        dofids[ii] = masterdis->dof(0, node)[k];
        ii += 1;
      }
    }
    Teuchos::RCP<CONTACT::Node> cnode = Teuchos::make_rcp<CONTACT::FriNode>(
        node->id(), node->x(), node->owner(), dofids, false, false, false);

    if (isnurbs)
    {
      Core::FE::Nurbs::ControlPoint* cp = dynamic_cast<Core::FE::Nurbs::ControlPoint*>(node);

      cnode->nurbs_w() = cp->w();
    }

    interface->add_node(cnode);
  }

  // feeding slave nodes to the interface including ghosted nodes
  for (nodeiter = slavegnodes.begin(); nodeiter != slavegnodes.end(); ++nodeiter)
  {
    Core::Nodes::Node* node = nodeiter->second;
    // vector containing only the gids of the coupled dofs (size numcoupleddof)
    std::vector<int> dofids(numcoupleddof);
    int ii = 0;
    for (unsigned int k = 0; k < coupleddof.size(); ++k)
    {
      // Should this dof be coupled? (==1)
      if (coupleddof[k] == 1)
      {
        // get the gid of the coupled dof (size dof)
        // and store it in the vector dofids containing only coupled dofs (size numcoupleddof)
        dofids[ii] = slavedis->dof(0, node)[k] + dofoffset;
        ii += 1;
      }
    }
    Teuchos::RCP<CONTACT::Node> cnode = Teuchos::make_rcp<CONTACT::FriNode>(
        node->id(), node->x(), node->owner(), dofids, true, true, false);

    if (isnurbs)
    {
      Core::FE::Nurbs::ControlPoint* cp = dynamic_cast<Core::FE::Nurbs::ControlPoint*>(node);

      cnode->nurbs_w() = cp->w();
    }

    interface->add_node(cnode);
  }
}


/*----------------------------------------------------------------------*
 |  add mortar elements                                      farah 10/14|
 *----------------------------------------------------------------------*/
void Adapter::CouplingNonLinMortar::add_mortar_elements(
    Teuchos::RCP<Core::FE::Discretization> masterdis,
    Teuchos::RCP<Core::FE::Discretization> slavedis, Teuchos::ParameterList& input,
    std::map<int, Teuchos::RCP<Core::Elements::Element>>& masterelements,
    std::map<int, Teuchos::RCP<Core::Elements::Element>>& slaveelements,
    Teuchos::RCP<CONTACT::Interface>& interface, int numcoupleddof)
{
  const bool isnurbs = input.get<bool>("NURBS");

  // get problem dimension (2D or 3D) and create (Mortar::Interface)
  const int dim = Global::Problem::instance()->n_dim();

  // ########## CHECK for a better implementation of this ###################
  // If this option is used, check functionality ... not sure if this is correct!
  // special case: sliding ale
  // In the sliding ale framework two mortar discretizations are generated from identical
  // masterelement and slaveelement sets. Since node-, dof- and element ids of the original
  // elements are the same, an offset have to be defined
  int nodeoffset = 0;
  // int dofoffset=0;
  //  if(slidingale==true)
  //  {
  //    nodeoffset = masterdis->NodeRowMap()->MaxAllGID()+1;
  //    dofoffset = masterdis->dof_row_map()->MaxAllGID()+1;
  //  }
  // ########## CHECK for a better implementation of this ###################


  // We need to determine an element offset to start the numbering of the slave
  // mortar elements AFTER the master mortar elements in order to ensure unique
  // eleIDs in the interface discretization. The element offset equals the
  // overall number of master mortar elements (which is not equal to the number
  // of elements in the field that is chosen as master side).
  //
  // If masterdis==slavedis, the element numbering is right without offset
  int eleoffset = 0;
  if (masterdis.get() != slavedis.get())
  {
    int nummastermtreles = masterelements.size();
    comm_->SumAll(&nummastermtreles, &eleoffset, 1);
  }

  //  if(slidingale==true)
  //    eleoffset = masterdis->ElementRowMap()->MaxAllGID()+1;

  // feeding master elements to the interface
  std::map<int, Teuchos::RCP<Core::Elements::Element>>::const_iterator elemiter;
  for (elemiter = masterelements.begin(); elemiter != masterelements.end(); ++elemiter)
  {
    Teuchos::RCP<Core::Elements::Element> ele = elemiter->second;
    Teuchos::RCP<CONTACT::Element> cele = Teuchos::make_rcp<CONTACT::Element>(
        ele->id(), ele->owner(), ele->shape(), ele->num_node(), ele->node_ids(), false, isnurbs);

    if (isnurbs)
    {
      Teuchos::RCP<Core::FE::Nurbs::NurbsDiscretization> nurbsdis =
          Teuchos::rcp_dynamic_cast<Core::FE::Nurbs::NurbsDiscretization>(masterdis);

      Teuchos::RCP<Core::FE::Nurbs::Knotvector> knots = (*nurbsdis).get_knot_vector();
      std::vector<Core::LinAlg::SerialDenseVector> parentknots(dim);
      std::vector<Core::LinAlg::SerialDenseVector> mortarknots(dim - 1);

      Teuchos::RCP<Core::Elements::FaceElement> faceele =
          Teuchos::rcp_dynamic_cast<Core::Elements::FaceElement>(ele, true);
      double normalfac = 0.0;
      bool zero_size = knots->get_boundary_ele_and_parent_knots(parentknots, mortarknots, normalfac,
          faceele->parent_master_element()->id(), faceele->face_master_number());

      // store nurbs specific data to node
      cele->zero_sized() = zero_size;
      cele->knots() = mortarknots;
      cele->normal_fac() = normalfac;
    }

    interface->add_element(cele);
  }

  // feeding slave elements to the interface
  for (elemiter = slaveelements.begin(); elemiter != slaveelements.end(); ++elemiter)
  {
    Teuchos::RCP<Core::Elements::Element> ele = elemiter->second;

    // Here, we have to distinguish between standard and sliding ale since mortar elements are
    // generated from the identical element sets in the case of sliding ale Therefore, we introduce
    // an element offset AND a node offset for the the slave mortar elements
    if (true)  //(slidingale==false)
    {
      Teuchos::RCP<CONTACT::Element> cele = Teuchos::make_rcp<CONTACT::Element>(
          ele->id(), ele->owner(), ele->shape(), ele->num_node(), ele->node_ids(), true, isnurbs);

      if (isnurbs)
      {
        Teuchos::RCP<Core::FE::Nurbs::NurbsDiscretization> nurbsdis =
            Teuchos::rcp_dynamic_cast<Core::FE::Nurbs::NurbsDiscretization>(slavedis);

        Teuchos::RCP<Core::FE::Nurbs::Knotvector> knots = (*nurbsdis).get_knot_vector();
        std::vector<Core::LinAlg::SerialDenseVector> parentknots(dim);
        std::vector<Core::LinAlg::SerialDenseVector> mortarknots(dim - 1);

        Teuchos::RCP<Core::Elements::FaceElement> faceele =
            Teuchos::rcp_dynamic_cast<Core::Elements::FaceElement>(ele, true);
        double normalfac = 0.0;
        bool zero_size = knots->get_boundary_ele_and_parent_knots(parentknots, mortarknots,
            normalfac, faceele->parent_master_element()->id(), faceele->face_master_number());

        // store nurbs specific data to node
        cele->zero_sized() = zero_size;
        cele->knots() = mortarknots;
        cele->normal_fac() = normalfac;
      }

      interface->add_element(cele);
    }
    else
    {
      std::vector<int> nidsoff;
      for (int i = 0; i < ele->num_node(); i++)
      {
        nidsoff.push_back(ele->node_ids()[ele->num_node() - 1 - i] + nodeoffset);
      }

      Teuchos::RCP<CONTACT::Element> cele = Teuchos::make_rcp<CONTACT::Element>(
          ele->id() + eleoffset, ele->owner(), ele->shape(), ele->num_node(), nidsoff.data(), true);

      interface->add_element(cele);
    }
  }

  return;
}


/*----------------------------------------------------------------------*
 |  Initialize matrices                                      farah 02/16|
 *----------------------------------------------------------------------*/
void Adapter::CouplingNonLinMortar::init_matrices()
{
  // safety check
  if (slavedofrowmap_ == Teuchos::null or slavenoderowmap_ == Teuchos::null)
    FOUR_C_THROW("ERROR: Maps not initialized!");

  // init as standard sparse matrix --> local assembly
  D_ = Teuchos::make_rcp<Core::LinAlg::SparseMatrix>(*slavedofrowmap_, 81, false, false);
  M_ = Teuchos::make_rcp<Core::LinAlg::SparseMatrix>(*slavedofrowmap_, 81, false, false);
  H_ = Teuchos::make_rcp<Core::LinAlg::SparseMatrix>(*slavedofrowmap_, 81, false, false);
  T_ = Teuchos::make_rcp<Core::LinAlg::SparseMatrix>(*slavedofrowmap_, 81, false, false);
  N_ = Teuchos::make_rcp<Core::LinAlg::SparseMatrix>(*slavedofrowmap_, 81, false, false);

  gap_ = Teuchos::make_rcp<Core::LinAlg::Vector<double>>(*slavenoderowmap_, true);

  // init as fe matrix --> nonlocal assembly
  DLin_ = Teuchos::make_rcp<Core::LinAlg::SparseMatrix>(
      *slavedofrowmap_, 81, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  MLin_ = Teuchos::make_rcp<Core::LinAlg::SparseMatrix>(
      *masterdofrowmap_, 81, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);

  // bye
  return;
}


/*----------------------------------------------------------------------*
 |  complete interface (also print and parallel redist.)     farah 02/16|
 *----------------------------------------------------------------------*/
void Adapter::CouplingNonLinMortar::complete_interface(
    Teuchos::RCP<Core::FE::Discretization> masterdis, Teuchos::RCP<CONTACT::Interface>& interface)
{
  const Teuchos::ParameterList& input =
      Global::Problem::instance()->mortar_coupling_params().sublist("PARALLEL REDISTRIBUTION");
  const Inpar::Mortar::ParallelRedist parallelRedist =
      Teuchos::getIntegralValue<Inpar::Mortar::ParallelRedist>(input, "PARALLEL_REDIST");

  /* Finalize the interface construction
   *
   * If this is the final parallel distribution, we need to assign degrees of freedom during
   * during fill_complete(). If parallel redistribution is enabled, there will be another call to
   * fill_complete(), so we skip this expensive operation here and do it later. DOFs have to be
   * assigned only once!
   */
  {
    bool isFinalDistribution = false;
    if (parallelRedist == Inpar::Mortar::ParallelRedist::redist_none || comm_->NumProc() == 1)
      isFinalDistribution = true;
    interface->fill_complete(Global::Problem::instance()->discretization_map(),
        Global::Problem::instance()->binning_strategy_params(),
        Global::Problem::instance()->output_control_file(),
        Global::Problem::instance()->spatial_approximation_type(), isFinalDistribution);
  }

  // create binary search tree
  interface->create_search_tree();

  // store old row maps (before parallel redistribution)
  pslavedofrowmap_ = Teuchos::make_rcp<Epetra_Map>(*interface->slave_row_dofs());
  pmasterdofrowmap_ = Teuchos::make_rcp<Epetra_Map>(*interface->master_row_dofs());
  pslavenoderowmap_ = Teuchos::make_rcp<Epetra_Map>(*interface->slave_row_nodes());
  psmdofrowmap_ = Core::LinAlg::merge_map(pslavedofrowmap_, pmasterdofrowmap_, false);

  // print parallel distribution
  interface->print_parallel_distribution();

  //**********************************************************************
  // PARALLEL REDISTRIBUTION OF INTERFACE
  //**********************************************************************
  if (parallelRedist != Inpar::Mortar::ParallelRedist::redist_none && comm_->NumProc() > 1)
  {
    // redistribute optimally among all procs
    interface->redistribute();

    // call fill complete again
    interface->fill_complete(Global::Problem::instance()->discretization_map(),
        Global::Problem::instance()->binning_strategy_params(),
        Global::Problem::instance()->output_control_file(),
        Global::Problem::instance()->spatial_approximation_type(), true);

    // re create binary search tree
    interface->create_search_tree();

    // print parallel distribution again
    interface->print_parallel_distribution();
  }

  // store row maps (after parallel redistribution)
  slavedofrowmap_ = Teuchos::make_rcp<Epetra_Map>(*interface->slave_row_dofs());
  masterdofrowmap_ = Teuchos::make_rcp<Epetra_Map>(*interface->master_row_dofs());
  slavenoderowmap_ = Teuchos::make_rcp<Epetra_Map>(*interface->slave_row_nodes());
  smdofrowmap_ = Core::LinAlg::merge_map(slavedofrowmap_, masterdofrowmap_, false);

  // store interface
  interface_ = interface;

  return;
}


/*----------------------------------------------------------------------*
 | setup contact elements for spring dashpot condition     pfaller Apr15|
 *----------------------------------------------------------------------*/
void Adapter::CouplingNonLinMortar::setup_spring_dashpot(
    Teuchos::RCP<Core::FE::Discretization> masterdis,
    Teuchos::RCP<Core::FE::Discretization> slavedis,
    Teuchos::RCP<Core::Conditions::Condition> spring, const int coupling_id,
    const Epetra_Comm& comm)
{
  if (comm.MyPID() == 0)
    std::cout << "Generating CONTACT interface for spring dashpot condition...\n" << std::endl;

  // initialize maps for row nodes
  std::map<int, Core::Nodes::Node*> slavenodes;
  std::map<int, Core::Nodes::Node*> masternodes;

  // initialize maps for column nodes
  std::map<int, Core::Nodes::Node*> slavegnodes;
  std::map<int, Core::Nodes::Node*> mastergnodes;

  // initialize maps for elements
  std::map<int, Teuchos::RCP<Core::Elements::Element>> slaveelements;
  std::map<int, Teuchos::RCP<Core::Elements::Element>> masterelements;

  // get the conditions for the current evaluation we use the SpringDashpot condition as a
  // substitute for the mortar slave surface
  std::vector<Core::Conditions::Condition*> conds_master(0);
  std::vector<Core::Conditions::Condition*> conds_slave(0);

  // Coupling condition is defined by "DESIGN SURF SPRING DASHPOT COUPLING CONDITIONS"
  std::vector<Core::Conditions::Condition*> coup_conds;
  slavedis->get_condition("RobinSpringDashpotCoupling", coup_conds);

  // number of coupling conditions
  const int n_coup_conds = (int)coup_conds.size();
  if (!n_coup_conds)
    FOUR_C_THROW("No section DESIGN SURF ROBIN SPRING DASHPOT COUPLING CONDITIONS found.");

  // slave surface = spring dashpot condition
  conds_slave.push_back(&(*spring));

  // find master surface: loop all coupling conditions
  for (int i = 0; i < n_coup_conds; i++)
  {
    // add one, since read in of COUPLING parameter in DESIGN SURF SPRING DASHPOT CONDITIONS
    // subtracts one
    if (coup_conds[i]->parameters().get<int>("COUPLING") == (coupling_id + 1))
      conds_master.push_back(coup_conds[i]);
  }
  if (!conds_master.size()) FOUR_C_THROW("Coupling ID not found.");

  Core::Conditions::find_condition_objects(
      *slavedis, slavenodes, slavegnodes, slaveelements, conds_slave);
  Core::Conditions::find_condition_objects(
      *masterdis, masternodes, mastergnodes, masterelements, conds_master);

  // get mortar coupling parameters
  Teuchos::ParameterList input;
  // set default values
  input.setParameters(Global::Problem::instance()->mortar_coupling_params());
  input.setParameters(Global::Problem::instance()->contact_dynamic_params());
  input.setParameters(Global::Problem::instance()->wear_params());
  input.set<int>("PROBTYPE", Inpar::CONTACT::other);

  // is this a nurbs problem?
  Core::FE::ShapeFunctionType distype = Global::Problem::instance()->spatial_approximation_type();
  switch (distype)
  {
    case Core::FE::ShapeFunctionType::nurbs:
    {
      // ***
      FOUR_C_THROW("nurbs for fsi mortar not supported!");
      input.set<bool>("NURBS", true);
      break;
    }
    default:
    {
      input.set<bool>("NURBS", false);
      break;
    }
  }

  // as two half pass approach is not implemented for this approach set false
  input.set<bool>("Two_half_pass", false);

  // get problem dimension (2D or 3D) and create (Mortar::Interface)
  const int dim = Global::Problem::instance()->n_dim();

  // generate contact interface
  Teuchos::RCP<CONTACT::Interface> interface =
      CONTACT::Interface::create(0, comm, dim, input, false);

  // feeding nodes to the interface including ghosted nodes
  std::map<int, Core::Nodes::Node*>::const_iterator nodeiter;

  // feeding elements to the interface
  std::map<int, Teuchos::RCP<Core::Elements::Element>>::const_iterator elemiter;

  // eleoffset is neccessary because slave and master elements are from different conditions
  const int eleoffset = masterdis->element_row_map()->MaxAllGID() + 1;

  // MASTER NODES
  // feeding master nodes to the interface including ghosted nodes
  for (nodeiter = mastergnodes.begin(); nodeiter != mastergnodes.end(); ++nodeiter)
  {
    Core::Nodes::Node* node = nodeiter->second;

    Teuchos::RCP<CONTACT::Node> mrtrnode = Teuchos::make_rcp<CONTACT::FriNode>(
        node->id(), node->x(), node->owner(), masterdis->dof(node), false, false, false);

    interface->add_node(mrtrnode);
  }

  // SLAVE NODES
  // feeding slave nodes to the interface including ghosted nodes
  for (nodeiter = slavegnodes.begin(); nodeiter != slavegnodes.end(); ++nodeiter)
  {
    Core::Nodes::Node* node = nodeiter->second;

    Teuchos::RCP<CONTACT::Node> mrtrnode = Teuchos::make_rcp<CONTACT::FriNode>(
        node->id(), node->x(), node->owner(), slavedis->dof(node), true, true, false);

    interface->add_node(mrtrnode);
  }

  // MASTER ELEMENTS
  // feeding master elements to the interface
  for (elemiter = masterelements.begin(); elemiter != masterelements.end(); ++elemiter)
  {
    Teuchos::RCP<Core::Elements::Element> ele = elemiter->second;

    Teuchos::RCP<CONTACT::Element> mrtrele = Teuchos::make_rcp<CONTACT::Element>(
        ele->id(), ele->owner(), ele->shape(), ele->num_node(), ele->node_ids(), false);

    interface->add_element(mrtrele);
  }

  // SLAVE ELEMENTS
  // feeding slave elements to the interface
  for (elemiter = slaveelements.begin(); elemiter != slaveelements.end(); ++elemiter)
  {
    Teuchos::RCP<Core::Elements::Element> ele = elemiter->second;

    Teuchos::RCP<CONTACT::Element> mrtrele = Teuchos::make_rcp<CONTACT::Element>(
        ele->id() + eleoffset, ele->owner(), ele->shape(), ele->num_node(), ele->node_ids(), true);

    interface->add_element(mrtrele);
  }

  /* Finalize the interface construction
   *
   * If this is the final parallel distribution, we need to assign degrees of freedom during
   * during fill_complete(). If parallel redistribution is enabled, there will be another call to
   * fill_complete(), so we skip this expensive operation here and do it later. DOFs have to be
   * assigned only once!
   */
  {
    bool isFinalDistribution = false;
    const Teuchos::ParameterList& input =
        Global::Problem::instance()->mortar_coupling_params().sublist("PARALLEL REDISTRIBUTION");
    if (Teuchos::getIntegralValue<Inpar::Mortar::ParallelRedist>(input, "PARALLEL_REDIST") ==
            Inpar::Mortar::ParallelRedist::redist_none or
        comm_->NumProc() == 1)
      isFinalDistribution = true;

    interface->fill_complete(Global::Problem::instance()->discretization_map(),
        Global::Problem::instance()->binning_strategy_params(),
        Global::Problem::instance()->output_control_file(),
        Global::Problem::instance()->spatial_approximation_type(), isFinalDistribution);
  }

  // store old row maps (before parallel redistribution)
  slavedofrowmap_ = Teuchos::make_rcp<Epetra_Map>(*interface->slave_row_dofs());
  masterdofrowmap_ = Teuchos::make_rcp<Epetra_Map>(*interface->master_row_dofs());

  // store interface
  interface_ = interface;

  // create binary search tree
  interface_->create_search_tree();

  // interface displacement (=0) has to be merged from slave and master discretization
  Teuchos::RCP<Epetra_Map> dofrowmap =
      Core::LinAlg::merge_map(masterdofrowmap_, slavedofrowmap_, false);
  Teuchos::RCP<Core::LinAlg::Vector<double>> dispn = Core::LinAlg::create_vector(*dofrowmap, true);

  // set displacement state in mortar interface
  interface_->set_state(Mortar::state_new_displacement, *dispn);

  // in the following two steps MORTAR does all the work
  interface_->initialize();

  // set setup flag
  issetup_ = true;

  return;
}


/*----------------------------------------------------------------------*
 |  print interface                                         farah 10/14|
 *----------------------------------------------------------------------*/
void Adapter::CouplingNonLinMortar::print_interface(std::ostream& os) { interface_->print(os); }


/*----------------------------------------------------------------------*
 |  Integrate slave-side matrix + linearization (D matrix)   farah 10/14|
 *----------------------------------------------------------------------*/
void Adapter::CouplingNonLinMortar::integrate_lin_d(const std::string& statename,
    const Teuchos::RCP<Core::LinAlg::Vector<double>> vec,
    const Teuchos::RCP<Core::LinAlg::Vector<double>> veclm)
{
  // safety check
  check_setup();

  // init matrices
  init_matrices();

  // set lagrange multiplier and displacement state
  interface_->set_state(Mortar::string_to_state_type(statename), *vec);
  interface_->set_state(Mortar::state_lagrange_multiplier, *veclm);

  // general interface init: data container etc...
  interface_->initialize();
  interface_->set_element_areas();

  // loop over all slave col elements and direct integration
  for (int j = 0; j < interface_->slave_col_elements()->NumMyElements(); ++j)
  {
    int gid = interface_->slave_col_elements()->GID(j);
    Core::Elements::Element* ele = interface_->discret().g_element(gid);
    if (!ele) FOUR_C_THROW("ERROR: Cannot find ele with gid %", gid);
    CONTACT::Element* cele = dynamic_cast<CONTACT::Element*>(ele);

    CONTACT::Integrator integrator(interface_->interface_params(), cele->shape(), *comm_);

    integrator.integrate_d(*cele, *comm_, true);
  }

  // assemble routine
  interface_->assemble_d(*D_);
  interface_->assemble_lin_d(*DLin_, false);

  // complete matrices
  D_->complete();
  DLin_->complete();

  // check for parallel redistribution
  bool parredist = false;
  const Teuchos::ParameterList& input =
      Global::Problem::instance()->mortar_coupling_params().sublist("PARALLEL REDISTRIBUTION");
  if (Teuchos::getIntegralValue<Inpar::Mortar::ParallelRedist>(input, "PARALLEL_REDIST") !=
      Inpar::Mortar::ParallelRedist::redist_none)
    parredist = true;

  // only for parallel redistribution case
  if (parredist)
  {
    if (pslavedofrowmap_ == Teuchos::null)
      FOUR_C_THROW("ERROR: Dof maps based on initial parallel distribution are wrong!");

    // transform everything back to old distribution
    D_ = Mortar::matrix_row_col_transform(*D_, *pslavedofrowmap_, *pslavedofrowmap_);
    DLin_ = Mortar::matrix_row_col_transform(*DLin_, *pslavedofrowmap_, *pslavedofrowmap_);
  }

  return;
}


/*----------------------------------------------------------------------*
 |  Integrate mortar matrices + linearization (D/M matrix)   farah 01/16|
 *----------------------------------------------------------------------*/
void Adapter::CouplingNonLinMortar::integrate_lin_dm(const std::string& statename,
    const Teuchos::RCP<Core::LinAlg::Vector<double>> vec,
    const Teuchos::RCP<Core::LinAlg::Vector<double>> veclm)
{
  // safety check
  check_setup();

  // init matrices with redistributed maps
  init_matrices();

  // set current lm and displ state
  interface_->set_state(Mortar::string_to_state_type(statename), *vec);
  interface_->set_state(Mortar::state_lagrange_multiplier, *veclm);

  // init internal data
  interface_->initialize();
  interface_->set_element_areas();

  // call interface evaluate (d,m,gap...)
  interface_->evaluate();

  // assemble mortar matrices and lin.
  interface_->assemble_dm(*D_, *M_);
  interface_->assemble_lin_dm(*DLin_, *MLin_);

  // complete
  D_->complete();
  M_->complete(*masterdofrowmap_, *slavedofrowmap_);
  DLin_->complete(*smdofrowmap_, *slavedofrowmap_);
  MLin_->complete(*smdofrowmap_, *masterdofrowmap_);

  // Dinv * M
  create_p();

  // transform to initial parallel distrib.
  matrix_row_col_transform();

  // bye bye
  return;
}


/*----------------------------------------------------------------------*
 |  transform all matrices and vectors                       farah 02/16|
 *----------------------------------------------------------------------*/
void Adapter::CouplingNonLinMortar::matrix_row_col_transform()
{
  // call base function
  Coupling::Adapter::CouplingMortar::matrix_row_col_transform();

  // safety check
  check_setup();

  // check for parallel redistribution
  bool parredist = false;
  const Teuchos::ParameterList& input =
      Global::Problem::instance()->mortar_coupling_params().sublist("PARALLEL REDISTRIBUTION");
  if (Teuchos::getIntegralValue<Inpar::Mortar::ParallelRedist>(input, "PARALLEL_REDIST") !=
      Inpar::Mortar::ParallelRedist::redist_none)
    parredist = true;

  // transform everything back to old distribution
  if (parredist)
  {
    if (pslavedofrowmap_ == Teuchos::null or pmasterdofrowmap_ == Teuchos::null or
        pslavenoderowmap_ == Teuchos::null or psmdofrowmap_ == Teuchos::null)
      FOUR_C_THROW("ERROR: Dof maps based on initial parallel distribution are wrong!");

    if (DLin_ != Teuchos::null)
      DLin_ = Mortar::matrix_row_col_transform(*DLin_, *pslavedofrowmap_, *psmdofrowmap_);

    if (MLin_ != Teuchos::null)
      MLin_ = Mortar::matrix_row_col_transform(*MLin_, *pmasterdofrowmap_, *psmdofrowmap_);

    if (H_ != Teuchos::null)
      H_ = Mortar::matrix_row_col_transform(*H_, *pslavedofrowmap_, *pslavedofrowmap_);

    if (T_ != Teuchos::null)
      T_ = Mortar::matrix_row_col_transform(*T_, *pslavedofrowmap_, *pslavedofrowmap_);

    if (N_ != Teuchos::null)
      N_ = Mortar::matrix_row_col_transform(*N_, *pslavedofrowmap_, *psmdofrowmap_);

    // transform gap vector
    if (gap_ != Teuchos::null)
    {
      Teuchos::RCP<Core::LinAlg::Vector<double>> pgap =
          Core::LinAlg::create_vector(*pslavenoderowmap_, true);
      Core::LinAlg::export_to(*gap_, *pgap);
      gap_ = pgap;
    }
  }  // end parredist

  return;
}


/*----------------------------------------------------------------------*
 |  Integrate mortar matrices + linearization (D/M matrix) + gap vector |
 |  + compute projection operator P                         wirtz 01/16 |
 *----------------------------------------------------------------------*/
void Adapter::CouplingNonLinMortar::integrate_all(const std::string& statename,
    const Teuchos::RCP<Core::LinAlg::Vector<double>> vec,
    const Teuchos::RCP<Core::LinAlg::Vector<double>> veclm)
{
  // safety check
  check_setup();

  // init matrices with redistributed maps
  init_matrices();

  // set current lm and displ state
  interface_->set_state(Mortar::string_to_state_type(statename), *vec);
  interface_->set_state(Mortar::state_lagrange_multiplier, *veclm);

  // init internal data
  interface_->initialize();
  interface_->set_element_areas();

  // call interface evaluate (d,m,gap...)
  interface_->evaluate();

  // assemble mortar matrices and lin.
  interface_->assemble_dm(*D_, *M_);
  interface_->assemble_lin_dm(*DLin_, *MLin_);
  interface_->assemble_g(*gap_);

  // complete
  D_->complete();
  M_->complete(*masterdofrowmap_, *slavedofrowmap_);
  DLin_->complete(*smdofrowmap_, *slavedofrowmap_);
  MLin_->complete(*smdofrowmap_, *masterdofrowmap_);

  // Dinv * M
  create_p();

  // transform to initial parallel distrib.
  matrix_row_col_transform();

  return;
}

/*----------------------------------------------------------------------*
 |  Evaluate all mortar matrices and vectors necessary for mesh sliding |
 |                                                          wirtz 02/16 |
 *----------------------------------------------------------------------*/
void Adapter::CouplingNonLinMortar::evaluate_sliding(const std::string& statename,
    const Teuchos::RCP<Core::LinAlg::Vector<double>> vec,
    const Teuchos::RCP<Core::LinAlg::Vector<double>> veclm)
{
  // safety check
  check_setup();

  // init matrices with redistributed maps
  init_matrices();

  // set current lm and displ state
  interface_->set_state(Mortar::string_to_state_type(statename), *vec);
  interface_->set_state(Mortar::state_lagrange_multiplier, *veclm);

  // init internal data
  interface_->initialize();
  interface_->set_element_areas();

  interface_->build_active_set(true);

  // call interface evaluate (d,m,gap...)
  interface_->evaluate();

  // assemble mortar matrices and lin.
  interface_->assemble_dm(*D_, *M_);
  interface_->assemble_lin_dm(*DLin_, *MLin_);
  interface_->assemble_t_nderiv(H_, Teuchos::null);
  interface_->assemble_tn(T_, Teuchos::null);
  interface_->assemble_s(*N_);
  interface_->assemble_g(*gap_);

  // complete
  D_->complete();
  M_->complete(*masterdofrowmap_, *slavedofrowmap_);
  DLin_->complete(*smdofrowmap_, *slavedofrowmap_);
  MLin_->complete(*smdofrowmap_, *masterdofrowmap_);
  H_->complete();
  T_->complete();
  N_->complete(*smdofrowmap_, *slavedofrowmap_);

  // Dinv * M
  create_p();

  // transform to initial parallel distrib.
  matrix_row_col_transform();

  return;
}

/*----------------------------------------------------------------------*
 |  compute projection operator P                            wirtz 02/16|
 *----------------------------------------------------------------------*/
void Adapter::CouplingNonLinMortar::create_p()
{
  // safety check
  check_setup();

  // check
  if (Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(
          interface()->interface_params(), "LM_SHAPEFCN") != Inpar::Mortar::shape_dual)
    FOUR_C_THROW("ERROR: Creation of P operator only for dual shape functions!");

  /********************************************************************/
  /* Multiply Mortar matrices: P = inv(D) * M         A               */
  /********************************************************************/
  D_->complete();
  Dinv_ = Teuchos::make_rcp<Core::LinAlg::SparseMatrix>(*D_);
  Teuchos::RCP<Core::LinAlg::Vector<double>> diag =
      Core::LinAlg::create_vector(*slavedofrowmap_, true);
  int err = 0;

  // extract diagonal of invd into diag
  Dinv_->extract_diagonal_copy(*diag);

  // set zero diagonal values to dummy 1.0
  for (int i = 0; i < diag->MyLength(); ++i)
  {
    if (abs((*diag)[i]) < 1e-12)
    {
      std::cout << "WARNING: Diagonal entry of D matrix is skipped because it is less than 1e-12!!!"
                << std::endl;
      (*diag)[i] = 1.0;
    }
  }

  // scalar inversion of diagonal values
  err = diag->Reciprocal(*diag);
  if (err > 0) FOUR_C_THROW("ERROR: Reciprocal: Zero diagonal entry!");

  // re-insert inverted diagonal into invd
  err = Dinv_->replace_diagonal_values(*diag);
  if (err > 0) FOUR_C_THROW("ERROR: replace_diagonal_values failed!");

  // complete inverse D matrix
  Dinv_->complete();

  // do the multiplication P = inv(D) * M
  P_ = Core::LinAlg::matrix_multiply(*Dinv_, false, *M_, false, false, false, true);

  // complete the matrix
  P_->complete(*masterdofrowmap_, *slavedofrowmap_);

  // bye
  return;
}

FOUR_C_NAMESPACE_CLOSE
