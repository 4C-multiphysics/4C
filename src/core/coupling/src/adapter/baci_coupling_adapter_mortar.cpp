/*----------------------------------------------------------------------*/
/*! \file

\brief A class providing coupling capabilities based on mortar methods

\level 2


*/
/*----------------------------------------------------------------------*/
#include "baci_coupling_adapter_mortar.H"

#include "baci_inpar_fluid.H"
#include "baci_io.H"
#include "baci_lib_condition_utils.H"
#include "baci_lib_discret.H"
#include "baci_linalg_multiply.H"
#include "baci_linalg_utils_densematrix_communication.H"
#include "baci_linalg_utils_sparse_algebra_assemble.H"
#include "baci_linalg_utils_sparse_algebra_create.H"
#include "baci_linalg_utils_sparse_algebra_manipulation.H"
#include "baci_mortar_coupling3d_classes.H"
#include "baci_mortar_element.H"
#include "baci_mortar_interface.H"
#include "baci_mortar_node.H"
#include "baci_mortar_utils.H"

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
CORE::ADAPTER::CouplingMortar::CouplingMortar(int spatial_dimension,
    Teuchos::ParameterList mortar_coupling_params, Teuchos::ParameterList contact_dynamic_params,
    CORE::FE::ShapeFunctionType shape_function_type)
    : spatial_dimension_(spatial_dimension),
      mortar_coupling_params_(mortar_coupling_params),
      contact_dynamic_params_(contact_dynamic_params),
      shape_function_type_(shape_function_type),
      issetup_(false)
{
  ;
}


/*----------------------------------------------------------------------*
 | setup routine for mortar framework                        fang 01/16 |
 *----------------------------------------------------------------------*/
void CORE::ADAPTER::CouplingMortar::Setup(
    const Teuchos::RCP<::DRT::Discretization>& masterdis,  ///< master discretization
    const Teuchos::RCP<::DRT::Discretization>& slavedis,   ///< slave discretization
    const Teuchos::RCP<::DRT::Discretization>& aledis,     ///< ALE discretization
    const std::vector<int>& coupleddof,  ///< vector defining coupled degrees of freedom
    const std::string& couplingcond,     ///< string for coupling condition
    const Epetra_Comm& comm,             ///< communicator
    const bool slavewithale,             ///< flag defining if slave is ALE
    const bool slidingale,               ///< flag indicating sliding ALE case
    const int nds_master,                ///< master dofset number
    const int nds_slave                  ///< slave dofset number
)
{
  // initialize maps for row nodes
  std::map<int, ::DRT::Node*> masternodes;
  std::map<int, ::DRT::Node*> slavenodes;

  // initialize maps for column nodes
  std::map<int, ::DRT::Node*> mastergnodes;
  std::map<int, ::DRT::Node*> slavegnodes;

  // initialize maps for elements
  std::map<int, Teuchos::RCP<::DRT::Element>> masterelements;
  std::map<int, Teuchos::RCP<::DRT::Element>> slaveelements;

  // Coupling condition is defined by "MORTAR COUPLING CONDITIONS"
  // There is only one discretization (masterdis == slavedis). Therefore, the node set have to be
  // separated beforehand.
  if (couplingcond == "Mortar" || couplingcond == "MortarMulti")
  {
    std::vector<::DRT::Condition*> conds;
    std::vector<::DRT::Condition*> conds_master(0);
    std::vector<::DRT::Condition*> conds_slave(0);
    masterdis->GetCondition(couplingcond, conds);

    for (unsigned i = 0; i < conds.size(); i++)
    {
      const std::string* side = conds[i]->Get<std::string>("Side");

      if (*side == "Master")
        conds_master.push_back(conds[i]);
      else if (*side == "Slave")
        conds_slave.push_back(conds[i]);
    }

    // Fill maps based on condition for master side (masterdis == slavedis)
    ::DRT::UTILS::FindConditionObjects(
        *masterdis, masternodes, mastergnodes, masterelements, conds_master);

    // Fill maps based on condition for slave side (masterdis == slavedis)
    ::DRT::UTILS::FindConditionObjects(
        *slavedis, slavenodes, slavegnodes, slaveelements, conds_slave);
  }
  // Coupling condition is defined by "FSI COUPLING CONDITIONS"
  // There are two discretizations for the master and slave side. Therefore, the master/slave nodes
  // are chosen based on the discretization.
  else
  {
    // Fill maps based on condition for master side (masterdis != slavedis)
    ::DRT::UTILS::FindConditionObjects(
        *masterdis, masternodes, mastergnodes, masterelements, couplingcond);

    // Fill maps based on condition for slave side (masterdis != slavedis)
    ::DRT::UTILS::FindConditionObjects(
        *slavedis, slavenodes, slavegnodes, slaveelements, couplingcond);
  }

  // number of coupled dofs (defined in coupleddof by a 1)
  int numcoupleddof = 0;
  for (unsigned ii = 0; ii < coupleddof.size(); ++ii)
    if (coupleddof[ii] == 1) ++numcoupleddof;

  // setup mortar interface
  SetupInterface(masterdis, slavedis, coupleddof, mastergnodes, slavegnodes, masterelements,
      slaveelements, comm, slavewithale, slidingale, nds_master, nds_slave);

  // all the following stuff has to be done once in setup
  // in order to get initial D_ and M_

  // processor ID
  const int myrank = masterdis->Comm().MyPID();

  // get mortar coupling parameters
  Teuchos::ParameterList inputmortar;
  inputmortar.setParameters(contact_dynamic_params_);
  inputmortar.setParameters(mortar_coupling_params_);

  // interface displacement (=0) has to be merged from slave and master discretization
  Teuchos::RCP<Epetra_Map> dofrowmap =
      CORE::LINALG::MergeMap(masterdofrowmap_, slavedofrowmap_, false);
  Teuchos::RCP<Epetra_Vector> dispn = CORE::LINALG::CreateVector(*dofrowmap, true);

  // set displacement state in mortar interface
  interface_->SetState(MORTAR::state_new_displacement, *dispn);

  // print message
  if (myrank == 0)
  {
    std::cout << "\nPerforming mortar coupling...............";
    fflush(stdout);
  }

  // evaluate interface
  Evaluate();

  // print message
  if (myrank == 0) std::cout << "done!" << std::endl << std::endl;

  // initial mesh relocation:
  // For curved internal or fsi coupling interfaces, a mesh relocation is critical,
  // since the integration over curved interface (generation of mortar coupling
  // matrices) results in inaccuracies. These inaccuracies may lead to undesired node
  // displacements.
  // Example: nodes at the interface are also moved for matching discretizations
  // (P should be "unity matrix")!
  if (::DRT::INPUT::IntegralValue<INPAR::MORTAR::MeshRelocation>(inputmortar, "MESH_RELOCATION") ==
      INPAR::MORTAR::relocation_initial)
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
    // spatial displacements. Therefore, the slave and master map could be also used
    // to store the coordinates of the interface nodes, which is necessary to perform
    // the mesh relocation. Hence, this method cannot be used for problem types such
    // as elch, scatra, etc., having less coupling degrees of freedom than spatial
    // dimensions.
    Teuchos::RCP<Epetra_Vector> idisp(Teuchos::null);
    MeshRelocation(slavedis, aledis, masterdofrowmap_, slavedofrowmap_, idisp, comm, slavewithale);
  }

  // matrix transformation to initial parallel distribution
  MatrixRowColTransform();

  // check if slave dofs have dirichlet constraints
  CheckSlaveDirichletOverlap(slavedis, comm);

  // bye
  return;
}


/*----------------------------------------------------------------------*
 | check for overlap of slave and Dirichlet boundaries      farah 02/16 |
 *----------------------------------------------------------------------*/
void CORE::ADAPTER::CouplingMortar::CheckSlaveDirichletOverlap(
    const Teuchos::RCP<::DRT::Discretization>& slavedis, const Epetra_Comm& comm)
{
  // safety check
  CheckSetup();

  // check for overlap of slave and Dirichlet boundaries
  // (this is not allowed in order to avoid over-constraint)
  bool overlap = false;
  Teuchos::ParameterList p;
  p.set("total time", 0.0);
  Teuchos::RCP<CORE::LINALG::MapExtractor> dbcmaps = Teuchos::rcp(new CORE::LINALG::MapExtractor());
  Teuchos::RCP<Epetra_Vector> temp = CORE::LINALG::CreateVector(*(slavedis->DofRowMap()), true);
  slavedis->EvaluateDirichlet(p, temp, Teuchos::null, Teuchos::null, Teuchos::null, dbcmaps);

  // loop over all slave row nodes of the interface
  for (int j = 0; j < interface_->SlaveRowNodes()->NumMyElements(); ++j)
  {
    int gid = interface_->SlaveRowNodes()->GID(j);
    ::DRT::Node* node = interface_->Discret().gNode(gid);
    if (!node) dserror("Cannot find node with gid %", gid);
    MORTAR::MortarNode* mtnode = static_cast<MORTAR::MortarNode*>(node);

    // check if this node's dofs are in dbcmap
    for (int k = 0; k < mtnode->NumDof(); ++k)
    {
      int currdof = mtnode->Dofs()[k];
      int lid = (dbcmaps->CondMap())->LID(currdof);

      // found slave node with dbc
      if (lid >= 0)
      {
        overlap = true;
        break;
      }
    }
  }

  // print warning message to screen
  if (overlap && comm.MyPID() == 0)
  {
    if (comm.MyPID() == 0)
      dserror(
          "Slave boundary and Dirichlet boundary conditions overlap!\n"
          "This leads to an over-constraint problem setup");
  }

  return;
}


/*----------------------------------------------------------------------*
 | setup routine for mortar framework                        ehrl 08/13 |
 *----------------------------------------------------------------------*/
void CORE::ADAPTER::CouplingMortar::SetupInterface(
    const Teuchos::RCP<::DRT::Discretization>& masterdis,  ///< master discretization
    const Teuchos::RCP<::DRT::Discretization>& slavedis,   ///< slave discretization
    const std::vector<int>& coupleddof,  ///< vector defining coupled degrees of freedom
    const std::map<int, ::DRT::Node*>& mastergnodes,  ///< master nodes, including ghosted nodes
    const std::map<int, ::DRT::Node*>& slavegnodes,   ///< slave nodes, including ghosted nodes
    const std::map<int, Teuchos::RCP<::DRT::Element>>& masterelements,  ///< master elements
    const std::map<int, Teuchos::RCP<::DRT::Element>>& slaveelements,   ///< slave elements
    const Epetra_Comm& comm,                                            ///< communicator
    const bool slavewithale,  ///< flag defining if slave is ALE
    const bool slidingale,    ///< flag indicating sliding ALE case
    const int nds_master,     ///< master dofset number
    const int nds_slave       ///< slave dofset number
)
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
  const bool nurbs = shape_function_type_ == CORE::FE::ShapeFunctionType::nurbs;
  input.set<bool>("NURBS", nurbs);

  // set valid parameter values
  input.set<std::string>("LM_SHAPEFCN", "dual");
  input.set<std::string>("LM_DUAL_CONSISTENT", "none");
  input.sublist("PARALLEL REDISTRIBUTION").set<std::string>("PARALLEL_REDIST", "none");
  input.set<int>("DIMENSION", spatial_dimension_);

  // create an empty mortar interface
  interface_ = MORTAR::MortarInterface::Create(0, comm, spatial_dimension_, input);

  // number of dofs per node based on the coupling vector coupleddof
  const int dof = coupleddof.size();
  if ((masterdis->NumMyRowNodes() > 0 and
          (masterdis->NumDof(nds_master, masterdis->lRowNode(0)) != dof and slavewithale == true and
              slidingale == false)) or
      (slavedis->NumMyRowNodes() > 0 and
          (slavedis->NumDof(nds_slave, slavedis->lRowNode(0)) != dof and slavewithale == false and
              slidingale == false)))
  {
    dserror(
        "The size of the coupling vector coupleddof and dof defined in the discretization does not "
        "fit!! \n"
        "dof defined in the discretization: %i \n"
        "length of coupleddof: %i",
        masterdis->NumDof(nds_master, masterdis->lRowNode(0)), dof);
  }

  // special case: sliding ale
  // In the sliding ale framework two mortar discretizations are generated from identical
  // masterelement and slaveelement sets. Since node-, dof- and element ids of the original elements
  // are the same, an offset have to be defined
  int nodeoffset = 0;
  int dofoffset = 0;
  if (slidingale == true)
  {
    nodeoffset = masterdis->NodeRowMap()->MaxAllGID() + 1;
    dofoffset = masterdis->DofRowMap(nds_master)->MaxAllGID() + 1;
  }

  // number of coupled dofs (defined in coupleddof by a 1)
  int numcoupleddof = 0;
  for (int ii = 0; ii < dof; ++ii)
    if (coupleddof[ii] == 1) ++numcoupleddof;

  // feeding master nodes to the interface including ghosted nodes
  std::map<int, ::DRT::Node*>::const_iterator nodeiter;
  for (nodeiter = mastergnodes.begin(); nodeiter != mastergnodes.end(); ++nodeiter)
  {
    ::DRT::Node* node = nodeiter->second;
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
        dofids[ii] = masterdis->Dof(nds_master, node)[k];
        ii += 1;
      }
    }
    Teuchos::RCP<MORTAR::MortarNode> mrtrnode =
        Teuchos::rcp(new MORTAR::MortarNode(node->Id(), node->X(), node->Owner(), dofids, false));

    if (nurbs) MORTAR::UTILS::PrepareNURBSNode(node, mrtrnode);
    interface_->AddMortarNode(mrtrnode);
  }

  // feeding slave nodes to the interface including ghosted nodes
  for (nodeiter = slavegnodes.begin(); nodeiter != slavegnodes.end(); ++nodeiter)
  {
    ::DRT::Node* node = nodeiter->second;
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
        dofids[ii] = slavedis->Dof(nds_slave, node)[k] + dofoffset;
        ii += 1;
      }
    }
    Teuchos::RCP<MORTAR::MortarNode> mrtrnode = Teuchos::rcp(
        new MORTAR::MortarNode(node->Id() + nodeoffset, node->X(), node->Owner(), dofids, true));

    if (nurbs) MORTAR::UTILS::PrepareNURBSNode(node, mrtrnode);
    interface_->AddMortarNode(mrtrnode);
  }

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
    comm.SumAll(&nummastermtreles, &eleoffset, 1);
  }

  if (slidingale == true) eleoffset = masterdis->ElementRowMap()->MaxAllGID() + 1;

  // feeding master elements to the interface
  std::map<int, Teuchos::RCP<::DRT::Element>>::const_iterator elemiter;
  for (elemiter = masterelements.begin(); elemiter != masterelements.end(); ++elemiter)
  {
    Teuchos::RCP<::DRT::Element> ele = elemiter->second;
    Teuchos::RCP<MORTAR::MortarElement> mrtrele = Teuchos::rcp(new MORTAR::MortarElement(
        ele->Id(), ele->Owner(), ele->Shape(), ele->NumNode(), ele->NodeIds(), false, nurbs));

    if (nurbs) MORTAR::UTILS::PrepareNURBSElement(*masterdis, ele, mrtrele, spatial_dimension_);
    interface_->AddMortarElement(mrtrele);
  }

  // feeding slave elements to the interface
  for (elemiter = slaveelements.begin(); elemiter != slaveelements.end(); ++elemiter)
  {
    Teuchos::RCP<::DRT::Element> ele = elemiter->second;

    // Here, we have to distinguish between standard and sliding ale since mortar elements are
    // generated from the identical element sets in the case of sliding ale Therefore, we introduce
    // an element offset AND a node offset for the the slave mortar elements
    if (slidingale == false)
    {
      Teuchos::RCP<MORTAR::MortarElement> mrtrele =
          Teuchos::rcp(new MORTAR::MortarElement(ele->Id() + eleoffset, ele->Owner(), ele->Shape(),
              ele->NumNode(), ele->NodeIds(), true, nurbs));

      if (nurbs) MORTAR::UTILS::PrepareNURBSElement(*slavedis, ele, mrtrele, spatial_dimension_);
      interface_->AddMortarElement(mrtrele);
    }
    else
    {
      std::vector<int> nidsoff;
      for (int i = 0; i < ele->NumNode(); i++)
      {
        nidsoff.push_back(ele->NodeIds()[ele->NumNode() - 1 - i] + nodeoffset);
      }

      Teuchos::RCP<MORTAR::MortarElement> mrtrele =
          Teuchos::rcp(new MORTAR::MortarElement(ele->Id() + eleoffset, ele->Owner(), ele->Shape(),
              ele->NumNode(), nidsoff.data(), true, nurbs));

      interface_->AddMortarElement(mrtrele);
    }
  }

  /* Finalize the interface construction
   *
   * If this is the final parallel distribution, we need to assign degrees of freedom during
   * during FillComplete(). If parallel redistribution is enabled, there will be another call to
   * FillComplete(), so we skip this expensive operation here and do it later. DOFs have to be
   * assigned only once!
   */
  const INPAR::MORTAR::ParallelRedist parallelRedist =
      Teuchos::getIntegralValue<INPAR::MORTAR::ParallelRedist>(
          input.sublist("PARALLEL REDISTRIBUTION"), "PARALLEL_REDIST");
  {
    bool isFinalDistribution = false;
    if (parallelRedist == INPAR::MORTAR::ParallelRedist::redist_none or comm.NumProc() == 1)
      isFinalDistribution = true;

    interface_->FillComplete(isFinalDistribution);
  }

  // set setup flag!
  issetup_ = true;

  // store old row maps (before parallel redistribution)
  pslavedofrowmap_ = Teuchos::rcp(new Epetra_Map(*interface_->SlaveRowDofs()));
  pmasterdofrowmap_ = Teuchos::rcp(new Epetra_Map(*interface_->MasterRowDofs()));

  // print parallel distribution
  interface_->PrintParallelDistribution();

  //**********************************************************************
  // PARALLEL REDISTRIBUTION OF INTERFACE
  //**********************************************************************
  if (parallelRedist != INPAR::MORTAR::ParallelRedist::redist_none and comm.NumProc() > 1)
  {
    // redistribute optimally among all procs
    interface_->Redistribute();

    // call fill complete again
    interface_->FillComplete(true);

    // print parallel distribution again
    interface_->PrintParallelDistribution();
  }
  //**********************************************************************

  // store row maps (after parallel redistribution)
  slavedofrowmap_ = Teuchos::rcp(new Epetra_Map(*interface_->SlaveRowDofs()));
  masterdofrowmap_ = Teuchos::rcp(new Epetra_Map(*interface_->MasterRowDofs()));

  // create binary search tree
  interface_->CreateSearchTree();

  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CORE::ADAPTER::CouplingMortar::MeshRelocation(Teuchos::RCP<::DRT::Discretization> slavedis,
    Teuchos::RCP<::DRT::Discretization> aledis, Teuchos::RCP<const Epetra_Map> masterdofrowmap,
    Teuchos::RCP<const Epetra_Map> slavedofrowmap, Teuchos::RCP<Epetra_Vector>& idisp,
    const Epetra_Comm& comm, bool slavewithale)
{
  // safety check
  CheckSetup();

  // problem dimension
  const int dim = spatial_dimension_;

  //**********************************************************************
  // (0) check constraints in reference configuration
  //**********************************************************************
  // build global vectors of slave and master coordinates
  Teuchos::RCP<Epetra_Vector> xs = CORE::LINALG::CreateVector(*slavedofrowmap, true);
  Teuchos::RCP<Epetra_Vector> xm = CORE::LINALG::CreateVector(*masterdofrowmap, true);

  // loop over all slave row nodes
  for (int j = 0; j < interface_->SlaveRowNodes()->NumMyElements(); ++j)
  {
    int gid = interface_->SlaveRowNodes()->GID(j);
    ::DRT::Node* node = interface_->Discret().gNode(gid);
    if (!node) dserror("Cannot find node with gid %", gid);
    MORTAR::MortarNode* mtnode = static_cast<MORTAR::MortarNode*>(node);

    // prepare assembly
    // minimum three coupling dof's otherwise this method is not working
    // since the slave map is based on dof<dim
    // -> spacial coordinates cannot be assembled in vector based on this map
    CORE::LINALG::SerialDenseVector val(dim);
    std::vector<int> lm(dim, 0.0);
    std::vector<int> lmowner(dim, 0.0);

    for (int k = 0; k < dim; ++k)
    {
      val[k] = mtnode->X()[k];
      lm[k] = mtnode->Dofs()[k];
      lmowner[k] = mtnode->Owner();
    }


    // add ALE displacements, if required
    if (idisp != Teuchos::null)
    {
      // get degrees of freedom of a node
      std::vector<int> gdofs = interface_->Discret().Dof(node);

      for (int k = 0; k < dim; ++k)
      {
        val[k] += (*idisp)[(idisp->Map()).LID(gdofs[k])];
      }
    }

    // do assembly
    CORE::LINALG::Assemble(*xs, val, lm, lmowner);
  }

  // loop over all master row nodes
  for (int j = 0; j < interface_->MasterRowNodes()->NumMyElements(); ++j)
  {
    int gid = interface_->MasterRowNodes()->GID(j);
    ::DRT::Node* node = interface_->Discret().gNode(gid);
    if (!node) dserror("Cannot find node with gid %", gid);
    MORTAR::MortarNode* mtnode = static_cast<MORTAR::MortarNode*>(node);

    // prepare assembly
    // minimum three coupling dof's otherwise this method is not working
    // since the slave map is based on dof<dim
    // -> spacial coordinates cannot be assembled in vector based on this map
    CORE::LINALG::SerialDenseVector val(dim);
    std::vector<int> lm(dim, 0.0);
    std::vector<int> lmowner(dim, 0.0);

    for (int k = 0; k < dim; ++k)
    {
      val[k] = mtnode->X()[k];
      lm[k] = mtnode->Dofs()[k];
      lmowner[k] = mtnode->Owner();
    }

    // add ALE displacements, if required
    if (idisp != Teuchos::null)
    {
      // get degrees of freedom of a node
      std::vector<int> gdofs = interface_->Discret().Dof(node);

      for (int k = 0; k < dim; ++k)
      {
        val[k] += (*idisp)[(idisp->Map()).LID(gdofs[k])];
      }
    }

    // do assembly
    CORE::LINALG::Assemble(*xm, val, lm, lmowner);
  }

  // compute g-vector at global level
  Teuchos::RCP<Epetra_Vector> Dxs = Teuchos::rcp(new Epetra_Vector(*slavedofrowmap));
  D_->Multiply(false, *xs, *Dxs);
  Teuchos::RCP<Epetra_Vector> Mxm = Teuchos::rcp(new Epetra_Vector(*slavedofrowmap));
  M_->Multiply(false, *xm, *Mxm);
  Teuchos::RCP<Epetra_Vector> gold = CORE::LINALG::CreateVector(*slavedofrowmap, true);
  gold->Update(1.0, *Dxs, 1.0);
  gold->Update(-1.0, *Mxm, 1.0);
  double gnorm = 0.0;
  gold->Norm2(&gnorm);
  gnorm /= sqrt((double)gold->GlobalLength());  // scale with length of vector

  const double tol = 1.0e-12;
  // no need to do mesh relocation if g already very small

  if (comm.MyPID() == 0)
  {
    std::cout << "Analyze interface quality: L2-norm of gap vector = " << gnorm
              << " whereas tol = " << tol << std::endl;

    if (gnorm < tol) std::cout << "  --> Mesh relocation is not necessary. " << std::endl;
  }

  if (gnorm < tol) return;

  // print message
  if (comm.MyPID() == 0)
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
  // evaluation! Thus, the modified slave positions are first exported
  // to the column map of the respective interface and the modification
  // loop is then also done with respect to this node column map!
  // A second concern is that we are dealing with a special interface
  // discretization (including special meshtying nodes, too) here, This
  // interface discretization has been set up for dealing with meshtying
  // ONLY, and there is still the underlying problem discretization
  // dealing with the classical finite element evaluation. Thus, it is
  // very important that we apply the nodal relocation to BOTH the
  // MortarNodes in the meshtying interface discretization AND to the
  // DRT:Nodes in the underlying problem discretization.
  // Finally, we have to ask ourselves whether the node column distribution
  // of the slave nodes in the interface discretization is IDENTICAL
  // to the distribution in the underlying problem discretization. This
  // is NOT necessarily the case, as we might have redistributed the
  // interface among all processors. Thus, we loop over the fully over-
  // lapping slave column map here to keep all processors around. Then,
  // the first modification (MortarNode) is always performed, but the
  // second modification (::DRT::Node) is only performed if the respective
  // node in contained in the problem node column map.
  //**********************************************************************

  //**********************************************************************
  // (1) get master positions on global level
  //**********************************************************************
  // fill Xmaster first
  Teuchos::RCP<Epetra_Vector> Xmaster = CORE::LINALG::CreateVector(*masterdofrowmap, true);

  // loop over all master row nodes on the current interface
  for (int j = 0; j < interface_->MasterRowNodes()->NumMyElements(); ++j)
  {
    int gid = interface_->MasterRowNodes()->GID(j);
    ::DRT::Node* node = interface_->Discret().gNode(gid);
    if (!node) dserror("Cannot find node with gid %", gid);
    MORTAR::MortarNode* mtnode = static_cast<MORTAR::MortarNode*>(node);

    // do assembly (overwrite duplicate nodes)
    // minimum three coupling dof's otherwise this method is not working
    // since the slave map is based on dof<dim
    // -> spacial coordinates cannot be assembled in vector based on this map
    for (int k = 0; k < dim; ++k)
    {
      int dof = mtnode->Dofs()[k];
      (*Xmaster)[(Xmaster->Map()).LID(dof)] = mtnode->X()[k];

      // add ALE displacements, if required
      if (idisp != Teuchos::null)
        (*Xmaster)[(Xmaster->Map()).LID(dof)] += (*idisp)[(idisp->Map()).LID(dof)];
    }
  }

  //**********************************************************************
  // (2) solve for modified slave positions on global level
  //**********************************************************************
  // relocate modified slave positions
  Teuchos::RCP<Epetra_Vector> Xslavemod = CORE::LINALG::CreateVector(*slavedofrowmap, true);

  // this is trivial for dual Lagrange multipliers
  P_->Multiply(false, *Xmaster, *Xslavemod);


  //**********************************************************************
  // (3) perform mesh relocation node by node
  //**********************************************************************
  // export Xslavemod to fully overlapping column map for current interface
  Teuchos::RCP<Epetra_Map> fullsdofs = CORE::LINALG::AllreduceEMap(*(interface_->SlaveRowDofs()));
  Teuchos::RCP<Epetra_Map> fullsnodes = CORE::LINALG::AllreduceEMap(*(interface_->SlaveRowNodes()));
  Epetra_Vector Xslavemodcol(*fullsdofs, false);
  CORE::LINALG::Export(*Xslavemod, Xslavemodcol);

  // loop over all slave nodes on the current interface
  for (int j = 0; j < fullsnodes->NumMyElements(); ++j)
  {
    // get global ID of current node
    int gid = fullsnodes->GID(j);

    // be careful to modify BOTH mtnode in interface discret ...
    // (check if the node is available on this processor)
    bool isininterfacecolmap = false;
    int ilid = interface_->SlaveColNodes()->LID(gid);
    if (ilid >= 0) isininterfacecolmap = true;
    ::DRT::Node* node = nullptr;
    MORTAR::MortarNode* mtnode = nullptr;
    if (isininterfacecolmap)
    {
      node = interface_->Discret().gNode(gid);
      if (!node) dserror("Cannot find node with gid %", gid);
      mtnode = static_cast<MORTAR::MortarNode*>(node);
    }

    // ... AND standard node in underlying slave discret
    // (check if the node is available on this processor)
    bool isinproblemcolmap = false;
    int lid = slavedis->NodeColMap()->LID(gid);
    if (lid >= 0) isinproblemcolmap = true;
    ::DRT::Node* pnode = nullptr;
    if (isinproblemcolmap)
    {
      pnode = slavedis->gNode(gid);
      if (!pnode) dserror("Cannot find node with gid %", gid);
    }

    // ... AND standard node in ALE discret if fluid=slave
    // (check if the node is available on this processor)
    bool isinproblemcolmap2 = false;
    ::DRT::Node* alenode = nullptr;
    if (aledis != Teuchos::null)
    {
      int lid2 = aledis->NodeColMap()->LID(gid);
      if (lid2 >= 0) isinproblemcolmap2 = true;
      if (isinproblemcolmap2)
      {
        alenode = aledis->gNode(gid);
        if (slavewithale and not alenode) dserror("Cannot find node with gid %", gid);
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
      if (comm.MyPID() == mtnode->Owner())
      {
        // get corresponding entries from Xslavemod
        int numdim = mtnode->Dim();

        // find DOFs of current node in Xslavemod and extract this node's position
        std::vector<int> locindex(numdim);

        for (int k = 0; k < numdim; ++k)
        {
          locindex[k] = (Xslavemodcol.Map()).LID(mtnode->Dofs()[k]);
          if (locindex[k] < 0) dserror("Did not find dof in map");
          Xnew[k] = Xslavemodcol[locindex[k]];
          Xold[k] = mtnode->X()[k];
          if (idisp != Teuchos::null)
            Xold[k] += (*idisp)[(idisp->Map()).LID(interface_->Discret().Dof(node)[k])];
        }

        // check is mesh distortion is still OK
        // (throw a dserror if length of relocation is larger than 80%
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
          dserror("Problem dimension must be either 2 or 3!");
        bool isok = mtnode->CheckMeshDistortion(relocation, limit);
        if (!isok) dserror("Mesh distortion generated by relocation is too large!");
      }
    }

    // communicate new position Xnew to all procs
    // (we can use SumAll here, as Xnew will be zero on all processors
    // except for the owner processor of the current node)
    comm.SumAll(Xnew.data(), Xnewglobal.data(), 3);

    // const_cast to force modifed X() into mtnode
    // const_cast to force modifed xspatial() into mtnode
    // const_cast to force modifed X() into pnode
    // const_cast to force modifed X() into alenode if fluid=slave
    // (remark: this is REALLY BAD coding)
    if (::DRT::INPUT::IntegralValue<INPAR::MORTAR::MeshRelocation>(
            mortar_coupling_params_, "MESH_RELOCATION") == INPAR::MORTAR::relocation_initial)
    {
      for (int k = 0; k < dim; ++k)
      {
        // modification in interface discretization
        if (isininterfacecolmap)
        {
          const_cast<double&>(mtnode->X()[k]) = Xnewglobal[k];
          const_cast<double&>(mtnode->xspatial()[k]) = Xnewglobal[k];
        }

        // modification in problem discretization
        if (isinproblemcolmap) const_cast<double&>(pnode->X()[k]) = Xnewglobal[k];

        // modification in ALE discretization
        if (isinproblemcolmap2 and slavewithale)
          const_cast<double&>(alenode->X()[k]) = Xnewglobal[k];
      }
    }
    else if (::DRT::INPUT::IntegralValue<INPAR::MORTAR::MeshRelocation>(
                 mortar_coupling_params_, "MESH_RELOCATION") == INPAR::MORTAR::relocation_timestep)
    {
      // modification of ALE displacements
      if (isininterfacecolmap and idisp != Teuchos::null)
      {
        // insertion solely done by owner processor of this node
        if (comm.MyPID() == node->Owner())
        {
          // define error variable
          int err(0);

          // get all degrees of freedom of this node
          std::vector<int> gdofs = interface_->Discret().Dof(node);

          // loop over spatial directions
          for (int k = 0; k < dim; ++k)
          {
            // get global ID of degree of freedom for this spatial direction
            int dofgid = (idisp->Map()).LID(gdofs[k]);
            // get new coordinate value for this spatial direction
            const double value = Xnewglobal[k] - node->X()[k];
            // replace respective value in displacement vector
            err = idisp->ReplaceMyValues(1, &value, &dofgid);
            // check whether there was a problem in the replacement process
            if (err != 0) dserror("error while inserting a value into ALE displacement vector!");
          }
        }
      }
    }
    else
      dserror("wrong input parameter for mortar-based MESH_RELOCATION!");
  }

  //**********************************************************************
  // (4) re-evaluate constraints in reference configuration
  //**********************************************************************
  // build global vectors of slave and master coordinates
  xs = CORE::LINALG::CreateVector(*slavedofrowmap, true);
  xm = CORE::LINALG::CreateVector(*masterdofrowmap, true);

  // loop over all slave row nodes
  for (int j = 0; j < interface_->SlaveRowNodes()->NumMyElements(); ++j)
  {
    int gid = interface_->SlaveRowNodes()->GID(j);
    ::DRT::Node* node = interface_->Discret().gNode(gid);
    if (!node) dserror("Cannot find node with gid %", gid);
    MORTAR::MortarNode* mtnode = static_cast<MORTAR::MortarNode*>(node);

    // prepare assembly
    CORE::LINALG::SerialDenseVector val(dim);
    std::vector<int> lm(dim);
    std::vector<int> lmowner(dim);

    for (int k = 0; k < dim; ++k)
    {
      val[k] = mtnode->X()[k];
      lm[k] = mtnode->Dofs()[k];
      lmowner[k] = mtnode->Owner();
    }

    // add ALE displacements, if required
    if (idisp != Teuchos::null)
    {
      // get degrees of freedom of a node
      std::vector<int> gdofs = interface_->Discret().Dof(node);

      for (int k = 0; k < dim; ++k)
      {
        val[k] += (*idisp)[(idisp->Map()).LID(gdofs[k])];
      }
    }

    // do assembly
    CORE::LINALG::Assemble(*xs, val, lm, lmowner);
  }

  // loop over all master row nodes
  for (int j = 0; j < interface_->MasterRowNodes()->NumMyElements(); ++j)
  {
    int gid = interface_->MasterRowNodes()->GID(j);
    ::DRT::Node* node = interface_->Discret().gNode(gid);
    if (!node) dserror("Cannot find node with gid %", gid);
    MORTAR::MortarNode* mtnode = static_cast<MORTAR::MortarNode*>(node);

    // prepare assembly
    CORE::LINALG::SerialDenseVector val(dim);
    std::vector<int> lm(dim);
    std::vector<int> lmowner(dim);

    for (int k = 0; k < dim; ++k)
    {
      val[k] = mtnode->X()[k];
      lm[k] = mtnode->Dofs()[k];
      lmowner[k] = mtnode->Owner();
    }

    // add ALE displacements, if required
    if (idisp != Teuchos::null)
    {
      // get degrees of freedom of a node
      std::vector<int> gdofs = interface_->Discret().Dof(node);

      for (int k = 0; k < dim; ++k)
      {
        val[k] += (*idisp)[(idisp->Map()).LID(gdofs[k])];
      }
    }

    // do assembly
    CORE::LINALG::Assemble(*xm, val, lm, lmowner);
  }

  // compute g-vector at global level
  Dxs = Teuchos::rcp(new Epetra_Vector(*slavedofrowmap));
  D_->Multiply(false, *xs, *Dxs);
  Mxm = Teuchos::rcp(new Epetra_Vector(*slavedofrowmap));
  M_->Multiply(false, *xm, *Mxm);
  Teuchos::RCP<Epetra_Vector> gnew = CORE::LINALG::CreateVector(*slavedofrowmap, true);
  gnew->Update(1.0, *Dxs, 1.0);
  gnew->Update(-1.0, *Mxm, 1.0);
  gnew->Norm2(&gnorm);
  gnorm /= sqrt((double)gnew->GlobalLength());  // scale with length of vector

  if (gnorm > tol)
    dserror(
        "Mesh relocation was not successful! \n "
        "Gap norm %e is larger than tolerance %e",
        gnorm, tol);

  //**********************************************************************
  // (5) re-relocate finite elements (if slave=structure)
  //**********************************************************************
  // if slave=fluid, we are lucky because fluid elements do not
  // need any re-relocation (unlike structural elements)
  // fluid elements: empty implementation (return 0)
  CORE::COMM::ParObjectFactory::Instance().InitializeElements(*slavedis);

  // print message
  if (comm.MyPID() == 0)
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
void CORE::ADAPTER::CouplingMortar::CreateP()
{
  // safety check
  CheckSetup();

  // check
  if (::DRT::INPUT::IntegralValue<INPAR::MORTAR::ShapeFcn>(
          Interface()->InterfaceParams(), "LM_SHAPEFCN") != INPAR::MORTAR::shape_dual)
    dserror("Creation of P operator only for dual shape functions!");

  /********************************************************************/
  /* Multiply Mortar matrices: P = inv(D) * M         A               */
  /********************************************************************/
  D_->Complete();
  Dinv_ = Teuchos::rcp(new CORE::LINALG::SparseMatrix(*D_));
  Teuchos::RCP<Epetra_Vector> diag = CORE::LINALG::CreateVector(*slavedofrowmap_, true);
  int err = 0;

  // extract diagonal of invd into diag
  Dinv_->ExtractDiagonalCopy(*diag);

  // set zero diagonal values to dummy 1.0
  for (int i = 0; i < diag->MyLength(); ++i)
  {
    if (abs((*diag)[i]) < 1e-12)
    {
      std::cout << "WARNING: Diagonal entry of D matrix (value = " << (*diag)[i]
                << ") is skipped because it is less than 1e-12!!!" << std::endl;
      (*diag)[i] = 1.0;
    }
  }

  // scalar inversion of diagonal values
  err = diag->Reciprocal(*diag);
  if (err != 0) dserror("Reciprocal: Zero diagonal entry!");

  // re-insert inverted diagonal into invd
  err = Dinv_->ReplaceDiagonalValues(*diag);
  if (err != 0) dserror("ReplaceDiagonalValues() failed with error code %d.", err);

  // complete inverse D matrix
  Dinv_->Complete();

  // do the multiplication P = inv(D) * M
  P_ = CORE::LINALG::MLMultiply(*Dinv_, false, *M_, false, false, false, true);

  // complete the matrix
  P_->Complete(*masterdofrowmap_, *slavedofrowmap_);

  // bye
  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CORE::ADAPTER::CouplingMortar::Evaluate(Teuchos::RCP<Epetra_Vector> idisp)
{
  // safety check
  CheckSetup();

  // set new displacement state in mortar interface
  interface_->SetState(MORTAR::state_new_displacement, *idisp);
  Evaluate();
  MatrixRowColTransform();

  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CORE::ADAPTER::CouplingMortar::Evaluate(
    Teuchos::RCP<Epetra_Vector> idispma, Teuchos::RCP<Epetra_Vector> idispsl)
{
  // safety checks
  CheckSetup();
  dsassert(idispma->Map().PointSameAs(*pmasterdofrowmap_),
      "Map of incoming master vector does not match the stored master dof row map.");
  dsassert(idispsl->Map().PointSameAs(*pslavedofrowmap_),
      "Map of incoming slave vector does not match the stored slave dof row map.");

  const Epetra_BlockMap stdmap = idispsl->Map();
  idispsl->ReplaceMap(*slavedofrowmap_);

  Teuchos::RCP<Epetra_Map> dofrowmap =
      CORE::LINALG::MergeMap(*pmasterdofrowmap_, *pslavedofrowmap_, false);
  Teuchos::RCP<Epetra_Import> master_importer =
      Teuchos::rcp(new Epetra_Import(*dofrowmap, *pmasterdofrowmap_));
  Teuchos::RCP<Epetra_Import> slaveImporter =
      Teuchos::rcp(new Epetra_Import(*dofrowmap, *pslavedofrowmap_));

  // Import master and slave displacements into a single vector
  int err = 0;
  Teuchos::RCP<Epetra_Vector> idisp_master_slave = CORE::LINALG::CreateVector(*dofrowmap, true);
  err = idisp_master_slave->Import(*idispma, *master_importer, Add);
  if (err != 0)
    dserror("Import failed with error code %d. See Epetra source code for details.", err);
  err = idisp_master_slave->Import(*idispsl, *slaveImporter, Add);
  if (err != 0)
    dserror("Import failed with error code %d. See Epetra source code for details.", err);

  // set new displacement state in mortar interface
  interface_->SetState(MORTAR::state_new_displacement, *idisp_master_slave);

  Evaluate();
  MatrixRowColTransform();

  idispsl->ReplaceMap(stdmap);

  return;
}


/*----------------------------------------------------------------------*
 *  Create integration cells for mortar interface           farah 01/16 |
 *----------------------------------------------------------------------*/
void CORE::ADAPTER::CouplingMortar::EvaluateGeometry(
    std::vector<Teuchos::RCP<MORTAR::IntCell>>& intcells  //!< vector of mortar integration cells
)
{
  // safety check
  CheckSetup();

  // evaluate geometry information
  interface_->EvaluateGeometry(intcells);

  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CORE::ADAPTER::CouplingMortar::Evaluate()
{
  // safety check
  CheckSetup();

  // in the following two steps MORTAR does all the work for new interface displacements
  interface_->Initialize();
  interface_->Evaluate();

  // preparation for AssembleDM
  // (Note that redistslave and redistmaster are the slave and master row maps
  // after parallel redistribution. If no redistribution was performed, they
  // are of course identical to slavedofrowmap_/masterdofrowmap_!)
  Teuchos::RCP<CORE::LINALG::SparseMatrix> dmatrix =
      Teuchos::rcp(new CORE::LINALG::SparseMatrix(*slavedofrowmap_, 10));
  Teuchos::RCP<CORE::LINALG::SparseMatrix> mmatrix =
      Teuchos::rcp(new CORE::LINALG::SparseMatrix(*slavedofrowmap_, 100));
  interface_->AssembleDM(*dmatrix, *mmatrix);

  // Complete() global Mortar matrices
  dmatrix->Complete();
  mmatrix->Complete(*masterdofrowmap_, *slavedofrowmap_);
  D_ = dmatrix;
  M_ = mmatrix;

  // create projection operator and Dinv
  CreateP();

  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CORE::ADAPTER::CouplingMortar::MatrixRowColTransform()
{
  // safety check
  CheckSetup();

  // check for parallel redistribution
  bool parredist = false;
  const Teuchos::ParameterList& input = mortar_coupling_params_.sublist("PARALLEL REDISTRIBUTION");
  if (Teuchos::getIntegralValue<INPAR::MORTAR::ParallelRedist>(input, "PARALLEL_REDIST") !=
      INPAR::MORTAR::ParallelRedist::redist_none)
    parredist = true;

  // only for parallel redistribution case
  if (parredist)
  {
    if (pslavedofrowmap_ == Teuchos::null or pmasterdofrowmap_ == Teuchos::null)
      dserror("Dof maps based on initial parallel distribution are wrong!");

    // transform everything back to old distribution
    D_ = MORTAR::MatrixRowColTransform(D_, pslavedofrowmap_, pslavedofrowmap_);
    M_ = MORTAR::MatrixRowColTransform(M_, pslavedofrowmap_, pmasterdofrowmap_);
    Dinv_ = MORTAR::MatrixRowColTransform(Dinv_, pslavedofrowmap_, pslavedofrowmap_);
    P_ = MORTAR::MatrixRowColTransform(P_, pslavedofrowmap_, pmasterdofrowmap_);
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CORE::ADAPTER::CouplingMortar::EvaluateWithMeshRelocation(
    Teuchos::RCP<::DRT::Discretization> slavedis, Teuchos::RCP<::DRT::Discretization> aledis,
    Teuchos::RCP<Epetra_Vector>& idisp, const Epetra_Comm& comm, bool slavewithale)
{
  // safety check
  CheckSetup();

  // set new displacement state in mortar interface
  interface_->SetState(MORTAR::state_new_displacement, *idisp);

  // in the following two steps MORTAR does all the work for new interface displacements
  interface_->Initialize();
  interface_->Evaluate();

  // preparation for AssembleDM
  // (Note that redistslave and redistmaster are the slave and master row maps
  // after parallel redistribution. If no redistribution was performed, they
  // are of course identical to slavedofrowmap_/masterdofrowmap_!)
  Teuchos::RCP<CORE::LINALG::SparseMatrix> dmatrix =
      Teuchos::rcp(new CORE::LINALG::SparseMatrix(*slavedofrowmap_, 10));
  Teuchos::RCP<CORE::LINALG::SparseMatrix> mmatrix =
      Teuchos::rcp(new CORE::LINALG::SparseMatrix(*slavedofrowmap_, 100));
  interface_->AssembleDM(*dmatrix, *mmatrix);

  // Complete() global Mortar matrices
  dmatrix->Complete();
  mmatrix->Complete(*masterdofrowmap_, *slavedofrowmap_);
  D_ = dmatrix;
  M_ = mmatrix;

  // Build Dinv
  Dinv_ = Teuchos::rcp(new CORE::LINALG::SparseMatrix(*D_));

  // extract diagonal of invd into diag
  Teuchos::RCP<Epetra_Vector> diag = CORE::LINALG::CreateVector(*slavedofrowmap_, true);
  Dinv_->ExtractDiagonalCopy(*diag);

  // set zero diagonal values to dummy 1.0
  for (int i = 0; i < diag->MyLength(); ++i)
    if ((*diag)[i] == 0.0) (*diag)[i] = 1.0;

  // scalar inversion of diagonal values
  diag->Reciprocal(*diag);
  Dinv_->ReplaceDiagonalValues(*diag);
  Dinv_->Complete(D_->RangeMap(), D_->DomainMap());
  P_ = MLMultiply(*Dinv_, *M_, false, false, true);

  // mesh relocation if required:
  // For curved internal or fsi coupling interfaces, a mesh relocation is critical,
  // since the integration over curved interface (generation of mortar coupling
  // matrices) results in inaccuracies. These inaccuracies may lead to undesired node
  // displacements.
  // Example: nodes at the interface are also moved for matching discretizations
  // (P should be "unity matrix")!
  if (::DRT::INPUT::IntegralValue<INPAR::MORTAR::MeshRelocation>(
          mortar_coupling_params_, "MESH_RELOCATION") == INPAR::MORTAR::relocation_timestep)
    MeshRelocation(slavedis, aledis, masterdofrowmap_, slavedofrowmap_, idisp, comm, slavewithale);

  // only for parallel redistribution case
  MatrixRowColTransform();

  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> CORE::ADAPTER::CouplingMortar::MasterToSlave(
    Teuchos::RCP<Epetra_Vector> mv) const
{
  // safety check
  CheckSetup();

  dsassert(masterdofrowmap_->SameAs(mv->Map()), "Vector with master dof map expected");

  Epetra_Vector tmp = Epetra_Vector(M_->RowMap());

  if (M_->Multiply(false, *mv, tmp)) dserror("M*mv multiplication failed");

  Teuchos::RCP<Epetra_Vector> sv = Teuchos::rcp(new Epetra_Vector(*pslavedofrowmap_));

  if (Dinv_->Multiply(false, tmp, *sv)) dserror("D^{-1}*v multiplication failed");

  return sv;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_MultiVector> CORE::ADAPTER::CouplingMortar::MasterToSlave(
    Teuchos::RCP<Epetra_MultiVector> mv) const
{
  // safety check
  CheckSetup();

  dsassert(masterdofrowmap_->SameAs(mv->Map()), "Vector with master dof map expected");

  Epetra_MultiVector tmp = Epetra_MultiVector(M_->RowMap(), mv->NumVectors());

  if (M_->Multiply(false, *mv, tmp)) dserror("M*mv multiplication failed");

  Teuchos::RCP<Epetra_MultiVector> sv =
      Teuchos::rcp(new Epetra_MultiVector(*pslavedofrowmap_, mv->NumVectors()));

  if (Dinv_->Multiply(false, tmp, *sv)) dserror("D^{-1}*v multiplication failed");

  return sv;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_MultiVector> CORE::ADAPTER::CouplingMortar::MasterToSlave(
    Teuchos::RCP<const Epetra_MultiVector> mv) const
{
  // safety check
  CheckSetup();

  dsassert(masterdofrowmap_->SameAs(mv->Map()), "Vector with master dof map expected");

  Epetra_MultiVector tmp = Epetra_MultiVector(M_->RowMap(), mv->NumVectors());

  if (M_->Multiply(false, *mv, tmp)) dserror("M*mv multiplication failed");

  Teuchos::RCP<Epetra_MultiVector> sv =
      Teuchos::rcp(new Epetra_MultiVector(*pslavedofrowmap_, mv->NumVectors()));

  if (Dinv_->Multiply(false, tmp, *sv)) dserror("D^{-1}*v multiplication failed");

  return sv;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> CORE::ADAPTER::CouplingMortar::MasterToSlave(
    Teuchos::RCP<const Epetra_Vector> mv) const
{
  // safety check
  CheckSetup();

  dsassert(masterdofrowmap_->SameAs(mv->Map()), "Vector with master dof map expected");

  Epetra_Vector tmp = Epetra_Vector(M_->RowMap());

  if (M_->Multiply(false, *mv, tmp)) dserror("M*mv multiplication failed");

  Teuchos::RCP<Epetra_Vector> sv = Teuchos::rcp(new Epetra_Vector(*pslavedofrowmap_));

  if (Dinv_->Multiply(false, tmp, *sv)) dserror("D^{-1}*v multiplication failed");

  return sv;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void CORE::ADAPTER::CouplingMortar::MasterToSlave(
    Teuchos::RCP<const Epetra_MultiVector> mv, Teuchos::RCP<Epetra_MultiVector> sv) const
{
#ifdef DEBUG
  if (not mv->Map().PointSameAs(P_->ColMap())) dserror("master dof map vector expected");
  if (not sv->Map().PointSameAs(D_->ColMap())) dserror("slave dof map vector expected");
#endif

  // safety check
  CheckSetup();

  // slave vector with auxiliary dofmap
  Epetra_MultiVector sv_aux(P_->RowMap(), sv->NumVectors());

  // project
  P_->Multiply(false, *mv, sv_aux);

  // copy from auxiliary to physical map (needed for coupling in fluid ale algorithm)
  std::copy(
      sv_aux.Values(), sv_aux.Values() + (sv_aux.MyLength() * sv_aux.NumVectors()), sv->Values());

  // in contrast to the ADAPTER::Coupling class we do not need to export here, as
  // the mortar interface itself has (or should have) guaranteed the same distribution of master and
  // slave dis on all procs
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void CORE::ADAPTER::CouplingMortar::SlaveToMaster(
    Teuchos::RCP<const Epetra_MultiVector> sv, Teuchos::RCP<Epetra_MultiVector> mv) const
{
#ifdef DEBUG
  if (not mv->Map().PointSameAs(P_->ColMap())) dserror("master dof map vector expected");
  if (not sv->Map().PointSameAs(D_->ColMap())) dserror("slave dof map vector expected");
#endif

  // safety check
  CheckSetup();

  Epetra_Vector tmp = Epetra_Vector(M_->RangeMap());
  std::copy(sv->Values(), sv->Values() + sv->MyLength(), tmp.Values());

  Teuchos::RCP<Epetra_Vector> tempm = Teuchos::rcp(new Epetra_Vector(*pmasterdofrowmap_));
  if (M_->Multiply(true, tmp, *tempm)) dserror("M^{T}*sv multiplication failed");

  // copy from auxiliary to physical map (needed for coupling in fluid ale algorithm)
  std::copy(
      tempm->Values(), tempm->Values() + (tempm->MyLength() * tempm->NumVectors()), mv->Values());

  // in contrast to the ADAPTER::Coupling class we do not need to export here, as
  // the mortar interface itself has (or should have) guaranteed the same distribution of master and
  // slave dis on all procs
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> CORE::ADAPTER::CouplingMortar::SlaveToMaster(
    Teuchos::RCP<const Epetra_Vector> sv) const
{
  // safety check
  CheckSetup();

  Epetra_Vector tmp = Epetra_Vector(M_->RangeMap());
  std::copy(sv->Values(), sv->Values() + sv->MyLength(), tmp.Values());

  Teuchos::RCP<Epetra_Vector> mv = Teuchos::rcp(new Epetra_Vector(*pmasterdofrowmap_));
  if (M_->Multiply(true, tmp, *mv)) dserror("M^{T}*sv multiplication failed");

  return mv;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> CORE::ADAPTER::CouplingMortar::SlaveToMaster(
    Teuchos::RCP<Epetra_Vector> sv) const
{
  // safety check
  CheckSetup();

  Epetra_Vector tmp = Epetra_Vector(M_->RangeMap());
  std::copy(sv->Values(), sv->Values() + sv->MyLength(), tmp.Values());

  Teuchos::RCP<Epetra_Vector> mv = Teuchos::rcp(new Epetra_Vector(*pmasterdofrowmap_));
  if (M_->Multiply(true, tmp, *mv)) dserror("M^{T}*sv multiplication failed");

  return mv;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_MultiVector> CORE::ADAPTER::CouplingMortar::SlaveToMaster(
    Teuchos::RCP<Epetra_MultiVector> sv) const
{
  // safety check
  CheckSetup();

  Epetra_MultiVector tmp = Epetra_MultiVector(M_->RangeMap(), sv->NumVectors());
  std::copy(sv->Values(), sv->Values() + sv->MyLength(), tmp.Values());

  Teuchos::RCP<Epetra_MultiVector> mv =
      Teuchos::rcp(new Epetra_MultiVector(*pmasterdofrowmap_, sv->NumVectors()));
  if (M_->Multiply(true, tmp, *mv)) dserror("M^{T}*sv multiplication failed");

  return mv;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_MultiVector> CORE::ADAPTER::CouplingMortar::SlaveToMaster(
    Teuchos::RCP<const Epetra_MultiVector> sv) const
{
  // safety check
  CheckSetup();

  Epetra_MultiVector tmp = Epetra_MultiVector(M_->RangeMap(), sv->NumVectors());
  std::copy(sv->Values(), sv->Values() + sv->MyLength(), tmp.Values());

  Teuchos::RCP<Epetra_MultiVector> mv =
      Teuchos::rcp(new Epetra_MultiVector(*pmasterdofrowmap_, sv->NumVectors()));
  if (M_->Multiply(true, tmp, *mv)) dserror("M^{T}*sv multiplication failed");

  return mv;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CORE::ADAPTER::CouplingMortar::MortarCondensation(
    Teuchos::RCP<CORE::LINALG::SparseMatrix>& k, Teuchos::RCP<Epetra_Vector>& rhs) const
{
  MORTAR::UTILS::MortarMatrixCondensation(k, P_, P_);
  MORTAR::UTILS::MortarRhsCondensation(rhs, P_);

  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CORE::ADAPTER::CouplingMortar::MortarRecover(
    Teuchos::RCP<CORE::LINALG::SparseMatrix>& k, Teuchos::RCP<Epetra_Vector>& inc) const
{
  MORTAR::UTILS::MortarRecover(inc, P_);
  return;
}
