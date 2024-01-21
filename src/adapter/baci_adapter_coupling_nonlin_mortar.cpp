/*----------------------------------------------------------------------*/
/*! \file

\brief A class providing coupling capabilities based on non-linear
       mortar methods

\level 2


*----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 |  headers                                                  farah 10/14|
 *----------------------------------------------------------------------*/
#include "baci_adapter_coupling_nonlin_mortar.H"

#include "baci_contact_element.H"
#include "baci_contact_friction_node.H"
#include "baci_contact_integrator.H"
#include "baci_contact_interface.H"
#include "baci_coupling_adapter_mortar.H"
#include "baci_global_data.H"
#include "baci_inpar_contact.H"
#include "baci_lib_condition_utils.H"
#include "baci_lib_discret.H"
#include "baci_linalg_multiply.H"
#include "baci_linalg_sparsematrix.H"
#include "baci_linalg_utils_sparse_algebra_create.H"
#include "baci_linalg_utils_sparse_algebra_manipulation.H"
#include "baci_mortar_utils.H"
#include "baci_nurbs_discret.H"
#include "baci_nurbs_discret_control_point.H"
#include "baci_nurbs_discret_knotvector.H"

#include <Epetra_Vector.h>

BACI_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 |  ctor                                                     farah 10/14|
 *----------------------------------------------------------------------*/
ADAPTER::CouplingNonLinMortar::CouplingNonLinMortar(int spatial_dimension,
    Teuchos::ParameterList mortar_coupling_params, Teuchos::ParameterList contact_dynamic_params,
    CORE::FE::ShapeFunctionType shape_function_type)
    : CORE::ADAPTER::CouplingMortar(
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
void ADAPTER::CouplingNonLinMortar::Setup(Teuchos::RCP<DRT::Discretization> masterdis,
    Teuchos::RCP<DRT::Discretization> slavedis, std::vector<int> coupleddof,
    const std::string& couplingcond)
{
  myrank_ = masterdis->Comm().MyPID();
  comm_ = Teuchos::rcp(masterdis->Comm().Clone());

  // ParameterList
  Teuchos::ParameterList input;

  // initialize maps for column nodes
  std::map<int, DRT::Node*> mastergnodes;
  std::map<int, DRT::Node*> slavegnodes;

  // initialize maps for elements
  std::map<int, Teuchos::RCP<DRT::Element>> masterelements;
  std::map<int, Teuchos::RCP<DRT::Element>> slaveelements;

  Teuchos::RCP<CONTACT::Interface> interface;

  // number of dofs per node based on the coupling vector coupleddof
  const int dof = coupleddof.size();

  // number of coupled dofs (defined in coupleddof by a 1)
  int numcoupleddof = 0;
  for (int ii = 0; ii < dof; ++ii)
    if (coupleddof[ii] == 1) numcoupleddof += 1;

  // read the mortar conditions and set probtype
  ReadMortarCondition(masterdis, slavedis, coupleddof, couplingcond, input, mastergnodes,
      slavegnodes, masterelements, slaveelements);

  // add contact nodes to interface discr.
  AddMortarNodes(masterdis, slavedis, coupleddof, input, mastergnodes, slavegnodes, masterelements,
      slaveelements, interface, numcoupleddof);

  // add contact eles to interface discr.
  AddMortarElements(
      masterdis, slavedis, input, masterelements, slaveelements, interface, numcoupleddof);

  // complete interface, store as int. var. and do
  // parallel red.
  CompleteInterface(masterdis, interface);

  // Initialize matrices
  InitMatrices();

  // create stratgy object if required
  CreateStrategy(masterdis, slavedis, input, numcoupleddof);

  // set setup flag
  issetup_ = true;

  return;
}


/*----------------------------------------------------------------------*
 |  read mortar condition                                    farah 10/14|
 *----------------------------------------------------------------------*/
void ADAPTER::CouplingNonLinMortar::CreateStrategy(Teuchos::RCP<DRT::Discretization> masterdis,
    Teuchos::RCP<DRT::Discretization> slavedis, Teuchos::ParameterList& input, int numcoupleddof)
{
  // nothing to do for pure adapter
  return;
}


/*----------------------------------------------------------------------*
 |  read mortar condition                                    farah 10/14|
 *----------------------------------------------------------------------*/
void ADAPTER::CouplingNonLinMortar::ReadMortarCondition(Teuchos::RCP<DRT::Discretization> masterdis,
    Teuchos::RCP<DRT::Discretization> slavedis, std::vector<int> coupleddof,
    const std::string& couplingcond, Teuchos::ParameterList& input,
    std::map<int, DRT::Node*>& mastergnodes, std::map<int, DRT::Node*>& slavegnodes,
    std::map<int, Teuchos::RCP<DRT::Element>>& masterelements,
    std::map<int, Teuchos::RCP<DRT::Element>>& slaveelements)
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
  std::map<int, DRT::Node*> masternodes;
  std::map<int, DRT::Node*> slavenodes;

  // Coupling condition is defined by "MORTAR COUPLING CONDITIONS"
  // There is only one discretization (masterdis == slavedis). Therefore, the node set have to be
  // separated beforehand.
  if (couplingcond == "Mortar" || couplingcond == "Contact" || couplingcond == "EHLCoupling")
  {
    std::vector<DRT::Condition*> conds;
    std::vector<DRT::Condition*> conds_master(0);
    std::vector<DRT::Condition*> conds_slave(0);
    masterdis->GetCondition(couplingcond, conds);

    for (unsigned i = 0; i < conds.size(); i++)
    {
      const std::string* side = conds[i]->Get<std::string>("Side");

      if (*side == "Master") conds_master.push_back(conds[i]);

      if (*side == "Slave") conds_slave.push_back(conds[i]);
    }

    // Fill maps based on condition for master side (masterdis == slavedis)
    DRT::UTILS::FindConditionObjects(
        *masterdis, masternodes, mastergnodes, masterelements, conds_master);

    // Fill maps based on condition for slave side (masterdis == slavedis)
    DRT::UTILS::FindConditionObjects(
        *slavedis, slavenodes, slavegnodes, slaveelements, conds_slave);
  }
  // Coupling condition is defined by "FSI COUPLING CONDITIONS"
  // There are two discretizations for the master and slave side. Therefore, the master/slave nodes
  // are chosen based on the discretization.
  else
  {
    // Fill maps based on condition for master side (masterdis != slavedis)
    //    if(masterdis!=Teuchos::null)
    //      DRT::UTILS::FindConditionObjects(*masterdis, masternodes, mastergnodes, masterelements,
    //      couplingcond);

    // Fill maps based on condition for slave side (masterdis != slavedis)
    if (slavedis != Teuchos::null)
      DRT::UTILS::FindConditionObjects(
          *slavedis, slavenodes, slavegnodes, slaveelements, couplingcond);
  }

  // get mortar coupling parameters
  const Teuchos::ParameterList& inputmortar = DRT::Problem::Instance()->MortarCouplingParams();
  const Teuchos::ParameterList& meshtying = DRT::Problem::Instance()->ContactDynamicParams();
  const Teuchos::ParameterList& wearlist = DRT::Problem::Instance()->WearParams();

  input.setParameters(inputmortar);
  input.setParameters(meshtying);
  input.setParameters(wearlist);

  input.set<int>("PROBTYPE", INPAR::CONTACT::other);  // if other probtypes, this will be
                                                      // overwritten in overloaded function

  // is this a nurbs problem?
  bool isnurbs = false;
  CORE::FE::ShapeFunctionType distype = DRT::Problem::Instance()->SpatialApproximationType();
  if (distype == CORE::FE::ShapeFunctionType::nurbs) isnurbs = true;
  input.set<bool>("NURBS", isnurbs);
  input.set<int>("DIMENSION", DRT::Problem::Instance()->NDim());

  // check for invalid parameter values
  if (INPUT::IntegralValue<INPAR::MORTAR::ShapeFcn>(input, "LM_SHAPEFCN") !=
          INPAR::MORTAR::shape_dual and
      INPUT::IntegralValue<INPAR::MORTAR::ShapeFcn>(input, "LM_SHAPEFCN") !=
          INPAR::MORTAR::shape_petrovgalerkin)
    if (myrank_ == 0) dserror("Mortar coupling adapter only works for dual shape functions");

  // as two half pass approach is not implemented for this approach set false
  input.set<bool>("Two_half_pass", false);

  return;
}


/*----------------------------------------------------------------------*
 |  add mortar nodes                                         farah 10/14|
 *----------------------------------------------------------------------*/
void ADAPTER::CouplingNonLinMortar::AddMortarNodes(Teuchos::RCP<DRT::Discretization> masterdis,
    Teuchos::RCP<DRT::Discretization> slavedis, std::vector<int> coupleddof,
    Teuchos::ParameterList& input, std::map<int, DRT::Node*>& mastergnodes,
    std::map<int, DRT::Node*>& slavegnodes,
    std::map<int, Teuchos::RCP<DRT::Element>>& masterelements,
    std::map<int, Teuchos::RCP<DRT::Element>>& slaveelements,
    Teuchos::RCP<CONTACT::Interface>& interface, int numcoupleddof)
{
  const bool isnurbs = input.get<bool>("NURBS");

  // get problem dimension (2D or 3D) and create (MORTAR::Interface)
  const int dim = DRT::Problem::Instance()->NDim();

  // create an empty mortar interface
  interface = CONTACT::Interface::Create(0, *comm_, dim, input, false);

  //  if((masterdis->NumDof(masterdis->lRowNode(0))!=dof and slavewithale==true and
  //  slidingale==false) or
  //      (slavedis->NumDof(slavedis->lRowNode(0))!=dof and slavewithale==false and
  //      slidingale==false))
  //  {
  //    dserror("The size of the coupling vector coupleddof and dof defined in the discretization
  //    does not fit!! \n"
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
  //    dofoffset = masterdis->DofRowMap()->MaxAllGID()+1;
  //  }
  // ########## CHECK for a better implementation of this ###################

  // feeding master nodes to the interface including ghosted nodes
  std::map<int, DRT::Node*>::const_iterator nodeiter;
  for (nodeiter = mastergnodes.begin(); nodeiter != mastergnodes.end(); ++nodeiter)
  {
    DRT::Node* node = nodeiter->second;
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
        dofids[ii] = masterdis->Dof(0, node)[k];
        ii += 1;
      }
    }
    Teuchos::RCP<CONTACT::Node> cnode = Teuchos::rcp(
        new CONTACT::FriNode(node->Id(), node->X(), node->Owner(), dofids, false, false, false));

    if (isnurbs)
    {
      DRT::NURBS::ControlPoint* cp = dynamic_cast<DRT::NURBS::ControlPoint*>(node);

      cnode->NurbsW() = cp->W();
    }

    interface->AddNode(cnode);
  }

  // feeding slave nodes to the interface including ghosted nodes
  for (nodeiter = slavegnodes.begin(); nodeiter != slavegnodes.end(); ++nodeiter)
  {
    DRT::Node* node = nodeiter->second;
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
        dofids[ii] = slavedis->Dof(0, node)[k] + dofoffset;
        ii += 1;
      }
    }
    Teuchos::RCP<CONTACT::Node> cnode = Teuchos::rcp(
        new CONTACT::FriNode(node->Id(), node->X(), node->Owner(), dofids, true, true, false));

    if (isnurbs)
    {
      DRT::NURBS::ControlPoint* cp = dynamic_cast<DRT::NURBS::ControlPoint*>(node);

      cnode->NurbsW() = cp->W();
    }

    interface->AddNode(cnode);
  }
}


/*----------------------------------------------------------------------*
 |  add mortar elements                                      farah 10/14|
 *----------------------------------------------------------------------*/
void ADAPTER::CouplingNonLinMortar::AddMortarElements(Teuchos::RCP<DRT::Discretization> masterdis,
    Teuchos::RCP<DRT::Discretization> slavedis, Teuchos::ParameterList& input,
    std::map<int, Teuchos::RCP<DRT::Element>>& masterelements,
    std::map<int, Teuchos::RCP<DRT::Element>>& slaveelements,
    Teuchos::RCP<CONTACT::Interface>& interface, int numcoupleddof)
{
  const bool isnurbs = input.get<bool>("NURBS");

  // get problem dimension (2D or 3D) and create (MORTAR::Interface)
  const int dim = DRT::Problem::Instance()->NDim();

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
  //    dofoffset = masterdis->DofRowMap()->MaxAllGID()+1;
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
  std::map<int, Teuchos::RCP<DRT::Element>>::const_iterator elemiter;
  for (elemiter = masterelements.begin(); elemiter != masterelements.end(); ++elemiter)
  {
    Teuchos::RCP<DRT::Element> ele = elemiter->second;
    Teuchos::RCP<CONTACT::Element> cele = Teuchos::rcp(new CONTACT::Element(
        ele->Id(), ele->Owner(), ele->Shape(), ele->NumNode(), ele->NodeIds(), false, isnurbs));

    if (isnurbs)
    {
      Teuchos::RCP<DRT::NURBS::NurbsDiscretization> nurbsdis =
          Teuchos::rcp_dynamic_cast<DRT::NURBS::NurbsDiscretization>(masterdis);

      Teuchos::RCP<DRT::NURBS::Knotvector> knots = (*nurbsdis).GetKnotVector();
      std::vector<CORE::LINALG::SerialDenseVector> parentknots(dim);
      std::vector<CORE::LINALG::SerialDenseVector> mortarknots(dim - 1);

      Teuchos::RCP<DRT::FaceElement> faceele =
          Teuchos::rcp_dynamic_cast<DRT::FaceElement>(ele, true);
      double normalfac = 0.0;
      bool zero_size = knots->GetBoundaryEleAndParentKnots(parentknots, mortarknots, normalfac,
          faceele->ParentMasterElement()->Id(), faceele->FaceMasterNumber());

      // store nurbs specific data to node
      cele->ZeroSized() = zero_size;
      cele->Knots() = mortarknots;
      cele->NormalFac() = normalfac;
    }

    interface->AddElement(cele);
  }

  // feeding slave elements to the interface
  for (elemiter = slaveelements.begin(); elemiter != slaveelements.end(); ++elemiter)
  {
    Teuchos::RCP<DRT::Element> ele = elemiter->second;

    // Here, we have to distinguish between standard and sliding ale since mortar elements are
    // generated from the identical element sets in the case of sliding ale Therefore, we introduce
    // an element offset AND a node offset for the the slave mortar elements
    if (true)  //(slidingale==false)
    {
      Teuchos::RCP<CONTACT::Element> cele = Teuchos::rcp(new CONTACT::Element(
          ele->Id(), ele->Owner(), ele->Shape(), ele->NumNode(), ele->NodeIds(), true, isnurbs));

      if (isnurbs)
      {
        Teuchos::RCP<DRT::NURBS::NurbsDiscretization> nurbsdis =
            Teuchos::rcp_dynamic_cast<DRT::NURBS::NurbsDiscretization>(slavedis);

        Teuchos::RCP<DRT::NURBS::Knotvector> knots = (*nurbsdis).GetKnotVector();
        std::vector<CORE::LINALG::SerialDenseVector> parentknots(dim);
        std::vector<CORE::LINALG::SerialDenseVector> mortarknots(dim - 1);

        Teuchos::RCP<DRT::FaceElement> faceele =
            Teuchos::rcp_dynamic_cast<DRT::FaceElement>(ele, true);
        double normalfac = 0.0;
        bool zero_size = knots->GetBoundaryEleAndParentKnots(parentknots, mortarknots, normalfac,
            faceele->ParentMasterElement()->Id(), faceele->FaceMasterNumber());

        // store nurbs specific data to node
        cele->ZeroSized() = zero_size;
        cele->Knots() = mortarknots;
        cele->NormalFac() = normalfac;
      }

      interface->AddElement(cele);
    }
    else
    {
      std::vector<int> nidsoff;
      for (int i = 0; i < ele->NumNode(); i++)
      {
        nidsoff.push_back(ele->NodeIds()[ele->NumNode() - 1 - i] + nodeoffset);
      }

      Teuchos::RCP<CONTACT::Element> cele = Teuchos::rcp(new CONTACT::Element(
          ele->Id() + eleoffset, ele->Owner(), ele->Shape(), ele->NumNode(), nidsoff.data(), true));

      interface->AddElement(cele);
    }
  }

  return;
}


/*----------------------------------------------------------------------*
 |  Initialize matrices                                      farah 02/16|
 *----------------------------------------------------------------------*/
void ADAPTER::CouplingNonLinMortar::InitMatrices()
{
  // safety check
  if (slavedofrowmap_ == Teuchos::null or slavenoderowmap_ == Teuchos::null)
    dserror("ERROR: Maps not initialized!");

  // init as standard sparse matrix --> local assembly
  D_ = Teuchos::rcp(new CORE::LINALG::SparseMatrix(*slavedofrowmap_, 81, false, false));
  M_ = Teuchos::rcp(new CORE::LINALG::SparseMatrix(*slavedofrowmap_, 81, false, false));
  H_ = Teuchos::rcp(new CORE::LINALG::SparseMatrix(*slavedofrowmap_, 81, false, false));
  T_ = Teuchos::rcp(new CORE::LINALG::SparseMatrix(*slavedofrowmap_, 81, false, false));
  N_ = Teuchos::rcp(new CORE::LINALG::SparseMatrix(*slavedofrowmap_, 81, false, false));

  gap_ = Teuchos::rcp(new Epetra_Vector(*slavenoderowmap_, true));

  // init as fe matrix --> nonlocal assembly
  DLin_ = Teuchos::rcp(new CORE::LINALG::SparseMatrix(
      *slavedofrowmap_, 81, true, false, CORE::LINALG::SparseMatrix::FE_MATRIX));
  MLin_ = Teuchos::rcp(new CORE::LINALG::SparseMatrix(
      *masterdofrowmap_, 81, true, false, CORE::LINALG::SparseMatrix::FE_MATRIX));

  // bye
  return;
}


/*----------------------------------------------------------------------*
 |  complete interface (also print and parallel redist.)     farah 02/16|
 *----------------------------------------------------------------------*/
void ADAPTER::CouplingNonLinMortar::CompleteInterface(
    Teuchos::RCP<DRT::Discretization> masterdis, Teuchos::RCP<CONTACT::Interface>& interface)
{
  const Teuchos::ParameterList& input =
      DRT::Problem::Instance()->MortarCouplingParams().sublist("PARALLEL REDISTRIBUTION");
  const INPAR::MORTAR::ParallelRedist parallelRedist =
      Teuchos::getIntegralValue<INPAR::MORTAR::ParallelRedist>(input, "PARALLEL_REDIST");

  /* Finalize the interface construction
   *
   * If this is the final parallel distribution, we need to assign degrees of freedom during
   * during FillComplete(). If parallel redistribution is enabled, there will be another call to
   * FillComplete(), so we skip this expensive operation here and do it later. DOFs have to be
   * assigned only once!
   */
  {
    bool isFinalDistribution = false;
    if (parallelRedist == INPAR::MORTAR::ParallelRedist::redist_none || comm_->NumProc() == 1)
      isFinalDistribution = true;

    interface->FillComplete(isFinalDistribution);
  }

  // create binary search tree
  interface->CreateSearchTree();

  // store old row maps (before parallel redistribution)
  pslavedofrowmap_ = Teuchos::rcp(new Epetra_Map(*interface->SlaveRowDofs()));
  pmasterdofrowmap_ = Teuchos::rcp(new Epetra_Map(*interface->MasterRowDofs()));
  pslavenoderowmap_ = Teuchos::rcp(new Epetra_Map(*interface->SlaveRowNodes()));
  psmdofrowmap_ = CORE::LINALG::MergeMap(pslavedofrowmap_, pmasterdofrowmap_, false);

  // print parallel distribution
  interface->PrintParallelDistribution();

  //**********************************************************************
  // PARALLEL REDISTRIBUTION OF INTERFACE
  //**********************************************************************
  if (parallelRedist != INPAR::MORTAR::ParallelRedist::redist_none && comm_->NumProc() > 1)
  {
    // redistribute optimally among all procs
    interface->Redistribute();

    // call fill complete again
    interface->FillComplete(true);

    // re create binary search tree
    interface->CreateSearchTree();

    // print parallel distribution again
    interface->PrintParallelDistribution();
  }

  // store row maps (after parallel redistribution)
  slavedofrowmap_ = Teuchos::rcp(new Epetra_Map(*interface->SlaveRowDofs()));
  masterdofrowmap_ = Teuchos::rcp(new Epetra_Map(*interface->MasterRowDofs()));
  slavenoderowmap_ = Teuchos::rcp(new Epetra_Map(*interface->SlaveRowNodes()));
  smdofrowmap_ = CORE::LINALG::MergeMap(slavedofrowmap_, masterdofrowmap_, false);

  // store interface
  interface_ = interface;

  return;
}


/*----------------------------------------------------------------------*
 | setup contact elements for spring dashpot condition     pfaller Apr15|
 *----------------------------------------------------------------------*/
void ADAPTER::CouplingNonLinMortar::SetupSpringDashpot(Teuchos::RCP<DRT::Discretization> masterdis,
    Teuchos::RCP<DRT::Discretization> slavedis, Teuchos::RCP<DRT::Condition> spring,
    const int coupling_id, const Epetra_Comm& comm)
{
  if (comm.MyPID() == 0)
    std::cout << "Generating CONTACT interface for spring dashpot condition...\n" << std::endl;

  // initialize maps for row nodes
  std::map<int, DRT::Node*> slavenodes;
  std::map<int, DRT::Node*> masternodes;

  // initialize maps for column nodes
  std::map<int, DRT::Node*> slavegnodes;
  std::map<int, DRT::Node*> mastergnodes;

  // initialize maps for elements
  std::map<int, Teuchos::RCP<DRT::Element>> slaveelements;
  std::map<int, Teuchos::RCP<DRT::Element>> masterelements;

  // get the conditions for the current evaluation we use the SpringDashpot condition as a
  // substitute for the mortar slave surface
  std::vector<DRT::Condition*> conds_master(0);
  std::vector<DRT::Condition*> conds_slave(0);

  // Coupling condition is defined by "DESIGN SURF SPRING DASHPOT COUPLING CONDITIONS"
  std::vector<DRT::Condition*> coup_conds;
  slavedis->GetCondition("RobinSpringDashpotCoupling", coup_conds);

  // number of coupling conditions
  const int n_coup_conds = (int)coup_conds.size();
  if (!n_coup_conds)
    dserror("No section DESIGN SURF ROBIN SPRING DASHPOT COUPLING CONDITIONS found.");

  // slave surface = spring dashpot condition
  conds_slave.push_back(&(*spring));

  // find master surface: loop all coupling conditions
  for (int i = 0; i < n_coup_conds; i++)
  {
    // add one, since read in of COUPLING parameter in DESIGN SURF SPRING DASHPOT CONDITIONS
    // subtracts one
    if (coup_conds[i]->GetInt("coupling id") == (coupling_id + 1))
      conds_master.push_back(coup_conds[i]);
  }
  if (!conds_master.size()) dserror("Coupling ID not found.");

  DRT::UTILS::FindConditionObjects(*slavedis, slavenodes, slavegnodes, slaveelements, conds_slave);
  DRT::UTILS::FindConditionObjects(
      *masterdis, masternodes, mastergnodes, masterelements, conds_master);

  // get mortar coupling parameters
  Teuchos::ParameterList input;
  // set default values
  input.setParameters(DRT::Problem::Instance()->MortarCouplingParams());
  input.setParameters(DRT::Problem::Instance()->ContactDynamicParams());
  input.setParameters(DRT::Problem::Instance()->WearParams());
  input.set<int>("PROBTYPE", INPAR::CONTACT::other);

  // is this a nurbs problem?
  CORE::FE::ShapeFunctionType distype = DRT::Problem::Instance()->SpatialApproximationType();
  switch (distype)
  {
    case CORE::FE::ShapeFunctionType::nurbs:
    {
      // ***
      dserror("nurbs for fsi mortar not supported!");
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

  // get problem dimension (2D or 3D) and create (MORTAR::Interface)
  const int dim = DRT::Problem::Instance()->NDim();

  // generate contact interface
  Teuchos::RCP<CONTACT::Interface> interface =
      CONTACT::Interface::Create(0, comm, dim, input, false);

  // feeding nodes to the interface including ghosted nodes
  std::map<int, DRT::Node*>::const_iterator nodeiter;

  // feeding elements to the interface
  std::map<int, Teuchos::RCP<DRT::Element>>::const_iterator elemiter;

  // eleoffset is neccessary because slave and master elements are from different conditions
  const int eleoffset = masterdis->ElementRowMap()->MaxAllGID() + 1;

  // MASTER NODES
  // feeding master nodes to the interface including ghosted nodes
  for (nodeiter = mastergnodes.begin(); nodeiter != mastergnodes.end(); ++nodeiter)
  {
    DRT::Node* node = nodeiter->second;

    Teuchos::RCP<CONTACT::Node> mrtrnode = Teuchos::rcp(new CONTACT::FriNode(
        node->Id(), node->X(), node->Owner(), masterdis->Dof(node), false, false, false));

    interface->AddNode(mrtrnode);
  }

  // SLAVE NODES
  // feeding slave nodes to the interface including ghosted nodes
  for (nodeiter = slavegnodes.begin(); nodeiter != slavegnodes.end(); ++nodeiter)
  {
    DRT::Node* node = nodeiter->second;

    Teuchos::RCP<CONTACT::Node> mrtrnode = Teuchos::rcp(new CONTACT::FriNode(
        node->Id(), node->X(), node->Owner(), slavedis->Dof(node), true, true, false));

    interface->AddNode(mrtrnode);
  }

  // MASTER ELEMENTS
  // feeding master elements to the interface
  for (elemiter = masterelements.begin(); elemiter != masterelements.end(); ++elemiter)
  {
    Teuchos::RCP<DRT::Element> ele = elemiter->second;

    Teuchos::RCP<CONTACT::Element> mrtrele = Teuchos::rcp(new CONTACT::Element(
        ele->Id(), ele->Owner(), ele->Shape(), ele->NumNode(), ele->NodeIds(), false));

    interface->AddElement(mrtrele);
  }

  // SLAVE ELEMENTS
  // feeding slave elements to the interface
  for (elemiter = slaveelements.begin(); elemiter != slaveelements.end(); ++elemiter)
  {
    Teuchos::RCP<DRT::Element> ele = elemiter->second;

    Teuchos::RCP<CONTACT::Element> mrtrele = Teuchos::rcp(new CONTACT::Element(
        ele->Id() + eleoffset, ele->Owner(), ele->Shape(), ele->NumNode(), ele->NodeIds(), true));

    interface->AddElement(mrtrele);
  }

  /* Finalize the interface construction
   *
   * If this is the final parallel distribution, we need to assign degrees of freedom during
   * during FillComplete(). If parallel redistribution is enabled, there will be another call to
   * FillComplete(), so we skip this expensive operation here and do it later. DOFs have to be
   * assigned only once!
   */
  {
    bool isFinalDistribution = false;
    const Teuchos::ParameterList& input =
        DRT::Problem::Instance()->MortarCouplingParams().sublist("PARALLEL REDISTRIBUTION");
    if (Teuchos::getIntegralValue<INPAR::MORTAR::ParallelRedist>(input, "PARALLEL_REDIST") ==
            INPAR::MORTAR::ParallelRedist::redist_none or
        comm_->NumProc() == 1)
      isFinalDistribution = true;

    interface->FillComplete(isFinalDistribution);
  }

  // store old row maps (before parallel redistribution)
  slavedofrowmap_ = Teuchos::rcp(new Epetra_Map(*interface->SlaveRowDofs()));
  masterdofrowmap_ = Teuchos::rcp(new Epetra_Map(*interface->MasterRowDofs()));

  // store interface
  interface_ = interface;

  // create binary search tree
  interface_->CreateSearchTree();

  // interface displacement (=0) has to be merged from slave and master discretization
  Teuchos::RCP<Epetra_Map> dofrowmap =
      CORE::LINALG::MergeMap(masterdofrowmap_, slavedofrowmap_, false);
  Teuchos::RCP<Epetra_Vector> dispn = CORE::LINALG::CreateVector(*dofrowmap, true);

  // set displacement state in mortar interface
  interface_->SetState(MORTAR::state_new_displacement, *dispn);

  // in the following two steps MORTAR does all the work
  interface_->Initialize();

  // set setup flag
  issetup_ = true;

  return;
}


/*----------------------------------------------------------------------*
 |  print interface                                         farah 10/14|
 *----------------------------------------------------------------------*/
void ADAPTER::CouplingNonLinMortar::PrintInterface(std::ostream& os) { interface_->Print(os); }


/*----------------------------------------------------------------------*
 |  Integrate slave-side matrix + linearization (D matrix)   farah 10/14|
 *----------------------------------------------------------------------*/
void ADAPTER::CouplingNonLinMortar::IntegrateLinD(const std::string& statename,
    const Teuchos::RCP<Epetra_Vector> vec, const Teuchos::RCP<Epetra_Vector> veclm)
{
  // safety check
  CheckSetup();

  // init matrices
  InitMatrices();

  // set lagrange multiplier and displacement state
  interface_->SetState(MORTAR::String2StateType(statename), *vec);
  interface_->SetState(MORTAR::state_lagrange_multiplier, *veclm);

  // general interface init: data container etc...
  interface_->Initialize();
  interface_->SetElementAreas();

  // loop over all slave col elements and direct integration
  for (int j = 0; j < interface_->SlaveColElements()->NumMyElements(); ++j)
  {
    int gid = interface_->SlaveColElements()->GID(j);
    DRT::Element* ele = interface_->Discret().gElement(gid);
    if (!ele) dserror("ERROR: Cannot find ele with gid %", gid);
    CONTACT::Element* cele = dynamic_cast<CONTACT::Element*>(ele);

    Teuchos::RCP<CONTACT::Integrator> integrator =
        Teuchos::rcp(new CONTACT::Integrator(interface_->InterfaceParams(), cele->Shape(), *comm_));

    integrator->IntegrateD(*cele, *comm_, true);
  }

  // assemble routine
  interface_->AssembleD(*D_);
  interface_->AssembleLinD(*DLin_, false);

  // complete matrices
  D_->Complete();
  DLin_->Complete();

  // check for parallel redistribution
  bool parredist = false;
  const Teuchos::ParameterList& input =
      DRT::Problem::Instance()->MortarCouplingParams().sublist("PARALLEL REDISTRIBUTION");
  if (Teuchos::getIntegralValue<INPAR::MORTAR::ParallelRedist>(input, "PARALLEL_REDIST") !=
      INPAR::MORTAR::ParallelRedist::redist_none)
    parredist = true;

  // only for parallel redistribution case
  if (parredist)
  {
    if (pslavedofrowmap_ == Teuchos::null)
      dserror("ERROR: Dof maps based on initial parallel distribution are wrong!");

    // transform everything back to old distribution
    D_ = MORTAR::MatrixRowColTransform(D_, pslavedofrowmap_, pslavedofrowmap_);
    DLin_ = MORTAR::MatrixRowColTransform(DLin_, pslavedofrowmap_, pslavedofrowmap_);
  }

  return;
}


/*----------------------------------------------------------------------*
 |  Integrate mortar matrices + linearization (D/M matrix)   farah 01/16|
 *----------------------------------------------------------------------*/
void ADAPTER::CouplingNonLinMortar::IntegrateLinDM(const std::string& statename,
    const Teuchos::RCP<Epetra_Vector> vec, const Teuchos::RCP<Epetra_Vector> veclm)
{
  // safety check
  CheckSetup();

  // init matrices with redistributed maps
  InitMatrices();

  // set current lm and displ state
  interface_->SetState(MORTAR::String2StateType(statename), *vec);
  interface_->SetState(MORTAR::state_lagrange_multiplier, *veclm);

  // init internal data
  interface_->Initialize();
  interface_->SetElementAreas();

  // call interface evaluate (d,m,gap...)
  interface_->Evaluate();

  // assemble mortar matrices and lin.
  interface_->AssembleDM(*D_, *M_);
  interface_->AssembleLinDM(*DLin_, *MLin_);

  // complete
  D_->Complete();
  M_->Complete(*masterdofrowmap_, *slavedofrowmap_);
  DLin_->Complete(*smdofrowmap_, *slavedofrowmap_);
  MLin_->Complete(*smdofrowmap_, *masterdofrowmap_);

  // Dinv * M
  CreateP();

  // transform to initial parallel distrib.
  MatrixRowColTransform();

  // bye bye
  return;
}


/*----------------------------------------------------------------------*
 |  transform all matrices and vectors                       farah 02/16|
 *----------------------------------------------------------------------*/
void ADAPTER::CouplingNonLinMortar::MatrixRowColTransform()
{
  // call base function
  CORE::ADAPTER::CouplingMortar::MatrixRowColTransform();

  // safety check
  CheckSetup();

  // check for parallel redistribution
  bool parredist = false;
  const Teuchos::ParameterList& input =
      DRT::Problem::Instance()->MortarCouplingParams().sublist("PARALLEL REDISTRIBUTION");
  if (Teuchos::getIntegralValue<INPAR::MORTAR::ParallelRedist>(input, "PARALLEL_REDIST") !=
      INPAR::MORTAR::ParallelRedist::redist_none)
    parredist = true;

  // transform everything back to old distribution
  if (parredist)
  {
    if (pslavedofrowmap_ == Teuchos::null or pmasterdofrowmap_ == Teuchos::null or
        pslavenoderowmap_ == Teuchos::null or psmdofrowmap_ == Teuchos::null)
      dserror("ERROR: Dof maps based on initial parallel distribution are wrong!");

    if (DLin_ != Teuchos::null)
      DLin_ = MORTAR::MatrixRowColTransform(DLin_, pslavedofrowmap_, psmdofrowmap_);

    if (MLin_ != Teuchos::null)
      MLin_ = MORTAR::MatrixRowColTransform(MLin_, pmasterdofrowmap_, psmdofrowmap_);

    if (H_ != Teuchos::null)
      H_ = MORTAR::MatrixRowColTransform(H_, pslavedofrowmap_, pslavedofrowmap_);

    if (T_ != Teuchos::null)
      T_ = MORTAR::MatrixRowColTransform(T_, pslavedofrowmap_, pslavedofrowmap_);

    if (N_ != Teuchos::null)
      N_ = MORTAR::MatrixRowColTransform(N_, pslavedofrowmap_, psmdofrowmap_);

    // transform gap vector
    if (gap_ != Teuchos::null)
    {
      Teuchos::RCP<Epetra_Vector> pgap = CORE::LINALG::CreateVector(*pslavenoderowmap_, true);
      CORE::LINALG::Export(*gap_, *pgap);
      gap_ = pgap;
    }
  }  // end parredist

  return;
}


/*----------------------------------------------------------------------*
 |  Integrate mortar matrices + linearization (D/M matrix) + gap vector |
 |  + compute projection operator P                         wirtz 01/16 |
 *----------------------------------------------------------------------*/
void ADAPTER::CouplingNonLinMortar::IntegrateAll(const std::string& statename,
    const Teuchos::RCP<Epetra_Vector> vec, const Teuchos::RCP<Epetra_Vector> veclm)
{
  // safety check
  CheckSetup();

  // init matrices with redistributed maps
  InitMatrices();

  // set current lm and displ state
  interface_->SetState(MORTAR::String2StateType(statename), *vec);
  interface_->SetState(MORTAR::state_lagrange_multiplier, *veclm);

  // init internal data
  interface_->Initialize();
  interface_->SetElementAreas();

  // call interface evaluate (d,m,gap...)
  interface_->Evaluate();

  // assemble mortar matrices and lin.
  interface_->AssembleDM(*D_, *M_);
  interface_->AssembleLinDM(*DLin_, *MLin_);
  interface_->AssembleG(*gap_);

  // complete
  D_->Complete();
  M_->Complete(*masterdofrowmap_, *slavedofrowmap_);
  DLin_->Complete(*smdofrowmap_, *slavedofrowmap_);
  MLin_->Complete(*smdofrowmap_, *masterdofrowmap_);

  // Dinv * M
  CreateP();

  // transform to initial parallel distrib.
  MatrixRowColTransform();

  return;
}

/*----------------------------------------------------------------------*
 |  Evaluate all mortar matrices and vectors necessary for mesh sliding |
 |                                                          wirtz 02/16 |
 *----------------------------------------------------------------------*/
void ADAPTER::CouplingNonLinMortar::EvaluateSliding(const std::string& statename,
    const Teuchos::RCP<Epetra_Vector> vec, const Teuchos::RCP<Epetra_Vector> veclm)
{
  // safety check
  CheckSetup();

  // init matrices with redistributed maps
  InitMatrices();

  // set current lm and displ state
  interface_->SetState(MORTAR::String2StateType(statename), *vec);
  interface_->SetState(MORTAR::state_lagrange_multiplier, *veclm);

  // init internal data
  interface_->Initialize();
  interface_->SetElementAreas();

  interface_->BuildActiveSet(true);

  // call interface evaluate (d,m,gap...)
  interface_->Evaluate();

  // assemble mortar matrices and lin.
  interface_->AssembleDM(*D_, *M_);
  interface_->AssembleLinDM(*DLin_, *MLin_);
  interface_->AssembleTNderiv(H_, Teuchos::null);
  interface_->AssembleTN(T_, Teuchos::null);
  interface_->AssembleS(*N_);
  interface_->AssembleG(*gap_);

  // complete
  D_->Complete();
  M_->Complete(*masterdofrowmap_, *slavedofrowmap_);
  DLin_->Complete(*smdofrowmap_, *slavedofrowmap_);
  MLin_->Complete(*smdofrowmap_, *masterdofrowmap_);
  H_->Complete();
  T_->Complete();
  N_->Complete(*smdofrowmap_, *slavedofrowmap_);

  // Dinv * M
  CreateP();

  // transform to initial parallel distrib.
  MatrixRowColTransform();

  return;
}

/*----------------------------------------------------------------------*
 |  compute projection operator P                            wirtz 02/16|
 *----------------------------------------------------------------------*/
void ADAPTER::CouplingNonLinMortar::CreateP()
{
  // safety check
  CheckSetup();

  // check
  if (INPUT::IntegralValue<INPAR::MORTAR::ShapeFcn>(
          Interface()->InterfaceParams(), "LM_SHAPEFCN") != INPAR::MORTAR::shape_dual)
    dserror("ERROR: Creation of P operator only for dual shape functions!");

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
      std::cout << "WARNING: Diagonal entry of D matrix is skipped because it is less than 1e-12!!!"
                << std::endl;
      (*diag)[i] = 1.0;
    }
  }

  // scalar inversion of diagonal values
  err = diag->Reciprocal(*diag);
  if (err > 0) dserror("ERROR: Reciprocal: Zero diagonal entry!");

  // re-insert inverted diagonal into invd
  err = Dinv_->ReplaceDiagonalValues(*diag);
  if (err > 0) dserror("ERROR: ReplaceDiagonalValues failed!");

  // complete inverse D matrix
  Dinv_->Complete();

  // do the multiplication P = inv(D) * M
  P_ = CORE::LINALG::MLMultiply(*Dinv_, false, *M_, false, false, false, true);

  // complete the matrix
  P_->Complete(*masterdofrowmap_, *slavedofrowmap_);

  // bye
  return;
}

BACI_NAMESPACE_CLOSE
