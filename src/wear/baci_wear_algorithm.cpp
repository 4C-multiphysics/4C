/*----------------------------------------------------------------------*/
/*! \file

\brief Basis of all WEAR algorithms that perform a coupling between the
       structural field equation and ALE field equations

\level 2


*/
/*----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 | headers                                                  farah 11/13 |
 *----------------------------------------------------------------------*/
#include "baci_wear_algorithm.H"

#include "baci_adapter_ale.H"
#include "baci_adapter_ale_wear.H"
#include "baci_adapter_str_fsiwrapper.H"
#include "baci_adapter_str_structure.H"
#include "baci_contact_aug_interface.H"
#include "baci_contact_element.H"
#include "baci_contact_friction_node.H"
#include "baci_contact_lagrange_strategy_wear.H"
#include "baci_contact_meshtying_contact_bridge.H"
#include "baci_contact_strategy_factory.H"
#include "baci_contact_utils.H"
#include "baci_contact_wear_interface.H"
#include "baci_global_data.H"
#include "baci_inpar_ale.H"
#include "baci_inpar_wear.H"
#include "baci_mortar_manager_base.H"
#include "baci_nurbs_discret.H"
#include "baci_nurbs_discret_control_point.H"
#include "baci_nurbs_discret_knotvector.H"

BACI_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 | Constructor                                              farah 11/13 |
 *----------------------------------------------------------------------*/
WEAR::Algorithm::Algorithm(const Epetra_Comm& comm)
    : AlgorithmBase(comm, GLOBAL::Problem::Instance()->StructuralDynamicParams())

{
  /*--------------------------------------------------------------------*
   | first create structure then ale --> important for discretization   |
   | numbering and therefore for the post_ensight.cpp                   |
   *--------------------------------------------------------------------*/

  // create structure
  Teuchos::RCP<ADAPTER::StructureBaseAlgorithm> structure = Teuchos::rcp(
      new ADAPTER::StructureBaseAlgorithm(GLOBAL::Problem::Instance()->StructuralDynamicParams(),
          const_cast<Teuchos::ParameterList&>(
              GLOBAL::Problem::Instance()->StructuralDynamicParams()),
          GLOBAL::Problem::Instance()->GetDis("structure")));
  structure_ = Teuchos::rcp_dynamic_cast<ADAPTER::FSIStructureWrapper>(structure->StructureField());
  structure_->Setup();

  if (structure_ == Teuchos::null)
    dserror("cast from ADAPTER::Structure to ADAPTER::FSIStructureWrapper failed");

  // ask base algorithm for the ale time integrator
  Teuchos::RCP<ADAPTER::AleBaseAlgorithm> ale = Teuchos::rcp(
      new ADAPTER::AleBaseAlgorithm(GLOBAL::Problem::Instance()->StructuralDynamicParams(),
          GLOBAL::Problem::Instance()->GetDis("ale")));
  ale_ = Teuchos::rcp_dynamic_cast<ADAPTER::AleWearWrapper>(ale->AleField());
  if (ale_ == Teuchos::null) dserror("cast from ADAPTER::Ale to ADAPTER::AleFsiWrapper failed");

  // create empty operator
  ale_->CreateSystemMatrix();

  // contact/meshtying manager
  cmtman_ = StructureField()->MeshtyingContactBridge()->ContactManager();

  // copy interfaces for material configuration
  // stactic cast of mortar strategy to contact strategy
  MORTAR::StrategyBase& strategy = cmtman_->GetStrategy();
  WEAR::LagrangeStrategyWear& cstrategy = static_cast<WEAR::LagrangeStrategyWear&>(strategy);

  // get dimension
  dim_ = strategy.Dim();

  // get vector of contact interfaces
  interfaces_ = cstrategy.ContactInterfaces();

  // create contact interfaces for material conf.
  CreateMaterialInterface();

  // input
  CheckInput();
}



/*----------------------------------------------------------------------*
 | Check compatibility of input parameters                  farah 09/14 |
 *----------------------------------------------------------------------*/
void WEAR::Algorithm::CheckInput()
{
  //  Teuchos::ParameterList apara = GLOBAL::Problem::Instance()->AleDynamicParams();
  //
  //  INPAR::ALE::AleDynamic aletype =
  //      INPUT::IntegralValue<INPAR::ALE::AleDynamic>(apara, "ALE_TYPE");

  return;
}


/*----------------------------------------------------------------------*
 | Create interfaces for material conf.                     farah 09/14 |
 *----------------------------------------------------------------------*/
void WEAR::Algorithm::CreateMaterialInterface()
{
  MORTAR::StrategyBase& strategy = cmtman_->GetStrategy();
  WEAR::LagrangeStrategyWear& cstrategy = static_cast<WEAR::LagrangeStrategyWear&>(strategy);

  // create some local variables (later to be stored in strategy)
  int dim = GLOBAL::Problem::Instance()->NDim();
  if (dim != 2 && dim != 3) dserror("Contact problem must be 2D or 3D");
  Teuchos::ParameterList cparams = cstrategy.Params();

  // check for FillComplete of discretization
  if (!structure_->Discretization()->Filled()) dserror("Discretization is not fillcomplete");

  // let's check for contact boundary conditions in discret
  // and detect groups of matching conditions
  // for each group, create a contact interface and store it
  if (Comm().MyPID() == 0)
  {
    std::cout << "Building contact interface(s) for Mat. conf. ...............";
    fflush(stdout);
  }

  std::vector<DRT::Condition*> contactconditions(0);
  structure_->Discretization()->GetCondition("Contact", contactconditions);

  // there must be more than one contact condition
  // unless we have a self contact problem!
  if ((int)contactconditions.size() < 1) dserror("Not enough contact conditions in discretization");
  if ((int)contactconditions.size() == 1)
  {
    const std::string* side = contactconditions[0]->Get<std::string>("Side");
    if (*side != "Selfcontact") dserror("Not enough contact conditions in discretization");
  }

  // find all pairs of matching contact conditions
  // there is a maximum of (conditions / 2) groups
  std::vector<int> foundgroups(0);
  int numgroupsfound = 0;

  // maximum dof number in discretization
  // later we want to create NEW Lagrange multiplier degrees of
  // freedom, which of course must not overlap with displacement dofs
  int maxdof = structure_->Discretization()->DofRowMap()->MaxAllGID();

  // get input par.
  INPAR::CONTACT::SolvingStrategy stype =
      INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(cparams, "STRATEGY");
  INPAR::WEAR::WearLaw wlaw = INPUT::IntegralValue<INPAR::WEAR::WearLaw>(cparams, "WEARLAW");
  INPAR::CONTACT::ConstraintDirection constr_direction =
      INPUT::IntegralValue<INPAR::CONTACT::ConstraintDirection>(cparams, "CONSTRAINT_DIRECTIONS");

  bool friplus = false;
  if ((wlaw != INPAR::WEAR::wear_none) || (cparams.get<int>("PROBTYPE") == INPAR::CONTACT::tsi))
    friplus = true;

  bool isanyselfcontact = false;

  for (int i = 0; i < (int)contactconditions.size(); ++i)
  {
    // initialize vector for current group of conditions and temp condition
    std::vector<DRT::Condition*> currentgroup(0);
    DRT::Condition* tempcond = nullptr;

    // try to build contact group around this condition
    currentgroup.push_back(contactconditions[i]);
    const std::vector<int>* group1v = currentgroup[0]->Get<std::vector<int>>("Interface ID");
    if (!group1v) dserror("Contact Conditions does not have value 'Interface ID'");
    int groupid1 = (*group1v)[0];
    bool foundit = false;

    // only one surface per group is ok for self contact
    const std::string* side = contactconditions[i]->Get<std::string>("Side");
    if (*side == "Selfcontact") foundit = true;

    for (int j = 0; j < (int)contactconditions.size(); ++j)
    {
      if (j == i) continue;  // do not detect contactconditions[i] again
      tempcond = contactconditions[j];
      const std::vector<int>* group2v = tempcond->Get<std::vector<int>>("Interface ID");
      if (!group2v) dserror("Contact Conditions does not have value 'Interface ID'");
      int groupid2 = (*group2v)[0];
      if (groupid1 != groupid2) continue;  // not in the group
      foundit = true;                      // found a group entry
      currentgroup.push_back(tempcond);    // store it in currentgroup
    }

    // now we should have found a group of conds
    if (!foundit) dserror("Cannot find matching contact condition for id %d", groupid1);

    // see whether we found this group before
    bool foundbefore = false;
    for (int j = 0; j < numgroupsfound; ++j)
      if (groupid1 == foundgroups[j])
      {
        foundbefore = true;
        break;
      }

    // if we have processed this group before, do nothing
    if (foundbefore) continue;

    // we have not found this group before, process it
    foundgroups.push_back(groupid1);
    ++numgroupsfound;

    // find out which sides are Master and Slave
    std::vector<bool> isslave(0);
    std::vector<bool> isself(0);
    CONTACT::UTILS::GetMasterSlaveSideInfo(isslave, isself, currentgroup);
    for (const bool is : isself)
      if (is)
      {
        isanyselfcontact = true;
        break;
      }

    // find out which sides are initialized as Active
    std::vector<bool> isactive(currentgroup.size());
    bool Two_half_pass(false);
    bool Check_nonsmooth_selfcontactsurface(false);
    bool Searchele_AllProc(false);

    CONTACT::UTILS::GetInitializationInfo(Two_half_pass, Check_nonsmooth_selfcontactsurface,
        Searchele_AllProc, isactive, isslave, isself, currentgroup);

    // create interface local parameter list (copy)
    Teuchos::ParameterList icparams = cparams;

    // find out if interface-specific coefficients of friction are given
    INPAR::CONTACT::FrictionType fric =
        INPUT::IntegralValue<INPAR::CONTACT::FrictionType>(cparams, "FRICTION");
    if (fric == INPAR::CONTACT::friction_tresca || fric == INPAR::CONTACT::friction_coulomb)
    {
      // read interface COFs
      std::vector<double> frcoeff((int)currentgroup.size());
      for (int j = 0; j < (int)currentgroup.size(); ++j)
        frcoeff[j] = currentgroup[j]->GetDouble("FrCoeffOrBound");

      // check consistency of interface COFs
      for (int j = 1; j < (int)currentgroup.size(); ++j)
        if (frcoeff[j] != frcoeff[0])
          dserror("Inconsistency in friction coefficients of interface %i", groupid1);

      // check for infeasible value of COF
      if (frcoeff[0] < 0.0) dserror("Negative FrCoeff / FrBound on interface %i", groupid1);

      // add COF locally to contact parameter list of this interface
      if (fric == INPAR::CONTACT::friction_tresca)
      {
        icparams.setEntry("FRBOUND", static_cast<Teuchos::ParameterEntry>(frcoeff[0]));
        icparams.setEntry("FRCOEFF", static_cast<Teuchos::ParameterEntry>(-1.0));
      }
      else if (fric == INPAR::CONTACT::friction_coulomb)
      {
        icparams.setEntry("FRCOEFF", static_cast<Teuchos::ParameterEntry>(frcoeff[0]));
        icparams.setEntry("FRBOUND", static_cast<Teuchos::ParameterEntry>(-1.0));
      }
    }

    // find out if interface-specific coefficients of friction are given
    INPAR::CONTACT::AdhesionType ad =
        INPUT::IntegralValue<INPAR::CONTACT::AdhesionType>(cparams, "ADHESION");
    if (ad == INPAR::CONTACT::adhesion_bound)
    {
      // read interface COFs
      std::vector<double> ad_bound((int)currentgroup.size());
      for (int j = 0; j < (int)currentgroup.size(); ++j)
        ad_bound[j] = currentgroup[j]->GetDouble("AdhesionBound");

      // check consistency of interface COFs
      for (int j = 1; j < (int)currentgroup.size(); ++j)
        if (ad_bound[j] != ad_bound[0])
          dserror("Inconsistency in adhesion bounds of interface %i", groupid1);

      // check for infeasible value of COF
      if (ad_bound[0] < 0.0) dserror("Negative adhesion bound on interface %i", groupid1);

      // add COF locally to contact parameter list of this interface
      icparams.setEntry("ADHESION_BOUND", static_cast<Teuchos::ParameterEntry>(ad_bound[0]));
    }

    // add information to parameter list of this interface
    icparams.set<bool>("Two_half_pass", Two_half_pass);
    icparams.set<bool>("Check_nonsmooth_selfcontactsurface", Check_nonsmooth_selfcontactsurface);
    icparams.set<bool>("Searchele_AllProc", Searchele_AllProc);

    // for structural contact we currently choose redundant master storage
    // the only exception is self contact where a redundant slave is needed, too
    INPAR::MORTAR::ExtendGhosting redundant =
        Teuchos::getIntegralValue<INPAR::MORTAR::ExtendGhosting>(
            icparams.sublist("PARALLEL REDISTRIBUTION"), "GHOSTING_STRATEGY");
    if (isanyselfcontact == true && redundant != INPAR::MORTAR::ExtendGhosting::redundant_all)
      dserror("Self contact requires fully redundant slave and master storage");

    // decide between contactinterface, augmented interface and wearinterface
    Teuchos::RCP<CONTACT::Interface> newinterface = CONTACT::STRATEGY::Factory::CreateInterface(
        groupid1, Comm(), dim, icparams, isself[0], Teuchos::null);
    interfacesMat_.push_back(newinterface);

    // get it again
    Teuchos::RCP<CONTACT::Interface> interface = interfacesMat_[(int)interfacesMat_.size() - 1];

    // note that the nodal ids are unique because they come from
    // one global problem discretization containing all nodes of the
    // contact interface.
    // We rely on this fact, therefore it is not possible to
    // do contact between two distinct discretizations here.

    // collect all intial active nodes
    std::vector<int> initialactive;

    //-------------------------------------------------- process nodes
    for (int j = 0; j < (int)currentgroup.size(); ++j)
    {
      // get all nodes and add them
      const std::vector<int>* nodeids = currentgroup[j]->GetNodes();
      if (!nodeids) dserror("Condition does not have Node Ids");
      for (int k = 0; k < (int)(*nodeids).size(); ++k)
      {
        int gid = (*nodeids)[k];
        // do only nodes that I have in my discretization
        if (!structure_->Discretization()->NodeColMap()->MyGID(gid)) continue;
        DRT::Node* node = structure_->Discretization()->gNode(gid);
        if (!node) dserror("Cannot find node with gid %", gid);

        // store initial active node gids
        if (isactive[j]) initialactive.push_back(gid);

        // find out if this node is initial active on another Condition
        // and do NOT overwrite this status then!
        bool foundinitialactive = false;
        if (!isactive[j])
        {
          for (int k = 0; k < (int)initialactive.size(); ++k)
            if (gid == initialactive[k])
            {
              foundinitialactive = true;
              break;
            }
        }

        // create Node object or FriNode object in the frictional case
        INPAR::CONTACT::FrictionType ftype =
            INPUT::IntegralValue<INPAR::CONTACT::FrictionType>(cparams, "FRICTION");

        // for the boolean variable initactive we use isactive[j]+foundinitialactive,
        // as this is true for BOTH initial active nodes found for the first time
        // and found for the second, third, ... time!
        if (ftype != INPAR::CONTACT::friction_none)
        {
          Teuchos::RCP<CONTACT::FriNode> cnode = Teuchos::rcp(new CONTACT::FriNode(node->Id(),
              node->X(), node->Owner(), structure_->Discretization()->Dof(0, node), isslave[j],
              isactive[j] + foundinitialactive, friplus));
          //-------------------
          // get nurbs weight!
          if (cparams.get<bool>("NURBS") == true)
          {
            DRT::NURBS::ControlPoint* cp = dynamic_cast<DRT::NURBS::ControlPoint*>(node);

            cnode->NurbsW() = cp->W();
          }

          // Check, if this node (and, in case, which dofs) are in the contact symmetry condition
          std::vector<DRT::Condition*> contactSymconditions(0);
          structure_->Discretization()->GetCondition("mrtrsym", contactSymconditions);

          for (unsigned j = 0; j < contactSymconditions.size(); j++)
            if (contactSymconditions.at(j)->ContainsNode(node->Id()))
            {
              const std::vector<int>* onoff =
                  contactSymconditions.at(j)->Get<std::vector<int>>("onoff");
              for (unsigned k = 0; k < onoff->size(); k++)
                if (onoff->at(k) == 1) cnode->DbcDofs()[k] = true;
              if (stype == INPAR::CONTACT::solution_lagmult &&
                  constr_direction != INPAR::CONTACT::constr_xyz)
                dserror(
                    "Contact symmetry with Lagrange multiplier method"
                    " only with contact constraints in xyz direction.\n"
                    "Set CONSTRAINT_DIRECTIONS to xyz in CONTACT input section");
            }

          // note that we do not have to worry about double entries
          // as the AddNode function can deal with this case!
          // the only problem would have occured for the initial active nodes,
          // as their status could have been overwritten, but is prevented
          // by the "foundinitialactive" block above!
          interface->AddNode(cnode);
        }
        else
        {
          Teuchos::RCP<CONTACT::Node> cnode = Teuchos::rcp(new CONTACT::Node(node->Id(), node->X(),
              node->Owner(), structure_->Discretization()->Dof(0, node), isslave[j],
              isactive[j] + foundinitialactive));
          //-------------------
          // get nurbs weight!
          if (cparams.get<bool>("NURBS") == true)
          {
            DRT::NURBS::ControlPoint* cp = dynamic_cast<DRT::NURBS::ControlPoint*>(node);

            cnode->NurbsW() = cp->W();
          }

          // Check, if this node (and, in case, which dofs) are in the contact symmetry condition
          std::vector<DRT::Condition*> contactSymconditions(0);
          structure_->Discretization()->GetCondition("mrtrsym", contactSymconditions);

          for (unsigned j = 0; j < contactSymconditions.size(); j++)
            if (contactSymconditions.at(j)->ContainsNode(node->Id()))
            {
              const std::vector<int>* onoff =
                  contactSymconditions.at(j)->Get<std::vector<int>>("onoff");
              for (unsigned k = 0; k < onoff->size(); k++)
                if (onoff->at(k) == 1) cnode->DbcDofs()[k] = true;
            }

          // note that we do not have to worry about double entries
          // as the AddNode function can deal with this case!
          // the only problem would have occured for the initial active nodes,
          // as their status could have been overwritten, but is prevented
          // by the "foundinitialactive" block above!
          interface->AddNode(cnode);
        }
      }
    }

    //----------------------------------------------- process elements
    int ggsize = 0;
    for (int j = 0; j < (int)currentgroup.size(); ++j)
    {
      // get elements from condition j of current group
      std::map<int, Teuchos::RCP<DRT::Element>>& currele = currentgroup[j]->Geometry();

      // elements in a boundary condition have a unique id
      // but ids are not unique among 2 distinct conditions
      // due to the way elements in conditions are build.
      // We therefore have to give the second, third,... set of elements
      // different ids. ids do not have to be continuous, we just add a large
      // enough number ggsize to all elements of cond2, cond3,... so they are
      // different from those in cond1!!!
      // note that elements in ele1/ele2 already are in column (overlapping) map
      int lsize = (int)currele.size();
      int gsize = 0;
      Comm().SumAll(&lsize, &gsize, 1);

      std::map<int, Teuchos::RCP<DRT::Element>>::iterator fool;
      for (fool = currele.begin(); fool != currele.end(); ++fool)
      {
        Teuchos::RCP<DRT::Element> ele = fool->second;
        Teuchos::RCP<CONTACT::Element> cele =
            Teuchos::rcp(new CONTACT::Element(ele->Id() + ggsize, ele->Owner(), ele->Shape(),
                ele->NumNode(), ele->NodeIds(), isslave[j], cparams.get<bool>("NURBS")));

        //------------------------------------------------------------------
        // get knotvector, normal factor and zero-size information for nurbs
        if (cparams.get<bool>("NURBS") == true)
        {
          DRT::NURBS::NurbsDiscretization* nurbsdis =
              dynamic_cast<DRT::NURBS::NurbsDiscretization*>(&(*(structure_->Discretization())));

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
      }  // for (fool=ele1.start(); fool != ele1.end(); ++fool)

      ggsize += gsize;  // update global element counter
    }

    //-------------------- finalize the contact interface construction
    interface->FillComplete(maxdof);

  }  // for (int i=0; i<(int)contactconditions.size(); ++i)
  if (Comm().MyPID() == 0) std::cout << "done!" << std::endl;

  return;
}

BACI_NAMESPACE_CLOSE
