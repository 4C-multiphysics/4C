/*!----------------------------------------------------------------------
\file contact_manager.cpp

\brief BACI implementation of main class to control all contact

<pre>
Maintainer: Alexander Popp
            popp@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15238
</pre>

*-----------------------------------------------------------------------*/

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include "contact_manager.H"
#include "contact_interface.H"
#include "contact_node.H"
#include "contact_element.H"
#include "contact_lagrange_strategy.H"
#include "contact_wear_lagrange_strategy.H"
#include "contact_poro_lagrange_strategy.H"
#include "contact_wear_interface.H"
#include "contact_penalty_strategy.H"
#include "contact_defines.H"
#include "friction_node.H"

#include "../drt_contact_aug/contact_augmented_strategy.H"
#include "../drt_contact_aug/contact_augmented_interface.H"

#include "../drt_mortar/mortar_defines.H"
#include "../linalg/linalg_utils.H"
#include "../drt_lib/drt_globalproblem.H"

#include "../drt_inpar/inpar_contact.H"
#include "../drt_inpar/inpar_mortar.H"
#include "../drt_inpar/inpar_wear.H"
#include "../drt_inpar/drt_validparameters.H"

#include "../drt_io/io_control.H"
#include "../drt_io/io.H"

/*----------------------------------------------------------------------*
 |  ctor (public)                                             popp 03/08|
 *----------------------------------------------------------------------*/
CONTACT::CoManager::CoManager(
    DRT::Discretization& discret,
    double alphaf) :
    MORTAR::ManagerBase(),
    discret_(discret)
{
  // overwrite base class communicator
  comm_ = Teuchos::rcp(Discret().Comm().Clone());

  // welcome message

  // create some local variables (later to be stored in strategy)
  int dim = DRT::Problem::Instance()->NDim();
  if (dim != 2 && dim != 3)
    dserror("ERROR: Contact problem must be 2D or 3D");
  std::vector<Teuchos::RCP<CONTACT::CoInterface> > interfaces;
  Teuchos::ParameterList cparams;

  // read and check contact input parameters
  if (Comm().MyPID() == 0)
  {
    std::cout << "Checking contact input parameters...........";
    fflush(stdout);
  }
  ReadAndCheckInput(cparams);
  if (Comm().MyPID() == 0)
    std::cout << "done!" << std::endl;

  // check for FillComplete of discretization
  if (!Discret().Filled())
    dserror("Discretization is not fillcomplete");

  // let's check for contact boundary conditions in discret
  // and detect groups of matching conditions
  // for each group, create a contact interface and store it
  if (Comm().MyPID() == 0)
  {
    std::cout << "Building contact interface(s)...............";
    fflush(stdout);
  }

  //Vector that contains solid-to-solid and beam-to-solid contact pairs
  std::vector<DRT::Condition*> beamandsolidcontactconditions(0);
  Discret().GetCondition("Contact", beamandsolidcontactconditions);

  //Vector that solely contains solid-to-solid contact pairs
  std::vector<DRT::Condition*> contactconditions(0);

  //Sort out beam-to-solid contact pairs, since these are treated in the beam3contact framework
  for (int i = 0; i < (int) beamandsolidcontactconditions.size(); ++i)
  {
    if(*(beamandsolidcontactconditions[i]->Get<std::string>("Application"))!="Beamtosolidcontact")
      contactconditions.push_back(beamandsolidcontactconditions[i]);
  }

  // there must be more than one contact condition
  // unless we have a self contact problem!
  if ((int) contactconditions.size() < 1)
    dserror("ERROR: Not enough contact conditions in discretization");
  if ((int) contactconditions.size() == 1)
  {
    const std::string* side = contactconditions[0]->Get<std::string>("Side");
    if (*side != "Selfcontact")
      dserror("ERROR: Not enough contact conditions in discretization");
  }

  // find all pairs of matching contact conditions
  // there is a maximum of (conditions / 2) groups
  std::vector<int> foundgroups(0);
  int numgroupsfound = 0;

  // maximum dof number in discretization
  // later we want to create NEW Lagrange multiplier degrees of
  // freedom, which of course must not overlap with displacement dofs
  int maxdof = Discret().DofRowMap()->MaxAllGID();

  // get input par.
  INPAR::CONTACT::SolvingStrategy stype = DRT::INPUT::IntegralValue<
      INPAR::CONTACT::SolvingStrategy>(cparams, "STRATEGY");
  INPAR::WEAR::WearLaw wlaw = DRT::INPUT::IntegralValue<
      INPAR::WEAR::WearLaw>(cparams, "WEARLAW");
  INPAR::WEAR::WearType wtype = DRT::INPUT::IntegralValue<
      INPAR::WEAR::WearType>(cparams, "WEARTYPE");
  INPAR::CONTACT::ConstraintDirection constr_direction =
      DRT::INPUT::IntegralValue<INPAR::CONTACT::ConstraintDirection>(cparams,"CONSTRAINT_DIRECTIONS");
  INPAR::CONTACT::FrictionType ftype = DRT::INPUT::IntegralValue<
      INPAR::CONTACT::FrictionType>(cparams, "FRICTION");
  INPAR::CONTACT::AdhesionType ad = DRT::INPUT::IntegralValue<
      INPAR::CONTACT::AdhesionType>(cparams, "ADHESION");
  const bool nurbs = cparams.get<bool>("NURBS");

  bool friplus = false;
  if ((wlaw != INPAR::WEAR::wear_none)
      || (cparams.get<int>("PROBTYPE") == INPAR::CONTACT::tsi))
    friplus = true;

  for (int i = 0; i < (int) contactconditions.size(); ++i)
  {
    // initialize vector for current group of conditions and temp condition
    std::vector<DRT::Condition*> currentgroup(0);
    DRT::Condition* tempcond = NULL;

    // try to build contact group around this condition
    currentgroup.push_back(contactconditions[i]);
    const std::vector<int>* group1v = currentgroup[0]->Get<std::vector<int> >(
        "Interface ID");
    if (!group1v)
      dserror("ERROR: Contact Conditions does not have value 'Interface ID'");
    int groupid1 = (*group1v)[0];
    bool foundit = false;

    // only one surface per group is ok for self contact
    const std::string* side = contactconditions[i]->Get<std::string>("Side");
    if (*side == "Selfcontact")
      foundit = true;

    for (int j = 0; j < (int) contactconditions.size(); ++j)
    {
      if (j == i)
        continue; // do not detect contactconditions[i] again
      tempcond = contactconditions[j];
      const std::vector<int>* group2v = tempcond->Get<std::vector<int> >(
          "Interface ID");
      if (!group2v)
        dserror("ERROR: Contact Conditions does not have value 'Interface ID'");
      int groupid2 = (*group2v)[0];
      if (groupid1 != groupid2)
        continue; // not in the group
      foundit = true; // found a group entry
      currentgroup.push_back(tempcond); // store it in currentgroup
    }

    // now we should have found a group of conds
    if (!foundit)
      dserror("ERROR: Cannot find matching contact condition for id %d", groupid1);

    // see whether we found this group before
    bool foundbefore = false;
    for (int j = 0; j < numgroupsfound; ++j)
      if (groupid1 == foundgroups[j])
      {
        foundbefore = true;
        break;
      }

    // if we have processed this group before, do nothing
    if (foundbefore)
      continue;

    // we have not found this group before, process it
    foundgroups.push_back(groupid1);
    ++numgroupsfound;

    // find out which sides are Master and Slave
    bool hasslave = false;
    bool hasmaster = false;
    std::vector<const std::string*> sides((int) currentgroup.size());
    std::vector<bool> isslave((int) currentgroup.size());
    std::vector<bool> isself((int) currentgroup.size());

    for (int j = 0; j < (int) sides.size(); ++j)
    {
      sides[j] = currentgroup[j]->Get<std::string>("Side");
      if (*sides[j] == "Slave")
      {
        hasslave = true;
        isslave[j] = true;
        isself[j] = false;
      }
      else if (*sides[j] == "Master")
      {
        hasmaster = true;
        isslave[j] = false;
        isself[j] = false;
      }
      else if (*sides[j] == "Selfcontact")
      {
        hasmaster = true;
        hasslave = true;
        isslave[j] = false;
        isself[j] = true;
      }
      else
        dserror("ERROR: CoManager: Unknown contact side qualifier!");
    }

    if (!hasslave)
      dserror("ERROR: Slave side missing in contact condition group!");
    if (!hasmaster)
      dserror("ERROR: Master side missing in contact condition group!");

    // check for self contact group
    if (isself[0])
    {
      for (int j = 1; j < (int) isself.size(); ++j)
        if (!isself[j])
          dserror("ERROR: Inconsistent definition of self contact condition group!");
    }

    // find out which sides are initialized as Active
    std::vector<const std::string*> active((int) currentgroup.size());
    std::vector<bool> isactive((int) currentgroup.size());

    for (int j = 0; j < (int) sides.size(); ++j)
    {
      active[j] = currentgroup[j]->Get<std::string>("Initialization");
      if (*sides[j] == "Slave")
      {
        // slave sides may be initialized as "Active" or as "Inactive"
        if (*active[j] == "Active")
          isactive[j] = true;
        else if (*active[j] == "Inactive")
          isactive[j] = false;
        else
          dserror("ERROR: Unknown contact init qualifier!");
      }
      else if (*sides[j] == "Master")
      {
        // master sides must NOT be initialized as "Active" as this makes no sense
        if (*active[j] == "Active")
          dserror("ERROR: Master side cannot be active!");
        else if (*active[j] == "Inactive")
          isactive[j] = false;
        else
          dserror("ERROR: Unknown contact init qualifier!");
      }
      else if (*sides[j] == "Selfcontact")
      {
        // Selfcontact surfs must NOT be initialized as "Active" as this makes no sense
        if (*active[j] == "Active")
          dserror("ERROR: Selfcontact surface cannot be active!");
        else if (*active[j] == "Inactive")
          isactive[j] = false;
        else
          dserror("ERROR: Unknown contact init qualifier!");
      }
      else
        dserror("ERROR: CoManager: Unknown contact side qualifier!");
    }

    // create interface local parameter list (copy)
    Teuchos::ParameterList icparams = cparams;

    // find out if interface-specific coefficients of friction are given
    if (ftype == INPAR::CONTACT::friction_tresca ||
        ftype == INPAR::CONTACT::friction_coulomb)
    {
      // read interface COFs
      std::vector<double> frcoeff((int) currentgroup.size());
      for (int j = 0; j < (int) currentgroup.size(); ++j)
        frcoeff[j] = currentgroup[j]->GetDouble("FrCoeffOrBound");

      // check consistency of interface COFs
      for (int j = 1; j < (int) currentgroup.size(); ++j)
        if (frcoeff[j] != frcoeff[0])
          dserror(
              "ERROR: Inconsistency in friction coefficients of interface %i",
              groupid1);

      // check for infeasible value of COF
      if (frcoeff[0] < 0.0)
        dserror("ERROR: Negative FrCoeff / FrBound on interface %i", groupid1);

      // add COF locally to contact parameter list of this interface
      if (ftype == INPAR::CONTACT::friction_tresca)
      {
        icparams.setEntry("FRBOUND",
            static_cast<Teuchos::ParameterEntry>(frcoeff[0]));
        icparams.setEntry("FRCOEFF",
            static_cast<Teuchos::ParameterEntry>(-1.0));
      }
      else if (ftype == INPAR::CONTACT::friction_coulomb)
      {
        icparams.setEntry("FRCOEFF",
            static_cast<Teuchos::ParameterEntry>(frcoeff[0]));
        icparams.setEntry("FRBOUND",
            static_cast<Teuchos::ParameterEntry>(-1.0));
      }
    }

    // find out if interface-specific coefficients of adhesion are given
    if (ad == INPAR::CONTACT::adhesion_bound)
    {
      // read interface COFs
      std::vector<double> ad_bound((int) currentgroup.size());
      for (int j = 0; j < (int) currentgroup.size(); ++j)
        ad_bound[j] = currentgroup[j]->GetDouble("AdhesionBound");

      // check consistency of interface COFs
      for (int j = 1; j < (int) currentgroup.size(); ++j)
        if (ad_bound[j] != ad_bound[0])
          dserror(
              "ERROR: Inconsistency in adhesion bounds of interface %i",
              groupid1);

      // check for infeasible value of COF
      if (ad_bound[0] < 0.0)
        dserror("ERROR: Negative adhesion bound on interface %i", groupid1);

      // add COF locally to contact parameter list of this interface
      icparams.setEntry("ADHESION_BOUND",static_cast<Teuchos::ParameterEntry>(ad_bound[0]));
    }

    // create an empty interface and store it in this Manager
    // create an empty contact interface and store it in this Manager
    // (for structural contact we currently choose redundant master storage)
    // (the only exception is self contact where a redundant slave is needed, too)
    INPAR::MORTAR::RedundantStorage redundant = DRT::INPUT::IntegralValue<
        INPAR::MORTAR::RedundantStorage>(icparams, "REDUNDANT_STORAGE");
//    if (isself[0]==false && redundant != INPAR::MORTAR::redundant_master)
//      dserror("ERROR: CoManager: Contact requires redundant master storage");
    if (isself[0] == true && redundant != INPAR::MORTAR::redundant_all)
      dserror("ERROR: CoManager: Self contact requires redundant slave and master storage");

    // decide between contactinterface, augmented interface and wearinterface
    Teuchos::RCP<CONTACT::CoInterface> newinterface=Teuchos::null;
    if (stype==INPAR::CONTACT::solution_augmented)
      newinterface=Teuchos::rcp(new CONTACT::AugmentedInterface(groupid1,Comm(),dim,icparams,isself[0],redundant));
    else if(wlaw!=INPAR::WEAR::wear_none)
      newinterface=Teuchos::rcp(new CONTACT::WearInterface(groupid1,Comm(),dim,icparams,isself[0],redundant));
    else
      newinterface = Teuchos::rcp(new CONTACT::CoInterface(groupid1, Comm(), dim, icparams, isself[0],redundant));
    interfaces.push_back(newinterface);

    // get it again
    Teuchos::RCP<CONTACT::CoInterface> interface =
        interfaces[(int) interfaces.size() - 1];

    // note that the nodal ids are unique because they come from
    // one global problem discretization containing all nodes of the
    // contact interface.
    // We rely on this fact, therefore it is not possible to
    // do contact between two distinct discretizations here.

    // collect all intial active nodes
    std::vector<int> initialactive;

    //-------------------------------------------------- process nodes
    for (int j = 0; j < (int) currentgroup.size(); ++j)
    {
      // get all nodes and add them
      const std::vector<int>* nodeids = currentgroup[j]->Nodes();
      if (!nodeids)
        dserror("ERROR: Condition does not have Node Ids");
      for (int k = 0; k < (int) (*nodeids).size(); ++k)
      {
        int gid = (*nodeids)[k];
        // do only nodes that I have in my discretization
        if (!Discret().NodeColMap()->MyGID(gid))
          continue;
        DRT::Node* node = Discret().gNode(gid);
        if (!node)
          dserror("ERROR: Cannot find node with gid %", gid);

        // store initial active node gids
        if (isactive[j])
          initialactive.push_back(gid);

        // find out if this node is initial active on another Condition
        // and do NOT overwrite this status then!
        bool foundinitialactive = false;
        if (!isactive[j])
        {
          for (int k = 0; k < (int) initialactive.size(); ++k)
            if (gid == initialactive[k])
            {
              foundinitialactive = true;
              break;
            }
        }

        // create CoNode object or FriNode object in the frictional case
        // for the boolean variable initactive we use isactive[j]+foundinitialactive,
        // as this is true for BOTH initial active nodes found for the first time
        // and found for the second, third, ... time!
        if (ftype != INPAR::CONTACT::friction_none)
        {
          Teuchos::RCP<CONTACT::FriNode> cnode = Teuchos::rcp(
              new CONTACT::FriNode(
                  node->Id(),
                  node->X(),
                  node->Owner(),
                  Discret().NumDof(0, node),
                  Discret().Dof(0, node),
                  isslave[j],
                  isactive[j] + foundinitialactive,
                  friplus));
          //-------------------
          // get nurbs weight!
          if (nurbs)
          {
            MORTAR::ManagerBase::PrepareNURBSNode(
                node,
                cnode);
          }

          // Check, if this node (and, in case, which dofs) are in the contact symmetry condition
          std::vector<DRT::Condition*> contactSymconditions(0);
          Discret().GetCondition("mrtrsym",contactSymconditions);

          for (unsigned j=0; j<contactSymconditions.size(); j++)
          if (contactSymconditions.at(j)->ContainsNode(node->Id()))
          {
            const std::vector<int>* onoff = contactSymconditions.at(j)->Get<std::vector<int> >("onoff");
            for (unsigned k=0; k<onoff->size(); k++)
            if (onoff->at(k)==1)
            cnode->DbcDofs()[k]=true;
             if (stype==INPAR::CONTACT::solution_lagmult && constr_direction!=INPAR::CONTACT::constr_xyz)
               dserror("Contact symmetry with Lagrange multiplier method"
                   " only with contact constraints in xyz direction.\n"
                   "Set CONSTRAINT_DIRECTIONS to xyz in CONTACT input section");
          }

          // note that we do not have to worry about double entries
          // as the AddNode function can deal with this case!
          // the only problem would have occured for the initial active nodes,
          // as their status could have been overwritten, but is prevented
          // by the "foundinitialactive" block above!
          interface->AddCoNode(cnode);
        }
        else
        {
          Teuchos::RCP<CONTACT::CoNode> cnode = Teuchos::rcp(
              new CONTACT::CoNode(
                  node->Id(),
                  node->X(),
                  node->Owner(),
                  Discret().NumDof(0, node),
                  Discret().Dof(0, node),
                  isslave[j],
                  isactive[j] + foundinitialactive));
          //-------------------
          // get nurbs weight!
          if (nurbs)
          {
            MORTAR::ManagerBase::PrepareNURBSNode(
                node,
                cnode);
          }

          // Check, if this node (and, in case, which dofs) are in the contact symmetry condition
          std::vector<DRT::Condition*> contactSymconditions(0);
          Discret().GetCondition("mrtrsym", contactSymconditions);

          for (unsigned j = 0; j < contactSymconditions.size(); j++)
            if (contactSymconditions.at(j)->ContainsNode(node->Id()))
            {
              const std::vector<int>* onoff = contactSymconditions.at(j)->Get<
                  std::vector<int> >("onoff");
              for (unsigned k = 0; k < onoff->size(); k++)
                if (onoff->at(k) == 1)
                {
                  cnode->DbcDofs()[k] = true;
                  if (stype==INPAR::CONTACT::solution_lagmult && constr_direction!=INPAR::CONTACT::constr_xyz)
                    dserror("Contact symmetry with Lagrange multiplier method"
                        " only with contact constraints in xyz direction.\n"
                        "Set CONSTRAINT_DIRECTIONS to xyz in CONTACT input section");
                }
            }

          // note that we do not have to worry about double entries
          // as the AddNode function can deal with this case!
          // the only problem would have occured for the initial active nodes,
          // as their status could have been overwritten, but is prevented
          // by the "foundinitialactive" block above!
          interface->AddCoNode(cnode);
        }
      }
    }

    //----------------------------------------------- process elements
    int ggsize = 0;
    for (int j = 0; j < (int) currentgroup.size(); ++j)
    {
      // get elements from condition j of current group
      std::map<int, Teuchos::RCP<DRT::Element> >& currele =
          currentgroup[j]->Geometry();

      // elements in a boundary condition have a unique id
      // but ids are not unique among 2 distinct conditions
      // due to the way elements in conditions are build.
      // We therefore have to give the second, third,... set of elements
      // different ids. ids do not have to be continuous, we just add a large
      // enough number ggsize to all elements of cond2, cond3,... so they are
      // different from those in cond1!!!
      // note that elements in ele1/ele2 already are in column (overlapping) map
      int lsize = (int) currele.size();
      int gsize = 0;
      Comm().SumAll(&lsize, &gsize, 1);

      std::map<int, Teuchos::RCP<DRT::Element> >::iterator fool;
      for (fool = currele.begin(); fool != currele.end(); ++fool)
      {
        Teuchos::RCP<DRT::Element> ele = fool->second;
        Teuchos::RCP<CONTACT::CoElement> cele = Teuchos::rcp(
            new CONTACT::CoElement(
                ele->Id() + ggsize,
                ele->Owner(),
                ele->Shape(),
                ele->NumNode(),
                ele->NodeIds(),
                isslave[j],
                nurbs));

        //store information about parent for porous contact (required for calculation of deformation gradient!)
        if (cparams.get<int>("PROBTYPE")==INPAR::CONTACT::poro)
        {
          Teuchos::RCP<DRT::FaceElement> faceele = Teuchos::rcp_dynamic_cast<DRT::FaceElement>(ele,true);
          if (faceele == Teuchos::null) dserror("Cast to FaceElement failed!");
          cele->SetParentMasterElement(faceele->ParentElement(), faceele->FaceParentNumber());
        }

        //------------------------------------------------------------------
        // get knotvector, normal factor and zero-size information for nurbs
        if (nurbs)
        {
          MORTAR::ManagerBase::PrepareNURBSElement(
              discret,
              ele,
              cele,
              dim);
        }

        cele->IsHermite() = DRT::INPUT::IntegralValue<int>(cparams,"HERMITE_SMOOTHING");

        interface->AddCoElement(cele);
      } // for (fool=ele1.start(); fool != ele1.end(); ++fool)

      ggsize += gsize; // update global element counter
    }

    //-------------------- finalize the contact interface construction
    interface->FillComplete(maxdof);

  } // for (int i=0; i<(int)contactconditions.size(); ++i)
  if (Comm().MyPID() == 0)
    std::cout << "done!" << std::endl;

  //**********************************************************************
  // create the solver strategy object
  // and pass all necessary data to it
  if (Comm().MyPID() == 0)
  {
    std::cout << "Building contact strategy object............";
    fflush(stdout);
  }

  // create WearLagrangeStrategy for wear as non-distinct quantity
  if ( stype == INPAR::CONTACT::solution_lagmult &&
       wlaw  != INPAR::WEAR::wear_none &&
      (wtype == INPAR::WEAR::wear_intstate ||
       wtype == INPAR::WEAR::wear_primvar))
  {
    strategy_ = Teuchos::rcp(new WearLagrangeStrategy(
        Discret().DofRowMap(),
        Discret().NodeRowMap(),
        cparams,
        interfaces,
        dim,
        comm_,
        alphaf,
        maxdof));
  }
  else if (stype == INPAR::CONTACT::solution_lagmult)
  {
    if (cparams.get<int>("PROBTYPE")!=INPAR::CONTACT::poro)
    {
      strategy_ = Teuchos::rcp(new CoLagrangeStrategy(
          Discret().DofRowMap(),
          Discret().NodeRowMap(),
          cparams,
          interfaces,
          dim,
          comm_,
          alphaf,
          maxdof));
    }
    else
    {
      strategy_ = Teuchos::rcp(new PoroLagrangeStrategy(
          Discret().DofRowMap(),
          Discret().NodeRowMap(),
          cparams,
          interfaces,
          dim,
          comm_,
          alphaf,
          maxdof));
    }
  }
  else if (stype == INPAR::CONTACT::solution_penalty)
  {
    strategy_ = Teuchos::rcp(new CoPenaltyStrategy(
        Discret().DofRowMap(),
        Discret().NodeRowMap(),
        cparams,
        interfaces,
        dim, comm_,
        alphaf,
        maxdof));
  }
  else if (stype == INPAR::CONTACT::solution_uzawa)
  {
    strategy_ = Teuchos::rcp(new CoPenaltyStrategy(
        Discret().DofRowMap(),
        Discret().NodeRowMap(),
        cparams,
        interfaces,
        dim,
        comm_,
        alphaf,
        maxdof));
  }
  else if (stype == INPAR::CONTACT::solution_augmented)
  {
    strategy_ = Teuchos::rcp(new AugmentedLagrangeStrategy(
        Discret().DofRowMap(),
        Discret().NodeRowMap(),
        cparams,
        interfaces,
        dim,
        comm_,
        alphaf,
        maxdof));
  }
  else
  {
    dserror("ERROR: Unrecognized strategy");
  }

  if (Comm().MyPID() == 0)
    std::cout << "done!" << std::endl;
  //**********************************************************************

  // print friction information of interfaces
  if (Comm().MyPID() == 0)
  {
    for (int i = 0; i < (int) interfaces.size(); ++i)
    {
      double checkfrcoeff = 0.0;
      if (ftype == INPAR::CONTACT::friction_tresca)
      {
        checkfrcoeff = interfaces[i]->IParams().get<double>("FRBOUND");
        std::cout << std::endl << "Interface         " << i + 1 << std::endl;
        std::cout << "FrBound (Tresca)  " << checkfrcoeff << std::endl;
      }
      else if (ftype == INPAR::CONTACT::friction_coulomb)
      {
        checkfrcoeff = interfaces[i]->IParams().get<double>("FRCOEFF");
        std::cout << std::endl << "Interface         " << i + 1 << std::endl;
        std::cout << "FrCoeff (Coulomb) " << checkfrcoeff << std::endl;
      }
    }
  }

  // print initial parallel redistribution
  for (int i = 0; i < (int) interfaces.size(); ++i)
    interfaces[i]->PrintParallelDistribution(i + 1);

  // create binary search tree
  for (int i = 0; i < (int) interfaces.size(); ++i)
    interfaces[i]->CreateSearchTree();

  // show default parameters
  if (Comm().MyPID() == 0)
  {
    std::cout << std::endl;
    DRT::INPUT::PrintDefaultParameters(IO::cout, GetStrategy().Params());
  }

  return;
}


/*----------------------------------------------------------------------*
 |  read and check input parameters (public)                  popp 04/08|
 *----------------------------------------------------------------------*/
bool CONTACT::CoManager::ReadAndCheckInput(Teuchos::ParameterList& cparams)
{
  // read parameter lists from DRT::Problem
  const Teuchos::ParameterList& mortar   = DRT::Problem::Instance()->MortarCouplingParams();
  const Teuchos::ParameterList& contact  = DRT::Problem::Instance()->ContactDynamicParams();
  const Teuchos::ParameterList& wearlist = DRT::Problem::Instance()->WearParams();
  const Teuchos::ParameterList& tsic     = DRT::Problem::Instance()->TSIContactParams();
  const Teuchos::ParameterList& stru     = DRT::Problem::Instance()->StructuralDynamicParams();

  // read Problem Type and Problem Dimension from DRT::Problem
  const PROBLEM_TYP problemtype = DRT::Problem::Instance()->ProblemType();
  std::string distype = DRT::Problem::Instance()->SpatialApproximation();
  const int dim       = DRT::Problem::Instance()->NDim();

  // *********************************************************************
  // invalid parallel strategies
  // *********************************************************************
  if(DRT::INPUT::IntegralValue<INPAR::MORTAR::RedundantStorage>(mortar,"REDUNDANT_STORAGE") == INPAR::MORTAR::redundant_master and
     DRT::INPUT::IntegralValue<INPAR::MORTAR::ParallelStrategy>(mortar,"PARALLEL_STRATEGY") != INPAR::MORTAR::ghosting_redundant )
    dserror("ERROR: Redundant storage only reasonable in combination with parallel strategy: ghosting_redundant !");

  if(DRT::INPUT::IntegralValue<INPAR::MORTAR::RedundantStorage>(mortar,"REDUNDANT_STORAGE") == INPAR::MORTAR::redundant_all and
     DRT::INPUT::IntegralValue<INPAR::MORTAR::ParallelStrategy>(mortar,"PARALLEL_STRATEGY") != INPAR::MORTAR::ghosting_redundant )
    dserror("ERROR: Redundant storage only reasonable in combination with parallel strategy: ghosting_redundant !");

  if((DRT::INPUT::IntegralValue<INPAR::MORTAR::ParallelStrategy>(mortar,"PARALLEL_STRATEGY") == INPAR::MORTAR::binningstrategy or
      DRT::INPUT::IntegralValue<INPAR::MORTAR::ParallelStrategy>(mortar,"PARALLEL_STRATEGY") == INPAR::MORTAR::roundrobinevaluate or
      DRT::INPUT::IntegralValue<INPAR::MORTAR::ParallelStrategy>(mortar,"PARALLEL_STRATEGY") == INPAR::MORTAR::roundrobinghost) and
      DRT::INPUT::IntegralValue<INPAR::MORTAR::RedundantStorage>(mortar,"REDUNDANT_STORAGE") != INPAR::MORTAR::redundant_none)
    dserror("ERROR: Parallel strategies only for none-redundant ghosting!");

  if (DRT::INPUT::IntegralValue<INPAR::MORTAR::ParRedist>(mortar,"PARALLEL_REDIST") != INPAR::MORTAR::parredist_none
      && mortar.get<int>("MIN_ELEPROC") < 0)
    dserror("ERROR: Minimum number of elements per processor for parallel redistribution must be >= 0");

  if (DRT::INPUT::IntegralValue<INPAR::MORTAR::ParRedist>(mortar,"PARALLEL_REDIST") == INPAR::MORTAR::parredist_dynamic
      && mortar.get<double>("MAX_BALANCE") < 1.0)
    dserror("ERROR: Maximum allowed value of load balance for dynamic parallel redistribution must be >= 1.0");

  if (problemtype == prb_tsi
      && DRT::INPUT::IntegralValue<INPAR::MORTAR::ParRedist>(mortar,"PARALLEL_REDIST") != INPAR::MORTAR::parredist_none)
    dserror("ERROR: Parallel redistribution not yet implemented for TSI problems");

  // *********************************************************************
  // adhesive contact
  // *********************************************************************
  if (DRT::INPUT::IntegralValue<INPAR::CONTACT::AdhesionType>(contact,"ADHESION") != INPAR::CONTACT::adhesion_none
      and DRT::INPUT::IntegralValue<INPAR::WEAR::WearLaw>(wearlist, "WEARLAW") != INPAR::WEAR::wear_none)
    dserror("ERROR: Adhesion combined with wear not yet tested!");

  if (DRT::INPUT::IntegralValue<INPAR::CONTACT::AdhesionType>(contact,"ADHESION") != INPAR::CONTACT::adhesion_none
      and DRT::INPUT::IntegralValue<INPAR::CONTACT::FrictionType>(contact,"FRICTION") != INPAR::CONTACT::friction_none)
    dserror("ERROR: Adhesion combined with friction not yet tested!");

  // *********************************************************************
  // generally invalid combinations (nts/mortar)
  // *********************************************************************
  if (DRT::INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact,"STRATEGY") == INPAR::CONTACT::solution_penalty
      && contact.get<double>("PENALTYPARAM") <= 0.0)
    dserror("ERROR: Penalty parameter eps = 0, must be greater than 0");

  if (DRT::INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact,"STRATEGY") == INPAR::CONTACT::solution_penalty
      && DRT::INPUT::IntegralValue<INPAR::CONTACT::FrictionType>(contact,"FRICTION") != INPAR::CONTACT::friction_none
      && contact.get<double>("PENALTYPARAMTAN") <= 0.0)
    dserror("ERROR: Tangential penalty parameter eps = 0, must be greater than 0");

  if (DRT::INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact,"STRATEGY") == INPAR::CONTACT::solution_uzawa
      && contact.get<double>("PENALTYPARAM") <= 0.0)
    dserror("ERROR: Penalty parameter eps = 0, must be greater than 0");

  if (DRT::INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact,"STRATEGY") == INPAR::CONTACT::solution_uzawa
      && DRT::INPUT::IntegralValue<INPAR::CONTACT::FrictionType>(contact,"FRICTION") != INPAR::CONTACT::friction_none
      && contact.get<double>("PENALTYPARAMTAN") <= 0.0)
    dserror("ERROR: Tangential penalty parameter eps = 0, must be greater than 0");

  if (DRT::INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact,"STRATEGY") == INPAR::CONTACT::solution_uzawa
      && contact.get<int>("UZAWAMAXSTEPS") < 2)
    dserror("ERROR: Maximum number of Uzawa / Augmentation steps must be at least 2");

  if (DRT::INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact,"STRATEGY") == INPAR::CONTACT::solution_uzawa
      && contact.get<double>("UZAWACONSTRTOL") <= 0.0)
    dserror("ERROR: Constraint tolerance for Uzawa / Augmentation scheme must be greater than 0");

  if (DRT::INPUT::IntegralValue<INPAR::CONTACT::FrictionType>(contact,"FRICTION") != INPAR::CONTACT::friction_none
      && contact.get<double>("SEMI_SMOOTH_CT") == 0.0)
    dserror("ERROR: Parameter ct = 0, must be greater than 0 for frictional contact");

  if (DRT::INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact,"STRATEGY") == INPAR::CONTACT::solution_augmented &&
      contact.get<double>("SEMI_SMOOTH_CN") <= 0.0)
    dserror("Regularization parameter cn, must be greater than 0 for contact problems");

  if (DRT::INPUT::IntegralValue<INPAR::CONTACT::FrictionType>(contact,"FRICTION") == INPAR::CONTACT::friction_tresca && dim == 3)
    dserror("ERROR: 3D frictional contact with Tresca's law not yet implemented");

  if (DRT::INPUT::IntegralValue<INPAR::CONTACT::FrictionType>(contact,"FRICTION") != INPAR::CONTACT::friction_none
      && DRT::INPUT::IntegralValue<int>(contact, "SEMI_SMOOTH_NEWTON") != 1
      && dim == 3)
    dserror("ERROR: 3D frictional contact only implemented with Semi-smooth Newton");

  if (DRT::INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact,"STRATEGY") == INPAR::CONTACT::solution_augmented &&
      DRT::INPUT::IntegralValue<INPAR::CONTACT::FrictionType>(contact,"FRICTION") != INPAR::CONTACT::friction_none)
    dserror("ERROR: Frictional contact is for the augmented Lagrange formulation not yet implemented!");

  if (DRT::INPUT::IntegralValue<int>(mortar,"CROSSPOINTS") == true && dim == 3)
    dserror("ERROR: Crosspoints / edge node modification not yet implemented for 3D");

  if (DRT::INPUT::IntegralValue<INPAR::CONTACT::FrictionType>(contact,"FRICTION") == INPAR::CONTACT::friction_tresca
      && DRT::INPUT::IntegralValue<int>(contact, "FRLESS_FIRST") == true)
    dserror("ERROR: Frictionless first contact step with Tresca's law not yet implemented"); // hopefully coming soon, when Coulomb and Tresca are combined

  if (DRT::INPUT::IntegralValue<INPAR::CONTACT::Regularization>(contact,"CONTACT_REGULARIZATION") != INPAR::CONTACT::reg_none
      && DRT::INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact,"STRATEGY") != INPAR::CONTACT::solution_lagmult)
    dserror("ERROR: Regularized Contact just available for Dual Mortar Contact with Lagrangean Multiplier!");

  if (DRT::INPUT::IntegralValue<INPAR::CONTACT::Regularization>(contact,"CONTACT_REGULARIZATION") != INPAR::CONTACT::reg_none
      && DRT::INPUT::IntegralValue<INPAR::CONTACT::FrictionType>(contact,"FRICTION") != INPAR::CONTACT::friction_none)
    dserror("ERROR: Regularized Contact for contact with friction not implemented yet!");

  // *********************************************************************
  // warnings
  // *********************************************************************
  if (mortar.get<double>("SEARCH_PARAM") == 0.0 && Comm().MyPID() == 0)
    std::cout << ("Warning: Contact search called without inflation of bounding volumes\n") << std::endl;

  if (DRT::INPUT::IntegralValue<INPAR::WEAR::WearSide>(wearlist,"WEAR_SIDE") != INPAR::WEAR::wear_slave)
    std::cout << ("\n \n Warning: Contact with both-sided wear is still experimental !") << std::endl;


  // *********************************************************************
  //                       MORTAR-SPECIFIC CHECKS
  // *********************************************************************
  if(DRT::INPUT::IntegralValue<INPAR::MORTAR::AlgorithmType>(mortar,"ALGORITHM") == INPAR::MORTAR::algorithm_mortar)
  {
    // *********************************************************************
    // invalid parameter combinations
    // *********************************************************************
    if (DRT::INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact,"STRATEGY") != INPAR::CONTACT::solution_lagmult
        && DRT::INPUT::IntegralValue<INPAR::MORTAR::ShapeFcn>(mortar, "LM_SHAPEFCN")   == INPAR::MORTAR::shape_petrovgalerkin)
      dserror("Petrov-Galerkin approach for LM only with Lagrange multiplier strategy");

    if (DRT::INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact,"STRATEGY") == INPAR::CONTACT::solution_lagmult
        && DRT::INPUT::IntegralValue<INPAR::MORTAR::ShapeFcn>(mortar, "LM_SHAPEFCN")   == INPAR::MORTAR::shape_standard
        && DRT::INPUT::IntegralValue<INPAR::CONTACT::SystemType>(contact,
            "SYSTEM") == INPAR::CONTACT::system_condensed)
      dserror("Condensation of linear system only possible for dual Lagrange multipliers");

    if (DRT::INPUT::IntegralValue<int>(mortar, "LM_DUAL_CONSISTENT") == true
        && DRT::INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact, "STRATEGY") != INPAR::CONTACT::solution_lagmult
        && DRT::INPUT::IntegralValue<INPAR::MORTAR::ShapeFcn>(mortar, "LM_SHAPEFCN") != INPAR::MORTAR::shape_standard)
      dserror("ERROR: Consistent dual shape functions in boundary elements only for Lagrange multiplier strategy.");

    if (DRT::INPUT::IntegralValue<int>(mortar, "LM_DUAL_CONSISTENT") == true
        && DRT::INPUT::IntegralValue<INPAR::MORTAR::IntType>(mortar, "INTTYPE") == INPAR::MORTAR::inttype_elements
        && (DRT::INPUT::IntegralValue<INPAR::MORTAR::ShapeFcn>(mortar, "LM_SHAPEFCN") == INPAR::MORTAR::shape_dual))
      dserror( "ERROR: Consistent dual shape functions in boundary elements not for purely element-based integration.");

    if (DRT::INPUT::IntegralValue<int>(mortar, "LM_NODAL_SCALE") == true
        && DRT::INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact,"STRATEGY") != INPAR::CONTACT::solution_lagmult)
      dserror("ERROR: Nodal scaling of Lagrange multipliers only for Lagrange multiplier strategy.");

    if (DRT::INPUT::IntegralValue<int>(mortar, "LM_NODAL_SCALE") == true
        && DRT::INPUT::IntegralValue<INPAR::MORTAR::IntType>(mortar, "INTTYPE") == INPAR::MORTAR::inttype_elements)
      dserror("ERROR: Nodal scaling of Lagrange multipliers not for purely element-based integration.");

    if ((DRT::INPUT::IntegralValue<int>(contact, "MESH_ADAPTIVE_CN") == true
        || DRT::INPUT::IntegralValue<int>(contact, "MESH_ADAPTIVE_CT") == true)
        && DRT::INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact,"STRATEGY") != INPAR::CONTACT::solution_lagmult)
      dserror("ERROR: Mesh adaptive cn and ct only for LM contact");

    if ((DRT::INPUT::IntegralValue<int>(contact, "MESH_ADAPTIVE_CN") == true
        || DRT::INPUT::IntegralValue<int>(contact, "MESH_ADAPTIVE_CT") == true)
        && DRT::INPUT::IntegralValue<int>(contact, "SEMI_SMOOTH_NEWTON") != 1)
      dserror("ERROR: Mesh adaptive cn and ct only for semi-smooth Newton strategy");

    if (DRT::INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact,"STRATEGY") == INPAR::CONTACT::solution_augmented &&
        DRT::INPUT::IntegralValue<INPAR::MORTAR::ShapeFcn>(mortar,"LM_SHAPEFCN") == INPAR::MORTAR::shape_dual)
      dserror("ERROR: The augmented Lagrange formulation does not support dual shape functions.");

    // *********************************************************************
    // not (yet) implemented combinations
    // *********************************************************************

    if (DRT::INPUT::IntegralValue<int>(mortar, "CROSSPOINTS") == true
        && DRT::INPUT::IntegralValue<INPAR::MORTAR::LagMultQuad>(mortar,"LM_QUAD") == INPAR::MORTAR::lagmult_lin)
      dserror("ERROR: Crosspoints and linear LM interpolation for quadratic FE not yet compatible");

    // check for self contact
    std::vector<DRT::Condition*> coco(0);
    Discret().GetCondition("Mortar", coco);
    bool self = false;

    for (int k = 0; k < (int) coco.size(); ++k)
    {
      const std::string* side = coco[k]->Get<std::string>("Side");
      if (*side == "Selfcontact")
        self = true;
    }

    if (self == true
        && DRT::INPUT::IntegralValue<INPAR::MORTAR::ParRedist>(mortar, "PARALLEL_REDIST") != INPAR::MORTAR::parredist_none)
      dserror("ERROR: Self contact and parallel redistribution not yet compatible");

    if (DRT::INPUT::IntegralValue<int>(contact, "INITCONTACTBYGAP") == true
        && contact.get<double>("INITCONTACTGAPVALUE") == 0.0)
      dserror("ERROR: For initialization of init contact with gap, the INITCONTACTGAPVALUE is needed.");

    if (DRT::INPUT::IntegralValue<int>(mortar, "LM_DUAL_CONSISTENT") == true
        && DRT::INPUT::IntegralValue<INPAR::MORTAR::LagMultQuad>(mortar,"LM_QUAD") != INPAR::MORTAR::lagmult_undefined
        && distype!="Nurbs")
      dserror("ERROR: Consistent dual shape functions in boundary elements only for linear shape functions or NURBS.");

    if (DRT::INPUT::IntegralValue<INPAR::WEAR::WearLaw>(wearlist, "WEARLAW") != INPAR::WEAR::wear_none
        && DRT::INPUT::IntegralValue<int>(contact, "FRLESS_FIRST") == true)
      dserror("ERROR: Frictionless first contact step with wear not yet implemented");

    if ((DRT::INPUT::IntegralValue<int>(contact, "MESH_ADAPTIVE_CN") == true
        || DRT::INPUT::IntegralValue<int>(contact, "MESH_ADAPTIVE_CT") == true)
        && DRT::INPUT::IntegralValue<INPAR::MORTAR::LagMultQuad>(mortar,"LM_QUAD") != INPAR::MORTAR::lagmult_undefined
        && distype!="Nurbs")
      dserror("ERROR: Mesh adaptive cn and ct only for first order elements or NURBS.");

    if ((DRT::INPUT::IntegralValue<int>(contact, "MESH_ADAPTIVE_CN") == true
        || DRT::INPUT::IntegralValue<int>(contact, "MESH_ADAPTIVE_CT") == true)
        && DRT::INPUT::IntegralValue<INPAR::CONTACT::FrictionType>(contact,"FRICTION") == INPAR::CONTACT::friction_tresca)
      dserror("ERROR: Mesh adaptive cn and ct only for frictionless contact and Coulomb friction");

    if ((DRT::INPUT::IntegralValue<int>(contact, "MESH_ADAPTIVE_CN") == true
        || DRT::INPUT::IntegralValue<int>(contact, "MESH_ADAPTIVE_CT") == true)
        && DRT::INPUT::IntegralValue<INPAR::WEAR::WearLaw>(wearlist, "WEARLAW")!= INPAR::WEAR::wear_none)
      dserror("ERROR: Mesh adaptive cn and ct not yet implemented for wear");

    // *********************************************************************
    // thermal-structure-interaction contact
    // *********************************************************************
    if (problemtype == prb_tsi
        && DRT::INPUT::IntegralValue<INPAR::MORTAR::ShapeFcn>(mortar, "LM_SHAPEFCN")
            != INPAR::MORTAR::shape_standard
        && DRT::INPUT::IntegralValue<int>(tsic, "THERMOLAGMULT") == false)
      dserror("ERROR: Thermal contact without Lagrange Multipliers only for standard shape functions");

    if (problemtype == prb_tsi
        && DRT::INPUT::IntegralValue<INPAR::MORTAR::ShapeFcn>(mortar, "LM_SHAPEFCN") == INPAR::MORTAR::shape_standard
        && DRT::INPUT::IntegralValue<int>(tsic, "THERMOLAGMULT") == true)
      dserror("ERROR: Thermal contact with Lagrange Multipliers only for dual shape functions");

    // no nodal scaling in for thermal-structure-interaction
    if (problemtype == prb_tsi
        && DRT::INPUT::IntegralValue<int>(mortar, "LM_NODAL_SCALE") == true)
      dserror("ERROR: Nodal scaling not yet implemented for TSI problems");

    // *********************************************************************
    // contact with wear
    // *********************************************************************
    if (DRT::INPUT::IntegralValue<INPAR::WEAR::WearLaw>(wearlist, "WEARLAW") == INPAR::WEAR::wear_none &&
        wearlist.get<double>("WEARCOEFF") != 0.0)
      dserror("ERROR: Wear coefficient only necessary in the context of wear.");

    if(problemtype == prb_structure and
       DRT::INPUT::IntegralValue<INPAR::WEAR::WearLaw>(wearlist, "WEARLAW")  != INPAR::WEAR::wear_none and
       DRT::INPUT::IntegralValue<INPAR::WEAR::WearTimInt>(wearlist,"WEARTIMINT") != INPAR::WEAR::wear_expl)
      dserror("ERROR: Wear calculation for pure structure problems only with explicit internal state variable approach reasonable!");

    if (DRT::INPUT::IntegralValue<INPAR::CONTACT::FrictionType>(contact,"FRICTION") == INPAR::CONTACT::friction_none
        && DRT::INPUT::IntegralValue<INPAR::WEAR::WearLaw>(wearlist, "WEARLAW")  != INPAR::WEAR::wear_none)
      dserror("ERROR: Wear models only applicable to frictional contact.");

    if (DRT::INPUT::IntegralValue<INPAR::WEAR::WearLaw>(wearlist, "WEARLAW") != INPAR::WEAR::wear_none &&
        wearlist.get<double>("WEARCOEFF") <= 0.0)
      dserror("ERROR: No valid wear coefficient provided, must be equal or greater 0.0");

    if (DRT::INPUT::IntegralValue<INPAR::WEAR::WearLaw>(wearlist, "WEARLAW") != INPAR::WEAR::wear_none
        && DRT::INPUT::IntegralValue<int>(mortar, "LM_NODAL_SCALE") == true)
      dserror("ERROR: Combination of LM_NODAL_SCALE and WEAR not (yet) implemented.");

    if (DRT::INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact,"STRATEGY") != INPAR::CONTACT::solution_lagmult
        && DRT::INPUT::IntegralValue<INPAR::WEAR::WearLaw>(wearlist, "WEARLAW")     != INPAR::WEAR::wear_none)
      dserror("ERROR: Wear model only applicable in combination with Lagrange multiplier strategy.");

    if (DRT::INPUT::IntegralValue<INPAR::CONTACT::FrictionType>(contact,"FRICTION") == INPAR::CONTACT::friction_tresca
        && DRT::INPUT::IntegralValue<INPAR::WEAR::WearLaw>(wearlist, "WEARLAW")  != INPAR::WEAR::wear_none)
      dserror("ERROR: Wear only for Coulomb friction!");

    // *********************************************************************
    // 3D quadratic mortar (choice of interpolation and testing fcts.)
    // *********************************************************************
    if ((DRT::INPUT::IntegralValue<INPAR::MORTAR::LagMultQuad>(mortar,"LM_QUAD") == INPAR::MORTAR::lagmult_pwlin
      || DRT::INPUT::IntegralValue<INPAR::MORTAR::LagMultQuad>(mortar,"LM_QUAD") == INPAR::MORTAR::lagmult_lin)
      && DRT::INPUT::IntegralValue<INPAR::MORTAR::ShapeFcn>(mortar, "LM_SHAPEFCN") == INPAR::MORTAR::shape_dual)
      dserror("ERROR: Only quadratic approach (for LM) implemented for quadratic contact with DUAL shape fct.");

    // *********************************************************************
    // Smooth contact
    // *********************************************************************
    if (DRT::INPUT::IntegralValue<int>(mortar,"HERMITE_SMOOTHING") == true and Comm().NumProc()!=1)
      dserror("ERROR: Hermit smoothing only for serial problems. It requires general overlap of 2!");

    if (DRT::INPUT::IntegralValue<int>(mortar,"HERMITE_SMOOTHING") == true and dim==3)
      dserror("ERROR: Hermit smoothing only for 2D cases!");

    if (DRT::INPUT::IntegralValue<int>(mortar,"HERMITE_SMOOTHING") == true and
        DRT::INPUT::IntegralValue<INPAR::MORTAR::IntType>(mortar, "INTTYPE")!= INPAR::MORTAR::inttype_elements)
      dserror("ERROR: Hermit smoothing only for element-based integration!");

    if (DRT::INPUT::IntegralValue<int>(mortar,"HERMITE_SMOOTHING") == true and
        DRT::INPUT::IntegralValue<int>(contact, "MESH_ADAPTIVE_CN") == false)
      dserror("ERROR: Use hermit smoothing with MESH_ADAPTIVE_CN!!!");

    if (DRT::INPUT::IntegralValue<int>(mortar,"HERMITE_SMOOTHING") == true)
      std::cout <<"\n \n Warning: Hermite smoothing still experimental!" << std::endl;

    // *********************************************************************
    // poroelastic contact
    // *********************************************************************
    if ((problemtype==prb_poroelast || problemtype==prb_fpsi || problemtype==prb_fpsi_xfem) &&
        (DRT::INPUT::IntegralValue<INPAR::MORTAR::ShapeFcn>(mortar,"LM_SHAPEFCN") != INPAR::MORTAR::shape_dual &&
        DRT::INPUT::IntegralValue<INPAR::MORTAR::ShapeFcn>(mortar,"LM_SHAPEFCN") != INPAR::MORTAR::shape_petrovgalerkin))
      dserror("POROCONTACT: Only dual and petrovgalerkin shape functions implemented yet!");

    if ((problemtype==prb_poroelast || problemtype==prb_fpsi || problemtype==prb_fpsi_xfem) &&
        DRT::INPUT::IntegralValue<INPAR::MORTAR::ParRedist>(mortar,"PARALLEL_REDIST") != INPAR::MORTAR::parredist_none)
      dserror("POROCONTACT: Parallel Redistribution not implemented yet!"); //Since we use Pointers to Parent Elements, which are not copied to other procs!

    if ((problemtype==prb_poroelast || problemtype==prb_fpsi || problemtype==prb_fpsi_xfem) &&
        DRT::INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact,"STRATEGY") != INPAR::CONTACT::solution_lagmult)
      dserror("POROCONTACT: Use Lagrangean Strategy for poro contact!");

    if ((problemtype==prb_poroelast || problemtype==prb_fpsi || problemtype==prb_fpsi_xfem) &&
        DRT::INPUT::IntegralValue<INPAR::CONTACT::FrictionType>(contact,"FRICTION") != INPAR::CONTACT::friction_none)
      dserror("POROCONTACT: Friction for poro contact not implemented!");

    if ((problemtype==prb_poroelast || problemtype==prb_fpsi || problemtype==prb_fpsi_xfem) &&
        DRT::INPUT::IntegralValue<int>(mortar,"LM_NODAL_SCALE")==true)
      dserror("POROCONTACT: Nodal scaling not yet implemented for poro contact problems");

    if ((problemtype==prb_poroelast || problemtype==prb_fpsi || problemtype==prb_fpsi_xfem) &&
        DRT::INPUT::IntegralValue<INPAR::CONTACT::SystemType>(contact,"SYSTEM") != INPAR::CONTACT::system_condensed)
      dserror("POROCONTACT: System has to be condensed for poro contact!");

    if ((problemtype==prb_poroelast || problemtype==prb_fpsi || problemtype==prb_fpsi_xfem) && dim != 3)
    {
      const Teuchos::ParameterList& porodyn = DRT::Problem::Instance()->PoroelastDynamicParams();
      if (DRT::INPUT::IntegralValue<int>(porodyn,"CONTACTNOPEN"))
        dserror("POROCONTACT: PoroContact with no penetration just tested for 3d!");
    }

  #ifdef MORTARTRAFO
    dserror("MORTARTRAFO not yet implemented for contact, only for meshtying");
  #endif // #ifndef MORTARTRAFO
    // *********************************************************************
    // element-based vs. segment-based mortar integration
    // *********************************************************************
    INPAR::MORTAR::IntType inttype = DRT::INPUT::IntegralValue<INPAR::MORTAR::IntType>(mortar, "INTTYPE");

    if ( inttype == INPAR::MORTAR::inttype_elements
        && mortar.get<int>("NUMGP_PER_DIM") <= 0)
      dserror("ERROR: Invalid Gauss point number NUMGP_PER_DIM for element-based integration.");

    if ( inttype == INPAR::MORTAR::inttype_elements_BS
        && mortar.get<int>("NUMGP_PER_DIM") <= 0)
      dserror("ERROR: Invalid Gauss point number NUMGP_PER_DIM for element-based integration with boundary segmentation."
              "\nPlease note that the value you have to provide only applies to the element-based integration"
              "\ndomain, while pre-defined default values will be used in the segment-based boundary domain.");

    if ((problemtype!=prb_tfsi_aero &&
        (inttype == INPAR::MORTAR::inttype_elements || inttype == INPAR::MORTAR::inttype_elements_BS) &&
        mortar.get<int>("NUMGP_PER_DIM") <= 1))
      dserror("ERROR: Invalid Gauss point number NUMGP_PER_DIM for element-based integration.");
  } // END MORTAR CHECKS

  // *********************************************************************
  //                       NTS-SPECIFIC CHECKS
  // *********************************************************************
  else if(DRT::INPUT::IntegralValue<INPAR::MORTAR::AlgorithmType>(mortar,"ALGORITHM") == INPAR::MORTAR::algorithm_nts)
  {
    if(DRT::INPUT::IntegralValue<int>(mortar,"HERMITE_SMOOTHING") == true)
      dserror("ERROR: Hermite smoothing only for mortar contact!");

    if(problemtype==prb_poroelast or problemtype==prb_fpsi or problemtype == prb_tsi)
      dserror("ERROR: NTS only for problem type: structure");
  } // END NTS CHECKS

  // *********************************************************************
  // store contents of BOTH ParameterLists in local parameter list
  // *********************************************************************
  cparams.setParameters(mortar);
  cparams.setParameters(contact);
  cparams.setParameters(wearlist);
  cparams.setParameters(tsic);
  cparams.set<double>("TIMESTEP", stru.get<double>("TIMESTEP"));

  // geometrically decoupled elements cannot be given via input file
  cparams.set<bool>("GEO_DECOUPLED", false);

  // *********************************************************************
  // NURBS contact
  // *********************************************************************
  if (distype == "Nurbs") cparams.set<bool>("NURBS", true);
  else                    cparams.set<bool>("NURBS", false);

  // *********************************************************************
  cparams.setName("CONTACT DYNAMIC / MORTAR COUPLING");

  // store relevant problem types
  if (problemtype == prb_structure or problemtype == prb_statmech)
  {
    cparams.set<int>("PROBTYPE", INPAR::CONTACT::structure);
  }
  else if (problemtype == prb_tsi)
  {
    cparams.set<int>("PROBTYPE", INPAR::CONTACT::tsi);
  }
  else if (problemtype == prb_struct_ale)
  {
    cparams.set<int>("PROBTYPE", INPAR::CONTACT::structalewear);
  }
  else if (problemtype == prb_poroelast or problemtype == prb_fpsi or problemtype == prb_fpsi_xfem)
  {
    cparams.set<int> ("PROBTYPE",INPAR::CONTACT::poro);
    //porotimefac = 1/(theta*dt) --- required for derivation of structural displacements!
    double porotimefac = 1/(stru.sublist("ONESTEPTHETA").get<double>("THETA") * stru.get<double>("TIMESTEP"));
    cparams.set<double> ("porotimefac", porotimefac);
  }
  else
  {
    cparams.set<int>("PROBTYPE", INPAR::CONTACT::other);
  }

  // no parallel redistribution in the serial case
  if (Comm().NumProc() == 1)
    cparams.set<std::string>("PARALLEL_REDIST", "None");

  return true;
}

/*----------------------------------------------------------------------*
 |  write restart information for contact (public)            popp 03/08|
 *----------------------------------------------------------------------*/
void CONTACT::CoManager::WriteRestart(IO::DiscretizationWriter& output,
    bool forcedrestart)
{
  // quantities to be written for restart
  Teuchos::RCP<Epetra_Vector> activetoggle;
  Teuchos::RCP<Epetra_Vector> sliptoggle;
  Teuchos::RCP<Epetra_Vector> weightedwear;
  Teuchos::RCP<Epetra_Vector> realwear;

  // quantities to be written for restart
  GetStrategy().DoWriteRestart(activetoggle, sliptoggle, weightedwear, realwear, forcedrestart);

  // export restart information for contact to problem dof row map
  Teuchos::RCP<Epetra_Map> problemdofs = GetStrategy().ProblemDofs();
  Teuchos::RCP<Epetra_Vector> lagrmultoldexp = Teuchos::rcp(new Epetra_Vector(*problemdofs));
  LINALG::Export(*(GetStrategy().LagrMultOld()), *lagrmultoldexp);
  Teuchos::RCP<Epetra_Vector> activetoggleexp = Teuchos::rcp(new Epetra_Vector(*problemdofs));
  LINALG::Export(*activetoggle, *activetoggleexp);

  // write restart information for contact
  output.WriteVector("lagrmultold", lagrmultoldexp);
  output.WriteVector("activetoggle", activetoggleexp);

  // friction
  if (GetStrategy().Friction())
  {
    Teuchos::RCP<Epetra_Vector> sliptoggleexp = Teuchos::rcp(new Epetra_Vector(*problemdofs));
    LINALG::Export(*sliptoggle, *sliptoggleexp);
    output.WriteVector("sliptoggle", sliptoggleexp);
  }

  // weighted wear
  if (weightedwear != Teuchos::null)
  {
    Teuchos::RCP<Epetra_Vector> weightedwearexp = Teuchos::rcp(new Epetra_Vector(*problemdofs));
    LINALG::Export(*weightedwear, *weightedwearexp);
    output.WriteVector("weightedwear", weightedwearexp);
  }

  // unweighted  wear
  if (realwear != Teuchos::null)
  {
    Teuchos::RCP<Epetra_Vector> realwearexp = Teuchos::rcp(new Epetra_Vector(*problemdofs));
    LINALG::Export(*realwear, *realwearexp);
    output.WriteVector("realwear", realwearexp);
  }

  return;
}

/*----------------------------------------------------------------------*
 |  read restart information for contact (public)             popp 03/08|
 *----------------------------------------------------------------------*/
void CONTACT::CoManager::ReadRestart(IO::DiscretizationReader& reader,
    Teuchos::RCP<Epetra_Vector> dis, Teuchos::RCP<Epetra_Vector> zero)
{
  // this is contact, thus we need the displacement state for restart
  // let strategy object do all the work
  GetStrategy().DoReadRestart(reader, dis);

  return;
}

/*----------------------------------------------------------------------*
 |  write interface tractions for postprocessing (public)     popp 03/08|
 *----------------------------------------------------------------------*/
void CONTACT::CoManager::PostprocessTractions(IO::DiscretizationWriter& output)
{
  // *********************************************************************
  // active contact set and slip set
  // *********************************************************************

  // evaluate active set and slip set
  Teuchos::RCP<Epetra_Vector> activeset = Teuchos::rcp( new Epetra_Vector(*GetStrategy().ActiveRowNodes()));
  activeset->PutScalar(1.0);
  if (GetStrategy().Friction())
  {
    Teuchos::RCP<Epetra_Vector> slipset = Teuchos::rcp(new Epetra_Vector(*GetStrategy().SlipRowNodes()));
    slipset->PutScalar(1.0);
    Teuchos::RCP<Epetra_Vector> slipsetexp = Teuchos::rcp(new Epetra_Vector(*GetStrategy().ActiveRowNodes()));
    LINALG::Export(*slipset, *slipsetexp);
    activeset->Update(1.0, *slipsetexp, 1.0);
  }

  // export to problem node row map
  Teuchos::RCP<Epetra_Map> problemnodes = GetStrategy().ProblemNodes();
  Teuchos::RCP<Epetra_Vector> activesetexp = Teuchos::rcp( new Epetra_Vector(*problemnodes));
  LINALG::Export(*activeset, *activesetexp);

  if (GetStrategy().WearBothDiscrete())
  {
    Teuchos::RCP<Epetra_Vector> mactiveset = Teuchos::rcp(new Epetra_Vector(*GetStrategy().MasterActiveNodes()));
    mactiveset->PutScalar(1.0);
    Teuchos::RCP<Epetra_Vector> slipset = Teuchos::rcp(new Epetra_Vector(*GetStrategy().MasterSlipNodes()));
    slipset->PutScalar(1.0);
    Teuchos::RCP<Epetra_Vector> slipsetexp = Teuchos::rcp( new Epetra_Vector(*GetStrategy().MasterActiveNodes()));
    LINALG::Export(*slipset, *slipsetexp);
    mactiveset->Update(1.0, *slipsetexp, 1.0);

    Teuchos::RCP<Epetra_Vector> mactivesetexp = Teuchos::rcp( new Epetra_Vector(*problemnodes));
    LINALG::Export(*mactiveset, *mactivesetexp);
    activesetexp->Update(1.0, *mactivesetexp, 1.0);
  }

  output.WriteVector("activeset", activesetexp);

  // *********************************************************************
  // contact tractions
  // *********************************************************************

  // evaluate contact tractions
  GetStrategy().OutputStresses();

  // export to problem dof row map
  Teuchos::RCP<Epetra_Map> problemdofs = GetStrategy().ProblemDofs();

  // normal direction
  Teuchos::RCP<Epetra_Vector> normalstresses    = GetStrategy().ContactNorStress();
  Teuchos::RCP<Epetra_Vector> normalstressesexp = Teuchos::rcp( new Epetra_Vector(*problemdofs));
  LINALG::Export(*normalstresses, *normalstressesexp);

  // tangential plane
  Teuchos::RCP<Epetra_Vector> tangentialstresses    = GetStrategy().ContactTanStress();
  Teuchos::RCP<Epetra_Vector> tangentialstressesexp = Teuchos::rcp(new Epetra_Vector(*problemdofs));
  LINALG::Export(*tangentialstresses, *tangentialstressesexp);

  // write to output
  // contact tractions in normal and tangential direction
  output.WriteVector("norcontactstress", normalstressesexp);
  output.WriteVector("tancontactstress", tangentialstressesexp);

#ifdef CONTACTFORCEOUTPUT

  // *********************************************************************
  // contact forces on slave non master side,
  // in normal and tangential direction
  // *********************************************************************
  // vectors for contact forces
  Teuchos::RCP<Epetra_Vector> fcslavenor = Teuchos::rcp(new Epetra_Vector(GetStrategy().DMatrix()->RowMap()));
  Teuchos::RCP<Epetra_Vector> fcslavetan = Teuchos::rcp(new Epetra_Vector(GetStrategy().DMatrix()->RowMap()));
  Teuchos::RCP<Epetra_Vector> fcmasternor = Teuchos::rcp(new Epetra_Vector(GetStrategy().MMatrix()->DomainMap()));
  Teuchos::RCP<Epetra_Vector> fcmastertan = Teuchos::rcp(new Epetra_Vector(GetStrategy().MMatrix()->DomainMap()));

  // vectors with problem dof row map
  Teuchos::RCP<Epetra_Vector> fcslavenorexp = Teuchos::rcp(new Epetra_Vector(*problemdofs));
  Teuchos::RCP<Epetra_Vector> fcslavetanexp = Teuchos::rcp(new Epetra_Vector(*problemdofs));
  Teuchos::RCP<Epetra_Vector> fcmasternorexp = Teuchos::rcp(new Epetra_Vector(*problemdofs));
  Teuchos::RCP<Epetra_Vector> fcmastertanexp = Teuchos::rcp(new Epetra_Vector(*problemdofs));

  // multiplication
  GetStrategy().DMatrix()->Multiply(true, *normalstresses, *fcslavenor);
  GetStrategy().DMatrix()->Multiply(true, *tangentialstresses, *fcslavetan);
  GetStrategy().MMatrix()->Multiply(true, *normalstresses, *fcmasternor);
  GetStrategy().MMatrix()->Multiply(true, *tangentialstresses, *fcmastertan);

#ifdef MASTERNODESINCONTACT
  //BEGIN: to output the global ID's of the master nodes in contact - devaal 02.2011

  int dim = DRT::Problem::Instance()->NDim();

  if (dim == 2)
  dserror("Only working for 3D");

  std::vector<int> lnid, gnid;

  //std::cout << "MasterNor" << fcmasternor->MyLength() << std::endl;

  for (int i=0; i<fcmasternor->MyLength(); i=i+3)
  {

    //check if master node in contact
    if (sqrt(((*fcmasternor)[i])*((*fcmasternor)[i])+((*fcmasternor)[i+1])*((*fcmasternor)[i+1])+((*fcmasternor)[i+2])*((*fcmasternor)[i]+2)) > 0.00001)
    {
      lnid.push_back((fcmasternor->Map()).GID(i)/3);
    }
  }

  // we want to gather data from on all procs
  std::vector<int> allproc(Comm().NumProc());
  for (int i=0; i<Comm().NumProc(); ++i) allproc[i] = i;

  // communicate all data to proc 0
  LINALG::Gather<int>(lnid,gnid,(int)allproc.size(),&allproc[0],Comm());

  //std::cout << " size of gnid:" << gnid.size() << std::endl;

  ////////////////
  ///// attempt at obtaining the nid and relative displacement u of master nodes in contact - devaal
  // define my own interface
  MORTAR::StrategyBase& myStrategy = GetStrategy();
  CoAbstractStrategy& myContactStrategy = dynamic_cast<CoAbstractStrategy&>(myStrategy);

  std::vector<Teuchos::RCP<CONTACT::CoInterface> > myInterface = myContactStrategy.ContactInterfaces();

  //check interface size - just doing this now for a single interface

  if (myInterface.size() != 1)
  dserror("Interface size should be 1");

  std::cout << "OUTPUT OF MASTER NODE IN CONTACT" << std::endl;
  //std::cout << "Master_node_in_contact x_dis y_dis z_dis" << std::endl;
  for (int i=0; i<(int)gnid.size(); ++i)
  {
    int myGid = gnid[i];
    std::cout << gnid[i] << std::endl; // << " " << myUx << " " << myUy << " " << myUz << std::endl;
  }

#endif  //MASTERNODESINCONTACT: to output the global ID's of the master nodes in contact
  // export
  LINALG::Export(*fcslavenor,*fcslavenorexp);
  LINALG::Export(*fcslavetan,*fcslavetanexp);
  LINALG::Export(*fcmasternor,*fcmasternorexp);
  LINALG::Export(*fcmastertan,*fcmastertanexp);

  // contact forces on slave and master side
  output.WriteVector("norslaveforce",fcslavenorexp);
  output.WriteVector("tanslaveforce",fcslavetanexp);
  output.WriteVector("normasterforce",fcmasternorexp);
  output.WriteVector("tanmasterforce",fcmastertanexp);

#ifdef CONTACTEXPORT
  // export averaged node forces to xxx.force
  double resultnor[fcslavenor->NumVectors()];
  double resulttan[fcslavetan->NumVectors()];
  fcslavenor->Norm2(resultnor);
  fcslavetan->Norm2(resulttan);

  if(Comm().MyPID()==0)
  {
    std::cout << "resultnor= " << resultnor[0] << std::endl;
    std::cout << "resulttan= " << resulttan[0] << std::endl;

    FILE* MyFile = NULL;
    std::ostringstream filename;
    const std::string filebase = DRT::Problem::Instance()->OutputControlFile()->FileNameOnlyPrefix();
    filename << filebase << ".force";
    MyFile = fopen(filename.str().c_str(), "at+");
    if (MyFile)
    {
      //fprintf(MyFile,valuename.c_str());
      fprintf(MyFile, "%g\t",resultnor[0]);
      fprintf(MyFile, "%g\n",resulttan[0]);
      fclose(MyFile);
    }
    else
    dserror("ERROR: File for Output could not be opened.");
  }
#endif //CONTACTEXPORT
#endif //CONTACTFORCEOUTPUT

  // Evaluate the interface forces for the augmented Lagrange formulation
  if (DRT::INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(DRT::Problem::Instance()->ContactDynamicParams(),"STRATEGY")
      ==INPAR::CONTACT::solution_augmented)
  {
    Teuchos::RCP<Epetra_Vector> augfs_lm = Teuchos::rcp(new Epetra_Vector(*problemdofs));
    Teuchos::RCP<Epetra_Vector> augfs_g  = Teuchos::rcp(new Epetra_Vector(*problemdofs));
    Teuchos::RCP<Epetra_Vector> augfm_lm = Teuchos::rcp(new Epetra_Vector(*problemdofs));
    Teuchos::RCP<Epetra_Vector> augfm_g  = Teuchos::rcp(new Epetra_Vector(*problemdofs));

    // evaluate augmented contact forces
    GetStrategy().AugForces(*augfs_lm,*augfs_g,*augfm_lm,*augfm_g);

    // contact forces on slave and master side
    output.WriteVector("norslaveforcelm",augfs_lm);
    output.WriteVector("norslaveforceg" ,augfs_g);
    output.WriteVector("normasterforcelm",augfm_lm);
    output.WriteVector("normasterforceg" ,augfm_g);
  }

  // *********************************************************************
  // wear with internal state variable approach
  // *********************************************************************
  bool wwear = GetStrategy().WeightedWear();
  if (wwear)
  {
    // ***************************************************************************
    // we do not compute the non-weighted wear here. we just write    farah 06/13
    // the output. the non-weighted wear will be used as dirichlet-b.
    // for the ale problem. n.w.wear will be called in stru_ale_algorithm.cpp
    // and computed in GetStrategy().OutputWear();
    // ***************************************************************************

    // evaluate wear (not weighted)
    GetStrategy().OutputWear();

    // write output
    Teuchos::RCP<Epetra_Vector> wearoutput    = GetStrategy().ContactWear();
    Teuchos::RCP<Epetra_Vector> wearoutputexp = Teuchos::rcp( new Epetra_Vector(*problemdofs));
    LINALG::Export(*wearoutput, *wearoutputexp);
    output.WriteVector("wear", wearoutputexp);
    GetStrategy().ContactWear()->Scale(0.0);
  }

  // *********************************************************************
  // poro contact
  // *********************************************************************
  bool poro = GetStrategy().HasPoroNoPenetration();
  if (poro)
  {
    //output of poro no penetration lagrange multiplier!
    CONTACT::PoroLagrangeStrategy& costrategy = dynamic_cast<CONTACT::PoroLagrangeStrategy&>(GetStrategy());
    Teuchos::RCP<Epetra_Vector> lambdaout     = costrategy.LambdaNoPen();
    Teuchos::RCP<Epetra_Vector> lambdaoutexp  = Teuchos::rcp(new Epetra_Vector(*problemdofs));
    LINALG::Export(*lambdaout, *lambdaoutexp);
    output.WriteVector("poronopen_lambda",lambdaoutexp);
  }
  return;
}
