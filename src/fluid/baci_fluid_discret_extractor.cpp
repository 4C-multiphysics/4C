/*-----------------------------------------------------------*/
/*! \file

\brief creates a second discretization as part of the complete discretization for inflow generation


\level 1

*/
/*-----------------------------------------------------------*/

#include "baci_fluid_discret_extractor.hpp"

#include "baci_global_data.hpp"
#include "baci_io.hpp"
#include "baci_lib_discret_xwall.hpp"
#include "baci_lib_dofset_transparent.hpp"
#include "baci_lib_periodicbc.hpp"
#include "baci_rebalance.hpp"

BACI_NAMESPACE_OPEN



/*----------------------------------------------------------------------*
 | Constructor (public)                                  rasthofer 05/11|
 *----------------------------------------------------------------------*/
FLD::FluidDiscretExtractor::FluidDiscretExtractor(
    Teuchos::RCP<DRT::Discretization> actdis, const std::string& condition, bool yescondition)
    : parentdiscret_(actdis)
{
  // get condition, i.e., do we have nodes that belong to a separate section of the domain
  std::vector<DRT::Condition*> sepcond;
  parentdiscret_->GetCondition(condition, sepcond);

  std::vector<int> allcnd_sepcondnodeids;

  // yes, we have nodes belonging to a separate section
  if (sepcond.size() != 0)
  {
    if (parentdiscret_->Comm().MyPID() == 0)
    {
      printf("+----------------\n");
      printf("|\n");
      printf("| Generating a second discretization containing all elements of ");
      printf("the separate section\n");
      printf("|\n");
    }

    // generate an empty child discretization
    // this discretization will contain all elements and nodes that are contained
    // in the separate section of the problem
    // add your discretization name here!
    if (condition == "TurbulentInflowSection")
    {
      DRT::DiscretizationXWall* xwall = dynamic_cast<DRT::DiscretizationXWall*>(&*actdis);
      if (nullptr != xwall)
        childdiscret_ = Teuchos::rcp(new DRT::DiscretizationXWall(
            (std::string) "inflow", Teuchos::rcp(parentdiscret_->Comm().Clone())));
      else
        childdiscret_ = Teuchos::rcp(new DRT::Discretization(
            (std::string) "inflow", Teuchos::rcp(parentdiscret_->Comm().Clone())));
    }
    else  // dummy discretization
      childdiscret_ = Teuchos::rcp(new DRT::Discretization(
          (std::string) "none", Teuchos::rcp(parentdiscret_->Comm().Clone())));

    // get set of ids of all child nodes
    std::set<int> sepcondnodeset;
    {
      // loop all separation conditions
      // there should be only one condition for turbulent flow problems
      // i.e., one volume in the input section FLUID TURBULENT INFLOW VOLUME
      //       or in the input section
      if ((sepcond.size() != 1) and (condition == "TurbulentInflowSection"))
        dserror("Only one separate section with condition TurbulentInflowSection expected!");
      // remark: however, more than one are already considered
      for (unsigned numcond = 0; numcond < sepcond.size(); ++numcond)
      {
        // get nodes ids of all nodes with separtion condition
        const std::vector<int>* sepcondnodeids = (*sepcond[numcond]).GetNodes();

        // and store them
        allcnd_sepcondnodeids.reserve(allcnd_sepcondnodeids.size() + sepcondnodeids->size());
        allcnd_sepcondnodeids.insert(
            allcnd_sepcondnodeids.end(), sepcondnodeids->begin(), sepcondnodeids->end());
      }

      // and change format
      for (std::vector<int>::iterator id = allcnd_sepcondnodeids.begin();
           id != allcnd_sepcondnodeids.end(); ++id)
      {
        sepcondnodeset.insert(*id);
      }
    }

    // determine sets of nodes which belong to separate section
    /*
     *  i.e., we are looking for elements which contain numelenodes with separtion condition
     *  as the two parts of the discretization, inflow section and problem domain, are separated
     *  this means:
     *
     *    *---------*         +---------+
     *    |         |         |         |
     *    |    1    |         |    3    |
     *    |         |         |         |
     *    *---------*         +---------+
     *    *: node with separation condition
     *    +: node without separtion condition
     *
     *    -> there should not be any elements containing numelenodes-1 or less nodes with separation
     * condition
     *
     *    *---------*---------+---------+
     *    |         |         |         |
     *    |    1    |    2    |    3    |
     *    |         |         |         |
     *    *---------*---------+---------+
     *
     *
     */
    std::set<int> sepcondelenodes_row;
    std::set<int> sepcondelenodes_col;

    // loop all column elements and label all row nodes of the separate section
    for (int i = 0; i < parentdiscret_->NumMyColElements(); ++i)
    {
      DRT::Element* actele = parentdiscret_->lColElement(i);

      // get the node ids of this element
      const int numnode = actele->NumNode();
      const int* nodeids = actele->NodeIds();

      bool found = false;

      // loop nodeids, check if a separation condition is active
      int counter = 0;
      for (int rr = 0; rr < numnode; ++rr)
      {
        int gid = nodeids[rr];

        std::set<int>::iterator curr = sepcondnodeset.find(gid);
        if (curr != sepcondnodeset.end())
        {
          counter++;
        }
      }

      // yes, we have a separation condition
      // element is part of the separate section
      if (counter == numnode)
        found = true;
      else if ((counter > 0) and (counter < numnode))
        dserror(
            "Turbulent inflow is a volume condition! All nodes of an element should have this "
            "condition!");

      if (found == true)
      {
        // loop nodeids
        for (int rr = 0; rr < numnode; ++rr)
        {
          int gid = nodeids[rr];

          if ((parentdiscret_->NodeRowMap())->LID(gid) > -1)
          {
            sepcondelenodes_row.insert(gid);
          }
          sepcondelenodes_col.insert(gid);
        }
      }
    }

    // all separation row nodes are now contained in the child discetization
    for (std::set<int>::iterator id = sepcondelenodes_row.begin(); id != sepcondelenodes_row.end();
         ++id)
    {
      DRT::Node* actnode = parentdiscret_->gNode(*id);

      Teuchos::RCP<DRT::Node> sepcondnode = Teuchos::rcp(actnode->Clone());

      childdiscret_->AddNode(sepcondnode);
    }

    // loop all row elements and add all elements with a separation node
    for (int i = 0; i < parentdiscret_->NumMyRowElements(); ++i)
    {
      DRT::Element* actele = parentdiscret_->lRowElement(i);

      // get the node ids of this element
      const int numnode = actele->NumNode();
      const int* nodeids = actele->NodeIds();

      bool found = false;

      // loop nodeids, check if a separation condition is active
      int counter = 0;
      for (int rr = 0; rr < numnode; ++rr)
      {
        int gid = nodeids[rr];

        std::set<int>::iterator curr = sepcondnodeset.find(gid);
        if (curr != sepcondnodeset.end())
        {
          counter++;
        }
      }
      // element is part of the separate section
      if (counter == numnode)
        found = true;
      else if ((counter > 0) and (counter < numnode))
        dserror(
            "Turbulent inflow is a volume condition! All nodes of an element should have this "
            "condition!");

      // yes, we have a turbulent separation condition (for this element)
      if (found == true)
      {
        Teuchos::RCP<DRT::Element> sepcondele = Teuchos::rcp(actele->Clone());

        childdiscret_->AddElement(sepcondele);
      }
    }

    // child discretization needs a full NodeRowMap and a NodeColMap
    Teuchos::RCP<Epetra_Map> newrownodemap;
    Teuchos::RCP<Epetra_Map> newcolnodemap;

    {
      std::vector<int> rownodes;

      // all row nodes with separation condition are now contained in the child discretization
      for (std::set<int>::iterator id = sepcondelenodes_row.begin();
           id != sepcondelenodes_row.end(); ++id)
      {
        rownodes.push_back(*id);
      }

      // build noderowmap for new distribution of nodes
      newrownodemap = Teuchos::rcp(
          new Epetra_Map(-1, rownodes.size(), rownodes.data(), 0, childdiscret_->Comm()));

      std::vector<int> colnodes;

      for (std::set<int>::iterator id = sepcondelenodes_col.begin();
           id != sepcondelenodes_col.end(); ++id)
      {
        colnodes.push_back(*id);
      }
      // build nodecolmap for new distribution of nodes
      newcolnodemap = Teuchos::rcp(
          new Epetra_Map(-1, colnodes.size(), colnodes.data(), 0, childdiscret_->Comm()));
    }

    if (childdiscret_->Comm().MyPID() == 0)
    {
      printf("| Distribute inflow discretization according to the initial nodemaps");
    }

    childdiscret_->Redistribute(*newrownodemap, *newcolnodemap, false, false, false);

    if (childdiscret_->Comm().MyPID() == 0)
    {
      std::cout << " ... done.\n";
    }

    // make all conditions known to the child discretization
    // i.e. periodic boundary conditions, dirichlet conditions, ...
    {
      if (childdiscret_->Comm().MyPID() == 0)
      {
        printf("| Inherit all boundary conditions");
      }

      // get all conditions types prescribed in the input file
      std::vector<std::string> allcond;
      parentdiscret_->GetConditionNames(allcond);
      // loop all conditions types
      for (unsigned numcond = 0; numcond < allcond.size(); ++numcond)
      {
        // get condition
        std::vector<DRT::Condition*> actcond;
        parentdiscret_->GetCondition(allcond[numcond], actcond);
        // loop all condition of the current type
        for (unsigned numactcond = 0; numactcond < actcond.size(); ++numactcond)
        {
          // we use the same nodal ids --- nevertheless, we just use a subset
          // of the node ids and thus cannot copy the conditions completely.
          std::vector<int> reduced_ids;

          // get all nodes of parent discretization having this condition
          const std::vector<int>* candidates = (*actcond[numactcond]).GetNodes();

          std::vector<int> mytoggle(candidates->size(), 0);
          std::vector<int> toggle(candidates->size(), 0);

          // loop all parent nodes with current condition
          // check if node is also contained in child discretization
          for (unsigned rr = 0; rr < candidates->size(); ++rr)
          {
            if (newrownodemap->LID((*candidates)[rr]) > -1)
            {
              mytoggle[rr] = 1;
            }
          }

          // combine marked nodes of all procs
          childdiscret_->Comm().SumAll(mytoggle.data(), toggle.data(), toggle.size());

          // and add nodes to the list of child nodes that will get the condition
          for (unsigned rr = 0; rr < candidates->size(); ++rr)
          {
            if (toggle[rr] > 0)
            {
              reduced_ids.push_back((*candidates)[rr]);
            }
          }

          // replace the nodes of the parent discretization by the nodes of the child discretization
          (*actcond[numactcond]).SetNodes(reduced_ids);

          // finally set condition
          childdiscret_->SetCondition(
              allcond[numcond], Teuchos::rcp(new DRT::Condition(*actcond[numactcond])));
        }

        // redistribute master and slave nodes
        // master and slave nodes are owned by one proc afterwards
        if (allcond[numcond] == "SurfacePeriodic")
        {
          PeriodicBoundaryConditions pbc(childdiscret_, false);
          pbc.UpdateDofsForPeriodicBoundaryConditions();
        }
      }

      if (childdiscret_->Comm().MyPID() == 0)
      {
        std::cout << " ... done.\n";
      }
    }

    if (childdiscret_->Comm().MyPID() == 0)
    {
      std::cout << "| Replace dofset by a transparent dofset that copies ";
      std::cout << "the dofs of the original";
      std::cout << " (parent) discretisation";
    }

    // idea: use a transparent dofset and hand through the dof numbering
    // get dof form parent discretization for child discretization
    childdiscret_->ReplaceDofSet(
        Teuchos::rcp(new DRT::TransparentDofSet(parentdiscret_, true)));  // true: parallel
    // and assign the dofs to nodes
    // remark: nothing is redistributed here
    childdiscret_->Redistribute(*newrownodemap, *newcolnodemap, true, true, true);

    if (childdiscret_->Comm().MyPID() == 0)
    {
      std::cout << " ... done.\n";
    }

    if (childdiscret_->Comm().MyPID() == 0)
    {
      std::cout << "| Call PARMETIS on the child discretization and ";
      std::cout << "redistribute according to";
      std::cout << " the new maps\n";
    }

    // this is the actual redistribution
    Teuchos::RCP<Epetra_Map> sepcondelenodesmap =
        Teuchos::rcp(new Epetra_Map(*childdiscret_->ElementRowMap()));
    Teuchos::Time time("", true);
    Teuchos::RCP<Epetra_Comm> comm = Teuchos::rcp(parentdiscret_->Comm().Clone());

    // Starting from the current partitioning of the discretization, compute nodal maps with a
    // hopefully better partitioning
    Teuchos::RCP<const Epetra_CrsGraph> sepcondnodemap =
        CORE::REBALANCE::BuildGraph(childdiscret_, sepcondelenodesmap);

    Teuchos::ParameterList rebalanceParams;
    rebalanceParams.set<std::string>("num parts", std::to_string(comm->NumProc()));

    const auto& [sepcondrownodes, sepcondcolnodes] =
        CORE::REBALANCE::RebalanceNodeMaps(sepcondnodemap, rebalanceParams);

    if (childdiscret_->Comm().MyPID() == 0)
    {
      std::cout << "| Redistributing .";
    }
    // redistribute accordingly to the adapted rowmap
    childdiscret_->Redistribute(*sepcondrownodes, *sepcondcolnodes, false, false);

    if (childdiscret_->Comm().MyPID() == 0)
    {
      std::cout << ".. done.\n";
    }

    // redistribute master and slave nodes
    // master and slave nodes are owned by one proc afterwards
    {
      if (childdiscret_->Comm().MyPID() == 0)
      {
        std::cout << "| Apply periodic boundary conditions to the redistributed";
        std::cout << " discretization and fetch slave nodes to the master's proc\n";
      }

      PeriodicBoundaryConditions pbc(childdiscret_, false);
      pbc.UpdateDofsForPeriodicBoundaryConditions();

      // get node to node coupling
      col_pbcmapmastertoslave_ = Teuchos::rcp(new std::map<int, std::vector<int>>());
      col_pbcmapmastertoslave_ = pbc.ReturnAllCoupledColNodes();
      row_pbcmapmastertoslave_ = Teuchos::rcp(new std::map<int, std::vector<int>>());
      row_pbcmapmastertoslave_ = pbc.ReturnAllCoupledRowNodes();
    }

    if (childdiscret_->Comm().MyPID() == 0)
    {
      std::cout << "| Assign the dofs for the redistributed layout, again using ";
      std::cout << "a parallel version";
      std::cout << " of the transparent dofset";
    }

    // idea: use a transparent dofset and hand through the dof numbering
    childdiscret_->ReplaceDofSet(Teuchos::rcp(new DRT::TransparentDofSet(parentdiscret_, true)));

    // set discretization writer
    childdiscret_->SetWriter(Teuchos::rcp(new IO::DiscretizationWriter(childdiscret_,
        GLOBAL::Problem::Instance()->OutputControlFile(),
        GLOBAL::Problem::Instance()->SpatialApproximationType())));

    // call FillComplete() to assign the dof
    // remark: equal Redistribute(*newrownodemap,*newcolnodemap,true,true,true) as
    //         it also calls FillComplete() at the end
    childdiscret_->FillComplete(true, true, true);

    if (childdiscret_->Comm().MyPID() == 0)
    {
      std::cout << " ... done.\n";
      printf("|\n");
      printf("+----------------\n\n");
    }

    // some output on the screen
    {
      const int numproc = parentdiscret_->Comm().NumProc();

      std::vector<int> my_n_nodes(numproc, 0);
      std::vector<int> n_nodes(numproc, 0);
      std::vector<int> my_n_elements(numproc, 0);
      std::vector<int> n_elements(numproc, 0);
      std::vector<int> my_n_ghostele(numproc, 0);
      std::vector<int> n_ghostele(numproc, 0);
      std::vector<int> my_n_dof(numproc, 0);
      std::vector<int> n_dof(numproc, 0);

      int myrank = childdiscret_->Comm().MyPID();

      my_n_nodes[myrank] = childdiscret_->NodeRowMap()->NumMyElements();
      my_n_elements[myrank] = childdiscret_->NumMyColElements();
      my_n_ghostele[myrank] = childdiscret_->NumMyColElements() - childdiscret_->NumMyRowElements();
      my_n_dof[myrank] = childdiscret_->DofRowMap()->NumMyElements();

      childdiscret_->Comm().SumAll(my_n_nodes.data(), n_nodes.data(), numproc);
      childdiscret_->Comm().SumAll(my_n_elements.data(), n_elements.data(), numproc);
      childdiscret_->Comm().SumAll(my_n_ghostele.data(), n_ghostele.data(), numproc);
      childdiscret_->Comm().SumAll(my_n_dof.data(), n_dof.data(), numproc);

      if (childdiscret_->Comm().MyPID() == 0)
      {
        printf("   +-----+---------------+-----------------+----------------+-----------------+\n");
        printf("   +                          child discretization                            +\n");
        printf("   +-----+---------------+-----------------+----------------+-----------------+\n");
        printf("   | PID |    n_nodes    |    n_elements   |   n_ghostele   |      n_dof      |\n");
        printf("   +-----+---------------+-----------------+----------------+-----------------+\n");
        for (int npid = 0; npid < numproc; ++npid)
        {
          printf("   | %3d | %13d | %15d | %14d | %15d |\n", npid, n_nodes[npid], n_elements[npid],
              n_ghostele[npid], n_dof[npid]);
          printf(
              "   +-----+---------------+-----------------+----------------+-----------------+\n");
        }
        std::cout << std::endl << std::endl;
      }
    }

    // The remaining part are just sanity checks for the redistributed discretisation
    {
      bool insane = false;

      // loop all column eles, check dofs for each node
      for (int i = 0; i < childdiscret_->NumMyColElements(); ++i)
      {
        DRT::Element* actele = childdiscret_->lColElement(i);

        // get the node ids of this element
        const int numnode = actele->NumNode();
        const int* nodeids = actele->NodeIds();

        // loop nodeids, check if a separation condition is active
        for (int rr = 0; rr < numnode; ++rr)
        {
          DRT::Node* node = childdiscret_->gNode(nodeids[rr]);
          std::vector<int> nodedofset = childdiscret_->Dof(node);

          for (unsigned index = 0; index < nodedofset.size(); ++index)
          {
            int gid = nodedofset[index];

            if (childdiscret_->DofColMap()->LID(gid) < 0)
            {
              insane = true;
              printf("myrank %d dof %d not in colmap\n", childdiscret_->Comm().MyPID(), gid);
            }
          }
        }
      }
      if (insane) dserror("invalid dof col map");

      {
        std::set<int> testset;
        for (int rr = 0; rr < childdiscret_->DofRowMap()->NumMyElements(); ++rr)
        {
          int id = childdiscret_->DofRowMap()->MyGlobalElements()[rr];

          std::set<int>::iterator curr = testset.find(id);
          if (curr != testset.end())
          {
            dserror("DofRowMap of child dis is not unique on this proc");
          }
          testset.insert(id);
        }

        if (!childdiscret_->DofRowMap()->UniqueGIDs())
        {
          std::cout << *childdiscret_->DofRowMap();

          dserror("DofRowMap  of child dis is not unique (global)");
        }
      }
    }
  }
  else
  {
    dserror("Nodes with separation condition expected!");
  }
}

BACI_NAMESPACE_CLOSE
