// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_discret_extractor.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_condition_periodic.hpp"
#include "4C_fem_dofset_transparent.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_rebalance_graph_based.hpp"
#include "4C_xfem_discretization_utils.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 | Constructor (public)                                  rasthofer 05/11|
 *----------------------------------------------------------------------*/
FLD::FluidDiscretExtractor::FluidDiscretExtractor(std::shared_ptr<Core::FE::Discretization> actdis,
    const std::string& condition, bool yescondition)
    : parentdiscret_(actdis)
{
  // get condition, i.e., do we have nodes that belong to a separate section of the domain
  std::vector<const Core::Conditions::Condition*> sepcond;
  parentdiscret_->get_condition(condition, sepcond);

  std::vector<int> allcnd_sepcondnodeids;

  // yes, we have nodes belonging to a separate section
  if (sepcond.size() != 0)
  {
    if (Core::Communication::my_mpi_rank(parentdiscret_->get_comm()) == 0)
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
      XFEM::DiscretizationXWall* xwall = dynamic_cast<XFEM::DiscretizationXWall*>(&*actdis);
      if (nullptr != xwall)
        childdiscret_ = std::make_shared<XFEM::DiscretizationXWall>(
            (std::string) "inflow", parentdiscret_->get_comm(), actdis->n_dim());
      else
        childdiscret_ = std::make_shared<Core::FE::Discretization>(
            (std::string) "inflow", parentdiscret_->get_comm(), actdis->n_dim());
    }
    else  // dummy discretization
      childdiscret_ = std::make_shared<Core::FE::Discretization>(
          (std::string) "none", parentdiscret_->get_comm(), actdis->n_dim());

    // get set of ids of all child nodes
    std::set<int> sepcondnodeset;
    {
      // loop all separation conditions
      // there should be only one condition for turbulent flow problems
      // i.e., one volume in the input section FLUID TURBULENT INFLOW VOLUME
      //       or in the input section
      if ((sepcond.size() != 1) and (condition == "TurbulentInflowSection"))
        FOUR_C_THROW("Only one separate section with condition TurbulentInflowSection expected!");
      // remark: however, more than one are already considered
      for (auto& sepc : sepcond)
      {
        // get nodes ids of all nodes with separtion condition
        const std::vector<int>* sepcondnodeids = (*sepc).get_nodes();

        // and store them
        allcnd_sepcondnodeids.reserve(allcnd_sepcondnodeids.size() + sepcondnodeids->size());
        allcnd_sepcondnodeids.insert(
            allcnd_sepcondnodeids.end(), sepcondnodeids->begin(), sepcondnodeids->end());
      }

      // and change format
      for (int& allcnd_sepcondnodeid : allcnd_sepcondnodeids)
      {
        sepcondnodeset.insert(allcnd_sepcondnodeid);
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
    for (int i = 0; i < parentdiscret_->num_my_col_elements(); ++i)
    {
      Core::Elements::Element* actele = parentdiscret_->l_col_element(i);

      // get the node ids of this element
      const int numnode = actele->num_node();
      const int* nodeids = actele->node_ids();

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
        FOUR_C_THROW(
            "Turbulent inflow is a volume condition! All nodes of an element should have this "
            "condition!");

      if (found == true)
      {
        // loop nodeids
        for (int rr = 0; rr < numnode; ++rr)
        {
          int gid = nodeids[rr];

          if ((parentdiscret_->node_row_map())->lid(gid) > -1)
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
      Core::Nodes::Node* actnode = parentdiscret_->g_node(*id);

      std::shared_ptr<Core::Nodes::Node> sepcondnode =
          std::shared_ptr<Core::Nodes::Node>(actnode->clone());

      childdiscret_->add_node(sepcondnode->x(), sepcondnode->id(), sepcondnode);
    }

    // loop all row elements and add all elements with a separation node
    for (int i = 0; i < parentdiscret_->num_my_row_elements(); ++i)
    {
      Core::Elements::Element* actele = parentdiscret_->l_row_element(i);

      // get the node ids of this element
      const int numnode = actele->num_node();
      const int* nodeids = actele->node_ids();

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
        FOUR_C_THROW(
            "Turbulent inflow is a volume condition! All nodes of an element should have this "
            "condition!");

      // yes, we have a turbulent separation condition (for this element)
      if (found == true)
      {
        std::shared_ptr<Core::Elements::Element> sepcondele =
            std::shared_ptr<Core::Elements::Element>(actele->clone());

        childdiscret_->add_element(sepcondele);
      }
    }

    // child discretization needs a full NodeRowMap and a NodeColMap
    std::shared_ptr<Core::LinAlg::Map> newrownodemap;
    std::shared_ptr<Core::LinAlg::Map> newcolnodemap;

    {
      std::vector<int> rownodes;

      // all row nodes with separation condition are now contained in the child discretization
      for (std::set<int>::iterator id = sepcondelenodes_row.begin();
          id != sepcondelenodes_row.end(); ++id)
      {
        rownodes.push_back(*id);
      }

      // build noderowmap for new distribution of nodes
      newrownodemap = std::make_shared<Core::LinAlg::Map>(
          -1, rownodes.size(), rownodes.data(), 0, childdiscret_->get_comm());

      std::vector<int> colnodes;

      for (std::set<int>::iterator id = sepcondelenodes_col.begin();
          id != sepcondelenodes_col.end(); ++id)
      {
        colnodes.push_back(*id);
      }
      // build nodecolmap for new distribution of nodes
      newcolnodemap = std::make_shared<Core::LinAlg::Map>(
          -1, colnodes.size(), colnodes.data(), 0, childdiscret_->get_comm());
    }

    if (Core::Communication::my_mpi_rank(childdiscret_->get_comm()) == 0)
    {
      printf("| Distribute inflow discretization according to the initial nodemaps");
    }

    childdiscret_->redistribute(
        {*newrownodemap, *newcolnodemap}, {.fill_complete = Core::FE::OptionsFillComplete::none()});

    if (Core::Communication::my_mpi_rank(childdiscret_->get_comm()) == 0)
    {
      std::cout << " ... done.\n";
    }

    // make all conditions known to the child discretization
    // i.e. periodic boundary conditions, dirichlet conditions, ...
    {
      if (Core::Communication::my_mpi_rank(childdiscret_->get_comm()) == 0)
      {
        printf("| Inherit all boundary conditions");
      }

      // get all conditions types prescribed in the input file
      std::vector<std::string> allcond;
      parentdiscret_->get_condition_names(allcond);
      // loop all conditions types
      for (unsigned numcond = 0; numcond < allcond.size(); ++numcond)
      {
        // get condition
        std::vector<const Core::Conditions::Condition*> actcond;
        parentdiscret_->get_condition(allcond[numcond], actcond);
        // loop all condition of the current type
        for (unsigned numactcond = 0; numactcond < actcond.size(); ++numactcond)
        {
          // we use the same nodal ids --- nevertheless, we just use a subset
          // of the node ids and thus cannot copy the conditions completely.
          std::vector<int> reduced_ids;

          // get all nodes of parent discretization having this condition
          const std::vector<int>* candidates = (*actcond[numactcond]).get_nodes();

          std::vector<int> mytoggle(candidates->size(), 0);
          std::vector<int> toggle(candidates->size(), 0);

          // loop all parent nodes with current condition
          // check if node is also contained in child discretization
          for (unsigned rr = 0; rr < candidates->size(); ++rr)
          {
            if (newrownodemap->lid((*candidates)[rr]) > -1)
            {
              mytoggle[rr] = 1;
            }
          }

          // combine marked nodes of all procs
          toggle = Core::Communication::sum_all(mytoggle, childdiscret_->get_comm());

          // and add nodes to the list of child nodes that will get the condition
          for (unsigned rr = 0; rr < candidates->size(); ++rr)
          {
            if (toggle[rr] > 0)
            {
              reduced_ids.push_back((*candidates)[rr]);
            }
          }

          // TODO this hacks the condition
          // replace the nodes of the parent discretization by the nodes of the child discretization
          const_cast<Core::Conditions::Condition&>(*actcond[numactcond]).set_nodes(reduced_ids);

          // finally set condition
          childdiscret_->set_condition(
              allcond[numcond], actcond[numactcond]->copy_without_geometry());
        }

        // redistribute master and slave nodes
        // master and slave nodes are owned by one proc afterwards
        if (allcond[numcond] == "SurfacePeriodic")
        {
          Core::Conditions::PeriodicBoundaryConditions pbc(childdiscret_, false);
          pbc.update_dofs_for_periodic_boundary_conditions();
        }
      }

      if (Core::Communication::my_mpi_rank(childdiscret_->get_comm()) == 0)
      {
        std::cout << " ... done.\n";
      }
    }

    if (Core::Communication::my_mpi_rank(childdiscret_->get_comm()) == 0)
    {
      std::cout << "| Replace dofset by a transparent dofset that copies ";
      std::cout << "the dofs of the original";
      std::cout << " (parent) discretisation";
    }

    // idea: use a transparent dofset and hand through the dof numbering
    // get dof form parent discretization for child discretization
    childdiscret_->replace_dof_set(std::make_shared<Core::DOFSets::TransparentDofSet>(
        parentdiscret_, true));  // true: parallel
    // and assign the dofs to nodes
    // remark: nothing is redistributed here
    childdiscret_->redistribute({*newrownodemap, *newcolnodemap});

    if (Core::Communication::my_mpi_rank(childdiscret_->get_comm()) == 0)
    {
      std::cout << " ... done.\n";
    }

    if (Core::Communication::my_mpi_rank(childdiscret_->get_comm()) == 0)
    {
      std::cout << "| Call PARMETIS on the child discretization and ";
      std::cout << "redistribute according to";
      std::cout << " the new maps\n";
    }

    // this is the actual redistribution
    Core::LinAlg::Map sepcondelenodesmap(*childdiscret_->element_row_map());
    Teuchos::Time time("", true);
    MPI_Comm comm(parentdiscret_->get_comm());

    // Starting from the current partitioning of the discretization, compute nodal maps with a
    // hopefully better partitioning
    std::shared_ptr<const Core::LinAlg::Graph> sepcondnodemap =
        Core::Rebalance::build_graph(*childdiscret_, sepcondelenodesmap);

    Teuchos::ParameterList rebalanceParams;
    rebalanceParams.set<std::string>(
        "num_global_parts", std::to_string(Core::Communication::num_mpi_ranks(comm)));

    const auto& [sepcondrownodes, sepcondcolnodes] =
        Core::Rebalance::rebalance_node_maps(*sepcondnodemap, rebalanceParams);

    if (Core::Communication::my_mpi_rank(childdiscret_->get_comm()) == 0)
    {
      std::cout << "| Redistributing .";
    }
    // redistribute accordingly to the adapted rowmap
    childdiscret_->redistribute({*sepcondrownodes, *sepcondcolnodes},
        {.fill_complete = Core::FE::OptionsFillComplete{
             .assign_degrees_of_freedom = false, .init_elements = false}});

    if (Core::Communication::my_mpi_rank(childdiscret_->get_comm()) == 0)
    {
      std::cout << ".. done.\n";
    }

    // redistribute master and slave nodes
    // master and slave nodes are owned by one proc afterwards
    {
      if (Core::Communication::my_mpi_rank(childdiscret_->get_comm()) == 0)
      {
        std::cout << "| Apply periodic boundary conditions to the redistributed";
        std::cout << " discretization and fetch slave nodes to the master's proc\n";
      }

      Core::Conditions::PeriodicBoundaryConditions pbc(childdiscret_, false);
      pbc.update_dofs_for_periodic_boundary_conditions();

      // get node to node coupling
      col_pbcmapmastertoslave_ = std::make_shared<std::map<int, std::vector<int>>>();
      col_pbcmapmastertoslave_ = pbc.return_all_coupled_col_nodes();
      row_pbcmapmastertoslave_ = std::make_shared<std::map<int, std::vector<int>>>();
      row_pbcmapmastertoslave_ = pbc.return_all_coupled_row_nodes();
    }

    if (Core::Communication::my_mpi_rank(childdiscret_->get_comm()) == 0)
    {
      std::cout << "| Assign the dofs for the redistributed layout, again using ";
      std::cout << "a parallel version";
      std::cout << " of the transparent dofset";
    }

    // idea: use a transparent dofset and hand through the dof numbering
    childdiscret_->replace_dof_set(
        std::make_shared<Core::DOFSets::TransparentDofSet>(parentdiscret_, true));

    // set discretization writer
    childdiscret_->set_writer(std::make_shared<Core::IO::DiscretizationWriter>(*childdiscret_,
        *Global::Problem::instance()->output_control_file(),
        Global::Problem::instance()->spatial_approximation_type()));

    // call fill_complete() to assign the dof
    // remark: equal redistribute(*newrownodemap,*newcolnodemap,true,true,true) as
    //         it also calls fill_complete() at the end
    childdiscret_->fill_complete();

    if (Core::Communication::my_mpi_rank(childdiscret_->get_comm()) == 0)
    {
      std::cout << " ... done.\n";
      printf("|\n");
      printf("+----------------\n\n");
    }

    // some output on the screen
    {
      const int numproc = Core::Communication::num_mpi_ranks(parentdiscret_->get_comm());

      std::vector<int> my_n_nodes(numproc, 0);
      std::vector<int> n_nodes(numproc, 0);
      std::vector<int> my_n_elements(numproc, 0);
      std::vector<int> n_elements(numproc, 0);
      std::vector<int> my_n_ghostele(numproc, 0);
      std::vector<int> n_ghostele(numproc, 0);
      std::vector<int> my_n_dof(numproc, 0);
      std::vector<int> n_dof(numproc, 0);

      int myrank = Core::Communication::my_mpi_rank(childdiscret_->get_comm());

      my_n_nodes[myrank] = childdiscret_->node_row_map()->num_my_elements();
      my_n_elements[myrank] = childdiscret_->num_my_col_elements();
      my_n_ghostele[myrank] =
          childdiscret_->num_my_col_elements() - childdiscret_->num_my_row_elements();
      my_n_dof[myrank] = childdiscret_->dof_row_map()->num_my_elements();

      n_nodes = Core::Communication::sum_all(my_n_nodes, childdiscret_->get_comm());
      n_elements = Core::Communication::sum_all(my_n_elements, childdiscret_->get_comm());
      n_ghostele = Core::Communication::sum_all(my_n_ghostele, childdiscret_->get_comm());
      n_dof = Core::Communication::sum_all(my_n_dof, childdiscret_->get_comm());

      if (Core::Communication::my_mpi_rank(childdiscret_->get_comm()) == 0)
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
      for (int i = 0; i < childdiscret_->num_my_col_elements(); ++i)
      {
        Core::Elements::Element* actele = childdiscret_->l_col_element(i);

        // get the node ids of this element
        const int numnode = actele->num_node();
        const int* nodeids = actele->node_ids();

        // loop nodeids, check if a separation condition is active
        for (int rr = 0; rr < numnode; ++rr)
        {
          Core::Nodes::Node* node = childdiscret_->g_node(nodeids[rr]);
          std::vector<int> nodedofset = childdiscret_->dof(node);

          for (unsigned index = 0; index < nodedofset.size(); ++index)
          {
            int gid = nodedofset[index];

            if (childdiscret_->dof_col_map()->lid(gid) < 0)
            {
              insane = true;
              printf("myrank %d dof %d not in colmap\n",
                  Core::Communication::my_mpi_rank(childdiscret_->get_comm()), gid);
            }
          }
        }
      }
      if (insane) FOUR_C_THROW("invalid dof col map");

      {
        std::set<int> testset;
        for (int rr = 0; rr < childdiscret_->dof_row_map()->num_my_elements(); ++rr)
        {
          int id = childdiscret_->dof_row_map()->my_global_elements()[rr];

          std::set<int>::iterator curr = testset.find(id);
          if (curr != testset.end())
          {
            FOUR_C_THROW("dof_row_map of child dis is not unique on this proc");
          }
          testset.insert(id);
        }

        if (!childdiscret_->dof_row_map()->unique_gids())
        {
          std::cout << *childdiscret_->dof_row_map();

          FOUR_C_THROW("dof_row_map  of child dis is not unique (global)");
        }
      }
    }
  }
  else
  {
    FOUR_C_THROW("Nodes with separation condition expected!");
  }
}

FOUR_C_NAMESPACE_CLOSE
