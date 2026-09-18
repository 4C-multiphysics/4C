// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_condition_periodic.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_dofset_pbc.hpp"
#include "4C_geometric_search_matchingoctree.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_io.hpp"
#include "4C_rebalance_graph_based.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::Conditions::PeriodicBoundaryConditions::PeriodicBoundaryConditions(
    std::shared_ptr<Core::FE::Discretization> actdis, bool verbose)
    : discret_(actdis), verbose_(verbose), pbcdofset_(nullptr)
{
  // get periodic surface boundary conditions
  discret_->get_condition("SurfacePeriodic", mysurfpbcs_);

  if (mysurfpbcs_.empty())
  {
    discret_->get_condition("LinePeriodic", mysurfpbcs_);
  }

  // set number of pairs of periodic boundary conditions
  numpbcpairs_ = mysurfpbcs_.size() / 2;

  // create map that will be connecting target to source nodes owned by
  // this proc
  //       target node -> list of its source node(s)
  allcoupledrownodes_ = std::make_shared<std::map<int, std::vector<int>>>();

  // create map that will be connecting target to source nodes owned or
  // ghosted by this proc
  //       target node -> list of its source node(s)
  allcoupledcolnodes_ = std::make_shared<std::map<int, std::vector<int>>>();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::Conditions::PeriodicBoundaryConditions::update_dofs_for_periodic_boundary_conditions()
{
  if (numpbcpairs_ > 0)
  {
    if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0 && verbose_)
    {
      std::cout << "Generate new dofset for discretization " << discret_->name();
      std::cout << std::endl << std::endl;
    }

    // fetch all sources to the proc of the target
    put_all_sources_to_targets_proc();

    // eventually  optimally distribute the nodes --- up to
    // now, a periodic boundary condition might remove all nodes from a proc ...
    balance_load();

    if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0 && verbose_)
    {
      std::cout << "---------------------------------------------\n";
      std::cout << "Repair Target->Source connection, generate final dofset";
      std::cout << std::endl << std::endl;
    }

    // assign the new dofs, make absolutely sure that we always have all sources to a target
    // the finite edge weights are not a 100% warranty for that...
    put_all_sources_to_targets_proc();

    if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0 && verbose_)
    {
      std::cout << std::endl << std::endl;
    }

    if (verbose_)
    {
      std::shared_ptr<const Teuchos::Comm<int>> TeuchosComm =
          Core::Communication::to_teuchos_comm<int>(discret_->get_comm());
      Teuchos::TimeMonitor::summarize(
          Teuchos::Ptr(TeuchosComm.get()), std::cout, false, true, false);
    }

    if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0 && verbose_)
    {
      std::cout << std::endl << std::endl;
    }

    {
      const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();
      const Core::LinAlg::Map* noderowmap = discret_->node_row_map();

      int mypid = Core::Communication::my_mpi_rank(discret_->get_comm());
      int numprocs = Core::Communication::num_mpi_ranks(discret_->get_comm());

      int countsource = 0;
      for (std::map<int, std::vector<int>>::iterator iter = allcoupledcolnodes_->begin();
          iter != allcoupledcolnodes_->end(); ++iter)
      {
        for (std::vector<int>::iterator viter = iter->second.begin(); viter != iter->second.end();
            ++viter)
        {
          ++countsource;
        }
      }

      std::vector<int> my_n_nodes(numprocs, 0);
      std::vector<int> n_nodes(numprocs, 0);
      std::vector<int> my_n_target(numprocs, 0);
      std::vector<int> n_target(numprocs, 0);
      std::vector<int> my_n_source(numprocs, 0);
      std::vector<int> n_source(numprocs, 0);
      std::vector<int> my_n_elements(numprocs, 0);
      std::vector<int> n_elements(numprocs, 0);
      std::vector<int> my_n_ghostele(numprocs, 0);
      std::vector<int> n_ghostele(numprocs, 0);
      std::vector<int> my_n_dof(numprocs, 0);
      std::vector<int> n_dof(numprocs, 0);

      my_n_nodes[mypid] = noderowmap->num_my_elements();
      my_n_target[mypid] = allcoupledcolnodes_->size();
      my_n_source[mypid] = countsource;
      my_n_elements[mypid] = discret_->num_my_col_elements();
      my_n_ghostele[mypid] = discret_->num_my_col_elements() - discret_->num_my_row_elements();
      my_n_dof[mypid] = dofrowmap->num_my_elements();

      n_nodes = Core::Communication::sum_all(my_n_nodes, discret_->get_comm());
      n_target = Core::Communication::sum_all(my_n_target, discret_->get_comm());
      n_source = Core::Communication::sum_all(my_n_source, discret_->get_comm());
      n_elements = Core::Communication::sum_all(my_n_elements, discret_->get_comm());
      n_ghostele = Core::Communication::sum_all(my_n_ghostele, discret_->get_comm());
      n_dof = Core::Communication::sum_all(my_n_dof, discret_->get_comm());

      if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0 && verbose_)
      {
        printf(
            "   "
            "+-----+---------------+--------------+-------------+-----------------+----------------"
            "+-----------------+\n");
        printf(
            "   | PID |    n_nodes    |   n_target   |   n_source   |    n_elements   |   "
            "n_ghostele   |      n_dof      |\n");
        printf(
            "   "
            "+-----+---------------+--------------+-------------+-----------------+----------------"
            "+-----------------+\n");
        for (int npid = 0; npid < numprocs; ++npid)
        {
          printf("   | %3d | %13d | %12d | %11d | %15d | %14d | %15d |\n", npid, n_nodes[npid],
              n_target[npid], n_source[npid], n_elements[npid], n_ghostele[npid], n_dof[npid]);
          printf(
              "   "
              "+-----+---------------+--------------+-------------+-----------------+--------------"
              "--+-----------------+\n");
        }
        std::cout << std::endl << std::endl;
      }
    }
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::Conditions::PeriodicBoundaryConditions::put_all_sources_to_targets_proc()
{
  if (numpbcpairs_ > 0)
  {
    // clear old data
    allcoupledrownodes_ = std::make_shared<std::map<int, std::vector<int>>>();
    allcoupledcolnodes_ = std::make_shared<std::map<int, std::vector<int>>>();

    // map from global targetnodeids (on this proc) to global sourcenodeids
    // for a single condition
    std::map<int, std::vector<int>> midtosid;

    // pointers to target and source condition
    const Core::Conditions::Condition* targetcond = nullptr;
    const Core::Conditions::Condition* sourcecond = nullptr;

    // global target node Ids and global source node Ids
    std::vector<int> targetnodeids;
    std::vector<int> sourcenodeids;

    //----------------------------------------------------------------------
    //                     LOOP PERIODIC DIRECTIONS
    //----------------------------------------------------------------------

    std::vector<std::string> planes;
    planes.emplace_back("xy");
    planes.emplace_back("xz");
    planes.emplace_back("yz");
    planes.emplace_back("xyz");

    // the id of the plane --- will be counted in the loop....
    int num = 0;

    // loop over periodic directions/planes
    for (auto& plane : planes)
    {
      // loop over all three layers (we allow three layers since
      // the code should be able to deal with up to cubic splines
      // which couple three layers of nodes)
      for (int nlayer = 0; nlayer < 3; ++nlayer)
      {
        //----------------------------------------------------
        // in the following, we loop all periodic boundary
        // conditions which have the prescribed periodic
        // direction.
        // For every Target condition, we add the nodes into
        // the set of all targetnodeids for periodic boundary
        // conditions with this homogeneous direction.
        // The same is done for the source conditions.
        // loop pairs of periodic boundary conditions
        for (int pbcid = 0; pbcid < numpbcpairs_; ++pbcid)
        {
          // target and source sets for this periodic direction
          std::set<int> targetset;
          std::set<int> sourceset;
          // possible angles of rotation for source plane for each pbc pair
          std::vector<double> rotangles(numpbcpairs_);

          // absolute node matching tolerance for octree
          double abs_tol = 0.0;

          // a toggle to indicate whether tolerance for octree was already set,
          // if so check if all values are equal
          bool tol_set = false;

          //--------------------------------------------------
          // get target and source condition pair with id pbcid

          for (auto& mysurfpbc : mysurfpbcs_)
          {
            const int id_zero_based = mysurfpbc->parameters().get<int>("ID") - 1;
            const int layer = mysurfpbc->parameters().get<int>("LAYER");
            // yes, I am the condition with id pbcid and in the desired layer

            if (id_zero_based == pbcid && layer == nlayer)
            {
              const auto& mytargetsourcetoggle =
                  mysurfpbc->parameters().get<std::string>("MASTER_OR_SLAVE");

              if (mytargetsourcetoggle == "Master")
              {
                targetcond = mysurfpbc;

                //--------------------------------------------------
                // check whether this periodic boundary condition belongs
                // to thisplane

                const auto& dofsforpbcplanename =
                    targetcond->parameters().get<std::string>("PLANE");

                if (dofsforpbcplanename == plane)
                {
                  // add all target nodes to targetset

                  //--------------------------------------------------
                  // get global target node Ids
                  const std::vector<int>* targetidstoadd;

                  targetidstoadd = targetcond->get_nodes();

                  for (int idtoadd : *targetidstoadd)
                  {
                    // we only add row nodes to the set
                    if (discret_->have_global_node(idtoadd))
                      if (discret_->g_node(idtoadd)->owner() ==
                          Core::Communication::my_mpi_rank(discret_->get_comm()))
                        targetset.insert(idtoadd);
                  }

                  // check for angle of rotation (has to be zero for target plane)
                  const double angle = targetcond->parameters().get<double>("ANGLE");
                  if (abs(angle) > 1e-13)
                    FOUR_C_THROW("Angle is not zero for target plane: {}", angle);
                }
              }
              else if (mytargetsourcetoggle == "Slave")
              {
                sourcecond = mysurfpbc;

                //--------------------------------------------------
                // check whether this periodic boundary condition belongs
                // to thisplane
                const auto& dofsforpbcplanename =
                    sourcecond->parameters().get<std::string>("PLANE");

                if (dofsforpbcplanename == plane)
                {
                  // add all source nodes to sourceset

                  //--------------------------------------------------
                  // get global source node Ids
                  const std::vector<int>* sourceidstoadd;

                  sourceidstoadd = sourcecond->get_nodes();

                  for (int idtoadd : *sourceidstoadd)
                  {
                    // we only add row nodes to the set
                    if (discret_->have_global_node(idtoadd))
                      if (discret_->g_node(idtoadd)->owner() ==
                          Core::Communication::my_mpi_rank(discret_->get_comm()))
                        sourceset.insert(idtoadd);
                  }

                  // check for angle of rotation of source plane and store it
                  const double angle = sourcecond->parameters().get<double>("ANGLE");
                  if (abs(angle) > 1e-13)
                  {
                    if ((plane != "xz") && (plane != "yz"))
                      FOUR_C_THROW(
                          "Rotation of source plane only implemented for xz and yz planes");
                    else
                    {
                      rotangles[pbcid] =
                          angle * std::numbers::pi / 180.0;  // convert from DEG to RAD!
                      if (pbcid > 0)
                      {
                        if (rotangles[pbcid] != rotangles[pbcid - 1])
                          FOUR_C_THROW("Angle has to be the same for all pairs in pbc");
                      }
                    }
                  }
                }
              }
              else
              {
                FOUR_C_THROW("pbc is neither target nor source");
              }


              // set tolerance for octree
              const double tol = mysurfpbc->parameters().get<double>("ABSTREETOL");

              if (!tol_set)
              {
                abs_tol = tol;

                tol_set = true;
              }
              else
              {
                if (fabs(abs_tol - tol) > 1e-5)
                {
                  FOUR_C_THROW(
                      "none matching tolerances {:12.5e} neq {:12.5e} for nodmatching octree. All "
                      "values in direction {} have to match\n",
                      abs_tol, tol, plane.c_str());
                }
              }
            }  // end if i am the right condition in the right layer
          }  // end loop over conditions



          //--------------------------------------------------
          // vector specifying the plane of this pair
          //
          //     |                           |
          //     |                           |
          //     |      parallel planes      |
          //     |-------------------------->|
          //     |                           |
          //     |                           |
          //     |                           |
          //   source                      target
          //
          //

          // we transform the three std::strings "xy", "xz", "yz" into integer
          // values dofsforpbcplanename
          std::vector<int> dofsforpbcplane(2);

          // this is a char-operation:
          //
          //       x -> 0
          //       y -> 1
          //       z -> 2
          //
          // it is based on the fact that the letters x, y and z are
          // consecutive in the ASCII table --- 'x' is the ASCII
          // value of x ....

          if (plane == "xyz")
          {
            // nodes in exact the same position are coupled
            dofsforpbcplane.clear();
          }
          else
          {
            dofsforpbcplane[0] = plane.c_str()[0] - 'x';
            dofsforpbcplane[1] = plane.c_str()[1] - 'x';
          }
          //--------------------------------------------------
          // we just write the sets into vectors
          (targetnodeids).clear();
          (sourcenodeids).clear();

          for (std::set<int>::iterator appendednode = targetset.begin();
              appendednode != targetset.end(); ++appendednode)
          {
            targetnodeids.push_back(*appendednode);
          }

          for (std::set<int>::iterator appendednode = sourceset.begin();
              appendednode != sourceset.end(); ++appendednode)
          {
            sourcenodeids.push_back(*appendednode);
          }

          //----------------------------------------------------------------------
          //      CONSTRUCT NODE MATCHING BETWEEN MASTER AND SLAVE NODES
          //                        FOR THIS DIRECTION
          //----------------------------------------------------------------------

          // clear map from global targetnodeids (on this proc) to global
          // sourcenodeids --- it belongs to this target source pair!!!
          midtosid.clear();

          if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0 && verbose_ &&
              pbcid == numpbcpairs_ - 1)
          {
            std::cout << " creating layer " << nlayer << " of midtosid-map in " << plane
                      << " direction ... ";
            fflush(stdout);
          }

          // get map target on this proc -> source on some proc
          create_node_coupling_for_single_pbc(
              midtosid, targetnodeids, sourcenodeids, dofsforpbcplane, rotangles[0], abs_tol);

          if (Core::Communication::num_mpi_ranks(discret_->get_comm()) == 1)
          {
            if (targetnodeids.size() != midtosid.size())
            {
              // before throwing FOUR_C_THROW, print helpful information to screen
              for (size_t i = 0; i < targetnodeids.size(); i++)
              {
                int mid = targetnodeids[i];
                bool found = false;
                std::map<int, std::vector<int>>::iterator curr;
                for (curr = midtosid.begin(); curr != midtosid.end(); ++curr)
                {
                  if (curr->first == mid)
                  {
                    found = true;
                    break;
                  }
                }
                if (not found)
                {
                  const auto& x = discret_->g_node(mid)->x();
                  std::cout << "\ntarget node not found in midtosid list: " << mid
                            << "  coord: x=" << x[0] << " y=" << x[1] << " z=" << x[2];
                }
              }
              // now it is time for the   FOUR_C_THROW
              FOUR_C_THROW("have {} targets in midtosid list, {} expected\n", midtosid.size(),
                  targetnodeids.size());
            }
          }

          if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0 && verbose_ &&
              pbcid == numpbcpairs_ - 1)
          {
            std::cout << "adding connectivity to previous pbcs ... ";
            fflush(stdout);
          }

          //----------------------------------------------------------------------
          //      ADD CONNECTIVITY TO CONNECTIVITY OF ALL PREVIOUS PBCS
          //----------------------------------------------------------------------
          // Add the connectivity from this condition to the connectivity
          // of all previously processed periodic boundary conditions.
          // Redistribute the nodes (rownodes+ghosting)
          // Assign the same degrees of freedom to coupled nodes
          add_connectivity(midtosid, num);

          if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0 && verbose_ &&
              pbcid == numpbcpairs_ - 1)
          {
            std::cout << "done.\n";
            fflush(stdout);
          }
        }  // end loop pairs of periodic boundary conditions
        ++num;
      }  // end loop over layers
    }  // end loop over planes

    //----------------------------------------------------------------------
    //         REDISTRIBUTE ACCORDING TO THE GENERATED CONNECTIVITY
    //----------------------------------------------------------------------

    if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0 && verbose_)
    {
      std::cout << "Redistributing: \n";
      fflush(stdout);
    }

    redistribute_and_create_dof_coupling();

    if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0 && verbose_)
    {
      std::cout << "... done\n";
      fflush(stdout);
    }
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::Conditions::PeriodicBoundaryConditions::create_node_coupling_for_single_pbc(
    std::map<int, std::vector<int>>& midtosid, const std::vector<int> targetnodeids,
    const std::vector<int> sourcenodeids, const std::vector<int> dofsforpbcplane,
    const double rotangle, const double abstol)
{
  // these are just parameter definitions for the octree search algorithm
  double tol = abstol;
  int maxnodeperleaf = 250;

  //----------------------------------------------------------------------
  //                   BUILD PROCESSOR LOCAL OCTREE
  //----------------------------------------------------------------------

  // build processor local octree
  auto nodematchingoctree = Core::GeometricSearch::NodeMatchingOctree();

  nodematchingoctree.init(*discret_, targetnodeids, maxnodeperleaf, tol);
  nodematchingoctree.setup();

  //----------------------------------------------------------------------
  //  SEARCH CLOSEST NODES IN OCTREES ON ALL PROCESSORS
  //----------------------------------------------------------------------

  // create connectivity for this condition in this direction
  {
    // create map from gid targetnode -> gid corresponding sourcenode
    nodematchingoctree.create_global_entity_matching(
        sourcenodeids, dofsforpbcplane, rotangle, midtosid);
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::Conditions::PeriodicBoundaryConditions::add_connectivity(
    std::map<int, std::vector<int>>& midtosid, const int pbcid)
{
  TEUCHOS_FUNC_TIME_MONITOR("Conditions::PeriodicBoundaryConditions::add_connectivity");

  // the "inverse" mapping of allcoupled(row/col)nodes
  //       source node -> its target node (list of size 1)
  std::shared_ptr<std::map<int, std::vector<int>>> inversenodecoupling;
  inversenodecoupling = std::make_shared<std::map<int, std::vector<int>>>();

  // Teuchos::rcp to the constructed rowmap
  std::shared_ptr<Core::LinAlg::Map> newrownodemap;

  //----------------------------------------------------------------------
  //  ADD THE CONNECTIVITY FROM THIS CONDITION TO THE CONNECTIVITY OF
  //                           ALL CONDITIONS
  //----------------------------------------------------------------------
  {
    std::map<int, std::vector<int>>::iterator iter;
    int targetid;
    int sourceid;
    bool alreadyinlist;

    for (iter = midtosid.begin(); iter != midtosid.end(); ++iter)
    {
      // get id of targetnode and sourcenode
      targetid = iter->first;

      std::vector<int>::iterator i;
      for (i = (iter->second).begin(); i != (iter->second).end(); ++i)
      {
        sourceid = *i;
        if (sourceid == targetid)
          FOUR_C_THROW(
              "Node {} is target AND source node of periodic boundary condition", targetid);

        // is targetid already in allcoupledrownodes?
        {
          alreadyinlist = false;

          std::map<int, std::vector<int>>::iterator found;

          found = allcoupledrownodes_->find(targetid);
          if (found != allcoupledrownodes_->end())
          {
            // targetid is already in the list --- i.e., the target is the
            // target of a previous condition. Simply append the source id here
            alreadyinlist = true;
            found->second.push_back(sourceid);
          }
          // targetid is not in the list yet. -> new entry
          if (alreadyinlist == false)
          {
            (*allcoupledrownodes_)[targetid].push_back(sourceid);
          }  // end if not in map
        }
      }
    }  // end insert entries of midtosid into the allcoupledrownodes map

    //---------------------------------------------------------------
    //     COMPLETE MATCHING FOR NODES WITH MORE THAN ONE PBC
    //---------------------------------------------------------------
    // complete matching --- we are able to do this because of the
    // communication step (first step is no problem at all ...)

    {
      if (pbcid > 0)
      {
        // 1) each proc generates a list of its multiple coupled targets
        // 2) the list is communicated in a round robin pattern to all the
        //    other procs.
        // 3) the proc checks the package from each proc and inserts missing
        //    links into the multiple coupling


        //--------------------------------------------------------------------
        // -> 1) create a list of multiple target
        // Communicate multiple couplings for completion...
        std::map<int, std::vector<int>> multiplecouplings;
        for (iter = midtosid.begin(); iter != midtosid.end(); ++iter)
        {
          // get id of targetnode and the node itself
          targetid = iter->first;
          Core::Nodes::Node* actnode = discret_->g_node(targetid);

          // get all periodic boundary conditions on this node

          std::vector<const Core::Conditions::Condition*> thiscond =
              discret_->get_conditions_on_node("SurfacePeriodic", actnode);

          if (thiscond.empty())
          {
            thiscond = discret_->get_conditions_on_node("LinePeriodic", actnode);
          }

          // loop them and check, whether this is a pbc pure target node
          // for all previous conditions
          unsigned ntimestarget = 0;
          for (unsigned numcond = 0; numcond < thiscond.size(); ++numcond)
          {
            const std::string& mytargetsourcetoggle =
                thiscond[numcond]->parameters().get<std::string>("MASTER_OR_SLAVE");

            if (mytargetsourcetoggle == "Master")
            {
              ++ntimestarget;
            }  // end is source?
          }  // end loop this conditions

          if (ntimestarget == thiscond.size())
          {
            // yes, we have such a pure target node
            std::vector<int> thiscoupling;
            for (std::vector<int>::iterator rr = (*allcoupledrownodes_)[targetid].begin();
                rr != (*allcoupledrownodes_)[targetid].end(); ++rr)
            {
              thiscoupling.push_back(*rr);
            }

            // add it to the list of multiple coupled targets on this proc
            multiplecouplings[targetid] = thiscoupling;
          }
        }

        //--------------------------------------------------------------------
        // -> 2) round robin loop

        const int numproc = Core::Communication::num_mpi_ranks(discret_->get_comm());
        const int myrank = Core::Communication::my_mpi_rank(discret_->get_comm());  // me
        const int torank = (myrank + 1) % numproc;                                  // to
        const int fromrank = (myrank + numproc - 1) % numproc;                      // from

        Core::Communication::Exporter exporter(discret_->get_comm());


        for (int irobin = 0; irobin < numproc; ++irobin)
        {
          std::vector<char> sdata;
          std::vector<char> rdata;

          // ---- pack data for sending -----
          {
            Core::Communication::PackBuffer data;

            std::vector<int> mids;
            for (std::map<int, std::vector<int>>::const_iterator iter = multiplecouplings.begin();
                iter != multiplecouplings.end(); ++iter)
              mids.push_back(iter->first);

            add_to_pack(data, mids);
            for (std::map<int, std::vector<int>>::const_iterator iter = multiplecouplings.begin();
                iter != multiplecouplings.end(); ++iter)
              add_to_pack(data, iter->second);

            std::swap(sdata, data());
          }

          // ---- send ----
          MPI_Request request;
          exporter.i_send(myrank, torank, sdata.data(), (int)sdata.size(), 1337, request);

          // ---- receive ----
          int length = rdata.size();
          int tag = -1;
          int from = -1;
          exporter.receive_any(from, tag, rdata, length);
          if (tag != 1337 or from != fromrank)
            FOUR_C_THROW("Received data from the wrong proc soll({} -> {}) is({} -> {})", fromrank,
                myrank, from, myrank);

          // ---- unpack ----
          {
            multiplecouplings.clear();
            Core::Communication::UnpackBuffer buffer(rdata);
            std::vector<int> mids;
            extract_from_pack(buffer, mids);

            for (int mid : mids)
            {
              std::vector<int> slvs;
              extract_from_pack(buffer, slvs);
              multiplecouplings[mid] = slvs;
            }
          }

          // wait for all communication to finish
          exporter.wait(request);
          Core::Communication::barrier(discret_->get_comm());  // I feel better this way ;-)

          //--------------------------------------------------
          // -> 3) Try to complete the matchings

          for (std::map<int, std::vector<int>>::iterator mciter = multiplecouplings.begin();
              mciter != multiplecouplings.end(); ++mciter)
          {
            size_t len = mciter->second.size();
            for (size_t mm = 0; mm < len; ++mm)  // this cannot be done through an iterator since we
                                                 // append to the vector while looping over it.
            {
              int possibletarget = (mciter->second)[mm];

              std::map<int, std::vector<int>>::iterator found =
                  allcoupledrownodes_->find(possibletarget);

              if (found != allcoupledrownodes_->end())
              {
                // close the connectivity using the source node which was the
                // targetnode of the previous condition
                for (std::vector<int>::iterator fsiter = found->second.begin();
                    fsiter != found->second.end(); ++fsiter)
                {
                  bool doit = true;
                  for (std::vector<int>::const_iterator innersiter = mciter->second.begin();
                      innersiter != mciter->second.end(); ++innersiter)
                    if (*fsiter == *innersiter) doit = false;

                  if (doit) mciter->second.push_back(*fsiter);
                }
                allcoupledrownodes_->erase(found);
              }  // end if we have a further connectivity information...
            }
          }
        }

        // add this information to the map of all coupled nodes
        for (std::map<int, std::vector<int>>::iterator mciter = multiplecouplings.begin();
            mciter != multiplecouplings.end(); ++mciter)
        {
          std::map<int, std::vector<int>>::iterator found =
              allcoupledrownodes_->find(mciter->first);

          if (found != allcoupledrownodes_->end())
          {
            for (size_t i = found->second.size(); i < mciter->second.size(); ++i)
              found->second.push_back(mciter->second[i]);
          }
        }
      }
    }  // end complete matching
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::Conditions::PeriodicBoundaryConditions::redistribute_and_create_dof_coupling()
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "Conditions::PeriodicBoundaryConditions::redistribute_and_create_dof_coupling");

  // the "inverse" mapping of allcoupled(row/col)nodes
  //       source node -> its target node (list of size 1)
  std::shared_ptr<std::map<int, std::vector<int>>> inversenodecoupling;
  inversenodecoupling = std::make_shared<std::map<int, std::vector<int>>>();

  // Teuchos::rcp to the constructed rowmap
  std::shared_ptr<Core::LinAlg::Map> newrownodemap;

  {
    // make sure we have a filled discretisation at this place
    // dofs are not required yet, they are assigned after redistribution
    // accessing the noderowmap requires a 'completed' discretization
    if (!discret_->filled())
    {
      discret_->fill_complete(Core::FE::OptionsFillComplete::none());
    }

    // a list of all nodes on this proc
    std::vector<int> nodesonthisproc(discret_->node_row_map()->num_my_elements());

    // get all node gids of nodes on this proc
    discret_->node_row_map()->my_global_elements(std::span<int>(nodesonthisproc));

    std::set<int> nodeset;

    for (std::vector<int>::const_iterator rr = nodesonthisproc.begin(); rr != nodesonthisproc.end();
        ++rr)
    {
      nodeset.insert(*rr);
    }

    // -----------------------------------------------
    // remove all node gids of source nodes on this proc

    // get all periodic boundary conditions on this node
    std::vector<const Core::Conditions::Condition*> thiscond;

    std::vector<const Core::Conditions::Condition*> linecond;
    discret_->get_condition("LinePeriodic", linecond);

    thiscond.insert(thiscond.end(), linecond.begin(), linecond.end());

    std::vector<const Core::Conditions::Condition*> surfcond;
    discret_->get_condition("SurfacePeriodic", surfcond);
    thiscond.insert(thiscond.end(), surfcond.begin(), surfcond.end());

    int myerase = 0;
    int numerase = 0;

    int mycerase = 0;
    int numcerase = 0;

    for (unsigned numcond = 0; numcond < thiscond.size(); ++numcond)
    {
      const std::string& mytargetsourcetoggle =
          thiscond[numcond]->parameters().get<std::string>("MASTER_OR_SLAVE");

      if (mytargetsourcetoggle == "Slave")
      {
        const std::vector<int>* sourceidstodel;

        sourceidstodel = thiscond[numcond]->get_nodes();

        for (std::vector<int>::const_iterator idtodel = (*sourceidstodel).begin();
            idtodel != (*sourceidstodel).end(); ++idtodel)
        {
          if (discret_->have_global_node(*idtodel))
          {
            // erase the coupled nodes from the map --- they are redundant
            allcoupledrownodes_->erase(*idtodel);

            Core::Nodes::Node* actnode = discret_->g_node(*idtodel);

            // check for row nodesactnodes ??????????????????
            if (actnode->owner() != Core::Communication::my_mpi_rank(discret_->get_comm()))
            {
              continue;
            }


            ++mycerase;

            std::set<int>::iterator curr = nodeset.find(*idtodel);
            if (curr != nodeset.end())
            {
              // erase id from vector
              nodeset.erase(curr);
              ++myerase;
            }
          }
        }
      }
    }

    numerase = Core::Communication::sum_all(myerase, discret_->get_comm());
    numcerase = Core::Communication::sum_all(mycerase, discret_->get_comm());
    if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0 && verbose_)
    {
      std::cout << " Erased " << numerase << " sources from nodeset.\n";
      std::cout << " Erased " << numcerase << " from the map of all ";
      std::cout << "coupled rownodes.\n";
      std::cout << "        (we want a target->sources map, so all entries ";
      std::cout << "source->... are deleted)\n";
    }

    nodesonthisproc.clear();

    for (std::set<int>::iterator rr = nodeset.begin(); rr != nodeset.end(); ++rr)
    {
      nodesonthisproc.push_back(*rr);
    }

    int mynumappend = 0;
    int numappend = 0;

    // append sourcenodes to this list of nodes on this proc
    {
      for (std::map<int, std::vector<int>>::iterator curr = allcoupledrownodes_->begin();
          curr != allcoupledrownodes_->end(); ++curr)
      {
        for (std::vector<int>::iterator iter = curr->second.begin(); iter != curr->second.end();
            ++iter)
        {
          int sourceid = *iter;

          nodesonthisproc.push_back(sourceid);
          ++mynumappend;
        }
      }
    }

    numappend = Core::Communication::sum_all(mynumappend, discret_->get_comm());
    if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0 && verbose_)
    {
      std::cout << " Appended " << numappend << " ids which belong to source ";
      std::cout << "nodes that are coupled to a target\n";
      std::cout << "        node. They will be fetched to the target's ";
      std::cout << "procs, an their total \n";
      std::cout << "        number has to equal the number of sources ";
      std::cout << "erased from the nodeset.\n";
    }


    {
      int mymax = 0;
      int max = 0;
      int mymin = 1000;
      int min = 0;

      int myallcouplednodes = allcoupledrownodes_->size();
      int allcouplednodes = 0;

      for (std::map<int, std::vector<int>>::iterator curr = allcoupledrownodes_->begin();
          curr != allcoupledrownodes_->end(); ++curr)
      {
        if ((int)curr->second.size() > mymax)
        {
          mymax = curr->second.size();
        }
        if ((int)curr->second.size() < mymin)
        {
          mymin = curr->second.size();
        }
      }

      allcouplednodes = Core::Communication::sum_all(myallcouplednodes, discret_->get_comm());
      max = Core::Communication::max_all(mymax, discret_->get_comm());
      min = Core::Communication::min_all(mymin, discret_->get_comm());

      if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0 && verbose_)
      {
        std::cout << " The layout is generated: " << allcouplednodes
                  << " targets are coupled to at least " << min << " and up to " << max
                  << " sources,\n";
        std::cout << "        all target/source couples are sent to the same proc.\n";
      }
    }


    {
      int myn = (int)nodesonthisproc.size();
      int gn = 0;

      gn = Core::Communication::sum_all(myn, discret_->get_comm());

      if (gn != discret_->num_global_nodes())
      {
        FOUR_C_THROW(
            "Unmatching numbers of nodes before and after call Redistribution. Nodemap constructor "
            "will crash.\n");
      }
    }

    // build noderowmap for new distribution of nodes
    newrownodemap = std::make_shared<Core::LinAlg::Map>(discret_->num_global_nodes(),
        std::span<const int>(nodesonthisproc), 0, discret_->get_comm());

    // create nodal graph of problem, according to old RowNodeMap
    std::shared_ptr<Core::LinAlg::Graph> oldnodegraph = discret_->build_node_graph();

    // export the graph to newrownodemap
    Core::LinAlg::Graph nodegraph(*newrownodemap, 108);

    {
      Core::LinAlg::Export exporter(*discret_->node_row_map(), *newrownodemap);
      nodegraph.export_to(*oldnodegraph, exporter, Core::LinAlg::CombineMode::add);
    }
    nodegraph.fill_complete();
    nodegraph.optimize_storage();

    // build nodecolmap for new distribution of nodes
    const Core::LinAlg::Map cntmp = nodegraph.col_map();

    std::shared_ptr<Core::LinAlg::Map> newcolnodemap;

    newcolnodemap = std::make_shared<Core::LinAlg::Map>(-1,
        std::span<const int>(cntmp.my_global_elements(), cntmp.num_my_elements()), 0,
        discret_->get_comm());

    //----------------------------------------------------------------------
    //       GHOSTED NODES NEED INFORMATION ON THEIR COUPLED NODES
    //----------------------------------------------------------------------

    // create the inverse map --- sourcenode -> targetnode
    inversenodecoupling->clear();

    for (std::map<int, std::vector<int>>::iterator curr = allcoupledrownodes_->begin();
        curr != allcoupledrownodes_->end(); ++curr)
    {
      for (unsigned rr = 0; rr < curr->second.size(); ++rr)
      {
        (*inversenodecoupling)[curr->second[rr]].push_back(curr->first);
      }
    }

    *allcoupledcolnodes_ = (*allcoupledrownodes_);
    {
      // create an exporter
      Core::Communication::Exporter exportconnectivity(
          *newrownodemap, *newcolnodemap, discret_->get_comm());

      // export information on all target->source couplings (with multiple
      // couplings)
      exportconnectivity.do_export(*allcoupledcolnodes_);

      // export the inverse source->target matching without multiple couplings
      exportconnectivity.do_export(*inversenodecoupling);
    }

    // to assign the degrees of freedom, we have to make sure that coupled
    // nodes are only ghosted in pairs --- so modify the colmap
    {
      // mycolnodes contains all nodes which will be stored on this proc
      // according to the colmap constructed
      std::vector<int> mycolnodes(newcolnodemap->num_my_elements());
      newcolnodemap->my_global_elements(std::span<int>(mycolnodes));

      // determine all ghosted source nodes in this vector which do not have
      // a ghosted target on this proc --- we have to fetch it to be able
      // to assign the dofs
      for (std::map<int, std::vector<int>>::iterator curr = inversenodecoupling->begin();
          curr != inversenodecoupling->end(); ++curr)
      {
        if (curr->second.empty())
        {
          FOUR_C_THROW("inverse source-target matching incomplete");
        }
        int mytarget = curr->second[0];
        if (newcolnodemap->lid(mytarget) < 0)
        {
          // was target already added to the list of (ghosted) nodes?
          std::vector<int>::iterator found;
          found = find(mycolnodes.begin(), mycolnodes.end(), mytarget);
          // no, it's not inside
          if (found == mycolnodes.end())
          {
            mycolnodes.push_back(mytarget);
          }
        }
      }

      // We need to do this again in order to get the new target-source pairs
      // that might have been added in the previous loop over the inversenodecoupling
      {
        // now reconstruct the extended colmap
        newcolnodemap = std::make_shared<Core::LinAlg::Map>(
            -1, std::span<const int>(mycolnodes), 0, discret_->get_comm());

        *allcoupledcolnodes_ = (*allcoupledrownodes_);

        // create an exporter
        Core::Communication::Exporter exportconnectivity(
            *newrownodemap, *newcolnodemap, discret_->get_comm());

        // export information on all target->source couplings (with multiple
        // couplings)
        exportconnectivity.do_export(*allcoupledcolnodes_);
      }

      // determine all ghosted target nodes in this vector which do not have
      // all their sources ghosted on this proc --- we have to fetch them to be able
      // to assign the dofs
      for (std::map<int, std::vector<int>>::iterator curr = allcoupledcolnodes_->begin();
          curr != allcoupledcolnodes_->end(); ++curr)
      {
        if (curr->second.empty())
        {
          FOUR_C_THROW("target-source matching incomplete");
        }
        for (size_t i = 0; i < curr->second.size(); ++i)
        {
          if (newcolnodemap->lid(curr->second[i]) < 0)
          {
            // was source already added to the list of (ghosted) nodes?
            std::vector<int>::iterator found;
            found = find(mycolnodes.begin(), mycolnodes.end(), curr->second[i]);
            // no, it's not inside
            if (found == mycolnodes.end())
            {
              mycolnodes.push_back(curr->second[i]);
            }
          }
        }
      }

      // now reconstruct the extended colmap
      newcolnodemap = std::make_shared<Core::LinAlg::Map>(
          -1, std::span<const int>(mycolnodes), 0, discret_->get_comm());

      *allcoupledcolnodes_ = (*allcoupledrownodes_);

      // the new target-ghost nodes need their information about
      // connectivity
      {
        // create an exporter
        Core::Communication::Exporter exportconnectivity(
            *newrownodemap, *newcolnodemap, discret_->get_comm());
        // export information on all source->target couplings (with multiple
        // couplings)
        exportconnectivity.do_export(*allcoupledcolnodes_);
      }
    }

    // check whether we have already passed a PBCDofSet to the discretization
    // If we did not the regular DofSet is replaced with a PBCDofSet. This will
    // lead to a new offset of the DofGIDs and therefore make exporting of vectors
    // impossible.
    // If we already passed a PBCDofSet to the discretization we simply update the
    // DofSet and therefore maintain a correct DofGIDs-offset.
    if (pbcdofset_ == nullptr)
    {
      // create a new dofset specialisation for periodic boundary conditions
      // the 'true' flag makes sure that the pbc dofset replaces the old
      // dofset also in the static_dofsets_.
      pbcdofset_ = std::make_shared<Core::DOFSets::PBCDofSet>(allcoupledcolnodes_);
      discret_->replace_dof_set(0, pbcdofset_, true);
    }
    else
    {
      // the discretization already has a pbc dofset, we merely need to update it
      // (a replace dofset is also not needed since we are working on pointer)
      pbcdofset_->set_coupled_nodes(allcoupledcolnodes_);
      pbcdofset_->reset();
    }

    // redistribute the nodes
    // this contains a call to fill_complete and assigns the same
    // degree of freedom to the matching nodes
    discret_->redistribute({*newrownodemap, *newcolnodemap});
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::Conditions::PeriodicBoundaryConditions::balance_load()
{
  TEUCHOS_FUNC_TIME_MONITOR("Conditions::PeriodicBoundaryConditions::balance_load");

  const Core::LinAlg::Map* node_row_map = discret_->node_row_map();

  // 1. set graph node weights
  auto node_weights = std::make_shared<Core::LinAlg::Vector<double>>(*node_row_map, true);
  {
    // set default node weights
    node_weights->put_scalar(10.0);

    // apply weight of special elements
    for (int node_lid = 0; node_lid < node_row_map->num_my_elements(); ++node_lid)
    {
      const int node_gid = node_row_map->gid(node_lid);
      Core::Nodes::Node* node = discret_->g_node(node_gid);
      double weight = 1.0;

      // loop over adjacent elements of this node and find element with highest cost
      for (auto ele : node->adjacent_elements())
        weight = std::max(weight, ele.user_element()->evaluation_cost());

      node_weights->replace_local_value(node_lid, weight);
    }

    // loop targetnodes to adjust weights of sourcenodes
    // they need a small weight since they do not contribute any dofs to the linear system
    for (const auto& targetsourcepair : *allcoupledcolnodes_)
    {
      const int target_gid = targetsourcepair.first;
      Core::Nodes::Node* target = discret_->g_node(target_gid);

      if (target->owner() != Core::Communication::my_mpi_rank(discret_->get_comm())) continue;

      std::vector<int> source_gids = targetsourcepair.second;
      for (const auto source_gid : source_gids)
      {
        const double weight = 1.0;
        node_weights->replace_global_values(1, &weight, &source_gid);
      }
    }
  }

  // 2. allocate graph
  Core::LinAlg::Graph node_graph(*node_row_map, 108);
  {
    // iterate all elements on this proc including ghosted ones and compute connectivity
    // standard part without target<->source coupling
    // Note:
    // if a proc stores the appropriate ghosted elements, the resulting graph will be the correct
    // and complete graph of the distributed discretization even if nodes are not ghosted.
    for (int ele_lid = 0; ele_lid < discret_->num_my_col_elements(); ++ele_lid)
    {
      Core::Elements::Element* ele = discret_->l_col_element(ele_lid);
      const int num_nodes_per_ele = ele->num_node();
      const int* node_gids_per_ele = ele->node_ids();

      for (int row = 0; row < num_nodes_per_ele; ++row)
      {
        const int node_row_gid = node_gids_per_ele[row];

        if (!node_row_map->my_gid(node_row_gid)) continue;

        for (int col = 0; col < num_nodes_per_ele; ++col)
        {
          int node_col_gid = node_gids_per_ele[col];
          auto neighbor_node = std::span(&node_col_gid, 1);
          node_graph.insert_global_indices(node_row_gid, neighbor_node);
        }
      }
    }

    // additional coupling between target and source
    // we do not only connect target and source nodes but if a target/source
    // is connected to a target/source, we connect the corresponding sources/target as well
    for (int ele_lid = 0; ele_lid < discret_->num_my_col_elements(); ++ele_lid)
    {
      Core::Elements::Element* ele = discret_->l_col_element(ele_lid);
      const int num_nodes_per_ele = ele->num_node();
      const int* node_gids_per_ele = ele->node_ids();

      for (int row = 0; row < num_nodes_per_ele; ++row)
      {
        int node_gid = node_gids_per_ele[row];

        if (!node_row_map->my_gid(node_gid)) continue;

        // only, if this node is a coupled node
        if (allcoupledcolnodes_->find(node_gid) != allcoupledcolnodes_->end())
        {
          // get all targetnodes of this element
          for (int col = 0; col < num_nodes_per_ele; ++col)
          {
            int neighbor_node = node_gids_per_ele[col];

            const auto othertargetsourcepair = allcoupledcolnodes_->find(neighbor_node);
            if (othertargetsourcepair != allcoupledcolnodes_->end())
            {
              const auto other_source_gids = othertargetsourcepair->second;
              // add connection to all sources
              for (auto other_source_gid : other_source_gids)
              {
                auto source_gid_index = std::span(&other_source_gid, 1);
                node_graph.insert_global_indices(node_gid, source_gid_index);

                if (node_row_map->my_gid(other_source_gid))
                {
                  auto targetindex = std::span(&node_gid, 1);
                  node_graph.insert_global_indices(other_source_gid, targetindex);
                }
              }
            }
          }
        }
      }
    }
  }
  node_graph.fill_complete();

  const int myrank = Core::Communication::my_mpi_rank(node_graph.get_comm());

  // get rowmap of the graph  (from blockmap -> map)
  const Core::LinAlg::Map& graph_row_map = node_graph.row_map();
  const Core::LinAlg::Map graph_rowmap(graph_row_map.num_global_elements(),
      std::span<const int>(graph_row_map.my_global_elements(), graph_row_map.num_my_elements()), 0,
      node_graph.get_comm());

  // 3. set graph edge weights
  auto edge_weights = std::make_shared<Core::LinAlg::SparseMatrix>(graph_rowmap, 15);
  {
    // set standard value of edge weight to 10.0
    for (int i = 0; i < node_graph.num_local_rows(); ++i)
    {
      const int grow = node_graph.row_map().gid(i);
      std::span<int> indices;
      node_graph.extract_local_row_view(i, indices);

      std::vector<double> values(indices.size(), 1.0);
      edge_weights->insert_global_values(grow, indices.size(), values.data(), indices.data());
    }

    // loop all target nodes on this proc
    for (const auto& targetsourcepair : *allcoupledcolnodes_)
    {
      Core::Nodes::Node* target = discret_->g_node(targetsourcepair.first);

      if (target->owner() != myrank) continue;

      // loop sourcenodes
      for (int source_gids : targetsourcepair.second)
      {
        Core::Nodes::Node* source = discret_->g_node(source_gids);

        // connections between target and sourcenodes are very strong
        // we do not want to partition between target and source nodes
        std::vector<int> target_gid(1, target->id());
        std::vector<int> source_gid(1, source->id());
        // set cost of strong edges to 100.0
        std::vector<double> value(1, 100.0);

        edge_weights->insert_global_values(target->id(), 1, value.data(), source_gid.data());
        edge_weights->insert_global_values(source->id(), 1, value.data(), target_gid.data());
      }
    }
  }
  edge_weights->complete();

  // 4. setup partitioner and redistribute
  Teuchos::ParameterList paramlist;
  paramlist.set("algorithm", "parmetis");
  paramlist.set("partitioning_approach", "repartition");

  auto newnodegraph =
      Core::Rebalance::rebalance_graph(node_graph, paramlist, node_weights, edge_weights);

  const Core::LinAlg::Map newnoderowmap(-1,
      std::span<const int>(
          newnodegraph->row_map().my_global_elements(), newnodegraph->row_map().num_my_elements()),
      0, discret_->get_comm());
  const Core::LinAlg::Map newnodecolmap(-1,
      std::span<const int>(
          newnodegraph->col_map().my_global_elements(), newnodegraph->col_map().num_my_elements()),
      0, discret_->get_comm());

  discret_->redistribute({newnoderowmap, newnodecolmap},
      {.fill_complete = FE::OptionsFillComplete{.assign_degrees_of_freedom = false}});
}

FOUR_C_NAMESPACE_CLOSE
