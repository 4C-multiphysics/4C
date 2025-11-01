// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_turbulence_transfer_turb_inflow.hpp"

#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_geometric_search_matchingoctree.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_fluid.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_utils_function_of_time.hpp"

FOUR_C_NAMESPACE_OPEN



//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Constructor (public)                                     gammi 03/10|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
FLD::TransferTurbulentInflowCondition::TransferTurbulentInflowCondition(
    std::shared_ptr<Core::FE::Discretization> dis,
    std::shared_ptr<Core::LinAlg::MapExtractor> dbcmaps)
    : dis_(dis), dbcmaps_(dbcmaps), curve_(-1), numveldof_(3)
{
  active_ = false;

  // vector of pointers to all node clouds i.e. conditions to couple
  std::vector<const Core::Conditions::Condition*> nodecloudstocouple;

  // get surfaces to couple
  dis_->get_condition("TransferTurbulentInflow", nodecloudstocouple);

  if (not nodecloudstocouple.empty())
  {
    // activate
    active_ = true;

    // master and slave sets to couple
    std::set<int> masterset;
    std::set<int> slaveset;

    // global master node Ids and global slave node Ids
    std::vector<int> masternodeids;
    std::vector<int> slavenodeids;

    // the (at the moment) one and only direction to couple
    int dir = -1;

    // loop all conditions and check whether they are of master or slave
    // type
    for (std::vector<const Core::Conditions::Condition*>::iterator cond =
             nodecloudstocouple.begin();
        cond != nodecloudstocouple.end(); ++cond)
    {
      // get id, direction info and toggle
      int id = -1;
      int direction = -1;
      ToggleType toggle = none;

      get_data(id, direction, toggle, *cond);

      if (dir == -1)
      {
        dir = direction;
      }
      else
      {
        if (dir != direction)
        {
          FOUR_C_THROW("multiple directions are not supported yet");
        }
      }

      if (id != 1)
      {
        FOUR_C_THROW("expecting only one group of coupling surfaces (up to now), its {}", id);
      }

      switch (toggle)
      {
        case master:
        {
          //--------------------------------------------------
          // get global master node Ids
          const std::vector<int>* masteridstoadd;

          masteridstoadd = (*cond)->get_nodes();

          for (int idtoadd : *masteridstoadd)
          {
            // we construct the local octree only with nodes owned by this proc
            if (dis_->have_global_node(idtoadd))
              if (dis_->g_node(idtoadd)->owner() ==
                  Core::Communication::my_mpi_rank(dis_->get_comm()))
                masterset.insert(idtoadd);
          }

          break;
        }
        case slave:
        {
          //--------------------------------------------------
          // get global slave node Ids
          const std::vector<int>* slaveidstoadd;

          slaveidstoadd = (*cond)->get_nodes();

          for (std::vector<int>::const_iterator idtoadd = (*slaveidstoadd).begin();
              idtoadd != (*slaveidstoadd).end(); ++idtoadd)
          {
            // we only try to match owned nodes of each proc
            if (dis_->have_global_node(*idtoadd))
              if (dis_->g_node(*idtoadd)->owner() ==
                  Core::Communication::my_mpi_rank(dis_->get_comm()))
                slaveset.insert(*idtoadd);
          }

          break;
        }
        default:
          FOUR_C_THROW("toggle non master or slave");
      }
    }

    //--------------------------------------------------
    // just write sets into vectors
    (masternodeids).clear();
    (slavenodeids).clear();

    for (std::set<int>::iterator appendednode = masterset.begin(); appendednode != masterset.end();
        ++appendednode)
    {
      masternodeids.push_back(*appendednode);
    }

    for (std::set<int>::iterator appendednode = slaveset.begin(); appendednode != slaveset.end();
        ++appendednode)
    {
      slavenodeids.push_back(*appendednode);
    }

    // these are just parameter definitions for the octree search algorithm
    const double tol = 1e-6;
    const int maxnodeperleaf = 250;
    const double rotangle = 0.0;

    std::vector<int> dofsforpbcplane(2);
    {
      int mm = 0;
      for (int rr = 0; rr < 3; ++rr)
      {
        if (rr != dir)
        {
          dofsforpbcplane[mm] = rr;
          ++mm;
        }
      }
    }

    // build processor local octree
    auto nodematchingoctree = Core::GeometricSearch::NodeMatchingOctree();
    nodematchingoctree.init(*dis_, masternodeids, maxnodeperleaf, tol);
    nodematchingoctree.setup();

    // create map from gid masternode -> gid corresponding slavenode
    nodematchingoctree.create_global_entity_matching(
        slavenodeids, dofsforpbcplane, rotangle, midtosid_);
    // sanity check
    for (std::map<int, std::vector<int>>::iterator pair = midtosid_.begin();
        pair != midtosid_.end(); ++pair)
    {
      if (pair->second.size() != 1)
      {
        FOUR_C_THROW("expected one node to match, got {} out of {}", pair->second.size(),
            slavenodeids.size());
      }
    }
  }
  return;
}  // TransferTurbulentInflowCondition::TransferTurbulentInflowCondition

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Perform transfer process (public)                        gammi 03/10|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void FLD::TransferTurbulentInflowCondition::transfer(
    const std::shared_ptr<Core::LinAlg::Vector<double>> veln,
    std::shared_ptr<Core::LinAlg::Vector<double>> velnp, const double time)
{
  const Core::LinAlg::Map* dofrowmap = dis_->dof_row_map();

  std::vector<int> mymasters;
  std::vector<std::vector<double>> mymasters_vel(numveldof_);

  if (active_)
  {
    // initialization of time curve factor
    double curvefac = 1.0;
    if (curve_ >= 0)  // yes, we have a time curve
    {
      // time factor for the intermediate step
      if (time >= 0.0)
      {
        curvefac = Global::Problem::instance()
                       ->function_by_id<Core::Utils::FunctionOfTime>(curve_)
                       .evaluate(time);
      }
      else
      {
        // do not compute an "alternative" curvefac here since a negative time value
        // indicates an error.
        FOUR_C_THROW("Negative time value: time = {}", time);
      }
    }
    else  // we do not have a time curve --- time curve factor is constant equal 1
    {
      // nothing to do
    }

    // collect masters on this proc and associated velocities
    for (std::map<int, std::vector<int>>::iterator pair = midtosid_.begin();
        pair != midtosid_.end(); ++pair)
    {
      int gid = pair->first;

      if (dis_->have_global_node(gid))
      {
        mymasters.push_back(gid);

        Core::Nodes::Node* master = dis_->g_node(gid);

        std::vector<int> masterdofs = dis_->dof(master);

        for (int rr = 0; rr < 3; ++rr)
        {
          int lid = dofrowmap->lid(masterdofs[rr]);

          (mymasters_vel[rr]).push_back(veln->local_values_as_span()[lid] * curvefac);
        }
      }
      else
      {
        FOUR_C_THROW("master {} in midtosid but not on proc. This was unexpected", gid);
      }
    }

    // create an exporter for point to point communication
    Core::Communication::Exporter exporter(dis_->get_comm());

    // necessary variables
    MPI_Request request;

    // define send and receive blocks
    std::vector<char> sblock;
    std::vector<char> rblock;

    // get number of processors and the current processors id
    int numproc = Core::Communication::num_mpi_ranks(dis_->get_comm());

    //----------------------------------------------------------------------
    // communication is done in a round robin loop
    //----------------------------------------------------------------------
    for (int np = 0; np < numproc + 1; ++np)
    {
      // in the first step, we cannot receive anything
      if (np > 0)
      {
        receive_block(rblock, exporter, request);

        // Unpack info from the receive block from the last proc
        unpack_local_master_values(mymasters, mymasters_vel, rblock);
      }

      // in the last step, we keep everything on this proc
      if (np < numproc)
      {
        // -----------------------
        // do what we wanted to do
        set_values_available_on_this_proc(mymasters, mymasters_vel, velnp);

        // Pack info into block to send
        Core::Communication::PackBuffer data;
        pack_local_master_values(mymasters, mymasters_vel, data);
        swap(sblock, data());

        send_block(sblock, exporter, request);
      }
    }
  }
  return;
}  // Transfer


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | get condition id etc (private)                            gammi 03/10|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void FLD::TransferTurbulentInflowCondition::get_data(
    int& id, int& direction, ToggleType& type, const Core::Conditions::Condition* cond)
{
  id = cond->parameters().get<int>("ID");

  direction =
      static_cast<int>(cond->parameters().get<Inpar::FLUID::TurbInflowDirection>("DIRECTION"));

  const auto mytoggle = cond->parameters().get<std::string>("toggle");
  if (mytoggle == "master")
  {
    type = master;
  }
  else if (mytoggle == "slave")
  {
    type = slave;
  }
  else
  {
    FOUR_C_THROW("expecting either master or slave");
  }

  // find out whether we will use a time curve
  if (curve_ == -1)
  {
    const auto curve = cond->parameters().get<std::optional<int>>("curve");

    // set zero based curve number
    curve_ = curve.value_or(-1) - 1;
  }
}


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | receive a block in the round robin communication pattern   (private) |
 |                                                          gammi 03/10 |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void FLD::TransferTurbulentInflowCondition::receive_block(
    std::vector<char>& rblock, Core::Communication::Exporter& exporter, MPI_Request& request)
{
  // get number of processors and the current processors id
  int numproc = Core::Communication::num_mpi_ranks(dis_->get_comm());
  int myrank = Core::Communication::my_mpi_rank(dis_->get_comm());

  // necessary variables

  int length = -1;
  int frompid = (myrank + numproc - 1) % numproc;
  int tag = frompid;

  // make sure that you do not think you received something if
  // you didn't
  if (rblock.empty() == false)
  {
    FOUR_C_THROW("rblock not empty");
  }

  // receive from predecessor
  exporter.receive_any(frompid, tag, rblock, length);

  if (tag != (myrank + numproc - 1) % numproc)
  {
    FOUR_C_THROW("received wrong message (ReceiveAny)");
  }

  exporter.wait(request);

  // for safety
  Core::Communication::barrier(exporter.get_comm());

  return;
}  // TransferTurbulentInflowCondition::receive_block


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | send a block in the round robin communication pattern      (private) |
 |                                                          gammi 03/10 |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void FLD::TransferTurbulentInflowCondition::send_block(
    std::vector<char>& sblock, Core::Communication::Exporter& exporter, MPI_Request& request)
{
  // get number of processors and the current processors id
  int numproc = Core::Communication::num_mpi_ranks(dis_->get_comm());
  int myrank = Core::Communication::my_mpi_rank(dis_->get_comm());

  // Send block to next proc.
  int tag = myrank;
  int frompid = myrank;
  int topid = (myrank + 1) % numproc;

  exporter.i_send(frompid, topid, sblock.data(), sblock.size(), tag, request);


  // for safety
  Core::Communication::barrier(exporter.get_comm());

  return;
}  // TransferTurbulentInflowCondition::send_block


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | unpack all master values contained in receive block        (private) |
 |                                                          gammi 03/10 |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void FLD::TransferTurbulentInflowCondition::unpack_local_master_values(std::vector<int>& mymasters,
    std::vector<std::vector<double>>& mymasters_vel, std::vector<char>& rblock)
{
  mymasters.clear();

  midtosid_.clear();

  if ((int)mymasters_vel.size() != numveldof_)
  {
    FOUR_C_THROW("expecting three spatial dimensions in mymasters_vel to unpack into");
  }

  for (int rr = 0; rr < numveldof_; ++rr)
  {
    (mymasters_vel[rr]).clear();
  }

  // position to extract


  Core::Communication::UnpackBuffer buffer(rblock);
  // extract size
  int size = 0;
  extract_from_pack(buffer, size);

  // extract master ids
  for (int i = 0; i < size; ++i)
  {
    int id;

    extract_from_pack(buffer, id);
    mymasters.push_back(id);

    std::map<int, std::vector<int>>::iterator iter = midtosid_.find(id);

    if (iter != midtosid_.end())
    {
      iter->second.clear();
    }
    else
    {
      midtosid_[id].clear();
    }
  }

  // extract slave ids
  for (int rr = 0; rr < size; ++rr)
  {
    int slavesize;

    extract_from_pack(buffer, slavesize);

    for (int ll = 0; ll < slavesize; ++ll)
    {
      int sid;
      extract_from_pack(buffer, sid);

      std::map<int, std::vector<int>>::iterator iter = midtosid_.find(mymasters[rr]);

      if (iter != midtosid_.end())
      {
        iter->second.push_back(sid);
      }
      else
      {
        FOUR_C_THROW("master id {} was not in midtosid_", mymasters[rr]);
      }
    }

    if (midtosid_[mymasters[rr]].size() < 1)
    {
      FOUR_C_THROW("require at least one slave to master {}, got {}", mymasters[rr],
          midtosid_[mymasters[rr]].size());
    }
  }

  // extract values (first u, then v, then w)
  for (int mm = 0; mm < numveldof_; ++mm)
  {
    for (int rr = 0; rr < size; ++rr)
    {
      double value;

      extract_from_pack(buffer, value);

      (mymasters_vel[mm]).push_back(value);
    }
  }

  rblock.clear();
  return;
}  // unpack_local_master_values



//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | pack all master values into a send block                   (private) |
 |                                                          gammi 03/10 |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void FLD::TransferTurbulentInflowCondition::pack_local_master_values(std::vector<int>& mymasters,
    std::vector<std::vector<double>>& mymasters_vel, Core::Communication::PackBuffer& sblock)
{
  int size = mymasters.size();

  if (mymasters_vel.size() != (unsigned)numveldof_)
  {
    FOUR_C_THROW("expecting three spatial dimensions in mymasters_vel to pack");
  }

  for (int rr = 0; rr < numveldof_; ++rr)
  {
    if ((int)(mymasters_vel[rr]).size() != size)
    {
      FOUR_C_THROW("at least one of the components of mymasters_vel has the wrong size");
    }
  }

  // add size  to sendblock
  add_to_pack(sblock, size);

  // add master ids
  for (int rr = 0; rr < size; ++rr)
  {
    add_to_pack(sblock, mymasters[rr]);
  }

  // add slave ids
  for (int rr = 0; rr < size; ++rr)
  {
    std::map<int, std::vector<int>>::iterator iter = midtosid_.find(mymasters[rr]);

    if (iter == midtosid_.end())
    {
      FOUR_C_THROW("tried to pack slaves to master master {}, got none", mymasters[rr]);
    }
    else
    {
      std::vector<int> slaves = iter->second;

      int slavesize = (int)slaves.size();

      add_to_pack(sblock, slavesize);
      for (int ll = 0; ll < slavesize; ++ll)
      {
        add_to_pack(sblock, slaves[ll]);
      }
    }
  }

  // add values (first u, then v, then w)
  for (int mm = 0; mm < numveldof_; ++mm)
  {
    for (int rr = 0; rr < size; ++rr)
    {
      add_to_pack(sblock, (mymasters_vel[mm])[rr]);
    }
  }

  return;
}  // pack_local_master_values

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | for all values available on the processor, do the final setting of the |
 | value                          (private)                 gammi 03/10 |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void FLD::TransferTurbulentInflowCondition::set_values_available_on_this_proc(
    std::vector<int>& mymasters, std::vector<std::vector<double>>& mymasters_vel,
    std::shared_ptr<Core::LinAlg::Vector<double>> velnp)
{
  const std::shared_ptr<const Core::LinAlg::Map> activedbcdofs = dbcmaps_->cond_map();

  for (unsigned nn = 0; nn < mymasters.size(); ++nn)
  {
    std::map<int, std::vector<int>>::iterator iter = midtosid_.find(mymasters[nn]);

    if (iter != midtosid_.end())
    {
      std::vector<int> myslaves(iter->second);

      for (std::vector<int>::iterator sid = myslaves.begin(); sid != myslaves.end(); ++sid)
      {
        // is this slave id on this proc?
        if (dis_->node_row_map()->my_gid(*sid))
        {
          Core::Nodes::Node* slave = dis_->g_node(*sid);

          // get dofs
          std::vector<int> slavedofs = dis_->dof(slave);

          for (int rr = 0; rr < numveldof_; ++rr)
          {
            int gid = slavedofs[rr];

            // only set if DBC is active, otherwise throw error
            if (activedbcdofs->my_gid(gid))
            {
              double value = (mymasters_vel[rr])[nn];

              velnp->replace_global_values(1, &value, &gid);
            }
            else
            {
              int id = slave->id();

              double x = slave->x()[0];
              double y = slave->x()[1];
              double z = slave->x()[2];

              FOUR_C_THROW(
                  "Dirichlet condition required on slave node ({:12.5e},{:12.5e},{:12.5e}), id {}, "
                  "dof "
                  "{} of transfer condition",
                  x, y, z, id, rr);
            }
          }
        }
      }
    }
  }

  return;
}  // set_values_available_on_this_proc


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Constructor (public)                                        bk 09/14|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
FLD::TransferTurbulentInflowConditionXW::TransferTurbulentInflowConditionXW(
    std::shared_ptr<Core::FE::Discretization> dis,
    std::shared_ptr<Core::LinAlg::MapExtractor> dbcmaps)
    : TransferTurbulentInflowCondition(dis, dbcmaps)
{
  numveldof_ = 6;
}


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Perform transfer process (public)                           bk 09/14|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void FLD::TransferTurbulentInflowConditionXW::transfer(
    const std::shared_ptr<Core::LinAlg::Vector<double>> veln,
    std::shared_ptr<Core::LinAlg::Vector<double>> velnp, const double time)
{
  const Core::LinAlg::Map* dofrowmap = dis_->dof_row_map();

  std::vector<int> mymasters;
  // there can be up to 6 velocity dofs per node (3 +3 virtual dofs)
  std::vector<std::vector<double>> mymasters_vel(numveldof_);

  if (active_)
  {
    // initialization of time curve factor
    double curvefac = 1.0;
    if (curve_ >= 0)  // yes, we have a time curve
    {
      // time factor for the intermediate step
      if (time >= 0.0)
      {
        curvefac = Global::Problem::instance()
                       ->function_by_id<Core::Utils::FunctionOfTime>(curve_ + 1)
                       .evaluate(time);
      }
      else
      {
        // do not compute an "alternative" curvefac here since a negative time value
        // indicates an error.
        FOUR_C_THROW("Negative time value: time = {}", time);
      }
    }
    else  // we do not have a time curve --- time curve factor is constant equal 1
    {
      // nothing to do
    }

    // collect masters on this proc and associated velocities
    for (std::map<int, std::vector<int>>::iterator pair = midtosid_.begin();
        pair != midtosid_.end(); ++pair)
    {
      int gid = pair->first;

      if (dis_->have_global_node(gid))
      {
        mymasters.push_back(gid);

        Core::Nodes::Node* master = dis_->g_node(gid);

        std::vector<int> masterdofs = dis_->dof(master);

        for (int rr = 0; rr < 3; ++rr)
        {
          int lid = dofrowmap->lid(masterdofs[rr]);

          (mymasters_vel[rr]).push_back(veln->local_values_as_span()[lid] * curvefac);
        }

        // in xwall, we have another virtual node right after this node
        if (dis_->num_dof(master) == 8)
        {
          for (int rr = 4; rr < 7; ++rr)
          {
            int lid = dofrowmap->lid(masterdofs[rr]);

            (mymasters_vel[rr - 1]).push_back(veln->local_values_as_span()[lid] * curvefac);
          }
        }
        else
          for (int rr = 4; rr < 7; ++rr)
          {
            (mymasters_vel[rr - 1]).push_back(0.0);
          }
      }
      else
      {
        FOUR_C_THROW("master {} in midtosid but not on proc. This was unexpected", gid);
      }
    }

    // create an exporter for point to point communication
    Core::Communication::Exporter exporter(dis_->get_comm());

    // necessary variables
    MPI_Request request;

    // define send and receive blocks
    std::vector<char> sblock;
    std::vector<char> rblock;

    // get number of processors and the current processors id
    int numproc = Core::Communication::num_mpi_ranks(dis_->get_comm());

    //----------------------------------------------------------------------
    // communication is done in a round robin loop
    //----------------------------------------------------------------------
    for (int np = 0; np < numproc + 1; ++np)
    {
      // in the first step, we cannot receive anything
      if (np > 0)
      {
        receive_block(rblock, exporter, request);

        // Unpack info from the receive block from the last proc
        unpack_local_master_values(mymasters, mymasters_vel, rblock);
      }

      // in the last step, we keep everything on this proc
      if (np < numproc)
      {
        // -----------------------
        // do what we wanted to do
        set_values_available_on_this_proc(mymasters, mymasters_vel, velnp);

        // Pack info into block to send
        Core::Communication::PackBuffer data;
        pack_local_master_values(mymasters, mymasters_vel, data);
        swap(sblock, data());

        send_block(sblock, exporter, request);
      }
    }
  }
  return;
}  // Transfer (XW)

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | for all values available on the processor, do the final setting of the |
 | value                          (private)                    bk 09/14 |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void FLD::TransferTurbulentInflowConditionXW::set_values_available_on_this_proc(
    std::vector<int>& mymasters, std::vector<std::vector<double>>& mymasters_vel,
    std::shared_ptr<Core::LinAlg::Vector<double>> velnp)
{
  const std::shared_ptr<const Core::LinAlg::Map> activedbcdofs = dbcmaps_->cond_map();

  for (unsigned nn = 0; nn < mymasters.size(); ++nn)
  {
    std::map<int, std::vector<int>>::iterator iter = midtosid_.find(mymasters[nn]);

    if (iter != midtosid_.end())
    {
      std::vector<int> myslaves(iter->second);

      for (std::vector<int>::iterator sid = myslaves.begin(); sid != myslaves.end(); ++sid)
      {
        // is this slave id on this proc?
        if (dis_->node_row_map()->my_gid(*sid))
        {
          Core::Nodes::Node* slave = dis_->g_node(*sid);

          // get dofs
          std::vector<int> slavedofs = dis_->dof(slave);

          for (int rr = 0; rr < 3; ++rr)
          {
            int gid = slavedofs[rr];

            // only set if DBC is active, otherwise throw error
            if (activedbcdofs->my_gid(gid))
            {
              double value = (mymasters_vel[rr])[nn];

              velnp->replace_global_values(1, &value, &gid);
            }
            else
            {
              int id = slave->id();

              double x = slave->x()[0];
              double y = slave->x()[1];
              double z = slave->x()[2];

              FOUR_C_THROW(
                  "Dirichlet condition required on slave node ({:12.5e},{:12.5e},{:12.5e}), id {}, "
                  "dof "
                  "{} of transfer condition",
                  x, y, z, id, rr);
            }
          }

          // and treat xwall dofs
          if (dis_->num_dof(slave) == 8)
            for (int rr = 4; rr < 7; ++rr)
            {
              int gid = slavedofs[rr];

              // only set if DBC is active, otherwise throw error
              if (activedbcdofs->my_gid(gid))
              {
                double value = (mymasters_vel[rr - 1])[nn];

                velnp->replace_global_values(1, &value, &gid);
              }
              else
                FOUR_C_THROW("xwall dofs don't have active dbc for transfer");
            }
        }
      }
    }
  }

  return;
}  // set_values_available_on_this_proc (XW)

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Constructor (public)                                        bk 09/14|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
FLD::TransferTurbulentInflowConditionNodal::TransferTurbulentInflowConditionNodal(
    std::shared_ptr<Core::FE::Discretization> dis,
    std::shared_ptr<Core::LinAlg::MapExtractor> dbcmaps)
    : TransferTurbulentInflowCondition(dis, dbcmaps)
{
  numveldof_ = 1;
}


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Perform transfer process (public)                           bk 09/14|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void FLD::TransferTurbulentInflowConditionNodal::transfer(
    const std::shared_ptr<Core::LinAlg::Vector<double>> invec,
    std::shared_ptr<Core::LinAlg::Vector<double>> outvec, const double time)
{
  std::vector<int> mymasters;
  std::vector<std::vector<double>> mymasters_vec(numveldof_);

  if (active_)
  {
    // collect masters on this proc and associated velocities
    for (std::map<int, std::vector<int>>::iterator pair = midtosid_.begin();
        pair != midtosid_.end(); ++pair)
    {
      int gid = pair->first;

      if (dis_->have_global_node(gid))
      {
        mymasters.push_back(gid);

        Core::Nodes::Node* master = dis_->g_node(gid);

        std::vector<int> masterdofs = dis_->dof(master);

        // and the 7th value is filled with the wall shear stress
        int lnodeid = dis_->node_row_map()->lid(gid);
        (mymasters_vec[0]).push_back(invec->local_values_as_span()[lnodeid]);
      }
      else
      {
        FOUR_C_THROW("master {} in midtosid but not on proc. This was unexpected", gid);
      }
    }

    // create an exporter for point to point communication
    Core::Communication::Exporter exporter(dis_->get_comm());

    // necessary variables
    MPI_Request request;

    // define send and receive blocks
    std::vector<char> sblock;
    std::vector<char> rblock;

    // get number of processors and the current processors id
    int numproc = Core::Communication::num_mpi_ranks(dis_->get_comm());

    //----------------------------------------------------------------------
    // communication is done in a round robin loop
    //----------------------------------------------------------------------
    for (int np = 0; np < numproc + 1; ++np)
    {
      // in the first step, we cannot receive anything
      if (np > 0)
      {
        receive_block(rblock, exporter, request);

        // Unpack info from the receive block from the last proc
        unpack_local_master_values(mymasters, mymasters_vec, rblock);
      }

      // in the last step, we keep everything on this proc
      if (np < numproc)
      {
        // -----------------------
        // do what we wanted to do
        set_values_available_on_this_proc(mymasters, mymasters_vec, outvec);

        // Pack info into block to send
        Core::Communication::PackBuffer data;
        pack_local_master_values(mymasters, mymasters_vec, data);
        swap(sblock, data());

        send_block(sblock, exporter, request);
      }
    }
  }
  return;
}  // Transfer (Nodal)

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | for all values available on the processor, do the final setting of the |
 | value                          (private)                    bk 09/14 |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void FLD::TransferTurbulentInflowConditionNodal::set_values_available_on_this_proc(
    std::vector<int>& mymasters, std::vector<std::vector<double>>& mymasters_vec,
    std::shared_ptr<Core::LinAlg::Vector<double>> outvec)
{
  for (unsigned nn = 0; nn < mymasters.size(); ++nn)
  {
    std::map<int, std::vector<int>>::iterator iter = midtosid_.find(mymasters[nn]);

    if (iter != midtosid_.end())
    {
      std::vector<int> myslaves(iter->second);

      for (std::vector<int>::iterator sid = myslaves.begin(); sid != myslaves.end(); ++sid)
      {
        // is this slave id on this proc?
        if (dis_->node_row_map()->my_gid(*sid))
          outvec->replace_global_value(*sid, mymasters_vec[0][nn]);
      }
    }
  }

  return;
}  // set_values_available_on_this_proc (Nodal)

FOUR_C_NAMESPACE_CLOSE
