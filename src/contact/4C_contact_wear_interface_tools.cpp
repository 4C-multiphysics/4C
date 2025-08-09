// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_defines.hpp"
#include "4C_contact_friction_node.hpp"
#include "4C_contact_input.hpp"
#include "4C_contact_integrator.hpp"
#include "4C_contact_selfcontact_binarytree.hpp"
#include "4C_contact_wear_interface.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_mortar_defines.hpp"
#include "4C_mortar_dofset.hpp"
#include "4C_mortar_element.hpp"
#include "4C_mortar_integrator.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 | Finite difference check for normal gap derivatives        farah 09/13|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::fd_check_gap_deriv()
{
  // FD checks only for serial case
  std::shared_ptr<Core::LinAlg::Map> snodefullmap = Core::LinAlg::allreduce_e_map(*snoderowmap_);
  std::shared_ptr<Core::LinAlg::Map> mnodefullmap = Core::LinAlg::allreduce_e_map(*mnoderowmap_);
  if (Core::Communication::num_mpi_ranks(get_comm()) > 1)
    FOUR_C_THROW("FD checks only for serial case");

  // create storage for gap values
  int nrow = snoderowmap_->num_my_elements();
  std::vector<double> refG(nrow);
  std::vector<double> newG(nrow);

  // problem dimension (2D or 3D)
  int dim = n_dim();

  // store reference
  // loop over proc's slave nodes
  for (int i = 0; i < snoderowmap_->num_my_elements(); ++i)
  {
    int gid = snoderowmap_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(node);

    // store gap-values into refG
    refG[i] = cnode->data().getg();
  }

  // global loop to apply FD scheme to all slave dofs (=dim*nodes)
  for (int fd = 0; fd < dim * snodefullmap->num_my_elements(); ++fd)
  {
    // store warnings for this finite difference
    int w = 0;

    // Initialize
    initialize();

    // now get the node we want to apply the FD scheme to
    int gid = snodefullmap->gid(fd / dim);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find slave node with gid %", gid);
    CONTACT::Node* snode = dynamic_cast<CONTACT::Node*>(node);

    int sdof = snode->dofs()[fd % dim];
    std::cout << "\nDERIVATIVE FOR S-NODE # " << gid << " DOF: " << sdof << std::endl;

    // apply finite difference scheme
    /*if (Core::Communication::my_mpi_rank(Comm())==snode->Owner())
    {
      std::cout << "\nBuilding FD for Slave Node: " << snode->Id() << " Dof(l): " << fd%dim
           << " Dof(g): " << snode->Dofs()[fd%dim] << std::endl;
    }*/

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd % dim == 0)
    {
      snode->xspatial()[0] += delta;
    }
    else if (fd % dim == 1)
    {
      snode->xspatial()[1] += delta;
    }
    else
    {
      snode->xspatial()[2] += delta;
    }

    // compute element areas
    set_element_areas();

    // *******************************************************************
    // contents of evaluate()
    // *******************************************************************
    evaluate();

    // compute finite difference derivative
    for (int k = 0; k < snoderowmap_->num_my_elements(); ++k)
    {
      int kgid = snoderowmap_->gid(k);
      Core::Nodes::Node* knode = idiscret_->g_node(kgid);
      if (!knode) FOUR_C_THROW("Cannot find node with gid %", kgid);
      CONTACT::Node* kcnode = dynamic_cast<CONTACT::Node*>(knode);

      // store gap-values into newG
      newG[k] = kcnode->data().getg();

      if (abs(newG[k] - refG[k]) > 1e-12 && newG[k] != 1.0e12 && refG[k] != 1.0e12)
      {
        double finit = (newG[k] - refG[k]) / delta;
        double analy = kcnode->data().get_deriv_g()[snode->dofs()[fd % dim]];
        double dev = finit - analy;

        // kgid: id of currently tested slave node
        // snode->Dofs()[fd%dim]: currently modified slave dof
        std::cout << "(" << kgid << "," << snode->dofs()[fd % dim] << ") : fd=" << finit
                  << " derivg=" << analy << " DEVIATION " << dev;

        if (abs(dev) > 1e-4)
        {
          std::cout << " ***** WARNING ***** ";
          w++;
        }
        else if (abs(dev) > 1e-5)
        {
          std::cout << " ***** warning ***** ";
          w++;
        }

        std::cout << std::endl;
      }
    }
    // undo finite difference modification
    if (fd % dim == 0)
    {
      snode->xspatial()[0] -= delta;
    }
    else if (fd % dim == 1)
    {
      snode->xspatial()[1] -= delta;
    }
    else
    {
      snode->xspatial()[2] -= delta;
    }

    std::cout << " ******************** GENERATED " << w << " WARNINGS ***************** "
              << std::endl;
  }

  // global loop to apply FD scheme to all master dofs (=dim*nodes)
  for (int fd = 0; fd < dim * mnodefullmap->num_my_elements(); ++fd)
  {
    // store warnings for this finite difference
    int w = 0;

    // Initialize
    // loop over all nodes to reset normals, closestnode and Mortar maps
    // (use fully overlapping column map)
    initialize();

    // now get the node we want to apply the FD scheme to
    int gid = mnodefullmap->gid(fd / dim);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find master node with gid %", gid);
    CONTACT::Node* mnode = dynamic_cast<CONTACT::Node*>(node);

    int mdof = mnode->dofs()[fd % dim];
    std::cout << "\nDERIVATIVE FOR M-NODE # " << gid << " DOF: " << mdof << std::endl;

    // apply finite difference scheme
    /*if (Core::Communication::my_mpi_rank(Comm())==mnode->Owner())
    {
      std::cout << "\nBuilding FD for Master Node: " << mnode->Id() << " Dof(l): " << fd%dim
           << " Dof(g): " << mnode->Dofs()[fd%dim] << std::endl;
    }*/

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd % dim == 0)
    {
      mnode->xspatial()[0] += delta;
    }
    else if (fd % dim == 1)
    {
      mnode->xspatial()[1] += delta;
    }
    else
    {
      mnode->xspatial()[2] += delta;
    }

    // compute element areas
    set_element_areas();

    // *******************************************************************
    // contents of evaluate()
    // *******************************************************************
    evaluate();

    // compute finite difference derivative
    for (int k = 0; k < snoderowmap_->num_my_elements(); ++k)
    {
      int kgid = snoderowmap_->gid(k);
      Core::Nodes::Node* knode = idiscret_->g_node(kgid);
      if (!knode) FOUR_C_THROW("Cannot find node with gid %", kgid);
      CONTACT::Node* kcnode = dynamic_cast<CONTACT::Node*>(knode);

      if (kcnode->active())
      {
        // check two versions of weighted gap

        std::map<int, double>& mmap = kcnode->mo_data().get_m();
        std::map<int, double>::const_iterator mcurr;

        for (int m = 0; m < mnodefullmap->num_my_elements(); ++m)
        {
          int gid = mnodefullmap->gid(m);
          Core::Nodes::Node* mnode = idiscret_->g_node(gid);
          if (!mnode) FOUR_C_THROW("Cannot find node with gid %", gid);
          CONTACT::Node* cmnode = dynamic_cast<CONTACT::Node*>(mnode);
          bool hasentry = false;

          // look for this master node in M-map of the active slave node
          for (mcurr = mmap.begin(); mcurr != mmap.end(); ++mcurr)
            if ((mcurr->first) == cmnode->id())
            {
              hasentry = true;
              break;
            }

          double mik = mmap[cmnode->id()];

          // get out of here, if master node not adjacent or coupling very weak
          if (!hasentry || abs(mik) < 1.0e-12) continue;
        }
      }

      // store gap-values into newG
      newG[k] = kcnode->data().getg();

      if (abs(newG[k] - refG[k]) > 1e-12 && newG[k] != 1.0e12 && refG[k] != 1.0e12)
      {
        double finit = (newG[k] - refG[k]) / delta;
        double analy = kcnode->data().get_deriv_g()[mnode->dofs()[fd % dim]];
        double dev = finit - analy;

        // kgid: id of currently tested slave node
        // mnode->Dofs()[fd%dim]: currently modified slave dof
        std::cout << "(" << kgid << "," << mnode->dofs()[fd % dim] << ") : fd=" << finit
                  << " derivg=" << analy << " DEVIATION " << dev;

        if (abs(dev) > 1e-4)
        {
          std::cout << " ***** WARNING ***** ";
          w++;
        }
        else if (abs(dev) > 1e-5)
        {
          std::cout << " ***** warning ***** ";
          w++;
        }

        std::cout << std::endl;
      }
    }

    // undo finite difference modification
    if (fd % dim == 0)
    {
      mnode->xspatial()[0] -= delta;
    }
    else if (fd % dim == 1)
    {
      mnode->xspatial()[1] -= delta;
    }
    else
    {
      mnode->xspatial()[2] -= delta;
    }

    std::cout << " ******************** GENERATED " << w << " WARNINGS ***************** "
              << std::endl;
  }

  // back to normal...
  initialize();
  evaluate();

  return;
}

/*----------------------------------------------------------------------*
 | Finite difference check for normal gap derivatives        farah 09/13|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::fd_check_gap_deriv_w()
{
  // FD checks only for serial case
  std::shared_ptr<Core::LinAlg::Map> snodefullmap = Core::LinAlg::allreduce_e_map(*snoderowmap_);
  std::shared_ptr<Core::LinAlg::Map> mnodefullmap = Core::LinAlg::allreduce_e_map(*mnoderowmap_);
  if (Core::Communication::num_mpi_ranks(get_comm()) > 1)
    FOUR_C_THROW("FD checks only for serial case");

  // create storage for gap values
  int nrow = snoderowmap_->num_my_elements();
  std::vector<double> refG(nrow);
  std::vector<double> newG(nrow);

  // problem dimension (2D or 3D)
  int dim = n_dim();

  // store reference
  // loop over proc's slave nodes
  for (int i = 0; i < snoderowmap_->num_my_elements(); ++i)
  {
    int gid = snoderowmap_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(node);

    // store gap-values into refG
    refG[i] = cnode->data().getg();
  }

  // global loop to apply FD scheme to all slave dofs (=dim*nodes)
  for (int fd = 0; fd < dim * snodefullmap->num_my_elements(); ++fd)
  {
    // store warnings for this finite difference
    int w = 0;

    // Initialize
    initialize();

    // now get the node we want to apply the FD scheme to
    int gid = snodefullmap->gid(fd / dim);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find slave node with gid %", gid);
    CONTACT::FriNode* snode = dynamic_cast<CONTACT::FriNode*>(node);

    int sdof = snode->dofs()[fd % dim];
    std::cout << "\nW --- DERIVATIVE FOR S-NODE # " << gid << " DOF: " << sdof << std::endl;

    double delta = 1e-8;
    if (fd % dim == 0) snode->wear_data().wcurr()[0] += delta;


    // compute element areas
    set_element_areas();

    // *******************************************************************
    // contents of evaluate()
    // *******************************************************************
    evaluate();

    // compute finite difference derivative
    for (int k = 0; k < snoderowmap_->num_my_elements(); ++k)
    {
      int kgid = snoderowmap_->gid(k);
      Core::Nodes::Node* knode = idiscret_->g_node(kgid);
      if (!knode) FOUR_C_THROW("Cannot find node with gid %", kgid);
      CONTACT::Node* kcnode = dynamic_cast<CONTACT::Node*>(knode);

      // store gap-values into newG
      newG[k] = kcnode->data().getg();

      if (abs(newG[k] - refG[k]) > 1e-12 && newG[k] != 1.0e12 && refG[k] != 1.0e12)
      {
        double finit = (newG[k] - refG[k]) / delta;
        double analy = kcnode->data().get_deriv_gw()[snode->dofs()[0]];
        double dev = finit - analy;

        // kgid: id of currently tested slave node
        // snode->Dofs()[fd%dim]: currently modified slave dof
        std::cout << "(" << kgid << "," << snode->dofs()[fd % dim] << ") : fd=" << finit
                  << " derivg=" << analy << " DEVIATION " << dev;

        if (abs(dev) > 1e-4)
        {
          std::cout << " ***** WARNING ***** ";
          w++;
        }
        else if (abs(dev) > 1e-5)
        {
          std::cout << " ***** warning ***** ";
          w++;
        }

        std::cout << std::endl;
      }
    }
    // undo finite difference modification
    if (fd % dim == 0) snode->wear_data().wcurr()[0] -= delta;


    std::cout << " ******************** GENERATED " << w << " WARNINGS ***************** "
              << std::endl;
  }

  // back to normal...
  initialize();
  evaluate();

  return;
}

/*----------------------------------------------------------------------*
 | Finite difference check of wear condition derivatives     farah 09/13|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::fd_check_deriv_e_d(Core::LinAlg::SparseMatrix& linedis)
{
  // FD checks only for serial case
  std::shared_ptr<Core::LinAlg::Map> snodefullmap = Core::LinAlg::allreduce_e_map(*snoderowmap_);
  std::shared_ptr<Core::LinAlg::Map> mnodefullmap = Core::LinAlg::allreduce_e_map(*mnoderowmap_);
  if (Core::Communication::num_mpi_ranks(get_comm()) > 1)
    FOUR_C_THROW("FD checks only for serial case");

  // create storage for values of complementary function C
  int nrow = snoderowmap_->num_my_elements();
  std::vector<double> reft(nrow);
  std::vector<double> newt(nrow);

  int dim = n_dim();
  using CI = std::map<int, double>::const_iterator;

  // store reference
  // loop over proc's slave nodes
  for (int i = 0; i < snoderowmap_->num_my_elements(); ++i)
  {
    reft[i] = 0.0;

    int gid = snoderowmap_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(node);

    if (cnode->wear_data().get_e().size() > 0)
    {
      std::map<int, double> scurr = cnode->wear_data().get_e()[0];

      for (CI p = scurr.begin(); p != scurr.end(); ++p)
      {
        int gid2 = (int)((p->first) / (dim));
        Core::Nodes::Node* node2 = idiscret_->g_node(gid2);
        if (!node2) FOUR_C_THROW("Cannot find node with gid %", gid2);
        CONTACT::FriNode* cnode2 = dynamic_cast<CONTACT::FriNode*>(node2);

        double w = cnode2->wear_data().wcurr()[0];

        reft[i] += (p->second) * w;
      }
    }
    else
    {
      reft[i] = 0.0;
    }
  }  // loop over procs slave nodes

  // ********************************************************************************************
  // global loop to apply FD scheme to all slave dofs (=3*nodes)
  // ********************************************************************************************
  for (int fd = 0; fd < dim * snodefullmap->num_my_elements(); ++fd)
  {
    // Initialize
    initialize();

    // now get the node we want to apply the FD scheme to
    int gid = snodefullmap->gid(fd / dim);
    int coldof = 0;
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find slave node with gid %", gid);
    CONTACT::FriNode* snode = dynamic_cast<CONTACT::FriNode*>(node);

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd % dim == 0)
    {
      snode->xspatial()[0] += delta;
      coldof = snode->dofs()[0];
    }
    else if (fd % dim == 1)
    {
      snode->xspatial()[1] += delta;
      coldof = snode->dofs()[1];
    }
    else
    {
      snode->xspatial()[2] += delta;
      coldof = snode->dofs()[2];
    }

    // compute element areas
    set_element_areas();

    // *******************************************************************
    // contents of evaluate()
    // *******************************************************************
    evaluate();

    // compute finite difference derivative
    for (int k = 0; k < snoderowmap_->num_my_elements(); ++k)
    {
      newt[k] = 0.0;

      int gid3 = snoderowmap_->gid(k);
      Core::Nodes::Node* node3 = idiscret_->g_node(gid3);
      if (!node3) FOUR_C_THROW("Cannot find node with gid %", gid3);
      CONTACT::FriNode* kcnode = dynamic_cast<CONTACT::FriNode*>(node3);

      if (kcnode->wear_data().get_e().size() > 0)
      {
        std::map<int, double> sscurr = kcnode->wear_data().get_e()[0];

        for (CI p = sscurr.begin(); p != sscurr.end(); ++p)
        {
          int gid2 = (int)((p->first) / (dim));
          Core::Nodes::Node* node2 = idiscret_->g_node(gid2);
          if (!node2) FOUR_C_THROW("Cannot find node with gid %", gid2);
          CONTACT::FriNode* cnode2 = dynamic_cast<CONTACT::FriNode*>(node2);

          double w = cnode2->wear_data().wcurr()[0];

          newt[k] += (p->second) * w;
        }
      }
      else
        newt[k] = 0.0;

      // ************************************************************************
      // Extract linearizations from sparse matrix !!!
      // ************************************************************************

      // ********************************* TXI
      double sparse_ij = 0.0;
      int sparsenumentries = 0;
      const int sparselength = linedis.num_global_entries(kcnode->dofs()[0]);
      std::vector<double> sparsevalues(sparselength);
      std::vector<int> sparseindices(sparselength);
      linedis.extract_global_row_copy(kcnode->dofs()[0], sparselength, sparsenumentries,
          sparsevalues.data(), sparseindices.data());

      for (int h = 0; h < sparselength; ++h)
      {
        if (sparseindices[h] == coldof)
        {
          sparse_ij = sparsevalues[h];
          break;
        }
        else
          sparse_ij = 0.0;
      }
      double analyt_txi = sparse_ij;


      // std::cout << "ref= " << reft[k] << "\t new= " << newt[k] << std::endl;

      if (abs(newt[k] - reft[k]) > 1e-12)
      {
        std::cout << "E WEAR DIS-Deriv: " << kcnode->dofs()[0]
                  << "\t w.r.t Slave: " << snode->dofs()[fd % dim]
                  << "\t FD= " << std::setprecision(4) << (newt[k] - reft[k]) / delta
                  << "\t analyt= " << std::setprecision(5) << analyt_txi
                  << "\t Error= " << analyt_txi - ((newt[k] - reft[k]) / delta);
        if (abs(analyt_txi - (newt[k] - reft[k]) / delta) > 1.0e-4)
          std::cout << "*** WARNING ***" << std::endl;
        else
          std::cout << " " << std::endl;
      }
    }
    // undo finite difference modification
    if (fd % dim == 0)
    {
      snode->xspatial()[0] -= delta;
    }
    else if (fd % dim == 1)
    {
      snode->xspatial()[1] -= delta;
    }
    else
    {
      snode->xspatial()[2] -= delta;
    }
  }  // loop over procs slave nodes

  // ********************************************************************************************
  // global loop to apply FD scheme to all master dofs (=3*nodes)
  // ********************************************************************************************
  for (int fd = 0; fd < dim * mnodefullmap->num_my_elements(); ++fd)
  {
    // Initialize
    initialize();

    // now get the node we want to apply the FD scheme to
    int gid = mnodefullmap->gid(fd / dim);
    int coldof = 0;
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find slave node with gid %", gid);
    CONTACT::FriNode* snode = dynamic_cast<CONTACT::FriNode*>(node);

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd % dim == 0)
    {
      snode->xspatial()[0] += delta;
      coldof = snode->dofs()[0];
    }
    else if (fd % dim == 1)
    {
      snode->xspatial()[1] += delta;
      coldof = snode->dofs()[1];
    }
    else
    {
      snode->xspatial()[2] += delta;
      coldof = snode->dofs()[2];
    }

    // compute element areas
    set_element_areas();

    // *******************************************************************
    // contents of evaluate()
    // *******************************************************************
    evaluate();

    // compute finite difference derivative
    for (int k = 0; k < snoderowmap_->num_my_elements(); ++k)
    {
      newt[k] = 0.0;

      int gid3 = snoderowmap_->gid(k);
      Core::Nodes::Node* node3 = idiscret_->g_node(gid3);
      if (!node3) FOUR_C_THROW("Cannot find node with gid %", gid3);
      CONTACT::FriNode* kcnode = dynamic_cast<CONTACT::FriNode*>(node3);

      if (kcnode->wear_data().get_e().size() > 0)
      {
        std::map<int, double> sscurr = kcnode->wear_data().get_e()[0];

        for (CI p = sscurr.begin(); p != sscurr.end(); ++p)
        {
          int gid2 = (int)((p->first) / (dim));
          Core::Nodes::Node* node2 = idiscret_->g_node(gid2);
          if (!node2) FOUR_C_THROW("Cannot find node with gid %", gid2);
          CONTACT::FriNode* cnode2 = dynamic_cast<CONTACT::FriNode*>(node2);

          double w = cnode2->wear_data().wcurr()[0];

          newt[k] += (p->second) * w;
        }
      }
      else
        newt[k] = 0.0;

      // ************************************************************************
      // Extract linearizations from sparse matrix !!!
      // ************************************************************************

      // ********************************* TXI
      double sparse_ij = 0.0;
      int sparsenumentries = 0;
      const int sparselength = linedis.num_global_entries(kcnode->dofs()[0]);
      std::vector<double> sparsevalues(sparselength);
      std::vector<int> sparseindices(sparselength);
      linedis.extract_global_row_copy(kcnode->dofs()[0], sparselength, sparsenumentries,
          sparsevalues.data(), sparseindices.data());

      for (int h = 0; h < sparselength; ++h)
      {
        if (sparseindices[h] == coldof)
        {
          sparse_ij = sparsevalues[h];
          break;
        }
        else
          sparse_ij = 0.0;
      }
      double analyt_txi = sparse_ij;


      // std::cout << "ref= " << reft[k] << "\t new= " << newt[k] << std::endl;

      if (abs(newt[k] - reft[k]) > 1e-12)
      {
        std::cout << "E WEAR DIS-Deriv: " << kcnode->dofs()[0]
                  << "\t w.r.t Master: " << snode->dofs()[fd % dim]
                  << "\t FD= " << std::setprecision(4) << (newt[k] - reft[k]) / delta
                  << "\t analyt= " << std::setprecision(5) << analyt_txi
                  << "\t Error= " << analyt_txi - ((newt[k] - reft[k]) / delta);
        if (abs(analyt_txi - (newt[k] - reft[k]) / delta) > 1.0e-4)
          std::cout << "*** WARNING ***" << std::endl;
        else
          std::cout << " " << std::endl;
      }
    }
    // undo finite difference modification
    if (fd % dim == 0)
    {
      snode->xspatial()[0] -= delta;
    }
    else if (fd % dim == 1)
    {
      snode->xspatial()[1] -= delta;
    }
    else
    {
      snode->xspatial()[2] -= delta;
    }
  }  // loop over procs slave nodes
  // back to normal...
  initialize();
  evaluate();

  return;
}

/*----------------------------------------------------------------------*
 | Finite difference check of wear condition derivatives     farah 11/13|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::fd_check_deriv_e_d_master(Core::LinAlg::SparseMatrix& linedis)
{
  // FD checks only for serial case
  std::shared_ptr<Core::LinAlg::Map> snodefullmap = Core::LinAlg::allreduce_e_map(*snoderowmap_);
  std::shared_ptr<Core::LinAlg::Map> mnodefullmap = Core::LinAlg::allreduce_e_map(*mnoderowmap_);
  if (Core::Communication::num_mpi_ranks(get_comm()) > 1)
    FOUR_C_THROW("FD checks only for serial case");

  // create storage for values of complementary function C
  int nrow = snoderowmap_->num_my_elements();
  std::vector<double> reft(nrow);
  std::vector<double> newt(nrow);

  int dim = n_dim();
  using CI = std::map<int, double>::const_iterator;

  // store reference
  // loop over proc's slave nodes
  for (int i = 0; i < mnoderowmap_->num_my_elements(); ++i)
  {
    reft[i] = 0.0;

    int gid = mnoderowmap_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(node);

    if (cnode->wear_data().get_e().size() > 0)
    {
      std::map<int, double> scurr = cnode->wear_data().get_e()[0];

      for (CI p = scurr.begin(); p != scurr.end(); ++p)
      {
        int gid2 = (int)((p->first) / (dim));
        Core::Nodes::Node* node2 = idiscret_->g_node(gid2);
        if (!node2) FOUR_C_THROW("Cannot find node with gid %", gid2);
        CONTACT::FriNode* cnode2 = dynamic_cast<CONTACT::FriNode*>(node2);

        double w = cnode2->wear_data().wcurr()[0];

        reft[i] += (p->second) * w;
      }
    }
    else
    {
      reft[i] = 0.0;
    }
  }  // loop over procs slave nodes

  // ********************************************************************************************
  // global loop to apply FD scheme to all slave dofs (=3*nodes)
  // ********************************************************************************************
  for (int fd = 0; fd < dim * snodefullmap->num_my_elements(); ++fd)
  {
    // Initialize
    initialize();

    // now get the node we want to apply the FD scheme to
    int gid = snodefullmap->gid(fd / dim);
    int coldof = 0;
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find slave node with gid %", gid);
    CONTACT::FriNode* snode = dynamic_cast<CONTACT::FriNode*>(node);

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd % dim == 0)
    {
      snode->xspatial()[0] += delta;
      coldof = snode->dofs()[0];
    }
    else if (fd % dim == 1)
    {
      snode->xspatial()[1] += delta;
      coldof = snode->dofs()[1];
    }
    else
    {
      snode->xspatial()[2] += delta;
      coldof = snode->dofs()[2];
    }

    // compute element areas
    set_element_areas();

    // *******************************************************************
    // contents of evaluate()
    // *******************************************************************
    evaluate();

    // compute finite difference derivative
    for (int k = 0; k < mnoderowmap_->num_my_elements(); ++k)
    {
      newt[k] = 0.0;

      int gid3 = mnoderowmap_->gid(k);
      Core::Nodes::Node* node3 = idiscret_->g_node(gid3);
      if (!node3) FOUR_C_THROW("Cannot find node with gid %", gid3);
      CONTACT::FriNode* kcnode = dynamic_cast<CONTACT::FriNode*>(node3);

      if (kcnode->wear_data().get_e().size() > 0)
      {
        std::map<int, double> sscurr = kcnode->wear_data().get_e()[0];

        for (CI p = sscurr.begin(); p != sscurr.end(); ++p)
        {
          int gid2 = (int)((p->first) / (dim));
          Core::Nodes::Node* node2 = idiscret_->g_node(gid2);
          if (!node2) FOUR_C_THROW("Cannot find node with gid %", gid2);
          CONTACT::FriNode* cnode2 = dynamic_cast<CONTACT::FriNode*>(node2);

          double w = cnode2->wear_data().wcurr()[0];

          newt[k] += (p->second) * w;
        }
      }
      else
        newt[k] = 0.0;

      // ************************************************************************
      // Extract linearizations from sparse matrix !!!
      // ************************************************************************

      // ********************************* TXI
      double sparse_ij = 0.0;
      int sparsenumentries = 0;
      const int sparselength = linedis.num_global_entries(kcnode->dofs()[0]);
      std::vector<double> sparsevalues(sparselength);
      std::vector<int> sparseindices(sparselength);
      linedis.extract_global_row_copy(kcnode->dofs()[0], sparselength, sparsenumentries,
          sparsevalues.data(), sparseindices.data());

      for (int h = 0; h < sparselength; ++h)
      {
        if (sparseindices[h] == coldof)
        {
          sparse_ij = sparsevalues[h];
          break;
        }
        else
          sparse_ij = 0.0;
      }
      double analyt_txi = sparse_ij;


      // std::cout << "ref= " << reft[k] << "\t new= " << newt[k] << std::endl;

      if (abs(newt[k] - reft[k]) > 1e-12)
      {
        std::cout << "E WEAR DIS-Deriv: " << kcnode->dofs()[0]
                  << "\t w.r.t Slave: " << snode->dofs()[fd % dim]
                  << "\t FD= " << std::setprecision(4) << (newt[k] - reft[k]) / delta
                  << "\t analyt= " << std::setprecision(5) << analyt_txi
                  << "\t Error= " << analyt_txi - ((newt[k] - reft[k]) / delta);
        if (abs(analyt_txi - (newt[k] - reft[k]) / delta) > 1.0e-4)
          std::cout << "*** WARNING ***" << std::endl;
        else
          std::cout << " " << std::endl;
      }
    }
    // undo finite difference modification
    if (fd % dim == 0)
    {
      snode->xspatial()[0] -= delta;
    }
    else if (fd % dim == 1)
    {
      snode->xspatial()[1] -= delta;
    }
    else
    {
      snode->xspatial()[2] -= delta;
    }
  }  // loop over procs slave nodes

  // ********************************************************************************************
  // global loop to apply FD scheme to all master dofs (=3*nodes)
  // ********************************************************************************************
  for (int fd = 0; fd < dim * mnodefullmap->num_my_elements(); ++fd)
  {
    // Initialize
    initialize();

    // now get the node we want to apply the FD scheme to
    int gid = mnodefullmap->gid(fd / dim);
    int coldof = 0;
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find slave node with gid %", gid);
    CONTACT::FriNode* snode = dynamic_cast<CONTACT::FriNode*>(node);

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd % dim == 0)
    {
      snode->xspatial()[0] += delta;
      coldof = snode->dofs()[0];
    }
    else if (fd % dim == 1)
    {
      snode->xspatial()[1] += delta;
      coldof = snode->dofs()[1];
    }
    else
    {
      snode->xspatial()[2] += delta;
      coldof = snode->dofs()[2];
    }

    // compute element areas
    set_element_areas();

    // *******************************************************************
    // contents of evaluate()
    // *******************************************************************
    evaluate();

    // compute finite difference derivative
    for (int k = 0; k < mnoderowmap_->num_my_elements(); ++k)
    {
      newt[k] = 0.0;

      int gid3 = mnoderowmap_->gid(k);
      Core::Nodes::Node* node3 = idiscret_->g_node(gid3);
      if (!node3) FOUR_C_THROW("Cannot find node with gid %", gid3);
      CONTACT::FriNode* kcnode = dynamic_cast<CONTACT::FriNode*>(node3);

      if (kcnode->wear_data().get_e().size() > 0)
      {
        std::map<int, double> sscurr = kcnode->wear_data().get_e()[0];

        for (CI p = sscurr.begin(); p != sscurr.end(); ++p)
        {
          int gid2 = (int)((p->first) / (dim));
          Core::Nodes::Node* node2 = idiscret_->g_node(gid2);
          if (!node2) FOUR_C_THROW("Cannot find node with gid %", gid2);
          CONTACT::FriNode* cnode2 = dynamic_cast<CONTACT::FriNode*>(node2);

          double w = cnode2->wear_data().wcurr()[0];

          newt[k] += (p->second) * w;
        }
      }
      else
        newt[k] = 0.0;

      // ************************************************************************
      // Extract linearizations from sparse matrix !!!
      // ************************************************************************

      // ********************************* TXI
      double sparse_ij = 0.0;
      int sparsenumentries = 0;
      const int sparselength = linedis.num_global_entries(kcnode->dofs()[0]);
      std::vector<double> sparsevalues(sparselength);
      std::vector<int> sparseindices(sparselength);
      linedis.extract_global_row_copy(kcnode->dofs()[0], sparselength, sparsenumentries,
          sparsevalues.data(), sparseindices.data());

      for (int h = 0; h < sparselength; ++h)
      {
        if (sparseindices[h] == coldof)
        {
          sparse_ij = sparsevalues[h];
          break;
        }
        else
          sparse_ij = 0.0;
      }
      double analyt_txi = sparse_ij;


      // std::cout << "ref= " << reft[k] << "\t new= " << newt[k] << std::endl;

      if (abs(newt[k] - reft[k]) > 1e-12)
      {
        std::cout << "E WEAR DIS-Deriv: " << kcnode->dofs()[0]
                  << "\t w.r.t Master: " << snode->dofs()[fd % dim]
                  << "\t FD= " << std::setprecision(4) << (newt[k] - reft[k]) / delta
                  << "\t analyt= " << std::setprecision(5) << analyt_txi
                  << "\t Error= " << analyt_txi - ((newt[k] - reft[k]) / delta);
        if (abs(analyt_txi - (newt[k] - reft[k]) / delta) > 1.0e-4)
          std::cout << "*** WARNING ***" << std::endl;
        else
          std::cout << " " << std::endl;
      }
    }
    // undo finite difference modification
    if (fd % dim == 0)
    {
      snode->xspatial()[0] -= delta;
    }
    else if (fd % dim == 1)
    {
      snode->xspatial()[1] -= delta;
    }
    else
    {
      snode->xspatial()[2] -= delta;
    }
  }  // loop over procs slave nodes
  // back to normal...
  initialize();
  evaluate();

  return;
}

/*----------------------------------------------------------------------*
 | Finite difference check of wear condition derivatives     farah 09/13|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::fd_check_deriv_t_d(Core::LinAlg::SparseMatrix& lintdis)
{
  // FD checks only for serial case
  std::shared_ptr<Core::LinAlg::Map> snodefullmap = Core::LinAlg::allreduce_e_map(*snoderowmap_);
  std::shared_ptr<Core::LinAlg::Map> mnodefullmap = Core::LinAlg::allreduce_e_map(*mnoderowmap_);
  if (Core::Communication::num_mpi_ranks(get_comm()) > 1)
    FOUR_C_THROW("FD checks only for serial case");

  // nothing to do if no slip nodes
  if (slipnodes_->num_my_elements() == 0) return;

  // create storage for values of complementary function C
  int nrow = slipnodes_->num_my_elements();
  std::vector<double> reft(nrow);
  std::vector<double> newt(nrow);

  int dim = n_dim();
  using CI = std::map<int, double>::const_iterator;

  // store reference
  // loop over proc's slave nodes
  for (int i = 0; i < slipnodes_->num_my_elements(); ++i)
  {
    reft[i] = 0.0;

    int gid = slipnodes_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(node);

    if (cnode->wear_data().get_t().size() > 0)
    {
      std::map<int, double> scurr = cnode->wear_data().get_t()[0];

      for (CI p = scurr.begin(); p != scurr.end(); ++p)
      {
        int gid2 = (int)((p->first) / (dim));
        Core::Nodes::Node* node2 = idiscret_->g_node(gid2);
        if (!node2) FOUR_C_THROW("Cannot find node with gid %", gid2);
        CONTACT::FriNode* cnode2 = dynamic_cast<CONTACT::FriNode*>(node2);

        double lmn = 0.0;
        for (int g = 0; g < dim; ++g)
          lmn += (cnode2->mo_data().n()[g]) * (cnode2->mo_data().lm()[g]);

        reft[i] += (p->second) * lmn;
      }
    }
    else
    {
      reft[i] = 0.0;
    }
  }  // loop over procs slave nodes

  // ********************************************************************************************
  // global loop to apply FD scheme to all slave dofs (=3*nodes)
  // ********************************************************************************************
  for (int fd = 0; fd < dim * snodefullmap->num_my_elements(); ++fd)
  {
    // Initialize
    initialize();

    // now get the node we want to apply the FD scheme to
    int gid = snodefullmap->gid(fd / dim);
    int coldof = 0;
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find slave node with gid %", gid);
    CONTACT::FriNode* snode = dynamic_cast<CONTACT::FriNode*>(node);

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd % dim == 0)
    {
      snode->xspatial()[0] += delta;
      coldof = snode->dofs()[0];
    }
    else if (fd % dim == 1)
    {
      snode->xspatial()[1] += delta;
      coldof = snode->dofs()[1];
    }
    else
    {
      snode->xspatial()[2] += delta;
      coldof = snode->dofs()[2];
    }

    // compute element areas
    set_element_areas();

    // *******************************************************************
    // contents of evaluate()
    // *******************************************************************
    evaluate();

    // compute finite difference derivative
    for (int k = 0; k < slipnodes_->num_my_elements(); ++k)
    {
      newt[k] = 0.0;

      int gid3 = slipnodes_->gid(k);
      Core::Nodes::Node* node3 = idiscret_->g_node(gid3);
      if (!node3) FOUR_C_THROW("Cannot find node with gid %", gid3);
      CONTACT::FriNode* kcnode = dynamic_cast<CONTACT::FriNode*>(node3);

      if (kcnode->wear_data().get_t().size() > 0)
      {
        std::map<int, double> sscurr = kcnode->wear_data().get_t()[0];

        for (CI p = sscurr.begin(); p != sscurr.end(); ++p)
        {
          int gid2 = (int)((p->first) / (dim));
          Core::Nodes::Node* node2 = idiscret_->g_node(gid2);
          if (!node2) FOUR_C_THROW("Cannot find node with gid %", gid2);
          CONTACT::FriNode* cnode2 = dynamic_cast<CONTACT::FriNode*>(node2);

          double lmn = 0.0;
          for (int g = 0; g < dim; ++g)
            lmn += (cnode2->mo_data().n()[g]) * (cnode2->mo_data().lm()[g]);

          newt[k] += (p->second) * lmn;
        }
      }
      else
        newt[k] = 0.0;

      // ************************************************************************
      // Extract linearizations from sparse matrix !!!
      // ************************************************************************

      // ********************************* TXI
      double sparse_ij = 0.0;
      int sparsenumentries = 0;
      const int sparselength = lintdis.num_global_entries(kcnode->dofs()[0]);
      std::vector<double> sparsevalues(sparselength);
      std::vector<int> sparseindices(sparselength);
      lintdis.extract_global_row_copy(kcnode->dofs()[0], sparselength, sparsenumentries,
          sparsevalues.data(), sparseindices.data());

      for (int h = 0; h < sparselength; ++h)
      {
        if (sparseindices[h] == coldof)
        {
          sparse_ij = sparsevalues[h];
          break;
        }
        else
          sparse_ij = 0.0;
      }
      double analyt_txi = sparse_ij;


      if (abs(newt[k] - reft[k]) > 1e-12)
      {
        std::cout << "WEAR DIS-Deriv: " << kcnode->dofs()[0]
                  << "\t w.r.t Slave: " << snode->dofs()[fd % dim]
                  << "\t FD= " << std::setprecision(4) << (newt[k] - reft[k]) / delta
                  << "\t analyt= " << std::setprecision(5) << analyt_txi
                  << "\t Error= " << analyt_txi - ((newt[k] - reft[k]) / delta);
        if (abs(analyt_txi - (newt[k] - reft[k]) / delta) > 1.0e-4)
          std::cout << "*** WARNING ***" << std::endl;
        else
          std::cout << " " << std::endl;
      }
    }
    // undo finite difference modification
    if (fd % dim == 0)
    {
      snode->xspatial()[0] -= delta;
    }
    else if (fd % dim == 1)
    {
      snode->xspatial()[1] -= delta;
    }
    else
    {
      snode->xspatial()[2] -= delta;
    }
  }  // loop over procs slave nodes

  // back to normal...
  initialize();
  evaluate();

  return;
}

/*----------------------------------------------------------------------*
 | Finite difference check of wear condition derivatives     farah 11/13|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::fd_check_deriv_t_d_master(Core::LinAlg::SparseMatrix& lintdis)
{
  // FD checks only for serial case
  std::shared_ptr<Core::LinAlg::Map> snodefullmap = Core::LinAlg::allreduce_e_map(*snoderowmap_);
  std::shared_ptr<Core::LinAlg::Map> mnodefullmap = Core::LinAlg::allreduce_e_map(*mnoderowmap_);
  if (Core::Communication::num_mpi_ranks(get_comm()) > 1)
    FOUR_C_THROW("FD checks only for serial case");

  // nothing to do if no slip nodes
  if (slipmasternodes_->num_my_elements() == 0) return;

  // create storage for values of complementary function C
  int nrow = slipmasternodes_->num_my_elements();
  std::vector<double> reft(nrow);
  std::vector<double> newt(nrow);

  int dim = n_dim();
  using CI = std::map<int, double>::const_iterator;

  // store reference
  // loop over proc's slave nodes
  for (int i = 0; i < slipmasternodes_->num_my_elements(); ++i)
  {
    reft[i] = 0.0;

    int gid = slipmasternodes_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(node);

    if (cnode->wear_data().get_t().size() > 0)
    {
      std::map<int, double> scurr = cnode->wear_data().get_t()[0];

      for (CI p = scurr.begin(); p != scurr.end(); ++p)
      {
        int gid2 = (int)((p->first) / (dim));
        Core::Nodes::Node* node2 = idiscret_->g_node(gid2);
        if (!node2) FOUR_C_THROW("Cannot find node with gid %", gid2);
        CONTACT::FriNode* cnode2 = dynamic_cast<CONTACT::FriNode*>(node2);

        double lmn = 0.0;
        for (int g = 0; g < dim; ++g)
          lmn += (cnode2->mo_data().n()[g]) * (cnode2->mo_data().lm()[g]);

        reft[i] += (p->second) * lmn;
      }
    }
    else
    {
      reft[i] = 0.0;
    }
  }  // loop over procs slave nodes

  // ********************************************************************************************
  // global loop to apply FD scheme to all slave dofs (=3*nodes)
  // ********************************************************************************************
  for (int fd = 0; fd < dim * snodefullmap->num_my_elements(); ++fd)
  {
    // Initialize
    initialize();

    // now get the node we want to apply the FD scheme to
    int gid = snodefullmap->gid(fd / dim);
    int coldof = 0;
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find slave node with gid %", gid);
    CONTACT::FriNode* snode = dynamic_cast<CONTACT::FriNode*>(node);

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd % dim == 0)
    {
      snode->xspatial()[0] += delta;
      coldof = snode->dofs()[0];
    }
    else if (fd % dim == 1)
    {
      snode->xspatial()[1] += delta;
      coldof = snode->dofs()[1];
    }
    else
    {
      snode->xspatial()[2] += delta;
      coldof = snode->dofs()[2];
    }

    // compute element areas
    set_element_areas();

    // *******************************************************************
    // contents of evaluate()
    // *******************************************************************
    evaluate();

    // compute finite difference derivative
    for (int k = 0; k < slipmasternodes_->num_my_elements(); ++k)
    {
      newt[k] = 0.0;

      int gid3 = slipmasternodes_->gid(k);
      Core::Nodes::Node* node3 = idiscret_->g_node(gid3);
      if (!node3) FOUR_C_THROW("Cannot find node with gid %", gid3);
      CONTACT::FriNode* kcnode = dynamic_cast<CONTACT::FriNode*>(node3);

      if (kcnode->wear_data().get_t().size() > 0)
      {
        std::map<int, double> sscurr = kcnode->wear_data().get_t()[0];

        for (CI p = sscurr.begin(); p != sscurr.end(); ++p)
        {
          int gid2 = (int)((p->first) / (dim));
          Core::Nodes::Node* node2 = idiscret_->g_node(gid2);
          if (!node2) FOUR_C_THROW("Cannot find node with gid %", gid2);
          CONTACT::FriNode* cnode2 = dynamic_cast<CONTACT::FriNode*>(node2);

          double lmn = 0.0;
          for (int g = 0; g < dim; ++g)
            lmn += (cnode2->mo_data().n()[g]) * (cnode2->mo_data().lm()[g]);

          newt[k] += (p->second) * lmn;
        }
      }
      else
        newt[k] = 0.0;

      // ************************************************************************
      // Extract linearizations from sparse matrix !!!
      // ************************************************************************

      // ********************************* TXI
      double sparse_ij = 0.0;
      int sparsenumentries = 0;
      const int sparselength = lintdis.num_global_entries(kcnode->dofs()[0]);
      std::vector<double> sparsevalues(sparselength);
      std::vector<int> sparseindices(sparselength);
      lintdis.extract_global_row_copy(kcnode->dofs()[0], sparselength, sparsenumentries,
          sparsevalues.data(), sparseindices.data());

      for (int h = 0; h < sparselength; ++h)
      {
        if (sparseindices[h] == coldof)
        {
          sparse_ij = sparsevalues[h];
          break;
        }
        else
          sparse_ij = 0.0;
      }
      double analyt_txi = sparse_ij;


      if (abs(newt[k] - reft[k]) > 1e-12)
      {
        std::cout << "WEAR DIS-Deriv: " << kcnode->dofs()[0]
                  << "\t w.r.t Slave: " << snode->dofs()[fd % dim]
                  << "\t FD= " << std::setprecision(4) << (newt[k] - reft[k]) / delta
                  << "\t analyt= " << std::setprecision(5) << analyt_txi
                  << "\t Error= " << analyt_txi - ((newt[k] - reft[k]) / delta);
        if (abs(analyt_txi - (newt[k] - reft[k]) / delta) > 1.0e-4)
          std::cout << "*** WARNING ***" << std::endl;
        else
          std::cout << " " << std::endl;
      }
    }
    // undo finite difference modification
    if (fd % dim == 0)
    {
      snode->xspatial()[0] -= delta;
    }
    else if (fd % dim == 1)
    {
      snode->xspatial()[1] -= delta;
    }
    else
    {
      snode->xspatial()[2] -= delta;
    }
  }  // loop over procs slave nodes

  // back to normal...
  initialize();
  evaluate();

  return;
}

/*----------------------------------------------------------------------*
 | Finite difference check of slip condition derivatives     farah 08/13|
 | Not for Wear Lin. or modifications concerning the compl.             |
 | fnc. !!! See flags CONSISTENTSTICK / CONSISTENTSLIP                  |
 *----------------------------------------------------------------------*/
void Wear::WearInterface::fd_check_slip_deriv(Core::LinAlg::SparseMatrix& linslipLMglobal,
    Core::LinAlg::SparseMatrix& linslipDISglobal, Core::LinAlg::SparseMatrix& linslipWglobal)
{
  // FD checks only for serial case
  std::shared_ptr<Core::LinAlg::Map> snodefullmap = Core::LinAlg::allreduce_e_map(*snoderowmap_);
  std::shared_ptr<Core::LinAlg::Map> mnodefullmap = Core::LinAlg::allreduce_e_map(*mnoderowmap_);
  if (Core::Communication::num_mpi_ranks(get_comm()) > 1)
    FOUR_C_THROW("FD checks only for serial case");

  // information from interface contact parameter list
  auto ftype = Teuchos::getIntegralValue<CONTACT::FrictionType>(interface_params(), "FRICTION");
  double frbound = interface_params().get<double>("FRBOUND");
  double frcoeff = interface_params().get<double>("FRCOEFF");
  double ct = interface_params().get<double>("SEMI_SMOOTH_CT");
  double cn = interface_params().get<double>("SEMI_SMOOTH_CN");

  // create storage for values of complementary function C
  int nrow = snoderowmap_->num_my_elements();
  std::vector<double> refCtxi(nrow);
  std::vector<double> refCteta(nrow);
  std::vector<double> newCtxi(nrow);
  std::vector<double> newCteta(nrow);

  int dim = n_dim();

  // store reference
  // loop over proc's slave nodes
  for (int i = 0; i < snoderowmap_->num_my_elements(); ++i)
  {
    int gid = snoderowmap_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(node);

    double jumptxi = 0;
    double jumpteta = 0;
    double ztxi = 0;
    double zteta = 0;
    double znor = 0;
    double euclidean = 0;

    if (cnode->fri_data().slip())
    {
      // calculate value of C-function
      double D = cnode->mo_data().get_d()[cnode->id()];
      double Dold = cnode->fri_data().get_d_old()[cnode->id()];

      for (int dim = 0; dim < cnode->num_dof(); ++dim)
      {
        jumptxi -= (cnode->data().txi()[dim]) * (D - Dold) * (cnode->xspatial()[dim]);
        jumpteta -= (cnode->data().teta()[dim]) * (D - Dold) * (cnode->xspatial()[dim]);
        ztxi += (cnode->data().txi()[dim]) * (cnode->mo_data().lm()[dim]);
        zteta += (cnode->data().teta()[dim]) * (cnode->mo_data().lm()[dim]);
        znor += (cnode->mo_data().n()[dim]) * (cnode->mo_data().lm()[dim]);
      }

      std::map<int, double>& mmap = cnode->mo_data().get_m();
      std::map<int, double>& mmapold = cnode->fri_data().get_m_old();

      std::map<int, double>::const_iterator colcurr;
      std::set<int> mnodes;

      for (colcurr = mmap.begin(); colcurr != mmap.end(); colcurr++) mnodes.insert(colcurr->first);

      for (colcurr = mmapold.begin(); colcurr != mmapold.end(); colcurr++)
        mnodes.insert(colcurr->first);

      std::set<int>::iterator mcurr;

      // loop over all master nodes (find adjacent ones to this slip node)
      for (mcurr = mnodes.begin(); mcurr != mnodes.end(); mcurr++)
      {
        int gid = *mcurr;
        Core::Nodes::Node* mnode = idiscret_->g_node(gid);
        if (!mnode) FOUR_C_THROW("Cannot find node with gid %", gid);
        CONTACT::FriNode* cmnode = dynamic_cast<CONTACT::FriNode*>(mnode);

        double mik = mmap[cmnode->id()];
        double mikold = mmapold[cmnode->id()];

        std::map<int, double>::iterator mcurr;

        for (int dim = 0; dim < cnode->num_dof(); ++dim)
        {
          jumptxi += (cnode->data().txi()[dim]) * (mik - mikold) * (cmnode->xspatial()[dim]);
          jumpteta += (cnode->data().teta()[dim]) * (mik - mikold) * (cmnode->xspatial()[dim]);
        }
      }  //  loop over master nodes

      // gp-wise slip !!!!!!!
      if (interface_params().get<bool>("GP_SLIP_INCR"))
      {
        jumptxi = cnode->fri_data().jump_var()[0];
        jumpteta = 0.0;

        if (n_dim() == 3) jumpteta = cnode->fri_data().jump_var()[1];
      }

      // evaluate euclidean norm ||vec(zt)+ct*vec(jump)||
      std::vector<double> sum1(n_dim() - 1, 0);
      sum1[0] = ztxi + ct * jumptxi;
      if (n_dim() == 3) sum1[1] = zteta + ct * jumpteta;
      if (n_dim() == 2) euclidean = abs(sum1[0]);
      if (n_dim() == 3) euclidean = sqrt(sum1[0] * sum1[0] + sum1[1] * sum1[1]);
    }  // if cnode == Slip

    // store C in vector
    if (ftype == CONTACT::FrictionType::tresca)
    {
      refCtxi[i] = euclidean * ztxi - frbound * (ztxi + ct * jumptxi);
      refCteta[i] = euclidean * zteta - frbound * (zteta + ct * jumpteta);
    }
    else if (ftype == CONTACT::FrictionType::coulomb)
    {
      refCtxi[i] = euclidean * ztxi - (frcoeff * znor) * (ztxi + ct * jumptxi);
      refCteta[i] = euclidean * zteta - (frcoeff * znor) * (zteta + ct * jumpteta);
    }
    else
      FOUR_C_THROW("Friction law is neither Tresca nor Coulomb");

    refCtxi[i] =
        euclidean * ztxi - (frcoeff * (znor - cn * cnode->data().getg())) * (ztxi + ct * jumptxi);
    refCteta[i] = euclidean * zteta -
                  (frcoeff * (znor - cn * cnode->data().getg())) * (zteta + ct * jumpteta);

  }  // loop over procs slave nodes

  // **********************************************************************************
  // global loop to apply FD scheme for LM to all slave dofs (=3*nodes)
  // **********************************************************************************
  for (int fd = 0; fd < dim * snodefullmap->num_my_elements(); ++fd)
  {
    // now get the node we want to apply the FD scheme to
    int gid = snodefullmap->gid(fd / dim);
    int coldof = 0;
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find slave node with gid %", gid);
    CONTACT::FriNode* snode = dynamic_cast<CONTACT::FriNode*>(node);

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd % dim == 0)
    {
      snode->mo_data().lm()[0] += delta;
      coldof = snode->dofs()[0];
    }
    else if (fd % dim == 1)
    {
      snode->mo_data().lm()[1] += delta;
      coldof = snode->dofs()[1];
    }
    else
    {
      snode->mo_data().lm()[2] += delta;
      coldof = snode->dofs()[2];
    }

    // compute finite difference derivative
    for (int k = 0; k < snoderowmap_->num_my_elements(); ++k)
    {
      int kgid = snoderowmap_->gid(k);
      Core::Nodes::Node* knode = idiscret_->g_node(kgid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", kgid);
      CONTACT::FriNode* kcnode = dynamic_cast<CONTACT::FriNode*>(knode);

      double jumptxi = 0;
      double jumpteta = 0;
      double ztxi = 0;
      double zteta = 0;
      double znor = 0;
      double euclidean = 0;

      if (kcnode->fri_data().slip())
      {
        // check two versions of weighted gap
        double D = kcnode->mo_data().get_d()[kcnode->id()];
        double Dold = kcnode->fri_data().get_d_old()[kcnode->id()];
        for (int dim = 0; dim < kcnode->num_dof(); ++dim)
        {
          jumptxi -= (kcnode->data().txi()[dim]) * (D - Dold) * (kcnode->xspatial()[dim]);
          jumpteta -= (kcnode->data().teta()[dim]) * (D - Dold) * (kcnode->xspatial()[dim]);
          ztxi += (kcnode->data().txi()[dim]) * (kcnode->mo_data().lm()[dim]);
          zteta += (kcnode->data().teta()[dim]) * (kcnode->mo_data().lm()[dim]);
          znor += (kcnode->mo_data().n()[dim]) * (kcnode->mo_data().lm()[dim]);
        }

        std::map<int, double> mmap = kcnode->mo_data().get_m();
        std::map<int, double> mmapold = kcnode->fri_data().get_m_old();

        std::map<int, double>::iterator colcurr;
        std::set<int> mnodes;

        for (colcurr = mmap.begin(); colcurr != mmap.end(); colcurr++)
          mnodes.insert(colcurr->first);

        for (colcurr = mmapold.begin(); colcurr != mmapold.end(); colcurr++)
          mnodes.insert(colcurr->first);

        std::set<int>::iterator mcurr;

        // loop over all master nodes (find adjacent ones to this stick node)
        for (mcurr = mnodes.begin(); mcurr != mnodes.end(); mcurr++)
        {
          int gid = *mcurr;
          Core::Nodes::Node* mnode = idiscret_->g_node(gid);
          if (!mnode) FOUR_C_THROW("Cannot find node with gid %", gid);
          CONTACT::FriNode* cmnode = dynamic_cast<CONTACT::FriNode*>(mnode);
          double mik = mmap[cmnode->id()];
          double mikold = mmapold[cmnode->id()];

          std::map<int, double>::iterator mcurr;

          for (int dim = 0; dim < kcnode->num_dof(); ++dim)
          {
            jumptxi += (kcnode->data().txi()[dim]) * (mik - mikold) * (cmnode->xspatial()[dim]);
            jumpteta += (kcnode->data().teta()[dim]) * (mik - mikold) * (cmnode->xspatial()[dim]);
          }
        }  //  loop over master nodes

        // gp-wise slip !!!!!!!
        if (interface_params().get<bool>("GP_SLIP_INCR"))
        {
          jumptxi = kcnode->fri_data().jump_var()[0];
          jumpteta = 0.0;

          if (n_dim() == 3) jumpteta = kcnode->fri_data().jump_var()[1];
        }

        // evaluate euclidean norm ||vec(zt)+ct*vec(jump)||
        std::vector<double> sum1(n_dim() - 1, 0);
        sum1[0] = ztxi + ct * jumptxi;
        if (n_dim() == 3) sum1[1] = zteta + ct * jumpteta;
        if (n_dim() == 2) euclidean = abs(sum1[0]);
        if (n_dim() == 3) euclidean = sqrt(sum1[0] * sum1[0] + sum1[1] * sum1[1]);
      }  // if cnode == Slip

      // store C in vector
      if (ftype == CONTACT::FrictionType::tresca)
      {
        newCtxi[k] = euclidean * ztxi - frbound * (ztxi + ct * jumptxi);
        newCteta[k] = euclidean * zteta - frbound * (zteta + ct * jumpteta);
      }
      else if (ftype == CONTACT::FrictionType::coulomb)
      {
        newCtxi[k] = euclidean * ztxi - (frcoeff * znor) * (ztxi + ct * jumptxi);
        newCteta[k] = euclidean * zteta - (frcoeff * znor) * (zteta + ct * jumpteta);
      }
      else
        FOUR_C_THROW("Friction law is neither Tresca nor Coulomb");

      newCtxi[k] = euclidean * ztxi -
                   (frcoeff * (znor - cn * kcnode->data().getg())) * (ztxi + ct * jumptxi);
      newCteta[k] = euclidean * zteta -
                    (frcoeff * (znor - cn * kcnode->data().getg())) * (zteta + ct * jumpteta);

      // ************************************************************************
      // Extract linearizations from sparse matrix !!!
      // ************************************************************************

      // ********************************* TXI
      double sparse_ij = 0.0;
      int sparsenumentries = 0;
      const int sparselength = linslipLMglobal.num_global_entries(kcnode->dofs()[1]);
      std::vector<double> sparsevalues(sparselength);
      std::vector<int> sparseindices(sparselength);
      linslipLMglobal.extract_global_row_copy(kcnode->dofs()[1], sparselength, sparsenumentries,
          sparsevalues.data(), sparseindices.data());

      for (int h = 0; h < sparselength; ++h)
      {
        if (sparseindices[h] == coldof)
        {
          sparse_ij = sparsevalues[h];
          break;
        }
        else
          sparse_ij = 0.0;
      }
      double analyt_txi = sparse_ij;

      // ********************************* TETA
      double sparse_2 = 0.0;
      int sparsenumentries2 = 0;
      const int sparselength2 = linslipLMglobal.num_global_entries(kcnode->dofs()[2]);
      std::vector<double> sparsevalues2(sparselength2);
      std::vector<int> sparseindices2(sparselength2);
      // int sparseextractionstatus =
      linslipLMglobal.extract_global_row_copy(kcnode->dofs()[2], sparselength2, sparsenumentries2,
          sparsevalues2.data(), sparseindices2.data());

      for (int h = 0; h < sparselength2; ++h)
      {
        if (sparseindices2[h] == coldof)
        {
          sparse_2 = sparsevalues2[h];
          break;
        }
        else
          sparse_2 = 0.0;
      }
      double analyt_teta = sparse_2;

      // print results (derivatives) to screen
      if (abs(newCtxi[k] - refCtxi[k]) > 1e-12)
      {
        std::cout << "SLIP LM-Deriv_xi: " << kcnode->id() << "\t w.r.t: " << snode->dofs()[fd % dim]
                  << "\t FD= " << std::setprecision(4) << (newCtxi[k] - refCtxi[k]) / delta
                  << "\t analyt= " << std::setprecision(4) << analyt_txi
                  << "\t Error= " << analyt_txi - ((newCtxi[k] - refCtxi[k]) / delta);
        if (abs(analyt_txi - (newCtxi[k] - refCtxi[k]) / delta) > 1.0e-4)
          std::cout << "*** WARNING ***" << std::endl;
        else
          std::cout << " " << std::endl;
      }

      // print results (derivatives) to screen
      if (abs(newCteta[k] - refCteta[k]) > 1e-12)
      {
        std::cout << "SLIP LM-Deriv_eta: " << kcnode->id()
                  << "\t w.r.t: " << snode->dofs()[fd % dim] << "\t FD= " << std::setprecision(4)
                  << (newCteta[k] - refCteta[k]) / delta << "\t analyt= " << std::setprecision(4)
                  << analyt_teta
                  << "\t Error= " << analyt_teta - ((newCteta[k] - refCteta[k]) / delta);
        if (abs(analyt_teta - (newCteta[k] - refCteta[k]) / delta) > 1.0e-4)
          std::cout << "*** WARNING ***" << std::endl;
        else
          std::cout << " " << std::endl;
      }
    }
    // undo finite difference modification
    if (fd % dim == 0)
    {
      snode->mo_data().lm()[0] -= delta;
    }
    else if (fd % dim == 1)
    {
      snode->mo_data().lm()[1] -= delta;
    }
    else
    {
      snode->mo_data().lm()[2] -= delta;
    }
  }  // loop over procs slave nodes


  // ********************************************************************************************
  // global loop to apply FD scheme to all slave dofs (=3*nodes)
  // ********************************************************************************************
  for (int fd = 0; fd < dim * snodefullmap->num_my_elements(); ++fd)
  {
    // Initialize
    initialize();

    // now get the node we want to apply the FD scheme to
    int gid = snodefullmap->gid(fd / dim);
    int coldof = 0;
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find slave node with gid %", gid);
    CONTACT::FriNode* snode = dynamic_cast<CONTACT::FriNode*>(node);

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd % dim == 0)
    {
      snode->xspatial()[0] += delta;
      coldof = snode->dofs()[0];
    }
    else if (fd % dim == 1)
    {
      snode->xspatial()[1] += delta;
      coldof = snode->dofs()[1];
    }
    else
    {
      snode->xspatial()[2] += delta;
      coldof = snode->dofs()[2];
    }

    // compute element areas
    set_element_areas();

    // *******************************************************************
    // contents of evaluate()
    // *******************************************************************
    evaluate();

    // compute finite difference derivative
    for (int k = 0; k < snoderowmap_->num_my_elements(); ++k)
    {
      int kgid = snoderowmap_->gid(k);
      Core::Nodes::Node* knode = idiscret_->g_node(kgid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", kgid);
      CONTACT::FriNode* kcnode = dynamic_cast<CONTACT::FriNode*>(knode);

      double jumptxi = 0;
      double jumpteta = 0;
      double ztxi = 0;
      double zteta = 0;
      double znor = 0;
      double euclidean = 0;

      if (kcnode->fri_data().slip())
      {
        // check two versions of weighted gap
        double D = kcnode->mo_data().get_d()[kcnode->id()];
        double Dold = kcnode->fri_data().get_d_old()[kcnode->id()];

        for (int dim = 0; dim < kcnode->num_dof(); ++dim)
        {
          jumptxi -= (kcnode->data().txi()[dim]) * (D - Dold) * (kcnode->xspatial()[dim]);
          jumpteta -= (kcnode->data().teta()[dim]) * (D - Dold) * (kcnode->xspatial()[dim]);
          ztxi += (kcnode->data().txi()[dim]) * (kcnode->mo_data().lm()[dim]);
          zteta += (kcnode->data().teta()[dim]) * (kcnode->mo_data().lm()[dim]);
          znor += (kcnode->mo_data().n()[dim]) * (kcnode->mo_data().lm()[dim]);
        }

        std::map<int, double> mmap = kcnode->mo_data().get_m();
        std::map<int, double> mmapold = kcnode->fri_data().get_m_old();

        std::map<int, double>::iterator colcurr;
        std::set<int> mnodes;

        for (colcurr = mmap.begin(); colcurr != mmap.end(); colcurr++)
          mnodes.insert(colcurr->first);

        for (colcurr = mmapold.begin(); colcurr != mmapold.end(); colcurr++)
          mnodes.insert(colcurr->first);

        std::set<int>::iterator mcurr;

        // loop over all master nodes (find adjacent ones to this stick node)
        for (mcurr = mnodes.begin(); mcurr != mnodes.end(); mcurr++)
        {
          int gid = *mcurr;
          Core::Nodes::Node* mnode = idiscret_->g_node(gid);
          if (!mnode) FOUR_C_THROW("Cannot find node with gid %", gid);
          CONTACT::FriNode* cmnode = dynamic_cast<CONTACT::FriNode*>(mnode);

          double mik = mmap[cmnode->id()];
          double mikold = mmapold[cmnode->id()];

          std::map<int, double>::iterator mcurr;

          for (int dim = 0; dim < kcnode->num_dof(); ++dim)
          {
            jumptxi += (kcnode->data().txi()[dim]) * (mik - mikold) * (cmnode->xspatial()[dim]);
            jumpteta += (kcnode->data().teta()[dim]) * (mik - mikold) * (cmnode->xspatial()[dim]);
          }
        }  //  loop over master nodes

        // gp-wise slip !!!!!!!
        if (interface_params().get<bool>("GP_SLIP_INCR"))
        {
          jumptxi = kcnode->fri_data().jump_var()[0];
          jumpteta = 0.0;

          if (n_dim() == 3) jumpteta = kcnode->fri_data().jump_var()[1];
        }

        // evaluate euclidean norm ||vec(zt)+ct*vec(jump)||
        std::vector<double> sum1(n_dim() - 1, 0);
        sum1[0] = ztxi + ct * jumptxi;
        if (n_dim() == 3) sum1[1] = zteta + ct * jumpteta;
        if (n_dim() == 2) euclidean = abs(sum1[0]);
        if (n_dim() == 3) euclidean = sqrt(sum1[0] * sum1[0] + sum1[1] * sum1[1]);

      }  // if cnode == Slip

      // store C in vector
      if (ftype == CONTACT::FrictionType::tresca)
      {
        newCtxi[k] = euclidean * ztxi - frbound * (ztxi + ct * jumptxi);
        newCteta[k] = euclidean * zteta - frbound * (zteta + ct * jumpteta);
      }
      else if (ftype == CONTACT::FrictionType::coulomb)
      {
        newCtxi[k] = euclidean * ztxi - (frcoeff * znor) * (ztxi + ct * jumptxi);
        newCteta[k] = euclidean * zteta - (frcoeff * znor) * (zteta + ct * jumpteta);
      }
      else
        FOUR_C_THROW("Friction law is neither Tresca nor Coulomb");

      newCtxi[k] = euclidean * ztxi -
                   (frcoeff * (znor - cn * kcnode->data().getg())) * (ztxi + ct * jumptxi);
      newCteta[k] = euclidean * zteta -
                    (frcoeff * (znor - cn * kcnode->data().getg())) * (zteta + ct * jumpteta);



      // ************************************************************************
      // Extract linearizations from sparse matrix !!!
      // ************************************************************************

      // ********************************* TXI
      double sparse_ij = 0.0;
      int sparsenumentries = 0;
      const int sparselength = linslipDISglobal.num_global_entries(kcnode->dofs()[1]);
      std::vector<double> sparsevalues(sparselength);
      std::vector<int> sparseindices(sparselength);
      linslipDISglobal.extract_global_row_copy(kcnode->dofs()[1], sparselength, sparsenumentries,
          sparsevalues.data(), sparseindices.data());

      for (int h = 0; h < sparselength; ++h)
      {
        if (sparseindices[h] == coldof)
        {
          sparse_ij = sparsevalues[h];
          break;
        }
        else
          sparse_ij = 0.0;
      }
      double analyt_txi = sparse_ij;

      // ********************************* TETA
      double sparse_2 = 0.0;
      int sparsenumentries2 = 0;
      const int sparselength2 = linslipDISglobal.num_global_entries(kcnode->dofs()[2]);
      std::vector<double> sparsevalues2(sparselength2);
      std::vector<int> sparseindices2(sparselength2);
      linslipDISglobal.extract_global_row_copy(kcnode->dofs()[2], sparselength2, sparsenumentries2,
          sparsevalues2.data(), sparseindices2.data());

      for (int h = 0; h < sparselength2; ++h)
      {
        if (sparseindices2[h] == coldof)
        {
          sparse_2 = sparsevalues2[h];
          break;
        }
        else
          sparse_2 = 0.0;
      }
      double analyt_teta = sparse_2;


      if (abs(newCtxi[k] - refCtxi[k]) > 1e-12)
      {
        std::cout << "SLIP DIS-Deriv_xi: " << kcnode->id()
                  << "\t w.r.t Slave: " << snode->dofs()[fd % dim]
                  << "\t FD= " << std::setprecision(4) << (newCtxi[k] - refCtxi[k]) / delta
                  << "\t analyt= " << std::setprecision(5) << analyt_txi
                  << "\t Error= " << analyt_txi - ((newCtxi[k] - refCtxi[k]) / delta);
        if (abs(analyt_txi - (newCtxi[k] - refCtxi[k]) / delta) > 1.0e-4)
          std::cout << "*** WARNING ***" << std::endl;
        else
          std::cout << " " << std::endl;
      }

      // print results (derivatives) to screen
      if (abs(newCteta[k] - refCteta[k]) > 1e-12)
      {
        std::cout << "SLIP DIS-Deriv_eta: " << kcnode->id()
                  << "\t w.r.t Slave: " << snode->dofs()[fd % dim]
                  << "\t FD= " << std::setprecision(4) << (newCteta[k] - refCteta[k]) / delta
                  << "\t analyt= " << std::setprecision(5) << analyt_teta
                  << "\t Error= " << analyt_teta - ((newCteta[k] - refCteta[k]) / delta);
        if (abs(analyt_teta - (newCteta[k] - refCteta[k]) / delta) > 1.0e-4)
          std::cout << "*** WARNING ***" << std::endl;
        else
          std::cout << " " << std::endl;
      }
    }
    // undo finite difference modification
    if (fd % dim == 0)
    {
      snode->xspatial()[0] -= delta;
    }
    else if (fd % dim == 1)
    {
      snode->xspatial()[1] -= delta;
    }
    else
    {
      snode->xspatial()[2] -= delta;
    }
  }  // loop over procs slave nodes

  // ********************************************************************************************
  // global loop to apply FD scheme to all master dofs (=3*nodes)
  // ********************************************************************************************
  for (int fd = 0; fd < dim * mnodefullmap->num_my_elements(); ++fd)
  {
    // Initialize
    initialize();

    // now get the node we want to apply the FD scheme to
    int gid = mnodefullmap->gid(fd / dim);
    int coldof = 0;
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find master node with gid %", gid);
    CONTACT::FriNode* mnode = dynamic_cast<CONTACT::FriNode*>(node);

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd % dim == 0)
    {
      mnode->xspatial()[0] += delta;
      coldof = mnode->dofs()[0];
    }
    else if (fd % dim == 1)
    {
      mnode->xspatial()[1] += delta;
      coldof = mnode->dofs()[1];
    }
    else
    {
      mnode->xspatial()[2] += delta;
      coldof = mnode->dofs()[2];
    }

    // compute element areas
    set_element_areas();

    // *******************************************************************
    // contents of evaluate()
    // *******************************************************************
    evaluate();

    // compute finite difference derivative
    for (int k = 0; k < snoderowmap_->num_my_elements(); ++k)
    {
      int kgid = snoderowmap_->gid(k);
      Core::Nodes::Node* knode = idiscret_->g_node(kgid);
      if (!knode) FOUR_C_THROW("Cannot find node with gid %", kgid);
      CONTACT::FriNode* kcnode = dynamic_cast<CONTACT::FriNode*>(knode);

      double jumptxi = 0;
      double jumpteta = 0;
      double ztxi = 0;
      double zteta = 0;
      double znor = 0;
      double euclidean = 0;

      if (kcnode->fri_data().slip())
      {
        // check two versions of weighted gap
        double D = kcnode->mo_data().get_d()[kcnode->id()];
        double Dold = kcnode->fri_data().get_d_old()[kcnode->id()];

        for (int dim = 0; dim < kcnode->num_dof(); ++dim)
        {
          jumptxi -= (kcnode->data().txi()[dim]) * (D - Dold) * (kcnode->xspatial()[dim]);
          jumpteta -= (kcnode->data().teta()[dim]) * (D - Dold) * (kcnode->xspatial()[dim]);
          ztxi += (kcnode->data().txi()[dim]) * (kcnode->mo_data().lm()[dim]);
          zteta += (kcnode->data().teta()[dim]) * (kcnode->mo_data().lm()[dim]);
          znor += (kcnode->mo_data().n()[dim]) * (kcnode->mo_data().lm()[dim]);
        }

        std::map<int, double> mmap = kcnode->mo_data().get_m();
        std::map<int, double> mmapold = kcnode->fri_data().get_m_old();

        std::map<int, double>::iterator colcurr;
        std::set<int> mnodes;

        for (colcurr = mmap.begin(); colcurr != mmap.end(); colcurr++)
          mnodes.insert(colcurr->first);

        for (colcurr = mmapold.begin(); colcurr != mmapold.end(); colcurr++)
          mnodes.insert(colcurr->first);

        std::set<int>::iterator mcurr;

        // loop over all master nodes (find adjacent ones to this stick node)
        for (mcurr = mnodes.begin(); mcurr != mnodes.end(); mcurr++)
        {
          int gid = *mcurr;
          Core::Nodes::Node* mnode = idiscret_->g_node(gid);
          if (!mnode) FOUR_C_THROW("Cannot find node with gid %", gid);
          CONTACT::FriNode* cmnode = dynamic_cast<CONTACT::FriNode*>(mnode);

          double mik = mmap[cmnode->id()];
          double mikold = mmapold[cmnode->id()];

          std::map<int, double>::iterator mcurr;

          for (int dim = 0; dim < kcnode->num_dof(); ++dim)
          {
            jumptxi += (kcnode->data().txi()[dim]) * (mik - mikold) * (cmnode->xspatial()[dim]);
            jumpteta += (kcnode->data().teta()[dim]) * (mik - mikold) * (cmnode->xspatial()[dim]);
          }
        }  //  loop over master nodes

        // gp-wise slip !!!!!!!
        if (interface_params().get<bool>("GP_SLIP_INCR"))
        {
          jumptxi = kcnode->fri_data().jump_var()[0];
          jumpteta = 0.0;

          if (n_dim() == 3) jumpteta = kcnode->fri_data().jump_var()[1];
        }

        // evaluate euclidean norm ||vec(zt)+ct*vec(jump)||
        std::vector<double> sum1(n_dim() - 1, 0);
        sum1[0] = ztxi + ct * jumptxi;
        if (n_dim() == 3) sum1[1] = zteta + ct * jumpteta;
        if (n_dim() == 2) euclidean = abs(sum1[0]);
        if (n_dim() == 3) euclidean = sqrt(sum1[0] * sum1[0] + sum1[1] * sum1[1]);

      }  // if cnode == Slip

      // store C in vector
      if (ftype == CONTACT::FrictionType::tresca)
      {
        newCtxi[k] = euclidean * ztxi - frbound * (ztxi + ct * jumptxi);
        newCteta[k] = euclidean * zteta - frbound * (zteta + ct * jumpteta);
      }
      else if (ftype == CONTACT::FrictionType::coulomb)
      {
        newCtxi[k] = euclidean * ztxi - (frcoeff * znor) * (ztxi + ct * jumptxi);
        newCteta[k] = euclidean * zteta - (frcoeff * znor) * (zteta + ct * jumpteta);
      }
      else
        FOUR_C_THROW("Friction law is neither Tresca nor Coulomb");

      newCtxi[k] = euclidean * ztxi -
                   (frcoeff * (znor - cn * kcnode->data().getg())) * (ztxi + ct * jumptxi);
      newCteta[k] = euclidean * zteta -
                    (frcoeff * (znor - cn * kcnode->data().getg())) * (zteta + ct * jumpteta);



      // ************************************************************************
      // Extract linearizations from sparse matrix !!!
      // ************************************************************************

      // ********************************* TXI
      double sparse_ij = 0.0;
      int sparsenumentries = 0;
      const int sparselength = linslipDISglobal.num_global_entries(kcnode->dofs()[1]);
      std::vector<double> sparsevalues(sparselength);
      std::vector<int> sparseindices(sparselength);
      linslipDISglobal.extract_global_row_copy(kcnode->dofs()[1], sparselength, sparsenumentries,
          sparsevalues.data(), sparseindices.data());

      for (int h = 0; h < sparselength; ++h)
      {
        if (sparseindices[h] == coldof)
        {
          sparse_ij = sparsevalues[h];
          break;
        }
        else
          sparse_ij = 0.0;
      }
      double analyt_txi = sparse_ij;

      // ********************************* TETA
      double sparse_2 = 0.0;
      int sparsenumentries2 = 0;
      const int sparselength2 = linslipDISglobal.num_global_entries(kcnode->dofs()[2]);
      std::vector<double> sparsevalues2(sparselength2);
      std::vector<int> sparseindices2(sparselength2);
      linslipDISglobal.extract_global_row_copy(kcnode->dofs()[2], sparselength2, sparsenumentries2,
          sparsevalues2.data(), sparseindices2.data());

      for (int h = 0; h < sparselength2; ++h)
      {
        if (sparseindices2[h] == coldof)
        {
          sparse_2 = sparsevalues2[h];
          break;
        }
        else
          sparse_2 = 0.0;
      }
      double analyt_teta = sparse_2;

      // print results (derivatives) to screen
      if (abs(newCtxi[k] - refCtxi[k]) > 1e-12)
      {
        std::cout << "SLIP DIS-Deriv_xi: " << kcnode->id()
                  << "\t w.r.t Master: " << mnode->dofs()[fd % dim]
                  << "\t FD= " << std::setprecision(4) << (newCtxi[k] - refCtxi[k]) / delta
                  << "\t analyt= " << std::setprecision(5) << analyt_txi
                  << "\t Error= " << analyt_txi - ((newCtxi[k] - refCtxi[k]) / delta);
        if (abs(analyt_txi - (newCtxi[k] - refCtxi[k]) / delta) > 1.0e-4)
          std::cout << "*** WARNING ***" << std::endl;
        else
          std::cout << " " << std::endl;
      }

      if (abs(newCteta[k] - refCteta[k]) > 1e-12)
      {
        std::cout << "SLIP DIS-Deriv_eta: " << kcnode->id()
                  << "\t w.r.t Master: " << mnode->dofs()[fd % dim]
                  << "\t FD= " << std::setprecision(4) << (newCteta[k] - refCteta[k]) / delta
                  << "\t analyt= " << std::setprecision(5) << analyt_teta
                  << "\t Error= " << analyt_teta - ((newCteta[k] - refCteta[k]) / delta);
        if (abs(analyt_teta - (newCteta[k] - refCteta[k]) / delta) > 1.0e-4)
          std::cout << "*** WARNING ***" << std::endl;
        else
          std::cout << " " << std::endl;
      }
    }

    // undo finite difference modification
    if (fd % dim == 0)
    {
      mnode->xspatial()[0] -= delta;
    }
    else if (fd % dim == 1)
    {
      mnode->xspatial()[1] -= delta;
    }
    else
    {
      mnode->xspatial()[2] -= delta;
    }
  }
  // ********************************************************************************************
  // global loop to apply FD scheme to all Wear dofs (=1*nodes)
  // ********************************************************************************************
  for (int fd = 0; fd < dim * snodefullmap->num_my_elements(); ++fd)
  {
    // Initialize
    initialize();

    // now get the node we want to apply the FD scheme to
    int gid = snodefullmap->gid(fd / dim);
    int coldof = 0;
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find master node with gid %", gid);
    CONTACT::FriNode* mnode = dynamic_cast<CONTACT::FriNode*>(node);

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd % dim == 0)
    {
      mnode->wear_data().wcurr()[0] += delta;
      coldof = mnode->dofs()[0];
    }

    // compute element areas
    set_element_areas();

    // *******************************************************************
    // contents of evaluate()
    // *******************************************************************
    evaluate();

    // compute finite difference derivative
    for (int k = 0; k < snoderowmap_->num_my_elements(); ++k)
    {
      int kgid = snoderowmap_->gid(k);
      Core::Nodes::Node* knode = idiscret_->g_node(kgid);
      if (!knode) FOUR_C_THROW("Cannot find node with gid %", kgid);
      CONTACT::FriNode* kcnode = dynamic_cast<CONTACT::FriNode*>(knode);

      double jumptxi = 0;
      double jumpteta = 0;
      double ztxi = 0;
      double zteta = 0;
      double znor = 0;
      double euclidean = 0;

      if (kcnode->fri_data().slip())
      {
        // check two versions of weighted gap
        double D = kcnode->mo_data().get_d()[kcnode->id()];
        double Dold = kcnode->fri_data().get_d_old()[kcnode->id()];

        for (int dim = 0; dim < kcnode->num_dof(); ++dim)
        {
          jumptxi -= (kcnode->data().txi()[dim]) * (D - Dold) * (kcnode->xspatial()[dim]);
          jumpteta -= (kcnode->data().teta()[dim]) * (D - Dold) * (kcnode->xspatial()[dim]);
          ztxi += (kcnode->data().txi()[dim]) * (kcnode->mo_data().lm()[dim]);
          zteta += (kcnode->data().teta()[dim]) * (kcnode->mo_data().lm()[dim]);
          znor += (kcnode->mo_data().n()[dim]) * (kcnode->mo_data().lm()[dim]);
        }

        std::map<int, double> mmap = kcnode->mo_data().get_m();
        std::map<int, double> mmapold = kcnode->fri_data().get_m_old();

        std::map<int, double>::iterator colcurr;
        std::set<int> mnodes;

        for (colcurr = mmap.begin(); colcurr != mmap.end(); colcurr++)
          mnodes.insert(colcurr->first);

        for (colcurr = mmapold.begin(); colcurr != mmapold.end(); colcurr++)
          mnodes.insert(colcurr->first);

        std::set<int>::iterator mcurr;

        // loop over all master nodes (find adjacent ones to this stick node)
        for (mcurr = mnodes.begin(); mcurr != mnodes.end(); mcurr++)
        {
          int gid = *mcurr;
          Core::Nodes::Node* mnode = idiscret_->g_node(gid);
          if (!mnode) FOUR_C_THROW("Cannot find node with gid %", gid);
          CONTACT::FriNode* cmnode = dynamic_cast<CONTACT::FriNode*>(mnode);

          double mik = mmap[cmnode->id()];
          double mikold = mmapold[cmnode->id()];

          std::map<int, double>::iterator mcurr;

          for (int dim = 0; dim < kcnode->num_dof(); ++dim)
          {
            jumptxi += (kcnode->data().txi()[dim]) * (mik - mikold) * (cmnode->xspatial()[dim]);
            jumpteta += (kcnode->data().teta()[dim]) * (mik - mikold) * (cmnode->xspatial()[dim]);
          }
        }  //  loop over master nodes

        // gp-wise slip !!!!!!!
        if (interface_params().get<bool>("GP_SLIP_INCR"))
        {
          jumptxi = kcnode->fri_data().jump_var()[0];
          jumpteta = 0.0;

          if (n_dim() == 3) jumpteta = kcnode->fri_data().jump_var()[1];
        }

        // evaluate euclidean norm ||vec(zt)+ct*vec(jump)||
        std::vector<double> sum1(n_dim() - 1, 0);
        sum1[0] = ztxi + ct * jumptxi;
        if (n_dim() == 3) sum1[1] = zteta + ct * jumpteta;
        if (n_dim() == 2) euclidean = abs(sum1[0]);
        if (n_dim() == 3) euclidean = sqrt(sum1[0] * sum1[0] + sum1[1] * sum1[1]);

      }  // if cnode == Slip

      // store C in vector
      if (ftype == CONTACT::FrictionType::tresca)
      {
        newCtxi[k] = euclidean * ztxi - frbound * (ztxi + ct * jumptxi);
        newCteta[k] = euclidean * zteta - frbound * (zteta + ct * jumpteta);
      }
      else if (ftype == CONTACT::FrictionType::coulomb)
      {
        newCtxi[k] = euclidean * ztxi - (frcoeff * znor) * (ztxi + ct * jumptxi);
        newCteta[k] = euclidean * zteta - (frcoeff * znor) * (zteta + ct * jumpteta);
      }
      else
        FOUR_C_THROW("Friction law is neither Tresca nor Coulomb");

      newCtxi[k] = euclidean * ztxi -
                   (frcoeff * (znor - cn * kcnode->data().getg())) * (ztxi + ct * jumptxi);
      newCteta[k] = euclidean * zteta -
                    (frcoeff * (znor - cn * kcnode->data().getg())) * (zteta + ct * jumpteta);


      // ************************************************************************
      // Extract linearizations from sparse matrix !!!
      // ************************************************************************

      // ********************************* TXI
      double sparse_ij = 0.0;
      int sparsenumentries = 0;
      const int sparselength = linslipWglobal.num_global_entries(kcnode->dofs()[1]);
      std::vector<double> sparsevalues(sparselength);
      std::vector<int> sparseindices(sparselength);
      linslipWglobal.extract_global_row_copy(kcnode->dofs()[1], sparselength, sparsenumentries,
          sparsevalues.data(), sparseindices.data());

      for (int h = 0; h < sparselength; ++h)
      {
        if (sparseindices[h] == coldof)
        {
          sparse_ij = sparsevalues[h];
          break;
        }
        else
          sparse_ij = 0.0;
      }
      double analyt_txi = sparse_ij;

      // ********************************* TETA
      double sparse_2 = 0.0;
      int sparsenumentries2 = 0;
      const int sparselength2 = linslipWglobal.num_global_entries(kcnode->dofs()[2]);
      std::vector<double> sparsevalues2(sparselength2);
      std::vector<int> sparseindices2(sparselength2);
      linslipWglobal.extract_global_row_copy(kcnode->dofs()[2], sparselength2, sparsenumentries2,
          sparsevalues2.data(), sparseindices2.data());

      for (int h = 0; h < sparselength2; ++h)
      {
        if (sparseindices2[h] == coldof)
        {
          sparse_2 = sparsevalues2[h];
          break;
        }
        else
          sparse_2 = 0.0;
      }
      double analyt_teta = sparse_2;

      // print results (derivatives) to screen
      if (abs(newCtxi[k] - refCtxi[k]) > 1e-12)
      {
        std::cout << "SLIP W-Deriv_xi: " << kcnode->id()
                  << "\t w.r.t Master: " << mnode->dofs()[fd % dim]
                  << "\t FD= " << std::setprecision(4) << (newCtxi[k] - refCtxi[k]) / delta
                  << "\t analyt= " << std::setprecision(5) << analyt_txi
                  << "\t Error= " << analyt_txi - ((newCtxi[k] - refCtxi[k]) / delta);
        if (abs(analyt_txi - (newCtxi[k] - refCtxi[k]) / delta) > 1.0e-4)
          std::cout << "*** WARNING ***" << std::endl;
        else
          std::cout << " " << std::endl;
      }

      if (abs(newCteta[k] - refCteta[k]) > 1e-12)
      {
        std::cout << "SLIP W-Deriv_eta: " << kcnode->id()
                  << "\t w.r.t Master: " << mnode->dofs()[fd % dim]
                  << "\t FD= " << std::setprecision(4) << (newCteta[k] - refCteta[k]) / delta
                  << "\t analyt= " << std::setprecision(5) << analyt_teta
                  << "\t Error= " << analyt_teta - ((newCteta[k] - refCteta[k]) / delta);
        if (abs(analyt_teta - (newCteta[k] - refCteta[k]) / delta) > 1.0e-4)
          std::cout << "*** WARNING ***" << std::endl;
        else
          std::cout << " " << std::endl;
      }
    }

    // undo finite difference modification
    if (fd % dim == 0)
    {
      mnode->wear_data().wcurr()[0] -= delta;
    }
  }

  // back to normal...
  initialize();
  evaluate();

  return;

}  // FDCheckSlipTrescaDeriv

/*----------------------------------------------------------------------*
 | Finite difference check for T-Mortar derivatives          farah 09/13|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::fd_check_mortar_t_deriv()
{
  // FD checks only for serial case
  std::shared_ptr<Core::LinAlg::Map> snodefullmap = Core::LinAlg::allreduce_e_map(*snoderowmap_);
  std::shared_ptr<Core::LinAlg::Map> mnodefullmap = Core::LinAlg::allreduce_e_map(*mnoderowmap_);
  if (Core::Communication::num_mpi_ranks(get_comm()) > 1)
    FOUR_C_THROW("FD checks only for serial case");

  // create storage for D-Matrix entries
  std::map<int, std::map<int, double>> refT;  // stores dof-wise the entries of D
  std::map<int, std::map<int, double>> newT;

  std::map<int, std::map<int, std::map<int, double>>>
      refDerivT;  // stores old derivm for every node

  // problem dimension (2D or 3D)
  int dim = n_dim();

  // print reference to screen (D-derivative-maps) and store them for later comparison
  // loop over proc's slave nodes
  for (int i = 0; i < snoderowmap_->num_my_elements(); ++i)
  {
    int gid = snoderowmap_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(node);

    if ((int)(cnode->wear_data().get_t().size()) == 0) continue;

    //    for( int d=0; d<dim; d++ )
    //    {
    int dof = cnode->dofs()[0];
    refT[dof] = cnode->wear_data().get_t()[0];
    //    }

    refDerivT[gid] = cnode->wear_data().get_deriv_tw();
  }

  // global loop to apply FD scheme to all SLAVE dofs (=dim*nodes)
  for (int fd = 0; fd < dim * snodefullmap->num_my_elements(); ++fd)
  {
    // store warnings for this finite difference
    int w = 0;

    // Initialize
    initialize();

    // now get the node we want to apply the FD scheme to
    int gid = snodefullmap->gid(fd / dim);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find slave node with gid %", gid);
    CONTACT::Node* snode = dynamic_cast<CONTACT::Node*>(node);

    int sdof = snode->dofs()[fd % dim];

    std::cout << "\nDERIVATIVE FOR S-NODE # " << gid << " DOF: " << sdof << std::endl;

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd % dim == 0)
    {
      snode->xspatial()[0] += delta;
    }
    else if (fd % dim == 1)
    {
      snode->xspatial()[1] += delta;
    }
    else
    {
      snode->xspatial()[2] += delta;
    }

    // compute element areas
    set_element_areas();

    // *******************************************************************
    // contents of evaluate()
    // *******************************************************************
    evaluate();

    // compute finite difference derivative
    for (int k = 0; k < snoderowmap_->num_my_elements(); ++k)
    {
      int kgid = snoderowmap_->gid(k);
      Core::Nodes::Node* knode = idiscret_->g_node(kgid);
      if (!knode) FOUR_C_THROW("Cannot find node with gid %", kgid);
      CONTACT::FriNode* kcnode = dynamic_cast<CONTACT::FriNode*>(knode);

      if ((int)(kcnode->wear_data().get_t().size()) == 0) continue;

      using CI = std::map<int, double>::const_iterator;

      for (int d = 0; d < 1; d++)
      {
        int dof = kcnode->dofs()[d];

        // store D-values into refD
        newT[dof] = kcnode->wear_data().get_t()[d];

        // print results (derivatives) to screen
        for (CI p = newT[dof].begin(); p != newT[dof].end(); ++p)
        {
          if (abs(newT[dof][p->first] - refT[dof][p->first]) > 1e-12)
          {
            double finit = (newT[dof][p->first] - refT[dof][p->first]) / delta;
            double analy = ((refDerivT[kgid])[(p->first) / n_dim()])[sdof];
            double dev = finit - analy;

            // kgid: currently tested dof of slave node kgid
            // (p->first)/Dim(): paired master
            // sdof: currently modified slave dof
            std::cout << "(" << dof << "," << (p->first) << "," << sdof << ") : fd=" << finit
                      << " derivT=" << analy << " DEVIATION " << dev;

            if (abs(dev) > 1e-4)
            {
              std::cout << " ***** WARNING ***** ";
              w++;
            }
            else if (abs(dev) > 1e-5)
            {
              std::cout << " ***** warning ***** ";
              w++;
            }

            std::cout << std::endl;
          }
        }
      }
    }

    // undo finite difference modification
    if (fd % dim == 0)
    {
      snode->xspatial()[0] -= delta;
    }
    else if (fd % dim == 1)
    {
      snode->xspatial()[1] -= delta;
    }
    else
    {
      snode->xspatial()[2] -= delta;
    }

    std::cout << " ******************** GENERATED " << w << " WARNINGS ***************** "
              << std::endl;
  }

  // back to normal...

  // Initialize
  initialize();

  // compute element areas
  set_element_areas();

  // *******************************************************************
  // contents of evaluate()
  // *******************************************************************
  evaluate();

  return;
}

/*----------------------------------------------------------------------*
 | Finite difference check for T-Mortar derivatives (Master) farah 11/13|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::fd_check_mortar_t_master_deriv()
{
  // FD checks only for serial case
  std::shared_ptr<Core::LinAlg::Map> snodefullmap = Core::LinAlg::allreduce_e_map(*snoderowmap_);
  std::shared_ptr<Core::LinAlg::Map> mnodefullmap = Core::LinAlg::allreduce_e_map(*mnoderowmap_);
  if (Core::Communication::num_mpi_ranks(get_comm()) > 1)
    FOUR_C_THROW("FD checks only for serial case");

  // create storage for D-Matrix entries
  std::map<int, std::map<int, double>> refT;  // stores dof-wise the entries of D
  std::map<int, std::map<int, double>> newT;

  std::map<int, std::map<int, std::map<int, double>>>
      refDerivT;  // stores old derivm for every node

  // problem dimension (2D or 3D)
  int dim = n_dim();

  // print reference to screen (D-derivative-maps) and store them for later comparison
  // loop over proc's slave nodes
  for (int i = 0; i < mnoderowmap_->num_my_elements(); ++i)
  {
    int gid = mnoderowmap_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(node);

    if ((int)(cnode->wear_data().get_t().size()) == 0) continue;

    int dof = cnode->dofs()[0];
    refT[dof] = cnode->wear_data().get_t()[0];

    refDerivT[gid] = cnode->wear_data().get_deriv_tw();
  }

  //**************************************************************
  //                            SLAVE
  //**************************************************************
  // global loop to apply FD scheme to all SLAVE dofs (=dim*nodes)
  for (int fd = 0; fd < dim * snodefullmap->num_my_elements(); ++fd)
  {
    // store warnings for this finite difference
    int w = 0;

    // Initialize
    initialize();

    // now get the node we want to apply the FD scheme to
    int gid = snodefullmap->gid(fd / dim);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find slave node with gid %", gid);
    CONTACT::Node* snode = dynamic_cast<CONTACT::Node*>(node);

    int sdof = snode->dofs()[fd % dim];

    std::cout << "\nDERIVATIVE FOR S-NODE # " << gid << " DOF: " << sdof << std::endl;

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd % dim == 0)
    {
      snode->xspatial()[0] += delta;
    }
    else if (fd % dim == 1)
    {
      snode->xspatial()[1] += delta;
    }
    else
    {
      snode->xspatial()[2] += delta;
    }

    // compute element areas
    set_element_areas();

    // *******************************************************************
    // contents of evaluate()
    // *******************************************************************
    evaluate();

    // compute finite difference derivative
    for (int k = 0; k < mnoderowmap_->num_my_elements(); ++k)
    {
      int kgid = mnoderowmap_->gid(k);
      Core::Nodes::Node* knode = idiscret_->g_node(kgid);
      if (!knode) FOUR_C_THROW("Cannot find node with gid %", kgid);
      CONTACT::FriNode* kcnode = dynamic_cast<CONTACT::FriNode*>(knode);

      if ((int)(kcnode->wear_data().get_t().size()) == 0) continue;

      using CI = std::map<int, double>::const_iterator;

      for (int d = 0; d < 1; d++)
      {
        int dof = kcnode->dofs()[d];

        // store D-values into refD
        newT[dof] = kcnode->wear_data().get_t()[d];

        // print results (derivatives) to screen
        for (CI p = newT[dof].begin(); p != newT[dof].end(); ++p)
        {
          if (abs(newT[dof][p->first] - refT[dof][p->first]) > 1e-12)
          {
            double finit = (newT[dof][p->first] - refT[dof][p->first]) / delta;
            double analy = ((refDerivT[kgid])[(p->first) / n_dim()])[sdof];
            double dev = finit - analy;

            // kgid: currently tested dof of slave node kgid
            // (p->first)/Dim(): paired master
            // sdof: currently modified slave dof
            std::cout << "(" << dof << "," << (p->first) << "," << sdof << ") : fd=" << finit
                      << " derivT=" << analy << " DEVIATION " << dev;

            if (abs(dev) > 1e-4)
            {
              std::cout << " ***** WARNING ***** ";
              w++;
            }
            else if (abs(dev) > 1e-5)
            {
              std::cout << " ***** warning ***** ";
              w++;
            }

            std::cout << std::endl;
          }
        }
      }
    }

    // undo finite difference modification
    if (fd % dim == 0)
    {
      snode->xspatial()[0] -= delta;
    }
    else if (fd % dim == 1)
    {
      snode->xspatial()[1] -= delta;
    }
    else
    {
      snode->xspatial()[2] -= delta;
    }

    std::cout << " ******************** GENERATED " << w << " WARNINGS ***************** "
              << std::endl;
  }

  //**************************************************************
  //                            MASTER
  //**************************************************************
  // global loop to apply FD scheme to all SLAVE dofs (=dim*nodes)
  for (int fd = 0; fd < dim * mnodefullmap->num_my_elements(); ++fd)
  {
    // store warnings for this finite difference
    int w = 0;

    // Initialize
    initialize();

    // now get the node we want to apply the FD scheme to
    int gid = mnodefullmap->gid(fd / dim);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find slave node with gid %", gid);
    CONTACT::Node* snode = dynamic_cast<CONTACT::Node*>(node);

    int sdof = snode->dofs()[fd % dim];

    std::cout << "\nDERIVATIVE FOR M-NODE # " << gid << " DOF: " << sdof << std::endl;

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd % dim == 0)
    {
      snode->xspatial()[0] += delta;
    }
    else if (fd % dim == 1)
    {
      snode->xspatial()[1] += delta;
    }
    else
    {
      snode->xspatial()[2] += delta;
    }

    // compute element areas
    set_element_areas();

    // *******************************************************************
    // contents of evaluate()
    // *******************************************************************
    evaluate();

    // compute finite difference derivative
    for (int k = 0; k < mnoderowmap_->num_my_elements(); ++k)
    {
      int kgid = mnoderowmap_->gid(k);
      Core::Nodes::Node* knode = idiscret_->g_node(kgid);
      if (!knode) FOUR_C_THROW("Cannot find node with gid %", kgid);
      CONTACT::FriNode* kcnode = dynamic_cast<CONTACT::FriNode*>(knode);

      if ((int)(kcnode->wear_data().get_t().size()) == 0) continue;

      using CI = std::map<int, double>::const_iterator;

      for (int d = 0; d < 1; d++)
      {
        int dof = kcnode->dofs()[d];

        // store D-values into refD
        newT[dof] = kcnode->wear_data().get_t()[d];

        // print results (derivatives) to screen
        for (CI p = newT[dof].begin(); p != newT[dof].end(); ++p)
        {
          if (abs(newT[dof][p->first] - refT[dof][p->first]) > 1e-12)
          {
            double finit = (newT[dof][p->first] - refT[dof][p->first]) / delta;
            double analy = ((refDerivT[kgid])[(p->first) / n_dim()])[sdof];
            double dev = finit - analy;

            // kgid: currently tested dof of slave node kgid
            // (p->first)/Dim(): paired master
            // sdof: currently modified slave dof
            std::cout << "(" << dof << "," << (p->first) << "," << sdof << ") : fd=" << finit
                      << " derivT=" << analy << " DEVIATION " << dev;

            if (abs(dev) > 1e-4)
            {
              std::cout << " ***** WARNING ***** ";
              w++;
            }
            else if (abs(dev) > 1e-5)
            {
              std::cout << " ***** warning ***** ";
              w++;
            }

            std::cout << std::endl;
          }
        }
      }
    }

    // undo finite difference modification
    if (fd % dim == 0)
    {
      snode->xspatial()[0] -= delta;
    }
    else if (fd % dim == 1)
    {
      snode->xspatial()[1] -= delta;
    }
    else
    {
      snode->xspatial()[2] -= delta;
    }

    std::cout << " ******************** GENERATED " << w << " WARNINGS ***************** "
              << std::endl;
  }

  // back to normal...

  // Initialize
  initialize();

  // compute element areas
  set_element_areas();

  // *******************************************************************
  // contents of evaluate()
  // *******************************************************************
  evaluate();

  return;
}

/*----------------------------------------------------------------------*
 | Finite difference check for E-Mortar derivatives          farah 09/13|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::fd_check_mortar_e_deriv()
{
  // FD checks only for serial case
  std::shared_ptr<Core::LinAlg::Map> snodefullmap = Core::LinAlg::allreduce_e_map(*snoderowmap_);
  std::shared_ptr<Core::LinAlg::Map> mnodefullmap = Core::LinAlg::allreduce_e_map(*mnoderowmap_);
  if (Core::Communication::num_mpi_ranks(get_comm()) > 1)
    FOUR_C_THROW("FD checks only for serial case");

  // create storage for D-Matrix entries
  std::map<int, std::map<int, double>> refE;  // stores dof-wise the entries of D
  std::map<int, std::map<int, double>> newE;

  std::map<int, std::map<int, std::map<int, double>>>
      refDerivE;  // stores old derivm for every node

  // problem dimension (2D or 3D)
  int dim = n_dim();

  // print reference to screen (D-derivative-maps) and store them for later comparison
  // loop over proc's slave nodes
  for (int i = 0; i < snoderowmap_->num_my_elements(); ++i)
  {
    int gid = snoderowmap_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(node);

    if ((int)(cnode->wear_data().get_e().size()) == 0) continue;

    int dof = cnode->dofs()[0];
    refE[dof] = cnode->wear_data().get_e()[0];

    refDerivE[gid] = cnode->wear_data().get_deriv_e();
  }

  // global loop to apply FD scheme to all SLAVE dofs (=dim*nodes)
  for (int fd = 0; fd < dim * snodefullmap->num_my_elements(); ++fd)
  {
    // store warnings for this finite difference
    int w = 0;

    // Initialize
    initialize();

    // now get the node we want to apply the FD scheme to
    int gid = snodefullmap->gid(fd / dim);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find slave node with gid %", gid);
    CONTACT::Node* snode = dynamic_cast<CONTACT::Node*>(node);

    int sdof = snode->dofs()[fd % dim];

    std::cout << "\nDERIVATIVE FOR S-NODE # " << gid << " DOF: " << sdof << std::endl;

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd % dim == 0)
    {
      snode->xspatial()[0] += delta;
    }
    else if (fd % dim == 1)
    {
      snode->xspatial()[1] += delta;
    }
    else
    {
      snode->xspatial()[2] += delta;
    }

    // compute element areas
    set_element_areas();

    // *******************************************************************
    // contents of evaluate()
    // *******************************************************************
    evaluate();

    // compute finite difference derivative
    for (int k = 0; k < snoderowmap_->num_my_elements(); ++k)
    {
      int kgid = snoderowmap_->gid(k);
      Core::Nodes::Node* knode = idiscret_->g_node(kgid);
      if (!knode) FOUR_C_THROW("Cannot find node with gid %", kgid);
      CONTACT::FriNode* kcnode = dynamic_cast<CONTACT::FriNode*>(knode);

      if ((int)(kcnode->wear_data().get_e().size()) == 0) continue;

      using CI = std::map<int, double>::const_iterator;

      for (int d = 0; d < 1; d++)
      {
        int dof = kcnode->dofs()[d];

        // store D-values into refD
        newE[dof] = kcnode->wear_data().get_e()[d];

        // print results (derivatives) to screen
        for (CI p = newE[dof].begin(); p != newE[dof].end(); ++p)
        {
          if (abs(newE[dof][p->first] - refE[dof][p->first]) > 1e-12)
          {
            double finit = (newE[dof][p->first] - refE[dof][p->first]) / delta;
            double analy = ((refDerivE[kgid])[(p->first) / n_dim()])[sdof];
            double dev = finit - analy;

            // kgid: currently tested dof of slave node kgid
            // (p->first)/Dim(): paired master
            // sdof: currently modified slave dof
            std::cout << "(" << dof << "," << (p->first) / n_dim() << "," << sdof
                      << ") : fd=" << finit << " derivE=" << analy << " DEVIATION " << dev;

            if (abs(dev) > 1e-4)
            {
              std::cout << " ***** WARNING ***** ";
              w++;
            }
            else if (abs(dev) > 1e-5)
            {
              std::cout << " ***** warning ***** ";
              w++;
            }

            std::cout << std::endl;
          }
        }
      }
    }

    // undo finite difference modification
    if (fd % dim == 0)
    {
      snode->xspatial()[0] -= delta;
    }
    else if (fd % dim == 1)
    {
      snode->xspatial()[1] -= delta;
    }
    else
    {
      snode->xspatial()[2] -= delta;
    }

    std::cout << " ******************** GENERATED " << w << " WARNINGS ***************** "
              << std::endl;
  }

  // back to normal...

  // Initialize
  initialize();

  // compute element areas
  set_element_areas();

  // *******************************************************************
  // contents of evaluate()
  // *******************************************************************
  evaluate();

  return;
}

/*----------------------------------------------------------------------*
 | Finite difference check for E-Mortar derivatives (Master) farah 11/13|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::fd_check_mortar_e_master_deriv()
{
  // FD checks only for serial case
  std::shared_ptr<Core::LinAlg::Map> snodefullmap = Core::LinAlg::allreduce_e_map(*snoderowmap_);
  std::shared_ptr<Core::LinAlg::Map> mnodefullmap = Core::LinAlg::allreduce_e_map(*mnoderowmap_);
  if (Core::Communication::num_mpi_ranks(get_comm()) > 1)
    FOUR_C_THROW("FD checks only for serial case");

  // create storage for D-Matrix entries
  std::map<int, std::map<int, double>> refE;  // stores dof-wise the entries of D
  std::map<int, std::map<int, double>> newE;

  std::map<int, std::map<int, std::map<int, double>>>
      refDerivE;  // stores old derivm for every node

  // problem dimension (2D or 3D)
  int dim = n_dim();

  // print reference to screen (D-derivative-maps) and store them for later comparison
  // loop over proc's slave nodes
  for (int i = 0; i < mnoderowmap_->num_my_elements(); ++i)
  {
    int gid = mnoderowmap_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(node);

    if ((int)(cnode->wear_data().get_e().size()) == 0) continue;

    int dof = cnode->dofs()[0];
    refE[dof] = cnode->wear_data().get_e()[0];

    refDerivE[gid] = cnode->wear_data().get_deriv_e();
  }

  //**************************************************************
  //                            SLAVE
  //**************************************************************
  // global loop to apply FD scheme to all SLAVE dofs (=dim*nodes)
  for (int fd = 0; fd < dim * snodefullmap->num_my_elements(); ++fd)
  {
    // store warnings for this finite difference
    int w = 0;

    // Initialize
    initialize();

    // now get the node we want to apply the FD scheme to
    int gid = snodefullmap->gid(fd / dim);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find slave node with gid %", gid);
    CONTACT::Node* snode = dynamic_cast<CONTACT::Node*>(node);

    int sdof = snode->dofs()[fd % dim];

    std::cout << "\nDERIVATIVE FOR S-NODE # " << gid << " DOF: " << sdof << std::endl;

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd % dim == 0)
    {
      snode->xspatial()[0] += delta;
    }
    else if (fd % dim == 1)
    {
      snode->xspatial()[1] += delta;
    }
    else
    {
      snode->xspatial()[2] += delta;
    }

    // compute element areas
    set_element_areas();

    // *******************************************************************
    // contents of evaluate()
    // *******************************************************************
    evaluate();

    // compute finite difference derivative
    for (int k = 0; k < mnoderowmap_->num_my_elements(); ++k)
    {
      int kgid = mnoderowmap_->gid(k);
      Core::Nodes::Node* knode = idiscret_->g_node(kgid);
      if (!knode) FOUR_C_THROW("Cannot find node with gid %", kgid);
      CONTACT::FriNode* kcnode = dynamic_cast<CONTACT::FriNode*>(knode);

      if ((int)(kcnode->wear_data().get_e().size()) == 0) continue;

      using CI = std::map<int, double>::const_iterator;

      for (int d = 0; d < 1; d++)
      {
        int dof = kcnode->dofs()[d];

        // store D-values into refD
        newE[dof] = kcnode->wear_data().get_e()[d];

        // print results (derivatives) to screen
        for (CI p = newE[dof].begin(); p != newE[dof].end(); ++p)
        {
          if (abs(newE[dof][p->first] - refE[dof][p->first]) > 1e-12)
          {
            double finit = (newE[dof][p->first] - refE[dof][p->first]) / delta;
            double analy = ((refDerivE[kgid])[(p->first) / n_dim()])[sdof];
            double dev = finit - analy;

            // kgid: currently tested dof of slave node kgid
            // (p->first)/Dim(): paired master
            // sdof: currently modified slave dof
            std::cout << "(" << dof << "," << (p->first) / n_dim() << "," << sdof
                      << ") : fd=" << finit << " derivE=" << analy << " DEVIATION " << dev;

            if (abs(dev) > 1e-4)
            {
              std::cout << " ***** WARNING ***** ";
              w++;
            }
            else if (abs(dev) > 1e-5)
            {
              std::cout << " ***** warning ***** ";
              w++;
            }

            std::cout << std::endl;
          }
        }
      }
    }

    // undo finite difference modification
    if (fd % dim == 0)
    {
      snode->xspatial()[0] -= delta;
    }
    else if (fd % dim == 1)
    {
      snode->xspatial()[1] -= delta;
    }
    else
    {
      snode->xspatial()[2] -= delta;
    }

    std::cout << " ******************** GENERATED " << w << " WARNINGS ***************** "
              << std::endl;
  }


  //**************************************************************
  //                            MASTER
  //**************************************************************
  // global loop to apply FD scheme to all SLAVE dofs (=dim*nodes)
  for (int fd = 0; fd < dim * mnodefullmap->num_my_elements(); ++fd)
  {
    // store warnings for this finite difference
    int w = 0;

    // Initialize
    initialize();

    // now get the node we want to apply the FD scheme to
    int gid = mnodefullmap->gid(fd / dim);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find slave node with gid %", gid);
    CONTACT::Node* snode = dynamic_cast<CONTACT::Node*>(node);

    int sdof = snode->dofs()[fd % dim];

    std::cout << "\nDERIVATIVE FOR M-NODE # " << gid << " DOF: " << sdof << std::endl;

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd % dim == 0)
    {
      snode->xspatial()[0] += delta;
    }
    else if (fd % dim == 1)
    {
      snode->xspatial()[1] += delta;
    }
    else
    {
      snode->xspatial()[2] += delta;
    }

    // compute element areas
    set_element_areas();

    // *******************************************************************
    // contents of evaluate()
    // *******************************************************************
    evaluate();

    // compute finite difference derivative
    for (int k = 0; k < mnoderowmap_->num_my_elements(); ++k)
    {
      int kgid = mnoderowmap_->gid(k);
      Core::Nodes::Node* knode = idiscret_->g_node(kgid);
      if (!knode) FOUR_C_THROW("Cannot find node with gid %", kgid);
      CONTACT::FriNode* kcnode = dynamic_cast<CONTACT::FriNode*>(knode);

      if ((int)(kcnode->wear_data().get_e().size()) == 0) continue;

      using CI = std::map<int, double>::const_iterator;

      for (int d = 0; d < 1; d++)
      {
        int dof = kcnode->dofs()[d];

        // store D-values into refD
        newE[dof] = kcnode->wear_data().get_e()[d];

        // print results (derivatives) to screen
        for (CI p = newE[dof].begin(); p != newE[dof].end(); ++p)
        {
          if (abs(newE[dof][p->first] - refE[dof][p->first]) > 1e-12)
          {
            double finit = (newE[dof][p->first] - refE[dof][p->first]) / delta;
            double analy = ((refDerivE[kgid])[(p->first) / n_dim()])[sdof];
            double dev = finit - analy;

            // kgid: currently tested dof of slave node kgid
            // (p->first)/Dim(): paired master
            // sdof: currently modified slave dof
            std::cout << "(" << dof << "," << (p->first) / n_dim() << "," << sdof
                      << ") : fd=" << finit << " derivE=" << analy << " DEVIATION " << dev;

            if (abs(dev) > 1e-4)
            {
              std::cout << " ***** WARNING ***** ";
              w++;
            }
            else if (abs(dev) > 1e-5)
            {
              std::cout << " ***** warning ***** ";
              w++;
            }

            std::cout << std::endl;
          }
        }
      }
    }

    // undo finite difference modification
    if (fd % dim == 0)
    {
      snode->xspatial()[0] -= delta;
    }
    else if (fd % dim == 1)
    {
      snode->xspatial()[1] -= delta;
    }
    else
    {
      snode->xspatial()[2] -= delta;
    }

    std::cout << " ******************** GENERATED " << w << " WARNINGS ***************** "
              << std::endl;
  }

  // back to normal...

  // Initialize
  initialize();

  // compute element areas
  set_element_areas();

  // *******************************************************************
  // contents of evaluate()
  // *******************************************************************
  evaluate();

  return;
}

/*----------------------------------------------------------------------*
 | Finite difference check for normal Wear-lm derivatives    farah 07/13|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::fd_check_wear_deriv_lm()
{
  double wcoeff = interface_params().get<double>("WEARCOEFF");

  // FD checks only for serial case
  std::shared_ptr<Core::LinAlg::Map> snodefullmap = Core::LinAlg::allreduce_e_map(*snoderowmap_);
  std::shared_ptr<Core::LinAlg::Map> mnodefullmap = Core::LinAlg::allreduce_e_map(*mnoderowmap_);
  if (Core::Communication::num_mpi_ranks(get_comm()) > 1)
    FOUR_C_THROW("FD checks only for serial case");

  // create storage for gap values
  int nrow = snoderowmap_->num_my_elements();
  std::vector<double> refW(nrow);
  std::vector<double> newW(nrow);

  // problem dimension (2D or 3D)
  int dim = n_dim();

  // store reference
  // loop over proc's slave nodes
  for (int i = 0; i < snoderowmap_->num_my_elements(); ++i)
  {
    int gid = snoderowmap_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(node);

    // store wear-values into refW
    refW[i] = cnode->wear_data().weighted_wear();
  }

  // global loop to apply FD scheme to all slave dofs (=dim*nodes)
  for (int fd = 0; fd < dim * snodefullmap->num_my_elements(); ++fd)
  {
    // store warnings for this finite difference
    int w = 0;

    // Initialize -- set WearData.Wear() = 0 !!!
    initialize();

    // now get the node we want to apply the FD scheme to
    int gid = snodefullmap->gid(fd / dim);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find slave node with gid %", gid);
    CONTACT::Node* snode = dynamic_cast<CONTACT::Node*>(node);

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd % dim == 0)
    {
      snode->mo_data().lm()[0] += delta;
    }
    else if (fd % dim == 1)
    {
      snode->mo_data().lm()[1] += delta;
    }
    else
    {
      snode->mo_data().lm()[2] += delta;
    }

    // compute element areas
    set_element_areas();

    // *******************************************************************
    // contents of evaluate()
    // *******************************************************************
    evaluate();


    // compute finite difference derivative
    for (int k = 0; k < snoderowmap_->num_my_elements(); ++k)
    {
      int kgid = snoderowmap_->gid(k);
      Core::Nodes::Node* knode = idiscret_->g_node(kgid);
      if (!knode) FOUR_C_THROW("Cannot find node with gid %", kgid);
      CONTACT::FriNode* kcnode = dynamic_cast<CONTACT::FriNode*>(knode);

      // store gap-values into newG
      newW[k] = wcoeff * kcnode->wear_data().delta_weighted_wear();

      //      if (abs(newW[k]-refW[k]) > 1e-12 && newW[k]!=1.0e12 && refW[k] != 1.0e12)
      //      {
      double finit = (newW[k] - refW[k]) / delta;
      double analy = kcnode->data().get_deriv_wlm()[snode->dofs()[fd % dim]];
      double dev = finit - analy;

      // kgid: id of currently tested slave node
      // snode->Dofs()[fd%dim]: currently modified slave dof
      std::cout << "(" << kgid << "," << snode->dofs()[fd % dim] << ") : fd=" << finit
                << " derivw=" << analy << " DEVIATION " << dev;

      if (abs(dev) > 1e-4)
      {
        std::cout << " ***** WARNING ***** ";
        w++;
      }
      else if (abs(dev) > 1e-5)
      {
        std::cout << " ***** warning ***** ";
        w++;
      }

      std::cout << std::endl;
      //      }
    }
    // undo finite difference modification
    if (fd % dim == 0)
    {
      snode->mo_data().lm()[0] -= delta;
    }
    else if (fd % dim == 1)
    {
      snode->mo_data().lm()[1] -= delta;
    }
    else
    {
      snode->mo_data().lm()[2] -= delta;
    }


    std::cout << " ******************** GENERATED " << w << " WARNINGS ***************** "
              << std::endl;
    //    if (w>0)
    //      FOUR_C_THROW("WARNING!!!");
  }

  // LM only on slave nodes!!!

  // back to normal...
  initialize();
  evaluate();

  // write the init. wear back into the node
  for (int i = 0; i < snoderowmap_->num_my_elements(); ++i)
  {
    int gid = snoderowmap_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(node);

    // store gap-values into refG
    cnode->wear_data().weighted_wear() = refW[i];
  }


  return;
}

/*----------------------------------------------------------------------*
 | Finite difference check for normal Wear-d derivatives     farah 07/13|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::fd_check_wear_deriv()
{
  double wcoeff = interface_params().get<double>("WEARCOEFF");

  // FD checks only for serial case
  std::shared_ptr<Core::LinAlg::Map> snodefullmap = Core::LinAlg::allreduce_e_map(*snoderowmap_);
  std::shared_ptr<Core::LinAlg::Map> mnodefullmap = Core::LinAlg::allreduce_e_map(*mnoderowmap_);
  if (Core::Communication::num_mpi_ranks(get_comm()) > 1)
    FOUR_C_THROW("FD checks only for serial case");

  // create storage for gap values
  int nrow = snoderowmap_->num_my_elements();
  std::vector<double> refW(nrow);
  std::vector<double> newW(nrow);

  // problem dimension (2D or 3D)
  int dim = n_dim();

  // store reference
  // loop over proc's slave nodes
  for (int i = 0; i < snoderowmap_->num_my_elements(); ++i)
  {
    int gid = snoderowmap_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(node);

    // store wear-values into refW
    refW[i] = cnode->wear_data().weighted_wear();
  }

  // global loop to apply FD scheme to all slave dofs (=dim*nodes)
  for (int fd = 0; fd < dim * snodefullmap->num_my_elements(); ++fd)
  {
    // store warnings for this finite difference
    int w = 0;

    // Initialize -- set WearData.Wear() = 0 !!!
    initialize();

    // now get the node we want to apply the FD scheme to
    int gid = snodefullmap->gid(fd / dim);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find slave node with gid %", gid);
    CONTACT::Node* snode = dynamic_cast<CONTACT::Node*>(node);

    int sdof = snode->dofs()[fd % dim];
    std::cout << "\nDERIVATIVE FOR S-NODE # " << gid << " DOF: " << sdof << std::endl;

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd % dim == 0)
    {
      snode->xspatial()[0] += delta;
    }
    else if (fd % dim == 1)
    {
      snode->xspatial()[1] += delta;
    }
    else
    {
      snode->xspatial()[2] += delta;
    }

    // compute element areas
    set_element_areas();

    // *******************************************************************
    // contents of evaluate()
    // *******************************************************************
    evaluate();


    // compute finite difference derivative
    for (int k = 0; k < snoderowmap_->num_my_elements(); ++k)
    {
      int kgid = snoderowmap_->gid(k);
      Core::Nodes::Node* knode = idiscret_->g_node(kgid);
      if (!knode) FOUR_C_THROW("Cannot find node with gid %", kgid);
      CONTACT::FriNode* kcnode = dynamic_cast<CONTACT::FriNode*>(knode);

      // store gap-values into newG
      newW[k] = wcoeff * kcnode->wear_data().delta_weighted_wear();

      if (abs(newW[k] - refW[k]) > 1e-12 && newW[k] != 1.0e12 && refW[k] != 1.0e12)
      {
        double finit = (newW[k] - refW[k]) / delta;
        double analy = kcnode->data().get_deriv_w()[snode->dofs()[fd % dim]];
        double dev = finit - analy;

        // kgid: id of currently tested slave node
        // snode->Dofs()[fd%dim]: currently modified slave dof
        std::cout << "(" << kgid << "," << snode->dofs()[fd % dim] << ") : fd=" << finit
                  << " derivw=" << analy << " DEVIATION " << dev;

        if (abs(dev) > 1e-4)
        {
          std::cout << " ***** WARNING ***** ";
          w++;
        }
        else if (abs(dev) > 1e-5)
        {
          std::cout << " ***** warning ***** ";
          w++;
        }

        std::cout << std::endl;
      }
    }
    // undo finite difference modification
    if (fd % dim == 0)
    {
      snode->xspatial()[0] -= delta;
    }
    else if (fd % dim == 1)
    {
      snode->xspatial()[1] -= delta;
    }
    else
    {
      snode->xspatial()[2] -= delta;
    }

    std::cout << " ******************** GENERATED " << w << " WARNINGS ***************** "
              << std::endl;
    //    if (w>0)
    //      FOUR_C_THROW("WARNING!!!");
  }

  // global loop to apply FD scheme to all master dofs (=dim*nodes)
  for (int fd = 0; fd < dim * mnodefullmap->num_my_elements(); ++fd)
  {
    // store warnings for this finite difference
    int w = 0;

    // Initialize
    // loop over all nodes to reset normals, closestnode and Mortar maps
    // (use fully overlapping column map)
    initialize();

    // now get the node we want to apply the FD scheme to
    int gid = mnodefullmap->gid(fd / dim);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find master node with gid %", gid);
    CONTACT::FriNode* mnode = dynamic_cast<CONTACT::FriNode*>(node);

    int mdof = mnode->dofs()[fd % dim];
    std::cout << "\nDERIVATIVE FOR M-NODE # " << gid << " DOF: " << mdof << std::endl;

    // do step forward (modify nodal displacement)
    double delta = 1e-6;
    if (fd % dim == 0)
    {
      mnode->xspatial()[0] += delta;
    }
    else if (fd % dim == 1)
    {
      mnode->xspatial()[1] += delta;
    }
    else
    {
      mnode->xspatial()[2] += delta;
    }

    // compute element areas
    set_element_areas();

    // *******************************************************************
    // contents of evaluate()
    // *******************************************************************
    evaluate();

    // compute finite difference derivative
    for (int k = 0; k < snoderowmap_->num_my_elements(); ++k)
    {
      int kgid = snoderowmap_->gid(k);
      Core::Nodes::Node* knode = idiscret_->g_node(kgid);
      if (!knode) FOUR_C_THROW("Cannot find node with gid %", kgid);
      CONTACT::FriNode* kcnode = dynamic_cast<CONTACT::FriNode*>(knode);

      // store gap-values into newG
      newW[k] = wcoeff * kcnode->wear_data().delta_weighted_wear();

      if (abs(newW[k] - refW[k]) > 1e-12 && newW[k] != 1.0e12 && refW[k] != 1.0e12)
      {
        double finit = (newW[k] - refW[k]) / delta;
        double analy = kcnode->data().get_deriv_w()[mnode->dofs()[fd % dim]];
        double dev = finit - analy;

        // kgid: id of currently tested slave node
        // mnode->Dofs()[fd%dim]: currently modified slave dof
        std::cout << "(" << kgid << "," << mnode->dofs()[fd % dim] << ") : fd=" << finit
                  << " derivw=" << analy << " DEVIATION " << dev;

        if (abs(dev) > 1e-4)
        {
          std::cout << " ***** WARNING ***** ";
          w++;
        }
        else if (abs(dev) > 1e-5)
        {
          std::cout << " ***** warning ***** ";
          w++;
        }

        std::cout << std::endl;
      }
    }

    // undo finite difference modification
    if (fd % dim == 0)
    {
      mnode->xspatial()[0] -= delta;
    }
    else if (fd % dim == 1)
    {
      mnode->xspatial()[1] -= delta;
    }
    else
    {
      mnode->xspatial()[2] -= delta;
    }

    std::cout << " ******************** GENERATED " << w << " WARNINGS ***************** "
              << std::endl;
    //    if (w>0)
    //      FOUR_C_THROW("WARNING!!!");
  }

  // back to normal...
  initialize();
  evaluate();

  // write the init. wear back into the node
  for (int i = 0; i < snoderowmap_->num_my_elements(); ++i)
  {
    int gid = snoderowmap_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(node);

    cnode->wear_data().weighted_wear() = refW[i];
  }

  return;
}

FOUR_C_NAMESPACE_CLOSE
