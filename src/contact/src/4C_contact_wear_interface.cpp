// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_wear_interface.hpp"

#include "4C_contact_defines.hpp"
#include "4C_contact_element.hpp"
#include "4C_contact_friction_node.hpp"
#include "4C_contact_interface.hpp"
#include "4C_contact_node.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_inpar_mortar.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_mortar_dofset.hpp"
#include "4C_mortar_element.hpp"
#include "4C_mortar_node.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  ctor (public)                                            farah 09/13|
 *----------------------------------------------------------------------*/
Wear::WearInterface::WearInterface(
    const std::shared_ptr<Mortar::InterfaceDataContainer>& interfaceData_ptr, const int id,
    MPI_Comm comm, const int dim, const Teuchos::ParameterList& icontact, bool selfcontact)
    : CONTACT::Interface(interfaceData_ptr, id, comm, dim, icontact, selfcontact),
      wear_(false),
      wearimpl_(false),
      wearpv_(false),
      wearboth_(false),
      sswear_(icontact.get<bool>("SSWEAR"))
{
  // set wear contact status
  auto wtype = Teuchos::getIntegralValue<Inpar::Wear::WearType>(icontact, "WEARTYPE");

  auto wtimint = Teuchos::getIntegralValue<Inpar::Wear::WearTimInt>(icontact, "WEARTIMINT");

  auto wside = Teuchos::getIntegralValue<Inpar::Wear::WearSide>(icontact, "WEAR_SIDE");

  auto wlaw = Teuchos::getIntegralValue<Inpar::Wear::WearLaw>(icontact, "WEARLAW");

  if (wlaw != Inpar::Wear::wear_none) wear_ = true;

  if (wtimint == Inpar::Wear::wear_impl) wearimpl_ = true;

  // set wear contact discretization
  if (wtype == Inpar::Wear::wear_primvar) wearpv_ = true;

  // set wear contact discretization
  if (wside == Inpar::Wear::wear_both) wearboth_ = true;

  return;
}

/*----------------------------------------------------------------------*
 |  Assemble Mortar wear matrices                            farah 09/13|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::assemble_te(
    Core::LinAlg::SparseMatrix& tglobal, Core::LinAlg::SparseMatrix& eglobal)
{
  /************************************************
   *  This function is only for discrete Wear !!! *
   ************************************************/

  // nodes for loop
  std::shared_ptr<Core::LinAlg::Map> considerednodes;

  // nothing to do if no active nodes
  if (sswear_)
  {
    if (activenodes_ == nullptr) return;
    considerednodes = activenodes_;
  }
  else
  {
    if (slipnodes_ == nullptr) return;
    considerednodes = slipnodes_;
  }

  // loop over proc's slave nodes of the interface for assembly
  // use standard row map to assemble each node only once
  for (int i = 0; i < considerednodes->num_my_elements(); ++i)
  {
    int gid = considerednodes->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* fnode = dynamic_cast<CONTACT::FriNode*>(node);

    if (fnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
      FOUR_C_THROW("AssembleTE: Node ownership inconsistency!");

    /**************************************************** T-matrix ******/
    if ((fnode->wear_data().get_t()).size() > 0)
    {
      std::vector<std::map<int, double>>& tmap = fnode->wear_data().get_t();
      int colsize = (int)tmap[0].size();

      std::map<int, double>::iterator colcurr;

      int row = fnode->dofs()[0];
      int k = 0;

      for (colcurr = tmap[0].begin(); colcurr != tmap[0].end(); ++colcurr)
      {
        int col = colcurr->first;
        double val = colcurr->second;

        // don't check for diagonality
        // since for standard shape functions, as in general when using
        // arbitrary shape function types, this is not the case
        // create the d matrix, do not assemble zeros
        if (abs(val) > 1.0e-12) tglobal.assemble(val, row, col);

        ++k;
      }

      if (k != colsize) FOUR_C_THROW("AssembleTE: k = {} but colsize = {}", k, colsize);
    }

    /**************************************************** E-matrix ******/
    if ((fnode->wear_data().get_e()).size() > 0)
    {
      std::vector<std::map<int, double>>& emap = fnode->wear_data().get_e();
      int rowsize = 1;  // fnode->NumDof();
      int colsize = (int)emap[0].size();

      for (int j = 0; j < rowsize - 1; ++j)
        if ((int)emap[j].size() != (int)emap[j + 1].size())
          FOUR_C_THROW("AssembleTE: Column dim. of nodal E-map is inconsistent!");

      std::map<int, double>::iterator colcurr;

      int row = fnode->dofs()[0];
      int k = 0;

      for (colcurr = emap[0].begin(); colcurr != emap[0].end(); ++colcurr)
      {
        int col = colcurr->first;
        double val = colcurr->second;

        // do not assemble zeros into m matrix
        if (wear_shape_fcn() == Inpar::Wear::wear_shape_standard)
        {
          if (abs(val) > 1.0e-12) eglobal.assemble(val, row, col);
          ++k;
        }
        else if (wear_shape_fcn() == Inpar::Wear::wear_shape_dual)
        {
          if (col == row)
            if (abs(val) > 1.0e-12) eglobal.assemble(val, row, col);
          ++k;
        }
        else
          FOUR_C_THROW("Chosen wear shape function not supported!");
      }

      if (k != colsize) FOUR_C_THROW("AssembleTE: k = {} but colsize = {}", k, colsize);
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Assemble Mortar wear matrices (for master side)          farah 11/13|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::assemble_te_master(
    Core::LinAlg::SparseMatrix& tglobal, Core::LinAlg::SparseMatrix& eglobal)
{
  /************************************************
   *  This function is only for discrete Wear !!! *
   ************************************************/

  if (!(wearboth_ and wearpv_)) FOUR_C_THROW("AssembleTE_Master only for discr both-sided wear!");

  //*******************************************************
  // assemble second D matrix for both-sided wear :
  // Loop over allreduced map, so that any proc join
  // the loop. Do non-local assembly by FEAssemble to
  // allow owned and ghosted nodes assembly. In integrator
  // only the nodes associated to the owned slave element
  // are filled with entries. Therefore, the integrated
  // entries are unique on nodes!
  //*******************************************************

  // nothing to do if no active nodes
  if (slipmasternodes_ == nullptr) return;

  const std::shared_ptr<Core::LinAlg::Map> slmasternodes =
      Core::LinAlg::allreduce_e_map(*(slipmasternodes_));

  // loop over proc's slave nodes of the interface for assembly
  // use standard row map to assemble each node only once
  for (int i = 0; i < slmasternodes->num_my_elements(); ++i)
  {
    int gid = slmasternodes->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* fnode = dynamic_cast<CONTACT::FriNode*>(node);

    /**************************************************** T-matrix ******/
    if ((fnode->wear_data().get_t()).size() > 0)
    {
      std::vector<std::map<int, double>>& tmap = fnode->wear_data().get_t();
      int colsize = (int)tmap[0].size();

      std::map<int, double>::iterator colcurr;

      int row = fnode->dofs()[0];
      int k = 0;

      for (colcurr = tmap[0].begin(); colcurr != tmap[0].end(); ++colcurr)
      {
        int col = colcurr->first;
        double val = colcurr->second;

        // don't check for diagonality
        // since for standard shape functions, as in general when using
        // arbitrary shape function types, this is not the case
        // create the d matrix, do not assemble zeros
        if (abs(val) > 1.0e-12) tglobal.fe_assemble(val, row, col);

        ++k;
      }

      if (k != colsize) FOUR_C_THROW("AssembleTE: k = {} but colsize = {}", k, colsize);
    }

    /**************************************************** E-matrix ******/
    if ((fnode->wear_data().get_e()).size() > 0)
    {
      std::vector<std::map<int, double>>& emap = fnode->wear_data().get_e();
      int rowsize = 1;  // fnode->NumDof();
      int colsize = (int)emap[0].size();

      for (int j = 0; j < rowsize - 1; ++j)
        if ((int)emap[j].size() != (int)emap[j + 1].size())
          FOUR_C_THROW("AssembleTE: Column dim. of nodal E-map is inconsistent!");

      std::map<int, double>::iterator colcurr;

      int row = fnode->dofs()[0];
      int k = 0;

      for (colcurr = emap[0].begin(); colcurr != emap[0].end(); ++colcurr)
      {
        int col = colcurr->first;
        double val = colcurr->second;

        // do not assemble zeros into m matrix
        if (wear_shape_fcn() == Inpar::Wear::wear_shape_standard)
        {
          if (abs(val) > 1.0e-12) eglobal.fe_assemble(val, row, col);
          ++k;
        }
        else if (wear_shape_fcn() == Inpar::Wear::wear_shape_dual)
        {
          if (col == row)
            if (abs(val) > 1.0e-12) eglobal.fe_assemble(val, row, col);
          ++k;
        }
        else
          FOUR_C_THROW("Chosen wear shape function not supported!");
      }

      if (k != colsize) FOUR_C_THROW("AssembleTE: k = {} but colsize = {}", k, colsize);
    }
  }


  return;
}

/*----------------------------------------------------------------------*
 |  Assemble matrix LinT containing disp derivatives         farah 09/13|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::assemble_lin_t_d(Core::LinAlg::SparseMatrix& lintglobal)
{
  /************************************************
   *  This function is only for discrete Wear !!! *
   ************************************************/

  // nodes for loop
  std::shared_ptr<Core::LinAlg::Map> considerednodes;

  // nothing to do if no active nodes
  if (sswear_)
  {
    if (activenodes_ == nullptr) return;
    considerednodes = activenodes_;
  }
  else
  {
    if (slipnodes_ == nullptr) return;
    considerednodes = slipnodes_;
  }

  /**********************************************************************/
  // we have: T_wj,c with j = Lagrange multiplier slave dof
  //                 with w = wear slave dof
  //                 with c = Displacement slave or master dof
  // we compute (LinT)_kc = T_wj,c * z_j
  /**********************************************************************/

  for (int j = 0; j < considerednodes->num_my_elements(); ++j)
  {
    int gid = considerednodes->gid(j);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* fnode = dynamic_cast<CONTACT::FriNode*>(node);

    // Mortar matrix Tw derivatives
    std::map<int, std::map<int, double>>& tderiv = fnode->wear_data().get_deriv_tw();

    // get sizes and iterator start
    int slavesize = (int)tderiv.size();  // column size
    std::map<int, std::map<int, double>>::iterator scurr = tderiv.begin();

    /********************************************** LinTMatrix **********/
    // loop over all DISP slave nodes in the DerivT-map of the current LM slave node
    for (int k = 0; k < slavesize; ++k)
    {
      int sgid = scurr->first;
      ++scurr;

      Core::Nodes::Node* snode = idiscret_->g_node(sgid);
      if (!snode) FOUR_C_THROW("Cannot find node with gid %", sgid);
      CONTACT::FriNode* csnode = dynamic_cast<CONTACT::FriNode*>(snode);

      // current Lagrange multipliers
      double lmn = 0.0;
      if (n_dim() == 2)
        lmn = (csnode->mo_data().lm()[0]) * (csnode->mo_data().n()[0]) +
              (csnode->mo_data().lm()[1]) * (csnode->mo_data().n()[1]);
      else if (n_dim() == 3)
        lmn = (csnode->mo_data().lm()[0]) * (csnode->mo_data().n()[0]) +
              (csnode->mo_data().lm()[1]) * (csnode->mo_data().n()[1]) +
              (csnode->mo_data().lm()[2]) * (csnode->mo_data().n()[2]);
      else
        FOUR_C_THROW("False Dimension!");

      // Mortar matrix T derivatives
      std::map<int, double>& thisdderive = fnode->wear_data().get_deriv_tw()[sgid];
      int mapsize = (int)(thisdderive.size());

      // we choose the first node dof as wear dof
      int row = fnode->dofs()[0];
      std::map<int, double>::iterator scolcurr = thisdderive.begin();

      // loop over all directional derivative entries
      for (int c = 0; c < mapsize; ++c)
      {
        int col = scolcurr->first;
        double val = lmn * (scolcurr->second);
        ++scolcurr;

        // owner of LM slave node can do the assembly, although it actually
        // might not own the corresponding rows in lindglobal (DISP slave node)
        // (FE_MATRIX automatically takes care of non-local assembly inside!!!)
        // std::cout << "Assemble LinE: " << row << " " << col << " " << val << std::endl;
        if (abs(val) > 1.0e-12) lintglobal.fe_assemble(val, row, col);
      }

      // check for completeness of DerivD-Derivatives-iteration
      if (scolcurr != thisdderive.end())
        FOUR_C_THROW("AssembleLinE_D: Not all derivative entries of DerivE considered!");
    }
    // check for completeness of DerivD-Slave-iteration
    if (scurr != tderiv.end())
      FOUR_C_THROW("AssembleLinE_D: Not all DISP slave entries of DerivE considered!");
    /******************************** Finished with LinTmatrix for delta T **********/
  }

  // *******************************************************************************
  //            Considering linearization of nodal normal vectors                 //
  // *******************************************************************************
  // loop over all LM slave nodes (row map)
  for (int j = 0; j < considerednodes->num_my_elements(); ++j)
  {
    int gid = considerednodes->gid(j);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* fnode = dynamic_cast<CONTACT::FriNode*>(node);

    if (fnode->wear_data().get_t().size() > 0)
    {
      // map iterator
      using CI = std::map<int, double>::const_iterator;

      std::map<int, double>& nmap = fnode->wear_data().get_t()[0];

      // loop over col entries
      for (CI z = nmap.begin(); z != nmap.end(); ++z)
      {
        // std::cout << "t-irst= " << z->first << std::endl;
        int gid3 = (int)((z->first) / n_dim());
        Core::Nodes::Node* snode = idiscret_->g_node(gid3);
        if (!snode) FOUR_C_THROW("Cannot find node with gid");
        CONTACT::FriNode* csnode = dynamic_cast<CONTACT::FriNode*>(snode);

        for (int u = 0; u < n_dim(); ++u)
        {
          Core::Gen::Pairedvector<int, double>& numap = csnode->data().get_deriv_n()[u];
          double lmu = csnode->mo_data().lm()[u];

          // multiply T-column entry with lin n*lambda
          for (auto b = numap.begin(); b != numap.end(); ++b)
          {
            int row = fnode->dofs()[0];
            int col = (b->first);
            double val = (z->second) * (b->second) * lmu;
            // owner of LM slave node can do the assembly, although it actually
            // might not own the corresponding rows in lindglobal (DISP slave node)
            // (FE_MATRIX automatically takes care of non-local assembly inside!!!)
            // std::cout << "Assemble LinT N: " << row << " " << col << " " << val << std::endl;
            if (abs(val) > 1.0e-12) lintglobal.fe_assemble(val, row, col);
          }
        }
      }
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Assemble matrix LinT containing disp derivatives         farah 11/13|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::assemble_lin_t_d_master(Core::LinAlg::SparseMatrix& lintglobal)
{
  /************************************************
   *  This function is only for discrete Wear !!! *
   ************************************************/

  // nothing to do if no active nodes
  if (slipmasternodes_ == nullptr) return;

  /**********************************************************************/
  // we have: T_wj,c with j = Lagrange multiplier slave dof
  //                 with w = wear slave dof
  //                 with c = Displacement slave or master dof
  // we compute (LinT)_kc = T_wj,c * z_j
  /**********************************************************************/
  const std::shared_ptr<Core::LinAlg::Map> slmasternodes =
      Core::LinAlg::allreduce_e_map(*(slipmasternodes_));

  for (int j = 0; j < slmasternodes->num_my_elements(); ++j)
  {
    int gid = slmasternodes->gid(j);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* fnode = dynamic_cast<CONTACT::FriNode*>(node);

    // Mortar matrix Tw derivatives
    std::map<int, std::map<int, double>>& tderiv = fnode->wear_data().get_deriv_tw();

    // get sizes and iterator start
    int slavesize = (int)tderiv.size();  // column size
    std::map<int, std::map<int, double>>::iterator scurr = tderiv.begin();

    /********************************************** LinTMatrix **********/
    // loop over all DISP slave nodes in the DerivT-map of the current LM slave node
    for (int k = 0; k < slavesize; ++k)
    {
      int sgid = scurr->first;
      ++scurr;

      Core::Nodes::Node* snode = idiscret_->g_node(sgid);
      if (!snode) FOUR_C_THROW("Cannot find node with gid %", sgid);
      CONTACT::FriNode* csnode = dynamic_cast<CONTACT::FriNode*>(snode);

      // current Lagrange multipliers
      double lmn = 0.0;
      if (n_dim() == 2)
        lmn = (csnode->mo_data().lm()[0]) * (csnode->mo_data().n()[0]) +
              (csnode->mo_data().lm()[1]) * (csnode->mo_data().n()[1]);
      else if (n_dim() == 3)
        lmn = (csnode->mo_data().lm()[0]) * (csnode->mo_data().n()[0]) +
              (csnode->mo_data().lm()[1]) * (csnode->mo_data().n()[1]) +
              (csnode->mo_data().lm()[2]) * (csnode->mo_data().n()[2]);
      else
        FOUR_C_THROW("False Dimension!");

      // Mortar matrix T derivatives
      std::map<int, double>& thisdderive = fnode->wear_data().get_deriv_tw()[sgid];
      int mapsize = (int)(thisdderive.size());

      // we choose the first node dof as wear dof
      int row = fnode->dofs()[0];
      std::map<int, double>::iterator scolcurr = thisdderive.begin();

      // loop over all directional derivative entries
      for (int c = 0; c < mapsize; ++c)
      {
        int col = scolcurr->first;
        double val = lmn * (scolcurr->second);
        ++scolcurr;

        // owner of LM slave node can do the assembly, although it actually
        // might not own the corresponding rows in lindglobal (DISP slave node)
        // (FE_MATRIX automatically takes care of non-local assembly inside!!!)
        // std::cout << "Assemble LinE: " << row << " " << col << " " << val << std::endl;
        if (abs(val) > 1.0e-12) lintglobal.fe_assemble(val, row, col);
      }

      // check for completeness of DerivD-Derivatives-iteration
      if (scolcurr != thisdderive.end())
        FOUR_C_THROW("AssembleLinE_D: Not all derivative entries of DerivE considered!");
    }
    // check for completeness of DerivD-Slave-iteration
    if (scurr != tderiv.end())
      FOUR_C_THROW("AssembleLinE_D: Not all DISP slave entries of DerivE considered!");
    /******************************** Finished with LinTmatrix for delta T **********/
  }

  // *******************************************************************************
  //            Considering linearization of nodal normal vectors                 //
  // *******************************************************************************
  // loop over all LM slave nodes (row map)
  for (int j = 0; j < slmasternodes->num_my_elements(); ++j)
  {
    int gid = slmasternodes->gid(j);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* fnode = dynamic_cast<CONTACT::FriNode*>(node);

    if (fnode->wear_data().get_t().size() > 0)
    {
      // map iterator
      using CI = std::map<int, double>::const_iterator;

      std::map<int, double>& nmap = fnode->wear_data().get_t()[0];

      // loop over col entries
      for (CI z = nmap.begin(); z != nmap.end(); ++z)
      {
        // std::cout << "t-irst= " << z->first << std::endl;
        int gid3 = (int)((z->first) / n_dim());
        Core::Nodes::Node* snode = idiscret_->g_node(gid3);
        if (!snode) FOUR_C_THROW("Cannot find node with gid");
        CONTACT::FriNode* csnode = dynamic_cast<CONTACT::FriNode*>(snode);

        for (int u = 0; u < n_dim(); ++u)
        {
          Core::Gen::Pairedvector<int, double>& numap = csnode->data().get_deriv_n()[u];
          double lmu = csnode->mo_data().lm()[u];

          // multiply T-column entry with lin n*lambda
          for (auto b = numap.begin(); b != numap.end(); ++b)
          {
            int row = fnode->dofs()[0];
            int col = (b->first);
            double val = (z->second) * (b->second) * lmu;
            // owner of LM slave node can do the assembly, although it actually
            // might not own the corresponding rows in lindglobal (DISP slave node)
            // (FE_MATRIX automatically takes care of non-local assembly inside!!!)
            // std::cout << "Assemble LinT N: " << row << " " << col << " " << val << std::endl;
            if (abs(val) > 1.0e-12) lintglobal.fe_assemble(val, row, col);
          }
        }
      }
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Assemble matrix LinE containing disp derivatives         farah 09/13|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::assemble_lin_e_d(Core::LinAlg::SparseMatrix& lineglobal)
{
  /************************************************
   *  This function is only for discrete Wear !!! *
   ************************************************/

  // nodes for loop
  std::shared_ptr<Core::LinAlg::Map> considerednodes;

  // nothing to do if no active nodes
  if (sswear_)
  {
    if (activenodes_ == nullptr) return;
    considerednodes = activenodes_;
  }
  else
  {
    if (slipnodes_ == nullptr) return;
    considerednodes = slipnodes_;
  }


  /**********************************************************************/
  // we have: E_wj,c with j = Lagrange multiplier slave dof
  //                 with w = wear slave dof
  //                 with c = Displacement slave or master dof
  // we compute (LinE)_kc = T_wj,c * z_j
  /**********************************************************************/

  // loop over all LM slave nodes (row map)
  for (int j = 0; j < considerednodes->num_my_elements(); ++j)
  {
    int gid = considerednodes->gid(j);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* fnode = dynamic_cast<CONTACT::FriNode*>(node);

    // Mortar matrix Tw derivatives
    std::map<int, std::map<int, double>>& ederiv = fnode->wear_data().get_deriv_e();

    // get sizes and iterator start
    int slavesize = (int)ederiv.size();  // column size
    std::map<int, std::map<int, double>>::iterator scurr = ederiv.begin();

    /********************************************** LinTMatrix **********/
    // loop over all DISP slave nodes in the DerivT-map of the current LM slave node
    for (int k = 0; k < slavesize; ++k)
    {
      int sgid = scurr->first;
      ++scurr;

      Core::Nodes::Node* snode = idiscret_->g_node(sgid);
      if (!snode) FOUR_C_THROW("Cannot find node with gid %", sgid);
      CONTACT::FriNode* csnode = dynamic_cast<CONTACT::FriNode*>(snode);

      // current wear - wear from last converged iteration step (partitioned scheme)
      double w = 0.0;
      w = (csnode->wear_data().wcurr()[0] + csnode->wear_data().wold()[0]);

      // Mortar matrix T derivatives
      std::map<int, double>& thisdderive = fnode->wear_data().get_deriv_e()[sgid];
      int mapsize = (int)(thisdderive.size());

      // we choose the first node dof as wear dof
      int row = fnode->dofs()[0];  // csnode->Dofs()[0];
      std::map<int, double>::iterator scolcurr = thisdderive.begin();

      // loop over all directional derivative entries
      for (int c = 0; c < mapsize; ++c)
      {
        int col = scolcurr->first;
        double val = w * (scolcurr->second);
        ++scolcurr;

        // owner of LM slave node can do the assembly, although it actually
        // might not own the corresponding rows in lindglobal (DISP slave node)
        // (FE_MATRIX automatically takes care of non-local assembly inside!!!)
        // std::cout << "Assemble LinE: " << row << " " << col << " " << val << std::endl;
        if (abs(val) > 1.0e-15) lineglobal.fe_assemble(val, row, col);
      }

      // check for completeness of DerivD-Derivatives-iteration
      if (scolcurr != thisdderive.end())
        FOUR_C_THROW("AssembleLinE_D: Not all derivative entries of DerivE considered!");
    }
    // check for completeness of DerivD-Slave-iteration
    if (scurr != ederiv.end())
      FOUR_C_THROW("AssembleLinE_D: Not all DISP slave entries of DerivE considered!");
    /******************************** Finished with LinTmatrix for delta T **********/
  }

  return;
}


/*----------------------------------------------------------------------*
 |  Assemble matrix LinE containing disp derivatives         farah 11/13|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::assemble_lin_e_d_master(Core::LinAlg::SparseMatrix& lineglobal)
{
  /************************************************
   *  This function is only for discrete Wear !!! *
   ************************************************/

  // nothing to do if no slip nodes
  if (slipmasternodes_->num_global_elements() == 0) return;

  /**********************************************************************/
  // we have: E_wj,c with j = Lagrange multiplier slave dof
  //                 with w = wear slave dof
  //                 with c = Displacement slave or master dof
  // we compute (LinE)_kc = T_wj,c * z_j
  /**********************************************************************/
  const std::shared_ptr<Core::LinAlg::Map> slmasternodes =
      Core::LinAlg::allreduce_e_map(*(slipmasternodes_));

  // loop over all LM slave nodes (row map)
  for (int j = 0; j < slmasternodes->num_my_elements(); ++j)
  {
    int gid = slmasternodes->gid(j);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* fnode = dynamic_cast<CONTACT::FriNode*>(node);

    // Mortar matrix Tw derivatives
    std::map<int, std::map<int, double>>& ederiv = fnode->wear_data().get_deriv_e();

    // get sizes and iterator start
    int slavesize = (int)ederiv.size();  // column size
    std::map<int, std::map<int, double>>::iterator scurr = ederiv.begin();

    /********************************************** LinTMatrix **********/
    // loop over all DISP slave nodes in the DerivT-map of the current LM slave node
    for (int k = 0; k < slavesize; ++k)
    {
      int sgid = scurr->first;
      ++scurr;

      Core::Nodes::Node* snode = idiscret_->g_node(sgid);
      if (!snode) FOUR_C_THROW("Cannot find node with gid %", sgid);
      CONTACT::FriNode* csnode = dynamic_cast<CONTACT::FriNode*>(snode);

      // current wear - wear from last converged iteration step (partitioned scheme)
      double w = 0.0;
      w = (csnode->wear_data().wcurr()[0] + csnode->wear_data().wold()[0]);

      // Mortar matrix T derivatives
      std::map<int, double>& thisdderive = fnode->wear_data().get_deriv_e()[sgid];
      int mapsize = (int)(thisdderive.size());

      // we choose the first node dof as wear dof
      int row = fnode->dofs()[0];  // csnode->Dofs()[0];
      std::map<int, double>::iterator scolcurr = thisdderive.begin();

      // loop over all directional derivative entries
      for (int c = 0; c < mapsize; ++c)
      {
        int col = scolcurr->first;
        double val = w * (scolcurr->second);
        ++scolcurr;

        // owner of LM slave node can do the assembly, although it actually
        // might not own the corresponding rows in lindglobal (DISP slave node)
        // (FE_MATRIX automatically takes care of non-local assembly inside!!!)
        // std::cout << "Assemble LinE: " << row << " " << col << " " << val << std::endl;
        if (abs(val) > 1.0e-12) lineglobal.fe_assemble(val, row, col);
      }

      // check for completeness of DerivD-Derivatives-iteration
      if (scolcurr != thisdderive.end())
        FOUR_C_THROW("AssembleLinE_D: Not all derivative entries of DerivE considered!");
    }
    // check for completeness of DerivD-Slave-iteration
    if (scurr != ederiv.end())
      FOUR_C_THROW("AssembleLinE_D: Not all DISP slave entries of DerivE considered!");
    /******************************** Finished with LinTmatrix for delta T **********/
  }

  return;
}



/*----------------------------------------------------------------------*
 |  Assemble matrix LinT containing lm derivatives           farah 09/13|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::assemble_lin_t_lm(Core::LinAlg::SparseMatrix& lintglobal)
{
  /************************************************
   *  This function is only for discrete Wear !!! *
   ************************************************/

  // nodes for loop
  std::shared_ptr<Core::LinAlg::Map> considerednodes;

  // nothing to do if no active nodes
  if (sswear_)
  {
    if (activenodes_ == nullptr) return;
    considerednodes = activenodes_;
  }
  else
  {
    if (slipnodes_ == nullptr) return;
    considerednodes = slipnodes_;
  }

  // loop over all LM slave nodes (row map)
  for (int j = 0; j < considerednodes->num_my_elements(); ++j)
  {
    int gid = considerednodes->gid(j);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* fnode = dynamic_cast<CONTACT::FriNode*>(node);

    using CI = std::map<int, double>::const_iterator;

    if (fnode->wear_data().get_t().size() > 0)
    {
      // column entries for row f
      std::map<int, double>& fmap = fnode->wear_data().get_t()[0];

      for (CI p = fmap.begin(); p != fmap.end(); ++p)
      {
        int gid2 = (int)((p->first) / n_dim());
        Core::Nodes::Node* node2 = idiscret_->g_node(gid2);
        if (!node2) FOUR_C_THROW("Cannot find node with gid %", gid2);
        CONTACT::FriNode* jnode = dynamic_cast<CONTACT::FriNode*>(node2);

        for (int iter = 0; iter < n_dim(); ++iter)
        {
          double n = 0.0;
          n = jnode->mo_data().n()[iter];

          int row = fnode->dofs()[0];
          int col = jnode->dofs()[iter];
          double val = n * (p->second);
          // std::cout << "Assemble LinT: " << row << " " << col << " " << val << std::endl;
          if (abs(val) > 1.0e-12) lintglobal.fe_assemble(val, row, col);
        }
      }
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Assemble matrix LinT containing lm derivatives           farah 11/13|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::assemble_lin_t_lm_master(Core::LinAlg::SparseMatrix& lintglobal)
{
  /************************************************
   *  This function is only for discrete Wear !!! *
   ************************************************/

  // nothing to do if no active nodes
  if (slipmasternodes_ == nullptr) return;

  const std::shared_ptr<Core::LinAlg::Map> slmasternodes =
      Core::LinAlg::allreduce_e_map(*(slipmasternodes_));


  // loop over all LM slave nodes (row map)
  for (int j = 0; j < slmasternodes->num_my_elements(); ++j)
  {
    int gid = slmasternodes->gid(j);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* fnode = dynamic_cast<CONTACT::FriNode*>(node);

    using CI = std::map<int, double>::const_iterator;

    if (fnode->wear_data().get_t().size() > 0)
    {
      // column entries for row f
      std::map<int, double>& fmap = fnode->wear_data().get_t()[0];

      for (CI p = fmap.begin(); p != fmap.end(); ++p)
      {
        int gid2 = (int)((p->first) / n_dim());
        Core::Nodes::Node* node2 = idiscret_->g_node(gid2);
        if (!node2) FOUR_C_THROW("Cannot find node with gid %", gid2);
        CONTACT::FriNode* jnode = dynamic_cast<CONTACT::FriNode*>(node2);

        for (int iter = 0; iter < n_dim(); ++iter)
        {
          double n = 0.0;
          n = jnode->mo_data().n()[iter];

          int row = fnode->dofs()[0];
          int col = jnode->dofs()[iter];
          double val = n * (p->second);
          // std::cout << "Assemble LinT: " << row << " " << col << " " << val << std::endl;
          if (abs(val) > 1.0e-12) lintglobal.fe_assemble(val, row, col);
        }
      }
    }
  }

  return;
}


/*----------------------------------------------------------------------*
 |  evaluate nodal normals (public)                          farah 11/13|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::evaluate_nodal_normals() const
{
  // call mortar function
  Mortar::Interface::evaluate_nodal_normals();

  // for both-sided discrete wear
  if (wearboth_ == true and wearpv_ == true)
  {
    for (int i = 0; i < mnoderowmap_->num_my_elements(); ++i)
    {
      int gid = mnoderowmap_->gid(i);
      Core::Nodes::Node* node = idiscret_->g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      Mortar::Node* mrtrnode = dynamic_cast<Mortar::Node*>(node);

      // build averaged normal at each master node
      mrtrnode->build_averaged_normal();
    }
  }

  return;
}


/*----------------------------------------------------------------------*
 |  export nodal normals (public)                            farah 11/13|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::export_nodal_normals() const
{
  // call contact function
  CONTACT::Interface::export_nodal_normals();

  std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>> triad;

  std::map<int, std::vector<int>> n_x_key;
  std::map<int, std::vector<int>> n_y_key;
  std::map<int, std::vector<int>> n_z_key;
  std::map<int, std::vector<int>> txi_x_key;
  std::map<int, std::vector<int>> txi_y_key;
  std::map<int, std::vector<int>> txi_z_key;
  std::map<int, std::vector<int>> teta_x_key;
  std::map<int, std::vector<int>> teta_y_key;
  std::map<int, std::vector<int>> teta_z_key;

  std::map<int, std::vector<double>> n_x_val;
  std::map<int, std::vector<double>> n_y_val;
  std::map<int, std::vector<double>> n_z_val;
  std::map<int, std::vector<double>> txi_x_val;
  std::map<int, std::vector<double>> txi_y_val;
  std::map<int, std::vector<double>> txi_z_val;
  std::map<int, std::vector<double>> teta_x_val;
  std::map<int, std::vector<double>> teta_y_val;
  std::map<int, std::vector<double>> teta_z_val;

  Core::Gen::Pairedvector<int, double>::iterator iter;

  // --------------------------------------------------------------------------------------
  // for both-sided discrete wear we need the same normal information on the master side:
  // --------------------------------------------------------------------------------------
  if (wearboth_ and wearpv_)
  {
    const std::shared_ptr<Core::LinAlg::Map> masternodes =
        Core::LinAlg::allreduce_e_map(*(mnoderowmap_));

    // build info on row map
    for (int i = 0; i < mnoderowmap_->num_my_elements(); ++i)
    {
      int gid = mnoderowmap_->gid(i);
      Core::Nodes::Node* node = idiscret_->g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(node);

      // fill nodal matrix
      std::shared_ptr<Core::LinAlg::SerialDenseMatrix> loc =
          std::make_shared<Core::LinAlg::SerialDenseMatrix>(3, 3);
      (*loc)(0, 0) = cnode->mo_data().n()[0];
      (*loc)(1, 0) = cnode->mo_data().n()[1];
      (*loc)(2, 0) = cnode->mo_data().n()[2];
      (*loc)(0, 1) = cnode->data().txi()[0];
      (*loc)(1, 1) = cnode->data().txi()[1];
      (*loc)(2, 1) = cnode->data().txi()[2];
      (*loc)(0, 2) = cnode->data().teta()[0];
      (*loc)(1, 2) = cnode->data().teta()[1];
      (*loc)(2, 2) = cnode->data().teta()[2];

      triad[gid] = loc;

      // fill nodal derivative vectors
      std::vector<Core::Gen::Pairedvector<int, double>>& derivn = cnode->data().get_deriv_n();
      std::vector<Core::Gen::Pairedvector<int, double>>& derivtxi = cnode->data().get_deriv_txi();
      std::vector<Core::Gen::Pairedvector<int, double>>& derivteta = cnode->data().get_deriv_teta();

      for (iter = derivn[0].begin(); iter != derivn[0].end(); ++iter)
      {
        n_x_key[gid].push_back(iter->first);
        n_x_val[gid].push_back(iter->second);
      }
      for (iter = derivn[1].begin(); iter != derivn[1].end(); ++iter)
      {
        n_y_key[gid].push_back(iter->first);
        n_y_val[gid].push_back(iter->second);
      }
      for (iter = derivn[2].begin(); iter != derivn[2].end(); ++iter)
      {
        n_z_key[gid].push_back(iter->first);
        n_z_val[gid].push_back(iter->second);
      }

      for (iter = derivtxi[0].begin(); iter != derivtxi[0].end(); ++iter)
      {
        txi_x_key[gid].push_back(iter->first);
        txi_x_val[gid].push_back(iter->second);
      }
      for (iter = derivtxi[1].begin(); iter != derivtxi[1].end(); ++iter)
      {
        txi_y_key[gid].push_back(iter->first);
        txi_y_val[gid].push_back(iter->second);
      }
      for (iter = derivtxi[2].begin(); iter != derivtxi[2].end(); ++iter)
      {
        txi_z_key[gid].push_back(iter->first);
        txi_z_val[gid].push_back(iter->second);
      }

      for (iter = derivteta[0].begin(); iter != derivteta[0].end(); ++iter)
      {
        teta_x_key[gid].push_back(iter->first);
        teta_x_val[gid].push_back(iter->second);
      }
      for (iter = derivteta[1].begin(); iter != derivteta[1].end(); ++iter)
      {
        teta_y_key[gid].push_back(iter->first);
        teta_y_val[gid].push_back(iter->second);
      }
      for (iter = derivteta[2].begin(); iter != derivteta[2].end(); ++iter)
      {
        teta_z_key[gid].push_back(iter->first);
        teta_z_val[gid].push_back(iter->second);
      }
    }

    // communicate from master node row to column map
    Core::Communication::Exporter ex(*mnoderowmap_, *masternodes, get_comm());
    ex.do_export(triad);

    ex.do_export(n_x_key);
    ex.do_export(n_x_val);
    ex.do_export(n_y_key);
    ex.do_export(n_y_val);
    ex.do_export(n_z_key);
    ex.do_export(n_z_val);

    ex.do_export(txi_x_key);
    ex.do_export(txi_x_val);
    ex.do_export(txi_y_key);
    ex.do_export(txi_y_val);
    ex.do_export(txi_z_key);
    ex.do_export(txi_z_val);

    ex.do_export(teta_x_key);
    ex.do_export(teta_x_val);
    ex.do_export(teta_y_key);
    ex.do_export(teta_y_val);
    ex.do_export(teta_z_key);
    ex.do_export(teta_z_val);

    // extract info on column map
    for (int i = 0; i < masternodes->num_my_elements(); ++i)
    {
      // only do something for ghosted nodes
      int gid = masternodes->gid(i);
      Core::Nodes::Node* node = idiscret_->g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(node);
      int linsize = cnode->get_linsize() + (int)(n_x_key[gid].size());

      if (cnode->owner() == Core::Communication::my_mpi_rank(get_comm())) continue;

      // extract info
      std::shared_ptr<Core::LinAlg::SerialDenseMatrix> loc = triad[gid];
      cnode->mo_data().n()[0] = (*loc)(0, 0);
      cnode->mo_data().n()[1] = (*loc)(1, 0);
      cnode->mo_data().n()[2] = (*loc)(2, 0);
      cnode->data().txi()[0] = (*loc)(0, 1);
      cnode->data().txi()[1] = (*loc)(1, 1);
      cnode->data().txi()[2] = (*loc)(2, 1);
      cnode->data().teta()[0] = (*loc)(0, 2);
      cnode->data().teta()[1] = (*loc)(1, 2);
      cnode->data().teta()[2] = (*loc)(2, 2);

      // extract derivative info
      std::vector<Core::Gen::Pairedvector<int, double>>& derivn = cnode->data().get_deriv_n();
      std::vector<Core::Gen::Pairedvector<int, double>>& derivtxi = cnode->data().get_deriv_txi();
      std::vector<Core::Gen::Pairedvector<int, double>>& derivteta = cnode->data().get_deriv_teta();

      for (int k = 0; k < (int)(derivn.size()); ++k) derivn[k].clear();
      derivn.resize(3, linsize);
      for (int k = 0; k < (int)(derivtxi.size()); ++k) derivtxi[k].clear();
      derivtxi.resize(3, linsize);
      for (int k = 0; k < (int)(derivteta.size()); ++k) derivteta[k].clear();
      derivteta.resize(3, linsize);

      cnode->data().get_deriv_n()[0].resize(linsize);
      cnode->data().get_deriv_n()[1].resize(linsize);
      cnode->data().get_deriv_n()[2].resize(linsize);

      cnode->data().get_deriv_txi()[0].resize(linsize);
      cnode->data().get_deriv_txi()[1].resize(linsize);
      cnode->data().get_deriv_txi()[2].resize(linsize);

      cnode->data().get_deriv_teta()[0].resize(linsize);
      cnode->data().get_deriv_teta()[1].resize(linsize);
      cnode->data().get_deriv_teta()[2].resize(linsize);

      for (int k = 0; k < (int)(n_x_key[gid].size()); ++k)
        (cnode->data().get_deriv_n()[0])[n_x_key[gid][k]] = n_x_val[gid][k];
      for (int k = 0; k < (int)(n_y_key[gid].size()); ++k)
        (cnode->data().get_deriv_n()[1])[n_y_key[gid][k]] = n_y_val[gid][k];
      for (int k = 0; k < (int)(n_z_key[gid].size()); ++k)
        (cnode->data().get_deriv_n()[2])[n_z_key[gid][k]] = n_z_val[gid][k];

      for (int k = 0; k < (int)(txi_x_key[gid].size()); ++k)
        (cnode->data().get_deriv_txi()[0])[txi_x_key[gid][k]] = txi_x_val[gid][k];
      for (int k = 0; k < (int)(txi_y_key[gid].size()); ++k)
        (cnode->data().get_deriv_txi()[1])[txi_y_key[gid][k]] = txi_y_val[gid][k];
      for (int k = 0; k < (int)(txi_z_key[gid].size()); ++k)
        (cnode->data().get_deriv_txi()[2])[txi_z_key[gid][k]] = txi_z_val[gid][k];

      for (int k = 0; k < (int)(teta_x_key[gid].size()); ++k)
        (cnode->data().get_deriv_teta()[0])[teta_x_key[gid][k]] = teta_x_val[gid][k];
      for (int k = 0; k < (int)(teta_y_key[gid].size()); ++k)
        (cnode->data().get_deriv_teta()[1])[teta_y_key[gid][k]] = teta_y_val[gid][k];
      for (int k = 0; k < (int)(teta_z_key[gid].size()); ++k)
        (cnode->data().get_deriv_teta()[2])[teta_z_key[gid][k]] = teta_z_val[gid][k];
    }

    // free memory
    triad.clear();

    n_x_key.clear();
    n_y_key.clear();
    n_z_key.clear();
    txi_x_key.clear();
    txi_y_key.clear();
    txi_z_key.clear();
    teta_x_key.clear();
    teta_y_key.clear();
    teta_z_key.clear();

    n_x_val.clear();
    n_y_val.clear();
    n_z_val.clear();
    txi_x_val.clear();
    txi_y_val.clear();
    txi_z_val.clear();
    teta_x_val.clear();
    teta_y_val.clear();
    teta_z_val.clear();
  }

  return;
}


/*----------------------------------------------------------------------*
 |  Assemble matrix S containing gap g~ derivatives          farah 09/13|
 |  PS: "AssembleS" is an outdated name which could make                |
 |  you confused.                                                       |
 *----------------------------------------------------------------------*/
void Wear::WearInterface::assemble_s(Core::LinAlg::SparseMatrix& sglobal)
{
  // call contact function
  CONTACT::Interface::assemble_s(sglobal);

  // nothing to do if no active nodes
  if (activenodes_ == nullptr) return;

  // loop over all active slave nodes of the interface
  for (int i = 0; i < activenodes_->num_my_elements(); ++i)
  {
    int gid = activenodes_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(node);

    if (cnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
      FOUR_C_THROW("AssembleS: Node ownership inconsistency!");

    // prepare assembly
    std::map<int, double>::iterator colcurr;
    int row = activen_->gid(i);

    /*************************************************************************
     * Wear implicit linearization   --> obviously, we need a new linear.    *
     *************************************************************************/
    if (wearimpl_ and !wearpv_)
    {
      // prepare assembly
      std::map<int, double>& dwmap = cnode->data().get_deriv_w();

      for (colcurr = dwmap.begin(); colcurr != dwmap.end(); ++colcurr)
      {
        int col = colcurr->first;
        double val = colcurr->second;
        // std::cout << "Assemble S: " << row << " " << col << " " << val << endl;
        // do not assemble zeros into s matrix
        if (abs(val) > 1.0e-12) sglobal.assemble(val, row, col);
      }
    }
  }  // for (int i=0;i<activenodes_->NumMyElements();++i)

  return;
}


/*----------------------------------------------------------------------*
 |  Assemble matrix S containing gap g~ w derivatives       farah 09/13|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::assemble_lin_g_w(Core::LinAlg::SparseMatrix& sglobal)
{
  // nothing to do if no active nodes
  if (activenodes_ == nullptr) return;

  // loop over all active slave nodes of the interface
  for (int i = 0; i < activenodes_->num_my_elements(); ++i)
  {
    int gid = activenodes_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(node);

    if (cnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
      FOUR_C_THROW("AssembleS: Node ownership inconsistency!");

    // prepare assembly
    std::map<int, double>& dgmap = cnode->data().get_deriv_gw();
    std::map<int, double>::iterator colcurr;
    int row = activen_->gid(i);

    for (colcurr = dgmap.begin(); colcurr != dgmap.end(); ++colcurr)
    {
      int col = colcurr->first;
      double val = colcurr->second;
      // std::cout << "Assemble S: " << row << " " << col << " " << val << std::endl;
      // do not assemble zeros into s matrix
      if (abs(val) > 1.0e-12) sglobal.assemble(val, row, col);
    }
  }  // for (int i=0;i<activenodes_->NumMyElements();++i)

  return;
}


/*----------------------------------------------------------------------*
 |  Assemble matrix LinStick with tangential+D+M derivatives  mgit 02/09|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::assemble_lin_stick(Core::LinAlg::SparseMatrix& linstickLMglobal,
    Core::LinAlg::SparseMatrix& linstickDISglobal, Core::LinAlg::Vector<double>& linstickRHSglobal)
{
  // create map of stick nodes
  std::shared_ptr<Core::LinAlg::Map> sticknodes =
      Core::LinAlg::split_map(*activenodes_, *slipnodes_);
  std::shared_ptr<Core::LinAlg::Map> stickt = Core::LinAlg::split_map(*activet_, *slipt_);

  // nothing to do if no stick nodes
  if (sticknodes->num_my_elements() == 0) return;

  // information from interface contact parameter list
  const double frcoeff = interface_params().get<double>("FRCOEFF");
  double ct = interface_params().get<double>("SEMI_SMOOTH_CT");
  double cn = interface_params().get<double>("SEMI_SMOOTH_CN");

  auto ftype = Teuchos::getIntegralValue<CONTACT::FrictionType>(interface_params(), "FRICTION");

  bool consistent = false;

#if defined(CONSISTENTSTICK) && defined(CONSISTENTSLIP)
  FOUR_C_THROW(
      "It's not reasonable to activate both, the consistent stick and slip branch, "
      "because both together will lead again to an inconsistent formulation!");
#endif
#ifdef CONSISTENTSTICK
  consistent = true;
#endif

  if (consistent && ftype == CONTACT::FrictionType::coulomb)
  {
    // loop over all stick nodes of the interface
    for (int i = 0; i < sticknodes->num_my_elements(); ++i)
    {
      int gid = sticknodes->gid(i);
      Core::Nodes::Node* node = idiscret_->g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(node);

      if (cnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
        FOUR_C_THROW("AssembleLinStick: Node ownership inconsistency!");

      cn = cn_ref().local_values_as_span()[cn_ref().get_map().lid(cnode->id())];
      ct = ct_ref().local_values_as_span()[ct_ref().get_map().lid(cnode->id())];

      // prepare assembly, get information from node
      std::vector<Core::Gen::Pairedvector<int, double>> dnmap = cnode->data().get_deriv_n();
      std::vector<Core::Gen::Pairedvector<int, double>> dtximap = cnode->data().get_deriv_txi();
      std::vector<Core::Gen::Pairedvector<int, double>> dtetamap = cnode->data().get_deriv_teta();
      std::map<int, double> dgmap = cnode->data().get_deriv_g();

      // check for Dimension of derivative maps
      for (int j = 0; j < n_dim() - 1; ++j)
        if ((int)dnmap[j].size() != (int)dnmap[j + 1].size())
          FOUR_C_THROW("AssembleLinStick: Column dim. of nodal DerivN-map is inconsistent!");

      for (int j = 0; j < n_dim() - 1; ++j)
        if ((int)dtximap[j].size() != (int)dtximap[j + 1].size())
          FOUR_C_THROW("AssembleLinStick: Column dim. of nodal DerivTxi-map is inconsistent!");

      if (n_dim() == 3)
      {
        for (int j = 0; j < n_dim() - 1; ++j)
          if ((int)dtximap[j].size() != (int)dtximap[j + 1].size())
            FOUR_C_THROW("AssembleLinStick: Column dim. of nodal DerivTeta-map is inconsistent!");
      }

      // more information from node
      double* n = cnode->mo_data().n();
      double* z = cnode->mo_data().lm();
      double& wgap = cnode->data().getg();

      // iterator for maps
      std::map<int, double>::iterator colcurr;
      Core::Gen::Pairedvector<int, double>::iterator _colcurr;

      // row number of entries
      std::vector<int> row(n_dim() - 1);
      if (n_dim() == 2)
      {
        row[0] = stickt->gid(i);
      }
      else if (n_dim() == 3)
      {
        row[0] = stickt->gid(2 * i);
        row[1] = stickt->gid(2 * i) + 1;
      }
      else
        FOUR_C_THROW("AssemblelinStick: Dimension not correct");

      // evaluation of specific components of entries to assemble
      double znor = 0;
      double jumptxi = 0;
      double jumpteta = 0;
      double* jump = cnode->fri_data().jump();
      double* txi = cnode->data().txi();
      double* teta = cnode->data().teta();

      for (int i = 0; i < n_dim(); i++) znor += n[i] * z[i];

      // for slip
      if (interface_params().get<bool>("GP_SLIP_INCR"))
      {
        jumptxi = cnode->fri_data().jump_var()[0];

        if (n_dim() == 3) jumpteta = cnode->fri_data().jump_var()[1];
      }
      else
      {
        // more information from node
        for (int i = 0; i < n_dim(); i++)
        {
          jumptxi += txi[i] * jump[i];
          jumpteta += teta[i] * jump[i];
        }
      }

      // check for dimensions
      if (n_dim() == 2 and (jumpteta != 0.0))
        FOUR_C_THROW("AssembleLinStick: jumpteta must be zero in 2D");

      //**************************************************************
      // calculation of matrix entries of linearized stick condition
      //**************************************************************

      // 1) Entries from differentiation with respect to LM
      /******************************************************************/

      // loop over the dimension
      const int numdof = cnode->num_dof();
      for (int dim = 0; dim < numdof; ++dim)
      {
        double valtxi = 0.0;
        double valteta = 0.0;
        int col = cnode->dofs()[dim];
        valtxi = -frcoeff * ct * jumptxi * n[dim];

        if (n_dim() == 3)
        {
          valteta = -frcoeff * ct * jumpteta * n[dim];
        }
        // do not assemble zeros into matrix
        if (abs(valtxi) > 1.0e-12) linstickLMglobal.assemble(valtxi, row[0], col);
        if (n_dim() == 3)
          if (abs(valteta) > 1.0e-12) linstickLMglobal.assemble(valteta, row[1], col);
      }

      // Entries on right hand side ****************************
      Core::LinAlg::SerialDenseVector rhsnode(n_dim() - 1);
      std::vector<int> lm(n_dim() - 1);
      std::vector<int> lmowner(n_dim() - 1);
      rhsnode(0) = frcoeff * (znor - cn * wgap) * ct * jumptxi;

      lm[0] = cnode->dofs()[1];
      lmowner[0] = cnode->owner();
      if (n_dim() == 3)
      {
        rhsnode(1) = frcoeff * (znor - cn * wgap) * ct * jumpteta;
        lm[1] = cnode->dofs()[2];
        lmowner[1] = cnode->owner();
      }

      Core::LinAlg::assemble(linstickRHSglobal, rhsnode, lm, lmowner);

      // 3) Entries from differentiation with respect to displacements
      /******************************************************************/

      if (interface_params().get<bool>("GP_SLIP_INCR"))
      {
        std::vector<std::map<int, double>> derivjump_ = cnode->fri_data().get_deriv_var_jump();

        // txi
        for (colcurr = derivjump_[0].begin(); colcurr != derivjump_[0].end(); ++colcurr)
        {
          int col = colcurr->first;
          double valtxi = -frcoeff * (znor - cn * wgap) * ct * (colcurr->second);
          if (abs(valtxi) > 1.0e-12) linstickDISglobal.assemble(valtxi, row[0], col);
        }
        // teta
        for (colcurr = derivjump_[1].begin(); colcurr != derivjump_[1].end(); ++colcurr)
        {
          int col = colcurr->first;
          double valteta = -frcoeff * (znor - cn * wgap) * ct * (colcurr->second);
          if (abs(valteta) > 1.0e-12) linstickDISglobal.assemble(valteta, row[1], col);
        }

        // ... old slip
        // get linearization of jump vector
        std::vector<std::map<int, double>> derivjump = cnode->fri_data().get_deriv_jump();

        // loop over dimensions
        for (int dim = 0; dim < numdof; ++dim)
        {
          // linearization of normal direction *****************************************
          // loop over all entries of the current derivative map
          for (_colcurr = dnmap[dim].begin(); _colcurr != dnmap[dim].end(); ++_colcurr)
          {
            int col = _colcurr->first;
            double valtxi = 0.0;
            valtxi = -frcoeff * z[dim] * _colcurr->second * ct * jumptxi;
            // do not assemble zeros into matrix
            if (abs(valtxi) > 1.0e-12) linstickDISglobal.assemble(valtxi, row[0], col);

            if (n_dim() == 3)
            {
              double valteta = 0.0;
              valteta = -frcoeff * z[dim] * _colcurr->second * ct * jumpteta;
              // do not assemble zeros into matrix
              if (abs(valteta) > 1.0e-12) linstickDISglobal.assemble(valteta, row[1], col);
            }
          }
        }  // loop over all dimensions
      }
      else  // std slip
      {
        // ... old slip
        // get linearization of jump vector
        std::vector<std::map<int, double>> derivjump = cnode->fri_data().get_deriv_jump();

        // loop over dimensions
        for (int dim = 0; dim < numdof; ++dim)
        {
          // loop over all entries of the current derivative map (jump)
          for (colcurr = derivjump[dim].begin(); colcurr != derivjump[dim].end(); ++colcurr)
          {
            int col = colcurr->first;

            double valtxi = 0.0;
            valtxi = -frcoeff * (znor - cn * wgap) * ct * txi[dim] * (colcurr->second);
            // do not assemble zeros into matrix
            if (abs(valtxi) > 1.0e-12) linstickDISglobal.assemble(valtxi, row[0], col);

            if (n_dim() == 3)
            {
              double valteta = 0.0;
              valteta = -frcoeff * (znor - cn * wgap) * ct * teta[dim] * (colcurr->second);
              // do not assemble zeros into matrix
              if (abs(valteta) > 1.0e-12) linstickDISglobal.assemble(valteta, row[1], col);
            }
          }

          // linearization first tangential direction *********************************
          // loop over all entries of the current derivative map (txi)
          for (_colcurr = dtximap[dim].begin(); _colcurr != dtximap[dim].end(); ++_colcurr)
          {
            int col = _colcurr->first;
            double valtxi = 0.0;
            valtxi = -frcoeff * (znor - cn * wgap) * ct * jump[dim] * _colcurr->second;

            // do not assemble zeros into matrix
            if (abs(valtxi) > 1.0e-12) linstickDISglobal.assemble(valtxi, row[0], col);
          }
          // linearization second tangential direction *********************************
          if (n_dim() == 3)
          {
            // loop over all entries of the current derivative map (teta)
            for (_colcurr = dtetamap[dim].begin(); _colcurr != dtetamap[dim].end(); ++_colcurr)
            {
              int col = _colcurr->first;
              double valteta = 0.0;
              valteta = -frcoeff * (znor - cn * wgap) * ct * jump[dim] * _colcurr->second;

              // do not assemble zeros into matrix
              if (abs(valteta) > 1.0e-12) linstickDISglobal.assemble(valteta, row[1], col);
            }
          }
          // linearization of normal direction *****************************************
          // loop over all entries of the current derivative map
          for (_colcurr = dnmap[dim].begin(); _colcurr != dnmap[dim].end(); ++_colcurr)
          {
            int col = _colcurr->first;
            double valtxi = 0.0;
            valtxi = -frcoeff * z[dim] * _colcurr->second * ct * jumptxi;
            // do not assemble zeros into matrix
            if (abs(valtxi) > 1.0e-12) linstickDISglobal.assemble(valtxi, row[0], col);

            if (n_dim() == 3)
            {
              double valteta = 0.0;
              valteta = -frcoeff * z[dim] * _colcurr->second * ct * jumpteta;
              // do not assemble zeros into matrix
              if (abs(valteta) > 1.0e-12) linstickDISglobal.assemble(valteta, row[1], col);
            }
          }
        }  // loop over all dimensions
      }

      // linearization of weighted gap**********************************************
      // loop over all entries of the current derivative map fixme
      for (colcurr = dgmap.begin(); colcurr != dgmap.end(); ++colcurr)
      {
        int col = colcurr->first;
        double valtxi = 0.0;
        valtxi = frcoeff * colcurr->second * ct * cn * jumptxi;
        // do not assemble zeros into matrix
        if (abs(valtxi) > 1e-12) linstickDISglobal.assemble(valtxi, row[0], col);
        if (n_dim() == 3)
        {
          double valteta = 0.0;
          valteta = frcoeff * colcurr->second * ct * cn * jumpteta;
          // do not assemble zeros into matrix
          if (abs(valteta) > 1.0e-12) linstickDISglobal.assemble(valteta, row[1], col);
        }
      }
      if (wearimpl_ and !wearpv_)
      {
        // linearization of weighted wear w.r.t. displacements **********************
        std::map<int, double>& dwmap = cnode->data().get_deriv_w();
        for (colcurr = dwmap.begin(); colcurr != dwmap.end(); ++colcurr)
        {
          int col = colcurr->first;
          double valtxi = 0.0;
          valtxi = frcoeff * colcurr->second * ct * cn * jumptxi;
          // do not assemble zeros into matrix
          if (abs(valtxi) > 1e-12) linstickDISglobal.assemble(valtxi, row[0], col);
          if (n_dim() == 3)
          {
            double valteta = 0.0;
            valteta = frcoeff * colcurr->second * ct * cn * jumpteta;
            // do not assemble zeros into matrix
            if (abs(valteta) > 1.0e-12) linstickDISglobal.assemble(valteta, row[1], col);
          }
        }
      }  // if wearimpl_
    }  // loop over stick nodes
  }
  else
  {
    // loop over all stick nodes of the interface
    for (int i = 0; i < sticknodes->num_my_elements(); ++i)
    {
      int gid = sticknodes->gid(i);
      Core::Nodes::Node* node = idiscret_->g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(node);

      if (cnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
        FOUR_C_THROW("AssembleLinStick: Node ownership inconsistency!");

      // prepare assembly, get information from node
      std::vector<Core::Gen::Pairedvector<int, double>> dtximap = cnode->data().get_deriv_txi();
      std::vector<Core::Gen::Pairedvector<int, double>> dtetamap = cnode->data().get_deriv_teta();

      for (int j = 0; j < n_dim() - 1; ++j)
        if ((int)dtximap[j].size() != (int)dtximap[j + 1].size())
          FOUR_C_THROW("AssembleLinStick: Column dim. of nodal DerivTxi-map is inconsistent!");

      if (n_dim() == 3)
      {
        for (int j = 0; j < n_dim() - 1; ++j)
          if ((int)dtximap[j].size() != (int)dtximap[j + 1].size())
            FOUR_C_THROW("AssembleLinStick: Column dim. of nodal DerivTeta-map is inconsistent!");
      }

      // iterator for maps
      std::map<int, double>::iterator colcurr;
      Core::Gen::Pairedvector<int, double>::iterator _colcurr;

      // row number of entries
      std::vector<int> row(n_dim() - 1);
      if (n_dim() == 2)
      {
        row[0] = stickt->gid(i);
      }
      else if (n_dim() == 3)
      {
        row[0] = stickt->gid(2 * i);
        row[1] = stickt->gid(2 * i) + 1;
      }
      else
        FOUR_C_THROW("AssemblelinStick: Dimension not correct");

      // evaluation of specific components of entries to assemble
      double jumptxi = 0;
      double jumpteta = 0;

      // more information from node
      double* txi = cnode->data().txi();
      double* teta = cnode->data().teta();
      double* jump = cnode->fri_data().jump();

      // slip
      if (interface_params().get<bool>("GP_SLIP_INCR"))
      {
        jumptxi = cnode->fri_data().jump_var()[0];

        if (n_dim() == 3) jumpteta = cnode->fri_data().jump_var()[1];
      }
      else
      {
        for (int i = 0; i < n_dim(); i++)
        {
          jumptxi += txi[i] * jump[i];
          jumpteta += teta[i] * jump[i];
        }
      }

      // check for dimensions
      if (n_dim() == 2 and (jumpteta != 0.0))
        FOUR_C_THROW("AssembleLinStick: jumpteta must be zero in 2D");

      // Entries on right hand side
      /************************************************ (-utxi, -uteta) ***/
      Core::LinAlg::SerialDenseVector rhsnode(n_dim() - 1);
      std::vector<int> lm(n_dim() - 1);
      std::vector<int> lmowner(n_dim() - 1);

      // modification to stabilize the convergence of the lagrange multiplier incr (hiermeier 08/13)
      if (abs(jumptxi) < 1e-15)
        rhsnode(0) = 0.0;
      else
        rhsnode(0) = -jumptxi;

      lm[0] = cnode->dofs()[1];
      lmowner[0] = cnode->owner();

      if (n_dim() == 3)
      {
        // modification to stabilize the convergence of the lagrange multiplier incr (hiermeier
        // 08/13)
        if (abs(jumpteta) < 1e-15)
          rhsnode(1) = 0.0;
        else
          rhsnode(1) = -jumpteta;

        lm[1] = cnode->dofs()[2];
        lmowner[1] = cnode->owner();
      }

      Core::LinAlg::assemble(linstickRHSglobal, rhsnode, lm, lmowner);

      // Entries from differentiation with respect to displacements
      /*** 1 ************************************** tangent.deriv(jump) ***/
      if (interface_params().get<bool>("GP_SLIP_INCR"))
      {
        std::map<int, double> derivjump1 = cnode->fri_data().get_deriv_var_jump()[0];
        std::map<int, double> derivjump2 = cnode->fri_data().get_deriv_var_jump()[1];

        for (colcurr = derivjump1.begin(); colcurr != derivjump1.end(); ++colcurr)
        {
          int col = colcurr->first;
          double valtxi = colcurr->second;

          if (abs(valtxi) > 1.0e-12) linstickDISglobal.assemble(valtxi, row[0], col);
        }

        if (n_dim() == 3)
        {
          for (colcurr = derivjump2.begin(); colcurr != derivjump2.end(); ++colcurr)
          {
            int col = colcurr->first;
            double valteta = colcurr->second;

            if (abs(valteta) > 1.0e-12) linstickDISglobal.assemble(valteta, row[1], col);
          }
        }
      }
      else  // std slip
      {
        // get linearization of jump vector
        std::vector<std::map<int, double>> derivjump = cnode->fri_data().get_deriv_jump();

        if (derivjump.size() < 1)
          FOUR_C_THROW("AssembleLinStick: Derivative of jump is not exiting!");

        // loop over dimensions
        for (int dim = 0; dim < cnode->num_dof(); ++dim)
        {
          // loop over all entries of the current derivative map (jump)
          for (colcurr = derivjump[dim].begin(); colcurr != derivjump[dim].end(); ++colcurr)
          {
            int col = colcurr->first;
            double valtxi = txi[dim] * colcurr->second;

            // do not assemble zeros into matrix
            if (abs(valtxi) > 1.0e-12) linstickDISglobal.assemble(valtxi, row[0], col);

            if (n_dim() == 3)
            {
              double valteta = teta[dim] * colcurr->second;
              if (abs(valteta) > 1.0e-12) linstickDISglobal.assemble(valteta, row[1], col);
            }
          }
        }

        /*** 2 ************************************** deriv(tangent).jump ***/
        // loop over dimensions
        for (int j = 0; j < n_dim(); ++j)
        {
          // loop over all entries of the current derivative map (txi)
          for (_colcurr = dtximap[j].begin(); _colcurr != dtximap[j].end(); ++_colcurr)
          {
            int col = _colcurr->first;
            double val = jump[j] * _colcurr->second;

            // do not assemble zeros into s matrix
            if (abs(val) > 1.0e-12) linstickDISglobal.assemble(val, row[0], col);
          }

          if (n_dim() == 3)
          {
            // loop over all entries of the current derivative map (teta)
            for (_colcurr = dtetamap[j].begin(); _colcurr != dtetamap[j].end(); ++_colcurr)
            {
              int col = _colcurr->first;
              double val = jump[j] * _colcurr->second;

              // do not assemble zeros into matrix
              if (abs(val) > 1.0e-12) linstickDISglobal.assemble(val, row[1], col);
            }
          }
        }
      }
    }
  }
  return;
}

/*----------------------------------------------------------------------*
|  Assemble matrix LinSlip with W derivatives               farah 09/13|
*----------------------------------------------------------------------*/
void Wear::WearInterface::assemble_lin_slip_w(Core::LinAlg::SparseMatrix& linslipWglobal)
{
  // nothing to do if no slip nodes
  if (slipnodes_->num_my_elements() == 0) return;

  // information from interface contact parameter list
  auto ftype = Teuchos::getIntegralValue<CONTACT::FrictionType>(interface_params(), "FRICTION");
  double frcoeff = interface_params().get<double>("FRCOEFF");
  double ct = interface_params().get<double>("SEMI_SMOOTH_CT");
  double cn = interface_params().get<double>("SEMI_SMOOTH_CN");

  //**********************************************************************
  //**********************************************************************
  //**********************************************************************
  // Coulomb Friction
  //**********************************************************************
  //**********************************************************************
  //**********************************************************************
  if (ftype == CONTACT::FrictionType::coulomb)
  {
    // loop over all slip nodes of the interface
    for (int i = 0; i < slipnodes_->num_my_elements(); ++i)
    {
      int gid = slipnodes_->gid(i);
      Core::Nodes::Node* node = idiscret_->g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(node);

      if (cnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
        FOUR_C_THROW("AssembleLinSlip: Node ownership inconsistency!");

      cn = cn_ref().local_values_as_span()[cn_ref().get_map().lid(cnode->id())];
      ct = ct_ref().local_values_as_span()[ct_ref().get_map().lid(cnode->id())];

      // prepare assembly, get information from node
      std::vector<Core::Gen::Pairedvector<int, double>> dnmap = cnode->data().get_deriv_n();
      std::vector<Core::Gen::Pairedvector<int, double>> dtximap = cnode->data().get_deriv_txi();
      std::vector<Core::Gen::Pairedvector<int, double>> dtetamap = cnode->data().get_deriv_teta();

      // check for Dimension of derivative maps
      for (int j = 0; j < n_dim() - 1; ++j)
        if ((int)dnmap[j].size() != (int)dnmap[j + 1].size())
          FOUR_C_THROW("AssembleLinSlip: Column dim. of nodal DerivTxi-map is inconsistent!");

      for (int j = 0; j < n_dim() - 1; ++j)
        if ((int)dtximap[j].size() != (int)dtximap[j + 1].size())
          FOUR_C_THROW("AssembleLinSlip: Column dim. of nodal DerivTxi-map is inconsistent!");

      if (n_dim() == 3)
      {
        for (int j = 0; j < n_dim() - 1; ++j)
          if ((int)dtximap[j].size() != (int)dtximap[j + 1].size())
            FOUR_C_THROW("AssembleLinSlip: Column dim. of nodal DerivTeta-map is inconsistent!");
      }

      // more information from node
      double* txi = cnode->data().txi();
      double* teta = cnode->data().teta();
      double* z = cnode->mo_data().lm();

      // iterator for maps
      std::map<int, double>::iterator colcurr;

      // row number of entries
      std::vector<int> row(n_dim() - 1);
      if (n_dim() == 2)
      {
        row[0] = slipt_->gid(i);
      }
      else if (n_dim() == 3)
      {
        row[0] = slipt_->gid(2 * i);
        row[1] = slipt_->gid(2 * i) + 1;
      }
      else
        FOUR_C_THROW("AssemblelinSlip: Dimension not correct");

      // boolean variable if flag "CONTACTFRICTIONLESSFIRST" AND
      // ActiveOld = true
      bool friclessandfirst = false;

      // evaluation of specific components of entries to assemble
      double ztxi = 0;
      double zteta = 0;
      double jumptxi = 0;
      double jumpteta = 0;
      double euclidean = 0;
      double* jump = cnode->fri_data().jump();

      // for slip
      if (interface_params().get<bool>("GP_SLIP_INCR"))
      {
        jumptxi = cnode->fri_data().jump_var()[0];

        if (n_dim() == 3) jumpteta = cnode->fri_data().jump_var()[1];

        for (int i = 0; i < n_dim(); i++)
        {
          ztxi += txi[i] * z[i];
          zteta += teta[i] * z[i];
        }
      }
      else
      {
        for (int i = 0; i < n_dim(); i++)
        {
          ztxi += txi[i] * z[i];
          zteta += teta[i] * z[i];
          jumptxi += txi[i] * jump[i];
          jumpteta += teta[i] * jump[i];
        }
      }

      // evaluate euclidean norm ||vec(zt)+ct*vec(jump)||
      std::vector<double> sum1(n_dim() - 1, 0);
      sum1[0] = ztxi + ct * jumptxi;
      if (n_dim() == 3) sum1[1] = zteta + ct * jumpteta;
      if (n_dim() == 2) euclidean = abs(sum1[0]);
      if (n_dim() == 3) euclidean = sqrt(sum1[0] * sum1[0] + sum1[1] * sum1[1]);

      // check of dimensions
      if (n_dim() == 2 and (zteta != 0.0 or jumpteta != 0.0))
        FOUR_C_THROW("AssemblelinSlip: zteta and jumpteta must be zero in 2D");

      // check of euclidean norm
      if (euclidean == 0.0) FOUR_C_THROW("AssemblelinSlip: Euclidean norm is zero");

      // this is not evaluated if "FRICTIONLESSFIRST" is flagged on AND the node
      // is just coming into contact
      if (friclessandfirst == false)
      {
        //****************************************************************
        // CONSISTENT TREATMENT OF CASE FRCOEFF=0 (FRICTIONLESS)
        //****************************************************************
        // popp 08/2012
        //
        // There is a problem with our frictional nonlinear complementarity
        // function when applied to the limit case frcoeff=0 (frictionless).
        // In this case, the simple frictionless sliding condition should
        // be consistently recovered, which unfortunately is not the case.
        // This fact is well-known (see PhD thesis S. Hueeber) and now
        // taken care of by a special treatment as can be seen below
        //
        //****************************************************************
        if (frcoeff == 0.0)
        {
          // coming soon....
        }

        //****************************************************************
        // STANDARD TREATMENT OF CASE FRCOEFF!=0 (FRICTIONAL)
        //****************************************************************
        else
        {
          //**************************************************************
          // calculation of matrix entries of linearized slip condition
          //**************************************************************

          /*** 1 ****************** frcoeff*cn*deriv (g).(ztan+ct*utan) ***/
          // prepare assembly
          std::map<int, double>& dgwmap = cnode->data().get_deriv_gw();

          // loop over all entries of the current derivative map
          for (colcurr = dgwmap.begin(); colcurr != dgwmap.end(); ++colcurr)
          {
            int col = colcurr->first;
            double valtxi = frcoeff * cn * (colcurr->second) * (ztxi + ct * jumptxi);
            double valteta = frcoeff * cn * (colcurr->second) * (zteta + ct * jumpteta);

            // do not assemble zeros into matrix
            if (abs(valtxi) > 1.0e-12) linslipWglobal.assemble(valtxi, row[0], col);
            if (abs(valteta) > 1.0e-12) linslipWglobal.assemble(valteta, row[1], col);
          }
        }  // if (frcoeff==0.0)
      }  // if (frictionlessandfirst == false)
    }  // loop over all slip nodes of the interface
  }  // Coulomb friction
  else
    FOUR_C_THROW("linslip wear only for coulomb friction!");
}

/*----------------------------------------------------------------------*
|  Assemble matrix LinSlip with tangential+D+M derivatives    mgit 02/09|
*----------------------------------------------------------------------*/
void Wear::WearInterface::assemble_lin_slip(Core::LinAlg::SparseMatrix& linslipLMglobal,
    Core::LinAlg::SparseMatrix& linslipDISglobal, Core::LinAlg::Vector<double>& linslipRHSglobal)
{
  // nothing to do if no slip nodes
  if (slipnodes_->num_my_elements() == 0) return;

  // information from interface contact parameter list
  auto ftype = Teuchos::getIntegralValue<CONTACT::FrictionType>(interface_params(), "FRICTION");
  double frcoeff = interface_params().get<double>("FRCOEFF");
  double ct = interface_params().get<double>("SEMI_SMOOTH_CT");
  double cn = interface_params().get<double>("SEMI_SMOOTH_CN");

  //**********************************************************************
  //**********************************************************************
  //**********************************************************************
  // Coulomb Friction
  //**********************************************************************
  //**********************************************************************
  //**********************************************************************
  if (ftype == CONTACT::FrictionType::coulomb)
  {
    // loop over all slip nodes of the interface
    for (int i = 0; i < slipnodes_->num_my_elements(); ++i)
    {
      int gid = slipnodes_->gid(i);
      Core::Nodes::Node* node = idiscret_->g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(node);

      if (cnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
        FOUR_C_THROW("AssembleLinSlip: Node ownership inconsistency!");

      cn = cn_ref().local_values_as_span()[cn_ref().get_map().lid(cnode->id())];
      ct = ct_ref().local_values_as_span()[ct_ref().get_map().lid(cnode->id())];

      // prepare assembly, get information from node
      std::vector<Core::Gen::Pairedvector<int, double>> dnmap = cnode->data().get_deriv_n();
      std::vector<Core::Gen::Pairedvector<int, double>> dtximap = cnode->data().get_deriv_txi();
      std::vector<Core::Gen::Pairedvector<int, double>> dtetamap = cnode->data().get_deriv_teta();

      // check for Dimension of derivative maps
      for (int j = 0; j < n_dim() - 1; ++j)
        if ((int)dnmap[j].size() != (int)dnmap[j + 1].size())
          FOUR_C_THROW("AssembleLinSlip: Column dim. of nodal DerivTxi-map is inconsistent!");

      for (int j = 0; j < n_dim() - 1; ++j)
        if ((int)dtximap[j].size() != (int)dtximap[j + 1].size())
          FOUR_C_THROW("AssembleLinSlip: Column dim. of nodal DerivTxi-map is inconsistent!");

      if (n_dim() == 3)
      {
        for (int j = 0; j < n_dim() - 1; ++j)
          if ((int)dtximap[j].size() != (int)dtximap[j + 1].size())
            FOUR_C_THROW("AssembleLinSlip: Column dim. of nodal DerivTeta-map is inconsistent!");
      }

      // more information from node
      double* n = cnode->mo_data().n();
      double* txi = cnode->data().txi();
      double* teta = cnode->data().teta();
      double* z = cnode->mo_data().lm();
      double& wgap = cnode->data().getg();

      // iterator for maps
      std::map<int, double>::iterator colcurr;
      Core::Gen::Pairedvector<int, double>::iterator _colcurr;

      // row number of entries
      std::vector<int> row(n_dim() - 1);
      if (n_dim() == 2)
      {
        row[0] = slipt_->gid(i);
      }
      else if (n_dim() == 3)
      {
        row[0] = slipt_->gid(2 * i);
        row[1] = slipt_->gid(2 * i) + 1;
      }
      else
        FOUR_C_THROW("AssemblelinSlip: Dimension not correct");

      // evaluation of specific components of entries to assemble
      double znor = 0;
      double ztxi = 0;
      double zteta = 0;
      double jumptxi = 0;
      double jumpteta = 0;
      double euclidean = 0;
      double* jump = cnode->fri_data().jump();

      // for gp slip
      if (interface_params().get<bool>("GP_SLIP_INCR"))
      {
        jumptxi = cnode->fri_data().jump_var()[0];

        if (n_dim() == 3) jumpteta = cnode->fri_data().jump_var()[1];

        for (int i = 0; i < n_dim(); i++)
        {
          znor += n[i] * z[i];
          ztxi += txi[i] * z[i];
          zteta += teta[i] * z[i];
        }
      }
      else
      {
        for (int i = 0; i < n_dim(); i++)
        {
          znor += n[i] * z[i];
          ztxi += txi[i] * z[i];
          zteta += teta[i] * z[i];
          jumptxi += txi[i] * jump[i];
          jumpteta += teta[i] * jump[i];
        }
      }

      // evaluate euclidean norm ||vec(zt)+ct*vec(jump)||
      std::vector<double> sum1(n_dim() - 1, 0);
      sum1[0] = ztxi + ct * jumptxi;
      if (n_dim() == 3) sum1[1] = zteta + ct * jumpteta;
      if (n_dim() == 2) euclidean = abs(sum1[0]);
      if (n_dim() == 3) euclidean = sqrt(sum1[0] * sum1[0] + sum1[1] * sum1[1]);

      // check of dimensions
      if (n_dim() == 2 and (zteta != 0.0 or jumpteta != 0.0))
        FOUR_C_THROW("AssemblelinSlip: zteta and jumpteta must be zero in 2D");

      // check of euclidean norm
      if (euclidean == 0.0)
      {
        std::cout << "owner= " << cnode->owner() << "  "
                  << Core::Communication::my_mpi_rank(get_comm()) << std::endl;
        FOUR_C_THROW("AssemblelinSlip: Euclidean norm is zero");
      }

      //****************************************************************
      // CONSISTENT TREATMENT OF CASE FRCOEFF=0 (FRICTIONLESS)
      //****************************************************************
      // popp 08/2012
      //
      // There is a problem with our frictional nonlinear complementarity
      // function when applied to the limit case frcoeff=0 (frictionless).
      // In this case, the simple frictionless sliding condition should
      // be consistently recovered, which unfortunately is not the case.
      // This fact is well-known (see PhD thesis S. Hueeber) and now
      // taken care of by a special treatment as can be seen below
      //
      //****************************************************************
      if (frcoeff == 0.0)
      {
        //**************************************************************
        // calculation of matrix entries of linearized slip condition
        //**************************************************************

        // 1) Entries from differentiation with respect to LM
        /******************************************************************/

        // loop over the dimension
        for (int dim = 0; dim < cnode->num_dof(); ++dim)
        {
          int col = cnode->dofs()[dim];
          double valtxi = txi[dim];

          double valteta = 0.0;
          if (n_dim() == 3) valteta = teta[dim];

          // do not assemble zeros into matrix
          if (abs(valtxi) > 1.0e-12) linslipLMglobal.assemble(valtxi, row[0], col);
          if (n_dim() == 3)
            if (abs(valteta) > 1.0e-12) linslipLMglobal.assemble(valteta, row[1], col);
        }

        // 2) Entries on right hand side
        /******************************************************************/

        Core::LinAlg::SerialDenseVector rhsnode(n_dim() - 1);
        std::vector<int> lm(n_dim() - 1);
        std::vector<int> lmowner(n_dim() - 1);

        lm[0] = cnode->dofs()[1];
        lmowner[0] = cnode->owner();

        rhsnode[0] = -ztxi;  // already negative rhs!!!

        if (n_dim() == 3)
        {
          rhsnode[1] = -zteta;  // already negative rhs!!!

          lm[1] = cnode->dofs()[2];
          lmowner[1] = cnode->owner();
        }

        Core::LinAlg::assemble(linslipRHSglobal, rhsnode, lm, lmowner);

        // 3) Entries from differentiation with respect to displacements
        /******************************************************************/

        // loop over dimensions
        for (int j = 0; j < n_dim(); ++j)
        {
          // loop over all entries of the current derivative map (txi)
          for (_colcurr = dtximap[j].begin(); _colcurr != dtximap[j].end(); ++_colcurr)
          {
            int col = _colcurr->first;
            double val = (_colcurr->second) * z[j];

            // do not assemble zeros into matrix
            if (abs(val) > 1.0e-12) linslipDISglobal.assemble(val, row[0], col);
          }

          if (n_dim() == 3)
          {
            // loop over all entries of the current derivative map (teta)
            for (_colcurr = dtetamap[j].begin(); _colcurr != dtetamap[j].end(); ++_colcurr)
            {
              int col = _colcurr->first;
              double val = (_colcurr->second) * z[j];

              // do not assemble zeros into s matrix
              if (abs(val) > 1.0e-12) linslipDISglobal.assemble(val, row[1], col);
            }
          }
        }
      }

      //****************************************************************
      // STANDARD TREATMENT OF CASE FRCOEFF!=0 (FRICTIONAL)
      //****************************************************************
      else
      {
        //**************************************************************
        // calculation of matrix entries of linearized slip condition
        //**************************************************************

        // 1) Entries from differentiation with respect to LM
        /******************************************************************/

        // loop over the dimension
        for (int dim = 0; dim < cnode->num_dof(); ++dim)
        {
          double valtxi = 0.0;
          int col = cnode->dofs()[dim];

          double valtxi0 = euclidean * txi[dim];
          double valtxi1 = ((ztxi + ct * jumptxi) / euclidean * ztxi) * txi[dim];
          double valtxi3 = (zteta + ct * jumpteta) / euclidean * ztxi * teta[dim];

#ifdef CONSISTENTSLIP
          valtxi0 = valtxi0 / (znor - cn * wgap);
          valtxi1 = valtxi1 / (znor - cn * wgap);
          valtxi3 = valtxi3 / (znor - cn * wgap);

          // Additional term
          valtxi0 -= euclidean * ztxi / pow(znor - cn * wgap, 2.0) * n[dim];

          double valtxi2 = -frcoeff * txi[dim];
#else
          double valtxi2 =
              -frcoeff * (znor - cn * wgap) * txi[dim] - frcoeff * (ztxi + ct * jumptxi) * n[dim];
#endif
          valtxi = valtxi0 + valtxi1 + valtxi2 + valtxi3;

          double valteta = 0.0;
          if (n_dim() == 3)
          {
            double valteta0 = euclidean * teta[dim];
            double valteta1 = ((ztxi + ct * jumptxi) / euclidean * zteta) * txi[dim];
            double valteta3 = (zteta + ct * jumpteta) / euclidean * zteta * teta[dim];
#ifdef CONSISTENTSLIP
            valteta0 = valteta0 / (znor - cn * wgap);
            valteta1 = valteta1 / (znor - cn * wgap);
            valteta3 = valteta3 / (znor - cn * wgap);

            // Additional term
            valteta0 -= euclidean * zteta / pow(znor - cn * wgap, 2.0) * n[dim];

            double valteta2 = -frcoeff * teta[dim];
#else
            double valteta2 = -frcoeff * (znor - cn * wgap) * teta[dim] -
                              frcoeff * (zteta + ct * jumpteta) * n[dim];
#endif
            valteta = valteta0 + valteta1 + valteta2 + valteta3;
          }

          // do not assemble zeros into matrix
          if (abs(valtxi) > 1.0e-12) linslipLMglobal.assemble(valtxi, row[0], col);
          if (n_dim() == 3)
            if (abs(valteta) > 1.0e-12) linslipLMglobal.assemble(valteta, row[1], col);
        }

        // 2) Entries on right hand side
        /******************************************************************/
        Core::LinAlg::SerialDenseVector rhsnode(n_dim() - 1);
        std::vector<int> lm(n_dim() - 1);
        std::vector<int> lmowner(n_dim() - 1);

#ifdef CONSISTENTSLIP
        double valuetxi1 = -(euclidean)*ztxi / (znor - cn * wgap) + frcoeff * (ztxi + ct * jumptxi);
#else
        double valuetxi1 =
            -(euclidean)*ztxi + (frcoeff * (znor - cn * wgap)) * (ztxi + ct * jumptxi);
#endif
        rhsnode(0) = valuetxi1;
        lm[0] = cnode->dofs()[1];
        lmowner[0] = cnode->owner();

        if (n_dim() == 3)
        {
#ifdef CONSISTENTSLIP
          double valueteta1 =
              -(euclidean)*zteta / (znor - cn * wgap) + frcoeff * (zteta + ct * jumpteta);
#else
          double valueteta1 =
              -(euclidean)*zteta + (frcoeff * (znor - cn * wgap)) * (zteta + ct * jumpteta);
#endif

          rhsnode(1) = valueteta1;

          lm[1] = cnode->dofs()[2];
          lmowner[1] = cnode->owner();
        }

        Core::LinAlg::assemble(linslipRHSglobal, rhsnode, lm, lmowner);

        // 3) Entries from differentiation with respect to displacements
        /******************************************************************/
        std::map<int, double> derivjump1, derivjump2;  // for gp slip
        std::vector<std::map<int, double>> derivjump;  // for dm slip

        /*** 01  ********* -Deriv(euclidean).ct.tangent.deriv(u)*ztan ***/
        if (interface_params().get<bool>("GP_SLIP_INCR"))
        {
          derivjump1 = cnode->fri_data().get_deriv_var_jump()[0];
          derivjump2 = cnode->fri_data().get_deriv_var_jump()[1];

          for (colcurr = derivjump1.begin(); colcurr != derivjump1.end(); ++colcurr)
          {
            int col = colcurr->first;
            double valtxi1 = (ztxi + ct * jumptxi) / euclidean * ct * colcurr->second * ztxi;
            double valteta1 = (ztxi + ct * jumptxi) / euclidean * ct * colcurr->second * zteta;

            if (abs(valtxi1) > 1.0e-12) linslipDISglobal.assemble(valtxi1, row[0], col);
            if (abs(valteta1) > 1.0e-12) linslipDISglobal.assemble(valteta1, row[1], col);
          }

          if (n_dim() == 3)
          {
            for (colcurr = derivjump2.begin(); colcurr != derivjump2.end(); ++colcurr)
            {
              int col = colcurr->first;
              double valtxi2 = (zteta + ct * jumpteta) / euclidean * ct * colcurr->second * ztxi;
              double valteta2 = (zteta + ct * jumpteta) / euclidean * ct * colcurr->second * zteta;

              if (abs(valtxi2) > 1.0e-12) linslipDISglobal.assemble(valtxi2, row[0], col);
              if (abs(valteta2) > 1.0e-12) linslipDISglobal.assemble(valteta2, row[1], col);
            }
          }
        }
        else  // std slip
        {
          // get linearization of jump vector
          derivjump = cnode->fri_data().get_deriv_jump();

          // loop over dimensions
          for (int dim = 0; dim < cnode->num_dof(); ++dim)
          {
            // loop over all entries of the current derivative map (jump)
            for (colcurr = derivjump[dim].begin(); colcurr != derivjump[dim].end(); ++colcurr)
            {
              int col = colcurr->first;

              double valtxi1 =
                  (ztxi + ct * jumptxi) / euclidean * ct * txi[dim] * colcurr->second * ztxi;
              double valteta1 =
                  (ztxi + ct * jumptxi) / euclidean * ct * txi[dim] * colcurr->second * zteta;
              double valtxi2 =
                  (zteta + ct * jumpteta) / euclidean * ct * teta[dim] * colcurr->second * ztxi;
              double valteta2 =
                  (zteta + ct * jumpteta) / euclidean * ct * teta[dim] * colcurr->second * zteta;

#ifdef CONSISTENTSLIP
              valtxi1 = valtxi1 / (znor - cn * wgap);
              valteta1 = valteta1 / (znor - cn * wgap);
              valtxi2 = valtxi2 / (znor - cn * wgap);
              valteta2 = valteta2 / (znor - cn * wgap);
#endif

              // do not assemble zeros into matrix
              if (abs(valtxi1) > 1.0e-12) linslipDISglobal.assemble(valtxi1, row[0], col);
              if (abs(valteta1) > 1.0e-12) linslipDISglobal.assemble(valteta1, row[1], col);
              if (abs(valtxi2) > 1.0e-12) linslipDISglobal.assemble(valtxi2, row[0], col);
              if (abs(valteta2) > 1.0e-12) linslipDISglobal.assemble(valteta2, row[1], col);
            }

#ifdef CONSISTENTSLIP
            /*** Additional Terms ***/
            // normal derivative
            for (colcurr = dnmap[dim].begin(); colcurr != dnmap[dim].end(); ++colcurr)
            {
              int col = colcurr->first;
              double valtxi =
                  -euclidean * ztxi * z[dim] / pow(znor - cn * wgap, 2.0) * (colcurr->second);
              double valteta =
                  -euclidean * zteta * z[dim] / pow(znor - cn * wgap, 2.0) * (colcurr->second);

              // do not assemble zeros into s matrix
              if (abs(valtxi) > 1.0e-12) linslipDISglobal.Assemble(valtxi, row[0], col);
              if (abs(valteta) > 1.0e-12) linslipDISglobal.Assemble(valteta, row[1], col);
            }
#endif
          }
        }


#ifdef CONSISTENTSLIP
        /*** Additional Terms ***/
        // wgap derivative
        std::map<int, double>& dgmap = cnode->Data().GetDerivG();

        for (colcurr = dgmap.begin(); colcurr != dgmap.end(); ++colcurr)
        {
          int col = colcurr->first;
          double valtxi = +euclidean * ztxi / pow(znor - cn * wgap, 2.0) * cn * (colcurr->second);
          double valteta = +euclidean * zteta / pow(znor - cn * wgap, 2.0) * cn * (colcurr->second);

          // do not assemble zeros into matrix
          if (abs(valtxi) > 1.0e-12) linslipDISglobal.Assemble(valtxi, row[0], col);
          if (abs(valteta) > 1.0e-12) linslipDISglobal.Assemble(valteta, row[1], col);
        }
#endif


        /*** 02 ***************** frcoeff*znor*ct*tangent.deriv(jump) ***/
        if (interface_params().get<bool>("GP_SLIP_INCR"))
        {
          for (colcurr = derivjump1.begin(); colcurr != derivjump1.end(); ++colcurr)
          {
            int col = colcurr->first;
            double valtxi = -frcoeff * (znor - cn * wgap) * ct * colcurr->second;

            if (abs(valtxi) > 1.0e-12) linslipDISglobal.assemble(valtxi, row[0], col);
          }

          if (n_dim() == 3)
          {
            for (colcurr = derivjump2.begin(); colcurr != derivjump2.end(); ++colcurr)
            {
              int col = colcurr->first;
              double valteta = -frcoeff * (znor - cn * wgap) * ct * colcurr->second;

              if (abs(valteta) > 1.0e-12) linslipDISglobal.assemble(valteta, row[1], col);
            }
          }
        }
        else  // std. slip
        {
          // loop over dimensions
          for (int dim = 0; dim < cnode->num_dof(); ++dim)
          {
            // loop over all entries of the current derivative map (jump)
            for (colcurr = derivjump[dim].begin(); colcurr != derivjump[dim].end(); ++colcurr)
            {
              int col = colcurr->first;

              // std::cout << "val " << colcurr->second << std::endl;
#ifdef CONSISTENTSLIP
              double valtxi = -frcoeff * ct * txi[dim] * colcurr->second;
              double valteta = -frcoeff * ct * teta[dim] * colcurr->second;
#else
              double valtxi =
                  (-1) * (frcoeff * (znor - cn * wgap)) * ct * txi[dim] * colcurr->second;
              double valteta =
                  (-1) * (frcoeff * (znor - cn * wgap)) * ct * teta[dim] * colcurr->second;
#endif
              // do not assemble zeros into matrix
              if (abs(valtxi) > 1.0e-12) linslipDISglobal.assemble(valtxi, row[0], col);

              if (n_dim() == 3)
              {
                if (abs(valteta) > 1.0e-12) linslipDISglobal.assemble(valteta, row[1], col);
              }
            }
          }
        }

        /*** 1 ********************************* euclidean.deriv(T).z ***/
        // loop over dimensions
        for (int j = 0; j < n_dim(); ++j)
        {
          // loop over all entries of the current derivative map (txi)
          for (_colcurr = dtximap[j].begin(); _colcurr != dtximap[j].end(); ++_colcurr)
          {
            int col = _colcurr->first;
            double val = euclidean * (_colcurr->second) * z[j];

#ifdef CONSISTENTSLIP
            val = val / (znor - cn * wgap);
#endif

            // do not assemble zeros into s matrix
            if (abs(val) > 1.0e-12) linslipDISglobal.assemble(val, row[0], col);
          }

          if (n_dim() == 3)
          {
            // loop over all entries of the current derivative map (teta)
            for (_colcurr = dtetamap[j].begin(); _colcurr != dtetamap[j].end(); ++_colcurr)
            {
              int col = _colcurr->first;
              double val = euclidean * (_colcurr->second) * z[j];

#ifdef CONSISTENTSLIP
              val = val / (znor - cn * wgap);
#endif

              // do not assemble zeros into s matrix
              if (abs(val) > 1.0e-12) linslipDISglobal.assemble(val, row[1], col);
            }
          }
        }

        /*** 2 ********************* deriv(euclidean).deriv(T).z.ztan ***/
        // loop over dimensions
        for (int j = 0; j < n_dim(); ++j)
        {
          // loop over all entries of the current derivative map (txi)
          for (_colcurr = dtximap[j].begin(); _colcurr != dtximap[j].end(); ++_colcurr)
          {
            int col = _colcurr->first;
            double valtxi = (ztxi + ct * jumptxi) / euclidean * (_colcurr->second) * z[j] * ztxi;
            double valteta = (ztxi + ct * jumptxi) / euclidean * (_colcurr->second) * z[j] * zteta;

#ifdef CONSISTENTSLIP
            valtxi = valtxi / (znor - cn * wgap);
            valteta = valteta / (znor - cn * wgap);
#endif

            // do not assemble zeros into matrix
            if (abs(valtxi) > 1.0e-12) linslipDISglobal.assemble(valtxi, row[0], col);
            if (n_dim() == 3)
              if (abs(valteta) > 1.0e-12) linslipDISglobal.assemble(valteta, row[1], col);
          }

          if (n_dim() == 3)
          {
            // 3D loop over all entries of the current derivative map (teta)
            for (_colcurr = dtetamap[j].begin(); _colcurr != dtetamap[j].end(); ++_colcurr)
            {
              int col = _colcurr->first;
              double valtxi =
                  (zteta + ct * jumpteta) / euclidean * (_colcurr->second) * z[j] * ztxi;
              double valteta =
                  (zteta + ct * jumpteta) / euclidean * (_colcurr->second) * z[j] * zteta;

#ifdef CONSISTENTSLIP
              valtxi = valtxi / (znor - cn * wgap);
              valteta = valteta / (znor - cn * wgap);
#endif

              // do not assemble zeros into matrix
              if (abs(valtxi) > 1.0e-12) linslipDISglobal.assemble(valtxi, row[0], col);
              if (abs(valteta) > 1.0e-12) linslipDISglobal.assemble(valteta, row[1], col);
            }
          }
        }

        /*** 3 ****************** deriv(euclidean).deriv(T).jump.ztan ***/
        if (interface_params().get<bool>("GP_SLIP_INCR"))
        {
          //!!!!!!!!!!!!!!! DO NOTHING !!!!!!!
        }
        else
        {
          // loop over dimensions
          for (int j = 0; j < n_dim(); ++j)
          {
            // loop over all entries of the current derivative map (txi)
            for (_colcurr = dtximap[j].begin(); _colcurr != dtximap[j].end(); ++_colcurr)
            {
              int col = _colcurr->first;
              double valtxi =
                  (ztxi + ct * jumptxi) / euclidean * ct * (_colcurr->second) * jump[j] * ztxi;
              double valteta =
                  (ztxi + ct * jumptxi) / euclidean * ct * (_colcurr->second) * jump[j] * zteta;

#ifdef CONSISTENTSLIP
              valtxi = valtxi / (znor - cn * wgap);
              valteta = valteta / (znor - cn * wgap);
#endif

              // do not assemble zeros into s matrix
              if (abs(valtxi) > 1.0e-12) linslipDISglobal.assemble(valtxi, row[0], col);
              if (abs(valteta) > 1.0e-12) linslipDISglobal.assemble(valteta, row[1], col);
            }

            if (n_dim() == 3)
            {
              // loop over all entries of the current derivative map (teta)
              for (_colcurr = dtetamap[j].begin(); _colcurr != dtetamap[j].end(); ++_colcurr)
              {
                int col = _colcurr->first;
                double valtxi =
                    (zteta + ct * jumpteta) / euclidean * ct * (_colcurr->second) * jump[j] * ztxi;
                double valteta =
                    (zteta + ct * jumpteta) / euclidean * ct * (_colcurr->second) * jump[j] * zteta;

#ifdef CONSISTENTSLIP
                valtxi = valtxi / (znor - cn * wgap);
                valteta = valteta / (znor - cn * wgap);
#endif

                // do not assemble zeros into matrix
                if (abs(valtxi) > 1.0e-12) linslipDISglobal.assemble(valtxi, row[0], col);
                if (abs(valteta) > 1.0e-12) linslipDISglobal.assemble(valteta, row[1], col);
              }
            }
          }
        }

        /*** 4 ************************** (frcoeff*znor).deriv(T).z ***/
        // loop over all dimensions
        for (int j = 0; j < n_dim(); ++j)
        {
          // loop over all entries of the current derivative map (txi)
          for (_colcurr = dtximap[j].begin(); _colcurr != dtximap[j].end(); ++_colcurr)
          {
            int col = _colcurr->first;
#ifdef CONSISTENTSLIP
            double val = -frcoeff * (_colcurr->second) * z[j];
#else
            double val = (-1) * (frcoeff * (znor - cn * wgap)) * (_colcurr->second) * z[j];
#endif
            // do not assemble zeros into matrix
            if (abs(val) > 1.0e-12) linslipDISglobal.assemble(val, row[0], col);
          }

          if (n_dim() == 3)
          {
            // loop over all entries of the current derivative map (teta)
            for (_colcurr = dtetamap[j].begin(); _colcurr != dtetamap[j].end(); ++_colcurr)
            {
              int col = _colcurr->first;
#ifdef CONSISTENTSLIP
              double val = -frcoeff * (_colcurr->second) * z[j];
#else
              double val = (-1.0) * (frcoeff * (znor - cn * wgap)) * (_colcurr->second) * z[j];
#endif
              // do not assemble zeros into matrix
              if (abs(val) > 1.0e-12) linslipDISglobal.assemble(val, row[1], col);
            }
          }
        }

        /*** 5 *********************** (frcoeff*znor).deriv(T).jump ***/
        if (interface_params().get<bool>("GP_SLIP_INCR"))
        {
          //!!!!!!!!!!!!!!! DO NOTHING !!!!!!!!!!!!!!!!!!!!!!
        }
        else
        {
          // loop over all dimensions
          for (int j = 0; j < n_dim(); ++j)
          {
            // loop over all entries of the current derivative map (txi)
            for (_colcurr = dtximap[j].begin(); _colcurr != dtximap[j].end(); ++_colcurr)
            {
              int col = _colcurr->first;
#ifdef CONSISTENTSLIP
              double val = -frcoeff * ct * (_colcurr->second) * jump[j];
#else
              double val =
                  (-1) * (frcoeff * (znor - cn * wgap)) * ct * (_colcurr->second) * jump[j];
#endif
              // do not assemble zeros into matrix
              if (abs(val) > 1.0e-12) linslipDISglobal.assemble(val, row[0], col);
            }

            if (n_dim() == 3)
            {
              // loop over all entries of the current derivative map (teta)
              for (_colcurr = dtetamap[j].begin(); _colcurr != dtetamap[j].end(); ++_colcurr)
              {
                int col = _colcurr->first;
#ifdef CONSISTENTSLIP
                double val = -frcoeff * ct * (_colcurr->second) * jump[j];
#else
                double val =
                    (-1) * (frcoeff * (znor - cn * wgap)) * ct * (_colcurr->second) * jump[j];
#endif
                // do not assemble zeros into s matrix
                if (abs(val) > 1.0e-12) linslipDISglobal.assemble(val, row[1], col);
              }
            }
          }
        }

#ifndef CONSISTENTSLIP
        /*** 6 ******************* -frcoeff.Deriv(n).z(ztan+ct*utan) ***/
        // loop over all dimensions
        for (int j = 0; j < n_dim(); ++j)
        {
          // loop over all entries of the current derivative map
          for (_colcurr = dnmap[j].begin(); _colcurr != dnmap[j].end(); ++_colcurr)
          {
            int col = _colcurr->first;
            double valtxi = (-1.0) * (ztxi + ct * jumptxi) * frcoeff * (_colcurr->second) * z[j];
            double valteta = (-1.0) * (zteta + ct * jumpteta) * frcoeff * (_colcurr->second) * z[j];

            // do not assemble zeros into s matrix
            if (abs(valtxi) > 1.0e-12) linslipDISglobal.assemble(valtxi, row[0], col);
            if (abs(valteta) > 1.0e-12) linslipDISglobal.assemble(valteta, row[1], col);
          }
        }

        /*** 7 ****************** frcoeff*cn*deriv (g).(ztan+ct*utan) ***/
        // prepare assembly
        std::map<int, double>& dgmap = cnode->data().get_deriv_g();

        // loop over all entries of the current derivative map
        for (colcurr = dgmap.begin(); colcurr != dgmap.end(); ++colcurr)
        {
          int col = colcurr->first;
          double valtxi = frcoeff * cn * (colcurr->second) * (ztxi + ct * jumptxi);
          double valteta = frcoeff * cn * (colcurr->second) * (zteta + ct * jumpteta);

          // do not assemble zeros into matrix
          if (abs(valtxi) > 1.0e-12) linslipDISglobal.assemble(valtxi, row[0], col);
          if (abs(valteta) > 1.0e-12) linslipDISglobal.assemble(valteta, row[1], col);
        }
#endif

        /*************************************************************************
         * Wear implicit linearization w.r.t. displ.                                          *
         *************************************************************************/
        if (wearimpl_ and !wearpv_)
        {
          std::map<int, double>& dwmap = cnode->data().get_deriv_w();
#ifdef CONSISTENTSLIP
          // loop over all entries of the current derivative map
          for (colcurr = dwmap.begin(); colcurr != dwmap.end(); ++colcurr)
          {
            int col = colcurr->first;
            double valtxi =
                euclidean * ztxi * (pow(znor - cn * wgap, -2.0) * cn * (colcurr->second));
            double valteta =
                euclidean * zteta * (pow(znor - cn * wgap, -2.0) * cn * (colcurr->second));

            // do not assemble zeros into matrix
            if (abs(valtxi) > 1.0e-12) linslipDISglobal.Assemble(valtxi, row[0], col);
            if (abs(valteta) > 1.0e-12) linslipDISglobal.Assemble(valteta, row[1], col);
          }
#else
          // loop over all entries of the current derivative map
          for (colcurr = dwmap.begin(); colcurr != dwmap.end(); ++colcurr)
          {
            int col = colcurr->first;
            double valtxi = frcoeff * cn * (colcurr->second) * (ztxi + ct * jumptxi);
            double valteta = frcoeff * cn * (colcurr->second) * (zteta + ct * jumpteta);

            // do not assemble zeros into matrix
            if (abs(valtxi) > 1.0e-12) linslipDISglobal.assemble(valtxi, row[0], col);
            if (abs(valteta) > 1.0e-12) linslipDISglobal.assemble(valteta, row[1], col);
          }
#endif
        }  // end wearimplicit
      }  // if (frcoeff==0.0)
    }  // loop over all slip nodes of the interface
  }  // Coulomb friction

  //**********************************************************************
  //**********************************************************************
  //**********************************************************************
  // Tresca Friction
  //**********************************************************************
  //**********************************************************************
  //**********************************************************************
  if (ftype == CONTACT::FrictionType::tresca)
  {
    FOUR_C_THROW("Tresca friction not implemented for wear !!!");
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Assemble matrix W_lm containing wear w~ derivatives      farah 07/13|
 |  w.r.t. lm       -- impl wear                                        |
 *----------------------------------------------------------------------*/
void Wear::WearInterface::assemble_lin_w_lm(Core::LinAlg::SparseMatrix& sglobal)
{
  /************************************************
   *  This function is only for Implicit Wear !!! *
   ************************************************/
  if (!wearimpl_) FOUR_C_THROW("This matrix deriv. is only required for implicit wear algorithm!");

  // nothing to do if no active nodes
  if (activenodes_ == nullptr) return;

  // loop over all active slave nodes of the interface
  for (int i = 0; i < activenodes_->num_my_elements();
      ++i)  //(int i=0;i<activenodes_->NumMyElements();++i)
  {
    int gid = activenodes_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(node);

    if (cnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
      FOUR_C_THROW("AssembleWLm: Node ownership inconsistency!");

    // prepare assembly
    std::map<int, double>& dwmap = cnode->data().get_deriv_wlm();
    std::map<int, double>::iterator colcurr;
    int row = activen_->gid(i);
    // row number of entries

    for (colcurr = dwmap.begin(); colcurr != dwmap.end(); ++colcurr)
    {
      int col = colcurr->first;
      double val = colcurr->second;

      if (abs(val) > 1.0e-12) sglobal.assemble(val, row, col);
    }

  }  // for (int i=0;i<activenodes_->NumMyElements();++i)

  return;
}

/*----------------------------------------------------------------------*
 |  Assemble matrix W_lmsl containing wear w~ derivatives    farah 07/13|
 |  w.r.t. lm  --> (ONLY !!!) for consistent stick                      |
 *----------------------------------------------------------------------*/
void Wear::WearInterface::assemble_lin_w_lm_st(Core::LinAlg::SparseMatrix& sglobal)
{
  /************************************************
   *  This function is only for Implicit Wear !!! *
   ************************************************/
  if (!wearimpl_) FOUR_C_THROW("This matrix deriv. is only required for implicit wear algorithm!");

  // create map of stick nodes
  std::shared_ptr<Core::LinAlg::Map> sticknodes =
      Core::LinAlg::split_map(*activenodes_, *slipnodes_);
  std::shared_ptr<Core::LinAlg::Map> stickt = Core::LinAlg::split_map(*activet_, *slipt_);

  // nothing to do if no stick nodes
  if (sticknodes->num_my_elements() == 0) return;

  // get input params
  double ct = interface_params().get<double>("SEMI_SMOOTH_CT");
  double cn = interface_params().get<double>("SEMI_SMOOTH_CN");
  double frcoeff = interface_params().get<double>("FRCOEFF");


  // loop over all stick slave nodes of the interface
  for (int i = 0; i < sticknodes->num_my_elements(); ++i)
  {
    int gid = sticknodes->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(node);

    if (cnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
      FOUR_C_THROW("Node ownership inconsistency!");

    cn = cn_ref().local_values_as_span()[cn_ref().get_map().lid(cnode->id())];
    ct = ct_ref().local_values_as_span()[ct_ref().get_map().lid(cnode->id())];

    // prepare assembly, get information from node
    std::map<int, double>& dwmap = cnode->data().get_deriv_wlm();

    double* jump = cnode->fri_data().jump();
    double* txi = cnode->data().txi();
    double* teta = cnode->data().teta();

    // iterator for maps
    std::map<int, double>::iterator colcurr;

    // row number of entries
    std::vector<int> row(n_dim() - 1);
    if (n_dim() == 2)
    {
      row[0] = stickt->gid(i);
    }
    else if (n_dim() == 3)
    {
      row[0] = stickt->gid(2 * i);
      row[1] = stickt->gid(2 * i) + 1;
    }
    else
      FOUR_C_THROW("AssemblelinSlip: Dimension not correct");

    // evaluation of specific components of entries to assemble
    double jumptxi = 0;
    double jumpteta = 0;
    for (int i = 0; i < n_dim(); i++)
    {
      jumptxi += txi[i] * jump[i];
      jumpteta += teta[i] * jump[i];
    }

    // loop over all entries of the current derivative map
    for (colcurr = dwmap.begin(); colcurr != dwmap.end(); ++colcurr)
    {
      int col = colcurr->first;
      double valtxi = frcoeff * cn * ct * jumptxi * (colcurr->second);

      // do not assemble zeros into matrix
      if (abs(valtxi) > 1.0e-12) sglobal.assemble(valtxi, row[0], col);

      if (n_dim() == 3)
      {
        double valteta = frcoeff * cn * ct * jumpteta * (colcurr->second);
        if (abs(valteta) > 1.0e-12) sglobal.assemble(valteta, row[1], col);
      }
    }
  }
  return;
}
/*----------------------------------------------------------------------*
 |  Assemble matrix W_lmsl containing wear w~ derivatives    farah 07/13|
 |  w.r.t. lm  --> for slip                                             |
 *----------------------------------------------------------------------*/
void Wear::WearInterface::assemble_lin_w_lm_sl(Core::LinAlg::SparseMatrix& sglobal)
{
  /************************************************
   *  This function is only for Implicit Wear !!! *
   ************************************************/

  // nothing to do if no slip nodes
  if (slipnodes_->num_my_elements() == 0) return;

  if (!wearimpl_) FOUR_C_THROW("This matrix deriv. is only required for implicit wear algorithm!");

  // get input params
  double ct = interface_params().get<double>("SEMI_SMOOTH_CT");
  double cn = interface_params().get<double>("SEMI_SMOOTH_CN");
  double frcoeff = interface_params().get<double>("FRCOEFF");

  // loop over all active slave nodes of the interface
  for (int i = 0; i < slipnodes_->num_my_elements(); ++i)
  {
    int gid = slipnodes_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(node);

    if (cnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
      FOUR_C_THROW("AssembleLinSlip: Node ownership inconsistency!");

    cn = cn_ref().local_values_as_span()[cn_ref().get_map().lid(cnode->id())];
    ct = ct_ref().local_values_as_span()[ct_ref().get_map().lid(cnode->id())];

    // prepare assembly, get information from node
    std::map<int, double>& dwmap = cnode->data().get_deriv_wlm();

    double* jump = cnode->fri_data().jump();
    double* txi = cnode->data().txi();
    double* teta = cnode->data().teta();
    double* z = cnode->mo_data().lm();

    // iterator for maps
    std::map<int, double>::iterator colcurr;

    // row number of entries
    std::vector<int> row(n_dim() - 1);
    if (n_dim() == 2)
    {
      row[0] = slipt_->gid(i);
    }
    else if (n_dim() == 3)
    {
      row[0] = slipt_->gid(2 * i);
      row[1] = slipt_->gid(2 * i) + 1;
    }
    else
      FOUR_C_THROW("AssemblelinSlip: Dimension not correct");

    // evaluation of specific components of entries to assemble
    double ztxi = 0;
    double zteta = 0;
    double jumptxi = 0;
    double jumpteta = 0;
    // double euclidean = 0;
    for (int i = 0; i < n_dim(); i++)
    {
      ztxi += txi[i] * z[i];
      zteta += teta[i] * z[i];
      jumptxi += txi[i] * jump[i];
      jumpteta += teta[i] * jump[i];
    }

#ifdef CONSISTENTSLIP

    double euclidean = 0;
    // evaluate euclidean norm ||vec(zt)+ct*vec(jump)||
    std::vector<double> sum1(Dim() - 1, 0);
    sum1[0] = ztxi + ct * jumptxi;
    if (Dim() == 3) sum1[1] = zteta + ct * jumpteta;
    if (Dim() == 2) euclidean = abs(sum1[0]);
    if (Dim() == 3) euclidean = sqrt(sum1[0] * sum1[0] + sum1[1] * sum1[1]);

    // loop over all entries of the current derivative map
    for (colcurr = dwmap.begin(); colcurr != dwmap.end(); ++colcurr)
    {
      int col = colcurr->first;
      double valtxi = euclidean * ztxi * (pow(znor - cn * wgap, -2.0) * cn * (colcurr->second));

      // do not assemble zeros into matrix
      if (abs(valtxi) > 1.0e-12) sglobal.Assemble(valtxi, row[0], col);

      if (Dim() == 3)
      {
        double valteta = euclidean * zteta * (pow(znor - cn * wgap, -2.0) * cn * (colcurr->second));
        if (abs(valteta) > 1.0e-12) sglobal.Assemble(valteta, row[1], col);
      }
    }
#else
    // loop over all entries of the current derivative map
    for (colcurr = dwmap.begin(); colcurr != dwmap.end(); ++colcurr)
    {
      int col = colcurr->first;
      double valtxi = frcoeff * cn * (colcurr->second) * (ztxi + ct * jumptxi);

      // do not assemble zeros into matrix
      if (abs(valtxi) > 1.0e-12) sglobal.assemble(valtxi, row[0], col);

      if (n_dim() == 3)
      {
        double valteta = frcoeff * cn * (colcurr->second) * (zteta + ct * jumpteta);
        if (abs(valteta) > 1.0e-12) sglobal.assemble(valteta, row[1], col);
      }
    }
#endif
  }
  return;
}

/*----------------------------------------------------------------------*
 |  Assemble wear                                         gitterle 12/10|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::assemble_wear(Core::LinAlg::Vector<double>& wglobal)
{
  // loop over proc's slave nodes of the interface for assembly
  // use standard row map to assemble each node only once
  for (int i = 0; i < snoderowmap_->num_my_elements(); ++i)
  {
    int gid = snoderowmap_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* frinode = dynamic_cast<CONTACT::FriNode*>(node);

    if (frinode->owner() != Core::Communication::my_mpi_rank(get_comm()))
      FOUR_C_THROW("AssembleWear: Node ownership inconsistency!");

    /**************************************************** w-vector ******/
    double wear = frinode->wear_data().weighted_wear();

    Core::LinAlg::SerialDenseVector wnode(1);
    std::vector<int> lm(1);
    std::vector<int> lmowner(1);

    wnode(0) = wear;
    lm[0] = frinode->id();
    lmowner[0] = frinode->owner();

    Core::LinAlg::assemble(wglobal, wnode, lm, lmowner);
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Assemble Mortar matrice for both sided wear              farah 06/13|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::assemble_d2(Core::LinAlg::SparseMatrix& dglobal)
{
  /**************************************************************
   *  This function is only for wear mapping by enforcing weak  *
   *  dirichlet bound. cond on master side !!! *                *
   **************************************************************/

  //*******************************************************
  // assemble second D matrix for both-sided wear :
  // Loop over allreduced map, so that any proc join
  // the loop. Do non-local assembly by FEAssemble to
  // allow owned and ghosted nodes assembly. In integrator
  // only the nodes associated to the owned slave element
  // are filled with entries. Therefore, the integrated
  // entries are unique on nodes!
  //*******************************************************
  const std::shared_ptr<Core::LinAlg::Map> masternodes =
      Core::LinAlg::allreduce_e_map(*(mnoderowmap_));

  for (int i = 0; i < masternodes->num_my_elements(); ++i)  // mnoderowmap_
  {
    int gid = masternodes->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(node);

    /**************************************************** D2-matrix ******/
    if ((cnode->wear_data().get_d2()).size() > 0)
    {
      std::vector<std::map<int, double>> dmap = cnode->wear_data().get_d2();
      int rowsize = cnode->num_dof();
      int colsize = (int)dmap[0].size();

      for (int j = 0; j < rowsize - 1; ++j)
        if ((int)dmap[j].size() != (int)dmap[j + 1].size())
          FOUR_C_THROW("AssembleDM: Column dim. of nodal D-map is inconsistent!");

      std::map<int, double>::iterator colcurr;

      for (int j = 0; j < rowsize; ++j)
      {
        int row = cnode->dofs()[j];
        int k = 0;

        for (colcurr = dmap[j].begin(); colcurr != dmap[j].end(); ++colcurr)
        {
          int col = colcurr->first;
          double val = colcurr->second;

          // do the assembly into global D matrix
          // check for diagonality
          if (row != col && abs(val) > 1.0e-12)
            FOUR_C_THROW("AssembleDM: D-Matrix is not diagonal!");

          // create an explicitly diagonal d matrix
          if (row == col) dglobal.fe_assemble(val, row, col);

          ++k;
        }

        if (k != colsize) FOUR_C_THROW("AssembleDM: k = {} but colsize = {}", k, colsize);
      }
    }
  }

  return;
}


/*----------------------------------------------------------------------*
 |  build active set (nodes / dofs) for Master               farah 11/13|
 *----------------------------------------------------------------------*/
bool Wear::WearInterface::build_active_set_master()
{
  //****************************************
  // for both-sided discr wear
  //****************************************

  // spread active and slip information to all procs
  std::vector<int> a;
  std::vector<int> sl;
  // loop over all slave nodes on the current interface
  for (int j = 0; j < slave_row_nodes()->num_my_elements(); ++j)
  {
    int gid = slave_row_nodes()->gid(j);
    Core::Nodes::Node* node = discret().g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* frinode = dynamic_cast<CONTACT::FriNode*>(node);

    if (frinode->active()) a.push_back(frinode->id());

    if (frinode->fri_data().slip()) sl.push_back(frinode->id());
  }
  Core::LinAlg::Map auxa(-1, (int)a.size(), a.data(), 0, get_comm());
  Core::LinAlg::Map auxsl(-1, (int)sl.size(), sl.data(), 0, get_comm());

  const std::shared_ptr<Core::LinAlg::Map> ara = Core::LinAlg::allreduce_e_map((auxa));
  const std::shared_ptr<Core::LinAlg::Map> arsl = Core::LinAlg::allreduce_e_map((auxsl));

  for (int j = 0; j < slave_col_nodes()->num_my_elements(); ++j)
  {
    int gid = slave_col_nodes()->gid(j);

    if (ara->lid(gid) == -1) continue;

    Core::Nodes::Node* node = discret().g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* frinode = dynamic_cast<CONTACT::FriNode*>(node);

    if (frinode->owner() != Core::Communication::my_mpi_rank(get_comm()))
    {
      frinode->active() = true;
    }
  }
  for (int j = 0; j < slave_col_nodes()->num_my_elements(); ++j)
  {
    int gid = slave_col_nodes()->gid(j);

    if (arsl->lid(gid) == -1) continue;

    Core::Nodes::Node* node = discret().g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* frinode = dynamic_cast<CONTACT::FriNode*>(node);

    if (frinode->owner() != Core::Communication::my_mpi_rank(get_comm()))
      frinode->fri_data().slip() = true;
  }

  // spread info for attached status...
  const std::shared_ptr<Core::LinAlg::Map> meleall =
      Core::LinAlg::allreduce_e_map(*(master_row_elements()));
  std::vector<int> eleatt;

  for (int j = 0; j < meleall->num_my_elements(); ++j)
  {
    int gid = meleall->gid(j);
    Core::Elements::Element* ele = discret().g_element(gid);
    if (!ele) FOUR_C_THROW("Cannot find node with gid %", gid);
    Mortar::Element* moele = dynamic_cast<Mortar::Element*>(ele);

    if (moele->is_attached() == true and
        moele->owner() == Core::Communication::my_mpi_rank(get_comm()))
      eleatt.push_back(moele->id());
  }

  Core::LinAlg::Map auxe(-1, (int)eleatt.size(), eleatt.data(), 0, get_comm());
  const std::shared_ptr<Core::LinAlg::Map> att = Core::LinAlg::allreduce_e_map((auxe));

  for (int j = 0; j < att->num_my_elements(); ++j)
  {
    int gid = att->gid(j);
    Core::Elements::Element* ele = discret().g_element(gid);
    if (!ele) FOUR_C_THROW("Cannot find node with gid %", gid);
    Mortar::Element* moele = dynamic_cast<Mortar::Element*>(ele);

    if (moele->is_attached() == false) moele->set_attached() = true;
  }

  // Detect maps
  std::vector<int> wa;
  std::vector<int> wsl;
  std::vector<int> wad;
  std::vector<int> wsln;

  // loop over all slave nodes on the current interface
  for (int j = 0; j < slave_row_nodes()->num_my_elements(); ++j)
  {
    int gid = slave_row_nodes()->gid(j);
    Core::Nodes::Node* node = discret().g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* frinode = dynamic_cast<CONTACT::FriNode*>(node);

    // get elements from node (SLAVE)
    for (int u = 0; u < (int)frinode->num_element(); ++u)
    {
      // all found MASTER elements:
      for (int k = 0;
          k < (int)dynamic_cast<Mortar::Element*>(frinode->adjacent_elements()[u].user_element())
                  ->mo_data()
                  .num_search_elements();
          ++k)
      {
        int gid2 = dynamic_cast<Mortar::Element*>(frinode->adjacent_elements()[u].user_element())
                       ->mo_data()
                       .search_elements()[k];
        Core::Elements::Element* ele2 = discret().g_element(gid2);
        if (!ele2) FOUR_C_THROW("Cannot find master element with gid %", gid2);
        Mortar::Element* celement = dynamic_cast<Mortar::Element*>(ele2);

        // nodes cor. to this master element
        if (celement->is_attached() == true)
        {
          for (int p = 0; p < celement->num_node(); ++p)
          {
            CONTACT::FriNode* mnode = dynamic_cast<CONTACT::FriNode*>(celement->nodes()[p]);

            if (mnode->is_detected() == false)
            {
              // active master nodes!
              if (frinode->active())
              {
                wa.push_back(celement->nodes()[p]->id());
                wad.push_back(dynamic_cast<Mortar::Node*>(celement->nodes()[p])->dofs()[0]);
              }

              // slip master nodes!
              if (frinode->fri_data().slip())
              {
                wsl.push_back(celement->nodes()[p]->id());
                wsln.push_back(dynamic_cast<Mortar::Node*>(celement->nodes()[p])->dofs()[0]);
              }

              // set detection status
              //              if(frinode->Active() || frinode->FriData().Slip())
              //                mnode->SetDetected()=true;
            }
          }
        }
      }
    }
  }  // node loop

  // reset nodes
  // loop over all master nodes on the current interface
  const std::shared_ptr<Core::LinAlg::Map> masternodes =
      Core::LinAlg::allreduce_e_map(*(mnoderowmap_));
  const std::shared_ptr<Core::LinAlg::Map> mastereles =
      Core::LinAlg::allreduce_e_map(*(melerowmap_));

  for (int j = 0; j < masternodes->num_my_elements(); ++j)
  {
    int gid = masternodes->gid(j);
    Core::Nodes::Node* node = discret().g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* mnode = dynamic_cast<CONTACT::FriNode*>(node);

    mnode->set_detected() = false;
  }
  for (int j = 0; j < mastereles->num_my_elements(); ++j)
  {
    int gid = mastereles->gid(j);
    Core::Elements::Element* ele = discret().g_element(gid);
    if (!ele) FOUR_C_THROW("Cannot find element with gid %", gid);
    Mortar::Element* mele = dynamic_cast<Mortar::Element*>(ele);

    mele->set_attached() = false;
  }

  Core::LinAlg::Map actmn(-1, (int)wa.size(), wa.data(), 0, get_comm());
  Core::LinAlg::Map slimn(-1, (int)wsl.size(), wsl.data(), 0, get_comm());
  Core::LinAlg::Map slimd(-1, (int)wsln.size(), wsln.data(), 0, get_comm());

  const std::shared_ptr<Core::LinAlg::Map> ARactmn =
      Core::LinAlg::allreduce_overlapping_e_map((actmn));
  const std::shared_ptr<Core::LinAlg::Map> ARslimn =
      Core::LinAlg::allreduce_overlapping_e_map((slimn));

  std::vector<int> ga;
  std::vector<int> gs;
  std::vector<int> gsd;

  for (int j = 0; j < ARactmn->num_my_elements(); ++j)
  {
    int gid = ARactmn->gid(j);
    Core::Nodes::Node* node = discret().g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* mnode = dynamic_cast<CONTACT::FriNode*>(node);

    if (mnode->owner() == Core::Communication::my_mpi_rank(get_comm()))
    {
      bool isin = false;
      for (int k = 0; k < (int)ga.size(); ++k)
      {
        if (ga[k] == mnode->id()) isin = true;
      }
      if (!isin) ga.push_back(mnode->id());
    }
  }


  for (int j = 0; j < ARslimn->num_my_elements(); ++j)
  {
    int gid = ARslimn->gid(j);
    Core::Nodes::Node* node = discret().g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* mnode = dynamic_cast<CONTACT::FriNode*>(node);

    if (mnode->owner() == Core::Communication::my_mpi_rank(get_comm()))
    {
      bool isin = false;
      for (int k = 0; k < (int)gs.size(); ++k)
      {
        if (gs[k] == mnode->id()) isin = true;
      }
      if (!isin)
      {
        gs.push_back(mnode->id());
        gsd.push_back(mnode->dofs()[0]);
      }
    }
  }

  activmasternodes_ =
      std::make_shared<Core::LinAlg::Map>(-1, (int)ga.size(), ga.data(), 0, get_comm());
  slipmasternodes_ =
      std::make_shared<Core::LinAlg::Map>(-1, (int)gs.size(), gs.data(), 0, get_comm());
  slipmn_ = std::make_shared<Core::LinAlg::Map>(-1, (int)gsd.size(), gsd.data(), 0, get_comm());

  for (int j = 0; j < slave_col_nodes()->num_my_elements(); ++j)
  {
    int gid = slave_col_nodes()->gid(j);

    if (ara->lid(gid) == -1) continue;

    Core::Nodes::Node* node = discret().g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* frinode = dynamic_cast<CONTACT::FriNode*>(node);

    if (frinode->owner() != Core::Communication::my_mpi_rank(get_comm())) frinode->active() = false;
  }
  for (int j = 0; j < slave_col_nodes()->num_my_elements(); ++j)
  {
    int gid = slave_col_nodes()->gid(j);

    if (arsl->lid(gid) == -1) continue;

    Core::Nodes::Node* node = discret().g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* frinode = dynamic_cast<CONTACT::FriNode*>(node);

    if (frinode->owner() != Core::Communication::my_mpi_rank(get_comm()))
      frinode->fri_data().slip() = false;
  }

  return true;
}


/*----------------------------------------------------------------------*
 |  build active set (nodes / dofs)                          farah 02/16|
 *----------------------------------------------------------------------*/
bool Wear::WearInterface::build_active_set(bool init)
{
  // call contact function
  CONTACT::Interface::build_active_set(init);

  // define local variables
  std::vector<int> mynodegids;
  std::vector<int> mydofgids;
  std::vector<int> myslipnodegids;
  std::vector<int> myslipdofgids;
  std::vector<int> mymnodegids;
  std::vector<int> mymdofgids;

  // *******************************************************
  // loop over all master nodes - both-sided wear specific
  if (wearboth_ and !wearpv_)
  {
    // The node and dof map of involved master nodes for both-sided wear should have an
    // analog distribution as the master node row map. Therefore, we loop over
    // allreduced map (all procs involved). If a node has the corresponding bool
    // "InvolvedM()" then this information will be communicated to all other procs.
    // Note, this node could be ghosted! At the end, the owning proc will take the
    // information to build the map.

    const std::shared_ptr<Core::LinAlg::Map> masternodes =
        Core::LinAlg::allreduce_e_map(*(mnoderowmap_));

    for (int k = 0; k < masternodes->num_my_elements(); ++k)  // mnoderowmap_
    {
      int gid = masternodes->gid(k);
      Core::Nodes::Node* node = idiscret_->g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(node);
      const int numdof = cnode->num_dof();

      int inv = 0;
      if (cnode->involved_m() == true) inv = 1;

      // TODO: not nice... alternative to sumall?
      int invglobal = 0;
      invglobal = Core::Communication::sum_all(inv, get_comm());
      Core::Communication::barrier(get_comm());

      if (cnode->owner() == Core::Communication::my_mpi_rank(get_comm()) && invglobal > 0)
      {
        mymnodegids.push_back(cnode->id());

        for (int j = 0; j < numdof; ++j)
        {
          mymdofgids.push_back(cnode->dofs()[j]);
        }
      }
      // reset it
      cnode->involved_m() = false;
    }

    // create map for all involved master nodes -- both-sided wear specific
    involvednodes_ = std::make_shared<Core::LinAlg::Map>(
        -1, (int)mymnodegids.size(), mymnodegids.data(), 0, get_comm());
    involveddofs_ = std::make_shared<Core::LinAlg::Map>(
        -1, (int)mymdofgids.size(), mymdofgids.data(), 0, get_comm());
  }

  return true;
}


/*----------------------------------------------------------------------*
 |  Initialize Data Container for nodes and elements         farah 02/16|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::initialize_data_container()
{
  // call contact class function which calls mortar function
  CONTACT::Interface::initialize_data_container();

  //***********************************************************
  // both-sided wear
  // here we need a datacontainer for the masternodes too
  // they have to know their involved nodes/dofs
  //***********************************************************
  if (wearboth_)
  {
    const std::shared_ptr<Core::LinAlg::Map> masternodes =
        Core::LinAlg::allreduce_e_map(*(master_row_nodes()));

    for (int i = 0; i < masternodes->num_my_elements(); ++i)  // MasterRowNodes()
    {
      int gid = masternodes->gid(i);
      Core::Nodes::Node* node = discret().g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid {}", gid);
      Mortar::Node* mnode = dynamic_cast<Mortar::Node*>(node);

      //********************************************************
      // NOTE: depending on which kind of node this really is,
      // i.e. mortar, contact or friction node, several derived
      // versions of the initialize_data_container() methods will
      // be called here, apart from the base class version.
      //********************************************************

      // initialize container if not yet initialized before
      mnode->initialize_data_container();
    }
  }

  return;
}


/*----------------------------------------------------------------------*
 |  Assemble inactive wear right hand side                   farah 09/13|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::assemble_inactive_wear_rhs(Core::LinAlg::Vector<double>& inactiverhs)
{
  /************************************************
   *  This function is only for discrete Wear !!! *
   ************************************************/

  // FIXME It's possible to improve the performance, if only recently active nodes of the inactive
  // node set, i.e. nodes, which were active in the last iteration, are considered. Since you know,
  // that the lagrange multipliers of former inactive nodes are still equal zero.

  std::shared_ptr<Core::LinAlg::Map> inactivenodes;

  if (sswear_)
    inactivenodes = Core::LinAlg::split_map(*snoderowmap_, *activenodes_);
  else
    inactivenodes = Core::LinAlg::split_map(*snoderowmap_, *slipnodes_);

  for (int i = 0; i < inactivenodes->num_my_elements(); ++i)
  {
    int gid = inactivenodes->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(node);

    if (cnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
      FOUR_C_THROW("AssembleInactiverhs: Node ownership inconsistency!");

    if (n_dim() == 2)
    {
      std::vector<int> w_gid(1);
      std::vector<int> w_owner(1);

      // calculate the tangential rhs
      Core::LinAlg::SerialDenseVector w_i(1);

      w_owner[0] = cnode->owner();
      w_i[0] =
          -cnode->wear_data().wold()[0] - cnode->wear_data().wcurr()[0];  // already negative rhs!!!
      w_gid[0] = cnode->dofs()[0];                                        // inactivedofs->GID(2*i);

      if (abs(w_i[0]) > 1e-15) Core::LinAlg::assemble(inactiverhs, w_i, w_gid, w_owner);
    }
    else if (n_dim() == 3)
    {
      std::vector<int> w_gid(1);
      std::vector<int> w_owner(1);

      // calculate the tangential rhs
      Core::LinAlg::SerialDenseVector w_i(1);

      w_owner[0] = cnode->owner();
      w_i[0] =
          -cnode->wear_data().wold()[0] - cnode->wear_data().wcurr()[0];  // already negative rhs!!!
      w_gid[0] = cnode->dofs()[0];                                        // inactivedofs->GID(3*i);

      if (abs(w_i[0]) > 1e-15) Core::LinAlg::assemble(inactiverhs, w_i, w_gid, w_owner);
    }
  }
}


/*----------------------------------------------------------------------*
 |  Assemble inactive wear right hand side                   farah 11/13|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::assemble_inactive_wear_rhs_master(
    Core::LinAlg::FEVector<double>& inactiverhs)
{
  /************************************************
   *  This function is only for discrete Wear !!! *
   ************************************************/

  std::shared_ptr<Core::LinAlg::Map> inactivenodes =
      Core::LinAlg::split_map(*mnoderowmap_, *slipmasternodes_);
  std::shared_ptr<Core::LinAlg::Map> inactivedofs = Core::LinAlg::split_map(*(mn_dofs()), *slipmn_);



  const std::shared_ptr<Core::LinAlg::Map> allredi = Core::LinAlg::allreduce_e_map(*(inactivedofs));

  std::shared_ptr<Core::LinAlg::Vector<double>> rhs = Core::LinAlg::create_vector(*allredi, true);

  for (int i = 0; i < inactivenodes->num_my_elements(); ++i)
  {
    int gid = inactivenodes->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(node);

    if (cnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
      FOUR_C_THROW("AssembleInactiverhs: Node ownership inconsistency!");

    if (n_dim() == 2)
    {
      std::vector<int> w_gid(1);
      std::vector<int> w_owner(1);

      // calculate the tangential rhs
      Core::LinAlg::SerialDenseVector w_i(1);

      w_owner[0] = Core::Communication::my_mpi_rank(get_comm());  // cnode->Owner();
      w_i[0] =
          -cnode->wear_data().wold()[0] - cnode->wear_data().wcurr()[0];  // already negative rhs!!!
      w_gid[0] = cnode->dofs()[0];                                        // inactivedofs->GID(2*i);

      if (abs(w_i[0]) > 1e-12) Core::LinAlg::assemble(*rhs, w_i, w_gid, w_owner);
    }
    else if (n_dim() == 3)
    {
      std::vector<int> w_gid(1);
      std::vector<int> w_owner(1);

      // calculate the tangential rhs
      Core::LinAlg::SerialDenseVector w_i(1);

      w_owner[0] = Core::Communication::my_mpi_rank(get_comm());  // cnode->Owner();
      w_i[0] =
          -cnode->wear_data().wold()[0] - cnode->wear_data().wcurr()[0];  // already negative rhs!!!
      w_gid[0] = cnode->dofs()[0];                                        // inactivedofs->GID(3*i);

      if (abs(w_i[0]) > 1e-12) Core::LinAlg::assemble(*rhs, w_i, w_gid, w_owner);
    }
  }

  Core::LinAlg::Export exp(*allredi, *inactivedofs);
  inactiverhs.export_to(*rhs, exp, Add);


  return;
}


/*----------------------------------------------------------------------*
 |  Assemble wear-cond. right hand side (discr)              farah 09/13|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::assemble_wear_cond_rhs(Core::LinAlg::Vector<double>& rhs)
{
  /************************************************
   *  This function is only for discrete Wear !!! *
   ************************************************/

  // nodes for loop
  std::shared_ptr<Core::LinAlg::Map> considerednodes;

  // nothing to do if no active nodes
  if (sswear_)
  {
    if (activenodes_ == nullptr) return;
    considerednodes = activenodes_;
  }
  else
  {
    if (slipnodes_ == nullptr) return;
    considerednodes = slipnodes_;
  }

  auto systype = Teuchos::getIntegralValue<CONTACT::SystemType>(interface_params(), "SYSTEM");

  double wcoeff = interface_params().get<double>("WEARCOEFF");

  using CI = std::map<int, double>::const_iterator;

  for (int i = 0; i < considerednodes->num_my_elements(); ++i)
  {
    int gid = considerednodes->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* fnode = dynamic_cast<CONTACT::FriNode*>(node);

    if (fnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
      FOUR_C_THROW("AssembleWearCondRhs: Node ownership inconsistency!");

    /**************************************************** E-matrix ******/
    if ((fnode->wear_data().get_e()).size() > 0)
    {
      std::map<int, double> emap = fnode->wear_data().get_e()[0];

      for (CI p = emap.begin(); p != emap.end(); ++p)
      {
        int gid3 = (int)((p->first) / n_dim());
        Core::Nodes::Node* snode = idiscret_->g_node(gid3);
        if (!snode) FOUR_C_THROW("Cannot find node with gid");
        CONTACT::FriNode* csnode = dynamic_cast<CONTACT::FriNode*>(snode);

        std::vector<int> w_gid(1);
        std::vector<int> w_owner(1);

        Core::LinAlg::SerialDenseVector w_i(1);

        w_owner[0] = fnode->owner();
        w_i[0] =
            (-(csnode->wear_data().wold()[0]) - (csnode->wear_data().wcurr()[0])) * (p->second);
        w_gid[0] = fnode->dofs()[0];

        if (abs(w_i[0]) > 1e-15) Core::LinAlg::assemble(rhs, w_i, w_gid, w_owner);
      }
    }

    /**************************************************** T-matrix ******/
    // for condensation of lm and wear we condense the system with absol. lm
    // --> therefore we do not need the lm^i term...
    if (((fnode->wear_data().get_t()).size() > 0) && systype != CONTACT::SystemType::condensed)
    {
      std::map<int, double> tmap = fnode->wear_data().get_t()[0];

      for (CI p = tmap.begin(); p != tmap.end(); ++p)
      {
        int gid3 = (int)((p->first) / n_dim());
        Core::Nodes::Node* snode = idiscret_->g_node(gid3);
        if (!snode) FOUR_C_THROW("Cannot find node with gid");
        CONTACT::FriNode* csnode = dynamic_cast<CONTACT::FriNode*>(snode);

        double lmn = 0.0;
        for (int u = 0; u < n_dim(); ++u)
          lmn += (csnode->mo_data().n()[u]) * (csnode->mo_data().lm()[u]);

        std::vector<int> w_gid(1);
        std::vector<int> w_owner(1);

        Core::LinAlg::SerialDenseVector w_i(1);

        w_owner[0] = fnode->owner();
        w_i[0] = wcoeff * lmn * (p->second);
        w_gid[0] = fnode->dofs()[0];

        if (abs(w_i[0]) > 1e-15) Core::LinAlg::assemble(rhs, w_i, w_gid, w_owner);
      }
    }
  }

  return;
}


/*----------------------------------------------------------------------*
 |  Assemble wear-cond. right hand side (discr)              farah 11/13|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::assemble_wear_cond_rhs_master(Core::LinAlg::FEVector<double>& RHS)
{
  /************************************************
   *  This function is only for discrete Wear !!! *
   ************************************************/

  // nothing to do if no active nodes
  if (slipmasternodes_ == nullptr) return;

  auto systype = Teuchos::getIntegralValue<CONTACT::SystemType>(interface_params(), "SYSTEM");

  double wcoeff = interface_params().get<double>("WEARCOEFF_MASTER");

  using CI = std::map<int, double>::const_iterator;

  const std::shared_ptr<Core::LinAlg::Map> slmasternodes =
      Core::LinAlg::allreduce_e_map(*(slipmasternodes_));
  const std::shared_ptr<Core::LinAlg::Map> slmastern = Core::LinAlg::allreduce_e_map(*(slipmn_));

  std::shared_ptr<Core::LinAlg::Vector<double>> rhs = Core::LinAlg::create_vector(*slmastern, true);

  for (int i = 0; i < slmasternodes->num_my_elements(); ++i)
  {
    int gid = slmasternodes->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::FriNode* fnode = dynamic_cast<CONTACT::FriNode*>(node);

    /**************************************************** E-matrix ******/
    if ((fnode->wear_data().get_e()).size() > 0)
    {
      std::map<int, double> emap = fnode->wear_data().get_e()[0];

      for (CI p = emap.begin(); p != emap.end(); ++p)
      {
        int gid3 = (int)((p->first) / n_dim());
        Core::Nodes::Node* snode = idiscret_->g_node(gid3);
        if (!snode) FOUR_C_THROW("Cannot find node with gid");
        CONTACT::FriNode* csnode = dynamic_cast<CONTACT::FriNode*>(snode);

        std::vector<int> w_gid(1);
        std::vector<int> w_owner(1);

        Core::LinAlg::SerialDenseVector w_i(1);

        w_owner[0] = Core::Communication::my_mpi_rank(get_comm());  // fnode->Owner();
        w_i[0] =
            (-(csnode->wear_data().wold()[0]) - (csnode->wear_data().wcurr()[0])) * (p->second);
        w_gid[0] = fnode->dofs()[0];

        if (abs(w_i[0]) > 1e-15) Core::LinAlg::assemble(*rhs, w_i, w_gid, w_owner);
      }
    }

    /**************************************************** T-matrix ******/
    // for condensation of lm and wear we condense the system with absol. lm
    // --> therefore we do not need the lm^i term...
    if (((fnode->wear_data().get_t()).size() > 0) && systype != CONTACT::SystemType::condensed)
    {
      std::map<int, double> tmap = fnode->wear_data().get_t()[0];

      for (CI p = tmap.begin(); p != tmap.end(); ++p)
      {
        int gid3 = (int)((p->first) / n_dim());
        Core::Nodes::Node* snode = idiscret_->g_node(gid3);
        if (!snode) FOUR_C_THROW("Cannot find node with gid");
        CONTACT::FriNode* csnode = dynamic_cast<CONTACT::FriNode*>(snode);

        double lmn = 0.0;
        for (int u = 0; u < n_dim(); ++u)
          lmn += (csnode->mo_data().n()[u]) * (csnode->mo_data().lm()[u]);

        std::vector<int> w_gid(1);
        std::vector<int> w_owner(1);

        Core::LinAlg::SerialDenseVector w_i(1);

        w_owner[0] = Core::Communication::my_mpi_rank(get_comm());  // fnode->Owner();
        w_i[0] = wcoeff * lmn * (p->second);
        w_gid[0] = fnode->dofs()[0];

        if (abs(w_i[0]) > 1e-15) Core::LinAlg::assemble(*rhs, w_i, w_gid, w_owner);
      }
    }
  }

  Core::LinAlg::Export exp(*slmastern, *slipmn_);
  RHS.export_to(*rhs, exp, Add);

  return;
}


/*----------------------------------------------------------------------*
 |  initialize / reset interface for wear                    farah 09/13|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::initialize()
{
  // loop over all nodes to reset stuff (fully overlapping column map)
  // (use fully overlapping column map)

  for (int i = 0; i < idiscret_->num_my_col_nodes(); ++i)
  {
    CONTACT::Node* node = dynamic_cast<CONTACT::Node*>(idiscret_->l_col_node(i));

    // reset feasible projection and segmentation status
    node->has_proj() = false;
    node->has_segment() = false;
  }

  //**************************************************
  // for both-sided wear
  //**************************************************
  if (wearboth_ and !wearpv_)
  {
    const std::shared_ptr<Core::LinAlg::Map> masternodes =
        Core::LinAlg::allreduce_e_map(*(master_row_nodes()));

    for (int i = 0; i < masternodes->num_my_elements();
        ++i)  // for (int i=0;i<MasterRowNodes()->NumMyElements();++i)
    {
      int gid = masternodes->gid(i);
      Core::Nodes::Node* node = discret().g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(node);

      if (cnode->is_slave() == false)
      {
        // reset nodal Mortar maps
        for (int j = 0; j < (int)((cnode->wear_data().get_d2()).size()); ++j)
          (cnode->wear_data().get_d2())[j].clear();

        (cnode->wear_data().get_d2()).resize(0);
      }
    }
  }

  // loop over all slave nodes to reset stuff (standard column map)
  // (include slave side boundary nodes / crosspoints)
  for (int i = 0; i < slave_col_nodes_bound()->num_my_elements(); ++i)
  {
    int gid = slave_col_nodes_bound()->gid(i);
    Core::Nodes::Node* node = discret().g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(node);

    // reset nodal Mortar maps
    cnode->mo_data().get_d().clear();
    cnode->mo_data().get_m().clear();
    cnode->mo_data().get_mmod().clear();

    // reset derivative maps of normal vector
    for (int j = 0; j < (int)((cnode->data().get_deriv_n()).size()); ++j)
      (cnode->data().get_deriv_n())[j].clear();
    (cnode->data().get_deriv_n()).resize(0, 0);

    // reset derivative maps of tangent vectors
    for (int j = 0; j < (int)((cnode->data().get_deriv_txi()).size()); ++j)
      (cnode->data().get_deriv_txi())[j].clear();
    (cnode->data().get_deriv_txi()).resize(0, 0);
    for (int j = 0; j < (int)((cnode->data().get_deriv_teta()).size()); ++j)
      (cnode->data().get_deriv_teta())[j].clear();
    (cnode->data().get_deriv_teta()).resize(0, 0);

    // reset derivative map of Mortar matrices
    (cnode->data().get_deriv_d()).clear();
    (cnode->data().get_deriv_m()).clear();

    // reset nodal weighted gap and derivative
    cnode->data().getg() = 1.0e12;
    (cnode->data().get_deriv_g()).clear();

    // reset derivative map of lagrange multipliers
    for (int j = 0; j < (int)((cnode->data().get_deriv_z()).size()); ++j)
      (cnode->data().get_deriv_z())[j].clear();
    (cnode->data().get_deriv_z()).resize(0);

    //************************************
    //              friction
    //*************************************
    if (friction_)
    {
      CONTACT::FriNode* frinode = dynamic_cast<CONTACT::FriNode*>(cnode);

      // reset SNodes and Mnodes
      frinode->fri_data().get_s_nodes().clear();
      frinode->fri_data().get_m_nodes().clear();

      // for gp slip
      if (interface_params().get<bool>("GP_SLIP_INCR"))
      {
        // reset jump deriv.
        for (int j = 0; j < (int)((frinode->fri_data().get_deriv_var_jump()).size()); ++j)
          (frinode->fri_data().get_deriv_var_jump())[j].clear();

        (frinode->fri_data().get_deriv_var_jump()).resize(2);

        // reset jumps
        frinode->fri_data().jump_var()[0] = 0.0;
        frinode->fri_data().jump_var()[1] = 0.0;
      }

      //************************************
      //              wear
      //*************************************
      // reset weighted wear increment and derivative
      // only for implicit wear algorithm
      if (wearimpl_ and !wearpv_)
      {
        (cnode->data().get_deriv_w()).clear();
        (cnode->data().get_deriv_wlm()).clear();
      }

      if (wearpv_)
      {
        // reset nodal Mortar wear maps
        for (int j = 0; j < (int)((frinode->wear_data().get_t()).size()); ++j)
          (frinode->wear_data().get_t())[j].clear();
        for (int j = 0; j < (int)((frinode->wear_data().get_e()).size()); ++j)
          (frinode->wear_data().get_e())[j].clear();

        (frinode->wear_data().get_t()).resize(0);
        (frinode->wear_data().get_e()).resize(0);

        (frinode->wear_data().get_deriv_tw()).clear();
        (frinode->wear_data().get_deriv_e()).clear();

        (frinode->data().get_deriv_gw()).clear();
      }
      if (wear_)
      {
        // reset wear increment
        if (!wearpv_) frinode->wear_data().delta_weighted_wear() = 0.0;

        // reset abs. wear.
        // for impl. wear algor. the abs. wear equals the
        // delta-wear
        if (wearimpl_ and !wearpv_) frinode->wear_data().weighted_wear() = 0.0;
      }
    }
  }

  // for both-sided wear with discrete wear
  if (wearboth_ and wearpv_)
  {
    const std::shared_ptr<Core::LinAlg::Map> masternodes =
        Core::LinAlg::allreduce_e_map(*(master_row_nodes()));

    for (int i = 0; i < masternodes->num_my_elements();
        ++i)  // for (int i=0;i<MasterRowNodes()->NumMyElements();++i)
    {
      int gid = masternodes->gid(i);
      Core::Nodes::Node* node = discret().g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      CONTACT::FriNode* frinode = dynamic_cast<CONTACT::FriNode*>(node);

      // reset nodal Mortar wear maps
      for (int j = 0; j < (int)((frinode->wear_data().get_t()).size()); ++j)
        (frinode->wear_data().get_t())[j].clear();
      for (int j = 0; j < (int)((frinode->wear_data().get_e()).size()); ++j)
        (frinode->wear_data().get_e())[j].clear();

      (frinode->wear_data().get_t()).resize(0);
      (frinode->wear_data().get_e()).resize(0);

      (frinode->wear_data().get_deriv_tw()).clear();
      (frinode->wear_data().get_deriv_e()).clear();
    }
  }

  //**********************************************************************
  // In general, it is sufficient to reset search candidates only for
  // all elements in the standard slave column map. However, self contact
  // is an exception here and we need to reset the search candidates of
  // all slave elements in the fully overlapping column map there. This
  // is due to the fact that self contact search is NOT parallelized.
  //**********************************************************************
  if (self_contact())
  {
    // loop over all elements to reset candidates / search lists
    // (use fully overlapping column map of S+M elements)
    for (int i = 0; i < idiscret_->num_my_col_elements(); ++i)
    {
      Core::Elements::Element* ele = idiscret_->l_col_element(i);
      Mortar::Element* mele = dynamic_cast<Mortar::Element*>(ele);

      mele->mo_data().search_elements().resize(0);

      // dual shape function coefficient matrix
      mele->mo_data().reset_dual_shape();
      mele->mo_data().reset_deriv_dual_shape();
    }
  }
  else
  {
    // loop over all elements to reset candidates / search lists
    // (use standard slave column map)
    for (int i = 0; i < slave_col_elements()->num_my_elements(); ++i)
    {
      int gid = slave_col_elements()->gid(i);
      Core::Elements::Element* ele = discret().g_element(gid);
      if (!ele) FOUR_C_THROW("Cannot find ele with gid {}", gid);
      Mortar::Element* mele = dynamic_cast<Mortar::Element*>(ele);

      mele->mo_data().search_elements().resize(0);

      // dual shape function coefficient matrix
      mele->mo_data().reset_dual_shape();
      mele->mo_data().reset_deriv_dual_shape();
    }
  }

  // reset s/m pairs and intcell counters
  smpairs_ = 0;
  smintpairs_ = 0;
  intcells_ = 0;

  return;
}


/*----------------------------------------------------------------------*
 |  create snode n_                                          farah 09/13|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::split_slave_dofs()
{
  // get out of here if active set is empty
  if (snoderowmap_ == nullptr)
  {
    sndofmap_ = std::make_shared<Core::LinAlg::Map>(0, 0, get_comm());
    return;
  }

  else if (snoderowmap_->num_global_elements() == 0)
  {
    sndofmap_ = std::make_shared<Core::LinAlg::Map>(0, 0, get_comm());
    return;
  }

  // define local variables
  int countN = 0;
  std::vector<int> myNgids(snoderowmap_->num_my_elements());

  // dimension check
  double dimcheck = (sdofrowmap_->num_global_elements()) / (snoderowmap_->num_global_elements());
  if (dimcheck != n_dim()) FOUR_C_THROW("SplitSlaveDofs: Nodes <-> Dofs dimension mismatch!");

  // loop over all slave nodes
  for (int i = 0; i < snoderowmap_->num_my_elements(); ++i)
  {
    int gid = snoderowmap_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(node);

    // add first dof to Nmap
    myNgids[countN] = cnode->dofs()[0];
    ++countN;
  }

  // resize the temporary vectors
  myNgids.resize(countN);

  // communicate countN and countT among procs
  int gcountN;
  gcountN = Core::Communication::sum_all(countN, get_comm());

  // check global dimensions
  if ((gcountN) != snoderowmap_->num_global_elements())
    FOUR_C_THROW("SplitSlaveDofs: Splitting went wrong!");

  // create Nmap and Tmap objects
  sndofmap_ = std::make_shared<Core::LinAlg::Map>(gcountN, countN, myNgids.data(), 0, get_comm());

  return;
}


/*----------------------------------------------------------------------*
 |  create mnode n_                                          farah 11/13|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::split_master_dofs()
{
  // get out of here if active set is empty
  if (mnoderowmap_ == nullptr)
  {
    mndofmap_ = std::make_shared<Core::LinAlg::Map>(0, 0, get_comm());
    return;
  }

  else if (mnoderowmap_->num_global_elements() == 0)
  {
    mndofmap_ = std::make_shared<Core::LinAlg::Map>(0, 0, get_comm());
    return;
  }

  // define local variables
  int countN = 0;
  std::vector<int> myNgids(mnoderowmap_->num_my_elements());

  // dimension check
  double dimcheck = (mdofrowmap_->num_global_elements()) / (mnoderowmap_->num_global_elements());
  if (dimcheck != n_dim()) FOUR_C_THROW("SplitMasterDofs: Nodes <-> Dofs dimension mismatch!");

  // loop over all slave nodes
  for (int i = 0; i < mnoderowmap_->num_my_elements(); ++i)
  {
    int gid = mnoderowmap_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(node);

    // add first dof to Nmap
    myNgids[countN] = cnode->dofs()[0];
    ++countN;
  }

  // resize the temporary vectors
  myNgids.resize(countN);

  // communicate countN and countT among procs
  int gcountN;
  gcountN = Core::Communication::sum_all(countN, get_comm());

  // check global dimensions
  if ((gcountN) != mnoderowmap_->num_global_elements())
    FOUR_C_THROW("SplitSlaveDofs: Splitting went wrong!");

  // create Nmap and Tmap objects
  mndofmap_ = std::make_shared<Core::LinAlg::Map>(gcountN, countN, myNgids.data(), 0, get_comm());

  return;
}


/*----------------------------------------------------------------------*
 |  compute element areas (public)                           farah 02/16|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::set_element_areas()
{
  //**********************************************************************
  // In general, it is sufficient to compute element areas only for
  // all elements in the standard slave column map. However, self contact
  // is an exception here and we need the element areas of all elements
  // (slave and master) in the fully overlapping column map there. At the
  // same time we initialize the element data containers for self contact.
  // This is due to the fact that self contact search is NOT parallelized.
  //**********************************************************************
  if (self_contact() or wearboth_)
  {
    // loop over all elements to set current element length / area
    // (use fully overlapping column map)
    for (int i = 0; i < idiscret_->num_my_col_elements(); ++i)
    {
      Mortar::Element* element = dynamic_cast<Mortar::Element*>(idiscret_->l_col_element(i));
      element->initialize_data_container();
      element->mo_data().area() = element->compute_area();
    }
  }
  else
  {
    // refer call back to base class version
    Mortar::Interface::set_element_areas();
  }

  return;
}


/*----------------------------------------------------------------------*
 |  update wear set (dofs)                                   farah 09/13|
 *----------------------------------------------------------------------*/
void Wear::WearInterface::update_w_sets(int offset_if, int maxdofwear, bool bothdiscr)
{
  //********************************************************************
  // WEAR DOFS --  one per node
  //********************************************************************

  //********************************************************************
  // temporary vector of W dofs
  std::vector<int> wdof;

  // gather information over all procs
  std::vector<int> localnumwdof(Core::Communication::num_mpi_ranks(get_comm()));
  std::vector<int> globalnumlmdof(Core::Communication::num_mpi_ranks(get_comm()));
  localnumwdof[Core::Communication::my_mpi_rank(get_comm())] =
      (int)((sdofrowmap_->num_my_elements()) / n_dim());
  globalnumlmdof = Core::Communication::sum_all(localnumwdof, get_comm());

  // compute offset for LM dof initialization for all procs
  int offset = 0;
  for (int k = 0; k < Core::Communication::my_mpi_rank(get_comm()); ++k)
    offset += globalnumlmdof[k];

  // loop over all slave dofs and initialize LM dofs
  for (int i = 0; i < (int)((sdofrowmap_->num_my_elements()) / n_dim()); ++i)
    wdof.push_back(maxdofwear + 1 + offset_if + offset + i);

  // create interface w map
  // (if maxdofglobal_ == 0, we do not want / need this)
  if (maxdofwear > 0)
    wdofmap_ =
        std::make_shared<Core::LinAlg::Map>(-1, (int)wdof.size(), wdof.data(), 0, get_comm());

  //********************************************************************
  // For discrete both-sided wear
  //********************************************************************
  if (bothdiscr)
  {
    // temporary vector of W dofs
    std::vector<int> wmdof;

    maxdofwear += wdofmap_->num_global_elements();

    // gather information over all procs
    std::vector<int> localnumwdof(Core::Communication::num_mpi_ranks(get_comm()));
    std::vector<int> globalnumlmdof(Core::Communication::num_mpi_ranks(get_comm()));
    localnumwdof[Core::Communication::my_mpi_rank(get_comm())] =
        (int)((mdofrowmap_->num_my_elements()) / n_dim());
    globalnumlmdof = Core::Communication::sum_all(localnumwdof, get_comm());

    // compute offset for LM dof initialization for all procs
    int offset = 0;
    for (int k = 0; k < Core::Communication::my_mpi_rank(get_comm()); ++k)
      offset += globalnumlmdof[k];

    // loop over all slave dofs and initialize LM dofs
    for (int i = 0; i < (int)((mdofrowmap_->num_my_elements()) / n_dim()); ++i)
      wmdof.push_back(maxdofwear + 1 + offset_if + offset + i);

    // create interface w map
    // (if maxdofglobal_ == 0, we do not want / need this)
    if (maxdofwear > 0)
      wmdofmap_ =
          std::make_shared<Core::LinAlg::Map>(-1, (int)wmdof.size(), wmdof.data(), 0, get_comm());
  }

  return;
}

FOUR_C_NAMESPACE_CLOSE
