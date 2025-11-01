// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_element.hpp"
#include "4C_contact_friction_node.hpp"
#include "4C_contact_interface.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------*
 |  Assemble slave coordinates (xs)                 gitterle 10/09|
 *----------------------------------------------------------------*/
void CONTACT::Interface::assemble_slave_coord(std::shared_ptr<Core::LinAlg::Vector<double>>& xsmod)
{
  // loop over all slave nodes
  for (int j = 0; j < snoderowmap_->num_my_elements(); ++j)
  {
    int gid = snoderowmap_->gid(j);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    FriNode* cnode = dynamic_cast<FriNode*>(node);

    int dim = cnode->num_dof();

    Core::LinAlg::SerialDenseVector xspatial(dim);
    std::vector<int> dof(dim);
    std::vector<int> owner(dim);

    for (int k = 0; k < dim; ++k)
    {
      xspatial(k) = cnode->xspatial()[k];
      dof[k] = cnode->dofs()[k];
      owner[k] = cnode->owner();
    }

    // do assembly
    Core::LinAlg::assemble(*xsmod, xspatial, dof, owner);
  }
  return;
}

/*----------------------------------------------------------------------*
 |  Evaluate regularized normal forces (nodes)                popp 05/09|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::assemble_reg_normal_forces(
    bool& localisincontact, bool& localactivesetchange)
{
  // penalty parameter
  double pp = interface_params().get<double>("PENALTYPARAM");

  // loop over all slave row nodes on the current interface
  for (int i = 0; i < slave_row_nodes()->num_my_elements(); ++i)
  {
    int gid = slave_row_nodes()->gid(i);
    Core::Nodes::Node* node = discret().g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Node* cnode = dynamic_cast<Node*>(node);

    int dim = cnode->num_dof();
    double gap = cnode->data().getg();

    // modified gap for zero initial gap
    // (if gap is below zero, it is explicitly set to zero)
    // double modgap = cnode->Data().Getg();
    // if (abs(modgap) < 1.0e-10) modgap=0.0;

    double kappa = cnode->data().kappa();

    double lmuzawan = 0.0;
    for (int k = 0; k < dim; ++k)
      lmuzawan += cnode->mo_data().lmuzawa()[k] * cnode->mo_data().n()[k];

#ifdef CONTACTFDPENALTYKC1
    // set lagrangian multipliers explicitly to constant
    // and corresponding derivatives to zero

    for (int j = 0; j < dim; ++j) cnode->MoData().lm()[j] = i * j;

    cnode->Data().GetDerivZ().clear();

    continue;
#endif

    //********************************************************************
    // Decision on active /  inactive nodes (regularization)
    //
    // CASE 1: Penalty approach
    // A node is activated if its weighted gap is negative or deactivated
    // if its gap is equal zero or positive.
    // -> the regularization reads: lambda_n = kappa * pp * < -gap >
    //
    // CASE 2: Uzawa augmented Lagrange approach
    // A node is activated if its Lagrange multiplier, stemming from the
    // last Uzawa Lagrange multiplier AND the current regularization is
    // negative or deactivated if its LM is equal zero or positive.
    // -> the regularization reads: lambda_n = < lmuzawa_n - kappa * pp * gap >
    //
    // As the Uzawa Lagrange multipliers are zero in the penalty approach,
    // the two cases can formally be treted identically, see below.
    // We do not need an explicit separation of cases!
    //
    //********************************************************************

    // Activate/Deactivate node and notice any change
    if ((cnode->active() == false) && (lmuzawan - kappa * pp * gap >= 0))
    {
      cnode->active() = true;
      localactivesetchange = true;

      // std::cout << "node #" << gid << " is now active (";
      // for( int j=0; j<dim; j++)
      //  std::cout << " " << cnode->Dofs()[j] << " ";
      // std::cout << ") gap=" << gap << std::endl;
    }

    else if ((cnode->active() == true) && (lmuzawan - kappa * pp * gap < 0))
    {
      cnode->active() = false;
      localactivesetchange = true;

      // std::cout << "node #" << gid << " is now inactive, gap=" << gap << std::endl;
    }
    //********************************************************************

    // Compute derivZ-entries with the Macauley-Bracket
    // of course, this is only done for active constraints in order
    // for linearization and r.h.s to match!
    if (cnode->active() == true)
    {
      //      std::cout << "GID " << gid << std::endl;
      //      std::cout << "LMUZAWAN " << lmuzawan << std::endl;
      //      std::cout << "GAP " << gap << std::endl;

      localisincontact = true;

      double* normal = cnode->mo_data().n();

      // compute lagrange multipliers and store into node
      for (int j = 0; j < dim; ++j)
        cnode->mo_data().lm()[j] = (lmuzawan - kappa * pp * gap) * normal[j];

      // compute derivatives of lagrange multipliers and store into node

      // contribution of derivative of weighted gap
      std::map<int, double>& derivg = cnode->data().get_deriv_g();
      std::map<int, double>::iterator gcurr;

      // contribution of derivative of normal
      std::vector<Core::Gen::Pairedvector<int, double>>& derivn = cnode->data().get_deriv_n();
      Core::Gen::Pairedvector<int, double>::iterator ncurr;

      for (int j = 0; j < dim; ++j)
      {
        for (gcurr = derivg.begin(); gcurr != derivg.end(); ++gcurr)
          cnode->add_deriv_z_value(j, gcurr->first, -kappa * pp * (gcurr->second) * normal[j]);
        for (ncurr = (derivn[j]).begin(); ncurr != (derivn[j]).end(); ++ncurr)
          cnode->add_deriv_z_value(j, ncurr->first, -kappa * pp * gap * ncurr->second);
        for (ncurr = (derivn[j]).begin(); ncurr != (derivn[j]).end(); ++ncurr)
          cnode->add_deriv_z_value(j, ncurr->first, +lmuzawan * ncurr->second);
      }
    }

    // be sure to remove all LM-related stuff from inactive nodes
    else
    {
      // clear lagrange multipliers
      for (int j = 0; j < dim; ++j) cnode->mo_data().lm()[j] = 0.0;

      // clear derivz
      cnode->data().get_deriv_z().clear();

    }  // Macauley-Bracket
  }  // loop over slave nodes

  return;
}

/*----------------------------------------------------------------------*
 |  Evaluate regularized tangential forces                gitterle 10/09|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::assemble_reg_tangent_forces_penalty()
{
  // penalty parameter in tangential direction
  double ppnor = interface_params().get<double>("PENALTYPARAM");
  double pptan = interface_params().get<double>("PENALTYPARAMTAN");
  double frcoeff = interface_params().get<double>("FRCOEFF");

  auto ftype = Teuchos::getIntegralValue<CONTACT::FrictionType>(interface_params(), "FRICTION");

  // loop over all slave row nodes on the current interface
  for (int i = 0; i < slave_row_nodes()->num_my_elements(); ++i)
  {
    int gid = slave_row_nodes()->gid(i);
    Core::Nodes::Node* node = discret().g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    FriNode* cnode = dynamic_cast<FriNode*>(node);

    // get some information form the node
    double gap = cnode->data().getg();
    int numdof = cnode->num_dof();
    double kappa = cnode->data().kappa();
    double* n = cnode->mo_data().n();

    // Lagrange multiplier from Uzawa algorithm
    Core::LinAlg::SerialDenseMatrix lmuzawa(numdof, 1);
    for (int k = 0; k < numdof; ++k) lmuzawa(k, 0) = cnode->mo_data().lmuzawa()[k];

    // Lagrange multiplier in normal direction
    double lmuzawan = 0.0;
    for (int k = 0; k < numdof; ++k)
      lmuzawan += cnode->mo_data().lmuzawa()[k] * cnode->mo_data().n()[k];

    // tangential plane
    Core::LinAlg::SerialDenseMatrix tanplane(numdof, numdof);
    if (numdof == 3)
    {
      tanplane(0, 0) = 1 - (n[0] * n[0]);
      tanplane(0, 1) = -(n[0] * n[1]);
      tanplane(0, 2) = -(n[0] * n[2]);
      tanplane(1, 0) = -(n[1] * n[0]);
      tanplane(1, 1) = 1 - (n[1] * n[1]);
      tanplane(1, 2) = -(n[1] * n[2]);

      tanplane(2, 0) = -(n[2] * n[0]);
      tanplane(2, 1) = -(n[2] * n[1]);
      tanplane(2, 2) = 1 - (n[2] * n[2]);
    }
    else if (numdof == 2)
    {
      tanplane(0, 0) = 1 - (n[0] * n[0]);
      tanplane(0, 1) = -(n[0] * n[1]);

      tanplane(1, 0) = -(n[1] * n[0]);
      tanplane(1, 1) = 1 - (n[1] * n[1]);
    }
    else
      FOUR_C_THROW("Error in AssembleTangentForces: Unknown dimension.");

    // evaluate traction
    Core::LinAlg::SerialDenseMatrix jumpvec(numdof, 1);

    for (int i = 0; i < numdof; i++) jumpvec(i, 0) = cnode->fri_data().jump()[i];

    // evaluate kappa.pptan.jumptan
    Core::LinAlg::SerialDenseMatrix temptrac(numdof, 1);
    Core::LinAlg::multiply(0.0, temptrac, kappa * pptan, tanplane, jumpvec);

    // fill vector tractionold
    std::vector<double> tractionold(numdof);
    for (int i = 0; i < numdof; i++) tractionold[i] = cnode->fri_data().tractionold()[i];

    // Evaluate trailtraction (tractionold+temptrac in penalty case)
    std::vector<double> trailtraction(numdof);
    double magnitude = 0;
    for (int i = 0; i < numdof; i++)
    {
      trailtraction[i] = tractionold[i] + temptrac(i, 0);
      magnitude += (trailtraction[i] * trailtraction[i]);
    }

    // evaluate magnitude of trailtraction
    magnitude = sqrt(magnitude);

    // evaluate maximal tangential traction
    double maxtantrac = frcoeff * (lmuzawan - kappa * ppnor * gap);

    if (cnode->active() == false)
    {
      // do nothing
      cnode->fri_data().slip() = false;
    }
    else if (cnode->active() == true &&
             ((abs(maxtantrac) - magnitude >= 0) or ftype == CONTACT::FrictionType::stick))
    {
      // std::cout << "Node " << gid << " is stick" << std::endl;
      cnode->fri_data().slip() = false;

      // in the stick case, traction is trailtraction
      for (int i = 0; i < numdof; i++) cnode->fri_data().traction()[i] = trailtraction[i];

      // compute lagrange multipliers and store into node
      for (int j = 0; j < numdof; ++j)
        cnode->mo_data().lm()[j] = n[j] * (-kappa * ppnor * gap) + trailtraction[j];
    }
    else
    {
      // std::cout << "Node " << gid << " is slip" << std::endl;
      cnode->fri_data().slip() = true;

      // in the slip case, traction is evaluated with a return map algorithm
      for (int i = 0; i < numdof; i++)
        cnode->fri_data().traction()[i] = maxtantrac / magnitude * trailtraction[i];

      // compute lagrange multipliers and store into node
      for (int j = 0; j < numdof; ++j)
        cnode->mo_data().lm()[j] =
            n[j] * (-kappa * ppnor * gap) + maxtantrac / magnitude * trailtraction[j];
    }

    // linearization of contact forces (lagrange multipliers)
    // this consists the linearization of the tangential part,
    // the normal part was already done in AssembleRegNormalTraction

    // stick nodes
    if (cnode->active() == true && cnode->fri_data().slip() == false)
    {
      /***************************************** tanplane.deriv(jump) ***/
      std::vector<std::map<int, double>>& derivjump = cnode->fri_data().get_deriv_jump();
      std::map<int, double>::iterator colcurr;
      Core::Gen::Pairedvector<int, double>::iterator _colcurr;

      // loop over dimensions
      for (int dimrow = 0; dimrow < numdof; ++dimrow)
      {
        for (int dim = 0; dim < numdof; ++dim)
        {
          // loop over all entries of the current derivative map
          for (colcurr = derivjump[dim].begin(); colcurr != derivjump[dim].end(); ++colcurr)
          {
            int col = colcurr->first;
            double val = pptan * kappa * (colcurr->second) * tanplane(dimrow, dim);
            cnode->add_deriv_z_value(dimrow, col, val);
          }
        }
      }

      /**************************************** deriv(tanplane).jump  ***/
      std::vector<Core::Gen::Pairedvector<int, double>>& derivn = cnode->data().get_deriv_n();

      // loop over dimensions
      for (int dimrow = 0; dimrow < numdof; ++dimrow)
      {
        // loop over all entries of the current derivative map
        for (_colcurr = derivn[dimrow].begin(); _colcurr != derivn[dimrow].end(); ++_colcurr)
        {
          for (int dim = 0; dim < numdof; ++dim)
          {
            int col = _colcurr->first;
            double val =
                -pptan * kappa * (_colcurr->second) * n[dim] * cnode->fri_data().jump()[dim];
            cnode->add_deriv_z_value(dimrow, col, val);
          }
        }
      }

      // loop over dimensions
      for (int dim = 0; dim < numdof; ++dim)
      {
        // loop over all entries of the current derivative map
        for (_colcurr = derivn[dim].begin(); _colcurr != derivn[dim].end(); ++_colcurr)
        {
          for (int dimrow = 0; dimrow < numdof; ++dimrow)
          {
            int col = _colcurr->first;
            double val =
                -pptan * kappa * (_colcurr->second) * n[dimrow] * cnode->fri_data().jump()[dim];
            cnode->add_deriv_z_value(dimrow, col, val);
          }
        }
      }
    }
    // slip nodes
    else if (cnode->active() == true && cnode->fri_data().slip() == true)
    {
      /******************** tanplane.deriv(jump).maxtantrac/magnidude ***/

      std::vector<std::map<int, double>>& derivjump = cnode->fri_data().get_deriv_jump();
      std::map<int, double>::iterator colcurr;
      Core::Gen::Pairedvector<int, double>::iterator _colcurr;

      // loop over dimensions
      for (int dimrow = 0; dimrow < numdof; ++dimrow)
      {
        for (int dim = 0; dim < numdof; ++dim)
        {
          // loop over all entries of the current derivative map
          for (colcurr = derivjump[dim].begin(); colcurr != derivjump[dim].end(); ++colcurr)
          {
            int col = colcurr->first;
            double val =
                pptan * kappa * (colcurr->second) * tanplane(dimrow, dim) * maxtantrac / magnitude;
            cnode->add_deriv_z_value(dimrow, col, val);
          }
        }
      }

      /******************** deriv(tanplane).jump.maxtantrac/magnitude ***/
      std::vector<Core::Gen::Pairedvector<int, double>>& derivn = cnode->data().get_deriv_n();

      // loop over dimensions
      for (int dimrow = 0; dimrow < numdof; ++dimrow)
      {
        // loop over all entries of the current derivative map
        for (_colcurr = derivn[dimrow].begin(); _colcurr != derivn[dimrow].end(); ++_colcurr)
        {
          for (int dim = 0; dim < numdof; ++dim)
          {
            int col = _colcurr->first;
            double val = -pptan * kappa * (_colcurr->second) * n[dim] *
                         cnode->fri_data().jump()[dim] * maxtantrac / magnitude;
            cnode->add_deriv_z_value(dimrow, col, val);
          }
        }
      }
      // loop over dimensions
      for (int dim = 0; dim < numdof; ++dim)
      {
        // loop over all entries of the current derivative map
        for (_colcurr = derivn[dim].begin(); _colcurr != derivn[dim].end(); ++_colcurr)
        {
          for (int dimrow = 0; dimrow < numdof; ++dimrow)
          {
            int col = _colcurr->first;
            double val = -pptan * kappa * (_colcurr->second) * n[dimrow] *
                         cnode->fri_data().jump()[dim] * maxtantrac / magnitude;
            cnode->add_deriv_z_value(dimrow, col, val);
          }
        }
      }

      /******************** tanplane.jump.deriv(maxtantrac)/magnitude ***/
      std::map<int, double>& derivg = cnode->data().get_deriv_g();
      std::map<int, double>::iterator gcurr;

      for (int j = 0; j < numdof; ++j)
      {
        for (gcurr = derivg.begin(); gcurr != derivg.end(); ++gcurr)
        {
          cnode->add_deriv_z_value(j, gcurr->first,
              -frcoeff * kappa * ppnor * (gcurr->second) * trailtraction[j] / magnitude);
        }
      }

      /******************** tanplane.jump.maxtantrac/deriv(magnitude) ***/
      // vector double temp
      std::vector<double> temp(numdof);
      for (int dim = 0; dim < numdof; ++dim)
        temp[dim] = -maxtantrac / (magnitude * magnitude) * trailtraction[dim];

      // loop over dimensions
      for (int dimout = 0; dimout < numdof; ++dimout)
      {
        double traction = 0;
        for (int dim = 0; dim < numdof; ++dim)
          traction += tanplane(dimout, dim) * cnode->fri_data().jump()[dim] * kappa * pptan;

        traction += tractionold[dimout];

        for (int dim = 0; dim < numdof; ++dim)
        {
          // loop over all entries of the current derivative map
          for (colcurr = derivjump[dim].begin(); colcurr != derivjump[dim].end(); ++colcurr)
          {
            int col = colcurr->first;
            double val =
                tanplane(dimout, dim) * pptan * kappa * (colcurr->second) * traction / magnitude;

            for (int dimrow = 0; dimrow < numdof; ++dimrow)
            {
              double val1 = val * temp[dimrow];
              cnode->add_deriv_z_value(dimrow, col, val1);
            }
          }
        }
      }

      // loop over dimensions
      for (int dimout = 0; dimout < numdof; ++dimout)
      {
        double traction = 0;
        for (int dim = 0; dim < numdof; ++dim)
          traction += tanplane(dimout, dim) * cnode->fri_data().jump()[dim] * kappa * pptan;

        traction += tractionold[dimout];

        // loop over all entries of the current derivative map
        for (_colcurr = derivn[dimout].begin(); _colcurr != derivn[dimout].end(); ++_colcurr)
        {
          int col = _colcurr->first;

          for (int dim = 0; dim < numdof; ++dim)
          {
            double val = -_colcurr->second * n[dim] * cnode->fri_data().jump()[dim] * traction /
                         magnitude * pptan * kappa;
            for (int dimrow = 0; dimrow < numdof; ++dimrow)
            {
              double val1 = val * temp[dimrow];
              cnode->add_deriv_z_value(dimrow, col, val1);
            }
          }
        }
      }

      // loop over dimensions
      for (int dimout = 0; dimout < numdof; ++dimout)
      {
        double traction = 0;
        for (int dim = 0; dim < numdof; ++dim)
          traction += tanplane(dimout, dim) * cnode->fri_data().jump()[dim] * kappa * pptan;

        traction += tractionold[dimout];

        for (int dim = 0; dim < numdof; ++dim)
        {
          // loop over all entries of the current derivative map
          for (_colcurr = derivn[dim].begin(); _colcurr != derivn[dim].end(); ++_colcurr)
          {
            int col = _colcurr->first;

            double val = -_colcurr->second * n[dimout] * cnode->fri_data().jump()[dim] * traction /
                         magnitude * pptan * kappa;

            for (int dimrow = 0; dimrow < numdof; ++dimrow)
            {
              double val1 = val * temp[dimrow];
              cnode->add_deriv_z_value(dimrow, col, val1);
            }
          }
        }
      }
    }  // if Slip == true
    else
    {
      // clear tractions
      for (int j = 0; j < numdof; ++j) cnode->mo_data().lm()[j] = 0;
      // clear derivz
      cnode->data().get_deriv_z().clear();
    }
  }  // loop over active nodes
}

/*----------------------------------------------------------------------*
 |  Evaluate regularized tangential forces                gitterle 10/09|
 |  (Uzawa Aug. Lagr.)                                                  |
 *----------------------------------------------------------------------*/
void CONTACT::Interface::assemble_reg_tangent_forces_uzawa()
{
  // penalty parameter in tangential direction
  double ppnor = interface_params().get<double>("PENALTYPARAM");
  double pptan = interface_params().get<double>("PENALTYPARAMTAN");
  double frcoeff = interface_params().get<double>("FRCOEFF");

  auto ftype = Teuchos::getIntegralValue<CONTACT::FrictionType>(interface_params(), "FRICTION");

  // loop over all slave row nodes on the current interface
  for (int i = 0; i < slave_row_nodes()->num_my_elements(); ++i)
  {
    int gid = slave_row_nodes()->gid(i);
    Core::Nodes::Node* node = discret().g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    FriNode* cnode = dynamic_cast<FriNode*>(node);

    // get some information form the node
    double gap = cnode->data().getg();
    int dim = cnode->num_dof();
    double kappa = cnode->data().kappa();
    double* n = cnode->mo_data().n();

    // Lagrange multiplier from Uzawa algorithm
    Core::LinAlg::SerialDenseMatrix lmuzawa(dim, 1);
    for (int k = 0; k < dim; ++k) lmuzawa(k, 0) = cnode->mo_data().lmuzawa()[k];

    // Lagrange multiplier in normal direction
    double lmuzawan = 0.0;
    for (int k = 0; k < dim; ++k)
      lmuzawan += cnode->mo_data().lmuzawa()[k] * cnode->mo_data().n()[k];

    // tangential plane
    Core::LinAlg::SerialDenseMatrix tanplane(dim, dim);
    if (dim == 3)
    {
      tanplane(0, 0) = 1 - (n[0] * n[0]);
      tanplane(0, 1) = -(n[0] * n[1]);
      tanplane(0, 2) = -(n[0] * n[2]);
      tanplane(1, 0) = -(n[1] * n[0]);
      tanplane(1, 1) = 1 - (n[1] * n[1]);
      tanplane(1, 2) = -(n[1] * n[2]);

      tanplane(2, 0) = -(n[2] * n[0]);
      tanplane(2, 1) = -(n[2] * n[1]);
      tanplane(2, 2) = 1 - (n[2] * n[2]);
    }
    else if (dim == 2)
    {
      tanplane(0, 0) = 1 - (n[0] * n[0]);
      tanplane(0, 1) = -(n[0] * n[1]);

      tanplane(1, 0) = -(n[1] * n[0]);
      tanplane(1, 1) = 1 - (n[1] * n[1]);
    }
    else
      FOUR_C_THROW("Error in AssembleTangentForces: Unknown dimension.");

    // Lagrange multiplier in tangential direction
    Core::LinAlg::SerialDenseMatrix lmuzawatan(dim, 1);
    Core::LinAlg::multiply(lmuzawatan, tanplane, lmuzawa);

    // evaluate traction
    Core::LinAlg::SerialDenseMatrix jumpvec(dim, 1);

    for (int i = 0; i < dim; i++) jumpvec(i, 0) = cnode->fri_data().jump()[i];

    // evaluate kappa.pptan.jumptan
    Core::LinAlg::SerialDenseMatrix temptrac(dim, 1);
    Core::LinAlg::multiply(0.0, temptrac, kappa * pptan, tanplane, jumpvec);

    // Evaluate trailtraction
    std::vector<double> trailtraction(dim);
    double magnitude = 0;
    for (int i = 0; i < dim; i++)
    {
      trailtraction[i] = lmuzawatan(i, 0) + temptrac(i, 0);
      magnitude += (trailtraction[i] * trailtraction[i]);
    }

    // evaluate magnitude of trailtraction
    magnitude = sqrt(magnitude);

    // evaluate maximal tangential traction
    double maxtantrac = frcoeff * (lmuzawan - kappa * ppnor * gap);

    if (cnode->active() == false)
    {
    }
    else if (cnode->active() == true &&
             ((abs(maxtantrac) - magnitude >= 0) or ftype == CONTACT::FrictionType::stick))
    {
      // std::cout << "Node " << gid << " is stick" << std::endl;
      cnode->fri_data().slip() = false;

      // compute lagrange multipliers and store into node
      for (int j = 0; j < dim; ++j)
        cnode->mo_data().lm()[j] = n[j] * (lmuzawan - kappa * ppnor * gap) + trailtraction[j];
    }
    else
    {
      // std::cout << "Node " << gid << " is slip" << std::endl;
      cnode->fri_data().slip() = true;

      // compute lagrange multipliers and store into node
      for (int j = 0; j < dim; ++j)
        cnode->mo_data().lm()[j] =
            n[j] * (lmuzawan - kappa * ppnor * gap) + trailtraction[j] * maxtantrac / magnitude;
    }

    // linearization of contact forces (lagrange multipliers)
    // this consists the linearization of the tangential part,
    // the normal part was already done in AssembleRegNormalTraction

    // stick nodes
    if (cnode->active() == true && cnode->fri_data().slip() == false)
    {
      /***************************************** tanplane.deriv(jump) ***/
      std::vector<std::map<int, double>>& derivjump = cnode->fri_data().get_deriv_jump();
      std::map<int, double>::iterator colcurr;
      Core::Gen::Pairedvector<int, double>::iterator _colcurr;
      const int numdof = cnode->num_dof();

      // loop over dimensions
      for (int dimrow = 0; dimrow < numdof; ++dimrow)
      {
        for (int dim = 0; dim < numdof; ++dim)
        {
          // loop over all entries of the current derivative map
          for (colcurr = derivjump[dim].begin(); colcurr != derivjump[dim].end(); ++colcurr)
          {
            int col = colcurr->first;
            double val = pptan * kappa * (colcurr->second) * tanplane(dimrow, dim);
            cnode->add_deriv_z_value(dimrow, col, val);
          }
        }
      }

      /******************************* deriv(tanplane).(lmuzawa+jump) ***/
      std::vector<Core::Gen::Pairedvector<int, double>>& derivn = cnode->data().get_deriv_n();

      // loop over dimensions
      for (int dimrow = 0; dimrow < numdof; ++dimrow)
      {
        // loop over all entries of the current derivative map
        for (_colcurr = derivn[dimrow].begin(); _colcurr != derivn[dimrow].end(); ++_colcurr)
        {
          for (int dim = 0; dim < numdof; ++dim)
          {
            int col = _colcurr->first;
            double val =
                -pptan * kappa * (_colcurr->second) * n[dim] * (cnode->fri_data().jump()[dim]);
            val = val - (_colcurr->second) * n[dim] * (cnode->mo_data().lmuzawa()[dim]);
            cnode->add_deriv_z_value(dimrow, col, val);
          }
        }
      }

      // loop over dimensions
      for (int dim = 0; dim < numdof; ++dim)
      {
        // loop over all entries of the current derivative map
        for (_colcurr = derivn[dim].begin(); _colcurr != derivn[dim].end(); ++_colcurr)
        {
          for (int dimrow = 0; dimrow < numdof; ++dimrow)
          {
            int col = _colcurr->first;
            double val =
                -pptan * kappa * (_colcurr->second) * n[dimrow] * (cnode->fri_data().jump()[dim]);
            val = val - (_colcurr->second) * n[dimrow] * (cnode->mo_data().lmuzawa()[dim]);
            cnode->add_deriv_z_value(dimrow, col, val);
          }
        }
      }
    }

    // slip nodes
    else if (cnode->active() == true && cnode->fri_data().slip() == true)
    {
      /***************************************** tanplane.deriv(jump) ***/
      std::vector<std::map<int, double>>& derivjump = cnode->fri_data().get_deriv_jump();
      std::map<int, double>::iterator colcurr;
      Core::Gen::Pairedvector<int, double>::iterator _colcurr;
      const int numdof = cnode->num_dof();

      // loop over dimensions
      for (int dimrow = 0; dimrow < numdof; ++dimrow)
      {
        for (int dim = 0; dim < numdof; ++dim)
        {
          // loop over all entries of the current derivative map
          for (colcurr = derivjump[dim].begin(); colcurr != derivjump[dim].end(); ++colcurr)
          {
            int col = colcurr->first;
            double val =
                pptan * kappa * (colcurr->second) * tanplane(dimrow, dim) * maxtantrac / magnitude;
            cnode->add_deriv_z_value(dimrow, col, val);
          }
        }
      }

      /******************************* deriv(tanplane).(lmuzawa+jump) ***/
      std::vector<Core::Gen::Pairedvector<int, double>>& derivn = cnode->data().get_deriv_n();

      // loop over dimensions
      for (int dimrow = 0; dimrow < numdof; ++dimrow)
      {
        // loop over all entries of the current derivative map
        for (_colcurr = derivn[dimrow].begin(); _colcurr != derivn[dimrow].end(); ++_colcurr)
        {
          for (int dim = 0; dim < numdof; ++dim)
          {
            int col = _colcurr->first;
            double val =
                -pptan * kappa * (_colcurr->second) * n[dim] * cnode->fri_data().jump()[dim];
            val = (val - (_colcurr->second) * n[dim] * (cnode->mo_data().lmuzawa()[dim])) *
                  maxtantrac / magnitude;
            cnode->add_deriv_z_value(dimrow, col, val);
          }
        }
      }

      // loop over dimensions
      for (int dim = 0; dim < numdof; ++dim)
      {
        // loop over all entries of the current derivative map
        for (_colcurr = derivn[dim].begin(); _colcurr != derivn[dim].end(); ++_colcurr)
        {
          for (int dimrow = 0; dimrow < numdof; ++dimrow)
          {
            int col = _colcurr->first;
            double val =
                -pptan * kappa * (_colcurr->second) * n[dimrow] * cnode->fri_data().jump()[dim];
            val = (val - (_colcurr->second) * n[dimrow] * (cnode->mo_data().lmuzawa()[dim])) *
                  maxtantrac / magnitude;
            cnode->add_deriv_z_value(dimrow, col, val);
          }
        }
      }

      /******************** tanplane.jump.deriv(maxtantrac)/magnitude ***/
      std::map<int, double>& derivg = cnode->data().get_deriv_g();
      std::map<int, double>::iterator gcurr;

      for (int j = 0; j < numdof; ++j)
      {
        for (gcurr = derivg.begin(); gcurr != derivg.end(); ++gcurr)
        {
          cnode->add_deriv_z_value(j, gcurr->first,
              -frcoeff * kappa * ppnor * (gcurr->second) * trailtraction[j] / magnitude);
        }
      }

      for (int j = 0; j < numdof; ++j)
      {
        for (_colcurr = (derivn[j]).begin(); _colcurr != (derivn[j]).end(); ++_colcurr)
        {
          for (int k = 0; k < numdof; ++k)
          {
            double val =
                frcoeff * (_colcurr->second) * lmuzawa(j, 0) * trailtraction[k] / magnitude;
            cnode->add_deriv_z_value(k, _colcurr->first, val);
          }
        }
      }

      /******************** tanplane.jump.maxtantrac/deriv(magnitude) ***/
      // vector double temp
      std::vector<double> temp(numdof);
      for (int dim = 0; dim < numdof; ++dim)
        temp[dim] = -maxtantrac / (magnitude * magnitude) * trailtraction[dim];

      // loop over dimensions
      for (int dimout = 0; dimout < numdof; ++dimout)
      {
        double traction = 0;
        for (int dim = 0; dim < numdof; ++dim)
          traction += tanplane(dimout, dim) *
                      (lmuzawa(dim, 0) + cnode->fri_data().jump()[dim] * kappa * pptan);

        for (int dim = 0; dim < numdof; ++dim)
        {
          // loop over all entries of the current derivative map
          for (colcurr = derivjump[dim].begin(); colcurr != derivjump[dim].end(); ++colcurr)
          {
            int col = colcurr->first;
            double val =
                tanplane(dimout, dim) * pptan * kappa * (colcurr->second) * traction / magnitude;

            for (int dimrow = 0; dimrow < numdof; ++dimrow)
            {
              double val1 = val * temp[dimrow];
              cnode->add_deriv_z_value(dimrow, col, val1);
            }
          }
        }
      }

      // loop over dimensions
      for (int dimout = 0; dimout < numdof; ++dimout)
      {
        double traction = 0;
        for (int dim = 0; dim < numdof; ++dim)
          traction += tanplane(dimout, dim) *
                      (lmuzawa(dim, 0) + cnode->fri_data().jump()[dim] * kappa * pptan);

        // loop over all entries of the current derivative map
        for (_colcurr = derivn[dimout].begin(); _colcurr != derivn[dimout].end(); ++_colcurr)
        {
          int col = _colcurr->first;

          for (int dim = 0; dim < numdof; ++dim)
          {
            double val = -_colcurr->second * n[dim] *
                         (lmuzawa(dim, 0) + cnode->fri_data().jump()[dim] * pptan * kappa) *
                         traction / magnitude;
            for (int dimrow = 0; dimrow < numdof; ++dimrow)
            {
              double val1 = val * temp[dimrow];
              cnode->add_deriv_z_value(dimrow, col, val1);
            }
          }
        }
      }

      // loop over dimensions
      for (int dimout = 0; dimout < numdof; ++dimout)
      {
        double traction = 0;
        for (int dim = 0; dim < numdof; ++dim)
          traction += tanplane(dimout, dim) *
                      (lmuzawa(dim, 0) + cnode->fri_data().jump()[dim] * kappa * pptan);

        for (int dim = 0; dim < numdof; ++dim)
        {
          // loop over all entries of the current derivative map
          for (_colcurr = derivn[dim].begin(); _colcurr != derivn[dim].end(); ++_colcurr)
          {
            int col = _colcurr->first;
            double val = -_colcurr->second * n[dimout] *
                         (lmuzawa(dim, 0) + cnode->fri_data().jump()[dim] * pptan * kappa) *
                         traction / magnitude;

            for (int dimrow = 0; dimrow < numdof; ++dimrow)
            {
              double val1 = val * temp[dimrow];
              cnode->add_deriv_z_value(dimrow, col, val1);
            }
          }
        }
      }
    }  // if Slip == true
    else
    {
      // clear tractions
      for (int j = 0; j < dim; ++j) cnode->mo_data().lm()[j] = 0;
      // clear derivz
      cnode->data().get_deriv_z().clear();
    }
  }  // loop over active nodes
}

/*----------------------------------------------------------------------*
 |  Assemble derivatives of lagrange multipliers              popp 05/09|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::assemble_lin_z(Core::LinAlg::SparseMatrix& linzglobal)
{
  // loop over all slave nodes (row map)
  for (int i = 0; i < snoderowmap_->num_my_elements(); ++i)
  {
    int gid = snoderowmap_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Node* cnode = dynamic_cast<Node*>(node);

    if (cnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
      FOUR_C_THROW("AssembleLinZ: Node ownership inconsistency!");

    // derivz is the std::vector<map> we want to assemble
    std::vector<std::map<int, double>>& derivz = cnode->data().get_deriv_z();

    if ((int)derivz.size() > 0)
    {
      int rowsize = cnode->num_dof();
      int colsize = (int)derivz[0].size();

      // consistency check
      for (int j = 0; j < rowsize - 1; ++j)
        if ((int)derivz[j].size() != (int)derivz[j + 1].size())
          FOUR_C_THROW("AssembleLinZ: Column dim. of nodal derivz-map is inconsistent!");

      std::map<int, double>::iterator colcurr;

      // loop over dofs
      for (int k = 0; k < rowsize; ++k)
      {
        int row = cnode->dofs()[k];  // row index equals global dof index of this #i node's dof k
        int l = 0;

        // loop over all directional derivative entries using the map iterator
        for (colcurr = derivz[k].begin(); colcurr != derivz[k].end(); ++colcurr)
        {
          int col =
              colcurr->first;  // col index equals global id of directional derivative component ,l
          double val = colcurr->second;
          linzglobal.assemble(val, row, col);
          l++;
        }

        if (l != colsize) FOUR_C_THROW("AssembleLinZ: l = {} but colsize = {}", k, colsize);
      }
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Assemble matrix with nodal tangents or/and normals         popp 01/08|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::assemble_tn(std::shared_ptr<Core::LinAlg::SparseMatrix> tglobal,
    std::shared_ptr<Core::LinAlg::SparseMatrix> nglobal)
{
  // nothing to do if no active nodes
  if (activenodes_ == nullptr) return;

  if (Interface::n_dim() != 2 && n_dim() != 3) FOUR_C_THROW("Dim() must be either 2 or 3!");

  // loop over all active slave nodes of the interface
  for (int i = 0; i < activenodes_->num_my_elements(); ++i)
  {
    int gid = activenodes_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    auto* cnode = dynamic_cast<Node*>(node);
    const int numdof = cnode->num_dof();

    if (cnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
      FOUR_C_THROW("AssembleTN: Node ownership inconsistency!");

    if (tglobal != nullptr)
    {
      if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
      {
        if (Interface::n_dim() == 2)
        {
          // prepare assembly
          std::vector<int> lmrowT(cnode->num_dof());
          std::vector<int> lmrowownerT(cnode->num_dof());
          std::vector<int> lmcol(cnode->num_dof());

          Core::LinAlg::SerialDenseMatrix Tnode(numdof, numdof);
          for (int i = 0; i < numdof; ++i)
          {
            lmrowT[i] = cnode->dofs()[i];
            lmrowownerT[i] = cnode->owner();
            for (int j = 0; j < numdof; ++j)
            {
              lmcol[j] = cnode->dofs()[j];
              Tnode(i, j) = cnode->data().txi()[i] * cnode->data().txi()[j];
            }
          }
          tglobal->assemble(-1, Tnode, lmrowT, lmrowownerT, lmcol);
        }
        else if (Interface::n_dim() == 3)
        {
          std::vector<int> lmrowT(cnode->num_dof());
          std::vector<int> lmrowownerT(cnode->num_dof());
          std::vector<int> lmcol(cnode->num_dof());

          Core::LinAlg::SerialDenseMatrix Tnode(numdof, numdof);

          for (int i = 0; i < numdof; ++i)
          {
            lmrowT[i] = cnode->dofs()[i];
            lmrowownerT[i] = cnode->owner();
            for (int j = 0; j < numdof; ++j)
            {
              lmcol[j] = cnode->dofs()[j];
              Tnode(i, j) = cnode->data().txi()[i] * cnode->data().txi()[j] +
                            cnode->data().teta()[i] * cnode->data().teta()[j];
            }
          }
          tglobal->assemble(-1, Tnode, lmrowT, lmrowownerT, lmcol);
        }
        else
          FOUR_C_THROW("Dim() must be either 2D or 3D");
      }
      else
      {
        if (Interface::n_dim() == 2)
        {
          // prepare assembly
          int colsize = numdof;
          std::vector<int> lmrowT(1);
          std::vector<int> lmrowownerT(1);
          std::vector<int> lmcol(colsize);

          lmrowT[0] = activet_->gid(i);
          lmrowownerT[0] = cnode->owner();

          /**************************************************** T-matrix ******/
          Core::LinAlg::SerialDenseMatrix Tnode(1, colsize);

          for (int j = 0; j < colsize; ++j)
          {
            lmcol[j] = cnode->dofs()[j];
            Tnode(0, j) = cnode->data().txi()[j];
          }

          // assemble into matrix of normal vectors T
          tglobal->assemble(-1, Tnode, lmrowT, lmrowownerT, lmcol);
        }

        else if (Interface::n_dim() == 3)
        {
          // prepare assembly
          int colsize = numdof;
          std::vector<int> lmrowT(2);
          std::vector<int> lmrowownerT(2);
          std::vector<int> lmcol(colsize);

          lmrowT[0] = activet_->gid(2 * i);
          lmrowT[1] = activet_->gid(2 * i + 1);
          lmrowownerT[0] = cnode->owner();
          lmrowownerT[1] = cnode->owner();

          /**************************************************** T-matrix ******/
          Core::LinAlg::SerialDenseMatrix Tnode(2, colsize);

          for (int j = 0; j < colsize; ++j)
          {
            lmcol[j] = cnode->dofs()[j];
            Tnode(0, j) = cnode->data().txi()[j];
            Tnode(1, j) = cnode->data().teta()[j];
          }

          // assemble into matrix of normal vectors T
          tglobal->assemble(-1, Tnode, lmrowT, lmrowownerT, lmcol);
        }
        else
          FOUR_C_THROW("Dim() must be either 2D or 3D");
      }
    }

    if (nglobal != nullptr)
    {
      // nodal normal
      double* nodalnormal = cnode->mo_data().n();

      int row = activen_->gid(i);

      // add normal to corresponding row in global matrix
      for (int k = 0; k < numdof; ++k)
        nglobal->assemble(
            nodalnormal[k], row, cnode->dofs()[k]);  // use the first dof for normal direction!!!
    }
  }
}

/*----------------------------------------------------------------------*
 |  Assemble matrix S containing gap g~ derivatives           popp 02/09|
 |  PS: "AssembleS" is an outdated name which could make                |
 |  you confused.                                                       |
 *----------------------------------------------------------------------*/
void CONTACT::Interface::assemble_s(Core::LinAlg::SparseMatrix& sglobal)
{
  // nothing to do if no active nodes
  if (activenodes_ == nullptr) return;

  // loop over all active slave nodes of the interface
  for (int i = 0; i < activenodes_->num_my_elements(); ++i)
  {
    int gid = activenodes_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    auto* cnode = dynamic_cast<Node*>(node);

    if (cnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
      FOUR_C_THROW("AssembleS: Node ownership inconsistency!");

    // prepare assembly
    std::map<int, double>& dgmap = cnode->data().get_deriv_g();
    std::map<int, double>::iterator colcurr;
    int row = activen_->gid(i);

    for (colcurr = dgmap.begin(); colcurr != dgmap.end(); ++colcurr)
    {
      int col = colcurr->first;
      double val = colcurr->second;

      // do not assemble zeros into s matrix
      if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
      {
        for (int j = 0; j < cnode->num_dof(); j++)
          if (abs(val * cnode->mo_data().n()[j]) > 1.0e-12)
            sglobal.assemble(val * cnode->mo_data().n()[j], cnode->dofs()[j], col);
      }
      else if (abs(val) > 1.0e-12)
        sglobal.assemble(val, row, col);
    }
  }  // for (int i=0;i<activenodes_->NumMyElements();++i)
}

/*----------------------------------------------------------------------*
 |  Assemble tangent deriv or/and normal deriv matrix         popp 05/08|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::assemble_t_nderiv(std::shared_ptr<Core::LinAlg::SparseMatrix> tderivglobal,
    std::shared_ptr<Core::LinAlg::SparseMatrix> nderivglobal, bool usePoroLM)
{
  // nothing to do if no active nodes
  if (activenodes_ == nullptr) return;

  if (Interface::n_dim() != 2 && n_dim() != 3) FOUR_C_THROW("Dim() must be either 2 or 3!");

  // loop over all active slave nodes of the interface
  for (int i = 0; i < activenodes_->num_my_elements(); ++i)
  {
    int gid = activenodes_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Node* cnode = dynamic_cast<Node*>(node);

    if (cnode->owner() !=
        Core::Communication::my_mpi_rank(get_comm()))  // move this check into debug?
      FOUR_C_THROW("AssembleTNderiv: Node ownership inconsistency!");

    if (tderivglobal != nullptr)  // assemble tangential derivs?
    {
      std::vector<Core::Gen::Pairedvector<int, double>>& dtmap = cnode->data().get_deriv_txi();
      Core::Gen::Pairedvector<int, double>::iterator colcurr;

      for (int d = 0; d < Interface::n_dim() - 1; ++d)  // for both tangents
      {
        if (d == 1)  // just for 3d case, 2nd tangent
          dtmap = cnode->data().get_deriv_teta();
        int colsize = (int)dtmap[0].size();
        int mapsize = (int)dtmap.size();

        int row = activet_->gid((Interface::n_dim() - 1) * i + d);

        if (Interface::n_dim() == 2 && mapsize == 3) mapsize = 2;  //??

        for (int j = 0; j < mapsize - 1; ++j)  // move this check into debug?
          if ((int)dtmap[j].size() != (int)dtmap[j + 1].size())
          {
            std::cout << "size j = " << dtmap[j].size() << "   size j+1 = " << dtmap[j + 1].size()
                      << std::endl;
            FOUR_C_THROW("AssembleTNderiv: Column dim. of nodal DerivT-map is inconsistent!");
          }

        // begin assembly of Tderiv-matrix
        // std::cout << std::endl << "->Assemble P for Node ID: " << cnode->Id() << std::endl;

        // loop over all derivative maps (=dimensions)
        for (int j = 0; j < mapsize; ++j)
        {
          int k = 0;

          // loop over all entries of the current derivative map
          for (colcurr = dtmap[j].begin(); colcurr != dtmap[j].end(); ++colcurr)
          {
            int col = colcurr->first;
            double val;
            if (!usePoroLM)
              val = cnode->mo_data().lm()[j] * (colcurr->second);
            else
              val = cnode->poro_data().poro_lm()[j] * (colcurr->second);
            // std::cout << "lm[" << j << "]=" << cnode->MoData().lm()[j] << " deriv=" <<
            // colcurr->second << std::endl; std::cout << "Assemble P: " << row << " " << col << " "
            // << val << std::endl;
            // do not assemble zeros into P matrix

            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int i = 0; i < cnode->num_dof(); ++i)
              {
                if (abs(val) > 1.0e-12)
                {
                  double t;
                  if (d == 0)
                    t = cnode->data().txi()[i];
                  else
                    t = cnode->data().teta()[i];
                  tderivglobal->assemble(val * t, cnode->dofs()[i], col);
                }
              }
            }
            else if (abs(val) > 1.0e-12)
              tderivglobal->assemble(val, row, col);
            ++k;
          }

          if (k != colsize) FOUR_C_THROW("AssembleTNderiv: k = {} but colsize = {}", k, colsize);
        }
      }
    }

    if (nderivglobal != nullptr)  // assemble normal derivs?
    {
      std::vector<Core::Gen::Pairedvector<int, double>>& dnmap = cnode->data().get_deriv_n();
      Core::Gen::Pairedvector<int, double>::iterator colcurr;

      int colsize = (int)dnmap[0].size();
      int mapsize = (int)dnmap.size();

      int row = activen_->gid(i);

      if (Interface::n_dim() == 2 && mapsize == 3) mapsize = 2;  //??

      for (int j = 0; j < mapsize - 1; ++j)  // move this check into debug?
        if ((int)dnmap[j].size() != (int)dnmap[j + 1].size())
          FOUR_C_THROW("AssembleTNderiv: Column dim. of nodal DerivN-map is inconsistent!");

      // loop over all derivative maps (=dimensions)
      for (int j = 0; j < mapsize; ++j)
      {
        int k = 0;

        // loop over all entries of the current derivative map
        for (colcurr = dnmap[j].begin(); colcurr != dnmap[j].end(); ++colcurr)
        {
          int col = colcurr->first;
          double val;
          if (!usePoroLM)
            val = cnode->mo_data().lm()[j] * (colcurr->second);
          else
            val = cnode->poro_data().poro_lm()[j] * (colcurr->second);

          // do not assemble zeros into P matrix

          if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
          {
            for (int i = 0; i < cnode->num_dof(); ++i)
              if (abs(val) > 1.0e-12)
                nderivglobal->assemble(val * cnode->mo_data().n()[i], cnode->dofs()[i], col);
          }
          else if (abs(val) > 1.0e-12)
            nderivglobal->assemble(val, row, col);
          ++k;
        }

        if (k != colsize) FOUR_C_THROW("AssembleTNderiv: k = {} but colsize = {}", k, colsize);
      }
    }

  }  // for (int i=0;i<activenodes_->NumMyElements();++i)

  return;
}

/*----------------------------------------------------------------------*
 |  Assemble matrices LinD containing fc derivatives          popp 06/08|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::assemble_lin_d(Core::LinAlg::SparseMatrix& lindglobal, bool usePoroLM)
{
  /**********************************************************************/
  // NEW VERSION (09/2010): No more communication, thanks to FE_MATRIX!
  /**********************************************************************/
  // we have: D_jk,c with j = Lagrange multiplier slave dof
  //                 with k = Displacement slave dof
  //                 with c = Displacement slave or master dof
  // we compute (LinD)_kc = D_jk,c * z_j
  /**********************************************************************/

  // loop over all LM slave nodes (row map)
  for (int j = 0; j < snoderowmap_->num_my_elements(); ++j)
  {
    int gid = snoderowmap_->gid(j);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Node* cnode = dynamic_cast<Node*>(node);
    int dim = cnode->num_dof();

    // Mortar matrix D and M derivatives
    std::map<int, std::map<int, double>>& dderiv = cnode->data().get_deriv_d();

    // current Lagrange multipliers
    double* lm;
    if (!usePoroLM)
      lm = cnode->mo_data().lm();
    else
      lm = cnode->poro_data().poro_lm();

    // get sizes and iterator start
    int slavesize = (int)dderiv.size();
    std::map<int, std::map<int, double>>::iterator scurr = dderiv.begin();

    /********************************************** LinDMatrix **********/
    // loop over all DISP slave nodes in the DerivD-map of the current LM slave node
    for (int k = 0; k < slavesize; ++k)
    {
      int sgid = scurr->first;
      ++scurr;

      Core::Nodes::Node* snode = idiscret_->g_node(sgid);
      if (!snode) FOUR_C_THROW("Cannot find node with gid %", sgid);
      Node* csnode = dynamic_cast<Node*>(snode);

      // Mortar matrix D derivatives
      std::map<int, double>& thisdderiv = cnode->data().get_deriv_d()[sgid];
      int mapsize = (int)(thisdderiv.size());

      // inner product D_{jk,c} * z_j for index j
      for (int prodj = 0; prodj < dim; ++prodj)
      {
        int row = csnode->dofs()[prodj];
        std::map<int, double>::iterator scolcurr = thisdderiv.begin();

        // loop over all directional derivative entries
        for (int c = 0; c < mapsize; ++c)
        {
          int col = scolcurr->first;
          double val = lm[prodj] * (scolcurr->second);
          ++scolcurr;

          // owner of LM slave node can do the assembly, although it actually
          // might not own the corresponding rows in lindglobal (DISP slave node)
          // (FE_MATRIX automatically takes care of non-local assembly inside!!!)
          // std::cout << "Assemble LinD: " << row << " " << col << " " << val << std::endl;
          if (abs(val) > 1.0e-12) lindglobal.fe_assemble(val, row, col);
        }

        // check for completeness of DerivD-Derivatives-iteration
        if (scolcurr != thisdderiv.end())
          FOUR_C_THROW("AssembleLinDM: Not all derivative entries of DerivD considered!");
      }
    }

    // check for completeness of DerivD-Slave-iteration
    if (scurr != dderiv.end())
      FOUR_C_THROW("AssembleLinDM: Not all DISP slave entries of DerivD considered!");
    /******************************** Finished with LinDMatrix **********/
  }

  return;
}


/*----------------------------------------------------------------------*
 |  Assemble matrices LinM containing fc derivatives          popp 06/08|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::assemble_lin_m(Core::LinAlg::SparseMatrix& linmglobal, bool usePoroLM)
{
  /**********************************************************************/
  // NEW VERSION (09/2010): No more communication, thanks to FE_MATRIX!
  /**********************************************************************/
  // we have: M_jl,c with j = Lagrange multiplier slave dof
  //                 with l = Displacement master dof
  //                 with c = Displacement slave or master dof
  // we compute (LinM)_lc = M_jl,c * z_j
  /**********************************************************************/

  // loop over all LM slave nodes (row map)
  for (int j = 0; j < snoderowmap_->num_my_elements(); ++j)
  {
    int gid = snoderowmap_->gid(j);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Node* cnode = dynamic_cast<Node*>(node);
    int dim = cnode->num_dof();

    // Mortar matrix D and M derivatives
    std::map<int, std::map<int, double>>& mderiv = cnode->data().get_deriv_m();

    // current Lagrange multipliers
    double* lm;
    if (!usePoroLM)
      lm = cnode->mo_data().lm();
    else
      lm = cnode->poro_data().poro_lm();

    // get sizes and iterator start
    int mastersize = (int)mderiv.size();
    std::map<int, std::map<int, double>>::iterator mcurr = mderiv.begin();

    /********************************************** LinMMatrix **********/
    // loop over all master nodes in the DerivM-map of the current LM slave node
    for (int l = 0; l < mastersize; ++l)
    {
      int mgid = mcurr->first;
      ++mcurr;

      Core::Nodes::Node* mnode = idiscret_->g_node(mgid);
      if (!mnode) FOUR_C_THROW("Cannot find node with gid %", mgid);
      Node* cmnode = dynamic_cast<Node*>(mnode);

      // Mortar matrix M derivatives
      std::map<int, double>& thismderiv = cnode->data().get_deriv_m()[mgid];
      int mapsize = (int)(thismderiv.size());

      // inner product M_{jl,c} * z_j for index j
      for (int prodj = 0; prodj < dim; ++prodj)
      {
        int row = cmnode->dofs()[prodj];
        std::map<int, double>::iterator mcolcurr = thismderiv.begin();

        // loop over all directional derivative entries
        for (int c = 0; c < mapsize; ++c)
        {
          int col = mcolcurr->first;
          double val = lm[prodj] * (mcolcurr->second);
          ++mcolcurr;

          // owner of LM slave node can do the assembly, although it actually
          // might not own the corresponding rows in lindglobal (DISP slave node)
          // (FE_MATRIX automatically takes care of non-local assembly inside!!!)
          // std::cout << "Assemble LinM: " << row << " " << col << " " << val << std::endl;
          if (abs(val) > 1.0e-12) linmglobal.fe_assemble(-val, row, col);
        }

        // check for completeness of DerivM-Derivatives-iteration
        if (mcolcurr != thismderiv.end())
          FOUR_C_THROW("AssembleLinDM: Not all derivative entries of DerivM considered!");
      }
    }

    // check for completeness of DerivM-Master-iteration
    if (mcurr != mderiv.end())
      FOUR_C_THROW("AssembleLinDM: Not all master entries of DerivM considered!");
    /******************************** Finished with LinMMatrix **********/
  }

  return;
}


/*----------------------------------------------------------------------*
 |  Assemble matrices LinDM containing fc derivatives        farah 02/16|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::assemble_lin_dm(
    Core::LinAlg::SparseMatrix& lindglobal, Core::LinAlg::SparseMatrix& linmglobal, bool usePoroLM)
{
  // call both sub functions
  assemble_lin_d(lindglobal, usePoroLM);
  assemble_lin_m(linmglobal, usePoroLM);

  return;
}


/*----------------------------------------------------------------------*
 |  Assemble normal weighted gap                              popp 01/08|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::assemble_g(Core::LinAlg::Vector<double>& gglobal)
{
  // loop over proc's slave nodes of the interface for assembly
  // use standard row map to assemble each node only once
  for (int i = 0; i < snoderowmap_->num_my_elements(); ++i)
  {
    int gid = snoderowmap_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Node* cnode = dynamic_cast<Node*>(node);

    if (cnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
      FOUR_C_THROW("AssembleG: Node ownership inconsistency!");

    /**************************************************** g-vector ******/
    if (cnode->data().getg() != 0.0)
    {
      double gap = cnode->data().getg();

      // std::cout << "Node ID: " << cnode->Id() << " HasProj: " << cnode->HasProj()
      //      << " is_active: " << cnode->Active() << " Gap: " << gap << std::endl;

      // check if this inactive node has a feasible projection
      // else, it cannot be in contact and weighted gap should be positive
      // (otherwise wrong results possible for g~ because of non-positivity
      // of dual shape functions!!!)
      //******************************************************************
      // TODO: This is only necessary for quadratic LM shape functions!
      // By the way, it makes the method slightly inconsistent
      // (e.g. patch tests with slave side being wider than master side).
      // However, we are able to solve many problems with this little trick.
      // But not all problems, e.g. dropping edge problems would still fail!!!
      // To solve this dilemma, we need a clever modification of the LM shape
      // functions such that they have positive integral values on the
      // "projecting" element part. Once we have this, the following trick
      // can (and should) also be removed in order to make the method
      // consistent again! (08/2013)
      //******************************************************************

      if (!nurbs_)  // only for Lagrange elements
      {
        bool node_has_quad_element = false;
        for (int i = 0; i < cnode->num_element(); i++)
        {
          if (dynamic_cast<Mortar::Element*>(cnode->adjacent_elements()[i].user_element())
                  ->is_quad() == true)
          {
            node_has_quad_element = true;
            break;
          }
        }
        if (!cnode->has_proj() && !cnode->active() && node_has_quad_element)
        {
          gap = 1.0e12;
          cnode->data().getg() = gap;
        }
      }

      if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
      {
        Core::LinAlg::SerialDenseVector gnode(Interface::n_dim());
        std::vector<int> lm(Interface::n_dim());
        std::vector<int> lmowner(Interface::n_dim());
        for (int i = 0; i < Interface::n_dim(); i++)
        {
          gnode(i) = gap * cnode->mo_data().n()[i];
          lm[i] = cnode->dofs()[i];
          lmowner[i] = cnode->owner();
        }
        Core::LinAlg::assemble(gglobal, gnode, lm, lmowner);
      }
      else
      {
        static Core::LinAlg::SerialDenseVector gnode(1);
        static std::vector<int> lm(1);
        static std::vector<int> lmowner(1);

        gnode(0) = gap;
        lm[0] = cnode->id();
        lmowner[0] = cnode->owner();
        Core::LinAlg::assemble(gglobal, gnode, lm, lmowner);
      }
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Assemble inactive right hand side                    hiermeier 08/13|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::assemble_inactiverhs(Core::LinAlg::Vector<double>& inactiverhs)
{
  // FIXME It's possible to improve the performance, if only recently active nodes of the inactive
  // node set, i.e. nodes, which were active in the last iteration, are considered. Since you know,
  // that the lagrange multipliers of former inactive nodes are still equal zero.

  std::shared_ptr<Core::LinAlg::Map> inactivenodes =
      Core::LinAlg::split_map(*snoderowmap_, *activenodes_);
  std::shared_ptr<Core::LinAlg::Map> inactivedofs =
      Core::LinAlg::split_map(*sdofrowmap_, *activedofs_);

  static std::vector<int> lm_gid(Interface::n_dim());
  static std::vector<int> lm_owner(Interface::n_dim());
  static Core::LinAlg::SerialDenseVector lm_i(Interface::n_dim());

  for (int i = 0; i < inactivenodes->num_my_elements(); ++i)
  {
    int gid = inactivenodes->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Node* cnode = dynamic_cast<Node*>(node);

    if (cnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
      FOUR_C_THROW("AssembleInactiverhs: Node ownership inconsistency!");
    if (Interface::n_dim() == 2)
    {
      // calculate the tangential rhs
      for (int j = 0; j < Interface::n_dim(); ++j)
      {
        lm_owner[j] = cnode->owner();
        lm_i[j] = -cnode->mo_data().lm()[j];  // already negative rhs!!!
      }
      lm_gid[0] = inactivedofs->gid(2 * i);
      lm_gid[1] = inactivedofs->gid(2 * i + 1);

      Core::LinAlg::assemble(inactiverhs, lm_i, lm_gid, lm_owner);
    }
    else if (Interface::n_dim() == 3)
    {
      // calculate the tangential rhs
      for (int j = 0; j < Interface::n_dim(); ++j)
      {
        lm_owner[j] = cnode->owner();
        lm_i[j] = -cnode->mo_data().lm()[j];  // already negative rhs!!!
      }
      lm_gid[0] = inactivedofs->gid(3 * i);
      lm_gid[1] = inactivedofs->gid(3 * i + 1);
      lm_gid[2] = inactivedofs->gid(3 * i + 2);

      Core::LinAlg::assemble(inactiverhs, lm_i, lm_gid, lm_owner);
    }
  }
}

/*----------------------------------------------------------------------*
 |  Assemble tangential right-hand side                  hiermeier 08/13|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::assemble_tangrhs(Core::LinAlg::Vector<double>& tangrhs)
{
  static std::vector<int> lm_gid(Interface::n_dim() - 1);
  static std::vector<int> lm_owner(Interface::n_dim() - 1);
  static Core::LinAlg::SerialDenseVector lm_t(Interface::n_dim() - 1);

  for (int i = 0; i < activenodes_->num_my_elements(); ++i)
  {
    int gid = activenodes_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Node* cnode = dynamic_cast<Node*>(node);

    if (cnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
      FOUR_C_THROW("AssembleTangrhs: Node ownership inconsistency!");
    if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
    {
      if (Interface::n_dim() == 2)
      {
        std::vector<int> lm_gid(2);
        std::vector<int> lm_owner(2);
        Core::LinAlg::SerialDenseVector lm_t(2);
        for (int i = 0; i < Interface::n_dim(); ++i)
        {
          lm_gid[i] = cnode->dofs()[i];
          lm_owner[i] = cnode->owner();
          lm_t[i] = 0.;
          for (int j = 0; j < Interface::n_dim(); ++j)
          {
            lm_t[i] -= cnode->data().txi()[i] * cnode->data().txi()[j] * cnode->mo_data().lm()[j];
          }
        }
        Core::LinAlg::assemble(tangrhs, lm_t, lm_gid, lm_owner);
      }
      else if (Interface::n_dim() == 3)
      {
        std::vector<int> lm_gid(3);
        std::vector<int> lm_owner(3);
        Core::LinAlg::SerialDenseVector lm_t(3);

        for (int i = 0; i < Interface::n_dim(); ++i)
        {
          lm_gid[i] = cnode->dofs()[i];
          lm_owner[i] = cnode->owner();
          lm_t[i] = 0.;
          for (int j = 0; j < Interface::n_dim(); ++j)
          {
            lm_t[i] -= cnode->data().txi()[i] * cnode->data().txi()[j] * cnode->mo_data().lm()[j];
            lm_t[i] -= cnode->data().teta()[i] * cnode->data().teta()[j] * cnode->mo_data().lm()[j];
          }
        }
        Core::LinAlg::assemble(tangrhs, lm_t, lm_gid, lm_owner);
      }
    }
    else
    {
      if (Interface::n_dim() == 2)
      {
        lm_gid[0] = activet_->gid(i);
        lm_owner[0] = cnode->owner();

        lm_t[0] = 0.0;
        for (int j = 0; j < Interface::n_dim(); ++j)
          lm_t[0] -= cnode->data().txi()[j] * cnode->mo_data().lm()[j];  // already negative rhs!!!

        Core::LinAlg::assemble(tangrhs, lm_t, lm_gid, lm_owner);
      }
      else if (Interface::n_dim() == 3)
      {
        lm_gid[0] = activet_->gid(2 * i);      // even
        lm_gid[1] = activet_->gid(2 * i + 1);  // odd
        lm_owner[0] = cnode->owner();
        lm_owner[1] = cnode->owner();

        // calculate the tangential rhs
        lm_t[0] = 0.0;
        lm_t[1] = 0.0;
        for (int j = 0; j < Interface::n_dim(); ++j)
        {
          lm_t[0] -= cnode->data().txi()[j] * cnode->mo_data().lm()[j];   // already negative rhs!!!
          lm_t[1] -= cnode->data().teta()[j] * cnode->mo_data().lm()[j];  // already negative rhs!!!
        }
        Core::LinAlg::assemble(tangrhs, lm_t, lm_gid, lm_owner);
      }
    }
  }
}

/*----------------------------------------------------------------------*
 |  Assemble matrix LinStick with tangential+D+M derivatives  mgit 02/09|
 *----------------------------------------------------------------------*/
void CONTACT::Interface::assemble_lin_stick(Core::LinAlg::SparseMatrix& linstickLMglobal,
    Core::LinAlg::SparseMatrix& linstickDISglobal, Core::LinAlg::Vector<double>& linstickRHSglobal)
{
  // FIXGIT: Assemble LinStick is containing a matrix for the de-
  // rivatives of the Lagrange multipliers. This is according to Hueeber.
  // Because of worse convergence, this is not implemented, but the
  // code is commented after the algorithm.

  // create map of stick nodes
  std::shared_ptr<Core::LinAlg::Map> sticknodes =
      Core::LinAlg::split_map(*activenodes_, *slipnodes_);
  std::shared_ptr<Core::LinAlg::Map> stickt = Core::LinAlg::split_map(*activet_, *slipt_);

  // nothing to do if no stick nodes
  if (sticknodes->num_my_elements() == 0) return;

  // information from interface contact parameter list
  const auto ftype =
      Teuchos::getIntegralValue<CONTACT::FrictionType>(interface_params(), "FRICTION");
  double frcoeff_in =
      interface_params().get<double>("FRCOEFF");  // the friction coefficient from the input
  const bool gp_slip = interface_params().get<bool>("GP_SLIP_INCR");
  const bool frilessfirst = interface_params().get<bool>("FRLESS_FIRST");

  double frcoeff = 0.;  // the friction coefficient actually used
  bool consistent = false;

#if defined(CONSISTENTSTICK) && defined(CONSISTENTSLIP)
  FOUR_C_THROW(
      "It's not reasonable to activate both, the consistent stick and slip branch, "
      "because both together will lead again to an inconsistent formulation!");
#endif
#ifdef CONSISTENTSTICK
  consistent = true;
#endif

  if (consistent && interface_params().get<bool>("REGULARIZED_NORMAL_CONTACT"))
    FOUR_C_THROW("no consistent stick for regularized contact");

  // loop over all stick nodes of the interface
  for (int i = 0; i < sticknodes->num_my_elements(); ++i)
  {
    int gid = sticknodes->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    FriNode* cnode = dynamic_cast<FriNode*>(node);

    if (cnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
      FOUR_C_THROW("AssembleLinStick: Node ownership inconsistency!");

    // get friction coefficient for this node
    // in case of TSI, the nodal temperature influences the local friction coefficient
    frcoeff = cnode->fr_coeff(frcoeff_in);

    double cn_input = get_cn_ref().local_values_as_span()[get_cn_ref().get_map().lid(cnode->id())];
    double ct_input = get_ct_ref().local_values_as_span()[get_ct_ref().get_map().lid(cnode->id())];

    // more information from node
    double* n = cnode->mo_data().n();
    double* txi = cnode->data().txi();
    double* teta = cnode->data().teta();
    double* z = cnode->mo_data().lm();

    // evaluation of specific components of entries to assemble
    double znor = 0.0;
    double ztxi = 0.0;
    double zteta = 0.0;

    for (int j = 0; j < Interface::n_dim(); j++)
    {
      znor += n[j] * z[j];
      ztxi += txi[j] * z[j];
      zteta += teta[j] * z[j];
    }

    // prepare assembly, get information from node
    std::vector<Core::Gen::Pairedvector<int, double>> dnmap = cnode->data().get_deriv_n();
    std::vector<Core::Gen::Pairedvector<int, double>> dtximap = cnode->data().get_deriv_txi();
    std::vector<Core::Gen::Pairedvector<int, double>> dtetamap = cnode->data().get_deriv_teta();
    std::map<int, double> dscmap;

    // iterator for maps
    std::map<int, double>::iterator colcurr;
    Core::Gen::Pairedvector<int, double>::iterator _colcurr;

    // row number of entries
    std::vector<int> row(Interface::n_dim() - 1);
    if (Interface::n_dim() == 2)
    {
      row[0] = stickt->gid(i);
    }
    else if (Interface::n_dim() == 3)
    {
      row[0] = stickt->gid(2 * i);
      row[1] = stickt->gid(2 * i) + 1;
    }
    else
      FOUR_C_THROW("AssemblelinStick: Dimension not correct");

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
    // seitz 11/2013
    // In case of a consistent treatment of the stick condition (i.e. using
    // the full linearization as in Diss. Gitterle eq.4.21) the case of
    // a vanishing frictional bound (frcoeff*znor==0) needs to be treated
    // as frictionless contact as well.
    // The FRLESS_FIRST option might end up here, depending on if nodes newly
    // in contact are initialized as stick or slip
    //
    //****************************************************************
    if (frcoeff * znor == 0.0 || (cnode->data().active_old() == false && frilessfirst))
    {
      //**************************************************************
      // calculation of matrix entries of linearized slip condition
      //**************************************************************

      // 1) Entries from differentiation with respect to LM
      /******************************************************************/

      // loop over the dimension
      for (int d = 0; d < cnode->num_dof(); ++d)
      {
        int col = cnode->dofs()[d];
        double valtxi = txi[d];

        double valteta = 0.0;
        if (Interface::n_dim() == 3) valteta = teta[d];

        if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
        {
          for (int j = 0; j < Interface::n_dim(); j++)
          {
            if (abs(valtxi * txi[j]) > 1.e-12)
              linstickLMglobal.assemble(valtxi * txi[j], cnode->dofs()[j], col);
            if (Interface::n_dim() == 3)
              if (abs(valteta * teta[j]) > 1.e-12)
                linstickLMglobal.assemble(valteta * teta[j], cnode->dofs()[j], col);
          }
        }
        else
        {
          if (abs(valtxi) > 1.0e-12) linstickLMglobal.assemble(valtxi, row[0], col);
          if (Interface::n_dim() == 3)
            if (abs(valteta) > 1.0e-12) linstickLMglobal.assemble(valteta, row[1], col);
        }
      }

      // 2) Entries on right hand side
      /******************************************************************/
      if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
      {
        Core::LinAlg::SerialDenseVector rhsnode(Interface::n_dim());
        std::vector<int> lm(Interface::n_dim());
        std::vector<int> lmowner(Interface::n_dim());
        for (int j = 0; j < Interface::n_dim(); j++)
        {
          lm[j] = cnode->dofs()[j];
          lmowner[j] = cnode->owner();
          rhsnode[j] -= ztxi * txi[j];
          if (Interface::n_dim() == 3) rhsnode[j] -= zteta * teta[j];
        }
        Core::LinAlg::assemble(linstickRHSglobal, rhsnode, lm, lmowner);
      }
      else
      {
        Core::LinAlg::SerialDenseVector rhsnode(Interface::n_dim() - 1);
        std::vector<int> lm(Interface::n_dim() - 1);
        std::vector<int> lmowner(Interface::n_dim() - 1);

        lm[0] = cnode->dofs()[1];
        lmowner[0] = cnode->owner();

        rhsnode[0] = -ztxi;  // already negative rhs!!!

        if (Interface::n_dim() == 3)
        {
          rhsnode[1] = -zteta;  // already negative rhs!!!

          lm[1] = cnode->dofs()[2];
          lmowner[1] = cnode->owner();
        }
        Core::LinAlg::assemble(linstickRHSglobal, rhsnode, lm, lmowner);
      }

      // 3) Entries from differentiation with respect to displacements
      /******************************************************************/

      // loop over dimensions
      for (int j = 0; j < Interface::n_dim(); ++j)
      {
        // loop over all entries of the current derivative map (txi)
        for (_colcurr = dtximap[j].begin(); _colcurr != dtximap[j].end(); ++_colcurr)
        {
          int col = _colcurr->first;
          double val = (_colcurr->second) * z[j];

          // do not assemble zeros into matrix
          if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
          {
            for (int j = 0; j < Interface::n_dim(); j++)
              if (abs(val * txi[j]) > 1.e-12)
                linstickDISglobal.assemble(val * txi[j], cnode->dofs()[j], col);
          }
          else if (abs(val) > 1.0e-12)
            linstickDISglobal.assemble(val, row[0], col);
        }

        if (Interface::n_dim() == 3)
        {
          // loop over all entries of the current derivative map (teta)
          for (_colcurr = dtetamap[j].begin(); _colcurr != dtetamap[j].end(); ++_colcurr)
          {
            int col = _colcurr->first;
            double val = (_colcurr->second) * z[j];

            // do not assemble zeros into s matrix
            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(val * teta[j]) > 1.e-12)
                  linstickDISglobal.assemble(val * teta[j], cnode->dofs()[j], col);
            }
            else if (abs(val) > 1.0e-12)
              linstickDISglobal.assemble(val, row[1], col);
          }
        }
      }
    }
    else if (consistent && ftype == CONTACT::FrictionType::coulomb)
    {
      std::map<int, double>& dgmap = cnode->data().get_deriv_g();

      // check for Dimension of derivative maps
      for (int j = 0; j < Interface::n_dim() - 1; ++j)
        if ((int)dnmap[j].size() != (int)dnmap[j + 1].size())
          FOUR_C_THROW("AssembleLinStick: Column dim. of nodal DerivN-map is inconsistent!");

      for (int j = 0; j < Interface::n_dim() - 1; ++j)
        if ((int)dtximap[j].size() != (int)dtximap[j + 1].size())
          FOUR_C_THROW("AssembleLinStick: Column dim. of nodal DerivTxi-map is inconsistent!");

      if (Interface::n_dim() == 3)
      {
        for (int j = 0; j < Interface::n_dim() - 1; ++j)
          if ((int)dtximap[j].size() != (int)dtximap[j + 1].size())
            FOUR_C_THROW("AssembleLinStick: Column dim. of nodal DerivTeta-map is inconsistent!");
      }

      double cn = cn_input;
      double ct = ct_input;

      double& wgap = cnode->data().getg();

      // evaluation of specific components of entries to assemble
      double jumptxi = 0;
      double jumpteta = 0;

      double* jump = cnode->fri_data().jump();

      // choose slip increment
      if (gp_slip)
      {
        jumptxi = cnode->fri_data().jump_var()[0];

        if (Interface::n_dim() == 3) jumpteta = cnode->fri_data().jump_var()[1];
      }
      else
      {
        // more information from node
        for (int i = 0; i < Interface::n_dim(); i++)
        {
          jumptxi += txi[i] * jump[i];
          jumpteta += teta[i] * jump[i];
        }
      }

      // check for dimensions
      if (Interface::n_dim() == 2 and (jumpteta != 0.0))
        FOUR_C_THROW("AssembleLinStick: jumpteta must be zero in 2D");

      //**************************************************************
      // calculation of matrix entries of linearized stick condition
      //**************************************************************

      // 1) Entries from differentiation with respect to LM
      /******************************************************************/

      // loop over the dimension
      for (int d = 0; d < cnode->num_dof(); ++d)
      {
        double valtxi = 0.0;
        double valteta = 0.0;
        int col = cnode->dofs()[d];
        valtxi = -frcoeff * ct * jumptxi * n[d];

        if (Interface::n_dim() == 3)
        {
          valteta = -frcoeff * ct * jumpteta * n[d];
        }

        // do not assemble zeros into matrix
        if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
        {
          for (int j = 0; j < Interface::n_dim(); j++)
          {
            if (abs(valtxi * txi[j]) > 1.e-12)
              linstickLMglobal.assemble(valtxi * txi[j], cnode->dofs()[j], col);
            if (Interface::n_dim() == 3)
              if (abs(valteta * teta[j]) > 1.e-12)
                linstickLMglobal.assemble(valteta * teta[j], cnode->dofs()[j], col);
          }
        }
        else
        {
          if (abs(valtxi) > 1.0e-12) linstickLMglobal.assemble(valtxi, row[0], col);
          if (Interface::n_dim() == 3)
            if (abs(valteta) > 1.0e-12) linstickLMglobal.assemble(valteta, row[1], col);
        }
      }

      // Entries on right hand side ****************************
      if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
      {
        Core::LinAlg::SerialDenseVector rhsnode(Interface::n_dim());
        std::vector<int> lm(Interface::n_dim());
        std::vector<int> lmowner(Interface::n_dim());

        for (int j = 0; j < Interface::n_dim(); j++)
        {
          lm[j] = cnode->dofs()[j];
          lmowner[j] = cnode->owner();
          rhsnode(j) += frcoeff * (znor - cn * wgap) * ct * jumptxi * txi[j];
          if (Interface::n_dim() == 3)
            rhsnode(j) += frcoeff * (znor - cn * wgap) * ct * jumpteta * teta[j];
        }
        Core::LinAlg::assemble(linstickRHSglobal, rhsnode, lm, lmowner);
      }
      else
      {
        Core::LinAlg::SerialDenseVector rhsnode(Interface::n_dim() - 1);
        std::vector<int> lm(Interface::n_dim() - 1);
        std::vector<int> lmowner(Interface::n_dim() - 1);
        rhsnode(0) = frcoeff * (znor - cn * wgap) * ct * jumptxi;

        lm[0] = cnode->dofs()[1];
        lmowner[0] = cnode->owner();
        if (Interface::n_dim() == 3)
        {
          rhsnode(1) = frcoeff * (znor - cn * wgap) * ct * jumpteta;
          lm[1] = cnode->dofs()[2];
          lmowner[1] = cnode->owner();
        }
        Core::LinAlg::assemble(linstickRHSglobal, rhsnode, lm, lmowner);
      }


      // 3) Entries from differentiation with respect to displacements
      /******************************************************************/

      if (gp_slip)
      {
        std::vector<std::map<int, double>> derivjump_ = cnode->fri_data().get_deriv_var_jump();

        // txi
        for (colcurr = derivjump_[0].begin(); colcurr != derivjump_[0].end(); ++colcurr)
        {
          int col = colcurr->first;
          double valtxi = -frcoeff * (znor - cn * wgap) * ct * (colcurr->second);

          if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
          {
            for (int j = 0; j < Interface::n_dim(); j++)
              if (abs(valtxi * txi[j]) > 1.e-12)
                linstickDISglobal.assemble(valtxi * txi[j], cnode->dofs()[j], col);
          }
          else if (abs(valtxi) > 1.0e-12)
            linstickDISglobal.assemble(valtxi, row[0], col);
        }
        // teta
        for (colcurr = derivjump_[1].begin(); colcurr != derivjump_[1].end(); ++colcurr)
        {
          int col = colcurr->first;
          double valteta = -frcoeff * (znor - cn * wgap) * ct * (colcurr->second);

          if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
          {
            for (int j = 0; j < Interface::n_dim(); j++)
              if (abs(valteta * teta[j]) > 1.e-12)
                linstickDISglobal.assemble(valteta * teta[j], cnode->dofs()[j], col);
          }
          else if (abs(valteta) > 1.0e-12)
            linstickDISglobal.assemble(valteta, row[1], col);
        }

        // ... old slip
        // get linearization of jump vector
        std::vector<std::map<int, double>> derivjump = cnode->fri_data().get_deriv_jump();

        // loop over dimensions
        for (int d = 0; d < cnode->num_dof(); ++d)
        {
          // linearization of normal direction *****************************************
          // loop over all entries of the current derivative map
          for (_colcurr = dnmap[d].begin(); _colcurr != dnmap[d].end(); ++_colcurr)
          {
            int col = _colcurr->first;
            double valtxi = 0.0;
            valtxi = -frcoeff * z[d] * _colcurr->second * ct * jumptxi;

            // do not assemble zeros into matrix
            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(valtxi * txi[j]) > 1.e-12)
                  linstickDISglobal.assemble(valtxi * txi[j], cnode->dofs()[j], col);
            }
            else if (abs(valtxi) > 1.0e-12)
              linstickDISglobal.assemble(valtxi, row[0], col);

            if (Interface::n_dim() == 3)
            {
              double valteta = 0.0;
              valteta = -frcoeff * z[d] * _colcurr->second * ct * jumpteta;

              // do not assemble zeros into matrix
              if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
              {
                for (int j = 0; j < Interface::n_dim(); j++)
                  if (abs(valteta * teta[j]) > 1.e-12)
                    linstickDISglobal.assemble(valteta * teta[j], cnode->dofs()[j], col);
              }
              else if (abs(valteta) > 1.0e-12)
                linstickDISglobal.assemble(valteta, row[1], col);
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
        for (int dim = 0; dim < cnode->num_dof(); ++dim)
        {
          // loop over all entries of the current derivative map (jump)
          for (colcurr = derivjump[dim].begin(); colcurr != derivjump[dim].end(); ++colcurr)
          {
            int col = colcurr->first;

            double valtxi = 0.0;
            valtxi = -frcoeff * (znor - cn * wgap) * ct * txi[dim] * (colcurr->second);

            // do not assemble zeros into matrix
            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(valtxi * txi[j]) > 1.e-12)
                  linstickDISglobal.assemble(valtxi * txi[j], cnode->dofs()[j], col);
            }
            else if (abs(valtxi) > 1.0e-12)
              linstickDISglobal.assemble(valtxi, row[0], col);

            if (Interface::n_dim() == 3)
            {
              double valteta = 0.0;
              valteta = -frcoeff * (znor - cn * wgap) * ct * teta[dim] * (colcurr->second);

              // do not assemble zeros into matrix
              if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
              {
                for (int j = 0; j < Interface::n_dim(); j++)
                  if (abs(valteta * teta[j]) > 1.e-12)
                    linstickDISglobal.assemble(valteta * teta[j], cnode->dofs()[j], col);
              }
              else if (abs(valteta) > 1.0e-12)
                linstickDISglobal.assemble(valteta, row[1], col);
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
            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(valtxi * txi[j]) > 1.e-12)
                  linstickDISglobal.assemble(valtxi * txi[j], cnode->dofs()[j], col);
            }
            else if (abs(valtxi) > 1.0e-12)
              linstickDISglobal.assemble(valtxi, row[0], col);
          }
          // linearization second tangential direction *********************************
          if (Interface::n_dim() == 3)
          {
            // loop over all entries of the current derivative map (teta)
            for (_colcurr = dtetamap[dim].begin(); _colcurr != dtetamap[dim].end(); ++_colcurr)
            {
              int col = _colcurr->first;
              double valteta = 0.0;
              valteta = -frcoeff * (znor - cn * wgap) * ct * jump[dim] * _colcurr->second;

              // do not assemble zeros into matrix
              if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
              {
                for (int j = 0; j < Interface::n_dim(); j++)
                  if (abs(valteta * teta[j]) > 1.e-12)
                    linstickDISglobal.assemble(valteta * teta[j], cnode->dofs()[j], col);
              }
              else if (abs(valteta) > 1.0e-12)
                linstickDISglobal.assemble(valteta, row[1], col);
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
            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(valtxi * txi[j]) > 1.e-12)
                  linstickDISglobal.assemble(valtxi * txi[j], cnode->dofs()[j], col);
            }
            else if (abs(valtxi) > 1.0e-12)
              linstickDISglobal.assemble(valtxi, row[0], col);

            if (Interface::n_dim() == 3)
            {
              double valteta = 0.0;
              valteta = -frcoeff * z[dim] * _colcurr->second * ct * jumpteta;

              // do not assemble zeros into matrix
              if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
              {
                for (int j = 0; j < Interface::n_dim(); j++)
                  if (abs(valteta * teta[j]) > 1.e-12)
                    linstickDISglobal.assemble(valteta * teta[j], cnode->dofs()[j], col);
              }
              else if (abs(valteta) > 1.0e-12)
                linstickDISglobal.assemble(valteta, row[1], col);
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
        if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
        {
          for (int j = 0; j < Interface::n_dim(); j++)
            if (abs(valtxi * txi[j]) > 1.e-12)
              linstickDISglobal.assemble(valtxi * txi[j], cnode->dofs()[j], col);
        }
        else if (abs(valtxi) > 1.0e-12)
          linstickDISglobal.assemble(valtxi, row[0], col);

        if (Interface::n_dim() == 3)
        {
          double valteta = 0.0;
          valteta = frcoeff * colcurr->second * ct * cn * jumpteta;

          // do not assemble zeros into matrix
          if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
          {
            for (int j = 0; j < Interface::n_dim(); j++)
              if (abs(valteta * teta[j]) > 1.e-12)
                linstickDISglobal.assemble(valteta * teta[j], cnode->dofs()[j], col);
          }
          else if (abs(valteta) > 1.0e-12)
            linstickDISglobal.assemble(valteta, row[1], col);
        }
      }
    }
    else  // not consistent stick
    {
      for (int j = 0; j < Interface::n_dim() - 1; ++j)
        if ((int)dtximap[j].size() != (int)dtximap[j + 1].size())
          FOUR_C_THROW("AssembleLinStick: Column dim. of nodal DerivTxi-map is inconsistent!");

      if (Interface::n_dim() == 3)
      {
        for (int j = 0; j < Interface::n_dim() - 1; ++j)
          if ((int)dtximap[j].size() != (int)dtximap[j + 1].size())
            FOUR_C_THROW("AssembleLinStick: Column dim. of nodal DerivTeta-map is inconsistent!");
      }

      // evaluation of specific components of entries to assemble
      double jumptxi = 0;
      double jumpteta = 0;

      // more information from node
      double* jump = cnode->fri_data().jump();

      if (gp_slip)
      {
        jumptxi = cnode->fri_data().jump_var()[0];

        if (Interface::n_dim() == 3) jumpteta = cnode->fri_data().jump_var()[1];
      }
      else
      {
        for (int i = 0; i < Interface::n_dim(); i++)
        {
          jumptxi += txi[i] * jump[i];
          jumpteta += teta[i] * jump[i];
        }
      }

      // check for dimensions
      if (Interface::n_dim() == 2 and (jumpteta != 0.0))
        FOUR_C_THROW("AssembleLinStick: jumpteta must be zero in 2D");

      // Entries on right hand side
      /************************************************ (-utxi, -uteta) ***/
      if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
      {
        Core::LinAlg::SerialDenseVector rhsnode(Interface::n_dim());
        std::vector<int> lm(Interface::n_dim());
        std::vector<int> lmowner(Interface::n_dim());
        for (int j = 0; j < Interface::n_dim(); j++)
        {
          lm[j] = cnode->dofs()[j];
          lmowner[j] = cnode->owner();
          rhsnode(j) -= jumptxi * txi[j];
          if (Interface::n_dim() == 3) rhsnode(j) -= jumpteta * teta[j];
        }
        Core::LinAlg::assemble(linstickRHSglobal, rhsnode, lm, lmowner);
      }
      else
      {
        Core::LinAlg::SerialDenseVector rhsnode(Interface::n_dim() - 1);
        std::vector<int> lm(Interface::n_dim() - 1);
        std::vector<int> lmowner(Interface::n_dim() - 1);

        // modification to stabilize the convergence of the lagrange multiplier incr (hiermeier
        // 08/13)
        if (abs(jumptxi) < 1e-15)
          rhsnode(0) = 0.0;
        else
          rhsnode(0) = -jumptxi;

        lm[0] = cnode->dofs()[1];
        lmowner[0] = cnode->owner();

        if (Interface::n_dim() == 3)
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
      }


      // The routine "ApplyDirichlet" in SaddlepointSolve can only set ones on the diagonal
      // if there has already been a diagonal entry in the sparse matrix
      for (int j = 0; j < Interface::n_dim(); j++)
      {
        if (cnode->dbc_dofs()[j] == true)
        {
          linstickLMglobal.assemble(1.e-12, cnode->dofs()[j], cnode->dofs()[j]);
        }
      }

      // Entries from differentiation with respect to displacements
      /*** 1 ************************************** tangent.deriv(jump) ***/
      if (gp_slip)
      {
        std::map<int, double> derivjump1 = cnode->fri_data().get_deriv_var_jump()[0];
        std::map<int, double> derivjump2 = cnode->fri_data().get_deriv_var_jump()[1];

        for (colcurr = derivjump1.begin(); colcurr != derivjump1.end(); ++colcurr)
        {
          int col = colcurr->first;
          double valtxi = colcurr->second;

          if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
          {
            for (int j = 0; j < Interface::n_dim(); j++)
              if (abs(valtxi * txi[j]) > 1.e-12)
                linstickDISglobal.assemble(valtxi * txi[j], cnode->dofs()[j], col);
          }
          else if (abs(valtxi) > 1.0e-12)
            linstickDISglobal.assemble(valtxi, row[0], col);
        }

        if (Interface::n_dim() == 3)
        {
          for (colcurr = derivjump2.begin(); colcurr != derivjump2.end(); ++colcurr)
          {
            int col = colcurr->first;
            double valteta = colcurr->second;

            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(valteta * teta[j]) > 1.e-12)
                  linstickDISglobal.assemble(valteta * teta[j], cnode->dofs()[j], col);
            }
            else if (abs(valteta) > 1.0e-12)
              linstickDISglobal.assemble(valteta, row[1], col);
          }
        }
      }
      else  // std. slip
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
            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(valtxi * txi[j]) > 1.e-12)
                  linstickDISglobal.assemble(valtxi * txi[j], cnode->dofs()[j], col);
            }
            else if (abs(valtxi) > 1.0e-12)
              linstickDISglobal.assemble(valtxi, row[0], col);

            if (Interface::n_dim() == 3)
            {
              double valteta = teta[dim] * colcurr->second;

              if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
              {
                for (int j = 0; j < Interface::n_dim(); j++)
                  if (abs(valteta * teta[j]) > 1.e-12)
                    linstickDISglobal.assemble(valteta * teta[j], cnode->dofs()[j], col);
              }
              else if (abs(valteta) > 1.0e-12)
                linstickDISglobal.assemble(valteta, row[1], col);
            }
          }
        }

        /*** 2 ************************************** deriv(tangent).jump ***/
        // loop over dimensions
        for (int j = 0; j < Interface::n_dim(); ++j)
        {
          // loop over all entries of the current derivative map (txi)
          for (_colcurr = dtximap[j].begin(); _colcurr != dtximap[j].end(); ++_colcurr)
          {
            int col = _colcurr->first;
            double val = jump[j] * _colcurr->second;

            // do not assemble zeros into s matrix
            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(val * txi[j]) > 1.e-12)
                  linstickDISglobal.assemble(val * txi[j], cnode->dofs()[j], col);
            }
            else if (abs(val) > 1.0e-12)
              linstickDISglobal.assemble(val, row[0], col);
          }

          if (Interface::n_dim() == 3)
          {
            // loop over all entries of the current derivative map (teta)
            for (_colcurr = dtetamap[j].begin(); _colcurr != dtetamap[j].end(); ++_colcurr)
            {
              int col = _colcurr->first;
              double val = jump[j] * _colcurr->second;

              // do not assemble zeros into matrix
              if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
              {
                for (int j = 0; j < Interface::n_dim(); j++)
                  if (abs(val * teta[j]) > 1.e-12)
                    linstickDISglobal.assemble(val * teta[j], cnode->dofs()[j], col);
              }
              else if (abs(val) > 1.0e-12)
                linstickDISglobal.assemble(val, row[1], col);
            }
          }
        }
      }
    }
  }
  return;
}

/*---------------------------------------------------------------------*
 | Assemble matrix LinSlip with tangential+D+M derivatives  mgit 02/09 |
 *---------------------------------------------------------------------*/
void CONTACT::Interface::assemble_lin_slip(Core::LinAlg::SparseMatrix& linslipLMglobal,
    Core::LinAlg::SparseMatrix& linslipDISglobal, Core::LinAlg::Vector<double>& linslipRHSglobal)
{
  // nothing to do if no slip nodes
  if (slipnodes_->num_my_elements() == 0) return;

  // information from interface contact parameter list
  auto ftype = Teuchos::getIntegralValue<CONTACT::FrictionType>(interface_params(), "FRICTION");
  double frbound = interface_params().get<double>("FRBOUND");
  double frcoeff_in =
      interface_params().get<double>("FRCOEFF");  // the friction coefficient from the input
  const bool gp_slip = interface_params().get<bool>("GP_SLIP_INCR");
  const bool frilessfirst = interface_params().get<bool>("FRLESS_FIRST");

  // the friction coefficient adapted by every node (eg depending on the local temperature)
  double frcoeff = 0.;

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
      FriNode* cnode = dynamic_cast<FriNode*>(node);

      if (cnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
        FOUR_C_THROW("AssembleLinSlip: Node ownership inconsistency!");

      // get friction coefficient for this node
      // in case of TSI, the nodal temperature influences the local friction coefficient
      frcoeff = cnode->fr_coeff(frcoeff_in);

      double cn_input =
          get_cn_ref().local_values_as_span()[get_cn_ref().get_map().lid(cnode->id())];
      double ct_input =
          get_ct_ref().local_values_as_span()[get_ct_ref().get_map().lid(cnode->id())];

      // prepare assembly, get information from node
      std::vector<Core::Gen::Pairedvector<int, double>> dnmap = cnode->data().get_deriv_n();
      std::vector<Core::Gen::Pairedvector<int, double>> dtximap = cnode->data().get_deriv_txi();
      std::vector<Core::Gen::Pairedvector<int, double>> dtetamap = cnode->data().get_deriv_teta();

      double cn = cn_input;
      double ct = ct_input;

      // check for Dimension of derivative maps
      for (int j = 0; j < Interface::n_dim() - 1; ++j)
        if ((int)dnmap[j].size() != (int)dnmap[j + 1].size())
          FOUR_C_THROW("AssembleLinSlip: Column dim. of nodal DerivTxi-map is inconsistent!");

      for (int j = 0; j < Interface::n_dim() - 1; ++j)
        if ((int)dtximap[j].size() != (int)dtximap[j + 1].size())
          FOUR_C_THROW("AssembleLinSlip: Column dim. of nodal DerivTxi-map is inconsistent!");

      if (Interface::n_dim() == 3)
      {
        for (int j = 0; j < Interface::n_dim() - 1; ++j)
          if ((int)dtximap[j].size() != (int)dtximap[j + 1].size())
            FOUR_C_THROW("AssembleLinSlip: Column dim. of nodal DerivTeta-map is inconsistent!");
      }

      // more information from node
      double* n = cnode->mo_data().n();
      double* txi = cnode->data().txi();
      double* teta = cnode->data().teta();
      double* z = cnode->mo_data().lm();
      double wgap = cnode->data().getg();

      // iterator for maps
      Core::Gen::Pairedvector<int, double>::iterator colcurr;
      std::map<int, double>::iterator _colcurr;

      // row number of entries
      std::vector<int> row(Interface::n_dim() - 1);
      if (Interface::n_dim() == 2)
      {
        row[0] = slipt_->gid(i);
      }
      else if (Interface::n_dim() == 3)
      {
        row[0] = slipt_->gid(2 * i);
        row[1] = slipt_->gid(2 * i) + 1;
      }
      else
        FOUR_C_THROW("AssemblelinSlip: Dimension not correct");

      // evaluation of specific components of entries to assemble
      double znor = 0.0;
      double ztxi = 0.0;
      double zteta = 0.0;
      double jumptxi = 0.0;
      double jumpteta = 0.0;
      double euclidean = 0.0;
      double* jump = cnode->fri_data().jump();

      if (gp_slip)
      {
        jumptxi = cnode->fri_data().jump_var()[0];

        if (Interface::n_dim() == 3) jumpteta = cnode->fri_data().jump_var()[1];

        for (int i = 0; i < Interface::n_dim(); i++)
        {
          znor += n[i] * z[i];
          ztxi += txi[i] * z[i];
          zteta += teta[i] * z[i];
        }
      }
      else  // std. slip
      {
        for (int i = 0; i < Interface::n_dim(); i++)
        {
          znor += n[i] * z[i];
          ztxi += txi[i] * z[i];
          zteta += teta[i] * z[i];
          jumptxi += txi[i] * jump[i];
          jumpteta += teta[i] * jump[i];
        }
      }

      // evaluate euclidean norm ||vec(zt)+ct*vec(jump)||
      std::vector<double> sum1(Interface::n_dim() - 1, 0);
      sum1[0] = ztxi + ct * jumptxi;
      if (Interface::n_dim() == 3) sum1[1] = zteta + ct * jumpteta;
      if (Interface::n_dim() == 2) euclidean = abs(sum1[0]);
      if (Interface::n_dim() == 3) euclidean = sqrt(sum1[0] * sum1[0] + sum1[1] * sum1[1]);

      // check of dimensions
      if (Interface::n_dim() == 2 and (zteta != 0.0 or jumpteta != 0.0))
        FOUR_C_THROW("AssemblelinSlip: zteta and jumpteta must be zero in 2D");

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
      // seitz11/13
      // Like that FRLESS_FIRST results in a frictionless constraint.
      // Nodes with a vanishing slip and tangential Lagrange multiplier
      // are treated as frictionless as well.
      //
      //****************************************************************
      if (frcoeff == 0.0 || (cnode->data().active_old() == false && frilessfirst) ||
          euclidean == 0.0)
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
          if (Interface::n_dim() == 3) valteta = teta[dim];

          // do not assemble zeros into matrix
          if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
          {
            for (int j = 0; j < Interface::n_dim(); j++)
            {
              if (abs(valtxi * txi[j]) > 1.e-12)
                linslipLMglobal.assemble(valtxi * txi[j], cnode->dofs()[j], col);
              if (Interface::n_dim() == 3)
                if (abs(valteta * teta[j]) > 1.e-12)
                  linslipLMglobal.assemble(valteta * teta[j], cnode->dofs()[j], col);
            }
          }
          else
          {
            if (abs(valtxi) > 1.0e-12) linslipLMglobal.assemble(valtxi, row[0], col);
            if (Interface::n_dim() == 3)
              if (abs(valteta) > 1.0e-12) linslipLMglobal.assemble(valteta, row[1], col);
          }
        }

        // 2) Entries on right hand side
        /******************************************************************/
        if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
        {
          Core::LinAlg::SerialDenseVector rhsnode(Interface::n_dim());
          std::vector<int> lm(Interface::n_dim());
          std::vector<int> lmowner(Interface::n_dim());
          for (int j = 0; j < Interface::n_dim(); j++)
          {
            lm[j] = cnode->dofs()[j];
            lmowner[j] = cnode->owner();
            rhsnode[j] -= ztxi * txi[j];
            if (Interface::n_dim() == 3) rhsnode[j] -= zteta * teta[j];
          }
          Core::LinAlg::assemble(linslipRHSglobal, rhsnode, lm, lmowner);
        }
        else
        {
          Core::LinAlg::SerialDenseVector rhsnode(Interface::n_dim() - 1);
          std::vector<int> lm(Interface::n_dim() - 1);
          std::vector<int> lmowner(Interface::n_dim() - 1);

          lm[0] = cnode->dofs()[1];
          lmowner[0] = cnode->owner();

          rhsnode[0] = -ztxi;  // already negative rhs!!!

          if (Interface::n_dim() == 3)
          {
            rhsnode[1] = -zteta;  // already negative rhs!!!

            lm[1] = cnode->dofs()[2];
            lmowner[1] = cnode->owner();
          }
          Core::LinAlg::assemble(linslipRHSglobal, rhsnode, lm, lmowner);
        }

        // 3) Entries from differentiation with respect to displacements
        /******************************************************************/

        // loop over dimensions
        for (int j = 0; j < Interface::n_dim(); ++j)
        {
          // loop over all entries of the current derivative map (txi)
          for (colcurr = dtximap[j].begin(); colcurr != dtximap[j].end(); ++colcurr)
          {
            int col = colcurr->first;
            double val = (colcurr->second) * z[j];

            // do not assemble zeros into matrix
            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(val * txi[j]) > 1.e-12)
                  linslipDISglobal.assemble(val * txi[j], cnode->dofs()[j], col);
            }
            else if (abs(val) > 1.0e-12)
              linslipDISglobal.assemble(val, row[0], col);
          }

          if (Interface::n_dim() == 3)
          {
            // loop over all entries of the current derivative map (teta)
            for (colcurr = dtetamap[j].begin(); colcurr != dtetamap[j].end(); ++colcurr)
            {
              int col = colcurr->first;
              double val = (colcurr->second) * z[j];

              // do not assemble zeros into s matrix
              if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
              {
                for (int j = 0; j < Interface::n_dim(); j++)
                  if (abs(val * teta[j]) > 1.e-12)
                    linslipDISglobal.assemble(val * teta[j], cnode->dofs()[j], col);
              }
              else if (abs(val) > 1.0e-12)
                linslipDISglobal.assemble(val, row[1], col);
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
          if (Interface::n_dim() == 3)
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
          if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
          {
            for (int j = 0; j < Interface::n_dim(); j++)
            {
              if (abs(valtxi * txi[j]) > 1.e-12)
                linslipLMglobal.assemble(valtxi * txi[j], cnode->dofs()[j], col);
              if (Interface::n_dim() == 3)
                if (abs(valteta * teta[j]) > 1.e-12)
                  linslipLMglobal.assemble(valteta * teta[j], cnode->dofs()[j], col);
            }
          }
          else
          {
            if (abs(valtxi) > 1.0e-12) linslipLMglobal.assemble(valtxi, row[0], col);
            if (Interface::n_dim() == 3)
              if (abs(valteta) > 1.0e-12) linslipLMglobal.assemble(valteta, row[1], col);
          }
        }

        // 2) Entries on right hand side
        /******************************************************************/
        if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
        {
          Core::LinAlg::SerialDenseVector rhsnode(Interface::n_dim());
          std::vector<int> lm(Interface::n_dim());
          std::vector<int> lmowner(Interface::n_dim());

#ifdef CONSISTENTSLIP
          double valuetxi1 =
              -(euclidean)*ztxi / (znor - cn * wgap) + frcoeff * (ztxi + ct * jumptxi);
#else
          double valuetxi1 =
              -(euclidean)*ztxi + (frcoeff * (znor - cn * wgap)) * (ztxi + ct * jumptxi);
#endif

          for (int j = 0; j < Interface::n_dim(); j++)
          {
            lm[j] = cnode->dofs()[j];
            lmowner[j] = cnode->owner();
            rhsnode(j) += valuetxi1 * txi[j];
          }

          if (Interface::n_dim() == 3)
          {
#ifdef CONSISTENTSLIP
            double valueteta1 =
                -(euclidean)*zteta / (znor - cn * wgap) + frcoeff * (zteta + ct * jumpteta);
#else
            double valueteta1 =
                -(euclidean)*zteta + (frcoeff * (znor - cn * wgap)) * (zteta + ct * jumpteta);
#endif

            for (int j = 0; j < Interface::n_dim(); j++) rhsnode(j) += valueteta1 * teta[j];
          }
          Core::LinAlg::assemble(linslipRHSglobal, rhsnode, lm, lmowner);
        }
        else
        {
          Core::LinAlg::SerialDenseVector rhsnode(Interface::n_dim() - 1);
          std::vector<int> lm(Interface::n_dim() - 1);
          std::vector<int> lmowner(Interface::n_dim() - 1);
#ifdef CONSISTENTSLIP
          double valuetxi1 =
              -(euclidean)*ztxi / (znor - cn * wgap) + frcoeff * (ztxi + ct * jumptxi);
#else
          double valuetxi1 =
              -(euclidean)*ztxi + (frcoeff * (znor - cn * wgap)) * (ztxi + ct * jumptxi);
#endif
          rhsnode(0) = valuetxi1;
          lm[0] = cnode->dofs()[1];
          lmowner[0] = cnode->owner();

          if (Interface::n_dim() == 3)
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
        }

        // 3) Entries from differentiation with respect to displacements
        /******************************************************************/
        std::map<int, double> derivjump1, derivjump2;  // for gp slip
        std::vector<std::map<int, double>> derivjump;  // for dm slip

        /*** 01  ********* -Deriv(euclidean).ct.tangent.deriv(u)*ztan ***/
        if (gp_slip)
        {
          derivjump1 = cnode->fri_data().get_deriv_var_jump()[0];
          derivjump2 = cnode->fri_data().get_deriv_var_jump()[1];

          for (_colcurr = derivjump1.begin(); _colcurr != derivjump1.end(); ++_colcurr)
          {
            int col = _colcurr->first;
            double valtxi1 = (ztxi + ct * jumptxi) / euclidean * ct * _colcurr->second * ztxi;
            double valteta1 = (ztxi + ct * jumptxi) / euclidean * ct * _colcurr->second * zteta;

            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
              {
                if (abs(valtxi1 * txi[j]) > 1.e-12)
                  linslipDISglobal.assemble(valtxi1 * txi[j], cnode->dofs()[j], col);
                if (abs(valteta1 * teta[j]) > 1.e-12)
                  linslipDISglobal.assemble(valteta1 * teta[j], cnode->dofs()[j], col);
              }
            }
            else
            {
              if (abs(valtxi1) > 1.0e-12) linslipDISglobal.assemble(valtxi1, row[0], col);
              if (abs(valteta1) > 1.0e-12) linslipDISglobal.assemble(valteta1, row[1], col);
            }
          }

          if (Interface::n_dim() == 3)
          {
            for (_colcurr = derivjump2.begin(); _colcurr != derivjump2.end(); ++_colcurr)
            {
              int col = _colcurr->first;
              double valtxi2 = (zteta + ct * jumpteta) / euclidean * ct * _colcurr->second * ztxi;
              double valteta2 = (zteta + ct * jumpteta) / euclidean * ct * _colcurr->second * zteta;

              if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
              {
                for (int j = 0; j < Interface::n_dim(); j++)
                {
                  if (abs(valtxi2 * txi[j]) > 1.e-12)
                    linslipDISglobal.assemble(valtxi2 * txi[j], cnode->dofs()[j], col);
                  if (abs(valteta2 * teta[j]) > 1.e-12)
                    linslipDISglobal.assemble(valteta2 * teta[j], cnode->dofs()[j], col);
                }
              }
              else
              {
                if (abs(valtxi2) > 1.0e-12) linslipDISglobal.assemble(valtxi2, row[0], col);
                if (abs(valteta2) > 1.0e-12) linslipDISglobal.assemble(valteta2, row[1], col);
              }
            }
          }
        }
        else  // std. slip increment
        {
          // get linearization of jump vector
          derivjump = cnode->fri_data().get_deriv_jump();

          // loop over dimensions
          for (int dim = 0; dim < cnode->num_dof(); ++dim)
          {
            // loop over all entries of the current derivative map (jump)
            for (_colcurr = derivjump[dim].begin(); _colcurr != derivjump[dim].end(); ++_colcurr)
            {
              int col = _colcurr->first;

              double valtxi1 =
                  (ztxi + ct * jumptxi) / euclidean * ct * txi[dim] * _colcurr->second * ztxi;
              double valteta1 =
                  (ztxi + ct * jumptxi) / euclidean * ct * txi[dim] * _colcurr->second * zteta;
              double valtxi2 =
                  (zteta + ct * jumpteta) / euclidean * ct * teta[dim] * _colcurr->second * ztxi;
              double valteta2 =
                  (zteta + ct * jumpteta) / euclidean * ct * teta[dim] * _colcurr->second * zteta;

#ifdef CONSISTENTSLIP
              valtxi1 = valtxi1 / (znor - cn * wgap);
              valteta1 = valteta1 / (znor - cn * wgap);
              valtxi2 = valtxi2 / (znor - cn * wgap);
              valteta2 = valteta2 / (znor - cn * wgap);
#endif

              // do not assemble zeros into matrix
              if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
              {
                for (int j = 0; j < Interface::n_dim(); j++)
                {
                  if (abs((valtxi1 + valtxi2) * txi[j]) > 1.e-12)
                    linslipDISglobal.assemble((valtxi1 + valtxi2) * txi[j], cnode->dofs()[j], col);
                  if (Interface::n_dim() == 3)
                    if (abs((valteta1 + valteta2) * teta[j]) > 1.e-12)
                      linslipDISglobal.assemble(
                          (valteta1 + valteta2) * teta[j], cnode->dofs()[j], col);
                }
              }
              else
              {
                if (abs(valtxi1) > 1.0e-12) linslipDISglobal.assemble(valtxi1, row[0], col);
                if (abs(valteta1) > 1.0e-12) linslipDISglobal.assemble(valteta1, row[1], col);
                if (abs(valtxi2) > 1.0e-12) linslipDISglobal.assemble(valtxi2, row[0], col);
                if (abs(valteta2) > 1.0e-12) linslipDISglobal.assemble(valteta2, row[1], col);
              }
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
              if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
              {
                for (int j = 0; j < Dim(); j++)
                {
                  if (abs(valtxi * txi[j]) > 1.e-12)
                    linslipDISglobal.Assemble(valtxi * txi[j], cnode->Dofs()[j], col);
                  if (Dim() == 3)
                    if (abs(valteta * teta[j]) > 1.e-12)
                      linslipDISglobal.Assemble(valteta * teta[j], cnode->Dofs()[j], col);
                }
              }
              else
              {
                if (abs(valtxi) > 1.0e-12) linslipDISglobal.Assemble(valtxi, row[0], col);
                if (abs(valteta) > 1.0e-12) linslipDISglobal.Assemble(valteta, row[1], col);
              }
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
          if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
          {
            for (int j = 0; j < Dim(); j++)
            {
              if (abs(valtxi * txi[j]) > 1.e-12)
                linslipDISglobal.Assemble(valtxi * txi[j], cnode->Dofs()[j], col);
              if (Dim() == 3)
                if (abs(valteta * teta[j]) > 1.e-12)
                  linslipDISglobal.Assemble(valteta * teta[j], cnode->Dofs()[j], col);
            }
          }
          else
          {
            if (abs(valtxi) > 1.0e-12) linslipDISglobal.Assemble(valtxi, row[0], col);
            if (abs(valteta) > 1.0e-12) linslipDISglobal.Assemble(valteta, row[1], col);
          }
        }
#endif


        /*** 02 ***************** frcoeff*znor*ct*tangent.deriv(jump) ***/
        if (gp_slip)
        {
          for (_colcurr = derivjump1.begin(); _colcurr != derivjump1.end(); ++_colcurr)
          {
            int col = _colcurr->first;
            double valtxi = -frcoeff * (znor - cn * wgap) * ct * _colcurr->second;
            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(valtxi * txi[j]) > 1.e-12)
                  linslipDISglobal.assemble(valtxi * txi[j], cnode->dofs()[j], col);
            }
            else if (abs(valtxi) > 1.0e-12)
              linslipDISglobal.assemble(valtxi, row[0], col);
          }

          if (Interface::n_dim() == 3)
          {
            for (_colcurr = derivjump2.begin(); _colcurr != derivjump2.end(); ++_colcurr)
            {
              int col = _colcurr->first;
              double valteta = -frcoeff * (znor - cn * wgap) * ct * _colcurr->second;

              if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
              {
                for (int j = 0; j < Interface::n_dim(); j++)
                  if (abs(valteta * teta[j]) > 1.e-12)
                    linslipDISglobal.assemble(valteta * teta[j], cnode->dofs()[j], col);
              }
              else if (abs(valteta) > 1.0e-12)
                linslipDISglobal.assemble(valteta, row[1], col);
            }
          }
        }
        else
        {
          // loop over dimensions
          for (int dim = 0; dim < cnode->num_dof(); ++dim)
          {
            // loop over all entries of the current derivative map (jump)
            for (_colcurr = derivjump[dim].begin(); _colcurr != derivjump[dim].end(); ++_colcurr)
            {
              int col = _colcurr->first;

              // std::cout << "val " << colcurr->second << std::endl;
#ifdef CONSISTENTSLIP
              double valtxi = -frcoeff * ct * txi[dim] * colcurr->second;
              double valteta = -frcoeff * ct * teta[dim] * colcurr->second;
#else
              double valtxi =
                  (-1) * (frcoeff * (znor - cn * wgap)) * ct * txi[dim] * _colcurr->second;
              double valteta =
                  (-1) * (frcoeff * (znor - cn * wgap)) * ct * teta[dim] * _colcurr->second;
#endif
              // do not assemble zeros into matrix
              if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
              {
                for (int j = 0; j < Interface::n_dim(); j++)
                  if (abs(valtxi * txi[j]) > 1.e-12)
                    linslipDISglobal.assemble(valtxi * txi[j], cnode->dofs()[j], col);
              }
              else if (abs(valtxi) > 1.0e-12)
                linslipDISglobal.assemble(valtxi, row[0], col);

              if (Interface::n_dim() == 3)
              {
                if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
                {
                  for (int j = 0; j < Interface::n_dim(); j++)
                    if (abs(valteta * teta[j]) > 1.e-12)
                      linslipDISglobal.assemble(valteta * teta[j], cnode->dofs()[j], col);
                }
                else if (abs(valteta) > 1.0e-12)
                  linslipDISglobal.assemble(valteta, row[1], col);
              }
            }
          }
        }
        /*** 1 ********************************* euclidean.deriv(T).z ***/
        // loop over dimensions
        for (int j = 0; j < Interface::n_dim(); ++j)
        {
          // loop over all entries of the current derivative map (txi)
          for (colcurr = dtximap[j].begin(); colcurr != dtximap[j].end(); ++colcurr)
          {
            int col = colcurr->first;
            double val = euclidean * (colcurr->second) * z[j];

#ifdef CONSISTENTSLIP
            val = val / (znor - cn * wgap);
#endif

            // do not assemble zeros into s matrix
            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(val * txi[j]) > 1.e-12)
                  linslipDISglobal.assemble(val * txi[j], cnode->dofs()[j], col);
            }
            else if (abs(val) > 1.0e-12)
              linslipDISglobal.assemble(val, row[0], col);
          }

          if (Interface::n_dim() == 3)
          {
            // loop over all entries of the current derivative map (teta)
            for (colcurr = dtetamap[j].begin(); colcurr != dtetamap[j].end(); ++colcurr)
            {
              int col = colcurr->first;
              double val = euclidean * (colcurr->second) * z[j];

#ifdef CONSISTENTSLIP
              val = val / (znor - cn * wgap);
#endif

              // do not assemble zeros into s matrix
              if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
              {
                for (int j = 0; j < Interface::n_dim(); j++)
                  if (abs(val * teta[j]) > 1.e-12)
                    linslipDISglobal.assemble(val * teta[j], cnode->dofs()[j], col);
              }
              else if (abs(val) > 1.0e-12)
                linslipDISglobal.assemble(val, row[1], col);
            }
          }
        }

        /*** 2 ********************* deriv(euclidean).deriv(T).z.ztan ***/
        // loop over dimensions
        for (int j = 0; j < Interface::n_dim(); ++j)
        {
          // loop over all entries of the current derivative map (txi)
          for (colcurr = dtximap[j].begin(); colcurr != dtximap[j].end(); ++colcurr)
          {
            int col = colcurr->first;
            double valtxi = (ztxi + ct * jumptxi) / euclidean * (colcurr->second) * z[j] * ztxi;
            double valteta = (ztxi + ct * jumptxi) / euclidean * (colcurr->second) * z[j] * zteta;

#ifdef CONSISTENTSLIP
            valtxi = valtxi / (znor - cn * wgap);
            valteta = valteta / (znor - cn * wgap);
#endif

            // do not assemble zeros into matrix
            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(valtxi * txi[j]) > 1.e-12)
                  linslipDISglobal.assemble(valtxi * txi[j], cnode->dofs()[j], col);
            }
            else if (abs(valtxi) > 1.0e-12)
              linslipDISglobal.assemble(valtxi, row[0], col);

            if (Interface::n_dim() == 3)
            {
              if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
              {
                for (int j = 0; j < Interface::n_dim(); j++)
                  if (abs(valteta * teta[j]) > 1.e-12)
                    linslipDISglobal.assemble(valteta * teta[j], cnode->dofs()[j], col);
              }
              else if (abs(valteta) > 1.0e-12)
                linslipDISglobal.assemble(valteta, row[1], col);
            }
          }

          if (Interface::n_dim() == 3)
          {
            // 3D loop over all entries of the current derivative map (teta)
            for (colcurr = dtetamap[j].begin(); colcurr != dtetamap[j].end(); ++colcurr)
            {
              int col = colcurr->first;
              double valtxi = (zteta + ct * jumpteta) / euclidean * (colcurr->second) * z[j] * ztxi;
              double valteta =
                  (zteta + ct * jumpteta) / euclidean * (colcurr->second) * z[j] * zteta;

#ifdef CONSISTENTSLIP
              valtxi = valtxi / (znor - cn * wgap);
              valteta = valteta / (znor - cn * wgap);
#endif

              // do not assemble zeros into matrix
              if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
              {
                for (int j = 0; j < Interface::n_dim(); j++)
                  if (abs(valtxi * txi[j]) > 1.e-12)
                    linslipDISglobal.assemble(valtxi * txi[j], cnode->dofs()[j], col);
              }
              else if (abs(valtxi) > 1.0e-12)
                linslipDISglobal.assemble(valtxi, row[0], col);

              if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
              {
                for (int j = 0; j < Interface::n_dim(); j++)
                  if (abs(valteta * teta[j]) > 1.e-12)
                    linslipDISglobal.assemble(valteta * teta[j], cnode->dofs()[j], col);
              }
              else if (abs(valteta) > 1.0e-12)
                linslipDISglobal.assemble(valteta, row[1], col);
            }
          }
        }

        /*** 3 ****************** deriv(euclidean).deriv(T).jump.ztan ***/
        if (gp_slip)
        {
          //!!!!!!!!!!!!!!! DO NOTHING !!!!!!!
        }
        else
        {
          // loop over dimensions
          for (int j = 0; j < Interface::n_dim(); ++j)
          {
            // loop over all entries of the current derivative map (txi)
            for (colcurr = dtximap[j].begin(); colcurr != dtximap[j].end(); ++colcurr)
            {
              int col = colcurr->first;
              double valtxi =
                  (ztxi + ct * jumptxi) / euclidean * ct * (colcurr->second) * jump[j] * ztxi;
              double valteta =
                  (ztxi + ct * jumptxi) / euclidean * ct * (colcurr->second) * jump[j] * zteta;

#ifdef CONSISTENTSLIP
              valtxi = valtxi / (znor - cn * wgap);
              valteta = valteta / (znor - cn * wgap);
#endif

              // do not assemble zeros into s matrix
              if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
              {
                for (int j = 0; j < Interface::n_dim(); j++)
                  if (abs(valtxi * txi[j]) > 1.e-12)
                    linslipDISglobal.assemble(valtxi * txi[j], cnode->dofs()[j], col);
                for (int j = 0; j < Interface::n_dim(); j++)
                  if (abs(valteta * teta[j]) > 1.e-12)
                    linslipDISglobal.assemble(valteta * teta[j], cnode->dofs()[j], col);
              }
              else
              {
                if (abs(valtxi) > 1.0e-12) linslipDISglobal.assemble(valtxi, row[0], col);
                if (abs(valteta) > 1.0e-12) linslipDISglobal.assemble(valteta, row[1], col);
              }
            }

            if (Interface::n_dim() == 3)
            {
              // loop over all entries of the current derivative map (teta)
              for (colcurr = dtetamap[j].begin(); colcurr != dtetamap[j].end(); ++colcurr)
              {
                int col = colcurr->first;
                double valtxi =
                    (zteta + ct * jumpteta) / euclidean * ct * (colcurr->second) * jump[j] * ztxi;
                double valteta =
                    (zteta + ct * jumpteta) / euclidean * ct * (colcurr->second) * jump[j] * zteta;

#ifdef CONSISTENTSLIP
                valtxi = valtxi / (znor - cn * wgap);
                valteta = valteta / (znor - cn * wgap);
#endif

                // do not assemble zeros into matrix
                if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
                {
                  for (int j = 0; j < Interface::n_dim(); j++)
                    if (abs(valtxi * txi[j]) > 1.e-12)
                      linslipDISglobal.assemble(valtxi * txi[j], cnode->dofs()[j], col);
                  for (int j = 0; j < Interface::n_dim(); j++)
                    if (abs(valteta * teta[j]) > 1.e-12)
                      linslipDISglobal.assemble(valteta * teta[j], cnode->dofs()[j], col);
                }
                else
                {
                  if (abs(valtxi) > 1.0e-12) linslipDISglobal.assemble(valtxi, row[0], col);
                  if (abs(valteta) > 1.0e-12) linslipDISglobal.assemble(valteta, row[1], col);
                }
              }
            }
          }
        }
        /*** 4 ************************** (frcoeff*znor).deriv(T).z ***/
        // loop over all dimensions
        for (int j = 0; j < Interface::n_dim(); ++j)
        {
          // loop over all entries of the current derivative map (txi)
          for (colcurr = dtximap[j].begin(); colcurr != dtximap[j].end(); ++colcurr)
          {
            int col = colcurr->first;
#ifdef CONSISTENTSLIP
            double val = -frcoeff * (colcurr->second) * z[j];
#else
            double val = (-1) * (frcoeff * (znor - cn * wgap)) * (colcurr->second) * z[j];
#endif
            // do not assemble zeros into matrix
            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(val * txi[j]) > 1.e-12)
                  linslipDISglobal.assemble(val * txi[j], cnode->dofs()[j], col);
            }
            else if (abs(val) > 1.0e-12)
              linslipDISglobal.assemble(val, row[0], col);
          }

          if (Interface::n_dim() == 3)
          {
            // loop over all entries of the current derivative map (teta)
            for (colcurr = dtetamap[j].begin(); colcurr != dtetamap[j].end(); ++colcurr)
            {
              int col = colcurr->first;
#ifdef CONSISTENTSLIP
              double val = -frcoeff * (colcurr->second) * z[j];
#else
              double val = (-1.0) * (frcoeff * (znor - cn * wgap)) * (colcurr->second) * z[j];
#endif
              // do not assemble zeros into matrix
              if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
              {
                for (int j = 0; j < Interface::n_dim(); j++)
                  if (abs(val * teta[j]) > 1.e-12)
                    linslipDISglobal.assemble(val * teta[j], cnode->dofs()[j], col);
              }
              else if (abs(val) > 1.0e-12)
                linslipDISglobal.assemble(val, row[1], col);
            }
          }
        }

        /*** 5 *********************** (frcoeff*znor).deriv(T).jump ***/
        if (gp_slip)
        {
          //!!!!!!!!!!!!!!! DO NOTHING !!!!!!!!!!
        }
        else
        {
          // loop over all dimensions
          for (int j = 0; j < Interface::n_dim(); ++j)
          {
            // loop over all entries of the current derivative map (txi)
            for (colcurr = dtximap[j].begin(); colcurr != dtximap[j].end(); ++colcurr)
            {
              int col = colcurr->first;
#ifdef CONSISTENTSLIP
              double val = -frcoeff * ct * (colcurr->second) * jump[j];
#else
              double val = (-1) * (frcoeff * (znor - cn * wgap)) * ct * (colcurr->second) * jump[j];
#endif
              // do not assemble zeros into matrix
              if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
              {
                for (int j = 0; j < Interface::n_dim(); j++)
                  if (abs(val * txi[j]) > 1.e-12)
                    linslipDISglobal.assemble(val * txi[j], cnode->dofs()[j], col);
              }
              else if (abs(val) > 1.0e-12)
                linslipDISglobal.assemble(val, row[0], col);
            }

            if (Interface::n_dim() == 3)
            {
              // loop over all entries of the current derivative map (teta)
              for (colcurr = dtetamap[j].begin(); colcurr != dtetamap[j].end(); ++colcurr)
              {
                int col = colcurr->first;
#ifdef CONSISTENTSLIP
                double val = -frcoeff * ct * (colcurr->second) * jump[j];
#else
                double val =
                    (-1) * (frcoeff * (znor - cn * wgap)) * ct * (colcurr->second) * jump[j];
#endif
                // do not assemble zeros into s matrix
                if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
                {
                  for (int j = 0; j < Interface::n_dim(); j++)
                    if (abs(val * teta[j]) > 1.e-12)
                      linslipDISglobal.assemble(val * teta[j], cnode->dofs()[j], col);
                }
                else if (abs(val) > 1.0e-12)
                  linslipDISglobal.assemble(val, row[1], col);
              }
            }
          }
        }

#ifndef CONSISTENTSLIP
        /*** 6 ******************* -frcoeff.Deriv(n).z(ztan+ct*utan) ***/
        // loop over all dimensions
        for (int j = 0; j < Interface::n_dim(); ++j)
        {
          // loop over all entries of the current derivative map
          for (colcurr = dnmap[j].begin(); colcurr != dnmap[j].end(); ++colcurr)
          {
            int col = colcurr->first;
            double valtxi = (-1) * (ztxi + ct * jumptxi) * frcoeff * (colcurr->second) * z[j];
            double valteta = (-1) * (zteta + ct * jumpteta) * frcoeff * (colcurr->second) * z[j];

            // do not assemble zeros into s matrix
            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(valtxi * txi[j]) > 1.e-12)
                  linslipDISglobal.assemble(valtxi * txi[j], cnode->dofs()[j], col);
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(valteta * teta[j]) > 1.e-12)
                  linslipDISglobal.assemble(valteta * teta[j], cnode->dofs()[j], col);
            }
            else
            {
              if (abs(valtxi) > 1.0e-12) linslipDISglobal.assemble(valtxi, row[0], col);
              if (abs(valteta) > 1.0e-12) linslipDISglobal.assemble(valteta, row[1], col);
            }
          }
        }

        /*** 7 ****************** frcoeff*cn*deriv (g).(ztan+ct*utan) ***/
        // prepare assembly
        std::map<int, double>& dgmap = cnode->data().get_deriv_g();

        // loop over all entries of the current derivative map
        for (_colcurr = dgmap.begin(); _colcurr != dgmap.end(); ++_colcurr)
        {
          int col = _colcurr->first;
          double valtxi = frcoeff * cn * (_colcurr->second) * (ztxi + ct * jumptxi);
          double valteta = frcoeff * cn * (_colcurr->second) * (zteta + ct * jumpteta);

          // do not assemble zeros into matrix
          if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
          {
            for (int j = 0; j < Interface::n_dim(); j++)
              if (abs(valtxi * txi[j]) > 1.e-12)
                linslipDISglobal.assemble(valtxi * txi[j], cnode->dofs()[j], col);
            for (int j = 0; j < Interface::n_dim(); j++)
              if (abs(valteta * teta[j]) > 1.e-12)
                linslipDISglobal.assemble(valteta * teta[j], cnode->dofs()[j], col);
          }
          else
          {
            if (abs(valtxi) > 1.0e-12) linslipDISglobal.assemble(valtxi, row[0], col);
            if (abs(valteta) > 1.0e-12) linslipDISglobal.assemble(valteta, row[1], col);
          }
        }
#endif
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
    // loop over all slip nodes of the interface
    for (int i = 0; i < slipnodes_->num_my_elements(); ++i)
    {
      int gid = slipnodes_->gid(i);
      Core::Nodes::Node* node = idiscret_->g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      FriNode* cnode = dynamic_cast<FriNode*>(node);

      if (cnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
        FOUR_C_THROW("AssembleLinSlip: Node ownership inconsistency!");

      double ct = get_ct_ref().local_values_as_span()[get_ct_ref().get_map().lid(cnode->id())];

      // preparation of assembly
      // get Deriv N and calculate DerivD form DerivN

      // only for 2D so far, in this case calculation is very easy
      // dty =  dnx
      // dtx = -dny
      // FIXGIT: in the future DerivD will be called directly form node

      std::vector<Core::Gen::Pairedvector<int, double>> dnmap = cnode->data().get_deriv_n();

      // iterator
      Core::Gen::Pairedvector<int, double>::iterator _colcurr;
      std::map<int, double>::iterator colcurr;

      std::vector<std::map<int, double>> dtmap(Interface::n_dim());

      for (_colcurr = dnmap[0].begin(); _colcurr != dnmap[0].end(); _colcurr++)
        dtmap[1].insert(std::pair<int, double>(_colcurr->first, _colcurr->second));

      for (_colcurr = dnmap[1].begin(); _colcurr != dnmap[1].end(); _colcurr++)
        dtmap[0].insert(std::pair<int, double>(_colcurr->first, (-1) * _colcurr->second));

      // get more information from node
      double* jump = cnode->fri_data().jump();
      double* txi = cnode->data().txi();
      double* xi = cnode->xspatial();
      double* z = cnode->mo_data().lm();
      int row = slipt_->gid(i);

      int colsize = (int)dtmap[0].size();
      int mapsize = (int)dtmap.size();

      for (int j = 0; j < mapsize - 1; ++j)
        if ((int)dtmap[j].size() != (int)dtmap[j + 1].size())
          FOUR_C_THROW("AssembleLinSlip: Column dim. of nodal DerivT-map is inconsistent!");

      // calculation of parts of the complementary function
      double ztan = txi[0] * z[0] + txi[1] * z[1];
      double jumptan = txi[0] * jump[0] + txi[1] * jump[1];
      // double temp = ztan + ct*jumptan;
      // double epk = frbound/abs(temp);
      // double Fpk = ztan*temp/(frbound*abs(temp));
      // double Mpk = epk*(1-Fpk);
      // double fac = 1/(abs(ztan+ct*jumptan))*1/(1-Mpk)*(-1);

      // calculation of |ztan+ct*utan|
      double sum = 0;
      int prefactor = 1;
      for (int dim = 0; dim < Interface::n_dim(); dim++)
        sum += txi[dim] * z[dim] + ct * txi[dim] * jump[dim];

      // calculate |sum| and prefactor
      if (sum < 0)
      {
        sum = -sum;
        prefactor = (-1);
      }

      //****************************************************************
      // CONSISTENT TREATMENT OF CASE FRBOUND=0 (FRICTIONLESS)
      //****************************************************************
      // popp 08/2012
      //
      // There is a problem with our frictional nonlinear complementarity
      // function when applied to the limit case frbound=0 (frictionless).
      // In this case, the simple frictionless sliding condition should
      // be consistently recovered, which unfortunately is not the case.
      // This fact is well-known (see PhD thesis S. Hueeber) and now
      // taken care of by a special treatment as can be seen below
      //
      //****************************************************************
      if (frbound == 0.0 || (cnode->data().active_old() == false && frilessfirst))
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

          // do not assemble zeros into matrix
          if (abs(valtxi) > 1.0e-12) linslipLMglobal.assemble(valtxi, row, col);
        }

        // 2) Entries on right hand side
        /******************************************************************/

        Core::LinAlg::SerialDenseVector rhsnode(1);
        std::vector<int> lm(1);
        std::vector<int> lmowner(1);

        rhsnode(0) = -ztan;
        lm[0] = cnode->dofs()[1];
        lmowner[0] = cnode->owner();

        Core::LinAlg::assemble(linslipRHSglobal, rhsnode, lm, lmowner);

        // 3) Entries from differentiation with respect to displacements
        /******************************************************************/

        // loop over dimensions
        for (int j = 0; j < Interface::n_dim(); ++j)
        {
          // loop over all entries of the current derivative map (txi)
          for (colcurr = dtmap[j].begin(); colcurr != dtmap[j].end(); ++colcurr)
          {
            int col = colcurr->first;
            double val = (colcurr->second) * z[j];

            // do not assemble zeros into matrix
            if (abs(val) > 1.0e-12) linslipDISglobal.assemble(val, row, col);
          }
        }
      }

      //****************************************************************
      // STANDARD TREATMENT OF CASE FRBOUND!=0 (FRICTIONAL)
      //****************************************************************
      else
      {
        //****************************************************************
        // calculation of matrix entries of the linearized slip condition
        //****************************************************************

        // 1) Entries from differentiation with respect to LM
        /**************** (Deriv(abs)*ztan+|ztan+ct*jumptan|-frbound).tan ***/

        // loop over the dimension
        for (int dim = 0; dim < cnode->num_dof(); ++dim)
        {
          int col = cnode->dofs()[dim];
          double val = (prefactor * ztan + sum - frbound) * txi[dim];

          // do not assemble zeros into matrix
          if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
          {
            for (int j = 0; j < Interface::n_dim(); j++)
              if (abs(val * txi[j]) > 1.e-12)
                linslipLMglobal.assemble(val * txi[j], cnode->dofs()[j], col);
          }
          else if (abs(val) > 1.0e-12)
            linslipLMglobal.assemble(val, row, col);
        }

        // 2) Entries on right hand side
        /************ -C + entries from writing Delta(z) as z(k+1)-z(k) ***/

        // -C and remaining terms
        double value1 = -(abs(ztan + ct * jumptan)) * ztan + frbound * (ztan + ct * jumptan);

        if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
        {
          Core::LinAlg::SerialDenseVector rhsnode(Interface::n_dim());
          std::vector<int> lm(Interface::n_dim());
          std::vector<int> lmowner(Interface::n_dim());

          for (int j = 0; j < Interface::n_dim(); j++)
          {
            lm[j] = cnode->dofs()[j];
            lmowner[j] = cnode->owner();
            rhsnode(j) = value1 * txi[j];
          }

          Core::LinAlg::assemble(linslipRHSglobal, rhsnode, lm, lmowner);
        }
        else
        {
          Core::LinAlg::SerialDenseVector rhsnode(1);
          rhsnode(0) = value1;

          std::vector<int> lm(1);
          std::vector<int> lmowner(1);

          lm[0] = cnode->dofs()[1];
          lmowner[0] = cnode->owner();

          Core::LinAlg::assemble(linslipRHSglobal, rhsnode, lm, lmowner);
        }

        // 3) Entries from differentiation with respect to displacements
        /***************************** -Deriv(abs)*ct*tan.(D-Dn-1)*ztan ***/

        // we need the nodal entries of the D-matrix and the old one
        double D = cnode->mo_data().get_d()[cnode->id()];
        double Dold = cnode->fri_data().get_d_old()[cnode->id()];

        if (abs(Dold) < 0.0001) FOUR_C_THROW("Error:No entry for Dold");

        // loop over all derivative maps (=dimensions)
        for (int dim = 0; dim < cnode->num_dof(); ++dim)
        {
          int col = cnode->dofs()[dim];
          double val = prefactor * (-1) * ct * txi[dim] * (D - Dold) * ztan;
          // std::cout << "01 GID " << gid << " row " << row << " col " << col << " val " << val <<
          // std::endl;

          // do not assemble zeros into matrix
          if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
          {
            for (int j = 0; j < Interface::n_dim(); j++)
              if (abs(val * txi[j]) > 1.e-12)
                linslipDISglobal.assemble(val * txi[j], cnode->dofs()[j], col);
          }
          else if (abs(val) > 1.0e-12)
            linslipDISglobal.assemble(val, row, col);
        }

        /***************************** -Deriv(abs)*ct*tan.(M-Mn-1)*ztan ***/

        // we need the nodal entries of the M-matrix and the old one
        std::map<int, double>& mmap = cnode->mo_data().get_m();
        std::map<int, double>& mmapold = cnode->fri_data().get_m_old();

        // create a set of nodes including nodes according to M entries
        // from current and previous time step
        std::set<int> mnodes;

        // iterator
        std::set<int>::iterator mcurr;

        std::set<int> mnodescurrent = cnode->fri_data().get_m_nodes();
        std::set<int> mnodesold = cnode->fri_data().get_m_nodes_old();

        for (mcurr = mnodescurrent.begin(); mcurr != mnodescurrent.end(); mcurr++)
          mnodes.insert(*mcurr);

        for (mcurr = mnodesold.begin(); mcurr != mnodesold.end(); mcurr++) mnodes.insert(*mcurr);

        // loop over all master nodes (find adjacent ones to this stick node)
        for (mcurr = mnodes.begin(); mcurr != mnodes.end(); mcurr++)
        {
          int gid = *mcurr;
          Core::Nodes::Node* mnode = idiscret_->g_node(gid);
          if (!mnode) FOUR_C_THROW("Cannot find node with gid %", gid);
          FriNode* cmnode = dynamic_cast<FriNode*>(mnode);

          double mik = mmap[cmnode->id()];
          double mikold = mmapold[cmnode->id()];

          // compute linstick-matrix entry of the current active node / master node pair
          // loop over all derivative maps (=dimensions)
          for (int dim = 0; dim < cnode->num_dof(); ++dim)
          {
            int col = cmnode->dofs()[dim];
            double val = prefactor * (+1) * ct * txi[dim] * (mik - mikold) * ztan;
            // std::cout << "02 GID " << gid << " row " << row << " col " << col << " val " << val
            // << std::endl;

            // do not assemble zeros into matrix
            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(val * txi[j]) > 1.e-12)
                  linslipDISglobal.assemble(val * txi[j], cnode->dofs()[j], col);
            }
            else if (abs(val) > 1.0e-12)
              linslipDISglobal.assemble(val, row, col);
          }
        }

        /************************************** frbound*ct*tan.(D-Dn-1) ***/

        // loop over all derivative maps (=dimensions)
        for (int dim = 0; dim < cnode->num_dof(); ++dim)
        {
          int col = cnode->dofs()[dim];
          double val = frbound * ct * txi[dim] * (D - Dold);
          // std::cout << "03 GID " << gid << " row " << row << " col " << col << " val " << val <<
          // std::endl;

          // do not assemble zeros into matrix
          if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
          {
            for (int j = 0; j < Interface::n_dim(); j++)
              if (abs(val * txi[j]) > 1.e-12)
                linslipDISglobal.assemble(val * txi[j], cnode->dofs()[j], col);
          }
          else if (abs(val) > 1.0e-12)
            linslipDISglobal.assemble(val, row, col);
        }

        /********************************** -frbound*ct*tan.(M-Mn-1).xm ***/

        // loop over all master nodes
        for (mcurr = mnodes.begin(); mcurr != mnodes.end(); mcurr++)
        {
          int gid = *mcurr;
          Core::Nodes::Node* mnode = idiscret_->g_node(gid);
          if (!mnode) FOUR_C_THROW("Cannot find node with gid %", gid);
          FriNode* cmnode = dynamic_cast<FriNode*>(mnode);

          double mik = mmap[cmnode->id()];
          double mikold = mmapold[cmnode->id()];

          // loop over all derivative maps (=dimensions)
          for (int dim = 0; dim < cnode->num_dof(); ++dim)
          {
            int col = cmnode->dofs()[dim];
            double val = frbound * (-1) * ct * txi[dim] * (mik - mikold);
            // std::cout << "04 GID " << gid << " row " << row << " col " << col << " val " << val
            // << std::endl;
            // do not assemble zeros into matrix
            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(val * txi[j]) > 1.e-12)
                  linslipDISglobal.assemble(val * txi[j], cnode->dofs()[j], col);
            }
            else if (abs(val) > 1.0e-12)
              linslipDISglobal.assemble(val, row, col);
          }
        }

        /************************************ |ztan+ct*utan|.DerivT.z ***/

        // loop over all derivative maps (=dimensions)
        for (int j = 0; j < mapsize; ++j)
        {
          int k = 0;

          // loop over all entries of the current derivative map
          for (colcurr = dtmap[j].begin(); colcurr != dtmap[j].end(); ++colcurr)
          {
            int col = colcurr->first;
            double val = sum * (colcurr->second) * z[j];
            // std::cout << "1 GID " << gid << " row " << row << " col " << col << " val " << val <<
            // std::endl;

            // do not assemble zeros into s matrix
            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(val * txi[j]) > 1.e-12)
                  linslipDISglobal.assemble(val * txi[j], cnode->dofs()[j], col);
            }
            else if (abs(val) > 1.0e-12)
              linslipDISglobal.assemble(val, row, col);

            ++k;
          }

          if (k != colsize) FOUR_C_THROW("AssembleLinSlip: k = {} but colsize = {}", k, colsize);
        }

        /*********************************** Deriv(abs)*DerivT.z*ztan ***/

        // loop over all derivative maps (=dimensions)
        for (int j = 0; j < mapsize; ++j)
        {
          int k = 0;

          // loop over all entries of the current derivative map
          for (colcurr = dtmap[j].begin(); colcurr != dtmap[j].end(); ++colcurr)
          {
            int col = colcurr->first;
            double val = prefactor * (colcurr->second) * z[j] * ztan;
            // std::cout << "2 GID " << gid << " row " << row << " col " << col << " val " << val <<
            // std::endl;

            // do not assemble zeros into matrix
            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(val * txi[j]) > 1.e-12)
                  linslipDISglobal.assemble(val * txi[j], cnode->dofs()[j], col);
            }
            else if (abs(val) > 1.0e-12)
              linslipDISglobal.assemble(val, row, col);

            ++k;
          }

          if (k != colsize) FOUR_C_THROW("AssembleLinSlip: k = {} but colsize = {}", k, colsize);
        }

        /******************************* Deriv(abs)*DerivT.jump+*ztan ***/

        // loop over all derivative maps (=dimensions)
        for (int j = 0; j < mapsize; ++j)
        {
          int k = 0;

          // loop over all entries of the current derivative map
          for (colcurr = dtmap[j].begin(); colcurr != dtmap[j].end(); ++colcurr)
          {
            int col = colcurr->first;
            double val = prefactor * ct * (colcurr->second) * jump[j] * ztan;
            // std::cout << "3 GID " << gid << " row " << row << " col " << col << " val " << val <<
            // std::endl;

            // do not assemble zeros into s matrix
            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(val * txi[j]) > 1.e-12)
                  linslipDISglobal.assemble(val * txi[j], cnode->dofs()[j], col);
            }
            else if (abs(val) > 1.0e-12)
              linslipDISglobal.assemble(val, row, col);

            ++k;
          }

          if (k != colsize) FOUR_C_THROW("AssembleLinSlip: k = {} but colsize = {}", k, colsize);
        }

        /*************************** -Deriv(abs).ct.tan.DerivD.x*ztan ***/

        // we need the dot product t*x of this node
        double tdotx = 0.0;
        for (int dim = 0; dim < cnode->num_dof(); ++dim) tdotx += txi[dim] * xi[dim];

        // prepare assembly
        std::map<int, double>& ddmap = cnode->data().get_deriv_d()[gid];

        // loop over all entries of the current derivative map
        for (colcurr = ddmap.begin(); colcurr != ddmap.end(); ++colcurr)
        {
          int col = colcurr->first;
          double val = (-1) * prefactor * ct * tdotx * colcurr->second * ztan;
          // std::cout << "4 GID " << gid << " row " << row << " col " << col << " val " << val <<
          // std::endl;

          // do not assemble zeros into matrix
          if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
          {
            for (int j = 0; j < Interface::n_dim(); j++)
              if (abs(val * txi[j]) > 1.e-12)
                linslipDISglobal.assemble(val * txi[j], cnode->dofs()[j], col);
          }
          else if (abs(val) > 1.0e-12)
            linslipDISglobal.assemble(val, row, col);
        }

        /**************************** Deriv(abs).ct.tan.DerivM.x*ztan ***/

        // we need the Lin(M-matrix) entries of this node
        std::map<int, std::map<int, double>>& dmmap = cnode->data().get_deriv_m();
        std::map<int, std::map<int, double>>::iterator dmcurr;

        // loop over all master nodes in the DerivM-map of the active slave node
        for (dmcurr = dmmap.begin(); dmcurr != dmmap.end(); ++dmcurr)
        {
          int gid = dmcurr->first;
          Core::Nodes::Node* mnode = idiscret_->g_node(gid);
          if (!mnode) FOUR_C_THROW("Cannot find node with gid %", gid);
          FriNode* cmnode = dynamic_cast<FriNode*>(mnode);
          double* mxi = cmnode->xspatial();

          // we need the dot product ns*xm of this node pair
          double tdotx = 0.0;
          for (int dim = 0; dim < cnode->num_dof(); ++dim) tdotx += txi[dim] * mxi[dim];

          // compute entry of the current active node / master node pair
          std::map<int, double>& thisdmmap = cnode->data().get_deriv_m(gid);

          // loop over all entries of the current derivative map
          for (colcurr = thisdmmap.begin(); colcurr != thisdmmap.end(); ++colcurr)
          {
            int col = colcurr->first;
            double val = prefactor * ct * tdotx * colcurr->second * ztan;
            // std::cout << "5 GID " << gid << " row " << row << " col " << col << " val " << val <<
            // std::endl;

            // do not assemble zeros into matrix
            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(val * txi[j]) > 1.e-12)
                  linslipDISglobal.assemble(val * txi[j], cnode->dofs()[j], col);
            }
            else if (abs(val) > 1.0e-12)
              linslipDISglobal.assemble(val, row, col);
          }
        }

        /****************************************** -frbound.DerivT.z ***/

        // loop over all derivative maps (=dimensions)
        for (int j = 0; j < mapsize; ++j)
        {
          int k = 0;

          // loop over all entries of the current derivative map
          for (colcurr = dtmap[j].begin(); colcurr != dtmap[j].end(); ++colcurr)
          {
            int col = colcurr->first;
            double val = (-1) * frbound * (colcurr->second) * z[j];
            // std::cout << "6 GID " << gid << " row " << row << " col " << col << " val " << val <<
            // std::endl;

            // do not assemble zeros into s matrix
            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(val * txi[j]) > 1.e-12)
                  linslipDISglobal.assemble(val * txi[j], cnode->dofs()[j], col);
            }
            else if (abs(val) > 1.0e-12)
              linslipDISglobal.assemble(val, row, col);

            ++k;
          }

          if (k != colsize) FOUR_C_THROW("AssembleLinSlip: k = {} but colsize = {}", k, colsize);
        }

        /************************************ -frbound.ct.DerivT.jump ***/

        // loop over all derivative maps (=dimensions)
        for (int j = 0; j < mapsize; ++j)
        {
          int k = 0;

          // loop over all entries of the current derivative map
          for (colcurr = dtmap[j].begin(); colcurr != dtmap[j].end(); ++colcurr)
          {
            int col = colcurr->first;
            double val = (-1) * frbound * ct * (colcurr->second) * jump[j];
            // std::cout << "7 GID " << gid << " row " << row << " col " << col << " val " << val <<
            // std::endl;

            // do not assemble zeros into s matrix
            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(val * txi[j]) > 1.e-12)
                  linslipDISglobal.assemble(val * txi[j], cnode->dofs()[j], col);
            }
            else if (abs(val) > 1.0e-12)
              linslipDISglobal.assemble(val, row, col);

            ++k;
          }

          if (k != colsize) FOUR_C_THROW("AssembleLinSlip: k = {} but colsize = {}", k, colsize);
        }

        /************************************* +frbound.ct.T.DerivD.x ***/

        // we need the dot product t*x of this node
        tdotx = 0.0;
        for (int dim = 0; dim < cnode->num_dof(); ++dim) tdotx += txi[dim] * xi[dim];

        // loop over all entries of the current derivative map
        for (colcurr = ddmap.begin(); colcurr != ddmap.end(); ++colcurr)
        {
          int col = colcurr->first;
          double val = (-1) * (-1) * frbound * ct * tdotx * colcurr->second;
          // std::cout << "8 GID " << gid << " row " << row << " col " << col << " val " << val <<
          // std::endl;

          // do not assemble zeros into matrix
          if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
          {
            for (int j = 0; j < Interface::n_dim(); j++)
              if (abs(val * txi[j]) > 1.e-12)
                linslipDISglobal.assemble(val * txi[j], cnode->dofs()[j], col);
          }
          else if (abs(val) > 1.0e-12)
            linslipDISglobal.assemble(val, row, col);
        }

        /********************************  -frbound.ct.T.DerivM.x ******/

        // loop over all master nodes in the DerivM-map of the active slave node
        for (dmcurr = dmmap.begin(); dmcurr != dmmap.end(); ++dmcurr)
        {
          int gid = dmcurr->first;
          Core::Nodes::Node* mnode = idiscret_->g_node(gid);
          if (!mnode) FOUR_C_THROW("Cannot find node with gid %", gid);
          FriNode* cmnode = dynamic_cast<FriNode*>(mnode);
          double* mxi = cmnode->xspatial();

          // we need the dot product ns*xm of this node pair
          double tdotx = 0.0;
          for (int dim = 0; dim < cnode->num_dof(); ++dim) tdotx += txi[dim] * mxi[dim];

          // compute entry of the current active node / master node pair
          std::map<int, double>& thisdmmap = cnode->data().get_deriv_m(gid);

          // loop over all entries of the current derivative map
          for (colcurr = thisdmmap.begin(); colcurr != thisdmmap.end(); ++colcurr)
          {
            int col = colcurr->first;
            double val = (-1) * frbound * ct * tdotx * colcurr->second;
            // std::cout << "9 GID " << gid << " row " << row << " col " << col << " val " << val <<
            // std::endl;


            // do not assemble zeros into matrix
            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(val * txi[j]) > 1.e-12)
                  linslipDISglobal.assemble(val * txi[j], cnode->dofs()[j], col);
            }
            else if (abs(val) > 1.0e-12)
              linslipDISglobal.assemble(val, row, col);
          }
        }
      }
    }
  }  // Tresca friction

  return;
}

void CONTACT::Interface::assemble_normal_contact_regularization(Core::LinAlg::SparseMatrix& d_disp,
    Core::LinAlg::SparseMatrix& d_lm, Core::LinAlg::Vector<double>& f)
{
  const bool regularization = interface_params().get<bool>("REGULARIZED_NORMAL_CONTACT");
  if (!regularization) FOUR_C_THROW("you should not be here");
  const double k = 1. / interface_params().get<double>("REGULARIZATION_STIFFNESS");
  const double gmax = interface_params().get<double>("REGULARIZATION_THICKNESS");
  const int dim = Interface::n_dim();
  static const auto constr_direction = Teuchos::getIntegralValue<CONTACT::ConstraintDirection>(
      interface_params(), "CONSTRAINT_DIRECTIONS");

  for (int i = 0; i < active_nodes()->num_my_elements(); ++i)
  {
    Core::Nodes::Node* node = discret().g_node(active_nodes()->gid(i));
    if (!node) FOUR_C_THROW("node not found");
    CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(node);
    if (!cnode) FOUR_C_THROW("not a contact node");

    Core::LinAlg::Matrix<3, 1> n(cnode->mo_data().n(), true);
    Core::LinAlg::Matrix<3, 1> lm(cnode->mo_data().lm(), true);
    const double lm_n = lm.dot(n);

    const double gLM = gmax * (1. - exp(-k / gmax * lm_n));
    const double d_gLM = k * exp(-k / gmax * lm_n);

    if (cnode->mo_data().get_d().size() != 1)
      FOUR_C_THROW(
          "we need to have a D-value for active contact nodes\nAnd exactly one due to "
          "biorthogonality");
    double dval = cnode->mo_data().get_d().at(cnode->id());

    if (constr_direction == CONTACT::ConstraintDirection::xyz)
      for (int d = 0; d < dim; ++d)
        f.get_values()[f.get_map().lid(cnode->dofs()[d])] += n(d) * dval * gLM;
    else
      f.get_values()[f.get_map().lid(cnode->dofs()[0])] += dval * gLM;

    for (int l = 0; l < dim; ++l)
    {
      if (constr_direction == CONTACT::ConstraintDirection::xyz)
        for (int d = 0; d < dim; ++d)
          d_lm.assemble(n(d) * dval * n(l) * d_gLM, cnode->dofs()[d], cnode->dofs()[l]);
      else
        d_lm.assemble(dval * d_gLM * n(l), cnode->dofs()[0], cnode->dofs()[l]);
    }

    for (auto p = cnode->data().get_deriv_d().at(cnode->id()).begin();
        p != cnode->data().get_deriv_d().at(cnode->id()).end(); ++p)
    {
      const int col = p->first;
      const double val = gLM * p->second;
      if (constr_direction == CONTACT::ConstraintDirection::xyz)
        for (int d = 0; d < dim; ++d) d_disp.assemble(val * n(d), cnode->dofs()[d], col);
      else
        d_disp.assemble(val, cnode->dofs()[0], col);
    }

    for (int l = 0; l < dim; ++l)
      for (auto p = cnode->data().get_deriv_n().at(l).begin();
          p != cnode->data().get_deriv_n().at(l).end(); ++p)
      {
        const int col = p->first;
        const double val = dval * d_gLM * lm(l) * p->second;
        if (constr_direction == CONTACT::ConstraintDirection::xyz)
          for (int d = 0; d < dim; ++d) d_disp.assemble(val * n(d), cnode->dofs()[d], col);
        else
          d_disp.assemble(val, cnode->dofs()[0], col);
      }
  }

  return;
}

void CONTACT::Interface::assemble_lin_slip_normal_regularization(
    Core::LinAlg::SparseMatrix& linslipLMglobal, Core::LinAlg::SparseMatrix& linslipDISglobal,
    Core::LinAlg::Vector<double>& linslipRHSglobal)
{
  // nothing to do if no slip nodes
  if (slipnodes_->num_my_elements() == 0) return;

  // information from interface contact parameter list
  auto ftype = Teuchos::getIntegralValue<CONTACT::FrictionType>(interface_params(), "FRICTION");
  double frcoeff_in =
      interface_params().get<double>("FRCOEFF");  // the friction coefficient from the input
  const bool gp_slip = interface_params().get<bool>("GP_SLIP_INCR");
  if (gp_slip) FOUR_C_THROW("not implemented");
  const bool frilessfirst = interface_params().get<bool>("FRLESS_FIRST");

  // the friction coefficient adapted by every node (eg depending on the local temperature)
  double frcoeff = 0.;

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
      FriNode* cnode = dynamic_cast<FriNode*>(node);

      if (cnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
        FOUR_C_THROW("AssembleLinSlip: Node ownership inconsistency!");

      // get friction coefficient for this node
      // in case of TSI, the nodal temperature influences the local friction coefficient
      frcoeff = cnode->fr_coeff(frcoeff_in);

      double cn = get_cn_ref().local_values_as_span()[get_cn_ref().get_map().lid(cnode->id())];
      double ct = get_ct_ref().local_values_as_span()[get_ct_ref().get_map().lid(cnode->id())];

      // prepare assembly, get information from node
      std::vector<Core::Gen::Pairedvector<int, double>> dnmap = cnode->data().get_deriv_n();
      std::vector<Core::Gen::Pairedvector<int, double>> dtximap = cnode->data().get_deriv_txi();
      std::vector<Core::Gen::Pairedvector<int, double>> dtetamap = cnode->data().get_deriv_teta();

      // check for Dimension of derivative maps
      for (int j = 0; j < Interface::n_dim() - 1; ++j)
        if ((int)dnmap[j].size() != (int)dnmap[j + 1].size())
          FOUR_C_THROW("AssembleLinSlip: Column dim. of nodal DerivTxi-map is inconsistent!");

      for (int j = 0; j < Interface::n_dim() - 1; ++j)
        if ((int)dtximap[j].size() != (int)dtximap[j + 1].size())
          FOUR_C_THROW("AssembleLinSlip: Column dim. of nodal DerivTxi-map is inconsistent!");

      if (Interface::n_dim() == 3)
      {
        for (int j = 0; j < Interface::n_dim() - 1; ++j)
          if ((int)dtximap[j].size() != (int)dtximap[j + 1].size())
            FOUR_C_THROW("AssembleLinSlip: Column dim. of nodal DerivTeta-map is inconsistent!");
      }

      // more information from node
      double* n = cnode->mo_data().n();
      double* txi = cnode->data().txi();
      double* teta = cnode->data().teta();
      double* z = cnode->mo_data().lm();
      double wgap = cnode->data().getg();

      // iterator for maps
      Core::Gen::Pairedvector<int, double>::iterator colcurr;
      std::map<int, double>::iterator _colcurr;

      // row number of entries
      std::vector<int> row(Interface::n_dim() - 1);
      if (Interface::n_dim() == 2)
      {
        row[0] = slipt_->gid(i);
      }
      else if (Interface::n_dim() == 3)
      {
        row[0] = slipt_->gid(2 * i);
        row[1] = slipt_->gid(2 * i) + 1;
      }
      else
        FOUR_C_THROW("AssemblelinSlip: Dimension not correct");

      // evaluation of specific components of entries to assemble
      double znor = 0.0;
      double ztxi = 0.0;
      double zteta = 0.0;
      double jumptxi = 0.0;
      double jumpteta = 0.0;
      double euclidean = 0.0;
      double* jump = cnode->fri_data().jump();

      for (int i = 0; i < Interface::n_dim(); i++)
      {
        znor += n[i] * z[i];
        ztxi += txi[i] * z[i];
        zteta += teta[i] * z[i];
        jumptxi += txi[i] * jump[i];
        jumpteta += teta[i] * jump[i];
      }

      // setup regularization
      static const bool regularization = interface_params().get<bool>("REGULARIZED_NORMAL_CONTACT");
      if (!regularization) FOUR_C_THROW("you should not be here");
      static const double k = 1. / interface_params().get<double>("REGULARIZATION_STIFFNESS");
      static const double gmax = interface_params().get<double>("REGULARIZATION_THICKNESS");
      if (cnode->mo_data().get_d().size() != 1)
        FOUR_C_THROW(
            "we need to have a D-value for active contact nodes\nAnd exactly one due to "
            "biorthogonality");
      double dval = cnode->mo_data().get_d().at(cnode->id());
      const double gLM = gmax * (1. - exp(-k / gmax * znor));
      const double d_gLM = k * exp(-k / gmax * znor);
      const double frbound = frcoeff * std::max(0., znor - cn * (wgap + dval * gLM));

      //      std::cout << "node: " << cnode->Id() << " wgap: " << wgap << " dval*gLM: " << dval*gLM
      //      << std::endl;

      // evaluate euclidean norm ||vec(zt)+ct*vec(jump)||
      std::vector<double> sum1(Interface::n_dim() - 1, 0);
      sum1[0] = ztxi + ct * jumptxi;
      if (Interface::n_dim() == 3) sum1[1] = zteta + ct * jumpteta;
      if (Interface::n_dim() == 2) euclidean = abs(sum1[0]);
      if (Interface::n_dim() == 3) euclidean = sqrt(sum1[0] * sum1[0] + sum1[1] * sum1[1]);

      // check of dimensions
      if (Interface::n_dim() == 2 and (zteta != 0.0 or jumpteta != 0.0))
        FOUR_C_THROW("AssemblelinSlip: zteta and jumpteta must be zero in 2D");

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
      // seitz11/13
      // Like that FRLESS_FIRST results in a frictionless constraint.
      // Nodes with a vanishing slip and tangential Lagrange multiplier
      // are treated as frictionless as well.
      //
      //****************************************************************
      if (frbound == 0.0 || (cnode->data().active_old() == false && frilessfirst) ||
          euclidean == 0.0)
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
          if (Interface::n_dim() == 3) valteta = teta[dim];

          // do not assemble zeros into matrix
          if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
          {
            for (int j = 0; j < Interface::n_dim(); j++)
            {
              if (abs(valtxi * txi[j]) > 1.e-12)
                linslipLMglobal.assemble(valtxi * txi[j], cnode->dofs()[j], col);
              if (Interface::n_dim() == 3)
                if (abs(valteta * teta[j]) > 1.e-12)
                  linslipLMglobal.assemble(valteta * teta[j], cnode->dofs()[j], col);
            }
          }
          else
          {
            if (abs(valtxi) > 1.0e-12) linslipLMglobal.assemble(valtxi, row[0], col);
            if (Interface::n_dim() == 3)
              if (abs(valteta) > 1.0e-12) linslipLMglobal.assemble(valteta, row[1], col);
          }
        }

        // 2) Entries on right hand side
        /******************************************************************/
        if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
        {
          Core::LinAlg::SerialDenseVector rhsnode(Interface::n_dim());
          std::vector<int> lm(Interface::n_dim());
          std::vector<int> lmowner(Interface::n_dim());
          for (int j = 0; j < Interface::n_dim(); j++)
          {
            lm[j] = cnode->dofs()[j];
            lmowner[j] = cnode->owner();
            rhsnode[j] -= ztxi * txi[j];
            if (Interface::n_dim() == 3) rhsnode[j] -= zteta * teta[j];
          }
          Core::LinAlg::assemble(linslipRHSglobal, rhsnode, lm, lmowner);
        }
        else
        {
          Core::LinAlg::SerialDenseVector rhsnode(Interface::n_dim() - 1);
          std::vector<int> lm(Interface::n_dim() - 1);
          std::vector<int> lmowner(Interface::n_dim() - 1);

          lm[0] = cnode->dofs()[1];
          lmowner[0] = cnode->owner();

          rhsnode[0] = -ztxi;  // already negative rhs!!!

          if (Interface::n_dim() == 3)
          {
            rhsnode[1] = -zteta;  // already negative rhs!!!

            lm[1] = cnode->dofs()[2];
            lmowner[1] = cnode->owner();
          }
          Core::LinAlg::assemble(linslipRHSglobal, rhsnode, lm, lmowner);
        }

        // 3) Entries from differentiation with respect to displacements
        /******************************************************************/

        // loop over dimensions
        for (int j = 0; j < Interface::n_dim(); ++j)
        {
          // loop over all entries of the current derivative map (txi)
          for (colcurr = dtximap[j].begin(); colcurr != dtximap[j].end(); ++colcurr)
          {
            int col = colcurr->first;
            double val = (colcurr->second) * z[j];

            // do not assemble zeros into matrix
            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(val * txi[j]) > 1.e-12)
                  linslipDISglobal.assemble(val * txi[j], cnode->dofs()[j], col);
            }
            else if (abs(val) > 1.0e-12)
              linslipDISglobal.assemble(val, row[0], col);
          }

          if (Interface::n_dim() == 3)
          {
            // loop over all entries of the current derivative map (teta)
            for (colcurr = dtetamap[j].begin(); colcurr != dtetamap[j].end(); ++colcurr)
            {
              int col = colcurr->first;
              double val = (colcurr->second) * z[j];

              // do not assemble zeros into s matrix
              if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
              {
                for (int j = 0; j < Interface::n_dim(); j++)
                  if (abs(val * teta[j]) > 1.e-12)
                    linslipDISglobal.assemble(val * teta[j], cnode->dofs()[j], col);
              }
              else if (abs(val) > 1.0e-12)
                linslipDISglobal.assemble(val, row[1], col);
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

          double valtxi2 = -frbound * txi[dim] -
                           frcoeff * (1. - cn * dval * d_gLM) * (ztxi + ct * jumptxi) * n[dim];
          valtxi = valtxi0 + valtxi1 + valtxi2 + valtxi3;

          double valteta = 0.0;
          if (Interface::n_dim() == 3)
          {
            double valteta0 = euclidean * teta[dim];
            double valteta1 = ((ztxi + ct * jumptxi) / euclidean * zteta) * txi[dim];
            double valteta3 = (zteta + ct * jumpteta) / euclidean * zteta * teta[dim];
            double valteta2 = -frbound * teta[dim] -
                              frcoeff * (1. - cn * dval * d_gLM) * (zteta + ct * jumpteta) * n[dim];
            valteta = valteta0 + valteta1 + valteta2 + valteta3;
          }

          // do not assemble zeros into matrix
          if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
          {
            for (int j = 0; j < Interface::n_dim(); j++)
            {
              if (abs(valtxi * txi[j]) > 1.e-12)
                linslipLMglobal.assemble(valtxi * txi[j], cnode->dofs()[j], col);
              if (Interface::n_dim() == 3)
                if (abs(valteta * teta[j]) > 1.e-12)
                  linslipLMglobal.assemble(valteta * teta[j], cnode->dofs()[j], col);
            }
          }
          else
          {
            if (abs(valtxi) > 1.0e-12) linslipLMglobal.assemble(valtxi, row[0], col);
            if (Interface::n_dim() == 3)
              if (abs(valteta) > 1.0e-12) linslipLMglobal.assemble(valteta, row[1], col);
          }
        }

        // 2) Entries on right hand side
        /******************************************************************/
        if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
        {
          Core::LinAlg::SerialDenseVector rhsnode(Interface::n_dim());
          std::vector<int> lm(Interface::n_dim());
          std::vector<int> lmowner(Interface::n_dim());

          double valuetxi1 = -(euclidean)*ztxi + (frbound) * (ztxi + ct * jumptxi);

          for (int j = 0; j < Interface::n_dim(); j++)
          {
            lm[j] = cnode->dofs()[j];
            lmowner[j] = cnode->owner();
            rhsnode(j) += valuetxi1 * txi[j];
          }

          if (Interface::n_dim() == 3)
          {
            double valueteta1 = -(euclidean)*zteta + (frbound) * (zteta + ct * jumpteta);

            for (int j = 0; j < Interface::n_dim(); j++) rhsnode(j) += valueteta1 * teta[j];
          }
          Core::LinAlg::assemble(linslipRHSglobal, rhsnode, lm, lmowner);
        }
        else
        {
          Core::LinAlg::SerialDenseVector rhsnode(Interface::n_dim() - 1);
          std::vector<int> lm(Interface::n_dim() - 1);
          std::vector<int> lmowner(Interface::n_dim() - 1);
          double valuetxi1 = -(euclidean)*ztxi + (frbound) * (ztxi + ct * jumptxi);
          rhsnode(0) = valuetxi1;
          lm[0] = cnode->dofs()[1];
          lmowner[0] = cnode->owner();

          if (Interface::n_dim() == 3)
          {
            double valueteta1 = -(euclidean)*zteta + (frbound) * (zteta + ct * jumpteta);
            rhsnode(1) = valueteta1;

            lm[1] = cnode->dofs()[2];
            lmowner[1] = cnode->owner();
          }
          Core::LinAlg::assemble(linslipRHSglobal, rhsnode, lm, lmowner);
        }

        // 3) Entries from differentiation with respect to displacements
        /******************************************************************/
        std::map<int, double> derivjump1, derivjump2;  // for gp slip
        std::vector<std::map<int, double>> derivjump;  // for dm slip

        // get linearization of jump vector
        derivjump = cnode->fri_data().get_deriv_jump();

        // loop over dimensions
        for (int dim = 0; dim < cnode->num_dof(); ++dim)
        {
          // loop over all entries of the current derivative map (jump)
          for (_colcurr = derivjump[dim].begin(); _colcurr != derivjump[dim].end(); ++_colcurr)
          {
            int col = _colcurr->first;

            double valtxi1 =
                (ztxi + ct * jumptxi) / euclidean * ct * txi[dim] * _colcurr->second * ztxi;
            double valteta1 =
                (ztxi + ct * jumptxi) / euclidean * ct * txi[dim] * _colcurr->second * zteta;
            double valtxi2 =
                (zteta + ct * jumpteta) / euclidean * ct * teta[dim] * _colcurr->second * ztxi;
            double valteta2 =
                (zteta + ct * jumpteta) / euclidean * ct * teta[dim] * _colcurr->second * zteta;

            // do not assemble zeros into matrix
            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
              {
                if (abs((valtxi1 + valtxi2) * txi[j]) > 1.e-12)
                  linslipDISglobal.assemble((valtxi1 + valtxi2) * txi[j], cnode->dofs()[j], col);
                if (Interface::n_dim() == 3)
                  if (abs((valteta1 + valteta2) * teta[j]) > 1.e-12)
                    linslipDISglobal.assemble(
                        (valteta1 + valteta2) * teta[j], cnode->dofs()[j], col);
              }
            }
            else
            {
              if (abs(valtxi1) > 1.0e-12) linslipDISglobal.assemble(valtxi1, row[0], col);
              if (abs(valteta1) > 1.0e-12) linslipDISglobal.assemble(valteta1, row[1], col);
              if (abs(valtxi2) > 1.0e-12) linslipDISglobal.assemble(valtxi2, row[0], col);
              if (abs(valteta2) > 1.0e-12) linslipDISglobal.assemble(valteta2, row[1], col);
            }
          }
        }

        // loop over dimensions
        for (int dim = 0; dim < cnode->num_dof(); ++dim)
        {
          // loop over all entries of the current derivative map (jump)
          for (_colcurr = derivjump[dim].begin(); _colcurr != derivjump[dim].end(); ++_colcurr)
          {
            int col = _colcurr->first;

            // std::cout << "val " << colcurr->second << std::endl;
            double valtxi = (-1) * (frbound)*ct * txi[dim] * _colcurr->second;
            double valteta = (-1) * (frbound)*ct * teta[dim] * _colcurr->second;

            // do not assemble zeros into matrix
            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(valtxi * txi[j]) > 1.e-12)
                  linslipDISglobal.assemble(valtxi * txi[j], cnode->dofs()[j], col);
            }
            else if (abs(valtxi) > 1.0e-12)
              linslipDISglobal.assemble(valtxi, row[0], col);

            if (Interface::n_dim() == 3)
            {
              if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
              {
                for (int j = 0; j < Interface::n_dim(); j++)
                  if (abs(valteta * teta[j]) > 1.e-12)
                    linslipDISglobal.assemble(valteta * teta[j], cnode->dofs()[j], col);
              }
              else if (abs(valteta) > 1.0e-12)
                linslipDISglobal.assemble(valteta, row[1], col);
            }
          }
        }
        /*** 1 ********************************* euclidean.deriv(T).z ***/
        // loop over dimensions
        for (int j = 0; j < Interface::n_dim(); ++j)
        {
          // loop over all entries of the current derivative map (txi)
          for (colcurr = dtximap[j].begin(); colcurr != dtximap[j].end(); ++colcurr)
          {
            int col = colcurr->first;
            double val = euclidean * (colcurr->second) * z[j];

            // do not assemble zeros into s matrix
            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(val * txi[j]) > 1.e-12)
                  linslipDISglobal.assemble(val * txi[j], cnode->dofs()[j], col);
            }
            else if (abs(val) > 1.0e-12)
              linslipDISglobal.assemble(val, row[0], col);
          }

          if (Interface::n_dim() == 3)
          {
            // loop over all entries of the current derivative map (teta)
            for (colcurr = dtetamap[j].begin(); colcurr != dtetamap[j].end(); ++colcurr)
            {
              int col = colcurr->first;
              double val = euclidean * (colcurr->second) * z[j];

              // do not assemble zeros into s matrix
              if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
              {
                for (int j = 0; j < Interface::n_dim(); j++)
                  if (abs(val * teta[j]) > 1.e-12)
                    linslipDISglobal.assemble(val * teta[j], cnode->dofs()[j], col);
              }
              else if (abs(val) > 1.0e-12)
                linslipDISglobal.assemble(val, row[1], col);
            }
          }
        }

        /*** 2 ********************* deriv(euclidean).deriv(T).z.ztan ***/
        // loop over dimensions
        for (int j = 0; j < Interface::n_dim(); ++j)
        {
          // loop over all entries of the current derivative map (txi)
          for (colcurr = dtximap[j].begin(); colcurr != dtximap[j].end(); ++colcurr)
          {
            int col = colcurr->first;
            double valtxi = (ztxi + ct * jumptxi) / euclidean * (colcurr->second) * z[j] * ztxi;
            double valteta = (ztxi + ct * jumptxi) / euclidean * (colcurr->second) * z[j] * zteta;

            // do not assemble zeros into matrix
            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(valtxi * txi[j]) > 1.e-12)
                  linslipDISglobal.assemble(valtxi * txi[j], cnode->dofs()[j], col);
            }
            else if (abs(valtxi) > 1.0e-12)
              linslipDISglobal.assemble(valtxi, row[0], col);

            if (Interface::n_dim() == 3)
            {
              if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
              {
                for (int j = 0; j < Interface::n_dim(); j++)
                  if (abs(valteta * teta[j]) > 1.e-12)
                    linslipDISglobal.assemble(valteta * teta[j], cnode->dofs()[j], col);
              }
              else if (abs(valteta) > 1.0e-12)
                linslipDISglobal.assemble(valteta, row[1], col);
            }
          }

          if (Interface::n_dim() == 3)
          {
            // 3D loop over all entries of the current derivative map (teta)
            for (colcurr = dtetamap[j].begin(); colcurr != dtetamap[j].end(); ++colcurr)
            {
              int col = colcurr->first;
              double valtxi = (zteta + ct * jumpteta) / euclidean * (colcurr->second) * z[j] * ztxi;
              double valteta =
                  (zteta + ct * jumpteta) / euclidean * (colcurr->second) * z[j] * zteta;

              // do not assemble zeros into matrix
              if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
              {
                for (int j = 0; j < Interface::n_dim(); j++)
                  if (abs(valtxi * txi[j]) > 1.e-12)
                    linslipDISglobal.assemble(valtxi * txi[j], cnode->dofs()[j], col);
              }
              else if (abs(valtxi) > 1.0e-12)
                linslipDISglobal.assemble(valtxi, row[0], col);

              if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
              {
                for (int j = 0; j < Interface::n_dim(); j++)
                  if (abs(valteta * teta[j]) > 1.e-12)
                    linslipDISglobal.assemble(valteta * teta[j], cnode->dofs()[j], col);
              }
              else if (abs(valteta) > 1.0e-12)
                linslipDISglobal.assemble(valteta, row[1], col);
            }
          }
        }

        // loop over dimensions
        for (int j = 0; j < Interface::n_dim(); ++j)
        {
          // loop over all entries of the current derivative map (txi)
          for (colcurr = dtximap[j].begin(); colcurr != dtximap[j].end(); ++colcurr)
          {
            int col = colcurr->first;
            double valtxi =
                (ztxi + ct * jumptxi) / euclidean * ct * (colcurr->second) * jump[j] * ztxi;
            double valteta =
                (ztxi + ct * jumptxi) / euclidean * ct * (colcurr->second) * jump[j] * zteta;

            // do not assemble zeros into s matrix
            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(valtxi * txi[j]) > 1.e-12)
                  linslipDISglobal.assemble(valtxi * txi[j], cnode->dofs()[j], col);
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(valteta * teta[j]) > 1.e-12)
                  linslipDISglobal.assemble(valteta * teta[j], cnode->dofs()[j], col);
            }
            else
            {
              if (abs(valtxi) > 1.0e-12) linslipDISglobal.assemble(valtxi, row[0], col);
              if (abs(valteta) > 1.0e-12) linslipDISglobal.assemble(valteta, row[1], col);
            }
          }

          if (Interface::n_dim() == 3)
          {
            // loop over all entries of the current derivative map (teta)
            for (colcurr = dtetamap[j].begin(); colcurr != dtetamap[j].end(); ++colcurr)
            {
              int col = colcurr->first;
              double valtxi =
                  (zteta + ct * jumpteta) / euclidean * ct * (colcurr->second) * jump[j] * ztxi;
              double valteta =
                  (zteta + ct * jumpteta) / euclidean * ct * (colcurr->second) * jump[j] * zteta;

              // do not assemble zeros into matrix
              if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
              {
                for (int j = 0; j < Interface::n_dim(); j++)
                  if (abs(valtxi * txi[j]) > 1.e-12)
                    linslipDISglobal.assemble(valtxi * txi[j], cnode->dofs()[j], col);
                for (int j = 0; j < Interface::n_dim(); j++)
                  if (abs(valteta * teta[j]) > 1.e-12)
                    linslipDISglobal.assemble(valteta * teta[j], cnode->dofs()[j], col);
              }
              else
              {
                if (abs(valtxi) > 1.0e-12) linslipDISglobal.assemble(valtxi, row[0], col);
                if (abs(valteta) > 1.0e-12) linslipDISglobal.assemble(valteta, row[1], col);
              }
            }
          }
        }
        /*** 4 ************************** (frcoeff*znor).deriv(T).z ***/
        // loop over all dimensions
        for (int j = 0; j < Interface::n_dim(); ++j)
        {
          // loop over all entries of the current derivative map (txi)
          for (colcurr = dtximap[j].begin(); colcurr != dtximap[j].end(); ++colcurr)
          {
            int col = colcurr->first;
            double val = (-1) * (frbound) * (colcurr->second) * z[j];

            // do not assemble zeros into matrix
            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(val * txi[j]) > 1.e-12)
                  linslipDISglobal.assemble(val * txi[j], cnode->dofs()[j], col);
            }
            else if (abs(val) > 1.0e-12)
              linslipDISglobal.assemble(val, row[0], col);
          }

          if (Interface::n_dim() == 3)
          {
            // loop over all entries of the current derivative map (teta)
            for (colcurr = dtetamap[j].begin(); colcurr != dtetamap[j].end(); ++colcurr)
            {
              int col = colcurr->first;
              double val = (-1.0) * (frbound) * (colcurr->second) * z[j];

              // do not assemble zeros into matrix
              if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
              {
                for (int j = 0; j < Interface::n_dim(); j++)
                  if (abs(val * teta[j]) > 1.e-12)
                    linslipDISglobal.assemble(val * teta[j], cnode->dofs()[j], col);
              }
              else if (abs(val) > 1.0e-12)
                linslipDISglobal.assemble(val, row[1], col);
            }
          }
        }

        /*** 5 *********************** (frcoeff*znor).deriv(T).jump ***/
        // loop over all dimensions
        for (int j = 0; j < Interface::n_dim(); ++j)
        {
          // loop over all entries of the current derivative map (txi)
          for (colcurr = dtximap[j].begin(); colcurr != dtximap[j].end(); ++colcurr)
          {
            int col = colcurr->first;
            double val = (-1) * (frbound)*ct * (colcurr->second) * jump[j];

            // do not assemble zeros into matrix
            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(val * txi[j]) > 1.e-12)
                  linslipDISglobal.assemble(val * txi[j], cnode->dofs()[j], col);
            }
            else if (abs(val) > 1.0e-12)
              linslipDISglobal.assemble(val, row[0], col);
          }

          if (Interface::n_dim() == 3)
          {
            // loop over all entries of the current derivative map (teta)
            for (colcurr = dtetamap[j].begin(); colcurr != dtetamap[j].end(); ++colcurr)
            {
              int col = colcurr->first;
              double val = (-1) * (frbound)*ct * (colcurr->second) * jump[j];

              // do not assemble zeros into s matrix
              if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
              {
                for (int j = 0; j < Interface::n_dim(); j++)
                  if (abs(val * teta[j]) > 1.e-12)
                    linslipDISglobal.assemble(val * teta[j], cnode->dofs()[j], col);
              }
              else if (abs(val) > 1.0e-12)
                linslipDISglobal.assemble(val, row[1], col);
            }
          }
        }

        /*** 6 ******************* -frcoeff.Deriv(n).z(ztan+ct*utan) ***/
        // loop over all dimensions
        for (int j = 0; j < Interface::n_dim(); ++j)
        {
          // loop over all entries of the current derivative map
          for (colcurr = dnmap[j].begin(); colcurr != dnmap[j].end(); ++colcurr)
          {
            int col = colcurr->first;
            double valtxi = (-1) * (ztxi + ct * jumptxi) * frcoeff * (1. - cn * dval * d_gLM) *
                            (colcurr->second) * z[j];
            double valteta = (-1) * (zteta + ct * jumpteta) * frcoeff * (1. - cn * dval * d_gLM) *
                             (colcurr->second) * z[j];

            // do not assemble zeros into s matrix
            if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
            {
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(valtxi * txi[j]) > 1.e-12)
                  linslipDISglobal.assemble(valtxi * txi[j], cnode->dofs()[j], col);
              for (int j = 0; j < Interface::n_dim(); j++)
                if (abs(valteta * teta[j]) > 1.e-12)
                  linslipDISglobal.assemble(valteta * teta[j], cnode->dofs()[j], col);
            }
            else
            {
              if (abs(valtxi) > 1.0e-12) linslipDISglobal.assemble(valtxi, row[0], col);
              if (abs(valteta) > 1.0e-12) linslipDISglobal.assemble(valteta, row[1], col);
            }
          }
        }

        /*** 7 ****************** deriv (g) ***/
        // prepare assembly
        std::map<int, double>& dgmap = cnode->data().get_deriv_g();

        // loop over all entries of the current derivative map
        for (_colcurr = dgmap.begin(); _colcurr != dgmap.end(); ++_colcurr)
        {
          int col = _colcurr->first;
          double valtxi = frcoeff * cn * (_colcurr->second) * (ztxi + ct * jumptxi);
          double valteta = frcoeff * cn * (_colcurr->second) * (zteta + ct * jumpteta);

          // do not assemble zeros into matrix
          if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
          {
            for (int j = 0; j < Interface::n_dim(); j++)
              if (abs(valtxi * txi[j]) > 1.e-12)
                linslipDISglobal.assemble(valtxi * txi[j], cnode->dofs()[j], col);
            for (int j = 0; j < Interface::n_dim(); j++)
              if (abs(valteta * teta[j]) > 1.e-12)
                linslipDISglobal.assemble(valteta * teta[j], cnode->dofs()[j], col);
          }
          else
          {
            if (abs(valtxi) > 1.0e-12) linslipDISglobal.assemble(valtxi, row[0], col);
            if (abs(valteta) > 1.0e-12) linslipDISglobal.assemble(valteta, row[1], col);
          }
        }

        std::map<int, double>& dd = cnode->data().get_deriv_d()[cnode->id()];
        for (auto p = dd.begin(); p != dd.end(); ++p)
        {
          int col = p->first;
          double valtxi = frcoeff * cn * gLM * p->second * (ztxi + ct * jumptxi);
          double valteta = frcoeff * cn * gLM * p->second * (zteta + ct * jumpteta);

          // do not assemble zeros into matrix
          if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
          {
            for (int j = 0; j < Interface::n_dim(); j++)
              if (abs(valtxi * txi[j]) > 1.e-12)
                linslipDISglobal.assemble(valtxi * txi[j], cnode->dofs()[j], col);
            for (int j = 0; j < Interface::n_dim(); j++)
              if (abs(valteta * teta[j]) > 1.e-12)
                linslipDISglobal.assemble(valteta * teta[j], cnode->dofs()[j], col);
          }
          else
          {
            if (abs(valtxi) > 1.0e-12) linslipDISglobal.assemble(valtxi, row[0], col);
            if (abs(valteta) > 1.0e-12) linslipDISglobal.assemble(valteta, row[1], col);
          }
        }
      }  // if (frcoeff==0.0)
    }  // loop over all slip nodes of the interface
  }  // Coulomb friction

  else
  {
    FOUR_C_THROW("no tresca friction with regularized normal constraint");
  }  // Tresca friction

  return;
}

/*---------------------------------------------------------------------------*
 |  Assemble normal coupling weighted condition for poro contact   ager 07/14|
 *--------------------------------------------------------------------------*/
void CONTACT::Interface::assemble_normal_coupling(Core::LinAlg::Vector<double>& gglobal)
{
  // loop over proc's slave nodes of the interface for assembly
  // use standard row map to assemble each node only once
  for (int i = 0; i < activenodes_->num_my_elements(); ++i)
  {
    int gid = activenodes_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Node* mrtnode = dynamic_cast<Node*>(node);

    if (mrtnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
      FOUR_C_THROW("AssembleDMG: Node ownership inconsistency!");

    /**************************************************** nCoup-vector ******/
    if (mrtnode->poro_data().getn_coup() != 0.0)
    {
      double nCoup = mrtnode->poro_data().getn_coup();

      Core::LinAlg::SerialDenseVector gnode(1);
      std::vector<int> lm(1);
      std::vector<int> lmowner(1);

      gnode(0) = nCoup;
      lm[0] = activen_->gid(i);

      lmowner[0] = mrtnode->owner();

      Core::LinAlg::assemble(gglobal, gnode, lm, lmowner);
    }
  }

  return;
}

/*---------------------------------------------------------------------*
 |  Assemble linearisation of normal coupling                          |
 |          weighted condition for poro contact              ager 07/14|
 *--------------------------------------------------------------------*/
void CONTACT::Interface::assemble_normal_coupling_linearisation(Core::LinAlg::SparseMatrix& sglobal,
    Coupling::Adapter::Coupling& coupfs, bool AssembleVelocityLin)
{
  // nothing to do if no active nodes
  if (activenodes_ == nullptr) return;

  std::shared_ptr<const Core::LinAlg::Map> MasterDofMap_full;
  std::shared_ptr<const Core::LinAlg::Map> PermSlaveDofMap_full;

  if (AssembleVelocityLin)
  {
    // store map on all processors, simple but expensive
    MasterDofMap_full = Core::LinAlg::allreduce_e_map(*coupfs.master_dof_map());
    PermSlaveDofMap_full = Core::LinAlg::allreduce_e_map(*coupfs.perm_slave_dof_map());
  }

  for (int i = 0; i < activenodes_->num_my_elements(); ++i)
  {
    int gid = activenodes_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Node* cnode = dynamic_cast<Node*>(node);

    if (cnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
      FOUR_C_THROW("AssembleS: Node ownership inconsistency!");

    std::map<int, double>::iterator colcurr;
    int row = activen_->gid(i);

    std::map<int, double>& dgmap = cnode->poro_data().get_derivn_coup();
    if (AssembleVelocityLin)
    {  // Assign fluid velocity linearization to matrix

      dgmap = cnode->poro_data().get_vel_derivn_coup();
      for (colcurr = dgmap.begin(); colcurr != dgmap.end(); ++colcurr)
      {
        int col = PermSlaveDofMap_full->gid(MasterDofMap_full->lid(colcurr->first));
        double val = colcurr->second;

        // do not assemble zeros into s matrix
        if (abs(val) > 1.0e-12) sglobal.assemble(val, row, col);
      }
      // Assign pressure linearization to matrix
      dgmap = cnode->poro_data().get_pres_derivn_coup();
      for (colcurr = dgmap.begin(); colcurr != dgmap.end(); ++colcurr)
      {
        int col =
            PermSlaveDofMap_full->gid(MasterDofMap_full->lid(colcurr->first)) + Interface::n_dim();
        double val = colcurr->second;

        // do not assemble zeros into s matrix
        if (abs(val) > 1.0e-12) sglobal.assemble(val, row, col);
      }
    }
    else
    {  // Assign skeleton displacement linearization to matrix
      for (colcurr = dgmap.begin(); colcurr != dgmap.end(); ++colcurr)
      {
        int col = colcurr->first;
        double val = colcurr->second;

        // do not assemble zeros into s matrix
        if (abs(val) > 1.0e-12) sglobal.assemble(val, row, col);
      }
    }
  }
  return;
}

/*------------------------------------------------------------------------*
 | Derivative of D-matrix multiplied with a slave dof vector              |
 *------------------------------------------------------------------------*/
void CONTACT::Interface::assemble_coup_lin_d(
    Core::LinAlg::SparseMatrix& CoupLin, const std::shared_ptr<Core::LinAlg::Vector<double>> x)
{
  // we have: D_jk,c with j = Slave dof
  //                 with k = Displacement slave dof
  //                 with c = Displacement slave or master dof
  // we compute (LinD)_kc = D_jk,c * x_j

  for (int j = 0; j < snoderowmap_->num_my_elements(); ++j)
  {
    int gid = snoderowmap_->gid(j);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Node* cnode = dynamic_cast<Node*>(node);

    // Mortar matrix D derivatives
    std::map<int, std::map<int, double>>& dderiv = cnode->data().get_deriv_d();

    // get sizes and iterator start
    int slavesize = (int)dderiv.size();
    std::map<int, std::map<int, double>>::iterator scurr = dderiv.begin();

    /********************************************** LinDMatrix **********/
    // loop over all DISP slave nodes in the DerivD-map of the current slave node
    for (int k = 0; k < slavesize; ++k)
    {
      int sgid = scurr->first;
      ++scurr;

      Core::Nodes::Node* snode = idiscret_->g_node(sgid);
      if (!snode) FOUR_C_THROW("Cannot find node with gid %", sgid);
      Node* csnode = dynamic_cast<Node*>(snode);  // current slave node

      // Mortar matrix D derivatives
      std::map<int, double>& thisdderiv = cnode->data().get_deriv_d()[sgid];
      int mapsize = (int)(thisdderiv.size());

      if (cnode->num_dof() != csnode->num_dof())
        FOUR_C_THROW("Mortar Nodes on interface must have same number of dofs!");

      // inner product D_{jk,c} * z_j for index j
      for (int prodj = 0; prodj < cnode->num_dof(); ++prodj)
      {
        int row = csnode->dofs()[prodj];
        std::map<int, double>::iterator scolcurr = thisdderiv.begin();

        // loop over all directional derivative entries
        for (int c = 0; c < mapsize; ++c)
        {
          int col = scolcurr->first;
          int slavedofgid = cnode->dofs()[prodj];

          int slavedoflid = x->get_map().lid(slavedofgid);
          if (slavedoflid < 0) FOUR_C_THROW("invalid slave dof lid");
          double val = x->local_values_as_span()[slavedoflid] * (scolcurr->second);

          ++scolcurr;


          //************   ASSEMBLY INTO THE MATRIX    **********************
          if (abs(val) > 1.0e-12) CoupLin.fe_assemble(val, row, col);
        }
        // check for completeness of DerivD-Derivatives-iteration
        if (scolcurr != thisdderiv.end())
          FOUR_C_THROW("AssembleCoupLin: Not all derivative entries of Lin(D*z_s) considered!");
      }
    }

    if (scurr != dderiv.end())
      FOUR_C_THROW("AssembleCoupLin: Not all connected slave nodes to Lin(D*z_s) considered!");
  }

  return;
}

/*-----------------------------------------------------------------------------------*
 | Derivative of transposed M-matrix multiplied with a slave dof vector  seitz 01/18 |
 *-----------------------------------------------------------------------------------*/
void CONTACT::Interface::assemble_coup_lin_m(
    Core::LinAlg::SparseMatrix& CoupLin, const std::shared_ptr<Core::LinAlg::Vector<double>> x)
{
  // we have: M_jl,c with j = Slave dof
  //                 with l = Displacement master dof
  //                 with c = Displacement slave or master dof
  // we compute (CoupLin)_lc = M_jl,c * x_j

  // loop over all slave nodes (row map)
  for (int j = 0; j < snoderowmap_->num_my_elements(); ++j)
  {
    int gid = snoderowmap_->gid(j);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Node* cnode = dynamic_cast<Node*>(node);

    // Mortar matrix M derivatives
    std::map<int, std::map<int, double>>& mderiv = cnode->data().get_deriv_m();

    // get sizes and iterator start
    int mastersize = (int)mderiv.size();
    std::map<int, std::map<int, double>>::iterator mcurr = mderiv.begin();

    /********************************************** LinMMatrix **********/
    // loop over all master nodes in the DerivM-map of the current LM slave node
    for (int l = 0; l < mastersize; ++l)
    {
      int mgid = mcurr->first;
      ++mcurr;

      Core::Nodes::Node* mnode = idiscret_->g_node(mgid);
      if (!mnode) FOUR_C_THROW("Cannot find node with gid %", mgid);
      Node* cmnode = dynamic_cast<Node*>(mnode);

      // Mortar matrix M derivatives
      std::map<int, double>& thismderiv = cnode->data().get_deriv_m()[mgid];
      int mapsize = (int)(thismderiv.size());

      if (cnode->num_dof() != cmnode->num_dof())
        FOUR_C_THROW("Mortar Nodes on interface must have same number of dofs!");

      // inner product M_{jl,c} * z_j for index j
      for (int prodj = 0; prodj < cmnode->num_dof(); ++prodj)
      {
        int row = cmnode->dofs()[prodj];
        std::map<int, double>::iterator mcolcurr = thismderiv.begin();

        // loop over all directional derivative entries
        for (int c = 0; c < mapsize; ++c)
        {
          int col = mcolcurr->first;
          int slavedofgid = cnode->dofs()[prodj];

          int slavedoflid = x->get_map().lid(slavedofgid);
          double val = (*x).local_values_as_span()[slavedoflid] * (mcolcurr->second);

          ++mcolcurr;

          ///************   ASSEMBLY INTO THE MATRIX    **********************
          if (abs(val) > 1.0e-12) CoupLin.fe_assemble(val, row, col);
        }

        // check for completeness of DerivM-Derivatives-iteration
        if (mcolcurr != thismderiv.end())
          FOUR_C_THROW("AssembleCoupLin: Not all derivative entries of DerivM considered!");
      }
    }  // loop over all master nodes, connected to this slave node => Lin(M*z_s) finished

    // check for completeness of DerivM-Master-iteration
    if (mcurr != mderiv.end())
      FOUR_C_THROW("AssembleCoupLin: Not all master entries of Lin(M*z_s) considered!");

    /*********************************************************************/
    /*******************       Finished with Lin(M*z_s)         **********/
    /*********************************************************************/
  }
}
FOUR_C_NAMESPACE_CLOSE
