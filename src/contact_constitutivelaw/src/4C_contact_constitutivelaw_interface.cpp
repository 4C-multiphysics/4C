// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_constitutivelaw_interface.hpp"

#include "4C_contact_constitutivelaw_contactconstitutivelaw.hpp"
#include "4C_contact_constitutivelaw_contactconstitutivelaw_parameter.hpp"
#include "4C_contact_input.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
CONTACT::ConstitutivelawInterface::ConstitutivelawInterface(
    const std::shared_ptr<Mortar::InterfaceDataContainer>& interfaceData, const int id,
    MPI_Comm comm, const int dim, const Teuchos::ParameterList& icontact, bool selfcontact,
    const int contactconstitutivelawid)
    : Interface(interfaceData, id, comm, dim, icontact, selfcontact)
{
  std::shared_ptr<CONTACT::CONSTITUTIVELAW::ConstitutiveLaw> coconstlaw =
      CONTACT::CONSTITUTIVELAW::ConstitutiveLaw::factory(contactconstitutivelawid);
  coconstlaw_ = coconstlaw;
}
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::ConstitutivelawInterface::assemble_reg_normal_forces(
    bool& localisincontact, bool& localactivesetchange)
{
  // loop over all slave row nodes on the current interface
  for (int i = 0; i < slave_row_nodes()->num_my_elements(); ++i)
  {
    const int gid = slave_row_nodes()->gid(i);
    Core::Nodes::Node* node = discret().g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %.", gid);
    Node* cnode = dynamic_cast<Node*>(node);

    const int dim = cnode->num_dof();
    const double gap = cnode->data().getg();

    const double kappa = cnode->data().kappa();

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

    // Activate/Deactivate node and notice any change
    if ((cnode->active() == false) && (-coconstlaw_->parameter()->get_offset() - kappa * gap >= 0))
    {
      cnode->active() = true;
      localactivesetchange = true;
    }

    else if ((cnode->active() == true) &&
             (-coconstlaw_->parameter()->get_offset() - kappa * gap < 0))
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
      // Evaluate pressure
      const double pressure = coconstlaw_->evaluate(kappa * gap, cnode);
      // Evaluate pressure derivative
      const double pressurederiv = coconstlaw_->evaluate_derivative(kappa * gap, cnode);

      localisincontact = true;

      double* normal = cnode->mo_data().n();

      // compute lagrange multipliers and store into node
      for (int j = 0; j < dim; ++j) cnode->mo_data().lm()[j] = (lmuzawan - pressure) * normal[j];

      // compute derivatives of lagrange multipliers and store into node
      // contribution of derivative of weighted gap
      std::map<int, double>& derivg = cnode->data().get_deriv_g();
      std::map<int, double>::iterator gcurr;
      // printf("lm=%f\n", -coconstlaw_->evaluate(kappa * gap));

      // contribution of derivative of normal
      std::vector<Core::Gen::Pairedvector<int, double>>& derivn = cnode->data().get_deriv_n();
      Core::Gen::Pairedvector<int, double>::iterator ncurr;

      for (int j = 0; j < dim; ++j)
      {
        for (gcurr = derivg.begin(); gcurr != derivg.end(); ++gcurr)
          cnode->add_deriv_z_value(
              j, gcurr->first, -kappa * pressurederiv * (gcurr->second) * normal[j]);
        for (ncurr = (derivn[j]).begin(); ncurr != (derivn[j]).end(); ++ncurr)
          cnode->add_deriv_z_value(j, ncurr->first, -pressure * ncurr->second);
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
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::ConstitutivelawInterface::assemble_reg_tangent_forces_penalty()
{
  FOUR_C_THROW("Frictional contact not yet implemented for rough surfaces.");
}

FOUR_C_NAMESPACE_CLOSE
