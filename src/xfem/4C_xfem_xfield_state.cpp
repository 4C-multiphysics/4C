// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_xfem_xfield_state.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_xfem_discretization.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
XFEM::XFieldState::XFieldState()
    : isinit_(false),
      issetup_(false),
      wizard_(Teuchos::null),
      condition_manager_(Teuchos::null),
      xdofset_(Teuchos::null),
      xfield_discret_ptr_(Teuchos::null),
      field_discret_ptr_(Teuchos::null)
{
  // intentionally left blank
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void XFEM::XFieldState::init(const Teuchos::RCP<XFEM::ConditionManager>& condition_manager,
    const Teuchos::RCP<Cut::CutWizard>& wizard, const Teuchos::RCP<XFEM::XFEMDofSet>& xdofset,
    const Teuchos::RCP<Core::FE::Discretization>& xfielddiscret,
    const Teuchos::RCP<Core::FE::Discretization>& fielddiscret)
{
  // Ensure, that the setup() routines are called afterwards.
  issetup_ = false;

  condition_manager_ = condition_manager;
  wizard_ = wizard;

  xdofset_ = xdofset;

  // store a pointer to the field discretization
  field_discret_ptr_ = fielddiscret;

  // store a pointer to the xfield discretization
  xfield_discret_ptr_ = xfielddiscret;

  isinit_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void XFEM::XFieldState::set_new_state(const XFEM::XFieldState& xstate)
{
  this->isinit_ = xstate.isinit_;
  this->issetup_ = xstate.issetup_;

  this->condition_manager_ = xstate.condition_manager_;
  this->wizard_ = xstate.wizard_;

  this->xdofset_ = xstate.xdofset_;

  this->field_discret_ptr_ = xstate.field_discret_ptr_;
  this->xfield_discret_ptr_ = xstate.xfield_discret_ptr_;
}

FOUR_C_NAMESPACE_CLOSE
