// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_adapter_ale_fpsi.hpp"

#include "4C_ale_utils_mapextractor.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Adapter::AleFpsiWrapper::AleFpsiWrapper(Teuchos::RCP<Ale> ale) : AleWrapper(ale)
{
  // create the FSI interface
  interface_ = Teuchos::make_rcp<ALE::Utils::MapExtractor>();
  interface_->setup(*discretization(), true);  // create overlapping maps for fpsi problem

  return;
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Adapter::AleFpsiWrapper::apply_interface_displacements(
    Teuchos::RCP<const Core::LinAlg::Vector<double>> idisp)
{
  interface_->insert_fpsi_cond_vector(*idisp, *write_access_dispnp());
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Adapter::AleFpsiWrapper::apply_fsi_interface_displacements(
    Teuchos::RCP<const Core::LinAlg::Vector<double>> idisp)
{
  interface_->insert_fsi_cond_vector(*idisp, *write_access_dispnp());
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Teuchos::RCP<const ALE::Utils::MapExtractor> Adapter::AleFpsiWrapper::interface() const
{
  return interface_;
}

FOUR_C_NAMESPACE_CLOSE
