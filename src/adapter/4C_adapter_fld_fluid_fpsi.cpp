// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_adapter_fld_fluid_fpsi.hpp"

#include "4C_adapter_fld_fluid_fsi.hpp"
#include "4C_fluid_implicit_integration.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_fpsi_utils.hpp"

FOUR_C_NAMESPACE_OPEN


/* constructor */
Adapter::FluidFPSI::FluidFPSI(Global::Problem& problem, std::shared_ptr<Fluid> fluid,
    std::shared_ptr<Core::FE::Discretization> dis, std::shared_ptr<Core::LinAlg::Solver> solver,
    std::shared_ptr<Teuchos::ParameterList> params,
    std::shared_ptr<Core::IO::DiscretizationWriter> output, bool isale, bool dirichletcond)
    : FluidFSI(problem, fluid, dis, solver, params, output, isale, dirichletcond),
      fpsiinterface_(std::make_shared<FLD::Utils::MapExtractor>())
{
  return;
}  // constructor


/* initialization */
void Adapter::FluidFPSI::init()
{
  // call base class init
  FluidFSI::init();

  fpsiinterface_->setup(*dis_, true, true);  // Always Create overlapping FPSI Interface

  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Adapter::FluidFPSI::setup_interface(const int target_dofset_number)
{
  // check target_dofset_number
  if (target_dofset_number != 0) FOUR_C_THROW("target_dofset_number is supposed to be 0 here");

  interface_->setup(*dis_, false, true);  // create overlapping maps for fpsi problem
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Adapter::FluidFPSI::use_block_matrix(
    bool splitmatrix, std::shared_ptr<FPSI::Utils::MapExtractor> const& shapederivSplitter)
{
  std::shared_ptr<std::set<int>> condelements =
      interface()->conditioned_element_map(*discretization());
  std::shared_ptr<std::set<int>> condelements_shape =
      shapederivSplitter->conditioned_element_map(*discretization());
  fluidimpl_->use_block_matrix(condelements, *interface(), *interface(), condelements_shape,
      *shapederivSplitter, *shapederivSplitter, splitmatrix);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Adapter::FluidFPSI::use_block_matrix(bool splitmatrix)
{
  std::shared_ptr<std::set<int>> condelements =
      interface()->conditioned_element_map(*discretization());
  fluidimpl_->use_block_matrix(condelements, *interface(), *interface(), condelements, *interface(),
      *interface(), splitmatrix);
}

FOUR_C_NAMESPACE_CLOSE
