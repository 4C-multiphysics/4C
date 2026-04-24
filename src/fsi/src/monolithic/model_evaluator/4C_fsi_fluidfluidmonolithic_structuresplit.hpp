// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FSI_FLUIDFLUIDMONOLITHIC_STRUCTURESPLIT_HPP
#define FOUR_C_FSI_FLUIDFLUIDMONOLITHIC_STRUCTURESPLIT_HPP

#include "4C_config.hpp"

#include "4C_fsi_monolithicstructuresplit.hpp"
#include "4C_xfem_input.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Adapter
{
  class FluidFluidFSI;
  class AleXFFsiWrapper;
}  // namespace Adapter

namespace FSI
{
  /// monolithic hybrid FSI algorithm with overlapping interface equations
  /*!
   * Monolithic fluid-fluid FSI with fluid-handled interface motion, employing XFEM and NOX.
   * Structural interface displacements are condensed.

   */
  class FluidFluidMonolithicStructureSplit : public MonolithicStructureSplit
  {
    friend class FSI::FSIResultTest;

   public:
    /// constructor
    explicit FluidFluidMonolithicStructureSplit(
        MPI_Comm comm, Global::Problem& problem, const Teuchos::ParameterList& timeparams);

    /// update subsequent fields, recover the Lagrange multiplier and relax the ALE-mesh
    void update() override;

    /// start a new time step
    void prepare_time_step() override;

   private:
    /// setup of extractor for merged Dirichlet maps
    void setup_dbc_map_extractor() override;

    /// access type-cast pointer to problem-specific fluid-wrapper
    const std::shared_ptr<Adapter::FluidFluidFSI>& fluid_field() { return fluid_; }

    /// access type-cast pointer to problem-specific ALE-wrapper
    const std::shared_ptr<Adapter::AleXFFsiWrapper>& ale_field() { return ale_; }

    /// type-cast pointer to problem-specific fluid-wrapper
    std::shared_ptr<Adapter::FluidFluidFSI> fluid_;

    /// type-cast pointer to problem-specific ALE-wrapper
    std::shared_ptr<Adapter::AleXFFsiWrapper> ale_;
  };
}  // namespace FSI

FOUR_C_NAMESPACE_CLOSE

#endif
