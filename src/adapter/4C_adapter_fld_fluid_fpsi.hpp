// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_ADAPTER_FLD_FLUID_FPSI_HPP
#define FOUR_C_ADAPTER_FLD_FLUID_FPSI_HPP

#include "4C_config.hpp"

#include "4C_adapter_fld_fluid_fsi.hpp"

FOUR_C_NAMESPACE_OPEN

namespace FPSI
{
  namespace Utils
  {
    class MapExtractor;
  }
}  // namespace FPSI

namespace Adapter
{
  /*! \brief Fluid field adapter for fpsi
   *
   *
   *  Can only be used in conjunction with #FLD::FluidImplicitTimeInt
   */
  class FluidFPSI : public FluidFSI
  {
   public:
    /// Constructor
    FluidFPSI(Global::Problem& problem, std::shared_ptr<Fluid> fluid,
        std::shared_ptr<Core::FE::Discretization> dis, std::shared_ptr<Core::LinAlg::Solver> solver,
        std::shared_ptr<Teuchos::ParameterList> params,
        std::shared_ptr<Core::IO::DiscretizationWriter> output, bool isale, bool dirichletcond);

    /// initialize algorithm
    void init() override;

    /// communication object at the interface without pressure dofs for FPSI problems
    std::shared_ptr<FLD::Utils::MapExtractor> const& fpsi_interface() const override
    {
      return fpsiinterface_;
    }

    void setup_interface(const int target_dofset_number = 0) override;

    void use_block_matrix(bool splitmatrix) override;
    virtual void use_block_matrix(
        bool splitmatrix, std::shared_ptr<FPSI::Utils::MapExtractor> const& shapederivSplitter);

   protected:
    /// the interface map setup for fpsi interface
    std::shared_ptr<FLD::Utils::MapExtractor> fpsiinterface_;

  };  // class FluidFPSI
}  // namespace Adapter

FOUR_C_NAMESPACE_CLOSE

#endif
