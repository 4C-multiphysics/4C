// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_ADAPTER_FLD_FBI_WRAPPER_HPP
#define FOUR_C_ADAPTER_FLD_FBI_WRAPPER_HPP

#include "4C_config.hpp"

#include "4C_adapter_fld_fluid_fsi.hpp"
#include "4C_fluid_meshtying.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class Solver;
  class SparseOperator;
}  // namespace Core::LinAlg

namespace Core::IO
{
  class DiscretizationWriter;
}


namespace Adapter
{
  /*! \brief Fluid field adapter for fluid beam interaction
   *
   *
   *  Can only be used in conjunction with #FLD::FluidImplicitTimeInt
   */
  class FluidFBI : public FluidFSI
  {
   public:
    /// Constructor
    FluidFBI(std::shared_ptr<Fluid> fluid, std::shared_ptr<Core::FE::Discretization> dis,
        std::shared_ptr<Core::LinAlg::Solver> solver,
        std::shared_ptr<Teuchos::ParameterList> params,
        std::shared_ptr<Core::IO::DiscretizationWriter> output, bool isale, bool dirichletcond);

    /** \brief Pass in additional contributions from coupling terms for the system matrix
     *
     * To enforce weak dirichlet conditions as they arise from meshtying for example, such
     * contributions can be handed to the fluid, which will store the pointer on the coupling
     * contributions to assemble them into the system matrix in each Newton iteration.
     *
     * \param[in] matrix (size fluid_dof x fluid_dof) matrix containing weak dirichlet entries that
     * need to be assembled into the overall fluid system matrix
     */
    virtual void set_coupling_contributions(
        std::shared_ptr<const Core::LinAlg::SparseOperator> matrix);

    /**
     * \brief Resets the external forces acting on the fluid to zero
     */
    virtual void reset_external_forces();

    virtual std::shared_ptr<const FLD::Meshtying> get_meshtying();
  };
}  // namespace Adapter

FOUR_C_NAMESPACE_CLOSE

#endif
