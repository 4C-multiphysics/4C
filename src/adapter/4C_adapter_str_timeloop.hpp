// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_ADAPTER_STR_TIMELOOP_HPP
#define FOUR_C_ADAPTER_STR_TIMELOOP_HPP

#include "4C_config.hpp"

#include "4C_adapter_str_wrapper.hpp"

FOUR_C_NAMESPACE_OPEN


namespace Adapter
{
  /*! \brief Time loop for stuctural simulations
   *
   *  This is a wrapper for the structural time integration which gives
   *  fine-grained access into the time loop by various pre- and post-operators.
   *
   *  To perform such pre- and post-operations, just derive from this class and
   *  overload the respective pre-/post-operator.
   *
   *  Implementations of pre-/post-operators in this class have to remain empty!
   */
  class StructureTimeLoop : public StructureWrapper
  {
   public:
    /// constructor
    StructureTimeLoop(std::shared_ptr<Structure> structure);

    /// actual time loop
    void integrate() override;

    /// wrapper for things that should be done before updating
    void pre_update() override {};

    /// wrapper for things that should be done after solving the update
    void post_update() override {};
  };

}  // namespace Adapter

FOUR_C_NAMESPACE_CLOSE

#endif
