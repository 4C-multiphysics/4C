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

namespace Global
{
  class Problem;
}  // namespace Global

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
    StructureTimeLoop(Global::Problem& problem, std::shared_ptr<Structure> structure);

    /// actual time loop
    int integrate() override;

    /// wrapper for things that should be done before prepare_time_step is called
    void pre_predict() override {};

    /// wrapper for things that should be done before solving the nonlinear iterations
    void pre_solve() override {};

    /// wrapper for things that should be done before updating
    void pre_update() override {};

    /// wrapper for things that should be done after solving the update
    void post_update() override {};

    /// wrapper for things that should be done after the output
    void post_output() override
    {
      // first forward to base class which forwards to structure_->post_output();
      StructureWrapper::post_output();
    };

   private:
    /// explicit global problem context
    Global::Problem& problem_;
  };

}  // namespace Adapter

FOUR_C_NAMESPACE_CLOSE

#endif
