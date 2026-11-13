// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_MODEL_HPP
#define FOUR_C_REDUCED_LUNG_MODEL_HPP

#include "4C_config.hpp"

#include "4C_rebalance.hpp"
#include "4C_reduced_lung_input.hpp"

#include <mpi.h>

#include <functional>
#include <memory>

namespace Teuchos
{
  class ParameterList;
}

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  class OutputControl;
}

namespace Core::Utils
{
  class FunctionManager;
  class ResultTest;
}  // namespace Core::Utils

namespace ReducedLung
{
  /**
   * @brief Everything a reduced-lung model needs, passed in explicitly.
   *
   * The model does not read from Global::Problem, so it can be driven both by the standalone
   * reduced-lung problem and by other fields that couple to it. The model keeps using the
   * referenced parameter lists, output control, and function manager after construction, so they
   * must outlive it.
   */
  struct ModelContext
  {
    ReducedLungParameters parameters;
    MPI_Comm comm;
    Core::Rebalance::RebalanceParameters rebalance_parameters;
    const Teuchos::ParameterList& io_parameters;
    const Teuchos::ParameterList& linear_solver_parameters;
    std::function<const Teuchos::ParameterList&(int)> solver_params_callback;
    std::shared_ptr<Core::IO::OutputControl> output_control_file;
    const Core::Utils::FunctionManager& function_manager;
  };

  /**
   * @brief One reduced-lung tree with its time integration, driven step by step from outside.
   *
   * A time step is split into a trial solve and an explicit acceptance:
   *
   *   model.solve_step(dt);  // may be repeated, e.g. in coupling iterations
   *   model.accept_step();   // accept the last trial solution
   *
   * solve_step() always starts from the last accepted state and leaves it untouched, so calling
   * it repeatedly within one time step needs no state save or restore. Only accept_step() advances
   * the history of the airway and terminal-unit models.
   */
  class Model
  {
   public:
    /**
     * @brief Build the discretization, model blocks, maps, and the configured nonlinear solver.
     */
    explicit Model(const ModelContext& context);

    ~Model();

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) = delete;
    Model& operator=(Model&&) = delete;

    /**
     * @brief Solve for the state at time() + @p dt, starting from the last accepted state.
     *
     * @param dt Step size from the last accepted state.
     * @return Number of nonlinear iterations performed.
     */
    unsigned int solve_step(double dt);

    /**
     * @brief Make the last trial solution the new accepted state and advance time().
     */
    void accept_step();

    /**
     * @brief Time of the last accepted state.
     */
    [[nodiscard]] double time() const;

    /**
     * @brief Write visualization output of the accepted state.
     *
     * Must not be called between solve_step() and accept_step(), where the model holds a trial
     * solution.
     */
    void write_output(int step) const;

    /**
     * @brief Create a result test that checks the model's current state.
     *
     * The result test refers to the model's state when it is evaluated, not when it is created, so
     * evaluate it after accept_step(). The model must outlive it.
     */
    [[nodiscard]] std::shared_ptr<Core::Utils::ResultTest> create_result_test() const;

   private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
  };
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

#endif
