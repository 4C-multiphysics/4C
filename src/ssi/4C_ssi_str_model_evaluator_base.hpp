// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SSI_STR_MODEL_EVALUATOR_BASE_HPP
#define FOUR_C_SSI_STR_MODEL_EVALUATOR_BASE_HPP

#include "4C_config.hpp"

#include "4C_linalg_vector.hpp"
#include "4C_structure_new_model_evaluator_generic.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Solid::ModelEvaluator
{
  class BaseSSI : public Generic
  {
   public:
    bool assemble_force(Core::LinAlg::Vector<double>& f, const double& timefac_np) const override
    {
      return true;
    }

    bool assemble_jacobian(
        Core::LinAlg::SparseOperator& jac, const double& timefac_np) const override
    {
      return true;
    }

    void determine_energy() override {}

    void determine_optional_quantity() override {}

    void determine_stress_strain() override;

    bool evaluate_force() override { return true; }

    bool evaluate_force_stiff() override { return true; }

    bool evaluate_stiff() override { return true; }

    [[nodiscard]] std::shared_ptr<const Core::LinAlg::Map> get_block_dof_row_map_ptr()
        const override;

    [[nodiscard]] std::shared_ptr<const Core::LinAlg::Vector<double>> get_current_solution_ptr()
        const override
    {
      FOUR_C_THROW("Not implemented!");
    }

    [[nodiscard]] std::shared_ptr<const Core::LinAlg::Vector<double>>
    get_last_time_step_solution_ptr() const override
    {
      FOUR_C_THROW("Not implemented!");
    }

    [[nodiscard]] std::shared_ptr<const Core::LinAlg::Vector<double>>
    get_mechanical_stress_state_n() const override
    {
      return mechanical_stress_state_n_;
    }

    [[nodiscard]] std::shared_ptr<const Core::LinAlg::Vector<double>>
    get_mechanical_stress_state_np() const override
    {
      return mechanical_stress_state_np_;
    }

    void output_step_state(Core::IO::DiscretizationWriter& iowriter) const override {}

    void post_evaluate() override {}

    void post_output() override {}

    void predict(const Inpar::Solid::PredEnum& pred_type) override {}

    void pre_evaluate() override {}

    void read_restart(Core::IO::DiscretizationReader& ioreader) override;

    void reset(const Core::LinAlg::Vector<double>& x) override {}

    void reset_step_state() override {}

    void run_post_compute_x(const Core::LinAlg::Vector<double>& xold,
        const Core::LinAlg::Vector<double>& dir, const Core::LinAlg::Vector<double>& xnew) override
    {
    }

    void run_post_iterate(const ::NOX::Solver::Generic& solver) override {}

    void run_pre_compute_x(const Core::LinAlg::Vector<double>& xold,
        Core::LinAlg::Vector<double>& dir_mutable, const NOX::Nln::Group& curr_grp) override
    {
    }

    void setup() override;

    [[nodiscard]] Inpar::Solid::ModelType type() const override
    {
      return Inpar::Solid::model_basic_coupling;
    }

    void update_step_element() override {}

    void update_step_state(const double& timefac_n) override
    {
      if (mechanical_stress_state_n_ != nullptr)
        mechanical_stress_state_n_->update(1.0, *mechanical_stress_state_np_, 0.0);
    }

    void write_restart(
        Core::IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const override;

   private:
    //! mechanical stress state at time n
    std::shared_ptr<Core::LinAlg::Vector<double>> mechanical_stress_state_n_;
    //! mechanical stress state at time np
    std::shared_ptr<Core::LinAlg::Vector<double>> mechanical_stress_state_np_;
  };
}  // namespace Solid::ModelEvaluator
FOUR_C_NAMESPACE_CLOSE

#endif