// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_MODEL_EVALUATOR_MULTISCALE_HPP
#define FOUR_C_STRUCTURE_NEW_MODEL_EVALUATOR_MULTISCALE_HPP

#include "4C_config.hpp"

#include "4C_io_discretization_visualization_writer_mesh.hpp"
#include "4C_io_visualization_parameters.hpp"
#include "4C_structure_new_model_evaluator_generic.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Solid
{
  namespace ModelEvaluator
  {
    class Multiscale : public Generic
    {
     public:
      //! constructor
      Multiscale() = default;

      void setup() override;

      Inpar::Solid::ModelType type() const override { return Inpar::Solid::model_multiscale; }

      void reset(const Core::LinAlg::Vector<double>& x) override {};

      bool evaluate_force() override { return true; };

      bool evaluate_stiff() override { return true; };

      bool evaluate_force_stiff() override { return true; };

      void pre_evaluate() override {};

      void post_evaluate() override {};

      bool assemble_force(Core::LinAlg::Vector<double>& f, const double& timefac_np) const override
      {
        return true;
      };

      bool assemble_jacobian(
          Core::LinAlg::SparseOperator& jac, const double& timefac_np) const override
      {
        return true;
      };

      void write_restart(
          Core::IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const override
      {
        write_restart(iowriter);
      };

      /*! \brief write model specific restart
       *
       *  \param iowriter            (in) : output writer
       */
      void write_restart(Core::IO::DiscretizationWriter& iowriter) const;

      void read_restart(Core::IO::DiscretizationReader& ioreader) override;

      void post_setup() override;

      void predict(const Inpar::Solid::PredEnum& pred_type) override {};

      void run_pre_compute_x(const Core::LinAlg::Vector<double>& xold,
          Core::LinAlg::Vector<double>& dir_mutable, const NOX::Nln::Group& curr_grp) override {};

      void run_post_compute_x(const Core::LinAlg::Vector<double>& xold,
          const Core::LinAlg::Vector<double>& dir,
          const Core::LinAlg::Vector<double>& xnew) override
      {
      }

      void run_post_iterate(const ::NOX::Solver::Generic& solver) override {};

      void update_step_state(const double& timefac_n) override {};

      void update_step_element() override {};

      void determine_stress_strain() override {};

      void determine_energy() override {};

      void determine_optional_quantity() override;

      //! derived
      void runtime_pre_output_step_state() override;

      //! derived
      void runtime_output_step_state() const override;

      //! derived
      void output_step_state(Core::IO::DiscretizationWriter& iowriter) const override;

      void reset_step_state() override {};

      void post_time_loop() override;

      std::shared_ptr<const Epetra_Map> get_block_dof_row_map_ptr() const override
      {
        return nullptr;
      };

      std::shared_ptr<const Core::LinAlg::Vector<double>> get_current_solution_ptr() const override
      {
        return nullptr;
      };

      std::shared_ptr<const Core::LinAlg::Vector<double>> get_last_time_step_solution_ptr()
          const override
      {
        return nullptr;
      };

      void post_output() override {};

     private:
      //! visualization parameters
      Core::IO::VisualizationParameters macro_visualization_params_;
    };
  }  // namespace ModelEvaluator
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
