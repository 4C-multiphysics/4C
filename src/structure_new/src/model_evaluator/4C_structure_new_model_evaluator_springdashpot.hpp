// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_MODEL_EVALUATOR_SPRINGDASHPOT_HPP
#define FOUR_C_STRUCTURE_NEW_MODEL_EVALUATOR_SPRINGDASHPOT_HPP

#include "4C_config.hpp"

#include "4C_constraint_springdashpot.hpp"
#include "4C_structure_new_model_evaluator_generic.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::IO
{
  class DiscretizationVisualizationWriterMesh;
}

namespace Core::LinAlg
{
  class SparseMatrix;
}

namespace Solid
{
  namespace ModelEvaluator
  {
    class SpringDashpot : public Generic
    {
     public:
      //! constructor
      SpringDashpot();

      void setup() override;
      [[nodiscard]] Solid::ModelType type() const override { return Solid::model_springdashpot; }
      void reset(const Core::LinAlg::Vector<double>& x) override;

      void evaluate_force() override;

      void evaluate_stiff() override;

      void evaluate_force_stiff() override;

      void pre_evaluate() override {}

      void post_evaluate() override {}

      void assemble_force(Core::LinAlg::Vector<double>& f, const double& timefac_np) const override;

      //! Assemble the jacobian at \f$t_{n+1}\f$
      void assemble_jacobian(
          Core::LinAlg::SparseOperator& jac, const double& timefac_np) const override;

      void write_restart(
          Core::IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const override;

      void read_restart(Core::IO::DiscretizationReader& ioreader) override;

      void predict(const Solid::PredEnum& pred_type) override {}

      void run_pre_compute_x(const Core::LinAlg::Vector<double>& xold,
          Core::LinAlg::Vector<double>& dir_mutable, const NOX::Nln::Group& curr_grp) override
      {
      }

      void run_post_compute_x(const Core::LinAlg::Vector<double>& xold,
          const Core::LinAlg::Vector<double>& dir,
          const Core::LinAlg::Vector<double>& xnew) override
      {
      }

      void run_post_iterate(const ::NOX::Solver::Generic& solver) override {}

      void update_step_state(const double& timefac_n) override;

      void update_step_element() override {}

      void determine_stress_strain() override {}

      void determine_energy() override {}

      void output_step_state(Core::IO::DiscretizationWriter& iowriter) const override {}

      void runtime_output_step_state() const override;

      void reset_step_state() override;

      std::shared_ptr<const Core::LinAlg::Map> get_block_dof_row_map_ptr() const override;

      std::shared_ptr<const Core::LinAlg::Vector<double>> get_current_solution_ptr() const override;


      std::shared_ptr<const Core::LinAlg::Vector<double>> get_last_time_step_solution_ptr()
          const override;

     private:
      //! all spring dashpot instances
      std::vector<std::shared_ptr<Constraints::SpringDashpot>> springs_;

      //! structural displacement at \f$t_{n+1}\f$
      std::shared_ptr<const Core::LinAlg::Vector<double>> disnp_ptr_;

      //! structural velocity at \f$t_{n+1}\f$
      std::shared_ptr<const Core::LinAlg::Vector<double>> velnp_ptr_;

      //! structural stiffness matrix
      std::shared_ptr<Core::LinAlg::SparseMatrix> stiff_spring_ptr_;

      //! spring forces at \f$t_{n+1}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>> fspring_np_ptr_;

      //! spring dashpot runtime output writer
      std::unique_ptr<Core::IO::DiscretizationVisualizationWriterMesh>
          spring_dashpot_vtu_writer_ptr_;
    };

  }  // namespace ModelEvaluator
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
