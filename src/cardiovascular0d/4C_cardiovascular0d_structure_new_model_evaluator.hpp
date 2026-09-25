// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CARDIOVASCULAR0D_STRUCTURE_NEW_MODEL_EVALUATOR_HPP
#define FOUR_C_CARDIOVASCULAR0D_STRUCTURE_NEW_MODEL_EVALUATOR_HPP

#include "4C_config.hpp"

#include "4C_cardiovascular0d_manager.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_structure_new_model_evaluator_generic.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::LinAlg
{
  class SparseMatrix;
}  // namespace Core::LinAlg

namespace Solid
{
  namespace ModelEvaluator
  {
    class Cardiovascular0D : public Generic
    {
     public:
      //! constructor
      Cardiovascular0D();


      void setup() override;

      //! derived
      Solid::ModelType type() const override { return Solid::model_cardiovascular0d; }

      //! derived
      void reset(const Core::LinAlg::Vector<double>& x) override;

      //! derived
      bool evaluate_force() override;

      //! derived
      bool evaluate_stiff() override;

      //! derived
      bool evaluate_force_stiff() override;

      //! derived
      void pre_evaluate() override { return; };

      //! derived
      void post_evaluate() override { return; };

      //! derived
      bool assemble_force(Core::LinAlg::Vector<double>& f, const double& timefac_np) const override;

      //! derived
      bool assemble_jacobian(
          Core::LinAlg::SparseOperator& jac, const double& timefac_np) const override;

      //! derived
      void write_restart(
          Core::IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const override;

      //! derived
      void read_restart(Core::IO::DiscretizationReader& ioreader) override;

      //! [derived]
      void predict() override {};

      //! derived
      void run_pre_compute_x(const Core::LinAlg::Vector<double>& xold,
          Core::LinAlg::Vector<double>& dir_mutable, const NOX::Nln::Group& curr_grp) override
      {
        return;
      };

      //! derived
      void run_post_compute_x(const Core::LinAlg::Vector<double>& xold,
          const Core::LinAlg::Vector<double>& dir,
          const Core::LinAlg::Vector<double>& xnew) override;

      //! derived
      void run_post_iterate(const ::NOX::Solver::Generic& solver) override { return; };

      //! derived
      void update_step_state(const double& timefac_n) override;

      //! derived
      void update_step_element() override;

      //! derived
      void determine_stress_strain() override;

      //! derived
      void determine_energy() override;

      //! derived
      void output_step_state(Core::IO::DiscretizationWriter& iowriter) const override {}

      //! derived
      void reset_step_state() override;

      //! derived
      std::shared_ptr<const Core::LinAlg::Map> get_block_dof_row_map_ptr() const override;

      //! derived
      std::shared_ptr<const Core::LinAlg::Vector<double>> get_current_solution_ptr() const override;

      //! derived
      std::shared_ptr<const Core::LinAlg::Vector<double>> get_last_time_step_solution_ptr()
          const override;

     private:
      std::shared_ptr<Utils::Cardiovascular0DManager> cardvasc0dman_;  //!< Cardiovascular0D manager

      //! structural displacement at \f$t_{n+1}\f$
      std::shared_ptr<const Core::LinAlg::Vector<double>> disnp_ptr_;

      //! cardio contributions to the structural stiffness matrix
      std::shared_ptr<Core::LinAlg::SparseMatrix> stiff_cardio_ptr_;

      //! structural rhs contributions of the cardio model at \f$t_{n+1}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>> fstructcardio_np_ptr_;
    };

  }  // namespace ModelEvaluator
}  // namespace Solid


FOUR_C_NAMESPACE_CLOSE

#endif
