// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONSTRAINT_FRAMEWORK_MODEL_EVALUATOR_HPP
#define FOUR_C_CONSTRAINT_FRAMEWORK_MODEL_EVALUATOR_HPP

#include "4C_config.hpp"

#include "4C_constraint_framework_submodelevaluator_base.hpp"
#include "4C_inpar_constraint_framework.hpp"
#include "4C_io_visualization_parameters.hpp"
#include "4C_structure_new_model_evaluator_generic.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Constraints::SubmodelEvaluator
{
  class RveMultiPointConstraintManager;

}
namespace Solid
{
  namespace ModelEvaluator
  {
    /**
     * \brief This class serves as a model evaluator for different types
     * of constraints applied to structural degrees of freedom.
     *
     * Through submodel evaluators, this class evaluates and assembles
     * the contributions resulting from periodic displacement boundary
     * conditions and coupling terms resulting from the constraint
     * enforcement for embedded mesh methods. The implementation of
     * these applications and their submodel evaluators is still a
     * work in progress.
     */
    class Constraint : public Generic
    {
     public:
      using SubmodelevaluatorVector =
          std::vector<std::shared_ptr<Constraints::SubmodelEvaluator::ConstraintBase>>;

      Constraint() = default;

      /*! \brief Setup of the model evaluator and submodel evaluator
       *
       */
      void setup() override;

      //! @name Derived public Solid::ModelEvaluator::Generic methods
      //! @{
      Inpar::Solid::ModelType type() const override { return Inpar::Solid::model_constraints; }

      /// check if the given model type is active.
      bool have_sub_model_type(Inpar::Constraints::SubModelType const& submodeltype) const;

      void reset(const Core::LinAlg::Vector<double>& x) override;

      bool evaluate_force() override;

      bool evaluate_stiff() override;

      bool evaluate_force_stiff() override;

      void pre_evaluate() override;

      void post_evaluate() override {}

      bool assemble_force(Core::LinAlg::Vector<double>& f, const double& timefac_np) const override;

      bool assemble_jacobian(
          Core::LinAlg::SparseOperator& jac, const double& timefac_np) const override;

      void write_restart(
          Core::IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const override;

      void read_restart(Core::IO::DiscretizationReader& ioreader) override;

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

      void predict(const Inpar::Solid::PredEnum& pred_type) override;

      void update_step_state(const double& timefac_n) override;

      void update_step_element() override;

      void determine_stress_strain() override;

      void determine_energy() override;

      void determine_optional_quantity() override;

      void reset_step_state() override;

      void output_step_state(Core::IO::DiscretizationWriter& iowriter) const override;

      void runtime_pre_output_step_state() override;

      void runtime_output_step_state() const override;

      std::shared_ptr<const Core::LinAlg::Map> get_block_dof_row_map_ptr() const override;

      std::shared_ptr<const Core::LinAlg::Vector<double>> get_current_solution_ptr() const override;

      std::shared_ptr<const Core::LinAlg::Vector<double>> get_last_time_step_solution_ptr()
          const override;

      void post_output() override;

      void evaluate_jacobian_contributions_from_element_level_for_ptc() override;

      void assemble_jacobian_contributions_from_element_level_for_ptc(
          std::shared_ptr<Core::LinAlg::SparseMatrix>& modjac, const double& timefac_n) override;

      void create_backup_state(const Core::LinAlg::Vector<double>& dir) override;

      void recover_from_backup_state() override;
      //! @}

     private:
      //!@name routines for submodel management
      //! @{

      //! Set Submodeltypes depending on input
      void set_sub_model_types();

      //! build, init and setup submodel evaluator
      void create_sub_model_evaluators();

      //! @}

     private:
      //!@name data for submodel management
      //! @{
      /// active model types for the model evaluator
      std::set<enum Inpar::Constraints::SubModelType> submodeltypes_;

      //! vector of submodelevaluators
      SubmodelevaluatorVector sub_model_vec_ptr_;

      //! constraint stiffness matrix
      std::shared_ptr<Core::LinAlg::SparseMatrix> constraint_stiff_ptr_;

      //! constraint force vector
      std::shared_ptr<Core::LinAlg::Vector<double>> constraint_force_ptr_;

      //! visualization parameters
      Core::IO::VisualizationParameters visualization_params_;
      //! @}
    };
  }  // namespace ModelEvaluator
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
