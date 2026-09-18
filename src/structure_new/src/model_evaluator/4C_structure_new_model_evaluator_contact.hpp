// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_MODEL_EVALUATOR_CONTACT_HPP
#define FOUR_C_STRUCTURE_NEW_MODEL_EVALUATOR_CONTACT_HPP

#include "4C_config.hpp"

#include "4C_io_discretization_visualization_writer_mesh.hpp"
#include "4C_structure_new_enum_lists.hpp"
#include "4C_structure_new_model_evaluator_generic.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace CONTACT
{
  class Manager;
  class AbstractStrategy;
}  // namespace CONTACT

namespace Mortar
{
  class StrategyBase;
}  // namespace Mortar

namespace Solid
{
  namespace ModelEvaluator
  {
    class ContactData;

    class Contact : public Generic
    {
     public:
      //! setup class variables [derived]
      void setup() override;

      //! @name Functions which are derived from the base generic class
      //!@{

      //! [derived]
      [[nodiscard]] Solid::ModelType type() const override { return Solid::model_contact; }

      //! reset class variables (without jacobian) [derived]
      void reset(const Core::LinAlg::Vector<double>& x) override;

      //! [derived]
      bool evaluate_force() override;

      //! [derived]
      bool evaluate_stiff() override;

      //! [derived]
      bool evaluate_force_stiff() override;

      //! [derived]
      void pre_evaluate() override;

      //! [derived]
      void post_evaluate() override;

      //! [derived]
      void remove_condensed_contributions_from_rhs(Core::LinAlg::Vector<double>& rhs) override;

      //! [derived]
      bool assemble_force(Core::LinAlg::Vector<double>& f, const double& timefac_np) const override;

      //! Assemble the jacobian at \f$t_{n+1}\f$
      bool assemble_jacobian(
          Core::LinAlg::SparseOperator& jac, const double& timefac_np) const override;

      //! [derived]
      void write_restart(
          Core::IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const override;

      //! [derived]
      void read_restart(Core::IO::DiscretizationReader& ioreader) override;

      //! [derived]
      void predict() override {};

      //! recover condensed Lagrange multipliers
      void run_post_compute_x(const Core::LinAlg::Vector<double>& xold,
          const Core::LinAlg::Vector<double>& dir,
          const Core::LinAlg::Vector<double>& xnew) override;

      //! [derived]
      void run_pre_compute_x(const Core::LinAlg::Vector<double>& xold,
          Core::LinAlg::Vector<double>& dir_mutable, const NOX::Nln::Group& curr_grp) override;

      //! [derived]
      void run_post_iterate(const ::NOX::Solver::Generic& solver) override;

      /// [derived]
      void run_pre_solve(const ::NOX::Solver::Generic& solver) override;

      //! [derived]
      void run_post_apply_jacobian_inverse(const Core::LinAlg::Vector<double>& rhs,
          Core::LinAlg::Vector<double>& result, const Core::LinAlg::Vector<double>& xold,
          const NOX::Nln::Group& grp) override;

      //! [derived]
      void run_pre_apply_jacobian_inverse(const Core::LinAlg::Vector<double>& rhs,
          Core::LinAlg::Vector<double>& result, const Core::LinAlg::Vector<double>& xold,
          const NOX::Nln::Group& grp) override;

      //! [derived]
      void update_step_state(const double& timefac_n) override;

      //! [derived]
      void update_step_element() override;

      //! [derived]
      void determine_stress_strain() override;

      //! [derived]
      void determine_energy() override;

      //! [derived]
      void output_step_state(Core::IO::DiscretizationWriter& iowriter) const override {}

      void runtime_pre_output_step_state() override;

      void runtime_output_step_state() const override;

      //! [derived]
      void reset_step_state() override;

      //! [derived]
      [[nodiscard]] std::shared_ptr<const Core::LinAlg::Map> get_block_dof_row_map_ptr()
          const override;

      //! [derived]
      [[nodiscard]] std::shared_ptr<const Core::LinAlg::Vector<double>> get_current_solution_ptr()
          const override;

      //! [derived]
      [[nodiscard]] std::shared_ptr<const Core::LinAlg::Vector<double>>
      get_last_time_step_solution_ptr() const override;

      //! [derived]
      bool evaluate_cheap_soc_rhs() override;

      //! [derived]
      bool assemble_cheap_soc_rhs(
          Core::LinAlg::Vector<double>& f, const double& timefac_np) const override;

      //! @}

      //! @name Call-back routines
      //!@{

      /** \brief Assemble the structural right-hand side vector
       *
       *  \param[in] without_these_models  Exclude all models defined in this vector
       *                                   during the assembly
       *  \param[in] apply_dbc             Apply Dirichlet boundary conditions
       *
       *  */
      std::shared_ptr<Core::LinAlg::Vector<double>> assemble_force_of_models(
          const std::vector<Solid::ModelType>* without_these_models = nullptr,
          const bool apply_dbc = false) const;

      [[nodiscard]] virtual std::shared_ptr<Core::LinAlg::SparseOperator> get_aux_displ_jacobian()
          const;

      void evaluate_weighted_gap_gradient_error();

      //!@}

      //! @name Accessors
      //!@{

      //! Returns a pointer to the underlying contact strategy object
      const std::shared_ptr<CONTACT::AbstractStrategy>& strategy_ptr();

      //! Returns the underlying contact strategy object
      CONTACT::AbstractStrategy& strategy();
      [[nodiscard]] const CONTACT::AbstractStrategy& strategy() const;

      //!@}

     protected:
      Solid::ModelEvaluator::ContactData& eval_contact();
      [[nodiscard]] const Solid::ModelEvaluator::ContactData& eval_contact() const;

      virtual void check_pseudo2d() const;

     private:
      void post_setup(Teuchos::ParameterList& cparams);

      /// correct time integration parameters within the contact strategy
      void set_time_integration_info(CONTACT::AbstractStrategy& strategy) const;

      void post_update_step_state();

      void extend_lagrange_multiplier_domain(
          std::shared_ptr<Core::LinAlg::Vector<double>>& lm_vec) const;

      //! contact evaluation data container
      std::shared_ptr<Solid::ModelEvaluator::ContactData> eval_contact_ptr_;

      //! contact strategy
      std::shared_ptr<CONTACT::AbstractStrategy> strategy_ptr_;

      //! contact runtime output writer
      std::unique_ptr<Core::IO::DiscretizationVisualizationWriterMesh> contact_vtu_writer_ptr_;

    };  // class Contact
  }  // namespace ModelEvaluator
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
