// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_MODEL_EVALUATOR_MESHTYING_HPP
#define FOUR_C_STRUCTURE_NEW_MODEL_EVALUATOR_MESHTYING_HPP

#include "4C_config.hpp"

#include "4C_structure_new_model_evaluator_generic.hpp"
#include "4C_structure_new_timint_basedataglobalstate.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace CONTACT
{
  class Manager;
  class MtAbstractStrategy;
}  // namespace CONTACT

namespace Mortar
{
  class StrategyBase;
}  // namespace Mortar

namespace Solid
{
  namespace ModelEvaluator
  {
    class MeshtyingData;

    /*! \brief Model evaluator for meshtying problems
     *
     */
    class Meshtying : public Generic
    {
     public:
      //! constructor
      Meshtying();


      /*! \brief Initialize class variables [derived]
       *
       * @param eval_data_ptr
       * @param gstate_ptr
       * @param gio_ptr
       * @param int_ptr
       * @param timint_ptr
       * @param dof_offset
       */
      void init(const Teuchos::RCP<Solid::ModelEvaluator::Data>& eval_data_ptr,
          const Teuchos::RCP<Solid::TimeInt::BaseDataGlobalState>& gstate_ptr,
          const Teuchos::RCP<Solid::TimeInt::BaseDataIO>& gio_ptr,
          const Teuchos::RCP<Solid::Integrator>& int_ptr,
          const Teuchos::RCP<const Solid::TimeInt::Base>& timint_ptr,
          const int& dof_offset) override;

      //! setup class variables [derived]
      void setup() override;

      //! @name Functions which are derived from the base generic class
      //!@{

      //! [derived]
      Inpar::Solid::ModelType type() const override { return Inpar::Solid::model_meshtying; }

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
      void predict(const Inpar::Solid::PredEnum& pred_type) override{};

      //! [derived]
      void run_post_compute_x(const Core::LinAlg::Vector<double>& xold,
          const Core::LinAlg::Vector<double>& dir,
          const Core::LinAlg::Vector<double>& xnew) override;

      //! [derived]
      void run_pre_compute_x(const Core::LinAlg::Vector<double>& xold,
          Core::LinAlg::Vector<double>& dir_mutable, const NOX::Nln::Group& curr_grp) override{};

      //! [derived]
      void run_post_iterate(const ::NOX::Solver::Generic& solver) override{};

      //! [derived]
      void run_post_apply_jacobian_inverse(const Core::LinAlg::Vector<double>& rhs,
          Core::LinAlg::Vector<double>& result, const Core::LinAlg::Vector<double>& xold,
          const NOX::Nln::Group& grp) override;

      //! [derived]
      void run_pre_apply_jacobian_inverse(const Core::LinAlg::Vector<double>& rhs,
          Core::LinAlg::Vector<double>& result, const Core::LinAlg::Vector<double>& xold,
          const NOX::Nln::Group& grp) override;

      //! [derived]
      void update_step_state(const double& timefac_n) override{};

      //! [derived]
      void update_step_element() override{};

      //! [derived]
      void determine_stress_strain() override{};

      //! [derived]
      void determine_energy() override{};

      //! [derived]
      void determine_optional_quantity() override{};

      //! [derived]
      void output_step_state(Core::IO::DiscretizationWriter& iowriter) const override{};

      //! [derived]
      void reset_step_state() override{};

      //! [derived]
      Teuchos::RCP<const Epetra_Map> get_block_dof_row_map_ptr() const override;

      //! [derived]
      Teuchos::RCP<const Core::LinAlg::Vector<double>> get_current_solution_ptr() const override;

      //! [derived]
      Teuchos::RCP<const Core::LinAlg::Vector<double>> get_last_time_step_solution_ptr()
          const override;

      //! [derived]
      void post_output() override{};

      /*! \brief Reset model specific variables (without jacobian) [derived]
       *
       * Nothing to do in case of meshtying.
       *
       * \param[in] x Current full state vector
       */
      void reset(const Core::LinAlg::Vector<double>& x) override{};

      //! \brief Perform actions just before the evaluate() call [derived]
      void pre_evaluate() override{};

      //! \brief Perform actions right after the evaluate() call [derived]
      void post_evaluate() override{};

      //! @}

      //! @name Call-back routines
      //!@{

      Teuchos::RCP<const Core::LinAlg::SparseMatrix> get_jacobian_block(
          const Solid::MatBlockType bt) const;

      /** \brief Assemble the structural right-hand side vector
       *
       *  \param[in] without_these_models  Exclude all models defined in this vector
       *                                   during the assembly
       *  \param[in] apply_dbc             Apply Dirichlet boundary conditions
       *
       *  \author hiermeier \date 08/17 */
      Teuchos::RCP<Core::LinAlg::Vector<double>> assemble_force_of_models(
          const std::vector<Inpar::Solid::ModelType>* without_these_models = nullptr,
          const bool apply_dbc = false) const;

      virtual Teuchos::RCP<Core::LinAlg::SparseOperator> get_aux_displ_jacobian() const
      {
        return Teuchos::null;
      };

      void evaluate_weighted_gap_gradient_error();

      //! [derived]
      bool evaluate_force() override;

      //! [derived]
      bool evaluate_stiff() override;

      //! [derived]
      bool evaluate_force_stiff() override;

      /*!
      \brief Apply results of mesh initialization to the underlying problem discretization

      \note This is only necessary in case of a mortar method.

      \warning This routine modifies the reference coordinates of slave nodes at the meshtying
      interface.

      @param[in] Xslavemod Vector with modified nodal positions
      */
      void apply_mesh_initialization(Teuchos::RCP<const Core::LinAlg::Vector<double>> Xslavemod);

      //!@}

      //! @name Accessors
      //!@{

      //! Returns a pointer to the underlying meshtying strategy object
      const Teuchos::RCP<CONTACT::MtAbstractStrategy>& strategy_ptr();

      //! Returns the underlying meshtying strategy object
      CONTACT::MtAbstractStrategy& strategy();
      const CONTACT::MtAbstractStrategy& strategy() const;

      //!@}

     protected:
     private:
      /// Set the correct time integration parameters within the meshtying strategy
      void set_time_integration_info(CONTACT::MtAbstractStrategy& strategy) const;

      //! meshtying strategy
      Teuchos::RCP<CONTACT::MtAbstractStrategy> strategy_ptr_;

      //! Mesh relocation for conservation of angular momentum
      Teuchos::RCP<Core::LinAlg::Vector<double>> mesh_relocation_;
    };  // namespace ModelEvaluator

  }  // namespace ModelEvaluator
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
