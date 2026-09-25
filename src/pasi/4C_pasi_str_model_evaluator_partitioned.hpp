// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PASI_STR_MODEL_EVALUATOR_PARTITIONED_HPP
#define FOUR_C_PASI_STR_MODEL_EVALUATOR_PARTITIONED_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_structure_new_model_evaluator_generic.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace Adapter
{
  class Structure;
}

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Solid
{
  namespace ModelEvaluator
  {
    class PartitionedPASI : public Generic
    {
     public:
      //! constructor
      PartitionedPASI();

      //! setup class variables [derived]
      void setup() override;

      //! get pointer to force vector at time level n+1 (full structural map)
      //! interface part is inserted in Adapter::PASIStructureWrapper
      const std::shared_ptr<Core::LinAlg::Vector<double>>& get_interface_force_np_ptr()
      {
        return interface_force_np_ptr_;
      };

      //! @name Functions which are derived from the base generic class
      //! @{

      //! [derived]
      Solid::ModelType type() const override { return Solid::model_partitioned_coupling; }

      //! reset class variables (without jacobian) [derived]
      void reset(const Core::LinAlg::Vector<double>& x) override { return; };

      //! [derived]
      bool evaluate_force() override { return true; };

      //! [derived]
      bool evaluate_stiff() override { return true; };

      //! [derived] not needed in partitioned scheme
      bool evaluate_force_stiff() override { return true; };

      //! [derived]
      void pre_evaluate() override { return; };

      //! [derived]
      void post_evaluate() override { return; };

      //! [derived]
      bool assemble_force(Core::LinAlg::Vector<double>& f, const double& timefac_np) const override;

      //! Assemble the jacobian at \f$t_{n+1}\f$ not needed in partitioned scheme
      bool assemble_jacobian(
          Core::LinAlg::SparseOperator& jac, const double& timefac_np) const override
      {
        return true;
      };

      //! [derived]
      void write_restart(
          Core::IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const override
      {
        return;
      };

      //! [derived]
      void read_restart(Core::IO::DiscretizationReader& ioreader) override { return; };

      //! [derived]
      void run_pre_compute_x(const Core::LinAlg::Vector<double>& xold,
          Core::LinAlg::Vector<double>& dir_mutable, const NOX::Nln::Group& curr_grp) override
      {
        return;
      };

      //! recover condensed Lagrange multipliers
      void run_post_compute_x(const Core::LinAlg::Vector<double>& xold,
          const Core::LinAlg::Vector<double>& dir,
          const Core::LinAlg::Vector<double>& xnew) override
      {
        return;
      };

      //! [derived]
      void run_post_iterate(const ::NOX::Solver::Generic& solver) override { return; };

      //! [derived]
      void update_step_state(const double& timefac_n) override;

      //! [derived]
      void update_step_element() override { return; };

      //! [derived]
      void determine_stress_strain() override { return; };

      //! [derived]
      void determine_energy() override { return; };

      //! [derived]
      void output_step_state(Core::IO::DiscretizationWriter& iowriter) const override {}

      //! [derived]
      void predict() override {};

      //! [derived]
      void reset_step_state() override { return; };

      //! [derived]
      std::shared_ptr<const Core::LinAlg::Map> get_block_dof_row_map_ptr() const override;

      //! [derived]
      std::shared_ptr<const Core::LinAlg::Vector<double>> get_current_solution_ptr() const override;

      //! [derived]
      std::shared_ptr<const Core::LinAlg::Vector<double>> get_last_time_step_solution_ptr()
          const override;

      //! @}

     private:
      //! pasi interface force at \f$t_{n+1}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>> interface_force_np_ptr_;
    };

  }  // namespace ModelEvaluator

}  // namespace Solid

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
