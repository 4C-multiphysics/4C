// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_IMPL_STATICS_HPP
#define FOUR_C_STRUCTURE_NEW_IMPL_STATICS_HPP

#include "4C_config.hpp"

#include "4C_structure_new_impl_generic.hpp"
#include "4C_utils_exceptions.hpp"

#include <NOX_Abstract_Vector.H>

FOUR_C_NAMESPACE_OPEN

namespace Solid
{
  namespace IMPLICIT
  {
    class Statics : public Generic
    {
     public:
      //! constructor
      Statics();

      //! Setup the class variables
      void setup() override;

      //! (derived)
      void post_setup() override;

      //! Reset state variables (derived)
      void set_state(const Core::LinAlg::Vector<double>& x) override;

      //! Apply the rhs only (derived)
      void apply_force(
          const Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& f) override;

      //! Apply the stiffness only (derived)
      void apply_stiff(
          const Core::LinAlg::Vector<double>& x, Core::LinAlg::SparseOperator& jac) override;

      //! Apply force and stiff at once (derived)
      void apply_force_stiff(const Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& f,
          Core::LinAlg::SparseOperator& jac) override;

      //! (derived)
      void assemble_force(Core::LinAlg::Vector<double>& f,
          const std::vector<Solid::ModelType>* without_these_models = nullptr) const override;

      //! (derived)
      void write_restart(
          Core::IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const override;

      //! (derived)
      void read_restart(Core::IO::DiscretizationReader& ioreader) override;

      //! (derived)
      double calc_ref_norm_force(const enum ::NOX::Abstract::Vector::NormType& type) const override;

      //! return time integration factor (derived)
      [[nodiscard]] double get_int_param() const override;

      //! @name Monolithic update routines
      //! @{
      //! things that should be done before updating (derived)
      void pre_update() override;

      //! Update configuration after time step (derived)
      void update_step_state() override;

      //! Update everything on element level after time step and after output (derived)
      void update_step_element() override;
      //! @}

      //! @name Predictor routines (dependent on the implicit integration scheme)
      //! @{
      /*! predict constant displacements, consistent velocities and accelerations (derived) */
      void predict_const_dis_consist_vel_acc(Core::LinAlg::Vector<double>& disnp,
          Core::LinAlg::Vector<double>& velnp, Core::LinAlg::Vector<double>& accnp) const override;

      /*! predict displacements based on constant velocities and consistent accelerations (derived)
       */
      bool predict_const_vel_consist_acc(Core::LinAlg::Vector<double>& disnp,
          Core::LinAlg::Vector<double>& velnp, Core::LinAlg::Vector<double>& accnp) const override;

      /*! predict displacements based on constant accelerations and consistent velocities (derived)
       */
      bool predict_const_acc(Core::LinAlg::Vector<double>& disnp,
          Core::LinAlg::Vector<double>& velnp, Core::LinAlg::Vector<double>& accnp) const override;
      //! @}

      //! @name Attribute access functions
      //@{

      //! Return name
      Solid::DynamicType method_name() const override { return Solid::DynamicType::Statics; }

      //! Provide number of steps, a single-step method returns 1
      int method_steps() const override { return 1; }

      //! Give local order of accuracy of displacement part
      int method_order_of_accuracy_dis() const override
      {
        FOUR_C_THROW("Sensible to ask?");
        return 0;
      }

      //! Give local order of accuracy of velocity part
      int method_order_of_accuracy_vel() const override
      {
        FOUR_C_THROW("Sensible to ask?");
        return 0;
      }

      //! Return linear error coefficient of displacements
      double method_lin_err_coeff_dis() const override
      {
        FOUR_C_THROW("Sensible to ask?");
        return 0.0;
      }

      //! Return linear error coefficient of velocities
      double method_lin_err_coeff_vel() const override
      {
        FOUR_C_THROW("Sensible to ask?");
        return 0.0;
      }

      //@}

     protected:
      /*! \brief Add the viscous and mass contributions to the right hand side (TR-rule)
       *
       * \remark Nothing needs to be done in the static case. */
      void add_visco_mass_contributions(Core::LinAlg::Vector<double>& f) const override {};

      /*! \brief Add the viscous and mass contributions to the jacobian (TR-rule)
       *
       * \remark Nothing needs to be done in the static case. */
      void add_visco_mass_contributions(Core::LinAlg::SparseOperator& jac) const override {};

      //! reset the time step dependent parameters for the element evaluation [derived]
      void reset_eval_params() override;
    };
  }  // namespace IMPLICIT
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
