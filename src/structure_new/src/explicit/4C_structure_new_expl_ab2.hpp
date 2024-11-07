// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_EXPL_AB2_HPP
#define FOUR_C_STRUCTURE_NEW_EXPL_AB2_HPP

#include "4C_config.hpp"

#include "4C_structure_new_expl_generic.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Solid
{
  namespace EXPLICIT
  {
    /*! \brief Adams-Bashforth-2 time integration for solid dynamics
     *
     */
    class AdamsBashforth2 : public Generic
    {
     public:
      //! constructor
      AdamsBashforth2();

      //! Setup class variables (derived)
      void setup() override;

      //! Post setup operation (compute initial equilibrium state) (derived)
      void post_setup() override;

      //! Set state variables (derived)
      void set_state(const Core::LinAlg::Vector<double>& x) override;

      //! return integration factor (derived)
      [[nodiscard]] double get_int_param() const override { return -1.0; }

      /*! \brief Add the viscous and mass contributions to the right hand side
       */
      void add_visco_mass_contributions(Core::LinAlg::Vector<double>& f) const override;

      /*! \brief Add the viscous and mass contributions to the jacobian (TR-rule)
       */
      void add_visco_mass_contributions(Core::LinAlg::SparseOperator& jac) const override;

      //! Update configuration after time step (derived)
      void update_step_state() override;

      //! (derived)
      void write_restart(
          Core::IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const override;

      /*! read restart information of the different time integration schemes
       *  and model evaluators (derived) */
      void read_restart(Core::IO::DiscretizationReader& ioreader) override;

      //! @name Attribute access functions
      //@{

      //! Return time integrator name
      [[nodiscard]] enum Inpar::Solid::DynamicType method_name() const override
      {
        return Inpar::Solid::dyna_ab2;
      }

      //! Provide number of steps, e.g. a single-step method returns 1,
      //! a m-multistep method returns m
      [[nodiscard]] int method_steps() const override { return 2; }

      //! Give local order of accuracy of displacement part
      [[nodiscard]] int method_order_of_accuracy_dis() const override { return 2; }

      //! Give local order of accuracy of velocity part
      [[nodiscard]] int method_order_of_accuracy_vel() const override { return 2; }

      //! Return linear error coefficient of displacements
      [[nodiscard]] double method_lin_err_coeff_dis() const override;

      //! Return linear error coefficient of velocities
      [[nodiscard]] double method_lin_err_coeff_vel() const override
      {
        return method_lin_err_coeff_dis();
      }

      //@}

     private:
      //! viscous force vector F_viscous F_{viscous;n+1}
      std::shared_ptr<Core::LinAlg::Vector<double>> fvisconp_ptr_;

      //! viscous force vector F_viscous F_{viscous;n}
      std::shared_ptr<Core::LinAlg::Vector<double>> fviscon_ptr_;

      //! pointer to inertial force vector F_{inertial,n+1} at new time
      std::shared_ptr<Core::LinAlg::Vector<double>> finertianp_ptr_;

      //! pointer to inertial force vector F_{inertial,n} at last time
      std::shared_ptr<Core::LinAlg::Vector<double>> finertian_ptr_;
    };
  }  // namespace EXPLICIT
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
