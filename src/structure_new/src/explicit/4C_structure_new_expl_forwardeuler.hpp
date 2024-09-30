/*-----------------------------------------------------------*/
/*! \file

\brief Explicit Euler time integration for solid dynamics


\level 3

*/
/*-----------------------------------------------------------*/

#ifndef FOUR_C_STRUCTURE_NEW_EXPL_FORWARDEULER_HPP
#define FOUR_C_STRUCTURE_NEW_EXPL_FORWARDEULER_HPP

#include "4C_config.hpp"

#include "4C_structure_new_expl_generic.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Solid
{
  namespace EXPLICIT
  {
    /*! \brief Explicit Euler time integration for solid dynamics
     *
     */
    class ForwardEuler : public Generic
    {
     public:
      //! constructor
      ForwardEuler();

      //! Setup class variables (derived)
      void setup() override;

      //! Post setup operation (compute initial equilibrium state) (derived)
      void post_setup() override;

      //! Set state variables (derived)
      void set_state(const Core::LinAlg::Vector& x) override;

      //! return integration factor (derived)
      double get_int_param() const override { return -1.0; }

      /*! \brief Add the viscous and mass contributions to the right hand side
       */
      void add_visco_mass_contributions(Core::LinAlg::Vector& f) const override;

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
        return Inpar::Solid::dyna_expleuler;
      }

      //! Provide number of steps, e.g. a single-step method returns 1,
      //! a m-multistep method returns m
      [[nodiscard]] int method_steps() const override { return 1; }

      //! Give local order of accuracy of displacement part
      [[nodiscard]] int method_order_of_accuracy_dis() const override { return 1; }

      //! Give local order of accuracy of velocity part
      [[nodiscard]] int method_order_of_accuracy_vel() const override { return 1; }

      /*! \brief Return linear error coefficient of displacements
       *
       *  The local discretization error reads
       *  \f[
       *  e \approx \frac{1}{2}\Delta t_n^2 \ddot{d_n} + HOT(\Delta t_n^3)
       *  \f]
       */
      [[nodiscard]] double method_lin_err_coeff_dis() const override { return 0.5; }

      /*! \brief Return linear error coefficient of velocities
       *
       *  The local discretization error reads
       *  \f[
       *  e \approx \frac{1}{2}\Delta t_n^2 \dddot{d_n} + HOT(\Delta t_n^3)
       *  \f]
       */
      [[nodiscard]] double method_lin_err_coeff_vel() const override { return 0.5; }

      //@}

     private:
      bool modexpleuler_;  //!< modified explicit Euler equation (veln_ instead of vel_ for
                           //!< calc of disn_), default: true

      //! viscous force vector F_viscous F_{viscous;n+1}
      Teuchos::RCP<Core::LinAlg::Vector> fvisconp_ptr_;

      //! viscous force vector F_viscous F_{viscous;n}
      Teuchos::RCP<Core::LinAlg::Vector> fviscon_ptr_;

      //! pointer to inertial force vector F_{inertial,n+1} at new time
      Teuchos::RCP<Core::LinAlg::Vector> finertianp_ptr_;

      //! pointer to inertial force vector F_{inertial,n} at last time
      Teuchos::RCP<Core::LinAlg::Vector> finertian_ptr_;
    };
  }  // namespace EXPLICIT
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
