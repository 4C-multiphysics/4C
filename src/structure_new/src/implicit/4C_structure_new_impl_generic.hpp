// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_IMPL_GENERIC_HPP
#define FOUR_C_STRUCTURE_NEW_IMPL_GENERIC_HPP

#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_abstract_prepostoperator.hpp"
#include "4C_structure_new_integrator.hpp"
// required because of used enums
#include <NOX_Abstract_Vector.H>

FOUR_C_NAMESPACE_OPEN

// forward declaration ...
namespace Core::LinAlg
{
  class SparseOperator;
  class SparseMatrix;
}  // namespace Core::LinAlg

namespace Solid
{
  namespace IMPLICIT
  {
    /*! \brief A generic fully implicit time integrator for solid dynamics
     *
     * This serves as a base class for all fully implicit time integration schemes.
     */
    class Generic : public Integrator
    {
     public:
      //! constructor
      Generic();

      //! Setup (has to be implemented by the derived classes)
      void setup() override;

      //! derived
      void apply_correction_system(const NOX::Nln::CorrectionType type,
          const std::vector<Solid::ModelType>& constraint_models,
          const Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& f,
          Core::LinAlg::SparseOperator& jac) override;

      //! derived
      void remove_condensed_contributions_from_rhs(
          Core::LinAlg::Vector<double>& rhs) const override;

      /*! \brief Calculate characteristic/reference norms for forces
       *
       *  The reference norms are used to scale the calculated iterative
       *  displacement norm and/or the residual force norm. For this
       *  purpose we only need the right order of magnitude, so we don't
       *  mind evaluating the corresponding norms at possibly different
       *  points within the time-step (end point, generalized midpoint). */
      double calc_ref_norm_force(
          const enum ::NOX::Abstract::Vector::NormType& type) const override = 0;

      //! return the default step length of the used ::NOX::LineSearch method
      double get_default_step_length() const;

      //! @name Monolithic update routines
      //! @{
      //! things that should be done before updating (derived)
      void pre_update() override { /* do nothing for now */ };

      //! things that should be done after updating (derived)
      void post_update() override { /* do nothing for now */ };

      //! update constant contributions of the current state for the new time step \f$t_{n+1}\f$
      void update_constant_state_contributions() override { /* do nothing for some integrators */ };
      //! @}

      //! @name Predictor routines (dependent on the implicit integration scheme)
      //!@{

      /*! \brief predict constant displacements with consistent velocities and accelerations
       *
       * The displacement field is kept constant, however the velocities
       * and accelerations are consistent to the time integration
       * if the constant displacements are taken as correct displacement solution.
       *
       * This method has to be implemented by the individual time
       * integrator since the calculation of consistent velocities and accelerations
       * depends on the actual time integration scheme.
       */
      virtual void predict_const_dis_consist_vel_acc(Core::LinAlg::Vector<double>& disnp,
          Core::LinAlg::Vector<double>& velnp, Core::LinAlg::Vector<double>& accnp) const = 0;

      /*! \brief predict displacements based on the assumption of constant velocities.
       *
       * Assuming constant velocities, new displacements are predicted.
       * Calculate consistent accelerations afterwards.
       *
       * This method has to be implemented by the individual time
       * integrator since the calculation of consistent velocities and accelerations
       * depends on the actual time integration scheme.
       *
       * \param[in/out] disnp Displacement vector
       * \param[in/out] velnp Velocity vector
       * \param[in/out] accnp Acceleration vector
       */
      virtual bool predict_const_vel_consist_acc(Core::LinAlg::Vector<double>& disnp,
          Core::LinAlg::Vector<double>& velnp, Core::LinAlg::Vector<double>& accnp) const = 0;

      /*! \brief predict displacements based on the assumption of constant accelerations.
       *
       * Assuming constant accelerations, new velocities and displacements are predicted.
       *
       * This method has to be implemented by the individual time
       * integrator since the calculation of consistent velocities and accelerations
       * depends on the actual time integration scheme.
       *
       * \param[in/out] disnp Displacement vector
       * \param[in/out] velnp Velocity vector
       * \param[in/out] accnp Acceleration vector
       */
      virtual bool predict_const_acc(Core::LinAlg::Vector<double>& disnp,
          Core::LinAlg::Vector<double>& velnp, Core::LinAlg::Vector<double>& accnp) const = 0;
      //!@}

      /*! \brief Set the predictor state flag
       *
       * \param[in] ispredictor_state Predictor state flag
       */
      void set_is_predictor_state(const bool ispredictor_state);

      //! Get the predictor state flag
      bool is_predictor_state() const;

      //! compute the scaling operator for element based scaling using PTC
      void compute_jacobian_contributions_from_element_level_for_ptc(
          std::shared_ptr<Core::LinAlg::SparseMatrix>& scalingMatrixOpPtr) override;

      /*! \brief Print jacbian into text file for later use in MATLAB
       *
       *  \param[in] NOX group containing the linear system with the Jacobian
       *
       *  */
      void print_jacobian_in_matlab_format(const NOX::Nln::Group& curr_grp) const;

      //! Get the NOX parameter list
      Teuchos::ParameterList& get_nox_params();

      //! @name Attribute access functions
      //@{

      //! Provide Name
      virtual Solid::DynamicType method_name() const = 0;

      //! Provide number of steps, e.g. a single-step method returns 1,
      //! a \f$m\f$-multistep method returns \f$m\f$
      virtual int method_steps() const = 0;

      //! Give local order of accuracy of displacement part
      virtual int method_order_of_accuracy_dis() const = 0;

      //! Give local order of accuracy of velocity part
      virtual int method_order_of_accuracy_vel() const = 0;

      //! Return linear error coefficient of displacements
      virtual double method_lin_err_coeff_dis() const = 0;

      //! Return linear error coefficient of velocities
      virtual double method_lin_err_coeff_vel() const = 0;

      //! Return a reference to the container of the structural dynamic data (read only)
      const Solid::TimeInt::BaseDataSDyn& get_data_sdyn() const;

      //! @}

     protected:
      //! reset the time step dependent parameters for the element evaluation [derived]
      void reset_eval_params() override;

     private:
      /*! \brief Flag indicating if the current state is the predictor state.
       *
       *  In the evaluation of the predictor state the set_state() routine is
       *  not allowed to calculate the consistent velocities and accelerations
       *  as usual. This is due to the fact, that the predictor might lead to
       *  velocities and accelerations that are not consistently computed from
       *  the displacements based on the time integration scheme. Instead we
       *  leave the predictor state untouched during the first evaluation. */
      bool ispredictor_state_;

    };  // namespace IMPLICIT
  }  // namespace IMPLICIT
}  // namespace Solid

namespace NOX
{
  namespace Nln
  {
    namespace PrePostOp
    {
      namespace IMPLICIT
      {
        /*! \brief Implicit time integration helper class
         *
         *  This class is an implementation of the NOX::Nln::Abstract::PrePostOperator
         *  and is used to modify the computeX() routine of the given NOX::Nln::Group.
         *  It's called by the wrapper class NOX::Nln::GROUP::PrePostOperator.
         *
         *  */
        class Generic : public NOX::Nln::Abstract::PrePostOperator
        {
         public:
          //! constructor
          Generic(const Solid::IMPLICIT::Generic& implicit)
              : impl_(implicit), default_step_(implicit.get_default_step_length()) { /* empty */ };

          /*! \brief Derived function, which is called at the very beginning of a call to
           *  NOX::Nln::Group::computeX()
           *
           *  This method is used to get access to the current direction vector and
           *  to augment/modify the direction vector before the actual solution update is
           *  performed. One possible scenario is the CONTACT::Aug::SteepestAscent::Strategy,
           *  where we calculate the Lagrange multiplier increment in a post-processing
           *  step.
           *
           *  */
          void run_pre_compute_x(const NOX::Nln::Group& input_grp,
              const Core::LinAlg::Vector<double>& dir, const double& step,
              const NOX::Nln::Group& curr_grp) override;

          /*! \brief Derived function, which is called at the end of a call to
           * NOX::Nln::Group::computeX()
           *
           *  This method is used to get access to the current direction vector. The
           *  direction vector is needed for different internal update routines. Two
           *  examples on element level are the EAS recovery, since the EAS DoFs are
           *  condensed and the calculation of the strain increments. The displacement
           *  increment plays also a role in other condensation approaches, like the
           *  mortar dual strategies.
           *
           *  */
          void run_post_compute_x(const NOX::Nln::Group& input_grp,
              const Core::LinAlg::Vector<double>& dir, const double& step,
              const NOX::Nln::Group& curr_grp) override;

          /*! \brief Derived function, which is called at the very end of a call to
           *  ::NOX::Solver::Generic::step()
           *
           *  This method gives you the opportunity to do something in the end of
           *  a successful or unsuccessful nonlinear solver step. It is called after
           *  a possible LineSearch or other globalization method has been used.
           *
           *  */
          void runPostIterate(const ::NOX::Solver::Generic& solver) override;

          /*! \brief Derived function, which is called at the very end of a call to
           *  NOX::Nln::Group::applyJacobianInverse()
           *
           *  This method gives you the opportunity to do something in the end of
           *  a successful or unsuccessful linear solver attempt.
           *
           *  \note The result vector is the actual result vector of the internal
           *  linear solver and, accordingly, due to the used sign convention in NOX,
           *  the NEGATIVE direction vector. The sign will be changed again in the
           *  NOX::Nln::GroupBase::computeNewton method.
           *
           *  \param rhs    : read-only access to the rhs vector
           *  \param result : full access to the result vector
           *  \param xold   : read-only access to the old state vector
           *  \param grp    : read only access to the group object
           *
           *  */
          void run_post_apply_jacobian_inverse(const ::NOX::Abstract::Vector& rhs,
              ::NOX::Abstract::Vector& result, const ::NOX::Abstract::Vector& xold,
              const NOX::Nln::Group& grp) override;

          /// \brief Called at the very beginning of a Newton loop
          /**
           *  */
          void runPreSolve(const ::NOX::Solver::Generic& solver) override;

          /*! \brief Derived function, which is called at the beginning of a call to
           *  NOX::Nln::Group::applyJacobianInverse()
           *
           *  \param rhs    : read-only access to the rhs vector
           *  \param result : full access to the result vector
           *  \param xold   : read-only access to the old state vector
           *  \param grp    : read only access to the group object
           *
           *  */
          void run_pre_apply_jacobian_inverse(const ::NOX::Abstract::Vector& rhs,
              ::NOX::Abstract::Vector& result, const ::NOX::Abstract::Vector& xold,
              const NOX::Nln::Group& grp) override;

         private:
          /// get the step length
          bool get_step(double& step, const ::NOX::Solver::Generic& solver) const;

         private:
          //! reference to the Solid::IMPLICIT::Generic object (read-only)
          const Solid::IMPLICIT::Generic& impl_;

          //! default step length
          const double default_step_;

        };  // class Generic
      }  // namespace IMPLICIT
    }  // namespace PrePostOp
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
