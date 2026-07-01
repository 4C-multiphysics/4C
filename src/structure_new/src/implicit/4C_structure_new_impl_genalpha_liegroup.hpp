// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_IMPL_GENALPHA_LIEGROUP_HPP
#define FOUR_C_STRUCTURE_NEW_IMPL_GENALPHA_LIEGROUP_HPP

#include "4C_config.hpp"

#include "4C_structure_new_impl_genalpha.hpp"


FOUR_C_NAMESPACE_OPEN

namespace Solid
{
  namespace IMPLICIT
  {
    /*! \brief Modified GenAlpha time integrator for Lie groups
     *
     *  We need this for all beam elements with rotation vector
     *  DoFs (currently beam3r, beam3k) because those are non-
     *  additive and the corresponding rotation tensors are
     *  members of the special orthogonal group SO3 which is no
     *  vector space!
     *
     *  The implementation is based on [Arnold, Bruels (2007)],
     *  [Bruels, Cardona, 2010] and [Bruels, Cardona, Arnold (2012)].
     *  See also Christoph Meier's dissertation for explanations,
     *  governing equations, differences to classical GenAlpha,
     *  comments on why we chose this time discretization, ...
     *
     *  In theory, this should be a generalization of classical
     *  GenAlpha, which yields the same results if applied to
     *  problems whose primary DoFs are elements of vector spaces.
     *
     *  The main difference to classical GenAlpha is that a
     *  modified acceleration vector is introduced (see accn_mod_)
     *  and that the weak form is evaluated at t_{n+1} only.
     *
     *  Furthermore, the update of non-additive rotation vector
     *  DoFs needs a special (multiplicative) treatment. However,
     *  for the implemented beam elements with rotation vector DoFs,
     *  the calculation of angular velocities and accelerations
     *  from temporal derivatives of the primary rotation vector DoFs
     *  is intricate (because of non-linear triad interpolation
     *  schemes). Therefore, these beam elements first discretize
     *  in time and apply the Newmark scheme to space-continuous
     *  angular velocities and accelerations. All this (including
     *  storage of old state variables) happens inside the beam
     *  elements. The beam elements are therefore independent from
     *  the velocity and acceleration vectors on time integrator level.
     *
     *  However, in mixed discretizations (beams and other elements),
     *  we need the global velocity and acceleration vector to evaluate
     *  the inertia terms of all other than beam elements in the usual manner.
     *  Note: because of the non-linear inertia forces and mass matrix,
     *        the beams require "MASSLIN = rotations" which evaluates
     *        inertia force and mass matrix in every iteration by calling
     *        Evaluate with action type "nlnstiffmass". Because this
     *        setting (ml_rotations) is true for entire discretization,
     *        all other than beam elements must adapt to this and also
     *        evaluate the inertia force and mass matrix inside the element.
     *
     *  Possible future work (to make this even nicer):
     *  We could use the global velocity and acceleration vectors
     *  for all translational (and hence additive) DoFs to compute
     *  the inertia terms inside the element but the rotation vector
     *  DoFs anyway need the element-internal treatment described above.
     *
     *  Note: - viscous Rayleigh damping is supported in principle, but
     *          never applied or tested yet!
     *
     *        - this is not yet rigorously tested with other than beam
     *          and rigidsphere elements and thus experimental code
     *
     */

    class GenAlphaLieGroup : public GenAlpha
    {
     public:
      //! constructor
      GenAlphaLieGroup();

      //! Setup the class variables [derived]
      void setup() override;

      //! [derived]
      void post_setup() override;

      //! Reset state variables [derived]
      void set_state(const Core::LinAlg::Vector<double>& x) override;

      //! [derived]
      void write_restart(
          Core::IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const override;

      //! [derived]
      void read_restart(Core::IO::DiscretizationReader& ioreader) override;

      //! [derived]
      double get_int_param() const override;

      //! Refresh cached force pointers and map-dependent helper vectors after redistribution.
      void remap_integrator_state_after_redistribution() override;

      //! @name Monolithic update routines
      //! @{
      //! Update configuration after time step [derived]
      void update_step_state() override;
      //! @}

      //! @name Predictor routines (dependent on the implicit integration scheme)
      //! @{
      /*! predict constant displacements, consistent velocities and accelerations [derived] */
      void predict_const_dis_consist_vel_acc(Core::LinAlg::Vector<double>& disnp,
          Core::LinAlg::Vector<double>& velnp, Core::LinAlg::Vector<double>& accnp) const override;

      /*! predict displacements based on constant velocities
       *  and consistent accelerations [derived] */
      bool predict_const_vel_consist_acc(Core::LinAlg::Vector<double>& disnp,
          Core::LinAlg::Vector<double>& velnp, Core::LinAlg::Vector<double>& accnp) const override;

      /*! predict displacements based on constant accelerations
       *  and consistent velocities [derived] */
      bool predict_const_acc(Core::LinAlg::Vector<double>& disnp,
          Core::LinAlg::Vector<double>& velnp, Core::LinAlg::Vector<double>& accnp) const override;
      //! @}
     protected:
      //! reset the time step dependent parameters for the element evaluation [derived]
      void reset_eval_params() override;

     private:
      /*! \brief Calculate the right-hand-side vector at \f$ t_{n+1} \f$  [derived]
       *
       *  \f[
       *    Res = F_{inert;n+1}
       *        + F_{visco;n+1}
       *        + F_{int;n+1}
       *        - F_{ext;n+1}
       *  \f]
       *     * Remark: In the case of a Lie group Gen-Alpha time integration scheme,
       *               all forces are evaluated at the end point n+1. */
      void add_visco_mass_contributions(Core::LinAlg::Vector<double>& f) const override;

      /*! \brief Calculate the structural stiffness block at \f$ t_{n+1} \f$  [derived]
       *
       *  \f[
       *    \boldsymbol{K}_{T,effdyn} = \frac{1 - \alpha_m}{\beta (\Delta t)^{2} (1 - \alpha_f) }
       * \boldsymbol{M}
       *                                + \frac{\gamma}{\beta \Delta t} \boldsymbol{C}
       *                                + \boldsymbol{K}_{T}
       *  \f] */
      void add_visco_mass_contributions(Core::LinAlg::SparseOperator& jac) const override;

      /*! \brief Update constant contributions of the current state for the new time step
       * \f$ t_{n+1} \f$ based on the generalized alpha scheme for Lie group extensions:
       *
       * \f[
       *     V_{n+1} = (1.0 - \gamma/(2.0 * \beta)) * dt * A_{mod,n} + (1.0 - \gamma/\beta) * V_{n}
       *             - \gamma/(\beta * dt) * D_{n} + \gamma/(\beta * dt) * D_{n+1}
       *     A_{n+1} = [ \alpha_m/(1.0 - \alpha_f) - (1.0 - \alpha_m) * (0.5 - \beta) /
       *               (\beta * (1.0 - \alpha_f) ] * A_{mod,n} - \alpha_f/(1.0 - \alpha_f) * A_{n}
       *             - (1.0 - \alpha_m)/(\beta * dt * (1.0 - \alpha_f)) * V_{n}
       *             - (1.0 - \alpha_m)/(\beta * dt^2 * (1.0 - \alpha_f) * D_{n}
       *             + (1.0 - \alpha_m)/(\beta * dt^2 * (1.0 - \alpha_f) * D_{n+1}
       * \f]
       *
       * Only the constant contributions, i.e. all components that depend on the state n are stored
       * in the const_vel_acc_update_ptr_ multi-vector pointer. The 1st entry represents the
       * velocity, and the 2nd the acceleration.
       *
       *  See the set_state() routine for the iterative update of the current state.
       *
       *  \f$ A_{mod,n} \f$ is updated in the method update_step_state() according to
       *  \f[
       *    A_{mod,n+1} =
       *    \frac{1}{1-\alpha_m} ( (1-\alpha_f) A_{n+1} + \alpha_f A_n - \alpha_m * A_{mod,n} )
       * \f]
       *
       * \note: For more information see e.g. the dissertation of Christoph Meier: "Geometrically
       * exact finite element formulations for slender beams and their contact interaction"
       * */
      void update_constant_state_contributions() override;

     private:
      /*! @name New vectors for internal use only
       *
       *  If an external use seems necessary, move these vectors to the
       *  global state data container and just store a pointer to the global
       *  state variable. */
      //! @{
      //! modified acceleration vector acc_{mod;n}
      std::shared_ptr<Core::LinAlg::Vector<double>> accn_mod_;

      //! @}
    };
  }  // namespace IMPLICIT
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
