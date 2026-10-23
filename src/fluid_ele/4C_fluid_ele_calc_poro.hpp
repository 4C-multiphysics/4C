// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FLUID_ELE_CALC_PORO_HPP
#define FOUR_C_FLUID_ELE_CALC_PORO_HPP


#include "4C_config.hpp"

#include "4C_fluid_ele_calc.hpp"
#include "4C_structure_new_input.hpp"
#include "4C_utils_singleton_owner.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  class StructPoro;
}

namespace Discret
{
  namespace Elements
  {
    //! Class for Evaluating boundary integrals for porous media problems
    /*!
     This class is derived from the FluidEleCalc class, i.e. it is capable of
     evaluated all integrals implemented there. It will do so, if the evaluate action
     given by the control routine is not known (see method Evaluate).

     The main methods are the Evaluate and the evaluate_od routines. Therein,
     the stiffness matrixes of a porous fluid problem are evaluated. OD means
     off diagonal, indicating linearizations with respect to structural degrees of freedom,
     that will be assembled into off diagonal entries in the global system matrix.
     The terms are eventually evaluated in the gauss_point_loop.. methods

     This a calculation class implemented as a singleton, like all calc classes in fluid
     (see comments on base classes for more details). In short this means that on instance
     exists for every discretization type of the boundary element (because of the template).
     */

    template <Core::FE::CellType distype>
    class FluidEleCalcPoro : public FluidEleCalc<distype>
    {
      using Base = Discret::Elements::FluidEleCalc<distype, Discret::Elements::Fluid::none>;
      using Base::numderiv2_;

     protected:
      using Base::nen_;
      using Base::nsd_;

      //! private Constructor since we are a Singleton.
      FluidEleCalcPoro();

     public:
      //! Singleton access method
      static FluidEleCalcPoro<distype>* instance(
          Core::Utils::SingletonAction action = Core::Utils::SingletonAction::create);

      /*!
        \brief calculate element matrix and rhs for porous flow

        \param eid              (i) element id
        \param discretization   (i) fluid discretization the element belongs to
        \param lm               (i) location matrix of element
        \param params           (i) element parameter list
        \param mat              (i) material
        \param elemat1          (o) element matrix to calculate
        \param elemat2          (o) element matrix to calculate
        \param elevec1          (o) element vector to calculate
        \param elevec2          (o) element vector to calculate
        \param elevec3          (o) element vector to calculate
        \param offdiag          (i) flag indicating whether diagonal or off diagonal blocks are to
        be calculated

       */
      int evaluate(Discret::Elements::Fluid* ele, Core::FE::Discretization& discretization,
          const std::vector<int>& lm, Teuchos::ParameterList& params,
          std::shared_ptr<Core::Mat::Material>& mat, Core::LinAlg::SerialDenseMatrix& elemat1,
          Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseVector& elevec2, Core::LinAlg::SerialDenseVector& elevec3,
          bool offdiag = false) override;

      /*!
        \brief calculate element matrix and rhs for porous flow (2)

        \param eid              (i) element id
        \param discretization   (i) fluid discretization the element belongs to
        \param lm               (i) location matrix of element
        \param params           (i) element parameter list
        \param mat              (i) material
        \param elemat1          (o) element matrix to calculate
        \param elemat2          (o) element matrix to calculate
        \param elevec1          (o) element vector to calculate
        \param elevec2          (o) element vector to calculate
        \param elevec3          (o) element vector to calculate
        \param intpoints        (i) Gaussian integration points

       */
      virtual int evaluate(Discret::Elements::Fluid* ele, Core::FE::Discretization& discretization,
          const std::vector<int>& lm, Teuchos::ParameterList& params,
          std::shared_ptr<Core::Mat::Material>& mat, Core::LinAlg::SerialDenseMatrix& elemat1,
          Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseVector& elevec2, Core::LinAlg::SerialDenseVector& elevec3,
          const Core::FE::GaussIntegration& intpoints);

      /*!
        \brief Evaluate coupling terms (off diagonal terms) of the element at specified gauss points
        for porous flow

        \param eid              (i) element id
        \param discretization   (i) fluid discretization the element belongs to
        \param lm               (i) location matrix of element
        \param params           (i) element parameter list
        \param mat              (i) material
        \param elemat1          (o) element matrix to calculate
        \param elemat2          (o) element matrix to calculate
        \param elevec1          (o) element vector to calculate
        \param elevec2          (o) element vector to calculate
        \param elevec3          (o) element vector to calculate
        \param intpoints        (i) Gaussian integration points

       */
      virtual int evaluate_od(Discret::Elements::Fluid* ele,
          Core::FE::Discretization& discretization, const std::vector<int>& lm,
          Teuchos::ParameterList& params, std::shared_ptr<Core::Mat::Material>& mat,
          Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
          Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
          Core::LinAlg::SerialDenseVector& elevec3, const Core::FE::GaussIntegration& intpoints);


      //! Evaluate supporting methods of the element
      /*!
          Interface function for supporting methods of the element
       */
      int evaluate_service(Discret::Elements::Fluid* ele, Teuchos::ParameterList& params,
          std::shared_ptr<Core::Mat::Material>& mat, Core::FE::Discretization& discretization,
          std::vector<int>& lm, Core::LinAlg::SerialDenseMatrix& elemat1,
          Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseVector& elevec2,
          Core::LinAlg::SerialDenseVector& elevec3) override;

     protected:
      /*!
          \brief evaluate function for Fluid element for porous flow

          Specific evaluate function without any knowledge about discretization objects. This
          way the element evaluation is independent of the specific mesh storage.

          \param params           (i) element parameter list
          \param ebofoaf          (i) body force at time step n+alpha_F/n+1
          \param elemat1          (o) element matrix to be filled
          \param elevec1          (o) element rhs vector to be filled
          \param evelaf           (i) nodal velocities at n+alpha_F/n+1
          \param epreaf           (i) nodal pressure at time step n+alpha_F/n+1
          \param evelnp           (i) nodal velocities time step at n+1
          \param eveln            (i) nodal velocities time step at n
          \param eprenp           (i) nodal pressure at time step n+alpha_F/n+1
          \param epren            (i) nodal pressure at time step n
          \param emhist           (i) time rhs for momentum equation
          \param echist           (i) time rhs for continuity equation
          \param epressnp_timederiv (i) nodal pressure time derivative at time step n+alpha_F/n+1
          \param eaccam           (i) nodal accelerations at time step n+alpha_M
          \param edispnp          (i) nodal displacements at time step n+1 (on moving mesh)
          \param edispn           (i) nodal displacements at time step n (on moving mesh)
          \param egridv           (i) grid velocity at time step n+1
          \param egridvn          (i) grid velocity at time step n
          \param escaaf           (i) nodal scalar at time step n+alpha_F/n+1
          \param material         (i) fluid material
          \param isale            (i) ALE flag
          \param intpoints        (i) Gaussian integration points
          \param ele             (i) element pointer
       */
      int evaluate(Teuchos::ParameterList& params, const Core::LinAlg::Matrix<nsd_, nen_>& ebofoaf,
          Core::LinAlg::Matrix<(nsd_ + 1) * nen_, (nsd_ + 1) * nen_>& elemat1,
          Core::LinAlg::Matrix<(nsd_ + 1) * nen_, 1>& elevec1,
          const Core::LinAlg::Matrix<nsd_, nen_>& evelaf,
          const Core::LinAlg::Matrix<nen_, 1>& epreaf,
          const Core::LinAlg::Matrix<nsd_, nen_>& evelnp,
          const Core::LinAlg::Matrix<nsd_, nen_>& eveln,
          const Core::LinAlg::Matrix<nen_, 1>& eprenp, const Core::LinAlg::Matrix<nen_, 1>& epren,
          const Core::LinAlg::Matrix<nsd_, nen_>& emhist,
          const Core::LinAlg::Matrix<nen_, 1>& echist,
          const Core::LinAlg::Matrix<nen_, 1>& epressnp_timederiv,
          const Core::LinAlg::Matrix<nen_, 1>& epressam_timederiv,
          const Core::LinAlg::Matrix<nen_, 1>& epressn_timederiv,
          const Core::LinAlg::Matrix<nsd_, nen_>& eaccam,
          const Core::LinAlg::Matrix<nsd_, nen_>& edispnp,
          const Core::LinAlg::Matrix<nsd_, nen_>& edispn,
          const Core::LinAlg::Matrix<nsd_, nen_>& egridv,
          const Core::LinAlg::Matrix<nsd_, nen_>& egridvn,
          const Core::LinAlg::Matrix<nen_, 1>& escaaf,
          const Core::LinAlg::Matrix<nen_, 1>* eporositynp,
          const Core::LinAlg::Matrix<nen_, 1>* eporositydot,
          const Core::LinAlg::Matrix<nen_, 1>* eporositydotn,
          std::shared_ptr<Core::Mat::Material> mat, bool isale,
          const Core::FE::GaussIntegration& intpoints,
          Discret::Elements::Fluid* ele);

      /*!
          \brief evaluate function for Fluid element for porous flow

          Specific evaluate function without any knowledge about discretization objects. This
          way the element evaluation is independent of the specific mesh storage.

              \param params           (i) element parameter list
              \param elemat1          (o) element matrix to be filled
              \param elevec1          (o) element rhs vector to be filled
              \param evelaf           (i) nodal velocities at n+alpha_F/n+1
              \param epreaf           (i) nodal pressure at n+alpha_F/n+1
              \param evelnp           (i) nodal velocities at n+1 (np_genalpha)
              \param eveln            (i) nodal velocities at n (np_genalpha)
              \param eprenp           (i) nodal pressure at n+1
              \param epren            (i) nodal pressure at n
              \param epressnp_timederiv (i) nodal pressure time derivative at n+alpha_F/n+1
              \param eaccam           (i) nodal accelerations at n+alpha_M
              \param edispnp          (i) nodal displacements at n+1 (on moving mesh)
              \param edispn           (i) nodal displacements at n (on moving mesh)
              \param egridv           (i) grid velocity at n+1
              \param egridvn          (i) grid velocity at n
              \param escaaf           (i) nodal scalar at n+alpha_F/n+1
              \param eporositynp      (i) nodal porosity at n+1
              \param mat              (i) fluid material
              \param isale            (i) ALE flag
              \param intpoints        (i) Gaussian integration points
              \param ele              (i) element pointer
       */
      int evaluate_od(Teuchos::ParameterList& params,
          const Core::LinAlg::Matrix<nsd_, nen_>& ebofoaf,
          Core::LinAlg::Matrix<(nsd_ + 1) * nen_, nsd_ * nen_>& elemat1,
          Core::LinAlg::Matrix<(nsd_ + 1) * nen_, 1>& elevec1,
          const Core::LinAlg::Matrix<nsd_, nen_>& evelaf,
          const Core::LinAlg::Matrix<nen_, 1>& epreaf,
          const Core::LinAlg::Matrix<nsd_, nen_>& evelnp,
          const Core::LinAlg::Matrix<nsd_, nen_>& eveln,
          const Core::LinAlg::Matrix<nen_, 1>& eprenp, const Core::LinAlg::Matrix<nen_, 1>& epren,
          const Core::LinAlg::Matrix<nen_, 1>& epressnp_timederiv,
          const Core::LinAlg::Matrix<nen_, 1>& epressam_timederiv,
          const Core::LinAlg::Matrix<nen_, 1>& epressn_timederiv,
          const Core::LinAlg::Matrix<nsd_, nen_>& eaccam,
          const Core::LinAlg::Matrix<nsd_, nen_>& edispnp,
          const Core::LinAlg::Matrix<nsd_, nen_>& edispn,
          const Core::LinAlg::Matrix<nsd_, nen_>& egridv,
          const Core::LinAlg::Matrix<nsd_, nen_>& egridvn,
          const Core::LinAlg::Matrix<nen_, 1>& escaaf,
          const Core::LinAlg::Matrix<nsd_, nen_>& emhist,
          const Core::LinAlg::Matrix<nen_, 1>& echist,
          const Core::LinAlg::Matrix<nen_, 1>* eporositynp,
          std::shared_ptr<Core::Mat::Material> mat, bool isale,
          const Core::FE::GaussIntegration& intpoints,
          Discret::Elements::Fluid* ele);

      /*!
        \brief calculate element matrix and rhs for porous flow

        \param params           (i) element parameter list
        \param ebofoaf          (i) body force at n+alpha_F/n+1
        \param evelaf           (i) nodal velocities at n+alpha_F/n+1
        \param evelnp           (i) nodal velocities at n+1 (np_genalpha)
        \param eveln            (i) nodal velocities at n (np_genalpha)
        \param epreaf           (i) nodal pressure at n+alpha_F/n+1
        \param eprenp           (i) nodal pressure at n+alpha_F/n+1
        \param epren            (i) nodal pressure at n
        \param eaccam           (i) nodal accelerations at n+alpha_M
        \param emhist           (i) time rhs for momentum equation
        \param echist           (i) time rhs for continuity equation
        \param epressnp_timederiv (i) nodal pressure time derivative at n+alpha_F/n+1
        \param edispnp          (i) nodal displacements at n+1 (on moving mesh)
        \param edispn           (i) nodal displacements at n (on moving mesh)
        \param egridv           (i) grid velocity at n+1
        \param egridvn          (i) grid velocity at n
        \param escaaf           (i) nodal scalar at n+alpha_F/n+1
        \param eporositynp      (i) nodal porosity at n+1
        \param eporositydot     (i) nodal porosity time derivative at n+1
        \param eporositydot     (i) nodal porosity time derivative at n
        \param estif            (o) element matrix to calculate
        \param eforce           (o) element rhs to calculate
        \param material         (i) fluid material
        \param isale            (i) ALE flag
        \param intpoints        (i) Gaussian integration points
        \param ele              (i) element pointer

       */
      void sysmat(Teuchos::ParameterList& params, const Core::LinAlg::Matrix<nsd_, nen_>& ebofoaf,
          const Core::LinAlg::Matrix<nsd_, nen_>& evelaf,
          const Core::LinAlg::Matrix<nsd_, nen_>& evelnp,
          const Core::LinAlg::Matrix<nsd_, nen_>& eveln,
          const Core::LinAlg::Matrix<nen_, 1>& epreaf, const Core::LinAlg::Matrix<nen_, 1>& eprenp,
          const Core::LinAlg::Matrix<nen_, 1>& epren,
          const Core::LinAlg::Matrix<nsd_, nen_>& eaccam,
          const Core::LinAlg::Matrix<nsd_, nen_>& emhist,
          const Core::LinAlg::Matrix<nen_, 1>& echist,
          const Core::LinAlg::Matrix<nen_, 1>& epressnp_timederiv,
          const Core::LinAlg::Matrix<nen_, 1>& epressam_timederiv,
          const Core::LinAlg::Matrix<nen_, 1>& epressn_timederiv,
          const Core::LinAlg::Matrix<nsd_, nen_>& edispnp,
          const Core::LinAlg::Matrix<nsd_, nen_>& edispn,
          const Core::LinAlg::Matrix<nsd_, nen_>& egridv,
          const Core::LinAlg::Matrix<nsd_, nen_>& egridvn,
          const Core::LinAlg::Matrix<nen_, 1>& escaaf,
          const Core::LinAlg::Matrix<nen_, 1>* eporositynp,
          const Core::LinAlg::Matrix<nen_, 1>* eporositydot,
          const Core::LinAlg::Matrix<nen_, 1>* eporositydotn,
          Core::LinAlg::Matrix<(nsd_ + 1) * nen_, (nsd_ + 1) * nen_>& estif,
          Core::LinAlg::Matrix<(nsd_ + 1) * nen_, 1>& eforce,
          std::shared_ptr<const Core::Mat::Material> material, bool isale,
          const Core::FE::GaussIntegration& intpoints,
          Discret::Elements::Fluid* ele);

      /*!
        \brief calculate off diagonal element matrix and rhs for porous flow

        \param params           (i) element parameter list
        \param evelaf           (i) nodal velocities at n+alpha_F/n+1
        \param evelnp           (i) nodal velocities at n+1 (np_genalpha)
        \param epreaf           (i) nodal pressure at n+alpha_F/n+1
        \param eprenp           (i) nodal pressure at n+alpha_F/n+1
        \param epren            (i) nodal pressure at n
        \param eaccam           (i) nodal accelerations at n+alpha_M
        \param epressnp_timederiv (i) nodal pressure time derivative at n+alpha_F/n+1
        \param edispnp          (i) nodal displacements at n+1 (on moving mesh)
        \param edispn           (i) nodal displacements at n (on moving mesh)
        \param egridv           (i) grid velocity at n+1
        \param egridvn          (i) grid velocity at n
        \param escaaf           (i) nodal scalar at n+alpha_F/n+1
        \param emhist           (i) time rhs for momentum equation
        \param echist           (i) time rhs for continuity equation
        \param eporositynp      (i) nodal porosity at n+1
        \param ecoupl           (o) element matrix to calculate
        \param eforce           (o) element rhs to calculate
        \param material         (i) fluid material
        \param isale            (i) ALE flag
        \param intpoints        (i) Gaussian integration points
        \param ele              (i) element pointer
       */
      void sysmat_od(Teuchos::ParameterList& params,
          const Core::LinAlg::Matrix<nsd_, nen_>& ebofoaf,
          const Core::LinAlg::Matrix<nsd_, nen_>& evelaf,
          const Core::LinAlg::Matrix<nsd_, nen_>& evelnp,
          const Core::LinAlg::Matrix<nsd_, nen_>& eveln,
          const Core::LinAlg::Matrix<nen_, 1>& epreaf, const Core::LinAlg::Matrix<nen_, 1>& eprenp,
          const Core::LinAlg::Matrix<nen_, 1>& epren,
          const Core::LinAlg::Matrix<nsd_, nen_>& eaccam,
          const Core::LinAlg::Matrix<nen_, 1>& epressnp_timederiv,
          const Core::LinAlg::Matrix<nen_, 1>& epressam_timederiv,
          const Core::LinAlg::Matrix<nen_, 1>& epressn_timederiv,
          const Core::LinAlg::Matrix<nsd_, nen_>& edispnp,
          const Core::LinAlg::Matrix<nsd_, nen_>& edispn,
          const Core::LinAlg::Matrix<nsd_, nen_>& egridv,
          const Core::LinAlg::Matrix<nsd_, nen_>& egridvn,
          const Core::LinAlg::Matrix<nen_, 1>& escaaf,
          const Core::LinAlg::Matrix<nsd_, nen_>& emhist,
          const Core::LinAlg::Matrix<nen_, 1>& echist,
          const Core::LinAlg::Matrix<nen_, 1>* eporositynp,
          Core::LinAlg::Matrix<(nsd_ + 1) * nen_, nsd_ * nen_>& ecoupl,
          Core::LinAlg::Matrix<(nsd_ + 1) * nen_, 1>& eforce,
          std::shared_ptr<const Core::Mat::Material> material, bool isale,
          const Core::FE::GaussIntegration& intpoints,
          Discret::Elements::Fluid* ele);

      /*!
        \brief linearisation of momentum equation in the case of mesh motion 3-D for Poroelasticity

        \param ecoupl_u           (o) coupling element matrix for fluid velocity
        \param dphi_dp            (i) derivative of porosity w.r.t. fluid pressure
        \param dphi_dJ            (i) derivative of porosity w.r.t. jacobian determinant
        \param refporositydot     (i) time derivative of reference porosity
        \param timefac            (i) time factor
        \param timefacfac         (i) time factor * integration factor
       */
      void lin_3d_mesh_motion_od(Core::LinAlg::Matrix<nsd_ * nen_, nsd_ * nen_>& ecoupl_u,
          const double& dphi_dp, const double& dphi_dJ, const double& refporositydot,
          const double& timefac, const double& timefacfac);

      /*!
        \brief linearisation of momentum equation in the case of mesh motion 2-D for Poroelasticity

        \param ecoupl_u           (o) coupling element matrix for fluid velocity
        \param dphi_dp            (i) derivative of porosity w.r.t. fluid pressure
        \param dphi_dJ            (i) derivative of porosity w.r.t. jacobian determinant
        \param refporositydot     (i) time derivative of reference porosity
        \param timefac            (i) time factor
        \param timefacfac         (i) time factor * integration factor
       */
      void lin_2d_mesh_motion_od(Core::LinAlg::Matrix<nsd_ * nen_, nsd_ * nen_>& ecoupl_u,
          const double& dphi_dp, const double& dphi_dJ, const double& refporositydot,
          const double& timefac, const double& timefacfac);

      /*!
        \brief linearisation of continuity equation in the case of mesh motion 2-D for
        Poroelasticity

        \param ecoupl_p           (o) coupling element matrix for fluid pressure
        \param dphi_dp            (i) derivative of porosity w.r.t. fluid pressure
        \param dphi_dJ            (i) derivative of porosity w.r.t. jacobian determinant
        \param refporositydot     (i) time derivative of reference porosity
        \param timefacfacpre      (i) time factor * integration factor
       */
      void lin_mesh_motion_2d_pres_od(Core::LinAlg::Matrix<nen_, nsd_ * nen_>& ecoupl_p,
          const double& dphi_dp, const double& dphi_dJ, const double& refporositydot,
          const double& timefacfacpre);

      /*!
        \brief linearisation of continuity equation in the case of mesh motion 3-D for
        Poroelasticity

        \param ecoupl_p           (o) coupling element matrix for fluid pressure
        \param dphi_dp            (i) derivative of porosity w.r.t. fluid pressure
        \param dphi_dJ            (i) derivative of porosity w.r.t. jacobian determinant
        \param refporositydot     (i) time derivative of reference porosity
        \param timefacfacpre      (i) time factor * integration factor
       */
      virtual void lin_mesh_motion_3d_pres_od(Core::LinAlg::Matrix<nen_, nsd_ * nen_>& ecoupl_p,
          const double& dphi_dp, const double& dphi_dJ, const double& refporositydot,
          const double& timefacfacpre);

      //! Compute element matrix entries: PSPG
      virtual void pspg(
          Core::LinAlg::Matrix<nen_, nen_ * nsd_>& estif_q_u,  //!< block (weighting function q x u)
          Core::LinAlg::Matrix<nen_, nen_>& ppmat,             //!< block (weighting function q x p)
          Core::LinAlg::Matrix<nen_, 1>& preforce,             //!< rhs forces pressure
          const Core::LinAlg::Matrix<nsd_ * nsd_, nen_>&
              lin_resM_Du,  //!< linearisation of the stabilization residual
          const Core::LinAlg::Matrix<nsd_ * nsd_, nen_>&
              lin_resMRea_Du,  //!< linearisation of the stabilization residual (reactive terms)
          const Core::LinAlg::Matrix<nsd_, nen_>&
              lin_resM_Dp,        //!< linearisation of the stabilization residual w.r.t. pressure
          const double& dphi_dp,  //!< linearisation of porosity w.r.t. pressure
          const double& fac3,     //!< factor for residual in current subgrid velocities
          const double& timefacfac,     //!< = timefac x fac
          const double& timefacfacpre,  //!< = timefacpre x fac
          const double& rhsfac          //!< right-hand-side factor for residuals
      );

      //! Compute element matrix entries: Biot
      virtual void stab_biot(
          Core::LinAlg::Matrix<nen_, nen_ * nsd_>& estif_q_u,  //!< block (weighting function q x u)
          Core::LinAlg::Matrix<nen_, nen_>& ppmat,             //!< block (weighting function q x p)
          Core::LinAlg::Matrix<nen_, 1>& preforce,             //!< rhs forces pressure
          const Core::LinAlg::Matrix<nsd_ * nsd_, nen_>&
              lin_resM_Du,  //!< linearisation of the stabilization residual
          const Core::LinAlg::Matrix<nsd_ * nsd_, nen_>&
              lin_resMRea_Du,  //!< linearisation of the stabilization residual (reactive terms)
          const Core::LinAlg::Matrix<nsd_, nen_>&
              lin_resM_Dp,        //!< linearisation of the stabilization residual w.r.t. pressure
          const double& dphi_dp,  //!< linearisation of porosity w.r.t. pressure
          const double& fac3,     //!< factor for residual in current subgrid velocities
          const double& timefacfac,     //!< = timefac x fac
          const double& timefacfacpre,  //!< = timefacpre x fac
          const double& rhsfac          //!< right-hand-side factor for residuals
      );

      //! Compute element matrix entries: reactive stabilization
      virtual void reac_stab(Core::LinAlg::Matrix<nen_ * nsd_, nen_ * nsd_>&
                                 estif_u,                      //!< block (weighting function v x u)
          Core::LinAlg::Matrix<nen_ * nsd_, nen_>& estif_p_v,  //!< block (weighting function v x p)
          Core::LinAlg::Matrix<nsd_, nen_>& velforce,          //!< rhs forces velocity
          Core::LinAlg::Matrix<nsd_ * nsd_, nen_>&
              lin_resM_Du,  //!< linearisation of the stabilization residual
          const Core::LinAlg::Matrix<nsd_, nen_>&
              lin_resM_Dp,        //!< linearisation of the stabilization residual w.r.t. pressure
          const double& dphi_dp,  //!< linearisation of porosity w.r.t. pressure
          const double& timefacfac,     //!< = timefac x fac
          const double& timefacfacpre,  //!< = timefacpre x fac
          const double& rhsfac,         //!< right-hand-side factor for residuals
          const double& fac3            //!< factor for residual in current subgrid velocities
      );

      //! Compute linearization of momentum residual w.r.t fluid velocities
      void compute_lin_res_m_du(const double& timefacfac,
          Core::LinAlg::Matrix<nsd_ * nsd_, nen_>& lin_resM_Du,
          Core::LinAlg::Matrix<nsd_ * nsd_, nen_>& lin_resMRea_Du);

      //! Compute linearization of momentum residual (stabilization) w.r.t fluid velocities
      void compute_lin_res_m_du_stabilization(
          const double& timefacfac, Core::LinAlg::Matrix<nsd_ * nsd_, nen_>& lin_resM_Du);

      //! Compute linearization of momentum residual w.r.t fluid pressure
      void compute_lin_res_m_dp(const double& timefacfacpre, const double& dphi_dp,
          Core::LinAlg::Matrix<nsd_, nen_>& lin_resM_Dp);

      //! calculate div(epsilon(u))
      void calc_div_eps(
          const Core::LinAlg::Matrix<nsd_, nen_>& evelaf  //!< velocity at time n+alpha_f / n+1
      );

      /*!
        \brief Compute derivatives of deformation gradient

        \param edispnp         (i) nodal displacements at n+1
        \param defgrd_inv      (i) inverse deformation gradient
        \param F_x             (o) derivative of deformation gradient w.r.t. current coordinates xyz
        \param F_X             (o) derivative of deformation gradient w.r.t. material coordinates
        XYZ
       */
      void compute_f_derivative(const Core::LinAlg::Matrix<nsd_, nen_>& edispnp,
          const Core::LinAlg::Matrix<nsd_, nsd_>& defgrd_inv,
          Core::LinAlg::Matrix<nsd_ * nsd_, nsd_>& F_x,
          Core::LinAlg::Matrix<nsd_ * nsd_, nsd_>& F_X);

      /*!
        \brief Compute spatial gradient of jacobian determinant and porosity

        \param J              (i) determinant of deformation gradient at gauss point
        \param dphidp         (i) derivative of porosity w.r.t. fluid pressure
        \param dphidJ         (i) derivative of porosity w.r.t. jacobian determinant
        \param defgrd_IT_vec  (i) inverse transposed deformation gradient in vector notation
        \param F_x            (i) derivative of deformation gradient w.r.t. current coordinates xyz
        \param gradp          (i) spatial gradient of fluid pressure
        \param eporositynp    (i) nodal porosities at n+1
        \param gradJ          (o) spatial gradient of jacobian determinant
        \param grad_porosity  (o) spatial gradient of porosity
        \param refgrad_porosity  (o) reference gradient of porosity
       */
      void compute_gradients(const double& J, const double& dphidp, const double& dphidJ,
          const Core::LinAlg::Matrix<nsd_ * nsd_, 1>& defgrd_IT_vec,
          const Core::LinAlg::Matrix<nsd_ * nsd_, nsd_>& F_x,
          const Core::LinAlg::Matrix<nsd_, 1>& gradp,
          const Core::LinAlg::Matrix<nen_, 1>* eporositynp, Core::LinAlg::Matrix<nsd_, 1>& gradJ,
          Core::LinAlg::Matrix<nsd_, 1>& grad_porosity,
          Core::LinAlg::Matrix<nsd_, 1>& refgrad_porosity);

      /*!
        \brief Compute spatial gradient of porosity

        \param dphidp         (i) derivative of porosity w.r.t. fluid pressure
        \param dphidJ         (i) derivative of porosity w.r.t. jacobian determinant
        \param gradJ          (i) spatial gradient of jacobian determinant
        \param gradJ          (i) spatial gradient of fluid pressure
        \param eporositynp    (i) nodal porosities at n+1
        \param grad_porosity  (o) spatial gradient of porosity
        \param refgrad_porosity  (o) reference gradient of porosity
       */
      virtual void compute_porosity_gradient(const double& dphidp, const double& dphidJ,
          const Core::LinAlg::Matrix<nsd_, 1>& gradJ, const Core::LinAlg::Matrix<nsd_, 1>& gradp,
          const Core::LinAlg::Matrix<nen_, 1>* eporositynp,
          Core::LinAlg::Matrix<nsd_, 1>& grad_porosity,
          Core::LinAlg::Matrix<nsd_, 1>& refgrad_porosity);

      /*!
        \brief Compute linearization needed for diagonal terms (lin. of porosity gradient w.r.t.
        fluid pressure)

        \param dphidp         (i) derivative of porosity w.r.t. fluid pressure
        \param dphi_dpp       (i) second derivative of porosity w.r.t. fluid pressure
        \param dphi_dJp       (i) mixed derivative of porosity w.r.t. fluid pressure and jacobian
                                  determinant
        \param gradJ          (i) spatial gradient of jacobian determinant
        \param dgradphi_dp    (o) derivate of spatial gradient of porosity w.r.t. fluid pressure
       */
      virtual void compute_linearization(const double& dphi_dp, const double& dphi_dpp,
          const double& dphi_dJdp, const Core::LinAlg::Matrix<nsd_, 1>& gradJ,
          Core::LinAlg::Matrix<nsd_, nen_>& dgradphi_dp);

      /*!
        \brief Compute linearization needed for off diagonal terms
          (lin. of jacobian, porosity and porosity gradient w.r.t. structure displacement)

        \param dphi_dJ        (i) derivative of porosity w.r.t. jacobian determinant
        \param dphi_dJJ       (i) second derivative of porosity w.r.t. jacobian determinant
        \param dphi_dJp       (i) mixed derivative of porosity w.r.t. fluid pressure and jacobian
                                  determinant
        \param defgrd_inv     (i) inverse deformation gradient
        \param defgrd_IT_vec  (i) inverse transposed deformation gradient in vector notation
        \param F_x            (i) derivative of deformation gradient w.r.t. current coordinates xyz
        \param F_X            (i) derivative of deformation gradient w.r.t. material coordinates
                                  XYZ
        \param gradJ          (i) spatial gradient of jacobian determinant
        \param dJ_dus         (o) derivative of jacobian determinant w.r.t. structure displacements
        \param dphi_dus       (o) derivative of porosity determinant w.r.t. structure displacements
        \param dgradphi_dus   (o) derivate of spatial gradient of porosity w.r.t. structure
                                  displacements
       */
      virtual void compute_linearization_od(const double& dphi_dJ, const double& dphi_dJJ,
          const double& dphi_dJp, const Core::LinAlg::Matrix<nsd_, nsd_>& defgrd_inv,
          const Core::LinAlg::Matrix<nsd_ * nsd_, 1>& defgrd_IT_vec,
          const Core::LinAlg::Matrix<nsd_ * nsd_, nsd_>& F_x,
          const Core::LinAlg::Matrix<nsd_ * nsd_, nsd_>& F_X,
          const Core::LinAlg::Matrix<nsd_, 1>& gradJ, Core::LinAlg::Matrix<1, nsd_ * nen_>& dJ_dus,
          Core::LinAlg::Matrix<1, nsd_ * nen_>& dphi_dus,
          Core::LinAlg::Matrix<nsd_, nen_ * nsd_>& dgradphi_dus);

      /*!
        \brief Compute porosity and derivatives

        \param params       (i) element parameter list
        \param press        (i) fluid pressure at gauss point
        \param J            (i) jacobian determinant at gauss point
        \param gp           (i) number of actual gauss point
        \param shapfct      (i) shape function values at gauss point
        \param myporosity   (i) nodal porosities
        \param porosity     (o) porosity at gauss point
        \param dphi_dp      (o) derivative of porosity gradient w.r.t. fluid pressure
        \param dphi_dJ      (o) derivative of porosity gradient w.r.t. jacobian determinant
        \param dphi_dJdp    (o) mixed derivative of porosity gradient w.r.t. fluid pressure and
                                jacobian determinant
        \param dphi_dJJ     (o) second derivative of porosity gradient w.r.t. jacobian determinant
        \param dphi_dpp     (o) second derivative of porosity gradient w.r.t. fluid pressure
        \param save         (i) flag for saving porosity within structure material
        \param eleGID       (i) element global ID
       */
      virtual void compute_porosity(Teuchos::ParameterList& params, const double& press,
          const double& J, const int& gp, const Core::LinAlg::Matrix<nen_, 1>& shapfct,
          const Core::LinAlg::Matrix<nen_, 1>* myporosity, double& porosity, double* dphi_dp,
          double* dphi_dJ, double* dphi_dJdp, double* dphi_dJJ, double* dphi_dpp, bool save, int eleGID);

      /*!
       \brief evaluate pressure equation (i.e. continuity equation for standard poro elements)
              This function is overwritten by the poro_p1 element

        \param params         (i) element parameter list
        \param timefacfacpre  (i) fluid pressure at gauss point
        \param rhsfac         (i) jacobian determinant at gauss point
        \param dphi_dp        (i) derivative of porosity gradient w.r.t. fluid pressure
        \param dphi_dJ        (i) derivative of porosity gradient w.r.t. jacobian determinant
        \param dphi_dJdp      (i) mixed derivative of porosity gradient w.r.t. fluid pressure and
                                  jacobian determinant
        \param dphi_dJJ       (i) second derivative of porosity gradient w.r.t. jacobian
                                  determinant
        \param dphi_dpp       (i) second derivative of porosity gradient w.r.t. fluid pressure
        \param eporositydot   (i) nodal values of porosity time derivative at time step n+1
        \param eporositydotn  (i) nodal values of porosity time derivative at time step n
        \param echist         (i) nodal values of history values of continuity equation
        \param dgradphi_dp    (i) derivative of porosity gradient w.r.t. fluid pressure
        \param estif_q_u      (o) element matrix (pressure - fluid velocity weighting) to be filled
        \param ppmat          (o) element matrix (pressure - pressure weighting) to be filled
        \param preforce       (o) element rhs vector to be filled
       * */
      virtual void evaluate_pressure_equation(Teuchos::ParameterList& params,
          const double& timefacfacpre, const double& rhsfac, const double& dphi_dp,
          const double& dphi_dJ, const double& dphi_dJdp, const double& dphi_dpp,
          const Core::LinAlg::Matrix<nen_, 1>* eporositydot,
          const Core::LinAlg::Matrix<nen_, 1>* eporositydotn,
          const Core::LinAlg::Matrix<nen_, 1>& echist,
          const Core::LinAlg::Matrix<nsd_, nen_>& dgradphi_dp,
          Core::LinAlg::Matrix<nen_, nen_ * nsd_>& estif_q_u,
          Core::LinAlg::Matrix<nen_, nen_>& ppmat, Core::LinAlg::Matrix<nen_, 1>& preforce);

      /*!
       \brief evaluate nontransient part of pressure equation (i.e. no time derivative of porosity)

        \param params         (i) element parameter list
        \param timefacfacpre  (i) fluid pressure at gauss point
        \param rhsfac         (i) jacobian determinant at gauss point
        \param dphi_dp        (i) derivative of porosity gradient w.r.t. fluid pressure
        \param dphi_dJ        (i) derivative of porosity gradient w.r.t. jacobian determinant
        \param dphi_dJdp      (i) mixed derivative of porosity gradient w.r.t. fluid pressure and
                                  jacobian determinant
        \param dphi_dJJ       (i) second derivative of porosity gradient w.r.t. jacobian
                                  determinant
        \param dphi_dpp       (i) second derivative of porosity gradient w.r.t. fluid pressure
        \param echist         (i) nodal values of history values of continuity equation
        \param dgradphi_dp    (i) derivative of porosity gradient w.r.t. fluid pressure
        \param estif_q_u      (o) element matrix (pressure - fluid velocity weighting) to be filled
        \param ppmat          (o) element matrix (pressure - pressure weighting) to be filled
        \param preforce       (o) element rhs vector to be filled
       * */
      virtual void evaluate_pressure_equation_non_transient(Teuchos::ParameterList& params,
          const double& timefacfacpre, const double& rhsfac, const double& dphi_dp,
          const double& dphi_dJ, const double& dphi_dJdp, const double& dphi_dpp,
          const Core::LinAlg::Matrix<nsd_, nen_>& dgradphi_dp,
          Core::LinAlg::Matrix<nen_, nen_ * nsd_>& estif_q_u,
          Core::LinAlg::Matrix<nen_, nen_>& ppmat, Core::LinAlg::Matrix<nen_, 1>& preforce);

      /*!
        \brief Gauss point loop for evaluation of diagonal terms

        \param params             (i) element parameter list
        \param evelaf             (i) nodal velocities at n+alpha_F/n+1
        \param evelnp             (i) nodal velocities at n+1 (np_genalpha)
        \param epreaf             (i) nodal pressure at n+alpha_F/n+1
        \param eprenp             (i) nodal pressure at n+alpha_F/n+1
        \param epressnp_timederiv (i) nodal pressure time derivative at n+alpha_F/n+1
        \param eaccam             (i) nodal accelerations at n+alpha_M
        \param edispnp            (i) nodal displacements at n+1 (on moving mesh)
        \param egridv             (i) grid velocity at n+1
        \param escaaf             (i) nodal scalar at n+alpha_F/n+1
        \param eporositynp        (i) nodal porosity at n+1
        \param eporositydot       (i) nodal porosity time derivative at n+1
        \param eporositydotn      (i) nodal porosity time derivative at n
        \param estif_u            (o) element matrix for fluid velocity
        \param estif_p_v          (o) element matrix for fluid velocity weighting - fluid pressure
        \param estif_q_u          (o) element matrix for pressure weighting - fluid velocity
        \param ppmat              (o) element matrix for fluid pressure
        \param preforce           (o) element rhs vector for fluid pressure
        \param velforce           (o) element rhs vector for fluid velocity
        \param material           (i) fluid material
        \param intpoints          (i) Gaussian integration points
        \param ele                (i) element pointer
       */
      void gauss_point_loop(Teuchos::ParameterList& params,
          const Core::LinAlg::Matrix<nsd_, nen_>& ebofoaf,
          const Core::LinAlg::Matrix<nsd_, nen_>& evelaf,
          const Core::LinAlg::Matrix<nsd_, nen_>& evelnp,
          const Core::LinAlg::Matrix<nsd_, nen_>& eveln,
          const Core::LinAlg::Matrix<nen_, 1>& epreaf, const Core::LinAlg::Matrix<nen_, 1>& eprenp,
          const Core::LinAlg::Matrix<nen_, 1>& epren,
          const Core::LinAlg::Matrix<nen_, 1>& epressnp_timederiv,
          const Core::LinAlg::Matrix<nen_, 1>& epressam_timederiv,
          const Core::LinAlg::Matrix<nen_, 1>& epressn_timederiv,
          const Core::LinAlg::Matrix<nsd_, nen_>& eaccam,
          const Core::LinAlg::Matrix<nsd_, nen_>& edispnp,
          const Core::LinAlg::Matrix<nsd_, nen_>& edispn,
          const Core::LinAlg::Matrix<nsd_, nen_>& egridv,
          const Core::LinAlg::Matrix<nsd_, nen_>& egridvn,
          const Core::LinAlg::Matrix<nen_, 1>& escaaf,
          const Core::LinAlg::Matrix<nsd_, nen_>& emhist,
          const Core::LinAlg::Matrix<nen_, 1>& echist,
          const Core::LinAlg::Matrix<nen_, 1>* eporositynp,
          const Core::LinAlg::Matrix<nen_, 1>* eporositydot,
          const Core::LinAlg::Matrix<nen_, 1>* eporositydotn,
          Core::LinAlg::Matrix<nen_ * nsd_, nen_ * nsd_>& estif_u,
          Core::LinAlg::Matrix<nen_ * nsd_, nen_>& estif_p_v,
          Core::LinAlg::Matrix<nen_, nen_ * nsd_>& estif_q_u,
          Core::LinAlg::Matrix<nen_, nen_>& ppmat, Core::LinAlg::Matrix<nen_, 1>& preforce,
          Core::LinAlg::Matrix<nsd_, nen_>& velforce,
          std::shared_ptr<const Core::Mat::Material> material,
          const Core::FE::GaussIntegration& intpoints,
          Discret::Elements::Fluid* ele);

      /*!
        \brief Gauss point loop for evaluation of off-diagonal terms

        \param params             (i) element parameter list
        \param evelaf             (i) nodal velocities at n+alpha_F/n+1
        \param evelnp             (i) nodal velocities at n+1 (np_genalpha)
        \param epreaf             (i) nodal pressure at n+alpha_F/n+1
        \param eprenp             (i) nodal pressure at n+alpha_F/n+1
        \param epren              (i) nodal pressure at n
        \param epressnp_timederiv (i) nodal pressure time derivative at n+alpha_F/n+1
        \param eaccam             (i) nodal accelerations at n+alpha_M
        \param edispnp            (i) nodal displacements at n+1 (on moving mesh)
        \param edispn             (i) nodal displacements at n
        \param egridv             (i) grid velocity at n+1
        \param egridvn            (i) grid velocity at n
        \param escaaf             (i) nodal scalar at n+alpha_F/n+1
        \param eporositynp        (i) nodal porosity at n+alpha_F/n+1
        \param eforce             (o) coupling rhs force vector
        \param ecoupl_u           (o) coupling element matrix for fluid velocity
        \param ecoupl_p           (o) coupling element matrix for fluid pressure
        \param material           (i) fluid material
        \param intpoints          (i) Gaussian integration points
        \param ele                (i) element pointer
       */
      void gauss_point_loop_od(Teuchos::ParameterList& params,
          const Core::LinAlg::Matrix<nsd_, nen_>& ebofoaf,
          const Core::LinAlg::Matrix<nsd_, nen_>& evelaf,
          const Core::LinAlg::Matrix<nsd_, nen_>& evelnp,
          const Core::LinAlg::Matrix<nsd_, nen_>& eveln,
          const Core::LinAlg::Matrix<nen_, 1>& epreaf, const Core::LinAlg::Matrix<nen_, 1>& eprenp,
          const Core::LinAlg::Matrix<nen_, 1>& epren,
          const Core::LinAlg::Matrix<nen_, 1>& epressnp_timederiv,
          const Core::LinAlg::Matrix<nen_, 1>& epressam_timederiv,
          const Core::LinAlg::Matrix<nen_, 1>& epressn_timederiv,
          const Core::LinAlg::Matrix<nsd_, nen_>& eaccam,
          const Core::LinAlg::Matrix<nsd_, nen_>& edispnp,
          const Core::LinAlg::Matrix<nsd_, nen_>& edispn,
          const Core::LinAlg::Matrix<nsd_, nen_>& egridv,
          const Core::LinAlg::Matrix<nsd_, nen_>& egridvn,
          const Core::LinAlg::Matrix<nen_, 1>& escaaf,
          const Core::LinAlg::Matrix<nsd_, nen_>& emhist,
          const Core::LinAlg::Matrix<nen_, 1>& echist,
          const Core::LinAlg::Matrix<nen_, 1>* eporositynp,
          Core::LinAlg::Matrix<(nsd_ + 1) * nen_, 1>& eforce,
          Core::LinAlg::Matrix<nen_ * nsd_, nen_ * nsd_>& ecoupl_u,
          Core::LinAlg::Matrix<nen_, nen_ * nsd_>& ecoupl_p,
          std::shared_ptr<const Core::Mat::Material> material,
          const Core::FE::GaussIntegration& intpoints,
          Discret::Elements::Fluid* ele);

      /*!
        \brief Evaluation of gauss point values (diagonal terms)

        \param params             (i) element parameter list
        \param evelaf             (i) nodal velocities at n+alpha_F/n+1
        \param evelnp             (i) nodal velocities at n+1 (np_genalpha)
        \param epreaf             (i) nodal pressure at n+alpha_F/n+1
        \param eprenp             (i) nodal pressure at n+alpha_F/n+1
        \param epressnp_timederiv (i) nodal pressure time derivative at n+alpha_F/n+1
        \param edispnp            (i) nodal displacements at n+1 (on moving mesh)
        \param edispn             (i) nodal displacements at n (on moving mesh)
        \param egridv             (i) grid velocity at n+1
        \param egridvn            (i) grid velocity at n
        \param escaaf             (i) nodal scalar at n+alpha_F/n+1
        \param eporositynp        (i) nodal porosity at n+1
        \param eporositydot       (i) nodal porosity time derivative at n+1
        \param eporositydotn      (i) nodal porosity time derivative at n
       */
      void evaluate_variables_at_gauss_point(Teuchos::ParameterList& params,
          const Core::LinAlg::Matrix<nsd_, nen_>& ebofoaf,
          const Core::LinAlg::Matrix<nsd_, nen_>& evelaf,
          const Core::LinAlg::Matrix<nsd_, nen_>& evelnp,
          const Core::LinAlg::Matrix<nsd_, nen_>& eveln,
          const Core::LinAlg::Matrix<nen_, 1>& epreaf, const Core::LinAlg::Matrix<nen_, 1>& eprenp,
          const Core::LinAlg::Matrix<nen_, 1>& epren,
          const Core::LinAlg::Matrix<nen_, 1>& epressnp_timederiv,
          const Core::LinAlg::Matrix<nen_, 1>& epressam_timederiv,
          const Core::LinAlg::Matrix<nen_, 1>& epressn_timederiv,
          const Core::LinAlg::Matrix<nsd_, nen_>& eaccam,
          const Core::LinAlg::Matrix<nsd_, nen_>& edispnp,
          const Core::LinAlg::Matrix<nsd_, nen_>& egridv,
          const Core::LinAlg::Matrix<nsd_, nen_>& egridvn,
          const Core::LinAlg::Matrix<nen_, 1>& escaaf,
          const Core::LinAlg::Matrix<nsd_, nen_>& emhist,
          const Core::LinAlg::Matrix<nen_, 1>& echist,
          const Core::LinAlg::Matrix<nen_, 1>* eporositynp,
          const Core::LinAlg::Matrix<nen_, 1>* eporositydot,
          const Core::LinAlg::Matrix<nen_, 1>* eporositydotn);

      /*!
        \brief Evaluation of gauss point values (off-diagonal terms)

        \param params             (i) element parameter list
        \param evelaf             (i) nodal velocities at n+alpha_F/n+1
        \param evelnp             (i) nodal velocities at n+1 (np_genalpha)
        \param epreaf             (i) nodal pressure at n+alpha_F/n+1
        \param eprenp             (i) nodal pressure at n+alpha_F/n+1
        \param epren              (i) nodal pressure at n
        \param epressnp_timederiv (i) nodal pressure time derivative at n+alpha_F/n+1
        \param edispnp            (i) nodal displacements at n+1 (on moving mesh)
        \param edispn             (i) nodal displacements at n (on moving mesh)
        \param egridv             (i) grid velocity at n+1
        \param egridvn            (i) grid velocity at n
        \param escaaf             (i) nodal scalar at n+alpha_F/n+1
        \param eporositynp        (i) nodal porosity at n+alpha_F/n+1
       */
      void evaluate_variables_at_gauss_point_od(Teuchos::ParameterList& params,
          const Core::LinAlg::Matrix<nsd_, nen_>& ebofoaf,
          const Core::LinAlg::Matrix<nsd_, nen_>& evelaf,
          const Core::LinAlg::Matrix<nsd_, nen_>& evelnp,
          const Core::LinAlg::Matrix<nsd_, nen_>& eveln,
          const Core::LinAlg::Matrix<nen_, 1>& epreaf, const Core::LinAlg::Matrix<nen_, 1>& eprenp,
          const Core::LinAlg::Matrix<nen_, 1>& epren,
          const Core::LinAlg::Matrix<nen_, 1>& epressnp_timederiv,
          const Core::LinAlg::Matrix<nen_, 1>& epressam_timederiv,
          const Core::LinAlg::Matrix<nen_, 1>& epressn_timederiv,
          const Core::LinAlg::Matrix<nsd_, nen_>& eaccam,
          const Core::LinAlg::Matrix<nsd_, nen_>& edispnp,
          const Core::LinAlg::Matrix<nsd_, nen_>& edispn,
          const Core::LinAlg::Matrix<nsd_, nen_>& egridv,
          const Core::LinAlg::Matrix<nsd_, nen_>& egridvn,
          const Core::LinAlg::Matrix<nen_, 1>& escaaf,
          const Core::LinAlg::Matrix<nsd_, nen_>& emhist,
          const Core::LinAlg::Matrix<nen_, 1>& echist,
          const Core::LinAlg::Matrix<nen_, 1>* eporositynp);

      /*!
        \brief Evaluate off diagonal terms in momentum equation

        \param timefacfac      (i) time factor * integration factor
        \param porosity        (i) porosity at gauss point
        \param gridvelint      (i) grid (structure) velocity at gauss point
        \param grad_porosity   (i) porosity gradient at gauss point
        \param dgradphi_dus    (i) derivative of porosity gradient w.r.t. structural displacements
                                   at gauss point
        \param dphi_dus        (i) derivative of porosity w.r.t. structural displacements at gauss
                                   point
        \param refporositydot  (i) time derivative of reference porosity
        \param lin_resM_Dus    (i) linearization of residual of momentum equation w.r.t. structural
                                   dofs
        \param ecoupl_u        (o) coupling element matrix of momentum equation
       */
      void fill_matrix_momentum_od(const double& timefacfac,
          const Core::LinAlg::Matrix<nsd_, nen_>& evelaf,
          const Core::LinAlg::Matrix<nsd_, nen_>& egridv,
          const Core::LinAlg::Matrix<nen_, 1>& epreaf,
          const Core::LinAlg::Matrix<nsd_, nen_ * nsd_>& dgradphi_dus, const double& dphi_dp,
          const double& dphi_dJ, const Core::LinAlg::Matrix<1, nsd_ * nen_>& dphi_dus,
          const double& refporositydot, Core::LinAlg::Matrix<nsd_, nen_ * nsd_>& lin_resM_Dus,
          Core::LinAlg::Matrix<nsd_, nen_ * nsd_>& lin_resM_Dus_gridvel,
          Core::LinAlg::Matrix<nen_ * nsd_, nen_ * nsd_>& ecoupl_u);

      /*!
        \brief Evaluate off diagonal terms in continuity equation

        \param timefacfacpre   (i) time factor (pressure) * integration factor
        \param dphi_dJ         (i) derivative of porosity gradient w.r.t. jacobian determinant at
                                   gauss point
        \param dphi_dJJ        (i) second derivative of porosity gradient w.r.t. jacobian
                                   determinant at gauss point
        \param dphi_dJdp       (i) mixed derivative of porosity gradient w.r.t. jacobian and
                                   pressure at gauss point
        \param dgradphi_dus    (i) derivative of porosity gradient w.r.t. structural displacements
                                   at gauss point
        \param dphi_dus        (i) derivative of porosity w.r.t. structural displacements at gauss
                                   point
        \param dJ_dus          (i) derivative of jacobian w.r.t. structural displacements at gauss
                                   point
        \param egridv          (i) nodal grid velocities
        \param lin_resM_Dus    (i) linearization of residual of momentum equation w.r.t. structural
                                   dofs
        \param ecoupl_u        (o) coupling element matrix of continuity equation
        \param eleGID          (i) element global ID
       */
      virtual void fill_matrix_conti_od(const double& timefacfacpre, const double& dphi_dp,
          const double& dphi_dJ, const double& dphi_dJJ, const double& dphi_dJdp,
          const double& refporositydot, const Core::LinAlg::Matrix<nsd_, nen_ * nsd_>& dgradphi_dus,
          const Core::LinAlg::Matrix<1, nsd_ * nen_>& dphi_dus,
          const Core::LinAlg::Matrix<1, nsd_ * nen_>& dJ_dus,
          const Core::LinAlg::Matrix<nsd_, nen_>& egridv,
          const Core::LinAlg::Matrix<nsd_, nen_ * nsd_>& lin_resM_Dus,
          const Core::LinAlg::Matrix<nsd_, nen_ * nsd_>& lin_resM_Dus_gridvel,
          Core::LinAlg::Matrix<nen_, nen_ * nsd_>& ecoupl_p,
          int eleGID
        );

      //! computation of material derivatives
      double setup_material_derivatives();

      //! access structure material of corresponding solid (poro) element
      void get_struct_material(Discret::Elements::Fluid* ele);

      //! get material parameters of poro fluid element
      void get_material_parameters(std::shared_ptr<const Core::Mat::Material> material);

      //! compute spatial reactive term (darcy term)
      void compute_spatial_reaction_terms(
          std::shared_ptr<const Core::Mat::Material> material,  //< fluid material
          const Core::LinAlg::Matrix<nsd_, nsd_>&
              invdefgrd  //!< inverse of deformationgradient at gausspoint
      );

      //! compute linearization of spatial reactive term (darcy term) w.r.t to structural
      //! displacements
      void compute_lin_spatial_reaction_terms(
          std::shared_ptr<const Core::Mat::Material> material,  //< fluid material
          const Core::LinAlg::Matrix<nsd_, nsd_>&
              defgrd_inv,  //!< inverse of deformationgradient at gausspoint
          const Core::LinAlg::Matrix<1, nsd_ * nen_>*
              dJ_dus,  //!< derivative of jacobian w.r.t. structural displacements at gauss point
          const Core::LinAlg::Matrix<1, nsd_ * nen_>*
              dphi_dus  //!< derivative of porosity w.r.t. structural displacements at gauss point
      );

      //! get compute RHS of momentum equation of time step n and subgrid-scale velocity
      void compute_old_rhs_and_subgrid_scale_velocity();

      //! get stabilization parameters
      void compute_stabilization_parameters(const double& vol);

      //! get compute RHS of contiuity equation of time step n
      void compute_old_rhs_conti(double dphi_dp);

      void compute_mixture_strong_residual(Teuchos::ParameterList& params,
          const Core::LinAlg::Matrix<nsd_, nsd_>& defgrd,
          const Core::LinAlg::Matrix<nsd_, nen_>& edispnp,
          const Core::LinAlg::Matrix<nsd_, nen_>& edispn,
          const Core::LinAlg::Matrix<nsd_ * nsd_, nsd_>& F_X, int gp, bool computeLinOD, int eleGID);

      virtual int compute_volume(Teuchos::ParameterList& params,  //!< parameters
          Discret::Elements::Fluid* ele,                          //!< current fluid element
          Core::FE::Discretization& discretization,               //!< fluid discretization
          std::vector<int>& lm,                     //!< location vector for DOF management
          Core::LinAlg::SerialDenseVector& elevec1  //!< reference to element vector to be filled
      );

      //! Compute deformation gradient
      void compute_def_gradient(
          Core::LinAlg::Matrix<nsd_, nsd_>& defgrd,  //!<<    (i) deformation gradient at gausspoint
          const Core::LinAlg::Matrix<nsd_, nen_>&
              N_XYZ,  //!<<    (i) derivatives of shape functions w.r.t. reference coordinates
          const Core::LinAlg::Matrix<nsd_, nen_>& xcurr  //!<<    (i) current position of gausspoint
      );

      //! Compute Jacobian determinant and the volume change
      void compute_jacobian_determinant_volume_change(double& J, double& volchange,
          const Core::LinAlg::Matrix<nsd_, nsd_>& defgrd,
          const Core::LinAlg::Matrix<nsd_, nen_>& N_XYZ,
          const Core::LinAlg::Matrix<nsd_, nen_>& nodaldisp);

      //! Compute deformation gradient
      double compute_effective_stiffness();

      //! Evaluate element ERROR
      /*!
          general function to compute the error (analytical solution) for particular problem type
       */
      int compute_error(Discret::Elements::Fluid* ele, Teuchos::ParameterList& params,
          std::shared_ptr<Core::Mat::Material>& mat, Core::FE::Discretization& discretization,
          std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec) override;

      int compute_error(Discret::Elements::Fluid* ele, Teuchos::ParameterList& params,
          std::shared_ptr<Core::Mat::Material>& mat, Core::FE::Discretization& discretization,
          std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
          const Core::FE::GaussIntegration& intpoints2) override;

      //! interpolate the anisotropic permeability coefficients at GP from nodal values
      std::vector<double> compute_anisotropic_permeability_coeffs_at_gp() const;

      //! first derivatives of shape functions w.r.t. material coordinates
      Core::LinAlg::Matrix<nsd_, nen_> N_XYZ_;
      /*!
      \brief second derivatives of shape functions w.r.t. material coordinates XYZ
        3D: (N,XX ; N,YY ; N,ZZ ; N,XY ; N,XZ ; N,YX ; N,YZ;  N,ZX ; N,ZY)
        2D: (N,XX ; N,YY ; N,XY ; N,YX )
       */
      Core::LinAlg::Matrix<numderiv2_, nen_> N_XYZ2_;
      //! second derivatives of shape functions w.r.t. material coordinates XYZ (ordered in
      //! symmetric matrix)
      Core::LinAlg::Matrix<nsd_ * nsd_, nen_> N_XYZ2full_;
      //! material coordinates
      Core::LinAlg::Matrix<nsd_, nen_> xyze0_;
      //! node coordinates at time n
      Core::LinAlg::Matrix<nsd_, nen_> xyzeold_;

      //! vector containing all values from previous timelevel n for continuity equation
      double hist_con_;

      //! porosity at gauss point (time step n+1)
      double porosity_;
      //! porosity gradient w.r.t. spatial coordinates at gauss point
      Core::LinAlg::Matrix<nsd_, 1> grad_porosity_;
      //! porosity gradient w.r.t. reference coordinates at gauss point
      Core::LinAlg::Matrix<nsd_, 1> refgrad_porosity_;
      //! grid (=structure) velocity at gauss point (time step n+1)
      Core::LinAlg::Matrix<nsd_, 1> gridvel_int_;
      //! grid (=structure) velocity at gauss point (time step n)
      Core::LinAlg::Matrix<nsd_, 1> gridvel_n_int_;
      //! convective velocity (fluid - structure) at gauss point
      Core::LinAlg::Matrix<nsd_, 1> convvel_;
      //! grid (=structure) velocity derivatives w.r.t. reference coordinates at integration point
      Core::LinAlg::Matrix<nsd_, nsd_> gridvel_deriv_;
      //! grid (=structure) velocity divergence at gauss point
      double gridvel_div_;
      //! determinant of deformation gradient (time step n+1)
      double J_;
      //! fluid pressure at gauss point (time step n+1)
      double press_;
      //! fluid pressure time derivative at gauss point (time step n+1 or n+\alpha_M)
      double press_dot_;
      //! fluid pressure gradient w.r.t to parameter space coordinates at gauss point
      Core::LinAlg::Matrix<nsd_, 1> refgrad_press_;

      //! material reactive tensor
      Core::LinAlg::Matrix<nsd_, nsd_> mat_reac_tensor_;

      //! linearisation of material reactive tensor w.r.t. porosity
      Core::LinAlg::Matrix<nsd_, nsd_> mat_reac_tensor_linporosity_;

      //! linearisation of material reactive tensor w.r.t. J
      Core::LinAlg::Matrix<nsd_, nsd_> mat_reac_tensor_linJ_;

      //! spatial reactive tensor
      Core::LinAlg::Matrix<nsd_, nsd_> reac_tensor_;

      //! linearisation of reactive tensor w.r.t. structural displacements * fluid velocity
      Core::LinAlg::Matrix<nsd_, nsd_ * nen_> reac_tensor_linOD_vel_;

      //! linearisation of reactive tensor w.r.t. structural displacements * grid velocity
      Core::LinAlg::Matrix<nsd_, nsd_ * nen_> reac_tensor_linOD_grid_vel_;

      //! reactive tensor x fluid velocity
      Core::LinAlg::Matrix<nsd_, 1> reac_tensor_vel_;

      //! reactive tensor x structural (grid) velocity
      Core::LinAlg::Matrix<nsd_, 1> reac_tensor_gridvel_;

      //! reactive tensor x convective velocity
      Core::LinAlg::Matrix<nsd_, 1> reac_tensor_convvel_;

      //! linearisation of reaction tensor w.r.t. porosity * fluid velocity
      Core::LinAlg::Matrix<nsd_, 1> lin_p_vel_;

      //! linearisation of reaction tensor w.r.t. porosity * grid velocity
      Core::LinAlg::Matrix<nsd_, 1> lin_p_vel_grid_;

      //! linearization of stabilisation parameters w.r.t. porosity -> it is a (3,1) vector for 2D
      //! and 3D
      Core::LinAlg::Matrix<3, 1> dtau_dphi_;

      //! the stabilisation parameter  for biot stabilization
      double tau_struct_;

      //! residual of mixture (structural) equation
      Core::LinAlg::Matrix<nsd_, 1> mixres_;

      //! linearisation residual of mixture (structural) equation
      Core::LinAlg::Matrix<nsd_, nsd_ * nen_> mixresLinOD_;

      //! corresponding poro structure material
      std::shared_ptr<Mat::StructPoro> struct_mat_;

      //! state if reaction/permeability tensor is constant
      bool const_permeability_;

      //! kinematic type
      Solid::KinemType kintype_;

      //! directions for anisotropic permeability
      std::vector<std::vector<double>> anisotropic_permeability_directions_;

      //! nodal coefficients for anisotropic permeability
      std::vector<std::vector<double>> anisotropic_permeability_nodal_coeffs_;

      //! pointer to parameter lists
      Discret::Elements::FluidEleParameterPoro* porofldpara_;
    };
  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
