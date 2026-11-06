// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FLUID_ELE_CALC_PORO_P1_HPP
#define FOUR_C_FLUID_ELE_CALC_PORO_P1_HPP

#include "4C_config.hpp"

#include "4C_fluid_ele_calc_poro.hpp"

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

     This is the poro P1 implementation, i.e. meant to be coupled with a structure problem
     which solves for the porosity. For the fluid not much changes compared with the
     standard implementation. Only the porosity is evaluated in a different way.
     */
    template <Core::FE::CellType distype>
    class FluidEleCalcPoroP1 : public FluidEleCalcPoro<distype>
    {
      using Base = FluidEleCalcPoro<distype>;

     protected:
      using Base::nen_;
      using Base::nsd_;

      //! private Constructor since we are a Singleton.
      FluidEleCalcPoroP1();

     public:
      //! Singleton access method
      static FluidEleCalcPoroP1<distype>* instance(
          Core::Utils::SingletonAction action = Core::Utils::SingletonAction::create);

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
      int evaluate(Discret::Elements::Fluid* ele, Core::FE::Discretization& discretization,
          const std::vector<int>& lm, Teuchos::ParameterList& params,
          std::shared_ptr<Core::Mat::Material>& mat, Core::LinAlg::SerialDenseMatrix& elemat1,
          Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseVector& elevec2, Core::LinAlg::SerialDenseVector& elevec3,
          const Core::FE::GaussIntegration& intpoints) override;

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
      int evaluate_od(Discret::Elements::Fluid* ele, Core::FE::Discretization& discretization,
          const std::vector<int>& lm, Teuchos::ParameterList& params,
          std::shared_ptr<Core::Mat::Material>& mat, Core::LinAlg::SerialDenseMatrix& elemat1,
          Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseVector& elevec2, Core::LinAlg::SerialDenseVector& elevec3,
          const Core::FE::GaussIntegration& intpoints) override;

     protected:
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
            \param eprenp           (i) nodal pressure at n+alpha_F/n+1
            \param epressnp_timederiv (i) nodal pressure time derivative at n+alpha_F/n+1
            \param eaccam           (i) nodal accelerations at n+alpha_M
            \param edispnp          (i) nodal displacements at n+1 (on moving mesh)
            \param egridv           (i) grid velocity at n+1
            \param escaaf           (i) nodal scalar at n+alpha_F/n+1
            \param material         (i) fluid material
            \param isale            (i) ALE flag
            \param intpoints        (i) Gaussian integration points
       */
      int evaluate_od(Teuchos::ParameterList& params,
          const Core::LinAlg::Matrix<nsd_, nen_>& ebofoaf,
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
          std::shared_ptr<Core::Mat::Material> mat, bool isale,
          const Core::FE::GaussIntegration& intpoints,
          Discret::Elements::Fluid* ele);

      /*!
        \brief calculate off diagonal element matrix and rhs for porous flow

        \param params           (i) element parameter list
        \param evelaf           (i) nodal velocities at n+alpha_F/n+1
        \param evelnp           (i) nodal velocities at n+1 (np_genalpha)
        \param epreaf           (i) nodal pressure at n+alpha_F/n+1
        \param eprenp           (i) nodal pressure at n+alpha_F/n+1
        \param epressnp_timederiv (i) nodal pressure time derivative at n+alpha_F/n+1
        \param eaccam           (i) nodal accelerations at n+alpha_M
        \param edispnp          (i) nodal displacements at n+1 (on moving mesh)
        \param egridv           (i) grid velocity at n+1
        \param escaaf           (i) nodal scalar at n+alpha_F/n+1
        \param ecoupl           (o) element matrix to calculate
        \param eforce           (o) element rhs to calculate
        \param material         (i) fluid material
        \param isale            (i) ALE flag
        \param intpoints        (i) Gaussian integration points
      */
      void sysmat_od(Teuchos::ParameterList& params,
          const Core::LinAlg::Matrix<nsd_, nen_>& ebofoaf,
          const Core::LinAlg::Matrix<nsd_, nen_>& evelaf,
          const Core::LinAlg::Matrix<nsd_, nen_>& evelnp,
          const Core::LinAlg::Matrix<nsd_, nen_>& eveln,
          const Core::LinAlg::Matrix<nen_, 1>& epreaf, const Core::LinAlg::Matrix<nen_, 1>& eprenp,
          const Core::LinAlg::Matrix<nen_, 1>& epren,
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
          Core::LinAlg::Matrix<(nsd_ + 1) * nen_, (nsd_ + 1) * nen_>& ecoupl,
          Core::LinAlg::Matrix<(nsd_ + 1) * nen_, 1>& eforce,
          std::shared_ptr<const Core::Mat::Material> material, bool isale,
          const Core::FE::GaussIntegration& intpoints,
          Discret::Elements::Fluid* ele);

      /*!
        \brief Gauss point loop for evaluation of off-diagonal terms

        \param params           ( i) element parameter list
        \param evelaf             (i) nodal velocities at n+alpha_F/n+1
        \param evelnp             (i) nodal velocities at n+1 (np_genalpha)
        \param epreaf             (i) nodal pressure at n+alpha_F/n+1
        \param eprenp             (i) nodal pressure at n+alpha_F/n+1
        \param epressnp_timederiv (i) nodal pressure time derivative at n+alpha_F/n+1
        \param eaccam             (i) nodal accelerations at n+alpha_M
        \param edispnp            (i) nodal displacements at n+1 (on moving mesh)
        \param egridv             (i) grid velocity at n+1
        \param escaaf             (i) nodal scalar at n+alpha_F/n+1
        \param eporositynp        (i) nodal porosity at n+alpha_F/n+1
        \param eforce             (o) coupling rhs force vector
        \param ecoupl_u           (o) coupling element matrix for fluid velocity
        \param ecoupl_p           (o) coupling element matrix for fluid pressure
        \param ecouplp1_u         (o) coupling element matrix for fluid velocity (porosity dependent
        terms) \param ecouplp1_p         (o) coupling element matrix for fluid pressure (porosity
        dependent terms) \param material           (i) fluid material \param intpoints          (i)
        Gaussian integration points
      */
      virtual void gauss_point_loop_p1_od(Teuchos::ParameterList& params,
          const Core::LinAlg::Matrix<nsd_, nen_>& ebofoaf,
          const Core::LinAlg::Matrix<nsd_, nen_>& evelaf,
          const Core::LinAlg::Matrix<nsd_, nen_>& evelnp,
          const Core::LinAlg::Matrix<nsd_, nen_>& eveln,
          const Core::LinAlg::Matrix<nen_, 1>& epreaf, const Core::LinAlg::Matrix<nen_, 1>& eprenp,
          const Core::LinAlg::Matrix<nen_, 1>& epren,
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
          Core::LinAlg::Matrix<(nsd_ + 1) * nen_, 1>& eforce,
          Core::LinAlg::Matrix<nen_ * nsd_, nen_ * nsd_>& ecoupl_u,
          Core::LinAlg::Matrix<nen_, nen_ * nsd_>& ecoupl_p,
          Core::LinAlg::Matrix<nen_ * nsd_, nen_>& ecouplp1_u,
          Core::LinAlg::Matrix<nen_, nen_>& ecouplp1_p,
          std::shared_ptr<const Core::Mat::Material> material,
          const Core::FE::GaussIntegration& intpoints,
          Discret::Elements::Fluid* ele);

      /*!
       \brief evaluate pressure equation (i.e. continuity equation for standard poro elements)
              This function is overwritten by the poro_p1 element

        \param params         (i) element parameter list
        \param timefacfacpre  (i) fluid pressure at gauss point
        \param rhsfac         (i) jacobian determinant at gauss point
        \param dphi_dp        (i) derivative of porosity gradient w.r.t. fluid pressure
        \param dphi_dJ        (i) derivative of porosity gradient w.r.t. jacobian determinant
        \param dphi_dJdp      (i) mixed derivative of porosity gradient w.r.t. fluid pressure and
       jacobian determinant \param dphi_dJJ       (i) second derivative of porosity gradient w.r.t.
       jacobian determinant \param dphi_dpp       (i) second derivative of porosity gradient w.r.t.
       fluid pressure \param eporositydot   (i) nodal values of porosity time derivative at time
       step n+1 \param eporositydotn  (i) nodal values of porosity time derivative at time step n
        \param echist         (i) nodal values of history values of continuity equation
        \param dgradphi_dp    (i) derivative of porosity gradient w.r.t. fluid pressure
        \param estif_q_u      (o) element matrix (pressure - fluid velocity weighting) to be filled
        \param ppmat          (o) element matrix (pressure - pressure weighting) to be filled
        \param preforce       (o) element rhs vector to be filled
       * */
      void evaluate_pressure_equation(Teuchos::ParameterList& params, const double& timefacfacpre,
          const double& rhsfac, const double& dphi_dp, const double& dphi_dJ,
          const double& dphi_dJdp, const double& dphi_dpp,
          const Core::LinAlg::Matrix<nen_, 1>* eporositydot,
          const Core::LinAlg::Matrix<nen_, 1>* eporositydotn,
          const Core::LinAlg::Matrix<nen_, 1>& echist,
          const Core::LinAlg::Matrix<nsd_, nen_>& dgradphi_dp,
          Core::LinAlg::Matrix<nen_, nen_ * nsd_>& estif_q_u,
          Core::LinAlg::Matrix<nen_, nen_>& ppmat,
          Core::LinAlg::Matrix<nen_, 1>& preforce) override;

      /*!
        \brief Compute porosity and derivatives

        \param params         (i) element parameter list
        \param press          (i) fluid pressure at gauss point
        \param J              (i) jacobian determinant at gauss point
        \param gp             (i) number of actual gauss point
        \param shapfct        (i) shape function values at gauss point
        \param myporosity     (i) nodal porosities
        \param porosity       (o) porosity at gauss point
        \param dphi_dp        (o) derivative of porosity gradient w.r.t. fluid pressure
        \param dphi_dJ        (o) derivative of porosity gradient w.r.t. jacobian determinant
        \param dphi_dJdp      (o) mixed derivative of porosity gradient w.r.t. fluid pressure and
        jacobian determinant \param dphi_dJJ       (o) second derivative of porosity gradient w.r.t.
        jacobian determinant \param dphi_dpp       (o) second derivative of porosity gradient w.r.t.
        fluid pressure \param save           (i) flag for saving porosity within structure material
        \param eleGID         (i) element global id
      */
      void compute_porosity(Teuchos::ParameterList& params, const double& press, const double& J,
          const int& gp, const Core::LinAlg::Matrix<nen_, 1>& shapfct,
          const Core::LinAlg::Matrix<nen_, 1>* myporosity, double& porosity, double* dphi_dp,
          double* dphi_dJ, double* dphi_dJdp, double* dphi_dJJ, double* dphi_dpp,
          bool save, int eleGID) override;

      /*!
        \brief Compute spatial gradient of porosity

        \param dphidp         (i) derivative of porosity w.r.t. fluid pressure
        \param dphidJ         (i) derivative of porosity w.r.t. jacobian determinant
        \param eporositynp    (i) nodal porosities at n+1
        \param gradJ          (i) spatial gradient of jacobian determinant
        \param grad_porosity  (o) spatial gradient of porosity
      */
      void compute_porosity_gradient(const double& dphidp, const double& dphidJ,
          const Core::LinAlg::Matrix<nsd_, 1>& gradJ, const Core::LinAlg::Matrix<nsd_, 1>& gradp,
          const Core::LinAlg::Matrix<nen_, 1>* eporositynp,
          Core::LinAlg::Matrix<nsd_, 1>& grad_porosity,
          Core::LinAlg::Matrix<nsd_, 1>& refgrad_porosity) override;

      /*!
        \brief Compute linearization needed for diagonal terms (lin. of porosity gradient w.r.t.
        fluid pressure)

        \param dphidp         (i) derivative of porosity w.r.t. fluid pressure
        \param dphi_dpp       (i) second derivative of porosity w.r.t. fluid pressure
        \param dphi_dJp       (i) mixed derivative of porosity w.r.t. fluid pressure and jacobian
        determinant \param gradJ          (i) spatial gradient of jacobian determinant \param
        dgradphi_dp    (o) derivate of spatial gradient of porosity w.r.t. fluid pressure
       */
      void compute_linearization(const double& dphi_dp, const double& dphi_dpp,
          const double& dphi_dJp, const Core::LinAlg::Matrix<nsd_, 1>& gradJ,
          Core::LinAlg::Matrix<nsd_, nen_>& dgradphi_dp) override;

      /*!
        \brief Compute linearization needed for off diagonal terms
          (lin. of jacobian, porosity and porosity gradient w.r.t. structure displacement)

        \param dphi_dJ        (i) derivative of porosity w.r.t. jacobian determinant
        \param dphi_dJJ       (i) second derivative of porosity w.r.t. jacobian determinant
        \param dphi_dJp       (i) mixed derivative of porosity w.r.t. fluid pressure and jacobian
        determinant \param defgrd_inv     (i) inverse deformation gradient \param defgrd_IT_vec  (o)
        inverse transposed deformation gradient in vector notation \param F_x            (i)
        derivative of deformation gradient w.r.t. current coordinates xyz \param F_X            (i)
        derivative of deformation gradient w.r.t. material coordinates XYZ \param gradJ          (i)
        spatial gradient of jacobian determinant \param dJ_dus         (o) derivative of jacobian
        determinant w.r.t. structure displacements \param dphi_dus       (o) derivative of porosity
        determinant w.r.t. structure displacements \param dgradphi_dus   (o) derivate of spatial
        gradient of porosity w.r.t. structure displacements
       */
      void compute_linearization_od(const double& dphi_dJ, const double& dphi_dJJ,
          const double& dphi_dJp, const Core::LinAlg::Matrix<nsd_, nsd_>& defgrd_inv,
          const Core::LinAlg::Matrix<nsd_ * nsd_, 1>& defgrd_IT_vec,
          const Core::LinAlg::Matrix<nsd_ * nsd_, nsd_>& F_x,
          const Core::LinAlg::Matrix<nsd_ * nsd_, nsd_>& F_X,
          const Core::LinAlg::Matrix<nsd_, 1>& gradJ, Core::LinAlg::Matrix<1, nsd_ * nen_>& dJ_dus,
          Core::LinAlg::Matrix<1, nsd_ * nen_>& dphi_dus,
          Core::LinAlg::Matrix<nsd_, nen_ * nsd_>& dgradphi_dus) override;

      //! Compute element matrix entries: PSPG
      void pspg(
          Core::LinAlg::Matrix<nen_, nen_ * nsd_>& estif_q_u,  //!< block (weighting function q x u)
          Core::LinAlg::Matrix<nen_, nen_>& ppmat,             //!< block (weighting function q x p)
          Core::LinAlg::Matrix<nen_, 1>& preforce,             //!< rhs forces pressure
          const Core::LinAlg::Matrix<nsd_ * nsd_, nen_>&
              lin_resM_Du,  //!< linearisation of the stabilization residual
          const Core::LinAlg::Matrix<nsd_ * nsd_, nen_>& lin_resMRea_Du,
          const Core::LinAlg::Matrix<nsd_, nen_>&
              lin_resM_Dp,        //!< linearisation of the stabilization residual w.r.t. pressure
          const double& dphi_dp,  //!< linearisation of porosity w.r.t. pressure
          const double& fac3,     //!< factor for residual in current subgrid velocities
          const double& timefacfac,     //!< = timefac x fac
          const double& timefacfacpre,  //!< = timefacpre x fac
          const double& rhsfac          //!< right-hand-side factor for residuals
          ) override;

      //! Compute element matrix entries: reactive stabilization
      void reac_stab(Core::LinAlg::Matrix<nen_ * nsd_, nen_ * nsd_>&
                         estif_u,                              //!< block (weighting function v x u)
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
          ) override;

      int compute_volume(Teuchos::ParameterList& params,
          Discret::Elements::Fluid* ele,             //!< current fluid element
          Core::FE::Discretization& discretization,  //!< fluid discretization
          std::vector<int>& lm,                      //!< location vector for DOF management
          Core::LinAlg::SerialDenseVector& elevec1   //!< reference to element vector to be filled
          ) override;
    };
  }  // namespace Elements
}  // namespace Discret


FOUR_C_NAMESPACE_CLOSE

#endif
