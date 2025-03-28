// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_ELE_CALC_ELCH_NP_HPP
#define FOUR_C_SCATRA_ELE_CALC_ELCH_NP_HPP

#include "4C_config.hpp"

#include "4C_scatra_ele_calc_elch.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Elements
  {
    // forward declaration
    template <int nsd, int nen>
    class ScaTraEleInternalVariableManagerElchNP;

    template <Core::FE::CellType distype>
    class ScaTraEleCalcElchNP : public ScaTraEleCalcElch<distype>
    {
     public:
      /// Singleton access method
      static ScaTraEleCalcElchNP<distype>* instance(
          const int numdofpernode, const int numscal, const std::string& disname);

     private:
      using my = ScaTraEleCalc<distype>;
      using myelch = ScaTraEleCalcElch<distype>;
      using my::nen_;
      using my::nsd_;

      /// private constructor, since we are a Singleton.
      ScaTraEleCalcElchNP(const int numdofpernode, const int numscal, const std::string& disname);

      /*========================================================================*/
      //! @name general framework
      /*========================================================================*/

      //! calculate contributions to matrix and rhs (inside loop over all scalars)
      void calc_mat_and_rhs(Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to calculate
          Core::LinAlg::SerialDenseVector& erhs,                    //!< element rhs to calculate+
          const int k,                                              //!< index of current scalar
          const double fac,                                         //!< domain-integration factor
          const double timefacfac,  //!< domain-integration factor times time-integration factor
          const double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
          const double taufac,  //!< tau times domain-integration factor
          const double
              timetaufac,  //!< domain-integration factor times tau times time-integration factor
          const double rhstaufac,  //!< time-integration factor for rhs times tau times
                                   //!< domain-integration factor
          Core::LinAlg::Matrix<nen_, 1>&
              tauderpot,  //!< derivatives of stabilization parameter w.r.t. electric potential
          double& rhsint  //!< rhs at Gauss point
          ) override;

      //! calculate contributions to matrix and rhs (outside loop over all scalars)
      void calc_mat_and_rhs_outside_scalar_loop(
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to calculate
          Core::LinAlg::SerialDenseVector& erhs,  //!< element rhs to calculate
          const double fac,                       //!< domain-integration factor
          const double timefacfac,  //!< domain-integration factor times time-integration factor
          const double rhsfac  //!< time-integration factor for rhs times domain-integration factor
          ) override;

      //! Correction for additional flux terms / currents across Dirichlet boundaries
      void correction_for_flux_across_dc(
          Core::FE::Discretization& discretization,  //!< discretization
          const std::vector<int>& lm,                //!< location vector
          Core::LinAlg::SerialDenseMatrix& emat,     //!< element matrix to calculate
          Core::LinAlg::SerialDenseVector& erhs      //!< element rhs to calculate
          ) override;

      /*========================================================================*/
      //! @name material and related functions
      /*========================================================================*/

      //! get the material parameters
      void get_material_params(
          const Core::Elements::Element* ele,  //!< the element we are dealing with
          std::vector<double>& densn,          //!< density at t_(n)
          std::vector<double>& densnp,         //!< density at t_(n+1) or t_(n+alpha_F)
          std::vector<double>& densam,         //!< density at t_(n+alpha_M)
          double& visc,                        //!< fluid viscosity
          const int iquad = -1                 //!< id of current gauss point (default = -1)
          ) override;


      //! evaluate material
      void materials(const std::shared_ptr<const Core::Mat::Material>
                         material,  //!< pointer to current material
          const int k,              //!< index of current scalar
          double& densn,            //!< density at t_(n)
          double& densnp,           //!< density at t_(n+1) or t_(n+alpha_F)
          double& densam,           //!< density at t_(n+alpha_M)
          double& visc,             //!< fluid viscosity
          const int iquad = -1      //!< id of current gauss point (default = -1)
          ) override;

      /*========================================================================*/
      //! @name stabilization and related functions
      /*========================================================================*/

      //! Calculate quantities used for stabilization
      void prepare_stabilization(
          std::vector<double>& tau,  //!< stabilization parameters (one per transported scalar)
          std::vector<Core::LinAlg::Matrix<nen_, 1>>&
              tauderpot,  //!< derivatives of stabilization parameters w.r.t. electric potential
          const std::vector<double>& densnp,  //!< density at t_(n+1) or t_(n+alpha_f)
          const double vol                    //!< element volume
          ) override;

      //! Calculate derivative of tau w.r.t. electric potential according to Taylor, Hughes and
      //! Zarins
      void calc_tau_der_pot_taylor_hughes_zarins(
          Core::LinAlg::Matrix<nen_, 1>&
              tauderpot,  //!< derivatives of stabilization parameter w.r.t. electric potential
          double& tau,    //!< stabilization parameter
          const double densnp,         //!< density at t_(n+1)
          const double frt,            //!< F/(RT)
          const double diffusvalence,  //!< diffusion coefficient times valence
          const Core::LinAlg::Matrix<nsd_, 1>&
              veleff  //!< effective convective velocity (fluid velocity
                      //!< plus migration velocity if applicable)
      );

      /*========================================================================*/
      //! @name methods for evaluation of individual terms
      /*========================================================================*/

      //! calc_res: Residual of Nernst-Planck equation in strong form
      double calc_res(const int k,  //!< index of current scalar
          const double conint,      //!< concentration at GP
          const double hist,        //!< history value at GP
          const double convphi,     //!< convective term (without convective part of migration term)
          const double frt,         //!< F/(RT)
          const Core::LinAlg::Matrix<nen_, 1>&
              migconv,         //!< migration operator: -F/(RT) \grad{\Phi} * \grad{N}
          const double rhsint  //!< rhs of Nernst-Planck equation (not of Newton-Raphson scheme) at
                               //!< Gauss point
      );

      //! CalcMat: SUPG Stabilization of convective term due to fluid flow and migration
      void calc_mat_conv_stab(
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to calculate
          const int k,                            //!< index of current scalar
          const double timefacfac,  //!< domain-integration factor times time-integration factor
          const double taufac,      //!< stabilization parameter tau times domain-integration factor
          const double
              timetaufac,  //!< domain-integration factor times tau times time-integration factor
          Core::LinAlg::Matrix<nen_, 1>&
              tauderpot,     //!< derivatives of stabilization parameter w.r.t. electric potential
          const double frt,  //!< F/(RT)
          const Core::LinAlg::Matrix<nen_, 1>&
              conv,  //!< convection operator: u_x*N,x + u_y*N,y + u_z*N,z
          const Core::LinAlg::Matrix<nen_, 1>&
              migconv,          //!< migration operator: -F/(RT) \grad{\Phi} * \grad{N}
          const double conint,  //!< concentration at GP
          const Core::LinAlg::Matrix<nsd_, 1>& gradphi,  //!< gradient of concentration at GP
          const double residual  //!< residual of Nernst-Planck equation in strong form
      );

      //! CalcMat: Migration term
      void calc_mat_migr(Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to be filled
          const int k,                                           //!< index of current scalar
          const double timefacfac,  //!< domain-integration factor times time-integration factor
          const double frt,         //!< F/RT
          const Core::LinAlg::Matrix<nen_, 1>& migconv,  //!< migration operator
          const double conint                            //!< concentration at GP
      );

      //! CalcMat: Electroneutrality condition in PDE form
      void calc_mat_pot_equ_encpde(
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to be filled
          const int k,                            //!< index of current scalar
          const double timefacfac,  //!< domain-integration factor times time-integration factor
          const double frt,         //!< F/RT
          const Core::LinAlg::Matrix<nen_, 1>& migconv,  //!< migration operator
          const double conint                            //!< concentration at GP
      );

      //! CalcMat: Electroneutrality condition in PDE form with Nernst-Planck equation for species m
      //! eliminated
      void calc_mat_pot_equ_encpde_elim(
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to be filled
          const int k,                            //!< index of current scalar
          const double timefacfac,  //!< domain-integration factor times time-integration factor
          const double frt,         //!< F/RT
          const Core::LinAlg::Matrix<nen_, 1>& migconv,  //!< migration operator
          const double conint                            //!< concentration at GP
      );

      //! CalcMat: Poisson equation for electric potential
      void calc_mat_pot_equ_poisson(
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to be filled
          const int k,                            //!< index of current scalar
          const double fac,                       //!< domain-integration factor
          const double epsilon,                   //!< dielectric constant
          const double faraday                    //!< Faraday constant
      );

      //! CalcMat: Laplace equation for electric potential
      void calc_mat_pot_equ_laplace(
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to be filled
          const double fac                        //!< domain-integration factor
      );

      //! CalcRhs: Additional contributions from conservative formulation of Nernst-Planck equations
      void calc_rhs_conv_add_cons(
          Core::LinAlg::SerialDenseVector& erhs,  //!< element vector to be filled
          const int k,                            //!< index of current scalar
          const double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
          const double conint,  //!< concentration at GP
          const double vdiv     //!< velocity divergence
      );

      //! CalcRhs: SUPG Stabilization of convective term due to fluid flow and migration
      void calc_rhs_conv_stab(
          Core::LinAlg::SerialDenseVector& erhs,  //!< element vector to be filled
          const int k,                            //!< index of current scalar
          const double rhstaufac,  //!< time-integration factor for rhs times tau times
                                   //!< domain-integration factor
          const Core::LinAlg::Matrix<nen_, 1>&
              conv,  //!< convection operator: u_x*N,x + u_y*N,y + u_z*N,z
          const Core::LinAlg::Matrix<nen_, 1>&
              migconv,           //!< migration operator: -F/(RT) \grad{\Phi} * \grad{N}
          const double residual  //!< residual of Nernst-Planck equation in strong form
      );

      //! CalcRhs: Migration term
      void calc_rhs_migr(Core::LinAlg::SerialDenseVector& erhs,  //!< element vector to be filled
          const int k,                                           //!< index of current scalar
          const double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
          const Core::LinAlg::Matrix<nen_, 1>& migconv,  //!< migration operator
          const double conint                            //!< concentration at GP
      );

      //! CalcRhs: Electroneutrality condition in PDE form
      void calc_rhs_pot_equ_encpde(
          Core::LinAlg::SerialDenseVector& erhs,         //!< element vector to be filled
          const int k,                                   //!< index of current scalar
          const double fac,                              //!< domain-integration factor
          const Core::LinAlg::Matrix<nen_, 1>& migconv,  //!< migration operator
          const double conint,                           //!< concentration at GP
          const Core::LinAlg::Matrix<nsd_, 1>& gradphi   //!< gradient of concentration at GP
      );

      //! CalcRhs: Electroneutrality condition in PDE form with Nernst-Planck equation for species m
      //! eliminated
      void calc_rhs_pot_equ_encpde_elim(
          Core::LinAlg::SerialDenseVector& erhs,         //!< element vector to be filled
          const int k,                                   //!< index of current scalar
          const double fac,                              //!< domain-integration factor
          const Core::LinAlg::Matrix<nen_, 1>& migconv,  //!< migration operator
          const double conint,                           //!< concentration at GP
          const Core::LinAlg::Matrix<nsd_, 1>& gradphi   //!< gradient of concentration at GP
      );

      //! CalcRhs: Poisson equation for electric potential
      void calc_rhs_pot_equ_poisson(
          Core::LinAlg::SerialDenseVector& erhs,        //!< element vector to be filled
          const int k,                                  //!< index of current scalar
          const double fac,                             //!< domain-integration factor
          const double epsilon,                         //!< dielectric constant
          const double faraday,                         //!< Faraday constant
          const double conint,                          //!< concentration at GP
          const Core::LinAlg::Matrix<nsd_, 1>& gradpot  //!< gradient of potential at GP
      );

      //! CalcRhs: Laplace equation for electric potential
      void calc_rhs_pot_equ_laplace(
          Core::LinAlg::SerialDenseVector& erhs,        //!< element vector to be filled
          const double fac,                             //!< domain-integration factor
          const Core::LinAlg::Matrix<nsd_, 1>& gradpot  //!< gradient of potential at GP
      );

      /*========================================================================*/
      //! @name additional service routines
      /*========================================================================*/

      //! validity check with respect to input parameters, degrees of freedom, number of scalars
      //! etc.
      void check_elch_element_parameter(Core::Elements::Element* ele  //!< current element
          ) override;

      //! evaluate an electrode boundary kinetics point condition
      void evaluate_elch_boundary_kinetics_point(
          const Core::Elements::Element* ele,     ///< current element
          Core::LinAlg::SerialDenseMatrix& emat,  ///< element matrix
          Core::LinAlg::SerialDenseVector& erhs,  ///< element right-hand side vector
          const std::vector<Core::LinAlg::Matrix<nen_, 1>>&
              ephinp,  ///< state variables at element nodes
          const std::vector<Core::LinAlg::Matrix<nen_, 1>>&
              ehist,       ///< history variables at element nodes
          double timefac,  ///< time factor
          std::shared_ptr<Core::Conditions::Condition>
              cond,                       ///< electrode kinetics boundary condition
          const int nume,                 ///< number of transferred electrons
          const std::vector<int> stoich,  ///< stoichiometry of the reaction
          const int kinetics,             ///< desired electrode kinetics model
          const double pot0,              ///< electrode potential on metal side
          const double frt,               ///< factor F/RT
          const double
              scalar  ///< scaling factor for element matrix and right-hand side contributions
          ) override;

      // Get conductivity from material
      void get_conductivity(const enum Inpar::ElCh::EquPot
                                equpot,  //!< type of closing equation for electric potential
          double& sigma_all,             //!< conductivity of electrolyte solution
          std::vector<double>&
              sigma,    //!< conductivity of a single ion + overall electrolyte solution
          bool effCond  //!< flag if effective conductivity should be calculated
          ) override;

      //!  calculate weighted mass flux (no reactive flux so far) -> elch-specific implementation
      void calculate_flux(Core::LinAlg::Matrix<nsd_, 1>& q,  //!< flux of species k
          const Inpar::ScaTra::FluxType fluxtype,            //!< type of flux
          const int k                                        //!< index of current scalar
          ) override;

      //! calculate error of numerical solution with respect to analytical solution
      void cal_error_compared_to_analyt_solution(
          const Core::Elements::Element* ele,      //!< the element we are dealing with
          Teuchos::ParameterList& params,          //!< parameter list
          Core::LinAlg::SerialDenseVector& errors  //!< vector containing L2-error norm
          ) override;

      //! set internal variables for Nernst-Planck formulation
      void set_internal_variables_for_mat_and_rhs() override;

      //! get internal variable manager for Nernst-Planck formulation
      std::shared_ptr<ScaTraEleInternalVariableManagerElchNP<nsd_, nen_>> var_manager()
      {
        return std::static_pointer_cast<ScaTraEleInternalVariableManagerElchNP<nsd_, nen_>>(
            my::scatravarmanager_);
      };

      /*========================================================================*/
      //! @name flags and enums
      /*========================================================================*/

      //! flag for migration operator in the stabilization terms
      /* The migration term in the ion transport equation can be split into a convective and a
         reactive part. Combining the convective part of the migration term with the convective term
         in the ion transport equation yields an "extended" convective term including not only the
         fluid velocity, but also the migration velocity. In case migrationstab_ == true, SUPG
         stabilization is performed for the extended convective term. In case migrationstab_ ==
         false, SUPG stabilization is performed only for the original convective term.
       */
      bool migrationstab_;

      /*========================================================================*/
      //! @name scalar degrees of freedom and related
      /*========================================================================*/
    };


    /// ScaTraEleInternalVariableManagerElchNP implementation
    /*!
      This class keeps all internal variables for the Nernst-Planck formulation.
    */
    template <int nsd, int nen>
    class ScaTraEleInternalVariableManagerElchNP
        : public ScaTraEleInternalVariableManagerElch<nsd, nen>
    {
     public:
      using vm = ScaTraEleInternalVariableManager<nsd, nen>;
      using vmelch = ScaTraEleInternalVariableManagerElch<nsd, nen>;

      ScaTraEleInternalVariableManagerElchNP(
          int numscal, const Discret::Elements::ScaTraEleParameterElch* elchpara)
          : ScaTraEleInternalVariableManagerElch<nsd, nen>(numscal, elchpara),
            // constant internal variables
            // empty

            //! internal variables evaluated at the Gauss point
            migvelint_(Core::LinAlg::Initialization::zero),
            migconv_(Core::LinAlg::Initialization::zero) {};

      //! compute and set internal variables for the Nernst-Planck formulation
      void set_internal_variables_elch_np(
          const Core::LinAlg::Matrix<nen, 1>& funct,  //!< array for shape functions
          const Core::LinAlg::Matrix<nsd, nen>&
              derxy,  //!< global derivatives of shape functions w.r.t x,y,z
          const std::vector<Core::LinAlg::Matrix<nen, 1>>&
              ephinp,  //!< nodal state variables at t_(n+1) or t_(n+alpha_F)
          const std::vector<Core::LinAlg::Matrix<nen, 1>>&
              ephin,  //!< nodal state variables at t_(n)
          const Core::LinAlg::Matrix<nsd, nen>&
              econvelnp,  //!< nodal convective velocity values at t_(n+1) or t_(n+alpha_F)
          const std::vector<Core::LinAlg::Matrix<nen, 1>>&
              ehist  //!< history vector of transported scalars
      )
      {
        // set internal variables in base variable manager
        vmelch::set_internal_variables_elch(funct, derxy, ephinp, ephin, econvelnp, ehist);

        // migration velocity vector (divided by D_k*z_k) at t_(n+1) or t_(n+alpha_F): -F/(RT)
        // \grad{\Phi}
        migvelint_.multiply(-vmelch::frt_, derxy, ephinp[vm::numscal_]);

        // convective part of migration term (divided by D_k*z_k): -F/(RT) \grad{\Phi}*\grad{N}
        migconv_.multiply_tn(-vmelch::frt_, derxy, vmelch::gradpot_);
      };

      /*========================================================================*/
      //! @name return constant internal variables
      /*========================================================================*/

      // empty

      /*========================================================================*/
      //! @name return methods for internal variables
      /*========================================================================*/

      //! return migration velocity vector (divided by D_k*z_k): -F/(RT) \grad{\Phi}
      const Core::LinAlg::Matrix<nsd, 1>& mig_vel_int() const { return migvelint_; };

      //! return convective part of migration term (divided by D_k*z_k): -F/(RT) \grad{\Phi} *
      //! \grad{N}
      Core::LinAlg::Matrix<nen, 1> mig_conv() { return migconv_; };

     private:
      /*========================================================================*/
      //! @name constant internal variables
      /*========================================================================*/


      /*========================================================================*/
      //! @name internal variables evaluated at element center or Gauss point
      /*========================================================================*/

      //! migration velocity vector (divided by D_k*z_k): -F/(RT) \grad{\Phi}
      Core::LinAlg::Matrix<nsd, 1> migvelint_;

      //! convective part of migration term (divided by D_k*z_k): -F/(RT) \grad{\Phi} * \grad{N}
      Core::LinAlg::Matrix<nen, 1> migconv_;

    };  // class ScaTraEleInternalVariableManagerElchNP

  }  // namespace Elements

}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
