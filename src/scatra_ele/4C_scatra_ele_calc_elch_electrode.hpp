// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_ELE_CALC_ELCH_ELECTRODE_HPP
#define FOUR_C_SCATRA_ELE_CALC_ELCH_ELECTRODE_HPP

#include "4C_config.hpp"

#include "4C_scatra_ele_calc_elch.hpp"
#include "4C_scatra_ele_utils_elch_electrode.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Elements
  {
    // forward declarations
    class ScaTraEleDiffManagerElchElectrode;
    template <int nsd, int nen>
    class ScaTraEleInternalVariableManagerElchElectrode;
    template <Core::FE::CellType distype>
    class ScaTraEleUtilsElchElectrode;

    // class implementation
    template <Core::FE::CellType distype, int probdim = Core::FE::dim<distype>>
    class ScaTraEleCalcElchElectrode : public ScaTraEleCalcElch<distype, probdim>
    {
     public:
      //! singleton access method
      static ScaTraEleCalcElchElectrode<distype, probdim>* instance(
          const int numdofpernode, const int numscal, const std::string& disname);



      //! evaluate action
      int evaluate_action(Core::Elements::Element* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, const ScaTra::Action& action,
          Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1,
          Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseVector& elevec2,
          Core::LinAlg::SerialDenseVector& elevec3) override;

     protected:
      //! protected constructor for singletons
      ScaTraEleCalcElchElectrode(
          const int numdofpernode, const int numscal, const std::string& disname);

      //! abbreviations
      using my = ScaTraEleCalc<distype, probdim>;
      using myelch = ScaTraEleCalcElch<distype, probdim>;
      using my::nen_;
      using my::nsd_;
      using my::nsd_ele_;

      /*========================================================================*/
      //! @name general framework
      /*========================================================================*/

      //! calculate contributions to element matrix and residual (inside loop over all scalars)
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

      //! calculate contributions to element matrix and residual (outside loop over all scalars)
      void calc_mat_and_rhs_outside_scalar_loop(
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to calculate
          Core::LinAlg::SerialDenseVector& erhs,  //!< element rhs to calculate
          const double fac,                       //!< domain-integration factor
          const double timefacfac,  //!< domain-integration factor times time-integration factor
          const double rhsfac  //!< time-integration factor for rhs times domain-integration factor
          ) override;

      //! compute additional flux terms across Dirichlet boundaries
      void correction_for_flux_across_dc(
          Core::FE::Discretization& discretization,  //!< discretization
          const std::vector<int>& lm,                //!< location vector
          Core::LinAlg::SerialDenseMatrix& emat,     //!< element matrix to calculate
          Core::LinAlg::SerialDenseVector& erhs      //!< element rhs to calculate
          ) override {};

      /*========================================================================*/
      //! @name material and related and related functions
      /*========================================================================*/

      //! get material parameters
      void get_material_params(
          const Core::Elements::Element* ele,  //!< the element we are dealing with
          std::vector<double>& densn,          //!< density at t_(n)
          std::vector<double>& densnp,         //!< density at t_(n+1) or t_(n+alpha_F)
          std::vector<double>& densam,         //!< density at t_(n+alpha_M)
          double& visc,                        //!< fluid viscosity
          const int iquad = -1                 //!< id of current gauss point (default = -1)
          ) override;

      /*========================================================================*/
      //! @name methods for evaluation of individual terms
      /*========================================================================*/

      //! CalcMat: linearizations of diffusion term and Ohmic overpotential w.r.t. structural
      //! displacements
      void calc_diff_od_mesh(Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix
          const int k,                                               //!< index of current scalar
          const int ndofpernodemesh,  //!< number of structural degrees of freedom per node
          const double diffcoeff,     //!< diffusion coefficient
          const double fac,           //!< domain-integration factor
          const double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
          const double J,       //!< Jacobian determinant det(dx/ds)
          const Core::LinAlg::Matrix<nsd_, 1>& gradphi,    //!< gradient of current scalar
          const Core::LinAlg::Matrix<nsd_, 1>& convelint,  //!< convective velocity
          const Core::LinAlg::Matrix<1, nsd_ * nen_>&
              dJ_dmesh  //!< derivatives of Jacobian determinant det(dx/ds) w.r.t. structural
                        //!< displacements
          ) override;

      //! CalcMat: linearization of diffusion coefficient in diffusion term
      void calc_mat_diff_coeff_lin(
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to be filled
          const int k,                            //!< index of current scalar
          const double timefacfac,  //!< domain-integration factor times time-integration factor
          const Core::LinAlg::Matrix<nsd_, 1>& gradphi,  //!< gradient of concentration at GP
          const double scalar  //!< scaling factor for element matrix contributions
      );

      //! CalcMat: potential equation div i with inserted current - ohmic overpotential
      void calc_mat_pot_equ_divi_ohm(
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to be filled
          const double timefacfac,  //!< domain-integration factor times time-integration factor
          const double invf,        //!< 1/F
          const Core::LinAlg::Matrix<nsd_, 1>& gradpot,  //!< gradient of potential at GP
          const double scalar  //!< scaling factor for element matrix contributions
      );

      /*!
       * @brief calculate the conservative part of the convective term of the right-hand side
       * vector, due to e.g. deformation of the body
       *
       * @param[in,out] erhs  element rhs vector to be filled
       * @param[in] k         index of current scalar
       * @param[in] rhsfac    time-integration factor for rhs times domain-integration factor
       * @param[in] vdiv      divergence of velocity
       */
      void calc_rhs_conservative_part_of_convective_term(Core::LinAlg::SerialDenseVector& erhs,
          const int k, const double rhsfac, const double vdiv);

      //! CalcRhs: potential equation div i with inserted current - ohmic overpotential
      virtual void calc_rhs_pot_equ_divi_ohm(
          Core::LinAlg::SerialDenseVector& erhs,  //!< element vector to be filled
          const double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
          const double invf,    //!< 1./F
          const Core::LinAlg::Matrix<nsd_, 1>& gradpot,  //!< gradient of potential at GP
          const double scalar  //!< scaling factor for element residual contributions
      );

      //! calculate weighted current density
      void calculate_current(Core::LinAlg::Matrix<nsd_, 1>& q,  //!< flux of species k
          const Inpar::ScaTra::FluxType fluxtype,               //!< type of flux
          const double fac                                      //!< integration factor
          ) override;

      //! get utility class supporting element evaluation for electrodes
      Discret::Elements::ScaTraEleUtilsElchElectrode<distype>* utils()
      {
        return static_cast<Discret::Elements::ScaTraEleUtilsElchElectrode<distype>*>(
            myelch::utils_);
      };

      /*========================================================================*/
      //! @name additional service routines
      /*========================================================================*/

      //! validity check with respect to input parameters, degrees of freedom, number of scalars
      //! etc.
      void check_elch_element_parameter(Core::Elements::Element* ele  //!< current element
          ) override;

      //! get conductivity
      void get_conductivity(
          const enum ElCh::EquPot equpot,  //!< type of closing equation for electric potential
          double& sigma_all,               //!< conductivity of electrolyte solution
          std::vector<double>&
              sigma,    //!< conductivity or a single ion + overall electrolyte solution
          bool effCond  //!< flag if effective conductivity should be calculated
          ) override;

      //! calculate electrode state of charge and C rate
      virtual void calculate_electrode_soc_and_c_rate(
          const Core::Elements::Element* const& ele,       //!< the element we are dealing with
          const Core::FE::Discretization& discretization,  //!< discretization
          Core::Elements::LocationArray& la,               //!< location array
          Core::LinAlg::SerialDenseVector&
              scalars  //!< result vector for scalar integrals to be computed
      );

      //! calculate mean concentration of micro discretization
      //! \param ele              the element we are dealing with
      //! \param discretization   discretization
      //! \param la               location array
      //! \param conc             result vector for scalar integrals to be computed
      virtual void calculate_mean_electrode_concentration(const Core::Elements::Element* const& ele,
          const Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
          Core::LinAlg::SerialDenseVector& conc);

      //! calculate weighted mass flux (no reactive flux so far)
      void calculate_flux(Core::LinAlg::Matrix<nsd_, 1>& q,  //!< flux of species k
          const Inpar::ScaTra::FluxType fluxtype,            //!< type of flux
          const int k                                        //!< index of current scalar
          ) override;

      //! calculate error of numerical solution with respect to analytical solution
      void cal_error_compared_to_analyt_solution(const Core::Elements::Element* ele,  //!< element
          Teuchos::ParameterList& params,          //!< parameter list
          Core::LinAlg::SerialDenseVector& errors  //!< vector containing L2 and H1 error norms
          ) override;

      //! set internal variables for electrodes
      void set_internal_variables_for_mat_and_rhs() override;

      //! get diffusion manager for electrodes
      std::shared_ptr<ScaTraEleDiffManagerElchElectrode> diff_manager()
      {
        return std::static_pointer_cast<ScaTraEleDiffManagerElchElectrode>(my::diffmanager_);
      };

     private:
      //! get internal variable manager for electrodes
      std::shared_ptr<ScaTraEleInternalVariableManagerElchElectrode<nsd_, nen_>> var_manager()
      {
        return std::static_pointer_cast<ScaTraEleInternalVariableManagerElchElectrode<nsd_, nen_>>(
            my::scatravarmanager_);
      };
    };


    //! ScaTraEleDiffManagerElchElectrode implementation
    /*!
      This class is derived from the standard ScaTraEleDiffManager and keeps all electrode-specific
      transport parameters.
    */
    class ScaTraEleDiffManagerElchElectrode : public ScaTraEleDiffManagerElch
    {
     public:
      ScaTraEleDiffManagerElchElectrode(int numscal)
          : ScaTraEleDiffManagerElch(numscal),
            concderivdiff_(numscal, std::vector<double>(numscal, 0.0)),
            tempderivdiff_(numscal, std::vector<double>(numscal, 0.0)),
            cond_(0.0),
            concderivcond_(numscal, 0.0),
            tempderivcond_(numscal, 0.0) {};

      /*========================================================================*/
      //! @name access methods
      /*========================================================================*/

      //! Set derivative of diffusion coefficients with respect to concentrations
      void set_conc_deriv_iso_diff_coef(const double concderivdiff, const int k, const int iscal)
      {
        (concderivdiff_[k])[iscal] = concderivdiff;
      };

      //! Set derivative of diffusion coefficient with respect to concentrations
      void set_temp_deriv_iso_diff_coef(const double tempderivdiff, const int k, const int iscal)
      {
        (tempderivdiff_[k])[iscal] = tempderivdiff;
      };

      //! Access routine for derivative of diffusion coefficients with respect to concentrations
      double get_conc_deriv_iso_diff_coef(const int k, const int iscal)
      {
        return (concderivdiff_[k])[iscal];
      };

      //! Access routine for derivative of diffusion coefficients with respect to concentrations
      double get_temp_deriv_iso_diff_coef(const int k, const int iscal)
      {
        return (tempderivdiff_[k])[iscal];
      };

      //! Set conductivity of the electrolyte solution and electrode
      void set_cond(const double cond) { cond_ = cond; };

      //! Access routine for conductivity of the electrolyte solution and electrode
      double get_cond() { return cond_; };

      //! Set derivative of the conductivity with respect to concentrations
      void set_conc_deriv_cond(const double concderivcond, const int k)
      {
        concderivcond_[k] = concderivcond;
      };

      //! Access routine for derivative of the conductivity with respect to concentrations
      double get_conc_deriv_cond(const int k) { return concderivcond_[k]; };

      //! Set derivative of the conductivity with respect to temperature
      void set_temp_deriv_cond(const double tempderivcond, const int k)
      {
        tempderivcond_[k] = tempderivcond;
      };

      //! Access routine for derivative of the conductivity with respect to temperature
      double get_temp_deriv_cond(const int k) { return tempderivcond_[k]; };

      /*========================================================================*/
      //! @name output
      /*========================================================================*/

      //! print transport parameters to screen
      virtual void output_transport_params(const int numscal)
      {
        for (int k = 0; k < numscal; ++k)
          std::cout << "diffusion coefficient " << k << ":   " << diff_[k] << std::endl;

        for (int k = 0; k < numscal; ++k)
        {
          for (int iscal = 0; iscal < numscal; ++iscal)
          {
            std::cout << "derivative of diffusion coefficient (" << k << "," << iscal
                      << "):  " << (concderivdiff_[k])[iscal] << std::endl;
            std::cout << "derivative of diffusion coefficient w.r.t. temperature (" << k << ","
                      << iscal << "):  " << (tempderivdiff_[k])[iscal] << std::endl;
          }
        }
        std::cout << std::endl;

        std::cout << "conductivity:   " << cond_ << std::endl;

        for (int k = 0; k < numscal; ++k)
        {
          std::cout << "derivative of conductivity " << k
                    << " w.r.t. concentration:   " << concderivcond_[k] << std::endl;
          std::cout << "derivative of conductivity " << k
                    << " w.r.t. temperature:   " << tempderivcond_[k] << std::endl;
        }
        std::cout << std::endl;
      };

     protected:
      /*========================================================================*/
      //! @name transport parameter
      /*========================================================================*/

      //! derivative of diffusion coefficients with respect to concentrations
      std::vector<std::vector<double>> concderivdiff_;

      //! derivative of diffusion coefficients with respect to temperature
      std::vector<std::vector<double>> tempderivdiff_;

      //! conductivity of the electrolyte solution
      double cond_;

      //! derivative of the conductivity with respect to concentrations
      std::vector<double> concderivcond_;

      //! derivative of the conductivity with respect to temperature
      std::vector<double> tempderivcond_;
    };

    //! ScaTraEleInternalVariableManagerElchElectrode implementation
    /*!
      This class keeps all internal electrode-specific variables.
    */
    template <int nsd, int nen>
    class ScaTraEleInternalVariableManagerElchElectrode
        : public ScaTraEleInternalVariableManagerElch<nsd, nen>
    {
     public:
      using vm = ScaTraEleInternalVariableManager<nsd, nen>;
      using vmelch = ScaTraEleInternalVariableManagerElch<nsd, nen>;

      //! constructor
      ScaTraEleInternalVariableManagerElchElectrode(
          int numscal, const Discret::Elements::ScaTraEleParameterElch* elchpara)
          : ScaTraEleInternalVariableManagerElch<nsd, nen>(numscal, elchpara),
            invf_(1. / vmelch::parameters_->faraday())
      {
      }


      //! compute and set internal electrode-specific variables
      void set_internal_variables_elch_electrode(
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
      };

      //! compute and set internal electrode-specific variables for evaluation of SOC and Crate
      void set_internal_variables_elch_electrode_soc_and_c_rate(
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
        // FRT not needed for calculation of SOC and C Rate. Temperature is not extracted in case of
        // non constant temperatures,
        const bool do_setfrt = false;
        // set internal variables in base variable manager
        vmelch::set_internal_variables_elch(
            funct, derxy, ephinp, ephin, econvelnp, ehist, do_setfrt);
      };

      //! return constant parameter 1./F
      double inv_f() { return invf_; };

     protected:
      //! constant parameter 1./F
      const double invf_;
    };  // class ScaTraEleInternalVariableManagerElchElectrode
  }  // namespace Elements
}  // namespace Discret
FOUR_C_NAMESPACE_CLOSE

#endif
