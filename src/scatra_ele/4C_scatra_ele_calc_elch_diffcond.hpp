// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_ELE_CALC_ELCH_DIFFCOND_HPP
#define FOUR_C_SCATRA_ELE_CALC_ELCH_DIFFCOND_HPP

#include "4C_config.hpp"

#include "4C_scatra_ele_calc_elch_electrode.hpp"
#include "4C_scatra_ele_parameter_elch_diffcond.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Elements
  {
    // forward declarations
    class ScaTraEleDiffManagerElchDiffCond;
    template <int nsd, int nen>
    class ScaTraEleInternalVariableManagerElchDiffCond;
    template <Core::FE::CellType distype>
    class ScaTraEleUtilsElchDiffCond;

    // class implementation
    template <Core::FE::CellType distype, int probdim = Core::FE::dim<distype>>
    class ScaTraEleCalcElchDiffCond : public ScaTraEleCalcElchElectrode<distype, probdim>
    {
     protected:
      /// protected constructor, since we are a Singleton, but a derived class exists
      ScaTraEleCalcElchDiffCond(
          const int numdofpernode, const int numscal, const std::string& disname);

      using my = ScaTraEleCalc<distype, probdim>;
      using myelch = ScaTraEleCalcElch<distype, probdim>;
      using myelectrode = ScaTraEleCalcElchElectrode<distype, probdim>;
      using my::nen_;
      using my::nsd_;
      using my::nsd_ele_;

     public:
      /// Singleton access method
      static ScaTraEleCalcElchDiffCond<distype, probdim>* instance(
          const int numdofpernode, const int numscal, const std::string& disname);

     protected:
      /*========================================================================*/
      //! @name general framework
      /*========================================================================*/

      //! calculate contributions to matrix and rhs (inside of loop over all scalars)
      void calc_mat_and_rhs(Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to calculate
          Core::LinAlg::SerialDenseVector& erhs,                    //!< element rhs to calculate
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

      //! calculate contributions to matrix and rhs (outside of loop over all scalars)
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
      //! @name material and related and related functions
      /*========================================================================*/

      //! get material parameters
      void get_material_params(const Core::Elements::Element* ele,  //!< current element
          std::vector<double>& densn,                               //!< density at t_(n)
          std::vector<double>& densnp,  //!< density at t_(n+1) or t_(n+alpha_F)
          std::vector<double>& densam,  //!< density at t_(n+alpha_M)
          double& visc,                 //!< fluid viscosity
          const int iquad = -1          //!< ID of current integration point
          ) override;

      /*========================================================================*/
      //! @name methods for evaluation of individual terms
      /*========================================================================*/

      //! CalcMat: Conduction term with inserted current - ohmic overpotential
      virtual void calc_mat_cond_ohm(
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to be filled
          const int k,                            //!< index of current scalar
          const double timefacfac,  //!< domain-integration factor times time-integration factor
          const double invfval,     //!< 1/(F z_k)
          const Core::LinAlg::Matrix<nsd_, 1>& gradpot  //!< gradient of potential at GP
      );

      //! CalcMat: Conduction term with inserted current - conc. overpotential
      void calc_mat_cond_conc(
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to be filled
          const int k,                            //!< index of current scalar
          const double timefacfac,      //!< domain-integration factor times time-integration factor
          const double rtffcval,        //!< RT/F^2/Newman_const_c/z_k
          const double newman_const_a,  //!< Newman constant a
          const double newman_const_b,  //!< Newman constant b
          const Core::LinAlg::Matrix<nsd_, 1>& gradphi,  //!< gradient of concentration at GP
          const std::vector<double>& conintinv           //!< inverted concentration at GP
      );

      //! CalcMat: Conduction term without inserted current
      virtual void calc_mat_cond(
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to be filled
          const int k,                            //!< index of current scalar
          const double timefacfac,  //!< domain-integration factor times time-integration factor
          const double invfval,     //!< 1/(F z_k)
          const Core::LinAlg::Matrix<nsd_, 1>& curint  //!< current at GP
      );

      //! CalcMat: Additional diffusion term without inserted current
      void calc_mat_cond_diff(
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to be filled
          const int k,                            //!< index of current scalar
          const double timefacfac,  //!< domain-integration factor times time-integration factor
          const double invfval,     //!< 1/(F z_k)
          const std::vector<Core::LinAlg::Matrix<nsd_, 1>>&
              gradphi  //!< gradient of concentration at GP
      );

      //! Potential equation div i inserted current - concentration overpotential
      void calc_mat_pot_equ_divi_conc(
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to be filled
          const int k,                            //!< index of current scalar
          const double timefacfac,      //!< domain-integration factor times time-integration factor
          const double rtffc,           //!< RT/(F^2 Newman_const_c)
          const double rtf,             //!< RT/F
          const double invf,            //!< 1/F
          const double newman_const_a,  //!< Newman constant a
          const double newman_const_b,  //!< Newman constant b
          const Core::LinAlg::Matrix<nsd_, 1>& gradphi,  //!< gradient of concentration at GP
          const double conintinv                         //!< inverted concentration at GP
      );

      //! Potential equation div i without inserted current
      void calc_mat_pot_equ_divi(
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to be filled
          const double timefacfac,  //!< domain-integration factor times time-integration factor
          const double invf         //!< 1/F
      );

      //! CalcMat: Current equation current
      virtual void calc_mat_cur_equ_cur(
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to be filled
          const double timefacfac,  //!< domain-integration factor times time-integration factor
          const double invf         //!< 1/F
      );

      //! CalcMat: Current equation ohmic overpotential
      virtual void calc_mat_cur_equ_ohm(
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to be filled
          const double timefacfac,  //!< domain-integration factor times time-integration factor
          const double invf,        //!< 1/F
          const Core::LinAlg::Matrix<nsd_, 1>& gradpot  //!< gradient of potential at GP
      );

      //! CalcMat: Current equation concentration overpotential
      void calc_mat_cur_equ_conc(
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to be filled
          const double timefacfac,  //!< domain-integration factor times time-integration factor
          const double rtf,         //!< RT/F
          const double rtffc,       //!< RT/(F^2 Newman_const_c)
          const std::vector<double>& invfval,  //!< 1/(F z_k)
          const double newman_const_a,         //!< Newman constant a
          const double newman_const_b,         //!< Newman constant b
          const std::vector<Core::LinAlg::Matrix<nsd_, 1>>&
              gradphi,                          //!< gradient of concentration at GP
          const std::vector<double>& conintinv  //!< inverted concentration at GP
      );

      //! CalcRhs: Conduction term with inserted current - ohmic overpotential
      virtual void calc_rhs_cond_ohm(
          Core::LinAlg::SerialDenseVector& erhs,  //!< element vector to be filled
          const int k,                            //!< index of current scalar
          const double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
          const double invfval,                         //!< 1/(F z_k)
          const Core::LinAlg::Matrix<nsd_, 1>& gradpot  //!< gradient of potential at GP
      );

      //! CalcRhs: Conduction term with inserted current - conc. overpotential
      void calc_rhs_cond_conc(
          Core::LinAlg::SerialDenseVector& erhs,  //!< element vector to be filled
          const int k,                            //!< index of current scalar
          const double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
          const double rtffcval,                         //!< RT/(F^2 Newman_const_c z_k)
          const double newman_const_a,                   //!< Newman constant a
          const double newman_const_b,                   //!< Newman constant b
          const Core::LinAlg::Matrix<nsd_, 1>& gradphi,  //!< gradient of concentration at GP
          const std::vector<double>& conintinv           //!< inverted concentration at GP
      );

      //! CalcRhs: Conduction term without inserted current
      virtual void calc_rhs_cond(
          Core::LinAlg::SerialDenseVector& erhs,  //!< element vector to be filled
          const int k,                            //!< index of current scalar
          const double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
          const double invfval,                        //!< 1/(F z_k)
          const Core::LinAlg::Matrix<nsd_, 1>& curint  //!< current at GP
      );

      //! CalcRhs: Additional diffusion term without inserted current
      void calc_rhs_cond_diff(
          Core::LinAlg::SerialDenseVector& erhs,  //!< element vector to be filled
          const int k,                            //!< index of current scalar
          const double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
          const std::vector<Core::LinAlg::Matrix<nsd_, 1>>&
              gradphi  //!< gradient of concentration at GP
      );

      //! CalcRhs: Potential equation div i inserted current - conc. overpotential
      void calc_rhs_pot_equ_divi_conc(
          Core::LinAlg::SerialDenseVector& erhs,  //!< element vector to be filled
          const int k,                            //!< index of current scalar
          const double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
          const double rtf,     //!< RT/F
          const std::vector<double>& invfval,            //!< 1/(F z_k)
          const double rtffc,                            //!< RT/(F^2 Newman_const_c)
          const double newman_const_a,                   //!< Newman constant a
          const double newman_const_b,                   //!< Newman constant b
          const Core::LinAlg::Matrix<nsd_, 1>& gradphi,  //!< gradient of concentration at GP
          const double conintinv                         //!< inverted concentration at GP
      );

      //! CalcRhs: Potential equation divi without inserted current
      void calc_rhs_pot_equ_divi(
          Core::LinAlg::SerialDenseVector& erhs,  //!< element vector to be filled
          const double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
          const double invf,    //!< 1/F
          const Core::LinAlg::Matrix<nsd_, 1>& curint  //!< current at GP
      );

      //! CalcRhs: Current equation - current
      virtual void calc_rhs_cur_equ_cur(
          Core::LinAlg::SerialDenseVector& erhs,  //!< element vector to be filled
          const double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
          const double invf,    //!< 1/F
          const Core::LinAlg::Matrix<nsd_, 1>& curint  //!< current at GP
      );

      //! CalcRhs: Current equation - ohmic overpotential
      virtual void calc_rhs_cur_equ_ohm(
          Core::LinAlg::SerialDenseVector& erhs,  //!< element vector to be filled
          const double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
          const double invf,    //!< 1/F
          const Core::LinAlg::Matrix<nsd_, 1>& gradpot  //!< gradient of potential at GP
      );

      //! Current equation - concentration overpotential
      void calc_rhs_cur_equ_conc(
          Core::LinAlg::SerialDenseVector& erhs,  //!< element vector to be filled
          const double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
          const double rtf,     //!< RT/F
          const std::vector<double>& invfval,  //!< 1/(F z_k)
          const double rtffc,                  //!< RT/(F^2 Newman_const_c)
          const double newman_const_a,         //!< Newman constant a
          const double newman_const_b,         //!< Newman constant b
          const std::vector<Core::LinAlg::Matrix<nsd_, 1>>&
              gradphi,                          //!< vector of gradient of concentration at GP
          const std::vector<double>& conintinv  //!< inverted concentration at GP
      );


      /*========================================================================*/
      //! @name additional service routines
      /*========================================================================*/

      //! evaluate action
      int evaluate_action(Core::Elements::Element* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, const ScaTra::Action& action,
          Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
          Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
          Core::LinAlg::SerialDenseVector& elevec1_epetra,
          Core::LinAlg::SerialDenseVector& elevec2_epetra,
          Core::LinAlg::SerialDenseVector& elevec3_epetra) override;

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

      //! evaluate electrode kinetics domain condition
      void evaluate_elch_domain_kinetics(
          const Core::Elements::Element* ele,     ///< the actual boundary element
          Core::LinAlg::SerialDenseMatrix& emat,  ///< element-matrix
          Core::LinAlg::SerialDenseVector& erhs,  ///< element-rhs
          const std::vector<Core::LinAlg::Matrix<nen_, 1>>&
              ephinp,  ///< nodal values of concentration and electric potential
          const std::vector<Core::LinAlg::Matrix<nen_, 1>>& ehist,  ///< nodal history vector
          double timefac,                                           ///< time factor
          Core::Conditions::Condition& cond,                        ///< the condition
          const int nume,                 ///< number of transferred electrons
          const std::vector<int> stoich,  ///< stoichiometry of the reaction
          const int kinetics,             ///< desired electrode kinetics model
          const double pot0               ///< actual electrode potential on metal side
      );

      //! validity check for all elements with respect to
      //! formulation-specific parameter, degree's of freedom, number of scalars, ...
      void check_elch_element_parameter(
          Core::Elements::Element* ele  //!< the element we are dealing with
          ) override;

      //! calculate element mass matrix and element residual for initial time derivative
      void calc_initial_time_derivative(Core::Elements::Element* ele,  //!< current element
          Core::LinAlg::SerialDenseMatrix& emat,                       //!< element matrix
          Core::LinAlg::SerialDenseVector& erhs,                       //!< element residual
          Teuchos::ParameterList& params,                              //!< parameter list
          Core::FE::Discretization& discretization,                    //!< discretization
          Core::Elements::LocationArray& la                            //!< location array
          ) override;

      //! Correct RHS calculated from calc_rhs_lin_mass() for the linearized mass term
      void correct_rhs_from_calc_rhs_lin_mass(
          Core::LinAlg::SerialDenseVector& erhs,  //!< element vector to be filled
          const int k,                            //!< index of current scalar
          const double fac,                       //!< domain-integration factor
          const double densnp,                    //!< density at time_(n+1)
          const double phinp                      //!< scalar at time_(n+1)
          ) override;

      //!  calculate weighted mass flux (no reactive flux so far) -> elch-specific implementation
      void calculate_flux(Core::LinAlg::Matrix<nsd_, 1>& q,  //!< flux of species k
          const Inpar::ScaTra::FluxType fluxtype,            //!< type of flux
          const int k                                        //!< index of current scalar
          ) override;

      //!  calculate weighted current flux (no reactive flux so far) -> elch-specific implementation
      void calculate_current(Core::LinAlg::Matrix<nsd_, 1>& q,  //!< flux of species k
          const Inpar::ScaTra::FluxType fluxtype,               //!< type of flux
          const double fac                                      //!< integration factor
          ) override;

      //! calculate error of numerical solution with respect to analytical solution
      void cal_error_compared_to_analyt_solution(
          const Core::Elements::Element* ele,      //!< the element we are dealing with
          Teuchos::ParameterList& params,          //!< parameter list
          Core::LinAlg::SerialDenseVector& errors  //!< vector containing L2-error norm
          ) override;

      void calc_elch_domain_kinetics(Core::Elements::Element* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, std::vector<int>& lm,
          Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
          Core::LinAlg::SerialDenseVector& elevec1_epetra);

      void evaluate_electrode_status(
          const Core::Elements::Element* ele,        ///< the actual boundary element
          Core::LinAlg::SerialDenseVector& scalars,  ///< scalars to be computed
          Teuchos::ParameterList& params,            ///< the parameter list
          Core::Conditions::Condition& cond,         ///< the condition
          const std::vector<Core::LinAlg::Matrix<nen_, 1>>&
              ephinp,  ///< nodal values of concentration and electric potential
          const std::vector<Core::LinAlg::Matrix<nen_, 1>>&
              ephidtnp,                   ///< nodal time derivative vector
          const int kinetics,             ///< desired electrode kinetics model
          const std::vector<int> stoich,  ///< stoichiometry of the reaction
          const int nume,                 ///<  number of transferred electrons
          const double pot0,              ///< actual electrode potential on metal side at t_{n+1}
          const double timefac            ///< factor due to time discretization
      );

      //! set internal variables for diffusion-conduction formulation
      void set_internal_variables_for_mat_and_rhs() override;

      //! get diffusion manager for diffusion-conduction formulation
      std::shared_ptr<ScaTraEleDiffManagerElchDiffCond> diff_manager()
      {
        return std::static_pointer_cast<ScaTraEleDiffManagerElchDiffCond>(my::diffmanager_);
      }

      //! get internal variable manager for diffusion-conduction formulation
      std::shared_ptr<ScaTraEleInternalVariableManagerElchDiffCond<nsd_, nen_>> var_manager()
      {
        return std::static_pointer_cast<ScaTraEleInternalVariableManagerElchDiffCond<nsd_, nen_>>(
            my::scatravarmanager_);
      }

      //! get utility class supporting element evaluation for diffusion-conduction formulation
      ScaTraEleUtilsElchDiffCond<distype>* utils()
      {
        return static_cast<ScaTraEleUtilsElchDiffCond<distype>*>(myelch::utils_);
      }

      void calculate_mean_electrode_concentration(const Core::Elements::Element* const& ele,
          const Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
          Core::LinAlg::SerialDenseVector& conc) override;

      //! flag for used element formulation (material based)
      Inpar::ElCh::DiffCondMat diffcondmat_;

      //! parameter class for diffusion-conduction formulation
      const ScaTraEleParameterElchDiffCond* diffcondparams_;
    };


    /// ScaTraEleDiffManagerElchDiffCond implementation
    /*!
      This class keeps all Diffusion-Conduction-specific transport parameter needed for the
      evaluation of an element.
    */
    class ScaTraEleDiffManagerElchDiffCond : public ScaTraEleDiffManagerElchElectrode
    {
     public:
      using dmelch = ScaTraEleDiffManagerElch;
      using dmelectrode = ScaTraEleDiffManagerElchElectrode;

      ScaTraEleDiffManagerElchDiffCond(int numscal)
          : ScaTraEleDiffManagerElchElectrode(numscal),
            transnum_(numscal, 0.0),
            derivtransnum_(numscal, std::vector<double>(numscal, 0.0)),
            thermfac_(1.0),
            derivthermfac_(numscal, 0.0),
            invval_(numscal, 0.),
            invfval_(numscal, 0.),
            // In the moment, we have only one phase but the framework is flexible
            eps_(1, 1.0),
            tort_(1, 1.0)
      {
      }

      /*========================================================================*/
      //! @name access methods for transport parameter
      /*========================================================================*/

      //! set valence and related parameters for single ionic species
      void set_valence(const double valence, const int k) override
      {
        // call base class routine
        dmelch::set_valence(valence, k);

        const double faraday =
            Discret::Elements::ScaTraEleParameterElch::instance("scatra")->faraday();

        // set additional parameters involving inverse of valence
        invval_[k] = 1. / valence;
        invfval_[k] = 1. / faraday / valence;
      };

      //! Set transference numbers with respect to single ionic species
      virtual void set_trans_num(const double transnum, const int k) { transnum_[k] = transnum; };

      //! Access routine for transference numbers with respect to single ionic species
      double get_trans_num(const int k) { return transnum_[k]; };

      //! Set derivative of transference numbers with respect to concentrations
      virtual void set_deriv_trans_num(const double derivtransnum, const int k, const int iscal)
      {
        (derivtransnum_[k])[iscal] = derivtransnum;
      };

      //! Access routine for derivative of transference numbers with respect to concentrations
      double get_deriv_trans_num(const int k, const int iscal)
      {
        return (derivtransnum_[k])[iscal];
      };

      //! Set thermodynamic factor for a specific electrolyte solution
      void set_therm_fac(const double thermfac) { thermfac_ = thermfac; };

      //! Access routine for the thermodynamic factor for a specific electrolyte solution
      double get_therm_fac() { return thermfac_; };

      //! Set derivative of thermodynamic factor with respect to concentrations
      void set_deriv_therm_fac(const double derivthermfac, const int k)
      {
        derivthermfac_[k] = derivthermfac;
      };

      //! Access routine for derivative of thermodynamic factor with respect to concentrations
      double get_deriv_therm_fac(const int k) { return derivthermfac_[k]; };

      //! Calculate conductivity based on diffusion coefficients
      void calc_conductivity(
          const int numscal, const double ffrt, const std::vector<double>& conint)
      {
        double cond = 0.0;
        for (int ispec = 0; ispec < numscal; ++ispec)
        {
          // conductivity
          cond += ffrt * valence_[ispec] * valence_[ispec] * diff_[ispec] * conint[ispec];
          // derivation of conductivity wrt concentrations
          concderivcond_[ispec] = ffrt * valence_[ispec] * valence_[ispec] * diff_[ispec];
        }

        cond_ = cond;
      };

      //! Calculate transference numbers based on diffusion coefficients
      void calc_trans_num(const int numscal, const std::vector<double>& conint)
      {
        // conductivity without ffrt
        double sum = 0.0;
        for (int k = 0; k < numscal; ++k)
        {
          sum += valence_[k] * valence_[k] * diff_[k] * conint[k];
        }
        double denomin = sum * sum;

        for (int k = 0; k < numscal; ++k)
        {
          transnum_[k] = valence_[k] * valence_[k] * diff_[k] * conint[k] / sum;

          for (int iscal = 0; iscal < numscal; ++iscal)
          {
            if (k == iscal)
            {
              (derivtransnum_[k])[iscal] =
                  (valence_[k] * valence_[k] * diff_[k] *
                      (sum - valence_[k] * valence_[k] * diff_[k] * conint[k])) /
                  denomin;
            }
            else
            {
              (derivtransnum_[k])[iscal] = (-valence_[k] * valence_[k] * diff_[k] * conint[k] *
                                               valence_[iscal] * valence_[iscal] * diff_[iscal]) /
                                           denomin;
            }
          }
        }
      };

      double inv_f_val(const int k) const { return invfval_[k]; };
      const std::vector<double>& inv_f_val() const { return invfval_; };

      /*========================================================================*/
      //! @name access methods for geometrical parameter
      /*========================================================================*/

      //! Set transference numbers with respect to single ionic species
      virtual void set_phase_poro(const double eps, const int phase) { eps_[phase] = eps; }

      double get_phase_poro(const int phase) const { return eps_[phase]; };

      //! Set transference numbers with respect to single ionic species
      virtual void set_phase_tort(const double tort, const int phase) { tort_[phase] = tort; }

      // get geometrical parameter: porosity*tortuosity
      double get_phase_poro_tort(const int phase) const
      {
        double epstort = eps_[phase] * tort_[phase];
        return epstort;
      };

      /*========================================================================*/
      //! @name output
      /*========================================================================*/

      //! Output of transport parameter (to screen)
      void output_transport_params(const int numscal) override
      {
        // call base class routine
        dmelectrode::output_transport_params(numscal);

        // additional outputs
        for (int k = 0; k < numscal; ++k)
          std::cout << "valence " << k << ":   " << valence_[k] << std::endl;

        for (int k = 0; k < numscal; ++k)
          std::cout << "transference number " << k << ":   " << transnum_[k] << std::endl;

        for (int k = 0; k < numscal; ++k)
        {
          for (int iscal = 0; iscal < numscal; ++iscal)
          {
            std::cout << "derivation transference number (" << k << "," << iscal
                      << "):  " << (derivtransnum_[k])[iscal] << std::endl;
          }
        }
        std::cout << std::endl;

        std::cout << "thermodynamic factor:   " << thermfac_ << std::endl;

        for (int k = 0; k < numscal; ++k)
          std::cout << "derivation of thermodynamic factor " << k << ":   " << derivthermfac_[k]
                    << std::endl;
        std::cout << std::endl;

        std::cout << "porosity species" << 0 << ":   " << eps_[0] << std::endl;
        std::cout << std::endl;

        std::cout << "tortuosity species" << 0 << ":   " << tort_[0] << std::endl;
        std::cout << std::endl;
      };

     protected:
      /*========================================================================*/
      //! @name transport parameter
      /*========================================================================*/

      //! transference numbers for single ionic species
      std::vector<double> transnum_;

      //! derivative of transference numbers with respect to concentrations
      std::vector<std::vector<double>> derivtransnum_;

      //! thermodynamic factor for a specific electrolyte solution
      //! transport parameter used only in the Newman model
      double thermfac_;

      //! derivative of thermodynamic factor with respect to concentrations
      std::vector<double> derivthermfac_;

      //! constant parameters 1/(z_k)
      std::vector<double> invval_;

      //! constant parameters 1/(F z_k)
      std::vector<double> invfval_;

      /*========================================================================*/
      //! @name geometrical parameters of the porous medium
      /*========================================================================*/

      //! porosity
      std::vector<double> eps_;
      //! tortuosity
      std::vector<double> tort_;
    };


    /// ScaTraEleInternalVariableManagerElchDiffCond implementation
    /*!
      This class keeps all internal variables needed for the diffusion-conduction formulation.
    */
    template <int nsd, int nen>
    class ScaTraEleInternalVariableManagerElchDiffCond
        : public ScaTraEleInternalVariableManagerElchElectrode<nsd, nen>
    {
     public:
      using vm = ScaTraEleInternalVariableManager<nsd, nen>;
      using vmelch = ScaTraEleInternalVariableManagerElch<nsd, nen>;
      using vmelectrode = ScaTraEleInternalVariableManagerElchElectrode<nsd, nen>;

      ScaTraEleInternalVariableManagerElchDiffCond(int numscal,
          const Discret::Elements::ScaTraEleParameterElch* elchparams,
          const Discret::Elements::ScaTraEleParameterElchDiffCond* diffcondparams)
          : ScaTraEleInternalVariableManagerElchElectrode<nsd, nen>(numscal, elchparams),
            diffcondparams_(diffcondparams),
            rtf_(0.),
            rtffc_(0.),
            curint_(Core::LinAlg::Initialization::zero)
      {
      }

      //! compute and set internal variables only used by the Diffusion-Conduction formulation
      void set_internal_variables_elch_diff_cond(
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
        vmelectrode::set_internal_variables_elch_electrode(
            funct, derxy, ephinp, ephin, econvelnp, ehist);

        // set constant parameter RT/F
        rtf_ = 1.0 / vmelch::frt_;

        // set constant parameter RT/(F^2*Newman_const_C)
        rtffc_ = rtf_ * vmelectrode::invf_ / diffcondparams_->newman_const_c();

        if (diffcondparams_->cur_sol_var())
          for (unsigned idim = 0; idim < nsd; ++idim)
            curint_(idim, 0) = ephinp[vm::numscal_ + 1 + idim].dot(funct);
      };

      /*========================================================================*/
      //! @name return constant internal variables
      /*========================================================================*/

      double rtf() const { return rtf_; };
      double rtffc() const { return rtffc_; };

      /*========================================================================*/
      //! @name return methods for GP values
      /*========================================================================*/

      //! return current density at GP
      const Core::LinAlg::Matrix<nsd, 1>& cur_int() { return curint_; };

     protected:
      //! parameter class for diffusion-conduction formulation
      const Discret::Elements::ScaTraEleParameterElchDiffCond* diffcondparams_;

      /*========================================================================*/
      //! @name constant internal variables
      /*========================================================================*/

      //! constant parameter RT/F
      double rtf_;
      //! constant parameter RT/(F^2*Newman_const_C)
      //! attention: this is a newman specific parameter
      double rtffc_;

      /*========================================================================*/
      //! @name internal variables evaluated at the Gauss point
      /*========================================================================*/

      //! current density at Gauss point
      Core::LinAlg::Matrix<nsd, 1> curint_;
    };  // class ScaTraEleInternalVariableManagerElchDiffCond
  }  // namespace Elements
}  // namespace Discret
FOUR_C_NAMESPACE_CLOSE

#endif
