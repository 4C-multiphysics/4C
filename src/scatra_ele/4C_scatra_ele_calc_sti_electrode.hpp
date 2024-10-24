// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_ELE_CALC_STI_ELECTRODE_HPP
#define FOUR_C_SCATRA_ELE_CALC_STI_ELECTRODE_HPP

#include "4C_config.hpp"

#include "4C_inpar_elch.hpp"
#include "4C_mat_electrode.hpp"
#include "4C_scatra_ele_calc.hpp"
#include "4C_scatra_ele_calc_elch_electrode.hpp"
#include "4C_scatra_ele_sti_elch.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Elements
  {
    // forward declarations
    class ScaTraEleDiffManagerElchElectrode;
    class ScaTraEleDiffManagerSTIElchElectrode;
    class ScaTraEleDiffManagerSTIThermo;
    template <Core::FE::CellType distype>
    class ScaTraEleUtilsElchElectrode;

    // class implementation
    template <Core::FE::CellType distype>
    class ScaTraEleCalcSTIElectrode : public ScaTraEleCalc<distype>,
                                      public ScaTraEleSTIElch<distype>
    {
     public:
      //! singleton access method
      static ScaTraEleCalcSTIElectrode<distype>* instance(
          const int numdofpernode, const int numscal, const std::string& disname);



     private:
      //! abbreviations
      using my = ScaTraEleCalc<distype>;
      using mystielch = ScaTraEleSTIElch<distype>;
      using my::nen_;
      using my::nsd_;
      using my::nsd_ele_;

      //! private constructor for singletons
      ScaTraEleCalcSTIElectrode(
          const int numdofpernode, const int numscal, const std::string& disname);

      //! evaluate action for off-diagonal system matrix block
      int evaluate_action_od(Core::Elements::Element* ele,  //!< current element
          Teuchos::ParameterList& params,                   //!< parameter list
          Core::FE::Discretization& discretization,         //!< discretization
          const ScaTra::Action& action,                     //!< action parameter
          Core::Elements::LocationArray& la,                //!< location array
          Core::LinAlg::SerialDenseMatrix& elemat1_epetra,  //!< element matrix 1
          Core::LinAlg::SerialDenseMatrix& elemat2_epetra,  //!< element matrix 2
          Core::LinAlg::SerialDenseVector& elevec1_epetra,  //!< element right-hand side vector 1
          Core::LinAlg::SerialDenseVector& elevec2_epetra,  //!< element right-hand side vector 2
          Core::LinAlg::SerialDenseVector& elevec3_epetra   //!< element right-hand side vector 3
          ) override;

      //! calculate element matrix and element right-hand side vector
      void sysmat(Core::Elements::Element* ele,       ///< current element
          Core::LinAlg::SerialDenseMatrix& emat,      ///< element matrix
          Core::LinAlg::SerialDenseVector& erhs,      ///< element right-hand side vector
          Core::LinAlg::SerialDenseVector& subgrdiff  ///< subgrid diffusivity scaling vector
          ) override;

      //! fill element matrix with linearizations of discrete thermo residuals w.r.t. scatra dofs
      void sysmat_od_thermo_scatra(Core::Elements::Element* ele,  //!< current element
          Core::LinAlg::SerialDenseMatrix& emat                   //!< element matrix
      );

      //! element matrix and right-hand side vector contributions arising from Joule's heat
      void calc_mat_and_rhs_joule(Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix
          Core::LinAlg::SerialDenseVector& erhs,  //!< element right-hand side vector
          const double& timefacfac,  //!< domain integration factor times time integration factor
          const double& rhsfac  //!< domain integration factor times time integration factor for
                                //!< right-hand side vector
          ) override;

      //! element matrix and right-hand side vector contributions arising from heat of mixing
      void calc_mat_and_rhs_mixing(Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix
          Core::LinAlg::SerialDenseVector& erhs,  //!< element right-hand side vector
          const double& timefacfac,  //!< domain integration factor times time integration factor
          const double& rhsfac  //!< domain integration factor times time integration factor for
                                //!< right-hand side vector
          ) override;

      //! element matrix and right-hand side vector contributions arising from Soret effect
      void calc_mat_and_rhs_soret(Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix
          Core::LinAlg::SerialDenseVector& erhs,  //!< element right-hand side vector
          const double& timefacfac,  //!< domain integration factor times time integration factor
          const double& rhsfac  //!< domain integration factor times time integration factor for
                                //!< right-hand side vector
          ) override;

      //! provide element matrix with linearizations of Joule's heat term in discrete thermo
      //! residuals w.r.t. scatra dofs
      void calc_mat_joule_od(Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix
          const double& timefacfac  //!< domain integration factor times time integration factor
          ) override;

      //! provide element matrix with linearizations of heat of mixing term in discrete thermo
      //! residuals w.r.t. scatra dofs
      void calc_mat_mixing_od(Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix
          const double& timefacfac  //!< domain integration factor times time integration factor
          ) override;

      //! provide element matrix with linearizations of Soret effect term in discrete thermo
      //! residuals w.r.t. scatra dofs
      void calc_mat_soret_od(Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix
          const double& timefacfac  //!< domain integration factor times time integration factor
          ) override;

      //! extract quantities for element evaluation
      void extract_element_and_node_values(Core::Elements::Element* ele,  //!< current element
          Teuchos::ParameterList& params,                                 //!< parameter list
          Core::FE::Discretization& discretization,                       //!< discretization
          Core::Elements::LocationArray& la                               //!< location array
          ) override;

      //! get material parameters
      void get_material_params(const Core::Elements::Element* ele,  //!< current element
          std::vector<double>& densn,                               //!< density at t_(n)
          std::vector<double>& densnp,  //!< density at t_(n+1) or t_(n+alpha_F)
          std::vector<double>& densam,  //!< density at t_(n+alpha_M)
          double& visc,                 //!< fluid viscosity
          const int iquad               //!< ID of current integration point
          ) override;

      //! evaluate Soret material
      void mat_soret(const Teuchos::RCP<const Core::Mat::Material> material,  //!< Soret material
          double& densn,   //!< density at time t_(n)
          double& densnp,  //!< density at time t_(n+1) or t_(n+alpha_F)
          double& densam   //!< density at time t_(n+alpha_M)
      );

      void mat_fourier(
          const Teuchos::RCP<const Core::Mat::Material> material,  //!< Fourier material
          double& densn,                                           //!< density at time t_(n)
          double& densnp,  //!< density at time t_(n+1) or t_(n+alpha_F)
          double& densam   //!< density at time t_(n+alpha_M)
      );

      //! set internal variables for element evaluation
      void set_internal_variables_for_mat_and_rhs() override;

      //! get thermo diffusion manager
      Teuchos::RCP<ScaTraEleDiffManagerSTIThermo> diff_manager()
      {
        return Teuchos::rcp_static_cast<ScaTraEleDiffManagerSTIThermo>(my::diffmanager_);
      };

      //! get internal variable manager for heat transfer within electrochemical substances
      Teuchos::RCP<ScaTraEleInternalVariableManagerSTIElch<nsd_, nen_>> var_manager()
      {
        return Teuchos::rcp_static_cast<ScaTraEleInternalVariableManagerSTIElch<nsd_, nen_>>(
            my::scatravarmanager_);
      };

      //! diffusion manager for thermodynamic electrodes
      Teuchos::RCP<ScaTraEleDiffManagerSTIElchElectrode> diffmanagerstielectrode_;

      //! utility class supporting element evaluation for electrodes
      Discret::Elements::ScaTraEleUtilsElchElectrode<distype>* utils_;
    };  // class ScaTraEleCalcSTIElectrode


    //! implementation of ScaTraEleDiffManagerSTIElchElectrode
    class ScaTraEleDiffManagerSTIElchElectrode : public ScaTraEleDiffManagerElchElectrode
    {
     public:
      //! constructor
      ScaTraEleDiffManagerSTIElchElectrode(int numscal)
          :  // constructor of base class
            ScaTraEleDiffManagerElchElectrode(numscal),

            // initialize internal member variables
            ocp_(0.),
            ocpderiv_(0.),
            ocpderiv2_(0.){};


      //! compute and store half cell open circuit potential and its first and second derivatives
      //! w.r.t. concentration
      void set_ocp_and_derivs(const Core::Elements::Element* ele, const double& concentration,
          const double& temperature)
      {
        const double faraday =
            Discret::Elements::ScaTraEleParameterElch::instance("scatra")->faraday();
        const double gasconstant =
            Discret::Elements::ScaTraEleParameterElch::instance("scatra")->gas_constant();
        // factor F/RT
        const double frt = faraday / (gasconstant * temperature);

        // access electrode material
        const Teuchos::RCP<const Mat::Electrode> matelectrode =
            Teuchos::rcp_dynamic_cast<const Mat::Electrode>(ele->material(1));
        if (matelectrode == Teuchos::null) FOUR_C_THROW("Invalid electrode material!");

        // no deformation available in this code part
        const double dummy_detF(1.0);
        // evaluate material
        ocp_ =
            matelectrode->compute_open_circuit_potential(concentration, faraday, frt, dummy_detF);
        ocpderiv_ = matelectrode->compute_d_open_circuit_potential_d_concentration(
            concentration, faraday, frt, dummy_detF);
        ocpderiv2_ =
            matelectrode->compute_d2_open_circuit_potential_d_concentration_d_concentration(
                concentration, faraday, frt, dummy_detF);
      };

      //! return half cell open circuit potential
      const double& get_ocp() { return ocp_; };

      //! return first derivative of half cell open circuit potential w.r.t. concentration
      const double& get_ocp_deriv() { return ocpderiv_; };

      //! return second derivative of half cell open circuit potential w.r.t. concentration
      const double& get_ocp_deriv2() { return ocpderiv2_; };

     protected:
      //! half cell open circuit potential
      double ocp_;

      //! first derivative of half cell open circuit potential w.r.t. concentration
      double ocpderiv_;

      //! second derivative of half cell open circuit potential w.r.t. concentration
      double ocpderiv2_;
    };  // class ScaTraEleDiffManagerSTIElchElectrode
  }     // namespace Elements
}  // namespace Discret
FOUR_C_NAMESPACE_CLOSE

#endif
