// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_ELE_CALC_LOMA_HPP
#define FOUR_C_SCATRA_ELE_CALC_LOMA_HPP

#include "4C_config.hpp"

#include "4C_mat_fourier.hpp"
#include "4C_scatra_ele_calc.hpp"

FOUR_C_NAMESPACE_OPEN


namespace Discret
{
  namespace Elements
  {
    template <Core::FE::CellType distype>
    class ScaTraEleCalcLoma : public ScaTraEleCalc<distype>
    {
     private:
      //! private constructor for singletons
      ScaTraEleCalcLoma(const int numdofpernode, const int numscal, const std::string& disname);

      using my = ScaTraEleCalc<distype>;
      using my::nen_;
      using my::nsd_;

     public:
      /// Singleton access method
      static ScaTraEleCalcLoma<distype>* instance(
          const int numdofpernode, const int numscal, const std::string& disname);

      int evaluate_action(Core::Elements::Element* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, const ScaTra::Action& action,
          Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1,
          Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseVector& elevec2,
          Core::LinAlg::SerialDenseVector& elevec3) override;

     private:
      /*========================================================================*/
      //! @name material and related and related functions
      /*========================================================================*/

      //! evaluate material
      void materials(const std::shared_ptr<const Core::Mat::Material>
                         material,  //!< pointer to current material
          const int k,              //!< id of current scalar
          double& densn,            //!< density at t_(n)
          double& densnp,           //!< density at t_(n+1) or t_(n+alpha_F)
          double& densam,           //!< density at t_(n+alpha_M)
          double& visc,             //!< fluid viscosity
          const int iquad = -1      //!< id of current gauss point (default = -1)
          ) override;

      //! material Sutherland
      void mat_sutherland(const std::shared_ptr<const Core::Mat::Material>
                              material,  //!< pointer to current material
          const int k,                   //!< id of current scalar
          double& densn,                 //!< density at t_(n)
          double& densnp,                //!< density at t_(n+1) or t_(n+alpha_F)
          double& densam,                //!< density at t_(n+alpha_M)
          double& visc                   //!< fluid viscosity
      );

      /*!
       * @brief Evaluate a Fourier material using the associated solid material's density.
       *
       * @param[in] material Fourier material of the scalar transport element
       * @param[in] solid_material_density Density from the associated solid material
       * @param[in] k Current scalar id
       * @param[out] densn Density at \f$t_n\f$
       * @param[out] densnp Density at \f$t_{n+1}\f$ or \f$t_{n+\alpha_F}\f$
       * @param[out] densam Density at \f$t_{n+\alpha_M}\f$
       * @param[out] visc Fluid viscosity
       */
      void mat_thermo_with_structural_density(const std::shared_ptr<const Mat::Fourier> material,
          const double solid_material_density, const int k, double& densn, double& densnp,
          double& densam, double& visc);

      /*========================================================================*/
      //! @name methods for evaluation of individual terms
      /*========================================================================*/

      //! compute rhs containing bodyforce
      void get_rhs_int(double& rhsint,  //!< rhs containing bodyforce at Gauss point
          const double densnp,          //!< density at t_(n+1)
          const int k                   //!< index of current scalar
          ) override;

      //! calculation of convective element matrix: add conservative contributions
      void calc_mat_conv_add_cons(
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to be filled
          const int k,                            //!< index of current scalar
          const double timefacfac,  //!< domain-integration factor times time-integration factor
          const Core::LinAlg::Matrix<nsd_, 1>& convelint,  //!< convective velocity at Gauss point
          const Core::LinAlg::Matrix<nsd_, 1>& gradphi,    //!< scalar gradient at Gauss point
          const double vdiv,                               //!< velocity divergence
          const double densnp,                             //!< density at time_(n+1)
          const double visc                                //!< viscosity
      );

      //! adaption of convective term for rhs
      void recompute_conv_phi_for_rhs(double& conv_phi,   //!< convective contribution
          const int k,                                    //!< index of current scalar
          const Core::LinAlg::Matrix<nsd_, 1>& sgvelint,  //!< subgrid-scale velocity at Gauss point
          const Core::LinAlg::Matrix<nsd_, 1>& gradphi,   //!< scalar gradient at Gauss point
          const double densnp,                            //!< density at time_(n+1)
          const double densn,                             //!< density at time_(n)
          const double phinp,                             //!< scalar at time_(n+1)
          const double phin,                              //!< scalar at time_(n)
          const double vdiv                               //!< velocity divergence
      );

      /*========================================================================*/
      //! @name additional service routines
      /*========================================================================*/

      //! calculate domain integral
      void calculate_domain_and_bodyforce(
          Core::LinAlg::SerialDenseVector& scalars, const Core::Elements::Element* ele);

      //! extract element based or nodal values and return extracted values of phinp
      void extract_element_and_node_values(Core::Elements::Element* ele,
          Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          Core::Elements::LocationArray& la) override;

      //! get density at integration point
      double get_density(const Core::Elements::Element* ele,
          std::shared_ptr<const Core::Mat::Material> material, Teuchos::ParameterList& params,
          const double tempnp) override;

      //! calculate viscous part of subgrid-scale velocity
      void calc_subgr_velocity_visc(Core::LinAlg::Matrix<nsd_, 1>& epsilonvel) override;

      /*========================================================================*/
      //! @name scalar degrees of freedom and related
      /*========================================================================*/

      //! scalar at t_(n+alpha_M)
      std::vector<Core::LinAlg::Matrix<nen_, 1>> ephiam_;
      //!
      std::vector<double> densgradfac_;

      //! thermodynamic pressure at t_(n+1) or t_(n+alpha_F) (LOMA specific)
      double thermpressnp_;
      //! thermodynamic pressure at t_(n+alpha_M) (LOMA specific)
      double thermpressam_;
      //! time derivative of thermodynamic pressure (LOMA specific)
      double thermpressdt_;

      //! specific heat capacity (either at constant pressure or at constant volume)
      double shc_;
    };

    /// Scatra reaction manager
    /*!
        special reaction manager for loma
    */
    class ScaTraEleReaManagerLoma : public ScaTraEleReaManager
    {
     public:
      ScaTraEleReaManagerLoma(int numscal) : ScaTraEleReaManager(numscal), reatemprhs_(numscal, 0.0)
      {
        return;
      }

      //! Set reaction / temperature term for rhs
      void set_rea_temp_rhs(const double reatemprhs, const int k)
      {
        reatemprhs_[k] = reatemprhs;
        return;
      }

      //! Return reaction / temperature term for rhs
      double get_rea_temp_rhs(const int k) { return reatemprhs_[k]; }

     private:
      //! reaction / temperature term for rhs
      std::vector<double> reatemprhs_;
    };

  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
