// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_VPLAST_REFORM_JOHNSONCOOK_HPP
#define FOUR_C_MAT_VPLAST_REFORM_JOHNSONCOOK_HPP
#include "4C_config.hpp"

#include "4C_comm_parobject.hpp"
#include "4C_comm_parobjectfactory.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_mat_vplast_law.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <Teuchos_RCP.hpp>

#include <cmath>
#include <memory>


FOUR_C_NAMESPACE_OPEN


namespace Mat
{

  namespace Viscoplastic
  {

    namespace PAR
    {
      /*----------------------------------------------------------------------*/
      /*! \class ReformulatedJohnsonCook
       *
       * Parameter class of ReformulatedJohnsonCook
       */
      class ReformulatedJohnsonCook : public Core::Mat::PAR::Parameter
      {
       public:
        explicit ReformulatedJohnsonCook(const Core::Mat::PAR::Parameter::Data& matdata);

        std::shared_ptr<Core::Mat::Material> create_material() override { return nullptr; };

        // getter methods
        //! get strain rate prefactor \f$ \dot{P}_0 \f$
        [[nodiscard]] double strain_rate_pre_fac() const { return strain_rate_prefac_; };
        //! get exponential factor \f$ C \f$
        [[nodiscard]] double strain_rate_exp_fac() const { return strain_rate_exp_fac_; };
        //! get initial yield strength (\f$ A_0 \f$ scaled by the temperature dependence factor)
        [[nodiscard]] double init_yield_strength() const { return init_yield_strength_; };
        //! get prefactor of the isotropic hardening stress (\f$ B_0 \f$ scaled by the temperature
        //! dependence factor)
        [[nodiscard]] double isotrop_harden_prefac() const { return isotrop_harden_prefac_; };
        //! get exponent of the isotropic hardening stress \f$ n \f$
        [[nodiscard]] double isotrop_harden_exp() const { return isotrop_harden_exp_; };
        //! get reference temperature
        [[nodiscard]] double ref_temperature() const { return ref_temperature_; };
        //! get melting temperature
        [[nodiscard]] double melt_temperature() const { return melt_temperature_; };
        //! get temperature sensitivity factor
        [[nodiscard]] double temperature_sens() const { return temperature_sens_; };

       private:
        //! strain rate prefactor \f$ \dot{P}_0 \f$
        const double strain_rate_prefac_;

        //! exponential factor \f$ C \f$
        const double strain_rate_exp_fac_;

        //! initial yield strength (\f$ A_0 \f$ scaled by the temperature dependence factor)
        double init_yield_strength_;

        //! prefactor of the isotropic hardening stress (\f$ B_0 \f$ scaled by the temperature
        //! dependence factor)
        double isotrop_harden_prefac_;

        //! exponent of the isotropic hardening stress \f$ n \f$
        const double isotrop_harden_exp_;

        //! reference temperature \f$ T_{\mathrm{ref}} \f$
        const double ref_temperature_;

        //! melting temperature \f$ T_{\mathrm{melt}} \f$
        const double melt_temperature_;

        //! temperature sensitivity \f$ M \f$
        const double temperature_sens_;
      };
    }  // namespace PAR

    /*----------------------------------------------------------------------*/
    /*! \class ReformulatedJohnsonCook
     *
     *  Viscoplasticity law associated to the Reformulated Johnson-Cook model proposed in
     *  -# Mareau et al., A thermodynamically consistent formulation of the Johnson-Cook model,
     *     Mechanics of Materials 143, 2020
     */
    class ReformulatedJohnsonCook : public Law
    {
     public:
      explicit ReformulatedJohnsonCook(Core::Mat::PAR::Parameter* params);

      Mat::Viscoplastic::PAR::ReformulatedJohnsonCook* parameter() const override
      {
        return dynamic_cast<Mat::Viscoplastic::PAR::ReformulatedJohnsonCook*>(
            Mat::Viscoplastic::Law::parameter());
      }

      Core::Materials::MaterialType material_type() const override
      {
        return Core::Materials::mvl_reformulated_Johnson_Cook;
      };

      double evaluate_stress_ratio(
          const double equiv_stress, const double equiv_plastic_strain) override;

      double compute_flow_resistance(const double equiv_stress, const double equiv_plastic_strain,
          Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType& err_status) override;

      double evaluate_plastic_strain_rate(const double equiv_stress,
          const double equiv_plastic_strain, const double dt, const double max_plastic_strain_incr,
          Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType& err_status,
          const bool update_hist_var) override;

      InelasticDefgradTransvIsotropElastViscoplastUtils::PlasticStrainRateDerivs
      evaluate_derivatives_of_plastic_strain_rate(const double equiv_stress,
          const double equiv_plastic_strain, const double dt,
          const double max_plastic_strain_deriv_incr,
          Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType& err_status,
          const bool update_hist_var) override;

      void setup(const int numgp, const Discret::Elements::Fibers& fibers,
          const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override;


      void pre_evaluate(const Teuchos::ParameterList& params, int gp) override;

      void update() override {};

      void update_gp_state(int gp) override {};

      void pack_viscoplastic_law(Core::Communication::PackBuffer& data) const override;

      void unpack_viscoplastic_law(Core::Communication::UnpackBuffer& buffer) override;

      void register_output_data_names(
          std::unordered_map<std::string, int>& names_and_size) const override;

      bool evaluate_output_data(
          const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const override;

     private:
      /// struct containing constant parameters to be evaluated only once
      struct ConstPars
      {
        /// perfect plasticity? \f$ B * N == 0 \f$
        bool is_perfect_plasticity;

        /// prefactor \f$ P \f$ of the plastic strain rate
        double p;

        /// prefactor logarithm \f$ \log(P) \f$
        double log_p;

        /// exponent \f$ E \f$ of the plastic strain rate
        double e;

        /// \f$ \log(P*E) \f$
        double log_p_e;

        /// hardening prefactor \f$ B \f$
        double B;

        /// hardening exponent \f$ N \f$
        double N;

        /// logarithm \f$ \log(B N) \f$
        double log_B_N;

        /// initial yield strength
        double sigma_Y0;

        /// constructor
        ConstPars(const double prefac, const double expon, const double harden_prefac,
            const double harden_expon, const double initial_yield_strength)
            : is_perfect_plasticity(harden_prefac == 0.0 || harden_expon == 0.0),
              p(prefac),
              log_p(std::log(p)),
              e(expon),
              log_p_e(std::log(prefac * expon)),
              B(harden_prefac),
              N(harden_expon),
              log_B_N(is_perfect_plasticity ? 0.0 : set_log_b_n(harden_prefac, harden_expon)),
              sigma_Y0(initial_yield_strength)
        {
          FOUR_C_ASSERT_ALWAYS(B >= 0.0,
              "The current Reformulated Johnson-Cook viscoplastic law only enables hardening "
              "prefactors >= 0: current specification: {}",
              B);
          FOUR_C_ASSERT_ALWAYS(N >= 0.0,
              "The current Reformulated Johnson-Cook viscoplastic law only enables hardening "
              "exponents >= 0: current specification: {}",
              N);
        }

        /*!
         * @brief Computes log(B * N) for the Johnson-Cook hardening law, assuming B * N > 0.0
         *
         *
         * @param[in] B Hardening prefactor.
         * @param[in] N Hardening exponent.
         * @return logarithm log(B * N)
         */
        double set_log_b_n(const double B, const double N)
        {
          FOUR_C_ASSERT_ALWAYS(
              B * N > 0.0, "You should not be here when setting B * N: {} * {} <= 0.0", B, N);
          return std::log(B * N);
        }
      };

      /// instance of ConstPars struct
      const ConstPars const_pars_;

      /// temperature ratio \f$ D_T = 1 - \frac{T^M - T_{\mathrm{ref}}^M}{T_{\mathrm{melt}}^M -
      /// T_{\mathrm{ref}}^M} \f$
      double temperature_ratio_ = 1.0;

      /// logarithm of temperature ratio \f$ \log(D_T) =  \log( 1 - \frac{T^M -
      /// T_{\mathrm{ref}}^M}{T_{\mathrm{melt}}^M - T_{\mathrm{ref}}^M} ) \f$
      double log_temperature_ratio_;

      /// logarithm of negative temperature ratio derivative wrt temperature \f$ \log( - \partial
      /// D_T / \partial T) = \log(M
      /// \frac{T^{M-1}}{T_{\mathrm{melt}}^M - T_{\mathrm{ref}}^M} ) \f$
      double log_neg_temperature_ratio_deriv_;


      //! struct containing quantities at different time points (i.e., "current" at \f[ t_{n+1} \f])
      struct TimeStepQuantities
      {
        //! yield strength (for all Gauss points)
        std::vector<double> current_yield_strength_;
      };
      TimeStepQuantities time_step_quantities_;
    };

  }  // namespace Viscoplastic

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
