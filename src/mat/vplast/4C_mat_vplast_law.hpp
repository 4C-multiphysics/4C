// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_VPLAST_LAW_HPP
#define FOUR_C_MAT_VPLAST_LAW_HPP

#include "4C_config.hpp"

#include "4C_comm_parobject.hpp"
#include "4C_comm_parobjectfactory.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_mat_inelastic_defgrad_factors_service.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_solid_ele_fibers.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <Teuchos_RCP.hpp>

#include <memory>
#include <optional>


FOUR_C_NAMESPACE_OPEN


namespace Mat
{
  namespace Viscoplastic
  {

    namespace PAR
    {

    };

    /*----------------------------------------------------------------------*/
    /*! \class Law
     * \brief Implementation of an interface called by the viscoplasticity finite strain model
     * InelasticDefgradTransvIsotropElastViscoplast to evaluate the viscoplastic flow rule (plastic
     * strain rate in terms of stress and plasticity state) and the hardening law (increase in yield
     * stress / flow resistance).
     *
     * This class implements an interface for viscoplasticity laws (combinations of flow rule and
     * hardening model), used by the (transversely) isotropic
     * material model InelasticDefgradTransvIsotropElastViscoplast to describe viscoplastic material
     * response. Specifically, it is used to evaluate the plastic strain rate for the given local
     * stress, taking into account the current plasticity state of the material (current plastic
     * strain). It also provides the linearization of this quantity with respect to both the stress
     * and the plastic strain, required in the linearization of the material referenced above. Both
     * viscoplasticity laws with and without yield surfaces can be implemented.
     *
     * For further information on the implemented interface, refer to :
     *    -# Master's Thesis : Dragos-Corneliu Ana, Continuum Modeling and Calibration of
     * Viscoplasticity in the Context of the Lithium Anode in Solid State Batteries, Supervisor:
     * Christoph Schmidt, 2024
     */
    class Law
    {
     public:
      /// construct viscoplastic laws with specific material params and error registration settings
      explicit Law(Core::Mat::PAR::Parameter* params,
          const InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorRegistrationSettings&
              error_registration_settings);
      /// construct empty viscoplastic law
      Law();

      /// virtual destructor
      virtual ~Law() = default;

      /*!
       * @brief create object by input parameter ID
       *
       * @param[in] matnum  material ID
       * @param[in] error_registration_settings  error registration settings for plastic strain
       * increments and derivative increments
       * @return pointer to material that is defined by material ID
       */
      static std::shared_ptr<Law> factory(int matnum,
          const InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorRegistrationSettings&
              error_registration_settings);

      /// provide material type
      [[nodiscard]] virtual Core::Materials::MaterialType material_type() const = 0;

      /// does the viscoplastic law use a yield surface formulation, or is it a no-yield-surface
      /// law?
      [[nodiscard]] virtual bool uses_yield_surface() const = 0;

      /*!
       * @brief Evaluates the yield stress, or more generally the flow resistance (to accommodate
       * viscoplastic laws with and without yield surfaces in a common framework)
       *
       *
       * @param[in] equiv_stress equivalent stress
       * @param[in] equiv_plastic_strain accumulated plastic strain
       * @param[in/out] err_status error status of the evaluation
       *
       */
      [[nodiscard]] virtual double compute_flow_resistance(double equiv_stress,
          double equiv_plastic_strain,
          Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType& err_status) const = 0;


      /*!
       * @brief Evaluate the equivalent plastic strain rate \f$ \dot{\varepsilon}^{\text{p}} \f$ for
       * a given equivalent stress \f$ \overflow{\sigma} \f$ and a given plastic strain \f$
       * \varepsilon^{\text{p}} \f$
       *
       * @note To use the viscoplasticity components in the time
       * integration of history variables within InelasticDefgradTransvIsotropElastViscoplast, we
       * may check for eventual overflow errors within them. Specifically, we focus on the term
       * \f$ \Delta t \dot{\varepsilon}^{\text{p}} \f$, which shall be
       * evaluable in the specific time integration used. Moreover,
       * for some viscoplasticity laws, we have to make sure that the given plastic strain is \f$
       * \varepsilon^{\text{p}} \ge 0 \f$, otherwise the evaluation of either flow rule or hardening
       * model fails. This depends on the employed viscoplastic law.
       *
       *
       * @param[in] equiv_stress Equivalent stress \f$ \overline{\sigma}  \f$
       * @param[in] equiv_plastic_strain Equivalent plastic strain \f$ \varepsilon^{\text{p}}\f$
       * @param[in] dt Time step size (used solely for overflow error checking, see @note)
       * @param[out] err_status Output variable: error due the term considered in @note?
       * @return Equivalent plastic strain rate \f$ \dot{\varepsilon}^{\text{p}} \f$
       */
      virtual double evaluate_plastic_strain_rate(double equiv_stress, double equiv_plastic_strain,
          double dt, Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType& err_status,
          bool update_hist_var = true) = 0;

      /*!
       * @brief Evaluate the derivatives of the equivalent plastic strain rate \f$
       * \dot{\varepsilon}^{\text{p}} \f$ for a given equivalent stress \f$ \overline{\sigma} \f$
       * and a given plastic strain \f$ \varepsilon^{\text{p}} \f$
       *
       * @note In addition to checking overflow based on the plastic strain increment (see
       * documentation of evaluate_plastic_strain_rate), we may also verify overflow based on the
       * derivatives of the plastic strain rate, scaled by the considered timestep.
       *
       *
       * @param[in] equiv_stress Equivalent stress \f$ \overline{\sigma}  \f$
       * @param[in] equiv_plastic_strain Equivalent plastic strain \f$ \varepsilon^{\text{p}}\f$
       * @param[in] dt Time step size (used solely for overflow error checking, see @note of
       * evaluate_plastic_strain_rate)
       * @param[out] err_status Output variable: error of the terms considered in @note?
       * @return Derivatives of the equivalent plastic strain rate w.r.t. the equivalent stress,
       *         plastic strain, and temperature.
       */
      [[nodiscard]] virtual InelasticDefgradTransvIsotropElastViscoplastUtils::
          PlasticStrainRateDerivs
          evaluate_derivatives_of_plastic_strain_rate(double equiv_stress,
              double equiv_plastic_strain, double dt,
              Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType& err_status)
              const = 0;

      /// Return material parameters
      [[nodiscard]] virtual Core::Mat::PAR::Parameter* parameter() const { return params_; }

      /*!
       * @brief Setup viscoplasticity law for the specific element
       *
       * @param[in] numgp Number of Gauss points
       * @param[in] container Input parameter Container
       */
      virtual void setup(unsigned int numgp, const Discret::Elements::Fibers& fibers,
          const std::optional<Discret::Elements::CoordinateSystem>& coord_system) {};

      /*!
       * @brief Pre-evaluation, intended to be used for stuff that has to be done only once per
       *        evaluate()
       *
       * @param[in] params  Various parameters, such as e.g. the temperature in TSI simulations
       * @param[in] gp      Current Gauss point
       */
      virtual void pre_evaluate(const Teuchos::ParameterList& params, unsigned int gp)
      {
        gp_ = gp;
      };

      /*!
       * @brief Update history variables of the viscoplasticity law for next time step at a given
       * Gauss point
       *
       * @param[in] gp Current Gauss point
       */
      virtual void update(unsigned int gp) {};

      /*!
       * @brief Update the history variables for a specific GP after a converged substep
       * (substepping procedure)
       *
       * @param[in] gp      Gauss point
       */
      virtual void update_gp_state_after_substep(unsigned int gp) {};

      /*!
       * @brief Reset the history variables for a specific GP to the state before substepping.
       * Specifically, it resets the substep history variables to the ones from the previous
       * converged time instant: \f$ S_{n + \tilde{m}} \gets S_{n} \f$ for history variables $S_{n +
       * \tilde{m}}$ and $S_{n}$ at the previous time instant in the current substep, and at the
       * previous time instant in the current timestep, respectively.
       *
       * @param[in] gp Gauss point
       */
      virtual void reset_gp_substep_state(unsigned int gp) {};

      virtual void pack_viscoplastic_law(Core::Communication::PackBuffer& data) const {};

      virtual void unpack_viscoplastic_law(Core::Communication::UnpackBuffer& buffer) {};


      /*!
       * @brief Register names of the internal data that should be saved during runtime output
       *
       * @param[out] name_and_size Unordered map of names of the data with the respective vector
       * size
       */
      virtual void register_output_data_names(
          std::unordered_map<std::string, int>& names_and_size) const
      {
      }

      /*!
       * @brief Evaluate internal data for every Gauss point saved for output during runtime
       * output
       *
       * @param[in] name  Name of the data to export
       * @param[out] data NUMGPxNUMDATA Matrix holding the data
       *
       * @return true if data is set by the material, otherwise false
       */
      virtual bool evaluate_output_data(
          const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const
      {
        return false;
      }

      //! struct containing information required for integrating the plastic strain according
      //! to its evolution equation / flow rule
      struct InputIntegratePlasticStrain
      {
        //! fixed equivalent stress \f$ \overline{\sigma}_{n+1} \f$
        double equiv_stress;

        //! plastic strain at the previous time instant \f$ \varepsilon_{\text{p},n} \f$
        double last_plastic_strain;

        //! integration timestep / substep \f$ \Delta t \f$ / \f$ \Delta \tilde{t} \f$
        double step;

        //! Newton-Raphson residual tolerance used for integration
        double residual_tolerance;

        //! maximum number of Newton-Raphson iterations used for integration
        unsigned int max_iter;
      };

      /*!
       * @brief Integrates the equivalent plastic strain for a fixed equivalent stress based on its
       * evolution equation / flow rule
       *
       * @note Relevant for the plastic strain update / "interpolation" within the Adaptive Estimate
       * Interpolation procedures
       *
       * @param[in] input_integrate_plastic_strain struct containing variables required for this
       * integration (particularly the fixed equivalent stress)
       * @return Integrated plastic strain if the integration was successful; nullopt if not
       */
      [[nodiscard]] std::optional<double> integrate_plastic_strain(
          const InputIntegratePlasticStrain& input_integrate_plastic_strain);

     protected:
      /// Gauss point index
      unsigned int gp_;

      /// error registration settings for the plastic strain increments and the derivative
      /// increments
      const Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorRegistrationSettings
          error_registration_settings_;


     private:
      /// material parameters
      Core::Mat::PAR::Parameter* params_;
    };
  }  // namespace Viscoplastic


}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
