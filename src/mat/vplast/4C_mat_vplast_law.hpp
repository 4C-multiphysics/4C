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
#include "4C_solid_3D_ele_fibers.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <Teuchos_RCP.hpp>

#include <memory>


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
      /// construct viscoplastic laws with specific material params
      explicit Law(Core::Mat::PAR::Parameter* params);
      /// construct empty viscoplastic law
      Law();

      /// virtual destructor
      virtual ~Law() = default;

      /*!
       * @brief create object by input parameter ID
       *
       * @param[in] matnum  material ID
       * @return pointer to material that is defined by material ID
       */
      static std::shared_ptr<Law> factory(int matnum);

      /// provide material type
      virtual Core::Materials::MaterialType material_type() const = 0;

      /*!
       * @brief Evaluate the ratio of the equivalent stress \f$ \overline{\sigma} \f$ to the yield
       * stress \f$ \sigma_{\text{Y}} \f$ (or for flow rules without yield surface: ratio of the
       * equivalent stress \f$ \overline{\sigma} \f$ to the flow resistance \f$ S \f$)
       *
       * @param[in] equiv_stress Equivalent stress \f$ \overline{\sigma}  \f$
       * @param[in] equiv_plastic_strain Equivalent plastic strain \f$ \varepsilon^{\text{p}}\f$
       * @return Stress ratio: \f$ \frac{\overline{\sigma}}{\sigma_{\text{Y}}} \f$ (for flow rules
       * with yield surface) or \f$ \frac{\overline{\sigma}}{S} \f$ (for flow rules
       * without yield surface)
       */
      virtual double evaluate_stress_ratio(
          const double equiv_stress, const double equiv_plastic_strain) = 0;

      /*!
       * @brief Evaluate the equivalent plastic strain rate \f$ \dot{\varepsilon}^{\text{p}} \f$ for
       * a given equivalent stress \f$ \overflow{\sigma} \f$ and a given plastic strain \f$
       * \varepsilon^{\text{p}} \f$
       *
       * @note To use the viscoplasticity components in the time
       * integration of history variables within InelasticDefgradTransvIsotropElastViscoplast, we
       * have to check for eventual overflow errors within them. Specifically, we focus on the term
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
       * @param[in] max_plastic_strain_incr maximum, numerically evaluable plastic
       * strain increment \f$ \Delta t \dot{\varepsilon}^{\text{p}}) \f$ (before throwing an
       * overflow error)
       * @param[out] err_status output variable: error of the terms considered in
       * @note?
       * @return Equivalent plastic strain rate \f$ \dot{\varepsilon}^{\text{p}} \f$
       */
      virtual double evaluate_plastic_strain_rate(const double equiv_stress,
          const double equiv_plastic_strain, const double dt, const double max_plastic_strain_incr,
          Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType& err_status,
          const bool update_hist_var = true) = 0;

      /*!
       * @brief Evaluate the derivatives of the equivalent plastic strain rate \f$
       * \dot{\varepsilon}^{\text{p}} \f$ for a given equivalent stress \f$ \overline{\sigma} \f$
       * and a given plastic strain \f$ \varepsilon^{\text{p}} \f$
       *
       * @param[in] equiv_stress Equivalent stress \f$ \overline{\sigma}  \f$
       * @param[in] equiv_plastic_strain Equivalent plastic strain \f$ \varepsilon^{\text{p}}\f$
       * @param[in] dt Time step size (used solely for overflow error checking, see @note of
       * evaluate_plastic_strain_rate)
       * @param[in] max_plastic_strain_deriv_incr Maximum
       * numerically evaluable increment of the
       * plastic strain derivatives (before throwing an overflow error),
       * i.e. \f$ \Delta t \frac{\partial \dot{\varepsilon}^{\text{p}}}{\partial s},~ s \in
       * \{\varepsilon^{\text{p}}, \overline{\sigma}\} \f$
       * @param[out] err_status output variable: error of the terms considered in @note?
       * @return Derivatives of the equivalent plastic strain rate w.r.t. the equivalent stress
       *         (element 0 of matrix) and the plastic strain (element 1 of matrix)
       */
      virtual Core::LinAlg::Matrix<2, 1> evaluate_derivatives_of_plastic_strain_rate(
          const double equiv_stress, const double equiv_plastic_strain, const double dt,
          const double max_plastic_strain_deriv_incr,
          Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType& err_status,
          const bool update_hist_var = true) = 0;

      /// Return material parameters
      virtual Core::Mat::PAR::Parameter* parameter() const { return params_; }

      /*!
       * @brief Setup viscoplasticity law for the specific element
       *
       * @param[in] numgp Number of Gauss points
       * @param[in] container Input parameter Container
       */
      virtual void setup(const int numgp, const Discret::Elements::Fibers& fibers,
          const std::optional<Discret::Elements::CoordinateSystem>& coord_system) = 0;

      /*!
       * @brief Pre-evaluation, intended to be used for stuff that has to be done only once per
       *        evaluate()
       *
       * @param[in] gp      Current Gauss point
       */
      virtual void pre_evaluate(int gp) = 0;

      /*!
       * @brief Update history variables of the viscoplasticity law for next time step
       */
      virtual void update() = 0;

      /*!
       * @brief Update the history variables for a specific GP after a converged substep
       * (substepping procedure)
       *
       * @param[in] gp      Gauss point
       */
      virtual void update_gp_state(int gp) = 0;

      virtual void pack_viscoplastic_law(Core::Communication::PackBuffer& data) const = 0;

      virtual void unpack_viscoplastic_law(Core::Communication::UnpackBuffer& buffer) = 0;

     private:
      /// material parameters
      Core::Mat::PAR::Parameter* params_;
    };
  }  // namespace Viscoplastic


}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
