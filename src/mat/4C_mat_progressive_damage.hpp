// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_PROGRESSIVE_DAMAGE_HPP
#define FOUR_C_MAT_PROGRESSIVE_DAMAGE_HPP

#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_damaged_solid.hpp"
#include "4C_material_parameter_base.hpp"

#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    //! Parameters of the progressive damage of a wrapped solid material
    class ProgressiveDamage : public Core::Mat::PAR::Parameter
    {
     public:
      explicit ProgressiveDamage(const Core::Mat::PAR::Parameter::Data& matdata);

      std::shared_ptr<Core::Mat::Material> create_material() override;

      //! material id of the wrapped (undamaged) solid material
      const int matid_;

      //! maximum principal Cauchy stress of the undamaged material at which damage initiates
      const double tensile_strength_;

      //! energy dissipated per unit crack area until the material is completely damaged
      const double fracture_energy_;

      //! ratio of the remaining to the initial strength at which the material point fails
      const double residual_strength_ratio_;
    };
  }  // namespace PAR

  class ProgressiveDamageType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "ProgressiveDamageType"; }

    static ProgressiveDamageType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static ProgressiveDamageType instance_;
  };

  /*!
   * @brief Isotropic progressive damage of a wrapped solid material at finite strains
   *
   * The stress of the wrapped (undamaged) material \f$\bar{\mathbf{S}}\f$ is degraded by the
   * scalar damage \f$D\f$ following the effective stress concept (Simo and Ju, 1987):
   * \f[ \mathbf{S} = (1 - D) \bar{\mathbf{S}}. \f]
   *
   * Damage initiates as soon as the maximum principal Cauchy stress of the undamaged material
   * reaches the tensile strength \f$f_t\f$ (Rankine criterion). It evolves with the maximum
   * principal logarithmic strain \f$\varepsilon_1 = \frac{1}{2} \ln \lambda_1\f$, where
   * \f$\lambda_1\f$ is the maximum eigenvalue of the right Cauchy-Green tensor, i.e., its
   * direction rotates with the deformation, which avoids the stress locking of a fixed crack
   * direction. With the history variable \f$\kappa = \max \varepsilon_1\f$ and the strain
   * \f$\varepsilon_0\f$ at initiation, the exponential softening law of the crack band model reads
   * (Jirasek and Bazant, Inelastic Analysis of Structures, 2002)
   * \f[ D(\kappa) = 1 - \frac{\varepsilon_0}{\kappa}
   *     \exp\left(-\frac{\kappa - \varepsilon_0}{\varepsilon_f - \varepsilon_0}\right), \quad
   *     \varepsilon_f = \frac{G_f}{f_t h} + \frac{\varepsilon_0}{2}, \f]
   * with the fracture energy \f$G_f\f$ and the extent \f$h\f$ of the element along the maximum
   * principal direction at initiation in the reference configuration, such that a linear elastic
   * band of width \f$h\f$ dissipates \f$G_f\f$ per unit crack area. For linear elements, the
   * extent equals the projected element length of Oliver (1989). It is fixed at initiation, since
   * the softening law must not change while the material softens. The material therefore needs the
   * reference coordinates of the element nodes.
   *
   * The material point fails, e.g., to deactivate its element, as soon as the remaining strength
   * \f$f_t \exp(-(\kappa - \varepsilon_0) / (\varepsilon_f - \varepsilon_0))\f$ drops to the
   * residual strength ratio \f$r\f$ times the tensile strength. Its damage does not grow anymore.
   * The failure is not formulated in terms of a critical damage, since the damage approaches one
   * long before the fracture energy is dissipated if \f$\varepsilon_f \gg \varepsilon_0\f$. A
   * critical damage would therefore remove the material point the earlier, the smaller the element
   * is. With the residual strength ratio, the fraction of the fracture energy dissipated until
   * failure is about \f$1 - r\f$ independent of the element.
   *
   * The thermal coupling terms are degraded like the stress, see DamagedSolid.
   */
  class ProgressiveDamage : public DamagedSolid
  {
   public:
    ProgressiveDamage() = default;

    explicit ProgressiveDamage(PAR::ProgressiveDamage* params);

    [[nodiscard]] int unique_par_object_id() const override
    {
      return ProgressiveDamageType::instance().unique_par_object_id();
    }

    void pack(Core::Communication::PackBuffer& data) const override;

    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    [[nodiscard]] Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_progressive_damage;
    }

    [[nodiscard]] std::shared_ptr<Core::Mat::Material> clone() const override;

    [[nodiscard]] Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    void evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) override;

    [[nodiscard]] bool is_failed(int gp) const override { return current_state_[gp].failed; }

    [[nodiscard]] double damage(int gp) const override { return current_state_[gp].damage; }

   private:
    void setup_damage(int numgp) override;

    void update_damage() override;

    void reset_damage() override;

    //! Damage state of a Gauss point
    struct GaussPointState
    {
      //! whether damage initiated
      bool initiated = false;

      //! extent of the element along the maximum principal direction at initiation in the
      //! reference configuration
      double characteristic_length = 0.0;

      //! maximum principal logarithmic strain at initiation
      double initiation_strain = 0.0;

      //! maximum of the maximum principal logarithmic strain since initiation
      double max_strain = 0.0;

      double damage = 0.0;

      //! whether the remaining strength dropped to the residual strength
      bool failed = false;
    };

    PAR::ProgressiveDamage* params_ = nullptr;

    //! damage state of the last converged step
    std::vector<GaussPointState> converged_state_;

    //! damage state of the last evaluation
    std::vector<GaussPointState> current_state_;
  };
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
