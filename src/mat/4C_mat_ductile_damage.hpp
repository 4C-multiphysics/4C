// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_DUCTILE_DAMAGE_HPP
#define FOUR_C_MAT_DUCTILE_DAMAGE_HPP

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
    //! Parameters of the ductile damage of a wrapped plastic solid material
    class DuctileDamage : public Core::Mat::PAR::Parameter
    {
     public:
      explicit DuctileDamage(const Core::Mat::PAR::Parameter::Data& matdata);

      std::shared_ptr<Core::Mat::Material> create_material() override;

      //! material id of the wrapped (undamaged) plastic solid material
      const int matid_;

      //! parameters of the fracture strain eps_f = d1 + d2 exp(d3 triaxiality) (Johnson-Cook)
      const double d1_;
      const double d2_;
      const double d3_;

      //! energy dissipated per unit crack area until the material is completely damaged
      const double fracture_energy_;

      //! ratio of the remaining to the initial strength at which the material point fails
      const double residual_strength_ratio_;
    };
  }  // namespace PAR

  class DuctileDamageType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "DuctileDamageType"; }

    static DuctileDamageType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static DuctileDamageType instance_;
  };

  /*!
   * @brief Isotropic ductile damage of a wrapped plastic solid material at finite strains
   *
   * The stress of the wrapped (undamaged) plastic material \f$\bar{\mathbf{S}}\f$ is degraded by
   * the scalar damage \f$D\f$ following the effective stress concept,
   * \f$\mathbf{S} = (1 - D) \bar{\mathbf{S}}\f$. The plastic flow of the wrapped material is driven
   * by the effective stress, i.e., the yield stress of the matrix is not affected by the damage
   * (Lemaitre, strain equivalence).
   *
   * Damage initiates as soon as the indicator \f$H = \int \mathrm{d}\bar{\varepsilon}_p /
   * \varepsilon_f(\eta)\f$ reaches one, with the equivalent plastic strain
   * \f$\bar{\varepsilon}_p\f$ of the wrapped material, the triaxiality \f$\eta\f$ of the effective
   * Cauchy stress and the fracture strain \f$\varepsilon_f = d_1 + d_2 \exp(d_3 \eta)\f$ of
   * Johnson and Cook (1985). It then evolves with the plastic displacement \f$\bar{u}_p = h
   * (\bar{\varepsilon}_p - \bar{\varepsilon}_{p,0})\f$ of a crack band of width \f$h\f$, the extent
   * of the element along the maximum principal direction at initiation in the reference
   * configuration:
   * \f[ D = 1 - \exp\left(-\frac{1}{G_f} \int \bar{\sigma}_\mathrm{eq} \, \mathrm{d}\bar{u}_p
   * \right), \f]
   * such that the dissipated energy \f$\int (1 - D) \bar{\sigma}_\mathrm{eq} \,
   * \mathrm{d}\bar{u}_p\f$ is \f$D G_f\f$ per unit crack area. The equivalent stress is taken from
   * the last converged step. The material point fails as soon as \f$1 - D\f$ drops to the residual
   * strength ratio.
   *
   * The wrapped material has to provide its equivalent plastic strain (Trait::Plastic). The
   * thermal coupling terms are degraded like the stress, see DamagedSolid.
   */
  class DuctileDamage : public DamagedSolid
  {
   public:
    DuctileDamage() = default;

    explicit DuctileDamage(PAR::DuctileDamage* params);

    [[nodiscard]] int unique_par_object_id() const override
    {
      return DuctileDamageType::instance().unique_par_object_id();
    }

    void pack(Core::Communication::PackBuffer& data) const override;

    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    [[nodiscard]] Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_ductile_damage;
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

    void register_output_data_names(
        std::unordered_map<std::string, int>& names_and_size) const override;

    bool evaluate_output_data(
        const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const override;

    //! Return the damage initiation indicator of the last evaluation at Gauss point @p gp
    [[nodiscard]] double initiation_indicator(int gp) const
    {
      return current_state_[gp].initiation_indicator;
    }

   private:
    void setup_damage(int numgp) override;

    void update_damage() override;

    void reset_damage() override;

    //! Damage state of a Gauss point
    struct GaussPointState
    {
      //! integral of the plastic strain increments over the fracture strain
      double initiation_indicator = 0.0;

      //! whether damage initiated
      bool initiated = false;

      //! extent of the element along the maximum principal direction at initiation in the
      //! reference configuration
      double characteristic_length = 0.0;

      //! equivalent plastic strain of the wrapped material
      double plastic_strain = 0.0;

      //! von Mises equivalent of the effective Cauchy stress
      double equivalent_stress = 0.0;

      //! integral of the equivalent stress over the plastic displacement since initiation
      double plastic_work = 0.0;

      double damage = 0.0;

      //! whether the remaining strength dropped to the residual strength
      bool failed = false;
    };

    PAR::DuctileDamage* params_ = nullptr;

    //! damage state of the last converged step
    std::vector<GaussPointState> converged_state_;

    //! damage state of the last evaluation
    std::vector<GaussPointState> current_state_;
  };
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
