// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_DAMAGED_SOLID_HPP
#define FOUR_C_MAT_DAMAGED_SOLID_HPP

#include "4C_config.hpp"

#include "4C_mat_trait_thermo_solid.hpp"

#include <memory>
#include <span>

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  /*!
   * @brief Base of materials that degrade the stress of a wrapped solid material by a scalar damage
   *
   * The wrapped (undamaged) material is evaluated with the full deformation, its stress is degraded
   * by the damage of the derived material. Everything else is passed through to the wrapped
   * material. The material is a thermo-solid material if the wrapped material is one, its thermal
   * coupling terms are degraded like the stress.
   */
  class DamagedSolid : public Trait::ThermoSolid
  {
   public:
    [[nodiscard]] double density() const override { return wrapped_material_->density(); }

    void valid_kinematics(Solid::KinemType kinem) override
    {
      wrapped_material_->valid_kinematics(kinem);
    }

    [[nodiscard]] double strain_energy(const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const EvaluationContext<3>& context, int gp, int eleGID) const override;

    void setup(int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override;

    void post_setup(const Teuchos::ParameterList& params, int eleGID) override;

    void update() override;

    bool uses_extended_update() override { return wrapped_material_->uses_extended_update(); }

    void update(const Core::LinAlg::Tensor<double, 3, 3>& defgrd, int gp,
        const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
        int eleGID) override;

    void reset_step() override;

    void store_history(int timestep) override { wrapped_material_->store_history(timestep); }

    void set_history(int timestep) override { wrapped_material_->set_history(timestep); }

    //! Return the damage of the last evaluation at Gauss point @p gp
    [[nodiscard]] virtual double damage(int gp) const = 0;

    void vis_names(std::map<std::string, int>& names) const override
    {
      wrapped_material_->vis_names(names);
    }

    bool vis_data(const std::string& name, std::vector<double>& data, int numgp) const override
    {
      return wrapped_material_->vis_data(name, data, numgp);
    }

    bool vis_data(
        const std::string& name, std::vector<double>& data, int numgp, int eleId) const override
    {
      return wrapped_material_->vis_data(name, data, numgp, eleId);
    }

    void register_output_data_names(
        std::unordered_map<std::string, int>& names_and_size) const override;

    bool evaluate_output_data(
        const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const override;

    //! @name Thermo-solid interface, only available if the wrapped material is a thermo-solid
    //! material
    //! @{
    void reinit(double temperature, unsigned gp) override;

    Core::LinAlg::SymmetricTensor<double, 3, 3> evaluate_d_stress_d_scalar(
        const Core::LinAlg::Tensor<double, 3, 3>& defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Teuchos::ParameterList& params, const EvaluationContext<3>& context, int gp,
        int eleGID) override;

    [[nodiscard]] std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>>
    evaluate_d_stress_d_scalars(const Core::LinAlg::Tensor<double, 3, 3>& defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Teuchos::ParameterList& params, const EvaluationContext<3>& context, int num_scalars,
        int gp, int eleGID) override;

    void stress_temperature_modulus_and_deriv(Core::LinAlg::SymmetricTensor<double, 3, 3>& stm,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& stm_dT, int gp) override;
    //! @}

   protected:
    DamagedSolid() = default;

    //! Create the wrapped material with material id @p matid
    explicit DamagedSolid(int matid);

    DamagedSolid(const DamagedSolid& other);

    DamagedSolid& operator=(const DamagedSolid& other);

    //! Allocate the damage state of @p numgp Gauss points
    virtual void setup_damage(int numgp) = 0;

    //! Make the damage state of the last evaluation the converged one
    virtual void update_damage() = 0;

    //! Reset the damage state of the last evaluation to the converged one
    virtual void reset_damage() = 0;

    //! Pack the wrapped material, e.g., after the damage state
    void pack_wrapped_material(Core::Communication::PackBuffer& data) const;

    //! Create the wrapped material with material id @p matid and unpack it
    void unpack_wrapped_material(Core::Communication::UnpackBuffer& buffer, int matid);

    [[nodiscard]] So3Material& wrapped_material() const { return *wrapped_material_; }

   private:
    //! Return the wrapped material as thermo-solid material or throw if it is none
    [[nodiscard]] Trait::ThermoSolid& wrapped_thermo_solid_material() const;

    std::shared_ptr<So3Material> wrapped_material_;
  };

  namespace DamageUtils
  {
    //! Extent of an element along @p direction, i.e., the width of a band of elements normal to
    //! it, from the reference coordinates of its nodes with three values per node
    [[nodiscard]] double evaluate_extent(std::span<const double> nodal_coordinates,
        const Core::LinAlg::Tensor<double, 3>& direction);

    struct MaxPrincipalValue
    {
      double value;
      Core::LinAlg::Tensor<double, 3> direction;
    };

    //! Maximum eigenvalue of a symmetric tensor and its direction
    [[nodiscard]] MaxPrincipalValue evaluate_max_principal_value(
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& tensor);
  }  // namespace DamageUtils
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
