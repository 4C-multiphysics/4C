// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_SOLID_SUPERPOSITION_HPP
#define FOUR_C_MAT_SOLID_SUPERPOSITION_HPP

#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  // forward declaration
  class SolidSuperposition;

  namespace PAR
  {
    /// parameter class for the superposition material
    class SolidSuperposition : public Core::Mat::PAR::Parameter
    {
      friend class Mat::SolidSuperposition;

     public:
      /// standard constructor
      explicit SolidSuperposition(const Core::Mat::PAR::Parameter::Data& matdata);

      std::shared_ptr<Core::Mat::Material> create_material() override;

      /// @name material parameters
      /// @{

      /// density of the material
      const double density_;

      /// list of the IDs of the materials to superpose
      const std::vector<int> matids_;

      /// @}
    };
  }  // namespace PAR


  class SolidSuperpositionType : public Core::Communication::ParObjectType
  {
   public:
    [[nodiscard]] std::string name() const override { return "SolidSuperpositionType"; }

    static SolidSuperpositionType& instance() { return instance_; }

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static SolidSuperpositionType instance_;
  };

  /*!
   * \brief Material that superposes multiple 3D solid materials.
   *
   * This material evaluates stress and consistent tangents by combining
   * contributions from multiple 3D solid materials.
   *
   * The second Piola-Kirchhoff stress is computed as a linear sum:
   * \f[
   * S = \sum_{i} S_i
   * \f]
   *
   * and similarly the material tangent:
   * \f[
   * \mathbb{C} = \sum_{i} \mathbb{C}_i
   * \f]
   *
   * Each constituent material is evaluated independently, and their responses are accumulated.
   */
  class SolidSuperposition : public So3Material
  {
   public:
    /// constructor for an empty material object
    SolidSuperposition();

    /// constructor for the material given the material parameters
    explicit SolidSuperposition(Mat::PAR::SolidSuperposition* params);

    [[nodiscard]] int unique_par_object_id() const override
    {
      return SolidSuperpositionType::instance().unique_par_object_id();
    }

    void pack(Core::Communication::PackBuffer& data) const override;

    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    void valid_kinematics(Solid::KinemType kinem) override
    {
      if (!(kinem == Solid::KinemType::nonlinearTotLag))
      {
        FOUR_C_THROW(
            "Element and material kinematics are not compatible. Use nonlinear total lagrangian"
            "kinematics (KINEM nonlinear) in your element definition.");
      }
    }

    [[nodiscard]] Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_superposition;
    }

    [[nodiscard]] std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<SolidSuperposition>(*this);
    }

    [[nodiscard]] Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    void setup(int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override;

    void post_setup(const Teuchos::ParameterList& params, const int eleGID) override;

    void update() override;

    void update(const Core::LinAlg::Tensor<double, 3, 3>& defgrd, int gp,
        const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
        int eleGID) override;

    bool uses_extended_update() override { return true; }

    void evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) override;

    [[nodiscard]] double density() const override { return params_->density_; }

    void register_output_data_names(
        std::unordered_map<std::string, int>& names_and_size) const override;

    bool evaluate_output_data(
        const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const override;

   private:
    /// material parameters, i.e., the density and the list of material IDs to superpose
    Mat::PAR::SolidSuperposition* params_;

    /// list of the references to the materials to superpose
    std::vector<std::shared_ptr<Mat::So3Material>> materials_;
  };

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif