// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MIXTURE_CONSTITUENT_SOLIDMATERIAL_HPP
#define FOUR_C_MIXTURE_CONSTITUENT_SOLIDMATERIAL_HPP

#include "4C_config.hpp"

#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_mixture_constituent.hpp"

FOUR_C_NAMESPACE_OPEN


namespace Mixture
{
  class MixtureConstituentSolidMaterial;

  namespace PAR
  {
    class MixtureConstituentSolidMaterial : public Mixture::PAR::MixtureConstituent
    {
     public:
      explicit MixtureConstituentSolidMaterial(const Core::Mat::PAR::Parameter::Data& matdata);
      /// create material instance of matching type with my parameters
      std::unique_ptr<Mixture::MixtureConstituent> create_constituent(int id) override;

      /// @name material parameters
      /// @{
      /// Id of the solid material
      const int matid_;
      /// @}
    };
  }  // namespace PAR

  /*!
   * \brief Constituent for any solid material
   *
   * This constituent represents any solid material from the material toolbox. It has to
   * be paired with the Mat::Mixture and a Mixture::MixtureRule.
   */
  class MixtureConstituentSolidMaterial : public Mixture::MixtureConstituent
  {
   public:
    /// Constructor for the material given the material parameters
    explicit MixtureConstituentSolidMaterial(
        Mixture::PAR::MixtureConstituentSolidMaterial* params, int id);

    void pack_constituent(Core::Communication::PackBuffer& data) const override;

    void unpack_constituent(Core::Communication::UnpackBuffer& buffer) override;

    Core::Materials::MaterialType material_type() const override;

    void read_element(int numgp, const Core::IO::InputParameterContainer& container) override;

    void update(Core::LinAlg::Tensor<double, 3, 3> const& defgrd,
        const Teuchos::ParameterList& params, const int gp, const int eleGID) override;

    void update() override;

    void evaluate(const Core::LinAlg::Tensor<double, 3, 3>& F,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& E_strain,
        const Teuchos::ParameterList& params, Core::LinAlg::SymmetricTensor<double, 3, 3>& S_stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) override;

    void register_output_data_names(
        std::unordered_map<std::string, int>& names_and_size) const override;

    bool evaluate_output_data(
        const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const override;

   private:
    /// my material parameters
    Mixture::PAR::MixtureConstituentSolidMaterial* params_;

    // reference to the so3 material
    std::shared_ptr<Mat::So3Material> material_;
  };

}  // namespace Mixture

FOUR_C_NAMESPACE_CLOSE

#endif