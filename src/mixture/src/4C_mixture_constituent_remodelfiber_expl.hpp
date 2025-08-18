// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MIXTURE_CONSTITUENT_REMODELFIBER_EXPL_HPP
#define FOUR_C_MIXTURE_CONSTITUENT_REMODELFIBER_EXPL_HPP

#include "4C_config.hpp"

#include "4C_mat_anisotropy_extension_default.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_mixture_constituent.hpp"
#include "4C_mixture_constituent_remodelfiber_material.hpp"
#include "4C_mixture_remodelfiber.hpp"

#include <cmath>

FOUR_C_NAMESPACE_OPEN

namespace Mixture
{
  class MixtureConstituent;
  template <typename T>
  class RemodelFiberMaterial;

  namespace PAR
  {
    class MixtureConstituentRemodelFiberExpl : public Mixture::PAR::MixtureConstituent
    {
     public:
      explicit MixtureConstituentRemodelFiberExpl(const Core::Mat::PAR::Parameter::Data& matdata);
      /// create material instance of matching type with my parameters
      std::unique_ptr<Mixture::MixtureConstituent> create_constituent(int id) override;

      const int fiber_id_;
      const int init_;
      const double gamma_;

      const int fiber_material_id_;
      const Mixture::PAR::RemodelFiberMaterial<double>* fiber_material_;

      const bool enable_growth_;
      const bool enable_basal_mass_production_;
      const double poisson_decay_time_;
      const double growth_constant_;

      const double deposition_stretch_;
      const int deposition_stretch_timefunc_num_;

      const bool inelastic_external_deformation_;
    };
  }  // namespace PAR

  /*!
   * \brief Remodel fiber constituent with an explicit update rule
   */
  class MixtureConstituentRemodelFiberExpl : public Mixture::MixtureConstituent
  {
   public:
    explicit MixtureConstituentRemodelFiberExpl(
        Mixture::PAR::MixtureConstituentRemodelFiberExpl* params, int id);

    [[nodiscard]] Core::Materials::MaterialType material_type() const override;

    void pack_constituent(Core::Communication::PackBuffer& data) const override;

    void unpack_constituent(Core::Communication::UnpackBuffer& buffer) override;

    void register_anisotropy_extensions(Mat::Anisotropy& anisotropy) override;

    void read_element(int numgp, const Core::IO::InputParameterContainer& container) override;

    void setup(const Teuchos::ParameterList& params, int eleGID) override;

    void update(const Core::LinAlg::Tensor<double, 3, 3>& F, const Teuchos::ParameterList& params,
        int gp, int eleGID) override;

    void update_elastic_part(const Core::LinAlg::Tensor<double, 3, 3>& F,
        const Core::LinAlg::Tensor<double, 3, 3>& iFext, const Teuchos::ParameterList& params,
        double dt, int gp, int eleGID) override;

    void evaluate(const Core::LinAlg::Tensor<double, 3, 3>& F,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& E, const Teuchos::ParameterList& params,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& S_stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) override;

    void evaluate_elastic_part(const Core::LinAlg::Tensor<double, 3, 3>& FM,
        const Core::LinAlg::Tensor<double, 3, 3>& iFextin, const Teuchos::ParameterList& params,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& S_stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) override;

    [[nodiscard]] double get_growth_scalar(int gp) const override;

    void register_output_data_names(
        std::unordered_map<std::string, int>& names_and_size) const override;

    bool evaluate_output_data(
        const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const override;

   private:
    [[nodiscard]] double evaluate_lambdaf(
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& C, int gp, int eleGID) const;
    [[nodiscard]] double evaluate_lambda_ext(
        const Core::LinAlg::Tensor<double, 3, 3>& iFext, int gp, int eleGID) const;

    [[nodiscard]] Core::LinAlg::SymmetricTensor<double, 3, 3> evaluate_current_p_k2(
        int gp, int eleGID) const;
    [[nodiscard]] Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> evaluate_current_cmat(
        int gp, int eleGID) const;

    [[nodiscard]] double evaluate_deposition_stretch(double time) const;
    void update_homeostatic_values(const Teuchos::ParameterList& params, int eleGID);

    void initialize();

    /// my material parameters
    Mixture::PAR::MixtureConstituentRemodelFiberExpl* params_;

    /// An instance of the remodel fiber
    std::vector<RemodelFiber<2>> remodel_fiber_;

    /// Handler for anisotropic input
    Mat::DefaultAnisotropyExtension<1> anisotropy_extension_;
  };
}  // namespace Mixture

FOUR_C_NAMESPACE_CLOSE

#endif
