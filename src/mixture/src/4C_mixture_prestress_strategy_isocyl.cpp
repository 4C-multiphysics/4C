// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mixture_prestress_strategy_isocyl.hpp"

#include "4C_linalg_fixedsizematrix_generators.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_anisotropy_coordinate_system_provider.hpp"
#include "4C_mat_anisotropy_cylinder_coordinate_system_provider.hpp"
#include "4C_mat_elast_isoneohooke.hpp"
#include "4C_mat_elast_volsussmanbathe.hpp"
#include "4C_mat_service.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_mixture_constituent_elasthyperbase.hpp"
#include "4C_mixture_rule.hpp"
#include "4C_mixture_rule_growthremodel.hpp"

FOUR_C_NAMESPACE_OPEN

Mixture::PAR::IsotropicCylinderPrestressStrategy::IsotropicCylinderPrestressStrategy(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : PrestressStrategy(matdata),
      inner_radius_(matdata.parameters.get<double>("INNER_RADIUS")),
      wall_thickness_(matdata.parameters.get<double>("WALL_THICKNESS")),
      axial_prestretch_(matdata.parameters.get<double>("AXIAL_PRESTRETCH")),
      circumferential_prestretch_(matdata.parameters.get<double>("CIRCUMFERENTIAL_PRESTRETCH")),
      pressure_(matdata.parameters.get<double>("PRESSURE")),
      radial(
          matdata.parameters.get<Core::IO::InterpolatedInputField<Core::LinAlg::Tensor<double, 3>,
              Mat::FiberInterpolation>>("RADIAL")),
      axial(matdata.parameters.get<Core::IO::InterpolatedInputField<Core::LinAlg::Tensor<double, 3>,
              Mat::FiberInterpolation>>("AXIAL")),
      circumferential(
          matdata.parameters.get<Core::IO::InterpolatedInputField<Core::LinAlg::Tensor<double, 3>,
              Mat::FiberInterpolation>>("CIRCUMFERENTIAL"))
{
}

std::unique_ptr<Mixture::PrestressStrategy>
Mixture::PAR::IsotropicCylinderPrestressStrategy::create_prestress_strategy()
{
  std::unique_ptr<Mixture::PrestressStrategy> prestressStrategy(
      new Mixture::IsotropicCylinderPrestressStrategy(this));
  return prestressStrategy;
}

Mixture::IsotropicCylinderPrestressStrategy::IsotropicCylinderPrestressStrategy(
    Mixture::PAR::IsotropicCylinderPrestressStrategy* params)
    : PrestressStrategy(params), params_(params)
{
}

void Mixture::IsotropicCylinderPrestressStrategy::setup(Mixture::MixtureConstituent& constituent,
    const Teuchos::ParameterList& params, int numgp, int eleGID)
{
  // nothing to do
}

void Mixture::IsotropicCylinderPrestressStrategy::evaluate_prestress(const MixtureRule& mixtureRule,
    const std::shared_ptr<const Mat::CoordinateSystemProvider> cosy,
    Mixture::MixtureConstituent& constituent, Core::LinAlg::SymmetricTensor<double, 3, 3>& G,
    const Teuchos::ParameterList& params, const Mat::EvaluationContext& context, int gp, int eleGID)
{
  // We evaluate the stress in the reference configuration with a prestretch. Hence, the
  // deformation gradient is the identity matrix and the inverse inelastic deformation gradient is
  // the prestretch. This results that the 2. Piola-Kirchhoff stress tensor is the same as the
  // Cauchy-stress tensor.

  // Currently, this prestressing technique implements only a certain material (IsoNeoHooke with
  // Sussman-Bathe penalty)

  auto& elhyper = dynamic_cast<Mixture::MixtureConstituentElastHyperBase&>(constituent);

  if (elhyper.summands().size() != 2)
  {
    FOUR_C_THROW(
        "Currently, the prestressing technique is only implemented for an ElastHyper constituent "
        "using an IsoNeoHooke summand with Sussman-Bathe penalty (Hence, exactly 2 summands are "
        "needed)");
  }

  // Let's assume that for simplicity, the first index is the IsoNeoHooke material and the second
  // index is the Sussman-Bathe penalty parameter
  auto matiso = std::dynamic_pointer_cast<Mat::Elastic::IsoNeoHooke>(elhyper.summands()[0]);
  auto matvol = std::dynamic_pointer_cast<Mat::Elastic::VolSussmanBathe>(elhyper.summands()[1]);

  if (!(matiso))
  {
    FOUR_C_THROW(
        "The first summand of the constituent needs to be an IsoNeoHooke material law. This is a "
        "requirement from the prestressing technique.");
  }

  if (!(matvol))
  {
    FOUR_C_THROW(
        "The second summand of the constituent needs to be a Sussman-Bathe penalty term. This is a "
        "requirement from the prestressing technique.");
  }

  // This prestress strategy is only valid for G&R simulations
  const auto& growth_remodel_rule =
      dynamic_cast<const Mixture::GrowthRemodelMixtureRule&>(mixtureRule);

  FOUR_C_ASSERT(context.ref_coords,
      "Reference coordinates not set in EvaluationContext, but required for function-based "
      "mixture rule!");
  const auto& reference_coordinates = *context.ref_coords;

  double r = reference_coordinates * params_->radial.interpolate(eleGID, context.xi->as_span());

  double initial_constituent_reference_density =
      growth_remodel_rule.get_constituent_initial_reference_mass_density(constituent);

  double Res = 1.0;
  double dResdlamb_pre;
  double lamb_pre = 1. / (params_->circumferential_prestretch_ * params_->axial_prestretch_);
  const double mue = matiso->mue(eleGID);
  while (std::abs(Res) > 1.0e-10)
  {
    Res =
        mue * initial_constituent_reference_density *
            std::pow(params_->circumferential_prestretch_ * params_->axial_prestretch_ * lamb_pre,
                -4. / 3.) *  // TODO: When deriving these equations by hand, I get -2.0 / 3.0. To be
                             //  compatible with the old implementation I decided for now to keep
                             //  this here. This has to be verified later.
            (lamb_pre * lamb_pre -
                (1. / 3.) *
                    (params_->circumferential_prestretch_ * params_->circumferential_prestretch_ +
                        params_->axial_prestretch_ * params_->axial_prestretch_ +
                        lamb_pre * lamb_pre)) +
        matvol->kappa() * initial_constituent_reference_density *
            ((params_->circumferential_prestretch_ * params_->axial_prestretch_ * lamb_pre) *
                    (params_->circumferential_prestretch_ * params_->axial_prestretch_ * lamb_pre) -
                (params_->circumferential_prestretch_ * params_->axial_prestretch_ * lamb_pre)) +
        ((1.0 - (r - params_->inner_radius_) / params_->wall_thickness_) * params_->pressure_);

    dResdlamb_pre =
        mue * initial_constituent_reference_density *
            (-(4. / 3.) *
                std::pow(
                    params_->circumferential_prestretch_ * params_->axial_prestretch_ * lamb_pre,
                    -7. / 3.) *
                params_->circumferential_prestretch_ * params_->axial_prestretch_) *
            (lamb_pre * lamb_pre -
                (1. / 3.) *
                    (params_->circumferential_prestretch_ * params_->circumferential_prestretch_ +
                        params_->axial_prestretch_ * params_->axial_prestretch_ +
                        lamb_pre * lamb_pre)) +
        mue * initial_constituent_reference_density *
            std::pow(params_->circumferential_prestretch_ * params_->circumferential_prestretch_ *
                         params_->axial_prestretch_ * params_->axial_prestretch_ * lamb_pre *
                         lamb_pre,
                -2. / 3.) *
            (2.0 * lamb_pre - (1. / 3.) * (2.0 * lamb_pre)) +
        matvol->kappa() * initial_constituent_reference_density *
            (2.0 * (params_->circumferential_prestretch_ * params_->axial_prestretch_ * lamb_pre) *
                    params_->circumferential_prestretch_ * params_->axial_prestretch_ -
                params_->circumferential_prestretch_ * params_->axial_prestretch_);

    lamb_pre = lamb_pre + (-Res / dResdlamb_pre);
  }

  // Build prestretch tensor
  G = lamb_pre *
      Core::LinAlg::self_dyadic<2>(params_->radial.interpolate(eleGID, context.xi->as_span()));
  G += params_->axial_prestretch_ *
       Core::LinAlg::self_dyadic<2>(params_->axial.interpolate(eleGID, context.xi->as_span()));
  G += params_->circumferential_prestretch_ *
       Core::LinAlg::self_dyadic<2>(
           params_->circumferential.interpolate(eleGID, context.xi->as_span()));
}

double Mixture::IsotropicCylinderPrestressStrategy::evaluate_mue_frac(MixtureRule& mixtureRule,
    const std::shared_ptr<const Mat::CoordinateSystemProvider> cosy,
    Mixture::MixtureConstituent& constituent, ElastinMembraneEvaluation& membraneEvaluation,
    const Teuchos::ParameterList& params, const Mat::EvaluationContext& context, int gp,
    int eleGID) const
{
  Core::LinAlg::Tensor<double, 3, 3> F =
      Core::LinAlg::get_full(Core::LinAlg::TensorGenerators::identity<double, 3, 3>);
  Core::LinAlg::SymmetricTensor<double, 3, 3> E_strain{};
  Core::LinAlg::SymmetricTensor<double, 3, 3> S_stress{};
  Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> cmat{};


  mixtureRule.evaluate(F, E_strain, params, context, S_stress, cmat, gp, eleGID);

  Core::LinAlg::SymmetricTensor<double, 3, 3> Acir = Core::LinAlg::self_dyadic(
      params_->circumferential.interpolate(eleGID, context.xi->as_span()));

  // This prestress strategy is only valid for G&R simulations
  const auto& growth_remodel_rule =
      dynamic_cast<const Mixture::GrowthRemodelMixtureRule&>(mixtureRule);
  double initial_constituent_reference_density =
      growth_remodel_rule.get_constituent_initial_reference_mass_density(constituent);

  Core::LinAlg::SymmetricTensor<double, 3, 3> Smembrane;
  membraneEvaluation.evaluate_membrane_stress(Smembrane, params, context, gp, eleGID);
  Smembrane *= initial_constituent_reference_density;

  double total_stress =
      Core::LinAlg::ddot(S_stress, Acir);  // stress of all constituents in circular direction
  double membrane_stress =
      Core::LinAlg::ddot(Smembrane, Acir);  // stress of the membrane in circular direction

  // Compute stress as a result of Barlow's formula ("Kesselformel")
  double target_stress = (params_->pressure_ * params_->inner_radius_) /
                         params_->wall_thickness_;  // stress that we need in circular direction

  return (target_stress - (total_stress - membrane_stress)) / membrane_stress;
}

void Mixture::IsotropicCylinderPrestressStrategy::update(
    const std::shared_ptr<const Mat::CoordinateSystemProvider> anisotropy,
    Mixture::MixtureConstituent& constituent, const Core::LinAlg::Tensor<double, 3, 3>& F,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& G, const Teuchos::ParameterList& params,
    const Mat::EvaluationContext& context, int gp, int eleGID)
{
}
FOUR_C_NAMESPACE_CLOSE
