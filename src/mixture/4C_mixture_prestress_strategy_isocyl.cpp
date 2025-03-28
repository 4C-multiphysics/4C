// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mixture_prestress_strategy_isocyl.hpp"

#include "4C_linalg_fixedsizematrix_generators.hpp"
#include "4C_mat_anisotropy.hpp"
#include "4C_mat_anisotropy_coordinate_system_provider.hpp"
#include "4C_mat_elast_isoneohooke.hpp"
#include "4C_mat_elast_volsussmanbathe.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_mixture_constituent_elasthyper.hpp"
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
      pressure_(matdata.parameters.get<double>("PRESSURE"))
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

void Mixture::IsotropicCylinderPrestressStrategy::setup(
    Mixture::MixtureConstituent& constituent, Teuchos::ParameterList& params, int numgp, int eleGID)
{
  // nothing to do
}

void Mixture::IsotropicCylinderPrestressStrategy::evaluate_prestress(const MixtureRule& mixtureRule,
    const std::shared_ptr<const Mat::CoordinateSystemProvider> cosy,
    Mixture::MixtureConstituent& constituent, Core::LinAlg::Matrix<3, 3>& G,
    Teuchos::ParameterList& params, int gp, int eleGID)
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


  std::shared_ptr<const Mat::CylinderCoordinateSystemProvider> cylinderCosy =
      cosy->get_cylinder_coordinate_system();

  if (!(cylinderCosy))
  {
    FOUR_C_THROW(
        "No cylinder coordinate system is defined but required by the cylinder prestress "
        "strategy!");
  }

  const auto& reference_coordinates = params.get<Core::LinAlg::Matrix<3, 1>>("gp_coords_ref");

  double r = 0;
  for (unsigned i = 0; i < 3; ++i)
  {
    r += cylinderCosy->get_rad()(i) * reference_coordinates(i);
  }

  double initial_constituent_reference_density =
      growth_remodel_rule.get_constituent_initial_reference_mass_density(constituent);

  double Res = 1.0;
  double dResdlamb_pre;
  double lamb_pre = 1. / (params_->circumferential_prestretch_ * params_->axial_prestretch_);
  while (std::abs(Res) > 1.0e-10)
  {
    Res =
        matiso->mue() * initial_constituent_reference_density *
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
        matiso->mue() * initial_constituent_reference_density *
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
        matiso->mue() * initial_constituent_reference_density *
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
  G.multiply_nt(lamb_pre, cylinderCosy->get_rad(), cylinderCosy->get_rad(), 0.0);
  G.multiply_nt(params_->axial_prestretch_, cylinderCosy->get_axi(), cylinderCosy->get_axi(), 1.0);
  G.multiply_nt(
      params_->circumferential_prestretch_, cylinderCosy->get_cir(), cylinderCosy->get_cir(), 1.0);
}

double Mixture::IsotropicCylinderPrestressStrategy::evaluate_mue_frac(MixtureRule& mixtureRule,
    const std::shared_ptr<const Mat::CoordinateSystemProvider> cosy,
    Mixture::MixtureConstituent& constituent, ElastinMembraneEvaluation& membraneEvaluation,
    Teuchos::ParameterList& params, int gp, int eleGID) const
{
  std::shared_ptr<const Mat::CylinderCoordinateSystemProvider> cylinderCosy =
      cosy->get_cylinder_coordinate_system();

  if (!(cylinderCosy))
  {
    FOUR_C_THROW(
        "No cylinder coordinate system is defined but required by the cylinder prestress "
        "strategy!");
  }

  Core::LinAlg::Matrix<3, 3> F = Core::LinAlg::identity_matrix<3>();
  Core::LinAlg::Matrix<6, 1> E_strain(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6, 1> S_stress(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6, 6> cmat(Core::LinAlg::Initialization::zero);


  mixtureRule.evaluate(F, E_strain, params, S_stress, cmat, gp, eleGID);

  Core::LinAlg::Matrix<6, 1> Acir(Core::LinAlg::Initialization::uninitialized);
  // Compute structural tensor
  for (int i = 0; i < 3; ++i) Acir(i) = cylinderCosy->get_cir()(i) * cylinderCosy->get_cir()(i);
  Acir(3) = 2.0 * cylinderCosy->get_cir()(0) * cylinderCosy->get_cir()(1);
  Acir(4) = 2.0 * cylinderCosy->get_cir()(1) * cylinderCosy->get_cir()(2);
  Acir(5) = 2.0 * cylinderCosy->get_cir()(0) * cylinderCosy->get_cir()(2);


  // This prestress strategy is only valid for G&R simulations
  const auto& growth_remodel_rule =
      dynamic_cast<const Mixture::GrowthRemodelMixtureRule&>(mixtureRule);
  double initial_constituent_reference_density =
      growth_remodel_rule.get_constituent_initial_reference_mass_density(constituent);

  Core::LinAlg::Matrix<6, 1> Smembrane(Core::LinAlg::Initialization::uninitialized);
  membraneEvaluation.evaluate_membrane_stress(Smembrane, params, gp, eleGID);
  Smembrane.scale(initial_constituent_reference_density);

  double total_stress = S_stress.dot(Acir);      // stress of all constituents in circular direction
  double membrane_stress = Smembrane.dot(Acir);  // stress of the membrane in circular direction

  // Compute stress as a result of Barlow's formula ("Kesselformel")
  double target_stress = (params_->pressure_ * params_->inner_radius_) /
                         params_->wall_thickness_;  // stress that we need in circular direction

  return (target_stress - (total_stress - membrane_stress)) / membrane_stress;
}

void Mixture::IsotropicCylinderPrestressStrategy::update(
    const std::shared_ptr<const Mat::CoordinateSystemProvider> anisotropy,
    Mixture::MixtureConstituent& constituent, const Core::LinAlg::Matrix<3, 3>& F,
    Core::LinAlg::Matrix<3, 3>& G, Teuchos::ParameterList& params, int gp, int eleGID)
{
}
FOUR_C_NAMESPACE_CLOSE
