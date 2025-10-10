// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mixture_constituent_elasthyper_damage.hpp"

#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_mat_elast_aniso_structuraltensor_strategy.hpp"
#include "4C_mat_elast_isoneohooke.hpp"
#include "4C_mat_multiplicative_split_defgrad_elasthyper_service.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mixture_elastin_membrane_prestress_strategy.hpp"
#include "4C_utils_function.hpp"

FOUR_C_NAMESPACE_OPEN

// Constructor for the parameter class
Mixture::PAR::MixtureConstituentElastHyperDamage::MixtureConstituentElastHyperDamage(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : MixtureConstituentElastHyperBase(matdata),
      damage_function_id_(matdata.parameters.get<int>("DAMAGE_FUNCT"))
{
  // nothing to do here
}

// Create an instance of Mixture::MixtureConstituentElastHyper from the parameters
std::unique_ptr<Mixture::MixtureConstituent>
Mixture::PAR::MixtureConstituentElastHyperDamage::create_constituent(int id)
{
  return std::unique_ptr<Mixture::MixtureConstituentElastHyperDamage>(
      new Mixture::MixtureConstituentElastHyperDamage(this, id));
}

// Constructor of the constituent holding the material parameters
Mixture::MixtureConstituentElastHyperDamage::MixtureConstituentElastHyperDamage(
    Mixture::PAR::MixtureConstituentElastHyperDamage* params, int id)
    : MixtureConstituentElastHyperBase(params, id), params_(params)
{
  // nothing to do here
}

// Returns the material type
Core::Materials::MaterialType Mixture::MixtureConstituentElastHyperDamage::material_type() const
{
  return Core::Materials::mix_elasthyper_damage;
}

// Pack the constituent
void Mixture::MixtureConstituentElastHyperDamage::pack_constituent(
    Core::Communication::PackBuffer& data) const
{
  MixtureConstituentElastHyperBase::pack_constituent(data);

  add_to_pack(data, current_reference_growth_);
}

// Unpack the constituent
void Mixture::MixtureConstituentElastHyperDamage::unpack_constituent(
    Core::Communication::UnpackBuffer& buffer)
{
  MixtureConstituentElastHyperBase::unpack_constituent(buffer);

  extract_from_pack(buffer, current_reference_growth_);
}

// Reads the element from the input file
void Mixture::MixtureConstituentElastHyperDamage::read_element(int numgp,
    const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  MixtureConstituentElastHyperBase::read_element(numgp, fibers, coord_system);

  current_reference_growth_.resize(numgp, 1.0);
}

// Updates all summands
void Mixture::MixtureConstituentElastHyperDamage::update(
    Core::LinAlg::Tensor<double, 3, 3> const& defgrd, const Teuchos::ParameterList& params,
    const Mat::EvaluationContext& context, const int gp, const int eleGID)
{
  FOUR_C_ASSERT(context.ref_coords,
      "Reference coordinates not set in EvaluationContext, but required for function-based "
      "mixture rule!");
  const auto& reference_coordinates = *context.ref_coords;

  FOUR_C_ASSERT(context.total_time, "Time not given in evaluation context.");
  double totaltime = *context.total_time;

  current_reference_growth_[gp] =
      Global::Problem::instance()
          ->function_by_id<Core::Utils::FunctionOfSpaceTime>(params_->damage_function_id_)
          .evaluate(reference_coordinates.as_span(), totaltime, 0);

  MixtureConstituentElastHyperBase::update(defgrd, params, context, gp, eleGID);
}

double Mixture::MixtureConstituentElastHyperDamage::get_growth_scalar(int gp) const
{
  return current_reference_growth_[gp];
}

void Mixture::MixtureConstituentElastHyperDamage::evaluate(
    const Core::LinAlg::Tensor<double, 3, 3>& F,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& E_strain,
    const Teuchos::ParameterList& params, const Mat::EvaluationContext& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& S_stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  FOUR_C_THROW("This constituent does not support Evaluation without an elastic part.");
}

void Mixture::MixtureConstituentElastHyperDamage::evaluate_elastic_part(
    const Core::LinAlg::Tensor<double, 3, 3>& F, const Core::LinAlg::Tensor<double, 3, 3>& iFextin,
    const Teuchos::ParameterList& params, const Mat::EvaluationContext& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& S_stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  // Compute total inelastic deformation gradient
  static Core::LinAlg::Tensor<double, 3, 3> iFin = iFextin * prestretch_tensor(gp);

  auto stress_view = Core::LinAlg::make_stress_like_voigt_view(S_stress);
  auto cmat_view = Core::LinAlg::make_stress_like_voigt_view(cmat);
  // Evaluate 3D elastic part
  Mat::elast_hyper_evaluate_elastic_part(Core::LinAlg::make_matrix_view(F),
      Core::LinAlg::make_matrix_view(iFin), stress_view, cmat_view, summands(),
      summand_properties(), gp, eleGID);
}
FOUR_C_NAMESPACE_CLOSE
