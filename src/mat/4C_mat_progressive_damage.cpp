// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_progressive_damage.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_par_bundle.hpp"

#include <cmath>

FOUR_C_NAMESPACE_OPEN

namespace
{
  struct DamageAndDerivative
  {
    double damage;

    //! ratio of the remaining to the initial strength
    double strength_ratio;

    //! derivative of the damage w.r.t. the history variable
    double derivative;
  };

  //! Exponential softening law of the crack band model
  DamageAndDerivative evaluate_exponential_softening(
      const double max_strain, const double initiation_strain, const double failure_strain)
  {
    const double softening =
        std::exp(-(max_strain - initiation_strain) / (failure_strain - initiation_strain));
    return {.damage = 1.0 - initiation_strain / max_strain * softening,
        .strength_ratio = softening,
        .derivative = initiation_strain / max_strain * softening *
                      (1.0 / max_strain + 1.0 / (failure_strain - initiation_strain))};
  }

  void add_to_pack(Core::Communication::PackBuffer& data, const auto& states)
  {
    Core::Communication::add_to_pack(data, static_cast<int>(states.size()));
    for (const auto& state : states)
    {
      Core::Communication::add_to_pack(data, state.initiated);
      Core::Communication::add_to_pack(data, state.characteristic_length);
      Core::Communication::add_to_pack(data, state.initiation_strain);
      Core::Communication::add_to_pack(data, state.max_strain);
      Core::Communication::add_to_pack(data, state.damage);
      Core::Communication::add_to_pack(data, state.failed);
    }
  }

  void extract_from_pack(Core::Communication::UnpackBuffer& buffer, auto& states)
  {
    int size = 0;
    Core::Communication::extract_from_pack(buffer, size);
    states.resize(size);
    for (auto& state : states)
    {
      Core::Communication::extract_from_pack(buffer, state.initiated);
      Core::Communication::extract_from_pack(buffer, state.characteristic_length);
      Core::Communication::extract_from_pack(buffer, state.initiation_strain);
      Core::Communication::extract_from_pack(buffer, state.max_strain);
      Core::Communication::extract_from_pack(buffer, state.damage);
      Core::Communication::extract_from_pack(buffer, state.failed);
    }
  }
}  // namespace

Mat::PAR::ProgressiveDamage::ProgressiveDamage(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      matid_(matdata.parameters.get<int>("MATID")),
      tensile_strength_(matdata.parameters.get<double>("TENSILE_STRENGTH")),
      fracture_energy_(matdata.parameters.get<double>("FRACTURE_ENERGY")),
      residual_strength_ratio_(matdata.parameters.get<double>("RESIDUAL_STRENGTH_RATIO"))
{
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::ProgressiveDamage::create_material()
{
  return std::make_shared<Mat::ProgressiveDamage>(this);
}

Mat::ProgressiveDamageType Mat::ProgressiveDamageType::instance_;

Core::Communication::ParObject* Mat::ProgressiveDamageType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* material = new Mat::ProgressiveDamage();
  material->unpack(buffer);
  return material;
}

Mat::ProgressiveDamage::ProgressiveDamage(PAR::ProgressiveDamage* params)
    : DamagedSolid(params->matid_), params_(params)
{
}

void Mat::ProgressiveDamage::pack(Core::Communication::PackBuffer& data) const
{
  Core::Communication::add_to_pack(data, unique_par_object_id());

  // matid, -1 in post-process mode
  const int matid = params_ != nullptr ? params_->id() : -1;
  Core::Communication::add_to_pack(data, matid);

  Core::Communication::PotentiallyUnusedBufferScope material_scope{data};
  if (params_ != nullptr)  // materials are not accessible in post-process mode
  {
    add_to_pack(data, converged_state_);
    add_to_pack(data, current_state_);
    pack_wrapped_material(data);
  }
}

void Mat::ProgressiveDamage::unpack(Core::Communication::UnpackBuffer& buffer)
{
  params_ = nullptr;

  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  int matid;
  Core::Communication::extract_from_pack(buffer, matid);
  if (Global::Problem::instance()->materials() != nullptr and
      Global::Problem::instance()->materials()->num() != 0)
  {
    const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
    Core::Mat::PAR::Parameter* mat =
        Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
    FOUR_C_ASSERT_ALWAYS(mat->type() == material_type(),
        "Type of parameter material {} does not fit to calling type {}", mat->type(),
        material_type());
    params_ = static_cast<PAR::ProgressiveDamage*>(mat);
  }

  Core::Communication::PotentiallyUnusedBufferScope material_scope{buffer};
  if (params_ != nullptr)  // materials are not accessible in post-process mode
  {
    extract_from_pack(buffer, converged_state_);
    extract_from_pack(buffer, current_state_);
    unpack_wrapped_material(buffer, params_->matid_);
  }
}

std::shared_ptr<Core::Mat::Material> Mat::ProgressiveDamage::clone() const
{
  return std::make_shared<ProgressiveDamage>(*this);
}

void Mat::ProgressiveDamage::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, const int gp, const int eleGID)
{
  Core::LinAlg::SymmetricTensor<double, 3, 3> effective_stress{};
  Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> effective_cmat{};
  wrapped_material().evaluate(
      defgrad, glstrain, params, context, effective_stress, effective_cmat, gp, eleGID);

  GaussPointState& state = current_state_[gp];
  state = converged_state_[gp];

  // derivative of the damage w.r.t. the Green-Lagrange strain while the damage grows
  double d_damage_d_max_strain = 0.0;
  Core::LinAlg::SymmetricTensor<double, 3, 3> d_max_strain_d_glstrain{};

  if (!state.failed)
  {
    // maximum principal logarithmic strain
    const auto [max_stretch_squared, max_stretch_direction] =
        DamageUtils::evaluate_max_principal_value(
            2.0 * glstrain + Core::LinAlg::TensorGenerators::identity<double, 3, 3>);
    const double max_strain = 0.5 * std::log(max_stretch_squared);

    if (!state.initiated)
    {
      FOUR_C_ASSERT_ALWAYS(
          defgrad != nullptr, "The progressive damage needs the deformation gradient.");
      const Core::LinAlg::SymmetricTensor<double, 3, 3> effective_cauchy_stress =
          Core::LinAlg::assume_symmetry(
              *defgrad * effective_stress * Core::LinAlg::transpose(*defgrad)) /
          Core::LinAlg::det(*defgrad);

      if (DamageUtils::evaluate_max_principal_value(effective_cauchy_stress).value >=
          params_->tensile_strength_)
      {
        FOUR_C_ASSERT_ALWAYS(max_strain > 0.0,
            "Damage initiates at Gauss point {} of element {} without tensile strain, the maximum "
            "principal logarithmic strain is {}.",
            gp, eleGID, max_strain);
        state.initiated = true;
        state.characteristic_length = DamageUtils::evaluate_extent(
            context.reference_nodal_coordinates, max_stretch_direction);
        state.initiation_strain = max_strain;
        state.max_strain = max_strain;
      }
    }

    if (state.initiated)
    {
      if (max_strain > state.max_strain)
      {
        const double length = state.characteristic_length;
        const double failure_strain =
            params_->fracture_energy_ / (params_->tensile_strength_ * length) +
            0.5 * state.initiation_strain;
        FOUR_C_ASSERT_ALWAYS(failure_strain > state.initiation_strain,
            "Element {} is too large for the fracture energy (snap-back): its extent {} along the "
            "maximum principal direction at initiation has to be smaller than {}.",
            eleGID, length,
            2.0 * params_->fracture_energy_ /
                (params_->tensile_strength_ * state.initiation_strain));

        state.max_strain = max_strain;
        const auto [damage, strength_ratio, derivative] =
            evaluate_exponential_softening(max_strain, state.initiation_strain, failure_strain);
        if (strength_ratio <= params_->residual_strength_ratio_)
        {
          // the damage at exactly the residual strength, independent of the overshoot of this step
          const double failure_max_strain =
              state.initiation_strain - (failure_strain - state.initiation_strain) *
                                            std::log(params_->residual_strength_ratio_);
          state.damage = evaluate_exponential_softening(
              failure_max_strain, state.initiation_strain, failure_strain)
                             .damage;
          state.failed = true;
        }
        else
        {
          state.damage = damage;
          d_damage_d_max_strain = derivative;
          d_max_strain_d_glstrain =
              Core::LinAlg::self_dyadic(max_stretch_direction) / max_stretch_squared;
        }
      }
    }
  }

  stress = (1.0 - state.damage) * effective_stress;
  cmat = (1.0 - state.damage) * effective_cmat -
         d_damage_d_max_strain * Core::LinAlg::dyadic(effective_stress, d_max_strain_d_glstrain);
}

void Mat::ProgressiveDamage::setup_damage(const int numgp)
{
  converged_state_.assign(numgp, GaussPointState{});
  current_state_ = converged_state_;
}

void Mat::ProgressiveDamage::update_damage() { converged_state_ = current_state_; }

void Mat::ProgressiveDamage::reset_damage() { current_state_ = converged_state_; }

FOUR_C_NAMESPACE_CLOSE
