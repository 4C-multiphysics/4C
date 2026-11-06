// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_ductile_damage.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_trait_plastic.hpp"

#include <cmath>

FOUR_C_NAMESPACE_OPEN

namespace
{
  struct StressInvariants
  {
    double equivalent_stress;
    double triaxiality;
  };

  //! von Mises equivalent stress and triaxiality of a Cauchy stress
  StressInvariants evaluate_stress_invariants(
      const Core::LinAlg::SymmetricTensor<double, 3, 3>& cauchy_stress)
  {
    const double mean_stress = Core::LinAlg::trace(cauchy_stress) / 3.0;
    const Core::LinAlg::SymmetricTensor<double, 3, 3> deviator =
        cauchy_stress - mean_stress * Core::LinAlg::TensorGenerators::identity<double, 3, 3>;
    const double equivalent_stress = std::sqrt(1.5 * Core::LinAlg::ddot(deviator, deviator));
    return {.equivalent_stress = equivalent_stress,
        .triaxiality = equivalent_stress > 0.0 ? mean_stress / equivalent_stress : 0.0};
  }

  void add_to_pack(Core::Communication::PackBuffer& data, const auto& states)
  {
    Core::Communication::add_to_pack(data, static_cast<int>(states.size()));
    for (const auto& state : states)
    {
      Core::Communication::add_to_pack(data, state.initiation_indicator);
      Core::Communication::add_to_pack(data, state.initiated);
      Core::Communication::add_to_pack(data, state.characteristic_length);
      Core::Communication::add_to_pack(data, state.plastic_strain);
      Core::Communication::add_to_pack(data, state.equivalent_stress);
      Core::Communication::add_to_pack(data, state.plastic_work);
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
      Core::Communication::extract_from_pack(buffer, state.initiation_indicator);
      Core::Communication::extract_from_pack(buffer, state.initiated);
      Core::Communication::extract_from_pack(buffer, state.characteristic_length);
      Core::Communication::extract_from_pack(buffer, state.plastic_strain);
      Core::Communication::extract_from_pack(buffer, state.equivalent_stress);
      Core::Communication::extract_from_pack(buffer, state.plastic_work);
      Core::Communication::extract_from_pack(buffer, state.damage);
      Core::Communication::extract_from_pack(buffer, state.failed);
    }
  }
}  // namespace

Mat::PAR::DuctileDamage::DuctileDamage(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      matid_(matdata.parameters.get<int>("MATID")),
      d1_(matdata.parameters.get<double>("D1")),
      d2_(matdata.parameters.get<double>("D2")),
      d3_(matdata.parameters.get<double>("D3")),
      fracture_energy_(matdata.parameters.get<double>("FRACTURE_ENERGY")),
      residual_strength_ratio_(matdata.parameters.get<double>("RESIDUAL_STRENGTH_RATIO"))
{
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::DuctileDamage::create_material()
{
  return std::make_shared<Mat::DuctileDamage>(this);
}

Mat::DuctileDamageType Mat::DuctileDamageType::instance_;

Core::Communication::ParObject* Mat::DuctileDamageType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* material = new Mat::DuctileDamage();
  material->unpack(buffer);
  return material;
}

Mat::DuctileDamage::DuctileDamage(PAR::DuctileDamage* params)
    : DamagedSolid(params->matid_), params_(params)
{
  FOUR_C_ASSERT_ALWAYS(dynamic_cast<const Trait::Plastic*>(&wrapped_material()) != nullptr,
      "The material {} wrapped by the ductile damage is no plastic material.", params->matid_);
}

void Mat::DuctileDamage::pack(Core::Communication::PackBuffer& data) const
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

void Mat::DuctileDamage::unpack(Core::Communication::UnpackBuffer& buffer)
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
    params_ = static_cast<PAR::DuctileDamage*>(mat);
  }

  Core::Communication::PotentiallyUnusedBufferScope material_scope{buffer};
  if (params_ != nullptr)  // materials are not accessible in post-process mode
  {
    extract_from_pack(buffer, converged_state_);
    extract_from_pack(buffer, current_state_);
    unpack_wrapped_material(buffer, params_->matid_);
  }
}

std::shared_ptr<Core::Mat::Material> Mat::DuctileDamage::clone() const
{
  return std::make_shared<DuctileDamage>(*this);
}

void Mat::DuctileDamage::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, const int gp, const int eleGID)
{
  Core::LinAlg::SymmetricTensor<double, 3, 3> effective_stress{};
  Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> effective_cmat{};
  wrapped_material().evaluate(
      defgrad, glstrain, params, context, effective_stress, effective_cmat, gp, eleGID);
  const auto& plastic_material = dynamic_cast<const Trait::Plastic&>(wrapped_material());

  const GaussPointState& converged = converged_state_[gp];
  GaussPointState& state = current_state_[gp];
  state = converged;

  // derivative of the damage w.r.t. the Green-Lagrange strain while the damage grows
  Core::LinAlg::SymmetricTensor<double, 3, 3> d_damage_d_glstrain{};

  if (!state.failed)
  {
    FOUR_C_ASSERT_ALWAYS(defgrad != nullptr, "The ductile damage needs the deformation gradient.");
    const auto [equivalent_stress, triaxiality] = evaluate_stress_invariants(
        Core::LinAlg::assume_symmetry(
            *defgrad * effective_stress * Core::LinAlg::transpose(*defgrad)) /
        Core::LinAlg::det(*defgrad));
    state.plastic_strain = plastic_material.equivalent_plastic_strain(gp);
    state.equivalent_stress = equivalent_stress;
    const double plastic_strain_increment = state.plastic_strain - converged.plastic_strain;

    if (!state.initiated)
    {
      const double fracture_strain =
          params_->d1_ + params_->d2_ * std::exp(params_->d3_ * triaxiality);
      FOUR_C_ASSERT_ALWAYS(fracture_strain > 0.0,
          "The fracture strain {} at Gauss point {} of element {} with triaxiality {} is not "
          "positive.",
          fracture_strain, gp, eleGID, triaxiality);
      state.initiation_indicator += plastic_strain_increment / fracture_strain;

      if (state.initiation_indicator >= 1.0)
      {
        state.initiated = true;
        state.characteristic_length =
            DamageUtils::evaluate_extent(context.reference_nodal_coordinates,
                DamageUtils::evaluate_max_principal_value(
                    2.0 * glstrain + Core::LinAlg::TensorGenerators::identity<double, 3, 3>)
                    .direction);
      }
    }
    else if (plastic_strain_increment > 0.0)
    {
      // work of the converged equivalent stress on the plastic displacement of the crack band
      const double length = state.characteristic_length;
      state.plastic_work += converged.equivalent_stress * length * plastic_strain_increment;
      const double damage = 1.0 - std::exp(-state.plastic_work / params_->fracture_energy_);

      if (1.0 - damage <= params_->residual_strength_ratio_)
      {
        state.damage = 1.0 - params_->residual_strength_ratio_;
        state.failed = true;
      }
      else
      {
        state.damage = damage;
        d_damage_d_glstrain = (1.0 - damage) / params_->fracture_energy_ *
                              converged.equivalent_stress * length *
                              plastic_material.d_equivalent_plastic_strain_d_glstrain(gp);
      }
    }
  }

  stress = (1.0 - state.damage) * effective_stress;
  cmat = (1.0 - state.damage) * effective_cmat -
         Core::LinAlg::dyadic(effective_stress, d_damage_d_glstrain);
}

void Mat::DuctileDamage::register_output_data_names(
    std::unordered_map<std::string, int>& names_and_size) const
{
  DamagedSolid::register_output_data_names(names_and_size);
  names_and_size["initiation_indicator"] = 1;
}

bool Mat::DuctileDamage::evaluate_output_data(
    const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const
{
  if (name == "initiation_indicator")
  {
    for (int gp = 0; gp < data.numRows(); ++gp)
      data(gp, 0) = current_state_[gp].initiation_indicator;
    return true;
  }
  return DamagedSolid::evaluate_output_data(name, data);
}

void Mat::DuctileDamage::setup_damage(const int numgp)
{
  converged_state_.assign(numgp, GaussPointState{});
  current_state_ = converged_state_;
}

void Mat::DuctileDamage::update_damage() { converged_state_ = current_state_; }

void Mat::DuctileDamage::reset_damage() { current_state_ = converged_state_; }

FOUR_C_NAMESPACE_CLOSE
