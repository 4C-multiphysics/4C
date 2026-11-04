// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_viscoelast_state.hpp"

#include <algorithm>

FOUR_C_NAMESPACE_OPEN

namespace
{
  [[nodiscard]] bool activation_matches(
      const Mat::ViscoElastState::ActiveModels& lhs, const Mat::ViscoElastState::ActiveModels& rhs)
  {
    return lhs.iso_rate == rhs.iso_rate && lhs.generalized_maxwell == rhs.generalized_maxwell &&
           lhs.quasi_linear_generalized_maxwell == rhs.quasi_linear_generalized_maxwell &&
           lhs.fsls == rhs.fsls;
  }


  Mat::ViscoElastState::StressVector make_zero_stress()
  {
    return Mat::ViscoElastState::StressVector(Core::LinAlg::Initialization::zero);
  }


  Mat::ViscoElastState::StressVector make_identity_strain_like()
  {
    Mat::ViscoElastState::StressVector identity(Core::LinAlg::Initialization::zero);
    for (int i = 0; i < 3; ++i) identity(i) = 1.;
    return identity;
  }


  void validate_iso_rate_state(
      const Mat::ViscoElastState::IsoRateState& iso_rate, const char* context)
  {
    const std::size_t gp_count = iso_rate.scg_current_.size();
    FOUR_C_ASSERT_ALWAYS(iso_rate.scg_previous_.size() == gp_count &&
                             iso_rate.modrcg_current_.size() == gp_count &&
                             iso_rate.modrcg_previous_.size() == gp_count,
        "Inconsistent iso-rate visco state sizes while {}: scg_current={}, scg_previous={}, "
        "modrcg_current={}, modrcg_previous={}",
        context, iso_rate.scg_current_.size(), iso_rate.scg_previous_.size(),
        iso_rate.modrcg_current_.size(), iso_rate.modrcg_previous_.size());
  }


  void validate_generalized_maxwell_state(
      const Mat::ViscoElastState::GeneralizedMaxwellState& generalized_maxwell,
      const std::size_t gp_count, const char* context)
  {
    FOUR_C_ASSERT_ALWAYS(
        generalized_maxwell.branch_stress_current_.size() == gp_count &&
            generalized_maxwell.branch_stress_previous_.size() == gp_count &&
            generalized_maxwell.branch_elastic_stress_current_.size() == gp_count &&
            generalized_maxwell.branch_elastic_stress_previous_.size() == gp_count,
        "Inconsistent generalized Maxwell state sizes while {}: branch_stress_current={}, "
        "branch_stress_previous={}, branch_elastic_stress_current={}, "
        "branch_elastic_stress_previous={}, expected={}",
        context, generalized_maxwell.branch_stress_current_.size(),
        generalized_maxwell.branch_stress_previous_.size(),
        generalized_maxwell.branch_elastic_stress_current_.size(),
        generalized_maxwell.branch_elastic_stress_previous_.size(), gp_count);

    std::size_t expected_branch_count = 0;
    for (std::size_t gp = 0; gp < gp_count; ++gp)
    {
      const auto& branch_stress_current = generalized_maxwell.branch_stress_current_.at(gp);
      const auto& branch_stress_previous = generalized_maxwell.branch_stress_previous_.at(gp);
      const auto& branch_elastic_stress_current =
          generalized_maxwell.branch_elastic_stress_current_.at(gp);
      const auto& branch_elastic_stress_previous =
          generalized_maxwell.branch_elastic_stress_previous_.at(gp);

      FOUR_C_ASSERT_ALWAYS(
          branch_stress_current.size() == branch_stress_previous.size() &&
              branch_stress_current.size() == branch_elastic_stress_current.size() &&
              branch_stress_current.size() == branch_elastic_stress_previous.size(),
          "Inconsistent generalized Maxwell branch sizes while {} at GP {}: "
          "branch_stress_current={}, branch_stress_previous={}, "
          "branch_elastic_stress_current={}, branch_elastic_stress_previous={}",
          context, gp, branch_stress_current.size(), branch_stress_previous.size(),
          branch_elastic_stress_current.size(), branch_elastic_stress_previous.size());

      if (gp == 0)
        expected_branch_count = branch_stress_current.size();
      else if (branch_stress_current.size() != expected_branch_count)
        FOUR_C_ASSERT_ALWAYS(branch_stress_current.size() == expected_branch_count,
            "Inconsistent generalized Maxwell branch count while {}: expected {} branches but "
            "GP {} has {}.",
            context, expected_branch_count, gp, branch_stress_current.size());
    }
  }


  void validate_quasi_linear_generalized_maxwell_state(
      const Mat::ViscoElastState::QuasiLinearGeneralizedMaxwellState& quasi_linear,
      const std::size_t gp_count, const char* context)
  {
    FOUR_C_ASSERT_ALWAYS(quasi_linear.branch_stress_current_.size() == gp_count &&
                             quasi_linear.branch_stress_previous_.size() == gp_count &&
                             quasi_linear.branch_elastic_stress_current_.size() == gp_count &&
                             quasi_linear.branch_elastic_stress_previous_.size() == gp_count &&
                             quasi_linear.dashpot_strain_current_.size() == gp_count &&
                             quasi_linear.dashpot_strain_previous_.size() == gp_count &&
                             quasi_linear.dashpot_stress_current_.size() == gp_count &&
                             quasi_linear.dashpot_stress_previous_.size() == gp_count,
        "Inconsistent quasi-linear generalized Maxwell state sizes while {}: "
        "branch_stress_current={}, branch_stress_previous={}, "
        "branch_elastic_stress_current={}, branch_elastic_stress_previous={}, "
        "dashpot_strain_current={}, dashpot_strain_previous={}, dashpot_stress_current={}, "
        "dashpot_stress_previous={}, expected={}",
        context, quasi_linear.branch_stress_current_.size(),
        quasi_linear.branch_stress_previous_.size(),
        quasi_linear.branch_elastic_stress_current_.size(),
        quasi_linear.branch_elastic_stress_previous_.size(),
        quasi_linear.dashpot_strain_current_.size(), quasi_linear.dashpot_strain_previous_.size(),
        quasi_linear.dashpot_stress_current_.size(), quasi_linear.dashpot_stress_previous_.size(),
        gp_count);

    std::size_t expected_branch_count = 0;
    for (std::size_t gp = 0; gp < gp_count; ++gp)
    {
      const auto& branch_stress_current = quasi_linear.branch_stress_current_.at(gp);
      const auto& branch_stress_previous = quasi_linear.branch_stress_previous_.at(gp);
      const auto& branch_elastic_stress_current =
          quasi_linear.branch_elastic_stress_current_.at(gp);
      const auto& branch_elastic_stress_previous =
          quasi_linear.branch_elastic_stress_previous_.at(gp);

      FOUR_C_ASSERT_ALWAYS(
          branch_stress_current.size() == branch_stress_previous.size() &&
              branch_stress_current.size() == branch_elastic_stress_current.size() &&
              branch_stress_current.size() == branch_elastic_stress_previous.size(),
          "Inconsistent quasi-linear generalized Maxwell branch sizes while {} at GP {}: "
          "branch_stress_current={}, branch_stress_previous={}, "
          "branch_elastic_stress_current={}, branch_elastic_stress_previous={}",
          context, gp, branch_stress_current.size(), branch_stress_previous.size(),
          branch_elastic_stress_current.size(), branch_elastic_stress_previous.size());

      if (gp == 0)
        expected_branch_count = branch_stress_current.size();
      else if (branch_stress_current.size() != expected_branch_count)
        FOUR_C_ASSERT_ALWAYS(branch_stress_current.size() == expected_branch_count,
            "Inconsistent quasi-linear generalized Maxwell branch count while {}: expected {} "
            "branches but GP {} has {}.",
            context, expected_branch_count, gp, branch_stress_current.size());
    }
  }


  void validate_fsls_state(const Mat::ViscoElastState::FslsState& fsls_state,
      const std::size_t gp_count, const char* context)
  {
    FOUR_C_ASSERT_ALWAYS(fsls_state.artificial_stress_current_.size() == gp_count &&
                             fsls_state.artificial_stress_previous_history_.size() == gp_count,
        "Inconsistent FSLS state sizes while {}: artificial_stress_current={}, "
        "artificial_stress_previous_history={}, expected={}",
        context, fsls_state.artificial_stress_current_.size(),
        fsls_state.artificial_stress_previous_history_.size(), gp_count);

    if (gp_count == 0) return;

    const std::size_t history_size = fsls_state.artificial_stress_previous_history_.at(0).size();
    for (std::size_t gp = 0; gp < gp_count; ++gp)
    {
      FOUR_C_ASSERT_ALWAYS(
          fsls_state.artificial_stress_previous_history_.at(gp).size() == history_size,
          "Inconsistent FSLS history sizes while {}: expected {} entries but GP {} has {}.",
          context, history_size, gp, fsls_state.artificial_stress_previous_history_.at(gp).size());
    }
  }
}  // namespace


bool Mat::ViscoElastState::initialized() const { return isinitvis_; }


void Mat::ViscoElastState::mark_initialized(const bool initialized) { isinitvis_ = initialized; }


void Mat::ViscoElastState::configure_active_models(const ActiveModels& active_models)
{
  has_iso_rate_state_ = active_models.iso_rate;
  has_generalized_maxwell_state_ = active_models.generalized_maxwell;
  has_quasi_linear_generalized_maxwell_state_ = active_models.quasi_linear_generalized_maxwell;
  has_fsls_state_ = active_models.fsls;
}


Mat::ViscoElastState::ActiveModels Mat::ViscoElastState::configured_active_models() const
{
  return ActiveModels{.iso_rate = has_iso_rate_state_,
      .generalized_maxwell = has_generalized_maxwell_state_,
      .quasi_linear_generalized_maxwell = has_quasi_linear_generalized_maxwell_state_,
      .fsls = has_fsls_state_};
}


void Mat::ViscoElastState::clear_iso_rate_state()
{
  iso_rate_.scg_previous_.clear();
  iso_rate_.modrcg_previous_.clear();
  iso_rate_.scg_current_.clear();
  iso_rate_.modrcg_current_.clear();
}


void Mat::ViscoElastState::clear_generalized_maxwell_state()
{
  generalized_maxwell_.branch_stress_previous_.clear();
  generalized_maxwell_.branch_elastic_stress_previous_.clear();
  generalized_maxwell_.branch_stress_current_.clear();
  generalized_maxwell_.branch_elastic_stress_current_.clear();
}


void Mat::ViscoElastState::clear_quasi_linear_generalized_maxwell_state()
{
  quasi_linear_generalized_maxwell_.branch_stress_previous_.clear();
  quasi_linear_generalized_maxwell_.branch_elastic_stress_previous_.clear();
  quasi_linear_generalized_maxwell_.branch_stress_current_.clear();
  quasi_linear_generalized_maxwell_.branch_elastic_stress_current_.clear();
  quasi_linear_generalized_maxwell_.dashpot_strain_previous_.clear();
  quasi_linear_generalized_maxwell_.dashpot_strain_current_.clear();
  quasi_linear_generalized_maxwell_.dashpot_stress_previous_.clear();
  quasi_linear_generalized_maxwell_.dashpot_stress_current_.clear();
}


void Mat::ViscoElastState::clear_fsls_state()
{
  fsls_.artificial_stress_current_.clear();
  fsls_.artificial_stress_previous_history_.clear();
}


void Mat::ViscoElastState::ensure_initialized(const char* context) const
{
  FOUR_C_ASSERT_ALWAYS(initialized(), "Attempted visco state {} before initialization.", context);
}


void Mat::ViscoElastState::ensure_model_active(
    const bool model_is_active, const char* model_name, const char* context) const
{
  FOUR_C_ASSERT_ALWAYS(model_is_active, "Attempted to {} {} state, but {} model is inactive.",
      context, model_name, model_name);
}


void Mat::ViscoElastState::ensure_gp_in_range(
    const int gp, const char* model_name, const char* context) const
{
  FOUR_C_ASSERT_ALWAYS(gp >= 0 && gp < gp_count_,
      "Invalid GP {} while attempting to {} {} state. Valid range is [0, {}).", gp, context,
      model_name, gp_count_);
}


int Mat::ViscoElastState::serialized_gp_count() const
{
  if (!initialized()) return 0;
  return gp_count_;
}


void Mat::ViscoElastState::serialize_state(
    Core::Communication::PackBuffer& data, const ActiveModels& active_models) const
{
  FOUR_C_ASSERT_ALWAYS(
      !initialized() || activation_matches(configured_active_models(), active_models),
      "Inconsistent visco model activation while serializing state: "
      "configured(iso_rate={}, generalized_maxwell={}, quasi_linear_generalized_maxwell={}, "
      "fsls={}), requested(iso_rate={}, generalized_maxwell={}, "
      "quasi_linear_generalized_maxwell={}, fsls={}).",
      has_iso_rate_state_, has_generalized_maxwell_state_,
      has_quasi_linear_generalized_maxwell_state_, has_fsls_state_, active_models.iso_rate,
      active_models.generalized_maxwell, active_models.quasi_linear_generalized_maxwell,
      active_models.fsls);

  const int gp_count = serialized_gp_count();
  add_to_pack(data, gp_count);

  if (active_models.iso_rate) serialize_iso_rate_state(data, gp_count);
  if (active_models.generalized_maxwell) serialize_generalized_maxwell_state(data, gp_count);
  if (active_models.quasi_linear_generalized_maxwell)
    serialize_quasi_linear_generalized_maxwell_state(data, gp_count);
  if (active_models.fsls) serialize_fsls_state(data, gp_count);
}


void Mat::ViscoElastState::deserialize_state(
    Core::Communication::UnpackBuffer& buffer, const ActiveModels& active_models)
{
  configure_active_models(active_models);

  mark_initialized(true);

  int gp_count = 0;
  extract_from_pack(buffer, gp_count);
  FOUR_C_ASSERT_ALWAYS(
      gp_count >= 0, "Invalid negative gp_count={} while deserializing visco state.", gp_count);

  gp_count_ = gp_count;
  if (gp_count == 0) mark_initialized(false);

  if (active_models.iso_rate)
    deserialize_iso_rate_state(buffer, gp_count);
  else
    clear_iso_rate_state();

  if (active_models.generalized_maxwell)
    deserialize_generalized_maxwell_state(buffer, gp_count);
  else
    clear_generalized_maxwell_state();

  if (active_models.quasi_linear_generalized_maxwell)
    deserialize_quasi_linear_generalized_maxwell_state(buffer, gp_count);
  else
    clear_quasi_linear_generalized_maxwell_state();

  if (active_models.fsls)
    deserialize_fsls_state(buffer, gp_count);
  else
    clear_fsls_state();
}


void Mat::ViscoElastState::initialize_from_setup(const int gp_count,
    const ActiveModels& active_models, const std::size_t generalized_maxwell_branch_count,
    const std::size_t quasi_linear_generalized_maxwell_branch_count)
{
  FOUR_C_ASSERT_ALWAYS(gp_count >= 0, "Invalid setup for visco state: gp_count={}.", gp_count);

  gp_count_ = gp_count;
  configure_active_models(active_models);

  if (active_models.iso_rate)
    initialize_iso_rate_state(gp_count);
  else
    clear_iso_rate_state();

  if (active_models.generalized_maxwell)
  {
    FOUR_C_ASSERT_ALWAYS(generalized_maxwell_branch_count > 0,
        "Invalid setup for generalized Maxwell visco state: branch count must be positive.");
    initialize_generalized_maxwell_state(gp_count, generalized_maxwell_branch_count);
  }
  else
    clear_generalized_maxwell_state();

  if (active_models.quasi_linear_generalized_maxwell)
    initialize_quasi_linear_generalized_maxwell_state(
        gp_count, quasi_linear_generalized_maxwell_branch_count);
  else
    clear_quasi_linear_generalized_maxwell_state();

  if (active_models.fsls)
    initialize_fsls_state(gp_count);
  else
    clear_fsls_state();

  mark_initialized(true);
}


void Mat::ViscoElastState::serialize_iso_rate_state(
    Core::Communication::PackBuffer& data, const int gp_count) const
{
  if (gp_count == 0) return;

  validate_iso_rate_state(iso_rate_, "serializing");
  FOUR_C_ASSERT_ALWAYS(gp_count == static_cast<int>(iso_rate_.scg_previous_.size()),
      "Invalid iso-rate state size for serialization: expected {}, got {}.", gp_count,
      iso_rate_.scg_previous_.size());

  for (int gp = 0; gp < gp_count; ++gp)
  {
    add_to_pack(data, iso_rate_.scg_previous_.at(gp));
    add_to_pack(data, iso_rate_.modrcg_previous_.at(gp));
    add_to_pack(data, iso_rate_.scg_current_.at(gp));
    add_to_pack(data, iso_rate_.modrcg_current_.at(gp));
  }
}


void Mat::ViscoElastState::serialize_generalized_maxwell_state(
    Core::Communication::PackBuffer& data, const int gp_count) const
{
  if (gp_count == 0) return;

  validate_generalized_maxwell_state(generalized_maxwell_, gp_count, "serializing");
  for (int gp = 0; gp < gp_count; ++gp)
  {
    add_to_pack(data, generalized_maxwell_.branch_stress_previous_.at(gp));
    add_to_pack(data, generalized_maxwell_.branch_elastic_stress_previous_.at(gp));
    add_to_pack(data, generalized_maxwell_.branch_stress_current_.at(gp));
    add_to_pack(data, generalized_maxwell_.branch_elastic_stress_current_.at(gp));
  }
}


void Mat::ViscoElastState::serialize_quasi_linear_generalized_maxwell_state(
    Core::Communication::PackBuffer& data, const int gp_count) const
{
  if (gp_count == 0) return;

  validate_quasi_linear_generalized_maxwell_state(
      quasi_linear_generalized_maxwell_, gp_count, "serializing");
  for (int gp = 0; gp < gp_count; ++gp)
  {
    add_to_pack(data, quasi_linear_generalized_maxwell_.branch_stress_previous_.at(gp));
    add_to_pack(data, quasi_linear_generalized_maxwell_.branch_elastic_stress_previous_.at(gp));
    add_to_pack(data, quasi_linear_generalized_maxwell_.branch_stress_current_.at(gp));
    add_to_pack(data, quasi_linear_generalized_maxwell_.branch_elastic_stress_current_.at(gp));
    add_to_pack(data, quasi_linear_generalized_maxwell_.dashpot_strain_previous_.at(gp));
    add_to_pack(data, quasi_linear_generalized_maxwell_.dashpot_strain_current_.at(gp));
    add_to_pack(data, quasi_linear_generalized_maxwell_.dashpot_stress_previous_.at(gp));
    add_to_pack(data, quasi_linear_generalized_maxwell_.dashpot_stress_current_.at(gp));
  }
}


void Mat::ViscoElastState::serialize_fsls_state(
    Core::Communication::PackBuffer& data, const int gp_count) const
{
  validate_fsls_state(fsls_, gp_count, "serializing");

  const bool have_previous_history = !fsls_.artificial_stress_previous_history_.empty();
  add_to_pack(data, have_previous_history);
  FOUR_C_ASSERT_ALWAYS(have_previous_history, "Missing FSLS previous history for serialization.");

  add_to_pack(data, fsls_.artificial_stress_previous_history_.at(0).size());
  for (std::size_t gp = 0; gp < fsls_.artificial_stress_previous_history_.size(); ++gp)
    for (std::size_t step = 0; step < fsls_.artificial_stress_previous_history_.at(gp).size();
        ++step)
      add_to_pack(data, fsls_.artificial_stress_previous_history_.at(gp).at(step));
}


void Mat::ViscoElastState::deserialize_iso_rate_state(
    Core::Communication::UnpackBuffer& buffer, const int gp_count)
{
  initialize_iso_rate_state_for_deserialize(gp_count);
  for (int gp = 0; gp < gp_count; ++gp)
  {
    extract_from_pack(buffer, iso_rate_.scg_previous_.at(gp));
    extract_from_pack(buffer, iso_rate_.modrcg_previous_.at(gp));
    extract_from_pack(buffer, iso_rate_.scg_current_.at(gp));
    extract_from_pack(buffer, iso_rate_.modrcg_current_.at(gp));
  }
}


void Mat::ViscoElastState::deserialize_generalized_maxwell_state(
    Core::Communication::UnpackBuffer& buffer, const int gp_count)
{
  initialize_generalized_maxwell_state_for_deserialize(gp_count);
  for (int gp = 0; gp < gp_count; ++gp)
  {
    extract_from_pack(buffer, generalized_maxwell_.branch_stress_previous_.at(gp));
    extract_from_pack(buffer, generalized_maxwell_.branch_elastic_stress_previous_.at(gp));
    extract_from_pack(buffer, generalized_maxwell_.branch_stress_current_.at(gp));
    extract_from_pack(buffer, generalized_maxwell_.branch_elastic_stress_current_.at(gp));
  }
}


void Mat::ViscoElastState::deserialize_quasi_linear_generalized_maxwell_state(
    Core::Communication::UnpackBuffer& buffer, const int gp_count)
{
  initialize_quasi_linear_generalized_maxwell_state_for_deserialize(gp_count);
  for (int gp = 0; gp < gp_count; ++gp)
  {
    extract_from_pack(buffer, quasi_linear_generalized_maxwell_.branch_stress_previous_.at(gp));
    extract_from_pack(
        buffer, quasi_linear_generalized_maxwell_.branch_elastic_stress_previous_.at(gp));
    extract_from_pack(buffer, quasi_linear_generalized_maxwell_.branch_stress_current_.at(gp));
    extract_from_pack(
        buffer, quasi_linear_generalized_maxwell_.branch_elastic_stress_current_.at(gp));
    extract_from_pack(buffer, quasi_linear_generalized_maxwell_.dashpot_strain_previous_.at(gp));
    extract_from_pack(buffer, quasi_linear_generalized_maxwell_.dashpot_strain_current_.at(gp));
    extract_from_pack(buffer, quasi_linear_generalized_maxwell_.dashpot_stress_previous_.at(gp));
    extract_from_pack(buffer, quasi_linear_generalized_maxwell_.dashpot_stress_current_.at(gp));
  }
}


void Mat::ViscoElastState::deserialize_fsls_state(
    Core::Communication::UnpackBuffer& buffer, const int gp_count)
{
  bool have_previous_history;
  extract_from_pack(buffer, have_previous_history);
  FOUR_C_ASSERT_ALWAYS(have_previous_history, "Missing FSLS previous history in serialized data.");

  std::size_t fsls_previous_history_size;
  extract_from_pack(buffer, fsls_previous_history_size);
  initialize_fsls_state_for_deserialize(gp_count, fsls_previous_history_size);
  for (std::size_t gp = 0; gp < static_cast<std::size_t>(gp_count); ++gp)
    for (std::size_t step = 0; step < fsls_previous_history_size; ++step)
      extract_from_pack(buffer, fsls_.artificial_stress_previous_history_.at(gp).at(step));
}


int Mat::ViscoElastState::gp_count() const { return gp_count_; }


void Mat::ViscoElastState::initialize_iso_rate_state(const int gp_count)
{
  const StressVector identity = make_identity_strain_like();

  iso_rate_.scg_current_.assign(gp_count, identity);
  iso_rate_.scg_previous_.assign(gp_count, identity);
  iso_rate_.modrcg_current_.assign(gp_count, identity);
  iso_rate_.modrcg_previous_.assign(gp_count, identity);
}


void Mat::ViscoElastState::initialize_generalized_maxwell_state(
    const int gp_count, const std::size_t branch_count)
{
  const PointHistory empty_branch_values(branch_count, make_zero_stress());
  generalized_maxwell_.branch_stress_current_.assign(gp_count, empty_branch_values);
  generalized_maxwell_.branch_stress_previous_.assign(gp_count, empty_branch_values);
  generalized_maxwell_.branch_elastic_stress_current_.assign(gp_count, empty_branch_values);
  generalized_maxwell_.branch_elastic_stress_previous_.assign(gp_count, empty_branch_values);
}


void Mat::ViscoElastState::initialize_quasi_linear_generalized_maxwell_state(
    const int gp_count, const std::size_t branch_count)
{
  const PointHistory empty_branch_values(branch_count, make_zero_stress());
  quasi_linear_generalized_maxwell_.branch_stress_current_.assign(gp_count, empty_branch_values);
  quasi_linear_generalized_maxwell_.branch_stress_previous_.assign(gp_count, empty_branch_values);
  quasi_linear_generalized_maxwell_.branch_elastic_stress_current_.assign(
      gp_count, empty_branch_values);
  quasi_linear_generalized_maxwell_.branch_elastic_stress_previous_.assign(
      gp_count, empty_branch_values);
  quasi_linear_generalized_maxwell_.dashpot_strain_current_.assign(gp_count, make_zero_stress());
  quasi_linear_generalized_maxwell_.dashpot_strain_previous_.assign(gp_count, make_zero_stress());
  quasi_linear_generalized_maxwell_.dashpot_stress_current_.assign(gp_count, make_zero_stress());
  quasi_linear_generalized_maxwell_.dashpot_stress_previous_.assign(gp_count, make_zero_stress());
}


void Mat::ViscoElastState::initialize_fsls_state(const int gp_count)
{
  const StressVector zero = make_zero_stress();
  fsls_.artificial_stress_current_.assign(gp_count, zero);
  fsls_.artificial_stress_previous_history_.assign(gp_count, PointHistory(1, zero));
}


void Mat::ViscoElastState::initialize_iso_rate_state_for_deserialize(const int gp_count)
{
  const StressVector zero = make_zero_stress();
  iso_rate_.scg_previous_.assign(gp_count, zero);
  iso_rate_.modrcg_previous_.assign(gp_count, zero);
  iso_rate_.scg_current_.assign(gp_count, zero);
  iso_rate_.modrcg_current_.assign(gp_count, zero);
}


void Mat::ViscoElastState::initialize_generalized_maxwell_state_for_deserialize(const int gp_count)
{
  generalized_maxwell_.branch_stress_previous_.assign(gp_count, PointHistory{});
  generalized_maxwell_.branch_elastic_stress_previous_.assign(gp_count, PointHistory{});
  generalized_maxwell_.branch_stress_current_.assign(gp_count, PointHistory{});
  generalized_maxwell_.branch_elastic_stress_current_.assign(gp_count, PointHistory{});
}


void Mat::ViscoElastState::initialize_quasi_linear_generalized_maxwell_state_for_deserialize(
    const int gp_count)
{
  quasi_linear_generalized_maxwell_.branch_stress_previous_.assign(gp_count, PointHistory{});
  quasi_linear_generalized_maxwell_.branch_elastic_stress_previous_.assign(
      gp_count, PointHistory{});
  quasi_linear_generalized_maxwell_.branch_stress_current_.assign(gp_count, PointHistory{});
  quasi_linear_generalized_maxwell_.branch_elastic_stress_current_.assign(gp_count, PointHistory{});
  quasi_linear_generalized_maxwell_.dashpot_strain_previous_.assign(gp_count, make_zero_stress());
  quasi_linear_generalized_maxwell_.dashpot_strain_current_.assign(gp_count, make_zero_stress());
  quasi_linear_generalized_maxwell_.dashpot_stress_previous_.assign(gp_count, make_zero_stress());
  quasi_linear_generalized_maxwell_.dashpot_stress_current_.assign(gp_count, make_zero_stress());
}


void Mat::ViscoElastState::initialize_fsls_state_for_deserialize(
    const int gp_count, const std::size_t history_size)
{
  const StressVector zero = make_zero_stress();
  fsls_.artificial_stress_current_.assign(gp_count, zero);
  fsls_.artificial_stress_previous_history_.assign(gp_count, PointHistory(history_size, zero));
}


void Mat::ViscoElastState::advance_time_step(const ActiveModels& active_models,
    const unsigned int fsls_max_history_size, const int visco_mat_id)
{
  if (!initialized())
  {
    FOUR_C_ASSERT_ALWAYS(!(active_models.iso_rate || active_models.generalized_maxwell ||
                             active_models.quasi_linear_generalized_maxwell || active_models.fsls),
        "Attempted to advance visco state before initialization for "
        "MAT_ViscoElastHyper (MAT {}).",
        visco_mat_id);

    configure_active_models(active_models);
    return;
  }

  FOUR_C_ASSERT_ALWAYS(activation_matches(configured_active_models(), active_models),
      "Inconsistent visco model activation while advancing state in "
      "MAT_ViscoElastHyper (MAT {}): "
      "configured(iso_rate={}, generalized_maxwell={}, quasi_linear_generalized_maxwell={}, "
      "fsls={}), requested(iso_rate={}, generalized_maxwell={}, "
      "quasi_linear_generalized_maxwell={}, fsls={}).",
      visco_mat_id, has_iso_rate_state_, has_generalized_maxwell_state_,
      has_quasi_linear_generalized_maxwell_state_, has_fsls_state_, active_models.iso_rate,
      active_models.generalized_maxwell, active_models.quasi_linear_generalized_maxwell,
      active_models.fsls);

  configure_active_models(active_models);

  if (active_models.iso_rate) commit_iso_rate_to_previous();
  if (active_models.fsls) append_fsls_current_to_previous_history(fsls_max_history_size);

  const int gp_count = this->gp_count();
  reset_current_state(gp_count, active_models.iso_rate, active_models.fsls);

  if (active_models.generalized_maxwell)
  {
    const std::size_t branch_count = rotate_generalized_maxwell_to_previous(gp_count, visco_mat_id);
    reset_generalized_maxwell_current_state(gp_count, branch_count);
  }

  if (active_models.quasi_linear_generalized_maxwell)
  {
    const std::size_t branch_count =
        rotate_quasi_linear_generalized_maxwell_to_previous(gp_count, visco_mat_id);
    reset_quasi_linear_generalized_maxwell_current_state(gp_count, branch_count);
  }
}


void Mat::ViscoElastState::commit_iso_rate_to_previous()
{
  validate_iso_rate_state(iso_rate_, "advancing time step");
  std::swap(iso_rate_.scg_previous_, iso_rate_.scg_current_);
  std::swap(iso_rate_.modrcg_previous_, iso_rate_.modrcg_current_);
}


void Mat::ViscoElastState::append_fsls_current_to_previous_history(
    const unsigned int max_history_size)
{
  FOUR_C_ASSERT_ALWAYS(max_history_size > 0,
      "Invalid FSLS max history size {} while advancing time step. Expected max history size "
      "to be positive.",
      max_history_size);

  validate_fsls_state(fsls_, fsls_.artificial_stress_current_.size(), "advancing time step");

  FOUR_C_ASSERT_ALWAYS(
      fsls_.artificial_stress_previous_history_.size() == fsls_.artificial_stress_current_.size(),
      "Inconsistent FSLS state sizes while advancing time step: previous history gauss points "
      "{}, current stress entries {}.",
      fsls_.artificial_stress_previous_history_.size(), fsls_.artificial_stress_current_.size());

  for (int gp = 0; gp < static_cast<int>(fsls_.artificial_stress_previous_history_.size()); ++gp)
  {
    auto& gp_history = fsls_.artificial_stress_previous_history_.at(gp);
    gp_history.push_back(fsls_.artificial_stress_current_.at(gp));

    if (gp_history.size() > max_history_size)
    {
      const std::size_t overflow = gp_history.size() - max_history_size;
      gp_history.erase(gp_history.begin(), gp_history.begin() + overflow);
    }
  }
}


void Mat::ViscoElastState::reset_current_state(
    const int gp_count, const bool has_iso_rate_state, const bool has_fsls_state)
{
  const StressVector identity = make_identity_strain_like();
  const StressVector zero = make_zero_stress();

  if (has_iso_rate_state)
  {
    if (iso_rate_.scg_current_.size() != static_cast<std::size_t>(gp_count))
      iso_rate_.scg_current_.assign(gp_count, identity);
    else
      std::fill(iso_rate_.scg_current_.begin(), iso_rate_.scg_current_.end(), identity);

    if (iso_rate_.modrcg_current_.size() != static_cast<std::size_t>(gp_count))
      iso_rate_.modrcg_current_.assign(gp_count, identity);
    else
      std::fill(iso_rate_.modrcg_current_.begin(), iso_rate_.modrcg_current_.end(), identity);
  }

  if (has_fsls_state)
  {
    if (fsls_.artificial_stress_current_.size() != static_cast<std::size_t>(gp_count))
      fsls_.artificial_stress_current_.assign(gp_count, zero);
    else
      std::fill(
          fsls_.artificial_stress_current_.begin(), fsls_.artificial_stress_current_.end(), zero);
  }
}


std::size_t Mat::ViscoElastState::rotate_generalized_maxwell_to_previous(
    const int gp_count, const int visco_mat_id)
{
  std::swap(
      generalized_maxwell_.branch_stress_previous_, generalized_maxwell_.branch_stress_current_);
  std::swap(generalized_maxwell_.branch_elastic_stress_previous_,
      generalized_maxwell_.branch_elastic_stress_current_);

  validate_generalized_maxwell_state(generalized_maxwell_, gp_count, "advancing time step");

  const std::size_t branch_count =
      gp_count > 0 ? generalized_maxwell_.branch_stress_previous_.at(0).size() : 0;
  FOUR_C_ASSERT_ALWAYS(branch_count > 0,
      "Invalid generalized Maxwell state in MAT_ViscoElastHyper (MAT {}) during time-step "
      "advance: branch count is zero.",
      visco_mat_id);

  for (int gp = 0; gp < gp_count; ++gp)
  {
    FOUR_C_ASSERT_ALWAYS(
        generalized_maxwell_.branch_stress_previous_.at(gp).size() == branch_count &&
            generalized_maxwell_.branch_elastic_stress_previous_.at(gp).size() == branch_count,
        "Inconsistent generalized Maxwell branch sizes in MAT_ViscoElastHyper (MAT {}) "
        "during time-step advance at GP {}: expected {}, got branch_stress={} and "
        "branch_elastic_stress={}",
        visco_mat_id, gp, branch_count, generalized_maxwell_.branch_stress_previous_.at(gp).size(),
        generalized_maxwell_.branch_elastic_stress_previous_.at(gp).size());
  }

  return branch_count;
}


void Mat::ViscoElastState::reset_generalized_maxwell_current_state(
    const int gp_count, const std::size_t branch_count)
{
  const StressVector zero = make_zero_stress();

  if (generalized_maxwell_.branch_stress_current_.size() != static_cast<std::size_t>(gp_count))
    generalized_maxwell_.branch_stress_current_.assign(
        gp_count, PointHistory(branch_count, make_zero_stress()));
  if (generalized_maxwell_.branch_elastic_stress_current_.size() !=
      static_cast<std::size_t>(gp_count))
    generalized_maxwell_.branch_elastic_stress_current_.assign(
        gp_count, PointHistory(branch_count, make_zero_stress()));

  for (int gp = 0; gp < gp_count; ++gp)
  {
    auto& branch_stress_current = generalized_maxwell_.branch_stress_current_.at(gp);
    auto& branch_elastic_stress_current =
        generalized_maxwell_.branch_elastic_stress_current_.at(gp);

    if (branch_stress_current.size() != branch_count)
      branch_stress_current.assign(branch_count, zero);
    else
      std::fill(branch_stress_current.begin(), branch_stress_current.end(), zero);

    if (branch_elastic_stress_current.size() != branch_count)
      branch_elastic_stress_current.assign(branch_count, zero);
    else
      std::fill(branch_elastic_stress_current.begin(), branch_elastic_stress_current.end(), zero);
  }
}


std::size_t Mat::ViscoElastState::rotate_quasi_linear_generalized_maxwell_to_previous(
    const int gp_count, const int visco_mat_id)
{
  std::swap(quasi_linear_generalized_maxwell_.branch_stress_previous_,
      quasi_linear_generalized_maxwell_.branch_stress_current_);
  std::swap(quasi_linear_generalized_maxwell_.branch_elastic_stress_previous_,
      quasi_linear_generalized_maxwell_.branch_elastic_stress_current_);
  std::swap(quasi_linear_generalized_maxwell_.dashpot_strain_previous_,
      quasi_linear_generalized_maxwell_.dashpot_strain_current_);
  std::swap(quasi_linear_generalized_maxwell_.dashpot_stress_previous_,
      quasi_linear_generalized_maxwell_.dashpot_stress_current_);

  validate_quasi_linear_generalized_maxwell_state(
      quasi_linear_generalized_maxwell_, gp_count, "advancing time step");

  const std::size_t branch_count =
      gp_count > 0 ? quasi_linear_generalized_maxwell_.branch_stress_previous_.at(0).size() : 0;

  for (int gp = 0; gp < gp_count; ++gp)
  {
    FOUR_C_ASSERT_ALWAYS(
        quasi_linear_generalized_maxwell_.branch_stress_previous_.at(gp).size() == branch_count &&
            quasi_linear_generalized_maxwell_.branch_elastic_stress_previous_.at(gp).size() ==
                branch_count,
        "Inconsistent quasi-linear generalized Maxwell branch sizes in MAT_ViscoElastHyper (MAT "
        "{}) during time-step advance at GP {}: expected {}, got branch_stress={} and "
        "branch_elastic_stress={}",
        visco_mat_id, gp, branch_count,
        quasi_linear_generalized_maxwell_.branch_stress_previous_.at(gp).size(),
        quasi_linear_generalized_maxwell_.branch_elastic_stress_previous_.at(gp).size());
  }

  return branch_count;
}


void Mat::ViscoElastState::reset_quasi_linear_generalized_maxwell_current_state(
    const int gp_count, const std::size_t branch_count)
{
  const StressVector zero = make_zero_stress();

  if (quasi_linear_generalized_maxwell_.branch_stress_current_.size() !=
      static_cast<std::size_t>(gp_count))
    quasi_linear_generalized_maxwell_.branch_stress_current_.assign(
        gp_count, PointHistory(branch_count, make_zero_stress()));
  if (quasi_linear_generalized_maxwell_.branch_elastic_stress_current_.size() !=
      static_cast<std::size_t>(gp_count))
    quasi_linear_generalized_maxwell_.branch_elastic_stress_current_.assign(
        gp_count, PointHistory(branch_count, make_zero_stress()));
  if (quasi_linear_generalized_maxwell_.dashpot_strain_current_.size() !=
      static_cast<std::size_t>(gp_count))
    quasi_linear_generalized_maxwell_.dashpot_strain_current_.assign(gp_count, make_zero_stress());
  if (quasi_linear_generalized_maxwell_.dashpot_stress_current_.size() !=
      static_cast<std::size_t>(gp_count))
    quasi_linear_generalized_maxwell_.dashpot_stress_current_.assign(gp_count, make_zero_stress());

  for (int gp = 0; gp < gp_count; ++gp)
  {
    auto& branch_stress_current = quasi_linear_generalized_maxwell_.branch_stress_current_.at(gp);
    auto& branch_elastic_stress_current =
        quasi_linear_generalized_maxwell_.branch_elastic_stress_current_.at(gp);

    if (branch_stress_current.size() != branch_count)
      branch_stress_current.assign(branch_count, zero);
    else
      std::fill(branch_stress_current.begin(), branch_stress_current.end(), zero);

    if (branch_elastic_stress_current.size() != branch_count)
      branch_elastic_stress_current.assign(branch_count, zero);
    else
      std::fill(branch_elastic_stress_current.begin(), branch_elastic_stress_current.end(), zero);

    quasi_linear_generalized_maxwell_.dashpot_strain_current_.at(gp) = zero;
    quasi_linear_generalized_maxwell_.dashpot_stress_current_.at(gp) = zero;
  }
}


Mat::ViscoElastState::IsoRatePrevPointState Mat::ViscoElastState::iso_rate_prev_point(
    const int gp) const
{
  ensure_initialized("read previous");
  ensure_model_active(has_iso_rate_state_, "iso-rate", "read previous");
  ensure_gp_in_range(gp, "iso-rate", "read previous");

  return {iso_rate_.scg_previous_.at(gp), iso_rate_.modrcg_previous_.at(gp)};
}


void Mat::ViscoElastState::set_iso_rate_current_point(
    const int gp, const StressVector& scg, const StressVector& modrcg)
{
  ensure_initialized("write current");
  ensure_model_active(has_iso_rate_state_, "iso-rate", "write current");
  ensure_gp_in_range(gp, "iso-rate", "write current");

  iso_rate_.scg_current_.at(gp) = scg;
  iso_rate_.modrcg_current_.at(gp) = modrcg;
}


const Mat::ViscoElastState::PointHistory&
Mat::ViscoElastState::generalized_maxwell_prev_branch_elastic_stress(const int gp) const
{
  ensure_initialized("read previous");
  ensure_model_active(has_generalized_maxwell_state_, "generalized Maxwell", "read previous");
  ensure_gp_in_range(gp, "generalized Maxwell", "read previous");

  return generalized_maxwell_.branch_elastic_stress_previous_.at(gp);
}


const Mat::ViscoElastState::PointHistory&
Mat::ViscoElastState::generalized_maxwell_prev_branch_stress(const int gp) const
{
  ensure_initialized("read previous");
  ensure_model_active(has_generalized_maxwell_state_, "generalized Maxwell", "read previous");
  ensure_gp_in_range(gp, "generalized Maxwell", "read previous");

  return generalized_maxwell_.branch_stress_previous_.at(gp);
}


void Mat::ViscoElastState::set_generalized_maxwell_current_point(
    const int gp, const PointHistory& branch_elastic_stress, const PointHistory& branch_stress)
{
  ensure_initialized("write current");
  ensure_model_active(has_generalized_maxwell_state_, "generalized Maxwell", "write current");
  ensure_gp_in_range(gp, "generalized Maxwell", "write current");

  FOUR_C_ASSERT_ALWAYS(branch_elastic_stress.size() == branch_stress.size(),
      "Inconsistent generalized Maxwell current state write at GP {}: "
      "branch_elastic_stress size {} does not match branch_stress size {}.",
      gp, branch_elastic_stress.size(), branch_stress.size());

  const std::size_t expected_branch_count =
      generalized_maxwell_.branch_stress_current_.at(gp).size();
  FOUR_C_ASSERT_ALWAYS(expected_branch_count == branch_stress.size(),
      "Invalid generalized Maxwell current state write at GP {}: expected {} branches but "
      "received {}.",
      gp, expected_branch_count, branch_stress.size());

  generalized_maxwell_.branch_elastic_stress_current_.at(gp) = branch_elastic_stress;
  generalized_maxwell_.branch_stress_current_.at(gp) = branch_stress;
}


const Mat::ViscoElastState::PointHistory&
Mat::ViscoElastState::quasi_linear_generalized_maxwell_prev_branch_elastic_stress(
    const int gp) const
{
  ensure_initialized("read previous");
  ensure_model_active(has_quasi_linear_generalized_maxwell_state_,
      "quasi-linear generalized Maxwell", "read previous");
  ensure_gp_in_range(gp, "quasi-linear generalized Maxwell", "read previous");

  return quasi_linear_generalized_maxwell_.branch_elastic_stress_previous_.at(gp);
}


const Mat::ViscoElastState::PointHistory&
Mat::ViscoElastState::quasi_linear_generalized_maxwell_prev_branch_stress(const int gp) const
{
  ensure_initialized("read previous");
  ensure_model_active(has_quasi_linear_generalized_maxwell_state_,
      "quasi-linear generalized Maxwell", "read previous");
  ensure_gp_in_range(gp, "quasi-linear generalized Maxwell", "read previous");

  return quasi_linear_generalized_maxwell_.branch_stress_previous_.at(gp);
}


const Mat::ViscoElastState::StressVector&
Mat::ViscoElastState::quasi_linear_generalized_maxwell_prev_dashpot_strain(const int gp) const
{
  ensure_initialized("read previous");
  ensure_model_active(has_quasi_linear_generalized_maxwell_state_,
      "quasi-linear generalized Maxwell", "read previous");
  ensure_gp_in_range(gp, "quasi-linear generalized Maxwell", "read previous");

  return quasi_linear_generalized_maxwell_.dashpot_strain_previous_.at(gp);
}


const Mat::ViscoElastState::StressVector&
Mat::ViscoElastState::quasi_linear_generalized_maxwell_prev_dashpot_stress(const int gp) const
{
  ensure_initialized("read previous");
  ensure_model_active(has_quasi_linear_generalized_maxwell_state_,
      "quasi-linear generalized Maxwell", "read previous");
  ensure_gp_in_range(gp, "quasi-linear generalized Maxwell", "read previous");

  return quasi_linear_generalized_maxwell_.dashpot_stress_previous_.at(gp);
}


void Mat::ViscoElastState::set_quasi_linear_generalized_maxwell_current_point(
    const int gp, const PointHistory& branch_elastic_stress, const PointHistory& branch_stress)
{
  ensure_initialized("write current");
  ensure_model_active(has_quasi_linear_generalized_maxwell_state_,
      "quasi-linear generalized Maxwell", "write current");
  ensure_gp_in_range(gp, "quasi-linear generalized Maxwell", "write current");

  FOUR_C_ASSERT_ALWAYS(branch_elastic_stress.size() == branch_stress.size(),
      "Inconsistent quasi-linear generalized Maxwell current state write at GP {}: "
      "branch_elastic_stress size {} does not match branch_stress size {}.",
      gp, branch_elastic_stress.size(), branch_stress.size());

  const std::size_t expected_branch_count =
      quasi_linear_generalized_maxwell_.branch_stress_current_.at(gp).size();
  FOUR_C_ASSERT_ALWAYS(expected_branch_count == branch_stress.size(),
      "Invalid quasi-linear generalized Maxwell current state write at GP {}: expected {} "
      "branches but received {}.",
      gp, expected_branch_count, branch_stress.size());

  quasi_linear_generalized_maxwell_.branch_elastic_stress_current_.at(gp) = branch_elastic_stress;
  quasi_linear_generalized_maxwell_.branch_stress_current_.at(gp) = branch_stress;
}


void Mat::ViscoElastState::set_quasi_linear_generalized_maxwell_current_dashpot_strain(
    const int gp, const StressVector& strain)
{
  ensure_initialized("write current");
  ensure_model_active(has_quasi_linear_generalized_maxwell_state_,
      "quasi-linear generalized Maxwell", "write current");
  ensure_gp_in_range(gp, "quasi-linear generalized Maxwell", "write current");

  quasi_linear_generalized_maxwell_.dashpot_strain_current_.at(gp) = strain;
}


void Mat::ViscoElastState::set_quasi_linear_generalized_maxwell_current_dashpot_stress(
    const int gp, const StressVector& stress)
{
  ensure_initialized("write current");
  ensure_model_active(has_quasi_linear_generalized_maxwell_state_,
      "quasi-linear generalized Maxwell", "write current");
  ensure_gp_in_range(gp, "quasi-linear generalized Maxwell", "write current");

  quasi_linear_generalized_maxwell_.dashpot_stress_current_.at(gp) = stress;
}


const Mat::ViscoElastState::FslsHistory& Mat::ViscoElastState::fsls_previous_history() const
{
  ensure_initialized("read previous");
  ensure_model_active(has_fsls_state_, "FSLS", "read previous");
  validate_fsls_state(fsls_, fsls_.artificial_stress_previous_history_.size(), "reading previous");

  return fsls_.artificial_stress_previous_history_;
}


void Mat::ViscoElastState::set_fsls_current_artificial_stress(
    const int gp, const StressVector& value)
{
  ensure_initialized("write current");
  ensure_model_active(has_fsls_state_, "FSLS", "write current");
  ensure_gp_in_range(gp, "FSLS", "write current");

  fsls_.artificial_stress_current_.at(gp) = value;
}


void Mat::ViscoElastState::clear()
{
  clear_iso_rate_state();
  clear_generalized_maxwell_state();
  clear_quasi_linear_generalized_maxwell_state();
  clear_fsls_state();

  gp_count_ = 0;
  configure_active_models(ActiveModels{});
  isinitvis_ = false;
}

FOUR_C_NAMESPACE_CLOSE
