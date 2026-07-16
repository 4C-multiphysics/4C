// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_plasticmohrcoulomb.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_symmetric_tensor_eigen.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_linalg_utils_densematrix_inverse.hpp"
#include "4C_mat_par_bundle.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

FOUR_C_NAMESPACE_OPEN

namespace
{
  /**
   * Result of the material-local active-set return mapping in ordered principal-stress space.
   *
   * This type and the following helper functions remain in the anonymous namespace because they
   * implement details of PlasticMohrCoulomb rather than reusable material or linear-algebra
   * operations.
   */
  struct ActiveSetResult
  {
    bool valid = false;
    bool numerical_failure = false;
    Core::LinAlg::Tensor<double, 3> stress{};
    Core::LinAlg::Tensor<double, 3> plastic_strain_increment{};
    Core::LinAlg::Tensor<double, 3, 3> tangent{};
    double accumulated_plastic_strain = 0.0;
  };

  double sum_components(const Core::LinAlg::Tensor<double, 3>& vector)
  {
    return vector(0) + vector(1) + vector(2);
  }

  Core::LinAlg::Tensor<double, 3> deviator(const Core::LinAlg::Tensor<double, 3>& vector)
  {
    const double mean = sum_components(vector) / 3.0;
    return Core::LinAlg::Tensor<double, 3>{{vector(0) - mean, vector(1) - mean, vector(2) - mean}};
  }

  double equivalent_plastic_strain(const Core::LinAlg::Tensor<double, 3>& plastic_strain_increment)
  {
    const auto deviatoric_increment = deviator(plastic_strain_increment);
    return std::sqrt(2.0 / 3.0) * Core::LinAlg::norm2(deviatoric_increment);
  }

  Core::LinAlg::Tensor<double, 3> yield_normal(
      int positive_index, int negative_index, double factor)
  {
    Core::LinAlg::Tensor<double, 3> normal{};
    normal(positive_index) = factor;
    normal(negative_index) = -1.0;
    return normal;
  }

  double max_yield_function(
      const Core::LinAlg::Tensor<double, 3>& stress, double friction_stress_ratio, double q)
  {
    double maximum = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        if (i != j) maximum = std::max(maximum, friction_stress_ratio * stress(i) - stress(j) - q);
    return maximum;
  }

  double cohesion(const Mat::PAR::PlasticMohrCoulomb& parameters, double plastic_strain)
  {
    double value = parameters.cohesion_ + parameters.linear_hardening_ * plastic_strain;
    if (parameters.saturation_hardening_ > 0.0)
      value += (parameters.saturation_hardening_ - parameters.cohesion_) *
               (1.0 - std::exp(-parameters.hardening_exponent_ * plastic_strain));
    return value;
  }

  double cohesion_derivative(const Mat::PAR::PlasticMohrCoulomb& parameters, double plastic_strain)
  {
    double value = parameters.linear_hardening_;
    if (parameters.saturation_hardening_ > 0.0)
      value += (parameters.saturation_hardening_ - parameters.cohesion_) *
               parameters.hardening_exponent_ *
               std::exp(-parameters.hardening_exponent_ * plastic_strain);
    return value;
  }

  Core::LinAlg::Tensor<double, 3, 3> elastic_principal_tangent(
      double shear_modulus, double lame_parameter)
  {
    Core::LinAlg::Tensor<double, 3, 3> tangent{};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        tangent(i, j) = lame_parameter + (i == j ? 2.0 * shear_modulus : 0.0);
    return tangent;
  }

  bool belongs_to_apex_flow_cone(const Core::LinAlg::Tensor<double, 3>& plastic_strain_increment,
      double dilatancy_stress_ratio, double tolerance)
  {
    if (Core::LinAlg::norm2(plastic_strain_increment) <= tolerance) return true;

    std::array<Core::LinAlg::Tensor<double, 3>, 6> flow_normals{};
    int normal_index = 0;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        if (i != j) flow_normals[normal_index++] = yield_normal(i, j, dilatancy_stress_ratio);

    for (int first = 0; first < 4; ++first)
      for (int second = first + 1; second < 5; ++second)
        for (int third = second + 1; third < 6; ++third)
        {
          Core::LinAlg::Tensor<double, 3, 3> flow_basis{};
          for (int row = 0; row < 3; ++row)
          {
            flow_basis(row, 0) = flow_normals[first](row);
            flow_basis(row, 1) = flow_normals[second](row);
            flow_basis(row, 2) = flow_normals[third](row);
          }
          const double determinant = Core::LinAlg::det(flow_basis);
          if (std::abs(determinant) <= std::numeric_limits<double>::epsilon()) continue;

          const auto determinant_with_column = [&](int column)
          {
            auto matrix = flow_basis;
            for (int row = 0; row < 3; ++row) matrix(row, column) = plastic_strain_increment(row);
            return Core::LinAlg::det(matrix);
          };

          const Core::LinAlg::Tensor<double, 3> multipliers{
              {determinant_with_column(0) / determinant, determinant_with_column(1) / determinant,
                  determinant_with_column(2) / determinant}};
          if (multipliers(0) < -tolerance || multipliers(1) < -tolerance ||
              multipliers(2) < -tolerance)
            continue;

          const auto reconstructed = multipliers(0) * flow_normals[first] +
                                     multipliers(1) * flow_normals[second] +
                                     multipliers(2) * flow_normals[third];
          const auto error = reconstructed - plastic_strain_increment;
          if (Core::LinAlg::norm2(error) <=
              tolerance * std::max(1.0, Core::LinAlg::norm2(plastic_strain_increment)))
            return true;
        }
    return false;
  }

  ActiveSetResult return_to_active_set(const Core::LinAlg::Tensor<double, 3>& trial_stress,
      double accumulated_plastic_strain_last,
      const Core::LinAlg::Tensor<double, 3, 3>& elastic_tangent, double friction_stress_ratio,
      double dilatancy_stress_ratio, const std::array<std::pair<int, int>, 2>& planes,
      int number_of_planes, const Mat::PAR::PlasticMohrCoulomb& parameters)
  {
    std::array<Core::LinAlg::Tensor<double, 3>, 2> yield_normals{};
    std::array<Core::LinAlg::Tensor<double, 3>, 2> flow_normals{};
    std::array<Core::LinAlg::Tensor<double, 3>, 2> elastic_yield_normals{};
    std::array<Core::LinAlg::Tensor<double, 3>, 2> elastic_flow_normals{};
    for (int alpha = 0; alpha < number_of_planes; ++alpha)
    {
      yield_normals[alpha] =
          yield_normal(planes[alpha].first, planes[alpha].second, friction_stress_ratio);
      flow_normals[alpha] =
          yield_normal(planes[alpha].first, planes[alpha].second, dilatancy_stress_ratio);
      elastic_yield_normals[alpha] = elastic_tangent * yield_normals[alpha];
      elastic_flow_normals[alpha] = elastic_tangent * flow_normals[alpha];
    }

    std::array<std::array<double, 2>, 2> active_matrix{};
    for (int alpha = 0; alpha < number_of_planes; ++alpha)
      for (int beta = 0; beta < number_of_planes; ++beta)
        active_matrix[alpha][beta] =
            Core::LinAlg::dot(yield_normals[alpha], elastic_flow_normals[beta]);

    const double determinant =
        number_of_planes == 1
            ? active_matrix[0][0]
            : active_matrix[0][0] * active_matrix[1][1] - active_matrix[0][1] * active_matrix[1][0];
    if (std::abs(determinant) <= std::numeric_limits<double>::epsilon())
      return {.numerical_failure = true};

    auto solve_active_matrix = [&](const std::array<double, 2>& right_hand_side)
    {
      std::array<double, 2> result{};
      if (number_of_planes == 1)
      {
        result[0] = right_hand_side[0] / active_matrix[0][0];
      }
      else
      {
        result[0] =
            (right_hand_side[0] * active_matrix[1][1] - active_matrix[0][1] * right_hand_side[1]) /
            determinant;
        result[1] =
            (active_matrix[0][0] * right_hand_side[1] - right_hand_side[0] * active_matrix[1][0]) /
            determinant;
      }
      return result;
    };

    const double q_factor = 2.0 * std::sqrt(friction_stress_ratio);
    auto state_at_accumulated_strain = [&](double accumulated_plastic_strain)
    {
      const double q = q_factor * cohesion(parameters, accumulated_plastic_strain);
      std::array<double, 2> right_hand_side{};
      for (int alpha = 0; alpha < number_of_planes; ++alpha)
        right_hand_side[alpha] = Core::LinAlg::dot(yield_normals[alpha], trial_stress) - q;
      const std::array<double, 2> plastic_multipliers = solve_active_matrix(right_hand_side);

      Core::LinAlg::Tensor<double, 3> plastic_strain_increment{};
      for (int alpha = 0; alpha < number_of_planes; ++alpha)
        plastic_strain_increment += plastic_multipliers[alpha] * flow_normals[alpha];

      return std::make_tuple(plastic_multipliers, plastic_strain_increment,
          accumulated_plastic_strain - accumulated_plastic_strain_last -
              equivalent_plastic_strain(plastic_strain_increment));
    };

    const auto [multipliers_at_lower, plastic_increment_at_lower, residual_at_lower] =
        state_at_accumulated_strain(accumulated_plastic_strain_last);
    if (residual_at_lower > parameters.tolerance_) return {};

    double lower = accumulated_plastic_strain_last;
    double upper = lower + std::max(2.0 * equivalent_plastic_strain(plastic_increment_at_lower),
                               10.0 * parameters.tolerance_);
    double residual_upper = std::get<2>(state_at_accumulated_strain(upper));
    for (int iteration = 0; residual_upper < 0.0 && iteration < parameters.max_iterations_;
        ++iteration)
    {
      upper = lower + 2.0 * (upper - lower);
      residual_upper = std::get<2>(state_at_accumulated_strain(upper));
    }
    if (residual_upper < 0.0) return {.numerical_failure = true};

    double accumulated_plastic_strain =
        std::clamp(lower + equivalent_plastic_strain(plastic_increment_at_lower), lower, upper);
    for (int iteration = 0; iteration < parameters.max_iterations_; ++iteration)
    {
      const auto [multipliers, plastic_strain_increment, residual] =
          state_at_accumulated_strain(accumulated_plastic_strain);
      if (std::abs(residual) <= parameters.tolerance_) break;

      if (residual > 0.0)
        upper = accumulated_plastic_strain;
      else
        lower = accumulated_plastic_strain;

      const double plastic_increment_norm = equivalent_plastic_strain(plastic_strain_increment);
      double residual_derivative = 1.0;
      if (plastic_increment_norm > parameters.tolerance_)
      {
        const auto plastic_increment_deviator = deviator(plastic_strain_increment);
        const std::array<double, 2> inverse_times_ones =
            solve_active_matrix({1.0, number_of_planes == 2 ? 1.0 : 0.0});
        Core::LinAlg::Tensor<double, 3> flow_correction{};
        for (int alpha = 0; alpha < number_of_planes; ++alpha)
          flow_correction += inverse_times_ones[alpha] * flow_normals[alpha];
        const double q_derivative =
            q_factor * cohesion_derivative(parameters, accumulated_plastic_strain);
        residual_derivative +=
            q_derivative * 2.0 / 3.0 *
            Core::LinAlg::dot(plastic_increment_deviator, deviator(flow_correction)) /
            plastic_increment_norm;
      }

      double candidate = accumulated_plastic_strain - residual / residual_derivative;
      if (!(candidate > lower && candidate < upper) || !std::isfinite(candidate))
        candidate = 0.5 * (lower + upper);
      accumulated_plastic_strain = candidate;
    }

    const auto [plastic_multipliers, plastic_strain_increment, residual] =
        state_at_accumulated_strain(accumulated_plastic_strain);
    if (std::abs(residual) > 10.0 * parameters.tolerance_) return {.numerical_failure = true};
    for (int alpha = 0; alpha < number_of_planes; ++alpha)
      if (plastic_multipliers[alpha] < -10.0 * parameters.tolerance_) return {};

    const auto updated_stress = trial_stress - elastic_tangent * plastic_strain_increment;
    const double q = q_factor * cohesion(parameters, accumulated_plastic_strain);
    if (max_yield_function(updated_stress, friction_stress_ratio, q) >
        100.0 * parameters.tolerance_)
      return {};
    if (updated_stress(0) + parameters.tolerance_ < updated_stress(1) ||
        updated_stress(1) + parameters.tolerance_ < updated_stress(2))
      return {};

    Core::LinAlg::Tensor<double, 3, 3> algorithmic_tangent = elastic_tangent;
    std::array<std::array<double, 2>, 2> hardening_matrix = active_matrix;
    const double plastic_increment_norm = equivalent_plastic_strain(plastic_strain_increment);
    if (plastic_increment_norm > parameters.tolerance_)
    {
      const Core::LinAlg::Tensor<double, 3> hardening_direction =
          2.0 / (3.0 * plastic_increment_norm) * deviator(plastic_strain_increment);
      const double q_derivative =
          q_factor * cohesion_derivative(parameters, accumulated_plastic_strain);
      for (int beta = 0; beta < number_of_planes; ++beta)
      {
        const double hardening_component =
            Core::LinAlg::dot(hardening_direction, flow_normals[beta]);
        for (int alpha = 0; alpha < number_of_planes; ++alpha)
          hardening_matrix[alpha][beta] += q_derivative * hardening_component;
      }
    }

    const double hardening_determinant = number_of_planes == 1
                                             ? hardening_matrix[0][0]
                                             : hardening_matrix[0][0] * hardening_matrix[1][1] -
                                                   hardening_matrix[0][1] * hardening_matrix[1][0];
    if (std::abs(hardening_determinant) <= std::numeric_limits<double>::epsilon())
      return {.numerical_failure = true};

    std::array<std::array<double, 2>, 2> inverse_hardening_matrix{};
    if (number_of_planes == 1)
    {
      inverse_hardening_matrix[0][0] = 1.0 / hardening_matrix[0][0];
    }
    else
    {
      inverse_hardening_matrix[0][0] = hardening_matrix[1][1] / hardening_determinant;
      inverse_hardening_matrix[0][1] = -hardening_matrix[0][1] / hardening_determinant;
      inverse_hardening_matrix[1][0] = -hardening_matrix[1][0] / hardening_determinant;
      inverse_hardening_matrix[1][1] = hardening_matrix[0][0] / hardening_determinant;
    }

    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        for (int alpha = 0; alpha < number_of_planes; ++alpha)
          for (int beta = 0; beta < number_of_planes; ++beta)
            algorithmic_tangent(i, j) -= elastic_flow_normals[alpha](i) *
                                         inverse_hardening_matrix[alpha][beta] *
                                         elastic_yield_normals[beta](j);

    return {.valid = true,
        .stress = updated_stress,
        .plastic_strain_increment = plastic_strain_increment,
        .tangent = algorithmic_tangent,
        .accumulated_plastic_strain = accumulated_plastic_strain};
  }
}  // namespace

Mat::PAR::PlasticMohrCoulomb::PlasticMohrCoulomb(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      youngs_(matdata.parameters.get<double>("YOUNG_MODULUS")),
      poisson_ratio_(matdata.parameters.get<double>("POISSON_RATIO")),
      density_(matdata.parameters.get<double>("DENSITY")),
      cohesion_(matdata.parameters.get<double>("COHESION")),
      friction_angle_(matdata.parameters.get<double>("FRICTION_ANGLE")),
      dilatancy_angle_(matdata.parameters.get<double>("DILATANCY_ANGLE")),
      linear_hardening_(matdata.parameters.get<double>("LINEAR_HARDENING")),
      saturation_hardening_(matdata.parameters.get<double>("SATURATION_HARDENING")),
      hardening_exponent_(matdata.parameters.get<double>("HARDENING_EXP")),
      tolerance_(matdata.parameters.get<double>("TOLERANCE")),
      max_iterations_(matdata.parameters.get<int>("MAX_ITERATIONS"))
{
  if (dilatancy_angle_ <= 0.0)
    FOUR_C_THROW("DILATANCY_ANGLE must be strictly positive for an admissible apex return.");
  if (dilatancy_angle_ > friction_angle_)
    FOUR_C_THROW("DILATANCY_ANGLE must not exceed FRICTION_ANGLE.");
  if (saturation_hardening_ > 0.0 && saturation_hardening_ < cohesion_)
    FOUR_C_THROW("SATURATION_HARDENING must be zero or at least COHESION.");
  if (saturation_hardening_ > 0.0 && hardening_exponent_ <= 0.0)
    FOUR_C_THROW("HARDENING_EXP must be positive when saturation hardening is enabled.");
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::PlasticMohrCoulomb::create_material()
{
  return std::make_shared<Mat::PlasticMohrCoulomb>(this);
}

Mat::PlasticMohrCoulombType Mat::PlasticMohrCoulombType::instance_;

Core::Communication::ParObject* Mat::PlasticMohrCoulombType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* material = new Mat::PlasticMohrCoulomb();
  material->unpack(buffer);
  return material;
}

Mat::PlasticMohrCoulomb::PlasticMohrCoulomb() : params_(nullptr) {}

Mat::PlasticMohrCoulomb::PlasticMohrCoulomb(Mat::PAR::PlasticMohrCoulomb* params) : params_(params)
{
}

void Mat::PlasticMohrCoulomb::pack(Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, unique_par_object_id());
  add_to_pack(data, params_ != nullptr ? params_->id() : -1);
  add_to_pack(data, inv_plastic_rcg_last_);
  add_to_pack(data, inv_plastic_rcg_current_);
  add_to_pack(data, accumulated_plastic_strain_last_);
  add_to_pack(data, accumulated_plastic_strain_current_);
  add_to_pack(data, accumulated_plastic_volumetric_strain_last_);
  add_to_pack(data, accumulated_plastic_volumetric_strain_current_);
  add_to_pack(data, dissipated_energy_last_);
  add_to_pack(data, dissipated_energy_current_);
}

void Mat::PlasticMohrCoulomb::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());
  int material_id;
  extract_from_pack(buffer, material_id);
  params_ = nullptr;
  if (Global::Problem::instance()->materials() != nullptr &&
      Global::Problem::instance()->materials()->num() != 0)
  {
    const int problem_instance = Global::Problem::instance()->materials()->get_read_from_problem();
    Core::Mat::PAR::Parameter* material =
        Global::Problem::instance(problem_instance)->materials()->parameter_by_id(material_id);
    if (material->type() != material_type())
      FOUR_C_THROW("Packed material type {} does not match Mohr-Coulomb type {}.", material->type(),
          material_type());
    params_ = static_cast<Mat::PAR::PlasticMohrCoulomb*>(material);
  }

  extract_from_pack(buffer, inv_plastic_rcg_last_);
  extract_from_pack(buffer, inv_plastic_rcg_current_);
  extract_from_pack(buffer, accumulated_plastic_strain_last_);
  extract_from_pack(buffer, accumulated_plastic_strain_current_);
  extract_from_pack(buffer, accumulated_plastic_volumetric_strain_last_);
  extract_from_pack(buffer, accumulated_plastic_volumetric_strain_current_);
  extract_from_pack(buffer, dissipated_energy_last_);
  extract_from_pack(buffer, dissipated_energy_current_);
}

void Mat::PlasticMohrCoulomb::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  const auto identity = Core::LinAlg::TensorGenerators::identity<double, 3, 3>;
  inv_plastic_rcg_last_.assign(numgp, identity);
  inv_plastic_rcg_current_.assign(numgp, identity);
  accumulated_plastic_strain_last_.assign(numgp, 0.0);
  accumulated_plastic_strain_current_.assign(numgp, 0.0);
  accumulated_plastic_volumetric_strain_last_.assign(numgp, 0.0);
  accumulated_plastic_volumetric_strain_current_.assign(numgp, 0.0);
  dissipated_energy_last_.assign(numgp, 0.0);
  dissipated_energy_current_.assign(numgp, 0.0);
}

void Mat::PlasticMohrCoulomb::update()
{
  inv_plastic_rcg_last_ = inv_plastic_rcg_current_;
  accumulated_plastic_strain_last_ = accumulated_plastic_strain_current_;
  accumulated_plastic_volumetric_strain_last_ = accumulated_plastic_volumetric_strain_current_;
  dissipated_energy_last_ = dissipated_energy_current_;
}

void Mat::PlasticMohrCoulomb::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  FOUR_C_ASSERT_ALWAYS(defgrad != nullptr, "Mohr-Coulomb requires the deformation gradient.");
  const double determinant_deformation_gradient = Core::LinAlg::det(*defgrad);
  FOUR_C_ASSERT_ALWAYS(determinant_deformation_gradient > 0.0,
      "Mohr-Coulomb requires a positive deformation-gradient determinant, got {}.",
      determinant_deformation_gradient);

  const double shear_modulus = params_->youngs_ / (2.0 * (1.0 + params_->poisson_ratio_));
  const double lame_parameter =
      params_->youngs_ * params_->poisson_ratio_ /
      ((1.0 + params_->poisson_ratio_) * (1.0 - 2.0 * params_->poisson_ratio_));
  const double bulk_modulus = params_->youngs_ / (3.0 * (1.0 - 2.0 * params_->poisson_ratio_));
  const auto elastic_tangent = elastic_principal_tangent(shear_modulus, lame_parameter);

  const double sin_friction = std::sin(params_->friction_angle_);
  const double sin_dilatancy = std::sin(params_->dilatancy_angle_);
  const double friction_stress_ratio = (1.0 + sin_friction) / (1.0 - sin_friction);
  const double dilatancy_stress_ratio = (1.0 + sin_dilatancy) / (1.0 - sin_dilatancy);

  const auto inverse_deformation_gradient = Core::LinAlg::inv(*defgrad);
  const auto elastic_lcg_trial = Core::LinAlg::assume_symmetry(
      *defgrad * inv_plastic_rcg_last_.at(gp) * Core::LinAlg::transpose(*defgrad));
  const auto& [ascending_stretch_squares, ascending_eigenvectors] =
      Core::LinAlg::eig(elastic_lcg_trial);

  std::array<double, 3> stretch_squares{};
  std::array<Core::LinAlg::Tensor<double, 3>, 3> spatial_principal_directions{};
  std::array<Core::LinAlg::Tensor<double, 3>, 3> material_principal_directions{};
  Core::LinAlg::Tensor<double, 3> trial_logarithmic_strain{};
  for (int i = 0; i < 3; ++i)
  {
    const int ascending_index = 2 - i;
    stretch_squares[i] = ascending_stretch_squares[ascending_index];
    FOUR_C_ASSERT_ALWAYS(stretch_squares[i] > 0.0,
        "Trial elastic left Cauchy-Green tensor is not positive definite.");
    trial_logarithmic_strain(i) = 0.5 * std::log(stretch_squares[i]);
    for (int j = 0; j < 3; ++j)
      spatial_principal_directions[i](j) = ascending_eigenvectors(j, ascending_index);
    material_principal_directions[i] =
        inverse_deformation_gradient * spatial_principal_directions[i];
  }

  const auto trial_stress = elastic_tangent * trial_logarithmic_strain;
  const double q_last = 2.0 * std::sqrt(friction_stress_ratio) *
                        cohesion(*params_, accumulated_plastic_strain_last_.at(gp));

  ActiveSetResult return_result{};
  // checking for plasticity
  if (max_yield_function(trial_stress, friction_stress_ratio, q_last) <= params_->tolerance_)
  {
    // elastic regime
    return_result.valid = true;
    return_result.stress = trial_stress;
    return_result.tangent = elastic_tangent;
    return_result.accumulated_plastic_strain = accumulated_plastic_strain_last_.at(gp);
  }
  else
  {
    // plastic regime
    constexpr std::array<std::pair<int, int>, 2> face_planes{{{0, 2}, {0, 0}}};
    return_result = return_to_active_set(trial_stress, accumulated_plastic_strain_last_.at(gp),
        elastic_tangent, friction_stress_ratio, dilatancy_stress_ratio, face_planes, 1, *params_);

    // An inadmissible smooth-face return requires testing the adjacent edge returns.
    if (!return_result.valid)
    {
      constexpr std::array<std::pair<int, int>, 2> compression_edge_planes{{{0, 2}, {1, 2}}};
      const ActiveSetResult compression_edge = return_to_active_set(trial_stress,
          accumulated_plastic_strain_last_.at(gp), elastic_tangent, friction_stress_ratio,
          dilatancy_stress_ratio, compression_edge_planes, 2, *params_);

      constexpr std::array<std::pair<int, int>, 2> tension_edge_planes{{{0, 2}, {0, 1}}};
      const ActiveSetResult tension_edge = return_to_active_set(trial_stress,
          accumulated_plastic_strain_last_.at(gp), elastic_tangent, friction_stress_ratio,
          dilatancy_stress_ratio, tension_edge_planes, 2, *params_);

      if (compression_edge.valid && tension_edge.valid)
      {
        const double compression_correction =
            Core::LinAlg::dot(compression_edge.plastic_strain_increment,
                elastic_tangent * compression_edge.plastic_strain_increment);
        const double tension_correction = Core::LinAlg::dot(tension_edge.plastic_strain_increment,
            elastic_tangent * tension_edge.plastic_strain_increment);
        return_result =
            compression_correction <= tension_correction ? compression_edge : tension_edge;
      }
      else if (compression_edge.valid)
      {
        return_result = compression_edge;
      }
      else if (tension_edge.valid)
      {
        return_result = tension_edge;
      }
      else if (compression_edge.numerical_failure || tension_edge.numerical_failure)
      {
        FOUR_C_THROW("Mohr-Coulomb edge return failed to converge within {} local iterations.",
            params_->max_iterations_);
      }
    }

    // If neither a smooth-face nor an edge return is admissible, return to the apex.
    if (!return_result.valid)
    {
      if (return_result.numerical_failure)
        FOUR_C_THROW("Mohr-Coulomb face return failed to converge within {} local iterations.",
            params_->max_iterations_);

      const auto trial_deviator = deviator(trial_stress);
      return_result.accumulated_plastic_strain =
          accumulated_plastic_strain_last_.at(gp) +
          std::sqrt(2.0 / 3.0) * Core::LinAlg::norm2(trial_deviator) / (2.0 * shear_modulus);
      const double q = 2.0 * std::sqrt(friction_stress_ratio) *
                       cohesion(*params_, return_result.accumulated_plastic_strain);
      const double apex_stress = q / (friction_stress_ratio - 1.0);
      return_result.stress =
          Core::LinAlg::Tensor<double, 3>{{apex_stress, apex_stress, apex_stress}};

      const auto stress_correction = trial_stress - return_result.stress;
      const auto stress_correction_deviator = deviator(stress_correction);
      const double mean_stress_correction = sum_components(stress_correction) / 3.0;
      return_result.plastic_strain_increment =
          1.0 / (2.0 * shear_modulus) * stress_correction_deviator +
          mean_stress_correction / (3.0 * bulk_modulus) *
              Core::LinAlg::Tensor<double, 3>{{1.0, 1.0, 1.0}};
      FOUR_C_ASSERT_ALWAYS(belongs_to_apex_flow_cone(return_result.plastic_strain_increment,
                               dilatancy_stress_ratio, 100.0 * params_->tolerance_),
          "Trial state has no admissible Mohr-Coulomb apex return for the selected dilatation "
          "angle.");

      return_result.tangent = Core::LinAlg::Tensor<double, 3, 3>{};
      const auto trial_logarithmic_strain_deviator = deviator(trial_logarithmic_strain);
      const double deviator_norm = Core::LinAlg::norm2(trial_logarithmic_strain_deviator);
      if (deviator_norm > params_->tolerance_)
      {
        const double apex_hardening =
            2.0 * std::sqrt(friction_stress_ratio) *
            cohesion_derivative(*params_, return_result.accumulated_plastic_strain) /
            (friction_stress_ratio - 1.0);
        const Core::LinAlg::Tensor<double, 3> accumulated_strain_derivative =
            std::sqrt(2.0 / 3.0) / deviator_norm * trial_logarithmic_strain_deviator;
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j)
            return_result.tangent(i, j) = apex_hardening * accumulated_strain_derivative(j);
      }
      return_result.valid = true;
    }
  }

  Core::LinAlg::SymmetricTensor<double, 3, 3> elastic_lcg{};
  for (int i = 0; i < 3; ++i)
  {
    const double updated_stretch_square =
        stretch_squares[i] * std::exp(-2.0 * return_result.plastic_strain_increment(i));
    elastic_lcg +=
        updated_stretch_square * Core::LinAlg::self_dyadic(spatial_principal_directions[i]);
    stress += return_result.stress(i) * Core::LinAlg::self_dyadic(material_principal_directions[i]);
  }

  for (int a = 0; a < 3; ++a)
  {
    const auto material_projector = Core::LinAlg::self_dyadic(material_principal_directions[a]);
    cmat += -2.0 * return_result.stress(a) *
            Core::LinAlg::dyadic(material_projector, material_projector);

    for (int b = 0; b < 3; ++b)
    {
      const auto material_projector_b = Core::LinAlg::self_dyadic(material_principal_directions[b]);
      cmat += return_result.tangent(a, b) *
              Core::LinAlg::dyadic(material_projector, material_projector_b);

      if (a != b)
      {
        const double denominator = stretch_squares[a] - stretch_squares[b];
        const double factor =
            std::abs(denominator) > 100.0 * std::numeric_limits<double>::epsilon()
                ? (return_result.stress(a) * stretch_squares[b] -
                      return_result.stress(b) * stretch_squares[a]) /
                      denominator
                : 0.5 * (return_result.tangent(b, b) - return_result.tangent(a, b)) -
                      return_result.stress(b);
        const auto mixed_projector = Core::LinAlg::dyadic(
            material_principal_directions[a], material_principal_directions[b]);
        cmat += factor * Core::LinAlg::assume_symmetry(
                             Core::LinAlg::dyadic(mixed_projector, mixed_projector));
        cmat += factor * Core::LinAlg::assume_symmetry(Core::LinAlg::dyadic(
                             mixed_projector, Core::LinAlg::transpose(mixed_projector)));
      }
    }
  }

  inv_plastic_rcg_current_.at(gp) =
      Core::LinAlg::assume_symmetry(inverse_deformation_gradient * elastic_lcg *
                                    Core::LinAlg::transpose(inverse_deformation_gradient));
  accumulated_plastic_strain_current_.at(gp) = return_result.accumulated_plastic_strain;
  accumulated_plastic_volumetric_strain_current_.at(gp) =
      accumulated_plastic_volumetric_strain_last_.at(gp) +
      sum_components(return_result.plastic_strain_increment);
  dissipated_energy_current_.at(gp) =
      dissipated_energy_last_.at(gp) +
      Core::LinAlg::dot(return_result.stress, return_result.plastic_strain_increment);
}

void Mat::PlasticMohrCoulomb::register_output_data_names(
    std::unordered_map<std::string, int>& names_and_size) const
{
  names_and_size["plastic_strain"] = 6;
  names_and_size["accumulated_plastic_strain"] = 1;
  names_and_size["accumulated_plastic_volumetric_strain"] = 1;
  names_and_size["local_dissipated_energy"] = 1;
}

bool Mat::PlasticMohrCoulomb::evaluate_output_data(
    const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const
{
  if (name == "accumulated_plastic_strain")
  {
    for (std::size_t gp = 0; gp < accumulated_plastic_strain_current_.size(); ++gp)
      data(gp, 0) = accumulated_plastic_strain_current_[gp];
    return true;
  }
  if (name == "accumulated_plastic_volumetric_strain")
  {
    for (std::size_t gp = 0; gp < accumulated_plastic_volumetric_strain_current_.size(); ++gp)
      data(gp, 0) = accumulated_plastic_volumetric_strain_current_[gp];
    return true;
  }
  if (name == "local_dissipated_energy")
  {
    for (std::size_t gp = 0; gp < dissipated_energy_current_.size(); ++gp)
      data(gp, 0) = dissipated_energy_current_[gp];
    return true;
  }
  if (name == "plastic_strain")
  {
    for (std::size_t gp = 0; gp < inv_plastic_rcg_current_.size(); ++gp)
    {
      const auto& [eigenvalues, eigenvectors] = Core::LinAlg::eig(inv_plastic_rcg_current_[gp]);
      Core::LinAlg::SymmetricTensor<double, 3, 3> logarithmic_plastic_strain{};
      for (int i = 0; i < 3; ++i)
      {
        FOUR_C_ASSERT_ALWAYS(eigenvalues[i] > 0.0,
            "Inverse plastic right Cauchy-Green tensor is not positive definite.");
        Core::LinAlg::Tensor<double, 3> direction{};
        for (int j = 0; j < 3; ++j) direction(j) = eigenvectors(j, i);
        logarithmic_plastic_strain +=
            -0.5 * std::log(eigenvalues[i]) * Core::LinAlg::self_dyadic(direction);
      }
      const Core::LinAlg::Matrix<6, 1> voigt =
          Core::LinAlg::make_strain_like_voigt_matrix(logarithmic_plastic_strain);
      for (int i = 0; i < 6; ++i) data(gp, i) = voigt(i, 0);
    }
    return true;
  }
  return false;
}

FOUR_C_NAMESPACE_CLOSE
