// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_functions.hpp"

#include "4C_global_data.hpp"
#include "4C_io_input_spec_builders.hpp"

#include <cmath>

FOUR_C_NAMESPACE_OPEN namespace
{
  std::shared_ptr<Core::Utils::FunctionOfSpaceTime> create_scatra_function(
      const Core::IO::InputParameterContainer& container)
  {
    const auto input_container = container.group("SCATRA_FUNCTION");
    const auto type = input_container.get<ScaTra::ScatraFunctionType>("type");
    const auto& parameters = input_container.group("parameters");
    const auto& magnetization_model_parameters =
        input_container.group("particle_magnetization_model");

    if (type == ScaTra::ScatraFunctionType::cylinder_magnet)
    {
      return std::make_shared<ScaTra::CylinderMagnetFunction>(ScaTra::CylinderMagnetParameters{
          .magnet_radius = parameters.get<double>("magnet_radius"),
          .magnet_length = parameters.get<double>("magnet_length"),
          .magnetic_permeability = parameters.get<double>("magnetic_permeability"),
          .magnet_magnetization = parameters.get<double>("magnet_magnetization"),
          .magnet_position =
              ScaTra::Position{
                  .x = parameters.get<std::vector<double>>("magnet_position")[0],
                  .y = parameters.get<std::vector<double>>("magnet_position")[1],
                  .z = parameters.get<std::vector<double>>("magnet_position")[2],
              },
          .dynamic_viscosity_fluid = parameters.get<double>("dynamic_viscosity_fluid"),
          .magnet_rotation =
              ScaTra::Rotation{
                  .x_axis = parameters.get<double>("rotation_around_x_axis"),
                  .y_axis = parameters.get<double>("rotation_around_y_axis"),
              },
          .particle_radius = parameters.get<double>("particle_radius"),
          .particle_magnetization_model_type =
              magnetization_model_parameters.get<ScaTra::ParticleMagnetizationModelType>("type"),
          .particle_magnetization_parameters =
              ScaTra::ParticleMagnetizationModelParameters{
                  .saturation_magnetization = magnetization_model_parameters.get_or<double>(
                      "saturation_magnetization", -1.0),
                  .susceptibility =
                      magnetization_model_parameters.get_or<double>("susceptibility", -1.0),
              },
      });
    }

    FOUR_C_THROW("Wrong type of SCATRA_FUNCTION");
  }


  std::shared_ptr<Core::Utils::FunctionOfSpaceTime> try_create_scatra_function(
      const std::vector<Core::IO::InputParameterContainer>& parameters)
  {
    if (parameters.size() != 1) return nullptr;

    if (const auto& container = parameters.front(); container.has_group("SCATRA_FUNCTION"))
    {
      return create_scatra_function(container);
    }
    return std::shared_ptr<Core::Utils::FunctionOfSpaceTime>(nullptr);
  }

}  // namespace


void ScaTra::add_valid_scatra_functions(Core::Utils::FunctionManager& function_manager)
{
  using namespace Core::IO::InputSpecBuilders;

  auto spec =
      group("SCATRA_FUNCTION",
          {
              parameter<ScatraFunctionType>("type"),
              group("parameters",
                  {
                      parameter<double>(
                          "magnet_radius", {.description = "Radius of the cylinder magnet"}),
                      parameter<double>(
                          "magnet_length", {.description = "Length of the cylinder magnet"}),
                      parameter<double>(
                          "magnetic_permeability", {.description = "Magnetic permeability"}),
                      parameter<double>("magnet_magnetization",
                          {.description = "Magnetization of the cylinder magnet"}),
                      parameter<std::vector<double>>("magnet_position",
                          {.description = "Position of the center of the cylinder magnet in the "
                                          "global coordinate system (X,Y,Z)",
                              .size = 3}),
                      parameter<double>("dynamic_viscosity_fluid",
                          {.description = "Dynamic viscosity of the fluid"}),
                      parameter<double>("rotation_around_x_axis",
                          {.description = "Rotation of the magnet around the x-axis in degrees",
                              .default_value = 0.0}),
                      parameter<double>("rotation_around_y_axis",
                          {.description = "Rotation of the magnet around the y-axis in degrees",
                              .default_value = 0.0}),
                      parameter<double>(
                          "particle_radius", {.description = "Radius of the magnetic particle"}),
                  }),
              selection<ParticleMagnetizationModelType>("particle_magnetization_model",
                  {
                      .selector = "type",
                      .choices =
                          {
                              {
                                  ParticleMagnetizationModelType::linear,
                                  parameter<double>(
                                      "susceptibility", {.description = "Magnetic susceptibility"}),
                              },
                              {
                                  ParticleMagnetizationModelType::linear_with_saturation,
                                  all_of(
                                      {
                                          parameter<double>("saturation_magnetization",
                                              {.description = "Saturation magnetization"}),
                                          parameter<double>("susceptibility",
                                              {.description = "Magnetic susceptibility"}),
                                      }),
                              },
                              {
                                  ParticleMagnetizationModelType::superparamagnetic,
                                  parameter<double>("saturation_magnetization",
                                      {.description = "Saturation magnetization"}),
                              },
                          },
                  },
                  {.description = "Magnetization model for the demagnetization factor f(|H|)"}),
          });

  function_manager.add_function_definition(std::move(spec), try_create_scatra_function);
}


ScaTra::CylinderMagnetFunction::CylinderMagnetFunction(const CylinderMagnetParameters& parameters)
    : parameters_(parameters)
{
}


double ScaTra::CylinderMagnetFunction::evaluate(
    const double* x, double t, std::size_t component) const
{
  const std::span<const double, 3> position(x, 3);
  const auto force_vector = evaluate_magnetic_force(position);
  return force_vector[component];
}


void ScaTra::CylinderMagnetFunction::evaluate_vector(
    const std::span<const double, 3> x, double t, std::span<double> values) const
{
  auto force = evaluate_magnetic_force(x);
  FOUR_C_ASSERT(force.size() == 3, "Internal error: force vector has wrong size");
  std::ranges::copy(force, values.begin());
}


std::array<double, 3> ScaTra::CylinderMagnetFunction::global_to_cylinder_coordinates(
    const std::span<const double, 3> x) const
{
  // convert rotation angles to radians
  const double gamma = parameters_.magnet_rotation.x_axis * M_PI / 180;
  const double beta = parameters_.magnet_rotation.y_axis * M_PI / 180;

  // translate coordinates to the magnet's coordinate system
  const double x_translated = x[0] - parameters_.magnet_position.x;
  const double y_translated = x[1] - parameters_.magnet_position.y;
  const double z_translated = x[2] - parameters_.magnet_position.z;

  // rotate coordinates to the magnet's coordinate system
  const double xi = x_translated * std::cos(beta) +
                    y_translated * std::sin(beta) * std::sin(gamma) +
                    z_translated * std::sin(beta) * std::cos(gamma);
  const double eta = y_translated * std::cos(gamma) - z_translated * std::sin(gamma);
  const double zeta = -x_translated * std::sin(beta) +
                      y_translated * std::cos(beta) * std::sin(gamma) +
                      z_translated * std::cos(beta) * std::cos(gamma);

  // transform to cylindrical coordinates
  // radial coordinate
  auto rho = std::sqrt(std::pow(xi, 2.0) + std::pow(eta, 2.0));
  // azimuthal coordinate
  const auto phi = std::atan2(eta, xi);
  // height coordinate
  const auto z = zeta;

  // the equation for the magnetic field/force is undefined at rho = 0:
  // outside the magnet, these singularities are removable and H_mag/F_mag are extendable
  if (std::abs(rho) < 1e-2) rho = 1e-2;

  return {rho, phi, z};
}


std::array<double, 3> ScaTra::CylinderMagnetFunction::cylinder_to_global_coordinates(
    const double rho_component, const double z_component, const double phi) const
{
  // convert rotation angles to radians
  const double gamma = parameters_.magnet_rotation.x_axis * M_PI / 180;
  const double beta = parameters_.magnet_rotation.y_axis * M_PI / 180;

  // transform to cartesian coordinates
  const double xi = rho_component * std::cos(phi);
  const double eta = rho_component * std::sin(phi);
  const double zeta = z_component;

  // rotate coordinates
  double X = xi * std::cos(beta) - zeta * std::sin(beta);
  double Y = xi * std::sin(beta) * std::sin(gamma) + eta * std::cos(gamma) +
             zeta * std::cos(beta) * std::sin(gamma);
  double Z = xi * std::sin(beta) * std::cos(gamma) - eta * std::sin(gamma) +
             zeta * std::cos(beta) * std::cos(gamma);

  if (std::abs(X) < 1e-14) X = 0.0;
  if (std::abs(Y) < 1e-14) Y = 0.0;
  if (std::abs(Z) < 1e-14) Z = 0.0;

  return {X, Y, Z};
}


std::array<double, 3> ScaTra::CylinderMagnetFunction::evaluate_magnetic_field(
    const std::span<const double, 3> x) const
{
  const double radius_magnet = parameters_.magnet_radius;
  // half-length L in Caciagli et al. (2018)
  const double half_length = 0.5 * parameters_.magnet_length;

  // transform coordinates to cylindrical coordinates in the magnet's coordinate system
  const auto cylinder_coordinates = global_to_cylinder_coordinates(x);
  const double rho = cylinder_coordinates[0];
  const double phi = cylinder_coordinates[1];
  const double z = cylinder_coordinates[2];

  // calculate auxiliary variables
  // the equation for the magnetic field is undefined at rho = R_mag:
  // outside the magnet, these singularities are removable and H_mag is extendable
  auto calc_rho_p = [radius_magnet](const double xi)
  {
    if (std::abs(radius_magnet + xi) < 1e-2) return 1e-2;
    return radius_magnet + xi;
  };
  auto calc_rho_m = [radius_magnet](const double xi)
  {
    if (std::abs(radius_magnet - xi) < 1e-2) return 1e-2;
    return radius_magnet - xi;
  };
  const double rho_p = calc_rho_p(rho);
  const double rho_m = calc_rho_m(rho);

  // see Eq. (5) in Caciagli et al. (2018)
  const double zeta_p = half_length + z;
  const double zeta_m = half_length - z;
  const double alpha_p = std::pow(std::sqrt(std::pow(zeta_p, 2.0) + std::pow(rho_p, 2.0)), -1.0);
  const double alpha_m = std::pow(std::sqrt(std::pow(zeta_m, 2.0) + std::pow(rho_p, 2.0)), -1.0);
  const double beta_p = zeta_p * alpha_p;
  const double beta_m = -zeta_m * alpha_m;  // different sign compared to Caciagli et al. (2018)
  const double gamma = -rho_m / rho_p;
  const double k_p = std::sqrt((std::pow(zeta_p, 2.0) + std::pow(rho_m, 2.0)) /
                               (std::pow(zeta_p, 2.0) + std::pow(rho_p, 2.0)));
  const double k_m = std::sqrt((std::pow(zeta_m, 2.0) + std::pow(rho_m, 2.0)) /
                               (std::pow(zeta_m, 2.0) + std::pow(rho_p, 2.0)));


  // calculate auxiliary functions with elliptic integrals
  // see Eq. (4) in Caciagli et al. (2018)
  auto P_1 = [](const double k)
  {
    const double k_squared = std::pow(k, 2.0);
    const double K = std::comp_ellint_1(std::sqrt(1.0 - k_squared));
    const double E = std::comp_ellint_2(std::sqrt(1.0 - k_squared));
    return K - (2.0 / (1.0 - k_squared)) * (K - E);
  };

  auto P_2 = [gamma](const double k)
  {
    const double gamma_squared = std::pow(gamma, 2.0);
    const double k_squared = std::pow(k, 2.0);
    const double K = std::comp_ellint_1(std::sqrt(1.0 - k_squared));
    const double P = safe_comp_ellint_3(std::sqrt(1.0 - k_squared), 1 - gamma_squared);

    return -(gamma / (1.0 - gamma_squared)) * (P - K) -
           (1.0 / (1.0 - gamma_squared)) * (gamma_squared * P - K);
  };

  // calculate magnetic field components
  // see Eq. (3) in Caciagli et al. (2018)
  const double H_rho = (radius_magnet * parameters_.magnet_magnetization / M_PI) *
                       (alpha_p * P_1(k_p) - alpha_m * P_1(k_m));
  const double H_z = (radius_magnet * parameters_.magnet_magnetization / (M_PI * rho_p)) *
                     (beta_p * P_2(k_p) - beta_m * P_2(k_m));

  // transform back to global cartesian coordinates
  return cylinder_to_global_coordinates(H_rho, H_z, phi);
}


double ScaTra::CylinderMagnetFunction::evaluate_magnetization_model(
    const std::span<const double, 3> x) const
{
  const auto magnetic_field = evaluate_magnetic_field(x);
  const auto magnetic_field_magnitude =
      std::sqrt(std::pow(magnetic_field[0], 2.0) + std::pow(magnetic_field[1], 2.0) +
                std::pow(magnetic_field[2], 2.0));

  const auto magnetization_parameters = parameters_.particle_magnetization_parameters;

  // linear without saturation
  if (parameters_.particle_magnetization_model_type == ParticleMagnetizationModelType::linear)
  {
    return 3.0 * magnetization_parameters.susceptibility /
           (3.0 + magnetization_parameters.susceptibility);
  }
  // linear with saturation (see Eq. (13) in Wirthl et al. (2024))
  if (parameters_.particle_magnetization_model_type ==
      ParticleMagnetizationModelType::linear_with_saturation)
  {
    const double magnetization_factor_linear =
        magnetization_parameters.susceptibility / (3.0 + magnetization_parameters.susceptibility);
    const double magnetization_factor_saturated =
        magnetization_parameters.saturation_magnetization / (3.0 * magnetic_field_magnitude);

    if (magnetization_factor_linear < magnetization_factor_saturated)
      return 3.0 * magnetization_factor_linear;
    else
      return 3.0 * magnetization_factor_saturated;
  }
  // superparamagnetic (see Eq. (17) in Wirthl et al. (2024))
  if (parameters_.particle_magnetization_model_type ==
      ParticleMagnetizationModelType::superparamagnetic)
  {
    // case of superparamagnetic particles: chi >> 1
    if (magnetic_field_magnitude < (1.0 / 3.0) * magnetization_parameters.saturation_magnetization)
      return 3.0;
    else
      return magnetization_parameters.saturation_magnetization / magnetic_field_magnitude;
  }
  FOUR_C_THROW("Magnetization model not implemented.");
}


std::array<double, 3> ScaTra::CylinderMagnetFunction::evaluate_magnetic_force(
    const std::span<const double, 3> x) const
{
  // magnet parameters
  const double radius_magnet = parameters_.magnet_radius;
  const double half_length_magnet = 0.5 * parameters_.magnet_length;

  // mobility zeta^-1 of the particle (based on Stokes law)
  const double mobility = std::pow(
      6.0 * M_PI * parameters_.dynamic_viscosity_fluid * parameters_.particle_radius, -1.0);
  const double volume_particle = (4.0 / 3.0) * M_PI * std::pow(parameters_.particle_radius, 3.0);

  // demagnetization factor f(|H|) as part of the magnetization model
  const double f_H = evaluate_magnetization_model(x);

  // transform to cylindrical coordinates
  const auto cylinder_coordinates = global_to_cylinder_coordinates(x);
  const double rho = cylinder_coordinates[0];
  const double phi = cylinder_coordinates[1];
  const double z = cylinder_coordinates[2];

  if (std::abs(rho) < radius_magnet && std::abs(z) < half_length_magnet)
    FOUR_C_THROW("Magnet inside the domain. Abort.");

  // calculate auxiliary variables
  // the equation for the magnetic force is undefined at rho = R_mag:
  // outside the magnet, these singularities are removable and F_mag is extendable
  auto calc_rho_p = [radius_magnet](const double xi)
  {
    if (std::abs(radius_magnet + xi) < 1e-2) return 1e-2;
    return radius_magnet + xi;
  };
  auto calc_rho_m = [radius_magnet](const double xi)
  {
    if (std::abs(radius_magnet - xi) < 1e-2) return 1e-2;
    return radius_magnet - xi;
  };
  const double rho_p = calc_rho_p(rho);
  const double rho_m = calc_rho_m(rho);

  const double zeta_p = half_length_magnet + z;
  const double zeta_m = half_length_magnet - z;

  const double beta = 4.0 * rho * radius_magnet / (std::pow(rho_p, 2.0));

  const double a_1 = std::pow(zeta_p, 2.0) + std::pow(rho_p, 2.0);
  const double a_2 = std::pow(zeta_m, 2.0) + std::pow(rho_p, 2.0);
  const double a_3 = std::pow(zeta_p, 2.0) + std::pow(rho_m, 2.0);
  const double a_4 = std::pow(zeta_m, 2.0) + std::pow(rho_m, 2.0);

  const double alpha_p = std::sqrt(a_1);
  const double alpha_m = std::sqrt(a_2);

  const double psi_p = 4.0 * rho * radius_magnet / a_1;
  const double psi_m = 4.0 * rho * radius_magnet / a_2;

  const double b_1 = std::pow(zeta_p, 2.0) + std::pow(radius_magnet, 2.0);
  const double b_2 = std::pow(zeta_m, 2.0) + std::pow(radius_magnet, 2.0);
  const double b_3 = std::pow(zeta_p, 2.0) - std::pow(radius_magnet, 2.0);
  const double b_4 = std::pow(zeta_m, 2.0) - std::pow(radius_magnet, 2.0);

  const double c_1 = b_1 + std::pow(rho, 2.0);
  const double c_2 = b_2 + std::pow(rho, 2.0);
  const double c_3 = b_3 + std::pow(rho, 2.0);
  const double c_4 = b_4 + std::pow(rho, 2.0);

  // precompute complete elliptic integrals
  const double K_p = std::comp_ellint_1(std::sqrt(psi_p));
  const double K_m = std::comp_ellint_1(std::sqrt(psi_m));
  const double E_p = std::comp_ellint_2(std::sqrt(psi_p));
  const double E_m = std::comp_ellint_2(std::sqrt(psi_m));

  // calculate auxiliary functions
  const double Q_1 =
      a_2 * alpha_p * E_m - a_1 * alpha_m * E_p + c_1 * alpha_m * K_p - c_2 * alpha_p * K_m;

  const double Q_2 = rho_p * zeta_p * alpha_m * K_p + rho_p * zeta_m * alpha_p * K_m +
                     rho_m * zeta_p * alpha_m * safe_comp_ellint_3(std::sqrt(psi_p), beta) +
                     rho_m * zeta_m * alpha_p * safe_comp_ellint_3(std::sqrt(psi_m), beta);

  const double F_rho =
      mobility * std::pow(parameters_.magnet_magnetization, 2.0) *
      parameters_.magnetic_permeability * volume_particle * f_H *
      (std::pow(rho, 2.0) * Q_2 *
              (c_1 * zeta_p * a_4 * alpha_m * E_p + c_2 * zeta_m * a_3 * alpha_p * E_m -
                  a_4 * a_3 * zeta_m * alpha_p * K_m - a_4 * a_3 * zeta_p * alpha_m * K_p) +
          rho_p * Q_1 *
              (a_4 * (std::pow(b_1, 2.0) + b_3 * std::pow(rho, 2.0)) * alpha_m * E_p -
                  a_3 * (std::pow(b_2, 2.0) + b_4 * std::pow(rho, 2.0)) * alpha_p * E_m +
                  a_4 * a_3 * b_2 * alpha_p * K_m - a_4 * a_3 * b_1 * alpha_m * K_p)) /
      (4.0 * std::pow(M_PI, 2.0) * std::pow(rho, 3.0) * rho_p * a_4 * a_2 * a_3 * a_1);

  const double F_z =
      mobility * std::pow(parameters_.magnet_magnetization, 2.0) *
      parameters_.magnetic_permeability * volume_particle * f_H *
      (((Q_1 / std::pow(rho, 2.0)) *
           (a_4 * a_3 * zeta_m * alpha_p * K_m + a_4 * a_3 * zeta_p * alpha_m * K_p -
               c_2 * zeta_m * a_3 * alpha_p * E_m - c_1 * zeta_p * a_4 * alpha_m * E_p)) +
          ((Q_2 / rho_p) * (c_4 * a_3 * alpha_p * E_m - c_3 * a_4 * alpha_m * E_p -
                               a_3 * a_4 * alpha_p * K_m + a_3 * a_4 * alpha_m * K_p))) /
      (4.0 * std::pow(M_PI, 2.0) * a_4 * a_2 * a_3 * a_1);

  // transform back to global cartesian coordinates
  return cylinder_to_global_coordinates(F_rho, F_z, phi);
}


double ScaTra::safe_comp_ellint_3(double k, double nu)
{
  constexpr double epsilon = 1e-10;
  if (std::abs(k) > 1.0 + epsilon)
  {
    FOUR_C_THROW("Evaluate cylindrical magnet function: Error in third elliptic integral: |k| > 1");
  }
  if (nu > 1.0 + epsilon)
  {
    FOUR_C_THROW(
        "Evaluate cylindrical magnet function: Error in third elliptic integral: nu must be < 1");
  }

  if (std::abs(std::abs(k) - 1.0) < epsilon || std::abs(nu - 1.0) < epsilon)
  {
    return 0.0;
  }

  constexpr double safe_margin = 1e-6;
  k = std::clamp(k, -1.0 + safe_margin, 1.0 - safe_margin);
  nu = std::min(nu, 1.0 - safe_margin);

  return std::comp_ellint_3(k, nu);
}

FOUR_C_NAMESPACE_CLOSE
