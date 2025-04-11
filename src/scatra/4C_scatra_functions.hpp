// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_FUNCTIONS_HPP
#define FOUR_C_SCATRA_FUNCTIONS_HPP

#include "4C_config.hpp"

#include "4C_utils_function.hpp"


FOUR_C_NAMESPACE_OPEN

namespace Core::Utils
{
  class FunctionManager;
}


namespace ScaTra
{
  /**
   * Type of the Scatra function
   */
  enum class ScatraFunctionType
  {
    cylinder_magnet,
  };


  enum class ParticleMagnetizationModelType
  {
    linear,
    linear_with_saturation,
    superparamagnetic,
  };

  /**
   * Position of the center of the cylinder magnet in the global coordinate system (X,Y,Z)
   */
  struct Position
  {
    double x{};
    double y{};
    double z{};
  };

  /**
   * Rotation of the magnet around the x and y-axis in degrees
   */
  struct Rotation
  {
    double x_axis{};
    double y_axis{};
  };

  /**
   * Parameters of the particle magnetization model
   */
  struct ParticleMagnetizationModelParameters
  {
    double saturation_magnetization{};
    double susceptibility{};
  };

  /**
   * Parameters of the cylinder magnet
   */
  struct CylinderMagnetParameters
  {
    double magnet_radius{};
    double magnet_length{};
    double magnetic_permeability{};
    double magnet_magnetization{};
    Position magnet_position{};
    double dynamic_viscosity_fluid{};
    Rotation magnet_rotation{};
    double particle_radius{};
    ParticleMagnetizationModelType particle_magnetization_model_type{};
    ParticleMagnetizationModelParameters particle_magnetization_parameters{};
  };

  /// Attach functions to the @p function_manager.
  void add_valid_scatra_functions(Core::Utils::FunctionManager& function_manager);


  /**
   * Functions for a cylindrical magnet of finite length with longitudinal magnetization
   */
  class CylinderMagnetFunction : public Core::Utils::FunctionOfSpaceTime
  {
   public:
    CylinderMagnetFunction(const CylinderMagnetParameters& parameters);

    /**
     * Evaluate the function at a given position x and time t for a given component
     *
     * @param x point in 3-dimensional space where the function will be evaluated
     * @param t point in time where the function will be evaluated
     * @param component index of the function-component which should be evaluated
     * @return function value
     */
    double evaluate(const double* x, double t, std::size_t component) const override;

    /**
     * Evaluate the function at a given position x and time t and return the vector of all
     * components
     *
     * @param x point in 3-dimensional space where the function will be evaluated
     * @param t point in time where the function will be evaluated
     * @param values function values
     */
    void evaluate_vector(
        std::span<const double, 3> x, double t, std::span<double> values) const override;

    /**
     * Number of components of the function (3 for the cylinder magnet)
     *
     * @return 3
     */
    [[nodiscard]] std::size_t number_components() const override { return (3); };

   private:
    /*!
     * @brief evaluate magnetic field H of a cylindrical magnet with longitudinal magnetization at a
     * given position (X,Y,Z) and parameters params
     *
     * The magnetic field is calculated according to the following reference:
     *     Caciagli, A., Baars, R. J., Philipse, A. P., & Kuipers, B. W. M. (2018). Exact expression
     *     for the magnetic field of a finite cylinder with arbitrary uniform magnetization. Journal
     *     of Magnetism and Magnetic Materials, 456, 423-432.
     *     https://doi.org/10.1016/j.jmmm.2018.02.003
     *
     * @param x global cartesian coordinates (X,Y,Z) of the point x
     * @return magnetic field vector
     */
    [[nodiscard]] std::array<double, 3> evaluate_magnetic_field(std::span<const double, 3> x) const;

    /*!
     * @brief evaluate demagnetization factor f(|H|) as part of the magnetization model at a given
     * position (x,y,z)
     *
     * @param x global cartesian coordinates (X,Y,Z) of the point x
     * @return magnetization model
     */
    [[nodiscard]] double evaluate_magnetization_model(std::span<const double, 3> x) const;

    /*!
     * @brief evaluate magnetic force at a given position (X,Y,Z)
     *
     * The magnetic force on a particle is calculated according to the following reference:
     *     Wirthl, B., Wirthl, V., & Wall, W. A. (2024). Efficient computational model of the
     *     in-flow capturing of magnetic nanoparticles by a cylindrical magnet for cancer
     *     nanomedicine. Physical Review E, 109, 065309. https://doi.org/10.1103/PhysRevE.109.065309
     *
     * @param x global cartesian coordinates (X,Y,Z) of the point x
     * @return magnetic force vector
     */
    [[nodiscard]] std::array<double, 3> evaluate_magnetic_force(std::span<const double, 3> x) const;


    /*!
     * \brief transform the global cartesian coordinates (X,Y,Z) of the point x to be evaluated to
     * coordinates in the coordinate system of the cylinder magnet: cylindrical coordinates (rho,
     * phi, z) with origin at the center of the magnet
     *
     * @param x global cartesian coordinates (X,Y,Z) of the point x
     * @return cylindrical coordinates (rho, phi, z)
     */
    [[nodiscard]] std::array<double, 3> global_to_cylinder_coordinates(
        std::span<const double, 3> x) const;

    /*!
     * \brief transform the cylindrical coordinates of a result vector to global cartesian
     * coordinates, e.g., for the magnetic field (H_rho, H_z) to (H_x, H_y, H_z) or for the
     * magnetic force (F_rho, F_z) to (F_x, F_y, F_z).
     * Note that the azimuthal component is zero for the magnetic field and the magnetic force due
     * to symmetry.
     *
     * @param rho_component radial component
     * @param z_component z component
     * @param phi angle
     * @return global cartesian coordinates (X,Y,Z)
     */
    [[nodiscard]] std::array<double, 3> cylinder_to_global_coordinates(
        double rho_component, double z_component, double phi) const;

    /// parameters of the cylinder magnet
    const CylinderMagnetParameters parameters_;
  };

  /*!
   * Evaluate the complete elliptic integral of the third kind
   * with
   * - checking if k is in the range [-1,1]
   * - checking if nu is in the range [0,1]
   * - avoiding numerical underflow for k and nue close to 1
   *
   * @param k elliptic modulus
   * @param nu characteristic parameter
   * @return complete elliptic integral of the third kind
   */
  double safe_comp_ellint_3(double k, double nu);

}  // namespace ScaTra

FOUR_C_NAMESPACE_CLOSE

#endif
