// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_mat_stvenantkirchhoff.hpp"
#include "4C_utils_function.hpp"

#ifndef FOUR_C_STRUCTURE_NEW_FUNCTIONS_HPP
#define FOUR_C_STRUCTURE_NEW_FUNCTIONS_HPP

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE
namespace Core::Utils
{
  class FunctionManager;
}

namespace Solid
{
  /// add valid structure-specific function lines
  void add_valid_structure_functions(Core::Utils::FunctionManager& function_manager);

  /// special implementation for weakly compressible flow - Etienne FSI problem
  class WeaklyCompressibleEtienneFSIStructureFunction : public Core::Utils::FunctionOfSpaceTime
  {
   public:
    WeaklyCompressibleEtienneFSIStructureFunction(const Mat::PAR::StVenantKirchhoff& fparams);

    double evaluate(std::span<const double> x, double t, std::size_t component) const override;

    std::vector<double> evaluate_time_derivative(
        std::span<const double> x, double t, unsigned deg, std::size_t component) const override;

    [[nodiscard]] std::size_t number_components() const override { return (2); };
  };

  /// special implementation for weakly compressible flow - Etienne FSI problem (force)
  class WeaklyCompressibleEtienneFSIStructureForceFunction : public Core::Utils::FunctionOfSpaceTime
  {
   public:
    WeaklyCompressibleEtienneFSIStructureForceFunction(const Mat::PAR::StVenantKirchhoff& fparams);

    double evaluate(std::span<const double> x, double t, std::size_t component) const override;

    std::vector<double> evaluate_time_derivative(
        std::span<const double> x, double t, unsigned deg, std::size_t component) const override;

    // Case with spatially varying material properties (inputfield)
    double evaluate(std::span<const double> x, double t, std::size_t component, int eleGID) const override;
    
    std::vector<double> evaluate_time_derivative(
        std::span<const double> x, double t, unsigned deg, std::size_t component, int eleGID) const override;

    [[nodiscard]] std::size_t number_components() const override { return (2); };

   private:
    Core::IO::InputField<double> youngmodulus_;
    double poissonratio_;
    double strucdensity_;
  };
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
