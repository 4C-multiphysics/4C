// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_SCATRA_UTILS_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_SCATRA_UTILS_HPP

#include "4C_config.hpp"

#include "4C_linalg_vector.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_algorithm_dependencies.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_input.hpp"

#include <Sacado.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <memory>
#include <set>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::LinAlg
{
  class MultiMapExtractor;
}
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace PoroPressureBased
{
  class PorofluidElastScatraBaseAlgorithm;
  class PorofluidElastScatraArteryCouplingBaseAlgorithm;

  //! setup discretizations and dofsets
  std::map<int, std::set<int>> setup_discretizations_and_field_coupling_porofluid_elast_scatra(
      const PorofluidElastAlgorithmDeps& porofluid_elast_algorithm_deps, MPI_Comm comm,
      const std::string& struct_disname, const std::string& fluid_disname,
      const std::string& scatra_disname, int& ndsporo_disp, int& ndsporo_vel,
      int& ndsporo_solidpressure, int& ndsporofluid_scatra, const bool artery_coupl);

  //! exchange material pointers of discretizations
  void assign_material_pointers_porofluid_elast_scatra(
      const PorofluidElastAlgorithmDeps& porofluid_elast_algorithm_deps,
      const std::string& struct_disname, const std::string& fluid_disname,
      const std::string& scatra_disname, const bool artery_coupl);

  //! create solution algorithm depending on input file
  std::shared_ptr<PorofluidElastScatraBaseAlgorithm> create_algorithm_porofluid_elast_scatra(
      Global::Problem& problem,                      //!< global problem instance (i)
      SolutionSchemePorofluidElastScatra solscheme,  //!< solution scheme to build (i)
      const Teuchos::ParameterList& timeparams,      //!< problem parameters (i)
      MPI_Comm comm                                  //!< communicator(i)
  );

  //! create coupling strategy for coupling with 1D network depending on input file
  std::shared_ptr<PorofluidElastScatraArteryCouplingBaseAlgorithm>
  create_and_init_artery_coupling_strategy(std::shared_ptr<Core::FE::Discretization> arterydis,
      std::shared_ptr<Core::FE::Discretization> contdis,
      const PorofluidElastScatraArteryCouplingDeps& artery_coupling_deps,
      const Teuchos::ParameterList& meshtyingparams, const std::string& condname,
      const bool evaluate_on_lateral_surface);

  //! backward-compatible overload that derives dependencies from Global::Problem
  std::shared_ptr<PorofluidElastScatraArteryCouplingBaseAlgorithm>
  create_and_init_artery_coupling_strategy(std::shared_ptr<Core::FE::Discretization> arterydis,
      std::shared_ptr<Core::FE::Discretization> contdis,
      const Teuchos::ParameterList& meshtyingparams, const std::string& condname,
      const bool evaluate_on_lateral_surface);

  namespace Internal
  {
    //! number of times the bisection fallback of the oxygen partial pressure inversion was needed
    inline std::atomic<unsigned long> oxy_partial_pressure_bisection_counter{0};

    //! the fallback warning is written only every n-th time the bisection is used (probably it
    //! makes sense to change it depending on your problem)
    inline constexpr unsigned long oxy_partial_pressure_bisection_warning_interval = 100;
  }  // namespace Internal

  /**
   * Get oxygen partial pressure from oxygen concentration via numerical inversion
   * templated to FAD and double
   *
   * The forward relation
   *   CaO2(Pb) = CaO2_max * Pb^n / (Pb^n + Pb50^n) + alpha_eff * Pb
   * is strictly increasing for Pb >= 0 (alpha_eff > 0), so every CaO2 >= 0 has exactly one root
   * Pb >= 0. We find it with Newton's method applied directly to g(Pb) = CaO2(Pb) - CaO2. Newton
   * can leave the physical range Pb > 0 (flat regions of the S-shaped curve give huge steps), in
   * which case we fall back to bisection, which cannot leave its bracket (a warning is written
   * every 100th time this happens). The FAD derivative
   * follows from dPb/dCaO2 = 1 / (dCaO2/dPb). Only first derivatives are supported (no nested
   * FAD).
   *
   * Requirements: alpha_eff > 0 and n >= 1
   *
   * @param Pb [out]: oxygen partial pressure
   * @param CaO2 [in]: oxygen concentration
   * @param CaO2_max [in]: maximum oxygen concentration
   * @param Pb50 [in]: partial pressure at 50% maximum oxygen concentration
   * @param n [in]: exponent in Hill equation (>= 1).
   * @param alpha_eff [in]: effective solubility of oxygen in blood (> 0)
   */
  template <typename T>
  void get_oxy_partial_pressure_from_concentration(T& Pb, const T& CaO2, const double& CaO2_max,
      const double& Pb50, const double& n, const double& alpha_eff)
  {
    const double target = Sacado::ScalarValue<T>::eval(CaO2);
    if (!std::isfinite(target) || target < 0.0)
      FOUR_C_THROW("invalid oxygen concentration {}", target);
    if (!(CaO2_max > 0.0) || !(Pb50 > 0.0) || !std::isfinite(CaO2_max) || !std::isfinite(Pb50))
      FOUR_C_THROW("CaO2_max and Pb50 must be positive and finite");

    const double Pb50_n = std::pow(Pb50, n);

    // forward relation CaO2(P) and its derivative dCaO2/dP (for P > 0)
    auto concentration = [&](double P)
    { return CaO2_max * std::pow(P, n) / (std::pow(P, n) + Pb50_n) + alpha_eff * P; };
    auto slope = [&](double P)
    {
      const double P_n = std::pow(P, n);
      return CaO2_max * n * std::pow(P, n - 1.0) * Pb50_n / ((P_n + Pb50_n) * (P_n + Pb50_n)) +
             alpha_eff;
    };

    // CaO2 = 0 has the exact root P = 0. The slope of CaO2(P) there is alpha_eff (n > 1) or
    // alpha_eff + CaO2_max / Pb50 (n = 1), so the derivative is always finite.
    if (target == 0.0)
    {
      Pb = (CaO2 - target) / (alpha_eff + (n == 1.0 ? CaO2_max / Pb50 : 0.0));
      return;
    }

    // Step 1: Newton method (fast). The result is only accepted if it is converged and physical
    // (P > 0); P = 0 is treated already above.
    double P = Pb50 * 2.0 * target / CaO2_max;
    bool valid = false;
    for (int i = 0; i < 20; ++i)
    {
      const double g = concentration(P) - target;
      if (std::abs(g) <= 1.0e-13 * target)
      {
        valid = true;
        break;
      }
      P -= g / slope(P);
      if (!std::isfinite(P) || P <= 0.0) break;
    }
    valid = valid && std::isfinite(P) && P > 0.0;

    // Step 2: fallback to bisection, which always finds the unique root P > 0. The dissolved part
    // alpha_eff * P alone reaches the target at P = target / alpha_eff, which bounds the root.
    if (!valid)
    {
      const unsigned long count = ++Internal::oxy_partial_pressure_bisection_counter;
      if (count % Internal::oxy_partial_pressure_bisection_warning_interval == 0)
      {
        std::cout << "WARNING: Newton for the oxygen partial pressure left the physical range "
                     "(Pb <= 0) or did not converge, bisection is used (occurrence "
                  << count << ", reported every "
                  << Internal::oxy_partial_pressure_bisection_warning_interval
                  << " occurrences, last CaO2 = " << target << ").\n";
      }
      double lower = 0.0;
      double upper = std::max(Pb50, target / alpha_eff);

      // bisect until the residual is below the Newton tolerance, the bracket cannot be refined any
      // further in floating point arithmetic, or the maximum number of iterations is reached
      double residual_abs = 0.0;
      for (int i = 0; i < 500; ++i)
      {
        P = 0.5 * (lower + upper);
        const double g = concentration(P) - target;
        residual_abs = std::abs(g);
        if (residual_abs <= 1.0e-13 * target || P <= lower || P >= upper) break;
        if (g < 0.0)
          lower = P;
        else
          upper = P;
      }

      // Roundoff can stall the residual slightly above the Newton tolerance, which is accepted
      // with a looser tolerance. Anything worse (e.g., NaN from overflow of P^n) is an error.
      if (!(residual_abs <= 1.0e-10 * target) || !std::isfinite(P) || !(P > 0.0))
      {
        FOUR_C_THROW(
            "Inversion of the oxygen partial pressure failed to converge (CaO2 = {}, CaO2_max = "
            "{}, Pb50 = {}, n = {}, alpha_eff = {}, last Pb = {}, relative residual = {}). Large "
            "values of n can cause overflow of Pb^n.",
            target, CaO2_max, Pb50, n, alpha_eff, P, residual_abs / target);
      }
    }

    // Step 3: carry the FAD derivative. One Newton step on T at the converged root yields the
    // implicit derivative dPb/dCaO2 = 1 / slope.
    Pb = P;
    const T Pb_n = std::pow(Pb, n);
    const T denominator = Pb50_n + Pb_n;
    const T residual = alpha_eff * Pb + CaO2_max * Pb_n / denominator - CaO2;
    const T derivative =
        alpha_eff + CaO2_max * n * Pb50_n * std::pow(Pb, n - 1.0) / (denominator * denominator);
    Pb -= residual / derivative;
  }
}  // namespace PoroPressureBased



FOUR_C_NAMESPACE_CLOSE

#endif
