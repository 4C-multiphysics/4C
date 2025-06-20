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
#include "4C_porofluid_pressure_based_elast_scatra_input.hpp"

#include <Sacado.hpp>

#include <cmath>
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
      MPI_Comm comm, const std::string& struct_disname, const std::string& fluid_disname,
      const std::string& scatra_disname, int& ndsporo_disp, int& ndsporo_vel,
      int& ndsporo_solidpressure, int& ndsporofluid_scatra, const bool artery_coupl);

  //! exchange material pointers of discretizations
  void assign_material_pointers_porofluid_elast_scatra(const std::string& struct_disname,
      const std::string& fluid_disname, const std::string& scatra_disname, const bool artery_coupl);

  //! create solution algorithm depending on input file
  std::shared_ptr<PorofluidElastScatraBaseAlgorithm> create_algorithm_porofluid_elast_scatra(
      SolutionSchemePorofluidElastScatra solscheme,  //!< solution scheme to build (i)
      const Teuchos::ParameterList& timeparams,      //!< problem parameters (i)
      MPI_Comm comm                                  //!< communicator(i)
  );

  //! create coupling strategy for coupling with 1D network depending on input file
  std::shared_ptr<PorofluidElastScatraArteryCouplingBaseAlgorithm>
  create_and_init_artery_coupling_strategy(std::shared_ptr<Core::FE::Discretization> arterydis,
      std::shared_ptr<Core::FE::Discretization> contdis,
      const Teuchos::ParameterList& meshtyingparams, const std::string& condname,
      const bool evaluate_on_lateral_surface);

  /**
   * Get oxygen partial pressure from oxygen concentration via numerical inversion
   * templated to FAD and double
   *
   * @param Pb [out]: oxygen partial pressure
   * @param CaO2 [in]: oxygen concentration
   * @param CaO2_max [in]: maximum oxygen concentration
   * @param Pb50 [in]: partial pressure at 50% maximum oxygen concentration
   * @param n [in]: exponent in Hill equation
   * @param alpha_eff [in]: effective solubility of oxygen in blood
   */
  template <typename T>
  void get_oxy_partial_pressure_from_concentration(T& Pb, const T& CaO2, const double& CaO2_max,
      const double& Pb50, const double& n, const double& alpha_eff)
  {
    // start value
    Pb = Pb50 * 2.0 * CaO2 / CaO2_max;

    bool converged = false;
    // Newton loop
    for (int i = 0; i < 20; i++)
    {
      // function
      T f = (std::pow(Pb50, n) + std::pow(Pb, n)) * CaO2 - CaO2_max * std::pow(Pb, n) -
            Pb * (std::pow(Pb, n) + std::pow(Pb50, n)) * alpha_eff;
      if (fabs(f) < 1.0e-10)
      {
        converged = true;
        break;
      }
      // deriv
      T dfdPb = n * std::pow(Pb, (n - 1)) * CaO2 - CaO2_max * n * std::pow(Pb, (n - 1)) -
                ((n + 1) * std::pow(Pb, n) + std::pow(Pb50, n)) * alpha_eff;
      // update
      Pb = Pb - f / dfdPb;
    }
    if (!converged)
    {
      FOUR_C_THROW("local Newton for computation of oxygen partial pressure unconverged");
    }
  }
}  // namespace PoroPressureBased



FOUR_C_NAMESPACE_CLOSE

#endif
