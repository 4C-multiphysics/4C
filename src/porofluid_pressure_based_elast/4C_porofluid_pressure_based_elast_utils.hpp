// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_UTILS_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_UTILS_HPP

#include "4C_config.hpp"

#include "4C_linalg_vector.hpp"
#include "4C_porofluid_pressure_based_elast_algorithm_dependencies.hpp"
#include "4C_porofluid_pressure_based_elast_base.hpp"
#include "4C_porofluid_pressure_based_elast_input.hpp"

#include <memory>
#include <set>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace PoroPressureBased
{
  /// setup discretizations and dofsets of porofluid and structure
  void setup_discretizations_and_field_coupling_porofluid_elast(
      const PorofluidElastAlgorithmDeps& algorithm_deps, const std::string& struct_disname,
      const std::string& fluid_disname, int& nds_disp, int& nds_vel, int& nds_solidpressure);

  /// setup discretizations and dofsets artery
  std::map<int, std::set<int>> setup_discretizations_and_field_coupling_artery(
      const PorofluidElastAlgorithmDeps& algorithm_deps, const std::string& struct_disname);

  //! exchange material pointers of both discretizations
  void assign_material_pointers_porofluid_elast(const PorofluidElastAlgorithmDeps& algorithm_deps,
      const std::string& struct_disname, const std::string& fluid_disname);

  /// create solution algorithm depending on input file
  std::shared_ptr<PorofluidElastAlgorithm> create_algorithm_porofluid_elast(
      SolutionSchemePorofluidElast solscheme,     //!< solution scheme to build (i)
      const Teuchos::ParameterList& timeparams,   //!< problem parameters (i)
      MPI_Comm comm,                              //!< communicator(i)
      PorofluidElastAlgorithmDeps algorithm_deps  //!< algorithm dependencies
  );
}  // namespace PoroPressureBased



FOUR_C_NAMESPACE_CLOSE

#endif
