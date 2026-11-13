// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_RESTART_HPP
#define FOUR_C_REDUCED_LUNG_RESTART_HPP

#include "4C_config.hpp"

#include <map>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}

namespace Core::IO
{
  class DiscretizationReader;
  class DiscretizationWriter;
}  // namespace Core::IO

namespace Core::LinAlg
{
  template <typename T>
  class Vector;
}

namespace ReducedLung::Airways
{
  struct AirwayContainer;
}

namespace ReducedLung::TerminalUnits
{
  struct TerminalUnitContainer;
}

namespace ReducedLung::Restart
{
  /**
   * @brief The parts of a reduced-lung model that carry state from one time step to the next.
   *
   * Everything else, e.g. junctions, boundary conditions, and quantities the state updaters derive
   * from the dofs, is rebuilt from the input or recomputed in the next solve.
   */
  struct State
  {
    const Core::FE::Discretization& discretization;
    Airways::AirwayContainer& airways;
    TerminalUnits::TerminalUnitContainer& terminal_units;
    //! First global dof id and number of dofs of every element, by global element id.
    const std::map<int, int>& first_global_dof_of_ele;
    const std::map<int, int>& global_dof_per_ele;
    //! Dofs of the accepted state, including the ghosted ones.
    Core::LinAlg::Vector<double>& locally_relevant_dofs;
    //! Owned dofs, i.e. the solution vector the nonlinear solver starts from.
    Core::LinAlg::Vector<double>& x;
  };

  /**
   * @brief Write the accepted state into the current step of @p writer.
   *
   * All data is stored per element, keyed by the global element id, so a restart may use a
   * different number of ranks or a different partitioning.
   */
  void write(Core::IO::DiscretizationWriter& writer, const State& state);

  /**
   * @brief Restore the accepted state written by write().
   */
  void read(Core::IO::DiscretizationReader& reader, State& state);
}  // namespace ReducedLung::Restart

FOUR_C_NAMESPACE_CLOSE

#endif
