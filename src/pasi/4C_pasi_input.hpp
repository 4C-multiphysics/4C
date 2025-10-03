// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PASI_INPUT_HPP
#define FOUR_C_PASI_INPUT_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_io_input_spec.hpp"


FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | Input parameters for particle structure interaction                       |
 *---------------------------------------------------------------------------*/
namespace PaSI
{
  //! type of partitioned coupling
  enum PartitionedCouplingType
  {
    partitioned_onewaycoup,                 //!< one-way coupling
    partitioned_twowaycoup,                 //!< two-way coupling
    partitioned_twowaycoup_disprelax,       //!< two-way coupling with constant relaxation
    partitioned_twowaycoup_disprelaxaitken  //!< two-way coupling with dynamic aitken relaxation
  };

  //! valid parameters for particle structure interaction
  Core::IO::InputSpec valid_parameters();

}  // namespace PaSI

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
