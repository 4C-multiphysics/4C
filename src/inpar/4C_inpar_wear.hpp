// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_INPAR_WEAR_HPP
#define FOUR_C_INPAR_WEAR_HPP

#include "4C_config.hpp"

#include "4C_io_input_spec.hpp"


FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
namespace Inpar
{
  namespace Wear
  {
    /// Type of contact wear law
    /// (this enum represents the input file parameter WEAR)
    enum WearLaw
    {
      wear_none,    ///< no wear
      wear_archard  ///< Archard wear law
    };

    /// Definition of contact wear surface
    /// (this enum represents the input file parameter WEAR_SIDE)
    enum WearSide
    {
      wear_slave,  ///< wear on slave side
      wear_both    ///< slave and master wear
    };

    /// Definition of contact wear algorithm
    /// (this enum represents the input file parameter WEARTYPE)
    enum WearType
    {
      wear_intstate,  ///< internal state variable approach for wear
      wear_primvar    ///< primary variable approach for wear
    };

    /// Definition of wear time integration
    /// (this enum represents the input file parameter WEARTIMINT)
    enum WearTimInt
    {
      wear_expl,  ///< implicit time integration
      wear_impl   ///< explicit time integration
    };

    /// Definition of wear shape functions (necessary for prim. var. approach)
    /// (this enum represents the input file parameter WEAR_SHAPEFCN)
    enum WearShape
    {
      wear_shape_dual,     ///< dual shape functions allowing for condensation
      wear_shape_standard  ///< std. shape functions
    };

    /// Definition of wear-ALE time scale coupling algorithm
    /// (this enum represents the input file parameter WEAR_TIMESCALE)
    enum WearTimeScale
    {
      wear_time_equal,     ///< shape evolution step after each structural step
      wear_time_different  ///< shape evolution for accumulated wear after predefined structural
                           ///< steps
    };

    /// wear parameters
    Core::IO::InputSpec valid_parameters();
  }  // namespace Wear
}  // namespace Inpar

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
