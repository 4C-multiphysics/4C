// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_INPUT_HPP
#define FOUR_C_REDUCED_LUNG_INPUT_HPP

#include "4C_config.hpp"

#include "4C_io_input_spec.hpp"

#include <map>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  namespace IO
  {
    /**
     * Enum class for the reduced lung dynamics.
     */
    enum class ReducedLungDyn
    {
      BackwardEuler,
    };

    /**
     * Enum to distinguish between airway and terminal unit elements in the reduced
     * lung implementation.
     */
    enum class ElementType
    {
      Airway,
      TerminalUnit,
    };

    /**
     * Enum to distinguish between different airway models in the reduced lung implementation.
     */
    enum class AirwayModel
    {
      Resistive,
      ViscoelasticRLC,
    };

    /**
     * Enum to distinguish between different airway wall models in the reduced lung implementation.
     */
    enum class WallModel
    {
      StandardWallModel
    };

    /**
     * Enum to distinguish between different rheological models for the terminal units in the
     * reduced lung implementation.
     */
    enum class RheologicalModel
    {
      KelvinVoigt,
      FourElementMaxwell,
    };

    /**
     * Enum to distinguish between different elasticity models for the terminal units in the reduced
     * lung implementation.
     */
    enum class ElasticityModel
    {
      Linear,
      Ogden
    };
  }  // namespace IO

  /// set the reduced airways parameters
  void set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list);

}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

#endif