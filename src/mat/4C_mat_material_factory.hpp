// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_MATERIAL_FACTORY_HPP
#define FOUR_C_MAT_MATERIAL_FACTORY_HPP

#include "4C_config.hpp"

#include "4C_comm_parobject.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_material_base.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN


/// MAT: materials
namespace Mat
{
  const int NUM_STRESS_3D = 6;  ///< 6 stresses for 3D

  /// create element material object given the number of a material definition
  std::shared_ptr<Core::Mat::Material> factory(int matnum  ///< material ID
  );

  /**
   * Create material parameter object from @p input_data. This function maps the material @p type
   * to a statically known material parameter class.
   */
  std::unique_ptr<Core::Mat::PAR::Parameter> make_parameter(int id,
      Core::Materials::MaterialType type, const Core::IO::InputParameterContainer& input_data);

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
