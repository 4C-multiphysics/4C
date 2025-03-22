// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beam3_euler_bernoulli.hpp"
#include "4C_legacy_enum_definitions_materials.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_material_base.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool Discret::Elements::Beam3eb::read_element(const std::string& eletype,
    const std::string& distype, const Core::IO::InputParameterContainer& container)
{
  // read number of material model
  int material_id = container.get<int>("MAT");
  set_material(0, Mat::factory(material_id));

  const auto mat_type = material()->parameter()->type();
  FOUR_C_ASSERT_ALWAYS(
      mat_type == Core::Materials::m_beam_kirchhoff_torsionfree_elast_hyper ||
          mat_type == Core::Materials::m_beam_kirchhoff_torsionfree_elast_hyper_bymodes,
      "The material parameter definition '{}' is not supported by Beam3eb element! "
      "Choose MAT_BeamKirchhoffTorsionFreeElastHyper or "
      "MAT_BeamKirchhoffTorsionFreeElastHyper_ByModes!",
      mat_type);

  return true;
}

FOUR_C_NAMESPACE_CLOSE
