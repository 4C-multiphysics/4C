// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beam3_kirchhoff.hpp"
#include "4C_fem_general_largerotations.hpp"
#include "4C_legacy_enum_definitions_materials.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_material_base.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN


/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
bool Discret::Elements::Beam3k::read_element(const std::string& eletype, const std::string& distype,
    const Core::IO::InputParameterContainer& container)
{
  // read number of material model and cross-sections specs
  int material_id = container.get<int>("MAT");
  set_material(0, Mat::factory(material_id));

  const auto mat_type = material()->parameter()->type();
  FOUR_C_ASSERT_ALWAYS(mat_type == Core::Materials::m_beam_kirchhoff_elast_hyper ||
                           mat_type == Core::Materials::m_beam_kirchhoff_elast_hyper_bymodes,
      "The material parameter definition '{}' is not supported by Beam3k element! "
      "Choose MAT_BeamKirchhoffElastHyper or MAT_BeamKirchhoffElastHyper_ByModes!",
      mat_type);

  rotvec_ = container.get<Beam3KirchhoffParametrizationType>("PARAMETRIZATION") ==
            Beam3KirchhoffParametrizationType::rot;
  weakkirchhoff_ = container.get<Beam3KirchhoffConstraintType>("CONSTRAINT") ==
                   Beam3KirchhoffConstraintType::weak;

  if (weakkirchhoff_)
  {
#ifdef CONSISTENTSPINSK
    FOUR_C_THROW(
        "The flag CONSISTENTSPINSK is only possible for strong Kirchhoff constraint enforcement "
        "(weakkirchhoff_=false)");
#endif
  }

  // extract triads at element nodes in reference configuration as rotation vectors and save them as
  // quaternions at each node, respectively
  auto nodal_thetas = container.get<std::vector<double>>("TRIADS");

  this->set_up_initial_rotations(nodal_thetas);

  // read whether automatic differentiation via Sacado::Fad package shall be used
  use_fad_ = container.get<bool>("USE_FAD");

  return true;
}

FOUR_C_NAMESPACE_CLOSE
