// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_anisotropy_cylinder_coordinate_system_manager.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_parobject.hpp"
#include "4C_io_linedefinition.hpp"
#include "4C_mat_anisotropy.hpp"
#include "4C_mat_anisotropy_utils.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

Mat::CylinderCoordinateSystemManager::CylinderCoordinateSystemManager() = default;

void Mat::CylinderCoordinateSystemManager::pack(Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, radial_);
  add_to_pack(data, axial_);
  add_to_pack(data, circumferential_);
  add_to_pack(data, static_cast<int>(is_defined_));
}

void Mat::CylinderCoordinateSystemManager::unpack(Core::Communication::UnpackBuffer& buffer)
{
  extract_from_pack(buffer, radial_);
  extract_from_pack(buffer, axial_);
  extract_from_pack(buffer, circumferential_);
  is_defined_ = static_cast<bool>(extract_int(buffer));
}

void Mat::CylinderCoordinateSystemManager::read_from_element_line_definition(
    const Core::IO::InputParameterContainer& container)
{
  if (container.get_if<std::vector<double>>("RAD") != nullptr and
      container.get_if<std::vector<double>>("AXI") != nullptr and
      container.get_if<std::vector<double>>("CIR") != nullptr)
  {
    read_anisotropy_fiber(container, "RAD", radial_);
    read_anisotropy_fiber(container, "AXI", axial_);
    read_anisotropy_fiber(container, "CIR", circumferential_);
    is_defined_ = true;
  }
}

void Mat::CylinderCoordinateSystemManager::evaluate_local_coordinate_system(
    Core::LinAlg::Matrix<3, 3>& cosy) const
{
  for (int i = 0; i < 3; ++i)
  {
    cosy(i, 0) = get_rad()(i);
    cosy(i, 1) = get_axi()(i);
    cosy(i, 2) = get_cir()(i);
  }
}

const Mat::CylinderCoordinateSystemManager&
Mat::Anisotropy::get_element_cylinder_coordinate_system() const
{
  return element_cylinder_coordinate_system_manager_.value();
}

const Mat::CylinderCoordinateSystemManager& Mat::Anisotropy::get_gp_cylinder_coordinate_system(
    const int gp) const
{
  return gp_cylinder_coordinate_system_managers_[gp];
}
FOUR_C_NAMESPACE_CLOSE
