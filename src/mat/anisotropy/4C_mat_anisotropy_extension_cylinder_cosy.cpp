// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_anisotropy_extension_cylinder_cosy.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_parobject.hpp"
#include "4C_mat_anisotropy.hpp"
#include "4C_mat_anisotropy_coordinate_system_provider.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_shared_ptr_from_ref.hpp"

FOUR_C_NAMESPACE_OPEN

Mat::CylinderCoordinateSystemAnisotropyExtension::CylinderCoordinateSystemAnisotropyExtension()
    : cosy_location_(CosyLocation::None)
{
}

void Mat::CylinderCoordinateSystemAnisotropyExtension::pack_anisotropy(
    Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, cosy_location_);
  add_to_pack(data, element_cosy_);
  add_to_pack(data, gp_cosy_);
}

void Mat::CylinderCoordinateSystemAnisotropyExtension::unpack_anisotropy(
    Core::Communication::UnpackBuffer& buffer)
{
  extract_from_pack(buffer, cosy_location_);
  extract_from_pack(buffer, element_cosy_);
  extract_from_pack(buffer, gp_cosy_);
}

void Mat::CylinderCoordinateSystemAnisotropyExtension::on_global_data_initialized(
    Mat::Anisotropy& anisotropy)
{
  if (anisotropy.has_gp_cylinder_coordinate_system())
  {
    cosy_location_ = CosyLocation::GPCosy;
    gp_cosy_.clear();
    for (int gp = 0; gp < anisotropy.get_number_of_gauss_points(); ++gp)
    {
      gp_cosy_.push_back(anisotropy.get_gp_cylinder_coordinate_system(gp));
    }
  }
  else if (anisotropy.has_element_cylinder_coordinate_system())
  {
    cosy_location_ = CosyLocation::ElementCosy;
    element_cosy_ = anisotropy.get_element_cylinder_coordinate_system();
  }
  else
  {
    cosy_location_ = CosyLocation::None;
  }
}

void Mat::CylinderCoordinateSystemAnisotropyExtension::on_global_element_data_initialized(
    Mat::Anisotropy& anisotropy)
{
  (void)anisotropy;
}

void Mat::CylinderCoordinateSystemAnisotropyExtension::on_global_gp_data_initialized(
    Mat::Anisotropy& anisotropy)
{
  (void)anisotropy;
}

const Mat::CylinderCoordinateSystemProvider&
Mat::CylinderCoordinateSystemAnisotropyExtension::get_cylinder_coordinate_system(int gp) const
{
  if (cosy_location_ == CosyLocation::None)
  {
    FOUR_C_THROW("No cylinder coordinate system defined!");
  }

  if (cosy_location_ == CosyLocation::ElementCosy)
  {
    FOUR_C_ASSERT(element_cosy_.has_value(),
        "Invalid state: CosyLocation=ElementCosy, but element_cosy_ is empty. "
        "Check initialization in on_global_data_initialized().");

    return *element_cosy_;
  }

  return gp_cosy_.at(gp);
}

std::shared_ptr<Mat::CoordinateSystemProvider>
Mat::CylinderCoordinateSystemAnisotropyExtension::get_coordinate_system_provider(int gp) const
{
  auto cosy = std::make_shared<CoordinateSystemHolder>();

  if (cosy_location_ != CosyLocation::None)
    cosy->set_cylinder_coordinate_system_provider(
        Core::Utils::shared_ptr_from_ref(get_cylinder_coordinate_system(gp)));

  return cosy;
}
FOUR_C_NAMESPACE_CLOSE
