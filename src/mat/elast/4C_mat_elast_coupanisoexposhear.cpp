// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_coupanisoexposhear.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_mat_elast_aniso_structuraltensor_strategy.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN


Mat::Elastic::CoupAnisoExpoShearAnisotropyExtension::CoupAnisoExpoShearAnisotropyExtension(
    const int init_mode, const std::array<int, 2> fiber_ids)
    : init_mode_(init_mode), fiber_ids_(fiber_ids)
{
}

void Mat::Elastic::CoupAnisoExpoShearAnisotropyExtension::pack_anisotropy(
    Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, scalar_products_);
  add_to_pack(data, structural_tensors_stress_);
  add_to_pack(data, structural_tensors_);
  add_to_pack(data, is_initialized_);
}

void Mat::Elastic::CoupAnisoExpoShearAnisotropyExtension::unpack_anisotropy(
    Core::Communication::UnpackBuffer& buffer)
{
  extract_from_pack(buffer, scalar_products_);
  extract_from_pack(buffer, structural_tensors_stress_);
  extract_from_pack(buffer, structural_tensors_);
  extract_from_pack(buffer, is_initialized_);
}

double Mat::Elastic::CoupAnisoExpoShearAnisotropyExtension::get_scalar_product(int gp) const
{
  if (!is_initialized_)
  {
    FOUR_C_THROW("Fibers have not been initialized yet.");
  }

  if (scalar_products_.size() == 1)
  {
    return scalar_products_[0];
  }

  return scalar_products_[gp];
}

const Core::LinAlg::Matrix<3, 3>&
Mat::Elastic::CoupAnisoExpoShearAnisotropyExtension::get_structural_tensor(int gp) const
{
  if (!is_initialized_)
  {
    FOUR_C_THROW("Fibers have not been initialized yet.");
  }

  if (structural_tensors_.size() == 1)
  {
    return structural_tensors_[0];
  }

  return structural_tensors_[gp];
}

const Core::LinAlg::Matrix<6, 1>&
Mat::Elastic::CoupAnisoExpoShearAnisotropyExtension::get_structural_tensor_stress(int gp) const
{
  if (!is_initialized_)
  {
    FOUR_C_THROW("Fibers have not been initialized yet.");
  }

  if (structural_tensors_stress_.size() == 1)
  {
    return structural_tensors_stress_[0];
  }

  return structural_tensors_stress_[gp];
}

void Mat::Elastic::CoupAnisoExpoShearAnisotropyExtension::on_global_data_initialized()
{
  // do nothing
}

void Mat::Elastic::CoupAnisoExpoShearAnisotropyExtension::on_global_element_data_initialized()
{
  if (init_mode_ == DefaultAnisotropyExtension<2>::INIT_MODE_NODAL_EXTERNAL ||
      init_mode_ == DefaultAnisotropyExtension<2>::INIT_MODE_NODAL_FIBERS)
  {
    // this is the initialization method for element fibers, so do nothing here
    return;
  }

  if (init_mode_ == DefaultAnisotropyExtension<2>::INIT_MODE_ELEMENT_EXTERNAL)
  {
    FOUR_C_THROW(
        "This material only supports the fiber prescription with the FIBER1 FIBER2 notation and "
        "INIT modes {} and {}.",
        DefaultAnisotropyExtension<2>::INIT_MODE_ELEMENT_FIBERS,
        DefaultAnisotropyExtension<2>::INIT_MODE_NODAL_FIBERS);
  }

  if (get_anisotropy()->get_element_fibers().empty())
  {
    FOUR_C_THROW("No element fibers are given with the FIBER1 FIBER2 notation");
  }

  scalar_products_.resize(1);
  structural_tensors_.resize(1);
  structural_tensors_stress_.resize(1);
  scalar_products_[0] = get_anisotropy()
                            ->get_element_fiber(fiber_ids_[0])
                            .dot(get_anisotropy()->get_element_fiber(fiber_ids_[1]));

  Core::LinAlg::Matrix<3, 3> fiber1fiber2T(Core::LinAlg::Initialization::uninitialized);
  fiber1fiber2T.multiply_nt(get_anisotropy()->get_element_fiber(fiber_ids_[0]),
      get_anisotropy()->get_element_fiber(fiber_ids_[1]));

  structural_tensors_[0].update(0.5, fiber1fiber2T);
  structural_tensors_[0].update_t(0.5, fiber1fiber2T, 1.0);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(
      structural_tensors_[0], structural_tensors_stress_[0]);

  is_initialized_ = true;
}

void Mat::Elastic::CoupAnisoExpoShearAnisotropyExtension::on_global_gp_data_initialized()
{
  if (init_mode_ == DefaultAnisotropyExtension<2>::INIT_MODE_ELEMENT_EXTERNAL ||
      init_mode_ == DefaultAnisotropyExtension<2>::INIT_MODE_ELEMENT_FIBERS)
  {
    // this is the initialization method for nodal fibers, so do nothing here
    return;
  }

  if (init_mode_ == DefaultAnisotropyExtension<2>::INIT_MODE_NODAL_EXTERNAL)
  {
    FOUR_C_THROW(
        "This material only supports the fiber prescription with the FIBER1 FIBER2 notation and "
        "INIT modes {} and {}.",
        DefaultAnisotropyExtension<2>::INIT_MODE_ELEMENT_FIBERS,
        DefaultAnisotropyExtension<2>::INIT_MODE_NODAL_FIBERS);
  }

  if (get_anisotropy()->get_number_of_gauss_point_fibers() == 0)
  {
    FOUR_C_THROW("No element fibers are given with the FIBER1 FIBER2 notation");
  }

  scalar_products_.resize(get_anisotropy()->get_number_of_gauss_points());
  structural_tensors_.resize(get_anisotropy()->get_number_of_gauss_points());
  structural_tensors_stress_.resize(get_anisotropy()->get_number_of_gauss_points());

  for (auto gp = 0; gp < get_anisotropy()->get_number_of_gauss_points(); ++gp)
  {
    scalar_products_[gp] = get_anisotropy()
                               ->get_gauss_point_fiber(gp, fiber_ids_[0])
                               .dot(get_anisotropy()->get_gauss_point_fiber(gp, fiber_ids_[1]));

    Core::LinAlg::Matrix<3, 3> fiber1fiber2T(Core::LinAlg::Initialization::uninitialized);
    fiber1fiber2T.multiply_nt(get_anisotropy()->get_gauss_point_fiber(gp, fiber_ids_[0]),
        get_anisotropy()->get_gauss_point_fiber(gp, fiber_ids_[1]));

    structural_tensors_[gp].update(0.5, fiber1fiber2T);
    structural_tensors_[gp].update_t(0.5, fiber1fiber2T, 1.0);
    Core::LinAlg::Voigt::Stresses::matrix_to_vector(
        structural_tensors_[gp], structural_tensors_stress_[gp]);
  }

  is_initialized_ = true;
}

Mat::Elastic::PAR::CoupAnisoExpoShear::CoupAnisoExpoShear(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Core::Mat::PAR::Parameter(matdata), Mat::Elastic::PAR::CoupAnisoExpoBase(matdata)
{
  std::copy_n(matdata.parameters.get<std::vector<int>>("FIBER_IDS").begin(), 2, fiber_id_.begin());

  for (int& i : fiber_id_) i -= 1;
}

Mat::Elastic::CoupAnisoExpoShear::CoupAnisoExpoShear(Mat::Elastic::PAR::CoupAnisoExpoShear* params)
    : Mat::Elastic::CoupAnisoExpoBase(params),
      params_(params),
      anisotropy_extension_(params_->init_, params->fiber_id_)
{
}

void Mat::Elastic::CoupAnisoExpoShear::register_anisotropy_extensions(Mat::Anisotropy& anisotropy)
{
  anisotropy.register_anisotropy_extension(anisotropy_extension_);
}

void Mat::Elastic::CoupAnisoExpoShear::pack_summand(Core::Communication::PackBuffer& data) const
{
  anisotropy_extension_.pack_anisotropy(data);
}

void Mat::Elastic::CoupAnisoExpoShear::unpack_summand(Core::Communication::UnpackBuffer& buffer)
{
  anisotropy_extension_.unpack_anisotropy(buffer);
}

void Mat::Elastic::CoupAnisoExpoShear::get_fiber_vecs(
    std::vector<Core::LinAlg::Matrix<3, 1>>& fibervecs) const
{
  // no fibers to export here
}

void Mat::Elastic::CoupAnisoExpoShear::set_fiber_vecs(const double newgamma,
    const Core::LinAlg::Matrix<3, 3>& locsys, const Core::LinAlg::Matrix<3, 3>& defgrd)
{
  FOUR_C_THROW("This function is not implemented for this summand!");
}

FOUR_C_NAMESPACE_CLOSE
