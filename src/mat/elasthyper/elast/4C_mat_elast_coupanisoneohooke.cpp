// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_coupanisoneohooke.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_elast_aniso_structuraltensor_strategy.hpp"
#include "4C_material_parameter_base.hpp"
FOUR_C_NAMESPACE_OPEN


Mat::Elastic::PAR::CoupAnisoNeoHooke::CoupAnisoNeoHooke(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : ParameterAniso(matdata),
      c_(matdata.parameters.get<double>("C")),
      gamma_(matdata.parameters.get<double>("GAMMA")),
      init_(matdata.parameters.get<int>("INIT")),
      adapt_angle_(matdata.parameters.get<bool>("ADAPT_ANGLE"))
{
}

Mat::Elastic::CoupAnisoNeoHooke::CoupAnisoNeoHooke(Mat::Elastic::PAR::CoupAnisoNeoHooke* params)
    : params_(params)
{
}

void Mat::Elastic::CoupAnisoNeoHooke::pack_summand(Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, a_);
  add_to_pack(data, structural_tensor_);
}

void Mat::Elastic::CoupAnisoNeoHooke::unpack_summand(Core::Communication::UnpackBuffer& buffer)
{
  extract_from_pack(buffer, a_);
  extract_from_pack(buffer, structural_tensor_);
}

void Mat::Elastic::CoupAnisoNeoHooke::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  // warning message
  std::cout << "Material does not respect a stress free reference state" << std::endl;

  // path if fibers aren't given in input file
  if (params_->init_ == 0)
  {
    // fibers aligned in YZ-plane with gamma around Z in global cartesian cosy
    Core::LinAlg::Matrix<3, 3> Id(Core::LinAlg::Initialization::zero);
    for (int i = 0; i < 3; i++) Id(i, i) = 1.0;
    set_fiber_vecs(-1.0,
        Core::LinAlg::get_full(Core::LinAlg::TensorGenerators::identity<double, 3, 3>),
        Core::LinAlg::get_full(Core::LinAlg::TensorGenerators::identity<double, 3, 3>));
  }

  // path if fibers are given in input file
  else if (params_->init_ == 1)
  {
    // CIR-AXI-RAD nomenclature
    if (coord_system.has_value())
    {
      // Read in of data
      Core::LinAlg::Tensor<double, 3, 3> locsys{};
      read_rad_axi_cir(*coord_system, locsys);
      // final setup of fiber data
      set_fiber_vecs(0.0, locsys,
          Core::LinAlg::get_full(Core::LinAlg::TensorGenerators::identity<double, 3, 3>));
    }

    // FIBER1 nomenclature
    else if (fibers.element_fibers.size() > 0)
    {
      // Read in of fiber data and setting fiber data
      a_ = fibers.element_fibers[0];
      params_->structural_tensor_strategy()->setup_structural_tensor(a_, structural_tensor_);
    }

    // error path
    else
    {
      FOUR_C_THROW("Reading of element local cosy for anisotropic materials failed");
    }
  }
  else
    FOUR_C_THROW("INIT mode not implemented");
}

void Mat::Elastic::CoupAnisoNeoHooke::add_stress_aniso_principal(
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& rcg,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress, const Teuchos::ParameterList& params,
    const int gp, const int eleGID)
{
  double c = params_->c_;

  double gamma = 2. * c;
  stress += gamma * structural_tensor_;

  // no contribution to cmat
  // double delta = 0.0;
  // cmat.multiply_nt(delta, A_, A_, 1.0);
}

void Mat::Elastic::CoupAnisoNeoHooke::get_fiber_vecs(
    std::vector<Core::LinAlg::Tensor<double, 3>>& fibervecs  ///< vector of all fiber vectors
) const
{
  fibervecs.push_back(a_);
}

void Mat::Elastic::CoupAnisoNeoHooke::set_fiber_vecs(const double newgamma,
    const Core::LinAlg::Tensor<double, 3, 3>& locsys,
    const Core::LinAlg::Tensor<double, 3, 3>& defgrd)
{
  if ((params_->gamma_ < -90) || (params_->gamma_ > 90))
    FOUR_C_THROW("Fiber angle not in [-90,90]");
  // convert
  double gamma = (params_->gamma_ * std::numbers::pi) / 180.;

  if (params_->adapt_angle_ && newgamma != -1.0)
  {
    if (gamma * newgamma < 0.0)
      gamma = -1.0 * newgamma;
    else
      gamma = newgamma;
  }

  Core::LinAlg::Tensor<double, 3> ca{};
  for (int i = 0; i < 3; ++i)
  {
    // a = cos gamma e3 + sin gamma e2
    ca(i) = cos(gamma) * locsys(i, 2) + sin(gamma) * locsys(i, 1);
  }
  // pull back in reference configuration
  Core::LinAlg::Tensor<double, 3> a_0{};
  Core::LinAlg::Tensor<double, 3, 3> idefgrd = Core::LinAlg::inv(defgrd);

  a_0 = idefgrd * ca;
  a_ = 1.0 / Core::LinAlg::norm2(a_0) * a_0;

  params_->structural_tensor_strategy()->setup_structural_tensor(a_, structural_tensor_);
}
FOUR_C_NAMESPACE_CLOSE
