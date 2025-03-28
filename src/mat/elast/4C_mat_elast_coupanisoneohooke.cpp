// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_coupanisoneohooke.hpp"

#include "4C_comm_pack_helpers.hpp"
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

void Mat::Elastic::CoupAnisoNeoHooke::setup(
    int numgp, const Core::IO::InputParameterContainer& container)
{
  // warning message
  std::cout << "Material does not respect a stress free reference state" << std::endl;

  // path if fibers aren't given in input file
  if (params_->init_ == 0)
  {
    // fibers aligned in YZ-plane with gamma around Z in global cartesian cosy
    Core::LinAlg::Matrix<3, 3> Id(Core::LinAlg::Initialization::zero);
    for (int i = 0; i < 3; i++) Id(i, i) = 1.0;
    set_fiber_vecs(-1.0, Id, Id);
  }

  // path if fibers are given in input file
  else if (params_->init_ == 1)
  {
    // CIR-AXI-RAD nomenclature
    if (container.get<std::optional<std::vector<double>>>("RAD").has_value() and
        container.get<std::optional<std::vector<double>>>("AXI").has_value() and
        container.get<std::optional<std::vector<double>>>("CIR").has_value())
    {
      // Read in of data
      Core::LinAlg::Matrix<3, 3> locsys(Core::LinAlg::Initialization::zero);
      read_rad_axi_cir(container, locsys);
      Core::LinAlg::Matrix<3, 3> Id(Core::LinAlg::Initialization::zero);
      for (int i = 0; i < 3; i++) Id(i, i) = 1.0;
      // final setup of fiber data
      set_fiber_vecs(0.0, locsys, Id);
    }

    // FIBER1 nomenclature
    else if (container.get<std::optional<std::vector<double>>>("FIBER1").has_value())
    {
      // Read in of fiber data and setting fiber data
      read_fiber(container, "FIBER1", a_);
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
    const Core::LinAlg::Matrix<6, 1>& rcg, Core::LinAlg::Matrix<6, 6>& cmat,
    Core::LinAlg::Matrix<6, 1>& stress, Teuchos::ParameterList& params, const int gp,
    const int eleGID)
{
  double c = params_->c_;

  double gamma = 2. * c;
  stress.update(gamma, structural_tensor_, 1.0);

  // no contribution to cmat
  // double delta = 0.0;
  // cmat.multiply_nt(delta, A_, A_, 1.0);
}

void Mat::Elastic::CoupAnisoNeoHooke::get_fiber_vecs(
    std::vector<Core::LinAlg::Matrix<3, 1>>& fibervecs  ///< vector of all fiber vectors
) const
{
  fibervecs.push_back(a_);
}

void Mat::Elastic::CoupAnisoNeoHooke::set_fiber_vecs(const double newgamma,
    const Core::LinAlg::Matrix<3, 3>& locsys, const Core::LinAlg::Matrix<3, 3>& defgrd)
{
  if ((params_->gamma_ < -90) || (params_->gamma_ > 90))
    FOUR_C_THROW("Fiber angle not in [-90,90]");
  // convert
  double gamma = (params_->gamma_ * M_PI) / 180.;

  if (params_->adapt_angle_ && newgamma != -1.0)
  {
    if (gamma * newgamma < 0.0)
      gamma = -1.0 * newgamma;
    else
      gamma = newgamma;
  }

  Core::LinAlg::Matrix<3, 1> ca(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; ++i)
  {
    // a = cos gamma e3 + sin gamma e2
    ca(i) = cos(gamma) * locsys(i, 2) + sin(gamma) * locsys(i, 1);
  }
  // pull back in reference configuration
  Core::LinAlg::Matrix<3, 1> a_0(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> idefgrd(Core::LinAlg::Initialization::zero);
  idefgrd.invert(defgrd);

  a_0.multiply(idefgrd, ca);
  a_.update(1. / a_0.norm2(), a_0);

  params_->structural_tensor_strategy()->setup_structural_tensor(a_, structural_tensor_);
}
FOUR_C_NAMESPACE_CLOSE
