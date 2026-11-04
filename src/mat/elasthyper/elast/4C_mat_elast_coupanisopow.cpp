// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_coupanisopow.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_elast_aniso_structuraltensor_strategy.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

Mat::Elastic::PAR::CoupAnisoPow::CoupAnisoPow(const Core::Mat::PAR::Parameter::Data& matdata)
    : ParameterAniso(matdata),
      k_(matdata.parameters.get<double>("K")),
      d1_(matdata.parameters.get<double>("D1")),
      d2_(matdata.parameters.get<double>("D2")),
      fibernumber_(matdata.parameters.get<int>("FIBER")),
      activethres_(matdata.parameters.get<double>("ACTIVETHRES")),
      gamma_(matdata.parameters.get<double>("GAMMA")),
      init_(matdata.parameters.get<int>("INIT")),
      adapt_angle_(matdata.parameters.get<bool>("ADAPT_ANGLE"))
{
}

Mat::Elastic::CoupAnisoPow::CoupAnisoPow(Mat::Elastic::PAR::CoupAnisoPow* params) : params_(params)
{
}

void Mat::Elastic::CoupAnisoPow::pack_summand(Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, a_);
  add_to_pack(data, structural_tensor_);
}

void Mat::Elastic::CoupAnisoPow::unpack_summand(Core::Communication::UnpackBuffer& buffer)
{
  extract_from_pack(buffer, a_);
  extract_from_pack(buffer, structural_tensor_);
}

void Mat::Elastic::CoupAnisoPow::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  // path if fibers aren't given in input file
  if (params_->init_ == 0)
  {
    // fibers aligned in YZ-plane with gamma around Z in global cartesian cosy
    constexpr auto id =
        Core::LinAlg::get_full(Core::LinAlg::TensorGenerators::identity<double, 3, 3>);
    set_fiber_vecs(-1.0, id, id);
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
      constexpr auto id =
          Core::LinAlg::get_full(Core::LinAlg::TensorGenerators::identity<double, 3, 3>);
      // final setup of fiber data
      set_fiber_vecs(0.0, locsys, id);
    }
    // FIBERi nomenclature
    else if (fibers.element_fibers.size() > 0)
    {
      // Read in of data
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

void Mat::Elastic::CoupAnisoPow::add_stress_aniso_principal(
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& rcg,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress, const Teuchos::ParameterList& params,
    const int gp, const int eleGID)
{
  // load params
  double k = params_->k_;
  double d1 = params_->d1_;
  double d2 = params_->d2_;
  double activethres = params_->activethres_;

  if (d2 <= 1.0)
  {
    FOUR_C_THROW(
        "exponential factor D2 should be greater than 1.0, since otherwise one can't achieve a "
        "stress free reference state");
  }

  // calc invariant I4
  double I4 = 0.0;
  I4 = Core::LinAlg::ddot(structural_tensor_, rcg);

  double lambda4 = pow(I4, 0.5);
  double pow_I4_d1 = pow(I4, d1);
  double pow_I4_d1m1 = pow(I4, d1 - 1.0);
  double pow_I4_d1m2 = pow(I4, d1 - 2.0);
  // Compute stress and material update
  // Beware that the fiber will be turned off in case of compression under activethres.
  // Hence, some compression (i.e. activethres<1.0) could be allow since the fibers are embedded in
  // the matrix and at usually at the microscale not fibers are allowed in the same given direction
  // by FIBER1
  double gamma = 0.0;
  double delta = 0.0;
  if (lambda4 > activethres)
  {
    // Coefficient for residual
    if (pow_I4_d1 > 1.0)
    {
      gamma = 2.0 * k * d2 * d1 * pow_I4_d1m1 * pow(pow_I4_d1 - 1.0, d2 - 1.0);
      // Coefficient for matrix
      delta = 4.0 * k * d2 * (d2 - 1) * d1 * pow_I4_d1m1 * d1 * pow_I4_d1m1 *
                  pow(pow_I4_d1 - 1.0, d2 - 2.0) +
              4.0 * k * d2 * d1 * (d1 - 1.0) * pow_I4_d1m2 * pow(pow_I4_d1 - 1.0, d2 - 1.0);
    }
    else
    {
      gamma = -2.0 * k * d2 * d1 * pow_I4_d1m1 *
              pow(1.0 - pow_I4_d1, d2 - 1.0);  // Note minus sign at the beginning
      // Coefficient for matrix
      delta = 4.0 * k * d2 * (d2 - 1) * d1 * pow_I4_d1m1 * d1 * pow_I4_d1m1 *
                  pow(1.0 - pow_I4_d1, d2 - 2.0) -  // Note minus sign
              4.0 * k * d2 * d1 * (d1 - 1.0) * pow_I4_d1m2 * pow(1.0 - pow_I4_d1, d2 - 1.0);
    }
  }
  stress += gamma * structural_tensor_;
  cmat += delta * Core::LinAlg::dyadic(structural_tensor_, structural_tensor_);
}

void Mat::Elastic::CoupAnisoPow::get_fiber_vecs(
    std::vector<Core::LinAlg::Tensor<double, 3>>& fibervecs  ///< vector of all fiber vectors
) const
{
  fibervecs.push_back(a_);
}

void Mat::Elastic::CoupAnisoPow::set_fiber_vecs(const double newgamma,
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
  Core::LinAlg::Tensor<double, 3, 3> idefgrd = Core::LinAlg::inv(defgrd);

  Core::LinAlg::Tensor<double, 3> a_0 = idefgrd * ca;

  a_ = 1.0 / Core::LinAlg::norm2(a_0) * a_0;
  params_->structural_tensor_strategy()->setup_structural_tensor(a_, structural_tensor_);
}
FOUR_C_NAMESPACE_CLOSE
