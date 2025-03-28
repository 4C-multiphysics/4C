// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_coupanisopow.hpp"

#include "4C_comm_pack_helpers.hpp"
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

void Mat::Elastic::CoupAnisoPow::setup(
    int numgp, const Core::IO::InputParameterContainer& container)
{
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
    std::ostringstream ss;
    ss << params_->fibernumber_;
    std::string fibername = "FIBER" + ss.str();  // FIBER Name
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
    // FIBERi nomenclature
    else if (container.get<std::optional<std::vector<double>>>(fibername).has_value())
    {
      // Read in of data
      read_fiber(container, fibername, a_);
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

void Mat::Elastic::CoupAnisoPow::add_stress_aniso_principal(const Core::LinAlg::Matrix<6, 1>& rcg,
    Core::LinAlg::Matrix<6, 6>& cmat, Core::LinAlg::Matrix<6, 1>& stress,
    Teuchos::ParameterList& params, const int gp, const int eleGID)
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
  I4 = structural_tensor_(0) * rcg(0) + structural_tensor_(1) * rcg(1) +
       structural_tensor_(2) * rcg(2) + structural_tensor_(3) * rcg(3) +
       structural_tensor_(4) * rcg(4) + structural_tensor_(5) * rcg(5);

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
  stress.update(gamma, structural_tensor_, 1.0);
  cmat.multiply_nt(delta, structural_tensor_, structural_tensor_, 1.0);
}

void Mat::Elastic::CoupAnisoPow::get_fiber_vecs(
    std::vector<Core::LinAlg::Matrix<3, 1>>& fibervecs  ///< vector of all fiber vectors
) const
{
  fibervecs.push_back(a_);
}

void Mat::Elastic::CoupAnisoPow::set_fiber_vecs(const double newgamma,
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
