// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_coupanisoneohooke_VarProp.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_elast_aniso_structuraltensor_strategy.hpp"
#include "4C_mat_service.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_utils_function.hpp"

FOUR_C_NAMESPACE_OPEN


Mat::Elastic::PAR::CoupAnisoNeoHookeVarProp::CoupAnisoNeoHookeVarProp(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : ParameterAniso(matdata),
      c_(matdata.parameters.get<double>("C")),
      sourceactiv_(matdata.parameters.get<int>("SOURCE_ACTIVATION")),
      gamma_(matdata.parameters.get<double>("GAMMA")),
      theta_(matdata.parameters.get<double>("THETA")),
      init_(matdata.parameters.get<int>("INIT")),
      adapt_angle_(matdata.parameters.get<bool>("ADAPT_ANGLE"))
{
}

Mat::Elastic::CoupAnisoNeoHookeVarProp::CoupAnisoNeoHookeVarProp(
    Mat::Elastic::PAR::CoupAnisoNeoHookeVarProp* params)
    : params_(params)
{
}

void Mat::Elastic::CoupAnisoNeoHookeVarProp::pack_summand(
    Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, a_);
  add_to_pack(data, structural_tensor_);
}

void Mat::Elastic::CoupAnisoNeoHookeVarProp::unpack_summand(
    Core::Communication::UnpackBuffer& buffer)
{
  extract_from_pack(buffer, a_);
  extract_from_pack(buffer, structural_tensor_);
}

void Mat::Elastic::CoupAnisoNeoHookeVarProp::setup(int numgp,
    const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  // path if fibers aren't given in input file
  if (params_->init_ == 0)
  {
    // fibers aligned in YZ-plane with gamma around Z in global cartesian cosy
    Core::LinAlg::Tensor<double, 3, 3> locsys{};

    // To realize a full rotated fiber orientation and to keep the general structure of
    // SetFiberVec() the input of locsys has to be adapted if one sets
    //               1           0                 sin(theta_)
    //  locsys := [  0       sin(theta_)               0                  ]
    //               0  sin(gamma)*cos(theta_)    cos(gamma_)*cos(theta_)
    // The call of SetFiberVec() will leed to the following fiber direction
    // a = cos(gamma_)*locsys(:,2) + sin(gamma_)*locsys(:,1)
    //         cos(gamma_)*sin(theta_)               0 cos(gamma_)*sin(theta_)
    //   = [             0              ] + [  sin(gamma_)*sin(theta_)  ] = [
    //   sin(gamma_)*sin(theta_) ] =: sperical coordinates
    //         cos(gamma)^2*cos(theta_)        sin(gamma_)^2*cos(theta_)               cos(theta_)
    //
    {
      // Local initialization of spherical angles
      double theta = (params_->theta_);
      double gamma = (params_->gamma_);
      if (gamma < 0.0 || gamma > 180.0 || abs(theta) > 180.0)
      {
        FOUR_C_THROW(
            "Wrong choice of sherical coordinates. Correct domain is gamma in [0,180], theta in "
            "[-180, 180]");
      }
      // conversion to radian measure
      theta = (theta * std::numbers::pi) / 180.0;
      gamma = (gamma * std::numbers::pi) / 180.0;
      locsys(1, 1) = sin(theta);
      locsys(2, 1) = sin(gamma) * cos(theta);
      locsys(0, 2) = sin(theta);
      locsys(2, 2) = cos(gamma) * cos(theta);
    }
    set_fiber_vecs(-1.0, locsys,
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
      Core::LinAlg::Matrix<3, 3> Id(Core::LinAlg::Initialization::zero);
      for (int i = 0; i < 3; i++) Id(i, i) = 1.0;
      // final setup of fiber data
      set_fiber_vecs(0.0, locsys,
          Core::LinAlg::get_full(Core::LinAlg::TensorGenerators::identity<double, 3, 3>));
    }

    // FIBER1 nomenclature
    else if (fibers.element_fibers.size() >= 1)
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

void Mat::Elastic::CoupAnisoNeoHookeVarProp::add_stress_aniso_principal(
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& rcg,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress, const Teuchos::ParameterList& params,
    const int gp, const int eleGID)
{
  double time_ = get_or<double>(params, "total time", 0.0);
  const auto& element_center_coordinates_ref =
      params.get<Core::LinAlg::Tensor<double, 3>>("elecenter_coords_ref");
  double stressFact_ = Global::Problem::instance()
                           ->function_by_id<Core::Utils::FunctionOfSpaceTime>(params_->sourceactiv_)
                           .evaluate(element_center_coordinates_ref.as_span(), time_, 0);


  // double stressFact_=params.get<double>("scalar");
  stress += 2 * (params_->c_) * stressFact_ * structural_tensor_;

  // no contribution to cmat
  // double delta = 0.0;
  // cmat.multiply_nt(delta, A_, A_, 1.0);
}

void Mat::Elastic::CoupAnisoNeoHookeVarProp::get_fiber_vecs(
    std::vector<Core::LinAlg::Tensor<double, 3>>& fibervecs  ///< vector of all fiber vectors
) const
{
  fibervecs.push_back(a_);
}

void Mat::Elastic::CoupAnisoNeoHookeVarProp::set_fiber_vecs(const double newgamma,
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
  a_ = 1 / Core::LinAlg::norm2(a_0) * a_0;

  params_->structural_tensor_strategy()->setup_structural_tensor(a_, structural_tensor_);
}

FOUR_C_NAMESPACE_CLOSE
