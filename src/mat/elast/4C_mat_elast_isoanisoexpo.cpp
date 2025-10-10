// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_isoanisoexpo.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_elast_aniso_structuraltensor_strategy.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_material_base.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN


Mat::Elastic::PAR::IsoAnisoExpo::IsoAnisoExpo(const Core::Mat::PAR::Parameter::Data& matdata)
    : ParameterAniso(matdata),
      k1_(matdata.parameters.get<double>("K1")),
      k2_(matdata.parameters.get<double>("K2")),
      gamma_(matdata.parameters.get<double>("GAMMA")),
      k1comp_(matdata.parameters.get<double>("K1COMP")),
      k2comp_(matdata.parameters.get<double>("K2COMP")),
      init_(matdata.parameters.get<int>("INIT")),
      adapt_angle_(matdata.parameters.get<bool>("ADAPT_ANGLE"))
{
}

Mat::Elastic::IsoAnisoExpo::IsoAnisoExpo(Mat::Elastic::PAR::IsoAnisoExpo* params) : params_(params)
{
}

void Mat::Elastic::IsoAnisoExpo::pack_summand(Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, a_);
  add_to_pack(data, structural_tensor_);
}

void Mat::Elastic::IsoAnisoExpo::unpack_summand(Core::Communication::UnpackBuffer& buffer)
{
  extract_from_pack(buffer, a_);
  extract_from_pack(buffer, structural_tensor_);
}

void Mat::Elastic::IsoAnisoExpo::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
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

    // FIBER1 nomenclature
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

void Mat::Elastic::IsoAnisoExpo::add_stress_aniso_modified(
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& rcg,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& icg,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress, double I3, const int gp, const int eleGID,
    const Teuchos::ParameterList& params)
{
  double incJ = std::pow(I3, -1.0 / 3.0);  // J^{-2/3}

  double J4 = incJ * Core::LinAlg::ddot(structural_tensor_, rcg);  // J4 = J^{-2/3} I4

  double k1 = params_->k1_;
  double k2 = params_->k2_;

  if (J4 < 1.0)
  {
    k1 = params_->k1comp_;
    k2 = params_->k2comp_;
  }

  double gammabar = 2. * (k1 * (J4 - 1.) * exp(k2 * (J4 - 1.) * (J4 - 1.)));  // 2 dW/dJ4
  auto Saniso = gammabar * structural_tensor_;                                // Sfbar

  double traceCSfbar = Core::LinAlg::ddot(Saniso, rcg);
  Saniso = -incJ / 3. * traceCSfbar * icg + incJ * Saniso;

  Core::LinAlg::Matrix<6, 1> incJvec = Core::LinAlg::make_strain_like_voigt_matrix(icg);
  const Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> Psl =
      Core::LinAlg::FourTensorOperations::holzapfel_product(icg) -
      1.0 / 3.0 * Core::LinAlg::dyadic(icg, icg);

  Core::LinAlg::SymmetricTensor<double, 3, 3> Aiso = -J4 / 3.0 * icg + incJ * structural_tensor_;
  Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> cmataniso{};  // isochoric elastic cmat
  double deltabar = 2. * (1. + 2. * k2 * (J4 - 1.) * (J4 - 1.)) * 2. * k1 *
                    exp(k2 * (J4 - 1.) * (J4 - 1.));  // 4 d^2Wf/dJ4dJ4
  cmataniso += deltabar * Core::LinAlg::dyadic(Aiso, Aiso);

  cmataniso += 2. / 3. * incJ * traceCSfbar * Psl;
  cmataniso += -2.0 / 3.0 * (Core::LinAlg::dyadic(icg, Saniso) + Core::LinAlg::dyadic(Saniso, icg));

  stress += Saniso;
  cmat += cmataniso;
}

void Mat::Elastic::IsoAnisoExpo::get_derivatives_aniso(Core::LinAlg::Matrix<2, 1>& dPI_aniso,
    Core::LinAlg::Matrix<3, 1>& ddPII_aniso, Core::LinAlg::Matrix<4, 1>& dddPIII_aniso,
    const double I4, const int gp, const int eleGID)
{
  double k1 = params_->k1_;
  double k2 = params_->k2_;

  if (I4 < 1.0)
  {
    k1 = params_->k1comp_;
    k2 = params_->k2comp_;
  }


  dPI_aniso(0) = k1 * (I4 - 1.0) * exp(k2 * (I4 - 1.0) * (I4 - 1.0));

  ddPII_aniso(0) =
      (1.0 + 2.0 * k2 * (I4 - 1.0) * (I4 - 1.0)) * k1 * exp(k2 * (I4 - 1.0) * (I4 - 1.0));

  dddPIII_aniso(0) = (3.0 + 2.0 * k2 * (I4 - 1.0) * (I4 - 1.0)) * 2.0 * k1 * k2 * (I4 - 1.0) *
                     exp(k2 * (I4 - 1.0) * (I4 - 1.0));
};

void Mat::Elastic::IsoAnisoExpo::get_fiber_vecs(
    std::vector<Core::LinAlg::Tensor<double, 3>>& fibervecs  ///< vector of all fiber vectors
) const
{
  fibervecs.push_back(a_);
}

void Mat::Elastic::IsoAnisoExpo::set_fiber_vecs(const double newgamma,
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
