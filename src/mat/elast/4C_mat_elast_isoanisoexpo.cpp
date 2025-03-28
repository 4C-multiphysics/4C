// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_isoanisoexpo.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
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

void Mat::Elastic::IsoAnisoExpo::setup(
    int numgp, const Core::IO::InputParameterContainer& container)
{
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
      // Read in of data
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

void Mat::Elastic::IsoAnisoExpo::add_stress_aniso_modified(const Core::LinAlg::Matrix<6, 1>& rcg,
    const Core::LinAlg::Matrix<6, 1>& icg, Core::LinAlg::Matrix<6, 6>& cmat,
    Core::LinAlg::Matrix<6, 1>& stress, double I3, const int gp, const int eleGID,
    Teuchos::ParameterList& params)
{
  double incJ = std::pow(I3, -1.0 / 3.0);  // J^{-2/3}

  double J4 = incJ * (structural_tensor_(0) * rcg(0) + structural_tensor_(1) * rcg(1) +
                         structural_tensor_(2) * rcg(2) + structural_tensor_(3) * rcg(3) +
                         structural_tensor_(4) * rcg(4) +
                         structural_tensor_(5) * rcg(5));  // J4 = J^{-2/3} I4

  double k1 = params_->k1_;
  double k2 = params_->k2_;

  if (J4 < 1.0)
  {
    k1 = params_->k1comp_;
    k2 = params_->k2comp_;
  }

  Core::LinAlg::Matrix<6, 1> Saniso(structural_tensor_);  // first compute Sfbar = 2 dW/dJ4 A_
  double gammabar = 2. * (k1 * (J4 - 1.) * exp(k2 * (J4 - 1.) * (J4 - 1.)));  // 2 dW/dJ4
  Saniso.scale(gammabar);                                                     // Sfbar

  double traceCSfbar = Saniso(0) * rcg(0) + Saniso(1) * rcg(1) + Saniso(2) * rcg(2) +
                       1. * (Saniso(3) * rcg(3) + Saniso(4) * rcg(4) + Saniso(5) * rcg(5));
  Saniso.update(-incJ / 3. * traceCSfbar, icg, incJ);

  Core::LinAlg::Matrix<6, 6> Psl(
      Core::LinAlg::Initialization::zero);  // Psl = Cinv o Cinv - 1/3 Cinv x Cinv
  Core::LinAlg::Tensor::add_holzapfel_product(Psl, icg, 1.0);
  Psl.multiply_nt(-1. / 3., icg, icg, 1.0);

  Core::LinAlg::Matrix<6, 1> Aiso(structural_tensor_);
  Aiso.update(-J4 / 3.0, icg, incJ);
  Core::LinAlg::Matrix<6, 6> cmataniso(
      Core::LinAlg::Initialization::zero);  // isochoric elastic cmat
  double deltabar = 2. * (1. + 2. * k2 * (J4 - 1.) * (J4 - 1.)) * 2. * k1 *
                    exp(k2 * (J4 - 1.) * (J4 - 1.));  // 4 d^2Wf/dJ4dJ4
  cmataniso.multiply_nt(deltabar, Aiso, Aiso);
  cmataniso.update(2. / 3. * incJ * traceCSfbar, Psl, 1.0);
  cmataniso.multiply_nt(-2. / 3., icg, Saniso, 1.0);
  cmataniso.multiply_nt(-2. / 3., Saniso, icg, 1.0);

  stress.update(1.0, Saniso, 1.0);
  cmat.update(1.0, cmataniso, 1.0);
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
    std::vector<Core::LinAlg::Matrix<3, 1>>& fibervecs  ///< vector of all fiber vectors
) const
{
  fibervecs.push_back(a_);
}

void Mat::Elastic::IsoAnisoExpo::set_fiber_vecs(const double newgamma,
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
