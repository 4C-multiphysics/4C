// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_coupanisoexpoactive.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_linalg_FADmatrix_utils.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_elast_aniso_structuraltensor_strategy.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN


Mat::Elastic::PAR::CoupAnisoExpoActive::CoupAnisoExpoActive(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : ParameterAniso(matdata),
      k1_(matdata.parameters.get<double>("K1")),
      k2_(matdata.parameters.get<double>("K2")),
      gamma_(matdata.parameters.get<double>("GAMMA")),
      k1comp_(matdata.parameters.get<double>("K1COMP")),
      k2comp_(matdata.parameters.get<double>("K2COMP")),
      init_(matdata.parameters.get<int>("INIT")),
      adapt_angle_(matdata.parameters.get<bool>("ADAPT_ANGLE")),
      s_(matdata.parameters.get<double>("S")),
      lambdamax_(matdata.parameters.get<double>("LAMBDAMAX")),
      lambda0_(matdata.parameters.get<double>("LAMBDA0")),
      dens_(matdata.parameters.get<double>("DENS"))
{
}

Mat::Elastic::CoupAnisoExpoActive::CoupAnisoExpoActive(
    Mat::Elastic::PAR::CoupAnisoExpoActive* params)
    : params_(params),
      anisotropy_extension_(params_->init_, params->gamma_, params_->adapt_angle_ != 0,
          params_->structural_tensor_strategy(), {0})
{
  d_p_iact_ = 0.0;
  lambdaact_ = 1.0;
  anisotropy_extension_.register_needed_tensors(
      FiberAnisotropyExtension<1>::FIBER_VECTORS | FiberAnisotropyExtension<1>::STRUCTURAL_TENSOR);
}

void Mat::Elastic::CoupAnisoExpoActive::register_anisotropy_extensions(Mat::Anisotropy& anisotropy)
{
  anisotropy.register_anisotropy_extension(anisotropy_extension_);
}

void Mat::Elastic::CoupAnisoExpoActive::pack_summand(Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, lambdaact_);
  anisotropy_extension_.pack_anisotropy(data);
}

void Mat::Elastic::CoupAnisoExpoActive::unpack_summand(Core::Communication::UnpackBuffer& buffer)
{
  extract_from_pack(buffer, lambdaact_);
  anisotropy_extension_.unpack_anisotropy(buffer);

  d_p_iact_ = evaluated_psi_active();
}

void Mat::Elastic::CoupAnisoExpoActive::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  // setup first derivative of active fiber potential w.r.t. active fiber stretch (const during the
  // whole simulation)
  lambdaact_ = 1.0;

  d_p_iact_ = evaluated_psi_active();
}

void Mat::Elastic::CoupAnisoExpoActive::add_strain_energy(double& psi,
    const Core::LinAlg::Matrix<3, 1>& prinv, const Core::LinAlg::Matrix<3, 1>& modinv,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, const int gp, const int eleGID)
{
  // right Cauchy Green in strain like Voigt notation
  double I4 = Core::LinAlg::ddot(anisotropy_extension_.get_structural_tensor(gp, 0), glstrain);

  double k1 = params_->k1_;
  double k2 = params_->k2_;

  if (prinv(0) < 1.0)
  {
    k1 = params_->k1comp_;
    k2 = params_->k2comp_;
  }

  // passive contribution
  psi += (k1 / (2.0 * k2)) * (exp(k2 * (I4 - 1.0) * (I4 - 1.0)) - 1.0);

  // active contribution
  psi += params_->s_ / params_->dens_ *
         (lambdaact_ + (1. / 3.) * (std::pow(params_->lambdamax_ - lambdaact_, 3.0) /
                                       std::pow(params_->lambdamax_ - params_->lambda0_, 2.0)));
}

template <typename T>
void Mat::Elastic::CoupAnisoExpoActive::evaluate_func(
    T& psi, Core::LinAlg::SymmetricTensor<T, 3, 3> const& rcg, const int gp, int const eleGID) const
{
  T I4_fad = Core::LinAlg::ddot(anisotropy_extension_.get_structural_tensor(gp, 0), rcg);

  T k1 = params_->k1_;
  T k2 = params_->k2_;

  if (I4_fad < 1.0)
  {
    k1 = params_->k1comp_;
    k2 = params_->k2comp_;
  }

  psi += (k1 / (2.0 * k2)) * (exp(k2 * (I4_fad - 1.0) * (I4_fad - 1.0)) - 1.0);
};

template <typename T>
void Mat::Elastic::CoupAnisoExpoActive::evaluate_active_stress_cmat_aniso(
    Core::LinAlg::SymmetricTensor<T, 3, 3> const& CM,
    Core::LinAlg::SymmetricTensor<T, 3, 3, 3, 3>& cmat,
    Core::LinAlg::SymmetricTensor<T, 3, 3>& stress, const int gp, const int eleGID) const
{
  T lambda_sq = 0.0;
  lambda_sq = Core::LinAlg::ddot(anisotropy_extension_.get_structural_tensor(gp, 0), CM);

  T dPIact_T = 0.0;
  dPIact_T = d_p_iact_;
  stress = T(dPIact_T / lambda_sq) * anisotropy_extension_.get_structural_tensor(gp, 0);
  cmat = T(-2.0 * dPIact_T * 1. / (lambda_sq * lambda_sq)) *
         Core::LinAlg::dyadic(anisotropy_extension_.get_structural_tensor(gp, 0),
             anisotropy_extension_.get_structural_tensor(gp, 0));
}

void Mat::Elastic::CoupAnisoExpoActive::add_active_stress_cmat_aniso(
    Core::LinAlg::SymmetricTensor<double, 3, 3> const& CM,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress, const int gp, const int eleGID) const
{
  double lambda_sq = Core::LinAlg::ddot(CM, anisotropy_extension_.get_structural_tensor(gp, 0));

  double dPIact_T = 0.0;
  dPIact_T = d_p_iact_;
  stress += dPIact_T * 1. / lambda_sq * anisotropy_extension_.get_structural_tensor(gp, 0);
  cmat += -2.0 * dPIact_T * 1. / (lambda_sq * lambda_sq) *
          Core::LinAlg::dyadic(anisotropy_extension_.get_structural_tensor(gp, 0),
              anisotropy_extension_.get_structural_tensor(gp, 0));
}

void Mat::Elastic::CoupAnisoExpoActive::evaluate_first_derivatives_aniso(
    Core::LinAlg::Matrix<2, 1>& dPI_aniso, Core::LinAlg::SymmetricTensor<double, 3, 3> const& rcg,
    int gp, int eleGID)
{
  double I4 = Core::LinAlg::ddot(anisotropy_extension_.get_structural_tensor(gp, 0), rcg);

  double k1 = params_->k1_;
  double k2 = params_->k2_;

  if (I4 < 1.0)
  {
    k1 = params_->k1comp_;
    k2 = params_->k2comp_;
  }

  dPI_aniso(0) = k1 * (I4 - 1.0) * exp(k2 * (I4 - 1.0) * (I4 - 1.0));
}

void Mat::Elastic::CoupAnisoExpoActive::evaluate_second_derivatives_aniso(
    Core::LinAlg::Matrix<3, 1>& ddPII_aniso, Core::LinAlg::SymmetricTensor<double, 3, 3> const& rcg,
    int gp, int eleGID)
{
  double I4 = Core::LinAlg::ddot(anisotropy_extension_.get_structural_tensor(gp, 0), rcg);

  double k1 = params_->k1_;
  double k2 = params_->k2_;

  if (I4 < 1.0)
  {
    k1 = params_->k1comp_;
    k2 = params_->k2comp_;
  }

  ddPII_aniso(0) =
      (1.0 + 2.0 * k2 * (I4 - 1.0) * (I4 - 1.0)) * k1 * exp(k2 * (I4 - 1.0) * (I4 - 1.0));
}

template <typename T>
void Mat::Elastic::CoupAnisoExpoActive::get_derivatives_aniso(
    Core::LinAlg::Matrix<2, 1, T>& dPI_aniso, Core::LinAlg::Matrix<3, 1, T>& ddPII_aniso,
    Core::LinAlg::Matrix<4, 1, T>& dddPIII_aniso, Core::LinAlg::SymmetricTensor<T, 3, 3> const& rcg,
    const int gp, const int eleGID) const
{
  T I4 = Core::LinAlg::ddot(anisotropy_extension_.get_structural_tensor(gp, 0), rcg);

  T k1 = params_->k1_;
  T k2 = params_->k2_;

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

void Mat::Elastic::CoupAnisoExpoActive::add_stress_aniso_principal(
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& rcg,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress, const Teuchos::ParameterList& params,
    const int gp, const int eleGID)
{
  double I4 = Core::LinAlg::ddot(anisotropy_extension_.get_structural_tensor(gp, 0), rcg);

  double k1 = params_->k1_;
  double k2 = params_->k2_;

  if (I4 < 1.0)
  {
    k1 = params_->k1comp_;
    k2 = params_->k2comp_;
  }

  double gamma = 2. * (k1 * (I4 - 1.) * exp(k2 * (I4 - 1.) * (I4 - 1.)));
  stress += gamma * anisotropy_extension_.get_structural_tensor(gp, 0);

  double delta =
      2. * (1. + 2. * k2 * (I4 - 1.) * (I4 - 1.)) * 2. * k1 * exp(k2 * (I4 - 1.) * (I4 - 1.));
  cmat += delta * Core::LinAlg::dyadic(anisotropy_extension_.get_structural_tensor(gp, 0),
                      anisotropy_extension_.get_structural_tensor(gp, 0));
}

void Mat::Elastic::CoupAnisoExpoActive::get_fiber_vecs(
    std::vector<Core::LinAlg::Tensor<double, 3>>& fibervecs) const
{
  // this method does not support Gauss point fibers
  fibervecs.push_back(anisotropy_extension_.get_fiber(BaseAnisotropyExtension::GPDEFAULT, 0));
}

void Mat::Elastic::CoupAnisoExpoActive::set_fiber_vecs(const double newgamma,
    const Core::LinAlg::Tensor<double, 3, 3>& locsys,
    const Core::LinAlg::Tensor<double, 3, 3>& defgrd)
{
  anisotropy_extension_.set_fiber_vecs(newgamma, locsys, defgrd);
}

double Mat::Elastic::CoupAnisoExpoActive::evaluated_psi_active() const
{
  return params_->s_ / params_->dens_ *
         (1.0 - std::pow(params_->lambdamax_ - lambdaact_, 2.0) /
                    std::pow(params_->lambdamax_ - params_->lambda0_, 2.0));
}


// explicit instantiation of template functions
template void Mat::Elastic::CoupAnisoExpoActive::get_derivatives_aniso<double>(
    Core::LinAlg::Matrix<2, 1, double>&, Core::LinAlg::Matrix<3, 1, double>&,
    Core::LinAlg::Matrix<4, 1, double>&, Core::LinAlg::SymmetricTensor<double, 3, 3> const&, int,
    const int) const;
template void Mat::Elastic::CoupAnisoExpoActive::get_derivatives_aniso<FAD>(
    Core::LinAlg::Matrix<2, 1, FAD>&, Core::LinAlg::Matrix<3, 1, FAD>&,
    Core::LinAlg::Matrix<4, 1, FAD>&, Core::LinAlg::SymmetricTensor<FAD, 3, 3> const&, int,
    const int) const;
template void Mat::Elastic::CoupAnisoExpoActive::evaluate_active_stress_cmat_aniso<double>(
    Core::LinAlg::SymmetricTensor<double, 3, 3> const&,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>&,
    Core::LinAlg::SymmetricTensor<double, 3, 3>&, int, const int) const;
template void Mat::Elastic::CoupAnisoExpoActive::evaluate_active_stress_cmat_aniso<FAD>(
    Core::LinAlg::SymmetricTensor<FAD, 3, 3> const&,
    Core::LinAlg::SymmetricTensor<FAD, 3, 3, 3, 3>&, Core::LinAlg::SymmetricTensor<FAD, 3, 3>&, int,
    const int) const;
template void Mat::Elastic::CoupAnisoExpoActive::evaluate_func<double>(
    double&, Core::LinAlg::SymmetricTensor<double, 3, 3> const&, int, const int) const;
template void Mat::Elastic::CoupAnisoExpoActive::evaluate_func<FAD>(
    FAD&, Core::LinAlg::SymmetricTensor<FAD, 3, 3> const&, int, const int) const;

FOUR_C_NAMESPACE_CLOSE
