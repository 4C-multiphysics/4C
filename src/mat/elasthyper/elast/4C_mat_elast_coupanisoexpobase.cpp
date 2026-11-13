// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_coupanisoexpobase.hpp"

#include "4C_linalg_FADmatrix_utils.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN


Mat::Elastic::PAR::CoupAnisoExpoBase::CoupAnisoExpoBase(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : k1_(matdata.parameters.get<double>("K1")),
      k2_(matdata.parameters.get<double>("K2")),
      gamma_(matdata.parameters.get<double>("GAMMA")),
      k1comp_(matdata.parameters.get<double>("K1COMP")),
      k2comp_(matdata.parameters.get<double>("K2COMP")),
      init_(matdata.parameters.get<int>("INIT"))
{
}

Mat::Elastic::PAR::CoupAnisoExpoBase::CoupAnisoExpoBase()
    : k1_(0.0), k2_(0.0), gamma_(0.0), k1comp_(0.0), k2comp_(0.0), init_(0.0)
{
}

Mat::Elastic::CoupAnisoExpoBase::CoupAnisoExpoBase(Mat::Elastic::PAR::CoupAnisoExpoBase* params)
    : params_(params)
{
}

void Mat::Elastic::CoupAnisoExpoBase::add_strain_energy(double& psi,
    const Core::LinAlg::Matrix<3, 1>& prinv, const Core::LinAlg::Matrix<3, 1>& modinv,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, const int gp, const int eleGID)
{
  // right Cauchy Green
  evaluate_func<double>(
      psi, 2.0 * glstrain + Core::LinAlg::TensorGenerators::identity<double, 3, 3>, gp, eleGID);
}

template <typename T>
void Mat::Elastic::CoupAnisoExpoBase::evaluate_func(
    T& psi, Core::LinAlg::SymmetricTensor<T, 3, 3> const& C, const int gp, int const eleGID) const
{
  Core::LinAlg::SymmetricTensor<T, 3, 3> A_T =
      get_coup_aniso_expo_base_interface().get_structural_tensor(gp);
  const double scalarProduct = get_coup_aniso_expo_base_interface().get_scalar_product(gp);

  T I4 = Core::LinAlg::ddot(C, A_T);

  T k1;
  T k2;
  if (I4 < 1.0)
  {
    k1 = params_->k1comp_;
    k2 = params_->k2comp_;
  }
  else
  {
    k1 = params_->k1_;
    k2 = params_->k2_;
  }

  psi += (k1 / (2.0 * k2)) * (std::exp(k2 * (I4 - scalarProduct) * (I4 - scalarProduct)) - 1.0);
}

void Mat::Elastic::CoupAnisoExpoBase::evaluate_first_derivatives_aniso(
    Core::LinAlg::Matrix<2, 1>& dPI_aniso, Core::LinAlg::SymmetricTensor<double, 3, 3> const& rcg,
    int gp, int eleGID)
{
  double I4 =
      Core::LinAlg::ddot(get_coup_aniso_expo_base_interface().get_structural_tensor(gp), rcg);
  const double scalarProduct = get_coup_aniso_expo_base_interface().get_scalar_product(gp);

  double k1 = params_->k1_;
  double k2 = params_->k2_;

  if (I4 < 1.0)
  {
    k1 = params_->k1comp_;
    k2 = params_->k2comp_;
  }

  dPI_aniso(0) =
      k1 * (I4 - scalarProduct) * std::exp(k2 * (I4 - scalarProduct) * (I4 - scalarProduct));
}

void Mat::Elastic::CoupAnisoExpoBase::evaluate_second_derivatives_aniso(
    Core::LinAlg::Matrix<3, 1>& ddPII_aniso, Core::LinAlg::SymmetricTensor<double, 3, 3> const& rcg,
    int gp, int eleGID)
{
  double I4 =
      Core::LinAlg::ddot(get_coup_aniso_expo_base_interface().get_structural_tensor(gp), rcg);
  const double scalarProduct = get_coup_aniso_expo_base_interface().get_scalar_product(gp);

  double k1 = params_->k1_;
  double k2 = params_->k2_;

  if (I4 < 1.0)
  {
    k1 = params_->k1comp_;
    k2 = params_->k2comp_;
  }

  ddPII_aniso(0) = (1.0 + 2.0 * k2 * std::pow((I4 - scalarProduct), 2)) * k1 *
                   std::exp(k2 * std::pow((I4 - scalarProduct), 2));
}

template <typename T>
void Mat::Elastic::CoupAnisoExpoBase::get_derivatives_aniso(
    Core::LinAlg::Matrix<2, 1, T>& dPI_aniso, Core::LinAlg::Matrix<3, 1, T>& ddPII_aniso,
    Core::LinAlg::Matrix<4, 1, T>& dddPIII_aniso, Core::LinAlg::SymmetricTensor<T, 3, 3> const& rcg,
    const int gp, const int eleGID) const
{
  const double scalarProduct = get_coup_aniso_expo_base_interface().get_scalar_product(gp);

  T I4 = Core::LinAlg::ddot(get_coup_aniso_expo_base_interface().get_structural_tensor(gp), rcg);

  T k1 = params_->k1_;
  T k2 = params_->k2_;

  if (I4 < 1.0)
  {
    k1 = params_->k1comp_;
    k2 = params_->k2comp_;
  }


  dPI_aniso(0) = k1 * (I4 - scalarProduct) * std::exp(k2 * std::pow((I4 - scalarProduct), 2));

  ddPII_aniso(0) = (1.0 + 2.0 * k2 * std::pow((I4 - scalarProduct), 2)) * k1 *
                   std::exp(k2 * std::pow((I4 - scalarProduct), 2));

  dddPIII_aniso(0) = (3.0 + 2.0 * k2 * std::pow((I4 - scalarProduct), 2)) * 2.0 * k1 * k2 *
                     (I4 - scalarProduct) * std::exp(k2 * std::pow((I4 - scalarProduct), 2));
}

void Mat::Elastic::CoupAnisoExpoBase::add_stress_aniso_principal(
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& rcg,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress, const Teuchos::ParameterList& params,
    const int gp, const int eleGID)
{
  double I4 =
      Core::LinAlg::ddot(get_coup_aniso_expo_base_interface().get_structural_tensor(gp), rcg);
  const double scalarProduct = get_coup_aniso_expo_base_interface().get_scalar_product(gp);

  double k1;
  double k2;
  if (I4 < 1.0)
  {
    k1 = params_->k1comp_;
    k2 = params_->k2comp_;
  }
  else
  {
    k1 = params_->k1_;
    k2 = params_->k2_;
  }

  double gamma =
      2. * (k1 * (I4 - scalarProduct) * std::exp(k2 * std::pow((I4 - scalarProduct), 2)));
  stress += gamma * get_coup_aniso_expo_base_interface().get_structural_tensor(gp);

  double delta = 2. * (1. + 2. * k2 * std::pow((I4 - scalarProduct), 2)) * 2. * k1 *
                 std::exp(k2 * std::pow((I4 - scalarProduct), 2));
  cmat +=
      delta * Core::LinAlg::dyadic(get_coup_aniso_expo_base_interface().get_structural_tensor(gp),
                  get_coup_aniso_expo_base_interface().get_structural_tensor(gp));
}

void Mat::Elastic::CoupAnisoExpoBase::get_fiber_vecs(
    std::vector<Core::LinAlg::Tensor<double, 3>>& fibervecs) const
{
  FOUR_C_THROW("Getting the fiber vectors is not implemented in the base version of CoupAnisoExpo");
}

void Mat::Elastic::CoupAnisoExpoBase::set_fiber_vecs(const double newgamma,
    const Core::LinAlg::Tensor<double, 3, 3>& locsys,
    const Core::LinAlg::Tensor<double, 3, 3>& defgrd)
{
  FOUR_C_THROW("Setting the fiber vectors is not implemented in the base version of CoupAnisoExpo");
}

void Mat::Elastic::CoupAnisoExpoBase::set_fiber_vecs(
    const Core::LinAlg::Tensor<double, 3>& fibervec)
{
  FOUR_C_THROW("Setting the fiber vectors is not implemented in the base version of CoupAnisoExpo");
}


// explicit instantiation of template functions
template void Mat::Elastic::CoupAnisoExpoBase::get_derivatives_aniso<double>(
    Core::LinAlg::Matrix<2, 1, double>&, Core::LinAlg::Matrix<3, 1, double>&,
    Core::LinAlg::Matrix<4, 1, double>&, Core::LinAlg::SymmetricTensor<double, 3, 3> const&, int,
    int) const;
template void Mat::Elastic::CoupAnisoExpoBase::evaluate_func<double>(
    double&, Core::LinAlg::SymmetricTensor<double, 3, 3> const&, int, int) const;

FOUR_C_NAMESPACE_CLOSE
