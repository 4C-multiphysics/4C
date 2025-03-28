// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_coupanisoexpobase.hpp"

#include "4C_linalg_FADmatrix_utils.hpp"
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
    const Core::LinAlg::Matrix<6, 1>& glstrain, const int gp, const int eleGID)
{
  // right Cauchy Green
  Core::LinAlg::Matrix<3, 3> C(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; ++i) C(i, i) = 2.0 * glstrain(i) + 1.0;
  C(0, 1) = C(1, 0) = glstrain(3);
  C(1, 2) = C(2, 1) = glstrain(4);
  C(0, 2) = C(2, 0) = glstrain(5);

  evaluate_func<double>(psi, C, gp, eleGID);
}

template <typename T>
void Mat::Elastic::CoupAnisoExpoBase::evaluate_func(
    T& psi, Core::LinAlg::Matrix<3, 3, T> const& C, const int gp, int const eleGID) const
{
  Core::LinAlg::Matrix<3, 3, T> A_T(
      get_coup_aniso_expo_base_interface().get_structural_tensor(gp).data());
  const double scalarProduct = get_coup_aniso_expo_base_interface().get_scalar_product(gp);

  T I4 = C.dot(A_T);

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
    Core::LinAlg::Matrix<2, 1>& dPI_aniso, Core::LinAlg::Matrix<3, 3> const& rcg, int gp,
    int eleGID)
{
  double I4 = get_coup_aniso_expo_base_interface().get_structural_tensor(gp).dot(rcg);
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
    Core::LinAlg::Matrix<3, 1>& ddPII_aniso, Core::LinAlg::Matrix<3, 3> const& rcg, int gp,
    int eleGID)
{
  double I4 = get_coup_aniso_expo_base_interface().get_structural_tensor(gp).dot(rcg);
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
    Core::LinAlg::Matrix<4, 1, T>& dddPIII_aniso, Core::LinAlg::Matrix<3, 3, T> const& rcg,
    const int gp, const int eleGID) const
{
  Core::LinAlg::Matrix<3, 3, T> AM(
      get_coup_aniso_expo_base_interface().get_structural_tensor(gp).data());
  const double scalarProduct = get_coup_aniso_expo_base_interface().get_scalar_product(gp);

  T I4 = AM.dot(rcg);

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
    const Core::LinAlg::Matrix<6, 1>& rcg, Core::LinAlg::Matrix<6, 6>& cmat,
    Core::LinAlg::Matrix<6, 1>& stress, Teuchos::ParameterList& params, const int gp,
    const int eleGID)
{
  double I4 = get_coup_aniso_expo_base_interface().get_structural_tensor_stress(gp).dot(rcg);
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
  stress.update(gamma, get_coup_aniso_expo_base_interface().get_structural_tensor_stress(gp), 1.0);

  double delta = 2. * (1. + 2. * k2 * std::pow((I4 - scalarProduct), 2)) * 2. * k1 *
                 std::exp(k2 * std::pow((I4 - scalarProduct), 2));
  cmat.multiply_nt(delta, get_coup_aniso_expo_base_interface().get_structural_tensor_stress(gp),
      get_coup_aniso_expo_base_interface().get_structural_tensor_stress(gp), 1.0);
}

void Mat::Elastic::CoupAnisoExpoBase::get_fiber_vecs(
    std::vector<Core::LinAlg::Matrix<3, 1>>& fibervecs) const
{
  FOUR_C_THROW("Getting the fiber vectors is not implemented in the base version of CoupAnisoExpo");
}

void Mat::Elastic::CoupAnisoExpoBase::set_fiber_vecs(const double newgamma,
    const Core::LinAlg::Matrix<3, 3>& locsys, const Core::LinAlg::Matrix<3, 3>& defgrd)
{
  FOUR_C_THROW("Setting the fiber vectors is not implemented in the base version of CoupAnisoExpo");
}

void Mat::Elastic::CoupAnisoExpoBase::set_fiber_vecs(const Core::LinAlg::Matrix<3, 1>& fibervec)
{
  FOUR_C_THROW("Setting the fiber vectors is not implemented in the base version of CoupAnisoExpo");
}


// explicit instantiation of template functions
template void Mat::Elastic::CoupAnisoExpoBase::get_derivatives_aniso<double>(
    Core::LinAlg::Matrix<2, 1, double>&, Core::LinAlg::Matrix<3, 1, double>&,
    Core::LinAlg::Matrix<4, 1, double>&, Core::LinAlg::Matrix<3, 3, double> const&, int, int) const;
template void Mat::Elastic::CoupAnisoExpoBase::evaluate_func<double>(
    double&, Core::LinAlg::Matrix<3, 3, double> const&, int, int) const;

FOUR_C_NAMESPACE_CLOSE
