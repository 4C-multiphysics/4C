// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_ogden_tca.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_symmetric_tensor_eigen.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_utils_exceptions.hpp"

#include <array>
#include <cmath>

FOUR_C_NAMESPACE_OPEN

Mat::PAR::OgdenTCA::OgdenTCA(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      c_(matdata.parameters.get<double>("C")),
      m_(matdata.parameters.get<double>("M")),
      q_(matdata.parameters.get<double>("Q")),
      kappa_(matdata.parameters.get<double>("KAPPA")),
      density_(matdata.parameters.get<double>("DENS"))
{
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::OgdenTCA::create_material()
{
  return std::make_shared<Mat::OgdenTCA>(this);
}

Mat::OgdenTCAType Mat::OgdenTCAType::instance_;

Core::Communication::ParObject* Mat::OgdenTCAType::create(Core::Communication::UnpackBuffer& buffer)
{
  auto* mat_ogden_tca = new Mat::OgdenTCA();
  mat_ogden_tca->unpack(buffer);
  return mat_ogden_tca;
}

Mat::OgdenTCA::OgdenTCA() : params_(nullptr) {}

Mat::OgdenTCA::OgdenTCA(Mat::PAR::OgdenTCA* params) : params_(params) {}

void Mat::OgdenTCA::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);
}

void Mat::OgdenTCA::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // make sure we have a pristine material
  params_ = nullptr;

  // matid and recover params_
  int matid;
  extract_from_pack(buffer, matid);

  if (Global::Problem::instance()->materials() != nullptr)
  {
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
      {
        params_ = static_cast<Mat::PAR::OgdenTCA*>(mat);
      }
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
  }
}

void Mat::OgdenTCA::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
}

void Mat::OgdenTCA::update(Core::LinAlg::Tensor<double, 3, 3> const& defgrd, int const gp,
    const Teuchos::ParameterList& params, const EvaluationContext<3>& context, int const eleGID)
{
}

void Mat::OgdenTCA::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  // make sure stresses and material tangent are zero
  stress = {};
  cmat = {};

  // parameters
  const double c = params_->c_;
  const double m = params_->m_;
  const double q = params_->q_;
  const double kappa = params_->kappa_;

  // right Cauchy Green tensor C = F^{T} F
  const Core::LinAlg::SymmetricTensor<double, 3, 3> C =
      Core::LinAlg::assume_symmetry(Core::LinAlg::transpose(*defgrad) * *defgrad);

  // determinant of C
  const double detC = Core::LinAlg::det(C);

  // inverse right Cauchy Green tensor C^-1
  const Core::LinAlg::SymmetricTensor<double, 3, 3> invC = Core::LinAlg::inv(C);

  // dyadic product of the inverse right Cauchy Green tensor
  const Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> dyadic_invC =
      Core::LinAlg::dyadic(invC, invC);

  // Part 1: Non-volumetric Ogden TCA contribution
  // Spectral decomposition of Cauchy-Green strain:
  // lambda_i^2 are the eigenvalues of C, i.e. the squares of the principal stretches
  // Ni are the eigenvectors of C, i.e. the principal referential directions
  const auto [eigenvalues, eigenvectors] = Core::LinAlg::eig(C);
  std::array<double, 3> lambda;                      // principal stretches
  std::array<Core::LinAlg::Tensor<double, 3>, 3> N;  // principal referential directions
  for (int i = 0; i < 3; ++i)
  {
    lambda[i] = std::sqrt(eigenvalues[i]);
    N[i] = Core::LinAlg::Tensor<double, 3>{
        {eigenvectors(0, i), eigenvectors(1, i), eigenvectors(2, i)}};
  }

  // Compute second Piola-Kirchhoff stresses and the material tangent tensor using the spectral
  // formula according to Holzapfel Eq. (6.52) and Eq. (6.180), respectively:
  // stress = sum_i [S_i * N_i dyadic N_i] with S_i = 1/lambda_i * dPsi/dlambda_i
  // cmat = cmat_1 + cmat_2 with
  // cmat_1 = sum_i sum_j [1/lambda_j * dS_i/dlambda_j N_i dyadic N_i dyadic N_j dyadic N_j]
  // cmat_2 = sum_i sum_j i!=j [(S_j - S_i)/(lambda_j^2 - lambda_i^2) (N_i dyadic N_j dyadic N_i
  // dyadic N_j + N_i dyadic N_j dyadic N_j dyadic N_i)]
  std::array<double, 3> S{0}, dSdlambda{0};
  for (int i = 0; i < 3; ++i)
  {
    const auto dyadic_Ni_Ni = Core::LinAlg::self_dyadic(N[i]);

    // Principal PK2 stress and its derivative w.r.t. principal stretch
    S[i] = c / m * (q * std::pow(lambda[i], m - 2) - (1 - q) * std::pow(lambda[i], -m - 2));
    dSdlambda[i] = c / m *
                   (q * (m - 2) * std::pow(lambda[i], m - 3) +
                       (1 - q) * (m + 2) * std::pow(lambda[i], -m - 3));

    // PK2 stress tensor
    stress += S[i] * dyadic_Ni_Ni;

    // cmat_1: In our case dS_i/dlambda_j = 0, so simplifies to: sum_i [1/lambda_i * dS_i/dlambda_i
    // N_i dyadic N_i dyadic N_i dyadic N_i]
    cmat += dSdlambda[i] / lambda[i] * Core::LinAlg::dyadic(dyadic_Ni_Ni, dyadic_Ni_Ni);
  }

  // cmat_2: For the prefactor, we consider the special case lambda_i == lambda_j.
  // We compute the limit of (S_j - S_i)/(lambda_j^2 - lambda_i^2) as lambda_j ->
  // lambda_i according to to Eq. (6.190): dS_j/d(lambda_j^2) - dS_i/d(lambda_j^2) =
  // dS_j/d(lambda_j^2) = 1/(2*lambda_j) * dS_j/dlambda_j
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      if (i != j)
      {
        const auto dyadic_Ni_Nj = Core::LinAlg::dyadic(N[i], N[j]);
        const auto dyadic_Nj_Ni = Core::LinAlg::transpose(dyadic_Ni_Nj);

        constexpr double equal_lambda_tol = 1e-12;
        const double fac = (std::abs(lambda[i] - lambda[j]) < equal_lambda_tol)
                               ? 0.5 / lambda[j] * dSdlambda[j]
                               : (S[j] - S[i]) / (lambda[j] * lambda[j] - lambda[i] * lambda[i]);
        cmat +=
            fac * Core::LinAlg::assume_symmetry(Core::LinAlg::dyadic(dyadic_Ni_Nj, dyadic_Ni_Nj));
        cmat +=
            fac * Core::LinAlg::assume_symmetry(Core::LinAlg::dyadic(dyadic_Ni_Nj, dyadic_Nj_Ni));
      }
    }
  }

  // Part 2: Volumetric contribution as proposed in Moerman et al. (2016)
  const double scalar_term_1 = kappa * (detC - std::sqrt(detC));
  const double scalar_term_2 = scalar_term_1 + c / m * (1 - 2 * q);

  stress += scalar_term_2 * invC;
  cmat += (scalar_term_1 + kappa * detC) * dyadic_invC +
          -2 * scalar_term_2 * Core::LinAlg::FourTensorOperations::holzapfel_product(invC);
}

FOUR_C_NAMESPACE_CLOSE
