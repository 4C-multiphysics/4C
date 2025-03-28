// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mixture_prestress_strategy_iterative.hpp"

#include "4C_linalg_fixedsizematrix_generators.hpp"
#include "4C_linalg_utils_densematrix_svd.hpp"
#include "4C_mat_anisotropy.hpp"
#include "4C_mat_elast_isoneohooke.hpp"
#include "4C_mat_elast_volsussmanbathe.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_mixture_constituent_elasthyper.hpp"
#include "4C_mixture_rule.hpp"

FOUR_C_NAMESPACE_OPEN

Mixture::PAR::IterativePrestressStrategy::IterativePrestressStrategy(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : PrestressStrategy(matdata),
      isochoric_(matdata.parameters.get<bool>("ISOCHORIC")),
      is_active_(matdata.parameters.get<bool>("ACTIVE"))
{
}

std::unique_ptr<Mixture::PrestressStrategy>
Mixture::PAR::IterativePrestressStrategy::create_prestress_strategy()
{
  std::unique_ptr<Mixture::PrestressStrategy> prestressStrategy(
      new Mixture::IterativePrestressStrategy(this));
  return prestressStrategy;
}

Mixture::IterativePrestressStrategy::IterativePrestressStrategy(
    Mixture::PAR::IterativePrestressStrategy* params)
    : PrestressStrategy(params), params_(params)
{
}

void Mixture::IterativePrestressStrategy::setup(
    Mixture::MixtureConstituent& constituent, Teuchos::ParameterList& params, int numgp, int eleGID)
{
  // nothing to do
}

void Mixture::IterativePrestressStrategy::evaluate_prestress(const MixtureRule& mixtureRule,
    const std::shared_ptr<const Mat::CoordinateSystemProvider> anisotropy,
    Mixture::MixtureConstituent& constituent, Core::LinAlg::Matrix<3, 3>& G,
    Teuchos::ParameterList& params, int gp, int eleGID)
{
  // Start with zero prestretch
  G = Core::LinAlg::identity_matrix<3>();
}

void Mixture::IterativePrestressStrategy::update(
    const std::shared_ptr<const Mat::CoordinateSystemProvider> anisotropy,
    Mixture::MixtureConstituent& constituent, const Core::LinAlg::Matrix<3, 3>& F,
    Core::LinAlg::Matrix<3, 3>& G, Teuchos::ParameterList& params, int gp, int eleGID)
{
  // only update prestress if it is active
  if (!params_->is_active_) return;

  // Compute isochoric part of the deformation
  Core::LinAlg::Matrix<3, 3> F_bar;
  if (params_->isochoric_)
  {
    F_bar.update(std::pow(F.determinant(), -1.0 / 3.0), F);
  }
  else
  {
    F_bar.update(F);
  }

  // Compute new predeformation gradient
  Core::LinAlg::Matrix<3, 3> G_old(G);
  G.multiply_nn(F_bar, G_old);


  // Compute polar decomposition of the prestretch deformation gradient

  // Singular value decomposition of F = RU
  Core::LinAlg::Matrix<3, 3> Q(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> S(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> VT(Core::LinAlg::Initialization::zero);

  Core::LinAlg::svd<3, 3>(G, Q, S, VT);

  // Compute stretch tensor G = U = V * S * VT
  Core::LinAlg::Matrix<3, 3> VS;

  VS.multiply_tn(VT, S);
  G.multiply_nn(VS, VT);
}
FOUR_C_NAMESPACE_CLOSE
