// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mixture_constituent_elasthyper.hpp"

#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_mat_mixture.hpp"
#include "4C_mat_multiplicative_split_defgrad_elasthyper_service.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mixture_prestress_strategy.hpp"

FOUR_C_NAMESPACE_OPEN

// Constructor for the parameter class
Mixture::PAR::MixtureConstituentElastHyper::MixtureConstituentElastHyper(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : MixtureConstituentElastHyperBase(matdata)
{
  // do nothing
}

// Create an instance of Mixture::MixtureConstituentElastHyper from the parameters
std::unique_ptr<Mixture::MixtureConstituent>
Mixture::PAR::MixtureConstituentElastHyper::create_constituent(int id)
{
  return std::unique_ptr<Mixture::MixtureConstituentElastHyper>(
      new Mixture::MixtureConstituentElastHyper(this, id));
}

// Constructor of the constituent holding the material parameters
Mixture::MixtureConstituentElastHyper::MixtureConstituentElastHyper(
    Mixture::PAR::MixtureConstituentElastHyper* params, int id)
    : MixtureConstituentElastHyperBase(params, id)
{
}

// Returns the material type
Core::Materials::MaterialType Mixture::MixtureConstituentElastHyper::material_type() const
{
  return Core::Materials::mix_elasthyper;
}

// Evaluates the stress of the constituent
void Mixture::MixtureConstituentElastHyper::evaluate(const Core::LinAlg::Tensor<double, 3, 3>& F,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& E_strain,
    const Teuchos::ParameterList& params, const Mat::EvaluationContext& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& S_stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, const int gp, const int eleGID)
{
  if (prestress_strategy() != nullptr)
  {
    Core::LinAlg::Matrix<6, 1> S_view = Core::LinAlg::make_stress_like_voigt_view(S_stress);
    Core::LinAlg::Matrix<6, 6> cmat_view = Core::LinAlg::make_stress_like_voigt_view(cmat);

    Mat::elast_hyper_evaluate_elastic_part(Core::LinAlg::make_matrix_view(F),
        Core::LinAlg::make_matrix(Core::LinAlg::get_full(prestretch_tensor(gp))), S_view, cmat_view,
        summands(), summand_properties(), gp, eleGID);
  }
  else
  {
    // Evaluate stresses using ElastHyper service functions
    Mat::elast_hyper_evaluate(
        F, E_strain, params, S_stress, cmat, gp, eleGID, summands(), summand_properties(), false);
  }
}

// Compute the stress resultant with incorporating an elastic and inelastic part of the deformation
void Mixture::MixtureConstituentElastHyper::evaluate_elastic_part(
    const Core::LinAlg::Tensor<double, 3, 3>& F, const Core::LinAlg::Tensor<double, 3, 3>& iFextin,
    const Teuchos::ParameterList& params, const Mat::EvaluationContext& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& S_stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  Core::LinAlg::Tensor<double, 3, 3> iFin = iFextin * prestretch_tensor(gp);

  Core::LinAlg::Matrix<6, 1> S_view = Core::LinAlg::make_stress_like_voigt_view(S_stress);
  Core::LinAlg::Matrix<6, 6> cmat_view = Core::LinAlg::make_stress_like_voigt_view(cmat);

  Mat::elast_hyper_evaluate_elastic_part(Core::LinAlg::make_matrix_view(F),
      Core::LinAlg::make_matrix_view(iFin), S_view, cmat_view, summands(), summand_properties(), gp,
      eleGID);
}
FOUR_C_NAMESPACE_CLOSE
