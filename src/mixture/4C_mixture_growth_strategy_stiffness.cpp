/*----------------------------------------------------------------------*/
/*! \file
\brief Definition of a penalty term like growth strategy

\level 3
*/
/*----------------------------------------------------------------------*/

#include "4C_mixture_growth_strategy_stiffness.hpp"

#include "4C_linalg_fixedsizematrix_generators.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_mat_service.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_mixture_growth_strategy.hpp"

FOUR_C_NAMESPACE_OPEN

MIXTURE::PAR::StiffnessGrowthStrategy::StiffnessGrowthStrategy(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : MIXTURE::PAR::MixtureGrowthStrategy(matdata), kappa_(matdata.parameters.get<double>("KAPPA"))
{
}

std::unique_ptr<MIXTURE::MixtureGrowthStrategy>
MIXTURE::PAR::StiffnessGrowthStrategy::create_growth_strategy()
{
  return std::make_unique<MIXTURE::StiffnessGrowthStrategy>(this);
}

MIXTURE::StiffnessGrowthStrategy::StiffnessGrowthStrategy(
    MIXTURE::PAR::StiffnessGrowthStrategy* params)
    : params_(params)
{
}

void MIXTURE::StiffnessGrowthStrategy::evaluate_inverse_growth_deformation_gradient(
    Core::LinAlg::Matrix<3, 3>& iFgM, const MIXTURE::MixtureRule& mixtureRule,
    double currentReferenceGrowthScalar, int gp) const
{
  iFgM = Core::LinAlg::identity_matrix<3>();
}

void MIXTURE::StiffnessGrowthStrategy::evaluate_growth_stress_cmat(
    const MIXTURE::MixtureRule& mixtureRule, double currentReferenceGrowthScalar,
    const Core::LinAlg::Matrix<1, 6>& dCurrentReferenceGrowthScalarDC,
    const Core::LinAlg::Matrix<3, 3>& F, const Core::LinAlg::Matrix<6, 1>& E_strain,
    Teuchos::ParameterList& params, Core::LinAlg::Matrix<6, 1>& S_stress,
    Core::LinAlg::Matrix<6, 6>& cmat, const int gp, const int eleGID) const
{
  Core::LinAlg::Matrix<3, 3> iC(false);
  iC.multiply_tn(F, F);
  iC.invert();

  Core::LinAlg::Matrix<6, 1> iC_stress(false);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(iC, iC_stress);

  const double kappa = params_->kappa_;
  const double detF = F.determinant();
  const double I3 = detF * detF;

  const double dPi = 0.5 * kappa * (1.0 - currentReferenceGrowthScalar / detF);
  const double ddPi = 0.25 * kappa * currentReferenceGrowthScalar / std::pow(detF, 3);

  const double ddPiDGrowthScalar = -0.5 * kappa / detF;

  const double gamma2 = 2.0 * I3 * dPi;
  const double dgamma2DGrowthScalar = 4.0 * I3 * ddPiDGrowthScalar;
  const double delta5 = 4. * (I3 * dPi + I3 * I3 * ddPi);
  const double delta6 = -4.0 * I3 * dPi;


  S_stress.update(gamma2, iC_stress, 0.0);

  // contribution: Cinv \otimes Cinv
  cmat.multiply_nt(delta5, iC_stress, iC_stress, 0.0);
  // contribution: Cinv \odot Cinv
  Mat::add_holzapfel_product(cmat, iC_stress, delta6);

  cmat.multiply_nn(dgamma2DGrowthScalar, iC_stress, dCurrentReferenceGrowthScalarDC, 1.0);
}
FOUR_C_NAMESPACE_CLOSE
