// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_aniso_structuraltensor_strategy.hpp"

#include "4C_fem_general_utils_integration.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_mat_service.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN


Mat::Elastic::PAR::StructuralTensorParameter::StructuralTensorParameter(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      c1_(matdata.parameters.get<double>("C1")),
      c2_(matdata.parameters.get<double>("C2")),
      c3_(matdata.parameters.get<double>("C3")),
      c4_(matdata.parameters.get<double>("C4")),
      distribution_type_(distr_type_undefined),
      strategy_type_(strategy_type_undefined)
{
  std::string strategy_type = matdata.parameters.get<std::string>("STRATEGY");
  if (strategy_type == "Standard")
    strategy_type_ = strategy_type_standard;
  else if (strategy_type == "ByDistributionFunction")
    strategy_type_ = strategy_type_bydistributionfunction;
  else if (strategy_type == "DispersedTransverselyIsotropic")
    strategy_type_ = strategy_type_dispersedtransverselyisotropic;
  else
    FOUR_C_THROW(
        "unknown strategy for evaluation of the structural tensor for anisotropic material.");

  std::string distr_type = matdata.parameters.get<std::string>("DISTR");
  if (distr_type == "vonMisesFisher")
  {
    distribution_type_ = distr_type_vonmisesfisher;

    if (c1_ == 0.0) FOUR_C_THROW("invalid parameter C1=0.0 for von Mises-Fisher distribution");
    if (c1_ > 500.0)
      FOUR_C_THROW(
          "von Mises-Fisher distribution with parameter C1>500 is too sharp.\n"
          "Mechanical behaviour is very close to fiber without any dispersion.\n"
          "Better switch to ELAST_StructuralTensor STRATEGY Standard");
  }
  else if (distr_type == "Bingham")
  {
    distribution_type_ = distr_type_bingham;

    if (c4_ == 0.0) FOUR_C_THROW("invalid parameter C4=0.0 for Bingham distribution");
  }
  else if (distr_type == "none" and strategy_type_ == strategy_type_bydistributionfunction)
    FOUR_C_THROW(
        "You chose structural tensor strategy 'ByDistributionFunction' but you forgot to specify "
        "the 'DISTR' parameter.\n"
        "Check the definitions of anisotropic materials in your input file.");
  else if (distr_type == "none" and
           (strategy_type_ == strategy_type_standard or
               strategy_type_ == strategy_type_dispersedtransverselyisotropic))
  { /* this is fine */
  }
  else
    FOUR_C_THROW("Invalid choice of parameter 'DISTR' in anisotropic material definition.");
}

Mat::Elastic::StructuralTensorStrategyBase::StructuralTensorStrategyBase(
    Mat::Elastic::PAR::StructuralTensorParameter* params)
    : params_(params)
{
}

Mat::Elastic::StructuralTensorStrategyStandard::StructuralTensorStrategyStandard(
    Mat::Elastic::PAR::StructuralTensorParameter* params)
    : StructuralTensorStrategyBase(params)
{
}

Mat::Elastic::StructuralTensorStrategyByDistributionFunction::
    StructuralTensorStrategyByDistributionFunction(
        Mat::Elastic::PAR::StructuralTensorParameter* params)
    : StructuralTensorStrategyBase(params)
{
}

Mat::Elastic::StructuralTensorStrategyDispersedTransverselyIsotropic::
    StructuralTensorStrategyDispersedTransverselyIsotropic(
        Mat::Elastic::PAR::StructuralTensorParameter* params)
    : StructuralTensorStrategyBase(params)
{
}

void Mat::Elastic::StructuralTensorStrategyBase::dyadic_product(
    const Core::LinAlg::Matrix<3, 1>& M, Core::LinAlg::Matrix<6, 1>& result)
{
  for (int i = 0; i < 3; ++i) result(i) = M(i) * M(i);

  result(3) = M(0) * M(1);
  result(4) = M(1) * M(2);
  result(5) = M(0) * M(2);
}

void Mat::Elastic::StructuralTensorStrategyBase::dyadic_product(
    const Core::LinAlg::Matrix<3, 1>& M, Core::LinAlg::Matrix<3, 3>& result)
{
  result.multiply_nt(M, M);
}

void Mat::Elastic::StructuralTensorStrategyStandard::setup_structural_tensor(
    const Core::LinAlg::Matrix<3, 1>& fiber_vector,
    Core::LinAlg::Matrix<6, 1>& structural_tensor_stress)
{
  dyadic_product(fiber_vector, structural_tensor_stress);
}

void Mat::Elastic::StructuralTensorStrategyStandard::setup_structural_tensor(
    const Core::LinAlg::Matrix<3, 1>& fiber_vector, Core::LinAlg::Matrix<3, 3>& structural_tensor)
{
  dyadic_product(fiber_vector, structural_tensor);
}

void Mat::Elastic::StructuralTensorStrategyByDistributionFunction::setup_structural_tensor(
    const Core::LinAlg::Matrix<3, 1>& fiber_vector,
    Core::LinAlg::Matrix<6, 1>& structural_tensor_stress)
{
  const Core::FE::IntegrationPoints1D gausspoints(Core::FE::GaussRule1D::line_50point);
  Core::LinAlg::Matrix<numbgp, twice> rho;

  // constants for distribution around fiber_vector
  double c1 = params_->c1_;
  double c2 = params_->c2_;
  double c3 = params_->c3_;
  double c4 = params_->c4_;

  // current direction for pair (i,j)
  Core::LinAlg::Matrix<3, 1> x;

  // aux mean direction of fiber
  // we evaluate the structural tensor with this mean direction.
  // note: used only in case of von mises-fisher distribution
  double theta_aux = acos(gausspoints.qxg[numbgp - 1][0]);
  Core::LinAlg::Matrix<3, 1> aux_fiber_vector;
  aux_fiber_vector(0) = sin(theta_aux);
  aux_fiber_vector(1) = 0.0;
  aux_fiber_vector(2) = gausspoints.qxg[numbgp - 1][0];  // = cos(theta_aux)

  // ensure that the structural tensor is empty on input
  structural_tensor_stress.clear();

  // gauss integration over sphere
  // see Atkinson 1982
  for (int j = 0; j < twice; j++)
  {
    for (int i = 0; i < numbgp; i++)
    {
      double theta = acos(gausspoints.qxg[i][0]);
      double phi = (((double)(j)) * M_PI) / ((double)gausspoints.nquad);

      x(0) = sin(theta) * cos(phi);
      x(1) = sin(theta) * sin(phi);
      x(2) = cos(theta);

      switch (params_->distribution_type_)
      {
        case Mat::Elastic::PAR::distr_type_vonmisesfisher:
        {
          double c = c1 / (sinh(c1) * 4.0 * M_PI);
          double arg = aux_fiber_vector.dot(x);
          rho(i, j) = c * exp(c1 * arg);
          break;
        }
        case Mat::Elastic::PAR::distr_type_bingham:
        {
          double K = (sin(theta) * cos(phi)) / cos(theta);
          double X1 = x(0) * x(0);
          double X2 = x(1) * x(1) * (pow(K, 2.0) / (1.0 + pow(K, 2.0)));
          double X3 = x(2) * x(2);
          rho(i, j) = (1.0 / c4) * exp(c1 * X1 + c2 * X2 + c3 * X3);
          break;
        }
        default:
          FOUR_C_THROW("Unknown type of distribution function requested.");
      }  // switch

      // integration fac
      double fac = (M_PI * gausspoints.qwgt[i] * rho(i, j)) / ((double)numbgp);

      structural_tensor_stress(0) += fac * x(0) * x(0);  // A_11
      structural_tensor_stress(1) += fac * x(1) * x(1);  // A_22
      structural_tensor_stress(2) += fac * x(2) * x(2);  // A_33
      structural_tensor_stress(3) += fac * x(0) * x(1);  // A_12
      structural_tensor_stress(4) += fac * x(1) * x(2);  // A_23
      structural_tensor_stress(5) += fac * x(0) * x(2);  // A_13
    }  // loop over i
  }  // loop over j

  // after we evaluated the structural tensor in the auxiliary direction,
  // we rotate the tensor to the desired fiber orientation.
  if (params_->distribution_type_ == Mat::Elastic::PAR::distr_type_vonmisesfisher)
  {
    // base vectors
    Core::LinAlg::Matrix<3, 1> e1(Core::LinAlg::Initialization::zero);
    e1(0) = 1.0;
    Core::LinAlg::Matrix<3, 1> e2(Core::LinAlg::Initialization::zero);
    e2(1) = 1.0;
    Core::LinAlg::Matrix<3, 1> e3(Core::LinAlg::Initialization::zero);
    e3(2) = 1.0;

    // x1-x2 plane projection of fiber_vector
    Core::LinAlg::Matrix<3, 1> fiber_vector_proj(fiber_vector);
    fiber_vector_proj(2) = 0.0;
    double norm = fiber_vector_proj.norm2();
    if (norm > 1e-12) fiber_vector_proj.scale(1.0 / norm);

    // angles phi and theta for desired mean fiber direction
    double phi_0 = -1.0;
    if (norm < 1e-12)
      phi_0 = 0.0;
    else
      phi_0 = acos(e1.dot(fiber_vector_proj));    //< azimuth (measured from e1)
    double theta_0 = acos(e3.dot(fiber_vector));  //< elevation (measured from e3)

    // rotation angles
    double alpha = -(theta_0 - theta_aux);  //< rotation around x1-axis
    double beta = -phi_0;                   //< rotation around x3-axis

    Core::LinAlg::Matrix<3, 3> rotation1(Core::LinAlg::Initialization::zero);
    rotation1(0, 0) = cos(alpha);
    rotation1(1, 1) = 1.0;
    rotation1(2, 2) = cos(alpha);
    rotation1(0, 2) = -sin(alpha);
    rotation1(2, 0) = sin(alpha);

    Core::LinAlg::Matrix<3, 3> rotation2(Core::LinAlg::Initialization::zero);
    rotation2(0, 0) = cos(beta);
    rotation2(1, 1) = cos(beta);
    rotation2(2, 2) = 1.0;
    rotation2(0, 1) = sin(beta);
    rotation2(1, 0) = -sin(beta);

    Core::LinAlg::Matrix<3, 3> rotation(Core::LinAlg::Initialization::zero);
    rotation.multiply(rotation2, rotation1);

    Core::LinAlg::Matrix<3, 3> tensor3x3;
    tensor3x3(0, 0) = structural_tensor_stress(0);
    tensor3x3(1, 1) = structural_tensor_stress(1);
    tensor3x3(2, 2) = structural_tensor_stress(2);
    tensor3x3(0, 1) = structural_tensor_stress(3);
    tensor3x3(1, 2) = structural_tensor_stress(4);
    tensor3x3(0, 2) = structural_tensor_stress(5);
    tensor3x3(1, 0) = tensor3x3(0, 1);
    tensor3x3(2, 0) = tensor3x3(0, 2);
    tensor3x3(2, 1) = tensor3x3(1, 2);

    Core::LinAlg::Matrix<3, 3> temp(Core::LinAlg::Initialization::zero);
    temp.multiply_nn(rotation, tensor3x3);
    tensor3x3.clear();
    tensor3x3.multiply_nt(temp, rotation);

    structural_tensor_stress(0) = tensor3x3(0, 0);
    structural_tensor_stress(1) = tensor3x3(1, 1);
    structural_tensor_stress(2) = tensor3x3(2, 2);
    structural_tensor_stress(3) = tensor3x3(0, 1);
    structural_tensor_stress(4) = tensor3x3(1, 2);
    structural_tensor_stress(5) = tensor3x3(0, 2);
  }  // if distr_type_vonmisesfisher

  // zero out small entries
  const double tol = get_residual_tol();
  for (unsigned i = 0; i < structural_tensor_stress.num_rows(); ++i)
    if (abs(structural_tensor_stress(i)) < tol) structural_tensor_stress(i) = 0.0;

  // scale whole structural tensor with its trace, because
  // the trace might deviate slightly from the value 1 due
  // to the integration error.
  double trace =
      structural_tensor_stress(0) + structural_tensor_stress(1) + structural_tensor_stress(2);
  structural_tensor_stress.scale(1.0 / trace);
}

void Mat::Elastic::StructuralTensorStrategyByDistributionFunction::setup_structural_tensor(
    const Core::LinAlg::Matrix<3, 1>& fiber_vector, Core::LinAlg::Matrix<3, 3>& structural_tensor)
{
  Core::LinAlg::Matrix<6, 1> structural_tensor_stress;
  setup_structural_tensor(fiber_vector, structural_tensor_stress);
  Core::LinAlg::Voigt::Stresses::vector_to_matrix(structural_tensor_stress, structural_tensor);
}

void Mat::Elastic::StructuralTensorStrategyDispersedTransverselyIsotropic::setup_structural_tensor(
    const Core::LinAlg::Matrix<3, 1>& fiber_vector,
    Core::LinAlg::Matrix<6, 1>& structural_tensor_stress)
{
  // constant for dispersion around fiber_vector
  double c1 = params_->c1_;

  dyadic_product(fiber_vector, structural_tensor_stress);

  Core::LinAlg::Matrix<6, 1> Identity(Core::LinAlg::Initialization::zero);
  Identity(0) = 1.0;
  Identity(1) = 1.0;
  Identity(2) = 1.0;

  structural_tensor_stress.update(c1, Identity, (1.0 - 3.0 * c1));
}

void Mat::Elastic::StructuralTensorStrategyDispersedTransverselyIsotropic::setup_structural_tensor(
    const Core::LinAlg::Matrix<3, 1>& fiber_vector, Core::LinAlg::Matrix<3, 3>& structural_tensor)
{
  Core::LinAlg::Matrix<6, 1> structural_tensor_stress;
  setup_structural_tensor(fiber_vector, structural_tensor_stress);
  Core::LinAlg::Voigt::Stresses::vector_to_matrix(structural_tensor_stress, structural_tensor);
}

double Mat::Elastic::StructuralTensorStrategyBase::get_residual_tol()
{
  double restol = -1.0;
  Global::Problem* gprob = Global::Problem::instance();
  restol = gprob->structural_dynamic_params().get<double>("TOLRES");

  return restol;
}
FOUR_C_NAMESPACE_CLOSE
