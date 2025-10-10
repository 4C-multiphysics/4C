// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_viscoplastic_no_yield_surface.hpp"

#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_utils_densematrix_eigen.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_fad.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

// struct definition
struct Mat::PreCalculatedTerms
{
  double equ_tens_stress_const;
  double flow_res_const;
  double equ_tens_stress_flow_res_ratio;
  double flow_res_sat_deviation;
};

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::PAR::ViscoPlasticNoYieldSurface::ViscoPlasticNoYieldSurface(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      density_(matdata.parameters.get<double>("DENS")),
      nue_(matdata.parameters.get<double>("NUE")),
      young_(matdata.parameters.get<double>("YOUNG")),
      temperature_(matdata.parameters.get<double>("TEMPERATURE")),
      pre_exp_fac_(matdata.parameters.get<double>("PRE_EXP_FAC")),
      activation_energy_(matdata.parameters.get<double>("ACTIVATION_ENERGY")),
      gas_constant_(matdata.parameters.get<double>("GAS_CONSTANT")),
      strain_rate_sensitivity_(matdata.parameters.get<double>("STRAIN_RATE_SENS")),
      init_flow_res_(matdata.parameters.get<double>("INIT_FLOW_RES")),
      flow_res_exp_(matdata.parameters.get<double>("FLOW_RES_EXP")),
      flow_res_pre_fac_(matdata.parameters.get<double>("FLOW_RES_PRE_FAC")),
      flow_res_sat_fac_(matdata.parameters.get<double>("FLOW_RES_SAT_FAC")),
      flow_res_sat_exp_(matdata.parameters.get<double>("FLOW_RES_SAT_EXP"))
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::ViscoPlasticNoYieldSurface::create_material()
{
  return std::make_shared<Mat::ViscoPlasticNoYieldSurface>(this);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::ViscoPlasticNoYieldSurfaceType Mat::ViscoPlasticNoYieldSurfaceType::instance_;

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::Communication::ParObject* Mat::ViscoPlasticNoYieldSurfaceType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* visco_plastic_no_yield_surface = new Mat::ViscoPlasticNoYieldSurface();
  visco_plastic_no_yield_surface->unpack(buffer);
  return visco_plastic_no_yield_surface;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::ViscoPlasticNoYieldSurface::ViscoPlasticNoYieldSurface() : params_(nullptr) {}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::ViscoPlasticNoYieldSurface::ViscoPlasticNoYieldSurface(
    Mat::PAR::ViscoPlasticNoYieldSurface* params)
    : params_(params)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::ViscoPlasticNoYieldSurface::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  // in case we are in post-process mode
  if (parameter() != nullptr) matid = parameter()->id();
  add_to_pack(data, matid);

  // insert last converged states
  add_to_pack(data, last_plastic_defgrd_inverse_);
  add_to_pack(data, last_flowres_isotropic_);

  // insert current iteration states
  add_to_pack(data, current_plastic_defgrd_inverse_);
  add_to_pack(data, current_flowres_isotropic_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::ViscoPlasticNoYieldSurface::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid and recover params_
  int matid;
  extract_from_pack(buffer, matid);
  params_ = nullptr;
  if (Global::Problem::instance()->materials() != nullptr)
  {
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
        params_ = dynamic_cast<Mat::PAR::ViscoPlasticNoYieldSurface*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
  }

  // last converged states are unpacked
  extract_from_pack(buffer, last_plastic_defgrd_inverse_);
  extract_from_pack(buffer, last_flowres_isotropic_);

  // current iteration states are unpacked
  extract_from_pack(buffer, current_plastic_defgrd_inverse_);
  extract_from_pack(buffer, current_flowres_isotropic_);
}

/*---------------------------------------------------------------------*
 *---------------------------------------------------------------------*/
void Mat::ViscoPlasticNoYieldSurface::setup(const int numgp,
    const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  // read initial flow resistance from input parameter container
  last_flowres_isotropic_.resize(numgp, params_->init_flow_res());

  // initialize last inverse plastic deformation gradient as identity
  Core::LinAlg::Matrix<3, 3> id2(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; ++i) id2(i, i) = 1.0;
  last_plastic_defgrd_inverse_.resize(numgp, id2);

  // initialize current history variables (values do not matter)
  current_flowres_isotropic_.resize(numgp, 0.0);
  current_plastic_defgrd_inverse_.resize(numgp, id2);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::ViscoPlasticNoYieldSurface::update()
{
  // this update is done after the global newton loop is converged
  last_plastic_defgrd_inverse_ = current_plastic_defgrd_inverse_;
  last_flowres_isotropic_ = current_flowres_isotropic_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::ViscoPlasticNoYieldSurface::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  const Core::LinAlg::Matrix<3, 3> defgrd_mat = Core::LinAlg::make_matrix_view(*defgrad);
  Core::LinAlg::Matrix<6, 1> stress_view = Core::LinAlg::make_stress_like_voigt_view(stress);
  Core::LinAlg::Matrix<6, 6> cmat_view = Core::LinAlg::make_stress_like_voigt_view(cmat);

  // read input and history variables
  FOUR_C_ASSERT(context.time_step_size, "Time step size not given in evaluation context.");
  const double dt = *context.time_step_size;
  const Core::LinAlg::Matrix<3, 3>& last_iFv = last_plastic_defgrd_inverse_[gp];

  // trial (purely elastic) deformation gradient
  static Core::LinAlg::Matrix<3, 3> Fe_trial;
  Fe_trial.multiply_nn(defgrd_mat, last_iFv);

  // define variables for eigenvalue analysis
  static Core::LinAlg::Matrix<3, 1> EigenvaluesFe_trial;
  static Core::LinAlg::Matrix<3, 3> EigenvectorsFe_trial;
  calculate_trial_elastic_defgrad_eigenvalues_and_eigenvectors(
      Fe_trial, EigenvaluesFe_trial, EigenvectorsFe_trial);

  // elastic trial rotation
  static Core::LinAlg::Matrix<3, 3> Re_trial;
  Re_trial = calculate_trial_elastic_rotation(Fe_trial, EigenvectorsFe_trial, EigenvaluesFe_trial);

  // logarithmic strain
  static Core::LinAlg::Matrix<6, 1> Ee_trial_Vstrain;
  Ee_trial_Vstrain = calculate_log_elastic_strain_in_strain_like_voigt_notation(
      EigenvectorsFe_trial, EigenvaluesFe_trial);

  // setup elasticity tensor (stress-stress-like)
  setup_cmat(cmat_view);

  // stress-like Voigt notation of stresses conjugated to logarithmic strains
  static Core::LinAlg::Matrix<6, 1> Me_trial_Vstress;
  // trial stress conjugate to logarithmic strain
  Me_trial_Vstress.multiply_nn(cmat_view, Ee_trial_Vstrain);

  // mean normal pressure (p = 1/3 * trace(Me_trial))
  const double p = 1.0 / 3.0 * (Me_trial_Vstress(0) + Me_trial_Vstress(1) + Me_trial_Vstress(2));

  static Core::LinAlg::Matrix<3, 3> Me_trial_dev;
  Me_trial_dev = calculate_deviatoric_trial_stresses(Me_trial_Vstress, p);
  const double Me_trial_eqv = calculate_trial_equivalent_stress(Me_trial_dev);

  // solution vector with variables x(0) = Me_eqv;  x(1) = S
  static Core::LinAlg::Matrix<2, 1> x;
  x(0) = Me_trial_eqv;
  x(1) = last_flowres_isotropic_[gp];

  // default value of eta = 1 (no plastic deformation) if trial equivalent stress is zero and no
  // mapping is needed
  double eta = 1.0;

  // do mapping if trial equivalent stress is not zero
  if (Me_trial_eqv != 0.0)
  {
    local_newton_loop(x, dt);
    // calculate ratio of actual elastic stress (first component of solution vector) to trial stress
    eta = x(0) / Me_trial_eqv;
  }

  // conjugate stress (Me = eta * Me_trial_dev - p * id2)
  static Core::LinAlg::Matrix<3, 3> Me;
  // identity matrix
  static Core::LinAlg::Matrix<3, 3> id2(Core::LinAlg::Initialization::zero);
  for (unsigned i = 0; i < 3; ++i) id2(i, i) = 1.0;
  Me.update(eta, Me_trial_dev, p, id2);

  static Core::LinAlg::Matrix<3, 3> PK2;
  PK2 = calculate_second_piola_kirchhoff_stresses(&defgrd_mat, Re_trial, Me);

  // current inverse plastic deformation gradient
  static Core::LinAlg::Matrix<3, 3> current_iFv;
  current_iFv = calculate_updated_inverse_viscous_defgrad(
      last_iFv, EigenvectorsFe_trial, EigenvaluesFe_trial, eta);

  // update history variables of current Gauss-point
  current_plastic_defgrd_inverse_[gp] = current_iFv;
  current_flowres_isotropic_[gp] = x(1);

  // transform stresses to stress-like Voigt notation
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(PK2, stress_view);

  auto cmatel = calculate_elastic_stiffness(EigenvectorsFe_trial, EigenvaluesFe_trial);

  cmat_view = Mat::pull_back_four_tensor(1.0 / current_iFv.determinant(), current_iFv, cmatel);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::Matrix<3, 3>& Mat::ViscoPlasticNoYieldSurface::calculate_deviatoric_trial_stresses(
    const Core::LinAlg::Matrix<6, 1>& Me_trial_Vstress, const double p) const
{
  // deviatoric trial stress in stress-like Voigt notation
  static Core::LinAlg::Matrix<6, 1> Me_trial_dev_Vstress;
  for (unsigned i = 0; i < 3; ++i) Me_trial_dev_Vstress(i) = Me_trial_Vstress(i) - p;
  for (unsigned i = 3; i < 6; ++i) Me_trial_dev_Vstress(i) = Me_trial_Vstress(i);

  static Core::LinAlg::Matrix<3, 3> Me_trial_dev;
  // transform deviatoric equivalent stress to matrix notation
  Core::LinAlg::Voigt::Stresses::vector_to_matrix(Me_trial_dev_Vstress, Me_trial_dev);

  return Me_trial_dev;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::Matrix<3, 3>&
Mat::ViscoPlasticNoYieldSurface::calculate_updated_inverse_viscous_defgrad(
    const Core::LinAlg::Matrix<3, 3>& last_iFv, const Core::LinAlg::Matrix<3, 3>& eigen_vectors,
    const Core::LinAlg::Matrix<3, 1>& eigen_values, const double eta) const
{
  static Core::LinAlg::Matrix<3, 3> scaledEigenvalues(Core::LinAlg::Initialization::zero);

  // cubic root of determinant
  const double cubicRoot_stretchTensorTrial_determinant =
      std::cbrt(eigen_values(0) * eigen_values(1) * eigen_values(2));

  for (unsigned i = 0; i < 3; ++i)
  {
    scaledEigenvalues(i, i) =
        std::pow((eigen_values(i) / cubicRoot_stretchTensorTrial_determinant), (eta - 1.0));
  }

  static Core::LinAlg::Matrix<3, 3> tmp3x3;
  static Core::LinAlg::Matrix<3, 3> scaled_iUeTrialDeviatoric;
  static Core::LinAlg::Matrix<3, 3> current_iFv;
  // calculate inverse scaled trial stretch tensor
  tmp3x3.multiply_nn(eigen_vectors, scaledEigenvalues);
  scaled_iUeTrialDeviatoric.multiply_nt(tmp3x3, eigen_vectors);
  current_iFv.multiply_nn(last_iFv, scaled_iUeTrialDeviatoric);

  return current_iFv;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::ViscoPlasticNoYieldSurface::calculate_trial_equivalent_stress(
    const Core::LinAlg::Matrix<3, 3>& Me_trial_dev) const
{
  static Core::LinAlg::Matrix<3, 3> Me_trial_dev_squared;
  Me_trial_dev_squared.multiply(Me_trial_dev, Me_trial_dev);

  // calculate trace
  double trace = 0.0;
  for (unsigned i = 0; i < 3; ++i)
  {
    trace += Me_trial_dev_squared(i, i);
  }

  // trial equivalent stress = sqrt(3/2 * trace(Me_trial_dev . Me_trial_dev))
  return std::sqrt(3.0 / 2.0 * trace);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::ViscoPlasticNoYieldSurface::calculate_trial_elastic_defgrad_eigenvalues_and_eigenvectors(
    const Core::LinAlg::Matrix<3, 3>& Fe_trial, Core::LinAlg::Matrix<3, 1>& eigen_values,
    Core::LinAlg::Matrix<3, 3>& eigen_vectors) const
{
  // as we only have a method to calculate the eigenvalues and eigenvectors from symmetric matrices,
  // we calculate the right Cauchy-Green tensor first and then make use of the fact that we get the
  // squared eigen values

  // trial right Cauchy-Green tensor
  static Core::LinAlg::Matrix<3, 3> Ce_trial;
  Ce_trial.multiply_tn(Fe_trial, Fe_trial);

  // squared principal stretches
  static Core::LinAlg::Matrix<3, 3> SquaredEigenvalues;
  // eigenvalue analysis of trial deformation
  Core::LinAlg::syev(Ce_trial, SquaredEigenvalues, eigen_vectors);

  // principal stretches
  for (unsigned i = 0; i < 3; ++i)
  {
    eigen_values(i) = std::sqrt(SquaredEigenvalues(i, i));
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D>&
Mat::ViscoPlasticNoYieldSurface::calculate_elastic_stiffness(
    const Core::LinAlg::Matrix<3, 3>& eigen_vectors,
    const Core::LinAlg::Matrix<3, 1>& eigen_values) const
{
  // init and clear elastic stiffness matrix
  static Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D> Ce;
  Ce.clear();
  const double eps(1.0e-12);

  // elastic parameters
  const double E = params_->young();
  const double nue = params_->nue();
  const double G = E / (2.0 * (1.0 + nue));
  const double K = E / (3.0 * (1.0 - 2.0 * nue));

  // pre-calculation of terms
  static Core::LinAlg::Matrix<3, 1> LogEigenValues;
  static Core::LinAlg::Matrix<3, 1> SquaredEigenValues;
  static Core::LinAlg::Matrix<3, 1> ithEigenVector;
  std::vector<Core::LinAlg::Matrix<3, 1>> EigenVectorsVec;
  for (unsigned i = 0; i < 3; ++i)
  {
    LogEigenValues(i) = std::log(eigen_values(i));
    SquaredEigenValues(i) = std::pow(eigen_values(i), 2.0);

    // extract ith-eigenvector and add it to the vector
    for (unsigned j = 0; j < 3; ++j) ithEigenVector(j) = eigen_vectors(j, i);
    EigenVectorsVec.push_back(ithEigenVector);
  }

  const double traceLogEigenvalues = LogEigenValues(0) + LogEigenValues(1) + LogEigenValues(2);

  // define outer products of eigenvectors
  static Core::LinAlg::Matrix<3, 3> temp;
  static Core::LinAlg::Matrix<6, 1> NaNaV, NbNbV, NaNbV, NbNaV;
  for (unsigned a = 0; a < 3; ++a)
  {
    temp.multiply_nt(EigenVectorsVec.at(a), EigenVectorsVec.at(a));
    Core::LinAlg::Voigt::Stresses::matrix_to_vector(temp, NaNaV);

    const double Se_a = (2.0 * G * (LogEigenValues(a) - 1.0 / 3.0 * traceLogEigenvalues) +
                            K * traceLogEigenvalues) /
                        SquaredEigenValues(a);

    for (unsigned b = 0; b < 3; ++b)
    {
      temp.multiply_nt(EigenVectorsVec.at(a), EigenVectorsVec.at(b));
      Core::LinAlg::Voigt::Stresses::matrix_to_vector(temp, NaNbV);
      temp.multiply_nt(EigenVectorsVec.at(b), EigenVectorsVec.at(a));
      Core::LinAlg::Voigt::Stresses::matrix_to_vector(temp, NbNaV);
      temp.multiply_nt(EigenVectorsVec.at(b), EigenVectorsVec.at(b));
      Core::LinAlg::Voigt::Stresses::matrix_to_vector(temp, NbNbV);

      const double Se_b = (2.0 * G * (LogEigenValues(b) - 1.0 / 3.0 * traceLogEigenvalues) +
                              K * traceLogEigenvalues) /
                          SquaredEigenValues(b);

      double Ce_term1_fac = -(2.0 / 3.0 * G - K) / (SquaredEigenValues(a) * SquaredEigenValues(b));
      double Ce_term2_fac(0.0);

      if (a == b)
      {
        Ce_term1_fac -=
            (2.0 * G * (LogEigenValues(a) - 1.0 / 3.0 * traceLogEigenvalues - 1.0 / 2.0) +
                K * traceLogEigenvalues) *
            2.0 / (SquaredEigenValues(a) * SquaredEigenValues(b));
      }
      else
      {
        // principle stretches are almost equal -> l'Hospital is used
        if (std::abs(SquaredEigenValues(a) - SquaredEigenValues(b)) < eps)
        {
          // add derivative of S_e,b w.r.t. lambda_e,b^2
          Ce_term2_fac = -(2.0 * G * (LogEigenValues(b) - 1.0 / 3.0 * (traceLogEigenvalues + 1.0)) +
                             K * (traceLogEigenvalues - 1.0 / 2.0)) *
                         2.0 / (SquaredEigenValues(b) * SquaredEigenValues(b));
          // subtract derivative of S_e,a w.r.t. lambda_e,bn^2
          Ce_term2_fac -= Ce_term1_fac;
        }
        // principle stretches differ -> normal formula can be used
        else
        {
          Ce_term2_fac = (Se_b - Se_a) / (SquaredEigenValues(b) - SquaredEigenValues(a));
        }
      }

      Ce.multiply_nt(Ce_term1_fac, NaNaV, NbNbV, 1.0);
      Ce.multiply_nt(Ce_term2_fac, NaNbV, NaNbV, 1.0);
      Ce.multiply_nt(Ce_term2_fac, NaNbV, NbNaV, 1.0);
    }
  }

  return Ce;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::Matrix<6, 1>&
Mat::ViscoPlasticNoYieldSurface::calculate_log_elastic_strain_in_strain_like_voigt_notation(
    const Core::LinAlg::Matrix<3, 3>& eigen_vectors,
    const Core::LinAlg::Matrix<3, 1>& eigen_values) const
{
  // trial elastic material logarithmic strains
  static Core::LinAlg::Matrix<3, 3> Ee_trial;
  static Core::LinAlg::Matrix<6, 1> Ee_trial_Vstrain;
  static Core::LinAlg::Matrix<3, 3> tmp3x3;
  static Core::LinAlg::Matrix<3, 3> logEigenValues(Core::LinAlg::Initialization::zero);

  for (unsigned i = 0; i < 3; ++i)
  {
    logEigenValues(i, i) = std::log(eigen_values(i));
  }

  tmp3x3.multiply_nn(eigen_vectors, logEigenValues);
  Ee_trial.multiply_nt(tmp3x3, eigen_vectors);

  // transform to strain-like Voigt notation of logarithmic elastic strain
  Core::LinAlg::Voigt::Strains::matrix_to_vector(Ee_trial, Ee_trial_Vstrain);

  return Ee_trial_Vstrain;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::Matrix<3, 3>& Mat::ViscoPlasticNoYieldSurface::calculate_trial_elastic_rotation(
    const Core::LinAlg::Matrix<3, 3>& Fe_trial, const Core::LinAlg::Matrix<3, 3>& eigen_vectors,
    const Core::LinAlg::Matrix<3, 1>& eigen_values) const
{
  static Core::LinAlg::Matrix<3, 3> Re_trial;
  static Core::LinAlg::Matrix<3, 3> tmp3x3;
  static Core::LinAlg::Matrix<3, 3> invEigenValues(Core::LinAlg::Initialization::zero);

  for (unsigned i = 0; i < 3; ++i)
  {
    invEigenValues(i, i) = 1.0 / eigen_values(i);
  }

  // trial elastic rotation tensor (F = R * U --> R = F * U^(-1))
  tmp3x3.multiply_nn(eigen_vectors, invEigenValues);
  // inverse trial stretch
  static Core::LinAlg::Matrix<3, 3> iUe_trial;
  iUe_trial.multiply_nt(tmp3x3, eigen_vectors);
  Re_trial.multiply_nn(Fe_trial, iUe_trial);

  return Re_trial;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::Matrix<3, 3>&
Mat::ViscoPlasticNoYieldSurface::calculate_second_piola_kirchhoff_stresses(
    const Core::LinAlg::Matrix<3, 3>* defgrd, const Core::LinAlg::Matrix<3, 3>& Re_trial,
    const Core::LinAlg::Matrix<3, 3>& Me) const
{
  static Core::LinAlg::Matrix<3, 3> iF;
  iF.invert(*defgrd);

  // calculate 2nd PK stress (PK2 = F^(-1) * Re_trial * Me * Re_trial^(T) * F^(-T))
  // from left to right
  static Core::LinAlg::Matrix<3, 3> PK2;
  static Core::LinAlg::Matrix<3, 3> tmp3x3;
  tmp3x3.multiply_nn(iF, Re_trial);
  PK2.multiply_nn(tmp3x3, Me);
  tmp3x3.multiply_nt(PK2, Re_trial);
  PK2.multiply_nt(tmp3x3, iF);

  return PK2;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::ViscoPlasticNoYieldSurface::local_newton_loop(
    Core::LinAlg::Matrix<2, 1>& x, const double dt)
{
  // predictor values
  const double equ_tens_trial_stress = x(0);
  const double flow_resistance_n = x(1);

  // Convergence tolerance for NR
  unsigned iter(0);
  const double tolNR(1.0e-12);
  const unsigned max_iter(100);
  // Jacobian matrix containing linearization
  static Core::LinAlg::Matrix<2, 2> J;
  // Inverse Jacobian
  static Core::LinAlg::Matrix<2, 2> iJ;
  // Increment of the solution variables
  static Core::LinAlg::Matrix<2, 1> dx;
  // Residual of both equations
  static Core::LinAlg::Matrix<2, 1> residual;

  // Newton Loop until convergence of the L2-norm, or maximum iteration reached
  while (true)
  {
    ++iter;

    // execute pre calculations
    const Mat::PreCalculatedTerms terms = pre_calculate_terms(x(0), x(1), dt);

    // Calculate residual and L2 Norm
    residual = calculate_residual(x(0), equ_tens_trial_stress, x(1), flow_resistance_n, terms);
    double residual_norm2 = residual.norm2();

    // skip if residual has converged
    if (residual_norm2 < tolNR) break;

    // Check if maximum iteration is reached
    if (iter > max_iter)
      FOUR_C_THROW("Local Newton Raphson in ViscoPlasticNoYieldSurface material not converged");

    // calculate linearization
    J = calculate_linearization(x(0), x(1), terms);

    // invert linearization and calculate increment
    iJ.invert(J);
    dx.multiply(-1.0, iJ, residual, 0.0);

    // update solution vector
    x.update(1.0, dx, 1.0);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::Matrix<2, 1>& Mat::ViscoPlasticNoYieldSurface::calculate_residual(
    const double equ_tens_stress_np, const double equ_tens_trial_stress,
    const double flow_resistance_np, const double flow_resistance_n,
    const PreCalculatedTerms& terms)
{
  // viscosity parameters
  const double m = params_->strain_rate_sensitivity();
  const double a = params_->flow_res_exp();

  static Core::LinAlg::Matrix<2, 1> residual;
  residual(0) =
      equ_tens_stress_np - equ_tens_trial_stress +
      terms.equ_tens_stress_const * std::pow(terms.equ_tens_stress_flow_res_ratio, (1.0 / m));

  const double temp_abs_pow = std::pow(std::abs(terms.flow_res_sat_deviation), a);
  const double flow_resistance_deriv_np = terms.flow_res_const * temp_abs_pow *
                                          Core::FADUtils::signum(terms.flow_res_sat_deviation) *
                                          std::pow(terms.equ_tens_stress_flow_res_ratio, (1.0 / m));

  residual(1) = flow_resistance_np - flow_resistance_n - flow_resistance_deriv_np;

  return residual;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::Matrix<2, 2>& Mat::ViscoPlasticNoYieldSurface::calculate_linearization(
    const double equ_tens_stress_np, const double flow_resistance_np,
    const Mat::PreCalculatedTerms& terms)
{
  static Core::LinAlg::Matrix<2, 2> J;

  // viscosity parameters
  const double m = params_->strain_rate_sensitivity();
  const double a = params_->flow_res_exp();
  const double b = params_->flow_res_sat_exp();
  const double flow_res_sat_fac = params_->flow_res_sat_fac();

  J(0, 0) = 1.0 + terms.equ_tens_stress_const / (m * equ_tens_stress_np) *
                      std::pow(terms.equ_tens_stress_flow_res_ratio, 1.0 / m);

  J(0, 1) = -terms.equ_tens_stress_const / (m * flow_resistance_np) *
            std::pow(terms.equ_tens_stress_flow_res_ratio, 1.0 / m);

  J(1, 0) = -terms.flow_res_const / (m * equ_tens_stress_np) *
                std::pow(std::abs(terms.flow_res_sat_deviation), a) *
                Core::FADUtils::signum(terms.flow_res_sat_deviation) *
                std::pow(terms.equ_tens_stress_flow_res_ratio, 1.0 / m) -
            terms.flow_res_const * b / (m * flow_res_sat_fac) * a *
                std::abs(terms.flow_res_sat_deviation) *
                std::pow(std::abs(terms.flow_res_sat_deviation), a - 2.0) *
                std::pow(terms.equ_tens_stress_flow_res_ratio, (1.0 - b - m) / m);

  J(1, 1) = 1.0 +
            terms.flow_res_const / (m * flow_resistance_np) *
                std::pow(std::abs(terms.flow_res_sat_deviation), a) *
                Core::FADUtils::signum(terms.flow_res_sat_deviation) *
                std::pow(terms.equ_tens_stress_flow_res_ratio, 1.0 / m) +
            terms.flow_res_const / flow_res_sat_fac *
                std::pow(terms.equ_tens_stress_flow_res_ratio, (1.0 - b) / m) * a *
                std::abs(terms.flow_res_sat_deviation) *
                std::pow(std::abs(terms.flow_res_sat_deviation), a - 2.0) * (1.0 + b / m);

  return J;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::PreCalculatedTerms Mat::ViscoPlasticNoYieldSurface::pre_calculate_terms(
    const double equ_tens_stress_np, const double flow_resistance_np, const double dt)
{
  Mat::PreCalculatedTerms terms;

  // Elasticity parameters
  const double E = params_->young();
  const double nu = params_->nue();
  const double G = E / (2.0 * (1.0 + nu));

  // viscosity parameters
  const double T = params_->temperature();
  const double A = params_->pre_exp_fac();
  const double Q = params_->activation_energy();
  const double R = params_->gas_constant();
  const double m = params_->strain_rate_sensitivity();
  const double H_0 = params_->flow_res_pre_fac();
  const double flow_res_sat_fac = params_->flow_res_sat_fac();
  const double b = params_->flow_res_sat_exp();

  const double expterm = std::exp(-Q / (R * T));
  terms.equ_tens_stress_const = 3.0 * G * dt * A * expterm;
  terms.flow_res_const = dt * H_0 * A * expterm;
  terms.equ_tens_stress_flow_res_ratio = equ_tens_stress_np / flow_resistance_np;
  terms.flow_res_sat_deviation =
      1.0 - flow_resistance_np /
                (flow_res_sat_fac * std::pow(terms.equ_tens_stress_flow_res_ratio, b / m));

  return terms;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::ViscoPlasticNoYieldSurface::setup_cmat(
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D>& cmat)
{
  // get material parameters
  // Young's modulus
  const double youngs_mod = params_->young();
  // Poisson's ratio
  const double nue = params_->nue();

  // isotropic elasticity tensor C in Voigt matrix notation
  //                       [ 1-nu     nu     nu |          0    0    0 ]
  //                       [        1-nu     nu |          0    0    0 ]
  //           E           [               1-nu |          0    0    0 ]
  //   C = --------------- [ ~~~~   ~~~~   ~~~~   ~~~~~~~~~~  ~~~  ~~~ ]
  //       (1+nu)*(1-2*nu) [                    | (1-2*nu)/2    0    0 ]
  //                       [                    |      (1-2*nu)/2    0 ]
  //                       [ symmetric          |           (1-2*nu)/2 ]
  //
  const double mfac = youngs_mod / ((1.0 + nue) * (1.0 - 2.0 * nue));  // factor

  cmat.clear();
  // write non-zero components --- axial
  cmat(0, 0) = mfac * (1.0 - nue);
  cmat(0, 1) = mfac * nue;
  cmat(0, 2) = mfac * nue;
  cmat(1, 0) = mfac * nue;
  cmat(1, 1) = mfac * (1.0 - nue);
  cmat(1, 2) = mfac * nue;
  cmat(2, 0) = mfac * nue;
  cmat(2, 1) = mfac * nue;
  cmat(2, 2) = mfac * (1.0 - nue);
  // write non-zero components --- shear
  cmat(3, 3) = mfac * 0.5 * (1.0 - 2.0 * nue);
  cmat(4, 4) = mfac * 0.5 * (1.0 - 2.0 * nue);
  cmat(5, 5) = mfac * 0.5 * (1.0 - 2.0 * nue);
}
FOUR_C_NAMESPACE_CLOSE
