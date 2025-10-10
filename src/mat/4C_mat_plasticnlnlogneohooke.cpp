// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_plasticnlnlogneohooke.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_utils_densematrix_eigen.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_function.hpp"
#include "4C_utils_local_newton.hpp"

FOUR_C_NAMESPACE_OPEN

namespace
{
  std::pair<double, double> residuum_and_jacobian_from_function(
      Mat::PAR::PlasticNlnLogNeoHooke* matparameter, double Dgamma, double dt, double accplstrain,
      double abs_dev_KH_trial, const Core::Utils::FunctionOfAnything& hardeningfunction)
  {
    const double ym = matparameter->youngs_;        // Young's modulus
    const double nu = matparameter->poissonratio_;  // Poisson's ratio
    const double G = ym / (2.0 * (1.0 + nu));       // shear modulus, mu=G
    // plastic material data
    const double visc = matparameter->visc_;            // viscosity
    const double eps = matparameter->rate_dependency_;  // rate dependency

    //! vector for input of accumulated strain to function
    std::vector<std::pair<std::string, double>> dp;
    dp.emplace_back("epsp", accplstrain + Dgamma);

    const double y_d = hardeningfunction.evaluate(dp, {}, 0);
    double y_d_visc = y_d * pow(visc * Dgamma / dt + 1., eps);

    const std::vector<double> dy_d_dgvector = hardeningfunction.evaluate_derivative(dp, {}, 0);
    const double dy_d_dgamma = dy_d_dgvector[0] * pow(visc * Dgamma / dt + 1., eps) +
                               y_d * eps * pow(visc * Dgamma / dt + 1., eps - 1) * visc / dt;

    double residuum = sqrt(3.0 / 2.0) * abs_dev_KH_trial - 3. * G * Dgamma - y_d_visc;
    double tangent = -3. * G - dy_d_dgamma;
    return {residuum, tangent};
  }

  std::pair<double, double> residuum_and_jacobian(Mat::PAR::PlasticNlnLogNeoHooke* matparameter,
      double Dgamma, double dt, double accplstrain_last, double abs_dev_KH_trial)
  {
    const double ym = matparameter->youngs_;        // Young's modulus
    const double nu = matparameter->poissonratio_;  // Poisson's ratio
    const double G = ym / (2.0 * (1.0 + nu));       // shear modulus, mu=G
    // plastic material data
    const double yield = matparameter->yield_;          // initial yield stress
    const double isohard = matparameter->isohard_;      // linear isotropic hardening
    const double infyield = matparameter->infyield_;    // saturation yield stress
    const double hardexp = matparameter->hardexp_;      // nonlinear hardening exponent
    const double visc = matparameter->visc_;            // viscosity
    const double eps = matparameter->rate_dependency_;  // rate dependency

    double accplstrain_curr = accplstrain_last + Dgamma;

    const double y_d = (yield + isohard * accplstrain_curr +
                        (infyield - yield) * (1. - exp(-hardexp * accplstrain_curr)));
    const double y_d_visc = y_d * pow(visc * Dgamma / dt + 1., eps);
    const double dy_d_dgamma =
        (isohard + (infyield - yield) * hardexp * exp(-hardexp * accplstrain_curr)) *
            pow(visc * Dgamma / dt + 1., eps) +
        y_d * eps * pow(visc * Dgamma / dt + 1., eps - 1) * visc / dt;

    double residuum = sqrt(3.0 / 2.0) * abs_dev_KH_trial - 3. * G * Dgamma - y_d_visc;
    double tangent = -3. * G - dy_d_dgamma;
    return {residuum, tangent};
  }

  std::pair<double, double> solve_newton_with_hardening_func(
      Mat::PAR::PlasticNlnLogNeoHooke* matparameter, double abs_dev_KH_trial,
      double accplstrain_last, double dt, const Core::Utils::FunctionOfAnything& hardening_function,
      bool error_tol)
  {
    const double visc = matparameter->visc_;            // viscosity
    const double eps = matparameter->rate_dependency_;  // rate dependency

    // local Newton iteration for nonlinear isotropic hardening
    const int maxiter = matparameter->max_iterations_;
    const double tol = matparameter->tolerance_nr_;

    auto residuumAndJacobianFromFunction = [&](double Dgamma)
    {
      return residuum_and_jacobian_from_function(
          matparameter, Dgamma, dt, accplstrain_last, abs_dev_KH_trial, hardening_function);
    };

    const double Dgamma =
        Core::Utils::solve_local_newton(residuumAndJacobianFromFunction, 0.0, tol, maxiter);

    //! vector for input of accumulated strain to function
    std::vector<std::pair<std::string, double>> dp;
    dp.emplace_back("epsp", accplstrain_last + Dgamma);
    const double y_d = hardening_function.evaluate(dp, {}, 0);

    std::vector<double> dy_d_dgvector = hardening_function.evaluate_derivative(dp, {}, 0);
    double dy_d_dgamma = dy_d_dgvector[0] * pow(visc * Dgamma / dt + 1., eps) +
                         y_d * eps * pow(visc * Dgamma / dt + 1., eps - 1) * visc / dt;

    return {Dgamma, dy_d_dgamma};
  }

  std::pair<double, double> solve_newton_with_parameters(
      Mat::PAR::PlasticNlnLogNeoHooke* matparameter, double abs_dev_KH_trial,
      double accplstrain_last, double dt, bool error_tol)
  {
    // plastic material data
    const double yield = matparameter->yield_;          // initial yield stress
    const double isohard = matparameter->isohard_;      // linear isotropic hardening
    const double infyield = matparameter->infyield_;    // saturation yield stress
    const double hardexp = matparameter->hardexp_;      // nonlinear hardening exponent
    const double visc = matparameter->visc_;            // viscosity
    const double eps = matparameter->rate_dependency_;  // rate dependency

    // local Newton iteration for nonlinear isotropic hardening
    const int maxiter = matparameter->max_iterations_;
    const double tol = matparameter->tolerance_nr_;

    auto residuumAndJacobian = [&](double Dgamma)
    { return residuum_and_jacobian(matparameter, Dgamma, dt, accplstrain_last, abs_dev_KH_trial); };

    const double Dgamma = Core::Utils::solve_local_newton(residuumAndJacobian, 0.0, tol, maxiter);

    const double accplstrain_curr = accplstrain_last + Dgamma;

    const double y_d = (yield + isohard * accplstrain_curr +
                        (infyield - yield) * (1. - exp(-hardexp * accplstrain_curr)));
    const double dy_d_dgamma =
        (isohard + (infyield - yield) * hardexp * exp(-hardexp * accplstrain_curr)) *
            pow(visc * Dgamma / dt + 1., eps) +
        y_d * eps * pow(visc * Dgamma / dt + 1., eps - 1) * visc / dt;

    return {Dgamma, dy_d_dgamma};
  }
}  // namespace



/*----------------------------------------------------------------------*
 | constructor (public)                                                 |
 *----------------------------------------------------------------------*/
Mat::PAR::PlasticNlnLogNeoHooke::PlasticNlnLogNeoHooke(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      youngs_(matdata.parameters.get<double>("YOUNG")),
      poissonratio_(matdata.parameters.get<double>("NUE")),
      density_(matdata.parameters.get<double>("DENS")),
      yield_(matdata.parameters.get<double>("YIELD")),
      isohard_(matdata.parameters.get<double>("ISOHARD")),
      infyield_(matdata.parameters.get<double>("SATHARDENING")),
      hardexp_(matdata.parameters.get<double>("HARDEXPO")),
      visc_(matdata.parameters.get<double>("VISC")),
      rate_dependency_(matdata.parameters.get<double>("RATE_DEPENDENCY")),
      tolerance_nr_(matdata.parameters.get<double>("TOL")),
      functionID_hardening_(matdata.parameters.get<int>("HARDENING_FUNC")),
      max_iterations_(50)
{
  if (yield_ == 0 && functionID_hardening_ == 0)
    FOUR_C_THROW(
        "You have to provide either a parameter for "
        "HARDENING_FUNC or YIELD in MAT_Struct_PlasticNlnLogNeoHooke");
}


/*----------------------------------------------------------------------*
 | is called in Material::Factory from read_materials()                  |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::PlasticNlnLogNeoHooke::create_material()
{
  return std::make_shared<Mat::PlasticNlnLogNeoHooke>(this);
}


Mat::PlasticNlnLogNeoHookeType Mat::PlasticNlnLogNeoHookeType::instance_;


/*----------------------------------------------------------------------*
 | is called in Material::Factory from read_materials()                  |
 *----------------------------------------------------------------------*/
Core::Communication::ParObject* Mat::PlasticNlnLogNeoHookeType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::PlasticNlnLogNeoHooke* plasticneo = new Mat::PlasticNlnLogNeoHooke();
  plasticneo->unpack(buffer);
  return plasticneo;
}


/*----------------------------------------------------------------------*
 | constructor (public)                                                 |
 *----------------------------------------------------------------------*/
Mat::PlasticNlnLogNeoHooke::PlasticNlnLogNeoHooke() : params_(nullptr) {}


/*----------------------------------------------------------------------*
 | copy-constructor (public)                                            |
 *----------------------------------------------------------------------*/
Mat::PlasticNlnLogNeoHooke::PlasticNlnLogNeoHooke(Mat::PAR::PlasticNlnLogNeoHooke* params)
    : params_(params)
{
}


/*----------------------------------------------------------------------*
 | pack (public)                                                        |
 *----------------------------------------------------------------------*/
void Mat::PlasticNlnLogNeoHooke::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);

  // pack history data
  int histsize;
  // if material is not initialized, i.e. start simulation, nothing to pack
  if (!initialized())
  {
    histsize = 0;
  }
  else
  {
    // if material is initialized (restart): size equates number of gausspoints
    histsize = accplstrainlast_.size();
  }
  add_to_pack(data, histsize);  // Length of history vector(s)
  for (int var = 0; var < histsize; ++var)
  {
    // insert last converged states
    add_to_pack(data, accplstrainlast_.at(var));
    add_to_pack(data, invplrcglast_.at(var));

    // insert current iteration states
    add_to_pack(data, accplstraincurr_.at(var));
    add_to_pack(data, invplrcgcurr_.at(var));
  }
}  // pack()


/*----------------------------------------------------------------------*
 | unpack (public)                                                      |
 *----------------------------------------------------------------------*/
void Mat::PlasticNlnLogNeoHooke::unpack(Core::Communication::UnpackBuffer& buffer)
{
  isinit_ = true;

  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid
  int matid;
  extract_from_pack(buffer, matid);
  params_ = nullptr;
  if (Global::Problem::instance()->materials() != nullptr)
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
        params_ = static_cast<Mat::PAR::PlasticNlnLogNeoHooke*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());

      // Extract the function for hardening again for unpack.
      const int functionID_hardening =
          params_->functionID_hardening_;  // function number for isotropic hardening
      if (functionID_hardening != 0)
      {
        hardening_function_ =
            &Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfAnything>(
                functionID_hardening);
      }
    }

  // history data
  int histsize;
  extract_from_pack(buffer, histsize);

  // if system is not yet initialized, the history vectors have to be initialized
  if (histsize == 0) isinit_ = false;

  for (int var = 0; var < histsize; ++var)
  {
    Core::LinAlg::Matrix<3, 3> tmp_matrix(Core::LinAlg::Initialization::zero);
    double tmp_scalar = 0.0;

    // last converged states are unpacked
    extract_from_pack(buffer, tmp_scalar);
    accplstrainlast_.push_back(tmp_scalar);
    extract_from_pack(buffer, tmp_matrix);
    invplrcglast_.push_back(tmp_matrix);

    // current iteration states are unpacked
    extract_from_pack(buffer, tmp_scalar);
    accplstraincurr_.push_back(tmp_scalar);
    extract_from_pack(buffer, tmp_matrix);
    invplrcgcurr_.push_back(tmp_matrix);
  }
}  // unpack()


/*---------------------------------------------------------------------*
 | initialise / allocate internal variables (public)                   |
 *---------------------------------------------------------------------*/
void Mat::PlasticNlnLogNeoHooke::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  // Extract the function for hardening only once because this is expensive.
  const int functionID_hardening =
      params_->functionID_hardening_;  // function number for isotropic hardening
  if (functionID_hardening != 0)
  {
    hardening_function_ =
        &Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfAnything>(
            functionID_hardening);
  }

  invplrcglast_.resize(numgp);
  invplrcgcurr_.resize(numgp);

  accplstrainlast_.resize(numgp);
  accplstraincurr_.resize(numgp);

  Core::LinAlg::Matrix<3, 3> emptymat(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++) emptymat(i, i) = 1.0;

  for (int i = 0; i < numgp; i++)
  {
    invplrcglast_.at(i) = emptymat;
    invplrcgcurr_.at(i) = emptymat;

    accplstrainlast_.at(i) = 0.0;
    accplstraincurr_.at(i) = 0.0;
  }

  isinit_ = true;
}  // setup()


/*----------------------------------------------------------------------*
 | update internal variables                                            |
 *----------------------------------------------------------------------*/
void Mat::PlasticNlnLogNeoHooke::update()
{
  // make current values at time step t_n+1 to values of last step t_n
  invplrcglast_ = invplrcgcurr_;
  accplstrainlast_ = accplstraincurr_;

  // empty vectors of current data
  invplrcgcurr_.clear();

  // get the size of the vector
  // (use the last vector, because it includes latest results, current is empty)
  const int histsize = invplrcglast_.size();
  invplrcgcurr_.resize(histsize);
  accplstraincurr_.resize(histsize);

  Core::LinAlg::Matrix<3, 3> emptymat(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++) emptymat(i, i) = 1.0;

  for (int i = 0; i < histsize; i++)
  {
    invplrcgcurr_.at(i) = emptymat;
    accplstraincurr_.at(i) = 0.0;
  }
}  // update()


/*----------------------------------------------------------------------*
 | calculate stress and constitutive tensor                             |
 *----------------------------------------------------------------------*/
void Mat::PlasticNlnLogNeoHooke::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  const Core::LinAlg::Matrix<3, 3> defgrd_mat = Core::LinAlg::make_matrix_view(*defgrad);

  // elastic material data
  // get material parameters
  const double ym = params_->youngs_;                  // Young's modulus
  const double nu = params_->poissonratio_;            // Poisson's ratio
  const double G = ym / (2.0 * (1.0 + nu));            // shear modulus, mu=G
  const double kappa = ym / (3.0 * (1.0 - 2.0 * nu));  // bulk modulus

  // plastic material data
  const double yield = params_->yield_;          // initial yield stress
  const double isohard = params_->isohard_;      // linear isotropic hardening
  const double infyield = params_->infyield_;    // saturation yield stress
  const double hardexp = params_->hardexp_;      // nonlinear hardening exponent
  const double visc = params_->visc_;            // viscosity
  const double eps = params_->rate_dependency_;  // rate dependency

  const double detF = defgrd_mat.determinant();

  FOUR_C_ASSERT(context.time_step_size, "Time step size not given in evaluation context.");
  const double dt = *context.time_step_size;
  // check, if errors are tolerated or should throw a FOUR_C_THROW
  bool error_tol = false;
  if (params.isParameter("tolerate_errors")) error_tol = params.get<bool>("tolerate_errors");

  // plastic increment
  double Dgamma = 0.0;
  double dy_d_dgamma = 0.0;

  Core::LinAlg::Matrix<3, 3> invdefgrd(defgrd_mat);
  invdefgrd.invert();

  // matrices for temporary stuff
  Core::LinAlg::Matrix<3, 3> tmp1;
  Core::LinAlg::Matrix<3, 3> tmp2;

  // 3x3 2nd-order identity matrix
  Core::LinAlg::Matrix<3, 3> id2(Core::LinAlg::Initialization::zero);
  // 3x3 2nd-order deviatoric identity matrix in principal directions
  Core::LinAlg::Matrix<3, 3> Idev;
  for (int i = 0; i < 3; i++)
  {
    id2(i, i) = 1.0;
    for (int j = 0; j < 3; j++)
    {
      if (i == j)
        Idev(i, j) = 2.0 / 3.0;
      else
        Idev(i, j) = -1.0 / 3.0;
    }
  }

  // linear elasticity tensor in principal directions
  Core::LinAlg::Matrix<3, 3> D_ep_principal(Idev);
  D_ep_principal.scale(2.0 * G);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) D_ep_principal(i, j) += kappa;

  // ------------------------------------------------------- trial strain
  // elastic left Cauchy-Green deformation tensor (LCG) at trial state
  // Be_trial = b_{e,n+1}^{trial} = F_{n+1} C_{p,n}^{-1} F_{n+1}
  Core::LinAlg::Matrix<3, 3> Be_trial;
  tmp1.multiply(defgrd_mat, invplrcglast_.at(gp));
  Be_trial.multiply_nt(tmp1, defgrd_mat);

  // elastic LCG at final state
  Core::LinAlg::Matrix<3, 3> Be;

  // ***************************************************
  // Here we start the principal stress based algorithm
  // ***************************************************

  // convert and solve eigenvalue problem
  // this matrix contains spatial eigen directions (the second
  // index corresponds to the eigenvalue)
  Core::LinAlg::SerialDenseMatrix n(3, 3);
  Core::LinAlg::SerialDenseVector lambda_trial_square(3);

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) n(i, j) = Be_trial(i, j);

  // calculate eigenvectors and eigenvalues
  Core::LinAlg::symmetric_eigen_problem(n, lambda_trial_square);
  // eigenvectors are stored in n, i.e. original matrix inserted in method is
  // destroyed.
  // eigenvalues correspond to the square of the stretches

  // spatial principal direction n_alpha (in Core::LINALG format)
  std::vector<Core::LinAlg::Matrix<3, 1>> spatial_principal_directions;
  spatial_principal_directions.resize(3);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) (spatial_principal_directions.at(i))(j) = n(j, i);

  // material principal directions N_alpha
  // note, that for convenience in the later programming this is NOT
  // based on the correct pull-back N_alpha = lambda_alpha * F^-1 * n_alpha
  // Instead we just do N_alpha = F^{-1} * n_alpha
  std::vector<Core::LinAlg::Matrix<3, 1>> material_principal_directions;
  material_principal_directions.resize(3);
  for (int i = 0; i < 3; i++)
    material_principal_directions.at(i).multiply(invdefgrd, spatial_principal_directions.at(i));

  // deviatoric Kirchhoff stress at trial state
  // tau^{trial} = 2 * G * log(sqrt(lambda^2)) - 2/3 * G * log(detF)
  Core::LinAlg::Matrix<3, 1> dev_KH_trial;
  for (int i = 0; i < 3; i++)
    dev_KH_trial(i) = G * std::log(lambda_trial_square(i)) - 2.0 / 3.0 * G * std::log(detF);

  // deviatoric Kirchhoff stress at final state
  // For now we store the trial states. In case of plastic
  // material reaction, we will update it later
  Core::LinAlg::Matrix<3, 1> dev_KH;

  // pressure (equal at trial state and final state) for the use with tau
  // tau = tau'_aa + p = tau'_aa + kappa * log(detF)
  // later used for S = F^{-1} . tau . F^{-T}, no scaling with detF required
  double pressure = kappa * std::log(detF);

  // ----------------------------------------------------- yield function
  // calculate von Mises equivalent stress at trial state
  double abs_dev_KH_trial = 0.0;
  for (int i = 0; i < 3; i++) abs_dev_KH_trial += dev_KH_trial(i) * dev_KH_trial(i);
  abs_dev_KH_trial = std::sqrt(abs_dev_KH_trial);

  //! vector for input of differential pressure to function
  std::vector<std::pair<std::string, double>> dp;
  dp.emplace_back("epsp", accplstrainlast_.at(gp));
  const double y_d = hardening_function_
                         ? hardening_function_->evaluate(dp, {}, 0)
                         : yield + isohard * accplstrainlast_.at(gp) +
                               (infyield - yield) * (1. - exp(-hardexp * accplstrainlast_.at(gp)));


  const double y_d_visc = y_d * pow(visc * Dgamma / dt + 1., eps);

  const double f_trial = std::sqrt(3.0 / 2.0) * abs_dev_KH_trial - y_d_visc;

  // switch elastic (f <= 0) and plastic (f > 0) state
  if (f_trial <= 0.0)  // ----------------------------------- elastic step
  {
    // trial state variables are final variables
    dev_KH.update(dev_KH_trial);
    Be.update(Be_trial);

    // plastic increment is zero
    Dgamma = 0.0;
  }

  else  // -------------------------------------------------- plastic step
  {
    std::pair<double, double> gamma_and_derivative =
        (hardening_function_) ? solve_newton_with_hardening_func(params_, abs_dev_KH_trial,
                                    accplstrainlast_.at(gp), dt, *hardening_function_, error_tol)
                              : solve_newton_with_parameters(params_, abs_dev_KH_trial,
                                    accplstrainlast_.at(gp), dt, error_tol);

    Dgamma = gamma_and_derivative.first;

    // flow vector (7.77) in de Souza Neta
    Core::LinAlg::Matrix<3, 1> flow_vector(dev_KH_trial);
    flow_vector.scale(1.0 / (sqrt(2.0 / 3.0) * abs_dev_KH_trial));

    // stress return mapping (7.89) in de Souza Neto
    double fac_dev_KH = 1.0 - 2.0 * G * Dgamma / (std::sqrt(2.0 / 3.0) * abs_dev_KH_trial);
    dev_KH.update(fac_dev_KH, dev_KH_trial, 0.0);

    // strain return mapping
    // b_{e,n+1} = sum_i^3 ( lambda_{n+1}^2 . n \otimes n )
    // lambda_{n+1} = lambda_{n+1}^{trial} / exp(Dgamma . flow_vector)
    Be.clear();
    for (int i = 0; i < 3; i++)
    {
      tmp1.multiply_nt(spatial_principal_directions.at(i), spatial_principal_directions.at(i));
      Be.update((lambda_trial_square(i) / exp(2.0 * flow_vector(i) * Dgamma)), tmp1, 1.0);
    }

    // update tangent modulus
    dy_d_dgamma = gamma_and_derivative.second;
    double fac_D_ep_1 = -4.0 * G * G * Dgamma / (sqrt(2.0 / 3.0) * abs_dev_KH_trial);
    D_ep_principal.update(fac_D_ep_1, Idev, 1.0);
    tmp1.multiply_nt(flow_vector, flow_vector);
    double fac_D_ep_2 =
        4.0 * G * G * (sqrt(2.0 / 3.0) * Dgamma / abs_dev_KH_trial - 1.0 / (3.0 * G + dy_d_dgamma));
    D_ep_principal.update(fac_D_ep_2, tmp1, 1.0);
  }

  // -------------------------------------------------- output PK2 stress
  // tau_ij = tau_ii * n_i \otimes n_i
  // or tau_ij = tau'_ii + kappa ln(J)
  // S_ij   = F^-1 tau_ij F^-T
  //        = tau_ij (F^-1 n_i) \otimes (F^-1 n_i)
  Core::LinAlg::Matrix<3, 3> PK2(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++)
  {
    tmp1.multiply_nt(material_principal_directions.at(i), material_principal_directions.at(i));
    PK2.update((dev_KH(i) + pressure), tmp1, 1.0);
  }

  Core::LinAlg::Matrix<6, 1> stress_view = Core::LinAlg::make_stress_like_voigt_view(stress);
  Core::LinAlg::Matrix<6, 6> cmat_view = Core::LinAlg::make_stress_like_voigt_view(cmat);

  // output stress in Voigt notation
  stress = Core::LinAlg::assume_symmetry(Core::LinAlg::make_tensor_view(PK2));

  // ---------------------------------------------------- tangent modulus
  // express coefficients of tangent in Kirchhoff stresses
  cmat_view.clear();
  for (int a = 0; a < 3; a++)
  {
    // - sum_1^3 (2 * tau N_aaaa)
    tmp1.multiply_nt(
        material_principal_directions.at(a), material_principal_directions.at(a));  // N_{aa}
    Core::LinAlg::FourTensorOperations::add_elasticity_tensor_product(
        cmat_view, -2.0 * (dev_KH(a) + pressure), tmp1, tmp1, 1.0);

    for (int b = 0; b < 3; b++)
    {
      // c_ab N_aabb
      // result of return mapping of deviatoric component c_ab
      tmp1.multiply_nt(
          material_principal_directions.at(a), material_principal_directions.at(a));  // N_{aa}
      tmp2.multiply_nt(
          material_principal_directions.at(b), material_principal_directions.at(b));  // N_{bb}
      Core::LinAlg::FourTensorOperations::add_elasticity_tensor_product(
          cmat_view, D_ep_principal(a, b), tmp1, tmp2, 1.0);

      if (a != b)
      {
        const double fac =
            (lambda_trial_square(a) != lambda_trial_square(b))
                ?  // (tau_aa * lambda_b^2 - tau_bb * lambda_a^2) / (lambda_a^2 - lambda_b^2)
                ((dev_KH(a) + pressure) * lambda_trial_square(b) -
                    (dev_KH(b) + pressure) * lambda_trial_square(a)) /
                    (lambda_trial_square(a) - lambda_trial_square(b))
                :  // 1/2 [(d^2 Psi)/(d ln lambda_b * d ln lambda_b) -
                   //                (d^2 Psi)/(d ln lambda_a * d ln lambda_b)]
                   // - tau_bb, cf. (6.91)
                0.5 * (D_ep_principal(b, b) - D_ep_principal(a, b)) - (dev_KH(b) + pressure);
        tmp1.multiply_nt(
            material_principal_directions.at(a), material_principal_directions.at(b));  // N_{ab}
        tmp2.multiply_nt(
            material_principal_directions.at(b), material_principal_directions.at(a));  // N_{ba}
        Core::LinAlg::FourTensorOperations::add_elasticity_tensor_product(
            cmat_view, fac, tmp1, tmp1, 1.0);  // N_{abab}
        Core::LinAlg::FourTensorOperations::add_elasticity_tensor_product(
            cmat_view, fac, tmp1, tmp2, 1.0);  // N_{abba}
      }  // end if (a!=b)
    }  // end loop b
  }  // end loop a

  // --------------------------------------------- update plastic history
  // plastic inverse of right Cauchy-Green deformation tensor (RCG)
  tmp1.multiply(invdefgrd, Be);
  invplrcgcurr_.at(gp).multiply_nt(tmp1, invdefgrd);

  // accumulated plastic strain
  accplstraincurr_.at(gp) = accplstrainlast_.at(gp) + Dgamma;

  // Green-Lagrange plastic strains can be easily calculated, in contrast
  // Euler-Almansi requires special treatment, which is not yet considered in the
  // element formulation
}  // evaluate()


/*---------------------------------------------------------------------*
 | return names of visualization data (public)                         |
 *---------------------------------------------------------------------*/
void Mat::PlasticNlnLogNeoHooke::vis_names(std::map<std::string, int>& names) const
{
  std::string accumulatedstrain = "accumulatedstrain";
  names[accumulatedstrain] = 1;  // scalar

}  // vis_names()


/*---------------------------------------------------------------------*
 | return names of visualization data for direct VTK output            |
 *---------------------------------------------------------------------*/
void Mat::PlasticNlnLogNeoHooke::register_output_data_names(
    std::unordered_map<std::string, int>& names_and_size) const
{
  names_and_size["accumulated_plastic_strain"] = 1;
  names_and_size["active_plasticity"] = 1;
  names_and_size["plastic_strain"] = 6;
}


/*---------------------------------------------------------------------*
 | return visualization data (public)                                  |
 *---------------------------------------------------------------------*/
bool Mat::PlasticNlnLogNeoHooke::vis_data(
    const std::string& name, std::vector<double>& data, int numgp, int eleID) const
{
  if (name == "accumulatedstrain")
  {
    if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");
    double temp = 0.0;
    for (int gp = 0; gp < numgp; ++gp) temp += accumulated_strain(gp);
    data[0] = temp / numgp;
  }
  return false;

}  // vis_data()


bool Mat::PlasticNlnLogNeoHooke::evaluate_output_data(
    const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const
{
  if (name == "accumulated_plastic_strain")
  {
    for (std::size_t gp = 0; gp < accplstraincurr_.size(); ++gp)
    {
      data(gp, 0) = accplstraincurr_.at(int(gp));
    }
    return true;
  }
  if (name == "plastic_strain")
  {
    for (std::size_t gp = 0; gp < invplrcgcurr_.size(); ++gp)
    {
      const double* values = invplrcgcurr_.at(gp).data();
      for (std::size_t i = 0; i < 6; ++i)
      {
        data(gp, i) = values[i];
      }
    }
    return true;
  }
  if (name == "active_plasticity")
  {
    for (std::size_t gp = 0; gp < invplrcgcurr_.size(); ++gp)
    {
      data(gp, 0) = (accplstraincurr_.at(int(gp)) > accplstrainlast_.at(int(gp))) ? 1 : 0;
    }
    return true;
  }
  return false;
}

/*----------------------------------------------------------------------*/

FOUR_C_NAMESPACE_CLOSE
