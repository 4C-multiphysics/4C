// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_plasticdruckerprager.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_parobject.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_linalg_FADmatrix_utils.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_stvenantkirchhoff.hpp"
#include "4C_material_base.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_local_newton.hpp"

FOUR_C_NAMESPACE_OPEN


Mat::PAR::PlasticDruckerPrager::PlasticDruckerPrager(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      youngs_(matdata.parameters.get<double>("YOUNG")),
      poissonratio_(matdata.parameters.get<double>("NUE")),
      density_(matdata.parameters.get<double>("DENS")),
      isohard_(matdata.parameters.get<double>("ISOHARD")),
      abstol_(matdata.parameters.get<double>("TOL")),
      cohesion_(matdata.parameters.get<double>("C")),
      eta_(matdata.parameters.get<double>("ETA")),
      xi_(matdata.parameters.get<double>("XI")),
      etabar_(matdata.parameters.get<double>("ETABAR")),
      tang_(matdata.parameters.get<std::string>("TANG")),
      itermax_(matdata.parameters.get<int>("MAXITER"))
{
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::PlasticDruckerPrager::create_material()
{
  return std::make_shared<Mat::PlasticDruckerPrager>(this);
}
Mat::PlasticDruckerPragerType Mat::PlasticDruckerPragerType::instance_;

Core::Communication::ParObject* Mat::PlasticDruckerPragerType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::PlasticDruckerPrager* plastic = new Mat::PlasticDruckerPrager();
  plastic->unpack(buffer);
  return plastic;
}

Mat::PlasticDruckerPrager::PlasticDruckerPrager() : params_(nullptr) {}

Mat::PlasticDruckerPrager::PlasticDruckerPrager(Mat::PAR::PlasticDruckerPrager* params)
    : params_(params)
{
}

void Mat::PlasticDruckerPrager::pack(Core::Communication::PackBuffer& data) const
{
  int type = unique_par_object_id();
  add_to_pack(data, type);
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();
  add_to_pack(data, matid);
  int histsize = initialized() ? strainpllast_.size() : 0;
  add_to_pack(data, histsize);
  for (int var = 0; var < histsize; ++var)
  {
    // insert last converged states
    add_to_pack(data, strainpllast_.at(var));
    add_to_pack(data, strainbarpllast_.at(var));

    // insert current iteration states
    add_to_pack(data, strainplcurr_.at(var));
    add_to_pack(data, strainbarplcurr_.at(var));
  }
}

void Mat::PlasticDruckerPrager::unpack(Core::Communication::UnpackBuffer& buffer)
{
  isinit_ = true;


  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

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
        params_ = static_cast<Mat::PAR::PlasticDruckerPrager*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }

    int histsize;
    extract_from_pack(buffer, histsize);

    if (histsize == 0) isinit_ = false;

    strainpllast_ = std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>();
    strainplcurr_ = std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>();
    strainbarpllast_ = std::vector<double>();
    strainbarplcurr_ = std::vector<double>();
    for (int var = 0; var < histsize; ++var)
    {
      Core::LinAlg::Matrix<NUM_STRESS_3D, 1> tmp_vector(Core::LinAlg::Initialization::zero);
      double tmp_scalar = 0.0;

      // last converged states are unpacked
      extract_from_pack(buffer, tmp_vector);
      strainpllast_.push_back(tmp_vector);
      extract_from_pack(buffer, tmp_scalar);
      strainbarpllast_.push_back(tmp_scalar);

      // current iteration states are unpacked
      extract_from_pack(buffer, tmp_vector);
      strainplcurr_.push_back(tmp_vector);
      extract_from_pack(buffer, tmp_scalar);
      strainbarplcurr_.push_back(tmp_scalar);
    }
  }
}
void Mat::PlasticDruckerPrager::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  strainpllast_.resize(numgp);
  strainplcurr_.resize(numgp);

  strainbarpllast_.resize(numgp);
  strainbarplcurr_.resize(numgp);

  isinit_ = true;
}

void Mat::PlasticDruckerPrager::update()
{
  strainpllast_ = strainplcurr_;
  strainbarpllast_ = strainbarplcurr_;

  std::for_each(strainplcurr_.begin(), strainplcurr_.end(), [](auto& item) { item.clear(); });
  std::fill(strainbarplcurr_.begin(), strainbarplcurr_.end(), 0.0);
}

void Mat::PlasticDruckerPrager::setup_cmat(
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat) const
{
  double young = params_->youngs_;

  double nu = params_->poissonratio_;

  cmat = Mat::StVenantKirchhoff::evaluate_stress_linearization(young, nu);
}

void Mat::PlasticDruckerPrager::setup_cmat_elasto_plastic_cone(
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, double Dgamma, double G, double Kappa,
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& devstrain, double xi, double Hiso, double eta,
    double etabar) const
{
  Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmat_view =
      Core::LinAlg::make_stress_like_voigt_view(cmat);
  cmat_view.clear();

  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> id2(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::identity_matrix(id2);
  Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> id4sharp(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++) id4sharp(i, i) = 1.0;
  for (int i = 3; i < 6; i++) id4sharp(i, i) = 0.5;

  const double normdevstrain =
      sqrt(devstrain(0) * devstrain(0) + devstrain(1) * devstrain(1) + devstrain(2) * devstrain(2) +
           2 * (devstrain(3) * devstrain(3) + devstrain(4) * devstrain(4) +
                   devstrain(5) * devstrain(5)));
  const double epfac = 2 * G * (1 - (Dgamma / sqrt(2) / normdevstrain));

  cmat_view.update(epfac, id4sharp, 1.0);

  double epfac1 = 0.0;
  double epfac2 = 0.0;
  double epfac3 = 0.0;
  double epfac4 = 0.0;
  epfac1 = epfac / (-3.0);
  cmat_view.multiply_nt(epfac1, id2, id2, 1.0);

  double A = 0.0;
  A = 1 / (G + Kappa * etabar * eta + xi * xi * Hiso);
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> D(Core::LinAlg::Initialization::zero);
  D.update(1 / normdevstrain, devstrain);
  epfac2 = 2 * G * (Dgamma / (sqrt(2) * normdevstrain) - G * A);

  for (int k = 0; k < 6; ++k)
  {
    for (int i = 0; i < 6; ++i)
    {
      cmat_view(i, k) += epfac2 * D(i) * D(k);
    }
  }

  epfac3 = -sqrt(2) * G * A * Kappa;

  for (int k = 0; k < 6; ++k)
  {
    for (int i = 0; i < 6; ++i)
    {
      cmat_view(k, i) += epfac3 * (eta * D(k) * id2(i) + etabar * id2(k) * D(i));
    }
  }

  epfac4 = Kappa * (1 - Kappa * eta * etabar * A);

  for (int k = 0; k < 6; ++k)
  {
    for (int i = 0; i < 6; ++i)
    {
      cmat_view(i, k) += epfac4 * id2(k) * id2(i);
    }
  }
}

void Mat::PlasticDruckerPrager::setup_cmat_elasto_plastic_apex(
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, double Kappa,
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& devstrain, double xi, double Hiso, double eta,
    double etabar) const
{
  Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmat_view =
      Core::LinAlg::make_stress_like_voigt_view(cmat);

  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> id2(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++) id2(i) = 1.0;
  double epfac = 0.0;
  epfac = Kappa * (1 - Kappa / (Kappa + xi / eta * xi / etabar * Hiso));
  cmat_view.clear();
  cmat_view.multiply_nt(epfac, id2, id2, 0.0);
}

template <typename ScalarT>
void Mat::PlasticDruckerPrager::evaluate_fad(const Core::LinAlg::Matrix<3, 3>* defgrd,
    const Core::LinAlg::Matrix<6, 1, ScalarT>& linstrain, const Teuchos::ParameterList& params,
    Core::LinAlg::Matrix<6, 1, ScalarT>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, const int gp, const int eleGID)
{
  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> plstrain(Core::LinAlg::Initialization::zero);

  ScalarT young = params_->youngs_;
  ScalarT nu = params_->poissonratio_;
  ScalarT Hiso = params_->isohard_;
  ScalarT cohesion = params_->cohesion_;
  ScalarT eta = params_->eta_;
  ScalarT xi = params_->xi_;
  ScalarT etabar = params_->etabar_;
  const int itermax = params_->itermax_;
  const std::string tang_str = params_->tang_;
  ScalarT G = 0.0;
  G = young / (2.0 * (1.0 + nu));
  ScalarT kappa = 0.0;
  kappa = young / (3.0 * (1.0 - 2.0 * nu));
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> id2(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++) id2(i) = 1.0;
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1, ScalarT> strain(linstrain);
  const int tang = std::invoke(
      [tang_str]() -> int
      {
        if (tang_str == "consistent")
          return 1;
        else if (tang_str == "elastic")
          return 0;
        else
          return 1;
      });

  Core::LinAlg::Matrix<NUM_STRESS_3D, 1, ScalarT> strain_p(
      Core::LinAlg::Initialization::uninitialized);
  for (int i = 0; i < 6; i++) strain_p(i, 0) = strainpllast_.at(gp)(i, 0);
  ScalarT strainbar_p = 0.0;
  strainbar_p = (strainbarpllast_.at(gp));
  if (strainbarpllast_.at(gp) < 0.0)
    FOUR_C_THROW("accumulated plastic strain has to be equal to or greater than zero!");

  for (int i = 3; i < 6; ++i) strain(i) /= 2.0;
  for (int i = 3; i < 6; ++i) strain_p(i) /= 2.0;

  Core::LinAlg::Matrix<NUM_STRESS_3D, 1, ScalarT> strain_e(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1, ScalarT> trialstrain_e(
      Core::LinAlg::Initialization::uninitialized);
  trialstrain_e.update(1.0, strain, (-1.0), strain_p);
  ScalarT tracestrain = trialstrain_e(0) + trialstrain_e(1) + trialstrain_e(2);
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1, ScalarT> volumetricstrain(
      Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1, ScalarT> id2Scalar(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < NUM_STRESS_3D; ++i) id2Scalar(i) = static_cast<ScalarT>(id2(i));
  volumetricstrain.update((tracestrain / 3.0), id2Scalar);
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1, ScalarT> devstrain(
      Core::LinAlg::Initialization::uninitialized);
  devstrain.update(1.0, trialstrain_e, (-1.0), volumetricstrain);

  ScalarT p = kappa * tracestrain;
  ScalarT p_trial = p;
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1, ScalarT> devstress(
      Core::LinAlg::Initialization::uninitialized);
  devstress.update((2.0 * G), devstrain);

  ScalarT J2 = 1.0 / 2.0 *
                   (devstress(0) * devstress(0) + devstress(1) * devstress(1) +
                       devstress(2) * devstress(2)) +
               devstress(3) * devstress(3) + devstress(4) * devstress(4) +
               devstress(5) * devstress(5);
  ScalarT Phi_trial = 0.0;
  Phi_trial = std::sqrt(J2) + eta * p - xi * cohesion - xi * Hiso * strainbar_p;
  ScalarT Dgamma = 0.0;
  ScalarT dstrainv = 0.0;
  if (Phi_trial / abs(cohesion) > params_->abstol_)
  {
    auto returnToConeFunctAndDeriv = [this, &G, &kappa, &Phi_trial](ScalarT Dgamma_init)
    { return this->return_to_cone_funct_and_deriv(Dgamma_init, G, kappa, Phi_trial); };

    const double tol = params_->abstol_;
    Dgamma =
        Core::Utils::solve_local_newton(returnToConeFunctAndDeriv, Dgamma, tol * cohesion, itermax);
    strainbar_p = (strainbarpllast_.at(gp)) + xi * Dgamma;
    devstress.scale(1.0 - (G * Dgamma / std::sqrt(J2)));
    p = p_trial - kappa * etabar * Dgamma;
    if ((std::sqrt(J2) - G * Dgamma) / abs(cohesion) < params_->abstol_)
    {
      strainbar_p = (strainbarpllast_.at(gp));
      auto returnToApexFunctAndDeriv = [this, &p_trial, &kappa, &strainbar_p](ScalarT dstrainv_init)
      { return this->return_to_apex_funct_and_deriv(dstrainv_init, p_trial, kappa, strainbar_p); };

      const double tol = params_->abstol_;
      dstrainv = Core::Utils::solve_local_newton(
          returnToApexFunctAndDeriv, dstrainv, tol * cohesion, itermax);
      strainbar_p = (strainbarpllast_.at(gp)) + xi / eta * dstrainv;
      p = p_trial - kappa * dstrainv;
      for (int i = 0; i < 6; i++) devstress(i) = 0.0;
    }
    PlasticDruckerPrager::stress(p, devstress, stress);
    strain_e.update(1 / G / 2, devstress, p / kappa / 3, id2Scalar);
    for (int i = 3; i < 6; ++i) strain_e(i) *= 2.0;
    for (int i = 3; i < 6; ++i) strain(i) *= 2.0;
    strain_p.update(1.0, strain, -1.0, strain_e);

    strainplcurr_.at(gp) = Core::FADUtils::cast_to_double(strain_p);
    strainbarplcurr_.at(gp) = Core::FADUtils::cast_to_double(strainbar_p);
  }
  else
  {
    PlasticDruckerPrager::stress(p, devstress, stress);
    strain_e.update(trialstrain_e);
    for (int i = 3; i < 6; ++i) strain_e(i) *= 2.0;
    for (int i = 3; i < 6; ++i) strain(i) *= 2.0;

    strainplcurr_.at(gp) = strainpllast_.at(gp);
    strainbarplcurr_.at(gp) = strainbarpllast_.at(gp);
  }
  if ((Phi_trial > 0) && (tang == 1))
  {
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> devstraindouble =
        Core::FADUtils::cast_to_double(devstrain);
    if (dstrainv != 0.0)
    {
      setup_cmat_elasto_plastic_apex(cmat, Core::FADUtils::cast_to_double(kappa), devstraindouble,
          Core::FADUtils::cast_to_double(xi), Core::FADUtils::cast_to_double(Hiso),
          Core::FADUtils::cast_to_double(eta), Core::FADUtils::cast_to_double(etabar));
    }
    else
    {
      setup_cmat_elasto_plastic_cone(cmat, Core::FADUtils::cast_to_double(Dgamma),
          Core::FADUtils::cast_to_double(G), Core::FADUtils::cast_to_double(kappa), devstraindouble,
          Core::FADUtils::cast_to_double(xi), Core::FADUtils::cast_to_double(Hiso),
          Core::FADUtils::cast_to_double(eta), Core::FADUtils::cast_to_double(etabar));
    }
  }
  else
  {
    setup_cmat(cmat);
  }
}

void Mat::PlasticDruckerPrager::register_output_data_names(
    std::unordered_map<std::string, int>& names_and_size) const
{
  names_and_size["accumulated_plastic_strain"] = 1;
  names_and_size["plastic_strain"] = 6;
}

bool Mat::PlasticDruckerPrager::evaluate_output_data(
    const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const
{
  if (name == "accumulated_plastic_strain")
  {
    for (std::size_t gp = 0; gp < strainbarplcurr_.size(); ++gp)
    {
      data(gp, 0) = strainbarplcurr_.at(int(gp));
    }
    return true;
  }
  if (name == "plastic_strain")
  {
    for (std::size_t gp = 0; gp < strainplcurr_.size(); ++gp)
    {
      const double* values = strainplcurr_.at(gp).data();
      for (std::size_t i = 0; i < 6; ++i)
      {
        data(gp, i) = values[i];
      }
    }
    return true;
  }
  return false;
}

template <typename T>
void Mat::PlasticDruckerPrager::stress(const T p,
    const Core::LinAlg::Matrix<NUM_STRESS_3D, 1, T>& devstress,
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1, T>& stress) const
{
  stress.update(devstress);
  for (int i = 0; i < 3; ++i) stress(i) += p;
}

template <typename T>
std::pair<T, T> Mat::PlasticDruckerPrager::return_to_cone_funct_and_deriv(
    T Dgamma, T G, T kappa, T Phi_trial)
{
  double Hiso = params_->isohard_;
  double eta = params_->eta_;
  double xi = params_->xi_;
  double etabar = params_->etabar_;
  T Res = Phi_trial - Dgamma * (G + eta * kappa * etabar) - (xi * xi * Dgamma * Hiso);
  T d = -G - (kappa * etabar * eta) - (xi * xi * Hiso);
  return {Res, d};
}

template <typename T>
std::pair<T, T> Mat::PlasticDruckerPrager::return_to_apex_funct_and_deriv(
    T dstrainv, T p, T kappa, T strainbar_p)
{
  double Hiso = params_->isohard_;
  double eta = params_->eta_;
  double xi = params_->xi_;
  double cohesion = params_->cohesion_;
  double etabar = params_->etabar_;
  double alpha = xi / eta;
  double beta = xi / etabar;
  T Res =
      beta * cohesion + beta * strainbar_p * Hiso - p + dstrainv * (alpha * beta * Hiso + kappa);
  T d = xi * xi / eta / etabar * Hiso + kappa;
  return {Res, d};
}

template void Mat::PlasticDruckerPrager::evaluate_fad<double>(const Core::LinAlg::Matrix<3, 3>*,
    const Core::LinAlg::Matrix<6, 1, double>&, const Teuchos::ParameterList&,
    Core::LinAlg::Matrix<6, 1, double>&, Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>&, int gp,
    int eleGID);
template void Mat::PlasticDruckerPrager::evaluate_fad<FAD>(const Core::LinAlg::Matrix<3, 3>*,
    const Core::LinAlg::Matrix<6, 1, FAD>&, const Teuchos::ParameterList&,
    Core::LinAlg::Matrix<6, 1, FAD>&, Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>&, int gp,
    int eleGID);
template void Mat::PlasticDruckerPrager::stress<double>(const double p,
    const Core::LinAlg::Matrix<NUM_STRESS_3D, 1, double>& devstress,
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1, double>& stress) const;
template void Mat::PlasticDruckerPrager::stress<FAD>(const FAD p,
    const Core::LinAlg::Matrix<NUM_STRESS_3D, 1, FAD>& devstress,
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1, FAD>& stress) const;
template std::pair<double, double>
Mat::PlasticDruckerPrager::return_to_cone_funct_and_deriv<double>(
    double Dgamma, double G, double kappa, double Phi_trial);
template std::pair<FAD, FAD> Mat::PlasticDruckerPrager::return_to_cone_funct_and_deriv<FAD>(
    FAD Dgamma, FAD G, FAD kappa, FAD Phi_trial);
template std::pair<double, double>
Mat::PlasticDruckerPrager::return_to_apex_funct_and_deriv<double>(
    double dstrainv, double p, double kappa, double strainbar_p);
template std::pair<FAD, FAD> Mat::PlasticDruckerPrager::return_to_apex_funct_and_deriv<FAD>(
    FAD dstrainv, FAD p, FAD kappa, FAD strainbar_p);

FOUR_C_NAMESPACE_CLOSE
