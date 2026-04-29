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
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_tensor_generators.hpp"
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
  FOUR_C_ASSERT_ALWAYS(cohesion_ > 1.0e-14, "Cohesion must be positive, but got {}.", cohesion_);
  FOUR_C_ASSERT_ALWAYS(poissonratio_ < 0.5 - 1.0e-14,
      "Poisson's ratio must be strictly less than 0.5, but got {}.", poissonratio_);
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

    strainpllast_ = std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>>();
    strainplcurr_ = std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>>();
    strainbarpllast_ = std::vector<double>();
    strainbarplcurr_ = std::vector<double>();
    for (int var = 0; var < histsize; ++var)
    {
      Core::LinAlg::SymmetricTensor<double, 3, 3> tmp_tensor{};
      double tmp_scalar = 0.0;

      // last converged states are unpacked
      extract_from_pack(buffer, tmp_tensor);
      strainpllast_.push_back(tmp_tensor);
      extract_from_pack(buffer, tmp_scalar);
      strainbarpllast_.push_back(tmp_scalar);

      // current iteration states are unpacked
      extract_from_pack(buffer, tmp_tensor);
      strainplcurr_.push_back(tmp_tensor);
      extract_from_pack(buffer, tmp_scalar);
      strainbarplcurr_.push_back(tmp_scalar);
    }
  }
}
void Mat::PlasticDruckerPrager::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  strainpllast_ = std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>>(numgp);
  strainplcurr_ = std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>>(numgp);

  strainbarpllast_.assign(numgp, 0.0);
  strainbarplcurr_.assign(numgp, 0.0);

  for (int i = 0; i < numgp; ++i)
  {
    strainpllast_.at(i).fill(0.0);
    strainplcurr_.at(i).fill(0.0);
  }

  isinit_ = true;
}

void Mat::PlasticDruckerPrager::update()
{
  strainpllast_ = strainplcurr_;
  strainbarpllast_ = strainbarplcurr_;

  std::for_each(strainplcurr_.begin(), strainplcurr_.end(), [](auto& item) { item.fill(0.0); });
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
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& devstrain, double xi, double Hiso,
    double eta, double etabar) const
{
  constexpr auto id2 = Core::LinAlg::TensorGenerators::identity<double, 3, 3>;
  constexpr auto Is = Core::LinAlg::TensorGenerators::symmetric_identity<double, 3, 3, 3, 3>;
  constexpr auto Id = Is - 1.0 / 3.0 * dyadic(id2, id2);

  const double nd = std::sqrt(ddot(devstrain, devstrain));
  const auto D = devstrain / nd;

  const double A = 1.0 / (G + Kappa * etabar * eta + xi * xi * Hiso);
  const double epfac = 2.0 * G * (1.0 - Dgamma / (std::sqrt(2.0) * nd));
  const double epfac2 = 2.0 * G * (Dgamma / (std::sqrt(2.0) * nd) - G * A);
  const double epfac3 = -std::sqrt(2.0) * G * A * Kappa;
  const double epfac4 = Kappa * (1.0 - Kappa * eta * etabar * A);

  cmat = epfac * Id + epfac2 * dyadic(D, D) +
         epfac3 * (eta * dyadic(D, id2) + etabar * dyadic(id2, D)) + epfac4 * dyadic(id2, id2);
}

void Mat::PlasticDruckerPrager::setup_cmat_elasto_plastic_apex(
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, double Kappa, double xi, double Hiso,
    double eta, double etabar) const
{
  constexpr auto id2 = Core::LinAlg::TensorGenerators::identity<double, 3, 3>;
  const double epfac = Kappa * (1.0 - Kappa / (Kappa + xi / eta * xi / etabar * Hiso));
  cmat = epfac * dyadic(id2, id2);
}

void Mat::PlasticDruckerPrager::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  const double young = params_->youngs_;
  const double nu = params_->poissonratio_;
  const double Hiso = params_->isohard_;
  const double cohesion = params_->cohesion_;
  const double eta = params_->eta_;
  const double xi = params_->xi_;
  const double etabar = params_->etabar_;
  const int itermax = params_->itermax_;
  const std::string& tang_str = params_->tang_;

  const double G = young / (2.0 * (1.0 + nu));
  const double kappa = young / (3.0 * (1.0 - 2.0 * nu));
  constexpr auto id2 = Core::LinAlg::TensorGenerators::identity<double, 3, 3>;

  const bool use_consistent_tangent = (tang_str != "elastic");

  FOUR_C_ASSERT_ALWAYS(strainbarpllast_.at(gp) >= 0.0,
      "accumulated plastic strain has to be equal to or greater than zero!");

  // Elastic predictor
  const Core::LinAlg::SymmetricTensor<double, 3, 3> trialstrain_e = glstrain - strainpllast_.at(gp);
  const double tracestrain = trace(trialstrain_e);
  const Core::LinAlg::SymmetricTensor<double, 3, 3> devstrain =
      trialstrain_e - id2 * (tracestrain / 3.0);

  double p = kappa * tracestrain;
  const double p_trial = p;
  Core::LinAlg::SymmetricTensor<double, 3, 3> devstress = 2.0 * G * devstrain;

  const double J2 = 0.5 * ddot(devstress, devstress);
  double strainbar_p = strainbarpllast_.at(gp);
  const double Phi_trial = std::sqrt(J2) + eta * p_trial - xi * cohesion - xi * Hiso * strainbar_p;

  double Dgamma = 0.0;
  double dstrainv = 0.0;

  if (Phi_trial / std::abs(cohesion) > params_->abstol_)
  {
    // Return to cone
    auto returnToConeFunctAndDeriv = [this, &G, &kappa, &Phi_trial](double Dgamma_init)
    { return this->return_to_cone_funct_and_deriv(Dgamma_init, G, kappa, Phi_trial); };

    Dgamma = Core::Utils::solve_local_newton(
        returnToConeFunctAndDeriv, Dgamma, params_->abstol_ * cohesion, itermax);

    if ((std::sqrt(J2) - G * Dgamma) / std::abs(cohesion) < params_->abstol_)
    {
      // Return to apex
      strainbar_p = strainbarpllast_.at(gp);
      auto returnToApexFunctAndDeriv = [this, &p_trial, &kappa, &strainbar_p](double dstrainv_init)
      { return this->return_to_apex_funct_and_deriv(dstrainv_init, p_trial, kappa, strainbar_p); };

      dstrainv = Core::Utils::solve_local_newton(
          returnToApexFunctAndDeriv, dstrainv, params_->abstol_ * cohesion, itermax);
      strainbar_p = strainbarpllast_.at(gp) + xi / eta * dstrainv;
      p = p_trial - kappa * dstrainv;
      devstress.fill(0.0);
    }
    else
    {
      strainbar_p = strainbarpllast_.at(gp) + xi * Dgamma;
      devstress *= (1.0 - G * Dgamma / std::sqrt(J2));
      p = p_trial - kappa * etabar * Dgamma;
    }

    stress = devstress + p * id2;

    // Recover elastic strain from updated stress: strain_e = s/(2G) + p/(3*kappa)*I
    const Core::LinAlg::SymmetricTensor<double, 3, 3> strain_e =
        devstress * (1.0 / (2.0 * G)) + id2 * (p / (3.0 * kappa));
    strainplcurr_.at(gp) = glstrain - strain_e;
    strainbarplcurr_.at(gp) = strainbar_p;
  }
  else
  {
    stress = devstress + p * id2;
    strainplcurr_.at(gp) = strainpllast_.at(gp);
    strainbarplcurr_.at(gp) = strainbarpllast_.at(gp);
  }

  if ((Phi_trial > 0.0) && use_consistent_tangent)
  {
    if (dstrainv != 0.0)
      setup_cmat_elasto_plastic_apex(cmat, kappa, xi, Hiso, eta, etabar);
    else
      setup_cmat_elasto_plastic_cone(cmat, Dgamma, G, kappa, devstrain, xi, Hiso, eta, etabar);
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
      const Core::LinAlg::Matrix<6, 1> voigt_strain =
          Core::LinAlg::make_strain_like_voigt_matrix(strainplcurr_.at(gp));
      for (std::size_t i = 0; i < 6; ++i)
      {
        data(gp, i) = voigt_strain(i, 0);
      }
    }
    return true;
  }
  return false;
}

std::pair<double, double> Mat::PlasticDruckerPrager::return_to_cone_funct_and_deriv(
    double Dgamma, double G, double kappa, double Phi_trial)
{
  const double Hiso = params_->isohard_;
  const double eta = params_->eta_;
  const double xi = params_->xi_;
  const double etabar = params_->etabar_;
  double Res = Phi_trial - Dgamma * (G + eta * kappa * etabar) - (xi * xi * Dgamma * Hiso);
  double d = -G - (kappa * etabar * eta) - (xi * xi * Hiso);
  return {Res, d};
}

std::pair<double, double> Mat::PlasticDruckerPrager::return_to_apex_funct_and_deriv(
    double dstrainv, double p, double kappa, double strainbar_p)
{
  const double Hiso = params_->isohard_;
  const double eta = params_->eta_;
  const double xi = params_->xi_;
  const double cohesion = params_->cohesion_;
  const double etabar = params_->etabar_;
  const double alpha = xi / eta;
  const double beta = xi / etabar;
  double Res =
      beta * cohesion + beta * strainbar_p * Hiso - p + dstrainv * (alpha * beta * Hiso + kappa);
  double d = xi * xi / eta / etabar * Hiso + kappa;
  return {Res, d};
}


FOUR_C_NAMESPACE_CLOSE
