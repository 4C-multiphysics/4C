// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_thermostvenantkirchhoff.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_stvenantkirchhoff.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN



/*----------------------------------------------------------------------*
 |                                                           dano 02/10 |
 *----------------------------------------------------------------------*/
Mat::PAR::ThermoStVenantKirchhoff::ThermoStVenantKirchhoff(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      youngs_((matdata.parameters.get<std::vector<double>>("YOUNG"))),
      poissonratio_(matdata.parameters.get<double>("NUE")),
      density_(matdata.parameters.get<double>("DENS")),
      thermexpans_(matdata.parameters.get<double>("THEXPANS")),
      thetainit_(matdata.parameters.get<double>("INITTEMP"))
{
  if (poissonratio_ >= 0.5 || poissonratio_ < -1.)
    FOUR_C_THROW("Poisson's ratio must be in [-1;0.5)");
}


/*----------------------------------------------------------------------*
 | is called in Material::Factory from read_materials()       dano 02/12 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::ThermoStVenantKirchhoff::create_material()
{
  return std::make_shared<Mat::ThermoStVenantKirchhoff>(this);
}


Mat::ThermoStVenantKirchhoffType Mat::ThermoStVenantKirchhoffType::instance_;


/*----------------------------------------------------------------------*
 | is called in Material::Factory from read_materials()       dano 02/12 |
 *----------------------------------------------------------------------*/
Core::Communication::ParObject* Mat::ThermoStVenantKirchhoffType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* thrstvenantk = new Mat::ThermoStVenantKirchhoff();
  thrstvenantk->unpack(buffer);
  return thrstvenantk;
}


/*----------------------------------------------------------------------*
 |  constructor (public)                                     dano 02/10 |
 *----------------------------------------------------------------------*/
Mat::ThermoStVenantKirchhoff::ThermoStVenantKirchhoff() : params_(nullptr) {}


/*----------------------------------------------------------------------*
 | constructor (public)                                      dano 02/10 |
 *----------------------------------------------------------------------*/
Mat::ThermoStVenantKirchhoff::ThermoStVenantKirchhoff(Mat::PAR::ThermoStVenantKirchhoff* params)
    : params_(params)
{
}


/*----------------------------------------------------------------------*
 |  Pack (public)                                            dano 02/10 |
 *----------------------------------------------------------------------*/
void Mat::ThermoStVenantKirchhoff::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);
}  // pack()


/*----------------------------------------------------------------------*
 |  Unpack (public)                                          dano 02/10 |
 *----------------------------------------------------------------------*/
void Mat::ThermoStVenantKirchhoff::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid and recover params_
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
        params_ = static_cast<Mat::PAR::ThermoStVenantKirchhoff*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }


}  // unpack()


/*----------------------------------------------------------------------*
 | calculates stresses using one of the above method to      dano 02/10 |
 | evaluate the elasticity tensor                                       |
 *----------------------------------------------------------------------*/
void Mat::ThermoStVenantKirchhoff::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  // fixme this backwards compatibility modification should be moved outside
  // use initial value as a default value
  const double temperature = [&]()
  {
    if (params.isParameter("temperature"))
    {
      return params.get<double>("temperature");
    }
    else
    {
      // std::cout << "using initial temperature" << std::endl;
      return params_->thetainit_;
    }
  }();

  setup_cmat(cmat, temperature);
  // purely mechanical part
  stress = Core::LinAlg::ddot(cmat, glstrain);

  // additive thermal part S_th = m(T) (T - T_0) I
  stress += thermal_stress_coefficient(temperature) * (temperature - params_->thetainit_) *
            Core::LinAlg::TensorGenerators::identity<double, 3, 3>;

}  // STR_Evaluate()

/*----------------------------------------------------------------------*
 | calculates strain energy                                 seitz 11/15 |
 *----------------------------------------------------------------------*/
double Mat::ThermoStVenantKirchhoff::strain_energy(
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const EvaluationContext<3>& context, const int gp, const int eleGID) const
{
  if (youngs_is_temp_dependent())
    FOUR_C_THROW("Calculation of strain energy only for constant Young's modulus");
  Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> cmat;
  setup_cmat(cmat, params_->thetainit_);
  Core::LinAlg::SymmetricTensor<double, 3, 3> s = Core::LinAlg::ddot(cmat, glstrain);
  return .5 * Core::LinAlg::ddot(s, glstrain);
}

Core::LinAlg::SymmetricTensor<double, 3, 3>
Mat::ThermoStVenantKirchhoff::evaluate_d_stress_d_scalar(
    const Core::LinAlg::Tensor<double, 3, 3>& defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext<3>& context, int gp, int eleGID)
{
  const double temperature = [&]()
  {
    if (params.isParameter("temperature"))
    {
      return params.get<double>("temperature");
    }
    else
    {
      return params_->thetainit_;
    }
  }();

  Core::LinAlg::SymmetricTensor<double, 3, 3> dS_dT{};

  // total derivative of stress (mechanical + thermal part) wrt. temperature
  // calculate derivative of cmat w.r.t. T_{n+1}
  Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> cmat_T;
  get_cmat_at_tempnp_t(cmat_T, temperature);

  // evaluate mechanical stress part
  // \f \sigma = {\mathbf C}_{,T} \,\varepsilon_{\rm GL} \f
  dS_dT = Core::LinAlg::ddot(cmat_T, glstrain);

  // calculate the temperature difference
  // Delta T = T - T_0
  const double deltaT = temperature - params_->thetainit_;

  // thermal stress part S_th = m(T) . Delta T . I
  dS_dT += (thermal_stress_coefficient_derivative(temperature) * deltaT +
               thermal_stress_coefficient(temperature)) *
           Core::LinAlg::TensorGenerators::identity<double, 3, 3>;

  return dS_dT;
}

Mat::StressTemperatureModulus Mat::ThermoStVenantKirchhoff::evaluate_stress_temperature_modulus(
    double temperature, const DeformationState& deformation_state,
    const EvaluationContext<3>& context, int gp, int eleGID)
{
  constexpr auto identity = Core::LinAlg::TensorGenerators::identity<double, 3, 3>;

  StressTemperatureModulus stress_temperature_modulus;
  stress_temperature_modulus.value = thermal_stress_coefficient(temperature) * identity;
  stress_temperature_modulus.derivative_wrt_temperature =
      thermal_stress_coefficient_derivative(temperature) * identity;
  return stress_temperature_modulus;
}

double Mat::ThermoStVenantKirchhoff::thermal_stress_coefficient(const double temperature) const
{
  const double youngs = youngs_is_temp_dependent()
                            ? get_mat_parameter_at_tempnp(&(params_->youngs_), temperature)
                            : params_->youngs_[0];

  // m = -(2 mu + 3 lambda) alpha_T = -E / (1 - 2 nu) alpha_T
  return -youngs * params_->thermexpans_ / (1.0 - 2.0 * params_->poissonratio_);
}

double Mat::ThermoStVenantKirchhoff::thermal_stress_coefficient_derivative(
    const double temperature) const
{
  if (!youngs_is_temp_dependent()) return 0.0;

  const double youngs_derivative = get_mat_parameter_at_tempnp_t(&(params_->youngs_), temperature);
  return -youngs_derivative * params_->thermexpans_ / (1.0 - 2.0 * params_->poissonratio_);
}

/*----------------------------------------------------------------------*
 | computes isotropic elasticity tensor in matrix notion     dano 02/10 |
 | for 3d                                                               |
 *----------------------------------------------------------------------*/
void Mat::ThermoStVenantKirchhoff::setup_cmat(
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, const double temperature) const
{
  // get material parameters
  // Young's modulus (modulus of elasticity)
  double Emod = 0.0;

  if (youngs_is_temp_dependent())
  {
    Emod = get_mat_parameter_at_tempnp(&(params_->youngs_), temperature);
  }
  // young's modulus is constant
  else
    Emod = params_->youngs_[0];

  // Poisson's ratio (Querdehnzahl)
  const double nu = params_->poissonratio_;

  cmat = StVenantKirchhoff::evaluate_stress_linearization(Emod, nu);
}



/*----------------------------------------------------------------------*
 | return temperature-dependent material parameter           dano 01/13 |
 | at current temperature --> polynomial type, cf. robinson material    |
 *----------------------------------------------------------------------*/
double Mat::ThermoStVenantKirchhoff::get_mat_parameter_at_tempnp(
    const std::vector<double>* paramvector,  // (i) given parameter is a vector
    const double& tempnp                     // tmpr (i) current temperature
) const
{
  // polynomial type

  // initialise the temperature dependent material parameter
  double parambytempnp = 0.0;
  double tempnp_pow = 1.0;

  // Param = a + b . T + c . T^2 + d . T^3 + ...
  // with T: current temperature
  for (double i : (*paramvector))
  {
    // calculate coefficient of variable T^i
    parambytempnp += i * tempnp_pow;
    // for the higher polynom increase the exponent of the temperature
    tempnp_pow *= tempnp;
  }

  // return temperature-dependent material parameter
  return parambytempnp;

}  // get_mat_parameter_at_tempnp()


/*----------------------------------------------------------------------*
 | calculate derivative of material parameter with respect   dano 01/13 |
 | to the current temperature --> polynomial type                       |
 *----------------------------------------------------------------------*/
double Mat::ThermoStVenantKirchhoff::get_mat_parameter_at_tempnp_t(
    const std::vector<double>* paramvector,  // (i) given parameter is a vector
    const double& tempnp                     // tmpr (i) current temperature
) const
{
  // polynomial type

  // initialise the temperature dependent material parameter
  double parambytempnp = 0.0;
  double tempnp_pow = 1.0;

  // Param = a + b . T + c . T^2 + d . T^3 + ...
  //       = a + b . N_T . T_{n+1} + c . (N_T . T_{n+1})^2 + d . (N_T . T_{n+1})^3 + ...
  // with T: current scalar-valued temperature, T = N_T . T_{n+1}

  // calculate derivative of E(T_{n+1}) w.r.t. T_{n+1}
  // d(Param)/dT . Delta T = b . N_T + 2 . c . T . N_T + 3 . d . T^2 . N_T + ...
  //                       = ( b + 2 . c . T + 3 . d . T^2 + ...) . N_T
  //                       = parambytempnp . N_T

  // the first, constant term has no influence for the derivative
  // --> start with i=1(!): neglect first term of paramvector
  for (unsigned i = 1; i < (*paramvector).size(); ++i)
  {
    // calculate coefficient of variable T^i
    parambytempnp += i * (*paramvector)[i] * tempnp_pow;
    // for the higher polynom increase the exponent of the temperature
    tempnp_pow *= tempnp;
  }

  // return derivative of temperature-dependent material parameter w.r.t. T_{n+1}
  return parambytempnp;

}  // get_mat_parameter_at_tempnp_t()


/*----------------------------------------------------------------------*
 | computes thermal derivative of the isotropic elasticity   dano 01/13 |
 | tensor in matrix notion for 3d for k_dT                              |
 *----------------------------------------------------------------------*/
void Mat::ThermoStVenantKirchhoff::get_cmat_at_tempnp_t(
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& derivcmat, const double temperature) const
{
  if (youngs_is_temp_dependent())
  {
    const double Ederiv = get_mat_parameter_at_tempnp_t(&(params_->youngs_), temperature);
    // Poisson's ratio (Querdehnzahl)
    const double nu = params_->poissonratio_;

    derivcmat = StVenantKirchhoff::evaluate_stress_linearization(Ederiv, nu);
  }
  else
  {
    derivcmat = {};
  }
}



/*----------------------------------------------------------------------*/

FOUR_C_NAMESPACE_CLOSE
