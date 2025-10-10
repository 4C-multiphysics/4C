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
      thetainit_(matdata.parameters.get<double>("INITTEMP")),
      thermomat_(matdata.parameters.get<int>("THERMOMAT"))
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
Mat::ThermoStVenantKirchhoff::ThermoStVenantKirchhoff() : params_(nullptr), thermo_(nullptr) {}


/*----------------------------------------------------------------------*
 | constructor (public)                                      dano 02/10 |
 *----------------------------------------------------------------------*/
Mat::ThermoStVenantKirchhoff::ThermoStVenantKirchhoff(Mat::PAR::ThermoStVenantKirchhoff* params)
    : params_(params), thermo_(nullptr)
{
  create_thermo_material_if_set();
}

void Mat::ThermoStVenantKirchhoff::create_thermo_material_if_set()
{
  const int thermoMatId = this->params_->thermomat_;
  if (thermoMatId != -1)
  {
    auto mat = Mat::factory(thermoMatId);
    if (mat == nullptr) FOUR_C_THROW("Failed to create thermo material, id={}", thermoMatId);
    thermo_ = std::dynamic_pointer_cast<Mat::Trait::Thermo>(mat);
  }
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

      create_thermo_material_if_set();
    }


}  // unpack()


/*----------------------------------------------------------------------*
 | calculates stresses using one of the above method to      dano 02/10 |
 | evaluate the elasticity tensor                                       |
 *----------------------------------------------------------------------*/
void Mat::ThermoStVenantKirchhoff::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext& context,
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

  reinit(defgrad, glstrain, temperature, gp);  // fixme call this before

  setup_cmat(cmat);
  // purely mechanical part
  stress = Core::LinAlg::ddot(cmat, glstrain);

  // additive thermal part
  double Tref = params_->thetainit_;
  double m = st_modulus();

  stress +=
      m * (current_temperature_ - Tref) * Core::LinAlg::TensorGenerators::identity<double, 3, 3>;

}  // STR_Evaluate()

/*----------------------------------------------------------------------*
 | calculates strain energy                                 seitz 11/15 |
 *----------------------------------------------------------------------*/
double Mat::ThermoStVenantKirchhoff::strain_energy(
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, const EvaluationContext& context,
    const int gp, const int eleGID) const
{
  if (youngs_is_temp_dependent())
    FOUR_C_THROW("Calculation of strain energy only for constant Young's modulus");
  Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> cmat;
  setup_cmat(cmat);
  Core::LinAlg::SymmetricTensor<double, 3, 3> s = Core::LinAlg::ddot(cmat, glstrain);
  return .5 * Core::LinAlg::ddot(s, glstrain);
}

void Mat::ThermoStVenantKirchhoff::evaluate(const Core::LinAlg::Matrix<3, 1>& gradtemp,
    Core::LinAlg::Matrix<3, 3>& cmat, Core::LinAlg::Matrix<3, 1>& heatflux, const int eleGID) const
{
  thermo_->evaluate(gradtemp, cmat, heatflux, eleGID);
}

void Mat::ThermoStVenantKirchhoff::evaluate(const Core::LinAlg::Matrix<2, 1>& gradtemp,
    Core::LinAlg::Matrix<2, 2>& cmat, Core::LinAlg::Matrix<2, 1>& heatflux, const int eleGID) const
{
  thermo_->evaluate(gradtemp, cmat, heatflux, eleGID);
}

void Mat::ThermoStVenantKirchhoff::evaluate(const Core::LinAlg::Matrix<1, 1>& gradtemp,
    Core::LinAlg::Matrix<1, 1>& cmat, Core::LinAlg::Matrix<1, 1>& heatflux, const int eleGID) const
{
  thermo_->evaluate(gradtemp, cmat, heatflux, eleGID);
}

std::vector<double> Mat::ThermoStVenantKirchhoff::conductivity(int eleGID) const
{
  return thermo_->conductivity(eleGID);
}

void Mat::ThermoStVenantKirchhoff::conductivity_deriv_t(Core::LinAlg::Matrix<3, 3>& dCondDT) const
{
  thermo_->conductivity_deriv_t(dCondDT);
}

void Mat::ThermoStVenantKirchhoff::conductivity_deriv_t(Core::LinAlg::Matrix<2, 2>& dCondDT) const
{
  thermo_->conductivity_deriv_t(dCondDT);
}

void Mat::ThermoStVenantKirchhoff::conductivity_deriv_t(Core::LinAlg::Matrix<1, 1>& dCondDT) const
{
  thermo_->conductivity_deriv_t(dCondDT);
}

double Mat::ThermoStVenantKirchhoff::capacity() const { return thermo_->capacity(); }

double Mat::ThermoStVenantKirchhoff::capacity_deriv_t() const
{
  return thermo_->capacity_deriv_t();
}

void Mat::ThermoStVenantKirchhoff::reinit(double temperature, unsigned gp)
{
  current_temperature_ = temperature;
  if (thermo_ != nullptr) thermo_->reinit(temperature, gp);
}
void Mat::ThermoStVenantKirchhoff::reset_current_state()
{
  if (thermo_ != nullptr) thermo_->reset_current_state();
}

void Mat::ThermoStVenantKirchhoff::commit_current_state()
{
  if (thermo_ != nullptr) thermo_->commit_current_state();
}

void Mat::ThermoStVenantKirchhoff::reinit(const Core::LinAlg::Tensor<double, 3, 3>* defgrd,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, double temperature, unsigned gp)
{
  reinit(temperature, gp);
}

Core::LinAlg::SymmetricTensor<double, 3, 3>
Mat::ThermoStVenantKirchhoff::evaluate_d_stress_d_scalar(
    const Core::LinAlg::Tensor<double, 3, 3>& defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext& context, int gp, int eleGID)
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

  reinit(&defgrad, glstrain, temperature, gp);  // fixme call this before

  Core::LinAlg::SymmetricTensor<double, 3, 3> dS_dT{};

  // total derivative of stress (mechanical + thermal part) wrt. temperature
  // calculate derivative of cmat w.r.t. T_{n+1}
  Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> cmat_T;
  get_cmat_at_tempnp_t(cmat_T);

  // evaluate mechanical stress part
  // \f \sigma = {\mathbf C}_{,T} \,\varepsilon_{\rm GL} \f
  dS_dT = Core::LinAlg::ddot(cmat_T, glstrain);

  // calculate the temperature difference
  // Delta T = T - T_0
  const double deltaT = current_temperature_ - params_->thetainit_;

  // calculate derivative of ctemp w.r.t. T_{n+1}
  Core::LinAlg::SymmetricTensor<double, 3, 3> ctemp_T;
  get_cthermo_at_tempnp_t(ctemp_T);

  // temperature dependent stress part
  // sigma = C_T . Delta T = m . I . Delta T
  dS_dT += deltaT * ctemp_T;

  setup_cthermo(ctemp_T);
  dS_dT += ctemp_T;

  return dS_dT;
}

void Mat::ThermoStVenantKirchhoff::stress_temperature_modulus_and_deriv(
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stm,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stm_dT, int gp)
{
  setup_cthermo(stm);
  get_cthermo_at_tempnp_t(stm_dT);
}

/*----------------------------------------------------------------------*
 | computes isotropic elasticity tensor in matrix notion     dano 02/10 |
 | for 3d                                                               |
 *----------------------------------------------------------------------*/
void Mat::ThermoStVenantKirchhoff::setup_cmat(
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat) const
{
  // get material parameters
  // Young's modulus (modulus of elasticity)
  double Emod = 0.0;

  if (youngs_is_temp_dependent())
  {
    Emod = get_mat_parameter_at_tempnp(&(params_->youngs_), current_temperature_);
  }
  // young's modulus is constant
  else
    Emod = params_->youngs_[0];

  // Poisson's ratio (Querdehnzahl)
  const double nu = params_->poissonratio_;

  cmat = StVenantKirchhoff::evaluate_stress_linearization(Emod, nu);
}

/*----------------------------------------------------------------------*
 | calculates stress-temperature modulus                     dano 04/10 |
 *----------------------------------------------------------------------*/
double Mat::ThermoStVenantKirchhoff::st_modulus() const
{
  const double Emod = youngs_is_temp_dependent()
                          ? get_mat_parameter_at_tempnp(&(params_->youngs_), current_temperature_)
                          : params_->youngs_[0];

  // initialise the parameters for the lame constants
  const double nu = params_->poissonratio_;

  // initialise the thermal expansion coefficient
  const double thermexpans = params_->thermexpans_;

  // plane strain, rotational symmetry
  // E / (1+nu)
  const double c1 = Emod / (1.0 + nu);
  // (E*nu) / ((1+nu)(1-2nu))
  const double b1 = c1 * nu / (1.0 - 2.0 * nu);

  // build the lame constants
  //            E
  //   mu = --------
  //        2*(1+nu)
  //                  E*nu
  //   lambda = ----------------
  //            (1+nu)*(1-2*nu)
  //
  //  \f \mu =  \frac{E}{2(1+\nu)} \f
  const double mu = 0.5 * c1;
  // lambda
  // \f \frac{E\,\nu}{(1-2\nu)(1+\nu)} \f
  const double lambda = b1;

  // stress-temperature modulus
  // \f m\, = \, -(2\,\cdot \mu \, +\, 3\cdot\lambda)\cdot\varalpha_T \f
  const double stmodulus = (-1.0) * (2.0 * mu + 3.0 * lambda) * thermexpans;

  return stmodulus;

}  // st_modulus()


/*----------------------------------------------------------------------*
 | computes temperature dependent isotropic                  dano 05/10 |
 | elasticity tensor in matrix notion for 3d, second(!) order tensor    |
 *----------------------------------------------------------------------*/
void Mat::ThermoStVenantKirchhoff::setup_cthermo(
    Core::LinAlg::SymmetricTensor<double, 3, 3>& ctemp) const
{
  double m = st_modulus();
  fill_cthermo(ctemp, m);
}

void Mat::ThermoStVenantKirchhoff::fill_cthermo(
    Core::LinAlg::SymmetricTensor<double, 3, 3>& ctemp, double m)
{
  // isotropic elasticity tensor C_temp in Voigt matrix notation C_temp = m I
  //
  // Matrix-notation for 3D case
  //       [ m      0      0 ]
  // C_T = [ 0      m      0 ]
  //       [ 0      0      m ]
  //
  // in Vector notation
  // C_T = [m, m, m, 0, 0, 0]^T
  //
  // write non-zero components

  // clear the material tangent, equal to PutScalar(0.0), but faster
  ctemp = m * Core::LinAlg::TensorGenerators::identity<double, 3, 3>;
  // else zeros
}
// setup_cthermo()


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
 | calculate linearisation of stress-temperature modulus     dano 04/10 |
 | w.r.t. T_{n+1} for k_dT, k_TT                                        |
 *----------------------------------------------------------------------*/
double Mat::ThermoStVenantKirchhoff::get_st_modulus_t() const
{
  // build the derivative of the stress-temperature modulus w.r.t. T_{n+1}
  // m = - (2 . mu + 3 . lambda) . varalpha_T
  //   = - (2 . nu / ((1+nu)(1-2nu)) + 3 / (2 . (1+nu))) . varalpha_T . E(T)
  double stmodulus_T = 0.0;

  if (youngs_is_temp_dependent())
  {
    const double Ederiv = get_mat_parameter_at_tempnp_t(&(params_->youngs_), current_temperature_);

    // initialise the parameters for the lame constants
    const double nu = params_->poissonratio_;

    // initialise the thermal expansion coefficient
    const double thermexpans = params_->thermexpans_;

    // plane strain, rotational symmetry
    // E / (1+nu)
    const double c1 = Ederiv / (1.0 + nu);
    // (E . nu) / ((1+nu)(1-2nu))
    const double b1 = c1 * nu / (1.0 - 2.0 * nu);

    // build the lame constants
    //         E
    // mu = --------  --> \f \mu = \frac{E}{2(1+\nu)} \f
    //      2*(1+nu)
    const double mu = 0.5 * c1;
    //              E*nu
    // lambda = --------------- --> \f \frac{E\,\nu}{(1-2\nu)(1+\nu)} \f
    //          (1+nu)*(1-2*nu)
    const double lambda = b1;

    // build the derivative of the stress-temperature modulus w.r.t. T_{n+1}
    // m = -(2 . mu + 3 . lambda) . varalpha_T
    stmodulus_T = (-1.0) * (2.0 * mu + 3.0 * lambda) * thermexpans;
  }
  // else (young_temp == false)
  // constant young's modulus, i.e. independent of T, no linearisation, return 0;

  return stmodulus_T;

}  // get_st_modulus_t()

/*----------------------------------------------------------------------*
 | computes thermal derivative of the isotropic elasticity   dano 01/13 |
 | tensor in matrix notion for 3d for k_dT                              |
 *----------------------------------------------------------------------*/
void Mat::ThermoStVenantKirchhoff::get_cmat_at_tempnp_t(
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& derivcmat) const
{
  if (youngs_is_temp_dependent())
  {
    const double Ederiv = get_mat_parameter_at_tempnp_t(&(params_->youngs_), current_temperature_);
    // Poisson's ratio (Querdehnzahl)
    const double nu = params_->poissonratio_;

    derivcmat = StVenantKirchhoff::evaluate_stress_linearization(Ederiv, nu);
  }
  else
  {
    derivcmat = {};
  }
}


/*----------------------------------------------------------------------*
 | computes linearisation of temperature-dependent isotropic dano 01/13 |
 | elasticity tensor in matrix notion for 3d, 2nd order tensor          |
 *----------------------------------------------------------------------*/
void Mat::ThermoStVenantKirchhoff::get_cthermo_at_tempnp_t(
    Core::LinAlg::SymmetricTensor<double, 3, 3>& derivctemp) const
{
  double m_T = get_st_modulus_t();

  fill_cthermo(derivctemp, m_T);
}


/*----------------------------------------------------------------------*/

FOUR_C_NAMESPACE_CLOSE
