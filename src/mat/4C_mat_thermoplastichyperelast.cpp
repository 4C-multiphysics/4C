// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_thermoplastichyperelast.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | constructor (public)                                      dano 03/13 |
 *----------------------------------------------------------------------*/
Mat::PAR::ThermoPlasticHyperElast::ThermoPlasticHyperElast(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      youngs_(matdata.parameters.get<double>("YOUNG")),
      poissonratio_(matdata.parameters.get<double>("NUE")),
      density_(matdata.parameters.get<double>("DENS")),
      cte_(matdata.parameters.get<double>("CTE")),
      inittemp_(matdata.parameters.get<double>("INITTEMP")),
      yield_(matdata.parameters.get<double>("YIELD")),
      isohard_(matdata.parameters.get<double>("ISOHARD")),
      sathardening_(matdata.parameters.get<double>("SATHARDENING")),
      hardexpo_(matdata.parameters.get<double>("HARDEXPO")),
      yieldsoft_(matdata.parameters.get<double>("YIELDSOFT")),
      hardsoft_(matdata.parameters.get<double>("HARDSOFT")),
      abstol_(matdata.parameters.get<double>("TOL")),
      thermomat_(matdata.parameters.get<int>("THERMOMAT"))
{
  if (sathardening_ < yield_)
    FOUR_C_THROW("Saturation hardening must not be less than initial yield stress!");
  if (hardexpo_ < 0.0) FOUR_C_THROW("Nonlinear hardening exponent must be non-negative!");
}


/*----------------------------------------------------------------------*
 | is called in Material::Factory from read_materials()       dano 03/13 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::ThermoPlasticHyperElast::create_material()
{
  return std::make_shared<Mat::ThermoPlasticHyperElast>(this);
}

Mat::ThermoPlasticHyperElastType Mat::ThermoPlasticHyperElastType::instance_;


/*----------------------------------------------------------------------*
 | is called in Material::Factory from read_materials()       dano 03/13 |
 *----------------------------------------------------------------------*/
Core::Communication::ParObject* Mat::ThermoPlasticHyperElastType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::ThermoPlasticHyperElast* thrplhyper = new Mat::ThermoPlasticHyperElast();
  thrplhyper->unpack(buffer);
  return thrplhyper;
}


/*----------------------------------------------------------------------*
 | constructor (public)                                      dano 03/13 |
 *----------------------------------------------------------------------*/
Mat::ThermoPlasticHyperElast::ThermoPlasticHyperElast() : params_(nullptr), thermo_(nullptr) {}


/*----------------------------------------------------------------------*
 | copy-constructor (public)                                 dano 03/13 |
 *----------------------------------------------------------------------*/
Mat::ThermoPlasticHyperElast::ThermoPlasticHyperElast(Mat::PAR::ThermoPlasticHyperElast* params)
    : params_(params), thermo_(nullptr), plastic_step_(false)
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
 | pack (public)                                             dano 03/13 |
 *----------------------------------------------------------------------*/
void Mat::ThermoPlasticHyperElast::pack(Core::Communication::PackBuffer& data) const
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
  // if material is not initialised, i.e. start simulation, nothing to pack
  if (!initialized())
  {
    histsize = 0;
  }
  else
  {
    // if material is initialised (restart): size equates number of gausspoints
    histsize = defgrdlast_->size();
  }

  add_to_pack(data, histsize);  // length of history vector(s)
  for (int var = 0; var < histsize; ++var)
  {
    // insert history vectors to add_to_pack
    add_to_pack(data, defgrdlast_->at(var));
    add_to_pack(data, bebarlast_->at(var));
    add_to_pack(data, accplstrainlast_->at(var));

    // variables corresponding to temperature-dependency
    add_to_pack(data, mechdiss_->at(var));
    add_to_pack(data, mechdiss_k_tt_->at(var));
    add_to_pack(data, mechdiss_k_td_->at(var));
    add_to_pack(data, cmat_kd_t_->at(var));
    add_to_pack(data, thrplheat_->at(var));
    add_to_pack(data, thrplheat_k_tt_->at(var));
    add_to_pack(data, thrplheat_k_td_->at(var));

    // insert current iteration states
    add_to_pack(data, defgrdcurr_->at(var));
    add_to_pack(data, bebarcurr_->at(var));
    add_to_pack(data, accplstraincurr_->at(var));
  }

  add_to_pack(data, plastic_step_);
}  // pack()


/*----------------------------------------------------------------------*
 | unpack (public)                                           dano 03/13 |
 *----------------------------------------------------------------------*/
void Mat::ThermoPlasticHyperElast::unpack(Core::Communication::UnpackBuffer& buffer)
{
  isinit_ = true;


  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid
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
        params_ = static_cast<Mat::PAR::ThermoPlasticHyperElast*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
  }

  // history data
  int histsize;
  extract_from_pack(buffer, histsize);

  // if system is not yet initialised, the history vectors have to be initialized
  if (histsize == 0) isinit_ = false;

  defgrdlast_ = std::make_shared<std::vector<Core::LinAlg::Tensor<double, 3, 3>>>();
  defgrdcurr_ = std::make_shared<std::vector<Core::LinAlg::Tensor<double, 3, 3>>>();

  bebarlast_ = std::make_shared<std::vector<Core::LinAlg::Matrix<3, 3>>>();
  bebarcurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<3, 3>>>();

  accplstrainlast_ = std::make_shared<std::vector<double>>();
  accplstraincurr_ = std::make_shared<std::vector<double>>();

  mechdiss_ = std::make_shared<std::vector<double>>();
  mechdiss_k_tt_ = std::make_shared<std::vector<double>>();
  mechdiss_k_td_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  cmat_kd_t_ = std::make_shared<std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>>>();
  thrplheat_ = std::make_shared<std::vector<double>>();
  thrplheat_k_tt_ = std::make_shared<std::vector<double>>();
  thrplheat_k_td_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();

  for (int var = 0; var < histsize; ++var)
  {
    // initialise
    Core::LinAlg::Matrix<3, 3> tmp_matrix(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Tensor<double, 3, 3> tmp_tensor;
    Core::LinAlg::SymmetricTensor<double, 3, 3> tmp_stensor;
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> tmp_vect(Core::LinAlg::Initialization::zero);
    double tmp_scalar = 0.0;

    extract_from_pack(buffer, tmp_tensor);
    defgrdlast_->push_back(tmp_tensor);

    extract_from_pack(buffer, tmp_matrix);
    bebarlast_->push_back(tmp_matrix);

    extract_from_pack(buffer, tmp_scalar);
    accplstrainlast_->push_back(tmp_scalar);

    extract_from_pack(buffer, tmp_scalar);
    mechdiss_->push_back(tmp_scalar);

    extract_from_pack(buffer, tmp_scalar);
    mechdiss_k_tt_->push_back(tmp_scalar);

    extract_from_pack(buffer, tmp_vect);
    mechdiss_k_td_->push_back(tmp_vect);

    extract_from_pack(buffer, tmp_stensor);
    cmat_kd_t_->push_back(tmp_stensor);

    extract_from_pack(buffer, tmp_scalar);
    thrplheat_->push_back(tmp_scalar);

    extract_from_pack(buffer, tmp_scalar);
    thrplheat_k_tt_->push_back(tmp_scalar);

    extract_from_pack(buffer, tmp_vect);
    thrplheat_k_td_->push_back(tmp_vect);

    // current iteration states are unpacked
    extract_from_pack(buffer, tmp_tensor);
    defgrdcurr_->push_back(tmp_tensor);
    extract_from_pack(buffer, tmp_matrix);
    bebarcurr_->push_back(tmp_matrix);
    extract_from_pack(buffer, tmp_scalar);
    accplstraincurr_->push_back(tmp_scalar);
  }

  extract_from_pack(buffer, plastic_step_);
}


/*---------------------------------------------------------------------*
 | initialise / allocate internal variables (public)                   |
 *---------------------------------------------------------------------*/
void Mat::ThermoPlasticHyperElast::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  // initialise hist variables
  defgrdlast_ = std::make_shared<std::vector<Core::LinAlg::Tensor<double, 3, 3>>>();
  defgrdcurr_ = std::make_shared<std::vector<Core::LinAlg::Tensor<double, 3, 3>>>();

  bebarlast_ = std::make_shared<std::vector<Core::LinAlg::Matrix<3, 3>>>();
  bebarcurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<3, 3>>>();

  accplstrainlast_ = std::make_shared<std::vector<double>>();
  accplstraincurr_ = std::make_shared<std::vector<double>>();

  mechdiss_ = std::make_shared<std::vector<double>>();
  mechdiss_k_tt_ = std::make_shared<std::vector<double>>();
  mechdiss_k_td_ = std::make_shared<std::vector<Core::LinAlg::Matrix<6, 1>>>();
  cmat_kd_t_ = std::make_shared<std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>>>();
  thrplheat_ = std::make_shared<std::vector<double>>();
  thrplheat_k_tt_ = std::make_shared<std::vector<double>>();
  thrplheat_k_td_ = std::make_shared<std::vector<Core::LinAlg::Matrix<6, 1>>>();

  defgrdlast_->resize(numgp);
  defgrdcurr_->resize(numgp);

  bebarlast_->resize(numgp);
  bebarcurr_->resize(numgp);

  accplstrainlast_->resize(numgp);
  accplstraincurr_->resize(numgp);

  mechdiss_->resize(numgp);
  mechdiss_k_tt_->resize(numgp);
  mechdiss_k_td_->resize(numgp);
  cmat_kd_t_->resize(numgp);
  thrplheat_->resize(numgp);
  thrplheat_k_tt_->resize(numgp);
  thrplheat_k_td_->resize(numgp);

  Core::LinAlg::Matrix<3, 3> emptymat(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++) emptymat(i, i) = 1.0;
  Core::LinAlg::Matrix<6, 1> emptyvect(Core::LinAlg::Initialization::zero);

  for (int i = 0; i < numgp; i++)
  {
    defgrdlast_->at(i) =
        Core::LinAlg::get_full(Core::LinAlg::TensorGenerators::identity<double, 3, 3>);
    defgrdcurr_->at(i) =
        Core::LinAlg::get_full(Core::LinAlg::TensorGenerators::identity<double, 3, 3>);

    bebarlast_->at(i) = emptymat;
    bebarcurr_->at(i) = emptymat;

    accplstrainlast_->at(i) = 0.0;
    accplstraincurr_->at(i) = 0.0;

    mechdiss_->at(i) = 0.0;
    mechdiss_k_tt_->at(i) = 0.0;
    mechdiss_k_td_->at(i) = emptyvect;
    cmat_kd_t_->at(i) = {};
    thrplheat_->at(i) = 0.0;
    thrplheat_k_tt_->at(i) = 0.0;
    thrplheat_k_td_->at(i) = emptyvect;
  }

  isinit_ = true;
}  // setup()


/*----------------------------------------------------------------------*
 | update internal variables                                 dano 03/13 |
 *----------------------------------------------------------------------*/
void Mat::ThermoPlasticHyperElast::update()
{
  // make current values at time step t_n+1 to values of last step t_n
  defgrdlast_ = defgrdcurr_;
  bebarlast_ = bebarcurr_;
  accplstrainlast_ = accplstraincurr_;

  // empty vectors of current data
  defgrdcurr_ = std::make_shared<std::vector<Core::LinAlg::Tensor<double, 3, 3>>>();
  bebarcurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<3, 3>>>();
  accplstraincurr_ = std::make_shared<std::vector<double>>();

  // get the size of the vector
  // (use the last vector, because it includes latest results, current is empty)
  const int histsize = defgrdlast_->size();
  defgrdcurr_->resize(histsize);
  bebarcurr_->resize(histsize);
  accplstraincurr_->resize(histsize);

  Core::LinAlg::Matrix<3, 3> emptymat(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < histsize; i++)
  {
    defgrdcurr_->at(i) = {};
    bebarcurr_->at(i) = emptymat;
    accplstraincurr_->at(i) = 0.0;
  }
}  // update()

/*----------------------------------------------------------------------*
 | Set current quantities for this material                             |
 *----------------------------------------------------------------------*/
void Mat::ThermoPlasticHyperElast::reinit(const Core::LinAlg::Tensor<double, 3, 3>* defgrd,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, double temperature, unsigned gp)
{
  reinit(temperature, gp);
}

/*----------------------------------------------------------------------*
 | calculate stress-temperature modulus and thermal derivative          |
 |   for coupled thermomechanics                                        |
 *----------------------------------------------------------------------*/
void Mat::ThermoPlasticHyperElast::stress_temperature_modulus_and_deriv(
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stm,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stm_dT, int gp)
{
  const auto& defgrad = (*defgrdcurr_)[gp];

  // inverse of right Cauchy-Green tensor = F^{-1} . F^{-T}
  Core::LinAlg::SymmetricTensor<double, 3, 3> cauchygreen =
      Core::LinAlg::assume_symmetry(Core::LinAlg::transpose(defgrad) * defgrad);
  Core::LinAlg::SymmetricTensor<double, 3, 3> Cinv = Core::LinAlg::inv(cauchygreen);

  setup_cthermo(stm, Core::LinAlg::det(defgrad), Cinv);

  stm_dT = cmat_kd_t_->at(gp);
}

/*----------------------------------------------------------------------*
 |  Evaluates the added derivatives of the stress w.r.t. all scalars    |
 *----------------------------------------------------------------------*/
Core::LinAlg::SymmetricTensor<double, 3, 3>
Mat::ThermoPlasticHyperElast::evaluate_d_stress_d_scalar(
    const Core::LinAlg::Tensor<double, 3, 3>& defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext& context, int gp, int eleGID)
{
  // obtain the temperature
  const double temperature = [&]()
  {
    if (params.isParameter("temperature"))
    {
      return params.get<double>("temperature");
    }
    else
    {
      return params_->inittemp_;
    }
  }();

  reinit(&defgrad, glstrain, temperature, gp);  // fixme call this before

  // inverse of right Cauchy-Green tensor = F^{-1} . F^{-T}
  Core::LinAlg::SymmetricTensor<double, 3, 3> cauchygreen =
      Core::LinAlg::assume_symmetry(Core::LinAlg::transpose(defgrad) * defgrad);
  Core::LinAlg::SymmetricTensor<double, 3, 3> Cinv = Core::LinAlg::inv(cauchygreen);

  // get the temperature-dependent mechanical material tangent
  Core::LinAlg::SymmetricTensor<double, 3, 3> dS_dT{};
  Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> cmat_T;
  setup_cmat_thermo(current_temperature_, cmat_T, defgrad);
  // evaluate mechanical stress part
  dS_dT = Core::LinAlg::ddot(cmat_T, glstrain);

  // get the temperature-dependent material tangent
  Core::LinAlg::SymmetricTensor<double, 3, 3> ctemp{};
  setup_cthermo(ctemp, Core::LinAlg::det(defgrad), Cinv);

  // add the derivatives of thermal stress w.r.t temperature
  dS_dT += ctemp;

  return dS_dT;
}

/*----------------------------------------------------------------------*
 | calculate stress and constitutive tensor                  dano 03/13 |
 *----------------------------------------------------------------------*/
void Mat::ThermoPlasticHyperElast::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, const int gp, const int eleGID)
{
  Core::LinAlg::Matrix<3, 3> defgrd_mat = Core::LinAlg::make_matrix_view(*defgrad);
  if (eleGID == -1) FOUR_C_THROW("no element provided in material");

  // elastic material data
  // -------------------------------------------------- get material parameters
  // Young's modulus
  const double ym = params_->youngs_;
  // Poisson's ratio
  const double nu = params_->poissonratio_;
  // shear modulus, mu=G
  const double G = ym / (2.0 * (1.0 + nu));
  // bulk modulus
  const double bulk = ym / (3.0 * (1.0 - 2.0 * nu));
  // linear isotropic hardening modulus
  double Hiso = params_->isohard_;
  // initial yield stress
  double sigma_y0 = params_->yield_;
  // yield stress softening
  double omega_0 = params_->yieldsoft_;
  // hardening softening
  double omega_h = params_->hardsoft_;
  // saturation hardening
  double sigma_y0infty = params_->sathardening_;
  // hardening exponent
  double hardexpo = params_->hardexpo_;
  // reference temperature
  double inittemp = params_->inittemp_;

  // 3x3 2nd-order identity matrix
  Core::LinAlg::Matrix<3, 3> id2(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++) id2(i, i) = 1.0;

  // start with current deformation
  defgrdcurr_->at(gp) = *defgrad;
  // get the inverse F^{-1}
  Core::LinAlg::Matrix<3, 3> invdefgrdcurr(defgrd_mat);
  invdefgrdcurr.invert();
  // calculate the Jacobi-determinant J = det(F_{n+1})
  double J = defgrd_mat.determinant();
  // determinant has to be >= 0
  // in case of too large dt, determinant is often negative --> reduce dt
  if (J < 0) FOUR_C_THROW("Jacobi determinant is not allowed to be less than zero!");

  // ------------------------------------------------ multiplicative split of F
  // split deformation gradient in elastic and plastic part
  // F_{n+1} = F_{n+1}^e . F_{n+1}^p

  // relative deformation gradient
  // f_{n+1} = F_{n+1} . (F_n)^-1
  Core::LinAlg::Tensor<double, 3, 3> invdefgrdlast = Core::LinAlg::inv(defgrdlast_->at(gp));
  Core::LinAlg::Tensor<double, 3, 3> defgrddelta = *defgrad * invdefgrdlast;

  // isochoric part of relative deformation gradient
  // fbar_{n+1} = Fbar_{n+1} . Fbar_n^{-1} = (J_{n+1}/J_n)^{-1/3}) . f_{n+1}
  // with J_{n+1}/J_n = det(fbar_)
  Core::LinAlg::Matrix<3, 3> defgrddeltabar = Core::LinAlg::make_matrix(defgrddelta);
  defgrddeltabar.scale(pow(Core::LinAlg::det(defgrddelta), -1.0 / 3.0));

  // --------------------------------------------------------------------------
  // elastic predictor (trial values)
  // --------------------------------------------------------------------------

  // ----------------------------------------------------- elastic trial strain

  // assume load step is elastic
  // elastic left Cauchy-Green (LCG) trial state (isochoric) (9.3.13)
  // bbar_{n+1}^{e,trial} = Fbar_{n+1} (Cbar_{n}^{p-1}) . Fbar_{n+1}^T
  //                      = fbar_{n+1} (bbar_{n} . fbar_{n+1}^T
  // with history variable Cbar_{n+1}^{p-1})^{trial} = Cbar_{n}^{p-1}
  Core::LinAlg::Matrix<3, 3> bebar_trial(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Matrix<3, 3> tmp(Core::LinAlg::Initialization::uninitialized);
  tmp.multiply(defgrddeltabar, bebarlast_->at(gp));
  bebar_trial.multiply_nt(tmp, defgrddeltabar);
  // trace of strain vector
  double tracebebar = bebar_trial(0, 0) + bebar_trial(1, 1) + bebar_trial(2, 2);

  // ------------------------------------------------------- trial stress

  // trial Kirchhoff stress deviator (9.3.9)
  // s_{n+1)^{trial} = G . dev_bebar_{n+1}^{e,trial}
  // dev_bebar_trial = bebar_trial - volstrain^e
  //                 = bebar_trial - 1/3 . tr( bebar_trial ) . id2
  Core::LinAlg::Matrix<3, 3> devtau_trial(bebar_trial);
  for (int i = 0; i < 3; i++) devtau_trial(i, i) -= 1.0 / 3.0 * tracebebar;
  devtau_trial.scale(G);

  // trial equivalent von Mises stress
  // q^{trial} = sqrt(s^{trial}_ij . s^{trial}_ij)
  double q_trial = 0.0;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) q_trial += devtau_trial(i, j) * devtau_trial(i, j);
  q_trial = sqrt(q_trial);

  // -------------------------- extract scalar-valued element temperature
  // initialise temperature
  double scalartemp = 0.0;
  if (params.getEntryPtr("temperature") != nullptr)
  {
    // TSI, i.e. temperature is available --> use this temperature
    scalartemp = params.get<double>("temperature");
  }
  // in case of purely structural analysis, i.e. isothermal: T = T_0, DeltaT = 0
  else
    scalartemp = inittemp;

  // ------------------------ temperature-dependent yield stress function

  // temperature-dependent isotropic hardening modulus
  double Hiso_temp = 0.0;
  Hiso_temp = Hiso * (1.0 - omega_h * (scalartemp - inittemp));

  // temperature-dependent yield stress
  double sigma_y0_temp = 0.0;
  sigma_y0_temp = sigma_y0 * (1.0 - omega_0 * (scalartemp - inittemp));

  // temperature-dependent saturation hardening
  double sigma_y0infty_temp = 0.0;
  sigma_y0infty_temp = sigma_y0infty * (1.0 - omega_h * (scalartemp - inittemp));

  // get old accumulated or equivalent plastic strain accplstrainlast
  double alpha = 0.0;
  alpha = accplstrainlast_->at(gp);
  if (accplstrainlast_->at(gp) < 0.0)
    FOUR_C_THROW("accumulated plastic strain has to be equal to or greater than 0!");

  // thermodynamic force / hardening flux describing nonlinear isotropic hardening
  // sigma_iso = rho (d psi^_p) / (d accplstrain)
  // rho psi^_p = 1/2 . h(T) . alpha^2
  //              + [sigma_y0infty(T) - sigma_y(T)] . (alpha - (1- exp(-delta . alpha))
  // --> sigma_iso = h(T) . alpha + [sigma_y0infty(T) - sigma_y0(T)] . (1 - exp(-delta . alpha)
  double sigma_iso =
      Hiso_temp * alpha + (sigma_y0infty_temp - sigma_y0_temp) * (1.0 - exp(-hardexpo * alpha));

  // complete yield stress
  // sigma_y = sigma_y0_temp + sigma_iso
  double sigma_y = sigma_y0_temp + sigma_iso;

  // calculate yield function at trial state
  // Phi = || s_{n+1}^{trial} || - sqrt(2/3) . sigma_y
  double Phi_trial = 0.0;
  Phi_trial = q_trial - sqrt(2.0 / 3.0) * sigma_y;

  // stress variables tau = J_{n+1} . p_{n+1} . I + s_{n+1}
  Core::LinAlg::Matrix<3, 3> devtau(Core::LinAlg::Initialization::uninitialized);

  // some computations
  // mubar = 1/3 mu tr(bebar_{n+1}^{e,trial})
  double mubar = G * 1.0 / 3.0 * tracebebar;

  // initialise incremental plastic multiplier Delta gamma
  double Dgamma = 0.0;

  // unit spatial flow vector
  // n = s^{trial}_{n+1} / || s^{trial}_{n+1} || = s^{trial}_{n+1} / q^{trial}
  Core::LinAlg::Matrix<3, 3> n(devtau_trial);
  if (q_trial != 0.0) n.scale(1.0 / q_trial);

  //-------------------------------------------------------------------
  // IF: elastic step (Phi_trial <= 0.0, Dgamma = 0.0)
  //-------------------------------------------------------------------
  if (Phi_trial <= 0.0)
  {
    // trial state vectors = result vectors of time step n+1

    // _n^{trial} --> _{n+1}
    bebarcurr_->at(gp) = bebar_trial;
    accplstraincurr_->at(gp) = accplstrainlast_->at(gp);
    devtau = devtau_trial;
    Dgamma = 0.0;

    // elastic load --> values are zero
    mechdiss_->at(gp) = 0.0;
    mechdiss_k_tt_->at(gp) = 0.0;
    mechdiss_k_td_->at(gp).put_scalar(0.0);
    cmat_kd_t_->at(gp) = {};
    thrplheat_->at(gp) = 0.0;
    thrplheat_k_tt_->at(gp) = 0.0;
    thrplheat_k_td_->at(gp).put_scalar(0.0);
  }  // end if (Phi_trial <= 0.0), i.e. elastic step

  //-------------------------------------------------------------------
  // ELSE consistency condition is violated, i.e. plastic load step
  // (Phi_trial > 0.0, Dgamma >= 0.0)
  //-------------------------------------------------------------------
  else
  {
    // only first plastic call is output at screen for every processor
    // visualisation of whole plastic behaviour via PLASTIC_STRAIN in postprocessing
    if (plastic_step_ == false)
    {
      plastic_ele_id_ = eleGID;

      if ((plastic_step_ == false) and (eleGID == plastic_ele_id_) and (gp == 0))
        std::cout << "plasticity starts in element = " << plastic_ele_id_ << std::endl;

      plastic_step_ = true;
    }

    // ------------ local Newton Raphson to determine plastic multiplier Dgamma

    // initialise
    const int itermax = 50;  // max. number of iterations
    int itnum = 0;           // iteration counter

    // Res:= residual of Newton iteration == yield function Phi
    double Res = 0.0;
    // calculate derivative of residual or tangent
    // ResTan = Phi' = d(Phi)/d(Dgamma)
    double ResTan = 0.0;

    // start iteration with index m for local Newton
    while (true)
    {
      itnum++;
      // check for convergence
      if (itnum > itermax)
      {
        FOUR_C_THROW(
            "local Newton iteration did not converge after iteration {:3d}/{:3d} with Res={:3f}",
            itnum, itermax, Res);
      }  // itnum > itermax
      // else: continue loop, i.e. m <= m_max

      // Res := Phi = qbar^{trial}_{n+1} - sqrt{2/3} (sigma_y0 + sigma_iso(alpha^{trial} )
      //      = qbar^{trial}_{n+1} - sqrt{2/3} (sigma_y0 + sigma_iso(alpha_n )
      Res = q_trial - 2.0 * mubar * Dgamma -
            sqrt(2.0 / 3.0) *
                (sigma_y0_temp + Hiso_temp * alpha +
                    (sigma_y0infty_temp - sigma_y0_temp) * (1.0 - exp(-hardexpo * alpha)));

      // check for convergence
      double norm = abs(Res);

      // check: absolute value of Res has to be smaller than given tolerance
      if (norm < (params_->abstol_))
      {
        break;
      }

      // tangent
      // ResTan = - 2 . mubar - sqrt(2/3) . [ H_iso . sqrt(2/3) +
      // (sigma_y0infty - sigma_y0) . (-exp(-delta . alpha)) . (-delta . sqrt(2/3))
      ResTan = -2.0 * mubar - (2.0 / 3.0) * Hiso_temp -
               2.0 / 3.0 * (sigma_y0infty_temp - sigma_y0_temp) * exp(-hardexpo * alpha) * hardexpo;

      // incremental plastic multiplier Dgamma
      // Dgamma^{m} = Dgamma^{m-1} - Phi / Phi'
      Dgamma += (-Res) / ResTan;
      // update accumulated plastic strain
      // alpha_{n+1} = alpha_n + sqrt{2/3} . Dgamma
      alpha = accplstrainlast_->at(gp) + (sqrt(2.0 / 3.0) * Dgamma);

      // q_trial and bebar (mubar) maintain constant during iteration

    }  // end of local Newton Raphson iteration

    // ------------------------------------------------ update Kirchhoff stress

    // update deviatoric Kirchhoff stress
    // s_{n+1} = s_{n+1}^{trial} - 2 . mubar . Delta gamma . n
    devtau.update(devtau_trial);
    devtau.update(-2.0 * mubar * Dgamma, n, 1.0);

    // --------------------------------------------------------- update history

    // update accumulated plastic strain
    accplstraincurr_->at(gp) = alpha;

    // update elastic LCG
    // bbar_{n+1}^e = bbar_{n+1}^{e,trial} - 2/3 . Dgamma . tr(bbar_{n+1}^{e,trial}) . n
    // see e.g. Simo, Comp.Inelaticity (9.3.7)
    // for nightly test case use present update procedure
    bebarcurr_->at(gp) = (bebar_trial);
    bebarcurr_->at(gp).update((-2.0 / 3.0 * Dgamma * tracebebar), n, 1.0);

    // plastic step update
    if (Dgamma != 0.0)
    {
      // ----------------------------------------- preliminary calculations

      // ------------------------------------- calculate plastic directions

      // pull-back of the spatial flow vector n to N
      // N = F^{-1} . n . F^{-T}
      Core::LinAlg::Matrix<3, 3> N(Core::LinAlg::Initialization::uninitialized);
      tmp.put_scalar(0.0);  // reuse tmp, but reset first
      tmp.multiply(invdefgrdcurr, n);
      N.multiply_nt(tmp, invdefgrdcurr);

      // pull-back of the deviatoric part of n^2 to dev[N^2]
      // dev (n^2)
      Core::LinAlg::Matrix<3, 3> devnsquare(Core::LinAlg::Initialization::uninitialized);
      devnsquare.multiply(n, n);
      double tracensquare = (devnsquare(0, 0) + devnsquare(1, 1) + devnsquare(2, 2));
      for (int i = 0; i < 3; i++) devnsquare(i, i) -= 1.0 / 3.0 * tracensquare;
      // dev (N^2) = F^{-1} . dev(n^2) . F^{-T}
      Core::LinAlg::Matrix<3, 3> devNsquare(Core::LinAlg::Initialization::uninitialized);
      tmp.put_scalar(0.0);  // reuse tmp, but reset first
      tmp.multiply(invdefgrdcurr, devnsquare);
      devNsquare.multiply_nt(tmp, invdefgrdcurr);

      // ------------------------------------------------------- linearisations
      // dsigma_y0_temp/dT_{n+1} = - omega_0 . sigma_y0 . N_T
      double dsigma_y0_temp_dT = sigma_y0 * (-omega_0);

      // dkappa_temp/dT_{n+1} = [- omega_h . Hiso . astrain^p +
      //                         + (- omega_h . sigma_y0infty + omega_0 . sigma_y0)
      //                            . (1 - exp(-delta . astrain^p)) ]  . N_T
      double dkappaT_dT =
          -omega_h * Hiso * alpha +
          (sigma_y0infty * (-omega_h) - sigma_y0 * (-omega_0)) * (1.0 - exp(-hardexpo * alpha));

      // dkappa_temp/dastrain_{n+1} = Hiso_temp +
      //      + (-delta) . (sigma_y0infty_temp - sigma_y0_temp) . [-exp(-delta . astrain)]
      double dkappa_dastrain = Hiso_temp + (sigma_y0infty_temp - sigma_y0_temp) *
                                               (-exp(-hardexpo * alpha)) * (-hardexpo);

      // beta_0 = 1 + 1/(3 mubar) . dkappa/dastrain^p
      double beta0 = 1.0 + 1.0 / (3.0 * mubar) * dkappa_dastrain;

      // dDgamma/dT_{n+1} = -sqrt(2/3) . dsigma_y/dT / [ 2 . mubar . beta0 ]
      double dDgamma_dT =
          -sqrt(2.0 / 3.0) * (dsigma_y0_temp_dT + dkappaT_dT) / (2.0 * mubar * beta0);

      // dkappa/dT = dkappa(T)/dT + dkappa(astrain)/dT . dastrain/dDgamma . dDgamma/dT
      //           = dkappaT_dT + dkappa_dastrain . \sqrt{2/3} . dDgamma_dT
      double dkappa_dT = dkappaT_dT + dkappa_dastrain * sqrt(2.0 / 3.0) * dDgamma_dT;

      // -------------------------------- calculate second derivatives of kappa
      // 2nd derivatives is required for K_TT/K_Td

      // d/dT(dkappa/dT) = d/dT(dkappa_T/dT) + d/dT(dkappa_dastrain . dastrain/dDgamma . dDgamma/dT)

      // purely thermal derivatives d/dT(dkappa_T/dT)
      // ddkappa_T_dTT = d/dT(dkappa_T/dT) = d/dT(0 + dkappaT_dastrain . \sqrt{2/3} . dDgamma_dT)
      //               = dT_dkappaT_dastrain . \sqrt{2/3} . dDgamma_dT)

      // implicit thermal derivatives d/dT(dkappa_dastrain . dastrain/dDgamma . dDgamma/dT)
      // d/dT(dkappa_dastrain . dastrain/dDgamma . dDgamma/dT)
      // = dT_dkappa_dastrain . sqrt(2/3) * dDgamma_dT + dkappa_dastrain . dastrain/dDgamma .
      // dT_dDgamma/dT

      // d/dT(dkappa/dT) = 2 . ddkappa_dastraindT . dastrain/dDgamma . dDgamma/dT
      double ddkappa_dTT = sqrt(2.0 / 3.0) * dDgamma_dT * 2.0 *
                           (Hiso * (-omega_h) + (sigma_y0infty * (-omega_h) + sigma_y0 * omega_0) *
                                                    (-exp(-hardexpo * alpha)) * (-hardexpo));
      // TODO in case of bad convergence: add dkappa_dastrain . dastrain/dDgamma . dT_dDgamma/dT

      double dkappa_dTdastrain = Hiso * (-omega_h) +
                                 (sigma_y0infty * (-omega_h) + sigma_y0 * omega_0) *
                                     (-exp(-hardexpo * alpha)) * (-hardexpo) +
                                 sqrt(2.0 / 3.0) * dDgamma_dT *
                                     (sigma_y0infty_temp - sigma_y0_temp) *
                                     (-exp(-hardexpo * alpha)) * hardexpo * hardexpo;

      // ------------------------------ calculate derivative of Dgamma w.r.t. E

      // spatial description:
      // 2 . dDgamma/dg = 1/beta0 . [ (1 - 2/3 || s || Dgamma / mubar ) . n
      //                          + || s || / mubar . dev[n^2] ]
      // material description:
      // dDgamma/dE = F^{-1} . (2 . dDgamma/dg) . F^{-T}
      //            = 1/beta0 . [ (1 - 2/3 . || s || . Dgamma / mubar ) . N
      //                          + || s || / mubar . dev[N^2] ]
      Core::LinAlg::Matrix<3, 3> dDgamma_dg(Core::LinAlg::Initialization::uninitialized);
      dDgamma_dg.update((1.0 - 2.0 / 3.0 * q_trial * Dgamma / mubar), N);
      dDgamma_dg.update((q_trial / mubar), devNsquare, 1.0);
      dDgamma_dg.scale(1.0 / beta0);

      // -------------------------------------- internal/mechanical dissipation

      // --------------------------------------------------------- D_mech
      // D_mech = tau . [-1/2  Lie(b_e)] . b_e^{-1} - kappa . alpha'
      //        = sqrt(2/3) . sigma_y0(T_{n+1}) . Dgamma/Dt
      // be aware: multiplication with Dt is done in thermo element
      double mechdiss = sqrt(2.0 / 3.0) * Dgamma * sigma_y0_temp;
      mechdiss_->at(gp) = mechdiss;

      // -------------------------------------------- dD_mech/dT for k_TT

      // k_TT += ... dD_mech/dT_{n+1} ...
      // with dD_mech/dT_{n+1} = 1/Dt . sqrt(2/3) . [ dDgamma/dT_{n+1} . sigma_y0_temp
      //                         + Dgamma . dsigma_y0_temp/dT_{n+1} ]

      // with sigma_y0_temp = sigma_y0 . (1.0 - omega_0 . (scalartemp - inittemp) )
      // calculate the derivative of sigma_y0(T_{n+1}) w.r.t. T_{n+1}
      // derivative of mechanical Dissipation w.r.t. temperatures
      mechdiss_k_tt_->at(gp) =
          sqrt(2.0 / 3.0) * (dDgamma_dT * sigma_y0_temp + Dgamma * dsigma_y0_temp_dT);

      // -------------------------------------------- dD_mech/dd for k_Td
      // k_Td += dD_mech/dd_{n+1}
      //      += 1/Dt . sqrt(2/3) . [ sigma_y0_temp . dDgamma/dE ]
      Core::LinAlg::Matrix<3, 3> mechdiss_kTd_matrix(Core::LinAlg::Initialization::uninitialized);
      mechdiss_kTd_matrix.update((sqrt(2.0 / 3.0) * sigma_y0_temp), dDgamma_dg);
      // Voigt notation
      Core::LinAlg::Matrix<6, 1> mechdiss_kTd_vct(Core::LinAlg::Initialization::uninitialized);
      mechdiss_kTd_vct(0) = mechdiss_kTd_matrix(0, 0);
      mechdiss_kTd_vct(1) = mechdiss_kTd_matrix(1, 1);
      mechdiss_kTd_vct(2) = mechdiss_kTd_matrix(2, 2);
      mechdiss_kTd_vct(3) = 0.5 * (mechdiss_kTd_matrix(0, 1) + mechdiss_kTd_matrix(1, 0));
      mechdiss_kTd_vct(4) = 0.5 * (mechdiss_kTd_matrix(1, 2) + mechdiss_kTd_matrix(2, 1));
      mechdiss_kTd_vct(5) = 0.5 * (mechdiss_kTd_matrix(0, 2) + mechdiss_kTd_matrix(2, 0));
      mechdiss_k_td_->at(gp).update(mechdiss_kTd_vct);

      // ------------------------------------------- thermoplastic heating term

      // ------------------------------------------------------------ H_p

      // H_p = T . dkappa/dT . astrain'
      // with astrain' = sqrt(2/3) . Dgamma/Dt
      // thrplheat_ = dkappa/dT . sqrt(2/3) . Dgamma
      thrplheat_->at(gp) = dkappa_dT * sqrt(2.0 / 3.0) * Dgamma;
      // H_p = T . thrplheat_ . 1/Dt
      // be aware: multiplication with 1/Dt and (T = N_T . T) is done in thermo
      //           element

      // ----------------------------------------------- dH_p/dT for k_TT

      // k_TT += dH_p/dT_{n+1}
      //       = dkappa/dT . astrain^p'
      //         + 1/Dt . T . [ d/dT(dkappa/dT) . astrain^p'
      //                        + dkappa/dT . dastrain^p'/dT ]
      //
      //       = [ thrplheat_ . 1/Dt + T . thrplheat_kTT_ . 1/Dt ] . N_T
      // be aware: multiplication with 1/Dt and T is done in thermo element
      thrplheat_k_tt_->at(gp) =
          ddkappa_dTT * sqrt(2.0 / 3.0) * Dgamma + dkappa_dT * sqrt(2.0 / 3.0) * dDgamma_dT;

      // ----------------------------------------------- dH_p/dd for k_Td
      // k_Td += ... dH_p/dE . dE/dd ...
      //
      // dH_p/dE = 1/Dt . [ ddkappa/(dT dE) . sqrt{2/3} . dDgamma + dkappa/dT. sqrt{2/3} .
      // dDgamma/dE
      //
      // with
      // ddkappa/(dT dE) = ddkappa/(dT dastrain) . dastrain/dDgamma . dDgamma/dE
      //                 = dkappa_dTdastrain . \sqrt(2/3) . dDgamma/dE
      //
      // dH_p/dE = [ dkappa_dTdastrain . (2/3) . dDgamma + dkappa/dT. sqrt{2/3} ]
      //           . 1/Dt . dDgamma/dE
      // be aware: multiplication with 1/Dt and dE/dd is done in thermo element
      double fac_thrpl_kTd = dkappa_dTdastrain * (2.0 / 3.0) * Dgamma + dkappa_dT * sqrt(2.0 / 3.0);
      Core::LinAlg::Matrix<3, 3> thrplheat_kTd_matrix(Core::LinAlg::Initialization::uninitialized);
      thrplheat_kTd_matrix.update(fac_thrpl_kTd, dDgamma_dg);
      // Voigt notation
      Core::LinAlg::Matrix<6, 1> thrplheat_kTd_vct(Core::LinAlg::Initialization::uninitialized);
      thrplheat_kTd_vct(0) = thrplheat_kTd_matrix(0, 0);
      thrplheat_kTd_vct(1) = thrplheat_kTd_matrix(1, 1);
      thrplheat_kTd_vct(2) = thrplheat_kTd_matrix(2, 2);
      thrplheat_kTd_vct(3) = 0.5 * (thrplheat_kTd_matrix(0, 1) + thrplheat_kTd_matrix(1, 0));
      thrplheat_kTd_vct(4) = 0.5 * (thrplheat_kTd_matrix(1, 2) + thrplheat_kTd_matrix(2, 1));
      thrplheat_kTd_vct(5) = 0.5 * (thrplheat_kTd_matrix(0, 2) + thrplheat_kTd_matrix(2, 0));
      thrplheat_k_td_->at(gp).update(thrplheat_kTd_vct);

      //------ linearisation of mechanical material tangent w.r.t. temperatures

      // ---------------------------------------------- dCmat_dT for k_dT

      // dCmat_dT += F^{-1} . ds_{n+1}/dT_{n+1} . F^{-T}
      // with ds_{n+1}/dT_{n+1} = - 2 . mubar . dDgamma/dT . n
      //                        = + sqrt(2/3) . dsigma_y_dT . 1/beta0 . N := beta5 . N
      Core::LinAlg::Matrix<3, 3> Cmat_kdT_matrix(Core::LinAlg::Initialization::uninitialized);
      Cmat_kdT_matrix.update((-2.0 * mubar * dDgamma_dT), N);
      // Voigt notation
      Core::LinAlg::SymmetricTensor<double, 3, 3> Cmat_kdT_vct;
      Cmat_kdT_vct(0, 0) = Cmat_kdT_matrix(0, 0);
      Cmat_kdT_vct(1, 1) = Cmat_kdT_matrix(1, 1);
      Cmat_kdT_vct(2, 2) = Cmat_kdT_matrix(2, 2);
      Cmat_kdT_vct(0, 1) = 0.5 * (Cmat_kdT_matrix(0, 1) + Cmat_kdT_matrix(1, 0));
      Cmat_kdT_vct(1, 2) = 0.5 * (Cmat_kdT_matrix(1, 2) + Cmat_kdT_matrix(2, 1));
      Cmat_kdT_vct(0, 2) = 0.5 * (Cmat_kdT_matrix(0, 2) + Cmat_kdT_matrix(2, 0));
      cmat_kd_t_->at(gp) = Cmat_kdT_vct;

    }  // (Dgamma != 0.0)


  }  // end plastic step

  // ------------------------------------------------------ update final stress
  // add mean stress to gain Kirchhoff stress tau (9.2.6)
  // tau = J . p . I + devtau
  // with p := U'(J) = 1/2 . bulk . (J^2 -1 ) / J
  // --> tau = 1/2 . bulk ( (J^2 -1 ) . I + devtau
  // different to Miehe (2.37): p = bulk/2 (J^2 - 1) / J
  double p = bulk / 2.0 * (J * J - 1.0) / J;
  Core::LinAlg::Matrix<3, 3> tau(devtau);
  for (int i = 0; i < 3; i++) tau(i, i) += J * p;

  // transform Kirchhoff stress to 2.PK-stress
  // PK2 = F^{-1} . tau . F^{-T}
  Core::LinAlg::Matrix<3, 3> PK2;
  tmp.put_scalar(0.0);  // reuse tmp, but reset first
  tmp.multiply(invdefgrdcurr, tau);
  PK2.multiply_nt(tmp, invdefgrdcurr);

  // output PK2-stress in Voigt-notation
  stress(0, 0) = PK2(0, 0);
  stress(1, 1) = PK2(1, 1);
  stress(2, 2) = PK2(2, 2);
  stress(0, 1) = 0.5 * (PK2(0, 1) + PK2(1, 0));
  stress(1, 2) = 0.5 * (PK2(1, 2) + PK2(2, 1));
  stress(0, 2) = 0.5 * (PK2(0, 2) + PK2(2, 0));

  // ----------------------- consistent elastoplastic tangent modulus (Box 9.2)
  Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmat_view =
      Core::LinAlg::make_stress_like_voigt_view(cmat);
  setup_cmat_elasto_plastic(cmat_view,  // (o) elasto-plastic tangent modulus
      Dgamma,                           // plastic multiplier
      Hiso_temp,                        // H(T)
      sigma_y0infty_temp,               // trial value of saturation yield stress
      sigma_y0_temp,                    // trial value of initial yield stress
      mubar, q_trial,
      defgrd_mat,     // F
      invdefgrdcurr,  // F^{-1}
      n,              // spatial flow vector
      bulk,           // isotropic thermodynamic force
      gp              // current Gauss point
  );

  // -------------------------------- add the temperature dependent stress part
  // inverse of right Cauchy-Green tensor = F^{-1} . F^{-T}
  Core::LinAlg::SymmetricTensor<double, 3, 3> cauchygreen =
      Core::LinAlg::assume_symmetry(Core::LinAlg::transpose(*defgrad) * *defgrad);
  Core::LinAlg::SymmetricTensor<double, 3, 3> Cinv = Core::LinAlg::inv(cauchygreen);

  // Delta T = T - T_0
  Core::LinAlg::Matrix<1, 1> deltaT(Core::LinAlg::Initialization::uninitialized);
  deltaT(0, 0) = scalartemp - params_->inittemp_;

  // get the temperature-dependent material tangent
  Core::LinAlg::SymmetricTensor<double, 3, 3> ctemp;
  ;
  setup_cthermo(ctemp, defgrd_mat.determinant(), Cinv);

  // calculate thermal stresses
  // tau = ctemp_AK . Delta T = m_0/2.0 . (J + 1/J) . I . Delta T
  // pull-back of Kirchhoff-stresses to PK2-stresses
  // PK2 = F^{-1} . tau . F^{-T}
  // --> PK2 = ctemp . Delta T = m_0/2.0 . (J + 1/J). Cinv . Delta T
  Core::LinAlg::SymmetricTensor<double, 3, 3> stresstemp = ctemp * deltaT(0, 0);

  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> stress_view =
      Core::LinAlg::make_stress_like_voigt_view(stress);
  stress += stresstemp;

}  // evaluate()


/*----------------------------------------------------------------------*
 | Calculation of consistent elastoplastic tangent modulus   dano 09/13 |
 *----------------------------------------------------------------------*/
void Mat::ThermoPlasticHyperElast::setup_cmat_elasto_plastic(
    Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
        cmat,  // elasto-plastic tangent modulus (out)
    double Dgamma, double Hiso_temp, double sigma_y0infty_temp, double sigma_y0_temp, double mubar,
    double q_trial,                            // || s_{n+1}^{trial} ||
    const Core::LinAlg::Matrix<3, 3>& defgrd,  // F
    Core::LinAlg::Matrix<3, 3> invdefgrdcurr, Core::LinAlg::Matrix<3, 3> n,
    double bulk,  // bulk modulus
    int gp        // current Gauss point
) const
{
  // ---------------------------------------------- initialise material tangents
  Core::LinAlg::Matrix<6, 6> Cmat(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6, 6> Cbar_trialMaterial(Core::LinAlg::Initialization::zero);

  // Cmat = C_ep = C_e + Cbar_trial + Cbar_p

  // Cbar_ep = (1 - beta1) . Cbar_{n+1}^{trial} - 2 mubar beta3 n \otimes n
  //           - 2 mubar beta4 sym[ n \otimes dev[n^2] ]^s
  // with Cbar_{n+1}^{trial} = 2 mubar I_d
  //              - 2/3 (I \otimes s_{n+1}^{trial} + s_{n+1}^{trial} \otimes I)

  // ---------------------------------------------------------- calculate terms
  // initialise some variables
  Core::LinAlg::Matrix<3, 3> tmp(Core::LinAlg::Initialization::uninitialized);
  // hardening exponent
  double hardexpo = params_->hardexpo_;
  // determinant of the deformation gradient
  double J = defgrd.determinant();

  // calculate the right Cauchy Green (RCG) deformation tensor and its inverse
  Core::LinAlg::Matrix<3, 3> RCG(Core::LinAlg::Initialization::uninitialized);
  RCG.multiply_tn(defgrd, defgrd);
  Core::LinAlg::Matrix<3, 3> invRCG;
  invRCG.invert(RCG);

  // --------------------------------------- calculate plastic directions
  // pull-back of the spatial flow vector n to N
  // N = F^{-1} n F^{-T}
  Core::LinAlg::Matrix<3, 3> N(Core::LinAlg::Initialization::uninitialized);
  tmp.multiply(invdefgrdcurr, n);
  N.multiply_nt(tmp, invdefgrdcurr);

  // dev (n^2)
  Core::LinAlg::Matrix<3, 3> devnsquare(Core::LinAlg::Initialization::uninitialized);
  devnsquare.multiply(n, n);
  double tracensquare = (devnsquare(0, 0) + devnsquare(1, 1) + devnsquare(2, 2));
  for (int i = 0; i < 3; i++) devnsquare(i, i) -= 1.0 / 3.0 * tracensquare;

  // pull-back of dev(n^2)
  Core::LinAlg::Matrix<3, 3> devNsquare(Core::LinAlg::Initialization::uninitialized);
  tmp.put_scalar(0.0);  // reuse tmp, but reset first
  tmp.multiply(invdefgrdcurr, devnsquare);
  devNsquare.multiply_nt(tmp, invdefgrdcurr);

  // ----------------------------------------------------------- calculate Cmat
  // ------------------------------------------------ isochoric part Cbar
  // Cbar
  // spatial: cbar_trial = 2 . mubar . I_d - 2/3 qbar [n \otimes I + I \otimes n]
  // with I_d = I_s - 1/3 . I \otimes I
  // pull-back of I --> invRCG
  // Cbar += Cbar_trial = 2 . mubar . pullback_I_d
  Core::LinAlg::FourTensorOperations::add_kronecker_tensor_product(
      Cbar_trialMaterial, 2.0 * mubar, invRCG, invRCG, 1.0);
  Core::LinAlg::FourTensorOperations::add_elasticity_tensor_product(
      Cbar_trialMaterial, -2.0 / 3.0 * mubar, invRCG, invRCG, 1.0);
  // Cbar += - 2/3 qbar [N \otimes C^{-1} + C^{-1} \otimes N]
  Core::LinAlg::FourTensorOperations::add_symmetric_elasticity_tensor_product(
      Cbar_trialMaterial, -2.0 / 3.0 * q_trial, N, invRCG, 1.0);

  // ------------------------------------------------ volumetric part C_e
  // spatial c_e = (J . U')' . J . I \otimes I - 2 J U' I4
  // with U'(J) = bulk/2 . (J^2 -1)  / J
  // C_e = bulk . J^2 [C^{-1} \otimes C^{-1}] - bulk ( J^2 -1 ) [C^{-1} \otimes C^{-1}]
  // with - bulk ( J^2 -1 ) [C^{-1} \otimes C^{-1}] = - bulk ( J^2 -1 ) [C^{-1} boeppel C^{-1}]
  Core::LinAlg::FourTensorOperations::add_elasticity_tensor_product(
      Cmat, bulk * J * J, invRCG, invRCG, 1.0);
  Core::LinAlg::FourTensorOperations::add_kronecker_tensor_product(
      Cmat, -1.0 * bulk * (J * J - 1.0), invRCG, invRCG, 1.0);
  Cmat.update(1.0, Cbar_trialMaterial, 1.0);

  // plastic step update
  if (Dgamma != 0.0)
  {
    // ------------------------------------ scaling factors for spatial tangent

    // beta_0 = 1 + 1/(3 mubar) . dkappa_{n+1}/dastrain_{n+1}
    // with dkappa_{n+1}/dastrain_{n+1} = Hiso_temp + (sigma_y0infty_temp - sigma_y0_temp)
    //                                    . [-exp(-delta . astrain)] . (-delta)
    double beta0 = 0.0;
    beta0 = 1.0 + (Hiso_temp + (sigma_y0infty_temp - sigma_y0_temp) *
                                   exp(-hardexpo * accplstraincurr_->at(gp)) * hardexpo) /
                      (3.0 * mubar);

    // beta_1 = 2 . mubar . Dgamma / || s_{n+1}^{trial} ||
    double beta1 = 0.0;
    beta1 = 2.0 * mubar * Dgamma / q_trial;

    // beta_2 = (1 - 1/beta_0) . 2/3 . Dgamma / mubar . || s_{n+1}^{trial} ||
    double beta2 = 0.0;
    beta2 = (1.0 - 1.0 / beta0) * 2.0 / 3.0 * Dgamma / mubar * q_trial;

    // beta_3 = 1/beta_0 - beta_1 + beta_2
    double beta3 = 0.0;
    beta3 = 1.0 / beta0 - beta1 + beta2;

    // beta_4 = (1/beta_0 - beta_1) . || s_{n+1}^{trial} || / mubar
    double beta4 = 0.0;
    beta4 = (1.0 / beta0 - beta1) * q_trial / mubar;

    // this is nonlinear mechanics
    Cmat.update((-1.0 * beta1), Cbar_trialMaterial, 1.0);
    Core::LinAlg::FourTensorOperations::add_elasticity_tensor_product(
        Cmat, (-2.0 * mubar * beta3), N, N, 1.0);
    Core::LinAlg::FourTensorOperations::add_elasticity_tensor_product(
        Cmat, (-2.0 * mubar * beta4), N, devNsquare, 1.0);
  }  // Dgamma != 0

  // update material tangent
  // cmat = C_ep = C_e + Cbar_trial + Cbar_p
  cmat = Cmat;

}  // setup_cmat_elasto_plastic()


/*----------------------------------------------------------------------*
 | calculate final isochoric elastic LCG bbar^e_{n+1}        dano 09/13 |
 *----------------------------------------------------------------------*/
void Mat::ThermoPlasticHyperElast::calculate_current_bebar(
    const Core::LinAlg::Matrix<3, 3>& devtau,  // s_{n+1}
    double G,                                  // shear modulus
    const Core::LinAlg::Matrix<3, 3>& id2,     // second-order identity
    int gp                                     // current Gauss-point
)
{
  // calculate equivalent von Mises stress || s_{n+1} ||
  double q = 0.0;
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      q += devtau(i, j) * devtau(i, j);
    }
  }

  // ------------------------------------------- calculate isochoric invariants
  // calculate isochoric second invariant of dev[bebar] J2_bar
  double J2_bar = 0.0;
  J2_bar = 1.0 / 2.0 * q * q / (G * G);

  // calculate isochoric third invariant of dev[bebar] (i.e. determinant) J3_bar
  double J3_bar = 0.0;
  J3_bar = 1.0 / (G * G * G) * devtau.determinant();

  // --------------------------------------------------- calculate coefficients
  // coefficient q
  // q = (1 - J3_bar) / 2
  double q_coeff = 0.0;
  q_coeff = 1.0 / 2.0 * (1.0 - J3_bar);

  // coefficient d
  // d = - J2_bar^3 / 27 + q^2
  double d_coeff = 0.0;
  d_coeff = -std::pow(J2_bar, 3.0) / 27.0 + std::pow(q_coeff, 2.0);

  // -------------------------------------------------- calculate updated bebar

  // calculate scaling factor for updated bebar
  // 1/3 tr(bebar) = 1/3 . Ibar_1
  double const third_Ibar_1 = std::pow((q_coeff + sqrt(d_coeff)), 1.0 / 3.0) +
                              std::pow((q_coeff - sqrt(d_coeff)), 1.0 / 3.0);

  // bebar_{n+1} = s_{n+1}/mu + 1/3 Ibar_1 . id2
  bebarcurr_->at(gp) = (devtau);
  bebarcurr_->at(gp).scale(1 / G);
  bebarcurr_->at(gp).update(third_Ibar_1, id2, 1.0);

}  // calculate_current_bebar()


/*----------------------------------------------------------------------*
 | computes temperature-dependent isotropic                  dano 09/13 |
 | elasticity tensor in matrix notion for 3d, second(!) order tensor    |
 *----------------------------------------------------------------------*/
void Mat::ThermoPlasticHyperElast::setup_cmat_thermo(const double temperature,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat_T,
    const Core::LinAlg::Tensor<double, 3, 3>& defgrd) const
{
  // temperature-dependent material tangent
  // cmat_T = cmat_vol,dT = dstresstemp/dE = 2 dstresstemp/dC
  //        = (T - T_0) . m_0/2.0 . (J - 1/J) (C^{-1} \otimes C^{-1}) -
  //          - (T - T_0) . m_0 . (J + 1/J) ( Cinv boeppel Cinv )

  // calculate the temperature difference
  // Delta T = T - T_0
  const double deltaT = temperature - (params_->inittemp_);

  // get stress-temperature modulus
  double m_0 = st_modulus();
  // get Jacobi
  double J = Core::LinAlg::det(defgrd);
  // calculate the right Cauchy Green (RCG) deformation tensor and its inverse
  Core::LinAlg::SymmetricTensor<double, 3, 3> RCG =
      Core::LinAlg::assume_symmetry(Core::LinAlg::transpose(defgrd) * defgrd);
  Core::LinAlg::SymmetricTensor<double, 3, 3> invRCG = Core::LinAlg::inv(RCG);

  // clear the material tangent
  Core::LinAlg::Matrix<6, 6> cmat_view = Core::LinAlg::make_stress_like_voigt_view(cmat_T);
  const Core::LinAlg::Matrix<3, 3> invRCG_mat =
      Core::LinAlg::make_matrix(Core::LinAlg::get_full(invRCG));
  cmat_view.clear();

  // cmat_T = 2 . dS_vol,dT/dd
  //        = (T - T_0) . m_0/2 . (J - 1/J) (C^{-1} \otimes C^{-1})
  //          - (T - T_0) . m_0 . (J + 1/J) ( Cinv boeppel Cinv )
  Core::LinAlg::FourTensorOperations::add_elasticity_tensor_product(
      cmat_view, (deltaT * m_0 / 2.0 * (J - 1 / J)), invRCG_mat, invRCG_mat, 1.0);
  Core::LinAlg::FourTensorOperations::add_kronecker_tensor_product(
      cmat_view, (-deltaT * m_0 * (J + 1 / J)), invRCG_mat, invRCG_mat, 1.0);


}  // SetupCmatThermo()


/*----------------------------------------------------------------------*
 | computes temperature-dependent isotropic                  dano 09/13 |
 | elasticity tensor in matrix notion for 3d, second(!) order tensor    |
 *----------------------------------------------------------------------*/
void Mat::ThermoPlasticHyperElast::setup_cthermo(Core::LinAlg::SymmetricTensor<double, 3, 3>& ctemp,
    const double J, const Core::LinAlg::SymmetricTensor<double, 3, 3>& Cinv_vct) const
{
  // temperature-dependent material tangent
  // C_T = m_0/2.0 . (J + 1/J) . Cinv

  // temperature-dependent stress temperature modulus
  // m = m(J) = m_0 .(J+1)/J = m_0 . (J + 1/J)
  const double m_0 = st_modulus();
  const double m = m_0 * (J + 1.0 / J);
  ctemp = m / 2.0 * Cinv_vct;

}  // setup_cthermo()


/*----------------------------------------------------------------------*
 | calculates stress-temperature modulus m_0                 dano 09/13 |
 *----------------------------------------------------------------------*/
double Mat::ThermoPlasticHyperElast::st_modulus() const
{
  // m_0 := -(2 . mu + 3 . lambda) . alpha_T = - 3 . bulk . alpha_T

  // initialise the parameters for the lame constants
  const double ym = params_->youngs_;                 // Young's modulus
  const double nu = params_->poissonratio_;           // Poisson's ratio
  const double bulk = ym / (3.0 * (1.0 - 2.0 * nu));  // bulk modulus
  const double cte = params_->cte_;

  // stress-temperature modulus
  const double stmodulus = (-1.0) * 3.0 * bulk * cte;

  return stmodulus;

}  // st_modulus()


/*---------------------------------------------------------------------*
 | return names of visualization data (public)                         |
 *---------------------------------------------------------------------*/
void Mat::ThermoPlasticHyperElast::vis_names(std::map<std::string, int>& names) const
{
  std::string accumulatedstrain = "accumulatedstrain";
  names[accumulatedstrain] = 1;  // scalar

  std::string mechdiss = "mechdiss";
  names[mechdiss] = 1;  // scalar

  std::string thrplheating = "thrplheating";
  names[thrplheating] = 1;  // scalar
}  // vis_names()


/*---------------------------------------------------------------------*
 | return visualization data (public)                                  |
 *---------------------------------------------------------------------*/
bool Mat::ThermoPlasticHyperElast::vis_data(
    const std::string& name, std::vector<double>& data, int numgp, int eleID) const
{
  // accumulated strain
  if (name == "accumulatedstrain")
  {
    if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");
    double temp = 0.0;
    for (int iter = 0; iter < numgp; iter++) temp += accumulated_strain(iter);
    data[0] = temp / numgp;
  }

  // mechanical dissipation
  if (name == "mechdiss")
  {
    if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");
    double temp = 0.0;
    for (int iter = 0; iter < numgp; iter++) temp += mech_diss(iter);
    data[0] = temp / numgp;
  }

  // thermoplastic heating term
  if (name == "thrplheating")
  {
    if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");
    double temp = 0.0;
    for (int iter = 0; iter < numgp; iter++) temp += thermo_plast_heating(iter);
    data[0] = temp / numgp;
  }

  return true;
}  // vis_data()


/*---------------------------------------------------------------------*
 | finite difference check for the material tangent.        dano 12/13 |
 | Meant for debugging only! (public)                                  |
 *---------------------------------------------------------------------*/
void Mat::ThermoPlasticHyperElast::fd_check(
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& stress,  // updated stress sigma_n+1
    Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
        cmat,  // material tangent calculated with FD of stresses
    Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
        cmatFD,  // material tangent calculated with FD of stresses
    const double temperature, const Teuchos::ParameterList& params) const
{
  // teste 2dS/dC see linearisation according to Holzapfel
  // extract F and Cinv from params
  Core::LinAlg::Matrix<3, 3> defgrd = params.get<Core::LinAlg::Matrix<3, 3>>("defgrd");
  Core::LinAlg::Matrix<6, 1> Cinv_vct = params.get<Core::LinAlg::Matrix<6, 1>>("Cinv_vct");

  // calculate the right Cauchy Green (RCG) deformation tensor and its inverse
  Core::LinAlg::Matrix<3, 3> RCG_disturb(Core::LinAlg::Initialization::uninitialized);
  RCG_disturb.multiply_tn(defgrd, defgrd);

  // value of disturbance
  const double delta = 1.0e-8;
  // disturb the respective strain quantities
  for (int i = 0; i < 3; ++i)
  {
    for (int k = 0; k < 3; ++k)
    {
      printf("-------------------------------------\n");
      printf("-------------------------------------\n");
      printf("STRAIN term %d\n", k);

      RCG_disturb(i, k) += delta / 2.0;
      RCG_disturb(k, i) += delta / 2.0;

      // calculate Jacobi-determinant of disturbed RCG
      double detRCG_disturb = RCG_disturb.determinant();
      double J_disturb = sqrt(detRCG_disturb);
      Core::LinAlg::Matrix<3, 3> invRCG_disturb;
      invRCG_disturb.invert(RCG_disturb);
      // use vector-notation
      Core::LinAlg::Matrix<6, 1> disturb_Cinv_vct(Core::LinAlg::Initialization::uninitialized);
      disturb_Cinv_vct(0) = invRCG_disturb(0, 0);
      disturb_Cinv_vct(1) = invRCG_disturb(1, 1);
      disturb_Cinv_vct(2) = invRCG_disturb(2, 2);
      disturb_Cinv_vct(3) = invRCG_disturb(0, 1);
      disturb_Cinv_vct(4) = invRCG_disturb(1, 2);
      disturb_Cinv_vct(5) = invRCG_disturb(2, 0);

      // calculate the temperature difference
      // Delta T = T - T_0
      Core::LinAlg::Matrix<1, 1> deltaT(Core::LinAlg::Initialization::uninitialized);
      deltaT(0, 0) = temperature - params_->inittemp_;

      // temperature-dependent stress temperature modulus
      // m = m(J) = m_0 .(J+1)/J = m_0 . (J + 1/J)
      double m_0 = st_modulus();
      double m = m_0 * (J_disturb + 1.0 / J_disturb);
      // in case of testing only the dJ/dd, use undisturbed m, but disturb_Cinv_vct
      // double J = defgrd.Determinant();
      // double m = m_0 * (J + 1.0 / J);
      // clear the material tangent
      Core::LinAlg::Matrix<NUM_STRESS_3D, 1> disturb_ctemp(Core::LinAlg::Initialization::zero);
      disturb_ctemp.clear();
      // C_T = m_0/2.0 . (J + 1/J) . Cinv
      disturb_ctemp.update((m / 2.0), disturb_Cinv_vct);
      // in case of testing only factor, use undisturbed Cinv_vct
      // disturb_ctemp.update( (m/2.0), Cinv_vct, 0.0);

      // calculate thermal stresses
      // tau = ctemp_AK . Delta T = m_0/2 . (J + 1/J) . I . Delta T
      // pull-back of Kirchhoff-stresses to PK2-stresses
      // PK2 = F^{-1} . tau . F^{-T}
      // --> PK2 = ctemp . Delta T = m_0/2 . (J + 1/J). Cinv . Delta T
      // initialise disturbed total stresses
      Core::LinAlg::Matrix<NUM_STRESS_3D, 1> disturb_stresstemp(Core::LinAlg::Initialization::zero);
      disturb_stresstemp.multiply_nn(disturb_ctemp, deltaT);
      // in case of testing only disturb_ctemp, ignore deltaT
      // disturb_stresstemp.update(disturb_ctemp);


      // be careful we save the disturbed RCG in tensor notation, i.e. (3x3)
      // to insert the corresponding terms in cmat (6x6) copy the terms to
      // their correct position using array VOIGT3X3SYM as is done, e.g. in
      // neohooke or so3_plast
      double array[3][3] = {{0, 3, 5}, {3, 1, 4}, {5, 4, 2}};

      for (int stress_comp = 0; stress_comp < 6; stress_comp++)
      {
        // build the finite difference tangent
        cmatFD(stress_comp, array[i][k]) = 0.0;
        // scale with factor 2 due to comparison with cmat_T and cmat_T = 2 dSvol/dC
        cmatFD(stress_comp, array[i][k]) +=
            2.0 * ((disturb_stresstemp(stress_comp) / (delta)-stress(stress_comp) / (delta)));

        std::cout << i << k << stress_comp << "fd: "
                  << 2.0 *
                         ((disturb_stresstemp(stress_comp) / (delta)-stress(stress_comp) / (delta)))
                  << "ref: " << cmat(stress_comp, array[i][k]) << std::endl;
      }
      // undisturb the respective strain quantities (disturbstrain=strain)
      RCG_disturb(i, k) -= delta / 2.0;
      RCG_disturb(k, i) -= delta / 2.0;
    }  // loop stresses

  }  // loop strains

}  // fd_check()


/*----------------------------------------------------------------------*/

void Mat::ThermoPlasticHyperElast::evaluate(const Core::LinAlg::Matrix<3, 1>& gradtemp,
    Core::LinAlg::Matrix<3, 3>& cmat, Core::LinAlg::Matrix<3, 1>& heatflux, const int eleGID) const
{
  thermo_->evaluate(gradtemp, cmat, heatflux, eleGID);
}

void Mat::ThermoPlasticHyperElast::evaluate(const Core::LinAlg::Matrix<2, 1>& gradtemp,
    Core::LinAlg::Matrix<2, 2>& cmat, Core::LinAlg::Matrix<2, 1>& heatflux, const int eleGID) const
{
  thermo_->evaluate(gradtemp, cmat, heatflux, eleGID);
}

void Mat::ThermoPlasticHyperElast::evaluate(const Core::LinAlg::Matrix<1, 1>& gradtemp,
    Core::LinAlg::Matrix<1, 1>& cmat, Core::LinAlg::Matrix<1, 1>& heatflux, const int eleGID) const
{
  thermo_->evaluate(gradtemp, cmat, heatflux, eleGID);
}

std::vector<double> Mat::ThermoPlasticHyperElast::conductivity(int eleGID) const
{
  return thermo_->conductivity(eleGID);
}


void Mat::ThermoPlasticHyperElast::conductivity_deriv_t(Core::LinAlg::Matrix<3, 3>& dCondDT) const
{
  thermo_->conductivity_deriv_t(dCondDT);
}

void Mat::ThermoPlasticHyperElast::conductivity_deriv_t(Core::LinAlg::Matrix<2, 2>& dCondDT) const
{
  thermo_->conductivity_deriv_t(dCondDT);
}

void Mat::ThermoPlasticHyperElast::conductivity_deriv_t(Core::LinAlg::Matrix<1, 1>& dCondDT) const
{
  thermo_->conductivity_deriv_t(dCondDT);
}

double Mat::ThermoPlasticHyperElast::capacity() const { return thermo_->capacity(); }

double Mat::ThermoPlasticHyperElast::capacity_deriv_t() const
{
  return thermo_->capacity_deriv_t();
}

void Mat::ThermoPlasticHyperElast::reinit(double temperature, unsigned gp)
{
  current_temperature_ = temperature;
  if (thermo_ != nullptr) thermo_->reinit(temperature, gp);
}
void Mat::ThermoPlasticHyperElast::reset_current_state()
{
  if (thermo_ != nullptr) thermo_->reset_current_state();
}

void Mat::ThermoPlasticHyperElast::commit_current_state()
{
  if (thermo_ != nullptr) thermo_->commit_current_state();
}

/*----------------------------------------------------------------------*/

FOUR_C_NAMESPACE_CLOSE
