// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_thermoplasticlinelast.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | constructor (public)                                      dano 08/11 |
 *----------------------------------------------------------------------*/
Mat::PAR::ThermoPlasticLinElast::ThermoPlasticLinElast(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      youngs_(matdata.parameters.get<double>("YOUNG")),
      poissonratio_(matdata.parameters.get<double>("NUE")),
      density_(matdata.parameters.get<double>("DENS")),
      thermexpans_(matdata.parameters.get<double>("THEXPANS")),
      thetainit_(matdata.parameters.get<double>("INITTEMP")),
      yield_(matdata.parameters.get<double>("YIELD")),
      isohard_(matdata.parameters.get<double>("ISOHARD")),
      kinhard_(matdata.parameters.get<double>("KINHARD")),
      sigma_y_((matdata.parameters.get<std::vector<double>>("SIGMA_Y"))),
      strainbar_p_ref_((matdata.parameters.get<std::vector<double>>("EPSBAR_P"))),
      abstol_(matdata.parameters.get<double>("TOL")),
      thermomat_(matdata.parameters.get<int>("THERMOMAT"))
{
}


/*----------------------------------------------------------------------*
 | is called in Material::Factory from read_materials()       dano 08/11 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::ThermoPlasticLinElast::create_material()
{
  return std::make_shared<Mat::ThermoPlasticLinElast>(this);
}


Mat::ThermoPlasticLinElastType Mat::ThermoPlasticLinElastType::instance_;


/*----------------------------------------------------------------------*
 | is called in Material::Factory from read_materials()       dano 08/11 |
 *----------------------------------------------------------------------*/
Core::Communication::ParObject* Mat::ThermoPlasticLinElastType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::ThermoPlasticLinElast* plastic = new Mat::ThermoPlasticLinElast();
  plastic->unpack(buffer);
  return plastic;
}


/*----------------------------------------------------------------------*
 | constructor (public)                                      dano 08/11 |
 *----------------------------------------------------------------------*/
Mat::ThermoPlasticLinElast::ThermoPlasticLinElast() : params_(nullptr), thermo_(nullptr) {}


/*----------------------------------------------------------------------*
 | constructor (public)                                      dano 08/11 |
 | called in read_materials --> create_material                           |
 *----------------------------------------------------------------------*/
Mat::ThermoPlasticLinElast::ThermoPlasticLinElast(Mat::PAR::ThermoPlasticLinElast* params)
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
 | pack (public)                                             dano 08/11 |
 *----------------------------------------------------------------------*/
void Mat::ThermoPlasticLinElast::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  // in case we are in post-process mode
  if (params_ != nullptr) matid = params_->id();
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
    histsize = strainpllast_->size();
  }
  add_to_pack(data, histsize);  // Length of history vector(s)
  for (int var = 0; var < histsize; ++var)
  {
    // insert last converged states
    add_to_pack(data, strainpllast_->at(var));
    add_to_pack(data, backstresslast_->at(var));
    add_to_pack(data, strainbarpllast_->at(var));

    add_to_pack(data, dmech_->at(var));
    add_to_pack(data, dmech_d_->at(var));

    add_to_pack(data, incstrainpl_->at(var));
    add_to_pack(data, strainelrate_->at(var));

    // insert current iteration states
    add_to_pack(data, strainplcurr_->at(var));
    add_to_pack(data, backstresscurr_->at(var));
    add_to_pack(data, strainbarplcurr_->at(var));
  }

  add_to_pack(data, plastic_step_);
}  // pack()


/*----------------------------------------------------------------------*
 | unpack (public)                                           dano 08/11 |
 *----------------------------------------------------------------------*/
void Mat::ThermoPlasticLinElast::unpack(Core::Communication::UnpackBuffer& buffer)
{
  isinit_ = true;


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
        params_ = static_cast<Mat::PAR::ThermoPlasticLinElast*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }

  // history data
  int histsize;
  extract_from_pack(buffer, histsize);

  // if system is not yet initialised, the history vectors have to be initialized
  if (histsize == 0) isinit_ = false;

  // unpack plastic history vectors
  strainpllast_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  strainplcurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();

  // unpack back stress vectors (for kinematic hardening)
  backstresslast_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  backstresscurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();

  strainbarpllast_ = std::make_shared<std::vector<double>>();
  strainbarplcurr_ = std::make_shared<std::vector<double>>();

  // unpack dissipation stuff
  dmech_ = std::make_shared<std::vector<double>>();
  dmech_d_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();

  incstrainpl_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  strainelrate_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();

  for (int var = 0; var < histsize; ++var)
  {
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> tmp_vector(Core::LinAlg::Initialization::zero);
    double tmp_scalar = 0.0;
    // vectors of last converged state are unpacked
    extract_from_pack(buffer, tmp_vector);
    strainpllast_->push_back(tmp_vector);
    extract_from_pack(buffer, tmp_vector);
    backstresslast_->push_back(tmp_vector);
    // scalar-valued vector of last converged state are unpacked
    extract_from_pack(buffer, tmp_scalar);
    strainbarpllast_->push_back(tmp_scalar);
    extract_from_pack(buffer, tmp_scalar);
    dmech_->push_back(tmp_scalar);
    extract_from_pack(buffer, tmp_vector);
    dmech_d_->push_back(tmp_vector);
    extract_from_pack(buffer, tmp_vector);
    incstrainpl_->push_back(tmp_vector);
    extract_from_pack(buffer, tmp_vector);
    strainelrate_->push_back(tmp_vector);

    // current iteration states are unpacked
    extract_from_pack(buffer, tmp_vector);
    strainplcurr_->push_back(tmp_vector);
    extract_from_pack(buffer, tmp_vector);
    backstresscurr_->push_back(tmp_vector);
    extract_from_pack(buffer, tmp_scalar);
    strainbarplcurr_->push_back(tmp_scalar);
  }

  extract_from_pack(buffer, plastic_step_);
}


/*---------------------------------------------------------------------*
 | initialise / allocate internal stress variables (public) dano 08/11 |
 *---------------------------------------------------------------------*/
void Mat::ThermoPlasticLinElast::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  // initialise history variables
  strainpllast_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  strainplcurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();

  backstresslast_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  backstresscurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();

  strainbarpllast_ = std::make_shared<std::vector<double>>();
  strainbarplcurr_ = std::make_shared<std::vector<double>>();

  dmech_ = std::make_shared<std::vector<double>>();
  dmech_d_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();

  incstrainpl_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  strainelrate_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();

  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> emptymat_vect(Core::LinAlg::Initialization::zero);
  strainpllast_->resize(numgp);
  strainplcurr_->resize(numgp);

  backstresslast_->resize(numgp);
  backstresscurr_->resize(numgp);

  strainbarpllast_->resize(numgp);
  strainbarplcurr_->resize(numgp);

  dmech_->resize(numgp);
  dmech_d_->resize(numgp);

  incstrainpl_->resize(numgp);
  strainelrate_->resize(numgp);

  for (int i = 0; i < numgp; i++)
  {
    strainpllast_->at(i) = emptymat_vect;
    strainplcurr_->at(i) = emptymat_vect;

    backstresslast_->at(i) = emptymat_vect;
    backstresscurr_->at(i) = emptymat_vect;

    strainbarpllast_->at(i) = 0.0;
    strainbarplcurr_->at(i) = 0.0;

    dmech_->at(i) = 0.0;
    dmech_d_->at(i) = emptymat_vect;

    incstrainpl_->at(i) = emptymat_vect;
    strainelrate_->at(i) = emptymat_vect;
  }

  isinit_ = true;

  return;

}  // setup()


/*---------------------------------------------------------------------*
 | update internal stress variables (public)                dano 08/11 |
 *---------------------------------------------------------------------*/
void Mat::ThermoPlasticLinElast::update()
{
  // make current values at time step t_n+1 to values of last step t_n
  strainpllast_ = strainplcurr_;
  backstresslast_ = backstresscurr_;

  strainbarpllast_ = strainbarplcurr_;

  // empty vectors of current data
  strainplcurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  backstresscurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();

  strainbarplcurr_ = std::make_shared<std::vector<double>>();

  // get the size of the vector
  // (use the last vector, because it includes latest results, current is empty)
  const int histsize = strainpllast_->size();
  strainplcurr_->resize(histsize);
  backstresscurr_->resize(histsize);

  strainbarplcurr_->resize(histsize);

  const Core::LinAlg::Matrix<NUM_STRESS_3D, 1> emptyvec(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < histsize; i++)
  {
    strainplcurr_->at(i) = emptyvec;
    backstresscurr_->at(i) = emptyvec;

    strainbarplcurr_->at(i) = 0.0;
  }

  return;
}  // update()


/*----------------------------------------------------------------------*
 | evaluate material (public)                                dano 08/11 |
 *----------------------------------------------------------------------*/
void Mat::ThermoPlasticLinElast::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  Core::LinAlg::Matrix<6, 1> stress_view = Core::LinAlg::make_stress_like_voigt_view(stress);
  Core::LinAlg::Matrix<6, 6> cmat_view = Core::LinAlg::make_stress_like_voigt_view(cmat);

  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> plstrain(Core::LinAlg::Initialization::zero);
  if (eleGID == -1) FOUR_C_THROW("no element provided in material");

  // get material parameters
  // Young's modulus
  double young = params_->youngs_;
  // Poisson's ratio
  double nu = params_->poissonratio_;
  // initial yield stress
  double sigma_y0 = params_->yield_;
  // linear isotropic hardening modulus
  double Hiso = params_->isohard_;
  // linear kinematic hardening modulus
  double Hkin = params_->kinhard_;

  // initialise scalars
  // lame constant
  // shear modulus parameter mu == G
  double G = 0.0;
  G = young / (2.0 * (1.0 + nu));
  // bulk modulus kappa = E /( 3 ( 1 - 2 nu) )= lambda + 2/3 * mu
  double kappa = 0.0;
  kappa = young / (3.0 * (1.0 - 2.0 * nu));

  // build Cartesian identity 2-tensor I_{AB}
  Core::LinAlg::Matrix<6, 1> id2(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++) id2(i) = 1.0;

  // glstrain (in): independent variable passed from the element
  //  strain^p: evolution is determined by the flow rule, history variable
  //  strain^e: definition of additive decomposition:
  //  strain^e = strain - strain^p
  // REMARK: stress-like 6-Voigt vector
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> strain =
      Core::LinAlg::make_strain_like_voigt_matrix(glstrain);

  //---------------------------------------------------------------------------
  // elastic predictor (trial values)
  //---------------------------------------------------------------------------

  // ------------------------------------------------- old plastic strain
  // strain^{p,trial}_{n+1} = strain^p_n
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> strain_p(Core::LinAlg::Initialization::uninitialized);
  for (int i = 0; i < 6; i++) strain_p(i, 0) = strainpllast_->at(gp)(i, 0);

  // get old equivalent plastic strain only in case of plastic step
  double strainbar_p = 0.0;
  // accumulated or equivalent plastic strain (scalar-valued)
  // astrain^p,trial}_{n+1} = astrain^p_n
  strainbar_p = (strainbarpllast_->at(gp));
  if (strainbarpllast_->at(gp) < 0.0)
    FOUR_C_THROW("accumulated plastic strain has to be equal to or greater than zero!");

  // ---------------------------------------------------- old back stress
  // beta^{trial}_{n+1} = beta_n
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> beta(Core::LinAlg::Initialization::uninitialized);
  for (int i = 0; i < 6; i++) beta(i, 0) = backstresslast_->at(gp)(i, 0);

  // --------------------------------------------------------- physical strains
  // convert engineering shear components into physical components
  // input strain is given in Voigt-notation

  // convert engineering shear component (in) into physical component
  for (int i = 3; i < 6; ++i) strain(i) /= 2.0;
  for (int i = 3; i < 6; ++i) strain_p(i) /= 2.0;

  // ----------------------------------------------- elastic trial strain
  // assume load step is elastic
  // strain^e_{n+1}
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> strain_e(Core::LinAlg::Initialization::zero);

  // strain^{e,trial}_{n+1} = strain_{n+1} - strain^p_n
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> trialstrain_e(Core::LinAlg::Initialization::uninitialized);
  trialstrain_e.update(1.0, strain, (-1.0), strain_p);

  // volumetric strain
  // trace of strain vector
  double tracestrain = trialstrain_e(0) + trialstrain_e(1) + trialstrain_e(2);
  // volstrain = 1/3 . tr( strain ) . Id
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> volumetricstrain(
      Core::LinAlg::Initialization::uninitialized);
  volumetricstrain.update((tracestrain / 3.0), id2);

  // deviatoric strain
  // devstrain^e = strain^e - volstrain^e
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> devstrain(Core::LinAlg::Initialization::uninitialized);
  devstrain.update(1.0, trialstrain_e, (-1.0), volumetricstrain);

  // ------------------------------------------------------- trial stress
  // pressure = kappa . tr( strain ): saved as scalar
  double p = kappa * tracestrain;

  // deviatoric stress = 2 . G . devstrain
  Core::LinAlg::SymmetricTensor<double, 3, 3> devstress;
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> devstress_view =
      Core::LinAlg::make_stress_like_voigt_view(devstress);
  devstress_view.update((2.0 * G), devstrain);
  // be careful for shear stresses (sigma_12)
  // in Voigt-notation the shear strains have to be scaled with 1/2
  // normally done in the material tangent (cf. id4sharp)

  // ------------------------------------------ relative effective stress
  // eta^{trial}_{n+1} = s^{trial}_{n+1} - beta^{trial}_{n+1}
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> eta(Core::LinAlg::Initialization::zero);
  rel_dev_stress(devstress_view, beta, eta);

  // J2 = 1/2 ( (eta11^{trial})^2 + (eta22^{trial})^2 + (eta33^{trial})^2
  //      + 2 . (eta12^{trial})^2 + 2 . (eta23^{trial})^2 + 2 . (eta13^{trial})^2)
  double J2 = 0.0;
  J2 = 1.0 / 2.0 * (eta(0) * eta(0) + eta(1) * eta(1) + eta(2) * eta(2)) + +eta(3) * eta(3) +
       eta(4) * eta(4) + eta(5) * eta(5);
  double etanorm = 0.0;
  etanorm = sqrt(eta(0) * eta(0) + eta(1) * eta(1) + eta(2) * eta(2) +
                 +2.0 * (eta(3) * eta(3) + eta(4) * eta(4) + eta(5) * eta(5)));

  // trial effective relative stress
  // qbar^{trial}_{n+1} := qbar(eta^{trial}_{n+1}) = \sqrt{ 3 . J2 }
  double qbar = 0.0;
  qbar = sqrt(3.0 * J2);

  // initialise the isotropic work hardening von Mises stress
  // sigma_yiso:= kappa = kappa(strainbar^p)
  double sigma_yiso = 0.0;

  //---------------------------------------------------------------------------
  // check plastic admissibility, Phi<=0 is admissble
  //---------------------------------------------------------------------------

  // ----------------------------------------------- trial yield function

  // calculate the yield stress
  // sigma_y = sigma_y0 + kappa(strainbar^p)
  // kappa == sigma_yiso, kappa is already used as material parameter
  double sigma_y = 0.0;

  bool bool_linisotrophard = false;
  // check if constant Hiso is given in input file
  if (((params_->sigma_y_).size()) == 0)
  {
    // so far: with linear isotropic hardening
    //         = sigma_y0 + Hiso . strainbar^p_{n}
    //         = sigma_y0 + Hiso . strainbar^{p, trial}_{n+1}
    sigma_yiso = Hiso * strainbar_p;
    sigma_y = sigma_y0 + sigma_yiso;

    bool_linisotrophard = true;
  }
  // calculate the isotropic hardening modulus and yield stress out of samples
  else
  {
    // calculate the isotropic hardening modulus with old plastic strains
    // Hiso = dsigma_y / d astrain^p
    Hiso = get_iso_hard_at_strainbarnp(strainbar_p);

    // calculate the uniaxial yield stress out of samples
    sigma_y = get_sigma_y_at_strainbarnp(strainbar_p);
  }

  // calculate the yield function with Dgamma = 0
  // Phi = \sqrt{ 3.0 . J2 } - sigma_y = q - sigma_y
  // with trial values: Phi_trial = q_trial - sigma_y
  double Phi_trial = 0.0;
  Phi_trial = qbar - sigma_y;

  // --------------------------------------------------------- initialise

  // if trial state is violated, i.e. it's a plastic load step, there are 2
  // possible states: plastic loading: heaviside = 1, elastic unloading = 0)
  double heaviside = 0.0;
  // incremental plastic multiplier Delta gamma
  double Dgamma = 0.0;

  // kinematic hardening curve of current time step and old time step
  // betabar = Hkin . strainbar_p
  // linear kinematic hardening: Hkin = const., else: Hkin = Hkin(strainnbar_p)
  double betabarold = 0.0;
  double betabar = 0.0;

  // unit flow vector Nbar (Prandtl-Reuss)
  // (using the updated relative stress eta_{n+1}, no longer eta_{n+1}^trial)
  // Nbar = ( eta^{trial}_{n+1} / || eta^{trial}_{n+1} || )
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Nbar(Core::LinAlg::Initialization::zero);

  // flow vector N (Prandtl-Reuss)
  // (using the updated relative stress eta_{n+1}, no longer eta^{trial}_{n+1})
  // N = sqrt{3/2} . ( eta^{trial}_{n+1} / || eta^{trial}_{n+1} || )
  //   = sqrt{3/2} . Nbar
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> N(Core::LinAlg::Initialization::zero);

  //---------------------------------------------------------------------------
  // IF consistency condition is violated, i.e. plastic load step
  // ( Phi^{trial} > 0.0, Dgamma >= 0.0 )
  //---------------------------------------------------------------------------
  if (Phi_trial > 1.0e-08)  // if (Phi^{trial} > 0.0)
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

    // calculate kinematic hardening stress of old time step
    // beta_{n} = Hkin . astrain^p_{n} = Hkin . strainbar^p, trial}_{n+1}
    betabarold = Hkin * strainbar_p;

    // --------------------------------------------------------- return-mapping

    // local Newton-Raphson

    // initialise
    const int itermax = 50;  // max. number of iterations
    int itnum = 0;           // iteration counter

    // Res:= residual of Newton iteration == yield function Phi
    double Res = 0.0;
    // calculate derivative of residual or tangent
    // ResTan = Phi' = d(Phi)/d(Dgamma)
    double ResTan = 0.0;
    // safety check: set to zero
    Dgamma = 0.0;
    betabar = 0.0;

    // start iteration with index m for local Newton
    while (true)
    {
      itnum++;
      // check for convergence

      // if not converged and (m > m_max)
      if (itnum > itermax)
      {
        FOUR_C_THROW(
            "local Newton iteration did not converge after iteration {:3d}/{:3d} with Res={:3f}",
            itnum, itermax, Res);
      }
      // else: continue loop (m <= m_max)

      // Res := Phi = qbar^{trial}_{n+1}
      //             - Delta gamma (3 . G + Hkin(strainbar_p + Dgamma) )
      //             - sigma_y(strainbar_p + Dgamma)
      // - Hiso is introduced in residual via sigma_y(strainbar_p + Dgamma)
      // - Hkin: using the relation: Delta gamma . Hkin := betabar - betabarold
      //
      // --> Res = qbar^{trial} - 3 * G * Dgamma
      //           - betabar(strainbar_p + Dgamma) + betabarold
      //           - sigma_y(strainbar_p + Dgamma)
      Res = qbar - 3.0 * G * Dgamma - betabar + betabarold - sigma_y;

      // check for convergence
      double norm = abs(Res);
      // check: absolute value of Res has to be smaller than given tolerance
      if (norm < (params_->abstol_))
      {
        break;
      }

      // plasticity with linear kinematic hardening
      // ResTan = -3G - Hkin(strainbar_p + Dgamma^{m-1}) - Hiso(strainbar_p + Dgamma^{m-1})
      // with Hiso = const. when considering LINEAR isotropic hardening
      ResTan = -3.0 * G - Hkin - Hiso;

      // incremental plastic multiplier Dgamma
      // Dgamma^{m} = Dgamma^{m-1} - Phi / Phi'
      Dgamma += (-Res) / ResTan;

      // -------------------------- local Newton update of plastic values

      // compute new residual of accumulatd plastic strains
      // astrain^p_{n+1} = astrain^p_n + Dgamma
      // astrain^p_{n+1} = SUM{Dgamma_n} from all time steps n
      // Kuhn-Tucker: Dgamma >= 0.0 --> astrain^p_{n+1} >= 0.0
      strainbar_p = strainbarpllast_->at(gp) + Dgamma;
      if (strainbar_p < 0.0)
        FOUR_C_THROW("accumulated plastic strain has to be equal or greater than zero");

      // Prager's linear kinemativ hardening rule
      // kinematic hardening stress betabar (scalar-valued)
      // beta_{n+1} = Hkin * astrain^p_{n+1}
      betabar = Hkin * strainbar_p;

      if (bool_linisotrophard == true)
      {
        // linear isotropic hardening
        // sigma = sigma_y0 + sigma_yiso(strainbar^p_{n+1})
        sigma_yiso = Hiso * strainbar_p;
        sigma_y = sigma_y0 + sigma_yiso;
      }
      else  // constant_Hiso == false
      {
        // Hiso = dsigma_y / d astrain^p_{n+1}
        Hiso = get_iso_hard_at_strainbarnp(strainbar_p);
        // sigma_y = sigma_y(astrain^p_{n+1})
        sigma_y = get_sigma_y_at_strainbarnp(strainbar_p);
      }


    }  // end of local Newton iteration

    // --------------------------------------------------- plastic update

    // ---------------------------------------------- update flow vectors
    // unit flow vector Nbar = eta_{n+1}^{trial} / || eta_{n+1}^{trial} ||
    Nbar.update(eta);
    Nbar.scale(1.0 / etanorm);

    // flow vector N = sqrt(3/2) eta_{n+1}^{trial} / || eta_{n+1}^{trial} ||
    N.update((sqrt(3.0 / 2.0)), Nbar);

    // update relative stress eta_{n+1}, cf. (7.193)
    // eta = ( 1 - (Delta gamma / qbar_{n+1}^{trial}) . [ 3 . G + Hkin] ) eta_{n+1}^{trial}
    // H_iso is not needed for update of the stress
    const double etafac = 1.0 - ((Dgamma / qbar) * (3.0 * G + Hkin));
    eta.scale(etafac);

    // update back stress, cf. (7.197)
    // beta_{n+1} = beta_n . sqrt(2/3) . (betabar - betabarold) . eta / etanorm;
    // sqrt(2/3) N =  2/3 . ( sqrt(3/2) eta / etanorm)
    const double facbeta = 2.0 / 3.0 * (betabar - betabarold);
    beta.update(facbeta, N, 1.0);

    // deviatoric stress
    // s = s_{n+1}^{trial} - 2 . G . Delta gamma . N
    const double facdevstress = (-2.0) * G * Dgamma;
    devstress_view.update(facdevstress, N, 1.0);

    // total stress
    // sigma_{n+1} = s_{n+1} + p_{n+1} . id2
    // pressure/volumetric stress no influence due to plasticity
    ThermoPlasticLinElast::stress(p, devstress, stress);

    // total strains
    // strain^e_{n+1} = strain^(e,trial)_{n+1} - Dgamma . N
    // compute converged engineering strain components (Voigt-notation)
    strain_e.update(1.0, trialstrain_e, (-Dgamma), N);

    // strain^p_{n+1} = strain^p_n + Dgamma . N
    strain_p.update(Dgamma, N, 1.0);

    // compute converged engineering strain components (Voigt-notation)
    for (int i = 3; i < 6; ++i) strain_e(i) *= 2.0;
    for (int i = 3; i < 6; ++i) strain_p(i) *= 2.0;

    // --------------------------------------------------- update history
    // plastic strain
    strainplcurr_->at(gp) = strain_p;

    // accumulated plastic strain
    strainbarplcurr_->at(gp) = strainbar_p;

    // back stress
    backstresscurr_->at(gp) = beta;


  }  // plastic corrector

  //---------------------------------------------------------------------------
  // ELSE: elastic step (Phi_trial <= 0.0, Dgamma = 0.0)
  //---------------------------------------------------------------------------
  else  // (Phi_trial <= 0.0)
  {
    // trial state vectors = result vectors of time step n+1
    // sigma^e_{n+1} = sigma^{e,trial}_{n+1} = s^{trial}_{n+1} + p . id2
    ThermoPlasticLinElast::stress(p, devstress, stress);

    // total strains
    // strain^e_{n+1} = strain^(e,trial)_{n+1}
    // compute converged engineering strain components (Voigt-notation)
    strain_e.update(trialstrain_e);
    for (int i = 3; i < 6; ++i) strain_e(i) *= 2.0;

    // no plastic yielding
    Dgamma = 0.0;
    // kinematic hardening curve of current time step and old time step
    // betabar = Hkin * strainbar_p
    // linear kinematic hardening: Hkin = const., else: Hkin = Hkin(strainnbar_p)
    betabarold = 0.0;
    betabar = 0.0;

    // pass the current plastic strains to the element (for visualisation)
    // compute converged engineering strain components (Voigt-notation)
    for (int i = 3; i < 6; ++i) strain_p(i) *= 2.0;

    // --------------------------------------------------------- update history
    // constant values for
    //  - plastic strains
    //  - accumulated plastic strains
    //  - back stress
    //    (--> relative stress)

    // as current history vectors are set to zero in update(), the old values
    // need to be set instead, otherwise no constant plastic values are possible
    strainplcurr_->at(gp) = strainpllast_->at(gp);
    strainbarplcurr_->at(gp) = strainbarpllast_->at(gp);
    backstresscurr_->at(gp) = backstresslast_->at(gp);


  }  // elastic step

  // -------------------------------- add the temperature dependent stress part

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

  // calculate the temperature difference
  // Delta T = T - T_0
  Core::LinAlg::Matrix<1, 1> deltaT(Core::LinAlg::Initialization::uninitialized);
  deltaT(0, 0) = temperature - params_->thetainit_;

  // temperature dependent stress
  // sigma = C_theta * Delta T = (m*I) * Delta T
  Core::LinAlg::SymmetricTensor<double, 3, 3> ctemp{};
  setup_cthermo(ctemp);
  stress += deltaT(0, 0) * ctemp;

  //---------------------------------------------------------------------------
  // --------------------------------- consistent elastoplastic tangent modulus
  //---------------------------------------------------------------------------

  // using an associative flow rule: C_ep is symmetric
  // ( generally C_ep is nonsymmetric )
  // if Phi^trial = 0: two tangent stress-strain relations exist
  // plastic loading --> C == C_ep
  if (Dgamma > 0.0) heaviside = 1.0;
  // elastic unloading --> C == C_e
  else
    heaviside = 0.0;

  // using an associative flow rule: C_ep is symmetric
  // ( generally C_ep is nonsymmetric )
  setup_cmat_elasto_plastic(cmat_view, Dgamma, G, qbar, N, Nbar, heaviside, Hiso, Hkin);


  //---------------------------------------------------------------------------
  // ------------------------------------------ internal/mechanical dissipation
  //---------------------------------------------------------------------------

  // ----------------------------------- compute plastic strain increment
  // strain^p_{n+1}' = gamma' . N
  // with implicit Euler:
  // (strain^p_{n+1}-strain^p_n)/dt = Dgamma/dt . N
  // Inc_strain^p_{n+1} := (strain^p_{n+1}-strain^p_n)
  //                     = Dgamma . N = Dgamma . sqrt{3/2} eta_{n+1} / || eta_{n+1}||
  for (int i = 0; i < 6; i++) incstrainpl_->at(gp)(i, 0) = Dgamma * N(i, 0);
  // --> plastic strain rate: strain^p_{n+1}' = Incstrainpl_/dt
  // scale with dt in StrainRateSplit()

  // ------------------------------------------------ dissipation for r_T
  // calculate mechanical dissipation required for thermo balance equation
  dissipation(gp, sigma_yiso, Dgamma, N, stress_view);

  // --------------------------------------- kinematic hardening for k_TT
  // temperature-dependent dissipated mechanical power
  // if (tr(strain^p) == 0) and (sigma_T(i,i)=const.) --> dot product of both is zero
  // safety check:
  double tracestrainp = 0.0;
  tracestrainp = strain_p(0) + strain_p(1) + strain_p(2);
  if (tracestrainp > 1.0E-8) FOUR_C_THROW("trace of plastic strains is not equal to zero!");

  // ----------------------------------- linearisation of D_mech for k_Td
  dissipation_coupl_cond(cmat_view, gp, G, Hiso, Hkin, heaviside, etanorm, Dgamma, N, stress_view);
}  // evaluate()

/*----------------------------------------------------------------------*
 | Set current quantities for this material                             |
 *----------------------------------------------------------------------*/
void Mat::ThermoPlasticLinElast::reinit(const Core::LinAlg::Tensor<double, 3, 3>* defgrd,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, double temperature, unsigned gp)
{
  reinit(temperature, gp);
}

/*----------------------------------------------------------------------*
 | calculate stress-temperature modulus and thermal derivative          |
 |   for coupled thermomechanics                                        |
 *----------------------------------------------------------------------*/
void Mat::ThermoPlasticLinElast::stress_temperature_modulus_and_deriv(
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stm,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stm_dT, int gp)
{
  setup_cthermo(stm);
  stm_dT = Core::LinAlg::TensorGenerators::full<3, 3>(0.0);
}

/*----------------------------------------------------------------------*
 |  Evaluates the added derivatives of the stress w.r.t. all scalars    |
 *----------------------------------------------------------------------*/
Core::LinAlg::SymmetricTensor<double, 3, 3> Mat::ThermoPlasticLinElast::evaluate_d_stress_d_scalar(
    const Core::LinAlg::Tensor<double, 3, 3>& defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext& context, int gp, int eleGID)
{
  // get the temperature-dependent material tangent
  Core::LinAlg::SymmetricTensor<double, 3, 3> ctemp;
  setup_cthermo(ctemp);

  return ctemp;
}

/*----------------------------------------------------------------------*
 | computes linear stress tensor                             dano 05/11 |
 *----------------------------------------------------------------------*/
void Mat::ThermoPlasticLinElast::stress(const double p,            // volumetric stress
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& devstress,  // deviatoric stress tensor
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress            // 2nd PK-stress
) const
{
  // total stress = deviatoric + hydrostatic pressure . I
  // sigma = s + p . I
  stress = devstress + p * Core::LinAlg::TensorGenerators::identity<double, 3, 3>;
}  // Stress()


/*----------------------------------------------------------------------*
 | compute relative deviatoric stress tensor                 dano 08/11 |
 *----------------------------------------------------------------------*/
void Mat::ThermoPlasticLinElast::rel_dev_stress(
    const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& devstress,  // deviatoric stress tensor
    const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& beta,       // back stress tensor
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& eta               // relative stress
) const
{
  // relative stress = deviatoric - back stress
  // eta = s - beta
  eta.update(1.0, devstress, (-1.0), beta);

}  // RelDevStress()


/*----------------------------------------------------------------------*
 | computes isotropic elasticity tensor in matrix notion     dano 08/11 |
 | for 3d                                                               |
 *----------------------------------------------------------------------*/
void Mat::ThermoPlasticLinElast::setup_cmat(
    Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>& cmat) const
{
  // get material parameters
  // Young's modulus (modulus of elasticity)
  double young = params_->youngs_;
  // Poisson's ratio
  double nu = params_->poissonratio_;

  // isotropic elasticity tensor C in Voigt matrix notation, cf. FEscript p.29
  //                       [ 1-nu     nu     nu |          0    0    0 ]
  //                       [        1-nu     nu |          0    0    0 ]
  //           E           [               1-nu |          0    0    0 ]
  //   C = --------------- [ ~~~~   ~~~~   ~~~~   ~~~~~~~~~~  ~~~  ~~~ ]
  //       (1+nu)*(1-2*nu) [                    | (1-2*nu)/2    0    0 ]
  //                       [                    |      (1-2*nu)/2    0 ]
  //                       [ symmetric          |           (1-2*nu)/2 ]
  //
  const double mfac = young / ((1.0 + nu) * (1.0 - 2.0 * nu));  // factor

  // clear the material tangent
  cmat.clear();
  // write non-zero components
  cmat(0, 0) = mfac * (1.0 - nu);
  cmat(0, 1) = mfac * nu;
  cmat(0, 2) = mfac * nu;
  cmat(1, 0) = mfac * nu;
  cmat(1, 1) = mfac * (1.0 - nu);
  cmat(1, 2) = mfac * nu;
  cmat(2, 0) = mfac * nu;
  cmat(2, 1) = mfac * nu;
  cmat(2, 2) = mfac * (1.0 - nu);
  // ~~~
  cmat(3, 3) = mfac * 0.5 * (1.0 - 2.0 * nu);
  cmat(4, 4) = mfac * 0.5 * (1.0 - 2.0 * nu);
  cmat(5, 5) = mfac * 0.5 * (1.0 - 2.0 * nu);

}  // setup_cmat()


/*----------------------------------------------------------------------*
 | computes isotropic elasticity tensor in matrix notion     dano 05/11 |
 | for 3d                                                               |
 *----------------------------------------------------------------------*/
void Mat::ThermoPlasticLinElast::setup_cmat_elasto_plastic(
    Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
        cmat,                                           // elasto-plastic tangent modulus (out)
    double Dgamma,                                      // plastic multiplier
    double G,                                           // shear modulus
    double q,                                           // elastic trial von Mises effective stress
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> flowvector,  // flow vector
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Nbar,        // unit flow vector
    double heaviside,                                   // Heaviside function
    double Hiso,                                        // isotropic hardening modulus
    double Hkin                                         // kinematic hardening modulus
) const
{
  // incremental constitutive function for the stress tensor
  // sigma_{n+1} = [ cmat - (Dgamma 6 G^2/q) I_d ] : strain^{e,trial}_{n+1}
  // consistent tangent operator
  // D^{ep} := dsigma_{n+1} / dstrain^{e,trial}_{n+1}

  // depending on the flow vector Cmat_ep can be a fully-occupied matrix

  // C_ep = C_e - ( H^ . Dgamma . 6 . G^2 ) / qbar^{trial} . I_d +
  //        +  H^ . 6 . G^2 ( Dgamma/qbar^{trial} - 1/(3 G + Hkin + Hiso) ) Nbar \otimes Nbar

  // ---------------------------------------------------------- Heaviside
  // if plastic loading:   heaviside = 1.0 --> use C_ep
  // if elastic unloading: heaviside = 0.0 --> use C_e

  // I_d = I_s - 1/3 I . I
  // I_d in Voigt-notation applied to symmetric problem, like stress calculation
  //         [ 2/3   -1/3  -1/3 | 0    0    0  ]
  //         [-1/3    2/3  -1/3 | 0    0    0  ]
  //         [-1/3   -1/3   2/3 | 0    0    0  ]
  //   I_d = [ ~~~~  ~~~~  ~~~~  ~~~  ~~~  ~~~ ]
  //         [                  | 1/2   0   0  ]
  //         [    symmetric     |      1/2  0  ]
  //         [                  |          1/2 ]
  //

  // build Cartesian identity 2-tensor I_{AB}
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> id2(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++) id2(i) = 1.0;

  // set Cartesian identity 4-tensor in 6-Voigt matrix notation
  // this is fully 'contra-variant' identity tensor, ie I^{ABCD}
  // REMARK: rows are stress-like 6-Voigt
  //         columns are stress-like 6-Voigt
  Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> id4sharp(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++) id4sharp(i, i) = 1.0;
  for (int i = 3; i < 6; i++) id4sharp(i, i) = 0.5;

  // unit flow vector Nbar (cf. de Souza Neto (7.117)/(7.210) )
  // Nbar = eta^{trial}_{n+1} / || eta^{trial}_{n+1} || = sqrt(2/3) . N
  double flowfac = sqrt(2.0 / 3.0);
  flowvector.scale(flowfac);

  // ------------------------------------------------------- elastic term
  // C_ep = C_e
  // add standard isotropic elasticity tensor C_e first
  setup_cmat(cmat);

  // ------------------------------------------------------ plastic terms

  // ------------------------------------------------- first plastic term
  // - ( H^ . Dgamma . 6 . G^2 ) / qbar^{trial} . I_d
  double epfac = 0.0;
  // elastic trial von Mises effective stress
  if (q != 0.0)
  {
    epfac = (-1.0) * heaviside * Dgamma * 6.0 * G * G / q;
  }
  // constitutive tensor
  // I_d = id4sharp - 1/3 Id \otimes Id
  // contribution: Id4^#
  cmat.update(epfac, id4sharp, 1.0);
  // contribution: Id \otimes Id
  double epfac1 = 0.0;
  epfac1 = epfac / (-3.0);
  cmat.multiply_nt(epfac1, id2, id2, 1.0);

  // ------------------------------------------------ second plastic term
  // +  H^ . 6 . G^2 ( Dgamma/qbar^{trial} - 1/(3 G + Hkin + Hiso) ) Nbar \otimes Nbar

  // unit flow vector (using co-linearity between trial and end state of eta)
  // Nbar = eta_{n+1} / || eta_{n+1} ||
  //      = eta^{trial}_{n+1} / || eta^{trial}_{n+1} ||

  // ------------------------------------------------------------ tangent

  if (q != 0.0)
  {
    double epfac2 = 0.0;
    // loop strains (columns)
    for (int k = 0; k < 6; ++k)
    {
      // loop stresses (rows)
      for (int i = 0; i < 6; ++i)
      {
        epfac2 = heaviside * 6.0 * G * G * (Dgamma / q - 1.0 / (3.0 * G + Hkin + Hiso));
        cmat(i, k) += epfac2 * Nbar(i) * Nbar(k);
      }  // end rows, loop i
    }  // end columns, loop k
  }  // (q != 0.0)


}  // setup_cmat_elasto_plastic()


/*----------------------------------------------------------------------*
 | split given strain rate into elastic and plastic term     dano 08/11 |
 *----------------------------------------------------------------------*/
void Mat::ThermoPlasticLinElast::strain_rate_split(int gp,    // current Gauss point
    const double stepsize,                                    // step size
    const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& strainrate  // total strain rate, i.e. B d'
)
{
  // elastic strain rate strain^e'
  // strain^e' = strain' - strain^p'
  for (int i = 0; i < NUM_STRESS_3D; i++)
  {
    // strain^e' = strain' - strain^p'

    // with strain^p' = Inc_strain^p / dt: use implicit Euler scheme
    strainelrate_->at(gp)(i) = strainrate(i, 0) - (1.0 / stepsize) * incstrainpl_->at(gp)(i);
  }

  return;

}  // StrainRateSplit


/*----------------------------------------------------------------------*
 | compute internal dissipation term                         dano 04/13 |
 *----------------------------------------------------------------------*/
void Mat::ThermoPlasticLinElast::dissipation(int gp,  // current Gauss point
    double sigma_yiso,                                // isotropic work hardening von Mises stress
    double Dgamma,                                    // plastic multiplier/increment
    const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& N,  // flow vector
    const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& stress  // total mechanical stress
)
{
  // D_mech = stress : strain^p' + beta : d(Phi)/d(beta) Dgamma
  //       = (stress - beta) : strain^p' = (stress - beta) : gamma' . N
  //       = eta : Dgamma/Dt . N

  // ----------------------------- dissipation due to kinematic hardening
  // (stress - beta) : strain^p_{n+1}'
  // with total stress: stress_d + stress_T

  // --------------------------------------- kinematic hardening for fint
  // stressdiff = stress_d_{n+1} - beta_{n+1} = s_{n+1} + p_{n+1} . I - beta_{n+1}
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> stressdiff(Core::LinAlg::Initialization::uninitialized);
  stressdiff.update(1.0, stress, (-1.0), (backstresscurr_->at(gp)));

  // Dmech = (stress_d + sigma_T - beta) : Inc_strain^p_{n+1}
  double stressIncstrainpl = 0.0;
  stressIncstrainpl =
      incstrainpl_->at(gp)(0) * stressdiff(0) + incstrainpl_->at(gp)(1) * stressdiff(1) +
      incstrainpl_->at(gp)(2) * stressdiff(2) + incstrainpl_->at(gp)(3) * stressdiff(3) +
      incstrainpl_->at(gp)(4) * stressdiff(4) + incstrainpl_->at(gp)(5) * stressdiff(5);

  // --------------------------------------- isotropic hardening for fint
  // kappa(strainbar^p) . strainbar^p' = sigma_yiso . Dgamma/dt
  double isotropicdis = sigma_yiso * Dgamma;

  // return mechanical dissipation term due to mixed hardening hardening
  dmech_->at(gp) = -stressIncstrainpl + isotropicdis;
  // time step not yet considered, i.e., Dmech_ is an energy, not a power
  // accumulated plastic strain

}  // Dissipation()


/*----------------------------------------------------------------------*
 | compute linearisation of internal dissipation for k_Td    dano 04/13 |
 *----------------------------------------------------------------------*/
void Mat::ThermoPlasticLinElast::dissipation_coupl_cond(
    const Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
        cmat,                                             // elasto-plastic tangent modulus (out)
    int gp,                                               // current Gauss point
    double G,                                             // shear modulus
    double Hiso,                                          // isotropic hardening modulus
    double Hkin,                                          // kinematic hardening modulus
    double heaviside,                                     // Heaviside function
    double etanorm,                                       // norm of eta^{trial}_{n+1}
    double Dgamma,                                        // plastic multiplier
    const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& N,      // flow vector
    const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& stress  // flow vector
)
{
  // ----------------------------------- linearisation of D_mech for k_Td

  // calculate the derivation of the dissipation w.r.t. to the strains
  // (dD_mech/dstrain)
  // = N_T^T . (- dDmech_kin/ dstrain + dDmech_iso/ dstrain )
  //
  // = - N_T^T . (d [ (sigma_{d,T} - beta) . strain^p' ]/ dstrain)
  //   + N_T^T . (d [ kappa(strainbar^p) . strainbar^p' ]/ dstrain)

  // ---------------------------------------------------------- Heaviside
  // if plastic loading:   heaviside = 1.0 --> use C_ep
  // if elastic unloading: heaviside = 0.0 --> use C_e

  // I_d = I_s - 1/3 I . I
  // I_d in Voigt-notation applied to symmetric problem, like stress calculation
  //         [ 2/3   -1/3  -1/3 | 0    0    0  ]
  //         [-1/3    2/3  -1/3 | 0    0    0  ]
  //         [-1/3   -1/3   2/3 | 0    0    0  ]
  //   I_d = [ ~~~~  ~~~~  ~~~~  ~~~  ~~~  ~~~ ]
  //         [                  | 1/2   0   0  ]
  //         [    symmetric     |      1/2  0  ]
  //         [                  |          1/2 ]
  //

  // build Cartesian identity 2-tensor I_{AB}
  // build Cartesian identity 2-tensor I_{AB}
  Core::LinAlg::Matrix<6, 1> id2(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++) id2(i) = 1.0;

  // set Cartesian identity 4-tensor in 6-Voigt matrix notation
  // this is fully 'contra-variant' identity tensor, ie I^{ABCD}
  // REMARK: rows are stress-like 6-Voigt
  //         columns are stress-like 6-Voigt
  Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> id4sharp(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++) id4sharp(i, i) = 1.0;
  for (int i = 3; i < 6; i++) id4sharp(i, i) = 0.5;

  // ---------------------- linearisation of KINEMATIC hardening for k_Td

  // (dD_mech_kin/dstrain) = (d [ (sigma_{d,T} - beta) . strain^p' ]/ dstrain)
  //
  // d[ (sigma_{d,T} - beta) . strain^p' ]/ dstrain
  // = d(sigma_{d,T} - beta)/dstrain . strain^p'
  //   + (sigma_{d,T} - beta) . (dstrain^p')/ dstrain)
  //
  // sigma_T is independent of deformation, i.e. strains: dsigma_T/dstrain = 0
  //
  // = d(sigma_d - beta)/dstrain . strain^p'
  //   + (sigma_{d,T} - beta) . [(dstrain^p')/ dstrain]

  // d(sigma_d - beta)/dstrain = dstress_d/dstrain = C_ep
  // calculate C_ep . Inc_strain^p_{n+1}
  Core::LinAlg::Matrix<6, 1> cmatstrainpinc(Core::LinAlg::Initialization::uninitialized);
  cmatstrainpinc.multiply(cmat, incstrainpl_->at(gp));
  // --> divide by dt in thermo_ele

  // (sigma_d - beta) . [(dstrain^p')/ dstrain] = eta_{n+1} . [(dstrain^p')/ dstrain]

  // ---------------------- linearisation of plastic strain
  // [(dstrain^p')/ dstrain] = 1/Dt . [(dstrain^p_{n+1})/ dstrain_{n+1}^{e,trial}]
  // = 2G/(3 G + Hkin + Hiso) . N \otimes N
  //   + Dgamma . 2G / || eta^{trial}_{n+1} || [sqrt(3/2) I_d - N \otimes N]

  Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> Dmech_kin_d(
      Core::LinAlg::Initialization::uninitialized);
  double fac_kinlin = 0.0;
  if (etanorm != 0.0)
  {
    fac_kinlin = heaviside * Dgamma * 2.0 * G / etanorm;
  }

  // I_d = id4sharp - 1/3 Id \otimes Id
  double fac_kinlin1 = 0.0;
  fac_kinlin1 = sqrt(3.0 / 2.0) * fac_kinlin;
  double fac_kinlin2 = 0.0;
  fac_kinlin2 = fac_kinlin1 / (-3.0);
  // contribution: Id4^#
  Dmech_kin_d.update(fac_kinlin1, id4sharp);
  // contribution: Id \otimes Id
  Dmech_kin_d.multiply_nt(fac_kinlin2, id2, id2, 1.0);

  double fac_lin_3 = 0.0;
  double fac_lin_4 = 0.0;
  fac_lin_4 = 3.0 * G * Hkin * Hiso;
  if (fac_lin_4 != 0) fac_lin_3 = heaviside * 2.0 * G / fac_lin_4;
  double fac_kinlin_flowvect = 0.0;
  fac_kinlin_flowvect = fac_lin_3 - fac_kinlin;

  // loop strains (columns)
  for (int k = 0; k < 6; ++k)
  {
    // loop stresses (rows)
    for (int i = 0; i < 6; ++i)
    {
      Dmech_kin_d(i, k) += fac_kinlin_flowvect * N(i) * N(k);
    }  // end rows, loop i
  }  // end columns, loop k

  // ---------------------- linearisation of ISOTROPIC hardening for k_Td

  // ----------------------------------------linearisation of Dmech_iso
  // dD_mech_iso/dstrain = (d [ kappa(strainbar^p) . strainbar^p' ]/ dstrain)
  //                     = Hiso . (d[ strainbar^p . strainbar^p' ]/dstrain)
  //
  // with linear isotropic hardening, i.e. sigma_yiso := kappa = Hiso . strainbar^p
  //
  // d[ sigma_yiso . strainbar^p' ]/dstrain^{trial}_{n+1}
  // = dkappa/dstrain^{trial}_{n+1} . strainbar^p' + kappa . dstrainbar^p'/dstrain^{trial}_{n+1}
  // = Hiso . dstrainbar^p/dstrain^{trial}_{n+1} . strainbar^p'
  //   + sigma_yiso . dstrainbar^p'/dstrain^{trial}_{n+1}
  //
  // dstrainbar^p/dstrain^{trial}_{n+1} = dDgamma/dstrain^{trial}_{n+1}
  //                                    = 2G/(3G + Hkin + Hiso) N_{n+1}
  // dstrainbar^p'/dstrain^{trial}_{n+1} = 1/Dt . 2G/(3G + Hkin + Hiso) N_{n+1}
  // --> as usual: calculation with dt is done in thermo_ele

  // dD_mech_iso/dstrain
  // = 2G/(3G + Hkin + Hiso) . (N_{n+1} . strainbar^p' + sigma_yiso . 1/Dt N_{n+1})
  // = 2G/(3G + Hkin + Hiso) . 1/Dt . (Inc_strainbar^p_{n+1} + sigma_yiso) N_{n+1}
  // = 2G/(3G + Hkin + Hiso) . 1/Dt . (strainbar^p_{n+1} - strainbar^p_n +
  //   + Hiso . strainbar^p_{n+1}) N_{n+1}
  // = 2G/(3G + Hkin + Hiso) . 1/Dt . (strainbar^p_{n+1} (1+Hiso) - strainbar^p_n) N_{n+1}
  double fac_liniso = 0.0;
  fac_liniso = fac_lin_3 * ((1.0 + Hiso) * strainbarplcurr_->at(gp) - -strainbarpllast_->at(gp));

  // ------------------------------------------------------ term for k_Td
  // add the linearisation term to D_mech_d
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> D_mech_d(Core::LinAlg::Initialization::uninitialized);
  D_mech_d.multiply(Dmech_kin_d, stress);
  D_mech_d.update((-1.0), cmatstrainpinc, (-1.0));
  D_mech_d.update((fac_liniso), N, 1.0);
  // update history
  dmech_d_->at(gp) = D_mech_d;

}  // dissipation_coupl_cond()


/*----------------------------------------------------------------------*
 | calculate stresses by evaluating the temperature tangent  dano 08/11 |
 *----------------------------------------------------------------------*/
void Mat::ThermoPlasticLinElast::evaluate(
    const Core::LinAlg::Matrix<1, 1>& Ntemp,  // shapefcts . temperatures
    Core::LinAlg::SymmetricTensor<double, 3, 3>& ctemp,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stresstemp)
{
  setup_cthermo(ctemp);

  // calculate the temperature difference
  Core::LinAlg::Matrix<1, 1> init(Core::LinAlg::Initialization::uninitialized);
  init(0, 0) = (params_->thetainit_);
  // Delta T = T - T_0
  Core::LinAlg::Matrix<1, 1> deltaT(Core::LinAlg::Initialization::uninitialized);
  deltaT.update(1.0, Ntemp, (-1.0), init);

  // temperature dependent stress
  // sigma = C_theta * Delta T = (m*I) * Delta T
  stresstemp = ctemp * deltaT(0, 0);

  // if stresstemp(i,i)=const.: (sigma_T : strainp' == 0), because (tr(strainp') == 0)
  // for different thermal stresses, term has to be considered!!!
  //  // calculate temperature-dependent stress term for dissipation
  //  double tempstressIncstrainpl = 0.0;
  //  tempstressIncstrainpl = stresstemp(0) * Incstrainpl_->at(gp)(0)
  //                          + stresstemp(1) * Incstrainpl_->at(gp)(1)
  //                          + stresstemp(2) * Incstrainpl_->at(gp)(2)
  //                          + stresstemp(3) * Incstrainpl_->at(gp)(3)
  //                          + stresstemp(4) * Incstrainpl_->at(gp)(4)
  //                          + stresstemp(5) * Incstrainpl_->at(gp)(5);
  //  // term enters as negative value into the balance equation

}  // Evaluate


/*----------------------------------------------------------------------*
 | computes temperature dependent isotropic                  dano 05/10 |
 | elasticity tensor in matrix notion for 3d, second(!) order tensor    |
 *----------------------------------------------------------------------*/
void Mat::ThermoPlasticLinElast::setup_cthermo(
    Core::LinAlg::SymmetricTensor<double, 3, 3>& ctemp) const
{
  double m = st_modulus();

  // isotropic elasticity tensor C_temp in Voigt matrix notation C_temp = m I
  //
  // Matrix-notation for 3D case
  //              [ m      0      0 ]
  //   C_temp =   [ 0      m      0 ]
  //              [ 0      0      m ]
  //
  //  in Vector notation
  //   C_temp =   [m, m, m, 0, 0, 0]^T
  //
  // write non-zero components

  // clear the material tangent
  ctemp = m * Core::LinAlg::TensorGenerators::identity<double, 3, 3>;
}  // setup_cthermo()


/*----------------------------------------------------------------------*
 | calculates stress-temperature modulus                     dano 08/11 |
 *----------------------------------------------------------------------*/
double Mat::ThermoPlasticLinElast::st_modulus() const
{
  // initialise the parameters for the lame constants
  const double ym = params_->youngs_;
  const double pv = params_->poissonratio_;

  // initialise the thermal expansion coefficient
  const double thermexpans = params_->thermexpans_;

  // plane strain, rotational symmetry
  // E / (1+nu)
  const double c1 = ym / (1.0 + pv);
  // (E*nu) / ((1+nu)(1-2nu))
  const double b1 = c1 * pv / (1.0 - 2.0 * pv);

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
 | return derivative of piecewise linear function for the    dano 02/14 |
 | yield stress, i.e. isotropic hardening modulus at current            |
 | accumulated plastic strain                                           |
 *----------------------------------------------------------------------*/
double Mat::ThermoPlasticLinElast::get_iso_hard_at_strainbarnp(
    const double strainbar_p  // current accumulated strain
) const
{
  // Hiso = d sigma_y / d astrain^p_{n+1}
  double Hiso = 0.0;

  // extract vectors of samples
  const std::vector<double> strainbar_p_ref = params_->strainbar_p_ref_;
  const std::vector<double> sigma_y_ref = params_->sigma_y_;
  // how many samples are available
  double samplenumber = sigma_y_ref.size();

  // loop over all samples
  for (int i = 0; i < samplenumber; ++i)
  {
    // astrain^{p}_{n+1} > astrain^{p}_ref^[i]
    if (strainbar_p >= strainbar_p_ref[i])
    {
      // astrain^{p}_{n+1} > astrain^{p}_ref^[i] --> sigma_y = sigma_ref[i]
      // --> Hiso = d sigma_y / d astrain^{p}_{n+1} = 0
      Hiso = 0.0;
      continue;
    }

    // (strainbar_p < strainbar_p_ref[i])
    else
    {
      // load is still elastic: astrain^{p}_{n+1} < astrain^{p}_ref^{i=0}
      if (i == 0)
      {
        // yield boundary is the initial yield stress (sigma_y^{i=0})
        Hiso = 0.0;
        continue;
      }
      // astrain^{p,i-1}_ref < astrain^{p}_{n+1} < astrain^{p,i}_ref
      else
      {
        //         sigma_y_n - sigma_y^{i-1}
        // Hiso =  ---------------------------------------
        //        astrain^{p,i}_ref - astrain^{p,i-1}_ref
        Hiso =
            (sigma_y_ref[i] - sigma_y_ref[i - 1]) / (strainbar_p_ref[i] - strainbar_p_ref[i - 1]);
        continue;
      }
    }  // load is plastic, hardening can occur
  }  // loop over samples

  // return current isotropic hardening modulus
  return Hiso;

}  // GetIsoHardeningModulus()


/*----------------------------------------------------------------------*
 | compute current yield stress sigma_y(astrain^p)           dano 02/14 |
 | calculate yield stress from (sigma_y-astrain^p)-samples              |
 *----------------------------------------------------------------------*/
double Mat::ThermoPlasticLinElast::get_sigma_y_at_strainbarnp(
    const double strainbar_p  // current accumulated strain, in case of dependent hardening
                              // if damage!=0: isotropic hardening internal variable
) const
{
  // extract vectors of samples
  const std::vector<double> sigma_y_ref = params_->sigma_y_;
  // how many samples are available
  double samplenumber = sigma_y_ref.size();
  // return the yield stress
  double sigma_y_interpol = 0.0;

  // uniaxial yield stress given by piecewise linear curve
  if (samplenumber > 0)
  {
    // get vector astrain^p_ref
    const std::vector<double> strainbar_p_ref = params_->strainbar_p_ref_;
    if (sigma_y_ref.size() != strainbar_p_ref.size())
      FOUR_C_THROW("Samples have to fit to each other!");

    // loop over samples
    for (int i = 0; i < samplenumber; ++i)
    {
      // astrain^{p}_{n+1} > astrain^{p}_ref^[i]
      if (strainbar_p >= strainbar_p_ref[i])
      {
        sigma_y_interpol = sigma_y_ref[i];
      }
      // current strains are <= strainbar_p_ref_max
      else  // astrain^{p}_{n+1} < astrain^{p}_ref^[i]
      {
        // astrain^{p}_{n+1} < astrain^{p}_ref^{i=0}, i.e. load is still elastic
        if (i == 0)
        {
          // yield boundary is the initial yield stress (sigma_y^{i=0})
          sigma_y_interpol = sigma_y_ref[0];
          continue;
        }
        // astrain^{p}_ref^{i=0} < astrain^{p}_{n+1} < astrain^{p}_ref^i
        else
        {
          // astrain^{p,i-1}_ref < astrain^{p}_{n+1} < astrain^{p,i}_ref
          if (strainbar_p < strainbar_p_ref[i])
          {
            // sigma_y_{n+1} = sigma_y^i +
            //                                        sigma_y^i - sigma_y^{i-1}
            // + (astrain^p_{n+1} - astrain^{p,i-1}) ---------------------------------------
            //                                      astrain^{p,i}_ref - astrain^{p,i-1}_ref
            sigma_y_interpol =
                sigma_y_ref[i - 1] + (strainbar_p - strainbar_p_ref[i - 1]) *
                                         (sigma_y_ref[i] - sigma_y_ref[i - 1]) /
                                         (strainbar_p_ref[i] - strainbar_p_ref[i - 1]);
          }  // current strains between strain^{i-1} and strain^i
        }  // plastic regime
      }
    }  // loop over all samples
  }  // samplenumber > 1

  // return current yield stress
  return sigma_y_interpol;

}  // get_sigma_y_at_strainbarnp()


/*---------------------------------------------------------------------*
 | return names of visualization data (public)              dano 03/13 |
 *---------------------------------------------------------------------*/
void Mat::ThermoPlasticLinElast::vis_names(std::map<std::string, int>& names) const
{
  std::string accumulatedstrain = "accumulatedstrain";
  names[accumulatedstrain] = 1;  // scalar
}  // vis_names()


/*---------------------------------------------------------------------*
 | return visualization data (public)                       dano 03/13 |
 *---------------------------------------------------------------------*/
bool Mat::ThermoPlasticLinElast::vis_data(
    const std::string& name, std::vector<double>& data, int numgp, int eleID) const
{
  if (name == "accumulatedstrain")
  {
    if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");
    double temp = 0.0;
    for (int iter = 0; iter < numgp; iter++) temp += accumulated_strain(iter);
    data[0] = temp / numgp;
  }
  return true;
}  // vis_data()

/*----------------------------------------------------------------------*/

void Mat::ThermoPlasticLinElast::evaluate(const Core::LinAlg::Matrix<3, 1>& gradtemp,
    Core::LinAlg::Matrix<3, 3>& cmat, Core::LinAlg::Matrix<3, 1>& heatflux, const int eleGID) const
{
  thermo_->evaluate(gradtemp, cmat, heatflux, eleGID);
}

void Mat::ThermoPlasticLinElast::evaluate(const Core::LinAlg::Matrix<2, 1>& gradtemp,
    Core::LinAlg::Matrix<2, 2>& cmat, Core::LinAlg::Matrix<2, 1>& heatflux, const int eleGID) const
{
  thermo_->evaluate(gradtemp, cmat, heatflux, eleGID);
}

void Mat::ThermoPlasticLinElast::evaluate(const Core::LinAlg::Matrix<1, 1>& gradtemp,
    Core::LinAlg::Matrix<1, 1>& cmat, Core::LinAlg::Matrix<1, 1>& heatflux, const int eleGID) const
{
  thermo_->evaluate(gradtemp, cmat, heatflux, eleGID);
}

std::vector<double> Mat::ThermoPlasticLinElast::conductivity(int eleGID) const
{
  return thermo_->conductivity(eleGID);
}

void Mat::ThermoPlasticLinElast::conductivity_deriv_t(Core::LinAlg::Matrix<3, 3>& dCondDT) const
{
  thermo_->conductivity_deriv_t(dCondDT);
}

void Mat::ThermoPlasticLinElast::conductivity_deriv_t(Core::LinAlg::Matrix<2, 2>& dCondDT) const
{
  thermo_->conductivity_deriv_t(dCondDT);
}

void Mat::ThermoPlasticLinElast::conductivity_deriv_t(Core::LinAlg::Matrix<1, 1>& dCondDT) const
{
  thermo_->conductivity_deriv_t(dCondDT);
}

double Mat::ThermoPlasticLinElast::capacity() const { return thermo_->capacity(); }

double Mat::ThermoPlasticLinElast::capacity_deriv_t() const { return thermo_->capacity_deriv_t(); }

void Mat::ThermoPlasticLinElast::reinit(double temperature, unsigned gp)
{
  current_temperature_ = temperature;
  if (thermo_ != nullptr) thermo_->reinit(temperature, gp);
}
void Mat::ThermoPlasticLinElast::reset_current_state()
{
  if (thermo_ != nullptr) thermo_->reset_current_state();
}

void Mat::ThermoPlasticLinElast::commit_current_state()
{
  if (thermo_ != nullptr) thermo_->commit_current_state();
}

/*----------------------------------------------------------------------*/

FOUR_C_NAMESPACE_CLOSE
