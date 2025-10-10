// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_damage.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_local_newton.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 | constructor (public)                                      dano 04/11 |
 *----------------------------------------------------------------------*/
Mat::PAR::Damage::Damage(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      youngs_(matdata.parameters.get<double>("YOUNG")),
      poissonratio_(matdata.parameters.get<double>("NUE")),
      density_(matdata.parameters.get<double>("DENS")),
      sigma_y_((matdata.parameters.get<std::vector<double>>("SIGMA_Y"))),
      strainbar_p_ref_((matdata.parameters.get<std::vector<double>>("EPSBAR_P"))),
      damden_(matdata.parameters.get<double>("DAMDEN")),
      damexp_(matdata.parameters.get<double>("DAMEXP")),
      epsbarD_(matdata.parameters.get<double>("DAMTHRESHOLD")),
      kinhard_(matdata.parameters.get<double>("KINHARD")),
      kinhard_rec_(matdata.parameters.get<double>("KINHARD_REC")),
      sathardening_(matdata.parameters.get<double>("SATHARDENING")),
      hardexpo_(matdata.parameters.get<double>("HARDEXPO")),
      abstol_(matdata.parameters.get<double>("TOL"))
{
  if (hardexpo_ < 0.0) FOUR_C_THROW("Nonlinear hardening exponent must be non-negative!");
  if (damden_ == 0.0)
    FOUR_C_THROW("Denominator has to be unequal to zero, otherwise floating point exception!");
}


/*----------------------------------------------------------------------*
 | is called in Material::Factory from read_materials()       dano 02/12 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::Damage::create_material()
{
  return std::make_shared<Mat::Damage>(this);
}


Mat::DamageType Mat::DamageType::instance_;


/*----------------------------------------------------------------------*
 | is called in Material::Factory from read_materials()       dano 02/12 |
 *----------------------------------------------------------------------*/
Core::Communication::ParObject* Mat::DamageType::create(Core::Communication::UnpackBuffer& buffer)
{
  Mat::Damage* plastic = new Mat::Damage();
  plastic->unpack(buffer);
  return plastic;
}


/*----------------------------------------------------------------------*
 | constructor (public)                                      dano 04/11 |
 *----------------------------------------------------------------------*/
Mat::Damage::Damage() : params_(nullptr), plastic_step_(false) {}


/*----------------------------------------------------------------------*
 | copy-constructor (public)                                 dano 04/11 |
 *----------------------------------------------------------------------*/
Mat::Damage::Damage(Mat::PAR::Damage* params) : params_(params), plastic_step_(false) {}


/*----------------------------------------------------------------------*
 | pack (public)                                             dano 04/11 |
 *----------------------------------------------------------------------*/
void Mat::Damage::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  // in case we are in post-process mode
  if (params_ != nullptr) matid = params_->id();
  add_to_pack(data, matid);

  add_to_pack(data, strainpllast_);
  add_to_pack(data, backstresslast_);

  add_to_pack(data, strainbarpllast_);
  add_to_pack(data, isohardvarlast_);
  add_to_pack(data, damagelast_);
  add_to_pack(data, failedlast_);

  add_to_pack(data, plastic_step_);
}


/*----------------------------------------------------------------------*
 | unpack (public)                                           dano 04/11 |
 *----------------------------------------------------------------------*/
void Mat::Damage::unpack(Core::Communication::UnpackBuffer& buffer)
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
        params_ = static_cast<Mat::PAR::Damage*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }

  // history data
  extract_from_pack(buffer, strainpllast_);
  extract_from_pack(buffer, backstresslast_);
  extract_from_pack(buffer, strainbarpllast_);
  extract_from_pack(buffer, isohardvarlast_);
  extract_from_pack(buffer, damagelast_);
  extract_from_pack(buffer, failedlast_);

  strainplcurr_ = strainpllast_;
  backstresscurr_ = backstresslast_;
  strainbarplcurr_ = strainbarpllast_;
  isohardvarcurr_ = isohardvarlast_;
  damagecurr_ = damagelast_;
  failedcurr_ = failedlast_;

  extract_from_pack(buffer, plastic_step_);
}


/*---------------------------------------------------------------------*
 | initialise / allocate internal stress variables (public)      04/11 |
 *---------------------------------------------------------------------*/
void Mat::Damage::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  // initialise history variables

  // set all history variables to zero
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> emptyvect(Core::LinAlg::Initialization::zero);
  strainpllast_.resize(numgp);
  strainplcurr_.resize(numgp);

  backstresslast_.resize(numgp);
  backstresscurr_.resize(numgp);

  strainbarpllast_.resize(numgp);
  strainbarplcurr_.resize(numgp);

  isohardvarlast_.resize(numgp);
  isohardvarcurr_.resize(numgp);

  damagelast_.resize(numgp);
  damagecurr_.resize(numgp);

  failedlast_.resize(numgp);
  failedcurr_.resize(numgp);

  for (int i = 0; i < numgp; i++)
  {
    strainpllast_.at(i) = emptyvect;
    strainplcurr_.at(i) = emptyvect;

    backstresslast_.at(i) = emptyvect;
    backstresscurr_.at(i) = emptyvect;

    strainbarpllast_.at(i) = 0.0;
    strainbarplcurr_.at(i) = 0.0;

    isohardvarlast_.at(i) = 0.0;
    isohardvarcurr_.at(i) = 0.0;

    damagelast_.at(i) = 0.0;
    damagecurr_.at(i) = 0.0;

    failedlast_.at(i) = false;
    failedcurr_.at(i) = false;
  }
  if (params_->sigma_y_.size() != params_->strainbar_p_ref_.size())
    FOUR_C_THROW("Samples have to fit to each other!");

  if (params_->sigma_y_.size() < 1)
    FOUR_C_THROW("You have to provide at least one stress-strain pair!");

  if (abs(params_->strainbar_p_ref_[0]) > 1.e-20)
    FOUR_C_THROW("the first plastic strain value must be zero!");

  for (std::size_t i = 1; i < params_->strainbar_p_ref_.size(); ++i)
  {
    if (params_->strainbar_p_ref_[i] < params_->strainbar_p_ref_[i - 1])
      FOUR_C_THROW("plastic strain values have to be in ascending order!");
    if (params_->sigma_y_[i] < params_->sigma_y_[i - 1])
      FOUR_C_THROW("yield stress values have to be in ascending order!");
  }
  isinit_ = true;
  return;

}  // setup()


/*---------------------------------------------------------------------*
 | update internal stress variables (public)                dano 04/11 |
 *---------------------------------------------------------------------*/
void Mat::Damage::update()
{
  // make current values at time step tlast+1 to values of last step tlast
  strainpllast_ = strainplcurr_;
  backstresslast_ = backstresscurr_;

  strainbarpllast_ = strainbarplcurr_;
  isohardvarlast_ = isohardvarcurr_;
  damagelast_ = damagecurr_;
  for (std::size_t gp = 0; gp < failedlast_.size(); ++gp)
  {
    if (failedcurr_.at(gp))
    {
      failedlast_.at(gp) = true;
    }
  }

  // get the size of the vector
  // (use the last vector, because it includes latest results, current is empty)
  const int histsize = strainpllast_.size();
  strainplcurr_.resize(histsize);
  backstresscurr_.resize(histsize);

  strainbarplcurr_.resize(histsize);
  isohardvarcurr_.resize(histsize);
  damagecurr_.resize(histsize);
  failedcurr_.resize(histsize);

  const Core::LinAlg::Matrix<NUM_STRESS_3D, 1> emptyvec(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < histsize; i++)
  {
    strainplcurr_.at(i) = emptyvec;
    backstresscurr_.at(i) = emptyvec;

    strainbarplcurr_.at(i) = 0.0;
    isohardvarcurr_.at(i) = 0.0;
    damagecurr_.at(i) = 0.0;
    failedcurr_.at(i) = false;
  }

  return;
}  // update()


//  evaluate material (public)
void Mat::Damage::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  const Core::LinAlg::Matrix<3, 3> defgrd_mat = Core::LinAlg::make_matrix_view(*defgrad);
  const Core::LinAlg::Matrix<6, 1> glstrain_mat =
      Core::LinAlg::make_strain_like_voigt_matrix(glstrain);

  Core::LinAlg::Matrix<6, 1> stress_view = Core::LinAlg::make_stress_like_voigt_view(stress);
  Core::LinAlg::Matrix<6, 6> cmat_view = Core::LinAlg::make_stress_like_voigt_view(cmat);

  // in case kinematic hardening is ignored, use implementation according to de
  // Souza Neto, Computational Methods for Plasticity
  if ((params_->kinhard_ == 0.0) and (params_->kinhard_rec_ == 0.0) and (params_->hardexpo_ == 0.0))
    evaluate_simplified_lemaitre(
        &defgrd_mat, &glstrain_mat, params, &stress_view, &cmat_view, gp, eleGID);
  // in case full Lemaitre material model is considered, i.e. including
  // kinematic hardening, use implementation according to Doghri
  else
    evaluate_full_lemaitre(
        &defgrd_mat, &glstrain_mat, params, &stress_view, &cmat_view, gp, eleGID);
}  // Evaluate


// evaluate material for pure isotropic hardening
void Mat::Damage::evaluate_simplified_lemaitre(const Core::LinAlg::Matrix<3, 3>* defgrd,
    const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* linstrain,  // linear strain vector
    const Teuchos::ParameterList& params,            // parameter list for communication & HISTORY
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* stress,  // 2nd PK-stress
    Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>* cmat,  // material stiffness matrix
    const int gp, const int eleGID)
{
  if (gp == -1) FOUR_C_THROW("no Gauss point number provided in material");
  if (eleGID == -1) FOUR_C_THROW("no element provided in material");

  // get material parameters
  // Young's modulus
  double young = params_->youngs_;
  // Poisson's ratio
  double nu = params_->poissonratio_;
  // damage threshold
  double strainbar_p_D = params_->epsbarD_;
  // newton tolerance
  double newton_tolerance = params_->abstol_;

  // calculate some more parameters
  // lame constant / shear modulus parameter mu == G
  double G = young / (2.0 * (1.0 + nu));
  // bulk modulus bulk = E /( 3 ( 1 - 2 nu) )= lambda + 2/3 * mu
  double bulk = young / (3.0 * (1.0 - 2.0 * nu));

  // build Cartesian identity 2-tensor I_{AB}
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> id2(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++) id2(i) = 1.0;

  // linstrain (in): independent variable passed from the element
  //  strain^p: evolution is determined by the flow rule, history variable
  //  strain^e: definition of additive decomposition:
  //  strain^e = strain - strain^p
  // REMARK: stress-like 6-Voigt vector
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> strain(*linstrain);

  //---------------------------------------------------------------------------
  // elastic predictor (trial values: consider old damage and/or plasticity)
  //---------------------------------------------------------------------------

  // -------------------------------- old damage internal state variables

  // to consider damage in the present material model we introduce the
  // '~'-operator to indicate UNDAMAGED values (e.g. s^{~,trial}_{n+1}) in
  // contrast to DAMAGED values (e.g. s^{trial}_{n+1})

  // set trial damage variable to old one
  // D^{trial}_{n+1} = D_n
  double damage = damagelast_.at(gp);
  bool failed = failedlast_.at(gp);

  // get old damaged isotropic hardening variable (scalar-valued)
  // R^{trial}_{n+1} = R_n
  double Rplast = isohardvarlast_.at(gp);
  if (isohardvarlast_.at(gp) < 0.0)
  {
    FOUR_C_THROW("damaged isotropic hardening variable has to be equal to or greater than zero!");
  }

  // get old integrity: omega_n = 1 - D_n
  double omegaold = 1.0 - damage;
  // The maximum damage before the gausspoint is regarded as failed:
  double omegamin = 1.e-8;

  // ------------------------------------------------ old plastic strains

  // plastic strain vector
  // strain^{p,trial}_{n+1} = strain^p_n
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> strain_p(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < NUM_STRESS_3D; i++) strain_p(i, 0) = strainpllast_.at(gp)(i, 0);

  // get old accumulated/equivalent plastic strain  (scalar-valued)
  // astrain^{p,trial}_{n+1} = astrain^p_n
  double strainbar_p = strainbarpllast_.at(gp);
  if (strainbarpllast_.at(gp) < 0.0)
    FOUR_C_THROW("accumulated plastic strain has to be equal to or greater than zero!");

  // --------------------------------------------------- physical strains
  // convert engineering shear components into physical components
  // input strain is given in Voigt-notation

  // convert engineering shear component (in) into physical component
  for (int i = 3; i < NUM_STRESS_3D; ++i) strain(i) /= 2.0;
  for (int i = 3; i < NUM_STRESS_3D; ++i) strain_p(i) /= 2.0;

  // ----------------------------------------------- elastic trial strain
  // assume load step is elastic
  // strain^{e,trial}_{n+1} = strain_n+1 - strain^p_n
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> trialstrain_e(Core::LinAlg::Initialization::uninitialized);
  trialstrain_e.update(1.0, strain, 0.0);
  trialstrain_e.update((-1.0), strain_p, 1.0);
  // volumetric strain
  // trace of strain vector
  double tracestrain = trialstrain_e(0) + trialstrain_e(1) + trialstrain_e(2);
  // volstrain = 1/3 . tr( strain ) . Id
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> volumetricstrain(
      Core::LinAlg::Initialization::uninitialized);
  volumetricstrain.update((tracestrain / 3.0), id2, 0.0);

  // deviatoric strain
  // devstrain^e = strain^e - volstrain^e
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> devstrain(Core::LinAlg::Initialization::uninitialized);
  devstrain.update(1.0, trialstrain_e, 0.0);
  devstrain.update(-1.0, volumetricstrain, 1.0);

  // --------------------------------------------- trial undamaged stress

  // undamaged scalar-valued pressure
  // p^{~} = bulk . tr( strain )
  double p_tilde = bulk * tracestrain;

  // deviatoric stress^{~} = 2 . G . devstrain
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> devstress_tilde(
      Core::LinAlg::Initialization::uninitialized);
  devstress_tilde.update(2.0 * G, devstrain);
  // be careful for shear stresses (sigma_12)
  // in Voigt-notation the shear strains have to be scaled with 1/2
  // normally done in the material tangent (cf. id4sharp)

  // --------------- trial (undamaged) elastic von Mises effective stress

  // q^{~,trial}_{n+1} := q(s^{trial}_{n+1}) / (1-D_n) = \sqrt{ 3 . J2 } / (1-D_n)
  //                    = sqrt{3/2} . || s^{trial}_{n+1} || / (1-D_n)
  // J2 = 1/2 (s11^2 + s22^2 + s33^2 + 2 . s12^2 + 2 . s23^2 + 2 . s13^2)
  double J2 = 0.0;
  J2 = 1.0 / 2.0 *
           (devstress_tilde(0) * devstress_tilde(0) + devstress_tilde(1) * devstress_tilde(1) +
               devstress_tilde(2) * devstress_tilde(2)) +
       devstress_tilde(3) * devstress_tilde(3) + devstress_tilde(4) * devstress_tilde(4) +
       devstress_tilde(5) * devstress_tilde(5);
  double q_tilde = 0.0;
  q_tilde = sqrt(3.0 * J2);

  // initialise final (damaged) deviatoric stresses
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> devstress(Core::LinAlg::Initialization::zero);

  //---------------------------------------------------------------------------
  // check plastic admissibility, Phi<=0 is admissble
  //---------------------------------------------------------------------------

  // ----------------------------------------------- trial yield function

  // initialise yield stress and its derivative
  double Hiso = 0.0;
  double sigma_y = 0.0;

  // --------------------------------------------------- damage threshold
  // Below damage threshold: current strainbar_p < strainbar_p_D
  if (strainbar_p < strainbar_p_D)
  {
    // --> no damage evolution: damage_{n+1} = damage_n == 0

    // calculate the isotropic hardening modulus with old plastic strains
    // Hiso = dsigma_y / d astrain^p
    Hiso = get_iso_hard_at_strainbarnp(params_, strainbar_p);

    // calculate the uniaxial yield stress out of samples
    sigma_y = get_sigma_y_at_strainbarnp(params_, strainbar_p);
  }
  else  // current strainbar_p > strainbar_p_D
  {
    // calculate the uniaxial yield stress out of samples
    sigma_y = get_sigma_y_at_strainbarnp(params_, Rplast);
  }

  // calculate the yield function
  // Phi = \sqrt{ 3.0 . J2 } - sigma_y = q - sigma_y
  // with trial values: Phi_trial = q_trial^{~} - sigma_y and Dgamma == 0
  double Phi_trial = q_tilde - sigma_y;

  // --------------------------------------------------------- initialise

  // if trial state is violated, i.e. it's a plastic load step, there are two
  // possible states covered by bool active_plasticity
  bool active_plasticity = false;
  // incremental plastic multiplier Delta gamma
  double Dgamma = 0.0;
  // damage energy release rate Y
  double energyrelrate = 0.0;
  // calculate derivative of engergy release rate Ytan w.r.t. Dgamma
  double Ytan = 0.0;
  // integrity
  // omega = 1 - D
  double omega = omegaold;  // if not damaged, omega == 1.0
  // flag indicating if damage evolution takes place or not
  bool damevolution = false;

  // unit flow vector Nbar (Prandtl-Reuss)
  // (using s_n+1^trial for undamaged, and s_n+1 for damaged load step)
  // Nbar = ( s^{trial}_{n+1} / || s^{trial}_{n+1} || )
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Nbar(Core::LinAlg::Initialization::zero);

  // flow vector N (Prandtl-Reuss)
  // (using the updated deviatoric stress s_n+1, no longer s_n+1^trial)
  // N = sqrt{3/2} / (1 - D_{n+1}) . ( s_{n+1} / || s_{n+1} || )
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> N(Core::LinAlg::Initialization::zero);

  //---------------------------------------------------------------------------
  // IF consistency condition is violated, i.e. plastic load step use return-mapping
  // (Phi_trial > 0.0, Dgamma >= 0.0)
  //---------------------------------------------------------------------------
  if (Phi_trial > 1.0e-08 and !failed)
  {
    // ------------------------------------------------ damage threshold
    // Below damage threshold: current strainbar_p < strainbar_p_D
    if (strainbar_p < strainbar_p_D)
    {
      // no damage evolution -> omega = 1 -D = 1 - 0 = 1
      omega = 1.0;

      // ------------------------------------------------------- return-mapping
      // local Newton-Raphson

      // initialise
      const int itermax = 50;  // max. number of iterations

      // new data
      auto residuumAndJacobianNoDamage = [&](double Dgamma)
      { return residuum_and_jacobian_no_damage(params_, Dgamma, strainbar_p, q_tilde); };

      Dgamma = Core::Utils::solve_local_newton(
          residuumAndJacobianNoDamage, 0.0, newton_tolerance, itermax);
      strainbar_p = strainbarpllast_.at(gp) + Dgamma / omega;

      // check if newest astrain^{p,m}_{n+1} is still smaller than threshold
      if (strainbar_p < strainbar_p_D)
      {
        // no damage occurs, i.e. current solution is correct, no return-map
        // considering damage is necessary
        damevolution = false;

        // --------------------------------------------------- plastic update
        // isotropic damage variable remains the same using D == 0 (omega=1)
        Rplast = isohardvarlast_.at(gp) + Dgamma;

        // Hiso = dsigma_y / d astrain^p_{n+1}
        Hiso = get_iso_hard_at_strainbarnp(params_, strainbar_p);

        // sigma_y = sigma_y(astrain^p_{n+1})
        sigma_y = get_sigma_y_at_strainbarnp(params_, strainbar_p);

        // ---------------------------------------------- update flow vectors
        // deviatoric stress norm || s^{trial}_{n+1} ||
        double devstress_tildenorm = 0.0;
        devstress_tildenorm =
            sqrt(devstress_tilde(0) * devstress_tilde(0) + devstress_tilde(1) * devstress_tilde(1) +
                 devstress_tilde(2) * devstress_tilde(2) +
                 2.0 * (devstress_tilde(3) * devstress_tilde(3) +
                           devstress_tilde(4) * devstress_tilde(4) +
                           devstress_tilde(5) * devstress_tilde(5)));

        // unit flow vector Nbar = s^{trial}_{n+1} / || s^{trial}_{n+1} ||
        Nbar.update((1.0 / devstress_tildenorm), devstress_tilde);

        // flow vector N = sqrt(3/2) . Nbar
        N.update((sqrt(3.0 / 2.0)), Nbar);

        // deviatoric stress
        // s = s_{n+1}^{trial} - 2 . G . Delta gamma . N
        const double facdevstress = (-2.0) * G * Dgamma;
        devstress.update(omega, devstress_tilde, facdevstress, N);

        // total stress
        // sigma_{n+1} = s_{n+1} + p_{n+1} . id2
        // pressure/volumetric stress no influence due to plasticity
        Damage::stress(p_tilde, devstress, *stress);

        // total strains
        // compute converged engineering strain components (Voigt-notation)
        // strain^p_{n+1} = strain^p_n + Dgamma . N
        strain_p.update(Dgamma, N, 1.0);

        // compute converged engineering strain components (Voigt-notation)
        for (int i = 3; i < NUM_STRESS_3D; ++i) strain_p(i) *= 2.0;

        // --------------------------------------------------- update history
        // plastic strain
        strainplcurr_.at(gp) = strain_p;

        // accumulated plastic strain
        strainbarplcurr_.at(gp) = strainbar_p;

        // update damaged isotropic hardening variable R_{n+1}
        isohardvarcurr_.at(gp) = Rplast;

        // update damage variable damage_{n+1}
        damagecurr_.at(gp) = damage;


      }  // (strainbar_p < strainbar_p_D)
      // updated strainbar_p not valid, recalculate step considering damage
      else  // (strainbar_p > strainbar_p_D)
      {
        // damage occurs, i.e. current solution is not correct, recalculate
        // load step considering damage
        damevolution = true;
      }  // (strainbar_p > strainbar_p_D)
    }  // no damage: (strainbar_p < strainbar_p_D)

    // ------------------------------------ threshold of trial step is exceeded
    // (strainbar_p > strainbar_p_D), i.e. damage evolves
    else
      damevolution = true;

    //-------------------------------------------------------------------
    // -------------------------------- return-mapping considering damage
    // damage has to be considered (strainbar_p > strainbar_p_D)
    //-------------------------------------------------------------------

    if (damevolution == true)
    {
      // ------------------------------------------------- return-mapping
      // local Newton-Raphson
      // ------------------------------- initial guess for Dgamma (12.49)

      // instead of initial guess Dgamma^{m=0} = 0.0 use perfectly plastic
      // solution with frozen yield surface at beginning of each load increment
      //
      // Dgamma^{m=0} = q^{~,trial} - sigma(Rplast_n) * omega_n / 3G
      Dgamma = omegaold * Phi_trial / (3.0 * G);

      // -------------------- initialise internal variables with Dgamma^0
      // Rplast = R^{p,m=0}_{n+1} = R^p_n + Dgamma
      Rplast = isohardvarlast_.at(gp) + Dgamma;

      const int itermax = 50;  // max. number of iterations
      auto residuumAndJacobianWithDamage = [&](double Dgamma)
      {
        return residuum_and_jacobian_with_damage(
            params_, Dgamma, Rplast, q_tilde, p_tilde, omegaold);
      };

      Dgamma = Core::Utils::solve_local_newton(
          residuumAndJacobianWithDamage, Dgamma, newton_tolerance, itermax);

      // Finally, calculate the important quantities
      Rplast = isohardvarlast_.at(gp) + Dgamma;
      sigma_y = get_sigma_y_at_strainbarnp(params_, Rplast);
      Hiso = get_iso_hard_at_strainbarnp(params_, Rplast);
      Ytan = -Hiso * sigma_y / (3.0 * G);
      omega = std::max(0.0, 3.0 * G / (q_tilde - sigma_y) * Dgamma);
      energyrelrate = -(sigma_y * sigma_y) / (6.0 * G) - p_tilde * p_tilde / (2.0 * bulk);
      // -------------------------- update hardening and damage variables

      // check if damage variable is acceptable
      // admissible values: (0 <= D < 1) or (1 >= omega > 0)
      // sanity check: omega < 1.0e-20
      if (omega < omegamin)
      {
        omega = omegamin;
        failed = true;
      }
      // update damage variable damage_{n+1}
      damage = 1.0 - omega;

      // --> damaged isotropic hardening variable has newest value (see L678)

      // final von Mises equivalent stress q_{n+1} = sqrt(3 J)
      // using consistency condition for admissible state
      // Phi_{n+1} == 0 = q / (1-D_{n+1}) - sigma_y(Rplast_{n+1})
      // --> q = sigma_y(Rplast_{n+1}) * omega_{n+1}
      // or alternatively: q = omega_n * q_tilde - 3.0 * G * Dgamma;
      double q = omega * sigma_y;

      // get damaged pressure
      double p = omega * p_tilde;

      // deviatoric stress
      // s_{n+1} = (1 - D_{n+1}) s_{n+1}^{trial}
      //           - 2 . G . Delta gamma . sqrt(3/2) . s^{~,trial}/||s^{~,trial}||
      // or alternative
      // s_{n+1} = s_{n+1}^{~,trial} * q / q_tilde
      // with deviatoric stress^{~} = 2 . G . devstrain
      devstress.update((q / q_tilde), devstress_tilde);
      // alternatively with identical results:
      // s_{n+1} = [omega - 3G Dgamma / q_tilde] . s_{n+1}^{~,trial}

      // total stress
      // sigma_{n+1} = s_{n+1} + p_{n+1} . id2
      // pressure/volumetric stress no influence due to plasticity
      Damage::stress(p, devstress, *stress);

      // -------------------------------------------- update flow vectors

      // unit flow vector Nbar = s_{n+1} / || s_{n+1} ||
      double devstressnorm = 0.0;
      devstressnorm = sqrt(devstress(0) * devstress(0) + devstress(1) * devstress(1) +
                           devstress(2) * devstress(2) +
                           2.0 * (devstress(3) * devstress(3) + devstress(4) * devstress(4) +
                                     devstress(5) * devstress(5)));
      Nbar.update((1.0 / devstressnorm), devstress);

      // flow vector N = sqrt(3/2) . Nbar . 1/omega (Box 12.3 (iv))
      N.update((sqrt(3.0 / 2.0) / omega), Nbar);

      // total strains
      // strain^p_{n+1} = strain^p_n + Dgamma . N
      // or alternatively
      //   strain^p_{n+1} = strain_{n+1} - strain^e_{n+1}
      strain_p.update(Dgamma, N, 1.0);

      // compute converged engineering strain components (Voigt-notation)
      for (int i = 3; i < NUM_STRESS_3D; ++i) strain_p(i) *= 2.0;

      // ------------------------------------------------- update history
      // plastic strain
      strainplcurr_.at(gp) = strain_p;

      // accumulated plastic strain
      strainbarplcurr_.at(gp) = strainbar_p;

      // update damaged isotropic hardening variable
      isohardvarcurr_.at(gp) = Rplast;

      // update damage variable damage_{n+1}
      damagecurr_.at(gp) = damage;

      // update failure flag;
      failedcurr_.at(gp) = failed;


    }  // damage evolution has to be considered, damage threshold exceeded

  }  // plastic corrector

  //------------------------------------------------------------------------
  // ELSE: elastic step (Phi_trial <= 0.0, Dgamma = 0.0)
  //------------------------------------------------------------------------
  else  // (Phi_trial <= 0.0)
  {
    // -------------------------- update stress using damaged elastic law
    // get damaged pressure
    // p = omega_{n+1} . p_tilde
    double p = p_tilde * omega;

    // get damaged deviatoric stresses
    // s_{n+1} = omega_{n+1} . s^{trial}_{n+1}
    devstress.update(omega, devstress_tilde);

    // result vectors of time step n+1 = omega . trial state vectors
    // sigma^e_n+1 = omega . sigma^(e,trial)_n+1
    //             = omega . (s^{trial}_{n+1} + p . id2)
    Damage::stress(p, devstress, *stress);

    // --------------------------------------------------------- update history
    // constant values for
    //  - plastic strains
    //  - accumulated (un)damaged plastic strains
    //  - stress

    // as current history vectors are set to zero in update(), the old values
    // need to be set instead, otherwise no constant plastic values are possible
    strainplcurr_.at(gp) = strainpllast_.at(gp);
    strainbarplcurr_.at(gp) = strainbarpllast_.at(gp);
    isohardvarcurr_.at(gp) = isohardvarlast_.at(gp);
    damagecurr_.at(gp) = damagelast_.at(gp);
    failedcurr_.at(gp) = failedlast_.at(gp);

  }  // elastic step

  //---------------------------------------------------------------------------
  // --------------------------------- consistent elastoplastic tangent modulus
  //---------------------------------------------------------------------------

  // using an associative flow rule: C_ep is symmetric
  // ( generally C_ep is nonsymmetric )
  // if Phi^trial = 0: two tangent stress-strain relations exist
  // plastic loading --> C == C_ep
  active_plasticity = (Dgamma > 0.0);

  // using an associative flow rule: C_ep is symmetric
  // ( generally C_ep is nonsymmetric )
  setup_cmat_elasto_plastic(*cmat, eleGID, Dgamma, G, bulk, p_tilde, q_tilde, energyrelrate, Ytan,
      sigma_y, Hiso, Nbar, gp, damevolution, active_plasticity);
}  // EvaluateSimplifiedLemaitre()



// derivative of yield stress (isotropic hardening modulus)
double Mat::Damage::get_iso_hard_at_strainbarnp(
    Mat::PAR::Damage* matparameter, const double strainbar_p)
{
  // Hiso = d sigma_y / d astrain^p_{n+1}
  double Hiso = 0.0;

  // extract vectors of samples
  const std::vector<double> strainbar_p_ref = matparameter->strainbar_p_ref_;
  const std::vector<double> sigma_y_ref = matparameter->sigma_y_;
  double hardexpo = matparameter->hardexpo_;  // hardening exponent
  double sigma_yinfty =
      matparameter->sathardening_;  // saturation hardening (=sigma_ysat - sigma_y0)
  // how many samples are available
  double samplenumber = sigma_y_ref.size();

  Hiso = sigma_yinfty * exp(-hardexpo * strainbar_p) * hardexpo;

  // loop over all samples
  for (int i = 1; i < samplenumber; ++i)
  {
    // strain^{p}_{n+1} < strain^{p}_ref^[i]?
    if (strainbar_p < strainbar_p_ref[i])
    {
      //         sigma_y_n - sigma_y^{i-1}
      // Hiso =  ---------------------------------------
      //        astrain^{p,i}_ref - astrain^{p,i-1}_ref
      Hiso += (sigma_y_ref[i] - sigma_y_ref[i - 1]) / (strainbar_p_ref[i] - strainbar_p_ref[i - 1]);
      break;
    }  // load is plastic, hardening can occur
  }  // loop over samples

  // return current isotropic hardening modulus
  return Hiso;
}  // get_iso_hard_at_strainbarnp()


// calculate yield stress from (sigma_y-astrain^p)-samples
double Mat::Damage::get_sigma_y_at_strainbarnp(
    Mat::PAR::Damage* matparameter, const double strainbar_p)
// current accumulated strain, in case of dependent hardening
// if damage!=0: isotropic hardening internal variable
{
  // extract vectors of samples
  const std::vector<double> sigma_y_ref = matparameter->sigma_y_;
  const std::vector<double> strainbar_p_ref = matparameter->strainbar_p_ref_;
  double hardexpo = matparameter->hardexpo_;  // hardening exponent
  double sigma_yinfty =
      matparameter->sathardening_;  // saturation hardening (=sigma_ysat - sigma_y0)
  // how many samples are available
  double samplenumber = sigma_y_ref.size();

  // kappa = sigma_yinfty . (1 - exp (-delta . astrain))
  double kappa = sigma_yinfty * (1.0 - exp(-hardexpo * strainbar_p));

  // loop over samples (starting from 1, since if it is smaller than 0,
  // we take the initial yield stress
  for (int i = 1; i < samplenumber; ++i)
  {
    // current strains are <= strainbar_p_ref_max
    if (strainbar_p < strainbar_p_ref[i])
    {
      // astrain^{p,i-1}_ref < astrain^{p}_{n+1} < astrain^{p,i}_ref
      // sigma_y_{n+1} = sigma_y^i +
      //                                        sigma_y^i - sigma_y^{i-1}
      // + (astrain^p_{n+1} - astrain^{p,i-1}) ---------------------------------------
      //                                      astrain^{p,i}_ref - astrain^{p,i-1}_ref
      return kappa + sigma_y_ref[i - 1] +
             (strainbar_p - strainbar_p_ref[i - 1]) * (sigma_y_ref[i] - sigma_y_ref[i - 1]) /
                 (strainbar_p_ref[i] - strainbar_p_ref[i - 1]);
    }
  }  // loop over all samples

  // return last yield stress in table if strain is larger than largest strain in table
  return kappa + sigma_y_ref[samplenumber - 1];

}  // get_sigma_y_at_strainbarnp()

// This is the residual and tangent calculation without consideration of damage
std::pair<double, double> Mat::Damage::residuum_and_jacobian_no_damage(
    Mat::PAR::Damage* matparameter, double Dgamma, double accplstrain_last, double q_tilde)
{
  const double G =
      matparameter->youngs_ / (2.0 * (1.0 + matparameter->poissonratio_));  // shear modulus, mu=G
  // plastic material data

  double accplstrain_curr = accplstrain_last + Dgamma;  //  / omega;

  const double y_d = get_sigma_y_at_strainbarnp(matparameter, accplstrain_curr);
  const double dy_d_dgamma = get_iso_hard_at_strainbarnp(matparameter, accplstrain_curr);

  double residuum = q_tilde - 3. * G * Dgamma - y_d;
  double tangent = -3. * G - dy_d_dgamma;
  return {residuum, tangent};
}  // residuum_and_jacobian_no_damage


// This is the residual and tangent calculation including damage
std::pair<double, double> Mat::Damage::residuum_and_jacobian_with_damage(
    Mat::PAR::Damage* matparameter, double Dgamma, double isohardvarlast, double q_tilde,
    double p_tilde, double omegaold)
{
  // elastic material parameters
  const double young = matparameter->youngs_;
  const double nu = matparameter->poissonratio_;
  const double damexp = matparameter->damexp_;
  const double damden = matparameter->damden_;
  const double G = young / (2.0 * (1.0 + nu));  // shear modulus, mu=G
  const double bulk = young / (3.0 * (1.0 - 2.0 * nu));

  // plastic material data
  double Rplast = isohardvarlast + Dgamma;
  const double y_d = get_sigma_y_at_strainbarnp(matparameter, Rplast);
  const double dy_d_dgamma = get_iso_hard_at_strainbarnp(matparameter, Rplast);

  // damage properties
  // get derivative of energy release rate w.r.t. Dgamma
  // d(-energyrelrate) / dDgamma = - Hiso(Rplast^m) . sigma_y(Rplast^m) / (3 . G)
  const double Ytan = -dy_d_dgamma * y_d / (3.0 * G);
  // omega_{n+1} = 3G / (q_tilde - sigma_y) * Dgamma = 1 - D_{n+1}
  const double omega = std::max(0.0, 3.0 * G / (q_tilde - y_d) * Dgamma);
  // damage energy release rate only implicitly depending on Dgamma (12.47)
  const double energyrelrate = -(y_d * y_d) / (6.0 * G) - p_tilde * p_tilde / (2.0 * bulk);

  // compute residual function (12.48)
  // Res := F(Dgamma) = omega(Dgamma) - omega_n
  //                    + Dgamma / omega(Dgamma) . (-Y(Dgamma)/r)^s
  //                      . (q_tilde - sigma_y) / (3 G)
  // here: it is important NOT to use Dgamma^{m=0}=0 --> omega=0 --> '1/0'
  double residuum = omega - omegaold +
                    std::pow((-energyrelrate / damden), damexp) / ((3.0 * G) / (q_tilde - y_d));

  double tangent = (3.0 * G) / (q_tilde - y_d) +
                   (3.0 * G) / (q_tilde - y_d) * Dgamma * dy_d_dgamma / (q_tilde - y_d) -
                   dy_d_dgamma / (3.0 * G) * std::pow((-energyrelrate / damden), damexp) -
                   damexp * Ytan / (((3.0 * G) / (q_tilde - y_d)) * damden) *
                       std::pow((-energyrelrate / damden), (damexp - 1.0));
  return {residuum, tangent};
}  // residuum_and_jacobian_with_damage


/*----------------------------------------------------------------------*
 | evaluate full Lemaitre material model (public)            dano 11/13 |
 *----------------------------------------------------------------------*/
void Mat::Damage::evaluate_full_lemaitre(const Core::LinAlg::Matrix<3, 3>* defgrd,
    const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* linstrain,  // linear strain vector
    const Teuchos::ParameterList& params,            // parameter list for communication & HISTORY
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* stress,  // 2nd PK-stress
    Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>* cmat,  // material stiffness matrix
    const int gp, const int eleGID)
{
  // --------- full Lemaitre material model requires solution of five equations

  // 1) fulfil consistency condition
  // Phi = \sqrt(3 . J2(s_tilde_{n+1} - beta_{n+1})) - sigma_y(R_{n+1}) <= 0
  //
  // 2) determine the stress
  // sigma_{n+1} = (1-D) . C^e : (strain_n+1^{e,trial} - Dgamma . N_{n+1} )
  //
  // 3) determine back stress
  // beta_{n+1} = beta_n + (Hkin . N_tilde_{n+1} - Hkin_rec . beta_n) . Dgamma/(1 + Hkin_rec Dgamma)
  //
  // 4) determine scalar-valued damage
  // D_{n+1} = D_n + 1/(1 - D_{n+1}) . y(s_tilde) . Dgamma / (1 - D)
  if (gp == -1) FOUR_C_THROW("no Gauss point number provided in material");

  if (eleGID == -1) FOUR_C_THROW("no element provided in material");

  // get material parameters
  // Young's modulus
  double young = params_->youngs_;
  // Poisson's ratio
  double nu = params_->poissonratio_;
  // damage evolution law denominator r
  double damden = params_->damden_;
  // damage evolution law exponent s
  double damexp = params_->damexp_;
  // damage threshold
  double strainbar_p_D = params_->epsbarD_;
  // kinematic hardening modulus, also defined as Hkin = k_1 = a
  double Hkin = params_->kinhard_;
  // scalar-value variable describing recovery, also defined as Hkin_rec = k_2 = b
  double Hkin_rec = params_->kinhard_rec_;

  // calculate some more parameters
  // lame constant / shear modulus parameter mu == G
  double G = young / (2.0 * (1.0 + nu));
  // bulk modulus bulk = E /( 3 ( 1 - 2 nu) ) = lambda + 2/3 * mu
  double bulk = young / (3.0 * (1.0 - 2.0 * nu));

  // build Cartesian identity 2-tensor I_{AB}
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> id2(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++) id2(i) = 1.0;

  // linstrain (in): independent variable passed from the element
  //  strain^p: evolution is determined by the flow rule, history variable
  //  strain^e: definition of additive decomposition:
  //  strain^e = strain - strain^p
  // REMARK: stress-like 6-Voigt vector
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> strain(*linstrain);

  //---------------------------------------------------------------------------
  // elastic predictor (trial values: consider old damage and/or plasticity)
  //---------------------------------------------------------------------------

  // -------------------------------- old damage internal state variables

  // to consider damage in the present material model we introduce the
  // '~'-operator to indicate UNDAMAGED values (e.g. s^{~,trial}_{n+1}) in
  // contrast to DAMAGED values (e.g. s^{trial}_{n+1})


  // damaged accumulated or equivalent plastic strain (scalar-valued)
  // R^{trial}_{n+1} = R_n
  double Rplast = isohardvarlast_.at(gp);
  if (isohardvarlast_.at(gp) < 0.0)
  {
    std::cout << "Rplast am ele = " << eleGID << ": " << Rplast << std::endl;
    FOUR_C_THROW("damaged isotropic hardening variable has to be equal to or greater than zero!");
  }

  // set trial damage variable to old one
  // D^{trial}_{n+1} = D_n
  double damage = damagelast_.at(gp);
  bool failed = failedlast_.at(gp);
  // get old integrity: omega_n = 1 - D_n
  double omegaold = 1.0 - damage;
  double omega = omegaold;
  // The maximum damage before the gausspoint is regarded as failed:
  double omegamin = 1.e-8;

  // ------------------------------------------------ old plastic strains

  // plastic strain vector
  // strain^{p,trial}_{n+1} = strain^p_n
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> strain_p(Core::LinAlg::Initialization::uninitialized);
  strain_p.update(strainpllast_.at(gp));

  // accumulated or equivalent plastic strain (scalar-valued)
  // astrain^{p,trial}_{n+1} = astrain^p_n
  double strainbar_p = strainbarpllast_.at(gp);
  if (strainbarpllast_.at(gp) < 0.0)
    FOUR_C_THROW("accumulated plastic strain has to be equal to or greater than zero!");

  // ------------------------------------------------ old back stress
  // beta^{trial}_{n+1} = beta_n
  // beta is a deviatoric tensor
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> beta(Core::LinAlg::Initialization::uninitialized);
  beta.update(backstresslast_.at(gp));

  // --------------------------------------------------- physical strains
  // convert engineering shear components into physical components
  // input strain is given in Voigt-notation

  // convert engineering shear component (in) into physical component
  for (int i = 3; i < NUM_STRESS_3D; ++i) strain(i) /= 2.0;
  for (int i = 3; i < NUM_STRESS_3D; ++i) strain_p(i) /= 2.0;

  // ----------------------------------------------- elastic trial strain
  // assume load step is elastic
  // strain^{e}_{n+1}
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> strain_e(Core::LinAlg::Initialization::zero);

  // strain^{e,trial}_{n+1} = strain_n+1 - strain^p_n
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> trialstrain_e(Core::LinAlg::Initialization::uninitialized);
  trialstrain_e.update(1.0, strain, (-1.0), strain_p);

  // volumetric strain
  // trace of strain vector
  double tracestrain = trialstrain_e(0) + trialstrain_e(1) + trialstrain_e(2);
  // volstrain = 1/3 . tr( strain ) . Id
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> volumetricstrain(
      Core::LinAlg::Initialization::uninitialized);
  volumetricstrain.update((tracestrain / 3.0), id2, 0.0);

  // deviatoric strain
  // devstrain^e = strain^e - volstrain^e
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> devstrain(Core::LinAlg::Initialization::uninitialized);
  devstrain.update(1.0, trialstrain_e, (-1.0), volumetricstrain);

  // --------------------------------------------- trial undamaged stress

  // undamaged scalar-valued pressure
  // p^{~} = bulk . tr( strain )
  double p_tilde = bulk * tracestrain;

  // undamaged deviatoric stress^{~} = 2 . G . devstrain
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> devstress_tilde(
      Core::LinAlg::Initialization::uninitialized);
  devstress_tilde.update((2.0 * G), devstrain);
  // be careful for shear stresses (sigma_12)
  // in Voigt-notation the shear strains have to be scaled with 1/2
  // normally done in the material tangent (cf. id4sharp)

  // -----------------------------------------  relative effective stress
  // eta^{~,trial}_{n+1} = s^{~,trial}_{n+1} - beta^{trial}_{n+1}
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> eta_tilde(Core::LinAlg::Initialization::zero);
  rel_stress(devstress_tilde, beta, eta_tilde);

  // --------------- trial (undamaged) elastic von Mises effective stress

  // q^{~,trial}_{n+1} := q(s^{trial}_{n+1}) / (1 - D_n) = \sqrt{ 3 . J2 } / (1-D_n)
  //                    = sqrt{3/2} . || s^{trial}_{n+1} || / (1 - D_n)
  // J2 = 1/2 (s11^2 + s22^2 + s33^2 + 2 . s12^2 + 2 . s23^2 + 2 . s13^2)
  double J2 =
      0.5 * (devstress_tilde(0) * devstress_tilde(0) + devstress_tilde(1) * devstress_tilde(1) +
                devstress_tilde(2) * devstress_tilde(2)) +
      +devstress_tilde(3) * devstress_tilde(3) + devstress_tilde(4) * devstress_tilde(4) +
      devstress_tilde(5) * devstress_tilde(5);
  double q_tilde = sqrt(3.0 * J2);

  // ------ trial (undamaged) relative elastic von Mises effective stress

  // qbar^{~,trial}_{n+1} := sqrt{3/2} . || s^{~,trial}_{n+1} - beta^{trial}_{n+1} ||
  //                       = sqrt{3/2} . || eta^{~,trial}_{n+1} ||
  //                       = qbar(eta^{trial}_{n+1}) / (1 - D_n)
  //                       = \sqrt{ 3 . J2bar(eta) } / (1 - D_n)

  // J2bar = 1/2 (eta11^2 + eta22^2 + eta33^2 + 2 . eta12^2 + 2 . eta23^2 + 2 . eta13^2)
  double J2bar = 0.5 * (eta_tilde(0) * eta_tilde(0) + eta_tilde(1) * eta_tilde(1) +
                           eta_tilde(2) * eta_tilde(2)) +
                 eta_tilde(3) * eta_tilde(3) + eta_tilde(4) * eta_tilde(4) +
                 eta_tilde(5) * eta_tilde(5);
  double qbar_tilde = sqrt(3.0 * J2bar);


  // initialise final (damaged) deviatoric stresses
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> devstress(Core::LinAlg::Initialization::zero);

  // ----------- initialise variables required due to kinematic hardening

  // initialise variables calculated within local Newton and required also for
  // the damaged elastoplastic material tangent C^{ep}
  // factor
  double g = 0.0;
  // denominator
  double h_alg = 0.0;
  // s_N = 2 G . (Dgamma / omega)^2 . dy/ds_tilde : N_tilde
  double s_N = 0.0;
  // Nbetaold = N_tilde : beta_n
  double Nbetaold = 0.0;
  // initialise energy-release rate y := (-Y/r)^s
  double y = 0.0;
  // dy/dsigma_tilde
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> dy_dsigma_tilde(Core::LinAlg::Initialization::zero);
  // b_NbetaoldN = beta_n - 2/3 . (N_tilde : beta_n) . N_tilde
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> b_NbetaoldN(Core::LinAlg::Initialization::zero);
  // correction for difference (back stress - deviatoric stress)
  // c_s - c_beta (49)
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> c_s_b(Core::LinAlg::Initialization::zero);

  //---------------------------------------------------------------------------
  // check plastic admissibility, Phi<=0 is admissble
  //---------------------------------------------------------------------------

  // ----------------------------------------------- trial yield function

  // calculate damage threshold (according to de Souza Neto, p.483ff)

  // bool which decide if threshold is passed or not
  // current strainbar_p < strainbar_p_D
  double hardening_variable = (strainbar_p < strainbar_p_D) ? strainbar_p : Rplast;

  // --> no damage evolution: damage_{n+1} = damage_n == 0

  // calculate the uniaxial yield stress out of samples
  double sigma_y = get_sigma_y_at_strainbarnp(params_, hardening_variable);

  // compute the derivative of the isotropic hardening kappa(R^p) w.r.t. R^p
  // dkappa_dR = sigma_infty . (- exp (-delta . R) ) . (-delta)
  //           = sigma_infty . exp (-delta . R) . delta
  double Hiso = get_iso_hard_at_strainbarnp(params_, hardening_variable);

  // calculate the yield function
  // Phi = \sqrt{ 3.0 . J2 } - sigma_y = qbar_tilde - sigma_y
  // with trial values: Phi_trial = qbar{~,trial} - sigma_y and Dgamma == 0
  double Phi_trial = qbar_tilde - sigma_y;


  // --------------------------------------------------------- initialise

  // if trial state is violated, i.e. it's a plastic load step, there are 2
  // possible states indicated by flag active_plasticity
  bool active_plasticity = false;
  // incremental plastic multiplier Delta gamma
  double Dgamma = 0.0;

  // unit flow vector Nbar (Prandtl-Reuss)
  // (using s_n+1^trial for undamaged, and s_n+1 for damaged load step)
  // Nbar = ( eta^{~,trial}_{n+1} / || eta^{~,trial}_{n+1} || )
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Nbar(Core::LinAlg::Initialization::zero);

  // flow vector N_tilde according to Doghri and their explicit updating
  // N_tilde = 3/2 . (s_tilde - beta) / qbar_tilde
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> N_tilde(Core::LinAlg::Initialization::zero);

  //---------------------------------------------------------------------------
  // IF consistency condition is violated, i.e. plastic load step use return-mapping
  // (Phi_trial > 0.0, Dgamma >= 0.0)
  //---------------------------------------------------------------------------
  if (Phi_trial > 1.0e-08 and !failed)  // if (Phi_trial > 0.0)
  {
    // deviatoric stress norm || eta^{~}_{n+1} ||
    double eta_tildenorm = 0.0;
    eta_tildenorm = sqrt(eta_tilde(0) * eta_tilde(0) + eta_tilde(1) * eta_tilde(1) +
                         eta_tilde(2) * eta_tilde(2) +
                         2.0 * (eta_tilde(3) * eta_tilde(3) + eta_tilde(4) * eta_tilde(4) +
                                   eta_tilde(5) * eta_tilde(5)));

    // undamaged unit flow vector
    // Nbar = eta^{trial}_{n+1} / || eta^{trial}_{n+1} || = eta_{n+1} / || eta_{n+1} ||
    // using the fact that eta_{n+1} is proportional to eta^{trial}_{n+1}
    Nbar.update((1.0 / eta_tildenorm), eta_tilde);

    // undamaged flow vector N_tilde according to Doghri
    // N_tilde^{trial} = 3/2 . (s_tilde^{trial} - beta_n) / sqrt{ 3/2 .
    //                   (s_tilde^{trial} - beta_n) : (s_tilde^{trial} - beta_n) }
    //                 = 3/2 . (s_tilde^{trial} - beta_n)
    //                   / ( sqrt{ 3/2 } . || (s_tilde^{trial} - beta_n) || )
    //                 = sqrt{ 3/2 } . (s_tilde^{trial} - beta_n)
    //                   /  || (s_tilde^{trial} - beta_n) || = sqrt{ 3/2 } . Nbar
    N_tilde.update((sqrt(3.0 / 2.0)), Nbar);

    // ------------------------------------------------ calculate corrections

    // 1.step) compute the corrections over the elastic trial predictor
    // 2.step) if residuals > tolerance criteria compute
    //         --> calculate the corrections in a loop over plastic iterations
    //             m until criteria is fulfilled, i.e. r_i < tolerance criteria

    // -------------------- calculate corrections for elastic trial predictor

    // calculate N : beta_n
    double N_tildebeta = 0.0;
    for (int i = 0; i < NUM_STRESS_3D; ++i)
      N_tildebeta += N_tilde(i, 0) * backstresslast_.at(gp)(i, 0);  // (6x1)(6x1)

    // calculate trial value of hardening modulus (27)
    // h_trial = 3 . G + (1 - D_n) . (dkappa/dR + 3/2 . a - b . N : beta_n)
    // dkappa/dR = Hiso with R = R_n, Dgamma = 0
    double h_trial = 3.0 * G + omegaold * (Hiso + 3.0 / 2.0 * Hkin - Hkin_rec * N_tildebeta);

    // correction for the internal hardening variable R (57)
    // R' = c_R/Dt:  c_R = (1-D_{n}) . Phi^{trial}_{n+1} / h^{trial}_{n+1}
    double c_R = omegaold * Phi_trial / h_trial;

    // correction for accumulated plastic strain (58a)
    // astrain' = c_astrain/Dt: c_astrain = c_R / (1 - D_n) = c_R / omega_n
    double c_strainbar = c_R / omegaold;

    // corrections for effective, undamaged deviatoric stress (58b)
    // s = c_s/Dt: c_s = -2G . N_tilde^{trial} . c_astrain
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> c_s(Core::LinAlg::Initialization::uninitialized);
    double fac_cs = -2.0 * G * c_strainbar;
    c_s.update(fac_cs, N_tilde);

    // corrections for back stress (58c)
    // beta' = c_beta/Dt: c_beta = (Hkin . N_tilde^{trial} - Hkin_rec . beta_n) . c_R
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> c_beta(Core::LinAlg::Initialization::uninitialized);
    c_beta.update(Hkin, N_tilde);
    c_beta.update((-Hkin_rec), beta, 1.0);
    c_beta.scale(c_R);

    // corrections for damage variable
    // damage energy release rate
    // Y = -1/2(1-D)^2 sigma : C^e : sigma
    //   = - q^2 / [6 G . (1 - D)^2] - p^2 / [2 bulk . (1 - D)^2]
    // with q^{~}_{n+1} = q_{n+1} / (1 - D_{n+1})
    // and p^{~}_{n+1} = p^{~, trial} = p_{n+1}  / (1 - D_{n+1})
    double Y = -q_tilde * q_tilde / (6.0 * G) - p_tilde * p_tilde / (2.0 * bulk);
    // y := (Y^trial_{n+1} / r)^s
    y = std::pow((-Y / damden), damexp);

    // c_D = (Y^trial_{n+1} / r)^s . c_astrain
    double c_D = y * c_strainbar;


    // ------------------------------------------------- return-mapping

    // local Newton-Raphson

    const int itermax = 50;  // max. number of iterations
    int itnum = 0;           // iteration counter

    // start iteration with index m for local Newton
    while (true)
    {
      itnum++;
      // check for convergence

      // if not converged m > m_max
      if (itnum > itermax)
      {
        FOUR_C_THROW(
            "local Newton iteration did not converge after iteration {:3d}/{:3d}", itnum, itermax);
      }
      // else: continue loop m <= m_max

      // ------------------------- update variables within local Newton
      // c_(.)/dt = ()'
      // --> c_(.) == (.)^{m+1} - (.)^m
      //     --> (.)^{m+1} = (.)^m + c_(.)

      // isotropic hardening variable
      // R^{p,m+1} = R^{p,m}_{n+1} = R^p_n + c_R
      Rplast = isohardvarlast_.at(gp) + c_R;

      // (undamaged) accumulated plastic strain
      // astrain^{m+1}_{n+1} = astrain^m_n + c_astrain
      strainbar_p = strainbarpllast_.at(gp) + c_strainbar;

      // (undamaged, effective) deviatoric stress
      // s^{~,m+1}_{n+1} = s^{~,trial/m}_{n+1} + c_s
      devstress_tilde.update(1.0, c_s, 1.0);

      // back stress
      // beta^{m+1}_{n+1} = beta^{trial/m}_{n+1} + c_beta
      beta.update(1.0, c_beta, 1.0);

      // check damage threshold
      if (strainbar_p < strainbar_p_D)
      {
        damage = 0.0;
        omega = 1.0;
      }
      else
      {
        // damage
        // D^{m+1}_{n+1} = D^m_n + c_D
        damage = damagelast_.at(gp) + c_D;
        // update omega
        omega = 1.0 - damage;
      }

      // update plastic multiplier
      // Dgamma = c_R (gamma' = R' = c_R/Dt)
      Dgamma = c_R;

      // update q^{~}_{n+1}
      // q^{~}_{n+1} := q(s^{~}_{n+1}) = \sqrt{ 3 . J2(s^{~} }
      //              = sqrt{3/2} . || s^{~}_{n+1} ||
      // J2 = 1/2 (s11^2 + s22^2 + s33^2 + 2 . s12^2 + 2 . s23^2 + 2 . s13^2)
      J2 = 1.0 / 2.0 *
               (devstress_tilde(0) * devstress_tilde(0) + devstress_tilde(1) * devstress_tilde(1) +
                   devstress_tilde(2) * devstress_tilde(2)) +
           +devstress_tilde(3) * devstress_tilde(3) + devstress_tilde(4) * devstress_tilde(4) +
           devstress_tilde(5) * devstress_tilde(5);
      q_tilde = sqrt(3.0 * J2);

      // update energy release rate
      // here: Y := Y(s_tilde), i.e. depends ONLY on stresses, beta NOT included!
      //
      // Y = - [ \sqrt(3/2) . J2(sigma_tilde) ]^2 / (6 G) - p_tilde^2 / (2 bulk)
      // here we assume that Y depends only on deviatoric part
      // --> q_tilde(sigma_tilde) == q_tilde(s_tilde)
      //
      // Y = - [ q_tilde ]^2 / (6 G) - p_tilde^2 / (2 bulk)
      Y = -q_tilde * q_tilde / (6.0 * G) - p_tilde * p_tilde / (2.0 * bulk);
      // y == (-Y(s^{~,m}_{n+1}) / r)^s
      y = std::pow((-Y / damden), damexp);

      // ----------------- calculate derivatives of energy-release rate

      // d(-Y)/ds_tilde
      //   = d{[ q_tilde(s_tilde) ]^2 / (6 G) + p_tilde^2 / (2 bulk)}/ds_tilde
      //   = d{[ q_tilde(s_tilde) ]^2 / (6 G) + 0 }/ds_tilde
      //   = d{[ q_tilde(s_tilde) ]^2 }/ds_tilde / (6 G)
      //   = 2 . q_tilde(s_tilde) . sqrt(3/2) . s_tilde/|| s_tilde || / (6 G)
      //   = s_tilde / (2 G)

      // dy_ds_tilde = s . (-Y / r)^{s-1} . (1 / r) . s_tilde / (2 G)
      //             = s / (2 G . r) . (-Y / r)^{s-1} .s_tilde
      Core::LinAlg::Matrix<NUM_STRESS_3D, 1> dy_dstilde(
          Core::LinAlg::Initialization::uninitialized);
      double fac_dy_dstilde = damexp / (damden * 2.0 * G) * std::pow((-Y / damden), (damexp - 1.0));
      dy_dstilde.update(fac_dy_dstilde, devstress_tilde);

      // d(-Y)/dsigma_tilde
      //   = d{[ q_tilde(s_tilde) ]^2 }/ds_tilde / (6 G) +
      //     + d{ p_tilde^2 / (2 bulk)}/d (tr sigma_tilde)
      //   = s_tilde / (2 G) + p_tilde / bulk . I
      Core::LinAlg::Matrix<NUM_STRESS_3D, 1> dYneg_dsigma_tilde(
          Core::LinAlg::Initialization::uninitialized);
      dYneg_dsigma_tilde.update((1.0 / (2.0 * G)), devstress_tilde);
      for (int i = 0; i < 3; ++i) dYneg_dsigma_tilde(i) += bulk * p_tilde;

      // dy_dsigma_tilde = s . (-Y / r)^{s-1} . (1 / r)
      //                   . [ s_tilde / (2 G) + p_tilde / bulk . I ]
      //
      // dy_dsigma_tilde = s . (-Y / r)^{s-1} . (1 / r) . d(-Y)/dsigma_tilde
      double fac_dy_dsigma_tilde = damexp / damden * std::pow((-Y / damden), (damexp - 1.0));
      dy_dsigma_tilde.update(fac_dy_dsigma_tilde, dYneg_dsigma_tilde);

      // update effective stress
      // eta^{~}_{n+1} = s^{~}_{n+1} - beta_{n+1}
      Core::LinAlg::Matrix<NUM_STRESS_3D, 1> eta_tilde(Core::LinAlg::Initialization::zero);
      rel_stress(devstress_tilde, beta, eta_tilde);
      // update the invariant of the stress deviator
      J2bar = 1.0 / 2.0 *
                  (eta_tilde(0) * eta_tilde(0) + eta_tilde(1) * eta_tilde(1) +
                      eta_tilde(1) * eta_tilde(1) + eta_tilde(2) * eta_tilde(2)) +
              eta_tilde(3) * eta_tilde(3) + eta_tilde(4) * eta_tilde(4) +
              eta_tilde(5) * eta_tilde(5);
      // qbar_tilde = sqrt( 3/2 eta : eta ) = sqrt( 3/2 ) . || eta ||
      qbar_tilde = sqrt(3.0 * J2bar);

      // calculate the uniaxial yield stress
      // in contrast to implementation of de Souza Neto, no pairs are used
      // sigma_y = sigma_y0 + kappa(R^p)
      // kappa(R^p) = sigma_infty . (1 - exp(-delta . R^p) )
      // initial yield stress sigma_y0 for R^p == 0;
      sigma_y = get_sigma_y_at_strainbarnp(params_, 0);

      // update flow vectors: deviatoric stress norm || eta^{~}_{n+1} ||
      double eta_tildenorm = 0.0;
      eta_tildenorm = sqrt(eta_tilde(0) * eta_tilde(0) + eta_tilde(1) * eta_tilde(1) +
                           eta_tilde(2) * eta_tilde(2) +
                           2.0 * (eta_tilde(3) * eta_tilde(3) + eta_tilde(4) * eta_tilde(4) +
                                     eta_tilde(5) * eta_tilde(5)));

      // update undamaged unit flow vector
      // Nbar = eta^{~}_{n+1} / || eta^{~}_{n+1} || = eta_{n+1} / || eta_{n+1} ||
      // using the fact that due eta_{n+1} is proportional to eta^{trial}_{n+1}
      Nbar.update((1.0 / eta_tildenorm), eta_tilde);

      // update undamaged flow vector N_tilde
      // N_tilde = 3/2 . (s^{~} - beta_n) / sqrt{ 3/2 . || s^{~} - beta_n || }
      N_tilde.update((sqrt(3.0 / 2.0)), Nbar);

      // -------------------------------------------------- compute residuals

      // ---------------- compute residual of deviatoric stresses (42a)

      // k_s_tilde = s^{~}_{n+1} - s^{~,trial} + 2G . N_tilde . Dgamma/omega
      // with s_tilde^{trial} = 2G . devstrain
      // k_s_tilde --> = 0
      Core::LinAlg::Matrix<NUM_STRESS_3D, 1> k_s_tilde(Core::LinAlg::Initialization::uninitialized);
      k_s_tilde.update((-2.0 * G), devstrain);
      k_s_tilde.update(1.0, devstress_tilde, 1.0);
      double fac_ks_tilde = 0.0;
      fac_ks_tilde = 2.0 * G * Dgamma / omega;
      k_s_tilde.update(fac_ks_tilde, N_tilde, 1.0);

      // -------------- compute residual of consistency condition (42b)

      // Phi = qbar_tilde - sigma_y(R^p)
      double Phi = qbar_tilde - sigma_y;

      // ------------------------ compute residual of back stress (42c)

      // k_b = beta_{n+1} - beta_n - (Hkin . N_tilde - Hkin_rec . beta_n)
      //                    . Dgamma / (1 + Hkin_rec . Dgamma)
      // k_b = beta_{n+1} - beta_n
      //       - (Hkin . Dgamma / (1 + Hkin_rec . Dgamma) . N_tilde
      //       + (Hkin_rec . Dgamma / (1 + Hkin_rec . Dgamma) . beta_n
      double fac_k_b1 = 0.0;
      fac_k_b1 = (-1.0) * Hkin * Dgamma / (1.0 + Hkin_rec * Dgamma);
      double fac_k_b2 = 0.0;
      fac_k_b2 = Hkin_rec * Dgamma / (1.0 + Hkin_rec * Dgamma);

      Core::LinAlg::Matrix<NUM_STRESS_3D, 1> k_b(Core::LinAlg::Initialization::uninitialized);
      k_b.update((-1.0), backstresslast_.at(gp));
      k_b.update(1.0, beta, 1.0);
      k_b.update(fac_k_b1, N_tilde, 1.0);
      k_b.update(fac_k_b2, backstresslast_.at(gp), 1.0);

      // ----------------------------- compute residual of damage (42d)

      // k_D = D_{n+1} - D_n - y . Dgamma / (1 - D_{n+1})
      double k_D = 0.0;
      k_D = damage - damagelast_.at(gp) - y * Dgamma / omega;

      // ------------------------------------- calculate residual norms

      // calculate norm of deviatoric residual k_s_tilde
      double norm_k_s_tilde = 0.0;
      norm_k_s_tilde = k_s_tilde.norm2();

      // calculate norm of consistency condition Phi
      double norm_Phi = 0.0;
      norm_Phi = abs(Phi_trial);

      // calculate norm of back stress evolution equation
      double norm_k_beta = 0.0;
      norm_k_beta = k_b.norm2();

      // calculate norm of damage evolution equation
      double norm_k_D = 0.0;
      norm_k_D = abs(k_D);

      // ---------------------------------------------- check for convergence

      // load step is convergence ONLY if all equations are convergenced!
      if ((norm_k_s_tilde < params_->abstol_) and (norm_Phi < params_->abstol_) and
          (norm_k_beta < params_->abstol_) and (norm_k_D < params_->abstol_))
      {
        if (omega < 0.0)
        {
          FOUR_C_THROW("Damage variable has converged to unacceptable value");
        }
        break;
      }

      // --------------- else: load step NOT converged, calculate corrections

      // the sequence of corrections is decisive due to the dependency of single
      // terms on the former ones

      // ------------------------------------- 1.) calculating c_R (54)
      // c_R = num / h_alg

      // calculate the factor g included in third term of num (47)
      // g = [ 2 G . Delta_astrain + Hkin . Dgamma / (1 + Hkin_rec . Dgamma) ] / qbar_tilde
      //   = [ 2 G .( Dgamma / omega) + Hkin . Dgamma / (1 + Hkin_rec . Dgamma) ] / qbar_tilde
      if (qbar_tilde != 0.0)
      {
        g = (2.0 * G * Dgamma / omega + Hkin * Dgamma / (1.0 + Hkin_rec * Dgamma)) / qbar_tilde;
      }
      else
        FOUR_C_THROW("do not divide by zero!");

      if (g <= 0.0) FOUR_C_THROW("factor g has to be greater zero! g =  {:<14.8E}", g);

      // h_alg = 3 G + [ (1 - D_{n+1}) - y . Delta_astrain +
      //         + 2 G . (Delta_astrain)^2 . dy/ds_tilde : N_tilde ]
      //           . [ dkappa/dR + (3/2 . Hkin - Hkin_rec . (N_tilde : beta_n) )
      //                             / (1 + Hkin_rec . Dgamma)^2 ]
      //        - 3 G . (Delta_astrain)^2 . 3 G / qbar_tilde . Delta_astrain
      //          / (1 + 3/2 . g) . Hkin_rec / (1 + Hkin_rec . Dgamma)^2
      //          . dy/ds_tilde : [ beta_n - 2/3 . (N_tilde : beta_n) . N^tilde ]
      //
      // h_alg = 3 G + [ omega - y . (Dgamma / omega) +
      //         + 2 G . (Dgamma / omega)^2 . dy/ds_tilde : N_tilde ]
      //           . [ dkappa/dR + (3/2 . Hkin - Hkin_rec . (N_tilde : beta_n) )
      //                             / (1 + Hkin_rec . Dgamma)^2 ]
      //        - 3 G . (Dgamma / omega)^2 . 3 G / qbar_tilde . (Dgamma / omega)
      //          / (1 + 3/2 . g) . Hkin_rec / (1 + Hkin_rec . Dgamma)^2
      //          . dy/ds_tilde : [ beta_n - 2/3 . (N_tilde : beta_n) . N^tilde ]

      // dy/ds_tilde : N_tilde
      double dydstilde_Ntilde = 0.0;
      for (int i = 0; i < NUM_STRESS_3D; ++i)
        dydstilde_Ntilde += dy_dstilde(i, 0) * N_tilde(i, 0);  // (6x1)(6x1)
      // s_N = 2 G . (Dgamma / omega)^2 . dy/ds_tilde : N_tilde
      s_N = 2.0 * G * dydstilde_Ntilde;

      // Nbetaold = N_tilde : beta_n
      for (int i = 0; i < NUM_STRESS_3D; ++i)
        Nbetaold += N_tilde(i, 0) * backstresslast_.at(gp)(i, 0);  // (6x1)(6x1)

      // b_NbetaoldN = beta_n - 2/3 . (N_tilde : beta_n) . N_tilde
      b_NbetaoldN.update(1.0, backstresslast_.at(gp), (-2.0 / 3.0 * Nbetaold), N_tilde);

      // dydsb_NbetaoldN = dy_dstilde : [ beta_n - 2/3 . (N_tilde : beta_n) . N_tilde ]
      //                  = dy/ds_tilde . b_NbetaoldN
      double dydsb_NbetaoldN = 0.0;
      for (int i = 0; i < NUM_STRESS_3D; ++i)
        dydsb_NbetaoldN += dy_dstilde(i, 0) * b_NbetaoldN(i, 0);

      // N_ksb = N_tilde : (k_s - k_b)
      double N_ksb = 0.0;
      for (int i = 0; i < NUM_STRESS_3D; ++i) N_ksb += N_tilde(i) * (k_s_tilde(i) - k_b(i));

      // s_k = dy/ds_tilde : k_s
      double s_k = 0.0;
      for (int i = 0; i < NUM_STRESS_3D; ++i) s_k += dy_dstilde(i, 0) * k_s_tilde(i, 0);

      // k_NkN = [ k_s - k_b - 2/3 . ( N_tilde : (k_s - k_b) ) . N_tilde ]
      Core::LinAlg::Matrix<NUM_STRESS_3D, 1> k_NkN(Core::LinAlg::Initialization::uninitialized);
      k_NkN.update(1.0, k_s_tilde, (-1.0), k_b);
      double fac_k_NkN = 0.0;
      fac_k_NkN = -2.0 / 3.0 * N_ksb;
      k_NkN.update(fac_k_NkN, N_tilde, 1.0);

      // dy_dstildekN = dy/ds_tilde . [ k_s - k_b - 2/3 . ( N_tilde : (k_s - k_b) ) . N_tilde ]
      //              = dy_dstilde . k_NkN
      double dy_dstildek_NkN = 0.0;
      for (int i = 0; i < NUM_STRESS_3D; ++i) dy_dstildek_NkN += dy_dstilde(i, 0) * k_NkN(i, 0);

      // ------------------------------ calculate numerator num for c_R

      // num = [ omega - y . (Dgamma / omega) + 2 G . (Dgamma / omega)^2
      //         . dy/ds_tilde : N_tilde ] .
      //         . [ Phi - N_tilde : (k_s_tilde - k_b) ]
      //       + 3 G . [ (Dgamma / omega) . k_D + (Dgamma / omega)^2 . s_k ]
      //       - 3 G . (Dgamma / omega)^2 . 3 G / qbar_tilde . (Dgamma / omega)
      //         / (1 + 3/2 . g) . dy_dstildek_NkN
      double num = (omega - y * (Dgamma / omega) + s_N) * (Phi - N_ksb) +
                   3.0 * G * ((Dgamma / omega) * k_D + (Dgamma / omega) * (Dgamma / omega) * s_k) -
                   3.0 * G * (Dgamma / omega) * (Dgamma / omega) * 3.0 * G / qbar_tilde *
                       (Dgamma / omega) / (1.0 + 3.0 / 2.0 * g) * dy_dstildek_NkN;

      // -------------------------- calculate denominator h_alg for c_R

      // h_alg = 3 G + [ omega - y . Dgamma / omega + s_N ] . [ dkappa/dR +
      //                 . (3/2 . Hkin - Hkin_rec . Nbetaold )
      //                 / (1 + Hkin_rec . Dgamma)^2 ]
      //        - 3 G (Dgamma / omega)^2 . 3 G / qbar_tilde . Dgamma / omega
      //          / (1 + 3/2 . g) . Hkin_rec / (1 + Hkin_rec . Dgamma)^2
      //          . dydsb_NbetaoldN
      h_alg = 3.0 * G +
              (omega - y * Dgamma / omega + s_N) *
                  (Hiso + (3.0 / 2.0 * Hkin - Hkin_rec * Nbetaold) /
                              ((1.0 + Hkin_rec * Dgamma) * (1.0 + Hkin_rec * Dgamma))) -
              3.0 * G * ((Dgamma / omega) * (Dgamma / omega)) * 3.0 * G / qbar_tilde *
                  (Dgamma / omega) / (1.0 + 3.0 / 2.0 * g) * Hkin_rec /
                  ((1.0 + Hkin_rec * Dgamma) * (1.0 + Hkin_rec * Dgamma)) * dydsb_NbetaoldN;

      // ------------------------------------- 1.) calculating c_R (54)

      // c_R = num / h_alg
      c_R = num / h_alg;

      // ------------------------------- 2.) calculate c_strainbar (53)

      // c_strainbar = 1/(3 G) . { Phi - N_tilde : (k_s -k_b)
      //           - [ dkappa_dR + (3/2 . Hkin - Hkin_rec . N_tilde : beta_n)
      //               / (1 + Hkin_rec . Dgamma)^2 ] . c_R }
      c_strainbar = (Phi - N_ksb -
                        (Hiso + (3.0 / 2.0 * Hkin - Hkin_rec * Nbetaold) /
                                    ((1 + Hkin_rec * Dgamma) * (1 + Hkin_rec * Dgamma))) *
                            c_R) /
                    (3.0 * G);

      // --------------------------------- 3.) calculate c_s_tilde (51)

      // dN_tilde/ds_tilde : (c_s - c_b) := dNds_csb (50)

      Core::LinAlg::Matrix<NUM_STRESS_3D, 1> bracket(Core::LinAlg::Initialization::uninitialized);
      // bracket = [ k_s - k_b - 2/3 . N_tilde : (k_s - k_b) . N_tilde
      //            - Hkin_rec / (1 + Hkin_rec . Dgamma)^2 .
      //              ( beta_n - 2/3 . (N_tilde : beta_n) . N_tilde ) . c_R ]
      // with k_NkN = [ k_s - k_b - 2/3 . ( N_tilde : (k_s - k_b) ) . N_tilde ]
      //
      // bracket = [ k_NkN + fac_bracket . ( beta_n - 2/3 . Nbetaold . N_tilde ) ]
      double fac_bracket = -Hkin_rec * c_R / ((1 + Hkin_rec * Dgamma) * (1 + Hkin_rec * Dgamma));
      bracket.update(1.0, k_NkN, fac_bracket, backstresslast_.at(gp));
      fac_bracket *= (-2 / 3.0) * Nbetaold;
      bracket.update(fac_bracket, N_tilde, 1.0);

      // dNds_csb = - 3/2 . 1/qbar_tilde . 1/(1 + 3/2 . g) . bracket (50)
      Core::LinAlg::Matrix<NUM_STRESS_3D, 1> dNds_csb(Core::LinAlg::Initialization::uninitialized);
      double fac_dNds_csb = -3.0 / 2.0 * (1.0 / qbar_tilde) * (1.0 / (1.0 + 3.0 / 2.0 * g));
      dNds_csb.update(fac_dNds_csb, bracket);

      // c_s_tilde = c_s =  - k_s - 2 G . N_tilde . c_strainbar
      //       + (3 G / qbar_tilde) . ( (Dgamma / omega) / (1 + 3/2 . g)) . dNds_csb
      double fac_c_s1 = (-2.0) * G * c_strainbar;
      c_s.update((-1.0), k_s_tilde, fac_c_s1, N_tilde);
      double fac_c_s2 = 3.0 * G / qbar_tilde * (Dgamma / omega) / (1.0 + 3.0 / 2.0 * g);
      c_s.update(fac_c_s2, dNds_csb, 1.0);

      // ---------------------- 4.) calculate (c_s_tilde - c_beta) (49)

      // c_s - c_b = - 1/(1 + 3/2 . g) . [ k_s - k_b + g . N_tilde : (k_s - k_b)
      //               . N_tilde ] -
      //             - [ 2 G . c_strainbar + Hkin_rec /
      //               (1 + Hkin_rec . Dgamma)^2 . c_R ] . N_tilde +
      //             + 1/(1 + 3/2 . g) . Hkin_rec / (1 + Hkin_rec . Dgamma)^2
      //               . [ beta_n + g . (N_tilde : beta_n) . N_tilde ] . c_R
      //
      // c_s - c_b = - fac_csb . [ k_s - k_b + g . N_ksb . N_tilde ]
      //             - fac_csb_1 . N_tilde +
      //             + fac_csb_2 . [ beta_n + g . Nbetaold . N_tilde ]
      double fac_csb = 1.0 / (1.0 + 3.0 / 2.0 * g);
      double fac_csb_1 = fac_csb * 2.0 * G * c_strainbar +
                         Hkin_rec * c_R / ((1.0 + Hkin_rec * Dgamma) * (1.0 + Hkin_rec * Dgamma));
      double fac_csb_2 =
          fac_csb * Hkin_rec * c_R / ((1.0 + Hkin_rec * Dgamma) * (1.0 + Hkin_rec * Dgamma));
      c_s_b.update((-fac_csb), k_s_tilde, fac_csb, k_b);
      c_s_b.update((-fac_csb * g * N_ksb), N_tilde, 1.0);
      c_s_b.update((-fac_csb_1), N_tilde, 1.0);
      c_s_b.update(fac_csb_2, backstresslast_.at(gp), 1.0);
      c_s_b.update((fac_csb_2 * g * Nbetaold), N_tilde, 1.0);

      // ------ 5.) calculate c_beta = c_s_tilde - (c_s_tilde - c_beta)

      // c_b = c_s - (c_s - c_b) = c_s - c_s_b
      c_beta.update(1.0, c_s, (-1.0), c_s_b);

      // --------------------------------------- 6.) calculate c_D (52)

      // c_D = - k_D + y . c_strainbar + (Dgamma / omega) .
      //         dy/ds_tilde : [ - k_s - 2 G . N_tilde . c_strainbar
      //          + 3 G / qbar_tilde . (Dgamma / omega) / (1 + 3/2 . g) . dNds_csb ]
      //
      // c_D = - k_D + y . c_strainbar + (Dgamma / omega) . dy/ds_tilde : matrix_c_D

      // matrix_c_D = [ - k_s - 2 G . N_tilde . c_strainbar
      //                + 3 G / qbar_tilde . (Dgamma / omega) / (1 + 3/2 . g) . dNds_csb ]
      Core::LinAlg::Matrix<NUM_STRESS_3D, 1> matrix_c_D(
          Core::LinAlg::Initialization::uninitialized);
      matrix_c_D.update((-1.0), k_s_tilde, (-2.0 * G * c_strainbar), N_tilde);
      double fac_c_D = 3.0 * G / qbar_tilde * (Dgamma / omega) / (1.0 + 3.0 / 2.0 * g);
      matrix_c_D.update(fac_c_D, dNds_csb, 1.0);

      // dyds_matrixcD = dy_ds_tilde : matrix_c_D
      double dyds_matrixcD = 0.0;
      for (int i = 0; i < NUM_STRESS_3D; ++i) dyds_matrixcD += dy_dstilde(i, 0) * matrix_c_D(i, 0);

      // c_D = - k_D + y . c_strainbar + (Dgamma / omega) dy_ds_matrixcD
      c_D = -k_D + y * c_strainbar + (Dgamma / omega) * dyds_matrixcD;


    }  // end of local Newton iteration

    // ------------------------------------------------- update plastic state

    // -------------------------- update hardening and damage variables

    // check if damage variable is acceptable
    // admissible values: (0 <= D < 1) or (1 >= omega > 0)
    // sanity check: omega < 1.0e-20
    if (omega < omegamin)
    {
      std::cout << "Inadmissible value of integrity: omega = " << omega << " in ele " << eleGID
                << "!\n Omega has to be greater than zero!\n";
      omega = omegamin;
      failed = true;
    }
    // update damage variable damage_{n+1}
    damage = 1.0 - omega;

    // --> damaged isotropic hardening variable has newest value (see L678)

    // get damaged pressure
    double p = omega * p_tilde;

    // deviatoric stress
    // s_{n+1} = omega_{n+1} . s^tilde_{n+1}
    devstress.update(omega, devstress_tilde);

    // total stress
    // sigma_{n+1} = s_{n+1} + p_{n+1} . id2
    // pressure/volumetric stress no influence due to plasticity
    Damage::stress(p, devstress, *stress);

    // ------------------------------------------------- update strains

    // strain^e_{n+1} = strain^(e,trial)_{n+1} - Dgamma . N_tilde
    strain_e.update(1.0, trialstrain_e, (-Dgamma), N_tilde);

    // strain^p_{n+1} = strain^p_n + Dgamma . N_tilde
    // or alternatively
    //   strain^p_{n+1} = strain_{n+1} - strain^e_{n+1}
    strain_p.update(Dgamma, N_tilde, 1.0);

    // compute converged engineering strain components (Voigt-notation)
    for (int i = 3; i < NUM_STRESS_3D; ++i) strain_e(i) *= 2.0;
    for (int i = 3; i < NUM_STRESS_3D; ++i) strain_p(i) *= 2.0;

    // ------------------------------------------------- update history
    // plastic strain
    strainplcurr_.at(gp) = strain_p;

    // backstress
    backstresscurr_.at(gp) = beta;

    // accumulated plastic strain
    strainbarplcurr_.at(gp) = strainbar_p;

    // update damaged isotropic hardening variable
    isohardvarcurr_.at(gp) = Rplast;

    // update damage variable damage_{n+1}
    damagecurr_.at(gp) = damage;

    // update failed state
    failedcurr_.at(gp) = failed;

  }  // plastic corrector

  //------------------------------------------------------------------------
  // ELSE: elastic step (Phi_trial <= 0.0, Dgamma = 0.0)
  //------------------------------------------------------------------------
  else  // (Phi_trial <= 0.0)
  {
    // -------------------------- update stress using damaged elastic law
    // get damaged pressure
    // p = omega_{n+1} . p_tilde
    double p = p_tilde * omega;

    // get damaged deviatoric stresses
    // s_{n+1} = omega_{n+1} . s^{trial}_{n+1}
    devstress.update(omega, devstress_tilde);

    // result vectors of time step n+1 = omega . trial state vectors
    // sigma^e_n+1 = omega . sigma^(e,trial)_n+1
    //             = omega . (s^{trial}_{n+1} + p . id2)
    Damage::stress(p, devstress, *stress);

    // total strains
    // strain^e_{n+1} = strain^(e,trial)_{n+1}
    // compute converged engineering strain components (Voigt-notation)
    strain_e.update(trialstrain_e);
    for (int i = 3; i < NUM_STRESS_3D; ++i) strain_e(i) *= 2.0;

    // --------------------------------------------------------- update history
    // constant values for
    //  - plastic strains
    //  - accumulated (un)damaged plastic strains
    //  - back stress
    //    (-. relative stress)
    //  - stress
    // as current history vectors are set to zero in update(), the old values
    // need to be set instead, otherwise no constant plastic values are possible
    strainplcurr_.at(gp) = strainpllast_.at(gp);
    strainbarplcurr_.at(gp) = strainbarpllast_.at(gp);
    backstresscurr_.at(gp) = backstresslast_.at(gp);
    isohardvarcurr_.at(gp) = isohardvarlast_.at(gp);
    damagecurr_.at(gp) = damagelast_.at(gp);
    failedcurr_.at(gp) = failedlast_.at(gp);

  }  // elastic step

  //---------------------------------------------------------------------------
  // --------------------------------- consistent elastoplastic tangent modulus
  //---------------------------------------------------------------------------

  // using an associative flow rule: C_ep is symmetric
  // ( generally C_ep is nonsymmetric )
  // if Phi^trial = 0: two tangent stress-strain relations exist
  // plastic loading --> C == C_ep
  active_plasticity = (Dgamma > 0.0);

  // using an associative flow rule: C_ep is symmetric
  // ( generally C_ep is nonsymmetric )
  setup_cmat_elasto_plastic_full_lemaitre(*cmat, N_tilde, *stress, active_plasticity, Dgamma, s_N,
      g, h_alg, G, Hiso, bulk, Hkin, Hkin_rec, Nbetaold, gp, qbar_tilde, y, dy_dsigma_tilde,
      b_NbetaoldN);


  return;

}  // evaluate_full_lemaitre()


/*----------------------------------------------------------------------*
 | computes stress tensor                                    dano 11/13 |
 *----------------------------------------------------------------------*/
void Mat::Damage::stress(const double p,                      // volumetric stress
    const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& devstress,  // deviatoric stress tensor
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& stress            // 2nd PK-stress
)
{
  // total stress = deviatoric + hydrostatic pressure . I
  // sigma = s + p . I
  stress.update(devstress);
  for (int i = 0; i < 3; ++i) stress(i) += p;

}  // Stress()


/*----------------------------------------------------------------------*
 | compute relative deviatoric stress tensor                 dano 08/11 |
 *----------------------------------------------------------------------*/
void Mat::Damage::rel_stress(
    const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& devstress,  // deviatoric stress tensor
    const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& beta,       // back stress tensor
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& eta               // relative stress
)
{
  // relative stress = deviatoric - back stress
  // eta = s - beta
  eta.update(1.0, devstress, (-1.0), beta);

}  // RelStress()


/*----------------------------------------------------------------------*
 | computes isotropic elasticity tensor in matrix notion     dano 04/11 |
 | for 3d                                                               |
 *----------------------------------------------------------------------*/
void Mat::Damage::setup_cmat(Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>& cmat)
{
  // get material parameters
  // Young's modulus (modulus of elasticity)
  double young = params_->youngs_;
  // Poisson's ratio
  double nu = params_->poissonratio_;

  // isotropic elasticity tensor C in Voigt matrix notation
  // e.g. see lecture notes of NiliFEM Script (3.134)
  // C = 2G I + lambda (id2 \otimes id2) with I == I_s
  //   = 2G I_d + bulk (id2 \otimes id2)
  //
  //                       [ 1-nu     nu     nu |          0    0    0 ]
  //                       [        1-nu     nu |          0    0    0 ]
  //           E           [               1-nu |          0    0    0 ]
  //   C = --------------- [ ~~~~   ~~~~   ~~~~   ~~~~~~~~~~  ~~~  ~~~ ]
  //       (1+nu)*(1-2*nu) [                    | (1-2*nu)/2    0    0 ]
  //                       [                    |      (1-2*nu)/2    0 ]
  //                       [ symmetric          |           (1-2*nu)/2 ]
  //
  const double mfac = young / ((1.0 + nu) * (1.0 - 2.0 * nu));

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
 | computes isotropic damaged elastoplastic tensor in        dano 05/11 |
 | matrix notion for 3d                                                 |
 *----------------------------------------------------------------------*/
void Mat::Damage::setup_cmat_elasto_plastic(Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
                                                cmat,  // elasto-plastic tangent modulus (out)
    int eleID,                                         // current element ID
    double Dgamma,                                     // plastic multiplier
    double G,                                          // shear modulus
    double bulk_modulus,                               // bulk modulus
    double p_tilde,                                    // undamaged pressure
    double q_tilde,        // undamaged trial von Mises equivalent stress
    double energyrelrate,  // damage energy release rate
    double Ytan,           // derivative of engergy release rate Ytan w.r.t. Dgamma
    double sigma_y,        // current yield stress
    double Hiso,           // isotropic hardening modulus
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Nbar,  // unit flow vector
    int gp,                                       // current Gauss point
    bool damevolution,                            // flag indicating if damage evolves or not
    bool active_plasticity                        // flag indicating active plasticity
)
{
  // damage threshold is still not passed, i.e. use undamaged material tangent
  if (damevolution == false)
  {
    // incremental constitutive function for the stress tensor
    // sigma_n+1 = [ cmat - (Dgamma 6 G^2/q) I_d ] : strain_n+1^{e,trial}
    // consistent tangent operator
    // C^{ep} := dsigma_n+1 / dstrain_n+1^{e,trial}

    // depending on the flow vector Cmat_ep can be a fully-occupied matrix

    // C^{ep} = C^e - ( H^ . Dgamma . 6 . G^2 ) / q . I_d +
    //        +  H^ . 6 . G^2 ( Dgamma/q - 1/(3 G + Hiso) ) Nbar \otimes Nbar

    // if active_plasticity --> use C_ep
    // if !active_plasticity: elastic unloading: --> use C_e

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

    // ------------------------------------------------------- elastic term
    // C_ep = C_e
    // add standard isotropic elasticity tensor C_e first
    setup_cmat(cmat);

    // ------------------------------------------------------ plastic terms

    // if active_plasticity --> use C_ep
    // if !active_plasticity: elastic unloading: --> use C_e

    // ------------------------------------------------- first plastic term
    // - ( H^ . Dgamma . 6 . G^2 ) / q^{trial} . I_d

    double epfac = 0.0;
    double epfac3 = 0.0;
    // elastic trial von Mises effective stress
    if (q_tilde != 0.0)
    {
      epfac = active_plasticity ? (-1.0) * Dgamma * 6.0 * G * G / q_tilde : 0.0;
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

    if (q_tilde != 0.0)
    {
      // loop strains (columns)
      for (int k = 0; k < NUM_STRESS_3D; ++k)
      {
        // ---------------------------------------------------------- tangent
        // loop stresses (rows)
        for (int i = 0; i < NUM_STRESS_3D; ++i)
        {
          epfac3 =
              active_plasticity ? 6.0 * G * G * (Dgamma / q_tilde - 1.0 / (3.0 * G + Hiso)) : 0;
          // here: Nbar = s^{trial}_{n+1} / || s^{trial}_{n+1} ||
          cmat(i, k) += epfac3 * Nbar(i) * Nbar(k);
        }  // end rows, loop i
      }  // end columns, loop k
    }  // (q != 0.0)

    // complete material tangent C_ep available

  }  // (damevolution == false)

  // material tangent differs for case damage or not
  // if no damage: use standard tangent of purely plastic behaviour
  else  // (damevolution == true)
  {
    // incremental constitutive function for the stress tensor
    // consistent tangent operator
    // C^{ep} := dsigma_n+1 / dstrain_n+1

    // depending on the flow vector Cmat_ep can be a fully-occupied matrix

    // if active_plasticity --> use C^{ep}
    // if !active_plasticity: elastic unloading --> use C^e

    // -------------------------------------------------- identity matrices

    // deviatoric projection tensor: I_d = I_s - 1/3 I \otimes I
    // I_d in Voigt-notation applied to symmetric problem, like stress calculation
    //         [ 2/3   -1/3  -1/3 | 0    0    0  ]
    //         [-1/3    2/3  -1/3 | 0    0    0  ]
    //         [-1/3   -1/3   2/3 | 0    0    0  ]
    //   I_d = [ ~~~~  ~~~~  ~~~~  ~~~  ~~~  ~~~ ]
    //         [                  | 1/2   0   0  ]
    //         [    symmetric     |      1/2  0  ]
    //         [                  |          1/2 ]

    // build Cartesian identity 2-tensor I_{AB}
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> id2(Core::LinAlg::Initialization::zero);
    for (int i = 0; i < 3; i++) id2(i) = 1.0;

    // set Cartesian identity 4-tensor in 6-Voigt matrix notation
    // this is fully 'contra-variant' identity tensor, i.e. I^{ABCD}
    // REMARK: rows are stress-like 6-Voigt
    //         columns are stress-like 6-Voigt
    Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> id4sharp(Core::LinAlg::Initialization::zero);
    for (int i = 0; i < 3; i++) id4sharp(i, i) = 1.0;
    for (int i = 3; i < 6; i++) id4sharp(i, i) = 0.5;

    // ------------------------------------- extract current history values
    // integrity omega_{n+1}
    double omega = 1.0 - damagecurr_.at(gp);

    // ----------------------------------------------------- damaged elastic term
    // C^{ep} = (1 - D_{n+1}) . C^e = omega_{n+1} . C^e
    //        = omega_{n+1} . 2G . I_d + omega_{n+1} . bulk_modulus . id2 \otimes id2
    // add standard isotropic elasticity tensor C^e first
    if (active_plasticity)
    {
      // C^{ep} = a . I_d + b . Nbar \otimes Nbar + c . Nbar \otimes id2
      //          + d . id2 \otimes Nbar + e . id2 \otimes id2

      // ----------------------------------- extract some material parameters

      // get material parameters
      // damage evolution law denominator r
      double damden = params_->damden_;
      // damage evolution law exponent s
      double damexp = params_->damexp_;

      // ---------------------------------------------------------- plastic terms

      // check if omega is admissible
      if (omega < 1.0e-20)
        FOUR_C_THROW("Omega has to be greater than zero! omega = {:<14.8E}\n", omega);

      // be aware: in the 1st equilibrium (Newton) iteration (i=0) C^{ep} is
      // indeterminate due to Dgamma == 0.0
      // a small perturbation Dgamma=1e-08 is used (instead of using a limiting
      // procedure) which was proposed by de Souza Neto
      if (Dgamma == 0) Dgamma = 1.0e-08;

      // ------------------------ factors required for the elasto-plastic tangent

      // some constants
      double aux = -energyrelrate / damden;
      double auxb = (q_tilde - sigma_y) / (3.0 * G);
      // PhiT = qtilde - sigma_y(Rplast_{n+1})
      // be careful: NOT Phi_trial which was calculated using sigma_y(Rplast_n)
      double PhiT = q_tilde - sigma_y;

      // ---------------------------------------- derivatives w.r.t. Dgamma
      // domega/dDgamma
      double Domega = (3.0 * G + omega * Hiso) / PhiT;
      // domega/dq_tilde . dq_tilde/dDgamma
      double DomegaDq_tilde = -omega / PhiT;

      // derivative of residual function dF/dDgamma
      double ResTan = Domega - Hiso / (3.0 * G) * std::pow(aux, damexp) -
                      auxb * damexp * Ytan / damden * std::pow(aux, (damexp - 1));
      // derivative of residual function dF/dp_tilde . dp_tilde/dDgamma
      // DResDp_tilde = s . (q_tilde - sigma_y)/(3G) . std::pow((-Y/r), s-1) . p_tilde /(r .
      // bulk_modulus)
      double DResDp_tilde =
          damexp * auxb * std::pow(aux, (damexp - 1.0)) * p_tilde / (damden * bulk_modulus);
      // derivative of residual function F w.r.t. q_tilde
      double DResDq_tilde = DomegaDq_tilde + std::pow(aux, damexp) / (3.0 * G);

      // --------- calculate the constants according to de Souza Neto (12.52)

      // a1 = - DResDq_tilde / ResTan
      //    = [ omega / (q_tilde - sigma_y) - (-energyrelrate/damden)^damexp
      //                                      / (3.0 * G) ] / ResTan;
      // or alternatively
      //   a1 = [1 - Dgamma / (omega . omega) . (-Y/r)^s] . omega
      //          / [ (q_tilde - sigma_y) . ResTan]
      double a1 = -DResDq_tilde / ResTan;

      // a2 = - DResDp_tilde / ResTan
      //    = - s . p_tilde . (q_tilde - sigma_y) / (3G . r . bulk_modulus . ResTan) .(-Y/r)^{s-1};
      double a2 = 0.0;
      a2 = -DResDp_tilde / ResTan;

      // a3 = a2 * Domega
      double a3 = 0.0;
      a3 = a2 * Domega;

      // a4 = a1 . Domega + DomegaDq_tilde
      //    = a1 . Domega - omega / (q_tilde - sigma_y)
      double a4 = 0.0;
      a4 = a1 * Domega + DomegaDq_tilde;

      // ------- calculate actual coefficients of elasto-plastic material tangent

      // a = 2G . sigma_y . omega / q_tilde
      double a = 0.0;
      // b = 2G . (a1 . Hiso . omega + a4 . sigma_y - sigma_y . omega / q_tilde
      double b = 0.0;

      if (q_tilde != 0.0)
      {
        a = (2.0 * G * sigma_y * omega) / q_tilde;
        b = 2.0 * G * (a1 * Hiso * omega + a4 * sigma_y - sigma_y * omega / q_tilde);
      }
      // c = bulk_modulus . (a2 . Hiso . omega + a3 . sigma_y) / sqrt(3/2)
      double c = 0.0;
      c = bulk_modulus * (a2 * Hiso * omega + a3 * sigma_y) / sqrt(3.0 / 2.0);
      // d = p_tilde . 2G . sqrt(3/2) . a4
      double d = 0.0;
      d = (p_tilde * 2.0 * G * sqrt(3.0 / 2.0) * a4);
      // e = bulk_modulus . (omega + p_tilde . a3)
      double e = 0.0;
      e = bulk_modulus * (omega + p_tilde * a3);

      // ------------------------------- assemble elasto-plastic material tangent

      // empty consistent tangent operator
      cmat.clear();
      // constitutive tensor
      // I_d = id4sharp - 1/3 Id \otimes Id
      // contribution: Id4^#
      cmat.update(a, id4sharp);
      // contribution: Id \otimes Id
      cmat.multiply_nt((a / (-3.0)), id2, id2, 1.0);
      cmat.multiply_nt(b, Nbar, Nbar, 1.0);
      cmat.multiply_nt(c, Nbar, id2, 1.0);
      cmat.multiply_nt(d, id2, Nbar, 1.0);
      cmat.multiply_nt(e, id2, id2, 1.0);

    }  // plastic load step
    else  // elastic (un-)loading
    {
      setup_cmat(cmat);
      cmat.scale(omega);
    }
    // complete material tangent C_ep available


  }  // damage evolves: (damevolution == true)

}  // setup_cmat_elasto_plastic()


/*----------------------------------------------------------------------*
 | computes isotropic elastoplastic tensor for full Lemaitre dano 11/13 |
 | model in matrix notion for 3d including ductile damage               |
 *----------------------------------------------------------------------*/
void Mat::Damage::setup_cmat_elasto_plastic_full_lemaitre(
    Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
        cmat,                                        // elasto-plastic tangent modulus (out)
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> N_tilde,  // flow vector
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& stress,  // stress
    bool active_plasticity,                          // flag indicating active plasticity
    double Dgamma,                                   // plastic multiplier
    double s_N,  // s_N = 2 G . (Dgamma / omega)^2 . dy/ds_tilde : N_tilde
    double g,    // g = (2 G . (Dgamma / omega) + Hkin . Dgamma / (1 + Hkin_rec . Dgamma) / q_tilde
    double h_alg,         // see definition of expression above
    double G,             // shear modulus
    double dkappa_dR,     // derivative of the hardening curve
    double bulk_modulus,  // bulk modulus
    double Hkin,          // kinematic hardening variable 1 (kinematic hardening modulus)
    double Hkin_rec,      // kinematic hardening variable 2 (saturation effect)
    double Nbetaold,      // N_beta = Hkin_rec . N_tilde : beta_n, CARO:Nbetaold =N_tilde : beta_n
    int gp,               // current Gauss point
    double qbar_tilde,    // effective trial stress ^tilde
    double y,             // (-Y / r)^s
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> dy_dsigma_tilde,
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1>
        b_NbetaoldN  // beta_n - 2/3 . (N_tilde : beta_n) . N_tilde
)
{
  // incremental constitutive function for the stress tensor
  // consistent tangent operator
  // C^{ep} := dsigma_n+1 / dstrain_n+1

  // depending on the flow vector Cmat_ep can be a fully-occupied matrix

  // if active_plasticity --> use C^{ep}
  // if !active_plasticity: --> use C^e

  // -------------------------------------------------- identity matrices

  // deviatoric projection tensor: I_d = I_s - 1/3 I \otimes I
  // I_d in Voigt-notation applied to symmetric problem, like stress calculation
  //         [ 2/3   -1/3  -1/3 | 0    0    0  ]
  //         [-1/3    2/3  -1/3 | 0    0    0  ]
  //         [-1/3   -1/3   2/3 | 0    0    0  ]
  //   I_d = [ ~~~~  ~~~~  ~~~~  ~~~  ~~~  ~~~ ]
  //         [                  | 1/2   0   0  ]
  //         [    symmetric     |      1/2  0  ]
  //         [                  |          1/2 ]

  // build Cartesian identity 2-tensor I_{AB}
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> id2(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++) id2(i) = 1.0;

  // set Cartesian identity 4-tensor in 6-Voigt matrix notation
  // this is fully 'contra-variant' identity tensor, i.e. I^{ABCD}
  // REMARK: rows are stress-like 6-Voigt
  //         columns are stress-like 6-Voigt
  Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> id4sharp(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++) id4sharp(i, i) = 1.0;
  for (int i = 3; i < 6; i++) id4sharp(i, i) = 0.5;

  // ------------------------------------- extract current history values
  // integrity omega_{n+1}
  double omega = 1.0 - damagecurr_.at(gp);

  // ------------------------------------------ elastic undamaged tangent
  // C^e = 2G . I_d + bulk_modulus . id2 \otimes id2
  setup_cmat(cmat);

  if (active_plasticity)
  {
    // ------------------------------------------ damaged elastoplastic tangent

    // consistent/algorithmic damaged elastoplastic tangent operator (70)
    // C^{ep} = omega . C_tilde^{ep} - stress_tilde \otimes .
    //          . { Dgamma / omega . C_tilde^{ep} : dy/dsigma_tilde
    //              + 2/3 . y . N_tilde
    //              - y / 3 G . [ dkappa_dR + (3/2 . Hkin - Hkin_rec .
    //                . N_tilde : beta_n ) / (1 + Hkin_rec . Dgamma)^2 ]
    //                 . n_alg / h_alg }

    // consistent tangent operator is achieved due to variation of effective,
    // undamaged stress sigma_tilde (68)
    // C^{~,ep} = C^e - (2 G)^2 . (Dgamma / omega) / (1 + 3/2 . g) . dPhi_dsigma_tilde_square
    //            - 2/3 . N_tilde \otimes (2 G . N_tilde)
    //            + { 2/3 . [ dkappa_dR + 3/2 . Hkin - Hkin_rec . (N_tilde : beta_n)
    //                        / (1 + Hkin_rec . Dgamma)^2 ] . N_tilde
    //                 - 3 G / qbar_tilde . (Dgamma / omega) / (1 + 3/2 . g) .
    //                     Hkin_rec / (1 + Hkin_rec . Dgamma)^2 .
    //                     [ beta_n - 2/3 . (N_tilde : beta_n) . N_tilde ]
    //                 } \otimes (n_alg / h_alg)

    // stress_tilde = stress / omega = sigma / omega
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> stress_tilde(
        Core::LinAlg::Initialization::uninitialized);
    stress_tilde.update((1.0 / omega), stress);

    // --------------------------- linearisation of delta_R = delta_gamma

    // Lin[ R ] = (1 - D) . Lin[ strainbar ] - Dgamma / omega . Lin[ D ]  (60e)
    //          = (n^alg : Lin[ strain ]) / h^alg                         (64)
    //
    // n_alg = [ omega - y . Dgamma / omega + 2 G . (Dgamma / omega)^2 .
    //           dy_ds_tilde : N_tilde ] . 2 G . N_tilde
    //         - 3 G . (Dgamma / omega)^2 . (C^e - (2 G)^2 . Dgamma / omega
    //           / (1 + 3/2 . g) . dPhi_dsigma_tilde_square ] : dy_dsigma_tilde
    //
    // n_alg = [ omega - y . Dgamma / omega + s_N ] . 2 G . N_tilde
    //         - 3 G . (Dgamma / omega)^2 . (C^e - (2 G)^2 . Dgamma / omega
    //           / (1 + 3/2 . g) . dPhi_dsigma_tilde_square ] : dy_dsigma_tilde

    // dPhi_dsigma_tilde_square = (3/2 . I_d - N_tilde \otimes N_tilde) / qbar_tilde
    Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> dPhi_dsigma_tilde_square(
        Core::LinAlg::Initialization::zero);
    double fac_dPhidsigma = 0.0;
    if (qbar_tilde != 0)
    {
      fac_dPhidsigma = 3.0 / 2.0 / qbar_tilde;
      // I_d = id4sharp - 1/3 Id \otimes Id
      // contribution: Id4^#
      dPhi_dsigma_tilde_square.update(fac_dPhidsigma, id4sharp);
      // contribution: Id \otimes Id
      double fac_dPhidsigma_1 = fac_dPhidsigma / (-3.0);
      dPhi_dsigma_tilde_square.multiply_nt(fac_dPhidsigma_1, id2, id2, 1.0);
      // loop strains (columns)
      for (int k = 0; k < NUM_STRESS_3D; ++k)
      {
        // ---------------------------------------------------------- tangent
        // loop stresses (rows)
        for (int i = 0; i < NUM_STRESS_3D; ++i)
        {
          // (- 1 / qbar_tilde) . N_tilde \otimes N_tilde
          dPhi_dsigma_tilde_square(i, k) += -1.0 / qbar_tilde * N_tilde(i) * N_tilde(k);
        }  // end rows, loop i
      }  // end columns, loop k
    }  // (qbar_tilde != 0.0)

    // n_alg = [ omega - y . Dgamma / omega + s_N ] . 2 G . N_tilde
    //         - 3 G . (Dgamma / omega)^2 . [ C^e - (2 G)^2 . Dgamma / omega
    //           / (1 + 3/2 . g) . dPhi_dsigma_tilde_square ] : dy_dsigma_tilde

    // n_alg = [ omega - y . Dgamma / omega + s_N ] . 2 G . N_tilde
    //         - 3 G . (Dgamma / omega)^2 . C^e : dy_dsigma_tilde
    //         - 3 G . (Dgamma / omega)^2 . - (2 G)^2 . Dgamma / omega
    //           / (1 + 3/2 . g) . dPhi_dsigma_tilde_square : dy_dsigma_tilde

    // n_alg = [ omega - y . Dgamma / omega + s_N ] . 2 G . N_tilde
    //
    // with s_N = = 2 G . (Dgamma / omega)^2 . dy/ds_tilde : N_tilde
    double fac_n_alg_1 = (omega - y * (Dgamma / omega) + s_N) * 2.0 * G;
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> n_alg(Core::LinAlg::Initialization::uninitialized);
    n_alg.update(fac_n_alg_1, N_tilde);

    // n_alg += - 3 G . (Dgamma / omega)^2 . C^e : dy_dsigma_tilde
    // calculate C^e : dy_dsigma_tilde
    //          (6x6)  (6x1)
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> ce_dydsigmatilde(
        Core::LinAlg::Initialization::uninitialized);
    ce_dydsigmatilde.multiply(cmat, dy_dsigma_tilde);
    double fac_n_alg_2 = -3.0 * G * (Dgamma / omega) * (Dgamma / omega);
    n_alg.update(fac_n_alg_2, ce_dydsigmatilde, 1.0);

    // n_alg += 3 G . (Dgamma / omega)^3 . (2 G)^2
    //           / (1 + 3/2 . g) . dPhi_dsigma_tilde_square : dy_dsigma_tilde
    // dPhi_dsigma_tilde_square : dy_dsigma_tilde
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> dPhidsigma_dydsigma(
        Core::LinAlg::Initialization::uninitialized);
    dPhidsigma_dydsigma.multiply(dPhi_dsigma_tilde_square, dy_dsigma_tilde);
    double fac_n_alg_3 =
        3.0 * G * std::pow((Dgamma / omega), 3) * std::pow((2 * G), 2) / (1 + 3.0 / 2.0 * g);
    n_alg.update(fac_n_alg_3, dPhidsigma_dydsigma, 1.0);

    // dkappa_HHNb = dkappa_dR + [ 3/2 . Hkin - Hkin_rec . (N_tilde : beta_n) ]
    //                             / (1 + Hkin_rec . Dgamma)^2
    double dkappa_HHNb = dkappa_dR + (3.0 / 2.0 * Hkin - Hkin_rec * Nbetaold) /
                                         ((1.0 + Hkin_rec * Dgamma) * (1.0 + Hkin_rec * Dgamma));

    // b_NbetaoldN = beta_n - 2/3 . (N_tilde : beta_n) . N_tilde

    // C^{~,ep} = C^e
    Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> Cep_tilde(
        Core::LinAlg::Initialization::uninitialized);
    Cep_tilde.update(cmat);

    // C^{~,ep} += - (2 G)^2 . (Dgamma / omega) / (1 + 3/2 . g) . dPhi_dsigma_tilde_square
    //          += fac_Cep_tilde1 . dPhi_dsigma_tilde_square
    // - (2 G)^2 . (Dgamma / omega) / (1 + 3/2 . g)
    double fac_Cep_tilde1 = -(2.0 * G) * (2.0 * G) * (Dgamma / omega) / (1.0 + 3.0 / 2.0 * g);
    Cep_tilde.update(fac_Cep_tilde1, dPhi_dsigma_tilde_square, 1.0);

    // C^{~,ep} += - 2/3 . 2 G . N_tilde \otimes  N_tilde)
    //          += fac_Cep_tilde2 . N_tilde \otimes  N_tilde)
    double fac_Cep_tilde2 = (-2.0) / 3.0 * 2.0 * G;
    Cep_tilde.multiply_nt(fac_Cep_tilde2, N_tilde, N_tilde, 1.0);

    // prematrix_n_alg = 2/3 . dkappa_HHNb . N_tilde
    //                   - 3 G / qbar_tilde . (Dgamma / omega) / (1 + 3/2 . g) .
    //                  . Hkin_rec / (1 + Hkin_rec . Dgamma)^2 . b_NbetaoldN
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> prematrix_n_alg(
        Core::LinAlg::Initialization::uninitialized);

    // fac_prematrix_n_alg = - 3 G / qbar_tilde . (Dgamma / omega) / (1 + 3/2 . g)
    //                       . Hkin_rec / (1 + Hkin_rec . Dgamma)^2
    double fac_prematrix_n_alg = -3.0 * G / qbar_tilde * (Dgamma / omega) / (1.0 + 3.0 / 2.0 * g) *
                                 Hkin_rec / ((1.0 + Hkin_rec * Dgamma) * (1.0 + Hkin_rec * Dgamma));
    // prematrix_n_alg = 2/3 . dkappa_HHNb . N_tilde
    //                   + fac_prematrix_n_alg . b_NbetaoldN
    prematrix_n_alg.update((2.0 / 3.0 * dkappa_HHNb), N_tilde);
    prematrix_n_alg.update(fac_prematrix_n_alg, b_NbetaoldN, 1.0);

    // C^{~,ep} += + prematrix_n_alg \otimes (n_alg / h_alg)
    double fac_Cep_tilde3 = 1.0 / h_alg;
    Cep_tilde.multiply_nt(fac_Cep_tilde3, prematrix_n_alg, n_alg, 1.0);

    // ------------- assemble consistent damaged elastoplastic material tangent

    // C^{ep} = omega . C^{~,ep}
    //          - stress_tilde \otimes cep_tilde_dydsigma_tilde
    //          - stress_tilde \otimes . 2/3 . y . N_tilde
    //          - stress_tilde \otimes . fac_cep_nalg . n_alg

    // C^{ep} = omega . C^{~,ep}
    cmat.update(omega, Cep_tilde);

    // cep_tilde_dydsigma_tilde = (Dgamma / omega) . C^{~,ep} : dy/dsigma_tilde
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> cep_tilde_dydsigma_tilde(
        Core::LinAlg::Initialization::uninitialized);
    cep_tilde_dydsigma_tilde.multiply((Dgamma / omega), Cep_tilde, dy_dsigma_tilde);
    // C^{ep} += - stress_tilde \otimes cep_tilde_dydsigma_tilde
    cmat.multiply_nt((-1.0), stress_tilde, cep_tilde_dydsigma_tilde, 1.0);

    // C^{ep} += - stress_tilde \otimes . (2/3 . y) . N_tilde }
    cmat.multiply_nt((-2.0 / 3.0 * y), stress_tilde, N_tilde, 1.0);

    // C^{ep} += - stress_tilde \otimes (fac_cep_nalg . n_alg)
    double fac_cep_nalg = -y / (3 * G) * dkappa_HHNb / h_alg;
    cmat.multiply_nt((-fac_cep_nalg), stress_tilde, n_alg, 1.0);

  }  // active_plasticity
  else
  {
    // ------------------------------------------------ damaged elastic tangent

    // C^e_D = (1 - D_{n+1}) . C^e = omega_{n+1} . C^e
    //        = omega_{n+1} . 2G . I_d + omega_{n+1} . bulk_modulus . id2 \otimes id2
    // add standard isotropic elasticity tensor C^e first
    cmat.scale(omega);
  }

}  // setup_cmat_elasto_plastic_full_lemaitre()


/*---------------------------------------------------------------------*
 | return names of visualization data (public)              dano 09/13 |
 *---------------------------------------------------------------------*/
void Mat::Damage::vis_names(std::map<std::string, int>& names) const
{
  std::string variablename = "accumulatedstrain";
  names[variablename] = 1;  // scalar

  variablename = "isohardeningvar";
  names[variablename] = 1;  // scalar

  variablename = "damage";
  names[variablename] = 1;  // scalar

  variablename = "failedFlag";
  names[variablename] = 1;  // scalar

}  // vis_names()


/*---------------------------------------------------------------------*
 | return visualization data (public)                       dano 09/13 |
 *---------------------------------------------------------------------*/
bool Mat::Damage::vis_data(
    const std::string& name, std::vector<double>& data, int numgp, int eleID) const
{
  if (name == "accumulatedstrain")
  {
    if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");
    double temp = 0.0;
    for (int gp = 0; gp < numgp; gp++) temp += accumulated_strain(gp);
    data[0] = temp / numgp;
  }

  if (name == "isohardeningvar")
  {
    if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");
    double temp = 0.0;
    for (int gp = 0; gp < numgp; gp++) temp += isotropic_hardening_variable(gp);
    data[0] = temp / numgp;
  }

  if (name == "damage")
  {
    if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");
    double temp = 0.0;
    for (int gp = 0; gp < numgp; gp++) temp += isotropic_damage(gp);
    data[0] = temp / numgp;
  }

  if (name == "failedFlag")
  {
    if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");
    double temp = 0.0;
    for (int gp = 0; gp < numgp; gp++) temp += failure_flag(gp);
    data[0] = temp / numgp;
  }

  return true;
}  // vis_data()


/*---------------------------------------------------------------------*
 | return names of visualization data for direct VTK output            |
 *---------------------------------------------------------------------*/
void Mat::Damage::register_output_data_names(
    std::unordered_map<std::string, int>& names_and_size) const
{
  names_and_size["accumulated_plastic_strain"] = 1;
  names_and_size["isohardeningvar"] = 1;
  names_and_size["damage"] = 1;
  names_and_size["failedFlag"] = 1;
}


bool Mat::Damage::evaluate_output_data(
    const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const
{
  if (name == "accumulated_plastic_strain")
  {
    for (std::size_t gp = 0; gp < strainbarplcurr_.size(); ++gp)
    {
      data(gp, 0) = accumulated_strain(gp);
    }
    return true;
  }
  if (name == "isohardeningvar")
  {
    for (std::size_t gp = 0; gp < isohardvarcurr_.size(); ++gp)
    {
      data(gp, 0) = isotropic_hardening_variable(gp);
    }
    return true;
  }
  if (name == "damage")
  {
    for (std::size_t gp = 0; gp < damagecurr_.size(); ++gp)
    {
      data(gp, 0) = damagecurr_.at(gp);
    }
    return true;
  }
  if (name == "failedFlag")
  {
    for (std::size_t gp = 0; gp < failedcurr_.size(); ++gp)
    {
      data(gp, 0) = failedcurr_.at(gp);
    }
    return true;
  }
  return false;
}


/*----------------------------------------------------------------------*/

FOUR_C_NAMESPACE_CLOSE
