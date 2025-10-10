// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_plasticlinelast.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | constructor (public)                                      dano 04/11 |
 *----------------------------------------------------------------------*/
Mat::PAR::PlasticLinElast::PlasticLinElast(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      youngs_(matdata.parameters.get<double>("YOUNG")),
      poissonratio_(matdata.parameters.get<double>("NUE")),
      density_(matdata.parameters.get<double>("DENS")),
      yield_(matdata.parameters.get<double>("YIELD")),
      isohard_(matdata.parameters.get<double>("ISOHARD")),
      kinhard_(matdata.parameters.get<double>("KINHARD")),
      abstol_(matdata.parameters.get<double>("TOL"))
{
}


/*----------------------------------------------------------------------*
 | is called in Material::Factory from read_materials()       dano 02/12 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::PlasticLinElast::create_material()
{
  return std::make_shared<Mat::PlasticLinElast>(this);
}


Mat::PlasticLinElastType Mat::PlasticLinElastType::instance_;


/*----------------------------------------------------------------------*
 | is called in Material::Factory from read_materials()       dano 02/12 |
 *----------------------------------------------------------------------*/
Core::Communication::ParObject* Mat::PlasticLinElastType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::PlasticLinElast* plastic = new Mat::PlasticLinElast();
  plastic->unpack(buffer);
  return plastic;
}


/*----------------------------------------------------------------------*
 | constructor (public)                                      dano 04/11 |
 *----------------------------------------------------------------------*/
Mat::PlasticLinElast::PlasticLinElast() : params_(nullptr), plastic_step_(false) {}


/*----------------------------------------------------------------------*
 | copy-constructor (public)                                 dano 04/11 |
 *----------------------------------------------------------------------*/
Mat::PlasticLinElast::PlasticLinElast(Mat::PAR::PlasticLinElast* params) : params_(params) {}


/*----------------------------------------------------------------------*
 | pack (public)                                             dano 04/11 |
 *----------------------------------------------------------------------*/
void Mat::PlasticLinElast::pack(Core::Communication::PackBuffer& data) const
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

    // insert current iteration states
    add_to_pack(data, strainplcurr_->at(var));
    add_to_pack(data, backstresscurr_->at(var));
    add_to_pack(data, strainbarplcurr_->at(var));
  }

  add_to_pack(data, plastic_step_);
}  // pack()


/*----------------------------------------------------------------------*
 | unpack (public)                                           dano 04/11 |
 *----------------------------------------------------------------------*/
void Mat::PlasticLinElast::unpack(Core::Communication::UnpackBuffer& buffer)
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
        params_ = static_cast<Mat::PAR::PlasticLinElast*>(mat);
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

  for (int var = 0; var < histsize; ++var)
  {
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> tmp_vector(Core::LinAlg::Initialization::zero);
    double tmp_scalar = 0.0;

    // last converged states are unpacked
    extract_from_pack(buffer, tmp_vector);
    strainpllast_->push_back(tmp_vector);
    extract_from_pack(buffer, tmp_vector);
    backstresslast_->push_back(tmp_vector);
    extract_from_pack(buffer, tmp_scalar);
    strainbarpllast_->push_back(tmp_scalar);

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
 | initialise / allocate internal stress variables (public)      04/11 |
 *---------------------------------------------------------------------*/
void Mat::PlasticLinElast::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  // initialise history variables
  strainpllast_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  strainplcurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();

  backstresslast_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  backstresscurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();

  strainbarpllast_ = std::make_shared<std::vector<double>>();
  strainbarplcurr_ = std::make_shared<std::vector<double>>();

  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> emptyvect(Core::LinAlg::Initialization::zero);
  strainpllast_->resize(numgp);
  strainplcurr_->resize(numgp);

  backstresslast_->resize(numgp);
  backstresscurr_->resize(numgp);

  strainbarpllast_->resize(numgp);
  strainbarplcurr_->resize(numgp);

  for (int i = 0; i < numgp; i++)
  {
    strainpllast_->at(i) = emptyvect;
    strainplcurr_->at(i) = emptyvect;

    backstresslast_->at(i) = emptyvect;
    backstresscurr_->at(i) = emptyvect;

    strainbarpllast_->at(i) = 0.0;
    strainbarplcurr_->at(i) = 0.0;
  }

  isinit_ = true;
  return;

}  // setup()


/*---------------------------------------------------------------------*
 | update internal stress variables (public)                dano 04/11 |
 *---------------------------------------------------------------------*/
void Mat::PlasticLinElast::update()
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
void Mat::PlasticLinElast::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  Core::LinAlg::Matrix<6, 1> stress_view = Core::LinAlg::make_stress_like_voigt_view(stress);
  Core::LinAlg::Matrix<6, 6> cmat_view = Core::LinAlg::make_stress_like_voigt_view(cmat);

  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> plstrain(Core::LinAlg::Initialization::zero);

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
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> id2(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++) id2(i) = 1.0;

  // linstrain (in): independent variable passed from the element
  //  strain^p: evolution is determined by the flow rule, history variable
  //  strain^e: definition of additive decomposition:
  //  strain^e = strain - strain^p
  // REMARK: stress-like 6-Voigt vector
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> strain =
      Core::LinAlg::make_strain_like_voigt_matrix(glstrain);

  //---------------------------------------------------------------------------
  // elastic predictor (trial values)
  //---------------------------------------------------------------------------

  // ------------------------------------------------ old plastic strains
  // strain^{p,trial}_{n+1} = strain^p_n
  // accumulated/equivalent plastic strain
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> strain_p(Core::LinAlg::Initialization::uninitialized);
  for (int i = 0; i < 6; i++) strain_p(i, 0) = strainpllast_->at(gp)(i, 0);

  // get old equivalent plastic strain only in case of plastic step
  double strainbar_p = 0.0;
  // accumulated or equivalent plastic strain (scalar-valued)
  // astrain^{p,trial}_{n+1} = astrain^p_n
  strainbar_p = (strainbarpllast_->at(gp));
  if (strainbarpllast_->at(gp) < 0.0)
    FOUR_C_THROW("accumulated plastic strain has to be equal to or greater than zero!");

  // ---------------------------------------------------- old back stress
  // beta^{trial}_{n+1} = beta_n
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> beta(Core::LinAlg::Initialization::uninitialized);
  for (int i = 0; i < 6; i++) beta(i, 0) = backstresslast_->at(gp)(i, 0);

  // ----------------------------------------------- physical strains
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
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> devstress(Core::LinAlg::Initialization::uninitialized);
  devstress.update((2.0 * G), devstrain);
  // be careful for shear stresses (sigma_12)
  // in Voigt-notation the shear strains have to be scaled with 1/2
  // normally done in the material tangent (cf. id4sharp)

  // ------------------------------------------ relative effective stress
  // eta^{trial}_{n+1} = s^{trial}_{n+1} - beta^{trial}_{n+1}
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> eta(Core::LinAlg::Initialization::zero);
  rel_stress(devstress, beta, eta);

  // J2 = 1/2 ( (eta11^{trial})^2 + (eta22^{trial})^2 + (eta33^{trial})^2
  //      + 2 . (eta12^{trial})^2 + 2 . (eta23^{trial})^2 + 2 . (eta13^{trial})^2)
  double J2 = 0.0;
  J2 = 1.0 / 2.0 * (eta(0) * eta(0) + eta(1) * eta(1) + eta(2) * eta(2)) + +eta(3) * eta(3) +
       eta(4) * eta(4) + eta(5) * eta(5);

  // trial effective relative stress
  // qbar^{trial}_{n+1} := qbar(eta^{trial}_{n+1}) = \sqrt{ 3 . J2 }
  double qbar = 0.0;
  qbar = sqrt(3.0 * J2);

  //---------------------------------------------------------------------------
  // check plastic admissibility, Phi<=0 is admissble
  //---------------------------------------------------------------------------

  // ----------------------------------------------- trial yield function

  // calculate the uniaxial yield stress considering linear isotropic hardening
  // sigma_y = sigma_y0 + Hiso . astrain^{p}_{n}
  //         = sigma_y0 + Hiso . astrain^{p, trial}_{n+1}
  double sigma_y = sigma_y0 + Hiso * strainbar_p;

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
  // (using the updated deviatoric stress eta_{n+1}, no longer eta^{trial}_{n+1})
  // N = sqrt{3/2} . Nbar = sqrt{3/2} . ( eta_{n+1} / || eta_{n+1} || )
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> N(Core::LinAlg::Initialization::zero);

  //-------------------------------------------------------------------
  // IF consistency condition is violated, i.e. plastic load step
  // (Phi_trial > 0.0, Dgamma >= 0.0)
  //-------------------------------------------------------------------
  if (Phi_trial > 1.0e-08)  // if (Phi^{trial} > 0.0)
  {
    // only first plastic call is output at screen for every processor
    // visualisation of whole plastic behaviour via PLASTIC_STRAIN in postprocessing
    if (plastic_step_ == false)
    {
      if (gp == 0) std::cout << "first plastic step: plastic strains unequal zero!" << std::endl;
      plastic_step_ = true;
    }

    // calculate kinematic hardening stress of old time step
    // beta_{n} = Hkin . astrain^p_{n} = Hkin . astrain^{p, trial}_{n+1}
    betabarold = Hkin * strainbar_p;

    // -------------------------------------------------- return-mapping

    // local Newton-Raphson

    // initialise
    const int itermax = 50;  // max. number of iterations
    int itnum = 0;           // iteration counter

    // Res:= residual of Newton iteration == yield function
    double Res = 0.0;
    // calculate residual derivative/tangent
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

      // linear isotropic hardening
      // sigma = sigma_y0 + Hiso . astrain^{p}_{n+1}
      sigma_y = sigma_y0 + Hiso * strainbar_p;


    }  // end of local Newton iteration

    // --------------------------------------------------- plastic update

    // ---------------------------------------------- update flow vectors

    // relative stress norm || eta_{n+1}^{trial} ||
    double etanorm = 0.0;
    etanorm = sqrt(eta(0) * eta(0) + eta(1) * eta(1) + eta(2) * eta(2) +
                   2.0 * (eta(3) * eta(3) + eta(4) * eta(4) + eta(5) * eta(5)));

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
    devstress.update(facdevstress, N, 1.0);

    // total stress
    // sigma_{n+1} = s_{n+1} + p_{n+1} . id2
    // pressure/volumetric stress no influence due to plasticity
    PlasticLinElast::stress(p, devstress, stress_view);

    // total strains
    // strain^e_{n+1} = strain^(e,trial)_{n+1} - Dgamma . N
    // compute converged engineering strain components (Voigt-notation)
    strain_e.update(1.0, trialstrain_e, (-Dgamma), N);

    // strain^p_{n+1} = strain^p_n + Dgamma . N
    strain_p.update(Dgamma, N, 1.0);

    // compute converged engineering strain components (Voigt-notation)
    for (int i = 3; i < 6; ++i) strain_e(i) *= 2.0;
    for (int i = 3; i < 6; ++i) strain_p(i) *= 2.0;

    // pass the current plastic strains to the element (for visualisation)
    plstrain.update(strain_p);

    // --------------------------------------------------- update history
    // plastic strain
    strainplcurr_->at(gp) = strain_p;

    // accumulated plastic strain
    strainbarplcurr_->at(gp) = strainbar_p;

    // back stress
    backstresscurr_->at(gp) = beta;


  }  // plastic corrector

  //-------------------------------------------------------------------
  // ELSE: elastic step (Phi_trial <= 0.0, Dgamma = 0.0)
  //-------------------------------------------------------------------
  else  // (Phi_trial <= 0.0)
  {
    // trial state vectors = result vectors of time step n+1
    // sigma^e_{n+1} = sigma^{e,trial}_{n+1} = s^{trial}_{n+1} + p . id2
    PlasticLinElast::stress(p, devstress, stress_view);

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
  setup_cmat_elasto_plastic(cmat_view, Dgamma, G, qbar, Nbar, heaviside, Hiso, Hkin);
}  // evaluate()


/*----------------------------------------------------------------------*
 | computes linear stress tensor                             dano 05/11 |
 *----------------------------------------------------------------------*/
void Mat::PlasticLinElast::stress(const double p,             // volumetric stress
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
void Mat::PlasticLinElast::rel_stress(
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
void Mat::PlasticLinElast::setup_cmat(
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
void Mat::PlasticLinElast::setup_cmat_elasto_plastic(
    Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
        cmat,                                     // elasto-plastic tangent modulus (out)
    double Dgamma,                                // plastic multiplier
    double G,                                     // shear modulus
    double q,                                     // elastic trial von Mises effective stress
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Nbar,  // unit flow vector
    double heaviside,                             // Heaviside function
    double Hiso,                                  // isotropic hardening modulus
    double Hkin                                   // kinematic hardening modulus
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

  // ------------------------------------------------------- elastic term
  // C_ep = C_e
  // add standard isotropic elasticity tensor C_e first
  setup_cmat(cmat);

  // ------------------------------------------------------ plastic terms

  // ------------------------------------------------- first plastic term
  // - ( H^ . Dgamma . 6 . G^2 ) / qbar^{trial} . I_d
  double epfac = 0.0;
  double epfac3 = 0.0;
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

  // loop strains (columns)
  for (int k = 0; k < 6; ++k)
  {
    // ---------------------------------------------------------- tangent
    // loop stresses (rows)
    for (int i = 0; i < 6; ++i)
    {
      if (q != 0.0)
      {
        epfac3 = heaviside * 6.0 * G * G * (Dgamma / q - 1.0 / (3.0 * G + Hkin + Hiso));
        cmat(i, k) += epfac3 * Nbar(i) * Nbar(k);
      }  // (q != 0.0)
    }  // end rows, loop i
  }  // end columns, loop k

  // complete material tangent C_ep available


}  // setup_cmat_elasto_plastic()


/*---------------------------------------------------------------------*
 | finite difference check for the material tangent.        dano 05/11 |
 | Meant for debugging only! (public)                                  |
 *---------------------------------------------------------------------*/
void Mat::PlasticLinElast::fd_check(
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& stress,  // updated stress sigma_n+1
    Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
        cmatFD,  // material tangent calculated with FD of stresses
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& beta,          // updated back stresses
    double p,                                              // volumetric stress
    const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& strain,  // elastic trial strain vector
    double Dgamma,                                         // plastic multiplier
    double G,                                              // shear modulus
    double qbar,                                // elastic trial von Mises effective stress
    double kappa,                               // bulk modulus
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& N,  // flow vector
    double heaviside                            // Heaviside function
)
{
  // *******************************************************************
  // FINITE DIFFERENCE check for cmat
  // *******************************************************************

  // clear the material tangent
  cmatFD.clear();

  // alloc the matrix that will store the perturbed values
  // strain matrices
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> disturbdevstrain(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> disturbstrain(Core::LinAlg::Initialization::zero);
  // initialise disturbed deviatoric stresses
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> devdisturbstress(Core::LinAlg::Initialization::zero);
  // initialise disturbed total stresses
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> disturbstress(Core::LinAlg::Initialization::zero);

  // second order identity
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> id2(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++) id2(i) = 1.0;

  // copy original strain to the storage matrix
  for (int i = 0; i < 6; ++i)
  {
    // insert total elastic strain for fd check
    disturbstrain(i) = strain(i);
  }
  // trial strains are already physical component (not Voigt notation), no Scaling

  // echo to screen
  printf("+-------------------------------------------+\n");
  printf("| FINITE DIFFERENCE CHECK FOR MATERIAL      |\n");
  printf("+-------------------------------------------+\n");
  printf("\n");

  // loop over all possible entries
  // cmat_ijkl = dsigma_ij /dstrain_kl
  // in matrix notation: cmat_ik = dsigma_i / dstrain_k
  // loop columns of matrix by looping strains(k) and rows by looping stresses(i)

  // loop strains (columns)
  for (int k = 0; k < 6; ++k)
  {
    printf("-------------------------------------\n");
    printf("-------------------------------------\n");
    printf("STRAIN term %d\n", k);

    // value of disturbance
    const double delta = 1.0e-8;
    // disturb the respective strain quantities
    disturbstrain(k) += delta;

    // ----------------------------------------------------------- strain
    // volumetric strain
    // trace of strain vector
    double tracestrain = (disturbstrain(0) + disturbstrain(1) + disturbstrain(2));
    // volstrain = 1/3 . tr( strain ) . Id
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> volumetricstrain(Core::LinAlg::Initialization::zero);
    volumetricstrain.update((tracestrain / 3.0), id2);

    // deviatoric strain
    // dev = strain - volstrain
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> devstrain(Core::LinAlg::Initialization::uninitialized);
    devstrain.update(1.0, disturbstrain, (-1.0), volumetricstrain);

    // ----------------------------------------------------------- stress
    // pressure = kappa . tr( strain ): saved as scalar
    double disturbp = kappa * tracestrain;

    // deviatoric stress = 2 . G . devstrain
    devdisturbstress.update(2.0 * G, devstrain);
    double devstressfac = 0.0;
    // update of trial state
    if (qbar != 0.0) devstressfac = 1.0 - heaviside * Dgamma * 3.0 * G / qbar;
    devdisturbstress.scale(devstressfac);

    // total disturb stress
    PlasticLinElast::stress(disturbp, devdisturbstress, disturbstress);

    // add the old back stress  to the total disturb stress
    double betafac = 0.0;
    if (qbar != 0.0) betafac = Dgamma * 3.0 * G / qbar;
    disturbstress.update(betafac, beta, 1.0);

    // ---------------------------------------------------------- tangent
    // loop stresses (rows)
    for (int i = 0; i < 6; ++i)
    {
      // build the finite difference tangent
      cmatFD(i, k) = 0.0;
      cmatFD(i, k) += (disturbstress(i) / (delta)-stress(i) / (delta));
    }  // loop stresses

    // undisturb the respective strain quantities (disturbstrain=strain)
    disturbstrain(k) -= delta;

  }  // loop strains

  // consider 1/2 in cmatFD for Voigt notation of the strains
  for (int i = 0; i < 6; ++i)
  {
    for (int k = 3; k < 6; ++k)
    {
      cmatFD(i, k) *= 0.5;
    }
  }  // loop stresses


  return;

}  // fd_check()


/*---------------------------------------------------------------------*
 | return names of visualization data (public)              dano 03/13 |
 *---------------------------------------------------------------------*/
void Mat::PlasticLinElast::vis_names(std::map<std::string, int>& names) const
{
  std::string accumulatedstrain = "accumulatedstrain";
  names[accumulatedstrain] = 1;  // scalar
}  // vis_names()


/*---------------------------------------------------------------------*
 | return visualization data (public)                       dano 03/13 |
 *---------------------------------------------------------------------*/
bool Mat::PlasticLinElast::vis_data(
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

FOUR_C_NAMESPACE_CLOSE
