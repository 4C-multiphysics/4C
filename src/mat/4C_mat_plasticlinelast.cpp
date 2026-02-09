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
#include "4C_linalg_tensor_generators.hpp"
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
    histsize = strainpllast_.size();
  }
  add_to_pack(data, histsize);  // Length of history vector(s)
  for (int var = 0; var < histsize; ++var)
  {
    // insert last converged states
    add_to_pack(data, strainpllast_.at(var));
    add_to_pack(data, backstresslast_.at(var));
    add_to_pack(data, strainbarpllast_.at(var));
    add_to_pack(data, betabarlast_.at(var));

    // insert current iteration states
    add_to_pack(data, strainplcurr_.at(var));
    add_to_pack(data, backstresscurr_.at(var));
    add_to_pack(data, strainbarplcurr_.at(var));
    add_to_pack(data, betabarcurr_.at(var));
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
  strainpllast_ = std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>>();
  strainplcurr_ = std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>>();

  // unpack back stress vectors (for kinematic hardening)
  backstresslast_ = std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>>();
  backstresscurr_ = std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>>();

  strainbarpllast_ = std::vector<double>();
  strainbarplcurr_ = std::vector<double>();

  betabarlast_ = std::vector<double>();
  betabarcurr_ = std::vector<double>();

  for (int var = 0; var < histsize; ++var)
  {
    Core::LinAlg::SymmetricTensor<double, 3, 3> tmp_tensor{};
    double tmp_scalar = 0.0;

    // last converged states are unpacked
    extract_from_pack(buffer, tmp_tensor);
    strainpllast_.push_back(tmp_tensor);
    extract_from_pack(buffer, tmp_tensor);
    backstresslast_.push_back(tmp_tensor);
    extract_from_pack(buffer, tmp_scalar);
    strainbarpllast_.push_back(tmp_scalar);
    extract_from_pack(buffer, tmp_scalar);
    betabarlast_.push_back(tmp_scalar);

    // current iteration states are unpacked
    extract_from_pack(buffer, tmp_tensor);
    strainplcurr_.push_back(tmp_tensor);
    extract_from_pack(buffer, tmp_tensor);
    backstresscurr_.push_back(tmp_tensor);
    extract_from_pack(buffer, tmp_scalar);
    strainbarplcurr_.push_back(tmp_scalar);
    extract_from_pack(buffer, tmp_scalar);
    betabarcurr_.push_back(tmp_scalar);
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
  strainpllast_ = std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>>();
  strainplcurr_ = std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>>();

  backstresslast_ = std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>>();
  backstresscurr_ = std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>>();

  strainbarpllast_ = std::vector<double>();
  strainbarplcurr_ = std::vector<double>();

  strainpllast_.resize(numgp);
  strainplcurr_.resize(numgp);

  backstresslast_.resize(numgp);
  backstresscurr_.resize(numgp);

  strainbarpllast_.resize(numgp);
  strainbarplcurr_.resize(numgp);

  betabarlast_.resize(numgp);
  betabarcurr_.resize(numgp);

  for (int i = 0; i < numgp; i++)
  {
    strainpllast_.at(i).fill(0.0);
    strainplcurr_.at(i).fill(0.0);

    backstresslast_.at(i).fill(0.0);
    backstresscurr_.at(i).fill(0.0);

    strainbarpllast_.at(i) = 0.0;
    strainbarplcurr_.at(i) = 0.0;

    betabarlast_.at(i) = 0.0;
    betabarcurr_.at(i) = 0.0;
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
  betabarlast_ = betabarcurr_;

  const int histsize = strainpllast_.size();

  for (int i = 0; i < histsize; i++)
  {
    strainplcurr_.at(i).fill(0.0);
    backstresscurr_.at(i).fill(0.0);

    strainbarplcurr_.at(i) = 0.0;
    betabarcurr_.at(i) = 0.0;
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
  double G = young / (2.0 * (1.0 + nu));
  // lambda = E * nu / ((1+nu) * (1-2*nu)) = K - 2/3 G
  const double lambda = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
  // bulk modulus kappa = E /( 3 ( 1 - 2 nu) )= lambda + 2/3 * mu
  double kappa = young / (3.0 * (1.0 - 2.0 * nu));

  // build Cartesian identity 2-tensor I_{AB}
  Core::LinAlg::SymmetricTensor<double, 3, 3> id2 =
      Core::LinAlg::TensorGenerators::identity<double, 3, 3>;

  // linstrain (in): independent variable passed from the element
  //  strain^p: evolution is determined by the flow rule, history variable
  //  strain^e: definition of additive decomposition:
  //  strain^e = strain - strain^p

  //---------------------------------------------------------------------------
  // elastic predictor (trial values)
  //---------------------------------------------------------------------------

  // ------------------------------------------------ old plastic strains
  // strain^{p,trial}_{n+1} = strain^p_n
  // accumulated/equivalent plastic strain
  strainplcurr_.at(gp) = strainpllast_.at(gp);

  // get old equivalent plastic strain only in case of plastic step
  // astrain^{p,trial}_{n+1} = astrain^p_n
  double strainbar_p = (strainbarpllast_.at(gp));
  if (strainbar_p < 0.0)
    FOUR_C_THROW("accumulated plastic strain has to be equal to or greater than zero!");

  // ----------------------------------------------- elastic trial strain
  // assume load step is elastic
  // strain^e_{n+1}
  Core::LinAlg::SymmetricTensor<double, 3, 3> strain_e{};

  // strain^{e,trial}_{n+1} = strain_{n+1} - strain^p_n
  Core::LinAlg::SymmetricTensor<double, 3, 3> trialstrain_e = glstrain - strainpllast_.at(gp);

  // volumetric strain
  double tracestrain = trace(trialstrain_e);
  Core::LinAlg::SymmetricTensor<double, 3, 3> volumetricstrain = id2 * (tracestrain / 3.0);

  // deviatoric strain
  Core::LinAlg::SymmetricTensor<double, 3, 3> devstrain = trialstrain_e - volumetricstrain;

  // ------------------------------------------------------- trial stress

  // pressure = kappa . tr( strain ): saved as scalar
  double p = kappa * tracestrain;

  // deviatoric stress = 2 . G . devstrain
  Core::LinAlg::SymmetricTensor<double, 3, 3> devstress = 2.0 * G * devstrain;

  // ------------------------------------------ relative effective stress
  // eta^{trial}_{n+1} = s^{trial}_{n+1} - beta^{trial}_{n+1}
  Core::LinAlg::SymmetricTensor<double, 3, 3> eta = devstress - backstresslast_.at(gp);

  // J2 = 1/2 ( (eta11^{trial})^2 + (eta22^{trial})^2 + (eta33^{trial})^2
  //      + 2 . (eta12^{trial})^2 + 2 . (eta23^{trial})^2 + 2 . (eta13^{trial})^2)
  double J2 = 0.5 * ddot(eta, eta);

  // trial effective relative stress
  // qbar^{trial}_{n+1} := qbar(eta^{trial}_{n+1}) = \sqrt{ 3 . J2 }
  double qbar = sqrt(3.0 * J2);

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
  double Phi_trial = qbar - sigma_y;

  // --------------------------------------------------------- initialise

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

  // flow vector N (Prandtl-Reuss)
  // (using the updated deviatoric stress eta_{n+1}, no longer eta^{trial}_{n+1})
  // N = sqrt{3/2} . Nbar = sqrt{3/2} . ( eta_{n+1} / || eta_{n+1} || )
  Core::LinAlg::SymmetricTensor<double, 3, 3> N{};

  //-------------------------------------------------------------------
  // IF consistency condition is violated, i.e. plastic load step
  // (Phi_trial > 0.0, Dgamma >= 0.0)
  //-------------------------------------------------------------------
  plastic_step_ = (Phi_trial > 1.0e-08);  // if (Phi^{trial} > 0.0)

  if (plastic_step_)  // PLASTIC LOADING
  {
    // calculate kinematic hardening stress of old time step
    // beta_{n} = Hkin . astrain^p_{n} = Hkin . astrain^{p, trial}_{n+1}
    betabarold = Hkin * strainbar_p;

    // -------------------------------------------------- return-mapping
    // local Newton-Raphson

    // Res:= residual of Newton iteration == yield function
    double Res = 0.0;
    // calculate residual derivative/tangent
    // ResTan = Phi' = d(Phi)/d(Dgamma)
    double ResTan = 0.0;

    // start iteration with index m for local Newton
    const int itermax = 50;  // max. number of iterations
    int itnum = 0;           // iteration counter
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

      // check for convergence (absolute value of Res has to be smaller than given tolerance)
      if (abs(Res) < (params_->abstol_))
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
      strainbar_p = strainbarpllast_.at(gp) + Dgamma;
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
    double etanorm = sqrt(ddot(eta, eta));

    // flow vector N = sqrt(3/2) eta_{n+1}^{trial} / || eta_{n+1}^{trial} ||
    N = sqrt(3.0 / 2.0) * eta / etanorm;  // N = sqrt(3/2) * Nbar

    // update relative stress eta_{n+1}, cf. (7.193)
    // eta = ( 1 - (Delta gamma / qbar_{n+1}^{trial}) . [ 3 . G + Hkin] ) eta_{n+1}^{trial}
    // H_iso is not needed for update of the stress
    eta *= 1.0 - ((Dgamma / qbar) * (3.0 * G + Hkin));

    // update back stress, cf. (7.197)
    // beta_{n+1} = beta_n + sqrt(2/3) . (betabar - betabarold) . eta / etanorm;
    // sqrt(2/3) N =  2/3 . ( sqrt(3/2) eta / etanorm)
    const double facbeta = 2.0 / 3.0 * (betabar - betabarold);
    backstresscurr_.at(gp) = backstresslast_.at(gp) + facbeta * N;
    betabarcurr_.at(gp) = betabar;

    // deviatoric stress
    // s = s_{n+1}^{trial} - 2 . G . Delta gamma . N
    const double facdevstress = (-2.0) * G * Dgamma;
    devstress += facdevstress * N;

    // total stress
    // sigma_{n+1} = s_{n+1} + p_{n+1} . id2
    // pressure/volumetric stress no influence due to plasticity
    stress = devstress + p * id2;

    // total strains
    // strain^e_{n+1} = strain^(e,trial)_{n+1} - Dgamma . N
    // compute converged engineering strain components
    strain_e = trialstrain_e - Dgamma * N;

    // update plastic strain
    // strain^p_{n+1} = strain^p_n + Dgamma . N
    strainplcurr_.at(gp) += Dgamma * N;
    // accumulated plastic strain
    strainbarplcurr_.at(gp) = strainbar_p;

  }  // plastic corrector

  //-------------------------------------------------------------------
  // ELSE: elastic step (Phi_trial <= 0.0, Dgamma = 0.0)
  //-------------------------------------------------------------------
  else  // (Phi_trial <= 0.0)
  {
    // trial state vectors = result vectors of time step n+1
    // sigma^e_{n+1} = sigma^{e,trial}_{n+1} = s^{trial}_{n+1} + p . id2
    stress = devstress + p * id2;

    // total strains
    // strain^e_{n+1} = strain^(e,trial)_{n+1}
    // compute converged engineering strain components (Voigt-notation)
    strain_e = trialstrain_e;

    // no plastic yielding
    Dgamma = 0.0;
    // kinematic hardening curve of current time step and old time step
    // betabar = Hkin * strainbar_p
    // linear kinematic hardening: Hkin = const., else: Hkin = Hkin(strainnbar_p)
    betabarold = 0.0;
    betabar = 0.0;

    // --------------------------------------------------------- update history
    // constant values for
    //  - plastic strains
    //  - accumulated plastic strains
    //  - back stress
    //    (--> relative stress)

    // as current history vectors are set to zero in update(), the old values
    // need to be set instead, otherwise no constant plastic values are possible
    strainplcurr_.at(gp) = strainpllast_.at(gp);
    strainbarplcurr_.at(gp) = strainbarpllast_.at(gp);
    backstresscurr_.at(gp) = backstresslast_.at(gp);
    betabarcurr_.at(gp) = betabarlast_.at(gp);


  }  // elastic step

  //---------------------------------------------------------------------------
  // --------------------------------- consistent elastoplastic tangent modulus
  //---------------------------------------------------------------------------
  setup_cmat(cmat, G, lambda);

  if (plastic_step_)
  {
    // using an associative flow rule: C_ep is symmetric
    // ( generally C_ep is nonsymmetric )
    // if Phi^trial = 0: two tangent stress-strain relations exist
    // plastic loading --> C == C_ep

    // using an associative flow rule: C_ep is symmetric
    // (generally C_ep is nonsymmetric)
    setup_cmat_elasto_plastic(cmat, Dgamma, G, qbar, N, Hiso, Hkin);
  }
}  // evaluate()



/*----------------------------------------------------------------------*
 | computes isotropic elasticity tensor in matrix notion     dano 04/11 |
 | for 3d                                                               |
 *----------------------------------------------------------------------*/
void Mat::PlasticLinElast::setup_cmat(
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, double mu, double lambda) const
{
  // isotropic elasticity tensor C in Voigt matrix notation, cf. FEscript p.29
  //                       [ 1-nu     nu     nu |          0    0    0 ]
  //                       [        1-nu     nu |          0    0    0 ]
  //           E           [               1-nu |          0    0    0 ]
  //   C = --------------- [ ~~~~   ~~~~   ~~~~   ~~~~~~~~~~  ~~~  ~~~ ]
  //       (1+nu)*(1-2*nu) [                    | (1-2*nu)/2    0    0 ]
  //                       [                    |      (1-2*nu)/2    0 ]
  //                       [ symmetric          |           (1-2*nu)/2 ]
  //

  constexpr auto id2 = Core::LinAlg::TensorGenerators::identity<double, 3, 3>;
  constexpr auto symmetric_id =
      Core::LinAlg::TensorGenerators::symmetric_identity<double, 3, 3, 3, 3>;

  // De = 2 G I_s + lambda I \otimes I   (acc. to eq. (4.51) with lambda = K - 2/3 G)
  cmat = 2 * mu * symmetric_id + lambda * dyadic(id2, id2);

}  // setup_cmat()


/*----------------------------------------------------------------------*
 | computes isotropic elasticity tensor in matrix notion     dano 05/11 |
 | for 3d                                                               |
 *----------------------------------------------------------------------*/
void Mat::PlasticLinElast::setup_cmat_elasto_plastic(
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>&
        cmat,       // elasto-plastic tangent modulus (in: elastic, out: elastoplastic)
    double Dgamma,  // plastic multiplier
    double G,       // shear modulus
    double q,       // elastic trial von Mises effective stress
    Core::LinAlg::SymmetricTensor<double, 3, 3> N,  // flow vector
    double Hiso,                                    // isotropic hardening modulus
    double Hkin                                     // kinematic hardening modulus
) const
{
  // incremental constitutive function for the stress tensor
  // sigma_{n+1} = [ cmat - (Dgamma 6 G^2/q) I_d ] : strain^{e,trial}_{n+1}
  // consistent tangent operator
  // D^{ep} := dsigma_{n+1} / dstrain^{e,trial}_{n+1}

  // depending on the flow vector Cmat_ep can be a fully-occupied matrix

  // C_ep = C_e - (Dgamma . 6 . G^2 ) / qbar^{trial} . I_d +
  //        +  6 . G^2 ( Dgamma/qbar^{trial} - 1/(3 G + Hkin + Hiso) ) Nbar \otimes Nbar
  // N \otimes N = (3/2) Nbar \otimes Nbar

  //         [ 2/3   -1/3  -1/3 | 0    0    0  ]
  //         [-1/3    2/3  -1/3 | 0    0    0  ]
  //         [-1/3   -1/3   2/3 | 0    0    0  ]
  //   I_d = [ ~~~~  ~~~~  ~~~~  ~~~  ~~~  ~~~ ]
  //         [                  | 1/2   0   0  ]
  //         [    symmetric     |      1/2  0  ]
  //         [                  |          1/2 ]
  //

  constexpr auto id2 = Core::LinAlg::TensorGenerators::identity<double, 3, 3>;
  constexpr auto Is = Core::LinAlg::TensorGenerators::symmetric_identity<double, 3, 3, 3, 3>;
  // I_d = Is - 1/3 id2 \otimes id2
  constexpr auto Id = Is - 1.0 / 3.0 * dyadic(id2, id2);

  // ------------------------------------------------- first plastic term
  // - ( Dgamma . 6 . G^2 ) / qbar^{trial} . I_d
  const double epfac = (-1.0) * Dgamma * 6.0 * G * G / q;
  cmat += epfac * Id;

  // ------------------------------------------------ second plastic term
  // +  H^ . 6 . G^2 ( Dgamma/qbar^{trial} - 1/(3 G + Hkin + Hiso) ) Nbar \otimes Nbar
  const double epfac3 = 6.0 * G * G * (Dgamma / q - 1.0 / (3.0 * G + Hkin + Hiso));
  cmat += epfac3 * 2. / 3. * dyadic(N, N);  // Nbar otimes Nbar = (2/3) N otimes N
  // complete material tangent C_ep available acc. to (7.213)

}  // setup_cmat_elasto_plastic()


/*---------------------------------------------------------------------*
 | finite difference check for the material tangent.        dano 05/11 |
 | Meant for debugging only! (public)                                  |
 *---------------------------------------------------------------------*/
void Mat::PlasticLinElast::fd_check(
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,  // updated stress sigma_n+1
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>&
        cmatFD,  // material tangent calculated with FD of stresses
    Core::LinAlg::SymmetricTensor<double, 3, 3>& beta,          // updated back stresses
    double p,                                                   // volumetric stress
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& strain,  // elastic trial strain vector
    double Dgamma,                                              // plastic multiplier
    double G,                                                   // shear modulus
    double qbar,                                    // elastic trial von Mises effective stress
    double kappa,                                   // bulk modulus
    Core::LinAlg::SymmetricTensor<double, 3, 3>& N  // flow vector
)
{
  // *******************************************************************
  // FINITE DIFFERENCE check for cmat
  // *******************************************************************

  // clear the material tangent
  cmatFD.fill(0);

  // alloc the matrix that will store the perturbed values
  // strain matrices
  Core::LinAlg::SymmetricTensor<double, 3, 3> disturbdevstrain{};
  Core::LinAlg::SymmetricTensor<double, 3, 3> disturbstrain{};
  // initialise disturbed deviatoric stresses
  Core::LinAlg::SymmetricTensor<double, 3, 3> devdisturbstress{};
  // initialise disturbed total stresses
  Core::LinAlg::SymmetricTensor<double, 3, 3> disturbstress{};

  // second order identity
  constexpr auto id2 = Core::LinAlg::TensorGenerators::identity<double, 3, 3>;

  // insert total elastic strain for fd check
  disturbstrain = strain;

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
  for (int k1 = 0; k1 < 3; ++k1)
    for (int k2 = 0; k2 < 3; ++k2)
    {
      printf("-------------------------------------\n");
      printf("-------------------------------------\n");
      printf("STRAIN term %d/%d\n", k1, k2);

      // value of disturbance
      const double delta = 1.0e-8;
      // disturb the respective strain quantities
      disturbstrain(k1, k2) += delta;

      // ----------------------------------------------------------- strain
      // volumetric strain
      // trace of strain vector
      double tracestrain = Core::LinAlg::trace(disturbstrain);
      // volstrain = 1/3 . tr( strain ) . Id
      Core::LinAlg::SymmetricTensor<double, 3, 3> volumetricstrain = tracestrain / 3.0 * id2;

      // deviatoric strain
      // dev = strain - volstrain
      disturbdevstrain = disturbstrain - volumetricstrain;

      // ----------------------------------------------------------- stress
      // deviatoric stress = 2 . G . devstrain
      devdisturbstress = 2.0 * G * disturbdevstrain;
      double devstressfac = 0.0;
      double betafac = 0.0;
      // update of trial state
      if (qbar != 0.0 && Dgamma > 0.0)
      {
        devstressfac = 1.0 - Dgamma * 3.0 * G / qbar;
        betafac = Dgamma * 3.0 * G / qbar;
        devdisturbstress *= devstressfac;
        devdisturbstress += betafac * beta;
      }
      // total stress = deviatoric + hydrostatic pressure . I
      // sigma = s + p . I
      disturbstress = devdisturbstress + p * id2;

      // ---------------------------------------------------------- tangent
      // loop stresses (rows)
      for (int i1 = 0; i1 < 3; ++i1)
        for (int i2 = 0; i2 < 3; ++i2)
        {
          // build the finite difference tangent
          cmatFD(i1, i2, k1, k2) += (disturbstress(i1, i2) - stress(i1, i2)) / (delta);
        }  // loop stresses

      // undisturb the respective strain quantities (disturbstrain=strain)
      disturbstrain(k1, k2) -= delta;

    }  // loop strains

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


/*---------------------------------------------------------------------*
 | return names of visualization data for direct VTK output            |
 *---------------------------------------------------------------------*/
void Mat::PlasticLinElast::register_output_data_names(
    std::unordered_map<std::string, int>& names_and_size) const
{
  names_and_size["accumulated_plastic_strain"] = 1;
  names_and_size["active_plasticity"] = 1;
  names_and_size["back_stress_magnitude"] = 1;
  names_and_size["back_stress"] = 6;
}


bool Mat::PlasticLinElast::evaluate_output_data(
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
  if (name == "back_stress_magnitude")
  {
    for (std::size_t gp = 0; gp < betabarcurr_.size(); ++gp)
    {
      data(gp, 0) = betabarcurr_.at(int(gp));
    }
    return true;
  }
  if (name == "back_stress")
  {
    for (std::size_t gp = 0; gp < backstresscurr_.size(); ++gp)
    {
      const double* values = backstresscurr_.at(gp).data();
      for (std::size_t i = 0; i < 6; ++i)
      {
        data(gp, i) = values[i];
      }
    }
    return true;
  }
  if (name == "active_plasticity")
  {
    for (std::size_t gp = 0; gp < strainbarplcurr_.size(); ++gp)
    {
      data(gp, 0) = (strainbarplcurr_.at(int(gp)) > strainbarpllast_.at(int(gp))) ? 1 : 0;
    }
    return true;
  }
  return false;
}


/*----------------------------------------------------------------------*/

FOUR_C_NAMESPACE_CLOSE
