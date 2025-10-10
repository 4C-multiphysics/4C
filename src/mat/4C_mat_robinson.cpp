// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_robinson.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix_solver.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_utils_enum.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | constructor (public)                                      dano 11/11 |
 *----------------------------------------------------------------------*/
Mat::PAR::Robinson::Robinson(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      kind_((matdata.parameters.get<std::string>("KIND"))),
      youngs_((matdata.parameters.get<std::vector<double>>("YOUNG"))),
      poissonratio_(matdata.parameters.get<double>("NUE")),
      density_(matdata.parameters.get<double>("DENS")),
      thermexpans_(matdata.parameters.get<double>("THEXPANS")),
      inittemp_(matdata.parameters.get<double>("INITTEMP")),
      hrdn_fact_(matdata.parameters.get<double>("HRDN_FACT")),
      hrdn_expo_(matdata.parameters.get<double>("HRDN_EXPO")),
      shrthrshld_((matdata.parameters.get<std::vector<double>>("SHRTHRSHLD"))),
      rcvry_(matdata.parameters.get<double>("RCVRY")),
      actv_ergy_(matdata.parameters.get<double>("ACTV_ERGY")),
      actv_tmpr_(matdata.parameters.get<double>("ACTV_TMPR")),
      g0_(matdata.parameters.get<double>("G0")),
      m_(matdata.parameters.get<double>("M_EXPO")),
      beta_((matdata.parameters.get<std::vector<double>>("BETA"))),
      h_(matdata.parameters.get<double>("H_FACT")),
      thermomat_(matdata.parameters.get<int>("THERMOMAT"))
{
}


/*----------------------------------------------------------------------*
 | is called in Material::Factory from read_materials()       dano 02/12 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::Robinson::create_material()
{
  return std::make_shared<Mat::Robinson>(this);
}


/*----------------------------------------------------------------------*
 |                                                           dano 02/12 |
 *----------------------------------------------------------------------*/
Mat::RobinsonType Mat::RobinsonType::instance_;


/*----------------------------------------------------------------------*
 | is called in Material::Factory from read_materials()       dano 02/12 |
 *----------------------------------------------------------------------*/
Core::Communication::ParObject* Mat::RobinsonType::create(Core::Communication::UnpackBuffer& buffer)
{
  Mat::Robinson* robinson = new Mat::Robinson();
  robinson->unpack(buffer);
  return robinson;
}


/*----------------------------------------------------------------------*
 | constructor (public) --> called in Create()               dano 11/11 |
 *----------------------------------------------------------------------*/
Mat::Robinson::Robinson() : params_(nullptr), thermo_(nullptr) {}


/*----------------------------------------------------------------------*
 | copy-constructor (public) --> called in create_material()  dano 11/11 |
 *----------------------------------------------------------------------*/
Mat::Robinson::Robinson(Mat::PAR::Robinson* params) : params_(params), thermo_(nullptr)
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
 | pack (public)                                             dano 11/11 |
 *----------------------------------------------------------------------*/
void Mat::Robinson::pack(Core::Communication::PackBuffer& data) const
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
  int numgp;
  // if material is not initialised, i.e. start simulation, nothing to pack
  if (!initialized())
  {
    numgp = 0;
  }
  else
  {
    // if material is initialised (restart): size equates number of gausspoints
    numgp = strainpllast_->size();
  }
  add_to_pack(data, numgp);  // Length of history vector(s)
  for (int var = 0; var < numgp; ++var)
  {
    // insert last converged states
    add_to_pack(data, strainpllast_->at(var));
    add_to_pack(data, backstresslast_->at(var));
    add_to_pack(data, kvarva_->at(var));
    add_to_pack(data, kvakvae_->at(var));
    add_to_pack(data, strain_last_.at(var));

    // insert current iteration states
    add_to_pack(data, strainplcurr_->at(var));
    add_to_pack(data, backstresscurr_->at(var));
  }
}  // pack()


/*----------------------------------------------------------------------*
 | unpack (public)                                           dano 11/11 |
 *----------------------------------------------------------------------*/
void Mat::Robinson::unpack(Core::Communication::UnpackBuffer& buffer)
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
        params_ = static_cast<Mat::PAR::Robinson*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }

  // history data
  int numgp;
  extract_from_pack(buffer, numgp);

  // if system is not yet initialised, the history vectors have to be initialized
  if (numgp == 0) isinit_ = false;

  // unpack strain vectors
  strainpllast_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  strainplcurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();

  // unpack back stress vectors (for kinematic hardening)
  backstresslast_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  backstresscurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();

  // unpack matrices needed for condensed system
  kvarva_ = std::make_shared<std::vector<Core::LinAlg::Matrix<2 * NUM_STRESS_3D, 1>>>();
  kvakvae_ =
      std::make_shared<std::vector<Core::LinAlg::Matrix<2 * NUM_STRESS_3D, NUM_STRESS_3D>>>();
  strain_last_.resize(0);

  for (int var = 0; var < numgp; ++var)
  {
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> tmp_vector(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<2 * Mat::NUM_STRESS_3D, 1> tmp_vector2(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<2 * Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D> tmp_matrix(
        Core::LinAlg::Initialization::zero);

    // unpack strain/stress vectors of last converged state
    extract_from_pack(buffer, tmp_vector);
    strainpllast_->push_back(tmp_vector);
    extract_from_pack(buffer, tmp_vector);
    backstresslast_->push_back(tmp_vector);

    // unpack matrices of last converged state
    extract_from_pack(buffer, tmp_vector2);
    kvarva_->push_back(tmp_vector2);
    extract_from_pack(buffer, tmp_matrix);
    kvakvae_->push_back(tmp_matrix);

    extract_from_pack(buffer, tmp_vector);
    strain_last_.push_back(tmp_vector);

    // current iteration states are unpacked
    extract_from_pack(buffer, tmp_vector);
    strainplcurr_->push_back(tmp_vector);
    extract_from_pack(buffer, tmp_vector);
    backstresscurr_->push_back(tmp_vector);
  }



  return;

}  // unpack()


/*---------------------------------------------------------------------*
 | initialise / allocate internal stress variables (public) dano 11/11 |
 *---------------------------------------------------------------------*/
void Mat::Robinson::setup(const int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  // temporary variable for read-in
  std::string buffer;

  // initialise history variables
  strainpllast_ = std::make_shared<std::vector<Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>>>();
  strainplcurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>>>();

  backstresslast_ = std::make_shared<std::vector<Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>>>();
  backstresscurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>>>();

  kvarva_ = std::make_shared<std::vector<Core::LinAlg::Matrix<2 * Mat::NUM_STRESS_3D, 1>>>();
  kvakvae_ = std::make_shared<
      std::vector<Core::LinAlg::Matrix<2 * Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D>>>();

  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> emptymat(Core::LinAlg::Initialization::zero);
  strainpllast_->resize(numgp);
  strainplcurr_->resize(numgp);
  strain_last_.resize(numgp, Core::LinAlg::Matrix<6, 1>(Core::LinAlg::Initialization::zero));

  backstresslast_->resize(numgp);
  backstresscurr_->resize(numgp);

  Core::LinAlg::Matrix<2 * Mat::NUM_STRESS_3D, 1> emptymat2(Core::LinAlg::Initialization::zero);
  kvarva_->resize(numgp);
  Core::LinAlg::Matrix<2 * Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D> emptymat3(
      Core::LinAlg::Initialization::zero);
  kvakvae_->resize(numgp);

  for (int i = 0; i < numgp; i++)
  {
    strainpllast_->at(i) = emptymat;
    strainplcurr_->at(i) = emptymat;

    backstresslast_->at(i) = emptymat;
    backstresscurr_->at(i) = emptymat;

    kvarva_->at(i) = emptymat2;
    kvakvae_->at(i) = emptymat3;
  }

  isinit_ = true;

  return;

}  // setup()


/*---------------------------------------------------------------------*
 | update after time step                          (public) dano 11/11 |
 *---------------------------------------------------------------------*/
void Mat::Robinson::update()
{
  // make current values at time step t_n+1 to values of last step t_n
  // x_n := x_n+1
  strainpllast_ = strainplcurr_;
  backstresslast_ = backstresscurr_;
  // the matrices do not have to be updated. They are reset after each time step

  // empty vectors of current data
  strainplcurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>>>();
  backstresscurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>>>();

  // get size of the vector
  // (use the last vector, because it includes latest results, current is empty)
  const int numgp = strainpllast_->size();
  strainplcurr_->resize(numgp);
  backstresscurr_->resize(numgp);

  const Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> emptymat(Core::LinAlg::Initialization::zero);
  const Core::LinAlg::Matrix<2 * Mat::NUM_STRESS_3D, 1> emptymat1(
      Core::LinAlg::Initialization::zero);
  const Core::LinAlg::Matrix<2 * Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D> emptymat2(
      Core::LinAlg::Initialization::zero);

  for (int i = 0; i < numgp; i++)
  {
    strainplcurr_->at(i) = emptymat;
    backstresscurr_->at(i) = emptymat;
  }

  return;

}  // update()


/*----------------------------------------------------------------------*
 | evaluate material (public)                                dano 11/11 |
 | select Robinson's material, integrate internal variables and return  |
 | stress and material tangent                                          |
 *----------------------------------------------------------------------*/
void Mat::Robinson::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  const Core::LinAlg::Matrix<6, 1> glstrain_mat =
      Core::LinAlg::make_strain_like_voigt_matrix(glstrain);
  Core::LinAlg::Matrix<6, 1> stress_view = Core::LinAlg::make_stress_like_voigt_view(stress);
  Core::LinAlg::Matrix<6, 6> cmat_view = Core::LinAlg::make_stress_like_voigt_view(cmat);

  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> straininc(glstrain_mat);
  straininc.update(-1., strain_last_[gp], 1.);
  strain_last_[gp] = glstrain_mat;
  // if no temperature has been set use the initial value
  const double scalartemp = get_or(params, "temperature", init_temp());

  // update history of the condensed variables plastic strain and back stress
  // iterative update of the current history vectors at current Gauss point gp
  iterative_update_of_internal_variables(gp, straininc);

  // name:                 4C
  // total strain:         strain
  // elastic strain:       strain_e
  // thermal strain:       strain_t
  // viscous strain:       strain_p
  // stress deviator:      devstress
  // back stress:          backstress/beta
  // over/relative stress: eta

  // ----------------------------------------------------------------
  // implementation is identical for linear and Green-Lagrange strains
  // strains are calculated on element level and is passed to the material
  // --> material has no idea what strains are passed
  //     --> NO kintype is needed

  // get material parameters
  FOUR_C_ASSERT(context.time_step_size, "Time step size not given in evaluation context.");
  const double dt_ = *context.time_step_size;

  // build Cartesian identity 2-tensor I_{AB}
  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> id2(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++) id2(i) = 1.0;
  // set Cartesian identity 4-tensor in 6x6-matrix notation (stress-like)
  // this is a 'mixed co- and contra-variant' identity 4-tensor, ie I^{AB}_{CD}
  // REMARK: rows are stress-like 6-Voigt
  //         columns are strain-like 6-Voigt
  Core::LinAlg::Matrix<6, 6> id4(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 6; i++) id4(i, i) = 1.0;

  // -------------------------------- temperatures and thermal strain
  // get initial and current temperature
  // initial temperature at Gauss point
  const double tempinit = params_->inittemp_;
  // initialise the thermal expansion coefficient
  const double thermexpans = params_->thermexpans_;
  // thermal strain vector
  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> strain_t(Core::LinAlg::Initialization::zero);
  // update current temperature at Gauss point
  for (int i = 0; i < 3; ++i) strain_t(i) = thermexpans * (scalartemp - tempinit);
  // for (int i=3; i<6; ++i){ strain_t(i) = 2E_xy = 2E_yz = 2E_zx = 0.0; }

  // ------------------------------------------------- viscous strain
  // viscous strain strain_{n+1}^{v,i} at t_{n+1}
  // use the newest plastic strains here, i.e., from latest Newton iteration
  // (strainplcurr_), not necessarily equal to newest temporal strains (last_)
  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> strain_on(
      Core::LinAlg::Initialization::uninitialized);
  strain_on.update(strainplcurr_->at(gp));
  // get history vector of old visco-plastic strain at t_n
  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> strain_p(Core::LinAlg::Initialization::uninitialized);
  strain_p.update(strainpllast_->at(gp));

  // ------------------------------------------------- elastic strain
  // elastic strain at t_{n+1}
  // strain^{e}_{n+1}
  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> strain_e(Core::LinAlg::Initialization::uninitialized);

  // strain^e_{n+1} = strain_n+1 - strain^p_n - strain^t
  strain_e.update(glstrain_mat);
  strain_e.update((-1.0), strain_on, (-1.0), strain_t, 1.0);

  // ---------------------------------------------- elasticity tensor
  // cmat = kee = pd(sig)/pd(eps)
  // pass the current temperature to calculate the current youngs modulus
  setup_cmat(scalartemp, cmat_view);

  // ------------------------------------ tangents of stress equation
  // declare single terms of elasto-plastic tangent Cmat_ep

  // kev = pd(sigma)/pd(eps^v)
  // tangent term resulting from linearisation \frac{\pd sig}{\pd eps^v}
  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D> kev(
      Core::LinAlg::Initialization::zero);
  // assign vector by another vector and scale it
  // (i): scale (-1.0)
  // (i): input matrix cmat
  // (o): output matrix kev
  kev.update((-1.0), cmat_view);

  // kea = pd(sigma)/pd(backstress)
  // tangent term resulting from linearisation \frac{\pd sig}{\pd al}
  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D> kea(
      Core::LinAlg::Initialization::zero);
  // initialise with 1 on the diagonals
  kea.update(id4);

  // ------------------------------------------------- elastic stress
  // stress sig_{n+1}^i at t_{n+1} */
  // (i): input matrix cmat
  // (i): input vector strain_e
  // (o): output vector stress
  // stress_{n+1} = cmat . strain^e_{n+1}
  stress_view.multiply_nn(cmat_view, strain_e);

  // ------------------------------------------------------ devstress
  // deviatoric stress s_{n+1}^i at t_{n+1}
  // calculate the deviator from stress
  // CAUTION: s = 2G . devstrain only in case of small strain
  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> devstress(
      Core::LinAlg::Initialization::uninitialized);
  // trace of stress vector
  double tracestress = (stress_view(0) + stress_view(1) + stress_view(2));
  for (int i = 0; i < 3; i++) devstress(i) = stress_view(i) - tracestress / 3.0;
  for (int i = 3; i < Mat::NUM_STRESS_3D; i++) devstress(i) = stress_view(i);
  // CAUTION: shear stresses (e.g., sigma_12)
  // in Voigt-notation the shear strains (e.g., strain_12) have to be scaled with
  // 1/2 normally considered in material tangent (using id4sharp, instead of id4)

  // ---------------------------------------------------- back stress
  // new back stress at t_{n+1} backstress_{n+1}^i
  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> backstress_n(
      Core::LinAlg::Initialization::uninitialized);
  backstress_n.update(backstresscurr_->at(gp));
  // old back stress at t_{n} backstress_{n}
  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> backstress(
      Core::LinAlg::Initialization::uninitialized);
  backstress.update(backstresslast_->at(gp));

  // ------------------------------------------ over/relativestress
  // overstress Sig_{n+1}^i = s_{n+1}^i - al_{n+1}^i
  // (i): input vector devstn
  // (i): input vector backstress_n
  // (o): output vector stsovr: subtract 2 vectors
  // eta_{n+1} = devstress_{n+1} - backstress_{n+1}
  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> eta(Core::LinAlg::Initialization::zero);
  rel_dev_stress(devstress, backstress_n, eta);

  // to calculate the new history vectors (strainplcurr_, backstresscurr_), the
  // submatrices of the complete problems, that are condensed later, have to be calculated

  // ---------------------- residual of viscous strain, kve, kvv, kva
  // residual of visc. strain eps_{n+1}^<i> and its consistent tangent for <i>
  // tangent term resulting from linearisation \frac{\pd res^v}{\pd eps}
  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D> kve(
      Core::LinAlg::Initialization::zero);
  // tangent term resulting from linearisation \frac{\pd res^v}{\pd eps^v}
  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D> kvv(
      Core::LinAlg::Initialization::zero);
  // tangent term resulting from linearisation \frac{\pd res^v}{\pd al}
  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D> kva(
      Core::LinAlg::Initialization::zero);
  // initialise the visco-plastic strain residual
  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> strain_pres(Core::LinAlg::Initialization::zero);
  calc_be_viscous_strain_rate(
      dt_, scalartemp, strain_p, strain_on, devstress, eta, strain_pres, kve, kvv, kva);

  // ------------------------- residual of back stress, kae, kav, kaa
  // residual of back stress al_{n+1} and its consistent tangent
  // initialise the sub matrices needed for evaluation of the complete coupled
  // problem
  // tangent term resulting from linearisation \frac{\pd res^al}{\pd eps}
  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D> kae(
      Core::LinAlg::Initialization::zero);
  // tangent term resulting from linearisation \frac{\pd res^al}{\pd eps^v}
  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D> kav(
      Core::LinAlg::Initialization::zero);
  // tangent term resulting from linearisation \frac{\pd res^al}{\pd al}
  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D> kaa(
      Core::LinAlg::Initialization::zero);
  // initialise the back stress residual
  // back stress (residual): beta/backstress --> bckstsr
  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> backstress_res(Core::LinAlg::Initialization::zero);
  calc_be_back_stress_flow(dt_, scalartemp, strain_p, strain_on, devstress, backstress,
      backstress_n, backstress_res, kae, kav, kaa);

  // build reduced system by condensing the evolution equations: only stress
  // equation remains
  // ------------------------------------- reduced stress and tangent
  // ==> static condensation
  calculate_condensed_system(stress_view,  // (o): sigma_red --> used to calculate f_int^{e} at GP
      cmat_view,                           // (o): C_red --> used to calculate stiffness k^{e} at GP
      kev, kea, strain_pres, kve, kvv, kva, backstress_res, kae, kav, kaa, (kvarva_->at(gp)),
      (kvakvae_->at(gp)));
}  // evaluate()

/*----------------------------------------------------------------------*
 | Set current quantities for this material                             |
 *----------------------------------------------------------------------*/
void Mat::Robinson::reinit(const Core::LinAlg::Tensor<double, 3, 3>* defgrd,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, double temperature, unsigned gp)
{
  reinit(temperature, gp);
}

/*----------------------------------------------------------------------*
 | calculate stress-temperature modulus and thermal derivative          |
 |   for coupled thermomechanics                                        |
 *----------------------------------------------------------------------*/
void Mat::Robinson::stress_temperature_modulus_and_deriv(
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stm,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stm_dT, int gp)
{
  stm = {};
  stm_dT = {};
}

/*----------------------------------------------------------------------*
 |  Evaluates the added derivatives of the stress w.r.t. all scalars    |
 *----------------------------------------------------------------------*/
Core::LinAlg::SymmetricTensor<double, 3, 3> Mat::Robinson::evaluate_d_stress_d_scalar(
    const Core::LinAlg::Tensor<double, 3, 3>& defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext& context, int gp, int eleGID)
{
  return {};
}

/*----------------------------------------------------------------------*
 | computes isotropic elasticity tensor in matrix notion     dano 11/11 |
 | for 3d                                                               |
 *----------------------------------------------------------------------*/
void Mat::Robinson::setup_cmat(
    double tempnp, Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D>& cmat) const
{
  // get material parameters
  // Young's modulus
  double emod = get_mat_parameter_at_tempnp(&(params_->youngs_), tempnp);
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
  const double mfac = emod / ((1.0 + nu) * (1.0 - 2.0 * nu));  // factor

  // clear the material tangent
  cmat.clear();
  // write non-zero components --- axial
  cmat(0, 0) = mfac * (1.0 - nu);
  cmat(0, 1) = mfac * nu;
  cmat(0, 2) = mfac * nu;
  cmat(1, 0) = mfac * nu;
  cmat(1, 1) = mfac * (1.0 - nu);
  cmat(1, 2) = mfac * nu;
  cmat(2, 0) = mfac * nu;
  cmat(2, 1) = mfac * nu;
  cmat(2, 2) = mfac * (1.0 - nu);
  // write non-zero components --- shear
  cmat(3, 3) = mfac * 0.5 * (1.0 - 2.0 * nu);
  cmat(4, 4) = mfac * 0.5 * (1.0 - 2.0 * nu);
  cmat(5, 5) = mfac * 0.5 * (1.0 - 2.0 * nu);

}  // setup_cmat()


/*----------------------------------------------------------------------*
 | computes linear stress tensor                             dano 11/11 |
 *----------------------------------------------------------------------*/
void Mat::Robinson::stress(const double p,                         //!< volumetric stress
    const Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>& devstress,  //!< deviatoric stress tensor
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>& stress            //!< 2nd PK-stress
) const
{
  // total stress = deviatoric + hydrostatic pressure . I
  // sigma = s + p . I
  stress.update(devstress);
  for (int i = 0; i < 3; ++i) stress(i) += p;

}  // Stress()


/*----------------------------------------------------------------------*
 | compute relative deviatoric stress tensor                 dano 11/11 |
 *----------------------------------------------------------------------*/
void Mat::Robinson::rel_dev_stress(
    const Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>& devstress,  // (i) deviatoric stress tensor
    const Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>& backstress_n,  // (i) back stress tensor
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>& eta                  // (o) relative stress
) const
{
  // relative stress = deviatoric - back stress
  // eta = s - backstress
  eta.update(1.0, devstress, (-1.0), backstress_n);

}  // RelDevStress()


/*----------------------------------------------------------------------*
 | residual of BE-discretised viscous strain rate            dano 11/11 |
 | at Gauss point                                                       |
 *----------------------------------------------------------------------*/
void Mat::Robinson::calc_be_viscous_strain_rate(const double dt,  // (i) time step size
    double tempnp,                                                // (i) current temperature
    const Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>& strain_p,  // (i) viscous strain at t_n
    const Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>&
        strain_on,  // (i) viscous strain at t_{n+1}^<i>
    const Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>&
        devstress,                                             // (i) stress deviator at t_{n+1}^<i>
    const Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>& eta,    // (i) over stress at t_{n+1}^<i>
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>& strain_pres,  // (i,o) viscous strain residual
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D>& kve,  // (i,o)
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D>& kvv,  // (i,o)
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D>& kva   // (i,o)
) const
{
  // strain_on' = f^v in a plastic load step
  // at the end of a time step, i.e. equilibrium, the residual has to vanish
  // r^v = ( strain_on - strain_p ) / dt - f^v != 0
  //
  // discretising the flow equation strain_on' with a backward Euler scheme
  // --> strain_on' = ( strain_on - strain_p ) / dt

  // initialise hardening exponent 'N' -->  nn
  double nn = params_->hrdn_expo_;

  // identity tensor in vector notation
  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> id2(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++) id2(i) = 1.0;

  // -------------------------------------------------- preliminaries
  // J2-invariant
  // J2 = 1/2 eta : eta
  // J2 = 1/2 (eta11^2 + eta22^2 + eta33^2 + 2 . eta12^2 + 2 . eta23^2 + 2 . eta13^2)
  //         2.) j2 *= 0.5;
  //         --> J_2 = 1/2 * Sig : Sig  with Sig...overstress
  double J2 = 0.0;
  J2 = 1.0 / 2.0 * (eta(0) * eta(0) + eta(1) * eta(1) + eta(2) * eta(2)) + +eta(3) * eta(3) +
       eta(4) * eta(4) + eta(5) * eta(5);

  // Bingham-Prager shear stress threshold at current temperature 'K^2'
  double kksq = 0.0;
  kksq = get_mat_parameter_at_tempnp(&(params_->shrthrshld_), tempnp);

  // F = (J_2 - K^2) / K^2 = (J_2 / K^2) - 1
  double ff = 0.0;
  if (fabs(kksq) <= 1e-10)  // shrthrshld = kksq
  {
    ff = -1.0;
    FOUR_C_THROW("Division by zero: Shear threshold very close to zero");
  }
  else
  {
    ff = (J2 - kksq) / kksq;
  }

  // hardening factor 'A' --> aa
  // calculate the temperature dependent material constant \bar{\mu} := aa
  double aa = 0.0;
  if (params_->kind_ == "Arya_CrMoSteel")
  {
    double mu = params_->hrdn_fact_;
    // calculate theta1 used for the material constant \bar{\mu}
    // \bar{\mu} = (23.8 . tempnp - 2635.0) . (1.0/811.0 - 1.0/tempnp)), vgl. (14)
    double th1 = (23.8 * tempnp - 2635.0) * (1.0 / 811.0 - 1.0 / tempnp);
    if (std::isinf(th1)) FOUR_C_THROW("Infinite theta1");
    // theory is the same as in literature, but present implementation differs, e.g.,
    // here: A == \bar{\mu} = 0.5/(mu exp(theta1)) = 1/(2 mu exp(theta1)
    // cf. Arya: \bar{\mu} := \mu . exp(- theta1), vgl. (12), f(F) includes mu
    aa = 0.5 / (mu * std::exp(-th1));
  }
  else  // "Butler","Arya","Arya_NarloyZ"
  {
    aa = params_->hrdn_fact_;
  }

  // se = 1/2 * devstress : eta
  // with  eta: overstress, devstress:.deviat.stress
  double se = 0.0;
  se = 1.0 / 2.0 * (devstress(0) * eta(0) + devstress(1) * eta(1) + devstress(2) * eta(2)) +
       +devstress(3) * eta(3) + devstress(4) * eta(4) + devstress(5) * eta(5);

  // viscous strain rate
  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> strainrate_p(Core::LinAlg::Initialization::zero);
  //-------------------------------------------------------------------
  // IF plastic step ( F > 0.0, (1/2 * devstress : eta) > 0.0 )
  //-------------------------------------------------------------------
  if ((ff > 0.0) and (se > 0.0))
  {
    //    std::cout << "plastic step: viscous strain rate strain_p'!= 0" << std::endl;

    // inelastic/viscous strain residual
    // epsilon_p' = F^n / sqrt(j2) . eta

    // fct = A . F^n / (J_2)^{1/2}
    double fct = aa * std::pow(ff, nn) / sqrt(J2);
    // calculate the viscous strain rate respecting that strain vector component
    // has a doubled shear component, but stress vectors not!
    // --> scale shear components accordingly
    for (unsigned i = 0; i < 3; i++)
    {
      strainrate_p(i) = eta(i);
    }
    for (int i = 3; i < Mat::NUM_STRESS_3D; i++)
    {
      strainrate_p(i) = 2.0 * eta(i);
    }
    strainrate_p.scale(fct);
  }  // inelastic
  //-------------------------------------------------------------------
  // ELSE IF elastic step (F <= 0.0, se <= 0.0)
  //-------------------------------------------------------------------
  else
  {
    // elastic step, no inelastic strains: strain_n^v' == 0
    // --> dvstcstn == strainrate_p
    strainrate_p.put_scalar(0.0);
  }  // elastic

  //-------------------------------------------------------------------
  // --------------------- residual of viscous strain rate at t_{n+1}
  // res_{n+1}^v = (strain_{n+1}^v - strain_n^v)/dt - d_eps_{n+1}^v
  // strainrate_p corresponds to f^v(d_{n+1}, strain^p_{n+1}, alpha_{n+1}, T_{n+1})
  for (int i = 0; i < Mat::NUM_STRESS_3D; i++)
  {
    // strain_pres = 1/dt * (strain_{n+1}^v - strain_{n}^v - dt * strain_n^v')

    // here is the essential!!!
    strain_pres(i) = (strain_on(i) - strain_p(i) - dt * strainrate_p(i)) / dt;
  }


  //-------------------------------------------------------------------

  // --------------------------------- derivative of viscous residual
  // derivative of viscous residual with respect to over stress eta
  // kvs = d(strain_pres) / d (eta)
  // kvs[NUMSTR_SOLID3][NUMSTR_SOLID3];
  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D> kvs(
      Core::LinAlg::Initialization::zero);
  //-------------------------------------------------------------------
  // IF plastic step ( F > 0.0, (1/2 * devstress : eta) > 0.0 )
  //-------------------------------------------------------------------
  if ((ff > 0.0) and (se > 0.0))
  {
    // add facu to all diagonal terms matrix kvs
    // facu = - A . F^n / sqrt(J2)
    double facu = -aa * std::pow(ff, nn) / sqrt(J2);
    for (int i = 0; i < Mat::NUM_STRESS_3D; i++) kvs(i, i) = facu;

    // update matrix with scaled dyadic vector product
    // contribution: kvs = kvs + faco . (eta \otimes eta^T)
    // faco = -n . a . F^(n-1) / (kappa . sqrt(J2)) + a . F^n / (2. J2^{1.5})
    double faco = -nn * aa * std::pow(ff, (nn - 1.0)) / (kksq * sqrt(J2)) +
                  aa * std::pow(ff, nn) / (2.0 * std::pow(J2, 1.5));
    kvs.multiply_nt(faco, eta, eta, 1.0);
    // multiply last 3 rows by 2 to conform with definition of strain vectors
    // consider the difference between physical strains and Voigt notation
    for (int i = 3; i < Mat::NUM_STRESS_3D; i++)
    {
      for (int j = 0; j < Mat::NUM_STRESS_3D; j++)
      {
        kvs(i, j) *= 2.0;
      }
    }  // rows 1--6
  }  // inelastic
  //-------------------------------------------------------------------
  // ELSE IF elastic step (F <= 0.0, 1/2 (devstress . eta) < 0.0)
  //-------------------------------------------------------------------
  else
  {
    // so3_mv6_m_zero(kvs);
    // in case of an elastic step, no contribution to kvs
    kvs.put_scalar(0.0);
  }

  //-------------------------------------------------------------------
  // - derivative of viscous residual with respect to total strain eps
  // kve = ( pd strain_pres^{n+1} )/ (pd strain^{n+1})|^<i>

  //-------------------------------------------------------------------
  // IF plastic step ( F > 0.0, (1/2 * devstress : eta) > 0.0 )
  //-------------------------------------------------------------------
  if ((ff > 0.0) and (se > 0.0))
  {
    // calculate elastic material tangent with temperature-dependent Young's modulus
    // kse = (pd eta) / (pd strain)
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D> kse(
        Core::LinAlg::Initialization::zero);
    // pass the current temperature to calculate the current youngs modulus
    setup_cmat(tempnp, kse);
    // Matrix vector product: cid2 = kse(i,j)*id2(j)
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> cid2(Core::LinAlg::Initialization::uninitialized);
    cid2.multiply(kse, id2);
    // update matrix with scaled dyadic vector product
    // contribution: kse = kse + (-1/3) . (id2 \otimes cid2^T)
    kse.multiply_nt((-1.0 / 3.0), id2, cid2, 1.0);

    // assign kve by matrix-matrix product kvs . kse (inner product)
    // kve = kvs . kse
    // kve(i,j) = kvs(i,k).kse(k,j)
    kve.multiply_nn(kvs, kse);
  }  // inelastic
  //-------------------------------------------------------------------
  // ELSE IF elastic step (F <= 0.0, (devstress . eta) <= 0.0)
  //-------------------------------------------------------------------
  else
  {
    // in case of an elastic step, no contribution to kvs
    kve.put_scalar(0.0);
  }  // elastic

  //-------------------------------------------------------------------
  // derivative of viscous residual with respect to viscous strain strain_p

  //-------------------------------------------------------------------
  // IF plastic step ( F > 0.0, (1/2 * devstress : eta) > 0.0 )
  //-------------------------------------------------------------------
  if ((ff > 0.0) and (se > 0.0))
  {
    // derivative ksv = (pd eta) / (pd strain_p)
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D> ksv(
        Core::LinAlg::Initialization::zero);
    // pass the current temperature to calculate the current youngs modulus
    setup_cmat(tempnp, ksv);

    // Matrix vector product: cid2 = kse(i,j)*id2(j)
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> cid2(Core::LinAlg::Initialization::uninitialized);
    cid2.multiply(ksv, id2);
    // update matrix with scaled dyadic vector product
    // contribution: ksv = ksv + (-1/3) . (id2 \otimes cid2^T)
    ksv.multiply_nt((-1.0 / 3.0), id2, cid2, 1.0);

    // kvv = d(res^v)/d(esp^v) + pd(res^v)/pd(Sig) . pd(Sig)/pd(eps^v)
    // kvv = 1/dt * Id  +  kvs . ksv
    // scale matrix kvv with 1.0/dt
    for (int i = 0; i < Mat::NUM_STRESS_3D; i++) kvv(i, i) = 1.0 / dt;
    // assign matrix kvv by scaled matrix-matrix product kvs, ksv (inner product)
    // kvv = kvv + (-1.0) . kvs . ksv;
    kvv.multiply_nn((-1.0), kvs, ksv, 1.0);
  }  // inelastic
  //-------------------------------------------------------------------
  // ELSE IF elastic step (F <= 0.0, (1/2 * devstress : eta) <= 0.0)
  //-------------------------------------------------------------------
  else
  {
    // scale diagonal terms of kvv with 1.0/dt
    for (int i = 0; i < Mat::NUM_STRESS_3D; i++) kvv(i, i) = 1.0 / dt;
  }  // elastic

  //-------------------------------------------------------------------
  // ----- derivative of viscous residual with respect to back stress
  // backstress --> alpha
  // kva = (pd res_{n+1}^v) / (pd back stress)
  //-------------------------------------------------------------------
  // IF plastic step ( F > 0.0, (1/2 * devstress : eta) > 0.0 )
  //-------------------------------------------------------------------
  if ((ff > 0.0) and (se > 0.0))
  {
    // assign vector by another vector and scale it
    // kva = kvs . ksa = kva . (-Id)
    // with ksa = (pd eta) / (pd backstress) = - Id; eta = s - backstress
    kva.update(-1.0, kvs);
  }  // inelastic
  //-------------------------------------------------------------------
  // ELSE IF elastic step (F <= 0.0, (1/2 * devstress : eta) <= 0.0)
  //-------------------------------------------------------------------
  else
  {
    for (int i = 0; i < 6; i++) kva(i, i) = 0.0;  // so3_mv6_m_zero(kva);
  }

}  // calc_be_viscous_strain_rate()


/*----------------------------------------------------------------------*
 | residual of BE-discretised back stress and its            dano 11/11 |
 | consistent tangent according to the flow rule at Gauss point         |
 *----------------------------------------------------------------------*/
void Mat::Robinson::calc_be_back_stress_flow(const double dt, const double tempnp,
    const Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>& strain_p,
    const Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>& strain_on,
    const Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>& devstress,
    const Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>& backstress,
    const Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>& backstress_n,
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>& backstress_res,
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D>& kae,
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D>& kav,
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D>& kaa) const
{
  // backstress_n' = f^alpha in a plastic load step
  // at the end of a time step, i.e. equilibrium, the residual has to vanish
  // r^alpha = ( backstress_n - backstress ) / dt - f^alpha != 0
  //
  // discretising the flow equation backstress_n' with a backward Euler scheme
  // --> backstress_n' = ( backstress_n - backstress ) / dt

  // -------------------------------------------------- preliminaries
  // set Cartesian identity 4-tensor in 6-Voigt matrix notation
  // this is fully 'contra-variant' identity tensor, ie I^{ABCD}
  // REMARK: rows are stress-like 6-Voigt
  //         columns are stress-like 6-Voigt
  Core::LinAlg::Matrix<6, 6> id4sharp(Core::LinAlg::Initialization::zero);
  for (unsigned i = 0; i < 3; i++) id4sharp(i, i) = 1.0;
  for (unsigned i = 3; i < 6; i++) id4sharp(i, i) = 0.5;

  // I_2 = 1/2 * Alpha : Alpha  with Alpha...back stress
  double i2 = 0.0;
  i2 = 1.0 / 2.0 *
           (backstress_n(0) * backstress_n(0) + backstress_n(1) * backstress_n(1) +
               backstress_n(2) * backstress_n(2)) +
       +backstress_n(3) * backstress_n(3) + +backstress_n(4) * backstress_n(4) +
       +backstress_n(5) * backstress_n(5);

  // Bingham-Prager shear stress threshold 'K_0^2' at activation temperature
  double kk0sq = 0.0;
  // activation temperature
  double tem0 = params_->actv_tmpr_;
  kk0sq = get_mat_parameter_at_tempnp(&(params_->shrthrshld_), tem0);

  // 'beta' at current temperature
  double beta = get_mat_parameter_at_tempnp(&(params_->beta_), tempnp);

  // 'H' at current temperature
  double hh = 0.0;
  hh = get_mat_parameter_at_tempnp(params_->h_, tempnp);
  if (params_->kind_ == "Arya_NarloyZ")
  {
    hh *= std::pow(6.896, (1.0 + beta)) / (3.0 * kk0sq);
  }
  if (params_->kind_ == "Arya_CrMoSteel")
  {
    double mu = params_->hrdn_fact_;
    hh *= 2.0 * mu;
  }
  else  // (*(params_->kind_) == "Butler", "Arya", "Arya_NarloyZ")
  {
    // go on, no further changes for H required
  }

  // recovery/softening factor 'R_0'
  double rr0 = 0.0;
  rr0 = get_mat_parameter_at_tempnp(params_->rcvry_, tempnp);
  // exponent 'm'
  double mm = params_->m_;
  if (params_->kind_ == "Arya_NarloyZ")
  {
    // pressure unit scale : cN/cm^2 = 10^-4 MPa
    const double pus = 1.0e-4;
    rr0 *= std::pow(6.896, (1.0 + beta + mm)) * std::pow(3.0 * kk0sq * pus * pus, (mm - beta));
  }
  else  // (*(params_->kind_) == "Butler", "Arya", "Arya_NarloyZ")
  {
    // go on, no further changes for R_0 required
  }

  // recovery/softening term 'R'
  // R = R_0 . exp[ Q_0 ( (T - Theta_0 )(T . Theta_0) )]
  // tem0: activation temperature
  double rr = 0.0;
  // activation energy 'Q_0'
  double q0 = params_->actv_ergy_;
  // T_{n+1} . T_0 < (1.0E-12)
  if (fabs(tempnp * tem0) <= 1e-12)
  {
    // T_0 < (1.0E-12)
    if (fabs(tem0) <= 1e-12)
    {
      rr = rr0;
    }
    // T_0 > (1.0E-12)
    else
    {
      rr = rr0 * exp(q0 / tem0);
    }
  }
  // T_{n+1} . T_0 > (1.0E-12)
  else
  {
    rr = rr0 * exp(q0 * (tempnp - tem0) / (tempnp * tem0));
    if (std::isinf(rr))
    {
      rr = rr0;
    }
  }

  // 'G_0'
  double gg0 = params_->g0_;

  // initialise 'G'
  // G = I_2/K_0^2
  double gg = 0.0;
  // K_0^2 < 1.0E-10
  if (fabs(kk0sq) <= 1e-10)
  {
    gg = 0.0;
    FOUR_C_THROW("Division by zero: Shear threshold very close to zero");
  }
  // K_0^2 > 1.0E-10
  else  // (fabs(kk0sq) > 1e-10)
  {
    gg = sqrt(i2 / kk0sq);
  }

  // sa = 1/2 * devstress : backstresscurr
  double sa = 0.0;
  sa = 1.0 / 2.0 *
           (backstress_n(0) * devstress(0) + backstress_n(1) * devstress(1) +
               backstress_n(2) * devstress(2)) +
       +backstress_n(3) * devstress(3) + backstress_n(4) * devstress(4) +
       +backstress_n(5) * devstress(5);

  // ----------------- difference of current and last viscous strains
  // (Delta strain_p)_{n+1} = strain_on - strain_p
  //  \incr \eps^v = \eps_{n+1}^v - \eps_{n}^v
  //  with halved entries to conform with stress vectors */
  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> strain_pd05(
      Core::LinAlg::Initialization::uninitialized);
  strain_pd05.update(1.0, strain_on, (-1.0), strain_p);
  // Due to the fact that strain vector component have a doubled shear
  // components, i.e.
  //   strain = [ a11 a22 a33 | 2*a12 2*a23 2*a31 ]
  // but stress vectors have not, i.e.
  //   stress = [ a11 a22 a33 | a12 a23 a31 ]
  // we need to scale the last three entries.
  for (int i = 3; i < Mat::NUM_STRESS_3D; i++)
  {
    strain_pd05(i) *= 0.5;
  }


  // ------------------------------------------ residual of back stress
  //-------------------------------------------------------------------
  // IF plastic step (G > G_0, 1/2 (devstress . backstress) > 0.0)
  //-------------------------------------------------------------------
  if ((gg > gg0) and (sa > 0.0))
  {
    //     std::cout << "plastic step: back stress rate alpha'!= 0" << std::endl;

    double fctv = hh / std::pow(gg, beta);
    double fcta = rr * std::pow(gg, (mm - beta)) / sqrt(i2);
    for (int i = 0; i < Mat::NUM_STRESS_3D; i++)
    {
      // alpha_res = (alpha_{n+1} - alpha_{n})/dt - f^alpha)

      // here is the essential!!!
      backstress_res(i) =
          (backstress_n(i) - backstress(i) - fctv * strain_pd05(i) + dt * fcta * backstress_n(i)) /
          dt;
    }
  }  // inelastic
  //-------------------------------------------------------------------
  // ELSE IF elastic step (G <= G_0, 1/2 (devstress . backstress) <= 0.0)
  //-------------------------------------------------------------------
  else
  {
    double fctv = hh / std::pow(gg0, beta);
    double fcta = 0.0;
    if (sqrt(i2) < 1e-10)
    {
      // sqrt(i2) := 1.0e6 assures units are OK
      fcta = rr * std::pow(gg0, (mm - beta)) / 1.0e6;
    }
    else  // (sqrt(i2) > 1e-10)
    {
      fcta = rr * std::pow(gg0, (mm - beta)) / sqrt(i2);
    }

    for (int i = 0; i < Mat::NUM_STRESS_3D; i++)
    {
      // here is the essential!!!
      backstress_res(i) =
          (backstress_n(i) - backstress(i) - fctv * strain_pd05(i) + dt * fcta * backstress_n(i)) /
          dt;
    }
  }  // elastic


  // ----------------------------- derivative of back stress residual

  // ----------------------- derivative with respect to total strains
  // kae = pd(res^al)/pd(eps) == 0
  kae.put_scalar(0.0);

  // --------------------- derivative with respect to viscous strains
  // kav = pd(res_{n+1}^al)/pd(eps_{n+1}^v)
  //-------------------------------------------------------------------
  // IF plastic step (G > G_0, 1/2 (devstress . backstress) > 0.0)
  //-------------------------------------------------------------------
  if ((gg > gg0) and (sa > 0.0))
  {
    double fctv = -hh / (pow(gg, beta) * dt);
    kav.update(fctv, id4sharp);
  }  // inelastic
  //-------------------------------------------------------------------
  // ELSE IF elastic step (G <= G_0, 1/2 (devstress . backstress) <= 0.0)
  //-------------------------------------------------------------------
  else
  {
    double fctv = -hh / (pow(gg0, beta) * dt);
    kav.update(fctv, id4sharp);
  }  // elastic

  // ------------------------- derivative with respect to back stress
  // derivative of back stress residual with respect to back stress
  // kaa = pd(res_{n+1}^al)/pd(al_{n+1})
  // set Cartesian identity 4-tensor in 6x6-matrix notation (stress-like)
  // this is a 'mixed co- and contra-variant' identity 4-tensor, ie I^{AB}_{CD}
  // REMARK: rows are stress-like 6-Voigt
  //         columns are strain-like 6-Voigt
  Core::LinAlg::Matrix<6, 6> id4(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 6; i++) id4(i, i) = 1.0;
  //-------------------------------------------------------------------
  // IF plastic step (G > G_0, 1/2 (devstress . backstress) > 0.0)
  //-------------------------------------------------------------------
  if ((gg > gg0) and (sa > 0.0))
  {
    double fctu = 1.0 / dt + rr * std::pow(gg, (mm - beta)) / sqrt(i2);
    double fctv = beta * hh / (std::pow(gg, (beta + 1.0)) * dt * kk0sq);
    double fcta = rr * (mm - beta) * std::pow(gg, (mm - beta - 1.0)) / (sqrt(i2) * kk0sq) -
                  rr * std::pow(gg, (mm - beta)) / (2.0 * std::pow(i2, 1.5));
    kaa.update(fctu, id4);
    kaa.multiply_nt(fctv, strain_pd05, backstress_n, 1.0);
    kaa.multiply_nt(fcta, backstress_n, backstress_n, 1.0);
  }  // inelastic
  //-------------------------------------------------------------------
  // ELSE IF elastic step (G <= G_0, 1/2 (devstress . backstress) <= 0.0)
  //-------------------------------------------------------------------
  else
  {
    double ii2;
    if (sqrt(i2) < 1e-10)
    {
      ii2 = 1.0e12; /* sqrt(i2) := 1.0e6 assures units are OK */
    }
    else
    {
      ii2 = i2;
    }
    double fctu = 1.0 / dt + rr * std::pow(gg0, (mm - beta)) / sqrt(ii2);
    double fcta = -rr * std::pow(gg0, (mm - beta)) / (2.0 * std::pow(ii2, 1.5));
    kaa.update(fctu, id4);
    kaa.multiply_nt(fcta, backstress_n, backstress_n, 1.0);
  }  // elastic

}  // calc_be_back_stress_flow()


/*----------------------------------------------------------------------*
 | return temperature-dependent material parameter           dano 02/12 |
 | at current temperature --> polynomial type                           |
 *----------------------------------------------------------------------*/
double Mat::Robinson::get_mat_parameter_at_tempnp(
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
  for (unsigned int i = 0; i < (*paramvector).size(); ++i)
  {
    // calculate coefficient of variable T^i
    parambytempnp += (*paramvector)[i] * tempnp_pow;
    // for the higher polynom increase the exponent of the temperature
    tempnp_pow *= tempnp;
  }

  return parambytempnp;

}  // get_mat_parameter_at_tempnp()


/*----------------------------------------------------------------------*
 | Get temperature-dependent material parameter at           dano 11/11 |
 | current temperature                                                  |
 *----------------------------------------------------------------------*/
double Mat::Robinson::get_mat_parameter_at_tempnp(
    const double paramconst,  // (i) given parameter is a constant
    const double& tempnp      // tmpr (i) current temperature
) const
{
  // initialise the temperature dependent material parameter
  double parambytempnp = 0.0;

  // constant
  if (paramconst != 0.0)
  {
    // now calculate the parameter
    parambytempnp = paramconst;
  }

  return parambytempnp;

}  // get_mat_parameter_at_tempnp()


/*----------------------------------------------------------------------*
 | Reduce (statically condense) system in strain, strain_p,  dano 01/12 |
 | backstress to purely strain                                          |
 *----------------------------------------------------------------------*/
void Mat::Robinson::calculate_condensed_system(
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>&
        stress,  // (io): updated stress with condensed terms
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D>&
        cmat,  // (io): updated tangent with condensed terms
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D>& kev,
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D>& kea,
    const Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>& strain_pres,
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D>& kve,
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D>& kvv,
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D>& kva,
    const Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>& backstress_res,
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D>& kae,
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D>& kav,
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D>& kaa,
    Core::LinAlg::Matrix<(2 * Mat::NUM_STRESS_3D), 1>& kvarva,  // (io) solved/condensed residual
    Core::LinAlg::Matrix<(2 * Mat::NUM_STRESS_3D), Mat::NUM_STRESS_3D>&
        kvakvae  // (io) solved/condensed tangent
) const
{
  // update vector for material internal variables (MIV) iterative increments
  // (stored as Fortranesque vector 2*NUMSTR_SOLID3x1)
  //             [ kvv  kva ]^{-1}   [ res^v  ]
  //    kvarva = [          ]      . [        ]
  //             [ kav  kaa ]      . [ res^al ]

  // update matrix for material internal variables (MIV)  iterative increments
  // (stored as Fortran-type (Fortranesque)vector (2*NUMSTR_SOLID3*NUMSTR_SOLID3)x1)
  //              [ kvv  kva ]^{-1}   [ kve ]
  //    kvakvae = [          ]      . [     ]
  //              [ kav  kaa ]        [ kae ]

  // build the matrix kvvkvakavkaa, consisting of the four submatrices, each
  // with size (6x6) --> kvvkvakavkaa: (12x12)
  //                [ kvv  kva ]
  // kvvkvakavkaa = [          ] and its inverse after factorisation
  //                [ kav  kaa ]
  Core::LinAlg::Matrix<2 * Mat::NUM_STRESS_3D, 2 * Mat::NUM_STRESS_3D> kvvkvakavkaa(
      Core::LinAlg::Initialization::zero);

  // build the matrix kevea (6x12)
  // kevea = [ kev  kea ] stored in column-major order
  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 2 * Mat::NUM_STRESS_3D> kevea(
      Core::LinAlg::Initialization::zero);

  // ------------------ build tangent and right hand side to reduce
  {
    // first Mat::NUM_STRESS_3D rows (i=1--6)
    for (int i = 0; i < Mat::NUM_STRESS_3D; i++)
    {
      // residual vector (i=1--6,j=1)
      kvarva(i) = strain_pres(i);

      // first Mat::NUM_STRESS_3D columns
      for (int j = 0; j < Mat::NUM_STRESS_3D; j++)
      {
        // tangent (i=1--6,j=1--6)
        kvvkvakavkaa(i, j) = kvv(i, j);

        // RHS (i=1--6,j=1--6)
        kvakvae(i, j) = kve(i, j);
        // intermediate matrix - position in column-major vector matrix (i=1--6,j=1--6)
        kevea(i, j) = kev(i, j);
      }  // column: j=1--6

      // second Mat::NUM_STRESS_3D columns
      for (int j = 0; j < Mat::NUM_STRESS_3D; j++)
      {
        // tangent (i=1--6,j=7--12)
        kvvkvakavkaa(i, (j + Mat::NUM_STRESS_3D)) = kva(i, j);
        // position in column-major vector matrix (i=1--6,j=6--12)
        kevea(i, (j + Mat::NUM_STRESS_3D)) = kva(i, j);
      }  // column: j=7--12
    }  //  rows: i=1--6

    // second Mat::NUM_STRESS_3D rows, i.e. (i=6--12)
    for (int i = 0; i < Mat::NUM_STRESS_3D; i++)
    {
      // residual vector (i=7--12,j=1)
      // add residual of backstresses
      kvarva(Mat::NUM_STRESS_3D + i, 0) = backstress_res(i);

      // first Mat::NUM_STRESS_3D columns
      for (int j = 0; j < Mat::NUM_STRESS_3D; j++)
      {
        // tangent (i=6--12,j=1--6)
        kvvkvakavkaa(Mat::NUM_STRESS_3D + i, j) = kav(i, j);
        // RHS (i=6--12,j=1--6)
        kvakvae(Mat::NUM_STRESS_3D + i, j) = kae(i, j);
      }  // column: j=1--6
      // second Mat::NUM_STRESS_3D columns
      for (int j = 0; j < Mat::NUM_STRESS_3D; j++)
      {
        // tangent (i=6--12,j=6--12)
        kvvkvakavkaa(Mat::NUM_STRESS_3D + i, (j + Mat::NUM_STRESS_3D)) = kaa(i, j);
      }  // column: j=7--12
    }  //  rows: i=6--12

  }  // end build tangent and rhs

  // ------------------------------- factorise kvvkvakavkaa and solve
  // adapt solving at utils_gder2.H implemented by Georg Bauer
  // solve x = A^{-1} . b

  // --------------------------------- back substitution of residuals
  // pass the size of the matrix and the number of columns to the solver
  // solve x = A^{-1} . b
  // with: A=kvvkvakavkaa(i), x=kvarva(o), b=kvarva (i)
  //           [ kvv  kva ]^{-1} [ res^v  ]^i
  // kvarva =  [          ]      [        ]
  //           [ kav  kaa ]      [ res^al ]
  Core::LinAlg::FixedSizeSerialDenseSolver<(2 * Mat::NUM_STRESS_3D), (2 * Mat::NUM_STRESS_3D), 1>
      solver_res;
  solver_res.set_matrix(kvvkvakavkaa);
  // No need for a separate rhs. We assemble the rhs to the solution vector.
  // The solver will destroy the rhs and return the solution.
  // x: vector of unknowns, b: right hand side vector
  // kvarva = kvvkvakavkaa^{-1} . kvarva
  solver_res.set_vectors(kvarva, kvarva);
  solver_res.solve();

  // ----------------------------------- back substitution of tangent
  // solve x = A^{-1} . b
  // with: A=kvvkvakavkaa(i), x=kvakvae(o), b=kvakvae (i)
  //            [ kvv  kva ]^{-1} [ kve ]^i
  // kvakvae =  [          ]      [     ]
  //            [ kav  kaa ]      [ kae ]
  Core::LinAlg::FixedSizeSerialDenseSolver<(2 * Mat::NUM_STRESS_3D), (2 * Mat::NUM_STRESS_3D),
      Mat::NUM_STRESS_3D>
      solver_tang;
  solver_tang.set_matrix(kvvkvakavkaa);
  // No need for a separate rhs. We assemble the rhs to the solution vector.
  // The solver will destroy the rhs and return the solution.
  // x: vector of unknowns, b: right hand side vector: here: x=b=kvakvae
  // kvakvae = kvvkvakavkaa^{-1} . kvakvae
  solver_tang.set_vectors(kvakvae, kvakvae);
  solver_tang.solve();

  // final condensed system expressed only in stress, strain, cmat
  // sig_red^i = kee_red^i . iinc eps --> stress_red = cmat_red . Delta strain

  // --------------------------------- reduce stress vector sigma_red
  // stress (6x6) = kevea (6x12) . kvarva (12x1)
  stress.multiply((-1.0), kevea, kvarva, 1.0);

  // ---------------------------------------- reduce tangent k_ee_red
  // cmat (6x6) = kevea (6x12) . kvakvae (12x6)
  cmat.multiply_nn((-1.0), kevea, kvakvae, 1.0);

}  // calculate_condensed_system()


/*----------------------------------------------------------------------*
 | iterative update of material internal variables that      dano 01/12 |
 | are condensed out of the system within calculate_condensed_system()    |
 *----------------------------------------------------------------------*/
void Mat::Robinson::iterative_update_of_internal_variables(const int gp,
    const Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> straininc  // total strain increment
)
{
  // get the condensed / reduced residual vector
  //           [ kvv  kva ]^{-1} [ res^v  ]^i
  // kvarva =  [          ]      [        ]
  //           [ kav  kaa ]      [ res^al ]
  // condensed vector of residual
  Core::LinAlg::Matrix<(2 * Mat::NUM_STRESS_3D), 1> kvarva(
      Core::LinAlg::Initialization::uninitialized);
  kvarva = kvarva_->at(gp);

  // get the condensed / reduced scaled tangent
  //            [ kvv  kva ]^{-1} [ kve ]^i
  // kvakvae =  [          ]      [     ]
  //            [ kav  kaa ]      [ kae ]
  Core::LinAlg::Matrix<(2 * Mat::NUM_STRESS_3D), Mat::NUM_STRESS_3D> kvakvae(
      Core::LinAlg::Initialization::uninitialized);
  kvakvae = kvakvae_->at(gp);

  // initialise updated vector with newest values strainp_np = strainp_n
  // strainp_np = strainp_n + Delta strain_np
  // viscous strain \f$\varepsilon_{n+1}\f$ at t_{n+1}^i
  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> strain_on(
      Core::LinAlg::Initialization::uninitialized);
  strain_on.update(strainplcurr_->at(gp));
  // back stress \f$alpha_{n+1}\f$ at t_{n+1}^i
  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> backstress_n(
      Core::LinAlg::Initialization::uninitialized);
  backstress_n.update(backstresscurr_->at(gp));

  // ---------------------------------  update current viscous strain
  // [ iinc eps^v ] = [ kvv  kva ]^{-1} (   [ res^v  ] - [ kve ] [ iinc eps ] )
  // Delta strain_on(i) = kvarva(i) - kvakvae(i) . Delta strain
  // with kvarva (12x1), kvakvae (12x6), Delta strain (6x1)
  {
    for (int i = 0; i < Mat::NUM_STRESS_3D; i++)
    {
      double rcsum = 0.0;
      double rcsum1 = 0.0;

      // viscous residual contribution
      // Delta strain_on(i) = [ kvv  kva ]^{-1} [ res^v  ] for i=1--6
      rcsum = kvarva(i);

      // tangent contribution
      // Delta strain_on(i)+ = [ kvv  kva ]^{-1} ( - [ kve ] [ iinc eps ] ) for i=1--6
      for (int j = 0; j < Mat::NUM_STRESS_3D; j++)
      {
        // Delta strain_on(i) += kvakvae(i,j) * straininc(j)
        // tangent kvakvae (12x6) (i=1--6,j=1--6)
        // increment of strain (j=1--6)
        rcsum1 = kvakvae(i, j) * straininc(j);
      }
      // update all terms on strain_on
      strain_on(i) += -rcsum - rcsum1;
    }
  }  // end update viscous strains

  // ------------------------------------  update current back stress
  {
    for (int i = 0; i < Mat::NUM_STRESS_3D; i++)
    {
      double rcsum = 0.0;
      double rcsum1 = 0.0;

      // back stress residual contribution
      // Delta backstress(i) = [ kav  kaa ]^{-1} [ res^v  ] for i=7--12
      rcsum = kvarva(Mat::NUM_STRESS_3D + i);

      // tangent contribution
      // Delta backstress(i)+ = [ kav  kaa ]^{-1} ( - [ kae ] [ iinc eps ] )
      //   for i=7--12, j=1--6
      for (int j = 0; j < Mat::NUM_STRESS_3D; j++)
      {
        // Delta strain_on(i) += kvakvae(i,j) * straininc(j)
        // tangent kvakvae (12x6) (i=7--12,j=1--6)
        // increment of strain (j=1--6)
        rcsum1 = kvakvae(Mat::NUM_STRESS_3D + i, j) * straininc(j);
      }

      backstress_n(i) += -rcsum - rcsum1;
    }
  }  // end update back stress

  // update the history vectors
  // strain_p^{n+1} = strain_p^{n} + Delta strain_p^{n+1}
  // with Delta strain_on(i) = [ kvv  kva ]^{-1} [ res^v  ]
  //                         + = [ kvv  kva ]^{-1} ( - [ kve ] [ iinc eps ] )
  //   for i=1--6, j=1--6
  strainplcurr_->at(gp) = strain_on;

  // backstress^{n+1} = backstress^{n} + Delta backstress^{n+1}
  // with Delta backstress^{n+1} = [ kav  kaa ]^{-1} ( [ res^v  ]
  //                               + [ kav  kaa ]^{-1} ( - [ kae ] [ iinc eps ] )
  //   for i=7--12, j=1--6
  backstresscurr_->at(gp) = backstress_n;

}  // iterative_update_of_internal_variables()

/*----------------------------------------------------------------------*/

void Mat::Robinson::evaluate(const Core::LinAlg::Matrix<3, 1>& gradtemp,
    Core::LinAlg::Matrix<3, 3>& cmat, Core::LinAlg::Matrix<3, 1>& heatflux, const int eleGID) const
{
  thermo_->evaluate(gradtemp, cmat, heatflux, eleGID);
}

void Mat::Robinson::evaluate(const Core::LinAlg::Matrix<2, 1>& gradtemp,
    Core::LinAlg::Matrix<2, 2>& cmat, Core::LinAlg::Matrix<2, 1>& heatflux, const int eleGID) const
{
  thermo_->evaluate(gradtemp, cmat, heatflux, eleGID);
}

void Mat::Robinson::evaluate(const Core::LinAlg::Matrix<1, 1>& gradtemp,
    Core::LinAlg::Matrix<1, 1>& cmat, Core::LinAlg::Matrix<1, 1>& heatflux, const int eleGID) const
{
  thermo_->evaluate(gradtemp, cmat, heatflux, eleGID);
}

std::vector<double> Mat::Robinson::conductivity(int eleGID) const
{
  return thermo_->conductivity(eleGID);
}

void Mat::Robinson::conductivity_deriv_t(Core::LinAlg::Matrix<3, 3>& dCondDT) const
{
  thermo_->conductivity_deriv_t(dCondDT);
}

void Mat::Robinson::conductivity_deriv_t(Core::LinAlg::Matrix<2, 2>& dCondDT) const
{
  thermo_->conductivity_deriv_t(dCondDT);
}

void Mat::Robinson::conductivity_deriv_t(Core::LinAlg::Matrix<1, 1>& dCondDT) const
{
  thermo_->conductivity_deriv_t(dCondDT);
}

double Mat::Robinson::capacity() const { return thermo_->capacity(); }

double Mat::Robinson::capacity_deriv_t() const { return thermo_->capacity_deriv_t(); }

void Mat::Robinson::reinit(double temperature, unsigned gp)
{
  current_temperature_ = temperature;
  if (thermo_ != nullptr) thermo_->reinit(temperature, gp);
}
void Mat::Robinson::reset_current_state()
{
  if (thermo_ != nullptr) thermo_->reset_current_state();
}

void Mat::Robinson::commit_current_state()
{
  if (thermo_ != nullptr) thermo_->commit_current_state();
}

/*----------------------------------------------------------------------*/

FOUR_C_NAMESPACE_CLOSE
