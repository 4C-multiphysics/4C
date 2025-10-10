// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_visconeohooke.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
Mat::PAR::ViscoNeoHooke::ViscoNeoHooke(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      youngs_slow_(matdata.parameters.get<double>("YOUNGS_SLOW")),
      poisson_(matdata.parameters.get<double>("POISSON")),
      density_(matdata.parameters.get<double>("DENS")),
      youngs_fast_(matdata.parameters.get<double>("YOUNGS_FAST")),
      relax_(matdata.parameters.get<double>("RELAX")),
      theta_(matdata.parameters.get<double>("THETA"))
{
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::ViscoNeoHooke::create_material()
{
  return std::make_shared<Mat::ViscoNeoHooke>(this);
}


Mat::ViscoNeoHookeType Mat::ViscoNeoHookeType::instance_;


Core::Communication::ParObject* Mat::ViscoNeoHookeType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::ViscoNeoHooke* visco = new Mat::ViscoNeoHooke();
  visco->unpack(buffer);
  return visco;
}


/*----------------------------------------------------------------------*
 |  Constructor                                   (public)         05/08|
 *----------------------------------------------------------------------*/
Mat::ViscoNeoHooke::ViscoNeoHooke() : params_(nullptr)
{
  isinit_ = false;
  histstresscurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  artstresscurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  histstresslast_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  artstresslast_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
}


/*----------------------------------------------------------------------*
 |  Copy-Constructor                             (public)          05/08|
 *----------------------------------------------------------------------*/
Mat::ViscoNeoHooke::ViscoNeoHooke(Mat::PAR::ViscoNeoHooke* params) : params_(params) {}


/*----------------------------------------------------------------------*
 |  Pack                                          (public)         05/08|
 *----------------------------------------------------------------------*/
void Mat::ViscoNeoHooke::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);

  //  pack history data
  int histsize = 0;
  if (initialized())
  {
    histsize = histstresslast_->size();
  }
  add_to_pack(data, histsize);  // Length of history vector(s)
  for (int var = 0; var < histsize; ++var)
  {
    // insert last converged states
    add_to_pack(data, histstresslast_->at(var));
    add_to_pack(data, artstresslast_->at(var));

    // insert current iteration states
    add_to_pack(data, histstresscurr_->at(var));
    add_to_pack(data, artstresscurr_->at(var));
  }
}


/*----------------------------------------------------------------------*
 |  Unpack                                        (public)         05/08|
 *----------------------------------------------------------------------*/
void Mat::ViscoNeoHooke::unpack(Core::Communication::UnpackBuffer& buffer)
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
        params_ = static_cast<Mat::PAR::ViscoNeoHooke*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }

  // history data
  int histsize;
  extract_from_pack(buffer, histsize);

  if (histsize == 0) isinit_ = false;

  histstresslast_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  artstresslast_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  histstresscurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  artstresscurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  for (int var = 0; var < histsize; ++var)
  {
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> tmp_vector(Core::LinAlg::Initialization::zero);

    // last converged states are unpacked
    extract_from_pack(buffer, tmp_vector);
    histstresslast_->push_back(tmp_vector);
    extract_from_pack(buffer, tmp_vector);
    artstresslast_->push_back(tmp_vector);

    // current iteration states are unpacked
    extract_from_pack(buffer, tmp_vector);
    histstresscurr_->push_back(tmp_vector);
    extract_from_pack(buffer, tmp_vector);
    artstresscurr_->push_back(tmp_vector);
  }
}

/*----------------------------------------------------------------------*
 |  Initialise/allocate internal stress variables (public)         05/08|
 *----------------------------------------------------------------------*/
void Mat::ViscoNeoHooke::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  histstresscurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  artstresscurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  histstresslast_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  artstresslast_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  const Core::LinAlg::Matrix<NUM_STRESS_3D, 1> emptyvec(Core::LinAlg::Initialization::zero);
  histstresscurr_->resize(numgp);
  histstresslast_->resize(numgp);
  artstresscurr_->resize(numgp);
  artstresslast_->resize(numgp);
  for (int j = 0; j < numgp; ++j)
  {
    histstresscurr_->at(j) = emptyvec;
    histstresslast_->at(j) = emptyvec;
    artstresscurr_->at(j) = emptyvec;
    artstresslast_->at(j) = emptyvec;
  }

  const double E_s = params_->youngs_slow_;
  double E_f = params_->youngs_fast_;
  double tau = params_->relax_;

  if (E_f < E_s) FOUR_C_THROW("Wrong ratio between fast and slow Young's modulus");
  if (tau <= 0.0) FOUR_C_THROW("Relaxation time tau has to be positive!");
  isinit_ = true;
  return;
}

/*----------------------------------------------------------------------*
 |  Update internal stress variables              (public)         05/08|
 *----------------------------------------------------------------------*/
void Mat::ViscoNeoHooke::update()
{
  histstresslast_ = histstresscurr_;
  artstresslast_ = artstresscurr_;
  const Core::LinAlg::Matrix<NUM_STRESS_3D, 1> emptyvec(Core::LinAlg::Initialization::zero);
  histstresscurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  artstresscurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  const int numgp = histstresslast_->size();
  histstresscurr_->resize(numgp);
  artstresscurr_->resize(numgp);
  for (int j = 0; j < numgp; ++j)
  {
    histstresscurr_->at(j) = emptyvec;
    artstresscurr_->at(j) = emptyvec;
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Evaluate Material                             (public)         05/08|
 *----------------------------------------------------------------------*/
void Mat::ViscoNeoHooke::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  const Core::LinAlg::Matrix<6, 1> glstrain_mat =
      Core::LinAlg::make_strain_like_voigt_matrix(glstrain);

  Core::LinAlg::Matrix<6, 1> stress_view = Core::LinAlg::make_stress_like_voigt_view(stress);
  Core::LinAlg::Matrix<6, 6> cmat_view = Core::LinAlg::make_stress_like_voigt_view(cmat);

  // get material parameters
  const double E_s = params_->youngs_slow_;
  const double nue = params_->poisson_;
  double E_f = params_->youngs_fast_;
  double tau = params_->relax_;
  const double theta = params_->theta_;

  // get time algorithmic parameters
  // NOTE: dt can be zero (in restart of STI) for Generalized Maxwell model
  // there is no special treatment required. Adaptation for Kelvin-Voigt were necessary.
  FOUR_C_ASSERT(context.time_step_size, "Time step size not given in evaluation context.");
  double dt = *context.time_step_size;

  // compute algorithmic relaxation time
  double tau1 = tau;
  // check for meaningful values
  if (E_f > E_s)
  {
    tau1 = tau * E_s / (E_f - E_s);
  }

  // initialize scalars
  double alpha0;
  double alpha1;
  double lambda;
  double mue;
  double kappa;
  double artscalar1;
  double artscalar2;
  double scalarvisco;

#define GEN_MAXWELL
#ifdef GEN_MAXWELL
  tau = tau1;
  // evaluate "alpha" factors which distribute stress or stiffness between parallel springs
  // sum_0^i alpha_j = 1
  alpha0 = E_s / E_f;
  alpha1 = 1.0 - alpha0;

  // evaluate Lame constants, bulk modulus
  lambda = nue * E_f / ((1.0 + nue) * (1.0 - 2.0 * nue));
  mue = E_f / (2.0 * (1.0 + nue));
  kappa = lambda + 2.0 / 3.0 * mue;

  // evaluate scalars to compute
  // Q^(n+1) = tau/(tau+theta*dt) [(tau-dt+theta*dt)/tau Q + S^(n+1) - S^n]
  artscalar1 = (tau - dt + theta * dt) / tau;
  artscalar2 = tau / (tau + theta * dt);

  // factor to calculate visco stiffness matrix from elastic stiffness matrix
  scalarvisco = alpha0 + alpha1 * tau / (tau + theta * dt);

#else
  // in this case we have a parallel layout of a spring and a dashpot,
  // so no stress distribution between parallel springs
  alpha0 = 1.;
  alpha1 = 1.;

  // do we have to propagate in time?
  if (dt > 0.0)
  {
    // evaluate scalars to compute
    // Q^(n+1) = tau/(theta*dt) [(-dt+theta*dt)/tau Q + S^(n+1) - S^n]
    artscalar1 = (-dt + theta * dt) / tau;
    artscalar2 = tau / (theta * dt);

    // factor to calculate visco stiffness matrix from elastic stiffness matrix
    scalarvisco = 1.0 + tau / (theta * dt);
  }
  else
  {
    // in case we do not want to propagate in time, Q^{n+1} = Q^{n}
    artscalar1 = 1.0;
    artscalar2 = 1.0;
    // factor to calculate visco stiffness matrix from elastic stiffness matrix
    scalarvisco = 2.0;
  }

#endif

  // right Cauchy-Green Tensor  C = 2 * E + I
  // build identity tensor I
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Id;
  for (int i = 0; i < 3; i++) Id(i) = 1.0;
  for (int i = 3; i < 6; i++) Id(i) = 0.0;
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> C(glstrain_mat);
  C.scale(2.0);
  C += Id;

  // invariants
  const double I1 = C(0) + C(1) + C(2);  // 1st invariant, trace
  const double I3 = C(0) * C(1) * C(2) + 0.25 * C(3) * C(4) * C(5) - 0.25 * C(1) * C(5) * C(5) -
                    0.25 * C(2) * C(3) * C(3) -
                    0.25 * C(0) * C(4) * C(4);  // 3rd invariant, determinant
  const double J = sqrt(I3);
  const double I3invcubroot = std::pow(I3, -1.0 / 3.0);

  // invert C
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Cinv;
  Cinv(0) = C(1) * C(2) - 0.25 * C(4) * C(4);
  Cinv(1) = C(0) * C(2) - 0.25 * C(5) * C(5);
  Cinv(2) = C(0) * C(1) - 0.25 * C(3) * C(3);
  Cinv(3) = 0.25 * C(5) * C(4) - 0.5 * C(3) * C(2);
  Cinv(4) = 0.25 * C(3) * C(5) - 0.5 * C(0) * C(4);
  Cinv(5) = 0.25 * C(3) * C(4) - 0.5 * C(5) * C(1);
  Cinv.scale(1.0 / I3);

  // elastic part: NeoHooke  ************************************************
  // NeoHooke with penalty W = W^dev(C) + U(J)
  // W = 1/2 mue (^I1-3) + 1/2 kappa (J-1)^2

  // Split into volumetric and deviatoric parts. Viscosity affects only deviatoric part
  // Volumetric part of PK2 stress
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> SVol(Cinv);
  SVol.scale(kappa * (J - 1.0) * J);
  stress_view += SVol;

  // Deviatoric elastic part (2 d W^dev/d C)
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> SDevEla(Cinv);
  SDevEla.scale(-1.0 / 3.0 * I1);
  SDevEla += Id;
  SDevEla.scale(mue * I3invcubroot);  // mue*I3^(-1/3) (Id-1/3*I1*Cinv)

  // visco part
  // read history
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> S_n(histstresslast_->at(gp));
  S_n.scale(-1.0);
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Q_n(artstresslast_->at(gp));

  // artificial visco stresses
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Q(Q_n);
  Q.scale(artscalar1);
  Q += SDevEla;
  Q += S_n;
  Q.scale(artscalar2);  // Q^(n+1) = artscalar2* [artscalar1* Q + S^(n+1) - S^n]

  // update history
  histstresscurr_->at(gp) = SDevEla;
  artstresscurr_->at(gp) = Q;

  // add visco PK2 stress, weighted with alphas
  SDevEla.scale(alpha0);
  stress_view += SDevEla;
  Q.scale(alpha1);
  stress_view += Q;
  // elasticity matrix
  double scalar1 = 2.0 * kappa * J * J - kappa * J;
  double scalar2 = -2.0 * kappa * J * J + 2.0 * kappa * J;
  double scalar3 = 2.0 / 3.0 * mue * I3invcubroot * I1;
  double scalar4 = 2.0 / 3.0 * mue * I3invcubroot;

  Core::LinAlg::SymmetricTensor<double, 3, 3> Cinv_t =
      Core::LinAlg::make_symmetric_tensor_from_stress_like_voigt_matrix(Cinv);
  // add volumetric elastic part 1
  // add scalar2 Cinv o Cinv (see Holzapfel p. 254)
  cmat += scalar2 * Core::LinAlg::FourTensorOperations::holzapfel_product(Cinv_t);

  // add visco-elastic deviatoric part 1
  cmat += scalarvisco * scalar3 * Core::LinAlg::FourTensorOperations::holzapfel_product(Cinv_t);

  for (int i = 0; i < 6; ++i)
  {
    for (int j = 0; j < 6; ++j)
    {
      // add volumetric elastic part 2
      cmat_view(i, j) += scalar1 * Cinv(i) * Cinv(j)  // add scalar Cinv x Cinv
                                                      // add visco-elastic deviatoric part 2
                         + scalarvisco * (-scalar4) * Id(i) * Cinv(j)  // add scalar Id x Cinv
                         + scalarvisco * (-scalar4) * Id(j) * Cinv(i)  // add scalar Cinv x Id
                         +
                         scalarvisco * (scalar3)*Cinv(i) * Cinv(j) / 3.0;  // add scalar Cinv x Cinv
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
