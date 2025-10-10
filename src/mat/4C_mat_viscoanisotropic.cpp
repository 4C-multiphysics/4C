// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_viscoanisotropic.hpp"

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
Mat::PAR::ViscoAnisotropic::ViscoAnisotropic(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      kappa_(matdata.parameters.get<double>("KAPPA")),
      mue_(matdata.parameters.get<double>("MUE")),
      density_(matdata.parameters.get<double>("DENS")),
      k1_(matdata.parameters.get<double>("K1")),
      k2_(matdata.parameters.get<double>("K2")),
      gamma_(matdata.parameters.get<double>("GAMMA")),
      numstresstypes_(3),
      beta_(),
      relax_(),
      minstretch_(matdata.parameters.get<double>("MINSTRETCH")),
      elethick_(matdata.parameters.get<int>("ELETHICKDIR"))
{
  beta_[0] = matdata.parameters.get<double>("BETA_ISO");
  beta_[1] = matdata.parameters.get<double>("BETA_ANISO");
  relax_[0] = matdata.parameters.get<double>("RELAX_ISO");
  relax_[1] = matdata.parameters.get<double>("RELAX_ANISO");
}


std::shared_ptr<Core::Mat::Material> Mat::PAR::ViscoAnisotropic::create_material()
{
  return std::make_shared<Mat::ViscoAnisotropic>(this);
}

Mat::ViscoAnisotropicType Mat::ViscoAnisotropicType::instance_;


Core::Communication::ParObject* Mat::ViscoAnisotropicType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::ViscoAnisotropic* visco = new Mat::ViscoAnisotropic();
  visco->unpack(buffer);
  return visco;
}


/*----------------------------------------------------------------------*
 |  Constructor                                   (public)         05/08|
 *----------------------------------------------------------------------*/
Mat::ViscoAnisotropic::ViscoAnisotropic() : params_(nullptr) {}


/*----------------------------------------------------------------------*
 |  Copy-Constructor                             (public)          05/08|
 *----------------------------------------------------------------------*/
Mat::ViscoAnisotropic::ViscoAnisotropic(Mat::PAR::ViscoAnisotropic* params) : params_(params) {}


/*----------------------------------------------------------------------*
 |  Pack                                          (public)         05/08|
 *----------------------------------------------------------------------*/
void Mat::ViscoAnisotropic::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);

  int numgp;
  int numhist;
  if (!initialized())
  {
    numgp = 0;
    numhist = 0;
  }
  else
  {
    numgp = a1_->size();  // size is number of gausspoints
    numhist = histstresscurr_->size();
  }
  add_to_pack(data, numgp);
  // Pack internal variables
  for (int gp = 0; gp < numgp; ++gp)
  {
    add_to_pack(data, a1_->at(gp));
    add_to_pack(data, a2_->at(gp));
    add_to_pack(data, ca1_->at(gp));
    add_to_pack(data, ca2_->at(gp));
  }
  // Pack history data
  if (numhist != 0) add_to_pack(data, numhist);  // Length of history vector(s)
  for (int var = 0; var < numhist; ++var)
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
void Mat::ViscoAnisotropic::unpack(Core::Communication::UnpackBuffer& buffer)
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
        params_ = static_cast<Mat::PAR::ViscoAnisotropic*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }

  int numgp, numhist;
  extract_from_pack(buffer, numgp);
  if (numgp == 0)
  {  // no history data to unpack
    isinit_ = false;

    return;
  }
  // unpack fiber internal variables
  a1_ = std::make_shared<std::vector<std::vector<double>>>(numgp);
  a2_ = std::make_shared<std::vector<std::vector<double>>>(numgp);
  ca1_ = std::make_shared<std::vector<std::vector<double>>>(numgp);
  ca2_ = std::make_shared<std::vector<std::vector<double>>>(numgp);

  for (int gp = 0; gp < numgp; ++gp)
  {
    std::vector<double> a;
    extract_from_pack(buffer, a);
    a1_->at(gp) = a;
    extract_from_pack(buffer, a);
    a2_->at(gp) = a;
    extract_from_pack(buffer, a);
    ca1_->at(gp) = a;
    extract_from_pack(buffer, a);
    ca2_->at(gp) = a;
  }

  // unpack history
  extract_from_pack(buffer, numhist);
  histstresscurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  artstresscurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  histstresslast_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  artstresslast_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  for (int var = 0; var < numhist; var++)
  {
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> tmp_vector(Core::LinAlg::Initialization::zero);

    // last vectors are unpacked
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
 *----------------------------------------------------------------------*/
void Mat::ViscoAnisotropic::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  /*fiber directions can be defined in the element line
    or by element thickness direction.
    Since we do not know know if thickness direction is defined, fibers are
    related to a local element cosy which has to be specified in the element line */

  a1_ = std::make_shared<std::vector<std::vector<double>>>(numgp);
  a2_ = std::make_shared<std::vector<std::vector<double>>>(numgp);
  ca1_ = std::make_shared<std::vector<std::vector<double>>>(numgp);
  ca2_ = std::make_shared<std::vector<std::vector<double>>>(numgp);

  if ((params_->gamma_ < 0) || (params_->gamma_ > 90)) FOUR_C_THROW("Fiber angle not in [0,90]");
  const double gamma = (params_->gamma_ * std::numbers::pi) / 180.;  // convert

  // read local (cylindrical) cosy-directions at current element
  FOUR_C_ASSERT_ALWAYS(coord_system, "Require RAD, AXI and CIR parameters.");

  Core::LinAlg::Matrix<3, 3> locsys;
  // basis is local cosy with third vec e3 = circumferential dir and e2 = axial dir
  double radnorm = std::sqrt(Core::LinAlg::norm2(coord_system->element_system[0]));
  double axinorm = std::sqrt(Core::LinAlg::norm2(coord_system->element_system[1]));
  double cirnorm = std::sqrt(Core::LinAlg::norm2(coord_system->element_system[2]));

  for (int i = 0; i < 3; ++i)
  {
    locsys(i, 0) = coord_system->element_system[0](i) / radnorm;
    locsys(i, 1) = coord_system->element_system[1](i) / axinorm;
    locsys(i, 2) = coord_system->element_system[2](i) / cirnorm;
  }
  for (int gp = 0; gp < numgp; ++gp)
  {
    a1_->at(gp).resize(3);
    a2_->at(gp).resize(3);
    ca1_->at(gp).resize(3);
    ca2_->at(gp).resize(3);
    for (int i = 0; i < 3; ++i)
    {
      // a1 = cos gamma e3 + sin gamma e2
      a1_->at(gp)[i] = cos(gamma) * locsys(i, 2) + sin(gamma) * locsys(i, 1);
      // a2 = cos gamma e3 - sin gamma e2
      a2_->at(gp)[i] = cos(gamma) * locsys(i, 2) - sin(gamma) * locsys(i, 1);
      ca1_->at(gp)[i] = a1_->at(gp)[i];
      ca2_->at(gp)[i] = a2_->at(gp)[i];
    }
  }

  // some input checking of viscous parameters for convenience
  if ((params_->beta_[0] < 0) || (params_->beta_[1] < 0) || (params_->relax_[0] <= 0) ||
      (params_->relax_[1] <= 0))
    FOUR_C_THROW("Check visocus parameters! Found beta < 0 or relax <= 0!");

  // initialize hist variables
  histstresscurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  artstresscurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  histstresslast_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  artstresslast_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  const Core::LinAlg::Matrix<NUM_STRESS_3D, 1> emptyvec(Core::LinAlg::Initialization::zero);

  // how many stress types are used?
  const int numst = params_->numstresstypes_;
  histstresscurr_->resize(numst * numgp);
  histstresslast_->resize(numst * numgp);
  artstresscurr_->resize(numst * numgp);
  artstresslast_->resize(numst * numgp);
  for (int j = 0; j < numst * numgp; ++j)
  {
    histstresscurr_->at(j) = emptyvec;
    histstresslast_->at(j) = emptyvec;
    artstresscurr_->at(j) = emptyvec;
    artstresslast_->at(j) = emptyvec;
  }

  isinit_ = true;
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::ViscoAnisotropic::setup(const int numgp, const std::vector<double> thickvec)
{
  // fiber directions can be defined by element thickness direction if specified
  // in material definition
  if (params_->elethick_ == 1)
  {
    a1_ = std::make_shared<std::vector<std::vector<double>>>(numgp);
    a2_ = std::make_shared<std::vector<std::vector<double>>>(numgp);
    ca1_ = std::make_shared<std::vector<std::vector<double>>>(numgp);
    ca2_ = std::make_shared<std::vector<std::vector<double>>>(numgp);

    if (abs(params_->gamma_) >= 1.0E-6)
      FOUR_C_THROW("Fibers can only be aligned in thickness direction for gamma = 0.0!");
    const double gamma = (params_->gamma_ * std::numbers::pi) / 180.;  // convert

    // Fibers are related to the element thickness direction
    std::vector<double> rad = thickvec;
    std::vector<double> axi = thickvec;
    std::vector<double> cir = thickvec;

    Core::LinAlg::Matrix<3, 3> locsys;
    // basis is local cosy with third vec e3 = circumferential dir and e2 = axial dir
    double radnorm = 0.;
    double axinorm = 0.;
    double cirnorm = 0.;
    for (int i = 0; i < 3; ++i)
    {
      radnorm += rad[i] * rad[i];
      axinorm += axi[i] * axi[i];
      cirnorm += cir[i] * cir[i];
    }
    radnorm = sqrt(radnorm);
    axinorm = sqrt(axinorm);
    cirnorm = sqrt(cirnorm);
    for (int i = 0; i < 3; ++i)
    {
      locsys(i, 0) = rad[i] / radnorm;
      locsys(i, 1) = axi[i] / axinorm;
      locsys(i, 2) = cir[i] / cirnorm;
    }
    for (int gp = 0; gp < numgp; ++gp)
    {
      a1_->at(gp).resize(3);
      a2_->at(gp).resize(3);
      ca1_->at(gp).resize(3);
      ca2_->at(gp).resize(3);
      for (int i = 0; i < 3; ++i)
      {
        // a1 = cos gamma e3 + sin gamma e2
        a1_->at(gp)[i] = cos(gamma) * locsys(i, 2) + sin(gamma) * locsys(i, 1);
        // a2 = cos gamma e3 - sin gamma e2
        a2_->at(gp)[i] = cos(gamma) * locsys(i, 2) - sin(gamma) * locsys(i, 1);
        ca1_->at(gp)[i] = a1_->at(gp)[i];
        ca2_->at(gp)[i] = a2_->at(gp)[i];
      }
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::ViscoAnisotropic::update()
{
  // make current values to values of last step
  histstresslast_ = histstresscurr_;
  artstresslast_ = artstresscurr_;

  // empty vectors of current data
  const Core::LinAlg::Matrix<NUM_STRESS_3D, 1> emptyvec(Core::LinAlg::Initialization::zero);
  histstresscurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  artstresscurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>();
  const int histsize = histstresslast_->size();
  histstresscurr_->resize(histsize);
  artstresscurr_->resize(histsize);
  for (int j = 0; j < histsize; ++j)
  {
    histstresscurr_->at(j) = emptyvec;
    artstresscurr_->at(j) = emptyvec;
  }

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::ViscoAnisotropic::update_fiber_dirs(const int gp, Core::LinAlg::Matrix<3, 3>* defgrad)
{
  // Loop over all gp and update fiber directions
  ca1_->at(gp).resize(3);
  ca2_->at(gp).resize(3);
  Core::LinAlg::DenseFunctions::multiply<double, 3, 3, 1>(
      ca1_->at(gp).data(), defgrad->data(), a1_->at(gp).data());
  Core::LinAlg::DenseFunctions::multiply<double, 3, 3, 1>(
      ca2_->at(gp).data(), defgrad->data(), a2_->at(gp).data());
  // std::cout << (ca1_->at(gp))[0] << ",  " << (ca1_->at(gp))[1] << ",  " << (ca1_->at(gp))[2] <<
  // std::endl; std::cout <<  (a1_->at(gp))[0] << ",  " <<  (a1_->at(gp))[1] << ",  " <<
  // (a1_->at(gp))[2] << std::endl;
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::ViscoAnisotropic::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  const Core::LinAlg::Matrix<6, 1> glstrain_mat =
      Core::LinAlg::make_strain_like_voigt_matrix(glstrain);
  Core::LinAlg::Matrix<6, 1> stress_view = Core::LinAlg::make_stress_like_voigt_view(stress);
  Core::LinAlg::Matrix<6, 6> cmat_view = Core::LinAlg::make_stress_like_voigt_view(cmat);

  const double mue = params_->mue_;
  const double kappa = params_->kappa_;
  const double k1 = params_->k1_;
  const double k2 = params_->k2_;
  const double minstretch = params_->minstretch_;

  // right Cauchy-Green Tensor  C = 2 * E + I
  // build identity tensor I
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Id(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++) Id(i) = 1.0;
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> C(glstrain_mat);
  C.scale(2.0);
  C += Id;

  // invariants
  const double I1 = C(0) + C(1) + C(2);  // 1st invariant, trace
  const double I3 = C(0) * C(1) * C(2) + 0.25 * C(3) * C(4) * C(5) - 0.25 * C(1) * C(5) * C(5) -
                    0.25 * C(2) * C(3) * C(3) -
                    0.25 * C(0) * C(4) * C(4);  // 3rd invariant, determinant
  const double J = sqrt(I3);
  const double incJ = std::pow(I3, -1.0 / 3.0);  // J^{-2/3}

  // invert C
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Cinv(Core::LinAlg::Initialization::zero);

  Cinv(0) = C(1) * C(2) - 0.25 * C(4) * C(4);
  Cinv(1) = C(0) * C(2) - 0.25 * C(5) * C(5);
  Cinv(2) = C(0) * C(1) - 0.25 * C(3) * C(3);
  Cinv(3) = 0.25 * C(5) * C(4) - 0.5 * C(3) * C(2);
  Cinv(4) = 0.25 * C(3) * C(5) - 0.5 * C(0) * C(4);
  Cinv(5) = 0.25 * C(3) * C(4) - 0.5 * C(5) * C(1);
  Cinv.scale(1.0 / I3);

  // isotropic part: NeoHooke  ************************************************
  // NeoHooke with penalty W = W^dev(C) + U(J)
  // W = 1/2 mue (^I1-3) + 1/2 kappa (J-1)^2

  // S = Svol + Siso
  // Svol = J*kappa*(J-1)
  // Isochoric (deviatoric) part via projection PP:Sbar, see Holzapfel p. 230
  // Siso = J^{-2/3}  Dev[Sbar] = J^{-2/3} [Sbar - 1/3 trace(Sbar C) Cinv
  // for this Wiso trace(C Sbar) = trace(mue I C) = mue I1
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> SisoEla_nh;  // isochoric elastic S from NeoHooke
  const double third = 1. / 3.;
  const double p = kappa * (J - 1);
  for (int i = 0; i < 6; ++i)
  {
    stress_view(i) = J * p * Cinv(i);  // volumetric part, not affected by viscosity
    SisoEla_nh(i) = incJ * (mue * Id(i) - third * mue * I1 * Cinv(i));
  }

  // Elasticity =  Cvol + Ciso, via projection see Holzapfel p. 255
  //             + viscous C

  // Cvol = J(p + J dp/dJ) Cinv x Cinv  -  2 J p Cinv o Cinv
  // Ciso = 0 + 2/3 J^{-2/3} Sbar:C Psl - 2/3 (Cinv x Siso + Siso x Cinv)
  // Cvol not affected by viscosity
  Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> CisoEla_nh(
      Core::LinAlg::Initialization::zero);  // isochoric elastic C from NeoHooke

  cmat += (-2 * J * p) *
          Core::LinAlg::FourTensorOperations::holzapfel_product(
              make_symmetric_tensor_from_stress_like_voigt_matrix(Cinv));  // -2 J p Cinv o Cinv

  const double fac = 2 * third * incJ * mue * I1;  // 2/3 J^{-2/3} Sbar:C
  // fac Psl = fac (Cinv o Cinv) - fac/3 (Cinv x Cinv)

  Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> Psl(
      Core::LinAlg::Initialization::zero);  // Psl = Cinv o Cinv - 1/3 Cinv x Cinv
  Core::LinAlg::FourTensorOperations::add_holzapfel_product(
      Psl, Cinv, 1.0);  // first part Psl = Cinv o Cinv

  for (int i = 0; i < 6; ++i)
  {
    for (int j = 0; j < 6; ++j)
    {
      double Siso_i = incJ * (mue * Id(i) - third * mue * I1 * Cinv(i));
      double Siso_j = incJ * (mue * Id(j) - third * mue * I1 * Cinv(j));
      cmat_view(i, j) += J * (p + J * kappa) * Cinv(i) * Cinv(j);  // J(p + J dp/dJ) Cinv x Cinv
      Psl(i, j) += (-third) * Cinv(i) * Cinv(j);          // on the fly complete Psl needed later
      CisoEla_nh(i, j) = fac * Psl(i, j)                  // fac Psl
                         - 2 * third * Cinv(i) * Siso_j   // -2/3 Cinv x Siso
                         - 2 * third * Cinv(j) * Siso_i;  // -2/3 Siso x Cinv
    }
  }

  // anisotropic part: ***************************************************
  // W_aniso=(k1/(2.0*k2))*(exp(k2*pow((Ibar_{4,6} - 1.0),2)-1.0)); fiber SEF

  // structural tensors in voigt notation
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> A1;
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> A2;
  for (int i = 0; i < 3; ++i)
  {
    A1(i) = a1_->at(gp)[i] * a1_->at(gp)[i];
    A2(i) = a2_->at(gp)[i] * a2_->at(gp)[i];
  }
  A1(3) = a1_->at(gp)[0] * a1_->at(gp)[1];
  A1(4) = a1_->at(gp)[1] * a1_->at(gp)[2];
  A1(5) = a1_->at(gp)[0] * a1_->at(gp)[2];
  A2(3) = a2_->at(gp)[0] * a2_->at(gp)[1];
  A2(4) = a2_->at(gp)[1] * a2_->at(gp)[2];
  A2(5) = a2_->at(gp)[0] * a2_->at(gp)[2];

  // modified (fiber-) invariants Ibar_{4,6} = J_{4,6} = J^{-2/3} I_{4,6}
  // Voigt: trace(AB) =  a11 b11 + 2 a12 b12 + 2 a13 b13 + a22 b22 + 2 a23 b23 + a33 b33
  // however factor 2 for shear terms is already in C
  double J4 =
      incJ * (A1(0) * C(0) + A1(1) * C(1) + A1(2) * C(2) +
                 1. * (A1(3) * C(3) + A1(4) * C(4) + A1(5) * C(5)));  // J4 = trace(A1:C^dev)
  double J6 =
      incJ * (A2(0) * C(0) + A2(1) * C(1) + A2(2) * C(2) +
                 1. * (A2(3) * C(3) + A2(4) * C(4) + A2(5) * C(5)));  // J6 = trace(A2:C^dev)

  // fibers can only stretch/compress down to a minimal value
  double fib1_tension = 1.;
  double fib2_tension = 1.;

  if (J4 < minstretch)
  {
    //    std::cout<<"Fiber compression exceeded minstretch! J4 = " << J4 <<std::endl;
    J4 = minstretch;
    fib1_tension = 0.;
  }
  if (J6 < minstretch)
  {
    // std::cout<<"Fiber compression exceeded minstretch! J6 = " << J6 <<std::endl;
    J6 = minstretch;
    fib2_tension = 0.;
  }

  // PK2 fiber part in split formulation, see Holzapfel p. 271
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> SisoEla_fib1(A1);  // first compute Sfbar1 = dWf/dJ4 A1
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> SisoEla_fib2(A2);  // first compute Sfbar2 = dWf/dJ6 A2
  const double exp1 = exp(k2 * (J4 - 1.) * (J4 - 1.));
  const double exp2 = exp(k2 * (J6 - 1.) * (J6 - 1.));
  const double fib1 = 2. * (k1 * (J4 - 1.) * exp1);  // 2 dWf/dJ4
  const double fib2 = 2. * (k1 * (J6 - 1.) * exp2);  // 2 dWf/dJ6
  SisoEla_fib1.scale(fib1);
  SisoEla_fib2.scale(fib2);

  const double traceCSfbar1 = SisoEla_fib1(0) * C(0) + SisoEla_fib1(1) * C(1) +
                              SisoEla_fib1(2) * C(2) +
                              1. * (SisoEla_fib1(3) * C(3) + SisoEla_fib1(4) * C(4) +
                                       SisoEla_fib1(5) * C(5));  // trace(Sfbar1 C)
  const double traceCSfbar2 = SisoEla_fib2(0) * C(0) + SisoEla_fib2(1) * C(1) +
                              SisoEla_fib2(2) * C(2) +
                              1. * (SisoEla_fib2(3) * C(3) + SisoEla_fib2(4) * C(4) +
                                       SisoEla_fib2(5) * C(5));  // trace(Sfbar2 C)
  // compute Sfiso_a = J^{-2/3} * (Sfbar_a - 1/3 trace(Sfbar_a C) Cinv
  for (int i = 0; i < 6; ++i)
  {
    SisoEla_fib1(i) = incJ * (SisoEla_fib1(i) - third * traceCSfbar1 * Cinv(i));
    SisoEla_fib2(i) = incJ * (SisoEla_fib2(i) - third * traceCSfbar2 * Cinv(i));
  }

  // Elasticity fiber part in split formulation, see Holzapfel p. 255 and 272
  const double delta7bar1 =
      fib1_tension * 4. *
      (k1 * exp1 + 2. * k1 * k2 * (J4 - 1.) * (J4 - 1.) * exp1);  // 4 d^2Wf/dJ4dJ4
  const double delta7bar2 =
      fib2_tension * 4. *
      (k1 * exp2 + 2. * k1 * k2 * (J6 - 1.) * (J6 - 1.) * exp2);  // 4 d^2Wf/dJ6dJ6

  Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> CisoEla_fib1;  // isochoric elastic C from Fib1
  Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> CisoEla_fib2;  // isochoric elastic C from Fib2

  for (int i = 0; i < 6; ++i)
  {
    for (int j = 0; j < 6; ++j)
    {
      double A1iso_i = incJ * A1(i) - third * J4 * Cinv(i);  // A1iso = J^{-2/3} A1 - 1/3 J4 Cinv
      double A1iso_j = incJ * A1(j) - third * J4 * Cinv(j);
      double A2iso_i = incJ * A2(i) - third * J6 * Cinv(i);  // A2iso = J^{-2/3} A2 - 1/3 J6 Cinv
      double A2iso_j = incJ * A2(j) - third * J6 * Cinv(j);
      CisoEla_fib1(i, j) =
          delta7bar1 * A1iso_i * A1iso_j                  // delta7bar1 A1iso x A1iso
          + 2. * third * incJ * traceCSfbar1 * Psl(i, j)  // 2/3 J^{-2/3} trace(Sfbar C) Psl
          - 2. * third *
                (Cinv(i) * SisoEla_fib1(j) +
                    Cinv(j) * SisoEla_fib1(i));  // -2/3 (Cinv x Sfiso1 + Sfiso1 x Cinv)
      CisoEla_fib2(i, j) =
          delta7bar2 * A2iso_i * A2iso_j                  // delta7bar2 A2iso x A2iso
          + 2. * third * incJ * traceCSfbar2 * Psl(i, j)  // 2/3 J^{-2/3} trace(Sfbar2 C) Psl
          - 2. * third *
                (Cinv(i) * SisoEla_fib2(j) +
                    Cinv(j) * SisoEla_fib2(i));  // -2/3 (Cinv x Sfiso2 + Sfiso2 x Cinv)
    }
  }


  /* -------[-------  Viscous Part -------[-------
   * based on paper "A viscoelastic model for fiber-reinforced composites..." Holzapfel&Gasser,
   * CMAME 2001
   */

  /* Layout for history vectors:
   * [SisoEla_nh(gp1) SisoEla_fib1(gp1) SisoEla_fib2(gp1)  SisoEla_nh(gp2) SisoEla_fib1(gp2)... ]
   * and the same for Q's
   */

  // read history
  const int numst = params_->numstresstypes_;
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> SisoEla_nh_old(histstresslast_->at(numst * gp + 0));
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> SisoEla_fib1_old(histstresslast_->at(numst * gp + 1));
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> SisoEla_fib2_old(histstresslast_->at(numst * gp + 2));
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Q_nh_old(artstresslast_->at(numst * gp + 0));
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Q_fib1_old(artstresslast_->at(numst * gp + 1));
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Q_fib2_old(artstresslast_->at(numst * gp + 2));

  // visco parameters
  /*
   * the betas control the ratio serial/parallel elasticity of the Maxwell-body
   * the taus control the relaxation time for each dashpot path with respect to the serial
   * elasticity
   */
  const double beta_nh = params_->beta_[0];
  const double beta_fib = params_->beta_[1];  // assume same beta for both fibers
  const double tau_nh = params_->relax_[0];
  const double tau_fib = params_->relax_[1];  // assume same tau for both fibers

  // get time algorithmic parameters
  FOUR_C_ASSERT(context.time_step_size, "Time step size not given in evaluation context.");
  double dt = *context.time_step_size;


  /* Time integration according Zien/Taylor and the viscoNeoHooke */
  const double theta = 0.5;
  const double artscalar1_nh = (tau_nh - dt + theta * dt) / tau_nh;
  const double artscalar2_nh = tau_nh / (tau_nh + theta * dt);
  const double artscalar1_fib = (tau_fib - dt + theta * dt) / tau_fib;
  const double artscalar2_fib = tau_fib / (tau_fib + theta * dt);


  // evaluate current Q's
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Q_nh(SisoEla_nh);
  Q_nh.scale(artscalar2_nh * beta_nh);
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Q_fib1(SisoEla_fib1);
  Q_fib1.scale(beta_fib * artscalar2_fib);
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Q_fib2(SisoEla_fib2);
  Q_fib2.scale(beta_fib * artscalar2_fib);

  // scale history
  SisoEla_nh_old.scale(-beta_nh * artscalar2_nh);
  Q_nh_old.scale(artscalar1_nh * artscalar2_nh);
  SisoEla_fib1_old.scale(-beta_fib * artscalar2_fib);
  Q_fib1_old.scale(artscalar1_fib * artscalar2_fib);
  SisoEla_fib2_old.scale(-beta_fib * artscalar2_fib);
  Q_fib2_old.scale(artscalar1_fib * artscalar2_fib);


  /* evaluate current stress */

  // elastic part
  stress_view += SisoEla_nh;  // S_{n+1} = S_vol^ela + S_iso^ela
  stress_view += SisoEla_fib1;
  stress_view += SisoEla_fib2;  // S_{n+1} = S_vol^ela + S_iso^ela

  // viscous part
  Q_nh += Q_nh_old;
  Q_nh += SisoEla_nh_old;  // H_nh = Q_nh_old + S_nh_old
  stress_view += Q_nh;     // S_{n+1} += H_nh + Q_nh_{n+1}

  Q_fib1 += Q_fib1_old;
  Q_fib1 += SisoEla_fib1_old;  // H_fib1 = Q_fib1_old + S_fib1_old
  stress_view += Q_fib1;       // S_{n+1} += H_fib1 + Q_fib1_{n+1}

  Q_fib2 += Q_fib2_old;
  Q_fib2 += SisoEla_fib2_old;  // H_fib2 = Q_fib2_old + S_fib2_old
  stress_view += Q_fib2;       // S_{n+1} += H_fib2 + Q_fib2_{n+1}

  /* evaluate current C-mat */

  /* Time integration according Holzapfel paper */
  /*
  CisoEla_nh.scale(1.+beta_nh*expfac_nh);
  CisoEla_fib1.scale(1.+beta_fib*expfac_fib);
  CisoEla_fib2.scale(1.+beta_fib*expfac_fib);
  */


  /* Time integration according Zien/Taylor and the viscoNeoHooke */
  CisoEla_nh.scale(1 + beta_nh * artscalar2_nh);
  CisoEla_fib1.scale(1 + beta_fib * artscalar2_fib);
  CisoEla_fib2.scale(1 + beta_fib * artscalar2_fib);


  cmat_view += CisoEla_nh;
  cmat_view += CisoEla_fib1;
  cmat_view += CisoEla_fib2;

  // update history
  histstresscurr_->at(numst * gp + 0) = SisoEla_nh;
  histstresscurr_->at(numst * gp + 1) = SisoEla_fib1;
  histstresscurr_->at(numst * gp + 2) = SisoEla_fib2;
  artstresscurr_->at(numst * gp + 0) = Q_nh;
  artstresscurr_->at(numst * gp + 1) = Q_fib1;
  artstresscurr_->at(numst * gp + 2) = Q_fib2;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::ViscoAnisotropic::vis_names(std::map<std::string, int>& names) const
{
  std::string fiber = "Fiber1";
  names[fiber] = 3;  // 3-dim vector
  fiber = "Fiber2";
  names[fiber] = 3;  // 3-dim vector
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Mat::ViscoAnisotropic::vis_data(
    const std::string& name, std::vector<double>& data, int numgp, int eleID) const
{
  std::vector<double> a1 = geta1()->at(0);  // get a1 of first gp
  std::vector<double> a2 = geta2()->at(0);  // get a2 of first gp
  if (name == "Fiber1")
  {
    if ((int)data.size() != 3) FOUR_C_THROW("size mismatch");
    data[0] = a1[0];
    data[1] = a1[1];
    data[2] = a1[2];
  }
  else if (name == "Fiber2")
  {
    if ((int)data.size() != 3) FOUR_C_THROW("size mismatch");
    data[0] = a2[0];
    data[1] = a2[1];
    data[2] = a2[2];
  }
  else
  {
    return false;
  }
  return true;
}

FOUR_C_NAMESPACE_CLOSE
