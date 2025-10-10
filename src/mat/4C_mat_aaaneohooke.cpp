// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_aaaneohooke.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
Mat::PAR::AAAneohooke::AAAneohooke(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata)
{
  Core::LinAlg::Map dummy_map(1, 1, 0,

      Global::Problem::instance()->get_communicators().local_comm());
  for (int i = first; i <= last; i++)
  {
    matparams_.push_back(std::make_shared<Core::LinAlg::Vector<double>>(dummy_map, true));
  }
  matparams_.at(young)->put_scalar(matdata.parameters.get<double>("YOUNG"));
  matparams_.at(nue)->put_scalar(matdata.parameters.get<double>("NUE"));
  matparams_.at(beta)->put_scalar(matdata.parameters.get<double>("BETA"));
  matparams_.at(density)->put_scalar(matdata.parameters.get<double>("DENS"));
}


std::shared_ptr<Core::Mat::Material> Mat::PAR::AAAneohooke::create_material()
{
  return std::make_shared<Mat::AAAneohooke>(this);
}

Mat::AAAneohookeType Mat::AAAneohookeType::instance_;



Core::Communication::ParObject* Mat::AAAneohookeType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::AAAneohooke* aaa = new Mat::AAAneohooke();
  aaa->unpack(buffer);
  return aaa;
}

/*----------------------------------------------------------------------*
 |  Constructor                                   (public)  chfoe 03/08 |
 *----------------------------------------------------------------------*/
Mat::AAAneohooke::AAAneohooke() : params_(nullptr) {}


/*----------------------------------------------------------------------*
 |  Constructor                             (public)   chfoe 03/08 |
 *----------------------------------------------------------------------*/
Mat::AAAneohooke::AAAneohooke(Mat::PAR::AAAneohooke* params) : params_(params) {}

/*----------------------------------------------------------------------*
 |  Pack                                          (public)  chfoe 03/08 |
 *----------------------------------------------------------------------*/
void Mat::AAAneohooke::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);
}

/*----------------------------------------------------------------------*
 |  Unpack                                        (public)  chfoe 03/08 |
 *----------------------------------------------------------------------*/
void Mat::AAAneohooke::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid
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
        params_ = static_cast<Mat::PAR::AAAneohooke*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
}


/*----------------------------------------------------------------------*
 |  Evaluate Material                   (public)  chfoe 03/08 gee 10/08 |
 *----------------------------------------------------------------------*

 plain strain energy function

 W    = alpha (Ic*IIIc^(-1/3) -3) + beta (Ic*IIIc^(-1/3)-3)^2

 taken from
 M.L. Raghavan, D.A. Vorp: Toward a biomechanical tool to evaluate rupture potential
 of abdominal aortic aneurysm: identification of a finite strain constitutive model
 and evaluation of its applicability, J. of Biomechanics 33 (2000) 475-482.

 and modified to slight compressibility

 here

 Ic   .. first invariant of right Cauchy-Green tensor C
 IIIc .. third invariant of right Cauchy-Green tensor C

 The volumetric part is done by a volumetric strain energy function taken from
 Holzapfel

 W_vol = K beta2^(-2) ( beta2 ln (J) + J^(-beta2) -1 )

 where

 K    .. bulk modulus
 beta2 =  -2.0 a parameter according to Doll and Schweizerhof; 9.0 according to Holzapfel,
 alternatively; numerical stability parameter J    .. det(F) determinante of the Jacobian matrix


 Note: Young's modulus is in the input just for convenience. Actually we need the
       parameter alpha (see W above) which is related to E by

     E = 6.0 * alpha.

       Correspondingly the bulk modulus is given by

     K = E / (3-6*nu) = 2*alpha / (1-2*nu)

     with nu = 0.495 we have K = 200 alpha
     with nu = 0.45  we have K =  20 alpha

 */
void Mat::AAAneohooke::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  // map in GetParameter can now calculate LID, so we do not need it here       05/2017 birzle
  // get element lID incase we have element specific material parameters
  //  int eleID = Global::Problem::instance()->GetDis("structure")->ElementColMap()->LID(eleGID);

  // material parameters for isochoric part
  const double youngs = params_->get_parameter(params_->young, eleGID);  // Young's modulus
  const double beta = params_->get_parameter(params_->beta, eleGID);     // second parameter
  const double nue = params_->get_parameter(params_->nue, eleGID);       // Poisson's ratio
  const double alpha = youngs * 0.1666666666666666667;                   // E = alpha * 6..

  // material parameters for volumetric part
  const double beta2 = -2.0;                                           // parameter from Holzapfel
  double komp = (nue != 0.5) ? 2.0 * alpha / (1.0 - 2.0 * nue) : 0.0;  // bulk modulus

  //--------------------------------------------------------------------------------------
  // build identity tensor I
  Core::LinAlg::Matrix<6, 1> identity(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++) identity(i) = 1.0;

  // right Cauchy-Green Tensor  C = 2 * E + I
  Core::LinAlg::Matrix<6, 1> rcg(Core::LinAlg::make_strain_like_voigt_matrix(glstrain));
  rcg.scale(2.0);
  rcg += identity;

  // invariants
  double inv = rcg(0) + rcg(1) + rcg(2);  // 1st invariant, trace
  double iiinv = rcg(0) * rcg(1) * rcg(2) + 0.25 * rcg(3) * rcg(4) * rcg(5) -
                 0.25 * rcg(1) * rcg(5) * rcg(5) - 0.25 * rcg(2) * rcg(3) * rcg(3) -
                 0.25 * rcg(0) * rcg(4) * rcg(4);  // 3rd invariant, determinante

  double detf = 0.0;
  //  if (iiinv < 0.0)
  //    FOUR_C_THROW("fatal failure in aneurysmatic artery wall material");
  //  else
  detf = sqrt(iiinv);  // determinate of deformation gradient

  //--------------------------------------------------------------------------------------
  // invert C
  Core::LinAlg::Matrix<6, 1> invc(Core::LinAlg::Initialization::uninitialized);

  double invdet = 1. / iiinv;

  invc(0) = rcg(1) * rcg(2) - 0.25 * rcg(4) * rcg(4);
  invc(1) = rcg(0) * rcg(2) - 0.25 * rcg(5) * rcg(5);
  invc(2) = rcg(0) * rcg(1) - 0.25 * rcg(3) * rcg(3);
  invc(3) = 0.25 * rcg(5) * rcg(4) - 0.5 * rcg(3) * rcg(2);
  invc(4) = 0.25 * rcg(3) * rcg(5) - 0.5 * rcg(0) * rcg(4);
  invc(5) = 0.25 * rcg(3) * rcg(4) - 0.5 * rcg(5) * rcg(1);

  invc.scale(invdet);

  //--- prepare some constants -----------------------------------------------------------
  const double third = 1.0 / 3.0;
  const double twthi = 2.0 / 3.0;

  double isochor1 = 0.0;
  double isochor2 = 0.0;

  //--- determine 2nd Piola Kirchhoff stresses pktwo -------------------------------------
  // 1st step: isochoric part
  //=========================
  isochor1 = 2.0 * (alpha * pow(iiinv, third) + 2.0 * beta * inv - 6.0 * beta * pow(iiinv, third)) *
             pow(iiinv, -twthi);
  isochor2 = -twthi * inv *
             (alpha * pow(iiinv, third) + 2.0 * beta * inv - 6.0 * beta * pow(iiinv, third)) *
             pow(iiinv, -twthi);

  // contribution: Cinv
  Core::LinAlg::Matrix<6, 1> pktwoiso(invc);
  pktwoiso.scale(isochor2);

  // contribution: I
  for (int i = 0; i < 3; i++) pktwoiso(i) += isochor1;


  // 2nd step: volumetric part
  //==========================
  double scalar = komp / beta2 * (1.0 - pow(detf, -beta2));

  // initialise PKtwo with volumetric part
  Core::LinAlg::Matrix<6, 1> pktwovol(invc);
  pktwovol.scale(scalar);

  // 3rd step: add everything up
  //============================
  auto stress_view = Core::LinAlg::make_stress_like_voigt_view(stress);
  stress_view = pktwoiso;
  stress_view += pktwovol;


  //--- do elasticity matrix -------------------------------------------------------------
  // ensure that cmat is zero when it enters the computation
  // It is an implicit law that cmat is zero upon input
  // cmat.PutScalar(0.0);

  // 1st step: isochoric part
  //=========================

  // deltas (see also Holzapfel p.261)
  // note that these deltas serve for the isochoric part only
  double delta1 = 8.0 * beta * pow(iiinv, -twthi);
  double delta3 = -4. / 3 *
                  (alpha * pow(iiinv, third) + 4. * beta * inv - 6 * beta * pow(iiinv, third)) *
                  pow(iiinv, -twthi);
  double delta6 = 4. / 9 * inv *
                  (alpha * pow(iiinv, third) + 4. * beta * inv - 6 * beta * pow(iiinv, third)) *
                  pow(iiinv, -twthi);
  double delta7 = 4. / 3 * inv *
                  (alpha * pow(iiinv, third) + 2. * beta * inv - 6 * beta * pow(iiinv, third)) *
                  pow(iiinv, -twthi);

  auto cmat_view = Core::LinAlg::make_stress_like_voigt_view(cmat);
  // contribution: I \obtimes I
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) cmat_view(i, j) = delta1;

  // contribution: Cinv \otimes Cinv
  for (int i = 0; i < 6; i++)
    for (int j = 0; j < 6; j++)
    {
      // contribution: Cinv \otimes I + I \otimes Cinv
      cmat_view(i, j) += delta3 * (identity(i) * invc(j) + invc(i) * identity(j));
      // contribution: Cinv \otimes Cinv
      cmat_view(i, j) += delta6 * invc(i) * invc(j);
    }

  // contribution: boeppel-product
  cmat += delta7 * Core::LinAlg::FourTensorOperations::holzapfel_product(
                       Core::LinAlg::make_symmetric_tensor_from_stress_like_voigt_matrix(invc));
  // 2nd step: volumetric part
  //==========================
  delta6 = komp * pow(detf, -beta2);
  delta7 = -2.0 * scalar;

  // contribution: Cinv \otimes Cinv
  for (int i = 0; i < 6; i++)
    for (int j = 0; j < 6; j++) cmat_view(i, j) += delta6 * invc(i) * invc(j);

  // contribution: boeppel-product
  cmat += delta7 * Core::LinAlg::FourTensorOperations::holzapfel_product(
                       Core::LinAlg::make_symmetric_tensor_from_stress_like_voigt_matrix(invc));
}

/*----------------------------------------------------------------------*
 |  calculate strain energy                                hemmler 02/17|
 *----------------------------------------------------------------------*/
double Mat::AAAneohooke::strain_energy(const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const EvaluationContext& context, int gp, int eleGID) const
{
  // material parameters for isochoric part
  const double youngs = params_->get_parameter(params_->young, eleGID);  // Young's modulus
  const double beta = params_->get_parameter(params_->beta, eleGID);     // second parameter
  const double nue = params_->get_parameter(params_->nue, eleGID);       // Poisson's ratio
  const double alpha = youngs * 0.1666666666666666667;                   // E = alpha * 6..

  // material parameters for volumetric part
  const double beta2 = -2.0;                                           // parameter from Holzapfel
  double komp = (nue != 0.5) ? 2.0 * alpha / (1.0 - 2.0 * nue) : 0.0;  // bulk modulus

  // build identity tensor I
  Core::LinAlg::Matrix<6, 1> identity(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++) identity(i) = 1.0;

  // right Cauchy-Green Tensor  C = 2 * E + I
  Core::LinAlg::Matrix<6, 1> rcg(Core::LinAlg::make_strain_like_voigt_matrix(glstrain));
  rcg.scale(2.0);
  rcg += identity;

  // invariants
  double inv = rcg(0) + rcg(1) + rcg(2);  // 1st invariant, trace
  double iiinv = rcg(0) * rcg(1) * rcg(2) + 0.25 * rcg(3) * rcg(4) * rcg(5) -
                 0.25 * rcg(1) * rcg(5) * rcg(5) - 0.25 * rcg(2) * rcg(3) * rcg(3) -
                 0.25 * rcg(0) * rcg(4) * rcg(4);  // 3rd invariant, determinante

  // isochoric part
  // W    = alpha (Ic*IIIc^(-1/3) -3) + beta (Ic*IIIc^(-1/3)-3)^2
  double psi = 0.0;
  psi += alpha * (inv * pow(iiinv, -1 / 3.) - 3) +
         beta * (inv * pow(iiinv, -1 / 3.) - 3) * (inv * pow(iiinv, -1 / 3.) - 3);
  // volumetric part
  // W_vol = K beta2^(-2) ( beta2 ln (J) + J^(-beta2) -1 )
  psi +=
      komp * pow(beta2, -2.) * (beta2 * log(pow(iiinv, -2.)) + pow(pow(iiinv, -2.), -beta2) - 1.0);
  return psi;
}



void Mat::AAAneohooke::vis_names(std::map<std::string, int>& names) const
{
  std::string fiber = "beta";
  names[fiber] = 1;  // scalar
  fiber = "youngs";
  names[fiber] = 1;  // scalar
  fiber = "FourCEleId";
  names[fiber] = 1;  // scalar
}


bool Mat::AAAneohooke::vis_data(
    const std::string& name, std::vector<double>& data, int numgp, int eleGID) const
{
  if (name == "beta")
  {
    if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");
    data[0] = params_->get_parameter(params_->beta, eleGID);
  }
  else if (name == "youngs")
  {
    if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");
    data[0] = params_->get_parameter(params_->young, eleGID);
  }
  else if (name == "FourCEleId")
  {
    if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");
    // get element lID incase we have element specific material parameters
    int eleID = Global::Problem::instance()->get_dis("structure")->element_col_map()->lid(eleGID);
    data[0] = eleID;
  }
  else
  {
    return false;
  }
  return true;
}

FOUR_C_NAMESPACE_CLOSE
