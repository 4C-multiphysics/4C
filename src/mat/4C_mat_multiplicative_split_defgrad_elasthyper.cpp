// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_multiplicative_split_defgrad_elasthyper.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_legacy_enum_definitions_materials.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_anisotropy.hpp"
#include "4C_mat_elast_couptransverselyisotropic.hpp"
#include "4C_mat_elasthyper_service.hpp"
#include "4C_mat_inelastic_defgrad_factors.hpp"
#include "4C_mat_multiplicative_split_defgrad_elasthyper_service.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_ssi_input.hpp"
#include "4C_structure_new_enum_lists.hpp"
#include "4C_utils_enum.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <memory>

FOUR_C_NAMESPACE_OPEN


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::MultiplicativeSplitDefgradElastHyper::MultiplicativeSplitDefgradElastHyper(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      nummat_elast_(matdata.parameters.get<int>("NUMMATEL")),
      matids_elast_(matdata.parameters.get<std::vector<int>>("MATIDSEL")),
      numfac_inel_(matdata.parameters.get<int>("NUMFACINEL")),
      inel_defgradfacids_(matdata.parameters.get<std::vector<int>>("INELDEFGRADFACIDS")),
      density_(matdata.parameters.get<double>("DENS"))
{
  // check if sizes fit
  if (nummat_elast_ != static_cast<int>(matids_elast_.size()))
    FOUR_C_THROW(
        "number of elastic materials {} does not fit to size of elastic material ID vector {}",
        nummat_elast_, matids_elast_.size());

  if (numfac_inel_ != static_cast<int>(inel_defgradfacids_.size()))
  {
    FOUR_C_THROW(
        "number of inelastic deformation gradient factors {} does not fit to size of inelastic "
        "deformation gradient ID vector {}",
        numfac_inel_, inel_defgradfacids_.size());
  }
}

std::shared_ptr<Core::Mat::Material>
Mat::PAR::MultiplicativeSplitDefgradElastHyper::create_material()
{
  return std::make_shared<Mat::MultiplicativeSplitDefgradElastHyper>(this);
}

Mat::MultiplicativeSplitDefgradElastHyperType
    Mat::MultiplicativeSplitDefgradElastHyperType::instance_;

Core::Communication::ParObject* Mat::MultiplicativeSplitDefgradElastHyperType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* splitdefgrad_elhy = new Mat::MultiplicativeSplitDefgradElastHyper();
  splitdefgrad_elhy->unpack(buffer);

  return splitdefgrad_elhy;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::MultiplicativeSplitDefgradElastHyper::MultiplicativeSplitDefgradElastHyper()
    : anisotropy_(std::make_shared<Mat::Anisotropy>()),
      inelastic_(std::make_shared<Mat::InelasticFactorsHandler>()),
      params_(nullptr),
      potsumel_(0),
      potsumel_transviso_(0)
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::MultiplicativeSplitDefgradElastHyper::MultiplicativeSplitDefgradElastHyper(
    Mat::PAR::MultiplicativeSplitDefgradElastHyper* params)
    : anisotropy_(std::make_shared<Mat::Anisotropy>()),
      inelastic_(std::make_shared<Mat::InelasticFactorsHandler>()),
      params_(params),
      potsumel_(0),
      potsumel_transviso_(0)
{
  // elastic materials
  for (int matid_elastic : params_->matids_elast_)
  {
    auto elastic_summand = Mat::Elastic::Summand::factory(matid_elastic);
    if (elastic_summand == nullptr) FOUR_C_THROW("Failed to allocate");
    if (elastic_summand->material_type() == Core::Materials::mes_couptransverselyisotropic)
    {
      potsumel_transviso_.push_back(
          std::dynamic_pointer_cast<Mat::Elastic::CoupTransverselyIsotropic>(elastic_summand));
    }
    else
    {
      potsumel_.push_back(elastic_summand);
    }
    elastic_summand->register_anisotropy_extensions(*anisotropy_);
  }

  inelastic_->assign_to_source(params);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::MultiplicativeSplitDefgradElastHyper::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);

  anisotropy_->pack_anisotropy(data);

  Core::Communication::PotentiallyUnusedBufferScope summand_scope{data};
  if (params_ != nullptr)  // summands are not accessible in postprocessing mode
  {
    // pack inelastic factors handler
    inelastic_->pack_inelastic(data);

    // loop map of associated potential summands
    for (const auto& p : potsumel_) p->pack_summand(data);
    for (const std::shared_ptr<Mat::Elastic::CoupTransverselyIsotropic>& p : potsumel_transviso_)
      p->pack_summand(data);
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::MultiplicativeSplitDefgradElastHyper::unpack(Core::Communication::UnpackBuffer& buffer)
{
  // make sure we have a pristine material
  params_ = nullptr;
  potsumel_.clear();
  potsumel_transviso_.clear();



  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid and recover params_
  int matid;
  extract_from_pack(buffer, matid);
  if (Global::Problem::instance()->materials() != nullptr)
  {
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
      auto* mat = Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
        params_ = dynamic_cast<Mat::PAR::MultiplicativeSplitDefgradElastHyper*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
  }

  anisotropy_->unpack_anisotropy(buffer);

  Core::Communication::PotentiallyUnusedBufferScope summand_scope{buffer};
  if (params_ != nullptr)  // summands are not accessible in postprocessing mode
  {
    // unpack inelastic deformation gradient factors
    inelastic_->assign_to_source(params_);
    inelastic_->unpack_inelastic(buffer);

    // elastic materials
    for (const auto& matid_elastic : params_->matids_elast_)
    {
      auto elastic_summand = Mat::Elastic::Summand::factory(matid_elastic);
      if (elastic_summand == nullptr) FOUR_C_THROW("Failed to allocate");
      if (elastic_summand->material_type() == Core::Materials::mes_couptransverselyisotropic)
      {
        potsumel_transviso_.push_back(
            std::dynamic_pointer_cast<Mat::Elastic::CoupTransverselyIsotropic>(elastic_summand));
      }
      else
      {
        potsumel_.push_back(elastic_summand);
      }
    }
    // loop map of associated potential summands
    for (const auto& elastic_summand : potsumel_)
    {
      elastic_summand->unpack_summand(buffer);
      elastic_summand->register_anisotropy_extensions(*anisotropy_);
    }
    for (const std::shared_ptr<Mat::Elastic::CoupTransverselyIsotropic>& elastic_summand :
        potsumel_transviso_)
    {
      elastic_summand->unpack_summand(buffer);
    }
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::MultiplicativeSplitDefgradElastHyper::evaluate(
    const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  const Core::LinAlg::Matrix<3, 3> defgrd_mat = Core::LinAlg::make_matrix_view(*defgrad);
  Core::LinAlg::Matrix<6, 1> stress_view = Core::LinAlg::make_stress_like_voigt_view(stress);
  Core::LinAlg::Matrix<6, 6> cmat_view = Core::LinAlg::make_stress_like_voigt_view(cmat);

  // do all stuff that only has to be done once per evaluate() call
  pre_evaluate(params, context, gp, eleGID);

  // compute kinematic quantities
  KinematicQuantities kinematic_quantities =
      evaluate_kinematic_quantities(*this, *inelastic_, defgrd_mat, gp, eleGID);

  // compute stress factors
  StressFactors stress_factors;
  Mat::calculate_gamma_delta(stress_factors.gamma, stress_factors.delta, kinematic_quantities.prinv,
      kinematic_quantities.dPIe, kinematic_quantities.ddPIIe);

  // derivative of 2nd Piola Kirchhoff stresses w.r.t. the inverse inelastic deformation
  // gradient
  Core::LinAlg::Matrix<6, 9> dSdiFin = evaluated_sdi_fin(kinematic_quantities, stress_factors);

  // right Cauchy-Green deformation tensor
  Core::LinAlg::Matrix<3, 3> CM(Core::LinAlg::Initialization::zero);
  CM.multiply_tn(1.0, defgrd_mat, defgrd_mat, 0.0);

  /// part of the elasticity tensor as shown in evaluate_stress_cmat_iso
  Core::LinAlg::Matrix<6, 6> cmatiso{Core::LinAlg::Initialization::zero};

  // cmat = 2 dS/dC = 2 \frac{\partial S}{\partial C} + 2 \frac{\partial S}{\partial F_{in}^{-1}}
  // : \frac{\partial F_{in}^{-1}}{\partial C} = cmatiso + cmatadd
  evaluate_stress_cmat_iso(kinematic_quantities, stress_factors, stress_view, cmatiso);
  // separate update coming from the transversely isotropic components
  if (!(potsumel_transviso_.empty()))
  {
    evaluate_transv_iso_quantities(
        kinematic_quantities, CM, params, gp, eleGID, stress_view, cmatiso, dSdiFin);
  }
  cmat_view.update(1.0, cmatiso, 0.0);

  // evaluate additional terms for the elasticity tensor
  // cmatadd = 2 \frac{\partial S}{\partial F_{in}^{-1}} : \frac{\partial F_{in}^{-1}}{\partial
  // C}, where F_{in}^{-1} can be multiplicatively composed of several inelastic contributions
  Core::LinAlg::Matrix<6, 6> cmatadd =
      evaluate_additional_cmat(&defgrd_mat, kinematic_quantities.iCV, dSdiFin);
  cmat_view.update(1.0, cmatadd, 1.0);
}

Core::LinAlg::SymmetricTensor<double, 3, 3>
Mat::MultiplicativeSplitDefgradElastHyper::evaluate_d_stress_d_scalar(
    const Core::LinAlg::Tensor<double, 3, 3>& defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext& context, int gp, int eleGID)
{
  Core::LinAlg::Matrix<3, 3> defgrad_mat = Core::LinAlg::make_matrix_view(defgrad);
  // do all stuff that only has to be done once per evaluate() call
  pre_evaluate(params, context, gp, eleGID);

  // get source of deformation for this OD block depending on the differentiation type
  auto source(PAR::InelasticSource::none);
  const int differentiationtype = get_or<int>(
      params, "differentiationtype", static_cast<int>(Solid::DifferentiationType::none));
  if (differentiationtype == static_cast<int>(Solid::DifferentiationType::elch))
    source = PAR::InelasticSource::concentration;
  else if (differentiationtype == static_cast<int>(Solid::DifferentiationType::temp))
    source = PAR::InelasticSource::temperature;
  else
    FOUR_C_THROW("unknown scalaratype");



  KinematicQuantities kinematic_quantities =
      evaluate_kinematic_quantities(*this, *inelastic_, defgrad_mat, gp, eleGID);
  StressFactors stress_factors;
  Mat::calculate_gamma_delta(stress_factors.gamma, stress_factors.delta, kinematic_quantities.prinv,
      kinematic_quantities.dPIe, kinematic_quantities.ddPIIe);
  Core::LinAlg::Matrix<6, 9> dSdiFin = evaluated_sdi_fin(kinematic_quantities, stress_factors);

  Core::LinAlg::SymmetricTensor<double, 3, 3> d_stress_d_scalar{};
  Core::LinAlg::Matrix<6, 1> d_stress_d_scalar_view =
      Core::LinAlg::make_stress_like_voigt_view(d_stress_d_scalar);
  evaluate_od_stiff_mat(source, &defgrad_mat, dSdiFin, d_stress_d_scalar_view);
  return d_stress_d_scalar;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
double Mat::MultiplicativeSplitDefgradElastHyper::evaluate_cauchy_n_dir_and_derivatives(
    const Core::LinAlg::Tensor<double, 3, 3>& defgrd, const Core::LinAlg::Tensor<double, 3>& n,
    const Core::LinAlg::Tensor<double, 3>& dir, Core::LinAlg::Matrix<3, 1>* d_cauchyndir_dn,
    Core::LinAlg::Matrix<3, 1>* d_cauchyndir_ddir, Core::LinAlg::Matrix<9, 1>* d_cauchyndir_dF,
    Core::LinAlg::Matrix<9, 9>* d2_cauchyndir_dF2, Core::LinAlg::Matrix<9, 3>* d2_cauchyndir_dF_dn,
    Core::LinAlg::Matrix<9, 3>* d2_cauchyndir_dF_ddir, const EvaluationContext& context, int eleGID,
    const double* concentration, const double* temp, double* d_cauchyndir_dT,
    Core::LinAlg::Matrix<9, 1>* d2_cauchyndir_dF_dT)
{
  if (concentration != nullptr) set_concentration_gp(*concentration);

  // reset sigma contracted with n and dir
  double cauchy_n_dir = 0.0;
  const Core::LinAlg::Matrix<3, 3> defgrd_mat = Core::LinAlg::make_matrix_view(defgrd);
  const Core::LinAlg::Matrix<3, 1> n_mat = Core::LinAlg::make_matrix_view<3, 1>(n);
  const Core::LinAlg::Matrix<3, 1> dir_mat = Core::LinAlg::make_matrix_view<3, 1>(dir);

  static Core::LinAlg::Matrix<6, 1> idV(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; ++i) idV(i) = 1.0;
  static Core::LinAlg::Matrix<3, 3> idM(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; ++i) idM(i, i) = 1.0;
  static Core::LinAlg::Matrix<3, 3> iFinM(Core::LinAlg::Initialization::zero);
  inelastic_->evaluate_inverse_inelastic_def_grad(&defgrd_mat, iFinM);
  static Core::LinAlg::Matrix<3, 3> FeM(Core::LinAlg::Initialization::zero);
  FeM.multiply_nn(1.0, defgrd_mat, iFinM, 0.0);

  // get elastic left cauchy-green tensor and corresponding principal invariants
  static Core::LinAlg::Matrix<3, 3> beM(Core::LinAlg::Initialization::zero);
  beM.multiply_nt(1.0, FeM, FeM, 0.0);
  static Core::LinAlg::Matrix<6, 1> beV_strain(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Strains::matrix_to_vector(beM, beV_strain);
  static Core::LinAlg::Matrix<3, 1> prinv(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Strains::invariants_principal(prinv, beV_strain);
  static Core::LinAlg::Matrix<6, 1> beV_stress(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(beM, beV_stress);

  static Core::LinAlg::Matrix<3, 1> beMdn(Core::LinAlg::Initialization::zero);
  beMdn.multiply(1.0, beM, n_mat, 0.0);
  const double beMdnddir = beMdn.dot(dir_mat);
  static Core::LinAlg::Matrix<3, 1> beMddir(Core::LinAlg::Initialization::zero);
  beMddir.multiply(1.0, beM, dir_mat, 0.0);

  static Core::LinAlg::Matrix<3, 3> ibeM(Core::LinAlg::Initialization::zero);
  ibeM.invert(beM);
  static Core::LinAlg::Matrix<6, 1> ibeV_stress(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(ibeM, ibeV_stress);
  static Core::LinAlg::Matrix<3, 1> ibeMdn(Core::LinAlg::Initialization::zero);
  ibeMdn.multiply(1.0, ibeM, n_mat, 0.0);
  const double ibeMdnddir = ibeMdn.dot(dir_mat);
  static Core::LinAlg::Matrix<3, 1> ibeMddir(Core::LinAlg::Initialization::zero);
  ibeMddir.multiply(1.0, ibeM, dir_mat, 0.0);

  // derivatives of principle invariants of elastic left cauchy-green tensor
  static Core::LinAlg::Matrix<3, 1> dPI(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<6, 1> ddPII(Core::LinAlg::Initialization::zero);
  constexpr int dummy_gp = -1;
  evaluate_invariant_derivatives(prinv, dummy_gp, eleGID, dPI, ddPII);

  const double detFe = FeM.determinant();
  const double nddir = n * dir;
  const double prefac = 2.0 / detFe;

  // calculate \mat{\sigma} \cdot \vec{n} \cdot \vec{v}
  cauchy_n_dir = prefac * (prinv(1) * dPI(1) * nddir + prinv(2) * dPI(2) * nddir +
                              dPI(0) * beMdnddir - prinv(2) * dPI(1) * ibeMdnddir);

  if (d_cauchyndir_dn)
  {
    d_cauchyndir_dn->update(prinv(1) * dPI(1) + prinv(2) * dPI(2), dir_mat, 0.0);
    d_cauchyndir_dn->update(dPI(0), beMddir, 1.0);
    d_cauchyndir_dn->update(-prinv(2) * dPI(1), ibeMddir, 1.0);
    d_cauchyndir_dn->scale(prefac);
  }

  if (d_cauchyndir_ddir)
  {
    d_cauchyndir_ddir->update(prinv(1) * dPI(1) + prinv(2) * dPI(2), n_mat, 0.0);
    d_cauchyndir_ddir->update(dPI(0), beMdn, 1.0);
    d_cauchyndir_ddir->update(-prinv(2) * dPI(1), ibeMdn, 1.0);
    d_cauchyndir_ddir->scale(prefac);
  }

  if (d_cauchyndir_dF)
  {
    static Core::LinAlg::Matrix<6, 1> d_I1_be(Core::LinAlg::Initialization::zero);
    d_I1_be = idV;
    static Core::LinAlg::Matrix<6, 1> d_I2_be(Core::LinAlg::Initialization::zero);
    d_I2_be.update(prinv(0), idV, -1.0, beV_stress);
    static Core::LinAlg::Matrix<6, 1> d_I3_be(Core::LinAlg::Initialization::zero);
    d_I3_be.update(prinv(2), ibeV_stress, 0.0);

    // calculation of \partial b_{el} / \partial F (elastic left cauchy-green w.r.t. deformation
    // gradient)
    static Core::LinAlg::Matrix<6, 9> d_be_dFe(Core::LinAlg::Initialization::zero);
    d_be_dFe.clear();
    Core::LinAlg::FourTensorOperations::add_right_non_symmetric_holzapfel_product_strain_like(
        d_be_dFe, idM, FeM, 1.0);
    static Core::LinAlg::Matrix<9, 9> d_Fe_dF(Core::LinAlg::Initialization::zero);
    d_Fe_dF.clear();
    Core::LinAlg::FourTensorOperations::add_non_symmetric_product(1.0, idM, iFinM, d_Fe_dF);
    static Core::LinAlg::Matrix<6, 9> d_be_dF(Core::LinAlg::Initialization::zero);
    d_be_dF.multiply(1.0, d_be_dFe, d_Fe_dF, 0.0);

    // calculation of \partial I_i / \partial F (Invariants of b_{el} w.r.t. deformation gradient)
    static Core::LinAlg::Matrix<9, 1> d_I1_dF(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<9, 1> d_I2_dF(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<9, 1> d_I3_dF(Core::LinAlg::Initialization::zero);
    d_I1_dF.multiply_tn(1.0, d_be_dF, d_I1_be, 0.0);
    d_I2_dF.multiply_tn(1.0, d_be_dF, d_I2_be, 0.0);
    d_I3_dF.multiply_tn(1.0, d_be_dF, d_I3_be, 0.0);

    // add d_cauchyndir_dI1 \odot d_I1_dF and clear static matrix
    d_cauchyndir_dF->update(prefac * (prinv(1) * ddPII(5) * nddir + prinv(2) * ddPII(4) * nddir +
                                         ddPII(0) * beMdnddir - prinv(2) * ddPII(5) * ibeMdnddir),
        d_I1_dF, 0.0);
    // add d_cauchyndir_dI2 \odot d_I2_dF
    d_cauchyndir_dF->update(
        prefac * (dPI(1) * nddir + prinv(1) * ddPII(1) * nddir + prinv(2) * ddPII(3) * nddir +
                     ddPII(5) * beMdnddir - prinv(2) * ddPII(1) * ibeMdnddir),
        d_I2_dF, 1.0);
    // add d_cauchyndir_dI3 \odot d_I3_dF
    d_cauchyndir_dF->update(
        prefac * (prinv(1) * ddPII(3) * nddir + dPI(2) * nddir + prinv(2) * ddPII(2) * nddir +
                     ddPII(4) * beMdnddir - dPI(1) * ibeMdnddir - prinv(2) * ddPII(3) * ibeMdnddir),
        d_I3_dF, 1.0);

    // next three updates add partial derivative of snt w.r.t. the deformation gradient F for
    // constant invariants first part is term arising from \partial Je^{-1} / \partial F
    static Core::LinAlg::Matrix<3, 3> iFeM(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<3, 3> iFeTM(Core::LinAlg::Initialization::zero);
    iFeM.invert(FeM);
    iFeTM.update_t(1.0, iFeM, 0.0);
    static Core::LinAlg::Matrix<9, 1> iFeTV(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Voigt::matrix_3x3_to_9x1(iFeTM, iFeTV);
    static Core::LinAlg::Matrix<1, 9> d_iJe_dFV(Core::LinAlg::Initialization::zero);
    d_iJe_dFV.multiply_tn(1.0, iFeTV, d_Fe_dF, 0.0);
    d_cauchyndir_dF->update_t(-cauchy_n_dir, d_iJe_dFV, 1.0);

    // second part is term arising from \partial b_el * n * v / \partial F
    static Core::LinAlg::Matrix<3, 3> FeMiFinTM(Core::LinAlg::Initialization::zero);
    FeMiFinTM.multiply_nt(1.0, FeM, iFinM, 0.0);
    static Core::LinAlg::Matrix<3, 1> tempvec(Core::LinAlg::Initialization::zero);
    tempvec.multiply_tn(1.0, FeMiFinTM, n_mat, 0.0);
    static Core::LinAlg::Matrix<3, 3> d_bednddir_dF(Core::LinAlg::Initialization::zero);
    d_bednddir_dF.multiply_nt(1.0, dir_mat, tempvec, 0.0);
    // now reuse tempvec
    tempvec.multiply_tn(1.0, FeMiFinTM, dir_mat, 0.0);
    d_bednddir_dF.multiply_nt(1.0, n_mat, tempvec, 1.0);
    static Core::LinAlg::Matrix<9, 1> d_bednddir_dFV(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Voigt::matrix_3x3_to_9x1(d_bednddir_dF, d_bednddir_dFV);
    d_cauchyndir_dF->update(prefac * dPI(0), d_bednddir_dFV, 1.0);

    // third part is term arising from \partial b_el^{-1} * n * v / \partial F
    static Core::LinAlg::Matrix<3, 3> iFM(Core::LinAlg::Initialization::zero);
    iFM.invert(defgrd_mat);
    static Core::LinAlg::Matrix<3, 1> tempvec2(Core::LinAlg::Initialization::zero);
    tempvec.multiply(1.0, ibeM, dir_mat, 0.0);
    tempvec2.multiply(1.0, iFM, n_mat, 0.0);
    static Core::LinAlg::Matrix<3, 3> d_ibednddir_dFM(Core::LinAlg::Initialization::zero);
    d_ibednddir_dFM.multiply_nt(1.0, tempvec, tempvec2, 0.0);
    // now reuse tempvecs
    tempvec.multiply(1.0, ibeM, n_mat, 0.0);
    tempvec2.multiply(1.0, iFM, dir_mat, 0.0);
    d_ibednddir_dFM.multiply_nt(1.0, tempvec, tempvec2, 1.0);
    d_ibednddir_dFM.scale(-1.0);
    static Core::LinAlg::Matrix<9, 1> d_ibednddir_dFV(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Voigt::matrix_3x3_to_9x1(d_ibednddir_dFM, d_ibednddir_dFV);
    d_cauchyndir_dF->update(-prefac * prinv(2) * dPI(1), d_ibednddir_dFV, 1.0);
  }

  return cauchy_n_dir;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::MultiplicativeSplitDefgradElastHyper::evaluate_linearization_od(
    const Core::LinAlg::Tensor<double, 3, 3>& defgrd, const double concentration,
    Core::LinAlg::Matrix<9, 1>& d_F_dx)
{
  set_concentration_gp(concentration);
  const Core::LinAlg::Matrix<3, 3> defgrd_mat = Core::LinAlg::make_matrix_view(defgrd);

  // References to vector of inelastic contributions and inelastic deformation gradients
  auto facdefgradin = inelastic_->fac_def_grad_in();

  // number of contributions for this source
  const int num_contributions = inelastic_->num_inelastic_def_grad();

  // build inverse inelastic deformation gradient
  Core::LinAlg::Tensor<double, 3, 3> iFin{};
  Core::LinAlg::Matrix<3, 3> iFinM = Core::LinAlg::make_matrix_view(iFin);

  inelastic_->evaluate_inverse_inelastic_def_grad(&defgrd_mat, iFinM);

  Core::LinAlg::Tensor<double, 3, 3> Fe = defgrd * iFin;

  // calculate the derivative of the deformation gradient w.r.t. the inelastic deformation gradient
  static Core::LinAlg::Tensor<double, 3, 3, 3, 3> d_F_dFin =
      Core::LinAlg::reorder_axis<0, 2, 3, 1>(Core::LinAlg::dyadic(
          Fe, Core::LinAlg::get_full(Core::LinAlg::TensorGenerators::identity<double, 3, 3>)));

  static Core::LinAlg::Tensor<double, 3, 3> d_Fin_dx{};

  // check number of factors the inelastic deformation gradient consists of and choose
  // implementation accordingly
  if (num_contributions == 1)
  {
    facdefgradin[0].second->evaluate_inelastic_def_grad_derivative(
        Core::LinAlg::det(defgrd), d_Fin_dx);
  }
  else
    FOUR_C_THROW("NOT YET IMPLEMENTED");

  auto dFdc = Core::LinAlg::ddot(d_F_dFin, d_Fin_dx);

  Core::LinAlg::Voigt::matrix_3x3_to_9x1(Core::LinAlg::make_matrix_view(dFdc), d_F_dx);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::MultiplicativeSplitDefgradElastHyper::evaluate_stress_cmat_iso(
    const Mat::MultiplicativeSplitDefgradElastHyper::KinematicQuantities& kinemat_quant,
    const Mat::MultiplicativeSplitDefgradElastHyper::StressFactors& stress_fact,
    Core::LinAlg::Matrix<6, 1>& stress, Core::LinAlg::Matrix<6, 6>& cmatiso) const
{
  // extract variables from kinemat_quant
  const Core::LinAlg::Matrix<6, 1>& iCV = kinemat_quant.iCV;
  const Core::LinAlg::Matrix<6, 1>& iCinV = kinemat_quant.iCinV;
  const Core::LinAlg::Matrix<6, 1>& iCinCiCinV = kinemat_quant.iCinCiCinV;
  const Core::LinAlg::Matrix<3, 1>& gamma = stress_fact.gamma;
  const Core::LinAlg::Matrix<8, 1>& delta = stress_fact.delta;
  const double detFin = kinemat_quant.detFin;

  // clear variables
  stress.clear();
  cmatiso.clear();

  // 2nd Piola Kirchhoff stresses
  stress.update(gamma(0), iCinV, 1.0);
  stress.update(gamma(1), iCinCiCinV, 1.0);
  stress.update(gamma(2), iCV, 1.0);
  stress.scale(detFin);

  // constitutive tensor
  cmatiso.multiply_nt(delta(0), iCinV, iCinV, 1.);
  cmatiso.multiply_nt(delta(1), iCinCiCinV, iCinV, 1.);
  cmatiso.multiply_nt(delta(1), iCinV, iCinCiCinV, 1.);
  cmatiso.multiply_nt(delta(2), iCinV, iCV, 1.);
  cmatiso.multiply_nt(delta(2), iCV, iCinV, 1.);
  cmatiso.multiply_nt(delta(3), iCinCiCinV, iCinCiCinV, 1.);
  cmatiso.multiply_nt(delta(4), iCinCiCinV, iCV, 1.);
  cmatiso.multiply_nt(delta(4), iCV, iCinCiCinV, 1.);
  cmatiso.multiply_nt(delta(5), iCV, iCV, 1.);
  Core::LinAlg::FourTensorOperations::add_holzapfel_product(cmatiso, iCV, delta(6));
  Core::LinAlg::FourTensorOperations::add_holzapfel_product(cmatiso, iCinV, delta(7));
  cmatiso.scale(detFin);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::MultiplicativeSplitDefgradElastHyper::evaluate_kin_quant_elast(
    const Core::LinAlg::Matrix<3, 3>* const defgrad,
    Mat::MultiplicativeSplitDefgradElastHyper::KinematicQuantities& kinemat_quant) const
{
  // extract variables from kinemat_quant
  const Core::LinAlg::Matrix<3, 3>& iFinM = kinemat_quant.iFinM;
  Core::LinAlg::Matrix<6, 1>& iCinV = kinemat_quant.iCinV;
  Core::LinAlg::Matrix<6, 1>& iCinCiCinV = kinemat_quant.iCinCiCinV;
  Core::LinAlg::Matrix<6, 1>& iCV = kinemat_quant.iCV;
  Core::LinAlg::Matrix<3, 3>& iCinCM = kinemat_quant.iCinCM;
  Core::LinAlg::Matrix<3, 3>& iFinCeM = kinemat_quant.iFinCeM;
  Core::LinAlg::Matrix<9, 1>& CiFin9x1 = kinemat_quant.CiFin9x1;
  Core::LinAlg::Matrix<9, 1>& CiFinCe9x1 = kinemat_quant.CiFinCe9x1;
  Core::LinAlg::Matrix<9, 1>& CiFiniCe9x1 = kinemat_quant.CiFiniCe9x1;
  Core::LinAlg::Matrix<3, 1>& prinv = kinemat_quant.prinv;
  Core::LinAlg::Matrix<6, 6>& dCedC = kinemat_quant.dCedC;
  Core::LinAlg::Matrix<6, 9>& dCediFin = kinemat_quant.dCediFin;


  // inverse inelastic right Cauchy-Green
  static Core::LinAlg::Matrix<3, 3> iCinM(Core::LinAlg::Initialization::zero);
  iCinM.multiply_nt(1.0, iFinM, iFinM, 0.0);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(iCinM, iCinV);

  // inverse right Cauchy-Green
  static Core::LinAlg::Matrix<3, 3> iCM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> CM(Core::LinAlg::Initialization::zero);
  CM.multiply_tn(1.0, *defgrad, *defgrad, 0.0);
  iCM.invert(CM);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(iCM, iCV);

  // C_{in}^{-1} * C * C_{in}^{-1}
  static Core::LinAlg::Matrix<3, 3> tmp(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> iCinCiCinM;
  Mat::evaluatei_cin_ci_cin(CM, iCinM, iCinCiCinM);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(iCinCiCinM, iCinCiCinV);

  // elastic right Cauchy-Green in strain-like Voigt notation.
  static Core::LinAlg::Matrix<3, 3> CeM(Core::LinAlg::Initialization::zero);
  Mat::evaluate_ce(*defgrad, iFinM, CeM);
  static Core::LinAlg::Matrix<6, 1> CeV_strain(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Strains::matrix_to_vector(CeM, CeV_strain);

  // principal invariants of elastic right Cauchy-Green strain
  Core::LinAlg::Voigt::Strains::invariants_principal(prinv, CeV_strain);

  // C_{in}^{-1} * C
  iCinCM.multiply_nn(1.0, iCinM, CM, 0.0);

  // F_{in}^{-1} * C_e
  iFinCeM.multiply_nn(1.0, iFinM, CeM, 0.0);

  // C * F_{in}^{-1}
  static Core::LinAlg::Matrix<3, 3> CiFinM(Core::LinAlg::Initialization::zero);
  CiFinM.multiply_nn(1.0, CM, iFinM, 0.0);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(CiFinM, CiFin9x1);

  // C * F_{in}^{-1} * C_e
  static Core::LinAlg::Matrix<3, 3> CiFinCeM(Core::LinAlg::Initialization::zero);
  tmp.multiply_nn(1.0, CM, iFinM, 0.0);
  CiFinCeM.multiply_nn(1.0, tmp, CeM, 0.0);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(CiFinCeM, CiFinCe9x1);

  // C * F_{in}^{-1} * C_e^{-1}
  static Core::LinAlg::Matrix<3, 3> CiFiniCeM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> iCeM(Core::LinAlg::Initialization::zero);
  iCeM.invert(CeM);
  tmp.multiply_nn(1.0, CM, iFinM, 0.0);
  CiFiniCeM.multiply_nn(1.0, tmp, iCeM, 0.0);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(CiFiniCeM, CiFiniCe9x1);

  // derivatives of the elastic right CG
  Mat::elast_hyper_get_derivs_of_elastic_right_cg_tensor(Core::LinAlg::make_tensor(iFinM),
      Core::LinAlg::assume_symmetry(Core::LinAlg::make_tensor(CM)), dCedC, dCediFin);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::MultiplicativeSplitDefgradElastHyper::evaluate_invariant_derivatives(
    const Core::LinAlg::Matrix<3, 1>& prinv, const int gp, const int eleGID,
    Core::LinAlg::Matrix<3, 1>& dPI, Core::LinAlg::Matrix<6, 1>& ddPII) const
{
  // clear variables
  dPI.clear();
  ddPII.clear();

  // loop over map of associated potential summands
  // derivatives of strain energy function w.r.t. principal invariants
  for (const auto& p : potsumel_)  // only for isotropic components
  {
    p->add_derivatives_principal(dPI, ddPII, prinv, gp, eleGID);
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<6, 9> Mat::MultiplicativeSplitDefgradElastHyper::evaluated_sdi_fin(
    const Mat::MultiplicativeSplitDefgradElastHyper::KinematicQuantities& kinemat_quant,
    const Mat::MultiplicativeSplitDefgradElastHyper::StressFactors& stress_fact) const
{
  // declare output variables
  Core::LinAlg::Matrix<6, 9> dSdiFin{Core::LinAlg::Initialization::zero};

  // extract variables from kinemat_quant
  const Core::LinAlg::Matrix<3, 1>& gamma = stress_fact.gamma;
  const Core::LinAlg::Matrix<8, 1>& delta = stress_fact.delta;
  const Core::LinAlg::Matrix<3, 3>& iFinM = kinemat_quant.iFinM;
  const Core::LinAlg::Matrix<3, 3>& iCinCM = kinemat_quant.iCinCM;
  const Core::LinAlg::Matrix<6, 1>& iCinV = kinemat_quant.iCinV;
  const Core::LinAlg::Matrix<9, 1>& CiFin9x1 = kinemat_quant.CiFin9x1;
  const Core::LinAlg::Matrix<9, 1>& CiFinCe9x1 = kinemat_quant.CiFinCe9x1;
  const Core::LinAlg::Matrix<6, 1>& iCinCiCinV = kinemat_quant.iCinCiCinV;
  const Core::LinAlg::Matrix<9, 1>& CiFiniCe9x1 = kinemat_quant.CiFiniCe9x1;
  const Core::LinAlg::Matrix<6, 1>& iCV = kinemat_quant.iCV;
  const Core::LinAlg::Matrix<3, 3>& iFinCeM = kinemat_quant.iFinCeM;
  const double detFin = kinemat_quant.detFin;

  // clear variable
  dSdiFin.clear();

  // calculate identity tensor
  static Core::LinAlg::Matrix<3, 3> id(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; ++i) id(i, i) = 1.0;

  // derivative of second Piola Kirchhoff stresses w.r.t. inverse growth deformation gradient
  // (contribution from iFin)
  Core::LinAlg::FourTensorOperations::add_right_non_symmetric_holzapfel_product(
      dSdiFin, id, iFinM, gamma(0));
  Core::LinAlg::FourTensorOperations::add_right_non_symmetric_holzapfel_product(
      dSdiFin, iCinCM, iFinM, gamma(1));
  dSdiFin.multiply_nt(delta(0), iCinV, CiFin9x1, 1.);
  dSdiFin.multiply_nt(delta(1), iCinV, CiFinCe9x1, 1.);
  dSdiFin.multiply_nt(delta(1), iCinCiCinV, CiFin9x1, 1.);
  dSdiFin.multiply_nt(delta(2), iCinV, CiFiniCe9x1, 1.);
  dSdiFin.multiply_nt(delta(2), iCV, CiFin9x1, 1.);
  dSdiFin.multiply_nt(delta(3), iCinCiCinV, CiFinCe9x1, 1.);
  dSdiFin.multiply_nt(delta(4), iCinCiCinV, CiFiniCe9x1, 1.);
  dSdiFin.multiply_nt(delta(4), iCV, CiFinCe9x1, 1.);
  dSdiFin.multiply_nt(delta(5), iCV, CiFiniCe9x1, 1.);
  Core::LinAlg::FourTensorOperations::add_right_non_symmetric_holzapfel_product(
      dSdiFin, id, iFinCeM, gamma(1));
  dSdiFin.scale(detFin);

  // derivative of second Piola Kirchhoff stresses w.r.t. inverse growth deformation gradient
  // (contribution from det(Fin))

  // dS/d(det(Fin))
  Core::LinAlg::Matrix<6, 1> dSddetFin(Core::LinAlg::Initialization::zero);
  dSddetFin.update(gamma(0), iCinV, 0.0);
  dSddetFin.update(gamma(1), iCinCiCinV, 1.0);
  dSddetFin.update(gamma(2), iCV, 1.0);

  // d(det(Fin))/diFin
  Core::LinAlg::Matrix<9, 1> ddetFindiFinV(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> ddetFindiFinM(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> FinM(Core::LinAlg::Initialization::zero);
  FinM.invert(iFinM);
  ddetFindiFinM.update_t((-1.0) * detFin, FinM);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(ddetFindiFinM, ddetFindiFinV);

  // chain rule to get dS/d(det(Fin)) * d(det(Fin))/diFin
  dSdiFin.multiply_nt(1.0, dSddetFin, ddetFindiFinV, 1.0);

  return dSdiFin;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::MultiplicativeSplitDefgradElastHyper::evaluate_transv_iso_quantities(
    const Mat::MultiplicativeSplitDefgradElastHyper::KinematicQuantities& kinemat_quant,
    const Core::LinAlg::Matrix<3, 3>& CM, const Teuchos::ParameterList& params, const int gp,
    const int eleGID, Core::LinAlg::Matrix<6, 1>& stress, Core::LinAlg::Matrix<6, 6>& cmatiso,
    Core::LinAlg::Matrix<6, 9>& dSdiFin) const
{
  // extract variables from kinemat_quant
  const Core::LinAlg::Matrix<3, 3>& iFinM = kinemat_quant.iFinM;
  const Core::LinAlg::Matrix<6, 6>& dCedC = kinemat_quant.dCedC;
  const Core::LinAlg::Matrix<6, 9>& dCediFin = kinemat_quant.dCediFin;

  // auxiliaries
  Core::LinAlg::Matrix<3, 3> id3x3(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; ++i) id3x3(i, i) = 1.0;
  Core::LinAlg::Matrix<3, 3> temp3x3(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6, 6> temp6x6(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6, 9> temp6x9(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<9, 9> temp9x9(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6, 1> temp6x1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::FourTensor<3> tempFourTensor(true);

  // compute elastic right CG tensor in matrix form (CeM) and in Voigt strain notation (CeV)
  Core::LinAlg::Matrix<3, 3> CeM(Core::LinAlg::Initialization::zero);
  temp3x3.multiply_tn(1.0, iFinM, CM, 0.0);
  CeM.multiply_nn(1.0, temp3x3, iFinM, 0.0);
  Core::LinAlg::SymmetricTensor<double, 3, 3> CeV =
      Core::LinAlg::assume_symmetry(Core::LinAlg::make_tensor(CeM));

  // initialize elastic 2nd PK stress elast_stress and elasticity tensor elast_stiffness (associated
  // to the transversely isotropic hyperelastic component)
  Core::LinAlg::SymmetricTensor<double, 3, 3> elast_stress{};  // Voigt stress notation of SeM
  Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> elast_stiffness{};  // Voigt stress-strain

  // loop through all transversely isotropic parts, and compute the total elastic stress and elastic
  // stiffness
  for (const auto& p : potsumel_transviso_)
  {
    p->add_stress_aniso_principal(CeV, elast_stiffness, elast_stress, params, gp, eleGID);
  }

  Core::LinAlg::Matrix<6, 6> elast_stiffness_6x6(
      Core::LinAlg::make_stress_like_voigt_view(elast_stiffness));

  // convert stiffness to stress-strain notation
  temp6x6 = elast_stiffness_6x6;
  elast_stiffness_6x6 = Core::LinAlg::Voigt::modify_voigt_representation(temp6x6, 1.0, 2.0);

  // calculate stress from elast_stress
  const Core::LinAlg::Matrix<3, 3> SeM =
      Core::LinAlg::make_matrix(Core::LinAlg::get_full(elast_stress));
  Core::LinAlg::Matrix<3, 3> SM(Core::LinAlg::Initialization::zero);
  temp3x3.multiply_nn(1.0, iFinM, SeM, 0.0);
  SM.multiply_nt(1.0, temp3x3, iFinM, 0.0);
  SM.scale(iFinM.determinant());
  Core::LinAlg::Voigt::VoigtUtils<Core::LinAlg::Voigt::NotationType::stress>::matrix_to_vector(
      SM, temp6x1);
  stress.update(1.0, temp6x1, 1.0);

  // F^{-1}_{in} S_{e} and S_{e} F^{-T}_{in}
  Core::LinAlg::Matrix<3, 3> iFinSeM(Core::LinAlg::Initialization::zero);
  iFinSeM.multiply_nn(1.0, iFinM, SeM, 0.0);
  Core::LinAlg::Matrix<3, 3> SeiFinTM(Core::LinAlg::Initialization::zero);
  SeiFinTM.multiply_nt(1.0, id3x3, iFinSeM, 0.0);

  // \frac{\partial S_{e}}{\partial F^{-1}_{in}}
  Core::LinAlg::Matrix<6, 9> dSediFin(Core::LinAlg::Initialization::zero);  // Voigt stress-form
  dSediFin.multiply_nn(1.0, elast_stiffness_6x6, dCediFin, 0.0);
  Core::LinAlg::FourTensor<3> dSediFin_FourTensor(true);
  Core::LinAlg::Voigt::setup_four_tensor_from_6x9_voigt_matrix(dSediFin_FourTensor, dSediFin);

  // \frac{\partial S_{e}}{\partial C}
  Core::LinAlg::Matrix<6, 6> dSedC(Core::LinAlg::Initialization::zero);  // Voigt stress-stress-form
  dSedC.multiply_nn(1.0, elast_stiffness_6x6, dCedC, 0.0);
  Core::LinAlg::FourTensor<3> dSedC_FourTensor(true);
  Core::LinAlg::Voigt::setup_four_tensor_from_6x6_voigt_matrix(dSedC_FourTensor, dSedC);

  // F_{in}^{-1} \frac{\partial S_{e}}{\partial F^{-1}_{in}}
  Core::LinAlg::FourTensor<3> iFindSediFin_FourTensor(true);
  Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
      iFindSediFin_FourTensor, iFinM, dSediFin_FourTensor, true);
  // (F_{in}^{-1} \frac{\partial S_{e}}{\partial F^{-1}_{in}})^{T_{12}}
  Core::LinAlg::FourTensor<3> iFindSediFin_FourTensor_T12(true);
  iFindSediFin_FourTensor_T12.transpose_12(iFindSediFin_FourTensor);
  // [F_{in}^{-1} (F_{in}^{-1} \frac{\partial S_{e}}{\partial F^{-1}_{in}})^T_{12}]
  Core::LinAlg::FourTensor<3> iFin_iFindSediFin_T12_FourTensor(true);
  Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
      iFin_iFindSediFin_T12_FourTensor, iFinM, iFindSediFin_FourTensor_T12, true);
  Core::LinAlg::Matrix<6, 9> iFin_iFindSediFin_T12(
      Core::LinAlg::Initialization::zero);  // Voigt stress-form
  Core::LinAlg::Voigt::setup_6x9_voigt_matrix_from_four_tensor(iFin_iFindSediFin_T12,
      iFin_iFindSediFin_T12_FourTensor);  // NOTE: we do not transpose ijkl->jikl anymore, this
                                          // should be symmetric!

  // [F_{in}^{-1} (F_{in}^{-1} \frac{\partial S_{e}}{\partial C})^T_{12}  ]^T_{12}
  Core::LinAlg::FourTensor<3> iFindSedC_FourTensor(true);
  Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
      iFindSedC_FourTensor, iFinM, dSedC_FourTensor, true);
  Core::LinAlg::FourTensor<3> iFindSedC_FourTensor_T12(true);
  iFindSedC_FourTensor_T12.transpose_12(iFindSedC_FourTensor);
  Core::LinAlg::FourTensor<3> iFin_iFindSedC_T12_FourTensor(true);
  Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
      iFin_iFindSedC_T12_FourTensor, iFinM, iFindSedC_FourTensor_T12, true);
  Core::LinAlg::Matrix<6, 6> iFin_iFindSedC_T12(
      Core::LinAlg::Initialization::zero);  // Voigt stress-stress-form
  Core::LinAlg::Voigt::setup_6x6_voigt_matrix_from_four_tensor(iFin_iFindSedC_T12,
      iFin_iFindSedC_T12_FourTensor);  // NOTE: we do not transpose ijkl->jikl anymore, this should
                                       // be symmetric!

  // \frac{\partial S}{\partial C}
  cmatiso.update(1.0, iFin_iFindSedC_T12, 1.0);

  // \frac{\partial S}{\partial F^{-1}_{in}}
  dSdiFin.update(1.0, iFin_iFindSediFin_T12, 1.0);
  temp9x9.clear();
  Core::LinAlg::FourTensorOperations::add_non_symmetric_product(1.0, id3x3, SeiFinTM, temp9x9);
  Core::LinAlg::FourTensorOperations::add_adbc_tensor_product(1.0, iFinSeM, id3x3, temp9x9);
  Core::LinAlg::Voigt::setup_four_tensor_from_9x9_voigt_matrix(tempFourTensor, temp9x9);
  Core::LinAlg::Voigt::setup_6x9_voigt_matrix_from_four_tensor(temp6x9, tempFourTensor);
  dSdiFin.update(1.0, temp6x9, 1.0);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<6, 6> Mat::MultiplicativeSplitDefgradElastHyper::evaluate_additional_cmat(
    const Core::LinAlg::Matrix<3, 3>* const defgrad, const Core::LinAlg::Matrix<6, 1>& iCV,
    const Core::LinAlg::Matrix<6, 9>& dSdiFin)
{
  // declare output variable
  Core::LinAlg::Matrix<6, 6> cmatadd;

  const auto& facdefgradin = inelastic_->fac_def_grad_in();
  const auto& iFinjM = inelastic_->geti_finj();
  const int num_contributions = inelastic_->num_inelastic_def_grad();

  // check amount of factors the inelastic deformation gradient consists of and choose
  // implementation accordingly
  if (num_contributions == 1)
  {
    Core::LinAlg::Matrix<3, 3> id3x3(Core::LinAlg::Initialization::zero);
    for (int i = 0; i < 3; ++i) id3x3(i, i) = 1.0;
    facdefgradin[0].second->evaluate_additional_cmat(
        defgrad, id3x3, iFinjM[0].second, iCV, dSdiFin, cmatadd);
  }
  else if (num_contributions > 1)
  {
    // static variables
    // dSdiFinj = dSdiFin : diFindiFinj
    // diFindiFinj = \Pi_(k=num_contributions_part)^(j+1) iFin_k : dFinjdFinj : \Pi_(k=j-1)^(0). The
    // double contraction and derivative in index notation for three contributions: diFindiFinj_abcd
    // = iFin_(j-1)_ae \delta_ec \delta_df iFin_(j+1)_fb = iFin_(j-1)_ac iFin_(j+1)_db. This is
    // performed by nonsymmetric product \Pi_(k=num_contributions_part)^(j+1) iFin_k (x)
    // \Pi_(k=j-1)^(0) iFin_k, where (x) denots the nonsymmetric product and \Pi is the
    // multiplication operator.
    static Core::LinAlg::Matrix<6, 9> dSdiFinj(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<9, 9> diFindiFinj(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<3, 3> id(Core::LinAlg::Initialization::zero);
    for (int i = 0; i < 3; ++i) id(i, i) = 1.0;

    // product of all iFinj, except for the one, that is currently evaluated. In case of inner
    // (neither first or last) we have two products
    static Core::LinAlg::Matrix<3, 3> producta(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<3, 3> productb(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<3, 3> producta_temp(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<3, 3> productb_temp(Core::LinAlg::Initialization::zero);

    for (int i = 0; i < num_contributions; ++i)
    {
      // clear static variable
      diFindiFinj.clear();

      // multiply all inelastic deformation gradients except for range between first and current
      producta = id;
      for (int j = num_contributions - 1; j > i; --j)
      {
        producta_temp.multiply(1.0, producta, iFinjM[j].second, 0.0);
        producta.update(1.0, producta_temp, 0.0);
      }

      // multiply all inelastic deformation gradients except for range between last and current
      productb = id;
      if (i > 0)
      {
        for (int j = i - 1; j >= 0; --j)
        {
          productb_temp.multiply(1.0, productb, iFinjM[j].second, 0.0);
          productb.update(1.0, productb_temp, 0.0);
        }
      }

      // evaluate additional contribution to C by applying chain rule
      Core::LinAlg::FourTensorOperations::add_non_symmetric_product(
          1.0, producta, productb, diFindiFinj);
      dSdiFinj.multiply(1.0, dSdiFin, diFindiFinj, 0.0);
      facdefgradin[i].second->evaluate_additional_cmat(
          defgrad, productb, iFinjM[i].second, iCV, dSdiFinj, cmatadd);
    }
  }
  else
    FOUR_C_THROW("You should not be here");

  return cmatadd;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::MultiplicativeSplitDefgradElastHyper::setup(const int numgp,
    const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  // Read anisotropy
  anisotropy_->set_number_of_gauss_points(numgp);
  anisotropy_->read_anisotropy_from_element(fibers, coord_system);

  // elastic materials
  for (const auto& summand : potsumel_) summand->setup(numgp, fibers, coord_system);
  for (const std::shared_ptr<Mat::Elastic::CoupTransverselyIsotropic>& summand :
      potsumel_transviso_)
    summand->setup(numgp, fibers, coord_system);

  // setup inelastic materials
  inelastic_->setup(numgp, fibers, coord_system);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::MultiplicativeSplitDefgradElastHyper::update()
{
  // loop map of associated potential summands
  for (const auto& summand : potsumel_) summand->update();
  for (const std::shared_ptr<Mat::Elastic::CoupTransverselyIsotropic>& summand :
      potsumel_transviso_)
    summand->update();

  // update inelastic materials
  inelastic_->update();
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::MultiplicativeSplitDefgradElastHyper::evaluate_od_stiff_mat(PAR::InelasticSource source,
    const Core::LinAlg::Matrix<3, 3>* const defgrad, const Core::LinAlg::Matrix<6, 9>& dSdiFin,
    Core::LinAlg::Matrix<6, 1>& dstressdx)
{
  // clear variable
  dstressdx.clear();

  // References to vector of inelastic contributions and inelastic deformation gradients
  const auto& facdefgradin = inelastic_->fac_def_grad_in();
  const auto& iFinjM = inelastic_->geti_finj();

  // number of contributions for this source
  const int num_contributions = inelastic_->num_inelastic_def_grad();

  // check number of factors the inelastic deformation gradient consists of and choose
  // implementation accordingly
  if (num_contributions == 1)
  {
    facdefgradin[0].second->evaluate_od_stiff_mat(defgrad, iFinjM[0].second, dSdiFin, dstressdx);
  }
  else if (num_contributions > 1)
  {
    // static variables
    // dSdiFinj = dSdiFin : diFindiFinj
    // diFindiFinj = \Pi_(k=num_contributions_part)^(j+1) iFin_k : dFinjdFinj : \Pi_(k=j-1)^(0). The
    // double contraction and derivative in index notation for three contributions: diFindiFinj_abcd
    // = iFin_(j-1)_ae \delta_ec \delta_df iFin_(j+1)_fb = iFin_(j-1)_ac iFin_(j+1)_db. This is
    // performed by nonsymmetric product \Pi_(k=num_contributions_part)^(j+1) iFin_k (x)
    // \Pi_(k=j-1)^(0) iFin_k, where (x) denots the nonsymmetric product and \Pi is the
    // multiplication operator.
    static Core::LinAlg::Matrix<6, 9> dSdiFinj(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<9, 9> diFindiFinj(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<3, 3> id(Core::LinAlg::Initialization::zero);
    for (int i = 0; i < 3; ++i) id(i, i) = 1.0;

    // product of all iFinj, except for the one, that is currently evaluated. In case of inner
    // (neither first or last) we have two products
    static Core::LinAlg::Matrix<3, 3> producta(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<3, 3> productb(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<3, 3> producta_temp(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<3, 3> productb_temp(Core::LinAlg::Initialization::zero);

    for (int i = 0; i < num_contributions; ++i)
    {
      // only if the contribution is from this source, the derivative is non-zero
      if (facdefgradin[i].first == source)
      {
        // clear static variable
        diFindiFinj.clear();

        // multiply all inelastic deformation gradients except for range between first and current
        // in reverse order
        producta = id;
        for (int j = num_contributions - 1; j > i; --j)
        {
          producta_temp.multiply(1.0, producta, iFinjM[j].second, 0.0);
          producta.update(1.0, producta_temp, 0.0);
        }

        // multiply all inelastic deformation gradients except for range between last and current
        productb = id;
        if (i > 0)
        {
          for (int j = i - 1; j >= 0; --j)
          {
            productb_temp.multiply(1.0, productb, iFinjM[j].second, 0.0);
            productb.update(1.0, productb_temp, 0.0);
          }
        }

        // evaluate additional contribution to OD block by applying chain rule
        Core::LinAlg::FourTensorOperations::add_non_symmetric_product(
            1.0, producta, productb, diFindiFinj);
        dSdiFinj.multiply(1.0, dSdiFin, diFindiFinj, 0.0);
        facdefgradin[i].second->evaluate_od_stiff_mat(
            defgrad, iFinjM[i].second, dSdiFinj, dstressdx);
      }
    }
  }
  else
    FOUR_C_THROW("You should not be here");
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::MultiplicativeSplitDefgradElastHyper::pre_evaluate(const Teuchos::ParameterList& params,
    const EvaluationContext& context, const int gp, const int eleGID) const
{
  // loop over all inelastic contributions
  for (int p = 0; p < inelastic_->num_inelastic_def_grad(); ++p)
    inelastic_->fac_def_grad_in()[p].second->pre_evaluate(params, context, gp, eleGID);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::MultiplicativeSplitDefgradElastHyper::set_concentration_gp(const double concentration)
{
  for (int p = 0; p < inelastic_->num_inelastic_def_grad(); ++p)
    inelastic_->fac_def_grad_in()[p].second->set_concentration_gp(concentration);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticFactorsHandler::assign_to_source(
    Mat::PAR::MultiplicativeSplitDefgradElastHyper* params)
{
  facdefgradin_.clear();
  i_finj_.clear();

  // get inelastic deformation gradient factors and assign them to their source
  for (int inelastic_matnum : params->inel_defgradfacids_)
  {
    auto inelastic_factor = Mat::InelasticDefgradFactors::factory(inelastic_matnum);
    if (inelastic_factor == nullptr) FOUR_C_THROW("Failed to allocate!");
    std::pair<PAR::InelasticSource, std::shared_ptr<Mat::InelasticDefgradFactors>> temppair(
        inelastic_factor->get_inelastic_source(), inelastic_factor);
    facdefgradin_.push_back(temppair);
  }

  i_finj_.resize(facdefgradin_.size());

  // safety checks
  // get the scatra structure control parameter list
  const auto& ssicontrol = Global::Problem::instance()->ssi_control_params();
  if (Teuchos::getIntegralValue<SSI::SolutionSchemeOverFields>(ssicontrol, "COUPALGO") ==
      SSI::SolutionSchemeOverFields::ssi_Monolithic)
  {
    for (const auto& inelasitc_factor : facdefgradin_)
    {
      const auto materialtype = inelasitc_factor.second->material_type();
      if ((materialtype != Core::Materials::mfi_lin_scalar_aniso) and
          (materialtype != Core::Materials::mfi_lin_scalar_iso) and
          (materialtype != Core::Materials::mfi_lin_temp_iso) and
          (materialtype != Core::Materials::mfi_no_growth) and
          (materialtype != Core::Materials::mfi_time_funct) and
          (materialtype != Core::Materials::mfi_poly_intercal_frac_aniso) and
          (materialtype != Core::Materials::mfi_poly_intercal_frac_iso) and
          (materialtype != Core::Materials::mfi_transv_isotrop_elast_viscoplast))
      {
        FOUR_C_THROW(
            "When you use the 'COUPALGO' 'ssi_Monolithic' from the 'SSI CONTROL' section, you need "
            "to use one of the materials derived from 'Mat::InelasticDefgradFactors'!"
            " If you want to use a different material, feel free to implement it! ;-)");
      }
    }
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticFactorsHandler::evaluate_inverse_inelastic_def_grad(
    const Core::LinAlg::Matrix<3, 3>* const defgrad, Core::LinAlg::Matrix<3, 3>& iFinM)
{
  // temporary variables
  static Core::LinAlg::Matrix<3, 3> iFinp(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> iFin_init_store(Core::LinAlg::Initialization::zero);

  // clear variables
  iFinM.clear();
  iFin_init_store.clear();

  for (int i = 0; i < 3; ++i) iFin_init_store(i, i) = 1.0;

  for (int i = 0; i < num_inelastic_def_grad(); ++i)
  {
    // clear tmp variable
    iFinp.clear();

    // calculate inelastic deformation gradient and its inverse
    facdefgradin_[i].second->evaluate_inverse_inelastic_def_grad(defgrad, iFin_init_store, iFinp);

    // store inelastic deformation gradient of p-th inelastic contribution
    i_finj_[i].second = iFinp;

    // update inverse inelastic deformation gradient
    iFinM.multiply(iFin_init_store, iFinp);

    // store result for next evaluation
    iFin_init_store.update(1.0, iFinM, 0.0);
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticFactorsHandler::update()
{
  // loop over all inelastic contributions
  for (const auto& [_, inelastic_defgrad_factor] : facdefgradin_)
  {
    inelastic_defgrad_factor->update();
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticFactorsHandler::setup(const int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  // loop over all inelastic contributions
  for (int i = 0; i < num_inelastic_def_grad(); ++i)
  {
    facdefgradin_[i].second->setup(numgp, fibers, coord_system);
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticFactorsHandler::pack_inelastic(Core::Communication::PackBuffer& data) const
{
  // loop over all inelastic contributions
  for (int i = 0; i < num_inelastic_def_grad(); ++i)
  {
    facdefgradin_[i].second->pack_inelastic(data);
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/

void Mat::InelasticFactorsHandler::unpack_inelastic(Core::Communication::UnpackBuffer& buffer)
{
  // loop over all inelastic contributions
  for (int i = 0; i < num_inelastic_def_grad(); ++i)
  {
    facdefgradin_[i].second->unpack_inelastic(buffer);
  }
}


Mat::MultiplicativeSplitDefgradElastHyper::KinematicQuantities
Mat::MultiplicativeSplitDefgradElastHyper::evaluate_kinematic_quantities(
    const Mat::MultiplicativeSplitDefgradElastHyper& splitdefgrd,
    Mat::InelasticFactorsHandler& inelastic_factors_handler,
    const Core::LinAlg::Matrix<3, 3>& defgrad, const int gp, const int eleGID)
{
  Mat::MultiplicativeSplitDefgradElastHyper::KinematicQuantities quantities{};

  // build inverse inelastic deformation gradient
  inelastic_factors_handler.evaluate_inverse_inelastic_def_grad(&defgrad, quantities.iFinM);

  // determinant of inelastic deformation gradient
  quantities.detFin = 1.0 / quantities.iFinM.determinant();

  splitdefgrd.evaluate_kin_quant_elast(&defgrad, quantities);

  // derivatives of principle invariants
  splitdefgrd.evaluate_invariant_derivatives(quantities.prinv, gp, eleGID, quantities.dPIe,
      quantities.ddPIIe);  // NOTE: we exclude the transversely isotropic hyperelastic
                           // components in this function --> we deal with them separately

  return quantities;
}


FOUR_C_NAMESPACE_CLOSE
