// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elasthyper.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_mat_elasthyper_service.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_utils_enum.hpp"

#include <iostream>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PAR::ElastHyper::ElastHyper(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      nummat_(matdata.parameters.get<int>("NUMMAT")),
      matids_(matdata.parameters.get<std::vector<int>>("MATIDS")),
      density_(matdata.parameters.get<double>("DENS")),
      polyconvex_(matdata.parameters.get<int>("POLYCONVEX"))

{
  // check if sizes fit
  if (nummat_ != (int)matids_.size())
    FOUR_C_THROW("number of materials {} does not fit to size of material vector {}", nummat_,
        matids_.size());

  // output, that polyconvexity is checked
  if (polyconvex_ != 0) std::cout << "Polyconvexity of your simulation is checked." << std::endl;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::ElastHyper::create_material()
{
  return std::make_shared<Mat::ElastHyper>(this);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::ElastHyperType Mat::ElastHyperType::instance_;


Core::Communication::ParObject* Mat::ElastHyperType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* elhy = new Mat::ElastHyper();
  elhy->unpack(buffer);

  return elhy;
}


/*----------------------------------------------------------------------*
 |  initialise static arrays                                 bborn 08/09|
 *----------------------------------------------------------------------*/



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::ElastHyper::ElastHyper() : summandProperties_(), params_(nullptr), potsum_(0), anisotropy_() {}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::ElastHyper::ElastHyper(Mat::PAR::ElastHyper* params)
    : summandProperties_(), params_(params), potsum_(0), anisotropy_()
{
  // make sure the referenced materials in material list have quick access parameters
  std::vector<int>::const_iterator m;
  for (m = params_->matids_.begin(); m != params_->matids_.end(); ++m)
  {
    const int matid = *m;
    std::shared_ptr<Mat::Elastic::Summand> sum = Mat::Elastic::Summand::factory(matid);
    if (sum == nullptr) FOUR_C_THROW("Failed to allocate");
    potsum_.push_back(sum);
    sum->register_anisotropy_extensions(anisotropy_);
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ElastHyper::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);
  summandProperties_.pack(data);

  anisotropy_.pack_anisotropy(data);

  Core::Communication::PotentiallyUnusedBufferScope potsum_scope(data);

  if (params_ != nullptr)  // summands are not accessible in postprocessing mode
  {
    // loop map of associated potential summands
    for (const auto& p : potsum_)
    {
      p->pack_summand(data);
    }
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ElastHyper::unpack(Core::Communication::UnpackBuffer& buffer)
{
  // make sure we have a pristine material
  params_ = nullptr;
  potsum_.clear();



  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid and recover params_
  int matid;
  extract_from_pack(buffer, matid);
  if (Global::Problem::instance()->materials() != nullptr)
  {
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
        params_ = dynamic_cast<Mat::PAR::ElastHyper*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
  }

  summandProperties_.unpack(buffer);

  // Pack anisotropy
  anisotropy_.unpack_anisotropy(buffer);

  Core::Communication::PotentiallyUnusedBufferScope potsum_scope(buffer);

  if (params_ != nullptr)  // summands are not accessible in postprocessing mode
  {
    // make sure the referenced materials in material list have quick access parameters
    std::vector<int>::const_iterator m;
    for (m = params_->matids_.begin(); m != params_->matids_.end(); ++m)
    {
      const int summand_matid = *m;
      std::shared_ptr<Mat::Elastic::Summand> sum = Mat::Elastic::Summand::factory(summand_matid);
      if (sum == nullptr) FOUR_C_THROW("Failed to allocate");
      potsum_.push_back(sum);
    }

    // loop map of associated potential summands
    for (auto& p : potsum_)
    {
      p->unpack_summand(buffer);
      p->register_anisotropy_extensions(anisotropy_);
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int Mat::ElastHyper::mat_id(const unsigned index) const
{
  if ((int)index >= params_->nummat_)
  {
    FOUR_C_THROW("Index too large");
  }

  return params_->matids_.at(index);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Mat::ElastHyper::shear_mod(int ele_gid) const
{
  // principal coefficients
  bool haveshearmod = false;
  double shearmod = 0.0;
  {
    // loop map of associated potential summands
    for (const auto& p : potsum_)
    {
      p->add_shear_mod(haveshearmod, shearmod, ele_gid);
    }
  }
  if (!haveshearmod)
  {
    FOUR_C_THROW("Cannot provide shear modulus equivalent");
  }
  return shearmod;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Mat::ElastHyper::get_young()
{
  double young;
  double shear;
  double bulk;
  young = shear = bulk = 0.;
  for (auto& p : potsum_) p->add_youngs_mod(young, shear, bulk);

  if (bulk != 0. || shear != 0.) young += 9. * bulk * shear / (3. * bulk + shear);

  return young;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ElastHyper::setup_aaa(const Teuchos::ParameterList& params, const int eleGID)
{
  // loop map of associated potential summands
  for (auto& p : potsum_)
  {
    p->setup_aaa(params, eleGID);
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ElastHyper::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  // Read anisotropy
  anisotropy_.set_number_of_gauss_points(numgp);
  anisotropy_.read_anisotropy_from_element(fibers, coord_system);

  // Setup summands
  for (auto& p : potsum_)
  {
    p->setup(numgp, fibers, coord_system);
  }
  summandProperties_.clear();
  elast_hyper_properties(potsum_, summandProperties_);

  if (summandProperties_.viscoGeneral)
  {
    FOUR_C_THROW(
        "Never use viscoelastic-materials in Elasthyper-Toolbox. Use Viscoelasthyper-Toolbox "
        "instead.");
  }
}

void Mat::ElastHyper::post_setup(const Teuchos::ParameterList& params, const int eleGID)
{
  anisotropy_.read_anisotropy_from_parameter_list(params);

  // Forward post_setup call to all summands
  for (auto& p : potsum_)
  {
    p->post_setup(params);
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ElastHyper::update()
{
  // loop map of associated potential summands
  for (auto& p : potsum_)
  {
    p->update();
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ElastHyper::get_fiber_vecs(std::vector<Core::LinAlg::Tensor<double, 3>>& fibervecs) const
{
  if (summandProperties_.anisoprinc || summandProperties_.anisomod)
  {
    for (auto& p : potsum_)
    {
      p->get_fiber_vecs(fibervecs);
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ElastHyper::evaluate_fiber_vecs(const double newgamma,
    const Core::LinAlg::Tensor<double, 3, 3>& locsys,
    const Core::LinAlg::Tensor<double, 3, 3>& defgrd)
{
  if (summandProperties_.anisoprinc || summandProperties_.anisomod)
  {
    for (auto& p : potsum_)
    {
      p->set_fiber_vecs(newgamma, locsys, defgrd);
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Mat::ElastHyper::strain_energy(const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const EvaluationContext& context, const int gp, const int eleGID) const
{
  Core::LinAlg::SymmetricTensor<double, 3, 3> C_strain{};
  static Core::LinAlg::Matrix<3, 1> prinv(Core::LinAlg::Initialization::zero);
  prinv.clear();
  static Core::LinAlg::Matrix<3, 1> modinv(Core::LinAlg::Initialization::zero);
  modinv.clear();

  evaluate_right_cauchy_green_strain_like_voigt(glstrain, C_strain);
  Core::LinAlg::Voigt::Stresses::invariants_principal(
      prinv, Core::LinAlg::make_stress_like_voigt_view(C_strain));
  invariants_modified(modinv, prinv);

  double psi = 0.0;
  // loop map of associated potential summands
  for (const auto& p : potsum_)
  {
    p->add_strain_energy(psi, prinv, modinv, glstrain, gp, eleGID);
  }

  return psi;
}



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ElastHyper::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  bool checkpolyconvexity = (params_ != nullptr and params_->polyconvex_ != 0);

  elast_hyper_evaluate(*defgrad, glstrain, params, stress, cmat, gp, eleGID, potsum_,
      summandProperties_, checkpolyconvexity);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ElastHyper::evaluate_cauchy_derivs(const Core::LinAlg::Matrix<3, 1>& prinv, const int gp,
    int eleGID, Core::LinAlg::Matrix<3, 1>& dPI, Core::LinAlg::Matrix<6, 1>& ddPII,
    Core::LinAlg::Matrix<10, 1>& dddPIII, const double* temp)
{
  for (auto& i : potsum_)
  {
    if (summandProperties_.isoprinc)
    {
      i->add_derivatives_principal(dPI, ddPII, prinv, gp, eleGID);
      i->add_third_derivatives_principal_iso(dddPIII, prinv, gp, eleGID);
    }
    if (summandProperties_.isomod || summandProperties_.anisomod || summandProperties_.anisoprinc)
      FOUR_C_THROW("not implemented for this form of strain energy function");
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Mat::ElastHyper::evaluate_cauchy_n_dir_and_derivatives(
    const Core::LinAlg::Tensor<double, 3, 3>& defgrd, const Core::LinAlg::Tensor<double, 3>& n,
    const Core::LinAlg::Tensor<double, 3>& dir, Core::LinAlg::Matrix<3, 1>* d_cauchyndir_dn,
    Core::LinAlg::Matrix<3, 1>* d_cauchyndir_ddir, Core::LinAlg::Matrix<9, 1>* d_cauchyndir_dF,
    Core::LinAlg::Matrix<9, 9>* d2_cauchyndir_dF2, Core::LinAlg::Matrix<9, 3>* d2_cauchyndir_dF_dn,
    Core::LinAlg::Matrix<9, 3>* d2_cauchyndir_dF_ddir, const EvaluationContext& context, int eleGID,
    const double* concentration, const double* temp, double* d_cauchyndir_dT,
    Core::LinAlg::Matrix<9, 1>* d2_cauchyndir_dF_dT)
{
  double cauchy_n_dir = 0.0;

  const Core::LinAlg::Matrix<3, 3> defgrd_mat = Core::LinAlg::make_matrix_view(defgrd);
  const Core::LinAlg::Matrix<3, 1> n_mat = Core::LinAlg::make_matrix_view<3, 1>(n);
  const Core::LinAlg::Matrix<3, 1> dir_mat = Core::LinAlg::make_matrix_view<3, 1>(dir);


  static Core::LinAlg::Matrix<3, 3> b(Core::LinAlg::Initialization::zero);
  b.multiply_nt(1.0, defgrd_mat, defgrd_mat, 0.0);
  static Core::LinAlg::Matrix<3, 1> bdn(Core::LinAlg::Initialization::zero);
  bdn.multiply(1.0, b, n_mat, 0.0);
  static Core::LinAlg::Matrix<3, 1> bddir(Core::LinAlg::Initialization::zero);
  bddir.multiply(1.0, b, dir_mat, 0.0);
  const double bdnddir = bdn.dot(dir_mat);

  static Core::LinAlg::Matrix<3, 3> ib(Core::LinAlg::Initialization::zero);
  ib.invert(b);
  static Core::LinAlg::Matrix<3, 1> ibdn(Core::LinAlg::Initialization::zero);
  ibdn.multiply(1.0, ib, n_mat, 0.0);
  static Core::LinAlg::Matrix<3, 1> ibddir(Core::LinAlg::Initialization::zero);
  ibddir.multiply(1.0, ib, dir_mat, 0.0);
  const double ibdnddir = ibdn.dot(dir_mat);
  const double nddir = n * dir;

  static Core::LinAlg::Matrix<6, 1> bV_strain(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Strains::matrix_to_vector(b, bV_strain);
  static Core::LinAlg::Matrix<3, 1> prinv(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Strains::invariants_principal(prinv, bV_strain);

  static Core::LinAlg::Matrix<3, 1> dPI(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<6, 1> ddPII(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<10, 1> dddPIII(Core::LinAlg::Initialization::zero);
  dPI.clear();
  ddPII.clear();
  dddPIII.clear();
  constexpr int dummy_gp_id = -1;
  evaluate_cauchy_derivs(prinv, dummy_gp_id, eleGID, dPI, ddPII, dddPIII, temp);

  const double prefac = 2.0 / std::sqrt(prinv(2));

  cauchy_n_dir = prefac * (prinv(1) * dPI(1) * nddir + prinv(2) * dPI(2) * nddir +
                              dPI(0) * bdnddir - prinv(2) * dPI(1) * ibdnddir);

  if (d_cauchyndir_dn != nullptr)
  {
    d_cauchyndir_dn->update(prinv(1) * dPI(1) + prinv(2) * dPI(2), dir_mat, 0.0);
    d_cauchyndir_dn->update(dPI(0), bddir, 1.0);
    d_cauchyndir_dn->update(-prinv(2) * dPI(1), ibddir, 1.0);
    d_cauchyndir_dn->scale(prefac);
  }

  if (d_cauchyndir_ddir != nullptr)
  {
    d_cauchyndir_ddir->update(prinv(1) * dPI(1) + prinv(2) * dPI(2), n_mat, 0.0);
    d_cauchyndir_ddir->update(dPI(0), bdn, 1.0);
    d_cauchyndir_ddir->update(-prinv(2) * dPI(1), ibdn, 1.0);
    d_cauchyndir_ddir->scale(prefac);
  }

  // calculate stuff that is needed for evaluations of derivatives w.r.t. F
  static Core::LinAlg::Matrix<9, 1> FV(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(defgrd_mat, FV);
  static Core::LinAlg::Matrix<3, 3> iF(Core::LinAlg::Initialization::zero);
  iF.invert(defgrd_mat);
  static Core::LinAlg::Matrix<3, 3> iFT(Core::LinAlg::Initialization::zero);
  iFT.update_t(iF);
  static Core::LinAlg::Matrix<9, 1> iFTV(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(iFT, iFTV);

  // calculation of dI_i/dF (derivatives of invariants of b w.r.t. deformation gradient)
  static Core::LinAlg::Matrix<3, 3> bdF(Core::LinAlg::Initialization::zero);
  bdF.multiply(1.0, b, defgrd_mat, 0.0);
  static Core::LinAlg::Matrix<9, 1> bdFV(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(bdF, bdFV);
  static Core::LinAlg::Matrix<3, 3> ibdF(Core::LinAlg::Initialization::zero);
  ibdF.multiply(1.0, ib, defgrd_mat, 0.0);
  static Core::LinAlg::Matrix<9, 1> ibdFV(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(ibdF, ibdFV);
  static Core::LinAlg::Matrix<9, 1> d_I1_dF(Core::LinAlg::Initialization::zero);
  d_I1_dF.update(2.0, FV, 0.0);
  static Core::LinAlg::Matrix<9, 1> d_I2_dF(Core::LinAlg::Initialization::zero);
  d_I2_dF.update(prinv(0), FV, 0.0);
  d_I2_dF.update(-1.0, bdFV, 1.0);
  d_I2_dF.scale(2.0);
  static Core::LinAlg::Matrix<9, 1> d_I3_dF(Core::LinAlg::Initialization::zero);
  d_I3_dF.update(2.0 * prinv(2), ibdFV, 0.0);

  // calculate d(b \cdot n \cdot t)/dF
  static Core::LinAlg::Matrix<3, 1> tempvec3x1(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<1, 3> tempvec1x3(Core::LinAlg::Initialization::zero);
  tempvec1x3.multiply_tn(1.0, dir_mat, defgrd_mat, 0.0);
  static Core::LinAlg::Matrix<3, 3> d_bdnddir_dF(Core::LinAlg::Initialization::zero);
  d_bdnddir_dF.multiply_nn(1.0, n_mat, tempvec1x3, 0.0);
  tempvec1x3.multiply_tn(1.0, n_mat, defgrd_mat, 0.0);
  d_bdnddir_dF.multiply_nn(1.0, dir_mat, tempvec1x3, 1.0);
  static Core::LinAlg::Matrix<9, 1> d_bdnddir_dFV(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(d_bdnddir_dF, d_bdnddir_dFV);

  // calculate d(b^{-1} \cdot n \cdot t)/dF
  static Core::LinAlg::Matrix<1, 3> dirdibdF(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<1, 3> ndibdF(Core::LinAlg::Initialization::zero);
  dirdibdF.multiply_tn(1.0, dir_mat, ibdF, 0.0);
  static Core::LinAlg::Matrix<3, 3> d_ibdnddir_dF(Core::LinAlg::Initialization::zero);
  d_ibdnddir_dF.multiply_nn(1.0, ibdn, dirdibdF, 0.0);
  ndibdF.multiply_tn(1.0, n_mat, ibdF, 0.0);
  d_ibdnddir_dF.multiply_nn(1.0, ibddir, ndibdF, 1.0);
  d_ibdnddir_dF.scale(-1.0);
  static Core::LinAlg::Matrix<9, 1> d_ibdnddir_dFV(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(d_ibdnddir_dF, d_ibdnddir_dFV);

  if (temp != nullptr)
    evaluate_cauchy_temp_deriv(prinv, nddir, bdnddir, ibdnddir, temp, d_cauchyndir_dT, iFTV,
        d_bdnddir_dFV, d_ibdnddir_dFV, d_I1_dF, d_I2_dF, d_I3_dF, d2_cauchyndir_dF_dT);

  if (d_cauchyndir_dF != nullptr)
  {
    // next 3 updates add partial derivative of (\sigma * n * v) w.r.t. F for constant invariants
    // 1. part is term arising from d(J^{-1})/dF
    d_cauchyndir_dF->update(-prefac * (prinv(1) * dPI(1) * nddir + prinv(2) * dPI(2) * nddir +
                                          dPI(0) * bdnddir - prinv(2) * dPI(1) * ibdnddir),
        iFTV, 0.0);  // DsntDF is cleared here
    // 2. part is term arising from d(b * n * v)/dF
    d_cauchyndir_dF->update(prefac * dPI(0), d_bdnddir_dFV, 1.0);
    // 3. part is term arising from d(b_el^{-1} * n * v)/dF
    d_cauchyndir_dF->update(-prefac * prinv(2) * dPI(1), d_ibdnddir_dFV, 1.0);
    // add d(sigma * n * v)/dI1 \otimes dI1/dF
    d_cauchyndir_dF->update(prefac * (prinv(1) * ddPII(5) * nddir + prinv(2) * ddPII(4) * nddir +
                                         ddPII(0) * bdnddir - prinv(2) * ddPII(5) * ibdnddir),
        d_I1_dF, 1.0);
    // add d(sigma * n * v)/dI2 \otimes dI2/dF
    d_cauchyndir_dF->update(
        prefac * (dPI(1) * nddir + prinv(1) * ddPII(1) * nddir + prinv(2) * ddPII(3) * nddir +
                     ddPII(5) * bdnddir - prinv(2) * ddPII(1) * ibdnddir),
        d_I2_dF, 1.0);
    // add d(sigma * n * v)/dI3 \otimes dI3/dF
    d_cauchyndir_dF->update(
        prefac * (prinv(1) * ddPII(3) * nddir + dPI(2) * nddir + prinv(2) * ddPII(2) * nddir +
                     ddPII(4) * bdnddir - dPI(1) * ibdnddir - prinv(2) * ddPII(3) * ibdnddir),
        d_I3_dF, 1.0);
  }

  if (d2_cauchyndir_dF_dn != nullptr)
  {
    // next three blocks add d/dn(d(\sigma * n * v)/dF) for constant invariants
    // first part is term arising from d/dn(dJ^{-1}/dF)
    tempvec3x1.update(prinv(1) * dPI(1) + prinv(2) * dPI(2), dir_mat, 0.0);
    tempvec3x1.update(dPI(0), bddir, 1.0);
    tempvec3x1.update(-prinv(2) * dPI(1), ibddir, 1.0);
    d2_cauchyndir_dF_dn->multiply_nt(-prefac, iFTV, tempvec3x1, 0.0);

    // second part is term arising from d/dn(d(b * n * v)/dF
    const double fac = prefac * dPI(0);
    tempvec1x3.multiply_tn(1.0, dir_mat, defgrd_mat, 0.0);
    for (int k = 0; k < 3; ++k)
    {
      for (int l = 0; l < 3; ++l)
      {
        for (int z = 0; z < 3; ++z)
          (*d2_cauchyndir_dF_dn)(
              Core::LinAlg::Voigt::IndexMappings::non_symmetric_tensor_to_voigt9_index(k, l), z) +=
              fac * (dir(k) * defgrd(z, l) + static_cast<double>(k == z) * tempvec1x3(0, l));
      }
    }

    // third part is term arising from d/dn(d(b^{-1} * n * t)/dF
    const double fac2 = prefac * prinv(2) * dPI(1);
    for (int k = 0; k < 3; ++k)
    {
      for (int l = 0; l < 3; ++l)
      {
        for (int z = 0; z < 3; ++z)
          (*d2_cauchyndir_dF_dn)(
              Core::LinAlg::Voigt::IndexMappings::non_symmetric_tensor_to_voigt9_index(k, l), z) +=
              fac2 * (ibddir(k, 0) * ibdF(z, l) + ib(z, k) * dirdibdF(0, l));
      }
    }

    // add parts originating from d/dn(d(sigma * n * t)/dI1 \otimes dI1/dF)
    tempvec3x1.update(prinv(1) * ddPII(5) + prinv(2) * ddPII(4), dir_mat, 0.0);
    tempvec3x1.update(ddPII(0), bddir, 1.0);
    tempvec3x1.update(-prinv(2) * ddPII(5), ibddir, 1.0);
    d2_cauchyndir_dF_dn->multiply_nt(prefac, d_I1_dF, tempvec3x1, 1.0);

    // add parts originating from d/dn(d(sigma * n * t)/dI2 \otimes dI2/dF)
    tempvec3x1.update(dPI(1) + prinv(1) * ddPII(1) + prinv(2) * ddPII(3), dir_mat, 0.0);
    tempvec3x1.update(ddPII(5), bddir, 1.0);
    tempvec3x1.update(-prinv(2) * ddPII(1), ibddir, 1.0);
    d2_cauchyndir_dF_dn->multiply_nt(prefac, d_I2_dF, tempvec3x1, 1.0);

    // add parts originating from d/dn(d(sigma * n * t)/dI3 \otimes dI3/dF)
    tempvec3x1.update(prinv(1) * ddPII(3) + dPI(2) + prinv(2) * ddPII(2), dir_mat, 0.0);
    tempvec3x1.update(ddPII(4), bddir, 1.0);
    tempvec3x1.update(-dPI(1) - prinv(2) * ddPII(3), ibddir, 1.0);
    d2_cauchyndir_dF_dn->multiply_nt(prefac, d_I3_dF, tempvec3x1, 1.0);
  }

  if (d2_cauchyndir_dF_ddir != nullptr)
  {
    // next three blocks add d/dt(d(\sigma * n * v)/dF) for constant invariants
    // first part is term arising from d/dt(dJ^{-1}/dF)
    tempvec3x1.update(prinv(1) * dPI(1) + prinv(2) * dPI(2), n_mat, 0.0);
    tempvec3x1.update(dPI(0), bdn, 1.0);
    tempvec3x1.update(-prinv(2) * dPI(1), ibdn, 1.0);
    d2_cauchyndir_dF_ddir->multiply_nt(-prefac, iFTV, tempvec3x1, 0.0);

    // second part is term arising from d/dt(d(b * n * v)/dF
    const double fac = prefac * dPI(0);
    tempvec1x3.multiply_tn(1.0, n_mat, defgrd_mat, 0.0);
    for (int k = 0; k < 3; ++k)
    {
      for (int l = 0; l < 3; ++l)
      {
        for (int z = 0; z < 3; ++z)
          (*d2_cauchyndir_dF_ddir)(
              Core::LinAlg::Voigt::IndexMappings::non_symmetric_tensor_to_voigt9_index(k, l), z) +=
              fac * (n_mat(k, 0) * defgrd(z, l) + static_cast<double>(k == z) * tempvec1x3(0, l));
      }
    }

    // third part is term arising from d/dn(d(b^{-1} * n * v)/dF
    const double fac2 = prefac * prinv(2) * dPI(1);
    for (int k = 0; k < 3; ++k)
    {
      for (int l = 0; l < 3; ++l)
      {
        for (int z = 0; z < 3; ++z)
          (*d2_cauchyndir_dF_ddir)(
              Core::LinAlg::Voigt::IndexMappings::non_symmetric_tensor_to_voigt9_index(k, l), z) +=
              fac2 * (ibdn(k, 0) * ibdF(z, l) + ib(z, k) * ndibdF(0, l));
      }
    }

    // add parts originating from d/dt(d(sigma * n * v)/dI1 \otimes dI1/dF)
    tempvec3x1.update(prinv(1) * ddPII(5) + prinv(2) * ddPII(4), n_mat, 0.0);
    tempvec3x1.update(ddPII(0), bdn, 1.0);
    tempvec3x1.update(-prinv(2) * ddPII(5), ibdn, 1.0);
    d2_cauchyndir_dF_ddir->multiply_nt(prefac, d_I1_dF, tempvec3x1, 1.0);

    // add parts originating from d/dt(d(sigma * n * v)/dI2 \otimes dI2/dF)
    tempvec3x1.update(dPI(1) + prinv(1) * ddPII(1) + prinv(2) * ddPII(3), n_mat, 0.0);
    tempvec3x1.update(ddPII(5), bdn, 1.0);
    tempvec3x1.update(-prinv(2) * ddPII(1), ibdn, 1.0);
    d2_cauchyndir_dF_ddir->multiply_nt(prefac, d_I2_dF, tempvec3x1, 1.0);

    // add parts originating from d/dt(d(sigma * n * v)/dI3 \otimes dI3/dF)
    tempvec3x1.update(prinv(1) * ddPII(3) + dPI(2) + prinv(2) * ddPII(2), n_mat, 0.0);
    tempvec3x1.update(ddPII(4), bdn, 1.0);
    tempvec3x1.update(-dPI(1) - prinv(2) * ddPII(3), ibdn, 1.0);
    d2_cauchyndir_dF_ddir->multiply_nt(prefac, d_I3_dF, tempvec3x1, 1.0);
  }

  if (d2_cauchyndir_dF2 != nullptr)
  {
    // define and fill all tensors that can not be calculated using multiply operations first
    static Core::LinAlg::Matrix<9, 9> d_iFT_dF(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<9, 9> d2_bdnddir_dF2(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<9, 9> d2_ibdnddir_dF2(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<9, 9> d2_I1_dF2(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<9, 9> d2_I2_dF2(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<9, 9> d2_I3_dF2(Core::LinAlg::Initialization::zero);
    d_iFT_dF.clear();
    d2_bdnddir_dF2.clear();
    d2_ibdnddir_dF2.clear();
    d2_I1_dF2.clear();
    d2_I2_dF2.clear();
    d2_I3_dF2.clear();

    static Core::LinAlg::Matrix<3, 3> C(Core::LinAlg::Initialization::zero);
    C.multiply_tn(1.0, defgrd_mat, defgrd_mat, 0.0);

    using map = Core::LinAlg::Voigt::IndexMappings;

    for (int k = 0; k < 3; ++k)
    {
      for (int l = 0; l < 3; ++l)
      {
        for (int m = 0; m < 3; ++m)
        {
          for (int a = 0; a < 3; ++a)
          {
            d_iFT_dF(map::non_symmetric_tensor_to_voigt9_index(k, l),
                map::non_symmetric_tensor_to_voigt9_index(m, a)) = -iF(l, m) * iF(a, k);
            d2_bdnddir_dF2(map::non_symmetric_tensor_to_voigt9_index(k, l),
                map::non_symmetric_tensor_to_voigt9_index(m, a)) =
                (dir(k) * n(m) + dir(m) * n(k)) * static_cast<double>(l == a);
            d2_ibdnddir_dF2(map::non_symmetric_tensor_to_voigt9_index(k, l),
                map::non_symmetric_tensor_to_voigt9_index(m, a)) =
                ibdF(k, a) * (ibddir(m, 0) * ndibdF(0, l) + ibdn(m, 0) * dirdibdF(0, l)) +
                ib(m, k) * (dirdibdF(0, a) * ndibdF(0, l) + dirdibdF(0, l) * ndibdF(0, a)) +
                ibdF(m, l) * (ibddir(k, 0) * ndibdF(0, a) + dirdibdF(0, a) * ibdn(k, 0));
            d2_I1_dF2(map::non_symmetric_tensor_to_voigt9_index(k, l),
                map::non_symmetric_tensor_to_voigt9_index(m, a)) =
                2.0 * static_cast<double>(k == m) * static_cast<double>(l == a);
            d2_I2_dF2(map::non_symmetric_tensor_to_voigt9_index(k, l),
                map::non_symmetric_tensor_to_voigt9_index(m, a)) =
                2.0 *
                (prinv(0) * static_cast<double>(k == m) * static_cast<double>(l == a) +
                    2.0 * defgrd(m, a) * defgrd(k, l) - static_cast<double>(k == m) * C(a, l) -
                    defgrd(k, a) * defgrd(m, l) - b(k, m) * static_cast<double>(l == a));
            d2_I3_dF2(map::non_symmetric_tensor_to_voigt9_index(k, l),
                map::non_symmetric_tensor_to_voigt9_index(m, a)) =
                2.0 * prinv(2) * (2.0 * ibdF(m, a) * ibdF(k, l) - ibdF(m, l) * ibdF(k, a));
          }
        }
      }
    }

    // terms below add contributions originating from d(1st term of DsntDF)/dF
    d2_cauchyndir_dF2->multiply_nt(prefac * (prinv(1) * dPI(1) * nddir + prinv(2) * dPI(2) * nddir +
                                                dPI(0) * bdnddir - prinv(2) * dPI(1) * ibdnddir),
        iFTV, iFTV, 0.0);  // D2sntDF2 is cleared here
    d2_cauchyndir_dF2->update(-prefac * (prinv(1) * dPI(1) * nddir + prinv(2) * dPI(2) * nddir +
                                            dPI(0) * bdnddir - prinv(2) * dPI(1) * ibdnddir),
        d_iFT_dF, 1.0);
    d2_cauchyndir_dF2->multiply_nt(-prefac * dPI(0), iFTV, d_bdnddir_dFV, 1.0);
    d2_cauchyndir_dF2->multiply_nt(prefac * prinv(2) * dPI(1), iFTV, d_ibdnddir_dFV, 1.0);

    d2_cauchyndir_dF2->multiply_nt(
        -prefac * (prinv(1) * ddPII(5) * nddir + prinv(2) * ddPII(4) * nddir + ddPII(0) * bdnddir -
                      prinv(2) * ddPII(5) * ibdnddir),
        iFTV, d_I1_dF, 1.0);
    d2_cauchyndir_dF2->multiply_nt(
        -prefac * (dPI(1) * nddir + prinv(1) * ddPII(1) * nddir + prinv(2) * ddPII(3) * nddir +
                      ddPII(5) * bdnddir - prinv(2) * ddPII(1) * ibdnddir),
        iFTV, d_I2_dF, 1.0);
    d2_cauchyndir_dF2->multiply_nt(
        -prefac * (prinv(1) * ddPII(3) * nddir + dPI(2) * nddir + prinv(2) * ddPII(2) * nddir +
                      ddPII(4) * bdnddir - dPI(1) * ibdnddir - prinv(2) * ddPII(3) * ibdnddir),
        iFTV, d_I3_dF, 1.0);

    // terms below add contributions originating from d(2nd term of DsntDF)/dF
    d2_cauchyndir_dF2->multiply_nt(-prefac * dPI(0), d_bdnddir_dFV, iFTV, 1.0);
    d2_cauchyndir_dF2->update(prefac * dPI(0), d2_bdnddir_dF2, 1.0);
    d2_cauchyndir_dF2->multiply_nt(prefac * ddPII(0), d_bdnddir_dFV, d_I1_dF, 1.0);
    d2_cauchyndir_dF2->multiply_nt(prefac * ddPII(5), d_bdnddir_dFV, d_I2_dF, 1.0);
    d2_cauchyndir_dF2->multiply_nt(prefac * ddPII(4), d_bdnddir_dFV, d_I3_dF, 1.0);

    // terms below add contributions originating from d(3rd term of DsntDF)/dF
    d2_cauchyndir_dF2->multiply_nt(prefac * prinv(2) * dPI(1), d_ibdnddir_dFV, iFTV, 1.0);
    d2_cauchyndir_dF2->update(-prefac * prinv(2) * dPI(1), d2_ibdnddir_dF2, 1.0);
    d2_cauchyndir_dF2->multiply_nt(-prefac * prinv(2) * ddPII(5), d_ibdnddir_dFV, d_I1_dF, 1.0);
    d2_cauchyndir_dF2->multiply_nt(-prefac * prinv(2) * ddPII(1), d_ibdnddir_dFV, d_I2_dF, 1.0);
    d2_cauchyndir_dF2->multiply_nt(
        -prefac * (dPI(1) + prinv(2) * ddPII(3)), d_ibdnddir_dFV, d_I3_dF, 1.0);

    // terms below add contributions originating from d(4th term of DsntDF)/dF
    d2_cauchyndir_dF2->multiply_nt(
        -prefac * (prinv(1) * ddPII(5) * nddir + prinv(2) * ddPII(4) * nddir + ddPII(0) * bdnddir -
                      prinv(2) * ddPII(5) * ibdnddir),
        d_I1_dF, iFTV, 1.0);
    d2_cauchyndir_dF2->multiply_nt(prefac * ddPII(0), d_I1_dF, d_bdnddir_dFV, 1.0);
    d2_cauchyndir_dF2->multiply_nt(-prefac * prinv(2) * ddPII(5), d_I1_dF, d_ibdnddir_dFV, 1.0);
    d2_cauchyndir_dF2->update(prefac * (prinv(1) * ddPII(5) * nddir + prinv(2) * ddPII(4) * nddir +
                                           ddPII(0) * bdnddir - prinv(2) * ddPII(5) * ibdnddir),
        d2_I1_dF2, 1.0);
    d2_cauchyndir_dF2->multiply_nt(
        prefac * (prinv(1) * dddPIII(5) * nddir + prinv(2) * dddPIII(6) * nddir +
                     dddPIII(0) * bdnddir - prinv(2) * dddPIII(5) * ibdnddir),
        d_I1_dF, d_I1_dF, 1.0);
    d2_cauchyndir_dF2->multiply_nt(
        prefac * (ddPII(5) * nddir + prinv(1) * dddPIII(3) * nddir + prinv(2) * dddPIII(9) * nddir +
                     dddPIII(5) * bdnddir - prinv(2) * dddPIII(3) * ibdnddir),
        d_I1_dF, d_I2_dF, 1.0);
    d2_cauchyndir_dF2->multiply_nt(
        prefac * (prinv(1) * dddPIII(9) * nddir + ddPII(4) * nddir + prinv(2) * dddPIII(4) * nddir +
                     dddPIII(6) * bdnddir - ddPII(5) * ibdnddir - prinv(2) * dddPIII(9) * ibdnddir),
        d_I1_dF, d_I3_dF, 1.0);

    // terms below add contributions originating from d(5th term of DsntDF)/dF
    d2_cauchyndir_dF2->multiply_nt(
        -prefac * (dPI(1) * nddir + prinv(1) * ddPII(1) * nddir + prinv(2) * ddPII(3) * nddir +
                      ddPII(5) * bdnddir - prinv(2) * ddPII(1) * ibdnddir),
        d_I2_dF, iFTV, 1.0);
    d2_cauchyndir_dF2->multiply_nt(prefac * ddPII(5), d_I2_dF, d_bdnddir_dFV, 1.0);
    d2_cauchyndir_dF2->multiply_nt(-prefac * prinv(2) * ddPII(1), d_I2_dF, d_ibdnddir_dFV, 1.0);
    d2_cauchyndir_dF2->update(
        prefac * (dPI(1) * nddir + prinv(1) * ddPII(1) * nddir + prinv(2) * ddPII(3) * nddir +
                     ddPII(5) * bdnddir - prinv(2) * ddPII(1) * ibdnddir),
        d2_I2_dF2, 1.0);
    d2_cauchyndir_dF2->multiply_nt(
        prefac * (ddPII(5) * nddir + prinv(1) * dddPIII(3) * nddir + prinv(2) * dddPIII(9) * nddir +
                     dddPIII(5) * bdnddir - prinv(2) * dddPIII(3) * ibdnddir),
        d_I2_dF, d_I1_dF, 1.0);
    d2_cauchyndir_dF2->multiply_nt(
        prefac * (2.0 * ddPII(1) * nddir + prinv(1) * dddPIII(1) * nddir +
                     prinv(2) * dddPIII(7) * nddir + dddPIII(3) * bdnddir -
                     prinv(2) * dddPIII(1) * ibdnddir),
        d_I2_dF, d_I2_dF, 1.0);
    d2_cauchyndir_dF2->multiply_nt(
        prefac * (2.0 * ddPII(3) * nddir + prinv(1) * dddPIII(7) * nddir +
                     prinv(2) * dddPIII(8) * nddir + dddPIII(9) * bdnddir - ddPII(1) * ibdnddir -
                     prinv(2) * dddPIII(7) * ibdnddir),
        d_I2_dF, d_I3_dF, 1.0);

    // terms below add contributions originating from d(6th term of DsntDF)/dF
    d2_cauchyndir_dF2->multiply_nt(
        -prefac * (prinv(1) * ddPII(3) * nddir + dPI(2) * nddir + prinv(2) * ddPII(2) * nddir +
                      ddPII(4) * bdnddir - dPI(1) * ibdnddir - prinv(2) * ddPII(3) * ibdnddir),
        d_I3_dF, iFTV, 1.0);
    d2_cauchyndir_dF2->multiply_nt(prefac * ddPII(4), d_I3_dF, d_bdnddir_dFV, 1.0);
    d2_cauchyndir_dF2->multiply_nt(
        -prefac * (dPI(1) + prinv(2) * ddPII(3)), d_I3_dF, d_ibdnddir_dFV, 1.0);
    d2_cauchyndir_dF2->update(
        prefac * (prinv(1) * ddPII(3) * nddir + dPI(2) * nddir + prinv(2) * ddPII(2) * nddir +
                     ddPII(4) * bdnddir - dPI(1) * ibdnddir - prinv(2) * ddPII(3) * ibdnddir),
        d2_I3_dF2, 1.0);
    d2_cauchyndir_dF2->multiply_nt(
        prefac * (prinv(1) * dddPIII(9) * nddir + ddPII(4) * nddir + prinv(2) * dddPIII(4) * nddir +
                     dddPIII(6) * bdnddir - ddPII(5) * ibdnddir - prinv(2) * dddPIII(9) * ibdnddir),
        d_I3_dF, d_I1_dF, 1.0);
    d2_cauchyndir_dF2->multiply_nt(
        prefac * (2.0 * ddPII(3) * nddir + prinv(1) * dddPIII(7) * nddir +
                     prinv(2) * dddPIII(8) * nddir + dddPIII(9) * bdnddir - ddPII(1) * ibdnddir -
                     prinv(2) * dddPIII(7) * ibdnddir),
        d_I3_dF, d_I2_dF, 1.0);
    d2_cauchyndir_dF2->multiply_nt(
        prefac * (prinv(1) * dddPIII(8) * nddir + 2.0 * ddPII(2) * nddir +
                     prinv(2) * dddPIII(2) * nddir + dddPIII(4) * bdnddir -
                     2.0 * ddPII(3) * ibdnddir - prinv(2) * dddPIII(8) * ibdnddir),
        d_I3_dF, d_I3_dF, 1.0);
  }

  return cauchy_n_dir;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ElastHyper::vis_names(std::map<std::string, int>& names) const
{
  if (anisotropic_principal() or anisotropic_modified())
  {
    std::vector<Core::LinAlg::Tensor<double, 3>> fibervecs;
    get_fiber_vecs(fibervecs);
    int vissize = fibervecs.size();
    std::string fiber;
    for (int i = 0; i < vissize; i++)
    {
      std::ostringstream s;
      s << "Fiber" << i + 1;
      fiber = s.str();
      names[fiber] = 3;  // 3-dim vector
    }
  }
  // do visualization for isotropic materials as well
  // loop map of associated potential summands
  for (auto& p : potsum_)
  {
    p->vis_names(names);
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool Mat::ElastHyper::vis_data(
    const std::string& name, std::vector<double>& data, int numgp, int eleID) const
{
  //
  int return_val = 0;
  if (anisotropic_principal() or anisotropic_modified())
  {
    std::vector<Core::LinAlg::Tensor<double, 3>> fibervecs;
    get_fiber_vecs(fibervecs);
    int vissize = fibervecs.size();
    for (int i = 0; i < vissize; i++)
    {
      std::ostringstream s;
      s << "Fiber" << i + 1;
      std::string fiber;
      fiber = s.str();
      if (name == fiber)
      {
        if ((int)data.size() != 3) FOUR_C_THROW("size mismatch");
        data[0] = fibervecs.at(i)(0);
        data[1] = fibervecs.at(i)(1);
        data[2] = fibervecs.at(i)(2);
      }
    }
    // return true;
    return_val = 1;
  }
  // loop map of associated potential summands
  for (auto& p : potsum_)
  {
    return_val += static_cast<int>(p->vis_data(name, data, numgp, eleID));
  }
  return (bool)return_val;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<const Mat::Elastic::Summand> Mat::ElastHyper::get_pot_summand_ptr(
    const Core::Materials::MaterialType& materialtype) const
{
  for (const auto& p : potsum_)
  {
    if (p->material_type() == materialtype) return p;
  }
  return nullptr;
}

FOUR_C_NAMESPACE_CLOSE
