// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_plasticelasthyper.hpp"

#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix_solver.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_linalg_utils_densematrix_eigen.hpp"
#include "4C_linalg_utils_densematrix_funct.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_SerialDenseSolver.hpp>

FOUR_C_NAMESPACE_OPEN

namespace
{
  constexpr double as_convergence_tol = 1e-12;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PAR::PlasticElastHyper::PlasticElastHyper(const Core::Mat::PAR::Parameter::Data& matdata)
    : Mat::PAR::ElastHyper(matdata),
      inityield_(matdata.parameters.get<double>("INITYIELD")),
      isohard_(matdata.parameters.get<double>("ISOHARD")),
      expisohard_(matdata.parameters.get<double>("EXPISOHARD")),
      infyield_(matdata.parameters.get<double>("INFYIELD")),
      kinhard_(matdata.parameters.get<double>("KINHARD")),
      visc_(matdata.parameters.get<double>("VISC")),
      rate_dependency_(matdata.parameters.get<double>("RATE_DEPENDENCY")),
      visc_soft_(matdata.parameters.get<double>("VISC_SOFT")),
      cte_(matdata.parameters.get<double>("CTE")),
      inittemp_(matdata.parameters.get<double>("INITTEMP")),
      yieldsoft_(matdata.parameters.get<double>("YIELDSOFT")),
      hardsoft_(matdata.parameters.get<double>("HARDSOFT")),
      taylor_quinney_(matdata.parameters.get<double>("TAYLOR_QUINNEY")),
      plspin_chi_(-1. * matdata.parameters.get<double>("PL_SPIN_CHI")),
      rY_11_(matdata.parameters.get<double>("rY_11")),
      rY_22_(matdata.parameters.get<double>("rY_22")),
      rY_33_(matdata.parameters.get<double>("rY_33")),
      rY_12_(matdata.parameters.get<double>("rY_12")),
      rY_23_(matdata.parameters.get<double>("rY_23")),
      rY_13_(matdata.parameters.get<double>("rY_13")),
      cpl_(0.),
      stab_s_(0.),
      dis_mode_(TSI::pl_multiplier)
{
  // check if sizes fit
  if (nummat_ != (int)matids_.size())
    FOUR_C_THROW("number of materials {} does not fit to size of material vector {}", nummat_,
        matids_.size());

  // check plastic parameter validity
  if (inityield_ <= 0.)
    std::cout << "***************************************************\n"
                 "Warning: negative initial yield stress detected!!!\n"
                 "this will be converted to a purely elastic response"
              << "\n***************************************************\n"
              << std::endl;

  // no infyield provided 0. is default
  if (infyield_ == 0.)
    if (expisohard_ != 0.) FOUR_C_THROW("hardening exponent provided without inf yield stress");
  if (expisohard_ < 0.) FOUR_C_THROW("Nonlinear hardening exponent must be non-negative");

  // polyconvexity check is just implemented for isotropic hyperlastic materials
  if (polyconvex_)
    FOUR_C_THROW(
        "This polyconvexity-check is just implemented for isotropic "
        "hyperelastic-materials (do not use for plastic materials).");
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::PlasticElastHyper::create_material()
{
  return std::make_shared<Mat::PlasticElastHyper>(this);
}


Mat::PlasticElastHyperType Mat::PlasticElastHyperType::instance_;


Core::Communication::ParObject* Mat::PlasticElastHyperType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::PlasticElastHyper* elhy = new Mat::PlasticElastHyper();
  elhy->unpack(buffer);

  return elhy;
}


/*----------------------------------------------------------------------*
 |  initialise static arrays                                 seitz 05/14|
 *----------------------------------------------------------------------*/
Core::LinAlg::Matrix<3, 1> Mat::PlasticElastHyper::prinv_;
Core::LinAlg::Matrix<9, 1> Mat::PlasticElastHyper::CFpiCei_;
Core::LinAlg::Matrix<9, 1> Mat::PlasticElastHyper::CFpi_;
Core::LinAlg::Matrix<9, 1> Mat::PlasticElastHyper::CFpiCe_;
Core::LinAlg::Matrix<6, 1> Mat::PlasticElastHyper::Cpi_;
Core::LinAlg::Matrix<6, 1> Mat::PlasticElastHyper::CpiCCpi_;
Core::LinAlg::Matrix<6, 1> Mat::PlasticElastHyper::ircg_;
Core::LinAlg::Matrix<6, 1> Mat::PlasticElastHyper::Ce_;
Core::LinAlg::Matrix<6, 1> Mat::PlasticElastHyper::Ce2_;
Core::LinAlg::Matrix<6, 1> Mat::PlasticElastHyper::id2V_;
Core::LinAlg::Matrix<6, 1> Mat::PlasticElastHyper::bev_;
Core::LinAlg::Matrix<6, 1> Mat::PlasticElastHyper::be2v_;
Core::LinAlg::Matrix<3, 3> Mat::PlasticElastHyper::invpldefgrd_;
Core::LinAlg::Matrix<3, 3> Mat::PlasticElastHyper::CeM_;
Core::LinAlg::Matrix<3, 3> Mat::PlasticElastHyper::id2_;
Core::LinAlg::Matrix<3, 3> Mat::PlasticElastHyper::CpiC_;
Core::LinAlg::Matrix<3, 3> Mat::PlasticElastHyper::FpiCe_;
Core::LinAlg::Matrix<3, 3> Mat::PlasticElastHyper::FpiTC_;
Core::LinAlg::Matrix<3, 3> Mat::PlasticElastHyper::CeFpiTC_;
Core::LinAlg::Matrix<3, 3> Mat::PlasticElastHyper::be_;
Core::LinAlg::Matrix<3, 3> Mat::PlasticElastHyper::be2_;
Core::LinAlg::Matrix<3, 3> Mat::PlasticElastHyper::Fe_;
Core::LinAlg::Matrix<3, 3> Mat::PlasticElastHyper::beF_;
Core::LinAlg::Matrix<3, 3> Mat::PlasticElastHyper::beFe_;
Core::LinAlg::Matrix<3, 3> Mat::PlasticElastHyper::FCpi_;
Core::LinAlg::Matrix<3, 3> Mat::PlasticElastHyper::beFCpi_;


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PlasticElastHyper::PlasticElastHyper() : params_(nullptr) {}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PlasticElastHyper::PlasticElastHyper(Mat::PAR::PlasticElastHyper* params)
    : params_(params),
      HepDiss_(nullptr),
      dHepDissdd_(nullptr),
      dHepDissdT_(nullptr),
      dHepDissdTeas_(nullptr)
{
  // make sure the referenced materials in material list have quick access parameters
  std::vector<int>::const_iterator m;
  for (m = mat_params()->matids_.begin(); m != mat_params()->matids_.end(); ++m)
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
void Mat::PlasticElastHyper::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // matid
  int matid = -1;
  if (mat_params() != nullptr) matid = mat_params()->id();  // in case we are in post-process mode
  add_to_pack(data, matid);
  summandProperties_.pack(data);

  // plastic anisotropy
  add_to_pack(data, PlAniso_full_);
  add_to_pack(data, InvPlAniso_full_);

  if (mat_params() != nullptr)  // summands are not accessible in postprocessing mode
  {
    // loop map of associated potential summands
    for (unsigned int p = 0; p < potsum_.size(); ++p)
    {
      potsum_[p]->pack_summand(data);
    }
  }

  // plastic history data
  add_to_pack(data, last_plastic_defgrd_inverse_);
  add_to_pack(data, last_alpha_isotropic_);
  add_to_pack(data, last_alpha_kinematic_);

  add_to_pack(data, activity_state_);
  add_to_pack(data, delta_alpha_i_);

  // tsi data
  bool tsi = HepDiss_ != nullptr;
  add_to_pack(data, tsi);
  bool tsi_eas = dHepDissdTeas_ != nullptr;
  add_to_pack(data, tsi_eas);
  if (tsi) add_to_pack(data, dHepDissdd_->at(0).numRows());

  // dissipation mode
  add_to_pack(data, dis_mode());

  add_to_pack(data, cpl());
  add_to_pack(data, s());

  anisotropy_.pack_anisotropy(data);

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::PlasticElastHyper::unpack(Core::Communication::UnpackBuffer& buffer)
{
  // make sure we have a pristine material
  params_ = nullptr;
  potsum_.clear();

  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid and recover MatParams()
  int matid;
  extract_from_pack(buffer, matid);
  if (Global::Problem::instance()->materials() != nullptr)
  {
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const unsigned int probinst =
          Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
        params_ = static_cast<Mat::PAR::PlasticElastHyper*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
  }

  summandProperties_.unpack(buffer);

  // plastic anisotropy
  extract_from_pack(buffer, PlAniso_full_);
  extract_from_pack(buffer, InvPlAniso_full_);

  if (mat_params() != nullptr)  // summands are not accessible in postprocessing mode
  {
    // make sure the referenced materials in material list have quick access parameters
    std::vector<int>::const_iterator m;
    for (m = mat_params()->matids_.begin(); m != mat_params()->matids_.end(); ++m)
    {
      const int matid = *m;
      std::shared_ptr<Mat::Elastic::Summand> sum = Mat::Elastic::Summand::factory(matid);
      if (sum == nullptr) FOUR_C_THROW("Failed to allocate");
      potsum_.push_back(sum);
    }

    // loop map of associated potential summands
    for (unsigned int p = 0; p < potsum_.size(); ++p)
    {
      potsum_[p]->unpack_summand(buffer);
      potsum_[p]->register_anisotropy_extensions(anisotropy_);
    }
  }

  // plastic history data
  extract_from_pack(buffer, last_plastic_defgrd_inverse_);
  extract_from_pack(buffer, last_alpha_isotropic_);
  extract_from_pack(buffer, last_alpha_kinematic_);

  extract_from_pack(buffer, activity_state_);
  extract_from_pack(buffer, delta_alpha_i_);

  bool tsi;
  extract_from_pack(buffer, tsi);
  bool tsi_eas;
  extract_from_pack(buffer, tsi_eas);
  if (!tsi)
  {
    HepDiss_ = nullptr;
    dHepDissdd_ = nullptr;
    dHepDissdT_ = nullptr;
    dHepDissdTeas_ = nullptr;
  }
  else
  {
    int ngp = last_alpha_isotropic_.size();
    HepDiss_ = std::make_shared<std::vector<double>>(ngp, 0.0);
    int numdofperelement;
    extract_from_pack(buffer, numdofperelement);
    dHepDissdd_ = std::make_shared<std::vector<Core::LinAlg::SerialDenseVector>>(
        ngp, Core::LinAlg::SerialDenseVector(numdofperelement));
    dHepDissdT_ = std::make_shared<std::vector<double>>(ngp, 0.0);
    if (tsi_eas)
      dHepDissdTeas_ = std::make_shared<std::vector<Core::LinAlg::SerialDenseVector>>(
          ngp, Core::LinAlg::SerialDenseVector(numdofperelement / 3));
  }

  // dissipation mode
  TSI::DissipationMode mode;
  extract_from_pack(buffer, mode);
  set_dissipation_mode(mode);

  double cpl;
  double s;
  extract_from_pack(buffer, cpl);
  extract_from_pack(buffer, s);
  get_params(s, cpl);

  anisotropy_.unpack_anisotropy(buffer);

  // in the postprocessing mode, we do not unpack everything we have packed
  // -> position check cannot be done in this case
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::PlasticElastHyper::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  // Read anisotropy
  anisotropy_.set_number_of_gauss_points(numgp);
  anisotropy_.read_anisotropy_from_element(fibers, coord_system);

  // Setup summands
  for (unsigned int p = 0; p < potsum_.size(); ++p)
  {
    potsum_[p]->setup(numgp, fibers, coord_system);
  }

  // find out which formulations are used

  summandProperties_.clear();

  elast_hyper_properties(potsum_, summandProperties_);

  // in this case the mandel stress become non-symmetric and the
  // calculated derivatives have to be extended.
  if (summandProperties_.anisomod == true || summandProperties_.anisoprinc == true)
    FOUR_C_THROW("PlasticElastHyper only for isotropic elastic material!");

  // no visco-elasticity (yet?)
  if (summandProperties_.viscoGeneral)
    FOUR_C_THROW("no visco-elasticity in PlasticElastHyper...yet(?)");

  // check if either zero or three fiber directions are given
  if (fibers.element_fibers.size() != 0 && fibers.element_fibers.size() != 3)
    FOUR_C_THROW("so3 expects no fibers or 3 fiber directions");

  // plastic anisotropy
  setup_hill_plasticity(fibers);

  // setup plastic history variables
  Core::LinAlg::Matrix<3, 3> tmp(Core::LinAlg::Initialization::zero);
  last_alpha_isotropic_.resize(numgp, 0.);
  last_alpha_kinematic_.resize(numgp, tmp);
  for (int i = 0; i < 3; i++) tmp(i, i) = 1.;
  last_plastic_defgrd_inverse_.resize(numgp, tmp);
  activity_state_.resize(numgp, false);
  delta_alpha_i_.resize(numgp, 0.);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::PlasticElastHyper::setup_tsi(
    const int numgp, const int numdofperelement, const bool eas, const TSI::DissipationMode mode)
{
  // dissipation mode
  if (mode == TSI::pl_multiplier)
    if (mat_params()->rY_11_ != 0. || mat_params()->rY_22_ != 0. || mat_params()->rY_33_ != 0. ||
        mat_params()->rY_12_ != 0. || mat_params()->rY_23_ != 0. || mat_params()->rY_13_ != 0.)
      FOUR_C_THROW("TSI with Hill plasticity not available with DISSIPATION_MODE pl_multiplier");
  set_dissipation_mode(mode);

  // allocate memory
  HepDiss_ = std::make_shared<std::vector<double>>(numgp, 0.0);
  dHepDissdd_ = std::make_shared<std::vector<Core::LinAlg::SerialDenseVector>>(
      numgp, Core::LinAlg::SerialDenseVector(numdofperelement));
  dHepDissdT_ = std::make_shared<std::vector<double>>(numgp, 0.0);
  if (eas)
    dHepDissdTeas_ = std::make_shared<std::vector<Core::LinAlg::SerialDenseVector>>(
        numgp, Core::LinAlg::SerialDenseVector(numdofperelement / 3));

  // no TSI with kinematic hardening yet
  // be aware that in that case, another dependency of the NCP function on the
  // temperature arises, namely via H^k=H^k(T) in the computation of the effective
  // stress eta. Without this term, the only dependency of the NCP function is
  // via the effective yield stress Y^pl=Y^pl(T)
  if (kinhard() != 0.) FOUR_C_THROW("no kinematic hardening for TSI (yet)");
  // no TSI with plastic spin yet
  // besides the kinematic hardening term (see above) there is not much to do
  // just add the derivatives of theating and NCP function in the
  // evaluate_nc_pand_spin(...) function
  if (pl_spin_chi() != 0.) FOUR_C_THROW("no thermo-plasticitiy with plastic spin");

  /// Hill TSI only with pl_flow dissipation
  if (mat_params()->rY_11_ != 0. && mode == TSI::pl_multiplier)
    FOUR_C_THROW("hill thermo plasticity not with dissipation mode pl_multiplier");

  /// viscoplastic TSI only with  pl_flow dissipation
  if (visc() != 0. && mode == TSI::pl_multiplier)
    FOUR_C_THROW("thermo-visco-plasticity not with dissipation mode pl_multiplier");
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::PlasticElastHyper::setup_hill_plasticity(const Discret::Elements::Fibers& fibers)
{
  // check if parameters are valid
  if (mat_params()->rY_11_ != 0. || mat_params()->rY_22_ != 0. || mat_params()->rY_33_ != 0. ||
      mat_params()->rY_12_ != 0. || mat_params()->rY_23_ != 0. || mat_params()->rY_13_ != 0.)
    if (mat_params()->rY_11_ <= 0. || mat_params()->rY_22_ <= 0. || mat_params()->rY_33_ <= 0. ||
        mat_params()->rY_12_ <= 0. || mat_params()->rY_23_ <= 0. || mat_params()->rY_13_ <= 0.)
      FOUR_C_THROW("Hill parameters all must be positive (incomplete set?)");

  // all (optional) Hill parameters are zero (default value)
  // --> we want to do von Mises plasticity
  if (mat_params()->rY_11_ == 0. && mat_params()->rY_22_ == 0. && mat_params()->rY_33_ == 0. &&
      mat_params()->rY_12_ == 0. && mat_params()->rY_23_ == 0. && mat_params()->rY_13_ == 0.)
  {
    PlAniso_full_.clear();
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        if (i == j)
          PlAniso_full_(i, j) = 2. / 3.;
        else
          PlAniso_full_(i, j) = -1. / 3.;
    for (int i = 3; i < 6; i++) PlAniso_full_(i, i) = 1.;
    InvPlAniso_full_.update(PlAniso_full_);
  }
  // we do Hill plasticity
  else
  {
    // anisotropy directions
    std::vector<Core::LinAlg::Matrix<3, 1>> directions(3);

    std::size_t dir_index = 0;
    for (const auto& fiber : fibers.element_fibers)
    {
      double fnorm = std::sqrt(Core::LinAlg::norm2(fiber));
      if (fnorm == 0.) FOUR_C_THROW("Fiber vector has norm zero");

      // fill final fiber vector
      for (int i = 0; i < 3; ++i) directions.at(dir_index)(i) = fiber(i) / fnorm;
      dir_index++;
    }

    // check orthogonality
    Core::LinAlg::Matrix<1, 1> matrix1;
    matrix1.multiply_tn(directions.at(0), directions.at(1));
    if (std::abs(matrix1(0, 0)) > 1.e-16) FOUR_C_THROW("fiber directions not orthogonal");
    matrix1.multiply_tn(directions.at(0), directions.at(2));
    if (std::abs(matrix1(0, 0)) > 1.e-16) FOUR_C_THROW("fiber directions not orthogonal");
    matrix1.multiply_tn(directions.at(2), directions.at(1));
    if (std::abs(matrix1(0, 0)) > 1.e-16) FOUR_C_THROW("fiber directions not orthogonal");

    // check right-handed trihedron
    Core::LinAlg::Matrix<3, 1> A0xA1;
    A0xA1(0) =
        (directions.at(0)(1) * directions.at(1)(2) - directions.at(0)(2) * directions.at(1)(1));
    A0xA1(1) =
        (directions.at(0)(2) * directions.at(1)(0) - directions.at(0)(0) * directions.at(1)(2));
    A0xA1(2) =
        (directions.at(0)(0) * directions.at(1)(1) - directions.at(0)(1) * directions.at(1)(0));
    A0xA1.update(-1., directions.at(2), 1.);
    if (A0xA1.norm2() > 1.e-8) FOUR_C_THROW("fibers don't form right-handed trihedron");

    // setup structural tensor for first and second direction
    // (as the directions are orthogonal, 2 structural tensors are sufficient)
    Core::LinAlg::Matrix<3, 3> M0;
    M0.multiply_nt(directions.at(0), directions.at(0));
    Core::LinAlg::Matrix<3, 3> M1;
    M1.multiply_nt(directions.at(1), directions.at(1));
    Core::LinAlg::Matrix<3, 3> M2;
    M2.multiply_nt(directions.at(2), directions.at(2));

    double alpha1 = 2. / 3. / mat_params()->rY_11_ / mat_params()->rY_11_;
    double alpha2 = 2. / 3. / mat_params()->rY_22_ / mat_params()->rY_22_;
    double alpha3 = 2. / 3. / mat_params()->rY_33_ / mat_params()->rY_33_;
    double alpha4 = 1. / 3. / mat_params()->rY_12_ / mat_params()->rY_12_;
    double alpha5 = 1. / 3. / mat_params()->rY_23_ / mat_params()->rY_23_;
    double alpha6 = 1. / 3. / mat_params()->rY_13_ / mat_params()->rY_13_;

    // calculate plastic anisotropy tensor
    PlAniso_full_.clear();
    Core::LinAlg::FourTensorOperations::add_elasticity_tensor_product(
        PlAniso_full_, alpha1, M0, M0, 1.);
    Core::LinAlg::FourTensorOperations::add_elasticity_tensor_product(
        PlAniso_full_, alpha2, M1, M1, 1.);
    Core::LinAlg::FourTensorOperations::add_elasticity_tensor_product(
        PlAniso_full_, alpha3, M2, M2, 1.);
    Core::LinAlg::FourTensorOperations::add_symmetric_elasticity_tensor_product(
        PlAniso_full_, 0.5 * (alpha3 - alpha1 - alpha2), M0, M1, 1.);
    Core::LinAlg::FourTensorOperations::add_symmetric_elasticity_tensor_product(
        PlAniso_full_, 0.5 * (alpha1 - alpha2 - alpha3), M1, M2, 1.);
    Core::LinAlg::FourTensorOperations::add_symmetric_elasticity_tensor_product(
        PlAniso_full_, 0.5 * (alpha2 - alpha3 - alpha1), M0, M2, 1.);
    Core::LinAlg::FourTensorOperations::add_symmetric_holzapfel_product(
        PlAniso_full_, M0, M1, alpha4);
    Core::LinAlg::FourTensorOperations::add_symmetric_holzapfel_product(
        PlAniso_full_, M1, M2, alpha5);
    Core::LinAlg::FourTensorOperations::add_symmetric_holzapfel_product(
        PlAniso_full_, M0, M2, alpha6);

    // we need this matrix to get rid of the zero eigenvalue to be able to invert
    // the anisotropy tensor. After the inversion we expand the tensor again to 6x6
    // so that we have the correct pseudo-inverse.
    Core::LinAlg::Matrix<6, 5> red(Core::LinAlg::Initialization::zero);
    red(0, 0) = 1.;
    red(1, 1) = 1.;
    red(2, 0) = -1.;
    red(2, 1) = -1.;
    red(3, 2) = 1.;
    red(4, 3) = 1.;
    red(5, 4) = 1.;

    // invert plastic anisotropy tensor
    Core::LinAlg::Matrix<6, 5> tmp;
    tmp.multiply(PlAniso_full_, red);
    Core::LinAlg::Matrix<5, 5> tmp55;
    tmp55.multiply_tn(red, tmp);
    Core::LinAlg::FixedSizeSerialDenseSolver<5, 5, 1> solver;
    solver.set_matrix(tmp55);
    int err2 = solver.factor();
    int err = solver.invert();
    if ((err != 0) || (err2 != 0)) FOUR_C_THROW("Inversion of plastic anisotropy tensor failed");
    tmp.multiply_nt(red, tmp55);
    InvPlAniso_full_.multiply_nt(tmp, red);
  }
}

/*----------------------------------------------------------------------*
 |  evaluate elastic stress and stiffness                   seitz 05/14 |
 *----------------------------------------------------------------------*/
void Mat::PlasticElastHyper::evaluate_elast(const Core::LinAlg::Matrix<3, 3>* defgrd,
    const Core::LinAlg::Matrix<3, 3>* deltaLp, Core::LinAlg::Matrix<6, 1>* pk2,
    Core::LinAlg::Matrix<6, 6>* cmat, const int gp, const int eleGID)
{
  Core::LinAlg::Matrix<3, 1> dPI(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6, 1> ddPII(Core::LinAlg::Initialization::zero);

  evaluate_kin_quant_elast(defgrd, deltaLp, gp);
  elast_hyper_evaluate_invariant_derivatives(
      prinv_, dPI, ddPII, potsum_, summandProperties_, gp, eleGID);

  // blank resulting quantities
  // ... even if it is an implicit law that cmat is zero upon input
  pk2->clear();
  cmat->clear();

  // isotropic elasticity in coupled strain energy format
  // isotropic elasticity in decoupled ("mod") format go here as well
  // as the modified gammas and deltas have been converted
  if (summandProperties_.isoprinc || summandProperties_.isomod)
    evaluate_isotropic_princ_elast(*pk2, *cmat, dPI, ddPII);
  else
    FOUR_C_THROW("only isotropic hyperelastic materials");

  return;
}

/*----------------------------------------------------------------------*
 |  evaluate elastic strain energy                          seitz 03/16 |
 *----------------------------------------------------------------------*/
double Mat::PlasticElastHyper::strain_energy_tsi(
    const Core::LinAlg::Matrix<3, 3>& defgrd, const int gp, const int eleGID, const double temp)
{
  double psi = 0.;

  Core::LinAlg::Matrix<3, 3> Fe;
  Fe.multiply(defgrd, last_plastic_defgrd_inverse_[gp]);
  Core::LinAlg::Matrix<3, 3> elRCG;
  elRCG.multiply_tn(Fe, Fe);
  Core::LinAlg::SymmetricTensor<double, 3, 3> elRCGv;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) elRCGv(i, j) = elRCG(i, j);
  Core::LinAlg::Matrix<3, 1> prinv;
  Core::LinAlg::Voigt::Stresses::invariants_principal(
      prinv, Core::LinAlg::make_stress_like_voigt_view(elRCGv));
  Core::LinAlg::Matrix<3, 1> modinv;
  invariants_modified(modinv, prinv);

  // loop map of associated potential summands
  Core::LinAlg::SymmetricTensor<double, 3, 3> glstrain{};
  Core::LinAlg::Matrix<6, 1> idv(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; ++i) idv(i) = 1.0;
  glstrain = 0.5 * elRCGv - Core::LinAlg::TensorGenerators::identity<double, 3, 3>;
  for (unsigned int p = 0; p < potsum_.size(); ++p)
    potsum_[p]->add_strain_energy(psi, prinv, modinv, glstrain, gp, eleGID);

  double dPj1 = 0.;
  for (unsigned int p = 0; p < potsum_.size(); ++p)
    potsum_[p]->add_coup_deriv_vol(sqrt(prinv(0)), &dPj1, nullptr, nullptr, nullptr);
  psi -= 3. * cte() * (temp - init_temp()) * dPj1;



  return psi;
}

/*----------------------------------------------------------------------*
 |  evaluate thermal stress and stiffness                   seitz 06/14 |
 *----------------------------------------------------------------------*/
void Mat::PlasticElastHyper::evaluate_thermal_stress(const Core::LinAlg::Matrix<3, 3>* defgrd,
    const double temp, Core::LinAlg::Matrix<6, 1>* pk2, Core::LinAlg::Matrix<6, 6>* cmat,
    const int gp, const int eleGID)
{
  // do TSI only for decoupled isotropic materials. By doing so, the stresses
  // due to thermal expansion can be easily calculated by the volumetric
  // part of the strain energy function
  if (summandProperties_.anisomod || summandProperties_.anisoprinc)
    FOUR_C_THROW(
        "TSI with semi-Smooth Newton type plasticity algorithm only "
        "with isotropic strain energy functions");

  // temperature difference
  double deltaT = temp - init_temp();

  // we are only interested in the volumetric response
  // which is for decoupled strain energy functions defined by
  // modinv_3 = J only.
  Core::LinAlg::Matrix<3, 1> modinv(Core::LinAlg::Initialization::zero);
  modinv(2) = defgrd->determinant();
  Core::LinAlg::Matrix<3, 1> dPmodI;
  Core::LinAlg::Matrix<6, 1> ddPmodII;
  double dddPmodIII = 0.;

  // loop map of associated potential summands
  for (unsigned int p = 0; p < potsum_.size(); ++p)
  {
    potsum_[p]->add_derivatives_modified(dPmodI, ddPmodII, modinv, gp, eleGID);
    potsum_[p]->add3rd_vol_deriv(modinv, dddPmodIII);
    potsum_[p]->add_coup_deriv_vol(modinv(2), &dPmodI(2), &ddPmodII(2), &dddPmodIII, nullptr);
  }

  // inverse RCG
  Core::LinAlg::Matrix<3, 3> invRCG;
  invRCG.multiply_tn(*defgrd, *defgrd);
  invRCG.invert();
  Core::LinAlg::Matrix<6, 1> icg;
  for (int i = 0; i < 3; ++i) icg(i) = invRCG(i, i);
  icg(3) = invRCG(0, 1);
  icg(4) = invRCG(1, 2);
  icg(5) = invRCG(0, 2);

  pk2->update(-3. * cte() * deltaT * modinv(2) * ddPmodII(2), icg, 1.);
  Core::LinAlg::FourTensorOperations::add_elasticity_tensor_product(
      *cmat, -3. * cte() * deltaT * modinv(2) * modinv(2) * dddPmodIII, invRCG, invRCG, 1.);
  Core::LinAlg::FourTensorOperations::add_elasticity_tensor_product(
      *cmat, -3. * cte() * deltaT * modinv(2) * ddPmodII(2), invRCG, invRCG, 1.);
  Core::LinAlg::FourTensorOperations::add_kronecker_tensor_product(
      *cmat, +6. * cte() * deltaT * modinv(2) * ddPmodII(2), invRCG, invRCG, 1.);

  return;
}

/*----------------------------------------------------------------------*
 |  evaluate thermal stress and stiffness                   seitz 06/14 |
 *----------------------------------------------------------------------*/
void Mat::PlasticElastHyper::evaluate_c_tvol(const Core::LinAlg::Matrix<3, 3>* defgrd,
    Core::LinAlg::Matrix<6, 1>* cTvol, Core::LinAlg::Matrix<6, 6>* dCTvoldE, const int gp,
    const int eleGID)
{
  // do TSI only for decoupled isotropic materials. By doing so, the stresses
  // due to thermal expansion can be easily calculated by the volumetric
  // part of the strain energy function
  if (summandProperties_.anisomod || summandProperties_.anisoprinc)
    FOUR_C_THROW(
        "TSI with semi-Smooth Newton type plasticity algorithm only "
        "with isotropic strain energy functions");

  // we are only interested in the volumetric response
  // which is for decoupled strain energy functions defined by
  // modinv_3 = J only.
  Core::LinAlg::Matrix<3, 1> modinv(Core::LinAlg::Initialization::zero);
  modinv(2) = defgrd->determinant();
  Core::LinAlg::Matrix<3, 1> dPmodI;
  Core::LinAlg::Matrix<6, 1> ddPmodII;
  double dddPmodIII = 0.;

  // loop map of associated potential summands
  for (unsigned int p = 0; p < potsum_.size(); ++p)
  {
    potsum_[p]->add_derivatives_modified(dPmodI, ddPmodII, modinv, gp, eleGID);
    potsum_[p]->add3rd_vol_deriv(modinv, dddPmodIII);
    potsum_[p]->add_coup_deriv_vol(modinv(2), &dPmodI(2), &ddPmodII(2), &dddPmodIII, nullptr);
  }

  // clear
  cTvol->clear();
  dCTvoldE->clear();

  // inverse RCG
  Core::LinAlg::Matrix<3, 3> invRCG;
  invRCG.multiply_tn(*defgrd, *defgrd);
  invRCG.invert();
  Core::LinAlg::Matrix<6, 1> icg;
  for (int i = 0; i < 3; ++i) icg(i) = invRCG(i, i);
  icg(3) = invRCG(0, 1);
  icg(4) = invRCG(1, 2);
  icg(5) = invRCG(0, 2);

  cTvol->update(-3. * cte() * modinv(2) * ddPmodII(2), icg, 1.);
  Core::LinAlg::FourTensorOperations::add_elasticity_tensor_product(
      *dCTvoldE, -3. * cte() * modinv(2) * modinv(2) * dddPmodIII, invRCG, invRCG, 1.);
  Core::LinAlg::FourTensorOperations::add_elasticity_tensor_product(
      *dCTvoldE, -3. * cte() * modinv(2) * ddPmodII(2), invRCG, invRCG, 1.);
  Core::LinAlg::FourTensorOperations::add_kronecker_tensor_product(
      *dCTvoldE, +6. * cte() * modinv(2) * ddPmodII(2), invRCG, invRCG, 1.);

  return;
}


/*----------------------------------------------------------------------*
 |  evaluate Gough Joule Effect                             seitz 10/15 |
 *----------------------------------------------------------------------*/
void Mat::PlasticElastHyper::evaluate_gough_joule(
    const double j, const int gp, const int eleGID, double& he_fac, double& he_fac_deriv)
{
  // we are only interested in the volumetric response
  // which is for decoupled strain energy functions defined by
  // modinv_3 = J only.
  Core::LinAlg::Matrix<3, 1> modinv(Core::LinAlg::Initialization::zero);
  modinv(2) = j;
  Core::LinAlg::Matrix<3, 1> dPmodI;
  Core::LinAlg::Matrix<6, 1> ddPmodII;
  double dddPmodIII = 0.;

  // loop map of associated potential summands
  for (unsigned int p = 0; p < potsum_.size(); ++p)
  {
    potsum_[p]->add_derivatives_modified(dPmodI, ddPmodII, modinv, gp, eleGID);
    potsum_[p]->add3rd_vol_deriv(modinv, dddPmodIII);
    potsum_[p]->add_coup_deriv_vol(modinv(2), &dPmodI(2), &ddPmodII(2), &dddPmodIII, nullptr);
  }

  he_fac = -3. * cte() * ddPmodII(2);
  he_fac_deriv = -3. * cte() * dddPmodIII;
  return;
}

/*----------------------------------------------------------------------*
 |  evaluate plastic stress and stiffness                   seitz 05/14 |
 *----------------------------------------------------------------------*/
void Mat::PlasticElastHyper::evaluate_plast(const Core::LinAlg::Matrix<3, 3>* defgrd,
    const Core::LinAlg::Matrix<3, 3>* deltaDp, const double* temp,
    const Teuchos::ParameterList& params, Core::LinAlg::Matrix<6, 6>* dPK2dDp,
    Core::LinAlg::Matrix<6, 1>* NCP, Core::LinAlg::Matrix<6, 6>* dNCPdC,
    Core::LinAlg::Matrix<6, 6>* dNCPdDp, bool* active, bool* elast, bool* as_converged,
    const int gp, Core::LinAlg::Matrix<6, 1>* dNCPdT, Core::LinAlg::Matrix<6, 1>* dHdC,
    Core::LinAlg::Matrix<6, 1>* dHdDp, const double dt, const int eleGID,
    Core::LinAlg::Matrix<6, 1>* cauchy, Core::LinAlg::Matrix<6, 6>* d_cauchy_ddp,
    Core::LinAlg::Matrix<6, 6>* d_cauchy_dC, Core::LinAlg::Matrix<6, 9>* d_cauchy_dF,
    Core::LinAlg::Matrix<6, 1>* d_cauchy_dT)
{
  int check = +(cauchy != nullptr) + (d_cauchy_dC != nullptr) + (d_cauchy_dF != nullptr) +
              (d_cauchy_ddp != nullptr);
  if (!(check == 0 || check == 4)) FOUR_C_THROW("some inconsistency with provided variables");

  Core::LinAlg::Matrix<3, 1> dPI;
  Core::LinAlg::Matrix<6, 1> ddPII;
  Core::LinAlg::Matrix<3, 1> gamma(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<8, 1> delta(Core::LinAlg::Initialization::zero);

  if (evaluate_kin_quant_plast(defgrd, deltaDp, gp, params)) return;
  elast_hyper_evaluate_invariant_derivatives(
      prinv_, dPI, ddPII, potsum_, summandProperties_, gp, eleGID);
  if (temp && cauchy) add_thermal_expansion_derivs(prinv_, dPI, ddPII, gp, eleGID, *temp);
  calculate_gamma_delta(gamma, delta, prinv_, dPI, ddPII);

  // blank resulting quantities
  // ... even if it is an implicit law that cmat is zero upon input
  dPK2dDp->clear();
  NCP->clear();
  dNCPdC->clear();
  dNCPdDp->clear();
  if (dNCPdT != nullptr) dNCPdT->clear();

  // new temporary matrices
  Core::LinAlg::Matrix<3, 3> mStr;  // Mandel stress tensor
  Core::LinAlg::Matrix<6, 6> dMdC;  // derivative of Mandel stress w.r.t. RCG
  Core::LinAlg::Matrix<6, 9>
      dMdFpinv;  // derivative of Mandel stress w.r.t. inverse plastic deformation gradient
  Core::LinAlg::Matrix<6, 9> dPK2dFpinv;
  Core::LinAlg::Matrix<6, 9> d_cauchy_dFpi;

  // isotropic elasticity in coupled strain energy format
  // isotropic elasticity in decoupled ("mod") format go here as well
  // as the modified gammas and deltas have been converted
  if (summandProperties_.isoprinc || summandProperties_.isomod)
    evaluate_isotropic_princ_plast(dPK2dFpinv, mStr, dMdC, dMdFpinv, gamma, delta);
  else
    FOUR_C_THROW("only isotropic hypereleastic materials");

  if (cauchy)
    evaluate_cauchy_plast(
        dPI, ddPII, defgrd, *cauchy, d_cauchy_dFpi, *d_cauchy_dC, *d_cauchy_dF, d_cauchy_dT);

  evaluate_ncp(&mStr, &dMdC, &dMdFpinv, &dPK2dFpinv, deltaDp, gp, temp, NCP, dNCPdC, dNCPdDp,
      dNCPdT, dPK2dDp, active, elast, as_converged, dHdC, dHdDp, params, dt, &d_cauchy_dFpi,
      d_cauchy_ddp);

  return;
}


/*----------------------------------------------------------------------*
 |  evaluate NCP function                                   seitz 05/14 |
 *----------------------------------------------------------------------*/
void Mat::PlasticElastHyper::evaluate_ncp(const Core::LinAlg::Matrix<3, 3>* mStr,
    const Core::LinAlg::Matrix<6, 6>* dMdC, const Core::LinAlg::Matrix<6, 9>* dMdFpinv,
    const Core::LinAlg::Matrix<6, 9>* dPK2dFpinv, const Core::LinAlg::Matrix<3, 3>* deltaDp,
    const int gp, const double* temp, Core::LinAlg::Matrix<6, 1>* NCP,
    Core::LinAlg::Matrix<6, 6>* dNCPdC, Core::LinAlg::Matrix<6, 6>* dNCPdDp,
    Core::LinAlg::Matrix<6, 1>* dNCPdT, Core::LinAlg::Matrix<6, 6>* dPK2dDp, bool* active,
    bool* elast, bool* as_converged, Core::LinAlg::Matrix<6, 1>* dHdC,
    Core::LinAlg::Matrix<6, 1>* dHdDp, const Teuchos::ParameterList& params, const double dt,
    const Core::LinAlg::Matrix<6, 9>* d_cauchy_dFpi, Core::LinAlg::Matrix<6, 6>* d_cauchy_ddp)
{
  const double sq = sqrt(2. / 3.);
  Core::LinAlg::Matrix<6, 1> tmp61;
  double dT = 0.;
  if (dNCPdT)
    dT = *temp - init_temp();
  else
    dT = 0.;

  // deviatoric projection tensor
  Core::LinAlg::Matrix<6, 6> pdev(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      if (i == j)
        pdev(i, j) = 2. / 3.;
      else
        pdev(i, j) = -1. / 3.;
  for (int i = 3; i < 6; i++) pdev(i, i) = 1.;

  // effective stress
  Core::LinAlg::Matrix<3, 3> eta(*mStr);
  for (int i = 0; i < 3; i++)
    eta(i, i) -= 1. / 3. * ((*mStr)(0, 0) + (*mStr)(1, 1) + (*mStr)(2, 2));
  eta.update(2. / 3. * kinhard(), last_alpha_kinematic_[gp], 1.);
  eta.update(-2. / 3. * kinhard(), *deltaDp, 1.);

  // in stress-like voigt notation
  Core::LinAlg::Matrix<6, 1> eta_v;      // in stress-like voigt notation
  Core::LinAlg::Matrix<6, 1> etatr_v;    // in stress-like voigt notation
  Core::LinAlg::Matrix<6, 1> eta_s_v;    // in stress-like voigt notation
  Core::LinAlg::Matrix<6, 1> deltaDp_v;  // in stress-like voigt notation
  for (int i = 0; i < 3; i++)
  {
    eta_v(i) = eta(i, i);
    deltaDp_v(i) = (*deltaDp)(i, i);
  }
  eta_v(3) = .5 * (eta(1, 0) + eta(0, 1));
  deltaDp_v(3) = .5 * ((*deltaDp)(0, 1) + (*deltaDp)(1, 0));
  eta_v(4) = .5 * (eta(1, 2) + eta(2, 1));
  deltaDp_v(4) = .5 * ((*deltaDp)(2, 1) + (*deltaDp)(1, 2));
  eta_v(5) = .5 * (eta(2, 0) + eta(0, 2));
  deltaDp_v(5) = .5 * ((*deltaDp)(0, 2) + (*deltaDp)(2, 0));

  // trial effective stress
  etatr_v.update(eta_v);
  etatr_v.multiply(cpl(), InvPlAniso_full_, deltaDp_v, 1.);

  // in strain-like voigt notation
  Core::LinAlg::Matrix<6, 1> eta_v_strainlike(eta_v);          // in strain-like voigt notation
  Core::LinAlg::Matrix<6, 1> etatr_v_strainlike(etatr_v);      // in strain-like voigt notation
  Core::LinAlg::Matrix<6, 1> deltaDp_v_strainlike(deltaDp_v);  // in strain-like voigt notation
  for (int i = 3; i < 6; i++)
  {
    eta_v_strainlike(i) *= 2.;
    etatr_v_strainlike(i) *= 2.;
    deltaDp_v_strainlike(i) *= 2.;
  }

  // different tensor norms
  tmp61.multiply(PlAniso_full_, eta_v);
  double absHeta = norm_stress_like(tmp61);
  double abseta_H = tmp61.dot(eta_v_strainlike);
  if (abseta_H < -1.e-16)
    FOUR_C_THROW("this should not happen. eta : H : eta ={} < 0", abseta_H);
  else if (abseta_H >= 0.)
    abseta_H = sqrt(abseta_H);
  else
    abseta_H = 0.;
  double dDpHeta = tmp61.dot(deltaDp_v_strainlike);
  tmp61.multiply(PlAniso_full_, etatr_v);
  double absetatr_H = tmp61.dot(etatr_v_strainlike);
  if (absetatr_H < -1.e-16)
    FOUR_C_THROW("this should not happen. eta_tr : H : eta_tr ={} < 0", absetatr_H);
  else if (absetatr_H >= 0.)
    absetatr_H = sqrt(absetatr_H);
  else
    absetatr_H = 0.;
  Core::LinAlg::Matrix<6, 1> HdDp;
  HdDp.multiply(PlAniso_full_, deltaDp_v);
  Core::LinAlg::Matrix<6, 1> HdDp_strainlike;
  HdDp_strainlike.multiply(PlAniso_full_, deltaDp_v_strainlike);
  Core::LinAlg::Matrix<6, 1> HetaH_strainlike;
  tmp61.multiply(PlAniso_full_, eta_v_strainlike);
  HetaH_strainlike.multiply(PlAniso_full_, tmp61);

  // isotropic hardening increment
  delta_alpha_i_[gp] = 0.;
  if (dDpHeta > 0. && absHeta > 0.)
    delta_alpha_i_[gp] = sq * dDpHeta * abseta_H / (absHeta * absHeta);
  // new isotropic hardening value
  const double aI = last_alpha_isotropic_[gp] + delta_alpha_i_[gp];

  // current yield stress equivalent (yield stress scaled by sqrt(2/3))
  double ypl =
      sq *
      ((infyield() * (1. - hard_soft() * dT) - inityield() * (1. - yield_soft() * dT)) *
              (1. - exp(-expisohard() * aI)) +
          isohard() * (1. - hard_soft() * dT) * aI + inityield() * (1. - yield_soft() * dT)) *
      pow(1. + visc() * (1. - visc_soft() * dT) * delta_alpha_i_[gp] / dt, visc_rate());

  double dYpldT = sq *
                  ((infyield() * (-hard_soft()) - inityield() * (-yield_soft())) *
                          (1. - exp(-expisohard() * aI)) -
                      isohard() * hard_soft() * aI - inityield() * yield_soft()) *
                  pow(1. + visc() * (1. - visc_soft() * dT) * delta_alpha_i_[gp] / dt, visc_rate());

  dYpldT += sq *
            ((infyield() * (1. - hard_soft() * dT) - inityield() * (1. - yield_soft() * dT)) *
                    (1. - exp(-expisohard() * aI)) +
                isohard() * (1. - hard_soft() * dT) * aI + inityield() * (1. - yield_soft() * dT)) *
            pow(1. + visc() * (1. - visc_soft() * dT) * delta_alpha_i_[gp] / dt, visc_rate() - 1.) *
            visc_rate() * delta_alpha_i_[gp] / dt * visc() * (-visc_soft());

  // Factor of derivative of Y^pl w.r.t. delta alpha ^i
  // we have added the factor sqrt(2/3) from delta_alpha_i=sq*... here
  double dYplDai =
      2. / 3. *
      (+isohard() * (1. - hard_soft() * dT) +
          (infyield() * (1. - hard_soft() * dT) - inityield() * (1. - yield_soft() * dT)) *
              expisohard() * exp(-expisohard() * aI)) *
      pow(1. + visc() * (1. - visc_soft() * dT) * delta_alpha_i_[gp] / dt, visc_rate());
  dYplDai +=
      2. / 3. *
      ((infyield() * (1. - hard_soft() * dT) - inityield() * (1. - yield_soft() * dT)) *
              (1. - exp(-expisohard() * aI)) +
          isohard() * (1. - hard_soft() * dT) * aI + inityield() * (1. - yield_soft() * dT)) *
      pow(1. + visc() * (1. - visc_soft() * dT) * delta_alpha_i_[gp] / dt, visc_rate() - 1) *
      visc_rate() * visc() * (1. - visc_soft() * dT) / dt;

  // safety check: due to thermal softening, we might get a negative yield stress
  // in that case, no viable solution to the plasticity problem exists, since we
  // need abs(dev(Sigma)) - Y = 0 --> abs(dev(Sigma))=Y, but abs(dev(Sigma))>0, Y<0 --> error
  if (ypl <= 0.)
    FOUR_C_THROW(
        "negative effective yield stress! Thermal softening and large temperature increase???");

  // activity state check
  if (ypl < absetatr_H)
  {
    if (activity_state_[gp] == false)  // gp switches state
    {
      if (abs(ypl - absetatr_H) > as_convergence_tol * inityield() ||
          deltaDp->norm_inf() > as_convergence_tol * inityield() / cpl())
        *as_converged = false;
    }
    activity_state_[gp] = true;
    *active = true;
  }
  else
  {
    if (activity_state_[gp] == true)  // gp switches state
    {
      if (abs(ypl - absetatr_H) > as_convergence_tol * inityield() ||
          deltaDp->norm_inf() > as_convergence_tol * inityield() / cpl())
        *as_converged = false;
    }
    activity_state_[gp] = false;
    *active = false;
  }

  // check, if gp is at the corner of the root of the NCP function
  if (abs(ypl - absetatr_H) > as_convergence_tol * inityield() ||
      deltaDp->norm_inf() > as_convergence_tol * inityield() / cpl())
  { /* gp not at the corner point */
  }
  else
  {
    /* gp at the corner point --> elastic and plastic branch are equally valid, take the elastic
     * one*/
    *active = false;
    activity_state_[gp] = false;
  }

  // these cases have some terms in common
  if (*active || dDpHeta > 0. || deltaDp->norm_inf() > 0.)
  {
    // damping parameter apl
    double apl = 1.;
    if (ypl < abseta_H) apl = ypl / abseta_H;

    // eta_s to abbreviate calculation of the derivatives
    if (absetatr_H > 0.) eta_s_v.update((1. - s()) * ypl / absetatr_H, etatr_v, 1.);
    eta_s_v.update(apl * s(), eta_v, 1.);

    // matrix exponential derivative
    Core::LinAlg::Matrix<3, 3> tmp(*deltaDp);
    tmp.scale(-1.);
    Core::LinAlg::MatrixFunctErrorType exp_deriv_err_status{
        Core::LinAlg::MatrixFunctErrorType::no_errors};
    Core::LinAlg::Matrix<6, 6> Dexp =
        Core::LinAlg::sym_matrix_3x3_exp_1st_deriv(tmp, exp_deriv_err_status);
    FOUR_C_ASSERT_ALWAYS(exp_deriv_err_status == Core::LinAlg::MatrixFunctErrorType::no_errors,
        "Matrix exponential derivative evaluation failed!");

    // Derivative of inverse plastic deformation gradient
    Core::LinAlg::Matrix<9, 6> dFpiDdeltaDp(Core::LinAlg::Initialization::zero);
    for (int A = 0; A < 3; A++)
      for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++)
          for (int i = 0; i < 6; i++)
            if (i <= 2)
              dFpiDdeltaDp(
                  Core::LinAlg::Voigt::IndexMappings::non_symmetric_tensor_to_voigt9_index(A, a),
                  i) -=
                  last_plastic_defgrd_inverse_[gp](A, b) *
                  Dexp(Core::LinAlg::Voigt::IndexMappings::symmetric_tensor_to_voigt6_index(b, a),
                      i);
            else
              dFpiDdeltaDp(
                  Core::LinAlg::Voigt::IndexMappings::non_symmetric_tensor_to_voigt9_index(A, a),
                  i) -=
                  2. * last_plastic_defgrd_inverse_[gp](A, b) *
                  Dexp(Core::LinAlg::Voigt::IndexMappings::symmetric_tensor_to_voigt6_index(b, a),
                      i);

    // derivative of mandel stress
    // we spare the deviatoric projection of the mandel stress derivative to get the effective
    // stress derivative. It is enforced implicitly as the "detaddp" is always contracted with
    // deviatoric tensors.
    Core::LinAlg::Matrix<6, 6> detaddp;
    detaddp.multiply(*dMdFpinv, dFpiDdeltaDp);
    detaddp.update(-2. / 3. * kinhard(), pdev, 1.);
    dPK2dDp->multiply(*dPK2dFpinv, dFpiDdeltaDp);
    if (d_cauchy_ddp) d_cauchy_ddp->multiply(*d_cauchy_dFpi, dFpiDdeltaDp);

    // TSI
    if (dNCPdT != nullptr)
    {
      if (dis_mode() == TSI::Taylor_Quinney)
      {
        double plHeating = taylor_quinney() * eta_v_strainlike.dot(deltaDp_v);
        Core::LinAlg::Matrix<6, 1> dHpDeta(Core::LinAlg::Initialization::zero);
        dHpDeta.update(taylor_quinney(), deltaDp_v_strainlike, 1.);
        dHdC->multiply_tn(*dMdC, dHpDeta);
        dHdDp->multiply_tn(detaddp, dHpDeta);
        dHdDp->update(taylor_quinney(), eta_v_strainlike, 1.);
        plHeating /= dt;
        dHdC->scale(1. / dt);
        dHdDp->scale(1. / dt);
        hep_diss(gp) += plHeating;
      }
      else
      {
        // plastic heating
        double plHeating = (0. - isohard() * hard_soft() * aI -
                               (infyield() * hard_soft() - inityield() * yield_soft()) *
                                   (1. - exp(-expisohard() * aI))) *
                           (*temp) * delta_alpha_i_[gp];
        switch (dis_mode())
        {
          case TSI::pl_multiplier:
            plHeating += delta_alpha_i_[gp] * (0. + inityield() * (1. - yield_soft() * dT) +
                                                  isohard() * (1. - hard_soft() * dT) * aI +
                                                  (infyield() * (1. - hard_soft() * dT) -
                                                      inityield() * (1. - yield_soft() * dT)) *
                                                      (1. - exp(-expisohard() * aI)));
            break;
          case TSI::pl_flow:
            plHeating += eta_v_strainlike.dot(deltaDp_v);
            break;
          default:
            FOUR_C_THROW("unknown plastic dissipation mode: {}", dis_mode());
            break;
        }

        // derivative w.r.t. temperature
        double dPlHeatingDT = (0. - isohard() * hard_soft() * aI +
                                  (infyield() * (-hard_soft()) - inityield() * (-yield_soft())) *
                                      (1. - exp(-expisohard() * aI))) *
                              delta_alpha_i_[gp];
        switch (dis_mode())
        {
          case TSI::pl_multiplier:
            dPlHeatingDT += -delta_alpha_i_[gp] *
                            (0. + inityield() * yield_soft() + isohard() * hard_soft() * aI +
                                (infyield() * hard_soft() - inityield() * yield_soft()) *
                                    (1. - exp(-expisohard() * aI)));
            break;
          case TSI::pl_flow:
            // do nothing
            break;
          default:
            FOUR_C_THROW("unknown plastic dissipation mode: {}", dis_mode());
            break;
        }

        // derivative w.r.t. Delta alpha i
        double dPlHeatingDdai =
            (*temp) * (0. - isohard() * hard_soft() * aI +
                          (infyield() * (-hard_soft()) - inityield() * (-yield_soft())) *
                              (1. - exp(-expisohard() * aI))) +
            (*temp) * delta_alpha_i_[gp] *
                (0. - isohard() * hard_soft() +
                    (-infyield() * hard_soft() + inityield() * yield_soft()) * expisohard() *
                        exp(-expisohard() * aI));
        switch (dis_mode())
        {
          case TSI::pl_multiplier:
            dPlHeatingDdai +=
                +inityield() * (1. - yield_soft() * dT) +
                isohard() * (1. - hard_soft() * dT) *
                    (last_alpha_isotropic_[gp] + 2. * delta_alpha_i_[gp]) +
                (infyield() * (1. - hard_soft() * dT) - inityield() * (1. - yield_soft() * dT)) *
                    ((1. - exp(-expisohard() * aI)) +
                        delta_alpha_i_[gp] * expisohard() * exp(-expisohard() * aI));
            break;
          case TSI::pl_flow:
            // do nothing
            break;
          default:
            FOUR_C_THROW("unknown plastic dissipation mode: {}", dis_mode());
            break;
        }

        // this factor is from the evolution equation for delta_alpha_i
        dPlHeatingDdai *= sq;

        // derivative w.r.t. eta
        Core::LinAlg::Matrix<6, 1> dHpDeta(Core::LinAlg::Initialization::zero);
        if (dDpHeta > 0.)
        {
          tmp61.multiply(PlAniso_full_, eta_v_strainlike);
          dHpDeta.update(dPlHeatingDdai * dDpHeta / (abseta_H * absHeta * absHeta), tmp61, 1.);
          dHpDeta.update(dPlHeatingDdai * abseta_H / (absHeta * absHeta), HdDp_strainlike, 1.);
          dHpDeta.update(
              -2. * dPlHeatingDdai * abseta_H * dDpHeta / (pow(absHeta, 4.)), HetaH_strainlike, 1.);
        }

        if (dis_mode() == TSI::pl_flow) dHpDeta.update(1., deltaDp_v_strainlike, 1.);

        // derivative w.r.t. C
        dHdC->multiply_tn(*dMdC, dHpDeta);

        // derivative w.r.t. Delta Dp
        dHdDp->multiply_tn(detaddp, dHpDeta);
        if (dDpHeta > 0.)
        {
          tmp61.multiply(PlAniso_full_, eta_v_strainlike);
          dHdDp->update(dPlHeatingDdai * abseta_H / (absHeta * absHeta), tmp61, 1.);
        }
        if (dis_mode() == TSI::pl_flow) dHdDp->update(1., eta_v_strainlike, 1.);

        // scaling with time step
        plHeating /= dt;
        dPlHeatingDT /= dt;
        dHdC->scale(1. / dt);
        dHdDp->scale(1. / dt);

        // communicate to the element via params (not nice)
        hep_diss(gp) += plHeating;
        (*dHepDissdT_)[gp] += dPlHeatingDT;
      }
    }  // TSI

    // plastic gp
    if (*active)
    {
      // this is a plastic gp
      *elast = false;

      // derivative of the complementarity function w.r.t. to the mandel stress tensor
      Core::LinAlg::Matrix<6, 6> dNCPdeta;
      dNCPdeta.update(1. - ypl / absetatr_H, pdev, 1.);
      tmp61.multiply(PlAniso_full_, etatr_v_strainlike);
      dNCPdeta.multiply_nt(1. / (absetatr_H * absetatr_H), eta_s_v, tmp61, 1.);
      if (dDpHeta > 0.)
      {
        tmp61.multiply(PlAniso_full_, eta_v_strainlike);
        dNCPdeta.multiply_nt(
            -dYplDai * dDpHeta / (abseta_H * absHeta * absHeta * absetatr_H), etatr_v, tmp61, 1.);
        dNCPdeta.multiply_nt(
            -dYplDai * abseta_H / (absHeta * absHeta * absetatr_H), etatr_v, HdDp_strainlike, 1.);
        dNCPdeta.multiply_nt(2. * dYplDai * abseta_H * dDpHeta / (pow(absHeta, 4.) * absetatr_H),
            etatr_v, HetaH_strainlike, 1.);
      }

      // derivative w.r.t. C
      dNCPdC->multiply(dNCPdeta, *dMdC);

      // derivative w.r.t. deltaDp
      dNCPdDp->multiply(dNCPdeta, detaddp);
      Core::LinAlg::Matrix<6, 6> dNCPdetatr;
      tmp61.multiply(PlAniso_full_, etatr_v_strainlike);
      dNCPdetatr.multiply_nt(cpl() / (absetatr_H * absetatr_H), eta_s_v, tmp61, 1.);
      dNCPdetatr.update(-cpl() * ypl / absetatr_H, pdev, 1.);
      dNCPdDp->multiply(1., dNCPdetatr, InvPlAniso_full_, 1.);
      if (dDpHeta > 0.)
      {
        tmp61.multiply(PlAniso_full_, eta_v_strainlike);
        dNCPdDp->multiply_nt(
            -dYplDai * abseta_H / (absetatr_H * absHeta * absHeta), etatr_v, tmp61, 1.);
      }

      // residual
      NCP->update(eta_v);
      NCP->update(-ypl / absetatr_H, etatr_v, 1.);

      // derivative w.r.t. temperature
      if (dNCPdT != nullptr) dNCPdT->update(-dYpldT / absetatr_H, etatr_v, 0.);
    }

    // not active but needs condensation due to activity in last iteration
    else if (dDpHeta > 0.)
    {
      // this is an elastic gp, but the flag "elast" is reserved for those
      // elastic GP that do not require a condensation
      *elast = false;

      // residual
      NCP->multiply(-cpl(), InvPlAniso_full_, deltaDp_v, 1.);

      // derivative of the complementarity function w.r.t. to the mandel stress tensor
      Core::LinAlg::Matrix<6, 6> dNCPdeta;

      tmp61.multiply(PlAniso_full_, eta_v_strainlike);
      dNCPdeta.multiply_nt(
          -s() * dYplDai * dDpHeta / (abseta_H * absHeta * absHeta * ypl), *NCP, tmp61, 1.);
      dNCPdeta.multiply_nt(
          -s() * dYplDai * abseta_H / (absHeta * absHeta * ypl), *NCP, HdDp_strainlike, 1.);
      dNCPdeta.multiply_nt(s() * 2. * dYplDai * abseta_H * dDpHeta / (pow(absHeta, 4.) * ypl), *NCP,
          HetaH_strainlike, 1.);

      // derivative w.r.t. C
      dNCPdC->multiply(dNCPdeta, *dMdC);

      // derivative w.r.t. deltaDp
      dNCPdDp->update(-cpl(), InvPlAniso_full_, 1.);
      dNCPdDp->multiply(1., dNCPdeta, detaddp, 1.);
      tmp61.multiply(PlAniso_full_, eta_v_strainlike);
      dNCPdDp->multiply_nt(dYplDai / (ypl * absHeta), *NCP, tmp61, 1.);

      // derivative w.r.t. temperature
      if (dNCPdT != nullptr)
        dNCPdT->multiply(-s() * cpl() / ypl * dYpldT, InvPlAniso_full_, deltaDp_v, 0.);
    }
    else
    {
      // Cpl = cpl* Delta D^p = 0
      // we don't build the matrix blocks here.
      // The trivial identity is enforced at the element
      *elast = true;
      // todo: currently, the coupling term dypldT is neglected
      // in the elastic realm. this is a slightly inconsistent linearization for s!=0
      // however, that way we can ensure that deltaLp=0 (exactly) at any Newton iteration

      if (dNCPdT != nullptr) dNCPdT->clear();
    }
  }
  // elastic gp
  else
  {
    *elast = true;
  }

  return;
}


/*----------------------------------------------------------------------*
 |  evaluate plastic stress and stiffness (with pl. spin)   seitz 05/14 |
 *----------------------------------------------------------------------*/
void Mat::PlasticElastHyper::evaluate_plast(const Core::LinAlg::Matrix<3, 3>* defgrd,
    const Core::LinAlg::Matrix<3, 3>* deltaLp, const double* temp,
    const Teuchos::ParameterList& params, Core::LinAlg::Matrix<6, 9>* dPK2dLp,
    Core::LinAlg::Matrix<9, 1>* NCP, Core::LinAlg::Matrix<9, 6>* dNCPdC,
    Core::LinAlg::Matrix<9, 9>* dNCPdLp, bool* active, bool* elast, bool* as_converged,
    const int gp, Core::LinAlg::Matrix<9, 1>* dNCPdT, Core::LinAlg::Matrix<6, 1>* dHdC,
    Core::LinAlg::Matrix<9, 1>* dHdLp, const double dt, const int eleGID,
    Core::LinAlg::Matrix<6, 1>* cauchy, Core::LinAlg::Matrix<6, 9>* d_cauchy_ddp,
    Core::LinAlg::Matrix<6, 6>* d_cauchy_dC, Core::LinAlg::Matrix<6, 9>* d_cauchy_dF,
    Core::LinAlg::Matrix<6, 1>* d_cauchy_dT)
{
  int check = +(cauchy != nullptr) + (d_cauchy_dC != nullptr) + (d_cauchy_dF != nullptr) +
              (d_cauchy_ddp != nullptr) + (d_cauchy_dT != nullptr);
  if (!(check == 0 || check == 5)) FOUR_C_THROW("some inconsistency with provided variables");

  Core::LinAlg::Matrix<3, 1> dPI(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6, 1> ddPII(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> gamma(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<8, 1> delta(Core::LinAlg::Initialization::zero);

  if (evaluate_kin_quant_plast(defgrd, deltaLp, gp, params)) return;

  elast_hyper_evaluate_invariant_derivatives(
      prinv_, dPI, ddPII, potsum_, summandProperties_, gp, eleGID);
  calculate_gamma_delta(gamma, delta, prinv_, dPI, ddPII);

  // blank resulting quantities
  // ... even if it is an implicit law that cmat is zero upon input
  dPK2dLp->clear();
  NCP->clear();
  dNCPdC->clear();
  dNCPdLp->clear();
  if (dNCPdT != nullptr) dNCPdT->clear();

  // new temporary matrices
  Core::LinAlg::Matrix<3, 3> mStr;  // Mandel stress tensor
  Core::LinAlg::Matrix<6, 6> dMdC;  // derivative of Mandel stress w.r.t. RCG
  Core::LinAlg::Matrix<6, 9>
      dMdFpinv;  // derivative of Mandel stress w.r.t. inverse plastic deformation gradient
  Core::LinAlg::Matrix<6, 9> dPK2dFpinv;
  Core::LinAlg::Matrix<6, 9> d_cauchy_dFpi;

  // isotropic elasticity in coupled strain energy format
  // isotropic elasticity in decoupled ("mod") format go here as well
  // as the modified gammas and deltas have been converted
  if (summandProperties_.isoprinc || summandProperties_.isomod)
  {
    evaluate_isotropic_princ_plast(dPK2dFpinv, mStr, dMdC, dMdFpinv, gamma, delta);
  }
  else
    FOUR_C_THROW("only isotropic hypereleastic materials");

  if (cauchy)
    evaluate_cauchy_plast(dPI, ddPII, defgrd, *cauchy, d_cauchy_dFpi, *d_cauchy_dC, *d_cauchy_dF);

  evaluate_nc_pand_spin(&mStr, &dMdC, &dMdFpinv, &dPK2dFpinv, deltaLp, gp, NCP, dNCPdC, dNCPdLp,
      dPK2dLp, active, elast, as_converged, dt, &d_cauchy_dFpi, d_cauchy_ddp);

  return;
}

/*----------------------------------------------------------------------*
 |  evaluate NCP function and plastic spin equation         seitz 05/14 |
 *----------------------------------------------------------------------*/
void Mat::PlasticElastHyper::evaluate_nc_pand_spin(const Core::LinAlg::Matrix<3, 3>* mStr,
    const Core::LinAlg::Matrix<6, 6>* dMdC, const Core::LinAlg::Matrix<6, 9>* dMdFpinv,
    const Core::LinAlg::Matrix<6, 9>* dPK2dFpinv, const Core::LinAlg::Matrix<3, 3>* deltaLp,
    const int gp, Core::LinAlg::Matrix<9, 1>* NCP, Core::LinAlg::Matrix<9, 6>* dNCPdC,
    Core::LinAlg::Matrix<9, 9>* dNCPdLp, Core::LinAlg::Matrix<6, 9>* dPK2dLp, bool* active,
    bool* elast, bool* as_converged, const double dt,
    const Core::LinAlg::Matrix<6, 9>* d_cauchy_dFpi, Core::LinAlg::Matrix<6, 9>* d_cauchy_ddp)
{
  const double sq = sqrt(2. / 3.);
  Core::LinAlg::Matrix<6, 1> tmp61;

  // deviatoric projection tensor
  Core::LinAlg::Matrix<6, 6> pdev(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      if (i == j)
        pdev(i, j) = +2. / 3.;
      else
        pdev(i, j) = -1. / 3.;
  for (int i = 3; i < 6; i++) pdev(i, i) = 1.;

  // deviatoric symmetric projection tensor (A-->dev(sym(A))
  Core::LinAlg::Matrix<6, 9> psymdev(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      if (i == j)
        psymdev(i, j) = +2. / 3.;
      else
        psymdev(i, j) = -1. / 3.;
  for (int i = 3; i < 6; i++) psymdev(i, i) = psymdev(i, i + 3) = .5;

  // symmetric identity
  Core::LinAlg::Matrix<6, 9> psym(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++) psym(i, i) = 1.;
  for (int i = 3; i < 6; i++) psym(i, i) = psym(i, i + 3) = .5;


  // effective stress
  Core::LinAlg::Matrix<3, 3> eta(*mStr);
  for (int i = 0; i < 3; i++)
    eta(i, i) -= 1. / 3. * ((*mStr)(0, 0) + (*mStr)(1, 1) + (*mStr)(2, 2));
  eta.update(2. / 3. * kinhard(), last_alpha_kinematic_[gp], 1.);
  eta.update(-1. / 3. * kinhard(), *deltaLp, 1.);
  eta.update_t(-1. / 3. * kinhard(), *deltaLp, 1.);

  // in stress-like voigt notation
  Core::LinAlg::Matrix<6, 1> eta_v;      // in stress-like voigt notation
  Core::LinAlg::Matrix<6, 1> etatr_v;    // in stress-like voigt notation
  Core::LinAlg::Matrix<6, 1> eta_s_v;    // in stress-like voigt notation
  Core::LinAlg::Matrix<6, 1> deltaDp_v;  // in stress-like voigt notation
  for (int i = 0; i < 3; i++)
  {
    eta_v(i) = eta(i, i);
    deltaDp_v(i) = (*deltaLp)(i, i);
  }
  eta_v(3) = .5 * (eta(1, 0) + eta(0, 1));
  deltaDp_v(3) = .5 * ((*deltaLp)(0, 1) + (*deltaLp)(1, 0));
  eta_v(4) = .5 * (eta(1, 2) + eta(2, 1));
  deltaDp_v(4) = .5 * ((*deltaLp)(2, 1) + (*deltaLp)(1, 2));
  eta_v(5) = .5 * (eta(2, 0) + eta(0, 2));
  deltaDp_v(5) = .5 * ((*deltaLp)(0, 2) + (*deltaLp)(2, 0));

  // trial effective stress
  etatr_v.update(eta_v);
  etatr_v.multiply(cpl(), InvPlAniso_full_, deltaDp_v, 1.);

  // in strain-like voigt notation
  Core::LinAlg::Matrix<6, 1> eta_v_strainlike(eta_v);          // in strain-like voigt notation
  Core::LinAlg::Matrix<6, 1> etatr_v_strainlike(etatr_v);      // in strain-like voigt notation
  Core::LinAlg::Matrix<6, 1> deltaDp_v_strainlike(deltaDp_v);  // in strain-like voigt notation
  for (int i = 3; i < 6; i++)
  {
    eta_v_strainlike(i) *= 2.;
    etatr_v_strainlike(i) *= 2.;
    deltaDp_v_strainlike(i) *= 2.;
  }

  tmp61.multiply(PlAniso_full_, eta_v);
  double absHeta = norm_stress_like(tmp61);
  double abseta_H = tmp61.dot(eta_v_strainlike);
  if (abseta_H < -1.e-16)
    FOUR_C_THROW("this should not happen. tmp={}", abseta_H);
  else if (abseta_H >= 0.)
    abseta_H = sqrt(abseta_H);
  else
    FOUR_C_THROW("this should not happen. tmp={}", abseta_H);
  double dDpHeta = tmp61.dot(deltaDp_v_strainlike);
  tmp61.multiply(PlAniso_full_, etatr_v);
  double absetatr_H = tmp61.dot(etatr_v_strainlike);
  if (absetatr_H < -1.e-16)
    FOUR_C_THROW("this should not happen. tmp={}", absetatr_H);
  else if (absetatr_H >= 0.)
    absetatr_H = sqrt(absetatr_H);
  else
    FOUR_C_THROW("this should not happen. tmp={}", absetatr_H);
  Core::LinAlg::Matrix<6, 1> HdDp;
  HdDp.multiply(PlAniso_full_, deltaDp_v);
  Core::LinAlg::Matrix<6, 1> HdDp_strainlike;
  HdDp_strainlike.multiply(PlAniso_full_, deltaDp_v_strainlike);
  Core::LinAlg::Matrix<6, 1> HetaH_strainlike;
  tmp61.multiply(PlAniso_full_, eta_v_strainlike);
  HetaH_strainlike.multiply(PlAniso_full_, tmp61);

  // isotropic hardening increment
  delta_alpha_i_[gp] = 0.;
  if (dDpHeta > 0. && absHeta > 0.)
    delta_alpha_i_[gp] = sq * dDpHeta * abseta_H / (absHeta * absHeta);

  // new isotropic hardening value
  const double aI = last_alpha_isotropic_[gp] + delta_alpha_i_[gp];

  // current yield stress equivalent (yield stress scaled by sqrt(2/3))
  double ypl =
      sq *
      ((infyield() - inityield()) * (1. - exp(-expisohard() * aI)) + isohard() * aI + inityield()) *
      pow(1. + visc() * delta_alpha_i_[gp] / dt, visc_rate());

  // check activity state
  if (ypl < absetatr_H)
  {
    if (activity_state_[gp] == false)  // gp switches state
    {
      if (abs(ypl - absetatr_H) > as_convergence_tol * inityield() ||
          deltaLp->norm_inf() > as_convergence_tol * inityield() / cpl())
        *as_converged = false;
    }
    activity_state_[gp] = true;
    *active = true;
  }
  else
  {
    if (activity_state_[gp] == true)  // gp switches state
    {
      if (abs(ypl - absetatr_H) > as_convergence_tol * inityield() ||
          deltaLp->norm_inf() > as_convergence_tol * inityield() / cpl())
        *as_converged = false;
    }
    activity_state_[gp] = false;
    *active = false;
  }

  // check, if gp is at the corner of the root of the NCP function
  if (abs(ypl - absetatr_H) > as_convergence_tol * inityield() ||
      deltaLp->norm_inf() > as_convergence_tol * inityield() / cpl())
  { /* gp not at the corner point */
  }
  else
  {
    /* gp at the corner point --> elastic and plastic branch are equally valid, take the elastic
     * one*/
    *active = false;
    activity_state_[gp] = false;
  }

  // these cases have some terms in common
  if (*active || dDpHeta > 0. || deltaLp->norm_inf() > 0.)
  {
    // derivative of the NCP function w.r.t. RCG / Delta Lp
    // without the lines corresponding to the plastic spin
    Core::LinAlg::Matrix<6, 6> dNCPdC_red;
    Core::LinAlg::Matrix<6, 9> dNCPdLp_red;
    Core::LinAlg::Matrix<6, 1> NCP_red;

    // damping parameter apl
    double apl = 1.;
    if (ypl / abseta_H < 1.) apl = ypl / abseta_H;

    eta_s_v.update((1. - s()) * ypl / absetatr_H, etatr_v, 1.);
    eta_s_v.update(apl * s(), eta_v, 1.);

    // matrix exponential derivative
    Core::LinAlg::Matrix<3, 3> tmp(*deltaLp);
    tmp.scale(-1.);
    Core::LinAlg::MatrixFunctErrorType exp_deriv_err_status{
        Core::LinAlg::MatrixFunctErrorType::no_errors};
    Core::LinAlg::Matrix<9, 9> Dexp =
        Core::LinAlg::matrix_3x3_exp_1st_deriv(tmp, exp_deriv_err_status);
    FOUR_C_ASSERT_ALWAYS(exp_deriv_err_status == Core::LinAlg::MatrixFunctErrorType::no_errors,
        "Matrix exponential derivative failed!");

    // Derivative of inverse plastic deformation gradient
    Core::LinAlg::Matrix<9, 9> dFpiDdeltaLp(Core::LinAlg::Initialization::zero);
    for (int A = 0; A < 3; A++)
      for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++)
          for (int i = 0; i < 9; i++)
            dFpiDdeltaLp(
                Core::LinAlg::Voigt::IndexMappings::non_symmetric_tensor_to_voigt9_index(A, a),
                i) -=
                last_plastic_defgrd_inverse_[gp](A, b) *
                Dexp(Core::LinAlg::Voigt::IndexMappings::non_symmetric_tensor_to_voigt9_index(b, a),
                    i);

    // derivative of mandel stress
    Core::LinAlg::Matrix<6, 9> dMdLp;
    dMdLp.multiply(*dMdFpinv, dFpiDdeltaLp);
    // we spare the deviatoric projection of the mandel stress derivative to get the effective
    // stress derivative. It is enforced implicitly as the "detadLp" is always contracted with
    // deviatoric tensors.
    Core::LinAlg::Matrix<6, 9> detadLp(dMdLp);
    detadLp.update(-2. / 3. * kinhard(), psymdev, 1.);
    dPK2dLp->multiply(*dPK2dFpinv, dFpiDdeltaLp);
    if (d_cauchy_ddp) d_cauchy_ddp->multiply(*d_cauchy_dFpi, dFpiDdeltaLp);

    // Factor of derivative of Y^pl w.r.t. delta alpha ^i
    // we have added the factor sqrt(2/3) from delta_alpha_i=sq*... here
    double dYplDai =
        2. / 3. *
        (+isohard() + (infyield() - inityield()) * expisohard() * exp(-expisohard() * aI)) *
        pow(1. + visc() * delta_alpha_i_[gp] / dt, visc_rate());
    dYplDai += 2. / 3. *
               ((infyield() - inityield()) * (1. - exp(-expisohard() * aI)) + isohard() * aI +
                   inityield()) *
               pow(1. + visc() * delta_alpha_i_[gp] / dt, visc_rate() - 1) * visc_rate() * visc() /
               dt;

    // plastic gp
    if (*active)
    {
      // this is a plastic gp
      *elast = false;

      // derivative of the complementarity function w.r.t. to the mandel stress tensor
      Core::LinAlg::Matrix<6, 6> dNCPdeta;
      dNCPdeta.update(1. - ypl / absetatr_H, pdev, 1.);
      tmp61.multiply(PlAniso_full_, etatr_v_strainlike);
      dNCPdeta.multiply_nt(1. / (absetatr_H * absetatr_H), eta_s_v, tmp61, 1.);
      if (dDpHeta > 0.)
      {
        tmp61.multiply(PlAniso_full_, eta_v_strainlike);
        dNCPdeta.multiply_nt(
            -dYplDai * dDpHeta / (abseta_H * absHeta * absHeta * absetatr_H), etatr_v, tmp61, 1.);
        dNCPdeta.multiply_nt(
            -dYplDai * abseta_H / (absHeta * absHeta * absetatr_H), etatr_v, HdDp_strainlike, 1.);
        dNCPdeta.multiply_nt(2. * dYplDai * abseta_H * dDpHeta / (pow(absHeta, 4.) * absetatr_H),
            etatr_v, HetaH_strainlike, 1.);
      }

      // derivative w.r.t. C
      dNCPdC_red.multiply(dNCPdeta, *dMdC);

      // derivative w.r.t. deltaLp
      dNCPdLp_red.multiply(dNCPdeta, detadLp);
      Core::LinAlg::Matrix<6, 6> dNCPdetatr;
      tmp61.multiply(PlAniso_full_, etatr_v_strainlike);
      dNCPdetatr.multiply_nt(cpl() / (absetatr_H * absetatr_H), eta_s_v, tmp61, 1.);
      dNCPdetatr.update(-cpl() * ypl / absetatr_H, pdev, 1.);
      Core::LinAlg::Matrix<6, 6> dNCPdDp;
      dNCPdDp.multiply(dNCPdetatr, InvPlAniso_full_);
      if (dDpHeta > 0.)
      {
        tmp61.multiply(PlAniso_full_, eta_v_strainlike);
        dNCPdDp.multiply_nt(
            -dYplDai * abseta_H / (absetatr_H * absHeta * absHeta), etatr_v, tmp61, 1.);
      }
      dNCPdLp_red.multiply(1., dNCPdDp, psym, 1.);

      // residual
      NCP_red.update(eta_v);
      NCP_red.update(-ypl / absetatr_H, etatr_v, 1.);
    }

    // not active but needs condensation due to activity in last iteration
    else if (dDpHeta > 0.)
    {
      // this is an elastic gp, but the flag "elast" is reserved for those
      // elastic GP that do not require a condensation
      *elast = false;

      // residual
      NCP_red.multiply(-cpl(), InvPlAniso_full_, deltaDp_v, 1.);

      // derivative of the complementarity function w.r.t. to the mandel stress tensor
      Core::LinAlg::Matrix<6, 6> dNCPdeta;

      tmp61.multiply(PlAniso_full_, eta_v_strainlike);
      dNCPdeta.multiply_nt(
          -s() * dYplDai * dDpHeta / (abseta_H * absHeta * absHeta * ypl), NCP_red, tmp61, 1.);
      dNCPdeta.multiply_nt(
          -s() * dYplDai * abseta_H / (absHeta * absHeta * ypl), NCP_red, HdDp_strainlike, 1.);
      dNCPdeta.multiply_nt(s() * 2. * dYplDai * abseta_H * dDpHeta / (pow(absHeta, 4.) * ypl),
          NCP_red, HetaH_strainlike, 1.);

      // derivative w.r.t. C
      dNCPdC_red.multiply(dNCPdeta, *dMdC);

      Core::LinAlg::Matrix<6, 6> dNCPdDp;
      dNCPdDp.update(-cpl(), InvPlAniso_full_, 1.);
      tmp61.multiply(PlAniso_full_, eta_v_strainlike);
      dNCPdDp.multiply_nt(dYplDai / (ypl * absHeta), NCP_red, tmp61, 1.);
      dNCPdLp_red.multiply(1., dNCPdDp, psym, 1.);
      dNCPdLp_red.multiply(1., dNCPdeta, detadLp, 1.);
    }
    else
    {
      // Cpl = cpl* Delta D^p = 0
      // we don't build the matrix blocks here.
      // The trivial identity is enforced at the element
      *elast = true;
    }

    // plastic spin equation
    // the tensor product Sigma.Dp is not made for voigt notation so we do it the hard way
    Core::LinAlg::Matrix<3, 1> spEq;
    spEq(0) = .5 * ((*deltaLp)(0, 1) - (*deltaLp)(1, 0)) -
              pl_spin_chi() / inityield() *
                  ((*mStr)(0, 0) * deltaDp_v(3) + (*mStr)(0, 1) * deltaDp_v(1) +
                      (*mStr)(0, 2) * deltaDp_v(4) - (*mStr)(0, 1) * deltaDp_v(0) -
                      (*mStr)(1, 1) * deltaDp_v(3) - (*mStr)(1, 2) * deltaDp_v(5));
    spEq(1) = .5 * ((*deltaLp)(1, 2) - (*deltaLp)(2, 1)) -
              pl_spin_chi() / inityield() *
                  ((*mStr)(0, 1) * deltaDp_v(5) + (*mStr)(1, 1) * deltaDp_v(4) +
                      (*mStr)(1, 2) * deltaDp_v(2) - (*mStr)(0, 2) * deltaDp_v(3) -
                      (*mStr)(1, 2) * deltaDp_v(1) - (*mStr)(2, 2) * deltaDp_v(4));
    spEq(2) = .5 * ((*deltaLp)(0, 2) - (*deltaLp)(2, 0)) -
              pl_spin_chi() / inityield() *
                  ((*mStr)(0, 0) * deltaDp_v(5) + (*mStr)(0, 1) * deltaDp_v(4) +
                      (*mStr)(0, 2) * deltaDp_v(2) - (*mStr)(0, 2) * deltaDp_v(0) -
                      (*mStr)(1, 2) * deltaDp_v(3) - (*mStr)(2, 2) * deltaDp_v(5));

    // Derivative of plastic spin equation w.r.t. mandel stress
    Core::LinAlg::Matrix<3, 6> dSpdM;
    dSpdM(0, 0) = +deltaDp_v(3);
    dSpdM(0, 1) = -deltaDp_v(3);
    dSpdM(0, 2) = 0.;
    dSpdM(0, 3) = +deltaDp_v(1) - deltaDp_v(0);
    dSpdM(0, 4) = -deltaDp_v(5);
    dSpdM(0, 5) = deltaDp_v(4);
    dSpdM(1, 0) = 0.;
    dSpdM(1, 1) = deltaDp_v(4);
    dSpdM(1, 2) = -deltaDp_v(4);
    dSpdM(1, 3) = deltaDp_v(5);
    dSpdM(1, 4) = deltaDp_v(2) - deltaDp_v(1);
    dSpdM(1, 5) = -deltaDp_v(3);
    dSpdM(2, 0) = deltaDp_v(5);
    dSpdM(2, 1) = 0.;
    dSpdM(2, 2) = -deltaDp_v(5);
    dSpdM(2, 3) = deltaDp_v(4);
    dSpdM(2, 4) = -deltaDp_v(3);
    dSpdM(2, 5) = deltaDp_v(2) - deltaDp_v(0);
    dSpdM.scale(-pl_spin_chi() / inityield());

    // derivative of plastic spin equation w.r.t. deltaDp
    Core::LinAlg::Matrix<3, 6> dSpdDp;
    dSpdDp(0, 0) = -(*mStr)(0, 1);
    dSpdDp(0, 1) = (*mStr)(0, 1);
    dSpdDp(0, 2) = 0.;
    dSpdDp(0, 3) = (*mStr)(0, 0) - (*mStr)(1, 1);
    dSpdDp(0, 4) = +(*mStr)(0, 2);
    dSpdDp(0, 5) = -(*mStr)(1, 2);
    dSpdDp(1, 0) = 0.;
    dSpdDp(1, 1) = -(*mStr)(1, 2);
    dSpdDp(1, 2) = +(*mStr)(1, 2);
    dSpdDp(1, 3) = -(*mStr)(0, 2);
    dSpdDp(1, 4) = +(*mStr)(1, 1) - (*mStr)(2, 2);
    dSpdDp(1, 5) = (*mStr)(0, 1);
    dSpdDp(2, 0) = -(*mStr)(0, 2);
    dSpdDp(2, 1) = 0.;
    dSpdDp(2, 2) = +(*mStr)(0, 2);
    dSpdDp(2, 3) = -(*mStr)(1, 2);
    dSpdDp(2, 4) = +(*mStr)(0, 1);
    dSpdDp(2, 5) = (*mStr)(0, 0) - (*mStr)(2, 2);
    dSpdDp.scale(-pl_spin_chi() / inityield());

    // derivative of plastic spin equation w.r.t. RCG
    Core::LinAlg::Matrix<3, 6> dSpdC;
    dSpdC.multiply(dSpdM, *dMdC);

    // derivative of plastic spin equation w.r.t. deltaLp
    Core::LinAlg::Matrix<3, 9> dSpddLp;
    dSpddLp.multiply(dSpdDp, psym);
    dSpddLp.multiply(1., dSpdM, dMdLp, 1.);
    dSpddLp(0, 3) += .5;
    dSpddLp(0, 6) -= .5;
    dSpddLp(1, 4) += .5;
    dSpddLp(1, 7) -= .5;
    dSpddLp(2, 5) += .5;
    dSpddLp(2, 8) -= .5;

    // combine NCP and plastic spin equations
    for (int i = 0; i < 6; i++) (*NCP)(i) = NCP_red(i);
    for (int i = 6; i < 9; i++) (*NCP)(i) = spEq(i - 6);

    for (int i = 0; i < 6; i++)
      for (int j = 0; j < 6; j++) (*dNCPdC)(i, j) = dNCPdC_red(i, j);
    for (int i = 6; i < 9; i++)
      for (int j = 0; j < 6; j++) (*dNCPdC)(i, j) = dSpdC(i - 6, j);

    for (int i = 0; i < 6; i++)
      for (int j = 0; j < 9; j++) (*dNCPdLp)(i, j) = dNCPdLp_red(i, j);
    for (int i = 6; i < 9; i++)
      for (int j = 0; j < 9; j++) (*dNCPdLp)(i, j) = dSpddLp(i - 6, j);
  }
  // elastic gp
  else
  {
    *elast = true;
  }

  return;
}

void Mat::PlasticElastHyper::evaluate_cauchy_plast(const Core::LinAlg::Matrix<3, 1>& dPI,
    const Core::LinAlg::Matrix<6, 1>& ddPII, const Core::LinAlg::Matrix<3, 3>* defgrd,
    Core::LinAlg::Matrix<6, 1>& cauchy, Core::LinAlg::Matrix<6, 9>& d_cauchy_dFpi,
    Core::LinAlg::Matrix<6, 6>& d_cauchy_dC, Core::LinAlg::Matrix<6, 9>& d_cauchy_dF,
    Core::LinAlg::Matrix<6, 1>* d_cauchy_dT)
{
  cauchy.clear();
  d_cauchy_dC.clear();
  d_cauchy_dF.clear();
  d_cauchy_dFpi.clear();

  cauchy.update(sqrt(prinv_(2)) * dPI(2), id2V_, 1.);
  cauchy.update((dPI(0) + prinv_(0) * dPI(1)) / sqrt(prinv_(2)), bev_, 1.);
  cauchy.update(-dPI(1) / sqrt(prinv_(2)), be2v_, 1.);
  cauchy.scale(2.);

  d_cauchy_dC.multiply_nt(sqrt(prinv_(2)) * (ddPII(4) + prinv_(0) * ddPII(3)), id2V_, Cpi_, 1.);
  d_cauchy_dC.multiply_nt(-sqrt(prinv_(2)) * ddPII(3), id2V_, CpiCCpi_, 1.);
  d_cauchy_dC.multiply_nt(sqrt(prinv_(2)) * (.5 * dPI(2) + prinv_(2) * ddPII(2)), id2V_, ircg_, 1.);

  d_cauchy_dC.multiply_nt(
      (ddPII(0) + dPI(1) + 2. * prinv_(0) * ddPII(5) + prinv_(0) * prinv_(0) * ddPII(1)) /
          sqrt(prinv_(2)),
      bev_, Cpi_, 1.);
  d_cauchy_dC.multiply_nt((-ddPII(5) - prinv_(0) * ddPII(1)) / sqrt(prinv_(2)), bev_, CpiCCpi_, 1.);
  d_cauchy_dC.multiply_nt((-.5 * dPI(0) - .5 * prinv_(0) * dPI(1) + prinv_(2) * ddPII(4) +
                              prinv_(0) * prinv_(2) * ddPII(3)) /
                              sqrt(prinv_(2)),
      bev_, ircg_, 1.);

  d_cauchy_dC.multiply_nt(-(ddPII(5) - prinv_(0) * ddPII(1)) / sqrt(prinv_(2)), be2v_, Cpi_, 1.);
  d_cauchy_dC.multiply_nt(ddPII(1) / sqrt(prinv_(2)), be2v_, CpiCCpi_, 1.);
  d_cauchy_dC.multiply_nt((.5 * dPI(1) / prinv_(2) - ddPII(3)) / sqrt(prinv_(2)), be2v_, ircg_, 1.);
  d_cauchy_dC.scale(4.);

  d_cauchy_dFpi.multiply_nt(sqrt(prinv_(2)) * (ddPII(4) + prinv_(0) * ddPII(3)), id2V_, CFpi_, 1.);
  d_cauchy_dFpi.multiply_nt(-sqrt(prinv_(2)) * ddPII(3), id2V_, CFpiCe_, 1.);
  d_cauchy_dFpi.multiply_nt(
      sqrt(prinv_(2)) * (.5 * dPI(2) + prinv_(2) * ddPII(2)), id2V_, CFpiCei_, 1.);

  d_cauchy_dFpi.multiply_nt(
      (ddPII(0) + dPI(1) + 2. * prinv_(0) * ddPII(5) + prinv_(0) * prinv_(0) * ddPII(1)) /
          sqrt(prinv_(2)),
      bev_, CFpi_, 1.);
  d_cauchy_dFpi.multiply_nt(
      (-ddPII(5) - prinv_(0) * ddPII(1)) / sqrt(prinv_(2)), bev_, CFpiCe_, 1.);
  d_cauchy_dFpi.multiply_nt((-.5 * dPI(0) - .5 * prinv_(0) * dPI(1) + prinv_(2) * ddPII(4) +
                                prinv_(0) * prinv_(2) * ddPII(3)) /
                                sqrt(prinv_(2)),
      bev_, CFpiCei_, 1.);

  d_cauchy_dFpi.multiply_nt(-(ddPII(5) - prinv_(0) * ddPII(1)) / sqrt(prinv_(2)), be2v_, CFpi_, 1.);
  d_cauchy_dFpi.multiply_nt(ddPII(1) / sqrt(prinv_(2)), be2v_, CFpiCe_, 1.);
  d_cauchy_dFpi.multiply_nt(
      (.5 * dPI(1) / prinv_(2) - ddPII(3)) / sqrt(prinv_(2)), be2v_, CFpiCei_, 1.);
  d_cauchy_dFpi.scale(2.);

  Core::LinAlg::FourTensorOperations::add_right_non_symmetric_holzapfel_product(
      d_cauchy_dFpi, *defgrd, Fe_, (dPI(0) + prinv_(0) * dPI(1)) / sqrt(prinv_(2)));
  Core::LinAlg::FourTensorOperations::add_right_non_symmetric_holzapfel_product(
      d_cauchy_dFpi, *defgrd, beFe_, -dPI(1) / sqrt(prinv_(2)));
  Core::LinAlg::FourTensorOperations::add_right_non_symmetric_holzapfel_product(
      d_cauchy_dFpi, beF_, Fe_, -dPI(1) / sqrt(prinv_(2)));

  Core::LinAlg::FourTensorOperations::add_right_non_symmetric_holzapfel_product(
      d_cauchy_dF, id2_, FCpi_, (dPI(0) + prinv_(0) * dPI(1)) / sqrt(prinv_(2)));
  Core::LinAlg::FourTensorOperations::add_right_non_symmetric_holzapfel_product(
      d_cauchy_dF, id2_, beFCpi_, -dPI(1) / sqrt(prinv_(2)));
  Core::LinAlg::FourTensorOperations::add_right_non_symmetric_holzapfel_product(
      d_cauchy_dF, be_, FCpi_, -dPI(1) / sqrt(prinv_(2)));
  d_cauchy_dF.scale(2.);
  d_cauchy_dFpi.scale(2.);

  if (d_cauchy_dT)
  {
    d_cauchy_dT->clear();

    const double j = sqrt(prinv_(2));
    double d_dPI2_dT = 0.;
    if (summandProperties_.isomod) FOUR_C_THROW("only coupled SEF are supposed to end up here");
    for (unsigned int p = 0; p < potsum_.size(); ++p)
      potsum_[p]->add_coup_deriv_vol(j, nullptr, &d_dPI2_dT, nullptr, nullptr);

    const double fac = -3. * cte();
    d_dPI2_dT *= fac * .5 / j;

    d_cauchy_dT->update(sqrt(prinv_(2)) * d_dPI2_dT, id2V_, 1.);
    d_cauchy_dT->scale(2.);
  }
}

void Mat::PlasticElastHyper::evaluate_cauchy_derivs(const Core::LinAlg::Matrix<3, 1>& prinv,
    const int gp, int eleGID, Core::LinAlg::Matrix<3, 1>& dPI, Core::LinAlg::Matrix<6, 1>& ddPII,
    Core::LinAlg::Matrix<10, 1>& dddPIII, const double* temp)
{
  for (unsigned i = 0; i < potsum_.size(); ++i)
  {
    if (summandProperties_.isoprinc)
    {
      potsum_[i]->add_derivatives_principal(dPI, ddPII, prinv, gp, eleGID);
      potsum_[i]->add_third_derivatives_principal_iso(dddPIII, prinv, gp, eleGID);
    }
    if (summandProperties_.isomod || summandProperties_.anisomod || summandProperties_.anisoprinc)
      FOUR_C_THROW("not implemented for this form of strain energy function");
  }
  if (temp)
  {
    if (summandProperties_.isomod || summandProperties_.anisomod || summandProperties_.anisoprinc)
      FOUR_C_THROW(
          "thermo-elastic Nitsche contact only with strain energies"
          "\ndepending on unmodified invariants");

    const double dT = *temp - init_temp();

    const double j = sqrt(prinv(2));
    double dPmodI = 0.;
    double ddPmodII = 0.;
    double dddPmodIII = 0.;
    double ddddPmodIIII = 0.;
    for (unsigned int p = 0; p < potsum_.size(); ++p)
      potsum_[p]->add_coup_deriv_vol(j, &dPmodI, &ddPmodII, &dddPmodIII, &ddddPmodIIII);

    const double fac = -3. * cte() * dT;
    dPI(2) += fac * .5 / j * ddPmodII;
    ddPII(2) += fac * (.25 / (j * j) * dddPmodIII - .25 / (j * j * j) * ddPmodII);
    dddPIII(2) +=
        fac * (+1. / (8. * j * j * j) * ddddPmodIIII - 3. / (8. * j * j * j * j) * dddPmodIII +
                  3. / (8. * j * j * j * j * j) * ddPmodII);
  }
}

void Mat::PlasticElastHyper::evaluate_cauchy_temp_deriv(const Core::LinAlg::Matrix<3, 1>& prinv,
    const double ndt, const double bdndt, const double ibdndt, const double* temp, double* DsntDT,
    const Core::LinAlg::Matrix<9, 1>& iFTV, const Core::LinAlg::Matrix<9, 1>& DbdndtDFV,
    const Core::LinAlg::Matrix<9, 1>& DibdndtDFV, const Core::LinAlg::Matrix<9, 1>& DI1DF,
    const Core::LinAlg::Matrix<9, 1>& DI2DF, const Core::LinAlg::Matrix<9, 1>& DI3DF,
    Core::LinAlg::Matrix<9, 1>* D2sntDFDT)
{
  const double sqI3 = sqrt(prinv(2));
  const double prefac = 2.0 / sqI3;
  double dPmodI = 0.0;
  double ddPmodII = 0.0;
  double dddPmodIII = 0.0;
  for (unsigned int p = 0; p < potsum_.size(); ++p)
    potsum_[p]->add_coup_deriv_vol(sqI3, &dPmodI, &ddPmodII, &dddPmodIII, nullptr);

  // those are actually the temperature derivatives of the coefficients
  // then we plug them into the cauchy stress derivative and voila
  // we keep many zero entries here for possible future extension to more
  // than just thermal expansion
  static Core::LinAlg::Matrix<3, 1> dPI(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<6, 1> ddPII(Core::LinAlg::Initialization::zero);
  dPI.clear();
  ddPII.clear();

  const double fac = -3.0 * cte();
  dPI(2) += fac * 0.5 / sqI3 * ddPmodII;
  ddPII(2) += fac * (0.25 / (sqI3 * sqI3) * dddPmodIII - 0.25 / (sqI3 * sqI3 * sqI3) * ddPmodII);

  *DsntDT = prefac * (prinv(1) * dPI(1) * ndt + prinv(2) * dPI(2) * ndt + dPI(0) * bdndt -
                         prinv(2) * dPI(1) * ibdndt);

  if (D2sntDFDT)
  {
    D2sntDFDT->update(-prefac * (prinv(1) * dPI(1) * ndt + prinv(2) * dPI(2) * ndt +
                                    dPI(0) * bdndt - prinv(2) * dPI(1) * ibdndt),
        iFTV, 0.0);  // D2sntDFDT is cleared here
    D2sntDFDT->update(prefac * dPI(0), DbdndtDFV, 1.0);
    D2sntDFDT->update(-prefac * prinv(2) * dPI(1), DibdndtDFV, 1.0);
    D2sntDFDT->update(prefac * (prinv(1) * ddPII(5) * ndt + prinv(2) * ddPII(4) * ndt +
                                   ddPII(0) * bdndt - prinv(2) * ddPII(5) * ibdndt),
        DI1DF, 1.0);
    D2sntDFDT->update(
        prefac * (dPI(1) * ndt + prinv(1) * ddPII(1) * ndt + prinv(2) * ddPII(3) * ndt +
                     ddPII(5) * bdndt - prinv(2) * ddPII(1) * ibdndt),
        DI2DF, 1.0);
    D2sntDFDT->update(
        prefac * (prinv(1) * ddPII(3) * ndt + dPI(2) * ndt + prinv(2) * ddPII(2) * ndt +
                     ddPII(4) * bdndt - dPI(1) * ibdndt - prinv(2) * ddPII(3) * ibdndt),
        DI3DF, 1.0);
  }
}


void Mat::PlasticElastHyper::add_thermal_expansion_derivs(const Core::LinAlg::Matrix<3, 1>& prinv,
    Core::LinAlg::Matrix<3, 1>& dPI, Core::LinAlg::Matrix<6, 1>& ddPII, const int gp, int eleGID,
    const double& temp)
{
  const double j = sqrt(prinv(2));
  Core::LinAlg::Matrix<3, 1> modinv(Core::LinAlg::Initialization::zero);
  modinv(2) = j;
  Core::LinAlg::Matrix<3, 1> dPmodI;
  Core::LinAlg::Matrix<6, 1> ddPmodII;
  double dddPmodIII = 0.;
  for (unsigned int p = 0; p < potsum_.size(); ++p)
  {
    potsum_[p]->add_derivatives_modified(dPmodI, ddPmodII, modinv, gp, eleGID);
    potsum_[p]->add3rd_vol_deriv(modinv, dddPmodIII);
    potsum_[p]->add_coup_deriv_vol(j, nullptr, &(ddPmodII(2)), &dddPmodIII, nullptr);
  }

  const double dT = temp - init_temp();
  const double fac = -3. * cte() * dT;
  dPI(2) += fac * .5 / j * ddPmodII(2);
  ddPII(2) += fac * (.25 / (j * j) * dddPmodIII - .25 / (j * j * j) * ddPmodII(2));
}

void Mat::PlasticElastHyper::update_gp(const int gp, const Core::LinAlg::Matrix<3, 3>* deltaDp)
{
  if (activity_state_[gp] == true)
  {
    // update plastic deformation gradient
    Core::LinAlg::Matrix<3, 3> tmp;
    tmp.update(-1., *deltaDp);
    Core::LinAlg::MatrixFunctErrorType exp_err_status{
        Core::LinAlg::MatrixFunctErrorType::no_errors};
    Core::LinAlg::Matrix<3, 3> exp_tmp = Core::LinAlg::matrix_exp(tmp, exp_err_status);
    FOUR_C_ASSERT_ALWAYS(exp_err_status == Core::LinAlg::MatrixFunctErrorType::no_errors,
        "Matrix exponential evaluation failed!");
    Core::LinAlg::Matrix<3, 3> fpi_last = last_plastic_defgrd_inverse_[gp];
    last_plastic_defgrd_inverse_[gp].multiply(fpi_last, exp_tmp);
    // update isotropic hardening
    last_alpha_isotropic_[gp] += delta_alpha_i_[gp];

    // update kinematic hardening
    last_alpha_kinematic_[gp].update(-.5, *deltaDp, 1.);
    last_alpha_kinematic_[gp].update_t(-.5, *deltaDp, 1.);
  }

  return;
}



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::PlasticElastHyper::evaluate_kin_quant_elast(const Core::LinAlg::Matrix<3, 3>* defgrd,
    const Core::LinAlg::Matrix<3, 3>* deltaLp, const int gp)
{
  Core::LinAlg::Matrix<3, 3> tmp;
  Core::LinAlg::Matrix<3, 3> invpldefgrd;
  Core::LinAlg::Matrix<3, 3>& InvPlasticDefgrdLast = last_plastic_defgrd_inverse_[gp];
  tmp.update(-1., *deltaLp);
  Core::LinAlg::MatrixFunctErrorType exp_err_status{Core::LinAlg::MatrixFunctErrorType::no_errors};
  Core::LinAlg::Matrix<3, 3> exp_tmp = Core::LinAlg::matrix_exp(tmp, exp_err_status);
  FOUR_C_ASSERT_ALWAYS(exp_err_status == Core::LinAlg::MatrixFunctErrorType::no_errors,
      "Matrix exponential evaluation failed!");
  invpldefgrd.multiply(InvPlasticDefgrdLast, exp_tmp);

  // inverse plastic right Cauchy-Green
  Core::LinAlg::Matrix<3, 3> CpiM;
  CpiM.multiply_nt(invpldefgrd, invpldefgrd);
  // stress-like Voigt notation
  for (int i = 0; i < 3; i++) Cpi_(i) = CpiM(i, i);
  Cpi_(3) = (CpiM(0, 1) + CpiM(1, 0)) / 2.;
  Cpi_(4) = (CpiM(2, 1) + CpiM(1, 2)) / 2.;
  Cpi_(5) = (CpiM(0, 2) + CpiM(2, 0)) / 2.;

  // inverse RCG
  Core::LinAlg::Matrix<3, 3> iRCG;
  Core::LinAlg::Matrix<3, 3> RCG;
  RCG.multiply_tn(*defgrd, *defgrd);
  iRCG.invert(RCG);
  // stress-like Voigt notation
  for (int i = 0; i < 3; i++) ircg_(i) = iRCG(i, i);
  ircg_(3) = (iRCG(0, 1) + iRCG(1, 0)) / 2.;
  ircg_(4) = (iRCG(2, 1) + iRCG(1, 2)) / 2.;
  ircg_(5) = (iRCG(0, 2) + iRCG(2, 0)) / 2.;

  // C_p^-1 * C * C_p^-1
  Core::LinAlg::Matrix<3, 3> CpiCCpiM;
  tmp.multiply(CpiM, RCG);
  CpiCCpiM.multiply(tmp, CpiM);
  // stress-like Voigt notation
  for (int i = 0; i < 3; i++) CpiCCpi_(i) = CpiCCpiM(i, i);
  CpiCCpi_(3) = (CpiCCpiM(0, 1) + CpiCCpiM(1, 0)) / 2.;
  CpiCCpi_(4) = (CpiCCpiM(2, 1) + CpiCCpiM(1, 2)) / 2.;
  CpiCCpi_(5) = (CpiCCpiM(0, 2) + CpiCCpiM(2, 0)) / 2.;

  tmp.multiply(*defgrd, invpldefgrd);
  Core::LinAlg::Matrix<3, 3> CeM;
  CeM.multiply_tn(tmp, tmp);
  // elastic right Cauchy-Green in strain-like Voigt notation.
  Core::LinAlg::Matrix<6, 1> elasticRCGv;
  for (int i = 0; i < 3; i++) elasticRCGv(i) = CeM(i, i);
  elasticRCGv(3) = (CeM(0, 1) + CeM(1, 0));
  elasticRCGv(4) = (CeM(2, 1) + CeM(1, 2));
  elasticRCGv(5) = (CeM(0, 2) + CeM(2, 0));

  // principal invariants of elastic Cauchy-Green strain
  Core::LinAlg::Voigt::Strains::invariants_principal(prinv_, elasticRCGv);

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int Mat::PlasticElastHyper::evaluate_kin_quant_plast(const Core::LinAlg::Matrix<3, 3>* defgrd,
    const Core::LinAlg::Matrix<3, 3>* deltaLp, const int gp, const Teuchos::ParameterList& params)
{
  id2_.clear();
  id2V_.clear();
  for (int i = 0; i < 3; i++)
  {
    id2V_(i) = 1.;
    id2_(i, i) = 1.;
  }
  Core::LinAlg::Matrix<3, 3> tmp;
  Core::LinAlg::Matrix<3, 3> tmp33;
  Core::LinAlg::Matrix<3, 3>& InvPlasticDefgrdLast = last_plastic_defgrd_inverse_[gp];
  tmp.update(-1., *deltaLp);
  Core::LinAlg::MatrixFunctErrorType exp_err_status{Core::LinAlg::MatrixFunctErrorType::no_errors};
  Core::LinAlg::Matrix<3, 3> exp_tmp = Core::LinAlg::matrix_exp(tmp, exp_err_status);
  FOUR_C_ASSERT_ALWAYS(exp_err_status == Core::LinAlg::MatrixFunctErrorType::no_errors,
      "Matrix exponential evaluation failed!");
  invpldefgrd_.multiply(InvPlasticDefgrdLast, exp_tmp);

  tmp33.multiply(*defgrd, invpldefgrd_);
  Fe_.update(tmp33);
  be_.multiply_nt(Fe_, Fe_);
  for (int i = 0; i < 3; ++i) bev_(i) = be_(i, i);
  bev_(3) = be_(0, 1);
  bev_(4) = be_(1, 2);
  bev_(5) = be_(0, 2);

  be2_.multiply(be_, be_);
  for (int i = 0; i < 3; ++i) be2v_(i) = be2_(i, i);
  be2v_(3) = be2_(0, 1);
  be2v_(4) = be2_(1, 2);
  be2v_(5) = be2_(0, 2);

  beF_.multiply(be_, *defgrd);
  beFe_.multiply(be_, Fe_);

  CeM_.multiply_tn(tmp33, tmp33);
  // elastic right Cauchy-Green in strain-like Voigt notation.
  Core::LinAlg::Matrix<6, 1> elasticRCGv;
  for (int i = 0; i < 3; i++) elasticRCGv(i) = CeM_(i, i);
  elasticRCGv(3) = (CeM_(0, 1) + CeM_(1, 0));
  elasticRCGv(4) = (CeM_(2, 1) + CeM_(1, 2));
  elasticRCGv(5) = (CeM_(0, 2) + CeM_(2, 0));
  // elastic right Cauchy-Green in stress-like Voigt notation.
  for (int i = 0; i < 3; i++) Ce_(i) = CeM_(i, i);
  Ce_(3) = (CeM_(0, 1) + CeM_(1, 0)) / 2.;
  Ce_(4) = (CeM_(2, 1) + CeM_(1, 2)) / 2.;
  Ce_(5) = (CeM_(0, 2) + CeM_(2, 0)) / 2.;

  // square of elastic right Cauchy-Green in stress-like Voigt notation.
  tmp.multiply(CeM_, CeM_);
  for (int i = 0; i < 3; i++) Ce2_(i) = tmp(i, i);
  Ce2_(3) = (tmp(0, 1) + tmp(1, 0)) / 2.;
  Ce2_(4) = (tmp(2, 1) + tmp(1, 2)) / 2.;
  Ce2_(5) = (tmp(0, 2) + tmp(2, 0)) / 2.;

  // principal invariants of elastic Cauchy-Green strain
  Core::LinAlg::Voigt::Strains::invariants_principal(prinv_, elasticRCGv);

  // inverse plastic right Cauchy-Green
  Core::LinAlg::Matrix<3, 3> CpiM;
  CpiM.multiply_nt(invpldefgrd_, invpldefgrd_);
  // stress-like Voigt notation
  for (int i = 0; i < 3; i++) Cpi_(i) = CpiM(i, i);
  Cpi_(3) = (CpiM(0, 1) + CpiM(1, 0)) / 2.;
  Cpi_(4) = (CpiM(2, 1) + CpiM(1, 2)) / 2.;
  Cpi_(5) = (CpiM(0, 2) + CpiM(2, 0)) / 2.;
  FCpi_.multiply(*defgrd, CpiM);
  beFCpi_.multiply(be_, FCpi_);

  // inverse RCG
  Core::LinAlg::Matrix<3, 3> iRCG;
  Core::LinAlg::Matrix<3, 3> RCG;
  RCG.multiply_tn(*defgrd, *defgrd);
  iRCG.invert(RCG);
  // stress-like Voigt notation
  for (int i = 0; i < 3; i++) ircg_(i) = iRCG(i, i);
  ircg_(3) = (iRCG(0, 1) + iRCG(1, 0)) / 2.;
  ircg_(4) = (iRCG(2, 1) + iRCG(1, 2)) / 2.;
  ircg_(5) = (iRCG(0, 2) + iRCG(2, 0)) / 2.;

  // C_p^-1 * C * C_p^-1
  Core::LinAlg::Matrix<3, 3> CpiCCpiM;
  tmp33.multiply(CpiM, RCG);
  CpiCCpiM.multiply(tmp33, CpiM);
  // stress-like Voigt notation
  for (int i = 0; i < 3; i++) CpiCCpi_(i) = CpiCCpiM(i, i);
  CpiCCpi_(3) = (CpiCCpiM(0, 1) + CpiCCpiM(1, 0)) / 2.;
  CpiCCpi_(4) = (CpiCCpiM(2, 1) + CpiCCpiM(1, 2)) / 2.;
  CpiCCpi_(5) = (CpiCCpiM(0, 2) + CpiCCpiM(2, 0)) / 2.;

  CpiC_.multiply(CpiM, RCG);
  FpiCe_.multiply(invpldefgrd_, CeM_);

  FpiTC_.multiply_tn(invpldefgrd_, RCG);
  CeFpiTC_.multiply(CeM_, FpiTC_);

  tmp.multiply(RCG, invpldefgrd_);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(tmp, CFpi_);
  tmp33.multiply(tmp, CeM_);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(tmp33, CFpiCe_);

  tmp.invert(CeM_);
  tmp33.multiply(invpldefgrd_, tmp);
  tmp.multiply(RCG, tmp33);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(tmp, CFpiCei_);

  return 0;
}

/*----------------------------------------------------------------------*
 |  tensor norm from stress-like Voigt-notation             seitz 05/14 |
 *----------------------------------------------------------------------*/
double Mat::PlasticElastHyper::norm_stress_like(const Core::LinAlg::Matrix<6, 1>& stress)
{
  double norm = 0;
  for (int i = 0; i < 3; ++i) norm += stress(i) * stress(i);
  for (int i = 3; i < 6; ++i) norm += 2. * stress(i) * stress(i);
  norm = sqrt(norm);
  return norm;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::PlasticElastHyper::evaluate_isotropic_princ_elast(
    Core::LinAlg::Matrix<6, 1>& stressisoprinc, Core::LinAlg::Matrix<6, 6>& cmatisoprinc,
    Core::LinAlg::Matrix<3, 1> dPI, Core::LinAlg::Matrix<6, 1> ddPII)
{
  // 2nd Piola Kirchhoff stress (according to Holzapfel-Nonlinear Solid Mechanics p. 216)
  // factors
  Core::LinAlg::Matrix<3, 1> gamma(Core::LinAlg::Initialization::zero);
  gamma(0) = 2. * (dPI(0) + prinv_(0) * dPI(1));
  gamma(1) = -2. * dPI(1);
  gamma(2) = 2. * prinv_(2) * dPI(2);

  //  // 2nd Piola Kirchhoff stresses
  stressisoprinc.update(gamma(0), Cpi_, 1.0);
  stressisoprinc.update(gamma(1), CpiCCpi_, 1.0);
  stressisoprinc.update(gamma(2), ircg_, 1.0);

  // constitutive tensor according to Holzapfel-Nonlinear Solid Mechanics p. 261)
  // factors
  Core::LinAlg::Matrix<8, 1> delta(Core::LinAlg::Initialization::zero);
  delta(0) =
      4. * (ddPII(0) + 2. * prinv_(0) * ddPII(5) + dPI(1) + prinv_(0) * prinv_(0) * ddPII(1));
  delta(1) = -4. * (ddPII(5) + prinv_(0) * ddPII(1));
  delta(2) = 4. * (prinv_(2) * ddPII(4) + prinv_(0) * prinv_(2) * ddPII(3));
  delta(3) = 4. * ddPII(1);
  delta(4) = -4. * prinv_(2) * ddPII(3);
  delta(5) = 4. * (prinv_(2) * dPI(2) + prinv_(2) * prinv_(2) * ddPII(2));
  delta(6) = -4. * prinv_(2) * dPI(2);
  delta(7) = -4. * dPI(1);

  // constitutive tensor
  cmatisoprinc.multiply_nt(delta(0), Cpi_, Cpi_, 1.);
  cmatisoprinc.multiply_nt(delta(1), CpiCCpi_, Cpi_, 1.);
  cmatisoprinc.multiply_nt(delta(1), Cpi_, CpiCCpi_, 1.);
  cmatisoprinc.multiply_nt(delta(2), Cpi_, ircg_, 1.);
  cmatisoprinc.multiply_nt(delta(2), ircg_, Cpi_, 1.);
  cmatisoprinc.multiply_nt(delta(3), CpiCCpi_, CpiCCpi_, 1.);
  cmatisoprinc.multiply_nt(delta(4), CpiCCpi_, ircg_, 1.);
  cmatisoprinc.multiply_nt(delta(4), ircg_, CpiCCpi_, 1.);
  cmatisoprinc.multiply_nt(delta(5), ircg_, ircg_, 1.);
  Core::LinAlg::FourTensorOperations::add_holzapfel_product(cmatisoprinc, ircg_, delta(6));
  Core::LinAlg::FourTensorOperations::add_holzapfel_product(cmatisoprinc, Cpi_, delta(7));

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::PlasticElastHyper::evaluate_isotropic_princ_plast(
    Core::LinAlg::Matrix<6, 9>& dPK2dFpinvIsoprinc,
    Core::LinAlg::Matrix<3, 3>& MandelStressIsoprinc, Core::LinAlg::Matrix<6, 6>& dMdCisoprinc,
    Core::LinAlg::Matrix<6, 9>& dMdFpinvIsoprinc, const Core::LinAlg::Matrix<3, 1>& gamma,
    const Core::LinAlg::Matrix<8, 1>& delta)
{
  // derivative of PK2 w.r.t. inverse plastic deformation gradient
  Core::LinAlg::FourTensorOperations::add_right_non_symmetric_holzapfel_product(
      dPK2dFpinvIsoprinc, id2_, invpldefgrd_, gamma(0));
  Core::LinAlg::FourTensorOperations::add_right_non_symmetric_holzapfel_product(
      dPK2dFpinvIsoprinc, CpiC_, invpldefgrd_, gamma(1));
  dPK2dFpinvIsoprinc.multiply_nt(delta(0), Cpi_, CFpi_, 1.);
  dPK2dFpinvIsoprinc.multiply_nt(delta(1), Cpi_, CFpiCe_, 1.);
  dPK2dFpinvIsoprinc.multiply_nt(delta(1), CpiCCpi_, CFpi_, 1.);
  dPK2dFpinvIsoprinc.multiply_nt(delta(2), Cpi_, CFpiCei_, 1.);
  dPK2dFpinvIsoprinc.multiply_nt(delta(2), ircg_, CFpi_, 1.);
  dPK2dFpinvIsoprinc.multiply_nt(delta(3), CpiCCpi_, CFpiCe_, 1.);
  dPK2dFpinvIsoprinc.multiply_nt(delta(4), CpiCCpi_, CFpiCei_, 1.);
  dPK2dFpinvIsoprinc.multiply_nt(delta(4), ircg_, CFpiCe_, 1.);
  dPK2dFpinvIsoprinc.multiply_nt(delta(5), ircg_, CFpiCei_, 1.);
  Core::LinAlg::FourTensorOperations::add_right_non_symmetric_holzapfel_product(
      dPK2dFpinvIsoprinc, id2_, FpiCe_, 0.5 * delta(7));

  // Mandel stress
  Core::LinAlg::Matrix<6, 1> Mv;
  Mv.update(gamma(0), Ce_);
  Mv.update(gamma(1), Ce2_, 1.);
  Mv.update(gamma(2), id2V_, 1.);
  for (int i = 0; i < 3; i++) MandelStressIsoprinc(i, i) += Mv(i);
  MandelStressIsoprinc(0, 1) += Mv(3);
  MandelStressIsoprinc(1, 0) += Mv(3);
  MandelStressIsoprinc(1, 2) += Mv(4);
  MandelStressIsoprinc(2, 1) += Mv(4);
  MandelStressIsoprinc(0, 2) += Mv(5);
  MandelStressIsoprinc(2, 0) += Mv(5);

  // derivative of Mandel stress w.r.t. GL
  Core::LinAlg::FourTensorOperations::add_symmetric_holzapfel_product(
      dMdCisoprinc, invpldefgrd_, invpldefgrd_, .5 * gamma(0));
  Core::LinAlg::FourTensorOperations::add_symmetric_holzapfel_product(
      dMdCisoprinc, invpldefgrd_, FpiCe_, gamma(1));
  dMdCisoprinc.multiply_nt(delta(0), Ce_, Cpi_, 1.);
  dMdCisoprinc.multiply_nt(delta(1), Ce_, CpiCCpi_, 1.);
  dMdCisoprinc.multiply_nt(delta(1), Ce2_, Cpi_, 1.);
  dMdCisoprinc.multiply_nt(delta(2), Ce_, ircg_, 1.);
  dMdCisoprinc.multiply_nt(delta(2), id2V_, Cpi_, 1.);
  dMdCisoprinc.multiply_nt(delta(3), Ce2_, CpiCCpi_, 1.);
  dMdCisoprinc.multiply_nt(delta(4), Ce2_, ircg_, 1.);
  dMdCisoprinc.multiply_nt(delta(4), id2V_, CpiCCpi_, 1.);
  dMdCisoprinc.multiply_nt(delta(5), id2V_, ircg_, 1.);

  // derivative of Mandel stress w.r.t. inverse plastic deformation gradient
  Core::LinAlg::FourTensorOperations::add_right_non_symmetric_holzapfel_product(
      dMdFpinvIsoprinc, FpiTC_, id2_, gamma(0));
  Core::LinAlg::FourTensorOperations::add_right_non_symmetric_holzapfel_product(
      dMdFpinvIsoprinc, FpiTC_, CeM_, gamma(1));
  Core::LinAlg::FourTensorOperations::add_right_non_symmetric_holzapfel_product(
      dMdFpinvIsoprinc, CeFpiTC_, id2_, gamma(1));
  dMdFpinvIsoprinc.multiply_nt(delta(0), Ce_, CFpi_, 1.);
  dMdFpinvIsoprinc.multiply_nt(delta(1), Ce_, CFpiCe_, 1.);
  dMdFpinvIsoprinc.multiply_nt(delta(1), Ce2_, CFpi_, 1.);
  dMdFpinvIsoprinc.multiply_nt(delta(2), Ce_, CFpiCei_, 1.);
  dMdFpinvIsoprinc.multiply_nt(delta(2), id2V_, CFpi_, 1.);
  dMdFpinvIsoprinc.multiply_nt(delta(3), Ce2_, CFpiCe_, 1.);
  dMdFpinvIsoprinc.multiply_nt(delta(4), Ce2_, CFpiCei_, 1.);
  dMdFpinvIsoprinc.multiply_nt(delta(4), id2V_, CFpiCe_, 1.);
  dMdFpinvIsoprinc.multiply_nt(delta(5), id2V_, CFpiCei_, 1.);

  return;
}

/*---------------------------------------------------------------------*
 | return names of visualization data (public)                         |
 *---------------------------------------------------------------------*/
void Mat::PlasticElastHyper::vis_names(std::map<std::string, int>& names) const
{
  std::string accumulatedstrain = "accumulatedstrain";
  names[accumulatedstrain] = 1;  // scalar
  std::string plastic_strain_incr = "plastic_strain_incr";
  names[plastic_strain_incr] = 1;  // scalar
  std::string plastic_zone = "plastic_zone";
  names[plastic_zone] = 1;  // scalar
  std::string kinematic_plastic_strain = "kinematic_plastic_strain";
  names[kinematic_plastic_strain] = 9;  // tensor

}  // vis_names()


/*---------------------------------------------------------------------*
 | return visualization data (public)                                  |
 *---------------------------------------------------------------------*/
bool Mat::PlasticElastHyper::vis_data(
    const std::string& name, std::vector<double>& data, int numgp, int eleID) const
{
  if (name == "accumulatedstrain")
  {
    if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");
    double tmp = 0.;
    for (unsigned gp = 0; gp < last_alpha_isotropic_.size(); gp++) tmp += accumulated_strain(gp);
    data[0] = tmp / last_alpha_isotropic_.size();
  }
  else if (name == "plastic_strain_incr")
  {
    if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");
    double tmp = 0.;
    for (unsigned gp = 0; gp < delta_alpha_i_.size(); gp++) tmp += delta_alpha_i_.at(gp);
    data[0] = tmp / delta_alpha_i_.size();
  }
  else if (name == "plastic_zone")
  {
    bool plastic_history = false;
    bool curr_active = false;
    if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");
    for (unsigned gp = 0; gp < last_alpha_isotropic_.size(); gp++)
    {
      if (accumulated_strain(gp) != 0.) plastic_history = true;
      if (active(gp)) curr_active = true;
    }
    data[0] = plastic_history + curr_active;
  }
  else if (name == "kinematic_plastic_strain")
  {
    if ((int)data.size() != 9) FOUR_C_THROW("size mismatch");
    std::vector<double> tmp(9, 0.);
    for (std::size_t gp = 0; gp < last_alpha_kinematic_.size(); ++gp)
    {
      const double* values = last_alpha_kinematic_[gp].data();
      for (std::size_t i = 0; i < 9; ++i)
      {
        tmp[i] += values[i];
      }
    }
    for (std::size_t i = 0; i < 9; ++i) data[i] = tmp[i] / last_alpha_kinematic_.size();
  }
  else
  {
    return false;
  }
  return true;

}  // vis_data()

/*---------------------------------------------------------------------*
 *---------------------------------------------------------------------*/
void Mat::PlasticElastHyper::register_output_data_names(
    std::unordered_map<std::string, int>& names_and_size) const
{
  names_and_size["accumulated_plastic_strain"] = 1;  // scalar
  names_and_size["plastic_strain_incr"] = 1;         // scalar
  names_and_size["plastic_zone"] = 1;                // scalar
  names_and_size["kinematic_plastic_strain"] = 9;    // tensor
}

/*---------------------------------------------------------------------*
 *---------------------------------------------------------------------*/
bool Mat::PlasticElastHyper::evaluate_output_data(
    const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const
{
  if (name == "accumulated_plastic_strain")
  {
    for (std::size_t gp = 0; gp < last_alpha_isotropic_.size(); ++gp)
    {
      data(gp, 0) = last_alpha_isotropic_[gp];
    }
    return true;
  }
  if (name == "plastic_strain_incr")
  {
    for (std::size_t gp = 0; gp < delta_alpha_i_.size(); ++gp)
    {
      data(gp, 0) = delta_alpha_i_[gp];
    }
    return true;
  }
  if (name == "plastic_zone")
  {
    bool plastic_history = false;
    bool curr_active = false;
    for (std::size_t gp = 0; gp < last_alpha_isotropic_.size(); ++gp)
    {
      if (accumulated_strain(gp) != 0.) plastic_history = true;
      if (active(gp)) curr_active = true;
      data(gp, 0) = plastic_history + curr_active;
    }
    return true;
  }
  if (name == "kinematic_plastic_strain")
  {
    for (std::size_t gp = 0; gp < last_alpha_kinematic_.size(); ++gp)
    {
      const double* values = last_alpha_kinematic_[gp].data();
      for (std::size_t i = 0; i < 9; ++i)
      {
        data(gp, i) = values[i];
      }
    }
    return true;
  }
  return false;
}

FOUR_C_NAMESPACE_CLOSE
