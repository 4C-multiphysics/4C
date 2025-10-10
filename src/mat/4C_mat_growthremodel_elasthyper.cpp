// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_growthremodel_elasthyper.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_elast_isoneohooke.hpp"
#include "4C_mat_elast_remodelfiber.hpp"
#include "4C_mat_elast_summand.hpp"
#include "4C_mat_elast_volsussmanbathe.hpp"
#include "4C_mat_elasthyper_service.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_solid_3D_ele_fibers.hpp"
#include "4C_utils_enum.hpp"

#include <Teuchos_SerialDenseSolver.hpp>

#include <cmath>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PAR::GrowthRemodelElastHyper::GrowthRemodelElastHyper(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      nummat_remodelfiber_(matdata.parameters.get<int>("NUMMATRF")),
      nummat_elastiniso_(matdata.parameters.get<int>("NUMMATEL3D")),
      nummat_elastinmem_(matdata.parameters.get<int>("NUMMATEL2D")),
      matids_remodelfiber_(matdata.parameters.get<std::vector<int>>("MATIDSRF")),
      matids_elastiniso_(matdata.parameters.get<std::vector<int>>("MATIDSEL3D")),
      matids_elastinmem_(matdata.parameters.get<std::vector<int>>("MATIDSEL2D")),
      matid_penalty_(matdata.parameters.get<int>("MATIDELPENALTY")),
      init_w_el_(matdata.parameters.get<double>("ELMASSFRAC")),
      density_(matdata.parameters.get<double>("DENS")),
      lamb_prestretch_cir_(matdata.parameters.get<double>("PRESTRETCHELASTINCIR")),
      lamb_prestretch_ax_(matdata.parameters.get<double>("PRESTRETCHELASTINAX")),
      t_ref_(matdata.parameters.get<double>("THICKNESS")),
      p_mean_(matdata.parameters.get<double>("MEANPRESSURE")),
      ri_(matdata.parameters.get<double>("RADIUS")),
      damage_(matdata.parameters.get<int>("DAMAGE")),
      growthtype_(matdata.parameters.get<int>("GROWTHTYPE")),
      loctimeint_(matdata.parameters.get<int>("LOCTIMEINT")),
      membrane_(matdata.parameters.get<int>("MEMBRANE")),
      cylinder_(matdata.parameters.get<int>("CYLINDER"))
{
  // check if sizes fit
  if (nummat_remodelfiber_ != (int)matids_remodelfiber_.size())
    FOUR_C_THROW(
        "number of remodelfiber materials {} does not fit to size of remodelfiber material vector "
        "{}",
        nummat_remodelfiber_, matids_remodelfiber_.size());

  if (nummat_elastinmem_ != (int)matids_elastinmem_.size())
    FOUR_C_THROW(
        "number of elastin materials {} does not fit to size of elastin material vector {}",
        nummat_elastinmem_, matids_elastinmem_.size());

  if (membrane_ == 1)
  {
    if ((growthtype_ != 1) || (loctimeint_ != 0))
      FOUR_C_THROW(
          "using membrane elements you can only simulate anisotropic growth in thickness direction"
          "and solve the local evolution equations with a Forward Euler Method");
  }
  else
  {
    if (nummat_elastiniso_ != (int)matids_elastiniso_.size())
      FOUR_C_THROW(
          "number of elastin materials {} does not fit to size of elastin material vector {}",
          nummat_elastiniso_, matids_elastiniso_.size());
    if (nummat_elastiniso_ == 0) FOUR_C_THROW("you have to set a 3D elastin material");
    if (matid_penalty_ == -1) FOUR_C_THROW("you have to set a volumetric penalty material");
    if ((p_mean_ == -1) || (ri_ == -1) || (t_ref_ == -1))
      FOUR_C_THROW(
          "you have to set the mean blood pressure, the inner radius of the vessel and thickness "
          "of the vessel wall");
  }

  if (cylinder_ != -1 && cylinder_ != 1 && cylinder_ != 2 && cylinder_ != 3)
    FOUR_C_THROW(
        "The parameter CYLINDER has to be either 1, 2 or 3. If you have defined a fiber direction "
        "in your input file then just skip this parameter");
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::GrowthRemodelElastHyper::create_material()
{
  return std::make_shared<Mat::GrowthRemodelElastHyper>(this);
}


Mat::GrowthRemodelElastHyperType Mat::GrowthRemodelElastHyperType::instance_;


Core::Communication::ParObject* Mat::GrowthRemodelElastHyperType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::GrowthRemodelElastHyper* gr_elhy = new Mat::GrowthRemodelElastHyper();
  gr_elhy->unpack(buffer);

  return gr_elhy;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::GrowthRemodelElastHyper::GrowthRemodelElastHyper()
    : params_(nullptr),
      potsumrf_(0),
      potsumeliso_(0),
      potsumelmem_(0),
      potsumelpenalty_(nullptr),
      t_tot_(0),
      nr_rf_tot_(0),
      anisotropy_()
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::GrowthRemodelElastHyper::GrowthRemodelElastHyper(Mat::PAR::GrowthRemodelElastHyper* params)
    : params_(params),
      potsumrf_(0),
      potsumeliso_(0),
      potsumelmem_(0),
      potsumelpenalty_(nullptr),
      t_tot_(0),
      nr_rf_tot_(0),
      anisotropy_()
{
  // make sure the referenced materials in material list have quick access parameters
  std::vector<int>::const_iterator m;

  // RemodelFiber
  for (m = params_->matids_remodelfiber_.begin(); m != params_->matids_remodelfiber_.end(); ++m)
  {
    const int matid = *m;
    std::shared_ptr<Mat::Elastic::RemodelFiber> sum =
        std::static_pointer_cast<Mat::Elastic::RemodelFiber>(Mat::Elastic::Summand::factory(matid));
    if (sum == nullptr) FOUR_C_THROW("Failed to allocate");
    potsumrf_.push_back(sum);
    sum->register_anisotropy_extensions(anisotropy_);
  }

  // 2d Elastin matrix
  for (m = params_->matids_elastinmem_.begin(); m != params_->matids_elastinmem_.end(); ++m)
  {
    const int matid = *m;
    std::shared_ptr<Mat::Elastic::Summand> sum = Mat::Elastic::Summand::factory(matid);
    if (sum == nullptr) FOUR_C_THROW("Failed to allocate");
    if (sum->material_type() != Core::Materials::mes_isoneohooke)
      FOUR_C_THROW(
          "2D Elastin Material: So far, you have to use a IsoNeoHooke material as the "
          "prestretching algorithm needs it. "
          "The prestretching algorithm can easily be expanded to other materials!");
    potsumelmem_.push_back(sum);
    sum->register_anisotropy_extensions(anisotropy_);
  }

  if (params_->membrane_ != 1)
  {
    // 3d Elastin matrix
    for (m = params_->matids_elastiniso_.begin(); m != params_->matids_elastiniso_.end(); ++m)
    {
      const int matid = *m;
      std::shared_ptr<Mat::Elastic::Summand> sum = Mat::Elastic::Summand::factory(matid);
      if (sum == nullptr) FOUR_C_THROW("Failed to allocate");
      if (sum->material_type() != Core::Materials::mes_isoneohooke)
        FOUR_C_THROW(
            "3D Elastin Material: So far, you have to use an IsoNeoHooke material as the "
            "prestretching algorithm needs it"
            "The prestretching algorithm can easily be expanded to other materials!");
      potsumeliso_.push_back(sum);
      sum->register_anisotropy_extensions(anisotropy_);
    }

    // VolPenalty
    std::shared_ptr<Mat::Elastic::Summand> sum =
        Mat::Elastic::Summand::factory(params_->matid_penalty_);
    if (sum == nullptr) FOUR_C_THROW("Failed to allocate");
    if (sum->material_type() != Core::Materials::mes_volsussmanbathe)
      FOUR_C_THROW(
          "Volumetric Penalty Material: So far, you have to use a CoupNeoHooke material as the "
          "prestretching algorithm needs it. "
          "This can easily be expanded to other materials!");
    potsumelpenalty_ = sum;
    sum->register_anisotropy_extensions(anisotropy_);
  }

  // initialize total simulation time
  t_tot_ = 0.0;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::GrowthRemodelElastHyper::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);


  // mass fraction of elastin
  add_to_pack(data, cur_rho_el_);
  add_to_pack(data, init_rho_el_);

  add_to_pack(data, v_);
  add_to_pack(data, gp_ax_);
  add_to_pack(data, gp_rad_);
  add_to_pack(data, acir_m_);
  add_to_pack(data, aax_m_);
  add_to_pack(data, arad_m_);
  add_to_pack(data, aradv_);
  add_to_pack(data, apl_m_);
  add_to_pack(data, ag_m_);
  add_to_pack(data, gm_);
  add_to_pack(data, radaxicirc_);
  add_to_pack(data, mue_frac_);
  add_to_pack(data, setup_);
  add_to_pack(data, nr_rf_tot_);

  anisotropy_.pack_anisotropy(data);

  Core::Communication::PotentiallyUnusedBufferScope summand_scope{data};
  if (params_ != nullptr)  // summands are not accessible in postprocessing mode
  {
    // loop map of associated potential summands
    for (const auto& p : potsumrf_) p->pack_summand(data);

    // loop map of associated potential summands
    for (const auto& p : potsumelmem_) p->pack_summand(data);

    if (params_->membrane_ != 1)
    {
      // loop map of associated potential summands
      for (const auto& p : potsumeliso_) p->pack_summand(data);

      potsumelpenalty_->pack_summand(data);
    }
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::GrowthRemodelElastHyper::unpack(Core::Communication::UnpackBuffer& buffer)
{
  // make sure we have a pristine material
  params_ = nullptr;
  potsumrf_.clear();
  potsumeliso_.clear();
  potsumelmem_.clear();



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
        params_ = static_cast<Mat::PAR::GrowthRemodelElastHyper*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
  }


  // mass fractions of elastin and ground matrix
  extract_from_pack(buffer, cur_rho_el_);
  extract_from_pack(buffer, init_rho_el_);

  extract_from_pack(buffer, v_);
  extract_from_pack(buffer, gp_ax_);
  extract_from_pack(buffer, gp_rad_);
  extract_from_pack(buffer, acir_m_);
  extract_from_pack(buffer, aax_m_);
  extract_from_pack(buffer, arad_m_);
  extract_from_pack(buffer, aradv_);
  extract_from_pack(buffer, apl_m_);
  extract_from_pack(buffer, ag_m_);
  extract_from_pack(buffer, gm_);
  extract_from_pack(buffer, radaxicirc_);
  extract_from_pack(buffer, mue_frac_);
  extract_from_pack(buffer, setup_);
  extract_from_pack(buffer, nr_rf_tot_);

  anisotropy_.unpack_anisotropy(buffer);

  Core::Communication::PotentiallyUnusedBufferScope summand_scope{buffer};
  if (params_ != nullptr)  // summands are not accessible in postprocessing mode
  {
    // make sure the referenced materials in material list have quick access parameters
    std::vector<int>::const_iterator m;

    // RemodelFiber
    for (m = params_->matids_remodelfiber_.begin(); m != params_->matids_remodelfiber_.end(); ++m)
    {
      const int matid = *m;
      std::shared_ptr<Mat::Elastic::RemodelFiber> sum =
          std::static_pointer_cast<Mat::Elastic::RemodelFiber>(
              Mat::Elastic::Summand::factory(matid));
      if (sum == nullptr) FOUR_C_THROW("Failed to allocate");
      potsumrf_.push_back(sum);
    }
    // loop map of associated potential summands
    for (auto& p : potsumrf_)
    {
      p->unpack_summand(buffer);
      p->register_anisotropy_extensions(anisotropy_);
    }

    // 2D Elastin matrix
    for (m = params_->matids_elastinmem_.begin(); m != params_->matids_elastinmem_.end(); ++m)
    {
      const int matid = *m;
      std::shared_ptr<Mat::Elastic::Summand> sum = Mat::Elastic::Summand::factory(matid);
      if (sum == nullptr) FOUR_C_THROW("Failed to allocate");
      if (sum->material_type() != Core::Materials::mes_isoneohooke)
        FOUR_C_THROW(
            "2D Elastin Material: So far, you have to use a IsoNeoHooke material as the "
            "prestretching algorithm needs it. "
            "This can easily be expanded to other materials!");
      potsumelmem_.push_back(sum);
    }
    // loop map of associated potential summands
    for (auto& p : potsumelmem_)
    {
      p->unpack_summand(buffer);
      p->register_anisotropy_extensions(anisotropy_);
    }

    if (params_->membrane_ != 1)
    {
      // 3D Elastin matrix
      for (m = params_->matids_elastiniso_.begin(); m != params_->matids_elastiniso_.end(); ++m)
      {
        const int matid = *m;
        std::shared_ptr<Mat::Elastic::Summand> sum = Mat::Elastic::Summand::factory(matid);
        if (sum == nullptr) FOUR_C_THROW("Failed to allocate");
        if (sum->material_type() != Core::Materials::mes_isoneohooke)
          FOUR_C_THROW(
              "3D Elastin Material: So far, you have to use an IsoNeoHooke material as the "
              "prestretching algorithm needs it"
              "This can easily be expanded to other materials!");
        potsumeliso_.push_back(sum);
      }
      // loop map of associated potential summands
      for (auto& p : potsumeliso_)
      {
        p->unpack_summand(buffer);
        p->register_anisotropy_extensions(anisotropy_);
      }

      // VolPenalty
      std::shared_ptr<Mat::Elastic::Summand> sum =
          Mat::Elastic::Summand::factory(params_->matid_penalty_);
      if (sum == nullptr) FOUR_C_THROW("Failed to allocate");
      if (sum->material_type() != Core::Materials::mes_volsussmanbathe)
        FOUR_C_THROW(
            "Volumetric Penalty Material: So far, you have to use a CoupNeoHooke material as the "
            "prestretching algorithm needs it. "
            "This can easily be expanded to other materials!");
      potsumelpenalty_ = sum;
      potsumelpenalty_->unpack_summand(buffer);

      // in the postprocessing mode, we do not unpack everything we have packed
      // -> position check cannot be done in this case
    }
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::GrowthRemodelElastHyper::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  // read anisotropy
  anisotropy_.set_number_of_gauss_points(numgp);
  anisotropy_.read_anisotropy_from_element(fibers, coord_system);

  // Initialize some variables
  v_.resize(numgp, 1.0);
  gp_ax_.resize(numgp, 0.0);
  gp_rad_.resize(numgp, 0.0);
  cur_rho_el_.resize(numgp);
  init_rho_el_.resize(numgp);
  gm_.resize(numgp, Core::LinAlg::Matrix<3, 3>(Core::LinAlg::Initialization::zero));
  setup_.resize(numgp, 1);

  for (int gp = 0; gp < numgp; ++gp)
  {
    init_rho_el_[gp] = params_->density_ * params_->init_w_el_;
    cur_rho_el_[gp] = init_rho_el_[gp];
  }

  // Setup summands
  // remodelfiber
  for (auto& p : potsumrf_) p->setup(numgp, params_->density_, fibers, coord_system);

  // 2D elastin matrix
  for (auto& p : potsumelmem_) p->setup(numgp, fibers, coord_system);

  if (params_->membrane_ != 1)
  {
    // 3D elastin matrix
    for (auto& p : potsumeliso_) p->setup(numgp, fibers, coord_system);

    // volpenalty
    potsumelpenalty_->setup(numgp, fibers, coord_system);
  }

  // Setup circumferential, radial and axial structural tensor
  setup_axi_cir_rad_structural_tensor(coord_system);

  // Setup growth tensors in the case of anisotropic growth (--> growth in thickness direction)
  if (params_->growthtype_ == 1) setup_aniso_growth_tensors();

  // variable which is multiplied with the material parameter of the elastin sheets
  mue_frac_.resize(numgp, 1.0);

  // total number of remodel fibers
  for (auto& p : potsumrf_) nr_rf_tot_ += p->get_num_fibers();
}

void Mat::GrowthRemodelElastHyper::post_setup(
    const Teuchos::ParameterList& params, const int eleGID)
{
  anisotropy_.read_anisotropy_from_parameter_list(params);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::GrowthRemodelElastHyper::setup_axi_cir_rad_structural_tensor(
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  // CIR-AXI-RAD nomenclature
  if (coord_system.has_value())
  {
    for (int i = 0; i < 3; ++i)
    {
      for (int j = 0; j < 3; ++j)
      {
        radaxicirc_(i, j) = coord_system->element_system[j](i);
      }
    }
    // Read in of data
    // read local (cylindrical) cosy-directions at current element
    // Axial direction
    aax_m_ = Core::LinAlg::Matrix<3, 3>(
        Core::LinAlg::dyadic(coord_system->element_system[1], coord_system->element_system[1])
            .data(),
        /*view=*/false);

    // Circumferential direction
    acir_m_ = Core::LinAlg::Matrix<3, 3>(
        Core::LinAlg::dyadic(coord_system->element_system[2], coord_system->element_system[2])
            .data(),
        /*view=*/false);

    // Radial direction
    arad_m_ = Core::LinAlg::Matrix<3, 3>(
        Core::LinAlg::dyadic(coord_system->element_system[0], coord_system->element_system[0])
            .data(),
        /*view=*/false);

    // radial structural tensor in "stress-like" Voigt notation
    Core::LinAlg::Voigt::Stresses::matrix_to_vector(arad_m_, aradv_);
  }
  // Cylinder flag defined in input file, dummy AXI, CIR and RAD directions are written till
  // location of element center in reference configuration is available
  else if (params_->cylinder_ == 1 || params_->cylinder_ == 2 || params_->cylinder_ == 3)
  {
    radaxicirc_(0, 0) = radaxicirc_(1, 1) = radaxicirc_(2, 2) = 1.0;
    aax_m_(0, 0) = 1.0;
    acir_m_(1, 1) = 1.0;
    arad_m_(2, 2) = 1.0;
    aradv_(2) = 1.0;
  }
  // No AXI CIR RAD-direction defined in input file and additionally no cylinder flag was set
  else
    FOUR_C_THROW(
        "Homogenized Constrained Mixture Model can so far only be used by defining AXI-, CIR- and "
        "RAD-direction in the input file or by defining the Cylinder flag!");
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::GrowthRemodelElastHyper::setup_aniso_growth_tensors()
{
  ag_m_.update(1.0, arad_m_, 0.0);

  // structural tensor of the plane in which all fibers are located
  Core::LinAlg::Matrix<3, 3> id(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; ++i) id(i, i) = 1.0;
  apl_m_.update(1.0, arad_m_, 0.0);
  apl_m_.update(1.0, id, -1.0);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::GrowthRemodelElastHyper::setup_axi_cir_rad_cylinder(
    Core::LinAlg::Tensor<double, 3> elecenter, double const dt)
{
  // Clear dummy directions
  radaxicirc_ = {};
  Core::LinAlg::Matrix<3, 1> axdir;
  Core::LinAlg::Matrix<3, 1> raddir;
  Core::LinAlg::Matrix<3, 1> cirdir;

  // Axial direction
  radaxicirc_(params_->cylinder_ - 1, 1) = 1.0;
  for (int i = 0; i < 3; ++i) axdir(i) = radaxicirc_(i, 1);
  aax_m_.multiply_nt(1.0, axdir, axdir, 0.0);

  // Radial direction
  elecenter(params_->cylinder_ - 1) = 0.0;
  raddir.update(
      1. / Core::LinAlg::norm2(elecenter), Core::LinAlg::make_matrix_view<3, 1>(elecenter), 0.0);
  for (int i = 0; i < 3; ++i) radaxicirc_(i, 0) = raddir(i);
  arad_m_.multiply_nt(1.0, raddir, raddir, 0.0);

  // Circumferential direction
  cirdir.cross_product(axdir, raddir);
  for (int i = 0; i < 3; ++i) radaxicirc_(i, 2) = cirdir(i);
  acir_m_.multiply_nt(1.0, cirdir, cirdir, 0.0);

  // radial structural tensor in "stress-like" Voigt notation
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(arad_m_, aradv_);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::GrowthRemodelElastHyper::update(const Core::LinAlg::Tensor<double, 3, 3>& defgrd,
    int const gp, const Teuchos::ParameterList& params, const EvaluationContext& context,
    int const eleGID)
{
  // Update individual volume of elastin
  FOUR_C_ASSERT(context.time_step_size, "Time step size not given in evaluation context.");
  static double dt_pre = *context.time_step_size;
  if (params_->damage_ == 1) evaluate_elastin_damage(dt_pre);

  // loop map of associated potential summands
  // remodelfiber
  for (auto& p : potsumrf_) p->update();

  // 2D elastin matrix
  for (auto& p : potsumelmem_) p->update();

  if (params_->membrane_ != 1)
  {
    // 3D elastin matrix
    for (auto& p : potsumeliso_) p->update();

    // volpenalty
    potsumelpenalty_->update();
  }


  // build inelastic growth deformation gradient
  Core::LinAlg::Matrix<3, 3> FgM(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> iFgM(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> dFgdrhoM(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> diFgdrhoM(Core::LinAlg::Initialization::zero);
  evaluate_growth_def_grad(FgM, iFgM, dFgdrhoM, diFgdrhoM, gp);

  // time step size
  FOUR_C_ASSERT(context.time_step_size, "Time step size not given in evaluation context.");
  double dt = *context.time_step_size;

  switch (params_->loctimeint_)
  {
    case 0:
      for (auto& p : potsumrf_)
        p->evaluate_growth_and_remodeling_expl(
            Core::LinAlg::make_matrix_view(defgrd), dt, iFgM, gp, eleGID);
      break;
    case 1:  // do nothing
      break;
    default:
      FOUR_C_THROW("LOCTIMEINT has to be either 1 (Backward Euler) or 0 (Forward Euler)");
      break;
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::GrowthRemodelElastHyper::evaluate_prestretch(
    Core::LinAlg::Matrix<3, 3> const* const defgrd, int const gp, int const eleGID)
{
  // setup prestretch of elastin in axial and circumferential direction
  gm_[gp].update(params_->lamb_prestretch_cir_, acir_m_, 0.0);
  gm_[gp].update(params_->lamb_prestretch_ax_, aax_m_, 1.0);

  // safety check
  if (potsumeliso_.size() != 1)
    FOUR_C_THROW(
        "So far, the prestretching routine does only work with ONE 3D isochoric elastin material");

  std::shared_ptr<Mat::Elastic::IsoNeoHooke> matiso;
  std::shared_ptr<Mat::Elastic::VolSussmanBathe> matvol;
  matiso = std::dynamic_pointer_cast<Mat::Elastic::IsoNeoHooke>(potsumeliso_[0]);
  matvol = std::dynamic_pointer_cast<Mat::Elastic::VolSussmanBathe>(potsumelpenalty_);
  double R = 1.0;
  double dRdlamb_pre = 0.0;
  double lamb_pre = 1. / (params_->lamb_prestretch_cir_ * params_->lamb_prestretch_ax_);
  const double mue = matiso->mue(eleGID);
  while (fabs(R) > 1.0e-10)
  {
    R = mue * init_rho_el_[gp] *
            std::pow(params_->lamb_prestretch_cir_ * params_->lamb_prestretch_cir_ *
                         params_->lamb_prestretch_ax_ * params_->lamb_prestretch_ax_ * lamb_pre *
                         lamb_pre,
                -2. / 3.) *  // ToDo: Verify that -1.0 / 3.0 would be correct here
            (lamb_pre * lamb_pre -
                (1. / 3.) * (params_->lamb_prestretch_cir_ * params_->lamb_prestretch_cir_ +
                                params_->lamb_prestretch_ax_ * params_->lamb_prestretch_ax_ +
                                lamb_pre * lamb_pre)) +
        matvol->kappa() * init_rho_el_[gp] *
            ((params_->lamb_prestretch_cir_ * params_->lamb_prestretch_ax_ * lamb_pre) *
                    (params_->lamb_prestretch_cir_ * params_->lamb_prestretch_ax_ * lamb_pre) -
                (params_->lamb_prestretch_cir_ * params_->lamb_prestretch_ax_ * lamb_pre)) +
        ((1.0 - (gp_rad_[gp] - 10.0e-3) / params_->t_ref_) * params_->p_mean_);

    dRdlamb_pre =
        mue * init_rho_el_[gp] *
            (-(4. / 3.) *
                std::pow(params_->lamb_prestretch_cir_ * params_->lamb_prestretch_ax_ * lamb_pre,
                    -7. / 3.) *
                params_->lamb_prestretch_cir_ * params_->lamb_prestretch_ax_) *
            (lamb_pre * lamb_pre -
                (1. / 3.) * (params_->lamb_prestretch_cir_ * params_->lamb_prestretch_cir_ +
                                params_->lamb_prestretch_ax_ * params_->lamb_prestretch_ax_ +
                                lamb_pre * lamb_pre)) +
        mue * init_rho_el_[gp] *
            std::pow(params_->lamb_prestretch_cir_ * params_->lamb_prestretch_cir_ *
                         params_->lamb_prestretch_ax_ * params_->lamb_prestretch_ax_ * lamb_pre *
                         lamb_pre,
                -2. / 3.) *
            (2.0 * lamb_pre - (1. / 3.) * (2.0 * lamb_pre)) +
        matvol->kappa() * init_rho_el_[gp] *
            (2.0 * (params_->lamb_prestretch_cir_ * params_->lamb_prestretch_ax_ * lamb_pre) *
                    params_->lamb_prestretch_cir_ * params_->lamb_prestretch_ax_ -
                params_->lamb_prestretch_cir_ * params_->lamb_prestretch_ax_);

    lamb_pre = lamb_pre + (-R / dRdlamb_pre);
  }

  // update radial prestretch of elastin
  gm_[gp].update(lamb_pre, arad_m_, 1.0);


  // calculate circumferential residual stress
  double sig = 0.0;
  Core::LinAlg::Matrix<6, 1> Acir_strain(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Strains::matrix_to_vector(acir_m_, Acir_strain);

  // total circumferential Cauchy stress ("membrane stress")
  double sig_tot = (params_->p_mean_ * params_->ri_) / params_->t_ref_;

  // evaluate anisotropic remodel fibers
  Core::LinAlg::Matrix<3, 3> CM(Core::LinAlg::Initialization::zero);
  CM.multiply_tn(1.0, *defgrd, *defgrd, 0.0);
  Core::LinAlg::Matrix<6, 1> stressaniso(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6, 6> cmataniso(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> id(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; ++i) id(i, i) = 1.0;
  for (auto& p : potsumrf_)
  {
    p->evaluate_anisotropic_stress_cmat(CM, id, cmataniso, stressaniso, gp, 1.0, eleGID);
    sig += stressaniso.dot(Acir_strain);
  }

  // build stress response and elasticity tensor of 3D material
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> stressiso(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmatiso(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<NUM_STRESS_3D, 9> dSdiFgiso(Core::LinAlg::Initialization::zero);
  evaluate_stress_cmat_iso(defgrd, gm_[gp], stressiso, cmatiso, dSdiFgiso, gp, eleGID);
  sig += stressiso.dot(Acir_strain);

  // residual stress
  double sig_res = sig_tot - sig;

  // safety check
  if (potsumelmem_.size() != 1)
    FOUR_C_THROW(
        "So far, only ONE CoupNeoHooke material can be chosen for the membrane \"material\"");

  // build stress response and elasticity tensor of membrane material
  Core::LinAlg::Matrix<6, 1> stressmem(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6, 6> cmatmem(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<NUM_STRESS_3D, 9> dSdiFgmem(Core::LinAlg::Initialization::zero);
  evaluate_stress_cmat_membrane(CM, id, stressmem, cmatmem, dSdiFgmem, gp, eleGID);

  mue_frac_[gp] = sig_res / stressmem.dot(Acir_strain);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::GrowthRemodelElastHyper::setup_g_r_3d(Core::LinAlg::Matrix<3, 3> const* const defgrd,
    const Teuchos::ParameterList& params, const EvaluationContext& context, const double dt,
    const int gp, const int eleGID)
{
  Core::LinAlg::Tensor<double, 3> axdir{};
  Core::LinAlg::Tensor<double, 3> raddir{};
  FOUR_C_ASSERT(context.ref_coords, "Reference coordinates not given in evaluation context.");
  const auto& gp_reference_coordinates = *context.ref_coords;

  if ((params_->cylinder_ == 1 || params_->cylinder_ == 2 || params_->cylinder_ == 3) &&
      (setup_[0] == 1))
  {
    const auto& ele_center_reference_coordinates =
        params.get<Core::LinAlg::Tensor<double, 3>>("elecenter_coords_ref");
    setup_axi_cir_rad_cylinder(ele_center_reference_coordinates, dt);
    if (params_->growthtype_ == 1) setup_aniso_growth_tensors();

    // Update fiber directions with new local coordinate system (radaxicirc_)
    for (auto& k : potsumrf_) k->update_fiber_dirs(radaxicirc_, dt);
  }

  // Evaluate radial and axial distance between origin and current Gauss-Point
  for (int i = 0; i < 3; ++i)
  {
    axdir(i) = radaxicirc_(i, 1);
    raddir(i) = radaxicirc_(i, 0);
  }
  gp_ax_[gp] = axdir * gp_reference_coordinates;
  gp_rad_[gp] = raddir * gp_reference_coordinates;

  // TODO: BE CAREFUL! So far, this prestretching procedure is only valid for certain materials and
  // a cylindrical geometry.
  //       The principle of the prestretching routine can easily be adapted to other materials or
  //       general geometries!!!
  evaluate_prestretch(defgrd, gp, eleGID);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::GrowthRemodelElastHyper::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
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

  // save current simulation time (used for the evaluation of elastin degradation)
  FOUR_C_ASSERT(context.total_time, "Time not given in evaluation context.");
  t_tot_ = *context.total_time;

  // time step size
  FOUR_C_ASSERT(context.time_step_size, "Time step size not given in evaluation context.");
  double dt = *context.time_step_size;

  if (setup_[gp] == 1)
  {
    setup_g_r_3d(&defgrd_mat, params, context, dt, gp, eleGID);
    setup_[gp] = 0;
  }

  // blank resulting quantities
  // ... even if it is an implicit law that cmat is zero upon input
  stress_view.clear();
  cmat_view.clear();

  // some static variables
  static Core::LinAlg::Matrix<NUM_STRESS_3D, 1> stressaniso(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmataniso(
      Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmatanisoadd(
      Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> iFinM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> CM(Core::LinAlg::Initialization::zero);
  CM.multiply_tn(1.0, defgrd_mat, defgrd_mat, 0.0);
  static Core::LinAlg::Matrix<6, 1> stressmem(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<6, 6> cmatmem(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<6, 9> dSdiFgmem(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<NUM_STRESS_3D, 1> stressiso(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmatiso(
      Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<6, 9> dSdiFgiso(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmatadd(
      Core::LinAlg::Initialization::zero);

  // build growth deformation gradient
  static Core::LinAlg::Matrix<3, 3> FgM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> iFgM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> dFgdrhoM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> diFgdrhoM(Core::LinAlg::Initialization::zero);
  evaluate_growth_def_grad(FgM, iFgM, dFgdrhoM, diFgdrhoM, gp);

  // some initialization
  // only do at first time step
  if (t_tot_ == dt)
  {
    // evaluate anisotropic remodel fibers
    for (auto& p : potsumrf_)
    {
      p->evaluate_anisotropic_stress_cmat(CM, iFgM, cmataniso, stressaniso, gp, dt, eleGID);
      stress_view.update(1.0, stressaniso, 1.0);
      cmat_view.update(1.0, cmataniso, 1.0);
    }

    // build stress response and elasticity tensor of 3D material
    iFinM.multiply_nn(1.0, iFgM, gm_[gp], 0.0);
    evaluate_stress_cmat_iso(&defgrd_mat, iFinM, stressiso, cmatiso, dSdiFgiso, gp, eleGID);
    stress_view.update(1.0, stressiso, 1.0);
    cmat_view.update(1.0, cmatiso, 1.0);

    // build stress response and elasticity tensor of membrane material
    evaluate_stress_cmat_membrane(CM, iFgM, stressmem, cmatmem, dSdiFgmem, gp, eleGID);
    stress_view.update(1.0, stressmem, 1.0);
    cmat_view.update(1.0, cmatmem, 1.0);
  }
  // Growth and Remodeling
  else
  {
    switch (params_->loctimeint_)
    {
      case 0:
      {
        // evaluate anisotropic remodel fibers
        for (auto& p : potsumrf_)
        {
          p->evaluate_anisotropic_stress_cmat(CM, iFgM, cmataniso, stressaniso, gp, dt, eleGID);
          stress_view.update(1.0, stressaniso, 1.0);
          cmat_view.update(1.0, cmataniso, 1.0);
        }

        // build stress response and elasticity tensor of 3D material
        iFinM.multiply_nn(1.0, iFgM, gm_[gp], 0.0);
        evaluate_stress_cmat_iso(&defgrd_mat, iFinM, stressiso, cmatiso, dSdiFgiso, gp, eleGID);
        stress_view.update(1.0, stressiso, 1.0);
        cmat_view.update(1.0, cmatiso, 1.0);

        // build stress response and elasticity tensor of membrane material
        evaluate_stress_cmat_membrane(CM, iFgM, stressmem, cmatmem, dSdiFgmem, gp, eleGID);
        stress_view.update(1.0, stressmem, 1.0);
        cmat_view.update(1.0, cmatmem, 1.0);

        break;
      }
      case 1:
      {
        static Core::LinAlg::SerialDenseMatrix K_T(2 * nr_rf_tot_, 2 * nr_rf_tot_, true);
        solve_for_rho_lambr(K_T, FgM, iFgM, dFgdrhoM, diFgdrhoM, &defgrd_mat, dt, gp, eleGID);

        // split solution vector in dlambda_r/dC and drho/dC
        static std::vector<Core::LinAlg::Matrix<1, 6>> drhodC(
            nr_rf_tot_, Core::LinAlg::Matrix<1, 6>(Core::LinAlg::Initialization::zero));
        static std::vector<Core::LinAlg::Matrix<1, 6>> dlambrdC(
            nr_rf_tot_, Core::LinAlg::Matrix<1, 6>(Core::LinAlg::Initialization::zero));
        static Core::LinAlg::Matrix<1, 6> sum_drhodC(Core::LinAlg::Initialization::zero);
        solve_fordrhod_cdlambrd_c(
            drhodC, dlambrdC, sum_drhodC, K_T, iFgM, &defgrd_mat, dt, gp, eleGID);

        // Evaluate anisotropic remodel fibers
        int nr_grf_proc = 0;
        for (auto& p : potsumrf_)
        {
          p->evaluate_anisotropic_stress_cmat(CM, iFgM, cmataniso, stressaniso, gp, dt, eleGID);
          stress_view.update(1.0, stressaniso, 1.0);
          cmat_view.update(1.0, cmataniso, 1.0);
          p->evaluate_additional_growth_remodel_cmat(&defgrd_mat, nr_grf_proc, iFgM, diFgdrhoM,
              drhodC, dlambrdC, cmatanisoadd, gp, eleGID);
          cmat_view.update(1.0, cmatanisoadd, 1.0);
          nr_grf_proc += p->get_num_fibers();
        }

        // Evaluate 3D elastin material
        iFinM.multiply_nn(1.0, iFgM, gm_[gp], 0.0);
        evaluate_stress_cmat_iso(&defgrd_mat, iFinM, stressiso, cmatiso, dSdiFgiso, gp, eleGID);
        stress_view.update(1.0, stressiso, 1.0);
        cmat_view.update(1.0, cmatiso, 1.0);

        // Build stress response and elasticity tensor of membrane material
        evaluate_stress_cmat_membrane(CM, iFgM, stressmem, cmatmem, dSdiFgmem, gp, eleGID);
        stress_view.update(1.0, stressmem, 1.0);
        cmat_view.update(1.0, cmatmem, 1.0);

        // Evaluate additional terms for the elasticity tensor
        static Core::LinAlg::Matrix<6, 9> dSdiFg_sum(Core::LinAlg::Initialization::zero);
        dSdiFg_sum.update(1.0, dSdiFgiso, 0.0);
        dSdiFg_sum.update(1.0, dSdiFgmem, 1.0);
        evaluate_additional_cmat(cmatadd, diFgdrhoM, sum_drhodC, dSdiFg_sum, gp);
        cmat_view.update(1.0, cmatadd, 1.0);

        break;
      }
      default:
        FOUR_C_THROW("LOCTIMEINT has to be 1 (Backward Euler Method) or 0 (Forward Euler Method)");
        break;
    }
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::GrowthRemodelElastHyper::solve_for_rho_lambr(Core::LinAlg::SerialDenseMatrix& K_T,
    Core::LinAlg::Matrix<3, 3>& FgM, Core::LinAlg::Matrix<3, 3>& iFgM,
    Core::LinAlg::Matrix<3, 3>& dFgdrhoM, Core::LinAlg::Matrix<3, 3>& diFgdrhoM,
    Core::LinAlg::Matrix<3, 3> const* const defgrd, double const& dt, int const gp,
    int const eleGID)
{
  // allocate some variables (allocate variables only once, therefore static)
  static std::vector<std::vector<double>> dWdrho(nr_rf_tot_, std::vector<double>(nr_rf_tot_, 0.0));
  static std::vector<std::vector<double>> dWdlambr(
      nr_rf_tot_, std::vector<double>(nr_rf_tot_, 0.0));
  static std::vector<double> W(nr_rf_tot_, 0.0);
  static std::vector<std::vector<double>> dEdrho(nr_rf_tot_, std::vector<double>(nr_rf_tot_, 0.0));
  static std::vector<std::vector<double>> dEdlambr(
      nr_rf_tot_, std::vector<double>(nr_rf_tot_, 0.0));
  static std::vector<double> E(nr_rf_tot_, 0.0);
  using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
  using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;
  static Teuchos::SerialDenseSolver<ordinalType, scalarType> solver;
  // residual vector of assembled system of equation
  static Core::LinAlg::SerialDenseMatrix R(2 * nr_rf_tot_, 1);
  for (unsigned i = 0; i < 2 * nr_rf_tot_; ++i) R(i, 0) = 1.0;
  // solution vector of assembled system of equation
  static Core::LinAlg::SerialDenseMatrix dsol(2 * nr_rf_tot_, 1);

  int nr_grf_proc = 0;
  int iter = 0;
  int l = 0;
  while (R.normOne() > 1.0e-10)
  {
    if (iter != 0)
    {
      // Solve linearized system of equations
      solver.setMatrix(Teuchos::rcpFromRef(K_T.base()));
      solver.setVectors(Teuchos::rcpFromRef(dsol.base()), Teuchos::rcpFromRef(R.base()));
      solver.solveToRefinedSolution(true);
      solver.factorWithEquilibration(true);
      solver.solve();

      l = 0;
      for (auto& p : potsumrf_)
      {
        for (unsigned k = 0; k < p->get_num_fibers(); ++k)
        {
          // update inelastic fiber stretch and current mass density
          p->update_growth_remodel_parameter(dsol(l, 0), dsol(nr_rf_tot_ + l, 0), k, gp);
          ++l;
        }
      }
    }
    // Update growth deformation gradient
    evaluate_growth_def_grad(FgM, iFgM, dFgdrhoM, diFgdrhoM, gp);

    // global number of fibers which were already processed
    nr_grf_proc = 0;
    for (auto& p : potsumrf_)
    {
      p->evaluate_derivatives_internal_newton(defgrd, nr_grf_proc, nr_rf_tot_, gp, dt, eleGID, iFgM,
          dFgdrhoM, diFgdrhoM, dWdrho, dWdlambr, W, dEdrho, dEdlambr, E);
      nr_grf_proc += p->get_num_fibers();
    }

    // Assembly
    for (unsigned i = 0; i < nr_rf_tot_; ++i)
    {
      R(i, 0) = -W[i];
      R(nr_rf_tot_ + i, 0) = -E[i];
      for (unsigned j = 0; j < nr_rf_tot_; ++j)
      {
        K_T(i, j) = dWdrho[i][j];
        K_T(i, nr_rf_tot_ + j) = dWdlambr[i][j];
        K_T(nr_rf_tot_ + i, j) = dEdrho[i][j];
        K_T(nr_rf_tot_ + i, nr_rf_tot_ + j) = dEdlambr[i][j];
      }
    }

    if (iter > 95)
    {
      std::cout << "iteration:  " << iter << std::endl
                << std::endl
                << "Vector of residuals: " << std::endl
                << R << std::endl
                << "Matrix of derivatives:" << std::endl
                << K_T;
      std::cout << "=================================================================" << std::endl
                << std::endl;
      if (iter > 100) FOUR_C_THROW("Internal Newton (at Gauss-Point) does not converge!");
    }
    ++iter;
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::GrowthRemodelElastHyper::solve_fordrhod_cdlambrd_c(
    std::vector<Core::LinAlg::Matrix<1, 6>>& drhodC,
    std::vector<Core::LinAlg::Matrix<1, 6>>& dlambrdC, Core::LinAlg::Matrix<1, 6>& sum_drhodC,
    Core::LinAlg::SerialDenseMatrix& K_T, Core::LinAlg::Matrix<3, 3> const& iFgM,
    Core::LinAlg::Matrix<3, 3> const* const defgrd, double const& dt, int const gp,
    int const eleGID) const
{
  static std::vector<Core::LinAlg::Matrix<1, 6>> dWdC(
      nr_rf_tot_, Core::LinAlg::Matrix<1, 6>(Core::LinAlg::Initialization::zero));
  static std::vector<Core::LinAlg::Matrix<1, 6>> dEdC(
      nr_rf_tot_, Core::LinAlg::Matrix<1, 6>(Core::LinAlg::Initialization::zero));
  int nr_grf_proc = 0;
  for (const auto& p : potsumrf_)
  {
    p->evaluate_derivatives_cauchy_green(defgrd, nr_grf_proc, gp, dt, iFgM, dWdC, dEdC, eleGID);
    nr_grf_proc += p->get_num_fibers();
  }

  // Assembly
  // Rcmat
  static Core::LinAlg::SerialDenseMatrix Rcmat(2 * nr_rf_tot_, 6);
  for (unsigned i = 0; i < nr_rf_tot_; ++i)
  {
    for (unsigned j = 0; j < 6; ++j)
    {
      Rcmat(i, j) = -dWdC[i](0, j);
      Rcmat(nr_rf_tot_ + i, j) = -dEdC[i](0, j);
    }
  }

  // Solve
  using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
  using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;
  static Teuchos::SerialDenseSolver<ordinalType, scalarType> solver;
  static Core::LinAlg::SerialDenseMatrix dsolcmat(2 * nr_rf_tot_, 6);
  solver.setMatrix(Teuchos::rcpFromRef(K_T.base()));
  solver.setVectors(Teuchos::rcpFromRef(dsolcmat.base()), Teuchos::rcpFromRef(Rcmat.base()));
  solver.solveToRefinedSolution(true);
  solver.factorWithEquilibration(true);
  solver.solve();

  sum_drhodC.clear();
  for (unsigned i = 0; i < nr_rf_tot_; ++i)
  {
    for (unsigned j = 0; j < 6; ++j)
    {
      drhodC[i](0, j) = dsolcmat(i, j);
      dlambrdC[i](0, j) = dsolcmat(nr_rf_tot_ + i, j);
    }
    sum_drhodC.update(1.0, drhodC[i], 1.0);
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::GrowthRemodelElastHyper::evaluate_stress_cmat_iso(
    Core::LinAlg::Matrix<3, 3> const* const defgrd, Core::LinAlg::Matrix<3, 3> const& iFinM,
    Core::LinAlg::Matrix<6, 1>& stressiso, Core::LinAlg::Matrix<6, 6>& cmatiso,
    Core::LinAlg::Matrix<6, 9>& dSdiFg, int const gp, int const eleGID) const
{
  // clear some variables
  stressiso.clear();
  cmatiso.clear();
  dSdiFg.clear();

  // Evaluate elastin matrix
  // some variables
  static Core::LinAlg::Matrix<6, 1> iCinv(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<6, 1> iCinCiCinv(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<6, 1> iCv(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 1> prinv(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> iCinCM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> iFinCeM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<9, 1> CiFin9x1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<9, 1> CiFinCe9x1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<9, 1> CiFiniCe9x1(Core::LinAlg::Initialization::zero);

  evaluate_kin_quant_elast(defgrd, iFinM, iCinv, iCinCiCinv, iCv, iCinCM, iFinCeM, CiFin9x1,
      CiFinCe9x1, CiFiniCe9x1, prinv, gp);

  Core::LinAlg::Matrix<3, 1> dPIe(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6, 1> ddPIIe(Core::LinAlg::Initialization::zero);
  evaluate_invariant_derivatives(prinv, dPIe, ddPIIe, gp, eleGID);

  // 2nd Piola Kirchhoff stress factors (according to Holzapfel-Nonlinear Solid Mechanics p. 216)
  static Core::LinAlg::Matrix<3, 1> gamma(Core::LinAlg::Initialization::zero);
  // constitutive tensor factors (according to Holzapfel-Nonlinear Solid Mechanics p. 261)
  static Core::LinAlg::Matrix<8, 1> delta(Core::LinAlg::Initialization::zero);

  // compose coefficients
  calculate_gamma_delta(gamma, delta, prinv, dPIe, ddPIIe);

  evaluate_isotropic_princ_elast(stressiso, cmatiso, iCinv, iCinCiCinv, iCv, gamma, delta);

  evaluated_sdi_fg(dSdiFg, gamma, delta, iFinM, iCinCM, iCinv, CiFin9x1, CiFinCe9x1, iCinCiCinv,
      CiFiniCe9x1, iCv, iFinCeM, gp);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::GrowthRemodelElastHyper::evaluate_kin_quant_elast(
    Core::LinAlg::Matrix<3, 3> const* const defgrd, Core::LinAlg::Matrix<3, 3> const& iFinM,
    Core::LinAlg::Matrix<6, 1>& iCinv, Core::LinAlg::Matrix<6, 1>& iCinCiCinv,
    Core::LinAlg::Matrix<6, 1>& iCv, Core::LinAlg::Matrix<3, 3>& iCinCM,
    Core::LinAlg::Matrix<3, 3>& iFinCeM, Core::LinAlg::Matrix<9, 1>& CiFin9x1,
    Core::LinAlg::Matrix<9, 1>& CiFinCe9x1, Core::LinAlg::Matrix<9, 1>& CiFiniCe9x1,
    Core::LinAlg::Matrix<3, 1>& prinv, const int gp)
{
  // inverse inelastic right Cauchy-Green
  static Core::LinAlg::Matrix<3, 3> iCinM(Core::LinAlg::Initialization::zero);
  iCinM.multiply_nt(1.0, iFinM, iFinM, 0.0);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(iCinM, iCinv);

  // inverse right Cauchy-Green
  static Core::LinAlg::Matrix<3, 3> iCM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> CM(Core::LinAlg::Initialization::zero);
  CM.multiply_tn(1.0, *defgrd, *defgrd, 0.0);
  iCM.invert(CM);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(iCM, iCv);

  // C_{in}^{-1} * C * C_{in}^{-1}
  static Core::LinAlg::Matrix<3, 3> tmp(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> iCinCiCinM;
  tmp.multiply_nn(1.0, iCinM, CM, 0.0);
  iCinCiCinM.multiply_nn(1.0, tmp, iCinM, 0.0);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(iCinCiCinM, iCinCiCinv);

  // elastic right Cauchy-Green in strain-like Voigt notation.
  tmp.multiply_nn(1.0, *defgrd, iFinM, 0.0);
  static Core::LinAlg::Matrix<3, 3> CeM(Core::LinAlg::Initialization::zero);
  CeM.multiply_tn(1.0, tmp, tmp, 0.0);
  static Core::LinAlg::Matrix<6, 1> Ce_strain(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Strains::matrix_to_vector(CeM, Ce_strain);

  // principal invariants of elastic right Cauchy-Green strain
  Core::LinAlg::Voigt::Strains::invariants_principal(prinv, Ce_strain);

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
}


/*----------------------------------------------------------------------/
 * Reads derivatives with respect to invariants and modified invariants
 * from all materials of the elasthyper-toolbox                         */
/*----------------------------------------------------------------------*/
void Mat::GrowthRemodelElastHyper::evaluate_invariant_derivatives(
    Core::LinAlg::Matrix<3, 1> const& prinv, Core::LinAlg::Matrix<3, 1>& dPIw,
    Core::LinAlg::Matrix<6, 1>& ddPIIw, int const gp, int const eleGID) const
{
  // derivatives of principal materials weighted with their mass fraction in the constraint mixture
  static Core::LinAlg::Matrix<3, 1> dPgrowthI(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<6, 1> ddPgrowthII(Core::LinAlg::Initialization::zero);

  // loop map of associated potential summands
  // derivatives of strain energy function w.r.t. principal invariants
  for (const auto& p : potsumeliso_)
  {
    dPgrowthI.clear();
    ddPgrowthII.clear();
    p->add_derivatives_principal(dPgrowthI, ddPgrowthII, prinv, gp, eleGID);
    dPIw.update(cur_rho_el_[gp], dPgrowthI, 1.0);
    ddPIIw.update(cur_rho_el_[gp], ddPgrowthII, 1.0);
  }

  // derivatives of decoupled (volumetric or isochoric) materials weighted with their mass fraction
  // in the constraint mixture
  static Core::LinAlg::Matrix<3, 1> modinv(Core::LinAlg::Initialization::zero);
  invariants_modified(modinv, prinv);
  Core::LinAlg::Matrix<3, 1> dPmodI(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6, 1> ddPmodII(Core::LinAlg::Initialization::zero);
  for (const auto& p : potsumeliso_)
  {
    dPgrowthI.clear();
    ddPgrowthII.clear();
    p->add_derivatives_modified(dPgrowthI, ddPgrowthII, modinv, gp, eleGID);
    dPmodI.update(cur_rho_el_[gp], dPgrowthI, 1.0);
    ddPmodII.update(cur_rho_el_[gp], ddPgrowthII, 1.0);
  }

  // volpenalty
  dPgrowthI.clear();
  ddPgrowthII.clear();
  potsumelpenalty_->add_derivatives_modified(dPgrowthI, ddPgrowthII, modinv, gp, eleGID);
  dPmodI.update(cur_rho_el_[gp], dPgrowthI, 1.0);
  ddPmodII.update(cur_rho_el_[gp], ddPgrowthII, 1.0);

  // convert decoupled derivatives to principal derivatives
  convert_mod_to_princ(prinv, dPmodI, ddPmodII, dPIw, ddPIIw);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::GrowthRemodelElastHyper::invariants_modified(
    Core::LinAlg::Matrix<3, 1>& modinv, Core::LinAlg::Matrix<3, 1> const& prinv) const
{
  // 1st invariant, trace
  modinv(0) = prinv(0) * std::pow(prinv(2), -1. / 3.);
  // 2nd invariant
  modinv(1) = prinv(1) * std::pow(prinv(2), -2. / 3.);
  // J
  modinv(2) = std::pow(prinv(2), 1. / 2.);
}


/*----------------------------------------------------------------------/
 * Converts derivatives with respect to modified invariants in derivatives
 * with respect to principal invariants                                 */
/*----------------------------------------------------------------------*/
void Mat::GrowthRemodelElastHyper::convert_mod_to_princ(Core::LinAlg::Matrix<3, 1> const& prinv,
    Core::LinAlg::Matrix<3, 1> const& dPmodI, Core::LinAlg::Matrix<6, 1> const& ddPmodII,
    Core::LinAlg::Matrix<3, 1>& dPI, Core::LinAlg::Matrix<6, 1>& ddPII) const
{
  // Conversions to dPI
  dPI(0) += std::pow(prinv(2), -1. / 3.) * dPmodI(0);
  dPI(1) += std::pow(prinv(2), -2. / 3.) * dPmodI(1);
  dPI(2) += 0.5 * std::pow(prinv(2), -0.5) * dPmodI(2) -
            1. / 3. * prinv(0) * std::pow(prinv(2), -4. / 3.) * dPmodI(0) -
            2. / 3. * prinv(1) * std::pow(prinv(2), -5. / 3.) * dPmodI(1);

  // Conversions to ddPII
  ddPII(0) += std::pow(prinv(2), -2. / 3.) * ddPmodII(0);
  ddPII(1) += std::pow(prinv(2), -4. / 3.) * ddPmodII(1);
  ddPII(2) += (1. / 9.) * std::pow(prinv(2), -8. / 3.) * prinv(0) * prinv(0) * ddPmodII(0) +
              (4. / 9.) * prinv(0) * prinv(1) * std::pow(prinv(2), -3.) * ddPmodII(5) -
              (1. / 3.) * std::pow(prinv(2), -11. / 6.) * prinv(0) * ddPmodII(4) +
              (4. / 9.) * std::pow(prinv(2), -7. / 3.) * prinv(0) * dPmodI(0) +
              (4. / 9.) * std::pow(prinv(2), -10. / 3.) * prinv(1) * prinv(1) * ddPmodII(1) -
              (2. / 3.) * std::pow(prinv(2), -13. / 6.) * prinv(1) * ddPmodII(3) +
              (10. / 9.) * std::pow(prinv(2), -8. / 3.) * prinv(1) * dPmodI(1) +
              0.25 * std::pow(prinv(2), -1.) * ddPmodII(2) -
              0.25 * std::pow(prinv(2), -1.5) * dPmodI(2);
  ddPII(3) += -(1. / 3.) * std::pow(prinv(2), -2.) * prinv(0) * ddPmodII(5) -
              (2. / 3.) * std::pow(prinv(2), -7. / 3.) * prinv(1) * ddPmodII(1) +
              0.5 * std::pow(prinv(2), -7. / 6.) * ddPmodII(3) -
              (2. / 3.) * std::pow(prinv(2), -5. / 3.) * dPmodI(1);
  ddPII(4) += -(1. / 3.) * std::pow(prinv(2), -5. / 3.) * prinv(0) * ddPmodII(0) -
              (2. / 3.) * std::pow(prinv(2), -2.) * prinv(1) * ddPmodII(5) +
              0.5 * std::pow(prinv(2), -5. / 6.) * ddPmodII(4) -
              (1. / 3.) * std::pow(prinv(2), -4. / 3.) * dPmodI(0);
  ddPII(5) += std::pow(prinv(2), -1.) * ddPmodII(5);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::GrowthRemodelElastHyper::evaluate_isotropic_princ_elast(
    Core::LinAlg::Matrix<6, 1>& stressisoprinc, Core::LinAlg::Matrix<6, 6>& cmatisoprinc,
    Core::LinAlg::Matrix<6, 1> const& iCinv, Core::LinAlg::Matrix<6, 1> const& iCinCiCinv,
    Core::LinAlg::Matrix<6, 1> const& iCv, Core::LinAlg::Matrix<3, 1> const& gamma,
    Core::LinAlg::Matrix<8, 1> const& delta) const
{
  // 2nd Piola Kirchhoff stresses
  stressisoprinc.update(gamma(0), iCinv, 1.0);
  stressisoprinc.update(gamma(1), iCinCiCinv, 1.0);
  stressisoprinc.update(gamma(2), iCv, 1.0);

  // constitutive tensor
  cmatisoprinc.multiply_nt(delta(0), iCinv, iCinv, 1.);
  cmatisoprinc.multiply_nt(delta(1), iCinCiCinv, iCinv, 1.);
  cmatisoprinc.multiply_nt(delta(1), iCinv, iCinCiCinv, 1.);
  cmatisoprinc.multiply_nt(delta(2), iCinv, iCv, 1.);
  cmatisoprinc.multiply_nt(delta(2), iCv, iCinv, 1.);
  cmatisoprinc.multiply_nt(delta(3), iCinCiCinv, iCinCiCinv, 1.);
  cmatisoprinc.multiply_nt(delta(4), iCinCiCinv, iCv, 1.);
  cmatisoprinc.multiply_nt(delta(4), iCv, iCinCiCinv, 1.);
  cmatisoprinc.multiply_nt(delta(5), iCv, iCv, 1.);
  Core::LinAlg::FourTensorOperations::add_holzapfel_product(cmatisoprinc, iCv, delta(6));
  Core::LinAlg::FourTensorOperations::add_holzapfel_product(cmatisoprinc, iCinv, delta(7));
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::GrowthRemodelElastHyper::evaluated_sdi_fg(Core::LinAlg::Matrix<6, 9>& dSdiFg,
    Core::LinAlg::Matrix<3, 1> const& gamma, Core::LinAlg::Matrix<8, 1> const& delta,
    Core::LinAlg::Matrix<3, 3> const& iFinM, Core::LinAlg::Matrix<3, 3> const& iCinCM,
    Core::LinAlg::Matrix<6, 1> const& iCinv, Core::LinAlg::Matrix<9, 1> const& CiFin9x1,
    Core::LinAlg::Matrix<9, 1> const& CiFinCe9x1, Core::LinAlg::Matrix<6, 1> const& iCinCiCinv,
    Core::LinAlg::Matrix<9, 1> const& CiFiniCe9x1, Core::LinAlg::Matrix<6, 1> const& iCv,
    Core::LinAlg::Matrix<3, 3> const& iFinCeM, int const gp) const
{
  // clear some variables
  dSdiFg.clear();

  static Core::LinAlg::Matrix<3, 3> id(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; ++i) id(i, i) = 1.0;
  static Core::LinAlg::Matrix<6, 9> dSdiFin(Core::LinAlg::Initialization::zero);
  dSdiFin.clear();

  // derivative of second Piola Kirchhoff stress w.r.t. inverse growth deformation gradient
  Core::LinAlg::FourTensorOperations::add_right_non_symmetric_holzapfel_product(
      dSdiFin, id, iFinM, gamma(0));
  Core::LinAlg::FourTensorOperations::add_right_non_symmetric_holzapfel_product(
      dSdiFin, iCinCM, iFinM, gamma(1));
  dSdiFin.multiply_nt(delta(0), iCinv, CiFin9x1, 1.);
  dSdiFin.multiply_nt(delta(1), iCinv, CiFinCe9x1, 1.);
  dSdiFin.multiply_nt(delta(1), iCinCiCinv, CiFin9x1, 1.);
  dSdiFin.multiply_nt(delta(2), iCinv, CiFiniCe9x1, 1.);
  dSdiFin.multiply_nt(delta(2), iCv, CiFin9x1, 1.);
  dSdiFin.multiply_nt(delta(3), iCinCiCinv, CiFinCe9x1, 1.);
  dSdiFin.multiply_nt(delta(4), iCinCiCinv, CiFiniCe9x1, 1.);
  dSdiFin.multiply_nt(delta(4), iCv, CiFinCe9x1, 1.);
  dSdiFin.multiply_nt(delta(5), iCv, CiFiniCe9x1, 1.);
  Core::LinAlg::FourTensorOperations::add_right_non_symmetric_holzapfel_product(
      dSdiFin, id, iFinCeM, 0.5 * delta(7));

  // diFin/diFg
  static Core::LinAlg::Matrix<9, 9> diFindiFg(Core::LinAlg::Initialization::zero);
  diFindiFg.clear();
  Core::LinAlg::FourTensorOperations::add_non_symmetric_product(1.0, id, gm_[gp], diFindiFg);

  dSdiFg.multiply_nn(1.0, dSdiFin, diFindiFg, 0.0);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::GrowthRemodelElastHyper::evaluate_additional_cmat(Core::LinAlg::Matrix<6, 6>& cmatadd,
    Core::LinAlg::Matrix<3, 3> const& diFgdrhoM, Core::LinAlg::Matrix<1, 6> const& sum_drhodC,
    Core::LinAlg::Matrix<6, 9> const& dSdiFg, int const gp) const
{
  // clear some variables
  cmatadd.clear();

  // diFg/dC
  Core::LinAlg::Matrix<9, 6> diFgdC(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<9, 1> diFgdrho9x1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(diFgdrhoM, diFgdrho9x1);
  diFgdC.multiply_nn(1.0, diFgdrho9x1, sum_drhodC, 0.0);

  // update elasticity tensor
  cmatadd.multiply_nn(2.0, dSdiFg, diFgdC, 0.0);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::GrowthRemodelElastHyper::evaluate_stress_cmat_membrane(
    Core::LinAlg::Matrix<3, 3> const& CM, Core::LinAlg::Matrix<3, 3> const& iFgM,
    Core::LinAlg::Matrix<6, 1>& stress, Core::LinAlg::Matrix<6, 6>& cmat,
    Core::LinAlg::Matrix<6, 9>& dSdiFg, const int gp, const int eleGID) const
{
  // clear some variables
  stress.clear();
  cmat.clear();
  dSdiFg.clear();

  // 2nd Piola Kirchhoff stress
  Core::LinAlg::FADMatrix<3, 3> iFgM_fad(Core::LinAlg::Initialization::zero);
  iFgM_fad = iFgM;
  iFgM_fad.diff(0, 9);
  static Core::LinAlg::FADMatrix<3, 3> CM_fad(Core::LinAlg::Initialization::zero);
  CM_fad = CM;
  static Core::LinAlg::FADMatrix<3, 3> FgM_fad(Core::LinAlg::Initialization::zero);
  FgM_fad.invert(iFgM_fad);
  static Core::LinAlg::FADMatrix<3, 3> GM_fad(Core::LinAlg::Initialization::zero);
  GM_fad = gm_[gp];
  static Core::LinAlg::FADMatrix<3, 3> iFinM_fad(Core::LinAlg::Initialization::zero);
  iFinM_fad.multiply_nn(1.0, iFgM_fad, GM_fad, 0.0);
  static Core::LinAlg::FADMatrix<3, 3> FinM_fad(Core::LinAlg::Initialization::zero);
  FinM_fad.invert(iFinM_fad);
  Core::LinAlg::FADMatrix<3, 3> CinM_fad(Core::LinAlg::Initialization::zero);
  CinM_fad.multiply_tn(1.0, FinM_fad, FinM_fad, 0.0);

  static Core::LinAlg::FADMatrix<3, 3> CeM_fad(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::FADMatrix<3, 3> tmp_fad(Core::LinAlg::Initialization::zero);
  tmp_fad.multiply_nn(1.0, CM_fad, iFinM_fad, 0.0);
  CeM_fad.multiply_tn(1.0, iFinM_fad, tmp_fad, 0.0);

  static Core::LinAlg::FADMatrix<3, 3> id_fad(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; ++i) id_fad(i, i) = 1.0;
  static Core::LinAlg::FADMatrix<3, 3> AradM_fad(Core::LinAlg::Initialization::zero);
  AradM_fad = arad_m_;
  static Core::LinAlg::FADMatrix<3, 3> AradgrM_fad(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::FADMatrix<3, 3> AorthgrM_fad(Core::LinAlg::Initialization::zero);
  tmp_fad.multiply_nn(1.0, FinM_fad, AradM_fad, 0.0);
  AradgrM_fad.multiply_nt(1.0 / CinM_fad.dot(AradM_fad), tmp_fad, FinM_fad, 0.0);
  AorthgrM_fad.update(1.0, id_fad, 0.0);
  AorthgrM_fad.update(-1.0, AradgrM_fad, 1.0);

  // X = Aorthgr*Ce*Aorthgr + Aradgr
  static Core::LinAlg::FADMatrix<3, 3> XM_fad(Core::LinAlg::Initialization::zero);
  tmp_fad.multiply_nn(1.0, AorthgrM_fad, CeM_fad, 0.0);
  XM_fad.multiply_nn(1.0, tmp_fad, AorthgrM_fad, 0.0);
  XM_fad.update(1.0, AradgrM_fad, 1.0);

  static Core::LinAlg::FADMatrix<3, 3> iFinAorthgriFinTM_fad(Core::LinAlg::Initialization::zero);
  tmp_fad.multiply_nn(1.0, iFinM_fad, AorthgrM_fad, 0.0);
  iFinAorthgriFinTM_fad.multiply_nt(1.0, tmp_fad, iFinM_fad, 0.0);

  double mue_el_mem = 0.0;
  std::shared_ptr<Mat::Elastic::IsoNeoHooke> matmem;
  for (const auto& k : potsumelmem_)
  {
    matmem = std::dynamic_pointer_cast<Mat::Elastic::IsoNeoHooke>(k);
    mue_el_mem += matmem->mue(eleGID);
  }

  FAD X_det = XM_fad(0, 0) * (XM_fad(1, 1) * XM_fad(2, 2) - XM_fad(1, 2) * XM_fad(2, 1)) -
              XM_fad(0, 1) * (XM_fad(1, 0) * XM_fad(2, 2) - XM_fad(1, 2) * XM_fad(2, 0)) +
              XM_fad(0, 2) * (XM_fad(1, 0) * XM_fad(2, 1) - XM_fad(1, 1) * XM_fad(2, 0));

  static Core::LinAlg::FADMatrix<3, 3> iXM_fad(Core::LinAlg::Initialization::zero);
  iXM_fad.invert(XM_fad);
  static Core::LinAlg::FADMatrix<3, 3> ZM_fad(
      Core::LinAlg::Initialization::zero);  // Z =
                                            // F_{in}^{-T}*A_{gr}^{T}*X^{-1}*A_{gr}^{T}*F_{in}^{-1}
  static Core::LinAlg::FADMatrix<3, 3> ZTM_fad(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::FADMatrix<3, 3> AorthgriFinM_fad(Core::LinAlg::Initialization::zero);
  AorthgriFinM_fad.multiply_nn(1.0, AorthgrM_fad, iFinM_fad, 0.0);
  tmp_fad.multiply_tt(1.0, AorthgriFinM_fad, iXM_fad, 0.0);
  ZTM_fad.multiply_nn(1.0, tmp_fad, AorthgriFinM_fad, 0.0);
  ZM_fad.update_t(1.0, ZTM_fad, 0.0);

  static Core::LinAlg::FADMatrix<3, 3> stress_fad(Core::LinAlg::Initialization::zero);
  stress_fad.update(mue_el_mem * cur_rho_el_[gp] * mue_frac_[gp], iFinAorthgriFinTM_fad, 0.0);
  stress_fad.update(-0.5 * mue_el_mem * cur_rho_el_[gp] * mue_frac_[gp] / X_det, ZM_fad, 1.0);
  stress_fad.update(-0.5 * mue_el_mem * cur_rho_el_[gp] * mue_frac_[gp] / X_det, ZTM_fad, 1.0);

  static Core::LinAlg::Matrix<3, 3> stressM(Core::LinAlg::Initialization::zero);
  stressM = stress_fad.convertto_double();
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(stressM, stress);

  // Derivative of 2nd Piola Kirchhoff stress w.r.t. the inverse growth deformation gradient
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 9; ++j) dSdiFg(i, j) = stress_fad(i, i).dx(j);
  for (int j = 0; j < 9; ++j)
  {
    dSdiFg(3, j) = 0.5 * (stress_fad(0, 1).dx(j) + stress_fad(1, 0).dx(j));
    dSdiFg(4, j) = 0.5 * (stress_fad(1, 2).dx(j) + stress_fad(2, 1).dx(j));
    dSdiFg(5, j) = 0.5 * (stress_fad(0, 2).dx(j) + stress_fad(2, 0).dx(j));
  }

  // Elasticity tensor
  static Core::LinAlg::Matrix<3, 3> ZM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> ZTM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> XM(Core::LinAlg::Initialization::zero);
  ZM = ZM_fad.convertto_double();
  ZTM = ZTM_fad.convertto_double();
  XM = XM_fad.convertto_double();
  // Y = Z^T + Z
  static Core::LinAlg::Matrix<3, 3> YM(Core::LinAlg::Initialization::zero);
  YM.update(1.0, ZM, 0.0);
  YM.update(1.0, ZTM, 1.0);
  static Core::LinAlg::Matrix<6, 1> Yv(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(YM, Yv);
  static Core::LinAlg::Matrix<6, 6> dYdC(Core::LinAlg::Initialization::zero);
  dYdC.clear();
  Core::LinAlg::FourTensorOperations::add_holzapfel_product(dYdC, Yv, -0.5);

  cmat.multiply_nt(0.5 * mue_el_mem * cur_rho_el_[gp] * mue_frac_[gp] / X_det.val(), Yv, Yv, 0.0);
  cmat.update(-mue_el_mem * cur_rho_el_[gp] * mue_frac_[gp] / X_det.val(), dYdC, 1.0);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::GrowthRemodelElastHyper::evaluate_elastin_damage(double const& dt_pre)
{
  double T_el = 101 * 365.25;  // in days
  double t_dam = 40.0;

  for (unsigned gp = 0; gp < cur_rho_el_.size(); ++gp)
  {
    cur_rho_el_[gp] = init_rho_el_[gp] * (exp(-(t_tot_ - dt_pre) / T_el) -
                                             0.5 * (1.0 - exp(-(t_tot_ - dt_pre) / t_dam)) *
                                                 exp(-0.5 * (100.0 * (fabs(gp_ax_[gp]) - 0.09)) *
                                                     (100.0 * (fabs(gp_ax_[gp]) - 0.09))));
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::GrowthRemodelElastHyper::evaluate_growth_def_grad(Core::LinAlg::Matrix<3, 3>& FgM,
    Core::LinAlg::Matrix<3, 3>& iFgM, Core::LinAlg::Matrix<3, 3>& dFgdrhoM,
    Core::LinAlg::Matrix<3, 3>& diFgdrhoM, const int gp)
{
  double rho_col_sum = 0.0;
  // evaluate volume change
  for (auto& p : potsumrf_)
    for (unsigned k = 0; k < p->get_num_fibers(); ++k)
      rho_col_sum += p->get_cur_mass_density(k, gp);
  v_[gp] = (rho_col_sum + cur_rho_el_[gp]) / params_->density_;

  switch (params_->growthtype_)
  {
    case 1:
      // build growth and inverse growth deformation gradient (and its derivative) for anisotropic
      // growth
      FgM.update(1.0, apl_m_, 0.0);
      FgM.update(v_[gp], ag_m_, 1.0);
      iFgM.update(1.0, apl_m_, 0.0);
      iFgM.update(1.0 / v_[gp], ag_m_, 1.0);
      diFgdrhoM.update(-(1. / (v_[gp] * v_[gp])) * (1. / params_->density_), ag_m_, 0.0);
      dFgdrhoM.update(1.0 / params_->density_, ag_m_, 0.0);
      break;
    case 0:
      // build growth and inverse growth deformation gradient (and its derivative) for isotropic
      // growth
      FgM.update(std::pow(v_[gp], 1. / 3.), acir_m_, 0.0);
      FgM.update(std::pow(v_[gp], 1. / 3.), arad_m_, 1.0);
      FgM.update(std::pow(v_[gp], 1. / 3.), aax_m_, 1.0);
      iFgM.update(std::pow(v_[gp], -1. / 3.), acir_m_, 0.0);
      iFgM.update(std::pow(v_[gp], -1. / 3.), arad_m_, 1.0);
      iFgM.update(std::pow(v_[gp], -1. / 3.), aax_m_, 1.0);
      diFgdrhoM.update(
          -1. / 3. * std::pow(v_[gp], -4. / 3.) * (1. / params_->density_), acir_m_, 0.0);
      diFgdrhoM.update(
          -1. / 3. * std::pow(v_[gp], -4. / 3.) * (1. / params_->density_), arad_m_, 1.0);
      diFgdrhoM.update(
          -1. / 3. * std::pow(v_[gp], -4. / 3.) * (1. / params_->density_), aax_m_, 1.0);
      dFgdrhoM.update(
          1. / 3. * std::pow(v_[gp], -2. / 3.) * (1. / params_->density_), acir_m_, 0.0);
      dFgdrhoM.update(
          1. / 3. * std::pow(v_[gp], -2. / 3.) * (1. / params_->density_), arad_m_, 1.0);
      dFgdrhoM.update(1. / 3. * std::pow(v_[gp], -2. / 3.) * (1. / params_->density_), aax_m_, 1.0);
      break;
    default:
      FOUR_C_THROW("growthtype has to be either 1: anisotropic growth or 0: isotropic growth");
      break;
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::GrowthRemodelElastHyper::setup_g_r_2d(const Teuchos::ParameterList& params,
    const EvaluationContext& context, const double dt, const int gp)
{
  Core::LinAlg::Tensor<double, 3> axdir{};
  FOUR_C_ASSERT(context.ref_coords, "Reference coordinates not given in evaluation context.");
  const auto& gp_coords_ref = *context.ref_coords;

  if ((params_->cylinder_ == 1 || params_->cylinder_ == 2 || params_->cylinder_ == 3) &&
      (setup_[0] == 1))
  {
    const auto& element_center_reference_coordinates =
        params.get<Core::LinAlg::Tensor<double, 3>>("elecenter_coords_ref");
    setup_axi_cir_rad_cylinder(element_center_reference_coordinates, dt);
    if (params_->growthtype_ == 1) setup_aniso_growth_tensors();

    // Update fiber directions with new local coordinate system (radaxicirc_)
    for (auto& k : potsumrf_) k->update_fiber_dirs(radaxicirc_, dt);
  }

  for (int i = 0; i < 3; ++i) axdir(i) = radaxicirc_(i, 1);
  gp_ax_[gp] = axdir * gp_coords_ref;

  // update elastin prestretch in radial direction
  gm_[gp].update(params_->lamb_prestretch_cir_, acir_m_, 0.0);
  gm_[gp].update(params_->lamb_prestretch_ax_, aax_m_, 1.0);
  gm_[gp].update(1. / (params_->lamb_prestretch_ax_ * params_->lamb_prestretch_cir_), arad_m_, 1.0);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::GrowthRemodelElastHyper::evaluate_membrane(Core::LinAlg::Matrix<3, 3> const& defgrd_glob,
    const Teuchos::ParameterList& params, const EvaluationContext& context,
    Core::LinAlg::Matrix<3, 3>& pk2M_glob, Core::LinAlg::Matrix<6, 6>& cmat_glob, const int gp,
    const int eleGID)
{
  // time step size
  FOUR_C_ASSERT(context.time_step_size, "Time step size not given in evaluation context.");
  double dt = *context.time_step_size;

  // blank resulting quantities
  pk2M_glob.clear();
  cmat_glob.clear();

  // Evaluate growth deformation gradient
  Core::LinAlg::Matrix<3, 3> FgM(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> iFgM(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> dFgdrhoM(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> diFgdrhoM(Core::LinAlg::Initialization::zero);
  evaluate_growth_def_grad(FgM, iFgM, dFgdrhoM, diFgdrhoM, gp);

  // Elastic deformation gradient
  Core::LinAlg::Matrix<3, 3> CM(Core::LinAlg::Initialization::zero);
  CM.multiply_tn(1.0, defgrd_glob, defgrd_glob, 0.0);

  // Evaluate anisotropic remodel fibers
  static Core::LinAlg::Matrix<6, 1> pk2v_glob(Core::LinAlg::Initialization::zero);
  pk2v_glob.clear();
  static Core::LinAlg::Matrix<NUM_STRESS_3D, 1> stressaniso(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmataniso(
      Core::LinAlg::Initialization::zero);
  for (auto& p : potsumrf_)
  {
    p->evaluate_anisotropic_stress_cmat(CM, iFgM, cmataniso, stressaniso, gp, dt, eleGID);
    pk2v_glob.update(1.0, stressaniso, 1.0);
    cmat_glob.update(1.0, cmataniso, 1.0);
  }

  // Build stress response and elasticity tensor of membrane material
  static Core::LinAlg::Matrix<NUM_STRESS_3D, 1> stressmem(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmatmem(
      Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<6, 9> dummy(Core::LinAlg::Initialization::zero);
  evaluate_stress_cmat_membrane(CM, iFgM, stressmem, cmatmem, dummy, gp, eleGID);
  pk2v_glob.update(1.0, stressmem, 1.0);
  cmat_glob.update(1.0, cmatmem, 1.0);
  Core::LinAlg::Voigt::Stresses::vector_to_matrix(pk2v_glob, pk2M_glob);
}


double Mat::GrowthRemodelElastHyper::evaluate_membrane_thickness_stretch(
    Core::LinAlg::Matrix<3, 3> const& defgrd_glob, const Teuchos::ParameterList& params,
    const EvaluationContext& context, const int gp, const int eleGID)
{
  // save current simulation time (used for the evaluation of elastin degradation)
  FOUR_C_ASSERT(context.total_time, "Time not given in evaluation context.");
  t_tot_ = *context.total_time;

  // time step size
  FOUR_C_ASSERT(context.time_step_size, "Time step size not given in evaluation context.");
  double dt = *context.time_step_size;

  if (setup_[gp] == 1)
  {
    setup_g_r_2d(params, context, dt, gp);
    setup_[gp] = 0;
  }

  // Evaluate growth deformation gradient
  // growth deformation gradient
  Core::LinAlg::Matrix<3, 3> FgM(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> iFgM(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> dFgdrhoM(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> diFgdrhoM(Core::LinAlg::Initialization::zero);
  evaluate_growth_def_grad(FgM, iFgM, dFgdrhoM, diFgdrhoM, gp);

  // Elastic deformation gradient
  Core::LinAlg::Matrix<3, 3> CM(Core::LinAlg::Initialization::zero);
  CM.multiply_tn(1.0, defgrd_glob, defgrd_glob, 0.0);
  Core::LinAlg::Matrix<3, 3> CeM(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> iFinM(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> tmp(Core::LinAlg::Initialization::zero);
  iFinM.multiply_nn(1.0, iFgM, gm_[gp], 0.0);
  tmp.multiply_tn(1.0, iFinM, CM, 0.0);
  CeM.multiply_nn(1.0, tmp, iFinM, 0.0);

  // Calculate adapted right Cauchy Green stretch in thickness direction
  Core::LinAlg::Matrix<3, 3> AplCeAplAradM(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> CinM(Core::LinAlg::Initialization::zero);
  tmp.multiply(1.0, apl_m_, CeM, 0.0);
  AplCeAplAradM.multiply_nn(1.0, tmp, apl_m_, 0.0);
  AplCeAplAradM.update(1.0, arad_m_, 1.0);
  tmp.multiply_nt(1.0, iFinM, iFinM, 0.0);
  CinM.invert(tmp);
  double rcg33 = CinM.dot(arad_m_) / AplCeAplAradM.determinant();

  return std::sqrt(rcg33);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::GrowthRemodelElastHyper::vis_names(std::map<std::string, int>& names) const
{
  std::string result_mass_fraction_el;
  result_mass_fraction_el = "mass_fraction_el";

  names[result_mass_fraction_el] = 1;


  std::string result_v_growth;
  result_v_growth = "v_growth";

  names[result_v_growth] = 1;


  // loop map of associated potential summands
  // remodelfiber
  for (unsigned int p = 0; p < potsumrf_.size(); ++p) potsumrf_[p]->vis_names(names, p);

  // 2D elastin matrix
  for (auto& p : potsumelmem_) p->vis_names(names);

  if (params_->membrane_ != 1)
  {
    // 3D elastin matrix
    for (auto& p : potsumeliso_) p->vis_names(names);

    // volpenalty
    potsumelpenalty_->vis_names(names);
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool Mat::GrowthRemodelElastHyper::vis_data(
    const std::string& name, std::vector<double>& data, int numgp, int eleID) const
{
  int return_val = 0;

  if (name == "mass_fraction_el")
  {
    if (data.size() != 1) FOUR_C_THROW("size mismatch");
    for (double i : cur_rho_el_)
    {
      data[0] += i;
    }
    data[0] = data[0] / cur_rho_el_.size();

    return true;
  }

  if (name == "v_growth")
  {
    if (data.size() != 1) FOUR_C_THROW("size mismatch");
    for (double i : v_)
    {
      data[0] += i;
    }
    data[0] = data[0] / v_.size();

    return true;
  }

  // loop map of associated potential summands
  // remodelfiber
  if (name.at(name.length() - 3) == '0')
    return_val += potsumrf_[0]->vis_data(name, data, numgp, eleID);
  if (name.at(name.length() - 3) == '1')
    return_val += potsumrf_[1]->vis_data(name, data, numgp, eleID);

  // 2D elastin matrix
  for (auto& p : potsumelmem_) return_val += p->vis_data(name, data, numgp, eleID);

  if (params_->membrane_ != 1)
  {
    // 3D elastin matrix
    for (auto& p : potsumeliso_) return_val += p->vis_data(name, data, numgp, eleID);

    // volpenalty
    return_val += potsumelpenalty_->vis_data(name, data, numgp, eleID);
  }

  return (bool)return_val;
}

FOUR_C_NAMESPACE_CLOSE
