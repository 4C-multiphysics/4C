// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_viscoelasthyper.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_utils_densematrix_inverse.hpp"
#include "4C_mat_elast_summand.hpp"
#include "4C_mat_elast_visco_generalizedgenmax.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_utils_enum.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PAR::ViscoElastHyper::ViscoElastHyper(const Core::Mat::PAR::Parameter::Data& matdata)
    : Mat::PAR::ElastHyper(matdata)
{
  // polyconvexity check is just implemented for isotropic hyperlastic materials
  if (polyconvex_)
    FOUR_C_THROW(
        "This polyconvexity-check is just implemented for isotropic "
        "hyperelastic-materials (do not use for viscoelastic materials).");
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::ViscoElastHyper::create_material()
{
  return std::make_shared<Mat::ViscoElastHyper>(this);
}


Mat::ViscoElastHyperType Mat::ViscoElastHyperType::instance_;


Core::Communication::ParObject* Mat::ViscoElastHyperType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::ViscoElastHyper* elhy = new Mat::ViscoElastHyper();
  elhy->unpack(buffer);

  return elhy;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::ViscoElastHyper::ViscoElastHyper() : Mat::ElastHyper()
{
  isovisco_ = false;
  viscogenmax_ = false;
  viscogeneralizedgenmax_ = false;
  viscofract_ = false;

  // history data 09/13
  isinitvis_ = false;
  histscgcurr_ = nullptr;
  histscglast_ = nullptr;
  histmodrcgcurr_ = nullptr;
  histmodrcglast_ = nullptr;
  histstresscurr_ = nullptr;
  histstresslast_ = nullptr;

  // genmax
  histartstresscurr_ = nullptr;
  histartstresslast_ = nullptr;

  // generalized genmax
  histbranchstresscurr_ = nullptr;
  histbranchstresslast_ = nullptr;
  histbranchelaststresscurr_ = nullptr;
  histbranchelaststresslast_ = nullptr;

  // fract
  histfractartstresscurr_ = nullptr;
  histfractartstresslastall_ = nullptr;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::ViscoElastHyper::ViscoElastHyper(Mat::PAR::ViscoElastHyper* params)
    : Mat::ElastHyper(params),
      isovisco_(false),
      viscogenmax_(false),
      viscogeneralizedgenmax_(false),
      viscofract_(false),
      isinitvis_(false)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ViscoElastHyper::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);
  summandProperties_.pack(data);
  add_to_pack(data, isovisco_);
  add_to_pack(data, viscogenmax_);
  add_to_pack(data, viscogeneralizedgenmax_);
  add_to_pack(data, viscofract_);

  anisotropy_.pack_anisotropy(data);

  Core::Communication::PotentiallyUnusedBufferScope summand_scope{data};
  if (params_ != nullptr)  // summands are not accessible in postprocessing mode
  {
    // loop map of associated potential summands
    for (unsigned int p = 0; p < potsum_.size(); ++p)
    {
      potsum_[p]->pack_summand(data);
    }

    //  pack history data
    int histsize = 0;
    if (initialized())
    {
      histsize = histscglast_->size();
    }
    add_to_pack(data, histsize);  // Length of history vector(s)
    for (int var = 0; var < histsize; ++var)
    {
      // insert last converged states
      add_to_pack(data, histscglast_->at(var));
      add_to_pack(data, histmodrcglast_->at(var));
      add_to_pack(data, histstresslast_->at(var));
      add_to_pack(data, histartstresslast_->at(var));

      // insert current iteration states
      add_to_pack(data, histscgcurr_->at(var));
      add_to_pack(data, histmodrcgcurr_->at(var));
      add_to_pack(data, histstresscurr_->at(var));
      add_to_pack(data, histartstresscurr_->at(var));
    }

    if (viscogeneralizedgenmax_)
    {
      for (int var = 0; var < histsize; ++var)
      {
        // insert last converged states
        add_to_pack(data, histbranchstresslast_->at(var));
        add_to_pack(data, histbranchelaststresslast_->at(var));

        // insert current iteration states
        add_to_pack(data, histbranchstresscurr_->at(var));
        add_to_pack(data, histbranchelaststresscurr_->at(var));
      }
    }

    // pack history of FSLS-model
    if (viscofract_)
    {
      // check if history exists
      add_to_pack(data, (histfractartstresslastall_ != nullptr));
      if (!(int)(histfractartstresslastall_ != nullptr))
        FOUR_C_THROW("Something got wrong with your history data.");

      // pack stepsize
      add_to_pack(data, histfractartstresslastall_->at(0).size());
      // pack history values
      for (std::size_t gp = 0; gp < histfractartstresslastall_->size(); ++gp)
        for (std::size_t step = 0; step < histfractartstresslastall_->at(gp).size(); ++step)
          add_to_pack(data, histfractartstresslastall_->at(gp).at(step));
    }
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ViscoElastHyper::unpack(Core::Communication::UnpackBuffer& buffer)
{
  // make sure we have a pristine material
  params_ = nullptr;
  potsum_.clear();

  summandProperties_.clear();
  isovisco_ = false;
  viscogenmax_ = false;
  viscogeneralizedgenmax_ = false;
  viscofract_ = false;

  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid and recover params_
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
        params_ = static_cast<Mat::PAR::ViscoElastHyper*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
  }

  summandProperties_.unpack(buffer);
  extract_from_pack(buffer, isovisco_);
  extract_from_pack(buffer, viscogenmax_);
  extract_from_pack(buffer, viscogeneralizedgenmax_);
  extract_from_pack(buffer, viscofract_);

  anisotropy_.unpack_anisotropy(buffer);

  Core::Communication::PotentiallyUnusedBufferScope summand_scope{buffer};
  if (params_ != nullptr)  // summands are not accessible in postprocessing mode
  {
    // make sure the referenced materials in material list have quick access parameters
    std::vector<int>::const_iterator m;
    for (m = params_->matids_.begin(); m != params_->matids_.end(); ++m)
    {
      const int matid = *m;
      std::shared_ptr<Mat::Elastic::Summand> sum = Mat::Elastic::Summand::factory(matid);
      if (sum == nullptr) FOUR_C_THROW("Failed to allocate");
      potsum_.push_back(sum);
    }

    // loop map of associated potential summands
    for (auto& p : potsum_)
    {
      p->unpack_summand(buffer);
      p->register_anisotropy_extensions(anisotropy_);
    }

    // history data 09/13
    isinitvis_ = true;
    int histsize;
    extract_from_pack(buffer, histsize);

    if (histsize == 0) isinitvis_ = false;

    // initialize last variables
    histscglast_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>(histsize);
    histmodrcglast_ =
        std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>(histsize);
    histstresslast_ =
        std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>(histsize);
    histartstresslast_ =
        std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>(histsize);

    // initialize current variables
    histscgcurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>(histsize);
    histmodrcgcurr_ =
        std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>(histsize);
    histstresscurr_ =
        std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>(histsize);
    histartstresscurr_ =
        std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>(histsize);


    for (int gp = 0; gp < histsize; ++gp)
    {
      // last vectors are unpacked
      extract_from_pack(buffer, histscglast_->at(gp));
      extract_from_pack(buffer, histmodrcglast_->at(gp));
      extract_from_pack(buffer, histstresslast_->at(gp));
      extract_from_pack(buffer, histartstresslast_->at(gp));

      // current iteration states are unpacked
      extract_from_pack(buffer, histscgcurr_->at(gp));
      extract_from_pack(buffer, histmodrcgcurr_->at(gp));
      extract_from_pack(buffer, histstresscurr_->at(gp));
      extract_from_pack(buffer, histartstresscurr_->at(gp));
    }

    if (viscogeneralizedgenmax_)
    {
      histbranchstresslast_ =
          std::make_shared<std::vector<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>>(
              histsize);
      histbranchelaststresslast_ =
          std::make_shared<std::vector<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>>(
              histsize);
      histbranchstresscurr_ =
          std::make_shared<std::vector<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>>(
              histsize);
      histbranchelaststresscurr_ =
          std::make_shared<std::vector<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>>(
              histsize);

      for (int gp = 0; gp < histsize; ++gp)
      {
        // last vectors are unpacked
        extract_from_pack(buffer, histbranchstresslast_->at(gp));
        extract_from_pack(buffer, histbranchelaststresslast_->at(gp));

        // current iteration states are unpacked
        extract_from_pack(buffer, histbranchstresscurr_->at(gp));
        extract_from_pack(buffer, histbranchelaststresscurr_->at(gp));
      }
    }

    // for FSLS-model
    if (viscofract_)
    {
      // check if history data is saved
      bool have_historyalldata;
      extract_from_pack(buffer, have_historyalldata);
      if (!have_historyalldata) FOUR_C_THROW("Something got wrong with your history data.");

      histfractartstresscurr_ =
          std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>(histsize);

      std::size_t histfractartstressall_stepsize;
      extract_from_pack(buffer, histfractartstressall_stepsize);
      histfractartstresslastall_ =
          std::make_shared<std::vector<std::vector<Core::LinAlg::Matrix<6, 1>>>>(
              histsize, std::vector<Core::LinAlg::Matrix<6, 1>>(histfractartstressall_stepsize));
      for (std::size_t gp = 0; gp < static_cast<std::size_t>(histsize); ++gp)
        for (std::size_t step = 0; step < histfractartstressall_stepsize; ++step)
          extract_from_pack(buffer, histfractartstresslastall_->at(gp).at(step));
    }
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ViscoElastHyper::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  // read anisotropy
  anisotropy_.set_number_of_gauss_points(numgp);
  anisotropy_.read_anisotropy_from_element(fibers, coord_system);

  // Setup summands
  for (auto& p : potsum_) p->setup(numgp, fibers, coord_system);

  // find out which formulations are used
  isovisco_ = false;
  viscogenmax_ = false;
  viscogeneralizedgenmax_ = false;
  viscofract_ = false;

  summandProperties_.clear();
  elast_hyper_properties(potsum_, summandProperties_);


  if (summandProperties_.viscoGeneral)
  {
    for (auto& p : potsum_)
    {
      p->specify_visco_formulation(isovisco_, viscogenmax_, viscogeneralizedgenmax_, viscofract_);
    }
  }

  // Initialise/allocate history variables 09/13
  const Core::LinAlg::Matrix<6, 1> emptyvec(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6, 1> idvec(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; ++i) idvec(i) = 1.;

  histscgcurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<6, 1>>>(numgp, idvec);
  histscglast_ = std::make_shared<std::vector<Core::LinAlg::Matrix<6, 1>>>(numgp, idvec);
  histmodrcgcurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<6, 1>>>(numgp, idvec);
  histmodrcglast_ = std::make_shared<std::vector<Core::LinAlg::Matrix<6, 1>>>(numgp, idvec);
  histstresscurr_ =
      std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>(numgp, emptyvec);
  histstresslast_ =
      std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>(numgp, emptyvec);
  histartstresscurr_ =
      std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>(numgp, emptyvec);
  histartstresslast_ =
      std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>(numgp, emptyvec);

  if (viscogeneralizedgenmax_)
  {
    const std::vector<Core::LinAlg::Matrix<6, 1>> emptybigvec(true);
    histbranchstresscurr_ =
        std::make_shared<std::vector<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>>(
            numgp, emptybigvec);
    histbranchstresslast_ =
        std::make_shared<std::vector<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>>(
            numgp, emptybigvec);
    histbranchelaststresscurr_ =
        std::make_shared<std::vector<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>>(
            numgp, emptybigvec);
    histbranchelaststresslast_ =
        std::make_shared<std::vector<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>>(
            numgp, emptybigvec);
  }

  // in case of FSLS-model
  if (viscofract_)
  {
    histfractartstresscurr_ =
        std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>(numgp, emptyvec);
    // set true that history size is known
    histfractartstresslastall_ =
        std::make_shared<std::vector<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>>(
            numgp, std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>(true));
  }

  isinitvis_ = true;

  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::ViscoElastHyper::update()
{
  Mat::ElastHyper::update();

  // Update history values 09/13
  histscglast_ = histscgcurr_;
  histmodrcglast_ = histmodrcgcurr_;
  histstresslast_ = histstresscurr_;
  histartstresslast_ = histartstresscurr_;

  // for FSLS-model
  if (viscofract_)
  {
    // To calculate the fractional derivative the history of all previous timesteps is saved in each
    // gauss-point

    // numsteps
    const Teuchos::ParameterList& sdyn = Global::Problem::instance()->structural_dynamic_params();
    const int numsteps = sdyn.get<int>("NUMSTEP");
    // maximal size of history (in time steps)
    const unsigned int max_hist = numsteps + 1;

    // for each gp
    for (int gp = 0; gp < (int)histfractartstresslastall_->size(); ++gp)
    {
      // add current stress (all gp) at the end of history data
      histfractartstresslastall_->at(gp).push_back(histfractartstresscurr_->at(gp));

      // this is in the moment not used, because all history data is saved
      // if maximal size of history-vector is reached
      if (histfractartstresslastall_->at(gp).size() > max_hist)
      {
        // Hint: If you want to use this functionality, reimplement it with a pointer (abirzle
        // 12/17)

        // save current history data in tmp_vec, but skip the first data
        // (= oldest data, which should deleted)
        std::vector<Core::LinAlg::Matrix<6, 1>> tmp_vec(
            ++histfractartstresslastall_->at(gp).begin(), histfractartstresslastall_->at(gp).end());
        // save data back to history-vector
        histfractartstresslastall_->at(gp) = tmp_vec;
      }
    }
  }

  // initialize current data
  const Core::LinAlg::Matrix<6, 1> emptyvec(Core::LinAlg::Initialization::zero);

  Core::LinAlg::Matrix<6, 1> idvec(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; ++i) idvec(i) = 1.;
  const int numgp = histscglast_->size();

  histscgcurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<6, 1>>>(numgp, idvec);
  histmodrcgcurr_ = std::make_shared<std::vector<Core::LinAlg::Matrix<6, 1>>>(numgp, idvec);
  histstresscurr_ =
      std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>(numgp, emptyvec);
  histartstresscurr_ =
      std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>(numgp, emptyvec);
  histfractartstresscurr_ =
      std::make_shared<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>(numgp, emptyvec);

  if (viscogeneralizedgenmax_)
  {
    histbranchstresslast_ = histbranchstresscurr_;
    histbranchelaststresslast_ = histbranchelaststresscurr_;

    const std::vector<Core::LinAlg::Matrix<6, 1>> emptybigvec(true);
    histbranchstresscurr_ =
        std::make_shared<std::vector<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>>(
            numgp, emptybigvec);
    histbranchelaststresscurr_ =
        std::make_shared<std::vector<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>>(
            numgp, emptybigvec);
  }

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ViscoElastHyper::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  const Core::LinAlg::Matrix<6, 1> glstrain_mat =
      Core::LinAlg::make_strain_like_voigt_matrix(glstrain);
  Core::LinAlg::Matrix<6, 1> stress_view = Core::LinAlg::make_stress_like_voigt_view(stress);
  Core::LinAlg::Matrix<6, 6> cmat_view = Core::LinAlg::make_stress_like_voigt_view(cmat);

  Core::LinAlg::Matrix<6, 1> modC_strain(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6, 1> id2(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6, 1> modrcg(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6, 6> id4(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6, 6> id4sharp(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> prinv(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> modinv(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<7, 1> rateinv(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<7, 1> modrateinv(Core::LinAlg::Initialization::zero);

  Core::LinAlg::Matrix<3, 1> dPI(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6, 1> ddPII(Core::LinAlg::Initialization::zero);

  Core::LinAlg::Matrix<6, 1> scgrate(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6, 1> modrcgrate(Core::LinAlg::Initialization::zero);

  // for extension: Core::LinAlg::Matrix<6,1> modicgrate(true);
  Core::LinAlg::Matrix<8, 1> mu(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<8, 1> modmu(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<33, 1> xi(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<33, 1> modxi(Core::LinAlg::Initialization::zero);

  Core::LinAlg::SymmetricTensor<double, 3, 3> C{};
  evaluate_right_cauchy_green_strain_like_voigt(glstrain, C);
  Core::LinAlg::SymmetricTensor<double, 3, 3> iC = Core::LinAlg::inv(C);

  Core::LinAlg::Matrix<6, 1> C_stress = Core::LinAlg::make_stress_like_voigt_view(C);
  Core::LinAlg::Matrix<6, 1> iC_stress = Core::LinAlg::make_stress_like_voigt_view(iC);
  Core::LinAlg::Matrix<6, 1> C_strain;
  Core::LinAlg::Voigt::Stresses::to_strain_like(C_stress, C_strain);
  Core::LinAlg::Matrix<6, 1> iC_strain;
  Core::LinAlg::Voigt::Stresses::to_strain_like(iC_stress, iC_strain);
  Core::LinAlg::Voigt::Stresses::invariants_principal(prinv, C_stress);


  Core::LinAlg::Voigt::identity_matrix(id2);

  using VoigtNotation = Core::LinAlg::Voigt::NotationType;
  Core::LinAlg::Voigt::fourth_order_identity_matrix<VoigtNotation::stress, VoigtNotation::stress>(
      id4sharp);
  Core::LinAlg::Voigt::fourth_order_identity_matrix<VoigtNotation::stress, VoigtNotation::strain>(
      id4);

  elast_hyper_evaluate_invariant_derivatives(
      prinv, dPI, ddPII, potsum_, summandProperties_, gp, eleGID);

  if (isovisco_)
  {
    if (summandProperties_.isomod)
    {
      // calculate modified invariants
      invariants_modified(modinv, prinv);
    }
    // calculate viscous quantities
    evaluate_kin_quant_vis(C_strain, C_stress, iC_stress, prinv, rateinv, modC_strain, params,
        context, scgrate, modrcgrate, modrateinv, gp);
    evaluate_mu_xi(prinv, modinv, mu, modmu, xi, modxi, rateinv, modrateinv, params, gp, eleGID);
  }

  // blank resulting quantities
  // ... even if it is an implicit law that cmat is zero upon input
  stress_view.clear();
  cmat_view.clear();

  // add isotropic part
  elast_hyper_add_isotropic_stress_cmat(stress, cmat, C, iC, prinv, dPI, ddPII);


  // add viscous part
  if (isovisco_)
  {
    if (summandProperties_.isomod)
    {
      // add viscous part decoupled
      Core::LinAlg::Matrix<NUM_STRESS_3D, 1> stressisomodisovisco(
          Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmatisomodisovisco(
          Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<NUM_STRESS_3D, 1> stressisomodvolvisco(
          Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmatisomodvolvisco(
          Core::LinAlg::Initialization::zero);
      evaluate_iso_visco_modified(stressisomodisovisco, stressisomodvolvisco, cmatisomodisovisco,
          cmatisomodvolvisco, prinv, modinv, modmu, modxi, C_strain, id2, iC_stress, id4,
          modrcgrate);
      stress_view.update(1.0, stressisomodisovisco, 1.0);
      stress_view.update(1.0, stressisomodvolvisco, 1.0);
      cmat_view.update(1.0, cmatisomodisovisco, 1.0);
      cmat_view.update(1.0, cmatisomodvolvisco, 1.0);
    }

    if (summandProperties_.isoprinc)
    {
      // add viscous part coupled
      Core::LinAlg::Matrix<NUM_STRESS_3D, 1> stressisovisco(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmatisovisco(
          Core::LinAlg::Initialization::zero);
      evaluate_iso_visco_principal(stressisovisco, cmatisovisco, mu, xi, id4sharp, scgrate);
      stress_view.update(1.0, stressisovisco, 1.0);
      cmat_view.update(1.0, cmatisovisco, 1.0);
    }
  }

  // add contribution of viscogenmax-material
  if (viscogenmax_)
  {
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Q(
        Core::LinAlg::Initialization::zero);  // artificial viscous stress
    Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmatq(Core::LinAlg::Initialization::zero);
    evaluate_visco_gen_max(&stress_view, &cmat_view, Q, cmatq, params, context, gp);
    stress_view.update(1.0, Q, 1.0);
    cmat_view.update(1.0, cmatq, 1.0);
  }

  // add contribution of generalized Maxwell model
  if (viscogeneralizedgenmax_)
  {
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Q(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmatq(Core::LinAlg::Initialization::zero);
    evaluate_visco_generalized_gen_max(Q, cmatq, params, context, &glstrain_mat, gp, eleGID);
    stress_view.update(1.0, Q, 1.0);
    cmat_view.update(1.0, cmatq, 1.0);
  }

  // add contribution of viscofract-material
  if (viscofract_)
  {
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Q(
        Core::LinAlg::Initialization::zero);  // artificial viscous stress
    Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmatq(Core::LinAlg::Initialization::zero);
    evaluate_visco_fract(stress_view, cmat_view, Q, cmatq, params, context, gp);
    stress_view.update(1.0, Q, 1.);
    cmat_view.update(1.0, cmatq, 1.);
  }


  /*----------------------------------------------------------------------*/
  // coefficients in principal stretches
  if (summandProperties_.coeffStretchesPrinc || summandProperties_.coeffStretchesMod)
  {
    elast_hyper_add_response_stretches(cmat, stress, C, potsum_, summandProperties_, gp, eleGID);
  }

  /*----------------------------------------------------------------------*/
  // Do all the anisotropic stuff!
  if (summandProperties_.anisoprinc)
  {
    elast_hyper_add_anisotropic_princ(stress, cmat, C, params, gp, eleGID, potsum_);
  }

  if (summandProperties_.anisomod)
  {
    elast_hyper_add_anisotropic_mod(stress, cmat, C, iC, prinv, gp, eleGID, params, potsum_);
  }
}

/*----------------------------------------------------------------------*/
/* Evaluate Quantities for viscous Part                           09/13 */
/*----------------------------------------------------------------------*/
void Mat::ViscoElastHyper::evaluate_kin_quant_vis(Core::LinAlg::Matrix<6, 1>& rcg,
    Core::LinAlg::Matrix<6, 1>& scg, Core::LinAlg::Matrix<6, 1>& icg,
    Core::LinAlg::Matrix<3, 1>& prinv, Core::LinAlg::Matrix<7, 1>& rateinv,
    Core::LinAlg::Matrix<6, 1>& modrcg, const Teuchos::ParameterList& params,
    const EvaluationContext& context, Core::LinAlg::Matrix<6, 1>& scgrate,
    Core::LinAlg::Matrix<6, 1>& modrcgrate, Core::LinAlg::Matrix<7, 1>& modrateinv, const int gp)
{
  // time derivative
  // -------------------------------------------------------------------
  // get time algorithmic parameters
  FOUR_C_ASSERT(context.time_step_size, "Time step size not given in evaluation context.");
  double dt = *context.time_step_size;

  // modrcg : \overline{C} = J^{-\frac{2}{3}} C
  const double modscale = std::pow(prinv(2), -1. / 3.);
  modrcg.update(modscale, rcg);

  // read history
  Core::LinAlg::Matrix<6, 1> scglast(histscglast_->at(gp));
  Core::LinAlg::Matrix<6, 1> modrcglast(histmodrcglast_->at(gp));

  // Update history of Cauchy-Green Tensor
  histscgcurr_->at(gp) = scg;        // principal material: store C^{n}
  histmodrcgcurr_->at(gp) = modrcg;  // decoupled material: store \overline{C}^{n}

  // rate of Cauchy-Green Tensor
  // REMARK: strain-like 6-Voigt vector
  scgrate.update(1.0, scg, 1.0);  // principal material: \dot{C} = \frac{C^n - C^{n-1}}{\Delta t}
  scgrate.update(-1.0, scglast, 1.0);
  scgrate.scale(1 / dt);

  modrcgrate.update(1.0, modrcg, 1.0);  // decoupled material: \overline{\dot{C}} =
                                        // \frac{\overline{C}^n - \overline{C}^{n-1}}{\Delta t}
  modrcgrate.update(-1.0, modrcglast, 1.0);
  modrcgrate.scale(1 / dt);

  // invariants
  // -------------------------------------------------------------------
  // Second Invariant of modrcgrate \bar{J}_2 = \frac{1}{2} \tr (\dot{\overline{C^2}}
  modrateinv(1) =
      0.5 * (modrcgrate(0) * modrcgrate(0) + modrcgrate(1) * modrcgrate(1) +
                modrcgrate(2) * modrcgrate(2) + .5 * modrcgrate(3) * modrcgrate(3) +
                .5 * modrcgrate(4) * modrcgrate(4) + .5 * modrcgrate(5) * modrcgrate(5));


  // For further extension of material law (not necessary at the moment)
  /*
  // necessary transfer variable: Core::LinAlg::Matrix<6,1>& modicgrate
  // \overline{J}_3 = determinant of modified rate of right Cauchy-Green-Tensor
  modrateinv(2) = modrcgrate(0)*modrcgrate(1)*modrcgrate(2)
      + 0.25 * modrcgrate(3)*modrcgrate(4)*modrcgrate(5)
      - 0.25 * modrcgrate(1)*modrcgrate(5)*modrcgrate(5)
      - 0.25 * modrcgrate(2)*modrcgrate(3)*modrcgrate(3)
      - 0.25 * modrcgrate(0)*modrcgrate(4)*modrcgrate(4);

  // invert modified rate of right Cauchy-Green tensor
  // REMARK: stress-like 6-Voigt vector
  {
    modicgrate(0) = ( modrcgrate(1)*modrcgrate(2) - 0.25*modrcgrate(4)*modrcgrate(4) ) /
  modrateinv(2); modicgrate(1) = ( modrcgrate(0)*modrcgrate(2) - 0.25*modrcgrate(5)*modrcgrate(5) )
  / modrateinv(2); modicgrate(2) = ( modrcgrate(0)*modrcgrate(1) - 0.25*modrcgrate(3)*modrcgrate(3)
  ) / modrateinv(2); modicgrate(3) = ( 0.25*modrcgrate(5)*modrcgrate(4) -
  0.5*modrcgrate(3)*modrcgrate(2) ) / modrateinv(2); modicgrate(4) = (
  0.25*modrcgrate(3)*modrcgrate(5) - 0.5*modrcgrate(0)*modrcgrate(4) ) / modrateinv(2);
    modicgrate(5) = ( 0.25*modrcgrate(3)*modrcgrate(4) - 0.5*modrcgrate(5)*modrcgrate(1) ) /
  modrateinv(2);
  }
   */
}

/*----------------------------------------------------------------------*/
/* Evaluate Factors for viscous Quantities                        09/13 */
/*----------------------------------------------------------------------*/
void Mat::ViscoElastHyper::evaluate_mu_xi(Core::LinAlg::Matrix<3, 1>& prinv,
    Core::LinAlg::Matrix<3, 1>& modinv, Core::LinAlg::Matrix<8, 1>& mu,
    Core::LinAlg::Matrix<8, 1>& modmu, Core::LinAlg::Matrix<33, 1>& xi,
    Core::LinAlg::Matrix<33, 1>& modxi, Core::LinAlg::Matrix<7, 1>& rateinv,
    Core::LinAlg::Matrix<7, 1>& modrateinv, const Teuchos::ParameterList& params, const int gp,
    const int eleGID)
{
  // principal materials
  if (summandProperties_.isoprinc)
  {
    // loop map of associated potential summands
    for (unsigned int p = 0; p < potsum_.size(); ++p)
    {
      potsum_[p]->add_coefficients_visco_principal(prinv, mu, xi, rateinv, params, gp, eleGID);
    }
  }

  // decoupled (volumetric or isochoric) materials
  if (summandProperties_.isomod)
  {
    // loop map of associated potential summands
    for (unsigned int p = 0; p < potsum_.size(); ++p)
    {
      potsum_[p]->add_coefficients_visco_modified(
          modinv, modmu, modxi, modrateinv, params, gp, eleGID);
    }
  }
}

/*----------------------------------------------------------------------*/
/* stress and constitutive tensor for principal viscous materials       */
/*                                                        pfaller May15 */
/*----------------------------------------------------------------------*/
void Mat::ViscoElastHyper::evaluate_iso_visco_principal(Core::LinAlg::Matrix<6, 1>& stress,
    Core::LinAlg::Matrix<6, 6>& cmat, Core::LinAlg::Matrix<8, 1>& mu,
    Core::LinAlg::Matrix<33, 1>& xi, Core::LinAlg::Matrix<6, 6>& id4sharp,
    Core::LinAlg::Matrix<6, 1>& scgrate)
{
  // contribution: \dot{C}
  stress.update(mu(2), scgrate, 1.0);

  // contribution: id4sharp_{ijkl} = 1/2 (\delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk})
  cmat.update(xi(2), id4sharp, 1.0);

  return;
}

/*----------------------------------------------------------------------*/
/* stress and constitutive tensor for decoupled viscous materials 09/13 */
/*----------------------------------------------------------------------*/
void Mat::ViscoElastHyper::evaluate_iso_visco_modified(
    Core::LinAlg::Matrix<6, 1>& stressisomodisovisco,
    Core::LinAlg::Matrix<6, 1>& stressisomodvolvisco,
    Core::LinAlg::Matrix<6, 6>& cmatisomodisovisco, Core::LinAlg::Matrix<6, 6>& cmatisomodvolvisco,
    Core::LinAlg::Matrix<3, 1>& prinv, Core::LinAlg::Matrix<3, 1>& modinv,
    Core::LinAlg::Matrix<8, 1>& modmu, Core::LinAlg::Matrix<33, 1>& modxi,
    Core::LinAlg::Matrix<6, 1>& rcg, Core::LinAlg::Matrix<6, 1>& id2,
    Core::LinAlg::Matrix<6, 1>& icg, Core::LinAlg::Matrix<6, 6>& id4,
    Core::LinAlg::Matrix<6, 1>& modrcgrate)
{
  // define necessary variables
  const double modscale = std::pow(prinv(2), -1. / 3.);

  // 2nd Piola Kirchhoff stresses

  // isochoric contribution
  Core::LinAlg::Matrix<6, 1> modstress(Core::LinAlg::Initialization::zero);
  modstress.update(modmu(1), id2);
  modstress.update(modmu(2), modrcgrate, 1.0);
  // build 4-tensor for projection as 6x6 tensor
  Core::LinAlg::Matrix<6, 6> Projection;
  Projection.multiply_nt(1. / 3., icg, rcg);
  Projection.update(1.0, id4, -1.0);
  // isochoric stress
  stressisomodisovisco.multiply_nn(modscale, Projection, modstress, 1.0);

  // volumetric contribution:
  // with visco_isoratedep: no volumetric part added --> always 0


  // Constitutive Tensor

  // isochoric contribution
  // modified constitutive tensor
  Core::LinAlg::Matrix<6, 6> modcmat(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6, 6> modcmat2(Core::LinAlg::Initialization::zero);
  // contribution:  Id \otimes \overline{\dot{C}} + \overline{\dot{C}} \otimes Id
  modcmat.multiply_nt(modxi(1), id2, modrcgrate);
  modcmat.multiply_nt(modxi(1), modrcgrate, id2, 1.0);
  // contribution: Id4
  modcmat.update(modxi(2), id4, 1.0);
  // scaling
  modcmat.scale(std::pow(modinv(2), -4. / 3.));
  // contribution: P:\overline{C}:P
  modcmat2.multiply_nn(Projection, modcmat);
  cmatisomodisovisco.multiply_nt(1.0, modcmat2, Projection, 1.0);
  // contribution: 2/3*Tr(J^(-2/3)modstress) (Cinv \odot Cinv - 1/3 Cinv \otimes Cinv)
  modcmat.clear();
  modcmat.multiply_nt(-1.0 / 3.0, icg, icg);
  Core::LinAlg::FourTensorOperations::add_holzapfel_product(modcmat, icg, 1.0);
  Core::LinAlg::Matrix<1, 1> tracemat;
  tracemat.multiply_tn(2. / 3. * std::pow(modinv(2), -2. / 3.), modstress, rcg);
  cmatisomodisovisco.update(tracemat(0, 0), modcmat, 1.0);
  // contribution: -2/3 (Cinv \otimes S_iso^v + S_iso^v \otimes Cinv)
  cmatisomodisovisco.multiply_nt(-2. / 3., icg, stressisomodisovisco, 1.0);
  cmatisomodisovisco.multiply_nt(-2. / 3., stressisomodisovisco, icg, 1.0);

  // volumetric contribution:
  // with visco_isoratedep: no volumetric part added --> always 0

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ViscoElastHyper::evaluate_visco_gen_max(Core::LinAlg::Matrix<6, 1>* stress,
    Core::LinAlg::Matrix<6, 6>* cmat, Core::LinAlg::Matrix<6, 1>& Q,
    Core::LinAlg::Matrix<6, 6>& cmatq, const Teuchos::ParameterList& params,
    const EvaluationContext& context, const int gp)
{
  // initialize material parameters
  double tau = -1.0;
  double beta = -1.0;
  std::string solve = "";
  // alpha not used in viscogenmax (viscofract with alpha=1 results in viscogenmax-material)
  double alpha(true);

  // read material parameters of viscogenmax material
  for (unsigned int p = 0; p < potsum_.size(); ++p)
    potsum_[p]->read_material_parameters_visco(tau, beta, alpha, solve);

  if (solve == "OST")
  {
    // initialize scalars
    double lambdascalar1(true);
    double lambdascalar2(true);
    double deltascalar(true);
    double theta = 0.5;

    // get theta of global time integration scheme to use it here
    // if global time integration scheme is not ONESTEPTHETA, theta is by default = 0.5 (abirzle
    // 09/14)
    const auto dyntype = Teuchos::getIntegralValue<Inpar::Solid::DynamicType>(
        Global::Problem::instance()->structural_dynamic_params(), "DYNAMICTYPE");
    if (dyntype == Inpar::Solid::DynamicType::OneStepTheta)
      theta = Global::Problem::instance()
                  ->structural_dynamic_params()
                  .sublist("ONESTEPTHETA")
                  .get<double>("THETA");

    // get time algorithmic parameters
    // NOTE: dt can be zero (in restart of STI) for Generalized Maxwell model
    // there is no special treatment required. Adaptation for Kelvin-Voigt were necessary.
    FOUR_C_ASSERT(context.time_step_size, "Time step size not given in evaluation context.");
    double dt = *context.time_step_size;  // TIMESTEP in the input file

    // evaluate scalars to compute
    // Q^(n+1) = tau/(tau+theta*dt) [(tau-dt+theta*dt)/tau Q + beta(S^(n+1) - S^n)]
    lambdascalar1 = tau / (tau + theta * dt);
    lambdascalar2 = (tau - dt + theta * dt) / tau;

    // factor to calculate visco stiffness matrix from elastic stiffness matrix
    // old Version: scalarvisco =
    // 1+beta_isoprinc*exp(-dt/(2*tau_isoprinc));//+alpha1*tau/(tau+theta*dt); Alines' version:
    // scalarvisco = beta_isoprinc*exp(-dt/(2*tau_isoprinc));//+alpha1*tau/(tau+theta*dt); Scalar
    // consistent to derivation of Q with one-step-theta-schema (abirzle 09/14):
    deltascalar = beta * lambdascalar1;

    // read history
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> S_n(histstresslast_->at(gp));
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Q_n(histartstresslast_->at(gp));

    // calculate artificial viscous stresses Q
    Q.update(lambdascalar2, Q_n, 1.0);
    Q.update(beta, *stress, 1.0);
    Q.update(-beta, S_n, 1.0);
    Q.scale(lambdascalar1);  // Q^(n+1) = lambdascalar1* [lambdascalar2* Q^n + beta*(S^(n+1) - S^n)]


    // update history
    histstresscurr_->at(gp) = *stress;
    histartstresscurr_->at(gp) = Q;

    // viscous constitutive tensor
    // contribution : Cmat_vis = Cmat_inf*deltascalar
    cmatq.update(deltascalar, *cmat, 1.0);
  }
  else if (solve == "CONVOL")
  {
    // initialize scalars
    double xiscalar1(true);
    double xiscalar2(true);

    // get time algorithmic parameters
    // NOTE: dt can be zero (in restart of STI) for Generalized Maxwell model
    // there is no special treatment required. Adaptation for Kelvin-Voigt were necessary.
    FOUR_C_ASSERT(context.time_step_size, "Time step size not given in evaluation context.");
    double dt = *context.time_step_size;  // TIMESTEP in the input file
    // evaluate scalars to compute
    // Q_(n+1)=\exp(2*(-dt/(2*tau)))*Q_n+exp(-dt/(2*tau))*beta*(S_(n+1)-S_n)

    xiscalar1 = exp(-dt / tau);
    xiscalar2 = exp(-dt / (2 * tau)) * beta;

    // read history
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> S_n(histstresslast_->at(gp));
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Q_n(histartstresslast_->at(gp));

    // calculate artificial viscous stresses Q
    Q.update(xiscalar1, Q_n, 1.0);
    Q.update(xiscalar2, *stress, 1.0);
    Q.update(-xiscalar2, S_n, 1.0);  // Q_(n+1) = xiscalar1* Q_n + xiscalar2*(S_(n+1) - S_n)

    // update history
    histstresscurr_->at(gp) = *stress;
    histartstresscurr_->at(gp) = Q;

    // viscous constitutive tensor
    // contribution : Cmat_vis = Cmat_inf*beta*exp(-dt/(2*tau))
    cmatq.update(xiscalar2, *cmat, 1.0);
  }
  else
    FOUR_C_THROW("Invalid input. Try valid input OST or CONVOL");
  return;
}  // end evaluate_visco_gen_max

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ViscoElastHyper::evaluate_visco_generalized_gen_max(Core::LinAlg::Matrix<6, 1>& Q,
    Core::LinAlg::Matrix<6, 6>& cmatq, const Teuchos::ParameterList& params,
    const Mat::EvaluationContext& context, const Core::LinAlg::Matrix<6, 1>* glstrain, const int gp,
    const int eleGID)
{
  int numbranch = -1;
  double tau = -1.0;
  std::string solve = "";
  const std::vector<int>* matids = nullptr;
  std::vector<std::shared_ptr<Mat::Elastic::Summand>> branchpotsum(
      0);  // vector of summands in one branch
  std::vector<std::vector<std::shared_ptr<Mat::Elastic::Summand>>> branchespotsum(
      0);  // vector for each branch of vectors of summands in each branch

  // get parameters of ViscoGeneralizedGenMax
  for (unsigned int p = 0; p < potsum_.size(); ++p)
  {
    std::shared_ptr<Mat::Elastic::GeneralizedGenMax> GeneralizedGenMax =
        std::dynamic_pointer_cast<Mat::Elastic::GeneralizedGenMax>(potsum_[p]);

    if (GeneralizedGenMax != nullptr)
    {
      GeneralizedGenMax->read_material_parameters(numbranch, matids, solve);
      branchespotsum = GeneralizedGenMax->get_branchespotsum();
    }
  }

  Core::LinAlg::Matrix<6, 6> cmatqbranch(Core::LinAlg::Initialization::zero);
  std::vector<Core::LinAlg::Matrix<6, 1>> S(numbranch);
  std::vector<Core::LinAlg::Matrix<6, 1>> Qbranch(numbranch);
  std::vector<Core::LinAlg::Matrix<6, 1>> S_n(numbranch);
  std::vector<Core::LinAlg::Matrix<6, 1>> Q_n(numbranch);

  // read history
  S_n = histbranchelaststresslast_->at(gp);
  Q_n = histbranchstresslast_->at(gp);


  // check size, correct if needed (only needed in the first time step)
  if (S_n.size() != (unsigned int)(abs(numbranch)))
  {
    std::vector<Core::LinAlg::Matrix<6, 1>> S_n(numbranch);
    for (int branch = 0; branch < numbranch; branch++) S_n.at(branch).clear();
    histbranchelaststresslast_->at(gp) = S_n;
  }

  if (Q_n.size() != (unsigned int)(abs(numbranch)))
  {
    std::vector<Core::LinAlg::Matrix<6, 1>> Q_n(numbranch);
    for (int branch = 0; branch < numbranch; branch++) Q_n.at(branch).clear();
    histbranchstresslast_->at(gp) = Q_n;
  }

  S_n = histbranchelaststresslast_->at(gp);
  Q_n = histbranchstresslast_->at(gp);

  // save switches
  SummandProperties branchProperties;

  /////////////////////////////////////////////////
  // Loop over all viscoelastic Maxwell branches //
  /////////////////////////////////////////////////
  for (int i = 0; i < numbranch; ++i)
  {
    // get parameter of each visco branch
    branchpotsum = branchespotsum[i];

    branchProperties.clear();
    elast_hyper_properties(branchpotsum, branchProperties);

    if (isovisco_)
      FOUR_C_THROW("case isovisco for branch in generalized Maxwell model not yet considered!");
    if (branchProperties.anisoprinc)
      FOUR_C_THROW("case anisoprinc for branch in generalized Maxwell model not yet considered!");
    if (branchProperties.anisomod)
      FOUR_C_THROW("case anisomod for branch in generalized Maxwell model not yet considered!");

    Core::LinAlg::Matrix<6, 1> modrcg(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1> prinv(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1> dPI(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 1> ddPII(Core::LinAlg::Initialization::zero);

    Core::LinAlg::Matrix<6, 1> gl_stress;
    Core::LinAlg::Voigt::Strains::to_stress_like(*glstrain, gl_stress);

    Core::LinAlg::SymmetricTensor<double, 3, 3> gl =
        Core::LinAlg::make_symmetric_tensor_from_stress_like_voigt_matrix(gl_stress);

    Core::LinAlg::SymmetricTensor<double, 3, 3> C;
    evaluate_right_cauchy_green_strain_like_voigt(gl, C);

    Core::LinAlg::SymmetricTensor<double, 3, 3> iC = Core::LinAlg::inv(C);

    Core::LinAlg::Matrix<6, 1> C_strain(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 1> iC_strain(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Voigt::Stresses::to_strain_like(
        Core::LinAlg::make_stress_like_voigt_view(C), C_strain);
    Core::LinAlg::Voigt::Stresses::to_strain_like(
        Core::LinAlg::make_stress_like_voigt_view(iC), iC_strain);

    Core::LinAlg::Voigt::Strains::invariants_principal(prinv, C_strain);
    elast_hyper_evaluate_invariant_derivatives(
        prinv, dPI, ddPII, branchpotsum, branchProperties, gp, eleGID);

    // blank resulting quantities
    // ... even if it is an implicit law that cmat is zero upon input
    S.at(i).clear();
    cmatqbranch.clear();

    // build stress response and elasticity tensor
    Core::LinAlg::SymmetricTensor<double, 3, 3> stressiso{};
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> cmatiso{};
    elast_hyper_add_isotropic_stress_cmat(stressiso, cmatiso, C, iC, prinv, dPI, ddPII);
    S.at(i).update(1.0, Core::LinAlg::make_stress_like_voigt_view(stressiso), 1.0);
    cmatqbranch.update(1.0, Core::LinAlg::make_stress_like_voigt_view(cmatiso), 1.0);

    // get Parameter of ViscoGeneralizedGenMax
    for (unsigned int q = 0; q < branchpotsum.size(); ++q)
    {
      std::shared_ptr<Mat::Elastic::ViscoPart> ViscoPart =
          std::dynamic_pointer_cast<Mat::Elastic::ViscoPart>(branchpotsum[q]);
      if (ViscoPart != nullptr) ViscoPart->read_material_parameters(tau);
    }

    // make sure Qbranch in this branch is empty
    Qbranch.at(i).clear();
    double deltascalar = 1.0;
    if (solve == "OST")
    {
      // initialize scalars
      double lambdascalar1(true);
      double lambdascalar2(true);
      double theta = 0.5;

      // get theta of global time integration scheme to use it here
      // if global time integration scheme is not ONESTEPTHETA, theta is by default = 0.5 (abirzle
      // 09/14)
      const auto dyntype = Teuchos::getIntegralValue<Inpar::Solid::DynamicType>(
          Global::Problem::instance()->structural_dynamic_params(), "DYNAMICTYPE");
      if (dyntype == Inpar::Solid::DynamicType::OneStepTheta)
        theta = Global::Problem::instance()
                    ->structural_dynamic_params()
                    .sublist("ONESTEPTHETA")
                    .get<double>("THETA");

      // get time algorithmic parameters
      // NOTE: dt can be zero (in restart of STI) for Generalized Maxwell model
      // there is no special treatment required. Adaptation for Kelvin-Voigt were necessary.
      FOUR_C_ASSERT(context.time_step_size, "Time step size not given in evaluation context.");
      double dt = *context.time_step_size;  // TIMESTEP in the input file

      // evaluate scalars to compute
      // Q^(n+1) = tau/(tau+theta*dt) [(tau-dt+theta*dt)/tau Q + beta(S^(n+1) - S^n)]
      lambdascalar1 = tau / (tau + theta * dt);
      lambdascalar2 = (tau - dt + theta * dt) / tau;

      // see evaluate_visco_gen_max
      deltascalar = lambdascalar1;

      // calculate artificial viscous stresses Q
      // Q_(n+1) = lambdascalar1*[lamdascalar2* Q_n + (Sa_(n+1) - Sa_n)]
      Qbranch.at(i).update(lambdascalar2, Q_n.at(i), 1.0);
      Qbranch.at(i).update(1.0, S.at(i), 1.0);
      Qbranch.at(i).update(-1.0, S_n.at(i), 1.0);
      Qbranch.at(i).scale(lambdascalar1);
    }
    else if (solve == "CONVOL")
    {
      // initialize scalars
      double xiscalar1(true);
      double xiscalar2(true);
      FOUR_C_ASSERT(context.time_step_size, "Time step size not given in evaluation context.");
      double dt = *context.time_step_size;

      xiscalar1 = exp(-dt / tau);
      xiscalar2 = exp(-dt / (2 * tau));

      deltascalar = xiscalar2;

      // calculate artificial stresses Q
      // Q_(n+1) = xiscalar1* Q_n + xiscalar2*(Sa_(n+1) - Sa_n)
      Qbranch.at(i).update(xiscalar1, Q_n.at(i), 1.0);
      Qbranch.at(i).update(xiscalar2, S.at(i), 1.0);
      Qbranch.at(i).update(-xiscalar2, S_n.at(i), 1.0);
    }
    else
      FOUR_C_THROW("Invalid input. Try valid input THETA or CONVOL");

    // sum up branches
    Q.update(1.0, Qbranch.at(i), 1.0);
    cmatq.update(deltascalar, cmatqbranch, 1.0);

  }  // end for loop over branches

  // update history
  histbranchelaststresscurr_->at(gp) = S;
  histbranchstresscurr_->at(gp) = Qbranch;
}  // evaluate_visco_generalized_gen_max


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ViscoElastHyper::evaluate_visco_fract(Core::LinAlg::Matrix<6, 1> stress,
    Core::LinAlg::Matrix<6, 6> cmat, Core::LinAlg::Matrix<6, 1>& Q,
    Core::LinAlg::Matrix<6, 6>& cmatq, const Teuchos::ParameterList& params,
    const EvaluationContext& context, const int gp)
{
  // initialize parameters
  double tau(true);
  double alpha(true);
  double beta(true);
  // string not used in viscofract
  std::string solve = "";

  // read material parameters of viscofract-material
  for (unsigned int p = 0; p < potsum_.size(); ++p)
  {
    potsum_[p]->read_material_parameters_visco(tau, beta, alpha, solve);
  }

  // safety checks for alpha
  if (alpha == 1)
    FOUR_C_THROW("Alpha cannot be 1 in fractional viscoelasticity. Use Genmax instead.");

  if (alpha < 0) FOUR_C_THROW("Alpha has to be between 0 and 1 in fractional viscoelasticity.");

  // get time algorithmic parameters
  // NOTE: dt can be zero (in restart of STI) for this model
  // there is no special treatment required.
  FOUR_C_ASSERT(context.time_step_size, "Time step size not given in evaluation context.");
  double dt = *context.time_step_size;


  // read history of last time step at gp
  // -> Q_n and history size
  int hs = histfractartstresslastall_->at(0).size();  // history size
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Q_n(histfractartstresslastall_->at(gp).at(hs - 1));


  // calculate artificial history stress Qq with weights b_j
  // Qq = sum[j=1 up to j=n][b_j*Q_(n+1-j)] (short: b*Qj)

  // b_j = gamma(j-alpha)/[gamma(-alpha)*gamma(j+1)]
  // with recurstion formula for gamma functions b_j shortens to
  // b_j = (j-1-alpha)/j * b_(j-1)
  // with b_0 = 1 and b_1 = -alpha ...
  double bj = 1.;   // b_0=1
  double fac = 1.;  // pre-factor (j-1-alpha)/j  for calculation of b
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Qq(Core::LinAlg::Initialization::zero);

  // j=1...n, hs=n
  for (int j = 1; j <= hs; j++)
  {
    fac = (j - 1. - alpha) / j;
    bj = bj * fac;

    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Qj(
        histfractartstresslastall_->at(gp).at(hs - j));  //(hs-1) correlates with last entry = n
    Qq.update(bj, Qj, 1.0);
  }


  // calculate artificial stress Q

  // Version 1: As in Adolfson and Enelund (2003): Fractional Derivative Visocelasticity at Large
  // Deformations
  //  // initialize and evaluate scalars to compute
  //  // Q^(n+1) = [((dt/tau)^alpha)/(1+theta*(dt/tau)^alpha)]*[theta*S^(n+1)+(1-theta)(S^n-Q^n)]-
  //  //           [1/(1+theta*(dt/tau)^alpha)]*Qq^n


  // Version 2: Anna's Version of calculation
  // Difference:  1.) No one-step theta schema necessary
  //              2.) Introduce beta
  // Q^(n+1) = (dt^alpha / (dt^alpha + tau^alpha))*S^(n+1) - (tau^alpha / (dt^alpha +
  // tau^alpha))*Qq^n
  double dtalpha = std::pow(dt, alpha);
  double taualpha = std::pow(tau, alpha);
  double lambdascalar1 = dtalpha / (dtalpha + taualpha);
  double lambdascalar2 = -1. * taualpha / (dtalpha + taualpha);

  Q.update(lambdascalar1 * beta, stress, 0.);
  Q.update(lambdascalar2, Qq, 1.);


  // update history for next step
  histfractartstresscurr_->at(gp) = Q;  // Q_n+1


  // calculate final stress here and in Evaluate
  // S = elastic stress of Psi
  // S_2 = S ; S_1 = beta*S ; Q = Q(S1) = Q(beta*S)
  // S_final = S + beta*S - Q(beta*S)
  Q.update(beta, stress, -1.);

  // viscos constitutive tensor
  cmatq.update(lambdascalar1 * beta, cmat, 0.);  // contribution of Q
  cmatq.update(beta, cmat, -1.);


  return;
}

FOUR_C_NAMESPACE_CLOSE
