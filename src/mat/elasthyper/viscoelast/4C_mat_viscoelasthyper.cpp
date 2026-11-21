// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_viscoelasthyper.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_utils_densematrix_inverse.hpp"
#include "4C_mat_elast_summand.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_mat_viscoelast_fsls.hpp"
#include "4C_mat_viscoelast_generalizedmaxwell.hpp"
#include "4C_mat_viscoelast_isoratedep.hpp"
#include "4C_mat_viscoelast_quasilineargeneralizedmaxwell.hpp"
#include "4C_mat_viscoelast_summand.hpp"

#include <optional>

FOUR_C_NAMESPACE_OPEN

namespace
{
  bool is_visco_material_type_for_split_input(const Core::Materials::MaterialType material_type)
  {
    switch (material_type)
    {
      case Core::Materials::mes_isoratedep:
      case Core::Materials::mes_fsls:
      case Core::Materials::mes_generalizedmaxwell:
      case Core::Materials::mes_quasilineargeneralizedmaxwell:
      case Core::Materials::mes_viscobranch:
      case Core::Materials::mes_coupmyocard:
        return true;
      default:
        return false;
    }
  }

}  // namespace

namespace ViscoElast = Mat::ViscoElast;

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PAR::ViscoElastHyper::ViscoElastHyper(const Core::Mat::PAR::Parameter::Data& matdata)
    : ViscoElastHyper(matdata, parse_summand_split(matdata))
{
}


Mat::PAR::ViscoElastHyper::SummandSplit Mat::PAR::ViscoElastHyper::parse_summand_split(
    const Core::Mat::PAR::Parameter::Data& matdata)
{
  const auto& numelast = matdata.parameters.get<std::optional<int>>("NUMELAST");
  const auto& elast_matids =
      matdata.parameters.get<std::optional<std::vector<int>>>("ELAST_MATIDS");
  const auto& numvisco = matdata.parameters.get<std::optional<int>>("NUMVISCO");
  const auto& visco_matids =
      matdata.parameters.get<std::optional<std::vector<int>>>("VISCO_MATIDS");

  const bool has_split_field = numelast.has_value() || elast_matids.has_value() ||
                               numvisco.has_value() || visco_matids.has_value();
  const bool has_complete_split = numelast.has_value() && elast_matids.has_value() &&
                                  numvisco.has_value() && visco_matids.has_value();

  FOUR_C_ASSERT_ALWAYS(!has_split_field || has_complete_split,
      "Split MAT_ViscoElastHyper input requires NUMELAST, ELAST_MATIDS, NUMVISCO and "
      "VISCO_MATIDS together (MAT {}).",
      matdata.id);

  if (has_complete_split)
  {
    FOUR_C_ASSERT_ALWAYS(*numelast >= 0,
        "Invalid MAT_ViscoElastHyper setup (MAT {}): NUMELAST={} must be non-negative.", matdata.id,
        *numelast);
    FOUR_C_ASSERT_ALWAYS(*numvisco >= 0,
        "Invalid MAT_ViscoElastHyper setup (MAT {}): NUMVISCO={} must be non-negative.", matdata.id,
        *numvisco);

    SummandSplit split;
    split.numelast = *numelast;
    split.elast_matids = *elast_matids;
    split.numvisco = *numvisco;
    split.visco_matids = *visco_matids;
    split.uses_legacy_matids = false;
    return split;
  }

  FOUR_C_ASSERT_ALWAYS(Global::Problem::instance()->materials() != nullptr,
      "Cannot derive MAT_ViscoElastHyper automatic summand split for MAT {} because no global "
      "material bundle is available.",
      matdata.id);

  FOUR_C_ASSERT_ALWAYS(Global::Problem::instance()->materials()->num() != 0,
      "Cannot derive MAT_ViscoElastHyper automatic summand split for MAT {} because the global "
      "material bundle is empty.",
      matdata.id);

  SummandSplit split;
  split.uses_legacy_matids = true;

  const std::vector<int>& matids = matdata.parameters.get<std::vector<int>>("MATIDS");
  const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
  for (const int summand_mat_id : matids)
  {
    auto* sumpar =
        Global::Problem::instance(probinst)->materials()->parameter_by_id(summand_mat_id);
    if (is_visco_material_type_for_split_input(sumpar->type()))
      split.visco_matids.push_back(summand_mat_id);
    else
      split.elast_matids.push_back(summand_mat_id);
  }

  split.numelast = static_cast<int>(split.elast_matids.size());
  split.numvisco = static_cast<int>(split.visco_matids.size());
  return split;
}


Mat::PAR::ViscoElastHyper::ViscoElastHyper(
    const Core::Mat::PAR::Parameter::Data& matdata, const SummandSplit& summand_split)
    : Mat::PAR::ElastHyper(matdata),
      numelast_(summand_split.numelast),
      elast_matids_(summand_split.elast_matids),
      numvisco_(summand_split.numvisco),
      visco_matids_(summand_split.visco_matids),
      uses_legacy_matids_(summand_split.uses_legacy_matids)
{
  FOUR_C_ASSERT_ALWAYS(numelast_ == static_cast<int>(elast_matids_.size()),
      "Invalid MAT_ViscoElastHyper setup (MAT {}): NUMELAST={} but ELAST_MATIDS has size {}.", id(),
      numelast_, elast_matids_.size());

  FOUR_C_ASSERT_ALWAYS(numvisco_ == static_cast<int>(visco_matids_.size()),
      "Invalid MAT_ViscoElastHyper setup (MAT {}): NUMVISCO={} but VISCO_MATIDS has size {}.", id(),
      numvisco_, visco_matids_.size());

  FOUR_C_ASSERT_ALWAYS(!uses_legacy_matids_ ||
                           static_cast<int>(elast_matids_.size() + visco_matids_.size()) == nummat_,
      "Invalid MAT_ViscoElastHyper automatic partitioning (MAT {}): NUMMAT={} but partitioned "
      "ELAST_MATIDS({}) + VISCO_MATIDS({}) do not match.",
      id(), nummat_, elast_matids_.size(), visco_matids_.size());

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
  visco_generalized_maxwell_ = false;
  visco_quasi_linear_generalized_maxwell_ = false;
  visco_fsls_ = false;

  state_.clear();
  contributions_.clear();
  rebuild_active_model_sequence();
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::ViscoElastHyper::ViscoElastHyper(Mat::PAR::ViscoElastHyper* params)
    : Mat::ElastHyper(),
      isovisco_(false),
      visco_generalized_maxwell_(false),
      visco_quasi_linear_generalized_maxwell_(false),
      visco_fsls_(false)
{
  params_ = params;
  rebuild_summand_sets();
  contributions_.clear();
  rebuild_active_model_sequence();
}


bool Mat::ViscoElastHyper::is_visco_material_type(const Core::Materials::MaterialType material_type)
{
  return is_visco_material_type_for_split_input(material_type);
}


const Mat::PAR::ViscoElastHyper* Mat::ViscoElastHyper::visco_params() const
{
  return static_cast<const Mat::PAR::ViscoElastHyper*>(params_);
}


int Mat::ViscoElastHyper::visco_mat_id(const unsigned index) const
{
  const Mat::PAR::ViscoElastHyper* visco_par = visco_params();
  if (visco_par == nullptr) return -1;
  FOUR_C_ASSERT_ALWAYS(index < visco_par->visco_matids_.size(),
      "Invalid visco summand index {} for MAT_ViscoElastHyper (MAT {}).", index, visco_par->id());
  return visco_par->visco_matids_.at(index);
}


void Mat::ViscoElastHyper::rebuild_summand_sets()
{
  elast_potsum_.clear();
  visco_potsum_.clear();

  const Mat::PAR::ViscoElastHyper* visco_par = visco_params();
  if (visco_par == nullptr) return;

  for (const int summand_mat_id : visco_par->elast_matids_)
  {
    auto sum = Mat::Elastic::Summand::factory(summand_mat_id);
    FOUR_C_ASSERT_ALWAYS(
        sum != nullptr, "Failed to allocate elastic summand MAT {}.", summand_mat_id);
    elast_potsum_.push_back(sum);
    sum->register_anisotropy_extensions(anisotropy_);
  }

  for (const int summand_mat_id : visco_par->visco_matids_)
  {
    auto sum = ViscoElast::Summand::factory(summand_mat_id);
    FOUR_C_ASSERT_ALWAYS(
        sum != nullptr, "Failed to allocate visco summand MAT {}.", summand_mat_id);
    visco_potsum_.push_back(sum);
  }
}


void Mat::ViscoElastHyper::rebuild_effective_summand_properties()
{
  effective_summand_properties_.clear();
  effective_summand_properties_.merge(elast_summand_properties_);

  SummandProperties visco_formulation_properties;
  visco_formulation_properties.clear();

  for (const auto& p : visco_potsum_)
  {
    p->specify_formulation(visco_formulation_properties.isoprinc,
        visco_formulation_properties.isomod, visco_formulation_properties.anisoprinc,
        visco_formulation_properties.anisomod, visco_formulation_properties.viscoGeneral);
  }

  effective_summand_properties_.merge(visco_formulation_properties);

  // Keep stretch coefficient flags tied to elastic constitutive summands.
  effective_summand_properties_.coeffStretchesPrinc = elast_summand_properties_.coeffStretchesPrinc;
  effective_summand_properties_.coeffStretchesMod = elast_summand_properties_.coeffStretchesMod;
}


Mat::ViscoElastHyper::EvaluateWorkspace::EvaluateWorkspace()
    : glstrain_mat(Core::LinAlg::Initialization::zero),
      c_stress(Core::LinAlg::Initialization::zero),
      i_c_stress(Core::LinAlg::Initialization::zero),
      c_strain(Core::LinAlg::Initialization::zero),
      mod_c_strain(Core::LinAlg::Initialization::zero),
      id2(Core::LinAlg::Initialization::zero),
      id4(Core::LinAlg::Initialization::zero),
      id4sharp(Core::LinAlg::Initialization::zero),
      prinv(Core::LinAlg::Initialization::zero),
      modinv(Core::LinAlg::Initialization::zero),
      rateinv(Core::LinAlg::Initialization::zero),
      modrateinv(Core::LinAlg::Initialization::zero),
      dPI(Core::LinAlg::Initialization::zero),
      ddPII(Core::LinAlg::Initialization::zero),
      scgrate(Core::LinAlg::Initialization::zero),
      modrcgrate(Core::LinAlg::Initialization::zero),
      mu(Core::LinAlg::Initialization::zero),
      modmu(Core::LinAlg::Initialization::zero),
      xi(Core::LinAlg::Initialization::zero),
      modxi(Core::LinAlg::Initialization::zero)
{
}


bool Mat::ViscoElastHyper::is_model_flag_enabled(const ViscoModelKind model_kind) const
{
  switch (model_kind)
  {
    case ViscoModelKind::iso_rate:
      return isovisco_;
    case ViscoModelKind::generalized_maxwell:
      return visco_generalized_maxwell_;
    case ViscoModelKind::quasi_linear_generalized_maxwell:
      return visco_quasi_linear_generalized_maxwell_;
    case ViscoModelKind::fsls:
      return visco_fsls_;
  }

  FOUR_C_THROW("Unsupported visco model kind.");
}


bool Mat::ViscoElastHyper::is_model_active(const ViscoModelKind model_kind) const
{
  for (const ViscoModelKind active_model_kind : active_model_sequence_)
  {
    if (active_model_kind == model_kind) return true;
  }

  return false;
}


void Mat::ViscoElastHyper::rebuild_active_model_sequence()
{
  active_model_sequence_.clear();
  for (const ViscoModelKind model_kind : visco_model_registry())
  {
    if (is_model_flag_enabled(model_kind)) active_model_sequence_.push_back(model_kind);
  }
}


void Mat::ViscoElastHyper::rebuild_contributions()
{
  contributions_.clear();

  for (const ViscoModelKind model_kind : active_model_sequence_)
  {
    switch (model_kind)
    {
      case ViscoModelKind::iso_rate:
        contributions_.push_back(std::make_shared<ViscoElast::IsoRateContribution>());
        break;
      case ViscoModelKind::generalized_maxwell:
        contributions_.push_back(std::make_shared<ViscoElast::GeneralizedMaxwellContribution>());
        break;
      case ViscoModelKind::quasi_linear_generalized_maxwell:
        contributions_.push_back(
            std::make_shared<ViscoElast::QuasiLinearGeneralizedMaxwellContribution>());
        break;
      case ViscoModelKind::fsls:
        contributions_.push_back(std::make_shared<ViscoElast::FslsContribution>());
        break;
    }
  }
}


void Mat::ViscoElastHyper::setup_contributions(const int gp, const int eleGID)
{
  const Mat::PAR::ViscoElastHyper* visco_par = visco_params();
  if (visco_par == nullptr)
  {
    FOUR_C_ASSERT_ALWAYS(contributions_.empty(),
        "Cannot setup visco contributions without MAT_ViscoElastHyper parameters (GP {}, ELE {}).",
        gp, eleGID);
    return;
  }

  const int visco_mat_id = params_ != nullptr ? params_->id() : -1;
  const ViscoElast::ContributionPointContext point{
      .visco_mat_id = visco_mat_id, .gp = gp, .ele_gid = eleGID};
  const ViscoElast::ContributionSetupContext base_context{.point = point,
      .visco_summands = visco_potsum_,
      .visco_summand_mat_ids = visco_par->visco_matids_,
      .active_models = active_models()};

  for (auto& contribution : contributions_)
  {
    if (contribution == nullptr) continue;

    if (contribution->kind() == ViscoModelKind::quasi_linear_generalized_maxwell)
    {
      auto* quasi_linear =
          dynamic_cast<ViscoElast::QuasiLinearGeneralizedMaxwellContribution*>(contribution.get());
      FOUR_C_ASSERT_ALWAYS(quasi_linear != nullptr,
          "Visco contribution type mismatch while setting up MAT_ViscoElastHyper (MAT {}, GP {}, "
          "ELE {}): expected quasi-linear generalized Maxwell contribution.",
          visco_mat_id, gp, eleGID);

      quasi_linear->setup(
          ViscoElast::QuasiLinearGeneralizedMaxwellSetupContext{.base = base_context,
              .elastic_summands = elast_potsum_,
              .elastic_summand_properties = elast_summand_properties_});
      continue;
    }

    contribution->setup(base_context);
  }
}


Mat::ViscoElastState::ActiveModels Mat::ViscoElastHyper::active_models_from_flags() const
{
  return Mat::ViscoElastState::ActiveModels{.iso_rate = isovisco_,
      .generalized_maxwell = visco_generalized_maxwell_,
      .quasi_linear_generalized_maxwell = visco_quasi_linear_generalized_maxwell_,
      .fsls = visco_fsls_};
}


Mat::ViscoElastState::ActiveModels Mat::ViscoElastHyper::active_models_from_sequence() const
{
  Mat::ViscoElastState::ActiveModels models;
  for (const ViscoModelKind model_kind : active_model_sequence_)
  {
    switch (model_kind)
    {
      case ViscoModelKind::iso_rate:
        models.iso_rate = true;
        break;
      case ViscoModelKind::generalized_maxwell:
        models.generalized_maxwell = true;
        break;
      case ViscoModelKind::quasi_linear_generalized_maxwell:
        models.quasi_linear_generalized_maxwell = true;
        break;
      case ViscoModelKind::fsls:
        models.fsls = true;
        break;
    }
  }

  return models;
}


void Mat::ViscoElastHyper::ensure_model_activation_consistency(const char* context) const
{
  const Mat::ViscoElastState::ActiveModels models_from_flags = active_models_from_flags();
  const Mat::ViscoElastState::ActiveModels models_from_sequence = active_models_from_sequence();

  FOUR_C_ASSERT_ALWAYS(
      models_from_flags.iso_rate == models_from_sequence.iso_rate &&
          models_from_flags.generalized_maxwell == models_from_sequence.generalized_maxwell &&
          models_from_flags.quasi_linear_generalized_maxwell ==
              models_from_sequence.quasi_linear_generalized_maxwell &&
          models_from_flags.fsls == models_from_sequence.fsls,
      "Inconsistent visco model activation while {} in MAT_ViscoElastHyper (MAT {}): "
      "flags=(iso_rate={}, generalized_maxwell={}, quasi_linear_generalized_maxwell={}, "
      "fsls={}), sequence=(iso_rate={}, generalized_maxwell={}, "
      "quasi_linear_generalized_maxwell={}, fsls={}).",
      context, params_ != nullptr ? params_->id() : -1, models_from_flags.iso_rate,
      models_from_flags.generalized_maxwell, models_from_flags.quasi_linear_generalized_maxwell,
      models_from_flags.fsls, models_from_sequence.iso_rate,
      models_from_sequence.generalized_maxwell,
      models_from_sequence.quasi_linear_generalized_maxwell, models_from_sequence.fsls);
}


Mat::ViscoElastState::ActiveModels Mat::ViscoElastHyper::active_models() const
{
  ensure_model_activation_consistency("deriving active model descriptor");
  return active_models_from_sequence();
}


std::size_t Mat::ViscoElastHyper::history_entry_count_for_setup(
    const ViscoModelKind model_kind) const
{
  if (!is_model_active(model_kind)) return 0;

  for (const auto& contribution : contributions_)
  {
    if (contribution != nullptr && contribution->kind() == model_kind)
      return contribution->history_entry_count_for_setup();
  }

  FOUR_C_THROW(
      "Missing visco contribution while reading setup history size in MAT_ViscoElastHyper (MAT "
      "{}).",
      params_ != nullptr ? params_->id() : -1);
  return 0;
}


unsigned int Mat::ViscoElastHyper::history_capacity_for_update(
    const ViscoModelKind model_kind) const
{
  if (!is_model_active(model_kind)) return 0;

  for (const auto& contribution : contributions_)
  {
    if (contribution != nullptr && contribution->kind() == model_kind)
      return contribution->history_capacity_for_update();
  }

  FOUR_C_THROW(
      "Missing visco contribution while reading update history capacity in MAT_ViscoElastHyper "
      "(MAT {}).",
      params_ != nullptr ? params_->id() : -1);
  return 0;
}


double Mat::ViscoElastHyper::read_visco_time_step_size(
    const EvaluationContext<3>& context, const int gp, const int eleGID) const
{
  FOUR_C_ASSERT_ALWAYS(context.time_step_size != nullptr,
      "Missing EvaluationContext::time_step_size in MAT_ViscoElastHyper (MAT {}, GP {}, ELE "
      "{}).",
      params_ != nullptr ? params_->id() : -1, gp, eleGID);

  const double dt = *context.time_step_size;
  FOUR_C_ASSERT_ALWAYS(dt > 0.0,
      "Invalid time step size dt={} in MAT_ViscoElastHyper (MAT {}, GP {}, ELE {}) for active "
      "viscoelastic response. Expected dt > 0.",
      dt, params_ != nullptr ? params_->id() : -1, gp, eleGID);

  return dt;
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
  elast_summand_properties_.pack(data);
  const Mat::ViscoElastState::ActiveModels models = active_models();
  add_to_pack(data, models.iso_rate);
  add_to_pack(data, models.generalized_maxwell);
  add_to_pack(data, models.quasi_linear_generalized_maxwell);
  add_to_pack(data, models.fsls);

  anisotropy_.pack_anisotropy(data);

  Core::Communication::PotentiallyUnusedBufferScope summand_scope{data};
  if (params_ != nullptr)  // summands are not accessible in postprocessing mode
  {
    for (auto& p : elast_potsum_) p->pack_summand(data);
    for (auto& p : visco_potsum_) p->pack_summand(data);

    state_.serialize_state(data, active_models());
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ViscoElastHyper::unpack(Core::Communication::UnpackBuffer& buffer)
{
  // make sure we have a pristine material
  params_ = nullptr;
  elast_potsum_.clear();
  visco_potsum_.clear();
  elast_summand_properties_.clear();
  effective_summand_properties_.clear();
  isovisco_ = false;
  visco_generalized_maxwell_ = false;
  visco_quasi_linear_generalized_maxwell_ = false;
  visco_fsls_ = false;
  state_.clear();
  active_model_sequence_.clear();
  contributions_.clear();

  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // Integration-boundary access: unpack resolves parameter material from the global
  // material bundle and then hands explicit data to visco internals.
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
      FOUR_C_ASSERT_ALWAYS(mat->type() == material_type(),
          "Type of parameter material {} does not fit to calling type {}", mat->type(),
          material_type());
      params_ = static_cast<Mat::PAR::ViscoElastHyper*>(mat);
    }
  }

  elast_summand_properties_.unpack(buffer);
  extract_from_pack(buffer, isovisco_);
  extract_from_pack(buffer, visco_generalized_maxwell_);
  extract_from_pack(buffer, visco_quasi_linear_generalized_maxwell_);
  extract_from_pack(buffer, visco_fsls_);
  rebuild_active_model_sequence();
  rebuild_contributions();
  ensure_model_activation_consistency("unpack model activation");

  anisotropy_.unpack_anisotropy(buffer);

  Core::Communication::PotentiallyUnusedBufferScope summand_scope{buffer};
  if (params_ != nullptr)  // summands are not accessible in postprocessing mode
  {
    rebuild_summand_sets();

    for (auto& p : elast_potsum_) p->unpack_summand(buffer);
    for (auto& p : visco_potsum_) p->unpack_summand(buffer);

    rebuild_effective_summand_properties();
    setup_contributions(-1, -1);

    state_.deserialize_state(buffer, active_models());
  }
  else
  {
    effective_summand_properties_.update(elast_summand_properties_);
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

  for (auto& p : elast_potsum_)
  {
    p->setup(numgp, fibers, coord_system);
  }
  for (auto& p : visco_potsum_) p->setup(numgp, fibers, coord_system);

  // find out which formulations are used
  isovisco_ = false;
  visco_generalized_maxwell_ = false;
  visco_quasi_linear_generalized_maxwell_ = false;
  visco_fsls_ = false;

  elast_summand_properties_.clear();
  elast_hyper_properties(elast_potsum_, elast_summand_properties_);
  rebuild_effective_summand_properties();

  for (auto& p : visco_potsum_)
    p->specify_visco_formulation(isovisco_, visco_generalized_maxwell_,
        visco_quasi_linear_generalized_maxwell_, visco_fsls_);

  rebuild_active_model_sequence();
  rebuild_contributions();
  ensure_model_activation_consistency("pre-loop setup orchestration");

  setup_contributions(-1, -1);

  const Mat::ViscoElastState::ActiveModels models = active_models();
  const std::size_t generalized_maxwell_numbranch =
      history_entry_count_for_setup(ViscoModelKind::generalized_maxwell);
  const std::size_t quasi_linear_generalized_maxwell_numbranch =
      history_entry_count_for_setup(ViscoModelKind::quasi_linear_generalized_maxwell);

  state_.initialize_from_setup(
      numgp, models, generalized_maxwell_numbranch, quasi_linear_generalized_maxwell_numbranch);

  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::ViscoElastHyper::update()
{
  for (auto& p : elast_potsum_) p->update();
  for (auto& p : visco_potsum_) p->update();

  const int visco_mat_id = params_ != nullptr ? params_->id() : -1;
  const ViscoElast::ContributionPointContext point{
      .visco_mat_id = visco_mat_id, .gp = -1, .ele_gid = -1};

  for (auto& contribution : contributions_)
  {
    if (contribution == nullptr) continue;
    contribution->update(ViscoElast::ContributionUpdateContext{.point = point, .state = state_});
  }

  ensure_model_activation_consistency("pre-loop update orchestration");

  const unsigned int max_hist = history_capacity_for_update(ViscoModelKind::fsls);

  state_.advance_time_step(active_models(), max_hist, visco_mat_id);

  return;
}

void Mat::ViscoElastHyper::prepare_evaluate_kinematics(
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const EvaluationContext<3>& context, EvaluateWorkspace& workspace, const int gp,
    const int eleGID) const
{
  workspace.glstrain_mat = Core::LinAlg::make_strain_like_voigt_matrix(glstrain);

  evaluate_right_cauchy_green_strain_like_voigt(glstrain, workspace.c);
  workspace.i_c = Core::LinAlg::inv(workspace.c);

  workspace.c_stress = Core::LinAlg::make_stress_like_voigt_view(workspace.c);
  workspace.i_c_stress = Core::LinAlg::make_stress_like_voigt_view(workspace.i_c);

  Core::LinAlg::Voigt::Stresses::to_strain_like(workspace.c_stress, workspace.c_strain);
  Core::LinAlg::Voigt::Stresses::invariants_principal(workspace.prinv, workspace.c_stress);

  workspace.dt = 0.0;
  if (!active_model_sequence_.empty())
    workspace.dt = read_visco_time_step_size(context, gp, eleGID);

  Core::LinAlg::Voigt::identity_matrix(workspace.id2);

  using VoigtNotation = Core::LinAlg::Voigt::NotationType;
  Core::LinAlg::Voigt::fourth_order_identity_matrix<VoigtNotation::stress, VoigtNotation::stress>(
      workspace.id4sharp);
  Core::LinAlg::Voigt::fourth_order_identity_matrix<VoigtNotation::stress, VoigtNotation::strain>(
      workspace.id4);

  elast_hyper_evaluate_invariant_derivatives(workspace.prinv, workspace.dPI, workspace.ddPII,
      elast_potsum_, elast_summand_properties_, gp, eleGID);
}


void Mat::ViscoElastHyper::initialize_elastic_response(const EvaluateWorkspace& workspace,
    Core::LinAlg::Matrix<6, 1>& stress_view, Core::LinAlg::Matrix<6, 6>& cmat_view,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat) const
{
  // blank resulting quantities
  // ... even if it is an implicit law that cmat is zero upon input
  stress_view.clear();
  cmat_view.clear();

  // add isotropic part
  elast_hyper_add_isotropic_stress_cmat(
      stress, cmat, workspace.c, workspace.i_c, workspace.prinv, workspace.dPI, workspace.ddPII);
}

void Mat::ViscoElastHyper::add_visco_contributions_in_sequence(const Teuchos::ParameterList& params,
    EvaluateWorkspace& workspace, Core::LinAlg::Matrix<6, 1>& stress_view,
    Core::LinAlg::Matrix<6, 6>& cmat_view, const int gp, const int eleGID)
{
  const int visco_mat_id = params_ != nullptr ? params_->id() : -1;
  const ViscoElast::ContributionPointContext point{
      .visco_mat_id = visco_mat_id, .gp = gp, .ele_gid = eleGID};

  auto require_contribution = [&]<typename ContributionType>(const ViscoModelKind kind,
                                  const char* model_name) -> ContributionType&
  {
    for (const auto& contribution : contributions_)
    {
      if (contribution == nullptr || contribution->kind() != kind) continue;

      auto* typed_contribution = dynamic_cast<ContributionType*>(contribution.get());
      FOUR_C_ASSERT_ALWAYS(typed_contribution != nullptr,
          "Visco contribution type mismatch while evaluating MAT_ViscoElastHyper (MAT {}, GP {}, "
          "ELE {}): expected {} contribution.",
          visco_mat_id, gp, eleGID, model_name);

      return *typed_contribution;
    }

    FOUR_C_THROW(
        "Missing {} contribution while evaluating MAT_ViscoElastHyper (MAT {}, GP {}, ELE {}).",
        model_name, visco_mat_id, gp, eleGID);
  };

  auto make_evaluate_base = [&]() -> ViscoElast::ContributionEvaluateBase
  {
    return ViscoElast::ContributionEvaluateBase{.point = point,
        .dt = workspace.dt,
        .state = state_,
        .stress = stress_view,
        .cmat = cmat_view};
  };

  for (const ViscoModelKind model_kind : active_model_sequence_)
  {
    switch (model_kind)
    {
      case ViscoModelKind::iso_rate:
      {
        auto& contribution =
            require_contribution.template operator()<ViscoElast::IsoRateContribution>(
                ViscoModelKind::iso_rate, "iso-rate");
        contribution.evaluate(ViscoElast::IsoRateEvaluateContext{.base = make_evaluate_base(),
            .params = params,
            .visco_summands = visco_potsum_,
            .effective_properties = effective_summand_properties_,
            .c_strain = workspace.c_strain,
            .c_stress = workspace.c_stress,
            .i_c_stress = workspace.i_c_stress,
            .prinv = workspace.prinv,
            .modinv = workspace.modinv,
            .rateinv = workspace.rateinv,
            .modrateinv = workspace.modrateinv,
            .mod_c_strain = workspace.mod_c_strain,
            .scgrate = workspace.scgrate,
            .modrcgrate = workspace.modrcgrate,
            .mu = workspace.mu,
            .modmu = workspace.modmu,
            .xi = workspace.xi,
            .modxi = workspace.modxi,
            .id2 = workspace.id2,
            .id4 = workspace.id4,
            .id4sharp = workspace.id4sharp});
        break;
      }
      case ViscoModelKind::generalized_maxwell:
      {
        auto& contribution =
            require_contribution.template operator()<ViscoElast::GeneralizedMaxwellContribution>(
                ViscoModelKind::generalized_maxwell, "generalized Maxwell");
        contribution.evaluate(ViscoElast::GeneralizedMaxwellEvaluateContext{
            .base = make_evaluate_base(), .glstrain_mat = workspace.glstrain_mat});
        break;
      }
      case ViscoModelKind::quasi_linear_generalized_maxwell:
      {
        auto& contribution =
            require_contribution
                .template operator()<ViscoElast::QuasiLinearGeneralizedMaxwellContribution>(
                    ViscoModelKind::quasi_linear_generalized_maxwell,
                    "quasi-linear generalized Maxwell");
        contribution.evaluate(ViscoElast::QuasiLinearGeneralizedMaxwellEvaluateContext{
            .base = make_evaluate_base(), .glstrain_mat = workspace.glstrain_mat});
        break;
      }
      case ViscoModelKind::fsls:
      {
        auto& contribution = require_contribution.template operator()<ViscoElast::FslsContribution>(
            ViscoModelKind::fsls, "FSLS");
        contribution.evaluate(ViscoElast::FslsEvaluateContext{.base = make_evaluate_base()});
        break;
      }
    }
  }
}


void Mat::ViscoElastHyper::add_post_elastic_composition_hooks(const Teuchos::ParameterList& params,
    const EvaluationContext<3>& context, const EvaluateWorkspace& workspace,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, const int gp, const int eleGID)
{
  /*----------------------------------------------------------------------*/
  // coefficients in principal stretches
  if (elast_summand_properties_.coeffStretchesPrinc || elast_summand_properties_.coeffStretchesMod)
  {
    elast_hyper_add_response_stretches(
        cmat, stress, workspace.c, elast_potsum_, elast_summand_properties_, gp, eleGID);
  }

  /*----------------------------------------------------------------------*/
  // Do all the anisotropic stuff!
  if (elast_summand_properties_.anisoprinc)
  {
    elast_hyper_add_anisotropic_princ(stress, cmat, workspace.c, params, gp, eleGID, elast_potsum_);
  }

  if (elast_summand_properties_.anisomod)
  {
    elast_hyper_add_anisotropic_mod(stress, cmat, workspace.c, workspace.i_c, workspace.prinv, gp,
        eleGID, context, elast_potsum_);
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ViscoElastHyper::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  ensure_model_activation_consistency("pre-loop evaluate orchestration");
  Core::LinAlg::Matrix<6, 1> stress_view = Core::LinAlg::make_stress_like_voigt_view(stress);
  Core::LinAlg::Matrix<6, 6> cmat_view = Core::LinAlg::make_stress_like_voigt_view(cmat);

  EvaluateWorkspace workspace;

  prepare_evaluate_kinematics(glstrain, context, workspace, gp, eleGID);
  initialize_elastic_response(workspace, stress_view, cmat_view, stress, cmat);
  add_visco_contributions_in_sequence(params, workspace, stress_view, cmat_view, gp, eleGID);
  add_post_elastic_composition_hooks(params, context, workspace, stress, cmat, gp, eleGID);
}
FOUR_C_NAMESPACE_CLOSE
