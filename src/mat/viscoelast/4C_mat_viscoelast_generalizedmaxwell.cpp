// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_viscoelast_generalizedmaxwell.hpp"

#include "4C_global_data.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_utils_densematrix_inverse.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_material_parameter_base.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <cmath>
#include <utility>

FOUR_C_NAMESPACE_OPEN

namespace Mat::ViscoElast
{

  PAR::GeneralizedMaxwell::GeneralizedMaxwell(const Core::Mat::PAR::Parameter::Data& matdata)
      : Parameter(matdata),
        numbranch_(matdata.parameters.get<int>("NUMBRANCH")),
        matids_(matdata.parameters.get<std::vector<int>>("MATIDS")),
        solve_(matdata.parameters.get<std::string>("SOLVE"))
  {
    FOUR_C_ASSERT_ALWAYS(numbranch_ > 0,
        "Invalid NUMBRANCH={} in VISCO_GeneralizedMaxwell (MAT {}). NUMBRANCH has to be "
        "positive.",
        numbranch_, matdata.id);

    FOUR_C_ASSERT_ALWAYS(static_cast<int>(matids_.size()) == numbranch_,
        "Invalid VISCO_GeneralizedMaxwell branch declaration in MAT {}: NUMBRANCH={} but "
        "MATIDS has size {}.",
        matdata.id, numbranch_, matids_.size());

    for (int branch_index = 0; branch_index < numbranch_; ++branch_index)
    {
      const int branch_mat = matids_.at(branch_index);
      FOUR_C_ASSERT_ALWAYS(branch_mat > 0,
          "Invalid MATIDS[{}]={} in VISCO_GeneralizedMaxwell (MAT {}). Branch material ids "
          "have to be positive.",
          branch_index, branch_mat, matdata.id);
    }

    FOUR_C_ASSERT_ALWAYS(solve_ == "OneStepTheta" || solve_ == "ExponentialTimeDiscretization",
        "Invalid input for SOLVE='{}' in VISCO_GeneralizedMaxwell (MAT {}). Use "
        "OneStepTheta or ExponentialTimeDiscretization.",
        solve_, matdata.id);
  }

  GeneralizedMaxwell::GeneralizedMaxwell(PAR::GeneralizedMaxwell* params)
      : params_(params), branchespotsum_(0), branchtau_(0), internalpotsum_(0)
  {
    FOUR_C_ASSERT_ALWAYS(params_->numbranch_ > 0,
        "Invalid VISCO_GeneralizedMaxwell setup for MAT {}: NUMBRANCH={} is not positive.",
        params_->id(), params_->numbranch_);

    FOUR_C_ASSERT_ALWAYS(static_cast<int>(params_->matids_.size()) == params_->numbranch_,
        "Invalid VISCO_GeneralizedMaxwell setup for MAT {}: NUMBRANCH={} but MATIDS has size "
        "{}.",
        params_->id(), params_->numbranch_, params_->matids_.size());

    // Integration-boundary access: branch material validation currently depends on the global
    // material bundle and remains outside constitutive evaluate/update hot paths.
    FOUR_C_ASSERT_ALWAYS(Global::Problem::instance()->materials() != nullptr,
        "Cannot validate VISCO_GeneralizedMaxwell branches for MAT {} because no global material "
        "bundle is available.",
        params_->id());

    FOUR_C_ASSERT_ALWAYS(Global::Problem::instance()->materials()->num() != 0,
        "Cannot validate VISCO_GeneralizedMaxwell branches for MAT {} because the global "
        "material bundle is empty.",
        params_->id());

    const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();

    // loop over materials of GeneralizedMaxwell (branches)
    std::vector<int>::const_iterator m;
    int branch_index = 0;
    for (m = params_->matids_.begin(); m != params_->matids_.end(); ++m)
    {
      // make sure the summands of the current branch is empty
      internalpotsum_.clear();

      // get parameters of each branch
      const int matid = *m;

      auto* branch_parameter =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      FOUR_C_ASSERT_ALWAYS(branch_parameter->type() == Core::Materials::mes_viscobranch,
          "Invalid branch declaration in VISCO_GeneralizedMaxwell (MAT {}): MATIDS[{}]={} "
          "refers to material type {}. Expected VISCO_GeneralizedMaxwellBranch.",
          params_->id(), branch_index, matid, branch_parameter->type());

      std::shared_ptr<Summand> visco_branch = Summand::factory(matid);
      std::shared_ptr<ViscoBranch> visco_branch_typed =
          std::dynamic_pointer_cast<ViscoBranch>(visco_branch);
      FOUR_C_ASSERT_ALWAYS(visco_branch_typed != nullptr,
          "Failed to create VISCO_GeneralizedMaxwellBranch summand from MAT {} referenced by "
          "VISCO_GeneralizedMaxwell (MAT {}, MATIDS[{}]).",
          matid, params_->id(), branch_index);

      double tau = -1.0;
      int branchmatid = -1;

      visco_branch_typed->read_material_parameters(tau, branchmatid);

      FOUR_C_ASSERT_ALWAYS(tau > 0.0,
          "Invalid TAU={} in VISCO_GeneralizedMaxwellBranch (MAT {}, referenced by "
          "VISCO_GeneralizedMaxwell MAT {}, MATIDS[{}]). TAU has to be positive.",
          tau, matid, params_->id(), branch_index);

      FOUR_C_ASSERT_ALWAYS(branchmatid > 0,
          "Invalid MATID={} in VISCO_GeneralizedMaxwellBranch (MAT {}, referenced by "
          "VISCO_GeneralizedMaxwell MAT {}, MATIDS[{}]). MATID has to be positive.",
          branchmatid, matid, params_->id(), branch_index);

      std::shared_ptr<Elastic::Summand> sum = Elastic::Summand::factory(branchmatid);
      FOUR_C_ASSERT_ALWAYS(sum != nullptr,
          "Failed to create branch elasticity summand for MATID {} in "
          "VISCO_GeneralizedMaxwellBranch (MAT {}, referenced by VISCO_GeneralizedMaxwell MAT "
          "{}, MATIDS[{}]).",
          branchmatid, matid, params_->id(), branch_index);

      // write summand in the vector of summands of each branch
      internalpotsum_.push_back(sum);

      // write into vector of summands of the GeneralizedMaxwell material
      branchespotsum_.push_back(internalpotsum_);
      branchtau_.push_back(tau);

      ++branch_index;

    }  // end for-loop over branches
  }

  void GeneralizedMaxwell::read_material_parameters(
      int& numbranch, const std::vector<int>*& matids, std::string& solve)
  {
    numbranch = params_->numbranch_;
    matids = &params_->matids_;
    solve = params_->solve_;
  }

  // Viscobranch
  PAR::ViscoBranch::ViscoBranch(const Core::Mat::PAR::Parameter::Data& matdata)
      : Parameter(matdata),
        tau_(matdata.parameters.get<double>("TAU")),
        matid_(matdata.parameters.get<int>("MATID"))
  {
    FOUR_C_ASSERT_ALWAYS(tau_ > 0.0,
        "Invalid TAU={} in VISCO_GeneralizedMaxwellBranch (MAT {}). TAU has to be positive.", tau_,
        matdata.id);

    FOUR_C_ASSERT_ALWAYS(matid_ > 0,
        "Invalid MATID={} in VISCO_GeneralizedMaxwellBranch (MAT {}). MATID has to be "
        "positive.",
        matid_, matdata.id);
  }

  ViscoBranch::ViscoBranch(PAR::ViscoBranch* params) : params_(params) {}

  void ViscoBranch::read_material_parameters(double& tau, int& matid)
  {
    tau = params_->tau_;
    matid = params_->matid_;
  }


  void GeneralizedMaxwellContribution::setup(const ContributionSetupContext& context)
  {
    build_metadata(context);
    build_runtime_context(context);
  }


  void GeneralizedMaxwellContribution::evaluate(const GeneralizedMaxwellEvaluateContext& context)
  {
    const auto& point = context.base.point;

    const Metadata& metadata =
        require_metadata("evaluating generalized Maxwell response", point.gp, point.ele_gid);

    double one_step_theta = 0.5;
    if (metadata.solve_kind == SolveKind::one_step_theta)
    {
      const RuntimeContext& runtime_context = require_runtime_context(
          "evaluating generalized Maxwell response", point.gp, point.ele_gid);
      one_step_theta = runtime_context.one_step_theta;
    }

    evaluate_branch_material_response(context, metadata, one_step_theta);
  }


  void GeneralizedMaxwellContribution::evaluate_branch_material_response(
      const GeneralizedMaxwellEvaluateContext& context, const Metadata& metadata,
      const double one_step_theta) const
  {
    const auto& point = context.base.point;
    const int numbranch = static_cast<int>(metadata.branches.size());

    Core::LinAlg::Matrix<6, 1> gl_stress(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Voigt::Strains::to_stress_like(context.glstrain_mat, gl_stress);

    const Core::LinAlg::SymmetricTensor<double, 3, 3> gl =
        Core::LinAlg::make_symmetric_tensor_from_stress_like_voigt_matrix(gl_stress);

    Core::LinAlg::SymmetricTensor<double, 3, 3> c;
    evaluate_right_cauchy_green_strain_like_voigt(gl, c);
    const Core::LinAlg::SymmetricTensor<double, 3, 3> i_c = Core::LinAlg::inv(c);

    Core::LinAlg::Matrix<6, 1> c_strain(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Voigt::Stresses::to_strain_like(
        Core::LinAlg::make_stress_like_voigt_view(c), c_strain);

    Core::LinAlg::Matrix<3, 1> prinv(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Voigt::Strains::invariants_principal(prinv, c_strain);

    const auto& previous_branch_elastic_stress =
        context.base.state.generalized_maxwell_prev_branch_elastic_stress(point.gp);
    const auto& previous_branch_stress =
        context.base.state.generalized_maxwell_prev_branch_stress(point.gp);

    const auto collect_branch_taus = [&]()
    {
      std::vector<double> branch_taus;
      branch_taus.reserve(numbranch);
      for (const auto& branch_metadata : metadata.branches)
        branch_taus.push_back(branch_metadata.tau);
      return branch_taus;
    };

    const auto make_kernel_input = [&]()
    {
      Kernels::GeneralizedMaxwellKernelInput kernel_input;
      kernel_input.visco_mat_id = point.visco_mat_id;
      kernel_input.gp = point.gp;
      kernel_input.ele_gid = point.ele_gid;
      kernel_input.dt = context.base.dt;
      kernel_input.one_step_theta = one_step_theta;
      kernel_input.previous_branch_elastic_stress = &previous_branch_elastic_stress;
      kernel_input.previous_branch_stress = &previous_branch_stress;
      switch (metadata.solve_kind)
      {
        case SolveKind::one_step_theta:
          kernel_input.solve_kind = Kernels::GeneralizedMaxwellSolveKind::one_step_theta;
          break;
        case SolveKind::exponential_time_discretization:
          kernel_input.solve_kind =
              Kernels::GeneralizedMaxwellSolveKind::exponential_time_discretization;
          break;
      }
      return kernel_input;
    };

    const std::vector<double> branch_taus = collect_branch_taus();
    const Kernels::GeneralizedMaxwellKernelInput kernel_input = make_kernel_input();

    Kernels::PointHistory current_branch_elastic_stress;
    Kernels::PointHistory current_branch_stress;

    const Kernels::BranchResponseEvaluator evaluate_branch_response =
        [&](const int branch_index, Kernels::StressVector& branch_elastic_stress,
            Kernels::TangentMatrix& branch_cmat)
    {
      const auto& branch_metadata = metadata.branches.at(branch_index);

      Core::LinAlg::Matrix<3, 1> dPI(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<6, 1> ddPII(Core::LinAlg::Initialization::zero);
      elast_hyper_evaluate_invariant_derivatives(prinv, dPI, ddPII, branch_metadata.summands,
          branch_metadata.properties, point.gp, point.ele_gid);

      branch_elastic_stress.clear();
      branch_cmat.clear();

      Core::LinAlg::SymmetricTensor<double, 3, 3> stressiso{};
      Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> cmatiso{};
      elast_hyper_add_isotropic_stress_cmat(stressiso, cmatiso, c, i_c, prinv, dPI, ddPII);
      branch_elastic_stress.update(1.0, Core::LinAlg::make_stress_like_voigt_view(stressiso), 1.0);
      branch_cmat.update(1.0, Core::LinAlg::make_stress_like_voigt_view(cmatiso), 1.0);
    };

    Core::LinAlg::Matrix<6, 1> q(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 6> cmatq(Core::LinAlg::Initialization::zero);
    Kernels::evaluate_generalized_maxwell_kernel(q, cmatq, current_branch_elastic_stress,
        current_branch_stress, branch_taus, kernel_input, evaluate_branch_response);

    context.base.state.set_generalized_maxwell_current_point(
        point.gp, current_branch_elastic_stress, current_branch_stress);
    context.base.stress.update(1.0, q, 1.0);
    context.base.cmat.update(1.0, cmatq, 1.0);
  }

  void GeneralizedMaxwellContribution::update(const ContributionUpdateContext& context)
  {
    (void)context;
  }


  std::size_t GeneralizedMaxwellContribution::history_entry_count_for_setup() const
  {
    return require_metadata("reading branch count", -1, -1).branches.size();
  }


  GeneralizedMaxwellContribution::SolveKind GeneralizedMaxwellContribution::parse_solve_kind(
      const std::string& solve, const int visco_mat_id, const int gp, const int ele_gid)
  {
    if (solve == "OneStepTheta") return SolveKind::one_step_theta;
    if (solve == "ExponentialTimeDiscretization") return SolveKind::exponential_time_discretization;

    FOUR_C_THROW(
        "Invalid SOLVE='{}' in VISCO_GeneralizedMaxwell for MAT_ViscoElastHyper (MAT {}, GP {}, "
        "ELE {}). Use OneStepTheta or ExponentialTimeDiscretization.",
        solve, visco_mat_id, gp, ele_gid);

    return SolveKind::exponential_time_discretization;
  }


  const GeneralizedMaxwellContribution::Metadata& GeneralizedMaxwellContribution::require_metadata(
      const char* context, const int gp, const int ele_gid) const
  {
    FOUR_C_ASSERT_ALWAYS(metadata_.has_value(),
        "Missing VISCO_GeneralizedMaxwell metadata cache while {} in MAT_ViscoElastHyper (GP {}, "
        "ELE {}). Run setup() before evaluation.",
        context, gp, ele_gid);

    FOUR_C_ASSERT_ALWAYS(!metadata_->branches.empty(),
        "Invalid VISCO_GeneralizedMaxwell metadata cache while {} in MAT_ViscoElastHyper (GP {}, "
        "ELE {}): no branch metadata available.",
        context, gp, ele_gid);

    return metadata_.value();
  }


  const GeneralizedMaxwellContribution::RuntimeContext&
  GeneralizedMaxwellContribution::require_runtime_context(
      const char* context, const int gp, const int ele_gid) const
  {
    FOUR_C_ASSERT_ALWAYS(runtime_context_.has_value(),
        "Missing generalized Maxwell runtime context while {} in MAT_ViscoElastHyper (GP {}, "
        "ELE {}). Run setup() before evaluation.",
        context, gp, ele_gid);

    return runtime_context_.value();
  }


  void GeneralizedMaxwellContribution::build_metadata(const ContributionSetupContext& context)
  {
    metadata_.reset();
    const auto& point = context.point;

    FOUR_C_ASSERT_ALWAYS(context.visco_summands.size() == context.visco_summand_mat_ids.size(),
        "Invalid generalized Maxwell setup context in MAT_ViscoElastHyper (MAT {}): visco "
        "summand count {} does not match MAT id count {}.",
        point.visco_mat_id, context.visco_summands.size(), context.visco_summand_mat_ids.size());

    int generalized_maxwell_model_count = 0;
    int generalized_maxwell_numbranch_value = -1;
    std::string generalized_maxwell_solve;
    const std::vector<int>* generalized_maxwell_matids = nullptr;

    std::shared_ptr<GeneralizedMaxwell> generalized_maxwell = nullptr;
    int generalized_maxwell_summand_mat_id = -1;

    for (std::size_t p = 0; p < context.visco_summands.size(); ++p)
    {
      auto current_generalized_maxwell =
          std::dynamic_pointer_cast<GeneralizedMaxwell>(context.visco_summands.at(p));
      if (current_generalized_maxwell != nullptr)
      {
        ++generalized_maxwell_model_count;
        generalized_maxwell = current_generalized_maxwell;
        generalized_maxwell_summand_mat_id = context.visco_summand_mat_ids.at(p);
        generalized_maxwell->read_material_parameters(generalized_maxwell_numbranch_value,
            generalized_maxwell_matids, generalized_maxwell_solve);
      }
    }

    FOUR_C_ASSERT_ALWAYS(generalized_maxwell_model_count == 1,
        "Invalid VISCO_GeneralizedMaxwell setup in MAT_ViscoElastHyper (MAT {}): expected "
        "exactly one VISCO_GeneralizedMaxwell summand but found {}.",
        point.visco_mat_id, generalized_maxwell_model_count);

    FOUR_C_ASSERT_ALWAYS(generalized_maxwell != nullptr,
        "Failed to resolve VISCO_GeneralizedMaxwell summand in MAT_ViscoElastHyper (MAT {}).",
        point.visco_mat_id);

    FOUR_C_ASSERT_ALWAYS(generalized_maxwell_numbranch_value > 0,
        "Invalid VISCO_GeneralizedMaxwell setup in MAT_ViscoElastHyper (MAT {}): NUMBRANCH={} "
        "is not positive.",
        point.visco_mat_id, generalized_maxwell_numbranch_value);

    FOUR_C_ASSERT_ALWAYS(generalized_maxwell_matids != nullptr,
        "Failed to read MATIDS for VISCO_GeneralizedMaxwell in MAT_ViscoElastHyper (MAT {}).",
        point.visco_mat_id);

    FOUR_C_ASSERT_ALWAYS(generalized_maxwell_matids->size() ==
                             static_cast<unsigned int>(generalized_maxwell_numbranch_value),
        "Invalid VISCO_GeneralizedMaxwell setup in MAT_ViscoElastHyper (MAT {}): NUMBRANCH={} "
        "but MATIDS has size {}.",
        point.visco_mat_id, generalized_maxwell_numbranch_value,
        generalized_maxwell_matids->size());

    const auto& branchespotsum = generalized_maxwell->get_branchespotsum();
    const auto& branchtau = generalized_maxwell->get_branchtaus();

    FOUR_C_ASSERT_ALWAYS(
        branchespotsum.size() == static_cast<unsigned int>(generalized_maxwell_numbranch_value) &&
            branchtau.size() == static_cast<unsigned int>(generalized_maxwell_numbranch_value),
        "Failed to initialize VISCO_GeneralizedMaxwell branches in MAT_ViscoElastHyper (MAT {}, "
        "GP {}, ELE {}). Expected {} branches, got {} branch definitions and {} branch relaxation "
        "times.",
        point.visco_mat_id, point.gp, point.ele_gid, generalized_maxwell_numbranch_value,
        branchespotsum.size(), branchtau.size());

    if (context.active_models.iso_rate)
      FOUR_C_THROW(
          "Unsupported branch formulation in VISCO_GeneralizedMaxwell for MAT_ViscoElastHyper "
          "(MAT {}, GP {}, ELE {}): isovisco branch response is not implemented.",
          point.visco_mat_id, point.gp, point.ele_gid);

    Metadata metadata;
    metadata.summand_mat_id = generalized_maxwell_summand_mat_id;
    metadata.solve_kind =
        parse_solve_kind(generalized_maxwell_solve, point.visco_mat_id, point.gp, point.ele_gid);
    metadata.branches.reserve(generalized_maxwell_numbranch_value);

    for (int i = 0; i < generalized_maxwell_numbranch_value; ++i)
    {
      BranchMetadata branch_metadata;
      branch_metadata.summands = branchespotsum.at(i);
      branch_metadata.tau = branchtau.at(i);

      FOUR_C_ASSERT_ALWAYS(branch_metadata.tau > 0.0,
          "Invalid branch relaxation time TAU={} in VISCO_GeneralizedMaxwell (MAT {}, branch {}, "
          "GP {}, ELE {}). Expected TAU > 0.",
          branch_metadata.tau, generalized_maxwell_summand_mat_id, i, point.gp, point.ele_gid);

      branch_metadata.properties.clear();
      elast_hyper_properties(branch_metadata.summands, branch_metadata.properties);

      if (branch_metadata.properties.anisoprinc)
        FOUR_C_THROW(
            "Unsupported branch formulation in VISCO_GeneralizedMaxwell for MAT_ViscoElastHyper "
            "(MAT {}, GP {}, ELE {}): anisoprinc branch response is not implemented.",
            point.visco_mat_id, point.gp, point.ele_gid);

      if (branch_metadata.properties.anisomod)
        FOUR_C_THROW(
            "Unsupported branch formulation in VISCO_GeneralizedMaxwell for MAT_ViscoElastHyper "
            "(MAT {}, GP {}, ELE {}): anisomod branch response is not implemented.",
            point.visco_mat_id, point.gp, point.ele_gid);

      metadata.branches.push_back(std::move(branch_metadata));
    }

    metadata_ = std::move(metadata);
  }

  void GeneralizedMaxwellContribution::build_runtime_context(
      const ContributionSetupContext& context)
  {
    runtime_context_.reset();
    const auto& point = context.point;

    const Teuchos::ParameterList& structural_dynamic_parameters =
        Global::Problem::instance()->structural_dynamic_params();

    FOUR_C_ASSERT_ALWAYS(structural_dynamic_parameters.isParameter("DYNAMICTYPE"),
        "Missing DYNAMICTYPE in STRUCTURAL DYNAMIC parameters while building generalized Maxwell "
        "runtime context for MAT_ViscoElastHyper (MAT {}, GP {}, ELE {}).",
        point.visco_mat_id, point.gp, point.ele_gid);

    RuntimeContext runtime_context;
    const auto dyntype = Teuchos::getIntegralValue<Inpar::Solid::DynamicType>(
        structural_dynamic_parameters, "DYNAMICTYPE");
    if (dyntype == Inpar::Solid::DynamicType::OneStepTheta)
      runtime_context.one_step_theta =
          structural_dynamic_parameters.sublist("ONESTEPTHETA").get<double>("THETA");

    runtime_context_ = runtime_context;
  }


  void Kernels::evaluate_generalized_maxwell_kernel(StressVector& q_total,
      TangentMatrix& cmatq_total, PointHistory& current_branch_elastic_stress,
      PointHistory& current_branch_stress, const std::vector<double>& branch_taus,
      const GeneralizedMaxwellKernelInput& input,
      const BranchResponseEvaluator& evaluate_branch_response)
  {
    FOUR_C_ASSERT_ALWAYS(input.dt > 0.0,
        "Invalid time step size dt={} in generalized Maxwell kernel evaluation (MAT {}, GP {}, "
        "ELE {}). Expected dt > 0.",
        input.dt, input.visco_mat_id, input.gp, input.ele_gid);

    FOUR_C_ASSERT_ALWAYS(input.previous_branch_elastic_stress != nullptr,
        "Missing previous generalized Maxwell elastic branch history in kernel evaluation (MAT "
        "{}, GP {}, ELE {}).",
        input.visco_mat_id, input.gp, input.ele_gid);

    FOUR_C_ASSERT_ALWAYS(input.previous_branch_stress != nullptr,
        "Missing previous generalized Maxwell viscous branch history in kernel evaluation (MAT "
        "{}, GP {}, ELE {}).",
        input.visco_mat_id, input.gp, input.ele_gid);

    const PointHistory& previous_branch_elastic_stress = *input.previous_branch_elastic_stress;
    const PointHistory& previous_branch_stress = *input.previous_branch_stress;

    const int numbranch = static_cast<int>(branch_taus.size());
    FOUR_C_ASSERT_ALWAYS(
        previous_branch_elastic_stress.size() == static_cast<unsigned int>(numbranch),
        "Invalid generalized Maxwell elastic branch history size in kernel evaluation (MAT {}, "
        "GP {}, ELE {}): expected {} entries but got {}.",
        input.visco_mat_id, input.gp, input.ele_gid, numbranch,
        previous_branch_elastic_stress.size());

    FOUR_C_ASSERT_ALWAYS(previous_branch_stress.size() == static_cast<unsigned int>(numbranch),
        "Invalid generalized Maxwell viscous branch history size in kernel evaluation (MAT {}, "
        "GP {}, ELE {}): expected {} entries but got {}.",
        input.visco_mat_id, input.gp, input.ele_gid, numbranch, previous_branch_stress.size());

    current_branch_elastic_stress.assign(
        numbranch, StressVector(Core::LinAlg::Initialization::zero));
    current_branch_stress.assign(numbranch, StressVector(Core::LinAlg::Initialization::zero));

    for (int branch_index = 0; branch_index < numbranch; ++branch_index)
    {
      const double tau = branch_taus.at(branch_index);
      FOUR_C_ASSERT_ALWAYS(tau > 0.0,
          "Invalid generalized Maxwell branch relaxation time TAU={} in kernel evaluation (MAT "
          "{}, GP {}, ELE {}, branch {}).",
          tau, input.visco_mat_id, input.gp, input.ele_gid, branch_index);

      StressVector& branch_elastic_stress = current_branch_elastic_stress.at(branch_index);
      TangentMatrix branch_cmat(Core::LinAlg::Initialization::zero);
      evaluate_branch_response(branch_index, branch_elastic_stress, branch_cmat);

      StressVector& branch_stress = current_branch_stress.at(branch_index);
      branch_stress.clear();

      double branch_cmat_scale = 1.0;
      switch (input.solve_kind)
      {
        case GeneralizedMaxwellSolveKind::one_step_theta:
        {
          const double theta = input.one_step_theta;
          const double lambda_1 = tau / (tau + theta * input.dt);
          const double lambda_2 = (tau - input.dt + theta * input.dt) / tau;

          branch_cmat_scale = lambda_1;
          branch_stress.update(lambda_2, previous_branch_stress.at(branch_index), 1.0);
          branch_stress.update(1.0, branch_elastic_stress, 1.0);
          branch_stress.update(-1.0, previous_branch_elastic_stress.at(branch_index), 1.0);
          branch_stress.scale(lambda_1);
          break;
        }
        case GeneralizedMaxwellSolveKind::exponential_time_discretization:
        {
          const double xi_1 = std::exp(-input.dt / tau);
          const double xi_2 = std::exp(-input.dt / (2.0 * tau));

          branch_cmat_scale = xi_2;
          branch_stress.update(xi_1, previous_branch_stress.at(branch_index), 1.0);
          branch_stress.update(xi_2, branch_elastic_stress, 1.0);
          branch_stress.update(-xi_2, previous_branch_elastic_stress.at(branch_index), 1.0);
          break;
        }
      }

      q_total.update(1.0, branch_stress, 1.0);
      cmatq_total.update(branch_cmat_scale, branch_cmat, 1.0);
    }
  }

}  // namespace Mat::ViscoElast

FOUR_C_NAMESPACE_CLOSE
