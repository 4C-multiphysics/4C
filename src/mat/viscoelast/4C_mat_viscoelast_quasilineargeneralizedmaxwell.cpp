// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_viscoelast_quasilineargeneralizedmaxwell.hpp"

#include "4C_global_data.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_utils_densematrix_inverse.hpp"
#include "4C_mat_service.hpp"
#include "4C_material_parameter_base.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <cmath>
#include <span>
#include <utility>

FOUR_C_NAMESPACE_OPEN

namespace Mat::ViscoElast
{
  namespace
  {
    enum class QuasiLinearSolveKind
    {
      one_step_theta,
      exponential_time_discretization
    };

    using StressVector = Core::LinAlg::Matrix<6, 1>;
    using TangentMatrix = Core::LinAlg::Matrix<6, 6>;
    using PointHistory = std::vector<StressVector>;

    struct QuasiLinearGeneralizedMaxwellKernelInput
    {
      int visco_mat_id = -1;
      int gp = -1;
      int ele_gid = -1;
      double dt = 0.0;
      double one_step_theta = 0.5;
      double viscosity = 0.0;
      QuasiLinearSolveKind solve_kind = QuasiLinearSolveKind::exponential_time_discretization;
      const PointHistory* previous_branch_elastic_stress = nullptr;
      const PointHistory* previous_branch_stress = nullptr;
      const StressVector* previous_dashpot_strain = nullptr;
      const StressVector* previous_dashpot_stress = nullptr;
    };


    void evaluate_quasi_linear_generalized_maxwell_kernel(StressVector& q_total,
        TangentMatrix& cmatq_total, PointHistory& current_branch_elastic_stress,
        PointHistory& current_branch_stress, StressVector& current_dashpot_strain,
        StressVector& current_dashpot_stress, const StressVector& base_stress,
        const TangentMatrix& base_cmat, const StressVector& current_strain,
        const TangentMatrix& strain_to_stress_identity, const std::span<const double> branch_betas,
        const std::span<const double> branch_taus,
        const QuasiLinearGeneralizedMaxwellKernelInput& input)
    {
      FOUR_C_ASSERT_ALWAYS(input.dt > 0.0,
          "Invalid time step size dt={} in quasi-linear generalized Maxwell kernel evaluation "
          "(MAT {}, GP {}, ELE {}). Expected dt > 0.",
          input.dt, input.visco_mat_id, input.gp, input.ele_gid);

      FOUR_C_ASSERT_ALWAYS(branch_betas.size() == branch_taus.size(),
          "Invalid quasi-linear generalized Maxwell branch declaration in kernel evaluation "
          "(MAT {}, GP {}, ELE {}): BETA has size {} but TAU has size {}.",
          input.visco_mat_id, input.gp, input.ele_gid, branch_betas.size(), branch_taus.size());

      FOUR_C_ASSERT_ALWAYS(input.previous_branch_elastic_stress != nullptr,
          "Missing previous quasi-linear generalized Maxwell elastic branch history in kernel "
          "evaluation (MAT {}, GP {}, ELE {}).",
          input.visco_mat_id, input.gp, input.ele_gid);

      FOUR_C_ASSERT_ALWAYS(input.previous_branch_stress != nullptr,
          "Missing previous quasi-linear generalized Maxwell branch stress history in kernel "
          "evaluation (MAT {}, GP {}, ELE {}).",
          input.visco_mat_id, input.gp, input.ele_gid);

      const PointHistory& previous_branch_elastic_stress = *input.previous_branch_elastic_stress;
      const PointHistory& previous_branch_stress = *input.previous_branch_stress;
      const int numbranch = static_cast<int>(branch_taus.size());

      FOUR_C_ASSERT_ALWAYS(previous_branch_elastic_stress.size() == branch_taus.size(),
          "Invalid quasi-linear generalized Maxwell elastic branch history size in kernel "
          "evaluation (MAT {}, GP {}, ELE {}): expected {} entries but got {}.",
          input.visco_mat_id, input.gp, input.ele_gid, numbranch,
          previous_branch_elastic_stress.size());

      FOUR_C_ASSERT_ALWAYS(previous_branch_stress.size() == branch_taus.size(),
          "Invalid quasi-linear generalized Maxwell branch stress history size in kernel "
          "evaluation (MAT {}, GP {}, ELE {}): expected {} entries but got {}.",
          input.visco_mat_id, input.gp, input.ele_gid, numbranch, previous_branch_stress.size());

      current_branch_elastic_stress.assign(
          numbranch, StressVector(Core::LinAlg::Initialization::zero));
      current_branch_stress.assign(numbranch, StressVector(Core::LinAlg::Initialization::zero));

      for (int branch_index = 0; branch_index < numbranch; ++branch_index)
      {
        const double beta = branch_betas[branch_index];
        const double tau = branch_taus[branch_index];

        StressVector& branch_elastic_stress = current_branch_elastic_stress.at(branch_index);
        branch_elastic_stress.update(beta, base_stress, 0.0);

        StressVector& branch_stress = current_branch_stress.at(branch_index);
        branch_stress.clear();

        double branch_cmat_scale = 0.0;
        switch (input.solve_kind)
        {
          case QuasiLinearSolveKind::one_step_theta:
          {
            const double theta = input.one_step_theta;
            const double lambda_1 = tau / (tau + theta * input.dt);
            const double lambda_2 = (tau - input.dt + theta * input.dt) / tau;

            branch_cmat_scale = beta * lambda_1;
            branch_stress.update(lambda_2, previous_branch_stress.at(branch_index), 1.0);
            branch_stress.update(1.0, branch_elastic_stress, 1.0);
            branch_stress.update(-1.0, previous_branch_elastic_stress.at(branch_index), 1.0);
            branch_stress.scale(lambda_1);
            break;
          }
          case QuasiLinearSolveKind::exponential_time_discretization:
          {
            const double h = input.dt / tau;
            const double decay = std::exp(-h);
            const double phi_1 = -std::expm1(-h) / h;

            branch_cmat_scale = beta * phi_1;
            branch_stress.update(decay, previous_branch_stress.at(branch_index), 1.0);
            branch_stress.update(phi_1, branch_elastic_stress, 1.0);
            branch_stress.update(-phi_1, previous_branch_elastic_stress.at(branch_index), 1.0);
            break;
          }
        }

        q_total.update(1.0, branch_stress, 1.0);
        cmatq_total.update(branch_cmat_scale, base_cmat, 1.0);
      }

      current_dashpot_strain.update(1.0, current_strain, 0.0);
      current_dashpot_stress.clear();
      if (input.viscosity > 0.0)
      {
        FOUR_C_ASSERT_ALWAYS(input.previous_dashpot_strain != nullptr,
            "Missing previous quasi-linear generalized Maxwell dashpot strain history in kernel "
            "evaluation (MAT {}, GP {}, ELE {}).",
            input.visco_mat_id, input.gp, input.ele_gid);

        double dashpot_tangent_scale = 0.0;
        current_dashpot_stress.update(1.0, current_strain, 0.0);
        current_dashpot_stress.update(-1.0, *input.previous_dashpot_strain, 1.0);

        switch (input.solve_kind)
        {
          case QuasiLinearSolveKind::one_step_theta:
          {
            FOUR_C_ASSERT_ALWAYS(input.previous_dashpot_stress != nullptr,
                "Missing previous quasi-linear generalized Maxwell dashpot stress history in "
                "kernel evaluation (MAT {}, GP {}, ELE {}).",
                input.visco_mat_id, input.gp, input.ele_gid);

            const double theta = input.one_step_theta;
            FOUR_C_ASSERT_ALWAYS(theta > 0.0,
                "Invalid one-step-theta value theta={} for quasi-linear generalized Maxwell "
                "parallel dashpot in kernel evaluation (MAT {}, GP {}, ELE {}). Expected "
                "theta > 0.",
                theta, input.visco_mat_id, input.gp, input.ele_gid);

            dashpot_tangent_scale = input.viscosity / (theta * input.dt);
            current_dashpot_stress.scale(dashpot_tangent_scale);
            current_dashpot_stress.update(
                -(1.0 - theta) / theta, *input.previous_dashpot_stress, 1.0);
            break;
          }
          case QuasiLinearSolveKind::exponential_time_discretization:
          {
            dashpot_tangent_scale = input.viscosity / input.dt;
            current_dashpot_stress.scale(dashpot_tangent_scale);
            break;
          }
        }

        q_total.update(1.0, current_dashpot_stress, 1.0);
        cmatq_total.update(dashpot_tangent_scale, strain_to_stress_identity, 1.0);
      }
    }
  }  // namespace


  PAR::QuasiLinearGeneralizedMaxwell::QuasiLinearGeneralizedMaxwell(
      const Core::Mat::PAR::Parameter::Data& matdata)
      : Parameter(matdata),
        beta_(matdata.parameters.get<std::vector<double>>("BETA")),
        tau_(matdata.parameters.get<std::vector<double>>("TAU")),
        solve_(matdata.parameters.get<std::string>("SOLVE")),
        viscosity_(matdata.parameters.get<double>("VISCOSITY"))
  {
    FOUR_C_ASSERT_ALWAYS(beta_.size() == tau_.size(),
        "Invalid VISCO_QuasiLinearGeneralizedMaxwell (MAT {}): BETA has size {} but TAU has "
        "size {}.",
        matdata.id, beta_.size(), tau_.size());

    FOUR_C_ASSERT_ALWAYS(!beta_.empty() || viscosity_ > 0.0,
        "Invalid VISCO_QuasiLinearGeneralizedMaxwell (MAT {}): provide at least one Maxwell "
        "branch or a positive VISCOSITY.",
        matdata.id);
  }


  QuasiLinearGeneralizedMaxwell::QuasiLinearGeneralizedMaxwell(
      PAR::QuasiLinearGeneralizedMaxwell* params)
      : params_(params)
  {
  }


  const PAR::QuasiLinearGeneralizedMaxwell& QuasiLinearGeneralizedMaxwell::parameters() const
  {
    return *params_;
  }


  void QuasiLinearGeneralizedMaxwellContribution::setup(const ContributionSetupContext& context)
  {
    FOUR_C_THROW(
        "VISCO_QuasiLinearGeneralizedMaxwell setup requires the QLV-specific setup context in "
        "MAT_ViscoElastHyper (MAT {}, GP {}, ELE {}).",
        context.point.visco_mat_id, context.point.gp, context.point.ele_gid);
  }


  void QuasiLinearGeneralizedMaxwellContribution::setup(
      const QuasiLinearGeneralizedMaxwellSetupContext& context)
  {
    build_metadata(context);
    build_runtime_context(context.base);
  }


  void QuasiLinearGeneralizedMaxwellContribution::evaluate(
      const QuasiLinearGeneralizedMaxwellEvaluateContext& context)
  {
    const auto& point = context.base.point;

    const Metadata& metadata = require_metadata(
        "evaluating quasi-linear generalized Maxwell response", point.gp, point.ele_gid);

    double one_step_theta = 0.5;
    if (metadata.solve_kind == SolveKind::one_step_theta)
    {
      const RuntimeContext& runtime_context = require_runtime_context(
          "evaluating quasi-linear generalized Maxwell response", point.gp, point.ele_gid);
      one_step_theta = runtime_context.one_step_theta;
    }

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

    Core::LinAlg::Matrix<3, 1> dPI(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 1> ddPII(Core::LinAlg::Initialization::zero);
    elast_hyper_evaluate_invariant_derivatives(prinv, dPI, ddPII, metadata.base_summands,
        metadata.base_properties, point.gp, point.ele_gid);

    Core::LinAlg::SymmetricTensor<double, 3, 3> base_stress_tensor{};
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> base_cmat_tensor{};
    elast_hyper_add_isotropic_stress_cmat(
        base_stress_tensor, base_cmat_tensor, c, i_c, prinv, dPI, ddPII);

    StressVector base_stress(Core::LinAlg::Initialization::zero);
    base_stress.update(1.0, Core::LinAlg::make_stress_like_voigt_view(base_stress_tensor), 0.0);
    TangentMatrix base_cmat(Core::LinAlg::Initialization::zero);
    base_cmat.update(1.0, Core::LinAlg::make_stress_like_voigt_view(base_cmat_tensor), 0.0);

    Core::LinAlg::Matrix<6, 6> strain_to_stress_identity(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Voigt::fourth_order_identity_matrix<Core::LinAlg::Voigt::NotationType::stress,
        Core::LinAlg::Voigt::NotationType::strain>(strain_to_stress_identity);

    const auto& previous_branch_elastic_stress =
        context.base.state.quasi_linear_generalized_maxwell_prev_branch_elastic_stress(point.gp);
    const auto& previous_branch_stress =
        context.base.state.quasi_linear_generalized_maxwell_prev_branch_stress(point.gp);
    const auto& previous_dashpot_strain =
        context.base.state.quasi_linear_generalized_maxwell_prev_dashpot_strain(point.gp);
    const auto& previous_dashpot_stress =
        context.base.state.quasi_linear_generalized_maxwell_prev_dashpot_stress(point.gp);

    QuasiLinearGeneralizedMaxwellKernelInput kernel_input;
    kernel_input.visco_mat_id = point.visco_mat_id;
    kernel_input.gp = point.gp;
    kernel_input.ele_gid = point.ele_gid;
    kernel_input.dt = context.base.dt;
    kernel_input.one_step_theta = one_step_theta;
    kernel_input.viscosity = metadata.viscosity;
    kernel_input.previous_branch_elastic_stress = &previous_branch_elastic_stress;
    kernel_input.previous_branch_stress = &previous_branch_stress;
    kernel_input.previous_dashpot_strain = &previous_dashpot_strain;
    kernel_input.previous_dashpot_stress = &previous_dashpot_stress;
    switch (metadata.solve_kind)
    {
      case SolveKind::one_step_theta:
        kernel_input.solve_kind = QuasiLinearSolveKind::one_step_theta;
        break;
      case SolveKind::exponential_time_discretization:
        kernel_input.solve_kind = QuasiLinearSolveKind::exponential_time_discretization;
        break;
    }

    PointHistory current_branch_elastic_stress;
    PointHistory current_branch_stress;
    StressVector current_dashpot_strain(Core::LinAlg::Initialization::zero);
    StressVector current_dashpot_stress(Core::LinAlg::Initialization::zero);
    StressVector q(Core::LinAlg::Initialization::zero);
    TangentMatrix cmatq(Core::LinAlg::Initialization::zero);

    evaluate_quasi_linear_generalized_maxwell_kernel(q, cmatq, current_branch_elastic_stress,
        current_branch_stress, current_dashpot_strain, current_dashpot_stress, base_stress,
        base_cmat, gl_stress, strain_to_stress_identity,
        std::span<const double>(metadata.beta.data(), metadata.beta.size()),
        std::span<const double>(metadata.tau.data(), metadata.tau.size()), kernel_input);

    context.base.state.set_quasi_linear_generalized_maxwell_current_point(
        point.gp, current_branch_elastic_stress, current_branch_stress);
    context.base.state.set_quasi_linear_generalized_maxwell_current_dashpot_strain(
        point.gp, current_dashpot_strain);
    context.base.state.set_quasi_linear_generalized_maxwell_current_dashpot_stress(
        point.gp, current_dashpot_stress);
    context.base.stress.update(1.0, q, 1.0);
    context.base.cmat.update(1.0, cmatq, 1.0);
  }


  void QuasiLinearGeneralizedMaxwellContribution::update(const ContributionUpdateContext& context)
  {
    (void)context;
  }


  std::size_t QuasiLinearGeneralizedMaxwellContribution::history_entry_count_for_setup() const
  {
    return require_metadata("reading branch count", -1, -1).tau.size();
  }


  QuasiLinearGeneralizedMaxwellContribution::SolveKind
  QuasiLinearGeneralizedMaxwellContribution::parse_solve_kind(
      const std::string& solve, const int visco_mat_id, const int gp, const int ele_gid)
  {
    if (solve == "OneStepTheta") return SolveKind::one_step_theta;
    if (solve == "ExponentialTimeDiscretization") return SolveKind::exponential_time_discretization;

    FOUR_C_THROW(
        "Invalid SOLVE='{}' in VISCO_QuasiLinearGeneralizedMaxwell for MAT_ViscoElastHyper "
        "(MAT {}, GP {}, ELE {}). Use OneStepTheta or ExponentialTimeDiscretization.",
        solve, visco_mat_id, gp, ele_gid);

    return SolveKind::exponential_time_discretization;
  }


  const QuasiLinearGeneralizedMaxwellContribution::Metadata&
  QuasiLinearGeneralizedMaxwellContribution::require_metadata(
      const char* context, const int gp, const int ele_gid) const
  {
    FOUR_C_ASSERT_ALWAYS(metadata_.has_value(),
        "Missing VISCO_QuasiLinearGeneralizedMaxwell metadata cache while {} in "
        "MAT_ViscoElastHyper (GP {}, ELE {}). Run setup() before evaluation.",
        context, gp, ele_gid);

    return metadata_.value();
  }


  const QuasiLinearGeneralizedMaxwellContribution::RuntimeContext&
  QuasiLinearGeneralizedMaxwellContribution::require_runtime_context(
      const char* context, const int gp, const int ele_gid) const
  {
    FOUR_C_ASSERT_ALWAYS(runtime_context_.has_value(),
        "Missing quasi-linear generalized Maxwell runtime context while {} in MAT_ViscoElastHyper "
        "(GP {}, ELE {}). Run setup() before evaluation.",
        context, gp, ele_gid);

    return runtime_context_.value();
  }


  void QuasiLinearGeneralizedMaxwellContribution::build_metadata(
      const QuasiLinearGeneralizedMaxwellSetupContext& context)
  {
    metadata_.reset();
    const ContributionSetupContext& base_context = context.base;
    const auto& point = base_context.point;

    FOUR_C_ASSERT_ALWAYS(
        base_context.visco_summands.size() == base_context.visco_summand_mat_ids.size(),
        "Invalid quasi-linear generalized Maxwell setup context in MAT_ViscoElastHyper (MAT {}): "
        "visco summand count {} does not match MAT id count {}.",
        point.visco_mat_id, base_context.visco_summands.size(),
        base_context.visco_summand_mat_ids.size());

    int quasi_linear_model_count = 0;
    std::shared_ptr<QuasiLinearGeneralizedMaxwell> quasi_linear = nullptr;
    int quasi_linear_summand_mat_id = -1;

    for (std::size_t p = 0; p < base_context.visco_summands.size(); ++p)
    {
      auto current_quasi_linear = std::dynamic_pointer_cast<QuasiLinearGeneralizedMaxwell>(
          base_context.visco_summands.at(p));
      if (current_quasi_linear != nullptr)
      {
        ++quasi_linear_model_count;
        quasi_linear = current_quasi_linear;
        quasi_linear_summand_mat_id = base_context.visco_summand_mat_ids.at(p);
      }
    }

    FOUR_C_ASSERT_ALWAYS(quasi_linear_model_count == 1,
        "Invalid VISCO_QuasiLinearGeneralizedMaxwell setup in MAT_ViscoElastHyper (MAT {}): "
        "expected exactly one VISCO_QuasiLinearGeneralizedMaxwell summand but found {}.",
        point.visco_mat_id, quasi_linear_model_count);

    FOUR_C_ASSERT_ALWAYS(quasi_linear != nullptr,
        "Failed to resolve VISCO_QuasiLinearGeneralizedMaxwell summand in MAT_ViscoElastHyper "
        "(MAT {}).",
        point.visco_mat_id);

    const PAR::QuasiLinearGeneralizedMaxwell& parameters = quasi_linear->parameters();
    const std::vector<double>& beta = parameters.beta_;
    const std::vector<double>& tau = parameters.tau_;
    const std::string& solve = parameters.solve_;
    const double viscosity = parameters.viscosity_;

    FOUR_C_ASSERT_ALWAYS(beta.size() == tau.size(),
        "Invalid VISCO_QuasiLinearGeneralizedMaxwell setup in MAT_ViscoElastHyper (MAT {}, GP "
        "{}, ELE {}): BETA has size {} but TAU has size {}.",
        point.visco_mat_id, point.gp, point.ele_gid, beta.size(), tau.size());

    FOUR_C_ASSERT_ALWAYS(!beta.empty() || viscosity > 0.0,
        "Invalid VISCO_QuasiLinearGeneralizedMaxwell setup in MAT_ViscoElastHyper (MAT {}, GP "
        "{}, ELE {}): provide at least one Maxwell branch or a positive VISCOSITY.",
        point.visco_mat_id, point.gp, point.ele_gid);

    Metadata metadata;
    metadata.summand_mat_id = quasi_linear_summand_mat_id;
    metadata.solve_kind = parse_solve_kind(solve, point.visco_mat_id, point.gp, point.ele_gid);
    metadata.base_summands = context.elastic_summands;
    metadata.base_properties = context.elastic_summand_properties;
    metadata.beta = beta;
    metadata.tau = tau;
    metadata.viscosity = viscosity;

    FOUR_C_ASSERT_ALWAYS(!metadata.base_summands.empty(),
        "Invalid VISCO_QuasiLinearGeneralizedMaxwell setup in MAT_ViscoElastHyper (MAT {}, GP "
        "{}, ELE {}): no hyperelastic base summand is available.",
        point.visco_mat_id, point.gp, point.ele_gid);

    if (metadata.base_properties.anisoprinc)
      FOUR_C_THROW(
          "Unsupported base formulation in VISCO_QuasiLinearGeneralizedMaxwell for "
          "MAT_ViscoElastHyper (MAT {}, GP {}, ELE {}): anisoprinc base response is not "
          "implemented.",
          point.visco_mat_id, point.gp, point.ele_gid);

    if (metadata.base_properties.anisomod)
      FOUR_C_THROW(
          "Unsupported base formulation in VISCO_QuasiLinearGeneralizedMaxwell for "
          "MAT_ViscoElastHyper (MAT {}, GP {}, ELE {}): anisomod base response is not "
          "implemented.",
          point.visco_mat_id, point.gp, point.ele_gid);

    if (metadata.base_properties.coeffStretchesPrinc || metadata.base_properties.coeffStretchesMod)
      FOUR_C_THROW(
          "Unsupported base formulation in VISCO_QuasiLinearGeneralizedMaxwell for "
          "MAT_ViscoElastHyper (MAT {}, GP {}, ELE {}): stretch-coefficient base response is not "
          "implemented.",
          point.visco_mat_id, point.gp, point.ele_gid);

    metadata_ = std::move(metadata);
  }


  void QuasiLinearGeneralizedMaxwellContribution::build_runtime_context(
      const ContributionSetupContext& context)
  {
    runtime_context_.reset();
    const auto& point = context.point;

    const Teuchos::ParameterList& structural_dynamic_parameters =
        Global::Problem::instance()->structural_dynamic_params();

    FOUR_C_ASSERT_ALWAYS(structural_dynamic_parameters.isParameter("DYNAMICTYPE"),
        "Missing DYNAMICTYPE in STRUCTURAL DYNAMIC parameters while building quasi-linear "
        "generalized Maxwell runtime context for MAT_ViscoElastHyper (MAT {}, GP {}, ELE {}).",
        point.visco_mat_id, point.gp, point.ele_gid);

    RuntimeContext runtime_context;
    const auto dyntype = Teuchos::getIntegralValue<Inpar::Solid::DynamicType>(
        structural_dynamic_parameters, "DYNAMICTYPE");
    if (dyntype == Inpar::Solid::DynamicType::OneStepTheta)
      runtime_context.one_step_theta =
          structural_dynamic_parameters.sublist("ONESTEPTHETA").get<double>("THETA");

    runtime_context_ = runtime_context;
  }
}  // namespace Mat::ViscoElast

FOUR_C_NAMESPACE_CLOSE
