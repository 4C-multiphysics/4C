// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_viscoelast_fsls.hpp"

#include "4C_global_data.hpp"
#include "4C_material_parameter_base.hpp"

#include <Teuchos_ParameterList.hpp>

#include <cmath>
#include <cstddef>

FOUR_C_NAMESPACE_OPEN

namespace Mat::ViscoElast
{

  PAR::Fsls::Fsls(const Core::Mat::PAR::Parameter::Data& matdata)
      : Parameter(matdata),
        tau_(matdata.parameters.get<double>("TAU")),
        alpha_(matdata.parameters.get<double>("ALPHA")),
        beta_(matdata.parameters.get<double>("BETA"))
  {
    FOUR_C_ASSERT_ALWAYS(tau_ > 0.0,
        "Invalid TAU={} in VISCO_FSLS (MAT {}). TAU has to be positive.", tau_, matdata.id);

    FOUR_C_ASSERT_ALWAYS(alpha_ >= 0.0 && alpha_ < 1.0,
        "Invalid ALPHA={} in VISCO_FSLS (MAT {}). Expected 0 <= ALPHA < 1.", alpha_, matdata.id);
  }

  Fsls::Fsls(PAR::Fsls* params) : params_(params) {}

  void Fsls::read_material_parameters_visco(
      double& tau, double& beta, double& alpha, std::string& solve)
  {
    tau = params_->tau_;
    alpha = params_->alpha_;
    beta = params_->beta_;
  }


  void FslsContribution::setup(const ContributionSetupContext& context)
  {
    build_metadata(context);
    build_runtime_context(context);
  }


  void FslsContribution::evaluate(const FslsEvaluateContext& context)
  {
    const auto& point = context.base.point;

    const Metadata& metadata =
        require_metadata("evaluating FSLS response", point.gp, point.ele_gid);
    const auto& fsls_previous_history = context.base.state.fsls_previous_history();

    Kernels::FslsKernelInput kernel_input;
    kernel_input.visco_mat_id = point.visco_mat_id;
    kernel_input.gp = point.gp;
    kernel_input.ele_gid = point.ele_gid;
    kernel_input.dt = context.base.dt;
    kernel_input.tau = metadata.tau;
    kernel_input.alpha = metadata.alpha;
    kernel_input.beta = metadata.beta;
    kernel_input.previous_history = &fsls_previous_history;

    Kernels::FslsStressVector q_current_for_history(Core::LinAlg::Initialization::zero);
    Kernels::FslsStressVector q_additive(Core::LinAlg::Initialization::zero);
    Kernels::FslsTangentMatrix cmatq_additive(Core::LinAlg::Initialization::zero);

    Kernels::evaluate_fsls_kernel(context.base.stress, context.base.cmat, q_current_for_history,
        q_additive, cmatq_additive, kernel_input);

    context.base.stress.update(1.0, q_additive, 1.0);
    context.base.cmat.update(1.0, cmatq_additive, 1.0);
    context.base.state.set_fsls_current_artificial_stress(point.gp, q_current_for_history);
  }


  void FslsContribution::update(const ContributionUpdateContext& context) { (void)context; }


  unsigned int FslsContribution::history_capacity_for_update() const
  {
    const RuntimeContext& runtime_context =
        require_runtime_context("reading FSLS history capacity", -1, -1);

    FOUR_C_ASSERT_ALWAYS(runtime_context.max_history_size > 0,
        "Invalid FSLS runtime history capacity {} in MAT_ViscoElastHyper. Expected a positive "
        "history capacity.",
        runtime_context.max_history_size);

    return runtime_context.max_history_size;
  }


  const FslsContribution::Metadata& FslsContribution::require_metadata(
      const char* context, const int gp, const int ele_gid) const
  {
    FOUR_C_ASSERT_ALWAYS(metadata_.has_value(),
        "Missing VISCO_FSLS metadata cache while {} in MAT_ViscoElastHyper (GP {}, ELE {}). Run "
        "setup() before evaluation.",
        context, gp, ele_gid);

    return metadata_.value();
  }


  const FslsContribution::RuntimeContext& FslsContribution::require_runtime_context(
      const char* context, const int gp, const int ele_gid) const
  {
    FOUR_C_ASSERT_ALWAYS(runtime_context_.has_value(),
        "Missing VISCO_FSLS runtime context while {} in MAT_ViscoElastHyper (GP {}, ELE {}). Run "
        "setup() before update.",
        context, gp, ele_gid);

    return runtime_context_.value();
  }


  void FslsContribution::build_metadata(const ContributionSetupContext& context)
  {
    const auto& point = context.point;

    FOUR_C_ASSERT_ALWAYS(context.visco_summands.size() == context.visco_summand_mat_ids.size(),
        "Invalid FSLS setup context in MAT_ViscoElastHyper (MAT {}): visco summand count {} does "
        "not match MAT id count {}.",
        point.visco_mat_id, context.visco_summands.size(), context.visco_summand_mat_ids.size());

    Metadata metadata;
    int fsls_model_count = 0;
    std::string solve;

    for (std::size_t p = 0; p < context.visco_summands.size(); ++p)
    {
      auto fsls = std::dynamic_pointer_cast<Fsls>(context.visco_summands.at(p));
      if (fsls == nullptr) continue;

      ++fsls_model_count;
      metadata.summand_mat_id = context.visco_summand_mat_ids.at(p);
      fsls->read_material_parameters_visco(metadata.tau, metadata.beta, metadata.alpha, solve);
    }

    FOUR_C_ASSERT_ALWAYS(fsls_model_count == 1,
        "Invalid VISCO_FSLS setup in MAT_ViscoElastHyper (MAT {}, GP {}, ELE {}): expected "
        "exactly one VISCO_FSLS summand but found {}.",
        point.visco_mat_id, point.gp, point.ele_gid, fsls_model_count);

    FOUR_C_ASSERT_ALWAYS(metadata.tau > 0.0,
        "Invalid TAU={} in VISCO_FSLS (MAT {}, referenced by MAT_ViscoElastHyper MAT {}, GP {}, "
        "ELE {}). TAU has to be positive.",
        metadata.tau, metadata.summand_mat_id, point.visco_mat_id, point.gp, point.ele_gid);

    FOUR_C_ASSERT_ALWAYS(metadata.alpha >= 0.0 && metadata.alpha < 1.0,
        "Invalid ALPHA={} in VISCO_FSLS (MAT {}, referenced by MAT_ViscoElastHyper MAT {}, GP "
        "{}, ELE {}). Expected 0 <= ALPHA < 1.",
        metadata.alpha, metadata.summand_mat_id, point.visco_mat_id, point.gp, point.ele_gid);

    metadata_ = metadata;
  }


  void FslsContribution::build_runtime_context(const ContributionSetupContext& context)
  {
    const auto& point = context.point;
    const Teuchos::ParameterList& structural_dynamic_parameters =
        Global::Problem::instance()->structural_dynamic_params();

    FOUR_C_ASSERT_ALWAYS(structural_dynamic_parameters.isParameter("NUMSTEP"),
        "Missing NUMSTEP in STRUCTURAL DYNAMIC parameters while deriving FSLS history capacity "
        "for MAT_ViscoElastHyper (MAT {}, GP {}, ELE {}).",
        point.visco_mat_id, point.gp, point.ele_gid);

    const int numsteps = structural_dynamic_parameters.get<int>("NUMSTEP");
    FOUR_C_ASSERT_ALWAYS(numsteps >= 0,
        "Invalid NUMSTEP={} while deriving FSLS history capacity for MAT_ViscoElastHyper (MAT "
        "{}, GP {}, ELE {}). Expected NUMSTEP >= 0.",
        numsteps, point.visco_mat_id, point.gp, point.ele_gid);

    runtime_context_ = RuntimeContext{.max_history_size = static_cast<unsigned int>(numsteps + 1)};
  }


  void Kernels::evaluate_fsls_kernel(const FslsStressVector& stress, const FslsTangentMatrix& cmat,
      FslsStressVector& q_current_for_history, FslsStressVector& q_additive,
      FslsTangentMatrix& cmatq_additive, const FslsKernelInput& input)
  {
    FOUR_C_ASSERT_ALWAYS(input.dt > 0.0,
        "Invalid time step size dt={} in FSLS kernel evaluation (MAT {}, GP {}, ELE {}). "
        "Expected dt > 0.",
        input.dt, input.visco_mat_id, input.gp, input.ele_gid);

    FOUR_C_ASSERT_ALWAYS(input.tau > 0.0,
        "Invalid TAU={} in FSLS kernel evaluation (MAT {}, GP {}, ELE {}). Expected TAU > 0.",
        input.tau, input.visco_mat_id, input.gp, input.ele_gid);

    FOUR_C_ASSERT_ALWAYS(input.alpha >= 0.0 && input.alpha < 1.0,
        "Invalid ALPHA={} in FSLS kernel evaluation (MAT {}, GP {}, ELE {}). Expected 0 <= "
        "ALPHA < 1.",
        input.alpha, input.visco_mat_id, input.gp, input.ele_gid);

    FOUR_C_ASSERT_ALWAYS(input.previous_history != nullptr,
        "Missing FSLS history state in kernel evaluation (MAT {}, GP {}, ELE {}).",
        input.visco_mat_id, input.gp, input.ele_gid);

    const FslsHistory& previous_history = *input.previous_history;
    FOUR_C_ASSERT_ALWAYS(!previous_history.empty(),
        "Missing FSLS history state in kernel evaluation (MAT {}, GP {}, ELE {}).",
        input.visco_mat_id, input.gp, input.ele_gid);

    FOUR_C_ASSERT_ALWAYS(input.gp >= 0 && input.gp < static_cast<int>(previous_history.size()),
        "Invalid Gauss point index GP={} for FSLS history in kernel evaluation (MAT {}, ELE "
        "{}). History container size is {}.",
        input.gp, input.visco_mat_id, input.ele_gid, previous_history.size());

    const auto& fsls_history_at_gp = previous_history.at(input.gp);
    const int hs = fsls_history_at_gp.size();
    FOUR_C_ASSERT_ALWAYS(hs > 0,
        "Invalid FSLS history size {} at GP {} in kernel evaluation (MAT {}, ELE {}). "
        "Expected at least one entry.",
        hs, input.gp, input.visco_mat_id, input.ele_gid);

    // calculate artificial history stress Qq with weights b_j
    // Qq = sum[j=1 up to j=n][b_j*Q_(n+1-j)] (short: b*Qj)
    // b_j = (j-1-alpha)/j * b_(j-1), with b_0 = 1
    double bj = 1.;
    double fac = 1.;
    FslsStressVector q_history_sum(Core::LinAlg::Initialization::zero);

    for (int j = 1; j <= hs; j++)
    {
      fac = (j - 1. - input.alpha) / j;
      bj = bj * fac;

      FslsStressVector qj(fsls_history_at_gp.at(hs - j));
      q_history_sum.update(bj, qj, 1.0);
    }

    const double dtalpha = std::pow(input.dt, input.alpha);
    const double taualpha = std::pow(input.tau, input.alpha);
    const double denominator = dtalpha + taualpha;
    FOUR_C_ASSERT_ALWAYS(denominator > 0.0,
        "Invalid FSLS update denominator dt^alpha + tau^alpha = {} in kernel evaluation (MAT "
        "{}, GP {}, ELE {}): dt={}, tau={}, alpha={}. Expected a positive denominator.",
        denominator, input.visco_mat_id, input.gp, input.ele_gid, input.dt, input.tau, input.alpha);

    const double lambda_1 = dtalpha / denominator;
    const double lambda_2 = -1. * taualpha / denominator;

    q_current_for_history.update(lambda_1 * input.beta, stress, 0.);
    q_current_for_history.update(lambda_2, q_history_sum, 1.);

    q_additive.update(1.0, q_current_for_history, 0.0);
    q_additive.update(input.beta, stress, -1.);

    cmatq_additive.update(lambda_1 * input.beta, cmat, 0.);
    cmatq_additive.update(input.beta, cmat, -1.);
  }

}  // namespace Mat::ViscoElast

FOUR_C_NAMESPACE_CLOSE
