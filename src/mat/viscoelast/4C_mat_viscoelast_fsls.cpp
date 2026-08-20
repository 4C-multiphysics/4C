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

#include <algorithm>
#include <cmath>
#include <cstddef>

FOUR_C_NAMESPACE_OPEN

namespace Mat::ViscoElast
{

  PAR::Fsls::Fsls(const Core::Mat::PAR::Parameter::Data& matdata)
      : Parameter(matdata),
        tau_(matdata.parameters.get<double>("TAU")),
        alpha_(matdata.parameters.get<double>("ALPHA")),
        beta_(matdata.parameters.get<double>("BETA")),
        solve_kind_(matdata.parameters.get<FslsSolveKind>("SOLVE"))
  {
  }

  Fsls::Fsls(PAR::Fsls* params) : params_(params) {}

  void Fsls::read_material_parameters_visco(
      double& tau, double& beta, double& alpha, FslsSolveKind& solve)
  {
    tau = params_->tau_;
    alpha = params_->alpha_;
    beta = params_->beta_;
    solve = params_->solve_kind_;
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

    kernel_input.solve_kind = metadata.solve_kind;
    if (metadata.solve_kind == FslsSolveKind::l1)
    {
      if (static_cast<std::size_t>(point.gp) >= l1_step_cache_.size())
        l1_step_cache_.resize(point.gp + 1);
      kernel_input.l1 = {&l1_weight_cache_, &l1_step_cache_[point.gp]};
    }

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

    for (std::size_t p = 0; p < context.visco_summands.size(); ++p)
    {
      auto fsls = std::dynamic_pointer_cast<Fsls>(context.visco_summands.at(p));
      if (fsls == nullptr) continue;

      ++fsls_model_count;
      metadata.summand_mat_id = context.visco_summand_mat_ids.at(p);
      fsls->read_material_parameters_visco(
          metadata.tau, metadata.beta, metadata.alpha, metadata.solve_kind);
    }

    FOUR_C_ASSERT_ALWAYS(fsls_model_count == 1,
        "Invalid VISCO_FSLS setup in MAT_ViscoElastHyper (MAT {}, GP {}, ELE {}): expected "
        "exactly one VISCO_FSLS summand but found {}.",
        point.visco_mat_id, point.gp, point.ele_gid, fsls_model_count);

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


  namespace
  {
    /// Common preconditions shared by every FSLS kernel scheme; returns the per-GP history.
    const Kernels::FslsPointHistory& require_fsls_history_at_gp(
        const Kernels::FslsKernelInput& input)
    {
      // TAU and ALPHA come from VISCO_FSLS material parameters and are already validated by that
      // InputSpec's validators; dt is a runtime time step, not an input parameter, so it still
      // needs its own check here.
      FOUR_C_ASSERT_ALWAYS(input.dt > 0.0,
          "Invalid time step size dt={} in FSLS kernel evaluation (MAT {}, GP {}, ELE {}). "
          "Expected dt > 0.",
          input.dt, input.visco_mat_id, input.gp, input.ele_gid);

      FOUR_C_ASSERT_ALWAYS(input.previous_history != nullptr,
          "Missing FSLS history state in kernel evaluation (MAT {}, GP {}, ELE {}).",
          input.visco_mat_id, input.gp, input.ele_gid);

      const Kernels::FslsHistory& previous_history = *input.previous_history;
      FOUR_C_ASSERT_ALWAYS(!previous_history.empty(),
          "Missing FSLS history state in kernel evaluation (MAT {}, GP {}, ELE {}).",
          input.visco_mat_id, input.gp, input.ele_gid);

      FOUR_C_ASSERT_ALWAYS(input.gp >= 0 && input.gp < static_cast<int>(previous_history.size()),
          "Invalid Gauss point index GP={} for FSLS history in kernel evaluation (MAT {}, ELE "
          "{}). History container size is {}.",
          input.gp, input.visco_mat_id, input.ele_gid, previous_history.size());

      const auto& fsls_history_at_gp = previous_history.at(input.gp);
      FOUR_C_ASSERT_ALWAYS(!fsls_history_at_gp.empty(),
          "Invalid FSLS history size {} at GP {} in kernel evaluation (MAT {}, ELE {}). "
          "Expected at least one entry.",
          fsls_history_at_gp.size(), input.gp, input.visco_mat_id, input.ele_gid);

      return fsls_history_at_gp;
    }


    /// Grunwald-Letnikov discretization: full convolution of the artificial-stress history with
    /// recursive binomial-type coefficients (Adolfsson and Enelund 2003).
    void evaluate_fsls_kernel_grunwald_letnikov(const Kernels::FslsStressVector& stress,
        const Kernels::FslsTangentMatrix& cmat, Kernels::FslsStressVector& q_current_for_history,
        Kernels::FslsStressVector& q_additive, Kernels::FslsTangentMatrix& cmatq_additive,
        const Kernels::FslsKernelInput& input)
    {
      const auto& fsls_history_at_gp = require_fsls_history_at_gp(input);
      const int hs = static_cast<int>(fsls_history_at_gp.size());

      // calculate artificial history stress Qq with weights b_j
      // Qq = sum[j=1 up to j=n][b_j*Q_(n+1-j)] (short: b*Qj)
      // b_j = (j-1-alpha)/j * b_(j-1), with b_0 = 1
      double bj = 1.;
      double fac = 1.;
      Kernels::FslsStressVector q_history_sum(Core::LinAlg::Initialization::zero);

      for (int j = 1; j <= hs; j++)
      {
        fac = (j - 1. - input.alpha) / j;
        bj = bj * fac;

        Kernels::FslsStressVector qj(fsls_history_at_gp.at(hs - j));
        q_history_sum.update(bj, qj, 1.0);
      }

      const double dtalpha = std::pow(input.dt, input.alpha);
      const double taualpha = std::pow(input.tau, input.alpha);
      const double denominator = dtalpha + taualpha;
      FOUR_C_ASSERT_ALWAYS(denominator > 0.0,
          "Invalid FSLS update denominator dt^alpha + tau^alpha = {} in kernel evaluation (MAT "
          "{}, GP {}, ELE {}): dt={}, tau={}, alpha={}. Expected a positive denominator.",
          denominator, input.visco_mat_id, input.gp, input.ele_gid, input.dt, input.tau,
          input.alpha);

      const double lambda_1 = dtalpha / denominator;
      const double lambda_2 = -1. * taualpha / denominator;

      q_current_for_history.update(lambda_1 * input.beta, stress, 0.);
      q_current_for_history.update(lambda_2, q_history_sum, 1.);

      q_additive.update(1.0, q_current_for_history, 0.0);
      q_additive.update(input.beta, stress, -1.);

      cmatq_additive.update(lambda_1 * input.beta, cmat, 0.);
      cmatq_additive.update(input.beta, cmat, -1.);
    }


    /// K_k^h = c * b_k with c = (tau/h)^alpha / Gamma(2-alpha) and
    /// b_k = k^(1-alpha) * expm1((1-alpha) * log1p(1/k)) for k >= 1, b_0 = 1.
    /// The expm1/log1p formulation avoids the catastrophic cancellation that direct evaluation of
    /// (k+1)^(1-alpha) - k^(1-alpha) suffers from for large k.
    double fsls_l1_weight_from_prefactor(const std::size_t k, const double alpha, const double c)
    {
      if (k == 0) return c;

      const double p = 1.0 - alpha;
      const double kd = static_cast<double>(k);
      const double bk = std::pow(kd, p) * std::expm1(p * std::log1p(1.0 / kd));
      return c * bk;
    }


    /// c = (tau/h)^alpha / Gamma(2-alpha), evaluated in log space to avoid overflow/underflow of
    /// the power for extreme tau/h ratios. alpha == 0 is special-cased to the exact value 1 so
    /// that a degenerate tau/h (e.g. from an extreme time step) can never produce a 0 * inf = NaN
    /// in alpha * log(tau/h).
    double fsls_l1_prefactor(const double alpha, const double tau, const double dt)
    {
      if (alpha == 0.0) return 1.0;

      const double log_c = alpha * std::log(tau / dt) - std::lgamma(2.0 - alpha);
      return std::exp(log_c);
    }


    /// Returns K_k^h from the contribution-wide cache, lazily computing and appending any weights
    /// up to index k that have not yet been cached. The L1 scheme is only valid on a uniform time
    /// grid, so a change of dt after the cache has been established is a hard user-facing error
    /// rather than a silent rebuild.
    double fsls_l1_weight_cached(
        FslsL1WeightCache& cache, const Kernels::FslsKernelInput& input, const std::size_t k)
    {
      const double alpha = input.alpha;
      const double tau = input.tau;
      const double dt = input.dt;

      if (!cache.valid)
      {
        cache.alpha = alpha;
        cache.tau = tau;
        cache.dt = dt;
        cache.valid = true;
      }
      else
      {
        constexpr double relative_grid_tolerance = 1.0e-12;
        const bool grid_matches =
            cache.alpha == alpha && cache.tau == tau &&
            std::abs(cache.dt - dt) <= relative_grid_tolerance * std::max(cache.dt, dt);
        FOUR_C_ASSERT_ALWAYS(grid_matches,
            "FSLS L1 scheme requires a uniform time step, but the time step size changed from "
            "{} to {} (MAT {}, GP {}, ELE {}). A variable-step L1 scheme is a different, "
            "unimplemented method; keep TIMESTEP constant while VISCO_FSLS uses SOLVE l1.",
            cache.dt, dt, input.visco_mat_id, input.gp, input.ele_gid);
      }

      if (k >= cache.weights.size())
      {
        const double c = fsls_l1_prefactor(alpha, tau, dt);
        cache.weights.reserve(k + 1);
        for (std::size_t j = cache.weights.size(); j <= k; ++j)
          cache.weights.push_back(fsls_l1_weight_from_prefactor(j, alpha, c));
      }

      return cache.weights[k];
    }


    /// Uniform-grid L1 discretization: closed-form affine update of the artificial stress,
    /// obtained by splitting the L1 convolution sum into a committed history remainder r_n^Q and
    /// a term linear in the current trial equilibrium stress (formula and caching rationale: see
    /// FslsL1StepCache).
    void evaluate_fsls_kernel_l1(const Kernels::FslsStressVector& stress,
        const Kernels::FslsTangentMatrix& cmat, Kernels::FslsStressVector& q_current_for_history,
        Kernels::FslsStressVector& q_additive, Kernels::FslsTangentMatrix& cmatq_additive,
        const Kernels::FslsKernelInput& input)
    {
      const auto& fsls_history_at_gp = require_fsls_history_at_gp(input);
      FOUR_C_ASSERT_ALWAYS(input.l1.weights != nullptr && input.l1.step != nullptr,
          "Missing FSLS L1 cache state in kernel evaluation (MAT {}, GP {}, ELE {}).",
          input.visco_mat_id, input.gp, input.ele_gid);

      // History holds committed values Q_0 .. Q_n (hs = n+1 entries); we are solving for Q_{n+1}.
      const std::size_t hs = fsls_history_at_gp.size();
      const std::size_t n = hs - 1;

      // K_0 is looked up (and, on a uniform grid, always re-validated against alpha/tau/dt) on
      // every call, independent of whether the r_n^Q cache below is reused: this is what
      // guarantees a dt change is still caught even when the history length happens to be
      // unchanged since the last call.
      const double k0 = fsls_l1_weight_cached(*input.l1.weights, input, 0);

      FslsL1StepCache& step_cache = *input.l1.step;
      if (!step_cache.valid || step_cache.history_size != hs)
      {
        // r_n^Q accumulation (see FslsL1StepCache for the formula).
        Kernels::FslsStressVector r_q(fsls_history_at_gp.at(n));
        r_q.scale(-k0);

        Kernels::FslsStressVector delta_q(Core::LinAlg::Initialization::zero);
        for (std::size_t k = 1; k <= n; ++k)
        {
          const double kk = fsls_l1_weight_cached(*input.l1.weights, input, k);

          delta_q.update(1.0, fsls_history_at_gp.at(n - k + 1), -1.0, fsls_history_at_gp.at(n - k));
          r_q.update(kk, delta_q, 1.0);
        }

        step_cache.r_q = r_q;
        step_cache.history_size = hs;
        step_cache.valid = true;
      }

      const Kernels::FslsStressVector& r_q = step_cache.r_q;

      const double denominator = 1.0 + k0;
      FOUR_C_ASSERT_ALWAYS(denominator > 0.0,
          "Invalid FSLS L1 update denominator 1 + K_0 = {} in kernel evaluation (MAT {}, GP {}, "
          "ELE {}). Expected a positive denominator.",
          denominator, input.visco_mat_id, input.gp, input.ele_gid);

      // Q_{n+1} = (beta * S_inf - r_n^Q) / (1 + K_0)
      Kernels::FslsStressVector q_next(Core::LinAlg::Initialization::zero);
      q_next.update(input.beta, stress, -1.0, r_q);
      q_next.scale(1.0 / denominator);

      q_current_for_history.update(q_next);

      // S_eff = (1+beta) * S_inf - Q  =>  additive = S_eff - S_inf = beta*S_inf - Q
      q_additive.update(1.0, q_next, 0.0);
      q_additive.update(input.beta, stress, -1.0);

      // C_eff = [(1+beta) - beta/(1+K0)] * C_inf  =>  additive = beta*K0/(1+K0) * C_inf
      const double tangent_factor = input.beta * k0 / denominator;
      cmatq_additive.update(tangent_factor, cmat);
    }
  }  // namespace


  void Kernels::evaluate_fsls_kernel(const FslsStressVector& stress, const FslsTangentMatrix& cmat,
      FslsStressVector& q_current_for_history, FslsStressVector& q_additive,
      FslsTangentMatrix& cmatq_additive, const FslsKernelInput& input)
  {
    switch (input.solve_kind)
    {
      case FslsSolveKind::grunwald_letnikov:
        evaluate_fsls_kernel_grunwald_letnikov(
            stress, cmat, q_current_for_history, q_additive, cmatq_additive, input);
        return;
      case FslsSolveKind::l1:
        evaluate_fsls_kernel_l1(
            stress, cmat, q_current_for_history, q_additive, cmatq_additive, input);
        return;
    }

    FOUR_C_THROW("Unhandled FSLS scheme in kernel evaluation (MAT {}, GP {}, ELE {}).",
        input.visco_mat_id, input.gp, input.ele_gid);
  }


  double Kernels::evaluate_l1_weight(
      const std::size_t k, const double alpha, const double tau, const double dt)
  {
    FOUR_C_ASSERT_ALWAYS(dt > 0.0,
        "Invalid time step size dt={} in FSLS L1 weight evaluation. Expected dt > 0.", dt);
    FOUR_C_ASSERT_ALWAYS(
        tau > 0.0, "Invalid TAU={} in FSLS L1 weight evaluation. Expected TAU > 0.", tau);
    FOUR_C_ASSERT_ALWAYS(alpha >= 0.0 && alpha < 1.0,
        "Invalid ALPHA={} in FSLS L1 weight evaluation. Expected 0 <= ALPHA < 1.", alpha);

    return fsls_l1_weight_from_prefactor(k, alpha, fsls_l1_prefactor(alpha, tau, dt));
  }

}  // namespace Mat::ViscoElast

FOUR_C_NAMESPACE_CLOSE
