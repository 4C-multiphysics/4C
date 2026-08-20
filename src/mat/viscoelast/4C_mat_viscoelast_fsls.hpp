// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_VISCOELAST_FSLS_HPP
#define FOUR_C_MAT_VISCOELAST_FSLS_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_mat_viscoelast_contribution.hpp"
#include "4C_mat_viscoelast_summand.hpp"
#include "4C_material_parameter_base.hpp"

#include <optional>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Mat::ViscoElast
{
  /**
   * \brief Cache of L1-scheme fractional-derivative weights shared by all Gauss points of one
   * FSLS contribution instance.
   *
   * The weights \f$K_k^h\f$ depend only on (alpha, tau, dt), not on Gauss point or history
   * content, so they are computed once per contribution and reused/extended as the history grows.
   * The L1 scheme requires a uniform time step; if dt changes between evaluations the cache is
   * invalidated and rebuilt from scratch rather than silently mixing weights from two different
   * grids. See FslsL1StepCache for the L1 discretization this cache feeds into, and
   * Kernels::evaluate_l1_weight for the formula and its numerically robust evaluation.
   */
  struct FslsL1WeightCache
  {
    /// weights[k] holds K_k^h, extended lazily as longer histories are encountered.
    std::vector<double> weights;
    double alpha = 0.0;
    double tau = 0.0;
    double dt = 0.0;
    bool valid = false;
  };

  /**
   * \brief Per-Gauss-point cache of the L1 scheme's committed-history remainder r_n^Q.
   *
   * The FSLS evolution equation
   * \f[
   *   \tau^\alpha \, {}^C\!D_t^\alpha \hat Q + \hat Q = \beta S^\infty, \qquad
   *   S^{\mathrm{eff}} = (1+\beta) S^\infty - \hat Q,
   * \f]
   * is discretized on a uniform time grid of step \f$h\f$ by the L1 scheme. With the fractional
   * weights \f$K_k^h\f$ (see Kernels::evaluate_l1_weight for the formula) and committed increments
   * \f$\Delta\hat Q_j = \hat Q_{j+1} - \hat Q_j\f$, the history remainder cached here is
   * \f[
   *   r_n^Q = -K_0^h \hat Q_n + \sum_{k=1}^{n} K_k^h \, \Delta\hat Q_{n-k}.
   * \f]
   * This is exactly what lets the update be written as a closed-form, non-iterative affine
   * function of the current trial equilibrium stress \f$S^\infty_{n+1}\f$ alone:
   * \f[
   *   \hat Q_{n+1} = \frac{\beta S^\infty_{n+1} - r_n^Q}{1 + K_0^h}, \qquad
   *   \mathbb{C}^{\mathrm{eff}} = \left[(1+\beta) - \frac{\beta}{1+K_0^h}\right] \mathbb{C}^\infty.
   * \f]
   * (See Mat::ViscoElast::Kernels::evaluate_fsls_kernel for where these are actually assembled.)
   *
   * \f$r_n^Q\f$ depends only on committed history, never on the current Newton trial, so it needs
   * to be recomputed only once per converged time step and can be reused across every Newton
   * iteration within that step. Committed history strictly grows by exactly one entry per
   * converged step, so comparing the stored history size against the current one is a sufficient
   * staleness check; no separate step counter is needed.
   */
  struct FslsL1StepCache
  {
    /// History size (number of committed Q entries) this remainder was computed for.
    std::size_t history_size = 0;
    /// Cached r_n^Q.
    Core::LinAlg::Matrix<6, 1> r_q;
    bool valid = false;
  };

  /// Bundles the two caches the L1 scheme needs during one kernel evaluation: the
  /// contribution-wide weight cache and this Gauss point's history-remainder cache. Both are
  /// null together (Grunwald-Letnikov scheme) or both non-null together (L1 scheme).
  struct FslsL1State
  {
    FslsL1WeightCache* weights = nullptr;
    FslsL1StepCache* step = nullptr;
  };

  /// Time-discretization scheme for the FSLS fractional derivative, selected by the material's
  /// SOLVE parameter.
  enum class FslsSolveKind
  {
    grunwald_letnikov,
    l1
  };

  /**
   * \brief Contribution implementation for the fractional standard linear solid model.
   *
   * Setup reads the single active FSLS summand and caches material parameters plus the maximum
   * artificial-stress history capacity. Evaluation computes the additive FSLS stress/tangent and
   * writes the current artificial stress that will be appended to history during update.
   *
   * The fractional derivative can be discretized with two schemes, selected by the material's
   * SOLVE parameter: the original Grunwald-Letnikov convolution, or the uniform-grid L1 scheme (a
   * closed-form, non-iterative affine update of the artificial stress; see FslsL1StepCache for the
   * full derivation).
   */
  class FslsContribution final : public Contribution
  {
   public:
    [[nodiscard]] ViscoModelKind kind() const override { return ViscoModelKind::fsls; }
    void setup(const ContributionSetupContext& context) override;
    void evaluate(const FslsEvaluateContext& context);
    void update(const ContributionUpdateContext& context) override;
    [[nodiscard]] unsigned int history_capacity_for_update() const override;

   private:
    /// Material parameters cached during setup.
    struct Metadata
    {
      double tau = 0.0;
      double alpha = 0.0;
      double beta = 0.0;
      int summand_mat_id = -1;
      FslsSolveKind solve_kind = FslsSolveKind::grunwald_letnikov;
    };

    /// Runtime history settings derived from the structural dynamic parameters.
    struct RuntimeContext
    {
      unsigned int max_history_size = 0;
    };

    [[nodiscard]] static FslsSolveKind parse_solve_kind(
        const std::string& solve, int visco_mat_id, int gp, int ele_gid);
    [[nodiscard]] const Metadata& require_metadata(const char* context, int gp, int ele_gid) const;
    [[nodiscard]] const RuntimeContext& require_runtime_context(
        const char* context, int gp, int ele_gid) const;

    void build_metadata(const ContributionSetupContext& context);
    void build_runtime_context(const ContributionSetupContext& context);

    std::optional<Metadata> metadata_;
    std::optional<RuntimeContext> runtime_context_;
    /// L1 weight cache, shared across Gauss points of this contribution. Unused for the
    /// Grunwald-Letnikov scheme.
    FslsL1WeightCache l1_weight_cache_;
    /// L1 history-remainder cache, indexed by Gauss point and lazily grown as new Gauss points
    /// are first evaluated. Unused for the Grunwald-Letnikov scheme.
    std::vector<FslsL1StepCache> l1_step_cache_;
  };

  namespace PAR
  {
    /*!
     * @brief Parameters for the fractional standard linear solid visco contribution.
     *
     * The parameter object stores relaxation time, fractional order, and viscous weighting. It is
     * consumed by FslsContribution during setup and does not create a standalone material object.
     */
    class Fsls : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      Fsls(const Core::Mat::PAR::Parameter::Data& matdata);

      /// @name material parameters
      //@{

      /// Positive relaxation time.
      double tau_;
      /// Fractional derivative order in the interval [0, 1).
      double alpha_;
      /// Weighting of the viscous contribution relative to the elastic stress.
      double beta_;
      /// Time-discretization scheme for the fractional derivative: "GrunwaldLetnikov" (default)
      /// or "L1".
      std::string solve_;

      //@}

      /// Override this method and throw error, as the material should be created in within the
      /// Factory method of the elastic summand
      std::shared_ptr<Core::Mat::Material> create_material() override
      {
        FOUR_C_THROW(
            "Cannot create a material from this method, as it should be created in "
            "Mat::ViscoElast::Summand::Factory.");
        return nullptr;
      };
    };  // class Fsls
  }  // namespace PAR


  /*!
   * @brief Parameter-backed summand for the fractional standard linear solid model.
   *
   * The model consists of one spring in parallel to one sequential branch of a spring and a
   * springpot. Within Mat::ViscoElastHyper, this summand activates FslsContribution and supplies
   * the cached scalar parameters used by the FSLS kernel.
   *
   * A springpot is between a spring and a dashpot. The parameter alpha regulates
   * how much damping is introduced.
   * Alpha=0, means the springpot is a spring
   * Alpha=1, means the springpot is a dashpot; this is equal to a generalized Maxwell branch
   *
   * <h3>References</h3>
   * <ul>
   * <li> [1] Adolfson and Enelund (2003): Fractional Derivative Viscoelasticity at
   *          Large Deformations
   * </ul>
   */
  class Fsls : public Summand
  {
   public:
    /// constructor with given material parameters
    Fsls(PAR::Fsls* params);

    /// @name Access material constants
    //@{

    /// material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::mes_fsls;
    }

    //@}

    /// Read material parameters
    void read_material_parameters_visco(double& tau,  ///< relaxation parameter tau
        double& beta,                                 ///< emphasis of viscous to elastic part
        double& alpha,                                ///< fractional order derivative (for FSLS)
        std::string& solve  ///< fractional-derivative time discretization scheme (for FSLS)
        ) override;

    /// Indicator for formulation
    void specify_formulation(
        bool& isoprinc,     ///< global indicator for isotropic principal formulation
        bool& isomod,       ///< global indicator for isotropic split formulation
        bool& anisoprinc,   ///< global indicator for anisotropic principal formulation
        bool& anisomod,     ///< global indicator for anisotropic split formulation
        bool& viscogeneral  ///< general indicator, if one viscoelastic formulation is used
        ) override
    {
      viscogeneral = true;
      return;
    };

    /// Indicator for the chosen viscoelastic formulations
    void specify_visco_formulation(
        bool& visco_iso_rate,  ///< global indicator for isotropic rate-dependent visco response
        bool& visco_generalized_maxwell,  ///< global indicator for generalized Maxwell model
        bool& visco_quasi_linear_generalized_maxwell,  ///< global indicator for QLV Maxwell model
        bool& visco_fsls                               ///< global indicator for FSLS model
        ) override
    {
      visco_fsls = true;
      return;
    };

   private:
    /// my material parameters
    PAR::Fsls* params_;
  };

  namespace Kernels
  {
    /// Stress-like vector used by the FSLS kernel and history containers.
    using FslsStressVector = Core::LinAlg::Matrix<6, 1>;
    /// Tangent matrix used by the FSLS kernel.
    using FslsTangentMatrix = Core::LinAlg::Matrix<6, 6>;
    /// Artificial-stress history for one Gauss point, indexed by stored time level.
    using FslsPointHistory = std::vector<FslsStressVector>;
    /// Artificial-stress history indexed by Gauss point and stored time level.
    using FslsHistory = std::vector<FslsPointHistory>;

    /// Input data required by the pure FSLS kernel.
    struct FslsKernelInput
    {
      int visco_mat_id = -1;
      int gp = -1;
      int ele_gid = -1;
      double dt = 0.0;
      double tau = 0.0;
      double alpha = 0.0;
      double beta = 0.0;
      FslsSolveKind solve_kind = FslsSolveKind::grunwald_letnikov;
      const FslsHistory* previous_history = nullptr;
      /// L1 scheme caches. Both non-null when solve_kind == l1, both null otherwise.
      FslsL1State l1;
    };

    /// Evaluate FSLS artificial stress, additive stress contribution, and additive tangent.
    void evaluate_fsls_kernel(const FslsStressVector& stress, const FslsTangentMatrix& cmat,
        FslsStressVector& q_current_for_history, FslsStressVector& q_additive,
        FslsTangentMatrix& cmatq_additive, const FslsKernelInput& input);

    /// Robustly evaluate the L1 weight
    /// \f$K_k^h = (\tau/h)^\alpha / \Gamma(2-\alpha) \, [(k+1)^{1-\alpha} - k^{1-\alpha}]\f$
    /// using a log/expm1 formulation that avoids catastrophic cancellation of the bracket for
    /// large k and avoids overflow of the prefactor for extreme tau/h ratios. Exposed for
    /// testing; production evaluation should go through the cached lookup in
    /// evaluate_fsls_kernel.
    double evaluate_l1_weight(std::size_t k, double alpha, double tau, double dt);
  }  // namespace Kernels
}  // namespace Mat::ViscoElast

FOUR_C_NAMESPACE_CLOSE

#endif
