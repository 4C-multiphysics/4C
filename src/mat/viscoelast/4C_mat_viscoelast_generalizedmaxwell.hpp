// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_VISCOELAST_GENERALIZEDMAXWELL_HPP
#define FOUR_C_MAT_VISCOELAST_GENERALIZEDMAXWELL_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_mat_elast_summand.hpp"
#include "4C_mat_viscoelast_contribution.hpp"
#include "4C_mat_viscoelast_summand.hpp"
#include "4C_material_parameter_base.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Mat::ViscoElast
{
  /**
   * \brief Contribution implementation for the generalized Maxwell visco model.
   *
   * Setup resolves the top-level generalized Maxwell summand, validates its branch materials, and
   * caches branch elastic summands, branch relaxation times, solve kind, and runtime integration
   * data. Evaluation computes the elastic response of each branch and advances the branch stress
   * history with the selected time integration rule.
   */
  class GeneralizedMaxwellContribution final : public Contribution
  {
   public:
    [[nodiscard]] ViscoModelKind kind() const override
    {
      return ViscoModelKind::generalized_maxwell;
    }

    void setup(const ContributionSetupContext& context) override;
    void evaluate(const GeneralizedMaxwellEvaluateContext& context);
    void update(const ContributionUpdateContext& context) override;
    [[nodiscard]] std::size_t history_entry_count_for_setup() const override;

   private:
    /// Cached data for one viscoelastic branch.
    struct BranchMetadata
    {
      /// Elastic summands that define this branch response.
      std::vector<std::shared_ptr<Elastic::Summand>> summands;
      /// Positive branch relaxation time.
      double tau = 0.0;
      /// Formulation flags of the branch elastic summands.
      SummandProperties properties;
    };

    enum class SolveKind
    {
      one_step_theta,
      exponential_time_discretization
    };

    /// Setup metadata shared by all Gauss-point evaluations of this contribution.
    struct Metadata
    {
      int summand_mat_id = -1;
      SolveKind solve_kind = SolveKind::exponential_time_discretization;
      std::vector<BranchMetadata> branches;
    };

    /// Runtime parameters read from the structural dynamic settings.
    struct RuntimeContext
    {
      double one_step_theta = 0.5;
    };

    [[nodiscard]] static SolveKind parse_solve_kind(
        const std::string& solve, int visco_mat_id, int gp, int ele_gid);
    [[nodiscard]] const Metadata& require_metadata(const char* context, int gp, int ele_gid) const;
    [[nodiscard]] const RuntimeContext& require_runtime_context(
        const char* context, int gp, int ele_gid) const;

    void build_metadata(const ContributionSetupContext& context);
    void build_runtime_context(const ContributionSetupContext& context);
    void evaluate_branch_material_response(const GeneralizedMaxwellEvaluateContext& context,
        const Metadata& metadata, double one_step_theta) const;

    std::optional<Metadata> metadata_;
    std::optional<RuntimeContext> runtime_context_;
  };

  namespace PAR
  {
    /*!
     * @brief Parameters for the top-level generalized Maxwell visco contribution.
     *
     * The material IDs stored in this parameter object reference
     * PAR::ViscoBranch entries. Each branch entry then references the elastic material that
     * defines the branch stiffness. The contribution setup caches the resolved branch data before
     * Gauss-point evaluation starts.
     */
    class GeneralizedMaxwell : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      GeneralizedMaxwell(const Core::Mat::PAR::Parameter::Data& matdata);

      /// @name material parameters
      //@{

      /// Number of viscoelastic branches.
      int numbranch_;
      /// Material IDs of PAR::ViscoBranch entries.
      const std::vector<int> matids_;
      /// Time integration rule for branch evolution.
      std::string solve_;
      //@}

      /// create material instance of matching type with my parameters

      std::shared_ptr<Core::Mat::Material> create_material() override { return nullptr; };
    };  // class GeneralizedMaxwell


    /*!
     * @brief Parameters for one branch of a generalized Maxwell contribution.
     *
     * A branch stores its relaxation time and the material ID of the elastic summand used as branch
     * stiffness. Branch materials are resolved through the generalized Maxwell contribution and are
     * not active top-level contributions on their own.
     */
    class ViscoBranch : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      ViscoBranch(const Core::Mat::PAR::Parameter::Data& matdata);

      /// @name material parameters
      //@{

      /// Positive branch relaxation time.
      double tau_;
      /// Material ID of the elastic summand that defines branch stiffness.
      int matid_;


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
    };  // class ViscoBranch

  }  // namespace PAR

  /**
   * \brief Parameter-backed summand for the top-level generalized Maxwell model.
   *
   * This adapter resolves the branch parameter objects and their elastic summands. The actual
   * stress/tangent contribution is implemented in GeneralizedMaxwellContribution, which consumes
   * the cached branch metadata during evaluation.
   */
  class GeneralizedMaxwell : public Summand
  {
   public:
    /// constructor with given material parameters
    GeneralizedMaxwell(PAR::GeneralizedMaxwell* params);

    /// @name Access material constants
    //@{

    /// material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::mes_generalizedmaxwell;
    }
    //@}

    /// Read material parameters
    void read_material_parameters(int& numbranch,  ///< number of viscoelastic branches
        const std::vector<int>*& matids,           ///< material IDs of the viscoelastic branches
        std::string& solve  /// variant of the solution of the evolution integral
        ) override;

    /// @name Resolved branch data used by GeneralizedMaxwellContribution
    //@{
    const std::vector<std::vector<std::shared_ptr<Elastic::Summand>>>& get_branchespotsum() const
    {
      return branchespotsum_;
    }

    const std::vector<double>& get_branchtaus() const { return branchtau_; }
    //@}

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
      visco_generalized_maxwell = true;
      return;
    };


   private:
    /// my material parameters
    PAR::GeneralizedMaxwell* params_;

   protected:
    /// Elastic summands of each generalized Maxwell branch.
    std::vector<std::vector<std::shared_ptr<Elastic::Summand>>> branchespotsum_;
    /// Relaxation time of each generalized Maxwell branch.
    std::vector<double> branchtau_;
    /// Temporary storage used while resolving one branch.
    std::vector<std::shared_ptr<Elastic::Summand>> internalpotsum_;
  };

  /// Parameter-backed summand representing one generalized Maxwell branch definition.
  class ViscoBranch : public Summand
  {
   public:
    /// constructor with given material parameters
    ViscoBranch(PAR::ViscoBranch* params);

    /// @name Access material constants
    //@{

    /// material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::mes_viscobranch;
    }

    //@}

    /// Read material parameters
    void read_material_parameters(double& tau,  ///< branch relaxation time
        int& matid                              ///< material ID of branch elasticity
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


   private:
    /// my material parameters
    PAR::ViscoBranch* params_;

  };  // class ViscoBranch


  namespace Kernels
  {
    /// Branch stress integration rule used by the pure generalized Maxwell kernel.
    enum class GeneralizedMaxwellSolveKind
    {
      one_step_theta,
      exponential_time_discretization
    };

    using StressVector = Core::LinAlg::Matrix<6, 1>;
    using TangentMatrix = Core::LinAlg::Matrix<6, 6>;
    using PointHistory = std::vector<StressVector>;

    /// Input data that is independent of branch elastic response evaluation.
    struct GeneralizedMaxwellKernelInput
    {
      int visco_mat_id = -1;
      int gp = -1;
      int ele_gid = -1;
      double dt = 0.0;
      double one_step_theta = 0.5;
      GeneralizedMaxwellSolveKind solve_kind =
          GeneralizedMaxwellSolveKind::exponential_time_discretization;
      const PointHistory* previous_branch_elastic_stress = nullptr;
      const PointHistory* previous_branch_stress = nullptr;
    };

    /// Callback that computes the current elastic response of one branch.
    using BranchResponseEvaluator = std::function<void(
        int branch_index, StressVector& branch_elastic_stress, TangentMatrix& branch_cmat)>;

    /// Evaluate generalized Maxwell branch stresses and additive tangent contribution.
    void evaluate_generalized_maxwell_kernel(StressVector& q_total, TangentMatrix& cmatq_total,
        PointHistory& current_branch_elastic_stress, PointHistory& current_branch_stress,
        const std::vector<double>& branch_taus, const GeneralizedMaxwellKernelInput& input,
        const BranchResponseEvaluator& evaluate_branch_response);

  }  // namespace Kernels
}  // namespace Mat::ViscoElast

FOUR_C_NAMESPACE_CLOSE

#endif
