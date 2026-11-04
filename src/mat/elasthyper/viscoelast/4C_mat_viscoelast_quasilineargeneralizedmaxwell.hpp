// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_VISCOELAST_QUASILINEARGENERALIZEDMAXWELL_HPP
#define FOUR_C_MAT_VISCOELAST_QUASILINEARGENERALIZEDMAXWELL_HPP

#include "4C_config.hpp"

#include "4C_mat_viscoelast_contribution.hpp"
#include "4C_mat_viscoelast_summand.hpp"
#include "4C_material_parameter_base.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Mat::Elastic
{
  class Summand;
}

namespace Mat::ViscoElast
{
  namespace PAR
  {
    /**
     * \brief Parameters for quasi-linear Fung-type generalized Maxwell viscoelasticity.
     *
     * The model evaluates the isotropic stress and tangent of the parallel hyperelastic base law
     * and uses the resulting second Piola-Kirchhoff stress \f$S_0\f$ and material tangent
     * \f$C_0\f$ as the common branch driver. Each Maxwell branch has the elastic stress
     * \f$S^e_i = \beta_i S_0\f$ and evolves according to
     * \f$\dot{Q}_i + Q_i / \tau_i = \dot{S}^e_i\f$. The total viscoelastic stress contribution is
     * \f$S_\mathrm{vis} = \sum_i Q_i + S_\eta\f$, where the optional parallel dashpot uses
     * \f$S_\eta = \eta \dot{E}\f$ with the Green-Lagrange strain \f$E\f$.
     *
     * In contrast to VISCO_GeneralizedMaxwell, this material does not reference separate branch
     * materials. All branches are scalar multiples of the one hyperelastic base response of the
     * surrounding MAT_ViscoElastHyper material.
     */
    class QuasiLinearGeneralizedMaxwell : public Core::Mat::PAR::Parameter
    {
     public:
      explicit QuasiLinearGeneralizedMaxwell(const Core::Mat::PAR::Parameter::Data& matdata);

      /// Relative branch weights.
      const std::vector<double> beta_;
      /// Positive branch relaxation times.
      const std::vector<double> tau_;
      /// Time integration rule for branch evolution.
      const std::string solve_;
      /// Parallel dashpot viscosity acting on Green-Lagrange strain rate.
      const double viscosity_;

      std::shared_ptr<Core::Mat::Material> create_material() override
      {
        FOUR_C_THROW(
            "Cannot create a material from this method, as it should be created in "
            "Mat::ViscoElast::Summand::Factory.");
        return nullptr;
      };
    };
  }  // namespace PAR


  /**
   * \brief Parameter-backed summand for quasi-linear generalized Maxwell viscoelasticity.
   */
  class QuasiLinearGeneralizedMaxwell : public Summand
  {
   public:
    explicit QuasiLinearGeneralizedMaxwell(PAR::QuasiLinearGeneralizedMaxwell* params);

    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::mes_quasilineargeneralizedmaxwell;
    }

    /// Access immutable input parameters owned by the matching parameter object.
    [[nodiscard]] const PAR::QuasiLinearGeneralizedMaxwell& parameters() const;

    void specify_formulation(
        bool& isoprinc, bool& isomod, bool& anisoprinc, bool& anisomod, bool& viscogeneral) override
    {
      (void)isoprinc;
      (void)isomod;
      (void)anisoprinc;
      (void)anisomod;
      viscogeneral = true;
    }

    void specify_visco_formulation(bool& visco_iso_rate, bool& visco_generalized_maxwell,
        bool& visco_quasi_linear_generalized_maxwell, bool& visco_fsls) override
    {
      (void)visco_iso_rate;
      (void)visco_generalized_maxwell;
      (void)visco_fsls;
      visco_quasi_linear_generalized_maxwell = true;
    }

   private:
    PAR::QuasiLinearGeneralizedMaxwell* params_;
  };


  /// Setup data needed only by the QLV model because its branches use the elastic base law.
  struct QuasiLinearGeneralizedMaxwellSetupContext
  {
    /// Generic setup data shared by all viscoelastic contributions.
    ContributionSetupContext base;
    /// Hyperelastic base summands that provide \f$S_0\f$ and \f$C_0\f$.
    const std::vector<std::shared_ptr<Mat::Elastic::Summand>>& elastic_summands;
    /// Formulation flags for the hyperelastic base summands.
    const SummandProperties& elastic_summand_properties;
  };


  /**
   * \brief Contribution implementation for Fung-type quasi-linear generalized Maxwell response.
   */
  class QuasiLinearGeneralizedMaxwellContribution final : public Contribution
  {
   public:
    [[nodiscard]] ViscoModelKind kind() const override
    {
      return ViscoModelKind::quasi_linear_generalized_maxwell;
    }

    /// Reject generic setup because QLV also needs the elastic base-law setup data.
    void setup(const ContributionSetupContext& context) override;
    /// Cache material parameters and elastic base-law data for later Gauss-point evaluations.
    void setup(const QuasiLinearGeneralizedMaxwellSetupContext& context);
    /// Add the QLV branch and optional dashpot stress/tangent to the current evaluation result.
    void evaluate(const QuasiLinearGeneralizedMaxwellEvaluateContext& context);
    /// QLV history rotation is handled by ViscoElastState, so no contribution-local update is
    /// needed.
    void update(const ContributionUpdateContext& context) override;
    /// Number of Maxwell branch history entries required per Gauss point.
    [[nodiscard]] std::size_t history_entry_count_for_setup() const override;

   private:
    enum class SolveKind
    {
      one_step_theta,
      exponential_time_discretization
    };

    /// Setup metadata shared by all Gauss-point evaluations of this contribution.
    struct Metadata
    {
      /// Material id of the VISCO_QuasiLinearGeneralizedMaxwell summand.
      int summand_mat_id = -1;
      /// Time integration rule selected by SOLVE.
      SolveKind solve_kind = SolveKind::exponential_time_discretization;
      /// Elastic base summands whose isotropic stress and tangent drive every branch.
      std::vector<std::shared_ptr<Elastic::Summand>> base_summands;
      /// Formulation flags for the base summands.
      SummandProperties base_properties;
      /// Relative branch spring weights.
      std::vector<double> beta;
      /// Positive branch relaxation times.
      std::vector<double> tau;
      /// Parallel Green-Lagrange strain-rate dashpot viscosity.
      double viscosity = 0.0;
    };

    /// Runtime parameters read from the structural dynamic settings.
    struct RuntimeContext
    {
      /// Structural one-step-theta value used when SOLVE is OneStepTheta.
      double one_step_theta = 0.5;
    };

    /// Convert the SOLVE input string into the internal integration-rule enum.
    [[nodiscard]] static SolveKind parse_solve_kind(
        const std::string& solve, int visco_mat_id, int gp, int ele_gid);
    /// Return cached setup metadata, failing if setup has not run.
    [[nodiscard]] const Metadata& require_metadata(const char* context, int gp, int ele_gid) const;
    /// Return cached runtime parameters, failing if setup has not run.
    [[nodiscard]] const RuntimeContext& require_runtime_context(
        const char* context, int gp, int ele_gid) const;

    /// Build the setup metadata from material input and elastic base-law data.
    void build_metadata(const QuasiLinearGeneralizedMaxwellSetupContext& context);
    /// Read time-integration parameters from the structural dynamic parameter list.
    void build_runtime_context(const ContributionSetupContext& context);

    /// Cached material and elastic base-law setup data.
    std::optional<Metadata> metadata_;
    /// Cached structural time-integration data.
    std::optional<RuntimeContext> runtime_context_;
  };
}  // namespace Mat::ViscoElast

FOUR_C_NAMESPACE_CLOSE

#endif
