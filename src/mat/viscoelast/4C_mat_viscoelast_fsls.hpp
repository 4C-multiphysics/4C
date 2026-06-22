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
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Mat::ViscoElast
{
  /**
   * \brief Contribution implementation for the fractional standard linear solid model.
   *
   * Setup reads the single active FSLS summand and caches material parameters plus the maximum
   * artificial-stress history capacity. Evaluation computes the additive FSLS stress/tangent and
   * writes the current artificial stress that will be appended to history during update.
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
    };

    /// Runtime history settings derived from the structural dynamic parameters.
    struct RuntimeContext
    {
      unsigned int max_history_size = 0;
    };

    [[nodiscard]] const Metadata& require_metadata(const char* context, int gp, int ele_gid) const;
    [[nodiscard]] const RuntimeContext& require_runtime_context(
        const char* context, int gp, int ele_gid) const;

    void build_metadata(const ContributionSetupContext& context);
    void build_runtime_context(const ContributionSetupContext& context);

    std::optional<Metadata> metadata_;
    std::optional<RuntimeContext> runtime_context_;
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
        std::string& solve  ///< unused for FSLS; kept for interface compatibility
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
    /// Artificial-stress history indexed by Gauss point and stored time level.
    using FslsHistory = std::vector<std::vector<FslsStressVector>>;

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
      const FslsHistory* previous_history = nullptr;
    };

    /// Evaluate FSLS artificial stress, additive stress contribution, and additive tangent.
    void evaluate_fsls_kernel(const FslsStressVector& stress, const FslsTangentMatrix& cmat,
        FslsStressVector& q_current_for_history, FslsStressVector& q_additive,
        FslsTangentMatrix& cmatq_additive, const FslsKernelInput& input);
  }  // namespace Kernels
}  // namespace Mat::ViscoElast

FOUR_C_NAMESPACE_CLOSE

#endif
