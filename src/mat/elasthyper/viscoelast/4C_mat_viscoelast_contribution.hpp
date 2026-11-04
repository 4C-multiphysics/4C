// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_VISCOELAST_CONTRIBUTION_HPP
#define FOUR_C_MAT_VISCOELAST_CONTRIBUTION_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_mat_elasthyper_service.hpp"
#include "4C_mat_viscoelast_state.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace Teuchos
{
  class ParameterList;
}

FOUR_C_NAMESPACE_OPEN

namespace Mat::ViscoElast
{
  class Summand;

  /// Identifies the viscoelastic model families orchestrated by Mat::ViscoElastHyper.
  enum class ViscoModelKind
  {
    iso_rate,
    generalized_maxwell,
    quasi_linear_generalized_maxwell,
    fsls
  };

  /**
   * \brief Location and material identifier shared by contribution operations.
   *
   * The point fields are diagnostic context. During material-wide setup/update calls, a value of
   * -1 denotes that the operation is not tied to a concrete Gauss point or element.
   */
  struct ContributionPointContext
  {
    int visco_mat_id = -1;
    int gp = -1;
    int ele_gid = -1;
  };

  /**
   * \brief Data available while a model contribution builds setup and runtime metadata.
   *
   * Contributions inspect the already constructed visco summands, select their own summand type,
   * validate model-specific assumptions, and cache metadata that is independent of the current
   * deformation state. The owning Mat::ViscoElastHyper remains responsible for constructing and
   * ordering the contribution objects.
   */
  struct ContributionSetupContext
  {
    ContributionPointContext point;
    const std::vector<std::shared_ptr<Summand>>& visco_summands;
    const std::vector<int>& visco_summand_mat_ids;
    ViscoElastState::ActiveModels active_models;
  };

  /// Data available during model-wide update hooks before history state is advanced.
  struct ContributionUpdateContext
  {
    ContributionPointContext point;
    ViscoElastState& state;
  };

  /**
   * \brief Common mutable evaluation state passed to each active contribution.
   *
   * The stress and tangent references are additive accumulators. Each contribution adds its
   * viscous response to the elastic response prepared by Mat::ViscoElastHyper and stores the
   * current internal variables in ViscoElastState.
   */
  struct ContributionEvaluateBase
  {
    ContributionPointContext point;
    double dt = 0.0;
    ViscoElastState& state;
    Core::LinAlg::Matrix<6, 1>& stress;
    Core::LinAlg::Matrix<6, 6>& cmat;
  };

  /**
   * \brief Evaluation workspace for iso-rate style viscous summands.
   *
   * This context intentionally exposes the hyperelastic invariant work arrays used by the iso-rate
   * coefficient hooks. The arrays are owned by Mat::ViscoElastHyper::EvaluateWorkspace and
   * reused across the active contribution sequence for the current Gauss point.
   */
  struct IsoRateEvaluateContext
  {
    ContributionEvaluateBase base;
    const Teuchos::ParameterList& params;
    const std::vector<std::shared_ptr<Summand>>& visco_summands;
    const SummandProperties& effective_properties;
    Core::LinAlg::Matrix<6, 1>& c_strain;
    Core::LinAlg::Matrix<6, 1>& c_stress;
    Core::LinAlg::Matrix<6, 1>& i_c_stress;
    Core::LinAlg::Matrix<3, 1>& prinv;
    Core::LinAlg::Matrix<3, 1>& modinv;
    Core::LinAlg::Matrix<7, 1>& rateinv;
    Core::LinAlg::Matrix<7, 1>& modrateinv;
    Core::LinAlg::Matrix<6, 1>& mod_c_strain;
    Core::LinAlg::Matrix<6, 1>& scgrate;
    Core::LinAlg::Matrix<6, 1>& modrcgrate;
    Core::LinAlg::Matrix<8, 1>& mu;
    Core::LinAlg::Matrix<8, 1>& modmu;
    Core::LinAlg::Matrix<33, 1>& xi;
    Core::LinAlg::Matrix<33, 1>& modxi;
    Core::LinAlg::Matrix<6, 1>& id2;
    Core::LinAlg::Matrix<6, 6>& id4;
    Core::LinAlg::Matrix<6, 6>& id4sharp;
  };

  /// Evaluation data needed by the generalized Maxwell contribution.
  struct GeneralizedMaxwellEvaluateContext
  {
    ContributionEvaluateBase base;
    const Core::LinAlg::Matrix<6, 1>& glstrain_mat;
  };

  /// Evaluation data needed by the quasi-linear generalized Maxwell contribution.
  struct QuasiLinearGeneralizedMaxwellEvaluateContext
  {
    ContributionEvaluateBase base;
    const Core::LinAlg::Matrix<6, 1>& glstrain_mat;
  };

  /// Evaluation data needed by the FSLS contribution.
  struct FslsEvaluateContext
  {
    ContributionEvaluateBase base;
  };

  /**
   * \brief Base interface for one active viscoelastic model contribution.
   *
   * Contributions encapsulate model-specific setup metadata, runtime context, and history-size
   * requirements. The owning material keeps the active ordering explicit and dispatches typed
   * evaluation contexts so each model receives only the data it needs.
   */
  class Contribution
  {
   public:
    virtual ~Contribution() = default;
    [[nodiscard]] virtual ViscoModelKind kind() const = 0;
    virtual void setup(const ContributionSetupContext& context) = 0;
    virtual void update(const ContributionUpdateContext& context) = 0;
    [[nodiscard]] virtual std::size_t history_entry_count_for_setup() const { return 0; }
    [[nodiscard]] virtual unsigned int history_capacity_for_update() const { return 0; }
  };
}  // namespace Mat::ViscoElast

FOUR_C_NAMESPACE_CLOSE

#endif
