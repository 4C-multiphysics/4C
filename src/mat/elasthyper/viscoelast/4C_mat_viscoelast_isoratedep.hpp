// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_VISCOELAST_ISORATEDEP_HPP
#define FOUR_C_MAT_VISCOELAST_ISORATEDEP_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_mat_viscoelast_contribution.hpp"
#include "4C_mat_viscoelast_summand.hpp"
#include "4C_material_parameter_base.hpp"

#include <memory>
#include <vector>

namespace Teuchos
{
  class ParameterList;
}

FOUR_C_NAMESPACE_OPEN

namespace Mat::ViscoElast
{
  /**
   * \brief Contribution implementation for iso-rate style viscous summands.
   *
   * The contribution evaluates rate-dependent kinematics, lets active visco summands add their
   * invariant coefficients, and adds the resulting isotropic viscous stress/tangent response to the
   * material accumulators. It uses previous/current right Cauchy-Green history stored in
   * ViscoElastState.
   */
  class IsoRateContribution final : public Contribution
  {
   public:
    [[nodiscard]] ViscoModelKind kind() const override { return ViscoModelKind::iso_rate; }
    void setup(const ContributionSetupContext& context) override;
    void evaluate(const IsoRateEvaluateContext& context);
    void update(const ContributionUpdateContext& context) override;

   private:
    static void evaluate_kinematics(const IsoRateEvaluateContext& context);
    static void evaluate_mu_xi(const IsoRateEvaluateContext& context);
    static void add_stress_tangent(const IsoRateEvaluateContext& context);
  };

  namespace PAR
  {
    /*!
     * @brief Parameters for an iso-rate dependent viscous summand.
     *
     * The parameter object stores the viscosity-like coefficient used by IsoRateDep and is consumed
     * through the visco summand factory.
     */
    class IsoRateDep : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      IsoRateDep(const Core::Mat::PAR::Parameter::Data& matdata);

      /// @name material parameters
      //@{

      /// Viscosity-like coefficient of the iso-rate response.
      double n_;

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
    };  // class IsoRateDep

  }  // namespace PAR

  /*!
   * @brief Parameter-backed summand for an isochoric iso-rate viscous response.
   *
   * This summand contributes modified invariant coefficients to IsoRateContribution. The response
   * depends on the modified invariants of the rate of the right Cauchy-Green tensor.
   *
   * Strain energy function is given by
   * \f[
   *   \Psi = n \overline{J}_2 (\overline{I}_1 -3).
   * \f]
   * (n = \f$\eta\f$)
   */
  class IsoRateDep : public Summand
  {
   public:
    /// constructor with given material parameters
    IsoRateDep(PAR::IsoRateDep* params);

    /// @name Access material constants
    //@{

    /// material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::mes_isoratedep;
    }

    //@}

    /// Add modified coeffiencts.
    void add_coefficients_visco_modified(
        const Core::LinAlg::Matrix<3, 1>&
            modinv,                          ///< modified invariants of right Cauchy-Green tensor
        Core::LinAlg::Matrix<8, 1>& modmu,   ///< necessary coefficients for piola-kirchhoff-stress
        Core::LinAlg::Matrix<33, 1>& modxi,  ///< necessary coefficients for viscosity tensor
        Core::LinAlg::Matrix<7, 1>& modrateinv, const Teuchos::ParameterList& params, double dt,
        int gp, int eleGID) override;

    /// Indicator for formulation
    void specify_formulation(
        bool& isoprinc,     ///< global indicator for isotropic principal formulation
        bool& isomod,       ///< global indicator for isotropic split formulation
        bool& anisoprinc,   ///< global indicator for anisotropic principal formulation
        bool& anisomod,     ///< global indicator for anisotropic split formulation
        bool& viscogeneral  ///< general indicator, if one viscoelastic formulation is used
        ) override
    {
      isomod = true;
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
      visco_iso_rate = true;
      return;
    };

   private:
    /// my material parameters
    PAR::IsoRateDep* params_;
  };

  namespace Kernels
  {
    using Matrix61 = Core::LinAlg::Matrix<6, 1>;
    using Matrix66 = Core::LinAlg::Matrix<6, 6>;
    using Matrix31 = Core::LinAlg::Matrix<3, 1>;
    using Matrix71 = Core::LinAlg::Matrix<7, 1>;
    using Matrix81 = Core::LinAlg::Matrix<8, 1>;
    using Matrix331 = Core::LinAlg::Matrix<33, 1>;

    /// Evaluate iso-rate viscous coefficient arrays from active summands.
    void evaluate_mu_xi_kernel(const std::vector<std::shared_ptr<Summand>>& summands,
        bool isoprinc_active, bool isomod_active, Matrix31& prinv, Matrix31& modinv, Matrix81& mu,
        Matrix81& modmu, Matrix331& xi, Matrix331& modxi, Matrix71& rateinv, Matrix71& modrateinv,
        const Teuchos::ParameterList& params, double dt, int gp, int ele_gid);

    /// Evaluate current kinematic rates and modified-rate invariants for iso-rate models.
    void evaluate_kin_quant_vis_kernel(const Matrix61& rcg, const Matrix61& scg,
        const Matrix31& prinv, const Matrix61& scg_previous, const Matrix61& modrcg_previous,
        double dt, Matrix61& modrcg, Matrix61& scgrate, Matrix61& modrcgrate, Matrix71& modrateinv,
        int visco_mat_id, int gp);

    /// Add principal iso-rate stress and tangent contributions.
    void evaluate_iso_visco_principal_kernel(Matrix61& stress, Matrix66& cmat, const Matrix81& mu,
        const Matrix331& xi, const Matrix66& id4sharp, const Matrix61& scgrate);

    /// Add modified-invariant iso-rate stress and tangent contributions.
    void evaluate_iso_visco_modified_kernel(Matrix61& stressisomodisovisco,
        Matrix61& stressisomodvolvisco, Matrix66& cmatisomodisovisco, Matrix66& cmatisomodvolvisco,
        const Matrix31& prinv, const Matrix31& modinv, const Matrix81& modmu,
        const Matrix331& modxi, const Matrix61& rcg, const Matrix61& id2, const Matrix61& icg,
        const Matrix66& id4, const Matrix61& modrcgrate);
  }  // namespace Kernels
}  // namespace Mat::ViscoElast

FOUR_C_NAMESPACE_CLOSE

#endif
