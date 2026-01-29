// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SSI_MONOLITHIC_CONVCHECK_STRATEGIES_HPP
#define FOUR_C_SSI_MONOLITHIC_CONVCHECK_STRATEGIES_HPP

#include "4C_config.hpp"

#include "4C_ssi_monolithic.hpp"

FOUR_C_NAMESPACE_OPEN

namespace SSI
{
  enum class L2norm : std::uint8_t
  {
    concdofnorm,
    concincnorm,
    concresnorm,
    manifoldpotdofnorm,
    manifoldpotincnorm,
    manifoldpotresnorm,
    manifoldconcdofnorm,
    manifoldconcincnorm,
    manifoldconcresnorm,
    potdofnorm,
    potincnorm,
    potresnorm,
    scatradofnorm,
    scatraincnorm,
    scatraresnorm,
    structuredofnorm,
    structureincnorm,
    structureresnorm
  };

  class SsiMono::ConvCheckStrategyBase
  {
   public:
    /**
     * Virtual destructor.
     */
    virtual ~ConvCheckStrategyBase() = default;

    explicit ConvCheckStrategyBase(const Teuchos::ParameterList& parameters);

    //! Check, if Newton-Raphson has converged and print residuals and increments to screen
    bool exit_newton_raphson(const SSI::SsiMono& ssi_mono);

    //! Check, if Newton-Raphson of initial potential calculation has converged and print residuals
    //! and increments to screen
    virtual bool exit_newton_raphson_init_pot_calc(const SSI::SsiMono& ssi_mono) = 0;

    //! print all time steps that have not been converged
    void print_non_converged_steps(int pid) const;

   protected:
    //! get L2 norms from structure field and check if they are reasonable
    //!
    //! \param ssi_mono  ssi time integration
    //! \param incnorm   (out) L2 norm of increment
    //! \param resnorm   (out) L2 norm of residual
    //! \param dofnorm   (out) L2 norm of displacement
    void get_and_check_l2_norm_structure(
        const SSI::SsiMono& ssi_mono, double& incnorm, double& resnorm, double& dofnorm) const;

    //! check, if L2 norm is inf or nan. For dofnom: check if it is numerical zero
    //!
    //! \param incnorm  (out) L2 norm of increment
    //! \param resnorm  (out) L2 norm of residual
    //! \param dofnorm  (out) L2 norm of state
    void check_l2_norm(const double& incnorm, const double& resnorm, double& dofnorm) const;

    //! decide, if Newton loop should be exited, if converged or maximum number of steps are reached
    //!
    //! \param ssi_mono   ssi time integration
    //! \param converged  convergence of Newton loop
    //! \return  decision on exit
    [[nodiscard]] bool compute_exit(const SSI::SsiMono& ssi_mono, bool converged) const;

    //! maximum number of Newton-Raphson iteration steps
    const int itermax_;

    //! relative tolerance for Newton-Raphson iteration
    const double itertol_;

    //! time steps that have not converged
    std::set<int> non_converged_steps_;

    //! absolute tolerance for residual vectors
    const double restol_;

   private:
    //! decide, if Newton loop has converged
    //!
    //! \param ssi_mono  ssi time integration
    //! \param norms     L2 norms of residual, increment, and state
    //! \return  decision on convergence
    [[nodiscard]] virtual bool check_convergence(
        const SSI::SsiMono& ssi_mono, const std::map<SSI::L2norm, double>& norms) const = 0;

    //! compute L2 norms and fill into a map
    //!
    //! \param ssi_mono   ssi time integration
    //! \return  map with string identifier of norm as key and its value
    [[nodiscard]] virtual std::map<SSI::L2norm, double> compute_norms(
        const SSI::SsiMono& ssi_mono) const = 0;

    //! print convergence table to screen
    //!
    //! \param ssi_mono   ssi time integration
    //! \param converged  convergence of Newton loop?
    //! \param exit       exit of Newton loop?
    //! \param norms      computed L2 norms
    virtual void print_newton_iteration_information(const SSI::SsiMono& ssi_mono, bool converged,
        bool exit, const std::map<L2norm, double>& norms) const = 0;
  };

  class SsiMono::ConvCheckStrategyStd : public SsiMono::ConvCheckStrategyBase
  {
   public:
    explicit ConvCheckStrategyStd(const Teuchos::ParameterList& parameters)
        : ConvCheckStrategyBase(parameters)
    {
    }

    bool exit_newton_raphson_init_pot_calc(const SSI::SsiMono& ssi_mono) override
    {
      FOUR_C_THROW("Calculation of initial potential only for Elch");
    }

   protected:
    //! get L2 norms from scatra field and check if they are reasonable
    void get_and_check_l2_norm_scatra(
        const SSI::SsiMono& ssi_mono, double& incnorm, double& resnorm, double& dofnorm) const;

   private:
    [[nodiscard]] bool check_convergence(
        const SSI::SsiMono& ssi_mono, const std::map<L2norm, double>& norms) const override;

    [[nodiscard]] std::map<SSI::L2norm, double> compute_norms(
        const SSI::SsiMono& ssi_mono) const override;

    void print_newton_iteration_information(const SSI::SsiMono& ssi_mono, bool converged, bool exit,
        const std::map<L2norm, double>& norms) const override;
  };


  class SsiMono::ConvCheckStrategyElch : public SsiMono::ConvCheckStrategyBase
  {
   public:
    explicit ConvCheckStrategyElch(const Teuchos::ParameterList& parameters)
        : ConvCheckStrategyBase(parameters)
    {
    }

    bool exit_newton_raphson_init_pot_calc(const SSI::SsiMono& ssi_mono) override;

   protected:
    //! get L2 norms from concentration field and check if they are reasonable
    void get_and_check_l2_norm_conc(
        const SSI::SsiMono& ssi_mono, double& incnorm, double& resnorm, double& dofnorm) const;

    //! get L2 norms from potential field and check if they are reasonable
    void get_and_check_l2_norm_pot(
        const SSI::SsiMono& ssi_mono, double& incnorm, double& resnorm, double& dofnorm) const;

   private:
    [[nodiscard]] bool check_convergence(
        const SSI::SsiMono& ssi_mono, const std::map<L2norm, double>& norms) const override;

    [[nodiscard]] std::map<SSI::L2norm, double> compute_norms(
        const SSI::SsiMono& ssi_mono) const override;

    void print_newton_iteration_information(const SSI::SsiMono& ssi_mono, bool converged, bool exit,
        const std::map<L2norm, double>& norms) const override;
  };

  class SsiMono::ConvCheckStrategyElchScaTraManifold : public SsiMono::ConvCheckStrategyElch
  {
   public:
    explicit ConvCheckStrategyElchScaTraManifold(const Teuchos::ParameterList& parameters)
        : ConvCheckStrategyElch(parameters)
    {
    }

    bool exit_newton_raphson_init_pot_calc(const SSI::SsiMono& ssi_mono) override;

   protected:
    //! get L2 norms from concentration field and check if they are reasonable
    void get_and_check_l2_norm_scatra_manifold_conc(
        const SSI::SsiMono& ssi_mono, double& incnorm, double& resnorm, double& dofnorm) const;

    //! get L2 norms from potential field and check if they are reasonable
    void get_and_check_l2_norm_scatra_manifold_pot(
        const SSI::SsiMono& ssi_mono, double& incnorm, double& resnorm, double& dofnorm) const;

   private:
    [[nodiscard]] bool check_convergence(
        const SSI::SsiMono& ssi_mono, const std::map<L2norm, double>& norms) const override;

    [[nodiscard]] std::map<SSI::L2norm, double> compute_norms(
        const SSI::SsiMono& ssi_mono) const override;

    void print_newton_iteration_information(const SSI::SsiMono& ssi_mono, bool converged, bool exit,
        const std::map<L2norm, double>& norms) const override;
  };
}  // namespace SSI
FOUR_C_NAMESPACE_CLOSE

#endif
