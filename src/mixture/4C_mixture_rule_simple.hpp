// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MIXTURE_RULE_SIMPLE_HPP
#define FOUR_C_MIXTURE_RULE_SIMPLE_HPP

#include "4C_config.hpp"

#include "4C_io_input_field.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_mixture_rule.hpp"

#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Mat
{
  namespace PAR
  {
    class Material;
  }
}  // namespace Mat

namespace Mixture
{
  // forward declaration
  class SimpleMixtureRule;

  namespace PAR
  {
    class SimpleMixtureRule : public Mixture::PAR::MixtureRule
    {
      friend class Mixture::SimpleMixtureRule;

     public:
      /// constructor
      explicit SimpleMixtureRule(const Core::Mat::PAR::Parameter::Data& matdata);

      /// Create mixturerule instance
      std::unique_ptr<Mixture::MixtureRule> create_rule() override;

      /// @name parameters of the mixture rule
      /// @{
      const double initial_reference_density_;

      const Core::IO::InputField<std::vector<double>> mass_fractions_;
      /// @}
    };

  }  // namespace PAR

  /*!
   * \brief This mixture rule controls the evaluation of growth and remodel simulations with
   * homogenized constrained mixture models
   */
  class SimpleMixtureRule : public Mixture::MixtureRule
  {
   public:
    /// Constructor for mixture rule given the input parameters
    explicit SimpleMixtureRule(Mixture::PAR::SimpleMixtureRule* params);

    void evaluate(const Core::LinAlg::Tensor<double, 3, 3>& F,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& E_strain,
        const Teuchos::ParameterList& params, Core::LinAlg::SymmetricTensor<double, 3, 3>& S_stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) override;

    void setup(const Teuchos::ParameterList& params, int eleGID) override;

    [[nodiscard]] double return_mass_density() const override
    {
      return params_->initial_reference_density_;
    };

   private:
    ///! Rule parameters as defined in the input file
    PAR::SimpleMixtureRule* params_{};
  };
}  // namespace Mixture

FOUR_C_NAMESPACE_CLOSE

#endif