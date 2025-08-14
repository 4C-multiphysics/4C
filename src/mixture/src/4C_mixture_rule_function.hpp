// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MIXTURE_RULE_FUNCTION_HPP
#define FOUR_C_MIXTURE_RULE_FUNCTION_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_mixture_rule.hpp"
#include "4C_utils_function.hpp"

#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Mixture
{
  // forward declaration
  class FunctionMixtureRule;

  namespace PAR
  {
    class FunctionMixtureRule : public Mixture::PAR::MixtureRule
    {
      friend class Mixture::FunctionMixtureRule;

     public:
      /// constructor
      explicit FunctionMixtureRule(const Core::Mat::PAR::Parameter::Data& matdata);

      /// Create mixturerule instance
      std::unique_ptr<Mixture::MixtureRule> create_rule() override;

      /// @name parameters of the mixture rule
      /// @{
      const double initial_reference_density_;

      const std::vector<int> mass_fractions_funct_ids_;
      /// @}
    };

  }  // namespace PAR

  /*!
   * \brief Mixture rule to be used in homogenized constrained mixture models. It scales the mass
   * fractions of the individual constituents by functions of space and time.
   */
  class FunctionMixtureRule : public Mixture::MixtureRule
  {
   public:
    /// Constructor for mixture rule given the input parameters
    explicit FunctionMixtureRule(Mixture::PAR::FunctionMixtureRule* params);

    void evaluate(const Core::LinAlg::Tensor<double, 3, 3>& F,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& E, const Teuchos::ParameterList& params,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& S,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) override;

    [[nodiscard]] double return_mass_density() const override
    {
      return params_->initial_reference_density_;
    };

    void setup(const Teuchos::ParameterList& params, const int eleGID) override;

    void unpack_mixture_rule(Core::Communication::UnpackBuffer& buffer) override;

   private:
    ///! Rule parameters as defined in the input file
    PAR::FunctionMixtureRule* params_{};
    std::vector<const Core::Utils::FunctionOfSpaceTime*> mass_fractions_functions_;
  };
}  // namespace Mixture

FOUR_C_NAMESPACE_CLOSE

#endif