#ifndef FOUR_C_MIXTURE_RULE_FUNCTION_HPP
#define FOUR_C_MIXTURE_RULE_FUNCTION_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_mixture_rule.hpp"
#include "4C_utils_function.hpp"

#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace MIXTURE
{
  // forward declaration
  class FunctionMixtureRule;

  namespace PAR
  {
    class FunctionMixtureRule : public MIXTURE::PAR::MixtureRule
    {
      friend class MIXTURE::FunctionMixtureRule;

     public:
      /// constructor
      explicit FunctionMixtureRule(const Core::Mat::PAR::Parameter::Data& matdata);

      /// Create mixturerule instance
      std::unique_ptr<MIXTURE::MixtureRule> create_rule() override;

      /// @name parameters of the mixture rule
      /// @{
      const double initial_reference_density_;

      const std::vector<int> mass_fractions_funct_ids_;
      /// @}
    };

  }  // namespace PAR

  /*!
   * \brief Mixture rule to be used in homogenized constrained mixture models. It scales the mass
   * fractions of the individual constitutents by functions of space and time.
   */
  class FunctionMixtureRule : public MIXTURE::MixtureRule
  {
   public:
    /// Constructor for mixture rule given the input parameters
    explicit FunctionMixtureRule(MIXTURE::PAR::FunctionMixtureRule* params);

    void evaluate(const Core::LinAlg::Matrix<3, 3>& F, const Core::LinAlg::Matrix<6, 1>& E_strain,
        Teuchos::ParameterList& params, Core::LinAlg::Matrix<6, 1>& S_stress,
        Core::LinAlg::Matrix<6, 6>& cmat, int gp, int eleGID) override;

    [[nodiscard]] double return_mass_density() const override
    {
      return params_->initial_reference_density_;
    };

    void setup(Teuchos::ParameterList& params, const int eleGID) override;

    void unpack_mixture_rule(Core::Communication::UnpackBuffer& buffer) override;

   private:
    ///! Rule parameters as defined in the input file
    PAR::FunctionMixtureRule* params_{};
    std::vector<const Core::Utils::FunctionOfSpaceTime*> mass_fractions_functions_;
  };
}  // namespace MIXTURE

FOUR_C_NAMESPACE_CLOSE

#endif