#ifndef FOUR_C_MIXTURE_RULE_SIMPLE_HPP
#define FOUR_C_MIXTURE_RULE_SIMPLE_HPP

#include "4C_config.hpp"

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

namespace MIXTURE
{
  // forward declaration
  class SimpleMixtureRule;

  namespace PAR
  {
    class SimpleMixtureRule : public MIXTURE::PAR::MixtureRule
    {
      friend class MIXTURE::SimpleMixtureRule;

     public:
      /// constructor
      explicit SimpleMixtureRule(const Core::Mat::PAR::Parameter::Data& matdata);

      /// Create mixturerule instance
      std::unique_ptr<MIXTURE::MixtureRule> create_rule() override;

      /// @name parameters of the mixture rule
      /// @{
      const double initial_reference_density_;

      const std::vector<double> mass_fractions_;
      /// @}
    };

  }  // namespace PAR

  /*!
   * \brief This mixture rule controls the evaluation of growth and remodel simulations with
   * homogenized constrained mixture models
   */
  class SimpleMixtureRule : public MIXTURE::MixtureRule
  {
   public:
    /// Constructor for mixture rule given the input parameters
    explicit SimpleMixtureRule(MIXTURE::PAR::SimpleMixtureRule* params);

    void evaluate(const Core::LinAlg::Matrix<3, 3>& F, const Core::LinAlg::Matrix<6, 1>& E_strain,
        Teuchos::ParameterList& params, Core::LinAlg::Matrix<6, 1>& S_stress,
        Core::LinAlg::Matrix<6, 6>& cmat, int gp, int eleGID) override;

    [[nodiscard]] double return_mass_density() const override
    {
      return params_->initial_reference_density_;
    };

   private:
    ///! Rule parameters as defined in the input file
    PAR::SimpleMixtureRule* params_{};
  };
}  // namespace MIXTURE

FOUR_C_NAMESPACE_CLOSE

#endif