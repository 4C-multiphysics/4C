#ifndef FOUR_C_MIXTURE_ELASTIN_MEMBRANE_PRESTRESS_STRATEGY_HPP
#define FOUR_C_MIXTURE_ELASTIN_MEMBRANE_PRESTRESS_STRATEGY_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

// forward declarations
FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  class CoordinateSystemProvider;
}

namespace MIXTURE
{
  // Forward declaration of mixture classes
  class MixtureConstituent;
  class MixtureRule;
  class ElastinMembraneEvaluation;

  /*!
   * \brief Prestress strategy for the elastin membrane material used for growth and remodeling
   * simulations with homogenized constrained mixtures
   */
  class ElastinMembranePrestressStrategy
  {
   public:
    /**
     * Virtual destructor.
     */
    virtual ~ElastinMembranePrestressStrategy() = default;

    /*!
     * \brief Evaluates the fraction of the elastin membrane contribution to the total elastin
     * response to ensure equilibrium in the reference configuration
     *
     * \param mixtureRule Mixture rule
     * \param cosy Coordinate systems
     * \param constituent Constituent that needs to be prestressed
     * \param membraneEvaluation Evaluator of the membrane material
     * \param params Container for additional information
     * \param gp Gauss point
     * \param eleGID Global element id
     * \return double Fraction of the elastin membrane contribution to the total elastin response
     */
    virtual double evaluate_mue_frac(MixtureRule& mixtureRule,
        const Teuchos::RCP<const Mat::CoordinateSystemProvider> cosy,
        MIXTURE::MixtureConstituent& constituent, ElastinMembraneEvaluation& membraneEvaluation,
        Teuchos::ParameterList& params, int gp, int eleGID) const = 0;
  };

  /*!
   * \brief Evaluation of the membrane material of elastin
   */
  class ElastinMembraneEvaluation
  {
   public:
    /**
     * Virtual destructor.
     */
    virtual ~ElastinMembraneEvaluation() = default;

    /*!
     * \brief Evaluates the 2D membrane stress
     *
     * \param S (out)     : Membrane 2. Piola-Kirchhoff stress tensor in stress like Voigt notation
     * \param params (in) : Container for additional information
     * \param gp (in)     : Gauss point
     * \param eleGID (in) : Global element id
     */
    virtual void evaluate_membrane_stress(
        Core::LinAlg::Matrix<6, 1>& S, Teuchos::ParameterList& params, int gp, int eleGID) = 0;
  };
}  // namespace MIXTURE
FOUR_C_NAMESPACE_CLOSE

#endif
