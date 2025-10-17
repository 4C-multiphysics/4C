// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MIXTURE_ELASTIN_MEMBRANE_PRESTRESS_STRATEGY_HPP
#define FOUR_C_MIXTURE_ELASTIN_MEMBRANE_PRESTRESS_STRATEGY_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

// forward declarations
FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  class CoordinateSystemProvider;
  struct EvaluationContext;
}  // namespace Mat

namespace Mixture
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
        const std::shared_ptr<const Mat::CoordinateSystemProvider> cosy,
        Mixture::MixtureConstituent& constituent, ElastinMembraneEvaluation& membraneEvaluation,
        const Teuchos::ParameterList& params, const Mat::EvaluationContext& context, int gp,
        int eleGID) const = 0;
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
    virtual void evaluate_membrane_stress(Core::LinAlg::SymmetricTensor<double, 3, 3>& S,
        const Teuchos::ParameterList& params, const Mat::EvaluationContext& context, int gp,
        int eleGID) = 0;
  };
}  // namespace Mixture
FOUR_C_NAMESPACE_CLOSE

#endif
