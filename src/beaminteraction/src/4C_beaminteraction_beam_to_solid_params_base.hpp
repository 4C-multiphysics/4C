// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_PARAMS_BASE_HPP
#define FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_PARAMS_BASE_HPP


#include "4C_config.hpp"

#include "4C_fem_general_utils_integration.hpp"
#include "4C_inpar_beam_to_solid.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN


namespace BeamInteraction
{
  /**
   * \brief Base class for beam to solid parameters.
   */
  class BeamToSolidParamsBase
  {
   public:
    /**
     * \brief Constructor.
     */
    BeamToSolidParamsBase();

    /**
     * \brief Destructor.
     */
    virtual ~BeamToSolidParamsBase() = default;

    /**
     * \brief Initialize with the stuff coming from input file.
     */
    virtual void init() = 0;

    /**
     * \brief Set the common beam-to-solid parameters.
     * @param beam_to_solid_params_list (in) parameter list with the common beam-to-solid
     * parameters.
     */
    virtual void set_base_params(const Teuchos::ParameterList& beam_to_solid_params_list);

    /**
     * \brief Setup member variables.
     */
    void setup();

    /**
     * \brief Returns the isinit_ flag.
     */
    inline const bool& is_init() const { return isinit_; };

    /**
     * \brief Returns the issetup_ flag.
     */
    inline const bool& is_setup() const { return issetup_; };

    /**
     * \brief Checks the init and setup status.
     */
    inline void check_init_setup() const
    {
      if (!is_init() or !is_setup()) FOUR_C_THROW("Call init() and setup() first!");
    }

    /**
     * \brief Checks the init status.
     */
    inline void check_init() const
    {
      if (!is_init()) FOUR_C_THROW("init() has not been called, yet!");
    }

    /**
     * \brief Returns the contact discretization method.
     */
    inline Inpar::BeamToSolid::BeamToSolidConstraintEnforcement get_constraint_enforcement() const
    {
      return constraint_enforcement_;
    }

    /**
     * \brief Returns constraints enforcement strategy.
     */
    inline Inpar::BeamToSolid::BeamToSolidContactDiscretization get_contact_discretization() const
    {
      return contact_discretization_;
    }

    /**
     * \brief Returns the type of lagrange formulation used.
     */
    inline Inpar::BeamToSolid::BeamToSolidLagrangeFormulation get_lagrange_formulation() const
    {
      return lagrange_formulation_;
    }

    /**
     * \brief Returns the shape function for the mortar Lagrange-multiplicators.
     */
    inline Inpar::BeamToSolid::BeamToSolidMortarShapefunctions get_mortar_shape_function_type()
        const
    {
      return mortar_shape_function_;
    }

    /**
     * \brief Returns the penalty parameter.
     * @return penalty parameter.
     */
    inline double get_penalty_parameter() const { return penalty_parameter_; }

    /**
     * \brief Returns the Gauss rule.
     * @return gauss rule.
     */
    inline Core::FE::GaussRule1D get_gauss_rule() const { return gauss_rule_; }

    /**
     * \brief Returns true if the coupling should be evaluated with FAD.
     */
    virtual inline bool get_is_fad() const { return false; }

    /**
     * \brief Returns the order for the FAD type.
     */
    virtual inline int get_fad_order() const { return 0; }

    /**
     * \brief Returns the order for the FAD type.
     */
    [[nodiscard]] bool is_rotational_coupling() const { return rotational_coupling_; }

   protected:
    //! Flag if Init was called.
    bool isinit_;

    //! Flag if Setup was called.
    bool issetup_;

    //! Enforcement strategy for constraints.
    Inpar::BeamToSolid::BeamToSolidConstraintEnforcement constraint_enforcement_;

    //! discretization used for the contact.
    Inpar::BeamToSolid::BeamToSolidContactDiscretization contact_discretization_;

    //! Shape function for the mortar Lagrange-multiplicators
    Inpar::BeamToSolid::BeamToSolidMortarShapefunctions mortar_shape_function_;

    //! Penalty parameter.
    double penalty_parameter_;

    //! Gauss rule to be used.
    Core::FE::GaussRule1D gauss_rule_;

    //! Flag for rotational coupling
    bool rotational_coupling_;

    //! Type of lagrange formulation
    Inpar::BeamToSolid::BeamToSolidLagrangeFormulation lagrange_formulation_;
  };

}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
