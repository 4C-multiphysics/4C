// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FBI_BEAM_TO_FLUID_MESHTYING_PARAMS_HPP
#define FOUR_C_FBI_BEAM_TO_FLUID_MESHTYING_PARAMS_HPP

#include "4C_config.hpp"

#include "4C_beaminteraction_contact_params.hpp"
#include "4C_fem_general_utils_integration.hpp"

FOUR_C_NAMESPACE_OPEN


namespace FBI
{
  enum class BeamToFluidConstraintEnforcement;
  enum class BeamToFluidDiscretization;
  enum class BeamToFluidMeshtingMortarShapefunctions;
}  // namespace FBI

namespace FBI
{
  /**
   * \brief class containing all relevant parameters for beam to fluid meshtying
   */
  class BeamToFluidMeshtyingVtkOutputParams;

  /**
   * \brief Class for beam to fluid meshtying parameters.
   */
  class BeamToFluidMeshtyingParams : public BeamInteraction::BeamContactParams
  {
   public:
    /**
     * \brief Constructor.
     */
    BeamToFluidMeshtyingParams();


    /**
     * \brief Initialize with the stuff coming from input file
     */
    void init();

    /**
     * \brief Setup
     */
    void setup();

    /// Sets the flag to compute only force contributions from the beam
    void set_weak_dirichlet_flag() { calcfluidweakdirichletforce_ = true; }

    /// Sets the flag to compute force contributions from beam and fluid
    void unset_weak_dirichlet_flag() { calcfluidweakdirichletforce_ = false; }

    /// Returns \ref calcfluidweakdirichletforce_
    bool get_weak_dirichlet_flag() { return calcfluidweakdirichletforce_; }

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
    inline FBI::BeamToFluidConstraintEnforcement get_constraint_enforcement() const
    {
      return constraint_enforcement_;
    }

    /**
     * \brief Returns constraints enforcement strategy.
     */
    inline FBI::BeamToFluidDiscretization get_contact_discretization() const
    {
      return meshtying_discretization_;
    }

    /**
     * \brief Returns the penalty parameter.
     * \returns penalty parameter.
     */
    inline double get_penalty_parameter() const { return penalty_parameter_; }

    /**
     * \brief Returns the Gauss rule.
     * \returns gauss rule.
     */
    inline Core::FE::GaussRule1D get_gauss_rule() const { return gauss_rule_; }

    /**
     * \brief Returns a pointer to the visualization output parameters.
     * @return Pointer to visualization output parameters.
     */
    std::shared_ptr<const FBI::BeamToFluidMeshtyingVtkOutputParams>
    get_visualization_output_params_ptr() const
    {
      return output_params_;
    }

    /**
     * \brief Returns the shape function for the mortar Lagrange-multiplicators.
     */
    inline FBI::BeamToFluidMeshtingMortarShapefunctions get_mortar_shape_function_type() const
    {
      return mortar_shape_function_;
    }


   private:
    /// Flag if Init was called.
    bool isinit_;

    /// Flag if Setup was called.
    bool issetup_;

    /// Enforcement strategy for constraints.
    FBI::BeamToFluidConstraintEnforcement constraint_enforcement_;

    /// discretization used for the contact.
    FBI::BeamToFluidDiscretization meshtying_discretization_;

    /// Penalty parameter.
    double penalty_parameter_;

    /// Gauss rule to be used.
    Core::FE::GaussRule1D gauss_rule_;

    /**
     * \brief Flag to keep track if the RHS contribution for the weak Dirichlet enforcement of the
     * kinematic continuity condition is requested
     *
     * In the case of a DirichletNeumann algorithm the right-hand side contribution C_fs*v_s leads
     * to the unnecessary costly creation of a non-local sparse matrix and matrix-vector product.
     * Instead, the contributions of the fluid "force" introduced by the beam DOFs can be calculated
     * directly on pair level
     */
    bool calcfluidweakdirichletforce_;

    /// Visualization output params. For now I see no reason to overload this
    std::shared_ptr<FBI::BeamToFluidMeshtyingVtkOutputParams> output_params_;

    //! Shape function for the mortar Lagrange-multiplicators
    FBI::BeamToFluidMeshtingMortarShapefunctions mortar_shape_function_;
  };

}  // namespace FBI

FOUR_C_NAMESPACE_CLOSE

#endif
