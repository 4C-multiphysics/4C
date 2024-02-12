/*----------------------------------------------------------------------*/
/*! \file

\brief Base data container holding data for beam-to-solid interactions.

\level 3
*/
// End doxygen header.


#ifndef BACI_BEAMINTERACTION_BEAM_TO_SOLID_PARAMS_BASE_HPP
#define BACI_BEAMINTERACTION_BEAM_TO_SOLID_PARAMS_BASE_HPP


#include "baci_config.hpp"

#include "baci_discretization_fem_general_utils_integration.hpp"
#include "baci_inpar_beam_to_solid.hpp"
#include "baci_utils_exceptions.hpp"

BACI_NAMESPACE_OPEN


namespace BEAMINTERACTION
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
    virtual void Init() = 0;

    /**
     * \brief Set the common beam-to-solid parameters.
     * @param beam_to_solid_params_list (in) parameter list with the common beam-to-solid
     * parameters.
     */
    virtual void SetBaseParams(const Teuchos::ParameterList& beam_to_solid_params_list);

    /**
     * \brief Setup member variables.
     */
    void Setup();

    /**
     * \brief Returns the isinit_ flag.
     */
    inline const bool& IsInit() const { return isinit_; };

    /**
     * \brief Returns the issetup_ flag.
     */
    inline const bool& IsSetup() const { return issetup_; };

    /**
     * \brief Checks the init and setup status.
     */
    inline void CheckInitSetup() const
    {
      if (!IsInit() or !IsSetup()) dserror("Call Init() and Setup() first!");
    }

    /**
     * \brief Checks the init status.
     */
    inline void CheckInit() const
    {
      if (!IsInit()) dserror("Init() has not been called, yet!");
    }

    /**
     * \brief Returns the contact discretization method.
     */
    inline INPAR::BEAMTOSOLID::BeamToSolidConstraintEnforcement GetConstraintEnforcement() const
    {
      return constraint_enforcement_;
    }

    /**
     * \brief Returns constraints enforcement strategy.
     */
    inline INPAR::BEAMTOSOLID::BeamToSolidContactDiscretization GetContactDiscretization() const
    {
      return contact_discretization_;
    }

    /**
     * \brief Returns the shape function for the mortar Lagrange-multiplicators.
     */
    inline INPAR::BEAMTOSOLID::BeamToSolidMortarShapefunctions GetMortarShapeFunctionType() const
    {
      return mortar_shape_function_;
    }

    /**
     * \brief Returns the penalty parameter.
     * @return penalty parameter.
     */
    inline double GetPenaltyParameter() const { return penalty_parameter_; }

    /**
     * \brief Returns the Gauss rule.
     * @return gauss rule.
     */
    inline CORE::FE::GaussRule1D GetGaussRule() const { return gauss_rule_; }

    /**
     * \brief Returns true if the coupling should be evaluated with FAD.
     */
    virtual inline bool GetIsFAD() const { return false; }

    /**
     * \brief Returns the order for the FAD type.
     */
    virtual inline int GetFADOrder() const { return 0; }

   protected:
    //! Flag if Init was called.
    bool isinit_;

    //! Flag if Setup was called.
    bool issetup_;

    //! Enforcement strategy for constraints.
    INPAR::BEAMTOSOLID::BeamToSolidConstraintEnforcement constraint_enforcement_;

    //! Discretization used for the contact.
    INPAR::BEAMTOSOLID::BeamToSolidContactDiscretization contact_discretization_;

    //! Shape function for the mortar Lagrange-multiplicators
    INPAR::BEAMTOSOLID::BeamToSolidMortarShapefunctions mortar_shape_function_;

    //! Penalty parameter.
    double penalty_parameter_;

    //! Gauss rule to be used.
    CORE::FE::GaussRule1D gauss_rule_;
  };

}  // namespace BEAMINTERACTION

BACI_NAMESPACE_CLOSE

#endif
