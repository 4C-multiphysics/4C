// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_CONDITIONS_HPP
#define FOUR_C_BEAMINTERACTION_CONDITIONS_HPP


#include "4C_config.hpp"

#include "4C_fem_condition.hpp"
#include "4C_inpar_beaminteraction.hpp"
#include "4C_utils_exceptions.hpp"

#include <map>
#include <memory>
#include <set>
#include <vector>

FOUR_C_NAMESPACE_OPEN

// Forward declarations.
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Elements
{
  class Element;
}

namespace BeamInteraction
{
  class BeamContactPair;
  class BeamContactParams;
  namespace SubmodelEvaluator
  {
    class BeamContactAssemblyManager;
  }
}  // namespace BeamInteraction
namespace GeometryPair
{
  class GeometryEvaluationDataBase;
}
namespace Solid
{
  namespace ModelEvaluator
  {
    class BeamInteractionDataState;
  }
}  // namespace Solid


namespace BeamInteraction
{
  /**
   * \brief This abstract base class represents a single beam interaction condition.
   */
  class BeamInteractionConditionBase
  {
   public:
    /**
     * \brief Constructor.
     *
     * @param condition_line (in) The line condition containing the beam elements.
     */
    BeamInteractionConditionBase(const Core::Conditions::Condition& condition_line);

    /**
     * \brief Destructor.
     */
    virtual ~BeamInteractionConditionBase() = default;

    /**
     * \brief Create the beam contact pair needed for this condition.
     *
     * @param ele_ptrs (in) Pointer to the two elements contained in the pair.
     * @return Pointer to the created pair.
     */
    virtual std::shared_ptr<BeamInteraction::BeamContactPair> create_contact_pair(
        const std::vector<Core::Elements::Element const*>& ele_ptrs) = 0;

    /**
     * \brief Build the ID sets for this condition. The ID sets will be used to check if an element
     * is in this condition.
     */
    virtual void build_id_sets(
        const std::shared_ptr<const Core::FE::Discretization>& discretization);

    /**
     * \brief Set the displacement state.
     *
     * @param discret (in) discretization.
     * @param beaminteraction_data_state (in) Datastate of the beaminteraction model evaluator.
     */
    virtual void set_state(const std::shared_ptr<const Core::FE::Discretization>& discret,
        const std::shared_ptr<const Solid::ModelEvaluator::BeamInteractionDataState>&
            beaminteraction_data_state)
    {
    }

    /**
     * \brief Setup geometry data.
     * @param discret (in) discretization.
     */
    virtual void setup(const std::shared_ptr<const Core::FE::Discretization>& discret);

    /**
     * \brief Clear not reusable data.
     */
    virtual void clear();

    /**
     * \brief Check if a combination of a beam element ID and another element (beam, solid, ...)
     * ID is in this condition.
     */
    virtual bool ids_in_condition(const int id_line, const int id_other) const = 0;

    /**
     * \brief Create the indirect assembly manager for this condition.
     * @param discret (in) discretization.
     * @return Pointer to created assembly manager.
     */
    virtual std::shared_ptr<SubmodelEvaluator::BeamContactAssemblyManager>
    create_indirect_assembly_manager(const std::shared_ptr<const Core::FE::Discretization>& discret)
    {
      return nullptr;
    };

   protected:
    //! Pointer to the beam condition.
    const Core::Conditions::Condition* condition_line_;

    //! Set containing the beam element IDs.
    std::set<int> line_ids_;
  };

  /**
   * \brief This class manages all beam interaction conditions.
   */
  class BeamInteractionConditions
  {
   public:
    /**
     * \brief Constructor.
     */
    BeamInteractionConditions();

    /**
     * \brief Destructor.
     */
    virtual ~BeamInteractionConditions() = default;

    /**
     * \brief Get all beam interaction conditions from the discretization.
     *
     * This method searches the discretization for input beam interaction conditions, finds the
     * correct line-to- line / surface / volume pairings and adds them to the class variable \ref
     * condition_map_.
     *
     * @param discret (in) pointer to the discretization
     * @param params (in) Pointer beam contact parameters.
     */
    void set_beam_interaction_conditions(
        const Core::FE::Discretization& discret, const BeamContactParams& params_ptr);

    /**
     * \brief Build the ID sets on all contained beam interaction conditions.
     */
    void build_id_sets(std::shared_ptr<Core::FE::Discretization> discretization);

    /**
     * \brief Set the displacement state.
     *
     * @param discret (in) discretization.
     * @param beaminteraction_data_state (in) Datastate of the beaminteraction model evaluator.
     */
    virtual void set_state(const std::shared_ptr<const Core::FE::Discretization>& discret,
        const std::shared_ptr<const Solid::ModelEvaluator::BeamInteractionDataState>&
            beaminteraction_data_state);

    /**
     * \brief Setup data in the conditions.
     * @param discret (in) discretization.
     */
    virtual void setup(const std::shared_ptr<const Core::FE::Discretization>& discret);

    /**
     * \brief Clear not reusable data in the conditions.
     */
    virtual void clear();

    /**
     * \brief Create the correct pair for the given element pointers.
     *
     * We assume, that each beam interaction condition can only have one interaction pair.
     * This function checks which interaction condition contains both elements of this pair and
     * creates the correct pair.
     *
     * @param ele_ptrs (in) Pointer to the two elements contained in the pair.
     */
    void create_contact_pairs(
        std::vector<std::shared_ptr<BeamInteraction::BeamContactPair>>& created_pairs,
        const std::vector<Core::Elements::Element const*>& ele_ptrs);

    /**
     * Create all needed indirect assembly managers.
     * @param discret (in) discretization.
     * @param assembly_managers (in/out) Pointer to assembly manager vector from the beam
     * interaction submodel evaluator.
     */
    void create_indirect_assembly_managers(
        const std::shared_ptr<const Core::FE::Discretization>& discret,
        std::vector<
            std::shared_ptr<BeamInteraction::SubmodelEvaluator::BeamContactAssemblyManager>>&
            assembly_managers);

    /**
     * \brief Return a const reference to the condition map.
     */
    inline const std::map<Inpar::BeamInteraction::BeamInteractionConditions,
        std::vector<std::shared_ptr<BeamInteractionConditionBase>>>&
    get_condition_map() const
    {
      return condition_map_;
    }

    /**
     * \brief Return a mutable reference to the condition map.
     */
    inline std::map<Inpar::BeamInteraction::BeamInteractionConditions,
        std::vector<std::shared_ptr<BeamInteractionConditionBase>>>&
    get_condition_map()
    {
      return condition_map_;
    }

    /**
     * \brief Get the total number of beam interaction conditions.
     */
    inline unsigned int get_total_number_of_conditions() const
    {
      unsigned int count = 0;
      for (const auto& map_pair : condition_map_) count += map_pair.second.size();
      return count;
    }

    /**
     * \brief Check if a combination of beam element and other element id is in any beam interaction
     * condition.
     */
    bool ids_in_conditions(const int id_line, const int id_other) const
    {
      for (auto& map_pair : condition_map_)
        for (auto& condition : map_pair.second)
          if (condition->ids_in_condition(id_line, id_other)) return true;
      return false;
    };

   private:
    //! A map containing all types of beam interaction conditions. The map keys are the beam
    //! interaction type, the values are vectors with conditions (since we can have multiple
    //! conditions of the same interaction type).
    std::map<Inpar::BeamInteraction::BeamInteractionConditions,
        std::vector<std::shared_ptr<BeamInteractionConditionBase>>>
        condition_map_;
  };

  /**
   * \brief Get the global element IDs of all elements in a condition.
   *
   * @param condition (in) A pointer to the condition.
   * @param element_ids (out) A vector with all global element IDs.
   */
  void condition_to_element_ids(
      const Core::Conditions::Condition& condition, std::vector<int>& element_ids);

  /**
   * \brief Create the element ID to element pointer map.
   */
  std::unordered_map<int, const Core::Elements::Element*> condition_to_element_id_map(
      const Core::Conditions::Condition& condition);

}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
