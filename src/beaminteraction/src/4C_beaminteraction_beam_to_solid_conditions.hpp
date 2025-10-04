// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_CONDITIONS_HPP
#define FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_CONDITIONS_HPP

#include "4C_config.hpp"

#include "4C_beaminteraction_conditions.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_utils_exceptions.hpp"

#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

FOUR_C_NAMESPACE_OPEN


// Forward declarations.
namespace Core::Elements
{
  class FaceElement;
}
namespace BeamInteraction
{
  class BeamToSolidParamsBase;
}
namespace Inpar
{
  namespace BeamToSolid
  {
    enum class BeamToSolidMortarShapefunctions;
  }
}  // namespace Inpar


namespace BeamInteraction
{
  /**
   * \brief This base class represents a single beam-to-solid interaction condition.
   */
  class BeamToSolidCondition : public BeamInteractionConditionBase
  {
   public:
    /**
     * \brief Constructor.
     *
     * @param condition_line (in) The line condition containing the beam elements.
     * @param condition_other (in) The surface / volume condition containing the solid elements
     * interacting with the beam.
     * @param beam_to_solid_params (in) Pointer to the beam-to-solid parameters.
     */
    BeamToSolidCondition(const Core::Conditions::Condition& condition_line,
        const Core::Conditions::Condition& condition_other,
        const std::shared_ptr<const BeamToSolidParamsBase>& beam_to_solid_params);


    /**
     * \brief Check if a combination of beam and solid id is in this condition.
     */
    bool ids_in_condition(const int id_line, const int id_other) const override;

    /**
     * \brief Clear not reusable data (derived).
     */
    void clear() override;

    /**
     * \brief Create the beam to solid pairs needed for this condition (derived).
     */
    std::shared_ptr<BeamInteraction::BeamContactPair> create_contact_pair(
        const std::vector<Core::Elements::Element const*>& ele_ptrs) override;

    /**
     * \brief Return the condition of the other geometry (volume or surface).
     */
    const Core::Conditions::Condition& get_other_condition() const { return *condition_other_; }

    /**
     * \brief Create the indirect assembly manager for this condition.
     * @param discret (in) discretization.
     */
    std::shared_ptr<SubmodelEvaluator::BeamContactAssemblyManager> create_indirect_assembly_manager(
        const std::shared_ptr<const Core::FE::Discretization>& discret) override;

    /**
     * \brief Return a pointer to the geometry evaluation data in this condition.
     */
    std::shared_ptr<const GeometryPair::GeometryEvaluationDataBase> get_geometry_evaluation_data()
        const
    {
      return geometry_evaluation_data_;
    }

   protected:
    /**
     * \brief Check if a solid ID is in this condition.
     */
    virtual bool id_in_other(const int id_other) const = 0;

    /**
     * \brief Return the created beam contact pair for this condition.
     *
     * This function is called by CreateContactPair where the geometry pair in the created contact
     * pair is initialized.
     *
     * @param ele_ptrs (in) Pointer to the two elements contained in the pair.
     * @return Pointer to the created pair.
     */
    virtual std::shared_ptr<BeamInteraction::BeamContactPair> create_contact_pair_internal(
        const std::vector<Core::Elements::Element const*>& ele_ptrs) = 0;

   protected:
    //! Pointer to the geometry evaluation data for this condition.
    std::shared_ptr<GeometryPair::GeometryEvaluationDataBase> geometry_evaluation_data_;

    //! Pointer to the solid condition.
    const Core::Conditions::Condition* condition_other_;

    //! Vector containing all beam contact pairs created by this condition.
    std::vector<std::shared_ptr<BeamContactPair>> condition_contact_pairs_;

    //! Pointer to the beam-to-solid parameters.
    std::shared_ptr<const BeamToSolidParamsBase> beam_to_solid_params_;
  };

  /**
   * \brief This base class represents a single beam-to-solid volume mesh tying interaction
   * condition.
   */
  class BeamToSolidConditionVolumeMeshtying : public BeamToSolidCondition
  {
   public:
    /**
     * \brief Constructor (derived).
     */
    BeamToSolidConditionVolumeMeshtying(const Core::Conditions::Condition& condition_line,
        const Core::Conditions::Condition& condition_other,
        const std::shared_ptr<const BeamToSolidParamsBase>& beam_to_solid_params);


    /**
     * \brief Build the volume ID sets for this condition.
     *
     * The BuildIdSets method from the base class is called to build the beam IDs.
     */
    void build_id_sets(
        const std::shared_ptr<const Core::FE::Discretization>& discretization) override;

   protected:
    /**
     * \brief Return the created beam contact pair for this condition. (derived)
     */
    std::shared_ptr<BeamInteraction::BeamContactPair> create_contact_pair_internal(
        const std::vector<Core::Elements::Element const*>& ele_ptrs) override;

    /**
     * \brief Check if a solid ID is in this condition.
     */
    inline bool id_in_other(const int id_other) const override
    {
      if (volume_ids_.find(id_other) != volume_ids_.end()) return true;
      return false;
    }

   private:
    //! Set containing the volume element IDs.
    std::set<int> volume_ids_;
  };

  /**
   * \brief This base class represents a single beam-to-solid surface mesh tying interaction
   * condition.
   */
  class BeamToSolidConditionSurface : public BeamToSolidCondition
  {
   public:
    /**
     * \brief Constructor (derived).
     */
    BeamToSolidConditionSurface(const Core::Conditions::Condition& condition_line,
        const Core::Conditions::Condition& condition_other,
        const std::shared_ptr<const BeamToSolidParamsBase>& beam_to_solid_params,
        const bool is_mesh_tying_in);


    /**
     * \brief Build the surface ID sets for this condition. The BuildIdSets method from the base
     * class is called to build the beam IDs.
     */
    void build_id_sets(
        const std::shared_ptr<const Core::FE::Discretization>& discretization) override;

    /**
     * \brief Here we get all face elements that are needed for the created pairs. This includes
     * elements which are not part of any pair, but share a node with a surface of a pair.
     *
     * @param discret (in) discretization.
     */
    void setup(const std::shared_ptr<const Core::FE::Discretization>& discret) override;

    /**
     * \brief Set the displacement state (derived).
     */
    void set_state(const std::shared_ptr<const Core::FE::Discretization>& discret,
        const std::shared_ptr<const Solid::ModelEvaluator::BeamInteractionDataState>&
            beaminteraction_data_state) override;

   protected:
    /**
     * \brief Return the created beam contact pair for this condition. (derived)
     */
    std::shared_ptr<BeamInteraction::BeamContactPair> create_contact_pair_internal(
        const std::vector<Core::Elements::Element const*>& ele_ptrs) override;

    /**
     * \brief Check if a solid ID is in this condition.
     */
    inline bool id_in_other(const int id_other) const override
    {
      if (surface_ids_.find(id_other) != surface_ids_.end()) return true;
      return false;
    }

    /**
     * \brief If this condition is a mesh tying condition.
     */
    inline bool is_mesh_tying() const { return is_mesh_tying_; }

    /**
     * \brief If this condition is a contact condition.
     */
    inline bool is_contact() const { return !is_mesh_tying_; }

   private:
    //! If the condition is mesh tying or contact.
    bool is_mesh_tying_;

    //! Map containing the global volume element IDs for each face element of the surface in this
    //! condition.
    std::unordered_map<int, const Core::Elements::Element*> surface_ids_;
  };


  /**
   * \brief Create a beam-to-solid volume pair depending on the solid volume shape.
   * @tparam BtsClass Beam-to-solid class to create
   * @tparam bts_template_arguments Template arguments when creating the class
   * @param shape (in) Shape of the solid volume
   * @return The created beam-to-solid pair
   */
  template <template <typename...> class BtsClass, typename... BtsTemplateArguments>
  std::shared_ptr<BeamInteraction::BeamContactPair> create_beam_to_solid_volume_pair_shape(
      const Core::FE::CellType shape);

  /**
   * \brief Create a beam-to-solid volume pair depending on the solid volume shape, without nurbs.
   *
   * This is for pairs which are not compatible with nurbs discretizations.
   *
   * @tparam BtsClass Beam-to-solid class to create
   * @tparam bts_template_arguments Template arguments when creating the class
   * @param shape
   * @return The created beam-to-solid pair
   */
  template <template <typename...> class BtsClass, typename... BtsTemplateArguments>
  std::shared_ptr<BeamInteraction::BeamContactPair> create_beam_to_solid_volume_pair_shape_no_nurbs(
      const Core::FE::CellType shape);

  /**
   * \brief Create a beam-to-solid volume mortar pair depending on the solid volume shape and mortar
   * shape function(s).
   * @tparam BtsClass Beam-to-solid class to create
   * @tparam bts_mortar_template_arguments
   * @tparam bts_mortar_shape Template collection of type of optional given (e.g. rotational) mortar
   * shape functions
   * @param shape (in) Shape of the solid volume
   * @param mortar_shape_function (in) Mortar shape function to be appended to the
   * bts_template_arguments in this call
   * @param other_mortar_shape_function (in) Other mortar shape functions
   * @return The created beam-to-solid pair
   */
  template <template <typename...> class BtsClass, typename... BtsMortarTemplateArguments,
      typename... BtsMortarShape>
  std::shared_ptr<BeamInteraction::BeamContactPair> create_beam_to_solid_volume_pair_mortar(
      const Core::FE::CellType shape,
      const Inpar::BeamToSolid::BeamToSolidMortarShapefunctions mortar_shape_function,
      BtsMortarShape... other_mortar_shape_function);

  /**
   * \brief Call the pair creation based on the shape once all mortar shape functions have been
   * converted to template arguments.
   * @tparam BtsClass Beam-to-solid class to create
   * @tparam bts_mortar_template_arguments Template arguments when creating the class
   * @param shape (in) Shape of the solid volume
   * @return The created beam-to-solid pair
   */
  template <template <typename...> class BtsClass, typename... BtsMortarTemplateArguments>
  std::shared_ptr<BeamInteraction::BeamContactPair> create_beam_to_solid_volume_pair_mortar(
      const Core::FE::CellType shape);
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
