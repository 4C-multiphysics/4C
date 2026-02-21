// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_CONDITION_DEFINITION_HPP
#define FOUR_C_FEM_CONDITION_DEFINITION_HPP

#include "4C_config.hpp"

#include "4C_fem_condition.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_io_input_spec.hpp"

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::IO
{
  class InputFile;
}

namespace Core::Conditions
{

  /**
   * @brief Definition of a condition.
   *
   * This class groups all data that is needed to create a Condition. It contains the InputSpec
   * defining the parameters of the condition and information about the associated geometry.
   */
  class ConditionDefinition
  {
   public:
    /// construction of a condition definition
    /*!
      \param sectionname name of input file section
      \param conditionname name of conditions in Core::FE::Discretization
      \param description description of condition type
      \param condtype type of conditions to be build
      \param buildgeometry whether we need conditions elements
      \param gtype type of geometry the condition lives on
     */
    ConditionDefinition(std::string sectionname, std::string conditionname, std::string description,
        Core::Conditions::ConditionType condtype, bool buildgeometry,
        Core::Conditions::GeometryType gtype);

    /**
     * Add an InputSpec @p spec as another component of this condition. The ordering of the
     * components is irrelevant.
     */
    void add_component(Core::IO::InputSpec&& spec);
    void add_component(const Core::IO::InputSpec& spec);

    /// read all conditions from my input file section
    /*!
      \param input the input file
      \param cmap an empty multimap to fill with the read conditions (key: entity id, value:
      condition)
      \param node_sets_names map of node set names to their ids from the external mesh file
     */
    void read(Core::IO::InputFile& input,
        std::multimap<int, std::shared_ptr<Core::Conditions::Condition>>& cmap,
        const std::map<std::string, std::vector<int>>& node_sets_names) const;

    /// name of my section in input file
    std::string section_name() const { return sectionname_; }

    /// my condition name
    std::string name() const { return conditionname_; }

    /// my GeometryType
    Core::Conditions::GeometryType geometry_type() const { return gtype_; }

    /// Get the InputSpec for this ConditionDefinition
    [[nodiscard]] Core::IO::InputSpec spec() const;

   private:
    std::string sectionname_;
    std::string conditionname_;
    std::string description_;
    Core::Conditions::ConditionType condtype_;
    bool buildgeometry_;
    Core::Conditions::GeometryType gtype_;

    std::vector<Core::IO::InputSpec> specs_;
  };

}  // namespace Core::Conditions


FOUR_C_NAMESPACE_CLOSE

#endif
