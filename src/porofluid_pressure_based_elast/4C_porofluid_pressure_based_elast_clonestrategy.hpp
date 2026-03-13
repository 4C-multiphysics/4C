// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_CLONESTRATEGY_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_CLONESTRATEGY_HPP

#include "4C_config.hpp"

#include <functional>
#include <map>
#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::Elements
{
  class Element;
}

namespace PoroPressureBased
{
  /*!
  \brief Implementation of a cloning strategy for automatic generation of a porofluid or scalar
  transport mesh from the original structure mesh
   */
  class PorofluidCloneStrategy
  {
   public:
    /// returns conditions names to be copied (source and target name)
    [[nodiscard]] static std::map<std::string, std::string> conditions_to_copy();

    static void set_material_validation_callback(std::function<void(int)> callback);

   protected:
    /// determine element type and whether the element is copied or not
    static bool determine_ele_type(
        Core::Elements::Element* actele, bool ismyele, std::vector<std::string>& eletype);

    /// set element-specific data (material etc.)
    static void set_element_data(std::shared_ptr<Core::Elements::Element> newele,
        Core::Elements::Element* oldele, int matid, bool isnurbs);

    /// check for correct material
    static void check_material_type(int matid);
  };

}  // namespace PoroPressureBased

FOUR_C_NAMESPACE_CLOSE

#endif
