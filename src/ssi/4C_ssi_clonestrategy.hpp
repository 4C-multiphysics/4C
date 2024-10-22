// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SSI_CLONESTRATEGY_HPP
#define FOUR_C_SSI_CLONESTRATEGY_HPP

#include "4C_config.hpp"

#include "4C_inpar_scatra.hpp"

#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace Core::Elements
{
  class Element;
}

namespace SSI
{
  /*!
  \brief strategy for cloning scatra discretization from structure discretization

  For some scatra-structure interaction problems, the scatra discretization is obtained through
  cloning from the structure discretization.

  \date 09/17
  */

  class ScatraStructureCloneStrategy
  {
   public:
    explicit ScatraStructureCloneStrategy() = default;

    virtual ~ScatraStructureCloneStrategy() = default;

    //! return map with original names of conditions to be cloned as key values, and final names of
    //! cloned conditions as mapped values
    virtual std::map<std::string, std::string> conditions_to_copy() const;

    //! get impltype of scatra element from structure element
    //!
    //! \param ele     element whose ScaTra::ImplType shall be determined
    //! \return        impltype of the scatra element
    virtual Inpar::ScaTra::ImplType get_impl_type(Core::Elements::Element* ele);

   protected:
    //! check material of cloned element
    //!
    //! \param matid     material of cloned element
    void check_material_type(const int matid);

    //! decide whether element should be cloned or not, and if so, determine type of cloned element
    //!
    //! \param actele       current element on source discretization
    //! \param ismyele      ownership flag
    //! \param eletype      vector storing types of cloned elements
    //! \return
    virtual bool determine_ele_type(
        Core::Elements::Element* actele, const bool ismyele, std::vector<std::string>& eletype);

    //! provide cloned element with element specific data (material etc.)
    //!
    //! \param newele    current cloned element on target discretization
    //! \param oldele    current element on source discretization
    //! \param matid     material of cloned element
    //! \param isnurbs   nurbs flag
    virtual void set_element_data(Teuchos::RCP<Core::Elements::Element> newele,
        Core::Elements::Element* oldele, const int matid, const bool isnurbs);
  };

  class ScatraStructureCloneStrategyManifold : public ScatraStructureCloneStrategy
  {
   public:
    std::map<std::string, std::string> conditions_to_copy() const override;

    void set_element_data(Teuchos::RCP<Core::Elements::Element> newele,
        Core::Elements::Element* oldele, const int matid, const bool isnurbs) override;
  };

}  // namespace SSI
FOUR_C_NAMESPACE_CLOSE

#endif
