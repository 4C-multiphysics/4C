// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_UTILS_HPP
#define FOUR_C_CONTACT_UTILS_HPP

#include "4C_config.hpp"

#include "4C_fem_condition.hpp"

#include <memory>
#include <set>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Nodes
{
  class Node;
}

namespace Core::Elements
{
  class Element;
}

namespace Core::LinAlg
{
  class SerialDenseMatrix;
}  // namespace Core::LinAlg

namespace CONTACT
{
  /// enum, which specifies the desired matrix block for the different models
  enum class MatBlockType
  {
    displ_displ,          ///< Kdd block (structural block)
    displ_lm,             ///< Kdz block (of the corresponding model evaluator)
    lm_displ,             ///< Kzd block (of the corresponding model evaluator)
    lm_lm,                ///< Kzz block (of the corresponding model evaluator)
    porofluid_porofluid,  ///< Kpp block (porofluid-porofluid)
    porofluid_displ,      ///< Kpd block (porofluid-structure)
    displ_porofluid,      ///< Kdp block (structure-porofluid)
    scatra_scatra,        ///< Kss block (scatra-scatra)
    scatra_displ,         ///< Ksd block (scatra-structure)
    displ_scatra,         ///< Kds block (structure-scatra)
    elch_elch,            ///< Kee block (elch-elch)
    elch_displ,           ///< Ked block (elch-structure)
    displ_elch            ///< Kde block (structure-elch)
  };

  //! enum, which specifies the desired vector blocks for the different models
  enum class VecBlockType
  {
    displ,       ///< displacement block (structural block)
    constraint,  ///< lagrange multiplier/constraint block of the corresponding model
    porofluid,   ///< porofluid block (porofluid block)
    scatra,      ///< scalar transport block (scatra block)
    elch         ///< electrochemistry block (elch block)
  };

  std::string vec_block_type_to_str(const VecBlockType bt);

  namespace Utils
  {
    /// Get the solid to solid contact conditions
    int get_contact_conditions(std::vector<const Core::Conditions::Condition*>& contact_conditions,
        const std::vector<const Core::Conditions::Condition*>& beamandsolidcontactconditions,
        const bool& throw_error = true);

    /// Find the solid to solid contact conditions and combine them to contact condition groups
    int get_contact_condition_groups(
        std::vector<std::vector<const Core::Conditions::Condition*>>& ccond_grps,
        const Core::FE::Discretization& discret_wrapper, const bool& throw_error = true);

    /// Combine the solid to solid contact conditions to contact condition groups
    void get_contact_condition_groups(
        std::vector<std::vector<const Core::Conditions::Condition*>>& ccond_grps,
        const std::vector<const Core::Conditions::Condition*>& cconds);

    /// Gather information which side is target and which side is source
    void get_target_source_side_info(std::vector<bool>& issource, std::vector<bool>& isself,
        const std::vector<const Core::Conditions::Condition*>& cond_grp);

    /**
     * \brief Gather information on initialization (Active/Inactive)
     *
     * \param [in,out]  Two_half_pass: two half pass approach applied for current condition group
     * \param [in,out]  Check_nonsmooth_selfcontactsurface: reference configuration check for
     *                  non-smooth self contact shall be performed for current condition group
     * \param [in,out]  Searchele_AllProc: Search elements on all processors
     * \param [in,out]  isactive:  condition is set active
     * \param [in]      issource:   condition is defined as source side
     * \param [in]      isself:    condition is self contact condition
     * \param [in]      cond_grp: current contact condition group (i.e. conditions with same ID)
     *
     * */
    void get_initialization_info(bool& Two_half_pass, bool& Check_nonsmooth_selfcontactsurface,
        bool& Searchele_AllProc, std::vector<bool>& isactive, std::vector<bool>& issource,
        std::vector<bool>& isself, const std::vector<const Core::Conditions::Condition*>& cond_grp);

  }  // namespace Utils
}  // namespace CONTACT


FOUR_C_NAMESPACE_CLOSE

#endif
