// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_MESHTYING_STRATEGY_FACTORY_HPP
#define FOUR_C_CONTACT_MESHTYING_STRATEGY_FACTORY_HPP

#include "4C_config.hpp"

#include "4C_contact_input.hpp"
#include "4C_mortar_element.hpp"
#include "4C_mortar_strategy_factory.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace CONTACT
{
  class MtAbstractStrategy;
}  // namespace CONTACT

namespace Mortar
{
  class InterfaceDataContainer;
  class MortarAbstractStrategy;
  class Element;
  class Interface;
  class StrategyDataContainer;

  namespace STRATEGY
  {
    /*! \brief Factory to create meshtying strategy
     *
     */
    class FactoryMT : public Mortar::STRATEGY::Factory
    {
     public:
      //! constructor
      FactoryMT() = default;

      //! derived
      void setup(int dim) override;

      /*! \brief Read and check meshtying/contact input parameters
       *
       * All specified contact-related input parameters are read from the
       * Global::Problem::instance() and stored into a local variable of
       * type Teuchos::ParameterList. Invalid parameter combinations are
       * sorted out and throw a FOUR_C_THROW.
       *
       * \param[in/out] params ParameterList with meshtying/contact parameters from input file
       *
       */
      void read_and_check_input(Teuchos::ParameterList& params) const;

      /** \brief Create the meshtying interfaces
       *
       * \param[in] params ParameterList with mortar meshtying/contact parameters from input file
       * \param[in/out] interfaces Collection of all mortar interfaces
       * \param porosource
       * \param porotarget
       *
       * \todo ToDo Get rid of porosource and porotarget parameters.
       *
       */
      void build_interfaces(const Teuchos::ParameterList& params,
          std::vector<std::shared_ptr<Mortar::Interface>>& interfaces, bool& porosource,
          bool& porotarget) const;

      /*! \brief Create the solver strategy object and pass all necessary data to it
       *
       * \param[in] params ParameterList with mortar meshtying/contact parameters from input file
       * \param[in] porosource
       * \param[in] porotarget
       * \param[in] dof_offset
       * \param interfaces Collection of all mortar interfaces
       *
       * \todo ToDo Get rid of porosource and porotarget parameters.
       *
       * */
      std::shared_ptr<CONTACT::MtAbstractStrategy> build_strategy(
          const Teuchos::ParameterList& params, const bool& porosource, const bool& porotarget,
          const int& dof_offset, std::vector<std::shared_ptr<Mortar::Interface>>& interfaces) const;

      /*! \brief Create the solver strategy object and pass all necessary data to it
       *
       * \param[in] stype Type of solution strategy
       * \param[in] params ParameterList with mortar meshtying/contact parameters from input file
       * \param[in] porosource
       * \param[in] porotarget
       * \param[in] dof_offset
       * \param[in] dof_row_map Dof row map
       * \param[in] node_row_map Node row map
       * \param[in] dim Spatial dimension
       * \param[in] comm_ptr Communicator
       *
       * \todo ToDo Get rid of porosource and porotarget parameters.
       *
       * \note This routine can be used like a non-member function. If you need
       * access to the class members, use the alternative call.
       *
       * */
      static std::shared_ptr<CONTACT::MtAbstractStrategy> build_strategy(
          CONTACT::SolvingStrategy stype, const Teuchos::ParameterList& params,
          const bool& porosource, const bool& porotarget, const int& dof_offset,
          std::vector<std::shared_ptr<Mortar::Interface>>& interfaces,
          const Core::LinAlg::Map* dof_row_map, const Core::LinAlg::Map* node_row_map, int dim,
          const MPI_Comm& comm_ptr);

     protected:
     private:
    };  // class FactoryMT
  }  // namespace STRATEGY
}  // namespace Mortar

FOUR_C_NAMESPACE_CLOSE

#endif
