// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_STRATEGY_FACTORY_HPP
#define FOUR_C_CONTACT_STRATEGY_FACTORY_HPP

#include "4C_config.hpp"

#include "4C_inpar_contact.hpp"
#include "4C_mortar_element.hpp"
#include "4C_mortar_strategy_factory.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

namespace CONTACT
{
  class AbstractStratDataContainer;
  class AbstractStrategy;
  class Element;
  class Interface;
  class InterfaceDataContainer;
  class ParamsInterface;

  namespace STRATEGY
  {
    /*! \brief Factory for contact strategies
     *
     */
    class Factory : public Mortar::STRATEGY::Factory
    {
     public:
      void setup(int dim) override;

      /*! \brief Read and check contact input parameters
       *
       * All specified contact-related input parameters are read from the
       * Global::Problem::instance() and stored into a local variable of
       * type Teuchos::ParameterList. Invalid parameter combinations are
       * sorted out and throw a FOUR_C_THROW.
       *
       * \param[in/out] params ParameterList with mortar/contact parameters from input file
       *
       * \author Popp */
      void read_and_check_input(Teuchos::ParameterList& params) const;

      /** \brief Create the contact interfaces
       *
       * \param[in/out] params ParameterList with mortar/contact parameters from input file
       * \param[in/out] interfaces Collection of all mortar contact interfaces
       * \param poroslave
       * \param poromaster
       *
       * \todo ToDo Get rid of poroslave and poromaster parameters.
       *
       * \author Popp */
      void build_interfaces(const Teuchos::ParameterList& params,
          std::vector<std::shared_ptr<CONTACT::Interface>>& interfaces, bool& poroslave,
          bool& poromaster) const;

      /** \brief Create a contact interface object based on the given information
       *
       *  \author hiermeier \date 03/17 */
      static std::shared_ptr<CONTACT::Interface> create_interface(const int id, MPI_Comm comm,
          const int dim, Teuchos::ParameterList& icparams, const bool selfcontact,
          std::shared_ptr<CONTACT::InterfaceDataContainer> interfaceData_ptr = nullptr,
          const int contactconstitutivelaw_id = -1);

      /** \brief Create a contact interface object based on the given information
       *
       *  \author hiermeier \date 03/17 */
      static std::shared_ptr<CONTACT::Interface> create_interface(
          const enum Inpar::CONTACT::SolvingStrategy stype, const int id, MPI_Comm comm,
          const int dim, Teuchos::ParameterList& icparams, const bool selfcontact,
          std::shared_ptr<CONTACT::InterfaceDataContainer> interface_data_ptr,
          const int contactconstitutivelaw_id = -1);

      /*! \brief Create the solver strategy object and pass all necessary data to it
       *
       * \author Popp */
      std::shared_ptr<CONTACT::AbstractStrategy> build_strategy(
          const Teuchos::ParameterList& params, const bool& poroslave, const bool& poromaster,
          const int& dof_offset, std::vector<std::shared_ptr<CONTACT::Interface>>& interfaces,
          CONTACT::ParamsInterface* cparams_interface = nullptr) const;

      /*! \brief Create the solver strategy object and pass all necessary data to it
       *
       *  \note This routine can be used like a non-member function. If you need
       *  access to the class members, use the alternative call.
       *
       * \author hiermeier \date 03/17 */
      static std::shared_ptr<CONTACT::AbstractStrategy> build_strategy(
          const Inpar::CONTACT::SolvingStrategy stype, const Teuchos::ParameterList& params,
          const bool& poroslave, const bool& poromaster, const int& dof_offset,
          std::vector<std::shared_ptr<CONTACT::Interface>>& interfaces,
          const Epetra_Map* dof_row_map, const Epetra_Map* node_row_map, const int dim,
          const MPI_Comm& comm_ptr, std::shared_ptr<CONTACT::AbstractStratDataContainer> data_ptr,
          CONTACT::ParamsInterface* cparams_interface = nullptr);

      //! Create the desired search tree object
      void build_search_tree(
          const std::vector<std::shared_ptr<CONTACT::Interface>>& interfaces) const;

      //! print some final screen output
      void print(const std::vector<std::shared_ptr<CONTACT::Interface>>& interfaces,
          CONTACT::AbstractStrategy& strategy_ptr, const Teuchos::ParameterList& params) const;

      /*! \brief print strategy banner
       *
       *  \param soltype (in) : contact solving strategy type */
      static void print_strategy_banner(const enum Inpar::CONTACT::SolvingStrategy soltype);

     protected:
     private:
      /*! @name Porous media
       *
       * \todo Move to some other place outside the pure contact factory.
       */
      //!@{

      /*! \brief Set Parent Elements for Poro Face Elements
       *
       *  \author Ager */
      void set_poro_parent_element(enum Mortar::Element::PhysicalType& slavetype,
          enum Mortar::Element::PhysicalType& mastertype, CONTACT::Element& cele,
          std::shared_ptr<Core::Elements::Element>& ele,
          const Core::FE::Discretization& discret) const;

      /*! \brief Find Physical Type (Poro or Structure) of Poro Interface
       *
       *  \author Ager */
      void find_poro_interface_types(bool& poromaster, bool& poroslave, bool& structmaster,
          bool& structslave, const enum Mortar::Element::PhysicalType slavetype,
          const enum Mortar::Element::PhysicalType mastertype) const;

      //!@}

      void fully_overlapping_interfaces(
          std::vector<std::shared_ptr<CONTACT::Interface>>& interfaces) const;

      int identify_full_subset(const Epetra_Map& map_0, const Epetra_Map& map_1,
          bool throw_if_partial_subset_on_proc = true) const;

      /*!
       * \brief Set parameters to contact interface parameter list
       *
       * @param[in] conditiongroupid:                ID of the current contact condition group
       * @param[in/out] contactinterfaceparameters:  the sublist 'ContactS2ICoupling' containing
       *                   the scatra-scatra interface parameters is added to this list if needed
       */
      void set_parameters_for_contact_condition(
          int conditiongroupid, Teuchos::ParameterList& contactinterfaceparameters) const;
    };
  }  // namespace STRATEGY
}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif
