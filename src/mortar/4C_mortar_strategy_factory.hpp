// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MORTAR_STRATEGY_FACTORY_HPP
#define FOUR_C_MORTAR_STRATEGY_FACTORY_HPP

#include "4C_config.hpp"

#include "4C_inpar_contact.hpp"

#include <mpi.h>

#include <memory>
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

namespace Core::IO
{
  class DiscretizationReader;
  class DiscretizationWriter;
}  // namespace Core::IO

namespace Mortar
{
  class Element;
  class Interface;
  class Node;
  class StrategyBase;

  namespace STRATEGY
  {
    /*! \brief Base class for mortar meshtying/contact factories
     *
     */
    class Factory
    {
     public:
      /**
       * Virtual destructor.
       */
      virtual ~Factory() = default;

      //! constructor
      Factory();

      //! initialization of class variables
      void init(std::shared_ptr<Core::FE::Discretization> dis);

      /*! \brief Setup of class variables
       *
       * \note Since this is an abstract class, the setup flag stays false. It has to be set by
       * the Setup routing of the derived class.
       */

      virtual void setup(int dim);

      /*! \brief Create the desired search tree object
       *
       * We loop over all \c interfaces and build the search tree for every interface.
       *
       * @param[in] interfaces All meshtying interfaces
       */
      void build_search_tree(
          const std::vector<std::shared_ptr<Mortar::Interface>>& interfaces) const;

      /*! \brief Check the problem dimension
       *
       * Mortar meshtying/contact problems are implemented for 2D and 3D only.
       * Throw an error in case of any other spatial dimension.
       */
      void check_dimension() const;

     protected:
      /*! \brief Set #issetup_ flag with user-given value
       *
       * @param[in] issetup Setup flag (default = true)
       */
      inline void set_is_setup(const bool issetup = true) { issetup_ = issetup; };

      //! Returns true, if init() has been called
      inline const bool& is_init() const { return isinit_; };

      //! Returns true, if setup() has been called
      inline const bool& is_setup() const { return issetup_; };

      //! Checks, if init() and setup() have been called
      void check_init_setup() const;

      //! Checks if init() has been called
      void check_init() const;

      //! @name NURBS related stuff
      //!@{

      /*! \brief Prepare mortar element for NURBS case
       *
       *  Stores knot vector, zerosized information and normal factor
       *
       *  \author Farah */
      void prepare_nurbs_element(const Core::FE::Discretization& discret,
          std::shared_ptr<Core::Elements::Element> ele, Mortar::Element& cele) const;

      /*! \brief Prepare mortar node for NURBS case
       *
       *  Stores control point weight
       *
       *  \author Farah */
      void prepare_nurbs_node(const Core::Nodes::Node* node, Mortar::Node& mnode) const;

      //!@}

      //! @name Accessors
      //!@{

      //! Returns the (structural) discretization
      Core::FE::Discretization& discret();
      const Core::FE::Discretization& discret() const;

      //! returns a reference to a copy of the structural communicator
      MPI_Comm get_comm() const;

      //! returns the problem dimension
      const int& n_dim() const;

      //!@}

      //! pointer to the structural problem discretization
      std::shared_ptr<Core::FE::Discretization> discret_ptr_;

     private:
      //! @name Status flags
      //!@{

      //! init flag
      bool isinit_;

      //! setup flag
      bool issetup_;

      //!@}
      //! pointer to a COPY of the structural communicator
      MPI_Comm comm_;

      int dim_;
    };  // namespace STRATEGY
  }  // namespace STRATEGY
}  // namespace Mortar

FOUR_C_NAMESPACE_CLOSE

#endif
