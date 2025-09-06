// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GENERAL_ELEMENTTYPE_HPP
#define FOUR_C_FEM_GENERAL_ELEMENTTYPE_HPP

#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_io_input_spec.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::LinAlg
{
  class SparseOperator;
}

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE
/// Subclass of ParObjectType that adds element type specific methods
/*!
  Element types need to be initialized. Furthermore, there is a pre_evaluate
  method and the ability to read elements from input files. And finally the
  element specific setup of null spaces for multi grid preconditioning is
  here, too.

  \note There are boundary elements that do not need all of this
  functionality.

 */

namespace Core::Elements
{
  class ElementType : public Core::Communication::ParObjectType
  {
   protected:
    // only derived classes might create an instance
    ElementType();

   public:
    /// setup the input file input line definitions for this type of element
    virtual void setup_element_definition(
        std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
    {
    }

    /// create an element from an input file specifier
    virtual std::shared_ptr<Element> create(
        const std::string eletype, const std::string eledistype, const int id, const int owner)
    {
      return nullptr;
    }

    /// create an empty element
    virtual std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) = 0;

    /// initialize the element type
    virtual int initialize(Core::FE::Discretization& dis);

    /*!
    \brief Get nodal block information to create a null space description

    \note All elements will fill \c nv, but \c np is only filled by some elements.

    @param[in] dwele Element pointer
    @param[out] numdf Number of degrees of freedom per node
    @param[out] dimns Nullspace dimension
    */
    virtual void nodal_block_information(
        Core::Elements::Element* dwele, int& numdf, int& dimns) = 0;

    /// do the null space computation
    virtual Core::LinAlg::SerialDenseMatrix compute_null_space(
        Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) = 0;
  };

}  // namespace Core::Elements

FOUR_C_NAMESPACE_CLOSE

#endif
