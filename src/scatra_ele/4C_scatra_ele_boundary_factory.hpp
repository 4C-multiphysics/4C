// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_ELE_BOUNDARY_FACTORY_HPP
#define FOUR_C_SCATRA_ELE_BOUNDARY_FACTORY_HPP

#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_inpar_scatra.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Elements
  {
    class ScaTraBoundaryInterface;

    class ScaTraBoundaryFactory
    {
     public:
      //! ctor
      ScaTraBoundaryFactory() { return; };

      //! dtor
      virtual ~ScaTraBoundaryFactory() = default;

      //! ProvideImpl
      static ScaTraBoundaryInterface* provide_impl(const Core::Elements::Element* ele,
          const enum Inpar::ScaTra::ImplType impltype, const int numdofpernode, const int numscal,
          const std::string& disname);

     private:
      //! return instance of element evaluation class depending on implementation type
      template <Core::FE::CellType distype, int probdim>
      static ScaTraBoundaryInterface* define_problem_type(
          const enum Inpar::ScaTra::ImplType impltype, const int numdofpernode, const int numscal,
          const std::string& disname);
    };  // class ScaTraBoundaryFactory
  }  // namespace Elements
}  // namespace Discret
FOUR_C_NAMESPACE_CLOSE

#endif
