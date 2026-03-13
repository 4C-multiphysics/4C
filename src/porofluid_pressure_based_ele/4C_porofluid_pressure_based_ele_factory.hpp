// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_ELE_FACTORY_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_ELE_FACTORY_HPP


#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Elements
  {
    // forward declaration
    class PoroFluidMultiPhaseEleInterface;

    class PoroFluidMultiPhaseFactory
    {
     public:
      //! ctor
      PoroFluidMultiPhaseFactory() { return; }

      //! dtor
      virtual ~PoroFluidMultiPhaseFactory() = default;
      //! ProvideImpl
      static PoroFluidMultiPhaseEleInterface* provide_impl(Core::FE::CellType distype,
          int spatial_dimension, const int numdofpernode, const std::string& disname);

     private:
      //! define PoroFluidMultiPhaseEle instances dependent on problem
      template <Core::FE::CellType distype>
      static PoroFluidMultiPhaseEleInterface* define_problem_type(
          const int numdofpernode, const std::string& disname);


    };  // end class PoroFluidMultiPhaseFactory

  }  // namespace Elements

}  // namespace Discret



FOUR_C_NAMESPACE_CLOSE

#endif
