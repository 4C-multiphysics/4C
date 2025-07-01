// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_ART_NET_ARTERY_ELE_FACTORY_HPP
#define FOUR_C_ART_NET_ARTERY_ELE_FACTORY_HPP

#include "4C_config.hpp"

#include "4C_art_net_artery_ele_interface.hpp"
#include "4C_art_net_input.hpp"
#include "4C_fem_general_element.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Elements
  {
    // forward declaration
    class ArteryEleInterface;

    class ArtNetFactory
    {
     public:
      //! ctor
      ArtNetFactory() { return; }

      //! dtor
      virtual ~ArtNetFactory() = default;
      //! ProvideImpl
      static ArteryEleInterface* provide_impl(
          Core::FE::CellType distype, ArtDyn::ImplType problem, const std::string& disname);

     private:
      //! define ArteryEle instances dependent on problem
      template <Core::FE::CellType distype>
      static ArteryEleInterface* define_problem_type(
          ArtDyn::ImplType problem, const std::string& disname);


    };  // end class ArtNetFactory

  }  // namespace Elements

}  // namespace Discret



FOUR_C_NAMESPACE_CLOSE

#endif
