// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_binstrategy_meshfree_bin.hpp"

FOUR_C_NAMESPACE_OPEN

/*--------------------------------------------------------------------------*
 |  ctor                                               (public) ghamm 11/12 |
 *--------------------------------------------------------------------------*/
template <typename Element>
Core::FE::MeshFree::MeshfreeBin<Element>::MeshfreeBin(int id, int owner) : Element(id, owner)
{
}

/*--------------------------------------------------------------------------*
 |  copy-ctor                                          (public) ghamm 11/12 |
 *--------------------------------------------------------------------------*/
template <typename Element>
Core::FE::MeshFree::MeshfreeBin<Element>::MeshfreeBin(
    const Core::FE::MeshFree::MeshfreeBin<Element>& old)
    : Element(old)
{
}


/*--------------------------------------------------------------------------*
 | Delete a single node from the element               (public) ghamm 11/12 |
 *--------------------------------------------------------------------------*/
template <typename Element>
void Core::FE::MeshFree::MeshfreeBin<Element>::delete_node(int gid)
{
  for (unsigned int i = 0; i < Element::nodeids_.size(); i++)
  {
    if (Element::nodeids_[i] == gid)
    {
      Element::nodeids_.erase(Element::nodeids_.begin() + i);
      Element::nodes_.erase(Element::nodes_.begin() + i);
      return;
    }
  }
  FOUR_C_THROW("Connectivity issues: No node with specified gid to delete in element. ");
}

/*--------------------------------------------------------------------------*
 | Explicit instantiations                                kronbichler 03/15 |
 *--------------------------------------------------------------------------*/
template class Core::FE::MeshFree::MeshfreeBin<Core::Elements::Element>;
template class Core::FE::MeshFree::MeshfreeBin<Core::Elements::FaceElement>;

FOUR_C_NAMESPACE_CLOSE
