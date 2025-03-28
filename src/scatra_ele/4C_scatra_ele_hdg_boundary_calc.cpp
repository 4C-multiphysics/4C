// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_hdg_boundary_calc.hpp"

#include "4C_fem_general_node.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_ele_hdg.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Discret::Elements::ScaTraHDGBoundaryImplInterface*
Discret::Elements::ScaTraHDGBoundaryImplInterface::impl(const Core::Elements::Element* ele)
{
  switch (ele->shape())
  {
    case Core::FE::CellType::quad4:
    {
      return ScaTraHDGBoundaryImpl<Core::FE::CellType::quad4>::instance();
    }
    case Core::FE::CellType::quad8:
    {
      return ScaTraHDGBoundaryImpl<Core::FE::CellType::quad8>::instance();
    }
    case Core::FE::CellType::quad9:
    {
      return ScaTraHDGBoundaryImpl<Core::FE::CellType::quad9>::instance();
    }
    case Core::FE::CellType::tri3:
    {
      return ScaTraHDGBoundaryImpl<Core::FE::CellType::tri3>::instance();
    }
    case Core::FE::CellType::tri6:
    {
      return ScaTraHDGBoundaryImpl<Core::FE::CellType::tri6>::instance();
    }
    case Core::FE::CellType::line2:
    {
      return ScaTraHDGBoundaryImpl<Core::FE::CellType::line2>::instance();
    }
    case Core::FE::CellType::line3:
    {
      return ScaTraHDGBoundaryImpl<Core::FE::CellType::line3>::instance();
    }
    case Core::FE::CellType::nurbs2:  // 1D nurbs boundary element
    {
      return ScaTraHDGBoundaryImpl<Core::FE::CellType::nurbs2>::instance();
    }
    case Core::FE::CellType::nurbs3:  // 1D nurbs boundary element
    {
      return ScaTraHDGBoundaryImpl<Core::FE::CellType::nurbs3>::instance();
    }
    case Core::FE::CellType::nurbs4:  // 2D nurbs boundary element
    {
      return ScaTraHDGBoundaryImpl<Core::FE::CellType::nurbs4>::instance();
    }
    case Core::FE::CellType::nurbs9:  // 2D nurbs boundary element
    {
      return ScaTraHDGBoundaryImpl<Core::FE::CellType::nurbs9>::instance();
    }
    default:
      FOUR_C_THROW(
          "Element shape {} ({} nodes) not activated. Just do it.", ele->shape(), ele->num_node());
      break;
  }
  return nullptr;
}

template <Core::FE::CellType distype>
Discret::Elements::ScaTraHDGBoundaryImpl<distype>*
Discret::Elements::ScaTraHDGBoundaryImpl<distype>::instance(Core::Utils::SingletonAction action)
{
  static auto singleton_owner = Core::Utils::make_singleton_owner(
      []()
      {
        return std::unique_ptr<Discret::Elements::ScaTraHDGBoundaryImpl<distype>>(
            new Discret::Elements::ScaTraHDGBoundaryImpl<distype>());
      });

  return singleton_owner.instance(action);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::ScaTraHDGBoundaryImpl<distype>::ScaTraHDGBoundaryImpl()
    : xyze_(Core::LinAlg::Initialization::zero),
      funct_(Core::LinAlg::Initialization::zero),
      deriv_(Core::LinAlg::Initialization::zero),
      unitnormal_(Core::LinAlg::Initialization::zero),
      velint_(Core::LinAlg::Initialization::zero),
      drs_(0.0),
      fac_(0.0)
{
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::ScaTraHDGBoundaryImpl<distype>::evaluate_neumann(
    Discret::Elements::ScaTraHDGBoundary* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra)
{
  Core::LinAlg::SerialDenseVector dummy_vec2, dummy_vec3;
  Core::LinAlg::SerialDenseMatrix dummy_mat2;

  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::project_neumann_field, params);

  const int* nodeids = ele->node_ids();

  Core::Elements::Element* parent = ele->parent_element();
  std::shared_ptr<Core::Elements::FaceElement>* faces = parent->faces();
  bool same = false;
  for (int i = 0; i < parent->num_face(); ++i)
  {
    const int* nodeidsfaces = faces[i]->node_ids();

    if (faces[i]->num_node() != ele->num_node()) break;

    for (int j = 0; j < ele->num_node(); ++j)
    {
      if (nodeidsfaces[j] == nodeids[j])
        same = true;
      else
      {
        same = false;
        break;
      }
    }
    if (same == true)
    {
      // i is the number we were searching for!!!!
      params.set<int>("face", i);
      ele->parent_element()->evaluate(params, discretization, la, elemat1_epetra, dummy_mat2,
          elevec1_epetra, dummy_vec2, dummy_vec3);
      // break;
    }
  }
  if (same == false && (faces[0]->num_node() != ele->num_node()))
    FOUR_C_THROW("Neumann boundary condition implemented only for surface elements");

  return 0;
}

FOUR_C_NAMESPACE_CLOSE
