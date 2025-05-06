// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_ele_boundary_calc.hpp"
#include "4C_fluid_ele_boundary_factory.hpp"
#include "4C_fluid_ele_poro.hpp"
#include "4C_inpar_fluid.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

int Discret::Elements::FluidPoroBoundary::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  // get the action required
  const auto act = Teuchos::getIntegralValue<FLD::BoundaryAction>(params, "action");

  // switch between different physical types as used below
  std::string impltype = "poro";
  switch (params.get<Inpar::FLUID::PhysicalType>("Physical Type", Inpar::FLUID::poro))
  {
    case Inpar::FLUID::poro:
      impltype = "poro";
      break;
    case Inpar::FLUID::poro_p1:
      impltype = "poro_p1";
      break;
    default:
      FOUR_C_THROW("invalid physical type for porous fluid!");
      break;
  }

  switch (act)
  {
    case FLD::calc_flowrate:
    case FLD::no_penetration:
    case FLD::no_penetrationIDs:
    case FLD::poro_boundary:
    case FLD::poro_prescoupl:
    case FLD::poro_splitnopenetration:
    case FLD::poro_splitnopenetration_OD:
    case FLD::poro_splitnopenetration_ODdisp:
    case FLD::poro_splitnopenetration_ODpres:
    case FLD::fpsi_coupling:
    {
      Discret::Elements::FluidBoundaryFactory::provide_impl(shape(), impltype)
          ->evaluate_action(
              this, params, discretization, lm, elemat1, elemat2, elevec1, elevec2, elevec3);
      break;
    }
    default:  // call standard fluid boundary element
    {
      FluidBoundary::evaluate(
          params, discretization, lm, elemat1, elemat2, elevec1, elevec2, elevec3);
      break;
    }
  }

  return 0;
}

void Discret::Elements::FluidPoroBoundary::location_vector(const Core::FE::Discretization& dis,
    Core::Elements::LocationArray& la, const std::string& condstring,
    Teuchos::ParameterList& params) const
{
  // get the action required
  const auto act = Teuchos::getIntegralValue<FLD::BoundaryAction>(params, "action");
  switch (act)
  {
    case FLD::poro_boundary:
    case FLD::fpsi_coupling:
    case FLD::calc_flowrate:
    case FLD::poro_splitnopenetration_ODdisp:
      // special cases: the boundary element assembles also into
      // the inner dofs of its parent element
      // note: using these actions, the element will get the parent location vector
      parent_element()->location_vector(dis, la);
      break;
    case FLD::poro_splitnopenetration:
    case FLD::poro_splitnopenetration_OD:
    {
      // This is a hack ...
      // Remove pressure dofs from location vector!!!
      // call standard fluid boundary element
      FluidBoundary::location_vector(dis, la, condstring, params);
      int dim = la[0].stride_[0] - 1;
      // extract velocity dofs from first dofset
      for (int i = num_node(); i > 0; --i)
      {
        la[0].lm_.erase(la[0].lm_.begin() + (i - 1) * (dim + 1) + dim);
        la[0].lmowner_.erase(la[0].lmowner_.begin() + (i - 1) * (dim + 1) + dim);
        la[0].stride_[i - 1] = dim;
      }
    }
    break;
    case FLD::boundary_none:
      FOUR_C_THROW("No action supplied");
      break;
    default:
      // call standard fluid boundary element
      FluidBoundary::location_vector(dis, la, condstring, params);
      break;
  }
}

FOUR_C_NAMESPACE_CLOSE
