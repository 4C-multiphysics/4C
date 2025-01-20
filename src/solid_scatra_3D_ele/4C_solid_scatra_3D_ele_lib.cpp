// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solid_scatra_3D_ele_lib.hpp"

#include "4C_io_input_parameter_container.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

Inpar::ScaTra::ImplType Discret::Elements::read_scatra_impl_type(
    const Core::IO::InputParameterContainer& container)
{
  auto impltype = container.get<std::string>("TYPE");

  if (impltype == "Undefined")
    return Inpar::ScaTra::impltype_undefined;
  else if (impltype == "AdvReac")
    return Inpar::ScaTra::impltype_advreac;
  else if (impltype == "CardMono")
    return Inpar::ScaTra::impltype_cardiac_monodomain;
  else if (impltype == "Chemo")
    return Inpar::ScaTra::impltype_chemo;
  else if (impltype == "ChemoReac")
    return Inpar::ScaTra::impltype_chemoreac;
  else if (impltype == "ElchDiffCond")
    return Inpar::ScaTra::impltype_elch_diffcond;
  else if (impltype == "ElchElectrode")
    return Inpar::ScaTra::impltype_elch_electrode;
  else if (impltype == "Loma")
    return Inpar::ScaTra::impltype_loma;
  else if (impltype == "RefConcReac")
    return Inpar::ScaTra::impltype_refconcreac;
  else if (impltype == "Std")
    return Inpar::ScaTra::impltype_std;

  FOUR_C_THROW("The input type %s is not valid for SOLIDSCATRA elements!", impltype.c_str());
}

FOUR_C_NAMESPACE_CLOSE
