// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solid_scatra_ele_lib.hpp"

#include "4C_io_input_parameter_container.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

ScaTra::ImplType Discret::Elements::read_scatra_impl_type(
    const Core::IO::InputParameterContainer& container)
{
  auto impltype = container.get<std::string>("TYPE");

  if (impltype == "Undefined")
    return ScaTra::impltype_undefined;
  else if (impltype == "AdvReac")
    return ScaTra::impltype_advreac;
  else if (impltype == "CardMono")
    return ScaTra::impltype_cardiac_monodomain;
  else if (impltype == "GR")
    return ScaTra::impltype_gr;
  else if (impltype == "NLS")
    return ScaTra::impltype_nl_stimulus;
  else if (impltype == "Chemo")
    return ScaTra::impltype_chemo;
  else if (impltype == "ChemoReac")
    return ScaTra::impltype_chemoreac;
  else if (impltype == "ElchDiffCond")
    return ScaTra::impltype_elch_diffcond;
  else if (impltype == "ElchElectrode")
    return ScaTra::impltype_elch_electrode;
  else if (impltype == "Loma")
    return ScaTra::impltype_loma;
  else if (impltype == "Std")
    return ScaTra::impltype_std;

  FOUR_C_THROW("The input type {} is not valid for SOLIDSCATRA elements!", impltype);
}

FOUR_C_NAMESPACE_CLOSE
