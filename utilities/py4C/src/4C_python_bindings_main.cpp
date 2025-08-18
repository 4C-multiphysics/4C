// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config_submodule_registry.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

FOUR_C_NAMESPACE_OPEN

PYBIND11_MODULE(py4C, module)
{
  module.doc() = "Python bindings for 4C";

  // initialize all submodules
  for (const auto& entry : Py4C::get_submodule_registry())
  {
    auto submodule = module.def_submodule(entry.name.c_str(), entry.doc.c_str());
    entry.init_funct(submodule);
  }
}
FOUR_C_NAMESPACE_CLOSE