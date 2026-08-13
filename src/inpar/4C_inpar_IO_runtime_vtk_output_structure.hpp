// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_INPAR_IO_RUNTIME_VTK_OUTPUT_STRUCTURE_HPP
#define FOUR_C_INPAR_IO_RUNTIME_VTK_OUTPUT_STRUCTURE_HPP


/*----------------------------------------------------------------------*/
/* headers */
#include "4C_config.hpp"

#include "4C_io_input_spec.hpp"
#include "4C_structure_new_input.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
namespace Inpar
{
  namespace IORuntimeOutput
  {
    namespace Solid
    {
      using GaussPointDataOutputType = FourC::Solid::GaussPointDataOutputType;
      using OptQuantityType = FourC::Solid::OptQuantityType;
      using FourC::Solid::optquantity_membranethickness;
      using FourC::Solid::optquantity_none;
      using FourC::Solid::optquantity_porosity;
      using FourC::Solid::optquantity_shell7pthickness;
      using FourC::Solid::optquantity_shell7pthicknessdirector;

      /// valid parameters related to writing of VTK output at runtime
      Core::IO::InputSpec valid_parameters();

    }  // namespace Solid
  }  // namespace IORuntimeOutput
}  // namespace Inpar

FOUR_C_NAMESPACE_CLOSE

#endif
