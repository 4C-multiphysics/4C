// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_IO_runtime_vtp_output_structure.hpp"

#include "4C_io_geometry_type.hpp"
#include "4C_utils_parameter_list.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Inpar
{
  namespace IORuntimeVTPStructure
  {
    /*----------------------------------------------------------------------*
     *----------------------------------------------------------------------*/
    void set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
    {
      using Teuchos::tuple;

      // related sublist
      Core::Utils::SectionSpecs sublist_IO{"IO"};
      Core::Utils::SectionSpecs sublist_IO_VTP_structure{
          sublist_IO, "RUNTIME VTP OUTPUT STRUCTURE"};


      // output interval regarding steps: write output every INTERVAL_STEPS steps
      Core::Utils::int_parameter("INTERVAL_STEPS", -1,
          "write VTP output at runtime every INTERVAL_STEPS steps", sublist_IO_VTP_structure);

      Core::Utils::int_parameter("STEP_OFFSET", 0,
          "An offset added to the current step to shift the steps to be written.",
          sublist_IO_VTP_structure);

      // whether to write output in every iteration of the nonlinear solver
      Core::Utils::bool_parameter("EVERY_ITERATION", false,
          "write output in every iteration of the nonlinear solver", sublist_IO_VTP_structure);

      // write owner at every visualization point
      Core::Utils::bool_parameter(
          "OWNER", false, "write owner of every point", sublist_IO_VTP_structure);

      // write orientation at every visualization point
      Core::Utils::bool_parameter("ORIENTATIONANDLENGTH", false, "write orientation at every point",
          sublist_IO_VTP_structure);

      // write number of bonds at every visualization point
      Core::Utils::bool_parameter(
          "NUMBEROFBONDS", false, "write number of bonds of every point", sublist_IO_VTP_structure);

      // write force actin in linker
      Core::Utils::bool_parameter(
          "LINKINGFORCE", false, "write force acting in linker", sublist_IO_VTP_structure);

      sublist_IO_VTP_structure.move_into_collection(list);
    }


  }  // namespace IORuntimeVTPStructure
}  // namespace Inpar

FOUR_C_NAMESPACE_CLOSE
