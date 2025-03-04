# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# At this point we deliberately turned on the option to use deal.II, so fail if it is not available.
find_package(deal.II REQUIRED)

if(NOT DEAL_II_WITH_TRILINOS)
  message(
    FATAL_ERROR
      "deal.II was not compiled with Trilinos support. Please recompile deal.II with Trilinos support."
    )
endif()

# N.B. We do NOT use dealii::dealii here, since this pulls in a lot of flags that we might not want.
# Instead, deal.II potentially provides two targets: dealii::dealii_release and/or dealii::dealii_debug.
if(TARGET dealii::dealii_release)
  message(STATUS "Using target dealii::dealii_release")
  target_link_libraries(four_c_all_enabled_external_dependencies INTERFACE dealii::dealii_release)
elseif(TARGET dealii::dealii_debug)
  message(STATUS "Using target dealii::dealii_debug")
  target_link_libraries(four_c_all_enabled_external_dependencies INTERFACE dealii::dealii_debug)
else()
  message(FATAL_ERROR "Could not find target dealii::dealii_release or dealii::dealii_debug.")
endif()

include(${DEAL_II_GIT_CONFIG})
set(FOUR_C_deal.II_GIT_HASH "${DEAL_II_GIT_REVISION}")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/deal.II.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake/templates/deal.II.cmake"
  @ONLY
  )
