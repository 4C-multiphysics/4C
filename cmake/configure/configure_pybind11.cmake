# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

if(NOT FOUR_C_WITH_PYTHON)
  message(FATAL_ERROR "FOUR_C_WITH_PYBIND11 requires FOUR_C_WITH_PYTHON to be enabled.")
endif()
if(NOT FOUR_C_PYTHON_ROOT)
  message(
    STATUS
      "FOUR_C_PYTHON_ROOT is not set. It can be risky that pybind11 picks up another Python version other than the one found by configure_Python.cmake. This can create inconsistencies and fail some tests."
    )
else()
  set(Python_ROOT ${FOUR_C_PYTHON_ROOT})
endif()

# Hand off the exact interpreter 4C found to FindPython3 (used by pybind11)
set(Python3_EXECUTABLE
    "${Python_EXECUTABLE}"
    CACHE FILEPATH "" FORCE
    )
set(Python3_FIND_STRATEGY LOCATION)
if(APPLE)
  set(Python3_FIND_FRAMEWORK LAST)
endif()

message(STATUS "Fetch content for pybind11")
fetchcontent_declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG 97bf890db679505a14dfe547a5e77bb2bd05dc90 # v3.1.0
  )
set(PYBIND11_FINDPYTHON NEW)
set(PYBIND11_INSTALL ON)
fetchcontent_makeavailable(pybind11)

set(FOUR_C_PYBIND11_ROOT "${CMAKE_INSTALL_PREFIX}")

four_c_add_external_dependency(
  four_c_all_enabled_external_dependencies pybind11::module pybind11::embed
  )

four_c_remember_variable_for_install(FOUR_C_PYBIND11_ROOT)
