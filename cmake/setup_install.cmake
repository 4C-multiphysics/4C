# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

##
# Append configuration for a package to FOUR_C_ACTIVATED_DEPENDENCIES_CONFIG
##
macro(_add_dependency_to_config _package_name)
  # Sanitize the package name: all upper case, no hyphens and dots.
  string(TOUPPER ${_package_name} _package_name_SANITIZED)
  string(REGEX REPLACE "[^A-Z0-9]" "_" _package_name_SANITIZED ${_package_name_SANITIZED})

  string(
    APPEND
    FOUR_C_ACTIVATED_DEPENDENCIES_CONFIG
    "set(FOUR_C_WITH_${_package_name_SANITIZED} ${FOUR_C_WITH_${_package_name_SANITIZED}})\n"
    )

  # Append the dependency info to the target settings
  if(FOUR_C_WITH_${_package_name_SANITIZED})
    # Append a tab at the start of each line of content
    file(READ ${PROJECT_BINARY_DIR}/cmake/templates/${_package_name}.cmake _content)
    string(REPLACE "\n" "\n\t" _content_with_tab "${_content}")
    string(
      APPEND
      FOUR_C_ACTIVATED_DEPENDENCIES_CONFIG
      "\nif(FOUR_C_WITH_${_package_name_SANITIZED})\n\t"
      )
    string(APPEND FOUR_C_ACTIVATED_DEPENDENCIES_CONFIG "${_content_with_tab}")
    string(APPEND FOUR_C_ACTIVATED_DEPENDENCIES_CONFIG "\nendif()\n")
  endif()
endmacro()

include(GNUInstallDirs)

# install the 4C executable
install(
  TARGETS ${FOUR_C_EXECUTABLE_NAME}
  EXPORT 4CTargets
  RUNTIME
  )

# add include libraries to 4C::lib4C
target_include_directories(
  ${FOUR_C_LIBRARY_NAME} INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

# install the targets for 4C::lib4C
install(
  TARGETS ${FOUR_C_LIBRARY_NAME}
  EXPORT 4CTargets
  ARCHIVE
  LIBRARY
  )

# install the targets for 4C dependencies
install(
  TARGETS four_c_all_enabled_external_dependencies
  EXPORT 4CTargets
  ARCHIVE
  LIBRARY
  )

# export the 4C targets
install(
  EXPORT 4CTargets
  NAMESPACE 4C::
  DESTINATION ${CMAKE_INSTALL_DATADIR}/cmake/4C
  )

# add the dependency info to settings
_add_dependency_to_config(HDF5)
_add_dependency_to_config(MPI)
_add_dependency_to_config(Qhull)
_add_dependency_to_config(Trilinos)
_add_dependency_to_config(VTK)
_add_dependency_to_config(deal.II)
_add_dependency_to_config(Boost)
_add_dependency_to_config(ArborX)
_add_dependency_to_config(FFTW)
_add_dependency_to_config(CLN)
_add_dependency_to_config(MIRCO)
_add_dependency_to_config(Backtrace)
_add_dependency_to_config(ryml)
_add_dependency_to_config(magic_enum)
_add_dependency_to_config(ZLIB)
_add_dependency_to_config(pybind11)

# install the Find modules
install(
  DIRECTORY ${PROJECT_SOURCE_DIR}/cmake/modules/
  DESTINATION ${CMAKE_INSTALL_DATADIR}/cmake/4C/modules
  )

# create and install the config file
include(CMakePackageConfigHelpers)
set(FOUR_C_VERSION_STRING "${FOUR_C_VERSION_MAJOR}.${FOUR_C_VERSION_MINOR}")
configure_package_config_file(
  cmake/templates/4CConfig.cmake.in ${PROJECT_BINARY_DIR}/cmake/templates/4CConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_DATADIR}/cmake/4C
  )
write_basic_package_version_file(
  ${PROJECT_BINARY_DIR}/cmake/templates/4CConfigVersion.cmake
  VERSION ${FOUR_C_VERSION_STRING}
  COMPATIBILITY ExactVersion
  )

install(
  FILES ${PROJECT_BINARY_DIR}/cmake/templates/4CConfig.cmake
        ${PROJECT_BINARY_DIR}/cmake/templates/4CConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_DATADIR}/cmake/4C
  )
