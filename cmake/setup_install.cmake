# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

function(_add_dependency_to_settings _package_name)
  # Sanitize the package name: all upper case, no hyphens and dots.
  string(TOUPPER ${_package_name} _package_name_SANITIZED)
  string(REGEX REPLACE "[^A-Z0-9]" "_" _package_name_SANITIZED ${_package_name_SANITIZED})

  # Append the dependency info to the target settings
  if(FOUR_C_WITH_${_package_name_SANITIZED})
    # Append a tab at the start of each line of content
    file(READ ${CMAKE_BINARY_DIR}/cmake/templates/${_package_name}.cmake _content)
    string(REPLACE "\n" "\n\t" _content_with_tab "${_content}")
    file(
      APPEND ${CMAKE_BINARY_DIR}/cmake/templates/4CSettings.cmake
      "\nif(FOUR_C_WITH_${_package_name_SANITIZED})\n\t"
      )
    file(APPEND ${CMAKE_BINARY_DIR}/cmake/templates/4CSettings.cmake "${_content_with_tab}")
    file(APPEND ${CMAKE_BINARY_DIR}/cmake/templates/4CSettings.cmake "\nendif()\n")
  endif()
endfunction()

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

# create the settings file
configure_file(
  ${CMAKE_SOURCE_DIR}/cmake/templates/4CSettings.cmake.in
  ${CMAKE_BINARY_DIR}/cmake/templates/4CSettings.cmake
  @ONLY
  )

# add the dependency info to settings
_add_dependency_to_settings(HDF5)
_add_dependency_to_settings(MPI)
_add_dependency_to_settings(Qhull)
_add_dependency_to_settings(Trilinos)
_add_dependency_to_settings(Boost)
_add_dependency_to_settings(ArborX)
_add_dependency_to_settings(FFTW)
_add_dependency_to_settings(CLN)
_add_dependency_to_settings(MIRCO)
_add_dependency_to_settings(Backtrace)
_add_dependency_to_settings(ryml)
_add_dependency_to_settings(magic_enum)

# install
install(
  FILES ${CMAKE_BINARY_DIR}/cmake/templates/4CSettings.cmake
  DESTINATION ${CMAKE_INSTALL_DATADIR}/cmake/4C
  )

# create and install the config file
include(CMakePackageConfigHelpers)
set(FOUR_C_VERSION_STRING "${FOUR_C_VERSION_MAJOR}.${FOUR_C_VERSION_MINOR}")
configure_package_config_file(
  cmake/templates/4CConfig.cmake.in ${CMAKE_BINARY_DIR}/cmake/templates/4CConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_DATADIR}/cmake/4C
  )
write_basic_package_version_file(
  ${CMAKE_BINARY_DIR}/cmake/templates/4CConfigVersion.cmake
  VERSION ${FOUR_C_VERSION_STRING}
  COMPATIBILITY ExactVersion
  )

install(
  FILES ${CMAKE_BINARY_DIR}/cmake/templates/4CConfig.cmake
        ${CMAKE_BINARY_DIR}/cmake/templates/4CConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_DATADIR}/cmake/4C
  )
