# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

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

install(
  FILES ${CMAKE_BINARY_DIR}/cmake/templates/4CSettings.cmake
  DESTINATION ${CMAKE_INSTALL_DATADIR}/cmake/4C
  )

# create and install the config file
include(CMakePackageConfigHelpers)
set(4C_VERSION_STRING "${FOUR_C_VERSION_MAJOR}.${FOUR_C_VERSION_MINOR}")
configure_package_config_file(
  cmake/templates/4CConfig.cmake.in ${CMAKE_BINARY_DIR}/cmake/templates/4CConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_DATADIR}/cmake/4C
  )
write_basic_package_version_file(
  ${CMAKE_BINARY_DIR}/cmake/templates/4CConfigVersion.cmake
  VERSION ${4C_VERSION_STRING}
  COMPATIBILITY AnyNewerVersion
  )

install(
  FILES ${CMAKE_BINARY_DIR}/cmake/templates/4CConfig.cmake
        ${CMAKE_BINARY_DIR}/cmake/templates/4CConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_DATADIR}/cmake/4C
  )

# create the test install script
set(FOURC_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/cmake)
configure_file(
  ${CMAKE_SOURCE_DIR}/tests/test_install.sh.in ${CMAKE_BINARY_DIR}/test_install.sh @ONLY
  )
