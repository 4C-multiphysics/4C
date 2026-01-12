# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

message(STATUS "Fetch content for Qhull")

cmake_policy(SET CMP0077 NEW)

set(QHULL_ENABLE_TESTING "OFF")
set(BUILD_APPLICATIONS "OFF")
set(BUILD_STATIC_LIBS "OFF")
set(LINK_APPS_SHARED "OFF")
set(BUILD_SHARED_LIBS "ON")

fetchcontent_declare(
  libqhull
  GIT_REPOSITORY https://github.com/qhull/qhull.git
  GIT_TAG a22c735d6a8d1b5eac5773790aeae28f3b088655 #v8.1-alpha1
  )

fetchcontent_makeavailable(libqhull)
set(FOUR_C_QHULL_ROOT "${CMAKE_INSTALL_PREFIX}")

four_c_add_external_dependency(four_c_all_enabled_external_dependencies qhull_r)

four_c_remember_variable_for_install(FOUR_C_QHULL_ROOT)
