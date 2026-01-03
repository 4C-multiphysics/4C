# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

message(STATUS "Fetch content for Qhull")

set(QHULL_ENABLE_TESTING "OFF")
set(BUILD_APPLICATIONS "OFF")
set(BUILD_STATIC_LIBS "OFF")
set(LINK_APPS_SHARED "OFF")
set(BUILD_SHARED_LIBS "ON")

fetchcontent_declare(
  libqhull
  GIT_REPOSITORY https://github.com/qhull/qhull.git
  GIT_TAG d1c2fc0caa5f644f3a0f220290d4a868c68ed4f6
  )

fetchcontent_makeavailable(libqhull)
set(FOUR_C_QHULL_ROOT "${CMAKE_INSTALL_PREFIX}")

four_c_add_external_dependency(four_c_all_enabled_external_dependencies qhull_r)

four_c_remember_variable_for_install(FOUR_C_QHULL_ROOT)
