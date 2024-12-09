# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

set(TESTNAME unittests_stl_extension)

set(SOURCE_LIST
    4C_utils_demangle_test.cpp 4C_utils_lazy_ptr_test.cpp 4C_utils_singleton_owner_test.cpp
    )

four_c_add_google_test_executable(
  ${TESTNAME}
  NP
  1
  SOURCE
  ${SOURCE_LIST}
  )
