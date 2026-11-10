# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Finder for Qhull
# Exports qhull::qhull as an imported target
# Note: The Qhull_ROOT variable is automatically considered by the find_ calls below.

find_path(QHULL_INCLUDE_DIR libqhull_r/libqhull_r.h)

find_library(QHULL_LIBRARY NAMES qhull_r)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Qhull DEFAULT_MSG QHULL_LIBRARY QHULL_INCLUDE_DIR)

if(QHULL_FOUND AND NOT TARGET qhull::qhull)
  add_library(qhull::qhull UNKNOWN IMPORTED)
  set_target_properties(
    qhull::qhull
    PROPERTIES IMPORTED_LOCATION "${QHULL_LIBRARY}"
               INTERFACE_INCLUDE_DIRECTORIES "${QHULL_INCLUDE_DIR}"
    )
endif()
