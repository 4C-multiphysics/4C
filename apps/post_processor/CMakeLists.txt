# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

set(OBJS_POST_PROCESSOR_MAIN
    # cmake-format: sortable
    4C_post_processor.cpp
    4C_post_processor_single_field_writers.cpp
    4C_post_processor_structure_stress.cpp
    )

add_executable(post_processor EXCLUDE_FROM_ALL ${OBJS_POST_PROCESSOR_MAIN})
set_target_properties(post_processor PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
four_c_set_up_executable(post_processor)

# various post-processing tools
include(scripts/create_post_scripts.cmake)
