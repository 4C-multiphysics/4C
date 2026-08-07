# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# This function unsets some properties to prevent the kokkos launch compiler from being used on a ${target}, and sets the clangcuda mode for the clangcuda++ compiler wrapper.
# clangcuda_mode can be either CLANGCUDA_MODE_HOST or CLANGCUDA_MODE_DEVICE
function(set_clangcuda_mode target clangcuda_mode)
  set_target_properties(
    ${target}
    PROPERTIES CXX_COMPILER_LAUNCHER ""
               C_COMPILER_LAUNCHER ""
               CUDA_COMPILER_LAUNCHER ""
               RULE_LAUNCH_COMPILE ""
               RULE_LAUNCH_LINK ""
    )
  target_compile_definitions(${target} PRIVATE ${clangcuda_mode})
endfunction()
