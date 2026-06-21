# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

function(check_trilinos_packages)
  unset(missing_packages)
  message(CHECK_START "Checking required Trilinos packages")
  list(APPEND CMAKE_MESSAGE_INDENT "  ")

  foreach(package_name IN LISTS ARGN)
    message(CHECK_START "Looking for ${package_name}")
    list(FIND Trilinos_PACKAGE_LIST "${package_name}" package_index)

    if(package_index GREATER -1)
      message(CHECK_PASS "found")
    else()
      list(APPEND missing_packages "${package_name}")
      message(CHECK_FAIL "not found")
    endif()
  endforeach()

  list(POP_BACK CMAKE_MESSAGE_INDENT)

  if(missing_packages)
    string(JOIN ", " missing_packages_str ${missing_packages})
    message(CHECK_FAIL "missing packages: ${missing_packages_str}")
    message(
      FATAL_ERROR "FourC requires the following Trilinos package(s): ${missing_packages_str}. "
                  "Please configure Trilinos according to the documentation."
      )
  else()
    message(CHECK_PASS "all required packages found")
  endif()
endfunction()

function(check_trilinos_tpl tpl_var tpl_name output_var)
  message(CHECK_START "Checking optional Trilinos TPL ${tpl_name}")

  if(${tpl_var})
    message(CHECK_PASS "enabled")
    set(${output_var}
        TRUE
        CACHE BOOL "Whether Trilinos was built with ${tpl_name} support" FORCE
        )
  else()
    message(CHECK_FAIL "disabled")
    set(${output_var}
        FALSE
        CACHE BOOL "Whether Trilinos was built with ${tpl_name} support" FORCE
        )
  endif()
endfunction()

# Kokkos is typically pulled in via Trilinos. If no location has been given,
# try the same location as Trilinos. If no Trilinos location exists, users
# will get an error to provide that one first.
set(Kokkos_FIND_QUIETLY TRUE)
if(Trilinos_ROOT AND NOT Kokkos_ROOT)
  set(Kokkos_ROOT
      ${Trilinos_ROOT}
      CACHE PATH "Path to Kokkos installation"
      )
endif()

# We only support Trilinos versions that provide a config file.
find_package(Trilinos REQUIRED)

message(STATUS "Trilinos version: ${Trilinos_VERSION}")
message(STATUS "Trilinos packages: ${Trilinos_PACKAGE_LIST}")

# Figure out the version.
if(EXISTS "${Trilinos_DIR}/../../../TrilinosRepoVersion.txt")
  file(STRINGS "${Trilinos_DIR}/../../../TrilinosRepoVersion.txt" TrilinosRepoVersionFile)
  # The hash is the first token on the second line.
  list(GET TrilinosRepoVersionFile 1 TrilinosRepoVersionFileLine2)
  separate_arguments(TrilinosRepoVersionFileLine2)
  list(GET TrilinosRepoVersionFileLine2 0 _sha)

  set(FOUR_C_Trilinos_GIT_HASH ${_sha})
else()
  set(FOUR_C_Trilinos_GIT_HASH "unknown")
endif()

# Check for some required Trilinos package dependencies
check_trilinos_packages(
  Teuchos
  Thyra
  Epetra
  EpetraExt
  Shards
  Sacado
  Intrepid2
  Zoltan2
  Stratimikos
  Belos
  Amesos2
  Ifpack
  MueLu
  Teko
  NOX
  )

# Check for some required Trilinos package TPL dependencies
# Set optional dependency on Amesos2, for solving and preconditioning
check_trilinos_tpl(Amesos2_ENABLE_SuperLUDist SuperLUDist FOUR_C_WITH_TRILINOS_SUPERLUDIST)
check_trilinos_tpl(Amesos2_ENABLE_UMFPACK UMFPACK FOUR_C_WITH_TRILINOS_UMFPACK)
check_trilinos_tpl(Amesos2_ENABLE_MUMPS MUMPS FOUR_C_WITH_TRILINOS_MUMPS)
# Set optional dependency on Zoltan2, for partitioning and rebalancing
check_trilinos_tpl(Zoltan2Core_ENABLE_ParMETIS ParMETIS FOUR_C_WITH_TRILINOS_ParMETIS)

# These variables should also be emitted into the generated config.h by
# four_c_configure_dependency().
set(Trilinos_additional_configuration
    FOUR_C_WITH_TRILINOS_SUPERLUDIST
    FOUR_C_WITH_TRILINOS_UMFPACK
    FOUR_C_WITH_TRILINOS_MUMPS
    FOUR_C_WITH_TRILINOS_ParMETIS
    )

target_link_libraries(
  four_c_all_enabled_external_dependencies INTERFACE Trilinos::all_selected_libs
  )

four_c_remember_variable_for_install(
  Trilinos_ROOT
  Kokkos_ROOT
  Kokkos_FIND_QUIETLY
  Trilinos_INCLUDE_DIRS
  Trilinos_LIBRARIES
  )
