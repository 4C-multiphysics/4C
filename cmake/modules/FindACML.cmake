# include(FindLibraryWithDebug)

if(ACML_INCLUDES AND ACML_LIBRARIES)
  set(ACML_FIND_QUIETLY TRUE)
endif(ACML_INCLUDES AND ACML_LIBRARIES)
find_path(
  ACML_INCLUDES
  NAMES acml.h
  PATHS ${ACMLDIR}/include ${ACML_DIR}/include ${INCLUDE_INSTALL_DIR}
  )

find_library(
  ACML_LIBRARIES
  NAMES acml_mp acml_mv
  PATHS ${ACMLDIR}/lib ${ACML_DIR}/lib ${LIB_INSTALL_DIR}
  )

find_file(
  ACML_LIBRARIES
  NAMES libacml_mp.so
  PATHS /usr/lib ${ACMLDIR}/lib ${LIB_INSTALL_DIR}
  )

if(NOT ACML_LIBRARIES)
  message(STATUS "Multi-threaded library not found, looking for single-threaded")
  find_library(
    ACML_LIBRARIES
    NAMES acml acml_mv
    PATHS ${ACMLDIR}/lib ${ACML_DIR}/lib ${LIB_INSTALL_DIR}
    )
  find_file(
    ACML_LIBRARIES libacml.so libacml_mv.so PATHS /usr/lib ${ACMLDIR}/lib ${LIB_INSTALL_DIR}
    )
endif()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(ACML DEFAULT_MSG ACML_INCLUDES ACML_LIBRARIES)
mark_as_advanced(ACML_INCLUDES ACML_LIBRARIES)

if(ACML_FOUND AND NOT TARGET acml::acml)
  add_library(acml::acml UNKNOWN IMPORTED)
  set_target_properties(
    acml::acml
    PROPERTIES IMPORTED_LOCATION "${ACML_LIBRARIES}"
               INTERFACE_INCLUDE_DIRECTORIES "${ACML_INCLUDES}"
    )
endif()

if(ACML_FOUND)
  list(APPEND BACI_ALL_ENABLED_EXTERNAL_LIBS acml::acml)
  message(STATUS "Found ACML (AMD Core Math Library): ${ACML_LIBRARIES}")
  message(STATUS "Found ACML (AMD Core Math Library): ${ACML_INCLUDES}")
endif()
