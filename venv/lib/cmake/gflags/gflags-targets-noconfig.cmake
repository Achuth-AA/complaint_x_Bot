#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "gflags::gflags_shared" for configuration ""
set_property(TARGET gflags::gflags_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(gflags::gflags_shared PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libgflags.2.2.2.dylib"
  IMPORTED_SONAME_NOCONFIG "@rpath/libgflags.2.2.dylib"
  )

list(APPEND _cmake_import_check_targets gflags::gflags_shared )
list(APPEND _cmake_import_check_files_for_gflags::gflags_shared "${_IMPORT_PREFIX}/lib/libgflags.2.2.2.dylib" )

# Import target "gflags::gflags_nothreads_shared" for configuration ""
set_property(TARGET gflags::gflags_nothreads_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(gflags::gflags_nothreads_shared PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libgflags_nothreads.2.2.2.dylib"
  IMPORTED_SONAME_NOCONFIG "@rpath/libgflags_nothreads.2.2.dylib"
  )

list(APPEND _cmake_import_check_targets gflags::gflags_nothreads_shared )
list(APPEND _cmake_import_check_files_for_gflags::gflags_nothreads_shared "${_IMPORT_PREFIX}/lib/libgflags_nothreads.2.2.2.dylib" )

# Import target "gflags::gflags_static" for configuration ""
set_property(TARGET gflags::gflags_static APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(gflags::gflags_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libgflags.a"
  )

list(APPEND _cmake_import_check_targets gflags::gflags_static )
list(APPEND _cmake_import_check_files_for_gflags::gflags_static "${_IMPORT_PREFIX}/lib/libgflags.a" )

# Import target "gflags::gflags_nothreads_static" for configuration ""
set_property(TARGET gflags::gflags_nothreads_static APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(gflags::gflags_nothreads_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libgflags_nothreads.a"
  )

list(APPEND _cmake_import_check_targets gflags::gflags_nothreads_static )
list(APPEND _cmake_import_check_files_for_gflags::gflags_nothreads_static "${_IMPORT_PREFIX}/lib/libgflags_nothreads.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
