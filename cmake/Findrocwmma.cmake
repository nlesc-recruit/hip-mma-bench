include(FindPackageHandleStandardArgs)

find_path(
  ROCWMMA_INCLUDE_DIR
  NAMES rocwmma/rocwmma.hpp)

find_package_handle_standard_args(
  rocwmma
  REQUIRED_VARS ROCWMMA_INCLUDE_DIR
  VERSION_VAR ROCWMMA_VERSION)

if(NOT TARGET rocwmma)
  add_library(rocwmma INTERFACE)
  target_include_directories(rocwmma INTERFACE ${ROCWMMA_INCLUDE_DIR})
endif()
