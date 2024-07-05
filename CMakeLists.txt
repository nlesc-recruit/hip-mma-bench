cmake_minimum_required(VERSION 3.17)
project(hip-mma-bench LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# default GPU support
# W7700 = gfx1101
# 7900XT = gfx1100
# MI210 = gfx90a
# MI300X = gfx940
if(NOT CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES
      "gfx1101,gfx1100,gfx90a,gfx940"
      CACHE STRING "HIP architecture")
endif()

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(hip REQUIRED)

find_package(rocwmma REQUIRED)

include(FetchContent)

FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG v3.2.0)
FetchContent_MakeAvailable(cxxopts)

include(FetchContent)
FetchContent_Declare(
  cudawrappers
  GIT_REPOSITORY https://github.com/nlesc-recruit/cudawrappers
  GIT_TAG hipified)
FetchContent_MakeAvailable(cudawrappers)

option(BUILD_WITH_PMT "Build with Power Measurement Toolkit support" OFF)
option(BUILD_WITH_FMT "Build with Frequency Measurement Toolkit support" OFF)
option(BUILD_KERNEL_DEBUG "Build HIP kernels in debug mode")

if(BUILD_WITH_PMT)
  set(PMT_BUILD_ROCM ON)
  FetchContent_Declare(pmt GIT_REPOSITORY https://git.astron.nl/RD/pmt)
  FetchContent_MakeAvailable(pmt)
  add_compile_definitions("HAVE_PMT")
endif()

if(BUILD_WITH_FMT)
  set(FMT_BUILD_AMD ON)
  FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://git.astron.nl/RD/fmt
    GIT_TAG main)
  FetchContent_MakeAvailable(fmt)
  add_compile_definitions("HAVE_FMT")
endif()

file(GLOB HIP_SOURCES *.hip)
list(REMOVE_ITEM HIP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/common.hip)

add_library(common OBJECT common.cpp common.hip)
target_link_libraries(common cxxopts cudawrappers::cu)
if(BUILD_WITH_PMT)
  target_link_libraries(common pmt)
endif()
if(BUILD_WITH_FMT)
  target_link_libraries(common fmt)
endif()

foreach(source_file ${HIP_SOURCES})
  get_filename_component(executable_name ${source_file} NAME_WE)
  get_filename_component(kernel_file ${source_file} NAME)
  set(KERNEL_FILE "${CMAKE_SOURCE_DIR}/kernels/${kernel_file}")
  add_executable(${executable_name} ${source_file} ${KERNEL_FILE})
  target_link_libraries(${executable_name} common)
  if(BUILD_KERNEL_DEBUG)
    target_compile_options(
      ${executable_name} PRIVATE $<$<COMPILE_LANGUAGE:HIP>: -g
                                 --generate-line-info >)
  endif()
endforeach()
