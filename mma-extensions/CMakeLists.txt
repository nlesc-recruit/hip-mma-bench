cmake_minimum_required(VERSION 3.17)
project(hip-mma-extensions LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(hip REQUIRED)

include(FetchContent)
FetchContent_Declare(
  cudawrappers
  GIT_REPOSITORY https://github.com/nlesc-recruit/cudawrappers
  GIT_TAG hipified)
FetchContent_MakeAvailable(cudawrappers)

option(BUILD_KERNEL_DEBUG "Build HIP kernels in debug mode" OFF)

file(GLOB HIP_SOURCES *.hip)

foreach(source_file ${HIP_SOURCES})
  get_filename_component(executable_name ${source_file} NAME_WE)
  add_executable(${executable_name} ${source_file})
  target_include_directories(${executable_name} PRIVATE "${CMAKE_SOURCE_DIR}/include")
  target_link_libraries(${executable_name} cudawrappers::cu)
  if(BUILD_KERNEL_DEBUG)
    target_compile_options(
      ${executable_name} PRIVATE $<$<COMPILE_LANGUAGE:HIP>: -g
                                 --generate-line-info >)
  endif()
endforeach()
