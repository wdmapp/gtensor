cmake_minimum_required(VERSION 3.13...3.17.2 FATAL_ERROR)
project(gtensor-examples
        VERSION 0.01
        LANGUAGES CXX
        HOMEPAGE_URL https://github.com/wdmapp/gtensor)

option(GTENSOR_EXAMPLES_USE_FIND_PACKAGE,
    "when ON, use find_package, when OFF use a copy of gtensor as subdir"
    OFF)

# support building as a subdirectory of the main project, or
# as an independent project with gtensor installed or gtensor
# as a subdirectory
if(NOT TARGET gtensor::gtensor)
  if (GTENSOR_EXAMPLES_USE_FIND_PACKAGE)
    message(STATUS "${PROJECT_NAME}: building with gtensor via find_package")
    find_package(gtensor)
  else()
    message(STATUS "${PROJECT_NAME}: building with gtensor as a subdir")
    file(MAKE_DIRECTORY external/gtensor)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../include
              ${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt
              ${CMAKE_CURRENT_SOURCE_DIR}/../gtensor-config.cmake.in
         DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/external/gtensor
         PATTERN examples EXCLUDE)
    add_subdirectory(external/gtensor)
  endif()
  if (${GTENSOR_DEVICE} STREQUAL "cuda")
    # Note: must be called in top level CMakeLists.txt
    enable_language(CUDA)
  endif()
else()
  message(STATUS "${PROJECT_NAME}: building as gtensor subdir")
endif()

add_executable(daxpy)
target_gtensor_sources(daxpy PRIVATE src/daxpy.cxx)
target_link_libraries(daxpy gtensor::gtensor)

add_executable(stencil1d)
target_gtensor_sources(stencil1d PRIVATE src/stencil1d.cxx)
target_link_libraries(stencil1d gtensor::gtensor)

add_executable(mult_table)
target_gtensor_sources(mult_table PRIVATE src/mult_table.cxx)
target_link_libraries(mult_table gtensor::gtensor)

if ("${GTENSOR_DEVICE}" STREQUAL "cuda")
  add_executable(trig)
  target_gtensor_sources(trig PRIVATE src/trig.cu)

  add_executable(trig_adapted)
  target_gtensor_sources(trig_adapted PRIVATE src/trig_adapted.cxx)
  target_link_libraries(trig_adapted gtensor::gtensor)
endif()
