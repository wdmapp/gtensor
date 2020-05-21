cmake_minimum_required(VERSION 3.13...3.17.2 FATAL_ERROR)
project(gtensor
        VERSION 0.01
        LANGUAGES CXX
        HOMEPAGE_URL https://github.com/wdmapp/gtensor)

option(USE_GTEST_DISCOVER_TESTS "use gtest_discover_tests()" ON)
set(GTENSOR_DEVICE "cuda" CACHE STRING "Device type 'none', 'cuda', or 'hip'")
set_property(CACHE GTENSOR_DEVICE PROPERTY STRINGS "none" "cuda" "hip")
set(GTENSOR_BUILD_EXAMPLES OFF CACHE BOOL "Build example programs")

if (NOT ${CMAKE_PROJECT_NAME} STREQUAL gtensor)
  set(BUILD_TESTING OFF)
endif()

add_library(gtensor INTERFACE)

if ("${GTENSOR_DEVICE}" STREQUAL "cuda")
  message(STATUS "${PROJECT_NAME}: device cuda")
  enable_language(CUDA)
  target_compile_definitions(gtensor INTERFACE GTENSOR_HAVE_DEVICE)
  target_compile_definitions(gtensor INTERFACE GTENSOR_DEVICE_CUDA)
  target_compile_options(gtensor INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
  target_compile_options(gtensor INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
elseif ("${GTENSOR_DEVICE}" STREQUAL "hip")
  message(STATUS "${PROJECT_NAME}: device hip")

  if(NOT (CMAKE_CXX_COMPILER MATCHES ".*/hcc$" OR CMAKE_CXX_COMPILER MATCHES ".*/hipcc$"))
    message(FATAL_ERROR "For GTENSOR_DEVICE=hip, 'hcc' or 'clang' must be used as C++ compiler.")
  endif()

  find_package(HIP REQUIRED CONFIG PATHS "/opt/rocm/hip")
  if(HIP_PLATFORM STREQUAL "nvcc")
    message(FATAL_ERROR "Error: use GTENSOR_DEVICE=cuda for nVidia GPU support")
  endif()

  find_package(rocprim REQUIRED CONFIG
               PATHS "/opt/rocm/rocprim")
  find_package(rocthrust REQUIRED CONFIG
               PATHS "/opt/rocm/rocthrust")

  target_compile_definitions(gtensor INTERFACE GTENSOR_HAVE_DEVICE)
  target_compile_definitions(gtensor INTERFACE GTENSOR_DEVICE_HIP)

  # Enable to see the full hcc command and include search paths
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -v")

  target_link_libraries(gtensor INTERFACE roc::rocthrust)
elseif ("${GTENSOR_DEVICE}" STREQUAL "none")
  message(STATUS "${PROJECT_NAME}: device none")
else()
  message(FATAL_ERROR "Unsupported GTENSOR_DEVICE: ${GTENSOR_DEVICE}")
endif()

include(CTest)
if (BUILD_TESTING)
  message(STATUS "${PROJECT_NAME}: build testing is ON")

  # try to find gtest, otherwise fetch and add to build
  find_package(GTest QUIET)

  if (NOT GTEST_FOUND)
    message(STATUS "${PROJECT_NAME}: googletest not found, fetching source and adding to build")
    include(FetchContent)
    FetchContent_Declare(googletest
      GIT_REPOSITORY    https://github.com/google/googletest.git
      GIT_TAG           release-1.10.0
      )
    FetchContent_GetProperties(googletest)
    if (NOT googletest_POPULATED)
      FetchContent_Populate(googletest)
      add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
    add_library(GTest::GTest INTERFACE IMPORTED)
    target_include_directories(GTest::GTest INTERFACE "${googletest_SOURCE_DIR}/googletest/include")
    target_link_libraries(GTest::GTest INTERFACE gtest)
    add_library(GTest::Main INTERFACE IMPORTED)
    target_link_libraries(GTest::Main INTERFACE gtest_main)
  endif()
else()
  message(STATUS "${PROJECT_NAME}: build testing is OFF")
endif()

function(target_gtensor_sources TARGET)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs PRIVATE)
  cmake_parse_arguments(target_gtensor_sources "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
  target_sources(${TARGET} PRIVATE ${target_gtensor_sources_PRIVATE})
  if ("${GTENSOR_DEVICE}" STREQUAL "cuda")
    set_source_files_properties(${target_gtensor_sources_PRIVATE} PROPERTIES LANGUAGE CUDA)
  endif()
endfunction()
  
target_include_directories(gtensor
    INTERFACE
       $<INSTALL_INTERFACE:include>
       $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_features(gtensor INTERFACE cxx_std_14)

# alias for using gtensor within build tree (tests, submodule usage)
add_library(gtensor::gtensor ALIAS gtensor)

if (BUILD_TESTING)
  add_subdirectory(tests)
endif()

if (GTENSOR_BUILD_EXAMPLES)
  message(STATUS "${PROJECT_NAME}: build examples is ON")
  add_subdirectory(examples)
endif()

# See https://github.com/pabloariasal/modern-cmake-sample
##############################################
# Installation instructions

include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/gtensor)

install(TARGETS gtensor
    EXPORT gtensor-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

set_target_properties(gtensor PROPERTIES EXPORT_NAME gtensor)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

#Export the targets to a script
install(EXPORT gtensor-targets
    FILE
        gtensor-targets.cmake
    NAMESPACE
        gtensor::
    DESTINATION
        ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/gtensor-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/gtensor-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/gtensor-config.cmake
    INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

#Install the config, configversion and custom find modules
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/gtensor-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/gtensor-config-version.cmake
    DESTINATION ${INSTALL_CONFIGDIR}
)

##############################################
## Exporting from the build tree

export(EXPORT gtensor-targets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/gtensor-targets.cmake
    NAMESPACE gtensor::)

#Register package in user's package registry
export(PACKAGE gtensor)
