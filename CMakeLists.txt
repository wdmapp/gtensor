cmake_minimum_required(VERSION 3.13...3.18 FATAL_ERROR)
project(gtensor
        VERSION 0.01
        LANGUAGES CXX
        HOMEPAGE_URL https://github.com/wdmapp/gtensor)

option(USE_GTEST_DISCOVER_TESTS "use gtest_discover_tests()" ON)
set(GTENSOR_DEVICE "cuda" CACHE STRING "Device type 'host', 'cuda', or 'hip'")
set_property(CACHE GTENSOR_DEVICE PROPERTY STRINGS "host" "cuda" "hip" "sycl")
set(GTENSOR_USE_THRUST ON CACHE BOOL "Use thrust (cuda and hip devices)")
set(GTENSOR_TEST_DEBUG OFF CACHE BOOL "Turn on debug printing for unit tests")
set(GTENSOR_BUILD_DEVICES "" CACHE STRING "List of device types 'host', 'cuda', 'hip', and 'sycl' (semicolon separated)")
set(GTENSOR_BUILD_EXAMPLES OFF CACHE BOOL "Build example programs")

# Experimental library features
set(GTENSOR_ENABLE_CLIB OFF CACHE BOOL "Enable libcgtensor")
set(GTENSOR_ENABLE_BLAS OFF CACHE BOOL "Enable gtblas")
set(GTENSOR_ENABLE_FFT  OFF CACHE BOOL "Enable gtfft")

# HIP specific configuration
set(HIP_GCC_TOOLCHAIN "" CACHE STRING "pass gcc-toolchain option to hipcc")
set(ROCM_PATH "/opt/rocm" CACHE STRING "path to ROCm installation")
set(HIP_PATH "/opt/rocm/hip" CACHE STRING "path to HIP installation")
set(HIP_GPU_ARCHITECTURES "gfx803,gfx906,gfx908" CACHE
    STRING "comma separated list of AMD gpu architectures to build for")

# SYCL specific configuration
set(GTENSOR_DEVICE_SYCL_SELECTOR "default" CACHE STRING
    "Use specified sycl device selector ('default', 'gpu', 'cpu',  or 'host')")
set(GTENSOR_DEVICE_SYCL_INTEL ON CACHE BOOL
    "Using Intel OneAPI SYCL implementation")
set(ONEAPI_PATH "/opt/intel/oneapi" CACHE STRING "path to oneAPI installation")
set(DPCPP_PATH "${ONEAPI_PATH}/compiler/latest/linux" CACHE STRING
    "Path to DPCPP compiler")
set(DPCPP_LIBDIR "${DPCPP_ROOT}/lib" CACHE STRING
    "Path to DPCPP compiler lib dir")
set(DPCPP_INCDIR "${DPCPP_ROOT}/include/sycl" CACHE STRING
    "Path to DPCPP include dir")
set(MKL_PATH "${ONEAPI_PATH}/mkl/latest" CACHE STRING "path to oneAPI MKL")
set(MKL_ARCH "intel64" CACHE STRING "MKL architecture (lib subdir)")

set(GTENSOR_SUPPORTED_DEVICES "host;cuda;hip;sycl")
set(GTENSOR_TARGETS "")

# by default, support host plus the default device
if (GTENSOR_BUILD_DEVICES STREQUAL "")
  set(GTENSOR_BUILD_DEVICES ${GTENSOR_DEVICE})
  if (NOT ${GTENSOR_DEVICE} STREQUAL "host")
    list(APPEND GTENSOR_BUILD_DEVICES "host")
  endif()
elseif(NOT ${GTENSOR_DEVICE} IN_LIST GTENSOR_BUILD_DEVICES)
  list(APPEND GTENSOR_BUILD_DEVICES ${GTENSOR_DEVICE})
endif()

# don't build tests if used as a subproject
set(IS_MAIN_PROJECT TRUE)
if (NOT ${CMAKE_PROJECT_NAME} STREQUAL gtensor)
  set(IS_MAIN_PROJECT FALSE)
endif()

function(device_is_supported DEVICE)
  if (NOT ${DEVICE} IN_LIST GTENSOR_SUPPORTED_DEVICES)
    message(FATAL_ERROR
      "${PROJECT_NAME}: device '${GTENSOR_DEVICE}' is not supported (${GTENSOR_SUPPORTED_DEVICES})")
  endif()
endfunction()

foreach(DEVICE IN LISTS GTENSOR_BUILD_DEVICES)
  device_is_supported(${DEVICE})
endforeach()

macro(add_gtensor_library DEVICE)
  device_is_supported(${DEVICE})
  add_library(gtensor_${DEVICE} INTERFACE)
  target_include_directories(gtensor_${DEVICE}
    INTERFACE
       $<INSTALL_INTERFACE:include>
       $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
  target_compile_features(gtensor_${DEVICE} INTERFACE cxx_std_14)

  list(APPEND GTENSOR_TARGETS gtensor_${DEVICE})

  # alias for using gtensor within build tree (tests, submodule usage)
  add_library(gtensor::gtensor_${DEVICE} ALIAS gtensor_${DEVICE})
endmacro()

message(STATUS "${PROJECT_NAME}: default device ${GTENSOR_DEVICE}")
if ("cuda" IN_LIST GTENSOR_BUILD_DEVICES)
  message(STATUS "${PROJECT_NAME}: adding gtensor_cuda target")
  add_gtensor_library(cuda)
  enable_language(CUDA)
  target_compile_definitions(gtensor_cuda INTERFACE GTENSOR_HAVE_DEVICE)
  target_compile_definitions(gtensor_cuda INTERFACE GTENSOR_DEVICE_CUDA)
  if (GTENSOR_USE_THRUST)
    target_compile_definitions(gtensor_cuda INTERFACE GTENSOR_USE_THRUST)
  endif()
  target_compile_options(gtensor_cuda INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
  target_compile_options(gtensor_cuda INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()

if ("hip" IN_LIST GTENSOR_BUILD_DEVICES)

  if (${CMAKE_VERSION} VERSION_LESS "3.18.0")
    message(FATAL_ERROR "gtensor HIP support requires cmake 3.18+")
  endif()

  message(STATUS "${PROJECT_NAME}: adding gtensor_hip target")
  add_gtensor_library(hip)

  if(NOT (CMAKE_CXX_COMPILER MATCHES ".*/hcc$" OR CMAKE_CXX_COMPILER MATCHES ".*/hipcc$"))
    message(FATAL_ERROR "For GTENSOR_BUILD_DEVICES=hip, 'hcc' or 'hipcc' must be used as C++ compiler.")
  endif()

  # The official ROCm/HIP cmake support causes problems on mixed language
  # executables, where hipcc specific flags leak into the other language
  # and break the build. In particular GENE with Fortran does not work when
  # using any ROCm cmake supplied targets, which depend on hip::device which
  # has these flags.
  #find_package(HIP REQUIRED CONFIG PATHS "${HIP_PATH}")
  #if(HIP_PLATFORM STREQUAL "nvcc")
  #  message(FATAL_ERROR "Error: use GTENSOR_DEVICE=cuda for nVidia GPU support")
  #endif()

  target_compile_definitions(gtensor_hip INTERFACE GTENSOR_HAVE_DEVICE)
  target_compile_definitions(gtensor_hip INTERFACE GTENSOR_DEVICE_HIP)

  if (GTENSOR_USE_THRUST)
    target_compile_definitions(gtensor_hip INTERFACE GTENSOR_USE_THRUST)

    add_library(rocthrust INTERFACE IMPORTED)
    target_include_directories(rocthrust INTERFACE
                               "${ROCM_PATH}/rocprim/include")
    target_include_directories(rocthrust INTERFACE
                               "${ROCM_PATH}/rocthrust/include")

    add_library(rocblas INTERFACE IMPORTED)
    target_link_libraries(rocblas INTERFACE "${ROCM_PATH}/lib/librocblas.so")
    target_include_directories(rocblas INTERFACE "${ROCM_PATH}/include")

    add_library(rocsolver INTERFACE IMPORTED)
    target_link_libraries(rocsolver INTERFACE
                          "${ROCM_PATH}/lib/librocsolver.so")
    target_include_directories(rocsolver INTERFACE "${ROCM_PATH}/include")

    add_library(rocfft INTERFACE IMPORTED)
    target_link_libraries(rocfft INTERFACE "${ROCM_PATH}/lib/librocfft.so")
    target_include_directories(rocfft INTERFACE "${ROCM_PATH}/include")

    #find_package(rocprim REQUIRED CONFIG
    #             PATHS "${ROCM_PATH}/rocprim")
    #find_package(rocthrust REQUIRED CONFIG
    #             PATHS "${ROCM_PATH}/rocthrust")

    target_link_libraries(gtensor_hip INTERFACE rocthrust)
    target_compile_options(gtensor_hip INTERFACE
      $<$<COMPILE_LANGUAGE:CXX>:--amdgpu-target=${HIP_GPU_ARCHITECTURES}>)
    target_link_options(gtensor_hip INTERFACE
      $<$<LINK_LANGUAGE:CXX>:--amdgpu-target=${HIP_GPU_ARCHITECTURES}>)

    if (HIP_GCC_TOOLCHAIN)
      target_compile_options(gtensor_hip INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-gcc-toolchain ${HIP_GCC_TOOLCHAIN}>)
      target_link_options(gtensor_hip INTERFACE
        $<$<LINK_LANGUAGE:CXX>:-gcc-toolchain ${HIP_GCC_TOOLCHAIN}>)
    endif()
  endif()

  # Enable to see the full hcc command and include search paths
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -v")
endif()

if ("sycl" IN_LIST GTENSOR_BUILD_DEVICES)
  message(STATUS "${PROJECT_NAME}: adding gtensor_sycl target")
  add_gtensor_library(sycl)

  target_compile_options(gtensor_sycl INTERFACE -fsycl)
  target_link_options(gtensor_sycl INTERFACE -fsycl)
  target_compile_definitions(gtensor_sycl INTERFACE GTENSOR_HAVE_DEVICE)
  target_compile_definitions(gtensor_sycl INTERFACE GTENSOR_DEVICE_SYCL)

  if (${GTENSOR_DEVICE_SYCL_SELECTOR} STREQUAL "default")
    target_compile_definitions(gtensor_sycl INTERFACE
                               GTENSOR_DEVICE_SYCL_DEFAULT)
  elseif (${GTENSOR_DEVICE_SYCL_SELECTOR} STREQUAL "gpu")
    target_compile_definitions(gtensor_sycl INTERFACE GTENSOR_DEVICE_SYCL_GPU)
  elseif (${GTENSOR_DEVICE_SYCL_SELECTOR} STREQUAL "cpu")
    target_compile_definitions(gtensor_sycl INTERFACE GTENSOR_DEVICE_SYCL_CPU)
  elseif (${GTENSOR_DEVICE_SYCL_SELECTOR} STREQUAL "host")
    target_compile_definitions(gtensor_sycl INTERFACE GTENSOR_DEVICE_SYCL_HOST)
  else()
    message(FATAL_ERROR "${PROJECT_NAME}: sycl selector '${GTENSOR_DEVICE_SYCL_SELECTOR}' is not supported")
  endif()

  target_include_directories(gtensor_sycl INTERFACE ${DPCPP_INCDIR})

  if (GTENSOR_DEVICE_SYCL_INTEL)
    if (NOT (${GTENSOR_DEVICE_SYCL_SELECTOR} STREQUAL "host"))
        target_link_options(gtensor_sycl INTERFACE
                            -fsycl-device-lib=libm-fp32,libm-fp64)
    endif()

    add_library(oneapi_mkl_sycl INTERFACE IMPORTED)
    target_compile_definitions(oneapi_mkl_sycl INTERFACE MKL_ILP64)
    target_include_directories(oneapi_mkl_sycl INTERFACE "${MKL_PATH}/include")
    target_link_libraries(oneapi_mkl_sycl INTERFACE
                          "${MKL_PATH}/lib/${MKL_ARCH}/libmkl_sycl.so")
    # NOTE: we could support gnu here to, but when using DPCPP it must be intel,
    # and that is the expected case here.
    target_link_libraries(oneapi_mkl_sycl INTERFACE
                          "${MKL_PATH}/lib/${MKL_ARCH}/libmkl_intel_ilp64.so")
    target_link_libraries(oneapi_mkl_sycl INTERFACE
                          "${MKL_PATH}/lib/${MKL_ARCH}/libmkl_sequential.so")
    target_link_libraries(oneapi_mkl_sycl INTERFACE
                          "${MKL_PATH}/lib/${MKL_ARCH}/libmkl_core.so")
  endif()
endif()

if ("host" IN_LIST GTENSOR_BUILD_DEVICES)
  message(STATUS "${PROJECT_NAME}: adding gtensor_host target")
  add_gtensor_library(host)
  target_compile_definitions(gtensor_host INTERFACE GTENSOR_DEVICE_HOST)
endif()

# define aliases for use in tests and examples subdirs
add_library(gtensor ALIAS gtensor_${GTENSOR_DEVICE})
add_library(gtensor::gtensor ALIAS gtensor_${GTENSOR_DEVICE})

include(CTest)
if (BUILD_TESTING AND IS_MAIN_PROJECT)
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
  else()
    set_source_files_properties(${target_gtensor_sources_PRIVATE} PROPERTIES LANGUAGE CXX)
  endif()
endfunction()

if (BUILD_TESTING AND IS_MAIN_PROJECT)
  add_subdirectory(tests)
endif()

if (GTENSOR_BUILD_EXAMPLES)
  message(STATUS "${PROJECT_NAME}: build examples is ON")
  add_subdirectory(examples)
endif()

if (GTENSOR_BUILD_BENCHMARKS)
  message(STATUS "${PROJECT_NAME}: build benchmarks is ON")
  add_subdirectory(benchmarks)
endif()

if (GTENSOR_ENABLE_CLIB)
  message(STATUS "${PROJECT_NAME}: CLIB is ENABLED")
  add_library(cgtensor)
  target_gtensor_sources(cgtensor PRIVATE src/cgtensor.cxx)
  target_link_libraries(cgtensor gtensor::gtensor)

  list(APPEND GTENSOR_TARGETS cgtensor)
  add_library(gtensor::cgtensor ALIAS cgtensor)
endif()

if (GTENSOR_ENABLE_BLAS)
  message(STATUS "${PROJECT_NAME}: BLAS is ENABLED")
  add_library(gtblas INTERFACE)
  #target_gtensor_sources(gtblas PRIVATE src/gtblas.cxx)
  target_link_libraries(gtblas INTERFACE gtensor::gtensor)

  if (${GTENSOR_DEVICE} STREQUAL "cuda")
    target_link_libraries(gtblas INTERFACE "cublas")
  elseif (${GTENSOR_DEVICE} STREQUAL "hip")
    target_link_libraries(gtblas INTERFACE rocblas rocsolver)
  elseif (${GTENSOR_DEVICE} STREQUAL "sycl")
    target_link_libraries(gtblas INTERFACE oneapi_mkl_sycl)
  endif()

  list(APPEND GTENSOR_TARGETS gtblas)
  add_library(gtensor::blas ALIAS gtblas)

  if (GTENSOR_ENABLE_CLIB)
    message(STATUS "${PROJECT_NAME}: CBLAS is ENABLED")
    add_library(cgtblas)
    target_gtensor_sources(cgtblas PRIVATE src/cgtblas.cxx)
    target_link_libraries(cgtblas gtblas)
    list(APPEND GTENSOR_TARGETS cgtblas)
    add_library(gtensor::cgtblas ALIAS cgtblas)
  endif()
endif()

if (GTENSOR_ENABLE_FFT)
  message(STATUS "${PROJECT_NAME}: FFT is ENABLED")
  add_library(gtfft INTERFACE)
  target_link_libraries(gtfft INTERFACE gtensor::gtensor)

  if (${GTENSOR_DEVICE} STREQUAL "cuda")
    target_link_libraries(gtfft INTERFACE cufft)
  elseif (${GTENSOR_DEVICE} STREQUAL "hip")
    target_link_libraries(gtfft INTERFACE rocfft)
  elseif (${GTENSOR_DEVICE} STREQUAL "sycl")
    target_link_libraries(gtfft INTERFACE oneapi_mkl_sycl)
  endif()

  list(APPEND GTENSOR_TARGETS gtfft)
  add_library(gtensor::gtfft ALIAS gtfft)
endif()

# See https://github.com/pabloariasal/modern-cmake-sample
##############################################
# Installation instructions

include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/gtensor)

install(TARGETS ${GTENSOR_TARGETS}
    EXPORT gtensor-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

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
