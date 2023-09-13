cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(gtensor
        VERSION 0.01
        LANGUAGES CXX
        HOMEPAGE_URL https://github.com/wdmapp/gtensor)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "gtensor: Setting build type to 'Release' since none specified.")
  set(CMAKE_BUILD_TYPE "Release"
      CACHE STRING "Choose the type of build." FORCE)
endif()

include(cmake/CPM.cmake)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")

option(USE_GTEST_DISCOVER_TESTS "use gtest_discover_tests()" ON)
set(GTENSOR_DEVICE "cuda" CACHE STRING "Device type 'host', 'cuda', or 'hip'")
set_property(CACHE GTENSOR_DEVICE PROPERTY STRINGS "host" "cuda" "hip" "sycl")
set(GTENSOR_BUILD_DEVICES "" CACHE STRING "List of device types 'host', 'cuda', 'hip', and 'sycl' (semicolon separated)")

set(GTENSOR_MANAGED_MEMORY_TYPE_DEFAULT "managed" CACHE STRING
   "One of 'managed', 'device', or for HIP 'managed_coarse' or 'managed_fine'")
set_property(CACHE GTENSOR_MANAGED_MEMORY_TYPE_DEFAULT
             PROPERTY STRINGS "managed" "device" "managed_coarse" "managed_fine")

set(GTENSOR_UMPIRE_STRATEGY "DynamicPoolList" CACHE STRING "class in umpire::strategy")

option(USE_GTEST_DISCOVER_TESTS "use gtest_discover_tests()" ON)
option(GTENSOR_USE_THRUST "Use thrust (cuda and hip devices)" OFF)
option(GTENSOR_USE_RMM "Use RMM (cuda devices only)" OFF)
option(GTENSOR_USE_UMPIRE "Use Umpire (cuda and hip devices only)" OFF)
option(GTENSOR_TEST_DEBUG "Turn on debug printing for unit tests" OFF)
option(GTENSOR_BUILD_EXAMPLES "Build example programs" OFF)

# Experimental library features
option(GTENSOR_ENABLE_CLIB "Enable libcgtensor" OFF)
option(GTENSOR_ENABLE_BLAS "Enable gtblas" OFF)
option(GTENSOR_ENABLE_FFT  "Enable gtfft" OFF)
option(GTENSOR_ENABLE_FORTRAN "Enable Fortran interoperability" OFF)
option(GTENSOR_ENABLE_SOLVER "Enable high level solver library" OFF)
option(GTENSOR_SOLVER_HIP_SPARSE_GENERIC "Use rocSPARSE generic API for sparse solver" OFF)

option(GTENSOR_PER_DIM_KERNELS
       "Enable per dim kernels (may break for large arrays)" OFF)

option(GTENSOR_DISABLE_PREFETCH
       "Make prefetch operations a no-op. Improves performance in some cases." OFF)

option(GTENSOR_ALLOCATOR_CACHING "Enable naive caching allocators" ON)

option(GTENSOR_BOUNDS_CHECK "Enable per access bounds checking" OFF)
option(GTENSOR_ADDRESS_CHECK "Enable address checking for device spans" OFF)
option(GTENSOR_SYNC_KERNELS "Enable host sync after assign and launch kernels" OFF)
option(GTENSOR_ENABLE_FP16 "Enable 16-bit floating point type gt::float16_t" OFF)

if (GTENSOR_ENABLE_FORTRAN)
  # do this early (here) since later the `enable_language(Fortran)` gives me trouble
  message(STATUS "${PROJECT_NAME}: Fortran is ENABLED")
  enable_language(Fortran)
  set(GTENSOR_ENABLE_CLIB ON)
endif()

# HIP specific configuration
set(HIP_GCC_TOOLCHAIN "" CACHE STRING "pass gcc-toolchain option to hipcc")
if (DEFINED ENV{ROCM_PATH})
   set(ROCM_PATH $ENV{ROCM_PATH} CACHE STRING "path to ROCm installation")
else()
   set(ROCM_PATH "/opt/rocm" CACHE STRING "path to ROCm installation")
endif()
set(HIP_PATH "${ROCM_PATH}/hip" CACHE STRING "path to HIP installation")
set(HIP_GPU_ARCHITECTURES "gfx803,gfx906,gfx908,gfx90a" CACHE
    STRING "comma separated list of AMD gpu architectures to build for")

# SYCL specific configuration
set(GTENSOR_DEVICE_SYCL_AOT_ARCHITECTURES "" CACHE STRING
    "AOT with specified sycl architecture")
set(GTENSOR_DEVICE_SYCL_INTEL ON CACHE BOOL
    "Using Intel OneAPI SYCL implementation")
set(GTENSOR_DEVICE_SYCL_ILP64 OFF CACHE BOOL
    "Link to ILP64 MKL; typically required for host/cpu backends")
set(GTENSOR_DEVICE_SYCL_BBFFT OFF CACHE BOOL
  "Using bbfft lib instead of oneMKL DFT for gt-fft")
set(ONEAPI_PATH "/opt/intel/oneapi" CACHE STRING "path to oneAPI installation")
set(DPCPP_PATH "${ONEAPI_PATH}/compiler/latest/linux" CACHE STRING
    "Path to DPCPP compiler")
set(DPCPP_LIBDIR "${DPCPP_PATH}/lib" CACHE STRING
    "Path to DPCPP compiler lib dir")
set(DPCPP_INCDIR "${DPCPP_PATH}/include/sycl" CACHE STRING
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
  if ("${GTENSOR_DEVICE}" STREQUAL "sycl")
    # Note: SYCL 2020 standard requires C++17, and gtensor takes advantage of this
    # in some SYCL backend specific code
    target_compile_features(gtensor_${DEVICE} INTERFACE cxx_std_17)
  else()
    target_compile_features(gtensor_${DEVICE} INTERFACE cxx_std_14)
  endif()

  list(APPEND GTENSOR_TARGETS gtensor_${DEVICE})

  # alias for using gtensor within build tree (tests, submodule usage)
  add_library(gtensor::gtensor_${DEVICE} ALIAS gtensor_${DEVICE})
endmacro()

message(STATUS "${PROJECT_NAME}: default device ${GTENSOR_DEVICE}")
if ("cuda" IN_LIST GTENSOR_BUILD_DEVICES)
  message(STATUS "${PROJECT_NAME}: adding gtensor_cuda target")
  add_gtensor_library(cuda)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(gtensor_cuda INTERFACE GTENSOR_HAVE_DEVICE)
  target_compile_definitions(gtensor_cuda INTERFACE GTENSOR_DEVICE_CUDA)
  if (GTENSOR_USE_THRUST)
    target_compile_definitions(gtensor_cuda INTERFACE GTENSOR_USE_THRUST)
  endif()
  target_compile_options(gtensor_cuda INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
  target_compile_options(gtensor_cuda INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

  if (GTENSOR_USE_RMM)
    message(STATUS "${PROJECT_NAME}: RMM is enabled")
    CPMAddPackage(
      NAME rmm
      GITHUB_REPOSITORY rapidsai/rmm
      VERSION 23.02.00
    )
    target_compile_definitions(gtensor_cuda INTERFACE
      GTENSOR_USE_MEMORY_POOL GTENSOR_USE_RMM)
    target_link_libraries(gtensor_cuda INTERFACE rmm::rmm)
  endif()
endif()

if ("hip" IN_LIST GTENSOR_BUILD_DEVICES)
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

  set(GPU_TARGETS "${HIP_GPU_ARCHITECTURES}"
      CACHE STRING "GPU targets to compile for")
  find_package(hip REQUIRED)
  find_package(rocthrust REQUIRED)

  if (GTENSOR_USE_THRUST)
    target_compile_definitions(gtensor_hip INTERFACE GTENSOR_USE_THRUST)
  endif()

  target_link_libraries(gtensor_hip INTERFACE roc::rocthrust)

  if (HIP_GCC_TOOLCHAIN)
    target_compile_options(gtensor_hip INTERFACE
      $<$<COMPILE_LANGUAGE:CXX>:--gcc-toolchain=${HIP_GCC_TOOLCHAIN}>)
    target_link_options(gtensor_hip INTERFACE
      $<$<LINK_LANGUAGE:CXX>:--gcc-toolchain=${HIP_GCC_TOOLCHAIN}>)
  endif()

  # Enable to see the full hcc command and include search paths
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -v")
endif()

if ("sycl" IN_LIST GTENSOR_BUILD_DEVICES)
  message(STATUS "${PROJECT_NAME}: adding gtensor_sycl target")
  add_gtensor_library(sycl)

  cmake_minimum_required(VERSION 3.23.0)
  find_package(IntelSYCL)
  if (IntelSYCL_FOUND)
      target_compile_options(gtensor_sycl INTERFACE
                             $<$<COMPILE_LANGUAGE:CXX>:-fsycl -x c++>)
      target_link_options(gtensor_sycl INTERFACE
                          $<$<COMPILE_LANGUAGE:CXX>:-fsycl>)
  else()
    find_package(IntelDPCPP REQUIRED)
  endif()

  target_compile_definitions(gtensor_sycl INTERFACE GTENSOR_HAVE_DEVICE)
  target_compile_definitions(gtensor_sycl INTERFACE GTENSOR_DEVICE_SYCL)

  target_include_directories(gtensor_sycl INTERFACE ${DPCPP_INCDIR})

  if (GTENSOR_DEVICE_SYCL_INTEL)
    # level zero headers should be available, even if different backend is
    # used at runtime
    target_compile_definitions(gtensor_sycl INTERFACE GTENSOR_DEVICE_SYCL_L0)
    target_compile_definitions(gtensor_sycl INTERFACE
                               GTENSOR_DEVICE_SYCL_OPENCL)
    target_link_libraries(gtensor_sycl INTERFACE ze_loader)
    target_link_libraries(gtensor_sycl INTERFACE OpenCL)

    if (GTENSOR_DEVICE_SYCL_AOT_ARCHITECTURES)
      target_compile_options(gtensor_sycl INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-fsycl-targets=spir64_gen
          -Xsycl-target-backend
            "-device ${GTENSOR_DEVICE_SYCL_AOT_ARCHITECTURES}" >)
      target_link_options(gtensor_sycl INTERFACE
        $<$<LINK_LANGUAGE:CXX>:-fsycl-targets=spir64_gen
          -Xsycl-target-backend
            "-device ${GTENSOR_DEVICE_SYCL_AOT_ARCHITECTURES}" >)
    endif()

    add_library(oneapi_mkl_sycl INTERFACE IMPORTED)
    target_include_directories(oneapi_mkl_sycl INTERFACE "${MKL_PATH}/include")
    target_link_libraries(oneapi_mkl_sycl INTERFACE
                          "${MKL_PATH}/lib/${MKL_ARCH}/libmkl_sycl.so")
    target_link_libraries(oneapi_mkl_sycl INTERFACE
                          "${MKL_PATH}/lib/${MKL_ARCH}/libmkl_sequential.so")
    target_link_libraries(oneapi_mkl_sycl INTERFACE
                          "${MKL_PATH}/lib/${MKL_ARCH}/libmkl_core.so")

    if (GTENSOR_DEVICE_SYCL_ILP64)
      target_compile_definitions(oneapi_mkl_sycl INTERFACE MKL_ILP64)
      target_link_libraries(oneapi_mkl_sycl INTERFACE
                            "${MKL_PATH}/lib/${MKL_ARCH}/libmkl_intel_ilp64.so")
    else()
      target_link_libraries(oneapi_mkl_sycl INTERFACE
                            "${MKL_PATH}/lib/${MKL_ARCH}/libmkl_intel_lp64.so")
    endif()
  endif()
endif()

if ("host" IN_LIST GTENSOR_BUILD_DEVICES)
  message(STATUS "${PROJECT_NAME}: adding gtensor_host target")
  add_gtensor_library(host)
  target_compile_definitions(gtensor_host INTERFACE GTENSOR_DEVICE_HOST)
endif()

if (GTENSOR_PER_DIM_KERNELS)
  message(STATUS "${PROJECT_NAME}: using per dim assign/launch kernels")
  target_compile_definitions(gtensor_${GTENSOR_DEVICE}
                             INTERFACE GTENSOR_PER_DIM_KERNELS)
else()
  message(STATUS "${PROJECT_NAME}: using N-d -> 1d assign/launch kernels")
endif()

if (GTENSOR_DISABLE_PREFETCH)
  message(STATUS "${PROJECT_NAME}: prefetch is DISABLED")
  target_compile_definitions(gtensor_${GTENSOR_DEVICE}
                             INTERFACE GTENSOR_DISABLE_PREFETCH)
else()
  message(STATUS "${PROJECT_NAME}: prefetch is ENABLED")
endif()

if (GTENSOR_ALLOCATOR_CACHING AND NOT GTENSOR_USE_RMM AND NOT GTENSOR_USE_UMPIRE)
  message(STATUS "${PROJECT_NAME}: caching allocator enabled")
else()
  target_compile_definitions(gtensor_${GTENSOR_DEVICE}
                             INTERFACE GTENSOR_ALLOCATOR_NO_CACHING)
  message(STATUS "${PROJECT_NAME}: caching allocator disabled")
endif()

if (GTENSOR_USE_UMPIRE)
  message(STATUS "${PROJECT_NAME}: UMPIRE enabled (${GTENSOR_UMPIRE_STRATEGY})")
  find_package(umpire REQUIRED)
  target_compile_definitions(gtensor_${GTENSOR_DEVICE} INTERFACE
    GTENSOR_USE_MEMORY_POOL GTENSOR_USE_UMPIRE)
  target_compile_definitions(gtensor_${GTENSOR_DEVICE} INTERFACE
    GTENSOR_UMPIRE_STRATEGY=${GTENSOR_UMPIRE_STRATEGY})
  target_link_libraries(gtensor_${GTENSOR_DEVICE} INTERFACE umpire)
endif()

if (GTENSOR_BOUNDS_CHECK)
  message(STATUS "${PROJECT_NAME}: bounds checking is ON")
  target_compile_definitions(gtensor_${GTENSOR_DEVICE}
                             INTERFACE GTENSOR_BOUNDS_CHECK)
else()
  message(STATUS "${PROJECT_NAME}: bounds checking is OFF")
endif()

if (GTENSOR_ADDRESS_CHECK)
  message(STATUS "${PROJECT_NAME}: address checking is ON")
  target_compile_definitions(gtensor_${GTENSOR_DEVICE}
                             INTERFACE GTENSOR_ADDRESS_CHECK)
else()
  message(STATUS "${PROJECT_NAME}: address checking is OFF")
endif()

if (GTENSOR_SYNC_KERNELS)
  message(STATUS "${PROJECT_NAME}: sync kernels is ON")
  target_compile_definitions(gtensor_${GTENSOR_DEVICE}
                             INTERFACE GTENSOR_SYNC_KERNELS)
else()
  message(STATUS "${PROJECT_NAME}: sync kernels is OFF")
endif()

if (GTENSOR_ENABLE_FP16)
  message(STATUS "${PROJECT_NAME}: gt::float16_t is ENABLED")
  target_compile_definitions(gtensor_${GTENSOR_DEVICE}
                             INTERFACE GTENSOR_ENABLE_FP16)
endif()


target_compile_definitions(gtensor_${GTENSOR_DEVICE} INTERFACE
  GTENSOR_MANAGED_MEMORY_TYPE_DEFAULT=${GTENSOR_MANAGED_MEMORY_TYPE_DEFAULT})
message(STATUS "${PROJECT_NAME}: default managed memory type '${GTENSOR_MANAGED_MEMORY_TYPE_DEFAULT}'")

# define aliases for use in tests and examples subdirs
add_library(gtensor ALIAS gtensor_${GTENSOR_DEVICE})
add_library(gtensor::gtensor ALIAS gtensor_${GTENSOR_DEVICE})

include(CTest)
if ((BUILD_TESTING AND IS_MAIN_PROJECT) OR GTENSOR_BUILD_BENCHMARKS)
  message(STATUS "${PROJECT_NAME}: build testing is ON")

  CPMAddPackage(
    NAME GTest
    GITHUB_REPOSITORY google/googletest
    # There is no release yet, but we want post 1.10.0 with
    # cmake_minimum_required updated
    GIT_TAG 32f4f52d95dc99c35f51deed552a6ba700567f94
    VERSION 1.10.0
    OPTIONS
      "INSTALL_GTEST OFF"
      "gtest_force_shared_crt ON"
    # googletest >= 1.10.0 provides a cmake config file -- use it if it exists
    FIND_PACKAGE_ARGUMENTS "CONFIG"
  )
  if (GTest_ADDED)
    add_library(GTest::gtest ALIAS gtest)
    add_library(GTest::gmock ALIAS gmock)
    add_library(GTest::gtest_main ALIAS gtest_main)
    add_library(GTest::gmock_main ALIAS gmock_main)
  endif()
  include(GoogleTest)
else()
  message(STATUS "${PROJECT_NAME}: build testing is OFF")
endif()

if (GTENSOR_BUILD_BENCHMARKS)
  CPMAddPackage(
    NAME benchmark
    GITHUB_REPOSITORY google/benchmark
    GIT_TAG v1.6.1
    VERSION v1.6.1
    GIT_SHALLOW TRUE
    OPTIONS
      "BENCHMARK_ENABLE_TESTING OFF"
      "BENCHMARK_ENABLE_INSTALL OFF"
      "BENCHMARK_ENABLE_WERROR OFF"
      "CMAKE_BUILD_TYPE Release"
  )
endif()

include(cmake/target-gtensor-sources-macro.cmake)

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
  target_link_libraries(cgtensor PUBLIC gtensor::gtensor)

  list(APPEND GTENSOR_TARGETS cgtensor)
  add_library(gtensor::cgtensor ALIAS cgtensor)
endif()

if (GTENSOR_ENABLE_BLAS)
  message(STATUS "${PROJECT_NAME}: BLAS is ENABLED")
  add_library(gtblas INTERFACE)
  #target_gtensor_sources(gtblas PRIVATE src/gtblas.cxx)
  target_link_libraries(gtblas INTERFACE gtensor::gtensor)

  if (${GTENSOR_DEVICE} STREQUAL "cuda")
    target_link_libraries(gtblas INTERFACE CUDA::cublas)
  elseif (${GTENSOR_DEVICE} STREQUAL "hip")
    find_package(rocblas REQUIRED)
    find_package(rocsolver REQUIRED)
    target_link_libraries(gtblas INTERFACE roc::rocblas roc::rocsolver)
  elseif (${GTENSOR_DEVICE} STREQUAL "sycl")
    target_link_libraries(gtblas INTERFACE oneapi_mkl_sycl)
  elseif (${GTENSOR_DEVICE} STREQUAL "host")
    # only openblas is supported at this time -- needs
    # adaptations for the various cblas interfaces
    set(BLA_VENDOR OpenBLAS)
    find_package(BLAS REQUIRED)
    get_filename_component(BLAS_DIR ${BLAS_LIBRARIES} DIRECTORY)
    # Hack, see https://gitlab.kitware.com/cmake/cmake/-/issues/20268
    find_path(BLAS_INCLUDE_DIRS cblas.h PATH_SUFFIXES openblas REQUIRED)
    target_include_directories(BLAS::BLAS INTERFACE ${BLAS_INCLUDE_DIRS})
    find_package(LAPACK REQUIRED)
    target_link_libraries(gtblas INTERFACE BLAS::BLAS LAPACK::LAPACK)
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

  if (GTENSOR_ENABLE_SOLVER)
    message(STATUS "${PROJECT_NAME}: SOLVER is ENABLED")
    add_library(gtsolver)
    target_gtensor_sources(gtsolver PRIVATE src/gtsolver.cxx)
    target_link_libraries(gtsolver gtblas)
    if (${GTENSOR_DEVICE} STREQUAL "cuda")
      target_link_libraries(gtsolver CUDA::cusparse)
    elseif (${GTENSOR_DEVICE} STREQUAL "hip")
      target_link_libraries(gtsolver rocsparse)
      if (GTENSOR_SOLVER_HIP_SPARSE_GENERIC)
        target_compile_definitions(gtsolver INTERFACE
                                   GTENSOR_SOLVER_HIP_SPARSE_GENERIC)
      endif()
    endif()
    list(APPEND GTENSOR_TARGETS gtsolver)
    add_library(gtensor::gtsolver ALIAS gtsolver)
  endif()
endif()

if (GTENSOR_ENABLE_FFT)
  message(STATUS "${PROJECT_NAME}: FFT is ENABLED")
  add_library(gtfft INTERFACE)
  target_link_libraries(gtfft INTERFACE gtensor::gtensor)

  if (${GTENSOR_DEVICE} STREQUAL "cuda")
    target_link_libraries(gtfft INTERFACE CUDA::cufft)
  elseif (${GTENSOR_DEVICE} STREQUAL "hip")
    find_package(rocfft REQUIRED)
    target_link_libraries(gtfft INTERFACE roc::rocfft)
  elseif (${GTENSOR_DEVICE} STREQUAL "sycl")
    if (GTENSOR_DEVICE_SYCL_BBFFT)
      #find_package(bbfft-sycl REQUIRED)
      CPMAddPackage(
        NAME bbfft
        GITHUB_REPOSITORY intel/double-batched-fft-library
        GIT_TAG v0.3.6
        OPTIONS
          "BUILD_SYCL ON"
          "BUILD_TESTING OFF"
          "SYCL_COMPILE_OPTIONS -fsycl"
          "SYCL_LINK_OPTIONS -fsycl"
        FIND_PACKAGE_ARGUMENTS "CONFIG"
      )
      target_compile_definitions(gtfft INTERFACE GTENSOR_DEVICE_SYCL_BBFFT)
      target_link_libraries(gtfft INTERFACE bbfft::bbfft-sycl)
    else()
      target_link_libraries(gtfft INTERFACE oneapi_mkl_sycl)
    endif()
  elseif (${GTENSOR_DEVICE} STREQUAL "host")
    find_package(FFTW REQUIRED)

    target_link_libraries(gtfft INTERFACE FFTW::Double FFTW::Float)
  endif()

  list(APPEND GTENSOR_TARGETS gtfft)
  add_library(gtensor::gtfft ALIAS gtfft)
endif()

if (GTENSOR_ENABLE_FORTRAN)
  # message(STATUS "${PROJECT_NAME}: Fortran is ENABLED")
  # enable_language(Fortran)
  add_library(fgtensor)
  set_target_properties(fgtensor PROPERTIES Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fortran)
  target_sources(fgtensor PRIVATE
    src/fortran/gpu_api_interface.F90)
  target_gtensor_sources(fgtensor PRIVATE
    src/fortran/gpu_api.cxx)

  target_include_directories(fgtensor
    INTERFACE
       $<INSTALL_INTERFACE:include/gtensor/fortran>
       $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/fortran>
  )
  target_link_libraries(fgtensor PUBLIC gtensor::cgtensor)

  list(APPEND GTENSOR_TARGETS fgtensor)
  add_library(gtensor::fgtensor ALIAS fgtensor)
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
if (GTENSOR_ENABLE_FORTRAN)
  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fortran DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gtensor)
endif()

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
    cmake/target-gtensor-sources-macro.cmake
    DESTINATION ${INSTALL_CONFIGDIR}
)
