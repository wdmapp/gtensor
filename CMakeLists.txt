cmake_minimum_required(VERSION 3.13)
project(gtensor LANGUAGES CXX)
set(VERSION 0.01)

option(USE_GTEST_DISCOVER_TESTS "use gtest_discover_tests()" ON)
set(GTENSOR_DEVICE "cuda" CACHE STRING "Device type 'none', 'cuda', or 'hip'")

add_library(gtensor INTERFACE)

if ("${GTENSOR_DEVICE}" STREQUAL "cuda")
  message(INFO "Gtensor device: cuda")
  enable_language(CUDA)
  list(APPEND CMAKE_CUDA_FLAGS "-DGTENSOR_HAVE_DEVICE")
  target_compile_options(gtensor INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
  target_compile_options(gtensor INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
elseif ("${GTENSOR_DEVICE}" STREQUAL "hip")
  message(INFO "Gtensor device: hip")

  list(APPEND CMAKE_MODULE_PATH
    ${HIP_PATH}/cmake /opt/rocm/hip/cmake # FindHIP.cmake
  )
  find_package(HIP REQUIRED CONFIG PATHS "/opt/rocm/hip")
  if(HIP_PLATFORM STREQUAL "nvcc")
    message(FATAL_ERROR "Error: use GTENSOR_DEVICE=cuda for nVidia GPU support")
  endif()

  # On ROCm rocThrust requires rocPRIM
  find_package(rocprim REQUIRED CONFIG PATHS "/opt/rocm/rocprim")

  # "/opt/rocm" - default install prefix
  find_package(rocthrust REQUIRED CONFIG PATHS "/opt/rocm/rocthrust")

  find_package(hcc REQUIRED CONFIG PATHS /opt/rocm)

  target_compile_definitions(gtensor INTERFACE GTENSOR_HAVE_DEVICE)
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -v")

  #target_link_libraries(gtensor INTERFACE rocthrust)
  target_include_directories(gtensor INTERFACE "/opt/rocm/rocprim/include")
  target_include_directories(gtensor INTERFACE "/opt/rocm/rocthrust/include")
elseif ("${GTENSOR_DEVICE}" STREQUAL "none")
  message(INFO "Gtensor device: none")
else()
  message(FATAL_ERROR "Unsupported GTENSOR_DEVICE: ${GTENSOR_DEVICE}")
endif()

find_package(GTest)
if (GTEST_FOUND)
  include(CTest)
endif()

function(target_cxx_sources TARGET)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs PRIVATE)
  cmake_parse_arguments(target_cxx_sources "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
  target_sources(${TARGET} PRIVATE ${target_cxx_sources_PRIVATE})
  if (GTENSOR_CUDA)
    set_source_files_properties(${target_cxx_sources_PRIVATE} PROPERTIES LANGUAGE CUDA)
  endif()
endfunction()
  
target_include_directories(gtensor INTERFACE include)
target_compile_features(gtensor INTERFACE cxx_std_14)
if (GTEST_FOUND AND BUILD_TESTING)
  add_subdirectory(tests)
endif()
