cmake_minimum_required(VERSION 3.13)
project(daxpy_example LANGUAGES CXX)
set(VERSION 0.01)

option(GTENSOR_CUDA "support CUDA" OFF)
option(GTENSOR_HIP "support CUDA" OFF)

add_library(gtensor INTERFACE)

if (GTENSOR_CUDA)
  enable_language(CUDA)
  list(APPEND CMAKE_CUDA_FLAGS "-DGTENSOR_HAVE_DEVICE")

  target_compile_options(gtensor INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
  target_compile_options(gtensor INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
elseif (GTENSOR_HIP)
  list(APPEND CMAKE_MODULE_PATH
    ${HIP_PATH}/cmake /opt/rocm/hip/cmake # FindHIP.cmake
  )
  find_package(HIP REQUIRED CONFIG PATHS "/opt/rocm/hip")
  if(HIP_PLATFORM STREQUAL "nvcc")
    message(FATAL_ERROR "Error: use GTENSOR_CUDA for nVidia GPU support")
  endif()

  # On ROCm rocThrust requires rocPRIM
  find_package(rocprim REQUIRED CONFIG PATHS "/opt/rocm/rocprim")

  # "/opt/rocm" - default install prefix
  find_package(rocthrust REQUIRED CONFIG PATHS "/opt/rocm/rocthrust")

  find_package(hcc REQUIRED CONFIG PATHS /opt/rocm)

  target_compile_definitions(gtensor INTERFACE GTENSOR_HAVE_DEVICE)
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -v")

  #target_compile_options(gtensor INTERFACE --expt-extended-lambda)
  #target_compile_options(gtensor INTERFACE --expt-relaxed-constexpr)

  #target_link_libraries(gtensor INTERFACE rocthrust)
  target_include_directories(gtensor INTERFACE "/opt/rocm/rocprim/include")
  target_include_directories(gtensor INTERFACE "/opt/rocm/rocthrust/include")
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

target_include_directories(gtensor INTERFACE ../include)
target_compile_features(gtensor INTERFACE cxx_std_14)

get_target_property(gtensor_INCLUDE_DIRS gtensor INTERFACE_INCLUDE_DIRECTORIES)

add_executable(daxpy)
target_cxx_sources(daxpy PRIVATE src/daxpy.cxx)
target_link_libraries(daxpy gtensor)

add_executable(stencil1d)
target_cxx_sources(stencil1d PRIVATE src/stencil1d.cxx)
target_link_libraries(stencil1d gtensor)

add_executable(mult_table)
target_cxx_sources(mult_table PRIVATE src/mult_table.cxx)
target_link_libraries(mult_table gtensor)
