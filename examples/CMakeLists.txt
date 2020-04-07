cmake_minimum_required(VERSION 3.13)
project(daxpy_example LANGUAGES CXX)
set(VERSION 0.01)

set(GTENSOR_DEVICE "cuda" CACHE STRING "Device type 'none', 'cuda', or 'hip'")
set_property(CACHE GTENSOR_DEVICE PROPERTY STRINGS "none" "cuda" "hip")

add_library(gtensor INTERFACE)

if ("${GTENSOR_DEVICE}" STREQUAL "cuda")
  message(INFO "Gtensor device: cuda")
  enable_language(CUDA)
  target_compile_definitions(gtensor INTERFACE GTENSOR_HAVE_DEVICE)
  target_compile_options(gtensor INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
  target_compile_options(gtensor INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
elseif ("${GTENSOR_DEVICE}" STREQUAL "hip")
  message(INFO "Gtensor device: hip")

  if(NOT (CMAKE_CXX_COMPILER MATCHES ".*/hcc$" OR CMAKE_CXX_COMPILER MATCHES ".*/hipcc$"))
    message(FATAL_ERROR "For GTENSOR_DEVICE=hip, 'hcc' or 'clang' must be used as C++ compiler.")
  endif()

  if(NOT DEFINED HIP_PATH)
    if(DEFINED "$ENV{HIP_PATH}")
      set(HIP_PATH "$ENV{HIP_PATH}")
    else()
      set(HIP_PATH "/opt/rocm/hip")
    endif()
  endif()

  # Get FindHIP.cmake from hip installation
  list(APPEND CMAKE_MODULE_PATH
    ${HIP_PATH}/cmake
  )
  find_package(HIP REQUIRED CONFIG PATHS "${HIP_PATH}")
  if(HIP_PLATFORM STREQUAL "nvcc")
    message(FATAL_ERROR "Error: use GTENSOR_DEVICE=cuda for nVidia GPU support")
  endif()

  # On ROCm rocThrust requires rocPRIM
  find_package(rocprim REQUIRED CONFIG
               HINTS "${HIP_PATH}/../rocprim" PATHS "/opt/rocm/rocprim")

  find_package(rocthrust REQUIRED CONFIG
               HINTS "${HIP_PATH}/../rocthrust" PATHS "/opt/rocm/rocthrust")

  find_package(hcc REQUIRED CONFIG
               HINTS "${HIP_PATH}/../" PATHS "/opt/rocm")

  target_compile_definitions(gtensor INTERFACE GTENSOR_HAVE_DEVICE)

  # Enable to see the full hcc command and include search paths
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -v")

  target_link_libraries(gtensor INTERFACE roc::rocthrust)
elseif ("${GTENSOR_DEVICE}" STREQUAL "none")
  message(INFO "Gtensor device: none")
else()
  message(FATAL_ERROR "Unsupported GTENSOR_DEVICE: ${GTENSOR_DEVICE}")
endif()

function(target_cxx_sources TARGET)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs PRIVATE)
  cmake_parse_arguments(target_cxx_sources "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
  target_sources(${TARGET} PRIVATE ${target_cxx_sources_PRIVATE})
  if ("${GTENSOR_DEVICE}" STREQUAL "cuda")
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
