function(target_gtensor_sources TARGET)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs PRIVATE)
  cmake_parse_arguments(target_gtensor_sources "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
  target_sources(${TARGET} PRIVATE ${target_gtensor_sources_PRIVATE})
  if ("${GTENSOR_DEVICE}" STREQUAL "cuda")
    set_source_files_properties(${target_gtensor_sources_PRIVATE}
                                TARGET_DIRECTORY ${TARGET}
                                PROPERTIES LANGUAGE CUDA)
  else()
    set_source_files_properties(${target_gtensor_sources_PRIVATE}
                                TARGET_DIRECTORY ${TARGET}
                                PROPERTIES LANGUAGE CXX)
  endif()
endfunction()
