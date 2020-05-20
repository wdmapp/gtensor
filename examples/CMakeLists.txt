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
