
add_executable(ij_deriv)
target_gtensor_sources(ij_deriv PRIVATE ij_deriv.cxx)
target_link_libraries(ij_deriv gtensor::gtensor)
