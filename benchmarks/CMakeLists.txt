
add_executable(ij_deriv)
target_gtensor_sources(ij_deriv PRIVATE ij_deriv.cxx)
target_link_libraries(ij_deriv gtensor::gtensor)

add_executable(bench_assign)
target_gtensor_sources(bench_assign PRIVATE bench_assign.cxx)
target_link_libraries(bench_assign PRIVATE gtensor::gtensor benchmark::benchmark)

add_executable(bench_hypz)
target_gtensor_sources(bench_hypz PRIVATE bench_hypz.cxx)
target_link_libraries(bench_hypz PRIVATE gtensor::gtensor benchmark::benchmark)