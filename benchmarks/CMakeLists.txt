
add_executable(ij_deriv)
target_gtensor_sources(ij_deriv PRIVATE ij_deriv.cxx)
target_link_libraries(ij_deriv gtensor::gtensor)

if (GTENSOR_ENABLE_BLAS)
  add_executable(getrs)
  target_gtensor_sources(getrs PRIVATE getrs.cxx)
  target_link_libraries(getrs gtensor::gtensor gtensor::blas)

  add_executable(bench_getrf)
  target_gtensor_sources(bench_getrf PRIVATE bench_getrf.cxx)
  target_link_libraries(bench_getrf gtensor::gtensor gtensor::blas benchmark::benchmark)

  add_executable(bench_getri)
  target_gtensor_sources(bench_getri PRIVATE bench_getri.cxx)
  target_link_libraries(bench_getri gtensor::gtensor gtensor::blas benchmark::benchmark)
endif()

add_executable(bench_assign)
target_gtensor_sources(bench_assign PRIVATE bench_assign.cxx)
target_link_libraries(bench_assign PRIVATE gtensor::gtensor benchmark::benchmark)

add_executable(bench_hypz)
target_gtensor_sources(bench_hypz PRIVATE bench_hypz.cxx)
target_link_libraries(bench_hypz PRIVATE gtensor::gtensor benchmark::benchmark)

add_executable(bench_stencil1d)
target_gtensor_sources(bench_stencil1d PRIVATE bench_stencil1d.cxx)
target_link_libraries(bench_stencil1d gtensor::gtensor benchmark::benchmark)

add_executable(bench_sparse)
target_gtensor_sources(bench_sparse PRIVATE bench_sparse.cxx)
target_link_libraries(bench_sparse gtensor::gtensor benchmark::benchmark)

if (GTENSOR_ENABLE_FFT)
  add_executable(bench_fft)
  target_gtensor_sources(bench_fft PRIVATE bench_fft.cxx)
  target_link_libraries(bench_fft gtensor::gtensor gtensor::gtfft benchmark::benchmark)
endif()
