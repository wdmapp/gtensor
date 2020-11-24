#include <gtensor/gtensor.h>

#ifdef GTENSOR_DEVICE_SYCL
#include <cstdlib>
#include <iostream>
#endif

#include "gtensor/gpufft.h"

void gpufft_plan_many(gpufft_handle_t* handle, int rank, int* n, int istride,
                      int idist, int ostride, int odist,
                      gpufft_transform_t type, int batchSize)
{
#ifdef GTENSOR_DEVICE_CUDA
  auto result = cufftPlanMany(handle, rank, n, nullptr, istride, idist, nullptr,
                              ostride, odist, type, batchSize);
  assert(result == CUFFT_SUCCESS);
#elif defined(GTENSOR_DEVICE_HIP)
  auto result = hipfftPlanMany(handle, rank, n, nullptr, istride, idist,
                               nullptr, ostride, odist, type, batchSize);
  assert(result == HIPFFT_SUCCESS);
  /*
  rocfft_plan_description desc = NULL;
  rocfft_plan_description_create(&desc);
  rocfft_plan_description_set_data_layout(
        desc,
        // input data format:
        rocfft_array_type_real,
        // output data format:
        rocfft_array_type_real,
        nullptr, // in offsets
        nullptr, // out offsets
        rank, // input stride length
        istride, // input stride data
        idist, // input batch distance
        rank, // output stride length
        ostride, // output stride data
        odist); // ouptut batch distance
  auto result = rocfft_plan_create(handle, rocfft_placment_inline, type,
                                   rocfft_precision_double, rank, n, batchSize,
  desc); assert(result == rocfft_success);
  */
#elif defined(GTENSOR_DEVICE_SYCL)
  std::int64_t fwd_distance, bwd_distance;

  if (type == GPUFFT_R2C || type == GPUFFT_D2Z) {
    fwd_distance = idist;
    bwd_distance = odist;
  } else {
    fwd_distance = odist;
    bwd_distance = idist;
  }
  // TODO: handle other types

  // TODO: handle single precision
  gpufft_double_descriptor_t* h;
  try {
    if (rank > 1) {
      std::vector<MKL_LONG> dims(rank);
      for (int i = 0; i < rank; i++) {
        dims[i] = n[i];
      }
      assert(dims.size() == rank);
      h = new gpufft_double_descriptor_t(dims);
    } else {
      h = new gpufft_double_descriptor_t(n[0]);
    }

    // set up strides arrays
    // TODO: the default generally does the right thing based on dims. Perhaps
    // we should simplify the interface, and do the same to calculate strides
    // and distance for the other backends.
    // TODO: is this correct column major?
    std::int64_t rstrides[rank + 1];
    std::int64_t cstrides[rank + 1];
    rstrides[0] = 0;
    cstrides[0] = 0;
    std::int64_t rs = istride;
    std::int64_t cs = ostride;
    for (int i = 1; i <= rank; i++) {
      rstrides[i] = rs;
      cstrides[i] = cs;
      rs *= n[i - 1];
      cs *= n[i - 1];
    }

    h->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                 batchSize);
    h->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, rstrides);
    h->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, cstrides);
    h->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, fwd_distance);
    h->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, bwd_distance);
    h->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
    if (type != GPUFFT_Z2Z && type != GPUFFT_C2C) {
      h->set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE,
                   DFTI_COMPLEX_COMPLEX);
    }
    h->commit(gt::backend::sycl::get_queue());
    *handle = h;
  } catch (std::exception const& e) {
    std::cerr << "Error creating dft descriptor:" << e.what() << std::endl;
    abort();
  }
#endif
}

void gpufft_plan_destroy(gpufft_handle_t handle)
{
#ifdef GTENSOR_DEVICE_CUDA
  auto result = cufftDestroy(handle);
  assert(result == CUFFT_SUCCESS);
#elif defined(GTENSOR_DEVICE_HIP)
  auto result = hipfftDestroy(handle);
  assert(result == rocfft_success);
#elif defined(GTENSOR_DEVICE_SYCL)
  delete handle;
#endif
}

void gpufft_exec_z2d(gpufft_handle_t handle, gpufft_double_complex_t* indata,
                     gpufft_double_real_t* outdata)
{
#ifdef GTENSOR_DEVICE_CUDA
  auto result = cufftExecZ2D(handle, indata, outdata);
  assert(result == CUFFT_SUCCESS);
#elif defined(GTENSOR_DEVICE_HIP)
  auto result = hipfftExecZ2D(handle, indata, outdata);
  assert(result == rocfft_success);
#elif defined(GTENSOR_DEVICE_SYCL)
  // auto h = static_cast<gpufft_double_descriptor_t*>(handle);
  auto indata_double = reinterpret_cast<double*>(indata);
  auto e = oneapi::mkl::dft::compute_backward(*handle, indata_double, outdata);
  e.wait();
#endif
}

void gpufft_exec_d2z(gpufft_handle_t handle, gpufft_double_real_t* indata,
                     gpufft_double_complex_t* outdata)
{
#ifdef GTENSOR_DEVICE_CUDA
  auto result = cufftExecD2Z(handle, indata, outdata);
  assert(result == CUFFT_SUCCESS);
#elif defined(GTENSOR_DEVICE_HIP)
  auto result = hipfftExecD2Z(handle, indata, outdata);
  assert(result == rocfft_success);
#elif defined(GTENSOR_DEVICE_SYCL)
  // auto h = static_cast<gpufft_double_descriptor_t*>(handle);
  auto outdata_double = reinterpret_cast<double*>(outdata);
  auto e = oneapi::mkl::dft::compute_forward(*handle, indata, outdata_double);
  e.wait();
#endif
}

void gpufft_exec_c2r(gpufft_handle_t handle, gpufft_complex_t* indata,
                     gpufft_real_t* outdata)
{
#ifdef GTENSOR_DEVICE_CUDA
  auto result = cufftExecC2R(handle, indata, outdata);
  assert(result == CUFFT_SUCCESS);
#elif defined(GTENSOR_DEVICE_HIP)
  auto result = hipfftExecC2R(handle, indata, outdata);
  assert(result == rocfft_success);
#elif defined(GTENSOR_DEVICE_SYCL)
  // auto h = static_cast<gpufft_descriptor_t*>(handle);
  auto indata_float = reinterpret_cast<float*>(indata);
  auto e = oneapi::mkl::dft::compute_backward(*handle, indata_float, outdata);
  e.wait();
#endif
}

void gpufft_exec_r2c(gpufft_handle_t handle, gpufft_real_t* indata,
                     gpufft_complex_t* outdata)
{
#ifdef GTENSOR_DEVICE_CUDA
  auto result = cufftExecR2C(handle, indata, outdata);
  assert(result == CUFFT_SUCCESS);
#elif defined(GTENSOR_DEVICE_HIP)
  auto result = hipfftExecR2C(handle, indata, outdata);
  assert(result == rocfft_success);
#elif defined(GTENSOR_DEVICE_SYCL)
  // auto h = static_cast<gpufft_descriptor_t*>(handle);
  auto outdata_float = reinterpret_cast<float*>(outdata);
  auto e = oneapi::mkl::dft::compute_forward(*handle, indata, outdata_float);
  e.wait();
#endif
}

void gpufft_exec_z2z(gpufft_handle_t handle, gpufft_double_complex_t* indata,
                     gpufft_double_complex_t* outdata, int direction)
{
#ifdef GTENSOR_DEVICE_CUDA
  auto result = cufftExecZ2Z(handle, indata, outdata, direction);
  assert(result == CUFFT_SUCCESS);
#elif defined(GTENSOR_DEVICE_HIP)
  auto result = hipfftExecZ2Z(handle, indata, outdata, direction);
  assert(result == rocfft_success);
#elif defined(GTENSOR_DEVICE_SYCL)
  cl::sycl::event e;
  if (direction == GPUFFT_FORWARD) {
    e = oneapi::mkl::dft::compute_forward(*handle, indata, outdata);
  } else {
    e = oneapi::mkl::dft::compute_backward(*handle, indata, outdata);
  }
  e.wait();
#endif
}

void gpufft_exec_c2c(gpufft_handle_t handle, gpufft_complex_t* indata,
                     gpufft_complex_t* outdata, int direction)
{
#ifdef GTENSOR_DEVICE_CUDA
  auto result = cufftExecC2C(handle, indata, outdata, direction);
  assert(result == CUFFT_SUCCESS);
#elif defined(GTENSOR_DEVICE_HIP)
  auto result = hipfftExecC2C(handle, indata, outdata, direction);
  assert(result == rocfft_success);
#elif defined(GTENSOR_DEVICE_SYCL)
  cl::sycl::event e;
  if (direction == GPUFFT_FORWARD) {
    e = oneapi::mkl::dft::compute_forward(*handle, indata, outdata);
  } else {
    e = oneapi::mkl::dft::compute_backward(*handle, indata, outdata);
  }
  e.wait();
#endif
}
