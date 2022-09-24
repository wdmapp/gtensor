/**
 * This file, along with gpu_api_interface.F90, is designed to replace
 * cuda_helpers.F90 and be GPU vendor independent. Initially it will only
 * support cuda, AMD / HIP support to follow shortly, and eventually Intel
 * GPU support. CUDA specific language features should be avoided - this is
 * a .cu only temporarily to make sure the correct headers are included by
 * the build system. C++ features are fine to use, and will likely be
 * required to support Intel GPUs.
 *
 * The current focus is on device selection, which needs to be called
 * directly from GENE fortran before starting any computation. Other
 * GPU operations will likely have more abstraction in the C/C++ implementation
 * and won't require calling device runtime API directly.
 *
 * Rather than handle the complexity of moving strings between C and Fortran,
 * most routines will print error to stderr and force an immediate exit. If
 * we encounter routines that have a use for non-fatal failure modes, we can
 * revist this. Error codes are also vendor specific, so returning them to the
 * Fortran layer just increases complexity in the absense of compelling
 * non-fatal error cases.
 *
 * See also:
 *  https://rocm-documentation.readthedocs.io/en/latest/ROCm_API_References/HIP_API/Device-management.html
 *  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#multi-device-system
 *  https://www.khronos.org/registry/SYCL/specs/sycl-1.2.pdf
 */

#ifdef GTENSOR_DEVICE_CUDA
#include "cuda_profiler_api.h"
#include "cuda_runtime_api.h"
#elif defined(GTENSOR_DEVICE_HIP)
#include "hip/hip_runtime.h"
#endif

#ifdef GTENSOR_DEVICE_SYCL
#include "gtensor/backend_sycl_device.h"
#endif

#include "gtensor/gtensor.h"

extern "C" void sycl_level_zero_mem_info(size_t* free, size_t* total);

extern "C" void gpuMemGetInfo(size_t* free, size_t* total)
{
#ifdef GTENSOR_DEVICE_CUDA
  gtGpuCheck(cudaMemGetInfo(free, total));
#elif defined(GTENSOR_DEVICE_HIP)
  gtGpuCheck(hipMemGetInfo(free, total));
#elif defined(GTENSOR_DEVICE_SYCL) && defined(GTENSOR_DEVICE_SYCL_L0)
  // Note: must set ZES_ENABLE_SYSMAN=1 in env for this to work
  sycl_level_zero_mem_info(free, total);
#else
  // fallback so compiles and not divide by zero
  *total = 1;
  *free = 1;
#endif
}

#ifdef GTENSOR_DEVICE_CUDA
extern "C" void gpuDeviceSetSharedMemConfig(cudaSharedMemConfig config)
{
  gtGpuCheck(cudaDeviceSetSharedMemConfig(config));
}
#elif defined(GTENSOR_DEVICE_HIP)
extern "C" void gpuDeviceSetSharedMemConfig(hipSharedMemConfig config)
{
  gtGpuCheck(hipDeviceSetSharedMemConfig(config));
}
#endif

extern "C" void gpuProfilerStart()
{
#ifdef GTENSOR_DEVICE_CUDA
  gtGpuCheck(cudaProfilerStart());
#endif
}

extern "C" void gpuProfilerStop()
{
#ifdef GTENSOR_DEVICE_CUDA
  gtGpuCheck(cudaProfilerStop());
#endif
}

extern "C" void gpuCheckLastError()
{
#ifdef GTENSOR_DEVICE_CUDA
  gtGpuCheck(cudaGetLastError());
#elif defined(GTENSOR_DEVICE_HIP)
  gtGpuCheck(hipGetLastError());
#elif defined(GTENSOR_DEVICE_SYCL)
  static int i = 10;
#endif
}

extern "C" void gpuDeviceReset()
{
#ifdef GTENSOR_DEVICE_CUDA
  cudaDeviceReset();
#elif defined(GTENSOR_DEVICE_HIP)
  hipDeviceReset();
#elif defined(GTENSOR_DEVICE_SYCL)
#warning "device reset not yet implemented for SYCL"
#endif
}

#ifdef GTENSOR_DEVICE_CUDA
extern "C" int gpuStreamCreate(cudaStream_t* pStream)
{
  return static_cast<int>(cudaStreamCreate(pStream));
}

extern "C" int gpuStreamDestroy(cudaStream_t stream)
{
  return static_cast<int>(cudaStreamDestroy(stream));
}

extern "C" int gpuStreamSynchronize(cudaStream_t stream)
{
  return static_cast<int>(cudaStreamSynchronize(stream));
}

extern "C" int gpuMemcpyAsync(void* dst, const void* src, size_t bytes,
                              cudaMemcpyKind kind, cudaStream_t stream)
{
  return static_cast<int>(cudaMemcpyAsync(dst, src, bytes, kind, stream));
}
#elif defined(GTENSOR_DEVICE_HIP)
extern "C" int gpuStreamCreate(hipStream_t* pStream)
{
  return static_cast<int>(hipStreamCreate(pStream));
}

extern "C" int gpuStreamDestroy(hipStream_t stream)
{
  return static_cast<int>(hipStreamDestroy(stream));
}

extern "C" int gpuStreamSynchronize(hipStream_t stream)
{
  return static_cast<int>(hipStreamSynchronize(stream));
}

extern "C" int gpuMemcpyAsync(void* dst, const void* src, size_t bytes,
                              hipMemcpyKind kind, hipStream_t stream)
{
  return static_cast<int>(hipMemcpyAsync(dst, src, bytes, kind, stream));
}

#elif defined(GTENSOR_DEVICE_SYCL)

#include "gtensor/gtensor.h"

// dummy implementation, one GPU only
extern "C" int gpuStreamCreate(sycl::queue** pStream)
{
  *pStream = &gt::backend::sycl::new_stream_queue();
  return 0;
}

extern "C" int gpuStreamDestroy(sycl::queue* stream)
{
  if (stream != nullptr) {
    gt::backend::sycl::delete_stream_queue(*stream);
  }
  return 0;
}

inline sycl::queue& sycl_get_queue(void* stream)
{
  if (stream == nullptr) {
    return gt::backend::sycl::get_queue();
  } else {
    return *(static_cast<sycl::queue*>(stream));
  }
}

inline sycl::queue& sycl_get_queue(void* stream, int device_id)
{
  if (stream == nullptr) {
    return gt::backend::sycl::get_queue(device_id);
  } else {
    return *(static_cast<sycl::queue*>(stream));
  }
}

#ifdef GTENSOR_DEVICE_SYCL_L0

extern "C" void sycl_level_zero_mem_info(size_t* free, size_t* total)
{
  zes_mem_state_t memory_props{
    ZES_STRUCTURE_TYPE_MEM_PROPERTIES,
  };

  auto q = sycl_get_queue(nullptr);
  auto d = q.get_device();

  // Get level-zero device handle
  auto ze_dev =
    cl::sycl::get_native<cl::sycl::backend::ext_oneapi_level_zero>(d);

  uint32_t n_mem_modules = 1;
  std::vector<zes_mem_handle_t> module_list(n_mem_modules);
  zesDeviceEnumMemoryModules(ze_dev, &n_mem_modules, module_list.data());

  zesMemoryGetState(module_list[0], &memory_props);
  *total = memory_props.size;
  *free = memory_props.free;
}

#endif // GTENSOR_DEVICE_SYCL_L0

extern "C" int gpuStreamSynchronize(sycl::queue* stream)
{
  sycl::queue& q = sycl_get_queue(stream);
  q.wait();
  return 0;
}

extern "C" int gpuMemcpyAsync(void* dst, const void* src, size_t bytes,
                              int kind, void* stream)
{
  sycl::queue& q = sycl_get_queue(stream);
  q.memcpy(dst, src, bytes);
  return 0;
}

#endif
