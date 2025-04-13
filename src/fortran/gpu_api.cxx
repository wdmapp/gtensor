/**
 * This file, along with gpu_api_interface.F90, is designed to provide
 * GPU vendor-independent functionality to Fortran.
 *
 * However, most of what's here should be moved to cgtensor and then
 * directly interfaced.
 *
 * The current focus is on device selection, which needs to be called
 * directly from GENE fortran before starting any computation. Other
 * GPU operations will likely have more abstraction in the C/C++ implementation
 * and won't require calling device runtime API directly.
 *
 * Rather than handle the complexity of moving strings between C and Fortran,
 * most routines will print error to stderr and force an immediate exit. If
 * we encounter routines that have a use for non-fatal failure modes, we can
 * revisit this. Error codes are also vendor specific, so returning them to the
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
#if HIP_VERSION_MAJOR >= 6
#include "roctracer/roctracer_ext.h"
#else
#include "roctracer_ext.h"
#endif
#endif

#ifdef GTENSOR_DEVICE_SYCL
#include "gtensor/backend_sycl_device.h"
#endif

#include "gtensor/gtensor.h"

extern "C" void gpuMemGetInfo(size_t* free, size_t* total)
{
  gt::backend::clib::mem_info(free, total);
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
#elif defined(GTENSOR_DEVICE_HIP)
  roctracer_start();
#endif
}

extern "C" void gpuProfilerStop()
{
#ifdef GTENSOR_DEVICE_CUDA
  gtGpuCheck(cudaProfilerStop());
#elif defined(GTENSOR_DEVICE_HIP)
  roctracer_stop();
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

extern "C" int gpuStreamCreateAsync(cudaStream_t* pStream)
{
  return static_cast<int>(
    cudaStreamCreateWithFlags(pStream, cudaStreamNonBlocking));
}

extern "C" int gpuStreamDestroy(cudaStream_t stream)
{
  return static_cast<int>(cudaStreamDestroy(stream));
}

extern "C" int gpuStreamSynchronize(cudaStream_t stream)
{
  return static_cast<int>(cudaStreamSynchronize(stream));
}

extern "C" int gpuEventCreate(cudaEvent_t* event)
{
  return static_cast<int>(cudaEventCreate(event));
}

extern "C" int gpuEventDestroy(cudaEvent_t event)
{
  return static_cast<int>(cudaEventDestroy(event));
}

extern "C" int gpuEventRecord(cudaEvent_t event, cudaStream_t stream)
{
  return static_cast<int>(cudaEventRecord(event, stream));
}

extern "C" int gpuEventSynchronize(cudaEvent_t event)
{
  return static_cast<int>(cudaEventSynchronize(event));
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

extern "C" int gpuStreamCreateAsync(hipStream_t* pStream)
{
  return static_cast<int>(
    hipStreamCreateWithFlags(pStream, hipStreamNonBlocking));
}

extern "C" int gpuStreamDestroy(hipStream_t stream)
{
  return static_cast<int>(hipStreamDestroy(stream));
}

extern "C" int gpuStreamSynchronize(hipStream_t stream)
{
  return static_cast<int>(hipStreamSynchronize(stream));
}

extern "C" int gpuEventCreate(hipEvent_t* event)
{
  return static_cast<int>(hipEventCreate(event));
}

extern "C" int gpuEventDestroy(hipEvent_t event)
{
  return static_cast<int>(hipEventDestroy(event));
}

extern "C" int gpuEventRecord(hipEvent_t event, hipStream_t stream)
{
  return static_cast<int>(hipEventRecord(event, stream));
}

extern "C" int gpuEventSynchronize(hipEvent_t event)
{
  return static_cast<int>(hipEventSynchronize(event));
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

extern "C" int gpuStreamCreateAsync(sycl::queue** pStream)
{
  // TODO: do we need to change queue properties for this
  // to be as close as possible to HIP/CUDA?
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

#elif defined(GTENSOR_DEVICE_HOST)

// dummy implementation, one GPU only
extern "C" int gpuStreamCreate(gt::stream_view::stream_t* pStream) { return 0; }

extern "C" int gpuStreamCreateAsync(gt::stream_view::stream_t* pStream)
{
  return 0;
}

extern "C" int gpuStreamDestroy(gt::stream_view::stream_t stream) { return 0; }

extern "C" int gpuStreamSynchronize(gt::stream_view::stream_t stream)
{
  return 0;
}

#endif
