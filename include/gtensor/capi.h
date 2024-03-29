#ifndef GTENSOR_CAPI_H
#define GTENSOR_CAPI_H

#include "device_runtime.h"
#include "macros.h"

/**
 * Expose a limited subset of gtensor API for calling from Fortran.
 *
 * The C++ namespaces are adapted to underscores when possible. For
 * allocation, the calls are simplified to exclude the allocator type.
 *
 * Also exposes macros gtLaunchKernel and gtGpuCheck for CUDA and AMD
 * backends. Note that these are not currently portable to SYCL, and may
 * be removed in the future.
 */

#ifdef __cplusplus
extern "C" {
#endif

void gt_synchronize();

int gt_backend_device_get_count();
void gt_backend_device_set(int device_id);
int gt_backend_device_get();
uint32_t gt_backend_device_get_vendor_id(int device_id);

void* gt_backend_host_allocate(size_t nbytes);
void* gt_backend_device_allocate(size_t nbytes);
void* gt_backend_managed_allocate(size_t nbytes);
void gt_backend_host_deallocate(void* p);
void gt_backend_device_deallocate(void* p);
void gt_backend_managed_deallocate(void* p);

#ifdef GTENSOR_HAVE_DEVICE

void gt_backend_memcpy_hh(void* dst, const void* src, size_t nbytes);
void gt_backend_memcpy_dd(void* dst, const void* src, size_t nbytes);
void gt_backend_memcpy_async_dd(void* dst, const void* src, size_t nbytes);
void gt_backend_memcpy_dh(void* dst, const void* src, size_t nbytes);
void gt_backend_memcpy_hd(void* dst, const void* src, size_t nbytes);

#endif // GTENSOR_HAVE_DEVICE

void gt_backend_memset(void* dst, int value, size_t nbytes);

bool gt_backend_is_device_accessible(void* p);
[[deprecated("use gt_backend_is_device_accessible() instead")]] bool
gt_backend_is_device_address(void* p);

void gt_backend_prefetch_device(void* p, size_t nbytes);
void gt_backend_prefetch_host(void* p, size_t nbytes);

#ifdef __cplusplus
}
#endif

#endif // GTENSOR_CAPI_H
