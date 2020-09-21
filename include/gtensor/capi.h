#ifndef GTENSOR_CAPI_H
#define GTENSOR_CAPI_H

/**
 * Expose a limited subset of gtensor API for calling from Fortran.
 *
 * The C++ namespaces are adapted to underscores when possible. For
 * allocation, the calls are simplified to exclude the allocator type.
 */

extern "C" void gt_synchronize();

#ifdef GTENSOR_HAVE_DEVICE

#ifndef GTENSOR_DEVICE_SYCL

extern "C" int gt_backend_device_get_count();
extern "C" void gt_backend_device_set(int device_id);
extern "C" int gt_backend_device_get();
extern "C" uint32_t gt_backend_device_get_vendor_id(int device_id);

#endif // not GTENSOR_DEVICE_SYCL

extern "C" void* gt_backend_device_allocate(int nbytes);
extern "C" void* gt_backend_managed_allocate(int nbytes);
extern "C" void gt_backend_device_deallocate(void* p);
extern "C" void gt_backend_managed_deallocate(void* p);

#endif // GTENSOR_HAVE_DEVICE

#endif // GTENSOR_CAPI_H
