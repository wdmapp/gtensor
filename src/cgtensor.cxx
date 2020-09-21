#include <stdint.h>

#include <gtensor/gtensor.h>

extern "C" void gt_synchronize()
{
  gt::synchronize();
}

#ifdef GTENSOR_HAVE_DEVICE

#ifndef GTENSOR_DEVICE_SYCL

extern "C" int gt_backend_device_get_count()
{
  return gt::backend::device_get_count();
}

extern "C" void gt_backend_device_set(int device_id)
{
  return gt::backend::device_set(device_id);
}

extern "C" int gt_backend_device_get()
{
  return gt::backend::device_get();
}

extern "C" uint32_t gt_backend_device_get_vendor_id(int device_id)
{
  return gt::backend::device_get_vendor_id(device_id);
}

#endif // not GTENSOR_DEVICE_SYCL

extern "C" void* gt_backend_device_allocate(int nbytes)
{
  return (void*)gt::backend::device_allocator<uint8_t>::allocate(nbytes);
}

extern "C" void* gt_backend_managed_allocate(int nbytes)
{
  return (void*)gt::backend::managed_allocator<uint8_t>::allocate(nbytes);
}

extern "C" void gt_backend_device_deallocate(void* p)
{
  gt::backend::device_allocator<uint8_t>::deallocate((uint8_t*)p);
}

extern "C" void gt_backend_managed_deallocate(void* p)
{
  gt::backend::managed_allocator<uint8_t>::deallocate((uint8_t*)p);
}

#endif
