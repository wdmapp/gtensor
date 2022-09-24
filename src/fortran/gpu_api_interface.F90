!> Provide vendor independent API for GPU device configuration and other
!! common functions.
!!
!! See gpu_api.cpp and cgtensor for the C/C++ implementations.
!!
!! Routines that wrap a similar cuda/hip function use the convention of
!! replacing cuda/hip with "gpu", for example cudaGetDevice -> gpuGetDevice.
!! However the API is different - subroutines are always used, and errors are
!! handled by abort() in C. This is to avoid the differences in error code
!! to meaning / string between vendors, and the complication of converting
!! from C strings to Fortran strings.

#define WITH_GPU
! Wrap non-void gtensor functions for use only inside gpu_api_m to define
! a subroutine version.
module cgtensor_private_m
   use,intrinsic :: iso_c_binding
   implicit none

#ifdef WITH_GPU
   interface

      function gt_backend_device_get_count() bind(c,name="gt_backend_device_get_count")
         import C_INT
         integer(C_INT) :: gt_backend_device_get_count
      end function gt_backend_device_get_count

      function gt_backend_device_get() bind(c,name="gt_backend_device_get")
         import C_INT
         integer(C_INT) :: gt_backend_device_get
      end function gt_backend_device_get

      function gt_backend_device_get_vendor_id(device_id) bind(c,name="gt_backend_device_get_vendor_id")
         import C_INT32_T, C_INT
         integer(C_INT),value :: device_id
         integer(C_INT32_T) :: gt_backend_device_get_vendor_id
      end function gt_backend_device_get_vendor_id

      function gt_backend_device_allocate(n_bytes) bind(c,name="gt_backend_device_allocate")
         import
         integer(C_SIZE_T),value :: n_bytes
         type(C_PTR) :: gt_backend_device_allocate
      end function gt_backend_device_allocate

      function gt_backend_managed_allocate(n_bytes) bind(c,name="gt_backend_managed_allocate")
         import
         integer(C_SIZE_T),value :: n_bytes
         type(C_PTR) :: gt_backend_managed_allocate
      end function gt_backend_managed_allocate

      function gt_backend_host_allocate(n_bytes) bind(c,name="gt_backend_host_allocate")
         import
         integer(C_SIZE_T),value :: n_bytes
         type(C_PTR) :: gt_backend_host_allocate
      end function gt_backend_host_allocate

      function gt_backend_is_device_address(ptr) bind(c,name="gt_backend_is_device_address")
         import
         type(C_PTR),value :: ptr
         logical(C_BOOL) :: gt_backend_is_device_address
      end function gt_backend_is_device_address

      subroutine gt_backend_prefetch_device(ptr, sz) bind(c)
         import
         type(C_PTR), value :: ptr
         integer(C_SIZE_T), value :: sz
      end subroutine gt_backend_prefetch_device

      subroutine gt_backend_prefetch_host(ptr, sz) bind(c)
         import
         type(C_PTR), value :: ptr
         integer(C_SIZE_T), value :: sz
      end subroutine gt_backend_prefetch_host

   end interface
#endif
end module cgtensor_private_m


module gpu_api_m
   use,intrinsic :: iso_c_binding
   use cgtensor_private_m
   implicit none

#ifdef WITH_GPU

   interface

      ! Defined in gpu_api.cu. Use is discouraged, as most of these do
      ! not have a clear SYCL analogue.
      subroutine gpuMemGetInfo(free,total) bind(c,name="gpuMemGetInfo")
         import
         integer(C_SIZE_T),intent(OUT) :: free,total
      end subroutine gpuMemGetInfo

      subroutine gpuDeviceSetSharedMemConfig( config ) &
           & bind(c,name="gpuDeviceSetSharedMemConfig")
         import C_INT
         integer(C_INT),value :: config
      end subroutine gpuDeviceSetSharedMemConfig

      subroutine gpuCheckLastError() bind(c,name="gpuCheckLastError")
         import
      end subroutine gpuCheckLastError

      subroutine gpuProfilerStart() bind(c,name="gpuProfilerStart")
         import
      end subroutine gpuProfilerStart

      subroutine gpuProfilerStop() bind(c,name="gpuProfilerStop")
         import
      end subroutine gpuProfilerStop

      function gpuStreamCreate(streamId) bind(c,name="gpuStreamCreate")
         import
         type(C_PTR),intent(INOUT) :: streamId
         integer(C_INT) :: gpuStreamCreate
      end function gpuStreamCreate

      function gpuStreamDestroy(streamId) bind(c,name="gpuStreamDestroy")
         import
         type(C_PTR),value :: streamId
         integer(C_INT) :: gpuStreamDestroy
      end function gpuStreamDestroy

      function gpuStreamSynchronize(streamid) bind(c,name="gpuStreamSynchronize")
         import
         type(C_PTR),value,intent(IN) :: streamid
         integer(C_INT) :: gpuStreamSynchronize
      end function gpuStreamSynchronize

      integer(C_INT) function gpuMemcpyAsync( dst, src, count, kind, streamId ) &
           & bind(c,name="gpuMemcpyAsync")
         import
         type(C_PTR),value :: dst
         type(C_PTR),value,intent(IN) :: src
         integer(C_SIZE_T),value :: count
         integer(C_INT),value :: kind
         type(C_PTR), value :: streamId
      end function gpuMemcpyAsync

      ! Wrap gtensor void functions as subroutines and expose them directly.
      ! Non-void functions are defined above in cgtensor_private_m and
      ! a "gpu" prefix named subroutine is defined below.
      subroutine gpuDeviceSet(device_id) bind(c,name="gt_backend_device_set")
         import C_INT
         integer(C_INT),value :: device_id
      end subroutine gpuDeviceSet

      subroutine gpuDeallocateDevice(device_ptr) bind(c,name="gt_backend_device_deallocate")
         import
         type(C_PTR),value :: device_ptr
      end subroutine gpuDeallocateDevice

      subroutine gpuDeallocateManaged(managed_ptr) bind(c,name="gt_backend_managed_deallocate")
         import
         type(C_PTR),value :: managed_ptr
      end subroutine gpuDeallocateManaged

      subroutine gpuDeallocateHost(host_ptr) bind(c,name="gt_backend_host_deallocate")
         import
         type(C_PTR),value :: host_ptr
      end subroutine gpuDeallocateHost

      subroutine gpuMemset( dst, val, n_bytes ) bind(c,name="gt_backend_memset")
         import
         type(C_PTR), value :: dst
         integer(C_INT),value :: val
         integer(C_SIZE_T),value :: n_bytes
      end subroutine gpuMemset

      subroutine gpuMemcpyDD( dst, src, n_bytes) bind(c,name="gt_backend_memcpy_dd")
         import
         type(C_PTR),value :: dst
         type(C_PTR),value,intent(IN) :: src
         integer(C_SIZE_T),value :: n_bytes
      end subroutine gpuMemcpyDD

      subroutine gpuMemcpyAsyncDD( dst, src, n_bytes ) &
           & bind(c,name="gt_backend_memcpy_async_dd")
         import
         type(C_PTR),value :: dst
         type(C_PTR),value,intent(IN) :: src
         integer(C_SIZE_T),value :: n_bytes
      end subroutine gpuMemcpyAsyncDD

      subroutine gpuMemcpyDH( dst, src, n_bytes) bind(c,name="gt_backend_memcpy_dh")
         import
         type(C_PTR),value :: dst
         type(C_PTR),value,intent(IN) :: src
         integer(C_SIZE_T),value :: n_bytes
      end subroutine gpuMemcpyDH

      subroutine gpuMemcpyHD( dst, src, n_bytes) bind(c,name="gt_backend_memcpy_hd")
         import
         type(C_PTR),value :: dst
         type(C_PTR),value,intent(IN) :: src
         integer(C_SIZE_T),value :: n_bytes
      end subroutine gpuMemcpyHD

      subroutine gpuDeviceSynchronize() bind(c,name="gt_synchronize")
      end subroutine gpuDeviceSynchronize

      subroutine gpuSynchronize() bind(c,name="gt_synchronize")
      end subroutine gpuSynchronize

      subroutine gpuDeviceReset() bind(c,name="gpuDeviceReset")
         import
      end subroutine gpuDeviceReset

   end interface
#endif
contains

   !> Get gpu device id based on MPI rank, ranks per node, and devices per node
   !! Devices are assigned linearly within a node, i.e. if there are
   !! 5 ranks per GPU, the first 5 ranks get GPU 0, the next GPU 1, etc.
   !! This is in contrast to round robin assignment, which could be
   !! added as an option in the future if needed.
   !! Assumptions:
   !!  - all nodes have same number of devices and same number of MPI ranks.
   !!  - ranks are assigned to nodes in a linear fashion
   function get_gpu_device_for_rank(rank, ranks_per_node, devices_per_node) result(device_id)
      integer, intent(IN) :: rank, ranks_per_node, devices_per_node
      integer :: device_id
      integer :: node_rank, ranks_per_device

      node_rank = modulo(rank, ranks_per_node)
      ! Note: purposely truncate, non-divisor case handled by modulo below
      ranks_per_device = ranks_per_node / devices_per_node
      if (ranks_per_device == 0) then
         ! ranks < devices per node
         ranks_per_device = 1
      end if

      ! Note: integer divide by ranks_per_device => linear assignment
      !       modulo devices per node => round robin
      ! Modulo is to handle uneven cases, e.g. 6 mpi 5 gpu
      device_id = modulo(node_rank / ranks_per_device, devices_per_node)
   end function get_gpu_device_for_rank

#ifdef WITH_GPU
   !> Get a 10 character string representing the vendor id of the GPU
   !! device at the specified gpu_id index.
   !! For CUDA and HIP, this will be the PCI Address, but other vendors may
   !! use something else, and if there is an error "00000000" is returned.
   !! This is primarily for debugging purposes to verify MPI rank to GPU
   !! mappings.
   function get_gpu_device_vendor_id(gpu_id) result(vendor_id)
      character(len=10) :: vendor_id
      integer, intent(IN) :: gpu_id
      integer(kind=C_INT32_T) :: packed

      packed = gt_backend_device_get_vendor_id(gpu_id)

      write(vendor_id, '(Z0.8)') packed
   end function get_gpu_device_vendor_id

   subroutine gpuAllocateDevice(out_ptr, n_bytes)
      type(C_PTR),intent(OUT) :: out_ptr
      integer(C_SIZE_T),value :: n_bytes

      out_ptr = gt_backend_device_allocate(n_bytes)
   end subroutine gpuAllocateDevice

   subroutine gpuAllocateManaged(out_ptr, n_bytes)
      type(C_PTR),intent(OUT) :: out_ptr
      integer(C_SIZE_T),value :: n_bytes

      out_ptr = gt_backend_managed_allocate(n_bytes)
   end subroutine gpuAllocateManaged

   subroutine gpuAllocateHost(out_ptr, n_bytes)
      type(C_PTR),intent(OUT) :: out_ptr
      integer(C_SIZE_T),value :: n_bytes

      out_ptr = gt_backend_host_allocate(n_bytes)
   end subroutine gpuAllocateHost

   subroutine gpuDeviceGetCount(out_count)
      integer(C_INT),intent(OUT) :: out_count

      out_count = gt_backend_device_get_count()
   end subroutine gpuDeviceGetCount

   subroutine gpuDeviceGet(out_device_id)
      integer(C_INT),intent(OUT) :: out_device_id

      out_device_id = gt_backend_device_get()
   end subroutine gpuDeviceGet

   function gpuIsDeviceAddressComplex(a)
      complex, dimension(*), target :: a
      logical(C_BOOL) :: gpuIsDeviceAddressComplex

      gpuIsDeviceAddressComplex = gt_backend_is_device_address(c_loc(a))
   end function gpuIsDeviceAddressComplex

   function gpuIsDeviceAddressReal(a)
      real, dimension(*), target :: a
      logical(C_BOOL) :: gpuIsDeviceAddressReal

      gpuIsDeviceAddressReal = gt_backend_is_device_address(c_loc(a))
   end function gpuIsDeviceAddressReal

   function gpuIsDeviceAddressInteger(a)
      integer, dimension(*), target :: a
      logical(C_BOOL) :: gpuIsDeviceAddressInteger

      gpuIsDeviceAddressInteger = gt_backend_is_device_address(c_loc(a))
   end function gpuIsDeviceAddressInteger

   subroutine printAddressComplex(varname, a, fname, line)
      character(len=*), intent(in) :: varname
      complex, dimension(*), intent(in), target :: a
      character(len=*), intent(in), optional :: fname
      integer, intent(in), optional :: line

      if (present(fname) .and. present(line)) then
         write(*,"(A,':',I0,'  ',A,'& 0x',Z0,' gpu? ',L)") fname, line, varname,&
              & loc(a), gpuIsDeviceAddressComplex(a)
      else
         write(*,"(A,'& 0x',Z0,' gpu? ',L)") varname, loc(a), &
              & gpuIsDeviceAddressComplex(a)
      end if
   end subroutine printAddressComplex

   subroutine printAddressReal(varname, a, fname, line)
      character(len=*), intent(in) :: varname
      real, dimension(*), intent(in), target :: a
      character(len=*), intent(in), optional :: fname
      integer, intent(in), optional :: line

      if (present(fname) .and. present(line)) then
         write(*,"(A,':',I0,'  ',A,'& 0x',Z0,' gpu? ',L)") fname, line, varname,&
              & loc(a), gpuIsDeviceAddressReal(a)
      else
         write(*,"(A,'& 0x',Z0,' gpu? ',L)") varname, loc(a), &
              & gpuIsDeviceAddressReal(a)
      end if
   end subroutine printAddressReal

   subroutine printAddressInteger(varname, a, fname, line)
      character(len=*), intent(in) :: varname
      integer, dimension(*), intent(in), target :: a
      character(len=*), intent(in), optional :: fname
      integer, intent(in), optional :: line

      if (present(fname) .and. present(line)) then
         write(*,"(A,':',I0,'  ',A,'& 0x',Z0,' gpu? ',L)") fname, line, varname,&
              & loc(a), gpuIsDeviceAddressInteger(a)
      else
         write(*,"(A,'& 0x',Z0,' gpu? ',L)") varname, loc(a), &
              & gpuIsDeviceAddressInteger(a)
      end if
   end subroutine printAddressInteger

   subroutine gpuAssertDeviceAddressComplex(varname, a, fname, line)
      character(len=*), intent(in) :: varname
      complex, dimension(*), target :: a
      character(len=*), intent(in), optional :: fname
      integer, intent(in), optional :: line

      if (.not. gpuIsDeviceAddressComplex(a)) then
         if (present(fname) .and. present(line)) then
            call printAddressComplex(varname, a, fname, line)
         else
            call printAddressComplex(varname, a)
         end if
         call abort()
      end if
   end subroutine gpuAssertDeviceAddressComplex

   subroutine gpuAssertDeviceAddressReal(varname, a, fname, line)
      character(len=*), intent(in) :: varname
      real, dimension(*), target :: a
      character(len=*), intent(in), optional :: fname
      integer, intent(in), optional :: line

      if (.not. gpuIsDeviceAddressReal(a)) then
         if (present(fname) .and. present(line)) then
            call printAddressReal(varname, a, fname, line)
         else
            call printAddressReal(varname, a)
         end if
         call abort()
      end if
   end subroutine gpuAssertDeviceAddressReal

   subroutine gpuAssertDeviceAddressInteger(varname, a, fname, line)
      character(len=*), intent(in) :: varname
      integer, dimension(*), target :: a
      character(len=*), intent(in), optional :: fname
      integer, intent(in), optional :: line

      if (.not. gpuIsDeviceAddressInteger(a)) then
         if (present(fname) .and. present(line)) then
            call printAddressInteger(varname, a, fname, line)
         else
            call printAddressInteger(varname, a)
         end if
         call abort()
      end if
   end subroutine gpuAssertDeviceAddressInteger

#endif

end module gpu_api_m
