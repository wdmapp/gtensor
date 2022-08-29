
module test_fortran_m
   use, intrinsic :: iso_c_binding
   use :: flcl_ndarray_mod
   implicit none

   private
   public :: f_test_arr2d

   interface
      subroutine c_test_arr2d(arr) bind(c)
         import
         type(nd_array_t) :: arr
      end subroutine c_test_arr2d
   end interface

contains

   subroutine f_test_arr2d(arr)
      real, dimension(:,:) :: arr

      call c_test_arr2d(to_nd_array(arr))
   end subroutine f_test_arr2d

end module test_fortran_m

program main
   use :: test_fortran_m
   implicit none

   integer, parameter :: ni = 4, nj = 3
   real :: arr2d(ni, nj)
   integer :: i, j

   print *, shape(arr2d)
   do j = 1, nj
      do i = 1, ni
         arr2d(i, j) = i + 10 * j
      end do
   end do

   print *, 'arr2d', arr2d
   call f_test_arr2d(arr2d)

   print *, 'arr2d(3:4, 1:2)', arr2d(3:4, 1:2)
   call f_test_arr2d(arr2d(3:4, 1:2))
end program
