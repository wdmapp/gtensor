/*
 * ============================================================================
 *
 *       Filename:  gtblas_axpy.cxx
 *
 *    Description: Example showing how to use gt-blas
 *
 *        Version:  1.0
 *        Created:  01/06/2020 10:28:50 AM
 *       Revision:  none
 *       Compiler:  gcc, nvcc
 *
 *         Author:  Bryce Allen (bdallen@uchicago.edu)
 *
 * ===========================================================================
 */

#include <iostream>

#include <gtensor/gtensor.h>

#include <gt-blas/blas.h>

// provides convenient shortcuts for common gtensor functions, for example
// underscore ('_') to represent open slice ends.
using namespace gt::placeholders;

// #define EXPLICIT_KERNEL

int main(int argc, char** argv)
{
  int n = 1024 * 1024;
  int nprint = 32;

  double a = 0.5;

  // Define and allocate two 1d vectors of size n on the host. Declare
  // but don't allocate a third 1d host vector for storing the result.
  gt::gtensor<double, 1, gt::space::host> h_x(gt::shape(n));
  gt::gtensor<double, 1, gt::space::host> h_y(gt::shape(n));
  gt::gtensor<double, 1, gt::space::host> h_axpy(gt::shape(n));

  // initialize the vectors, x is twice it's index values and y is equal
  // to it's index values. We will perform .5*x + y, so the result should be
  // axpy(i) = 2i.
  for (int i = 0; i < n; i++) {
    h_x(i) = 2.0 * static_cast<double>(i);
    h_y(i) = static_cast<double>(i);
  }

  // Define and allocate device versions of h_x and h_y, and declare
  // a varaible for the result on gpu.
  gt::gtensor_device<double, 1> d_x(gt::shape(n));
  gt::gtensor_device<double, 1> d_axpy(gt::shape(n));

  // Explicit copies of input from host to device. Note that this is an
  // overload of the copy function for gtensor and gtensor_span types, not
  // std::copy which has a different signature. The source is the first
  // argument and destination the second argument. Currently thrust::copy is
  // used under the hood in the implementation.
  gt::copy(h_x, d_x);
  gt::copy(h_y, d_axpy);

  gt::blas::handle_t h;

  gt::blas::axpy(h, a, d_x, d_axpy);
  gt::copy(d_axpy, h_axpy);

  // Define a slice to print a subset of elements for spot checking the
  // result.
  auto print_slice = gt::gslice(_, _, n / nprint);
  std::cout << "a       = " << a << std::endl;
  std::cout << "x       = " << h_x.view(print_slice) << std::endl;
  std::cout << "y       = " << h_y.view(print_slice) << std::endl;
  std::cout << "a*x + y = " << h_axpy.view(print_slice) << std::endl;
}
