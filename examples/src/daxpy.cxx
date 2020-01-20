/*
 * =====================================================================================
 *
 *       Filename:  daxpy.cxx
 *
 *    Description: Example showing how element-wise operations are simple to
 *                 perform in Gtensor 
 *
 *        Version:  1.0
 *        Created:  01/06/2020 10:28:50 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include <iostream>
#include <thread>
#include <chrono>

#include <gtensor/gtensor.h>

using namespace std;

// provides convenient shortcuts for commont gtensor functions, for example
// _s for gt::gslice
using namespace gt::placeholders;


/**
 * Templated function for daxpy that can work with host gtensor or device
 * gtensor. Relies on C++11 mandatory copy elision to move the result into the
 * LHS it is assigned to, rather than copying all the data (very important for
 * large arrays).
 */
template <typename S>
gt::gtensor<double, 1, S> daxpy(double a, gt::gtensor<double, 1, S> x,
                                gt::gtensor<double, 1, S> y) {
    return a * x + y;
}


int main(int argc, char **argv)
{
    int n = 2048;

    double a = 0.5;
    gt::gtensor<double, 1, gt::space::host> h_x(gt::shape(n));
    gt::gtensor<double, 1, gt::space::host> h_y = gt::empty_like(h_x);
    gt::gtensor<double, 1, gt::space::host> h_axpy = gt::empty_like(h_x);

    for (int i=0; i<n; i++) {
        h_x(i) = 2.0 * static_cast<double>(i);
        h_y(i) = static_cast<double>(i);
    }

#ifdef WITH_CUDA
    gt::gtensor<double, 1, gt::space::device> d_x(gt::shape(n));
    gt::gtensor<double, 1, gt::space::device> d_y = gt::empty_like(d_x);
    gt::gtensor<double, 1, gt::space::device> d_axpy = gt::empty_like(d_x);

    copy(h_x, d_x);
    copy(h_y, d_y);

    auto k_x = d_x.to_kernel();
    auto k_y = d_y.to_kernel();
    auto k_axpy = d_axpy.to_kernel();

    gt::launch<1>(d_x.shape(), GT_LAMBDA(int i) mutable {
      k_axpy(i) = a * k_x(i) + k_y(i);
    });

    copy(d_axpy, h_axpy);
#else
    h_axpy = daxpy(a, h_x, h_y);
#endif

    auto print_slice = gt::gslice(_, _, 128);
    cout << "a       = " << a << endl;
    cout << "x       = " << h_x.view(print_slice)  << endl;
    cout << "y       = " << h_y.view(print_slice)  << endl;
    cout << "a*x + y = " << h_axpy.view(print_slice) << endl;
}
