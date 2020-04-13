# gtensor

GTensor is a multi-dimensional array C++14 header-only library for hybrid GPU
development. It was inspired by
[xtensor](https://xtensor.readthedocs.io/en/latest/), and designed to support
the GPU port of the [GENE](http://genecode.org) fusion code.

Features:
- multi-dimensional arrays and array views, with easy interoperability
  with Fortran and thrust
- automatically generate GPU kernels based on array operations
- define complex re-usable operations with lazy evaluation. This allows
  operations to be composed in different ways and evaluated once as a single
  kernel
- easily support both CPU-only and GPU-CPU hybrid code in the same code base,
  with only minimal use of #ifdef.
- multi-dimensional array slicing similar to numpy
- GPU support for nVidia via CUDA and AMD via HIP/rocm. Future plans to
  support Intel GPUs

## Getting Started

## Installation

GTensor is a header only library - to use, simply add the gtensor base
directory to your projects include path. Building with CUDA or HIP support
requires some more complex setup - see the examples
(CMakeLists.txt)[examples/CMakeLists.txt] as a starting point for how to do
this using cmake. See also the build line below for the daxpy example.

### Basic Example (host CPU only)

Here is a simple example that computes a matrix with the multiplication
table and prints it out row by row using array slicing:

```c++
#include <iostream>

#include <gtensor/gtensor.h>

int main(int argc, char **argv) {
    const int n = 9;
    gt::gtensor<int, 2> mult_table(gt::shape(n, n));

    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            mult_table(i,j) = (i+1)*(j+1);
        }
    }

    for (int i=0; i<n; i++) {
        std::cout << mult_table.view(i, gt::all()) << std::endl;
    }
}

```

It can be built like this, using gcc version 5 or later:
```
g++ -std=c++14 -I /path/to/gtensor/include -o mult_table mult_table.cxx
```

and produces the following output:
```
{ 1 2 3 4 5 6 7 8 9 }
{ 2 4 6 8 10 12 14 16 18 }
{ 3 6 9 12 15 18 21 24 27 }
{ 4 8 12 16 20 24 28 32 36 }
{ 5 10 15 20 25 30 35 40 45 }
{ 6 12 18 24 30 36 42 48 54 }
{ 7 14 21 28 35 42 49 56 63 }
{ 8 16 24 32 40 48 56 64 72 }
{ 9 18 27 36 45 54 63 72 81 }
```

See the full [mult\_table example](examples/src/mult_table.cxx) for different
ways of performing this operation, taking advantage of more gtensor features.

### GPU and CPU Example

The following program computed vector product `a*x + y`, where `a` is a scalar
and `x` and `y` are vectors. If build with `GTENSOR_HAVE_DEVICE` defined and
using the appropriate compiler (currently either nvcc or hipcc), it will run
the computation on a GPU device.

See the full [daxpy example](examples/src/daxpy.cxx) for more detailed comments
and an example of using an explicit kernel.

```
#include <iostream>

#include <gtensor/gtensor.h>

using namespace std;

// provides convenient shortcuts for common gtensor functions, for example
// underscore ('_') to represent open slice ends.
using namespace gt::placeholders;

template <typename S>
gt::gtensor<double, 1, S> daxpy(double a, const gt::gtensor<double, 1, S> &x,
                                const gt::gtensor<double, 1, S> &y) {
    return a * x + y;
}

int main(int argc, char **argv)
{
    int n = 1024 * 1024;
    int nprint = 32;

    double a = 0.5;

    // Define and allocate two 1d vectors of size n on the host.
    gt::gtensor<double, 1, gt::space::host> h_x(gt::shape(n));
    gt::gtensor<double, 1, gt::space::host> h_y = gt::empty_like(h_x);
    gt::gtensor<double, 1, gt::space::host> h_axpy;

    // initialize host vectors
    for (int i=0; i<n; i++) {
        h_x(i) = 2.0 * static_cast<double>(i);
        h_y(i) = static_cast<double>(i);
    }

#ifdef GTENSOR_HAVE_DEVICE
    cout << "gtensor have device" << endl;

    // Define and allocate device versions of h_x and h_y, and declare
    // a varaible for the result on gpu.
    gt::gtensor<double, 1, gt::space::device> d_x(gt::shape(n));
    gt::gtensor<double, 1, gt::space::device> d_y = gt::empty_like(d_x);
    gt::gtensor<double, 1, gt::space::device> d_axpy;
 
    // Explicit copies of input from host to device.
    copy(h_x, d_x);
    copy(h_y, d_y);

    // This automatically generates a computation kernel to run on the
    // device.
    d_axpy = daxpy(a, d_x, d_y);

    // Explicit copy of result to host
    h_axpy = gt::empty_like(h_x);
    copy(d_axpy, h_axpy);
#else
    // host implementation - simply call directly using host gtensors
    h_axpy = daxpy(a, h_x, h_y);
#endif // GTENSOR_HAVE_DEVICE

    // Define a slice to print a subset of elements for checking result
    auto print_slice = gt::gslice(_, _, n/nprint);
    cout << "a       = " << a << endl;
    cout << "x       = " << h_x.view(print_slice)  << endl;
    cout << "y       = " << h_y.view(print_slice)  << endl;
    cout << "a*x + y = " << h_axpy.view(print_slice) << endl;
}
```

Example build for nVidia GPU using nvcc:
```
GTENSOR_HOME=/path/to/gtensor
nvcc -x cu -std=c++14 --expt-extended-lambda --expt-relaxed-constexpr \
 -DGTENSOR_HAVE_DEVICE -DNDEBUG -O3 \
 -I $GTENSOR_HOME/include \
 -o daxpy_cuda daxpy.cxx
```

Build for AMD GPU using hipcc:
```
hipcc -hc -std=c++14 \
 -DGTENSOR_HAVE_DEVICE -DNDEBUG -O3 \
 -I $GTENSOR_HOME/include \
 -isystem /opt/rocm/rocthrust/include \
 -isystem /opt/rocm/include \
 -isystem /opt/rocm/rocprim/include \
 -isystem /opt/rocm/hip/include \
 -o daxpy_hip daxpy.cxx
```

Build for host CPU:
```
g++ -std=c++14 \
 -DNDEBUG -O3 \
 -I $GTENSOR_HOME/include \
 -o daxpy_host daxpy.cxx
```

### Example using gtensor with existing GPU code

If you have existing code written in CUDA or HIP, you can use the `gt::adapt`
and `gt::adapt_device` functions to wrap existing allocated host and device
memory in gtensor view containers. This allows you to use the convenience of
gtensor for new code without having to do an extensive rewrite.

See [trig.cu](examples/src/trig.cu) and
[trig_adapted.cxx](examples/src/trig_adapted.cxx). The same approach will work
for HIP with minor modifications.
