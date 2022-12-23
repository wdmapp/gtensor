# gtensor

![CCPP workflow](https://github.com/wdmapp/gtensor/actions/workflows/ccpp.yml/badge.svg)
![SYCL workflow](https://github.com/wdmapp/gtensor/actions/workflows/sycl.yml/badge.svg)

gtensor is a multi-dimensional array C++14 header-only library for hybrid GPU
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
- GPU support for nVidia via CUDA and AMD via HIP/ROCm,
  and experimental Intel GPU support via SYCL.
- [Experimental] C library cgtensor with wrappers around common GPU operations
  (allocate and deallocate, device management, memory copy and set)
- [Experimental] lightweight wrappers around GPU BLAS, LAPACK, and FFT
  routines.

## License

gtensor is licensed under the 3-clause BSD license. See the [LICENSE](LICENSE)
file for details.

## Installation (cmake)

gtensor uses cmake 3.13+ to build the tests and install:
```sh
git clone https://github.com/wdmapp/gtensor.git
cd gtensor
cmake -S . -B build -DGTENSOR_DEVICE=cuda \
  -DCMAKE_INSTALL_PREFIX=/opt/gtensor \
  -DBUILD_TESTING=OFF
cmake --build build --target install
```
To build for cpu/host only, use `-DGTENSOR_DEVICE=host`, for AMD/HIP use
`-DGTENSOR_DEVICE=hip -DCMAKE_CXX_COMPILER=$(which hipcc)`, and for
Intel/SYCL use `-DGTENSOR_DEVICE=sycl -DCMAKE_CXX_COMPILER=$(which dpcpp)`
See sections below for more device specific requirements.

Note that gtensor can still be used by applications not using cmake -
see [Usage (GNU make)](#usage-gnu-make) for an example.

To use the internal data vector implementation instead of thrust, set
`-DGTENSOR_USE_THRUST=OFF`. This has the advantage that device array
allocations will not be zero initialized, which can improve performance
significantly for some workloads, particularly when temporary arrays are
used.

To enable experimental C/C++ library features,`GTENSOR_BUILD_CLIB`,
`GTENSOR_BUILD_BLAS`, or `GTENSOR_BUILD_FFT` to `ON`. Note that BLAS
includes some LAPACK routines for LU factorization.

### nVidia CUDA requirements

gtensor for nVidia GPUs with CUDA requires
[CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 10.0+.

### AMD HIP requirements

gtensor for AMD GPUs with HIP requires ROCm 4.5.0+, and rocthrust and
rocprim. See the [ROCm installation
guide](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html)
for details. In Ubuntu, after setting up the ROCm repository, the required
packages can be installed like this:
```
sudo apt install rocm-dkms rocm-dev rocthrust
```
The official packages install to `/opt/rocm`. If using a different install
location, set the `ROCM_PATH` cmake variable. To use coarse grained
managed memory, ROCm 5.0+ is required.

To use gt-fft and gt-blas, rocsolver, rocblas, and rocfft packages need
to be installed as well.

### Intel SYCL requirements

The current SYCL implementation requires Intel OneAPI/DPC++ 2022.0 or later, with some known issues in
gt-blas and gt-fft (npvt getrf/rs, 2d fft). Using the latest available release is recommended.
When using the instructions at
[install via package managers](https://www.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/installation/install-using-package-managers.html),
installing the `intel-oneapi-dpcpp-compiler` package will pull in all required packages
(the rest of basekit is not required).

The reason for the dependence on Intel OneAPI is that the implementation uses
the USM extension, which is not part of the current SYCL standard.
CodePlay ComputeCpp 2.0.0 has an experimental implementation that is
sufficiently different to require extra work to support.

The default device selector is always used. To control device section, set the `SYCL_DEVICE_FILTER`
environment variable. See the
[intel llvm documentation](https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#sycl_device_filter)
for details.

The port is tested with an Intel iGPU, specifically UHD Graphics 630. It
may also work with the experimental CUDA backend for nVidia GPUs, but this
is untested and it's recommended to use the gtensor CUDA backend instead.

Better support for other SYCL implementations like hipSYCL and ComputCPP should be possible to add, with the
possible exception of gt-blas and gt-fft sub-libraries which require oneMKL.

### HOST CPU (no device) requirements

gtensor should build with any C++ compiler supporting C++14. It has been
tested with g++ 7, 8, and 9 and clang++ 8, 9, and 10.

### Advanced multi-device configuration

By default, gtensor will install support for the device specified by
the `GTENSOR_DEVICE` variable (default `cuda`), and also the `host` (cpu only)
device. This can be configured with `GTENSOR_BUILD_DEVICES` as a semicolon (;)
separated list. For example, to build support for all four backends
(assuming a machine with multi-vendor GPUs and associated toolkits installed).
```
cmake -S . -B build -DGTENSOR_DEVICE=cuda \
  -DGTENSOR_BUILD_DEVICES=host;cuda;hip;sycl \
  -DCMAKE_INSTALL_PREFIX=/opt/gtensor \
  -DBUILD_TESTING=OFF
```

This will cause targets to be created for each device: `gtensor::gtensor_cuda`,
`gtensor::gtensor_host`, `gtensor::gtensor_hip`, and `gtensor::gtensor_sycl`.
The main `gtensor::gtensor` target will be an alias for the default set by
`GTENSOR_DEVICE` (the cuda target in the above example).

## Usage (cmake)

Once installed, gtensor can be used by adding this to a project's
`CMakeLists.txt`:

```cmake
# if using GTENSOR_DEVICE=cuda
enable_language(CUDA)

find_library(gtensor)

# for each C++ target using gtensor
target_gtensor_sources(myapp PRIVATE src/myapp.cxx)
target_link_libraries(myapp gtensor::gtensor)
```

When running `cmake` for a project, add the gtensor
install prefix to `CMAKE_PREFIX_PATH`. For example:
```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/gtensor
```

The default gtensor device, set with the `GTENSOR_DEVICE` cmake variable
when installing gtensor, can be overridden by setting `GTENSOR_DEVICE`
again in the client application before the call to `find_library(gtensor)`,
typically via the `-D` cmake command line option. This can be useful to debug
an application by setting `-DGTENSOR_DEVICE=host`, to see if the problem is
related to the hybrid device model or is an algorithmic problem, or to run a
host-only interactive debugger. Note that only devices specified with
`GTENSOR_BUILD_DEVICES` at gtensor install time are available (the default
device and `host` if no option was specified).

### Using gtensor as a subdirectory or git submodule

gtensor also supports usage as a subdiretory of another cmake project. This
is typically done via git submodules. For example:
```sh
cd /path/to/app
git submodule add https://github.com/wdmapp/gtensor.git external/gtensor
```

In the application's `CMakeLists.txt`:
```cmake
# set here or on the cmake command-line with `-DGTENSOR_DEVICE=...`.
set(GTENSOR_DEVICE "cuda" CACHE STRING "")

if (${GTENSOR_DEVICE} STREQUAL "cuda")
  enable_language(CUDA)
endif()

# after setting GTENSOR_DEVICE
add_subdirectory(external/gtensor)

# for each C++ target using gtensor
target_gtensor_sources(myapp PRIVATE src/myapp.cxx)
target_link_libraries(myapp gtensor::gtensor)
```

## Usage (GNU make)

As a header only library, gtensor can be integrated into an existing
GNU make project as a subdirectory fairly easily for cuda and host devices.

The subdirectory is typically managed via git submodules, for example:
```sh
cd /path/to/app
git submodule add https://github.com/wdmapp/gtensor.git external/gtensor
```

See [examples/Makefile](examples/Makefile) for a good way of organizing a
project's Makefile to provide cross-device support. The examples can be
built for different devices by setting the `GTENSOR_DEVICE` variable,
e.g. `cd examples; make GTENSOR_DEVICE=host`.

## Getting Started

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
        std::cout << mult_table.view(i, gt::all) << std::endl;
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
 -DGTENSOR_HAVE_DEVICE -DGTENSOR_DEVICE_CUDA -DGTENSOR_USE_THRUST \
 -DNDEBUG -O3 \
 -I $GTENSOR_HOME/include \
 -o daxpy_cuda daxpy.cxx
```

Build for AMD GPU using hipcc:
```
hipcc -hc -std=c++14 \
 -DGTENSOR_HAVE_DEVICE -DGTENSOR_DEVICE_HIP -DGTENSOR_USE_THRUST \
 -DNDEBUG -O3 \
 -I $GTENSOR_HOME/include \
 -isystem /opt/rocm/rocthrust/include \
 -isystem /opt/rocm/include \
 -isystem /opt/rocm/rocprim/include \
 -isystem /opt/rocm/hip/include \
 -o daxpy_hip daxpy.cxx
```

Build for Intel GPU using dpcpp:
```
dpcpp -fsycl -std=c++14 \
 -DGTENSOR_HAVE_DEVICE -DGTENSOR_DEVICE_SYCL \
 -DGTENSOR_DEVICE_SYCL_GPU \
 -DNDEBUG -O3 \
 -I $GTENSOR_HOME/include \
 -o daxpy_sycl daxpy.cxx
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
memory in gtensor span containers. This allows you to use the convenience of
gtensor for new code without having to do an extensive rewrite.

See [trig.cu](examples/src/trig.cu) and
[trig_adapted.cxx](examples/src/trig_adapted.cxx). The same approach will work
for HIP with minor modifications.

# Data Types and mutability

gtensor has two types of data objects - those which are containers that own the
underlying data, like `gtensor`, and those which behave like span objects or
pointers, like `gtensor_span`. The `gview` objects, which are generally
constructed via the helper method `gt::view` or the convenience `view` methods
on `gtensor`, implement the slicing, broadcasting, and axis manipulation
functions, and have hybrid behavior based on the underlying expression. In
particular, a `gview` wrapping a `gtensor_span` object will have span-like
behavior, and in most other cases will have owning container behavior.

Before a data object can be passed to a GPU kernel, it must be converted to a
span-like object, and must be resident on the device. This generally happens
automatically when using expression evaluation and `gtensor_device`, but must
be done manually by calling the `to_kernel()` method when using custom kernels
with `gt::launch<N>`. What typically happens is that the underlying `gtensor`
objects get transformed to `gtensor_span` of the appropriate type. This happens
even when they are wrapped inside complex `gview` and `gfunction` objects.

The objects with span like behavior also have shallow const behavior. This
means that even if the outer object is const, they allow modification of the
underlying data. This is consistent with `std::span` standardized in C++20. The
idea is that if copying does not copy the underlying data (shallow copy), all
other aspects of the interface should behave similarly. This is called
"regularity". This also allows non-mutable lambdas to be used for launch
kernels. Non-mutable lambdas are important because SYCL requires const kernel
functions, so the left hand side of expressions must allow mutation of the
underlying data even when const because they may be contained inside a
non-mutable lambda and forced to be const.

To ensure const-correctness whenever possible, the `to_kernel()` routine on
`const gtensor<T, N, S>` is special cased to return a `gtensor_span<const T,
N, S>`. This makes it so even though a non-const reference is returned from the
element accessors (shallow const behavior of span like object), modification is
still not allowed since the underlying type is const.

To make this more concrete, here are some examples:

```
gtensor_device<int, 1> a{1, 2, 3};
const gtensor_device<int, 1> a_const_copy = a;

a(0) = 10; // fine
a_const_copy(0) = 1; // won't compile, because a_const_copy(0) is const int&

const auto k_a = a.to_kernel(); // const gtensor_span<int, 1>
k_a(0) = -1; // allowed, gtensor_span has shallow const behavior

auto k_a_const_copy = a_const_copy.to_kernel(); // gtensor_span<const int, 1>
k_a_const_copy(0) = 10; // won't compile, type of LHS is const int&

```

# Streams (experimental)

To facilitate interoperability with existing libraries and allow
experimentation with some advanced multi-stream use cases, there are classes
`gt::stream` and `gt::stream_view`. The `gt::stream` will create a new stream
in the default device backend and destroy the stream when the object is
destructed. The `gt::stream_view` is constructed with an existing native stream
object in the default backend (e.g. a `cudaStream\_t` for the CUDA backend).
They can be used as optional arguments to `gt::launch` and `gt::assign`, in
which case they will execute asynchronously with the default stream on device.
Note that the equals operator form of assign does not work with alternate
streams - it will always use the default stream. For the SYCL backend, the
native stream object is a `sycl::queue`.

See also `tests/test_stream.cxx`. Note that this API is likely to change; in
particular, the stream objects will become templated on space type.

# Library Wrapper Extensions

## gt-blas

Provides wrappers around commonly used blas routines. Requires cuBLAS, rocblas,
or oneMKL, depending on the GPU backend. Interface is mostly C style taking
raw pointers, for easy interoperability with Fortran, with a few higher level
gtensor specific helpers.

```
#include "gt-blas/blas.h"

void blas()
{
  gt::blas::handle_t h;
  gt::gtensor_device<double, 1> x = gt::arange<double>(1, 11);
  gt::gtensor_device<double, 1> y = gt::arange<double>(1, 11);
  gt::blas::axpy(h, 2.0, x, y);
  std::cout << "a*x+y = " << y << std::endl;
  /* a*x+y = { 3 6 9 12 15 18 21 24 27 30 } */
}
```

A naive banded LU solver implementation is also provided, useful in cases where
the matrices are banded and the native GPU batched LU solve has not been
optimized yet. Parallelism for this implementaiton is on batch only.

## gt-fft

Provides high level C++ style interface around cuFFT, rocFFT, and oneMKL DFT.

```
#include "gt-fft/fft.h"

void fft()
{
  // python: x = np.array([2., 3., -1., 4.])
  gt::gtensor_device<double, 1> x = {2, 3, -1, 4};
  auto y = gt::empty_device<gt::complex<double>>({3});

  // python: y = np.fft.fft(x)
  gt::fft::FFTPlanMany<gt::fft::Domain::REAL, double> plan({x.shape(0)}, 1);
  plan(x, y);
  std::cout << y << std::endl;
  /* { (8,0) (3,1) (-6,0) } */
}
```

## gt-solver

Provides a high level C++ style interface around batched LU solve, in
particular the case where a single set of matrices is used repeatedly to solve
with different right hand side vectors. It maintains it's own contiguous copy
of the factored matrices in device memory, and device buffers for staging input
and output right hand side vectors. This allows the application to build the
matrices on host and pass them off to the solver interface for all the device
specific handling. On some platforms, there are performance issues with solving
out of managed memory, and the internal device buffers can significantly
improve performance on these platforms.

This should be preferred over directly calling gt-blas routines like getrf and
getrs, when the use case matches (single factor and many solves with different
right hand sides).
