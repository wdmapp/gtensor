# gtensor

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
- GPU support for nVidia via CUDA and AMD via HIP/rocm. Future plans to
  support Intel GPUs

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
to build for cpu/host only, use `-DGTENSOR_DEVICE=host`, and for AMD/HIP use
`-DGTENSOR_DEVICE=hip -DCMAKE_CXX_COMPILER=$(which hipcc)`
(see also further HIP requirements below).

Note that gtensor can still be used by applications not using cmake -
see [Usage (GNU make)](#usage-gnu-make) for an example.

### nVidia CUDA requirements

gtensor for nVidia GPUs with CUDA requires
[CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 10.0+.

### AMD HIP requirements

gtensor for AMD GPUs with HIP requires ROCm 3.3.0+
with rocthrust and rocprim. See the
[ROCm installation guide](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html)
for details. In Ubuntu, after setting up the ROCm repository, the required
packages can be installed like this:
```
sudo apt install rocm-dkms rocm-dev rocthrust
```
The official packages install to `/opt/rocm`. If using a different install
location, add it to `CMAKE_PREFIX_PATH` when running cmake for the application.

### HOST CPU (no device) requirements

gtensor should build with any C++ compiler supporting C++14. It has been
tested with g++ 7, 8, and 9 and clang++ 8, 9, and 10.

### Advanced multi-device configuration

By default, gtensor will install support for the device specified by
the `GTENSOR_DEVICE` variable (default `cuda`), and also the `host` (cpu only)
device. This can be configured with `GTENSOR_BUILD_DEVICES` as a semicolon (;)
separated list. For example, to build support for all three backends
(i.e. assuming a machine with both nVidia and AMD GPUs):
```
cmake -S . -B build -DGTENSOR_DEVICE=cuda \
  -DGTENSOR_BUILD_DEVICES=host;cuda;hip \
  -DCMAKE_INSTALL_PREFIX=/opt/gtensor \
  -DBUILD_TESTING=OFF
```

This will cause targets to be created for each device: `gtensor::gtensor_cuda`,
`gtensor::gtensor_host`, and `gtensor::gtensor_hip`. The main
`gtensor::gtensor` target will be an alias for the default set by
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