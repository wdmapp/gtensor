# gtensor

GTensor is a multi-dimensional array C++14 header-only library for hybrid GPU
development. It was inspired by
(xtensor)[https://xtensor.readthedocs.io/en/latest/], and designed to support
the GPU port of the (GENE)[http://genecode.org] fusion code.

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
- currently the GPU support is based on CUDA; AMD and Intel GPU platforms
  will be supported in the future.

## Getting Started

## Installation

GTensor is a header only library - to use, simply add the gtensor base
directory to your projects include path. Building with CUDA support requires
some more complex setup - see the examples
(CMakeLists.txt)[examples/CMakeLists.txt] as a starting point for how to do
this using cmake.

### Basic Example (host only)

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
