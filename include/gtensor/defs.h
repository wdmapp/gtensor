
#ifndef GTENSOR_DEFS_H
#define GTENSOR_DEFS_H

#include <cstddef>

namespace gt
{

using size_type = std::size_t;

// forward declarations

template <typename T, size_type N>
class sarray;

// some commonly used types

template <size_type N>
using shape_type = sarray<int, N>;

template <typename T>
T div_ceil(const T n, const T d)
{
  return (n - 1) / d + 1;
}

} // namespace gt

#endif
