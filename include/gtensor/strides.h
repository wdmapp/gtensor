
#ifndef GTENSOR_STRIDES_H
#define GTENSOR_STRIDES_H

namespace gt
{

// ======================================================================
// calc_strides
//
// calculates strides corresponding to col-major layout of given shape

template <typename S>
GT_INLINE S calc_strides(const S& shape)
{
  S strides;
  int stride = 1;
  for (int i = 0; i < shape.size(); i++) {
    if (shape[i] == 1) {
      strides[i] = 0;
    } else {
      strides[i] = stride;
    }
    stride *= shape[i];
  }
  return strides;
}

// ======================================================================
// unravel
//
// given 1-d index and strides, calculate multi-d index
//
template <typename S>
GT_INLINE S unravel(size_type i, const S& strides)
{
  S idx;
  for (int d = strides.size() - 1; d >= 0; d--) {
    idx[d] = strides[d] == 0 ? 0 : (i / strides[d]);
    i -= idx[d] * strides[d];
  }
  return idx;
}

// ======================================================================
// calc_index
//
// given strides and multi-d index, calculates 1-d index into storage

namespace detail
{

template <size_type dim, typename S>
GT_INLINE size_type calc_index(const S&)
{
  return 0;
}

template <size_type dim, typename S, typename Arg, typename... Args>
GT_INLINE size_type calc_index(const S& strides, Arg arg, Args... args)
{
  return strides[dim] * arg + calc_index<dim + 1>(strides, args...);
}

} // namespace detail

template <typename S, typename... Args>
GT_INLINE size_type calc_index(const S& strides, Args... args)
{
  static_assert(sizeof...(Args) == S::size(),
                "calc_index: need matching number of args");
  return detail::calc_index<0>(strides, args...);
}

// ======================================================================
// calc_size

template <typename S>
GT_INLINE constexpr size_type calc_size(const S& shape, size_type i = 0)
{
  return (i < S::size()) ? shape[i] * calc_size(shape, i + 1) : size_type(1);
}

// ======================================================================
// bounds_check
//
// checks that given multi-d index is in-bounds w.r.t giveen shape

namespace detail
{

template <size_type dim, typename S>
GT_INLINE void bounds_check(const S&)
{}

template <size_type dim, typename S, class Arg, class... Args>
GT_INLINE void bounds_check(const S& shape, Arg arg, Args... args)
{
  if (shape[dim] != 1 && (arg < 0 || arg >= shape[dim])) {
    printf("out-of-bounds error: dim = %d, arg = %d, shape = %d\n", int(dim),
           int(arg), int(shape[dim]));
    assert(0);
  }
  bounds_check<dim + 1>(shape, args...);
}

} // namespace detail

template <typename S, class... Args>
GT_INLINE void bounds_check(const S& shape, Args... args)
{
  static_assert(sizeof...(Args) == S::size(),
                "bounds_check: dims do not match");
  detail::bounds_check<0>(shape, args...);
}

} // namespace gt

#endif
