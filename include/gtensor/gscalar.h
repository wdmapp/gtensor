
#ifndef GTENSOR_GSCALAR_H
#define GTENSOR_GSCALAR_H

#include "expression.h"
#include "sarray.h"

namespace gt
{

template <typename T>
class gscalar : public expression<gscalar<T>>
{
public:
  using value_type = std::decay_t<T>;
  using shape_type = gt::shape_type<0>;
  using space_type = space::any;

  constexpr static size_type dimension() { return 0; }

  gscalar(T value) : value_(value) {}

  shape_type shape() const { return {}; }

  template <typename... Args>
  GT_INLINE value_type operator()(Args... args) const
  {
    return value_;
  }

  gscalar<value_type> to_kernel() const { return gscalar<value_type>(value_); }

private:
  value_type value_;
};

template <typename T>
gscalar<T> scalar(T&& t)
{
  return gscalar<T>(std::forward<T>(t));
}

} // namespace gt

#endif
