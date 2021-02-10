
#ifndef GTENSOR_GSTRIDED_H
#define GTENSOR_GSTRIDED_H

#include "defs.h"
#include "expression.h"
#include "gslice.h"
#include "strides.h"

namespace gt
{

template <typename E, typename... Args>
constexpr int view_dimension()
{
  constexpr std::size_t N_new = detail::count_convertible<gnewaxis, Args...>();
  constexpr std::size_t N_value = detail::count_convertible<int, Args...>();
  constexpr std::size_t N_expr = expr_dimension<E>();
  constexpr std::size_t N_args = sizeof...(Args);
  static_assert(N_args <= N_expr + N_new, "too many view args for expression");
  return N_expr - N_value + N_new;
}

template <typename E, typename... Args>
auto view(E&& e, Args&&... args);

template <typename D>
struct gtensor_inner_types;

// ======================================================================
// gstrided

template <typename D>
class gstrided : public expression<D>
{
public:
  using derived_type = D;
  using base_type = expression<D>;
  using inner_types = gtensor_inner_types<D>;
  using space_type = typename inner_types::space_type;

  using value_type = typename inner_types::value_type;
  using reference = typename inner_types::reference;
  using const_reference = typename inner_types::const_reference;

  constexpr static size_type dimension() { return inner_types::dimension; }

  using shape_type = gt::shape_type<dimension()>;
  using strides_type = gt::shape_type<dimension()>;

  using base_type::derived;

  gstrided() = default;
  gstrided(const shape_type& shape, const strides_type& strides);

  GT_INLINE int shape(int i) const;
  GT_INLINE const shape_type& shape() const;
  GT_INLINE const strides_type& strides() const;
  GT_INLINE size_type size() const;

  template <typename... Args>
  inline auto view(Args&&... args) &;
  template <typename... Args>
  inline auto view(Args&&... args) const&;
  template <typename... Args>
  inline auto view(Args&&... args) &&;

protected:
  template <typename... Args>
  GT_INLINE size_type index(Args&&... args) const;

  shape_type shape_;
  strides_type strides_;
};

// ----------------------------------------------------------------------
// gstrided implementation

template <typename D>
inline gstrided<D>::gstrided(const shape_type& shape,
                             const strides_type& strides)
  : shape_(shape), strides_(strides)
{}

template <typename D>
GT_INLINE int gstrided<D>::shape(int i) const
{
  return shape_[i];
}

template <typename D>
GT_INLINE auto gstrided<D>::shape() const -> const shape_type&
{
  return shape_;
}

template <typename D>
GT_INLINE auto gstrided<D>::strides() const -> const strides_type&
{
  return strides_;
}

template <typename D>
GT_INLINE size_type gstrided<D>::size() const
{
  return calc_size(shape());
}

template <typename D>
template <typename... Args>
inline auto gstrided<D>::view(Args&&... args) const&
{
  return gt::view(derived(), std::forward<Args>(args)...);
}

template <typename D>
template <typename... Args>
inline auto gstrided<D>::view(Args&&... args) &
{
  return gt::view(derived(), std::forward<Args>(args)...);
}

template <typename D>
template <typename... Args>
inline auto gstrided<D>::view(Args&&... args) &&
{
  return gt::view(std::move(*this).derived(), std::forward<Args>(args)...);
}

template <typename D>
template <typename... Args>
GT_INLINE size_type gstrided<D>::index(Args&&... args) const
{
#ifdef GT_BOUNDSCHECK
  bounds_check(this->shape(), std::forward<Args>(args)...);
#endif
  return calc_index(this->strides_, std::forward<Args>(args)...);
}

} // namespace gt

#endif
