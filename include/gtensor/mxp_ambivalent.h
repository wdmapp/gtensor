#ifndef MXP_AMBIVALENT_H
#define MXP_AMBIVALENT_H

#include "mxp_truncated_mantissa_t.h"

// __________________________________________________________________________ //

namespace mxp
{

// -------------------------------------------------------------------------- //

namespace detail
{

// -------------------------------------------------------------------------- //

/*  ambivalent_t<CT, ST> below implements typecast compute_type() as:
    storage_type --> intermediate_compute_type --> compute_type
    for CT builtin, intermediate_compute_type = compute_type = CT */

// -------------------------------------------------------------------------- //

template <typename CT, typename ST>
struct accessor
{
  typedef CT type;
};

template <typename fp_t, std::uint8_t bits, typename ST>
struct accessor<truncated_mantissa_t<fp_t, bits>, ST>
{
  typedef std::enable_if_t<
    std::is_same<std::decay_t<fp_t>, std::decay_t<ST>>::value,
    std::decay_t<fp_t>>
    type;
};

template <typename CT, typename ST>
using accessor_t = typename accessor<CT, ST>::type;

// -------------------------------------------------------------------------- //

/*  An instance of the auxiliary type ambivalent_t
    doesn't know yet in which precision it will be used
    as this will depend on the operator context.
    Roughly, i.e., are you on LHS or on RHS of assignment?
    It is designed s.t. mxp::mxp_span (mxp::adapt)
    can be used for both, evaluation and assignment */

// -------------------------------------------------------------------------- //

template <typename CT, typename ST>
class ambivalent_t
{
public:
  using storage_type = ST;
  using intermediate_compute_type = CT;
  using compute_type = accessor_t<CT, ST>;

  // ------------------------------------------------------------------------ //

  // construct from reference
  GT_INLINE explicit ambivalent_t(storage_type& value_ref)
    : value_ref_(value_ref)
  {}

  // ------------------------------------------------------------------------ //

  // (compound) assignment ops [=, +=, -=, *=, /=]
#define DEFINE_ASSIGNMENT_OPERATOR(op)                                         \
  template <typename T>                                                        \
  GT_INLINE ambivalent_t& operator op(const T& rhs)                            \
  {                                                                            \
    value_ref_ op rhs;                                                         \
    return *this;                                                              \
  }

  DEFINE_ASSIGNMENT_OPERATOR(=)
  DEFINE_ASSIGNMENT_OPERATOR(+=)
  DEFINE_ASSIGNMENT_OPERATOR(-=)
  DEFINE_ASSIGNMENT_OPERATOR(*=)
  DEFINE_ASSIGNMENT_OPERATOR(/=)
#undef DEFINE_ASSIGNMENT_OPERATOR

  // ------------------------------------------------------------------------ //

  // the only type cast guarantees computations in desired precision
  template <typename T>
  GT_INLINE explicit operator T() const = delete;

  GT_INLINE operator compute_type() const
  {
    return static_cast<intermediate_compute_type>(value_ref_);
  }

  // ------------------------------------------------------------------------ //

private:
  storage_type& value_ref_;
};

// -------------------------------------------------------------------------- //

template <typename CT, typename ST>
std::ostream& operator<<(std::ostream& s, const ambivalent_t<CT, ST>& a)
{
  using compute_type = typename ambivalent_t<CT, ST>::compute_type;
  return s << static_cast<compute_type>(a);
}

// -------------------------------------------------------------------------- //

/* Binary operators: ambivalent_t + ambivalent_t
                     ambivalent_t + scalar_t
                     scalar_t     + ambivalent_t */

#define MAKE_AMBIVALENT_BINARY_OPERATOR(op)                                    \
  template <typename CT1, typename ST1, typename CT2, typename ST2>            \
  GT_INLINE auto operator op(const ambivalent_t<CT1, ST1>& lhs,                \
                             const ambivalent_t<CT2, ST2>& rhs)                \
  {                                                                            \
    using A1_compute_t = typename ambivalent_t<CT1, ST1>::compute_type;        \
    using A2_compute_t = typename ambivalent_t<CT2, ST2>::compute_type;        \
    return A1_compute_t(lhs) op A2_compute_t(rhs);                             \
  }                                                                            \
                                                                               \
  template <typename T, typename CT, typename ST>                              \
  GT_INLINE auto operator op(const T& lhs, const ambivalent_t<CT, ST>& rhs)    \
  {                                                                            \
    using A_compute_t = typename ambivalent_t<CT, ST>::compute_type;           \
    return lhs op A_compute_t(rhs);                                            \
  }                                                                            \
                                                                               \
  template <typename CT, typename ST, typename T>                              \
  GT_INLINE auto operator op(const ambivalent_t<CT, ST>& lhs, const T& rhs)    \
  {                                                                            \
    using A_compute_t = typename ambivalent_t<CT, ST>::compute_type;           \
    return A_compute_t(lhs) op rhs;                                            \
  }

MAKE_AMBIVALENT_BINARY_OPERATOR(+)
MAKE_AMBIVALENT_BINARY_OPERATOR(-)
MAKE_AMBIVALENT_BINARY_OPERATOR(*)
MAKE_AMBIVALENT_BINARY_OPERATOR(/)

#undef MAKE_AMBIVALENT_BINARY_OPERATOR

// -------------------------------------------------------------------------- //

} // namespace detail

// -------------------------------------------------------------------------- //

} // namespace mxp

// __________________________________________________________________________ //

#endif // MXP_AMBIVALENT_H
