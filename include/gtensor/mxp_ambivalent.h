#ifndef MXP_AMBIVALENT_H
#define MXP_AMBIVALENT_H

// __________________________________________________________________________ //

namespace mxp
{

// -------------------------------------------------------------------------- //

namespace detail
{

// -------------------------------------------------------------------------- //

/*  An instance of the auxiliary type ambivalent_t
    doesn't know yet in which precision it will be used
    as this will depend on the operator context.
    Roughly, i.e., are you on LHS or on RHS of assignment?
    It is designed s.t. mxp::mxp_span (mxp::adapt)
    can be used for both, evaluation and assignment */

// -------------------------------------------------------------------------- //

template <typename compute_type, typename storage_type>
class ambivalent_t
{
public:
  // ------------------------------------------------------------------------ //

  // construct from reference
  inline explicit ambivalent_t(storage_type& value_ref) : value_ref_(value_ref)
  {}

  // ------------------------------------------------------------------------ //

  // (compound) assignment ops [=, +=, -=, *=, /=]
#define DEFINE_ASSIGNMENT_OPERATOR(op)                                         \
  template <typename T>                                                        \
  inline ambivalent_t& operator op(const T& rhs)                               \
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
  inline explicit operator T() const = delete;

  inline operator compute_type() const
  {
    return static_cast<compute_type>(value_ref_);
  }

  // ------------------------------------------------------------------------ //

private:
  storage_type& value_ref_;
};

// -------------------------------------------------------------------------- //

template <typename compute_type, typename storage_type>
std::ostream& operator<<(std::ostream& s,
                         const ambivalent_t<compute_type, storage_type>& a)
{
  return s << static_cast<compute_type>(a);
}

} // namespace detail

// -------------------------------------------------------------------------- //

} // namespace mxp

// __________________________________________________________________________ //

#endif // MXP_AMBIVALENT_H
