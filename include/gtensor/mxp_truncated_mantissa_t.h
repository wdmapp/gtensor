#ifndef GTENSOR_MXP_TRUNCATED_MANTISSA_H
#define GTENSOR_MXP_TRUNCATED_MANTISSA_H

#include <cstdint>
#include <cstring>
#include <type_traits>

// __________________________________________________________________________ //

namespace gt
{

// __________________________________________________________________________ //

namespace mxp_detail
{

template <typename fp_t>
constexpr bool is_decay_float{std::is_same<std::decay_t<fp_t>, float>::value};

template <typename fp_t>
constexpr bool is_decay_double{std::is_same<std::decay_t<fp_t>, double>::value};

// -------------------------------------------------------------------------- //

template <
  typename fp_t,
  typename = std::enable_if_t<
    (is_decay_float<fp_t> && (sizeof(float) == sizeof(std::uint32_t))) ||
    (is_decay_double<fp_t> && (sizeof(double) == sizeof(std::uint64_t)))>>
using uint_t =
  std::conditional_t<is_decay_float<fp_t>, std::uint32_t, std::uint64_t>;

// -------------------------------------------------------------------------- //

template <typename fp_t>
constexpr uint_t<fp_t> sign_mask{is_decay_float<fp_t> ? 0x8000'0000
                                                      : 0x8000'0000'0000'0000};

template <typename fp_t>
constexpr uint_t<fp_t> exponent_mask{
  is_decay_float<fp_t> ? 0x7F80'0000 : 0x7FF0'0000'0000'0000};

template <typename fp_t>
constexpr uint_t<fp_t> mantissa_mask{
  is_decay_float<fp_t> ? 0x007F'FFFF : 0x000F'FFFF'FFFF'FFFF};

// -------------------------------------------------------------------------- //

// clang-format off
// (messes with triple consecutive angle brackets)

template <typename fp_t, std::uint16_t bits>
constexpr uint_t<fp_t> reduced_mantissa_mask{mantissa_mask<fp_t> ^
                                             (mantissa_mask<fp_t> >> bits)};

template <typename fp_t, std::uint16_t bits>
constexpr uint_t<fp_t> reduced_rounding_mask{mantissa_mask<fp_t> >> (bits + 1)};

// clang-format on

// -------------------------------------------------------------------------- //

template <typename fp_t>
struct decay_strip_complex
{
  typedef std::decay_t<fp_t> type;
};

template <typename fp_t>
struct decay_strip_complex<gt::complex<fp_t>>
{
  typedef std::decay_t<fp_t> type;
};

template <typename fp_t>
using decay_strip_complex_t = typename decay_strip_complex<fp_t>::type;

// -------------------------------------------------------------------------- //

template <typename fp_t, std::uint16_t bits>
struct mantissa_bits_available
  : public std::conditional_t<
      (is_decay_float<decay_strip_complex_t<fp_t>> && (bits <= 23)) ||
        (is_decay_double<decay_strip_complex_t<fp_t>> && (bits <= 52)),
      std::true_type, std::false_type>
{};

template <typename fp_t, std::uint16_t bits>
constexpr bool mantissa_bits_available_v =
  mantissa_bits_available<fp_t, bits>::value;

// -------------------------------------------------------------------------- //

} // namespace mxp_detail

// __________________________________________________________________________ //

template <typename fp_t, std::uint16_t bits>
class mxp_truncated_mantissa_t
{
public:
  using enclosing_fp_t = std::decay_t<fp_t>;
  using underlying_fp_t =
    std::enable_if_t<mxp_detail::mantissa_bits_available_v<fp_t, bits>,
                     mxp_detail::decay_strip_complex_t<fp_t>>;
  using uint_t = mxp_detail::uint_t<underlying_fp_t>;

  // ------------------------------------------------------------------------ //

  GT_INLINE mxp_truncated_mantissa_t(const fp_t& FP_src) : FP_src_(FP_src) {}

  // ------------------------------------------------------------------------ //
  // the only type cast guarantees computations
  // desired precision (enclosing_fp_t) after rounding mantissa on read-access

  template <typename T>
  GT_INLINE explicit operator T() const = delete;

  // returns value rounded to truncated mantissa
  GT_INLINE operator enclosing_fp_t() const
  {
    return get_truncated_mantissa_value(FP_src_);
  }

  // ------------------------------------------------------------------------ //

private:
  const fp_t& FP_src_;

  // ------------------------------------------------------------------------ //

  template <typename Real>
  static GT_INLINE enclosing_fp_t get_truncated_mantissa_value(const Real& arg)
  {
    return get_truncated_mantissa_value_impl(arg);
  }

  // ------------------------------------------------------------------------ //

  template <typename Real>
  static GT_INLINE enclosing_fp_t
  get_truncated_mantissa_value(const gt::complex<Real>& arg)
  {
    return {get_truncated_mantissa_value_impl(arg.real()),
            get_truncated_mantissa_value_impl(arg.imag())};
  }

  // ------------------------------------------------------------------------ //

  static GT_INLINE underlying_fp_t
  get_truncated_mantissa_value_impl(const underlying_fp_t& FP_src)
  {
    // clang-format off
    // (messes with bitwise arithmetic operator&)
    uint_t BIN_src;
    general_memcpy_single_val(&FP_src, &BIN_src);

    // binary BIN_oneexp: S E...E (1)0...0
    // [S, E like src; mantissa: implicit one, then all zeroes]
    uint_t BIN_oneexp{(mxp_detail::sign_mask<underlying_fp_t> |
                       mxp_detail::exponent_mask<underlying_fp_t>) & BIN_src};
    underlying_fp_t FP_oneexp;
    general_memcpy_single_val(&BIN_oneexp, &FP_oneexp);

    // binary BIN_rounding: S E...E (1)0...01...1
    // [S, E like src; mantissa: implicit one, bits+1 zeroes, then all ones]
    uint_t BIN_rounding{
      BIN_oneexp | mxp_detail::reduced_rounding_mask<underlying_fp_t, bits>};
    underlying_fp_t FP_rounding;
    general_memcpy_single_val(&BIN_rounding, &FP_rounding);

    // in FP arithmetic add rounding value
    underlying_fp_t FP_tmp = FP_src + (FP_rounding - FP_oneexp);
    uint_t BIN_tmp;
    general_memcpy_single_val(&FP_tmp, &BIN_tmp);

    // truncate mantissa to length 'bits'
    uint_t BIN_result{
      (mxp_detail::sign_mask<underlying_fp_t> |
       mxp_detail::exponent_mask<underlying_fp_t> |
       mxp_detail::reduced_mantissa_mask<underlying_fp_t, bits>) & BIN_tmp};

    underlying_fp_t FP_result;
    general_memcpy_single_val(&BIN_result, &FP_result);
    return FP_result;
    // clang-format on
  }

  // ------------------------------------------------------------------------ //

  static GT_INLINE void general_memcpy_single_val(const underlying_fp_t* src,
                                                  uint_t* dest)
  {
    static_assert(sizeof(underlying_fp_t) == sizeof(uint_t));
#ifdef GTENSOR_DEVICE_HIP
    __builtin_memcpy(dest, src, sizeof(underlying_fp_t));
#else
    std::memcpy(dest, src, sizeof(underlying_fp_t));
#endif
  }

  static GT_INLINE void general_memcpy_single_val(const uint_t* src,
                                                  underlying_fp_t* dest)
  {
    static_assert(sizeof(underlying_fp_t) == sizeof(uint_t));
#ifdef GTENSOR_DEVICE_HIP
    __builtin_memcpy(dest, src, sizeof(underlying_fp_t));
#else
    std::memcpy(dest, src, sizeof(underlying_fp_t));
#endif
  }

  // ------------------------------------------------------------------------ //
};

// -------------------------------------------------------------------------- //

} // namespace gt

// __________________________________________________________________________ //

#endif // GTENSOR_MXP_TRUNCATED_MANTISSA_H
