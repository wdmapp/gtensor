#ifndef GTENSOR_MXP_TRUNCATED_MANTISSA_H
#define GTENSOR_MXP_TRUNCATED_MANTISSA_H

// __________________________________________________________________________ //

namespace mxp
{

// __________________________________________________________________________ //

namespace detail
{

template <typename fp_t>
constexpr bool is_decay_float{std::is_same<std::decay_t<fp_t>, float>::value};

template <typename fp_t>
constexpr bool is_decay_double{std::is_same<std::decay_t<fp_t>, double>::value};

// -------------------------------------------------------------------------- //

template <typename fp_t, typename = std::enable_if_t<is_decay_float<fp_t> ||
                                                     is_decay_double<fp_t>>>
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

template <typename fp_t, std::uint8_t bits>
constexpr uint_t<fp_t> reduced_mantissa_mask{mantissa_mask<fp_t> ^
                                             (mantissa_mask<fp_t> >> bits)};

template <typename fp_t, std::uint8_t bits>
constexpr uint_t<fp_t> reduced_rounding_mask{mantissa_mask<fp_t> >> (bits + 1)};

// -------------------------------------------------------------------------- //

} // namespace detail

// __________________________________________________________________________ //

template <typename fp_t, std::uint8_t bits>
class truncated_mantissa_t
{
public:
  using underlying_fp_t = std::decay_t<fp_t>;
  using uint_t = detail::uint_t<underlying_fp_t>;

  // ------------------------------------------------------------------------ //

  GT_INLINE truncated_mantissa_t(const fp_t& FP_src) : FP_src_(FP_src) {}

  // ------------------------------------------------------------------------ //
  // the only type cast guarantees computations
  // desired precision (underlying_fp_t) after rounding mantissa on read-access

  template <typename T>
  GT_INLINE explicit operator T() const = delete;

  // returns value rounded to truncated mantissa
  GT_INLINE operator underlying_fp_t() const
  {
    return get_truncated_mantissa_value(FP_src_);
  }

  // ------------------------------------------------------------------------ //

private:
  const fp_t& FP_src_;

  // ------------------------------------------------------------------------ //

  static GT_INLINE underlying_fp_t
  get_truncated_mantissa_value(const underlying_fp_t& FP_src)
  {
    uint_t BIN_src;
    std::memcpy(&BIN_src, &FP_src, sizeof(underlying_fp_t));

    // binary BIN_oneexp: S E...E (1)0...0
    // [S, E like src; mantissa: implicit one, then all zeroes]
    uint_t BIN_oneexp{(detail::sign_mask<underlying_fp_t> |
                       detail::exponent_mask<underlying_fp_t>)&BIN_src};
    underlying_fp_t FP_oneexp;
    std::memcpy(&FP_oneexp, &BIN_oneexp, sizeof(underlying_fp_t));

    // binary BIN_rounding: S E...E (1)0...01...1
    // [S, E like src; mantissa: implicit one, bits+1 zeroes, then all ones]
    uint_t BIN_rounding{BIN_oneexp |
                        detail::reduced_rounding_mask<underlying_fp_t, bits>};
    underlying_fp_t FP_rounding;
    std::memcpy(&FP_rounding, &BIN_rounding, sizeof(underlying_fp_t));

    // in FP arithmetic add rounding value
    underlying_fp_t FP_tmp = FP_src + (FP_rounding - FP_oneexp);
    uint_t BIN_tmp;
    std::memcpy(&BIN_tmp, &FP_tmp, sizeof(underlying_fp_t));

    // truncate mantissa to length 'bits'
    uint_t BIN_result{
      (detail::sign_mask<underlying_fp_t> |
       detail::exponent_mask<underlying_fp_t> |
       detail::reduced_mantissa_mask<underlying_fp_t, bits>)&BIN_tmp};

    underlying_fp_t FP_result;
    std::memcpy(&FP_result, &BIN_result, sizeof(underlying_fp_t));
    return FP_result;
  }

  // ------------------------------------------------------------------------ //
};

// -------------------------------------------------------------------------- //

} // namespace mxp

// __________________________________________________________________________ //

#endif // GTENSOR_MXP_TRUNCATED_MANTISSA_H
