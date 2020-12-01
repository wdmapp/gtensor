
#define Z_MAX_ERR 1e-14

template <typename T>
inline void expect_complex_eq(T x, T y)
{
  EXPECT_NEAR(x.real(), y.real(), Z_MAX_ERR);
  EXPECT_NEAR(x.imag(), y.imag(), Z_MAX_ERR);
}

template <typename T1>
inline void expect_complex_eq(T1 x, double y)
{
  EXPECT_NEAR(x.real(), y, Z_MAX_ERR);
  EXPECT_NEAR(x.imag(), 0.0, Z_MAX_ERR);
}
