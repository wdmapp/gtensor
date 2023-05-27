template <>
class fftw<R>
{
public:
  using plan_type = ::FFTW_(plan);
  using complex_type = ::FFTW_(complex);
  using real_type = R;

  fftw() = default;
  fftw(plan_type plan) : plan_{plan} {}

  // move only
  fftw(const fftw& other) = delete;
  fftw(fftw&& other) : plan_{other.plan_} { other.plan_ = {}; }

  fftw& operator=(const fftw& other) = delete;
  fftw& operator=(fftw&& other)
  {
    if (&other != this) {
      using std::swap;
      swap(plan_, other.plan_);
    }
    return *this;
  }

  ~fftw()
  {
    if (plan_) {
      FFTW_(destroy_plan)(plan_);
    }
  }

  static fftw plan_many_dft(int rank, const int* n, int howmany,
                            complex_type* in, const int* inembed, int istride,
                            int idist, complex_type* out, const int* onembed,
                            int ostride, int odist, int sign, unsigned flags)
  {
    return fftw{::FFTW_(plan_many_dft)(rank, n, howmany, in, inembed, istride,
                                       idist, out, onembed, ostride, odist,
                                       sign, flags)};
  }

  static fftw plan_many_dft_r2c(int rank, const int* n, int howmany,
                                real_type* in, const int* inembed, int istride,
                                int idist, complex_type* out,
                                const int* onembed, int ostride, int odist,
                                unsigned flags)
  {
    return fftw{::FFTW_(plan_many_dft_r2c)(rank, n, howmany, in, inembed,
                                           istride, idist, out, onembed,
                                           ostride, odist, flags)};
  }

  static fftw plan_many_dft_c2r(int rank, const int* n, int howmany,
                                complex_type* in, const int* inembed,
                                int istride, int idist, real_type* out,
                                const int* onembed, int ostride, int odist,
                                unsigned flags)
  {
    return fftw{::FFTW_(plan_many_dft_c2r)(rank, n, howmany, in, inembed,
                                           istride, idist, out, onembed,
                                           ostride, odist, flags)};
  }

  void execute_dft(complex_type* in, complex_type* out) const
  {
    if (!plan_) {
      throw std::runtime_error("can't use a moved-from plan");
    }
    return ::FFTW_(execute_dft)(plan_, in, out);
  }

  void execute_dft_r2c(real_type* in, complex_type* out) const
  {
    if (!plan_) {
      throw std::runtime_error("can't use a moved-from plan");
    }
    return ::FFTW_(execute_dft_r2c)(plan_, in, out);
  }

  void execute_dft_c2r(complex_type* in, real_type* out) const
  {
    if (!plan_) {
      throw std::runtime_error("can't use a moved-from plan");
    }
    return ::FFTW_(execute_dft_c2r)(plan_, in, out);
  }

private:
  plan_type plan_ = {};
};
