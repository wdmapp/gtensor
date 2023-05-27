template <>
class fftw<R>
{
public:
  using plan_type = ::FFTW_(plan);
  using complex_type = ::FFTW_(complex);

  fftw() = default;
  fftw(plan_type plan) : plan_{plan}, is_valid_{true} {}

  // move only
  fftw(const fftw& other) = delete;
  fftw(fftw&& other) : plan_{other.plan_}, is_valid_{other.is_valid_}
  {
    other.is_valid_ = false;
  }

  fftw& operator=(const fftw& other) = delete;
  fftw& operator=(fftw&& other)
  {
    if (&other != this) {
      using std::swap;
      swap(plan_, other.plan_);
      swap(is_valid_, other.is_valid_);
    }
    return *this;
  }

  ~fftw()
  {
    if (is_valid_) {
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

  void execute_dft(complex_type* in, complex_type* out) const
  {
    assert(is_valid_);
    return ::FFTW_(execute_dft)(plan_, in, out);
  }

private:
  plan_type plan_;
  bool is_valid_ = false;
};
