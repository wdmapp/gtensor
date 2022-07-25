#include <gtensor/gtensor.h>

namespace gt
{

namespace bm
{
// little hack to make tests parameterizable on managed vs device memory

template <typename T, gt::size_type N, typename S = gt::space::device>
struct gthelper
{
  using gtensor = gt::gtensor<T, N, S>;
};

#ifdef GTENSOR_HAVE_DEVICE

template <typename T, gt::size_type N>
struct gthelper<T, N, gt::space::managed>
{
  using gtensor = gt::gtensor_container<gt::space::managed_vector<T>, N>;
};

#endif

template <typename T, gt::size_type N, typename S = gt::space::device>
using gtensor2 = typename gthelper<T, N, S>::gtensor;

} // namespace bm

} // namespace gt
