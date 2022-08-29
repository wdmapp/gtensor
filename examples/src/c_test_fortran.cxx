
#include <gtensor/fortran.h>

#include <iostream>

extern "C" void c_test_arr2d(flcl_ndarray<const float>* nd_arr)
{
  auto arr = gt::adapt<2>(nd_arr);

  std::cout << "arr " << arr.shape() << "\n";
  std::cout << arr << "\n";
}
