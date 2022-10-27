
#include <gtensor/fortran.h>

#include <iostream>

extern "C" void c_test_arr2d(gt::farray<const float, 2>* nd_arr)
{
  auto arr = gt::adapt<2>(nd_arr);

  std::cout << "arr " << arr.shape() << "\n";
  std::cout << arr << "\n";
}
