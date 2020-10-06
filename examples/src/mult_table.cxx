#include <iostream>

#include <gtensor/gtensor.h>

template <typename T, typename S>
gt::gtensor<T, 2, S> outer_product(gt::gtensor<T, 1, S>& a,
                                   gt::gtensor<T, 1, S>& b)
{
  int n = a.shape(0);
  assert(n == b.shape(0));
  return gt::reshape(a, gt::shape(1, n)) * gt::reshape(b, gt::shape(n, 1));
}

template <typename T, typename S>
auto outer_product_expr(gt::gtensor<T, 1, S>& a, gt::gtensor<T, 1, S>& b)
{
  int n = a.shape(0);
  assert(n == b.shape(0));
  return gt::reshape(a, gt::shape(1, n)) * gt::reshape(b, gt::shape(n, 1));
}

int main(int argc, char** argv)
{
  const int n = 9;

  std::cout << "element access" << std::endl;
  gt::gtensor<int, 2> mult_table(gt::shape(n, n));

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      mult_table(i, j) = (i + 1) * (j + 1);
    }
  }

  for (int i = 0; i < n; i++) {
    std::cout << mult_table.view(i, gt::all) << std::endl;
  }
  std::cout << std::endl;

  std::cout << "broadcast" << std::endl;
  gt::gtensor<int, 2> a(gt::shape(1, n));
  gt::gtensor<int, 2> b(gt::shape(n, 1));

  for (int i = 0; i < n; i++) {
    a(0, i) = i + 1;
    b(i, 0) = i + 1;
  }

  gt::gtensor<int, 2> ab = a * b;
  for (int i = 0; i < n; i++) {
    std::cout << ab.view(i, gt::all) << std::endl;
  }
  std::cout << std::endl;

  std::cout << "broadcast with reshape" << std::endl;
  gt::gtensor<int, 1> v(gt::shape(n));

  for (int i = 0; i < n; i++) {
    v(i) = i + 1;
  }

  gt::gtensor<int, 2> vv =
    gt::reshape(v, gt::shape(1, n)) * gt::reshape(v, gt::shape(n, 1));
  for (int i = 0; i < n; i++) {
    std::cout << vv.view(i, gt::all) << std::endl;
  }
  std::cout << std::endl;

  std::cout << "broadcast with reshape (helper fn)" << std::endl;
  gt::gtensor<int, 2> vv2 = outer_product(v, v);
  for (int i = 0; i < n; i++) {
    std::cout << vv2.view(i, gt::all) << std::endl;
  }
  std::cout << std::endl;

  std::cout << "broadcast with reshape (helper fn expr)" << std::endl;
  auto vv_expr = outer_product_expr(v, v);
  auto vv3 = gt::eval(vv_expr);
  for (int i = 0; i < n; i++) {
    std::cout << vv3.view(i, gt::all) << std::endl;
  }
}
