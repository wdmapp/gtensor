#include <iostream>
#include <vector>
#include <gtensor/gtensor.h>
#include <gt-fft/fft.h>
#include <gtensor/blas.h>
#include <gtensor/reductions.h>

using namespace gt::placeholders;

void derivative()
{
double dx = 0.1;
// python: x = np.arange(0, 1, dx)
// python: y = x**2
gt::gtensor_device<double, 1> x = gt::arange<double>(0, 1, dx);
gt::gtensor_device<double, 1> y = x * x;

// python: dydx = -0.5/dx * y[:-2] + 0.5/dx * y[2:]
auto dydx = -0.5/dx * y.view(_s(_, -2)) + 0.5/dx * y.view(_s(2,  _));

std::cout << "x    : " << x.view(_s(1,-1)) << std::endl;
std::cout << "dy/dx: " << gt::eval(dydx) << std::endl;
/* x    : { 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 }
   dy/dx: { 0.2 0.4 0.6 0.8 1 1.2 1.4 1.6 } */
}

void mult_table()
{
// python: a = np.arange(1, 11)
gt::gtensor<int, 1> a = gt::arange<int>(1, 11);
// python: mtable = np.reshape(a, (1, 10)) * np.reshape(a, (10, 1))
gt::gtensor<int, 2> mtable = a.view(_newaxis, _all)
                           * a.view(_all, _newaxis);
// python: print("multiples of 4", mtable[3,:])
std::cout << "multiples of 4 " << mtable.view(3, _all) << std::endl;
/* multiples of 4 { 4 8 12 16 20 24 28 32 36 40 } */
}

void reduction()
{
gt::gtensor_device<int, 1> a = gt::arange<int>(1, 11);
std::cout << "sum " << gt::sum(a) << std::endl;
/* sum 55 */
}

void launch()
{
auto m = gt::full_device<int>({2, 3}, 7);
auto k_m = m.to_kernel();
gt::launch<1>(gt::shape(3), GT_LAMBDA(int i) {
    k_m(0, i) = (1  + i) * k_m(0, i);
    k_m(1, i) = (11 + i) * k_m(1, i);
});
std::cout << m << std::endl;
/* {{ 7 77 }
   { 14 84 }
   { 21 91 }} */
}

void blas()
{
gt::blas::handle_t* h = gt::blas::create();
gt::gtensor_device<double, 1> x = gt::arange<double>(1, 11);
gt::gtensor_device<double, 1> y = gt::arange<double>(1, 11);
gt::blas::axpy(h, 2.0, x, y);
std::cout << "a*x+y = " << y << std::endl;
/* a*x+y = { 3 6 9 12 15 18 21 24 27 30 } */
}

void fft()
{
// python: x = np.array([2., 3., -1., 4.])
gt::gtensor_device<double, 1> x = { 2, 3, -1, 4 };
auto y = gt::empty_device<gt::complex<double>>({3});

// python: y = np.fft.fft(x)
gt::fft::FFTPlanMany<gt::fft::Domain::REAL, double> plan({x.shape(0)}, 1);
plan(x, y);
std::cout << y << std::endl;
/* { (8,0) (3,1) (-6,0) } */
}

int main(int argc, char** argv)
{
  derivative();
  mult_table();
  reduction();
  launch();
  blas();
  fft();
}
