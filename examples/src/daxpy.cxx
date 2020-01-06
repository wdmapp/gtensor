/*
 * =====================================================================================
 *
 *       Filename:  daxpy.cxx
 *
 *    Description: Example implementing BLAS vector operation DAXPY with Gtensor 
 *
 *        Version:  1.0
 *        Created:  01/06/2020 10:28:50 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include <iostream>

#include <gtensor/gtensor.h>

using namespace std;

int main(int argc, char **argv)
{
    double a = 1.5;
    gt::gtensor<double, 1> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    gt::gtensor<double, 1> y = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto axpy = a * x + y;
    cout << "a       = " << a << endl;
    cout << "x       = " << x << endl;
    cout << "y       = " << y << endl;
    cout << "a*x + y = " << axpy << endl;
}
