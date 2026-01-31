#include "parameter.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char *argv[])
{
  float a, b, c, d, x;
  a = 1;
  b = -2;
  c = -1;
  d = 4;
  x = 0;
  float lr = 0.1;
  float beta1 = 0.9;
  float beta2 = 0.999;
  int num_iter = 100;
  float tol = 0.00001;
  float eps = 1e-8;
  
  printf("Starting Adam optimization...\n");
  printf("Target function: f(x) = %.2fx^4 + %.2fx^3 + %.2fx^2 + %.2fx\n", a, b, c, d);
  printf("Initial x: %.6f\n", x);
  printf("Learning rate: %.6f\n", lr);
  printf("Beta1: %.6f, Beta2: %.6f\n", beta1, beta2);
  printf("Max iterations: %d, Tolerance: %.2e\n\n", num_iter, tol);
  
  x = adam(a, b, c, d, x, lr, beta1, beta2, num_iter, tol, eps);
  
  printf("\n\n\t\t\tResult: %e, f(%e)=%e\n", x, x, target_(a, b, c, d, x));
  
  return EXIT_SUCCESS;
}