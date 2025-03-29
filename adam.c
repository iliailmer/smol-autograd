#include "parameter.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

Parameter quad(float a, float b, float c, Parameter x) {
  // TODO: this function needs to be debugged, esp. gradient computation
  Parameter sq = mult(&x, &x);
  Parameter monom_1 = mul_num(&sq, a);
  Parameter monom_2 = mul_num(&x, b);
  Parameter res = add(&monom_1, &monom_2);
  return add_num(&res, c);
}

float target_(float a, float b, float c, float d, float x) {
  return a * x * x * x * x + b * x * x * x + c * x * x + d * x;
}
float grad_target_(float a, float b, float c, float d, float x) {
  return 4 * a * x * x * x + 3 * b * x * x + 2 * c * x + d;
}

float adam(float a, float b, float c, float d, float x, float lr, float beta1, float beta2, int num_iter, float tol, float eps) {
  float m = 0;
  float m_hat = 0;
  float v = 0;
  float v_hat = 0;
  int t = 0;
  float grad = 0;
  for (int iter = 0; iter < num_iter; iter++) {
    t += 1;
    grad = grad_target_(a, b, c, d, x);
    m = beta1 * m + (1 - beta1) * grad;
    v = beta2 * v + (1 - beta2) * grad * grad;
    m_hat = m / (1 - pow(beta1, t));
    v_hat = v / (1 - pow(beta2, t));
    x = x - lr * m_hat / (sqrt(v_hat) + eps);
    printf("%e, f(%e)=%e\n", x, x, target_(a, b, c, d, x));
  }
  return x;
}

int main(int argc, char *argv[]) {
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
  x = adam(a, b, c, d, x, lr, beta1, beta2, num_iter, tol, eps);
  printf("\n\n\n\t\t\tResult: %e, f(%e)=%e", x, x, target_(a, b, c, d, x));
  // Parameter x, y;
  // init_parameter(&x, 1.);
  // init_parameter(&y, 1.);
  // Parameter s = add(&x, &y);
  // Parameter quad_p = quad(a, b, c, x);

  // backward(&quad_p);
  // display(&quad_p);
  // display(&x);
  return EXIT_SUCCESS;
}
