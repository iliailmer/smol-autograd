#include "parameter.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

float target_(float a, float b, float c, float d, float x)
{
  return a * x * x * x * x + b * x * x * x + c * x * x + d * x;
}
float grad_target_(float a, float b, float c, float d, float x)
{
  return 4 * a * x * x * x + 3 * b * x * x + 2 * c * x + d;
}

float adam(float a, float b, float c, float d, float x, float lr, float beta1,
           float beta2, int num_iter, float tol, float eps)
{
  float m = 0;
  float m_hat = 0;
  float v = 0;
  float v_hat = 0;
  float beta1_power = beta1; // beta1^t
  float beta2_power = beta2; // beta2^t
  float grad = 0;

  for (int iter = 0; iter < num_iter; iter++) {
    grad = grad_target_(a, b, c, d, x);
    m = beta1 * m + (1 - beta1) * grad;
    v = beta2 * v + (1 - beta2) * grad * grad;

    // Use cached powers instead of expensive pow() calls
    m_hat = m / (1 - beta1_power);
    v_hat = v / (1 - beta2_power);

    x = x - lr * m_hat / (sqrt(v_hat) + eps);

    // Check for convergence
    float curr_val = target_(a, b, c, d, x);
    printf("%e, f(%e)=%e\n", x, x, curr_val);

    if (fabs(grad) < tol) {
      printf("Converged after %d iterations\n", iter + 1);
      break;
    }

    // Update powers for next iteration
    beta1_power *= beta1;
    beta2_power *= beta2;
  }
  return x;
}
