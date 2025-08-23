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

adam_error_t adam_optimizer(Parameter **params, int n_params, float lr, 
                           float beta1, float beta2, int num_iter, 
                           float tol, float eps)
{
  if (!params || n_params <= 0) {
    return ADAM_ERROR_NULL_POINTER;
  }
  
  for (int i = 0; i < n_params; i++) {
    if (!params[i]) {
      return ADAM_ERROR_NULL_POINTER;
    }
  }

  float *m = calloc(n_params, sizeof(float));
  float *v = calloc(n_params, sizeof(float));
  if (!m || !v) {
    free(m);
    free(v);
    return ADAM_ERROR_MEMORY_ALLOCATION;
  }

  float beta1_power = beta1;
  float beta2_power = beta2;

  for (int iter = 0; iter < num_iter; iter++) {
    float max_grad = 0.0f;
    
    for (int i = 0; i < n_params; i++) {
      float grad = params[i]->grad;
      
      m[i] = beta1 * m[i] + (1.0f - beta1) * grad;
      v[i] = beta2 * v[i] + (1.0f - beta2) * grad * grad;
      
      float m_hat = m[i] / (1.0f - beta1_power);
      float v_hat = v[i] / (1.0f - beta2_power);
      
      params[i]->value -= lr * m_hat / (sqrtf(v_hat) + eps);
      
      if (fabsf(grad) > max_grad) {
        max_grad = fabsf(grad);
      }
    }
    
    if (max_grad < tol) {
      free(m);
      free(v);
      return ADAM_SUCCESS;
    }
    
    beta1_power *= beta1;
    beta2_power *= beta2;
  }
  
  free(m);
  free(v);
  return ADAM_SUCCESS;
}

void print_parameters(const char *title, Parameter **params, int n_params)
{
  if (!title || !params) {
    return;
  }
  
  printf("\n=== %s ===\n", title);
  for (int i = 0; i < n_params; i++) {
    if (params[i]) {
      printf("param[%d]: value=%.6f, grad=%.6f\n", 
             i, params[i]->value, params[i]->grad);
    }
  }
  printf("\n");
}

void print_mlp_parameters(Parameter w[2][2], Parameter b[2])
{
  printf("\n=== MLP Parameters ===\n");
  printf("Weights:\n");
  for (int i = 0; i < 2; i++) {
    printf("  ");
    for (int j = 0; j < 2; j++) {
      printf("w[%d][%d]=%.6f ", i, j, w[i][j].value);
    }
    printf("\n");
  }
  printf("Biases:\n");
  printf("  b[0]=%.6f b[1]=%.6f\n", b[0].value, b[1].value);
  printf("\n");
}

void zero_grad(Parameter **params, int n_params)
{
  if (!params) {
    return;
  }
  
  for (int i = 0; i < n_params; i++) {
    if (params[i]) {
      params[i]->grad = 0.0f;
    }
  }
}
