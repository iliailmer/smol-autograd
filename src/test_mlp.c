#include "parameter.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
  adam_error_t err;
  const int in_channels = 2;
  const int out_channels = 2;
  Parameter w[in_channels][out_channels], b[out_channels], x[in_channels],
      tmp_prod[in_channels][out_channels], w_by_x[out_channels],
      h[out_channels], h_relu[out_channels], y[out_channels],
      err_terms[out_channels], err_sq[out_channels], result;
  float W[in_channels][out_channels] = {{1.0, 2.0}, {3.0, 4.0}};

  for (int i = 0; i < in_channels; i++) {
    for (int j = 0; j < out_channels; j++) {
      init_parameter(&w[i][j], W[i][j]);
    }
  }
  for (int i = 0; i < in_channels; i++) {
    for (int j = 0; j < out_channels; j++) {
      init_parameter(&tmp_prod[i][j], 0.0);
    }
  }
  for (int j = 0; j < out_channels; j++) {
    init_parameter(&w_by_x[j], 0.0);
    init_parameter(&b[j], 0.0);
    init_parameter(&h[j], 0.0);
    init_parameter(&h_relu[j], 0.0);
    init_parameter(&err_sq[j], 0.0);
  }

  init_parameter(&x[0], 1.0);
  init_parameter(&x[1], 1.0);

  init_parameter(&y[0], 10.0);
  init_parameter(&y[1], 14.0);

  for (int i = 0; i < in_channels; i++) {
    for (int j = 0; j < out_channels; j++) {
      mult(&w[i][j], &x[i], &tmp_prod[i][j]);
    }
  }
  for (int j = 0; j < out_channels; j++) {
    add(&tmp_prod[0][j], &tmp_prod[1][j], &w_by_x[j]);
  }
  for (int i = 0; i < out_channels; i++) {
    add(&w_by_x[i], &b[i], &h[i]);
  }

  // Apply ReLU nonlinearity
  for (int i = 0; i < out_channels; i++) {
    relu_(&h[i], &h_relu[i]);
  }

  sub(&h_relu[0], &y[0], &err_terms[0]);
  sub(&h_relu[1], &y[1], &err_terms[1]);

  mult(&err_terms[0], &err_terms[0], &err_sq[0]);
  mult(&err_terms[1], &err_terms[1], &err_sq[1]);

  add(&err_sq[0], &err_sq[1], &result);

  printf("Initial loss: %.2f\n", result.value);
  print_mlp_parameters(w, b);

  backward(&result);
  Parameter *params[] = {&w[0][0], &w[0][1], &w[1][0], &w[1][1], &b[0], &b[1]};
  print_parameters("Before Optimization", params, 6);

  adam_optimizer(params, 6, 0.01f, 0.9f, 0.999f, 100, 1e-6f, 1e-8f);

  print_mlp_parameters(w, b);

  // Recompute forward pass with optimized parameters
  Parameter new_tmp_prod[in_channels][out_channels], new_w_by_x[out_channels],
      new_h[out_channels], new_h_relu[out_channels],
      new_err_terms[out_channels], new_err_sq[out_channels], new_result;

  // Initialize new computation parameters
  for (int i = 0; i < in_channels; i++) {
    for (int j = 0; j < out_channels; j++) {
      init_parameter(&new_tmp_prod[i][j], 0.0);
    }
  }
  for (int j = 0; j < out_channels; j++) {
    init_parameter(&new_w_by_x[j], 0.0);
    init_parameter(&new_h[j], 0.0);
    init_parameter(&new_h_relu[j], 0.0);
    init_parameter(&new_err_terms[j], 0.0);
    init_parameter(&new_err_sq[j], 0.0);
  }
  init_parameter(&new_result, 0.0);

  // Forward pass with optimized weights
  for (int i = 0; i < in_channels; i++) {
    for (int j = 0; j < out_channels; j++) {
      mult(&w[i][j], &x[i], &new_tmp_prod[i][j]);
    }
  }
  for (int j = 0; j < out_channels; j++) {
    add(&new_tmp_prod[0][j], &new_tmp_prod[1][j], &new_w_by_x[j]);
  }
  for (int i = 0; i < out_channels; i++) {
    add(&new_w_by_x[i], &b[i], &new_h[i]);
  }

  // Apply ReLU nonlinearity
  for (int i = 0; i < out_channels; i++) {
    relu_(&new_h[i], &new_h_relu[i]);
  }

  sub(&new_h_relu[0], &y[0], &new_err_terms[0]);
  sub(&new_h_relu[1], &y[1], &new_err_terms[1]);

  mult(&new_err_terms[0], &new_err_terms[0], &new_err_sq[0]);
  mult(&new_err_terms[1], &new_err_terms[1], &new_err_sq[1]);

  add(&new_err_sq[0], &new_err_sq[1], &new_result);

  printf("Actual final loss: %.2f\n", new_result.value);
  printf("Loss reduction: %.2f -> %.2f (%.1f%% improvement)\n", result.value,
         new_result.value,
         100.0f * (result.value - new_result.value) / result.value);

  err = save_graph(&new_result, "build/mlp_graph.dot");
  if (err != ADAM_SUCCESS) {
    fprintf(stderr, "Error saving graph: %d\n", err);
    return err;
  }

  return EXIT_SUCCESS;
}
