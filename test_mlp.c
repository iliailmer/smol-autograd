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
      h[out_channels], y[out_channels], err_terms[out_channels],
      err_sq[out_channels], result;
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

  // skip nonlinearity

  sub(&h[0], &y[0], &err_terms[0]);
  sub(&h[1], &y[1], &err_terms[1]);

  mult(&err_terms[0], &err_terms[0], &err_sq[0]);
  mult(&err_terms[1], &err_terms[1], &err_sq[1]);

  add(&err_sq[0], &err_sq[1], &result);

  backward(&result);
  err = save_graph(&result, "mlp_graph.dot");
  if (err != ADAM_SUCCESS) {
    fprintf(stderr, "Error saving graph: %d\n", err);
    return err;
  }

  return EXIT_SUCCESS;
}
