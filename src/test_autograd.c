#include "parameter.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
  adam_error_t err;

  Parameter x, x_squared, result;

  printf("Initializing parameters...\n");
  init_parameter(&x, 3.0);

  printf("Computing: f(x) = x² + x where x=%.1f\n", x.value);

  err = mult(&x, &x, &x_squared);
  if (err != ADAM_SUCCESS) {
    fprintf(stderr, "Error in x*x: %d\n", err);
    return err;
  }
  printf("x² = %.1f\n", x_squared.value);

  // result = x_squared + x (x is reused again)
  err = add(&x_squared, &x, &result);
  if (err != ADAM_SUCCESS) {
    fprintf(stderr, "Error in x² + x: %d\n", err);
    return err;
  }
  printf("f(x) = x² + x = %.1f\n", result.value);

  printf("Forward pass completed. Result: %f\n", result.value);

  err = backward(&result);
  if (err != ADAM_SUCCESS) {
    fprintf(stderr, "Error in backward pass: %d\n", err);
    return err;
  }

  printf("Backward pass completed successfully.\n");
  printf("Gradient: x.grad=%.1f (should be df/dx = 2x + 1 = %.1f)\n", x.grad,
         2 * x.value + 1);

  err = save_graph(&result, "build/quadratic_graph.dot");
  if (err != ADAM_SUCCESS) {
    fprintf(stderr, "Error saving graph: %d\n", err);
    return err;
  }

  free_parameter_graph(&result);

  printf("Graph saved to build/quadratic_graph.dot\n");

  return EXIT_SUCCESS;
}
