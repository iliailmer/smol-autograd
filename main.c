#include "src/autograd.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

// f(a,b) = a * (a + b), df/da = 2a+b, df/db = a
static void example_scalar() {
  printf("=== scalar: f = a * (a + b), a=0.5 b=1.5 ===\n\n");
  Parameter *a = malloc(sizeof(Parameter));
  Parameter *b = malloc(sizeof(Parameter));
  Parameter *ab = malloc(sizeof(Parameter));
  Parameter *f = malloc(sizeof(Parameter));
  init_0d(a, "a");
  init_0d(b, "b");
  init_0d(ab, "a+b");
  init_0d(f, "f");
  a->data[0] = 0.5f;
  b->data[0] = 1.5f;
  add(a, b, ab);
  mul(a, ab, f);

  printf("-- graph before backward --\n");
  print_graph(f);
  f->grad[0] = 1.0f;
  dyn_array *topo = malloc(sizeof(dyn_array));
  dyn_array_init(topo);
  topo_sort(f, topo);
  backward(topo);
  printf("\n-- graph after backward --\n");
  print_graph(f);
  printf("\nexpected: f=1.00  da=2.50  db=0.50\n");
  free_parameter(a);
  free_parameter(b);
  free_parameter(ab);
  free_parameter(f);
}

// C = A @ B
// A = [[1,2],[3,4]]  B = [[1,1],[1,1]]
// C = [[3,3],[7,7]]
// dA (dC=ones) = B^T summed = [[2,2],[2,2]]
// dB (dC=ones) = A^T summed = [[4,4],[6,6]]
static void example_matmul(size_t n) {
  printf("\n=== matmul %zux%zu: C = A @ B ===\n\n", n, n);
  Parameter *a = malloc(sizeof(Parameter));
  Parameter *b = malloc(sizeof(Parameter));
  Parameter *c = malloc(sizeof(Parameter));
  init_2d(a, n, n, "A");
  init_2d(b, n, n, "B");
  init_2d(c, n, n, "C");

  // A: sequential 1..n*n, B: all ones
  for (size_t i = 0; i < n * n; i++) {
    a->data[i] = (float)(i + 1);
    b->data[i] = 1.0f;
  }

  matmul(a, b, c);

  printf("-- inputs --\n");
  print_parameter(a);
  print_parameter(b);

  printf("\n-- graph (before backward) --\n");
  print_graph(c);

  // seed dC = ones and run backward
  for (size_t i = 0; i < n * n; i++)
    c->grad[i] = 1.0f;
  dyn_array *topo = malloc(sizeof(dyn_array));
  dyn_array_init(topo);
  topo_sort(c, topo);
  backward(topo);

  printf("\n-- C (forward) --\n");
  print_parameter(c);
  printf("\n-- gradients --\n");
  print_parameter(a);
  print_parameter(b);
}

int main(void) {
  example_scalar();
  example_matmul(2);
  example_matmul(4);
  return 0;
}
