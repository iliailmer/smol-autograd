#include "src/autograd.h"
#include <stdlib.h>

int main(int argc, char *argv[]) {
  Parameter *a = malloc(sizeof(Parameter));
  Parameter *b = malloc(sizeof(Parameter));
  Parameter *s = malloc(sizeof(Parameter));
  Parameter *c = malloc(sizeof(Parameter));
  init_0d(a);
  init_0d(b);
  init_0d(s);
  init_0d(c);

  a->data[0] = 0.5;
  a->grad[0] = 0;
  b->data[0] = 1.5;
  b->grad[0] = 0;

  s->grad[0] = 1;
  c->grad[0] = 1;

  add_0d(a, b, s);
  add_0d_backward(s);

  print_parameter(a);
  print_parameter(b);
  print_parameter(s);

  zero_grad(a);
  zero_grad(b);

  mul_0d(a, b, c);
  mul_0d_backward(c);

  print_parameter(a);
  print_parameter(b);
  print_parameter(c);

  free_parameter(a);
  free_parameter(b);
  free_parameter(s);
  return 0;
}
