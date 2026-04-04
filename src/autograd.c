#include "autograd.h"
#include <stdio.h>
#include <stdlib.h>

void init_0d(Parameter *p) {
  p->data = (float *)malloc(sizeof(float));
  p->grad = (float *)malloc(sizeof(float));
  p->op = op_leaf;
  p->rank = Scalar;
  p->inputs = NULL;
  p->n_inputs = 0;
  p->shape = NULL;
}
void init_1d(Parameter *p, size_t width) {
  p->data = (float *)malloc(width * sizeof(float));
  p->grad = (float *)malloc(width * sizeof(float));
  p->op = op_leaf;
  p->rank = Vector;
  p->inputs = NULL;
  p->n_inputs = 0;
  p->shape = (size_t *)malloc(p->rank * sizeof(size_t));
  p->shape[0] = width;
}

void add_0d(Parameter *a, Parameter *b, Parameter *output) {
  output->data[0] = a->data[0] + b->data[0];
  output->inputs = malloc(2 * sizeof(Parameter *));
  output->inputs[0] = a;
  output->inputs[1] = b;
  output->n_inputs = 2;
  output->op = op_add;
}

void add_0d_backward(Parameter *a) {
  a->inputs[0]->grad[0] += a->grad[0];
  a->inputs[1]->grad[0] += a->grad[0];
};

void mul_0d(Parameter *a, Parameter *b, Parameter *output) {
  output->data[0] = a->data[0] * b->data[0];
  output->inputs = malloc(2 * sizeof(Parameter *));
  output->inputs[0] = a;
  output->inputs[1] = b;
  output->n_inputs = 2;
  output->op = op_mul;
}

void mul_0d_backward(Parameter *a) {
  a->inputs[0]->grad[0] += a->grad[0] * a->inputs[1]->data[0];
  a->inputs[1]->grad[0] += a->grad[0] * a->inputs[0]->data[0];
};

void add_1d(Parameter *a, Parameter *b, Parameter *output) {
  for (size_t i = 0; i < output->shape[0]; i++) {
    output->data[i] = a->data[i] + b->data[i];
  }
  output->inputs = malloc(2 * sizeof(Parameter *));
  output->inputs[0] = a;
  output->inputs[1] = b;
  output->n_inputs = 2;
  output->op = op_leaf;
}

void add_1d_backward(Parameter *a) {
  for (size_t i = 0; i < a->shape[0]; i++) {
    a->inputs[0]->grad[i] += a->grad[i];
    a->inputs[1]->grad[i] += a->grad[i];
  }
}

void mul_1d(Parameter *a, Parameter *b, Parameter *output) {
  for (size_t i = 0; i < output->shape[0]; i++) {
    output->data[i] = a->data[i] * b->data[i];
  }
  output->inputs = malloc(2 * sizeof(Parameter *));
  output->inputs[0] = a;
  output->inputs[1] = b;
  output->n_inputs = 2;
  output->op = op_mul;
}

void mul_1d_backward(Parameter *a) {
  for (size_t i = 0; i < a->shape[0]; i++) {
    a->inputs[0]->grad[i] += a->grad[i] * a->inputs[1]->data[i];
    a->inputs[1]->grad[i] += a->grad[i] * a->inputs[0]->data[i];
  }
}

static const char *op_code_name(op_code op) {
  switch (op) {
  case op_leaf:
    return "leaf";
  case op_add:
    return "add";
  case op_mul:
    return "mul";
  case op_div:
    return "div";
  case op_sub:
    return "sub";
  case op_pow:
    return "pow";
  case op_exp:
    return "exp";
  }
  return "unknown";
}

void print_parameter(Parameter *p) {
  size_t len = (p->rank == Scalar) ? 1 : p->shape[0];
  printf("Parameter(op=%s, inputs=%d, data=[", op_code_name(p->op),
         p->n_inputs);
  for (size_t i = 0; i < len; i++) {
    printf("%.4f%s", p->data[i], i < len - 1 ? ", " : "");
  }
  printf("], grad=[");
  for (size_t i = 0; i < len; i++) {
    printf("%.4f%s", p->grad[i], i < len - 1 ? ", " : "");
  }
  printf("])\n");
}

void zero_grad(Parameter *p) {
  size_t len = (p->rank == Scalar) ? 1 : p->shape[0];
  for (size_t i = 0; i < len; i++) {
    p->grad[i] = 0.0f;
  }
}

void free_parameter(Parameter *a) {
  free(a->data);
  free(a->grad);
  free(a->shape);
  free(a->inputs);
  free(a);
}
