#include "autograd.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
// TODO: global backward method
void init_0d(Parameter *p, char *name) {
  p->data = (float *)calloc(1, sizeof(float));
  p->grad = (float *)calloc(1, sizeof(float));
  p->op = op_leaf;
  p->rank = Scalar;
  p->inputs = NULL;
  p->n_inputs = 0;
  p->shape = NULL;
  p->name = name;
  p->visited = 0;
}
void init_1d(Parameter *p, size_t width, char *name) {
  p->data = (float *)calloc(width, sizeof(float));
  p->grad = (float *)calloc(width, sizeof(float));
  p->op = op_leaf;
  p->rank = Vector;
  p->inputs = NULL;
  p->n_inputs = 0;
  p->shape = (size_t *)malloc(p->rank * sizeof(size_t));
  p->shape[0] = width;
  p->name = name;
  p->visited = 0;
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
  output->op = op_add;
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
    printf("%.2f%s", p->data[i], i < len - 1 ? ", " : "");
  }
  printf("], shape=[");
  for (size_t i = 0; i < p->rank; i++) {
    printf("%zu%s", p->shape[i], i < p->rank - 1 ? ", " : "");
  }
  printf("], grad=[");
  for (size_t i = 0; i < len; i++) {
    printf("%.2f%s", p->grad[i], i < len - 1 ? ", " : "");
  }
  printf("], name=%s)\n", p->name);
  // printf("])\n");
}

void zero_grad(Parameter *p) {
  size_t len = (p->rank == Scalar) ? 1 : p->shape[0];
  for (size_t i = 0; i < len; i++) {
    p->grad[i] = 0.0f;
  }
}

void dyn_array_init(dyn_array *da) {
  da->cap = 1;
  da->len = 0;
  da->params = (Parameter **)malloc(da->cap * sizeof(Parameter));
  for (size_t i = 0; i < da->len; i++) {
    da->params[i] = (Parameter *)malloc(sizeof(Parameter));
  }
};

void dyn_array_free(dyn_array *da) {
  for (size_t i = 0; i < da->len; i++) {
    free(da->params[i]);
  }
  free(da->params);
  free(da);
}

void dyn_array_append(dyn_array *da, Parameter *p) {
  if (da->len < da->cap) {
    da->len += 1;
    da->params[da->len - 1] = p;
  } else {
    da->cap = da->cap * 2;
    Parameter **tmp = realloc(da->params, da->cap * sizeof(Parameter));
    da->params = tmp;
    da->len += 1;
    da->params[da->len - 1] = p;
  }
}
void topo_sort(Parameter *p, dyn_array *topo) {
  p->visited = 1;
  for (size_t i = 0; i < p->n_inputs; i++) {
    if (p->inputs[i]->visited == 0) {
      topo_sort(p->inputs[i], topo);
    }
  }
  dyn_array_append(topo, p);
}

void backward(dyn_array *topo) {
  for (size_t i = topo->len - 1; i >= 0; i--) {
    // TODO: need to figure out how to dispatch the correct
    // backward function based on op code and inputs into
    // params[i]; Possibly a switch?
    // switch (topo->params[i]->op) {
    // case (op_add):
  }
}

void free_parameter(Parameter *p) {
  free(p->data);
  free(p->grad);
  free(p->shape);
  free(p->inputs);
  free(p);
}
