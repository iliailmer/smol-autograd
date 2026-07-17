#include "autograd.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

static size_t param_size(const Parameter *p) {
  return p->rank == Scalar ? 1 : p->shape[0];
}

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
  p->backward = NULL;
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
  p->backward = NULL;
}

static void add_backward(Parameter *p) {
  size_t n = param_size(p);
  for (size_t i = 0; i < n; i++) {
    p->inputs[0]->grad[i] += p->grad[i];
    p->inputs[1]->grad[i] += p->grad[i];
  }
}

static void mul_backward(Parameter *p) {
  size_t n = param_size(p);
  for (size_t i = 0; i < n; i++) {
    p->inputs[0]->grad[i] += p->grad[i] * p->inputs[1]->data[i];
    p->inputs[1]->grad[i] += p->grad[i] * p->inputs[0]->data[i];
  }
}

void add(Parameter *a, Parameter *b, Parameter *output) {
  size_t n = param_size(output);
  for (size_t i = 0; i < n; i++)
    output->data[i] = a->data[i] + b->data[i];
  output->inputs = malloc(2 * sizeof(Parameter *));
  output->inputs[0] = a;
  output->inputs[1] = b;
  output->n_inputs = 2;
  output->op = op_add;
  output->backward = add_backward;
}

void mul(Parameter *a, Parameter *b, Parameter *output) {
  size_t n = param_size(output);
  for (size_t i = 0; i < n; i++)
    output->data[i] = a->data[i] * b->data[i];
  output->inputs = malloc(2 * sizeof(Parameter *));
  output->inputs[0] = a;
  output->inputs[1] = b;
  output->n_inputs = 2;
  output->op = op_mul;
  output->backward = mul_backward;
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

static void print_node_inline(const Parameter *p) {
  if (p->rank == Scalar)
    printf("%s [%s  data=%.2f  grad=%.2f]", p->name, op_code_name(p->op),
           p->data[0], p->grad[0]);
  else
    printf("%s [%s  data=%.2f...  grad=%.2f...]", p->name, op_code_name(p->op),
           p->data[0], p->grad[0]);
}

static void print_graph_impl(const Parameter *p, const char *prefix,
                              int is_last) {
  printf("%s%s", prefix, is_last ? "└── " : "├── ");
  print_node_inline(p);
  printf("\n");

  char child_prefix[256];
  snprintf(child_prefix, sizeof(child_prefix), "%s%s", prefix,
           is_last ? "    " : "│   ");
  for (int i = 0; i < p->n_inputs; i++)
    print_graph_impl(p->inputs[i], child_prefix, i == p->n_inputs - 1);
}

void print_graph(const Parameter *p) {
  print_node_inline(p);
  printf("\n");
  for (int i = 0; i < p->n_inputs; i++)
    print_graph_impl(p->inputs[i], "", i == p->n_inputs - 1);
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
  da->params = (Parameter **)malloc(da->cap * sizeof(Parameter *));
};

void dyn_array_free(dyn_array *da) { free(da->params); }

void dyn_array_display(dyn_array *da) {
  printf("len: %lu; cap: %lu\n", da->len, da->cap);
  for (size_t i = 0; i < da->len; i++) {
    print_parameter(da->params[i]);
  }
}

void dyn_array_append(dyn_array *da, Parameter *p) {
  if (da->len >= da->cap) {
    da->cap = da->cap * 2;
    Parameter **tmp = realloc(da->params, da->cap * sizeof(Parameter *));
    da->params = tmp;
  }
  da->params[da->len++] = p;
}

void topo_sort(Parameter *p, dyn_array *topo) {
  p->visited = 1;
  for (int i = 0; i < p->n_inputs; i++) {
    if (p->inputs[i]->visited == 0)
      topo_sort(p->inputs[i], topo);
  }
  dyn_array_append(topo, p);
}

void backward(dyn_array *topo) {
  if (topo->len == 0) return;
  size_t n = param_size(topo->params[topo->len - 1]);
  for (size_t i = 0; i < n; i++)
    topo->params[topo->len - 1]->grad[i] = 1.0f;
  for (int i = (int)topo->len - 1; i >= 0; i--) {
    Parameter *p = topo->params[i];
    if (p->backward != NULL)
      p->backward(p);
  }
}

void free_parameter(Parameter *p) {
  free(p->data);
  free(p->grad);
  free(p->shape);
  free(p->inputs);
  free(p);
}
