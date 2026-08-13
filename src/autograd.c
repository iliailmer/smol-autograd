#include "autograd.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

static size_t param_size(const Parameter *p) {
  if (p->rank == Scalar)
    return 1;
  if (p->rank == Vector)
    return p->shape[0];
  return p->shape[0] * p->shape[1];
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
  p->exponent = 1.0;
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
  p->exponent = 1.0;
}

void init_2d(Parameter *p, size_t rows, size_t cols, char *name) {
  p->data = (float *)calloc(rows * cols, sizeof(float));
  p->grad = (float *)calloc(rows * cols, sizeof(float));
  p->op = op_leaf;
  p->rank = Matrix;
  p->inputs = NULL;
  p->n_inputs = 0;
  p->shape = (size_t *)malloc(2 * sizeof(size_t));
  p->shape[0] = rows;
  p->shape[1] = cols;
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

void pow_(Parameter *a, float b, Parameter *output) {
  size_t n = param_size(output);
  for (size_t i = 0; i < n; i++)
    output->data[i] = pow(a->data[i], b);
  output->inputs = malloc(1 * sizeof(Parameter *));
  output->inputs[0] = a;
  output->exponent = b;
  output->n_inputs = 1;
  output->op = op_pow;
  output->backward = add_backward;
}

static void sub_backward(Parameter *p) {
  size_t n = param_size(p);
  for (size_t i = 0; i < n; i++) {
    p->inputs[0]->grad[i] += p->grad[i];
    p->inputs[1]->grad[i] -= p->grad[i];
  }
}
void sub(Parameter *a, Parameter *b, Parameter *output) {
  size_t n = param_size(output);
  for (size_t i = 0; i < n; i++)
    output->data[i] = a->data[i] - b->data[i];
  output->inputs = malloc(2 * sizeof(Parameter *));
  output->inputs[0] = a;
  output->inputs[1] = b;
  output->n_inputs = 2;
  output->op = op_sub;
  output->backward = sub_backward;
}

static void matmul_backward(Parameter *p) {
  Parameter *a = p->inputs[0]; // m×k
  Parameter *b = p->inputs[1]; // k×n
  size_t m = a->shape[0], k = a->shape[1], n = b->shape[1];
  // dL/dA[i,l] = sum_j dL/dC[i,j] * B[l,j]
  for (size_t i = 0; i < m; i++)
    for (size_t l = 0; l < k; l++)
      for (size_t j = 0; j < n; j++)
        a->grad[i * k + l] += p->grad[i * n + j] * b->data[l * n + j];
  // dL/dB[l,j] = sum_i A[i,l] * dL/dC[i,j]
  for (size_t l = 0; l < k; l++)
    for (size_t j = 0; j < n; j++)
      for (size_t i = 0; i < m; i++)
        b->grad[l * n + j] += a->data[i * k + l] * p->grad[i * n + j];
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

void matmul(Parameter *a, Parameter *b, Parameter *output) {
  size_t m = a->shape[0], k = a->shape[1], n = b->shape[1];
  for (size_t i = 0; i < m; i++)
    for (size_t j = 0; j < n; j++) {
      output->data[i * n + j] = 0.0f;
      for (size_t l = 0; l < k; l++)
        output->data[i * n + j] += a->data[i * k + l] * b->data[l * n + j];
    }
  output->inputs = malloc(2 * sizeof(Parameter *));
  output->inputs[0] = a;
  output->inputs[1] = b;
  output->n_inputs = 2;
  output->op = op_matmul;
  output->backward = matmul_backward;
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
  case op_matmul:
    return "matmul";
  }
  return "unknown";
}

static void print_node_inline(const Parameter *p) {
  const char *op = op_code_name(p->op);
  if (p->rank == Scalar)
    printf("%s [%s  data=%.2f  grad=%.2f]", p->name, op, p->data[0],
           p->grad[0]);
  else if (p->rank == Vector)
    printf("%s [%s  n=%zu  data=%.2f...  grad=%.2f...]", p->name, op,
           p->shape[0], p->data[0], p->grad[0]);
  else
    printf("%s [%s  %zux%zu  data[0,0]=%.2f  grad[0,0]=%.2f]", p->name, op,
           p->shape[0], p->shape[1], p->data[0], p->grad[0]);
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
  if (p->rank == Matrix) {
    size_t r = p->shape[0], c = p->shape[1];
    printf("%s [%s  %zux%zu]\n", p->name, op_code_name(p->op), r, c);
    printf("  data:\n");
    for (size_t i = 0; i < r; i++) {
      printf("    [");
      for (size_t j = 0; j < c; j++)
        printf(" %6.2f", p->data[i * c + j]);
      printf(" ]\n");
    }
    printf("  grad:\n");
    for (size_t i = 0; i < r; i++) {
      printf("    [");
      for (size_t j = 0; j < c; j++)
        printf(" %6.2f", p->grad[i * c + j]);
      printf(" ]\n");
    }
  } else {
    size_t len = param_size(p);
    printf("%s [%s] data=[", p->name, op_code_name(p->op));
    for (size_t i = 0; i < len; i++)
      printf("%.2f%s", p->data[i], i < len - 1 ? ", " : "");
    printf("] grad=[");
    for (size_t i = 0; i < len; i++)
      printf("%.2f%s", p->grad[i], i < len - 1 ? ", " : "");
    printf("]\n");
  }
}

void zero_grad(Parameter *p) {
  size_t n = param_size(p);
  for (size_t i = 0; i < n; i++)
    p->grad[i] = 0.0f;
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
  if (topo->len == 0)
    return;
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
