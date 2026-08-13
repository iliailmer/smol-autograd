#ifndef AUTOGRAD_H
#define AUTOGRAD_H
#include <stddef.h>

typedef enum {
  op_leaf = -1,
  op_add = 0,
  op_mul = 1,
  op_div = 2,
  op_sub = 3,
  op_pow = 4,
  op_exp = 5,
  op_matmul = 6,
} op_code;

typedef enum { Scalar = 0, Vector = 1, Matrix = 2 } tensor_rank;

typedef struct Parameter Parameter;
typedef void (*backward_fn)(Parameter *p);
struct Parameter {
  float *data;
  float *grad;
  float exponent;
  tensor_rank rank;
  size_t *shape;
  op_code op;
  struct Parameter **inputs;
  backward_fn backward;
  int n_inputs;
  int visited;
  char *name;
};

typedef struct {
  size_t cap;
  size_t len;
  Parameter **params;
} dyn_array;

// dynamic array
void dyn_array_init(dyn_array *da);
void dyn_array_free(dyn_array *da);
void dyn_array_append(dyn_array *da, Parameter *p);
void dyn_array_display(dyn_array *da);

// Parameter
void init_0d(Parameter *p, char *name);
void init_1d(Parameter *p, size_t width, char *name);
void init_2d(Parameter *p, size_t rows, size_t cols, char *name);
void free_parameter(Parameter *p);
void print_parameter(Parameter *p);
void print_graph(const Parameter *p);

// operations
void add(Parameter *a, Parameter *b, Parameter *output);
void sub(Parameter *a, Parameter *b, Parameter *output);
void mul(Parameter *a, Parameter *b, Parameter *output);
void matmul(Parameter *a, Parameter *b, Parameter *output);
void pow_(Parameter *a, float b, Parameter *output);
// gradients
void zero_grad(Parameter *p);

// graph
void topo_sort(Parameter *p, dyn_array *topo);
void backward(dyn_array *topo);

#endif
