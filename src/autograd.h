#ifndef AUTOGRAD_H
#define AUTOGRAD_H
#include <stddef.h>
#define DUMP(varname) fprintf(stderr, "%s = %x", #varname, varname);
typedef enum {
  op_leaf = -1,
  op_add = 0,
  op_mul = 1,
  op_div = 2,
  op_sub = 3,
  op_pow = 4,
  op_exp = 5,
} op_code;

typedef enum { Leaf = 0, Unary = 1, Binary = 2 } op_input_size;
typedef enum { Scalar = 0, Vector = 1, Matrix = 2 } tensor_rank;
typedef void (*forward_fn)(const float *input1, const float *input2, float *out,
                           int size);
typedef void (*backward_fn)(float *grad_in1, float *grad_in2,
                            const float *grad_out, const float *input1,
                            const float *input2, int size);
// typedef struct {
//   op_code op;
//   const char *name;
//   forward_fn forward;
//   backward_fn backward;
//   op_input_size n_input;
// } op_descriptor;

typedef struct Parameter {
  float *data;
  float *grad;
  tensor_rank rank;
  size_t *shape;
  op_code op;
  struct Parameter **inputs;
  int n_inputs;
  // op_descriptor op_desc;
  int visited;
  char *name;
} Parameter;

typedef struct {
  size_t cap;
  size_t len;
  Parameter **params;
} dyn_array;

// dynamic array

void dyn_array_init(dyn_array *da);
void dyn_array_append(dyn_array *da, Parameter *p);
void dyn_array_append(dyn_array *da, Parameter *p);

// Parameter
void init_0d(Parameter *p, char *name);
void init_1d(Parameter *p, size_t width, char *name);
void free_parameter(Parameter *p);
void print_parameter(Parameter *p);

// operations
void add_0d(Parameter *a, Parameter *b, Parameter *output);
void add_1d(Parameter *a, Parameter *b, Parameter *output);

void mul_0d(Parameter *a, Parameter *b, Parameter *output);
void mul_1d(Parameter *a, Parameter *b, Parameter *output);

// gradients
void zero_grad(Parameter *p);

// backward
void add_0d_backward(Parameter *a);
void add_1d_backward(Parameter *a);

void mul_0d_backward(Parameter *a);
void mul_1d_backward(Parameter *a);

void topo_sort(Parameter *p, dyn_array *topo);
#endif
