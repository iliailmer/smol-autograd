// implementing Autograd (basic)
// TODO: zero grad, relu, sigmoid, mse,
#include <stdio.h>
#include <stdlib.h>
#ifndef PARAMETER_H
#define PARAMETER_H
#define T2D(t, i, j) (t->data[(i) * (t)->cols + (j)])

typedef enum {
  ADAM_SUCCESS = 0,
  ADAM_ERROR_NULL_POINTER = -1,
  ADAM_ERROR_DIVISION_BY_ZERO = -2,
  ADAM_ERROR_MEMORY_ALLOCATION = -3,
  ADAM_ERROR_INVALID_INPUT = -4,
  ADAM_ERROR_GRAPH_TOO_LARGE = -5
} adam_error_t;

typedef enum { BINARY, UNARY } op_type;
typedef enum { ADD, SUB, NEG, MUL, DIV, POW, EXP, TANH, RELU } op_name;

typedef struct Parameter {
  float value;
  float grad;
  struct OperationNode *prev;
  int visited;         // will be used for topological sort;
  int cleanup_visited; // for memory cleanup to avoid interfering with
                       // computation
  int export_visited;  // for graph export to avoid infinite recursion
  int exponent;        // for the pow operation
  int requires_grad;
} Parameter;

typedef struct Tensor {
  Parameter *data;
  size_t rows;
  size_t cols;
} Tensor;

typedef struct OperationNode {
  op_name _op_name;
  op_type _op_type;
  Parameter **inputs;
  size_t n_inputs;
  void (*backward_fn)(struct Parameter *self);
} OperationNode;
#define MAX_GRAPH_SIZE 1024

void init_parameter(Parameter *p, float value);
void init_tensor(Tensor *t, size_t rows, size_t cols);
void free_tensor(Tensor *t);
void free_operation_node(OperationNode *node);
void free_parameter_graph(Parameter *p);
void reset_export_visited(Parameter *p);
adam_error_t matmul(Tensor *t1, Tensor *t2, Tensor *result);

adam_error_t add(Parameter *p1, Parameter *p2, Parameter *result);
adam_error_t sub(Parameter *p1, Parameter *p2, Parameter *result);
adam_error_t neg(Parameter *p1, Parameter *result);
adam_error_t mult(Parameter *p1, Parameter *p2, Parameter *result);
adam_error_t divide(Parameter *p1, Parameter *p2, Parameter *result);
adam_error_t power(Parameter *p1, int exponent, Parameter *result);
adam_error_t exp_(Parameter *p1, Parameter *result);
adam_error_t tanh_(Parameter *p1, Parameter *result);
adam_error_t relu_(Parameter *p1, Parameter *result);

adam_error_t backward(Parameter *p);
adam_error_t save_graph(Parameter *p, const char *filename);

// Adam optimizer functions
float target_(float a, float b, float c, float d, float x);
float grad_target_(float a, float b, float c, float d, float x);
float adam(float a, float b, float c, float d, float x, float lr, float beta1,
           float beta2, int num_iter, float tol, float eps);
#endif
