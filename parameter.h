// implementing Autograd (basic)
// TODO: zero grad, relu, sigmoid, mse,
#include <stdio.h>
#include <stdlib.h>
#ifndef PARAMETER_H
#define PARAMETER_H
typedef enum { BINARY, UNARY } op_type;
typedef enum { ADD, MUL, DIV, POW, EXP, TANH, RELU } op_name;

typedef struct Parameter {
  float value;
  float grad;
  struct OperationNode *prev;
  int visited;  // will be used for topological sort;
  int exponent; // for the pow operation
} Parameter;

typedef struct OperationNode {
  op_name _op_name;
  op_type _op_type;
  Parameter **inputs;
  size_t n_inputs;
  void (*backward_fn)(struct Parameter *self);
} OperationNode;
#define MAX_GRAPH_SIZE 1024

void init_parameter(Parameter *p, float value);
OperationNode *add(Parameter *p1, Parameter *p2, Parameter *result);
OperationNode *mult(Parameter *p1, Parameter *p2, Parameter *result);
OperationNode *divide(Parameter *p1, Parameter *p2, Parameter *result);
OperationNode *power(Parameter *p1, int exponent, Parameter *result);
OperationNode *exp_(Parameter *p1, Parameter *result);

void backward(Parameter *p);
void save_graph(Parameter *p, const char *filename);
#endif
