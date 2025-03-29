// implementing Autograd (basic)
#include <stdio.h>
#include <stdlib.h>
#ifndef PARAMETER_H
#define PARAMETER_H
#define ADD "+"
#define SUB "-"
#define MULT "*"
#define DIVIDE "/"
typedef struct Parameter {
  float value;
  float grad;                              // gradient value
  float cnst;                              // constant factor in case we multiply by a number
  char *op;                                // '+', '*', etc
  struct Parameter **args;                 // inputs to op
  void (*grad_fn)(struct Parameter *self); // function to compute gradient
} Parameter;

void zero_grad(Parameter *p);
void init_parameter(Parameter *p1, float value);
Parameter add(Parameter *p1, Parameter *p2);
Parameter add_num(Parameter *p1, float p2);

Parameter sub(Parameter *p1, Parameter *p2);
Parameter sub_num(Parameter *p1, float p2);

Parameter mult(Parameter *p1, Parameter *p2);
Parameter mul_num(Parameter *p1, float p2);

Parameter divide(Parameter *p1, Parameter *p2);
Parameter div_num(Parameter *p1, float p2);
void backward(Parameter *p);
void display(const Parameter *p);
#endif
