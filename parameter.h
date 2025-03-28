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
  float grad;                             // gradient value
  char *op;                               // '+', '*', etc
  struct Parameter **args;                // inputs to op
  void (*grad_fn)(struct Parameter self); // function to compute gradient
} Parameter;

Parameter add(Parameter p1, Parameter p2);
Parameter sub(Parameter p1, Parameter p2);
Parameter mult(Parameter p1, Parameter p2);
Parameter divide(Parameter p1, Parameter p2);

void display(Parameter p);
#endif
