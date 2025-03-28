#include "parameter.h"
#include <stdlib.h>

Parameter add(Parameter p1, Parameter p2) {
  Parameter out;
  out.value = p1.value + p2.value;
  out.op = ADD;
  out.args = (Parameter **)malloc(sizeof(Parameter *) * 2);
  out.args[0] = p1;
  out.args[1] = p2;
  return out;
}
Parameter sub(Parameter p1, Parameter p2) {
  Parameter out;
  printf("here");
  out.value = p1.value - p2.value;
  out.op = SUB;
  out.args = (Parameter *)malloc(sizeof(Parameter) * 2);
  out.args[0] = p1;
  out.args[1] = p2;
  return out;
}
Parameter mult(Parameter p1, Parameter p2) {
  Parameter out;
  out.value = p1.value * p2.value;
  out.args = (Parameter *)malloc(sizeof(Parameter) * 2);
  out.op = MULT;
  return out;
}
Parameter divide(Parameter p1, Parameter p2) {
  Parameter out;
  out.value = p1.value / p2.value;
  out.op = DIVIDE;
  return out;
}

void display(Parameter p) {
  Parameter next = p;
  if (next.op != NULL) {
    printf("Parameter: {value: %e, op: %s}", next.value, next.op);
  } else {
    printf("Parameter: {value: %e, op: NULL}", next.value);
  }
}
