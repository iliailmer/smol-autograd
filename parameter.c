#include "parameter.h"
#include <stdlib.h>
#include <string.h>

void init_parameter(Parameter *p, float value) {
  p->value = value;
  p->grad = 0.0;
  p->op = "None";
}

void backward(Parameter *p) {
  p->grad = 1.0;
  p->grad_fn(p);
}

void add_grad(Parameter *p) {
  p->args[0]->grad += 1 * p->grad;
  p->args[1]->grad += 1 * p->grad;
}
Parameter add(Parameter *p1, Parameter *p2) {
  Parameter out;
  out.value = p1->value + p2->value;
  out.op = ADD;
  out.args = (Parameter **)malloc(sizeof(Parameter *) * 2);
  out.args[0] = p1;
  out.args[1] = p2;
  out.grad_fn = add_grad;
  return out;
}

void add_num_grad(Parameter *p) {
  p->args[0]->grad += 1 * p->grad;
}
Parameter add_num(Parameter *p1, float p2) {
  Parameter out;
  out.value = p1->value + p2;
  out.op = ADD;
  out.args = (Parameter **)malloc(sizeof(Parameter *) * 2);
  out.args[0] = p1;
  out.args[1] = NULL;
  out.grad_fn = add_num_grad;
  return out;
}

void sub_grad(Parameter *p) {
  p->args[0]->grad -= 1 * p->grad;
  p->args[1]->grad -= 1 * p->grad;
}
Parameter sub(Parameter *p1, Parameter *p2) {
  Parameter out;
  out.value = p1->value - p2->value;
  out.op = SUB;
  out.args = (Parameter **)malloc(sizeof(Parameter *) * 2);
  out.args[0] = p1;
  out.args[1] = p2;
  out.grad_fn = sub_grad;
  return out;
}

void sub_num_grad(Parameter *p) {
  p->args[0]->grad -= 1 * p->grad;
}
Parameter sub_num(Parameter *p1, float p2) {
  Parameter out;
  out.value = p1->value - p2;
  out.op = SUB;
  out.args = (Parameter **)malloc(sizeof(Parameter *) * 2);
  out.args[0] = p1;
  out.args[1] = NULL;
  out.grad_fn = sub_num_grad;
  return out;
}
void zero_grad(Parameter *p) {
  p->grad = 0.0f;
  if (p->args[0] != NULL) {
    zero_grad(p->args[0]);
  }
  if (p->args[1] != NULL) {
    zero_grad(p->args[1]);
  }
}

void mul_grad(Parameter *p) {
  p->args[0]->grad += p->args[1]->value * p->grad;
  p->args[1]->grad += p->args[0]->value * p->grad;
}
Parameter mult(Parameter *p1, Parameter *p2) {
  Parameter out;
  out.value = p1->value * p2->value;
  out.args = (Parameter **)malloc(sizeof(Parameter *) * 2);
  out.op = MULT;
  out.args[0] = p1;
  out.args[1] = p2;
  out.grad_fn = mul_grad;
  return out;
}

void mul_num_grad(Parameter *p) {
  p->args[0]->grad += p->args[0]->cnst * p->grad;
}
Parameter mul_num(Parameter *p1, float p2) {
  Parameter out;
  out.value = p1->value * p2;
  out.cnst = p2;
  out.op = MULT;
  out.args = (Parameter **)malloc(sizeof(Parameter *) * 2);
  out.args[0] = p1;
  out.args[1] = NULL;
  out.grad_fn = mul_num_grad;
  return out;
}

void div_grad(Parameter *p) {
  float x = p->args[0]->value;
  float y = p->args[1]->value;
  float incoming_grad = p->grad;

  p->args[0]->grad += (1.0f / y) * incoming_grad;
  p->args[1]->grad += (-x / (y * y)) * incoming_grad;
}
Parameter divide(Parameter *p1, Parameter *p2) {
  Parameter out;
  out.op = DIVIDE;
  out.value = p1->value / p2->value;
  out.args = (Parameter **)malloc(sizeof(Parameter *) * 2);
  out.args[0] = p1;
  out.args[1] = p2;
  out.grad_fn = div_grad;
  return out;
}

void display(const Parameter *p) {
  if (p == NULL) {
    printf("Parameter: {NULL pointer}\n");
    return;
  }

  // Count arguments if args is not NULL and null-terminated
  size_t arg_count = 0;
  if (p->args != NULL) {
    while (p->args[arg_count] != NULL) {
      arg_count++;
    }
  }

  // Display different formats based on whether op is set
  if (p->op != NULL) {
    if (p->args != NULL) {
      printf("Parameter: {value: %e, grad: %e, op: %s, # args: %zu}\n",
             p->value, p->grad, p->op, arg_count);
    } else {
      printf("Parameter: {value: %e, grad: %e, op: %s, # args: None}\n",
             p->value, p->grad, p->op);
    }
  } else {
    printf("Parameter: {value: %e, grad: %e, op: ''}\n",
           p->value, p->grad);
  }
}
