// TODO: Subtraction, negation
#include "parameter.h"
#include <math.h>

void add_grad(Parameter *result)
{
  printf("inside add_grad, value=%.2f, grad=%.2f\n", result->value,
         result->grad);

  result->prev->inputs[0]->grad += result->grad;
  result->prev->inputs[1]->grad += result->grad;
}
OperationNode *add(Parameter *p1, Parameter *p2, Parameter *result)
{
  OperationNode *add_node = malloc(sizeof(OperationNode));
  if (p1 == NULL) {
    printf("Uninitialized or NULL parameter p1");
    exit(1);
  }
  if (p2 == NULL) {
    printf("Uninitialized or NULL parameter p2");
    exit(1);
  }
  if (result == NULL) {
    printf("Uninitialized or NULL parameter result");
    exit(1);
  }
  add_node->_op_name = ADD;
  add_node->_op_type = BINARY;
  add_node->inputs = malloc(sizeof(Parameter *) * 2);
  add_node->inputs[0] = p1;
  add_node->inputs[1] = p2;
  add_node->n_inputs = 2;
  add_node->backward_fn = add_grad;

  result->value = p1->value + p2->value;
  result->prev = add_node;
  result->grad = 0.0;
  result->visited = 0;
  result->exponent = 1;

  return add_node;
}

void mult_grad(Parameter *result)
{
  printf("inside mult_grad, value=%.2f, grad=%.2f\n", result->value,
         result->grad);
  result->prev->inputs[0]->grad +=
      result->grad * result->prev->inputs[1]->value;
  result->prev->inputs[1]->grad +=
      result->grad * result->prev->inputs[0]->value;
}
OperationNode *mult(Parameter *p1, Parameter *p2, Parameter *result)
{
  OperationNode *mult_node = malloc(sizeof(OperationNode));
  if (p1 == NULL) {
    printf("Uninitialized or NULL parameter p1");
    exit(1);
  }
  if (p2 == NULL) {
    printf("Uninitialized or NULL parameter p2");
    exit(1);
  }
  if (result == NULL) {
    printf("Uninitialized or NULL parameter result");
    exit(1);
  }
  mult_node->_op_name = MUL;
  mult_node->_op_type = BINARY;
  mult_node->inputs = malloc(sizeof(Parameter *) * 2);
  mult_node->inputs[0] = p1;
  mult_node->inputs[1] = p2;
  mult_node->n_inputs = 2;
  mult_node->backward_fn = mult_grad;

  result->value = p1->value * p2->value;
  result->prev = mult_node;
  result->grad = 0.0;
  result->visited = 0;
  result->exponent = 1;

  return mult_node;
}
void divide_grad(Parameter *result)
{
  printf("inside divide_grad, value=%.2f, grad=%.2f\n", result->value,
         result->grad);
  // numerator is idx 0
  result->prev->inputs[0]->grad +=
      result->grad / result->prev->inputs[1]->value;
  // denominator is idx 1
  result->prev->inputs[1]->grad +=
      result->grad * result->prev->inputs[0]->value /
      (result->prev->inputs[1]->value * result->prev->inputs[1]->value);
}
OperationNode *divide(Parameter *p1, Parameter *p2, Parameter *result)
{
  OperationNode *div_node = malloc(sizeof(OperationNode));
  if (p1 == NULL) {
    printf("Uninitialized or NULL parameter p1");
    exit(1);
  }
  if (p2 == NULL) {
    printf("Uninitialized or NULL parameter p2");
    exit(1);
  }
  if (result == NULL) {
    printf("Uninitialized or NULL parameter result");
    exit(1);
  }
  div_node->_op_name = DIV;
  div_node->_op_type = BINARY;
  div_node->inputs = malloc(sizeof(Parameter *) * 2);
  div_node->inputs[0] = p1;
  div_node->inputs[1] = p2;
  div_node->n_inputs = 2;
  div_node->backward_fn = divide_grad;

  result->value = p1->value / p2->value;
  result->prev = div_node;
  result->grad = 0.0;
  result->visited = 0;
  result->exponent = 1;

  return div_node;
}

void pow_grad(Parameter *result)
{
  printf("inside pow_grad, value=%.2f, grad=%.2f\n", result->value,
         result->grad);
  int exponent = result->prev->inputs[0]->exponent;
  result->prev->inputs[0]->grad +=
      result->grad * exponent *
      pow(result->prev->inputs[0]->value, exponent - 1);
}
OperationNode *power(Parameter *p1, int exponent, Parameter *result)
{
  OperationNode *pow_node = malloc(sizeof(OperationNode));
  if (p1 == NULL) {
    printf("Uninitialized or NULL parameter p1");
    exit(1);
  }
  if (result == NULL) {
    printf("Uninitialized or NULL parameter result");
    exit(1);
  }
  pow_node->_op_name = POW;
  pow_node->_op_type = BINARY;
  pow_node->inputs = malloc(sizeof(Parameter *) * 1);
  pow_node->inputs[0] = p1;
  pow_node->n_inputs = 1;
  pow_node->backward_fn = pow_grad;

  result->value = pow(p1->value, exponent);
  result->prev = pow_node;
  result->grad = 0.0;
  result->visited = 0;
  result->exponent = exponent;

  return pow_node;
}

void exp_grad(Parameter *result)
{
  printf("inside exp_grad, value=%.2f, grad=%.2f\n", result->value,
         result->grad);
  result->prev->inputs[0]->grad += exp(result->prev->inputs[0]->value);
}
OperationNode *exp_(Parameter *p1, Parameter *result)
{

  OperationNode *exp_node = malloc(sizeof(OperationNode));
  if (p1 == NULL) {
    printf("Uninitialized or NULL parameter p1");
    exit(1);
  }
  if (result == NULL) {
    printf("Uninitialized or NULL parameter result");
    exit(1);
  }
  exp_node->_op_name = EXP;
  exp_node->_op_type = UNARY;
  exp_node->backward_fn = exp_grad;
  exp_node->inputs = malloc(sizeof(Parameter *) * 1);
  exp_node->inputs[0] = p1;
  exp_node->n_inputs = 1;

  result->value = exp(p1->value);
  result->prev = exp_node;
  result->grad = 0.0;
  result->visited = 0;
  result->exponent = 1;
  return exp_node;
}

void tanh_grad(Parameter *result)
{
  float tanh_value = tanh(result->prev->inputs[0]->value);
  result->prev->inputs[0]->grad += result->grad * (1 - tanh_value * tanh_value);
}
OperationNode *tanh_(Parameter *p1, Parameter *result)
{
  OperationNode *tanh_node = malloc(sizeof(OperationNode));
  tanh_node->_op_name = TANH;
  tanh_node->_op_type = UNARY;
  tanh_node->backward_fn = tanh_grad;
  tanh_node->inputs = malloc(sizeof(Parameter *) * 1);
  tanh_node->inputs[0] = p1;
  tanh_node->n_inputs = 1;

  result->value = tanh(p1->value);
  result->exponent = 1;
  result->grad = 0.0;
  result->visited = 0;
  result->prev = tanh_node;
  return tanh_node;
}
