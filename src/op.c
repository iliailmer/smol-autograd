#include "parameter.h"
#include <math.h>

void add_grad(Parameter *result)
{
  if (!result || !result->prev || !result->prev->inputs[0] ||
      !result->prev->inputs[1]) {
    return;
  }

  result->prev->inputs[0]->grad += result->grad;
  result->prev->inputs[1]->grad += result->grad;
}

adam_error_t add(Parameter *p1, Parameter *p2, Parameter *result)
{
  if (!p1 || !p2 || !result) {
    return ADAM_ERROR_NULL_POINTER;
  }

  OperationNode *add_node = malloc(sizeof(OperationNode));
  if (!add_node) {
    return ADAM_ERROR_MEMORY_ALLOCATION;
  }

  add_node->inputs = malloc(sizeof(Parameter *) * 2);
  if (!add_node->inputs) {
    free(add_node);
    return ADAM_ERROR_MEMORY_ALLOCATION;
  }

  add_node->_op_name = ADD;
  add_node->_op_type = BINARY;
  add_node->inputs[0] = p1;
  add_node->inputs[1] = p2;
  add_node->n_inputs = 2;
  add_node->backward_fn = add_grad;

  result->value = p1->value + p2->value;
  result->prev = add_node;
  result->grad = 0.0;
  result->visited = 0;
  result->exponent = 1;

  return ADAM_SUCCESS;
}

void sub_grad(Parameter *result)
{
  if (!result || !result->prev || !result->prev->inputs[0] ||
      !result->prev->inputs[1]) {
    return;
  }

  result->prev->inputs[0]->grad += result->grad;
  result->prev->inputs[1]->grad -= result->grad;
}

adam_error_t sub(Parameter *p1, Parameter *p2, Parameter *result)
{
  if (!p1 || !p2 || !result) {
    return ADAM_ERROR_NULL_POINTER;
  }

  OperationNode *sub_node = malloc(sizeof(OperationNode));
  if (!sub_node) {
    return ADAM_ERROR_MEMORY_ALLOCATION;
  }

  sub_node->inputs = malloc(sizeof(Parameter *) * 2);
  if (!sub_node->inputs) {
    free(sub_node);
    return ADAM_ERROR_MEMORY_ALLOCATION;
  }

  sub_node->_op_name = SUB;
  sub_node->_op_type = BINARY;
  sub_node->inputs[0] = p1;
  sub_node->inputs[1] = p2;
  sub_node->n_inputs = 2;
  sub_node->backward_fn = sub_grad;

  result->value = p1->value - p2->value;
  result->prev = sub_node;
  result->grad = 0.0;
  result->visited = 0;
  result->exponent = 1;

  return ADAM_SUCCESS;
}

void neg_grad(Parameter *result)
{
  if (!result || !result->prev || !result->prev->inputs[0]) {
    return;
  }

  result->prev->inputs[0]->grad -= result->grad;
}

adam_error_t neg(Parameter *p1, Parameter *result)
{
  if (!p1 || !result) {
    return ADAM_ERROR_NULL_POINTER;
  }

  OperationNode *neg_node = malloc(sizeof(OperationNode));
  if (!neg_node) {
    return ADAM_ERROR_MEMORY_ALLOCATION;
  }

  neg_node->inputs = malloc(sizeof(Parameter *) * 1);
  if (!neg_node->inputs) {
    free(neg_node);
    return ADAM_ERROR_MEMORY_ALLOCATION;
  }

  neg_node->_op_name = NEG;
  neg_node->_op_type = UNARY;
  neg_node->inputs[0] = p1;
  neg_node->n_inputs = 1;
  neg_node->backward_fn = neg_grad;

  result->value = (-1) * p1->value;
  result->prev = neg_node;
  result->grad = 0.0;
  result->visited = 0;
  result->exponent = 1;

  return ADAM_SUCCESS;
}

void mult_grad(Parameter *result)
{
  if (!result || !result->prev || !result->prev->inputs[0] ||
      !result->prev->inputs[1]) {
    return;
  }

  result->prev->inputs[0]->grad +=
      result->grad * result->prev->inputs[1]->value;
  result->prev->inputs[1]->grad +=
      result->grad * result->prev->inputs[0]->value;
}

adam_error_t mult(Parameter *p1, Parameter *p2, Parameter *result)
{
  if (!p1 || !p2 || !result) {
    return ADAM_ERROR_NULL_POINTER;
  }

  OperationNode *mult_node = malloc(sizeof(OperationNode));
  if (!mult_node) {
    return ADAM_ERROR_MEMORY_ALLOCATION;
  }

  mult_node->inputs = malloc(sizeof(Parameter *) * 2);
  if (!mult_node->inputs) {
    free(mult_node);
    return ADAM_ERROR_MEMORY_ALLOCATION;
  }

  mult_node->_op_name = MUL;
  mult_node->_op_type = BINARY;
  mult_node->inputs[0] = p1;
  mult_node->inputs[1] = p2;
  mult_node->n_inputs = 2;
  mult_node->backward_fn = mult_grad;

  result->value = p1->value * p2->value;
  result->prev = mult_node;
  result->grad = 0.0;
  result->visited = 0;
  result->exponent = 1;

  return ADAM_SUCCESS;
}
void divide_grad(Parameter *result)
{
  if (!result || !result->prev || !result->prev->inputs[0] ||
      !result->prev->inputs[1]) {
    return;
  }

  float denom = result->prev->inputs[1]->value;
  if (denom == 0.0f) {
    return; // Avoid division by zero in gradient
  }

  // numerator is idx 0
  result->prev->inputs[0]->grad += result->grad / denom;
  // denominator is idx 1
  result->prev->inputs[1]->grad +=
      result->grad * result->prev->inputs[0]->value / (denom * denom);
}

adam_error_t divide(Parameter *p1, Parameter *p2, Parameter *result)
{
  if (!p1 || !p2 || !result) {
    return ADAM_ERROR_NULL_POINTER;
  }

  if (p2->value == 0.0f) {
    return ADAM_ERROR_DIVISION_BY_ZERO;
  }

  OperationNode *div_node = malloc(sizeof(OperationNode));
  if (!div_node) {
    return ADAM_ERROR_MEMORY_ALLOCATION;
  }

  div_node->inputs = malloc(sizeof(Parameter *) * 2);
  if (!div_node->inputs) {
    free(div_node);
    return ADAM_ERROR_MEMORY_ALLOCATION;
  }

  div_node->_op_name = DIV;
  div_node->_op_type = BINARY;
  div_node->inputs[0] = p1;
  div_node->inputs[1] = p2;
  div_node->n_inputs = 2;
  div_node->backward_fn = divide_grad;

  result->value = p1->value / p2->value;
  result->prev = div_node;
  result->grad = 0.0;
  result->visited = 0;
  result->exponent = 1;

  return ADAM_SUCCESS;
}

void pow_grad(Parameter *result)
{
  if (!result || !result->prev || !result->prev->inputs[0]) {
    return;
  }

  int exponent = result->exponent;
  if (exponent == 0) {
    return; // derivative of constant is 0
  }

  result->prev->inputs[0]->grad +=
      result->grad * exponent *
      pow(result->prev->inputs[0]->value, exponent - 1);
}

adam_error_t power(Parameter *p1, int exponent, Parameter *result)
{
  if (!p1 || !result) {
    return ADAM_ERROR_NULL_POINTER;
  }

  // Check for invalid power operations
  if (p1->value < 0 && exponent != (int)exponent) {
    return ADAM_ERROR_INVALID_INPUT;
  }

  OperationNode *pow_node = malloc(sizeof(OperationNode));
  if (!pow_node) {
    return ADAM_ERROR_MEMORY_ALLOCATION;
  }

  pow_node->inputs = malloc(sizeof(Parameter *) * 1);
  if (!pow_node->inputs) {
    free(pow_node);
    return ADAM_ERROR_MEMORY_ALLOCATION;
  }

  pow_node->_op_name = POW;
  pow_node->_op_type = UNARY;
  pow_node->inputs[0] = p1;
  pow_node->n_inputs = 1;
  pow_node->backward_fn = pow_grad;

  result->value = pow(p1->value, exponent);
  result->prev = pow_node;
  result->grad = 0.0;
  result->visited = 0;
  result->exponent = exponent;

  return ADAM_SUCCESS;
}

void exp_grad(Parameter *result)
{
  if (!result || !result->prev || !result->prev->inputs[0]) {
    return;
  }

  result->prev->inputs[0]->grad +=
      exp(result->prev->inputs[0]->value) * result->grad;
}

adam_error_t exp_(Parameter *p1, Parameter *result)
{
  if (!p1 || !result) {
    return ADAM_ERROR_NULL_POINTER;
  }

  OperationNode *exp_node = malloc(sizeof(OperationNode));
  if (!exp_node) {
    return ADAM_ERROR_MEMORY_ALLOCATION;
  }

  exp_node->inputs = malloc(sizeof(Parameter *) * 1);
  if (!exp_node->inputs) {
    free(exp_node);
    return ADAM_ERROR_MEMORY_ALLOCATION;
  }

  exp_node->_op_name = EXP;
  exp_node->_op_type = UNARY;
  exp_node->backward_fn = exp_grad;
  exp_node->inputs[0] = p1;
  exp_node->n_inputs = 1;

  result->value = exp(p1->value);
  result->prev = exp_node;
  result->grad = 0.0;
  result->visited = 0;
  result->exponent = 1;

  return ADAM_SUCCESS;
}

void tanh_grad(Parameter *result)
{
  if (!result || !result->prev || !result->prev->inputs[0]) {
    return;
  }

  float tanh_value = tanh(result->prev->inputs[0]->value);
  result->prev->inputs[0]->grad += result->grad * (1 - tanh_value * tanh_value);
}

adam_error_t tanh_(Parameter *p1, Parameter *result)
{
  if (!p1 || !result) {
    return ADAM_ERROR_NULL_POINTER;
  }

  OperationNode *tanh_node = malloc(sizeof(OperationNode));
  if (!tanh_node) {
    return ADAM_ERROR_MEMORY_ALLOCATION;
  }

  tanh_node->inputs = malloc(sizeof(Parameter *) * 1);
  if (!tanh_node->inputs) {
    free(tanh_node);
    return ADAM_ERROR_MEMORY_ALLOCATION;
  }

  tanh_node->_op_name = TANH;
  tanh_node->_op_type = UNARY;
  tanh_node->backward_fn = tanh_grad;
  tanh_node->inputs[0] = p1;
  tanh_node->n_inputs = 1;

  result->value = tanh(p1->value);
  result->exponent = 1;
  result->grad = 0.0;
  result->visited = 0;
  result->prev = tanh_node;

  return ADAM_SUCCESS;
}

void relu_grad(Parameter *result)
{
  if (!result || !result->prev || !result->prev->inputs[0]) {
    return;
  }

  float val = result->prev->inputs[0]->value;
  float relu_deriv = val > 0.0 ? 1.0 : 0.0;
  result->prev->inputs[0]->grad += result->grad * (relu_deriv);
}

adam_error_t relu_(Parameter *p1, Parameter *result)
{
  if (!p1 || !result) {
    return ADAM_ERROR_NULL_POINTER;
  }

  OperationNode *relu_node = malloc(sizeof(OperationNode));
  if (!relu_node) {
    return ADAM_ERROR_MEMORY_ALLOCATION;
  }

  relu_node->inputs = malloc(sizeof(Parameter *) * 1);
  if (!relu_node->inputs) {
    free(relu_node);
    return ADAM_ERROR_MEMORY_ALLOCATION;
  }

  relu_node->_op_name = RELU;
  relu_node->_op_type = UNARY;
  relu_node->backward_fn = relu_grad;
  relu_node->inputs[0] = p1;
  relu_node->n_inputs = 1;

  result->value = (p1->value > 0) ? p1->value : 0.0;
  result->exponent = 1;
  result->grad = 0.0;
  result->visited = 0;
  result->prev = relu_node;

  return ADAM_SUCCESS;
}

void sigmoid_grad(Parameter *result)
{
  if (!result || !result->prev || !result->prev->inputs[0]) {
    return;
  }

  float sigmoid_val = result->value;
  float sigmoid_deriv = sigmoid_val * (1.0f - sigmoid_val);
  result->prev->inputs[0]->grad += result->grad * sigmoid_deriv;
}

adam_error_t sigmoid_(Parameter *p1, Parameter *result)
{
  if (!p1 || !result) {
    return ADAM_ERROR_NULL_POINTER;
  }

  OperationNode *sigmoid_node = malloc(sizeof(OperationNode));
  if (!sigmoid_node) {
    return ADAM_ERROR_MEMORY_ALLOCATION;
  }

  sigmoid_node->inputs = malloc(sizeof(Parameter *) * 1);
  if (!sigmoid_node->inputs) {
    free(sigmoid_node);
    return ADAM_ERROR_MEMORY_ALLOCATION;
  }

  sigmoid_node->_op_name = SIGMOID;
  sigmoid_node->_op_type = UNARY;
  sigmoid_node->backward_fn = sigmoid_grad;
  sigmoid_node->inputs[0] = p1;
  sigmoid_node->n_inputs = 1;

  result->value = 1.0f / (1.0f + expf(-p1->value));
  result->exponent = 1;
  result->grad = 0.0f;
  result->visited = 0;
  result->prev = sigmoid_node;

  return ADAM_SUCCESS;
}
