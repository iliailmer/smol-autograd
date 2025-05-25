#include "parameter.h"

void add_grad(Parameter *result)
{
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
  return add_node;
}

void mult_grad(Parameter *result)
{
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
  return mult_node;
}
