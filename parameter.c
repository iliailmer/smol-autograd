#include "parameter.h"
#include <stdlib.h>
#include <string.h>

static void _free_parameter_graph_recursive(Parameter *p, int cleanup_id);

void init_parameter(Parameter *p, float value)
{
  p->value = value;
  p->grad = 0.0;
  p->prev = NULL;
  p->visited = 0;
  p->cleanup_visited = 0;
  p->export_visited = 0;
  p->exponent = 1;
}

void free_operation_node(OperationNode *node)
{
  if (!node)
    return;

  if (node->inputs) {
    free(node->inputs);
  }
  free(node);
}

void free_parameter_graph(Parameter *p)
{
  if (!p)
    return;

  // Use a separate visited flag for cleanup to avoid interfering with
  // computation
  static int cleanup_id = 0;
  cleanup_id++;

  _free_parameter_graph_recursive(p, cleanup_id);
}

static void _free_parameter_graph_recursive(Parameter *p, int cleanup_id)
{
  if (!p || p->cleanup_visited == cleanup_id)
    return;

  p->cleanup_visited = cleanup_id;

  if (p->prev) {
    for (size_t i = 0; i < p->prev->n_inputs; i++) {
      _free_parameter_graph_recursive(p->prev->inputs[i], cleanup_id);
    }
    free_operation_node(p->prev);
    p->prev = NULL;
  }
}

void reset_export_visited(Parameter *p)
{
  if (!p || p->export_visited == 0) return;
  
  p->export_visited = 0;
  
  if (p->prev) {
    for (size_t i = 0; i < p->prev->n_inputs; i++) {
      reset_export_visited(p->prev->inputs[i]);
    }
  }
}

int export_to_dot(Parameter *p, FILE *f, int *global_id)
{
  if (!p)
    return -1;

  // Check if this parameter was already exported
  if (p->export_visited > 0) {
    return p->export_visited; // Return the previously assigned ID
  }

  int my_id = (*global_id)++;
  p->export_visited = my_id; // Mark as visited with its ID
  
  fprintf(f, "  node%d [label=\"value=%.2f, grad=%.2f\"];\n", my_id, p->value,
          p->grad);

  if (p->prev) {
    int op_id = (*global_id)++;
    const char *label = "?";
    switch (p->prev->_op_name) {
    case ADD:
      label = "+";
      break;
    case SUB:
      label = "-";
      break;
    case NEG:
      label = "-1";
      break;
    case MUL:
      label = "*";
      break;
    case DIV:
      label = "/";
      break;
    case POW:
      label = "pow";
      break;
    case TANH:
      label = "tanh";
      break;
    case RELU:
      label = "relu";
      break;
    case EXP:
      label = "exp";
      break;
    }

    fprintf(f, "  op%d [label=\"%s\", shape=ellipse];\n", op_id, label);
    fprintf(f, "  op%d -> node%d;\n", op_id, my_id);

    for (size_t i = 0; i < p->prev->n_inputs; i++) {
      int input_id = export_to_dot(p->prev->inputs[i], f, global_id);
      if (input_id != -1)
        fprintf(f, "  node%d -> op%d;\n", input_id, op_id);
    }
  }

  return my_id;
}

void topo_sort(Parameter *p, Parameter **sorted, int *index)
{
  if (!p || p->visited)
    return;

  p->visited = 1;

  if (p->prev) {
    for (size_t i = 0; i < p->prev->n_inputs; i++) {
      topo_sort(p->prev->inputs[i], sorted, index);
    }
  }

  sorted[(*index)] = p;
  (*index)++;
}

adam_error_t backward(Parameter *p)
{
  if (!p)
    return ADAM_ERROR_NULL_POINTER;

  Parameter *sorted[MAX_GRAPH_SIZE];
  int idx = 0;
  topo_sort(p, sorted, &idx);

  if (idx >= MAX_GRAPH_SIZE) {
    return ADAM_ERROR_GRAPH_TOO_LARGE;
  }

  p->grad = 1.0; // dp/dp

  // traverse in reverse
  for (int i = idx - 1; i >= 0; i--) {
    if (sorted[i]->prev &&
        sorted[i]->prev->backward_fn) { // if there was an operation perfomed
      sorted[i]->prev->backward_fn(sorted[i]);
    }
  }

  return ADAM_SUCCESS;
}

adam_error_t save_graph(Parameter *p, const char *filename)
{
  if (!p || !filename)
    return ADAM_ERROR_NULL_POINTER;

  FILE *f = fopen(filename, "w");
  if (!f)
    return ADAM_ERROR_INVALID_INPUT;

  fprintf(f, "digraph G {\n  rankdir=BT;\n");

  // Reset export visited flags to ensure clean export
  reset_export_visited(p);

  int global_id = 0;
  export_to_dot(p, f, &global_id);

  fprintf(f, "}\n");
  fclose(f);

  return ADAM_SUCCESS;
}
