#include "parameter.h"
#include <stdlib.h>
#include <string.h>

#define MAX_QUEUE 100

void init_parameter(Parameter *p, float value)
{
  p->value = value;
  p->grad = 0.0;
  p->prev = NULL;
  p->visited = 0;
  p->exponent = 1;
}

void export_to_dot(Parameter *p, FILE *f, int *global_id)
{
  if (!p)
    return;

  int my_id = *global_id;
  (*global_id)++;
  fprintf(f, "  node%d [label=\"value=%.2f, grad=%.2f\"];\n", my_id, p->value,
          p->grad);

  if (p->prev) {
    int op_id = (*global_id);
    (*global_id)++;
    const char *label = "?";
    switch (p->prev->_op_name) {
    case ADD:
      label = "+";
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
    case EXP:
      label = "exp";
      break;
    }

    fprintf(f, "  op%d [label=\"%s\", shape=ellipse];\n", op_id, label);
    fprintf(f, "  op%d -> node%d;\n", op_id, my_id);

    for (size_t i = 0; i < p->prev->n_inputs; i++) {
      int input_id = *global_id;
      export_to_dot(p->prev->inputs[i], f, global_id);
      fprintf(f, "  node%d -> op%d;\n", input_id, op_id);
    }
  }
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

void backward(Parameter *p)
{
  Parameter *sorted[MAX_GRAPH_SIZE];
  int idx = 0;
  topo_sort(p, sorted, &idx);
  p->grad = 1.0; // dp/dp
  printf("idx=%d\n", idx);
  // traverse in reverse
  for (int i = idx - 1; i >= 0; i--) {
    if (sorted[i]->prev &&
        sorted[i]->prev->backward_fn) { // if there was an operation perfomed
      sorted[i]->prev->backward_fn(sorted[i]);
    }
  }
}

void save_graph(Parameter *p, const char *filename)
{
  FILE *f = fopen(filename, "w");
  fprintf(f, "digraph G {\n  rankdir=BT;\n");

  int global_id = 0;
  export_to_dot(p, f, &global_id);

  fprintf(f, "}\n");
  fclose(f);
}

int main(int argc, char *argv[])
{
  // (a+b)*c^2/d;
  Parameter w11, w12, w21, w22, b1, b2, x1, x2, y1, y2, y_hat1, y_hat2, result;
  init_parameter(&w11, 1.0);
  init_parameter(&w12, 1.0);
  init_parameter(&w21, 1.0);
  init_parameter(&w22, 1.0);
  init_parameter(&b1, 1.0);
  init_parameter(&b2, 1.0);
  init_parameter(&x1, 1.0);
  init_parameter(&x2, 1.0);
  init_parameter(&y1, 0.0);
  init_parameter(&y2, 1.0);

  OperationNode *prod = mult(&w11, &x1, &result);
  backward(&result);
  return EXIT_SUCCESS;
}
