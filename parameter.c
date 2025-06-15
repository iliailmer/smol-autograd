#include "parameter.h"
#include <stdlib.h>
#include <string.h>

void init_parameter(Parameter *p, float value)
{
  p->value = value;
  p->grad = 0.0;
  p->prev = NULL;
  p->visited = 0;
  p->exponent = 1;
}

int export_to_dot(Parameter *p, FILE *f, int *global_id)
{
  if (!p)
    return -1;

  int my_id = (*global_id)++;
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
  Parameter w1[2][2], w2[2][2], b1[2], b2[2], x[2], y[2], y_hat[2];
  Parameter h1[2];
  init_parameter(&y[0], 1.0);
  init_parameter(&y[1], 4.0);
  init_parameter(&y_hat[0], 1.0);
  init_parameter(&y_hat[1], 4.0);
  init_parameter(&x[0], 1.0);
  init_parameter(&x[1], 2.0);
  for (size_t i = 0; i < 2; i++) {
    init_parameter(&b1[i], 1.0);
    init_parameter(&b2[i], 1.0);
    init_parameter(&h1[i], 1.0);
    for (size_t j = 0; j < 2; j++) {
      init_parameter(&w1[i][j], 1.0);
      init_parameter(&w2[i][j], 1.0);
    }
  }
  // forward pass
  // first layer
  for (size_t i = 0; i < 2; i++) {
    Parameter sum;
    init_parameter(&sum, 0.0); // sum = 0
    for (size_t j = 0; j < 2; j++) {
      Parameter prod;
      init_parameter(&prod, 0.0);
      mult(&w1[i][j], &x[j], &prod); // prod =  w[i,j] * x[j]
      Parameter new_sum;
      init_parameter(&new_sum, 0.0);
      add(&prod, &sum, &new_sum);
      sum = new_sum; // safe copy
    }
    Parameter biased_sum;
    add(&sum, &b1[i], &biased_sum);
    tanh_(&biased_sum, &h1[i]);
  }
  // second layer

  for (size_t i = 0; i < 2; i++) {
    Parameter sum;
    init_parameter(&sum, 0.0); // sum = 0
    for (size_t j = 0; j < 2; j++) {
      Parameter prod;
      init_parameter(&prod, 0.0);
      mult(&w2[i][j], &h1[j], &prod); // prod =  w[i,j] * x[j]
      Parameter new_sum;
      init_parameter(&new_sum, 0.0);
      add(&prod, &sum, &new_sum);
      sum = new_sum; // safe copy
    }
    add(&sum, &b2[i], &y_hat[i]); // y_hat[i] = sum + b2[i]
  }

  // calculate loss
  Parameter loss;
  init_parameter(&loss, 0.0f);
  for (size_t i = 0; i < 2; i++) {
    Parameter diff;
    sub(&y[i], &y_hat[i], &diff);
    Parameter pow_diff;
    power(&diff, 2, &pow_diff);
    Parameter new_loss;
    add(&loss, &pow_diff, &new_loss);
    loss = new_loss;
  }
  backward(&loss);
  save_graph(&loss, "graph.dot");
  return EXIT_SUCCESS;
}
