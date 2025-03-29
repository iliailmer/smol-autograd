#include "parameter.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  Parameter p1; //= {2, NULL, NULL};
  Parameter p2; //= {2, NULL, NULL};
  init_parameter(&p1, 4);
  init_parameter(&p2, 2);
  Parameter p3 = mult(&p1, &p2);
  backward(&p3);
  display(&p3);
  display(&p1);
  return EXIT_SUCCESS;
}
