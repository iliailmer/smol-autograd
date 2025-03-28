#include "parameter.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  Parameter p1 = {2, NULL, NULL};
  Parameter p2 = {2, NULL, NULL};
  Parameter p3 = add(p1, p2);
  display(p3);
  return EXIT_SUCCESS;
}
