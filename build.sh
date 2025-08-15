gcc -g -Wall -Wextra -std=c99 -o test_autograd test_autograd.c parameter.c op.c tensor.c -lm

./test_autograd
dot -Tpng quadratic_graph.dot -o quadratic_graph.png
open quadratic_graph.png # macOS
