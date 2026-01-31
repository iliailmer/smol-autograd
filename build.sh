#!/bin/bash

# Create build directory if it doesn't exist
mkdir -p build

# Build test_autograd
# gcc -g -Wall -Wextra -std=c11 -o build/test_autograd src/test_autograd.c src/parameter.c src/op.c src/tensor.c -lm
# ./build/test_autograd
# dot -Tpng build/quadratic_graph.dot -o build/quadratic_graph.png
# open build/quadratic_graph.png # macOS

# Build test_mlp
gcc -g -Wall -Wextra -std=c11 -o build/test_mlp src/test_mlp.c src/parameter.c src/op.c src/tensor.c src/adam.c -lm
./build/test_mlp
dot -Tpng build/mlp_graph.dot -o build/mlp_graph.png
open build/mlp_graph.png # macOS
