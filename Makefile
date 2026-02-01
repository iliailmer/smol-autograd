# Compiler and flags
CC = gcc
CFLAGS = -g -Wall -Wextra -std=c11
LDFLAGS = -lm

# Directories
SRC_DIR = src
BUILD_DIR = build

# Source files
CORE_SRCS = $(SRC_DIR)/parameter.c $(SRC_DIR)/op.c $(SRC_DIR)/tensor.c
ADAM_SRCS = $(CORE_SRCS) $(SRC_DIR)/adam.c

# Object files
CORE_OBJS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(CORE_SRCS))
ADAM_OBJS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(ADAM_SRCS))

# Test objects
TEST_MLP_OBJ = $(BUILD_DIR)/test_mlp.o
TEST_AUTOGRAD_OBJ = $(BUILD_DIR)/test_autograd.o

# Executables
TEST_MLP = $(BUILD_DIR)/test_mlp
TEST_AUTOGRAD = $(BUILD_DIR)/test_autograd

# Targets
TARGETS = $(TEST_MLP) $(TEST_AUTOGRAD)

# Default target
all: $(TARGETS)

# Create build directory
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Compile object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(SRC_DIR)/parameter.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Link test_mlp
$(TEST_MLP): $(TEST_MLP_OBJ) $(ADAM_OBJS)
	$(CC) $^ -o $@ $(LDFLAGS)

# Link test_autograd
$(TEST_AUTOGRAD): $(TEST_AUTOGRAD_OBJ) $(CORE_OBJS)
	$(CC) $^ -o $@ $(LDFLAGS)

# Run targets
run-mlp: $(TEST_MLP)
	./$(TEST_MLP)

run-autograd: $(TEST_AUTOGRAD)
	./$(TEST_AUTOGRAD)

# Visualization targets
viz-mlp: run-mlp
	@if [ -f $(BUILD_DIR)/mlp_graph.dot ]; then \
		dot -Tpng $(BUILD_DIR)/mlp_graph.dot -o $(BUILD_DIR)/mlp_graph.png; \
		echo "Generated $(BUILD_DIR)/mlp_graph.png"; \
		open $(BUILD_DIR)/mlp_graph.png 2>/dev/null || echo "Open $(BUILD_DIR)/mlp_graph.png to view"; \
	else \
		echo "Error: mlp_graph.dot not found"; \
	fi

viz-autograd: run-autograd
	@if [ -f $(BUILD_DIR)/quadratic_graph.dot ]; then \
		dot -Tpng $(BUILD_DIR)/quadratic_graph.dot -o $(BUILD_DIR)/quadratic_graph.png; \
		echo "Generated $(BUILD_DIR)/quadratic_graph.png"; \
		open $(BUILD_DIR)/quadratic_graph.png 2>/dev/null || echo "Open $(BUILD_DIR)/quadratic_graph.png to view"; \
	else \
		echo "Error: quadratic_graph.dot not found"; \
	fi

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

# Rebuild everything
rebuild: clean all

# Show help
help:
	@echo "Available targets:"
	@echo "  all           - Build all executables (default)"
	@echo "  run-mlp       - Build and run MLP test"
	@echo "  run-autograd  - Build and run autograd test"
	@echo "  viz-mlp       - Build, run, and visualize MLP graph"
	@echo "  viz-autograd  - Build, run, and visualize autograd graph"
	@echo "  clean         - Remove all build artifacts"
	@echo "  rebuild       - Clean and rebuild everything"
	@echo "  help          - Show this help message"

# Phony targets
.PHONY: all clean rebuild run-mlp run-autograd viz-mlp viz-autograd help
