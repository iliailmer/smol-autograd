CC = gcc
CFLAGS = -g -Wall -Wextra -std=c11
LDFLAGS = -lm

SRCS = main.c src/autograd.c
TARGET = build/main

all: $(TARGET)

$(TARGET): $(SRCS) src/autograd.h
	@mkdir -p build
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

build/test_ops: tests/test_ops.c src/autograd.c src/autograd.h
	@mkdir -p build
	$(CC) $(CFLAGS) tests/test_ops.c src/autograd.c -o build/test_ops $(LDFLAGS)

test: build/test_ops
	./build/test_ops

clean:
	rm -rf build

.PHONY: all run test clean
