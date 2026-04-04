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

clean:
	rm -rf build

.PHONY: all run clean
