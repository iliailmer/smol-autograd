#include "../src/autograd.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define TOL 1e-5f

#define ASSERT_NEAR(got, expected)                                             \
  do {                                                                         \
    if (fabsf((got) - (expected)) > TOL) {                                     \
      fprintf(stderr, "FAIL %s:%d: expected %.6f, got %.6f\n", __FILE__,      \
              __LINE__, (float)(expected), (float)(got));                       \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

#define PASS(name) printf("PASS %s\n", name)

static void test_add_scalar() {
  Parameter *a = malloc(sizeof(Parameter));
  Parameter *b = malloc(sizeof(Parameter));
  Parameter *s = malloc(sizeof(Parameter));
  init_0d(a, "a"); init_0d(b, "b"); init_0d(s, "s");
  a->data[0] = 0.5f;
  b->data[0] = 1.5f;
  add(a, b, s);
  s->grad[0] = 1.0f;
  s->backward(s);
  ASSERT_NEAR(s->data[0], 2.0f);
  ASSERT_NEAR(a->grad[0], 1.0f);
  ASSERT_NEAR(b->grad[0], 1.0f);
  free_parameter(a); free_parameter(b); free_parameter(s);
  PASS("add_scalar");
}

static void test_mul_scalar() {
  Parameter *a = malloc(sizeof(Parameter));
  Parameter *b = malloc(sizeof(Parameter));
  Parameter *c = malloc(sizeof(Parameter));
  init_0d(a, "a"); init_0d(b, "b"); init_0d(c, "c");
  a->data[0] = 0.5f;
  b->data[0] = 1.5f;
  mul(a, b, c);
  c->grad[0] = 1.0f;
  c->backward(c);
  ASSERT_NEAR(c->data[0], 0.75f);
  ASSERT_NEAR(a->grad[0], 1.5f); // dc/da = b
  ASSERT_NEAR(b->grad[0], 0.5f); // dc/db = a
  free_parameter(a); free_parameter(b); free_parameter(c);
  PASS("mul_scalar");
}

static void test_add_vector() {
  int n = 4;
  Parameter *a = malloc(sizeof(Parameter));
  Parameter *b = malloc(sizeof(Parameter));
  Parameter *s = malloc(sizeof(Parameter));
  init_1d(a, n, "a"); init_1d(b, n, "b"); init_1d(s, n, "s");
  for (int i = 0; i < n; i++) { a->data[i] = 0.5f; b->data[i] = 1.5f; }
  add(a, b, s);
  for (int i = 0; i < n; i++) s->grad[i] = 1.0f;
  s->backward(s);
  for (int i = 0; i < n; i++) {
    ASSERT_NEAR(s->data[i], 2.0f);
    ASSERT_NEAR(a->grad[i], 1.0f);
    ASSERT_NEAR(b->grad[i], 1.0f);
  }
  free_parameter(a); free_parameter(b); free_parameter(s);
  PASS("add_vector");
}

static void test_mul_vector() {
  int n = 4;
  Parameter *a = malloc(sizeof(Parameter));
  Parameter *b = malloc(sizeof(Parameter));
  Parameter *c = malloc(sizeof(Parameter));
  init_1d(a, n, "a"); init_1d(b, n, "b"); init_1d(c, n, "c");
  for (int i = 0; i < n; i++) { a->data[i] = 0.5f; b->data[i] = 1.5f; }
  mul(a, b, c);
  for (int i = 0; i < n; i++) c->grad[i] = 1.0f;
  c->backward(c);
  for (int i = 0; i < n; i++) {
    ASSERT_NEAR(c->data[i], 0.75f);
    ASSERT_NEAR(a->grad[i], 1.5f);
    ASSERT_NEAR(b->grad[i], 0.5f);
  }
  free_parameter(a); free_parameter(b); free_parameter(c);
  PASS("mul_vector");
}

static void test_grad_accumulates() {
  // f(a) = a * a: df/da = 2a, but since a is shared, grad accumulates twice
  Parameter *a = malloc(sizeof(Parameter));
  Parameter *c = malloc(sizeof(Parameter));
  init_0d(a, "a"); init_0d(c, "c");
  a->data[0] = 3.0f;
  mul(a, a, c);
  c->grad[0] = 1.0f;
  c->backward(c);
  ASSERT_NEAR(c->data[0], 9.0f);
  ASSERT_NEAR(a->grad[0], 6.0f); // 2 * a
  free_parameter(a); free_parameter(c);
  PASS("grad_accumulates (a*a)");
}

int main(void) {
  test_add_scalar();
  test_mul_scalar();
  test_add_vector();
  test_mul_vector();
  test_grad_accumulates();
  printf("All tests passed.\n");
  return 0;
}
