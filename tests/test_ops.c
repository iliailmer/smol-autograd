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

// out = a - b: da = +dout, db = -dout
static void test_sub_scalar() {
  Parameter *a = malloc(sizeof(Parameter));
  Parameter *b = malloc(sizeof(Parameter));
  Parameter *s = malloc(sizeof(Parameter));
  init_0d(a, "a"); init_0d(b, "b"); init_0d(s, "s");
  a->data[0] = 3.0f;
  b->data[0] = 1.0f;
  sub(a, b, s);
  s->grad[0] = 1.0f;
  s->backward(s);
  ASSERT_NEAR(s->data[0],  2.0f);
  ASSERT_NEAR(a->grad[0],  1.0f);
  ASSERT_NEAR(b->grad[0], -1.0f);
  free_parameter(a); free_parameter(b); free_parameter(s);
  PASS("sub_scalar");
}

// A = [[1,2],[3,4]]  B = [[1,1],[1,1]]
// C = A@B = [[3,3],[7,7]]
// dC = ones  =>  dA = [[2,2],[2,2]]  dB = [[4,4],[6,6]]
static void test_matmul_2x2() {
  Parameter *a = malloc(sizeof(Parameter));
  Parameter *b = malloc(sizeof(Parameter));
  Parameter *c = malloc(sizeof(Parameter));
  init_2d(a, 2, 2, "A"); init_2d(b, 2, 2, "B"); init_2d(c, 2, 2, "C");
  a->data[0]=1; a->data[1]=2; a->data[2]=3; a->data[3]=4;
  b->data[0]=1; b->data[1]=1; b->data[2]=1; b->data[3]=1;
  matmul(a, b, c);
  ASSERT_NEAR(c->data[0], 3.0f); ASSERT_NEAR(c->data[1], 3.0f);
  ASSERT_NEAR(c->data[2], 7.0f); ASSERT_NEAR(c->data[3], 7.0f);
  for (int i = 0; i < 4; i++) c->grad[i] = 1.0f;
  c->backward(c);
  // dA
  ASSERT_NEAR(a->grad[0], 2.0f); ASSERT_NEAR(a->grad[1], 2.0f);
  ASSERT_NEAR(a->grad[2], 2.0f); ASSERT_NEAR(a->grad[3], 2.0f);
  // dB
  ASSERT_NEAR(b->grad[0], 4.0f); ASSERT_NEAR(b->grad[1], 4.0f);
  ASSERT_NEAR(b->grad[2], 6.0f); ASSERT_NEAR(b->grad[3], 6.0f);
  free_parameter(a); free_parameter(b); free_parameter(c);
  PASS("matmul_2x2");
}

// 4x4 identity: A @ I = A,  dA = dC @ I^T = dC
static void test_matmul_4x4_identity() {
  Parameter *a = malloc(sizeof(Parameter));
  Parameter *id = malloc(sizeof(Parameter));
  Parameter *c  = malloc(sizeof(Parameter));
  init_2d(a,  4, 4, "A");
  init_2d(id, 4, 4, "I");
  init_2d(c,  4, 4, "C");
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) {
      a->data[i*4+j]  = (float)(i*4+j+1);
      id->data[i*4+j] = (i == j) ? 1.0f : 0.0f;
    }
  matmul(a, id, c);
  for (int i = 0; i < 16; i++) ASSERT_NEAR(c->data[i], a->data[i]);
  for (int i = 0; i < 16; i++) c->grad[i] = 1.0f;
  c->backward(c);
  // dA = dC @ I^T = ones @ I = ones
  for (int i = 0; i < 16; i++) ASSERT_NEAR(a->grad[i], 1.0f);
  free_parameter(a); free_parameter(id); free_parameter(c);
  PASS("matmul_4x4_identity");
}

int main(void) {
  test_add_scalar();
  test_mul_scalar();
  test_add_vector();
  test_mul_vector();
  test_grad_accumulates();
  test_sub_scalar();
  test_matmul_2x2();
  test_matmul_4x4_identity();
  printf("All tests passed.\n");
  return 0;
}
