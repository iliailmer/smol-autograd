#include "parameter.h"
#include <stdlib.h>

void init_tensor(Tensor *t, size_t rows, size_t cols) 
{
  if (!t || rows == 0 || cols == 0) {
    return;
  }
  
  t->rows = rows;
  t->cols = cols;
  t->data = malloc(sizeof(Parameter) * rows * cols);
  
  if (t->data) {
    // Initialize all parameters to zero
    for (size_t i = 0; i < rows * cols; i++) {
      init_parameter(&t->data[i], 0.0f);
    }
  }
}

void free_tensor(Tensor *t)
{
  if (!t || !t->data) {
    return;
  }
  
  // Only free the tensor data, not the computation graphs
  // User should call free_parameter_graph() explicitly if needed
  free(t->data);
  t->data = NULL;
  t->rows = 0;
  t->cols = 0;
}

adam_error_t matmul(Tensor *t1, Tensor *t2, Tensor *result)
{
  if (!t1 || !t2 || !result || !t1->data || !t2->data) {
    return ADAM_ERROR_NULL_POINTER;
  }
  
  // Check dimensions for matrix multiplication
  if (t1->cols != t2->rows) {
    return ADAM_ERROR_INVALID_INPUT;
  }
  
  // Initialize result tensor
  init_tensor(result, t1->rows, t2->cols);
  if (!result->data) {
    return ADAM_ERROR_MEMORY_ALLOCATION;
  }
  
  // Perform matrix multiplication
  for (size_t i = 0; i < t1->rows; i++) {
    for (size_t j = 0; j < t2->cols; j++) {
      Parameter *sum = &T2D(result, i, j);
      init_parameter(sum, 0.0f);
      
      for (size_t k = 0; k < t1->cols; k++) {
        Parameter prod;
        init_parameter(&prod, 0.0f);
        
        adam_error_t err = mult(&T2D(t1, i, k), &T2D(t2, k, j), &prod);
        if (err != ADAM_SUCCESS) {
          free_tensor(result);
          return err;
        }
        
        Parameter new_sum;
        init_parameter(&new_sum, 0.0f);
        err = add(sum, &prod, &new_sum);
        if (err != ADAM_SUCCESS) {
          free_tensor(result);
          return err;
        }
        
        *sum = new_sum;
      }
    }
  }
  
  return ADAM_SUCCESS;
}
