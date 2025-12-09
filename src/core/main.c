#include <stdlib.h>
#include <stdio.h>
#include "core/tensor.h"

tensor_type_t random(tensor_type_t _) {
  return rand() / (tensor_type_t)RAND_MAX;
}

int main() {
  tensor_t* tensor_a = tensor_create(2, TENSOR_DECLARE_SHAPE(5, 2));
  tensor_t* tensor_b = tensor_create(2, TENSOR_DECLARE_SHAPE(3, 5));
  
  tensor_t* tensor_am = tensor_emap(NULL, tensor_a, random);
  tensor_t* tensor_bm = tensor_emap(NULL, tensor_b, random);
  tensor_destroy(&tensor_a);
  tensor_destroy(&tensor_b);

  tensor_t* tensor_c = tensor_contract(NULL, tensor_am, tensor_bm);
  tensor_t* scalar_a = tensor_create(0, NULL);
  scalar_a->data[0] = 50;
  tensor_t* tensor_d = tensor_eadd(NULL, tensor_c, scalar_a);
  printf("A:\n");
  dbg_tensor_print(tensor_am);
  printf("B:\n");
  dbg_tensor_print(tensor_bm);
  printf("C:\n");
  dbg_tensor_print(tensor_c);
  printf("D:\n");
  dbg_tensor_print(tensor_d);
}