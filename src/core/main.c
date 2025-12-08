#include <stdlib.h>
#include <stdio.h>

#include "core/tensor.h"
#include "utils/utils.h"

int main() {
  tensor_size_t shape_a[TENSOR_MAX_RANK] = {3, 1};
  tensor_t *tensor_a = tensor_create(2, shape_a);
  REQUIRE(tensor_a, return -1);

  tensor_size_t shape_b[TENSOR_MAX_RANK] = {3, 2};
  tensor_t *tensor_b = tensor_create(2, shape_b);
  REQUIRE(tensor_b, return -1);

  for (tensor_size_t i = 0; i < tensor_a->rank_data[1].shape; ++i) {
    for (tensor_size_t j = 0; j < tensor_a->rank_data[0].shape; ++j) {
      tensor_a->data[i * tensor_a->rank_data[1].stride + j * tensor_a->rank_data[0].stride] =
          (tensor_type_t)(rand() % 10);
    }
  }
  for (tensor_size_t i = 0; i < tensor_b->rank_data[1].shape; ++i) {
    for (tensor_size_t j = 0; j < tensor_b->rank_data[0].shape; ++j) {
      tensor_b->data[i * tensor_b->rank_data[1].stride + j * tensor_b->rank_data[0].stride] =
          (tensor_type_t)(rand() % 10);
    }
  }
  printf("A\n");
  dbg_tensor_print(tensor_a);

  tensor_transpose_inplace(tensor_a);
  dbg_tensor_print(tensor_a);

}