/**
 * tensor.h
 *
 * Declaration for the tensor_t struct and its
 * related functions.
 * 
 * NOTE: Implementation only supports TENSOR_MAX_RANK = 2.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#define TENSOR_MAX_RANK 2
#define TENSOR_MAX_SIZE UINT32_MAX

typedef uint32_t tensor_size_t;
typedef float tensor_type_t;

typedef struct {
  tensor_size_t stride;
  tensor_size_t shape;
} tensor_rank_data_t;

typedef struct {
  tensor_size_t rank;
  tensor_rank_data_t rank_data[TENSOR_MAX_RANK];
  tensor_type_t data[];
} tensor_t;

tensor_t *tensor_create(tensor_size_t rank, const tensor_size_t shape[rank]);
void tensor_destroy(tensor_t **tensor);

tensor_t *tensor_contract(tensor_t *a, tensor_t *b);
tensor_t *tensor_emul(tensor_t *a, tensor_t *b);
tensor_t *tensor_eadd(tensor_t *a, tensor_t *b);
tensor_t *tensor_esub(tensor_t *a, tensor_t *b);
tensor_t *tensor_ediv(tensor_t *a, tensor_t *b);
tensor_t *tensor_emap(tensor_t *a, tensor_type_t (*f)(tensor_type_t));
tensor_t *tensor_transpose(tensor_t *a);
void tensor_transpose_inplace(tensor_t *a); 

tensor_type_t sigmoid(tensor_type_t x);
tensor_type_t relu(tensor_type_t x);
tensor_type_t leaky_relu(tensor_type_t x);

tensor_type_t sigmoid_dx(tensor_type_t x);
tensor_type_t relu_dx(tensor_type_t x);
tensor_type_t leaky_relu_dx(tensor_type_t x);

void dbg_tensor_print(tensor_t* tensor);