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

#define TENSOR_DECLARE_SHAPE(a, b)   \
  (tensor_size_t[TENSOR_MAX_RANK]) { \
    a, b                             \
  }

#define TENSOR_STRIDE(tensor, dimension) (tensor->rank_data[dimension].stride)
#define TENSOR_SHAPE(tensor, dimension) (tensor->rank_data[dimension].shape)

typedef uint32_t tensor_size_t;
typedef float tensor_type_t;

// Tensor rank data.
typedef struct {
  tensor_size_t stride;
  tensor_size_t shape;
} tensor_rank_data_t;

// Tensor object.
typedef struct {
  tensor_size_t rank;
  tensor_rank_data_t rank_data[TENSOR_MAX_RANK];
  tensor_type_t data[];
} tensor_t;

// Creates a tensor with the specified characteristics.
tensor_t *tensor_create(tensor_size_t rank, const tensor_size_t shape[rank]);

// Destroys a tensor and sets its pointer to NULL.
void tensor_destroy(tensor_t **tensor);

// Tensor contraction. Only supports up to TENSOR_MAX_RANK.
tensor_t *tensor_contract(
    tensor_t *restrict dst, const tensor_t *restrict a, const tensor_t *restrict b
);

// Tensor element-wise multiplication.
tensor_t *tensor_emul(tensor_t *dst, const tensor_t *a, const tensor_t *restrict b);

// Tensor element-wise addition.
tensor_t *tensor_eadd(tensor_t *dst, const tensor_t *a, const tensor_t *restrict b);

// Tensor element-wise subtraction.
tensor_t *tensor_esub(tensor_t *dst, const tensor_t *a, const tensor_t *restrict b);

// Tensor element-wise division.
tensor_t *tensor_ediv(tensor_t *dst, const tensor_t *a, const tensor_t *restrict b);

// Tensor element-wise function mapping.
tensor_t *tensor_emap(tensor_t *dst, const tensor_t *a, tensor_type_t (*f)(tensor_type_t));

// Swaps metadata if a == dst. If not, result is stored on dst if dst != NULL.
tensor_t *tensor_transpose(tensor_t *dst, tensor_t *a);

// Calculates the mean-squared error from a given output and expected output. 
// This returns a scalar value.
tensor_t *tensor_mse(tensor_t *restrict dst, const tensor_t *restrict output, const tensor_t *restrict expected);

tensor_type_t sigmoid(tensor_type_t x);
tensor_type_t relu(tensor_type_t x);
tensor_type_t leaky_relu(tensor_type_t x);

// Sigmoid derivative.
tensor_type_t sigmoid_dx(tensor_type_t x);

// ReLU derivative.
tensor_type_t relu_dx(tensor_type_t x);

// Leaky ReLU derivative.
tensor_type_t leaky_relu_dx(tensor_type_t x);

// Always returns 0.
tensor_type_t zeroes(tensor_type_t x);

void dbg_tensor_print(tensor_t *tensor);