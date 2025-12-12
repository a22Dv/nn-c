/**
 * tensor.h
 *
 * BRIEF:
 * Declaration for the tensor_t struct and
 * its related functions and macro definitions.
 * 
 * NOTE:
 * Implementation relies on TNSR_MAX_RANK being 2.
 */

#pragma once

#include <math.h>
#include <stdint.h>
#include <stdlib.h>  // IWYU pragma: export

typedef uint32_t tnsr_size_t;
typedef float tnsr_type_t;

#define TNSR_MAX_RANK 2
#define TNSR_MAX_SIZE UINT32_MAX

/* -------------------------------- Accessors ------------------------------- */

#define TNSR_STRD(tensor, n) (tensor->stride[n])
#define TNSR_SHPE(tensor, n) (tensor->shape[n])
#define TNSR_DATA(tensor, i, j) (tensor->data[i * tensor->stride[0] + j * tensor->stride[1]])
#define TNSR_DSTR(tensor) (tnsr_destroy(&tensor))

/* ----------------------------------- API ---------------------------------- */

#define TNSR_MATRIX(n, m) tnsr_create(n, m)
#define TNSR_COLVEC(n) tnsr_create(n, 1)
#define TNSR_ROWVEC(m) tnsr_create(1, m)
#define TNSR_SCALAR() tnsr_create(1, 1)
#define TNSR_FROM_ARRAY(t, a) memcpy(t->data, a, sizeof(t->data))

// Generic tensor type.
typedef struct {
  tnsr_size_t shape[TNSR_MAX_RANK];
  tnsr_size_t stride[TNSR_MAX_RANK];
  tnsr_type_t data[];
} tnsr_t;

// Creates a zero-initialized tensor with the specified dimensions. NULL upon failure.
tnsr_t *tnsr_create(tnsr_size_t n, tnsr_size_t m);

// Destroys the given tensor. Passing NULL is a no-op.
void tnsr_destroy(tnsr_t **t);

// Tensor contraction.
tnsr_t *tnsr_contract(tnsr_t *restrict dst, tnsr_t *restrict a, tnsr_t *restrict b);

// Tensor element-wise addition.
tnsr_t *tnsr_eadd(tnsr_t *dst, tnsr_t *a, tnsr_t *restrict b);

// Tensor element-wise subtraction.
tnsr_t *tnsr_esub(tnsr_t *dst, tnsr_t *a, tnsr_t *restrict b);

// Tensor element-wise multiplication.
tnsr_t *tnsr_emul(tnsr_t *dst, tnsr_t *a, tnsr_t *restrict b);

// Tensor element-wise division.
tnsr_t *tnsr_ediv(tnsr_t *dst, tnsr_t *a, tnsr_t *restrict b);

// Tensor element-wise function mapping.
tnsr_t *tnsr_emap(tnsr_t *dst, tnsr_t *a, tnsr_type_t (*f)(tnsr_type_t, void*), void *restrict ctx);

// Tensor transpose.
tnsr_t *tnsr_transpose(tnsr_t *dst, tnsr_t *a);

// Sets all fields of the tensor to the specified value.
void tnsr_set(tnsr_type_t x);

// Resets the tensor's values to zero. Equivalent to `tnsr_set(0)`.
void tnsr_reset(tnsr_t *t);

// Prints the tensor to stderr.
void tnsr_dbgprint(tnsr_t *t);

