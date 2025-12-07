/**
 * tensor.h
 * 
 * Definition for the tensor_t struct and its 
 * related functions.
 */

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

#define TENSOR_MAX_RANK 2

typedef float tensor_type_t;

typedef enum {
  ALLOC_NEW,
  STORE_ON_A,
  STORE_ON_B,
} tensor_opmode_t;

typedef struct {
  size_t stride;
  size_t shape;
} tensor_rank_t;

typedef struct {
  size_t rank;
  tensor_rank_t rank_data[TENSOR_MAX_RANK];
  tensor_type_t data[];
} tensor_t;

// Creates a zero-initialized tensor with the given rank and shape.
tensor_t* tensor_create(size_t rank, size_t shape[rank]);

// Deallocates the tensor and sets the given pointer to NULL.
void tensor_destroy(tensor_t** tensor);

tensor_t* tensor_contract(tensor_t* a, tensor_t* b, tensor_opmode_t mode);
tensor_t* tensor_emult(tensor_t* a, tensor_t* b, tensor_opmode_t mode);
tensor_t* tensor_eadd(tensor_t* a, tensor_t* b, tensor_opmode_t mode);
tensor_t* tensor_esub(tensor_t* a, tensor_t* b, tensor_opmode_t mode);
tensor_t* tensor_ediv(tensor_t* a, tensor_t* b, tensor_opmode_t mode);
tensor_t* tensor_eapply(tensor_t* a, tensor_type_t(*func)(tensor_type_t val), tensor_opmode_t mode);
tensor_t* tensor_transpose(tensor_t* a, tensor_opmode_t mode);
