/**
 * tensor.c
 *
 * Implementation for tensor.h.
 *
 * NOTE:
 *
 * Implementations rely on the fact that TENSOR_MAX_RANK = 2.
 * Higher dimensional tensors are not supported.
 */

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/tensor.h"
#include "utils/utils.h"

typedef enum {
  TENSOR_BROADCAST_INCOMPATIBLE,
  TENSOR_BROADCAST_A_TO_B,
  TENSOR_BROADCAST_B_TO_A,
  TENSOR_BROADCAST_COMPATIBLE,
  TENSOR_CONTRACT_COMPATIBLE,
  TENSOR_CONTRACT_INCOMPATIBLE
} tensor_compatible_t;

tensor_compatible_t tensor_broadcast_compatible(const tensor_t *a, const tensor_t *b) {
  REQUIRE(a && b, return TENSOR_BROADCAST_INCOMPATIBLE);
  tensor_compatible_t cmpt = TENSOR_BROADCAST_INCOMPATIBLE;

  const tensor_rank_data_t *a_rank = a->rank_data;
  const tensor_rank_data_t *b_rank = b->rank_data;
  const tensor_rank_data_t *broadcast_source = NULL;
  const tensor_rank_data_t *broadcast_target = NULL;
  const tensor_size_t min_rank = min(a->rank, b->rank);
  const tensor_size_t max_rank = max(a->rank, b->rank);

  if (a->rank < b->rank) {
    broadcast_source = a_rank;
    broadcast_target = b_rank;
    cmpt = TENSOR_BROADCAST_A_TO_B;
  } else if (a->rank > b->rank) {
    broadcast_source = b_rank;
    broadcast_target = a_rank;
    cmpt = TENSOR_BROADCAST_B_TO_A;
  } else {
    broadcast_source = a_rank;
    broadcast_target = b_rank;
    cmpt = TENSOR_BROADCAST_COMPATIBLE;
  }

  for (tensor_size_t i = 0; i < min_rank; ++i) {
    const tensor_size_t trailing_target = max_rank - i - 1;
    const tensor_size_t trailing_source = min_rank - i - 1;
    bool equal = broadcast_source[trailing_source].shape == broadcast_target[trailing_target].shape;
    bool has_one = broadcast_source[trailing_source].shape == 1 ||
                   broadcast_target[trailing_target].shape == 1;
    if (!equal && !has_one) {
      cmpt = TENSOR_BROADCAST_INCOMPATIBLE;
      break;
    }
    // Handles edge-case where both ranks are the same but one of them has a 1.
    if (has_one && !equal && cmpt == TENSOR_BROADCAST_COMPATIBLE) {
      cmpt = broadcast_source[trailing_source].shape == 1 ? TENSOR_BROADCAST_A_TO_B
                                                          : TENSOR_BROADCAST_B_TO_A;
    }
  }
  return cmpt;
}

tensor_compatible_t tensor_contract_compatible(const tensor_t *a, const tensor_t *b) {
  REQUIRE(a && b, return TENSOR_CONTRACT_INCOMPATIBLE);
  const tensor_rank_data_t *min_rank_data = NULL;
  const tensor_rank_data_t *max_rank_data = NULL;
  tensor_size_t min_rank = 0;
  tensor_size_t max_rank = 0;
  bool cmpt = true;

  if (a->rank <= b->rank) {
    min_rank_data = a->rank_data;
    max_rank_data = b->rank_data;
    min_rank = a->rank;
    max_rank = b->rank;
  } else {
    min_rank_data = b->rank_data;
    max_rank_data = a->rank_data;
    min_rank = b->rank;
    max_rank = a->rank;
  }

  if (min_rank == 0) {
    return TENSOR_CONTRACT_INCOMPATIBLE;
  }
  switch (max_rank) {
    case 1:
      cmpt = min_rank_data[0].shape == max_rank_data[0].shape;
      break;

    // Depends on argument order. If minimum is on left (min == a),
    // then minimum rows must match maximum columns.
    case 2:
      cmpt = min_rank_data[0].shape == max_rank_data[min_rank_data == a->rank_data].shape;
      break;
  }
  return cmpt ? TENSOR_CONTRACT_COMPATIBLE : TENSOR_CONTRACT_INCOMPATIBLE;
}

tensor_compatible_t tensor_broadcast_strides(
    const tensor_t *a,
    const tensor_t *b,
    tensor_rank_data_t a_stride[TENSOR_MAX_RANK],
    tensor_rank_data_t b_stride[TENSOR_MAX_RANK]
) {
  REQUIRE(a && b, return TENSOR_BROADCAST_INCOMPATIBLE);
  tensor_compatible_t cmpt = tensor_broadcast_compatible(a, b);
  REQUIRE(cmpt != TENSOR_BROADCAST_INCOMPATIBLE, return TENSOR_BROADCAST_INCOMPATIBLE);

  memcpy(a_stride, a->rank_data, sizeof(tensor_rank_data_t[TENSOR_MAX_RANK]));
  memcpy(b_stride, b->rank_data, sizeof(tensor_rank_data_t[TENSOR_MAX_RANK]));

  tensor_rank_data_t *source_strides = NULL;
  switch (cmpt) {
    case TENSOR_BROADCAST_B_TO_A:
      source_strides = b_stride;
      break;
    case TENSOR_BROADCAST_A_TO_B:
    case TENSOR_BROADCAST_COMPATIBLE:
      source_strides = a_stride;
      break;
    default:
      return TENSOR_BROADCAST_INCOMPATIBLE;
  }

  if (cmpt != TENSOR_BROADCAST_COMPATIBLE) {
    for (tensor_size_t i = 0; i < TENSOR_MAX_RANK; ++i) {
      tensor_rank_data_t *entry = &source_strides[i];
      if (entry->shape == 1) {
        entry->stride = 0;
      }
    }
  }
  return cmpt;
}

tensor_t *tensor_create(tensor_size_t rank, const tensor_size_t shape[rank]) {
  REQUIRE(rank <= TENSOR_MAX_RANK, return NULL);
  tensor_size_t size = 1;
  for (tensor_size_t i = 0; i < rank; ++i) {
    REQUIRE(shape[i] && TENSOR_MAX_SIZE / shape[i] > size, return NULL);
    size *= shape[i];
  }

  tensor_t *tensor = malloc(sizeof(tensor_t) + sizeof(tensor_type_t[size]));
  REQUIRE(tensor, return NULL);

  tensor->rank = rank;
  tensor_size_t cumulative_stride = 1;
  for (tensor_size_t i = 0; i < rank; ++i) {
    tensor->rank_data[i] = (tensor_rank_data_t){cumulative_stride, shape[i]};
    cumulative_stride *= shape[i];
  }
  for (int i = rank; i < TENSOR_MAX_RANK; ++i) {
    tensor->rank_data[i] = (tensor_rank_data_t){1, 1};
  }
  memset(tensor->data, 0, sizeof(tensor_type_t[size]));
  return tensor;
}

void tensor_destroy(tensor_t **tensor) {
  REQUIRE(tensor && *tensor, return);
  free(*tensor);
  *tensor = NULL;
}

tensor_t *tensor_contract(const tensor_t *a, const tensor_t *b) {
  REQUIRE(a && b, return NULL);
  REQUIRE(tensor_contract_compatible(a, b) == TENSOR_CONTRACT_COMPATIBLE, return NULL);

  tensor_size_t contract_axis_size = TENSOR_SHAPE(a, 0);
  tensor_t *tensor_r = tensor_create(
      max(a->rank, b->rank), TENSOR_DECLARE_SHAPE(TENSOR_SHAPE(b, 0), TENSOR_SHAPE(a, 1))
  );
  REQUIRE(tensor_r, return NULL);
  
  for (tensor_size_t i = 0; i < TENSOR_SHAPE(tensor_r, 1); ++i) {
    for (tensor_size_t j = 0; j < TENSOR_SHAPE(tensor_r, 0); ++j) {
      tensor_type_t point = 0.0;
      for (tensor_size_t k = 0; k < contract_axis_size; ++k) {
        const tensor_size_t ai = i * TENSOR_STRIDE(a, 1) + k * TENSOR_STRIDE(a, 0);
        const tensor_size_t bi = k * TENSOR_STRIDE(b, 1) + j * TENSOR_STRIDE(b, 0);
        point += a->data[ai] * b->data[bi];
      }
      tensor_r->data[i * TENSOR_STRIDE(tensor_r, 1) + j * TENSOR_STRIDE(a, 0)] = point;
    }
  }
  return tensor_r;
}

#define _TENSOR_E_2D_IMPL(op)                                                                     \
  REQUIRE(a &&b, return NULL);                                                                    \
  tensor_rank_data_t a_stride[TENSOR_MAX_RANK] = {};                                              \
  tensor_rank_data_t b_stride[TENSOR_MAX_RANK] = {};                                              \
  tensor_compatible_t cmpt = tensor_broadcast_strides(a, b, a_stride, b_stride);                  \
  REQUIRE(cmpt != TENSOR_BROADCAST_INCOMPATIBLE, return NULL);                                    \
                                                                                                  \
  tensor_size_t shape[TENSOR_MAX_RANK] = {};                                                      \
  for (tensor_size_t i = 0; i < TENSOR_MAX_RANK; ++i) {                                           \
    shape[i] = max(TENSOR_SHAPE(a, i), TENSOR_SHAPE(b, i));                                       \
  }                                                                                               \
                                                                                                  \
  tensor_t *tensor_r = tensor_create(max(a->rank, b->rank), shape);                               \
  REQUIRE(tensor_r, return NULL);                                                                 \
                                                                                                  \
  for (tensor_size_t i = 0; i < tensor_r->rank_data[1].shape; ++i) {                              \
    for (tensor_size_t j = 0; j < tensor_r->rank_data[0].shape; ++j) {                            \
      tensor_type_t av = a->data[i * a_stride[1].stride + j * a_stride[0].stride];                \
      tensor_type_t bv = b->data[i * b_stride[1].stride + j * b_stride[0].stride];                \
      tensor_r->data[i * TENSOR_STRIDE(tensor_r, 1) + j * TENSOR_STRIDE(tensor_r, 0)] = av op bv; \
    }                                                                                             \
  }                                                                                               \
  return tensor_r;

tensor_t *tensor_emul(const tensor_t *a, const tensor_t *b) {
  _TENSOR_E_2D_IMPL(*);
}

tensor_t *tensor_eadd(const tensor_t *a, const tensor_t *b) {
  _TENSOR_E_2D_IMPL(+);
}

tensor_t *tensor_esub(const tensor_t *a, const tensor_t *b) {
  _TENSOR_E_2D_IMPL(-);
}

tensor_t *tensor_ediv(const tensor_t *a, const tensor_t *b) {
  _TENSOR_E_2D_IMPL(/);
}

tensor_t *tensor_emap(const tensor_t *a, tensor_type_t (*f)(tensor_type_t)) {
  REQUIRE(a && f, return NULL);
  tensor_t *tensor_r =
      tensor_create(a->rank, TENSOR_DECLARE_SHAPE(TENSOR_SHAPE(a, 0), TENSOR_SHAPE(a, 1)));
  REQUIRE(tensor_r, return NULL);

  for (tensor_size_t i = 0; i < TENSOR_SHAPE(tensor_r, 1); ++i) {
    for (tensor_size_t j = 0; j < TENSOR_SHAPE(tensor_r, 0); ++j) {
      tensor_size_t ti = i * TENSOR_STRIDE(tensor_r, 1) + j * TENSOR_STRIDE(tensor_r, 0);
      tensor_size_t ai = i * TENSOR_STRIDE(a, 0) + j * TENSOR_STRIDE(a, 1);
      tensor_type_t val = a->data[ai];
      tensor_r->data[ti] = f(val);
    }
  }
  return tensor_r;
}

tensor_t *tensor_transpose(tensor_t *a) {
  tensor_size_t shape[TENSOR_MAX_RANK] = {TENSOR_SHAPE(a, 1), TENSOR_SHAPE(a, 0)};

  // When rank is 1, transposition promotes it to rank 2, else its the minimum.
  tensor_t *transposed = tensor_create(shape[1] != 1 ? 2 : min(a->rank, 1), shape);
  for (tensor_size_t i = 0; i < TENSOR_SHAPE(a, 1); ++i) {
    for (tensor_size_t j = 0; j < TENSOR_SHAPE(a, 0); ++j) {
      tensor_size_t ai = i * TENSOR_STRIDE(a, 1) + j * TENSOR_STRIDE(a, 0);
      tensor_size_t ti = j * TENSOR_STRIDE(transposed, 1) + i * TENSOR_STRIDE(transposed, 0);
      transposed->data[ti] = a->data[ai];
    }
  }
  return transposed;
}

void tensor_transpose_inplace(tensor_t *a) {
  tensor_rank_data_t tmp = a->rank_data[0];
  a->rank_data[0] = a->rank_data[1];
  a->rank_data[1] = tmp;
};

tensor_type_t sigmoid(tensor_type_t x) {
  return (tensor_type_t)(1.0 / (1.0 + exp(-x)));
}

tensor_type_t relu(tensor_type_t x) {
  return (tensor_type_t)(x > 0.0 ? x : 0.0);
}

tensor_type_t leaky_relu(tensor_type_t x) {
  return (tensor_type_t)(x > 0.0 ? x : x * 0.01);
}

tensor_type_t sigmoid_dx(tensor_type_t x) {
  tensor_type_t ex = (tensor_type_t)exp(-x);
  return (tensor_type_t)(ex * pow((1.0 + ex), -2.0));
}

tensor_type_t relu_dx(tensor_type_t x) {
  return (tensor_type_t)(x > 0.0 ? 1.0 : 0.0);
}

tensor_type_t leaky_relu_dx(tensor_type_t x) {
  return (tensor_type_t)(x > 0.0 ? 1.0 : 0.01);
}

void dbg_tensor_print(tensor_t *tensor) {
  for (tensor_size_t i = 0; i < TENSOR_SHAPE(tensor, 1); ++i) {
    for (tensor_size_t j = 0; j < TENSOR_SHAPE(tensor, 0); ++j) {
      tensor_size_t it = i * TENSOR_STRIDE(tensor, 1) + j * TENSOR_STRIDE(tensor, 0);
      tensor_type_t val = tensor->data[it];
      printf("%.1f ", val);
    }
    printf("\n");
  }
}
