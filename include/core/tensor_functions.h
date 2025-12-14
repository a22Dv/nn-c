/**
 * tensor_functions.h
 *
 * BRIEF:
 * Tensor-applied inline utility functions.
 */

#include "core/tensor.h"
#include "utils/utils.h"

#pragma once

typedef struct {
  int fan_in;
  int fan_out;
} glorot_ctx_t;

typedef struct {
  int fan_in;
} he_ctx_t;

FRCINL tnsr_type_t pow_2(tnsr_type_t x, void *ctx) {
  (void)ctx;
  return x * x;
}

FRCINL tnsr_type_t pow_neg1(tnsr_type_t x, void *ctx) {
  (void)ctx;
  return 1 / x;
}

FRCINL tnsr_type_t pow_neg2(tnsr_type_t x, void *ctx) {
  (void)ctx;
  return 1 / (x * x);
}

FRCINL tnsr_type_t sigmoid(tnsr_type_t x, void *ctx) {
  (void)ctx;
  return 1 / (1 + exp(-x));
}

FRCINL tnsr_type_t relu(tnsr_type_t x, void *ctx) {
  (void)ctx;
  return max(0, x);
}

FRCINL tnsr_type_t leaky_relu(tnsr_type_t x, void *ctx) {
  (void)ctx;
  return x < 0 ? 0.01 * x : x;
}

FRCINL tnsr_type_t sigmoid_dx(tnsr_type_t x, void *ctx) {
  (void)ctx;
  tnsr_type_t ex = expf(-x);
  return ex / ((1 + ex) * (1 + ex));
}

FRCINL tnsr_type_t sigmoid_odx(tnsr_type_t ox, void *ctx) {
  (void)ctx;
  return ox * (1 - ox);
}

FRCINL tnsr_type_t relu_dx(tnsr_type_t x, void *ctx) {
  (void)ctx;
  return x > 0;
}

FRCINL tnsr_type_t leaky_relu_dx(tnsr_type_t x, void *ctx) {
  (void)ctx;
  return x > 0 ? 1 : 0.01;
}

FRCINL tnsr_type_t rand_uniform(tnsr_type_t x, void *ctx) {
  (void)ctx;
  (void)x;
  return (((tnsr_type_t)(rand()) / RAND_MAX) -0.5f) * 2.0f;  // Range [0.0-1.0]
}

FRCINL tnsr_type_t glorot(tnsr_type_t x, void *ctx) {
  ASSERT(ctx);
  glorot_ctx_t *gctx = ctx;
  return rand_uniform(x, NULL) * sqrt(6.0f / (gctx->fan_in + gctx->fan_out));
}

FRCINL tnsr_type_t he(tnsr_type_t x, void *ctx) {
  ASSERT(ctx);
  he_ctx_t *hctx = ctx;
  return rand_uniform(x, NULL) * sqrt(6.0f / (hctx->fan_in));
}

FRCINL tnsr_type_t euler(tnsr_type_t x, void *ctx) {
  (void)ctx;
  return expf(x);
}

FRCINL tnsr_type_t ln(tnsr_type_t x, void *ctx) {
  (void)ctx;
  return logf(x);
}

FRCINL tnsr_type_t mul_neg1(tnsr_type_t x, void *ctx) {
  (void)ctx;
  return -x;
}
