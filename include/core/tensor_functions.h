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

typedef struct {
  tnsr_type_t n;
} gen_ctx_t;

// Returns the result for powf(x, n);
FRCINL tnsr_type_t tnsr_powf(tnsr_type_t x, void *ctx) { 
  ASSERT(ctx); 
  return powf(x, ((gen_ctx_t*)ctx)->n);
}

// Returns the value of the sigmoid function at x.
FRCINL tnsr_type_t tnsr_sigmoid(tnsr_type_t x, void *ctx) {
  (void)ctx;
  return 1 / (1 + exp(-x));
}

// Returns the result of tanhf(x).
FRCINL tnsr_type_t tnsr_tanh(tnsr_type_t x, void *ctx) {
  (void)ctx;
  return tanhf(x);
}

// Returns the value of the ReLU function at x.
FRCINL tnsr_type_t tnsr_relu(tnsr_type_t x, void *ctx) {
  (void)ctx;
  return x < 0 ? 0 : x;
}

// Returns the value of the Leaky ReLU function at x.
// Note that this function uses the context pointer as the alpha value.
// Such that calculating the derivative of this function requires the same alpha value to 
// be called with.
FRCINL tnsr_type_t tnsr_leaky_relu(tnsr_type_t x, void *ctx) {  
  ASSERT(ctx);
  return x < 0 ? ((gen_ctx_t*)ctx)->n * x : x;
}

// Returns the value of the derivative of the sigmoid function at x.
FRCINL tnsr_type_t tnsr_sigmoid_dx(tnsr_type_t x, void *ctx) {
  (void)ctx;
  tnsr_type_t ex = expf(-x);
  return ex / ((1 + ex) * (1 + ex));
}

// Returns the value of the derivative of the sigmoid function at x.
// Note that x is assumed to be the output of the previous sigmoid function.
FRCINL tnsr_type_t tnsr_sigmoid_odx(tnsr_type_t x, void *ctx) {
  (void)ctx;
  return x * (1 - x);
}

// Returns the derivative of the ReLU function at x.
FRCINL tnsr_type_t tnsr_relu_dx(tnsr_type_t x, void *ctx) {
  (void)ctx;
  return x > 0;
}

// Returns the derivative of the tanh function at x. 
FRCINL tnsr_type_t tnsr_tanh_dx(tnsr_type_t x, void *ctx) {
  (void)ctx;
  const tnsr_type_t tanhx = tanhf(x);
  return 1 - (tanhx * tanhx);
}

// Returns the derivative of the tanh function at x. 
// Note that x is assumed to be the output of the tanh function.
FRCINL tnsr_type_t tnsr_tanh_odx(tnsr_type_t x, void *ctx) {
  (void)ctx;
  return 1 - (x * x);
}

// Returns the derivative of the Leaky ReLU function at x.
// Note that this function is highly dependent on the context given, and must
// be given the same value that the non-dx function has been called with to prevent 
// garbage output.
FRCINL tnsr_type_t tnsr_leaky_relu_dx(tnsr_type_t x, void *ctx) {
  (void)ctx;
  return x < 0 ? ((gen_ctx_t*)ctx)->n: 1;
}

// Returns a random number based on C's stdlib rand(). Passing a non-zero x
// offsets the range from [-1 -> 1] to [-1 + x -> 1 + x].
FRCINL tnsr_type_t tnsr_rand_uniform(tnsr_type_t x, void *ctx) {
  (void)x;
  return ((((tnsr_type_t)(rand()) / RAND_MAX) - 0.5f) * 2.0f) + ((gen_ctx_t*)ctx)->n;
}

// Uses the uniform variation for Glorot initialization.
FRCINL tnsr_type_t tnsr_glorot(tnsr_type_t x, void *ctx) {
  ASSERT(ctx);
  (void)x;
  glorot_ctx_t *gctx = ctx;
  gen_ctx_t rand_ctx = {0};
  return tnsr_rand_uniform(0, &rand_ctx) * sqrtf(6.0f / (gctx->fan_in + gctx->fan_out));
}

// Uses the uniform variation for He initialization.
FRCINL tnsr_type_t tnsr_he(tnsr_type_t x, void *ctx) {
  ASSERT(ctx);
  he_ctx_t *hctx = ctx;
  gen_ctx_t rand_ctx = {0};
  return tnsr_rand_uniform(x, &rand_ctx) * sqrtf(6.0f / (hctx->fan_in));
}

// Returns the value of expf(x).
FRCINL tnsr_type_t tnsr_expf(tnsr_type_t x, void *ctx) {
  (void)ctx;
  return expf(x);
}

// Returns the value of logf(x) (natural logarithm).
FRCINL tnsr_type_t tnsr_ln(tnsr_type_t x, void *ctx) {
  (void)ctx;
  return logf(x);
}

// Returns the value of x multiplied by n. 
FRCINL tnsr_type_t tnsr_mul_n(tnsr_type_t x, void *ctx) {
  ASSERT(ctx);
  return x * ((gen_ctx_t*)ctx)->n;
}

// Returns the value of x added by n.
FRCINL tnsr_type_t tnsr_add_n(tnsr_type_t x, void *ctx) {
  ASSERT(ctx);
  return x + ((gen_ctx_t*)ctx)->n;
}

// Returns the value of n - x, where x acts as the subtrahend. 
FRCINL tnsr_type_t tnsr_as_subtrahend(tnsr_type_t x, void *ctx) {
  ASSERT(ctx);
  return ((gen_ctx_t*)ctx)->n - x;
}

// Returns the value of x - n, where x acts as the minuend.
FRCINL tnsr_type_t tnsr_as_minuend(tnsr_type_t x, void *ctx) {
  ASSERT(ctx);
  return x - ((gen_ctx_t*)ctx)->n;
}

// Returns the value of x / n, where x acts as the minuend.
FRCINL tnsr_type_t tnsr_as_dividend(tnsr_type_t x, void *ctx) {
  ASSERT(ctx);
  return x / ((gen_ctx_t*)ctx)->n;
}

// Returns the value of n / x, where x acts as the divisor.
FRCINL tnsr_type_t tnsr_as_divisor(tnsr_type_t x, void *ctx) {
  ASSERT(ctx);
  return ((gen_ctx_t*)ctx)->n / x;
}



