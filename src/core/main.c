/**
 * main.c
 *
 * TODO:
 * Implement batching support.
 */

#define DEFAULT_LR 0.05f

#include <Windows.h>
#include <stdio.h>
#include <time.h>

#include "core/graph.h"
#include "core/network.h"
#include "core/node.h"  // IWYU pragma: export
#include "core/tensor.h"

#define DBGPRINT_BATCHED                                              \
  printf("NETWORK INPUT:\n");                                         \
  tnsr_dbgprint(input);                                               \
  printf("NETWORK OUT:\n");                                           \
  tnsr_dbgprint(GRPH_NODE_DATA(graph, GRPH_NODE_DEPS(graph, n)[0]));  \
  printf("EXPECTED OUT:\n");                                          \
  tnsr_dbgprint(expected);                                            \
  printf("LOSS: %+.3f  \n", TNSR_DATA(GRPH_NODE_DATA(graph, n), 0, 0)); \
  for (size_t i = 0; i < nlayers; ++i) {                              \
    dense_layer_dbgprint(layers[i]);                                  \
  }

#define DBGPRINT_INDIV                                                \
  printf("[%llu]\n", passes);                                         \
  printf("NETWORK INPUT:\n");                                         \
  tnsr_dbgprint(inputs[smp]);                                         \
  printf("NETWORK OUT:\n");                                           \
  tnsr_dbgprint(GRPH_NODE_DATA(graph, GRPH_NODE_DEPS(graph, n)[0]));  \
  printf("EXPECTED OUT:\n");                                          \
  tnsr_dbgprint(expected[smp]);                                       \
  printf("LOSS: %+.3f  \n", TNSR_DATA(GRPH_NODE_DATA(graph, n), 0, 0)); \
  // for (size_t i = 0; i < nlayers; ++i) {                              \
  //   dense_layer_dbgprint(layers[i]);                                  \
  // }

int xor_model_batched();
int xor_model_indiv();

int main() {
  srand(6767);
  system("cls");

  return xor_model_indiv();
}

int xor_model_batched() {
  tnsr_type_t inputr[] = {0, 0, 0, 1, 1, 0, 1, 1};
  tnsr_type_t expectedr[] = {0, 1, 1, 0};
  tnsr_t *input = TNSR_MATRIX(4, 2);
  TNSR_FROM_ARRAY(input, inputr);
  tnsr_t *expected = TNSR_COLVEC(4);
  TNSR_FROM_ARRAY(expected, expectedr);
  dense_layer_t *layers[] = {
      dense_layer_create(2, 4, INIT_GLOROT, NDTYPE_ELEAKYRELU, OPT_NONE),
      dense_layer_create(4, 1, INIT_GLOROT, NDTYPE_ESIGMOID, OPT_NONE),
  };
  const size_t nlayers = sizeof(layers) / sizeof(dense_layer_t *);
  const size_t epochs = 10000;
  size_t passes = 0;
  while (passes < epochs) {
    grph_t *graph = grph_create(0);
    for (size_t i = 0; i < nlayers; ++i)
      dense_layer_add_to_graph(&graph, layers[i]);

    grph_size_t n = grph_append_data(&graph, input);
    const grph_size_t m = grph_append_data(&graph, expected);
    for (size_t i = 0; i < nlayers; ++i) {
      n = dense_layer_passthrough(&graph, layers[i], n);
    }
    n = grph_execute(&graph, n, m, NDTYPE_BINARY_CROSS_ENTROPY_LOSS);
    grph_trace(graph);
    for (size_t i = 0; i < nlayers; ++i) {
      dense_layer_update(&graph, layers[i]);
    }

    for (size_t i = 0; i < nlayers; ++i) {
      dense_layer_remove_from_graph(layers[i]);
    }
    grph_destroy(&graph);
    ++passes;
  }
  return 0;
}

// 2-4-1. XOR demo prototype. Individual no-batch.
// Converges ~500 epochs with LR=0.01
int xor_model_indiv() {
  tnsr_type_t inputs_raw[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  tnsr_type_t expected_raw[][1] = {{0}, {1}, {1}, {0}};
  tnsr_t *inputs[] = {
      TNSR_ROWVEC(2),
      TNSR_ROWVEC(2),
      TNSR_ROWVEC(2),
      TNSR_ROWVEC(2),
  };
  tnsr_t *expected[] = {
      TNSR_SCALAR(),
      TNSR_SCALAR(),
      TNSR_SCALAR(),
      TNSR_SCALAR(),
  };
  for (size_t i = 0; i < 4; ++i) {
    TNSR_FROM_ARRAY(inputs[i], inputs_raw[i]);
    TNSR_FROM_ARRAY(expected[i], expected_raw[i]);
  }
  dense_layer_t *layers[] = {
      dense_layer_create(2, 32, INIT_HE, NDTYPE_ELEAKYRELU, OPT_NONE),
      dense_layer_create(32, 1, INIT_HE, NDTYPE_ESIGMOID, OPT_NONE),
  };
  const size_t nlayers = sizeof(layers) / sizeof(dense_layer_t *);
  const size_t epochs = 10000;

  // tnsr_t *biased = TNSR_SCALAR();
  // tnsr_set(biased, 1.0);
  // for (size_t i = 0; i < nlayers; ++i) {
  //   tnsr_eadd(layers[i]->biases, layers[i]->biases, biased);
  // }

  size_t passes = 0;
  while (passes / 4 < epochs) {
    size_t smp = rand() % 4;
    grph_t *graph = grph_create(0);
    for (size_t i = 0; i < nlayers; ++i) {
      dense_layer_add_to_graph(&graph, layers[i]);
    }
    grph_size_t n = grph_append_data(&graph, inputs[smp]);
    const grph_size_t m = grph_append_data(&graph, expected[smp]);
    for (size_t i = 0; i < nlayers; ++i) {
      n = dense_layer_passthrough(&graph, layers[i], n);
    }

    n = grph_execute(&graph, n, m, NDTYPE_BINARY_CROSS_ENTROPY_LOSS);
    grph_trace(graph);
    if (passes % 11 == 0) {
      DBGPRINT_INDIV
      printf("\033[H");
      Sleep(100);
    }
    for (size_t i = 0; i < nlayers; ++i) {
      dense_layer_update(&graph, layers[i]);
    }
    for (size_t i = 0; i < nlayers; ++i) {
      dense_layer_remove_from_graph(layers[i]);
    }
    grph_destroy(&graph);
    ++passes;
  }
  return 0;
}