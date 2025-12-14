#include <Windows.h>
#include <time.h>

#include "core/graph.h"
#include "core/network.h"
#include "core/tensor.h"

int xor_model_indiv();

int main() {
  srand(time(NULL));

  return xor_model_indiv();
}

int xor_model_indiv() {  // 2-2-1. XOR demo prototype. Individual no-batch.
  tnsr_type_t inputs_raw[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  tnsr_t *inputs[] = {
    TNSR_ROWVEC(2),
    TNSR_ROWVEC(2),
    TNSR_ROWVEC(2),
    TNSR_ROWVEC(2),
  };
  for (size_t i = 0; i < 4; ++i) {
    TNSR_FROM_ARRAY(inputs[i], inputs_raw[i]);
  }
  dense_layer_t *layers[] = {
      dense_layer_create(2, 2, INIT_HE, NDTYPE_ERELU, OPT_NONE),
      dense_layer_create(2, 1, INIT_HE, NDTYPE_MSE, OPT_NONE),
  };

  const size_t epochs = 100;
  size_t passes = 0;
  while (passes / 4 < epochs) {
    grph_t *graph = grph_create(0);
    for (size_t i = 0; i < 2; ++i) {
      dense_layer_add_to_graph(&graph, layers[i]);
    }
    grph_size_t n = grph_append_data(&graph, inputs[passes % 4]);
    for (size_t i = 0; i < 2; ++i) {
      n = dense_layer_passthrough(&graph, layers[i], n);
    }
    grph_trace(graph);
    for (size_t i = 0; i < 2; ++i) {
      dense_layer_update(&graph, layers[i]);
    }
    ++passes;
  }
  return 0;
}