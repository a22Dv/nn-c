// clang-format off

#include <Windows.h>
#include <time.h>

#include "core/graph.h"
#include "core/tensor.h"
#include "utils/utils.h"
#include "core/tensor_functions.h"

int xor_model_indiv();

int main() {
  srand(time(NULL));
  return xor_model_indiv();
}

int xor_model_indiv() {  // 2-2-1.
  tnsr_t *inputs[] = {TNSR_ROWVEC(2), TNSR_ROWVEC(2), TNSR_ROWVEC(2), TNSR_ROWVEC(2)};
  tnsr_t *expected[] = {TNSR_SCALAR(), TNSR_SCALAR(), TNSR_SCALAR(), TNSR_SCALAR()};
  tnsr_t *weights[] = {TNSR_MATRIX(2, 2), TNSR_MATRIX(2, 1)};  // i, j Node_i -> Node_j
  tnsr_t *biases[] = {TNSR_MATRIX(1, 2), TNSR_MATRIX(1, 1)};
  grph_size_t weights_ids[] = {0, 0, 0, 0};
  grph_size_t biases_ids[] = {0, 0};

  tnsr_set(inputs[0], 0);  // [0, 0]
  tnsr_set(expected[0], 0);

  TNSR_DATA(inputs[1], 0, 0) = 1;  // [1, 0]
  tnsr_set(expected[1], 1);

  TNSR_DATA(inputs[2], 0, 1) = 1;  // [0, 1]
  tnsr_set(expected[2], 1);

  tnsr_set(inputs[3], 1);  // [1, 1]
  tnsr_set(expected[3], 0);

  tnsr_t *lr = TNSR_SCALAR();
  tnsr_set(lr, -0.05);

  size_t sample_n = 0;
  size_t samples_ran = 0;

  size_t nweights = sizeof(weights) / sizeof(tnsr_t *);
  for (size_t i = 0; i < nweights; ++i) {
    REQUIRE(tnsr_emap(weights[i], weights[i], rand_uniform, NULL), goto error);
    REQUIRE(tnsr_emap(biases[i], biases[i], rand_uniform, NULL), goto error);
  }
  
  system("cls");
  while (true) {

    grph_t *graph = grph_create(16);

    for (size_t i = 0; i < nweights; ++i) {
      REQUIRE((weights_ids[i] = grph_append_data(&graph, weights[i])) != GRPH_ERR_ID, goto error);
      REQUIRE((biases_ids[i] = grph_append_data(&graph, biases[i])) != GRPH_ERR_ID, goto error);
    }

    grph_size_t cId = grph_append_data(&graph, inputs[sample_n]);
    for (size_t i = 0; i < nweights; ++i) {
      REQUIRE((cId = grph_execute(&graph, cId, weights_ids[i], NDTYPE_CONTRACT)) != GRPH_ERR_ID, goto error);
      REQUIRE((cId = grph_execute(&graph, biases_ids[i], cId, NDTYPE_EADD)) != GRPH_ERR_ID, goto error);
      if (i == nweights - 1) {
        continue;
      }
      REQUIRE((cId = grph_execute(&graph, cId, GRPH_NO_INPUT_ID, NDTYPE_ELEAKYRELU)) != GRPH_ERR_ID, goto error);
    } 

    grph_size_t ground = grph_append_data(&graph, expected[sample_n]);
    grph_size_t mse = 0;
    REQUIRE((mse = grph_execute(&graph, cId, ground, NDTYPE_MSE)) != GRPH_ERR_ID, goto error);

    REQUIRE(grph_trace(graph), goto error);
    
    if (samples_ran % 11 == 0) {
      printf("SAMPLE NO. %lld\n", samples_ran);
      printf("INPUT: [%.3f, %.3f]  \n", TNSR_DATA(inputs[sample_n], 0, 0), TNSR_DATA(inputs[sample_n], 0, 1));
      printf("EXPECTED OUT: %.3f \n", TNSR_DATA(expected[sample_n], 0, 0));
      printf("NETWORK OUT: %.3f \n",  TNSR_DATA(GRPH_NODE_DATA(graph, GRPH_NODE_DEPS(graph, mse)[0]), 0, 0));
      printf("NETWORK LOSS: %.3f \n", TNSR_DATA(GRPH_NODE_DATA(graph, mse), 0, 0));
      for (size_t i = 0; i < nweights; ++i) {
        printf("W%llu:\n", i+1);
        tnsr_dbgprint(weights[i]);
        printf("\nB%llu:\n", i+1);
        tnsr_dbgprint(biases[i]);
      }
      printf("\033[H");
      Sleep(100);
    }
    

    for (size_t i = 0; i < nweights; ++i) {
      tnsr_t* w_g = GRPH_NODE_GRAD(graph, weights_ids[i]);
      tnsr_t* b_g = GRPH_NODE_GRAD(graph, biases_ids[i]);
      REQUIRE(tnsr_emul(w_g, w_g, lr), goto error);
      REQUIRE(tnsr_emul(b_g, b_g, lr), goto error);
      REQUIRE(tnsr_eadd(weights[i], weights[i], w_g), goto error);
      REQUIRE(tnsr_eadd(biases[i], biases[i], b_g), goto error);
    }
    
    grph_destroy(&graph);
    sample_n = (sample_n + 1) % 4;
    ++samples_ran;
  }
error:
  return EXIT_FAILURE;
}