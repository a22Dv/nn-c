#include <stdio.h>
#include <stdlib.h>
#include <Windows.h>

#include "core/cgraph.h"
#include "core/tensor.h"
#include "utils/utils.h"

tensor_type_t random(tensor_type_t _) {
  return rand() / (tensor_type_t)RAND_MAX;
}

// Testing Prototype. 2-2-1 network. XOR problem. Converges in ~1000 runs. Final loss: 0.0.
int main() {
  // Initialization.
  cgraph_t graph = {};
  tensor_t *data = tensor_create(2, TENSOR_DECLARE_SHAPE(1, 2));  // (Inner, Outer) dimensions
  tensor_t *gradient =
      tensor_create(2, TENSOR_DECLARE_SHAPE(1, 2));               // Not really used w.r.t input.
  REQUIRE(data && gradient, return EXIT_FAILURE);

  graph.head = cnode_create(0, 1, NULL, NULL, data, gradient, CN_DATA);
  REQUIRE(graph.head, return EXIT_FAILURE);

  tensor_t *w1t = tensor_create(2, TENSOR_DECLARE_SHAPE(2, 2));
  tensor_t *w1g = tensor_create(2, TENSOR_DECLARE_SHAPE(2, 2));
  tensor_t *b1t = tensor_create(2, TENSOR_DECLARE_SHAPE(1, 2));
  tensor_t *b1g = tensor_create(2, TENSOR_DECLARE_SHAPE(1, 2));
  REQUIRE(w1t && w1g && b1t && b1g, return EXIT_FAILURE);

  cnode_t *w1 = cnode_create(0, 1, NULL, NULL, w1t, w1g, CN_DATA);
  cnode_t *b1 = cnode_create(0, 1, NULL, NULL, b1t, b1g, CN_DATA);
  REQUIRE(w1 && b1, return EXIT_FAILURE);

  tensor_t *w2t = tensor_create(2, TENSOR_DECLARE_SHAPE(2, 1));
  tensor_t *w2g = tensor_create(2, TENSOR_DECLARE_SHAPE(2, 1));
  tensor_t *b2t = tensor_create(2, TENSOR_DECLARE_SHAPE(1, 1));
  tensor_t *b2g = tensor_create(2, TENSOR_DECLARE_SHAPE(1, 1));
  REQUIRE(w2t && w2g && b2t && b2g, return EXIT_FAILURE);

  cnode_t *w2 = cnode_create(0, 1, NULL, NULL, w2t, w2g, CN_DATA);
  cnode_t *b2 = cnode_create(0, 1, NULL, NULL, b2t, b2g, CN_DATA);
  REQUIRE(w2 && b2, return EXIT_FAILURE);

  // Graph setup.
  cnode_t *w[] = {w1, w2};
  cnode_t *b[] = {b1, b2};
  cnode_t *prv_node = graph.head;
  for (tensor_size_t i = 0; i < 2; ++i) {
    tensor_size_t od = TENSOR_SHAPE(w[i]->data, 1);

    tensor_t *dataw = tensor_create(2, TENSOR_DECLARE_SHAPE(1, od));
    tensor_t *gradientw = tensor_create(2, TENSOR_DECLARE_SHAPE(1, od));
    prv_node =
        cnode_attach((cnode_t *[CN_MAX_INDEGREE]){w[i], prv_node}, CN_CONTRACT, dataw, gradientw);
    REQUIRE(prv_node, return EXIT_FAILURE);

    tensor_t *datab = tensor_create(2, TENSOR_DECLARE_SHAPE(1, od));
    tensor_t *gradientb = tensor_create(2, TENSOR_DECLARE_SHAPE(1, od));
    prv_node = cnode_attach((cnode_t *[CN_MAX_INDEGREE]){b[i], prv_node}, CN_ADD, datab, gradientb);
    REQUIRE(prv_node, return EXIT_FAILURE);

    tensor_t *dataa = tensor_create(2, TENSOR_DECLARE_SHAPE(1, od));
    tensor_t *gradienta = tensor_create(2, TENSOR_DECLARE_SHAPE(1, od));
    prv_node = cnode_attach((cnode_t *[CN_MAX_INDEGREE]){prv_node}, CN_RELU, dataa, gradienta);
    REQUIRE(prv_node, return EXIT_FAILURE);
  }

  // MSE setup.
  tensor_t *tdata = tensor_create(2, TENSOR_DECLARE_SHAPE(1, 1));
  tensor_t *tgrad = tensor_create(2, TENSOR_DECLARE_SHAPE(1, 1));  // unused.
  cnode_t *tnode = cnode_create(0, 1, NULL, NULL, tdata, tgrad, CN_DATA);
  REQUIRE(tdata && tgrad && tnode, return -1);

  tensor_t *mset = tensor_create(0, TENSOR_DECLARE_SHAPE(1, 1));
  tensor_t *mseg = tensor_create(0, TENSOR_DECLARE_SHAPE(1, 1));
  cnode_t *msen = cnode_attach((cnode_t *[CN_MAX_INDEGREE]){prv_node, tnode}, CN_MSE, mset, mseg);
  mseg->data[0] = 1;  // Derivative of loss w.r.t loss = 1. Seed value.

  // Data setup.
  tensor_type_t data_d1[] = {1.0, 0.0, 1.0, 0.0};
  tensor_type_t data_d2[] = {1.0, 1.0, 0.0, 0.0};
  tensor_type_t expected[] = {0.0, 1.0, 1.0, 0.0};

  // Randomize weights and biases.
  tensor_emap(w1t, w1t, random);
  tensor_emap(b1t, b1t, random);
  tensor_emap(w2t, w2t, random);
  tensor_emap(b2t, b2t, random);

  tensor_size_t sample = 0;

  tensor_t *n1 = tensor_create(0, TENSOR_DECLARE_SHAPE(1, 1));
  tensor_t *lr = tensor_create(0, TENSOR_DECLARE_SHAPE(1, 1));
  n1->data[0] = -1;
  lr->data[0] = 0.05;

  system("cls");
  while (true) {  // training loop.
    cnode_reset_data(graph.head);
    cnode_reset_gradients(graph.head);
    mseg->data[0] = 1;  // Derivative of loss w.r.t loss = 1. Gradient seed value.

    graph.head->data->data[0] = data_d1[sample % 4];
    graph.head->data->data[1] = data_d2[sample % 4];
    tnode->data->data[0] = expected[sample % 4];

    // Forward and back.
    REQUIRE(cnode_traverse_and_perform(graph.head), return EXIT_FAILURE);
    REQUIRE(cnode_traverse_gradient(msen), return EXIT_FAILURE);

    if (sample % 30 == 0) {
      printf("LOSS [RUN no. %u] = %.2f\n", sample, msen->data->data[0]);
      printf("NETWORK OUT: %.5f\n", msen->prev[0]->data->data[0]);
      printf("EXPECTED OUT: %.5f\n", msen->prev[1]->data->data[0]);

      printf("\nW1:\n");
      dbg_tensor_print(w1->data);

      printf("B1:\n");
      dbg_tensor_print(b1->data);

      printf("\nW2:\n");
      dbg_tensor_print(w2->data);

      printf("\nB2:\n");
      dbg_tensor_print(b2->data);
      printf("\033[H");

      Sleep(100);
    }
    ++sample;

    for (int i = 0; i < 2; ++i) {
      tensor_emul(w[i]->gradient, w[i]->gradient, n1);
      tensor_emul(w[i]->gradient, w[i]->gradient, lr);
      tensor_eadd(w[i]->data, w[i]->data, w[i]->gradient);
      tensor_emap(w[i]->gradient, w[i]->gradient, zeroes);

      tensor_emul(b[i]->gradient, b[i]->gradient, n1);
      tensor_emul(b[i]->gradient, b[i]->gradient, lr);
      tensor_eadd(b[i]->data, b[i]->data, b[i]->gradient);
      tensor_emap(b[i]->gradient, b[i]->gradient, zeroes);
    }
  }
  tensor_destroy(&n1);
  tensor_destroy(&lr);
}