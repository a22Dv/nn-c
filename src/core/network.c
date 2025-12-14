/**
 * network.c
 *
 * BRIEF:
 * Implementation for network.h
 */

#include "core/network.h"
#include "core/graph.h"
#include "core/node.h"  // IWYU pragma: export
#include "core/tensor.h"
#include "core/tensor_functions.h"
#include "utils/utils.h"

dense_layer_t *dense_layer_create(
    grph_size_t fan_in,
    grph_size_t fan_out,
    initialization_t init,
    node_type_t function,
    optimizer_t optimizer
) {
  dense_layer_t *layer = malloc(sizeof(dense_layer_t));
  layer->weights = TNSR_MATRIX(fan_in, fan_out);  // X(b,f) * W(f,o) -> b,o
  layer->biases = TNSR_ROWVEC(fan_out);           // B(1, o)
  REQUIRE(layer->weights && layer->biases, goto error);

  layer->function_type = function;
  layer->weights_id = GRPH_NO_INPUT_ID;
  layer->biases_id = GRPH_NO_INPUT_ID;

  switch (optimizer) {  /// TODO: Implement optimizers.
    case OPT_NONE:
    default:
      layer->optimizer = NULL;
      layer->optimizer_data = NULL;
      layer->opt_dtor = NULL;
      break;
  }
  switch (init) {
    case INIT_HE:
      he_ctx_t he_ctx = (he_ctx_t){fan_in};
      REQUIRE(tnsr_emap(layer->weights, layer->weights, he, &he_ctx), goto error);
      REQUIRE(tnsr_emap(layer->biases, layer->biases, he, &he_ctx), goto error);
      break;
    case INIT_GLOROT:
      glorot_ctx_t glorot_ctx = (glorot_ctx_t){fan_in, fan_out};
      REQUIRE(tnsr_emap(layer->weights, layer->weights, glorot, &glorot_ctx), goto error);
      REQUIRE(tnsr_emap(layer->biases, layer->biases, glorot, &glorot_ctx), goto error);
      break;
    case INIT_RANDOM_UNIFORM:
      REQUIRE(tnsr_emap(layer->weights, layer->weights, rand_uniform, NULL), goto error);
      REQUIRE(tnsr_emap(layer->biases, layer->biases, rand_uniform, NULL), goto error);
      break;
  }

  return layer;
error:
  if (layer) {
    tnsr_destroy(&layer->weights);
    tnsr_destroy(&layer->biases);
  }
  free(layer);
  return NULL;
}

void dense_layer_destroy(dense_layer_t **dl) {
  REQUIRE(dl && *dl, return);
  (*dl)->opt_dtor((*dl)->optimizer_data);
  tnsr_destroy(&(*dl)->weights);
  tnsr_destroy(&(*dl)->biases);
  free(*dl);
  *dl = NULL;
}

bool dense_layer_add_to_graph(grph_t **g, dense_layer_t *dl) {
  ASSERT(g && *g && dl);
  ASSERT(dl->weights_id == GRPH_NO_INPUT_ID && dl->biases_id == GRPH_NO_INPUT_ID);
  dl->weights_id = grph_append_data(g, dl->weights);
  dl->biases_id = grph_append_data(g, dl->biases);
  REQUIRE(dl->weights_id != GRPH_ERR_ID && dl->biases_id != GRPH_ERR_ID, goto error);
  return true;
error:
  return false;
}

void dense_layer_remove_from_graph(dense_layer_t *dl) {
  ASSERT(dl);
  dl->weights_id = GRPH_NO_INPUT_ID;
  dl->biases_id = GRPH_NO_INPUT_ID;
}

grph_size_t dense_layer_passthrough(grph_t **g, dense_layer_t *dl, grph_size_t input) {
  ASSERT(g && *g && dl && input != GRPH_NO_INPUT_ID);
  ASSERT(dl->weights_id != GRPH_NO_INPUT_ID && dl->biases_id != GRPH_NO_INPUT_ID);

  grph_size_t nd = grph_execute(g, input, dl->weights_id, NDTYPE_CONTRACT);
  REQUIRE(nd != GRPH_ERR_ID, goto error);
  nd = grph_execute(g, nd, dl->biases_id, NDTYPE_EADD);
  REQUIRE(nd != GRPH_ERR_ID, goto error);
  switch (dl->function_type) {
    case NDTYPE_ELEAKYRELU:
    case NDTYPE_ERELU:
    case NDTYPE_ESIGMOID:
    case NDTYPE_SOFTMAX:
      nd = grph_execute(g, nd, GRPH_NO_INPUT_ID, dl->function_type);
      REQUIRE(nd, goto error);
      break;
    default:
      ASSERT(false);  // Unreachable.
  }
  return nd;
error:
  return GRPH_ERR_ID;
}

bool dense_layer_update(grph_t **g, dense_layer_t *dl) {
  ASSERT(g && *g && dl);
  ASSERT(dl->weights_id != GRPH_NO_INPUT_ID && dl->biases_id != GRPH_NO_INPUT_ID);
  tnsr_t *ml = NULL;
  if (!dl->optimizer) {
    ml = TNSR_SCALAR();
    REQUIRE(ml, goto error);
    tnsr_set(ml, -DEFAULT_LR);
    tnsr_t *wgrad = GRPH_NODE_GRAD(*g, dl->weights_id);
    tnsr_t *bgrad = GRPH_NODE_GRAD(*g, dl->biases_id);
    REQUIRE(tnsr_emul(wgrad, wgrad, ml), goto error);
    REQUIRE(tnsr_emul(bgrad, bgrad, ml), goto error);
    REQUIRE(tnsr_eadd(dl->weights, dl->weights, wgrad), goto error);
    REQUIRE(tnsr_eadd(dl->biases, dl->biases, bgrad), goto error);
  } else {
    REQUIRE(dl->optimizer(dl), goto error);
  }
  tnsr_destroy(&ml);
  return true;
error:
  tnsr_destroy(&ml);
  return false;
}

void dense_layer_dbgprint(dense_layer_t *dl) {
  ASSERT(dl);
  printf("W:\n");
  tnsr_dbgprint(dl->weights);
  printf("B:\n");
  tnsr_dbgprint(dl->biases);
}