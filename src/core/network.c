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

typedef struct {
  tnsr_t *moment_w;
  tnsr_t *moment_b;
} momentum_data_t;

typedef struct {
  tnsr_t *moment_w;
  tnsr_t *moment_b;
} rms_prop_data_t;

typedef struct {
  tnsr_t *moment1_w;
  tnsr_t *moment1_b;
  tnsr_t *moment2_w;
  tnsr_t *moment2_b;
  tnsr_type_t timestamp;
} adam_data_t;

dense_layer_t *dense_layer_create(
    grph_size_t fan_in,
    grph_size_t fan_out,
    initialization_t init,
    node_type_t function,
    optimizer_t optimizer,
    tnsr_type_t learning_rate
) {
  dense_layer_t *layer = malloc(sizeof(dense_layer_t));
  layer->weights = TNSR_MATRIX(fan_in, fan_out);  // X(b,f) * W(f,o) -> b,o
  layer->biases = TNSR_ROWVEC(fan_out);           // B(1, o)
  REQUIRE(layer->weights && layer->biases, goto error);

  layer->function_type = function;
  layer->weights_id = GRPH_NO_INPUT_ID;
  layer->biases_id = GRPH_NO_INPUT_ID;
  layer->learning_rate = learning_rate;

  tnsr_size_t size[TNSR_MAX_RANK] = {fan_in, fan_out};
  switch (optimizer) {
    case OPT_SGD:
      layer->optimizer = dense_layer_sgd;
      layer->opt_dtor = dense_layer_sgd_dtor;
      REQUIRE(dense_layer_sgd_create(size, &layer->optimizer_data), goto error);
      break;
    case OPT_SGD_MOMENTUM:
      layer->optimizer = dense_layer_sgd_momentum;
      layer->opt_dtor = dense_layer_sgd_momentum_dtor;
      REQUIRE(dense_layer_sgd_momentum_create(size, &layer->optimizer_data), goto error);
      break;
    case OPT_SGD_RMS_PROP:
      layer->optimizer = dense_layer_sgd_rms_prop;
      layer->opt_dtor = dense_layer_sgd_rms_prop_dtor;
      REQUIRE(dense_layer_sgd_rms_prop_create(size, &layer->optimizer_data), goto error);
      break;
    case OPT_SGD_ADAM:
      layer->optimizer = dense_layer_sgd_adam;
      layer->opt_dtor = dense_layer_sgd_adam_dtor;
      REQUIRE(dense_layer_sgd_adam_create(size, &layer->optimizer_data), goto error);
      break;
    default:
      ASSERT(false);  // Unreachable.
      break;
  }

  switch (init) {
    case INIT_HE:
      he_ctx_t he_ctx = (he_ctx_t){fan_in};
      REQUIRE(tnsr_emap(layer->weights, layer->weights, tnsr_he, &he_ctx), goto error);
      REQUIRE(tnsr_emap(layer->biases, layer->biases, tnsr_he, &he_ctx), goto error);
      break;
    case INIT_GLOROT:
      glorot_ctx_t glorot_ctx = (glorot_ctx_t){fan_in, fan_out};
      REQUIRE(tnsr_emap(layer->weights, layer->weights, tnsr_glorot, &glorot_ctx), goto error);
      REQUIRE(tnsr_emap(layer->biases, layer->biases, tnsr_glorot, &glorot_ctx), goto error);
      break;
    case INIT_RANDOM_UNIFORM:
      gen_ctx_t rand_ctx = {0};  // No offset.
      REQUIRE(tnsr_emap(layer->weights, layer->weights, tnsr_rand_uniform, &rand_ctx), goto error);
      REQUIRE(tnsr_emap(layer->biases, layer->biases, tnsr_rand_uniform, &rand_ctx), goto error);
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
  (*dl)->opt_dtor(&(*dl)->optimizer_data);
  tnsr_destroy(&(*dl)->weights);
  tnsr_destroy(&(*dl)->biases);
  free(*dl);
  *dl = NULL;
}

bool dense_layer_add_to_graph(grph_t **g, dense_layer_t *dl) {
  ASSERT(g && *g && dl);
  // ASSERT(dl->weights_id == GRPH_NO_INPUT_ID && dl->biases_id == GRPH_NO_INPUT_ID);
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
    case NDTYPE_ETANH:
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
  REQUIRE(dl->optimizer(g, dl), goto error);
  return true;
error:
  return false;
}

bool dense_layer_sgd_create(tnsr_size_t tnsr_shape[TNSR_MAX_RANK], void **dataptr) {
  ASSERT(dataptr);
  (void)tnsr_shape;
  *dataptr = NULL;  // Stateless.
  return true;
}

bool dense_layer_sgd_momentum_create(tnsr_size_t tnsr_shape[TNSR_MAX_RANK], void **dataptr) {
  ASSERT(dataptr);
  momentum_data_t *data = NULL;
  REQUIRE(*dataptr = malloc(sizeof(momentum_data_t)), goto error);
  data = *dataptr;
  REQUIRE(data->moment_w = tnsr_create(tnsr_shape[0], tnsr_shape[1]), goto error);
  REQUIRE(data->moment_b = tnsr_create(1, tnsr_shape[1]), goto error);
  return true;
error:
  return false;
}

bool dense_layer_sgd_rms_prop_create(tnsr_size_t tnsr_shape[TNSR_MAX_RANK], void **dataptr) {
  ASSERT(dataptr);
  rms_prop_data_t *data = NULL;
  REQUIRE(*dataptr = malloc(sizeof(rms_prop_data_t)), goto error);
  data = *dataptr;
  REQUIRE(data->moment_w = tnsr_create(tnsr_shape[0], tnsr_shape[1]), goto error);
  REQUIRE(data->moment_b = tnsr_create(1, tnsr_shape[1]), goto error);
  return true;
error:
  return false;
}

bool dense_layer_sgd_adam_create(tnsr_size_t tnsr_shape[TNSR_MAX_RANK], void **dataptr) {
  ASSERT(dataptr);
  adam_data_t *data = NULL;
  REQUIRE(*dataptr = malloc(sizeof(adam_data_t)), goto error);
  data = *dataptr;
  REQUIRE(data->moment1_w = tnsr_create(tnsr_shape[0], tnsr_shape[1]), goto error);
  REQUIRE(data->moment1_b = tnsr_create(1, tnsr_shape[1]), goto error);
  REQUIRE(data->moment2_w = tnsr_create(tnsr_shape[0], tnsr_shape[1]), goto error);
  REQUIRE(data->moment2_b = tnsr_create(1, tnsr_shape[1]), goto error);
  data->timestamp = 0;
  return true;
error:
  return false;
}

bool dense_layer_sgd(grph_t **g, dense_layer_t *dl) {
  ASSERT(g && *g && dl);
  tnsr_t *wgrad = GRPH_NODE_GRAD(*g, dl->weights_id);
  tnsr_t *bgrad = GRPH_NODE_GRAD(*g, dl->biases_id);

  gen_ctx_t ctx = {-dl->learning_rate};
  REQUIRE(tnsr_emap(wgrad, wgrad, tnsr_mul_n, &ctx), goto error);
  REQUIRE(tnsr_emap(bgrad, bgrad, tnsr_mul_n, &ctx), goto error);
  REQUIRE(tnsr_eadd(dl->weights, dl->weights, wgrad), goto error);
  REQUIRE(tnsr_eadd(dl->biases, dl->biases, bgrad), goto error);
  return true;
error:
  return false;
}

bool dense_layer_sgd_momentum(grph_t **g, dense_layer_t *dl) {
  ASSERT(g && *g && dl);
  gen_ctx_t beta = {0.9f};
  gen_ctx_t i_beta = {0.1f};
  gen_ctx_t lr = {-dl->learning_rate};
  tnsr_t *wgrad = GRPH_NODE_GRAD(*g, dl->weights_id);
  tnsr_t *bgrad = GRPH_NODE_GRAD(*g, dl->biases_id);
  momentum_data_t *mdata = dl->optimizer_data;

  REQUIRE(tnsr_emap(mdata->moment_w, mdata->moment_w, tnsr_mul_n, &beta), goto error);
  REQUIRE(tnsr_emap(mdata->moment_b, mdata->moment_b, tnsr_mul_n, &beta), goto error);
  REQUIRE(tnsr_emap(wgrad, wgrad, tnsr_mul_n, &i_beta), goto error);
  REQUIRE(tnsr_emap(bgrad, bgrad, tnsr_mul_n, &i_beta), goto error);
  REQUIRE(tnsr_eadd(mdata->moment_w, mdata->moment_w, wgrad), goto error);
  REQUIRE(tnsr_eadd(mdata->moment_b, mdata->moment_b, bgrad), goto error);
  REQUIRE(tnsr_emap(wgrad, mdata->moment_w, tnsr_mul_n, &lr), goto error);
  REQUIRE(tnsr_emap(bgrad, mdata->moment_b, tnsr_mul_n, &lr), goto error);
  REQUIRE(tnsr_eadd(dl->weights, dl->weights, wgrad), goto error);
  REQUIRE(tnsr_eadd(dl->biases, dl->biases, bgrad), goto error);

  return true;
error:
  return false;
}

/**
 * NOTE:
 * Runaway weights when loss has converged to 0.0.
 */
bool dense_layer_sgd_rms_prop(grph_t **g, dense_layer_t *dl) {
  ASSERT(g && *g && dl);
  gen_ctx_t beta = {0.9f};
  gen_ctx_t i_beta = {0.1f};
  gen_ctx_t epsilon = {1e-8f};
  gen_ctx_t lr = {-dl->learning_rate};
  tnsr_t *wgrad = GRPH_NODE_GRAD(*g, dl->weights_id);
  tnsr_t *bgrad = GRPH_NODE_GRAD(*g, dl->biases_id);
  rms_prop_data_t *mdata = dl->optimizer_data;

  gen_ctx_t pow = {2.0f};
  tnsr_t *tmp_w = tnsr_emap(NULL, wgrad, tnsr_powf, &pow);
  tnsr_t *tmp_b = tnsr_emap(NULL, bgrad, tnsr_powf, &pow);
  REQUIRE(tmp_w && tmp_b, goto error);

  REQUIRE(tnsr_emap(mdata->moment_w, mdata->moment_w, tnsr_mul_n, &beta), goto error);
  REQUIRE(tnsr_emap(mdata->moment_b, mdata->moment_b, tnsr_mul_n, &beta), goto error);
  REQUIRE(tnsr_emap(tmp_w, tmp_w, tnsr_mul_n, &i_beta), goto error);
  REQUIRE(tnsr_emap(tmp_b, tmp_b, tnsr_mul_n, &i_beta), goto error);
  REQUIRE(tnsr_eadd(mdata->moment_w, mdata->moment_w, tmp_w), goto error);
  REQUIRE(tnsr_eadd(mdata->moment_b, mdata->moment_b, tmp_b), goto error);
  REQUIRE(tnsr_emap(tmp_w, mdata->moment_w, tnsr_sqrt, NULL), goto error);
  REQUIRE(tnsr_emap(tmp_b, mdata->moment_b, tnsr_sqrt, NULL), goto error);
  REQUIRE(tnsr_emap(tmp_w, tmp_w, tnsr_add_n, &epsilon), goto error);
  REQUIRE(tnsr_emap(tmp_b, tmp_b, tnsr_add_n, &epsilon), goto error);
  REQUIRE(tnsr_emap(tmp_w, tmp_w, tnsr_as_divisor, &lr), goto error);
  REQUIRE(tnsr_emap(tmp_b, tmp_b, tnsr_as_divisor, &lr), goto error);
  REQUIRE(tnsr_emul(tmp_w, tmp_w, wgrad), goto error);
  REQUIRE(tnsr_emul(tmp_b, tmp_b, bgrad), goto error);
  REQUIRE(tnsr_eadd(dl->weights, dl->weights, tmp_w), goto error);
  REQUIRE(tnsr_eadd(dl->biases, dl->biases, tmp_b), goto error);

  tnsr_destroy(&tmp_w);
  tnsr_destroy(&tmp_b);
  return true;
error:
  tnsr_destroy(&tmp_w);
  tnsr_destroy(&tmp_b);
  return false;
}

bool dense_layer_sgd_adam(grph_t **g, dense_layer_t *dl) {
  ASSERT(g && *g && dl);
  gen_ctx_t beta1 = {0.9f};
  gen_ctx_t i_beta1 = {0.1f};
  gen_ctx_t beta2 = {0.999f};
  gen_ctx_t i_beta2 = {0.001f};
  gen_ctx_t epsilon = {1e-8f};
  gen_ctx_t lr = {-dl->learning_rate};
  tnsr_t *wgrad = GRPH_NODE_GRAD(*g, dl->weights_id);
  tnsr_t *bgrad = GRPH_NODE_GRAD(*g, dl->biases_id);
  adam_data_t *data = dl->optimizer_data;

  ++data->timestamp;  // Must be incremented first to prevent div by 0.

  gen_ctx_t i_beta1t = {1 - powf(beta1.n, data->timestamp)};
  gen_ctx_t i_beta2t = {1 - powf(beta2.n, data->timestamp)};
  gen_ctx_t sqrt_ib2t = {sqrtf(i_beta2t.n)};
  gen_ctx_t adj_epsilon = {epsilon.n * sqrt_ib2t.n};
  gen_ctx_t step = {lr.n * sqrt_ib2t.n / i_beta1t.n};

  gen_ctx_t pow = {2.0f};
  tnsr_t *tmp1 = tnsr_emap(NULL, wgrad, tnsr_mul_n, &i_beta1);
  tnsr_t *tmp2 = tnsr_emap(NULL, bgrad, tnsr_mul_n, &i_beta1);
  REQUIRE(tmp1 && tmp2, goto error);

  REQUIRE(tnsr_emap(data->moment1_w, data->moment1_w, tnsr_mul_n, &beta1), goto error);
  REQUIRE(tnsr_emap(data->moment1_b, data->moment1_b, tnsr_mul_n, &beta1), goto error);
  REQUIRE(tnsr_eadd(data->moment1_w, data->moment1_w, tmp1), goto error);
  REQUIRE(tnsr_eadd(data->moment1_b, data->moment1_b, tmp2), goto error);

  REQUIRE(tnsr_emap(data->moment2_w, data->moment2_w, tnsr_mul_n, &beta2), goto error);
  REQUIRE(tnsr_emap(data->moment2_b, data->moment2_b, tnsr_mul_n, &beta2), goto error);
  REQUIRE(tnsr_emap(tmp1, wgrad, tnsr_powf, &pow), goto error);
  REQUIRE(tnsr_emap(tmp2, bgrad, tnsr_powf, &pow), goto error);
  REQUIRE(tnsr_emap(tmp1, tmp1, tnsr_mul_n, &i_beta2), goto error);
  REQUIRE(tnsr_emap(tmp2, tmp2, tnsr_mul_n, &i_beta2), goto error);
  REQUIRE(tnsr_eadd(data->moment2_w, data->moment2_w, tmp1), goto error);
  REQUIRE(tnsr_eadd(data->moment2_b, data->moment2_b, tmp2), goto error);

  REQUIRE(tnsr_emap(tmp1, data->moment2_w, tnsr_sqrt, NULL), goto error);
  REQUIRE(tnsr_emap(tmp1, tmp1, tnsr_add_n, &adj_epsilon), goto error);

  // This div-operation is non-standard. Equivalent to b = a / b. Not a /= b.
  REQUIRE(tnsr_ediv(tmp1, data->moment1_w, tmp1), goto error);

  REQUIRE(tnsr_emap(tmp1, tmp1, tnsr_mul_n, &step), goto error);
  REQUIRE(tnsr_eadd(dl->weights, dl->weights, tmp1), goto error);

  REQUIRE(tnsr_emap(tmp2, data->moment2_b, tnsr_sqrt, NULL), goto error);
  REQUIRE(tnsr_emap(tmp2, tmp2, tnsr_add_n, &adj_epsilon), goto error);

  // This div-operation is non-standard. Equivalent to b = a / b. Not a /= b.
  REQUIRE(tnsr_ediv(tmp2, data->moment1_b, tmp2), goto error);
  REQUIRE(tnsr_emap(tmp2, tmp2, tnsr_mul_n, &step), goto error);
  REQUIRE(tnsr_eadd(dl->biases, dl->biases, tmp2), goto error);

  tnsr_destroy(&tmp1);
  tnsr_destroy(&tmp2);
  return true;

error:
  tnsr_destroy(&tmp1);
  tnsr_destroy(&tmp2);
  return false;
}

void dense_layer_sgd_dtor(void **dataptr) {
  (void)dataptr;  // Stateless.
}

void dense_layer_sgd_momentum_dtor(void **dataptr) {
  REQUIRE(dataptr && *dataptr, return);
  momentum_data_t *data = *dataptr;
  tnsr_destroy(&data->moment_w);
  tnsr_destroy(&data->moment_b);
  free(data);
  *dataptr = NULL;
}

void dense_layer_sgd_rms_prop_dtor(void **dataptr) {
  REQUIRE(dataptr && *dataptr, return);
  rms_prop_data_t *data = *dataptr;
  tnsr_destroy(&data->moment_w);
  tnsr_destroy(&data->moment_b);
  free(data);
  *dataptr = NULL;
}

void dense_layer_sgd_adam_dtor(void **dataptr) {
  REQUIRE(dataptr && *dataptr, return);
  adam_data_t *data = *dataptr;
  tnsr_destroy(&data->moment1_w);
  tnsr_destroy(&data->moment1_b);
  tnsr_destroy(&data->moment2_w);
  tnsr_destroy(&data->moment2_b);
  free(data);
  *dataptr = NULL;
}

void dense_layer_dbgprint(dense_layer_t *dl) {
  ASSERT(dl);
  printf("W:\n");
  tnsr_dbgprint(dl->weights);
  printf("B:\n");
  tnsr_dbgprint(dl->biases);
}
