/**
 * model.c
 *
 * BRIEF:
 * Implementation for model.h
 */

#define _CRT_SECURE_NO_WARNINGS

#include <float.h>
#include <math.h>
#include <string.h>
#include <threads.h>

#include "core/graph.h"
#include "core/model.h"
#include "core/network.h"
#include "core/node.h"  // IWYU pragma: export
#include "core/tensor.h"
#include "core/tensor_functions.h"
#include "utils/utils.h"

#define MODEL_LOSS_HISTORY_LENGTH 60
#define MODEL_LOSS_BINS 12

typedef struct {
  uint64_t magicn;
  uint64_t epochs;
  uint64_t network_depth;
  uint64_t batch_size;
  uint64_t data_size;
  uint64_t input_size;
  uint64_t output_size;
  uint64_t epoch_count;
  uint64_t pass_count;
  double training_loss;
  double learning_rate;
  int64_t optimizer_method;
  int64_t loss_function_type;
} model_serial_header_config_t;

typedef struct {
  uint64_t neuron_count;
  int64_t initialization_function;
  int64_t activation_function;
} layer_serial_config_t;

typedef struct {
  uint64_t layer_count;  // Same as network_depth.
  layer_serial_config_t network[];
} model_serial_layer_config_t;

typedef struct {
  // Parameter dump. Arranged in L0[W, B] -> L1[W, B].
  // Refer to serial_layer_config to find out sizes which are just
  // (previous neuron count or input_size) * neuron_count (weights) + neuron_count (bias)
  tnsr_type_t paremeters[];
} model_serial_parameters_t;

static bool model_update_status(
    model_t *model, grph_t **grph, grph_size_t loss_node, size_t epoch_count, size_t pass_count
) {
  ASSERT(model && grph && *grph);
  model->state.epoch_count = epoch_count;
  model->state.pass_count = pass_count;
  model->state.training_loss = TNSR_DATA(GRPH_NODE_DATA(*grph, loss_node), 0, 0);
  return true;
}

static grph_size_t model_forward_pass(model_t *model, grph_t **grph, tnsr_t *input) {
  ASSERT(model && grph && *grph && input);
  grph_size_t n = grph_append_data(grph, input);
  for (size_t i = 0; i < model->config.network_depth; ++i) {
    n = dense_layer_passthrough(grph, model->layers[i], n);
    REQUIRE(n != GRPH_ERR_ID, goto error);
  }
  return n;
error:
  return GRPH_ERR_ID;
}

static grph_size_t model_backward_pass(
    model_t *model, grph_size_t lnode, grph_t **grph, tnsr_t *expected
) {
  ASSERT(grph && *grph && expected && model && lnode != GRPH_ERR_ID);
  grph_size_t expg = grph_append_data(grph, expected);
  REQUIRE(expg != GRPH_ERR_ID, goto error);

  grph_size_t loss = grph_execute(grph, lnode, expg, model->config.loss_function_type);
  REQUIRE(loss != GRPH_ERR_ID, goto error);
  REQUIRE(grph_trace(*grph), goto error);
  return loss;
error:
  return GRPH_ERR_ID;
}

static bool model_optimize(model_t *model, grph_t **grph) {
  ASSERT(model && grph && *grph);
  for (size_t i = 0; i < model->config.network_depth; ++i) {
    REQUIRE(dense_layer_update(grph, model->layers[i]), goto error);
  }
  return true;
error:
  return false;
}

static bool model_fit_one_epoch(model_t *m, size_t epoch_n) {
  ASSERT(m);
  grph_t *graph = NULL;
  tnsr_t *input = NULL;
  tnsr_t *expected = NULL;
  dashboard_config_t dconfig = m->config.dashboard;
  size_t iters = m->config.data_size / m->config.batch_size;
  for (size_t i = 0; i < iters; ++i) {
    bool data_status = m->config.data_callback(
        m->config.batch_size,
        &input,
        &expected,
        m->config.context  // Context pointer.
    );
    REQUIRE(data_status, goto error);
    graph = grph_create(0);
    REQUIRE(graph, goto error);
    for (size_t j = 0; j < m->config.network_depth; ++j) {
      dense_layer_add_to_graph(&graph, m->layers[j]);
    }
    grph_size_t node = model_forward_pass(m, &graph, input);
    REQUIRE(node != GRPH_ERR_ID, goto error);
    grph_size_t loss_node = model_backward_pass(m, node, &graph, expected);
    REQUIRE(loss_node, goto error);
    REQUIRE(model_update_status(m, &graph, loss_node, epoch_n, i), goto error);
    REQUIRE(model_optimize(m, &graph), goto error);
    if (dconfig.show_dashboard && i % dconfig.passes_interval == 0) {
      dconfig.dashboard_callback(
          graph, m, input, GRPH_NODE_DATA(graph, GRPH_NODE_DEPS(graph, loss_node)[0]), expected
      );
    }
    for (size_t j = 0; j < m->config.network_depth; ++j) {
      dense_layer_remove_from_graph(m->layers[j]);
    }
    tnsr_destroy(&expected);
    tnsr_destroy(&input);
    grph_destroy(&graph);
  }
  tnsr_destroy(&expected);
  tnsr_destroy(&input);
  grph_destroy(&graph);
  return true;
error:
  tnsr_destroy(&expected);
  tnsr_destroy(&input);
  grph_destroy(&graph);
  return false;
}

model_t *model_create(model_config_t *config) {
  ASSERT(config);
  model_t *model = malloc(sizeof(model_t) + sizeof(dense_layer_t[config->network_depth]));
  REQUIRE(model, goto error);
  memcpy(&model->config, config, sizeof(model_config_t));
  model->state.epoch_count = 0;
  model->state.pass_count = 0;
  model->state.training_loss = NAN;

  grph_size_t input = config->input_size;
  for (size_t i = 0; i < model->config.network_depth; ++i) {
    model->layers[i] = dense_layer_create(
        input,
        config->network[i].neuron_count,
        config->network[i].initialization_function,
        config->network[i].activation_function,
        config->optimizer_method,
        config->learning_rate
    );
    input = config->network[i].neuron_count;
    REQUIRE(model->layers[i], goto error);
  }
  return model;
error:
  return NULL;
}

void model_destroy(model_t **m) {
  REQUIRE(m && *m, return);
  model_t *model = *m;
  for (size_t i = 0; i < model->config.network_depth; ++i) {
    dense_layer_destroy(&model->layers[i]);
  }
  free(model);
  *m = NULL;
}

/**
 * Will NOT preserve transposed data.
 * Model weights must be placed in row-major contiguous order.
 * NOTE:
 * Uses twice as much RAM as needed. Ignored for now, not like
 * I'm running gigabyte-scale models on a CPU.
 */
bool model_save(model_t *m, char *location) {
  ASSERT(m && location);
  FILE *stream = NULL;
  model_serial_header_config_t scfg = {
      .magicn = 0x4004,
      .epochs = m->config.epochs,
      .network_depth = m->config.network_depth,
      .batch_size = m->config.batch_size,
      .data_size = m->config.data_size,
      .input_size = m->config.input_size,
      .output_size = m->config.output_size,
      .epoch_count = m->state.epoch_count,
      .pass_count = m->state.pass_count,
      .training_loss = m->state.training_loss,
      .learning_rate = m->config.learning_rate,
      .optimizer_method = m->config.optimizer_method,
      .loss_function_type = m->config.loss_function_type
  };
  model_serial_parameters_t *params = NULL;
  size_t lcfg_size = {
      sizeof(model_serial_layer_config_t) + m->config.network_depth * sizeof(layer_serial_config_t)
  };
  model_serial_layer_config_t *lcfg = malloc(lcfg_size);
  size_t tparam_size = 0;
  REQUIRE(lcfg, goto error);
  lcfg->layer_count = m->config.network_depth;
  for (size_t i = 0; i < m->config.network_depth; ++i) {
    layer_config_t layer = m->config.network[i];
    lcfg->network[i] = (layer_serial_config_t){
        .activation_function = layer.activation_function,
        .initialization_function = layer.initialization_function,
        .neuron_count = layer.neuron_count,
    };
    tnsr_t *w = m->layers[i]->weights;
    tnsr_t *b = m->layers[i]->biases;
    tparam_size += TNSR_SHPE(w, 0) * TNSR_SHPE(w, 1);
    tparam_size += TNSR_SHPE(b, 0) * TNSR_SHPE(b, 1);
  }
  params = malloc(tparam_size * sizeof(tnsr_type_t));
  REQUIRE(params, goto error);
  size_t offset = 0;
  for (size_t i = 0; i < m->config.network_depth; ++i) {
    tnsr_t *w = m->layers[i]->weights;
    tnsr_t *b = m->layers[i]->biases;
    size_t cpy_size = TNSR_SHPE(w, 0) * TNSR_SHPE(w, 1) * sizeof(tnsr_type_t);
    memcpy((char *)(params->paremeters) + offset, &TNSR_DATA(w, 0, 0), cpy_size);
    offset += cpy_size;
    cpy_size = TNSR_SHPE(b, 0) * TNSR_SHPE(b, 1) * sizeof(tnsr_type_t);
    memcpy((char *)(params->paremeters) + offset, &TNSR_DATA(b, 0, 0), cpy_size);
    offset += cpy_size;
  }

  stream = fopen(location, "wb");
  REQUIRE(stream, goto error);
  size_t rw = fwrite(&scfg, sizeof(scfg), 1, stream);
  REQUIRE(rw == 1, goto error);
  rw = fwrite(lcfg, lcfg_size, 1, stream);
  REQUIRE(rw == 1, goto error);
  rw = fwrite(params, tparam_size * sizeof(tnsr_type_t), 1, stream);
  REQUIRE(rw == 1, goto error);
  free(lcfg);
  free(params);
  fclose(stream);
  return true;
error:
  free(lcfg);
  free(params);
  if (stream) {
    fclose(stream);
  }
  return false;
}

model_t *model_load(char *location) {
  ASSERT(location);
  model_t *model = NULL;
  dense_layer_t **layers = NULL;
  FILE *stream = fopen(location, "rb");
  REQUIRE(stream, goto error);
  model_serial_header_config_t header = {};
  REQUIRE(fread(&header, sizeof(header), 1, stream), goto error);
  REQUIRE(header.magicn == 0x4004, goto error);
  size_t lcfg_size = {
      sizeof(model_serial_layer_config_t) +  // Layer count.
      sizeof(layer_serial_config_t) * header.network_depth
  };
  model_serial_layer_config_t *lcfg = malloc(lcfg_size);
  REQUIRE(lcfg, goto error);
  REQUIRE(fread(lcfg, lcfg_size, 1, stream) == 1, goto error);

  layers = calloc(header.network_depth, sizeof(dense_layer_t *));
  REQUIRE(layers, goto error);
  size_t prev_nsize = header.input_size;
  for (size_t i = 0; i < header.network_depth; ++i) {
    layer_serial_config_t lc = lcfg->network[i];
    layers[i] = dense_layer_create(
        prev_nsize,
        lc.neuron_count,
        lc.initialization_function,
        lc.activation_function,
        header.optimizer_method,
        header.learning_rate
    );
    REQUIRE(layers[i], goto error);
    size_t sw = {
        TNSR_SHPE(layers[i]->weights, 0) *
        TNSR_SHPE(layers[i]->weights, 1) *  // Weights stored first.
        sizeof(tnsr_type_t)
    };
    size_t sb = {
        TNSR_SHPE(layers[i]->biases, 0) *
        TNSR_SHPE(layers[i]->biases, 1) *  // Biases stored afterwards.
        sizeof(tnsr_type_t)
    };
    REQUIRE(fread(layers[i]->weights->data, sw, 1, stream) == 1, goto error);
    REQUIRE(fread(layers[i]->biases->data, sb, 1, stream) == 1, goto error);
    prev_nsize = lc.neuron_count;
  }
  model = malloc(sizeof(model_t) + sizeof(dense_layer_t *) * header.network_depth);
  REQUIRE(model, goto error);
  model->state.epoch_count = header.epoch_count;
  model->state.pass_count = header.pass_count;
  model->state.training_loss = header.training_loss;
  memcpy(model->layers, layers, sizeof(dense_layer_t *) * header.network_depth);
  model->config.epochs = header.epochs;
  model->config.network_depth = header.network_depth;
  model->config.batch_size = header.batch_size;
  model->config.data_size = header.data_size;
  model->config.input_size = header.input_size;
  model->config.output_size = header.output_size;
  model->config.optimizer_method = header.optimizer_method;
  model->config.learning_rate = header.learning_rate;
  model->config.loss_function_type = header.loss_function_type;
  model->config.data_callback = NULL;
  model->config.context = NULL;

  // Byte-for-byte memcpy not allowed due to standardized 64-bit size in serialization.
  model->config.network = malloc(sizeof(layer_config_t) * model->config.network_depth);
  REQUIRE(model->config.network, goto error);
  for (size_t i = 0; i < model->config.network_depth; ++i) {
    model->config.network[i] =
        (layer_config_t){.activation_function = lcfg->network[i].activation_function,
                         .initialization_function = lcfg->network[i].initialization_function,
                         .neuron_count = lcfg->network[i].neuron_count};
  }

  model->config.dashboard.dashboard_callback = NULL;
  model->config.dashboard.show_dashboard = false;
  model->config.dashboard.passes_interval = 0;
  free(lcfg);
  fclose(stream);
  return model;
error:
  if (layers) {
    for (size_t i = 0; i < header.network_depth; ++i) {
      dense_layer_destroy(&layers[i]);
    }
    free(layers);
  }
  free(lcfg);
  if (model) {
    free(model->config.network);
    free(model);
  }

  if (stream) {
    fclose(stream);
  }
  return NULL;
}

bool model_fit(model_t *m) {
  ASSERT(m);
  REQUIRE(m, goto error);
  for (size_t i = 0; i < m->config.epochs; ++i) {
    REQUIRE(model_fit_one_epoch(m, i), goto error);
  }
  return true;
error:
  return false;
}

tnsr_t *model_infer(model_t *m, tnsr_t *data) {
  ASSERT(m && data);
  grph_t *grph = grph_create(0);
  REQUIRE(grph, goto error);
  for (size_t j = 0; j < m->config.network_depth; ++j) {
    dense_layer_add_to_graph(&grph, m->layers[j]);
  }
  grph_size_t raw = model_forward_pass(m, &grph, data);
  REQUIRE(raw, goto error);
  tnsr_t *raw_data = GRPH_NODE_DATA(grph, raw);
  tnsr_t *result = tnsr_emap(NULL, raw_data, tnsr_cpy, NULL);
  REQUIRE(result, goto error);
  grph_destroy(&grph);
  return result;
error:
  grph_destroy(&grph);
  tnsr_destroy(&result);
  return NULL;
}

/// NOTE: Needs refactoring to smaller helper functions.
void model_generic_dashboard(
    const grph_t *graph,
    const model_t *model,
    const tnsr_t *input,
    const tnsr_t *output,
    const tnsr_t *expected
) {
  thread_local static tnsr_type_t loss_history[MODEL_LOSS_HISTORY_LENGTH] = {};
  thread_local static size_t loss_boundary = 0;
  thread_local static tnsr_type_t max_loss = 0.0f;
  thread_local static tnsr_type_t min_loss = FLT_MAX;
  thread_local static size_t max_loss_i = 0;
  thread_local static size_t min_loss_i = 0;
  thread_local static bool initialized = false;

  ASSERT(model && input && output && expected);

  if (!initialized) {
    for (size_t i = 0; i < MODEL_LOSS_HISTORY_LENGTH; ++i) {
      loss_history[i] = FLT_MAX;
    }
    initialized = true;
  }
  (void)input;
  (void)output;
  (void)expected;

  printf("EPOCH COUNT: %5llu\n", model->state.epoch_count);
  printf("PASS COUNT: %6llu\n", model->state.pass_count);
  printf("LOSS: %+11.2f%% \n", model->state.training_loss * 100.0f);
  printf("ACCURACY: %+7.2f%% \n", (1.0f - model->state.training_loss) * 100.0f);

  bool nmax = loss_boundary == max_loss_i;
  bool nmin = loss_boundary == min_loss_i;
  tnsr_type_t nmax_loss = 0.0f;
  tnsr_type_t nmin_loss = FLT_MAX;
  for (size_t i = (loss_boundary + 1) % MODEL_LOSS_HISTORY_LENGTH;
       (nmax || nmin) && i != loss_boundary;
       i = (i + 1) % MODEL_LOSS_HISTORY_LENGTH) {
    if (nmax && nmax_loss < loss_history[i] && loss_history[i] != FLT_MAX) {
      nmax_loss = loss_history[i];
      max_loss_i = i;
    }
    if (nmin && nmin_loss > loss_history[i]) {
      nmin_loss = loss_history[i];
      min_loss_i = i;
    }
  }
  if (nmax) {
    max_loss = nmax_loss;
  }
  if (nmin) {
    min_loss = nmin_loss;
  }
  loss_history[loss_boundary] = model->state.training_loss;
  if (max_loss < loss_history[loss_boundary]) {
    max_loss = loss_history[loss_boundary];
    max_loss_i = loss_boundary;
  }
  if (min_loss > loss_history[loss_boundary]) {
    min_loss = loss_history[loss_boundary];
    min_loss_i = loss_boundary;
  }
  char loss_graph[MODEL_LOSS_BINS][MODEL_LOSS_HISTORY_LENGTH] = {};
  size_t l = loss_boundary;
  size_t gid = MODEL_LOSS_HISTORY_LENGTH - 1;
  do {
    int bin = (int)(loss_history[l] / max_loss * MODEL_LOSS_BINS);
    bin = bin >= MODEL_LOSS_BINS ? MODEL_LOSS_BINS - 1 : bin;
    if (loss_history[l] != FLT_MAX) {
      for (int i = 0; i < bin; ++i) {
        loss_graph[MODEL_LOSS_BINS - i - 1][gid] = '.';
      }
    }
    l = l ? l - 1 : MODEL_LOSS_HISTORY_LENGTH - 1;
    --gid;
  } while (l != loss_boundary);
  for (size_t i = 0; i <= MODEL_LOSS_BINS; ++i) {
    if (i == 0) {
      printf("\n%+7.2f%%\n", max_loss * 100.0f);
    } else if (i == MODEL_LOSS_BINS) {
      printf("%+7.2f%% ", min_loss * 100.0f);
      for (size_t j = 0; j < MODEL_LOSS_HISTORY_LENGTH - 3; ++j) {
        printf("-");
      }
    } else {
      printf("    | ");
      for (size_t j = 0; j < MODEL_LOSS_HISTORY_LENGTH; ++j) {
        printf("%c", loss_graph[i][j] ? loss_graph[i][j] : ' ');
      }
      printf("\n");
    }
  }
  printf("\n\n   +-----------------+------------------+-------------------+----------+\n");
  printf("   | LAYER ID.       | GRADIENT L2 NORM | PARAMETER L2 NORM | SPARSITY |\n");
  printf("   +-----------------+------------------+-------------------+----------+\n");
  for (size_t i = 0; i < model->config.network_depth; ++i) {
    dense_layer_t *layer = model->layers[i];
    tnsr_t *wgrad = GRPH_NODE_GRAD(graph, layer->weights_id);
    tnsr_t *bgrad = GRPH_NODE_GRAD(graph, layer->biases_id);
    tnsr_t *wlayer = GRPH_NODE_DATA(graph, layer->weights_id);
    tnsr_t *blayer = GRPH_NODE_DATA(graph, layer->biases_id);

    tnsr_type_t gwsquared_sum = 0.0f;
    tnsr_type_t gwsparsity = 0.0f;
    tnsr_type_t wsquared_sum = 0.0f;
    for (grph_size_t j = 0; j < TNSR_SHPE(wgrad, 0); ++j) {
      for (grph_size_t k = 0; k < TNSR_SHPE(wgrad, 1); ++k) {
        gwsquared_sum += TNSR_DATA(wgrad, j, k) * TNSR_DATA(wgrad, j, k);
        gwsparsity += fabsf(TNSR_DATA(wgrad, j, k)) > 1e-8f;
        wsquared_sum += TNSR_DATA(wlayer, j, k) * TNSR_DATA(wlayer, j, k);
      }
    }
    tnsr_type_t gbsquared_sum = 0.0f;
    tnsr_type_t gbsparsity = 0.0f;
    tnsr_type_t bsquared_sum = 0.0f;
    for (grph_size_t j = 0; j < TNSR_SHPE(bgrad, 0); ++j) {
      for (grph_size_t k = 0; k < TNSR_SHPE(bgrad, 1); ++k) {
        gbsquared_sum += TNSR_DATA(bgrad, j, k) * TNSR_DATA(bgrad, j, k);
        gbsparsity += fabsf(TNSR_DATA(bgrad, j, k)) > 1e-8f;
        bsquared_sum += TNSR_DATA(blayer, j, k) * TNSR_DATA(blayer, j, k);
      }
    }
    gwsparsity /= TNSR_SHPE(wgrad, 0) * TNSR_SHPE(wgrad, 1);
    gbsparsity /= TNSR_SHPE(bgrad, 0) * TNSR_SHPE(bgrad, 1);
    gwsparsity *= 100;
    gbsparsity *= 100;
    gwsparsity = 100 - gwsparsity;
    gbsparsity = 100 - gbsparsity;

    tnsr_type_t gmeanl2norm = (sqrtf(gwsquared_sum) + sqrtf(gbsquared_sum)) / 2;
    tnsr_type_t gsparsity = (gwsparsity + gbsparsity) / 2;
    tnsr_type_t pmeanl2norm = (sqrtf(wsquared_sum) + sqrtf(bsquared_sum)) / 2;
    char fid[32] = {};
    switch (layer->function_type) {
      case NDTYPE_ESIGMOID:
        memcpy(fid, "SIGMD", sizeof("SIGMD"));
        break;
      case NDTYPE_ETANH:
        memcpy(fid, "TANH", sizeof("TANH"));
        break;
      case NDTYPE_ERELU:
        memcpy(fid, "RELU", sizeof("RELU"));
        break;
      case NDTYPE_ELEAKYRELU:
        memcpy(fid, "LRELU", sizeof("LRELU"));
        break;
      case NDTYPE_SOFTMAX:
        memcpy(fid, "SFTMX", sizeof("SFTMX"));
        break;
      default:
        ASSERT(false);
        break;
    }
    printf(
        "   | #L%llu-N%04u-%-5s | %-16.4e | %-17.4e | %-7.2f%% |\n",
        i,
        TNSR_SHPE(blayer, 1),
        fid,
        gmeanl2norm,
        pmeanl2norm,
        gsparsity
    );
    printf("   +-----------------+------------------+-------------------+----------+\n");
  }
  loss_boundary = (loss_boundary + 1) % MODEL_LOSS_HISTORY_LENGTH;
}
