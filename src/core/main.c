/**
 * main.c
 *
 * MNIST demo.
 * Converges ~3 epochs.
 * 92-95% accuracy
 */

#define _CRT_SECURE_NO_WARNINGS

#include <Windows.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>

#include "core/graph.h"
#include "core/model.h"

typedef struct {
  bool use_testing;
  uint32_t mn_train_lbl;
  uint32_t mn_train_img;
  uint32_t ne_train_lbl;
  uint32_t ne_train_img;
  uint32_t mn_test_lbl;
  uint32_t mn_test_img;
  uint32_t ne_test_lbl;
  uint32_t ne_test_img;
  uint8_t *train_images;
  uint8_t *train_labels;
  uint8_t *test_images;
  uint8_t *test_labels;
} callback_ctx_t;

void mnist_dash(
    const grph_t *g, const model_t *m, const tnsr_t *in, const tnsr_t *out, const tnsr_t *expected
) {
  printf("INPUT IMAGE:\n");
  for (size_t i = 0; i < 28; ++i) {
    for (size_t j = 0; j < 28; ++j) {
      printf("%c ", TNSR_DATA(in, 0, i * 28 + j) < 0.5 ? ' ' : '#');
    }
    printf("\n");
  }
  printf("\n\n");
  model_generic_dashboard(g, m, in, out, expected);
  printf("\033[H");
}

bool mnist_data(size_t batch_size, tnsr_t **in, tnsr_t **expected, void *ctx) {
  callback_ctx_t *context = ctx;
  tnsr_t *label = TNSR_MATRIX(batch_size, 10);
  tnsr_t *image = TNSR_MATRIX(batch_size, 28 * 28);
  uint8_t *target_img = context->use_testing ? context->test_images : context->train_images;
  uint8_t *target_lbl = context->use_testing ? context->test_labels : context->train_labels;
  size_t ne = context->use_testing ? context->ne_test_img : context->ne_train_img;
  for (size_t i = 0; i < batch_size; ++i) {
    size_t idx = rand() % ne;
    uint8_t lbl = target_lbl[idx];
    TNSR_DATA(label, i, lbl) = 1.0f;
    for (size_t j = 0; j < 28 * 28; ++j) {
      TNSR_DATA(image, i, j) = target_img[idx * 28 * 28 + j] / 255.0f;
    }
  }
  *in = image;
  *expected = label;
  return true;
}

int main() {
  /* ---------------------------------- Setup --------------------------------- */
  system("cls");
  callback_ctx_t ctx = {};
  FILE *tr_imgstream = fopen("C:/dev/repositories/nn-c/data/train-images.idx3-ubyte", "rb");
  FILE *tr_lblstream = fopen("C:/dev/repositories/nn-c/data/train-labels.idx1-ubyte", "rb");
  FILE *tst_imgstream = fopen("C:/dev/repositories/nn-c/data/t10k-images.idx3-ubyte", "rb");
  FILE *tst_lblstream = fopen("C:/dev/repositories/nn-c/data/t10k-labels.idx1-ubyte", "rb");
  if (!tr_imgstream || !tr_lblstream || !tst_imgstream || !tst_lblstream) {
    goto error;
  }
  uint32_t *mn[] = {&ctx.mn_train_img, &ctx.mn_train_lbl, &ctx.mn_test_img, &ctx.mn_test_lbl};
  uint32_t *ne[] = {&ctx.ne_train_img, &ctx.ne_train_lbl, &ctx.ne_test_img, &ctx.ne_test_lbl};
  uint8_t **dt[] = {&ctx.train_images, &ctx.train_labels, &ctx.test_images, &ctx.test_labels};
  FILE *streams[] = {tr_imgstream, tr_lblstream, tst_imgstream, tst_lblstream};
  for (size_t i = 0; i < sizeof(streams) / sizeof(FILE *); ++i) {
    if (fread(mn[i], sizeof(uint32_t), 1, streams[i]) != 1) {
      goto error;
    }
    if (fread(ne[i], sizeof(uint32_t), 1, streams[i]) != 1) {
      goto error;
    }
    if (i % 2 == 0) {
      uint32_t b = 0;  // Consume row/cols. Hardcoded to be 28*28.
      fread(&b, sizeof(uint32_t), 1, streams[i]);
      fread(&b, sizeof(uint32_t), 1, streams[i]);
    }
    *mn[i] = _bswap(*mn[i]);
    *ne[i] = _bswap(*ne[i]);
    size_t cpos = ftell(streams[i]);
    fseek(streams[i], 0, SEEK_END);
    size_t epos = ftell(streams[i]);
    fseek(streams[i], cpos, SEEK_SET);
    *dt[i] = malloc(epos - cpos);
    if (!*dt[i]) {
      goto error;
    }
    if (fread(*dt[i], sizeof(uint8_t), epos - cpos, streams[i]) != epos - cpos) {
      goto error;
    }
  }

  /* -------------------------------- Training -------------------------------- */

  dashboard_config_t dash = {
      .show_dashboard = true,
      .passes_interval = 1024,  // Show dashboard every 1024 mini-batches.
      .dashboard_callback = mnist_dash
  };
  layer_config_t layers[] = {
      (layer_config_t){64, INIT_HE, NDTYPE_ELEAKYRELU},
      (layer_config_t){64, INIT_HE, NDTYPE_ELEAKYRELU},
      (layer_config_t){10, INIT_GLOROT, NDTYPE_SOFTMAX}
  };
  size_t lsize = sizeof(layers) / sizeof(layer_config_t);
  model_config_t config = {
      .epochs = 3,
      .network_depth = lsize,
      .batch_size = 8,
      .data_size = 60000,
      .network = layers,
      .dashboard = dash,
      .input_size = 28 * 28,
      .output_size = 10,
      .optimizer_method = OPT_SGD_MOMENTUM,
      .learning_rate = 0.01f,
      .loss_function_type = NDTYPE_CATEGORICAL_CROSS_ENTROPY_LOSS,
      .data_callback = mnist_data,
      .context = &ctx,
  };
  model_t *model = model_create(&config);
  if (!model) {
    goto error;
  }
  if (!model_fit(model)) {
    goto error;
  }
  if (!model_save(model, "C:/dev/repositories/nn-c/mnist32.weights")) {
    goto error;
  }
  /* -------------------------------- Inference ------------------------------- */

  system("cls");
  model_t *model_inf = model_load("C:/dev/repositories/nn-c/mnist32.weights");
  if (!model_inf) {
    goto error;
  }
  ctx.use_testing = true;
  for (size_t i = 0; i < 50; ++i) {
    bool rt = true;
    tnsr_t *in = NULL;
    tnsr_t *expected = NULL;
    if (!mnist_data(1, &in, &expected, &ctx)) {
      goto end;
    }
    tnsr_t *out = model_infer(model_inf, in);
    if (!out) {
      goto end;
    }
    tnsr_type_t expconf = 0.0f;
    size_t exppred = 0;
    tnsr_type_t conf = 0.0f;
    size_t pred = 0;
    for (size_t j = 0; j < 10; ++j) {
      if (TNSR_DATA(out, 0, j) > conf) {
        pred = j;
        conf = TNSR_DATA(out, 0, j);
      }
      if (TNSR_DATA(expected, 0, j) > expconf) {
        exppred = j;
        expconf = TNSR_DATA(expected, 0, j);
      }
    }
    printf("EXPECTED: %zu | %.2f ? NETWORK PREDICTED: %zu | %.2f\n", exppred, expconf, pred, conf);
  end:
    tnsr_destroy(&in);
    tnsr_destroy(&expected);
    tnsr_destroy(&out);
    if (!rt) {
      goto error;
    }
    Sleep(100);
  }
  fclose(tr_imgstream);
  fclose(tr_lblstream);
  fclose(tst_imgstream);
  fclose(tst_lblstream);
  free(ctx.train_images);
  free(ctx.train_labels);
  free(ctx.test_images);
  free(ctx.test_labels);
  model_destroy(&model);
  model_destroy(&model_inf);
  return EXIT_SUCCESS;
error:
  if (tr_imgstream)
    fclose(tr_imgstream);
  if (tr_lblstream) {
    fclose(tr_lblstream);
  }
  if (tst_imgstream) {
    fclose(tst_imgstream);
  }
  if (tst_lblstream) {
    fclose(tst_lblstream);
  }
  free(ctx.train_images);
  free(ctx.train_labels);
  free(ctx.test_images);
  free(ctx.test_labels);
  model_destroy(&model);
  model_destroy(&model_inf);
  return EXIT_FAILURE;
}