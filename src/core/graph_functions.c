/**
 * graph_functions.c
 *
 * BRIEF:
 * Implementation for graph_functions.h
 */

#include <stdbool.h>
#include <stddef.h>

#include "core/graph.h"
#include "core/graph_functions.h"
#include "core/tensor.h"
#include "core/tensor_functions.h"
#include "utils/utils.h"

static grph_size_t input_req[] = {_GRPH_INPUT_TBLE};

static grph_size_t output_size[] = {
    [NDTYPE_DATA] = OUTSIZE_INDEPENDENT,
    [NDTYPE_TRANSPOSE] = OUTSIZE_TRANSPOSED,
    [NDTYPE_CONTRACT] = OUTSIZE_DEP_ON_A0 | OUTSIZE_DEP_ON_B1,
    [NDTYPE_EADD] = OUTSIZE_DEP_SAMEAS,
    [NDTYPE_ESUB] = OUTSIZE_DEP_SAMEAS,
    [NDTYPE_EMUL] = OUTSIZE_DEP_SAMEAS,
    [NDTYPE_EDIV] = OUTSIZE_DEP_SAMEAS,
    [NDTYPE_ESIGMOID] = OUTSIZE_DEP_SAMEAS,
    [NDTYPE_ERELU] = OUTSIZE_DEP_SAMEAS,
    [NDTYPE_ELEAKYRELU] = OUTSIZE_DEP_SAMEAS,
    [NDTYPE_MSE] = OUTSIZE_SCALAR,
    [NDTYPE_CROSS_ENTROPY_LOSS] = OUTSIZE_SCALAR,
    [NDTYPE_SOFTMAX] = OUTSIZE_DEP_SAMEAS,
};

node_t *node_create(grph_t *g, tnsr_t *data, grph_size_t a, grph_size_t b, node_type_t type) {
  ASSERT(g);
  bool transient = false;
  grph_size_t ndependencies = input_req[type];
  tnsr_t *node_data = NULL;
  tnsr_t *node_grad = NULL;

  switch (output_size[type]) {
    case OUTSIZE_INDEPENDENT: {
      ASSERT(data && a == GRPH_NO_INPUT_ID && b == GRPH_NO_INPUT_ID);
      node_data = data;
      node_grad = tnsr_create(TNSR_SHPE(data, 0), TNSR_SHPE(data, 1));
      REQUIRE(node_grad, goto error);
      transient = false;
      break;
    }
    case OUTSIZE_TRANSPOSED: {
      ASSERT(!data && a != GRPH_NO_INPUT_ID && b == GRPH_NO_INPUT_ID);
      const tnsr_t *a_tnsr = GRPH_NODE_DATA(g, a);
      node_data = tnsr_create(TNSR_SHPE(a_tnsr, 1), TNSR_SHPE(a_tnsr, 0));
      node_grad = tnsr_create(TNSR_SHPE(a_tnsr, 1), TNSR_SHPE(a_tnsr, 0));
      REQUIRE(node_data && node_grad, goto error);
      transient = true;
      break;
    }
    case OUTSIZE_SCALAR: {
      ASSERT(!data);
      node_data = TNSR_SCALAR();
      node_grad = TNSR_SCALAR();
      REQUIRE(node_data && node_grad, goto error);
      transient = true;
      break;
    }
    case OUTSIZE_DEP_SAMEAS: {
      ASSERT(!data && a != GRPH_NO_INPUT_ID);
      const tnsr_t *a_tnsr = GRPH_NODE_DATA(g, a);
      node_data = tnsr_create(TNSR_SHPE(a_tnsr, 0), TNSR_SHPE(a_tnsr, 1));
      node_grad = tnsr_create(TNSR_SHPE(a_tnsr, 0), TNSR_SHPE(a_tnsr, 1));
      REQUIRE(node_data && node_grad, goto error);
      transient = true;
      break;
    }
    case OUTSIZE_DEP_ON_A0 | OUTSIZE_DEP_ON_B1: {
      ASSERT(!data && a != GRPH_NO_INPUT_ID && b != GRPH_NO_INPUT_ID);
      const tnsr_t *a_tnsr = GRPH_NODE_DATA(g, a);
      const tnsr_t *b_tnsr = GRPH_NODE_DATA(g, b);
      node_data = tnsr_create(TNSR_SHPE(a_tnsr, 0), TNSR_SHPE(b_tnsr, 1));
      node_grad = tnsr_create(TNSR_SHPE(a_tnsr, 0), TNSR_SHPE(b_tnsr, 1));
      REQUIRE(node_data && node_grad, goto error);
      transient = true;
      break;
    }
    default: {
      ASSERT(false);
      break;  // Unreachable.
    }
  }
  node_t *node = malloc(sizeof(node_t) + sizeof(grph_size_t[NODE_INIT_DEP_CPCTY]));
  REQUIRE(node, goto error);

  node->n_deps_capacity = NODE_INIT_DEP_CPCTY;
  node->n_dependencies = ndependencies;
  node->dependencies[0] = a;
  node->dependencies[1] = b;
  node->data = node_data;
  node->grad = node_grad;
  node->type = type;
  node->transient = transient;

  return node;

error:
  if (node_data && !data) {
    free(node_data);
  }
  if (node_grad) {
    free(node_grad);
  }
  return NULL;
}

void node_destroy(node_t **n) {
  REQUIRE(n && *n, return);
  tnsr_destroy(&(*n)->data);
  tnsr_destroy(&(*n)->grad);
  free(*n);
  *n = NULL;
}

node_t *node_transpose(grph_t *g, grph_size_t a, grph_size_t b) {
  ASSERT(g && a != GRPH_NO_INPUT_ID && b == GRPH_NO_INPUT_ID);
  node_t *node = node_create(g, NULL, a, b, NDTYPE_TRANSPOSE);
  REQUIRE(node, goto error);
  REQUIRE(tnsr_transpose(node->data, GRPH_NODE_DATA(g, a)), goto error);
  return node;
error:
  node_destroy(&node);
  return NULL;
}

node_t *node_contract(grph_t *g, grph_size_t a, grph_size_t b) {
  ASSERT(g && a != GRPH_NO_INPUT_ID && b != GRPH_NO_INPUT_ID);
  node_t *node = node_create(g, NULL, a, b, NDTYPE_CONTRACT);
  REQUIRE(node, goto error);
  REQUIRE(tnsr_contract(node->data, GRPH_NODE_DATA(g, a), GRPH_NODE_DATA(g, b)), goto error);
  return node;
error:
  node_destroy(&node);
  return NULL;
}

node_t *node_eadd(grph_t *g, grph_size_t a, grph_size_t b) {
  ASSERT(g && a != GRPH_NO_INPUT_ID && b != GRPH_NO_INPUT_ID);
  node_t *node = node_create(g, NULL, a, b, NDTYPE_EADD);
  REQUIRE(node, goto error);
  REQUIRE(tnsr_eadd(node->data, GRPH_NODE_DATA(g, a), GRPH_NODE_DATA(g, b)), goto error);
  return node;
error:
  node_destroy(&node);
  return NULL;
}

node_t *node_esub(grph_t *g, grph_size_t a, grph_size_t b) {
  ASSERT(g && a != GRPH_NO_INPUT_ID && b != GRPH_NO_INPUT_ID);
  node_t *node = node_create(g, NULL, a, b, NDTYPE_ESUB);
  REQUIRE(node, goto error);
  REQUIRE(tnsr_esub(node->data, GRPH_NODE_DATA(g, a), GRPH_NODE_DATA(g, b)), goto error);
  return node;
error:
  node_destroy(&node);
  return NULL;
}

node_t *node_emul(grph_t *g, grph_size_t a, grph_size_t b) {
  ASSERT(g && a != GRPH_NO_INPUT_ID && b != GRPH_NO_INPUT_ID);
  node_t *node = node_create(g, NULL, a, b, NDTYPE_EMUL);
  REQUIRE(node, goto error);
  REQUIRE(tnsr_emul(node->data, GRPH_NODE_DATA(g, a), GRPH_NODE_DATA(g, b)), goto error);
  return node;
error:
  node_destroy(&node);
  return NULL;
}

node_t *node_ediv(grph_t *g, grph_size_t a, grph_size_t b) {
  ASSERT(g && a != GRPH_NO_INPUT_ID && b != GRPH_NO_INPUT_ID);
  node_t *node = node_create(g, NULL, a, b, NDTYPE_EDIV);
  REQUIRE(node, goto error);
  REQUIRE(tnsr_ediv(node->data, GRPH_NODE_DATA(g, a), GRPH_NODE_DATA(g, b)), goto error);
  return node;
error:
  node_destroy(&node);
  return NULL;
}

node_t *node_esigmoid(grph_t *g, grph_size_t a, grph_size_t b) {
  ASSERT(g && a != GRPH_NO_INPUT_ID && b == GRPH_NO_INPUT_ID);
  node_t *node = node_create(g, NULL, a, b, NDTYPE_ESIGMOID);
  REQUIRE(node, goto error);
  REQUIRE(tnsr_emap(node->data, GRPH_NODE_DATA(g, a), sigmoid, NULL), goto error);
  return node;
error:
  node_destroy(&node);
  return NULL;
}

node_t *node_erelu(grph_t *g, grph_size_t a, grph_size_t b) {
  ASSERT(g && a != GRPH_NO_INPUT_ID && b == GRPH_NO_INPUT_ID);
  node_t *node = node_create(g, NULL, a, b, NDTYPE_ERELU);
  REQUIRE(node, goto error);
  REQUIRE(tnsr_emap(node->data, GRPH_NODE_DATA(g, a), relu, NULL), goto error);
  return node;
error:
  node_destroy(&node);
  return NULL;
}

node_t *node_eleakyrelu(grph_t *g, grph_size_t a, grph_size_t b) {
  ASSERT(g && a != GRPH_NO_INPUT_ID && b == GRPH_NO_INPUT_ID);
  node_t *node = node_create(g, NULL, a, b, NDTYPE_ELEAKYRELU);
  REQUIRE(node, goto error);
  REQUIRE(tnsr_emap(node->data, GRPH_NODE_DATA(g, a), leaky_relu, NULL), goto error);
  return node;
error:
  node_destroy(&node);
  return NULL;
}

node_t *node_mse(grph_t *g, grph_size_t a, grph_size_t b) {
  ASSERT(g && a != GRPH_NO_INPUT_ID && b != GRPH_NO_INPUT_ID);
  tnsr_t *diff = NULL;
  tnsr_t *sum = NULL;
  node_t *node = node_create(g, NULL, a, b, NDTYPE_MSE);
  REQUIRE(node, goto error);
  diff = tnsr_esub(NULL, GRPH_NODE_DATA(g, a), GRPH_NODE_DATA(g, b));
  REQUIRE(diff, goto error);
  REQUIRE(tnsr_emap(diff, diff, pow_2, NULL), goto error);
  sum = tnsr_sum_over_axis(NULL, diff, 1);
  REQUIRE(sum, goto error);
  REQUIRE(tnsr_mean(node->data, sum), goto error);

  tnsr_destroy(&sum);
  tnsr_destroy(&diff);
  return node;
error:
  tnsr_destroy(&sum);
  tnsr_destroy(&diff);
  node_destroy(&node);
  return NULL;
}

node_t *node_cross_entropy_loss(grph_t *g, grph_size_t a, grph_size_t b) {
  ASSERT(g && a != GRPH_NO_INPUT_ID && b != GRPH_NO_INPUT_ID);

  tnsr_t *y_logp = NULL;
  tnsr_t *sum = NULL;
  node_t *node = node_create(g, NULL, a, b, NDTYPE_CROSS_ENTROPY_LOSS);
  REQUIRE(node, goto error);
  y_logp = tnsr_emap(NULL, GRPH_NODE_DATA(g, a), ln, NULL);
  REQUIRE(y_logp, goto error);
  REQUIRE(tnsr_emul(y_logp, y_logp, GRPH_NODE_DATA(g, b)), goto error);
  sum = tnsr_sum_over_axis(NULL, y_logp, 1);
  REQUIRE(sum, goto error);
  REQUIRE(tnsr_mean(node->data, sum), goto error);
  REQUIRE(tnsr_emap(node->data, node->data, mul_neg1, NULL), goto error);

  tnsr_destroy(&y_logp);
  tnsr_destroy(&sum);
  return node;
error:
  tnsr_destroy(&y_logp);
  tnsr_destroy(&sum);
  node_destroy(&node);
  return NULL;
}

node_t *node_softmax(grph_t *g, grph_size_t a, grph_size_t b) {
  ASSERT(g && a != GRPH_NO_INPUT_ID && b == GRPH_NO_INPUT_ID);
  tnsr_t *max = NULL;
  tnsr_t *sum = NULL;
  node_t *node = node_create(g, NULL, a, b, NDTYPE_SOFTMAX);
  REQUIRE(node, goto error);

  max = tnsr_max_over_axis(NULL, GRPH_NODE_DATA(g, a), 1);
  REQUIRE(max, goto error);
  REQUIRE(tnsr_esub(node->data, GRPH_NODE_DATA(g, a), max), goto error);

  REQUIRE(tnsr_emap(node->data, node->data, euler, NULL), goto error);
  sum = tnsr_sum_over_axis(NULL, node->data, 1);
  REQUIRE(sum, goto error);
  REQUIRE(tnsr_ediv(node->data, node->data, sum), goto error);

  tnsr_destroy(&max);
  tnsr_destroy(&sum);
  return node;

error:
  tnsr_destroy(&sum);
  tnsr_destroy(&max);
  node_destroy(&node);
  return NULL;
}

bool node_transpose_dx(grph_t *g, grph_size_t a) {
  ASSERT(g && a != GRPH_NO_INPUT_ID && GRPH_NODE_TYPE(g, a) == NDTYPE_TRANSPOSE);
  tnsr_t *grad_a_dep0 = GRPH_NODE_GRAD(g, GRPH_NODE_DEPS(g, a)[0]);
  tnsr_t *local_deriv = tnsr_transpose(NULL, GRPH_NODE_GRAD(g, a));
  REQUIRE(local_deriv, goto error);
  REQUIRE(tnsr_eadd(grad_a_dep0, grad_a_dep0, local_deriv), goto error);
  tnsr_destroy(&local_deriv);
  return true;
error:
  return false;
}

bool node_contract_dx(grph_t *g, grph_size_t a) {
  ASSERT(g && a != GRPH_NO_INPUT_ID && GRPH_NODE_TYPE(g, a) == NDTYPE_CONTRACT);
  tnsr_t *grad_a_dep0 = GRPH_NODE_GRAD(g, GRPH_NODE_DEPS(g, a)[0]);
  tnsr_t *grad_a_dep1 = GRPH_NODE_GRAD(g, GRPH_NODE_DEPS(g, a)[1]);

  tnsr_t *grad_a0 = NULL;
  tnsr_t *grad_a1 = NULL;
  tnsr_t *dep0_t = tnsr_transpose(NULL, GRPH_NODE_DATA(g, GRPH_NODE_DEPS(g, a)[0]));
  tnsr_t *dep1_t = tnsr_transpose(NULL, GRPH_NODE_DATA(g, GRPH_NODE_DEPS(g, a)[1]));
  REQUIRE(dep0_t && dep1_t, goto error);

  grad_a0 = tnsr_contract(NULL, GRPH_NODE_GRAD(g, a), dep1_t);
  grad_a1 = tnsr_contract(NULL, dep0_t, GRPH_NODE_GRAD(g, a));
  REQUIRE(grad_a0 && grad_a1, goto error);

  REQUIRE(tnsr_eadd(grad_a_dep0, grad_a_dep0, grad_a0), goto error);
  REQUIRE(tnsr_eadd(grad_a_dep1, grad_a_dep1, grad_a1), goto error);

  tnsr_destroy(&dep0_t);
  tnsr_destroy(&dep1_t);
  tnsr_destroy(&grad_a0);
  tnsr_destroy(&grad_a1);
  return true;
error:
  tnsr_destroy(&grad_a0);
  tnsr_destroy(&grad_a1);
  tnsr_destroy(&dep0_t);
  tnsr_destroy(&dep1_t);
  return false;
}

bool node_eadd_dx(grph_t *g, grph_size_t a) {
  ASSERT(g && a != GRPH_NO_INPUT_ID && GRPH_NODE_TYPE(g, a) == NDTYPE_EADD);
  tnsr_t *grad_a_dep0 = GRPH_NODE_GRAD(g, GRPH_NODE_DEPS(g, a)[0]);
  tnsr_t *grad_a_dep1 = GRPH_NODE_GRAD(g, GRPH_NODE_DEPS(g, a)[1]);

  REQUIRE(tnsr_eadd(grad_a_dep0, grad_a_dep0, GRPH_NODE_GRAD(g, a)), goto error);
  REQUIRE(tnsr_eadd(grad_a_dep1, grad_a_dep1, GRPH_NODE_GRAD(g, a)), goto error);
  return true;
error:
  return false;
}

bool node_esub_dx(grph_t *g, grph_size_t a) {
  ASSERT(g && a != GRPH_NO_INPUT_ID && GRPH_NODE_TYPE(g, a) == NDTYPE_ESUB);
  tnsr_t *grad_a_dep0 = GRPH_NODE_GRAD(g, GRPH_NODE_DEPS(g, a)[0]);
  tnsr_t *grad_a_dep1 = GRPH_NODE_GRAD(g, GRPH_NODE_DEPS(g, a)[1]);

  REQUIRE(tnsr_eadd(grad_a_dep0, grad_a_dep0, GRPH_NODE_GRAD(g, a)), goto error);
  REQUIRE(tnsr_esub(grad_a_dep1, grad_a_dep1, GRPH_NODE_GRAD(g, a)), goto error);
  return true;
error:
  return false;
}

bool node_emul_dx(grph_t *g, grph_size_t a) {
  ASSERT(g && a != GRPH_NO_INPUT_ID && GRPH_NODE_TYPE(g, a) == NDTYPE_EMUL);
  tnsr_t *grad_a_dep0 = GRPH_NODE_GRAD(g, GRPH_NODE_DEPS(g, a)[0]);
  tnsr_t *grad_a_dep1 = GRPH_NODE_GRAD(g, GRPH_NODE_DEPS(g, a)[1]);
  tnsr_t *data_a_dep0 = GRPH_NODE_DATA(g, GRPH_NODE_DEPS(g, a)[0]);
  tnsr_t *data_a_dep1 = GRPH_NODE_DATA(g, GRPH_NODE_DEPS(g, a)[1]);

  tnsr_t *dep0_inter = tnsr_emul(NULL, data_a_dep1, GRPH_NODE_GRAD(g, a));
  tnsr_t *dep1_inter = tnsr_emul(NULL, data_a_dep0, GRPH_NODE_GRAD(g, a));
  REQUIRE(dep0_inter && dep1_inter, goto error);

  REQUIRE(tnsr_eadd(grad_a_dep0, grad_a_dep0, dep0_inter), goto error);
  REQUIRE(tnsr_eadd(grad_a_dep1, grad_a_dep1, dep1_inter), goto error);
  tnsr_destroy(&dep0_inter);
  tnsr_destroy(&dep1_inter);
  return true;
error:
  tnsr_destroy(&dep0_inter);
  tnsr_destroy(&dep1_inter);
  return false;
}

bool node_ediv_dx(grph_t *g, grph_size_t a) {
  ASSERT(g && a != GRPH_NO_INPUT_ID && GRPH_NODE_TYPE(g, a) == NDTYPE_EDIV);
  tnsr_t *grad_a_dep0 = GRPH_NODE_GRAD(g, GRPH_NODE_DEPS(g, a)[0]);
  tnsr_t *grad_a_dep1 = GRPH_NODE_GRAD(g, GRPH_NODE_DEPS(g, a)[1]);
  tnsr_t *data_a_dep0 = GRPH_NODE_DATA(g, GRPH_NODE_DEPS(g, a)[0]);
  tnsr_t *data_a_dep1 = GRPH_NODE_DATA(g, GRPH_NODE_DEPS(g, a)[1]);

  tnsr_t *dep0_inter = tnsr_emap(NULL, data_a_dep1, pow_neg1, NULL);
  tnsr_t *dep1_inter = tnsr_emap(NULL, data_a_dep1, pow_neg2, NULL);
  REQUIRE(dep0_inter && dep1_inter, goto error);

  REQUIRE(tnsr_emap(dep1_inter, dep1_inter, mul_neg1, NULL), goto error);
  REQUIRE(tnsr_emul(dep1_inter, dep1_inter, data_a_dep0), goto error);

  REQUIRE(tnsr_emul(dep0_inter, dep0_inter, GRPH_NODE_GRAD(g, a)), goto error);
  REQUIRE(tnsr_emul(dep1_inter, dep1_inter, GRPH_NODE_GRAD(g, a)), goto error);

  REQUIRE(tnsr_eadd(grad_a_dep0, grad_a_dep0, dep0_inter), goto error);
  REQUIRE(tnsr_eadd(grad_a_dep1, grad_a_dep1, dep1_inter), goto error);
  tnsr_destroy(&dep0_inter);
  tnsr_destroy(&dep1_inter);
  return true;
error:
  tnsr_destroy(&dep0_inter);
  tnsr_destroy(&dep1_inter);
  return false;
}

bool node_esigmoid_dx(grph_t *g, grph_size_t a) {
  ASSERT(g && a != GRPH_NO_INPUT_ID && GRPH_NODE_TYPE(g, a) == NDTYPE_ESIGMOID);
  tnsr_t *grad_a_dep0 = GRPH_NODE_GRAD(g, GRPH_NODE_DEPS(g, a)[0]);

  tnsr_t *inter = tnsr_emap(NULL, GRPH_NODE_DATA(g, a), sigmoid_odx, NULL);
  REQUIRE(inter, goto error);
  REQUIRE(tnsr_emul(inter, inter, GRPH_NODE_GRAD(g, a)), goto error);
  REQUIRE(tnsr_eadd(grad_a_dep0, grad_a_dep0, inter), goto error);
  tnsr_destroy(&inter);
  return true;
error:
  tnsr_destroy(&inter);
  return false;
}

bool node_erelu_dx(grph_t *g, grph_size_t a) {
  ASSERT(g && a != GRPH_NO_INPUT_ID && GRPH_NODE_TYPE(g, a) == NDTYPE_ERELU);
  tnsr_t *grad_a_dep0 = GRPH_NODE_GRAD(g, GRPH_NODE_DEPS(g, a)[0]);
  tnsr_t *data_a_dep0 = GRPH_NODE_DATA(g, GRPH_NODE_DEPS(g, a)[0]);

  tnsr_t *inter = tnsr_emap(NULL, data_a_dep0, relu_dx, NULL);
  REQUIRE(inter, goto error);
  REQUIRE(tnsr_emul(inter, inter, GRPH_NODE_GRAD(g, a)), goto error);
  REQUIRE(tnsr_eadd(grad_a_dep0, grad_a_dep0, inter), goto error);
  tnsr_destroy(&inter);
  return true;
error:
  tnsr_destroy(&inter);
  return false;
}

bool node_eleakyrelu_dx(grph_t *g, grph_size_t a) {
  ASSERT(g && a != GRPH_NO_INPUT_ID && GRPH_NODE_TYPE(g, a) == NDTYPE_ELEAKYRELU);
  tnsr_t *grad_a_dep0 = GRPH_NODE_GRAD(g, GRPH_NODE_DEPS(g, a)[0]);
  tnsr_t *data_a_dep0 = GRPH_NODE_DATA(g, GRPH_NODE_DEPS(g, a)[0]);

  tnsr_t *inter = tnsr_emap(NULL, data_a_dep0, leaky_relu_dx, NULL);
  REQUIRE(inter, goto error);
  REQUIRE(tnsr_emul(inter, inter, GRPH_NODE_GRAD(g, a)), goto error);
  REQUIRE(tnsr_eadd(grad_a_dep0, grad_a_dep0, inter), goto error);
  tnsr_destroy(&inter);
  return true;
error:
  tnsr_destroy(&inter);
  return false;
}

bool node_mse_dx(grph_t *g, grph_size_t a) {
  ASSERT(g && a != GRPH_NO_INPUT_ID && GRPH_NODE_TYPE(g, a) == NDTYPE_MSE);
  tnsr_t *grad_a_dep0 = GRPH_NODE_GRAD(g, GRPH_NODE_DEPS(g, a)[0]);
  tnsr_t *grad_a_dep1 = GRPH_NODE_GRAD(g, GRPH_NODE_DEPS(g, a)[1]);
  tnsr_t *data_a_dep0 = GRPH_NODE_DATA(g, GRPH_NODE_DEPS(g, a)[0]);
  tnsr_t *data_a_dep1 = GRPH_NODE_DATA(g, GRPH_NODE_DEPS(g, a)[1]);
  tnsr_set(GRPH_NODE_GRAD(g, a), 1);

  tnsr_t *nc = TNSR_SCALAR();
  REQUIRE(nc, goto error);
  tnsr_t *diff = tnsr_esub(NULL, data_a_dep0, data_a_dep1);
  REQUIRE(diff, goto error);

  tnsr_set(nc, 2.0f / TNSR_SHPE(grad_a_dep0, 1));
  REQUIRE(tnsr_emul(diff, diff, nc), goto error);
  REQUIRE(tnsr_eadd(grad_a_dep0, grad_a_dep0, diff), goto error);
  tnsr_set(nc, -1);
  REQUIRE(tnsr_emul(diff, diff, nc), goto error);
  REQUIRE(tnsr_eadd(grad_a_dep1, grad_a_dep1, diff), goto error);

  tnsr_destroy(&diff);
  tnsr_destroy(&nc);
  return true;
error:
  tnsr_destroy(&diff);
  tnsr_destroy(&nc);
  return false;
}

bool node_cross_entropy_loss_dx(grph_t *g, grph_size_t a) {
  ASSERT(g && a != GRPH_NO_INPUT_ID && GRPH_NODE_TYPE(g, a) == NDTYPE_CROSS_ENTROPY_LOSS);
  tnsr_t *grad_a_dep0 = GRPH_NODE_GRAD(g, GRPH_NODE_DEPS(g, a)[0]);
  tnsr_t *grad_a_dep1 = GRPH_NODE_GRAD(g, GRPH_NODE_DEPS(g, a)[1]);
  tnsr_t *data_a_dep0 = GRPH_NODE_DATA(g, GRPH_NODE_DEPS(g, a)[0]);
  tnsr_t *data_a_dep1 = GRPH_NODE_DATA(g, GRPH_NODE_DEPS(g, a)[1]);

  tnsr_set(GRPH_NODE_GRAD(g, a), 1);
  tnsr_t *inter = tnsr_create(TNSR_SHPE(data_a_dep1, 0), TNSR_SHPE(data_a_dep1, 1));
  REQUIRE(inter, goto error);
  tnsr_set(inter, -1.0f / TNSR_SHPE(data_a_dep0, 0));
  REQUIRE(tnsr_emul(inter, inter, data_a_dep1), goto error);
  REQUIRE(tnsr_ediv(inter, inter, data_a_dep0), goto error);

  REQUIRE(tnsr_eadd(grad_a_dep0, grad_a_dep0, inter), goto error);
  REQUIRE(tnsr_emap(inter, data_a_dep0, ln, NULL), goto error);
  REQUIRE(tnsr_emap(inter, inter, mul_neg1, NULL), goto error);
  REQUIRE(tnsr_eadd(grad_a_dep1, grad_a_dep1, inter), goto error);

  tnsr_destroy(&inter);
  return true;

error:
  tnsr_destroy(&inter);
  return false;
}

bool node_softmax_dx(grph_t *g, grph_size_t a) {
  ASSERT(g && a != GRPH_NO_INPUT_ID && GRPH_NODE_TYPE(g, a) == NDTYPE_SOFTMAX);
  tnsr_t *grad_a_dep0 = GRPH_NODE_GRAD(g, GRPH_NODE_DEPS(g, a)[0]);

  tnsr_t *sub = NULL;
  tnsr_t *dot = NULL;
  tnsr_t *inter = tnsr_emul(NULL, GRPH_NODE_GRAD(g, a), GRPH_NODE_DATA(g, a));
  REQUIRE(inter, goto error);
  dot = tnsr_sum_over_axis(NULL, inter, 1);
  REQUIRE(dot, goto error);
  sub = tnsr_esub(NULL, GRPH_NODE_GRAD(g, a), dot);
  
  REQUIRE(sub, goto error);
  REQUIRE(tnsr_emul(sub, sub, GRPH_NODE_DATA(g, a)), goto error);
  REQUIRE(tnsr_eadd(grad_a_dep0, grad_a_dep0, sub), goto error);
  
  tnsr_destroy(&inter);
  tnsr_destroy(&dot);
  tnsr_destroy(&sub);
  return true;
error:
  tnsr_destroy(&inter);
  tnsr_destroy(&dot);
  tnsr_destroy(&sub);
  return false;
}

