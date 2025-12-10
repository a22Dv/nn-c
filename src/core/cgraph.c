/**
 * cgraph.c
 *
 * Implementation for cgraph.h
 */

#include <stdbool.h>
#include <stdlib.h>

#include "core/cgraph.h"
#include "core/tensor.h"
#include "utils/utils.h"

cnode_t *cnode_create(
    tensor_size_t parents,
    tensor_size_t children,
    cnode_t *prev[CN_MAX_INDEGREE],
    cnode_t *next[CN_MAX_OUTDEGREE],
    tensor_t *data,
    tensor_t *gradient,
    cnode_type_t type
) {
  REQUIRE(data && gradient, return NULL);
  REQUIRE(parents <= CN_MAX_INDEGREE && children <= CN_MAX_OUTDEGREE, return NULL);
  cnode_t *node = malloc(sizeof(cnode_t));
  REQUIRE(node, return NULL);

  node->parents = parents;
  node->children = children;
  node->data = data;
  node->gradient = gradient;
  node->type = type;

  for (tensor_size_t i = 0; i < CN_MAX_INDEGREE; ++i) {
    node->prev[i] = i < parents ? prev[i] : NULL;
    if (!node->prev[i]) {
      continue;
    }
    bool set = false;
    for (tensor_size_t j = 0; j < CN_MAX_OUTDEGREE; ++j) {
      if (node->prev[i]->next[j]) {
        continue;
      }
      node->prev[i]->next[j] = node;
      set = true;
      break;
    }
    REQUIRE(set, goto failure);
  }
  for (tensor_size_t i = 0; i < CN_MAX_OUTDEGREE; ++i) {
    node->next[i] = i < children ? next[i] : NULL;
    if (!node->next[i]) {
      continue;
    }
    bool set = false;
    for (tensor_size_t j = 0; j < CN_MAX_INDEGREE; ++j) {
      if (node->next[i]->prev[j]) {
        continue;
      }
      node->next[i]->prev[j] = node;
      set = true;
      break;
    }
    REQUIRE(set, goto failure);
  }
  return node;

failure:
  cnode_destroy(&node);
  return NULL;
}

void cnode_destroy(cnode_t **node) {
  REQUIRE(node && *node, return);
  cnode_t *rnode = *node;
  for (tensor_size_t i = 0; i < rnode->children; ++i) {
    for (tensor_size_t j = 0; j < CN_MAX_INDEGREE; ++j) {
      if (rnode->next[i] && rnode->next[i]->prev[j] == rnode) {
        rnode->next[i]->prev[j] = NULL;
        break;
      }
    }
  }
  for (tensor_size_t i = 0; i < rnode->parents; ++i) {
    for (tensor_size_t j = 0; j < CN_MAX_OUTDEGREE; ++j) {
      if (rnode->prev[i] && rnode->prev[i]->next[j] == rnode) {
        rnode->prev[i]->next[j] = NULL;
        break;
      }
    }
  }
  free(rnode);
  *node = NULL;
}

cnode_t *cnode_attach(
    cnode_t *operands[CN_MAX_INDEGREE], cnode_type_t type, tensor_t *data, tensor_t *gradient
) {
  tensor_size_t opc = 0;
  switch (type) {
    case CN_ADD:
    case CN_SUB:
    case CN_MUL:
    case CN_DIV:
    case CN_MSE:
    case CN_CONTRACT:
      opc = 2;
      break;
    case CN_LEAKY_RELU:
    case CN_RELU:
    case CN_SIGMOID:
    case CN_TRANSPOSE:
      opc = 1;
      break;
    case CN_DATA:
      opc = 0;
      break;
  }
  tensor_size_t fopc = 0;
  for (tensor_size_t i = 0; i < CN_MAX_INDEGREE; ++i) {
    fopc += !!operands[i];
  }
  REQUIRE(fopc == opc, return NULL);
  cnode_t *node = cnode_create(opc, 0, operands, NULL, data, gradient, type);
  return node;
}

bool cnode_traverse_and_perform(cnode_t *head) {
  if (head == NULL) {
    return true;
  }
  cnode_t *p1 = head->prev[0];
  cnode_t *p2 = head->prev[1];
  tensor_t *y1 = p1 ? p1->data : NULL;
  tensor_t *y2 = p2 ? p2->data : NULL;
  tensor_t *d = head->data;

  switch (head->type) {
    case CN_ADD:
      REQUIRE(tensor_eadd(d, y1, y2), return false);
      break;
    case CN_SUB:
      REQUIRE(tensor_esub(d, y1, y2), return false);
      break;
    case CN_MUL:
      REQUIRE(tensor_emul(d, y1, y2), return false);
      break;
    case CN_DIV:
      REQUIRE(tensor_ediv(d, y1, y2), return false);
      break;
    case CN_CONTRACT:
      REQUIRE(tensor_contract(d, y1, y2), return false);
      break;
    case CN_MSE:
      REQUIRE(tensor_mse(d, y1, y2), return false);
      break;
    case CN_LEAKY_RELU:
      REQUIRE(tensor_emap(d, y1, leaky_relu), return false);
      break;
    case CN_RELU:
      REQUIRE(tensor_emap(d, y1, relu), return false);
      break;
    case CN_SIGMOID:
      REQUIRE(tensor_emap(d, y1, sigmoid), return false);
      break;
    case CN_TRANSPOSE:
      REQUIRE(tensor_transpose(d, y1), return false);
      break;
    case CN_DATA:  // No-op.
      break;
  }
  bool ops = true;
  for (tensor_size_t i = 0; i < CN_MAX_OUTDEGREE; ++i) {
    if (!head->next[i]) {
      continue;
    }
    if (!(ops = cnode_traverse_and_perform(head->next[i]))) {
      break;
    }
  }
  return ops;
}

bool cnode_traverse_gradient(cnode_t *tail) {
  if (tail == NULL) {
    return true;
  }

  tensor_t *y1g = tail->prev[0]->gradient;
  tensor_t *y2g = tail->prev[1]->gradient;
  tensor_t *y1 = tail->prev[0]->data;
  tensor_t *y2 = tail->prev[1]->data;
  tensor_t *g = tail->gradient;
  tensor_t *d = tail->data;

  switch (tail->type) {
    case CN_ADD:  // 𝛛L/𝛛a = 𝛁, 𝛛L/𝛛b = 𝛁
      REQUIRE(tensor_eadd(y1g, y1g, g), return false);
      REQUIRE(tensor_eadd(y2g, y2g, g), return false);
      break;
    case CN_SUB:  // 𝛛L/𝛛a = 𝛁, 𝛛L/𝛛b = -𝛁
      REQUIRE(tensor_eadd(y1g, y1g, g), return false);
      REQUIRE(tensor_esub(y2g, y2g, g), return false);
      break;
    case CN_MUL:  // 𝛛L/𝛛a = 𝛁b, 𝛛L/𝛛b = 𝛁a
      REQUIRE(tensor_eadd(y1g, y1g, g), return false);
      REQUIRE(tensor_eadd(y2g, y2g, g), return false);
      REQUIRE(tensor_emul(y1g, y1g, y2), return false);
      REQUIRE(tensor_emul(y2g, y2g, y1), return false);
      break;
    case CN_DIV:  // 𝛛L/𝛛a = 𝛁(1/b), 𝛛L/𝛛b = 𝛁(-a/b^2)
      tensor_t *one = tensor_create(0, TENSOR_DECLARE_SHAPE(1, 1));
      REQUIRE(one, return false);
      one->data[0] = 1;

      REQUIRE(tensor_eadd(y1g, y1g, one), return false);
      REQUIRE(tensor_ediv(y1g, y1g, y2), return false);
      REQUIRE(tensor_emul(y1g, y1g, g), return false);

      REQUIRE(tensor_esub(y2g, y2g, y1), return false);
      REQUIRE(tensor_ediv(y2g, y2g, y2), return false);
      REQUIRE(tensor_ediv(y2g, y2g, y2), return false);
      REQUIRE(tensor_emul(y2g, y2g, g), return false);
      tensor_destroy(&one);
      break;
    case CN_CONTRACT:  // 𝛛L/𝛛a = 𝛁(b^T),  𝛛L/𝛛b = (a^T)𝛁
      tensor_t* y1t = tensor_transpose(NULL, y1);
      tensor_t* y2t = tensor_transpose(NULL, y2);
      REQUIRE(y1t, return false);
      REQUIRE(y2t, return false);
      REQUIRE(tensor_contract(y1g, g, y2t), return false);
      REQUIRE(tensor_contract(y2g, y1t, g), return false);
      tensor_destroy(&y1t);
      tensor_destroy(&y2t);
      break;
    case CN_MSE:       // 𝛛L/𝛛a = 2/n * (y1-y2), 𝛛L/𝛛L = 1
      tensor_t *scalar = tensor_create(0, TENSOR_DECLARE_SHAPE(1, 1));
      REQUIRE(scalar, return false);
      scalar->data[0] = 2.0f / TENSOR_SHAPE(y1, 1);

      REQUIRE(tensor_esub(y1g, y1, y2), return false);
      REQUIRE(tensor_emul(y1g, y1g, scalar), return false);
      REQUIRE(tensor_emul(y1g, y1g, g), return false);  // Implicit 1.
      tensor_destroy(&scalar);
      break;
    case CN_LEAKY_RELU:  // 𝛛L/𝛛a = 𝛁 > 0 -> 1, 𝛁 <= 0 -> 0.01
      REQUIRE(tensor_emap(y1g, d, leaky_relu_dx), return false);
      REQUIRE(tensor_emul(y1g, y1g, g), return false);
      break;
    case CN_RELU:  // 𝛛L/𝛛a = 𝛁 > 0 -> 1, 𝛁 <= 0 -> 0
      REQUIRE(tensor_emap(y1g, d, relu_dx), return false);
      REQUIRE(tensor_emul(y1g, y1g, g), return false);
      break;
    case CN_SIGMOID:  // 𝛛L/𝛛a = (e^(-𝛁))(1+e^(-𝛁))^2
      REQUIRE(tensor_emap(y1g, d, sigmoid_dx), return false);
      REQUIRE(tensor_emul(y1g, y1g, g), return false);
      break;
    case CN_TRANSPOSE:  // 𝛛L/𝛛a = 𝛁^T
      REQUIRE(tensor_transpose(y1g, g), return false);
      break;
    case CN_DATA:  // No-op.
      break;
  }
  bool ops = true;
  for (tensor_size_t i = 0; i < CN_MAX_INDEGREE; ++i) {
    if (!tail->prev[i]) {
      continue;
    }
    if (!(ops = cnode_traverse_gradient(tail->prev[i]))) {
      break;
    }
  }
  return ops;
}

void cnode_reset_gradient(cnode_t *head) {
  if (head == NULL) {
    return;
  }
  tensor_emap(head->gradient, head->gradient, zeroes);
  for (tensor_size_t i = 0; i < CN_MAX_OUTDEGREE; ++i) {
    cnode_reset_data(head->next[i]);
  }
}

void cnode_reset_data(cnode_t *head) {
  if (head == NULL) {
    return;
  }
  tensor_emap(head->data, head->data, zeroes);
  for (tensor_size_t i = 0; i < CN_MAX_OUTDEGREE; ++i) {
    cnode_reset_data(head->next[i]);
  }
}
