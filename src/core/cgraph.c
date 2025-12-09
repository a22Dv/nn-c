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
  switch (head->type) {
    case CN_ADD:
      REQUIRE(head->prev[0] && head->prev[1], return false);
      REQUIRE(tensor_eadd(head->data, head->prev[0]->data, head->prev[1]->data), return false);
      break;
    case CN_SUB:
      REQUIRE(head->prev[0] && head->prev[1], return false);
      REQUIRE(tensor_esub(head->data, head->prev[0]->data, head->prev[1]->data), return false);
      break;
    case CN_MUL:
      REQUIRE(head->prev[0] && head->prev[1], return false);
      REQUIRE(tensor_emul(head->data, head->prev[0]->data, head->prev[1]->data), return false);
      break;
    case CN_DIV:
      REQUIRE(head->prev[0] && head->prev[1], return false);
      REQUIRE(tensor_ediv(head->data, head->prev[0]->data, head->prev[1]->data), return false);
      break;
    case CN_CONTRACT:
      REQUIRE(head->prev[0] && head->prev[1], return false);
      REQUIRE(tensor_contract(head->data, head->prev[0]->data, head->prev[1]->data), return false);
      break;
    case CN_MSE:
      REQUIRE(head->prev[0] && head->prev[1], return false);
      REQUIRE(tensor_mse(head->data, head->prev[0]->data, head->prev[1]->data), return false);
      break;
    case CN_LEAKY_RELU:
      REQUIRE(head->prev[0], return false);
      REQUIRE(tensor_emap(head->data, head->prev[0]->data, leaky_relu), return false);
      break;
    case CN_RELU:
      REQUIRE(head->prev[0], return false);
      REQUIRE(tensor_emap(head->data, head->prev[0]->data, relu), return false);
      break;
    case CN_SIGMOID:
      REQUIRE(head->prev[0], return false);
      REQUIRE(tensor_emap(head->data, head->prev[0]->data, sigmoid), return false);
      break;
    case CN_TRANSPOSE:
      REQUIRE(head->prev[0], return false);
      REQUIRE(tensor_transpose(head->data, head->prev[0]->data), return false);
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


/// TODO: Implementation.
/// NOTE: Equation comments do not use standard notation. Just for reference.
bool cnode_traverse_gradient(cnode_t *tail) {
  if (tail == NULL) {
    return true;
  }
  switch (tail->type) {
    case CN_ADD: // 𝛛L/𝛛a = 𝛁, 𝛛L/𝛛b = 𝛁
      break;
    case CN_SUB: // 𝛛L/𝛛a = 𝛁, 𝛛L/𝛛b = -𝛁
      break;
    case CN_MUL: // 𝛛L/𝛛a = 𝛁b, 𝛛L/𝛛b = 𝛁a
      break;
    case CN_DIV: // 𝛛L/𝛛a = 𝛁(1/b), 𝛛L/𝛛b = 𝛁(-a/b^2)
      break;
    case CN_CONTRACT: // 𝛛L/𝛛a = 𝛁(b^T),  𝛛L/𝛛b = (a^T)𝛁
      break;
    case CN_MSE: // 𝛛L/𝛛a = 2(y1-y2)/n, 𝛛L/𝛛L = 1
      break;
    case CN_LEAKY_RELU: // 𝛛L/𝛛a = 𝛁 > 0 -> 1, 𝛁 <= 0 -> 0.01
      break;
    case CN_RELU: // 𝛛L/𝛛a = 𝛁 > 0 -> 1, 𝛁 <= 0 -> 0
      break;
    case CN_SIGMOID: // 𝛛L/𝛛a = (e^(-𝛁))(1+e^(-𝛁))^2
      break;
    case CN_TRANSPOSE: // 𝛛L/𝛛a = 𝛁^T
      break;
    case CN_DATA: // No-op.
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
