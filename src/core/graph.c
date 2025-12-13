/**
 * graph.c
 *
 * BRIEF:
 * Implementation for graph.h
 */

#include <string.h>

#include "core/graph.h"
#include "core/graph_functions.h"
#include "utils/utils.h"

static grph_size_t input_req[] = {
    _GRPH_INPUT_TBLE
};

static node_t *(*node_functions[])(grph_t *, grph_size_t, grph_size_t) = {
    [NDTYPE_TRANSPOSE] = node_transpose,
    [NDTYPE_CONTRACT] = node_contract,
    [NDTYPE_EADD] = node_eadd,
    [NDTYPE_ESUB] = node_esub,
    [NDTYPE_EMUL] = node_emul,
    [NDTYPE_EDIV] = node_ediv,
    [NDTYPE_ESIGMOID] = node_esigmoid,
    [NDTYPE_ERELU] = node_erelu,
    [NDTYPE_ELEAKYRELU] = node_eleakyrelu,
    [NDTYPE_MSE] = node_mse,
    [NDTYPE_CROSS_ENTROPY_LOSS] = node_cross_entropy_loss,
    [NDTYPE_SOFTMAX] = node_softmax,
};

static grph_t *grph_resize(grph_t *g) {
  ASSERT(g);

  grph_t *rg = NULL;
  REQUIRE(GRPH_CPCTY(g) < GRPH_MAX_SIZE / 2, goto error);
  rg = realloc(g, sizeof(grph_t) + sizeof(node_t * [2 * GRPH_CPCTY(g)]));
  REQUIRE(rg, goto error);
  memset(&rg->adj_list[GRPH_CPCTY(rg)], 0, sizeof(node_t * [GRPH_CPCTY(rg)]));
  rg->capacity = 2 * GRPH_CPCTY(rg);

  return rg;
error:
  return NULL;
}

grph_t *grph_create(grph_size_t cpcty) {
  ASSERT(cpcty != GRPH_MAX_SIZE);
  if (!cpcty) {
    cpcty = GRPH_INITCPCTY;
  }
  grph_t *graph = calloc(1, sizeof(grph_t) + sizeof(node_t *[cpcty]));
  REQUIRE(graph, goto error);
  graph->capacity = cpcty;

  return graph;
error:
  return NULL;
}

void grph_destroy(grph_t **g) {
  REQUIRE(g && *g, return);
  free(*g);
  *g = NULL;
}

grph_size_t grph_append_data(grph_t **g, tnsr_t *data) {
  ASSERT(g && *g && data);

  if (GRPH_NODES(*g) == GRPH_CPCTY(*g)) {
    grph_t *rg = grph_resize(*g);
    REQUIRE(rg, goto error);
    *g = rg;
  }
  grph_t *graph = *g;
  node_t *node = node_create(graph, data, GRPH_NO_INPUT_ID, GRPH_NO_INPUT_ID, NDTYPE_DATA);
  REQUIRE(node, goto error);

  GRPH_LIST(graph)[GRPH_NODES(graph)] = node;
  ++GRPH_NODES(graph);
  return GRPH_NODES(graph) - 1;
error:
  return GRPH_ERR_ID;
}

grph_size_t grph_execute(grph_t **g, grph_size_t a, grph_size_t b, node_type_t ntype) {
  ASSERT(g && *g && ntype != NDTYPE_DATA);
  ASSERT((a != GRPH_NO_INPUT_ID) + (b != GRPH_NO_INPUT_ID) == input_req[ntype]);
  ASSERT((input_req[ntype] == 1 ? a != GRPH_NO_INPUT_ID : true));

  if (GRPH_NODES(*g) == GRPH_CPCTY(*g)) {
    grph_t *rg = grph_resize(*g);
    REQUIRE(rg, goto error);
    *g = rg;
  }
  grph_t *graph = *g;
  node_t *node = node_functions[ntype](graph, a, b);
  REQUIRE(node, goto error);

  GRPH_LIST(graph)[GRPH_NODES(graph)] = node;
  ++GRPH_NODES(graph);
  return GRPH_NODES(graph) - 1;

error:
  return GRPH_ERR_ID;
}

bool grph_trace(grph_t *g) {
  ASSERT(g);

  return false;
}