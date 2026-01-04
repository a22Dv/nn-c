/**
 * graph.c
 *
 * BRIEF:
 * Implementation for graph.h
 */

#include <string.h>

#include "core/graph.h"
#include "core/node.h"
#include "utils/utils.h"


static grph_size_t input_req[] = {_GRPH_INPUT_TBLE};

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
    [NDTYPE_ETANH] = node_etanh,
    [NDTYPE_MSE] = node_mse,
    [NDTYPE_CATEGORICAL_CROSS_ENTROPY_LOSS] = node_categorical_cross_entropy_loss,
    [NDTYPE_BINARY_CROSS_ENTROPY_LOSS] = node_binary_cross_entropy_loss,
    [NDTYPE_SOFTMAX] = node_softmax,
};

static bool (*node_functions_dx[])(grph_t *, grph_size_t) = {
    [NDTYPE_TRANSPOSE] = node_transpose_dx,
    [NDTYPE_CONTRACT] = node_contract_dx,
    [NDTYPE_EADD] = node_eadd_dx,
    [NDTYPE_ESUB] = node_esub_dx,
    [NDTYPE_EMUL] = node_emul_dx,
    [NDTYPE_EDIV] = node_ediv_dx,
    [NDTYPE_ESIGMOID] = node_esigmoid_dx,
    [NDTYPE_ERELU] = node_erelu_dx,
    [NDTYPE_ELEAKYRELU] = node_eleakyrelu_dx,
    [NDTYPE_ETANH] = node_etanh_dx,
    [NDTYPE_MSE] = node_mse_dx,
    [NDTYPE_CATEGORICAL_CROSS_ENTROPY_LOSS] = node_categorical_cross_entropy_loss_dx,
    [NDTYPE_BINARY_CROSS_ENTROPY_LOSS] = node_binary_cross_entropy_loss_dx,
    [NDTYPE_SOFTMAX] = node_softmax_dx,
}; 

typedef enum {
  ND_NOT_VISITED,
  ND_VISITING,
  ND_VISITED,
} node_visited_t;

static bool topological_sort(
    grph_t *g, grph_size_t n, grph_size_t *topological, node_visited_t *visited, grph_size_t *count
) {
  ASSERT(g && topological && visited && count);
  visited[n] = ND_VISITING;
  for (grph_size_t i = 0; i < GRPH_NODE_NDEP(g, n); ++i) {
    const grph_size_t node_id = GRPH_NODE_DEPS(g, n)[i];
    ASSERT(visited[node_id] != ND_VISITING);
    if (visited[node_id] == ND_VISITED) {
      continue;
    }
    REQUIRE(topological_sort(g, node_id, topological, visited, count), goto error);
  }
  visited[n] = ND_VISITED;
  topological[*count] = n;
  ++(*count);
  return true;
error:
  return false;
}

static grph_size_t grph_tail(grph_t *g) {
  ASSERT(g);
  grph_size_t *outdegs = calloc(1, sizeof(grph_size_t[GRPH_NODES(g)]));
  REQUIRE(outdegs, goto error);

  for (grph_size_t i = 0; i < GRPH_NODES(g); ++i) {
    grph_size_t ndep = GRPH_NODE_NDEP(g, i);
    for (grph_size_t j = 0; j < ndep; ++j) {
      ++outdegs[GRPH_NODE_DEPS(g, i)[j]];
    }
  }
  grph_size_t tail = 0;
  for (grph_size_t i = 0; i < GRPH_NODES(g); ++i) {
    if (!outdegs[i]) {
      REQUIRE(tail == 0, goto error);
      tail = i;
    }
  }
  free(outdegs);
  return tail;

error:
  free(outdegs);
  return GRPH_ERR_ID;
}

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
  if (!g || !*g) {
    return;
  }
  grph_t *graph = *g;
  for (grph_size_t i = 0; i < GRPH_NODES(graph); ++i) {
    if (GRPH_NODE_TRANSIENT(graph, i)) {
      node_destroy(&GRPH_NODE(graph, i));
    } else {
      tnsr_destroy(&GRPH_NODE(graph, i)->grad);
      free(GRPH_NODE(graph, i));
      GRPH_NODE(graph, i) = NULL;
    }
  }
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
  (void)input_req;
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
  grph_size_t tail = grph_tail(g);
  grph_size_t *topological = NULL;
  node_visited_t *visited = NULL;
  REQUIRE(tail != GRPH_ERR_ID, goto error);

  topological = malloc(sizeof(grph_size_t[GRPH_NODES(g)]));
  visited = calloc(1, sizeof(node_visited_t[GRPH_NODES(g)]));

  REQUIRE(topological && visited, goto error);

  grph_size_t found = 0;
  REQUIRE(topological_sort(g, tail, topological, visited, &found), goto error);

  for (int i = GRPH_NODES(g); i-- > 0;) {
    const grph_size_t node_id = topological[i];
    const node_type_t ntype = GRPH_NODE_TYPE(g, node_id);
    if (ntype == NDTYPE_DATA) {
      continue;
    }
    REQUIRE(node_functions_dx[ntype](g, node_id), goto error);
  }

  free(topological);
  free(visited);
  return true;

error:
  free(topological);
  free(visited);
  return false;
}