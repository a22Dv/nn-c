/**
 * graph.h
 *
 * BRIEF:
 * Declaration for grph_t and node_t structs,
 * as well as related functions.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "core/tensor.h"

typedef uint16_t grph_size_t;

#define GRPH_INITCPCTY 64
#define GRPH_MAX_SIZE UINT16_MAX
#define GRPH_ERR_ID UINT16_MAX
#define GRPH_NO_INPUT_ID UINT16_MAX
#define NODE_INIT_DEP_CPCTY 2

/* -------------------------------- Accessors ------------------------------- */

#define GRPH_NODE(g, i) ((g)->adj_list[i])
#define GRPH_NODE_TRANSIENT(g, i) ((g)->adj_list[i]->transient)
#define GRPH_NODE_DATA(g, i) ((g)->adj_list[i]->data)
#define GRPH_NODE_GRAD(g, i) ((g)->adj_list[i]->grad)
#define GRPH_NODE_TYPE(g, i) ((g)->adj_list[i]->type)
#define GRPH_NODE_NDEP(g, i) ((g)->adj_list[i]->ndependencies)
#define GRPH_NODE_DEPS(g, i) ((g)->adj_list[i]->dependencies)

#define GRPH_CPCTY(g) ((g)->capacity)
#define GRPH_NODES(g) ((g)->nodes)
#define GRPH_LIST(g) ((g)->adj_list)

/* ----------------------------------- API ---------------------------------- */


typedef enum {
  NDTYPE_DATA,
  NDTYPE_TRANSPOSE,
  NDTYPE_CONTRACT,
  NDTYPE_EADD,
  NDTYPE_ESUB,
  NDTYPE_EMUL,
  NDTYPE_EDIV,
  NDTYPE_ESIGMOID,
  NDTYPE_ERELU,
  NDTYPE_ELEAKYRELU,
  NDTYPE_MSE,
  NDTYPE_CROSS_ENTROPY_LOSS,
  NDTYPE_SOFTMAX,
} node_type_t;

#define _GRPH_INPUT_TBLE                                                                          \
  [NDTYPE_DATA] = 0, [NDTYPE_TRANSPOSE] = 1, [NDTYPE_CONTRACT] = 2, [NDTYPE_EADD] = 2,            \
  [NDTYPE_ESUB] = 2, [NDTYPE_EMUL] = 2, [NDTYPE_EDIV] = 2, [NDTYPE_ESIGMOID] = 1,                 \
  [NDTYPE_ERELU] = 1, [NDTYPE_ELEAKYRELU] = 1, [NDTYPE_MSE] = 2, [NDTYPE_CROSS_ENTROPY_LOSS] = 2, \
  [NDTYPE_SOFTMAX] = 1,


typedef enum {
  OUTSIZE_DEP_ON_A0 = 1,
  OUTSIZE_DEP_ON_B0 = (1 << 1),
  OUTSIZE_DEP_ON_A1 = (1 << 2),
  OUTSIZE_DEP_ON_B1 = (1 << 3),
  OUTSIZE_DEP_SAMEAS = (1 << 4),
  OUTSIZE_TRANSPOSED = (1 << 5),
  OUTSIZE_SCALAR = (1 << 6),
  OUTSIZE_INDEPENDENT = (1 << 7),
} grph_outsize_t;

typedef struct {
  bool transient;
  tnsr_t *data;
  tnsr_t *grad;
  node_type_t type;

  grph_size_t n_dependencies;
  grph_size_t n_deps_capacity;
  grph_size_t dependencies[];
} node_t;

typedef struct {
  grph_size_t nodes;
  grph_size_t capacity;
  node_t *adj_list[];
} grph_t;

// Initializes a graph with a set capacity.
// Passing 0 defaults to GRPH_INITCPCTY.
grph_t *grph_create(grph_size_t cpcty);

// Deallocates the graph and sets its pointer to NULL.
// Passing NULL is a no-op.
void grph_destroy(grph_t **g);

// Appends a data node to the graph,
// and returns its index in the node list.
// This operation INVALIDATES existing pointers. As the graph
// can reallocate should it exceed capacity.
grph_size_t grph_append_data(grph_t **g, tnsr_t *node);

// Eagerly executes the operation specified by ntype on A and B. Unary operations must
// set A and set B to GRPH_NO_INPUT. The operation is then appended to the graph afterwards,
// and returns its index in the adjacency list. This operation INVALIDATES existing pointers.
// As the graph can reallocate should it exceed its current capacity.
grph_size_t grph_execute(grph_t **g, grph_size_t a, grph_size_t b, node_type_t ntype);

// Traces the graph backwards and fills in the gradient fields for each node.
// Does a Topological sort of the graph and returns the order of execution.
bool grph_trace(grph_t *g);
