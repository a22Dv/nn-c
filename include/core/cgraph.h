/**
 * cgraph.h
 *
 * Declaration for the cnode_t and cgraph_t struct.
 */

#pragma once

#define CN_MAX_INDEGREE 2
#define CN_MAX_OUTDEGREE 2

#include "core/tensor.h"

typedef enum {
  CN_DATA,        // Generic data.
  CN_SIGMOID,     // Sigmoid operation. Get prev[0] for operand.
  CN_RELU,        // ReLU operation. Get prev[0] for operand.
  CN_LEAKY_RELU,  // Leaky ReLU operation. Get prev[0] for operand.
  CN_MUL,         // Multiplication operation. Get prev[0], prev[1] for operands.
  CN_SUB,         // Subtraction operation. Get prev[0], prev[1] for operands.
  CN_ADD,         // Addition operation. Get prev[0], prev[1] for operands.
  CN_DIV,         // Division operation. Get prev[0], prev[1] for operands.
  CN_MSE,         // Mean-squared-error operation. Get prev[0], prev[1] for operands.
  CN_CONTRACT,    // Tensor contraction operation. Get prev[0], prev[1] for operands.
} cnode_type_t;

typedef struct cnode {
  tensor_size_t children;
  tensor_size_t parents;
  cnode_type_t type;
  tensor_t *data;      // Result of operation, or as-is for CN_DATA.
  tensor_t *gradient;  // Gradient attributed to the operation. 
  struct cnode *next[CN_MAX_OUTDEGREE];
  struct cnode *prev[CN_MAX_INDEGREE];
} cnode_t;

typedef struct {
  cnode_t *head;
} cgraph_t;

// Does not perform the operation, defines and links nodes together.
void cnode_attach(cnode_t *operands[CN_MAX_INDEGREE], cnode_type_t type);

// Performs the operations required by the graph.
void cnode_traverse_and_perform(cnode_t *head);

// Calculates the gradients of the graph.
void cnode_traverse_gradient(cnode_t *tail);

// Resets the gradients on the given flags.
void cnode_reset_gradients(cnode_t *head, int flags);

// Resets the data on nodes with the given flags.
void cnode_reset_data(cnode_t *head, int flags);

// Eliminates all pointers to it. Including immediate parents.
void cnode_destroy(cnode_t **node);

// Creates a node connected to the given previous and next nodes.
void cnode_create(
    tensor_size_t parents,
    tensor_size_t children,
    cnode_t *prev[CN_MAX_INDEGREE],
    cnode_t *next[CN_MAX_OUTDEGREE],
    tensor_t *data
);
