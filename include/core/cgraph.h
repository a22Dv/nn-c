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
  CN_TRANSPOSE,   // Tensor transposition operation. Get prev[0] for operands.
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

// Creates a node connected to the given previous and next nodes.
// Data or gradient cannot be NULL. The node will reuse the same memory
// block for the computations, so it must be sized in advance.
cnode_t *cnode_create(
    tensor_size_t parents,
    tensor_size_t children,
    cnode_t *prev[CN_MAX_INDEGREE],
    cnode_t *next[CN_MAX_OUTDEGREE],
    tensor_t *data,
    tensor_t *gradient,
    cnode_type_t type
);

// NULLs all pointers to it in immediate parents and children.
void cnode_destroy(cnode_t **node);

// Does not perform the operation, defines and links nodes together.
// Tensors for data and gradient are reused and must be sized in advance.
// returns a pointer to the attached node if successful.
cnode_t *cnode_attach(
    cnode_t *operands[CN_MAX_INDEGREE], 
    cnode_type_t type, 
    tensor_t *data, 
    tensor_t *gradient
);

// Performs the operations required by the graph.
// Goes downstream starting at the head.
bool cnode_traverse_and_perform(cnode_t *head);

// Calculates the gradients of the graph.
// Goes upstream and calculates the derivatives to place in the
// gradient field of the nodes. Any previous gradient is
// overwritten.
bool cnode_traverse_gradient(cnode_t *tail);

// Resets the gradients on the given flags.
// This function won't climb "back up" to other parents or
// spread to the entire graph. Necessary behavior to avoid
// resetting the weights themselves.
void cnode_reset_gradients(cnode_t *head);

// Resets the data on nodes. Traverses downstream.
// This function won't climb "back up" to other parents or
// spread to the entire graph. Necessary behavior to avoid
// resetting the weights themselves.
void cnode_reset_data(cnode_t *head);
