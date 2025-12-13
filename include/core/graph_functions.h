/**
 * graph_functions.h
 *
 * BRIEF:
 * Declarations for graph-related functions
 * that operate on given nodes.
 */

#pragma once

#include "core/graph.h"

// Creates the appropriate node based on the node type.
node_t *node_create(grph_t *g, tnsr_t *data, grph_size_t a, grph_size_t b, node_type_t type);

// Deallocates a given node. Passing NULL is a no-op.
void node_destroy(node_t **n);

// Applies a transposition on A and returns the result
// in a new node. B must be set to GRPH_NO_INPUT_ID.
node_t *node_transpose(grph_t *g, grph_size_t a, grph_size_t b);

// Applies a tensor contraction on A and B and returns the result in a new node.
node_t *node_contract(grph_t *g, grph_size_t a, grph_size_t b);

// Applies element-wise addition on A and B and returns the result in a new node.
node_t *node_eadd(grph_t *g, grph_size_t a, grph_size_t b);

// Applies element-wise subtraction on A and B and returns the result in a new node.
node_t *node_esub(grph_t *g, grph_size_t a, grph_size_t b);

// Applies element-wise multiplication on A and B and returns the result in a new node.
node_t *node_emul(grph_t *g, grph_size_t a, grph_size_t b);

// Applies element-wise division on A and B and returns the result in a new node.
node_t *node_ediv(grph_t *g, grph_size_t a, grph_size_t b);

// Applies the sigmoid function on A and returns the result
// in a new node. B must be set to GRPH_NO_INPUT_ID.
node_t *node_esigmoid(grph_t *g, grph_size_t a, grph_size_t b);

// Applies ReLU on A and returns the result
// in a new node. B must be set to GRPH_NO_INPUT_ID.
node_t *node_erelu(grph_t *g, grph_size_t a, grph_size_t b);

// Applies Leaky ReLU on A and returns the result
// in a new node. B must be set to GRPH_NO_INPUT_ID.
node_t *node_eleakyrelu(grph_t *g, grph_size_t a, grph_size_t b);

// Applies Mean-Squared-Error on A and B and returns the
// result in a new node.
node_t *node_mse(grph_t *g, grph_size_t a, grph_size_t b);

// Applies Cross-Entropy-Loss on A and B and returns the result in a
// new node.
node_t *node_cross_entropy_loss(grph_t *g, grph_size_t a, grph_size_t b);

// Applies softmax on A and returns the result in a new node.
// B must e set to GRPH_NO_INPUT_ID.
node_t *node_softmax(grph_t *g, grph_size_t a, grph_size_t b);

// Pushes the gradient from a transpose node to its dependencies and multiplies it
// with the upstream gradient stored in A's grad field.
bool node_transpose_dx(grph_t *g, grph_size_t a);

// Pushes the gradient from a tensor contraction node to its dependencies and multiplies it
// with the upstream gradient stored in A's grad field.
bool node_contract_dx(grph_t *g, grph_size_t a);

// Pushes the gradient from an element-wise addition node to its dependencies and multiplies it
// with the upstream gradient stored in A's grad field.
bool node_eadd_dx(grph_t *g, grph_size_t a);

// Pushes the gradient from an element-wise subtraction node to its dependencies and multiplies it
// with the upstream gradient stored in A's grad field.
bool node_esub_dx(grph_t *g, grph_size_t a);

// Pushes the gradient from an element-wise multiplication node to its dependencies and multiplies it
// with the upstream gradient stored in A's grad field.
bool node_emul_dx(grph_t *g, grph_size_t a);

// Pushes the gradient from an element-wise division node to its dependencies and multiplies it
// with the upstream gradient stored in A's grad field.
bool node_ediv_dx(grph_t *g, grph_size_t a);

// Pushes the gradient from a sigmoid node to its dependencies and multiplies it
// with the upstream gradient stored in A's grad field.
bool node_esigmoid_dx(grph_t *g, grph_size_t a);

// Pushes the gradient from a ReLU node to its dependencies and multiplies it
// with the upstream gradient stored in A's grad field.
bool node_erelu_dx(grph_t *g, grph_size_t a);

// Pushes the gradient from a Leaky ReLU node to its dependencies and multiplies it
// with the upstream gradient stored in A's grad field.
bool node_eleakyrelu_dx(grph_t *g, grph_size_t a);

// Pushes the gradient from a Mean-Squared-Error node to its dependencies and multiplies it
// with the upstream gradient stored in A's grad field.
bool node_mse_dx(grph_t *g, grph_size_t a);

// Pushes the gradient from a Cross-Entropy-Loss node to its dependencies and multiplies it
// with the upstream gradient stored in A's grad field.
bool node_cross_entropy_loss_dx(grph_t *g, grph_size_t a);

// Pushes the gradient from a Softmax node to its dependencies and multiplies it
// with the upstream gradient stored in A's grad field.
bool node_softmax_dx(grph_t *g, grph_size_t a);
