/**
 * network.h
 *
 * BRIEF:
 * Declarations for layer type
 * structs and their functions.
 * 
 * TODO:
 * Implement optimizers.
 */

#pragma once

#include "core/graph.h"

typedef enum {
  OPT_SGD,
  OPT_SGD_MOMENTUM,
  OPT_SGD_RMS_PROP,
  OPT_SGD_ADAM,
} optimizer_t;

typedef enum {
  INIT_HE,
  INIT_GLOROT,
  INIT_RANDOM_UNIFORM
} initialization_t;

typedef struct dense_layer {
  tnsr_t *weights;
  tnsr_t *biases;
  grph_size_t weights_id;
  grph_size_t biases_id;
  node_type_t function_type;
  void *optimizer_data;
  bool (*optimizer)(struct dense_layer *);
  void (*opt_dtor)(void *);
} dense_layer_t;

// Creates a dense layer with the specified characteristics,
// then initializes it depending on the function.
// Falls back to He initialization otherwise.
dense_layer_t *dense_layer_create(
    grph_size_t fan_in,
    grph_size_t fan_out,
    initialization_t init,
    node_type_t function,
    optimizer_t optimizer
);

// Deallocates the dense layer.
void dense_layer_destroy(dense_layer_t **dl);

// Adds the layer to the graph.
bool dense_layer_add_to_graph(grph_t **g, dense_layer_t *dl);

// Resets the layer's graph IDs. Must be called before adding to another instance
// of the graph.
void dense_layer_remove_from_graph(dense_layer_t *dl);

// Passes the input through the layer and returns the operation index on the graph.
grph_size_t dense_layer_passthrough(grph_t **g, dense_layer_t *dl, grph_size_t input);

// Updates the layer through the gradients of its nodes
// as well as running any optimizers. Modifes node gradients.
bool dense_layer_update(grph_t **g, dense_layer_t *dl);

// Prints the layer's weights and biases to stdout.
void dense_layer_dbgprint(dense_layer_t *dl);

// Default stochastic gradient descent.
bool dense_layer_sgd(dense_layer_t *dl);

// Stochastic gradient descent with momentum.
bool dense_layer_sgd_momentum(dense_layer_t *dl);

// Stochastic gradient descent with Root Mean Square Propagation.
bool dense_layer_sgd_rms_prop(dense_layer_t *dl);

// Stochastic gradient descent with Adaptive Moment Estimation.
bool dense_layer_sgd_adam(dense_layer_t *dl);