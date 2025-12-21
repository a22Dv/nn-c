/**
 * model.h
 *
 * BRIEF:
 * Declarations for the model_t struct and its related functions.
 * Handles the user-facing API and training dashboard.
 */

#pragma once

#include <stddef.h>
#include "core/graph.h"
#include "core/network.h"

typedef struct {
  grph_size_t neuron_count;
  initialization_t initialization_function;
  node_type_t activation_function;
} layer_config_t;

typedef struct model model_t;

typedef struct {
  size_t epoch_count;
  size_t pass_count;
  tnsr_type_t training_loss;
} model_state_t;

typedef struct {
  bool show_dashboard;
  size_t passes_interval;
  void (*dashboard_callback)(
      const grph_t *graph,
      const model_t *model,
      const tnsr_t *input,
      const tnsr_t *output,
      const tnsr_t *expected
  );
} dashboard_config_t;

typedef struct {
  size_t epochs;                   
  size_t network_depth;          
  size_t batch_size;              
  size_t data_size;            
  layer_config_t *network;     
  dashboard_config_t dashboard;    
  grph_size_t input_size;         
  grph_size_t output_size;         
  optimizer_t optimizer_method;   
  tnsr_type_t learning_rate;       
  node_type_t loss_function_type;  
  bool (*data_callback)(
      size_t batch_size,
      tnsr_t **input_out,
      tnsr_t **expected_out,
      void *context  // Context pointer.
  );
  void *context;
} model_config_t;

typedef struct model {
  model_config_t config;    
  model_state_t state;     
  dense_layer_t *layers[];  
} model_t;

// Creates a model based on a given configuration.
model_t *model_create(model_config_t *config);

// Deallocates the model and sets the pointer to NULL.
// Passing NULL is a no-op.
void model_destroy(model_t **m);

// Saves the model weights to disk.
// DOES NOT save:
// dashboard_callback,
// context,
// data_callback.
bool model_save(model_t *m, char *location);

// Loads a model from disk.
model_t *model_load(char *location);

// Trains a model.
bool model_fit(model_t *m);

// Does a forward-pass on the given model with the given data.
tnsr_t *model_infer(model_t *m, tnsr_t *data);

// Generic training debugging dashboard.
void model_generic_dashboard(
    const grph_t *graph,
    const model_t *model,
    const tnsr_t *input,
    const tnsr_t *output,
    const tnsr_t *expected
);