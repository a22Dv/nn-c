# nn-c
![Language](https://img.shields.io/badge/C%2D17-blue?logo=c&logoColor=white)
![MIT License](https://img.shields.io/badge/License-MIT-green)

![Demo](./public/demo-mnist-128x128x10.gif)
A mini neural network framework built entirely in C.

This framework is built using only the C standard library (Along with Windows-specific headers for console handling)

This framework is more of a personal demonstration that a functional neural network can be built without heavy external dependencies.

## Features

- Minimal dependencies
- Declarative configuration pattern (similar to PyTorch/TensorFlow)
- Support for the most common optimizers, as well as activation functions.

## How To

A demonstration is found in `main.c`, however, the core logic is simple as declaring a model is straightforward, all that is needed is to specify the requirements of the model using structs.

Note that you must provide your own data callbacks, and dashboard function *(Though there is a default dashboard function)*.

```C
dashboard_config_t dash = {
      .show_dashboard = true,
      .passes_interval = 12,  // Show dashboard every 12 mini-batches.
      .dashboard_callback = mnist_dash
  };
  layer_config_t layers[] = {
      (layer_config_t){128, INIT_HE, NDTYPE_ELEAKYRELU},
      (layer_config_t){128, INIT_HE, NDTYPE_ELEAKYRELU},
      (layer_config_t){10, INIT_GLOROT, NDTYPE_SOFTMAX}
  };
  size_t lsize = sizeof(layers) / sizeof(layer_config_t);
  model_config_t config = {
      .epochs = 2,
      .network_depth = lsize,
      .batch_size = 8,
      .data_size = 60000,
      .network = layers,
      .dashboard = dash,
      .input_size = 28 * 28,
      .output_size = 10,
      .optimizer_method = OPT_SGD_ADAM,
      .learning_rate = 0.001f,
      .loss_function_type = NDTYPE_CATEGORICAL_CROSS_ENTROPY_LOSS,
      .data_callback = mnist_data,
      .context = &ctx,
  };
  model_t *model = model_create(&config);
  if (!model) {
    goto error;
  }
  if (!model_fit(model)) {
    goto error;
  }
  if (!model_save(model, "C:/dev/repositories/nn-c/mnist128.weights")) {
    goto error;
  }
```

## License

This project is licensed under the MIT License - see LICENSE for more details.

## Author

a22Dv - a22dev.gl@gmail.com