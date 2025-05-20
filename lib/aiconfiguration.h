#ifndef AICONFIGURATION_H
#define AICONFIGURATION_H

#include <stdbool.h>

#define MAX_LAYERS 128

#define PARAM_KERNEL_SIZE      0
#define PARAM_KERNEL_STRIDE    1
#define PARAM_KERNEL_DILATION  2
#define PARAM_KERNEL_PADDING   3
#define PARAM_POOL_SIZE        4
#define CHANNEL_AXIS           5
#define FILTERS                6
#define NEURONS                7

typedef enum {
    SIGMOID, RELU, SOFTMAX, LEAKY_RELU, ELU, TANH, SOFTSIGN
} Activation_Fun;

typedef enum {
    DENSE, CONV2D, BATCH_NORM, MAXPOOL2D, RESHAPE, FLATTEN
} Layer_type;

typedef enum {
    ADAM, SGD
} Optimizer;

typedef enum {
    MSE, CROSSENTROPY
} Loss;

typedef struct ayconfigurationlayer{
    Layer_type type;
    Activation_Fun activation;
    int params[8][2];
} aiconfigurationlayer_t;

typedef struct aiconfiguration{
    int num_layer;
    int epochs;
    int batch_size;
    int input_shape[3];

    char *basedir;
    int sample_train;
    int sample_test;

    Loss loss;
    Optimizer optimizer;
    aiconfigurationlayer_t layers[MAX_LAYERS];
} aiconfiguration_t;

bool load_config(const char *filename, aiconfiguration_t *ctx);

const char *layer_type_to_string(Layer_type type);
const char *activation_to_string(Activation_Fun act);
const char *optimizer_to_string(Optimizer opt);
const char *loss_to_string(Loss opt);

#endif //AICONFIGURATION_H
