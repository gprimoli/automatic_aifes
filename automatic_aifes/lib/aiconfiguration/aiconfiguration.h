#ifndef AICONFIGURATION_H
#define AICONFIGURATION_H

#include <stdbool.h>
#include <stdint.h>
#include "basic/base/aimath/aimath_f32.h"

#define MAX_LAYERS 128

typedef struct {
    uint32_t neurons;
} DenseParams;

typedef struct {
    aiscalar_f32_t alpha;
} LeakyReLUParams;

typedef struct {
    aiscalar_f32_t alpha;
} EluParams;

typedef struct {
    uint32_t filter;
    uint16_t kernel_size[2];
    uint16_t stride[2];
    uint16_t dilation[2];
    uint16_t padding[2];
} Conv2DParams;

typedef struct {
    aiscalar_f32_t momentum;
    aiscalar_f32_t eps;
} BatchNormalizationParams;

typedef struct {
    uint16_t pool_size[2];
    uint16_t stride[2];
    uint16_t padding[2];
} MaxPool2DParams;

typedef struct {
    uint8_t output_dim;
    uint8_t infer_axis;
    uint16_t output_shape[2];
} ReshapeParams;

typedef union {
    DenseParams dense;
    LeakyReLUParams leaky_relu;
    EluParams elu;
    Conv2DParams conv2d;
    BatchNormalizationParams batch_norm;
    MaxPool2DParams maxpool2d;
    ReshapeParams reshape;
} LayerParams;

typedef enum {
    UNKNOWN_LAYER,

    DENSE, CONV2D, BATCH_NORM, MAXPOOL2D, RESHAPE, FLATTEN,

    SIGMOID, RELU, SOFTMAX, LEAKY_RELU, ELU, TANH, SOFTSIGN,
} Layer_type;

typedef enum {
    UNKNOWN_OPTIMIZER, ADAM, SGD
} Optimizer;

typedef enum {
    UNKNOWN_LOSS, MSE, CROSSENTROPY
} Loss;

typedef struct ayconfigurationlayer {
    Layer_type type;
    Layer_type activation;
    LayerParams params;
} aiconfigurationlayer_t;

typedef struct aiconfiguration {
    char *basedir;
    bool training;
    char *load;
    char *save;

    Loss loss;
    Optimizer optimizer;

    uint32_t epochs;
    uint32_t batch_size;

    uint16_t input_shape[4]; // 1) t 2) z 3) x 4) y
    aiconfigurationlayer_t layers[MAX_LAYERS];

    //Calcolati
    uint32_t num_layer;
    uint32_t sample_train;
    uint32_t sample_test;
    aitensor_t *x;
    aitensor_t *y;
} aiconfiguration_t;

bool load_config(const char *filename, aiconfiguration_t *ctx);

#endif //AICONFIGURATION_H

/*
// TODO implement;





const char *optimizer_to_string(Optimizer opt);
const char *loss_to_string(Loss opt);
const char *layer_type_to_string(Layer_type type);
const char *activation_to_string(Activation_Fun act);

const char *layer_type_to_string(Layer_type type) {
    switch (type) {
        case DENSE: return "dense";
        case CONV2D: return "conv2d";
        case MAXPOOL2D: return "maxpool2d";
        case FLATTEN: return "flatten";
        default: return "unknown";
    }
}

const char *activation_to_string(Activation_Fun act) {
    switch (act) {
        case RELU: return "relu";
        case SIGMOID: return "sigmoid";
        case SOFTMAX: return "softmax";
        default: return "unknown";
    }
}

const char *optimizer_to_string(Optimizer opt) {
    switch (opt) {
        case ADAM: return "adam";
        case SGD: return "sgd";
        default: return "unknown";
    }
}

const char *loss_to_string(Loss opt){
    switch (opt) {
        case MSE: return "Mean Squared Error";
        case CROSSENTROPY: return "Crossentropy";
        default: return "unknown";
    }
}
 */
