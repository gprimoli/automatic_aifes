#include <aiconfiguration.h>
#include <aiconfiguration_intern.h>

#include "ini.h"

bool load_config(aiconfiguration_t *conf, const char *filename) {
    conf->best_acc = 0;
    return ini_parse(filename, handler, conf);
}

const char *quantization_to_str(Quantization quantization) {
    switch (quantization) {
        case Q31: return "Q31";
        case Q7: return "Q7";
        default: return "F32";
    }
}

const char *layer_type_to_string(Layer_type type) {
    switch (type) {
        case DENSE: return "dense";
        case CONV2D: return "Conv2D";
        case BATCH_NORM: return "batch normalizzation";
        case MAXPOOL2D: return "maxpool2d";
        case RESHAPE: return "reshape";
        case FLATTEN: return "flatten";

        case SIGMOID: return "sigmoid";
        case RELU: return "relu";
        case SOFTMAX: return "softmax";
        case LEAKY_RELU: return "leaky relu";
        case ELU: return "elu";
        case TANH: return "tanh";
        case SOFTSIGN: return "softsign";
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