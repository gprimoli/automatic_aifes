#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

#include "Log.h"
#include "ini.h"
#include "aiconfiguration.h"



Layer_type parse_layer_type(const char *value);
Activation_Fun parse_activation(const char *value);
Optimizer parse_optimizer(const char *value);
Loss parse_loss(const char *value);
void parse_int_string_to_array(const char *value, int *arr, int max_count);
int handler(void *data, const char *section, const char *name, const char *value);


bool load_config(const char *filename, aiconfiguration_t *ctx) {
    if (ini_parse(filename, handler, ctx) < 0) {
        LOG_ERROR("Cannot load INI file: %s", filename);
        return false;
    }
    return true;
}


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


/*-------------------------PRIVATE-------------------------*/


int handler(void *data, const char *section, const char *name, const char *value) {
    aiconfiguration_t *ctx = data;

    if (strcmp(section, "network") == 0) {
        if (strcmp(name, "epochs") == 0) ctx->epochs = atoi(value);
        else if (strcmp(name, "batch_size") == 0) ctx->batch_size = atoi(value);
        else if (strcmp(name, "optimizer") == 0) ctx->optimizer = parse_optimizer(value);
        else if (strcmp(name, "loss") == 0) ctx->loss = parse_loss(value);
        return 1;
    }

    if (strcmp(section, "dataset") == 0) {
        if (strcmp(name, "input") == 0) parse_int_string_to_array(value, ctx->input_shape, 3);
        else if (strcmp(name, "sample_train") == 0) ctx->sample_train = atoi(value);
        else if (strcmp(name, "sample_test") == 0) ctx->sample_test = atoi(value);
        else if (strcmp(name, "basedir") == 0) ctx->basedir = strdup(value);
    }

    if (strncmp(section, "layer", 5) == 0) {
        int i = atoi(section + 5); // Aritmetica dei puntatori se incremento mi sposto nella parola strlen("layer") = 5
        if (i < 0 || i >= MAX_LAYERS) {
            LOG_ERROR("Layer index %d out of range", i);
            return 0;
        }
        aiconfigurationlayer_t *l = &ctx->layers[i];
        if (strcmp(name, "type") == 0) l->type = parse_layer_type(value);
        else if (strcmp(name, "activation") == 0) l->activation = parse_activation(value);
        else if (strcmp(name, "neurons") == 0) l->params[NEURONS][0] = atoi(value);
        else if (strcmp(name, "filters") == 0) l->params[FILTERS][0] = atoi(value);
        else if (strcmp(name, "channel_axis") == 0) l->params[CHANNEL_AXIS][0] = atoi(value);
        else if (strcmp(name, "kernel_size") == 0) parse_int_string_to_array(value, l->params[PARAM_KERNEL_SIZE], 2);
        else if (strcmp(name, "kernel_stride") == 0) parse_int_string_to_array(value, l->params[PARAM_KERNEL_STRIDE], 2);
        else if (strcmp(name, "kernel_dilation") == 0) parse_int_string_to_array(value, l->params[PARAM_KERNEL_DILATION], 2);
        else if (strcmp(name, "kernel_padding") == 0) parse_int_string_to_array(value, l->params[PARAM_KERNEL_PADDING], 2);
        else if (strcmp(name, "kernel_pool_size") == 0) parse_int_string_to_array(value, l->params[PARAM_POOL_SIZE], 2);

        if (i >= ctx->num_layer) ctx->num_layer = i + 1;
        return 1;
    }

    return 0;
}

Layer_type parse_layer_type(const char *value) {
    if (strcmp(value, "dense") == 0 || strcmp(value, "output") == 0) return DENSE;
    if (strcmp(value, "conv2d") == 0) return CONV2D;
    if (strcmp(value, "maxpool2d") == 0) return MAXPOOL2D;
    return FLATTEN;//if (strcmp(value, "flatten") == 0)
}

Activation_Fun parse_activation(const char *value) {
    if (strcmp(value, "relu") == 0) return RELU;
    if (strcmp(value, "sigmoid") == 0) return SIGMOID;
    return SOFTMAX; //if (strcmp(value, "softmax") == 0)
}

Optimizer parse_optimizer(const char *value) {
    if (strcmp(value, "adam") == 0) return ADAM;
    if (strcmp(value, "sgd") == 0) return SGD;
    return SGD;
}

Loss parse_loss(const char *value) {
    if (strcmp(value, "mse") == 0) return MSE;
    return CROSSENTROPY; //if (strcmp(value, "crossentropy") == 0)
}

void parse_int_string_to_array(const char *value, int *arr, int max_count) {
    char buf[128];
    strncpy(buf, value, sizeof(buf));
    buf[sizeof(buf) - 1] = '\0';

    char *token = strtok(buf, ",");
    int i = 0;
    while (token != NULL && i < max_count) {
        arr[i++] = atoi(token);
        token = strtok(NULL, ",");
    }
}
/*---------------------------------------------------------*/
