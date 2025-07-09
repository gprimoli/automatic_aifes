#ifndef AICONFIGURATION_INTERN_H
#define AICONFIGURATION_INTERN_H

#include <ctype.h>
#include <float.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

#include "log.h"
#include "csv.h"
#include "main.h"
#include "memmanager.h"
#include "aifescustom.h"

static int strcmp_ignorecase(const char *s1, const char *s2) {
    while (*s1 && *s2) {
        char c1 = tolower((unsigned char) *s1);
        char c2 = tolower((unsigned char) *s2);
        if (c1 != c2)
            return c1 - c2;
        s1++;
        s2++;
    }
    return tolower((unsigned char) *s1) - tolower((unsigned char) *s2);
}

static Loss parse_loss(const char *value) {
    if (strcmp_ignorecase(value, "mse") == 0) return MSE;
    if (strcmp_ignorecase(value, "crossentropy") == 0) return CROSSENTROPY;
    return UNKNOWN_LOSS;
}

static Optimizer parse_optimizer(const char *value) {
    if (strcmp_ignorecase(value, "adam") == 0) return ADAM;
    if (strcmp_ignorecase(value, "sgd") == 0) return SGD;
    return UNKNOWN_OPTIMIZER;
}

static Quantization parse_quantization(const char *value) {
    if (strcmp_ignorecase(value, "Q31") == 0) return Q31;
    if (strcmp_ignorecase(value, "Q7") == 0) return Q7;
    if (strcmp_ignorecase(value, "Q1") == 0) return Q1;
    return F32;
}

static void parse_int_string_to_array(const char *value, uint16_t *arr, uint32_t max_count) {
    char buf[BUF_MIN];
    strncpy(buf, value, sizeof(buf));
    buf[sizeof(buf) - 1] = '\0';

    char *token = strtok(buf, ",");
    int i = 0;
    while (token != NULL && i < max_count) {
        arr[i++] = atoi(token);
        token = strtok(NULL, ",");
    }
}


static uint32_t count_lines_in_file(const char *filepath) {
    FILE *f = fopen(filepath, "r");
    if (f == NULL) return 0;

    int lines = 0;
    int ch;
    while ((ch = fgetc(f)) != EOF) {
        if (ch == '\n') lines++;
    }

    fclose(f);
    return lines;
}

static Layer_type parse_layer_type(const char *value) {
    if (strcmp_ignorecase(value, "dense") == 0 || strcmp_ignorecase(value, "output") == 0) return DENSE;
    if (strcmp_ignorecase(value, "conv2d") == 0) return CONV2D;
    if (strcmp_ignorecase(value, "batch_normalization") == 0) return BATCH_NORM;
    if (strcmp_ignorecase(value, "maxpool2d") == 0) return MAXPOOL2D;
    if (strcmp_ignorecase(value, "reshape") == 0) return RESHAPE;
    if (strcmp_ignorecase(value, "flatten") == 0) return FLATTEN;
    if (strcmp_ignorecase(value, "sigmoid") == 0) return SIGMOID;
    if (strcmp_ignorecase(value, "relu") == 0) return RELU;
    if (strcmp_ignorecase(value, "softmax") == 0) return SOFTMAX;
    if (strcmp_ignorecase(value, "leaky_relu") == 0) return LEAKY_RELU;
    if (strcmp_ignorecase(value, "elu") == 0) return ELU;
    if (strcmp_ignorecase(value, "tanh") == 0) return TANH;
    if (strcmp_ignorecase(value, "softsign") == 0) return SOFTSIGN;
    return UNKNOWN_LAYER;
}

static int handler(void *data, const char *section, const char *name, const char *value) {
    aiconfiguration_t *conf = data;

    if (strcmp_ignorecase(section, "general") == 0) {
        if (conf->basedir == NULL && strcmp_ignorecase(name, "basedir") != 0) {
            LOG_ERROR("basedir first param in [general]");
            return 0;
        }

        if (strcmp_ignorecase(name, "basedir") == 0) {
            conf->basedir = mem_strdup(value);

            char filepath[BUF_MIN] = {0};
            sprintf(filepath, "%s%c%s", conf->basedir, DIR_SEPARATOR, "y_train.csv");
            conf->sample_train = count_lines_in_file(filepath);

            sprintf(filepath, "%s%c%s", conf->basedir, DIR_SEPARATOR, "y_test.csv");
            conf->sample_test = count_lines_in_file(filepath);

            if (conf->sample_train == 0 || conf->sample_test == 0) {
                return 0;
            }

            if (!open_csv(&x_train, conf->basedir, "x_train.csv", "r")
                || !open_csv(&y_train, conf->basedir, "y_train.csv", "r")
                || !open_csv(&x_test, conf->basedir, "x_test.csv", "r")
                || !open_csv(&y_test, conf->basedir, "y_test.csv", "r")
                || !initLogFile(conf->basedir)) {
                SAFE_EXIT_FAILURE("Errore apertura file CSV");
            }
        } else if (strcmp_ignorecase(name, "training") == 0) {
            conf->training = atoi(value) != 0;
        } else if (strcmp_ignorecase(name, "load") == 0) {
            conf->load = mem_strdup(value);
        } else if (strcmp_ignorecase(name, "save") == 0) {
            conf->save = mem_strdup(value);
        } else if (strcmp_ignorecase(name, "pruning") == 0) {
            conf->pruning = atof(value);
        } else if (strcmp_ignorecase(name, "quantization") == 0) {
            conf->quantization = parse_quantization(value);
        } else {
            return 0;
        }
        return 1;
    }

    if (strcmp_ignorecase(section, "network") == 0) {
        if (strcmp_ignorecase(name, "epochs") == 0) {
            conf->epochs = atoi(value);
        } else if (strcmp_ignorecase(name, "batch_size") == 0) {
            conf->batch_size = atoi(value);
        } else if (strcmp_ignorecase(name, "loss") == 0) {
            conf->loss = parse_loss(value);
        } else if (strcmp_ignorecase(name, "optimizer") == 0) {
            conf->optimizer = parse_optimizer(value);
        } else {
            return 0;
        }
        return 1;
    }

    if (strcmp_ignorecase(section, "dataset") == 0) {
        if (strcmp_ignorecase(name, "input") == 0) {
            parse_int_string_to_array(value, conf->input_shape, 4);
        } else {
            return 0;
        }
        return 1;
    }

    if (strncmp(section, "layer", 5) == 0) {
        int i = atoi(section + 5);
        if (i < 0 || i >= MAX_LAYERS) {
            LOG_ERROR("Layer index %d out of range", i);
            return 0;
        }

        if (i > conf->num_layer) {
            LOG_ERROR("Layer order");
            return 0;
        }

        aiconfigurationlayer_t *l = &conf->layers[i];

        if (l->type == UNKNOWN_LAYER && strcmp_ignorecase(name, "type") == 0) {
            l->type = parse_layer_type(value);
        } else if (strcmp_ignorecase(name, "activation") == 0) {
            l->activation = parse_layer_type(value);
        } else {
            switch (l->type) {
                case DENSE: {
                    if (strcmp_ignorecase(name, "neurons") == 0)
                        l->params.dense.neurons = atoi(value);
                    break;
                }
                case CONV2D: {
                    if (strcmp_ignorecase(name, "filters") == 0)
                        l->params.conv2d.filter = atoi(value);
                    else if (strcmp_ignorecase(name, "kernel_size") == 0)
                        parse_int_string_to_array(value, l->params.conv2d.kernel_size, 2);
                    else if (strcmp_ignorecase(name, "stride") == 0)
                        parse_int_string_to_array(value, l->params.conv2d.stride, 2);
                    else if (strcmp_ignorecase(name, "dilation") == 0)
                        parse_int_string_to_array(value, l->params.conv2d.dilation, 2);
                    else if (strcmp_ignorecase(name, "padding") == 0)
                        parse_int_string_to_array(value, l->params.conv2d.padding, 2);
                    break;
                }
                case BATCH_NORM: {
                    if (strcmp_ignorecase(name, "momentum") == 0)
                        l->params.batch_norm.momentum = atof(value);
                    else if (strcmp_ignorecase(name, "eps") == 0)
                        l->params.batch_norm.eps = strtof(value, NULL);
                    break;
                }
                case MAXPOOL2D: {
                    if (strcmp_ignorecase(name, "pool_size") == 0)
                        parse_int_string_to_array(value, l->params.maxpool2d.pool_size, 2);
                    else if (strcmp_ignorecase(name, "stride") == 0)
                        parse_int_string_to_array(value, l->params.maxpool2d.stride, 2);
                    else if (strcmp_ignorecase(name, "padding") == 0)
                        parse_int_string_to_array(value, l->params.maxpool2d.padding, 2);
                    break;
                }
                case RESHAPE: {
                    if (strcmp_ignorecase(name, "output_dim") == 0)
                        l->params.reshape.output_dim = atoi(value);
                    else if (strcmp_ignorecase(name, "infer_axis") == 0)
                        l->params.reshape.infer_axis = atoi(value);
                    else if (strcmp_ignorecase(name, "output_shape") == 0)
                        parse_int_string_to_array(value, l->params.reshape.output_shape, 2);
                    break;
                }
                default: ;
            }

            switch (l->activation) {
                case LEAKY_RELU: {
                    if (strcmp_ignorecase(name, "alpha") == 0)
                        l->params.leaky_relu.alpha = atof(value);
                    break;
                }
                case ELU: {
                    if (strcmp_ignorecase(name, "alpha") == 0)
                        l->params.elu.alpha = atof(value);
                    break;
                default: ;
                }
            }
        }

        if (i >= conf->num_layer)
            conf->num_layer = i + 1;

        return 1;
    }

    return 1;
}

#endif //AICONFIGURATION_INTERN_H
