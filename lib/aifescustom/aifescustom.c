#include <string.h>

#include "log.h"
#include "csv.h"
#include "aifes.h"
#include "memmanager.h"
#include "aifescustom.h"
#include "aifescustom_internal.h"

FILE *x_train = NULL, *y_train = NULL, *x_test = NULL, *y_test = NULL;

aiopti_t *build_model(aiconfiguration_t *conf, aimodel_t *model) {
    aiopti_t *optimizer = NULL;

    uint16_t *input_shape = mem_calloc(4, sizeof(uint16_t));
    input_shape[0] = conf->input_shape[0];
    input_shape[1] = conf->input_shape[1];
    input_shape[2] = conf->input_shape[2];
    input_shape[3] = conf->input_shape[3];

    ailayer_input_f32_t *input_layer = mem_calloc(1, sizeof(ailayer_input_f32_t));
    input_layer->input_dim = conf->input_shape[2] == 0 && conf->input_shape[3] == 0 ? 2 : 4;
    input_layer->input_shape = input_shape;

    model->input_layer = ailayer_input_f32_default(input_layer);

    ailayer_t *layers = model->input_layer;

    for (uint32_t i = 0; i < conf->num_layer; i++) {
        aiconfigurationlayer_t current = conf->layers[i];
        switch (current.type) {
            case DENSE: {
                ailayer_dense_f32_t *l = mem_calloc(1, sizeof(ailayer_dense_f32_t));

                l->neurons = current.params.dense.neurons;
                layers = ailayer_dense_f32_default(l, layers);

                switch (conf->quantization) {
                    case Q31: {
                        l->base.forward = ailayer_dense_forward_Q31;
                        break;
                    }
                    case Q7: {
                        l->base.forward = ailayer_dense_forward_Q7;
                        break;
                    }
                    case Q1: {
                        l->base.forward = ailayer_dense_forward_Q1;
                        break;
                    }
                    default: ;
                }

                break;
            }
            case CONV2D: {
                ailayer_conv2d_f32_t *l = mem_calloc(1, sizeof(ailayer_conv2d_f32_t));

                l->channel_axis = 1;
                l->filter_count = current.params.conv2d.filter;

                l->kernel_size[0] = current.params.conv2d.kernel_size[0];
                l->kernel_size[1] = current.params.conv2d.kernel_size[1];

                l->stride[0] = current.params.conv2d.stride[0];
                l->stride[1] = current.params.conv2d.stride[1];

                l->dilation[0] = current.params.conv2d.dilation[0];
                l->dilation[1] = current.params.conv2d.dilation[1];

                l->padding[0] = current.params.conv2d.padding[0];
                l->padding[1] = current.params.conv2d.padding[1];

                layers = ailayer_conv2d_f32_default(l, layers);

                switch (conf->quantization) {
                    case Q31: {
                        l->base.forward = ailayer_conv2d_forward_Q31;
                        break;
                    }
                    case Q7: {
                        l->base.forward = ailayer_conv2d_forward_Q7;
                        break;
                    }
                    case Q1: {
                        l->base.forward = ailayer_conv2d_forward_Q1;
                        break;
                    }
                    default: ;
                }

                break;
            }
            case BATCH_NORM: {
                ailayer_batch_norm_f32_t *l = mem_calloc(1, sizeof(ailayer_batch_norm_f32_t));
                l->eps = current.params.batch_norm.eps;
                l->momentum = current.params.batch_norm.momentum;
                layers = ailayer_batch_norm_f32_default(l, layers);
                break;
            }
            case MAXPOOL2D: {
                ailayer_maxpool2d_f32_t *l = mem_calloc(1, sizeof(ailayer_maxpool2d_f32_t));

                l->channel_axis = 1;
                l->pool_size[0] = current.params.maxpool2d.pool_size[0];
                l->pool_size[1] = current.params.maxpool2d.pool_size[1];
                l->stride[0] = current.params.maxpool2d.stride[0];
                l->stride[1] = current.params.maxpool2d.stride[1];
                l->padding[0] = current.params.maxpool2d.padding[0];
                l->padding[1] = current.params.maxpool2d.padding[1];
                layers = ailayer_maxpool2d_f32_default(l, layers);
                break;
            }
            case RESHAPE: {
                ailayer_reshape_f32_t *l = mem_calloc(1, sizeof(ailayer_reshape_f32_t));
                l->output_dim = current.params.reshape.output_dim;
                l->infer_axis = current.params.reshape.infer_axis;
                l->output_shape = current.params.reshape.output_shape;
                layers = ailayer_reshape_f32_default(l, layers);
                break;
            }
            case FLATTEN: {
                ailayer_flatten_f32_t *l = mem_calloc(1, sizeof(ailayer_flatten_f32_t));
                layers = ailayer_flatten_f32_default(l, layers);
                break;
            }
            case UNKNOWN_LAYER: break;
            default: return false;
        }

        if (current.type == MAXPOOL2D || current.type == FLATTEN || current.type == BATCH_NORM || current.type ==
            RESHAPE) {
            continue;
        }

        switch (current.activation) {
            case SIGMOID: {
                ailayer_sigmoid_f32_t *l = mem_calloc(1, sizeof(ailayer_sigmoid_f32_t));
                layers = ailayer_sigmoid_f32_default(l, layers);
                break;
            }
            case RELU: {
                ailayer_relu_f32_t *l = mem_calloc(1, sizeof(ailayer_relu_f32_t));
                layers = ailayer_relu_f32_default(l, layers);
                break;
            }
            case SOFTMAX: {
                ailayer_softmax_f32_t *l = mem_calloc(1, sizeof(ailayer_softmax_f32_t));
                layers = ailayer_softmax_f32_default(l, layers);
                break;
            }
            case LEAKY_RELU: {
                ailayer_leaky_relu_f32_t *l = mem_calloc(1, sizeof(ailayer_leaky_relu_f32_t));
                l->alpha = current.params.leaky_relu.alpha;
                layers = ailayer_leaky_relu_f32_default(l, layers);
                break;
            }
            case ELU: {
                ailayer_elu_f32_t *l = mem_calloc(1, sizeof(ailayer_elu_f32_t));
                l->alpha = current.params.elu.alpha;
                layers = ailayer_elu_f32_default(l, layers);
                break;
            }
            case TANH: {
                ailayer_tanh_f32_t *l = mem_calloc(1, sizeof(ailayer_tanh_f32_t));
                layers = ailayer_tanh_f32_default(l, layers);
                break;
            }
            case SOFTSIGN: {
                ailayer_softsign_f32_t *l = mem_calloc(1, sizeof(ailayer_softsign_f32_t));
                layers = ailayer_softsign_f32_default(l, layers);
                break;
            }
            case UNKNOWN_LAYER: break;
            default: return NULL;
        }
    }

    model->output_layer = layers;

    uint32_t output_size = conf->batch_size * conf->layers[conf->num_layer - 1].params.dense.neurons;
    uint32_t input_size = conf->input_shape[2] == 0 && conf->input_shape[3] == 0
                              ? conf->batch_size * conf->input_shape[1]
                              : conf->batch_size * conf->input_shape[1] * conf->input_shape[2] * conf->input_shape[3];

    uint16_t *input_shape_training = mem_calloc(4, sizeof(uint16_t));
    input_shape_training[0] = conf->batch_size;
    input_shape_training[1] = conf->input_shape[1];
    input_shape_training[2] = conf->input_shape[2];
    input_shape_training[3] = conf->input_shape[3];

    uint16_t *output_shape_training = mem_calloc(2, sizeof(uint16_t));
    output_shape_training[0] = conf->batch_size;
    output_shape_training[1] = conf->layers[conf->num_layer - 1].params.dense.neurons;

    conf->x = mem_calloc(1, sizeof(aitensor_t));
    conf->x->dtype = aif32;
    conf->x->dim = conf->input_shape[2] == 0 && conf->input_shape[3] == 0 ? 2 : 4;
    conf->x->shape = input_shape_training;
    conf->x->data = mem_calloc(input_size, sizeof(float));

    conf->y = mem_calloc(1, sizeof(aitensor_t));
    conf->y->dtype = aif32;
    conf->y->dim = 2;
    conf->y->shape = output_shape_training;
    conf->y->data = mem_calloc(output_size, sizeof(float));

    aialgo_compile_model(model);

    uint32_t parameter_memory_size = aialgo_sizeof_parameter_memory(model);
    void *parameter_memory = mem_calloc(parameter_memory_size, sizeof(void));

    aialgo_distribute_parameter_memory(model, parameter_memory, parameter_memory_size);

    switch (conf->loss) {
        case MSE: {
            ailoss_mse_f32_t *loss = mem_calloc(1, sizeof(ailoss_mse_f32_t));
            model->loss = ailoss_mse_f32_default(loss, model->output_layer);
            break;
        }
        case CROSSENTROPY: {
            ailoss_crossentropy_f32_t *loss = mem_calloc(1, sizeof(ailoss_crossentropy_f32_t));
            model->loss = ailoss_crossentropy_f32_default(loss, model->output_layer);
            break;
        }
        default: ;
    }

    aiopti_adam_f32_t *adam_opti = mem_calloc(1, sizeof(aiopti_adam_f32_t));
    adam_opti->learning_rate = 0.01f;
    adam_opti->beta1 = 0.9f;
    adam_opti->beta2 = 0.999f;
    adam_opti->eps = 1e-7f;

    aialgo_initialize_parameters_model(model);

    optimizer = aiopti_adam_f32_default(adam_opti);

    uint32_t memory_size = aialgo_sizeof_training_memory(model, optimizer);
    void *memory_ptr = mem_calloc(memory_size, sizeof(void));

    aialgo_schedule_training_memory(model, optimizer, memory_ptr, memory_size);

    aialgo_init_model_for_training(model, optimizer);

    return optimizer;
}

void save_model(aiconfiguration_t *conf, aimodel_t *model) {
    FILE *f;
    open_csv(&f, conf->basedir, conf->save, "w");

    const ailayer_t *layer = model->input_layer;
    for (int i = 1; i < model->layer_count; i++) {
        layer = layer->output_layer;

        if (strcmp(layer->layer_type->name, "Conv2D") == 0) {
            ailayer_conv2d_t *l = (ailayer_conv2d_t *) layer;
            write_aitensor_to_csv(&(l->weights), f);
            write_aitensor_to_csv(&(l->bias), f);
        } else if (strcmp(layer->layer_type->name, "Dense") == 0) {
            ailayer_dense_t *l = (ailayer_dense_t *) layer;
            write_aitensor_to_csv(&(l->weights), f);
            write_aitensor_to_csv(&(l->bias), f);
        }
    }

    CLOSE_ALL_FILES(f);
}

void load_model(aiconfiguration_t *conf, aimodel_t *model) {
    FILE *f;
    open_csv(&f, conf->basedir, conf->load, "r");

    const ailayer_t *layer = model->input_layer;
    for (int i = 1; i < model->layer_count; i++) {
        layer = layer->output_layer;
        if (strcmp(layer->layer_type->name, "Conv2D") == 0) {
            ailayer_conv2d_t *l = (ailayer_conv2d_t *) layer;
            read_aitensor_from_csv(&(l->weights), f);
            read_aitensor_from_csv(&(l->bias), f);
        } else if (strcmp(layer->layer_type->name, "Dense") == 0) {
            ailayer_dense_t *l = (ailayer_dense_t *) layer;
            read_aitensor_from_csv(&(l->weights), f);
            read_aitensor_from_csv(&(l->bias), f);
        }
    }

    CLOSE_ALL_FILES(f);
}

void run_training(aiconfiguration_t *conf, aimodel_t *model, aiopti_t *optimizer) {
    uint32_t input_elements = (conf->input_shape[2] == 0 && conf->input_shape[3] == 0)
                                  ? conf->batch_size * conf->input_shape[1]
                                  : conf->batch_size * conf->input_shape[1] * conf->input_shape[2] * conf->input_shape
                                    [3];
    uint32_t output_elements = conf->batch_size * conf->layers[conf->num_layer - 1].params.dense.neurons;

    uint32_t batch_train = conf->sample_train / conf->batch_size;

    for (int epoch = 0; epoch < conf->epochs; epoch++) {
        LOG_INFO("Inizio Training");
        LOG_INFO("Epoch %d/%d", epoch + 1, conf->epochs);
        for (int batch = 0; batch < batch_train; batch++) {
            if (!csv_read(conf->x->data, input_elements, x_train) ||
                !csv_read(conf->y->data, output_elements, y_train)) {
                SAFE_EXIT_FAILURE("Errore lettura batch da CSV");
            }

            aialgo_train_model(model, conf->x, conf->y, optimizer, conf->batch_size);
        }
        LOG_INFO("Fine Training\n");

        if (conf->pruning > 0) {
            const uint8_t pruning_steps = 5;
            const uint32_t pruning_step_size = conf->epochs / pruning_steps;

            if (pruning_step_size > 0 && ((epoch + 1) % pruning_step_size == 0)) {
                LOG_INFO("Inizio Pruning");

                const float step = (float) (epoch + 1) / (float) pruning_step_size;
                const float prune_fraction = (conf->pruning * step) / (100.0f * (float) pruning_steps);

                prune_global(model, prune_fraction);

                LOG_INFO("Fine Pruning\t%s\n", get_timestamp());
            }
        }

        run_evaluation(conf, model);

        RESET_ALL_FILES(x_train, y_train);
    }

    if (conf->pruning > 0) {
        LOG_INFO("Inizio Pruning finale");
        prune_global(model, conf->pruning / 100.0f);
        run_evaluation(conf, model);
        LOG_INFO("Fine Pruning finale\t%s\n", get_timestamp());
    }
}

void run_evaluation(aiconfiguration_t *conf, aimodel_t *model) {
    uint32_t input_elements = (conf->input_shape[2] == 0 && conf->input_shape[3] == 0)
                                  ? conf->batch_size * conf->input_shape[1]
                                  : conf->batch_size * conf->input_shape[1] * conf->input_shape[2] * conf->input_shape
                                    [3];
    uint32_t output_elements = conf->batch_size * conf->layers[conf->num_layer - 1].params.dense.neurons;

    uint32_t batch_test = conf->sample_test / conf->batch_size;

    LOG_INFO("Inizio Testing");
    float loss, acc;
    loss = acc = 0.0f;
    for (int batch = 0; batch < batch_test; batch++) {
        if (!csv_read(conf->x->data, input_elements, x_test) ||
            !csv_read(conf->y->data, output_elements, y_test)) {
            SAFE_EXIT_FAILURE("Errore lettura batch da CSV");
        }
        if (!aialgo_calc_loss_acc_model_f32(conf, model, &loss, &acc))
            SAFE_EXIT_FAILURE("Acc loss error");
    }

    LOG_INFO("Loss: %.5f\tAccuracy: %.5f", loss, acc);
    LOG_INFO("Fine Testing\n");
    RESET_ALL_FILES(x_test, y_test);
}
