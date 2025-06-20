#include "aifes.h"
#include "aifescustom.h"

#include <float.h>
#include <string.h>

#include "log.h"
#include "csv.h"
#include "main.h"
#include "memmanager.h"

/* ------------- Private ------------- */

int argmax(aitensor_t *aitensor);

bool write_aitensor_to_csv(aitensor_t *l, FILE *f);

bool read_aitensor_from_csv(aitensor_t *l, FILE *f);

/* ------------- End Private ------------- */

bool aialgo_calc_loss_acc_model_f32(aiconfiguration_t *ctx, aimodel_t *model, float *loss_result,
                                    float *accuracy_result) {
    const aitensor_t *input_tensor = ctx->x;
    const aitensor_t *target_tensor = ctx->y;
    const uint16_t batch_size = input_tensor->shape[0];
    const uint16_t batch_slice_size = model->input_layer->result.shape[0];

    uint32_t i;
    float loss;

    aitensor_t input_batch;
    uint16_t input_batch_shape[input_tensor->dim];
    input_batch.dtype = input_tensor->dtype;
    input_batch.dim = input_tensor->dim;
    input_batch.shape = input_batch_shape;
    input_batch.tensor_params = input_tensor->tensor_params;
    aitensor_t target_batch;
    uint16_t target_batch_shape[target_tensor->dim];
    target_batch.dtype = target_tensor->dtype;
    target_batch.dim = target_tensor->dim;
    target_batch.shape = target_batch_shape;
    target_batch.tensor_params = target_tensor->tensor_params;

    uint32_t input_multiplier = 1;
    for (i = input_tensor->dim - 1; i > 0; i--) {
        input_multiplier *= input_tensor->shape[i];
        input_batch_shape[i] = input_tensor->shape[i];
    }
    input_multiplier *= input_tensor->dtype->size;
    input_batch_shape[0] = batch_slice_size;
    uint32_t target_multiplier = 1;
    for (i = target_tensor->dim - 1; i > 0; i--) {
        target_multiplier *= target_tensor->shape[i];
        target_batch_shape[i] = target_tensor->shape[i];
    }
    target_multiplier *= target_tensor->dtype->size;
    target_batch_shape[0] = batch_slice_size;

    aialgo_set_training_mode_model(model, FALSE);
    aialgo_set_batch_mode_model(model, FALSE);

    uint32_t num_batches = batch_size / batch_slice_size;
    int correct = 0;

    *loss_result = 0;
    for (i = 0; i < num_batches; i++) {
        input_batch.data = input_tensor->data + i * batch_slice_size * input_multiplier;
        target_batch.data = target_tensor->data + i * batch_slice_size * target_multiplier;

        aitensor_t *result_tensor = aialgo_forward_model(model, &input_batch);
        model->loss->calc_loss(model->loss, &target_batch, &loss);

        int pred_label = argmax(result_tensor);
        int true_label = argmax(&target_batch);

        if (pred_label == true_label) {
            correct++;
        }

        *loss_result += loss;
    }
    *accuracy_result = (float) correct / (float) i;
    *loss_result = (float) *loss_result / (float) num_batches;
    return 0;
}

aiopti_t *build_model(aiconfiguration_t *ctx, aimodel_t *model) {
    aiopti_t *optimizer = NULL;

    uint16_t *input_shape = mem_calloc(4, sizeof(uint16_t));
    input_shape[0] = ctx->input_shape[0];
    input_shape[1] = ctx->input_shape[1];
    input_shape[2] = ctx->input_shape[2];
    input_shape[3] = ctx->input_shape[3];

    ailayer_input_f32_t *input_layer = mem_calloc(1, sizeof(ailayer_input_f32_t));
    input_layer->input_dim = ctx->input_shape[2] == 0 && ctx->input_shape[3] == 0 ? 2 : 4;
    input_layer->input_shape = input_shape;

    model->input_layer = ailayer_input_f32_default(input_layer);

    ailayer_t *layers = model->input_layer;

    for (uint32_t i = 0; i < ctx->num_layer; i++) {
        aiconfigurationlayer_t current = ctx->layers[i];
        switch (current.type) {
            case DENSE: {
                ailayer_dense_f32_t *l = mem_calloc(1, sizeof(ailayer_dense_f32_t));

                l->neurons = current.params.dense.neurons;
                layers = ailayer_dense_f32_default(l, layers);

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

    uint32_t output_size = ctx->batch_size * ctx->layers[ctx->num_layer - 1].params.dense.neurons;
    uint32_t input_size = ctx->input_shape[2] == 0 && ctx->input_shape[3] == 0
                              ? ctx->batch_size * ctx->input_shape[1]
                              : ctx->batch_size * ctx->input_shape[1] * ctx->input_shape[2] * ctx->input_shape[3];

    uint16_t *input_shape_training = mem_calloc(4, sizeof(uint16_t));
    input_shape_training[0] = ctx->batch_size;
    input_shape_training[1] = ctx->input_shape[1];
    input_shape_training[2] = ctx->input_shape[2];
    input_shape_training[3] = ctx->input_shape[3];

    uint16_t *output_shape_training = mem_calloc(2, sizeof(uint16_t));
    output_shape_training[0] = ctx->batch_size;
    output_shape_training[1] = ctx->layers[ctx->num_layer - 1].params.dense.neurons;

    ctx->x = mem_calloc(1, sizeof(aitensor_t));
    ctx->x->dtype = aif32;
    ctx->x->dim = ctx->input_shape[2] == 0 && ctx->input_shape[3] == 0 ? 2 : 4;
    ctx->x->shape = input_shape_training;
    ctx->x->data = mem_calloc(input_size, sizeof(float));

    ctx->y = mem_calloc(1, sizeof(aitensor_t));
    ctx->y->dtype = aif32;
    ctx->y->dim = 2;
    ctx->y->shape = output_shape_training;
    ctx->y->data = mem_calloc(output_size, sizeof(float));

    aialgo_compile_model(model);

    uint32_t parameter_memory_size = aialgo_sizeof_parameter_memory(model);
    void *parameter_memory = mem_calloc(parameter_memory_size, sizeof(void));

    aialgo_distribute_parameter_memory(model, parameter_memory, parameter_memory_size);

    switch (ctx->loss) {
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

void prune(aimodel_t *model, float percentage) {
    ailayer_t *last = model->output_layer;
    while (strcmp(last->layer_type->name, "Dense") != 0) {
        last = last->input_layer;
    }

    aitensor_t *weights = &((ailayer_dense_f32_t *) last)->weights;
    unsigned int hist[(BUF_MIN * 2)] = {0};
    float *w = (float *) weights->data;
    size_t N = aimath_tensor_elements(weights);
    float min = fabsf(w[0]), max = fabsf(w[0]);

    for (size_t i = 1; i < N; i++) {
        float val = fabsf(w[i]);
        if (val < min) min = val;
        if (val > max) max = val;
    }
    if (max == min) return;

    for (size_t i = 0; i < N; i++) {
        float val = fabsf(w[i]);
        int bin = (int) ((val - min) / (max - min) * ((BUF_MIN * 2) - 1));
        hist[bin]++;
    }

    //istogramma
    size_t target = (size_t) (percentage * N);
    size_t acc = 0, bin = 0;
    for (; bin < (BUF_MIN * 2); ++bin) {
        acc += hist[bin];
        if (acc >= target) break;
    }

    float pruning_threshold = min + (max - min) * bin / ((BUF_MIN * 2) - 1);
    for (int i = 0; i < N; i++) {
        if (fabsf(w[i]) < pruning_threshold) {
            w[i] = 0.0f;
        }
    }
}

void custom_ailayer_dense_forward(ailayer_t *self) {
    aitensor_t *x_in = &(self->input_layer->result);
    aitensor_t *x_out = &(self->result);
    ailayer_dense_t *layer = (ailayer_dense_t *) (self->layer_configuration);
    aitensor_t *weights = &(layer->weights);
    aitensor_t *bias = &(layer->bias);

    if (self->settings != 2) {
        uint32_t x_in_size = aimath_tensor_elements(x_in);
        uint32_t weights_size = aimath_tensor_elements(weights);

        // --- Quantizzazione INPUT (affine) ---
        aitensor_t *x_in_q = mem_calloc(1, sizeof(aitensor_t));
        x_in_q->dim = x_in->dim;
        x_in_q->shape = x_in->shape;
        x_in_q->dtype = x_in->dtype;
        x_in_q->data = mem_calloc(x_in_size, sizeof(float));

        float input_min = FLT_MAX, input_max = -FLT_MAX;
        for (int i = 0; i < x_in_size; i++) {
            float val = ((float *) x_in->data)[i];
            if (val > input_max) input_max = val;
            if (val < input_min) input_min = val;
        }

        float input_scale = (input_max - input_min) / 255.0f;
        if (input_scale == 0.0f) input_scale = 0.1f;

        int32_t input_zero_point = (int32_t) roundf(-input_min / input_scale);
        if (input_zero_point < 0) input_zero_point = 0;
        if (input_zero_point > 255) input_zero_point = 255;

        for (int i = 0; i < x_in_size; i++) {
            float val = ((float *) x_in->data)[i];
            int32_t q = (int32_t) roundf(val / input_scale) + input_zero_point;
            if (q < 0) q = 0;
            if (q > 255) q = 255;
            ((float *) x_in_q->data)[i] = (float) (q - input_zero_point) * input_scale;
        }

        // --- Quantizzazione PESI (simmetrica int8) ---
        aitensor_t *weights_q = mem_calloc(1, sizeof(aitensor_t));
        weights_q->dim = weights->dim;
        weights_q->shape = weights->shape;
        weights_q->dtype = weights->dtype;
        weights_q->data = mem_calloc(weights_size, sizeof(float));

        float weight_min = FLT_MAX, weight_max = -FLT_MAX;
        for (int i = 0; i < weights_size; i++) {
            float val = ((float *) weights->data)[i];
            if (val > weight_max) weight_max = val;
            if (val < weight_min) weight_min = val;
        }

        float weight_scale = fmaxf(fabsf(weight_max), fabsf(weight_min)) / 127.0f;
        if (weight_scale == 0.0f) weight_scale = 0.1f;

        for (int i = 0; i < weights_size; i++) {
            float val = ((float *) weights->data)[i];
            int8_t q = (int8_t) roundf(val / weight_scale);
            ((float *) weights_q->data)[i] = (float) q * weight_scale;
        }

        // --- Operazione lineare ---
        layer->linear(x_in_q, weights_q, bias, x_out);

        // --- Cleanup ---
        mem_dealloc(x_in_q);
        mem_dealloc(weights_q);
    } else {
        layer->linear(x_in, weights, bias, x_out);
    }
}

void save_model(const aimodel_t *model, FILE *f) {
    const ailayer_t *layer = model->input_layer;
    for (int i = 1; i < model->layer_count; i++) {
        layer = layer->output_layer;

        if (strcmp(layer->layer_type->name, "Conv2d") == 0) {
            ailayer_conv2d_t *l = (ailayer_conv2d_t *) layer;
            write_aitensor_to_csv(&(l->weights), f);
            write_aitensor_to_csv(&(l->bias), f);
        } else if (strcmp(layer->layer_type->name, "Dense") == 0) {
            ailayer_dense_t *l = (ailayer_dense_t *) layer;
            write_aitensor_to_csv(&(l->weights), f);
            write_aitensor_to_csv(&(l->bias), f);
        }
    }
}

void load_model(const aimodel_t *model, FILE *f) {
    const ailayer_t *layer = model->input_layer;
    for (int i = 1; i < model->layer_count; i++) {
        layer = layer->output_layer;
        if (strcmp(layer->layer_type->name, "Conv2d") == 0) {
            ailayer_conv2d_t *l = (ailayer_conv2d_t *) layer;
            read_aitensor_from_csv(&(l->weights), f);
            read_aitensor_from_csv(&(l->bias), f);
        } else if (strcmp(layer->layer_type->name, "Dense") == 0) {
            ailayer_dense_t *l = (ailayer_dense_t *) layer;
            read_aitensor_from_csv(&(l->weights), f);
            read_aitensor_from_csv(&(l->bias), f);
        }
    }
}

void run_training_loop(aiconfiguration_t *ctx, aimodel_t *model, aiopti_t *optimizer,
                       FILE *x_train, FILE *y_train, FILE *x_test, FILE *y_test) {
    uint32_t input_elements = (ctx->input_shape[2] == 0 && ctx->input_shape[3] == 0)
                                  ? ctx->batch_size * ctx->input_shape[1]
                                  : ctx->batch_size * ctx->input_shape[1] * ctx->input_shape[2] * ctx->input_shape
                                    [3];
    uint32_t output_elements = ctx->batch_size * ctx->layers[ctx->num_layer - 1].params.dense.neurons;

    uint32_t batch_train = ctx->sample_train / ctx->batch_size;
    uint32_t batch_test = ctx->sample_test / ctx->batch_size;

    for (int epoch = 0; epoch < ctx->epochs; epoch++) {
        LOG_INFO("Epoch: %d", epoch);

        for (int batch = 0; batch < batch_train; batch++) {
            LOG_INFO("Batch: %d/%d", batch + 1, batch_train);

            if (!csv_read(ctx->x->data, input_elements, x_train) ||
                !csv_read(ctx->y->data, output_elements, y_train)) {
                SAFE_EXIT_FAILURE("Errore lettura batch da CSV");
            }

            aialgo_train_model(model, ctx->x, ctx->y, optimizer, ctx->batch_size);
        }

        float loss, acc;
        for (int batch = 0; batch < batch_test; batch++) {
            if (!csv_read(ctx->x->data, input_elements, x_test) ||
                !csv_read(ctx->y->data, output_elements, y_test)) {
                SAFE_EXIT_FAILURE("Errore lettura batch da CSV");
            }

            aialgo_calc_loss_acc_model_f32(ctx, model, &loss, &acc);
        }

        LOG_INFO("Test loss: %f\tTest acc:%f", loss, acc);
        RESET_ALL_FILES(x_train, y_train, x_test, y_test);
    }
}

void run_evaluation(aiconfiguration_t *ctx, aimodel_t *model, FILE *x_test, FILE *y_test) {
    uint32_t input_elements = (ctx->input_shape[2] == 0 && ctx->input_shape[3] == 0)
                                  ? ctx->batch_size * ctx->input_shape[1]
                                  : ctx->batch_size * ctx->input_shape[1] * ctx->input_shape[2] * ctx->input_shape
                                    [3];
    uint32_t output_elements = ctx->batch_size * ctx->layers[ctx->num_layer - 1].params.dense.neurons;

    uint32_t batch_test = ctx->sample_test / ctx->batch_size;

    float loss, acc;
    for (int batch = 0; batch < batch_test; batch++) {
        if (!csv_read(ctx->x->data, input_elements, x_test) ||
            !csv_read(ctx->y->data, output_elements, y_test)) {
            SAFE_EXIT_FAILURE("Errore lettura batch da CSV");
        }
        aialgo_calc_loss_acc_model_f32(ctx, model, &loss, &acc);
    }

    LOG_INFO("Test loss: %f\tTest acc:%f", loss, acc);
    RESET_ALL_FILES(x_test, y_test);
}

/* ------------- Private ------------- */

int argmax(aitensor_t *aitensor) {
    const float *data = aitensor->data;
    float max = data[0];
    int ixd = 0;

    for (int i = 1; i < aitensor->shape[1]; i++) {
        if (data[i] > max) {
            max = data[i];
            ixd = i;
        }
    }
    return ixd;
}

bool write_aitensor_to_csv(aitensor_t *l, FILE *f) {
    if (!l || !f) return false;

    int len = 0;
    if (l->dim == 2) {
        len = l->shape[0] * l->shape[1];
    } else if (l->dim == 4) {
        len = l->shape[0] * l->shape[1] *
              l->shape[2] * l->shape[3];
    }

    return csv_write(l->data, len, f);
}

bool read_aitensor_from_csv(aitensor_t *l, FILE *f) {
    if (!l || !f) return false;

    int len = 0;
    if (l->dim == 2) {
        len = l->shape[0] * l->shape[1];
    } else if (l->dim == 4) {
        len = l->shape[0] * l->shape[1] *
              l->shape[2] * l->shape[3];
    }

    return csv_read(l->data, len, f);
}

/* ------------- End Private ------------- */
