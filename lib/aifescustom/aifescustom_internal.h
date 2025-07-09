#ifndef AIFESCUSTOM_INTERNAL_H
#define AIFESCUSTOM_INTERNAL_H

#include <stdbool.h>
#include <float.h>

#include "main.h"
#include "aifes.h"

#define CLAMP(x, low, high) ((x) < (low) ? (low) : ((x) > (high) ? (high) : (x)))

static int argmax(aitensor_t *aitensor) {
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

static bool aialgo_calc_loss_acc_model_f32(aiconfiguration_t *conf, aimodel_t *model, float *loss_result,
                                           float *accuracy_result) {
    float loss = 0.0f;
    const aitensor_t *input_tensor = conf->x;
    const aitensor_t *target_tensor = conf->y;
    const uint16_t batch_size = input_tensor->shape[0];
    const uint16_t batch_slice_size = model->input_layer->result.shape[0];
    const uint32_t num_batches = batch_size / batch_slice_size;

    if (num_batches == 0) return false;


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
    for (uint32_t i = input_tensor->dim - 1; i > 0; i--) {
        input_multiplier *= input_tensor->shape[i];
        input_batch_shape[i] = input_tensor->shape[i];
    }
    input_multiplier *= input_tensor->dtype->size;
    input_batch_shape[0] = batch_slice_size;
    uint32_t target_multiplier = 1;
    for (uint32_t i = target_tensor->dim - 1; i > 0; i--) {
        target_multiplier *= target_tensor->shape[i];
        target_batch_shape[i] = target_tensor->shape[i];
    }
    target_multiplier *= target_tensor->dtype->size;
    target_batch_shape[0] = batch_slice_size;

    aialgo_set_training_mode_model(model, FALSE);
    aialgo_set_batch_mode_model(model, FALSE);

    float tmp_loss = 0;
    float tmp_acc = 0;

    for (uint32_t i = 0; i < num_batches; i++) {
        input_batch.data = input_tensor->data + i * batch_slice_size * input_multiplier;
        target_batch.data = target_tensor->data + i * batch_slice_size * target_multiplier;

        aitensor_t *result_tensor = aialgo_forward_model(model, &input_batch);

        int pred_label, true_label;
        if (conf->loss == CROSSENTROPY) {
            pred_label = argmax(result_tensor);
            true_label = argmax(&target_batch);
        } else {
            pred_label = ((float *) result_tensor->data)[0] > 0.5 ? 1 : 0;
            true_label = ((float *) target_batch.data)[0] == 1.f ? 1 : 0;
        }

        if (pred_label == true_label) {
            tmp_acc++;
        }

        model->loss->calc_loss(model->loss, &target_batch, &loss);
        tmp_loss += loss;
    }

    tmp_loss = tmp_loss / (float) num_batches;
    tmp_acc = tmp_acc / (float) num_batches;

    const float alpha = 0.1f; //Exponential Moving Average (EMA)
    *loss_result = (*loss_result != 0) ? (alpha * tmp_loss + (1.0f - alpha) * (*loss_result)) : tmp_loss;
    *accuracy_result = *accuracy_result != 0 ? (tmp_acc + *accuracy_result) / 2 : tmp_acc;

    return true;
}

/*--------------------------------------------------------------------*/
/*####################################################################*/
/*####################################################################*/
/*####################################################################*/
/*####################################################################*/
/*####################################################################*/
/*####################################################################*/
/*####################################################################*/
/*-------------------------- Pruning ---------------------------------*/

static void prune_global(aimodel_t *model, float prune_percentage) {
    unsigned int hist[256] = {0};
    float min = 0, max = 0;
    size_t total_weights = 0;

    ailayer_t *layer = model->input_layer;
    for (int i = 1; i < model->layer_count; i++) {
        layer = layer->output_layer;

        const aitensor_t *weights = NULL;
        if (strcmp(layer->layer_type->name, "Dense") == 0) {
            weights = &((ailayer_dense_f32_t *) layer)->weights;
        } else if (strcmp(layer->layer_type->name, "Conv2D") == 0) {
            weights = &((ailayer_conv2d_f32_t *) layer)->weights;
        }

        if (weights && weights->data) {
            float *w = (float *) weights->data;
            size_t N = aimath_tensor_elements(weights);

            for (size_t y = 0; y < N; y++) {
                float val = fabsf(w[y]);
                if (val < min) min = val;
                if (val > max) max = val;
            }

            total_weights += N;
        }
    }

    if (min == max || total_weights == 0) {
        SAFE_EXIT_FAILURE("Pruning non eseguito: tutti i pesi hanno lo stesso valore o sono assenti.");
    }

    layer = model->input_layer;
    for (int i = 1; i < model->layer_count; i++) {
        layer = layer->output_layer;

        const aitensor_t *weights = NULL;
        if (strcmp(layer->layer_type->name, "Dense") == 0) {
            weights = &((ailayer_dense_f32_t *) layer)->weights;
        } else if (strcmp(layer->layer_type->name, "Conv2D") == 0) {
            weights = &((ailayer_conv2d_f32_t *) layer)->weights;
        }

        if (weights && weights->data) {
            float *w = (float *) weights->data;
            size_t N = aimath_tensor_elements(weights);

            for (size_t y = 0; y < N; y++) {
                float val = fabsf(w[y]);
                int bin = (int) (((val - min) / (max - min)) * (256 - 1));
                if (bin > 255) bin = 255;
                hist[bin]++;
            }
        }
    }


    size_t target = (size_t) (prune_percentage * total_weights);
    LOG_INFO("Pesi prunati %zu/%zu", target, total_weights);

    size_t acc = 0;
    int bin_cutoff = 0;

    for (; bin_cutoff < 256; bin_cutoff++) {
        acc += hist[bin_cutoff];
        if (acc >= target) break;
    }

    float threshold = min + (max - min) * bin_cutoff / (256 - 1);

    layer = model->input_layer;
    for (int i = 1; i < model->layer_count; i++) {
        layer = layer->output_layer;

        aitensor_t *weights = NULL;
        if (strcmp(layer->layer_type->name, "Dense") == 0) {
            weights = &((ailayer_dense_f32_t *) layer)->weights;
        } else if (strcmp(layer->layer_type->name, "Conv2D") == 0) {
            weights = &((ailayer_conv2d_f32_t *) layer)->weights;
        }

        if (weights && weights->data) {
            float *w = (float *) weights->data;
            size_t N = aimath_tensor_elements(weights);

            for (size_t y = 0; y < N; y++) {
                if (fabsf(w[y]) < threshold) {
                    w[y] = 0.0f;
                }
            }
        }
    }
}

/*--------------------------------------------------------------------*/
/*####################################################################*/
/*####################################################################*/
/*####################################################################*/
/*####################################################################*/
/*####################################################################*/
/*####################################################################*/
/*####################################################################*/
/*-------------------------- Quantization ----------------------------*/


static bool calc_scale_and_zero_point(Quantization qType, const aitensor_t *t, aimath_q31_params_t *qParam) {
    if (!t || !t->data || !qParam) return false;

    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;

    float *data = (float *) t->data;
    int size = aimath_tensor_elements(t);

    for (int i = 0; i < size; i++) {
        float v = data[i];
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }

    if (min_val == max_val) {
        qParam->shift = 0;
        qParam->zero_point = 0;
        return true;
    }

    int qmin, qmax;
    switch (qType) {
        case Q31:
            qmin = INT32_MIN;
            qmax = INT32_MAX;
            break;
        case Q7:
            qmin = INT8_MIN;
            qmax = INT8_MAX;
            break;
        case Q1:
            qmin = INT2_MIN;
            qmax = INT2_MAX;
            break;
        default: return false;
    }

    float float_range = max_val - min_val;
    int shift = 0;

    while ((float_range * (1 << shift)) < (qmax - qmin) && shift < 31) {
        shift++;
    }

    int32_t zp = (int32_t) (qmin - roundf(min_val * (1 << shift)));

    qParam->shift = (uint16_t) shift;
    qParam->zero_point = zp;

    return true;
}

static float fake_quantize_value_shift(Quantization qType, float value, aimath_q31_params_t *qParam) {
    int qmax, qmin;
    qmax = qmin = 0;
    switch (qType) {
        case Q31:
            qmax = INT32_MAX;
            qmin = INT32_MIN;
            break;

        case Q7:
            qmax = INT8_MAX;
            qmin = INT8_MIN;
            break;

        case Q1:
            qmax = INT2_MAX;
            qmin = INT2_MIN;
            break;

        default:
            SAFE_EXIT_FAILURE("Errore quantizzazione: tipo non supportato");
    }

    int quantized = (int) roundf(value * (float) (1 << qParam->shift)) + qParam->zero_point;
    quantized = quantized < qmin ? qmin : (quantized > qmax ? qmax : quantized);
    return ((float) (quantized - qParam->zero_point)) / (float) (1 << qParam->shift);
}

static void quantize(aimodel_t *model, Quantization qType) {
    static bool already_quantized = false;
    if (already_quantized) return;

    ailayer_t *layer = model->input_layer;
    for (int i = 1; i < model->layer_count; i++) {
        layer = layer->output_layer;
        aitensor_t *weights = NULL;
        if (strcmp(layer->layer_type->name, "Dense") == 0) {
            weights = &((ailayer_dense_f32_t *) layer)->weights;
        } else if (strcmp(layer->layer_type->name, "Conv2D") == 0) {
            weights = &((ailayer_conv2d_f32_t *) layer)->weights;
        }

        if (weights && weights->data) {
            float *w = (float *) weights->data;
            size_t N = aimath_tensor_elements(weights);

            aimath_q31_params_t qParams;
            if (!calc_scale_and_zero_point(qType, weights, &qParams))
                SAFE_EXIT_FAILURE("Errore quantizzazione");

            for (size_t y = 0; y < N; y++) {
                w[y] = fake_quantize_value_shift(qType, w[y], &qParams);
            }
        }
    }

    already_quantized = true;
}


/*-------------------------- Dense --------------------------*/

static void aimath_f32_default_linear_quantized(Quantization qType, const aitensor_t *a, const aitensor_t *b,
                                                const aitensor_t *c,
                                                aitensor_t *result) {
    uint16_t i, j, k;
    float sum;

    float *a_data = (float *) a->data;
    float *b_data = (float *) b->data;
    float *c_data = c != 0 ? (float *) c->data : 0;
    float *result_data = (float *) result->data;

    aimath_q31_params_t qParam_a, qParam_b;

    if (!calc_scale_and_zero_point(qType, a, &qParam_a))
        SAFE_EXIT_FAILURE("Errore quantizzazione");
    if (!calc_scale_and_zero_point(qType, b, &qParam_b))
        SAFE_EXIT_FAILURE("Errore quantizzazione");

    for (i = 0; i < a->shape[0]; i++) {
        for (j = 0; j < b->shape[1]; j++) {
            sum = 0.0f;
            for (k = 0; k < a->shape[1]; k++) {
                float a_val = a_data[i * a->shape[1] + k];
                float b_val = b_data[k * b->shape[1] + j];
                float a_q = fake_quantize_value_shift(qType, a_val, &qParam_a);
                float b_q = fake_quantize_value_shift(qType, b_val, &qParam_b);
                sum += a_q * b_q;
            }
            if (c != 0) {
                // Bias add
                sum += c_data[j];
            }
            result_data[i * result->shape[1] + j] = sum;
        }
    }
    return;
}


static void ailayer_dense_forward_Q31(ailayer_t *self) {
    aitensor_t *x_in = &(self->input_layer->result);
    aitensor_t *x_out = &(self->result);
    ailayer_dense_t *layer = (ailayer_dense_t *) (self->layer_configuration);
    aitensor_t *weights = &(layer->weights);
    aitensor_t *bias = &(layer->bias);

    aimath_f32_default_linear_quantized(Q31, x_in, weights, bias, x_out);
    return;
}

static void ailayer_dense_forward_Q7(ailayer_t *self) {
    aitensor_t *x_in = &(self->input_layer->result);
    aitensor_t *x_out = &(self->result);
    ailayer_dense_t *layer = (ailayer_dense_t *) (self->layer_configuration);
    aitensor_t *weights = &(layer->weights);
    aitensor_t *bias = &(layer->bias);

    aimath_f32_default_linear_quantized(Q7, x_in, weights, bias, x_out);
    return;
}

static void ailayer_dense_forward_Q1(ailayer_t *self) {
    aitensor_t *x_in = &(self->input_layer->result);
    aitensor_t *x_out = &(self->result);
    ailayer_dense_t *layer = (ailayer_dense_t *) (self->layer_configuration);
    aitensor_t *weights = &(layer->weights);
    aitensor_t *bias = &(layer->bias);

    aimath_f32_default_linear_quantized(Q1, x_in, weights, bias, x_out);
    return;
}

/*-------------------------- Conv2d --------------------------*/

static void aimath_f32_default_conv2d_add_quantized(
    Quantization qType,
    const aitensor_t *input,
    const uint16_t stride[2], // [s_h, s_w]
    const uint16_t dilation[2], // [d_h, d_w]
    const int16_t padding[2][2],
    const aitensor_t *kernel,
    const void *bias,
    const uint8_t rotated_kernel, // wether or not the kernel is rotated by 180°; default is TRUE (1)
    const int16_t *input_use_dims, // Indices of dimensions for h and w
    const int16_t *output_use_dims, // Indices of dimensions for h and w
    const int16_t *kernel_use_dims, // Indices of dimensions for h and w
    aitensor_t *output) {
    // Input format: NCHW
    // Weights format: FCHW

    uint16_t n_h; // Input height
    uint16_t n_w; // Input width
    uint16_t o_h; // Output height
    uint16_t o_w; // Output width
    uint16_t k_h;
    uint16_t k_w;

    uint32_t acc;
    int8_t i;

    // ------------- Calculate the index multiplier for the given dimensions -----------------
    uint32_t x_1;
    uint32_t x_2;
    uint32_t x_3 = 0;
    acc = 1;
    for (i = input->dim - 1; i >= 0; i--) {
        if (input_use_dims[i] == -1) {
            x_1 = acc;
            n_h = input->shape[i];
        } else if (input_use_dims[i] == -2) {
            x_2 = acc;
            n_w = input->shape[i];
        } else x_3 += input_use_dims[i] * acc;
        acc *= input->shape[i];
    }

    uint32_t k_1;
    uint32_t k_2;
    uint32_t k_3 = 0;
    acc = 1;
    for (i = kernel->dim - 1; i >= 0; i--) {
        if (kernel_use_dims[i] == -1) {
            k_1 = acc;
            k_h = kernel->shape[i];
        } else if (kernel_use_dims[i] == -2) {
            k_2 = acc;
            k_w = kernel->shape[i];
        } else k_3 += kernel_use_dims[i] * acc;
        acc *= kernel->shape[i];
    }

    uint32_t y_1;
    uint32_t y_2;
    uint32_t y_3 = 0;
    acc = 1;
    for (i = output->dim - 1; i >= 0; i--) {
        if (output_use_dims[i] == -1) {
            y_1 = acc;
            o_h = output->shape[i];
        } else if (output_use_dims[i] == -2) {
            y_2 = acc;
            o_w = output->shape[i];
        } else {
            y_3 += output_use_dims[i] * acc;
        }
        acc *= output->shape[i];
    }
    // -------------------------------------------------------------

    // Stride
    uint16_t s_h = stride[0];
    uint16_t s_w = stride[1];
    // Dilation
    uint16_t d_h = dilation[0];
    uint16_t d_w = dilation[1];

    int16_t p_h0 = padding[0][0];
    int16_t p_h1 = padding[0][1];
    int16_t p_w0 = padding[1][0];
    int16_t p_w1 = padding[1][1];


    int32_t y_h_idx, y_w_idx, k_h_idx, k_w_idx;
    // Output dimensions: floor((N + P0 + P1 - (D * (K - 1) + 1)) / S + 1)
    // N: Input shape
    // P0 and P1: Left and right input padding
    // D: Kernel dilation
    // K: Kernel size
    // S: Stride
    uint32_t out_h = (uint32_t) ((n_h + p_h0 + p_h1 - (d_h * (k_h - 1) + 1)) / s_h) + 1;
    uint32_t out_w = (uint32_t) ((n_w + p_w0 + p_w1 - (d_w * (k_w - 1) + 1)) / s_w) + 1;

    float *kernel_data = kernel->data;
    float *y_data = output->data;
    float *x_data = input->data;
    float sum;
    uint32_t idx_h, idx_w; // Auxilary index variables


    aimath_q31_params_t qParam_a, qParam_b;
    if (!calc_scale_and_zero_point(qType, input, &qParam_a))
        SAFE_EXIT_FAILURE("Errore quantizzazione");
    if (!calc_scale_and_zero_point(qType, kernel, &qParam_b))
        SAFE_EXIT_FAILURE("Errore quantizzazione");


    for (y_h_idx = 0; y_h_idx < out_h; y_h_idx++) {
        for (y_w_idx = 0; y_w_idx < out_w; y_w_idx++) {
            sum = 0.0f;
            if (rotated_kernel == FALSE) {
                for (k_h_idx = 0; k_h_idx < k_h; k_h_idx++) {
                    for (k_w_idx = 0; k_w_idx < k_w; k_w_idx++) {
                        idx_h = s_h * y_h_idx + d_h * k_h_idx;
                        idx_w = s_w * y_w_idx + d_w * k_w_idx;
                        // Check for padding constraints
                        // (Crop some values at zero to make comparing signed and unsigned integers possible)
                        if (idx_h >= (p_h0 < 0 ? 0 : p_h0) && idx_w >= (p_w0 < 0 ? 0 : p_w0)
                            && idx_h < ((n_h + p_h0) < 0 ? 0 : (n_h + p_h0)) && idx_w < ((n_w + p_w0) < 0
                                    ? 0
                                    : (n_w + p_w0))) {
                            float a_val = x_data[x_1 * (idx_h - p_h0) + x_2 * (idx_w - p_w0) + x_3];
                            float b_val = kernel_data[k_1 * k_h_idx + k_2 * k_w_idx + k_3];
                            float a_q = fake_quantize_value_shift(qType, a_val, &qParam_a);
                            float b_q = fake_quantize_value_shift(qType, b_val, &qParam_b);
                            sum += a_q * b_q;
                        }
                    }
                }
            } else {
                // 180° Rotated kernel
                for (k_h_idx = 0; k_h_idx < k_h; k_h_idx++) {
                    for (k_w_idx = 0; k_w_idx < k_w; k_w_idx++) {
                        idx_h = s_h * y_h_idx + d_h * k_h_idx;
                        idx_w = s_w * y_w_idx + d_w * k_w_idx;
                        // Check for padding constraints
                        // (Crop some values at zero to make comparing signed and unsigned integers possible)
                        if (idx_h >= (p_h0 < 0 ? 0 : p_h0) && idx_w >= (p_w0 < 0 ? 0 : p_w0)
                            && idx_h < ((n_h + p_h0) < 0 ? 0 : (n_h + p_h0)) && idx_w < ((n_w + p_w0) < 0
                                    ? 0
                                    : (n_w + p_w0))) {
                            float a_val = x_data[x_1 * (idx_h - p_h0) + x_2 * (idx_w - p_w0) + x_3];
                            float b_val = kernel_data[k_1 * (k_h - k_h_idx - 1) + k_2 * (k_w - k_w_idx - 1) + k_3];
                            float a_q = fake_quantize_value_shift(qType, a_val, &qParam_a);
                            float b_q = fake_quantize_value_shift(qType, b_val, &qParam_b);
                            sum += a_q * b_q;
                        }
                    }
                }
            }
            if (bias != 0) {
                sum += *((float *) bias);
            }
            y_data[y_1 * y_h_idx + y_2 * y_w_idx + y_3] += sum;
        }
    }
    return;
}

static void aimath_f32_default_conv2d_fwd_quantized(
    Quantization qType,
    const aitensor_t *input,
    const uint16_t stride[2], // [s_h, s_w]
    const uint16_t dilation[2], // [d_h, d_w]
    const uint16_t padding[2],
    const aitensor_t *weights,
    const aitensor_t *bias,
    int8_t channel_axis,
    void *work_space,
    aitensor_t *output) {
    // Dimensions for h and w are fixed for input, kernel and result
    // The dynamic parts are the channel and filter numbers
    int16_t input_dims[4];
    int16_t weights_dims[4];
    int16_t output_dims[4];

    uint8_t channel_uaxis = channel_axis < 0 ? input->dim + channel_axis : channel_axis;
    // Negative axis = indexing from the end
    uint8_t h_ax, w_ax;
    uint16_t n_idx, f_idx, c_idx;
    uint16_t N = input->shape[0], F = weights->shape[0], C = weights->shape[channel_uaxis];
    int16_t fwd_padding[2][2];

    float *bias_ptr;

    if (channel_uaxis == 1) {
        // Channels first
        h_ax = 2;
        w_ax = 3;
    } else if (channel_uaxis == 3) {
        // Channels last
        h_ax = 1;
        w_ax = 2;
    } else {
        // ERROR
        return;
    }
    input_dims[h_ax] = -1;
    input_dims[w_ax] = -2;
    output_dims[h_ax] = -1;
    output_dims[w_ax] = -2;
    weights_dims[h_ax] = -1;
    weights_dims[w_ax] = -2;

    if (padding[0] == AIFES_PADDING_SAME) {
        // Output shape should equal input shape
        // P = ceil(0.5 * ((N-1) * S - N + D * (K-1) + 1))
        fwd_padding[0][0] = (uint16_t) ((((uint32_t) 1 << 15) * (uint32_t) ((input->shape[h_ax] - 1) * stride[0]
                                                                            - input->shape[h_ax] + dilation[0] * (
                                                                                weights->shape[h_ax] - 1) + 1)
                                         + ((uint32_t) 1 << 16) - 1) >> 16);
        fwd_padding[0][1] = fwd_padding[0][0];
    } else {
        fwd_padding[0][0] = padding[0];
        fwd_padding[0][1] = padding[0];
    }
    if (padding[1] == AIFES_PADDING_SAME) {
        // Output shape should equal input shape
        // P = ceil(0.5 * ((N-1) * S - N + D * (K-1) + 1))
        fwd_padding[1][0] = (uint16_t) ((((uint32_t) 1 << 15) * (uint32_t) ((input->shape[w_ax] - 1) * stride[1]
                                                                            - input->shape[w_ax] + dilation[1] * (
                                                                                weights->shape[w_ax] - 1) + 1)
                                         + ((uint32_t) 1 << 16) - 1) >> 16);
        fwd_padding[1][1] = fwd_padding[1][0];
    } else {
        fwd_padding[1][0] = padding[1];
        fwd_padding[1][1] = padding[1];
    }

    // Init result with zeros
    aimath_f32_default_init_zeros(output);

    // Iterate over all samples
    for (n_idx = 0; n_idx < N; n_idx++) {
        input_dims[0] = n_idx;
        output_dims[0] = n_idx;

        // for all f: y_f = sum_c{x_c * k_fc}
        for (f_idx = 0; f_idx < F; f_idx++) {
            output_dims[channel_uaxis] = f_idx;
            weights_dims[0] = f_idx;
            for (c_idx = 0; c_idx < C; c_idx++) {
                input_dims[channel_uaxis] = c_idx;
                weights_dims[channel_uaxis] = c_idx;
                if (c_idx + 1 == C) {
                    // Add the bias only when the last channel is reached
                    bias_ptr = (float *) (bias->data) + f_idx;
                } else {
                    bias_ptr = 0;
                }

                aimath_f32_default_conv2d_add_quantized(qType,
                                                        input,
                                                        stride,
                                                        dilation,
                                                        fwd_padding,
                                                        weights,
                                                        bias_ptr,
                                                        FALSE,
                                                        input_dims,
                                                        output_dims,
                                                        weights_dims,
                                                        output);
            }
        }
    }
}

static void ailayer_conv2d_forward_Q31(ailayer_t *self) {
    aitensor_t *x_in = &(self->input_layer->result);
    aitensor_t *x_out = &(self->result);
    ailayer_conv2d_t *layer = (ailayer_conv2d_t *) (self->layer_configuration);
    aitensor_t *weights = &layer->weights;
    aitensor_t *bias = &layer->bias;

    aimath_f32_default_conv2d_fwd_quantized(Q31,
                                            x_in,
                                            layer->stride,
                                            layer->dilation,
                                            layer->padding,
                                            weights,
                                            bias,
                                            layer->channel_axis,
                                            0,
                                            x_out);

    return;
}

static void ailayer_conv2d_forward_Q7(ailayer_t *self) {
    aitensor_t *x_in = &(self->input_layer->result);
    aitensor_t *x_out = &(self->result);
    ailayer_conv2d_t *layer = (ailayer_conv2d_t *) (self->layer_configuration);
    aitensor_t *weights = &layer->weights;
    aitensor_t *bias = &layer->bias;

    aimath_f32_default_conv2d_fwd_quantized(Q7,
                                            x_in,
                                            layer->stride,
                                            layer->dilation,
                                            layer->padding,
                                            weights,
                                            bias,
                                            layer->channel_axis,
                                            0,
                                            x_out);

    return;
}

static void ailayer_conv2d_forward_Q1(ailayer_t *self) {
    aitensor_t *x_in = &(self->input_layer->result);
    aitensor_t *x_out = &(self->result);
    ailayer_conv2d_t *layer = (ailayer_conv2d_t *) (self->layer_configuration);
    aitensor_t *weights = &layer->weights;
    aitensor_t *bias = &layer->bias;

    aimath_f32_default_conv2d_fwd_quantized(Q1,
                                            x_in,
                                            layer->stride,
                                            layer->dilation,
                                            layer->padding,
                                            weights,
                                            bias,
                                            layer->channel_axis,
                                            0,
                                            x_out);

    return;
}

/*-------------------------------------------------------------*/






#endif //AIFESCUSTOM_INTERNAL_H
