#ifndef AIFESCUSTOM_QUANTIZZATION_H
#define AIFESCUSTOM_QUANTIZZATION_H

#include "aifes.h"
#include "aifescustom.h"

#define CLAMP(x, low, high) ((x) < (low) ? (low) : ((x) > (high) ? (high) : (x)))

static bool aimath_quantize_tensor(aitensor_t *t, Quantization qType) {
    if (!t || !(t->data)) return false;
    float *data = (float *) t->data;

    switch (qType) {
        case Q31: {
            t->tensor_params = (aimath_q31_params_t *) mem_calloc(1, sizeof(aimath_q31_params_t));
            aimath_q31_params_t *q = (aimath_q31_params_t *) t->tensor_params;
            if (!calc_scale_and_zero_point(Q31, t, q)) return false;
            break;
        }
        case Q7: {
            t->tensor_params = (aimath_q7_params_t *) mem_calloc(1, sizeof(aimath_q7_params_t));
            aimath_q7_params_t *q = (aimath_q7_params_t *) t->tensor_params;
            if (!calc_scale_and_zero_point(Q31, t, q)) return false;
            break;
        }
        default:
            SAFE_EXIT_FAILURE("Errore quantizzazione: tipo non supportato");
    }


    for (int i = 0; i < aimath_tensor_elements(t); i++) {
        switch (qType) {
            case Q31: {
                aimath_q31_params_t *q = (aimath_q31_params_t *) t->tensor_params;
                data[i] = FLOAT_TO_Q31(data[i], q->shift, q->zero_point);
                break;
            }
            case Q7: {
                aimath_q7_params_t *q = (aimath_q7_params_t *) t->tensor_params;
                data[i] = FLOAT_TO_Q7(data[i], q->shift, q->zero_point);
                break;
            }
            default:
                SAFE_EXIT_FAILURE("Errore quantizzazione: tipo non supportato");
        }
    }
    return true;
}

static float fake_quantize_value_shift(Quantization qType, float value, void *qParam) {
    switch (qType) {
        case Q31: {
            aimath_q31_params_t *q = (aimath_q31_params_t *) qParam;
            return Q31_TO_FLOAT(FLOAT_TO_Q31(value, q->shift, q->zero_point), q->shift, q->zero_point);
            break;
        }
        case Q7: {
            aimath_q7_params_t *q = (aimath_q7_params_t *) qParam;
            return Q7_TO_FLOAT(FLOAT_TO_Q7(value, q->shift, q->zero_point), q->shift, q->zero_point);
            break;
        }
        default:
            SAFE_EXIT_FAILURE("Errore quantizzazione: tipo non supportato");
    }
    return value;
}

static void quantize(aimodel_t *model, Quantization qType) {//TODO NON FUNZIONA!!!
    static bool already_quantized = false;
    if (already_quantized) return;

    ailayer_t *layer = model->input_layer;
    for (int i = 1; i < model->layer_count; i++) {
        layer = layer->output_layer;
        aitensor_t *weights = NULL;
        if (strcmp(layer->layer_type->name, "Dense") == 0) {
            ailayer_dense_f32_t *layer_cast = (ailayer_dense_f32_t *) layer;
            if (!aimath_quantize_tensor(&(layer_cast->weights), qType)) {
                SAFE_EXIT_FAILURE("Errore quantizzazione");
            }
            if (!aimath_quantize_tensor(&(layer_cast->bias), qType)) {
                SAFE_EXIT_FAILURE("Errore quantizzazione");
            }
        } else if (strcmp(layer->layer_type->name, "Conv2D") == 0) {
            ailayer_conv2d_f32_t *layer_cast = (ailayer_conv2d_f32_t *) layer;
            if (!aimath_quantize_tensor(&(layer_cast->weights), qType)) {
                SAFE_EXIT_FAILURE("Errore quantizzazione");
            }
            if (!aimath_quantize_tensor(&(layer_cast->bias), qType)) {
                SAFE_EXIT_FAILURE("Errore quantizzazione");
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

    void *qParam_a = NULL, *qParam_b = NULL;
    aimath_q31_params_t q31_a, q31_b;
    aimath_q7_params_t q7_a, q7_b;

    switch (qType) {
        case Q31: {
            qParam_a = &q31_a;
            qParam_b = &q31_b;
            break;
        }
        case Q7: {
            qParam_a = &q7_a;
            qParam_b = &q7_b;
            break;
        }
        default:
            SAFE_EXIT_FAILURE("Tipo di quantizzazione non supportato");
    }

    if (!calc_scale_and_zero_point(qType, a, &qParam_a) || !calc_scale_and_zero_point(qType, b, &qParam_b)) {
        SAFE_EXIT_FAILURE("Errore quantizzazione");
    }

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


    void *qParam_a = NULL, *qParam_b = NULL;
    aimath_q31_params_t q31_a, q31_b;
    aimath_q7_params_t q7_a, q7_b;

    switch (qType) {
        case Q31: {
            qParam_a = &q31_a;
            qParam_b = &q31_b;
            break;
        }
        case Q7: {
            qParam_a = &q7_a;
            qParam_b = &q7_b;
            break;
        }
        default:
            SAFE_EXIT_FAILURE("Tipo di quantizzazione non supportato");
    }

    if (!calc_scale_and_zero_point(qType, input, &qParam_a) || !calc_scale_and_zero_point(qType, kernel, &qParam_b)) {
        SAFE_EXIT_FAILURE("Errore quantizzazione");
    }


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


/*-------------------------------------------------------------*/

#endif //AIFESCUSTOM_INTERNAL_H
