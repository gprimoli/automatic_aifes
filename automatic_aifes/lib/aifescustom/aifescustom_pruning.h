#ifndef AIFESCUSTOM_PRUNING_H
#define AIFESCUSTOM_PRUNING_H

#include <string.h>

#include "log.h"
#include "aifes.h"
#include "memmanager.h"

static void prune_global(aimodel_t *model, float prune_percentage) {
    LOG_INFO("Inizio Pruning");
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
    LOG_INFO("Fine Pruning");
}

#endif //AIFESCUSTOM_INTERNAL_H
