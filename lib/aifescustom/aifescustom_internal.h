#ifndef AIFESCUSTOM_INTERNAL_H
#define AIFESCUSTOM_INTERNAL_H

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

static void run_inference(aiconfiguration_t *conf, aimodel_t *model, FILE *f_x, FILE *f_y, uint32_t sample_number, float *acc, float *loss) {
    uint32_t input_elements = (conf->input_shape[2] == 0 && conf->input_shape[3] == 0)
                                  ? conf->batch_size * conf->input_shape[1]
                                  : conf->batch_size * conf->input_shape[1] * conf->input_shape[2] * conf->input_shape
                                    [3];
    uint32_t output_elements = conf->batch_size * conf->layers[conf->num_layer - 1].params.dense.neurons;

    uint32_t batch_test = sample_number / conf->batch_size;

    *loss = *acc = 0.0f;
    for (int batch = 0; batch < batch_test; batch++) {
        if (!csv_read(conf->x->data, input_elements, f_x) || !csv_read(conf->y->data, output_elements, f_y)) {
            SAFE_EXIT_FAILURE("Errore lettura batch da CSV");
        }

        if (!aialgo_calc_loss_acc_model_f32(conf, model, loss, acc)) {
            SAFE_EXIT_FAILURE("Acc loss error");
        }
    }

    LOG_INFO("Inference Loss: %.5f\tAccuracy: %.5f", *loss, *acc);
    RESET_ALL_FILES(f_x, f_y);
}

#endif //AIFESCUSTOM_INTERNAL_H
