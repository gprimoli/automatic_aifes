#include <time.h>

#include "Log.h"
#include "aifes.h"
#include "aidataset.h"
#include "MemManager.h"
#include "aiconfiguration.h"

bool build_model(aiconfiguration_t *ctx, aimodel_t *model);

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

bool aialgo_calc_loss_acc_model_f32(aimodel_t *model, aitensor_t *input_tensor, aitensor_t *target_tensor,
                                    float *loss_result, float *accuracy_result) {
    uint32_t i;
    float loss;
    uint16_t batch_size = input_tensor->shape[0];
    uint16_t batch_slice_size = model->input_layer->result.shape[0]; // Size of a batch that is processed by one forward pass

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
    for(i = input_tensor->dim - 1; i > 0; i--)
    {
        input_multiplier *= input_tensor->shape[i];
        input_batch_shape[i] = input_tensor->shape[i];
    }
    input_multiplier *= input_tensor->dtype->size;
    input_batch_shape[0] = batch_slice_size;
    uint32_t target_multiplier = 1;
    for(i = target_tensor->dim - 1; i > 0; i--)
    {
        target_multiplier *= target_tensor->shape[i];
        target_batch_shape[i] = target_tensor->shape[i];
    }
    target_multiplier *= target_tensor->dtype->size;
    target_batch_shape[0] = batch_slice_size;

    aialgo_set_training_mode_model(model, FALSE);
    aialgo_set_batch_mode_model(model, FALSE);

    int correct = 0;
    *loss_result = 0;
    for (i = 0; i < batch_size / batch_slice_size; i++) {
        input_batch.data = input_tensor->data + i * batch_slice_size * input_multiplier;
        target_batch.data = target_tensor->data + i * batch_slice_size * target_multiplier;

        aitensor_t *result_tensor = aialgo_forward_model(model, &input_batch);
        model->loss->calc_loss(model->loss, &target_batch, &loss);
        *loss_result += loss;

        int pred_label = argmax(result_tensor);
        int true_label = argmax(&target_batch);

        if (pred_label == true_label) {
            correct++;
        }
    }

    *accuracy_result = (float) correct / (float) i;
    return true;
}


int main(int argc, char *argv[]) {
    srand(time(NULL));

    aiconfiguration_t ctx = {0};
    aidataset_t d = {0};
    aimodel_t model = {0};

    if (argc < 2 || !load_config(argv[1], &ctx) || !load_dataset(ctx, &d)) {
        SAFE_EXIT_FAILURE;
    }


    uint16_t input_shape[] = {1, ctx.input_shape[0], ctx.input_shape[1], ctx.input_shape[2]};
    ailayer_input_f32_t input_layer = AILAYER_INPUT_F32_A(4, input_shape);
    model.input_layer = ailayer_input_f32_default(&input_layer);

    if (!build_model(&ctx, &model)) {
        SAFE_EXIT_FAILURE;
    }

    uint16_t input_shape_training[] = {ctx.sample_train, ctx.input_shape[0], ctx.input_shape[1], ctx.input_shape[2]};
    uint16_t output_shape_training[] = {ctx.sample_train, ctx.layers[ctx.num_layer - 1].params[NEURONS][0]};

    uint16_t input_shape_test[] = {ctx.sample_test, ctx.input_shape[0], ctx.input_shape[1], ctx.input_shape[2]};
    uint16_t output_shape_test[] = {ctx.sample_test, ctx.layers[ctx.num_layer - 1].params[NEURONS][0]};

    aitensor_t x_train = AITENSOR_4D_F32(input_shape_training, d.x_train);
    aitensor_t y_train = AITENSOR_2D_F32(output_shape_training, d.y_train);
    aitensor_t x_test = AITENSOR_4D_F32(input_shape_test, d.x_test);
    aitensor_t y_test = AITENSOR_2D_F32(output_shape_test, d.y_test);

    aialgo_compile_model(&model);
    uint32_t parameter_memory_size = aialgo_sizeof_parameter_memory(&model);

    void *parameter_memory = mem_calloc(parameter_memory_size, sizeof(void));

    aialgo_distribute_parameter_memory(&model, parameter_memory, parameter_memory_size);

    ailoss_crossentropy_f32_t crossentropy_loss;
    model.loss = ailoss_crossentropy_f32_default(&crossentropy_loss, model.output_layer);

    aiopti_adam_f32_t adam_opti = {
        .learning_rate = 0.01f,
        .beta1 = 0.9f,
        .beta2 = 0.999f,
        .eps = 1e-7f
    };

    aialgo_initialize_parameters_model(&model);

    aiopti_t *optimizer;
    optimizer = aiopti_adam_f32_default(&adam_opti);

    uint32_t memory_size = aialgo_sizeof_training_memory(&model, optimizer);

    void *memory_ptr = mem_calloc(memory_size, sizeof(void));

    aialgo_schedule_training_memory(&model, optimizer, memory_ptr, memory_size);

    aialgo_init_model_for_training(&model, optimizer);

    aiprint("\n-------------- Model structure ---------------\n");
    aialgo_print_model_structure(&model);
    aiprint("----------------------------------------------\n\n");

    // float loss, acc;
    // for (int i = 0; i < ctx.epochs; i++) {
    //     LOG_INFO("Epoc: %d", i);
    //     aialgo_train_model(&model, &x_train, &y_train, optimizer, ctx.batch_size);
    //     aialgo_calc_loss_acc_model_f32(&model, &x_test, &y_test, &loss, &acc);
    //     LOG_INFO("Test loss: %f\nTest acc:%f", loss, acc);
    //
    //     // aialgo_calc_loss_model_f32(&model, &x_test, &y_test, &loss); //original
    //     // if (loss < 5) break;
    // }

    LOG_INFO("Freed %llu byte", mem_total());
    SAFE_EXIT_SUCCESS;
}

bool build_model(aiconfiguration_t *ctx, aimodel_t *model) {
    ailayer_t *layers = model->input_layer;

    for (uint32_t i = 0; i < ctx->num_layer; i++) {
        aiconfigurationlayer_t current = ctx->layers[i];
        switch (current.type) {
            case DENSE: {
                ailayer_dense_f32_t *l = mem_calloc(1, sizeof(ailayer_dense_f32_t));

                l->neurons = current.params[NEURONS][0];
                layers = ailayer_dense_f32_default(l, layers);
            }
            break;
            case CONV2D: {
                ailayer_conv2d_f32_t *l = mem_calloc(1, sizeof(ailayer_conv2d_f32_t));

                l->channel_axis = current.params[CHANNEL_AXIS][0];
                l->filter_count = current.params[FILTERS][0];

                l->kernel_size[0] = current.params[PARAM_KERNEL_SIZE][0];
                l->kernel_size[1] = current.params[PARAM_KERNEL_SIZE][1];

                l->stride[0] = current.params[PARAM_KERNEL_STRIDE][0];
                l->stride[1] = current.params[PARAM_KERNEL_STRIDE][1];

                l->dilation[0] = current.params[PARAM_KERNEL_DILATION][0];
                l->dilation[1] = current.params[PARAM_KERNEL_DILATION][1];

                l->padding[0] = current.params[PARAM_KERNEL_PADDING][0];
                l->padding[1] = current.params[PARAM_KERNEL_PADDING][1];

                layers = ailayer_conv2d_f32_default(l, layers);
            }
            break;
            case MAXPOOL2D: {
                ailayer_maxpool2d_f32_t *l = mem_calloc(1, sizeof(ailayer_maxpool2d_f32_t));

                l->channel_axis = current.params[CHANNEL_AXIS][0];
                l->pool_size[0] = current.params[PARAM_POOL_SIZE][0];
                l->pool_size[1] = current.params[PARAM_POOL_SIZE][1];
                l->stride[0] = current.params[PARAM_KERNEL_STRIDE][0];
                l->stride[1] = current.params[PARAM_KERNEL_STRIDE][1];
                l->padding[0] = current.params[PARAM_KERNEL_PADDING][0];
                l->padding[1] = current.params[PARAM_KERNEL_PADDING][1];
                layers = ailayer_maxpool2d_f32_default(l, layers);
            }
            break;
            case FLATTEN: {
                ailayer_flatten_f32_t *l = mem_calloc(1, sizeof(ailayer_flatten_f32_t));
                layers = ailayer_flatten_f32_default(l, layers);
            }
            break;
            default: {
                LOG_ERROR("Non implemented layer! layer: %s", layer_type_to_string(current.type));
                return false;
            }
        }

        if (current.type == MAXPOOL2D || current.type == FLATTEN) {
            continue;
        }

        switch (current.activation) {
            case RELU: {
                ailayer_relu_f32_t *l = mem_calloc(1, sizeof(ailayer_relu_f32_t));
                layers = ailayer_relu_f32_default(l, layers);
            }
            break;
            case SIGMOID: {
                ailayer_sigmoid_f32_t *l = mem_calloc(1, sizeof(ailayer_sigmoid_f32_t));
                layers = ailayer_sigmoid_f32_default(l, layers);
            }
            break;
            case SOFTMAX: {
                ailayer_softmax_f32_t *l = mem_calloc(1, sizeof(ailayer_softmax_f32_t));
                layers = ailayer_softmax_f32_default(l, layers);
            }
            break;
            default: {
                LOG_ERROR("Non implemented layer! fun: %s", activation_to_string(current.activation));
                return false;
            }
        }
    }

    model->output_layer = layers;
    return true;
}


/*
ailoss_mse_t mse_loss; //Loss: mean squared error
    ailoss_crossentropy_t crossentropy_loss; //Loss: crossentropy

    switch(AIFES_E_fnn_training->loss){
        case AIfES_E_mse:
            model.loss = ailoss_mse_f32_default(&mse_loss, model.output_layer);
            break;
        case AIfES_E_crossentropy:
            model.loss = ailoss_crossentropy_f32_default(&crossentropy_loss, model.output_layer);
            break;
        default :
            //printf("ERROR! Unknown loss function\n" );
            return(-10);
    }


aiopti_t *optimizer; // Object for the optimizer
    aiopti_adam_f32_t adam_opti;
    aiopti_sgd_f32_t sgd_opti;

    switch(AIFES_E_fnn_training->optimizer){
        case AIfES_E_adam:
            adam_opti.learning_rate = AIFES_E_fnn_training->learn_rate;
            adam_opti.beta1 = 0.9f;
            adam_opti.beta2 = 0.999f;
            adam_opti.eps = 1e-7;
            optimizer = aiopti_adam_f32_default(&adam_opti);
            break;
        case AIfES_E_sgd:
            sgd_opti.learning_rate = AIFES_E_fnn_training->learn_rate;
            sgd_opti.momentum = AIFES_E_fnn_training->sgd_momentum;
            optimizer = aiopti_sgd_f32_default(&sgd_opti);
            break;
        default :
            //printf("ERROR! Unknown optimizer\n" );
            return(-12);
    }

aialgo_train_model(&model, input_tensor, target_tensor, optimizer, AIFES_E_fnn_training->batch_size);

case AIfES_E_crossentropy:
                // loss = loss / NUMBER_DATASETS
                loss = loss / input_tensor->shape[0];
                (*AIFES_E_fnn_training->loss_print_function)(loss);

                if(AIFES_E_fnn_training->early_stopping == AIfES_E_early_stopping_on)
                {
                    if(loss <= AIFES_E_fnn_training->early_stopping_target_loss)
                    {
                            free(memory_ptr);
                            return(0);
                    }
                }
                break;
 */
