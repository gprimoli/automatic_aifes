#include <time.h>

#include "Log.h"
#include "aifes.h"
#include "aidataset.h"
#include "MemManager.h"
#include "aiconfiguration.h"

bool build_model(aiconfiguration_t *ctx, aimodel_t *model);

int main(int argc, char *argv[]) {
    srand(time(NULL));

    aiconfiguration_t ctx = {0};
    aidataset_t d = {0};
    aimodel_t model = {0};

    if (argc < 2 || !load_config(argv[1], &ctx) || !load_dataset(ctx, &d)) {
        SAFE_EXIT_FAILURE;
    }


    uint16_t input_shape[] = {ctx.batch_size, ctx.input_shape[0], ctx.input_shape[1], ctx.input_shape[2]};
    uint16_t output_shape[] = {ctx.batch_size, ctx.layers[ctx.num_layer - 1].params[NEURONS][0]};

    ailayer_input_f32_t input_layer = AILAYER_INPUT_F32_A(4, input_shape);
    model.input_layer = ailayer_input_f32_default(&input_layer);

    if (!build_model(&ctx, &model)) {
        SAFE_EXIT_FAILURE;
    }

    aitensor_t x_train = AITENSOR_4D_F32(input_shape, d.x_train);
    aitensor_t y_train = AITENSOR_2D_F32(output_shape, d.y_train);
    aitensor_t x_test = AITENSOR_4D_F32(input_shape, d.x_test);
    aitensor_t y_test = AITENSOR_2D_F32(output_shape, d.y_test);

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


    // for (int i = 0; i < ctx.epochs; i++) {
    //     float loss;
    //
    //     aialgo_train_model(&model, &x_train, &y_train, optimizer, ctx.batch_size);
    //
    //     aialgo_calc_loss_model_f32(&model, &x_test, &y_test, &loss);
    //     LOG_INFO("Test loss: %f", loss);
    // }


    LOG_INFO("%llu byte", mem_total());
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
