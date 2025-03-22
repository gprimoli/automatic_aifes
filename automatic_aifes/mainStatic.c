#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "aifes.h"
#include "lib/GenericFuncs.h"

// ---- SETUP ---- //
#define INPUT 2
#define OUTPUT 1
#define BATCH_SIZE 4
#define EPOCH 1000
#define LOG_LOSS ( (int) EPOCH / 200)


#define LAYER_COMPOSITION { INPUT, 3, OUTPUT }
#define HIDDEN_LAYER 1
#define TOT_LAYER (HIDDEN_LAYER + 2)

// ---- SETUP ---- //

void printLoss(float loss)
{
    printf("loss: %f\n", loss);
}

int main(int argc, char *argv[]) {
    srand(time(NULL));
    // if (argc != 2) handle_error("Usage: %s <input_file> <output_file>", argv[0]);


    uint32_t nnLayerComposition[] = LAYER_COMPOSITION;
    AIFES_E_activations activationsFunc[] = {
        AIfES_E_relu,
        AIfES_E_sigmoid,
    };

    if (ARRAY_LEN(nnLayerComposition) != TOT_LAYER || ARRAY_LEN(activationsFunc) != TOT_LAYER - 1)
        handle_error("nnStructure (%d), activationsFunc (%d),  ", ARRAY_LEN(nnLayerComposition), ARRAY_LEN(activationsFunc));

    uint32_t weights_number = AIFES_E_flat_weights_number_fnn_f32(nnLayerComposition,TOT_LAYER);
    float weights[weights_number];

    /* ---- Wrapper ---- */
    AIFES_E_model_parameter_fnn_f32 nnStructure;
    nnStructure.layer_count = TOT_LAYER;
    nnStructure.fnn_structure = nnLayerComposition;
    nnStructure.fnn_activations = activationsFunc;
    nnStructure.flat_weights = weights;

    AIFES_E_init_weights_parameter_fnn_f32 INIT_WEIGHTS;
    INIT_WEIGHTS.init_weights_method = AIfES_E_init_glorot_uniform; // OR AIfES_E_init_no_init OR AIfES_E_init_uniform

    AIFES_E_training_parameter_fnn_f32 trainStructure;
    trainStructure.optimizer = AIfES_E_adam; // OR AIfES_E_sgd
    trainStructure.loss = AIfES_E_mse; // OR AIfES_E_crossentropy
    trainStructure.learn_rate = 0.05f;
    trainStructure.batch_size = BATCH_SIZE;
    trainStructure.epochs = EPOCH;
    trainStructure.epochs_loss_print_interval = LOG_LOSS;
    trainStructure.loss_print_function = printLoss;
    trainStructure.early_stopping = AIfES_E_early_stopping_off; // OR AIfES_E_early_stopping_on
    // trainStructure.early_stopping_target_loss = 0.004f;


    /* ---- Wrapper ---- */

    float input_data[BATCH_SIZE][INPUT] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };
    uint16_t input_shape[] = {BATCH_SIZE, INPUT};
    aitensor_t input_tensor = AITENSOR_2D_F32(input_shape, input_data);

    float target_data[BATCH_SIZE][OUTPUT] = {
        {0.0},
        {1.0},
        {1.0},
        {0.0},
    };
    uint16_t target_shape[] = {BATCH_SIZE, OUTPUT};
    aitensor_t target_tensor = AITENSOR_2D_F32(target_shape, target_data);

    float output_data[BATCH_SIZE];
    uint16_t output_shape[] = {BATCH_SIZE, OUTPUT};
    aitensor_t output_tensor = AITENSOR_2D_F32(output_shape, output_data);

    int8_t error = 0;
    error = AIFES_E_training_fnn_f32(&input_tensor,&target_tensor,&nnStructure,&trainStructure,&INIT_WEIGHTS,&output_tensor);

    printf("ok %d ", error);

    return 1;
}
