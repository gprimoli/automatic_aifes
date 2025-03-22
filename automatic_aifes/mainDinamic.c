#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include "aifes.h"
#include "lib/GenericFuncs.h"


void printLoss(float loss);

int parse_layers(char *arg, uint32_t **LAYER_COMPOSITION, uint32_t *TOT_LAYER);

int parse_activation_func(char *arg, AIFES_E_activations **ACTIVATION_FUNCTION, uint32_t TOT_LAYER);

int runTraining(uint32_t *LAYER_COMPOSITION, AIFES_E_activations *ACTIVATION_FUNCTION, uint32_t TOT_LAYER,
                uint32_t BATCH_SIZE, uint32_t EPOCH, uint32_t LOG_LOSS, float *weights, float *input, float *target,
                float *output, char *weights_filename);

int initArrA(float **arr, uint32_t colSize, uint32_t rowSize, char *initFile);

int initArrB(float **arr, uint32_t size, char *initFile);

int main(int argc, char *argv[]) {
    srand(time(NULL));

    uint32_t BATCH_SIZE = 0;
    uint32_t EPOCH = 0;
    uint32_t LOG_LOSS = 0;
    uint32_t TOT_LAYER = 0;

    uint32_t *LAYER_COMPOSITION = NULL;
    AIFES_E_activations *ACTIVATION_FUNCTION = NULL;
    float *weights = NULL, *input_data = NULL, *targhet_data = NULL, *output_data = NULL;

    char *weights_filename = NULL, *input_filename = NULL, *targhet_filename = NULL;
    char *layerDefString = NULL, *activationFunctionString = NULL;

    int opt;
    while ((opt = getopt(argc, argv, "b:e:l:a:w:i:t:")) != -1) {
        switch (opt) {
            case 'b':
                BATCH_SIZE = atoi(optarg);
                break;
            case 'e':
                EPOCH = atoi(optarg);
                LOG_LOSS = EPOCH / 5;
                break;
            case 'l':
                layerDefString = optarg;
                break;
            case 'a':
                activationFunctionString = optarg;
                break;
            case 'w':
                weights_filename = optarg;
                break;
            case 'i':
                input_filename = optarg;
                break;
            case 't':
                targhet_filename = optarg;
                break;
            default:
                fprintf(
                    stderr,
                    "Usage: %s -b <batch_size> -e <epoch> -l <layer_composition> -a <activation_function_composition> "
                    "-i <input_file> -t <targhet_file> [-w <weights_file>]\n",
                    argv[0]);
                return EXIT_FAILURE;
        }
    }

    if ((parse_layers(layerDefString, &LAYER_COMPOSITION, &TOT_LAYER) != EXIT_SUCCESS) ||
        (parse_activation_func(activationFunctionString, &ACTIVATION_FUNCTION, TOT_LAYER) != EXIT_SUCCESS) ||
        (initArrB(&weights, AIFES_E_flat_weights_number_fnn_f32(LAYER_COMPOSITION, TOT_LAYER), weights_filename) !=
         EXIT_SUCCESS) ||
        (initArrA(&input_data, LAYER_COMPOSITION[0], BATCH_SIZE, input_filename) != EXIT_SUCCESS) ||
        (initArrA(&targhet_data, LAYER_COMPOSITION[TOT_LAYER - 1], BATCH_SIZE, targhet_filename) != EXIT_SUCCESS) ||
        (initArrA(&output_data, LAYER_COMPOSITION[TOT_LAYER - 1], BATCH_SIZE, NULL) != EXIT_SUCCESS)
    ) {
        safeFree((void **) &LAYER_COMPOSITION);
        safeFree((void **) &ACTIVATION_FUNCTION);
        safeFree((void **) &weights);
        safeFree((void **) &input_data);
        safeFree((void **) &targhet_data);
        safeFree((void **) &output_data);
        return EXIT_FAILURE;
    }

    printf("Neural Network Configuration:\n");
    printf("- Input Neurons: %d\n", LAYER_COMPOSITION[0]);
    printf("- Output Neurons: %d\n", LAYER_COMPOSITION[TOT_LAYER - 1]);
    printf("- Total Layers: %d\n", TOT_LAYER);
    printf("- Batch Size: %d\n", BATCH_SIZE);
    printf("- Epochs: %d\n", EPOCH);
    printf("- Log every: %d epochs\n\n", LOG_LOSS);

    runTraining(LAYER_COMPOSITION, ACTIVATION_FUNCTION, TOT_LAYER, BATCH_SIZE, EPOCH, LOG_LOSS, weights, input_data,
                targhet_data, output_data, weights_filename);

    safeFree((void **) &LAYER_COMPOSITION);
    safeFree((void **) &ACTIVATION_FUNCTION);
    safeFree((void **) &weights);
    safeFree((void **) &input_data);
    safeFree((void **) &targhet_data);
    safeFree((void **) &output_data);
    return EXIT_SUCCESS;
}


void printLoss(float loss) {
    printf("Loss: %f\n", loss);
}

int parse_layers(char *arg, uint32_t **LAYER_COMPOSITION, uint32_t *TOT_LAYER) {
    *TOT_LAYER = 0;
    char *backup = strdup(arg);
    if (backup == NULL) {
        safeFree((void **) &backup);
        safeFree((void **) &LAYER_COMPOSITION);
        printf("Memory allocation failed for backup\n");
        return EXIT_FAILURE;
    }

    for (char *token = strtok(backup, ","); token != NULL; token = strtok(NULL, ",")) {
        (*TOT_LAYER)++;
    }

    *LAYER_COMPOSITION = (uint32_t *) malloc((*TOT_LAYER) * sizeof(uint32_t));
    if (*LAYER_COMPOSITION == NULL) {
        safeFree((void **) &backup);
        safeFree((void **) &LAYER_COMPOSITION);
        printf("Memory allocation failed for LAYER_COMPOSITION\n");
        return EXIT_FAILURE;
    }

    strcpy(backup, arg);
    char *token = strtok(backup, ",");
    for (int i = 0; i < *TOT_LAYER; i++) {
        (*LAYER_COMPOSITION)[i] = atoi(token);
        token = strtok(NULL, ",");
    }

    safeFree((void **) &backup);
    return EXIT_SUCCESS;
}

int parse_activation_func(char *arg, AIFES_E_activations **ACTIVATION_FUNCTION, uint32_t TOT_LAYER) {
    *ACTIVATION_FUNCTION = (AIFES_E_activations *) malloc((TOT_LAYER - 1) * sizeof(AIFES_E_activations));
    if (*ACTIVATION_FUNCTION == NULL) {
        safeFree((void **) &ACTIVATION_FUNCTION);
        printf("Memory allocation failed for ACTIVATION_FUNCTION\n");
        return EXIT_FAILURE;
    }

    char *token = strtok(arg, ","); // modifies the original string by replacing delimiters with \0 !!!
    for (int i = 0; i < TOT_LAYER - 1; i++) {
        if (token == NULL)
            return EXIT_FAILURE;
        if (strcmp(token, "relu") == 0) {
            (*ACTIVATION_FUNCTION)[i] = AIfES_E_relu;
        } else if (strcmp(token, "sigmoid") == 0) {
            (*ACTIVATION_FUNCTION)[i] = AIfES_E_sigmoid;
        } else if (strcmp(token, "softmax") == 0) {
            (*ACTIVATION_FUNCTION)[i] = AIfES_E_softmax;
        } else if (strcmp(token, "leaky_relu") == 0) {
            (*ACTIVATION_FUNCTION)[i] = AIfES_E_leaky_relu;
        } else if (strcmp(token, "elu") == 0) {
            (*ACTIVATION_FUNCTION)[i] = AIfES_E_elu;
        } else if (strcmp(token, "tanh") == 0) {
            (*ACTIVATION_FUNCTION)[i] = AIfES_E_tanh;
        } else if (strcmp(token, "softsign") == 0) {
            (*ACTIVATION_FUNCTION)[i] = AIfES_E_softsign;
        } else if (strcmp(token, "linear") == 0) {
            (*ACTIVATION_FUNCTION)[i] = AIfES_E_linear;
        }
        token = strtok(NULL, ",");
    }
    return EXIT_SUCCESS;
}

int initArrB(float **arr, uint32_t size, char *initFile) {
    *arr = (float *) malloc(size * sizeof(float));
    if (*arr == NULL) {
        fprintf(stderr, "Error allocating memory for an array (%s)\n", initFile);
        return EXIT_FAILURE;
    }
    if (initFile != NULL && readCSVB(initFile, *arr, size) != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}


int initArrA(float **arr, uint32_t colSize, uint32_t rowSize, char *initFile) {
    *arr = (float *) malloc(colSize * rowSize * sizeof(float));
    if (*arr == NULL) {
        fprintf(stderr, "Error allocating memory for an array (%s)\n", initFile);
        return EXIT_FAILURE;
    }
    if (initFile != NULL && readCSVA(initFile, *arr, colSize, rowSize) != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int runTraining(uint32_t *LAYER_COMPOSITION, AIFES_E_activations *ACTIVATION_FUNCTION, uint32_t TOT_LAYER,
                uint32_t BATCH_SIZE, uint32_t EPOCH, uint32_t LOG_LOSS, float *weights, float *input, float *target,
                float *output, char *weights_filename) {
    AIFES_E_model_parameter_fnn_f32 NN_STRUCTURE = {
        .layer_count = TOT_LAYER,
        .fnn_structure = LAYER_COMPOSITION,
        .fnn_activations = ACTIVATION_FUNCTION,
        .flat_weights = weights,
    };
    AIFES_E_init_weights_parameter_fnn_f32 INIT_WEIGHTS = {
        .init_weights_method = weights_filename != NULL ? AIfES_E_init_no_init : AIfES_E_init_glorot_uniform,
    };

    AIFES_E_training_parameter_fnn_f32 TRAIN_STRUCTURE = {
        .optimizer = AIfES_E_adam, // OR AIfES_E_sgd
        .loss = AIfES_E_crossentropy, // OR AIfES_E_mse
        .learn_rate = 0.001f,
        .batch_size = BATCH_SIZE,
        .epochs = EPOCH,
        .epochs_loss_print_interval = LOG_LOSS,
        .loss_print_function = printLoss,
        .early_stopping = AIfES_E_early_stopping_off, // OR AIfES_E_early_stopping_on
        .early_stopping_target_loss = 0.004f,
    };

    uint16_t input_shape[] = {BATCH_SIZE, LAYER_COMPOSITION[0]};
    uint16_t target_shape[] = {BATCH_SIZE, LAYER_COMPOSITION[TOT_LAYER - 1]};

    aitensor_t input_tensor = AITENSOR_2D_F32(input_shape, input);
    aitensor_t target_tensor = AITENSOR_2D_F32(target_shape, target);
    aitensor_t output_tensor = AITENSOR_2D_F32(target_shape, output);

    int8_t error = AIFES_E_training_fnn_f32(&input_tensor, &target_tensor, &NN_STRUCTURE, &TRAIN_STRUCTURE,
                                            &INIT_WEIGHTS, &output_tensor);

    fprintf(stdout, "\n");
    switch (error) {
        case 0: fprintf(stdout, "Training OK\n");
            break;
        case -1: fprintf(stderr, "ERROR! Tensor dtype\n");
            break;
        case -2: fprintf(stderr, "ERROR! Tensor shape: Data Number\n");
            break;
        case -3: fprintf(stderr, "ERROR! Input tensor shape does not correspond to ANN inputs\n");
            break;
        case -4: fprintf(stderr, "ERROR! Output tensor shape does not correspond to ANN outputs\n");
            break;
        case -5: fprintf(stderr, "ERROR! Use the crossentropy as loss for softmax\n");
            break;
        case -6: fprintf(stderr, "ERROR! learn_rate or sgd_momentum negative\n");
            break;
        case -7: fprintf(stderr, "ERROR! Init uniform weights min - max wrongn");
            break;
        case -8: fprintf(stderr, "ERROR! batch_size: min = 1 / max = Number of training data\n");
            break;
        case -9: fprintf(stderr, "ERROR! Unknown activation function\n");
            break;
        case -10: fprintf(stderr, "ERROR! Unknown loss function\n");
            break;
        case -11: fprintf(stderr, "ERROR! Unknown init weights method\n");
            break;
        case -12: fprintf(stderr, "ERROR! Unknown optimizer\n");
            break;
        case -13: fprintf(stderr, "ERROR! Not enough memory\n");
            break;
        default: fprintf(stderr, "Unknown error\n");
    }

    if (!error) {
        saveArray(weights_filename, weights, AIFES_E_flat_weights_number_fnn_f32(LAYER_COMPOSITION, TOT_LAYER));
    }

    return error == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
