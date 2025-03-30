#include <time.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdbool.h>

#include "aifes.h"
#include "Log.h"
#include "GenericFuncs.h"

void print_usage(char *prog_name) {
    LOG_ERROR(
        "Usage: %s [OPTIONS]\n\n"
        "Options:\n"
        "  -b <batch_size>              Number of samples per training batch (e.g., 32, 64)\n"
        "  -e <epoch>                   Number of training epochs (e.g., 100)\n"
        "  -s <dataset_size>            Number of samples in input\n"
        "  -l <layer_composition>       Layer structure, e.g., \"784,128,64,10\"\n"
        "  -a <activation_functions>    Activations per layer, e.g., \"relu-relu-softmax\"\n"
        "  -i <input_file>              Path to input CSV file with features\n"
        "  -t <target_file>             (Optional) Path to CSV file with expected outputs\n"
        "  -w <weights_file>            (Optional) File to load/store weights\n"
        "  -v <log_level>               (Optional) Log level (1,2,3)\n"
        "  -h                           Show this help message and exit\n\n"
        "Example:\n"
        "  %s -b 64 -e 100 -s 200 -l \"784,64,10\" -a \"relu,softmax\" -i input.csv -t target.csv -w weights.bin\n",
        prog_name, prog_name
    );
}

void printLoss(float loss) {
    printf("Loss: %f\n", loss);
}

bool parse_layers(char *arg, uint32_t **LAYER_COMPOSITION, uint32_t *TOT_LAYER) {
    if (!arg || !LAYER_COMPOSITION || !TOT_LAYER) return false;

    *TOT_LAYER = 0;
    char *backup = strdup(arg);
    if (backup == NULL) {
        LOG_ERROR("Memory allocation failed for backup\n");
        return false;
    }

    for (char *token = strtok(backup, ","); token != NULL; token = strtok(NULL, ",")) {
        (*TOT_LAYER)++;
    }
    FREE_ALL_RESOURCES(backup);

    *LAYER_COMPOSITION = (uint32_t *) malloc((*TOT_LAYER) * sizeof(uint32_t));
    if (*LAYER_COMPOSITION == NULL) {
        LOG_ERROR("Memory allocation failed for LAYER_COMPOSITION\n");
        return false;
    }

    char *token = strtok(arg, ",");
    for (int i = 0; i < *TOT_LAYER; i++) {
        (*LAYER_COMPOSITION)[i] = atoi(token);
        token = strtok(NULL, ",");
    }
    return true;
}

bool parse_activations(char *arg, AIFES_E_activations **ACTIVATION_FUNCTION, uint32_t TOT_LAYER) {
    if (!arg || !ACTIVATION_FUNCTION || TOT_LAYER < 2) return false;

    *ACTIVATION_FUNCTION = (AIFES_E_activations *) malloc((TOT_LAYER - 1) * sizeof(AIFES_E_activations));
    if (*ACTIVATION_FUNCTION == NULL) {
        LOG_ERROR("Memory allocation failed for ACTIVATION_FUNCTION\n");
        return false;
    }

    char *token = strtok(arg, ","); // modifies the original string by replacing delimiters with \0 !!!
    for (int i = 0; i < TOT_LAYER - 1; i++) {
        if (token == NULL) {
            LOG_ERROR("Not enough activation functions provided.\n");
            FREE_ALL_RESOURCES(ACTIVATION_FUNCTION);
            return false;
        }
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
        } else {
            LOG_ERROR("Unknown activation function: '%s'\n", token);
            FREE_ALL_RESOURCES(ACTIVATION_FUNCTION);
            return false;
        }
        token = strtok(NULL, ",");
    }
    return true;
}

bool parse_args(int argc, char *argv[], uint32_t *batch_size, uint32_t *epoch, uint32_t *log_loss,
                uint32_t *dataset_size, char **layer_str, char **activation_str, char **weights_filename,
                char **input_filename, char **target_filename) {
    int opt;
    while ((opt = getopt(argc, argv, "b:e:l:a:w:i:t:s:v:h")) != -1) {
        switch (opt) {
            case 'b':
                *batch_size = atoi(optarg);
                break;
            case 'e':
                *epoch = atoi(optarg);
                *log_loss = *epoch / 5;
                break;
            case 's':
                *dataset_size = atoi(optarg);
                break;
            case 'l':
                *layer_str = optarg;
                break;
            case 'a':
                *activation_str = optarg;
                break;
            case 'w':
                *weights_filename = optarg;
                break;
            case 'i':
                *input_filename = optarg;
                break;
            case 't':
                *target_filename = optarg;
                break;
            case 'v':
                CURRENT_LOG_LEVEL = atoi(optarg);
                break;
            case 'h': default:
                print_usage(argv[0]);
                return false;
        }
    }

    if (!(*layer_str) || !(*activation_str) || !(*input_filename) || !(*target_filename) || !(*dataset_size)) {
        LOG_ERROR("Missing required arguments.");
        print_usage(argv[0]);
        return false;
    }

    return true;
}

bool runTraining(uint32_t *LAYER_COMPOSITION, AIFES_E_activations *ACTIVATION_FUNCTION, uint32_t TOT_LAYER,
                 uint32_t DATASET_SIZE, uint32_t BATCH_SIZE, uint32_t EPOCH, uint32_t LOG_LOSS, float *weights,
                 float *input, float *target, float *output, char *weights_filename) {
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

    uint16_t input_shape[] = {DATASET_SIZE, LAYER_COMPOSITION[0]};
    uint16_t target_shape[] = {DATASET_SIZE, LAYER_COMPOSITION[TOT_LAYER - 1]};

    aitensor_t input_tensor = AITENSOR_2D_F32(input_shape, input);
    aitensor_t target_tensor = AITENSOR_2D_F32(target_shape, target);
    aitensor_t output_tensor = AITENSOR_2D_F32(target_shape, output);

    int8_t error = AIFES_E_training_fnn_f32(&input_tensor, &target_tensor, &NN_STRUCTURE, &TRAIN_STRUCTURE,
                                            &INIT_WEIGHTS, &output_tensor);

    switch (error) {
        case 0: LOG_INFO("Training OK\n");
            break;
        case -1: LOG_ERROR("ERROR! Tensor dtype\n");
            break;
        case -2: LOG_ERROR("ERROR! Tensor shape: Data Number\n");
            break;
        case -3: LOG_ERROR("ERROR! Input tensor shape does not correspond to ANN inputs\n");
            break;
        case -4: LOG_ERROR("ERROR! Output tensor shape does not correspond to ANN outputs\n");
            break;
        case -5: LOG_ERROR("ERROR! Use the crossentropy as loss for softmax\n");
            break;
        case -6: LOG_ERROR("ERROR! learn_rate or sgd_momentum negative\n");
            break;
        case -7: LOG_ERROR("ERROR! Init uniform weights min - max wrongn");
            break;
        case -8: LOG_ERROR("ERROR! batch_size: min = 1 / max = Number of training data\n");
            break;
        case -9: LOG_ERROR("ERROR! Unknown activation function\n");
            break;
        case -10: LOG_ERROR("ERROR! Unknown loss function\n");
            break;
        case -11: LOG_ERROR("ERROR! Unknown init weights method\n");
            break;
        case -12: LOG_ERROR("ERROR! Unknown optimizer\n");
            break;
        case -13: LOG_ERROR("ERROR! Not enough memory\n");
            break;
        default: LOG_ERROR("Unknown error\n");
    }

    if (!error) {
        save_float_array(weights_filename, weights, AIFES_E_flat_weights_number_fnn_f32(LAYER_COMPOSITION, TOT_LAYER));
    }
    return error == 0 ? true : false;
}

int main(int argc, char *argv[]) {
    srand(time(NULL));

    uint32_t BATCH_SIZE = 0;
    uint32_t EPOCH = 0;
    uint32_t LOG_LOSS = 0;
    uint32_t TOT_LAYER = 0;
    uint32_t DATASET_SIZE = 0;

    uint32_t *LAYER_COMPOSITION = NULL;
    AIFES_E_activations *ACTIVATION_FUNCTION = NULL;
    float *weights_data = NULL, *input_data = NULL, *target_data = NULL, *output_data = NULL;

    char *weights_filename = NULL, *input_filename = NULL, *target_filename = NULL;
    char *layer_string = NULL, *activation_string = NULL;

    if (!parse_args(argc, argv, &BATCH_SIZE, &EPOCH, &LOG_LOSS, &DATASET_SIZE, &layer_string, &activation_string,
                    &weights_filename, &input_filename, &target_filename)) {
        return EXIT_FAILURE;
    }

    if (!parse_layers(layer_string, &LAYER_COMPOSITION, &TOT_LAYER) ||
        !parse_activations(activation_string, &ACTIVATION_FUNCTION, TOT_LAYER) ||
        !INIT_AND_FILL_ARR(weights_filename, weights_data, float,
                           AIFES_E_flat_weights_number_fnn_f32(LAYER_COMPOSITION, TOT_LAYER), readCSV) ||
        !INIT_AND_FILL_ARR(input_filename, input_data, float, LAYER_COMPOSITION[0] * DATASET_SIZE, readCSV) ||
        !INIT_AND_FILL_ARR(target_filename, target_data, float, LAYER_COMPOSITION[TOT_LAYER - 1] * DATASET_SIZE,
                           readCSV) ||
        !INIT_ARR(output_data, float, LAYER_COMPOSITION[TOT_LAYER - 1] * DATASET_SIZE)
    ) {
        FREE_ALL_RESOURCES(LAYER_COMPOSITION, ACTIVATION_FUNCTION, weights_data, input_data, target_data, output_data);
        return EXIT_FAILURE;
    }


    LOG_INFO("Neural Network Configuration:");
    LOG_INFO("- Input Neurons: %d", LAYER_COMPOSITION[0]);
    LOG_INFO("- Output Neurons: %d", LAYER_COMPOSITION[TOT_LAYER - 1]);
    LOG_INFO("- Total Layers: %d", TOT_LAYER);
    LOG_INFO("- Batch Size: %d", BATCH_SIZE);
    LOG_INFO("- Epochs: %d", EPOCH);
    LOG_INFO("- Log every: %d epochs", LOG_LOSS);

    runTraining(LAYER_COMPOSITION, ACTIVATION_FUNCTION, TOT_LAYER, DATASET_SIZE, BATCH_SIZE, EPOCH, LOG_LOSS,
                weights_data,
                input_data, target_data, output_data, weights_filename);

    FREE_ALL_RESOURCES(LAYER_COMPOSITION, ACTIVATION_FUNCTION, weights_data, input_data, target_data, output_data);
    LOG_INFO("Resources freed. Exiting :)");
    return EXIT_SUCCESS;
}
