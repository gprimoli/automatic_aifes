#include <time.h>
#include <unistd.h>
#include <stdbool.h>
#include <string.h>

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
        "  %s -l 784,64,10 -a relu,softmax -e 100 -b 64 -i input.csv -t target.csv -s 200 -w weights.bin",
        prog_name, prog_name
    );
}

void printLoss(float loss) {
    LOG_INFO("Loss: %f", loss);
}

bool parse_layers(char *arg, uint32_t **LAYER_COMPOSITION, uint32_t *TOT_LAYER) {
    if (!arg || !LAYER_COMPOSITION || !TOT_LAYER) return false;

    *TOT_LAYER = 0;
    char *backup = strdup(arg);
    if (backup == NULL) {
        LOG_ERROR("Memory allocation failed for backup");
        return false;
    }

    for (char *token = strtok(backup, ","); token != NULL; token = strtok(NULL, ",")) {
        (*TOT_LAYER)++;
    }
    FREE_ALL_RESOURCES(backup);

    *LAYER_COMPOSITION = (uint32_t *) malloc((*TOT_LAYER) * sizeof(uint32_t));
    if (*LAYER_COMPOSITION == NULL) {
        LOG_ERROR("Memory allocation failed for LAYER_COMPOSITION");
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
        LOG_ERROR("Memory allocation failed for ACTIVATION_FUNCTION");
        return false;
    }

    char *token = strtok(arg, ","); // modifies the original string by replacing delimiters with \0 !!!
    for (int i = 0; i < TOT_LAYER - 1; i++) {
        if (token == NULL) {
            LOG_ERROR("Not enough activation functions provided.");
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
            LOG_ERROR("Unknown activation function: '%s'", token);
            FREE_ALL_RESOURCES(ACTIVATION_FUNCTION);
            return false;
        }
        token = strtok(NULL, ",");
    }
    return true;
}

bool parse_args(int argc, char *argv[], uint32_t *batch_size, uint32_t *epoch, uint32_t *log_loss,
                uint32_t *dataset_training, uint32_t *dataset_testing, char **layer_str, char **activation_str,
                char **weights_filename,
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
                uint32_t dataset_size = atoi(optarg);
                *dataset_training = (uint32_t) dataset_size * 0.8;
                *dataset_testing = (uint32_t) dataset_size - *dataset_training;
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

    if (!(*layer_str) || !(*activation_str) || !(*input_filename) || !(*target_filename) || !(*dataset_training) || !(*
            dataset_testing) || *batch_size > *dataset_training) {
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
        case 0: LOG_INFO("Training OK");
            break;
        case -1: LOG_ERROR("Tensor dtype");
            break;
        case -2: LOG_ERROR("Tensor shape: Data Number");
            break;
        case -3: LOG_ERROR("Input tensor shape does not correspond to ANN inputs");
            break;
        case -4: LOG_ERROR("Output tensor shape does not correspond to ANN outputs");
            break;
        case -5: LOG_ERROR("Use the crossentropy as loss for softmax");
            break;
        case -6: LOG_ERROR("learn_rate or sgd_momentum negative");
            break;
        case -7: LOG_ERROR("Init uniform weights min - max wrongn");
            break;
        case -8: LOG_ERROR("batch_size: min = 1 / max = Number of training data");
            break;
        case -9: LOG_ERROR("Unknown activation function");
            break;
        case -10: LOG_ERROR("Unknown loss function");
            break;
        case -11: LOG_ERROR("Unknown init weights method");
            break;
        case -12: LOG_ERROR("Unknown optimizer");
            break;
        case -13: LOG_ERROR("Not enough memory");
            break;
        default: LOG_ERROR("Unknown error");
    }

    if (!error) {
        save_float_array(weights_filename, weights, AIFES_E_flat_weights_number_fnn_f32(LAYER_COMPOSITION, TOT_LAYER));
    }
    return error == 0 ? true : false;
}

bool runInference(uint32_t *LAYER_COMPOSITION, AIFES_E_activations *ACTIVATION_FUNCTION, uint32_t TOT_LAYER,
                  uint32_t DATASET_SIZE, float *weights, float *input, float *target, float *output) {
    AIFES_E_model_parameter_fnn_f32 NN_STRUCTURE = {
        .layer_count = TOT_LAYER,
        .fnn_structure = LAYER_COMPOSITION,
        .fnn_activations = ACTIVATION_FUNCTION,
        .flat_weights = weights,
    };

    uint16_t input_shape[] = {DATASET_SIZE, LAYER_COMPOSITION[0]};
    uint16_t output_shape[] = {DATASET_SIZE, LAYER_COMPOSITION[TOT_LAYER - 1]};

    aitensor_t input_tensor = AITENSOR_2D_F32(input_shape, input);
    aitensor_t output_tensor = AITENSOR_2D_F32(output_shape, output);

    int8_t error = AIFES_E_inference_fnn_f32(&input_tensor, &NN_STRUCTURE, &output_tensor);

    switch (error) {
        case 0:
            //TODO: do it better
            LOG_INFO("Inference OK");

            if (LAYER_COMPOSITION[TOT_LAYER - 1] == 1) {
                uint32_t TP = 0, FP = 0, FN = 0, TN = 0;
                for (int i = 0; i < DATASET_SIZE; ++i) {
                    const int pred = output[i] >= 0.5 ? 1 : 0;
                    if (pred == 1 && target[i] == 1) TP++;
                    else if (pred == 1 && target[i] == 0) FP++;
                    else if (pred == 0 && target[i] == 1) FN++;
                    else if (pred == 0 && target[i] == 0) TN++;
                }

                const float precision = TP + FP == 0 ? 0 : (float)TP / (float) (TP + FP);
                const float recall    = TP + FN == 0 ? 0 : (float)TP / (float) (TP + FN);
                const float f1  = precision + recall == 0 ? 0 : 2 * (precision * recall) / (precision + recall);
                LOG_INFO("Accuracy: %f", ((float)(TP + TN) / (float) DATASET_SIZE) * 100);
                LOG_INFO("Precision: %f  Recall: %f  F1 Score: %f", precision, recall, f1);
            }else {
                const uint32_t num_classes = LAYER_COMPOSITION[TOT_LAYER - 1];
                uint32_t *TP = calloc(num_classes, sizeof(uint32_t));
                uint32_t *FP = calloc(num_classes, sizeof(uint32_t));
                uint32_t *FN = calloc(num_classes, sizeof(uint32_t));

                if (TP != NULL && FP != NULL && FN != NULL) {
                    uint32_t correct = 0;
                    for (int i = 0; i < DATASET_SIZE; i++) {
                        const int pred = argmax(&output[i * num_classes], num_classes);
                        const int true_label = (int) target[i];

                        if (pred == true_label) {
                            correct++;
                            TP[pred]++;
                        } else {
                            FP[pred]++;
                            FN[true_label]++;
                        }
                    }

                    LOG_INFO("Accuracy: %f", ((float) correct / (float) DATASET_SIZE) * 100);

                    for (int c = 0; c < num_classes; c++) {
                        const uint32_t tp = TP[c];
                        const uint32_t fp = FP[c];
                        const uint32_t fn = FN[c];
                        const float precision = tp + fp == 0 ? 0 : (float) tp / (float) (tp + fp);
                        const float recall = tp + fn == 0 ? 0 : (float) tp / (float) (tp + fn);
                        const float f1 = precision + recall == 0 ? 0 : 2 * (precision * recall) / (precision + recall);

                        LOG_INFO("Class %d - Precision: %f  Recall: %f  F1 Score: %f", c, precision, recall, f1);
                    }
                }
                FREE_ALL_RESOURCES(TP, FP, FN);
            }
            break;
        case -1:
            LOG_ERROR("Tensor dtype");
            break;
        case -2:
            LOG_ERROR("Tensor shape: Data Number");
            break;
        case -3:
            LOG_ERROR("Input tensor shape does not correspond to ANN inputs");
            break;
        case -4:
            LOG_ERROR("Output tensor shape does not correspond to ANN outputs");
            break;
        case -5:
            LOG_ERROR("Unknown activation function");
            break;
        case -6:
            LOG_ERROR("Not enough memory");
            break;
        default:
            LOG_ERROR("Unknown error");
    }

    return error == 0 ? true : false;
}

int main(int argc, char *argv[]) {
    srand(time(NULL));

    uint32_t BATCH_SIZE = 0;
    uint32_t EPOCH = 0;
    uint32_t LOG_LOSS = 0;
    uint32_t TOT_LAYER = 0;
    uint32_t DATASET_TRAINING = 0;
    uint32_t DATASET_TESTING = 0;
    uint32_t DATASET_SIZE = 0;

    uint32_t *LAYER_COMPOSITION = NULL;
    AIFES_E_activations *ACTIVATION_FUNCTION = NULL;
    float *weights_data = NULL;
    float *input_data = NULL, *target_data = NULL, *output_data = NULL;
    float *test_input = NULL, *test_target = NULL, *test_output = NULL;

    char *weights_filename = NULL, *input_filename = NULL, *target_filename = NULL;
    char *layer_string = NULL, *activation_string = NULL;

    if (!parse_args(argc, argv, &BATCH_SIZE, &EPOCH, &LOG_LOSS, &DATASET_TRAINING, &DATASET_TESTING, &layer_string,
                    &activation_string, &weights_filename, &input_filename, &target_filename)) {
        return EXIT_FAILURE;
    }


    DATASET_SIZE = DATASET_TRAINING + DATASET_TESTING;

    if (!parse_layers(layer_string, &LAYER_COMPOSITION, &TOT_LAYER) ||
        !parse_activations(activation_string, &ACTIVATION_FUNCTION, TOT_LAYER) ||
        !INIT_AND_FILL_ARR(weights_filename, weights_data, float,
                           AIFES_E_flat_weights_number_fnn_f32(LAYER_COMPOSITION, TOT_LAYER), readCSV) ||
        !INIT_AND_FILL_ARR(input_filename, input_data, float, LAYER_COMPOSITION[0] * DATASET_SIZE,
                           readCSV) ||
        !INIT_AND_FILL_ARR(target_filename, target_data, float, LAYER_COMPOSITION[TOT_LAYER - 1] * DATASET_SIZE,
                           readCSV) ||
        !INIT_ARR(output_data, float, LAYER_COMPOSITION[TOT_LAYER - 1] * DATASET_SIZE)
    ) {
        FREE_ALL_RESOURCES(LAYER_COMPOSITION, ACTIVATION_FUNCTION, weights_data, input_data, target_data, output_data);
        return EXIT_FAILURE;
    }

    test_input = input_data + DATASET_TRAINING;
    test_target = target_data + DATASET_TRAINING;
    test_output = output_data + DATASET_TRAINING;

    LOG_INFO("Neural Network Configuration:");
    LOG_INFO("- Input Neurons: %d", LAYER_COMPOSITION[0]);
    LOG_INFO("- Output Neurons: %d", LAYER_COMPOSITION[TOT_LAYER - 1]);
    LOG_INFO("- Total Layers: %d", TOT_LAYER);
    LOG_INFO("- Batch Size: %d", BATCH_SIZE);
    LOG_INFO("- Epochs: %d", EPOCH);
    LOG_INFO("- Log every: %d epochs", LOG_LOSS);

    runTraining(LAYER_COMPOSITION, ACTIVATION_FUNCTION, TOT_LAYER, DATASET_TRAINING, BATCH_SIZE, EPOCH, LOG_LOSS,
                weights_data, input_data, target_data, output_data, weights_filename);

    runInference(LAYER_COMPOSITION, ACTIVATION_FUNCTION, TOT_LAYER, DATASET_TESTING, weights_data, test_input,
                 test_target, test_output);

    FREE_ALL_RESOURCES(LAYER_COMPOSITION, ACTIVATION_FUNCTION, weights_data, input_data, target_data, output_data);
    LOG_INFO("Resources freed. Exiting :)");
    return EXIT_SUCCESS;
}
