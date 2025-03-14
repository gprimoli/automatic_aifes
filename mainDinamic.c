#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include "aifes.h"
#include "lib/GenericFuncs.h"

void printLoss(float loss);

int parse_layers(char *arg, uint32_t **LAYER_COMPOSITION, AIFES_E_activations **ACTIVATION_FUNCTION, int *INPUT,
                 int *OUTPUT, int *TOT_LAYER);

int parse_activation_func(char *arg, AIFES_E_activations *ACTIVATION_FUNCTION, int TOT_LAYER);


int main(int argc, char *argv[]) {
    int INPUT;
    int OUTPUT;

    int BATCH_SIZE;
    int EPOCH;
    int LOG_LOSS;

    uint32_t *LAYER_COMPOSITION;
    AIFES_E_activations *ACTIVATION_FUNCTION;
    int TOT_LAYER;


    srand(time(NULL));
    int opt;

    while ((opt = getopt(argc, argv, "b:e:l:a:x:")) != -1) {
        switch (opt) {
            case 'b':
                BATCH_SIZE = atoi(optarg);
                break;
            case 'e':
                EPOCH = atoi(optarg);
                break;
            case 'l':
                if (!parse_layers(optarg, &LAYER_COMPOSITION, &ACTIVATION_FUNCTION, &INPUT, &OUTPUT, &TOT_LAYER)) {
                    fprintf(stderr, "Error on parse_layers function\n");
                    return EXIT_FAILURE;
                }

                break;
            case 'a':
                if (!parse_activation_func(optarg, ACTIVATION_FUNCTION, TOT_LAYER)) {
                    free(LAYER_COMPOSITION);
                    free(ACTIVATION_FUNCTION);
                }
                break;
            case 'x':
                LOG_LOSS = atoi(optarg);
                break;
            default:
                fprintf(
                    stderr,
                    "Usage: %s -b <batch_size> -a <activation_function_composition> -e <epoch> -l <layer_composition>\n",
                    argv[0]);
                return 1;
        }
    }


    printf("Configurazione della Rete Neurale:\n");
    printf("- Input Neurons: %d\n", INPUT);
    printf("- Output Neurons: %d\n", OUTPUT);
    printf("- Total Layers: %d\n", TOT_LAYER);
    printf("- Batch Size: %d\n", BATCH_SIZE);
    printf("- Epochs: %d\n", EPOCH);
    printf("- Log Loss every %d epochs\n", LOG_LOSS);
    printf("\n");

    uint32_t weights_number = AIFES_E_flat_weights_number_fnn_f32(LAYER_COMPOSITION, TOT_LAYER);
    float *weights = malloc(weights_number * sizeof(double));
    if (weights == NULL) {
        free(LAYER_COMPOSITION);
        free(ACTIVATION_FUNCTION);
        printf("Memory allocation failed for weights\n");
        return EXIT_FAILURE;
    }


    /* ---- Wrapper ---- */
    AIFES_E_model_parameter_fnn_f32 nnStructure;
    nnStructure.layer_count = TOT_LAYER;
    nnStructure.fnn_structure = LAYER_COMPOSITION;
    nnStructure.fnn_activations = ACTIVATION_FUNCTION;
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
    trainStructure.early_stopping_target_loss = 0.004f;
    /* ---- Wrapper ---- */


    float *input_data = malloc(BATCH_SIZE * INPUT * sizeof(float));
    if (input_data == NULL) {
        free(LAYER_COMPOSITION);
        free(ACTIVATION_FUNCTION);
        free(weights);
        printf("Memory allocation failed for input_data\n");
        return EXIT_FAILURE;
    }
    uint16_t input_shape[] = {BATCH_SIZE, INPUT};
    aitensor_t input_tensor = AITENSOR_2D_F32(input_shape, input_data);


    float *target_data = malloc(BATCH_SIZE * OUTPUT * sizeof(float));
    if (target_data == NULL) {
        free(input_data);
        free(LAYER_COMPOSITION);
        free(ACTIVATION_FUNCTION);
        free(weights);
        printf("Memory allocation failed for target_data\n");
        return EXIT_FAILURE;
    }
    uint16_t target_shape[] = {BATCH_SIZE, OUTPUT};
    aitensor_t target_tensor = AITENSOR_2D_F32(target_shape, target_data);


    float *output_data = malloc(BATCH_SIZE * OUTPUT * sizeof(float));
    if (output_data == NULL) {
        free(target_data);
        free(input_data);
        free(LAYER_COMPOSITION);
        free(ACTIVATION_FUNCTION);
        free(weights);
        printf("Memory allocation failed for output_data\n");
        return EXIT_FAILURE;
    }
    uint16_t output_shape[] = {BATCH_SIZE, OUTPUT};
    aitensor_t output_tensor = AITENSOR_2D_F32(output_shape, output_data);

    int8_t error = AIFES_E_training_fnn_f32(&input_tensor, &target_tensor, &nnStructure, &trainStructure, &INIT_WEIGHTS,
                                            &output_tensor);
    printf("%d ", error);

    free(input_data);
    free(output_data);
    free(target_data);
    free(LAYER_COMPOSITION);
    free(ACTIVATION_FUNCTION);
    free(weights);

    return EXIT_SUCCESS;
}


// #define MAX_LINE_LENGTH 256
//
// void read_csv(const char *filename);
// void read_csv(const char *filename) {
//     FILE *file = fopen(filename, "r");
//     if (file != NULL) {
//         char line[MAX_LINE_LENGTH];
//         int row = 0;
//
//         while (fgets(line, sizeof(line), file)) {
//             row++;
//             char *token;
//             int column = 0;
//
//             token = strtok(line, ",");
//             while (token != NULL) {
//                 column++;
//                 printf("Riga %d, Colonna %d: %s\n", row, column, token);
//                 token = strtok(NULL, ",");
//             }
//         }
//         fclose(file);
//     } else {
//         exit(EXIT_FAILURE);
//     }
// }


void printLoss(float loss) {
    printf("loss: %f\n", loss);
}

int parse_layers(char *arg, uint32_t **LAYER_COMPOSITION, AIFES_E_activations **ACTIVATION_FUNCTION, int *INPUT,
                 int *OUTPUT, int *TOT_LAYER) {
    *TOT_LAYER = 0;
    char *backup = strdup(arg);
    if (backup == NULL) {
        printf("Memory allocation failed for backup\n");
        return 0;
    }


    char *token = strtok(backup, ","); // modifies the original string by replacing delimiters with \0 !!!
    while (token != NULL) {
        (*TOT_LAYER)++;
        token = strtok(NULL, ",");
    }

    *LAYER_COMPOSITION = (uint32_t *) malloc((*TOT_LAYER) * sizeof(uint32_t));
    if (*LAYER_COMPOSITION == NULL) {
        free(backup);
        printf("Memory allocation failed for LAYER_COMPOSITION\n");
        return 0;
    }

    *ACTIVATION_FUNCTION = (AIFES_E_activations *) malloc((*TOT_LAYER - 1) * sizeof(AIFES_E_activations));
    if (*ACTIVATION_FUNCTION == NULL) {
        free(backup);
        free(*LAYER_COMPOSITION);
        printf("Memory allocation failed for ACTIVATION_FUNCTION\n");
        return 0;
    }

    strcpy(backup, arg);
    token = strtok(backup, ",");
    for (int i = 0; i < *TOT_LAYER; i++) {
        if (i == 0) {
            *INPUT = atoi(token);
        } else if (i == *TOT_LAYER - 1) {
            *OUTPUT = atoi(token);
        }

        (*LAYER_COMPOSITION)[i] = atoi(token);
        token = strtok(NULL, ",");
    }

    free(backup);
    return 1;
}

int parse_activation_func(char *arg, AIFES_E_activations *ACTIVATION_FUNCTION, int TOT_LAYER) {
    if (ACTIVATION_FUNCTION == NULL)
        return 0;

    char *token = strtok(arg, ","); // modifies the original string by replacing delimiters with \0 !!!
    for (int i = 0; i < TOT_LAYER - 1; i++) {
        if (token == NULL)
            return 0;
        if (strcmp(token, "relu") == 0) {
            ACTIVATION_FUNCTION[i] = AIfES_E_relu;
        } else if (strcmp(token, "sigmoid") == 0) {
            ACTIVATION_FUNCTION[i] = AIfES_E_sigmoid;
        } else if (strcmp(token, "softmax") == 0) {
            ACTIVATION_FUNCTION[i] = AIfES_E_softmax;
        } else if (strcmp(token, "leaky_relu") == 0) {
            ACTIVATION_FUNCTION[i] = AIfES_E_leaky_relu;
        } else if (strcmp(token, "elu") == 0) {
            ACTIVATION_FUNCTION[i] = AIfES_E_elu;
        } else if (strcmp(token, "tanh") == 0) {
            ACTIVATION_FUNCTION[i] = AIfES_E_tanh;
        } else if (strcmp(token, "softsign") == 0) {
            ACTIVATION_FUNCTION[i] = AIfES_E_softsign;
        } else if (strcmp(token, "linear") == 0) {
            ACTIVATION_FUNCTION[i] = AIfES_E_linear;
        }
        token = strtok(NULL, ",");
    }
    return 1;
}
