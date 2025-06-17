#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "aidataset.h"

#include "Log.h"
#include "MemManager.h"
#include "aiconfiguration.h"



#ifdef _WIN32
#define DIR_SEPARATOR '\\'
#else
  #define DIR_SEPARATOR '/'
#endif

bool readCSV(char *filepath, float **arr, size_t size) {
    FILE *f = fopen(filepath, "r");

    if (f == NULL) {
        LOG_ERROR("Failed to open file %s", filepath);
        fclose(f);
        SAFE_EXIT_FAILURE;
    }

    *arr = mem_calloc(size, sizeof(float));
    char line[15360] = {0};

    int i = 0;
    while (fgets(line, sizeof(line), f) != NULL) {
        char *col = strtok(line, ",");
        while (col) {
            float val = strtof(col, NULL);
            (*arr)[i++] = val;
            col = strtok(NULL, ",");
        }
    }

    if (i != size) {
        LOG_ERROR("Dataset error %d/%zu!", i, size);
        return false;
    }

    fclose(f);
    return true;
}

bool load_dataset(aiconfiguration_t ctx, aidataset_t *d) {
    char filepath[255] = {0};
    /*------- TRAINING -------*/
    const size_t x_train_size = ctx.sample_train * ctx.input_shape[0] * ctx.input_shape[1] * ctx.input_shape[2];
    const size_t y_train_size = ctx.sample_train * ctx.layers[ctx.num_layer - 1].params[NEURONS][0];

    sprintf(filepath, "%s%c%s", ctx.basedir, DIR_SEPARATOR, "x_train.csv");
    if (!readCSV(filepath, &(d->x_train), x_train_size)) return false;

    sprintf(filepath, "%s%c%s", ctx.basedir, DIR_SEPARATOR, "y_train.csv");
    if (!readCSV(filepath, &(d->y_train), y_train_size)) return false;

    // /*------- TESTING -------*/
    const size_t x_test_size = ctx.sample_test * ctx.input_shape[0] * ctx.input_shape[1] * ctx.input_shape[2];
    const size_t y_test_size = ctx.sample_test * ctx.layers[ctx.num_layer - 1].params[NEURONS][0];

    sprintf(filepath, "%s%c%s", ctx.basedir, DIR_SEPARATOR, "x_test.csv");
    if (!readCSV(filepath, &(d->x_test), x_test_size)) return false;

    sprintf(filepath, "%s%c%s", ctx.basedir, DIR_SEPARATOR, "y_test.csv");
    if (!readCSV(filepath, &(d->y_test), y_test_size)) return false;

    return true;
}
