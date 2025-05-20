#ifndef AIDATASET_H
#define AIDATASET_H

#include "aiconfiguration.h"
#include <stdbool.h>

typedef struct aidataset {
    float *x_train;
    float *y_train;
    float *x_test;
    float *y_test;
} aidataset_t;

bool load_dataset(aiconfiguration_t ctx, aidataset_t *d);

#endif //AIDATASET_H
