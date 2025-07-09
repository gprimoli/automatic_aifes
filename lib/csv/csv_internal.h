#ifndef CSV_INTERNAL_H
#define CSV_INTERNAL_H

#include <stdbool.h>

#include "aifes.h"

static bool write_aitensor_to_csv(aitensor_t *l, FILE *f) {
    if (!l || !f) return false;

    int len = 0;
    switch (l->dim) {
        case 1: len = l->shape[0];
            break;
        case 2: len = l->shape[0] * l->shape[1];
            break;
        case 3: len = l->shape[0] * l->shape[1] * l->shape[2];
            break;
        case 4: default: len = l->shape[0] * l->shape[1] * l->shape[2] * l->shape[3];
    }

    return csv_write(l->data, len, f);
}

static bool read_aitensor_from_csv(aitensor_t *l, FILE *f) {
    if (!l || !f) return false;

    int len = 0;
    switch (l->dim) {
        case 1: len = l->shape[0];
            break;
        case 2: len = l->shape[0] * l->shape[1];
            break;
        case 3: len = l->shape[0] * l->shape[1] * l->shape[2];
            break;
        case 4: default: len = l->shape[0] * l->shape[1] * l->shape[2] * l->shape[3];
    }

    return csv_read(l->data, len, f);
}

#endif //CSV_INTERNAL_H
