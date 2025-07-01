#ifndef CSV_INTERNAL_H
#define CSV_INTERNAL_H

#include <stdbool.h>

#include "aifes.h"

static bool write_aitensor_to_csv(aitensor_t *l, FILE *f) {
    if (!l || !f) return false;

    int len = 0;
    if (l->dim == 2) {
        len = l->shape[0] * l->shape[1];
    } else if (l->dim == 4) {
        len = l->shape[0] * l->shape[1] *
              l->shape[2] * l->shape[3];
    }

    return csv_write(l->data, len, f);
}

static bool read_aitensor_from_csv(aitensor_t *l, FILE *f) {
    if (!l || !f) return false;

    int len = 0;
    if (l->dim == 2) {
        len = l->shape[0] * l->shape[1];
    } else if (l->dim == 4) {
        len = l->shape[0] * l->shape[1] *
              l->shape[2] * l->shape[3];
    }

    return csv_read(l->data, len, f);
}

#endif //CSV_INTERNAL_H
