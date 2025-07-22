#ifndef CSV_INTERNAL_H
#define CSV_INTERNAL_H

#include <stdbool.h>

#include "aifes.h"
#include "aiconfiguration.h"

static bool write_aitensor_to_csv(aitensor_t *l, FILE *f, Quantization qType) {
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

    float *data = (float *) l->data;

    switch (qType) {
        case Q31: {
            aimath_q31_params_t *qp = (aimath_q31_params_t *) l->tensor_params;
            if (qp == NULL) return false;
            fprintf(f, "%hu, %d\n", qp->shift, qp->zero_point);
            break;
        }
        case Q7: {
            aimath_q7_params_t *qp = (aimath_q7_params_t *) l->tensor_params;
            if (qp == NULL) return false;
            fprintf(f, "%hu, %d\n", qp->shift, qp->zero_point);
            break;
        }
        default: break;
    }

    return csv_write(data, len, f);
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
